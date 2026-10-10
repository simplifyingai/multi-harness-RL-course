# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Imitation (SFT) on the toy: a student copies an answer-key teacher.

The teacher knows the right query and plays the agent: 120 demonstrations on
the easy training questions. The student, chapter 4's softmax policy with one
set of weights, minimises -log p(teacher's action) at every call the teacher
made, in batches of 8 for one epoch (15 steps, the shape of the project's
real SFT run), and never scores what the environment sends back. Then it is
tested on questions about cities no demonstration used, including medium and
hard ones, which need more looking than any demonstration did.

Experiments: ``--fixed-length`` (every demonstration is LOOK, QUERY_A, WRITE,
FINISH), ``--varied`` (the project's four paths; the default),
``--teacher-never-checks``, ``--lr-high``, and ``--then-rl`` (chapter 6's
GRPO on the hard questions, started from the imitator and from scratch).

Run it with ``uv run sft_toy.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import report
import rl
import sft
from environment import ACTIONS, BUDGET, State, tables_needed
from evaluate import LevelResult, held_out_test
from knobs import (
    BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_RL_STEPS,
    DEFAULT_SEED,
    DEMO_LEVEL,
    DEMONSTRATIONS,
    EPOCHS,
    HIGH_LR,
    IMITATOR,
    RL_LR,
    SCRATCH,
)
from policy import features, weight_rows
from questions import HARD, LEVELS, Question, make_questions
from rl import RlStep, train_grpo
from sft import Demonstration, StepLog, collect_demonstrations, demo_items, train_sft
from tasks import CHAPTER, course_home, short_path, work_root
from teacher import FIXED_LENGTH, NEVER_CHECKS, VARIED

PAUSE_S = 0.3  # after each demonstration printed in full
STEP_PAUSE_S = 0.1  # after each step's line
NO_PAUSE_S = 0.0
LOG_DECIMALS = 4
RUNS_DIR = "runs"
RUN_PREFIX = "run-"
DEMOS_FILE = "demos.jsonl"
METRICS_FILE = "metrics.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"


@dataclass(frozen=True)
class Settings:
    """The knobs of one run (a plain record, read from the flags).

    Attributes:
        teacher: ``"varied"``, ``"fixed-length"`` or ``"never-checks"``.
        seed: Seeds the order the teacher takes the questions in, the test's
            sampled picks and RL's question order and picks. (The batches are
            shuffled with the epoch as their seed, as the project's are.)
        lr: The SFT learning rate.
        then_rl: After SFT, run GRPO on the hard questions from scratch and
            from the imitator.
        rl_steps: How many GRPO steps each start gets.
        calls_as_share: Give the student chapters 4 to 7's "calls used" share
            instead of the call number.
        pause_s: Seconds to wait after a demonstration printed in full.
        step_pause_s: Seconds to wait after each step's line.
    """

    teacher: str = VARIED
    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    then_rl: bool = False
    rl_steps: int = DEFAULT_RL_STEPS
    calls_as_share: bool = False
    pause_s: float = NO_PAUSE_S
    step_pause_s: float = NO_PAUSE_S


@dataclass(frozen=True)
class RlResult:
    """What the RL stage ends with, for both starts.

    Attributes:
        questions: How many hard questions it trained on.
        scratch: Every step from scratch.
        imitator: Every step from the imitator.
        after_scratch: The held-out test after RL from scratch.
        after_imitator: The held-out test after RL from the imitator.
    """

    questions: int
    scratch: tuple[RlStep, ...]
    imitator: tuple[RlStep, ...]
    after_scratch: tuple[LevelResult, ...]
    after_imitator: tuple[LevelResult, ...]


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        settings: The knobs it ran with.
        demos: Every demonstration.
        rows: How many training rows the trainer kept.
        sft_steps: Every SFT step.
        weights: The imitator's weights after SFT.
        before: The held-out test of the untrained student (every weight 0).
        after: The held-out test after SFT.
        rl: The RL stage, with ``--then-rl``.
        logs: Each log's path and md5, in the order printed.
    """

    settings: Settings
    demos: tuple[Demonstration, ...]
    rows: int
    sft_steps: tuple[StepLog, ...]
    weights: np.ndarray
    before: tuple[LevelResult, ...]
    after: tuple[LevelResult, ...]
    rl: RlResult | None
    logs: tuple[tuple[Path, str], ...]


@contextmanager
def run_folder() -> Iterator[Path]:
    """Make the folder this run's rollout folders go in, and remove it after.

    Its name is random, so two runs at once never share a folder; the folders
    inside are numbered, so the printed output repeats exactly.

    Yields:
        The run's folder under ``$COURSE_HOME/08-sft-toy/work``.

    Raises:
        OSError: If the folder cannot be made.
    """
    root = work_root()
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=RUN_PREFIX, dir=root))
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


def rounded(values: Any) -> Any:
    """Round numbers for the logs, so the md5 does not hang on tiny digits.

    Args:
        values: A number, or nested lists or arrays of numbers.

    Returns:
        The same shape, as plain rounded floats.
    """
    if isinstance(values, np.ndarray | list | tuple):
        return [rounded(value) for value in values]
    return round(float(values), LOG_DECIMALS)


def demo_record(demo: Demonstration) -> dict[str, Any]:
    """Describe one demonstration for demos.jsonl, as the trainer sees it.

    Args:
        demo: The demonstration.

    Returns:
        A JSON-ready record: the question, the path, each piece with its loss
        mask, and the verdict.
    """
    return {
        "number": demo.number,
        "question": demo.question.task.question,
        "level": demo.question.level,
        "path": demo.path,
        "pieces": [[piece.kind, piece.text] for piece in demo.sequence.pieces],
        "loss_mask": list(demo.sequence.loss_mask),
        "correct": demo.correct,
        "reason": demo.reason,
    }


def sft_metrics(log: StepLog) -> dict[str, Any]:
    """Describe one SFT step for metrics.jsonl, with the project's field names.

    The project logs ``loss_per_token`` and ``tokens``; one call is the toy's
    token, so here they are ``loss_per_call`` and ``calls``.

    Args:
        log: The step's log.

    Returns:
        A JSON-ready record.
    """
    return {
        "stage": sft.STAGE,
        "epoch": log.epoch,
        "step": log.step,
        "loss_per_call": rounded(log.loss_per_call),
        "calls": log.calls,
        "rows": log.rows,
    }


def rl_metrics(step: RlStep) -> dict[str, Any]:
    """Describe one GRPO step for metrics.jsonl, with the project's field names.

    Args:
        step: The step.

    Returns:
        A JSON-ready record: everything plot.py needs to redraw the run.
    """
    rollouts = step.rollouts
    return {
        "stage": rl.STAGE,
        "start": step.start,
        "step": step.number,
        "accuracy": sum(r.verdict.correct for r in rollouts) / len(rollouts),
        "mean_reward": sum(r.reward for r in rollouts) / len(rollouts),
        "mean_calls": sum(len(r.calls) for r in rollouts) / len(rollouts),
        "groups_without_contrast": step.no_signal,
        "groups": len(step.groups),
        "group_rewards": [group.rewards for group in step.groups],
        "rollouts": len(rollouts),
        "correct": sum(r.verdict.correct for r in rollouts),
        "train_calls": step.calls,
    }


def rl_records(step: RlStep) -> list[dict[str, Any]]:
    """Describe one GRPO step's rollouts for rollouts.jsonl.

    Args:
        step: The step.

    Returns:
        One JSON-ready record per rollout.
    """
    return [
        {
            "stage": rl.STAGE,
            "start": step.start,
            "step": step.number,
            "group": place,
            "question": group.question.task.question,
            "calls": [[call.action, call.observation] for call in rollout.calls],
            "answer": rollout.verdict.submitted,
            "reward": rollout.reward,
            "advantage": None if group.advantages is None else rounded(advantage),
        }
        for place, group in enumerate(step.groups, start=1)
        for rollout, advantage in zip(
            group.rollouts, group.advantages or [0.0] * len(group.rollouts), strict=True
        )
    ]


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> str:
    """Write records as JSON lines and return the file's md5.

    Args:
        path: Where to write.
        entries: The records.

    Returns:
        The md5 of the text written.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    path.write_text(text, encoding="utf-8")
    # md5 here is a fingerprint for comparing runs, not a security measure.
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def run_name(settings: Settings) -> str:
    """Name a run's folder after the settings that decide its contents.

    Args:
        settings: The run's knobs.

    Returns:
        For example ``"varied-seed0-lr3"`` or ``"varied-seed0-lr3-then-rl100"``.
    """
    name = f"{settings.teacher}-seed{settings.seed}-lr{settings.lr:g}"
    if settings.calls_as_share:
        name += "-calls-as-share"
    if settings.then_rl:
        name += f"-then-rl{settings.rl_steps}"
    return name


def looks_needed(questions: Sequence[Question]) -> dict[str, int]:
    """Count the LOOKs each level's questions need before their query works.

    Args:
        questions: Questions covering every level.

    Returns:
        The LOOKs a question of each level needs (the same for every
        question of a level).
    """
    return {
        level: tables_needed(
            next(q for q in questions if q.level == level).task.gold_sql
        )
        for level in LEVELS
    }


def run(settings: Settings, printing: bool = True) -> RunResult:
    """Collect demonstrations, run SFT, test, optionally run RL, write the logs.

    Args:
        settings: The run's knobs.
        printing: Print the run (``seeds.py`` turns it off).

    Returns:
        Everything the run did, and where its logs went.

    Raises:
        OSError: If a folder, a file or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
        CourseError: If a rollout or demonstration stops without a verdict
            (never expected).
    """
    questions = make_questions()
    held_out = questions.held_out
    describe: Callable[[State], np.ndarray] = partial(
        features, calls_as_share=settings.calls_as_share
    )
    demo_questions = [q for q in questions.training if q.level == DEMO_LEVEL]
    if printing:
        report.print_header(
            settings,
            len(demo_questions),
            len(questions.test_cities),
            {level: sum(q.level == level for q in held_out) for level in LEVELS},
            looks_needed(held_out),
        )
    log: list[dict[str, Any]] = []
    zeros = np.zeros((weight_rows(settings.calls_as_share), len(ACTIONS)))
    rl = None
    with run_folder() as parent:
        demos = collect_demonstrations(
            demo_questions,
            settings.teacher,
            DEMONSTRATIONS,
            settings.seed,
            parent,
            describe,
        )
        items = demo_items(demos, BUDGET)
        if printing:
            report.print_demonstrations(demos)
            time.sleep(settings.pause_s)
            report.print_rows(demos, len(items), sum(i.n_calls for i in items))

        def show(step: StepLog) -> None:
            """Print one SFT step's line, then pause.

            Args:
                step: The step's log.
            """
            report.print_sft_step(step)
            time.sleep(settings.step_pause_s)

        weights, sft_steps = train_sft(
            items,
            EPOCHS,
            settings.lr,
            BATCH_SIZE,
            zeros,
            on_step=show if printing else None,
        )
        test = partial(
            held_out_test,
            held_out=held_out,
            parent=parent,
            seed=settings.seed,
            describe=describe,
            log=log,
        )
        before = test(zeros, weights_name="before sft")
        after = test(weights, weights_name="after sft")
        if settings.then_rl:
            hard = [q for q in questions.training if q.level == HARD]
            runs = {}
            for start, initial in ((SCRATCH, zeros), (IMITATOR, weights)):
                runs[start] = train_grpo(
                    initial,
                    hard,
                    start,
                    settings.rl_steps,
                    RL_LR,
                    settings.seed,
                    parent,
                    describe,
                )
            rl = RlResult(
                questions=len(hard),
                scratch=tuple(runs[SCRATCH][1]),
                imitator=tuple(runs[IMITATOR][1]),
                after_scratch=test(runs[SCRATCH][0], weights_name="rl from scratch"),
                after_imitator=test(runs[IMITATOR][0], weights_name="sft then rl"),
            )
    folder = course_home() / CHAPTER / RUNS_DIR / run_name(settings)
    metrics = [sft_metrics(step) for step in sft_steps]
    if rl is not None:
        for step in (*rl.scratch, *rl.imitator):
            metrics.append(rl_metrics(step))
            log.extend(rl_records(step))
    log.append({"weights after sft": rounded(weights)})
    files = (
        (folder / DEMOS_FILE, [demo_record(demo) for demo in demos]),
        (folder / METRICS_FILE, metrics),
        (folder / ROLLOUTS_FILE, log),
    )
    logs = tuple((path, write_jsonl(path, entries)) for path, entries in files)
    result = RunResult(
        settings=settings,
        demos=tuple(demos),
        rows=len(items),
        sft_steps=tuple(sft_steps),
        weights=weights,
        before=before,
        after=after,
        rl=rl,
        logs=logs,
    )
    if printing:
        report.print_run(result)
        report.print_logs(
            [(short_path(path), md5) for path, md5 in logs],
            short_path(folder / METRICS_FILE),
        )
    return result


def non_negative(text: str) -> float:
    """Read a flag that must be an ordinary number, 0 or more.

    Args:
        text: What was typed after the flag.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If it is not a number, is negative, or is
            not finite (``nan`` or ``inf``).
    """
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be 0 or more")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the flags.

    Args:
        argv: The flags (None reads the command line).

    Returns:
        The run's knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    teachers = parser.add_mutually_exclusive_group()
    teachers.add_argument(
        "--fixed-length",
        action="store_true",
        help="every demonstration is LOOK, QUERY_A, WRITE, FINISH",
    )
    teachers.add_argument(
        "--varied",
        action="store_true",
        help="the project's four paths: direct, peek, tables, verify (the default)",
    )
    teachers.add_argument(
        "--teacher-never-checks",
        action="store_true",
        help="the four paths, but verify never CHECKs answer.txt",
    )
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument(
        "--lr-high",
        action="store_true",
        help=f"SFT at lr {HIGH_LR:g}, ten times the default {DEFAULT_LR:g}",
    )
    rates.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="the SFT learning rate"
    )
    parser.add_argument(
        "--then-rl",
        action="store_true",
        help="then GRPO on the hard questions, from the imitator and from scratch",
    )
    parser.add_argument(
        "--rl-steps",
        type=int,
        default=DEFAULT_RL_STEPS,
        metavar="N",
        help="GRPO steps for each start",
    )
    parser.add_argument(
        "--calls-as-share",
        action="store_true",
        help="give the student chapters 4 to 7's calls-used share, not the call number",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.rl_steps < 1:
        parser.error("seed must be 0 or more, and rl-steps 1 or more")
    return args


def settings_from(args: argparse.Namespace) -> Settings:
    """Turn the flags into a run's settings.

    Args:
        args: The parsed flags.

    Returns:
        The settings.
    """
    teacher = VARIED
    if args.fixed_length:
        teacher = FIXED_LENGTH
    elif args.teacher_never_checks:
        teacher = NEVER_CHECKS
    return Settings(
        teacher=teacher,
        seed=args.seed,
        lr=HIGH_LR if args.lr_high else args.lr,
        then_rl=args.then_rl,
        rl_steps=args.rl_steps,
        calls_as_share=args.calls_as_share,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
        step_pause_s=NO_PAUSE_S if args.fast else STEP_PAUSE_S,
    )


def main() -> None:
    """Parse the flags and run."""
    run(settings_from(parse_args()))


if __name__ == "__main__":
    main()
