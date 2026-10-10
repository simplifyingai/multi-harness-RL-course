# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""GRPO on chapter 4's tiny agent: try each question 4 times, compare within the group.

Every step picks 2 training questions (easy, medium or hard) and runs 4
rollouts of each at temperature 1, each in a new folder. Each group's rewards
become advantages with the project's group_advantages(): (reward - mean) /
(spread + 0.0001), or None when all four tie, in which case the group teaches
nothing and is skipped. Then one gradient-descent step on chapter 3's
policy-gradient loss, with every call sharing its rollout's advantage.

Every step appends one line to metrics.jsonl, and plot.py redraws the curves
from that file alone. ``--group 1,0,1,1`` works out one group's advantages by
hand and stops.

Run it with ``uv run grpo_groups.py --trace`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import report
from environment import ACTIONS, Call, MiniAnalyst, NotGraded, State, fresh_folder
from grader import Grade
from grpo import (
    ALL_RIGHT,
    COUNTS,
    GROUP_SIZE,
    QUESTIONS_PER_STEP,
    RIGHT,
    ROLLOUTS,
    SIGNAL,
    Trained,
    group_advantages,
    group_outcome,
    step_update,
)
from policy import (
    FEATURE_NAMES,
    action_scores,
    features,
    gradient_step,
    greedy,
    log_prob,
    pick,
    softmax,
    weight_rows,
)
from questions import LEVELS, TEST, TRAIN, Question, make_questions
from tasks import CHAPTER, course_home, short_path, work_root

DEFAULT_SEED = 0
DEFAULT_LR = 3.0  # the loss is an average over calls, so steps are bigger
DEFAULT_STEPS = 200
STRETCHES = 4  # the table splits the run into quarters (report.py says so)
PAUSE_S = 0.4  # after each rollout printed in full
STEP_PAUSE_S = 0.05  # after each step's line
NO_PAUSE_S = 0.0
PICK_STREAM = 1  # the random stream for the agent's picks
LOG_DECIMALS = 4
STAGE = "grpo"
RUNS_DIR = "runs"
RUN_PREFIX = "run-"
SHARED_PREFIX = "shared-"
METRICS_FILE = "metrics.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"
ROLLOUT_FOLDER = "step-{:03d}-group-{}-rollout-{}"
TEST_FOLDER = "greedy-{}-{:03d}"


@dataclass(frozen=True)
class Settings:
    """The knobs of one run (a plain record, read from the flags).

    Attributes:
        seed: Seeds the question order and the random picks; the same seed
            gives the same run.
        lr: The learning rate: how big each gradient-descent step is.
        steps: How many steps to train for (one update each).
        shared: One set of weights for every level instead of one per level.
        trace: Print three groups in full: the first, the first with signal
            and the first that was all right.
        pause_s: Seconds to wait after each rollout printed in full.
        step_pause_s: Seconds to wait after each step's line.
    """

    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    steps: int = DEFAULT_STEPS
    shared: bool = False
    trace: bool = False
    pause_s: float = NO_PAUSE_S
    step_pause_s: float = NO_PAUSE_S


@dataclass(frozen=True)
class Rollout:
    """One attempt at one question, from its first call to its reward.

    Attributes:
        question: The question answered.
        folder: The name of the rollout's own folder.
        calls: Every call: the action and what came back.
        situations: The features the agent saw before each call.
        scores: The action scores at each call.
        choices: The action taken at each call (an index into the actions).
        chances: The probability the taken action had.
        log_probs: Its log-probability.
        stop: Why the rollout stopped: ``"FINISH"`` or ``"budget used"``.
        verdict: The grader's verdict on answer.txt.
        files: The files in the folder when the rollout stopped.
        states: The state before each call.
    """

    question: Question
    folder: str
    calls: tuple[Call, ...]
    situations: tuple[np.ndarray, ...]
    scores: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    chances: tuple[float, ...]
    log_probs: tuple[float, ...]
    stop: str
    verdict: Grade
    files: tuple[str, ...]
    states: tuple[State, ...]

    @property
    def reward(self) -> int:
        """The rollout's one reward: 1 for a correct answer.txt, else 0."""
        return self.verdict.reward


@dataclass(frozen=True)
class Group:
    """One question's group: its rollouts, their rewards and advantages.

    Attributes:
        step: The step the group belongs to, from 1.
        number: Its place in the step, from 1.
        question: The question all its rollouts answered.
        rollouts: The rollouts, in order.
        advantages: One advantage per rollout, or None when all rewards tied.
    """

    step: int
    number: int
    question: Question
    rollouts: tuple[Rollout, ...]
    advantages: tuple[float, ...] | None

    @property
    def rewards(self) -> list[int]:
        """The rollouts' rewards, in order."""
        return [rollout.reward for rollout in self.rollouts]

    @property
    def outcome(self) -> str:
        """``"all wrong"``, ``"all right"`` or ``"signal"``."""
        return group_outcome(self.rewards)


@dataclass(frozen=True)
class Step:
    """One step: its groups and the update they caused.

    Attributes:
        number: The step's number, from 1.
        groups: Its groups, in order.
        calls: How many calls the update learned from.
        loss: The step's loss, or None when no group had contrast.
    """

    number: int
    groups: tuple[Group, ...]
    calls: int
    loss: float | None

    @property
    def rollouts(self) -> list[Rollout]:
        """Every rollout of the step, group by group."""
        return [rollout for group in self.groups for rollout in group.rollouts]

    @property
    def no_signal(self) -> int:
        """How many of the step's groups had no contrast."""
        return sum(group.advantages is None for group in self.groups)


@dataclass(frozen=True)
class LevelScore:
    """How the greedy agent did on one level at the end.

    Attributes:
        level: ``"easy"``, ``"medium"`` or ``"hard"``.
        training: Correct answers on the training questions of that level.
        training_total: How many training questions it has.
        held_out: Correct answers on the held-out questions of that level.
        held_out_total: How many held-out questions it has.
        actions: The calls of the first greedy rollout of that level.
        untrained: Whether that level's weights are all still 0.
    """

    level: str
    training: int
    training_total: int
    held_out: int
    held_out_total: int
    actions: tuple[str, ...]
    untrained: bool


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        weights: The trained weights.
        steps: Every step, in order.
        table: Per level, per stretch of steps, how many groups were all
            wrong, all right or had signal, and how many rollouts were right.
        edges: Where the stretches start and end: steps ``edges[i] + 1`` to
            ``edges[i + 1]``.
        scores: The greedy test at the end, per level.
        metrics_path: Where metrics.jsonl was written.
        metrics_md5: Its md5.
        rollouts_path: Where rollouts.jsonl was written.
        rollouts_md5: Its md5.
    """

    weights: np.ndarray
    steps: tuple[Step, ...]
    table: dict[str, list[dict[str, int]]]
    edges: tuple[int, ...]
    scores: tuple[LevelScore, ...]
    metrics_path: Path
    metrics_md5: str
    rollouts_path: Path
    rollouts_md5: str


@contextmanager
def run_folder() -> Iterator[Path]:
    """Make the folder this run's rollout folders go in, and remove it after.

    Its name is random, so two runs at once never share a folder; the rollout
    folders inside are numbered, so the printed output repeats exactly.

    Yields:
        The run's folder under ``$COURSE_HOME/05-grpo-groups/work``.

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


def run_rollout(
    weights: np.ndarray,
    question: Question,
    parent: Path,
    name: str,
    choose: Callable[[np.ndarray], int],
    shared: bool,
) -> Rollout:
    """Run one rollout in its own new folder, call by call, until it stops.

    Args:
        weights: The policy's weights.
        question: The question to answer.
        parent: The run's folder; the rollout's folder is made inside it.
        name: The rollout folder's name.
        choose: Turns action scores into an action: a random pick while
            training, the top score in the greedy test.
        shared: One set of weights for every level.

    Returns:
        Everything that happened, with the grader's verdict.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If the rollout stopped without a verdict (never expected).
    """
    situations, scores, choices, states, chances, log_probs = [], [], [], [], [], []
    with fresh_folder(parent, name) as folder:
        env = MiniAnalyst(question, folder)
        while not env.done:
            states.append(replace(env.state))  # a copy, as it is before the call
            situation = features(env.state, shared)
            row = action_scores(weights, situation)
            choice = choose(row)
            env.step(ACTIONS[choice])
            situations.append(situation)
            scores.append(row)
            choices.append(choice)
            chances.append(float(softmax(row)[choice]))
            log_probs.append(log_prob(row, choice))
        files = tuple(sorted(path.name for path in folder.iterdir()))
    # A stopped rollout always has both; checking says so to a type checker too.
    if env.grade is None or env.state.stop is None:
        raise NotGraded(f"{name} stopped without a verdict")
    return Rollout(
        question=question,
        folder=name,
        calls=tuple(env.calls),
        situations=tuple(situations),
        scores=tuple(scores),
        choices=tuple(choices),
        chances=tuple(chances),
        log_probs=tuple(log_probs),
        stop=env.state.stop,
        verdict=env.grade,
        files=files,
        states=tuple(states),
    )


def rounded(values: Any) -> Any:
    """Round numbers for the logs, so the md5 does not hang on tiny digits.

    Args:
        values: A number, None, or nested lists or arrays of numbers.

    Returns:
        The same shape, as plain rounded floats (None stays None).
    """
    if values is None:
        return None
    if isinstance(values, np.ndarray | list | tuple):
        return [rounded(value) for value in values]
    return round(float(values), LOG_DECIMALS)


def rollout_records(group: Group) -> list[dict[str, Any]]:
    """Describe a group's rollouts for the rollout log, one record each.

    Args:
        group: The group.

    Returns:
        One JSON-ready record per rollout.
    """
    advantages = group.advantages or [None] * len(group.rollouts)
    return [
        {
            "step": group.step,
            "group": group.number,
            "level": group.question.level,
            "question": group.question.task.question,
            "calls": [[call.action, call.observation] for call in rollout.calls],
            "p": rounded(rollout.chances),
            "stop": rollout.stop,
            "answer": rollout.verdict.submitted,
            "reward": rollout.reward,
            "advantage": rounded(advantage),
        }
        for rollout, advantage in zip(group.rollouts, advantages, strict=True)
    ]


def step_metrics(step: Step) -> dict[str, Any]:
    """Describe one step for metrics.jsonl, with the project's field names.

    plot.py reads these fields by name, so renaming one breaks the plot.

    Args:
        step: The step.

    Returns:
        One JSON-ready record: everything plot.py needs to redraw the run.
    """
    rollouts = step.rollouts
    return {
        "stage": STAGE,
        "step": step.number,
        "accuracy": sum(r.verdict.correct for r in rollouts) / len(rollouts),
        "mean_reward": sum(r.reward for r in rollouts) / len(rollouts),
        "mean_calls": sum(len(r.calls) for r in rollouts) / len(rollouts),
        "groups_without_contrast": step.no_signal,
        "groups": len(step.groups),
        "group_levels": [group.question.level for group in step.groups],
        "group_rewards": [group.rewards for group in step.groups],
        "train_rollouts": sum(
            len(g.rollouts) for g in step.groups if g.advantages is not None
        ),
        "train_calls": step.calls,
        "loss_per_call": rounded(step.loss),
    }


class Tracer:
    """Decides which groups ``--trace`` prints in full, and prints them.

    It prints three: the very first group, the first group with signal in
    which exactly one rollout failed, and the first group whose rollouts were
    all right.
    """

    def __init__(self, pause_s: float) -> None:
        """Start with nothing printed yet.

        Args:
            pause_s: Seconds to wait after each rollout printed.
        """
        self.pause_s = pause_s
        self.first_done = False
        self.one_wrong_done = False
        self.all_right_done = False

    def see(self, group: Group) -> None:
        """Print a group in full if it is one of the three worth seeing.

        Args:
            group: A group that has just been scored.
        """
        show = not self.first_done
        self.first_done = True
        one_wrong = sum(group.rewards) == len(group.rollouts) - 1
        if group.outcome == SIGNAL and one_wrong and not self.one_wrong_done:
            self.one_wrong_done = True
            show = True
        if group.outcome == ALL_RIGHT and not self.all_right_done:
            self.all_right_done = True
            show = True
        if show:
            report.print_group(group, self.pause_s)


def run_group(
    weights: np.ndarray,
    question: Question,
    step: int,
    place: int,
    parent: Path,
    pick_rng: np.random.Generator,
    settings: Settings,
) -> Group:
    """Run one question's group: 4 rollouts, their rewards and advantages.

    Args:
        weights: The policy's weights, the same for all four rollouts.
        question: The question every rollout answers.
        step: The step's number.
        place: The group's place in the step (1 or 2).
        parent: The run's folder.
        pick_rng: The random generator for the agent's picks.
        settings: The run's knobs.

    Returns:
        The group, with None for advantages when its rewards tie.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    rollouts = tuple(
        run_rollout(
            weights,
            question,
            parent,
            ROLLOUT_FOLDER.format(step, place, index),
            lambda row: pick(row, pick_rng),
            settings.shared,
        )
        for index in range(1, GROUP_SIZE + 1)
    )
    advantages = group_advantages([r.reward for r in rollouts])
    return Group(
        step,
        place,
        question,
        rollouts,
        None if advantages is None else tuple(advantages),
    )


def trained_rollouts(groups: Sequence[Group]) -> list[Trained]:
    """Collect what the step learns from: every rollout of a group with contrast.

    Groups with no contrast are skipped. Every call of every other rollout
    shares its rollout's advantage.

    Args:
        groups: The step's groups.

    Returns:
        One item per rollout learned from.
    """
    return [
        Trained(r.situations, r.scores, r.choices, r.log_probs, advantage)
        for group in groups
        if group.advantages is not None
        for r, advantage in zip(group.rollouts, group.advantages, strict=True)
    ]


def train(
    settings: Settings,
    training: tuple[Question, ...],
    parent: Path,
    log: list[dict[str, Any]],
    printing: bool = True,
) -> tuple[np.ndarray, tuple[Step, ...]]:
    """Run the GRPO loop: 2 questions x 4 rollouts, advantages, one step, again.

    Args:
        settings: The run's knobs.
        training: The questions to train on.
        parent: The run's folder, where every rollout gets its own folder.
        log: A list every rollout is appended to.
        printing: Print one line per step (and, with ``trace``, three groups).

    Returns:
        The trained weights and every step.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    weights = np.zeros((weight_rows(settings.shared), len(ACTIONS)))
    # The project's step structure: the training questions in one shuffled
    # order, taken two at a time and cycled.
    order = list(training)
    random.Random(settings.seed).shuffle(order)
    pick_rng = np.random.default_rng([settings.seed, PICK_STREAM])
    tracer = Tracer(settings.pause_s) if printing and settings.trace else None
    steps: list[Step] = []
    cursor = 0
    for number in range(1, settings.steps + 1):
        groups: list[Group] = []
        for place in range(1, QUESTIONS_PER_STEP + 1):
            question = order[cursor % len(order)]
            cursor += 1
            # Both groups are sampled with the same weights: the update comes
            # only after every rollout of the step.
            group = run_group(
                weights, question, number, place, parent, pick_rng, settings
            )
            groups.append(group)
            log.extend(rollout_records(group))
            if tracer is not None:
                tracer.see(group)
        update = step_update(trained_rollouts(groups), weights.shape)
        weights = gradient_step(weights, update.gradient, settings.lr)
        step = Step(number, tuple(groups), update.calls, update.loss)
        steps.append(step)
        if printing:
            report.print_step(step, step_metrics(step))
            time.sleep(settings.step_pause_s)
    return weights, tuple(steps)


def stretch_edges(steps: int) -> tuple[int, ...]:
    """Split a run into (at most) four stretches of steps.

    Args:
        steps: How many steps the run has.

    Returns:
        The boundaries, e.g. ``(0, 50, 100, 150, 200)`` for 200 steps.
    """
    size = math.ceil(steps / STRETCHES)
    return (*range(0, steps, size), steps)


def signal_table(
    steps: Sequence[Step], edges: Sequence[int]
) -> dict[str, list[dict[str, int]]]:
    """Count, per level and stretch, the groups by outcome and the rollouts right.

    Args:
        steps: Every step of a run.
        edges: The stretch boundaries.

    Returns:
        For each level and stretch: how many groups were all wrong, all right
        or had signal, how many rollouts there were and how many were right.
    """
    table = {level: [dict.fromkeys(COUNTS, 0) for _ in edges[1:]] for level in LEVELS}
    for step in steps:
        stretch = next(i for i, end in enumerate(edges[1:]) if step.number <= end)
        for group in step.groups:
            counts = table[group.question.level][stretch]
            counts[group.outcome] += 1
            counts[ROLLOUTS] += len(group.rollouts)
            counts[RIGHT] += sum(r.verdict.correct for r in group.rollouts)
    return table


def level_weights(weights: np.ndarray, level: str, shared: bool) -> np.ndarray:
    """Return the weights one level uses.

    Args:
        weights: All the weights.
        level: ``"easy"``, ``"medium"`` or ``"hard"``.
        shared: Whether every level shares one set of weights.

    Returns:
        That level's rows (all of them when shared).
    """
    if shared:
        return weights
    rows = len(FEATURE_NAMES)
    start = LEVELS.index(level) * rows
    return weights[start : start + rows]


def greedy_test(
    weights: np.ndarray,
    training: tuple[Question, ...],
    held_out: tuple[Question, ...],
    parent: Path,
    log: list[dict[str, Any]],
    shared: bool,
) -> tuple[LevelScore, ...]:
    """Answer every question greedily, each in its own new folder.

    Args:
        weights: The trained weights.
        training: The training questions.
        held_out: The held-out questions.
        parent: The run's folder.
        log: The rollout log, which gets one record per question.
        shared: One set of weights for every level.

    Returns:
        One score per level.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    results: dict[str, list[tuple[Question, Rollout]]] = {}
    for split, questions in ((TRAIN, training), (TEST, held_out)):
        rollouts = []
        for number, question in enumerate(questions, start=1):
            name = TEST_FOLDER.format(split, number)
            rollout = run_rollout(weights, question, parent, name, greedy, shared)
            rollouts.append((question, rollout))
            log.append(
                {
                    "test": question.task.question,
                    "level": question.level,
                    "calls": [call.action for call in rollout.calls],
                    "answer": rollout.verdict.submitted,
                    "correct": rollout.verdict.correct,
                }
            )
        results[split] = rollouts
    scores = []
    for level in LEVELS:
        mine = {
            split: [r for q, r in pairs if q.level == level]
            for split, pairs in results.items()
        }
        scores.append(
            LevelScore(
                level=level,
                training=sum(r.verdict.correct for r in mine[TRAIN]),
                training_total=len(mine[TRAIN]),
                held_out=sum(r.verdict.correct for r in mine[TEST]),
                held_out_total=len(mine[TEST]),
                actions=tuple(call.action for call in mine[TRAIN][0].calls),
                untrained=not level_weights(weights, level, shared).any(),
            )
        )
    return tuple(scores)


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
        For example ``"seed0-lr3-steps200"``.
    """
    shared = SHARED_PREFIX if settings.shared else ""
    return f"{shared}seed{settings.seed}-lr{settings.lr:g}-steps{settings.steps}"


def run(settings: Settings, printing: bool = True) -> RunResult:
    """Train, print each step, tabulate the groups, test greedily, write the logs.

    Args:
        settings: The run's knobs.
        printing: Print the run (``seeds.py`` turns it off).

    Returns:
        The weights, every step, the table, the greedy test and the logs.

    Raises:
        OSError: If a folder, a file or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    questions = make_questions()
    if printing:
        report.print_header(settings, questions)
    log: list[dict[str, Any]] = []
    with run_folder() as parent:
        weights, steps = train(settings, questions.training, parent, log, printing)
        scores = greedy_test(
            weights,
            questions.training,
            questions.held_out,
            parent,
            log,
            settings.shared,
        )
    edges = stretch_edges(settings.steps)
    table = signal_table(steps, edges)
    log.append({"weights": rounded(weights)})
    folder = course_home() / CHAPTER / RUNS_DIR / run_name(settings)
    metrics_path = folder / METRICS_FILE
    metrics_md5 = write_jsonl(metrics_path, [step_metrics(step) for step in steps])
    rollouts_path = folder / ROLLOUTS_FILE
    rollouts_md5 = write_jsonl(rollouts_path, log)
    if printing:
        report.print_table(table, edges)
        report.print_critic_note()
        report.print_greedy(scores)
        report.print_logs(
            short_path(metrics_path),
            metrics_md5,
            short_path(rollouts_path),
            rollouts_md5,
        )
    return RunResult(
        weights,
        steps,
        table,
        edges,
        scores,
        metrics_path,
        metrics_md5,
        rollouts_path,
        rollouts_md5,
    )


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


def rewards_list(text: str) -> list[float]:
    """Read ``--group``: rewards separated by commas, e.g. ``1,0,1,1``.

    Args:
        text: What was typed after the flag.

    Returns:
        The rewards.

    Raises:
        argparse.ArgumentTypeError: If any of them is not an ordinary number.
    """
    rewards = []
    for part in text.split(","):
        try:
            value = float(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{part!r} is not a number") from error
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"{part!r} is not an ordinary number")
        rewards.append(value)
    return rewards


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The run's knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print the first group, the first with signal and the first all right",
    )
    parser.add_argument(
        "--group",
        type=rewards_list,
        default=None,
        metavar="R1,R2,...",
        help="work out one group's advantages by hand, then stop",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="one set of weights for every level instead of one per level",
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="steps (updates) to train for"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("steps must be 1 or more, and seed 0 or more")
    return args


def main() -> None:
    """Parse the flags, then work out one group or train."""
    args = parse_args()
    if args.group is not None:
        report.print_group_arithmetic(args.group, group_advantages(args.group))
        return
    settings = Settings(
        seed=args.seed,
        lr=args.lr,
        steps=args.steps,
        shared=args.shared,
        trace=args.trace,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
        step_pause_s=NO_PAUSE_S if args.fast else STEP_PAUSE_S,
    )
    run(settings)


if __name__ == "__main__":
    main()
