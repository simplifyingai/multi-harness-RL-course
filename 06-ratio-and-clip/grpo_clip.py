# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""GRPO with the importance ratio and the clip, on chapter 5's tiny agent.

Every call keeps the log-probability its action had when it was sampled (old).
The learner works it out again with the weights it has now (new). The ratio
new p / old p reweights each call, and the clip keeps that ratio within 0.8
to 1.2: loss = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), the project's
loss_fn line for line.

With one fresh batch and one update per step (the default) every ratio is
exactly 1 and the clip never acts. Three flags make the sampler and the
learner differ: ``--updates-per-batch 8`` learns from the same batch eight
times, ``--stale 2`` samples with weights two updates old, and
``--recompute-old`` is the bug that hides the difference. ``--no-clip`` takes
the clip away. ``--token 0.20,0.30`` works out one call by hand and stops.

Run it with ``uv run grpo_clip.py --stale 2`` (``--fast`` skips the pauses).
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
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import report
from environment import ACTIONS, Call, MiniAnalyst, NotGraded, fresh_folder
from grader import Grade
from grpo import (
    GROUP_SIZE,
    QUESTIONS_PER_STEP,
    Trained,
    call_logprobs,
    clip_active,
    group_advantages,
    group_outcome,
    outside_clip_range,
    step_update,
)
from policy import (
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
DEFAULT_LR = 3.0  # chapter 5's learning rate
DEFAULT_STEPS = 200
DEFAULT_UPDATES = 1  # one update per fresh batch, as the project does
DEFAULT_STALE = 0  # the sampler uses the learner's latest weights
STRETCHES = 4  # the tables split the run into quarters
PAUSE_S = 0.4  # after a call printed in full
STEP_PAUSE_S = 0.05  # after each step's line
NO_PAUSE_S = 0.0
PICK_STREAM = 1  # the random stream for the agent's picks (chapter 5's)
LOG_DECIMALS = 4
STAGE = "grpo"
RUNS_DIR = "runs"
RUN_PREFIX = "run-"
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
        steps: How many steps to train for (one batch each).
        updates: How many updates each batch is used for (1 by default; the
            chapter also tries 4 and 8).
        stale: How many updates behind the learner the sampler's weights are.
        recompute_old: The bug: score the "old" probabilities with the
            learner's weights instead of keeping the sampler's.
        clip: Whether the clip is on (``--no-clip`` turns it off).
        trace: Print the first call the clip switches off, in full.
        pause_s: Seconds to wait after a call printed in full.
        step_pause_s: Seconds to wait after each step's line.
    """

    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    steps: int = DEFAULT_STEPS
    updates: int = DEFAULT_UPDATES
    stale: int = DEFAULT_STALE
    recompute_old: bool = False
    clip: bool = True
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
        choices: The action taken at each call (an index into the actions).
        chances: The probability the taken action had when it was sampled.
        log_probs: Its log-probability: the "old" one, recorded at sampling.
        stop: Why the rollout stopped: ``"FINISH"`` or ``"budget used"``.
        verdict: The grader's verdict on answer.txt.
    """

    question: Question
    folder: str
    calls: tuple[Call, ...]
    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    chances: tuple[float, ...]
    log_probs: tuple[float, ...]
    stop: str
    verdict: Grade

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
class Use:
    """One use of a batch: one update, and every call's ratio in it.

    Attributes:
        ratios: Every call's ratio, as the loss saw it.
        clipped: For every call, whether the clip switched its push off.
        loss: The update's loss per call, or None with nothing to learn.
        recorded_ratios: With ``--recompute-old`` only: the same calls' ratios
            against the probabilities recorded at sampling, which the bug hides.
        recorded_clipped: With ``--recompute-old`` only: which of those the
            clip would have switched off.
    """

    ratios: np.ndarray
    clipped: np.ndarray
    loss: float | None
    recorded_ratios: np.ndarray | None = None
    recorded_clipped: np.ndarray | None = None


@dataclass(frozen=True)
class Step:
    """One step: its groups, how stale its sampler was, and every use of the batch.

    Attributes:
        number: The step's number, from 1.
        groups: Its groups, in order.
        calls: How many calls each update learned from.
        lag: How many updates behind the learner the sampler's weights were.
        uses: One record per update made with the batch.
    """

    number: int
    groups: tuple[Group, ...]
    calls: int
    lag: int
    uses: tuple[Use, ...]

    @property
    def rollouts(self) -> list[Rollout]:
        """Every rollout of the step, group by group."""
        return [rollout for group in self.groups for rollout in group.rollouts]

    @property
    def no_signal(self) -> int:
        """How many of the step's groups had no contrast."""
        return sum(group.advantages is None for group in self.groups)

    @property
    def ratios(self) -> np.ndarray:
        """Every ratio of every use, as the loss saw them."""
        return np.concatenate([use.ratios for use in self.uses])

    @property
    def clipped(self) -> np.ndarray:
        """Whether each of those calls was clipped."""
        return np.concatenate([use.clipped for use in self.uses])


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
    """

    level: str
    training: int
    training_total: int
    held_out: int
    held_out_total: int
    actions: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        settings: The knobs it ran with.
        weights: The trained weights.
        steps: Every step, in order.
        edges: Where the quarters start and end: steps ``edges[i] + 1`` to
            ``edges[i + 1]``.
        scores: The greedy test at the end, per level.
        metrics_path: Where metrics.jsonl was written.
        metrics_md5: Its md5.
        rollouts_path: Where rollouts.jsonl was written.
        rollouts_md5: Its md5.
    """

    settings: Settings
    weights: np.ndarray
    steps: tuple[Step, ...]
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
        The run's folder under ``$COURSE_HOME/06-ratio-and-clip/work``.

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
) -> Rollout:
    """Run one rollout in its own new folder, call by call, until it stops.

    Args:
        weights: The weights the sampler acts with.
        question: The question to answer.
        parent: The run's folder; the rollout's folder is made inside it.
        name: The rollout folder's name.
        choose: Turns action scores into an action: a random pick while
            training, the top score in the greedy test.

    Returns:
        Everything that happened, with each action's probability as it was
        sampled and the grader's verdict.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If the rollout stopped without a verdict (never expected).
    """
    situations, choices, chances, log_probs = [], [], [], []
    with fresh_folder(parent, name) as folder:
        env = MiniAnalyst(question, folder)
        while not env.done:
            situation = features(env.state)
            row = action_scores(weights, situation)
            choice = choose(row)
            env.step(ACTIONS[choice])
            situations.append(situation)
            choices.append(choice)
            # Recorded now, with the weights that chose: the "old" probability.
            chances.append(float(softmax(row)[choice]))
            log_probs.append(log_prob(row, choice))
    # A stopped rollout always has both; checking says so to a type checker too.
    if env.grade is None or env.state.stop is None:
        raise NotGraded(f"{name} stopped without a verdict")
    return Rollout(
        question=question,
        folder=name,
        calls=tuple(env.calls),
        situations=tuple(situations),
        choices=tuple(choices),
        chances=tuple(chances),
        log_probs=tuple(log_probs),
        stop=env.state.stop,
        verdict=env.grade,
    )


def run_group(
    weights: np.ndarray,
    question: Question,
    step: int,
    place: int,
    parent: Path,
    pick_rng: np.random.Generator,
) -> Group:
    """Run one question's group: GROUP_SIZE rollouts, their rewards and advantages.

    Args:
        weights: The sampler's weights, the same for every rollout.
        question: The question every rollout answers.
        step: The step's number.
        place: The group's place in the step, from 1.
        parent: The run's folder.
        pick_rng: The random generator for the agent's picks.

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
    """Collect what the step learns from, with the probabilities recorded at sampling.

    Groups with no contrast are skipped. Every call of every other rollout
    shares its rollout's advantage.

    Args:
        groups: The step's groups.

    Returns:
        One item per rollout learned from.
    """
    return [
        Trained(r.situations, r.choices, r.log_probs, advantage)
        for group in groups
        if group.advantages is not None
        for r, advantage in zip(group.rollouts, group.advantages, strict=True)
    ]


def recompute_old(trained: Sequence[Trained], weights: np.ndarray) -> list[Trained]:
    """The bug: replace each recorded "old" log p with one the learner works out.

    The learner scores the calls with its own weights when the batch arrives.
    If the sampler used other weights, the probabilities it really used are
    gone, and the first update's ratios all read exactly 1.

    Args:
        trained: The rollouts, with the probabilities recorded at sampling.
        weights: The learner's weights when the batch arrives.

    Returns:
        The same rollouts with recomputed "old" log-probabilities.
    """
    return [
        replace(item, old_logprobs=tuple(call_logprobs(weights, item)))
        for item in trained
    ]


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


def ratio_summary(ratios: np.ndarray, clipped: np.ndarray) -> dict[str, Any]:
    """Sum up some ratios for the logs.

    ``importance_ratio_mean`` is the project's field name. The counts let
    plot.py add steps together; the histogram uses report.py's seven bins.

    Args:
        ratios: The ratios.
        clipped: Whether each was clipped.

    Returns:
        How many ratios, how many fell outside 0.8 to 1.2 and how many were
        clipped, their histogram, and their mean, smallest and largest (None
        when there are no ratios).
    """
    present = bool(ratios.size)
    return {
        "ratio_calls": int(ratios.size),
        "ratios_outside": int(outside_clip_range(ratios).sum()),
        "clipped_calls": int(clipped.sum()),
        "ratio_histogram": report.histogram(ratios),
        "importance_ratio_mean": rounded(ratios.mean()) if present else None,
        "ratio_min": rounded(ratios.min()) if present else None,
        "ratio_max": rounded(ratios.max()) if present else None,
    }


def step_metrics(step: Step) -> dict[str, Any]:
    """Describe one step for metrics.jsonl, with the project's field names.

    plot.py reads these fields by name, so renaming one breaks the plot.

    Args:
        step: The step.

    Returns:
        One JSON-ready record: everything plot.py needs to redraw the run.
    """
    rollouts = step.rollouts
    uses = []
    for number, use in enumerate(step.uses, start=1):
        entry = {
            "use": number,
            "loss_per_call": rounded(use.loss),
            **ratio_summary(use.ratios, use.clipped),
        }
        if use.recorded_ratios is not None and use.recorded_clipped is not None:
            hidden = ratio_summary(use.recorded_ratios, use.recorded_clipped)
            entry["recorded"] = hidden
        uses.append(entry)
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
        "sampler_lag": step.lag,
        **ratio_summary(step.ratios, step.clipped),
        "uses": uses,
    }


def rollout_records(group: Group, lag: int) -> list[dict[str, Any]]:
    """Describe a group's rollouts for the rollout log, one record each.

    Args:
        group: The group.
        lag: How many updates behind the learner its sampler was.

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
            "sampler_lag": lag,
            "calls": [[call.action, call.observation] for call in rollout.calls],
            "old_p": rounded(rollout.chances),
            "stop": rollout.stop,
            "answer": rollout.verdict.submitted,
            "reward": rollout.reward,
            "advantage": rounded(advantage),
        }
        for rollout, advantage in zip(group.rollouts, advantages, strict=True)
    ]


class Tracer:
    """Prints, once, the first call the clip switches off, with its arithmetic."""

    def __init__(self, pause_s: float, clip: bool) -> None:
        """Start with nothing printed yet.

        Args:
            pause_s: Seconds to wait after the call is printed.
            clip: Whether the clip is on (with ``--no-clip`` nothing is clipped).
        """
        self.pause_s = pause_s
        self.clip = clip
        self.done = False

    def finish(self) -> None:
        """Say so when the run ended without the clip switching any call off."""
        if not self.done:
            report.print_nothing_clipped(self.clip)

    def see(
        self, step: int, use: int, trained: Sequence[Trained], weights: np.ndarray
    ) -> None:
        """Print the first clipped call of this update, if none was printed yet.

        Args:
            step: The step's number.
            use: Which use of the batch this update is, from 1.
            trained: The rollouts the update learns from.
            weights: The learner's weights before the update.
        """
        if self.done or not self.clip:
            return
        for item in trained:
            # The same ratios the update is about to see, worked out again
            # here so the call can be printed with its two probabilities.
            new = call_logprobs(weights, item)
            ratio = np.exp(new - np.array(item.old_logprobs))
            clipped = clip_active(ratio, item.advantage)
            if not clipped.any():
                continue
            index = int(np.argmax(clipped))  # the first clipped call
            self.done = True
            report.print_real_call(
                step,
                use,
                ACTIONS[item.choices[index]],
                math.exp(item.old_logprobs[index]),
                math.exp(new[index]),
                item.advantage,
            )
            time.sleep(self.pause_s)
            return


def train(
    settings: Settings,
    training: tuple[Question, ...],
    parent: Path,
    log: list[dict[str, Any]],
    printing: bool = True,
) -> tuple[np.ndarray, tuple[Step, ...]]:
    """Run GRPO with the ratio and the clip: sample, score, update, again.

    Args:
        settings: The run's knobs.
        training: The questions to train on.
        parent: The run's folder, where every rollout gets its own folder.
        log: A list every rollout is appended to.
        printing: Print one line per step (and, with ``trace``, one call).

    Returns:
        The trained weights and every step.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    weights = np.zeros((weight_rows(), len(ACTIONS)))
    # The weights after each of the last few updates, newest last. An
    # asynchronous trainer's sampler lags behind its learner, so --stale N
    # samples with the weights from N updates before the learner's.
    history = deque([weights], maxlen=settings.stale + 1)
    order = list(training)
    random.Random(settings.seed).shuffle(order)
    pick_rng = np.random.default_rng([settings.seed, PICK_STREAM])
    tracer = None
    if printing and settings.trace:
        tracer = Tracer(settings.pause_s, settings.clip)
    steps: list[Step] = []
    cursor = 0
    for number in range(1, settings.steps + 1):
        sampler = history[0]  # the oldest kept: N updates old once N are made
        lag = len(history) - 1
        groups: list[Group] = []
        for place in range(1, QUESTIONS_PER_STEP + 1):
            question = order[cursor % len(order)]
            cursor += 1
            group = run_group(sampler, question, number, place, parent, pick_rng)
            groups.append(group)
            log.extend(rollout_records(group, lag))
        recorded = trained_rollouts(groups)
        trained = recorded
        if settings.recompute_old:
            trained = recompute_old(recorded, weights)
        uses: list[Use] = []
        update = None
        for use in range(1, settings.updates + 1):
            if tracer is not None:
                tracer.see(number, use, trained, weights)
            update = step_update(trained, weights, settings.clip)
            record = Use(update.ratios, update.clipped, update.loss)
            if settings.recompute_old:
                # What the recorded probabilities would have shown (never used).
                shadow = step_update(recorded, weights, settings.clip)
                record = replace(
                    record,
                    recorded_ratios=shadow.ratios,
                    recorded_clipped=shadow.clipped,
                )
            uses.append(record)
            weights = gradient_step(weights, update.gradient, settings.lr)
            if update.calls:  # as in the project, learning nothing is no update
                history.append(weights)
        calls = 0 if update is None else update.calls
        step = Step(number, tuple(groups), calls, lag, tuple(uses))
        steps.append(step)
        if printing:
            report.print_step(step, step_metrics(step), settings.clip)
            time.sleep(settings.step_pause_s)
    if tracer is not None:
        tracer.finish()
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


def greedy_test(
    weights: np.ndarray,
    training: tuple[Question, ...],
    held_out: tuple[Question, ...],
    parent: Path,
    log: list[dict[str, Any]],
) -> tuple[LevelScore, ...]:
    """Answer every question greedily, each in its own new folder.

    Args:
        weights: The trained weights.
        training: The training questions.
        held_out: The held-out questions.
        parent: The run's folder.
        log: The rollout log, which gets one record per question.

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
            rollout = run_rollout(weights, question, parent, name, greedy)
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
        For example ``"seed0-lr3-steps200-updates8-stale0"``, with
        ``-recompute-old`` or ``-no-clip`` added when those are on.
    """
    name = (
        f"seed{settings.seed}-lr{settings.lr:g}-steps{settings.steps}"
        f"-updates{settings.updates}-stale{settings.stale}"
    )
    if settings.recompute_old:
        name += "-recompute-old"
    if not settings.clip:
        name += "-no-clip"
    return name


def run(settings: Settings, printing: bool = True) -> RunResult:
    """Train, print each step, sum up the ratios, test greedily, write the logs.

    Args:
        settings: The run's knobs.
        printing: Print the run (``seeds.py`` turns it off).

    Returns:
        The weights, every step, the greedy test and the logs.

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
            weights, questions.training, questions.held_out, parent, log
        )
    edges = stretch_edges(settings.steps)
    log.append({"weights": rounded(weights)})
    folder = course_home() / CHAPTER / RUNS_DIR / run_name(settings)
    metrics_path = folder / METRICS_FILE
    metrics_md5 = write_jsonl(metrics_path, [step_metrics(step) for step in steps])
    rollouts_path = folder / ROLLOUTS_FILE
    rollouts_md5 = write_jsonl(rollouts_path, log)
    result = RunResult(
        settings,
        weights,
        steps,
        edges,
        scores,
        metrics_path,
        metrics_md5,
        rollouts_path,
        rollouts_md5,
    )
    if printing:
        report.print_run_end(result)
        report.print_logs(
            short_path(metrics_path),
            metrics_md5,
            short_path(rollouts_path),
            rollouts_md5,
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


def probability_pair(text: str) -> tuple[float, float]:
    """Read ``--token``: the old and the new probability, e.g. ``0.20,0.30``.

    Args:
        text: What was typed after the flag.

    Returns:
        The old and the new probability.

    Raises:
        argparse.ArgumentTypeError: If it is not two numbers above 0 and at
            most 1.
    """
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{text!r} is not two numbers, OLD,NEW")
    pair = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{part!r} is not a number") from error
        if not 0 < value <= 1:
            raise argparse.ArgumentTypeError(f"{part!r} is not a probability above 0")
        pair.append(value)
    return pair[0], pair[1]


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The run's knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--token",
        type=probability_pair,
        default=None,
        metavar="OLD,NEW",
        help="work out one call's ratio and clip by hand, then stop",
    )
    parser.add_argument(
        "--updates-per-batch",
        type=int,
        default=DEFAULT_UPDATES,
        metavar="N",
        help="how many updates each batch is used for (1 by default; try 4 or 8)",
    )
    parser.add_argument(
        "--stale",
        type=int,
        default=DEFAULT_STALE,
        metavar="N",
        help="sample with the weights from N updates earlier",
    )
    parser.add_argument(
        "--recompute-old",
        action="store_true",
        help="the bug: recompute the old probabilities with the learner's weights",
    )
    parser.add_argument(
        "--no-clip", action="store_true", help="loss = -ratio x A, with no clip"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print the first call the clip switches off, in full",
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="steps (batches) to train for"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0:
        parser.error("steps must be 1 or more, and seed 0 or more")
    if args.updates_per_batch < 1 or args.stale < 0:
        parser.error("updates-per-batch must be 1 or more, and stale 0 or more")
    return args


def main() -> None:
    """Parse the flags, then work out one call or train."""
    args = parse_args()
    if args.token is not None:
        report.print_token(*args.token)
        return
    settings = Settings(
        seed=args.seed,
        lr=args.lr,
        steps=args.steps,
        updates=args.updates_per_batch,
        stale=args.stale,
        recompute_old=args.recompute_old,
        clip=not args.no_clip,
        trace=args.trace,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
        step_pause_s=NO_PAUSE_S if args.fast else STEP_PAUSE_S,
    )
    run(settings)


if __name__ == "__main__":
    main()
