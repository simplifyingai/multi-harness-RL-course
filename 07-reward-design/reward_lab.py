# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Five reward rules, each trained with chapter 6's GRPO on the tiny agent.

The agent is chapter 6's, with these changes: SUBMIT hands in the result in
hand (chapter 6's WRITE and FINISH in one action), and SUBMIT_GUESS hands in
a guess at once; every reply has a length in tokens, drawn around a typical
length the agent learns (a long reply holds two actions, and one past the
output cap is cut off before its actions run, which ends the rollout); and,
with ``--crashes``, sandboxes die before grading. One set of weights serves
every level (chapter 5's ``--shared``).

``--reward`` picks what the agent is paid for: ``correct``, ``submit``,
``call-bonus``, ``token-bonus`` or ``bonus-when-wrong``. GRPO learns to raise
the reward, whatever it measures, and each rule teaches something different.
``--show-group`` works out one group's advantages by hand and stops.

Run it with ``uv run reward_lab.py --reward call-bonus`` (``--fast`` skips
the pauses).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import report
from environment import (
    ACTIONS,
    CRASHED,
    GUESSED,
    Call,
    MiniAnalyst,
    NotGraded,
    fresh_folder,
)
from grader import Grade
from grpo import (
    GROUP_SIZE,
    QUESTIONS_PER_STEP,
    Trained,
    Weights,
    apply_update,
    group_advantages,
    step_update,
)
from policy import action_scores, features, greedy, log_prob, pick, weight_rows
from questions import (
    GUESS,
    LEVELS,
    TEST,
    TRAIN,
    Question,
    QuestionSet,
    guess_hits,
    make_questions,
)
from replies import (
    START_WEIGHT,
    length_log_prob,
    reply_tokens,
    sample_length,
    typical_length,
)
from rewards import CORRECT, LEFT_OUT, RULES, SCORED_ZERO, Outcome, reward
from tasks import CHAPTER, course_home, short_path, work_root

DEFAULT_SEED = 0
DEFAULT_LR = 3.0  # chapter 5's learning rate, for the action weights
DEFAULT_LENGTH_LR = 0.3  # the length weight's learning rate
SHARED = True  # one set of weights for every level (chapter 5's --shared)
DEFAULT_STEPS = 200
DEFAULT_CRASHES = 0.0
PAUSE_S = 0.4  # after a rollout printed in full
STEP_PAUSE_S = 0.05  # after each step's line
NO_PAUSE_S = 0.0
PICK_STREAM = 1  # the random stream for the agent's picks (chapter 5's)
CRASH_STREAM = 2  # a separate stream decides which sandboxes die
LOG_DECIMALS = 4
STAGE = "grpo"
RUNS_DIR = "runs"
RUN_PREFIX = "run-"
METRICS_FILE = "metrics.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"
ROLLOUT_FOLDER = "step-{:03d}-group-{}-rollout-{}"
TEST_FOLDER = "greedy-{}-{:03d}"
# An all-right group paid by the token bonus for 80, 160, 120 and 200 tokens.
SHOW_GROUP = (1.08, 1.06, 1.07, 1.05)
CRASH_SCORE = 0.0  # what a dead sandbox earns, unless it is left unscored
MAX_CHANCE = 1.0  # --crashes is a chance, at most 1


@dataclass(frozen=True)
class Settings:
    """The knobs of one run (a plain record, read from the flags).

    Attributes:
        rule: The reward rule, one of ``rewards.RULES``.
        seed: Seeds the question order and the random picks; the same seed
            gives the same run.
        lr: The action weights' learning rate.
        length_lr: The length weight's learning rate.
        steps: How many steps to train for (one batch, one update each).
        guessable: Keep the questions a guess of 0 answers.
        crashes: The chance that a rollout's sandbox dies before grading.
        unscored: Leave a crashed rollout out of its group instead of
            scoring it 0.
        trace: Print rollouts in full at a few moments of the run.
        pause_s: Seconds to wait after a rollout printed in full.
        step_pause_s: Seconds to wait after each step's line.
    """

    rule: str = CORRECT
    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    length_lr: float = DEFAULT_LENGTH_LR
    steps: int = DEFAULT_STEPS
    guessable: bool = False
    crashes: float = DEFAULT_CRASHES
    unscored: bool = False
    trace: bool = False
    pause_s: float = NO_PAUSE_S
    step_pause_s: float = NO_PAUSE_S


@dataclass(frozen=True)
class Rollout:
    """One attempt at one question, from its first call to its reward.

    Attributes:
        question: The question answered.
        folder: The name of the rollout's own folder.
        calls: Every call: its tokens, the actions it held, what came back.
        situations: The features the agent saw before each action.
        choices: Each action taken (an index into the actions).
        lengths: Each reply length drawn, before rounding.
        old_logprobs: ``log p`` of every decision as it was sampled: the
            actions first, then the lengths.
        stop: Why the rollout stopped.
        verdict: The grader's verdict, or None when the sandbox died.
        lost: When the sandbox died, the verdict it would have had (only the
            report reads it; training never sees it).
        reward: The rule's reward, or None when a crash is left unscored.
    """

    question: Question
    folder: str
    calls: tuple[Call, ...]
    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    lengths: tuple[float, ...]
    old_logprobs: tuple[float, ...]
    stop: str
    verdict: Grade | None
    lost: Grade | None
    reward: float | None

    @property
    def correct(self) -> bool:
        """Whether the grader accepted the answer (never, when it crashed)."""
        return self.verdict is not None and self.verdict.correct

    @property
    def tokens(self) -> int:
        """Every token of every reply, added up."""
        return sum(call.tokens for call in self.calls)


@dataclass(frozen=True)
class Group:
    """One question's group: its rollouts, their rewards and advantages.

    Attributes:
        step: The step the group belongs to, from 1.
        number: Its place in the step, from 1.
        question: The question all its rollouts answered.
        rollouts: The rollouts, in order.
        advantages: One advantage per rollout (None for a rollout left out),
            or None when the group has no contrast.
    """

    step: int
    number: int
    question: Question
    rollouts: tuple[Rollout, ...]
    advantages: tuple[float | None, ...] | None

    @property
    def rewards(self) -> list[float | None]:
        """The rollouts' rewards, in order (None for a crash left unscored)."""
        return [rollout.reward for rollout in self.rollouts]


@dataclass(frozen=True)
class Step:
    """One step: its groups, its update, and the weights after it.

    Attributes:
        number: The step's number, from 1.
        groups: Its groups, in order.
        decisions: How many decisions the update learned from.
        ratios: Every decision's ratio, new p / old p.
        clipped: For every decision, whether the clip switched it off.
        typical_length: The typical reply length after the step's update.
    """

    number: int
    groups: tuple[Group, ...]
    decisions: int
    ratios: np.ndarray
    clipped: np.ndarray
    typical_length: float

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
        calls: The calls of the first greedy rollout of that level.
    """

    level: str
    training: int
    training_total: int
    held_out: int
    held_out_total: int
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        settings: The knobs it ran with.
        questions: The questions it trained and was tested on.
        guess: What SUBMIT_GUESS wrote, and how many training questions that
            answer is right for.
        weights: The trained weights.
        steps: Every step, in order.
        edges: Where the quarters start and end: steps ``edges[i] + 1`` to
            ``edges[i + 1]``.
        scores: The greedy test at the end, per level.
        greedy_tokens: How long every greedy reply is (the typical length).
        metrics_path: Where metrics.jsonl was written.
        metrics_md5: Its md5.
        rollouts_path: Where rollouts.jsonl was written.
        rollouts_md5: Its md5.
    """

    settings: Settings
    questions: QuestionSet
    guess: tuple[str, int]
    weights: Weights
    steps: tuple[Step, ...]
    edges: tuple[int, ...]
    scores: tuple[LevelScore, ...]
    greedy_tokens: int
    metrics_path: Path
    metrics_md5: str
    rollouts_path: Path
    rollouts_md5: str


@dataclass(frozen=True)
class Chooser:
    """How a rollout picks: by chance while training, the top in the test.

    Attributes:
        action: Turns action scores into an action.
        length: Turns the length weight into one reply's log-length.
    """

    action: Callable[[np.ndarray], int]
    length: Callable[[float], float]


GREEDY = Chooser(greedy, lambda w: w)  # the top action, the typical length


@contextmanager
def run_folder() -> Iterator[Path]:
    """Make the folder this run's rollout folders go in, and remove it after.

    Its name is random, so two runs at once never share a folder; the rollout
    folders inside are numbered, so the printed output repeats exactly.

    Yields:
        The run's folder under ``$COURSE_HOME/07-reward-design/work``.

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


def sampler(rng: np.random.Generator) -> Chooser:
    """Pick by chance: actions by their probabilities, lengths off the curve.

    Args:
        rng: The random generator for picks.

    Returns:
        The training chooser.
    """
    return Chooser(lambda row: pick(row, rng), lambda w: sample_length(w, rng))


@dataclass
class Decisions:
    """Every decision of a rollout, each with its log p as it was sampled.

    Attributes:
        situations: The features before each action.
        choices: Each action taken (an index into the actions).
        action_logps: ``log p`` of each action, with the weights that chose.
        lengths: Each reply's drawn log-length.
        length_logps: ``log p`` of each of those.
    """

    situations: list[np.ndarray] = field(default_factory=list)
    choices: list[int] = field(default_factory=list)
    action_logps: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    length_logps: list[float] = field(default_factory=list)


def play_call(
    env: MiniAnalyst, weights: Weights, choose: Chooser, made: Decisions
) -> None:
    """Make one call: draw the reply's length, then pick the actions it holds.

    A reply past the output cap holds no action. Otherwise the agent picks
    one or two, the second from the state the first left.

    Args:
        env: The rollout's environment.
        weights: The weights the agent acts with.
        choose: How actions and lengths are picked.
        made: Where every decision is recorded.

    Raises:
        OSError: If answer.txt cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    length = choose.length(weights.length)
    made.lengths.append(length)
    # Recorded now, with the weights that chose: the "old" log p.
    made.length_logps.append(length_log_prob(length, weights.length))
    room = env.open_call(reply_tokens(length))
    for _ in range(room):
        situation = features(env.state, SHARED)
        row = action_scores(weights.actions, situation)
        choice = choose.action(row)
        made.situations.append(situation)
        made.choices.append(choice)
        made.action_logps.append(log_prob(row, choice))
        env.act(ACTIONS[choice])
        if env.done:
            break
    env.close_call()


def payment(rule: str, env: MiniAnalyst, crash: str | None) -> float | None:
    """Pay a finished rollout by the rule, or by what its crash means.

    Args:
        rule: The reward rule.
        env: The rollout's environment, stopped and graded.
        crash: None, ``SCORED_ZERO`` or ``LEFT_OUT``.

    Returns:
        The reward, or None for a crash left out of its group.
    """
    if crash == LEFT_OUT:
        return None
    if crash == SCORED_ZERO:
        return CRASH_SCORE
    verdict = env.grade
    outcome = Outcome(
        correct=verdict is not None and verdict.correct,
        submitted=verdict is not None and bool(verdict.submitted),
        calls=len(env.calls),
        tokens=env.tokens,
    )
    return reward(rule, outcome)


def run_rollout(
    weights: Weights,
    question: Question,
    parent: Path,
    name: str,
    guess: str,
    choose: Chooser,
    rule: str,
    crash: str | None = None,
) -> Rollout:
    """Run one rollout in its own new folder, call by call, until it stops.

    Args:
        weights: The weights the agent acts with.
        question: The question to answer.
        parent: The run's folder; the rollout's folder is made inside it.
        name: The rollout folder's name.
        guess: What SUBMIT_GUESS hands in.
        choose: How actions and lengths are picked.
        rule: The reward rule.
        crash: None, or how a sandbox that dies before grading is treated:
            ``SCORED_ZERO`` or ``LEFT_OUT``.

    Returns:
        Everything that happened, with each decision's log-probability as it
        was sampled, the verdict and the reward.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If the rollout stopped without a verdict (never expected).
    """
    made = Decisions()
    with fresh_folder(parent, name) as folder:
        env = MiniAnalyst(question, folder, guess)
        while not env.done:
            play_call(env, weights, choose, made)
    if env.state.stop is None or env.grade is None:
        raise NotGraded(f"{name} stopped without a verdict")
    paid = payment(rule, env, crash)
    lost = None
    if crash is not None:
        lost = env.grade  # kept for the report; training never sees it
        env.crash()
    return Rollout(
        question=question,
        folder=name,
        calls=tuple(env.calls),
        situations=tuple(made.situations),
        choices=tuple(made.choices),
        lengths=tuple(made.lengths),
        old_logprobs=tuple(made.action_logps + made.length_logps),
        stop=env.state.stop,
        verdict=env.grade,
        lost=lost,
        reward=paid,
    )


def scored_advantages(
    rewards: Sequence[float | None],
) -> tuple[float | None, ...] | None:
    """GRPO's advantages over the rollouts that have a reward.

    A crash left unscored is left out: the group's mean and spread come from
    the others, and it gets no advantage, so it teaches nothing.

    Args:
        rewards: One reward per rollout, None for a rollout left out.

    Returns:
        One advantage per rollout (None where left out), or None when the
        scored rollouts have no contrast.
    """
    scored = [r for r in rewards if r is not None]
    advantages = group_advantages(scored)
    if advantages is None:
        return None
    given = iter(advantages)
    return tuple(None if r is None else next(given) for r in rewards)


def run_group(
    weights: Weights,
    question: Question,
    step: int,
    place: int,
    parent: Path,
    settings: Settings,
    guess: str,
    rngs: tuple[np.random.Generator, np.random.Generator],
) -> Group:
    """Run one question's group: GROUP_SIZE rollouts, their rewards and advantages.

    Args:
        weights: The weights, the same for every rollout.
        question: The question every rollout answers.
        step: The step's number.
        place: The group's place in the step, from 1.
        parent: The run's folder.
        settings: The run's knobs.
        guess: What SUBMIT_GUESS hands in.
        rngs: The random generators for the picks and for the crashes.

    Returns:
        The group, with None for advantages when its rewards tie.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    pick_rng, crash_rng = rngs
    choose = sampler(pick_rng)
    rollouts = []
    for index in range(1, GROUP_SIZE + 1):
        crash = None
        if settings.crashes and crash_rng.random() < settings.crashes:
            crash = LEFT_OUT if settings.unscored else SCORED_ZERO
        rollouts.append(
            run_rollout(
                weights,
                question,
                parent,
                ROLLOUT_FOLDER.format(step, place, index),
                guess,
                choose,
                settings.rule,
                crash,
            )
        )
    advantages = scored_advantages([r.reward for r in rollouts])
    return Group(step, place, question, tuple(rollouts), advantages)


def trained_rollouts(groups: Sequence[Group]) -> list[Trained]:
    """Collect what the step learns from, with the log p recorded at sampling.

    Groups with no contrast are skipped, and so is a crash left unscored.
    Every decision of every other rollout shares its rollout's advantage.

    Args:
        groups: The step's groups.

    Returns:
        One item per rollout learned from.
    """
    return [
        Trained(r.situations, r.choices, r.lengths, r.old_logprobs, advantage)
        for group in groups
        if group.advantages is not None
        for r, advantage in zip(group.rollouts, group.advantages, strict=True)
        if advantage is not None
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


def step_metrics(step: Step) -> dict[str, Any]:
    """Describe one step for metrics.jsonl, with the project's field names.

    plot.py reads these fields by name, so renaming one breaks the plot.

    Args:
        step: The step.

    Returns:
        One JSON-ready record: everything plot.py needs to redraw the run.
    """
    rollouts = step.rollouts
    calls = [call for r in rollouts for call in r.calls]
    paid = [r.reward for r in rollouts if r.reward is not None]
    return {
        "stage": STAGE,
        "step": step.number,
        "accuracy": sum(r.correct for r in rollouts) / len(rollouts),
        "mean_reward": sum(paid) / len(paid) if paid else None,
        "mean_calls": len(calls) / len(rollouts),
        "groups_without_contrast": step.no_signal,
        "groups": len(step.groups),
        "group_levels": [group.question.level for group in step.groups],
        "group_rewards": [rounded(group.rewards) for group in step.groups],
        "rollouts": len(rollouts),
        "scored": len(paid),
        "correct": sum(r.correct for r in rollouts),
        "replies": len(calls),
        "reply_tokens": sum(call.tokens for call in calls),
        "cut_off_replies": sum(call.cut_off for call in calls),
        "two_action_replies": sum(call.long for call in calls),
        "guesses": sum(r.stop == GUESSED for r in rollouts),
        "crashed": sum(r.stop == CRASHED for r in rollouts),
        "typical_length": rounded(step.typical_length),
        "train_decisions": step.decisions,
        "importance_ratio_mean": rounded(step.ratios.mean())
        if step.ratios.size
        else None,
        "clipped_decisions": int(step.clipped.sum()),
    }


def rollout_records(group: Group) -> list[dict[str, Any]]:
    """Describe a group's rollouts for the rollout log, one record each.

    Args:
        group: The group.

    Returns:
        One JSON-ready record per rollout.
    """
    advantages = group.advantages or (None,) * len(group.rollouts)
    return [
        {
            "step": group.step,
            "group": group.number,
            "level": group.question.level,
            "question": group.question.task.question,
            "calls": [
                [call.tokens, list(call.actions), list(call.observations)]
                for call in rollout.calls
            ],
            "stop": rollout.stop,
            "answer": None if rollout.verdict is None else rollout.verdict.submitted,
            "correct": rollout.correct,
            "reward": rounded(rollout.reward),
            "advantage": rounded(advantage),
        }
        for rollout, advantage in zip(group.rollouts, advantages, strict=True)
    ]


def train(
    settings: Settings,
    questions: QuestionSet,
    guess: str,
    parent: Path,
    log: list[dict[str, Any]],
    printing: bool = True,
) -> tuple[Weights, tuple[Step, ...]]:
    """Run GRPO under one reward rule: sample, pay, update, again.

    Args:
        settings: The run's knobs.
        questions: The questions; it trains on the training ones.
        guess: What SUBMIT_GUESS hands in.
        parent: The run's folder, where every rollout gets its own folder.
        log: A list every rollout is appended to.
        printing: Print one line per step (and, with ``trace``, rollouts).

    Returns:
        The trained weights and every step.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    weights = Weights(np.zeros((weight_rows(SHARED), len(ACTIONS))), START_WEIGHT)
    order = list(questions.training)
    random.Random(settings.seed).shuffle(order)
    rngs = (
        np.random.default_rng([settings.seed, PICK_STREAM]),
        np.random.default_rng([settings.seed, CRASH_STREAM]),
    )
    tracer = report.Tracer(settings.steps) if printing and settings.trace else None
    steps: list[Step] = []
    cursor = 0
    for number in range(1, settings.steps + 1):
        groups: list[Group] = []
        for place in range(1, QUESTIONS_PER_STEP + 1):
            question = order[cursor % len(order)]
            cursor += 1
            group = run_group(
                weights, question, number, place, parent, settings, guess, rngs
            )
            groups.append(group)
            log.extend(rollout_records(group))
        trained = trained_rollouts(groups)
        # One fresh batch and one update per step, as the project does: every
        # ratio is exactly 1 and chapter 6's clip never acts.
        update = step_update(trained, weights)
        weights = apply_update(
            weights, update.gradient, settings.lr, settings.length_lr
        )
        step = Step(
            number,
            tuple(groups),
            update.decisions,
            update.ratios,
            update.clipped,
            typical_length(weights.length),
        )
        steps.append(step)
        if printing:
            report.print_step(step, step_metrics(step))
            time.sleep(settings.step_pause_s)
            if tracer is not None and tracer.see(step):
                time.sleep(settings.pause_s)
    return weights, tuple(steps)


def stretch_edges(steps: int) -> tuple[int, ...]:
    """Split a run into (at most) four stretches of steps.

    Args:
        steps: How many steps the run has.

    Returns:
        The boundaries, e.g. ``(0, 50, 100, 150, 200)`` for 200 steps.
    """
    size = math.ceil(steps / report.QUARTERS)
    return (*range(0, steps, size), steps)


def greedy_test(
    weights: Weights,
    questions: QuestionSet,
    guess: str,
    rule: str,
    parent: Path,
    log: list[dict[str, Any]],
) -> tuple[LevelScore, ...]:
    """Answer every question greedily, each in its own new folder.

    Greedy means the most likely action at every pick and the mean length
    for every reply.

    Args:
        weights: The trained weights.
        questions: The training and held-out questions.
        guess: What SUBMIT_GUESS hands in.
        rule: The reward rule (the test only reports right or wrong).
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
    for split, chosen in ((TRAIN, questions.training), (TEST, questions.held_out)):
        rollouts = []
        for number, question in enumerate(chosen, start=1):
            name = TEST_FOLDER.format(split, number)
            rollout = run_rollout(weights, question, parent, name, guess, GREEDY, rule)
            rollouts.append((question, rollout))
            log.append(
                {
                    "test": question.task.question,
                    "level": question.level,
                    "calls": [
                        [call.tokens, list(call.actions)] for call in rollout.calls
                    ],
                    "answer": None
                    if rollout.verdict is None
                    else rollout.verdict.submitted,
                    "correct": rollout.correct,
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
                training=sum(r.correct for r in mine[TRAIN]),
                training_total=len(mine[TRAIN]),
                held_out=sum(r.correct for r in mine[TEST]),
                held_out_total=len(mine[TEST]),
                # Every level has training questions in this toy; the
                # guard only keeps a changed split from crashing the print.
                calls=mine[TRAIN][0].calls if mine[TRAIN] else (),
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
        For example ``"call-bonus-seed0-lr3-length-lr0.3-steps200"``, with
        ``-guessable`` and ``-crashes0.1`` (and ``-unscored``) when on.
    """
    name = (
        f"{settings.rule}-seed{settings.seed}-lr{settings.lr:g}"
        f"-length-lr{settings.length_lr:g}-steps{settings.steps}"
    )
    if settings.guessable:
        name += "-guessable"
    if settings.crashes:
        name += f"-crashes{settings.crashes:g}"
        if settings.unscored:
            name += "-unscored"
    return name


def run(settings: Settings, printing: bool = True) -> RunResult:
    """Train under one rule, print each step, test greedily, write the logs.

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
    questions = make_questions(settings.guessable)
    guess = (GUESS, guess_hits(questions.training, GUESS))
    if printing:
        report.print_header(settings, questions, guess)
    log: list[dict[str, Any]] = []
    with run_folder() as parent:
        weights, steps = train(settings, questions, guess[0], parent, log, printing)
        scores = greedy_test(weights, questions, guess[0], settings.rule, parent, log)
    edges = stretch_edges(settings.steps)
    log.append(
        {
            "weights": rounded(weights.actions),
            "length_weight": rounded(weights.length),
        }
    )
    folder = course_home() / CHAPTER / RUNS_DIR / run_name(settings)
    metrics_path = folder / METRICS_FILE
    metrics_md5 = write_jsonl(metrics_path, [step_metrics(step) for step in steps])
    rollouts_path = folder / ROLLOUTS_FILE
    rollouts_md5 = write_jsonl(rollouts_path, log)
    result = RunResult(
        settings,
        questions,
        guess,
        weights,
        steps,
        edges,
        scores,
        reply_tokens(weights.length),
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


def whole_number(text: str, smallest: int) -> int:
    """Read a flag that must be a whole number, at least ``smallest``.

    Args:
        text: What was typed after the flag.
        smallest: The smallest number allowed.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If it is not a whole number that large.
    """
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from error
    if value < smallest:
        raise argparse.ArgumentTypeError(f"{text!r} must be {smallest} or more")
    return value


def positive(text: str) -> int:
    """Read a flag that must be a whole number, 1 or more (``--steps``).

    Args:
        text: What was typed after the flag.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If it is not a whole number of 1 or more.
    """
    return whole_number(text, 1)


def seed_number(text: str) -> int:
    """Read ``--seed``: a whole number, 0 or more.

    Args:
        text: What was typed after the flag.

    Returns:
        The seed.

    Raises:
        argparse.ArgumentTypeError: If it is not a whole number of 0 or more.
    """
    return whole_number(text, 0)


def probability(text: str) -> float:
    """Read ``--crashes``: a chance between 0 and 1.

    Args:
        text: What was typed after the flag.

    Returns:
        The chance.

    Raises:
        argparse.ArgumentTypeError: If it is not a number from 0 to 1.
    """
    value = non_negative(text)
    if value > MAX_CHANCE:
        raise argparse.ArgumentTypeError(f"{text!r} is not a chance from 0 to 1")
    return value


def reward_list(text: str) -> tuple[float, ...]:
    """Read ``--show-group``: a group's rewards, e.g. ``1.08,1.06,1.07,1.05``.

    Args:
        text: What was typed after the flag.

    Returns:
        The rewards.

    Raises:
        argparse.ArgumentTypeError: If they are not at least two numbers.
    """
    parts = text.split(",")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"{text!r} is not two or more rewards")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{part!r} is not a number") from error
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"{part!r} is not a finite number")
        values.append(value)
    return tuple(values)


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The run's knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reward", choices=RULES, default=CORRECT, help="what the agent is paid for"
    )
    parser.add_argument(
        "--show-group",
        type=reward_list,
        nargs="?",
        const=SHOW_GROUP,
        default=None,
        metavar="R,R,R,R",
        help="work out one group's advantages by hand, then stop",
    )
    parser.add_argument(
        "--guessable",
        action="store_true",
        help="add a question whose answer is the same for nearly every city",
    )
    parser.add_argument(
        "--crashes",
        type=probability,
        default=DEFAULT_CRASHES,
        metavar="P",
        help="the chance a rollout's sandbox dies before grading (scored 0)",
    )
    parser.add_argument(
        "--unscored",
        action="store_true",
        help="leave a crashed rollout out of its group instead of scoring it 0",
    )
    parser.add_argument(
        "--trace", action="store_true", help="print rollouts in full along the way"
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="action weights' step size"
    )
    parser.add_argument(
        "--length-lr",
        type=non_negative,
        default=DEFAULT_LENGTH_LR,
        help="the length weight's step size",
    )
    parser.add_argument(
        "--steps", type=positive, default=DEFAULT_STEPS, help="steps to train for"
    )
    parser.add_argument(
        "--seed", type=seed_number, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    if args.unscored and not args.crashes:
        parser.error("--unscored needs --crashes")
    return args


def main() -> None:
    """Parse the flags, then work out one group or train."""
    args = parse_args()
    if args.show_group is not None:
        report.print_show_group(list(args.show_group))
        return
    settings = Settings(
        rule=args.reward,
        seed=args.seed,
        lr=args.lr,
        length_lr=args.length_lr,
        steps=args.steps,
        guessable=args.guessable,
        crashes=args.crashes,
        unscored=args.unscored,
        trace=args.trace,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
        step_pause_s=NO_PAUSE_S if args.fast else STEP_PAUSE_S,
    )
    run(settings)


if __name__ == "__main__":
    main()
