# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""A tiny agent: several calls, one reward, trained with chapter 3's REINFORCE.

Each rollout is one attempt at one question in a new folder. The agent makes
calls (LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE, FINISH) until it says FINISH or
its 8 calls are used up, and only then does chapter 1's grader give a reward.
The policy is a softmax over the actions, computed from a few features of the
agent's situation. After each rollout, one gradient-descent step lowers
chapter 3's loss with a per-question baseline, and every action in the
rollout shares the rollout's advantage, because nothing says which call
earned the reward.

``--memorise`` gives the agent a memory: after each rollout it is shown the
right answer, and a seventh action, RECALL, writes the answer it saw last time
for the same question.

Run it with ``uv run minianalyst.py --trace`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import report
from environment import (
    ACTIONS,
    MEMORY_ACTIONS,
    Call,
    MiniAnalyst,
    NotGraded,
    State,
    fresh_folder,
)
from grader import Grade
from policy import (
    FEATURE_NAMES,
    action_scores,
    features,
    gradient_step,
    greedy,
    log_prob,
    pick,
    rollout_gradient,
    rollout_loss,
    softmax,
)
from questions import TEST, TRAIN, Question, QuestionSet, make_questions
from tasks import ANSWER_FILE, CHAPTER, course_home, short_path, work_root

DEFAULT_SEED = 0
DEFAULT_LR = 0.25  # chapter 3's default; 0.5 sometimes collapses (see README)
DEFAULT_ROLLOUTS = 600
DEFAULT_SHOW = 3  # with --trace: the first three rollouts, then the last
STRETCH = 50  # print how training goes every 50 rollouts
PAUSE_S = 0.4
NO_PAUSE_S = 0.0
QUESTION_STREAM = 0  # the same two random streams as chapters 2 and 3, so
PICK_STREAM = 1  # the same seed asks the same questions in the same order
LOG_DECIMALS = 4
RUNS_DIR = "runs"
RUN_PREFIX = "run-"
MEMORISE_PREFIX = "memorise-"
ROLLOUT_FOLDER = "rollout-{:04d}"
TEST_FOLDER = "greedy-{}-{:04d}"
UNTRAINED = "untrained: every action has the same chance, 1 in {}"
TRAINED = "after {} steps of training"
TRAINING_TITLE = "training questions"
HELD_OUT_TITLE = "cities never used in training"


@dataclass(frozen=True)
class Settings:
    """The knobs of one run (a plain record, read from the flags).

    Attributes:
        seed: Seeds the question order and the random picks; the same seed
            gives the same run.
        lr: The learning rate: how big each gradient-descent step is.
        rollouts: How many rollouts to train for (one step after each).
        memorise: Give the agent a memory of answers and the RECALL action.
        trace: Print the first ``show`` rollouts and the last one in full.
        show: How many of the first rollouts ``trace`` prints.
        pause_s: Seconds to wait after each printed call and stretch.
    """

    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    rollouts: int = DEFAULT_ROLLOUTS
    memorise: bool = False
    trace: bool = False
    show: int = DEFAULT_SHOW
    pause_s: float = NO_PAUSE_S


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
        states: The state before each call (for printing the greedy path).
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
class Update:
    """A training rollout and the gradient-descent step it caused.

    Attributes:
        number: The rollout's number, from 1.
        rollout: What happened.
        baseline: What the agent expected to earn on this question.
        advantage: ``reward - baseline``, shared by every action.
        loss: ``-advantage x`` the sum of the actions' log-probabilities.
        untrained: Whether every weight was still 0 when the rollout ran.
    """

    number: int
    rollout: Rollout
    baseline: float
    advantage: float
    loss: float
    untrained: bool


@dataclass(frozen=True)
class GreedyResult:
    """One question of the greedy test.

    Attributes:
        question: The question asked.
        answer: What answer.txt held at the end, or None without the file.
        correct: The grader's verdict.
        reason: Why the grader said no, or ``"correct"``.
        actions: The actions of every call, in order.
        observations: What every call sent back, in order.
    """

    question: Question
    answer: str | None
    correct: bool
    reason: str
    actions: tuple[str, ...]
    observations: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        weights: The trained weights: one row per feature, one column per
            action.
        updates: Every training rollout and its step, in order.
        training_test: The greedy test on the training questions.
        held_out_test: The greedy test on the held-out cities.
        log_path: Where the run log was written.
        digest: The run log's md5, the same for the same settings.
    """

    weights: np.ndarray
    updates: tuple[Update, ...]
    training_test: tuple[GreedyResult, ...]
    held_out_test: tuple[GreedyResult, ...]
    log_path: Path
    digest: str


class QuestionBaseline:
    """Chapter 3's per-question baseline: the average reward on this question.

    A question here is one of the four kinds (any city), as in chapters 2 and
    3. Before a question has any earlier reward there is nothing to compare
    with, so the baseline is the rollout's own reward: advantage 0.
    """

    def __init__(self) -> None:
        """Start with no rewards seen."""
        self._totals: defaultdict[int, float] = defaultdict(float)
        self._counts: defaultdict[int, int] = defaultdict(int)

    def expect(self, row: int, reward: float) -> float:
        """Return what the agent expected to earn on this question.

        Args:
            row: The question's kind (0 to 3).
            reward: This rollout's reward, used only when there is nothing to
                average yet.

        Returns:
            The average reward of the earlier rollouts of this question, or
            this rollout's own reward before there are any.
        """
        if self._counts[row] == 0:
            return reward
        return self._totals[row] / self._counts[row]

    def add(self, row: int, reward: float) -> None:
        """Count a rollout's reward in its question's average, after its step.

        Args:
            row: The question's kind.
            reward: The rollout's reward.
        """
        self._totals[row] += reward
        self._counts[row] += 1


@contextmanager
def run_folder() -> Iterator[Path]:
    """Make the folder this run's rollout folders go in, and remove it after.

    Its name is random, so two runs at once never share a folder; the rollout
    folders inside are numbered, so the printed output repeats exactly.

    Yields:
        The run's folder under ``$COURSE_HOME/04-episodes/work``.

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
    memory: Mapping[str, str] | None,
) -> Rollout:
    """Run one rollout in its own new folder, call by call, until it stops.

    Args:
        weights: The policy's weights.
        question: The question to answer.
        parent: The run's folder; the rollout's folder is made inside it.
        name: The rollout folder's name, e.g. ``"rollout-0001"``.
        choose: Turns action scores into an action: a random pick while
            training, the top score in the greedy test.
        memory: Remembered answers for the RECALL action, or None.

    Returns:
        Everything that happened, with the grader's verdict.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If the rollout stopped without a verdict (never expected).
    """
    actions = ACTIONS if memory is None else MEMORY_ACTIONS
    situations, scores, choices, states, chances, log_probs = [], [], [], [], [], []
    with fresh_folder(parent, name) as folder:
        env = MiniAnalyst(question, folder, memory)
        while not env.done:
            states.append(replace(env.state))  # a copy, as it is before the call
            situation = features(env.state)
            row = action_scores(weights, situation)
            choice = choose(row)
            env.step(actions[choice])
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
    """Round numbers for the run log, so the md5 does not hang on tiny digits.

    Args:
        values: A number, or nested lists or arrays of numbers.

    Returns:
        The same shape, as plain rounded floats.
    """
    if isinstance(values, np.ndarray | list | tuple):
        return [rounded(value) for value in values]
    return round(float(values), LOG_DECIMALS)


def record(update: Update) -> dict[str, Any]:
    """Describe one training rollout for the run log.

    Args:
        update: The rollout and its step.

    Returns:
        One JSON-ready record.
    """
    rollout = update.rollout
    return {
        "rollout": update.number,
        "question": rollout.question.task.question,
        "calls": [[call.action, call.observation] for call in rollout.calls],
        "p": rounded(rollout.chances),
        "stop": rollout.stop,
        "answer": rollout.verdict.submitted,
        "reward": rollout.reward,
        "baseline": rounded(update.baseline),
        "advantage": rounded(update.advantage),
        "loss": rounded(update.loss),
    }


def train(
    settings: Settings,
    questions: QuestionSet,
    parent: Path,
    log: list[dict[str, Any]],
    printing: bool = True,
) -> tuple[np.ndarray, tuple[Update, ...], dict[str, str] | None]:
    """Run the loop: a rollout, its reward, one gradient-descent step, again.

    Args:
        settings: The run's knobs.
        questions: The fair questions; only the training ones are asked.
        parent: The run's folder, where every rollout gets its own folder.
        log: A list every rollout and stretch is appended to.
        printing: Print the stretches (and, with ``trace``, rollouts in full).

    Returns:
        The trained weights, every rollout and its step, and the memory
        (None without ``memorise``).

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    actions = MEMORY_ACTIONS if settings.memorise else ACTIONS
    weights = np.zeros((len(FEATURE_NAMES), len(actions)))  # every action equal
    memory: dict[str, str] | None = {} if settings.memorise else None
    baseline = QuestionBaseline()
    question_rng = np.random.default_rng([settings.seed, QUESTION_STREAM])
    pick_rng = np.random.default_rng([settings.seed, PICK_STREAM])
    updates: list[Update] = []
    # Stretches of 50 rollouts (the last may be shorter): 0-50, 50-100, ...
    stretches = [0, *range(STRETCH, settings.rollouts, STRETCH), settings.rollouts]
    for start, end in pairwise(stretches):
        for number in range(start + 1, end + 1):
            asked = question_rng.integers(len(questions.training))
            question = questions.training[asked]
            untrained = not weights.any()
            rollout = run_rollout(
                weights,
                question,
                parent,
                ROLLOUT_FOLDER.format(number),
                lambda row: pick(row, pick_rng),
                memory,
            )
            expected = baseline.expect(question.row, rollout.reward)
            advantage = rollout.reward - expected
            gradient = rollout_gradient(
                rollout.situations, rollout.scores, rollout.choices, advantage
            )
            weights = gradient_step(weights, gradient, settings.lr)
            baseline.add(question.row, rollout.reward)
            if memory is not None:
                # The memoriser is shown the right answer after every rollout.
                memory[question.task.task_id] = question.task.answer
            update = Update(
                number,
                rollout,
                expected,
                advantage,
                rollout_loss(advantage, rollout.log_probs),
                untrained,
            )
            updates.append(update)
            log.append(record(update))
            if printing and settings.trace and number <= settings.show:
                report.print_rollout(update, when(update, actions), settings.pause_s)
        if printing and start == 0:
            print()
        log.append({"after": end, "correct": sum(u.rollout.reward for u in updates)})
        if printing:
            report.print_stretch(start + 1, end, updates[start:end])
            time.sleep(settings.pause_s)
    if printing and settings.trace and settings.rollouts > settings.show:
        last = updates[-1]
        report.print_rollout(last, when(last, actions), settings.pause_s)
    return weights, tuple(updates), memory


def when(update: Update, actions: tuple[str, ...]) -> str:
    """Say how trained the agent was when a rollout ran.

    Args:
        update: The rollout and its step.
        actions: The actions the agent may take.

    Returns:
        ``"untrained: ..."`` while every weight is 0, else how many
        gradient-descent steps came before it.
    """
    if update.untrained:
        return UNTRAINED.format(len(actions))
    return TRAINED.format(update.number - 1)


def greedy_test(
    weights: np.ndarray,
    questions: tuple[Question, ...],
    parent: Path,
    memory: Mapping[str, str] | None,
    log: list[dict[str, Any]],
    split: str,
) -> tuple[tuple[GreedyResult, ...], tuple[Rollout, ...]]:
    """Answer every question greedily, each in its own new folder.

    Args:
        weights: The trained weights.
        questions: The questions to ask.
        parent: The run's folder.
        memory: The memoriser's remembered answers, or None.
        log: The run log, which gets one record per question.
        split: ``"train"`` or ``"test"``, for the folder names.

    Returns:
        One result per question, and every rollout.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    results = []
    rollouts = []
    for number, question in enumerate(questions, start=1):
        name = TEST_FOLDER.format(split, number)
        rollout = run_rollout(weights, question, parent, name, greedy, memory)
        rollouts.append(rollout)
        verdict = rollout.verdict
        results.append(
            GreedyResult(
                question=question,
                answer=verdict.submitted if ANSWER_FILE in rollout.files else None,
                correct=verdict.correct,
                reason=verdict.reason,
                actions=tuple(call.action for call in rollout.calls),
                observations=tuple(call.observation for call in rollout.calls),
            )
        )
        log.append(
            {
                "test": question.task.question,
                "calls": [call.action for call in rollout.calls],
                "answer": verdict.submitted,
                "correct": verdict.correct,
            }
        )
    return tuple(results), tuple(rollouts)


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


def log_name(settings: Settings) -> str:
    """Name a run's log after the settings that decide its contents.

    Args:
        settings: The run's knobs.

    Returns:
        For example ``"seed0-lr0.25-rollouts600.jsonl"``.
    """
    memorise = MEMORISE_PREFIX if settings.memorise else ""
    return (
        f"{memorise}seed{settings.seed}-lr{settings.lr:g}"
        f"-rollouts{settings.rollouts}.jsonl"
    )


def run(settings: Settings) -> RunResult:
    """Train, print how it went, test greedily on both splits, write the log.

    Args:
        settings: The run's knobs.

    Returns:
        The weights, every rollout, both tests and the log's md5.

    Raises:
        OSError: If a folder, a file or the run log cannot be written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    questions = make_questions()
    actions = MEMORY_ACTIONS if settings.memorise else ACTIONS
    report.print_header(settings, questions, actions)
    log: list[dict[str, Any]] = []
    with run_folder() as parent:
        weights, updates, memory = train(settings, questions, parent, log)
        training_test, seen = greedy_test(
            weights, questions.training, parent, memory, log, TRAIN
        )
        held_out_test, _ = greedy_test(
            weights, questions.held_out, parent, memory, log, TEST
        )
    report.print_greedy_path(weights, seen[0].states, actions)
    print()
    report.print_test(TRAINING_TITLE, training_test, show=False)
    report.print_test(HELD_OUT_TITLE, held_out_test, show=True)
    log.append({"weights": rounded(weights)})
    log_path = course_home() / CHAPTER / RUNS_DIR / log_name(settings)
    digest = write_jsonl(log_path, log)
    print(f"\nwrote {short_path(log_path)}")
    print(f"md5   {digest}")
    return RunResult(weights, updates, training_test, held_out_test, log_path, digest)


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
        help="print the first rollouts and the last one in full",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=DEFAULT_SHOW,
        help="with --trace, how many of the first rollouts to print",
    )
    parser.add_argument(
        "--memorise",
        action="store_true",
        help="show the right answer after each rollout and add RECALL",
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=DEFAULT_ROLLOUTS,
        help="rollouts to train for, one step after each",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    if args.rollouts < 1 or args.show < 0 or args.seed < 0:
        parser.error("rollouts must be 1 or more, and show and seed 0 or more")
    return args


def main() -> None:
    """Parse the flags, then train and test."""
    args = parse_args()
    settings = Settings(
        seed=args.seed,
        lr=args.lr,
        rollouts=args.rollouts,
        memorise=args.memorise,
        trace=args.trace,
        show=args.show,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
    )
    run(settings)


if __name__ == "__main__":
    main()
