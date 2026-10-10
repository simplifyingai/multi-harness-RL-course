# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Policy gradient: chapter 2's bandit, written as a loss, trained by gradient descent.

Same four questions, same four queries each, same grader and same city split
as chapter 2. Each try picks a query, gets the grader's reward, and turns it
into an advantage: the reward minus a baseline (what the agent expected). The
loss for the try is ``-advantage * log p(chosen)``, and one gradient-descent
step moves the row's scores against the loss's slope. With no baseline the
advantage is the reward, and every step is chapter 2's nudge, bit for bit.

Run it with ``uv run reinforce.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import report
from grader import Grade
from policy import (
    chances,
    gradient_step,
    greedy,
    log_prob,
    loss,
    loss_gradient,
    pick,
    softmax,
)
from questions import LETTERS, TEMPLATES, Question, QuestionSet, attempt, make_questions
from tasks import CHAPTER, course_home, short_path

DEFAULT_SEED = 0
DEFAULT_LR = 0.25  # chapter 2's default
DEFAULT_EPISODES = 500
DEFAULT_SHOW = 3
TABLE_AFTER = (50, 200, 500)  # the table is printed after these many tries
PAUSE_S = 0.4
NO_PAUSE_S = 0.0
QUESTION_STREAM = 0  # the same two random streams as chapter 2, so the same
PICK_STREAM = 1  # seed asks the same questions in the same order
LOG_DECIMALS = 4
RUNS_DIR = "runs"

# Three kinds of baseline: what the agent expects to earn before it tries.
NO_BASELINE = "none"  # expects 0, so the advantage is the reward (chapter 2)
AVERAGE = "average"  # the average reward of all earlier tries
PER_QUESTION = "question"  # the average reward of earlier tries at this question
BASELINES = (NO_BASELINE, AVERAGE, PER_QUESTION)
DEFAULT_BASELINE = AVERAGE
BASELINE_NAMES = {
    NO_BASELINE: "no baseline",
    AVERAGE: "running average",
    PER_QUESTION: "per-question average",
}
ALL_QUESTIONS = -1  # the one shared key the running average is kept under

# Chapter 2's rewards, and shifted ones: every answer earns about 1, a right
# one only a little more.
PLAIN_WRONG = 0.0
PLAIN_RIGHT = 1.0
SHIFTED_WRONG = 1.00
SHIFTED_RIGHT = 1.05
LOCKED_AT = 0.9  # a wrong query this likely is almost the only one tried again


@dataclass(frozen=True)
class Settings:
    """The knobs of one run.

    Attributes:
        seed: Seeds the question order and the random picks; the same seed
            gives the same run.
        lr: The learning rate: how big each gradient-descent step is.
        episodes: How many tries to train for.
        baseline: ``"none"``, ``"average"`` or ``"question"``.
        shifted: Reward 1.00 for a wrong answer and 1.05 for a right one,
            instead of 0 and 1.
        show: How many tries to print in full after each table.
        pause_s: Seconds to wait after each printed try, so a viewer can follow.
        remember: Grade each question-and-query pair once and reuse the
            result, instead of a fresh folder for every try. The comparison
            scripts train 30 runs, and a query on the same database always
            gives the same answer.
    """

    seed: int = DEFAULT_SEED
    lr: float = DEFAULT_LR
    episodes: int = DEFAULT_EPISODES
    baseline: str = DEFAULT_BASELINE
    shifted: bool = False
    show: int = DEFAULT_SHOW
    pause_s: float = NO_PAUSE_S
    remember: bool = False


@dataclass(frozen=True)
class Update:
    """One try and the gradient-descent step it caused.

    Attributes:
        number: The try's number, from 1.
        question: The question asked.
        choice: The query picked (0 for A).
        answer: What the query wrote to answer.txt.
        correct: The grader's verdict.
        reward: The reward the verdict earned.
        baseline: What the agent expected to earn.
        advantage: ``reward - baseline``.
        p: The choice's probability before the step.
        log_p: Its log-probability before the step.
        loss: ``-advantage * log_p``.
        before: The question's row of scores before the step.
        after: The row after the step.
    """

    number: int
    question: Question
    choice: int
    answer: str
    correct: bool
    reward: float
    baseline: float
    advantage: float
    p: float
    log_p: float
    loss: float
    before: np.ndarray
    after: np.ndarray


@dataclass(frozen=True)
class Training:
    """What training ends with.

    Attributes:
        scores: The 4 x 4 table of scores (the policy's weights).
        updates: Every try, in order.
    """

    scores: np.ndarray
    updates: tuple[Update, ...]


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        scores: The final table of scores.
        updates: Every try, in order.
        test_correct: How many held-out questions the greedy test got right.
        test_total: How many held-out questions there were.
        log_path: Where the run log was written.
        digest: The run log's md5, the same for the same settings.
    """

    scores: np.ndarray
    updates: tuple[Update, ...]
    test_correct: int
    test_total: int
    log_path: Path
    digest: str


class Baseline:
    """What the agent expects to earn: an average of the rewards so far.

    Attributes:
        kind: ``"none"``, ``"average"`` or ``"question"``.
    """

    def __init__(self, kind: str) -> None:
        """Start with no rewards seen.

        Args:
            kind: ``"none"``, ``"average"`` or ``"question"``.

        Raises:
            ValueError: If the kind is not one of those three.
        """
        if kind not in BASELINES:
            raise ValueError(f"unknown baseline {kind!r}")
        self.kind = kind
        self._totals: defaultdict[int, float] = defaultdict(float)
        self._counts: defaultdict[int, int] = defaultdict(int)

    def _key(self, row: int) -> int:
        """Say which average a question's rewards go into.

        Args:
            row: The question's row.

        Returns:
            The row itself for a per-question baseline, else one shared key.
        """
        return row if self.kind == PER_QUESTION else ALL_QUESTIONS

    def expect(self, row: int, reward: float) -> float:
        """Return what the agent expected to earn on this try.

        Args:
            row: The question's row.
            reward: This try's reward, used only when there is nothing to
                average yet.

        Returns:
            0 with no baseline; otherwise the average reward of the earlier
            tries it covers. Before any earlier try there is nothing to
            compare with, so it returns this try's own reward: an advantage
            of 0, and the first try changes nothing.
        """
        if self.kind == NO_BASELINE:
            return 0.0
        key = self._key(row)
        if self._counts[key] == 0:
            return reward
        return self._totals[key] / self._counts[key]

    def add(self, row: int, reward: float) -> None:
        """Count a try's reward in the average, after its step.

        Args:
            row: The question's row.
            reward: The try's reward.
        """
        key = self._key(row)
        self._totals[key] += reward
        self._counts[key] += 1


@functools.cache
def remembered_attempt(question: Question, choice: int) -> tuple[str, Grade]:
    """Run and grade a question-and-query pair once, then reuse the result.

    Args:
        question: The question being answered.
        choice: Which candidate query to run (0 for A).

    Returns:
        The answer and the grader's verdict, as ``attempt`` returns them.

    Raises:
        OSError: If the work folder cannot be made the first time.
        sqlite3.Error: If the query fails.
    """
    return attempt(question, choice)


def graded(question: Question, choice: int, remember: bool) -> tuple[str, Grade]:
    """Run the picked query and grade it: in a fresh folder, or remembered.

    Args:
        question: The question being answered.
        choice: Which candidate query to run (0 for A).
        remember: Reuse the result of an earlier identical try.

    Returns:
        The answer written to answer.txt, and the grader's verdict.

    Raises:
        OSError: If the work folder cannot be made.
        sqlite3.Error: If the query fails.
    """
    return (
        remembered_attempt(question, choice) if remember else attempt(question, choice)
    )


def reward_for(correct: bool, shifted: bool) -> float:
    """Turn the grader's verdict into a reward.

    Args:
        correct: The grader's verdict.
        shifted: Use 1.00 and 1.05 instead of 0 and 1.

    Returns:
        The reward.
    """
    if shifted:
        return SHIFTED_RIGHT if correct else SHIFTED_WRONG
    return PLAIN_RIGHT if correct else PLAIN_WRONG


def one_try(
    scores: np.ndarray,
    question: Question,
    number: int,
    settings: Settings,
    baseline: Baseline,
    pick_rng: np.random.Generator,
) -> Update:
    """Pick, run, grade, then take one gradient-descent step on the loss.

    Args:
        scores: The table; the question's row is replaced by the new row.
        question: The question asked.
        number: The try's number.
        settings: The run's knobs.
        baseline: The running averages, updated after the step.
        pick_rng: The random generator for picks.

    Returns:
        Everything about the try and its step.

    Raises:
        OSError: If the work folder cannot be made.
        sqlite3.Error: If the query fails.
    """
    row = question.row
    before = scores[row].copy()
    choice = pick(before, pick_rng)
    answer, verdict = graded(question, choice, settings.remember)
    reward = reward_for(verdict.correct, settings.shifted)
    expected = baseline.expect(row, reward)
    advantage = reward - expected
    gradient = loss_gradient(before, choice, advantage)
    scores[row] = gradient_step(before, gradient, settings.lr)
    baseline.add(row, reward)
    return Update(
        number=number,
        question=question,
        choice=choice,
        answer=answer,
        correct=verdict.correct,
        reward=reward,
        baseline=expected,
        advantage=advantage,
        p=float(softmax(before)[choice]),
        log_p=log_prob(before, choice),
        loss=loss(before, choice, advantage),
        before=before,
        after=scores[row].copy(),
    )


def table_after(episodes: int) -> list[int]:
    """List the try counts after which the probability table is printed.

    Args:
        episodes: How many tries the run has.

    Returns:
        0, then every count in ``TABLE_AFTER`` below ``episodes``, then
        ``episodes`` itself.
    """
    return [0, *(count for count in TABLE_AFTER if count < episodes), episodes]


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
    """Describe one try for the run log.

    Args:
        update: The try.

    Returns:
        One JSON-ready record.
    """
    return {
        "try": update.number,
        "question": update.question.task.question,
        "chose": LETTERS[update.choice],
        "answer": update.answer,
        "reward": update.reward,
        "baseline": rounded(update.baseline),
        "advantage": rounded(update.advantage),
        "p": rounded(update.p),
        "log_p": rounded(update.log_p),
        "loss": rounded(update.loss),
        "new_p": rounded(softmax(update.after)),
    }


def train(
    settings: Settings,
    questions: QuestionSet,
    log: list[dict[str, Any]],
    printing: bool = True,
) -> Training:
    """Run the loop: ask, pick, run, grade, step down the loss, again and again.

    Args:
        settings: The run's knobs.
        questions: The fair questions; only the training ones are asked.
        log: A list every try and table is appended to, for the run log.
        printing: Print the tables and the first tries of each stretch.

    Returns:
        The final table of scores and every try.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    scores = np.zeros((len(TEMPLATES), len(LETTERS)))  # all equal: 25% each
    tries = np.zeros(len(TEMPLATES), dtype=int)
    baseline = Baseline(settings.baseline)
    question_rng = np.random.default_rng([settings.seed, QUESTION_STREAM])
    pick_rng = np.random.default_rng([settings.seed, PICK_STREAM])
    updates: list[Update] = []
    if printing:
        report.print_table(chances(scores), tries, 0)
    for start, end in pairwise(table_after(settings.episodes)):
        if printing:
            print()
        for number in range(start + 1, end + 1):
            asked = question_rng.integers(len(questions.training))
            question = questions.training[asked]
            update = one_try(scores, question, number, settings, baseline, pick_rng)
            tries[question.row] += 1
            updates.append(update)
            log.append(record(update))
            if printing and number - start <= settings.show:
                report.print_update(update, settings.shifted)
                time.sleep(settings.pause_s)
        log.append({"after": end, "chances": rounded(chances(scores))})
        if printing:
            correct = sum(u.correct for u in updates[start:end])
            report.print_stretch(start + 1, end, correct)
            report.print_table(chances(scores), tries, end)
            time.sleep(settings.pause_s)
    return Training(scores, tuple(updates))


def locked_rows(scores: np.ndarray) -> list[int]:
    """List the rows where a wrong query has taken over.

    Args:
        scores: The table of scores.

    Returns:
        Every row whose top query is wrong and holds at least ``LOCKED_AT``
        of the chance.
    """
    table = chances(scores)
    return [
        row
        for row, template in enumerate(TEMPLATES)
        if greedy(scores[row]) != template.right and table[row].max() >= LOCKED_AT
    ]


@dataclass(frozen=True)
class Outcome:
    """How a quiet run went, for comparing runs side by side.

    Attributes:
        pushes: For every try, how far its step moved the right query's score
            (positive means toward the right query).
        pushed_up: How many tries pushed the query they tried up (advantage
            above 0).
        wrong_pushed_up: How many of those tried a wrong query.
        right_chances: Each question's chance of its right query at the end.
        locked: How many questions a wrong query has taken over.
        test_correct: The greedy test's score on the held-out cities.
    """

    pushes: tuple[float, ...]
    pushed_up: int
    wrong_pushed_up: int
    right_chances: tuple[float, ...]
    locked: int
    test_correct: int


def train_quietly(settings: Settings, questions: QuestionSet) -> Outcome:
    """Train and test without printing, and measure how it went.

    Args:
        settings: The run's knobs.
        questions: The fair questions.

    Returns:
        The run's measurements.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    log: list[dict[str, Any]] = []
    settings = replace(settings, remember=True)
    training = train(settings, questions, log, printing=False)
    test_correct = greedy_test(
        training.scores, questions.held_out, log, printing=False, remember=True
    )
    pushes = []
    pushed_up = wrong_pushed_up = 0
    for update in training.updates:
        right = TEMPLATES[update.question.row].right
        pushes.append(float(update.after[right] - update.before[right]))
        if update.advantage > 0:
            pushed_up += 1
            if update.choice != right:
                wrong_pushed_up += 1
    final = chances(training.scores)
    return Outcome(
        pushes=tuple(pushes),
        pushed_up=pushed_up,
        wrong_pushed_up=wrong_pushed_up,
        right_chances=tuple(
            float(final[row][template.right]) for row, template in enumerate(TEMPLATES)
        ),
        locked=len(locked_rows(training.scores)),
        test_correct=test_correct,
    )


def greedy_test(
    scores: np.ndarray,
    held_out: tuple[Question, ...],
    log: list[dict[str, Any]],
    printing: bool = True,
    remember: bool = False,
) -> int:
    """Ask every held-out question greedily and count the right answers.

    Args:
        scores: The trained table of scores.
        held_out: Questions about cities never used in training.
        log: The run log, which gets one record per question.
        printing: Print one line per question and the total.
        remember: Reuse the result of an earlier identical try.

    Returns:
        How many held-out questions were answered correctly.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    if printing:
        print("\ngreedy test (always the top score) on cities never used in training")
    correct = 0
    for row in range(len(TEMPLATES)):
        choice = greedy(scores[row])
        asked = [question for question in held_out if question.row == row]
        right_here = 0
        for question in asked:
            answer, verdict = graded(question, choice, remember)
            right_here += verdict.reward
            log.append(
                {
                    "test": question.task.question,
                    "chose": LETTERS[choice],
                    "answer": answer,
                    "correct": verdict.correct,
                }
            )
        if printing:
            report.print_test_row(row, choice, right_here, len(asked))
        correct += right_here
    if printing:
        print(f"held-out score: {correct} of {len(held_out)}")
    return correct


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


def rewards_name(shifted: bool) -> str:
    """Name the reward scheme, for titles and file names.

    Args:
        shifted: Whether the rewards are shifted.

    Returns:
        ``"shifted"`` or ``"0-1"``.
    """
    return "shifted" if shifted else "0-1"


def run(settings: Settings) -> RunResult:
    """Train, print the tables, test on held-out cities and write the log.

    Args:
        settings: The run's knobs.

    Returns:
        The final scores, every try, the test score and the log's md5.

    Raises:
        OSError: If a work folder or the run log cannot be written.
        sqlite3.Error: If a query fails.
    """
    questions = make_questions()
    rewards = (
        f"rewards {SHIFTED_WRONG:.2f} and {SHIFTED_RIGHT:.2f}"
        if settings.shifted
        else "rewards 0 and 1"
    )
    title = (
        f"policy gradient · seed {settings.seed} · lr {settings.lr:g}"
        f" · {BASELINE_NAMES[settings.baseline]} · {rewards}"
        f" · {settings.episodes} tries"
    )
    report.print_header(title, questions)
    log: list[dict[str, Any]] = []
    training = train(settings, questions, log)
    report.print_weights(training.scores)
    locked = locked_rows(training.scores)
    if locked:
        report.print_locked(locked, training.scores)
    test_correct = greedy_test(training.scores, questions.held_out, log)
    log.append({"weights": rounded(training.scores)})
    name = (
        f"seed{settings.seed}-lr{settings.lr:g}-baseline-{settings.baseline}"
        f"-rewards-{rewards_name(settings.shifted)}-tries{settings.episodes}.jsonl"
    )
    path = course_home() / CHAPTER / RUNS_DIR / name
    digest = write_jsonl(path, log)
    print(f"\nwrote {short_path(path)}")
    print(f"md5   {digest}")
    return RunResult(
        training.scores,
        training.updates,
        test_correct,
        len(questions.held_out),
        path,
        digest,
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


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The run's knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINES,
        default=DEFAULT_BASELINE,
        help="what the reward is compared with (default: the running average)",
    )
    parser.add_argument(
        "--shifted",
        action="store_true",
        help="reward 1.00 for a wrong answer and 1.05 for a right one",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="tries to train for"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="same seed, same run"
    )
    parser.add_argument(
        "--show", type=int, default=DEFAULT_SHOW, help="tries to print per stretch"
    )
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    if args.episodes < 1 or args.show < 0:
        parser.error("episodes must be 1 or more, and show 0 or more")
    return args


def main() -> None:
    """Parse the flags, then train and test."""
    args = parse_args()
    settings = Settings(
        seed=args.seed,
        lr=args.lr,
        episodes=args.episodes,
        baseline=args.baseline,
        shifted=args.shifted,
        show=args.show,
        pause_s=NO_PAUSE_S if args.fast else PAUSE_S,
    )
    run(settings)


if __name__ == "__main__":
    main()
