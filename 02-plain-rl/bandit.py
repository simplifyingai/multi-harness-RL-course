# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Plain RL: an agent learns which of four queries to trust, from rewards alone.

For each of four questions the agent keeps four scores, one per candidate
query. Softmax turns a row of scores into probabilities. The agent picks a
query at random by those probabilities (the temperature flattens or sharpens
them first), the query runs on a fresh copy of shop.db, and chapter 1's grader
turns the answer into a reward: 1 or 0. Then the row gets a nudge,
``row += lr * reward * (onehot(choice) - probabilities)``, so whatever earned
a reward becomes more likely. Chapter 3 shows where that nudge comes from.

Run it with ``uv run bandit.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import report
from questions import LETTERS, TEMPLATES, Question, QuestionSet, attempt, make_questions
from tasks import CHAPTER, course_home, short_path

DEFAULT_SEED = 0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LR = 0.25
DEFAULT_EPISODES = 500
DEFAULT_SHOW = 3
TABLE_AFTER = (50, 200, 500)  # the table is printed after these many tries
GREEDY = 0.0  # temperature 0: no chance involved, always the top score
PAUSE_S = 0.4
NO_PAUSE_S = 0.0
QUESTION_STREAM = 0
PICK_STREAM = 1
LOG_DECIMALS = 4
RUNS_DIR = "runs"


@dataclass(frozen=True)
class Settings:
    """The knobs of one run.

    Attributes:
        seed: Seeds the question order and the random picks; the same seed
            gives the same run.
        temperature: 1 picks by the plain probabilities, above 1 flattens
            them (more exploring), below 1 sharpens them, 0 always takes the
            top score.
        lr: The learning rate: how big each nudge is.
        episodes: How many tries to train for.
        show: How many tries to print after each table.
        pause_s: Seconds to wait after each printed try, so a viewer can follow.
    """

    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    lr: float = DEFAULT_LR
    episodes: int = DEFAULT_EPISODES
    show: int = DEFAULT_SHOW
    pause_s: float = NO_PAUSE_S


@dataclass(frozen=True)
class RunResult:
    """What one run ends with.

    Attributes:
        scores: The 4 x 4 table of scores (the policy's weights).
        rewards: How many training tries earned reward 1.
        test_correct: How many held-out questions the greedy test got right.
        test_total: How many held-out questions there were.
        log_path: Where the run log was written.
        digest: The run log's md5, the same for the same settings.
    """

    scores: np.ndarray
    rewards: int
    test_correct: int
    test_total: int
    log_path: Path
    digest: str


def softmax(scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Turn a row of scores into probabilities that add up to 1.

    Args:
        scores: Any real numbers.
        temperature: Divides the scores first: above 1 makes the
            probabilities more even, below 1 makes the top score dominate.

    Returns:
        One probability per score.
    """
    # Subtracting the largest score changes no probability, but keeps exp()
    # from overflowing when scores grow large or the temperature is tiny.
    scaled = (scores - scores.max()) / temperature
    exps = np.exp(scaled)
    return exps / exps.sum()


def greedy(scores: np.ndarray) -> int:
    """Take the top score. When scores tie, take the first (A before B).

    Args:
        scores: One row of the table.

    Returns:
        The index of the top score.
    """
    return int(np.argmax(scores))  # argmax returns the first of equal values


def pick(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Choose a query: greedily at temperature 0, otherwise at random.

    Args:
        scores: One row of the table.
        temperature: 0 for greedy; otherwise how much to flatten the odds.
        rng: The random generator for picks.

    Returns:
        The chosen index (0 for A).
    """
    if temperature == GREEDY:  # exactly 0, as typed: no chance involved
        return greedy(scores)
    return int(rng.choice(len(scores), p=softmax(scores, temperature)))


def nudge(scores: np.ndarray, choice: int, reward: int, lr: float) -> np.ndarray:
    """Move a row toward the choice if it earned a reward.

    ``onehot - probabilities`` is positive for the choice and negative for
    the others, so a reward raises the choice and lowers the rest. A reward
    of 0 changes nothing. The probabilities are the scores' own (temperature
    1): the temperature only changes how a query is picked.

    Args:
        scores: The row of the question that was asked.
        choice: The query that was picked.
        reward: The grader's reward, 1 or 0.
        lr: The learning rate.

    Returns:
        The new row.
    """
    onehot = np.zeros_like(scores)
    onehot[choice] = 1.0
    return scores + lr * reward * (onehot - softmax(scores))


def chances(scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Apply softmax to every row of the table.

    Args:
        scores: The table of scores.
        temperature: As for ``softmax``.

    Returns:
        One row of probabilities per question.
    """
    return np.array([softmax(row, temperature) for row in scores])


def table_after(episodes: int) -> list[int]:
    """List the try counts after which the probability table is printed.

    Args:
        episodes: How many tries the run has.

    Returns:
        0, then every count in ``TABLE_AFTER`` below ``episodes``, then
        ``episodes`` itself.
    """
    return [0, *(count for count in TABLE_AFTER if count < episodes), episodes]


def record(question: Question, choice: int, answer: str, reward: int) -> dict[str, Any]:
    """Describe one answered question for the run log.

    Args:
        question: The question asked.
        choice: The query picked.
        answer: What the query wrote to answer.txt.
        reward: The grader's reward.

    Returns:
        One JSON-ready record.
    """
    return {
        "question": question.task.question,
        "picked": LETTERS[choice],
        "answer": answer,
        "reward": reward,
    }


def rounded(table: np.ndarray) -> list[list[float]]:
    """Round a table for the run log, so the md5 does not hang on tiny digits.

    Args:
        table: Rows of numbers.

    Returns:
        The table as nested lists of plain floats.
    """
    return [[round(float(value), LOG_DECIMALS) for value in row] for row in table]


def train(
    settings: Settings, questions: QuestionSet, log: list[dict[str, Any]]
) -> tuple[np.ndarray, int]:
    """Run the RL loop: ask, pick, run, grade, nudge, again and again.

    Args:
        settings: The run's knobs.
        questions: The fair questions; only the training ones are asked.
        log: A list every try and table is appended to, for the run log.

    Returns:
        The final table of scores and how many tries earned a reward.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    scores = np.zeros((len(TEMPLATES), len(LETTERS)))  # all equal: 25% each
    tries = np.zeros(len(TEMPLATES), dtype=int)
    # Two random streams: changing the temperature never changes which
    # questions come up, only which queries get picked.
    question_rng = np.random.default_rng([settings.seed, QUESTION_STREAM])
    pick_rng = np.random.default_rng([settings.seed, PICK_STREAM])
    checkpoints = table_after(settings.episodes)
    report.print_table(chances(scores), tries, 0)
    total = 0
    for start, end in pairwise(checkpoints):
        print()
        rewarded = 0
        for number in range(start + 1, end + 1):
            asked = question_rng.integers(len(questions.training))
            question = questions.training[asked]
            row = question.row
            choice = pick(scores[row], settings.temperature, pick_rng)
            answer, verdict = attempt(question, choice)
            reward = verdict.reward  # 1 if the grader says correct, else 0
            scores[row] = nudge(scores[row], choice, reward, settings.lr)
            tries[row] += 1
            rewarded += reward
            log.append({"try": number, **record(question, choice, answer, reward)})
            if number - start <= settings.show:
                report.print_try(number, question, choice, answer, reward)
                time.sleep(settings.pause_s)
        report.print_stretch(start + 1, end, rewarded)
        report.print_table(chances(scores), tries, end)
        log.append({"after": end, "chances": rounded(chances(scores))})
        total += rewarded
        time.sleep(settings.pause_s)
    return scores, total


def greedy_test(
    scores: np.ndarray, held_out: tuple[Question, ...], log: list[dict[str, Any]]
) -> int:
    """Ask every held-out question at temperature 0 and count the right answers.

    Args:
        scores: The trained table of scores.
        held_out: Questions about cities never used in training.
        log: The run log, which gets one record per question.

    Returns:
        How many held-out questions were answered correctly.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    print("\ngreedy test (temperature 0) on cities never used in training")
    correct = 0
    for row in range(len(TEMPLATES)):
        choice = greedy(scores[row])
        asked = [question for question in held_out if question.row == row]
        right_here = 0
        for question in asked:
            answer, verdict = attempt(question, choice)
            right_here += verdict.reward
            entry = record(question, choice, answer, verdict.reward)
            log.append({"test": True, **entry})
        report.print_test_row(row, choice, right_here, len(asked))
        correct += right_here
    print(f"held-out score: {correct} of {len(held_out)}")
    return correct


def write_log(settings: Settings, log: list[dict[str, Any]]) -> tuple[Path, str]:
    """Write the run log as JSON lines and return its path and md5.

    Args:
        settings: The run's knobs, which name the file.
        log: Every try, table and test answer.

    Returns:
        The file's path and the md5 of its text.

    Raises:
        OSError: If the file cannot be written.
    """
    name = (
        f"seed{settings.seed}-temperature{settings.temperature}"
        f"-lr{settings.lr}-tries{settings.episodes}.jsonl"
    )
    path = course_home() / CHAPTER / RUNS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in log)
    path.write_text(text, encoding="utf-8")
    # md5 here is a fingerprint for comparing runs, not a security measure.
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return path, digest


def run(settings: Settings) -> RunResult:
    """Train, print the tables, test on held-out cities and write the log.

    Args:
        settings: The run's knobs.

    Returns:
        The final scores, the reward count, the test score and the log's md5.

    Raises:
        OSError: If a work folder or the run log cannot be written.
        sqlite3.Error: If a query fails.
    """
    questions = make_questions()
    title = (
        f"plain RL · seed {settings.seed} · temperature {settings.temperature:g}"
        f" · lr {settings.lr:g} · {settings.episodes} tries"
    )
    report.print_header(title, questions, settings.temperature == GREEDY)
    log: list[dict[str, Any]] = []
    scores, rewards = train(settings, questions, log)
    if settings.temperature not in (GREEDY, DEFAULT_TEMPERATURE):
        picking = chances(scores, settings.temperature)
        right = [picking[row][t.right] for row, t in enumerate(TEMPLATES)]
        report.print_temperature(right, settings.temperature)
    report.print_weights(scores)
    test_correct = greedy_test(scores, questions.held_out, log)
    log.append({"weights": rounded(scores)})
    path, digest = write_log(settings, log)
    print(f"\nwrote {short_path(path)}")
    print(f"md5   {digest}")
    return RunResult(
        scores, rewards, test_correct, len(questions.held_out), path, digest
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


def main() -> None:
    """Parse the flags, then train, list the queries or print the words box."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--temperature",
        type=non_negative,
        default=DEFAULT_TEMPERATURE,
        help="1 = the plain odds, above 1 = flatter, 0 = always the top score",
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each nudge is"
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
    only = parser.add_mutually_exclusive_group()
    only.add_argument(
        "--queries", action="store_true", help="list the questions and queries"
    )
    only.add_argument(
        "--words", action="store_true", help="map the RL words onto this code"
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.show < 0:
        parser.error("episodes must be 1 or more, and show 0 or more")
    if args.queries:
        report.print_queries(make_questions())
    elif args.words:
        report.print_words()
    else:
        pause_s = NO_PAUSE_S if args.fast else PAUSE_S
        settings = Settings(
            args.seed, args.temperature, args.lr, args.episodes, args.show, pause_s
        )
        run(settings)


if __name__ == "__main__":
    main()
