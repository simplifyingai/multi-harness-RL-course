"""How this chapter prints: tries in full, tables, the test, and the comparisons.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so the scripts can stay about the math. The table, weights and
test printers are chapter 2's, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from policy import softmax
from questions import LETTERS, TEMPLATES, QuestionSet, snippet

if TYPE_CHECKING:  # only for type hints: reinforce.py imports this module
    from reinforce import Update

PERCENT = 100
WEIGHT_DECIMALS = 2
COLUMN_WIDTH = 6
WEIGHT_WIDTH = 8
RIGHT_MARK = "*"
INDENT = " " * 8
QUESTION_WIDTH = max(len(template.question) for template in TEMPLATES) + 2
SEPARATOR = " · "
LOSS_LINE = (
    "loss for each try: -advantage x log p(chosen), advantage = reward - baseline"
)


def print_header(title: str, questions: QuestionSet) -> None:
    """Print the run's settings, the loss, and which cities it uses.

    Args:
        title: The settings, on one line.
        questions: The fair questions, split by city.
    """
    print(title)
    print(LOSS_LINE)
    print(f"training cities: {', '.join(questions.train_cities)}")
    print(f"held out for the test: {', '.join(questions.test_cities)}")
    print(f"{len(questions.left_out)} unfair questions left out, as in chapter 2")


def print_table(chances: np.ndarray, tries: np.ndarray, done: int) -> None:
    """Print each query's chance for each question, in percent.

    Args:
        chances: One row of probabilities per question.
        tries: How many tries each question has had.
        done: How many tries have run so far.
    """
    print(f"\nafter {done} tries: each query's chance (%), {RIGHT_MARK} = right")
    header = "".join(f"{letter:>{COLUMN_WIDTH}}" for letter in LETTERS)
    print(f"{'':<{QUESTION_WIDTH}}{header}{'tries':>{COLUMN_WIDTH + 1}}")
    for row, template in enumerate(TEMPLATES):
        cells = ""
        for column, chance in enumerate(chances[row]):
            mark = RIGHT_MARK if column == template.right else ""
            cells += f"{mark + str(round(PERCENT * chance)):>{COLUMN_WIDTH}}"
        count = f"{tries[row]:>{COLUMN_WIDTH + 1}}"
        print(f"{template.question:<{QUESTION_WIDTH}}{cells}{count}")


def reward_text(reward: float, shifted: bool) -> str:
    """Write a reward: 1 or 0, or with two decimals when shifted.

    Args:
        reward: The reward.
        shifted: Whether the rewards are 1.00 and 1.05.

    Returns:
        The reward as text.
    """
    return f"{reward:.2f}" if shifted else f"{reward:g}"


def probabilities_text(scores: np.ndarray) -> str:
    """Write a row's probabilities with their letters.

    Args:
        scores: One row of scores.

    Returns:
        For example ``"A 0.250 · B 0.250 · C 0.250 · D 0.250"``.
    """
    return SEPARATOR.join(
        f"{letter} {chance:.3f}"
        for letter, chance in zip(LETTERS, softmax(scores), strict=True)
    )


def tidy(value: float, decimals: int) -> float:
    """Round a number for printing, so a tiny negative never shows as -0.000.

    Args:
        value: The number.
        decimals: How many decimals will be shown.

    Returns:
        The rounded number; adding 0.0 turns -0.0 into 0.0.
    """
    return round(value, decimals) + 0.0


def print_update(update: Update, shifted: bool) -> None:
    """Print one try in full: the pick, the numbers of the loss, and the step.

    Args:
        update: The try and its step.
        shifted: Whether the rewards are 1.00 and 1.05.
    """
    question = update.question
    verdict = "correct" if update.correct else "wrong"
    shown = snippet(TEMPLATES[question.row], update.choice, question.city)
    print(f"try {update.number} · {question.task.question}")
    print(
        f"{INDENT}chose {LETTERS[update.choice]}: {shown}"
        f" → {update.answer or '(nothing)'} · {verdict}"
    )
    print(
        f"{INDENT}p = {update.p:.2f} · log p = {tidy(update.log_p, 3):.3f}"
        f" · reward {reward_text(update.reward, shifted)}"
        f" · baseline {update.baseline:.2f}"
        f" · advantage {tidy(update.advantage, 2):+.2f}"
    )
    print(
        f"{INDENT}loss {tidy(update.loss, 3):+.3f} → new probabilities "
        f"{probabilities_text(update.after)}"
    )


def print_stretch(start: int, end: int, correct: int) -> None:
    """Print how many tries in a stretch were answered correctly.

    Args:
        start: The stretch's first try.
        end: Its last try.
        correct: How many of its tries the grader called correct.
    """
    share = round(PERCENT * correct / (end - start + 1))
    print(f"tries {start}-{end}: {correct} correct ({share}%)")


def print_weights(scores: np.ndarray) -> None:
    """Print the scores themselves: the policy's weights.

    Args:
        scores: The table of scores.
    """
    print(f"\nthe {scores.size} weights: the scores softmax turns into those chances")
    header = "".join(f"{letter:>{WEIGHT_WIDTH}}" for letter in LETTERS)
    print(f"{'':<{QUESTION_WIDTH}}{header}")
    for row, template in enumerate(TEMPLATES):
        # Adding 0.0 turns a rounded -0.00 into 0.00.
        values = [round(float(w), WEIGHT_DECIMALS) + 0.0 for w in scores[row]]
        cells = "".join(f"{v:>+{WEIGHT_WIDTH}.{WEIGHT_DECIMALS}f}" for v in values)
        print(f"{template.question:<{QUESTION_WIDTH}}{cells}")


def print_locked(rows: list[int], scores: np.ndarray) -> None:
    """Name the questions where a wrong query has taken over.

    Args:
        rows: The locked rows.
        scores: The table of scores.
    """
    print("\nlocked onto a wrong query (it is now almost the only one tried):")
    for row in rows:
        chance = softmax(scores[row])
        top = int(np.argmax(chance))
        print(
            f"{TEMPLATES[row].question:<{QUESTION_WIDTH}}{LETTERS[top]} at "
            f"{round(PERCENT * chance[top])}%, right query "
            f"{LETTERS[TEMPLATES[row].right]} at "
            f"{round(PERCENT * chance[TEMPLATES[row].right])}%"
        )


def print_test_row(row: int, choice: int, correct: int, asked: int) -> None:
    """Print the greedy test's result for one question.

    Args:
        row: Which question (0 to 3).
        choice: The query greedy picked for it.
        correct: How many held-out cities it answered correctly.
        asked: How many held-out cities it was asked about.
    """
    question = TEMPLATES[row].question
    print(
        f"{question:<{QUESTION_WIDTH}}picked {LETTERS[choice]} · "
        f"{correct} of {asked} correct"
    )
