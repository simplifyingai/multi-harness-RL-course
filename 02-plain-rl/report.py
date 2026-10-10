"""How bandit.py prints what happens: tries, tables, the test and the words box.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``bandit.py`` can stay about the RL loop itself.
"""

from __future__ import annotations

import re

import numpy as np
from questions import LETTERS, TEMPLATES, Question, QuestionSet, candidate_sql, snippet

PERCENT = 100
WEIGHT_DECIMALS = 2
COLUMN_WIDTH = 6
WEIGHT_WIDTH = 8
RIGHT_MARK = "*"
INDENT = " " * 8
QUERY_INDENT = " " * 5
BOX_WIDTH = 72
BOX_CORNER = "+"
WORD_WIDTH = 9  # the longest word, "weights", plus room
MODEL = "LFM2.5-2.6B"
MODEL_WEIGHTS = "about 2.6 billion"
QUESTION_WIDTH = max(len(template.question) for template in TEMPLATES) + 2
SQL_BREAK_RE = re.compile(r"\s+(?=(FROM|JOIN|WHERE)\b)")
WORDS = (
    ("state", "the question: which row of the table (the city does not count)"),
    ("action", "the query it picks: A, B, C or D"),
    ("reward", "the grader's verdict: 1 for a right answer, 0 for a wrong one"),
    ("policy", "the score table: softmax of a row gives each query's chance"),
    (
        "weights",
        f"the table's {len(TEMPLATES) * len(LETTERS)} numbers; "
        f"{MODEL} has {MODEL_WEIGHTS}",
    ),
)


def print_header(title: str, questions: QuestionSet, greedy: bool) -> None:
    """Print the run's settings and which cities it trains and tests on.

    Args:
        title: The settings, on one line.
        questions: The fair questions, split by city.
        greedy: Whether the run picks at temperature 0.
    """
    print(title)
    print(f"training cities: {', '.join(questions.train_cities)}")
    print(f"held out for the test: {', '.join(questions.test_cities)}")
    print(
        f"{len(questions.left_out)} unfair questions left out "
        "(uv run bandit.py --queries lists them)"
    )
    if greedy:
        print("temperature 0: no chance involved; it always takes the top score,")
        print("and when the scores tie it takes the first, A")


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


def print_try(
    number: int, question: Question, choice: int, answer: str, reward: int
) -> None:
    """Print one try on two lines: the question, then the pick and the verdict.

    Args:
        number: The try's number.
        question: The question asked.
        choice: The query picked.
        answer: What the query wrote to answer.txt.
        reward: The grader's reward.
    """
    verdict = "correct" if reward else "wrong"
    shown = snippet(TEMPLATES[question.row], choice, question.city)
    print(f"try {number} · {question.task.question}")
    print(
        f"{INDENT}picked {LETTERS[choice]}: {shown} → {answer or '(nothing)'}"
        f" · {verdict} · reward {reward}"
    )


def print_stretch(start: int, end: int, rewarded: int) -> None:
    """Print how many tries in a stretch earned a reward.

    Args:
        start: The stretch's first try.
        end: Its last try.
        rewarded: How many of its tries earned reward 1.
    """
    share = round(PERCENT * rewarded / (end - start + 1))
    print(f"tries {start}-{end}: {rewarded} rewarded ({share}%)")


def print_temperature(right_chances: list[float], temperature: float) -> None:
    """Say how often a temperature other than 1 actually picks the right query.

    Args:
        right_chances: For each question, the chance of picking its right
            query at this temperature.
        temperature: The run's temperature.
    """
    low, high = round(PERCENT * min(right_chances)), round(PERCENT * max(right_chances))
    print(
        f"but at temperature {temperature:g} it picks the right query "
        f"only {low}% to {high}% of the time"
    )


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


def pretty_sql(sql: str) -> str:
    """Break a query before each clause so it reads in a narrow terminal.

    Args:
        sql: One line of SQL.

    Returns:
        The query over several indented lines.
    """
    return SQL_BREAK_RE.sub("\n" + QUERY_INDENT + "  ", sql)


def print_queries(questions: QuestionSet) -> None:
    """Print every question's four candidate queries and which questions are used.

    Args:
        questions: The fair questions, split by city, and the left-out ones.
    """
    for template in TEMPLATES:
        print(f"\n{template.question}")
        for choice, candidate in enumerate(template.candidates):
            sql = pretty_sql(candidate_sql(template, choice, "{city}"))
            print(f"  {LETTERS[choice]}  {sql}")
            print(f"{QUERY_INDENT}-- {candidate.note}")
    print(f"\ntraining cities: {', '.join(questions.train_cities)}")
    print(f"held out for the test: {', '.join(questions.test_cities)}")
    print(
        f"{len(questions.training)} training questions, "
        f"{len(questions.held_out)} held-out questions"
    )
    print("left out, because a wrong query gets the right answer too, or none does:")
    for question, reason in questions.left_out:
        print(f"  {question} · {reason}")


def print_words() -> None:
    """Print the box that maps reinforcement-learning words onto bandit.py."""
    title = f"{BOX_CORNER}-- RL words, mapped onto bandit.py "
    print(title + "-" * (BOX_WIDTH - len(title)))
    for word, meaning in WORDS:
        print(f"| {word:<{WORD_WIDTH}}{meaning}")
    print(BOX_CORNER + "-" * (BOX_WIDTH - len(BOX_CORNER)))
