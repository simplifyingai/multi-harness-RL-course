"""The four questions, the four queries the agent may pick for each, and one try.

Each question has a ``{city}`` slot and four candidate SQL queries: one right
and three plausible mistakes. The agent in ``bandit.py`` sees only the letters
A to D. It never reads the SQL or which letter is right: the grader decides
that, by comparing what the picked query returned with the gold answer, which
is what the right query returns for that city.

A filled-in question is kept only when it is fair: the right query returns an
answer, and no mistake happens to return the same one (chapter 1's task
generator throws such drafts away too). Every third city in alphabetical order
is held out, so the final test asks about cities the agent never trained on.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grader import Grade, check_answer, grade
from tasks import (
    ANSWER_FILE,
    DATA_DIR,
    DATABASE_FILE,
    NUMBER,
    TEXT,
    Task,
    work_root,
)

SHOP_DB = DATA_DIR / "shop.db"
CITIES_SQL = "SELECT DISTINCT city FROM customers ORDER BY city"
LETTERS = "ABCD"
HELD_OUT_EVERY = 3  # every third city, in alphabetical order, is kept for the test
DOMAIN = "shop"
DIFFICULTY = "easy"
TRAIN = "train"
TEST = "test"
DECIMALS = 2
READ_ONLY = "?mode=ro"
TRY_PREFIX = "try-"
SNIPPET_WIDTH = 60  # a longer query is shown by the first words that set it apart
ELLIPSIS = " ..."
NO_ANSWER = "no answer"

# Two joins several queries share: one city's order lines, and its orders.
ITEMS_OF_CITY = (
    "FROM order_items JOIN orders ON orders.id = order_items.order_id "
    "JOIN customers ON customers.id = orders.customer_id "
    "WHERE customers.city = '{city}'"
)
ORDERS_OF_CITY = (
    "FROM orders JOIN customers ON customers.id = orders.customer_id "
    "WHERE customers.city = '{city}'"
)


@dataclass(frozen=True)
class Candidate:
    """One query the agent may pick.

    Attributes:
        sql: The query, with a ``{city}`` slot.
        note: What it really computes, in plain words. Only people read this.
    """

    sql: str
    note: str


@dataclass(frozen=True)
class Template:
    """A question with a ``{city}`` slot and its four candidate queries.

    Attributes:
        key: A short name, used in task ids.
        question: The question, e.g. ``"How many customers live in {city}?"``.
        answer_type: ``"number"`` or ``"text"``, so the grader knows how to compare.
        candidates: Queries A to D.
        right: Which candidate is right (0 means A). It makes the gold answer;
            the agent never reads it.
    """

    key: str
    question: str
    answer_type: str
    candidates: tuple[Candidate, ...]
    right: int


# The right query sits in a different place in every row (B, D, A, C), so no
# letter is a safe bet.
TEMPLATES = (
    Template(
        "customers",
        "How many customers live in {city}?",
        NUMBER,
        (
            Candidate(
                "SELECT COUNT(*) FROM customers",
                "no WHERE filter: counts every customer",
            ),
            Candidate(
                "SELECT COUNT(*) FROM customers WHERE city = '{city}'",
                "right: counts the customers in the city",
            ),
            Candidate(
                "SELECT SUM(quantity) " + ITEMS_OF_CITY,
                "SUM instead of COUNT: adds up the items they bought",
            ),
            Candidate(
                "SELECT COUNT(*) " + ORDERS_OF_CITY,
                "the JOIN counts their orders, not the customers",
            ),
        ),
        right=1,
    ),
    Template(
        "orders",
        "How many orders did customers from {city} place?",
        NUMBER,
        (
            Candidate(
                "SELECT COUNT(*) FROM orders", "no WHERE filter: counts every order"
            ),
            Candidate(
                "SELECT SUM(quantity) " + ITEMS_OF_CITY,
                "SUM instead of COUNT: adds up the items in those orders",
            ),
            Candidate(
                "SELECT COUNT(*) " + ITEMS_OF_CITY,
                "the JOIN counts order lines, not orders",
            ),
            Candidate(
                "SELECT COUNT(*) " + ORDERS_OF_CITY, "right: counts their orders"
            ),
        ),
        right=3,
    ),
    Template(
        "spend",
        "How much did customers from {city} spend?",
        NUMBER,
        (
            Candidate(
                "SELECT SUM(quantity * unit_price) " + ITEMS_OF_CITY,
                "right: adds up quantity times price",
            ),
            Candidate(
                "SELECT SUM(quantity * unit_price) FROM order_items",
                "no WHERE filter: what every customer spent",
            ),
            Candidate(
                "SELECT SUM(unit_price) " + ITEMS_OF_CITY,
                "forgets the quantity: adds up prices only",
            ),
            Candidate(
                "SELECT COUNT(*) " + ITEMS_OF_CITY,
                "COUNT instead of SUM: counts order lines",
            ),
        ),
        right=0,
    ),
    Template(
        "first-order",
        "When was the first order from {city}?",
        TEXT,
        (
            Candidate(
                "SELECT MIN(signup_date) FROM customers WHERE city = '{city}'",
                "the wrong date: the first sign-up, not the first order",
            ),
            Candidate(
                "SELECT MAX(order_date) " + ORDERS_OF_CITY,
                "MAX instead of MIN: the last order",
            ),
            Candidate(
                "SELECT MIN(order_date) " + ORDERS_OF_CITY, "right: the first order"
            ),
            Candidate(
                "SELECT MIN(order_date) FROM orders",
                "no WHERE filter: the first order from anywhere",
            ),
        ),
        right=2,
    ),
)


@dataclass(frozen=True)
class Question:
    """One template filled with one city.

    Attributes:
        row: Which template (0 to 3), and so which row of the score table.
        city: The city in the slot.
        task: The question text, gold SQL and gold answer, for the grader.
    """

    row: int
    city: str
    task: Task


@dataclass(frozen=True)
class QuestionSet:
    """Every fair question, split by city, and what was left out and why.

    Attributes:
        train_cities: Cities the agent trains on.
        test_cities: Cities kept for the final test.
        training: Fair questions about the training cities.
        held_out: Fair questions about the test cities.
        left_out: ``(question, reason)`` for every unfair question.
    """

    train_cities: tuple[str, ...]
    test_cities: tuple[str, ...]
    training: tuple[Question, ...]
    held_out: tuple[Question, ...]
    left_out: tuple[tuple[str, str], ...]


def sql_literal(value: str) -> str:
    """Write a value the way it appears inside SQL text.

    Args:
        value: A string taken from the database.

    Returns:
        The value as SQL source; quotes are doubled, as SQL requires.
    """
    return value.replace("'", "''")


def candidate_sql(template: Template, choice: int, city: str) -> str:
    """Fill one candidate query with a city.

    Args:
        template: The question's template.
        choice: Which candidate (0 for A).
        city: The city for the ``{city}`` slot.

    Returns:
        The query, ready to run.
    """
    return template.candidates[choice].sql.format(city=sql_literal(city))


def format_answer(rows: list[tuple[Any, ...]]) -> str:
    """Turn a query result into the text that goes into answer.txt.

    Args:
        rows: What the query returned.

    Returns:
        The first value as text (a decimal number with two places), or "" when
        the query returned nothing.
    """
    if not rows or not rows[0] or rows[0][0] is None:
        return ""
    value = rows[0][0]
    if isinstance(value, float):
        return f"{value:.{DECIMALS}f}"
    return str(value)


def run_query(database: Path, sql: str) -> str:
    """Run one query on a database, opened read-only, and format its result.

    Args:
        database: The SQLite file.
        sql: The query.

    Returns:
        The answer text, as ``format_answer`` writes it.

    Raises:
        sqlite3.Error: If the query fails.
    """
    with closing(sqlite3.connect(database.as_uri() + READ_ONLY, uri=True)) as db:
        return format_answer(db.execute(sql).fetchall())


def unfair_reason(template: Template, answers: list[str], task: Task) -> str | None:
    """Say why a filled-in question is unfair, or None when it is fair.

    Args:
        template: The question's template.
        answers: What each candidate returns for this city, A to D.
        task: The question with its gold answer.

    Returns:
        A short reason, or None.
    """
    if not task.answer:
        return NO_ANSWER
    for choice, answer in enumerate(answers):
        if choice != template.right and check_answer(task, answer).correct:
            return f"query {LETTERS[choice]} also gives {task.answer}"
    return None


def make_questions() -> QuestionSet:
    """Fill every template with every city and split the fair ones by city.

    Returns:
        The training and held-out questions, and what was left out.

    Raises:
        sqlite3.Error: If a query fails on the bundled database.
    """
    with closing(sqlite3.connect(SHOP_DB.as_uri() + READ_ONLY, uri=True)) as db:
        cities = tuple(row[0] for row in db.execute(CITIES_SQL))
    test_cities = cities[HELD_OUT_EVERY - 1 :: HELD_OUT_EVERY]
    training: list[Question] = []
    held_out: list[Question] = []
    left_out: list[tuple[str, str]] = []
    for row, template in enumerate(TEMPLATES):
        for city in cities:
            answers = [
                run_query(SHOP_DB, candidate_sql(template, choice, city))
                for choice in range(len(template.candidates))
            ]
            split = TEST if city in test_cities else TRAIN
            task = Task(
                task_id=f"{template.key}-{city}",
                split=split,
                domain=DOMAIN,
                difficulty=DIFFICULTY,
                db_file=SHOP_DB.name,
                question=template.question.format(city=city),
                gold_sql=candidate_sql(template, template.right, city),
                answer=answers[template.right],
                answer_type=template.answer_type,
            )
            reason = unfair_reason(template, answers, task)
            if reason:
                left_out.append((task.question, reason))
            elif split == TEST:
                held_out.append(Question(row, city, task))
            else:
                training.append(Question(row, city, task))
    train_cities = tuple(city for city in cities if city not in test_cities)
    return QuestionSet(
        train_cities, test_cities, tuple(training), tuple(held_out), tuple(left_out)
    )


def attempt(question: Question, choice: int) -> tuple[str, Grade]:
    """Do one try: run the picked query in a fresh folder and grade the answer.

    The folder holds a copy of shop.db as data.db. The query's result goes
    into answer.txt, chapter 1's grader judges it, and the folder is deleted.

    Args:
        question: The question being answered.
        choice: Which candidate query to run (0 for A).

    Returns:
        The answer written to answer.txt, and the grader's verdict.

    Raises:
        OSError: If the work folder cannot be made or the database copied.
        sqlite3.Error: If the query fails.
    """
    root = work_root()
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=TRY_PREFIX, dir=root))
    try:
        shutil.copyfile(SHOP_DB, workdir / DATABASE_FILE)
        template = TEMPLATES[question.row]
        sql = candidate_sql(template, choice, question.city)
        answer = run_query(workdir / DATABASE_FILE, sql)
        (workdir / ANSWER_FILE).write_text(answer + "\n", encoding="utf-8")
        return answer, grade(question.task, workdir)
    finally:
        shutil.rmtree(workdir)


def snippet(template: Template, choice: int, city: str) -> str:
    """Shorten a query for one line of output, keeping what sets it apart.

    A short query is shown whole. A long one is cut after the first words
    that no other candidate for the same question starts with.

    Args:
        template: The question's template.
        choice: Which candidate (0 for A).
        city: The city in the question.

    Returns:
        The query, or its first words followed by " ...".
    """
    words = [
        candidate_sql(template, index, city).split()
        for index in range(len(template.candidates))
    ]
    mine = words[choice]
    if len(" ".join(mine)) <= SNIPPET_WIDTH:
        return " ".join(mine)
    others = [other for index, other in enumerate(words) if index != choice]
    for count in range(1, len(mine)):
        if all(other[:count] != mine[:count] for other in others):
            return " ".join(mine[:count]) + ELLIPSIS
    return " ".join(mine)
