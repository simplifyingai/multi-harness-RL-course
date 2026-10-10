"""Six questions at three difficulties, three queries each, and the split by city.

Chapter 4's questions, now labelled the way the project labels its own (by
what the SQL needs, not by how the sentence reads):

* easy: one table, a filter and one aggregate;
* medium: a two-table join with a filter;
* hard: three or more tables (here all four).

Each question has a ``{city}`` slot and three candidate queries. QUERY_A runs
the right one, QUERY_B and QUERY_C two plausible mistakes. The agent is not
told which is which: it sees only the names of its actions, and the grader
decides by comparing answer.txt with the gold answer.

A filled-in question is kept only when it is fair: the right query returns an
answer, and no mistake happens to return the same one. Every third city in
alphabetical order is held out, as in chapter 4, so the final test asks about
cities the agent never trained on.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grader import check_answer
from tasks import DATA_DIR, NUMBER, TEXT, Task

SHOP_DB = DATA_DIR / "shop.db"
CITIES_SQL = "SELECT DISTINCT city FROM customers ORDER BY city"
LETTERS = "ABC"
HELD_OUT_EVERY = 3  # every third city, in alphabetical order, is kept for the test
DOMAIN = "shop"
TRAIN = "train"
TEST = "test"
DECIMALS = 2
READ_ONLY = "?mode=ro"
NO_ANSWER = "no answer"

# The three difficulties, in order.
EASY = "easy"
MEDIUM = "medium"
HARD = "hard"
LEVELS = (EASY, MEDIUM, HARD)
LEVEL_NOTES = {  # what the SQL needs, as the project's generator labels it
    EASY: "one table, a filter, one aggregate",
    MEDIUM: "a two-table join with a filter",
    HARD: "three or more tables (here four)",
}

# Joins several queries share: one city's orders, its order lines, and its
# order lines for electronics (all four tables).
ORDERS_OF_CITY = (
    "FROM orders JOIN customers ON customers.id = orders.customer_id "
    "WHERE customers.city = '{city}'"
)
ITEMS_OF_CITY = (
    "FROM order_items JOIN orders ON orders.id = order_items.order_id "
    "JOIN customers ON customers.id = orders.customer_id "
    "WHERE customers.city = '{city}'"
)
ELECTRONICS_OF_CITY = (
    "FROM order_items JOIN orders ON orders.id = order_items.order_id "
    "JOIN customers ON customers.id = orders.customer_id "
    "JOIN products ON products.id = order_items.product_id "
    "WHERE customers.city = '{city}' AND products.category = 'Electronics'"
)


@dataclass(frozen=True)
class Candidate:
    """One query the agent may run.

    Attributes:
        sql: The query, with a ``{city}`` slot.
        note: What it really computes, in plain words. Only people read this.
    """

    sql: str
    note: str


@dataclass(frozen=True)
class Template:
    """A question with a ``{city}`` slot and its three candidate queries.

    Attributes:
        key: A short name, used in task ids.
        level: ``"easy"``, ``"medium"`` or ``"hard"``.
        question: The question, e.g. ``"How many customers live in {city}?"``.
        answer_type: ``"number"`` or ``"text"``, so the grader knows how to compare.
        candidates: Queries A to C; A is the right one.
    """

    key: str
    level: str
    question: str
    answer_type: str
    candidates: tuple[Candidate, ...]


# The right query first, then two mistakes. The medium questions and the
# first easy one are chapter 4's.
TEMPLATES = (
    Template(
        "customers",
        EASY,
        "How many customers live in {city}?",
        NUMBER,
        (
            Candidate(
                "SELECT COUNT(*) FROM customers WHERE city = '{city}'",
                "right: counts the customers in the city",
            ),
            Candidate(
                "SELECT COUNT(*) FROM customers",
                "no WHERE filter: counts every customer",
            ),
            Candidate(
                "SELECT SUM(quantity) " + ITEMS_OF_CITY,
                "SUM instead of COUNT: adds up the items they bought",
            ),
        ),
    ),
    Template(
        "newest-customer",
        EASY,
        "When did the newest customer from {city} sign up?",
        TEXT,
        (
            Candidate(
                "SELECT MAX(signup_date) FROM customers WHERE city = '{city}'",
                "right: the latest sign-up in the city",
            ),
            Candidate(
                "SELECT MAX(signup_date) FROM customers",
                "no WHERE filter: the shop's newest customer",
            ),
            Candidate(
                "SELECT MIN(signup_date) FROM customers WHERE city = '{city}'",
                "MIN instead of MAX: the first to sign up",
            ),
        ),
    ),
    Template(
        "orders",
        MEDIUM,
        "How many orders did customers from {city} place?",
        NUMBER,
        (
            Candidate(
                "SELECT COUNT(*) " + ORDERS_OF_CITY, "right: counts their orders"
            ),
            Candidate(
                "SELECT COUNT(*) FROM orders", "no WHERE filter: counts every order"
            ),
            Candidate(
                "SELECT SUM(quantity) " + ITEMS_OF_CITY,
                "SUM instead of COUNT: adds up the items in those orders",
            ),
        ),
    ),
    Template(
        "first-order",
        MEDIUM,
        "When was the first order from {city}?",
        TEXT,
        (
            Candidate(
                "SELECT MIN(order_date) " + ORDERS_OF_CITY, "right: the first order"
            ),
            Candidate(
                "SELECT MIN(signup_date) FROM customers WHERE city = '{city}'",
                "the wrong date: the first sign-up, not the first order",
            ),
            Candidate(
                "SELECT MAX(order_date) " + ORDERS_OF_CITY,
                "MAX instead of MIN: the last order",
            ),
        ),
    ),
    Template(
        "electronics-spend",
        HARD,
        "How much did customers from {city} spend on electronics?",
        NUMBER,
        (
            Candidate(
                "SELECT SUM(quantity * unit_price) " + ELECTRONICS_OF_CITY,
                "right: quantity times price, electronics only",
            ),
            Candidate(
                "SELECT SUM(quantity * unit_price) " + ITEMS_OF_CITY,
                "forgets the category: everything they spent",
            ),
            Candidate(
                "SELECT SUM(unit_price) " + ELECTRONICS_OF_CITY,
                "forgets the quantity: adds up prices only",
            ),
        ),
    ),
    Template(
        "electronics-items",
        HARD,
        "How many electronics items did customers from {city} buy?",
        NUMBER,
        (
            Candidate(
                "SELECT SUM(quantity) " + ELECTRONICS_OF_CITY,
                "right: adds up the quantities, electronics only",
            ),
            Candidate(
                "SELECT COUNT(*) " + ELECTRONICS_OF_CITY,
                "COUNT instead of SUM: counts order lines, not items",
            ),
            Candidate(
                "SELECT SUM(quantity) " + ITEMS_OF_CITY,
                "forgets the category: every item they bought",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class Question:
    """One template filled with one city.

    Attributes:
        row: Which template (0 to 5): the question's kind, whatever the city.
        city: The city in the slot.
        task: The question text, gold SQL and gold answer, for the grader.
    """

    row: int
    city: str
    task: Task

    @property
    def level(self) -> str:
        """The question's difficulty: ``"easy"``, ``"medium"`` or ``"hard"``."""
        return TEMPLATES[self.row].level


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
    # The cities come from the bundled database, and sql_literal doubles any
    # quote, so a city's name can never end its string early.
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


def unfair_reason(answers: list[str], task: Task) -> str | None:
    """Say why a filled-in question is unfair, or None when it is fair.

    Args:
        answers: What each candidate returns for this city, A to C.
        task: The question with its gold answer (query A's).

    Returns:
        A short reason, or None.
    """
    if not task.answer:
        return NO_ANSWER
    for choice, answer in enumerate(answers[1:], start=1):
        if check_answer(task, answer).correct:
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
                difficulty=template.level,
                db_file=SHOP_DB.name,
                question=template.question.format(city=city),
                gold_sql=candidate_sql(template, 0, city),
                answer=answers[0],
                answer_type=template.answer_type,
            )
            reason = unfair_reason(answers, task)
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


def count_by_level(questions: tuple[Question, ...]) -> dict[str, int]:
    """Count questions per difficulty.

    Args:
        questions: Any questions.

    Returns:
        How many are easy, medium and hard, in that order.
    """
    return {level: sum(q.level == level for q in questions) for level in LEVELS}
