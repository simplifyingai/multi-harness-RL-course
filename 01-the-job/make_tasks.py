# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Draft data questions from templates, each with the SQL that answers it.

A small, seeded copy of the project's task generator (``mhrl/tasks/
generate.py``), over the shop database only. Every question is stored with
its *gold SQL*: the query whose result is the correct answer. Drafts a guess
could pass are thrown away. The same seed always writes the same file, which
the md5 at the end proves.

Run it with ``uv run make_tasks.py --seed 0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tasks import CHAPTER, DATA_DIR, LIST, NUMBER, TEXT, course_home, short_path

SHOP_DB = DATA_DIR / "dbs" / "shop.db"
OUTPUT_DIR = "made-tasks"
DEFAULT_SEED = 0
DEFAULT_COUNT = 6
DIFFICULTIES = ("easy", "medium", "hard")

# The project's rules for a fair question (mhrl/tasks/questions.py).
MIN_INTEGER_ANSWER = 2  # writing "0" or "1" must never be rewarded
TIE_MARGIN = 0.01  # "which is the most" needs a clear winner
MIN_LIST_ITEMS = 2
MAX_LIST_ITEMS = 8
MAX_SAME_ANSWER = 2  # no answer worth guessing
MAX_DRAWS = 200
DECIMALS = 2
SQL_BREAK_RE = re.compile(r"\s+(?=(FROM|JOIN|WHERE|GROUP BY|HAVING|ORDER BY)\b)")
INDENT = " " * 8


@dataclass(frozen=True)
class Template:
    """A question pattern and the SQL that answers it.

    Attributes:
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        question: The question, with ``{name}`` slots.
        sql: The gold SQL, with the same slots.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.
        slots: For each slot, a query listing the values it may take.
        ranking: For "which is the most" questions, a query returning
            ``(name, score)`` best first, used to refuse ties.
    """

    difficulty: str
    question: str
    sql: str
    answer_type: str
    slots: dict[str, str] = field(default_factory=dict)
    ranking: str | None = None


CITY = "SELECT DISTINCT city FROM customers ORDER BY city"
CATEGORY = "SELECT DISTINCT category FROM products ORDER BY category"
METHOD = "SELECT DISTINCT payment_method FROM orders ORDER BY payment_method"
YEAR = "SELECT DISTINCT strftime('%Y', order_date) FROM orders ORDER BY 1"
SIGNUP_YEAR = "SELECT DISTINCT strftime('%Y', signup_date) FROM customers ORDER BY 1"
ORDER_THRESHOLD = "SELECT value FROM (SELECT 2 AS value UNION SELECT 3 UNION SELECT 4)"

TEMPLATES = (
    Template(
        "easy",
        "How many customers live in {city}?",
        "SELECT COUNT(*) FROM customers WHERE city = '{city}'",
        NUMBER,
        {"city": CITY},
    ),
    Template(
        "easy",
        "How many products are in the {category} category?",
        "SELECT COUNT(*) FROM products WHERE category = '{category}'",
        NUMBER,
        {"category": CATEGORY},
    ),
    Template(
        "easy",
        "How many orders were paid with {method}?",
        "SELECT COUNT(*) FROM orders WHERE payment_method = '{method}'",
        NUMBER,
        {"method": METHOD},
    ),
    Template(
        "easy",
        "What is the average price of a {category} product?",
        "SELECT ROUND(AVG(price), 2) FROM products WHERE category = '{category}'",
        NUMBER,
        {"category": CATEGORY},
    ),
    Template(
        "medium",
        "Which city had the most new customers sign up in {year}?",
        "SELECT city FROM customers WHERE strftime('%Y', signup_date) = '{year}' "
        "GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1",
        TEXT,
        {"year": SIGNUP_YEAR},
        "SELECT city, COUNT(*) FROM customers "
        "WHERE strftime('%Y', signup_date) = '{year}' GROUP BY city ORDER BY 2 DESC",
    ),
    Template(
        "medium",
        "How many orders did customers from {city} place?",
        "SELECT COUNT(*) FROM orders JOIN customers ON customers.id = "
        "orders.customer_id WHERE customers.city = '{city}'",
        NUMBER,
        {"city": CITY},
    ),
    Template(
        "medium",
        "Which payment method was used most often in {year}?",
        "SELECT payment_method FROM orders WHERE strftime('%Y', order_date) = "
        "'{year}' GROUP BY payment_method ORDER BY COUNT(*) DESC LIMIT 1",
        TEXT,
        {"year": YEAR},
        "SELECT payment_method, COUNT(*) FROM orders "
        "WHERE strftime('%Y', order_date) = '{year}' GROUP BY 1 ORDER BY 2 DESC",
    ),
    Template(
        "hard",
        "What was the total revenue from {category} products in {year}?",
        "SELECT ROUND(SUM(order_items.quantity * order_items.unit_price), 2) "
        "FROM order_items JOIN orders ON orders.id = order_items.order_id "
        "JOIN products ON products.id = order_items.product_id "
        "WHERE products.category = '{category}' "
        "AND strftime('%Y', orders.order_date) = '{year}'",
        NUMBER,
        {"category": CATEGORY, "year": YEAR},
    ),
    Template(
        "hard",
        "Which customer from {city} spent the most in total?",
        "SELECT customers.name FROM customers "
        "JOIN orders ON orders.customer_id = customers.id "
        "JOIN order_items ON order_items.order_id = orders.id "
        "WHERE customers.city = '{city}' GROUP BY customers.id "
        "ORDER BY SUM(order_items.quantity * order_items.unit_price) DESC LIMIT 1",
        TEXT,
        {"city": CITY},
        "SELECT customers.name, SUM(order_items.quantity * order_items.unit_price) "
        "FROM customers JOIN orders ON orders.customer_id = customers.id "
        "JOIN order_items ON order_items.order_id = orders.id "
        "WHERE customers.city = '{city}' GROUP BY customers.id ORDER BY 2 DESC",
    ),
    Template(
        "hard",
        "Which {category} products were never ordered?",
        "SELECT name FROM products WHERE category = '{category}' AND id NOT IN "
        "(SELECT product_id FROM order_items) ORDER BY name",
        LIST,
        {"category": CATEGORY},
    ),
    Template(
        "hard",
        "How many customers placed more than {count} orders?",
        "SELECT COUNT(*) FROM (SELECT customer_id FROM orders "
        "GROUP BY customer_id HAVING COUNT(*) > {count})",
        NUMBER,
        {"count": ORDER_THRESHOLD},
    ),
)


@dataclass(frozen=True)
class Draft:
    """One drafted task, stored as one JSON line.

    Attributes:
        task_id: For example ``"seed0-easy-01"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        question: The question.
        gold_sql: The query whose result is the answer.
        answer: The correct answer as text.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.
    """

    task_id: str
    difficulty: str
    question: str
    gold_sql: str
    answer: str
    answer_type: str


def sql_literal(value: object) -> str:
    """Write a value the way it appears inside SQL text.

    Args:
        value: A string or a number taken from the database.

    Returns:
        The value as SQL source; quotes are doubled, as SQL requires.
    """
    return str(value).replace("'", "''")


def format_answer(rows: list[tuple[Any, ...]], answer_type: str) -> str | None:
    """Turn a query result into the answer text, or None if it is unusable.

    Args:
        rows: The rows the gold SQL returned.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.

    Returns:
        The answer text, or None when the result is empty.
    """
    values = [row[0] for row in rows if row and row[0] is not None]
    if not values:
        return None
    if answer_type == LIST:
        return ", ".join(str(value) for value in values)
    value = values[0]
    if answer_type == NUMBER and isinstance(value, float):
        return f"{value:.{DECIMALS}f}"
    return str(value)


def rejection(
    template: Template,
    answer: str | None,
    ranking: list[tuple[Any, ...]],
    used: Counter[str],
) -> str | None:
    """Say why a draft is unfair, or None when it is a good question.

    Args:
        template: The template the draft came from.
        answer: The gold SQL's answer.
        ranking: ``(name, score)`` rows best first, for "the most" questions.
        used: How often each answer has been used already.

    Returns:
        A short reason, or None.
    """
    if answer is None:
        return "empty answer"
    if template.answer_type == NUMBER:
        number = float(answer)
        if number == int(number) and number < MIN_INTEGER_ANSWER:
            return "answer 0 or 1"
    if template.answer_type == LIST:
        size = len(answer.split(", "))
        if not MIN_LIST_ITEMS <= size <= MAX_LIST_ITEMS:
            return "list too short or long"
    if len(ranking) > 1 and abs(ranking[0][1] - ranking[1][1]) <= TIE_MARGIN:
        return "tie for first"
    if used[answer] >= MAX_SAME_ANSWER:
        return "answer used too often"
    return None


@dataclass
class Drafting:
    """What drafting has used so far, so no question or answer repeats.

    Attributes:
        connection: A read-only connection to the shop database.
        rng: The seeded random generator.
        questions: Questions already drafted.
        used: How often each answer has been used.
        rejected: Thrown-away drafts, counted by reason.
    """

    connection: sqlite3.Connection
    rng: random.Random
    questions: set[str] = field(default_factory=set)
    used: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)


def pick(state: Drafting, query: str) -> str:
    """Pick one value for a slot from the values its query lists.

    Args:
        state: Holds the database connection and the seeded random generator.
        query: A query listing the slot's possible values.

    Returns:
        The chosen value, as text.

    Raises:
        sqlite3.Error: If the query fails.
    """
    rows = state.connection.execute(query).fetchall()
    chosen = state.rng.choice(rows)
    return str(chosen[0])


def draw(state: Drafting, template: Template) -> tuple[str, str, str] | None:
    """Fill a template's slots until it makes a fair question.

    Args:
        state: What has been drafted so far.
        template: The template to fill.

    Returns:
        The question, its gold SQL and its answer, or None if no fair draft
        turned up in ``MAX_DRAWS`` tries.

    Raises:
        sqlite3.Error: If a template's SQL fails.
    """
    for _ in range(MAX_DRAWS):
        values = {name: pick(state, query) for name, query in template.slots.items()}
        question = template.question.format(**values)
        if question in state.questions:
            state.rejected["question already drafted"] += 1
            continue
        # Inside SQL text a quote must be doubled; in the question it must not.
        in_sql = {name: sql_literal(value) for name, value in values.items()}
        gold_sql = template.sql.format(**in_sql)
        answer = format_answer(
            state.connection.execute(gold_sql).fetchall(), template.answer_type
        )
        ranking = (
            state.connection.execute(template.ranking.format(**in_sql)).fetchall()
            if template.ranking
            else []
        )
        reason = rejection(template, answer, ranking, state.used)
        if reason:
            state.rejected[reason] += 1
            continue
        state.questions.add(question)
        state.used[answer] += 1
        return question, gold_sql, answer
    return None


def share_for(level: int, count: int) -> int:
    """Split ``count`` tasks as evenly as possible over the difficulties.

    Args:
        level: 0 for easy, 1 for medium, 2 for hard.
        count: How many tasks in total.

    Returns:
        How many tasks this difficulty gets; the first ones get any extra.
    """
    base, extra = divmod(count, len(DIFFICULTIES))
    return base + 1 if level < extra else base


def draft_one(
    state: Drafting, templates: list[Template], number: int, task_id: str
) -> Draft | None:
    """Draft one task, letting the templates take turns.

    Task 1 starts with the first template, task 2 with the second, and so on.
    When a template cannot make a fair question, the next one tries.

    Args:
        state: What has been drafted so far.
        templates: This difficulty's templates, in shuffled order.
        number: Which task of this difficulty (1, 2, ...).
        task_id: The id to give the task.

    Returns:
        The task, or None when no template could make a fair question.
    """
    start = (number - 1) % len(templates)
    for template in templates[start:] + templates[:start]:
        made = draw(state, template)
        if made:
            question, gold_sql, answer = made
            return Draft(
                task_id=task_id,
                difficulty=template.difficulty,
                question=question,
                gold_sql=gold_sql,
                answer=answer,
                answer_type=template.answer_type,
            )
    return None


def draft_tasks(seed: int, count: int) -> tuple[list[Draft], Counter[str]]:
    """Draft ``count`` fair tasks, spread over the three difficulties.

    Args:
        seed: The random seed; the same seed gives the same tasks.
        count: How many tasks to draft.

    Returns:
        The tasks and a count of rejected drafts by reason, including
        ``"no fair draft"`` for every task that could not be made: a shortfall
        is reported, never hidden.
    """
    state = Drafting(
        connection=sqlite3.connect(f"file:{SHOP_DB}?mode=ro", uri=True),
        rng=random.Random(seed),
    )
    drafts: list[Draft] = []
    try:
        for level, difficulty in enumerate(DIFFICULTIES):
            templates = [t for t in TEMPLATES if t.difficulty == difficulty]
            state.rng.shuffle(templates)
            for number in range(1, share_for(level, count) + 1):
                task_id = f"seed{seed}-{difficulty}-{number:02d}"
                draft = draft_one(state, templates, number, task_id)
                if draft:
                    drafts.append(draft)
                else:
                    state.rejected["no fair draft"] += 1
    finally:
        state.connection.close()
    return drafts, state.rejected


def pretty_sql(sql: str) -> str:
    """Break a query before each clause so it reads in a narrow terminal.

    Args:
        sql: One line of SQL.

    Returns:
        The query over several indented lines.
    """
    return SQL_BREAK_RE.sub("\n" + INDENT + "  ", sql)


def main() -> None:
    """Draft the tasks, print them, write them and print the file's md5."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()

    drafts, rejected = draft_tasks(args.seed, args.count)
    print(f"seed {args.seed} · {len(drafts)} tasks from {len(TEMPLATES)} templates\n")
    for draft in drafts:
        print(f"{draft.difficulty:<8}{draft.question}")
        print(f"{INDENT}{pretty_sql(draft.gold_sql)}")
        print(f"{INDENT}answer: {draft.answer}\n")
    if len(drafts) < args.count:
        print(f"drafted only {len(drafts)} of {args.count}: see 'no fair draft' below")
    if rejected:
        reasons = ", ".join(f"{n} {why}" for why, n in sorted(rejected.items()))
        print(f"thrown away: {reasons}")

    output = course_home() / CHAPTER / OUTPUT_DIR / f"seed-{args.seed}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in drafts)
    output.write_text(text, encoding="utf-8")
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    print(f"wrote {short_path(Path(output))}")
    print(f"md5   {digest}")


if __name__ == "__main__":
    main()
