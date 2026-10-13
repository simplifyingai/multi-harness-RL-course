# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Make a new task family: seeded spreadsheet questions, each with its gold answer.

Chapter 1's ``make_tasks.py`` pointed at new data. The machinery is
unchanged: each template is a question with slots, the gold SQL that
answers it, and for each slot a query listing the values it may take; the
same fairness rules throw drafts away (an answer of 0 or 1, a tie for
first, a list of one item or of more than eight, an answer used more than
twice). What changes is the data and the templates: twelve questions over
``data/sales.csv``, whose gold SQL runs on the spreadsheet loaded with typed
columns (``sheet.py``). The same seed always writes the same file, which the
md5 at the end proves.

Run it with ``uv run make_csv_tasks.py`` (seed 0, 20 tasks); ``--seed`` and
``--count`` make others. Only Python's standard library is needed.
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

from lfm_files import short_path
from sheet import SHEET, open_sheet
from tasks import (
    DEFAULT_SEED,
    DIFFICULTIES,
    DOMAIN,
    LIST,
    NUMBER,
    TEXT,
    Task,
    load_tasks,
    tasks_path,
)

DEFAULT_COUNT = 20
EASY, MEDIUM, HARD = DIFFICULTIES
ID_PREFIX = "csv"  # task ids: csv-s<seed>-<difficulty>-<number>

# Chapter 1's rules for a fair question (the project's mhrl/tasks/questions.py).
MIN_INTEGER_ANSWER = 2  # writing "0" or "1" must never be rewarded
TIE_MARGIN = 0.01  # "which is the most" needs a clear winner
MIN_LIST_ITEMS = 2
MAX_LIST_ITEMS = 8
MAX_SAME_ANSWER = 2  # no answer worth guessing
MAX_DRAWS = 200
# Why a draft was thrown away, as the counts print them.
EMPTY_ANSWER = "empty answer"
ZERO_OR_ONE = "answer 0 or 1"
LIST_SIZE = "list too short or long"
TIE = "tie for first"
USED_TOO_OFTEN = "answer used too often"
ALREADY_DRAFTED = "question already drafted"
NO_FAIR_DRAFT = "no fair draft"
DECIMALS = 2
PARTIAL_SUFFIX = ".partial"
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


REGION = "SELECT DISTINCT region FROM sales ORDER BY region"
REP = "SELECT DISTINCT rep FROM sales ORDER BY rep"
PRODUCT = "SELECT DISTINCT product FROM sales ORDER BY product"
CATEGORY = "SELECT DISTINCT category FROM sales ORDER BY category"
UNITS_THRESHOLD = (
    "SELECT value FROM (SELECT 20 AS value UNION SELECT 30 UNION SELECT 40)"
)
AMOUNT = "SELECT value FROM (SELECT 200 AS value UNION SELECT 400 UNION SELECT 600)"
REVENUE = "units * unit_price"  # an order's worth; the questions spell it out

TEMPLATES = (
    Template(
        EASY,
        "How many orders came from the {region} region?",
        "SELECT COUNT(*) FROM sales WHERE region = '{region}'",
        NUMBER,
        {"region": REGION},
    ),
    Template(
        EASY,
        "How many units of {product} were sold in total?",
        "SELECT SUM(units) FROM sales WHERE product = '{product}'",
        NUMBER,
        {"product": PRODUCT},
    ),
    Template(
        EASY,
        "How many orders did {rep} take?",
        "SELECT COUNT(*) FROM sales WHERE rep = '{rep}'",
        NUMBER,
        {"rep": REP},
    ),
    Template(
        EASY,
        "What is the highest unit price paid for a {category} product?",
        "SELECT MAX(unit_price) FROM sales WHERE category = '{category}'",
        NUMBER,
        {"category": CATEGORY},
    ),
    Template(
        MEDIUM,
        "What was the total revenue (units times unit price) in the {region} region?",
        f"SELECT ROUND(SUM({REVENUE}), 2) FROM sales WHERE region = '{{region}}'",
        NUMBER,
        {"region": REGION},
    ),
    Template(
        MEDIUM,
        "Which sales rep sold the most units in the {region} region?",
        "SELECT rep FROM sales WHERE region = '{region}' "
        "GROUP BY rep ORDER BY SUM(units) DESC LIMIT 1",
        TEXT,
        {"region": REGION},
        "SELECT rep, SUM(units) FROM sales WHERE region = '{region}' "
        "GROUP BY rep ORDER BY 2 DESC",
    ),
    Template(
        MEDIUM,
        "Which {category} product brought in the most revenue "
        "(units times unit price)?",
        f"SELECT product FROM sales WHERE category = '{{category}}' "
        f"GROUP BY product ORDER BY SUM({REVENUE}) DESC LIMIT 1",
        TEXT,
        {"category": CATEGORY},
        f"SELECT product, SUM({REVENUE}) FROM sales "
        f"WHERE category = '{{category}}' GROUP BY product ORDER BY 2 DESC",
    ),
    Template(
        MEDIUM,
        "What is the average number of units per order for {product}?",
        "SELECT ROUND(AVG(units), 2) FROM sales WHERE product = '{product}'",
        NUMBER,
        {"product": PRODUCT},
    ),
    Template(
        HARD,
        "Which sales reps took no orders in the {region} region?",
        "SELECT DISTINCT rep FROM sales WHERE rep NOT IN "
        "(SELECT rep FROM sales WHERE region = '{region}') ORDER BY rep",
        LIST,
        {"region": REGION},
    ),
    Template(
        HARD,
        "Which sales reps sold more than {count} units of {category} products "
        "in total?",
        "SELECT rep FROM sales WHERE category = '{category}' "
        "GROUP BY rep HAVING SUM(units) > {count} ORDER BY rep",
        LIST,
        {"count": UNITS_THRESHOLD, "category": CATEGORY},
    ),
    Template(
        HARD,
        "How many orders were worth more than {amount} (units times unit price)?",
        f"SELECT COUNT(*) FROM sales WHERE {REVENUE} > {{amount}}",
        NUMBER,
        {"amount": AMOUNT},
    ),
    Template(
        HARD,
        "What was the total revenue (units times unit price) from {product} "
        "in the {region} region?",
        f"SELECT ROUND(SUM({REVENUE}), 2) FROM sales "
        f"WHERE product = '{{product}}' AND region = '{{region}}'",
        NUMBER,
        {"product": PRODUCT, "region": REGION},
    ),
)


def sql_literal(value: object) -> str:
    """Write a value the way it appears inside SQL text.

    Args:
        value: A string or a number taken from the spreadsheet.

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
        return EMPTY_ANSWER
    if template.answer_type == NUMBER:
        number = float(answer)
        if number == int(number) and number < MIN_INTEGER_ANSWER:
            return ZERO_OR_ONE
    if template.answer_type == LIST:
        size = len(answer.split(", "))
        if not MIN_LIST_ITEMS <= size <= MAX_LIST_ITEMS:
            return LIST_SIZE
    if len(ranking) > 1 and abs(ranking[0][1] - ranking[1][1]) <= TIE_MARGIN:
        return TIE
    if used[answer] >= MAX_SAME_ANSWER:
        return USED_TOO_OFTEN
    return None


@dataclass
class Drafting:
    """What drafting has used so far, so no question or answer repeats.

    Attributes:
        connection: The spreadsheet, loaded with typed columns.
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
        state: Holds the spreadsheet and the seeded random generator.
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
            state.rejected[ALREADY_DRAFTED] += 1
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
        if reason is not None or answer is None:  # rejection() names an empty one
            state.rejected[reason or EMPTY_ANSWER] += 1
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
) -> Task | None:
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
            return Task(
                task_id=task_id,
                domain=DOMAIN,
                difficulty=template.difficulty,
                data_file=SHEET.name,
                question=question,
                gold_sql=gold_sql,
                answer=answer,
                answer_type=template.answer_type,
            )
    return None


def draft_tasks(seed: int, count: int) -> tuple[list[Task], Counter[str]]:
    """Draft ``count`` fair tasks, spread over the three difficulties.

    Args:
        seed: The random seed; the same seed gives the same tasks.
        count: How many tasks to draft.

    Returns:
        The tasks and a count of rejected drafts by reason, including
        ``"no fair draft"`` for every task that could not be made: a shortfall
        is reported, never hidden.
    """
    state = Drafting(connection=open_sheet(), rng=random.Random(seed))
    drafts: list[Task] = []
    try:
        for level, difficulty in enumerate(DIFFICULTIES):
            templates = [t for t in TEMPLATES if t.difficulty == difficulty]
            state.rng.shuffle(templates)
            for number in range(1, share_for(level, count) + 1):
                task_id = f"{ID_PREFIX}-s{seed}-{difficulty}-{number:02d}"
                draft = draft_one(state, templates, number, task_id)
                if draft:
                    drafts.append(draft)
                else:
                    state.rejected[NO_FAIR_DRAFT] += 1
    finally:
        state.connection.close()
    return drafts, state.rejected


def tasks_text(tasks: list[Task]) -> str:
    """Write tasks as JSON lines, the file's exact bytes.

    Args:
        tasks: The tasks.

    Returns:
        One JSON object per line.
    """
    return "".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in tasks)


def write_tasks(seed: int, tasks: list[Task]) -> tuple[Path, str]:
    """Write one seed's tasks and fingerprint the file.

    Args:
        seed: The seed they were drafted with.
        tasks: The tasks.

    Returns:
        The file and its md5.
    """
    text = tasks_text(tasks)
    output = tasks_path(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Written beside it, then renamed: a stopped write never leaves a short
    # file that would read as a complete one.
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    partial.write_text(text, encoding="utf-8")
    partial.replace(output)
    return output, hashlib.md5(text.encode("utf-8")).hexdigest()


def reasons_text(rejected: Counter[str]) -> str:
    """Write the thrown-away drafts, counted by reason.

    Args:
        rejected: Counts by reason.

    Returns:
        Such as ``2 empty answer, 1 question already drafted``.
    """
    return ", ".join(f"{n} {why}" for why, n in sorted(rejected.items()))


def ensure_tasks(seed: int = DEFAULT_SEED, count: int | None = None) -> list[Task]:
    """Load one seed's tasks, making them first when the file is missing.

    A script that runs the family never sends you back to make its input. A
    file already there is used as it is; when ``count`` is given and the file
    holds another number of tasks, that is said, never silently ignored.

    Args:
        seed: The seed.
        count: How many to make when the file is missing (``DEFAULT_COUNT``
            when None); when given, also how many the file should hold.

    Returns:
        The seed's tasks.

    Raises:
        SystemExit: If no task could be made.
    """
    wanted = DEFAULT_COUNT if count is None else count
    path = tasks_path(seed)
    if not path.is_file():
        tasks, rejected = draft_tasks(seed, wanted)
        if not tasks:
            raise SystemExit(f"seed {seed}: no fair task could be drafted")
        path, digest = write_tasks(seed, tasks)
        print(f"made      {len(tasks)} tasks · {short_path(path)} · md5 {digest}")
        if len(tasks) < wanted:
            print(f"  only {len(tasks)} of {wanted}: {reasons_text(rejected)}")
    found = load_tasks(path)
    if count is not None and len(found) != count:
        print(
            f"tasks     {short_path(path)} holds {len(found)} tasks, not the {count} "
            "asked: delete it to make them again"
        )
    return found


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
    if args.count < 1:
        parser.error("--count: at least 1 task")

    tasks, rejected = draft_tasks(args.seed, args.count)
    print(
        f"seed {args.seed} · {len(tasks)} tasks from {len(TEMPLATES)} templates · "
        f"over {SHEET.name}\n"
    )
    for task in tasks:
        print(f"{task.difficulty:<8}{task.question}")
        print(f"{INDENT}{pretty_sql(task.gold_sql)}")
        print(f"{INDENT}answer: {task.answer}\n")
    if len(tasks) < args.count:
        print(f"drafted only {len(tasks)} of {args.count}: see 'no fair draft' below")
    if rejected:
        print(f"thrown away: {reasons_text(rejected)}")
    kinds = Counter(task.answer_type for task in tasks)
    print("answers   " + ", ".join(f"{kinds[k]} {k}" for k in (NUMBER, TEXT, LIST)))
    output, digest = write_tasks(args.seed, tasks)
    print(f"wrote {short_path(output)}")
    print(f"md5   {digest}")


if __name__ == "__main__":
    main()
