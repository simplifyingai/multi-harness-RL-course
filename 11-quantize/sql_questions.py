"""The SQL questions compare.py asks every build, and how a reply is graded.

Three fixed sets, the same for every build:

- **24 held-out questions** for the pass rate: the project's own evaluation
  set, 24 of the 50 test questions picked round-robin over (database,
  difficulty) exactly as the project's ``mhrl/cli.py::balanced`` does (a plain
  first 24 would be one database's easy questions). These are single-turn: the
  model sees the database's tables and the question, and gets one reply to
  write one SQL query whose result is the answer alone, one column. The
  query runs on a read-only copy of the database, its result becomes an
  answer as chapter 1's ``make_tasks.py`` formats one (from the first
  column), and chapter 1's grader judges it.
- **20 training questions** for agreement, asked the way chapter 10 asked its
  question: "<question> Answer with one SQL query."
- **The gold SQL of those 20** as the texts whose log-probability is measured.

This module needs only the standard library.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grader import check_answer
from tasks import DATA_DIR, LIST, NUMBER, TEST_FILE, TRAIN_FILE, Task, load_tasks

QUERY_ONLY = "PRAGMA query_only = ON"  # no statement may write, to any database
REFUSED_ACTIONS = (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)

PASS_RATE_QUESTIONS = 24  # the project evaluates 24 held-out tasks per harness
AGREEMENT_QUESTIONS = 20
AGREEMENT_SUFFIX = " Answer with one SQL query."  # chapter 10's wording
SCHEMA_QUERY = "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
# Like chapter 1's instruction ("write ONLY the answer"), the prompt asks for
# the answer alone: the answer is read from the result's first column, as
# chapter 1's make_tasks.py formats gold answers, so a query that lists an id
# first would be graded on the id.
SINGLE_TURN_PROMPT = (
    "A SQLite database has these tables:\n\n{schema}\n\n{question}\n"
    "Answer with one SQL query whose result is the answer and nothing else: one "
    "column, one row per item if it is a list. Put it in a ```sql code block."
)
THINK_END = "</think>"
SQL_BLOCK_RE = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_BLOCK_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)
QUERY_SECONDS = 10.0  # a query running longer is stopped
PROGRESS_EVERY = 10_000  # SQLite virtual-machine steps between time checks
STOP_QUERY = 1  # what a progress handler returns to interrupt the query
MAX_ROWS = 1_000  # rows read from a result; the answer needs the first few
DECIMALS = 2  # chapter 1's make_tasks.py writes float answers to 2 places

REASON_NO_SQL = "no SQL in the reply"
REASON_NO_ROWS = "the query returned nothing"
REASON_CUT_OFF = "cut off by the length cap while still reasoning"


@dataclass(frozen=True)
class Graded:
    """The verdict on one reply.

    Attributes:
        correct: Whether the query's answer matches.
        sql: The query found in the reply, or None.
        answer: The answer its result gave, or None.
        reason: ``"correct"`` or why the reply was wrong.
    """

    correct: bool
    sql: str | None
    answer: str | None
    reason: str


class QueryError(Exception):
    """Raised when a model's query fails or runs too long."""


def balanced(tasks: list[Task], limit: int) -> list[Task]:
    """Take up to ``limit`` tasks, round-robin over (database, difficulty).

    A copy of the project's ``mhrl/cli.py::balanced``.

    Args:
        tasks: Candidate tasks, in file order.
        limit: How many to keep.

    Returns:
        A balanced subset, the same for the same input order.
    """
    if limit >= len(tasks):
        return list(tasks)
    buckets: dict[tuple[str, str], list[Task]] = {}
    for task in tasks:
        buckets.setdefault((task.domain, task.difficulty), []).append(task)
    queues = [list(buckets[key]) for key in sorted(buckets)]
    picked: list[Task] = []
    while len(picked) < limit:
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def pass_rate_tasks(folder: Path = DATA_DIR) -> list[Task]:
    """Pick the 24 held-out questions the project evaluates on.

    Args:
        folder: Where the task files are.

    Returns:
        The 24 tasks.
    """
    return balanced(load_tasks(TEST_FILE, folder), PASS_RATE_QUESTIONS)


def agreement_tasks(folder: Path = DATA_DIR) -> list[Task]:
    """Pick the 20 training questions used for agreement and log-probability.

    Args:
        folder: Where the task files are.

    Returns:
        The 20 tasks.
    """
    return balanced(load_tasks(TRAIN_FILE, folder), AGREEMENT_QUESTIONS)


def agreement_question(task: Task) -> str:
    """Ask a question the way chapter 10 asked its first one.

    Args:
        task: The task.

    Returns:
        Such as ``"How many customers live in Paris? Answer with one SQL query."``.
    """
    return task.question + AGREEMENT_SUFFIX


def database_path(task: Task, folder: Path = DATA_DIR) -> Path:
    """Return where a task's database is bundled.

    Args:
        task: The task.
        folder: The data folder.

    Returns:
        The database file.
    """
    return folder / task.db_file


def refuse_attach(action: int, *_: object) -> int:
    """Tell SQLite to refuse ATTACH and DETACH, and allow everything else.

    ``mode=ro`` guards only the database opened: a model-written
    ``ATTACH 'x.db' AS x`` would otherwise create a file wherever the process
    can write (found by chapter 14's code review).

    Args:
        action: The kind of operation SQLite is about to prepare.
        *_: The operation's details (names), not needed here.

    Returns:
        ``sqlite3.SQLITE_DENY`` for ATTACH and DETACH, else ``SQLITE_OK``.
    """
    return sqlite3.SQLITE_DENY if action in REFUSED_ACTIONS else sqlite3.SQLITE_OK


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a database so that no query can change it, or any other file.

    Args:
        path: The database file.

    Returns:
        The connection.

    Raises:
        sqlite3.Error: If it cannot be opened.
    """
    # A file: URI, escaped, so a folder name with # or ? cannot cut it short.
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.execute(QUERY_ONLY)
    connection.set_authorizer(refuse_attach)
    return connection


def schema_text(path: Path) -> str:
    """Read a database's tables as the CREATE TABLE statements that made them.

    Args:
        path: The database file.

    Returns:
        One statement per line, ending with a semicolon.
    """
    connection = connect_read_only(path)
    try:
        rows = connection.execute(SCHEMA_QUERY).fetchall()
    finally:
        connection.close()
    return "\n".join(f"{row[0]};" for row in rows)


def single_turn_prompt(task: Task, folder: Path = DATA_DIR) -> str:
    """Write the one message a build gets for a pass-rate question.

    Args:
        task: The task.
        folder: The data folder.

    Returns:
        The tables, the question and what to reply with.
    """
    schema = schema_text(database_path(task, folder))
    return SINGLE_TURN_PROMPT.format(schema=schema, question=task.question)


def extract_sql(reply: str) -> str | None:
    """Find the query in a reply: the last ```sql block after the reasoning.

    Args:
        reply: The model's whole reply, reasoning included.

    Returns:
        The query, or None if the reply holds no code block.
    """
    answer = reply.rsplit(THINK_END, 1)[-1]
    for pattern in (SQL_BLOCK_RE, ANY_BLOCK_RE):
        blocks = pattern.findall(answer)
        if blocks:
            return blocks[-1].strip() or None
    return None


def run_query(path: Path, sql: str) -> list[tuple[Any, ...]]:
    """Run a query on a read-only database, stopping it if it runs too long.

    Args:
        path: The database file.
        sql: The query.

    Returns:
        Up to MAX_ROWS rows of its result.

    Raises:
        QueryError: If SQLite rejects the query or it runs out of time.
    """
    deadline = time.monotonic() + QUERY_SECONDS

    def out_of_time() -> int:
        """Tell SQLite to stop once the deadline passes.

        Returns:
            STOP_QUERY after the deadline, else 0.
        """
        return STOP_QUERY if time.monotonic() > deadline else 0

    try:
        connection = connect_read_only(path)
    except sqlite3.Error as error:
        raise QueryError(f"could not open {path.name}: {error}") from error
    try:
        connection.set_progress_handler(out_of_time, PROGRESS_EVERY)
        return connection.execute(sql).fetchmany(MAX_ROWS)
    except sqlite3.Error as error:
        raise QueryError(f"SQL error: {error}") from error
    finally:
        connection.close()


def format_answer(rows: list[tuple[Any, ...]], answer_type: str) -> str | None:
    """Turn a query result into answer text, as chapter 1's make_tasks.py does.

    Args:
        rows: The rows the query returned.
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


def grade_reply(
    task: Task, reply: str, cut_off: bool = False, folder: Path = DATA_DIR
) -> Graded:
    """Find the query in a reply, run it, and let chapter 1's grader judge it.

    Args:
        task: The task the reply answers.
        reply: The model's whole reply.
        cut_off: True if the length cap ended the reply. A reply cut off
            before its reasoning ended has no answer yet: a query drafted
            inside the reasoning is not taken as one.
        folder: The data folder.

    Returns:
        The verdict.
    """
    if cut_off and THINK_END not in reply:
        return Graded(False, None, None, REASON_CUT_OFF)
    sql = extract_sql(reply)
    if sql is None:
        return Graded(False, None, None, REASON_NO_SQL)
    try:
        rows = run_query(database_path(task, folder), sql)
    except QueryError as error:
        return Graded(False, sql, None, str(error))
    answer = format_answer(rows, task.answer_type)
    if answer is None:
        return Graded(False, sql, None, REASON_NO_ROWS)
    grade = check_answer(task, answer)
    return Graded(grade.correct, sql, answer, grade.reason)
