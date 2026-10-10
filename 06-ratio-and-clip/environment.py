"""The environment: chapter 4's MiniAnalyst, with one change: LOOK shows one table.

An RL environment is everything the agent acts in. This one has chapter 4's
parts:

* tasks: the questions in ``questions.py``, easy, medium and hard;
* actions: one per call, LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE or FINISH;
* observations: what each call sends back (a table, a result, an error);
* state: the folder with its data.db and answer.txt, plus the calls used, how
  many tables have been seen, the result in hand and whether an answer has
  been written;
* reward rule: chapter 1's grader on answer.txt, 1 or 0, given once, at the end;
* stop rule: FINISH, or the budget of 8 calls is used up.

The one change from chapter 4: there, one LOOK showed every table. Here each
LOOK shows one more table and its columns, following the links between them:
customers, then the orders they placed, then the items in those orders, then
the products. A query can only use tables the agent has seen, so a question
that needs more tables needs more looking first. The toy models that by
trying the query on a database holding only the tables seen so far (empty
copies), so the error is SQLite's own: ``no such table: orders``.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from grader import REASON_CORRECT, Grade, grade
from questions import SHOP_DB, TEMPLATES, Question, candidate_sql, run_query
from tasks import ANSWER_FILE, DATABASE_FILE, CourseError

# The actions, one per call. Each QUERY runs one of the question's three
# candidate queries: QUERY_A the right one, QUERY_B and QUERY_C two mistakes.
LOOK = "LOOK"
QUERY_A = "QUERY_A"
QUERY_B = "QUERY_B"
QUERY_C = "QUERY_C"
WRITE = "WRITE"
FINISH = "FINISH"
ACTIONS = (LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE, FINISH)
QUERIES = {QUERY_A: 0, QUERY_B: 1, QUERY_C: 2}  # the candidate each one runs

BUDGET = 8  # calls per rollout, like the project's 8-call budget
FINISHED = "FINISH"  # why a rollout stopped: the agent said so...
BUDGET_USED = "budget used"  # ...or it ran out of calls

# The order LOOK shows the tables in: each one links to the one before it
# (orders.customer_id, order_items.order_id, order_items.product_id).
LINKED_TABLES = ("customers", "orders", "order_items", "products")
KNOWN_TABLES = ":memory:"  # where a query is tried first: only the tables seen
COLUMNS_SQL = "SELECT name FROM pragma_table_info(?) ORDER BY cid"
SCHEMA_SQL = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?"
ALL_SEEN = "every table seen already"
NOTHING = "(nothing)"  # a query that returned no value
NOTHING_TO_WRITE = "nothing to write yet"
NOT_GRADED = "not graded yet"


class RolloutOver(CourseError):
    """Raised when an action is sent after the rollout has stopped."""


class UnknownAction(CourseError):
    """Raised when an action is not one this environment offers."""


class NotGraded(CourseError):
    """Raised when a rollout that has stopped has no verdict (a bug)."""


@dataclass
class State:
    """Everything about a rollout so far, apart from the files in its folder.

    Attributes:
        question: The question being answered.
        calls: How many calls have been made.
        tables_seen: How many tables LOOK has shown (0 to 4).
        result: The last query result in hand, or None before any result.
        written: Whether an answer has been written to answer.txt.
        stop: Why the rollout stopped, or None while it is running.
    """

    question: Question
    calls: int = 0
    tables_seen: int = 0
    result: str | None = None
    written: bool = False
    stop: str | None = None


@dataclass(frozen=True)
class Call:
    """One action and what came back.

    Attributes:
        number: The call's number, from 1.
        action: The action, e.g. ``"LOOK"``.
        observation: What the environment sent back.
    """

    number: int
    action: str
    observation: str


@dataclass
class MiniAnalyst:
    """One rollout: a question, its own folder, and the rules of the game.

    Attributes:
        question: The question being answered.
        folder: This rollout's folder, holding data.db (a copy of shop.db).
        state: The calls used, what has been seen, held and written.
        calls: Every call so far, in order.
        grade: The grader's verdict, or None until the rollout stops.
    """

    question: Question
    folder: Path
    state: State = field(init=False)
    calls: list[Call] = field(default_factory=list, init=False)
    grade: Grade | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Start with nothing seen, held or written, and no calls used."""
        self.state = State(self.question)

    @property
    def done(self) -> bool:
        """Whether the rollout has stopped (FINISH, or the budget is used up)."""
        return self.state.stop is not None

    @property
    def reward(self) -> int | None:
        """The reward: None while the rollout runs, then the grader's 1 or 0."""
        return None if self.grade is None else self.grade.reward

    def step(self, action: str) -> str:
        """Take one call: do the action and send back what it shows.

        Args:
            action: One of ``ACTIONS``.

        Returns:
            The observation. For FINISH it is the grader's verdict.

        Raises:
            RolloutOver: If the rollout has already stopped.
            UnknownAction: If the action is not on offer.
            OSError: If answer.txt cannot be written.
            sqlite3.Error: If a query fails for a reason other than a table
                not seen yet.
        """
        if self.done:
            raise RolloutOver(f"the rollout stopped ({self.state.stop})")
        if action not in ACTIONS:
            raise UnknownAction(f"unknown action {action!r}")
        number = self.state.calls + 1
        if action == FINISH:
            self.state.calls = number
            self._stop(FINISHED)
            observation = verdict_text(self.grade)
        else:
            observation = self._act(action)
            self.state.calls = number  # counted once the action has worked
            if number >= BUDGET:
                self._stop(BUDGET_USED)
        self.calls.append(Call(number, action, observation))
        return observation

    def _act(self, action: str) -> str:
        """Do one action other than FINISH.

        Args:
            action: LOOK, a QUERY or WRITE.

        Returns:
            The observation.

        Raises:
            OSError: If answer.txt cannot be written.
            sqlite3.Error: If a query fails for a reason other than a table
                not seen yet.
        """
        if action == LOOK:
            return self._look()
        if action in QUERIES:
            return self._query(QUERIES[action])
        # Only WRITE is left.
        if self.state.result is None:
            return NOTHING_TO_WRITE
        return self._write(self.state.result)

    def _look(self) -> str:
        """Show the next table and its columns, following the links.

        Returns:
            For example ``"orders: id, customer_id, order_date, payment_method"``,
            or a note that every table has been seen.

        Raises:
            sqlite3.Error: If the database cannot be read.
        """
        if self.state.tables_seen == len(LINKED_TABLES):
            return ALL_SEEN
        table = LINKED_TABLES[self.state.tables_seen]
        self.state.tables_seen += 1
        columns = column_names(self.folder / DATABASE_FILE, table)
        return f"{table}: " + ", ".join(columns)

    def _query(self, choice: int) -> str:
        """Run one of the question's candidate queries.

        Args:
            choice: Which candidate (0 for QUERY_A).

        Returns:
            The result, or SQLite's error when the query uses a table the
            agent has not seen yet.

        Raises:
            sqlite3.Error: If the query fails on the real database.
        """
        template = TEMPLATES[self.question.row]
        sql = candidate_sql(template, choice, self.question.city)
        seen = LINKED_TABLES[: self.state.tables_seen]
        if len(seen) < len(LINKED_TABLES):
            # The agent knows only some tables: try the query on empty copies
            # of those, and if it names any other table, SQLite says so.
            error = missing_table_error(self.folder / DATABASE_FILE, seen, sql)
            if error is not None:
                return error
        self.state.result = run_query(self.folder / DATABASE_FILE, sql)
        return self.state.result or NOTHING

    def _write(self, answer: str) -> str:
        """Write an answer to answer.txt, replacing any earlier one.

        Args:
            answer: The text to write.

        Returns:
            The observation, e.g. ``"answer.txt ← 13"``.

        Raises:
            OSError: If the file cannot be written.
        """
        (self.folder / ANSWER_FILE).write_text(answer + "\n", encoding="utf-8")
        self.state.written = True
        return f"{ANSWER_FILE} ← {answer or NOTHING}"

    def _stop(self, why: str) -> None:
        """Stop the rollout and grade answer.txt: the only reward it gets.

        Args:
            why: ``FINISHED`` or ``BUDGET_USED``.
        """
        self.state.stop = why
        self.grade = grade(self.question.task, self.folder)


def column_names(database: Path, table: str) -> list[str]:
    """List a table's columns, as LOOK shows them.

    Args:
        database: The SQLite file.
        table: The table's name.

    Returns:
        The column names in the table's own order.

    Raises:
        sqlite3.Error: If the file is not a database.
    """
    with closing(sqlite3.connect(database)) as db:
        return [row[0] for row in db.execute(COLUMNS_SQL, (table,))]


def tables_needed(sql: str) -> int:
    """Count how many LOOKs a query needs before it can run.

    LOOK shows the tables in a fixed order, so a query needs every table up
    to the last one it names.

    Args:
        sql: The query.

    Returns:
        1 for a query on customers alone, up to 4 for one that uses products.
    """
    named = [
        place
        for place, table in enumerate(LINKED_TABLES, start=1)
        if re.search(rf"\b{table}\b", sql)
    ]
    return max(named, default=0)


def missing_table_error(database: Path, seen: tuple[str, ...], sql: str) -> str | None:
    """Try a query on empty copies of the tables seen so far.

    Args:
        database: The real database, whose table definitions are copied.
        seen: The tables the agent has seen.
        sql: The query.

    Returns:
        SQLite's error from the empty copies, which is ``"no such table: ..."``
        when the query names a table not seen yet; None when it runs.

    Raises:
        sqlite3.Error: If a table definition cannot be read or copied.
    """
    with closing(sqlite3.connect(database)) as real:
        schemas = [real.execute(SCHEMA_SQL, (name,)).fetchone()[0] for name in seen]
    with closing(sqlite3.connect(KNOWN_TABLES)) as known:
        for schema in schemas:
            known.execute(schema)
        try:
            known.execute(sql)
        except sqlite3.OperationalError as error:
            return str(error)
    return None


def verdict_text(verdict: Grade | None) -> str:
    """Write the grader's verdict and reward in a few words.

    Args:
        verdict: The grade, once the rollout has stopped.

    Returns:
        For example ``"correct · reward 1"``.
    """
    if verdict is None:
        return NOT_GRADED
    said = REASON_CORRECT if verdict.correct else verdict.reason
    return f"{said} · reward {verdict.reward}"


@contextmanager
def fresh_folder(parent: Path, name: str) -> Iterator[Path]:
    """Make a new folder holding a copy of shop.db, and delete it afterwards.

    Every rollout gets its own folder, so two rollouts can never see each
    other's answer.txt, and nothing is left behind.

    Args:
        parent: Where the folder goes (one folder per run, under COURSE_HOME).
        name: The folder's name, e.g. ``"step-001-group-1-rollout-1"``.

    Yields:
        The folder, holding only data.db.

    Raises:
        FileExistsError: If the folder already exists.
        OSError: If the folder cannot be made or the database copied.
    """
    folder = parent / name
    folder.mkdir(parents=True)
    try:
        shutil.copyfile(SHOP_DB, folder / DATABASE_FILE)
        yield folder
    finally:
        shutil.rmtree(folder)
