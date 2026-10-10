"""The environment: chapter 6's MiniAnalyst, with replies that have a length.

An RL environment is everything the agent acts in. This one keeps chapter 6's
parts, with these changes:

* tasks: the questions in ``questions.py``, easy, medium and hard;
* actions: LOOK, QUERY_A, QUERY_B and QUERY_C as before; SUBMIT, which writes
  the result in hand to answer.txt and hands it in (chapter 6's WRITE and
  FINISH in one action); FINISH, which stops without handing anything in;
  and SUBMIT_GUESS, which writes a guess (0) and hands it in at once,
  without looking or querying;
* observations: what each action sends back (a table, a result, an error);
* state: the folder with its data.db and answer.txt, plus the calls used, how
  many tables have been seen and the result in hand;
* reward rule: none here. The environment only says what happened (right or
  wrong, handed in or not, how many calls, how many tokens); ``rewards.py``
  turns that into a reward, five different ways;
* stop rule: SUBMIT, SUBMIT_GUESS or FINISH; a reply that is cut off; or the
  budget of 8 calls used up. Only SUBMIT and SUBMIT_GUESS hand in an answer.

New: every call is a reply with a length in tokens. A real model writes text,
and its tool call sits somewhere in that text. Here a reply of up to 79
tokens holds one action. A reply of 80 tokens or more holds two, as a model
may run a query and hand in its result in one message, so two actions cost
one call. A reply longer than the output cap of 120 tokens is cut off at 120,
before it gets to its actions (the project's engine does the same at 1,024).
A reply with no action ends the rollout, the way Claude Code takes a reply
with no tool call as the agent's last word.

One more thing can happen to a rollout, with ``--crashes``: its sandbox dies
before the grader can read answer.txt. Whatever the agent did, there is no
verdict.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from grader import REASON_CORRECT, Grade, grade
from questions import ALL_TEMPLATES, SHOP_DB, Question, candidate_sql, run_query
from tasks import ANSWER_FILE, DATABASE_FILE, CourseError

# The actions. Each QUERY runs one of the question's three candidate queries:
# QUERY_A the right one, QUERY_B and QUERY_C two mistakes.
LOOK = "LOOK"
QUERY_A = "QUERY_A"
QUERY_B = "QUERY_B"
QUERY_C = "QUERY_C"
SUBMIT = "SUBMIT"
FINISH = "FINISH"
SUBMIT_GUESS = "SUBMIT_GUESS"
ACTIONS = (LOOK, QUERY_A, QUERY_B, QUERY_C, SUBMIT, FINISH, SUBMIT_GUESS)
QUERIES = {QUERY_A: 0, QUERY_B: 1, QUERY_C: 2}  # the candidate each one runs

BUDGET = 8  # calls per rollout, like the project's 8-call budget
# How long a reply may be, in tokens.
LONG_REPLY = 80  # a reply this long or longer holds two actions...
ACTIONS_PER_LONG_REPLY = 2
OUTPUT_CAP = 120  # ...and one longer than this is cut off before its actions

SUBMITTED = "SUBMIT"  # why a rollout stopped: it handed in its result,
GUESSED = "SUBMIT_GUESS"  # or handed in a guess,
FINISHED = "FINISH"  # or stopped without an answer,
CUT = "cut off"  # or its reply was cut off before any action,
BUDGET_USED = "budget used"  # or it ran out of calls,
CRASHED = "sandbox died"  # or its sandbox died before grading

# The order LOOK shows the tables in: each one links to the one before it
# (orders.customer_id, order_items.order_id, order_items.product_id).
LINKED_TABLES = ("customers", "orders", "order_items", "products")
KNOWN_TABLES = ":memory:"  # where a query is tried first: only the tables seen
COLUMNS_SQL = "SELECT name FROM pragma_table_info(?) ORDER BY cid"
SCHEMA_SQL = "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?"
ALL_SEEN = "every table seen already"
NOTHING = "(nothing)"  # a query that returned no value
NOTHING_TO_SUBMIT = "nothing to submit yet"
NOTHING_HANDED_IN = "stopped, nothing handed in"
VERDICT_JOIN = " · "
NOT_GRADED = "not graded"
CUT_OFF = f"cut off at {OUTPUT_CAP} tokens, before its action"


class RolloutOver(CourseError):
    """Raised when a call is made after the rollout has stopped."""


class UnknownAction(CourseError):
    """Raised when an action is not one this environment offers."""


class CallNotOpen(CourseError):
    """Raised when an action is sent outside a call, or too many in one call."""


class NotGraded(CourseError):
    """Raised when a rollout that has stopped has no verdict (a bug)."""


@dataclass
class State:
    """Everything about a rollout so far, apart from the files in its folder.

    Attributes:
        question: The question being answered.
        calls: How many calls have been made (finished ones).
        tables_seen: How many tables LOOK has shown (0 to 4).
        result: The last query result in hand, or None before any result.
        written: Whether answer.txt has been written. SUBMIT writes it and
            ends the rollout at once, so the agent never sees this as 1; it
            stays because chapter 6's policy reads it.
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
    """One call: a reply of some length, and the actions it held.

    Attributes:
        number: The call's number, from 1.
        tokens: How many tokens the reply had (at most the output cap).
        actions: The actions it held, in order (none when it was cut off).
        observations: What each action sent back.
        cut_off: Whether the reply passed the cap and was cut off.
    """

    number: int
    tokens: int
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    cut_off: bool

    @property
    def long(self) -> bool:
        """Whether the reply was long enough to hold two actions, and held them."""
        return len(self.actions) == ACTIONS_PER_LONG_REPLY


@dataclass
class MiniAnalyst:
    """One rollout: a question, its own folder, and the rules of the game.

    A call goes: ``open_call(tokens)`` says how many actions the reply holds
    (0 when it is cut off), ``act`` runs each of them, ``close_call`` counts
    the call.

    Attributes:
        question: The question being answered.
        folder: This rollout's folder, holding data.db (a copy of shop.db).
        guess: What SUBMIT_GUESS hands in.
        state: The calls used, what has been seen and held.
        calls: Every finished call, in order.
        grade: The grader's verdict, or None until the rollout stops (and
            for good when the sandbox died).
    """

    question: Question
    folder: Path
    guess: str
    state: State = field(init=False)
    calls: list[Call] = field(default_factory=list, init=False)
    grade: Grade | None = field(default=None, init=False)
    _open: bool = field(default=False, init=False)
    _room: int = field(default=0, init=False)  # actions the open call can hold
    _tokens: int = field(default=0, init=False)
    _actions: list[str] = field(default_factory=list, init=False)
    _observations: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Start with nothing seen or held, and no calls used."""
        self.state = State(self.question)

    @property
    def done(self) -> bool:
        """Whether the rollout has stopped."""
        return self.state.stop is not None

    @property
    def tokens(self) -> int:
        """Every token the rollout's replies held, added up."""
        return sum(call.tokens for call in self.calls)

    def open_call(self, tokens: int) -> int:
        """Start the next call with a reply of ``tokens`` tokens.

        Args:
            tokens: The reply's length.

        Returns:
            How many actions the reply holds: 1, 2 for a long reply, or 0 when
            it passed the output cap and was cut off.

        Raises:
            RolloutOver: If the rollout has already stopped.
            CallNotOpen: If the previous call was not closed.
        """
        if self.done:
            raise RolloutOver(f"the rollout stopped ({self.state.stop})")
        if self._open:
            raise CallNotOpen("the previous call is still open")
        self._open = True
        self._tokens = min(tokens, OUTPUT_CAP)
        if tokens > OUTPUT_CAP:
            self._room = 0
        elif tokens >= LONG_REPLY:
            self._room = ACTIONS_PER_LONG_REPLY
        else:
            self._room = 1
        return self._room

    def act(self, action: str) -> str:
        """Do one action inside the open call and send back what it shows.

        Args:
            action: One of ``ACTIONS``.

        Returns:
            The observation. When the action ends the rollout, the verdict
            follows it.

        Raises:
            RolloutOver: If the rollout has already stopped.
            CallNotOpen: If no call is open, or it holds no more actions.
            UnknownAction: If the action is not on offer.
            OSError: If answer.txt cannot be written.
            sqlite3.Error: If a query fails for a reason other than a table
                not seen yet.
        """
        if self.done:
            raise RolloutOver(f"the rollout stopped ({self.state.stop})")
        if not self._open or len(self._actions) >= self._room:
            raise CallNotOpen("no open call has room for another action")
        if action not in ACTIONS:
            raise UnknownAction(f"unknown action {action!r}")
        observation = self._act(action)
        if self.done:
            observation = f"{observation}{VERDICT_JOIN}{verdict_text(self.grade)}"
        self._actions.append(action)
        self._observations.append(observation)
        return observation

    def close_call(self) -> Call:
        """Finish the open call: count it, and stop the rollout if it must.

        A reply that held no action (it was cut off) ends the rollout, and so
        does the last call of the budget.

        Returns:
            The finished call.

        Raises:
            CallNotOpen: If no call is open.
        """
        if not self._open:
            raise CallNotOpen("no call is open")
        number = self.state.calls + 1
        cut_off = self._room == 0
        call = Call(
            number,
            self._tokens,
            tuple(self._actions),
            (CUT_OFF,) if cut_off else tuple(self._observations),
            cut_off,
        )
        self.calls.append(call)
        self.state.calls = number
        self._open, self._room, self._actions, self._observations = False, 0, [], []
        if not self.done and cut_off:
            self._stop(CUT)
        elif not self.done and number >= BUDGET:
            self._stop(BUDGET_USED)
        return call

    def crash(self) -> None:
        """The sandbox dies: the rollout stops, and nothing is ever graded."""
        self.state.stop = CRASHED
        self.grade = None

    def _act(self, action: str) -> str:
        """Do one action.

        Args:
            action: One of ``ACTIONS``.

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
        if action == SUBMIT_GUESS:
            observation = self._write(self.guess)
            self._stop(GUESSED)
            return observation
        if action == FINISH:
            self._stop(FINISHED)
            return NOTHING_HANDED_IN
        # Only SUBMIT is left.
        if self.state.result is None:
            return NOTHING_TO_SUBMIT
        observation = self._write(self.state.result)
        self._stop(SUBMITTED)
        return observation

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
        template = ALL_TEMPLATES[self.question.row]
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
        """Write an answer to answer.txt.

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
        """Stop the rollout and grade answer.txt (missing unless handed in).

        Args:
            why: Why it stopped: ``SUBMITTED``, ``GUESSED``, ``FINISHED``,
                ``CUT`` or ``BUDGET_USED``.
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
    """Write the grader's verdict in a few words.

    The reward is not part of it: in this chapter the reward depends on the
    rule, so ``rewards.py`` works it out after the rollout.

    Args:
        verdict: The grade, once the rollout has stopped.

    Returns:
        For example ``"correct"``, or why the answer was wrong.
    """
    if verdict is None:
        return NOT_GRADED
    return REASON_CORRECT if verdict.correct else verdict.reason


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
