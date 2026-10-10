"""The environment: one question, a fresh folder, six actions and an 8-call budget.

An RL environment is everything the agent acts in. This one has the parts the
project's real environment has:

* tasks: the questions in ``questions.py``, split by city;
* actions: one per call, LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE or FINISH;
* observations: what each call sends back (table names, a result, an error);
* state: the folder with its data.db and answer.txt, plus the calls used,
  whether the tables have been seen, the result in hand and whether an answer
  has been written;
* reward rule: chapter 1's grader on answer.txt, 1 or 0, given once, at the end;
* stop rule: FINISH, or the budget of 8 calls is used up.

Until the agent has looked, it does not know what the tables are called. The
toy models that by sending a query made before LOOK to an empty database, so
the error is SQLite's own: ``no such table: customers``.

With ``memory`` (the ``--memorise`` agent) there is a seventh action, RECALL:
write the answer remembered from an earlier rollout of the same question.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from grader import REASON_CORRECT, Grade, grade
from questions import SHOP_DB, TEMPLATES, Question, candidate_sql, run_query
from tasks import ANSWER_FILE, DATABASE_FILE, CourseError

# The actions, one per call. Each QUERY runs one of the question's three
# candidate queries (chapter 2's, minus each question's last mistake).
LOOK = "LOOK"
QUERY_A = "QUERY_A"
QUERY_B = "QUERY_B"
QUERY_C = "QUERY_C"
WRITE = "WRITE"
FINISH = "FINISH"
RECALL = "RECALL"  # only for the --memorise agent
ACTIONS = (LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE, FINISH)
MEMORY_ACTIONS = (*ACTIONS, RECALL)
QUERIES = {QUERY_A: 0, QUERY_B: 1, QUERY_C: 2}  # the candidate each one runs

BUDGET = 8  # calls per rollout, like the project's 8-call budget
FINISHED = "FINISH"  # why a rollout stopped: the agent said so...
BUDGET_USED = "budget used"  # ...or it ran out of calls

EMPTY_DATABASE = ":memory:"  # where a query goes before the tables are known
TABLES_SQL = "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
TABLES = "tables: "  # what LOOK's observation starts with
NOTHING = "(nothing)"  # a query that returned no value
NOTHING_TO_WRITE = "nothing to write yet"
NOTHING_REMEMBERED = "nothing remembered for this question"
REMEMBERED = " (remembered)"
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
        schema_seen: Whether LOOK has shown the table names.
        result: The last query result in hand, or None before any result.
        written: Whether an answer has been written to answer.txt.
        stop: Why the rollout stopped, or None while it is running.
    """

    question: Question
    calls: int = 0
    schema_seen: bool = False
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
        memory: Answers remembered per question id, or None for an agent
            without RECALL.
        state: The calls used, what has been seen, held and written.
        calls: Every call so far, in order.
        grade: The grader's verdict, or None until the rollout stops.
    """

    question: Question
    folder: Path
    memory: Mapping[str, str] | None = None
    state: State = field(init=False)
    calls: list[Call] = field(default_factory=list, init=False)
    grade: Grade | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Start with nothing seen, held or written, and no calls used."""
        self.state = State(self.question)

    @property
    def actions(self) -> tuple[str, ...]:
        """The actions this agent may take: six, or seven with a memory."""
        return ACTIONS if self.memory is None else MEMORY_ACTIONS

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
            action: One of ``self.actions``.

        Returns:
            The observation. For FINISH it is the grader's verdict.

        Raises:
            RolloutOver: If the rollout has already stopped.
            UnknownAction: If the action is not on offer.
            OSError: If answer.txt cannot be written.
            sqlite3.Error: If a query fails for a reason other than a missing
                table.
        """
        if self.done:
            raise RolloutOver(f"the rollout stopped ({self.state.stop})")
        if action not in self.actions:
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
            action: LOOK, a QUERY, WRITE or RECALL.

        Returns:
            The observation.

        Raises:
            OSError: If answer.txt cannot be written.
            sqlite3.Error: If a query fails for a reason other than a missing
                table.
        """
        if action == LOOK:
            self.state.schema_seen = True
            return TABLES + ", ".join(table_names(self.folder / DATABASE_FILE))
        if action in QUERIES:
            return self._query(QUERIES[action])
        if action == WRITE:
            if self.state.result is None:
                return NOTHING_TO_WRITE
            return self._write(self.state.result)
        # Only RECALL is left, and only an agent with a memory is offered it.
        memory = self.memory or {}
        remembered = memory.get(self.question.task.task_id)
        if remembered is None:
            return NOTHING_REMEMBERED
        return self._write(remembered) + REMEMBERED

    def _query(self, choice: int) -> str:
        """Run one of the question's candidate queries.

        Args:
            choice: Which candidate (0 for QUERY_A).

        Returns:
            The result, or SQLite's error when the tables are not known yet.

        Raises:
            sqlite3.Error: If the query fails on the real database.
        """
        template = TEMPLATES[self.question.row]
        sql = candidate_sql(template, choice, self.question.city)
        if not self.state.schema_seen:
            # The agent has not looked, so it does not know the tables: the
            # query goes to an empty database and SQLite says so.
            with closing(sqlite3.connect(EMPTY_DATABASE)) as empty:
                try:
                    empty.execute(sql)
                except sqlite3.OperationalError as error:
                    return str(error)
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


def table_names(database: Path) -> list[str]:
    """List a database's tables, as LOOK shows them.

    Args:
        database: The SQLite file.

    Returns:
        The table names in alphabetical order.

    Raises:
        sqlite3.Error: If the file is not a database.
    """
    with closing(sqlite3.connect(database)) as db:
        return [row[0] for row in db.execute(TABLES_SQL)]


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
        name: The folder's name, e.g. ``"rollout-0001"``.

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
