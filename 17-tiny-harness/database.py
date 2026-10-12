"""Read a task's database so that nothing the model types can change any file.

Copied from chapter 15's ``database.py``, which is chapter 11's guard with two
gaps closed. Every database is opened read only (``mode=ro``); that guards
only the file opened, so the connection is also made query-only and refuses
``ATTACH`` (which would otherwise open or create another file wherever the
process can write). A query running longer than 10 seconds is stopped, and
text SQLite cannot encode is refused as a ``QueryError``.

Chapter 17 runs every ``sqlite3`` command the model sends through this guard,
so it keeps what the sqlite3 program prints and the guard did not need: each
result's column names (for ``-header``), several statements in one command,
and the table names and ``CREATE`` statements behind ``.tables`` and
``.schema``. Because the model now types the SQL, the guard also refuses to
let a query switch ``query_only`` back off (with two other settings), and
caps one value at a million bytes. This module needs only the standard
library.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUERY_SECONDS = 10.0  # a query running longer is stopped
PROGRESS_EVERY = 10_000  # SQLite virtual-machine steps between time checks
STOP_QUERY = 1  # what a progress handler returns to interrupt the query
CONTINUE_QUERY = 0  # and to let it go on
MAX_ROWS = 1_000  # rows read from one result; the harness cuts long output anyway
QUERY_ONLY = "PRAGMA query_only = ON"  # no statement may write, to any database
REFUSED_ACTIONS = (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)
# Settings a query could otherwise change on its own connection: turning
# query_only off, moving temporary files, editing the schema table.
GUARDED_PRAGMAS = {"query_only", "temp_store_directory", "writable_schema"}
MAX_VALUE_BYTES = 1_000_000  # the largest text or blob one query may build
STATEMENT_END = ";"
# What sqlite3's .tables and .schema read: the tables, by name and in the
# order they were made (sqlite_ tables are SQLite's own and never shown).
TABLES_QUERY = (
    "SELECT name FROM sqlite_master WHERE type = 'table' "
    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
)
SCHEMA_QUERY = (
    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
    "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
)
SCHEMA_OF_QUERY = (
    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
    "AND (name = ? OR tbl_name = ?) ORDER BY rowid"
)


class QueryError(Exception):
    """Raised when a query fails or runs too long."""


@dataclass(frozen=True)
class Result:
    """What one statement returned.

    Attributes:
        columns: The result's column names (empty for a statement with no rows).
        rows: Up to MAX_ROWS rows.
    """

    columns: list[str]
    rows: list[tuple[Any, ...]]


def refuse_attach(action: int, name: str | None, value: str | None, *_: object) -> int:
    """Tell SQLite to refuse ATTACH, DETACH and changes to the guard's settings.

    ``VACUUM INTO`` writes a copy of the database through ATTACH, so refusing
    ATTACH refuses it too.

    Args:
        action: The kind of operation SQLite is about to prepare.
        name: Its first detail: for a PRAGMA, the setting's name.
        value: Its second detail: for a PRAGMA, the new value (None when the
            setting is only read).
        *_: The other details, not needed here.

    Returns:
        ``sqlite3.SQLITE_DENY`` for ATTACH, DETACH and a change to one of
        GUARDED_PRAGMAS, else ``SQLITE_OK``.
    """
    if action in REFUSED_ACTIONS:
        return sqlite3.SQLITE_DENY
    changes_guard = name is not None and name.lower() in GUARDED_PRAGMAS
    if action == sqlite3.SQLITE_PRAGMA and changes_guard and value is not None:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


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
    connection.set_authorizer(refuse_attach)  # mode=ro guards only this file
    # The default allows a 1 GB value: one query could fill the memory the
    # model needs. A million bytes is far more than any answer.
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    return connection


def split_statements(sql: str) -> list[str]:
    """Cut SQL text into whole statements, as the sqlite3 program runs them.

    A semicolon inside a quoted string does not end a statement:
    ``sqlite3.complete_statement`` says when the text so far is a whole one.

    Args:
        sql: One or more statements.

    Returns:
        The statements, each ending with a semicolon; empty ones are dropped.
    """
    statements, pending = [], ""
    for piece in sql.split(STATEMENT_END):
        pending += piece + STATEMENT_END
        if sqlite3.complete_statement(pending):
            if pending.strip(STATEMENT_END + " \t\r\n"):
                statements.append(pending.strip())
            pending = ""
    # Text left over never ended: the last statement had no semicolon (the
    # loop added one too many), or it opened a quote it never closed.
    leftover = pending[: -len(STATEMENT_END)].strip()
    if leftover:
        statements.append(leftover)
    return statements


def guarded(path: Path, work: Callable[[sqlite3.Connection], Any]) -> Any:
    """Open the database read only, do some work, and close it again.

    Args:
        path: The database file.
        work: What to do with the connection.

    Returns:
        What ``work`` returned.

    Raises:
        QueryError: If the database cannot be opened, SQLite rejects a
            statement, or the work runs longer than QUERY_SECONDS.
    """
    deadline = time.monotonic() + QUERY_SECONDS

    def out_of_time() -> int:
        """Tell SQLite to stop once the deadline passes.

        Returns:
            STOP_QUERY after the deadline, else CONTINUE_QUERY.
        """
        return STOP_QUERY if time.monotonic() > deadline else CONTINUE_QUERY

    try:
        connection = connect_read_only(path)
    except sqlite3.Error as error:
        raise QueryError(f"unable to open database {path.name!r}: {error}") from error
    try:
        connection.set_progress_handler(out_of_time, PROGRESS_EVERY)
        return work(connection)
    except (sqlite3.Error, ValueError) as error:  # ValueError: text it cannot encode
        raise QueryError(str(error)) from error
    finally:
        connection.close()


def run_statements(path: Path, sql: str) -> list[Result]:
    """Run each statement of some SQL text on a read-only database.

    Args:
        path: The database file.
        sql: One or more statements.

    Returns:
        One result per statement, in order.

    Raises:
        QueryError: At the first statement SQLite rejects (or that runs too
            long); the results before it are lost, as in a refused command.
    """

    def run(connection: sqlite3.Connection) -> list[Result]:
        """Run the statements one by one on the open connection.

        Args:
            connection: The guarded connection.

        Returns:
            One result per statement.
        """
        results = []
        for statement in split_statements(sql):
            cursor = connection.execute(statement)
            columns = [entry[0] for entry in cursor.description or []]
            results.append(Result(columns, cursor.fetchmany(MAX_ROWS)))
        return results

    return guarded(path, run)


def table_names(path: Path) -> list[str]:
    """List a database's tables by name, as sqlite3's ``.tables`` does.

    Args:
        path: The database file.

    Returns:
        The names, sorted.

    Raises:
        QueryError: If the database cannot be read.
    """
    return [row[0] for row in run_statements(path, TABLES_QUERY)[0].rows]


def schema_lines(path: Path, table: str | None = None) -> list[str]:
    """Read the CREATE statements behind a database, as ``.schema`` prints them.

    Args:
        path: The database file.
        table: Only this table's statements, or every one when None.

    Returns:
        One statement per item, ending with a semicolon.

    Raises:
        QueryError: If the database cannot be read.
    """

    def read(connection: sqlite3.Connection) -> list[str]:
        """Read the statements on the open connection.

        Args:
            connection: The guarded connection.

        Returns:
            The statements.
        """
        if table is None:
            rows = connection.execute(SCHEMA_QUERY).fetchall()
        else:
            rows = connection.execute(SCHEMA_OF_QUERY, (table, table)).fetchall()
        return [f"{row[0]};" for row in rows]

    return guarded(path, read)
