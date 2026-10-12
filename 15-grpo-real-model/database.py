"""Read a bundled database: its tables, and the result of a query run on it.

Copied from chapter 11's ``sql_questions.py``, with one gap closed. Every
database is opened read only, so no query, the model's or the gold one, can
change it; ``mode=ro`` guards only the database opened, so the connection is
also made query-only and refuses ``ATTACH`` (which would otherwise open or
create another file wherever the process can write). A query running longer
than 10 seconds is stopped. Chapter 15 adds one more gap closed: text SQLite
cannot encode (a lone surrogate) is refused as a QueryError instead of
stopping the run. This module needs only the standard library.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA_QUERY = "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
QUERY_SECONDS = 10.0  # a query running longer is stopped
PROGRESS_EVERY = 10_000  # SQLite virtual-machine steps between time checks
STOP_QUERY = 1  # what a progress handler returns to interrupt the query
CONTINUE_QUERY = 0  # and to let it go on
MAX_ROWS = 1_000  # rows read from a result; the answer needs the first few
QUERY_ONLY = "PRAGMA query_only = ON"  # no statement may write, to any database
REFUSED_ACTIONS = (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH)


class QueryError(Exception):
    """Raised when a query fails or runs too long."""


def refuse_attach(action: int, *_: object) -> int:
    """Tell SQLite to refuse ATTACH and DETACH, and allow everything else.

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
    connection.set_authorizer(refuse_attach)  # mode=ro guards only this file
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
            STOP_QUERY after the deadline, else CONTINUE_QUERY.
        """
        return STOP_QUERY if time.monotonic() > deadline else CONTINUE_QUERY

    try:
        connection = connect_read_only(path)
    except sqlite3.Error as error:
        raise QueryError(f"could not open {path.name}: {error}") from error
    try:
        connection.set_progress_handler(out_of_time, PROGRESS_EVERY)
        return connection.execute(sql).fetchmany(MAX_ROWS)
    except (sqlite3.Error, ValueError) as error:  # ValueError: text it cannot encode
        raise QueryError(f"SQL error: {error}") from error
    finally:
        connection.close()
