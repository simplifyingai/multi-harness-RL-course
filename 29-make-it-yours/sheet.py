"""The spreadsheet a new task family asks about, read the same way every time.

``data/sales.csv`` is an invented spreadsheet: 96 orders of a small shop,
one row each, with its date, region, sales rep, product, category, units
and unit price. A CSV file has no types: every value in it is text. So the
gold answers are computed from a copy loaded into an in-memory SQLite table
whose columns are declared (``units`` an integer, ``unit_price`` a real
number), because as text ``'10' < '9'`` is true and a sum of prices would
be a sum of strings. The scripted engine's solving command loads the file
the same way (``gold_command``), so it prints exactly the gold answer.

Only Python's standard library is needed (and, for ``gold_command``, the
``sqlite3`` program macOS ships).
"""

from __future__ import annotations

import csv
import shlex
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEET = HERE / "data" / "sales.csv"
TABLE = "sales"
# Each column of the spreadsheet and the type its values are read as.
COLUMNS = (
    ("order_id", "INTEGER"),
    ("date", "TEXT"),
    ("region", "TEXT"),
    ("rep", "TEXT"),
    ("product", "TEXT"),
    ("category", "TEXT"),
    ("units", "INTEGER"),
    ("unit_price", "REAL"),
)
# What each number column's text must read as.
NUMBER_TYPES = {"INTEGER": int, "REAL": float}
CREATE_TABLE = (
    f"CREATE TABLE {TABLE} ("
    + ", ".join(f"{name} {kind}" for name, kind in COLUMNS)
    + ")"
)


class SheetError(Exception):
    """Raised when the spreadsheet's header is not the columns this family reads."""


def read_rows(path: Path = SHEET) -> list[dict[str, str]]:
    """Read the spreadsheet as text, one dictionary per row.

    Args:
        path: The CSV file.

    Returns:
        Its rows, every value still text.

    Raises:
        SheetError: If its header row is not ``COLUMNS``' names, in order, or
            a row has a missing or extra field, an empty value, or a value
            that is not a number in a number column (the message names the
            line): a gold answer must never come from a half-read sheet.
        OSError: If the file cannot be read.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [name for name, _ in COLUMNS]
        if reader.fieldnames != expected:
            raise SheetError(f"{path.name}: header {reader.fieldnames}, not {expected}")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        if None in row or any(row[name] in (None, "") for name in expected):
            raise SheetError(f"{path.name}:{line}: a missing, extra or empty field")
        for name, kind in COLUMNS:
            if kind in NUMBER_TYPES:
                try:
                    NUMBER_TYPES[kind](row[name])
                except ValueError as error:
                    raise SheetError(
                        f"{path.name}:{line}: {name} {row[name]!r} is not a number"
                    ) from error
    return rows


def open_sheet(path: Path = SHEET) -> sqlite3.Connection:
    """Load the spreadsheet into an in-memory table with typed columns.

    Args:
        path: The CSV file.

    Returns:
        A connection holding one table, ``sales``.

    Raises:
        SheetError: If the header is not the expected columns.
        OSError: If the file cannot be read.
    """
    rows = read_rows(path)
    connection = sqlite3.connect(":memory:")
    connection.execute(CREATE_TABLE)
    names = [name for name, _ in COLUMNS]
    slots = ", ".join("?" for _ in names)
    # SQLite converts each text value to its column's declared type.
    connection.executemany(
        f"INSERT INTO {TABLE} VALUES ({slots})",
        [[row[name] for name in names] for row in rows],
    )
    return connection


def gold_command(gold_sql: str, data_file: str) -> str:
    """The shell command that loads a CSV file as typed columns and runs a query.

    It is what the scripted engine's "solve" script types, so a scripted run
    computes its answer from the task folder's file, as an agent would.

    Args:
        gold_sql: The query.
        data_file: The CSV file's name in the task folder.

    Returns:
        A ``sqlite3`` command line: an empty in-memory database, the typed
        table, the file imported into it (header skipped), then the query.
    """
    # Quoted, so a file name with a space stays one argument of the dot-command.
    load = f'.import --csv --skip 1 "{data_file}" {TABLE}'
    return shlex.join(
        ["sqlite3", ":memory:", "-cmd", CREATE_TABLE, "-cmd", load, gold_sql]
    )
