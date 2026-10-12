"""The one tool the harness offers: a 'bash' that runs three commands, safely.

The model may ask for any command; the harness decides what runs. This tool
is a stand-in for chapter 19's sandbox, and it is not a shell: nothing the
model types ever reaches bash. The command is cut into words the way a shell
cuts them (``shlex``), then checked, then carried out in Python:

- ``sqlite3 [options] data.db "<SQL or .tables or .schema>"`` runs through
  chapter 11's guard (``database.py``): read only, query-only, no ATTACH, a
  10-second limit. Its output looks like the sqlite3 program's.
- ``ls [-a] [-l] [path]`` lists the task folder.
- ``echo "<answer>" > answer.txt`` writes the answer file, and nothing else.

Every path must stay inside the task folder. Pipes, ``;``, ``&&``, ``||``,
any other redirection and command substitution are refused (a backtick
anywhere, even inside quoted SQL), so one call is one command. Neither
variables nor ``~`` are expanded. Nothing the model types makes the tool
raise: a refusal is an ``Error: ...`` output. Output longer than 4,000
characters is cut, as the project cuts Claude Code's (``bashOutputMaxChars``).
This module needs only the standard library.
"""

from __future__ import annotations

import math
import re
import shlex
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database import QueryError, Result, run_statements, schema_lines, table_names
from tasks import ANSWER_FILE

SQLITE = "sqlite3"
LS = "ls"
ECHO = "echo"
ALLOWED = (SQLITE, LS, ECHO)
REDIRECT = ">"
OPERATOR_CHARS = set("();<>|&")  # shlex returns runs of these as their own words
SUBSTITUTIONS = ("`", "$(")  # command substitution: a command inside a command
BRACKETS = "()"
QUOTE_SQL_HINT = ' (put the SQL in quotes: sqlite3 data.db "SELECT ...")'
OUTPUT_LIMIT = 4_000  # characters of one output the model sees
NO_NEWLINE = "-n"  # echo's flag: leave out the final newline
ECHO_FLAG = re.compile(r"-[neE]+")  # every flag word bash's echo understands
NEWLINE = "\n"
# sqlite3's output modes and the options that pick them.
LIST_MODE, CSV_MODE, COLUMN_MODE = "list", "csv", "column"
MODE_OPTIONS = {"-list": LIST_MODE, "-csv": CSV_MODE, "-column": COLUMN_MODE}
HEADER_OPTIONS = {"-header": True, "-noheader": False}
# Options that change nothing here: the database is always read only and safe.
IGNORED_OPTIONS = {"-readonly", "-safe", "-batch", "-bail"}
LIST_SEPARATOR = "|"
COLUMN_GAP = "  "
# csv mode: sqlite3 quotes a text only when it holds one of these, a space,
# a control character or anything past plain ASCII (code 127 and up).
CSV_SEPARATOR = ","
QUOTE = '"'
CSV_SPECIAL = {QUOTE, "'", CSV_SEPARATOR}
DELETE_CODE = 127
TABLES_WIDTH = 80  # sqlite3 lays .tables out in columns across 80 characters
# Dot-commands sqlite3 -safe refuses: they reach files or other programs.
UNSAFE_DOT_COMMANDS = {
    ".backup",
    ".cd",
    ".excel",
    ".import",
    ".load",
    ".log",
    ".once",
    ".open",
    ".output",
    ".read",
    ".restore",
    ".save",
    ".shell",
    ".system",
}
SETTING_ON = ("on", "yes", "true", "1")
# sqlite3's own message for a dot-command it does not know (two spaces and all).
UNKNOWN_DOT_COMMAND = (
    'unknown command or invalid arguments:  "{word}". Enter ".help" for help'
)
# ls: the flags understood. -a and -A both show names starting with a dot.
LS_FLAGS = set("aAl1h")
LS_HIDDEN_FLAGS = set("aA")
LS_LONG_FLAG = "l"
HIDDEN_PREFIX = "."
SIZE_WIDTH = 8


class Refused(Exception):
    """Raised when a command is not one this tool will run."""


@dataclass(frozen=True)
class ToolResult:
    """What running one command gave back.

    Attributes:
        output: The text the model will see as the observation.
        ok: False when the command was refused or failed (a shell's exit
            code would not be 0).
    """

    output: str
    ok: bool


def split_command(command: str) -> list[str]:
    """Cut a command into words as a shell would, refusing what is not simple.

    Args:
        command: The command the model sent.

    Returns:
        Its words; quotes are removed, and an operator such as ``>`` or
        ``|`` is a word of its own.

    Raises:
        Refused: If it substitutes a command, does not close a quote, or is
            empty.
    """
    if any(mark in command for mark in SUBSTITUTIONS):
        raise Refused("command substitution (` or $( ) is not run")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""  # shlex would drop everything after a #: keep it
    try:
        words = list(lexer)
    except ValueError as error:  # shlex: "No closing quotation"
        raise Refused(f"the command could not be read: {error}") from error
    if not words:
        raise Refused("the command is empty")
    return words


def is_operator(word: str) -> bool:
    """Say whether a word is a shell operator (``>``, ``|``, ``&&``, ``;``...).

    A quoted ``">"`` comes out of shlex looking the same as a real one, so it
    counts as an operator too: refusing it is the safe side.

    Args:
        word: One word from ``split_command``.

    Returns:
        True when every character is an operator character.
    """
    return bool(word) and set(word) <= OPERATOR_CHARS


def inside(workdir: Path, name: str) -> Path:
    """Resolve a path the model named, refusing one outside the task folder.

    Args:
        workdir: The task folder.
        name: The path as typed (relative to the folder, or absolute).

    Returns:
        The resolved path.

    Raises:
        Refused: If it leads outside the task folder, or cannot be a path
            at all (a NUL character, say).
    """
    # No ~ expansion: like variables, it is left as typed (a missing file).
    try:
        path = (workdir / name).resolve()
    except (OSError, ValueError) as error:
        raise Refused(f"{name!r}: not a usable path ({error})") from error
    if not path.is_relative_to(workdir.resolve()):
        raise Refused(f"{name}: only the task folder can be used")
    return path


def format_value(value: Any) -> str:
    """Write one result value as the sqlite3 program writes it.

    Args:
        value: A value from SQLite.

    Returns:
        The text: nothing for NULL, a float with up to 15 significant digits
        and always a decimal point (sqlite3's ``%!.15g``).
    """
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.15g}"
        if not math.isfinite(value):  # infinity has no decimal point to add
            return text
        # 523.0 stays 523.0 and 1e+20 becomes 1.0e+20, as sqlite3 prints them.
        mantissa, exponent, power = text.partition("e")
        point = "" if "." in mantissa else ".0"
        return mantissa + point + exponent + power
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def csv_cell(value: Any) -> str:
    """Write one value as sqlite3's csv mode does, quoting text only if needed.

    Args:
        value: A value from SQLite, or a column name.

    Returns:
        Numbers and NULL as in list mode; text in double quotes (each quote
        doubled) when it is empty or holds a space, a control or non-ASCII
        character, a quote or a comma.
    """
    if value is None or isinstance(value, int | float):
        return format_value(value)
    text = format_value(value)
    needs_quotes = not text or any(
        ord(char) <= ord(" ") or ord(char) >= DELETE_CODE or char in CSV_SPECIAL
        for char in text
    )
    return f'"{text.replace(QUOTE, QUOTE * 2)}"' if needs_quotes else text


def format_result(result: Result, mode: str, header: bool) -> list[str]:
    """Lay out one result as sqlite3 does in list, csv or column mode.

    Args:
        result: The columns and rows.
        mode: ``"list"``, ``"csv"`` or ``"column"``.
        header: Whether to print the column names first.

    Returns:
        The output lines.
    """
    if not result.rows:
        return []
    names = list(result.columns)
    if mode == CSV_MODE:
        table = ([[csv_cell(name) for name in names]] if header else []) + [
            [csv_cell(value) for value in row] for row in result.rows
        ]
        return [CSV_SEPARATOR.join(row) for row in table]
    rows = [[format_value(value) for value in row] for row in result.rows]
    if mode == COLUMN_MODE:
        # Each column is as wide as its widest value or its name, header or not.
        columns = zip(names, *rows, strict=True)
        widths = [max(len(cell) for cell in column) for column in columns]
        table = ([names, ["-" * width for width in widths]] if header else []) + rows
        return [
            COLUMN_GAP.join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
            for row in table
        ]
    return ([LIST_SEPARATOR.join(names)] if header else []) + [
        LIST_SEPARATOR.join(row) for row in rows
    ]


def tables_layout(names: Sequence[str]) -> list[str]:
    """Lay out table names in columns, down then across, as ``.tables`` does.

    Args:
        names: The table names, sorted.

    Returns:
        The output lines.
    """
    if not names:
        return []
    width = max(len(name) for name in names)
    per_line = max(1, TABLES_WIDTH // (width + len(COLUMN_GAP)))
    lines = -(-len(names) // per_line)  # rounded up
    return [
        COLUMN_GAP.join(name.ljust(width) for name in names[row::lines]).rstrip()
        for row in range(lines)
    ]


def dot_command(database: Path, text: str, settings: dict[str, Any]) -> list[str]:
    """Carry out one of sqlite3's dot-commands.

    Args:
        database: The database file.
        text: The command, such as ``.tables`` or ``.schema doctors``.
        settings: The output ``mode`` and ``header``, which ``.mode`` and
            ``.headers`` change for the arguments after them.

    Returns:
        The output lines.

    Raises:
        Refused: For a dot-command that reaches files or programs, or one
            this tool does not know (with sqlite3's own message).
        QueryError: If the database cannot be read.
    """
    name, *args = text.split()
    if name in UNSAFE_DOT_COMMANDS:
        raise Refused(f"cannot run {name} in safe mode")  # sqlite3 -safe's words
    if name == ".tables" and not args:
        return tables_layout(table_names(database))
    if name == ".schema" and len(args) <= 1:
        return schema_lines(database, args[0] if args else None)
    if name in (".headers", ".header") and len(args) == 1:
        settings["header"] = args[0].lower() in SETTING_ON
        return []
    if name == ".mode" and len(args) == 1 and args[0] in MODE_OPTIONS.values():
        settings["mode"] = args[0]
        return []
    raise Refused(UNKNOWN_DOT_COMMAND.format(word=name.removeprefix(".")))


def run_sqlite(words: list[str], workdir: Path) -> str:
    """Run ``sqlite3 [options] <database> <SQL or dot-command>...``.

    Args:
        words: The command's words, ``sqlite3`` first.
        workdir: The task folder.

    Returns:
        The output text.

    Raises:
        Refused: For an unknown option, a missing database or query, or a
            database outside the task folder.
        QueryError: If SQLite rejects a statement.
    """
    settings: dict[str, Any] = {"mode": LIST_MODE, "header": False}
    rest = words[1:]
    # sqlite3 accepts -option and --option alike.
    while rest and rest[0].startswith("-"):
        option = "-" + rest.pop(0).lstrip("-")
        if option in MODE_OPTIONS:
            settings["mode"] = MODE_OPTIONS[option]
        elif option in HEADER_OPTIONS:
            settings["header"] = HEADER_OPTIONS[option]
        elif option not in IGNORED_OPTIONS:
            raise Refused(f"sqlite3: unknown option: {option}")
    if not rest:
        raise Refused('sqlite3 needs a database: sqlite3 data.db "SELECT ..."')
    database = inside(workdir, rest[0])
    if not database.is_file():  # the real sqlite3 would make a new, empty file
        raise Refused(f'unable to open database "{rest[0]}": no such file')
    if len(rest) == 1:
        raise Refused(
            "sqlite3 needs a query after the database: it cannot wait for typed input"
        )
    lines: list[str] = []
    for text in rest[1:]:
        if text.strip().startswith("."):
            lines += dot_command(database, text.strip(), settings)
            continue
        for result in run_statements(database, text):
            lines += format_result(result, settings["mode"], settings["header"])
    return NEWLINE.join(lines)


def run_ls(words: list[str], workdir: Path) -> str:
    """Run ``ls [-a] [-l] [path...]`` inside the task folder.

    Args:
        words: The command's words, ``ls`` first.
        workdir: The task folder.

    Returns:
        One name per line (a pipe, not a screen, receives it), or with
        ``-l`` the mode, the size and the name (never the owner).

    Raises:
        Refused: For a flag this tool does not know, or a path outside the
            task folder.
    """
    flags = set("".join(word[1:] for word in words[1:] if word.startswith("-")))
    if not flags <= LS_FLAGS:
        raise Refused(f"ls: option not supported here: -{''.join(flags - LS_FLAGS)}")
    names = [word for word in words[1:] if not word.startswith("-")] or ["."]
    show_hidden = bool(flags & LS_HIDDEN_FLAGS)
    lines = []
    for name in names:
        path = inside(workdir, name)
        if not path.exists():
            raise Refused(f"ls: {name}: No such file or directory")
        entries = sorted(path.iterdir()) if path.is_dir() else [path]
        for entry in entries:
            if entry.name.startswith(HIDDEN_PREFIX) and not show_hidden:
                continue
            if LS_LONG_FLAG in flags:
                info = entry.lstat()
                mode = stat.filemode(info.st_mode)
                lines.append(f"{mode} {info.st_size:>{SIZE_WIDTH}} {entry.name}")
            else:
                lines.append(entry.name)
    return NEWLINE.join(lines)


def run_echo(words: list[str], workdir: Path) -> str:
    """Run ``echo [-n] <words> > answer.txt``: write the answer file.

    Args:
        words: The command's words, ``echo`` first.
        workdir: The task folder.

    Returns:
        Nothing (an echo into a file prints nothing).

    Raises:
        Refused: Unless the command ends with ``> answer.txt`` in the task
            folder.
    """
    if len(words) < 3 or words[-2] != REDIRECT:
        raise Refused('echo is allowed only as: echo "<answer>" > answer.txt')
    target = inside(workdir, words[-1])
    if target != workdir.resolve() / ANSWER_FILE or target.is_symlink():
        raise Refused(f"echo may write only {ANSWER_FILE}, in the task folder")
    text = words[1:-2]
    ending = NEWLINE
    if text[:1] == [NO_NEWLINE]:
        text, ending = text[1:], ""
    # bash's echo also reads -e and -E (backslash escapes); this one does not,
    # so it refuses them rather than write something bash would not. A word
    # such as -5 is not a flag and is written as it is.
    if text and text[0] != NO_NEWLINE and ECHO_FLAG.fullmatch(text[0]):
        raise Refused(f"echo: only {NO_NEWLINE} is understood here, not {text[0]}")
    target.write_text(" ".join(text) + ending, encoding="utf-8")
    return ""


RUNNERS: dict[str, Callable[[list[str], Path], str]] = {
    SQLITE: run_sqlite,
    LS: run_ls,
    ECHO: run_echo,
}


def cut(output: str, limit: int = OUTPUT_LIMIT) -> str:
    """Cut a long output to what the model will see, saying so.

    Args:
        output: The whole output.
        limit: The most characters kept.

    Returns:
        The output, or its first ``limit`` characters and a note.
    """
    if len(output) <= limit:
        return output
    return f"{output[:limit]}\n[output cut to {limit:,} of {len(output):,} characters]"


def run_command(
    command: str, workdir: Path, limit: int | None = OUTPUT_LIMIT
) -> ToolResult:
    """Check one command and, if it is allowed, carry it out.

    Args:
        command: The command the model sent.
        workdir: The task folder (it holds data.db; answer.txt goes here).
        limit: The most characters of output kept, or None to keep it all
            (for a harness that cuts long output its own way).

    Returns:
        The output the model will see, and whether the command succeeded.
        Nothing the model types makes it raise: a refusal or a failure is
        an ``Error: ...`` output.
    """
    try:
        words = split_command(command)
        # The only operator allowed is echo's "> answer.txt", checked there.
        operators = [word for word in words if is_operator(word)]
        if operators and not (words[0] == ECHO and operators == [REDIRECT]):
            # An unquoted bracket in SQL also lands here: say what to do.
            hint = QUOTE_SQL_HINT if set(operators[0]) <= set(BRACKETS) else ""
            raise Refused(
                f"{operators[0]!r} is not run: one command per call, and the only "
                f"redirection is echo ... > {ANSWER_FILE}{hint}"
            )
        if words[0] not in RUNNERS:
            allowed = ", ".join(ALLOWED)
            raise Refused(f"{words[0]}: not allowed here (this tool runs {allowed})")
        output = RUNNERS[words[0]](words, workdir)
        return ToolResult(output if limit is None else cut(output, limit), ok=True)
    except (Refused, QueryError) as error:
        return ToolResult(f"Error: {error}", ok=False)
    except OSError as error:  # a file the system would not read or write
        return ToolResult(f"Error: {error.strerror or error}", ok=False)
