"""Measure a checkout of the project: every mirrored file and name, with sizes.

A file's size is its line count, as ``wc -l`` gives it. A name's size is the
lines of its definition, from its ``def`` (or ``class``, or assignment) to its
last line, docstring included. Names are found by reading the Python file's
syntax tree (``ast``), so nothing of the project is imported or run; shell
functions are found by their ``name() {`` line, and ``python -m mhrl.cli``
subcommands in the ``choices`` of the command line's parser.

A **snapshot** is the same measurements written as JSON. One is bundled in
``data/project_snapshot.json`` (the project at the commit it names), so
``map.py`` works without a checkout; with one, ``map.py`` measures it live
and says whether it still matches. Only the standard library is needed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from course_map import CHAPTERS, OPENENV, PROJECT, Mirror

HERE = Path(__file__).resolve().parent
SNAPSHOT_FILE = HERE / "data" / "project_snapshot.json"
SNAPSHOT_VERSION = 1
# OpenEnv's capture package inside the project's environment.
OPENENV_GLOB = ".venv/lib/python3*/site-packages/openenv/core/harness/capture"
# The OpenEnv commit the project pins (pyproject.toml, [tool.uv.sources]).
OPENENV_PIN = re.compile(r'openenv\s*=\s*\{[^}]*rev\s*=\s*"(\w+)"')
SHELL_FUNCTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{", re.MULTILINE)
# The project's own code: what "every file" means when counting coverage.
CODE_FOLDERS = ("mhrl", "scripts", "configs", "tests")
CODE_SUFFIXES = (".py", ".sh", ".sb", ".yaml")
TOP_LEVEL_CODE = ("main.py",)
GIT_TIMEOUT_S = 10
SHORT_SHA = 7


class ProjectMismatchError(Exception):
    """A mirrored file or name is missing from the checkout."""


@dataclass(frozen=True)
class Measured:
    """Every measurement ``map.py`` prints.

    Attributes:
        commit: The checkout's commit (short), or "" when not a git checkout.
        openenv_commit: The OpenEnv commit the project pins.
        files: Line count per mirrored file, keyed ``origin:path``.
        names: Line count per mirrored name, keyed ``origin:path`` then name.
        commands: The ``mhrl.cli`` subcommands its parser accepts.
        code: Line count of every file of the project's own code.
        missing: Mirrored files or names the checkout does not have.
    """

    commit: str
    openenv_commit: str
    files: dict[str, int]
    names: dict[str, dict[str, int]]
    commands: tuple[str, ...]
    code: dict[str, int]
    missing: tuple[str, ...]


def key_of(mirror: Mirror) -> str:
    """Return the key a mirrored file is stored under.

    Args:
        mirror: The mirrored file.

    Returns:
        ``origin:path``.
    """
    return f"{mirror.origin}:{mirror.path}"


def count_lines(text: str) -> int:
    """Count lines as ``wc -l`` does (newline characters).

    Args:
        text: A file's text.

    Returns:
        Its line count.
    """
    return text.count("\n")


def _span(node: ast.AST) -> int:
    """Return how many lines a definition takes.

    Args:
        node: A ``def``, ``class`` or assignment node.

    Returns:
        Its last line minus its first, plus one.
    """
    end = getattr(node, "end_lineno", None) or node.lineno
    return end - node.lineno + 1


def _defined(node: ast.AST) -> dict[str, ast.AST]:
    """Return what a module or class body defines, by name.

    Args:
        node: A module or class node.

    Returns:
        Each function, class and assigned name, mapped to its node.
    """
    found: dict[str, ast.AST] = {}
    for child in getattr(node, "body", []):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found[child.name] = child
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = child
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            found[child.target.id] = child
    return found


def _nested(function: ast.AST, name: str) -> ast.AST | None:
    """Find a function defined inside another one.

    Args:
        function: The outer function's node.
        name: The inner function's name.

    Returns:
        Its node, or None.
    """
    for node in ast.walk(function):
        is_function = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        if node is not function and is_function and node.name == name:
            return node
    return None


def python_name_lines(source: str, name: str) -> int | None:
    """Measure one name in a Python file.

    Args:
        source: The file's text.
        name: ``f``, ``Class.member`` or ``outer.inner``.

    Returns:
        The lines of its definition, or None when the file has no such name.
    """
    node: ast.AST | None = ast.parse(source)
    for part in name.split("."):
        if node is None:
            return None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            node = _nested(node, part)
        else:
            node = _defined(node).get(part)
    return None if node is None else _span(node)


def shell_function_lines(source: str, name: str) -> int | None:
    """Measure one function of a shell script.

    It runs from its ``name() {`` line to the first line holding only ``}``.

    Args:
        source: The script's text.
        name: The function's name.

    Returns:
        Its lines, or None when the script has no such function.
    """
    lines = source.splitlines()
    for start, line in enumerate(lines):
        match = SHELL_FUNCTION.match(line)
        if match and match.group(1) == name:
            if line.rstrip().endswith("}"):
                return 1  # a one-line function
            for end in range(start + 1, len(lines)):
                if lines[end].strip() == "}":
                    return end - start + 1
    return None


def cli_commands(source: str) -> tuple[str, ...]:
    """Read the subcommands ``mhrl/cli.py``'s parser accepts.

    Args:
        source: The text of ``mhrl/cli.py``.

    Returns:
        The strings in its ``choices=[...]`` list.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "choices":
            value = node.value
            if isinstance(value, ast.List | ast.Tuple):
                return tuple(
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
    return ()


def openenv_folder(project: Path) -> Path | None:
    """Find OpenEnv's capture package in the project's environment.

    Args:
        project: The checkout.

    Returns:
        The package folder, or None when the environment is not installed.
    """
    found = sorted(project.glob(OPENENV_GLOB))
    return found[0] if found else None


def file_of(project: Path, mirror: Mirror) -> Path | None:
    """Return where a mirrored file is on disk.

    Args:
        project: The checkout.
        mirror: The mirrored file.

    Returns:
        Its path, or None when its origin is not installed.
    """
    if mirror.origin == PROJECT:
        return project / mirror.path
    if mirror.origin == OPENENV:
        folder = openenv_folder(project)
        return folder / mirror.path if folder else None
    raise ValueError(f"unknown origin {mirror.origin!r}")


def name_lines(path: Path, source: str, name: str) -> int | None:
    """Measure one name in a file of either kind.

    Args:
        path: The file.
        source: Its text.
        name: The name.

    Returns:
        Its lines, or None when it is not there.
    """
    if path.suffix == ".py":
        return python_name_lines(source, name)
    if path.suffix == ".sh":
        return shell_function_lines(source, name)
    return None


def git_commit(project: Path) -> str:
    """Return the checkout's commit, short, if it is a git checkout.

    Args:
        project: The checkout.

    Returns:
        The short commit, or "" when git cannot tell.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", f"--short={SHORT_SHA}", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def code_files(project: Path) -> list[Path]:
    """List every file of the project's own code.

    Args:
        project: The checkout.

    Returns:
        Paths relative to the checkout, sorted.
    """
    found = [Path(name) for name in TOP_LEVEL_CODE if (project / name).is_file()]
    for folder in CODE_FOLDERS:
        for path in (project / folder).rglob("*"):
            relative = path.relative_to(project)
            hidden = any(part.startswith((".", "__")) for part in relative.parts[:-1])
            if path.is_file() and path.suffix in CODE_SUFFIXES and not hidden:
                found.append(relative)
    return sorted(found)


def measure(project: Path) -> Measured:
    """Measure every mirrored file and name in a checkout.

    Args:
        project: The checkout.

    Returns:
        The measurements, with anything missing listed in ``missing``.
    """
    files: dict[str, int] = {}
    names: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    commands: tuple[str, ...] = ()
    for entry in CHAPTERS:
        for mirror in entry.mirrors:
            key = key_of(mirror)
            path = file_of(project, mirror)
            if path is None or not path.is_file():
                missing.append(f"{key} (file)")
                continue
            source = path.read_text(encoding="utf-8")
            files[key] = count_lines(source)
            for name in mirror.names:
                lines = name_lines(path, source, name)
                if lines is None:
                    missing.append(f"{key} {name}")
                else:
                    names.setdefault(key, {})[name] = lines
            if mirror.commands:
                commands = cli_commands(source)
                missing += [
                    f"{key} command {c}" for c in mirror.commands if c not in commands
                ]
    pyproject = project / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8") if pyproject.is_file() else ""
    pin = OPENENV_PIN.search(text)
    code = {
        str(path): count_lines((project / path).read_text(encoding="utf-8"))
        for path in code_files(project)
    }
    return Measured(
        commit=git_commit(project),
        openenv_commit=pin.group(1) if pin else "",
        files=files,
        names=names,
        commands=commands,
        code=code,
        missing=tuple(dict.fromkeys(missing)),
    )


def to_json(measured: Measured) -> dict[str, Any]:
    """Write measurements as a snapshot.

    Args:
        measured: What ``measure`` found.

    Returns:
        The snapshot, ready for ``json.dumps``.

    Raises:
        ProjectMismatchError: If anything mirrored was missing: a snapshot
            must be complete.
    """
    if measured.missing:
        raise ProjectMismatchError("missing: " + ", ".join(measured.missing))
    return {
        "version": SNAPSHOT_VERSION,
        "commit": measured.commit,
        "openenv_commit": measured.openenv_commit,
        "files": measured.files,
        "names": measured.names,
        "commands": list(measured.commands),
        "code": measured.code,
    }


def from_json(data: dict[str, Any]) -> Measured:
    """Read a snapshot back.

    Args:
        data: A snapshot, as ``to_json`` wrote it.

    Returns:
        The measurements it holds.

    Raises:
        ValueError: If it is another snapshot version.
    """
    if data.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"snapshot version {data.get('version')!r} is not mine")
    return Measured(
        commit=data["commit"],
        openenv_commit=data["openenv_commit"],
        files=dict(data["files"]),
        names={key: dict(value) for key, value in data["names"].items()},
        commands=tuple(data["commands"]),
        code=dict(data["code"]),
        missing=(),
    )


def load_snapshot(path: Path = SNAPSHOT_FILE) -> Measured:
    """Read the bundled snapshot.

    Args:
        path: The snapshot file.

    Returns:
        Its measurements.
    """
    return from_json(json.loads(path.read_text(encoding="utf-8")))


def differences(live: Measured, snapshot: Measured) -> list[str]:
    """Say where a live checkout differs from a snapshot.

    Args:
        live: The checkout's measurements.
        snapshot: The snapshot's.

    Returns:
        One line per difference; empty when they agree.
    """
    found: list[str] = []
    for label, now, then in (
        ("lines", live.files, snapshot.files),
        ("code lines", live.code, snapshot.code),
    ):
        for key in sorted(set(now) | set(then)):
            if now.get(key) != then.get(key):
                found.append(f"{label} {key}: {then.get(key)} -> {now.get(key)}")
    for key in sorted(set(live.names) | set(snapshot.names)):
        now, then = live.names.get(key, {}), snapshot.names.get(key, {})
        for name in sorted(set(now) | set(then)):
            if now.get(name) != then.get(name):
                found.append(f"{key} {name}: {then.get(name)} -> {now.get(name)}")
    if live.commands != snapshot.commands:
        found.append(f"commands: {snapshot.commands} -> {live.commands}")
    return found
