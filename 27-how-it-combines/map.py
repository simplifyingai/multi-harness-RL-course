# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Every chapter beside the project file and function it mirrors, with sizes.

Chapters 1 to 26 each built one piece of the multi-harness-rl project on its
own. This prints, chapter by chapter, which project files and names that
piece is, how the chapter's code relates to them (a copy, a cut-down
version, the idea on the toy, or the project's file run as it is), and how
many lines each one is. The footer lists the project code no chapter
mirrors and the share that the chapters cover.

``uv run map.py`` prints it from the bundled snapshot (``data/
project_snapshot.json``: the project at the commit it names). ``--project
PATH`` (or ``MHRL_PROJECT``) measures a checkout instead, checks that every
file and name is still there, and says where it differs from the snapshot.
``--chapter N`` or ``--part N`` prints one chapter or one part. Only
Python's standard library is needed; nothing of the project is imported or
run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import project_scan
from course_map import (
    CHAPTERS,
    OPENENV,
    PARTS,
    PROJECT,
    PUBLISHED,
    Chapter,
    Mirror,
    chapter,
    part_of,
)
from paths import ProjectNotFoundError, check_project, named_project, short_path

SCRIPT = "map.py"
WIDTH = 100  # wrap name lists here, so a part fits a recording's screen
PATH_COLUMN = 30
LINES_COLUMN = 5
INDENT = "   "
NAME_SEPARATOR = " · "
DIFFERENCES_SHOWN = 6
PERCENT = 100
EXIT_MISSING = 1
LINES_LINE = "lines     a file's lines (wc -l) · each name's, from its def to its end"
KIND_LINES = (
    "kinds     copy: the project's code line for line · cut down: the same steps,"
    " less around them",
    "          toy: the idea on the toy, full size in the project · runs: the"
    " chapter runs or reads it",
)


def folder_label(entry: Chapter) -> str:
    """Return a chapter's label: its number and its folder's name.

    Args:
        entry: The chapter.

    Returns:
        For example ``01 the-job``.
    """
    return f"{entry.number:02d} {entry.folder.split('-', 1)[1]}"


def display_path(mirror: Mirror) -> str:
    """Return how a mirrored file is shown.

    Args:
        mirror: The mirrored file.

    Returns:
        Its project path, or ``openenv/<file>`` for OpenEnv's.
    """
    return f"openenv/{mirror.path}" if mirror.origin == OPENENV else mirror.path


def wrap_names(items: list[str], first: str, rest: str) -> list[str]:
    """Lay ``name lines`` items out after a prefix, wrapping at ``WIDTH``.

    Args:
        items: The items, e.g. ``["grade 33", "check_answer 42"]``.
        first: The prefix of the first line.
        rest: The prefix of every later line (spaces, as wide as ``first``).

    Returns:
        The lines.
    """
    lines: list[str] = []
    current = first
    started = False
    for item in items:
        piece = (NAME_SEPARATOR if started else "") + item
        if started and len(current) + len(piece) > WIDTH:
            lines.append(current)
            current, piece = rest, item
        current += piece
        started = True
    lines.append(current)
    return lines


def mirror_lines(mirror: Mirror, measured: project_scan.Measured) -> list[str]:
    """Print one mirrored file: its path, its lines and its names' lines.

    Args:
        mirror: The mirrored file.
        measured: The measurements.

    Returns:
        The lines.
    """
    key = project_scan.key_of(mirror)
    size = measured.files.get(key)
    size_text = f"{size:,}" if size is not None else "?"
    path = display_path(mirror)
    first = f"{INDENT}{path:<{PATH_COLUMN}}{size_text:>{LINES_COLUMN}}  "
    rest = " " * len(first)
    named = measured.names.get(key, {})
    items = [f"{n} {named[n]}" if n in named else n for n in mirror.names]
    items += [f"`{command}`" for command in mirror.commands]
    if not items:
        items = ["the whole file"]
    return wrap_names(items, first, rest)


def chapter_lines(entry: Chapter, measured: project_scan.Measured) -> list[str]:
    """Print one chapter: its label, kind, files, note and mirrored files.

    Args:
        entry: The chapter.
        measured: The measurements.

    Returns:
        The lines.
    """
    files = ", ".join(entry.files) if entry.files else "(not written yet)"
    lines = [f"{folder_label(entry)} · {entry.kind} · {files}", f"{INDENT}{entry.note}"]
    for mirror in entry.mirrors:
        lines += mirror_lines(mirror, measured)
    return lines


def mapped_paths(numbers: Iterable[int]) -> set[str]:
    """Return the project files some of the chapters mirror.

    Args:
        numbers: The chapters.

    Returns:
        Their project paths.
    """
    wanted = set(numbers)
    return {
        mirror.path
        for entry in CHAPTERS
        if entry.number in wanted
        for mirror in entry.mirrors
        if mirror.origin == PROJECT
    }


def coverage_lines(measured: project_scan.Measured) -> list[str]:
    """Print the share of the project's code the published chapters mirror.

    Then the files they leave: those this chapter and the next take up, and
    those no chapter mirrors at all.

    Args:
        measured: The measurements.

    Returns:
        The footer's lines.
    """
    published = mapped_paths(PUBLISHED)
    every = mapped_paths(entry.number for entry in CHAPTERS)
    code = measured.code
    covered = sum(n for path, n in code.items() if path in published)
    total = sum(code.values())
    files = sum(1 for path in code if path in published)
    lines = [
        f"the project's code: {len(code)} files, {total:,} lines; chapters "
        f"{PUBLISHED.start}-{PUBLISHED.stop - 1} mirror {files} files, "
        f"{covered:,} lines ({PERCENT * covered / total:.1f}%)"
    ]
    later = [
        f"{path} {n}"
        for path, n in sorted(code.items())
        if path in every and path not in published
    ]
    if later:
        lines.append("left for this chapter and the next")
        lines += wrap_names(later, INDENT, INDENT)
    nowhere = {path: n for path, n in code.items() if path not in every}
    items = [f"{path} {n}" for path, n in sorted(nowhere.items()) if n]
    empty = sum(1 for n in nowhere.values() if n == 0)
    if items or empty:
        lines.append("mirrored by no chapter")
        if empty:
            items.append(f"{empty} empty __init__.py files")
        lines += wrap_names(items, INDENT, INDENT)
    return lines


def render(
    measured: project_scan.Measured,
    source: list[str],
    numbers: Iterable[int] | None = None,
) -> list[str]:
    """Print the whole map, or the chapters asked for.

    Args:
        measured: The measurements.
        source: Lines saying where the measurements came from.
        numbers: Chapter numbers to print; None prints every chapter and the
            coverage footer.

    Returns:
        The lines.
    """
    wanted = None if numbers is None else set(numbers)
    lines = [f"{SCRIPT} · each chapter beside the project code it mirrors", *source]
    lines.append(LINES_LINE)
    lines += KIND_LINES
    shown_part = None
    for entry in CHAPTERS:
        if wanted is not None and entry.number not in wanted:
            continue
        part, title = part_of(entry.number)
        if part != shown_part:
            lines += ["", f"Part {part} · {title}"]
            shown_part = part
        lines += chapter_lines(entry, measured)
    if wanted is None:
        lines += ["", *coverage_lines(measured)]
    return lines


def snapshot_source(snapshot: project_scan.Measured) -> list[str]:
    """Say that the map comes from the bundled snapshot.

    Args:
        snapshot: The snapshot.

    Returns:
        The header lines.
    """
    return [
        f"project   the bundled snapshot of multi-harness-rl at {snapshot.commit} "
        f"(OpenEnv {snapshot.openenv_commit})",
        "          --project PATH or MHRL_PROJECT measures a checkout instead",
    ]


def live_source(
    project: Path, live: project_scan.Measured, snapshot: project_scan.Measured
) -> list[str]:
    """Say which checkout was measured and how it compares with the snapshot.

    Args:
        project: The checkout.
        live: Its measurements.
        snapshot: The bundled snapshot.

    Returns:
        The header lines.
    """
    commit = live.commit or "no git commit"
    lines = [
        f"project   {short_path(project)} at {commit}, measured now "
        f"(OpenEnv {live.openenv_commit or 'pin not found'})"
    ]
    if live.missing:
        lines.append(f"          MISSING from the checkout: {len(live.missing)}")
        lines += [f"{INDENT}{INDENT}{item}" for item in live.missing]
        return lines
    found = project_scan.differences(live, snapshot)
    if not found:
        lines.append(
            "          every mirrored file and name is there; the bundled "
            f"snapshot ({snapshot.commit}) agrees"
        )
        return lines
    lines.append(
        f"          every mirrored file and name is there; {len(found)} "
        f"differences from the bundled snapshot ({snapshot.commit}):"
    )
    lines += [f"{INDENT}{INDENT}{item}" for item in found[:DIFFERENCES_SHOWN]]
    if len(found) > DIFFERENCES_SHOWN:
        lines.append(f"{INDENT}{INDENT}and {len(found) - DIFFERENCES_SHOWN} more")
    return lines


def chosen_numbers(args: argparse.Namespace) -> list[int] | None:
    """Turn ``--chapter`` and ``--part`` into chapter numbers.

    Args:
        args: The parsed arguments.

    Returns:
        The chapter numbers, or None for the whole map.

    Raises:
        KeyError: If the chapter or part does not exist.
    """
    if args.chapter is not None:
        return [chapter(args.chapter).number]
    if args.part is not None:
        for part, _, numbers in PARTS:
            if part == args.part:
                return list(numbers)
        raise KeyError(f"no part {args.part}")
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments, or None for ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", help="a checkout to measure (or MHRL_PROJECT)")
    parser.add_argument("--chapter", type=int, help="print one chapter")
    parser.add_argument("--part", type=int, help="print one part of the course")
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="write the checkout's measurements as a snapshot to this file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, say: Callable[[str], None] = print) -> int:
    """Print the map.

    Args:
        argv: The arguments, or None for ``sys.argv``.
        say: Where each line goes.

    Returns:
        0, or 1 when a mirrored file or name is missing from the checkout.

    Raises:
        SystemExit: On a bad ``--chapter`` or ``--part``, or a ``--project``
            that is not a checkout.
    """
    args = parse_args(argv)
    try:
        numbers = chosen_numbers(args)
    except KeyError as error:
        raise SystemExit(f"{SCRIPT}: {error.args[0]}") from error
    snapshot = project_scan.load_snapshot()
    project = named_project(args.project)
    if project is None:
        if args.save_snapshot:
            raise SystemExit(f"{SCRIPT}: --save-snapshot needs --project")
        for line in render(snapshot, snapshot_source(snapshot), numbers):
            say(line)
        return 0
    try:
        check_project(project)
    except ProjectNotFoundError as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error
    live = project_scan.measure(project)
    for line in render(live, live_source(project, live, snapshot), numbers):
        say(line)
    if live.missing:
        return EXIT_MISSING
    if args.save_snapshot:
        data = project_scan.to_json(live)
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        say(f"snapshot  written to {short_path(args.save_snapshot.resolve())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
