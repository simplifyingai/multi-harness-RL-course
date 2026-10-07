# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check that this laptop can run the course, and say where its files will go.

Run it with ``uv run check_machine.py``. Nothing is installed or written.
"""

from __future__ import annotations

import platform
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

from tasks import COURSE_HOME_ENV, course_home, short_path

BYTES_PER_GB = 1024**3
RECOMMENDED_MEMORY_GB = 16  # the training chapters peak near 9 GB
RECOMMENDED_FREE_DISK_GB = 15  # models, adapters and runs, all chapters
APPLE_SILICON_MACHINE = "arm64"
COMMAND_TIMEOUT_S = 10
LABEL_WIDTH = 14
VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")
FIRST_LOCAL_MODEL_CHAPTER = 10  # from here on the model runs on this Mac


def run(command: list[str]) -> str | None:
    """Run a small command and return its first output line.

    Args:
        command: The program and its arguments.

    Returns:
        The first line of output, or None if the program is missing or fails.
    """
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0] if result.returncode == 0 and lines else None


def memory_gb() -> float | None:
    """Return the machine's memory in GB (macOS), or None elsewhere.

    Returns:
        Unified memory in GB, or None when the system does not report it.
    """
    value = run(["sysctl", "-n", "hw.memsize"])
    return int(value) / BYTES_PER_GB if value and value.isdigit() else None


def show(label: str, value: str) -> None:
    """Print one aligned line.

    Args:
        label: What is being reported.
        value: Its value.
    """
    print(f"{label:<{LABEL_WIDTH}}{value}")


def free_disk_gb(folder: Path) -> float:
    """Return the free space on the disk that will hold ``folder``.

    Args:
        folder: A folder that may not exist yet.

    Returns:
        Free space in GB.
    """
    probe = folder
    while not probe.exists():  # walk up to a folder that exists
        probe = probe.parent
    return shutil.disk_usage(probe).free / BYTES_PER_GB


def version_number(line: str | None) -> str | None:
    """Pull the version out of a line like ``uv 0.11.21 (...)``.

    Args:
        line: A program's version line, or None.

    Returns:
        The first ``N.N`` or ``N.N.N`` in the line, or None.
    """
    match = VERSION_RE.search(line or "")
    return match.group(0) if match else None


def report(home: Path, memory: float | None, free_gb: float, uv: str | None) -> None:
    """Print one line per fact about this machine.

    Args:
        home: The course's cache folder.
        memory: Unified memory in GB, or None.
        free_gb: Free disk space in GB.
        uv: uv's version, or None when it is missing.
    """
    chip = run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    sqlite_cli = version_number(run(["sqlite3", "--version"]))
    show("chip", f"{chip} ({platform.machine()})")
    show("memory", f"{memory:.0f} GB unified" if memory else "unknown")
    show("macOS", platform.mac_ver()[0] or f"not macOS ({platform.system()})")
    show("python", platform.python_version())
    show("uv", uv or "MISSING: see https://docs.astral.sh/uv/")
    show("sqlite3", f"{sqlite_cli or 'MISSING'} (command line)")
    show("", f"{sqlite3.sqlite_version} (inside Python)")
    show("free disk", f"{free_gb:.0f} GB")
    show(COURSE_HOME_ENV, short_path(home))
    show("MLX", f"not needed until chapter {FIRST_LOCAL_MODEL_CHAPTER}")


def find_problems(uv: str | None) -> list[str]:
    """List what must be fixed before chapter 1 can run.

    Args:
        uv: uv's version, or None when it is missing.

    Returns:
        One short instruction per problem.
    """
    problems = []
    if uv is None:
        problems.append("install uv")
    if run(["sqlite3", "--version"]) is None:
        problems.append("install the sqlite3 command-line tool")
    return problems


def find_warnings(memory: float | None, free_gb: float) -> list[str]:
    """List what later chapters will need that this machine lacks.

    Args:
        memory: Unified memory in GB, or None.
        free_gb: Free disk space in GB.

    Returns:
        One short note per gap.
    """
    later = f"chapter {FIRST_LOCAL_MODEL_CHAPTER}"
    warnings = []
    if platform.machine() != APPLE_SILICON_MACHINE or not platform.mac_ver()[0]:
        warnings.append(f"{later} and later need an Apple-silicon Mac")
    if memory is not None and memory < RECOMMENDED_MEMORY_GB:
        warnings.append(f"{RECOMMENDED_MEMORY_GB} GB of memory is recommended")
    if free_gb < RECOMMENDED_FREE_DISK_GB:
        warnings.append(f"free {RECOMMENDED_FREE_DISK_GB} GB before {later}")
    return warnings


def main() -> None:
    """Print the machine report and a verdict."""
    home = course_home()
    memory = memory_gb()
    free_gb = free_disk_gb(home)
    uv = version_number(run(["uv", "--version"]))
    report(home, memory, free_gb, uv)
    problems = find_problems(uv)
    print()
    print("not ready: " + "; ".join(problems) if problems else "ready for chapter 1")
    for warning in find_warnings(memory, free_gb):
        print(f"note: {warning}")


if __name__ == "__main__":
    main()
