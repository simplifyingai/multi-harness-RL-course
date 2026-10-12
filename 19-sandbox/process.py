# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run one command: logged to a file, time-limited, its whole tree reaped.

A copy of the project's ``mhrl/harnesses/process.py``, plus a report of what
was seen, killed and left alive, so the chapter can show it. A harness is a
process tree, not a process: mini-swe-agent starts every shell command in a
NEW session, and a model can background a job that outlives its shell.
Killing only the first process would leave its children running after a
timeout. So on exit or timeout the runner kills:

1. the command's own process group (while the command is unreaped, so its id
   cannot have been recycled);
2. every descendant still attached to it, and every process in a process
   group a descendant was seen using (the table is polled every second while
   it runs, and read once more just before the kill);
3. every process whose environment carries this run's marker
   (``MHRL_RUN_ID=<hex>``), which catches orphans that left the tree and its
   group before a poll saw them. macOS hides the environment of Apple's own
   programs (``/bin/sh``, ``/bin/sleep``), so this sweep sees Python and Node
   orphans but not those; step 2 is what catches them.

The whole sweep runs twice (two passes), to catch a child forked while the
first pass was killing its parent. Steps 2 and 3 only signal processes that
started after the run began, so a process older than the run is never
touched, even if its id was recycled into the tree's groups. (A pid recycled
DURING the run is not ruled out this way; on a laptop that is very unlikely.)

Afterwards ``ps`` is asked whether anything seen, or anything carrying the
marker, is still running; if ``ps`` itself fails, the outcome says so instead
of reporting a clean reap.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PS = "/bin/ps"
# lstart is five words ("Mon Oct  6 05:47:12 2026"); comm (the program) is last.
PS_COLUMNS = "pid=,ppid=,pgid=,lstart=,comm="
PS_ID_FIELDS = 3
PS_LSTART_WORDS = 5
PS_ENV_COLUMNS = "pid=,command="
PS_TIMEOUT_S = 10
LSTART_FORMAT = "%a %b %d %H:%M:%S %Y"
C_LOCALE_ENV = {"LC_ALL": "C"}

POLL_INTERVAL_S = 1.0
# lstart has one-second resolution; allow for the rounding.
START_SLACK_S = 2.0
# Two passes catch a child forked while the first pass was killing its parent.
REAP_PASSES = 2
# pid 1 is launchd; nothing at or below it is ever signalled.
LAUNCHD_PID = 1
# A killed process can stay in the table for a moment while it exits.
SURVIVOR_WAIT_S = 2.0
SURVIVOR_POLL_S = 0.1


@dataclass(frozen=True)
class ProcessInfo:
    """One row of the process table.

    Attributes:
        pid: Process id.
        ppid: Parent process id.
        pgid: Process group id.
        started: Start time, in epoch seconds.
        name: The program's name (the last part of its path).
    """

    pid: int
    ppid: int
    pgid: int
    started: float
    name: str


@dataclass(frozen=True)
class ProcessOutcome:
    """How a command ended, and what the reaper found.

    Attributes:
        exit_code: The exit status, or None when it was killed on timeout.
        seconds: Wall-clock duration.
        timed_out: Whether the hard timeout fired.
        root_pid: The command's process id (also its process group id).
        seen: Every process seen in the tree while it ran, by pid.
        signalled: The pids the sweeps sent SIGKILL to.
        survivors: Pids still alive after the reaping: seen ones, and any
            process carrying the run's marker.
        ps_failed: Whether ``ps`` could not be read afterwards, so
            ``survivors`` could not be checked.
    """

    exit_code: int | None
    seconds: float
    timed_out: bool
    root_pid: int
    seen: dict[int, ProcessInfo] = field(default_factory=dict)
    signalled: frozenset[int] = frozenset()
    survivors: frozenset[int] = frozenset()
    ps_failed: bool = False


def _ps(columns: str, *extra: str) -> str:
    """Run ``ps -A`` with the given columns in the C locale.

    Args:
        columns: The ``-o`` column spec.
        *extra: Additional ``ps`` flags.

    Returns:
        Its stdout; empty (and a warning logged) if ``ps`` failed.
    """
    try:
        result = subprocess.run(
            [PS, "-A", *extra, "-o", columns],
            capture_output=True,
            text=True,
            check=True,
            timeout=PS_TIMEOUT_S,
            env={**os.environ, **C_LOCALE_ENV},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ps failed (%s); process cleanup may be incomplete", exc)
        return ""
    return result.stdout


def parse_ps_line(line: str) -> ProcessInfo | None:
    """Read one line of ``ps -o pid=,ppid=,pgid=,lstart=,comm=``.

    Args:
        line: One line of output.

    Returns:
        The row, or None when the line cannot be read.
    """
    words = line.split()
    date_end = PS_ID_FIELDS + PS_LSTART_WORDS
    if len(words) <= date_end:
        return None
    try:
        pid, ppid, pgid = (int(word) for word in words[:PS_ID_FIELDS])
        lstart = " ".join(words[PS_ID_FIELDS:date_end])
        started = time.mktime(time.strptime(lstart, LSTART_FORMAT))
    except ValueError:
        return None
    name = Path(" ".join(words[date_end:])).name
    return ProcessInfo(pid, ppid, pgid, started, name)


def process_table() -> list[ProcessInfo]:
    """Read every process's pid, parent, group, start time and name.

    Returns:
        The rows ``ps`` reported; empty if ``ps`` could not be run.
    """
    rows = []
    for line in _ps(PS_COLUMNS).splitlines():
        row = parse_ps_line(line)
        if row is not None:
            rows.append(row)
    return rows


def processes_with_marker(marker: str) -> set[int]:
    """Find processes whose environment contains a marker string.

    Args:
        marker: A unique ``NAME=value`` token placed in the run's environment.

    Returns:
        Their pids (Apple's own programs never match: macOS hides their env).
    """
    found: set[int] = set()
    for line in _ps(PS_ENV_COLUMNS, "-E", "-ww").splitlines():
        if marker not in line:
            continue
        pid, _, _ = line.strip().partition(" ")
        if pid.isdigit():
            found.add(int(pid))
    return found


def descendants(root_pid: int, table: Iterable[ProcessInfo]) -> list[ProcessInfo]:
    """Return every process below ``root_pid`` in the parent tree.

    Args:
        root_pid: The ancestor.
        table: A process table snapshot.

    Returns:
        The descendants, in no particular order (the root is excluded).
    """
    children: dict[int, list[ProcessInfo]] = {}
    for row in table:
        children.setdefault(row.ppid, []).append(row)
    found: list[ProcessInfo] = []
    stack = [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child.pid)
    return found


def _kill(pid: int) -> bool:
    """Send SIGKILL to one process, ignoring ones already gone.

    Args:
        pid: Process id.

    Returns:
        Whether the signal was sent. Pids 0 and 1 are refused: 0 would mean
        the caller's own group, 1 is launchd.
    """
    if pid <= LAUNCHD_PID:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill_group(pgid: int) -> None:
    """Send SIGKILL to a whole process group, ignoring an empty one.

    Args:
        pgid: Process group id.
    """
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def kill_tree(
    root_pid: int,
    groups: Iterable[int],
    started_after: float,
    *,
    root_reaped: bool,
    marker: str | None = None,
) -> set[int]:
    """Kill a command's process tree and everything it left behind.

    Args:
        root_pid: The command's process; also the id of the group it leads.
        groups: Process groups its descendants were seen using.
        started_after: Epoch seconds when the run began. Only processes that
            started after this (less ``START_SLACK_S``) are signalled from the
            table and marker sweeps.
        root_reaped: Whether the command was already waited for. Only an
            unreaped root (whose pid cannot be recycled) is killed
            unconditionally, with its group.
        marker: The run's ``NAME=value`` environment marker, if any.

    Returns:
        The pids signalled through the sweeps.
    """
    known_groups = {root_pid, *groups}
    cutoff = started_after - START_SLACK_S
    signalled: set[int] = set()
    for _ in range(REAP_PASSES):
        if not root_reaped:
            _kill_group(root_pid)
            _kill(root_pid)
        table = process_table()
        recent = {row.pid for row in table if row.started >= cutoff}
        targets = {row.pid for row in descendants(root_pid, table)}
        targets.update(row.pid for row in table if row.pgid in known_groups)
        if marker:
            targets.update(processes_with_marker(marker))
        targets.discard(os.getpid())
        for pid in targets & recent:
            if _kill(pid):
                signalled.add(pid)
    return signalled


def still_alive(seen: Mapping[int, ProcessInfo], *, wait: bool = True) -> set[int]:
    """Return the seen processes that are still in the process table.

    A row counts only if its start time matches the one seen, so a recycled
    pid is not mistaken for a survivor.

    Args:
        seen: Processes recorded while the command ran, by pid.
        wait: Give killed processes up to two seconds to leave the table.

    Returns:
        The pids still running.
    """
    deadline = time.monotonic() + (SURVIVOR_WAIT_S if wait else 0)
    while True:
        alive = {
            row.pid
            for row in process_table()
            if row.pid in seen and row.started == seen[row.pid].started
        }
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(SURVIVOR_POLL_S)


def kill_survivors(seen: Mapping[int, ProcessInfo]) -> None:
    """Kill processes a ``group_only`` demo left running.

    Each is checked again (same pid, same start time) right before the kill.

    Args:
        seen: The survivors' rows, by pid.
    """
    for pid in still_alive(seen, wait=False):
        _kill(pid)


def record_tree(
    seen: dict[int, ProcessInfo],
    root_pid: int,
    table: Iterable[ProcessInfo],
    cutoff: float,
) -> None:
    """Add the root, its descendants and its group's members to ``seen``.

    Args:
        seen: Processes recorded so far, by pid (updated in place).
        root_pid: The command's process, which leads its group.
        table: A process table snapshot.
        cutoff: Only processes started at or after this count (epoch s).
    """
    rows = list(table)
    found = descendants(root_pid, rows)
    found += [row for row in rows if row.pid == root_pid or row.pgid == root_pid]
    for row in found:
        if row.started >= cutoff and row.pid != os.getpid():
            seen[row.pid] = row


def marked(marker: str | None, cutoff: float) -> dict[int, ProcessInfo]:
    """Find the live processes that carry the run's marker.

    Args:
        marker: The run's ``NAME=value`` token, or None.
        cutoff: Only processes started at or after this count (epoch s).

    Returns:
        Their rows, by pid (never this process).
    """
    if not marker:
        return {}
    pids = processes_with_marker(marker) - {os.getpid()}
    return {
        row.pid: row
        for row in process_table()
        if row.pid in pids and row.started >= cutoff
    }


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_s: float,
    marker: str | None = None,
    group_only: bool = False,
) -> ProcessOutcome:
    """Run a command with stdout and stderr in a log file and a hard timeout.

    The command starts in its own session, so it leads a fresh process group
    that never contains the caller. Whether it exits or times out, its whole
    tree is killed afterwards (see ``kill_tree``) and checked with ``ps``.

    Args:
        command: The argv to run.
        cwd: Working directory.
        env: The complete environment (nothing is inherited); it should
            contain ``marker``.
        log_path: File that receives stdout and stderr (truncated first).
        timeout_s: Seconds before the tree is killed.
        marker: A unique ``NAME=value`` token in ``env``, used to find
            orphans that left the process tree.
        group_only: Kill only the run's process group (and only while the
            command is unreaped), as naive code would, to show what the
            sweeps catch. Survivors are left running.

    Returns:
        The exit status, duration, whether the timeout fired, and what the
        reaper saw, killed and found still alive.

    Raises:
        OSError: If the command cannot be started.
    """
    started_wall = time.time()
    cutoff = started_wall - START_SLACK_S
    start = time.monotonic()
    deadline = start + timeout_s
    seen: dict[int, ProcessInfo] = {}
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(POLL_INTERVAL_S, remaining))
                    break
                except subprocess.TimeoutExpired:
                    record_tree(seen, process.pid, process_table(), cutoff)
        finally:
            # One last look, so a child forked since the last poll is known.
            record_tree(seen, process.pid, process_table(), cutoff)
            signalled: set[int] = set()
            if group_only:
                # The naive way, for the demo: one signal to the run's group,
                # sent only while the root is unreaped (its id is still ours).
                if process.returncode is None:
                    _kill_group(process.pid)
            else:
                signalled = kill_tree(
                    process.pid,
                    {row.pgid for row in seen.values()},
                    started_wall,
                    root_reaped=process.returncode is not None,
                    marker=marker,
                )
            process.wait()
    exit_code = None if timed_out else process.returncode
    # An empty table means ps failed: then nothing can be said about survivors.
    ps_failed = not process_table()
    leftovers = marked(marker, cutoff)
    seen.update(leftovers)
    survivors = still_alive(seen) if not ps_failed else set()
    return ProcessOutcome(
        exit_code=exit_code,
        seconds=time.monotonic() - start,
        timed_out=timed_out,
        root_pid=process.pid,
        seen=seen,
        signalled=frozenset(signalled),
        survivors=frozenset(survivors),
        ps_failed=ps_failed,
    )
