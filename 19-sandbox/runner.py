# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run one command inside the box, the way the project runs every harness.

``uv run runner.py [--timeout S] [--profile project|v1] -- COMMAND ...``

Each run gets a fresh rollout:

* a task folder, ``$CH/rollouts/<id>``, holding a copy of the task's database
  as ``data.db``; it is the command's working folder and one of the only two
  places it may write;
* a private state folder, ``$CH/state/<id>``, with a fake ``home/`` and its
  own ``tmp/``, the other place it may write;
* a clean environment: ``PATH``, the fake ``HOME``, the private ``TMPDIR``,
  ``LANG`` and the run's marker ``MHRL_RUN_ID``; nothing else from your shell
  (no API keys, no tokens) reaches the box;
* its own session, a log file ``$CH/logs/<id>.txt``, and a hard timeout after
  which every process of the run is killed, in two passes, and ``ps`` is
  asked whether any survived.

If ``sandbox-exec`` or the profile is missing, nothing runs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from process import (
    REAP_PASSES,
    ProcessOutcome,
    kill_survivors,
    run_logged,
    still_alive,
)
from sandbox import (
    PROFILES,
    PROJECT_PROFILE,
    SandboxError,
    sandbox_command,
    sandbox_params,
)
from tasks import (
    DATA_DIR,
    DATABASE_FILE,
    LOGS_DIR,
    ROLLOUTS_DIR,
    STATE_DIR,
    TEXT_ENCODING,
    chapter_home,
    legend,
    load_task,
    show,
)

# The project's limit for one rollout (mhrl/config.py rollout_timeout_s).
DEFAULT_TIMEOUT_S = 300.0
# What a command in the box may find on PATH: the system, plus Homebrew.
BASE_PATH_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)
BOX_SHELL = "/bin/bash"
BOX_LANG = "en_US.UTF-8"
# Marks every process of a run, so orphans that leave its tree can be found.
RUN_ID_VAR = "MHRL_RUN_ID"
RUN_ID_HEX_CHARS = 8
HOME_SUBDIR = "home"
TMP_SUBDIR = "tmp"
LOG_SUFFIX = ".txt"
EXIT_SANDBOX_UNAVAILABLE = 2
EXIT_TIMED_OUT = 124  # what coreutils' timeout returns
EXIT_SIGNAL_BASE = 128  # a shell reports death by signal N as 128 + N
UNSAFE_NAMES = frozenset({"", ".", ".."})
POLL_NOTE = "ps is read every second, and once more before the kill"


@dataclass(frozen=True)
class Rollout:
    """One rollout's folders and the settings its box is built from.

    Attributes:
        name: The rollout's folder name, e.g. ``"run-1a2b3c4d"``.
        workdir: The task folder (the command's working folder).
        state_dir: The private folder holding ``home/`` and ``tmp/``.
        run_id: The marker value every process of the run carries.
        profile: Which profile the box uses (``"project"`` or ``"v1"``).
        home: The home folder whose secrets are hidden (None: your real one).
        tasks_dir: The folder holding the gold answers.
    """

    name: str
    workdir: Path
    state_dir: Path
    run_id: str
    profile: str = PROJECT_PROFILE
    home: Path | None = None
    tasks_dir: Path = DATA_DIR

    @property
    def fake_home(self) -> Path:
        """The ``$HOME`` the command sees."""
        return self.state_dir / HOME_SUBDIR

    @property
    def tmp(self) -> Path:
        """The ``$TMPDIR`` the command sees."""
        return self.state_dir / TMP_SUBDIR

    @property
    def marker(self) -> str:
        """The ``MHRL_RUN_ID=<id>`` token in every process's environment."""
        return f"{RUN_ID_VAR}={self.run_id}"

    def environment(self) -> dict[str, str]:
        """Build the complete environment of a command in the box.

        Returns:
            The few variables the command gets; nothing is inherited.
        """
        return {
            "PATH": os.pathsep.join(BASE_PATH_DIRS),
            "HOME": str(self.fake_home),
            "TMPDIR": str(self.tmp),
            "LANG": BOX_LANG,
            RUN_ID_VAR: self.run_id,
        }

    def log_path(self, label: str) -> Path:
        """Return the log file for one command of this rollout.

        Args:
            label: A short name for the command.

        Returns:
            ``$CH/logs/<rollout>/<label>.txt``, its folder created.
        """
        folder = chapter_home() / LOGS_DIR / self.name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{label}{LOG_SUFFIX}"


def new_rollout(
    name: str | None = None,
    *,
    profile: str = PROJECT_PROFILE,
    home: Path | None = None,
    tasks_dir: Path = DATA_DIR,
) -> Rollout:
    """Make a fresh rollout: an empty task folder with the database, a state folder.

    Args:
        name: The folder name; a new ``run-<hex>`` when None. An existing
            rollout of that name is deleted first.
        profile: Which profile its box uses.
        home: The home folder whose secrets are hidden (None: your real one).
        tasks_dir: The folder holding the gold answers.

    Returns:
        The rollout.

    Raises:
        ValueError: If the name is not a plain folder name.
        OSError: If the folders cannot be made or the database copied.
    """
    run_id = uuid.uuid4().hex
    name = name or f"run-{run_id[:RUN_ID_HEX_CHARS]}"
    # The name becomes a folder that is deleted first: never a path.
    if Path(name).name != name or name in UNSAFE_NAMES:
        raise ValueError(f"a rollout name must be a plain folder name, not {name!r}")
    home_dir = chapter_home()
    workdir = home_dir / ROLLOUTS_DIR / name
    state_dir = home_dir / STATE_DIR / name
    for folder in (workdir, state_dir):
        shutil.rmtree(folder, ignore_errors=True)
    workdir.mkdir(parents=True)
    (state_dir / HOME_SUBDIR).mkdir(parents=True)
    (state_dir / TMP_SUBDIR).mkdir()
    # Copied OUTSIDE the box: the box itself may not read the tasks folder.
    shutil.copyfile(load_task().database, workdir / DATABASE_FILE)
    return Rollout(name, workdir, state_dir, run_id, profile, home, tasks_dir)


def box_argv(rollout: Rollout, command: Sequence[str]) -> list[str]:
    """Wrap a command in ``sandbox-exec`` for one rollout.

    Args:
        rollout: The rollout.
        command: The program and its arguments.

    Returns:
        The full argv.

    Raises:
        SandboxError: If the box cannot be applied, or a folder is unsafe.
    """
    params = sandbox_params(
        workdir=rollout.workdir,
        state_dir=rollout.state_dir,
        runs_dir=rollout.workdir.parent,
        home=rollout.home,
        tasks_dir=rollout.tasks_dir,
    )
    return sandbox_command(command, profile=PROFILES[rollout.profile], params=params)


def run_in_box(
    rollout: Rollout,
    command: Sequence[str],
    *,
    label: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    group_only: bool = False,
) -> tuple[ProcessOutcome, str]:
    """Run one command in a rollout's box and read back its output.

    Args:
        rollout: The rollout.
        command: The program and its arguments.
        label: A short name for the command's log file.
        timeout_s: Seconds before the whole tree is killed.
        group_only: Kill only the run's process group (the naive way).

    Returns:
        How it ended, and everything it printed (stdout and stderr).

    Raises:
        SandboxError: If the box cannot be applied; nothing ran.
        OSError: If the command cannot be started.
    """
    log_path = rollout.log_path(label)
    outcome = run_logged(
        box_argv(rollout, command),
        cwd=rollout.workdir,
        env=rollout.environment(),
        log_path=log_path,
        timeout_s=timeout_s,
        marker=rollout.marker,
        group_only=group_only,
    )
    return outcome, log_path.read_text(encoding=TEXT_ENCODING, errors="replace")


def describe_processes(outcome: ProcessOutcome, *, group_only: bool) -> list[str]:
    """Describe what the reaper saw and did, one line per fact.

    Args:
        outcome: The run's outcome.
        group_only: Whether only the run's group was killed.

    Returns:
        Lines for the terminal.
    """
    seen = sorted(outcome.seen.values(), key=lambda row: row.pid)
    if outcome.ps_failed:
        return ["survivors  UNKNOWN: ps could not be read after the run"]
    if not seen:
        return [f"tree       nothing of it was running at any look ({POLL_NOTE})"]
    lines = [
        f"tree       {len(seen)} processes seen while it ran "
        f"({POLL_NOTE}; root pid {outcome.root_pid})"
    ]
    for row in seen:
        group = (
            f"group {row.pgid} (the run's group)"
            if row.pgid == outcome.root_pid
            else f"group {row.pgid} (LEFT the run's group)"
        )
        lines.append(f"  pid {row.pid:<7} {row.name:<10} parent {row.ppid:<7} {group}")
    count = len(outcome.signalled)
    if group_only:
        lines.append(f"reaped     group {outcome.root_pid} killed, and nothing else")
    else:
        lines.append(
            f"reaped     group {outcome.root_pid} killed, then {REAP_PASSES} "
            "sweeps signalled "
            f"{count} {'process' if count == 1 else 'processes'} "
            "(the tree, every group it used, the marker)"
        )
    pids = ",".join(str(row.pid) for row in seen)
    if not outcome.survivors:
        lines.append(f"survivors  0 of {len(seen)} (ps -p {pids} lists none of them)")
        return lines
    for pid in sorted(outcome.survivors):
        row = outcome.seen[pid]
        lines.append(f"survivor   pid {pid} {row.name} is still running")
    return lines


def main() -> None:
    """Run the command after ``--`` in a fresh rollout and report how it ended."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=PROJECT_PROFILE)
    parser.add_argument(
        "--group-only",
        action="store_true",
        help="kill only the run's process group, to see what the sweeps catch",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give a command after --")
    rollout = new_rollout(profile=args.profile)
    for line in legend():
        print(line)
    print(
        f"rollout    {rollout.name} · profile {args.profile} "
        f"· timeout {args.timeout:g} s"
    )
    print(f"task dir   {show(str(rollout.workdir))}  (holds {DATABASE_FILE})")
    print(f"fake HOME  {show(str(rollout.fake_home))}")
    print(f"TMPDIR     {show(str(rollout.tmp))}")
    print(f"env        {' '.join(sorted(rollout.environment()))}")
    print(f"marker     {rollout.marker}")
    print(f"log        {show(str(rollout.log_path('command')))}")
    print(f"command    {show(shlex.join(command))}")
    try:
        outcome, output = run_in_box(
            rollout,
            command,
            label="command",
            timeout_s=args.timeout,
            group_only=args.group_only,
        )
    except SandboxError as exc:
        print(f"refused    {exc}; nothing ran")
        sys.exit(EXIT_SANDBOX_UNAVAILABLE)
    print("output     (the log file)")
    for line in show(output).splitlines():
        print(f"  | {line}")
    if outcome.timed_out:
        print(f"timeout    after {outcome.seconds:.1f} s the run was stopped")
    else:
        print(f"exit       {outcome.exit_code} after {outcome.seconds:.1f} s")
    for line in describe_processes(outcome, group_only=args.group_only):
        print(line)
    if outcome.survivors:
        rows = {pid: outcome.seen[pid] for pid in outcome.survivors}
        kill_survivors(rows)
        left = still_alive(rows)
        print(f"cleanup    the runner killed them now; {len(left)} still running")
    sys.exit(exit_status(outcome))


def exit_status(outcome: ProcessOutcome) -> int:
    """Turn an outcome into the runner's own exit status, the way a shell does.

    Args:
        outcome: The run's outcome.

    Returns:
        124 after a timeout, 128 + N for a death by signal N, else the code.
    """
    if outcome.timed_out or outcome.exit_code is None:
        return EXIT_TIMED_OUT
    if outcome.exit_code < 0:
        return EXIT_SIGNAL_BASE - outcome.exit_code
    return outcome.exit_code


if __name__ == "__main__":
    main()
