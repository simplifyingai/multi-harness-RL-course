# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Wrap a command in macOS ``sandbox-exec`` with the project's profile.

Chapter 19's ``sandbox.py`` (a copy of the project's
``mhrl/harnesses/sandbox.py``), pointed at this chapter's folders, with only
the project's profile (chapter 19's first, too-strict one is left out).
Chapter 29 adds two rules: the contents of files in other rollouts' folders,
in the family's run records and in this repository are denied (their
metadata is not, so paths still resolve), and loopback is open only on the
proxy's port.
Claude Code runs the model's shell commands on your laptop with
``bypassPermissions``, so it never runs unsandboxed: when the box cannot be
applied, this module raises instead of falling back, and nothing runs.

``uv run sandbox.py`` prints the ``sandbox-exec`` command a run would use,
with every parameter the profile reads. ``uv run sandbox.py --workdir
<task folder> -- <command>`` runs one command in the box a rollout in that
folder would get, and prints its exit status and output: the way to check
what the box lets a rollout see.
"""

from __future__ import annotations

import argparse
import os
import pwd
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from lfm_files import chapter_home, short_path

HERE = Path(__file__).resolve().parent
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SUPPORTED_PLATFORM = "darwin"

PROFILE = HERE / "sandbox.sb"  # the project's profile, two rules added

# Parameter names the profile reads with (param "...").
WORKDIR_PARAM = "WORKDIR"
STATE_DIR_PARAM = "STATE_DIR"
HOME_DIR_PARAM = "HOME_DIR"
ENV_FILE_REGEX_PARAM = "ENV_FILE_REGEX"
CLAUDE_JSON_REGEX_PARAM = "CLAUDE_JSON_REGEX"
CLAUDE_CREDS_REGEX_PARAM = "CLAUDE_CREDS_REGEX"
TASKS_DIR_PARAM = "TASKS_DIR"
RUNS_DIR_PARAM = "RUNS_DIR"
FAMILY_DIR_PARAM = "FAMILY_DIR"
REPO_DIR_PARAM = "REPO_DIR"
PROXY_PORT_PARAM = "PROXY_PORT"
EXTRA_PORT_PARAM = "EXTRA_PORT"
# Where the chapter keeps every rollout's folder and the family's run records
# (run_one.py and try_family.py name their folders from these).
RUNS_FOLDER = "runs"
FAMILY_FOLDER = "family"
REPO_DIR = HERE.parent  # this repository: its generator can rebuild the gold
DEFAULT_PROXY_PORT = 8200  # the project's proxy port
SYSTEM_TMP_PARAM = "SYSTEM_TMP"
SYSTEM_TMP_ON = "1"
EXAMPLE_RUN = "example"
EXAMPLE_FOLDERS = ("task", "state")  # where an example run would write
TRY_TIMEOUT_S = 30

# Secret-file patterns, appended to the escaped home folder:
# .env, .envrc, name.env, each with optional suffixes (.env.local); not .venv.
ENV_FILE_PATTERN = r"/(.+/)?(\.env(rc)?|[^/]+\.env)(\.[^/]*)?$"
CLAUDE_JSON_PATTERN = r"/\.claude\.json"
CLAUDE_CREDS_PATTERN = r"/\.claude/\.credentials"
# Characters with a meaning in the POSIX extended regexes Seatbelt uses.
REGEX_SPECIALS = frozenset(".[]()*+?{}|^$\\")

# Credential paths (relative to the home folder) the profile denies reading.
# Keep in sync with sandbox.sb; a test checks each one. A writable folder may
# not overlap them, because the profile re-allows reads inside the task folder
# and the state folder after the credential denies.
CREDENTIAL_PATHS = (
    ".ssh",
    ".aws",
    ".config/gh",
    ".config/gcloud",
    ".azure",
    ".gnupg",
    ".docker",
    ".kube",
    "Library/Keychains",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".cache/huggingface/token",
    ".huggingface/token",
    ".claude.json",
    ".claude/.credentials.json",
)


# The chapter's own exception root (the course is standard library only, so
# there is no application-wide base class to inherit from).
class SandboxError(Exception):
    """Base class for every reason the box will not run a command."""


class SandboxUnavailableError(SandboxError):
    """The box cannot be applied here (not macOS, or a file is missing)."""


class UnsafeSandboxPathError(SandboxError):
    """A folder handed to the profile would weaken it."""


def real_home() -> Path:
    """Return the user's real home folder, ignoring ``$HOME``.

    A program inside the box gets a fake ``$HOME``, so the real one is read
    from the password database, as the project does.

    Returns:
        The resolved home folder.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def regex_escape(text: str) -> str:
    """Escape text for a POSIX extended regex.

    Args:
        text: Literal text, such as a path.

    Returns:
        The text with every regex metacharacter backslash-escaped.
    """
    return "".join(f"\\{char}" if char in REGEX_SPECIALS else char for char in text)


def check_writable_root(path: Path, home: Path) -> Path:
    """Resolve a folder the box will make writable, refusing broad ones.

    Seatbelt matches real paths, so the path is resolved first. A folder that
    contains a home folder (``/`` or ``/Users``) would make that home
    writable; a folder inside or around a credential path (``~/.ssh``,
    ``~/.config``) would make the credential readable again, because the
    profile re-allows reads in its writable folders. Both are refused.

    Args:
        path: A folder the box may write to.
        home: The home folder the profile protects (normally your real one).

    Returns:
        The resolved path.

    Raises:
        UnsafeSandboxPathError: If the path is not an existing folder, contains
            a home folder, or overlaps a credential path.
    """
    resolved = path.resolve()
    if not resolved.is_dir():
        raise UnsafeSandboxPathError(f"not a folder: {resolved}")
    for protected_home in {home, real_home()}:
        if protected_home.is_relative_to(resolved):
            raise UnsafeSandboxPathError(
                f"{resolved} contains the home folder; refusing to make it writable"
            )
        for relative in CREDENTIAL_PATHS:
            secret = protected_home / relative
            if resolved.is_relative_to(secret) or secret.is_relative_to(resolved):
                raise UnsafeSandboxPathError(
                    f"{resolved} overlaps the credential path {secret}"
                )
    return resolved


def check_hidden_root(path: Path, writable: Sequence[Path]) -> Path:
    """Resolve a folder the box will hide, refusing one that overlaps a writable one.

    Hiding a folder above the task folder makes every path through it
    unreadable: that is exactly the first profile's bug (``sandbox_v1.sb``).
    A hidden folder INSIDE a writable one would not be hidden at all, because
    the profile re-allows reads in the writable folders after the deny.

    Args:
        path: A folder the profile denies reading.
        writable: The resolved task folder and state folder.

    Returns:
        The resolved path.

    Raises:
        UnsafeSandboxPathError: If the folder contains, or sits inside, a
            writable folder.
    """
    resolved = path.resolve()
    for folder in writable:
        if folder.is_relative_to(resolved):
            raise UnsafeSandboxPathError(
                f"{resolved} contains {folder}; hiding it would break every "
                "tool that resolves paths through it"
            )
        if resolved.is_relative_to(folder):
            raise UnsafeSandboxPathError(
                f"{resolved} is inside {folder}, where reads are allowed again"
            )
    return resolved


def check_contents_root(path: Path, writable: Sequence[Path]) -> Path:
    """Resolve a folder whose file contents the box denies, metadata kept.

    Unlike ``check_hidden_root``, the folder may contain the task folder
    (``runs/`` does): only ``file-read-data`` is denied, so paths through it
    still resolve, and the own folders are re-allowed after it. It may not
    sit inside a writable folder, where reads are allowed again.

    Args:
        path: A folder whose file contents the profile denies.
        writable: The resolved task folder and state folder.

    Returns:
        The resolved path.

    Raises:
        UnsafeSandboxPathError: If the folder sits inside a writable folder.
    """
    resolved = path.resolve()
    for folder in writable:
        if resolved != folder and resolved.is_relative_to(folder):
            raise UnsafeSandboxPathError(
                f"{resolved} is inside {folder}, where reads are allowed again"
            )
    return resolved


def sandbox_params(
    *,
    workdir: Path,
    state_dir: Path,
    home: Path | None = None,
    tasks_dir: Path | None = None,
    system_tmp: bool = False,
    proxy_port: int = DEFAULT_PROXY_PORT,
    extra_port: int | None = None,
) -> dict[str, str]:
    """Compute every parameter the profile reads.

    Args:
        workdir: The task folder the command may write to.
        state_dir: The run's private HOME/TMPDIR folder it may write to.
        home: The home folder whose credentials are hidden; your real one
            unless a decoy home is being tested.
        tasks_dir: The folder holding the gold answers (the family's
            ``tasks.gold_folder()`` when None).
        system_tmp: Also allow writes to the shared system temp folders.
        proxy_port: The one loopback port the command may connect to.
        extra_port: A second one (a probe's catcher); ``proxy_port`` if None.

    Returns:
        Parameter name to value.

    Raises:
        UnsafeSandboxPathError: If a writable folder is too broad, the tasks
            folder overlaps a writable folder, or a folder whose contents are
            denied sits inside one.
    """
    if tasks_dir is None:
        from tasks import gold_folder  # here, since tasks reads lfm_files too

        tasks_dir = gold_folder()
    protected = (home or real_home()).resolve()
    escaped = regex_escape(str(protected))
    writable = [
        check_writable_root(workdir, protected),
        check_writable_root(state_dir, protected),
    ]
    params = {
        WORKDIR_PARAM: str(writable[0]),
        STATE_DIR_PARAM: str(writable[1]),
        HOME_DIR_PARAM: str(protected),
        ENV_FILE_REGEX_PARAM: f"^{escaped}{ENV_FILE_PATTERN}",
        CLAUDE_JSON_REGEX_PARAM: f"^{escaped}{CLAUDE_JSON_PATTERN}",
        CLAUDE_CREDS_REGEX_PARAM: f"^{escaped}{CLAUDE_CREDS_PATTERN}",
        TASKS_DIR_PARAM: str(check_hidden_root(tasks_dir, writable)),
        RUNS_DIR_PARAM: str(
            check_contents_root(chapter_home() / RUNS_FOLDER, writable)
        ),
        FAMILY_DIR_PARAM: str(
            check_contents_root(chapter_home() / FAMILY_FOLDER, writable)
        ),
        REPO_DIR_PARAM: str(check_contents_root(REPO_DIR, writable)),
        PROXY_PORT_PARAM: str(proxy_port),
        EXTRA_PORT_PARAM: str(proxy_port if extra_port is None else extra_port),
    }
    if system_tmp:
        params[SYSTEM_TMP_PARAM] = SYSTEM_TMP_ON
    return params


def check_available(profile: Path) -> None:
    """Refuse to go on unless ``sandbox-exec`` and the profile exist.

    Args:
        profile: The Seatbelt profile file.

    Raises:
        SandboxUnavailableError: If this is not macOS, or ``sandbox-exec`` or
            the profile is missing.
    """
    if sys.platform != SUPPORTED_PLATFORM:
        raise SandboxUnavailableError(
            f"the box needs macOS sandbox-exec, and this is {sys.platform}"
        )
    if not SANDBOX_EXEC.is_file():
        raise SandboxUnavailableError(f"{SANDBOX_EXEC} is missing")
    if not profile.is_file():
        raise SandboxUnavailableError(f"sandbox profile {profile} is missing")


def sandbox_command(
    command: Sequence[str], *, params: dict[str, str], profile: Path = PROFILE
) -> list[str]:
    """Prefix a command with ``sandbox-exec``, the profile and its parameters.

    ``sandbox-exec`` applies the profile and then becomes the command, so the
    command's process id is the one ``Popen`` returns. If the profile cannot
    be applied (a syntax error, a missing parameter), ``sandbox-exec`` exits
    with an error and the command never starts.

    Args:
        command: The program and its arguments.
        params: The profile's parameters (``sandbox_params``).
        profile: The Seatbelt profile file.

    Returns:
        The full argv to execute.

    Raises:
        SandboxUnavailableError: If the box cannot be applied on this machine.
    """
    check_available(profile)
    defines: list[str] = []
    for name, value in params.items():
        defines += ["-D", f"{name}={value}"]
    return [str(SANDBOX_EXEC), "-f", str(profile), *defines, *command]


def try_in_box(workdir: Path, command: list[str], proxy_port: int) -> int:
    """Run one command in the box a rollout in ``workdir`` would get.

    Args:
        workdir: An existing task folder.
        command: The command and its arguments.
        proxy_port: The loopback port the box leaves open.

    Returns:
        The command's exit status.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    home = str(Path.home())
    with tempfile.TemporaryDirectory(prefix="mhrl-box-check-") as state:
        params = sandbox_params(
            workdir=workdir, state_dir=Path(state), proxy_port=proxy_port
        )
        argv = sandbox_command(command, params=params)
        done = subprocess.run(
            argv,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TRY_TIMEOUT_S,
            check=False,
        )
    print(f"$ {shlex.join(command)}".replace(home, "~"))
    print(f"  (in the box of {short_path(workdir.resolve())})")
    for line in (done.stdout + done.stderr).splitlines():
        print("  " + line.replace(home, "~"))
    print(f"  exit {done.returncode}")
    return done.returncode


def main() -> None:
    """Print the command a run would use, or run one command in a rollout's box."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, help="a task folder to try a box in")
    parser.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then a command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.workdir is not None:
        if not command:
            parser.error("--workdir needs a command after --")
        try_in_box(args.workdir, command, args.proxy_port)
        return
    example = chapter_home() / EXAMPLE_RUN
    workdir, state = (example / name for name in EXAMPLE_FOLDERS)
    workdir.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    params = sandbox_params(workdir=workdir, state_dir=state)
    argv = sandbox_command(["/bin/bash", "-c", "pwd"], params=params)
    home = str(Path.home())
    print(shlex.join(argv[:3]).replace(home, "~") + " \\")
    for index in range(3, len(argv) - 3, 2):
        print("    " + shlex.join(argv[index : index + 2]).replace(home, "~") + " \\")
    print("    " + shlex.join(argv[-3:]))
    print(f"(the example folders are in {short_path(example)})")


if __name__ == "__main__":
    main()
