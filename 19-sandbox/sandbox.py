# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Wrap a command in macOS ``sandbox-exec`` with the chapter's profile.

A copy of the project's ``mhrl/harnesses/sandbox.py``, pointed at the course's
folders. The policy is a small model that runs shell commands on your laptop,
so nothing ever runs unsandboxed: when the box cannot be applied, this module
raises instead of falling back, and nothing runs.

``uv run sandbox.py`` prints the ``sandbox-exec`` command a rollout would run,
with every parameter the profile reads.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from tasks import DATA_DIR, chapter_home, real_home, show

HERE = Path(__file__).resolve().parent
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SUPPORTED_PLATFORM = "darwin"

# The two profiles: the project's, and the first, too-strict version.
PROJECT_PROFILE = "project"
V1_PROFILE = "v1"
PROFILES = {
    PROJECT_PROFILE: HERE / "sandbox.sb",
    V1_PROFILE: HERE / "sandbox_v1.sb",
}

# Parameter names the profile reads with (param "...").
WORKDIR_PARAM = "WORKDIR"
STATE_DIR_PARAM = "STATE_DIR"
HOME_DIR_PARAM = "HOME_DIR"
ENV_FILE_REGEX_PARAM = "ENV_FILE_REGEX"
CLAUDE_JSON_REGEX_PARAM = "CLAUDE_JSON_REGEX"
CLAUDE_CREDS_REGEX_PARAM = "CLAUDE_CREDS_REGEX"
TASKS_DIR_PARAM = "TASKS_DIR"
RUNS_DIR_PARAM = "RUNS_DIR"
SYSTEM_TMP_PARAM = "SYSTEM_TMP"
SYSTEM_TMP_ON = "1"

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


class SandboxError(Exception):
    """Base class for every reason the box will not run a command."""


class SandboxUnavailableError(SandboxError):
    """The box cannot be applied here (not macOS, or a file is missing)."""


class UnsafeSandboxPathError(SandboxError):
    """A folder handed to the profile would weaken it."""


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


def check_hidden_root(path: Path, workdir: Path) -> Path:
    """Resolve a folder the box will hide, refusing an ancestor of the workdir.

    Hiding a folder above the task folder makes every path through it
    unreadable: that is exactly the first profile's bug (``sandbox_v1.sb``).

    Args:
        path: A folder the profile denies reading.
        workdir: The resolved task folder.

    Returns:
        The resolved path.

    Raises:
        UnsafeSandboxPathError: If the folder contains the task folder.
    """
    resolved = path.resolve()
    if workdir.is_relative_to(resolved):
        raise UnsafeSandboxPathError(
            f"{resolved} contains the task folder {workdir}; hiding it would "
            "break every tool that resolves paths through it"
        )
    return resolved


def sandbox_params(
    *,
    workdir: Path,
    state_dir: Path,
    runs_dir: Path,
    home: Path | None = None,
    tasks_dir: Path = DATA_DIR,
    system_tmp: bool = False,
) -> dict[str, str]:
    """Compute every parameter the profile reads.

    Args:
        workdir: The task folder the command may write to.
        state_dir: The rollout's private HOME/TMPDIR folder it may write to.
        runs_dir: The folder holding every rollout's task folder (only the
            first, too-strict profile reads it).
        home: The home folder whose credentials are hidden; your real one
            unless a decoy home is being tested.
        tasks_dir: The folder holding the gold answers.
        system_tmp: Also allow writes to the shared system temp folders.

    Returns:
        Parameter name to value.

    Raises:
        UnsafeSandboxPathError: If a writable folder is too broad, or the tasks
            folder contains the task folder.
    """
    protected = (home or real_home()).resolve()
    escaped = regex_escape(str(protected))
    resolved_workdir = check_writable_root(workdir, protected)
    params = {
        WORKDIR_PARAM: str(resolved_workdir),
        STATE_DIR_PARAM: str(check_writable_root(state_dir, protected)),
        HOME_DIR_PARAM: str(protected),
        ENV_FILE_REGEX_PARAM: f"^{escaped}{ENV_FILE_PATTERN}",
        CLAUDE_JSON_REGEX_PARAM: f"^{escaped}{CLAUDE_JSON_PATTERN}",
        CLAUDE_CREDS_REGEX_PARAM: f"^{escaped}{CLAUDE_CREDS_PATTERN}",
        TASKS_DIR_PARAM: str(check_hidden_root(tasks_dir, resolved_workdir)),
        # Not checked: the first profile hides it although it IS an ancestor.
        RUNS_DIR_PARAM: str(runs_dir.resolve()),
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
    command: Sequence[str], *, profile: Path, params: dict[str, str]
) -> list[str]:
    """Prefix a command with ``sandbox-exec``, the profile and its parameters.

    ``sandbox-exec`` applies the profile and then becomes the command, so the
    command's process id is the one ``Popen`` returns. If the profile cannot
    be applied (a syntax error, a missing parameter), ``sandbox-exec`` exits
    with an error and the command never starts.

    Args:
        command: The program and its arguments.
        profile: The Seatbelt profile file.
        params: The profile's parameters (``sandbox_params``).

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


def main() -> None:
    """Print the command a rollout would run, one parameter per line."""
    home = chapter_home()
    example = home / "rollouts" / "example"
    state = home / "state" / "example"
    example.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    params = sandbox_params(workdir=example, state_dir=state, runs_dir=example.parent)
    argv = sandbox_command(
        ["/bin/bash", "-c", "pwd"], profile=PROFILES[PROJECT_PROFILE], params=params
    )
    print(show(shlex.join(argv[:3])) + " \\")
    for index in range(3, len(argv) - 3, 2):
        print("    " + show(shlex.join(argv[index : index + 2])) + " \\")
    print("    " + show(shlex.join(argv[-3:])))


if __name__ == "__main__":
    main()
