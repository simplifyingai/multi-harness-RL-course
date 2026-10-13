# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Install the three harnesses at their pinned versions, and refuse any other.

A score means nothing without the harness that produced it, so each harness
is pinned to the version the project measured (``mhrl/harnesses/``), and a
run refuses to start on any other:

* **mini-swe-agent 2.4.6**, a Python program: installed with ``uv`` into
  this chapter's own environment, ``COURSE_HOME/25-sft-in-harnesses/
  mini-swe-agent-env``, on the project's Python (3.13.7), from
  ``mini-swe-agent-requirements.txt`` (every package at the version in the
  project's lock, installed with ``--no-deps``, as ``start.sh`` installs the
  proxy, and compiled to bytecode once, since the sandbox lets nothing write
  into it later);
* **Claude Code 2.1.290** and **Pi 1.0.4**, Node programs: installed with
  ``npm install --prefix`` into ``COURSE_HOME/25-sft-in-harnesses/claude-code``
  and ``.../pi``. npm runs each package's own install script (Claude Code's
  puts its binary in place) outside the sandbox, as any npm install does.

Both installers run with their settings made private: no settings file of
yours is read (your ``~/.npmrc`` may name a private registry and its token;
uv's ``uv.toml``, ``~/.netrc`` and ``UV_*`` variables may name another
index), their caches and uv's Python live in the chapter's folder, and they
fetch from the public registries.

Each install command and the version it resolved are kept in
``installs.json`` beside them and printed. The version is then asked of the
program itself, inside chapter 19's sandbox with a private ``HOME``, so the
probe cannot read or write your own configuration either.

``uv run install.py`` installs what is missing and checks all three;
``--check`` installs nothing. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lfm_files import chapter_home, short_path, tilde
from sandbox import SandboxError, sandbox_command, sandbox_params

HERE = Path(__file__).resolve().parent
MINI = "mini-swe-agent"
CLAUDE = "claude-code"
PI = "pi"
HARNESSES = (MINI, CLAUDE, PI)

PYPI, NPM = "pypi", "npm"
NPM_REGISTRY = "https://registry.npmjs.org/"
NPM_FOLDER = "npm"  # npm's private settings file and cache
NPM_SETTINGS_FILE = "npmrc"  # empty: your own npm settings are never read
# Empty too, in place of Homebrew's etc/npmrc. npm refuses one file as both.
NPM_GLOBAL_SETTINGS_FILE = "npmrc-global"
NPM_CACHE_FOLDER = "cache"
PYPI_INDEX = "https://pypi.org/simple"
UV_FOLDER = "uv"  # uv's private cache and the Python it installs
RECORD_FILE = "installs.json"
PYTHON_VERSION = "3.13.7"  # the project's Python
MINI_ENV_FOLDER = "mini-swe-agent-env"
MINI_REQUIREMENTS = HERE / "mini-swe-agent-requirements.txt"
MINI_MODULE = "minisweagent"
NODE_MODULES = "node_modules"
PACKAGE_JSON = "package.json"
CLAUDE_BINARY = Path(NODE_MODULES) / ".bin" / "claude"
PI_CLI = Path("dist") / "bundle" / "cli.js"  # the file the project runs
SAFE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
INSTALL_TIMEOUT_S = 900
VERSION_TIMEOUT_S = 60
PROBE_FOLDER = "probe"  # where a version probe runs: task/ and state/
MINI_VERSION_CODE = "import importlib.metadata as m; print(m.version('mini-swe-agent'))"
FOUND_EARLIER = "found installed (this record was not made by its install)"
# The settings worth printing beside each command (the rest is plumbing).
SHOWN_SETTINGS = {
    PYPI: ("UV_NO_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"),
    NPM: ("npm_config_userconfig", "npm_config_cache", "npm_config_registry"),
}
ERROR_TAIL_CHARS = 400  # how much of a failed command's output is shown
NAME_WIDTH = 15
# Claude Code's native binary comes in a package named after the platform.
CLAUDE_PLATFORMS = {"arm64": "darwin-arm64", "x86_64": "darwin-x64"}


@dataclass(frozen=True)
class Pin:
    """One harness and the only version this chapter runs.

    Attributes:
        name: The harness's name in this course.
        version: The pinned version.
        kind: ``pypi`` or ``npm``.
        package: The package that holds it.
        folder: Its install folder's name under the chapter's folder.
    """

    name: str
    version: str
    kind: str
    package: str
    folder: str


PINS = {
    MINI: Pin(MINI, "2.4.6", PYPI, "mini-swe-agent", MINI_ENV_FOLDER),
    CLAUDE: Pin(CLAUDE, "2.1.290", NPM, "@anthropic-ai/claude-code", "claude-code"),
    PI: Pin(PI, "1.0.4", NPM, "@earendil-works/pi-coding-agent", "pi"),
}


@dataclass
class Installed:
    """What one install did, as kept in ``installs.json``.

    Attributes:
        harness: The harness's name.
        pinned: The version it must be.
        resolved: The version the install left on disk.
        commands: Every command the install ran, as typed (home as ``~``).
        program: The command line that starts the harness.
        seconds: How long the install took (0 when it was already there).
        note: How the record was made.
        extra: Other versions worth keeping (the platform binary, node).
    """

    harness: str
    pinned: str
    resolved: str
    commands: list[str]
    program: list[str]
    seconds: float
    note: str = "installed by install.py"
    extra: dict[str, str] = field(default_factory=dict)


class HarnessInstallError(Exception):
    """Raised when a harness cannot be installed, found or run."""


class HarnessVersionError(HarnessInstallError):
    """Raised when a harness on disk is not its pinned version."""


def install_root() -> Path:
    """Where this chapter installs the harnesses.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses``.
    """
    return chapter_home()


def install_folder(name: str) -> Path:
    """Where one harness is installed.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        Its folder.
    """
    return install_root() / PINS[name].folder


def package_folder(name: str) -> Path:
    """The installed npm package's own folder.

    Args:
        name: ``claude-code`` or ``pi``.

    Returns:
        ``<prefix>/node_modules/<package>``.
    """
    return install_folder(name) / NODE_MODULES / PINS[name].package


def program(name: str) -> list[str]:
    """The command line that starts a harness, before its own arguments.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        mini-swe-agent: the chapter environment's Python with ``-m
        minisweagent``; Claude Code: its binary; Pi: node with its CLI.
    """
    if name == MINI:
        return [str(install_folder(MINI) / "bin" / "python"), "-m", MINI_MODULE]
    if name == CLAUDE:
        return [str(install_folder(CLAUDE) / CLAUDE_BINARY)]
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    return [node, str(package_folder(PI) / PI_CLI)]


def version_command(name: str) -> list[str]:
    """The command that makes a harness say its version.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        The argv (mini-swe-agent has no flag, so its Python reads its metadata).
    """
    if name == MINI:
        return [program(MINI)[0], "-c", MINI_VERSION_CODE]
    return [*program(name), "--version"]


def parse_version(output: str) -> str:
    """Read the version out of what a version command printed.

    Args:
        output: Its standard output, e.g. ``2.1.290 (Claude Code)``.

    Returns:
        The first word of the first line, or "" when it printed nothing.
    """
    lines = output.strip().splitlines()
    return lines[0].split()[0] if lines and lines[0].split() else ""


def check_version(name: str, reported: str) -> None:
    """Refuse any version but the pinned one.

    Args:
        name: A harness in ``HARNESSES``.
        reported: The version the program reported.

    Raises:
        HarnessVersionError: If it is not the pinned version.
    """
    pinned = PINS[name].version
    if reported != pinned:
        raise HarnessVersionError(
            f"{name} reports {reported or 'nothing'!r}; this chapter is pinned to "
            f"{pinned}: remove {short_path(install_folder(name))} and run "
            "`uv run install.py`"
        )


def probe_folders() -> tuple[Path, Path]:
    """Make the folders a version probe runs in, outside any run.

    Returns:
        The probe's task folder and state folder.
    """
    task = install_root() / PROBE_FOLDER / "task"
    state = install_root() / PROBE_FOLDER / "state"
    for folder in (task, state / "home", state / "tmp"):
        folder.mkdir(parents=True, exist_ok=True)
    return task, state


def probe_environment(state: Path) -> dict[str, str]:
    """The clean environment a version probe runs with.

    Args:
        state: The probe's state folder.

    Returns:
        A private HOME and TMPDIR, a safe PATH and nothing of yours.
    """
    return {
        "PATH": SAFE_PATH,
        "HOME": str(state / "home"),
        "TMPDIR": str(state / "tmp"),
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def probe_version(
    name: str, workdir: Path, state: Path, env: dict[str, str] | None = None
) -> str:
    """Ask a harness its version, inside the sandbox.

    Args:
        name: A harness in ``HARNESSES``.
        workdir: The folder it runs in (writable in the box).
        state: Its private state folder (writable in the box).
        env: Its environment; a clean probe environment when None.

    Returns:
        The version it reported.

    Raises:
        HarnessInstallError: If the box cannot be applied, or the program
            cannot run.
    """
    try:
        params = sandbox_params(workdir=workdir, state_dir=state)
        argv = sandbox_command(version_command(name), params=params)
    except SandboxError as error:
        raise HarnessInstallError(f"no sandbox, so no run: {error}") from error
    try:
        answer = subprocess.run(
            argv,
            cwd=workdir,
            env=env or probe_environment(state),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HarnessInstallError(f"cannot run {name}: {error}") from error
    return parse_version(answer.stdout)


def npm_environment() -> dict[str, str]:
    """Only what npm needs: your own npm settings and tokens never reach it.

    Returns:
        The environment for ``npm install`` (nothing is written here).
    """
    folder = install_root() / NPM_FOLDER
    return {
        "PATH": os.environ.get("PATH", SAFE_PATH),
        "HOME": str(folder),  # npm looks for ~/.npmrc here, and finds none
        "npm_config_userconfig": str(folder / NPM_SETTINGS_FILE),
        "npm_config_globalconfig": str(folder / NPM_GLOBAL_SETTINGS_FILE),
        "npm_config_cache": str(folder / NPM_CACHE_FOLDER),
        "npm_config_registry": NPM_REGISTRY,
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    }


def uv_environment() -> dict[str, str]:
    """Only what uv needs: no uv.toml, ~/.netrc or UV_* variable of yours.

    Returns:
        The environment for ``uv venv`` and ``uv pip install``.
    """
    folder = install_root() / UV_FOLDER
    return {
        "PATH": os.environ.get("PATH", SAFE_PATH),
        "HOME": str(folder),  # uv looks for ~/.netrc here, and finds none
        "UV_NO_CONFIG": "1",
        "UV_CACHE_DIR": str(folder / "cache"),
        "UV_PYTHON_INSTALL_DIR": str(folder / "python"),
    }


def installer_environment(kind: str) -> dict[str, str]:
    """The private environment one kind of install runs with.

    Args:
        kind: ``pypi`` or ``npm``.

    Returns:
        The environment.
    """
    return uv_environment() if kind == PYPI else npm_environment()


def prepare_installer(kind: str) -> None:
    """Make the installer's private folder, and npm's empty settings file.

    Args:
        kind: ``pypi`` or ``npm``.
    """
    folder = install_root() / (UV_FOLDER if kind == PYPI else NPM_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    if kind == NPM:
        for name in (NPM_SETTINGS_FILE, NPM_GLOBAL_SETTINGS_FILE):
            (folder / name).write_text("", encoding="utf-8")


def install_commands(name: str) -> list[list[str]]:
    """The commands that install a harness, in order.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        Each command's argv.
    """
    pin = PINS[name]
    folder = install_folder(name)
    if pin.kind == PYPI:
        python = folder / "bin" / "python"
        return [
            [
                "uv", "venv", "--quiet", "--managed-python",
                "--python", PYTHON_VERSION, str(folder),
            ],
            [
                "uv", "pip", "install", "--quiet", "--python", str(python),
                "--default-index", PYPI_INDEX,
                "--no-deps", "--compile-bytecode", "-r", str(MINI_REQUIREMENTS),
            ],
        ]  # fmt: skip
    return [
        [
            "npm", "install", "--prefix", str(folder), "--save-exact",
            "--no-audit", "--no-fund", f"{pin.package}@{pin.version}",
        ]
    ]  # fmt: skip


def shown_command(argv: list[str], kind: str) -> str:
    """Write a command as it would be typed, its private settings in front.

    Args:
        argv: The command.
        kind: ``pypi`` or ``npm``: whose settings to show.

    Returns:
        One line, the home folder as ``~``.
    """
    env = installer_environment(kind)
    prefix = " ".join(f"{key}={env[key]}" for key in SHOWN_SETTINGS[kind])
    text = shlex.join(argv).replace(str(HERE) + "/", "")
    return tilde(f"{prefix} {text}")


def resolved_version(name: str) -> str:
    """Read the version an install left on disk, from the package's own record.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        The version, or "" when the harness is not there.
    """
    if name == MINI:
        records = install_folder(MINI).glob(
            "lib/python*/site-packages/mini_swe_agent-*.dist-info"
        )
        found = sorted(
            record.name.removesuffix(".dist-info").split("-")[1] for record in records
        )
        return found[0] if len(found) == 1 else ""
    manifest = package_folder(name) / PACKAGE_JSON
    if not manifest.is_file():
        return ""
    return str(json.loads(manifest.read_text(encoding="utf-8")).get("version", ""))


def extra_versions(name: str) -> dict[str, str]:
    """Versions beside the harness's own that a reader may want.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        Claude Code's platform binary, or the node Pi runs on.
    """
    if name == CLAUDE:
        suffix = CLAUDE_PLATFORMS.get(platform.machine(), platform.machine())
        binary = f"{PINS[CLAUDE].package}-{suffix}"
        manifest = install_folder(CLAUDE) / NODE_MODULES / binary / PACKAGE_JSON
        if not manifest.is_file():
            return {"binary": f"{binary}: not found"}
        version = json.loads(manifest.read_text(encoding="utf-8"))["version"]
        return {"binary": f"{binary}@{version}"}
    if name == PI:
        node = program(PI)[0]
        try:
            answer = subprocess.run(
                [node, "--version"],
                capture_output=True,
                text=True,
                timeout=VERSION_TIMEOUT_S,
                check=False,
                env={"PATH": SAFE_PATH},
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"node": "not found"}
        return {"node": f"{tilde(node)} {answer.stdout.strip()}"}
    return {}


def run_install(name: str) -> Installed:
    """Install one harness at its pinned version.

    Args:
        name: A harness in ``HARNESSES``.

    Returns:
        What was done.

    Raises:
        HarnessInstallError: If a command fails or a tool is missing.
        HarnessVersionError: If the install resolved another version.
    """
    pin = PINS[name]
    folder = install_folder(name)
    commands = install_commands(name)
    tool = commands[0][0]
    if shutil.which(tool) is None:
        raise HarnessInstallError(f"{tool} is not installed; {name} needs it")
    if folder.exists():
        shutil.rmtree(folder)  # a half-done install: start again
    prepare_installer(pin.kind)
    env = installer_environment(pin.kind)
    started = time.perf_counter()
    for argv in commands:
        print(f"  $ {shown_command(argv, pin.kind)}", flush=True)
        try:
            answer = subprocess.run(
                argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT_S,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise HarnessInstallError(f"{argv[0]} failed: {error}") from error
        if answer.returncode != 0:
            tail = (answer.stderr or answer.stdout).strip()[-ERROR_TAIL_CHARS:]
            raise HarnessInstallError(f"{argv[0]} failed for {name}: {tail}")
    resolved = resolved_version(name)
    check_version(name, resolved)
    return Installed(
        harness=name,
        pinned=pin.version,
        resolved=resolved,
        commands=[shown_command(argv, pin.kind) for argv in commands],
        program=[tilde(part) for part in program(name)],
        seconds=round(time.perf_counter() - started, 1),
        extra=extra_versions(name),
    )


def read_record() -> dict[str, dict[str, Any]]:
    """Read ``installs.json``.

    Returns:
        Each harness's record, by name ({} when there is none).
    """
    path = install_root() / RECORD_FILE
    if not path.is_file():
        return {}
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def write_record(records: dict[str, dict[str, Any]]) -> Path:
    """Write ``installs.json``.

    Args:
        records: Each harness's record, by name.

    Returns:
        The file.
    """
    path = install_root() / RECORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path


def ensure(name: str, install: bool, records: dict[str, dict[str, Any]]) -> Installed:
    """Install a harness when it is missing, then hold it to its pin.

    Args:
        name: A harness in ``HARNESSES``.
        install: False to only look.
        records: ``installs.json`` so far (updated in place).

    Returns:
        Its record.

    Raises:
        HarnessInstallError: If it is missing and ``install`` is False.
        HarnessVersionError: If the version on disk is another one.
    """
    resolved = resolved_version(name)
    if resolved:
        check_version(name, resolved)  # never replaced silently
        kept = records.get(name)
        if isinstance(kept, dict) and kept.get("resolved") == resolved:
            with contextlib.suppress(TypeError):  # a record of another shape
                return Installed(**kept)
        found = Installed(
            harness=name,
            pinned=PINS[name].version,
            resolved=resolved,
            commands=[
                shown_command(argv, PINS[name].kind) for argv in install_commands(name)
            ],
            program=[tilde(part) for part in program(name)],
            seconds=0.0,
            note=FOUND_EARLIER,
            extra=extra_versions(name),
        )
    elif not install:
        raise HarnessInstallError(f"{name} is not installed: run `uv run install.py`")
    else:
        print(f"installing {name} {PINS[name].version}", flush=True)
        found = run_install(name)
    records[name] = asdict(found)
    write_record(records)
    return found


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("harness", nargs="*", help=f"any of {', '.join(HARNESSES)}")
    parser.add_argument("--check", action="store_true", help="install nothing")
    args = parser.parse_args()
    unknown = sorted(set(args.harness) - set(HARNESSES))
    if unknown:
        parser.error(f"unknown harness {', '.join(unknown)}: choose {HARNESSES}")
    return args


def main() -> None:
    """Install and check the harnesses, printing the record."""
    args = parse_args()
    names = args.harness or list(HARNESSES)
    records = read_record()
    task, state = probe_folders()
    failed = False
    for name in names:
        fresh = not resolved_version(name)  # its commands print as they run
        try:
            found = ensure(name, not args.check, records)
            reported = probe_version(name, task, state)
            check_version(name, reported)
        except HarnessInstallError as error:
            print(f"{name:<{NAME_WIDTH}} REFUSED · {error}")
            failed = True
            continue
        note = f"{found.note} in {found.seconds} s" if fresh else found.note
        print(f"{name:<{NAME_WIDTH}} {reported} (pinned {found.pinned}) · {note}")
        for command in [] if fresh else found.commands:
            print(f"  install   {command}")
        print(f"  runs as   {' '.join(found.program)}")
        for label, value in found.extra.items():
            print(f"  {label:<9} {value}")
    print(f"record    {short_path(install_root() / RECORD_FILE)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
