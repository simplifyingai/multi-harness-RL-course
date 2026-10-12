# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Claude Code 2.1.290, headless, pointed at the capture proxy, in the sandbox.

Chapter 16 ran the pinned Claude Code against a stand-in model; here it
reaches the model through the capture proxy, as the project runs it
(``mhrl/harnesses/claude_code.py``):

* the pinned version, installed with ``npx`` into
  ``COURSE_HOME/22-silent-capture-failures/claude-code`` with a private npm cache, and
  refused if ``claude --version`` says anything else (``CLAUDE_BIN`` may
  name an install already on disk, such as chapter 16's);
* an empty environment plus a private ``HOME``, ``TMPDIR`` and
  ``CLAUDE_CONFIG_DIR`` in the run's state folder: your own ``~/.claude`` is
  never read, and no real key is used. ``ANTHROPIC_BASE_URL`` is the proxy
  and ``ANTHROPIC_API_KEY`` the session id, so every call lands in the
  session;
* the project's flags: one tool (``--tools Bash``), no MCP servers, no
  settings files, Bash output capped at 4,000 characters, Bash run without
  asking, and one JSON line per event (``--output-format stream-json
  --verbose``). One flag more than the project: ``--include-partial-messages``
  adds each reply's raw stream events to the log, the only place it states
  the tokens of each reply (it changes nothing Claude Code sends: the
  proxy's tokens were the same with and without it);
* the project's box: Bash runs whatever the model writes, without asking, so
  Claude Code runs inside chapter 19's sandbox (``sandbox.py``): it may write
  only to the run's task and state folders, read no credentials and no gold
  answers, and reach nothing but this machine. When it stops, or overruns
  its time limit, its whole process group is killed, so no command the model
  started outlives the run (chapter 19's ``process.py`` sweeps further; a
  command that leaves the group is still boxed);
* ``CLAUDE_CODE_TOTAL_TOKENS_REMINDER`` is left to the caller: unset is
  Claude Code's default, ``off`` is the project's.

``Relay`` is a recording relay for the other side of the conversation: put
between Claude Code and the proxy, it passes every request and answer
through and keeps each request's path and JSON body, so a script can show
what Claude Code itself sent before the proxy rewrote it. It reads each
answer whole before passing it on (with its status and headers), so a
streamed answer arrives in one piece rather than event by event. Given a
``rewrite`` function, it also changes each JSON request on its way, and
keeps what it forwarded beside what Claude Code sent: ``breakages.py``
summarises the history this way, as a harness that compacts does. Only
Python's standard library is needed.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lfm_files import chapter_home
from sandbox import SandboxError, sandbox_command, sandbox_params

CLAUDE_VERSION = "2.1.290"
CLAUDE_PACKAGE = f"@anthropic-ai/claude-code@{CLAUDE_VERSION}"
NPM_REGISTRY = "https://registry.npmjs.org/"
INSTALL_FOLDER = "claude-code"
PATH_FILE = "path.txt"  # where the installed binary's path is kept
NPM_SETTINGS_FILE = "npmrc"  # empty: your own npm settings are never read
NPM_CACHE_FOLDER = "npm-cache"
CLAUDE_BIN_ENV = "CLAUDE_BIN"  # an install already on disk, to skip npx
REMINDER_ENV = "CLAUDE_CODE_TOTAL_TOKENS_REMINDER"
REMINDER_OFF = "off"
MODEL_NAME = "lfm2.5-2.6b"  # the name the engine serves, as the project passes it
TOOLS = "Bash"  # the one tool, as the project offers it
PERMISSION_MODE = "bypassPermissions"  # Bash runs without asking: the box is the guard
RUN_TIMEOUT_S = 900  # one rollout; the real model can take minutes per call
VERSION_TIMEOUT_S = 60
INSTALL_TIMEOUT_S = 900
STOP_WAIT_S = 10  # after the time limit, how long to wait for the group to die
API_TIMEOUT_MS = 600_000  # the project's REQUEST_TIMEOUT_MS
MAX_OUTPUT_TOKENS = 1_024  # the project's max_completion_tokens
BASH_OUTPUT_MAX_CHARS = 4_000
FLAG_SETTINGS = {
    "attribution": {"commit": "", "pr": ""},
    "includeGitInstructions": False,
    "bashOutputMaxChars": BASH_OUTPUT_MAX_CHARS,
}
SAFE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
SHELL = "/bin/bash"
LOCALE = "en_US.UTF-8"
PLAIN_TERMINAL = "dumb"  # no colours, no cursor tricks in the log
ENABLED = "1"
STREAM_FILE = "stream.jsonl"  # what Claude Code printed
STDERR_FILE = "stderr.txt"
TASK_FOLDER = "task"  # Claude Code's working folder: data.db, answer.txt
STATE_FOLDER = "state"  # its private HOME, TMPDIR and config
STATE_PARTS = ("home", "tmp", "config")
RELAY_TIMEOUT_S = 600
BAD_GATEWAY = 502  # what the relay answers when the proxy cannot be reached
TEXT_PLAIN = "text/plain"
# Headers a relay must not copy: they describe one hop, not the message.
HOP_HEADERS = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
)


class ClaudeNotInstalledError(Exception):
    """Raised when the pinned Claude Code cannot be found, run or installed."""


@dataclass(frozen=True)
class ClaudeRun:
    """One finished Claude Code run.

    Attributes:
        folder: The run's folder (task/, state/, the logs).
        exit_code: Claude Code's exit status (None when the time limit hit).
        seconds: How long it ran.
    """

    folder: Path
    exit_code: int | None
    seconds: float

    @property
    def task_folder(self) -> Path:
        """The folder Claude Code worked in."""
        return self.folder / TASK_FOLDER

    @property
    def stream(self) -> Path:
        """Claude Code's stream-json log."""
        return self.folder / STREAM_FILE


@dataclass(frozen=True)
class RelayedRequest:
    """One request the relay passed on.

    Attributes:
        path: Such as ``/v1/messages``.
        body: Its JSON body (an object).
    """

    path: str
    body: dict[str, Any]


def install_folder() -> Path:
    """Where this chapter installs Claude Code.

    Returns:
        ``COURSE_HOME/22-silent-capture-failures/claude-code``.
    """
    return chapter_home() / INSTALL_FOLDER


def installed_version(binary: Path) -> str:
    """Ask a ``claude`` binary its version, from an empty environment.

    Args:
        binary: The executable.

    Returns:
        Its first word, such as ``2.1.290``.

    Raises:
        ClaudeNotInstalledError: If it cannot be run, or does not answer.
    """
    try:
        answer = subprocess.run(
            [str(binary), "--version"],
            env={"PATH": SAFE_PATH, "HOME": str(install_folder())},
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ClaudeNotInstalledError(f"cannot run {binary}: {error}") from error
    return answer.stdout.split(" ", 1)[0].strip()


def recorded_binary(path_file: Path) -> Path | None:
    """Read the binary an earlier install recorded, if it is still there.

    Args:
        path_file: The install's ``path.txt``.

    Returns:
        The executable, or None when nothing was recorded or npx's cache no
        longer holds it.
    """
    if not path_file.is_file():
        return None
    binary = Path(path_file.read_text(encoding="utf-8").strip())
    return binary if binary.is_file() else None


def find_claude(install: bool = True) -> Path:
    """Find the pinned Claude Code, installing it with npx when needed.

    Args:
        install: False to only look.

    Returns:
        The ``claude`` executable, at version ``CLAUDE_VERSION``.

    Raises:
        ClaudeNotInstalledError: If it is missing, cannot run, or is another
            version.
    """
    named = os.environ.get(CLAUDE_BIN_ENV)
    folder = install_folder()
    if named:
        binary = Path(named)
        if not binary.is_file():
            raise ClaudeNotInstalledError(f"{CLAUDE_BIN_ENV}={binary} is not a file")
    else:
        found = recorded_binary(folder / PATH_FILE)
        if found is None and not install:
            raise ClaudeNotInstalledError(f"no Claude Code under {folder}")
        # Nothing recorded, or npx's cache was cleaned since: install again.
        binary = found or npx_install(folder)
    version = installed_version(binary)
    if version != CLAUDE_VERSION:
        raise ClaudeNotInstalledError(
            f"{binary} reports {version!r}; this chapter is pinned to {CLAUDE_VERSION}"
        )
    return binary


def npx_install(folder: Path) -> Path:
    """Install the pinned package with npx, its npm settings private.

    Args:
        folder: Where the npm cache and settings go.

    Returns:
        The installed ``claude`` executable.

    Raises:
        ClaudeNotInstalledError: If npx is missing or fails.
    """
    npx = shutil.which("npx")
    if npx is None:
        raise ClaudeNotInstalledError("npx is not installed (it comes with Node.js)")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / NPM_SETTINGS_FILE).write_text("", encoding="utf-8")
    # Only what npx needs: no tokens or settings from your own environment.
    env = {
        "PATH": os.environ.get("PATH", SAFE_PATH),
        "HOME": str(folder),
        "npm_config_userconfig": str(folder / NPM_SETTINGS_FILE),
        "npm_config_cache": str(folder / NPM_CACHE_FOLDER),
        "npm_config_registry": NPM_REGISTRY,
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
    }
    print(f"installing {CLAUDE_PACKAGE} with npx into {folder}", flush=True)
    try:
        answer = subprocess.run(
            [npx, "--yes", "--package", CLAUDE_PACKAGE, "-c", "command -v claude"],
            env=env,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ClaudeNotInstalledError(f"npx failed: {error}") from error
    lines = [line.strip() for line in answer.stdout.splitlines() if line.strip()]
    if answer.returncode != 0 or not lines:
        raise ClaudeNotInstalledError(f"npx failed: {answer.stderr.strip()[-300:]}")
    binary = Path(lines[-1])  # `command -v claude` prints the path last
    (folder / PATH_FILE).write_text(f"{binary}\n", encoding="utf-8")
    return binary


def clean_environment(
    folder: Path,
    base_url: str,
    api_key: str,
    reminder: str | None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the only environment Claude Code gets.

    Args:
        folder: The run's folder (its state folder holds home/, tmp/, config/).
        base_url: The proxy's root (Claude Code adds ``/v1/messages``).
        api_key: The capture session's id.
        reminder: ``CLAUDE_CODE_TOTAL_TOKENS_REMINDER``, or None to leave it
            unset (Claude Code's default).
        extra: More settings, such as ``compaction.py``'s, or None.

    Returns:
        The environment.
    """
    state = folder / STATE_FOLDER
    user = os.environ.get("USER", "learner")
    env = {
        "PATH": SAFE_PATH,
        "HOME": str(state / "home"),
        "TMPDIR": str(state / "tmp"),
        "USER": user,
        "LOGNAME": user,
        "SHELL": SHELL,
        "LANG": LOCALE,
        "TERM": PLAIN_TERMINAL,
        "NO_COLOR": ENABLED,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_MODEL": MODEL_NAME,
        # Any "small fast model" call goes to the same model, as in the project.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL_NAME,
        "CLAUDE_CONFIG_DIR": str(state / "config"),
        "CLAUDE_CODE_TMPDIR": str(state / "tmp"),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
        "API_TIMEOUT_MS": str(API_TIMEOUT_MS),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": ENABLED,
        "DISABLE_TELEMETRY": ENABLED,
        "DISABLE_ERROR_REPORTING": ENABLED,
        "DISABLE_AUTOUPDATER": ENABLED,
    }
    if reminder is not None:
        env[REMINDER_ENV] = reminder
    env.update(extra or {})
    return env


def command(binary: Path, instruction: str) -> list[str]:
    """Write the headless command line, the project's flags plus one.

    Args:
        binary: The ``claude`` executable.
        instruction: The task, as ``claude -p``'s argument.

    Returns:
        The argv.
    """
    return [
        str(binary),
        "-p",
        instruction,
        "--tools",
        TOOLS,
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--settings",
        json.dumps(FLAG_SETTINGS),
        "--permission-mode",
        PERMISSION_MODE,
        "--disable-slash-commands",
        "--no-session-persistence",
        "--model",
        MODEL_NAME,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]


def prepare_run(folder: Path, database: Path) -> Path:
    """Make a fresh run folder holding the task's database as data.db.

    Args:
        folder: The run's folder (emptied first).
        database: The database file to copy.

    Returns:
        The task folder Claude Code works in.
    """
    if folder.exists():
        shutil.rmtree(folder)  # a failure here is shown, not hidden
    task = folder / TASK_FOLDER
    task.mkdir(parents=True)
    for part in STATE_PARTS:
        (folder / STATE_FOLDER / part).mkdir(parents=True)
    shutil.copy(database, task / "data.db")
    return task


def boxed_command(binary: Path, folder: Path, instruction: str) -> list[str]:
    """Wrap the command line in the sandbox, writable only in the run's folders.

    Args:
        binary: The ``claude`` executable.
        folder: The run's folder.
        instruction: The task.

    Returns:
        The argv, ``sandbox-exec`` first.

    Raises:
        ClaudeNotInstalledError: If the box cannot be applied (then nothing runs).
    """
    try:
        params = sandbox_params(
            workdir=folder / TASK_FOLDER, state_dir=folder / STATE_FOLDER
        )
        return sandbox_command(command(binary, instruction), params=params)
    except SandboxError as error:
        raise ClaudeNotInstalledError(f"no sandbox, so no run: {error}") from error


def kill_group(process: subprocess.Popen) -> None:
    """Kill a run's whole process group: Claude Code and every command it started.

    Args:
        process: The run, started in a session of its own.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # the group is already gone


def run_claude(
    binary: Path,
    folder: Path,
    instruction: str,
    base_url: str,
    api_key: str,
    reminder: str | None,
    extra_env: dict[str, str] | None = None,
) -> ClaudeRun:
    """Run Claude Code once, headless and boxed, in a prepared run folder.

    Args:
        binary: The ``claude`` executable.
        folder: The run's folder, from ``prepare_run``.
        instruction: The task.
        base_url: Where Claude Code sends its calls (the proxy, or a relay).
        api_key: The capture session's id.
        reminder: The total-tokens note's setting, or None for the default.
        extra_env: More settings for Claude Code's environment, or None.

    Returns:
        The finished run; its log is ``stream.jsonl`` in the folder.

    Raises:
        ClaudeNotInstalledError: If the sandbox cannot be applied.
    """
    argv = boxed_command(binary, folder, instruction)
    env = clean_environment(folder, base_url, api_key, reminder, extra_env)
    started = time.perf_counter()
    exit_code: int | None = None
    with (
        (folder / STREAM_FILE).open("w", encoding="utf-8") as out,
        (folder / STDERR_FILE).open("w", encoding="utf-8") as err,
    ):
        # Its own session, so one signal reaches every process it starts.
        process = subprocess.Popen(
            argv,
            cwd=folder / TASK_FOLDER,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            exit_code = None
        finally:
            kill_group(process)  # also any background command left running
            # A process killed in its sleep is reaped at once; never wait forever.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=STOP_WAIT_S)
    return ClaudeRun(folder, exit_code, time.perf_counter() - started)


Rewrite = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class Relay:
    """A recording relay: Claude Code on one side, the proxy on the other.

    Attributes:
        upstream: The proxy's root.
        rewrite: Changes a request's JSON body on its way (path, body ->
            body), or None to pass every request unchanged.
        requests: Every request with a JSON object body, as Claude Code sent it.
        forwarded: The same requests as they reached the proxy (the same
            bodies unless ``rewrite`` changed them).
    """

    upstream: str
    rewrite: Rewrite | None = None
    requests: list[RelayedRequest] = field(default_factory=list)
    forwarded: list[RelayedRequest] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> str:
        """Listen on a free port of this machine.

        Returns:
            The relay's root, to give Claude Code as its base URL.
        """
        relay = self

        class Handler(BaseHTTPRequestHandler):
            """Hands every request to the relay."""

            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                """Relay a GET (no body)."""
                relay.forward(self, None)

            def do_POST(self) -> None:  # noqa: N802
                """Relay a POST with its body."""
                size = int(self.headers.get("Content-Length") or 0)
                relay.forward(self, self.rfile.read(size))

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Stay quiet: the relay is not part of the output.

                Args:
                    format: The message's format.
                    *args: Its values.
                """

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def keep(self, path: str, body: bytes | None) -> bytes | None:
        """Keep a request whose body is a JSON object, rewritten if asked.

        Args:
            path: The request's path.
            body: Its body, or None.

        Returns:
            The body to forward: the rewritten one, or the original bytes.
        """
        if not body:
            return body
        try:
            found = json.loads(body)
        except ValueError:  # not JSON (or not UTF-8): relayed, not kept
            return body
        if not isinstance(found, dict):
            return body
        sent = found
        if self.rewrite is not None:
            sent = self.rewrite(path, found)
        with self.lock:
            self.requests.append(RelayedRequest(path, found))
            self.forwarded.append(RelayedRequest(path, sent))
        return body if sent is found else json.dumps(sent).encode("utf-8")

    def forward(self, handler: BaseHTTPRequestHandler, body: bytes | None) -> None:
        """Pass one request to the proxy and its answer back, keeping the body.

        Args:
            handler: The request Claude Code made.
            body: Its body, or None for a GET.
        """
        try:
            body = self.keep(handler.path, body)
        except (KeyError, TypeError, ValueError) as error:  # a rewrite that failed
            self.answer(
                handler,
                BAD_GATEWAY,
                [("Content-Type", TEXT_PLAIN)],
                f"relay: the rewrite failed: {error!r}".encode(),
            )
            return
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in HOP_HEADERS
        }
        request = urllib.request.Request(
            self.upstream + handler.path,
            data=body,
            headers=headers,
            method=handler.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=RELAY_TIMEOUT_S) as answer:
                status, data = answer.status, answer.read()
                kept = answer.headers.items()
        except urllib.error.HTTPError as error:  # the proxy answered: a refusal
            status, kept, data = error.code, error.headers.items(), error.read()
        except (urllib.error.URLError, OSError) as error:  # nobody answered
            status, kept = BAD_GATEWAY, [("Content-Type", TEXT_PLAIN)]
            data = f"relay: cannot reach {self.upstream}: {error}".encode()
        self.answer(handler, status, kept, data)

    def answer(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        headers: Any,
        data: bytes,
    ) -> None:
        """Send Claude Code an answer: a status, the headers that travel, a body.

        Args:
            handler: The request Claude Code made.
            status: The HTTP status.
            headers: ``(name, value)`` pairs; per-hop ones are left out.
            data: The body.
        """
        handler.send_response(status)
        for name, value in headers:  # Retry-After, request ids, the content type...
            if name.lower() not in HOP_HEADERS:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def stop(self) -> None:
        """Stop listening."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
