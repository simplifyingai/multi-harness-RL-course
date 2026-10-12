# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Claude Code 2.1.290, headless, pointed at the capture proxy: chapter 16's setup.

Chapter 16 ran the pinned Claude Code against a stand-in model; here it
reaches the model through the capture proxy, as the project runs it
(``mhrl/harnesses/claude_code.py``):

* the pinned version, installed with ``npx`` into
  ``COURSE_HOME/21-calls-to-rows/claude-code`` with a private npm cache, and
  refused if ``claude --version`` says anything else (``CLAUDE_BIN`` may
  name an install already on disk, such as chapter 16's);
* an empty environment plus a private ``HOME``, ``TMPDIR`` and
  ``CLAUDE_CONFIG_DIR`` under the run's folder: your own ``~/.claude`` is
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
* ``CLAUDE_CODE_TOTAL_TOKENS_REMINDER`` is left to the caller: unset is
  Claude Code's default, ``off`` is the project's.

``Relay`` is a recording relay for the other side of the conversation: put
between Claude Code and the proxy, it passes every request and answer
through unchanged and keeps each request's JSON body, so a script can show
what Claude Code itself sent before the proxy rewrote it. Only Python's
standard library is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lfm_files import chapter_home

CLAUDE_VERSION = "2.1.290"
CLAUDE_PACKAGE = f"@anthropic-ai/claude-code@{CLAUDE_VERSION}"
NPM_REGISTRY = "https://registry.npmjs.org/"
INSTALL_FOLDER = "claude-code"
PATH_FILE = "path.txt"  # where the installed binary's path is kept
CLAUDE_BIN_ENV = "CLAUDE_BIN"  # an install already on disk, to skip npx
REMINDER_ENV = "CLAUDE_CODE_TOTAL_TOKENS_REMINDER"
REMINDER_OFF = "off"
MODEL_NAME = "lfm2.5-2.6b"  # the name the engine serves, as the project passes it
RUN_TIMEOUT_S = 900  # one rollout; the real model can take minutes per call
INSTALL_TIMEOUT_S = 900
API_TIMEOUT_MS = 600_000  # the project's REQUEST_TIMEOUT_MS
MAX_OUTPUT_TOKENS = 1_024  # the project's max_completion_tokens
BASH_OUTPUT_MAX_CHARS = 4_000
FLAG_SETTINGS = {
    "attribution": {"commit": "", "pr": ""},
    "includeGitInstructions": False,
    "bashOutputMaxChars": BASH_OUTPUT_MAX_CHARS,
}
SAFE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ENABLED = "1"
STREAM_FILE = "stream.jsonl"  # what Claude Code printed
STDERR_FILE = "stderr.txt"
TASK_FOLDER = "task"
RELAY_TIMEOUT_S = 600
BAD_GATEWAY = 502  # what the relay answers when the proxy cannot be reached
# Headers a relay must not copy: they describe one hop, not the message.
HOP_HEADERS = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
)


class ClaudeNotInstalledError(Exception):
    """Raised when the pinned Claude Code cannot be found or installed."""


@dataclass(frozen=True)
class ClaudeRun:
    """One finished Claude Code run.

    Attributes:
        folder: The run's folder (task/, config/, home/, tmp/, the logs).
        exit_code: Claude Code's exit status (None when it was stopped).
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


def install_folder() -> Path:
    """Where this chapter installs Claude Code.

    Returns:
        ``COURSE_HOME/21-calls-to-rows/claude-code``.
    """
    return chapter_home() / INSTALL_FOLDER


def installed_version(binary: Path) -> str:
    """Ask a ``claude`` binary its version, from an empty environment.

    Args:
        binary: The executable.

    Returns:
        Its first word, such as ``2.1.290``.
    """
    answer = subprocess.run(
        [str(binary), "--version"],
        env={"PATH": SAFE_PATH, "HOME": str(install_folder())},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return answer.stdout.split(" ", 1)[0].strip()


def find_claude(install: bool = True) -> Path:
    """Find the pinned Claude Code, installing it with npx when asked.

    Args:
        install: False to only look.

    Returns:
        The ``claude`` executable, at version ``CLAUDE_VERSION``.

    Raises:
        ClaudeNotInstalledError: If it is missing, or another version.
    """
    named = os.environ.get(CLAUDE_BIN_ENV)
    folder = install_folder()
    path_file = folder / PATH_FILE
    if named:
        binary = Path(named)
    elif path_file.is_file():
        binary = Path(path_file.read_text(encoding="utf-8").strip())
    elif install:
        binary = npx_install(folder)
    else:
        raise ClaudeNotInstalledError(f"no Claude Code under {folder}")
    if not binary.is_file():
        raise ClaudeNotInstalledError(f"{binary} is not a file")
    version = installed_version(binary)
    if version != CLAUDE_VERSION:
        reason = f"{binary} reports {version!r}; this chapter is pinned to "
        raise ClaudeNotInstalledError(reason + CLAUDE_VERSION)
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
    (folder / "npmrc").write_text("", encoding="utf-8")  # your npm settings unread
    env = {
        **os.environ,
        "npm_config_userconfig": str(folder / "npmrc"),
        "npm_config_cache": str(folder / "npm-cache"),
        "npm_config_registry": NPM_REGISTRY,
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
    }
    print(f"installing {CLAUDE_PACKAGE} with npx into {folder}", flush=True)
    answer = subprocess.run(
        [npx, "--yes", "--package", CLAUDE_PACKAGE, "-c", "command -v claude"],
        env=env,
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT_S,
        check=False,
    )
    if answer.returncode != 0 or not answer.stdout.strip():
        raise ClaudeNotInstalledError(f"npx failed: {answer.stderr.strip()[-300:]}")
    binary = Path(answer.stdout.strip())
    (folder / PATH_FILE).write_text(f"{binary}\n", encoding="utf-8")
    return binary


def clean_environment(
    folder: Path, base_url: str, api_key: str, reminder: str | None
) -> dict[str, str]:
    """Build the only environment Claude Code gets.

    Args:
        folder: The run's folder (home/, tmp/ and config/ are made in it).
        base_url: The proxy's root (Claude Code adds ``/v1/messages``).
        api_key: The capture session's id.
        reminder: ``CLAUDE_CODE_TOTAL_TOKENS_REMINDER``, or None to leave it
            unset (Claude Code's default).

    Returns:
        The environment.
    """
    user = os.environ.get("USER", "learner")
    env = {
        "PATH": SAFE_PATH,
        "HOME": str(folder / "home"),
        "TMPDIR": str(folder / "tmp"),
        "USER": user,
        "LOGNAME": user,
        "SHELL": "/bin/bash",
        "LANG": "en_US.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": ENABLED,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_MODEL": MODEL_NAME,
        # Any "small fast model" call goes to the same model, as in the project.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL_NAME,
        "CLAUDE_CONFIG_DIR": str(folder / "config"),
        "CLAUDE_CODE_TMPDIR": str(folder / "tmp"),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
        "API_TIMEOUT_MS": str(API_TIMEOUT_MS),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": ENABLED,
        "DISABLE_TELEMETRY": ENABLED,
        "DISABLE_ERROR_REPORTING": ENABLED,
        "DISABLE_AUTOUPDATER": ENABLED,
    }
    if reminder is not None:
        env[REMINDER_ENV] = reminder
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
        "Bash",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--settings",
        json.dumps(FLAG_SETTINGS),
        "--permission-mode",
        "bypassPermissions",
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
    shutil.rmtree(folder, ignore_errors=True)
    for name in (TASK_FOLDER, "config", "home", "tmp"):
        (folder / name).mkdir(parents=True)
    task = folder / TASK_FOLDER
    shutil.copy(database, task / "data.db")
    return task


def run_claude(
    binary: Path,
    folder: Path,
    instruction: str,
    base_url: str,
    api_key: str,
    reminder: str | None,
) -> ClaudeRun:
    """Run Claude Code once, headless, in a prepared run folder.

    Args:
        binary: The ``claude`` executable.
        folder: The run's folder, from ``prepare_run``.
        instruction: The task.
        base_url: Where Claude Code sends its calls (the proxy, or a relay).
        api_key: The capture session's id.
        reminder: The total-tokens note's setting, or None for the default.

    Returns:
        The finished run; its log is ``stream.jsonl`` in the folder.
    """
    env = clean_environment(folder, base_url, api_key, reminder)
    started = time.perf_counter()
    with (
        (folder / STREAM_FILE).open("w", encoding="utf-8") as out,
        (folder / STDERR_FILE).open("w", encoding="utf-8") as err,
    ):
        try:
            done = subprocess.run(
                command(binary, instruction),
                cwd=folder / TASK_FOLDER,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=RUN_TIMEOUT_S,
                check=False,
            )
            exit_code: int | None = done.returncode
        except subprocess.TimeoutExpired:  # subprocess.run has killed it
            exit_code = None
    return ClaudeRun(folder, exit_code, time.perf_counter() - started)


@dataclass
class Relay:
    """A recording relay: Claude Code on one side, the proxy on the other.

    Attributes:
        upstream: The proxy's root.
        requests: Every POST body received, as JSON, in order.
    """

    upstream: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> str:
        """Listen on a free port of this machine.

        Returns:
            The relay's root, to give Claude Code as its base URL.
        """
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                relay.forward(self, None)

            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers.get("Content-Length") or 0)
                relay.forward(self, self.rfile.read(size))

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Stay quiet: the relay is not part of the output."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def forward(self, handler: BaseHTTPRequestHandler, body: bytes | None) -> None:
        """Pass one request to the proxy and its answer back, keeping the body.

        Args:
            handler: The request Claude Code made.
            body: Its body, or None for a GET.
        """
        if body:
            try:
                with self.lock:
                    self.requests.append(json.loads(body))
            except json.JSONDecodeError:
                pass  # not JSON: relayed, not kept
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
                status, kind, data = answer.status, answer.headers, answer.read()
        except urllib.error.HTTPError as error:  # the proxy answered: a refusal
            status, kind, data = error.code, error.headers, error.read()
        except (urllib.error.URLError, OSError) as error:  # nobody answered
            status, kind = BAD_GATEWAY, {"Content-Type": "text/plain"}
            data = f"relay: cannot reach {self.upstream}: {error}".encode()
        handler.send_response(status)
        handler.send_header("Content-Type", kind.get("Content-Type", "text/plain"))
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def stop(self) -> None:
        """Stop listening."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
