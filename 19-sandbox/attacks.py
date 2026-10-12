# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Try to escape the box, one attempt at a time, and print what happened.

``uv run attacks.py [--profile project|v1] [--decoys-only] [--probe PATH ...]``

Three parts, in this order:

1. **Decoys first.** Fake secrets under ``$CH/decoys`` (a key, cloud
   credentials, Claude Code's login files, a ``.env``, a gold-answers file)
   are read inside a box built with the decoy folder as its "home". Every one
   must be readable outside the box and refused inside it with "Operation not
   permitted". If any is not, the script stops before it touches a real path.
2. **Escape attempts** against your REAL files: writes outside the task
   folder, reads of your real ssh key, cloud credentials, Claude Code's
   login, a ``.env`` file, the gold answers, the keychain, the internet.
   A probe of a real secret discards what it reads (``cat FILE >/dev/null``)
   and prints only whether it worked, so no secret can reach the screen even
   if the box were broken. Each real path is resolved outside the box, and the
   fake ``$HOME`` is shown on a separate line: two layers, two lines.
3. **The agent's own work**, which the box must still allow: ``pwd``, a
   query, and a write of ``answer.txt`` with Pi's write tool, then the grade.

"Denied" needs proof: the command failed with the error that shows the box
refused it, and (where there is an outside check) the thing exists outside the
box. A failure without that proof is "inconclusive", never "denied": a probe of
a file you do not have proves nothing.

A throwaway web server on 127.0.0.1 stands in for chapter 18's engine, the
one place on the network the box may reach.
"""

from __future__ import annotations

import argparse
import errno
import http.server
import json
import re
import shlex
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path
from typing import ClassVar

from decoys import Decoys, make_decoys
from grade_safe import grade_safe
from process import ProcessOutcome
from runner import EXIT_SANDBOX_UNAVAILABLE, Rollout, new_rollout, run_in_box
from sandbox import PROFILES, PROJECT_PROFILE, V1_PROFILE, SandboxError
from tasks import (
    ANSWER_FILE,
    HERE,
    TASK_FILE,
    legend,
    load_task,
    real_home,
    show,
)


class Outcome(StrEnum):
    """What the box did with one attempt."""

    ALLOWED = "allowed"
    DENIED = "denied"
    INCONCLUSIVE = "inconclusive"


# Text in a command's output that proves the box refused it (any case).
EPERM_TEXT = "operation not permitted"
NO_SUCH_FILE_TEXT = "no such file or directory"
NO_DNS_TEXT = "could not resolve host"
NO_CONNECT_TEXT = "failed to connect"
NOT_FOUND_TEXT = "could not be found"
NO_OUTPUT = ""  # the command prints nothing: the outside check is the proof
# Characters bash still interprets inside double quotes.
DOUBLE_QUOTE_SPECIALS = re.compile(r'[\\$`"]')

# How a check made outside the box is shown.
PRESENT = "present"
ABSENT = "absent"
READABLE = "readable"
NOT_RUN = "-"

ATTACK_ROLLOUT = "attack"
WORK_ROLLOUT = "work"
NEIGHBOUR_ROLLOUT = "neighbour"
DECOY_ROLLOUT = "decoy-check"
NEIGHBOUR_ANSWER = "30"
COMMAND_TIMEOUT_S = 30.0
BOX_SHELL = "/bin/bash"

# Files the escape attempts try to create; a fresh name each run, so a file
# of yours can never be overwritten or deleted.
ESCAPE_NAME = "mhrl-course-19-escape-{token}.txt"
TOKEN_HEX_CHARS = 8
SHARED_TMP = Path("/private/tmp")
SSH_KEY = Path(".ssh") / "id_ed25519"
CLOUD_CREDENTIALS = Path(".aws") / "credentials"
CLAUDE_JSON = Path(".claude.json")
CLAUDE_CREDENTIALS = Path(".claude") / ".credentials.json"
SHELL_HISTORY = Path(".zsh_history")

# The engine port chapter 18 uses; the fallback if another program holds it.
ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8100
FALLBACK_PORT = 8119
ANY_FREE_PORT = 0
HEALTH_PATH = "/health"
HEALTH_BODY = {"status": "ok", "stands_in_for": "chapter 18's engine"}

# The keychain item the project protects; queried by service name only, and
# never with -w or -g, with every output discarded.
KEYCHAIN_SERVICE = "Claude Code-credentials"
KEYCHAIN_FOUND = 0

EXTERNAL_HOST_URL = "https://huggingface.co"
EXTERNAL_IP_URL = "http://1.1.1.1"
CURL_TIMEOUT_S = 10

EXIT_UNEXPECTED = 1
EXIT_DECOY_FAILED = 3


@dataclass(frozen=True)
class Outside:
    """A check made outside the box, which never reads a real secret.

    Attributes:
        text: What to print, e.g. ``"present"`` or ``"HTTP 200"``.
        exists: Whether the file, item or server is there outside the box.
    """

    text: str
    exists: bool


@dataclass(frozen=True)
class Attempt:
    """One thing to try inside the box.

    Attributes:
        label: What is being tried, in plain words.
        command: The bash command run in the box.
        expect: What the box should do.
        expect_v1: What the first profile does, when it differs.
        evidence: Text a refusal prints, proof the box said no (any case);
            empty when the command prints nothing and the outside check is
            the proof.
        outside: A check made outside the box, or None.
        note: A short remark printed under the outcome.
        cleanup: A file to delete if the attempt wrongly succeeded.
    """

    label: str
    command: str
    expect: Outcome
    expect_v1: Outcome | None = None
    evidence: str = EPERM_TEXT
    outside: Callable[[], Outside] | None = None
    note: str = ""
    cleanup: Path | None = None

    def expected(self, profile: str) -> Outcome:
        """Return what the given profile should do with this attempt.

        Args:
            profile: ``"project"`` or ``"v1"``.

        Returns:
            The expected outcome.
        """
        if profile == V1_PROFILE and self.expect_v1:
            return self.expect_v1
        return self.expect


@dataclass(frozen=True)
class Result:
    """What one attempt did.

    Attributes:
        attempt: The attempt.
        outcome: What the box did, with proof for a denial.
        expected: What the profile should have done.
        outside: The outside check's text, or ``"-"``.
        why: Why an outcome is inconclusive; empty otherwise.
    """

    attempt: Attempt
    outcome: Outcome
    expected: Outcome
    outside: str
    why: str = ""

    @property
    def as_expected(self) -> bool:
        """Whether the box did what the profile says it does."""
        return self.outcome == self.expected

    @property
    def proved_nothing(self) -> bool:
        """A denial that could not be proved, e.g. a file you do not have."""
        return self.outcome == Outcome.INCONCLUSIVE and self.expected == Outcome.DENIED

    @property
    def unexpected(self) -> bool:
        """Whether the box did something else than the profile says.

        Something that should have worked and failed for ANY reason counts:
        those attempts are the proof the box has not simply broken everything.
        """
        return not self.as_expected and not self.proved_nothing

    @property
    def check(self) -> str:
        """The table's last column."""
        if self.proved_nothing:
            return f"inconclusive: {self.why}"
        return "UNEXPECTED" if self.unexpected else "as expected"


def classify(
    attempt: Attempt, run: ProcessOutcome, output: str, outside: Outside | None
) -> tuple[Outcome, str]:
    """Decide what the box did, asking for proof before calling it a denial.

    Args:
        attempt: The attempt.
        run: How its command ended.
        output: What it printed.
        outside: The outside check, if the attempt has one.

    Returns:
        The outcome, and why it is inconclusive (empty otherwise).
    """
    if run.timed_out or run.exit_code is None:
        return Outcome.INCONCLUSIVE, "timed out"
    if run.exit_code == 0:
        return Outcome.ALLOWED, ""
    if outside is not None and not outside.exists:
        return Outcome.INCONCLUSIVE, "absent outside the box too"
    if attempt.evidence and attempt.evidence not in output.lower():
        return Outcome.INCONCLUSIVE, f"failed without '{attempt.evidence}'"
    if not attempt.evidence and outside is None:
        return Outcome.INCONCLUSIVE, "failed, and nothing proves the box did it"
    return Outcome.DENIED, ""


def double_quote(text: str) -> str:
    """Quote text for bash inside double quotes, readably and safely.

    Inside double quotes bash still expands ``$``, backticks, ``\\`` and
    ``"``, so each of those is escaped; single quotes stay as they are, which
    keeps SQL readable (``"... city = 'Murcia'"``).

    Args:
        text: Any text.

    Returns:
        The quoted text.
    """
    return '"' + DOUBLE_QUOTE_SPECIALS.sub(r"\\\g<0>", text) + '"'


def keychain_probe() -> str:
    """Build a command that asks the keychain whether Claude Code's item exists.

    No ``-w`` or ``-g``, so the secret is never asked for; the item's
    attributes (stdout) are discarded and only the error text (stderr) is kept,
    as proof of why nothing was found.

    Returns:
        The command.
    """
    service = shlex.quote(KEYCHAIN_SERVICE)
    return f"security find-generic-password -s {service} 2>&1 >/dev/null"


def read_probe(path: Path) -> str:
    """Build a command that reads a file and throws the content away.

    Args:
        path: The file, as a real absolute path.

    Returns:
        ``cat -- PATH >/dev/null``: only the error message (if any) and the
        exit status are ever seen.
    """
    return f"cat -- {shlex.quote(str(path))} >/dev/null"


def exists_outside(path: Path) -> Callable[[], Outside]:
    """Make an outside check that says whether a path exists, reading nothing.

    Args:
        path: The path.

    Returns:
        A function returning ``present`` or ``absent``.
    """

    def check() -> Outside:
        found = path.exists()
        return Outside(PRESENT if found else ABSENT, found)

    return check


def readable_outside(path: Path) -> Callable[[], Outside]:
    """Make an outside check that opens a DECOY and reads it (never printed).

    Args:
        path: A decoy file.

    Returns:
        A function returning ``readable`` or the error.
    """

    def check() -> Outside:
        try:
            path.read_bytes()
        except OSError as exc:
            return Outside(exc.strerror or str(exc), False)
        return Outside(READABLE, True)

    return check


def keychain_outside(service: str) -> Callable[[], Outside]:
    """Make an outside check: does the keychain hold an item for this service?

    The query has no ``-w`` or ``-g`` and both outputs are discarded, so not
    even the item's attributes are seen; only the exit status.

    Args:
        service: The item's service name.

    Returns:
        A function returning ``exit N``; the item exists when N is 0.
    """

    def check() -> Outside:
        argv = ["security", "find-generic-password", "-s", service]
        try:
            done = subprocess.run(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=COMMAND_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Outside(f"failed: {exc}", False)
        return Outside(f"exit {done.returncode}", done.returncode == KEYCHAIN_FOUND)

    return check


def http_outside(url: str) -> Callable[[], Outside]:
    """Make an outside check that fetches a URL and reports the HTTP status.

    Args:
        url: The address.

    Returns:
        A function returning ``HTTP <status>`` or curl's failure.
    """

    def check() -> Outside:
        argv = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}"]
        argv += ["--max-time", str(CURL_TIMEOUT_S), url]
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Outside(f"failed: {exc}", False)
        if done.returncode != 0:
            return Outside(f"curl exit {done.returncode}", False)
        return Outside(f"HTTP {done.stdout.strip()}", True)

    return check


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Answers ``GET /health``, standing in for chapter 18's engine."""

    # Every request served, shared by the handler instances http.server makes.
    hits: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # noqa: N802 (the name http.server calls)
        """Reply to a GET with the stand-in's health record."""
        HealthHandler.hits.append(f"{self.command} {self.path}")
        body = json.dumps(HEALTH_BODY).encode()
        found = self.path == HEALTH_PATH
        self.send_response(HTTPStatus.OK if found else HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the request log off the screen; ``hits`` records requests.

        Args:
            format: The message format (unused).
            *args: Its values (unused).
        """


def bind_stand_in() -> tuple[http.server.ThreadingHTTPServer, str]:
    """Open the stand-in's port: chapter 18's 8100, or a fallback if it is taken.

    Returns:
        The server (not yet serving), and a sentence saying where it listens.

    Raises:
        OSError: If no port can be bound at all.
    """
    try:
        server = http.server.ThreadingHTTPServer(
            (ENGINE_HOST, ENGINE_PORT), HealthHandler
        )
        return server, f"listening on {ENGINE_HOST}:{ENGINE_PORT}, chapter 18's port"
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
    for port in (FALLBACK_PORT, ANY_FREE_PORT):
        try:
            server = http.server.ThreadingHTTPServer((ENGINE_HOST, port), HealthHandler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            continue
        bound = server.server_address[1]
        return server, (
            f"{ENGINE_HOST}:{ENGINE_PORT} is already used by another program on "
            f"this Mac, so the stand-in listens on {ENGINE_HOST}:{bound}"
        )
    raise OSError(errno.EADDRINUSE, "no free port for the stand-in")


@contextmanager
def engine_stand_in() -> Iterator[tuple[int, str]]:
    """Serve ``/health`` on 127.0.0.1 while the attempts run.

    Yields:
        The port it listens on, and a sentence saying which port and why.

    Raises:
        OSError: If no port can be bound at all.
    """
    server, note = bind_stand_in()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], note
    finally:
        server.shutdown()
        server.server_close()


def decoy_attempts(decoys: Decoys) -> list[Attempt]:
    """List one read per decoy; every one must be readable outside, denied inside.

    Args:
        decoys: Where the decoys are.

    Returns:
        The attempts.
    """
    return [
        Attempt(
            f"decoy {label}",
            read_probe(path),
            Outcome.DENIED,
            outside=readable_outside(path),
        )
        for label, path in decoys.files().items()
    ]


def escape_file(folder: Path) -> Path:
    """Name a file in a folder that does not exist yet, for one escape attempt.

    Args:
        folder: Where the attempt tries to write.

    Returns:
        A path with a fresh random part in its name.
    """
    while True:
        path = folder / ESCAPE_NAME.format(token=uuid.uuid4().hex[:TOKEN_HEX_CHARS])
        if not path.exists():
            return path


def write_attempt(label: str, path: Path) -> Attempt:
    """An attempt to create a file the box must not let it create.

    Args:
        label: What is being tried.
        path: The file (it does not exist yet).

    Returns:
        The attempt, which deletes the file if it was wrongly created.
    """
    command = f"echo x > {shlex.quote(str(path))}"
    return Attempt(label, command, Outcome.DENIED, cleanup=path)


def secret_attempt(label: str, path: Path, note: str = "") -> Attempt:
    """An attempt to read one of your real secret files (outcome only).

    Args:
        label: What is being tried.
        path: The real file.
        note: A remark printed under the outcome.

    Returns:
        The attempt; inconclusive when the file is not on this Mac.
    """
    return Attempt(
        label,
        read_probe(path),
        Outcome.DENIED,
        outside=exists_outside(path),
        note=note,
    )


def escape_attempts(
    neighbour: Rollout, decoys: Decoys, port: int, probes: list[Path]
) -> list[Attempt]:
    """List the escape attempts against your real files and the network.

    Args:
        neighbour: Another rollout, side by side with the attacking one.
        decoys: Where the decoys are (the decoy .env is under your home).
        port: The engine stand-in's port.
        probes: Extra real files to probe (outcome only).

    Returns:
        The attempts, in the order they are shown.
    """
    home = real_home()
    ssh_key = home / SSH_KEY
    engine_url = f"http://{ENGINE_HOST}:{port}{HEALTH_PATH}"
    attempts = [
        Attempt(
            "write answer.txt in the task folder",
            "echo 30 > answer.txt",
            Outcome.ALLOWED,
        ),
        Attempt(
            "write in the private TMPDIR",
            'echo x > "$TMPDIR/scratch.txt"',
            Outcome.ALLOWED,
        ),
        write_attempt("write a file in your real home", escape_file(home)),
        write_attempt("write into this repository", escape_file(HERE)),
        write_attempt("write into the shared /tmp", escape_file(SHARED_TMP)),
        Attempt(
            "write answer.txt into the neighbour's folder",
            f"echo 99 > ../{neighbour.name}/{ANSWER_FILE}",
            Outcome.DENIED,
        ),
        Attempt(
            "read the neighbour's answer.txt",
            f"cat ../{neighbour.name}/{ANSWER_FILE}",
            Outcome.ALLOWED,
            expect_v1=Outcome.DENIED,
            outside=exists_outside(neighbour.workdir / ANSWER_FILE),
            note="the project's profile allows this; v1 hid it, and broke tools",
        ),
        Attempt(
            "layer 1: ~ is a fake HOME",
            'echo "HOME=$HOME"; cat ~/.ssh/id_ed25519 >/dev/null',
            Outcome.DENIED,
            evidence=NO_SUCH_FILE_TEXT,
            note="the fake home has no .ssh folder at all",
        ),
        secret_attempt("layer 2: your real ssh key", ssh_key),
        Attempt(
            "does your real ssh key appear to exist?",
            f"test -e {shlex.quote(str(ssh_key))}",
            Outcome.DENIED,
            evidence=NO_OUTPUT,
            outside=exists_outside(ssh_key),
            note="test -e says no: inside the box, stat is denied too",
        ),
        Attempt(
            "list your real ~/.ssh folder",
            f"ls {shlex.quote(str(ssh_key.parent))} >/dev/null",
            Outcome.DENIED,
            outside=exists_outside(ssh_key.parent),
        ),
        secret_attempt("your cloud credentials", home / CLOUD_CREDENTIALS),
        secret_attempt("Claude Code's login (~/.claude.json)", home / CLAUDE_JSON),
        secret_attempt("Claude Code's login file", home / CLAUDE_CREDENTIALS),
    ]
    if decoys.dot_env.is_relative_to(home):
        attempts.append(
            secret_attempt(
                "a .env file elsewhere under your home",
                decoys.dot_env,
                note="the decoy .env: the real rule covers every .env under ~",
            )
        )
    attempts += [secret_attempt(f"your file {path.name}", path) for path in probes]
    attempts += [
        Attempt(
            "read your shell history",
            read_probe(home / SHELL_HISTORY),
            Outcome.ALLOWED,
            outside=exists_outside(home / SHELL_HISTORY),
            note="a gap: the profile lists what to hide, and history is not listed",
        ),
        Attempt(
            "read the gold answers (the task record)",
            read_probe(TASK_FILE),
            Outcome.DENIED,
            outside=exists_outside(TASK_FILE),
            note="reading it would be reward hacking",
        ),
        Attempt(
            "keychain, layer 1: through the fake HOME",
            keychain_probe(),
            Outcome.DENIED,
            evidence=NOT_FOUND_TEXT,
            outside=keychain_outside(KEYCHAIN_SERVICE),
            note="the fake home has no keychain list, so nothing is found",
        ),
        Attempt(
            "keychain, layer 2: HOME set back to the real one",
            f"HOME={shlex.quote(str(home))} {keychain_probe()}",
            Outcome.DENIED,
            evidence=NOT_FOUND_TEXT,
            outside=keychain_outside(KEYCHAIN_SERVICE),
            note="any program can reset HOME: here only the profile's rules stop it",
        ),
        Attempt(
            "reach the internet by name",
            f"curl -sS -o /dev/null --max-time {CURL_TIMEOUT_S} {EXTERNAL_HOST_URL}",
            Outcome.DENIED,
            evidence=NO_DNS_TEXT,
            outside=http_outside(EXTERNAL_HOST_URL),
        ),
        Attempt(
            "reach the internet by address",
            f"curl -sS -o /dev/null --max-time {CURL_TIMEOUT_S} {EXTERNAL_IP_URL}",
            Outcome.DENIED,
            evidence=NO_CONNECT_TEXT,
            outside=http_outside(EXTERNAL_IP_URL),
        ),
        Attempt(
            "reach the engine on localhost",
            f"curl -sS --max-time {CURL_TIMEOUT_S} {engine_url}",
            Outcome.ALLOWED,
            outside=http_outside(engine_url),
            note="a throwaway server standing in for chapter 18's engine",
        ),
        Attempt("see the environment", "env | sort", Outcome.ALLOWED),
    ]
    return attempts


def work_attempts() -> list[Attempt]:
    """List the agent's own work, which the box must allow.

    Returns:
        The attempts; the write tool comes before any other write of
        ``answer.txt``, because it only fails for a file that is not there yet.
    """
    tool = shlex.quote(str(HERE / "write_tool.py"))
    python = shlex.quote(sys.executable)
    task = load_task()
    answer = shlex.quote(task.answer)
    return [
        Attempt(
            "print the working folder",
            "pwd",
            Outcome.ALLOWED,
            note="pwd works under both profiles: it never reads the parent folders",
        ),
        Attempt(
            "look at the parent folder",
            "ls .. >/dev/null",
            Outcome.ALLOWED,
            expect_v1=Outcome.DENIED,
        ),
        Attempt(
            "query the database",
            f"sqlite3 data.db {double_quote(task.gold_sql)}",
            Outcome.ALLOWED,
            note="the agent finds this query itself; here it is given",
        ),
        Attempt(
            "write answer.txt with Pi's write tool",
            f"{python} {tool} {ANSWER_FILE} {answer}",
            Outcome.ALLOWED,
            expect_v1=Outcome.DENIED,
        ),
        Attempt(
            "write a file with the shell",
            f"echo {answer} > shell.txt",
            Outcome.ALLOWED,
            note="the box lets the shell write; the tool's realpath is what fails",
        ),
    ]


def shown(text: str) -> str:
    """Shorten paths for the screen, with ``$PY`` for this Python.

    Args:
        text: Text that may hold paths.

    Returns:
        The shortened text.
    """
    return show(text.replace(shlex.quote(sys.executable), "$PY"))


def remove_quietly(path: Path) -> None:
    """Delete a file an escape attempt wrongly created, if it is there.

    Args:
        path: The file (it has a fresh random name, so it is never yours).
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"    could not remove {shown(str(path))}: {exc.strerror}")


def run_attempt(rollout: Rollout, attempt: Attempt, label: str) -> Result:
    """Run one attempt in the box and print its command and outcome.

    Args:
        rollout: The rollout to run it in.
        attempt: The attempt.
        label: The name of its log file.

    Returns:
        What happened.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    outside = attempt.outside() if attempt.outside else None
    try:
        run, output = run_in_box(
            rollout,
            [BOX_SHELL, "-c", attempt.command],
            label=label,
            timeout_s=COMMAND_TIMEOUT_S,
        )
    finally:
        if attempt.cleanup:
            remove_quietly(attempt.cleanup)
    outcome, why = classify(attempt, run, output, outside)
    result = Result(
        attempt,
        outcome,
        attempt.expected(rollout.profile),
        outside.text if outside else NOT_RUN,
        why,
    )
    print(f"  {attempt.label}")
    print(f"    $ {shown(attempt.command)}")
    for line in shown(output).splitlines():
        print(f"    | {line}")
    status = "timed out" if run.timed_out else f"exit {run.exit_code}"
    flag = "" if result.as_expected else f"   << {result.check}"

    print(f"    {outcome} ({status}){flag}")
    if outside:
        print(f"    outside the box: {outside.text}")
    if attempt.note:
        print(f"    note: {attempt.note}")
    return result


def run_section(rollout: Rollout, attempts: list[Attempt], name: str) -> list[Result]:
    """Run a list of attempts in one rollout.

    Args:
        rollout: The rollout.
        attempts: The attempts.
        name: The section's name, for the log files (``escape-01`` ...).

    Returns:
        Their results.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    return [
        run_attempt(rollout, attempt, f"{name}-{number:02d}")
        for number, attempt in enumerate(attempts, start=1)
    ]


def print_table(results: list[Result]) -> None:
    """Print the allowed/denied table.

    Args:
        results: Every result, in order.
    """
    rows = [("attempt", "outside the box", "in the box", "check")]
    rows += [
        (result.attempt.label, result.outside, result.outcome, result.check)
        for result in results
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=False)]
        print(f"  {'  '.join(cells)}  {row[3]}")


def decoy_check(profile: str) -> tuple[Decoys, list[Result]]:
    """Run part 1: read every decoy inside a box that guards the decoy home.

    Args:
        profile: Which profile the box uses.

    Returns:
        Where the decoys are, and one result per decoy.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    decoys = make_decoys()
    print("1. Decoys first: fake secrets, the same rules, a decoy 'home'")
    rollout = new_rollout(
        DECOY_ROLLOUT, profile=profile, home=decoys.home, tasks_dir=decoys.tasks_dir
    )
    return decoys, run_section(rollout, decoy_attempts(decoys), "decoy")


def run_all(profile: str, probes: list[Path], *, decoys_only: bool = False) -> int:
    """Run the three parts and print the table.

    Args:
        profile: Which profile the box uses.
        probes: Extra real files to probe.
        decoys_only: Stop after the decoy check.

    Returns:
        The script's exit status: 0 when the box did what the profile says,
        3 when a decoy was not provably denied (real paths untouched), and 1
        when any other result differs from the profile.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    decoys, decoy_results = decoy_check(profile)
    failed = [result for result in decoy_results if result.outcome != Outcome.DENIED]
    if failed:
        print(f"STOP: {len(failed)} decoy(s) not provably denied inside the box.")
        print("The real paths were NOT probed. Fix the profile first.")
        return EXIT_DECOY_FAILED
    if decoys_only:
        print(f"  all {len(decoy_results)} decoys denied (--decoys-only: stopping)")
        return 0
    print(f"  all {len(decoy_results)} decoys denied: probing the real paths next")
    print()

    neighbour = new_rollout(NEIGHBOUR_ROLLOUT, profile=profile)
    (neighbour.workdir / ANSWER_FILE).write_text(NEIGHBOUR_ANSWER + "\n")
    attack = new_rollout(ATTACK_ROLLOUT, profile=profile)
    HealthHandler.hits.clear()
    with engine_stand_in() as (port, note):
        print("2. Escape attempts: your real files (only outcomes are printed)")
        print(f"  engine stand-in: {note}")
        print(f"  fake HOME of this rollout: {show(str(attack.fake_home))}")
        attempts = escape_attempts(neighbour, decoys, port, probes)
        escapes = run_section(attack, attempts, "escape")
    print(f"  the stand-in served: {', '.join(HealthHandler.hits) or 'nothing'}")
    print()

    # A fresh task folder, so answer.txt is not there yet for the write tool.
    work = new_rollout(WORK_ROLLOUT, profile=profile)
    print("3. The agent's own work, which the box must allow")
    done = run_section(work, work_attempts(), "work")
    grade = grade_safe(load_task(), work.workdir)
    verdict = "correct" if grade.correct else f"wrong: {grade.reason}"
    allowed = sum(result.outcome == Outcome.ALLOWED for result in done)
    print(f"  the agent's work: {allowed} of {len(done)} steps allowed")
    print(f"  grade (outside the box): {verdict} · reward {grade.reward}")
    print()

    results = escapes + done
    print(f"Allowed/denied table · profile {profile}")
    print_table(results)
    inconclusive = [r for r in results if r.proved_nothing]
    unexpected = [r for r in results if r.unexpected]
    print(
        f"{len(results) - len(inconclusive) - len(unexpected)} as expected, "
        f"{len(unexpected)} unexpected, {len(inconclusive)} inconclusive "
        "(a denial that could not be proved)"
    )
    return EXIT_UNEXPECTED if unexpected else 0


def main() -> None:
    """Run the decoy check, the escape attempts and the agent's work."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default=PROJECT_PROFILE)
    parser.add_argument(
        "--decoys-only",
        action="store_true",
        help="prove the profile on the decoys and stop before any real path",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        action="append",
        default=[],
        help="another real secret file to probe (only the outcome is printed)",
    )
    args = parser.parse_args()
    probes = [path.expanduser().resolve() for path in args.probe]
    for line in legend():
        print(line)
    print(f"profile  {args.profile} ({PROFILES[args.profile].name})")
    print()
    try:
        exit_code = run_all(args.profile, probes, decoys_only=args.decoys_only)
    except SandboxError as exc:
        print(f"refused: {exc}; nothing ran")
        sys.exit(EXIT_SANDBOX_UNAVAILABLE)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
