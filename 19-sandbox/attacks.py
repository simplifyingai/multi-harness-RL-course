# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Try to escape the box, one attempt at a time, and print what happened.

``uv run attacks.py [--profile project|v1] [--probe PATH ...]``

Three parts, in this order:

1. **Decoys first.** Fake secrets under ``$CH/decoys`` (a key, cloud
   credentials, Claude Code's login files, a ``.env``, a gold-answers file)
   are read inside a box built with the decoy folder as its "home". Every one
   must be denied. If any is readable, the script stops before it touches a
   real path.
2. **Escape attempts** against your REAL files: writes outside the task
   folder, reads of your real ssh key, cloud credentials, Claude Code's
   login, a ``.env`` file, the gold answers, the keychain, the internet.
   A probe of a real secret discards what it reads (``cat FILE >/dev/null``)
   and prints only whether it worked, so no secret can reach the screen even
   if the box were broken. Each real path is resolved outside the box, and the
   fake ``$HOME`` is shown on a separate line: two layers, two lines.
3. **The agent's own work**, which the box must still allow: ``pwd``, a
   query, and a write of ``answer.txt`` with Pi's write tool, then the grade.

A throwaway web server on 127.0.0.1 stands in for chapter 18's engine, the
one place on the network the box may reach.
"""

from __future__ import annotations

import argparse
import errno
import http.server
import json
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from decoys import Decoys, make_decoys
from grade_safe import grade_safe
from runner import Rollout, new_rollout, run_in_box
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

ALLOWED = "allowed"
DENIED = "denied"
# Outcome of a check made outside the box.
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

# The engine port chapter 18 uses; the fallback if another program holds it.
ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8100
FALLBACK_PORT = 8119
ANY_FREE_PORT = 0
HEALTH_PATH = "/health"
HEALTH_BODY = {"status": "ok", "stands_in_for": "chapter 18's engine"}

# The keychain item the project protects; queried by service name only, and
# never with -w or -g, so not even its attributes are printed.
KEYCHAIN_SERVICE = "Claude Code-credentials"
KEYCHAIN_NOT_FOUND = 44  # security's exit status for "could not be found"

EXTERNAL_HOST_URL = "https://huggingface.co"
EXTERNAL_IP_URL = "http://1.1.1.1"
CURL_TIMEOUT_S = 10

EXIT_UNEXPECTED = 1
EXIT_DECOY_READABLE = 3
EXIT_SANDBOX_UNAVAILABLE = 2


@dataclass(frozen=True)
class Attempt:
    """One thing to try inside the box.

    Attributes:
        label: What is being tried, in plain words.
        command: The bash command run in the box.
        expect: What the box should do: ``"allowed"`` or ``"denied"``.
        expect_v1: What the first profile does, when it differs.
        outside: A check made outside the box (never reads a secret), or None.
        note: A short remark printed under the outcome.
        cleanup: A file to delete if the attempt wrongly succeeded.
    """

    label: str
    command: str
    expect: str
    expect_v1: str | None = None
    outside: Callable[[], str] | None = None
    note: str = ""
    cleanup: Path | None = None

    def expected(self, profile: str) -> str:
        """Return what the given profile should do with this attempt.

        Args:
            profile: ``"project"`` or ``"v1"``.

        Returns:
            ``"allowed"`` or ``"denied"``.
        """
        if profile == V1_PROFILE and self.expect_v1:
            return self.expect_v1
        return self.expect


@dataclass(frozen=True)
class Result:
    """What one attempt did.

    Attributes:
        attempt: The attempt.
        outcome: ``"allowed"`` or ``"denied"``, from the exit status.
        expected: What the profile should have done.
        outside: The outside check's result, or ``"-"``.
    """

    attempt: Attempt
    outcome: str
    expected: str
    outside: str

    @property
    def as_expected(self) -> bool:
        """Whether the box did what the profile says it does."""
        return self.outcome == self.expected


def read_probe(path: Path) -> str:
    """Build a command that reads a file and throws the content away.

    Args:
        path: The file, as a real absolute path.

    Returns:
        ``cat -- PATH >/dev/null``: only the error message (if any) and the
        exit status are ever seen.
    """
    return f"cat -- {shlex.quote(str(path))} >/dev/null"


def exists_outside(path: Path) -> Callable[[], str]:
    """Make an outside check that says whether a path exists, reading nothing.

    Args:
        path: The path.

    Returns:
        A function returning ``"present"`` or ``"absent"``.
    """
    return lambda: PRESENT if path.exists() else ABSENT


def readable_outside(path: Path) -> Callable[[], str]:
    """Make an outside check that opens a DECOY and reads it (never printed).

    Args:
        path: A decoy file.

    Returns:
        A function returning ``"readable"`` or the error.
    """

    def check() -> str:
        try:
            path.read_bytes()
        except OSError as exc:
            return exc.strerror or str(exc)
        return READABLE

    return check


def command_outside(argv: list[str]) -> Callable[[], str]:
    """Make an outside check that runs a command and reports its exit status.

    Args:
        argv: The command; its output is discarded.

    Returns:
        A function returning ``"exit N"``.
    """

    def check() -> str:
        try:
            done = subprocess.run(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=COMMAND_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"failed: {exc}"
        return f"exit {done.returncode}"

    return check


def http_outside(url: str) -> Callable[[], str]:
    """Make an outside check that fetches a URL and reports the HTTP status.

    Args:
        url: The address.

    Returns:
        A function returning ``"HTTP 200"`` or curl's failure.
    """

    def check() -> str:
        argv = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}"]
        argv += ["--max-time", str(CURL_TIMEOUT_S), url]
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"failed: {exc}"
        if done.returncode != 0:
            return f"curl exit {done.returncode}"
        return f"HTTP {done.stdout.strip()}"

    return check


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Answers ``GET /health``, standing in for chapter 18's engine."""

    # Every request served, shared by the handler instances http.server makes.
    hits: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # noqa: N802 (the name http.server calls)
        """Reply to a GET with the stand-in's health record."""
        HealthHandler.hits.append(f"{self.command} {self.path}")
        body = json.dumps(HEALTH_BODY).encode()
        self.send_response(200 if self.path == HEALTH_PATH else 404)
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
    """List one read per decoy; every one must be denied.

    Args:
        decoys: Where the decoys are.

    Returns:
        The attempts.
    """
    return [
        Attempt(
            f"decoy {label}",
            read_probe(path),
            DENIED,
            outside=readable_outside(path),
        )
        for label, path in decoys.files().items()
    ]


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
    repo_file = HERE / "escape.txt"
    home_file = home / "escape.txt"
    tmp_file = Path("/private/tmp/mhrl-course-19-escape.txt")
    neighbour_answer = neighbour.workdir / ANSWER_FILE
    ssh_key = home / ".ssh" / "id_ed25519"
    attempts = [
        Attempt("write answer.txt in the task folder", "echo 30 > answer.txt", ALLOWED),
        Attempt(
            "write in the private TMPDIR", 'echo x > "$TMPDIR/scratch.txt"', ALLOWED
        ),
        Attempt(
            "write ~/escape.txt (your real home)",
            f"echo x > {shlex.quote(str(home_file))}",
            DENIED,
            cleanup=home_file,
        ),
        Attempt(
            "write into this repository",
            f"echo x > {shlex.quote(str(repo_file))}",
            DENIED,
            cleanup=repo_file,
        ),
        Attempt(
            "write into the shared /tmp",
            f"echo x > {tmp_file}",
            DENIED,
            cleanup=tmp_file,
        ),
        Attempt(
            "write answer.txt into the neighbour's folder",
            f"echo 99 > ../{neighbour.name}/{ANSWER_FILE}",
            DENIED,
        ),
        Attempt(
            "read the neighbour's answer.txt",
            f"cat ../{neighbour.name}/{ANSWER_FILE}",
            ALLOWED,
            expect_v1=DENIED,
            outside=exists_outside(neighbour_answer),
            note="the project's profile allows this; v1 hid it, and broke tools",
        ),
        Attempt(
            "layer 1: ~ is a fake HOME",
            'echo "HOME=$HOME"; cat ~/.ssh/id_ed25519 >/dev/null',
            DENIED,
            note="the fake home has no .ssh folder at all",
        ),
        Attempt(
            "layer 2: your real ssh key",
            read_probe(ssh_key),
            DENIED,
            outside=exists_outside(ssh_key),
        ),
        Attempt(
            "does your real ssh key appear to exist?",
            f"test -e {shlex.quote(str(ssh_key))}",
            DENIED,
            outside=exists_outside(ssh_key),
            note="test -e fails: inside the box, stat is denied too",
        ),
        Attempt(
            "list your real ~/.ssh folder",
            f"ls {shlex.quote(str(ssh_key.parent))} >/dev/null",
            DENIED,
            outside=exists_outside(ssh_key.parent),
        ),
    ]
    for label, relative in (
        ("your cloud credentials", ".aws/credentials"),
        ("Claude Code's login (~/.claude.json)", ".claude.json"),
        ("Claude Code's login file", ".claude/.credentials.json"),
    ):
        path = home / relative
        attempts.append(
            Attempt(label, read_probe(path), DENIED, outside=exists_outside(path))
        )
    if decoys.dot_env.is_relative_to(home):
        attempts.append(
            Attempt(
                "a .env file elsewhere under your home",
                read_probe(decoys.dot_env),
                DENIED,
                outside=exists_outside(decoys.dot_env),
                note="the decoy .env: the real rule covers every .env under ~",
            )
        )
    for path in probes:
        attempts.append(
            Attempt(
                f"your file {path.name}",
                read_probe(path),
                DENIED,
                outside=exists_outside(path),
            )
        )
    attempts += [
        Attempt(
            "read the gold answers (the task record)",
            read_probe(TASK_FILE),
            DENIED,
            outside=exists_outside(TASK_FILE),
            note="reading it would be reward hacking",
        ),
        Attempt(
            "ask the keychain for Claude Code's login",
            f"security find-generic-password -s {shlex.quote(KEYCHAIN_SERVICE)} "
            ">/dev/null 2>&1",
            DENIED,
            outside=command_outside(
                ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE]
            ),
            note=f"exit {KEYCHAIN_NOT_FOUND} means 'could not be found'",
        ),
        Attempt(
            "reach the internet by name",
            f"curl -sS -o /dev/null --max-time {CURL_TIMEOUT_S} {EXTERNAL_HOST_URL}",
            DENIED,
            outside=http_outside(EXTERNAL_HOST_URL),
        ),
        Attempt(
            "reach the internet by address",
            f"curl -sS -o /dev/null --max-time {CURL_TIMEOUT_S} {EXTERNAL_IP_URL}",
            DENIED,
            outside=http_outside(EXTERNAL_IP_URL),
        ),
        Attempt(
            "reach the engine on localhost",
            f"curl -sS --max-time {CURL_TIMEOUT_S} "
            f"http://{ENGINE_HOST}:{port}{HEALTH_PATH}",
            ALLOWED,
            outside=http_outside(f"http://{ENGINE_HOST}:{port}{HEALTH_PATH}"),
            note="a throwaway server standing in for chapter 18's engine",
        ),
        Attempt("see the environment", "env | sort", ALLOWED),
    ]
    return attempts


def work_attempts() -> list[Attempt]:
    """List the agent's own work, which the box must allow.

    Returns:
        The attempts; the write tool comes before any other write of
        ``answer.txt``, because it only fails for a file that is not there yet.
    """
    tool = HERE / "write_tool.py"
    python = shlex.quote(sys.executable)
    task = load_task()
    return [
        Attempt(
            "print the working folder",
            "pwd",
            ALLOWED,
            note="pwd works under both profiles: it never reads the parent folders",
        ),
        Attempt(
            "look at the parent folder",
            "ls .. >/dev/null",
            ALLOWED,
            expect_v1=DENIED,
        ),
        Attempt(
            "query the database",
            f'sqlite3 data.db "{task.gold_sql}"',
            ALLOWED,
            note="the agent finds this query itself; here it is given",
        ),
        Attempt(
            "write answer.txt with Pi's write tool",
            f"{python} {shlex.quote(str(tool))} {ANSWER_FILE} {task.answer}",
            ALLOWED,
            expect_v1=DENIED,
        ),
        Attempt(
            "write a file with the shell",
            "echo 30 > shell.txt",
            ALLOWED,
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


def run_attempt(rollout: Rollout, attempt: Attempt, index: int) -> Result:
    """Run one attempt in the box and print its command and outcome.

    Args:
        rollout: The rollout to run it in.
        attempt: The attempt.
        index: Its number, for the log file's name.

    Returns:
        What happened.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    outside = attempt.outside() if attempt.outside else NOT_RUN
    outcome, output = run_in_box(
        rollout,
        ["/bin/bash", "-c", attempt.command],
        label=f"{index:02d}",
        timeout_s=COMMAND_TIMEOUT_S,
    )
    measured = ALLOWED if outcome.exit_code == 0 else DENIED
    if attempt.cleanup and attempt.cleanup.exists():
        attempt.cleanup.unlink()
    result = Result(attempt, measured, attempt.expected(rollout.profile), outside)
    flag = (
        "" if result.as_expected else f"   << UNEXPECTED (expected {result.expected})"
    )
    print(f"  {attempt.label}")
    print(f"    $ {shown(attempt.command)}")
    for line in shown(output).splitlines():
        print(f"    | {line}")
    status = "timed out" if outcome.timed_out else f"exit {outcome.exit_code}"
    print(f"    {measured} ({status}){flag}")
    if outside != NOT_RUN:
        print(f"    outside the box: {outside}")
    if attempt.note:
        print(f"    note: {attempt.note}")
    return result


def run_section(rollout: Rollout, attempts: list[Attempt], start: int) -> list[Result]:
    """Run a list of attempts in one rollout.

    Args:
        rollout: The rollout.
        attempts: The attempts.
        start: The first attempt's number.

    Returns:
        Their results.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    return [
        run_attempt(rollout, attempt, start + offset)
        for offset, attempt in enumerate(attempts)
    ]


def print_table(results: list[Result]) -> None:
    """Print the allowed/denied table.

    Args:
        results: Every result, in order.
    """
    width = max(len(result.attempt.label) for result in results)
    print(f"  {'attempt':<{width}}  {'outside the box':<16}  {'in the box':<10}  check")
    for result in results:
        check = "as expected" if result.as_expected else "UNEXPECTED"
        print(
            f"  {result.attempt.label:<{width}}  {result.outside:<16}  "
            f"{result.outcome:<10}  {check}"
        )


def main() -> None:
    """Run the decoy check, the escape attempts and the agent's work."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default=PROJECT_PROFILE)
    parser.add_argument(
        "--probe",
        type=Path,
        action="append",
        default=[],
        help="another real secret file to probe (only the outcome is printed)",
    )
    parser.add_argument(
        "--decoys-only",
        action="store_true",
        help="prove the profile on the decoys and stop before any real path",
    )
    args = parser.parse_args()
    profile = args.profile
    probes = [path.expanduser().resolve() for path in args.probe]
    for line in legend():
        print(line)
    print(f"profile  {profile} ({PROFILES[profile].name})")
    print()
    try:
        exit_code = run_all(profile, probes, decoys_only=args.decoys_only)
    except SandboxError as exc:
        print(f"refused: {exc}; nothing ran")
        sys.exit(EXIT_SANDBOX_UNAVAILABLE)
    sys.exit(exit_code)


def run_all(profile: str, probes: list[Path], *, decoys_only: bool = False) -> int:
    """Run the three parts and print the table.

    Args:
        profile: Which profile the box uses.
        probes: Extra real files to probe.
        decoys_only: Stop after the decoy check.

    Returns:
        The script's exit status: 0 when the box did exactly what the profile
        says, 3 when a decoy was readable (real paths untouched), 1 otherwise.

    Raises:
        SandboxError: If the box cannot be applied.
    """
    decoys = make_decoys()
    print("1. Decoys first: fake secrets, the same rules, a decoy 'home'")
    decoy = new_rollout(
        DECOY_ROLLOUT, profile=profile, home=decoys.home, tasks_dir=decoys.tasks_dir
    )
    decoy_results = run_section(decoy, decoy_attempts(decoys), 1)
    readable = [result for result in decoy_results if result.outcome != DENIED]
    if readable:
        print(f"STOP: {len(readable)} decoy(s) readable inside the box.")
        print("The real paths were NOT probed. Fix the profile first.")
        return EXIT_DECOY_READABLE
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
        escapes = run_section(
            attack, escape_attempts(neighbour, decoys, port, probes), 10
        )
    print(f"  the stand-in served: {', '.join(HealthHandler.hits) or 'nothing'}")
    print()

    # A fresh task folder, so answer.txt is not there yet for the write tool.
    work = new_rollout(WORK_ROLLOUT, profile=profile)
    print("3. The agent's own work, which the box must allow")
    done = run_section(work, work_attempts(), 50)
    grade = grade_safe(load_task(), work.workdir)
    verdict = "correct" if grade.correct else f"wrong: {grade.reason}"
    allowed = sum(result.outcome == ALLOWED for result in done)
    print(f"  the agent's work: {allowed} of {len(done)} steps allowed")
    print(f"  grade (outside the box): {verdict} · reward {grade.reward}")
    print()

    results = escapes + done
    print(f"Allowed/denied table · profile {profile}")
    print_table(results)
    surprises = [r for r in decoy_results + results if not r.as_expected]
    print(f"{len(surprises)} unexpected results")
    return EXIT_UNEXPECTED if surprises else 0


if __name__ == "__main__":
    main()
