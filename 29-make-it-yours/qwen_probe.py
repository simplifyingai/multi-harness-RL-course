# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""What Qwen Code does around the model, measured before it ever meets one.

Wiring a new harness is more than a base URL and a key. Before the fourth
harness met the model, this probe measured six things about it, each run
of Qwen Code 0.24.7 inside chapter 19's sandbox with a private ``HOME``:

1. **its own switches left on**: where it tries to connect besides the
   model. ``HTTPS_PROXY`` points at a **catcher** on this machine, which
   writes down the host of every outside connection asked for and refuses
   it, so nothing leaves the machine and the box is never tested;
2. **the same, with no catcher**: only the sandbox stands in the way, and
   its own log says how many direct connections it refused;
3. **with wire.py's switches**: usage statistics, telemetry, the early
   connection to the API and session titles switched off;
4. **``--bare``** (wire.py's setup): the tools it offers, the size of its
   first request, how many requests it makes and the reply length it asks;
5. **without ``--bare``**: the same numbers;
6. **``read_file`` excluded too**: what its permission rules then do to a
   shell command that reads a file.

After every run the sandbox's refusals of that process are read from the
system log (``log show``): the catcher sees only clients that honour
``HTTPS_PROXY``, the log sees the rest.

The model is a **stand-in** on this machine that speaks OpenAI's chat API,
records every request and answers from a short script: look at the file,
write ``42`` to ``answer.txt``, then a last reply with no tool call. Sizes
are characters of JSON (chapter 23's scripted engine counts tokens; this
probe needs no tokenizer). Only Python's standard library is needed.

Run ``uv run qwen_probe.py`` after ``uv run install.py qwen-code``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from install import PINS, QWEN, HarnessInstallError, ensure, read_record
from lfm_files import chapter_home, tilde
from run_one import read_jsonl, read_qwen_log
from sandbox import SandboxError, sandbox_command, sandbox_params
from tasks import DATA_FILE, gold_folder
from wire import MODEL_NAME, QWEN_EXCLUDED, RunFolders, Wiring, wire

LOOPBACK = "127.0.0.1"
CHAT_PATH = "/v1/chat/completions"
PROBE_FOLDER = "qwen-probe"
RUN_TIMEOUT_S = 120
INSTRUCTION = (
    "There is a CSV file named data.csv in the current folder. How many rows "
    "does it have? Write ONLY the answer to a file named answer.txt."
)
SAMPLE_CSV = "a,b\n1,2\n"
LOOK = f"head -3 {DATA_FILE}"
WRITE = "echo 42 > answer.txt"
SHELL_TOOL = "run_shell_command"
# Its own switches, the ones wire.py turns off.
QWEN_SWITCHES = (
    "QWEN_USAGE_STATISTICS_ENABLED",
    "QWEN_TELEMETRY_ENABLED",
    "QWEN_CODE_DISABLE_PRECONNECT",
    "QWEN_DISABLE_AUTO_TITLE",
)
PROXY_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
NO_PROXY = "127.0.0.1,localhost"  # the stand-in model is reached directly
REFUSED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
RULE = "━━"
# The sandbox's own log of what it refused (macOS's unified log).
LOG_COMMAND = ("/usr/bin/log", "show", "--style", "compact", "--start")
LOG_PREDICATE = 'sender == "Sandbox" AND eventMessage CONTAINS "network-outbound"'
LOG_TIMEOUT_S = 60
LOG_DELAY_S = 1.0  # the log takes a moment to hold the last refusals
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DENY_RE = r"Sandbox: [^(]+\({pid}\) deny\(\d+\) network-outbound"
PROBE_KEY = "probe-key"  # the stand-in model takes any key
SWITCH_RUNS = ("1-", "2-", "3-")  # the runs read for outside connections
INCOMPLETE = "run did not complete: 0 means unobserved"
PROXY_ONLY = (
    "(through HTTPS_PROXY only; a client that ignores it is stopped by the "
    "sandbox and not seen here)"
)
HTTP_OK = 200


@dataclass(frozen=True)
class Run:
    """One way of running Qwen Code, changed from wire.py's in at most one way.

    Attributes:
        name: The run's folder name.
        title: What it shows.
        switches: Keep wire.py's switches off.
        bare: Keep ``--bare``.
        extra: A tool to exclude besides wire.py's.
        catcher: Point ``HTTPS_PROXY`` at the catcher (False: only the
            sandbox stands between it and the internet).
    """

    name: str
    title: str
    switches: bool = True
    bare: bool = True
    extra: str = ""
    catcher: bool = True

    def reader(self, probe: Probe) -> list[str]:
        """The lines that describe this run's result.

        Args:
            probe: What the run did.

        Returns:
            Outside connections for the switch runs, the refusals for the
            exclusion run, the requests for the others.
        """
        if self.name.startswith(SWITCH_RUNS):
            return outside_lines(probe)
        if self.extra:
            return refusal_lines(probe)
        return request_lines(probe)


@dataclass
class Probe:
    """What one run of Qwen Code did, as the stand-in and the catcher saw it.

    Attributes:
        name: Which run.
        requests: Every request body the stand-in model received, in order.
        outside: The first line of every outside connection asked for.
        exit_code: Qwen Code's exit status (None when the time limit hit).
        log: Its stream-json log.
        answered: What ``answer.txt`` holds after the run, or None.
        refused: Direct connections the sandbox refused its process, from
            the system log (None when the log could not be read).
    """

    name: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    outside: list[str] = field(default_factory=list)
    exit_code: int | None = None
    log: Path = Path()
    answered: str | None = None
    refused: int | None = None


def reply_for(index: int) -> dict[str, Any]:
    """The stand-in model's reply to its ``index``-th request (from 0).

    Args:
        index: Which request.

    Returns:
        An assistant message: a shell call to look, one to write, then
        (for this request and any after it) text with no tool call.
    """
    commands = (LOOK, WRITE)
    if index < len(commands):
        call = {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": SHELL_TOOL,
                "arguments": json.dumps({"command": commands[index]}),
            },
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}
    return {"role": "assistant", "content": "42"}


def chunks_for(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Cut a reply into the two stream events OpenAI's chat API would send.

    Args:
        message: The assistant message.

    Returns:
        A chunk with the whole message as its delta, then one with the
        finish reason and the usage.
    """
    delta = {key: value for key, value in message.items() if value is not None}
    if "tool_calls" in delta:
        delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
    finish = "tool_calls" if "tool_calls" in message else "stop"
    base = {"id": "probe", "object": "chat.completion.chunk", "model": MODEL_NAME}
    return [
        base | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        base
        | {
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    ]


class StandIn(ThreadingHTTPServer):
    """A stand-in model: OpenAI's chat API, every request kept."""

    def __init__(self) -> None:
        """Listen on a free loopback port."""
        super().__init__((LOOPBACK, 0), StandInHandler)
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()


class StandInHandler(BaseHTTPRequestHandler):
    """Answers chat completions from ``reply_for``, streamed or not."""

    server: StandIn

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Print nothing per request.

        Args:
            format: The message's format.
            *args: Its values.
        """

    def do_POST(self) -> None:  # noqa: N802 (http.server's name)
        """Record the request and answer it."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            index = len(self.server.requests)
            self.server.requests.append(body)
        message = reply_for(index)
        self.send_response(HTTP_OK)
        if body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks_for(message):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        finish = "tool_calls" if "tool_calls" in message else "stop"
        answer = {
            "id": "probe",
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        data = json.dumps(answer).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Catcher(socketserver.ThreadingTCPServer):
    """An HTTPS proxy that notes where each connection was going, and refuses it."""

    daemon_threads = True

    def __init__(self) -> None:
        """Listen on a free loopback port."""
        super().__init__((LOOPBACK, 0), CatcherHandler)
        self.asked: list[str] = []


class CatcherHandler(socketserver.StreamRequestHandler):
    """Reads the proxy request's first line (``CONNECT host:443 ...``), answers 403."""

    server: Catcher

    def handle(self) -> None:
        """Keep the first line, refuse the connection."""
        line = self.rfile.readline().decode("latin-1").strip()
        if line:
            self.server.asked.append(line)
        self.wfile.write(REFUSED)


@contextlib.contextmanager
def serving(server: socketserver.BaseServer) -> Iterator[str]:
    """Run a server on a thread while the block runs.

    Args:
        server: The server.

    Yields:
        Its root, ``http://127.0.0.1:<port>``.
    """
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def run_once(
    name: str, wiring: Wiring, folders: RunFolders, stand_in: StandIn, catcher: Catcher
) -> Probe:
    """Run Qwen Code once in the sandbox, and collect what it did.

    Args:
        name: Which run.
        wiring: Its command line and environment.
        folders: A fresh task folder (holding data.csv) and state folder.
        stand_in: The stand-in model (its requests are taken from it).
        catcher: The catcher (its lines are taken from it).

    Returns:
        The probe's account.

    Raises:
        HarnessInstallError: If the sandbox cannot be applied.
    """
    stand_in.requests.clear()
    catcher.asked.clear()
    try:
        params = sandbox_params(
            workdir=folders.task,
            state_dir=folders.state,
            tasks_dir=gold_folder(),
            proxy_port=wiring.proxy_port,  # the stand-in model
            extra_port=catcher.server_address[1],
        )
        argv = sandbox_command(wiring.argv, params=params)
    except SandboxError as error:
        raise HarnessInstallError(f"no sandbox, so no run: {error}") from error
    log = folders.task.parent / f"{name}.jsonl"
    exit_code: int | None
    since = datetime.now() - timedelta(seconds=1)
    with log.open("w", encoding="utf-8") as out:
        # sandbox-exec becomes Qwen Code, so this pid is its own process.
        process = subprocess.Popen(
            argv,
            cwd=folders.task,
            env=wiring.env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.DEVNULL,
        )
        try:
            exit_code = process.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code = None
    time.sleep(LOG_DELAY_S)  # late connections reach the catcher and the log
    answer = folders.task / "answer.txt"
    return Probe(
        name=name,
        requests=list(stand_in.requests),
        outside=list(catcher.asked),
        exit_code=exit_code,
        log=log,
        answered=answer.read_text().strip() if answer.is_file() else None,
        refused=sandbox_refusals(process.pid, since),
    )


def sandbox_refusals(pid: int, since: datetime) -> int | None:
    """Count the outbound connections the sandbox refused one process.

    A client that ignores ``HTTPS_PROXY`` never reaches the catcher; the
    sandbox refuses its connect, and says so in the system log.

    Args:
        pid: The process.
        since: When its run started.

    Returns:
        How many refusals the log holds for it, or None when the log could
        not be read.
    """
    command = [
        *LOG_COMMAND,
        since.strftime(LOG_TIME_FORMAT),
        "--predicate",
        LOG_PREDICATE,
    ]
    try:
        found = subprocess.run(
            command, capture_output=True, text=True, timeout=LOG_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if found.returncode != 0:
        return None
    return len(re.findall(DENY_RE.format(pid=pid), found.stdout))


def fresh_folders(root: Path, name: str) -> RunFolders:
    """Make a run's folders: a task folder with a small data.csv, a state folder.

    The state folder goes in the system's temp folder, as chapter 24 found
    a harness's state must.

    Args:
        root: The probe's folder under ``COURSE_HOME``.
        name: Which run.

    Returns:
        The folders.
    """
    task = root / name / "task"
    if task.parent.exists():
        shutil.rmtree(task.parent)
    task.mkdir(parents=True)
    (task / DATA_FILE).write_text(SAMPLE_CSV, encoding="utf-8")
    state = Path(tempfile.mkdtemp(prefix="mhrl-harness-qwen-probe-")).resolve()
    folders = RunFolders(task=task.resolve(), state=state)
    folders.make()
    return folders


def variant(
    base: Wiring,
    proxy: str | None,
    *,
    switches: bool,
    bare: bool,
    also_excluded: str = "",
) -> Wiring:
    """wire.py's Qwen Code wiring, changed in one way, with the catcher as its proxy.

    Args:
        base: wire.py's wiring.
        proxy: The catcher's root, or None for no proxy at all.
        switches: Keep wire.py's switches off (False: Qwen Code's defaults).
        bare: Keep ``--bare``.
        also_excluded: A tool to add to the excluded ones.

    Returns:
        The changed wiring.
    """
    env = {k: v for k, v in base.env.items() if switches or k not in QWEN_SWITCHES}
    if proxy is not None:
        env |= dict.fromkeys(PROXY_VARIABLES, proxy) | {"NO_PROXY": NO_PROXY}
    argv = [a for a in base.argv if bare or a != "--bare"]
    if also_excluded:
        where = argv.index("--exclude-tools") + 1
        argv[where] = ",".join([*QWEN_EXCLUDED, also_excluded])
    return Wiring(
        base.harness,
        argv,
        env,
        stdout_name=base.stdout_name,
        proxy_port=base.proxy_port,
    )


def tools_of(body: dict[str, Any]) -> list[str]:
    """The names of the tools one request offered.

    Args:
        body: A chat request.

    Returns:
        The tool names, sorted.
    """
    return sorted(t["function"]["name"] for t in body.get("tools") or [])


def size_of(body: dict[str, Any]) -> tuple[int, int]:
    """How big one request's messages and tool descriptions are.

    Args:
        body: A chat request.

    Returns:
        Characters of JSON: the messages, then the tools.
    """
    messages = json.dumps(body.get("messages") or [], ensure_ascii=False)
    tools = json.dumps(body.get("tools") or [], ensure_ascii=False)
    return len(messages), len(tools)


def outside_lines(probe: Probe) -> list[str]:
    """Say where a run tried to connect besides the stand-in model.

    Args:
        probe: The run.

    Returns:
        Lines: a warning first when the run did not complete (a count of 0
        then means nothing was observed, not that nothing was asked).
    """
    lines = []
    if probe.exit_code != 0 or not probe.requests:
        lines.append(f"  {INCOMPLETE}")
    hosts = sorted({line.split()[1] for line in probe.outside if " " in line})
    count = f"  outside connections asked for: {len(probe.outside)}"
    lines.append(count if probe.outside else f"{count} {PROXY_ONLY}")
    lines += [f"    {host} (refused by the catcher)" for host in hosts]
    return lines


def sandbox_line(probe: Probe) -> str:
    """Say how many direct connections the sandbox refused the run.

    Args:
        probe: The run.

    Returns:
        One line.
    """
    if probe.refused is None:
        return "  sandbox        log unreadable"
    return f"  sandbox        refused {probe.refused} direct connections (its own log)"


def request_lines(probe: Probe) -> list[str]:
    """Describe a run's requests: their number, tools, size and reply length.

    Args:
        probe: The run.

    Returns:
        Lines; a request offering other tools than the first is named.
    """
    if not probe.requests:
        return ["  no request reached the stand-in model"]
    first = probe.requests[0]
    messages, tools = size_of(first)
    names = tools_of(first)
    asked = sorted({str(r.get("max_tokens")) for r in probe.requests})
    lines = [
        f"  requests       {len(probe.requests)}",
        f"  first request  {messages + tools:,} characters: messages {messages:,}, "
        f"tool descriptions {tools:,}",
        f"  tools offered  {len(names)}: {', '.join(names)}",
        f"  max_tokens     {', '.join(asked)} asked",
    ]
    for number, body in enumerate(probe.requests, start=1):
        other = tools_of(body)
        if other != names:
            last = " (after the last reply)" if number == len(probe.requests) else ""
            lines.append(
                f"  request {number}      {sum(size_of(body)):,} characters, other "
                f"tools{last}: {', '.join(other)}"
            )
    return lines


def refusal_lines(probe: Probe) -> list[str]:
    """What happened to each shell command in a run, from Qwen Code's own log.

    Args:
        probe: The run.

    Returns:
        Lines: each command and its result, then the log's own account.
    """
    results: dict[str, str] = {}
    calls: dict[str, str] = {}
    for event in read_jsonl(probe.log):
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                calls[str(block.get("id"))] = str(
                    (block.get("input") or {}).get("command", "")
                )
            elif block.get("type") == "tool_result":
                text = str(block.get("content") or "")
                verdict = f"refused: {text}" if block.get("is_error") else "ran"
                results[str(block.get("tool_use_id"))] = verdict
    lines = [f"  {calls[i]:<22} {results.get(i, 'no result')}" for i in calls]
    lines.append(f"  its log        {read_qwen_log(probe.log)[0]}")
    lines.append(f"  answer.txt     {probe.answered!r}")
    return lines


def main() -> None:
    """Run the six probes and print what each measured."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    try:
        ensure(QWEN, install=False, records=read_record())
    except HarnessInstallError as error:
        raise SystemExit(f"{error} qwen-code") from error
    root = chapter_home() / PROBE_FOLDER
    stand_in, catcher = StandIn(), Catcher()
    with serving(stand_in) as model_url, serving(catcher) as proxy_url:
        print(
            f"qwen_probe.py · what Qwen Code {PINS[QWEN].version} does around the model"
        )
        print(
            f"  stand-in model  {model_url}/v1 on this machine: keeps every request, "
            f"answers `{LOOK}`, `{WRITE}`, then text"
        )
        print(
            f"  catcher         HTTPS_PROXY={proxy_url}: writes down each outside "
            "connection asked for and refuses it"
        )
        print("  every run       inside chapter 19's sandbox, a private HOME")
        runs = [
            Run("1-own-switches", "its own switches left on", switches=False),
            Run(
                "2-own-switches-no-catcher",
                "its own switches left on, no HTTPS_PROXY: only the sandbox",
                switches=False,
                catcher=False,
            ),
            Run("3-wire-switches", "with wire.py's switches"),
            Run("4-bare", "--bare (wire.py's setup)"),
            Run("5-not-bare", "without --bare", bare=False),
            Run("6-read-file-excluded", "read_file excluded too", extra="read_file"),
        ]
        probes = {}
        for run in runs:
            run_folders = fresh_folders(root, run.name)
            # wire.py's wiring for this run's folders, then the run's one change.
            base = wire(QWEN, run_folders, INSTRUCTION, model_url, PROBE_KEY)
            wiring = variant(
                base,
                proxy_url if run.catcher else None,
                switches=run.switches,
                bare=run.bare,
                also_excluded=run.extra,
            )
            try:
                probe = run_once(run.name, wiring, run_folders, stand_in, catcher)
            finally:
                shutil.rmtree(run_folders.state, ignore_errors=True)
            probes[run.name] = probe
            print("")
            print(f"{RULE} {run.name[0]}. {run.title}")
            print(f"  exit           {probe.exit_code}")
            for line in run.reader(probe):
                print(line)
            print(sandbox_line(probe))
    print("")
    print(f"logs      {tilde(str(root))}/<run>/<run>.jsonl (Qwen Code's stream-json)")


if __name__ == "__main__":
    main()
