# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run one real harness on one task, through the capture proxy, inside the sandbox.

This is what the project's ``run_rollout`` does for each attempt
(``mhrl/rollout.py``), one harness at a time, printed:

1. **version**: the harness says its version, inside the sandbox; any
   version but the pinned one is refused, so a score always names the
   harness that made it;
2. **session**: the proxy opens an evaluation session (temperature 0, an
   8-call budget, the project's settings); its id is the API key;
3. **wiring**: ``wire.py`` writes the harness's own settings, pointing it at
   the proxy and nowhere else;
4. **run**: the harness runs, unmodified, in chapter 19's sandbox, with a
   private ``HOME``; ``lsof`` watches every connection it opens;
5. **verdict**: the grader reads ``answer.txt``; the proxy says which API
   dialect the harness spoke, what each call read and what stopped the run;
   the harness's own log says the same from its side.

With the scripted engine (``ENGINE_PORT=8119 ./start.sh scripted``) the
replies come from a script, queued here before the run: ``--script solve``
answers the task in the harness's own tool names, ``--script mistakes``
makes the mistakes the untrained model made (a tool name in the wrong case,
a binary file read whole, replies with no usable call), so the harness's
reactions can be read without a model. With the model (``./start.sh``)
nothing is queued. Each run is kept in
``COURSE_HOME/24-measuring/runs/<harness>[-variant]``: the task
folder, the harness's settings and its own log, the proxy's record and
``run.json``. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from capture_client import PURPOSE_EVAL, CaptureClient, ProxyError
from grader import Grade, grade
from install import (
    CLAUDE,
    HARNESSES,
    MINI,
    PI,
    PINS,
    HarnessInstallError,
    check_version,
    probe_version,
    resolved_version,
)
from lfm_files import MAX_CALLS, chapter_home, short_path, tilde
from loopback import NetworkReport, NetworkWatch, describe
from sandbox import SandboxError, sandbox_command, sandbox_params
from scripted_replies import (
    TABLES_COMMAND,
    NotScriptedError,
    Reply,
    ScriptClient,
    ScriptError,
    answer_reply,
    call_reply,
    tool_reply,
)
from tasks import DATABASE_FILE, Task, find_task, prepare_workdir, task_instruction
from wire import (
    CLAUDE_SETUPS,
    MAX_PROMPT_TOKENS,
    PI_API,
    PI_APIS,
    PI_PROVIDER,
    ROLLOUT_TIMEOUT_S,
    Options,
    RunFolders,
    Wiring,
    wire,
)

PROXY_URL = "http://127.0.0.1:8200"
DEFAULT_TASK = "clinic-medium-001"  # held out; the project's recorded demo task
RUNS_FOLDER = "runs"
TASK_FOLDER, STATE_FOLDER = "task", "state"
RUN_FILE, ROLLOUT_FILE, STDERR_FILE = "run.json", "rollout.json", "stderr.txt"
STOP_WAIT_S = 10
SCRIPTS = ("auto", "solve", "mistakes", "none")
FINISH_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BUDGET_STOP_TEXT = "Step budget exhausted; stopping."  # the proxy's stop reply
CONTEXT_FINDING = "context_budget_exhausted"
REFUSED_TOKENS_RE = re.compile(r"prompt is ([\d,]+) tokens")
# A provider name becomes a JSON key, a command-line value and part of the
# run's folder name, so it is kept to plain characters.
PROVIDER_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
MINI_FORMAT_ERRORS = 3  # mini_swe_agent.yaml's max_consecutive_format_errors
# The tool names each harness offers the model (what --script solve calls).
BASH_TOOL = {MINI: "bash", CLAUDE: "Bash", PI: "bash"}
# mini-swe-agent's trajectory says how it ended, in its own words.
MINI_EXITS = {
    "Submitted": f"mini-swe-agent's finishing command: `{FINISH_COMMAND}`",
    "LimitsExceeded": f"mini-swe-agent's step_limit ({MAX_CALLS} calls)",
    "RepeatedFormatError": (
        f"{MINI_FORMAT_ERRORS} replies in a row without a usable bash call "
        "(mini-swe-agent's max_consecutive_format_errors)"
    ),
    "TimeExceeded": "mini-swe-agent's own wall-time limit",
}
LOG_NAMES = {
    MINI: "mini-swe-agent's trajectory",
    CLAUDE: "Claude Code's stream-json log",
    PI: "Pi's JSON events",
}
ERRORS_SHOWN = 3
# Each scripted reply takes this long, as the model's would: long enough for
# lsof, sampling every 0.1 s, to see the call's open connection.
SCRIPT_SECONDS = 0.5


@dataclass
class Facts:
    """What one run did, from the proxy's record and the harness's own log.

    Attributes:
        dialect: The API dialect the proxy detected.
        agent_calls: Calls that offered the model tools (the project's
            ``count_calls``).
        side_calls: Calls that offered none (titles, summaries).
        first_prompt: Prompt tokens of the first agent call, or None.
        budget_stops: Calls the proxy answered itself (budget or window).
        refused_tokens: Size of a prompt the engine refused, or None.
        stop: The rule that ended the run, in words.
        log_says: What the harness's own log says about its end.
        tool_errors: Error messages the harness gave the model.
        findings: The proxy's warnings, information notes left out.
    """

    dialect: str
    agent_calls: int
    side_calls: int
    first_prompt: int | None
    budget_stops: int
    refused_tokens: int | None
    stop: str = ""
    log_says: str = ""
    tool_errors: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Process:
    """How the harness's process ended.

    Attributes:
        exit_code: Its exit status, or None when the time limit killed it.
        seconds: How long it ran.
        network: What ``lsof`` saw.
    """

    exit_code: int | None
    seconds: float
    network: NetworkReport


def answer_text(task: Task) -> str:
    """Write the task's gold answer as an agent would write it to answer.txt.

    Args:
        task: The task.

    Returns:
        The answer on one line (a list's items joined by commas).
    """
    answer = task.answer
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    return str(answer)


def solve_script(harness: str, task: Task) -> list[Reply]:
    """Replies that solve the task in the harness's own tool names.

    The script reads the schema, runs the task's gold query, writes the
    answer, and then ends the way the harness expects an agent to end:
    mini-swe-agent with its finishing command, Claude Code and Pi with a
    reply that calls no tool.

    Args:
        harness: One of ``HARNESSES``.
        task: The task.

    Returns:
        The replies, in order.
    """
    tool = BASH_TOOL[harness]
    query = f"sqlite3 {DATABASE_FILE} {shlex.quote(task.gold_sql)}"
    write = f"echo {shlex.quote(answer_text(task))} > answer.txt"
    replies = [
        call_reply(TABLES_COMMAND, tool, "Which tables are there?"),
        call_reply(query, tool, "This query answers it."),
        call_reply(write, tool, "Write the answer."),
    ]
    if harness == MINI:
        replies.append(call_reply(FINISH_COMMAND, tool, "Finish."))
    else:
        replies.append(answer_reply(answer_text(task)))
    return replies


def mistakes_script(harness: str, task: Task) -> list[Reply]:
    """Replies that make the mistakes the untrained model made in each harness.

    * Claude Code: ``bash`` in lower case (its tool is ``Bash``), then a
      ``write`` tool it does not have, then a plain answer: no answer.txt.
    * Pi: its ``read`` tool on the binary ``data.db``; the next prompt holds
      the whole file and overflows the 16,384-token window.
    * mini-swe-agent: three replies in a row with no usable bash call.

    Args:
        harness: One of ``HARNESSES``.
        task: The task.

    Returns:
        The replies, in order.
    """
    if harness == CLAUDE:
        return [
            call_reply(TABLES_COMMAND, "bash", "List the tables."),
            call_reply(TABLES_COMMAND, "Bash", "List the tables."),
            tool_reply(
                "write",
                {"file": "answer.txt", "content": answer_text(task)},
                "Write the answer.",
            ),
            answer_reply(answer_text(task)),
        ]
    if harness == PI:
        return [tool_reply("read", {"path": DATABASE_FILE}, "Read the database.")]
    return [
        tool_reply("sqlite3", {"query": ".tables"}, "List the tables."),
        answer_reply("I will look at the tables."),
        answer_reply("The tables are there."),
    ]


def chosen_script(name: str, engine: str) -> str:
    """Resolve ``--script auto``: solve on the scripted engine, nothing on the model.

    Args:
        name: One of ``SCRIPTS``.
        engine: ``scripted`` or ``model``.

    Returns:
        ``solve``, ``mistakes`` or ``none``.

    Raises:
        SystemExit: If a script is asked of a real model.
    """
    if name == "auto":
        return "solve" if engine == "scripted" else "none"
    if name != "none" and engine != "scripted":
        raise SystemExit(f"--script {name} needs the scripted engine: the model writes")
    return name


def variant_name(harness: str, script: str, options: Options) -> str:
    """Name a run's folder after what it changed from the plain run.

    Args:
        harness: One of ``HARNESSES``.
        script: The script used.
        options: The run's options.

    Returns:
        Such as ``claude-code``, ``claude-code-bare`` or ``pi-mistakes``.
    """
    parts = [harness]
    if harness == CLAUDE and options.claude_setup != "project":
        parts.append(options.claude_setup)
    if harness == PI and options.pi_provider != PI_PROVIDER:
        parts.append(f"provider-{options.pi_provider}")
    if harness == PI and options.pi_api != PI_API:
        parts.append(f"api-{options.pi_api}")
    if script == "mistakes":
        parts.append("mistakes")
    return "-".join(parts)


def prepare_run(folder: Path, task: Task) -> RunFolders:
    """Make a fresh run folder: the task folder with data.db, an empty state.

    Args:
        folder: The run's folder (emptied first).
        task: The task.

    Returns:
        The run's folders.

    Raises:
        ValueError: If the folder is not inside the chapter's runs folder.
    """
    runs = chapter_home() / RUNS_FOLDER
    if not folder.resolve().is_relative_to(runs.resolve()):
        raise ValueError(f"{folder} is not inside {runs}")
    if folder.exists():
        shutil.rmtree(folder)  # an old answer.txt must never be graded again
    folders = RunFolders(task=folder / TASK_FOLDER, state=folder / STATE_FOLDER)
    prepare_workdir(task, folders.task)
    folders.make()
    return folders


def detect_engine(client: CaptureClient) -> tuple[str, str, str]:
    """Ask the proxy what it is in front of, and that engine what it is.

    Args:
        client: The proxy's client.

    Returns:
        The engine's URL, ``scripted`` or ``model``, and how it counts
        tokens (the scripted engine's tokenizer, or ``lfm2.5``).

    Raises:
        SystemExit: If the proxy cannot be reached.
    """
    try:
        health = client.health()
    except ProxyError as error:
        raise SystemExit(
            f"{error}\nstart the stack first: ENGINE_PORT=8119 ./start.sh scripted "
            "(no model) or ENGINE_PORT=8119 ./start.sh (the model)"
        ) from error
    engine_url = str(health.get("upstream", ""))
    try:
        state = ScriptClient(engine_url).state()
    except NotScriptedError:
        return engine_url, "model", "lfm2.5"
    except ScriptError as error:
        raise SystemExit(f"the proxy's engine does not answer: {error}") from error
    return engine_url, "scripted", str(state.get("tokenizer", "scripted"))


def kill_group(process: subprocess.Popen[bytes]) -> None:
    """Kill the run's whole process group: the harness and every command it started.

    Args:
        process: The harness, started in a session of its own.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def run_boxed(wiring: Wiring, folders: RunFolders, timeout_s: int) -> Process:
    """Run the harness once inside the sandbox, watching its connections.

    Args:
        wiring: Its command line, environment and settings.
        folders: The run's folders.
        timeout_s: The time limit; after it the whole process group is killed.

    Returns:
        How it ended.

    Raises:
        HarnessInstallError: If the sandbox cannot be applied (nothing runs).
    """
    try:
        params = sandbox_params(workdir=folders.task, state_dir=folders.state)
        argv = sandbox_command(wiring.argv, params=params)
    except SandboxError as error:
        raise HarnessInstallError(f"no sandbox, so no run: {error}") from error
    run_folder = folders.task.parent
    started = time.perf_counter()
    exit_code: int | None = None
    with (
        (run_folder / wiring.stdout_name).open("w", encoding="utf-8") as out,
        (run_folder / STDERR_FILE).open("w", encoding="utf-8") as err,
    ):
        # Its own session, so one signal reaches every process it starts.
        process = subprocess.Popen(
            argv,
            cwd=folders.task,
            env=wiring.env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        watch = NetworkWatch(process.pid)
        try:
            watch.start()
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            exit_code = None
        finally:
            # Kill first: nothing below may leave the harness running.
            kill_group(process)  # also any background command left running
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=STOP_WAIT_S)
            network = watch.stop()
    return Process(exit_code, time.perf_counter() - started, network)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines log, skipping lines that are not JSON objects.

    Args:
        path: The log.

    Returns:
        Its objects, in order ([] when the file is missing).
    """
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        with contextlib.suppress(ValueError):
            found = json.loads(line)
            if isinstance(found, dict):
                events.append(found)
    return events


def text_of(content: Any) -> str:
    """Flatten a message's content (a string or a list of blocks) to text.

    Args:
        content: The content.

    Returns:
        Its text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            text_of(block.get("text", block.get("content", "")))
            for block in content
            if isinstance(block, dict)
        )
    return ""


def read_claude_log(path: Path) -> tuple[str, list[str], str]:
    """Read Claude Code's stream-json log: its result, its tool errors, its last words.

    Args:
        path: ``stream.jsonl``.

    Returns:
        A one-line account of the result event, the tool errors it sent the
        model, and the last assistant text.
    """
    errors, last_text, says = [], "", "no result event in the log"
    for event in read_jsonl(path):
        message = event.get("message") or {}
        if event.get("type") == "assistant":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    last_text = block.get("text", "")
        elif event.get("type") == "user":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("is_error"):
                    errors.append(clean_error(text_of(block.get("content"))))
        elif event.get("type") == "result":
            says = (
                f"result {event.get('subtype')} · {event.get('num_turns')} turns · "
                f"stop_reason {event.get('stop_reason')}"
            )
    return says, errors, last_text


def read_pi_log(path: Path) -> tuple[str, list[str], str]:
    """Read Pi's JSON events: its tool calls, their errors, its last words.

    Args:
        path: ``events.jsonl``.

    Returns:
        A one-line account of the run, the tool errors, and the last
        assistant text.
    """
    errors, last_text, calls, ended = [], "", [], False
    for event in read_jsonl(path):
        kind = event.get("type")
        if kind == "tool_execution_start":
            calls.append(str(event.get("toolName")))
        elif kind == "tool_execution_end" and event.get("isError"):
            errors.append(
                clean_error(text_of((event.get("result") or {}).get("content")))
            )
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                last_text = text_of(
                    [
                        block
                        for block in message.get("content") or []
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                )
        elif kind == "agent_end":
            ended = True
    tools = ", ".join(calls) if calls else "none"
    says = f"agent_end {'reached' if ended else 'missing'} · tools run: {tools}"
    return says, errors, last_text


def read_mini_log(path: Path) -> tuple[str, list[str], str]:
    """Read mini-swe-agent's trajectory: how it says it ended.

    Args:
        path: ``trajectory.json``.

    Returns:
        Its exit status and call count, the format errors it sent the model,
        and the exit status alone.
    """
    if not path.is_file():
        return "no trajectory saved", [], ""
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return "trajectory is not JSON", [], ""
    info = found.get("info") or {}
    status = str(info.get("exit_status", ""))
    calls = (info.get("model_stats") or {}).get("api_calls")
    errors = [
        clean_error(str(message.get("content", "")))
        for message in found.get("messages") or []
        if (message.get("extra") or {}).get("interrupt_type") == "FormatError"
    ]
    return f"exit_status {status} · api_calls {calls}", errors, status


def clean_error(text: str) -> str:
    """Shorten an error message to its first meaningful line.

    Args:
        text: The message.

    Returns:
        One line, without the ``<tool_use_error>`` wrapper or mini-swe-agent's
        guidance after it.
    """
    text = text.replace("<tool_use_error>", "").replace("</tool_use_error>", "")
    for marker in ("<error>", "</error>"):
        text = text.replace(marker, "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    useful = [line for line in lines if line not in ("Tool call error:",)]
    return useful[0] if useful else ""


def dialect_of(document: dict[str, Any]) -> str:
    """Read the API dialect the proxy detected, as it recorded it.

    Args:
        document: The proxy's rollout document.

    Returns:
        Such as ``anthropic (the proxy read it from the request)``.
    """
    found = (document.get("metadata") or {}).get("api_type")
    if found:
        return f"{found} (the proxy read it from the request)"
    return "not recorded (the proxy records it with the first answered call)"


def read_facts(
    harness: str, document: dict[str, Any], log: Path, process: Process
) -> Facts:
    """Work out what the run did and what stopped it.

    Args:
        harness: One of ``HARNESSES``.
        document: The proxy's rollout document.
        log: The harness's own log of the run.
        process: How the process ended.

    Returns:
        The facts.
    """
    turns = [t for t in document.get("turns") or [] if not t.get("discarded")]
    with_tools = [t for t in turns if t.get("n_tools")]
    agent = with_tools or turns
    findings = [str(f) for f in document.get("validation") or []]
    refused = None
    for finding in findings:
        match = REFUSED_TOKENS_RE.search(finding)
        if CONTEXT_FINDING in finding and match:
            refused = int(match.group(1).replace(",", ""))
    facts = Facts(
        dialect=dialect_of(document),
        agent_calls=len(agent),
        side_calls=len(turns) - len(agent),
        first_prompt=int(agent[0].get("n_prompt") or 0) if agent else None,
        budget_stops=int(document.get("budget_stop_count") or 0),
        refused_tokens=refused,
        findings=[f for f in findings if not f.startswith("[INFO]")],
    )
    readers = {MINI: read_mini_log, CLAUDE: read_claude_log, PI: read_pi_log}
    says, errors, status = readers[harness](log)
    facts.log_says, facts.tool_errors = says, errors
    facts.stop = stop_rule(harness, facts, process, status)
    return facts


def stop_rule(harness: str, facts: Facts, process: Process, status: str) -> str:
    """Name the rule that ended the run, the strongest evidence first.

    Args:
        harness: One of ``HARNESSES``.
        facts: What the proxy recorded.
        process: How the process ended.
        status: mini-swe-agent's exit status, or the last assistant text.

    Returns:
        The rule, in words.
    """
    if process.exit_code is None:
        return "the time limit: the whole process group was killed"
    if facts.refused_tokens is not None:
        return (
            f"the context window: the engine refused a {facts.refused_tokens:,}-token "
            f"prompt (limit {MAX_PROMPT_TOKENS:,}), and the proxy answered "
            f"'{BUDGET_STOP_TEXT}', a reply with no tool call"
        )
    if harness == MINI:
        return MINI_EXITS.get(status, f"mini-swe-agent ended with {status!r}")
    if facts.budget_stops:
        return (
            f"the proxy's {MAX_CALLS}-call budget: it answered '{BUDGET_STOP_TEXT}', "
            "a reply with no tool call"
        )
    if process.exit_code != 0:
        return f"{harness} exited with status {process.exit_code}"
    return "a reply with no tool call: the harness's loop ends there"


def first_call(facts: Facts) -> str:
    """Say what the first agent call read, or why nothing was read.

    Args:
        facts: What the run did.

    Returns:
        One line.
    """
    if facts.first_prompt is not None:
        return (
            f"call 1 read {facts.first_prompt:,} prompt tokens "
            "(system prompt, tools, task)"
        )
    if facts.refused_tokens is not None:
        return (
            f"call 1 was refused: {facts.refused_tokens:,} prompt tokens, over the "
            f"{MAX_PROMPT_TOKENS:,}-token window"
        )
    return "no call reached the engine"


def proxy_address(url: str) -> str:
    """Write the proxy's address as lsof writes a remote end.

    Args:
        url: Such as ``http://127.0.0.1:8200``.

    Returns:
        ``127.0.0.1:8200``.
    """
    return url.split("://", 1)[-1].rstrip("/")


def own_log(wiring: Wiring, folders: RunFolders) -> Path:
    """Where the harness kept its own log of the run.

    Args:
        wiring: How it was wired.
        folders: The run's folders.

    Returns:
        mini-swe-agent's trajectory, or the file its standard output went to.
    """
    return wiring.own_log or folders.task.parent / wiring.stdout_name


def print_report(
    wiring: Wiring,
    facts: Facts,
    verdict: Grade,
    process: Process,
    proxy: str,
    log: Path,
) -> None:
    """Print what the run did, one fact per line.

    Args:
        wiring: How it was wired.
        facts: What it did.
        verdict: The grade.
        process: How its process ended.
        proxy: The proxy's URL.
        log: The harness's own log (the run's folder holds everything else).
    """
    for index, line in enumerate(wiring.settings):
        print(f"{'wired' if index == 0 else '':<10}{line}")
    print(f"dialect   {facts.dialect}")
    print(f"first     {first_call(facts)}")
    print(
        f"calls     {facts.agent_calls} agent calls · {facts.side_calls} side calls · "
        f"{facts.budget_stops} answered by the proxy itself"
    )
    print(f"stop      {facts.stop}")
    print(f"its log   {facts.log_says}")
    for error in facts.tool_errors[:ERRORS_SHOWN]:
        print(f"  told    {error}")
    if len(facts.tool_errors) > ERRORS_SHOWN:
        print(f"  told    ... and {len(facts.tool_errors) - ERRORS_SHOWN} more")
    submitted = "none" if verdict.submitted is None else repr(verdict.submitted)
    print(f"answer    answer.txt {submitted} · {verdict.reason}")
    exit_text = "killed" if process.exit_code is None else str(process.exit_code)
    print(f"exit      {exit_text} after {process.seconds:.1f} s")
    print(f"network   {describe(process.network, proxy_address(proxy))}")
    print(f"log       {short_path(log)} ({LOG_NAMES[wiring.harness]})")


def provider_name(value: str) -> str:
    """Check a provider name for Pi's models.json.

    Args:
        value: The name typed.

    Returns:
        The name.

    Raises:
        argparse.ArgumentTypeError: If it is not a plain name.
    """
    if not PROVIDER_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r}: a provider name is a letter, then letters, digits, - or _"
        )
    return value


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("harness", choices=HARNESSES)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--script", choices=SCRIPTS, default="auto")
    parser.add_argument("--claude-setup", choices=CLAUDE_SETUPS, default="project")
    parser.add_argument("--pi-provider", type=provider_name, default=PI_PROVIDER)
    parser.add_argument("--pi-api", choices=PI_APIS, default=PI_API)
    parser.add_argument("--timeout", type=int, default=ROLLOUT_TIMEOUT_S)
    return parser.parse_args()


def checked_version(
    harness: str, folders: RunFolders, instruction: str, options: Options
) -> str:
    """Ask the harness its version, inside the sandbox, and refuse any other.

    Args:
        harness: One of ``HARNESSES``.
        folders: The run's folders.
        instruction: The task.
        options: The run's options.

    Returns:
        The version, which is the pinned one.

    Raises:
        SystemExit: If the harness is missing or is another version.
    """
    try:
        if not resolved_version(harness):
            raise HarnessInstallError(
                f"{harness} is not installed: run `uv run install.py`"
            )
        # The run's own environment, before any session: no key yet.
        probe = wire(harness, folders, instruction, PROXY_URL, "none-yet", options)
        reported = probe_version(harness, folders.task, folders.state, probe.env)
        check_version(harness, reported)
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    pinned = PINS[harness].version
    print(f"version   {reported}, asked inside the sandbox · pinned {pinned} · ok")
    return reported


def queue_script(script: str, harness: str, task: Task, engine_url: str) -> None:
    """Queue the scripted replies for this run, after emptying the queue.

    Args:
        script: ``solve``, ``mistakes`` or ``none``.
        harness: One of ``HARNESSES``.
        task: The task.
        engine_url: The scripted engine's root.
    """
    if script == "none":
        return
    writer = solve_script if script == "solve" else mistakes_script
    replies = [replace(r, seconds=SCRIPT_SECONDS) for r in writer(harness, task)]
    client = ScriptClient(engine_url)
    client.clear()
    client.push(replies)


def save_run(
    folders: RunFolders, document: dict[str, Any], record: dict[str, Any]
) -> None:
    """Keep the proxy's record and the run's summary beside the run.

    Args:
        folders: The run's folders.
        document: The proxy's rollout document.
        record: The run's summary.
    """
    run_folder = folders.task.parent
    (run_folder / ROLLOUT_FILE).write_text(json.dumps(document, indent=1), "utf-8")
    (run_folder / RUN_FILE).write_text(tilde(json.dumps(record, indent=2)), "utf-8")


def outcome(verdict: Grade, process: Process, unused: int) -> dict[str, Any]:
    """The run's results, as run.json keeps them.

    Args:
        verdict: The grade.
        process: How the harness's process ended.
        unused: Scripted replies never asked for.

    Returns:
        The grade, the exit, the time, the network report, the unused replies.
    """
    return {
        "grade": asdict(verdict),
        "exit_code": process.exit_code,
        "seconds": round(process.seconds, 1),
        "network": process.network.as_dict(),
        "unused_replies": unused,
    }


def main() -> None:
    """Run one harness once and print what happened.

    Raises:
        SystemExit: If the harness is missing, another version, or the box
            cannot be applied (nothing runs then); status 1 when a connection
            left this machine.
    """
    args = parse_args()
    harness, task = args.harness, find_task(args.task)
    client = CaptureClient(args.proxy_url)
    engine_url, engine, tokens = detect_engine(client)
    script = chosen_script(args.script, engine)
    options = Options(args.claude_setup, args.pi_provider, args.pi_api, args.timeout)
    run_folder = chapter_home() / RUNS_FOLDER / variant_name(harness, script, options)
    print(f"run_one.py · {harness} through the capture proxy at {args.proxy_url}")
    print(f"  task      {task.task_id} (held out): {task.question}")
    print(f"  engine    {engine} at {engine_url} ({tokens} tokens) · script: {script}")
    folders = prepare_run(run_folder, task)
    instruction = task_instruction(task)
    version = checked_version(harness, folders, instruction, options)
    labels = {"task_id": task.task_id, "harness": harness}
    session = client.open_session(PURPOSE_EVAL, labels, MAX_CALLS)
    print(
        f"session   {session.session_id} · {session.rollout_type} · temperature 0 · "
        f"budget {session.max_model_calls} calls"
    )
    try:
        queue_script(script, harness, task, engine_url)
        wiring = wire(
            harness, folders, instruction, args.proxy_url, session.session_id, options
        )
        wiring.write()
        process = run_boxed(wiring, folders, args.timeout)
        verdict = grade(task, folders.task)
        document = client.rollout(session.session_id)
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    finally:
        client.close(session.session_id)  # on every way out
    left = ScriptClient(engine_url).clear()["queued"] if script != "none" else 0
    record = {
        "harness": harness,
        "version": version,
        "task": task.task_id,
        "engine": engine,
        "script": script,
        "options": asdict(options),
        **outcome(verdict, process, left),
        "own_log": LOG_NAMES[harness],
        "facts": "not read yet",  # replaced below; kept if reading fails
    }
    save_run(folders, document, record)  # before anything else can fail
    log = own_log(wiring, folders)
    facts = read_facts(harness, document, log, process)
    record["facts"] = asdict(facts)
    save_run(folders, document, record)
    print_report(wiring, facts, verdict, process, args.proxy_url, log)
    if left:
        print(f"script    {left} queued replies were never asked for")
    print(f"kept      {short_path(run_folder)}")
    if not process.network.loopback_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
