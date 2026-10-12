# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Claude Code through the capture proxy, its total-tokens note on and then off.

Claude Code 2.1.290 tells the model how much of its budget is left, in a
note written ``<total_tokens>N tokens left</total_tokens>``. The project
turns it off (``CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off``) because, it says,
the note splits every rollout into one row per call. This script runs the
same task three times through the proxy: with Claude Code's default (the
variable unset: the number counts down), with ``fixed`` (the number never
changes) and with ``off``, and prints for each:

* what Claude Code sent, call by call: a relay between Claude Code and the
  proxy records each request as it left Claude Code (which messages, and
  every note in them);
* what the engine received, call by call: the proxy's own record, after it
  rewrote the request into OpenAI's format (the notes again, and how each
  call's system prompt differs from the call before);
* what the proxy built from it: the graph, the rows, the findings.

With the scripted engine (``ENGINE_PORT=8119 ./start.sh scripted``) the
replies are four scripted ``Bash`` calls; with the model
(``ENGINE_PORT=8119 ./start.sh``) LFM2.5 answers. Each run's folder,
``COURSE_HOME/21-calls-to-rows/runs/reminder-<setting>``, keeps Claude
Code's own log (``stream.jsonl``), what it sent (``sent.jsonl``) and the
proxy's record (``rollout.json``), which ``crosscheck.py`` compares.
Claude Code is installed on first use (see ``claude_code.py``). Only Python's
standard library is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capture_client import PROXY_URL, PURPOSE_TRAIN, CaptureClient, ProxyError
from claude_code import (
    CLAUDE_VERSION,
    REMINDER_ENV,
    REMINDER_OFF,
    TOOLS,
    ClaudeNotInstalledError,
    Relay,
    RelayedRequest,
    find_claude,
    prepare_run,
    run_claude,
)
from grader import Grade, grade
from lfm_files import chapter_home, short_path
from rollout_view import (
    COLOR_CHOICES,
    INDENT,
    Painter,
    clip,
    print_findings,
    print_graph,
    print_rows,
    print_stats,
)
from scripted_replies import (
    MURCIA_COMMAND,
    TABLES_COMMAND,
    NotScriptedError,
    ScriptClient,
    ScriptError,
    answer_reply,
    call_reply,
)
from tasks import DATA_DIR, Task, find_task, task_instruction
from training_rows import count_calls, n_sampled

SCRIPT = "reminder.py"
DEFAULT_TASK = "clinic-easy-001"
RUNS_FOLDER = "runs"
RUN_PREFIX = "reminder-"
ROLLOUT_FILE = "rollout.json"  # the proxy's record, every row and message
SENT_FILE = "sent.jsonl"  # what Claude Code sent, request by request
# The note's settings, in the order they run (None: the variable unset).
# "fixed" always shows 5000000: does the number's change split the rollout?
SETTINGS = {"default": None, "fixed": "fixed", "off": REMINDER_OFF}
ALL = "all"
NOTE = re.compile(r"<total_tokens>(\d+) tokens left</total_tokens>")
MESSAGES_PATH = "/v1/messages"  # the model calls (not .../count_tokens)
MARK = "*"  # after a message that carries a note
DIFF_CHARS = 70
RULE = "━━"
# The scripted model's replies: Claude Code's tool is named Bash.
SCRIPTED = (
    call_reply(TABLES_COMMAND, tool=TOOLS),
    call_reply(MURCIA_COMMAND, tool=TOOLS, think="Count them."),
    call_reply("echo 30 > answer.txt", tool=TOOLS, think="Write it."),
    answer_reply("30 patients live in Murcia; answer.txt holds 30."),
)


@dataclass(frozen=True)
class Result:
    """One setting's run, for the closing table.

    Attributes:
        setting: ``default``, ``fixed`` or ``off``.
        grade: The verdict on answer.txt.
        trainer: The rollout document the trainer gets.
    """

    setting: str
    grade: Grade
    trainer: dict[str, Any]


def text_of(content: Any) -> str:
    """Flatten a message's content (text, or a list of blocks) into text.

    Args:
        content: A string, or Anthropic or OpenAI content blocks.

    Returns:
        The text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or block.get("content") or "")
            for block in content
            if isinstance(block, dict)
        )
    return ""


def notes(text: str) -> list[str]:
    """Find every total-tokens note in a text.

    Args:
        text: Any text.

    Returns:
        The numbers the notes show, in order.
    """
    return NOTE.findall(text)


def role_line(messages: list[dict[str, Any]]) -> str:
    """List a request's message roles, marking those that carry a note.

    Args:
        messages: The request's messages.

    Returns:
        Such as ``user system* assistant user system*``.
    """
    names = []
    for message in messages:
        marked = bool(notes(text_of(message.get("content"))))
        names.append(message.get("role", "?") + (MARK if marked else ""))
    return " ".join(names)


def all_notes(messages: list[dict[str, Any]], system: Any = None) -> list[str]:
    """Collect every note of a request, its system prompt first.

    Args:
        messages: The request's messages.
        system: Anthropic's separate system prompt, if any.

    Returns:
        The numbers, in order.
    """
    found = notes(text_of(system)) if system is not None else []
    for message in messages:
        found += notes(text_of(message.get("content")))
    return found


def model_calls(requests: list[RelayedRequest]) -> list[dict[str, Any]]:
    """Keep the requests that asked the model for a reply.

    Args:
        requests: Everything the relay kept.

    Returns:
        The bodies sent to ``/v1/messages`` with a list of messages.
    """
    return [
        request.body
        for request in requests
        if request.path.split("?", 1)[0] == MESSAGES_PATH
        and isinstance(request.body.get("messages"), list)
    ]


def print_sent(requests: list[RelayedRequest]) -> None:
    """Print what Claude Code sent, call by call (the relay's record).

    Args:
        requests: Everything the relay kept, in order.
    """
    calls = model_calls(requests)
    others = len(requests) - len(calls)
    extra = f"; {others} other requests" if others else ""
    print(f"what Claude Code sent ({len(calls)} model calls{extra}, before the proxy)")
    for number, body in enumerate(calls, start=1):
        system = body.get("system")
        blocks = len(system) if isinstance(system, list) else int(bool(system))
        print(
            f"{INDENT}call {number}  system prompt: {blocks} blocks, notes "
            f"{' '.join(notes(text_of(system))) or 'none'} · messages: "
            f"{role_line(body['messages'])}"
        )
        later = all_notes(body["messages"])
        print(f"{INDENT}        notes in the messages: {' '.join(later) or 'none'}")


def print_received(document: dict[str, Any]) -> None:
    """Print what the engine received, call by call (the proxy's record).

    Args:
        document: The rollout document, with the messages.
    """
    print("what the engine received (the proxy's record of each call)")
    previous = ""
    for turn in document["turns"]:
        messages = turn.get("request_messages") or []
        first = text_of(messages[0].get("content")) if messages else ""
        found = " ".join(all_notes(messages)) or "none"
        print(
            f"{INDENT}call {turn['index'] + 1}  messages: {role_line(messages)} "
            f"· notes {found}"
        )
        if previous:
            print(f"{INDENT}        {system_change(previous, first)}")
        previous = first


def system_change(before: str, after: str) -> str:
    """Say how one call's system prompt differs from the call before.

    Args:
        before: The earlier call's system message.
        after: This call's.

    Returns:
        One line: unchanged, grown at its end (and by what), or where it differs.
    """
    if before == after:
        return "system prompt: the same as the call before"
    if after.startswith(before):
        added = json.dumps(after[len(before) :])
        added = clip(added, DIFF_CHARS)
        return f"system prompt: the call before's, plus {added} at its end"
    same = 0
    while same < min(len(before), len(after)) and before[same] == after[same]:
        same += 1
    return f"system prompt: differs from the call before at character {same:,}"


def run_setting(
    setting: str,
    task: Task,
    client: CaptureClient,
    binary: Path,
    engine: ScriptClient | None,
    painter: Painter,
) -> Result:
    """Run Claude Code once with one setting of the note, and print what happened.

    Args:
        setting: ``default``, ``fixed`` or ``off``.
        task: The task.
        client: The proxy's session client.
        binary: The pinned ``claude``.
        engine: The scripted engine's remote control, or None for the model.
        painter: Colours, or none.

    Returns:
        The run's numbers.

    Raises:
        ClaudeNotInstalledError: If the sandbox cannot be applied.
    """
    value = SETTINGS[setting]
    shown = f"{REMINDER_ENV}={value}"
    if value is None:
        shown = f"{REMINDER_ENV} unset (the default)"
    print(painter.paint(f"\n{RULE} {shown}", "head"))
    folder = chapter_home() / RUNS_FOLDER / f"{RUN_PREFIX}{setting}"
    prepare_run(folder, DATA_DIR / task.db_file)
    session = client.open_session(PURPOSE_TRAIN, {"task_id": task.task_id})
    relay = Relay(client.proxy_url)
    unused = 0
    try:
        if engine is not None:
            engine.clear()
            engine.push(list(SCRIPTED))
        base_url = relay.start()
        run = run_claude(
            binary, folder, task_instruction(task), base_url, session.session_id, value
        )
        if engine is not None:
            unused = engine.state()["queued"]
            engine.clear()
        trainer = client.rollout(session.session_id)
        full = client.rollout(
            session.session_id, include_discarded=True, include_messages=True
        )
    finally:
        relay.stop()
        with contextlib.suppress(ProxyError):
            client.close(session.session_id)
    verdict = grade(task, run.task_folder)
    save(folder, full, relay.requests)
    ending = "stopped at the time limit" if run.exit_code is None else "exit"
    code = "" if run.exit_code is None else f" {run.exit_code}"
    print(
        f"claude    {ending}{code} after {run.seconds:.0f} s · "
        f"{len(full['turns'])} calls reached the proxy · grade "
        f"{'correct' if verdict.correct else 'wrong'} (answer.txt "
        f"{verdict.submitted!r})"
    )
    if unused:  # the conversation did not go as scripted
        print(f"{INDENT}note: {unused} scripted replies were never asked for")
    print_sent(relay.requests)
    print_received(full)
    print_graph(full, painter)
    print_rows(full, painter)
    print_findings(trainer, painter)
    print_stats(trainer)
    print(f"saved     {short_path(folder)}")
    return Result(setting, verdict, trainer)


def save(
    folder: Path, document: dict[str, Any], requests: list[RelayedRequest]
) -> None:
    """Keep the proxy's record and what Claude Code sent, for crosscheck.py.

    Args:
        folder: The run's folder.
        document: The rollout document, every row and message included.
        requests: What the relay kept: each request's path and body.
    """
    text = json.dumps(document, indent=1) + "\n"
    (folder / ROLLOUT_FILE).write_text(text, encoding="utf-8")
    lines = "".join(
        json.dumps({"path": request.path, "body": request.body}) + "\n"
        for request in requests
    )
    (folder / SENT_FILE).write_text(lines, encoding="utf-8")


def print_table(results: list[Result]) -> None:
    """Print the runs side by side.

    Args:
        results: One result per setting.
    """
    print(f"\n{RULE} the same task, the note's three settings")
    print(
        f"{INDENT}{'setting':<8} {'grade':<8} calls roots rows row-tokens mask-1 "
        "trainable n_calls n_sampled"
    )
    for result in results:
        doc = result.trainer
        stats = doc["stats"]
        row_tokens = sum(len(row["input_ids"]) for row in doc["sequences"])
        verdict = "correct" if result.grade.correct else "wrong"
        print(
            f"{INDENT}{result.setting:<8} {verdict:<8} {stats['n_turns']:>5} "
            f"{stats['n_roots']:>5} {stats['n_sequences']:>4} {row_tokens:>10,} "
            f"{stats['n_trainable_tokens']:>6,} {str(doc['trainable']).lower():>9} "
            f"{count_calls(doc):>7} {n_sampled(doc):>9,}"
        )


def scripted_engine(client: CaptureClient) -> ScriptClient | None:
    """Find out whether the proxy is in front of the scripted engine.

    Args:
        client: The proxy's session client.

    Returns:
        The engine's remote control, or None when it is a real engine.

    Raises:
        SystemExit: If the proxy is not running, or its engine does not answer.
    """
    try:
        upstream = client.health().get("upstream")
    except ProxyError as error:
        raise SystemExit(f"{error}: start the stack with ./start.sh") from error
    if not upstream:
        raise SystemExit(f"{client.proxy_url}/health names no engine: is it the proxy?")
    engine = ScriptClient(upstream)
    try:
        engine.state()
    except NotScriptedError:
        return None  # a real engine: the model answers
    except ScriptError as error:
        raise SystemExit(f"the proxy's engine does not answer: {error}") from error
    return engine


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("setting", nargs="?", default=ALL, choices=(ALL, *SETTINGS))
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--color", default="auto", choices=COLOR_CHOICES)
    return parser.parse_args()


def main() -> None:
    """Run Claude Code with each setting of the note, and print what the proxy built."""
    args = parse_args()
    painter = Painter(args.color)
    client = CaptureClient(args.proxy_url)
    engine = scripted_engine(client)
    try:
        binary = find_claude()
    except ClaudeNotInstalledError as error:
        raise SystemExit(str(error)) from error
    model = "the scripted engine" if engine else "the model"
    print(
        f"{SCRIPT} · Claude Code {CLAUDE_VERSION} through the capture proxy at "
        f"{client.proxy_url}, in front of {model}"
    )
    task = find_task(DEFAULT_TASK)
    print(f"{INDENT}task {task.task_id}: {task.question}")
    settings = list(SETTINGS) if args.setting == ALL else [args.setting]
    results = []
    for setting in settings:
        try:
            results.append(run_setting(setting, task, client, binary, engine, painter))
        except (ProxyError, ScriptError, ClaudeNotInstalledError) as error:
            raise SystemExit(f"{setting}: {error}") from error
    if len(results) > 1:
        print_table(results)


if __name__ == "__main__":
    main()
