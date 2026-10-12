# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print the untrained model's recorded runs in three harnesses, from their own logs.

The project ran LFM2.5-2.6B, as released, on one held-out question in all
three harnesses (greedy decoding, the 8-call budget) and kept what each
harness printed. Each harness keeps its own kind of log, and that log is
where a run's failure is read:

* **mini-swe-agent** prints every step as text (its trajectory file was not
  kept in this run, so this is its console output);
* **Claude Code** writes stream-json: one JSON object per event, with each
  tool call and the error it sent back;
* **Pi** writes JSON events (``--mode json``): each message, each tool run,
  each tool result.

This script reads all three formats and prints, call by call, what the
model asked for and what the harness answered, then the project's verdict.
The logs are in ``data/transcripts``, with personal paths replaced (``<project>``,
``<task folder>``, ``<tmp>``, ``user``). Pi's copy is cut: its one tool
result is the whole binary database, 36,864 characters, repeated in six
events; each copy keeps its first 400 characters and says how many were
cut. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "data" / "transcripts"
TASK_ID = "clinic-medium-001"
QUESTION = "Which specialty has the most doctors?"
EXPECTED = "General Practice"
RECORDED = "2026-10-06"  # the date in the logs' own timestamps
MINI_LOG = TRANSCRIPTS / f"{TASK_ID}.mini-swe-agent.txt"
CLAUDE_LOG = TRANSCRIPTS / f"{TASK_ID}.claude-code.jsonl"
PI_LOG = TRANSCRIPTS / f"{TASK_ID}.pi.jsonl"
PROJECT_DEMO = TRANSCRIPTS / f"{TASK_ID}.project-demo.txt"
HARNESSES = ("mini-swe-agent", "claude-code", "pi")
BUDGET_STOP_TEXT = "Step budget exhausted; stopping."  # the proxy's stop reply
FINISH_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
CUT_RE = re.compile(r"\n?\[course copy: ([\d,]+) of ([\d,]+) characters cut[^\]]*\]")
SEPARATOR = "─" * 80  # mini-swe-agent's console draws one between steps
STEP_RE = re.compile(r"^mini-swe-agent \(step (\d+), ")
BINARY_MARK = "\x00"  # text holding a NUL byte is a binary file read as text
LINES_SHOWN = 3
WIDTH = 96
LABEL_WIDTH = 8
INDENT = " " * (2 + 6 + 2 + LABEL_WIDTH)


@dataclass
class Call:
    """One model call as the harness's log tells it.

    Attributes:
        number: Its place in the run (1 = first call).
        tool: The tool the model asked for, or "" for a reply with none.
        request: The command or arguments, or the reply's text.
        result: What the harness sent back.
        error: True when the harness refused the call.
        note: Anything else worth a word (prompt size, a cut copy).
    """

    number: int
    tool: str
    request: str
    result: str = ""
    error: bool = False
    note: str = ""


@dataclass
class Transcript:
    """One harness's recorded run.

    Attributes:
        harness: Its name.
        version: Its version, from the log's header.
        log_kind: What kind of log it keeps.
        calls: The model's calls, in order.
        ending: How the run ended, in the log's own terms.
        project: The project's one-line verdict for the run.
    """

    harness: str
    version: str
    log_kind: str
    calls: list[Call] = field(default_factory=list)
    ending: str = ""
    project: str = ""


def header_version(text: str) -> str:
    """Read the version from a log's first line, ``[mhrl] harness=... version=...``.

    Args:
        text: The log.

    Returns:
        The version, such as ``2.1.290``.
    """
    match = re.search(r"version=(\S+)", text.splitlines()[0] if text else "")
    return match.group(1) if match else "unknown"


def unwrap(lines: list[str]) -> list[str]:
    """Join the lines the console wrapped at 80 columns.

    The console breaks a long line after a space, so a line that ends in a
    space and is followed by text continues on the next line.

    Args:
        lines: The lines as printed.

    Returns:
        The lines as written.
    """
    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith(" ") and line.strip():
            out[-1] += line
        else:
            out.append(line)
    return [line.rstrip() for line in out]


def between(lines: list[str], start: str, stop: str | None) -> list[str]:
    """Take the lines after a marker line, up to another marker or the end.

    Args:
        lines: The block's lines.
        start: The line that opens the part.
        stop: The line that closes it, or None for the end of the block.

    Returns:
        The lines in between, empty ones at either end dropped.
    """
    if start not in lines:
        return []
    rest = lines[lines.index(start) + 1 :]
    if stop is not None and stop in rest:
        rest = rest[: rest.index(stop)]
    while rest and not rest[0].strip():
        rest.pop(0)
    while rest and not rest[-1].strip():
        rest.pop()
    return rest


def read_mini(path: Path) -> Transcript:
    """Read mini-swe-agent's console output, step by step.

    Args:
        path: The log.

    Returns:
        The transcript.

    Raises:
        OSError: If the log cannot be read.
    """
    text = path.read_text(encoding="utf-8")
    run = Transcript("mini-swe-agent", header_version(text), "its console output")
    last = 0
    for block in text.split(SEPARATOR)[1:]:
        lines = unwrap(block.strip("\n").splitlines())
        step = STEP_RE.match(lines[0]) if lines else None
        if step:
            last = int(step.group(1))
            command = "\n".join(between(lines, "```", "```"))
            output = between(lines, "<output>", "<exception_info>")
            exception = between(lines, "<exception_info>", "Exit:")
            call = Call(last, "bash", command, "\n".join(output))
            if FINISH_MARK in command:
                # It ran: mini-swe-agent saw the marker it printed and stopped.
                # Its console then shows its placeholder for a step whose
                # output was never filed: "action was not executed".
                call.note = "the finishing command: it ran, and the run ended"
                shown = "\n".join(exception) or call.result
                call.result = f"{shown} (its placeholder: no output was filed)"
            run.calls.append(call)
            if "Exit:" in lines:
                run.ending = "Exit: (mini-swe-agent's own word for a finished run)"
        elif lines[:2] == ["User:", "Tool call error:"]:
            last += 1
            error = " ".join(between(lines, "<error>", "</error>"))
            run.calls.append(Call(last, "", "a reply mini-swe-agent could not use"))
            run.calls[-1].result, run.calls[-1].error = error, True
    return run


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines log, skipping its header lines.

    Args:
        path: The log.

    Returns:
        Its JSON objects, in order.

    Raises:
        OSError: If the log cannot be read.
        ValueError: If a line that opens like JSON is not JSON.
    """
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
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
        parts = [
            text_of(b.get("text", b.get("content", "")))
            for b in content
            if isinstance(b, dict)
        ]
        return "\n".join(part for part in parts if part)
    return ""


def tidy_error(text: str) -> str:
    """Strip Claude Code's ``<tool_use_error>`` wrapper.

    Args:
        text: A tool result.

    Returns:
        The message inside.
    """
    return re.sub(r"</?tool_use_error>", "", text).strip()


def read_claude(path: Path) -> Transcript:
    """Read Claude Code's stream-json log: each call, each result, the end.

    Args:
        path: The log.

    Returns:
        The transcript.
    """
    run = Transcript(
        "claude-code", header_version(path.read_text("utf-8")), "stream-json"
    )
    by_id: dict[str, Call] = {}
    for event in read_events(path):
        message = event.get("message") or {}
        if event.get("type") == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    request = json.dumps(block.get("input") or {})
                    command = (block.get("input") or {}).get("command")
                    call = Call(len(run.calls) + 1, block["name"], command or request)
                    run.calls.append(call)
                    by_id[block["id"]] = call
                elif block.get("type") == "text":
                    run.ending = stop_words(block.get("text", ""))
        elif event.get("type") == "user":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call = by_id.get(block.get("tool_use_id", ""))
                    if call is not None:
                        call.result = tidy_error(text_of(block.get("content")))
                        call.error = bool(block.get("is_error"))
        elif event.get("type") == "result":
            run.ending += (
                f" · result {event.get('subtype')}, {event.get('num_turns')} turns"
            )
    return run


def stop_words(text: str) -> str:
    """Say who wrote a run's last reply.

    Args:
        text: The last reply with no tool call.

    Returns:
        The reply, labelled as the proxy's when it is the budget stop.
    """
    if text.strip() == BUDGET_STOP_TEXT:
        return f"'{BUDGET_STOP_TEXT}' (the proxy's reply, not the model's)"
    return f"the model's reply with no tool call: {text.strip()!r}"


def read_pi(path: Path) -> Transcript:
    """Read Pi's JSON events: each tool call, its result, the end.

    Args:
        path: The log.

    Returns:
        The transcript.
    """
    run = Transcript("pi", header_version(path.read_text("utf-8")), "JSON events")
    by_id: dict[str, Call] = {}
    for event in read_events(path):
        kind = event.get("type")
        message = event.get("message") or {}
        if kind == "message_end" and message.get("role") == "assistant":
            used = (message.get("usage") or {}).get("input")
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    arguments = block.get("arguments") or {}
                    request = arguments.get("command") or json.dumps(arguments)
                    call = Call(len(run.calls) + 1, block["name"], request)
                    if used:
                        call.note = f"Pi's count: the call read {used:,} tokens"
                    run.calls.append(call)
                    by_id[block["id"]] = call
                elif block.get("type") == "text":
                    run.ending = stop_words(block.get("text", ""))
        elif kind == "tool_execution_end":
            call = by_id.get(event.get("toolCallId", ""))
            if call is not None:
                call.result = text_of((event.get("result") or {}).get("content"))
                call.error = bool(event.get("isError"))
        elif kind == "agent_end":
            run.ending += " · agent_end"
    return run


def read_project_lines(path: Path) -> dict[str, str]:
    """Read the project's one-line verdict for each harness's run.

    Args:
        path: The project's demo log, ``[harness] task ... · calls N · S s``.

    Returns:
        The verdict line, by harness.
    """
    verdicts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\[([\w-]+)\] task \S+ · (.*)$", line)
        if match:
            verdicts[match.group(1)] = match.group(2)
    return verdicts


def shorten(text: str, width: int = WIDTH) -> str:
    """Cut one line to the page's width.

    Args:
        text: The line.
        width: The most characters kept.

    Returns:
        The line, ending in ``…`` when cut.
    """
    return text if len(text) <= width else text[: width - 1] + "…"


def result_lines(call: Call) -> list[str]:
    """The first lines of a result, and a word about a cut copy.

    Args:
        call: The call.

    Returns:
        At most ``LINES_SHOWN`` lines, then one saying what is left out.
    """
    cut = CUT_RE.search(call.result)
    text = CUT_RE.sub("", call.result).rstrip("\n ")
    if cut or BINARY_MARK in text:  # a binary file: its bytes are not lines
        printable = "".join(ch if ch.isprintable() else "·" for ch in text[:60])
        size = cut.group(2) if cut else f"{len(text):,}"
        kept = f" (this copy keeps {len(text):,})" if cut else ""
        return [f"{printable}… a binary file, {size} characters{kept}"]
    lines = text.splitlines() or ["(no output)"]
    shown = [shorten(line) for line in lines[:LINES_SHOWN]]
    if len(lines) > LINES_SHOWN:
        shown.append(f"... {len(lines) - LINES_SHOWN} more lines")
    return shown


def print_transcript(run: Transcript) -> None:
    """Print one harness's run, call by call.

    Args:
        run: The transcript.
    """
    print(f"━━ {run.harness} {run.version} · its log: {run.log_kind}")
    for call in run.calls:
        tool = call.tool or "(none)"
        request = shorten(call.request.replace("\n", " "), WIDTH - LABEL_WIDTH)
        print(f"  call {call.number:<2}  {tool:<{LABEL_WIDTH}}{request}")
        if call.note:
            print(f"{INDENT}({call.note})")
        label = "error" if call.error else "out"
        for index, line in enumerate(result_lines(call)):
            print(f"{' ' * 12}{label if index == 0 else '':<{LABEL_WIDTH - 2}}{line}")
    print(f"  end       {run.ending.strip(' ·')}")
    print(f"  project   {run.project}")


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("harness", nargs="*", help=f"any of {', '.join(HARNESSES)}")
    args = parser.parse_args()
    unknown = sorted(set(args.harness) - set(HARNESSES))
    if unknown:
        parser.error(f"unknown harness {', '.join(unknown)}: choose {HARNESSES}")
    return args


def main() -> None:
    """Print the recorded runs."""
    args = parse_args()
    readers = {"mini-swe-agent": read_mini, "claude-code": read_claude, "pi": read_pi}
    logs = {"mini-swe-agent": MINI_LOG, "claude-code": CLAUDE_LOG, "pi": PI_LOG}
    verdicts = read_project_lines(PROJECT_DEMO)
    print(f"replay.py · LFM2.5-2.6B as released (no training) on {TASK_ID}: {QUESTION}")
    print(
        f"  recorded by the project on {RECORDED} · greedy decoding · 8-call budget "
        f"· expected {EXPECTED!r}"
    )
    for name in args.harness or HARNESSES:
        run = readers[name](logs[name])
        run.project = verdicts.get(name, "no verdict recorded")
        print_transcript(run)


if __name__ == "__main__":
    main()
