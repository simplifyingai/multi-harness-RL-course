# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay three scripted mishaps, and show what Claude Code did with each.

Each mishap is a stand-in script (``SCRIPTS`` in stand_in.py) run through
``run_claude.sh``; then the run's request log says what happened:

1. ``not-streamed``: the provider ignores ``"stream": true`` and answers with
   one JSON body. Does Claude Code still get the reply?
2. ``long-output``: one command prints a whole joined table. How much of it
   reaches the model's next call? Run twice: with Claude Code's default
   limit, and with ours (``bashOutputMaxChars`` 4,000).
3. ``lowercase-bash``: the model calls ``bash``; the only tool is ``Bash``.

    uv run mishaps.py                     all three
    uv run mishaps.py --only long-output  one of them
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from records import (
    CALLS_FILE,
    STREAM_FILE,
    TOOL_USE,
    RunNotFoundError,
    block_text,
    clip,
    first_loop_call,
    model_calls,
    read_jsonl,
    short,
    tool_results,
)
from tasks import runs_root

HERE = Path(__file__).resolve().parent
RUN_SCRIPT = HERE / "run_claude.sh"
RUN_OUTPUT = "run.txt"
NAME_PREFIX = "mishap-"
DEFAULT_SUFFIX = "-default"
OUR_BASH_CAP = 4000  # run_claude.sh's default, passed explicitly here
# The lines of run_claude.sh's output this report repeats.
REPORTED_PREFIXES = ("claude    exit", "verdict")
# Only one shape of command is ever re-run here, `sqlite3 data.db "<SQL>"`,
# and it is run as a list of arguments, never through a shell.
QUERY_RE = re.compile(r'sqlite3 data\.db "([^"]+)"')
SQLITE = "sqlite3"
DATABASE = "data.db"
SAVED_RE = re.compile(r"Full output saved to: (\S+)")
PREVIEW_RE = re.compile(
    r"(Preview \(first [^)]*\):\n)(.*)(\n\.\.\.\n</persisted-output>)", re.S
)
SHOWN_CHARS = 100
NEWLINE = "\n"


@dataclass(frozen=True)
class Exchange:
    """One tool call the stand-in made, and what came back in a later call.

    Attributes:
        call: The number of the call whose reply made the tool call.
        tool: The tool's name, as the reply wrote it.
        command: The command it was given.
        result: The tool_result text the harness sent back ("" if none).
        is_error: Whether the harness marked that result as an error.
        carried_by: The number of the call that carried the result (0 if none).
    """

    call: int
    tool: str
    command: str
    result: str
    is_error: bool
    carried_by: int


@dataclass(frozen=True)
class Run:
    """One run_claude.sh run made by this script.

    Attributes:
        folder: The run folder.
        calls: Its model calls, from calls.jsonl.
        reported: The lines of run_claude.sh's output worth repeating.
    """

    folder: Path
    calls: list[dict[str, Any]]
    reported: list[str]


def find_result(
    calls: list[dict[str, Any]], start: int, tool_use_id: str
) -> tuple[dict[str, Any] | None, int]:
    """Find the tool_result answering one tool_use, in the calls after it.

    Args:
        calls: A run's model calls, in order.
        start: How many calls to skip (the tool_use's own call and those before).
        tool_use_id: The id of the tool_use.

    Returns:
        The tool_result block and the number of the call carrying it, or
        (None, 0) when no later call carries one.
    """
    for number, call in enumerate(calls[start:], start=start + 1):
        for result in tool_results(call["body"]):
            if result.get("tool_use_id") == tool_use_id:
                return result, number
    return None, 0


def exchanges(calls: list[dict[str, Any]]) -> list[Exchange]:
    """Pair each scripted tool call with the result the harness sent back.

    Args:
        calls: A run's model calls, in order.

    Returns:
        One exchange per tool_use the stand-in sent.
    """
    found = []
    for number, call in enumerate(calls, start=1):
        for block in call["reply"].get("message", {}).get("content", []):
            if block["type"] != TOOL_USE:
                continue
            result, carried_by = find_result(calls, number, block["id"])
            found.append(
                Exchange(
                    call=number,
                    tool=block["name"],
                    command=block["input"].get("command", ""),
                    result=block_text(result) if result else "",
                    is_error=bool(result and result.get("is_error")),
                    carried_by=carried_by,
                )
            )
    return found


def resent_without_stream(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Tell whether a request repeats the previous one with streaming off.

    Args:
        before: A logged call.
        after: The call after it.

    Returns:
        True when ``before`` asked for a stream, ``after`` did not, and the
        two bodies are otherwise identical.
    """
    first, second = before["body"], after["body"]
    if not first.get("stream") or second.get("stream"):
        return False
    return first | {"stream": None} == second | {"stream": None}


def outcome(folder: Path) -> str:
    """Say how Claude Code's run ended, from what it printed.

    Args:
        folder: The run folder.

    Returns:
        How many replies it printed, and its result event's subtype and text.
    """
    events = read_jsonl(folder / STREAM_FILE)
    replies = sum(1 for event in events if event.get("type") == "assistant")
    results = [event for event in events if event.get("type") == "result"]
    if not results:
        return f"printed {replies} replies and no result"
    last = results[-1]
    text = last.get("result")
    return f"printed {replies} replies; result {last.get('subtype')}: {text!r}"


def run_claude(script: str, name: str, *extra: str) -> Run:
    """Make one scripted run with run_claude.sh, keeping its full output.

    Args:
        script: The stand-in script.
        name: The run's folder name.
        *extra: More options for run_claude.sh.

    Returns:
        The run.

    Raises:
        OSError: If run_claude.sh cannot be started or its output not saved.
    """
    options = ["--script", script, "--name", name, *extra]
    print(f"  ran      ./run_claude.sh {' '.join(options)}")
    finished = subprocess.run(
        ["bash", str(RUN_SCRIPT), *options], capture_output=True, text=True, check=False
    )
    output = finished.stdout + finished.stderr
    folder = runs_root() / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RUN_OUTPUT).write_text(output, encoding="utf-8")
    print(f"           exit {finished.returncode}; its output is in ", end="")
    print(short(str(folder / RUN_OUTPUT)))
    lines = output.splitlines()
    reported = [line for line in lines if line.startswith(REPORTED_PREFIXES)]
    return Run(folder, model_calls(read_jsonl(folder / CALLS_FILE)), reported)


def report_end(run: Run) -> None:
    """Print how a run ended: what Claude Code printed, its exit code, the grade.

    Args:
        run: The run.
    """
    print(f"  Claude Code {outcome(run.folder)}")
    for line in run.reported:
        print(f"  {line}")


def give_up(run: Run, what: str) -> NoReturn:
    """Stop with a plain message pointing at the run's saved output.

    Args:
        run: The run that did not produce what was needed.
        what: What was missing.

    Raises:
        SystemExit: Always.
    """
    sys.exit(f"{what}; see {short(str(run.folder / RUN_OUTPUT))}")


def not_streamed() -> None:
    """Mishap 1: the provider answers a stream request with one JSON body."""
    print('mishap 1 · not-streamed: the provider ignores "stream": true')
    run = run_claude("not-streamed", f"{NAME_PREFIX}not-streamed")
    previous: dict[str, Any] | None = None
    resent = 0
    for number, call in enumerate(run.calls, start=1):
        asked = "a stream" if call["body"].get("stream") else "no stream"
        got = "events" if call["reply"].get("streamed") else "one JSON body"
        again = ""
        if previous and resent_without_stream(previous, call):
            again = " · the same request again, with stream false"
            resent += 1
        print(f"  call {number}   asked for {asked:<9} · got {got}{again}")
        previous = call
    print(f"  {len(run.calls)} requests; {resent} of them repeat the request ", end="")
    print("before them, with streaming off")
    report_end(run)


def show_default(run: Run, dump: Exchange, printed: str) -> None:
    """Report the long output under Claude Code's default limit.

    Args:
        run: The run without our cap.
        dump: The exchange of the long command in that run.
        printed: What the command prints when we run it ourselves.
    """
    print(f"  command  {clip(dump.command, SHOWN_CHARS)}")
    print(f"  run by us in the task folder, it prints {len(printed):,} ", end="")
    print(f"characters in {printed.count(NEWLINE):,} lines")
    print(f"  Claude Code's default: call {dump.carried_by} carries ", end="")
    print(f"{len(dump.result):,} characters of it")
    report_end(run)


def show_capped(run: Run, same: Exchange) -> None:
    """Report the long output under our cap: a note, a preview, a saved file.

    Args:
        run: The run with our cap.
        same: The exchange of the same command in that run.
    """
    print(f"  bashOutputMaxChars {OUR_BASH_CAP} (ours): ", end="")
    print(f"call {same.carried_by} carries {len(same.result):,} characters:")
    note = PREVIEW_RE.sub(r"\1<the preview>\3", same.result)
    for line in note.splitlines():
        print(f"    | {short(line)}")
    preview = PREVIEW_RE.search(same.result)
    if preview:
        shown = preview.group(2)
        lines = shown.count(NEWLINE) + 1
        print(f"  the preview: the first {len(shown):,} characters ({lines} lines)")
    saved = SAVED_RE.search(same.result)
    if saved and Path(saved.group(1)).is_file():
        size = len(Path(saved.group(1)).read_text(encoding="utf-8"))
        print(f"  the saved file: {size:,} characters, the whole output")
    report_end(run)


def long_output() -> None:
    """Mishap 2: a command prints far more than the harness passes on."""
    print("mishap 2 · long-output: one command prints a whole joined table")
    name = f"{NAME_PREFIX}long-output"
    default = run_claude("long-output", name + DEFAULT_SUFFIX, "--bash-max", "default")
    capped = run_claude("long-output", name, "--bash-max", str(OUR_BASH_CAP))
    found = exchanges(default.calls)
    if not found:
        give_up(default, "the run made no tool calls")
    dump = max(found, key=lambda exchange: len(exchange.result))
    query = QUERY_RE.fullmatch(dump.command)
    if not query:
        give_up(default, f"not re-running {dump.command!r}")
    task = default.folder / "task"
    if not (task / DATABASE).is_file():
        give_up(default, f"no {DATABASE} in the run's task folder")
    printed = subprocess.run(
        [SQLITE, DATABASE, query.group(1)],
        cwd=task,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    show_default(default, dump, printed)
    same = [e for e in exchanges(capped.calls) if e.command == dump.command]
    if not same:
        give_up(capped, "the capped run never ran the long command")
    show_capped(capped, same[0])


def lowercase_bash() -> None:
    """Mishap 3: the model calls a tool the harness does not have."""
    print("mishap 3 · lowercase-bash: the model calls 'bash'; the tool is 'Bash'")
    run = run_claude("lowercase-bash", f"{NAME_PREFIX}lowercase-bash")
    try:
        tools = first_loop_call(run.calls)["body"].get("tools") or []
    except RunNotFoundError:
        give_up(run, "no model call offered tools")
    print(f"  tools offered: {[tool['name'] for tool in tools]}")
    found = exchanges(run.calls)
    for exchange in found:
        status = "error" if exchange.is_error else "ok"
        print(f"  call {exchange.call} reply   {exchange.tool}: ", end="")
        print(clip(exchange.command, SHOWN_CHARS))
        print(f"  call {exchange.carried_by} carries tool_result ({status}): ", end="")
        print(clip(exchange.result, SHOWN_CHARS))
    errors = sum(1 for exchange in found if exchange.is_error)
    print(f"  {len(run.calls)} calls; {errors} tool call came back as an error")
    report_end(run)


MISHAPS: dict[str, Callable[[], None]] = {
    "not-streamed": not_streamed,
    "long-output": long_output,
    "lowercase-bash": lowercase_bash,
}


def main() -> None:
    """Run the chosen mishaps, one Claude Code run at a time."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(MISHAPS))
    args = parser.parse_args()
    for name, mishap in MISHAPS.items():
        if args.only in (None, name):
            mishap()
            print()


if __name__ == "__main__":
    main()
