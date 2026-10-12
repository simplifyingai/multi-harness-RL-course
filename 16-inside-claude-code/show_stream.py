# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print, one line per event, what Claude Code printed during a run.

``claude -p --output-format stream-json --verbose`` prints one JSON object
per line: an ``init`` event (its tools, model and settings), then each
message as it happens (the model's replies as ``assistant``, the tool
outputs Claude Code sends back as ``user``), then a ``result``.
``run_claude.sh`` calls this after every run; by hand:
``uv run show_stream.py <run name or folder>``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from records import (
    CALLS_FILE,
    KIND_SIDE,
    STDERR_FILE,
    STREAM_FILE,
    TOOL_RESULT,
    TOOL_USE,
    RunNotFoundError,
    block_text,
    blocks,
    clip,
    model_calls,
    read_jsonl,
    run_folder,
)

SHOWN_CHARS = 90  # longer texts are cut, with their full length given
KIND_WIDTH = 10  # the column the event kinds are printed in


def one_line(text: str) -> str:
    """Show a text on one line, cut to ``SHOWN_CHARS`` with its full length.

    Args:
        text: Any text.

    Returns:
        The shortened text.
    """
    return clip(text, SHOWN_CHARS, with_length=True)


def describe(event: dict[str, Any]) -> list[str]:
    """Describe one stream-json event.

    Args:
        event: The decoded event.

    Returns:
        One line per thing the event carries.
    """
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init":
        tools = ", ".join(event.get("tools", []))
        return [
            f"init       tools [{tools}] · model {event.get('model')} · "
            f"{event.get('permissionMode')} · key from {event.get('apiKeySource')} · "
            f"version {event.get('claude_code_version')}"
        ]
    if kind in ("assistant", "user"):
        lines = []
        for block in blocks(event["message"]["content"]):
            label = f"{kind:<{KIND_WIDTH}}"
            if block["type"] == TOOL_USE:
                command = block["input"].get("command", block_text(block))
                lines.append(f"{label} tool_use {block['name']}: {one_line(command)}")
            elif block["type"] == TOOL_RESULT:
                error = " (error)" if block.get("is_error") else ""
                text = one_line(block_text(block))
                lines.append(f"{label} tool_result{error}: {text}")
            else:
                lines.append(f"{label} text: {one_line(block_text(block))}")
        return lines
    if kind == "result":
        text = one_line(str(event.get("result", "")))
        return [
            f"result     {event.get('subtype')} · {event.get('num_turns')} turns · "
            f"stop {event.get('stop_reason')} · {text}"
        ]
    return [str(kind)]


def main() -> None:
    """Print a run's stream-json events, its error stream and its request count.

    Raises:
        SystemExit: With a plain message when the run does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a run name (e.g. solve) or a run folder")
    args = parser.parse_args()
    try:
        folder = run_folder(args.run)
    except RunNotFoundError as error:
        sys.exit(str(error))

    events = read_jsonl(folder / STREAM_FILE)
    print(f"stream    {len(events)} events printed by Claude Code ({STREAM_FILE})")
    for event in events:
        for line in describe(event):
            print(f"  {line}")
    stderr_path = folder / STDERR_FILE
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
    for line in stderr.splitlines():
        print(f"stderr    {one_line(line)}")
    calls = model_calls(read_jsonl(folder / CALLS_FILE))
    sides = sum(1 for call in calls if call["reply"]["kind"] == KIND_SIDE)
    print(f"requests  {len(calls)} model calls reached the stand-in", end="")
    print(f" ({sides} side calls)")


if __name__ == "__main__":
    main()
