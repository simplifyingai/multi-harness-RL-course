"""Read what one Claude Code run left behind in its run folder.

``run_claude.sh`` gives every run a folder, ``runs/<name>/``, holding:

* ``calls.jsonl``: every request Claude Code sent, written by the stand-in
  (one JSON object per line: path, headers, body, and the reply it got);
* ``stream.jsonl``: what Claude Code printed (``--output-format stream-json``);
* ``stderr.txt``: what it printed on its error stream.

These helpers and names are shared by the stand-in and the three readers
(show_stream.py, inspect_calls.py and mishaps.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tasks import runs_root

CALLS_FILE = "calls.jsonl"
STREAM_FILE = "stream.jsonl"
STDERR_FILE = "stderr.txt"
MESSAGES_PATH = "/v1/messages"
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
# What the stand-in did with a request, as written in the log.
KIND_LOOP = "loop"
KIND_SIDE = "side"
KIND_COUNT = "count_tokens"
KIND_UNKNOWN = "unknown route"
# The kinds of content block a message can hold.
TEXT = "text"
TOOL_USE = "tool_use"
TOOL_RESULT = "tool_result"
ASSISTANT = "assistant"
HOME = str(Path.home())
# Claude Code names a project's folder after its path, "/" and "." becoming "-".
ENCODED_HOME = HOME.replace("/", "-").replace(".", "-")
NEWLINE_SHOWN = "⏎"


class RunNotFoundError(Exception):
    """Raised when a run folder, its request log or its model calls are missing."""


def run_folder(name: str) -> Path:
    """Find a run's folder by name (under ``runs/``), or take a path as it is.

    Args:
        name: A run name (``solve``) or a path to a run folder.

    Returns:
        The folder.

    Raises:
        RunNotFoundError: If it holds no ``calls.jsonl``.
    """
    folder = runs_root() / name
    if not (folder / CALLS_FILE).is_file() and Path(name).is_dir():
        folder = Path(name)
    if not (folder / CALLS_FILE).is_file():
        raise RunNotFoundError(
            f"no run called {name!r} (looked for {short(str(folder / CALLS_FILE))}); "
            "make one with ./run_claude.sh"
        )
    return folder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a file holding one JSON object per line.

    Args:
        path: The file.

    Returns:
        The objects; an empty list when the file does not exist.

    Raises:
        OSError: If the file exists but cannot be read.
        ValueError: If a line is not JSON (for example a run killed while it
            was writing its last line).
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def short(text: str) -> str:
    """Write your home folder as ``~``, so printed paths are not personal.

    Args:
        text: Any text.

    Returns:
        The text with the home folder replaced, also inside the folder names
        Claude Code makes from a path (``-Users-you--cache-...``).
    """
    return text.replace(HOME, "~").replace(ENCODED_HOME, "~")


def clip(text: str, width: int, with_length: bool = False) -> str:
    """Show a text on one line, cut to a width.

    Args:
        text: Any text.
        width: The most characters shown.
        with_length: Add the full length when the text is cut.

    Returns:
        The text with new lines shown as ``⏎`` and the home folder as ``~``.
    """
    flat = short(text).replace("\n", NEWLINE_SHOWN)
    if len(flat) <= width:
        return flat
    length = f" ({len(text):,} characters)" if with_length else ""
    return f"{flat[:width]}…{length}"


def model_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the requests that asked for a reply (``POST /v1/messages``).

    Args:
        records: Every logged request.

    Returns:
        The model calls, in order.
    """
    return [r for r in records if r["path"] == MESSAGES_PATH]


def first_loop_call(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the first call of the agent's loop (the first that offers tools).

    A harness may open with a side call (one that offers no tools), so the
    first logged call is not always the loop's.

    Args:
        calls: A run's model calls.

    Returns:
        The call.

    Raises:
        RunNotFoundError: If no call offers tools.
    """
    for call in calls:
        if call["body"].get("tools"):
            return call
    raise RunNotFoundError("the run has no model call that offers tools")


def blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a message's content as a list of blocks.

    The API accepts a plain string as shorthand for one text block.

    Args:
        content: A message's ``content``.

    Returns:
        The blocks.
    """
    if isinstance(content, str):
        return [{"type": TEXT, "text": content}]
    return content


def block_text(block: dict[str, Any]) -> str:
    """Return the text a content block carries.

    Args:
        block: A text, tool_use or tool_result block.

    Returns:
        The text; for a tool_use, its input as JSON; for a tool_result, the
        tool's output as the model will read it.
    """
    kind = block.get("type")
    if kind == TOOL_USE:
        return json.dumps(block.get("input", {}), ensure_ascii=False)
    if kind == TOOL_RESULT:
        return "".join(block_text(inner) for inner in blocks(block.get("content", "")))
    return str(block.get("text", ""))


def message_label(message: dict[str, Any]) -> str:
    """Name a message by its role and, for tool traffic, its kind.

    Args:
        message: One entry of a request's ``messages``.

    Returns:
        ``user``, ``system``, ``assistant tool_use``, ``user tool_result``...
    """
    kinds = [block.get("type") for block in blocks(message["content"])]
    for special in (TOOL_USE, TOOL_RESULT):
        if special in kinds:
            return f"{message['role']} {special}"
    return message["role"]


def message_shape(messages: list[dict[str, Any]]) -> str:
    """Show a conversation as the list of its message labels.

    Args:
        messages: A request's ``messages``.

    Returns:
        For example ``[user, system, assistant tool_use, user tool_result]``.
    """
    return "[" + ", ".join(message_label(message) for message in messages) + "]"


def system_text(body: dict[str, Any]) -> str:
    """Return a request's system prompt as one string.

    Args:
        body: The request body.

    Returns:
        The text of every system block, joined by new lines.
    """
    system = body.get("system") or []
    if isinstance(system, str):
        return system
    return "\n".join(block.get("text", "") for block in system)


def reply_summary(record: dict[str, Any]) -> str:
    """Say in one line what the stand-in answered.

    Args:
        record: One logged model call.

    Returns:
        ``Bash: <command>`` for a tool call, the text for a plain reply.
    """
    parts = []
    for block in record["reply"].get("message", {}).get("content", []):
        if block["type"] == TOOL_USE:
            parts.append(f"{block['name']}: {block['input'].get('command', '')}")
        else:
            parts.append(f"text: {block.get('text', '')}")
    return " + ".join(parts) or "(nothing)"


def tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool_result blocks in a request's last message.

    Args:
        body: The request body.

    Returns:
        The blocks (empty when the last message carries none).
    """
    messages = body.get("messages") or []
    if not messages:
        return []
    found = blocks(messages[-1]["content"])
    return [block for block in found if block.get("type") == TOOL_RESULT]
