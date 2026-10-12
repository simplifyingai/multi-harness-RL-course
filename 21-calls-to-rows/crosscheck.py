# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check the proxy's record against Claude Code's own log, call by call.

The capture proxy says what it recorded; the harness keeps its own account
of what it did. They are written by different programs, so where they agree
the capture can be trusted, and where they disagree something was lost. The
article checks its captures this way against Harbor's trajectory files;
here the harness's account is Claude Code's stream-json log.

For a run folder written by ``reminder.py`` (``rollout.json``, the proxy's
record, and ``stream.jsonl``, Claude Code's), it compares:

* **turns**: every call the proxy recorded (the discarded ones aside)
  against every reply in Claude Code's log, in order;
* **replies**: the tool call (or text) of each, on both sides;
* **completion tokens**: what the engine sampled for each call (the
  proxy's ``n_sampled``) against what Claude Code was told it received
  (each reply's ``message_delta`` usage, in the log's raw stream events);
* **trainable coverage**: the reply lengths the agent rows train
  (``turn_lengths``, the proxy's join key against a harness trace) against
  the same per-call token counts. A turn without log-probabilities, a lost
  turn or a split row shows up here, though every reply matched.

Run ``uv run crosscheck.py`` (the ``reminder-off`` run),
``uv run crosscheck.py reminder-default``, or give a folder. It exits with
status 1 when anything disagrees. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm_files import chapter_home, short_path
from rollout_view import INDENT, clip

SCRIPT = "crosscheck.py"
RUNS_FOLDER = "runs"
DEFAULT_RUN = "reminder-off"
ROLLOUT_FILE = "rollout.json"
STREAM_FILE = "stream.jsonl"
AGENT = "agent"
REPLY_CHARS = 40
OK, MISMATCH = "ok", "MISMATCH"


class RunError(Exception):
    """Raised when a run folder lacks the files to compare."""


@dataclass(frozen=True)
class LoggedReply:
    """One reply, as Claude Code's own log tells it.

    Attributes:
        message_id: The reply's id.
        summary: Its tool calls (or text), as one line.
        output_tokens: The tokens Claude Code was told the reply held, or
            None when the log has no stream events.
    """

    message_id: str
    summary: str
    output_tokens: int | None


@dataclass(frozen=True)
class Check:
    """The verdict of one comparison.

    Attributes:
        name: What was compared.
        ok: Whether both sides agree.
        detail: The numbers behind it.
    """

    name: str
    ok: bool
    detail: str


def run_folder(name: str) -> Path:
    """Find a run folder by name or path.

    Args:
        name: Such as ``reminder-off``, or a folder.

    Returns:
        The folder.

    Raises:
        RunError: If it does not hold both files.
    """
    folder = Path(name)
    if not folder.is_dir():
        folder = chapter_home() / RUNS_FOLDER / name
    for needed in (ROLLOUT_FILE, STREAM_FILE):
        if not (folder / needed).is_file():
            raise RunError(f"{folder} has no {needed}: run reminder.py first")
    return folder


def call_summary(name: str, arguments: dict[str, Any] | str | None) -> str:
    """Write one tool call as ``name: command``.

    Args:
        name: The tool's name.
        arguments: Its input (a mapping) or its JSON text.

    Returns:
        The line.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{name}: {arguments}"
    command = arguments.get("command") if isinstance(arguments, dict) else None
    return f"{name}: {command if command is not None else json.dumps(arguments)}"


def logged_replies(events: list[dict[str, Any]]) -> list[LoggedReply]:
    """Read every reply out of Claude Code's stream-json log, in order.

    Claude Code prints one ``assistant`` event per finished part of a reply,
    all with the reply's id; with ``--include-partial-messages`` it also
    prints the raw stream: a ``message_start`` naming the id, and a
    ``message_delta`` with the reply's ``output_tokens``.

    Args:
        events: The log's events.

    Returns:
        One entry per reply.
    """
    parts: dict[str, list[str]] = {}
    tokens: dict[str, int] = {}
    current = ""
    for event in events:
        kind = event.get("type")
        if kind == "assistant":
            message = event["message"]
            pieces = parts.setdefault(message["id"], [])
            for block in message.get("content") or []:
                if block.get("type") == "tool_use":
                    pieces.append(call_summary(block["name"], block.get("input")))
                elif block.get("type") == "text":
                    pieces.append(f"text: {block.get('text', '')}")
        elif kind == "stream_event":
            raw = event.get("event") or {}
            if raw.get("type") == "message_start":
                current = (raw.get("message") or {}).get("id", "")
            elif raw.get("type") == "message_delta" and current:
                usage = raw.get("usage") or {}
                tokens[current] = int(usage.get("output_tokens") or 0)
    return [
        LoggedReply(message_id, "; ".join(pieces), tokens.get(message_id))
        for message_id, pieces in parts.items()
    ]


def recorded_summary(turn: dict[str, Any]) -> str:
    """Write a recorded turn's reply the same way as a logged one.

    Args:
        turn: One of the proxy's turns (with messages).

    Returns:
        Its tool calls (or text), as one line.
    """
    message = turn.get("response_message") or {}
    pieces = [
        call_summary(
            (call.get("function") or {}).get("name", ""),
            (call.get("function") or {}).get("arguments"),
        )
        for call in message.get("tool_calls") or []
    ]
    if not pieces and message.get("content"):
        pieces.append(f"text: {message['content']}")
    return "; ".join(pieces)


def same_reply(recorded: str, logged: str) -> bool:
    """Say whether two reply summaries name the same thing.

    Args:
        recorded: The proxy's.
        logged: Claude Code's.

    Returns:
        True when equal once their spacing is normalised.
    """
    return " ".join(recorded.split()) == " ".join(logged.split())


def compare(document: dict[str, Any], replies: list[LoggedReply]) -> list[Check]:
    """Compare the proxy's record with Claude Code's log, printing each call.

    Args:
        document: The proxy's rollout document (with messages, every row).
        replies: Claude Code's replies, in order.

    Returns:
        The four verdicts.
    """
    turns = [turn for turn in document["turns"] if not turn.get("discarded")]
    dropped = len(document["turns"]) - len(turns)
    print(f"{INDENT}call  proxy: wrote  Claude Code: told  reply")
    replies_ok = tokens_ok = True
    for number, (turn, reply) in enumerate(zip(turns, replies, strict=False), start=1):
        recorded = recorded_summary(turn)
        reply_match = same_reply(recorded, reply.summary)
        token_match = reply.output_tokens == turn["n_sampled"]
        replies_ok &= reply_match
        tokens_ok &= token_match
        told = "-" if reply.output_tokens is None else f"{reply.output_tokens:,}"
        verdict = OK if reply_match and token_match else MISMATCH
        print(
            f"{INDENT}{number:>4}  {turn['n_sampled']:>12,}  {told:>17}  "
            f"{clip(recorded, REPLY_CHARS):<{REPLY_CHARS}}  {verdict}"
        )
        if not reply_match:
            print(f"{INDENT}      Claude Code's reply: {clip(reply.summary, 60)}")
    extra = f" ({dropped} discarded turns set aside)" if dropped else ""
    counted = [r.output_tokens for r in replies]
    sampled = [turn["n_sampled"] for turn in turns]
    trained = [
        length
        for row in document["sequences"]
        if row["role"] == AGENT
        for length in row["turn_lengths"]
    ]
    return [
        Check(
            "turns",
            len(turns) == len(replies),
            f"{len(turns)}/{len(replies)}: {len(turns)} recorded{extra}, "
            f"{len(replies)} in Claude Code's log",
        ),
        Check("replies", replies_ok and len(turns) == len(replies), "call by call"),
        Check(
            "completion tokens",
            tokens_ok and counted == sampled,
            f"proxy {sum(sampled):,}, Claude Code {sum(c or 0 for c in counted):,}",
        ),
        Check(
            "trainable coverage",
            trained == counted,
            f"agent rows train {trained}; Claude Code's replies {counted}",
        ),
    ]


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines log, skipping lines that are not JSON.

    Args:
        path: The file.

    Returns:
        The events.
    """
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Claude Code prints only JSON here; anything else is noise
    return events


def result_line(events: list[dict[str, Any]]) -> str:
    """Say what Claude Code's own result event reports, if there is one.

    Args:
        events: The log's events.

    Returns:
        Its turn count and output tokens, or a note that it is missing.
    """
    for event in reversed(events):
        if event.get("type") == "result":
            usage = event.get("usage") or {}
            written, read = usage.get("output_tokens"), usage.get("input_tokens")
            return (
                f"Claude Code's result: {event.get('subtype')} · "
                f"{event.get('num_turns')} turns · output_tokens {written} · "
                f"input_tokens {read}"
            )
    return "Claude Code's log has no result event: it did not finish"


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default=DEFAULT_RUN)
    return parser.parse_args()


def main() -> None:
    """Compare one run's two accounts and print the verdicts.

    Raises:
        SystemExit: If the run is missing, or (status 1) anything disagrees.
    """
    args = parse_args()
    try:
        folder = run_folder(args.run)
    except RunError as error:
        raise SystemExit(str(error)) from error
    document = json.loads((folder / ROLLOUT_FILE).read_text(encoding="utf-8"))
    events = read_events(folder / STREAM_FILE)
    print(f"{SCRIPT} · {short_path(folder)}: the proxy's record vs Claude Code's log")
    checks = compare(document, logged_replies(events))
    print(f"{INDENT}{result_line(events)}")
    for check in checks:
        print(f"{check.name:<19} {OK if check.ok else MISMATCH} · {check.detail}")
    if not all(check.ok for check in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
