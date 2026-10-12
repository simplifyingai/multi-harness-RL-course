# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check the proxy's record against Claude Code's own log, call by call and row by row.

The capture proxy says what it recorded; the harness keeps its own account
of what it did. They are written by different programs, so where they agree
the capture can be trusted, and where they disagree something was lost. The
article checks its captures this way against Harbor's trajectory files;
here the harness's account is Claude Code's stream-json log.

For a run folder (``rollout.json``, the proxy's record, and
``stream.jsonl``, Claude Code's), it compares, as chapter 21's copy did:

* **turns**: every call the proxy recorded (the discarded ones aside)
  against every reply in Claude Code's log, in order. A reply with nothing
  in it gets no message in the log, only its raw stream events, so replies
  are counted from those; the proxy's own answers once a session's call
  budget is spent are in the log but were never recorded, so they are set
  aside;
* **replies**: the tool call (or text) of each, on both sides;
* **completion tokens**: what the engine sampled for each call (the
  proxy's ``n_sampled``) against what Claude Code was told it received
  (each reply's ``message_delta`` usage, in the log's raw stream events);
* **trainable coverage**: how many of each call's tokens the agent rows
  train (mask 1) against the tokens Claude Code was told that reply held.
  A call the rows do not train is named, with the proxy's finding about it.

and adds the check chapter 21's copy lacked, which passed a rollout split
into one row per call:

* **calls per row**: Claude Code's log is one conversation, or one more for
  each compaction it reports (a ``compact_boundary`` event), and each
  conversation should be one agent row holding all its calls. A row that
  starts where the log shows no new conversation is a split nothing
  announced; the check says where, and why, from the proxy's own record of
  that call against the call before (a system prompt that grew, a history
  that was replaced, or tokens that no longer continue).

The proxy's own verdict, ``trainable``, is printed last for contrast.

Run ``uv run crosscheck.py`` (the capture), ``uv run crosscheck.py note``
(any run ``breakages.py`` kept), a folder, or ``uv run crosscheck.py all``
(the capture and every case, then a table of which check caught what). It
exits with status 1 when anything disagrees. Only Python's standard library
is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from captures import (
    CAPTURE,
    CASE_RUNS,
    ROLLOUT_FILE,
    clip_text,
    named_run,
    system_change,
    unreadable,
)
from captures import text_of as message_text
from lfm_files import short_path
from rollout_view import INDENT, clip

SCRIPT = "crosscheck.py"
STREAM_FILE = "stream.jsonl"
ALL = "all"
# The capture, then breakages.py's cases, in the order breakages.py runs them.
RUN_ORDER = (CAPTURE, *CASE_RUNS)
AGENT = "agent"
REPLY_CHARS = 40
OK, MISMATCH = "ok", "MISMATCH"
NO_COUNTS_NOTE = (
    " (the log states no count for some replies: was Claude Code run with "
    "--include-partial-messages?)"
)
UNNAMED = "?"
MAX_TOKENS = "max_tokens"  # Anthropic's stop reason for a reply the cap cut off
EMPTY = "(empty)"
CUT_EMPTY = "(empty, cut at the output cap)"
SHOWN_REPLY_CHARS = 60  # Claude Code's reply, when it differs from the proxy's
COMPACT_BOUNDARY = "compact_boundary"  # Claude Code's mark for a compaction
SYSTEM = "system"
SHOWN_MESSAGE_CHARS = 50
PROXY_ANSWER_CHARS = 50  # the proxy's own answer, when shown
RULE = "━━"
CHECK_NAMES = (
    "turns",
    "replies",
    "completion tokens",
    "trainable coverage",
    "calls per row",
)
SHORT_NAMES = ("turns", "replies", "tokens", "coverage", "calls/row")


class RunError(Exception):
    """Raised when a run folder lacks the files to compare."""


@dataclass(frozen=True)
class LoggedReply:
    """One reply, as Claude Code's own log tells it.

    Attributes:
        message_id: The reply's id.
        summary: Its tool calls (or text), as one line; "" for an empty reply.
        output_tokens: The tokens Claude Code was told the reply held, or
            None when the log has no stream events.
        stop_reason: Why the reply ended (``tool_use``, ``end_turn``,
            ``max_tokens``), or None when the log does not say.
    """

    message_id: str
    summary: str
    output_tokens: int | None
    stop_reason: str | None = None

    @property
    def cut_and_empty(self) -> bool:
        """True for a reply with nothing in it that the output cap cut off."""
        return not self.summary and self.stop_reason == MAX_TOKENS


@dataclass(frozen=True)
class Check:
    """The verdict of one comparison.

    Attributes:
        name: What was compared.
        ok: Whether both sides agree.
        detail: The numbers behind it.
        why: Lines naming each disagreement's cause, when there are any.
    """

    name: str
    ok: bool
    detail: str
    why: tuple[str, ...] = ()


@dataclass(frozen=True)
class Log:
    """What Claude Code's own log says about a run.

    Attributes:
        replies: Every reply, in order.
        compactions: For each compaction it reports, how many replies came
            before it.
        events: All its events.
    """

    replies: list[LoggedReply]
    compactions: list[int]
    events: list[dict[str, Any]]


def read_rollout(folder: Path) -> dict[str, Any]:
    """Read the proxy's record from a run folder.

    Args:
        folder: The run's folder.

    Returns:
        The rollout document.

    Raises:
        RunError: If it cannot be read or is not a rollout document.
    """
    path = folder / ROLLOUT_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunError(unreadable(path, error)) from error
    if not isinstance(document, dict) or "turns" not in document:
        raise RunError(f"{short_path(path)} is not a rollout document")
    return document


def run_folder(name: str) -> Path:
    """Find a run folder by name or path.

    Args:
        name: Such as ``capture`` or ``note``, or a folder.

    Returns:
        The folder.

    Raises:
        RunError: If it does not hold both files.
    """
    folder = named_run(name)
    for needed in (ROLLOUT_FILE, STREAM_FILE):
        if not (folder / needed).is_file():
            raise RunError(
                f"{short_path(folder)} has no {needed}: run capture.py "
                "(and breakages.py) first"
            )
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


def read_log(events: list[dict[str, Any]]) -> Log:
    """Read every reply, and every compaction, out of Claude Code's log.

    Claude Code prints one ``assistant`` event per finished part of a reply,
    all with the reply's id, and none for a reply with nothing in it; with
    ``--include-partial-messages`` it also prints the raw stream of every
    reply: a ``message_start`` naming the id, and a ``message_delta`` with
    the reply's ``output_tokens`` and its stop reason. So the replies are
    counted from both, in the order they first appear. A compaction is a
    ``system`` event with the subtype ``compact_boundary``.

    Args:
        events: The log's events.

    Returns:
        The replies (empty ones included) and the compactions.
    """
    parts: dict[str, list[str]] = {}  # insertion order is the replies' order
    tokens: dict[str, int] = {}
    stops: dict[str, str] = {}
    compactions: list[int] = []
    current = ""
    for event in events:
        kind = event.get("type")
        if kind == SYSTEM and event.get("subtype") == COMPACT_BOUNDARY:
            compactions.append(len(parts))
        elif kind == "assistant":
            message = event.get("message") or {}
            if not message.get("id"):
                continue  # not a model reply (Claude Code wrote it itself)
            pieces = parts.setdefault(message["id"], [])
            for block in message.get("content") or []:
                if block.get("type") == "tool_use":
                    name = block.get("name", UNNAMED)
                    pieces.append(call_summary(name, block.get("input")))
                elif block.get("type") == "text":
                    pieces.append(f"text: {block.get('text', '')}")
        elif kind == "stream_event":
            raw = event.get("event") or {}
            if raw.get("type") == "message_start":
                current = (raw.get("message") or {}).get("id", "")
                if current:
                    parts.setdefault(current, [])  # seen, even if never printed
            elif raw.get("type") == "message_delta" and current:
                told = (raw.get("usage") or {}).get("output_tokens")
                if isinstance(told, int):  # absent stays absent, not 0
                    tokens[current] = told
                stop = (raw.get("delta") or {}).get("stop_reason")
                if isinstance(stop, str):
                    stops[current] = stop
    replies = [
        LoggedReply(
            message_id, "; ".join(pieces), tokens.get(message_id), stops.get(message_id)
        )
        for message_id, pieces in parts.items()
    ]
    return Log(replies, compactions, events)


def logged_replies(events: list[dict[str, Any]]) -> list[LoggedReply]:
    """Read every reply out of Claude Code's stream-json log, in order.

    Args:
        events: The log's events.

    Returns:
        One entry per reply, empty ones included.
    """
    return read_log(events).replies


def proxy_answers(
    turns: list[dict[str, Any]], replies: list[LoggedReply], budget_stops: int
) -> list[LoggedReply]:
    """Find the replies the proxy wrote itself: its answers once the budget ran out.

    The proxy answers a call over the session's budget with a fixed text and
    records no turn for it, so those replies are in Claude Code's log only:
    at its end, with no tokens.

    Args:
        turns: The proxy's turns (the discarded ones aside).
        replies: Claude Code's replies, in order.
        budget_stops: The document's ``budget_stop_count``.

    Returns:
        The trailing replies that are the proxy's own, or none.
    """
    extra = replies[len(turns) :]
    if extra and len(extra) <= budget_stops and not any(r.output_tokens for r in extra):
        return extra
    return []


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


def trained_per_call(
    document: dict[str, Any], turns: list[dict[str, Any]]
) -> list[int]:
    """Count, for each call, how many of its reply's tokens an agent row trains.

    A row is built along its path, so a call's reply is the row's tokens
    from its ``n_prompt`` on, ``n_sampled`` of them.

    Args:
        document: The proxy's record (every row).
        turns: The calls to count, in order.

    Returns:
        One count per call: its tokens with mask 1 in the agent row that
        trains most of them (0 when no agent row holds it).
    """
    counts = []
    for turn in turns:
        start, end = turn["n_prompt"], turn["n_prompt"] + turn["n_sampled"]
        counts.append(
            max(
                (
                    sum(row["loss_mask"][start:end])
                    for row in document["sequences"]
                    if row["role"] == AGENT and turn["node_id"] in row["node_ids"]
                ),
                default=0,
            )
        )
    return counts


def turn_findings(document: dict[str, Any], turn: dict[str, Any]) -> list[str]:
    """The proxy's findings about one call.

    Args:
        document: The proxy's record.
        turn: The call (the proxy names it by its ``index``).

    Returns:
        Its findings (``... turn <index>: ...``).
    """
    tag = f" turn {turn['index']}:"
    return [f for f in document.get("validation") or [] if tag in f]


def agent_rows(
    document: dict[str, Any], turns: list[dict[str, Any]]
) -> list[list[int]]:
    """List the calls each agent row holds, rows in call order.

    Calls are numbered as Claude Code's log numbers its replies: by their
    place among the turns compared (the discarded ones set aside).

    Args:
        document: The proxy's record (every row).
        turns: The calls compared, in order.

    Returns:
        One list of call numbers per agent row.
    """
    number = {turn["node_id"]: n for n, turn in enumerate(turns, start=1)}
    rows = [
        [number[node] for node in row["node_ids"] if node in number]
        for row in document["sequences"]
        if row["role"] == AGENT
    ]
    return sorted(row for row in rows if row)


def conversations(calls: int, compactions: list[int]) -> list[list[int]]:
    """Cut a run's calls into the conversations Claude Code's log reports.

    Args:
        calls: How many calls the run made.
        compactions: How many replies came before each compaction.

    Returns:
        One list of call numbers (from 1) per conversation.
    """
    cuts = sorted({c for c in compactions if 0 < c < calls})
    edges = [0, *cuts, calls]
    return [list(range(a + 1, b + 1)) for a, b in zip(edges, edges[1:], strict=False)]


def calls_text(groups: list[list[int]]) -> str:
    """Write groups of calls compactly.

    Args:
        groups: Lists of call numbers.

    Returns:
        Such as ``calls 1-4`` or ``call 1 | call 2``.
    """
    shown = []
    for group in groups:
        if len(group) == 1:
            shown.append(f"call {group[0]}")
        elif group == list(range(group[0], group[-1] + 1)):
            shown.append(f"calls {group[0]}-{group[-1]}")
        else:
            shown.append("calls " + ", ".join(map(str, group)))
    return " | ".join(shown) or "none"


def why_new_row(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, str]:
    """Say why a call starts a row of its own, from the proxy's record of both.

    Continuing the call before means a prompt that begins with that call's
    prompt, then its reply and what came back. So the first message where
    the two prompts part says what happened.

    Args:
        before: The call before (with its messages).
        after: The call that starts a new row.

    Returns:
        A kind (``system``, ``history``, ``message`` or ``tokens``) and the
        detail.
    """
    old = before.get("request_messages") or []
    new = after.get("request_messages") or []
    parted = next(
        (i for i, pair in enumerate(zip(old, new, strict=False)) if pair[0] != pair[1]),
        None,
    )
    if parted is None and len(new) < len(old) + 2:  # cut short, or the same
        return (
            "history",
            f"its history was cut: {len(new)} messages, where continuing the call "
            f"before takes at least {len(old) + 2}",
        )
    if parted is None:  # the messages continue: the tokens do not
        return (
            "tokens",
            "its messages continue the call before's, its tokens do not "
            "(the reply came back as other tokens)",
        )
    if parted == 0 and new[0].get("role") == SYSTEM:
        change = system_change(
            message_text(old[0].get("content")), message_text(new[0].get("content"))
        )
        return "system", f"its system prompt {change}"
    if len(new) < len(old) + 2:
        start = clip_text(message_text(new[parted].get("content")), SHOWN_MESSAGE_CHARS)
        return (
            "history",
            f"its history was replaced: {len(new)} messages, where continuing the "
            f"call before takes at least {len(old) + 2}; message {parted + 1} now "
            f"begins {json.dumps(start, ensure_ascii=False)}",
        )
    return (
        "message",
        f"its message {parted + 1} ({new[parted].get('role')}) differs from the "
        "call before's",
    )


def split_causes(
    turns: list[dict[str, Any]], rows: list[list[int]], expected: list[list[int]]
) -> tuple[str, ...]:
    """Name why each unannounced row starts where it does, one line per cause.

    Calls with the same kind of cause are told together, with the first
    one's detail.

    Args:
        turns: The calls, in order (with messages).
        rows: The calls each agent row holds.
        expected: The calls each logged conversation holds.

    Returns:
        The lines.
    """
    announced = {group[0] for group in expected}
    found_starts = {row[0] for row in rows}
    starts = sorted(found_starts - announced)
    lines = []
    held = {call for row in rows for call in row}
    missing = [call for group in expected for call in group if call not in held]
    if missing:
        lines.append(f"calls {', '.join(map(str, missing))} are in no agent row")
    for call in sorted(announced - found_starts - {1}):
        lines.append(
            f"Claude Code's log compacts before call {call}, yet no row starts there"
        )
    grouped: dict[str, list[tuple[int, str]]] = {}
    for call in starts:
        if call < 2 or call > len(turns):
            continue
        kind, detail = why_new_row(turns[call - 2], turns[call - 1])
        grouped.setdefault(kind, []).append((call, detail))
    for found in grouped.values():
        calls = [call for call, _ in found]
        first, detail = found[0]
        if len(calls) == 1:
            lines.append(f"call {first} starts a row nothing announced: {detail}")
            continue
        named = ", ".join(map(str, calls))
        lines.append(
            f"calls {named} each start a row nothing announced; call {first}: {detail}"
        )
    return tuple(lines)


def compare(document: dict[str, Any], log: Log) -> list[Check]:
    """Compare the proxy's record with Claude Code's log, printing each call.

    Args:
        document: The proxy's rollout document (with messages, every row).
        log: Claude Code's account.

    Returns:
        The five verdicts.
    """
    turns = sorted(
        (turn for turn in document["turns"] if not turn.get("discarded")),
        key=lambda t: t["index"],
    )
    dropped = len(document["turns"]) - len(turns)
    budget_stops = int(document.get("budget_stop_count") or 0)
    replies = log.replies
    by_proxy = proxy_answers(turns, replies, budget_stops)
    replies = replies[: len(replies) - len(by_proxy)]
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
        shown = recorded or (CUT_EMPTY if reply.cut_and_empty else EMPTY)
        print(
            f"{INDENT}{number:>4}  {turn['n_sampled']:>12,}  {told:>17}  "
            f"{clip(shown, REPLY_CHARS):<{REPLY_CHARS}}  {verdict}"
        )
        if not reply_match:
            shown = clip(reply.summary, SHOWN_REPLY_CHARS)
            print(f"{INDENT}      Claude Code's reply: {shown}")
    for reply in by_proxy:
        shown = clip(reply.summary, PROXY_ANSWER_CHARS)
        print(f"{INDENT}   +  answered by the proxy itself: {shown}")
    paired = replies[: len(turns)]  # a reply the proxy never recorded has no size
    cut = [n for n, r in enumerate(paired, start=1) if r.cut_and_empty]
    if cut:
        held = sum(turns[n - 1]["n_sampled"] for n in cut)
        print(
            f"{INDENT}note: calls {', '.join(map(str, cut))} came back empty, cut at "
            f"the output cap: Claude Code printed no message for them, and the "
            f"rows hold their {held:,} tokens"
        )
    extra = f" ({dropped} discarded turns set aside)" if dropped else ""
    budget_note = ""
    if by_proxy:
        budget_note = (
            f" (+{len(by_proxy)} answered by the proxy itself: the call budget)"
        )
    counted = [r.output_tokens for r in replies]
    sampled = [turn["n_sampled"] for turn in turns]
    trained = trained_per_call(document, turns)
    untrained = tuple(
        f"call {n}: Claude Code was told {told:,} tokens, the rows train "
        f"{got:,}" + "".join(f" ({f})" for f in turn_findings(document, turn))
        for n, (turn, told, got) in enumerate(
            zip(turns, counted, trained, strict=False), start=1
        )
        if told is not None and got != told
    )
    expected = conversations(len(turns), log.compactions)
    rows = agent_rows(document, turns)
    compacted = ""
    if log.compactions:
        compacted = f", compacted {len(log.compactions)} time(s)"
    groups = len(expected)
    return [
        Check(
            "turns",
            len(turns) == len(replies),
            f"{len(turns)}/{len(replies)}: {len(turns)} recorded{extra}, "
            f"{len(replies)} in Claude Code's log{budget_note}",
        ),
        Check("replies", replies_ok and len(turns) == len(replies), "call by call"),
        Check(
            "completion tokens",
            tokens_ok and counted == sampled,
            f"proxy {sum(sampled):,}, Claude Code {sum(c or 0 for c in counted):,}"
            + (NO_COUNTS_NOTE if None in counted else ""),
        ),
        Check(
            "trainable coverage",
            trained == counted,
            f"agent rows train {trained}; Claude Code's replies {counted}",
            untrained,
        ),
        Check(
            "calls per row",
            rows == expected,
            f"Claude Code's log: {groups} conversation{'s' * (groups != 1)} "
            f"({calls_text(expected)}{compacted}); agent rows: {calls_text(rows)}",
            () if rows == expected else split_causes(turns, rows, expected),
        ),
    ]


def read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSON-lines log, counting the lines that are not JSON objects.

    Args:
        path: The file.

    Returns:
        The events, and how many lines were skipped (a log cut off mid-line
        ends with one).

    Raises:
        RunError: If the file cannot be read.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RunError(unreadable(path, error)) from error
    events, skipped = [], 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict):
            events.append(event)
        elif line.strip():
            skipped += 1
    return events, skipped


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


def proxy_verdict(document: dict[str, Any]) -> str:
    """Say what the proxy itself concluded, for contrast with the checks.

    Args:
        document: The proxy's record.

    Returns:
        Its verdict, roots, rows and the findings it raised.
    """
    stats = document.get("stats") or {}
    findings = [
        f.split(":", 1)[0]
        for f in document.get("validation") or []
        if not f.startswith("[INFO]")
    ]
    raised = ", ".join(dict.fromkeys(findings)) or "none"
    return (
        f"the proxy's verdict: trainable {str(document.get('trainable')).lower()} · "
        f"n_roots {stats.get('n_roots')} · n_sequences {stats.get('n_sequences')} · "
        f"findings {raised}"
    )


def check_run(folder: Path) -> list[Check]:
    """Compare one run's two accounts and print every verdict.

    Args:
        folder: The run's folder.

    Returns:
        The checks.

    Raises:
        RunError: If the proxy's record or Claude Code's log cannot be read.
    """
    document = read_rollout(folder)
    events, skipped = read_events(folder / STREAM_FILE)
    print(f"{SCRIPT} · {short_path(folder)}: the proxy's record vs Claude Code's log")
    checks = compare(document, read_log(events))
    print(f"{INDENT}{result_line(events)}")
    if skipped:
        print(f"{INDENT}note: {skipped} line(s) of Claude Code's log were not JSON")
    for check in checks:
        print(f"{check.name:<19} {OK if check.ok else MISMATCH} · {check.detail}")
        for line in check.why:
            print(f"{'':<19} why: {line}")
    failed = sum(not check.ok for check in checks)
    print(proxy_verdict(document))
    verdict = "agree" if not failed else f"disagree on {failed} of {len(checks)} checks"
    print(f"the cross-check: the two accounts {verdict}")
    return checks


def print_table(results: list[tuple[str, list[Check], dict[str, Any]]]) -> None:
    """Print which check caught what, run by run.

    Args:
        results: Each run's name, checks and the proxy's record.
    """
    width = max(len(name) for name, _, _ in results) + 1
    print(f"\n{RULE} the proxy's record against Claude Code's log, run by run")
    header = " ".join(f"{name:<9}" for name in SHORT_NAMES)
    print(f"{INDENT}{'run':<{width}} {header} proxy: trainable")
    for name, checks, document in results:
        cells = " ".join(f"{OK if c.ok else MISMATCH:<9}" for c in checks)
        trainable = str(document.get("trainable")).lower()
        print(f"{INDENT}{name:<{width}} {cells} {trainable}")


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default=CAPTURE, help="a run, or all")
    return parser.parse_args()


def main() -> None:
    """Check one run, or every run, against Claude Code's own log.

    Raises:
        SystemExit: If a run is missing, or (status 1) anything disagrees.
    """
    args = parse_args()
    names = [args.run]
    if args.run == ALL:
        kept = [name for name in RUN_ORDER if (named_run(name) / STREAM_FILE).is_file()]
        names = kept
        if not names:
            raise SystemExit("no runs yet: run capture.py, then breakages.py")
    results = []
    for number, name in enumerate(names):
        if number:
            print()
        try:
            folder = run_folder(name)
            results.append((name, check_run(folder), read_rollout(folder)))
        except RunError as error:
            raise SystemExit(str(error)) from error
    if len(results) > 1:
        print_table(results)
    if not all(check.ok for _, checks, _ in results for check in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
