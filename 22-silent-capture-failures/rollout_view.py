# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print a capture proxy's rollout document as data: graph, rows, findings, stats.

The proxy turns every model call of a rollout into one **turn** (a node),
links each turn to the earlier turn whose prompt and reply its own prompt
begins with (its parent), and flattens every path from a root to a leaf into
one training **row**: ``input_ids``, a ``loss_mask`` (1 on the model's own
tokens, 0 on everything it only read) and the ``logprobs`` of the trained
tokens. This module prints those pieces the way the chapter reads them:

* the graph, one line per turn: its parent, its size, its reply, and
  whether it was discarded;
* each row, span by span: the context it read (mask 0) and each reply it
  wrote (mask 1, with its first log-probabilities, or mask 0 when the turn
  came without them);
* the proxy's findings, one line each, and its stats.

The document's ``turns`` carry no parent, so parents are read off the rows'
paths (``node_ids``): ask the proxy for every row
(``include_discarded=true``) to see the whole graph. Replies are shown when
the document was asked for with ``include_messages=true``. Only Python's
standard library is needed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from lfm_files import tilde

INDENT = "  "
NO_PARENT = "-"
CONTEXT = "context"
MODEL = "model"
DISCARDED = "discarded"
LOGPROBS_SHOWN = 4  # the first few log-probabilities of a reply
REPLY_CHARS = 46  # a reply's summary is cut to this many characters
FINDING_CHARS = 110  # a finding is cut at its first sentence, or here
SENTENCE_END = ". "
MISSING = "-"  # a log-probability the row does not have
NEGATIVE_NOTE = "(shorter than its parent: the turns do not add up to this row)"
ELLIPSIS = "…"
# ANSI colours, only when asked or when printing to a terminal.
STYLES = {
    "trained": "\033[32m",  # green: tokens a trainer learns from
    "untrained": "\033[33m",  # yellow: the model's tokens, masked out
    "context": "\033[2m",  # dim: tokens the model only read
    "fatal": "\033[31m",
    "warn": "\033[33m",
    "head": "\033[1m",
}
RESET = "\033[0m"
COLOR_CHOICES = ("auto", "always", "never")


@dataclass(frozen=True)
class Span:
    """One stretch of a row: what a turn added before its reply, or the reply.

    Attributes:
        turn: The turn's label, such as ``t0``.
        kind: ``context`` (mask 0: the model read it) or ``model`` (its reply).
        start: The span's first position in the row.
        end: One past its last position.
        ones: How many of its positions have mask 1.
    """

    turn: str
    kind: str
    start: int
    end: int
    ones: int

    @property
    def size(self) -> int:
        """Number of tokens in the span."""
        return self.end - self.start

    @property
    def trained(self) -> bool:
        """True when the mask is 1 over the whole (non-empty) span."""
        return self.size > 0 and self.ones == self.size

    def mask_label(self) -> str:
        """Say the span's mask: 0, 1, or how much of it is 1.

        Returns:
            Such as ``mask 1``, ``mask 0`` or ``mask 1 on 3 of 40``.
        """
        if self.trained:
            return "mask 1"
        if self.ones == 0:
            return "mask 0"
        return f"mask 1 on {self.ones:,} of {self.size:,}"


class Painter:
    """Colours text for a terminal, or leaves it plain."""

    def __init__(self, color: str = "auto") -> None:
        """Decide once whether to colour.

        Args:
            color: ``auto`` (only on a terminal), ``always`` or ``never``.
        """
        self.on = color == "always" or (color == "auto" and sys.stdout.isatty())

    def paint(self, text: str, style: str) -> str:
        """Wrap text in a colour when colouring is on.

        Args:
            text: The text.
            style: A key of ``STYLES``.

        Returns:
            The text, coloured or not.
        """
        return f"{STYLES[style]}{text}{RESET}" if self.on else text


def labels(document: dict[str, Any]) -> dict[str, str]:
    """Name each turn by its arrival order, as the proxy's findings do.

    Args:
        document: A rollout document.

    Returns:
        ``node_id -> "t<index>"``.
    """
    return {turn["node_id"]: f"t{turn['index']}" for turn in document["turns"]}


def parents(document: dict[str, Any]) -> dict[str, str | None]:
    """Read each turn's parent off the rows' paths.

    Args:
        document: A rollout document (with ``include_discarded=true`` for the
            whole graph).

    Returns:
        ``node_id -> parent node_id``, None for a root (or a turn on no row).
    """
    found: dict[str, str | None] = {turn["node_id"]: None for turn in document["turns"]}
    for row in document["sequences"]:
        path = row["node_ids"]
        for parent, child in zip(path, path[1:], strict=False):
            found[child] = parent
    return found


def shape(document: dict[str, Any]) -> str:
    """Say the graph's shape in one line: roots, leaves, forks, discards.

    Args:
        document: A rollout document, every row included.

    Returns:
        Such as ``1 root (t0) · 2 leaves (t1, t4) · fork t0 -> t1, t2``.
    """
    names = labels(document)
    up = parents(document)
    children: dict[str, list[str]] = {}
    for child, parent in up.items():
        if parent is not None:
            children.setdefault(parent, []).append(child)
    order = [turn["node_id"] for turn in document["turns"]]
    roots = [names[n] for n in order if up[n] is None]
    leaves = [names[n] for n in order if n not in children]
    parts = [
        f"{len(roots)} root{'s' * (len(roots) != 1)} ({', '.join(roots)})",
        f"{len(leaves)} lea{'ves' if len(leaves) != 1 else 'f'} ({', '.join(leaves)})",
    ]
    for parent in order:
        kids = children.get(parent, [])
        if len(kids) > 1:
            parts.append(f"fork {names[parent]} -> {', '.join(names[k] for k in kids)}")
    gone = [names[t["node_id"]] for t in document["turns"] if t.get("discarded")]
    parts.append(f"discarded: {', '.join(gone) if gone else 'none'}")
    return " · ".join(parts)


def clip(text: str, limit: int) -> str:
    """Cut a text to one line of at most ``limit`` characters.

    Args:
        text: Any text.
        limit: The most characters kept.

    Returns:
        The text on one line (the home folder as ``~``), cut with an ellipsis
        when longer.
    """
    line = " ".join(tilde(str(text)).split())
    return line if len(line) <= limit else line[: limit - 1] + ELLIPSIS


def reply_summary(turn: dict[str, Any]) -> str:
    """Say what a turn's reply was: its tool calls, else its text.

    Args:
        turn: One entry of the document's ``turns`` (with messages).

    Returns:
        Such as ``bash: sqlite3 data.db .tables``, or "" when unknown.
    """
    message = turn.get("response_message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        detail = arguments.get("command") if isinstance(arguments, dict) else None
        calls.append(f"{function.get('name')}: {detail or function.get('arguments')}")
    if calls:
        return clip("; ".join(calls), REPLY_CHARS)
    content = message.get("content")
    return clip(f"text: {content}", REPLY_CHARS) if content else ""


def short_finding(finding: str) -> str:
    """Cut a finding at its first sentence.

    Args:
        finding: Such as ``[WARN] code: first sentence. More.``.

    Returns:
        The finding's level, code and first sentence.
    """
    head, _, _ = finding.partition(SENTENCE_END)
    return clip(head, FINDING_CHARS)


def finding_style(finding: str) -> str | None:
    """Pick the colour of a finding by its level.

    Args:
        finding: The finding.

    Returns:
        ``fatal``, ``warn`` or None.
    """
    if "FATAL" in finding[:12]:
        return "fatal"
    if "WARN" in finding[:12]:
        return "warn"
    return None


def spans(row: dict[str, Any], turns: dict[str, dict[str, Any]]) -> list[Span]:
    """Cut a row into what each turn added before its reply, and the reply.

    The proxy builds a row turn by turn along its path: a root contributes
    its whole prompt, a child the tokens its prompt adds after its parent's
    prompt and reply; then each turn's reply. So the spans follow from the
    turns' sizes alone.

    Args:
        row: One entry of the document's ``sequences``.
        turns: The document's turns, by node id.

    Returns:
        The spans, in order.
    """
    found: list[Span] = []
    position = 0
    previous_end = 0  # tokens of the parent's prompt plus its reply
    for node_id in row["node_ids"]:
        turn = turns[node_id]
        label = f"t{turn['index']}"
        mask = row["loss_mask"]
        context = turn["n_prompt"] - previous_end
        ones = sum(mask[position : position + max(context, 0)])
        found.append(Span(label, CONTEXT, position, position + context, ones))
        position += context
        sampled = turn["n_sampled"]
        ones = sum(mask[position : position + sampled])
        found.append(Span(label, MODEL, position, position + sampled, ones))
        position += sampled
        previous_end = turn["n_prompt"] + sampled
    return found


def logprob_line(row: dict[str, Any], span: Span) -> str:
    """Show a reply's first log-probabilities, as the row carries them.

    Args:
        row: The row.
        span: A ``model`` span of it.

    Returns:
        Such as ``logprobs -0.125 -0.250 -0.375 -0.500 … (30)``.
    """
    if span.ones == 0:
        return "no logprobs: mask 0"
    values = row["logprobs"][span.start : span.end]
    shown = " ".join(
        MISSING if value is None else f"{value:.3f}"
        for value in values[:LOGPROBS_SHOWN]
    )
    more = f" {ELLIPSIS}" if len(values) > LOGPROBS_SHOWN else ""
    return f"logprobs {shown}{more}"


def print_graph(document: dict[str, Any], painter: Painter) -> None:
    """Print the graph: its shape, then one line per turn.

    Args:
        document: A rollout document, every row and the messages included.
        painter: Colours, or none.
    """
    stats = document["stats"]
    print(
        f"graph     {stats['n_turns']} turns · {shape(document)}",
    )
    names = labels(document)
    up = parents(document)
    print(f"{INDENT}turn  parent  read   wrote  tools  reply")
    for turn in document["turns"]:
        parent = up[turn["node_id"]]
        note = painter.paint(f"  {DISCARDED}", "untrained") if turn["discarded"] else ""
        parent_name = names[parent] if parent else NO_PARENT
        print(
            f"{INDENT}{names[turn['node_id']]:<5} {parent_name:<7}"
            f"{turn['n_prompt']:>5,}  {turn['n_sampled']:>5,}  {turn['n_tools']:>5}  "
            f"{reply_summary(turn)}{note}"
        )


def print_row(
    number: int,
    row: dict[str, Any],
    turns: dict[str, dict[str, Any]],
    painter: Painter,
) -> None:
    """Print one row: its label, then each span with its mask.

    Args:
        number: The row's number, from 1.
        row: The row.
        turns: The document's turns, by node id.
        painter: Colours, or none.
    """
    verdict = "trainable" if row["trainable"] else "not trainable"
    trained = sum(row["loss_mask"])
    print(
        f"{INDENT}row {number} · {row['role']} · {verdict} · "
        f"{len(row['input_ids']):,} tokens, {trained:,} with mask 1"
    )
    for span in spans(row, turns):
        if span.kind == CONTEXT:
            text = f"{span.turn:<4} context {span.size:>6,}  {span.mask_label()}"
            if span.size < 0:  # the turns' sizes do not add up to this row
                text += f"  {NEGATIVE_NOTE}"
            print(f"{INDENT * 2}{painter.paint(text, 'context')}")
            continue
        style = "trained" if span.trained else "untrained"
        text = f"{span.turn:<4} reply   {span.size:>6,}  {span.mask_label()}"
        print(f"{INDENT * 2}{painter.paint(text, style)}  {logprob_line(row, span)}")
    for finding in row["validation"]:
        if not finding.startswith("[INFO]"):
            style = finding_style(finding)
            line = short_finding(finding)
            print(f"{INDENT * 2}{painter.paint(line, style) if style else line}")


def print_rows(document: dict[str, Any], painter: Painter) -> None:
    """Print every row, the discarded ones last and marked.

    Args:
        document: A rollout document (every row, to see discarded ones too).
        painter: Colours, or none.
    """
    turns = {turn["node_id"]: turn for turn in document["turns"]}
    rows = document["sequences"]
    kept = [row for row in rows if row["role"] != DISCARDED]
    gone = len(rows) - len(kept)
    extra = ""
    if gone:
        extra = f" (+{gone} discarded, asked for with include_discarded=true)"
    print(f"rows      {len(kept)} for the trainer{extra}")
    for number, row in enumerate(rows, start=1):
        print_row(number, row, turns, painter)


def print_findings(document: dict[str, Any], painter: Painter) -> None:
    """Print the proxy's findings on the rollout, one line each.

    Args:
        document: A rollout document.
        painter: Colours, or none.
    """
    findings = document["validation"]
    if not findings:
        print("findings  none")
        return
    for index, finding in enumerate(findings):
        label = "findings  " if index == 0 else INDENT * 5
        style = finding_style(finding)
        line = short_finding(finding)
        print(f"{label}{painter.paint(line, style) if style else line}")


def print_stats(document: dict[str, Any]) -> None:
    """Print the trainer's numbers: roots, rows, trained tokens, the verdict.

    Args:
        document: The rollout document the trainer gets (the default one).
    """
    stats = document["stats"]
    print(
        f"stats     n_roots {stats['n_roots']} · n_sequences {stats['n_sequences']} · "
        f"n_trainable_tokens {stats['n_trainable_tokens']:,} · "
        f"trainable {str(document['trainable']).lower()}"
    )
