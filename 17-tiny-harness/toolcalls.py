"""Read LFM2.5's tool calls out of a reply's text.

LFM2.5 writes a tool call as a Python-style list between two special tokens::

    <|tool_call_start|>[bash(command='sqlite3 data.db .tables')]<|tool_call_end|>

A copy of the project's ``mhrl/engine/toolcalls.py`` (``split_reply`` and
``parse_calls``), kept to what an in-process harness needs:

- everything before ``</think>`` is **reasoning**; a reply that never closes
  its thinking is all reasoning, so a call written inside it is not a call;
- each tool-call block is parsed with Python's ``ast`` module, which reads
  the text without running it. Positional arguments are named from the
  tool's JSON schema (``bash('ls')`` becomes ``command='ls'``), an argument
  that is not a plain value is kept as its source text, and a block that
  does not parse is dropped, as the project's engine drops it;
- what is left after the reasoning, without the blocks, is the **visible**
  text.

The project's engine also gives every call an id, because a harness talking
over HTTP must match each result to its call; in one process the order is
enough. Chapter 17 adds one count the engine did not need: how many blocks
were opened, so the harness can tell an unreadable call from no call at all.
This module needs only the standard library.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_OPEN = "<think>"  # the chat template's generation prompt ends with it
THINK_CLOSE = "</think>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
END_OF_TURN = "<|im_end|>"
TOOL_BLOCK = re.compile(
    re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL
)
POSITIONAL_KEY = "arg{index}"  # a positional argument the schema has no name for
# What ast.parse raises on text it cannot read: bad syntax, a NUL character
# (ValueError before Python 3.12) and nesting too deep for the parser.
UNPARSEABLE = (SyntaxError, ValueError, RecursionError, MemoryError)


@dataclass(frozen=True)
class ToolCall:
    """One tool call, as the harness runs it.

    Attributes:
        name: The tool's name, exactly as the model wrote it.
        arguments: Its arguments by name.
    """

    name: str
    arguments: dict[str, Any]

    def as_json(self) -> str:
        """Write the call as an OpenAI-style harness receives it.

        Returns:
            Such as ``bash {"command": "ls"}``.
        """
        return f"{self.name} {json.dumps(self.arguments, default=repr)}"


@dataclass(frozen=True)
class ParsedReply:
    """A reply split into what a harness needs.

    Attributes:
        reasoning: What the model wrote before ``</think>``.
        content: The visible text, without reasoning or tool-call blocks.
        calls: The tool calls that parsed, in order.
        opened: How many tool-call blocks the visible part opened, finished
            or not (a reply cut off by the length cap has no end marker).
    """

    reasoning: str = ""
    content: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    opened: int = 0


def literal(node: ast.AST) -> Any:
    """Evaluate one argument node, falling back to its source text.

    Args:
        node: An AST expression node.

    Returns:
        The Python value, or the unparsed source when it is not a literal.
    """
    try:
        return ast.literal_eval(node)  # values only: it never runs code
    except UNPARSEABLE + (TypeError,):
        return ast.unparse(node)


def call_name(func: ast.AST) -> str:
    """Return a dotted function name from a call's ``func`` node.

    Args:
        func: The ``func`` attribute of an ``ast.Call``.

    Returns:
        The name, such as ``bash`` or ``tools.bash``.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return f"{call_name(func.value)}.{func.attr}"
    return ast.unparse(func)


def positional_names(name: str, tools: list[dict[str, Any]] | None) -> list[str]:
    """Return a tool's parameter names in schema order, for positional args.

    Args:
        name: Tool name.
        tools: The tool schemas offered to the model.

    Returns:
        Parameter names, possibly empty (an unknown tool has none).
    """
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name") == name:
            properties = (function.get("parameters") or {}).get("properties") or {}
            return list(properties)
    return []


def parse_calls(source: str, tools: list[dict[str, Any]] | None) -> list[ToolCall]:
    """Parse the inside of one tool-call block into tool calls.

    Args:
        source: Text between the markers, e.g. ``[bash(command='ls')]``.
        tools: The tool schemas offered, used to name positional arguments.

    Returns:
        The calls; empty when the text does not parse. An item of the list
        that is not a call (``[42]``) is skipped.
    """
    try:
        tree = ast.parse(source.strip(), mode="eval").body
    except UNPARSEABLE:
        return []
    nodes = tree.elts if isinstance(tree, ast.List | ast.Tuple) else [tree]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node.func)
        arguments = {kw.arg: literal(kw.value) for kw in node.keywords if kw.arg}
        names = positional_names(name, tools)
        for index, arg in enumerate(node.args):
            named = index < len(names)
            key = names[index] if named else POSITIONAL_KEY.format(index=index)
            arguments.setdefault(key, literal(arg))
        calls.append(ToolCall(name, arguments))
    return calls


def split_reply(text: str, tools: list[dict[str, Any]] | None) -> ParsedReply:
    """Split a raw reply (the prompt ended in ``<think>``) into its parts.

    Args:
        text: The reply as the model wrote it, special tokens kept.
        tools: The tool schemas offered.

    Returns:
        The reasoning, the visible text, the calls and the blocks opened.
    """
    text = text.removesuffix(END_OF_TURN)  # a reply ends at its end of turn
    reasoning, closed, answer = text.partition(THINK_CLOSE)
    if not closed:
        # The model never closed its thinking: everything is reasoning.
        reasoning, answer = text, ""
    calls = []
    for match in TOOL_BLOCK.finditer(answer):
        calls.extend(parse_calls(match.group(1), tools))
    content = TOOL_BLOCK.sub("", answer).strip()
    return ParsedReply(
        reasoning=reasoning.strip(),
        content=content,
        calls=calls,
        opened=answer.count(TOOL_CALL_START),
    )
