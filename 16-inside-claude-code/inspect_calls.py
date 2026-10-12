# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Show what Claude Code sent the model, call by call, with token counts.

Reads a run's ``calls.jsonl`` (every request the stand-in received) and
prints, for each call: the address and the model asked for, the system
prompt (size, and whether it changed), the tools offered, the conversation
so far as a list of messages, the size of each part in tokens, and what the
stand-in answered. A call that offers no tools, or names another model than
the loop's, is flagged as a side call: a request the harness makes for
itself, outside the agent's loop.

Tokens are counted with LFM2.5-2.6B's tokenizer, part by part (the system
text, the tools as JSON, the text of each message). A server would first
render the request with the model's chat template (chapter 9), which adds a
few marker tokens per message, so these are close to, not exactly, what the
model reads.

    uv run inspect_calls.py                         the run called "solve"
    uv run inspect_calls.py mishap-lowercase-bash   any other run
    uv run inspect_calls.py solve --system          also print the prompt in full
    uv run inspect_calls.py --compare yours ours-extras
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lfm_tokens import TokenizerError, find_tokenizer, load_counter
from records import (
    CALLS_FILE,
    RunNotFoundError,
    block_text,
    blocks,
    first_loop_call,
    message_shape,
    model_calls,
    read_jsonl,
    reply_summary,
    run_folder,
    short,
    system_text,
)

DEFAULT_RUN = "solve"
SKILLS_HEADER = "The following skills are available for use with the Skill tool:"
CLAUDE_MD_MARKER = "CLAUDE.md (project instructions"
MCP_PREFIX = "mcp__"
MAIN_TOOL = "Bash"
YES, NO, NONE = "yes", "no", "none"
LABEL_WIDTH = 22  # columns of --compare's table
VALUE_WIDTH = 16

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class CallSize:
    """How big one request is, part by part.

    Attributes:
        system_chars: Characters in the system prompt.
        system_tokens: Its tokens.
        tools_tokens: Tokens of the tools, as the JSON that was sent.
        messages_tokens: Tokens of the text every message carries.
    """

    system_chars: int
    system_tokens: int
    tools_tokens: int
    messages_tokens: int

    @property
    def total(self) -> int:
        """All three parts together."""
        return self.system_tokens + self.tools_tokens + self.messages_tokens


def system_chars(body: dict[str, Any]) -> int:
    """Count the characters of a request's system prompt.

    Args:
        body: The request body.

    Returns:
        The characters of its blocks, added up.
    """
    system = body.get("system") or []
    if isinstance(system, str):
        return len(system)
    return sum(len(block.get("text", "")) for block in system)


def measure(body: dict[str, Any], count: TokenCounter) -> CallSize:
    """Measure one request.

    The tools are counted as the JSON that was sent, with every character
    written as itself (``ensure_ascii=False``), not as ``\\u`` escapes.

    Args:
        body: The request body.
        count: A function from text to its number of tokens.

    Returns:
        The sizes.
    """
    tools = body.get("tools") or []
    texts = [
        block_text(block)
        for message in body.get("messages", [])
        for block in blocks(message["content"])
    ]
    return CallSize(
        system_chars=system_chars(body),
        system_tokens=count(system_text(body)),
        tools_tokens=count(json.dumps(tools, ensure_ascii=False)) if tools else 0,
        messages_tokens=sum(count(text) for text in texts),
    )


def tool_names(body: dict[str, Any]) -> list[str]:
    """List the names of the tools a request offers.

    Args:
        body: The request body.

    Returns:
        The names, in the order sent.
    """
    return [tool["name"] for tool in body.get("tools") or []]


def loop_model(calls: list[dict[str, Any]]) -> str:
    """Find the model the agent's loop asks for.

    Args:
        calls: A run's model calls.

    Returns:
        The model named by the first call that offers tools ("" if none does).
    """
    for call in calls:
        if call["body"].get("tools"):
            return call["body"].get("model", "")
    return ""


def side_call_reason(body: dict[str, Any], main_model: str) -> str:
    """Say why a request is a side call, if it is one.

    Args:
        body: The request body.
        main_model: The model the agent's loop asks for.

    Returns:
        "offers no tools", "names <model>", or "" for a call of the loop.
    """
    if not body.get("tools"):
        return "offers no tools"
    if body.get("model") != main_model:
        return f"names {body.get('model')}"
    return ""


def schema_line(tool: dict[str, Any]) -> str:
    """Summarise a tool's input schema on one line.

    Args:
        tool: One entry of a request's ``tools``.

    Returns:
        Each parameter with its type, ``required`` where it is.
    """
    schema = tool.get("input_schema", {})
    required = set(schema.get("required", []))
    parts = []
    for name, spec in schema.get("properties", {}).items():
        flag = ", required" if name in required else ""
        parts.append(f"{name} ({spec.get('type', '?')}{flag})")
    return " · ".join(parts)


def skills_listed(body: dict[str, Any]) -> list[str]:
    """Find the skills a request lists for the model.

    Claude Code lists each skill's name and description under one header,
    in a message of the conversation.

    Args:
        body: The request body.

    Returns:
        The skill names.
    """
    names = []
    for message in body.get("messages", []):
        for block in blocks(message["content"]):
            text = block_text(block)
            if SKILLS_HEADER not in text:
                continue
            listing = text.split(SKILLS_HEADER, 1)[1].strip().split("\n\n")[0]
            for line in listing.splitlines():
                if line.startswith("- "):
                    names.append(line[2:].split(":", 1)[0])
    return names


def sends_claude_md(body: dict[str, Any]) -> bool:
    """Tell whether a request carries the contents of a CLAUDE.md.

    Args:
        body: The request body.

    Returns:
        True when a message quotes a CLAUDE.md file.
    """
    return any(
        CLAUDE_MD_MARKER in block_text(block)
        for message in body.get("messages", [])
        for block in blocks(message["content"])
    )


def print_call(
    index: int,
    call: dict[str, Any],
    size: CallSize,
    previous: tuple[dict[str, Any], CallSize] | None,
    main_model: str,
) -> None:
    """Print one call.

    Args:
        index: Its number in the run, from 1.
        call: The logged call.
        size: Its measured size.
        previous: The call before it and its size, or None for the first.
        main_model: The model the agent's loop asks for.
    """
    body = call["body"]
    side = side_call_reason(body, main_model)
    role = f"SIDE CALL: {side}" if side else "the agent's loop"
    stream = YES if body.get("stream") else NO
    print(
        f"call {index}  {call['method']} {call['path']} · model {body.get('model')} "
        f"· stream {stream} · {role}"
    )
    if previous and system_text(previous[0]["body"]) == system_text(body):
        print(f"  system    the same {size.system_chars:,} characters as before")
    else:
        system = body.get("system") or []
        parts = len(system) if isinstance(system, list) else 1
        print(
            f"  system    {parts} blocks · {size.system_chars:,} characters "
            f"· {size.system_tokens:,} tokens"
        )
    names = tool_names(body)
    if previous and tool_names(previous[0]["body"]) == names:
        print("  tools     the same")
    else:
        print(f"  tools     [{', '.join(names)}] · {size.tools_tokens:,} tokens")
    grown = ""
    if previous:
        grown = f" (+{size.messages_tokens - previous[1].messages_tokens:,})"
    shape = message_shape(body.get("messages", []))
    print(f"  messages  {shape} · {size.messages_tokens:,} tokens{grown}")
    print(f"  total     {size.total:,} tokens")
    print(f"  reply     {short(reply_summary(call))}")


def print_every_call(first: dict[str, Any]) -> None:
    """Print what every call carries besides the conversation.

    Args:
        first: The run's first call of the loop.
    """
    headers, body = first["headers"], first["body"]
    betas = [b for b in headers.get("anthropic-beta", "").split(",") if b]
    thinking = body.get("thinking", {}).get("type", NONE)
    effort = body.get("output_config", {}).get("effort", NONE)
    print(
        f"headers   x-api-key {headers.get('x-api-key')} · anthropic-version "
        f"{headers.get('anthropic-version')} · user-agent {headers.get('user-agent')}"
    )
    print(f"          anthropic-beta: {len(betas)} flags")
    print(
        f"body      max_tokens {body.get('max_tokens')} · thinking {thinking} "
        f"· effort {effort} · metadata {', '.join(body.get('metadata', {}))}"
    )


def report_run(name: str, count: TokenCounter, show_system: bool) -> None:
    """Print every call of one run.

    Args:
        name: The run's name or folder.
        count: A function from text to its number of tokens.
        show_system: Also print the system prompt and the Bash tool in full.

    Raises:
        RunNotFoundError: If the run does not exist.
    """
    folder = run_folder(name)
    calls = model_calls(read_jsonl(folder / CALLS_FILE))
    if not calls:
        print(f"run {name}: no model calls in {CALLS_FILE}")
        return
    print(f"run {name} · {len(calls)} model calls · {short(str(folder / CALLS_FILE))}")
    main_model = loop_model(calls)
    previous = None
    for index, call in enumerate(calls, start=1):
        size = measure(call["body"], count)
        print_call(index, call, size, previous, main_model)
        previous = (call, size)
    loop = [c for c in calls if not side_call_reason(c["body"], main_model)]
    prompts = {system_text(c["body"]) for c in loop}
    same = YES if len(prompts) == 1 else f"no ({len(prompts)} versions)"
    print(f"side calls          {len(calls) - len(loop)} of {len(calls)}")
    print(f"same system prompt  {same}, across the {len(loop)} calls of the loop")
    first = first_loop_call(calls)
    print_every_call(first)
    for tool in first["body"].get("tools") or []:
        if tool["name"] == MAIN_TOOL:
            chars = len(tool.get("description", ""))
            print(f"{MAIN_TOOL} tool  description {chars:,} characters")
            print(f"          input: {schema_line(tool)}")
    if show_system:
        print_system(first["body"])


def print_system(body: dict[str, Any]) -> None:
    """Print a request's system prompt, every message and the Bash tool.

    Args:
        body: The request body.
    """
    system = body.get("system") or []
    for index, block in enumerate(system if isinstance(system, list) else []):
        text = block.get("text", "")
        print(f"\n===== system block {index + 1} ({len(text):,} characters)")
        print(short(text))
    for number, message in enumerate(body.get("messages", []), start=1):
        for part, block in enumerate(blocks(message["content"]), start=1):
            text = block_text(block)
            where = f"message {number} (role {message['role']}), block {part}"
            print(f"\n===== {where}: {len(text):,} characters")
            print(short(text))
    for tool in body.get("tools") or []:
        if tool["name"] == MAIN_TOOL:
            print(f"\n===== the {MAIN_TOOL} tool, as sent")
            print(json.dumps(tool, indent=2, ensure_ascii=False))


def compare_runs(first: str, second: str, count: TokenCounter) -> None:
    """Compare the first call of the loop of two runs, side by side.

    Args:
        first: One run's name or folder.
        second: The other's.
        count: A function from text to its number of tokens.

    Raises:
        RunNotFoundError: If either run does not exist or has no call that
            offers tools.
    """
    folders = [run_folder(name) for name in (first, second)]
    runs = [model_calls(read_jsonl(folder / CALLS_FILE)) for folder in folders]
    bodies = [first_loop_call(calls)["body"] for calls in runs]
    sizes = [measure(body, count) for body in bodies]

    def row(label: str, values: list[str]) -> None:
        """Print one line of the table."""
        cells = "".join(f"{value:>{VALUE_WIDTH}}" for value in values)
        print(f"{label:<{LABEL_WIDTH}}{cells}")

    print("the first call of the loop in each run")
    row("", [first, second])
    row("model calls", [str(len(calls)) for calls in runs])
    row("tools offered", [str(len(tool_names(body))) for body in bodies])
    row("system characters", [f"{s.system_chars:,}" for s in sizes])
    row("system tokens", [f"{s.system_tokens:,}" for s in sizes])
    row("tools tokens", [f"{s.tools_tokens:,}" for s in sizes])
    row("messages tokens", [f"{s.messages_tokens:,}" for s in sizes])
    row("total tokens", [f"{s.total:,}" for s in sizes])
    row("CLAUDE.md sent", [YES if sends_claude_md(b) else NO for b in bodies])
    skills = [skills_listed(b) for b in bodies]
    row("skills listed", [str(len(names)) for names in skills])
    mcp = [[n for n in tool_names(b) if n.startswith(MCP_PREFIX)] for b in bodies]
    row("MCP tools", [str(len(names)) for names in mcp])
    only = [n for n in tool_names(bodies[0]) if n not in tool_names(bodies[1])]
    print(f"tools only in {first}: {', '.join(only) or NONE}")
    print(f"skills listed in {first}: {', '.join(skills[0]) or NONE}")
    if sizes[1].total:
        ratio = sizes[0].total / sizes[1].total
        print(
            f"the first call of {first} is {ratio:.1f} times the tokens of {second}'s"
        )


def main() -> None:
    """Report one run, or compare two.

    Raises:
        SystemExit: With a plain message when a run or the tokenizer is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--system", action="store_true", help="print the prompt")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args()
    try:
        path = find_tokenizer()
        count = load_counter(path)
        print(f"tokens    counted with LFM2.5-2.6B's tokenizer ({short(str(path))})")
        if args.compare:
            compare_runs(*args.compare, count)
        else:
            report_run(args.run, count, args.system)
    except (RunNotFoundError, TokenizerError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
