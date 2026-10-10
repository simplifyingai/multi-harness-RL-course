# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2", "jinja2==3.1.6"]
# ///
"""Every call sends the whole conversation again, and it must fit the window.

A model remembers nothing between calls. So for each call, the harness sends
the whole conversation so far: the task, every earlier reply and every tool
output, and the template renders all of it again. This replays three calls of
a tiny agent on the shop database, counts the tokens each call reads and
writes, checks that each prompt starts with every token of the call before,
and measures the result against the context window: the model's 131,072
positions and the project engine's 16,384-token cap. Then it shows the command
that blew the window in the project.

Run it with ``uv run count.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import report
from chats import (
    AGENT_CALLS,
    BASH_TOOL,
    END_OF_TURN,
    THINK_OPEN,
    Message,
    agent_chat,
    assistant,
    tool_result,
)
from lfm import LfmTokenizer, load_tokenizer

SCRIPT = "count.py"
TITLE = "every call sends the whole conversation again"
HERE = Path(__file__).resolve().parent
# Chapter 1's shop database, copied. The agent sees it as data.db.
DATABASE = HERE / "data" / "shop.db"

# How far the model can read: max_position_embeddings in its config.json
# (chapter 10 reads it). Rotary positions make longer prompts possible in
# principle; this is what LFM2.5-2.6B was built and trained for.
MODEL_POSITIONS = 131_072
# The project's engine refuses longer prompts (mhrl/config.py,
# Settings.max_prompt_tokens, checked in mhrl/engine/server.py; chapter 18).
ENGINE_CAP = 16_384
TOOLS = [BASH_TOOL]

# What the untrained model kept doing in Pi: read the database file itself
# (Pi has a read tool; with a bash tool it is cat).
READ_DATABASE = (
    "<think>Let me look at the database file.</think>"
    "<|tool_call_start|>[bash(command='cat data.db')]<|tool_call_end|>"
)


@dataclass(frozen=True)
class Call:
    """One call of the agent: what it read and what it wrote.

    Attributes:
        number: The call's number, from 1.
        prompt: The prompt's token ids.
        reply: The reply's token ids, end of turn included.
        output: The tool's output, as text.
        prefix_kept: Whether the next call's prompt starts with this prompt
            and this reply, token for token (None for the last call).
    """

    number: int
    prompt: list[int]
    reply: list[int]
    output: str
    prefix_kept: bool | None


@dataclass(frozen=True)
class Overflow:
    """A prompt that holds the whole database file as a tool's output.

    Attributes:
        file_bytes: The file's size.
        file_tokens: Its bytes read as text, in tokens.
        prompt_tokens: The next call's prompt, in tokens.
    """

    file_bytes: int
    file_tokens: int
    prompt_tokens: int


def prompt_text(
    tokenizer: LfmTokenizer, messages: list[Message], preserve_thinking: bool
) -> str:
    """Render the agent's conversation as the prompt for its next call.

    Args:
        tokenizer: The model's tokenizer and template.
        messages: The conversation so far.
        preserve_thinking: Keep the reasoning of earlier replies.

    Returns:
        The prompt, ending with the generation prompt.
    """
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        preserve_thinking=preserve_thinking,
    )


def replay(tokenizer: LfmTokenizer, preserve_thinking: bool = True) -> list[Call]:
    """Render the prompt of each call, as the harness would send it.

    Args:
        tokenizer: The model's tokenizer and template.
        preserve_thinking: Keep the reasoning of earlier replies (the project's
            setting); False gives the template's default.

    Returns:
        One record per call.
    """
    prompts = [
        tokenizer.encode(prompt_text(tokenizer, agent_chat(done), preserve_thinking))
        for done in range(len(AGENT_CALLS))
    ]
    calls = []
    for index, (reply_text, output) in enumerate(AGENT_CALLS):
        # The prompt ends with <think>, so the model's reply is what follows
        # it, up to and including its end-of-turn token.
        written = reply_text.removeprefix(THINK_OPEN) + END_OF_TURN
        reply = tokenizer.encode(written)
        prefix_kept = None
        if index + 1 < len(prompts):
            so_far = prompts[index] + reply
            prefix_kept = prompts[index + 1][: len(so_far)] == so_far
        calls.append(Call(index + 1, prompts[index], reply, output, prefix_kept))
    return calls


def last_prompt_text(tokenizer: LfmTokenizer) -> str:
    """Render the last call's prompt as text.

    Args:
        tokenizer: The model's tokenizer and template.

    Returns:
        The prompt the third call reads.
    """
    return prompt_text(tokenizer, agent_chat(len(AGENT_CALLS) - 1), True)


def overflow(tokenizer: LfmTokenizer, database: Path) -> Overflow:
    """Measure a prompt whose first tool output is the database file itself.

    The file is binary; read as text, every byte that is not valid text
    becomes the replacement character.

    Args:
        tokenizer: The model's tokenizer and template.
        database: The SQLite file.

    Returns:
        The file's size and the prompt's length in tokens.

    Raises:
        FileNotFoundError: If the bundled data/shop.db is missing.
    """
    raw = database.read_bytes()
    as_text = raw.decode("utf-8", errors="replace")
    messages = agent_chat(0) + [assistant(READ_DATABASE), tool_result(as_text)]
    prompt = tokenizer.render(messages, TOOLS)
    return Overflow(len(raw), len(tokenizer.encode(as_text)), len(prompt))


def main() -> None:
    """Replay three calls, then measure them against the context window."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    pause_s = report.NO_PAUSE_S if args.fast else report.PAUSE_S
    tokenizer = load_tokenizer()

    report.print_title(SCRIPT, TITLE)
    calls = replay(tokenizer)
    default = replay(tokenizer, preserve_thinking=False)
    same_without = [call.prompt for call in calls] == [c.prompt for c in default]
    report.print_calls(calls, tokenizer.decode, same_without)
    report.pause(pause_s)
    report.print_last_prompt(len(calls), last_prompt_text(tokenizer))
    report.pause(pause_s)
    last = calls[-1]
    report.print_window(last.number, len(last.prompt), MODEL_POSITIONS, ENGINE_CAP)
    report.pause(pause_s)
    report.print_overflow(overflow(tokenizer, DATABASE), MODEL_POSITIONS, ENGINE_CAP)


if __name__ == "__main__":
    main()
