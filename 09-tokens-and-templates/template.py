# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2", "jinja2==3.1.6"]
# ///
"""Chats become one token string, through the model's own chat template.

A chat is a list of messages, each with a role: system, user, assistant or
tool. The model reads one string of tokens, so a chat template (a Jinja file
that ships with the model) writes the messages one after another between
special tokens, adds the tool list, and ends with the cue for the model's
reply. This renders three chats the way the project's engine does
(``apply_chat_template`` with ``add_generation_prompt`` and
``preserve_thinking``): one user message, the same message with a bash tool,
and a chat with three user turns, where the template's default drops the
reasoning of earlier replies.

Run it with ``uv run template.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import report
from chats import (
    BASH_TOOL,
    THINK_OPEN,
    Message,
    Tool,
    one_question,
    three_user_turns,
)
from lfm import LfmTokenizer, load_tokenizer

SCRIPT = "template.py"
TITLE = "chats become one string, through LFM2.5's chat_template.jinja"
GENERATION_PROMPT = f"<|im_start|>assistant\n{THINK_OPEN}"


@dataclass(frozen=True)
class Rendered:
    """One chat rendered into a prompt.

    Attributes:
        text: The prompt as one string.
        ids: Its token ids.
    """

    text: str
    ids: list[int]


@dataclass(frozen=True)
class ThinkingComparison:
    """The same chat rendered with and without preserve_thinking.

    Attributes:
        dropped: Rendered with the template's default (reasoning dropped).
        kept: Rendered with preserve_thinking=True, as the project does.
        shared: How many ids the two have in common from the start.
    """

    dropped: Rendered
    kept: Rendered
    shared: int


def render_prompt(
    tokenizer: LfmTokenizer,
    messages: list[Message],
    tools: list[Tool] | None = None,
    preserve_thinking: bool = True,
) -> Rendered:
    """Render a chat with the generation prompt, keeping the text as well.

    With ``preserve_thinking=True`` the ids are what ``tokenizer.render`` (the
    project's own call) gives; False shows the template's default.

    Args:
        tokenizer: The model's tokenizer and template.
        messages: The chat.
        tools: Tool definitions, or None.
        preserve_thinking: Keep the reasoning of every earlier reply.

    Returns:
        The prompt and its ids.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        preserve_thinking=preserve_thinking,
    )
    return Rendered(text, tokenizer.encode(text))


def common_prefix(first: list[int], second: list[int]) -> int:
    """Count how many ids two sequences share from the start.

    Args:
        first: One sequence.
        second: Another.

    Returns:
        The length of their common start.
    """
    shared = 0
    for a, b in zip(first, second, strict=False):
        if a != b:
            break
        shared += 1
    return shared


def compare_thinking(
    tokenizer: LfmTokenizer, messages: list[Message]
) -> ThinkingComparison:
    """Render a chat with and without preserve_thinking.

    Args:
        tokenizer: The model's tokenizer and template.
        messages: The chat.

    Returns:
        Both renders and where they part.
    """
    dropped = render_prompt(tokenizer, messages, preserve_thinking=False)
    kept = render_prompt(tokenizer, messages, preserve_thinking=True)
    return ThinkingComparison(dropped, kept, common_prefix(dropped.ids, kept.ids))


def main() -> None:
    """Render the three chats and say what the template did to each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    pause_s = report.NO_PAUSE_S if args.fast else report.PAUSE_S
    tokenizer = load_tokenizer()
    cue = tokenizer.encode(GENERATION_PROMPT)
    think_id = tokenizer.require_id(THINK_OPEN)

    report.print_title(SCRIPT, TITLE)
    question = one_question()
    plain = render_prompt(tokenizer, question)
    report.print_plain(question, plain.text, plain.ids, tokenizer.pieces(cue))
    report.pause(pause_s)
    with_tool = render_prompt(tokenizer, question, [BASH_TOOL])
    report.print_with_tool(with_tool.text, len(with_tool.ids), len(plain.ids), think_id)
    report.pause(pause_s)
    chat = three_user_turns()
    result = compare_thinking(tokenizer, chat)
    report.print_drop(
        chat,
        (result.dropped.text, len(result.dropped.ids)),
        (result.kept.text, len(result.kept.ids)),
        result.shared,
    )


if __name__ == "__main__":
    main()
