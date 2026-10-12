"""The seam between the harness and the model: chat in, reply text out.

The harness never touches weights. It needs two things from a policy (the
model, in the course's words): turn the conversation into the exact text the
model reads (its chat template does that, printing the tools into the system
prompt), and write a reply to that text. ``lfm.py`` is the real policy,
LFM2.5-2.6B in 4 bits; the tests give the harness a scripted one, reply
texts written in advance with the special tokens in them, so the loop, the
parser, the tool and the stop rules are checked without loading a model.
This module needs only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

FINISH_STOP = "stop"  # the model wrote its end-of-turn token
FINISH_LENGTH = "length"  # the length cap cut the reply off

Message = dict[str, Any]  # {"role": ..., "content": ...}
Tool = dict[str, Any]  # a tool's JSON schema, as OpenAI's API writes it


@dataclass(frozen=True)
class Reply:
    """One reply, exactly as the model wrote it.

    Attributes:
        text: The reply's text with its special tokens kept, the end-of-turn
            token included when the model wrote one. The prompt ended with
            ``<think>``, so the reply starts inside the model's reasoning.
        prompt_tokens: How many tokens the model read first.
        reply_tokens: How many tokens it wrote.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        seconds: How long reading the prompt and writing the reply took.
    """

    text: str
    prompt_tokens: int
    reply_tokens: int
    finish_reason: str
    seconds: float


class Policy(Protocol):
    """What the harness needs from a model."""

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Turn a conversation into the text the model reads.

        Args:
            messages: The conversation so far.
            tools: The tools offered, or None for none.

        Returns:
            The prompt, ending where the model's reply begins.
        """
        ...

    def complete(self, prompt: str) -> Reply:
        """Write a reply to a prompt.

        Args:
            prompt: What ``render`` returned.

        Returns:
            The reply.
        """
        ...
