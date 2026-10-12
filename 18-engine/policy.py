"""The seam between the engine and the model: tokens in, tokens out.

Chapter 17's seam was text: render a chat, write a reply. An engine that
keeps the receipts needs the tokens themselves, so this seam is cut lower.
The engine (``engine.py``) decides what the model reads, remembers what it
wrote and keeps the prompt caches; the model only:

- renders a chat with its chat template, and turns text into token ids and
  back, special tokens kept;
- makes an empty prompt cache, and says how many tokens a cache holds;
- samples a reply after the tokens a cache already holds, and reports every
  sampled token's id and log-probability (and, when asked, the likeliest
  alternatives at each position).

``lfm.py`` is the real model, LFM2.5-2.6B in 4 bits on MLX. The tests plug in
a scripted model instead, so the request handling, the token fields, the
restore, the caches and the queue are all checked without loading weights.
This module needs only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

FINISH_STOP = "stop"  # the model wrote its end-of-turn token (or a stop string)
FINISH_LENGTH = "length"  # the length cap cut the reply off
GREEDY = 0.0  # temperature 0: always the likeliest token

Message = dict[str, Any]  # {"role": ..., "content": ...}, as OpenAI's API writes it
Tool = dict[str, Any]  # a tool's JSON schema, as OpenAI's API writes it


class RenderError(Exception):
    """Raised when the chat template refuses a conversation."""


@dataclass(frozen=True)
class Sampling:
    """How one reply is drawn.

    Attributes:
        max_tokens: The most tokens the reply may have.
        temperature: The scores are divided by it before the softmax; 0
            takes the likeliest token every time.
        top_logprobs: How many of the likeliest alternatives to report at
            each position (0 for none).
        stop: Texts that end the reply when the model writes one.
    """

    max_tokens: int
    temperature: float
    top_logprobs: int = 0
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sampled:
    """One reply, as token ids, with the numbers training needs.

    Attributes:
        token_ids: The tokens drawn, the end-of-turn token included when the
            model wrote one.
        logprobs: Each token's log-probability under the distribution it was
            drawn from (``log_softmax(scores / temperature)``).
        top_logprobs: Per position, the likeliest ``(token id,
            log-probability)`` pairs, best first; empty when none were asked.
        finish_reason: ``"stop"`` or ``"length"``.
        first_token_seconds: How long reading the new prompt tokens and
            drawing the first token took.
    """

    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[list[tuple[int, float]]]
    finish_reason: str
    first_token_seconds: float


class Model(Protocol):
    """What the engine needs from a model.

    Attributes:
        eos_ids: The token ids that end a turn.
    """

    eos_ids: frozenset[int]

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Turn a conversation into the text the model reads.

        Args:
            messages: The conversation, as OpenAI's API writes it.
            tools: The tools offered, or None.

        Returns:
            The prompt, ending where the model's reply begins.

        Raises:
            RenderError: If the chat template refuses the conversation.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Cut text into token ids, adding no tokens of its own.

        Args:
            text: Any text; special tokens written in it become one id each.

        Returns:
            The ids.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Write token ids back as text, special tokens kept.

        Args:
            token_ids: The ids.

        Returns:
            The text.
        """
        ...

    def new_cache(self) -> Any:
        """Make an empty prompt cache.

        Returns:
            A cache that holds no tokens yet.
        """
        ...

    def cache_size(self, cache: Any) -> int:
        """Say how many tokens a cache holds.

        Args:
            cache: A cache from ``new_cache``, used by ``sample`` or not.

        Returns:
            The number of tokens whose work the cache keeps.
        """
        ...

    def sample(self, new_ids: list[int], cache: Any, sampling: Sampling) -> Sampled:
        """Read the new prompt tokens after what the cache holds, then draw a reply.

        Args:
            new_ids: The prompt tokens the cache does not hold yet (at least
                one).
            cache: The cache; on return it also holds the new tokens and the
                reply's.
            sampling: How to draw.

        Returns:
            The reply.
        """
        ...
