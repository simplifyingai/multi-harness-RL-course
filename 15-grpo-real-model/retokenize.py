"""What training on re-tokenized text does: decode the reply, encode it again.

A pipeline that keeps only text has to turn the reply back into tokens before
it can train. Chapter 9 showed why that is a guess: ``answer.txt`` can be
spelled in 168 token sequences, and encoding the text gives back one of them,
not necessarily the one the model drew.

``align_by_position`` lines the re-encoded reply up with the sampled one,
position by position. Where the two ids agree, the sampled token's recorded
log-probability is kept as its old log-probability; where they differ (or
one reply is longer), the position is masked out of training and counted.
Positions after a difference can agree again by coincidence: they keep an
old log-probability that was recorded for another context, which is what
moves the importance ratio.

This module needs only the standard library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rows import READ, SAMPLED

MASKED_LOGPROB = 0.0  # what a masked position records: it is never trained


@dataclass(frozen=True)
class Aligned:
    """A re-encoded reply lined up with the sampled one.

    Attributes:
        ids: The re-encoded reply's token ids.
        logprobs: Per position, the sampled token's old log-probability where
            the ids agree, else 0.
        mask: Per position, 1 where the ids agree (trained), else 0.
        first_difference: The first position where they differ, or None when
            the two replies are identical.
    """

    ids: list[int]
    logprobs: list[float]
    mask: list[int]
    first_difference: int | None

    @property
    def differs(self) -> bool:
        """Whether re-encoding changed the reply at all.

        Returns:
            True when the ids are not the sampled ones.
        """
        return self.first_difference is not None

    @property
    def masked(self) -> int:
        """How many re-encoded positions were masked out.

        Returns:
            The count of mask-0 positions.
        """
        return self.mask.count(READ)

    @property
    def after_difference(self) -> int:
        """How many kept positions come after the first difference.

        Their old log-probabilities were recorded after other tokens: these
        are the ones that move the ratio.

        Returns:
            The count of mask-1 positions past ``first_difference``.
        """
        if self.first_difference is None:
            return 0
        return sum(self.mask[self.first_difference :])


def align_by_position(
    sampled: list[int], logprobs: list[float], reencoded: list[int]
) -> Aligned:
    """Line a re-encoded reply up with the sampled one, position by position.

    Args:
        sampled: The token ids the model drew.
        logprobs: Their recorded log-probabilities.
        reencoded: The ids that encoding the decoded reply gave.

    Returns:
        The re-encoded reply, with the old log-probabilities it can borrow
        and the positions it must mask.

    Raises:
        ValueError: If ``sampled`` and ``logprobs`` differ in length.
    """
    if len(sampled) != len(logprobs):
        raise ValueError("one log-probability per sampled token is needed")
    kept, mask, first = [], [], None
    for position, token in enumerate(reencoded):
        if position < len(sampled) and sampled[position] == token:
            kept.append(logprobs[position])
            mask.append(SAMPLED)
        else:
            kept.append(MASKED_LOGPROB)
            mask.append(READ)
            first = position if first is None else first
    if first is None and len(reencoded) != len(sampled):
        first = len(reencoded)  # the re-encoded reply stopped early
    return Aligned(list(reencoded), kept, mask, first)


def reencode(
    sampled: list[int],
    decode: Callable[[list[int]], str],
    encode: Callable[[str], list[int]],
) -> list[int]:
    """Turn a sampled reply into text and back into tokens.

    Args:
        sampled: The token ids the model drew.
        decode: Turns ids into text (special tokens kept as text).
        encode: Turns text into ids, adding nothing.

    Returns:
        The ids encoding gives for the reply's text.
    """
    return encode(decode(sampled))
