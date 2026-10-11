"""Small sums over a drawn reply: running totals, grids, and what a cut changed.

Everything here works on plain lists of numbers, after the model is done, so
it needs only the standard library (and the tests can check it anywhere).
sample_lab.py measures; this module adds up what it measured; report.py
prints it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate

FINEST_GRID_POWER = 24  # look for grids down to 1/2**24 (float32 has 24 bits)
MILLION = 1_000_000
BILLION = 1_000_000_000
NEARLY_CERTAIN_NATS = 0.1  # an entropy below this: one token has almost all


@dataclass(frozen=True)
class Ending:
    """How a reply ended, and its sums.

    Attributes:
        tokens: How many tokens were drawn, the end of turn included.
        total: The reply's log-probability: the sum of its tokens'.
        mean_entropy: The average entropy per step, in nats.
        nearly_certain: Steps whose entropy was below ``NEARLY_CERTAIN_NATS``.
    """

    tokens: int
    total: float
    mean_entropy: float
    nearly_certain: int


@dataclass(frozen=True)
class Cut:
    """What top-k or top-p changed, measured at the same steps both ways.

    Attributes:
        steps: How many steps there were.
        higher: Steps where the drawn token was likelier as drawn than uncut.
        equal: Steps where the two were the same number.
        lower: Steps where it was less likely as drawn: float32 rounding
            alone, since a cut can only raise the kept tokens' probabilities.
        largest_drop: The largest such shortfall, in log-probability.
        kept_mean: The average share of probability the kept tokens held.
        kept_lowest: The smallest such share.
        entropy_uncut: The average entropy before the cut.
        entropy_cut: The average entropy of the distribution drawn from.
        lower_entropy: Steps where the cut lowered the entropy.
        total_cut: The reply's log-probability as drawn.
        total_uncut: The same tokens' log-probability before the cut.
    """

    steps: int
    higher: int
    equal: int
    lower: int
    largest_drop: float
    kept_mean: float
    kept_lowest: float
    entropy_uncut: float
    entropy_cut: float
    lower_entropy: int
    total_cut: float
    total_uncut: float


@dataclass(frozen=True)
class RawSide:
    """The drawn tokens' log-probabilities, processed and raw.

    Attributes:
        steps: How many steps there were.
        differ: Steps where the two numbers differ.
        processed_higher: Steps where the processed one is higher.
        total_processed: The reply's log-probability as drawn.
        total_raw: The same tokens' log-probability under the raw scores.
    """

    steps: int
    differ: int
    processed_higher: int
    total_processed: float
    total_raw: float


@dataclass(frozen=True)
class Grid:
    """Where bfloat16 log-probabilities land, beside float32 ones.

    Attributes:
        steps: How many steps there were.
        power: Every bfloat16 value is a multiple of 1 / 2**power (None if
            even a grid as fine as 1 / 2**24 does not hold them all).
        float32_power: The same for the float32 values.
        zeros: Steps whose bfloat16 log-probability is exactly 0.
        zero_float32_low: The lowest float32 value at those steps.
        zero_float32_high: The highest float32 value at those steps.
        distinct: How many different bfloat16 values there are.
        distinct_float32: How many different float32 values there are.
        score_low: The lowest score among the drawn tokens.
        score_high: The highest score among the drawn tokens.
        total: The reply's log-probability in bfloat16.
        total_float32: The same in float32.
    """

    steps: int
    power: int | None
    float32_power: int | None
    zeros: int
    zero_float32_low: float
    zero_float32_high: float
    distinct: int
    distinct_float32: int
    score_low: float
    score_high: float
    total: float
    total_float32: float


def running_sums(values: list[float]) -> list[float]:
    """Add the values up one by one.

    Args:
        values: Log-probabilities, in order.

    Returns:
        After each value, the sum so far.
    """
    return list(accumulate(values))


def grid_power(values: list[float]) -> int | None:
    """Find the coarsest grid 1 / 2**k that holds every value exactly.

    Args:
        values: The numbers.

    Returns:
        The smallest k such that every value is a whole multiple of
        1 / 2**k, or None if even 1 / 2**24 is not fine enough.
    """
    for power in range(FINEST_GRID_POWER + 1):
        step = 2.0**-power
        if all((value / step).is_integer() for value in values):
            return power
    return None


def one_in(total: float) -> str:
    """Write the chance e**total as "1 in N", in words a person can read.

    Args:
        total: A log-probability (0 or below).

    Returns:
        Such as ``"1 in 1,612"``, ``"1 in 3.4 million"`` or ``"1 in 10^41.2"``.
    """
    power_form = f"1 in 10^{-total / math.log(10):.1f}"
    if -total >= math.log(BILLION):
        return power_form
    against = math.exp(-total)
    if round(against) < MILLION:
        return f"1 in {against:,.0f}"
    millions = round(against / MILLION, 1)  # rounded first, so never "1000.0"
    if millions < BILLION / MILLION:
        return f"1 in {millions:.1f} million"
    return power_form


def ending(logprobs: list[float], entropies: list[float]) -> Ending:
    """Sum a reply's log-probabilities and describe its entropy.

    Args:
        logprobs: Each drawn token's log-probability.
        entropies: Each step's entropy.

    Returns:
        The sums.
    """
    certain = sum(1 for value in entropies if value < NEARLY_CERTAIN_NATS)
    mean = sum(entropies) / len(entropies) if entropies else 0.0
    return Ending(len(logprobs), math.fsum(logprobs), mean, certain)


def require_steps(values: list[float]) -> None:
    """Stop if there is nothing to sum: every reply has at least one token.

    Args:
        values: One number per step.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("a reply has at least one step; got none")


def cut(
    drawn: list[float],
    uncut: list[float],
    entropy_cut: list[float],
    entropy_uncut: list[float],
) -> Cut:
    """Compare the drawn tokens' numbers with and without top-k or top-p.

    The kept tokens' share of the probability is exp(uncut - drawn): the cut
    divides every kept probability by that share.

    Args:
        drawn: Each drawn token's log-probability as drawn.
        uncut: The same before the cut.
        entropy_cut: Each step's entropy as drawn.
        entropy_uncut: The same before the cut.

    Returns:
        The comparison.

    Raises:
        ValueError: If there are no steps, or the lists differ in length.
    """
    require_steps(drawn)
    both = list(zip(drawn, uncut, strict=True))
    kept = [math.exp(before - after) for after, before in both]
    drops = [before - after for after, before in both if after < before]
    pairs = list(zip(entropy_cut, entropy_uncut, strict=True))
    return Cut(
        steps=len(drawn),
        higher=sum(1 for after, before in both if after > before),
        equal=sum(1 for after, before in both if after == before),
        lower=len(drops),
        largest_drop=max(drops, default=0.0),
        kept_mean=sum(kept) / len(kept),
        kept_lowest=min(kept),
        entropy_uncut=sum(entropy_uncut) / len(entropy_uncut),
        entropy_cut=sum(entropy_cut) / len(entropy_cut),
        lower_entropy=sum(1 for after, before in pairs if after < before),
        total_cut=math.fsum(drawn),
        total_uncut=math.fsum(uncut),
    )


def raw_side(processed: list[float], raw: list[float]) -> RawSide:
    """Compare the drawn tokens' processed and raw log-probabilities.

    Args:
        processed: Each drawn token's log-probability as drawn.
        raw: The same under the model's own distribution.

    Returns:
        The comparison.

    Raises:
        ValueError: If there are no steps, or the lists differ in length.
    """
    require_steps(processed)
    pairs = list(zip(processed, raw, strict=True))
    return RawSide(
        steps=len(pairs),
        differ=sum(1 for one, other in pairs if one != other),
        processed_higher=sum(1 for one, other in pairs if one > other),
        total_processed=math.fsum(processed),
        total_raw=math.fsum(raw),
    )


def grid(low_precision: list[float], float32: list[float], scores: list[float]) -> Grid:
    """Describe where bfloat16 log-probabilities land, beside float32 ones.

    Args:
        low_precision: Each drawn token's log-probability in bfloat16.
        float32: The same from the same scores in float32.
        scores: Each drawn token's score.

    Returns:
        The description.

    Raises:
        ValueError: If there are no steps, or the lists differ in length.
    """
    require_steps(low_precision)
    pairs = zip(low_precision, float32, strict=True)
    at_zero = [wide for low, wide in pairs if low == 0]
    return Grid(
        steps=len(low_precision),
        power=grid_power(low_precision),
        float32_power=grid_power(float32),
        zeros=len(at_zero),
        zero_float32_low=min(at_zero, default=0.0),
        zero_float32_high=max(at_zero, default=0.0),
        distinct=len(set(low_precision)),
        distinct_float32=len(set(float32)),
        score_low=min(scores),
        score_high=max(scores),
        total=math.fsum(low_precision),
        total_float32=math.fsum(float32),
    )
