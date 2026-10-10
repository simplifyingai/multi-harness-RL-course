"""The reply length: drawn around a typical length the agent learns.

Chapter 6's agent only chose actions. This one also chooses how long each
reply is, the way a model chooses how much to write before its tool call.
Every reply's length is drawn from a bell curve around the agent's **typical
reply length**, which starts at 40 tokens. The curve is on a log scale, as
real reply lengths are: a reply 30% longer than typical is as likely as one
30% shorter, so the spread grows with the length.

The typical length is one more number the agent learns, like a weight. It is
kept as its natural log, the **length weight**, so that a step of the same
size moves a 40-token reply and a 100-token reply by the same share. A reply
length that did better than its group pulls the length weight toward it; one
that did worse pushes it away.

The math is chapter 3's again, for a bell curve instead of softmax. Write x
for the log of a reply's length, w for the length weight and s for the
spread. Then ``log p(x) = -(x - w)**2 / (2 s**2)`` plus a constant, and its
slope with respect to w is ``(x - w) / s**2``: positive for a reply longer
than typical, negative for a shorter one.
"""

from __future__ import annotations

import math

import numpy as np

START_LENGTH = 40  # tokens: the typical reply length at the start
START_WEIGHT = math.log(START_LENGTH)  # the length weight at the start
SPREAD = 0.3  # on the log scale: replies vary by about 30% around typical
MIN_REPLY = 20  # tokens: the fewest a reply that names an action can have
FLOOR_WEIGHT = math.log(MIN_REPLY)  # the length weight never goes below this
# log(s x sqrt(2 pi)): the bell curve's constant, so log p is the real one.
LOG_SCALE = math.log(SPREAD * math.sqrt(2.0 * math.pi))


def typical_length(weight: float) -> float:
    """Turn a length weight back into tokens.

    Args:
        weight: The length weight (the log of the typical length).

    Returns:
        The typical reply length, in tokens.
    """
    return math.exp(weight)


def sample_length(weight: float, rng: np.random.Generator) -> float:
    """Draw one reply's length, as its log, from the bell curve.

    Args:
        weight: The length weight.
        rng: The random generator for picks.

    Returns:
        The log of the reply's length (any real number).
    """
    return weight + SPREAD * float(rng.standard_normal())


def reply_tokens(drawn: float) -> int:
    """Turn a drawn log-length into the reply's tokens.

    Args:
        drawn: The log of the reply's length.

    Returns:
        The length in whole tokens, never below ``MIN_REPLY``: however short
        the agent aims, a reply must name its action.
    """
    return max(MIN_REPLY, round(math.exp(drawn)))


def length_log_prob(drawn: float, weight: float) -> float:
    """Return log p of a drawn log-length: how likely the bell curve made it.

    Args:
        drawn: The log of the reply's length.
        weight: The length weight it was drawn around.

    Returns:
        The log of the bell curve's density at ``drawn``.
    """
    distance = (drawn - weight) / SPREAD
    return -0.5 * distance * distance - LOG_SCALE


def length_slope(drawn: float, weight: float) -> float:
    """Return the slope of that log p with respect to the length weight.

    Raising the length weight makes longer replies more likely and shorter
    ones less likely, so the slope is positive for a reply longer than
    typical and negative for a shorter one.

    Args:
        drawn: The log of the reply's length.
        weight: The length weight.

    Returns:
        ``(drawn - weight) / SPREAD**2``.
    """
    return (drawn - weight) / (SPREAD * SPREAD)
