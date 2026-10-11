"""Rewards, group advantages and the clip test, as the project has them.

- ``token_bonus`` and ``reward_for`` are the project's ``mhrl/rollout.py``:
  a right answer earns 1 plus a bonus of up to 0.1 for generating fewer
  tokens (the whole bonus at 0 tokens, none at 2,000); a wrong answer earns
  0 however short it is, so the bonus cannot be bought by giving up early.
- ``group_advantages`` is the project's ``mhrl/training/grpo.py`` (chapter
  5's): each reply's reward minus its group's mean, divided by the group's
  spread, or None when every reward in the group is the same.
- ``clip_active`` is chapter 6's test for whether the clip has switched a
  token's push off.

This module needs only the standard library.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

RIGHT_REWARD = 1.0  # what a right answer earns before its bonus
WRONG_REWARD = 0.0
ALL_WRONG = "all wrong"
ALL_RIGHT = "all right"
MIXED = "right and wrong"


class Defaults:
    """The project's constants this chapter uses (``mhrl/config.py``)."""

    # Two rewards closer than this count as a tie, and it keeps the division
    # safe when the spread is tiny.
    ADVANTAGE_EPS = 1e-4
    # How far the ratio may move from 1 before the clip stops the push:
    # clip(ratio, 1 - 0.2, 1 + 0.2), the range 0.8 to 1.2.
    CLIP_EPSILON = 0.2
    # The largest efficiency bonus a right answer can add.
    MAX_TOOL_BONUS = 0.1


def token_bonus(n_sampled: int, budget: int) -> float:
    """Our efficiency bonus: at most 0.1, more for fewer generated tokens.

    The project's ``mhrl/rollout.py::token_bonus``. The article pays for
    fewer tool calls; in the project's first RL run that traded calls for
    longer replies, so the project pays for fewer tokens.

    Args:
        n_sampled: Tokens the model generated.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        A bonus in [0, ``Defaults.MAX_TOOL_BONUS``].
    """
    return Defaults.MAX_TOOL_BONUS * max(0.0, 1.0 - n_sampled / max(budget, 1))


def reward_for(correct: bool, n_sampled: int, budget: int) -> float:
    """Correctness (1 or 0) plus the bonus, which only a correct answer earns.

    The project's ``mhrl/rollout.py::reward_for``.

    Args:
        correct: Whether the answer was right.
        n_sampled: Tokens the model generated.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        The reward GRPO trains on.
    """
    if not correct:
        return WRONG_REWARD
    return RIGHT_REWARD + token_bonus(n_sampled, budget)


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Group-relative advantages, or None when the group has no contrast.

    Args:
        rewards: One reward per rollout in the group.

    Returns:
        Standardized advantages, or None if every reward is equal.
    """
    if len(rewards) < 2 or max(rewards) - min(rewards) < Defaults.ADVANTAGE_EPS:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + Defaults.ADVANTAGE_EPS) for r in rewards]


def group_outcome(rewards: Sequence[float]) -> str:
    """Say which answers a group holds: all right, all wrong, or both.

    Args:
        rewards: One reward per reply in the group.

    Returns:
        ``"all right"``, ``"all wrong"`` or ``"right and wrong"``.
    """
    right = [reward >= RIGHT_REWARD for reward in rewards]
    if all(right):
        return ALL_RIGHT
    return MIXED if any(right) else ALL_WRONG


def clip_active(ratio: float, advantage: float) -> bool:
    """Say whether the clip has switched one token's push off (chapter 6).

    ``min()`` picks the clipped term, which no longer depends on the weights,
    only when the token has already moved far enough the way the advantage
    wants: more likely than 1.2 times its old probability when A > 0, less
    likely than 0.8 times when A < 0.

    Args:
        ratio: The token's new probability divided by its old one.
        advantage: Its reply's advantage.

    Returns:
        True where the clip is active.
    """
    if advantage > 0:
        return ratio > 1.0 + Defaults.CLIP_EPSILON
    if advantage < 0:
        return ratio < 1.0 - Defaults.CLIP_EPSILON
    return False
