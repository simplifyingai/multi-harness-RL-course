"""The two rewards the project trained with, and GRPO's group advantages.

- ``reward_for`` is the project's reward today (``mhrl/rollout.py``): a right
  answer earns 1 plus a **token bonus** of up to 0.1, falling in a straight
  line from 0.1 at 0 generated tokens to 0 at 2,000; a wrong answer earns 0
  however short it is (chapter 7).
- ``call_reward_for`` is the reward of the project's FIRST RL attempt, the
  article's: a right answer earns 1 plus a **call bonus** of up to 0.1 for
  fewer tool calls (the whole bonus at 2 calls, none at the 8-call budget).
  A rollout cannot make fewer calls than it needs, but it can say more in
  each one; ``grpo_harness.py --attempt1`` trains with this reward again.
- ``group_advantages`` is the project's (``mhrl/training/grpo.py``, chapter
  5): each reward minus its group's mean, divided by the group's spread, or
  None when every reward is the same (a **no-contrast group**: skipped).

Both rewards reproduce the project's recorded ones exactly (the tests check
every rollout of both attempts). This module needs only the standard library.
"""

from __future__ import annotations

import statistics

RIGHT_REWARD = 1.0  # what a right answer earns before its bonus
WRONG_REWARD = 0.0
MAX_BONUS = 0.1  # the project's Defaults.MAX_TOOL_BONUS: the largest bonus
TOKEN_BUDGET = 2_000  # the project's token_budget: the bonus is 0 at this many
MIN_TOOL_CALLS = 2  # look at the data, then answer: the fewest a solve can take
MAX_CALLS = 8  # the project's max_model_calls: the proxy's per-rollout budget
ADVANTAGE_EPS = 1e-4  # rewards closer than this tie; keeps the division safe


def token_bonus(n_sampled: int, budget: int = TOKEN_BUDGET) -> float:
    """The project's bonus: at most 0.1, more for fewer generated tokens.

    Args:
        n_sampled: Tokens the model generated across the rollout.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        A bonus in [0, 0.1].
    """
    return MAX_BONUS * max(0.0, 1.0 - n_sampled / max(budget, 1))


def tool_bonus(n_calls: int, max_calls: int = MAX_CALLS) -> float:
    """The article's bonus: at most 0.1, more for fewer tool calls.

    Args:
        n_calls: Model calls the rollout made.
        max_calls: The per-rollout call budget.

    Returns:
        A bonus in [0, 0.1].
    """
    span = max(max_calls - MIN_TOOL_CALLS, 1)
    unused = max_calls - max(n_calls, MIN_TOOL_CALLS)
    return MAX_BONUS * max(0.0, min(1.0, unused / span))


def reward_for(correct: bool, n_sampled: int, budget: int = TOKEN_BUDGET) -> float:
    """Correctness plus the token bonus, which only a right answer earns.

    Args:
        correct: Whether the grader accepted the answer.
        n_sampled: Tokens the model generated across the rollout.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        The reward GRPO trains on (attempt 2's, and the project's today).
    """
    if not correct:
        return WRONG_REWARD
    return RIGHT_REWARD + token_bonus(n_sampled, budget)


def call_reward_for(correct: bool, n_calls: int, max_calls: int = MAX_CALLS) -> float:
    """Correctness plus the call bonus, which only a right answer earns.

    Args:
        correct: Whether the grader accepted the answer.
        n_calls: Model calls the rollout made.
        max_calls: The per-rollout call budget.

    Returns:
        The reward attempt 1 trained on (the article's).
    """
    if not correct:
        return WRONG_REWARD
    return RIGHT_REWARD + tool_bonus(n_calls, max_calls)


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Group-relative advantages, or None when the group has no contrast.

    Args:
        rewards: One reward per rollout in the group.

    Returns:
        Standardized advantages, or None if every reward is equal.
    """
    if len(rewards) < 2 or max(rewards) - min(rewards) < ADVANTAGE_EPS:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + ADVANTAGE_EPS) for r in rewards]
