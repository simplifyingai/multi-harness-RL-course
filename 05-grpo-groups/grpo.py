"""GRPO's two pieces: advantages from a group, and one update per step.

``group_advantages`` is the project's (``mhrl/training/grpo.py``) line for
line: each rollout's reward minus its group's mean, divided by the group's
spread (its standard deviation), or None when every reward in the group is
the same. A test checks the code is the same.

``step_update`` is chapter 3's policy gradient with those advantages. Every
call of a rollout shares the rollout's advantage (chapter 4), every group with
contrast is learned from, and the step's loss is the average over every call
it learns from, as the project averages its loss over every token.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from policy import rollout_gradient, rollout_loss

# The project's step structure: each step picks 2 questions and runs a group
# of 4 rollouts for each (mhrl/cli.py: GRPO_TASKS_PER_STEP, GRPO_GROUP).
QUESTIONS_PER_STEP = 2
GROUP_SIZE = 4

# What a group can be: no contrast because all wrong or all right, or signal.
ALL_WRONG = "all wrong"
ALL_RIGHT = "all right"
SIGNAL = "signal"
OUTCOMES = (ALL_WRONG, ALL_RIGHT, SIGNAL)
# Beside the groups, the tables count rollouts and how many were right.
ROLLOUTS = "rollouts"
RIGHT = "right"
COUNTS = (*OUTCOMES, ROLLOUTS, RIGHT)
CORRECT_REWARD = 1  # what a right answer earns in this chapter


class Defaults:
    """The project's GRPO constant this chapter uses (``mhrl/config.py``)."""

    # Two rewards closer than this count as a tie, and it keeps the division
    # safe when the spread is tiny.
    ADVANTAGE_EPS = 1e-4


@dataclass(frozen=True)
class Trained:
    """What one rollout contributes to a step: its choices and its advantage.

    Attributes:
        situations: The features the agent saw before each call.
        scores: The action scores at each call.
        choices: The action taken at each call (an index into the actions).
        log_probs: The log-probability of each action taken.
        advantage: The rollout's advantage, shared by all its calls.
    """

    situations: tuple[np.ndarray, ...]
    scores: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    log_probs: tuple[float, ...]
    advantage: float


@dataclass(frozen=True)
class Update:
    """One step's update.

    Attributes:
        gradient: The loss's slope for every weight (zeros with nothing to
            learn from).
        calls: How many calls the step learned from.
        loss: The step's loss, the average over those calls, or None when no
            group had contrast.
    """

    gradient: np.ndarray
    calls: int
    loss: float | None


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


def group_outcome(rewards: list[float]) -> str:
    """Say what a group's rewards can teach.

    Args:
        rewards: One reward per rollout in the group.

    Returns:
        ``"signal"`` when group_advantages finds contrast, else ``"all right"``
        or ``"all wrong"`` (a tie on the top reward, or on anything less).
    """
    if group_advantages(rewards) is not None:
        return SIGNAL
    return ALL_RIGHT if min(rewards) >= CORRECT_REWARD else ALL_WRONG


def step_update(trained: Sequence[Trained], shape: tuple[int, int]) -> Update:
    """Work out one step's update from every rollout it learns from.

    The step's loss is ``-advantage x log p`` summed over every call of every
    rollout learned from, divided by how many calls that is.

    Args:
        trained: The rollouts of the groups that had contrast.
        shape: The shape of the weights, for an all-zero gradient.

    Returns:
        The gradient, the number of calls and the loss.
    """
    calls = sum(len(item.choices) for item in trained)
    if calls == 0:
        return Update(np.zeros(shape), 0, None)
    total = sum(
        rollout_gradient(item.situations, item.scores, item.choices, item.advantage)
        for item in trained
    )
    loss = sum(rollout_loss(item.advantage, item.log_probs) for item in trained)
    return Update(np.asarray(total) / calls, calls, float(loss) / calls)
