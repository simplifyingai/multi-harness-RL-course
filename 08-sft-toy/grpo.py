"""GRPO's pieces: advantages from a group, and the clipped loss the project uses.

``group_advantages`` is chapter 5's, which is the project's
(``mhrl/training/grpo.py``) line for line: each rollout's reward minus its
group's mean, divided by the group's spread, or None when every reward in the
group is the same.

``loss_fn`` is new, and it too is the project's line for line. Every call
keeps the log-probability its action had when the rollout was sampled (the
"old" one). The learner works out the same action's log-probability with the
weights it has now (the "new" one). Their ratio, new probability / old
probability, says how much more or less likely the action has become since it
was sampled. The loss is ``-min(ratio x A, clip(ratio, 0.8, 1.2) x A)``.

NumPy has no automatic gradient, so ``loss_gradient`` works out the loss's
slope by hand, and a test checks it against slopes measured by moving each
weight a hair. When the ratio is exactly 1 the slope is chapter 5's,
number for number: one update on a fresh batch is REINFORCE with a group
baseline.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from policy import action_scores, log_prob, weight_gradient

# The project's step structure: each step picks 2 questions and runs a group
# of 4 rollouts for each (mhrl/cli.py: GRPO_TASKS_PER_STEP, GRPO_GROUP).
QUESTIONS_PER_STEP = 2
GROUP_SIZE = 4

# What a group can be: no contrast because all wrong or all right, or signal.
ALL_WRONG = "all wrong"
ALL_RIGHT = "all right"
SIGNAL = "signal"
CORRECT_REWARD = 1  # what a right answer earns in this chapter


class Defaults:
    """The project's GRPO constants this chapter uses (``mhrl/config.py``)."""

    # Two rewards closer than this count as a tie, and it keeps the division
    # safe when the spread is tiny.
    ADVANTAGE_EPS = 1e-4
    # How far the ratio may move from 1 before the clip stops the push:
    # clip(ratio, 1 - 0.2, 1 + 0.2), the range 0.8 to 1.2.
    CLIP_EPSILON = 0.2


@dataclass(frozen=True)
class Trained:
    """What one rollout brings to a step: its calls, their old log p, its advantage.

    Attributes:
        situations: The features the agent saw before each call.
        choices: The action taken at each call (an index into the actions).
        old_logprobs: ``log p`` of each action taken, as recorded when the
            rollout was sampled (or, with the bug, recomputed later).
        advantage: The rollout's advantage, shared by all its calls.
    """

    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    advantage: float


@dataclass(frozen=True)
class Update:
    """One update: the gradient, and what every call's ratio was.

    Attributes:
        gradient: The loss's slope for every weight (zeros with nothing to
            learn from).
        calls: How many calls the update learned from.
        loss: The update's loss, the average over those calls, or None when
            no group had contrast.
        ratios: Every call's ratio, new p / old p, with the weights used.
        clipped: For every call, whether the clip switched its push off.
    """

    gradient: np.ndarray
    calls: int
    loss: float | None
    ratios: np.ndarray
    clipped: np.ndarray


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


def call_logprobs(weights: np.ndarray, item: Trained) -> np.ndarray:
    """Work out ``log p`` of each call's action with the given weights (the "new").

    The toy's version of the project's ``token_logprobs``: in the toy one call
    is one action, where the real model writes many tokens per call.

    Args:
        weights: The weights to score with (the learner's, as they are now).
        item: The rollout's calls.

    Returns:
        One log-probability per call.
    """
    return np.array(
        [
            log_prob(action_scores(weights, situation), choice)
            for situation, choice in zip(item.situations, item.choices, strict=True)
        ]
    )


def loss_fn(weights: np.ndarray, item: Trained) -> tuple[float, np.ndarray]:
    """The clipped loss of one rollout, and each call's ratio.

    ``mhrl/training/grpo.py``'s ``loss_fn`` line for line, with NumPy (``np``)
    in place of MLX (``mx``): NumPy has no gradient to stop, so the ratio is
    returned as it is.

    Args:
        weights: The learner's weights.
        item: The rollout: its calls, their old log p and its advantage.

    Returns:
        The loss summed over the rollout's calls, and every call's ratio.
    """
    eps = Defaults.CLIP_EPSILON
    new = call_logprobs(weights, item)
    ratio = np.exp(new - np.array(item.old_logprobs))
    unclipped = ratio * item.advantage
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
    return -np.minimum(unclipped, clipped).sum(), ratio


def unclipped_loss_fn(weights: np.ndarray, item: Trained) -> tuple[float, np.ndarray]:
    """The same loss without the clip: ``-ratio x A``, for ``--no-clip``.

    Its first two lines repeat loss_fn's on purpose: loss_fn stays exactly the
    project's, so the version without the clip is written out beside it.

    Args:
        weights: The learner's weights.
        item: The rollout: its calls, their old log p and its advantage.

    Returns:
        The loss summed over the rollout's calls, and every call's ratio.
    """
    new = call_logprobs(weights, item)
    ratio = np.exp(new - np.array(item.old_logprobs))
    return -(ratio * item.advantage).sum(), ratio


def clip_active(ratio: np.ndarray, advantage: float) -> np.ndarray:
    """Say, for each call, whether the clip has switched its push off.

    ``min()`` picks the clipped term, which no longer depends on the weights,
    only when the action has already moved far enough the way the advantage
    wants: more likely than 1.2 times its old probability when A > 0, less
    likely than 0.8 times when A < 0. A move the wrong way is never clipped.

    Args:
        ratio: Every call's ratio.
        advantage: The rollout's advantage.

    Returns:
        True where the clip is active.
    """
    eps = Defaults.CLIP_EPSILON
    if advantage > 0:
        return ratio > 1.0 + eps
    if advantage < 0:
        return ratio < 1.0 - eps
    return np.zeros(ratio.shape, dtype=bool)


def outside_clip_range(ratio: np.ndarray) -> np.ndarray:
    """Say, for each call, whether its ratio lies outside 0.8 to 1.2.

    Outside is not the same as clipped: a ratio of 1.5 on a rollout with
    A < 0 is outside the range, but the clip leaves it alone.

    Args:
        ratio: Every call's ratio.

    Returns:
        True where the ratio is below 0.8 or above 1.2.
    """
    eps = Defaults.CLIP_EPSILON
    return (ratio < 1.0 - eps) | (ratio > 1.0 + eps)


def loss_gradient(weights: np.ndarray, item: Trained, clip: bool = True) -> np.ndarray:
    """Work out the slope of one rollout's loss for every weight, by hand.

    For a call the clip has not switched off, the loss is ``-ratio x A`` and
    ``ratio = exp(new log p - old log p)``, so its slope is ``-A x ratio``
    times the slope of ``log p`` (chapter 4's ``weight_gradient``). A clipped
    call adds nothing. With every ratio exactly 1 this is chapter 5's
    ``rollout_gradient``, number for number.

    Args:
        weights: The learner's weights.
        item: The rollout.
        clip: Whether the clip is on (``--no-clip`` turns it off).

    Returns:
        One slope per weight, shaped like the weights.
    """
    ratio = np.exp(call_logprobs(weights, item) - np.array(item.old_logprobs))
    switched_off = clip_active(ratio, item.advantage) & clip
    pushes = np.where(switched_off, 0.0, ratio)
    total = sum(
        push * weight_gradient(situation, action_scores(weights, situation), choice)
        for push, situation, choice in zip(
            pushes, item.situations, item.choices, strict=True
        )
    )
    return -item.advantage * np.asarray(total)


def step_update(
    trained: Sequence[Trained], weights: np.ndarray, clip: bool = True
) -> Update:
    """Work out one update from every rollout it learns from.

    The loss is summed over every call of every rollout learned from and
    divided by how many calls that is, as the project divides by its tokens.

    Args:
        trained: The rollouts of the groups that had contrast.
        weights: The learner's weights.
        clip: Whether the clip is on.

    Returns:
        The gradient, the number of calls, the loss, and every call's ratio
        and whether it was clipped.
    """
    calls = sum(len(item.choices) for item in trained)
    if calls == 0:
        nothing = np.zeros(0)
        return Update(np.zeros(weights.shape), 0, None, nothing, nothing.astype(bool))
    total = sum(loss_gradient(weights, item, clip) for item in trained)
    loss_of: Callable[[np.ndarray, Trained], tuple[float, np.ndarray]] = (
        loss_fn if clip else unclipped_loss_fn
    )
    losses, ratios = zip(*(loss_of(weights, item) for item in trained), strict=True)
    clipped = [
        clip_active(ratio, item.advantage) & clip
        for ratio, item in zip(ratios, trained, strict=True)
    ]
    return Update(
        np.asarray(total) / calls,
        calls,
        float(sum(losses)) / calls,
        np.concatenate(ratios),
        np.concatenate(clipped),
    )
