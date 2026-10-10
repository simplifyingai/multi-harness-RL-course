"""GRPO's pieces: chapter 6's advantages and clipped loss, now with reply lengths.

``group_advantages`` is the project's (``mhrl/training/grpo.py``) line for
line, as in chapters 5 and 6: each rollout's reward minus its group's mean,
divided by the group's spread, or None when every reward in the group is the
same. ``loss_fn`` is the project's too, line for line, as in chapter 6:
``-min(ratio x A, clip(ratio, 0.8, 1.2) x A)``.

What is new is what a rollout learns from. In chapter 6 every call was one
decision, its action. Here a call is a reply: first its length, then the one
or two actions it holds. Each of those is a **decision** with its own
log-probability, the way a real reply is many tokens, each with its own. The
loss weighs every decision by its ratio, as the project weighs every token.

The agent's weights are now two things: chapter 5's action weights, and its
length weight, the log of its typical reply length (``replies.py``).
``loss_gradient`` works out the slope for both by hand, and a test checks it
against slopes measured by moving each number a hair.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from policy import action_scores, log_prob, weight_gradient
from replies import FLOOR_WEIGHT, length_log_prob, length_slope

# The project's step structure: each step picks 2 questions and runs a group
# of 4 rollouts for each (mhrl/cli.py: GRPO_TASKS_PER_STEP, GRPO_GROUP).
QUESTIONS_PER_STEP = 2
GROUP_SIZE = 4


class Defaults:
    """The project's GRPO constants this chapter uses (``mhrl/config.py``)."""

    # Two rewards closer than this count as a tie, and it keeps the division
    # safe when the spread is tiny.
    ADVANTAGE_EPS = 1e-4
    # How far the ratio may move from 1 before the clip stops the push:
    # clip(ratio, 1 - 0.2, 1 + 0.2), the range 0.8 to 1.2.
    CLIP_EPSILON = 0.2


@dataclass(frozen=True)
class Weights:
    """Everything the agent learns.

    Attributes:
        actions: Chapter 5's action weights: one row per feature (one set of
            five per level), one column per action.
        length: The length weight: the log of the typical reply length.
    """

    actions: np.ndarray
    length: float


@dataclass(frozen=True)
class Trained:
    """What one rollout brings to a step: its decisions, their old log p, its A.

    Attributes:
        situations: The features before each action decision.
        choices: The action taken at each action decision (an index).
        lengths: The log-length drawn at each reply-length decision.
        old_logprobs: ``log p`` of every decision as recorded when the rollout
            was sampled: the action decisions first, then the lengths.
        advantage: The rollout's advantage, shared by all its decisions.
    """

    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    lengths: tuple[float, ...]
    old_logprobs: tuple[float, ...]
    advantage: float

    @property
    def decisions(self) -> int:
        """How many decisions the rollout made: actions plus reply lengths."""
        return len(self.choices) + len(self.lengths)


@dataclass(frozen=True)
class Update:
    """One update: the gradient, and what every decision's ratio was.

    Attributes:
        gradient: The loss's slope for every weight and for the length weight
            (zeros with nothing to learn from).
        decisions: How many decisions the update learned from.
        loss: The update's loss, the average over those decisions, or None
            when no group had contrast.
        ratios: Every decision's ratio, new p / old p.
        clipped: For every decision, whether the clip switched its push off.
    """

    gradient: Weights
    decisions: int
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


def call_logprobs(weights: Weights, item: Trained) -> np.ndarray:
    """Work out ``log p`` of every decision with the given weights (the "new").

    The toy's version of the project's ``token_logprobs``: where the real
    model scores every token of a reply, the toy scores every decision.

    Args:
        weights: The weights to score with (the learner's, as they are now).
        item: The rollout's decisions.

    Returns:
        One log-probability per decision: the actions first, then the lengths.
    """
    actions = [
        log_prob(action_scores(weights.actions, situation), choice)
        for situation, choice in zip(item.situations, item.choices, strict=True)
    ]
    lengths = [length_log_prob(x, weights.length) for x in item.lengths]
    return np.array(actions + lengths)


def loss_fn(weights: Weights, item: Trained) -> tuple[float, np.ndarray]:
    """The clipped loss of one rollout, and each decision's ratio.

    ``mhrl/training/grpo.py``'s ``loss_fn`` line for line, with NumPy (``np``)
    in place of MLX (``mx``): NumPy has no gradient to stop, so the ratio is
    returned as it is.

    Args:
        weights: The learner's weights.
        item: The rollout: its decisions, their old log p and its advantage.

    Returns:
        The loss summed over the rollout's decisions, and every ratio.
    """
    eps = Defaults.CLIP_EPSILON
    new = call_logprobs(weights, item)
    ratio = np.exp(new - np.array(item.old_logprobs))
    unclipped = ratio * item.advantage
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
    return -np.minimum(unclipped, clipped).sum(), ratio


def clip_active(ratio: np.ndarray, advantage: float) -> np.ndarray:
    """Say, for each decision, whether the clip has switched its push off.

    Chapter 6's rule: ``min()`` picks the flat clipped term only when the
    decision has already moved more than 20% the way its advantage wants.

    Args:
        ratio: Every decision's ratio.
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


def loss_gradient(weights: Weights, item: Trained) -> Weights:
    """Work out the slope of one rollout's loss, by hand.

    For a decision the clip has not switched off, the loss is ``-ratio x A``,
    so its slope is ``-A x ratio`` times the slope of its ``log p``: chapter
    4's ``weight_gradient`` for an action, ``length_slope`` for a length. A
    clipped decision adds nothing.

    Args:
        weights: The learner's weights.
        item: The rollout.

    Returns:
        One slope per action weight, and one for the length weight.
    """
    ratio = np.exp(call_logprobs(weights, item) - np.array(item.old_logprobs))
    pushes = np.where(clip_active(ratio, item.advantage), 0.0, ratio)
    count = len(item.choices)
    action_slope = np.zeros(weights.actions.shape)
    for push, situation, choice in zip(
        pushes[:count], item.situations, item.choices, strict=True
    ):
        scores = action_scores(weights.actions, situation)
        action_slope += push * weight_gradient(situation, scores, choice)
    length_slope_sum = float(
        sum(
            push * length_slope(x, weights.length)
            for push, x in zip(pushes[count:], item.lengths, strict=True)
        )
    )
    return Weights(-item.advantage * action_slope, -item.advantage * length_slope_sum)


def step_update(trained: Sequence[Trained], weights: Weights) -> Update:
    """Work out one update from every rollout it learns from.

    The loss is summed over every decision of every rollout learned from and
    divided by how many decisions that is, as the project divides by tokens.

    Args:
        trained: The rollouts of the groups that had contrast.
        weights: The learner's weights.

    Returns:
        The gradient, the number of decisions, the loss, and every decision's
        ratio and whether it was clipped.
    """
    decisions = sum(item.decisions for item in trained)
    if decisions == 0:
        nothing = np.zeros(0)
        zero = Weights(np.zeros(weights.actions.shape), 0.0)
        return Update(zero, 0, None, nothing, nothing.astype(bool))
    slopes = [loss_gradient(weights, item) for item in trained]
    losses, ratios = zip(*(loss_fn(weights, item) for item in trained), strict=True)
    clipped = [
        clip_active(ratio, item.advantage)
        for ratio, item in zip(ratios, trained, strict=True)
    ]
    gradient = Weights(
        sum(slope.actions for slope in slopes) / decisions,
        sum(slope.length for slope in slopes) / decisions,
    )
    return Update(
        gradient,
        decisions,
        float(sum(losses)) / decisions,
        np.concatenate(ratios),
        np.concatenate(clipped),
    )


def apply_update(
    weights: Weights, gradient: Weights, lr: float, length_lr: float
) -> Weights:
    """Take one gradient-descent step on both parts of the weights.

    The length weight never goes below ``FLOOR_WEIGHT`` (a typical reply of
    20 tokens): a reply cannot be shorter than that, so below it no reply
    would differ and nothing could ever pull the weight back up.

    Args:
        weights: The weights before the step.
        gradient: The loss's slope for each of them.
        lr: The action weights' learning rate (chapter 5's 3).
        length_lr: The length weight's learning rate.

    Returns:
        The new weights.
    """
    length = weights.length - length_lr * gradient.length
    return Weights(weights.actions - lr * gradient.actions, max(length, FLOOR_WEIGHT))
