"""The student: chapter 4's scores from features, with the call number as switches.

Chapter 4's agent described its situation with a few numbers, its features,
weighed each action's score from them, and turned the scores into
probabilities with softmax. All of that is unchanged, and like chapter 7's
agent it keeps one set of weights for every question, as a real model does.

One feature changes. Chapters 4 to 7 gave "calls used" as one number, the
share of the budget (0 at the first call, 7/8 at the last). A real model sees
its whole conversation, so it knows exactly which reply it is writing. Here
the call number is eight switches, "call 1" to "call 8", exactly one of them
on. That lets the student tie an action to a call number as sharply as to
anything else it sees, which is what imitation can latch onto. With
``calls_as_share=True`` the student gets chapters 4 to 7's single share
instead (``--calls-as-share``), to show what that changes.

The math is chapter 3's. The gradient of ``log p(action)`` with respect to the
scores is ``onehot(action) - probabilities``; with respect to the weights it is
that, times each feature.
"""

from __future__ import annotations

import numpy as np
from environment import BUDGET, LINKED_TABLES, State

DEFAULT_TEMPERATURE = 1.0
CALL_NAMES = tuple(f"call {number}" for number in range(1, BUDGET + 1))
BASE_NAMES = ("always 1", "tables seen", "result in hand", "answer written")
FEATURE_NAMES = (*BASE_NAMES, *CALL_NAMES)
SHARE_NAMES = (*BASE_NAMES, "calls used")  # chapters 4 to 7's features
ALWAYS = 1.0  # the constant feature: what the agent prefers before anything


# softmax, log_prob, onehot, log_prob_gradient, gradient_step, greedy and pick
# are chapter 3's, line for line (a test checks the code is the same).
def softmax(scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Turn a row of scores into probabilities that add up to 1.

    Args:
        scores: Any real numbers, one per action.
        temperature: Divides the scores first: above 1 makes the
            probabilities more even, below 1 makes the top score dominate.

    Returns:
        One probability per score.
    """
    # Subtracting the largest score changes no probability, but keeps exp()
    # from overflowing when scores grow large or the temperature is tiny.
    scaled = (scores - scores.max()) / temperature
    exps = np.exp(scaled)
    return exps / exps.sum()


def log_prob(scores: np.ndarray, choice: int) -> float:
    """Return log p(choice): the natural log of the choice's probability.

    ``log p(c) = score_c - log(sum of e**score)``, which is the log of
    softmax's ``e**score_c / sum of e**score`` without ever dividing.

    Args:
        scores: One score per action.
        choice: Which action (0 for the first).

    Returns:
        A number at or below 0; 0 means certain.
    """
    top = scores.max()  # as in softmax: keeps exp() from overflowing
    log_total = top + np.log(np.exp(scores - top).sum())
    return float(scores[choice] - log_total)


def onehot(choice: int, size: int) -> np.ndarray:
    """Return a row of zeros with a 1 at the choice.

    Args:
        choice: Where the 1 goes.
        size: How long the row is.

    Returns:
        The row.
    """
    row = np.zeros(size)
    row[choice] = 1.0
    return row


def log_prob_gradient(scores: np.ndarray, choice: int) -> np.ndarray:
    """Return the gradient of log p(choice) with respect to the scores.

    How much log p(choice) rises per unit each score rises:
    ``onehot(choice) - probabilities``. Raising the chosen score helps it by
    ``1 - p``; raising any other score hurts it by that score's ``p``.

    Args:
        scores: One score per action.
        choice: The action that was taken.

    Returns:
        One slope per score.
    """
    return onehot(choice, len(scores)) - softmax(scores)


def gradient_step(scores: np.ndarray, gradient: np.ndarray, lr: float) -> np.ndarray:
    """Take one gradient-descent step: move every number against its slope.

    Args:
        scores: The numbers being trained (here, the weights).
        gradient: The loss's slope for each of them.
        lr: The learning rate: how big the step is.

    Returns:
        The new numbers.
    """
    return scores - lr * gradient


def greedy(scores: np.ndarray) -> int:
    """Take the top score. When scores tie, take the first.

    Args:
        scores: One score per action.

    Returns:
        The index of the top score.
    """
    return int(np.argmax(scores))  # argmax returns the first of equal values


def pick(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Choose an action at random by the probabilities (temperature 1).

    Args:
        scores: One score per action.
        rng: The random generator for picks.

    Returns:
        The chosen index.
    """
    return int(rng.choice(len(scores), p=softmax(scores)))


def features(state: State, calls_as_share: bool = False) -> np.ndarray:
    """Describe the student's situation with twelve numbers.

    Args:
        state: The rollout's state before the next call.
        calls_as_share: Give the calls used as one number, the share of the
            budget, as chapters 4 to 7 did (five numbers in all).

    Returns:
        A constant 1; the share of the tables seen (0 to 1); 1 or 0 for
        result in hand and for answer written; then eight switches, one per
        call of the budget, with only the coming call's switched on (or the
        share). The question and its city are not among them.

    Raises:
        IndexError: If the rollout has already used its whole budget (never
            expected: a stopped rollout is not described).
    """
    seen = [
        ALWAYS,
        state.tables_seen / len(LINKED_TABLES),
        float(state.result is not None),
        float(state.written),
    ]
    if calls_as_share:
        return np.array([*seen, state.calls / BUDGET])
    calls = np.zeros(BUDGET)
    calls[state.calls] = 1.0  # state.calls counts finished calls: 0 before call 1
    return np.concatenate([seen, calls])


def weight_rows(calls_as_share: bool = False) -> int:
    """Return how many rows of weights the student has: one per feature.

    Args:
        calls_as_share: The calls used as one number instead of eight switches.

    Returns:
        12 (four features, then the eight call switches), or 5 with the share.
    """
    return len(SHARE_NAMES if calls_as_share else FEATURE_NAMES)


def action_scores(weights: np.ndarray, situation: np.ndarray) -> np.ndarray:
    """Compute one score per action: a weighted sum of the features.

    Args:
        weights: One row per feature, one column per action.
        situation: The features, as ``features`` returns them.

    Returns:
        One score per action.
    """
    return situation @ weights


def weight_gradient(
    situation: np.ndarray, scores: np.ndarray, choice: int
) -> np.ndarray:
    """Return the gradient of log p(choice) with respect to the weights.

    Each weight joins one feature to one action, so its slope is the
    feature's value times chapter 3's slope for that action's score.

    Args:
        situation: The features at this call.
        scores: The action scores at this call.
        choice: The action that was taken.

    Returns:
        One slope per weight, shaped like the weights.
    """
    return np.outer(situation, log_prob_gradient(scores, choice))
