"""The policy: one score per action, computed from what the agent can see.

In chapters 2 and 3 the policy was a table with one row of scores per
question. An agent that makes several calls is in a different situation at
every call, so here the row is computed instead of looked up. The agent
describes its situation with a few numbers, its **features**: whether it has
seen the tables, whether it holds a result, whether it has written an answer,
and how many calls it has used, plus a constant 1 so that it can have a
favourite action before it has done anything. Each action's score is a
weighted sum of those features, and softmax turns the scores into one
probability per action, exactly as before.

The math is chapter 3's. The gradient of ``log p(action)`` with respect to the
scores is ``onehot(action) - probabilities``; with respect to the weights it is
that, times each feature. A rollout's loss is ``-advantage`` times the sum of
the log-probabilities of all its actions, so every action in the rollout gets
the same push.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from environment import BUDGET, State

DEFAULT_TEMPERATURE = 1.0
FEATURE_NAMES = (
    "always 1",
    "schema seen",
    "result in hand",
    "answer written",
    "calls used",
)
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


def features(state: State) -> np.ndarray:
    """Describe the agent's situation with five numbers.

    Args:
        state: The rollout's state before the next call.

    Returns:
        A constant 1; then 1 or 0 for schema seen, result in hand and answer
        written; then the calls used as a share of the budget (0 at the first
        call, 7/8 at the last). The question and its city are not among them.
    """
    return np.array(
        [
            ALWAYS,
            float(state.schema_seen),
            float(state.result is not None),
            float(state.written),
            state.calls / BUDGET,
        ]
    )


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


def rollout_loss(advantage: float, log_probs: Sequence[float]) -> float:
    """Return a rollout's loss: ``-advantage x`` the sum of its log-probabilities.

    Args:
        advantage: How much better the rollout did than expected.
        log_probs: ``log p`` of each action the rollout took.

    Returns:
        The loss.
    """
    return -advantage * float(sum(log_probs))


def rollout_gradient(
    situations: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    choices: Sequence[int],
    advantage: float,
) -> np.ndarray:
    """Return the gradient of the rollout's loss with respect to the weights.

    Every action shares the rollout's one advantage: the reward came once, at
    the end, and nothing says which call earned it.

    Args:
        situations: The features at each call.
        scores: The action scores at each call.
        choices: The action taken at each call.
        advantage: The rollout's advantage.

    Returns:
        One slope per weight.
    """
    total = sum(
        weight_gradient(situation, row, choice)
        for situation, row, choice in zip(situations, scores, choices, strict=True)
    )
    return -advantage * np.asarray(total)
