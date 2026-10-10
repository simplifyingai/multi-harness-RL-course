"""The policy, its log-probabilities, the loss and one gradient-descent step.

The whole chapter's math sits in this file. A row of four scores is the
policy for one question: softmax turns it into one probability per query.
Training lowers a loss, ``-advantage * log p(chosen)``, by stepping every
score against the loss's gradient. The gradient of ``log p(chosen)`` with
respect to the scores is ``onehot(chosen) - probabilities``; ``grad_check.py``
measures it two other ways and gets the same numbers.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEMPERATURE = 1.0


# Copied from 02-plain-rl/bandit.py, unchanged (a test checks the two match).
def softmax(scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Turn a row of scores into probabilities that add up to 1.

    Args:
        scores: Any real numbers.
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
        scores: One row of the table.
        choice: Which query (0 for A).

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
    """Return the gradient of log p(choice), worked out by hand.

    How much log p(choice) rises per unit each score rises:
    ``onehot(choice) - probabilities``. Raising the chosen score helps it by
    ``1 - p``; raising any other score hurts it by that score's ``p``.

    Args:
        scores: One row of the table.
        choice: The query that was picked.

    Returns:
        One slope per score.
    """
    return onehot(choice, len(scores)) - softmax(scores)


def loss(scores: np.ndarray, choice: int, advantage: float) -> float:
    """Return the policy-gradient loss for one try: ``-advantage * log p(choice)``.

    Lowering it raises log p(choice) when the advantage is positive (do more
    of that) and lowers it when the advantage is negative (do less).

    Args:
        scores: One row of the table.
        choice: The query that was picked.
        advantage: How much better the try did than expected.

    Returns:
        The loss.
    """
    return -advantage * log_prob(scores, choice)


def loss_gradient(scores: np.ndarray, choice: int, advantage: float) -> np.ndarray:
    """Return the gradient of the loss: ``-advantage * (onehot - probabilities)``.

    Args:
        scores: One row of the table.
        choice: The query that was picked.
        advantage: How much better the try did than expected.

    Returns:
        One slope per score.
    """
    return -advantage * log_prob_gradient(scores, choice)


def gradient_step(scores: np.ndarray, gradient: np.ndarray, lr: float) -> np.ndarray:
    """Take one gradient-descent step: move every score against its slope.

    Args:
        scores: One row of the table.
        gradient: The loss's slope for each score.
        lr: The learning rate: how big the step is.

    Returns:
        The new row.
    """
    return scores - lr * gradient


def greedy(scores: np.ndarray) -> int:
    """Take the top score. When scores tie, take the first (A before B).

    Args:
        scores: One row of the table.

    Returns:
        The index of the top score.
    """
    return int(np.argmax(scores))  # argmax returns the first of equal values


def pick(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Choose a query at random by the row's probabilities (temperature 1).

    Args:
        scores: One row of the table.
        rng: The random generator for picks.

    Returns:
        The chosen index (0 for A).
    """
    return int(rng.choice(len(scores), p=softmax(scores)))


def chances(scores: np.ndarray) -> np.ndarray:
    """Apply softmax to every row of the table.

    Args:
        scores: The table of scores.

    Returns:
        One row of probabilities per question.
    """
    return np.array([softmax(row) for row in scores])
