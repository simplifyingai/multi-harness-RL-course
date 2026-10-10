# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.3",
#     "mlx==0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""Log-probabilities, then the gradient of log p three ways, then chapter 2's nudge.

First the probabilities beside their log-probabilities, and why adding logs
beats multiplying chances. Then the gradient of log p(C), the slope of
log p(C) for each score, three ways: measured by moving one score a hair
down and up, worked out by hand (``onehot - probabilities``), and computed
by MLX, an automatic-gradient library. The last line shows that chapter 2's
nudge was exactly reward times the gradient of log p.

Run it with ``uv run grad_check.py``. MLX installs only on Apple-silicon Macs;
anywhere else the script says so and still runs the other two ways.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import numpy as np
from policy import (
    gradient_step,
    log_prob,
    log_prob_gradient,
    loss_gradient,
    onehot,
    softmax,
)
from questions import LETTERS

try:
    import mlx.core as mx
except ImportError:  # MLX installs only on Apple-silicon Macs
    mx = None  # every MLX step checks this and says MLX is missing

DEFAULT_SEED = 0
EPSILON = 0.001  # how far a score is nudged, down and up, to measure a slope
RANDOM_ROWS = 1000
SCORE_RANGE = 5.0  # random scores fall between -5 and +5
LR = 0.25  # chapter 2's default learning rate
REWARD_EVERY = 2  # every other random try is rewarded, so both 0 and 1 are checked
REWARDED = 1.0
CHOSEN = LETTERS.index("C")  # the walk-through follows a pick of C
PROBABILITIES = (1.00, 0.90, 0.50, 0.25, 0.10, 0.01)
NOTES = {1.00: "certain", 0.25: "one query in four: where every row starts"}
PAIR = (0.90, 0.25)  # two picks: B at 90% for one question, C at 25% for another
MANY_PICKS = 1100
COIN = 0.5
DIGITS = 3
CLOSE_DIGITS = 6
COLUMN = 15

Gradient = Callable[[np.ndarray, int], np.ndarray]


# Copied from 02-plain-rl/bandit.py, unchanged (a test checks the two match).
def nudge(scores: np.ndarray, choice: int, reward: int, lr: float) -> np.ndarray:
    """Move a row toward the choice if it earned a reward.

    ``onehot - probabilities`` is positive for the choice and negative for
    the others, so a reward raises the choice and lowers the rest. A reward
    of 0 changes nothing. The probabilities are the scores' own (temperature
    1): the temperature only changes how a query is picked.

    Args:
        scores: The row of the question that was asked.
        choice: The query that was picked.
        reward: The grader's reward, 1 or 0.
        lr: The learning rate.

    Returns:
        The new row.
    """
    onehot = np.zeros_like(scores)
    onehot[choice] = 1.0
    return scores + lr * reward * (onehot - softmax(scores))


def moved(scores: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two copies of a row: one score moved down a hair, and up a hair.

    Args:
        scores: One row of scores.
        index: Which score to move.

    Returns:
        The row with that score lowered by ``EPSILON``, and with it raised.
    """
    down, up = scores.copy(), scores.copy()
    down[index] -= EPSILON
    up[index] += EPSILON
    return down, up


def measured_gradient(scores: np.ndarray, choice: int) -> np.ndarray:
    """Measure the gradient of log p(choice) by moving each score a hair.

    Each score is moved down by ``EPSILON`` and up by ``EPSILON``; the change
    in log p(choice), divided by the change in the score, is that score's
    slope. Nothing here knows the formula.

    Args:
        scores: One row of scores.
        choice: The query whose log-probability is watched.

    Returns:
        One measured slope per score.
    """
    slopes = np.zeros(len(scores))
    for index in range(len(scores)):
        down, up = moved(scores, index)
        change = log_prob(up, choice) - log_prob(down, choice)
        slopes[index] = change / (2 * EPSILON)  # the score moved 2 x EPSILON
    return slopes


def mlx_gradient(scores: np.ndarray, choice: int) -> np.ndarray:
    """Ask MLX, an automatic-gradient library, for the gradient of log p(choice).

    ``mx.grad`` takes a function and returns a new function that computes
    its gradient. We only write log p; MLX works out the slopes.

    Args:
        scores: One row of scores.
        choice: The query whose log-probability is differentiated.

    Returns:
        One slope per score.

    Raises:
        RuntimeError: If MLX is not installed.
    """
    if mx is None:
        raise RuntimeError("MLX is not installed")

    def log_p(row: mx.array) -> mx.array:
        # The same formula as policy.log_prob: score minus log of the total.
        return row[choice] - mx.logsumexp(row)

    # Four numbers need no GPU; the CPU also gives the same digits every run.
    with mx.stream(mx.cpu):
        gradient = mx.grad(log_p)(mx.array(scores, dtype=mx.float32))
        return np.array(gradient, dtype=np.float64)


def numbers(values: np.ndarray, digits: int = 2, signed: bool = True) -> str:
    """Write a row of numbers in brackets.

    Args:
        values: The numbers.
        digits: Decimals to show.
        signed: Show a + before positive numbers (slopes and steps have a
            direction; probabilities do not).

    Returns:
        For example ``"[-0.25 -0.25 +0.75 -0.25]"``.
    """
    sign = "+" if signed else ""
    # Adding 0.0 turns a rounded -0.00 into 0.00.
    cells = [f"{round(float(v), digits) + 0.0:{sign}.{digits}f}" for v in values]
    return "[" + " ".join(cells) + "]"


def show_log_probabilities() -> None:
    """Print probabilities beside log-probabilities, and why logs are added."""
    print("log-probabilities: log p is the natural log of a probability p")
    print("     p     log p")
    for p in PROBABILITIES:
        note = NOTES.get(p, "")
        print(f"  {p:.2f}  {math.log(p) + 0.0:>7.{DIGITS}f}   {note}".rstrip())
    first, second = PAIR
    both = first * second
    print(
        f"two picks, B for one question (p {first:.2f}) and C for another"
        f" (p {second:.2f}):"
    )
    print(
        f"  the chance of both is {first:.2f} x {second:.2f} = {both:.3f},"
        f" and log {both:.3f} = {math.log(both):.{DIGITS}f}"
    )
    print(
        f"  the two log-probabilities add up to it: {math.log(first):.{DIGITS}f}"
        f" + {math.log(second):.{DIGITS}f} = "
        f"{math.log(first) + math.log(second):.{DIGITS}f}"
    )
    print(f"{MANY_PICKS:,} picks at p {COIN:.2f} each:")
    print(
        f"  multiplied, the chance is {COIN**MANY_PICKS}"
        " (too small for the computer to hold)"
    )
    print(f"  added, the log-probability is {MANY_PICKS * math.log(COIN):.{DIGITS}f}")


def show_three_ways(scores: np.ndarray) -> None:
    """Print the gradient of log p(C) on one row: measured, by hand, with MLX.

    Args:
        scores: The row to differentiate at.
    """
    letter = LETTERS[CHOSEN]
    chances = softmax(scores)
    print(f"\nthe gradient of log p({letter}): for each score, how fast")
    print(f"log p({letter}) changes when that score changes (its slope)")
    print(
        f"row scores {numbers(scores)} · probabilities {numbers(chances, signed=False)}"
    )
    print(f"\n1. measured: move one score {EPSILON} down and up, watch log p({letter})")
    print(
        f"   {'score':<8}{'after -' + str(EPSILON):<{COLUMN}}"
        f"{'after +' + str(EPSILON):<{COLUMN}}slope = change / {2 * EPSILON:g}"
    )
    measured = measured_gradient(scores, CHOSEN)
    for index, name in enumerate(LETTERS):
        down, up = moved(scores, index)
        print(
            f"   {name:<8}{log_prob(down, CHOSEN):<{COLUMN}.{CLOSE_DIGITS}f}"
            f"{log_prob(up, CHOSEN):<{COLUMN}.{CLOSE_DIGITS}f}"
            f"{measured[index]:+.{DIGITS}f}"
        )
    print(f"   measured gradient: {numbers(measured, DIGITS)}")
    picked = onehot(CHOSEN, len(scores))
    print(
        f"2. by hand: onehot({letter}) - probabilities"
        f" = {numbers(picked, 0, signed=False)} - {numbers(chances, signed=False)}"
    )
    print(f"   = {numbers(log_prob_gradient(scores, CHOSEN), DIGITS)}")
    if mx is None:
        print("3. MLX is not installed here (it needs an Apple-silicon Mac):")
        print("   skipping the automatic gradient; ways 1 and 2 still ran")
    else:
        print(f"3. MLX, an automatic-gradient library: mx.grad of log p({letter})")
        print(f"   = {numbers(mlx_gradient(scores, CHOSEN), DIGITS)}")


def largest_difference(
    rows: np.ndarray, choices: np.ndarray, first: Gradient, second: Gradient
) -> float:
    """Return the largest gap between two ways of computing the gradient.

    Args:
        rows: Rows of scores.
        choices: One chosen query per row.
        first: One way to compute the gradient.
        second: The other way.

    Returns:
        The largest absolute difference over every row and score.
    """
    return max(
        float(np.abs(first(row, choice) - second(row, choice)).max())
        for row, choice in zip(rows, choices, strict=True)
    )


def show_random_rows(rows: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    """Compare the three ways on many random rows and print the largest gaps.

    Args:
        rows: Rows of scores.
        choices: One chosen query per row.

    Returns:
        The largest gap from the by-hand gradient, per way that ran (the
        tests read it; the script only prints it).
    """
    print(f"\nthe same check on {len(rows):,} random rows and choices")
    gaps = {
        "measured": largest_difference(
            rows, choices, measured_gradient, log_prob_gradient
        )
    }
    print(f"  measured vs by hand: largest difference {gaps['measured']:.1e}")
    if mx is not None:
        gaps["MLX"] = largest_difference(rows, choices, mlx_gradient, log_prob_gradient)
        print(f"  MLX vs by hand:      largest difference {gaps['MLX']:.1e}")
        print("  (MLX works with 32-bit numbers, about 7 significant digits)")
    return gaps


def gap_text(gap: float) -> str:
    """Write a difference: exactly 0 as "0", anything else like "1.6e-08".

    Args:
        gap: The difference.

    Returns:
        The difference as text.
    """
    return "0" if gap == 0 else f"{gap:.1e}"


def nudge_gap(
    rows: np.ndarray, choices: np.ndarray, rewards: np.ndarray, gradient: Gradient
) -> float:
    """Return the largest gap between chapter 2's nudge and a gradient step.

    Args:
        rows: Rows of scores.
        choices: One chosen query per row.
        rewards: One reward, 0 or 1, per row.
        gradient: A way to compute the gradient of log p.

    Returns:
        The largest absolute difference over every row and score.
    """
    gaps = []
    for row, choice, reward in zip(rows, choices, rewards, strict=True):
        nudged = nudge(row, choice, int(reward), LR)
        # The loss -reward x log p has gradient -reward x the gradient of log p.
        stepped = gradient_step(row, -reward * gradient(row, choice), LR)
        gaps.append(float(np.abs(nudged - stepped).max()))
    return max(gaps)


def show_chapter2(scores: np.ndarray, rows: np.ndarray, choices: np.ndarray) -> None:
    """Show one gradient-descent step equal chapter 2's nudge, then the last line.

    Args:
        scores: The walk-through row.
        rows: Random rows of scores.
        choices: One chosen query per random row.
    """
    letter = LETTERS[CHOSEN]
    gradient = loss_gradient(scores, CHOSEN, REWARDED)
    stepped = gradient_step(scores, gradient, LR)
    nudged = nudge(scores, CHOSEN, int(REWARDED), LR)
    print(f"\nthe loss for a rewarded {letter}: -reward x log p({letter}), reward 1")
    print(f"  its gradient, -reward x that slope:  {numbers(gradient)}")
    print(f"  one gradient-descent step, lr {LR:g}:  {numbers(stepped - scores, 4)}")
    print(f"  chapter 2's nudge, lr {LR:g}:          {numbers(nudged - scores, 4)}")
    rewards = np.arange(len(rows)) % REWARD_EVERY  # rewards 0, 1, 0, 1, ...
    ways = {"by hand": log_prob_gradient, "measured": measured_gradient}
    if mx is not None:
        ways["with MLX"] = mlx_gradient
    gaps = [
        f"{gap_text(nudge_gap(rows, choices, rewards, way))} {name}"
        for name, way in ways.items()
    ]
    print(
        f"on the {len(rows):,} random tries, the step and the nudge differ by at most"
    )
    print("  " + ", ".join(gaps))
    print("chapter 2's nudge was exactly lr x reward x the gradient of log p")


def main() -> None:
    """Parse the flags and print the three parts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seeds the random rows; same seed, same output",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = rng.uniform(-SCORE_RANGE, SCORE_RANGE, size=(RANDOM_ROWS, len(LETTERS)))
    choices = rng.integers(len(LETTERS), size=RANDOM_ROWS)
    even = np.zeros(len(LETTERS))  # every row starts here: 25% each
    show_log_probabilities()
    show_three_ways(even)
    show_random_rows(rows, choices)
    show_chapter2(even, rows, choices)


if __name__ == "__main__":
    main()
