"""The arithmetic of reading a score: pass@1, its noise, gained, lost and p.

Chapter 14's ``stats.py`` (pass@1, the error bar, gained and lost, the exact
p-value), with what chapter 24 adds to read 24 tasks per harness honestly:

- **Standard error**: sqrt(p x (1 - p) / n), in percentage points: how far
  a score typically moves by luck alone if another n tasks like these were
  asked. At 24/24 it says 0, as if 100% were certain.
- **Wilson interval**: a 95% range for the true rate that stays sensible at
  0% and 100% (Wilson, 1927). At 24/24 its lower end is 86%, not 100%.
- **Two identical models**: both solve each task with the same chance, so
  any gap between their scores is luck. ``gap_chances`` gives the exact
  chance of each gap; ``simulate_gaps`` draws the same thing with a seeded
  random generator, so a reader can watch it happen.
- **Gained, lost and the exact p-value**: two runs compared task by task.
  If they were equally good, each task on which they disagree would be a
  coin flip between gained and lost; p is the chance of a split at least
  this uneven (McNemar's exact test, two-sided).

This module needs only the standard library.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

PERCENT = 100
HALF = 2  # the exact test is two-sided
# 1.96: the normal distribution's 97.5th percentile, so a 95% interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Comparison:
    """Two runs on the same cells, compared cell by cell.

    Attributes:
        gained: Cells right in the second run and wrong in the first.
        lost: Cells right in the first run and wrong in the second.
        both: Cells right in both.
        neither: Cells wrong in both.
    """

    gained: tuple[str, ...]
    lost: tuple[str, ...]
    both: int
    neither: int

    @property
    def flipped(self) -> int:
        """Cells whose verdict changed, either way.

        Returns:
            ``gained + lost``.
        """
        return len(self.gained) + len(self.lost)

    @property
    def p_value(self) -> float:
        """The exact two-sided p-value for this split of gained and lost.

        Returns:
            The p-value (1 when nothing changed).
        """
        return exact_p(len(self.gained), len(self.lost))


def exact_p(gained: int, lost: int) -> float:
    """The exact two-sided McNemar p-value for cells gained and lost.

    Args:
        gained: Cells right after but not before.
        lost: Cells right before but not after.

    Returns:
        The p-value (1 when nothing changed).
    """
    changed = gained + lost
    if not changed:
        return 1.0
    tail = sum(math.comb(changed, k) for k in range(min(gained, lost) + 1))
    return min(1.0, HALF * tail / HALF**changed)


def pass_at_1(correct: int, cells: int) -> float:
    """The share of cells right, as a percentage.

    Args:
        correct: Cells answered right.
        cells: Cells run.

    Returns:
        The percentage (0 when nothing ran).
    """
    return PERCENT * correct / cells if cells else 0.0


def standard_error(correct: int, cells: int) -> float:
    """One binomial standard error of a pass rate, in percentage points.

    Args:
        correct: Cells answered right.
        cells: Cells run.

    Returns:
        ``100 x sqrt(p x (1 - p) / n)``; 0 when nothing ran.
    """
    if not cells:
        return 0.0
    share = correct / cells
    return PERCENT * math.sqrt(share * (1 - share) / cells)


def wilson_interval(correct: int, cells: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for a pass rate, in percent.

    Unlike "score plus or minus two standard errors", it never leaves 0-100%
    and never shrinks to a point at 0% or 100%.

    Args:
        correct: Cells answered right.
        cells: Cells run.
        z: How many standard deviations wide (1.96 for 95%).

    Returns:
        The low and high ends, in percent; (0, 100) when nothing ran.
    """
    if not cells:
        return 0.0, float(PERCENT)
    share = correct / cells
    spread = z * z / cells
    centre = (share + spread / 2) / (1 + spread)
    half = (
        z * math.sqrt(share * (1 - share) / cells + spread / (4 * cells)) / (1 + spread)
    )
    # At 0 and at every cell solved, the matching end is exactly 0 or 100%.
    low = 0.0 if correct == 0 else max(0.0, centre - half)
    high = 1.0 if correct == cells else min(1.0, centre + half)
    return PERCENT * low, PERCENT * high


def binomial(cells: int, rate: float) -> list[float]:
    """The chance of each score from 0 to ``cells``, when each cell is a coin.

    Args:
        cells: Cells run.
        rate: The chance of solving each one (0 to 1).

    Returns:
        ``chances[k]`` = the chance of exactly k solved.
    """
    return [
        math.comb(cells, k) * rate**k * (1 - rate) ** (cells - k)
        for k in range(cells + 1)
    ]


def gap_chances(cells: int, rate: float) -> list[float]:
    """The exact chance of each gap between two identical models' scores.

    Args:
        cells: Cells each model runs.
        rate: The chance both have of solving each cell.

    Returns:
        ``chances[g]`` = the chance their scores differ by exactly g cells.
    """
    chances = binomial(cells, rate)
    gaps = [0.0] * (cells + 1)
    for first, p_first in enumerate(chances):
        for second, p_second in enumerate(chances):
            gaps[abs(first - second)] += p_first * p_second
    return gaps


def simulate_gaps(cells: int, rate: float, pairs: int, seed: int) -> list[int]:
    """Score two identical models many times and count each gap.

    Each model solves each cell with chance ``rate``, independently: the
    luck of which tasks were picked, and of small prompt changes (the date,
    the folder) that send greedy decoding down another path.

    Args:
        cells: Cells each model runs.
        rate: The chance both have of solving each cell.
        pairs: How many pairs of runs to draw.
        seed: The random generator's seed, so the draw can be repeated.

    Returns:
        ``counts[g]`` = how many pairs differed by exactly g cells.
    """
    rng = random.Random(seed)
    counts = [0] * (cells + 1)
    for _ in range(pairs):
        first = sum(rng.random() < rate for _ in range(cells))
        second = sum(rng.random() < rate for _ in range(cells))
        counts[abs(first - second)] += 1
    return counts


def compare(before: Mapping[str, bool], after: Mapping[str, bool]) -> Comparison:
    """Compare two runs cell by cell.

    Args:
        before: Each cell's verdict in the first run.
        after: Each cell's verdict in the second run (the same cells).

    Returns:
        What was gained, lost, kept and missed.

    Raises:
        ValueError: If the two runs ran different cells.
    """
    if set(before) != set(after):
        raise ValueError("the two runs ran different cells")
    ids = sorted(before)
    return Comparison(
        gained=tuple(i for i in ids if after[i] and not before[i]),
        lost=tuple(i for i in ids if before[i] and not after[i]),
        both=sum(before[i] and after[i] for i in ids),
        neither=sum(not before[i] and not after[i] for i in ids),
    )
