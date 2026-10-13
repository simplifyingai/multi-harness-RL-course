"""The arithmetic of reading a score: pass@1, its noise, gained, lost and p.

Chapter 24's ``stats.py``, cut to what chapter 28 reads the final table with:

- **Wilson interval**: a 95% range for the true rate that stays sensible at
  0% and 100% (Wilson, 1927). At 24/24 its lower end is 86%, not 100%.
- **Gained, lost and the exact p-value**: two runs compared task by task.
  If they were equally good, each task on which they disagree would be a
  coin flip between gained and lost; p is the chance of a split at least
  this uneven (McNemar's exact test, two-sided). The same coin-flip test
  reads "fewer calls or more" on the cells two checkpoints both solved.

This module needs only the standard library.
"""

from __future__ import annotations

import math
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
