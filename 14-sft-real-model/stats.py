"""The arithmetic of reading a score: pass@1, its error bar, gained, lost and p.

- **pass@1**: the share of questions answered right on the one greedy try.
- **Error bar**: plus or minus one standard error, sqrt(p x (1 - p) / n),
  in percentage points: how far the score would typically move by luck
  alone if another n questions like these were asked.
- **Gained, lost and the exact p-value**: two runs compared question by
  question. If they were equally good, each question on which they disagree
  would be a coin flip between gained and lost; p is the chance of a split
  at least this uneven (McNemar's exact test, two-sided, as chapter 8's
  ``real_sft.py`` and chapter 11's ``compare.py`` compute it).

This module needs only the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

PERCENT = 100
HALF = 2  # the exact test is two-sided


@dataclass(frozen=True)
class Comparison:
    """Two runs on the same questions, compared question by question.

    Attributes:
        gained: Questions right in the second run and wrong in the first.
        lost: Questions right in the first run and wrong in the second.
        both: Questions right in both.
        neither: Questions wrong in both.
    """

    gained: tuple[str, ...]
    lost: tuple[str, ...]
    both: int
    neither: int

    @property
    def p_value(self) -> float:
        """The exact two-sided p-value for this split of gained and lost.

        Returns:
            The p-value (1 when nothing changed).
        """
        return exact_p(len(self.gained), len(self.lost))


def exact_p(gained: int, lost: int) -> float:
    """The exact two-sided McNemar p-value for questions gained and lost.

    Args:
        gained: Questions right after but not before.
        lost: Questions right before but not after.

    Returns:
        The p-value (1 when nothing changed).
    """
    changed = gained + lost
    if not changed:
        return 1.0
    tail = sum(math.comb(changed, k) for k in range(min(gained, lost) + 1))
    return min(1.0, HALF * tail / HALF**changed)


def pass_at_1(correct: int, questions: int) -> float:
    """The share of questions right, as a percentage.

    Args:
        correct: Questions answered right.
        questions: Questions asked.

    Returns:
        The percentage (0 when nothing was asked).
    """
    return PERCENT * correct / questions if questions else 0.0


def error_bar(correct: int, questions: int) -> float:
    """One standard error of a pass rate, in percentage points.

    Args:
        correct: Questions answered right.
        questions: Questions asked.

    Returns:
        ``100 x sqrt(p x (1 - p) / n)``; 0 when nothing was asked.
    """
    if not questions:
        return 0.0
    share = correct / questions
    return PERCENT * math.sqrt(share * (1 - share) / questions)


def compare(before: Mapping[str, bool], after: Mapping[str, bool]) -> Comparison:
    """Compare two runs question by question.

    Args:
        before: Each question's verdict in the first run.
        after: Each question's verdict in the second run (the same questions).

    Returns:
        What was gained, lost, kept and missed.

    Raises:
        ValueError: If the two runs answered different questions.
    """
    if set(before) != set(after):
        raise ValueError("the two runs answered different questions")
    ids = list(before)
    return Comparison(
        gained=tuple(i for i in ids if after[i] and not before[i]),
        lost=tuple(i for i in ids if before[i] and not after[i]),
        both=sum(before[i] and after[i] for i in ids),
        neither=sum(not before[i] and not after[i] for i in ids),
    )


def median(values: list[int]) -> float:
    """The middle value of a list (the mean of the middle two if it is even).

    Args:
        values: Numbers.

    Returns:
        The median, or 0 for an empty list.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
