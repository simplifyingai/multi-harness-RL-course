# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Redraw a run's curves in the terminal, from its metrics.jsonl alone.

grpo_clip.py appends one JSON line per step to metrics.jsonl: each group's
difficulty and four rewards, and how many of the ratios the loss saw fell
outside 0.8 to 1.2 and how many the clip switched off. This script reads
nothing else, so any run can be redrawn later, or on another machine, from
that one file. It draws two text charts with the standard library only (no
plotting library):

* the share correct, one letter per difficulty (e, m, h);
* the ratios outside 0.8 to 1.2, split into clipped and not clipped.

Run it with ``uv run plot.py <path to metrics.jsonl>``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

WIDTH = 50  # columns of a chart; each column covers a run of steps
HEIGHT = 10  # rows above the zero line
PERCENT = 100
# Two numbers copied on purpose, so this script stands alone; a test keeps
# them equal to grpo.CORRECT_REWARD and grpo.Defaults.CLIP_EPSILON.
CORRECT_REWARD = 1  # what a right answer earns
CLIP_EPSILON = 0.2  # the clip keeps the ratio within 1 - 0.2 and 1 + 0.2
LEVEL_MARKS = {"easy": "e", "medium": "m", "hard": "h"}
OVERLAP = "*"  # two or more difficulties in the same place
CLIPPED_MARK = "c"
OUTSIDE_MARK = "o"
EMPTY = " "
LABEL_WIDTH = 5  # "100% "
WINDOW_SHARE = 10  # each point averages the last tenth of the run


class PlotError(Exception):
    """Raised when the file is not a metrics.jsonl this chapter wrote.

    It does not derive from the chapter's CourseError, because this script
    imports nothing from the folder.
    """


@dataclass(frozen=True)
class StepRecord:
    """What plot.py reads from one line of metrics.jsonl.

    Attributes:
        step: The step's number.
        levels: Each group's difficulty.
        rewards: Each group's rewards.
        ratio_calls: How many ratios the loss saw in that step (every use).
        ratios_outside: How many of them fell outside 0.8 to 1.2.
        clipped_calls: How many the clip switched off.
    """

    step: int
    levels: tuple[str, ...]
    rewards: tuple[tuple[float, ...], ...]
    ratio_calls: int
    ratios_outside: int
    clipped_calls: int


def read_metrics(path: Path) -> list[StepRecord]:
    """Read every step of a run from its metrics.jsonl.

    Args:
        path: The file.

    Returns:
        One record per step, in order.

    Raises:
        PlotError: If the file is missing, unreadable, empty or not this
            chapter's format.
    """
    if not path.is_file():
        raise PlotError(f"no such file: {path}")
    foreign = f"{path} is not a metrics.jsonl from grpo_clip.py"
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            records.append(
                StepRecord(
                    step=int(entry["step"]),
                    levels=tuple(entry["group_levels"]),
                    rewards=tuple(
                        tuple(float(r) for r in group)
                        for group in entry["group_rewards"]
                    ),
                    ratio_calls=int(entry["ratio_calls"]),
                    ratios_outside=int(entry["ratios_outside"]),
                    clipped_calls=int(entry["clipped_calls"]),
                )
            )
    except OSError as error:
        raise PlotError(f"{path} could not be read") from error
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PlotError(foreign) from error
    if not records:
        raise PlotError(f"{path} has no steps")
    for record in records:
        # Every step needs one level per group and at least one reward each,
        # and no more ratios outside the range (or clipped) than ratios.
        has_groups = bool(record.rewards) and all(record.rewards)
        counts_fit = (
            0 <= record.clipped_calls <= record.ratios_outside <= record.ratio_calls
        )
        if not has_groups or len(record.levels) != len(record.rewards):
            raise PlotError(foreign)
        if not counts_fit:
            raise PlotError(foreign)
    return records


def window_size(steps: int) -> int:
    """Return how many steps each point of a chart averages.

    Args:
        steps: How many steps the run has.

    Returns:
        A tenth of the run, and never less than one column's worth.
    """
    return max(math.ceil(steps / WIDTH), math.ceil(steps / WINDOW_SHARE))


def columns(records: Sequence[StepRecord]) -> list[list[StepRecord]]:
    """Pick the steps each column averages: the window that ends at the column.

    A single step has only 8 rollouts, so one step per point would be mostly
    noise. Each column instead covers the last tenth of the run up to it.

    Args:
        records: Every step.

    Returns:
        The steps of each column, in order.
    """
    size = math.ceil(len(records) / WIDTH)
    window = window_size(len(records))
    ends = [min(end, len(records)) for end in range(size, len(records) + size, size)]
    return [list(records[max(0, end - window) : end]) for end in ends]


def share_correct(steps: Sequence[StepRecord], level: str) -> float | None:
    """Return the share of a level's rollouts that were right in some steps.

    Args:
        steps: The steps of one column.
        level: The difficulty.

    Returns:
        A share from 0 to 1, or None when the level had no group there.
    """
    rewards = [
        reward
        for step in steps
        for group_level, group in zip(step.levels, step.rewards, strict=True)
        if group_level == level
        for reward in group
    ]
    if not rewards:
        return None
    return sum(r >= CORRECT_REWARD for r in rewards) / len(rewards)


def ratio_shares(steps: Sequence[StepRecord]) -> tuple[float, float] | None:
    """Return the share of ratios clipped, and outside the range but not clipped.

    Args:
        steps: The steps of one column.

    Returns:
        The two shares, each from 0 to 1, or None when those steps learned
        from nothing (no ratios at all).
    """
    total = sum(step.ratio_calls for step in steps)
    if not total:
        return None
    clipped = sum(step.clipped_calls for step in steps)
    outside = sum(step.ratios_outside for step in steps)
    return clipped / total, (outside - clipped) / total


def row_of(value: float, top: float) -> int:
    """Turn a value into a chart row, 0 at the bottom and HEIGHT at the top.

    Args:
        value: The value.
        top: The value drawn on the top row.

    Returns:
        The row.
    """
    return round(HEIGHT * min(max(value / top, 0.0), 1.0))


def draw(
    title: str, grid: list[list[str]], labels: dict[int, str], last_step: int
) -> list[str]:
    """Lay out a chart: a title, labelled rows and a step axis.

    Args:
        title: The line above the chart.
        grid: One list of characters per row, bottom row first.
        labels: Text for some rows' left edge, by row.
        last_step: The step the right edge stands for.

    Returns:
        The chart's lines.
    """
    width = len(grid[0])
    lines = [title]
    for row in range(HEIGHT, -1, -1):
        label = labels.get(row, "")
        lines.append(f"{label:>{LABEL_WIDTH}}|{''.join(grid[row])}".rstrip())
    lines.append(f"{'':>{LABEL_WIDTH}}+{'-' * width}")
    # The first step at the left edge, the last at the right (one column
    # holds both when the run is that short).
    axis = f"1{str(last_step):>{width - 1}}" if width > 1 else str(last_step)
    lines.append(f"{'':>{LABEL_WIDTH}} {axis}  step")
    return lines


def blank(width: int) -> list[list[str]]:
    """Make an empty chart grid.

    Args:
        width: How many columns.

    Returns:
        HEIGHT + 1 rows of spaces, bottom row first.
    """
    return [[EMPTY] * width for _ in range(HEIGHT + 1)]


def percent_labels() -> dict[int, str]:
    """Label the top, middle and bottom rows of a percentage chart.

    Returns:
        ``"100%"``, ``"50%"`` and ``"0%"`` by row.
    """
    return {HEIGHT: f"{PERCENT}%", HEIGHT // 2: f"{PERCENT // 2}%", 0: "0%"}


def correct_chart(cols: Sequence[Sequence[StepRecord]], last_step: int) -> list[str]:
    """Draw the share correct per column, one letter per difficulty.

    Args:
        cols: The steps of each column.
        last_step: The run's last step.

    Returns:
        The chart's lines.
    """
    grid = blank(len(cols))
    for x, steps in enumerate(cols):
        for level, mark in LEVEL_MARKS.items():
            share = share_correct(steps, level)
            if share is None:
                continue
            row = row_of(share, 1.0)
            grid[row][x] = mark if grid[row][x] == EMPTY else OVERLAP
    marks = ", ".join(f"{mark} {level}" for level, mark in LEVEL_MARKS.items())
    title = f"correct, by difficulty ({marks}, {OVERLAP} where they meet)"
    return draw(title, grid, percent_labels(), last_step)


def ratio_chart(cols: Sequence[Sequence[StepRecord]], last_step: int) -> list[str]:
    """Draw the ratios outside the clip's range per column, clipped ones below.

    Args:
        cols: The steps of each column.
        last_step: The run's last step.

    Returns:
        The chart's lines. A column whose steps learned from nothing stays
        empty.
    """
    grid = blank(len(cols))
    for x, steps in enumerate(cols):
        shares = ratio_shares(steps)
        if shares is None:
            continue
        clipped, rest = shares
        clipped_top = row_of(clipped, 1.0)
        outside_top = row_of(clipped + rest, 1.0)
        for row in range(1, clipped_top + 1):
            grid[row][x] = CLIPPED_MARK
        for row in range(clipped_top + 1, outside_top + 1):
            grid[row][x] = OUTSIDE_MARK
    low, high = 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON
    title = (
        f"ratios outside {low:g}-{high:g} ({CLIPPED_MARK} clipped,"
        f" {OUTSIDE_MARK} not clipped), as a share of the ratios the loss saw"
    )
    return draw(title, grid, percent_labels(), last_step)


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The path to the metrics file.

    Raises:
        SystemExit: If no path is given (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", type=Path, help="a metrics.jsonl to redraw")
    return parser.parse_args()


def main() -> None:
    """Read the log and print the two charts."""
    args = parse_args()
    try:
        records = read_metrics(args.metrics)
    except PlotError as error:
        raise SystemExit(f"plot.py: {error}") from error
    cols = columns(records)
    last_step = records[-1].step
    ratios = sum(record.ratio_calls for record in records)
    print(
        f"{len(records)} steps, {ratios:,} ratios; each point is the average of"
        f" the {window_size(len(records))} steps up to it"
    )
    for chart in (correct_chart, ratio_chart):
        print()
        print("\n".join(chart(cols, last_step)))


if __name__ == "__main__":
    main()
