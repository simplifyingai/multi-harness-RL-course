# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Redraw a run's curves in the terminal, from its metrics.jsonl alone.

reward_lab.py appends one JSON line per step to metrics.jsonl: how many
rollouts there were and how many were right, the average reward, and how
many replies there were, how many tokens they held and how many were cut
off. This script reads nothing else, so any run can be redrawn later, or on
another machine, from that one file. It draws three text charts with the
standard library only (no plotting library):

* the share right and the reward, one letter each (c, r);
* tokens per reply, up to the output cap;
* calls per rollout, up to the budget.

Run it with ``uv run plot.py <path to metrics.jsonl>``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

WIDTH = 50  # columns of a chart; each column covers a run of steps
HEIGHT = 10  # rows above the zero line
WINDOW_SHARE = 10  # each point averages the last tenth of the run
# Two numbers copied on purpose, so this script stands alone; a test keeps
# them equal to environment.OUTPUT_CAP and environment.BUDGET.
OUTPUT_CAP = 120  # tokens: a longer reply is cut off
BUDGET = 8  # calls per rollout
REWARD_TOP = 1.2  # the top of the first chart: above any reward a rule pays
CORRECT_MARK = "c"
REWARD_MARK = "r"
TOKENS_MARK = "t"
CALLS_MARK = "n"
CAP_MARK = "-"
OVERLAP = "*"  # two marks in the same place
EMPTY = " "
LABEL_WIDTH = 5  # "1.2  "


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
        rollouts: How many rollouts the step ran.
        correct: How many of them were right.
        scored: How many were paid a reward.
        mean_reward: Their average reward, or None when none was paid.
        replies: How many replies (calls) they made.
        reply_tokens: How many tokens those replies held.
        cut_off: How many replies were cut off.
    """

    step: int
    rollouts: int
    correct: int
    scored: int
    mean_reward: float | None
    replies: int
    reply_tokens: int
    cut_off: int


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
    foreign = f"{path} is not a metrics.jsonl from reward_lab.py"
    records = []
    # The field names are reward_lab.py's step_metrics keys, written out here
    # because this script imports nothing; a test redraws a real run's file.
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            reward = entry["mean_reward"]
            records.append(
                StepRecord(
                    step=int(entry["step"]),
                    rollouts=int(entry["rollouts"]),
                    correct=int(entry["correct"]),
                    scored=int(entry["scored"]),
                    mean_reward=None if reward is None else float(reward),
                    replies=int(entry["replies"]),
                    reply_tokens=int(entry["reply_tokens"]),
                    cut_off=int(entry["cut_off_replies"]),
                )
            )
    except OSError as error:
        raise PlotError(f"{path} could not be read") from error
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PlotError(foreign) from error
    if not records:
        raise PlotError(f"{path} has no steps")
    for record in records:
        # Every step ran rollouts, and no count can exceed what it counts.
        counts_fit = (
            0 <= record.correct <= record.rollouts
            and 0 <= record.scored <= record.rollouts
            and 0 <= record.cut_off <= record.replies
        )
        if record.rollouts < 1 or not counts_fit:
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


def share_correct(steps: Sequence[StepRecord]) -> float:
    """Return the share of rollouts that were right in some steps.

    Args:
        steps: The steps of one column.

    Returns:
        A share from 0 to 1.
    """
    return sum(s.correct for s in steps) / sum(s.rollouts for s in steps)


def average_reward(steps: Sequence[StepRecord]) -> float | None:
    """Return the average reward of the rollouts paid in some steps.

    Args:
        steps: The steps of one column.

    Returns:
        The average, or None when no rollout was paid.
    """
    paid = [(s.mean_reward, s.scored) for s in steps if s.mean_reward is not None]
    scored = sum(count for _, count in paid)
    if not scored:
        return None
    return sum(reward * count for reward, count in paid) / scored


def tokens_per_reply(steps: Sequence[StepRecord]) -> float | None:
    """Return the average reply length in some steps (cut off: the cap).

    Args:
        steps: The steps of one column.

    Returns:
        Tokens per reply, or None when no reply was made.
    """
    replies = sum(s.replies for s in steps)
    return sum(s.reply_tokens for s in steps) / replies if replies else None


def calls_per_rollout(steps: Sequence[StepRecord]) -> float:
    """Return the average number of calls per rollout in some steps.

    Args:
        steps: The steps of one column.

    Returns:
        Calls per rollout.
    """
    return sum(s.replies for s in steps) / sum(s.rollouts for s in steps)


def row_of(value: float, top: float) -> int:
    """Turn a value into a chart row, 0 at the bottom and HEIGHT at the top.

    Args:
        value: The value.
        top: The value drawn on the top row.

    Returns:
        The row.
    """
    return round(HEIGHT * min(max(value / top, 0.0), 1.0))


def blank(width: int) -> list[list[str]]:
    """Make an empty chart grid.

    Args:
        width: How many columns.

    Returns:
        HEIGHT + 1 rows of spaces, bottom row first.
    """
    return [[EMPTY] * width for _ in range(HEIGHT + 1)]


def place(grid: list[list[str]], x: int, row: int, mark: str) -> None:
    """Put a mark in the grid, or the overlap mark if one is there already.

    Args:
        grid: The chart grid.
        x: The column.
        row: The row.
        mark: The mark.
    """
    grid[row][x] = mark if grid[row][x] in (EMPTY, CAP_MARK) else OVERLAP


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


def scale_labels(top: float, written: Callable[[float], str]) -> dict[int, str]:
    """Label the top, middle and bottom rows of a chart.

    Args:
        top: The value of the top row.
        written: How to write a value.

    Returns:
        The three labels by row.
    """
    return {HEIGHT: written(top), HEIGHT // 2: written(top / 2), 0: written(0)}


def correct_chart(cols: Sequence[Sequence[StepRecord]], last_step: int) -> list[str]:
    """Draw the share right and the average reward per column.

    Args:
        cols: The steps of each column.
        last_step: The run's last step.

    Returns:
        The chart's lines.
    """
    grid = blank(len(cols))
    for x, steps in enumerate(cols):
        place(grid, x, row_of(share_correct(steps), REWARD_TOP), CORRECT_MARK)
        reward = average_reward(steps)
        if reward is not None:
            place(grid, x, row_of(reward, REWARD_TOP), REWARD_MARK)
    title = (
        f"share right ({CORRECT_MARK}, 1 = every rollout) and reward"
        f" ({REWARD_MARK}), {OVERLAP} where they meet"
    )
    return draw(title, grid, scale_labels(REWARD_TOP, lambda v: f"{v:.1f}"), last_step)


def tokens_chart(cols: Sequence[Sequence[StepRecord]], last_step: int) -> list[str]:
    """Draw tokens per reply per column, with the output cap as the top row.

    Args:
        cols: The steps of each column.
        last_step: The run's last step.

    Returns:
        The chart's lines.
    """
    grid = blank(len(cols))
    grid[HEIGHT] = [CAP_MARK] * len(cols)
    for x, steps in enumerate(cols):
        tokens = tokens_per_reply(steps)
        if tokens is not None:
            place(grid, x, row_of(tokens, OUTPUT_CAP), TOKENS_MARK)
    title = (
        f"tokens per reply ({TOKENS_MARK}); the top row ({CAP_MARK}) is the"
        f" output cap, {OUTPUT_CAP}"
    )
    return draw(title, grid, scale_labels(OUTPUT_CAP, lambda v: f"{v:.0f}"), last_step)


def calls_chart(cols: Sequence[Sequence[StepRecord]], last_step: int) -> list[str]:
    """Draw calls per rollout per column, with the budget as the top row.

    Args:
        cols: The steps of each column.
        last_step: The run's last step.

    Returns:
        The chart's lines.
    """
    grid = blank(len(cols))
    for x, steps in enumerate(cols):
        place(grid, x, row_of(calls_per_rollout(steps), BUDGET), CALLS_MARK)
    title = f"calls per rollout ({CALLS_MARK}); the top row is the budget, {BUDGET}"
    return draw(title, grid, scale_labels(BUDGET, lambda v: f"{v:.0f}"), last_step)


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
    """Read the log and print the three charts.

    Raises:
        SystemExit: If the file is missing or not this chapter's format.
    """
    args = parse_args()
    try:
        records = read_metrics(args.metrics)
    except PlotError as error:
        raise SystemExit(f"plot.py: {error}") from error
    cols = columns(records)
    last_step = records[-1].step
    rollouts = sum(record.rollouts for record in records)
    print(
        f"{len(records)} steps, {rollouts:,} rollouts; each point is the average"
        f" of the {window_size(len(records))} steps up to it"
    )
    for chart in (correct_chart, tokens_chart, calls_chart):
        print()
        print("\n".join(chart(cols, last_step)))


if __name__ == "__main__":
    main()
