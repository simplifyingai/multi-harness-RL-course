# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Redraw a run's curves in the terminal, from its metrics.jsonl alone.

sft_toy.py appends one JSON line per SFT step (its loss per call) and, with
``--then-rl``, one per GRPO step of each start (how many rollouts were right,
how many groups had contrast). This script reads nothing else, so any run can
be redrawn later, or on another machine, from that one file. It draws text
charts with the standard library only (no plotting library):

* SFT: the loss per call at every step;
* RL, if the run had it: the share of rollouts right, and the share of
  groups with signal, from scratch (s) and from the imitator (i).

Run it with ``uv run plot.py <path to metrics.jsonl>``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

WIDTH = 50  # columns of an RL chart; each column covers a run of steps
HEIGHT = 10  # rows above the zero line
WINDOW_SHARE = 10  # each RL point averages the last tenth of the run
LOSS_FLOOR_TOP = 2.0  # the loss chart's top: at least the untrained loss, ln 7
SFT = "sft"  # the stage names sft_toy.py writes
GRPO = "grpo"
SCRATCH = "scratch"
IMITATOR = "imitator"
STARTS = (SCRATCH, IMITATOR)
MARKS = {SCRATCH: "s", IMITATOR: "i"}
LOSS_MARK = "l"
OVERLAP = "*"  # two marks in the same place
EMPTY = " "
LABEL_WIDTH = 5  # "100% "
PERCENT = 100
LOSS_DECIMALS = 2


class PlotError(Exception):
    """Raised when the file is not a metrics.jsonl this chapter wrote.

    It does not derive from the chapter's CourseError, because this script
    imports nothing from the folder.
    """


@dataclass(frozen=True)
class SftRecord:
    """What plot.py reads from one SFT line.

    Attributes:
        step: The step's number.
        loss: The loss per call.
    """

    step: int
    loss: float


@dataclass(frozen=True)
class RlRecord:
    """What plot.py reads from one GRPO line.

    Attributes:
        start: ``"scratch"`` or ``"imitator"``.
        step: The step's number.
        right: How many rollouts were right.
        rollouts: How many rollouts the step ran.
        signal: How many groups had contrast.
        groups: How many groups the step had.
    """

    start: str
    step: int
    right: int
    rollouts: int
    signal: int
    groups: int


def read_metrics(path: Path) -> tuple[list[SftRecord], dict[str, list[RlRecord]]]:
    """Read every step of a run from its metrics.jsonl.

    Args:
        path: The file.

    Returns:
        The SFT steps in order, and each start's GRPO steps in order (empty
        lists without ``--then-rl``).

    Raises:
        PlotError: If the file is missing, unreadable, empty or not this
            chapter's format.
    """
    if not path.is_file():
        raise PlotError(f"no such file: {path}")
    foreign = f"{path} is not a metrics.jsonl from sft_toy.py"
    sft: list[SftRecord] = []
    rl: dict[str, list[RlRecord]] = {start: [] for start in STARTS}
    # The field names are sft_toy.py's sft_metrics and rl_metrics keys,
    # written out here because this script imports nothing.
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["stage"] == SFT:
                loss = float(entry["loss_per_call"])
                if not math.isfinite(loss):  # JSON allows NaN and Infinity
                    raise PlotError(foreign)
                sft.append(SftRecord(int(entry["step"]), loss))
                continue
            if entry["stage"] != GRPO or entry["start"] not in rl:
                raise PlotError(foreign)
            groups = int(entry["groups"])
            rl[entry["start"]].append(
                RlRecord(
                    start=entry["start"],
                    step=int(entry["step"]),
                    right=int(entry["correct"]),
                    rollouts=int(entry["rollouts"]),
                    signal=groups - int(entry["groups_without_contrast"]),
                    groups=groups,
                )
            )
    except OSError as error:
        raise PlotError(f"{path} could not be read") from error
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PlotError(foreign) from error
    if not sft:
        raise PlotError(f"{path} has no SFT steps")
    if len(rl[SCRATCH]) != len(rl[IMITATOR]):
        raise PlotError(foreign)
    for runs in rl.values():
        for record in runs:
            counts_fit = (
                0 <= record.right <= record.rollouts
                and 0 <= record.signal <= record.groups
            )
            if record.rollouts < 1 or record.groups < 1 or not counts_fit:
                raise PlotError(foreign)
    return sft, rl


def window_size(steps: int) -> int:
    """Return how many steps each RL point averages.

    Args:
        steps: How many steps the run has.

    Returns:
        A tenth of the run, and never less than one column's worth.
    """
    return max(math.ceil(steps / WIDTH), math.ceil(steps / WINDOW_SHARE))


def columns(records: Sequence[RlRecord]) -> list[list[RlRecord]]:
    """Pick the steps each column averages: the window that ends at the column.

    Args:
        records: Every step of one start.

    Returns:
        The steps of each column, in order.
    """
    size = math.ceil(len(records) / WIDTH)
    window = window_size(len(records))
    ends = [min(end, len(records)) for end in range(size, len(records) + size, size)]
    return [list(records[max(0, end - window) : end]) for end in ends]


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
    grid[row][x] = mark if grid[row][x] == EMPTY else OVERLAP


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


def loss_chart(records: Sequence[SftRecord]) -> list[str]:
    """Draw the loss per call at every SFT step, one column per step.

    Args:
        records: Every SFT step.

    Returns:
        The chart's lines.
    """
    top = max(LOSS_FLOOR_TOP, math.ceil(max(r.loss for r in records)))
    grid = blank(len(records))
    for x, record in enumerate(records):
        place(grid, x, row_of(record.loss, top), LOSS_MARK)
    title = f"SFT loss per call ({LOSS_MARK}), one column per step"
    return draw(title, grid, scale_labels(top, lambda v: f"{v:.1f}"), records[-1].step)


def rl_chart(
    rl: dict[str, list[RlRecord]],
    title: str,
    part: Callable[[RlRecord], int],
    whole: Callable[[RlRecord], int],
) -> list[str]:
    """Draw one share per column for both starts, as a percentage.

    Args:
        rl: Each start's GRPO steps.
        title: The line above the chart.
        part: What is counted in a step (rollouts right, groups with signal).
        whole: What it is counted out of.

    Returns:
        The chart's lines.
    """
    per_start = {start: columns(rl[start]) for start in STARTS}
    grid = blank(len(per_start[SCRATCH]))
    for start, cols in per_start.items():
        for x, steps in enumerate(cols):
            value = sum(map(part, steps)) / sum(map(whole, steps))
            place(grid, x, row_of(value, 1.0), MARKS[start])
    labels = scale_labels(1.0, lambda v: f"{round(PERCENT * v)}%")
    return draw(title, grid, labels, rl[SCRATCH][-1].step)


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
    """Read the log and print its charts.

    Raises:
        SystemExit: If the file is missing or not this chapter's format.
    """
    args = parse_args()
    try:
        sft, rl = read_metrics(args.metrics)
    except PlotError as error:
        raise SystemExit(f"plot.py: {error}") from error
    print(
        f"{len(sft)} SFT steps; loss per call {sft[0].loss:.{LOSS_DECIMALS}f}"
        f" -> {sft[-1].loss:.{LOSS_DECIMALS}f}"
    )
    print()
    print("\n".join(loss_chart(sft)))
    if not rl[SCRATCH]:
        return
    steps = len(rl[SCRATCH])
    print()
    print(
        f"{steps} RL steps from each start; each point is the average of the"
        f" {window_size(steps)} steps up to it"
    )
    marks = f"{MARKS[SCRATCH]} from scratch, {MARKS[IMITATOR]} from the imitator"
    print()
    title = f"rollouts right ({marks}, {OVERLAP} where they meet)"
    print("\n".join(rl_chart(rl, title, lambda r: r.right, lambda r: r.rollouts)))
    print()
    title = f"groups with signal ({marks})"
    print("\n".join(rl_chart(rl, title, lambda r: r.signal, lambda r: r.groups)))


if __name__ == "__main__":
    main()
