# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Read a finished run honestly: the final table, its noise, its cost.

``capstone.sh`` runs the project's whole pipeline; this script reads what it
left in ``runs/<name>/``. With no arguments it reads the project's own
recorded run (``data/project_run/``, cut from its ``runs/main/``), so it
runs before any model does:

1. **the final table**: pass@1 per harness with a Wilson 95% interval, and
   the plain average over harnesses;
2. **calls per solve**: model calls on solved cells, per harness and
   averaged, what each stage changed, and the same thing read pairwise on
   the cells two checkpoints both solved (fewer, more, the exact p);
3. **every pair of checkpoints, cell by cell**: gained, lost and the exact
   McNemar p (chapter 24), and where flips cancel out (churn);
4. **the harness gap**: two harnesses on the same tasks, paired the same way;
5. **memory**: MLX's own peak from the trainers' metrics, and the
   processes' footprint when ``capstone.sh`` sampled it (``memory.tsv``);
6. **wall time**: the stage clock (``stages.txt``) and the records' own;
7. **the network check** (``network_check.txt``);
8. **the reading**, worded from the tables above, never typed by hand;
9. **the article's own numbers**, in a box of their own.

``--run-dir`` reads another run, such as the one ``capstone.sh`` made.
Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from paths import HERE, short_path
from stats import (
    PERCENT,
    Comparison,
    compare,
    exact_p,
    pass_at_1,
    wilson_interval,
)

DESCRIPTION = "Read a finished run honestly: the final table, its noise, its cost."
PROJECT_RUN = HERE / "data" / "project_run"
PROJECT_RUN_NAME = "the project's run of 2026-10-06"
# Where each bundled file came from, printed only for the bundled run.
PROJECT_RUN_NOTES = (
    "eval_*.jsonl, sft/ and grpo/metrics.jsonl and network_check.txt are cut "
    "from the project's runs/main/; stages.txt is the stage banners of its "
    "runs/logs/session_main.log, whose GRPO stage is attempt 1 (stopped by "
    "hand after step 8). runs/main/grpo is attempt 2, so its clock is "
    "grpo/metrics.jsonl's own. network_check.txt is one look, at 08:42: by the "
    "project's file times, during RL attempt 1's evaluation, not the final one."
)
CHECKPOINTS = ("base", "sft", "rl")  # mhrl/cli.py CliDefaults.CHECKPOINTS
HARNESS_ORDER = ("mini-swe-agent", "claude-code", "pi")
TRAIN_HARNESSES = "mini-swe-agent,claude-code"  # train_from_scratch.sh's default
RECORDS = "eval_{label}.jsonl"
PI_FIXED = "eval_{label}-pi-fixed.jsonl"
PI = "pi"
STAGES_FILE = "stages.txt"
MEMORY_FILE = "memory.tsv"
ROLE_ORDER = ("engine", "teacher", "proxy", "stage")  # memory.tsv's roles
ROLE_COLUMNS = ("time", "stage", "role", "pid", "mb")  # memory.tsv's header
NETWORK_FILE = "network_check.txt"
SFT_METRICS = Path("sft") / "metrics.jsonl"
GRPO_METRICS = Path("grpo") / "metrics.jsonl"
BANNER = "==="
CLOCK_FORMATS = ("%Y-%m-%d %H:%M:%S", "%H:%M:%S")
SIGNIFICANT = 0.05
CHURN_SHARE = 3  # churn: flips that cancel out are at least 1/3 of the cells
SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 24 * 60 * 60
DONE_STAGE = "done"  # capstone.sh's last banner: it closes the clock
UNDATED_YEAR = 1900  # what strptime gives a clock printed without a date
MB_PER_GB = 1024
RULE = "━━"
WIDTH = 78  # the boxed note's width
LABEL = 12  # the checkpoint column
CELL = 21  # one harness's column in the final table: "24/24 100% (86-100)"
PAIR = 12  # "base -> sft"
NAME = 15  # "mini-swe-agent" and a space
STAGE = 58  # a stage banner's column in the wall-time table
SOURCE = 6  # "sft", "grpo" in the memory and clock lines
STAGE_KEY = 8  # "1/5", "results" in the footprint lines
EVAL_LABEL = 5  # "base", "sft", "rl" in the clock lines
NOTE_INDENT = "              "  # under the header's values
# The article's own figures (Hugging Face and Liquid AI, "The ultimate guide
# to multi-harness RL", 2026), checked sentence by sentence against the
# article on 2026-10-13 (https://fineenvs-multi-harness-rl.hf.space/, the
# Space at https://huggingface.co/spaces/FineEnvs/multi-harness-rl). Its
# tasks, harnesses and hardware are not ours: shown in a box of their own,
# never set in a table beside ours.
ARTICLE_TITLE = "the article's own numbers (Hugging Face + Liquid AI)"
ARTICLE_SOURCE = 'from "The ultimate guide to multi-harness RL" (2026), its words:'
ARTICLE_LINES = (
    '"Trained across all four, it went from solving 42% of held-out tasks to '
    "54%, averaged over the four harnesses, and used 31% fewer tool calls on "
    'the tasks it already solved."',
    '"Two H100s per run on the Hugging Face cluster, and one E2B sandbox per '
    'rollout with 1 CPU and 4 GB of memory."',
    '"Training took about 32 hours for the OpenCode-only run and 46 hours for '
    'the multi-harness run, not counting evaluation, queueing and restarts."',
)


@dataclass(frozen=True)
class Cell:
    """One (task, harness) rollout of an evaluation.

    Attributes:
        task_id: The held-out task.
        harness: The harness it ran in.
        correct: Whether the grader accepted the answer.
        n_calls: Model calls the proxy counted.
        seconds: The rollout's own wall time.
    """

    task_id: str
    harness: str
    correct: bool
    n_calls: int
    seconds: float


@dataclass(frozen=True)
class Checkpoint:
    """One evaluated checkpoint: its label and its cells.

    Attributes:
        label: ``base``, ``sft`` or ``rl``.
        cells: Each cell, keyed by (harness, task id).
    """

    label: str
    cells: dict[tuple[str, str], Cell]

    def harness_cells(self, harness: str) -> list[Cell]:
        """The cells of one harness, in task order.

        Args:
            harness: The harness.

        Returns:
            Its cells.
        """
        return [self.cells[key] for key in sorted(self.cells) if key[0] == harness]

    def verdicts(self, harness: str) -> dict[str, bool]:
        """Each task's verdict in one harness.

        Args:
            harness: The harness.

        Returns:
            ``{task_id: correct}``.
        """
        return {cell.task_id: cell.correct for cell in self.harness_cells(harness)}


@dataclass(frozen=True)
class Run:
    """Everything read from one run folder.

    Attributes:
        folder: The run folder.
        checkpoints: The evaluated checkpoints, in pipeline order.
        missing: Checkpoints with no records yet.
        others: Other evaluations in the folder, not in the table.
        notes: What was substituted or skipped while reading.
    """

    folder: Path
    checkpoints: tuple[Checkpoint, ...]
    missing: tuple[str, ...]
    others: tuple[str, ...]
    notes: tuple[str, ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file.

    Args:
        path: The file.

    Returns:
        One dictionary per non-empty line ([] when the file is missing).

    Raises:
        ValueError: If a line is not JSON (the message names the line).
    """
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{short_path(path)}:{number}: {error}") from error
    return rows


def cells_from(path: Path) -> tuple[dict[tuple[str, str], Cell], int]:
    """Read one evaluation file into cells.

    A cell recorded twice (a resumed run that appended again) keeps its last
    record, and the count of such repeats is returned, so it gets printed.

    Args:
        path: ``eval_<label>.jsonl``.

    Returns:
        Its cells, keyed by (harness, task id), and how many rows repeated
        a cell already read.

    Raises:
        ValueError: If a row is not JSON or lacks a field (the message names
            the file and the row), such as a last line still being written.
    """
    cells: dict[tuple[str, str], Cell] = {}
    repeats = 0
    for number, row in enumerate(read_jsonl(path), 1):
        try:
            cell = Cell(
                task_id=row["task_id"],
                harness=row["harness"],
                correct=bool(row["correct"]),
                n_calls=int(row["n_calls"]),
                seconds=float(row.get("seconds") or 0.0),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{short_path(path)}: row {number}: {error!r}") from error
        key = (cell.harness, cell.task_id)
        repeats += key in cells
        cells[key] = cell
    return cells, repeats


def read_run(folder: Path) -> Run:
    """Read every evaluation in a run folder.

    Pi's cells come from ``eval_<label>-pi-fixed.jsonl`` when that file
    exists: the project re-ran Pi at every checkpoint after chapter 19's
    sandbox fix, and its report takes Pi from the re-run. That choice is
    kept in ``notes``, so it is printed, not hidden.

    Args:
        folder: ``runs/<name>``.

    Returns:
        What it holds.

    Raises:
        ValueError: If a records file holds a broken row (see ``cells_from``).
    """
    checkpoints, missing, fixed, notes = [], [], [], []
    for label in CHECKPOINTS:
        cells, repeats = cells_from(folder / RECORDS.format(label=label))
        rerun, rerun_repeats = cells_from(folder / PI_FIXED.format(label=label))
        rerun = {key: cell for key, cell in rerun.items() if key[0] == PI}
        if repeats + rerun_repeats:
            notes.append(
                f"{label}: {repeats + rerun_repeats} row(s) repeat a cell; "
                "the last record of each was kept"
            )
        if not cells:
            missing.append(label)
            if rerun:
                notes.append(f"{label}: only a pi re-run exists, so it is left out")
            continue
        if rerun:
            cells = {key: cell for key, cell in cells.items() if key[0] != PI}
            cells.update(rerun)
            fixed.append(label)
        checkpoints.append(Checkpoint(label, cells))
    known = {RECORDS.format(label=label) for label in CHECKPOINTS}
    others = sorted(
        path.stem.removeprefix("eval_")
        for path in folder.glob("eval_*.jsonl")
        if path.name not in known and not path.stem.endswith("-pi-fixed")
    )
    if fixed:
        notes.insert(
            0,
            f"pi for {', '.join(fixed)} from {PI_FIXED.format(label='<label>')}: "
            "the re-run after chapter 19's sandbox fix (as the project's report)",
        )
    return Run(folder, tuple(checkpoints), tuple(missing), tuple(others), tuple(notes))


def harnesses_of(run: Run) -> list[str]:
    """The harnesses the run evaluated, in the project's order.

    Args:
        run: The run.

    Returns:
        Harness names.
    """
    seen = {harness for c in run.checkpoints for harness, _ in c.cells}
    known = [h for h in HARNESS_ORDER if h in seen]
    return known + sorted(seen - set(known))


def task_sizes(run: Run, harnesses: list[str]) -> list[int]:
    """How many cells each harness ran, over every checkpoint.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        The distinct sizes, smallest first (one size in a complete run).
    """
    return sorted(
        {len(c.harness_cells(h)) for c in run.checkpoints for h in harnesses} - {0}
    )


def percent(value: float) -> str:
    """Write a percentage the way the project's tables do (no decimals).

    Args:
        value: A percentage.

    Returns:
        Such as ``62%``.
    """
    return f"{value:.0f}%"


def calls_when_solved(cells: list[Cell]) -> float | None:
    """Mean model calls over solved cells (the project's calls_when_solved).

    Args:
        cells: One harness's cells at one checkpoint.

    Returns:
        The mean, or None when nothing was solved.
    """
    solved = [cell.n_calls for cell in cells if cell.correct]
    return statistics.fmean(solved) if solved else None


def average(values: Iterable[float | None]) -> float | None:
    """The plain mean of the values that exist (each harness weighs the same).

    Args:
        values: One value per harness, None where there is none.

    Returns:
        The mean, or None when no value exists.
    """
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def header(run: Run, harnesses: list[str], train: list[str]) -> list[str]:
    """The run, its checkpoints, and what reading it substituted.

    Args:
        run: The run.
        harnesses: Its harnesses.
        train: The harnesses the pipeline trained in.

    Returns:
        Lines.
    """
    bundled = run.folder.resolve() == PROJECT_RUN.resolve()
    name = f"{PROJECT_RUN_NAME} ({short_path(run.folder)}/)" if bundled else None
    lines = ["read_results.py · the final table, read honestly"]
    lines.append(f"  run         {name or short_path(run.folder)}")
    if bundled:
        lines += wrap(PROJECT_RUN_NOTES, "              ")
    held_out = [h for h in harnesses if h not in train]
    shown = [f"{h} (never trained in)" if h in held_out else h for h in harnesses]
    lines.append(f"  harnesses   {', '.join(shown)}")
    labels = ", ".join(c.label for c in run.checkpoints) or "none"
    lines.append(f"  checkpoints {labels}")
    if run.missing:
        lines.append(f"  missing     {', '.join(run.missing)}: no records yet")
    if run.others:
        lines.append(
            f"  also here   {', '.join(run.others)} (evaluated, not in the table)"
        )
    for note in run.notes:
        first, *rest = wrap(note, NOTE_INDENT)
        lines.append("  note        " + first.strip())
        lines += rest
    return lines


def wrap(text: str, indent: str) -> list[str]:
    """Wrap a paragraph to the page width.

    Args:
        text: The paragraph.
        indent: Every line's indent.

    Returns:
        Lines.
    """
    return textwrap.wrap(
        text,
        WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_on_hyphens=False,
    )


def score_text(cells: list[Cell]) -> tuple[str, float | None]:
    """One harness's entry in the final table.

    Args:
        cells: Its cells at one checkpoint.

    Returns:
        Such as ``15/24 62% (43-79)``, and the pass@1 (None when not run).
    """
    if not cells:
        return "not run", None
    solved = sum(cell.correct for cell in cells)
    low, high = wilson_interval(solved, len(cells))
    score = pass_at_1(solved, len(cells))
    return f"{solved}/{len(cells)} {percent(score)} ({low:.0f}-{high:.0f})", score


def final_table(run: Run, harnesses: list[str]) -> list[str]:
    """Pass@1 per harness with Wilson intervals, and the average.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    sizes = task_sizes(run, harnesses)
    lines = [
        "",
        f"{RULE} the final table: pass@1 with a Wilson 95% interval "
        f"({'/'.join(map(str, sizes))} cells per harness)",
    ]
    row = f"  {'checkpoint':<{LABEL}}" + "".join(f"{h:<{CELL}}" for h in harnesses)
    lines.append(row + "average")
    for checkpoint in run.checkpoints:
        entries = [score_text(checkpoint.harness_cells(h)) for h in harnesses]
        mean = average(score for _, score in entries)
        lines.append(
            f"  {checkpoint.label:<{LABEL}}"
            + "".join(f"{text:<{CELL}}" for text, _ in entries)
            + (percent(mean) if mean is not None else "none")
        )
    lines.append(
        "  average = the plain mean over harnesses, each weighing the same; "
        "(low-high) = Wilson 95%"
    )
    for size in sizes:
        lines.append(
            f"  at {size} tasks per harness, one task = {PERCENT / size:.1f} points"
        )
    return lines


def calls_section(run: Run, harnesses: list[str]) -> list[str]:
    """Calls per solve per checkpoint, by stage, and paired on shared solves.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    lines = ["", f"{RULE} calls per solve: model calls on solved cells only"]
    lines += calls_table(run, harnesses)
    lines.append(
        "  each checkpoint counts only the cells IT solved, so the sets differ; "
        "the pairs below"
    )
    lines += ["  count only cells both checkpoints solved", ""]
    lines += calls_pairs(run, harnesses)
    return lines + pooled_calls(run, harnesses)


def calls_table(run: Run, harnesses: list[str]) -> list[str]:
    """Calls per solve per checkpoint and harness, and what each stage changed.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    lines = [
        f"  {'checkpoint':<{LABEL}}"
        + "".join(f"{h:<{NAME}}" for h in harnesses)
        + "average over harnesses"
    ]
    averages: dict[str, float | None] = {}
    for checkpoint in run.checkpoints:
        per = [calls_when_solved(checkpoint.harness_cells(h)) for h in harnesses]
        mean = averages[checkpoint.label] = average(per)
        texts = [f"{'none':<{NAME}}" if v is None else f"{v:<{NAME}.1f}" for v in per]
        lines.append(
            f"  {checkpoint.label:<{LABEL}}"
            + "".join(texts)
            + ("none" if mean is None else f"{mean:.1f}")
        )
    stages = []
    for before, after in zip(run.checkpoints, run.checkpoints[1:], strict=False):
        first, second = averages[before.label], averages[after.label]
        if first is not None and second is not None:
            stages.append(f"{after.label} {second - first:+.1f}")
    if stages:
        lines.append(f"  by stage (average over harnesses): {', '.join(stages)} calls")
    return lines


def calls_pairs(run: Run, harnesses: list[str]) -> list[str]:
    """Calls before and after on the cells both checkpoints solved, paired.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    lines = [
        f"  {'pair':<{PAIR}} {'harness':<{NAME}}both  mean before -> after  "
        "median  fewer  more  same  p"
    ]
    for first, second in combinations(run.checkpoints, 2):
        for harness in harnesses:
            lines.append(paired_calls_line(first, second, harness))
    lines.append(
        "  p: the exact sign test on fewer against more (same dropped), as for "
        "gained and lost"
    )
    return lines


def shared_solves(
    first: Checkpoint, second: Checkpoint, harnesses: Iterable[str]
) -> list[tuple[int, int]]:
    """Calls before and after on the cells both checkpoints solved.

    Args:
        first: The earlier checkpoint.
        second: The later one.
        harnesses: The harnesses to include.

    Returns:
        One (calls before, calls after) pair per shared solve.
    """
    pairs = []
    for harness in harnesses:
        for cell in first.harness_cells(harness):
            other = second.cells.get((harness, cell.task_id))
            if cell.correct and other is not None and other.correct:
                pairs.append((cell.n_calls, other.n_calls))
    return pairs


def paired_calls_line(first: Checkpoint, second: Checkpoint, harness: str) -> str:
    """One line of the paired calls table.

    Args:
        first: The earlier checkpoint.
        second: The later one.
        harness: The harness.

    Returns:
        The line.
    """
    pair = f"{first.label} -> {second.label}"
    shared = shared_solves(first, second, [harness])
    if not shared:
        return f"  {pair:<{PAIR}} {harness:<{NAME}}   0  no cell solved by both"
    before = [b for b, _ in shared]
    after = [a for _, a in shared]
    fewer = sum(a < b for b, a in shared)
    more = sum(a > b for b, a in shared)
    same = len(shared) - fewer - more
    means = f"{statistics.fmean(before):.1f} -> {statistics.fmean(after):.1f}"
    medians = f"{statistics.median(before):g}->{statistics.median(after):g}"
    return (
        f"  {pair:<{PAIR}} {harness:<{NAME}}{len(shared):4d}  {means:<20} "
        f"{medians:<7} {fewer:5d} {more:5d} {same:5d}  {exact_p(fewer, more):.3f}"
    )


def pooled_calls(run: Run, harnesses: list[str]) -> list[str]:
    """The first and last checkpoints' calls on every cell both solved.

    The article counts its "fewer tool calls" this way (on the tasks both
    solved); our figure is printed here, never beside the article's.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines (none when fewer than two checkpoints exist).
    """
    if len(run.checkpoints) < 2:
        return []
    first, last = run.checkpoints[0], run.checkpoints[-1]
    shared = shared_solves(first, last, harnesses)
    if not shared:
        return []
    before = statistics.fmean(b for b, _ in shared)
    after = statistics.fmean(a for _, a in shared)
    if before <= 0:  # no calls recorded: there is nothing to count fewer of
        return []
    return [
        f"  {first.label} -> {last.label}, every harness pooled: {len(shared)} cells "
        f"solved by both, {before:.1f} -> {after:.1f} calls "
        f"({100 * (before - after) / before:.0f}% fewer)"
    ]


def pair_reading(comparison: Comparison) -> str:
    """Name what a split of gained and lost shows.

    Args:
        comparison: One pair's comparison.

    Returns:
        ``nothing changed``, ``gained: beyond noise``, ``lost: beyond noise``
        or ``within noise``.
    """
    if not comparison.flipped:
        return "nothing changed"
    if comparison.p_value < SIGNIFICANT:
        side = "gained" if len(comparison.gained) > len(comparison.lost) else "lost"
        return f"{side}: beyond noise"
    return "within noise"


@dataclass(frozen=True)
class PairResult:
    """One checkpoint pair in one harness, compared cell by cell.

    Attributes:
        pair: Such as ``base -> sft``.
        harness: The harness.
        before: Cells solved by the first checkpoint.
        after: Cells solved by the second.
        comparison: Gained, lost and p.
    """

    pair: str
    harness: str
    before: int
    after: int
    comparison: Comparison


def pair_results(run: Run, harnesses: list[str]) -> list[PairResult]:
    """Every pair of checkpoints, in every harness, on the cells both ran.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        One result per (pair, harness) that shares at least one cell.
    """
    results = []
    for first, second in combinations(run.checkpoints, 2):
        for harness in harnesses:
            before, after = first.verdicts(harness), second.verdicts(harness)
            shared = sorted(set(before) & set(after))
            if not shared:
                continue
            comparison = compare(
                {t: before[t] for t in shared}, {t: after[t] for t in shared}
            )
            results.append(
                PairResult(
                    f"{first.label} -> {second.label}",
                    harness,
                    sum(before[t] for t in shared),
                    sum(after[t] for t in shared),
                    comparison,
                )
            )
    return results


def pairs_section(results: list[PairResult]) -> list[str]:
    """The gained / lost / p table, then the churn lines.

    Args:
        results: Every pair's result.

    Returns:
        Lines.
    """
    lines = ["", f"{RULE} every pair of checkpoints, cell by cell (exact McNemar p)"]
    lines.append(
        f"  {'pair':<{PAIR}} {'harness':<{NAME}}cells  before  after  gained  "
        "lost  flipped  p      reading"
    )
    for r in results:
        c = r.comparison
        cells = c.both + c.neither + c.flipped
        lines.append(
            f"  {r.pair:<{PAIR}} {r.harness:<{NAME}}{cells:5d}  {r.before:6d}  "
            f"{r.after:5d}  {len(c.gained):6d}  {len(c.lost):4d}  {c.flipped:7d}  "
            f"{c.p_value:.3f}  {pair_reading(c)}"
        )
    lines.append(
        f"  p < {SIGNIFICANT:g} reads as beyond noise; anything else, within it"
    )
    for r in results:
        c = r.comparison
        cells = c.both + c.neither + c.flipped
        cancelled = 2 * min(len(c.gained), len(c.lost))
        if cancelled and cancelled * CHURN_SHARE >= cells:
            lines.append(
                f"  churn: {r.harness} {r.pair}: {c.flipped} of {cells} cells "
                f"flipped, {cancelled} of them cancel out: score "
                f"{r.after - r.before:+d}"
            )
    return lines


@dataclass(frozen=True)
class GapResult:
    """Two harnesses on the same tasks at one checkpoint.

    Attributes:
        label: The checkpoint.
        first: The first harness.
        second: The second harness.
        tasks: Tasks both ran.
        only_first: Tasks only the first solved.
        only_second: Tasks only the second solved.
    """

    label: str
    first: str
    second: str
    tasks: int
    only_first: int
    only_second: int

    @property
    def p_value(self) -> float:
        """The exact p for this split (McNemar, as for checkpoints).

        Returns:
            The p-value.
        """
        return exact_p(self.only_first, self.only_second)


def gap_results(run: Run, harnesses: list[str]) -> list[GapResult]:
    """Every pair of harnesses at every checkpoint, paired by task.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        One result per (checkpoint, harness pair).
    """
    results = []
    for checkpoint in run.checkpoints:
        for first, second in combinations(harnesses, 2):
            a, b = checkpoint.verdicts(first), checkpoint.verdicts(second)
            shared = sorted(set(a) & set(b))
            if shared:
                results.append(
                    GapResult(
                        checkpoint.label,
                        first,
                        second,
                        len(shared),
                        sum(a[t] and not b[t] for t in shared),
                        sum(b[t] and not a[t] for t in shared),
                    )
                )
    return results


def gap_section(results: list[GapResult]) -> list[str]:
    """The harness-gap table.

    Args:
        results: Every (checkpoint, harness pair) result.

    Returns:
        Lines.
    """
    lines = [
        "",
        f"{RULE} the harness gap: the same weights, the same tasks, two harnesses",
        f"  {'checkpoint':<{LABEL}}{'first':<{NAME}}{'second':<{NAME}}tasks  "
        "only first  only second  p      reading",
    ]
    for r in results:
        reading = "beyond noise" if r.p_value < SIGNIFICANT else "within noise"
        lines.append(
            f"  {r.label:<{LABEL}}{r.first:<{NAME}}{r.second:<{NAME}}{r.tasks:5d}  "
            f"{r.only_first:10d}  {r.only_second:11d}  {r.p_value:.3f}  {reading}"
        )
    return lines


def metric_peak(rows: list[dict[str, Any]]) -> tuple[float, int] | None:
    """The highest ``peak_gb`` in a metrics file, and its step.

    Args:
        rows: The metrics rows.

    Returns:
        (peak, step), or None when no row has a peak.
    """
    peaks = [(row["peak_gb"], row["step"]) for row in rows if "peak_gb" in row]
    return max(peaks, key=lambda item: item[0]) if peaks else None


def memory_section(folder: Path) -> list[str]:
    """MLX's own peaks from the metrics, and the sampled footprints.

    Args:
        folder: The run folder.

    Returns:
        Lines.
    """
    lines = [
        "",
        f"{RULE} memory: MLX's own peak (mx.get_peak_memory, the trainer's process)",
    ]
    for name, path in (("sft", SFT_METRICS), ("grpo", GRPO_METRICS)):
        rows = read_jsonl(folder / path)
        peak = metric_peak(rows)
        if peak is None:
            lines.append(f"  {name:<{SOURCE}} no {path} with a peak")
            continue
        lines.append(
            f"  {name:<{SOURCE}} {peak[0]:.2f} GB at step {peak[1]} "
            f"({path}, {steps(len(rows))})"
        )
    lines += footprint_lines(folder / MEMORY_FILE)
    return lines


def read_footprints(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    """Read ``memory.tsv``: each stage's largest footprint per role.

    ``capstone.sh`` writes one line per process per look: the time, the
    stage, the process's role, its pid and its footprint in MB (top's MEM
    column, which counts the GPU memory MLX allocates, unlike ps's RSS). A
    line the sampler was still writing when the run stopped is skipped.

    Args:
        path: ``runs/<name>/memory.tsv``.

    Returns:
        The stages in order, each stage's largest MB per role, and each
        stage's largest MB summed over the processes of one look.
    """
    order: list[str] = []
    largest: dict[str, dict[str, float]] = {}
    looks: dict[tuple[str, str], float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != len(ROLE_COLUMNS) or parts[0] == ROLE_COLUMNS[0]:
            continue
        when, stage, role, _, megabytes = parts
        try:
            size = float(megabytes)
        except ValueError:
            continue
        if stage not in largest:
            largest[stage] = {}
            order.append(stage)
        role = role or "unknown"
        largest[stage][role] = max(largest[stage].get(role, 0.0), size)
        looks[(stage, when)] = looks.get((stage, when), 0.0) + size
    together = {
        stage: max(size for (s, _), size in looks.items() if s == stage)
        for stage in order
    }
    return order, largest, together


def footprint_lines(path: Path) -> list[str]:
    """Summarise ``memory.tsv``: each stage's largest footprints.

    Args:
        path: ``runs/<name>/memory.tsv``.

    Returns:
        Lines.
    """
    if not path.is_file():
        return [
            f"  the servers: no {MEMORY_FILE} here (the project records no engine "
            "peak; capstone.sh samples one)"
        ]
    order, largest, together = read_footprints(path)
    lines = [
        f"  footprint sampled by capstone.sh ({MEMORY_FILE}; top's MEM, in GB), "
        "per stage:"
    ]
    for stage in order:
        seen = largest[stage]
        ranked = [r for r in ROLE_ORDER if r in seen]
        ranked += sorted(set(seen) - set(ROLE_ORDER))
        roles = " · ".join(f"{role} {seen[role] / MB_PER_GB:.2f}" for role in ranked)
        lines.append(
            f"  {stage:<{STAGE_KEY}} {roles} · all at once "
            f"{together[stage] / MB_PER_GB:.2f} (largest)"
        )
    return lines


def parse_clock(text: str) -> datetime | None:
    """Read a stage banner's time.

    Args:
        text: ``2026-10-13 20:31:00`` (capstone.sh) or ``07:31:00`` (the
            project's ``train_from_scratch.sh``).

    Returns:
        The time, or None when the line is not one.
    """
    for pattern in CLOCK_FORMATS:
        try:
            return datetime.strptime(text.strip(), pattern)
        except ValueError:
            continue
    return None


def read_stages(path: Path) -> list[tuple[str, datetime]]:
    """Read the stage banners and the time printed under each.

    Args:
        path: ``stages.txt``: ``=== name ===`` lines, each followed by its
            start time.

    Returns:
        (stage name, start) in order.
    """
    if not path.is_file():
        return []
    stages: list[tuple[str, datetime]] = []
    name = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(BANNER) and line.endswith(BANNER):
            name = line.strip("= ").strip()
            continue
        clock = parse_clock(line)
        if name is not None and clock is not None:
            stages.append((name, clock))
            name = None
    return stages


def steps(count: int) -> str:
    """Write a number of steps.

    Args:
        count: How many.

    Returns:
        Such as ``1 step`` or ``15 steps``.
    """
    return f"{count} step" if count == 1 else f"{count} steps"


def duration(seconds: float) -> str:
    """Write a duration in minutes and seconds.

    Args:
        seconds: The duration.

    Returns:
        Such as ``21 min 50 s``.
    """
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    return f"{minutes} min {rest:02d} s"


def stage_lines(stages: list[tuple[str, datetime]]) -> list[str]:
    """One line per stage: its time from banner to banner.

    Args:
        stages: The banners and their times.

    Returns:
        Lines.
    """
    lines = []
    for index, (name, start) in enumerate(stages):
        if index + 1 == len(stages):
            if name != DONE_STAGE:
                lines.append(f"  {name:<{STAGE}} no later banner: the clock ends here")
            continue
        end = stages[index + 1][1]
        if end < start:  # a clock without a date passed midnight
            end += timedelta(days=1)
        seconds = (end - start).total_seconds()
        span = f"{start:%H:%M:%S} -> {end:%H:%M:%S}"
        lines.append(f"  {name:<{STAGE}} {duration(seconds):>12}  {span}")
    return lines


def time_section(run: Run, stages_file: Path) -> list[str]:
    """The stage clock, then the records' own clocks.

    Args:
        run: The run.
        stages_file: The stage clock to read.

    Returns:
        Lines.
    """
    lines = ["", f"{RULE} wall time"]
    lines += clock_lines(stages_file)
    lines.append("  the records' own clocks:")
    return lines + record_clock_lines(run)


def clock_lines(stages_file: Path) -> list[str]:
    """The stage clock, banner to banner, and its total when it closed.

    Args:
        stages_file: The stage clock to read.

    Returns:
        Lines.
    """
    stages = read_stages(stages_file)
    if not stages:
        return [f"  no stage clock ({short_path(stages_file)} is missing)"]
    lines = [f"  the stage clock, banner to banner ({short_path(stages_file)}):"]
    lines += stage_lines(stages)
    if stages[-1][0] == DONE_STAGE:
        first, last = stages[0][1], stages[-1][1]
        total = (last - first).total_seconds()
        if first.year == UNDATED_YEAR:  # no date: at most one midnight passed
            total %= SECONDS_PER_DAY
        lines.append(f"  {'all stages':<{STAGE}} {duration(total):>12}")
    return lines


def record_clock_lines(run: Run) -> list[str]:
    """The trainers' own step clocks and the evaluations' added-up rollouts.

    Args:
        run: The run.

    Returns:
        Lines.
    """
    lines = []
    sft = read_jsonl(run.folder / SFT_METRICS)
    if sft:
        seconds = sum(row.get("seconds", 0.0) for row in sft)
        lines.append(
            f"  {'sft':<{SOURCE}} {steps(len(sft))}, {duration(seconds)} of "
            "training steps (the model's loading not counted)"
        )
    grpo = [row for row in read_jsonl(run.folder / GRPO_METRICS) if "step" in row]
    if grpo:
        seconds = sum(row.get("step_seconds", 0.0) for row in grpo)
        rollouts = sum(row.get("rollout_seconds", 0.0) for row in grpo)
        lines.append(
            f"  {'grpo':<{SOURCE}} {steps(len(grpo))}, {duration(seconds)} "
            f"({duration(rollouts)} of it rollouts)"
        )
    for checkpoint in run.checkpoints:
        seconds = sum(cell.seconds for cell in checkpoint.cells.values())
        lines.append(
            f"  {'eval':<{SOURCE}} {checkpoint.label:<{EVAL_LABEL}} "
            f"{len(checkpoint.cells)} cells, {duration(seconds)} of rollouts added up"
        )
    if run.checkpoints:
        lines.append(
            "  the project evaluates several cells side by side "
            "(Settings.concurrency), so the"
        )
        lines.append("  added-up time is longer than the stage's wall time")
    return lines


def network_section(folder: Path) -> list[str]:
    """What the network check found, and what it was.

    Args:
        folder: The run folder.

    Returns:
        Lines.
    """
    path = folder / NETWORK_FILE
    lines = ["", f"{RULE} the network check ({NETWORK_FILE})"]
    if not path.is_file():
        return lines + ["  none in this run folder: nothing was checked"]
    text = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in text if line.startswith("#")]
    found = [line for line in text if line.strip() and not line.startswith("#")]
    lines += [f"  {line}" for line in comments]
    if found:
        lines.append(f"  {len(found)} connection line(s) off this machine:")
        lines += [f"    {line}" for line in found]
    else:
        lines.append("  no connection lines: nothing left this machine while it looked")
    return lines


def reading_section(
    run: Run,
    harnesses: list[str],
    train: list[str],
    pairs: list[PairResult],
    gaps: list[GapResult],
) -> list[str]:
    """The reading, worded from the tables (nothing here is typed by hand).

    Args:
        run: The run.
        harnesses: Its harnesses.
        train: The harnesses the pipeline trained in.
        pairs: Every checkpoint pair's result.
        gaps: Every harness pair's result.

    Returns:
        Lines.
    """
    lines = ["", f"{RULE} the reading, worded from the tables above"]
    lines += gap_reading(gaps)
    lines += checkpoint_reading(pairs)
    lines += average_lines(run, harnesses)
    lines += calls_reading(run, harnesses)
    for harness in (h for h in harnesses if h not in train):
        lines += held_out_lines(run, harness, pairs)
    return lines


def gap_reading(gaps: list[GapResult]) -> list[str]:
    """The harness gaps that are beyond noise, each in one sentence.

    Args:
        gaps: Every (checkpoint, harness pair) result.

    Returns:
        Lines (none when no gap is beyond noise).
    """
    beyond = [g for g in gaps if g.p_value < SIGNIFICANT]
    lines = ["  the harness gap, beyond noise:"] if beyond else []
    for g in beyond:
        first_ahead = g.only_first > g.only_second
        ahead, behind = (g.first, g.second) if first_ahead else (g.second, g.first)
        most, least = sorted((g.only_first, g.only_second), reverse=True)
        lines.append(
            f"    {g.label}: {ahead} ahead of {behind}, {most} tasks to "
            f"{least} (p {g.p_value:.3f})"
        )
    return lines


def checkpoint_reading(pairs: list[PairResult]) -> list[str]:
    """Every checkpoint pair that changed, beyond noise or within it.

    Args:
        pairs: Every checkpoint pair's result.

    Returns:
        Lines.
    """
    clear = [p for p in pairs if p.comparison.p_value < SIGNIFICANT]
    noise = [p for p in pairs if p.comparison.flipped and p not in clear]
    lines = [f"  checkpoints, beyond noise (p < {SIGNIFICANT:g}):"]
    lines += [] if clear else ["    none"]
    for p in clear:
        c = p.comparison
        lines.append(
            f"    {p.pair}, {p.harness}: {p.before} -> {p.after} solved, "
            f"{len(c.gained)} gained, {len(c.lost)} lost (p {c.p_value:.3f})"
        )
    lines.append("  checkpoints, within noise:")
    lines += [] if noise else ["    none"]
    for p in noise:
        c = p.comparison
        lines.append(
            f"    {p.pair}, {p.harness}: {p.after - p.before:+d} "
            f"({len(c.gained)} gained, {len(c.lost)} lost, p {c.p_value:.3f})"
        )
    return lines


def average_lines(run: Run, harnesses: list[str]) -> list[str]:
    """The average's path, with what it rests on.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    path = []
    for checkpoint in run.checkpoints:
        scores = [
            pass_at_1(sum(c.correct for c in cells), len(cells))
            for cells in (checkpoint.harness_cells(h) for h in harnesses)
            if cells
        ]
        mean = average(scores)
        if mean is not None:
            path.append(percent(mean))
    if len(path) < 2:
        return []
    sizes = task_sizes(run, harnesses)
    return [
        f"  the average, {' -> '.join(path)}: one seed, "
        f"{'/'.join(map(str, sizes))} tasks per harness, read only through "
        "the pairs above"
    ]


def calls_reading(run: Run, harnesses: list[str]) -> list[str]:
    """Calls per solve between consecutive checkpoints, pooled and paired.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    lines = []
    for first, second in zip(run.checkpoints, run.checkpoints[1:], strict=False):
        shared = shared_solves(first, second, harnesses)
        if not shared:
            continue
        fewer = sum(a < b for b, a in shared)
        more = sum(a > b for b, a in shared)
        p = exact_p(fewer, more)
        reading = "beyond noise" if p < SIGNIFICANT else "within noise"
        lines.append(
            f"  calls, {first.label} -> {second.label}, every harness pooled: "
            f"fewer on {fewer} of {len(shared)} shared solves, more on {more} "
            f"(p {p:.3f}): {reading}"
        )
    return lines


def held_out_lines(run: Run, harness: str, pairs: list[PairResult]) -> list[str]:
    """What happened in a harness the pipeline never trained in.

    Args:
        run: The run.
        harness: The held-out harness.
        pairs: Every checkpoint pair's result.

    Returns:
        Lines.
    """
    if len(run.checkpoints) < 2:
        return []
    first, last = run.checkpoints[0].label, run.checkpoints[-1].label
    wanted = f"{first} -> {last}"
    match = [p for p in pairs if p.harness == harness and p.pair == wanted]
    if not match:
        return []
    p = match[0]
    c = p.comparison
    reading = "shown" if c.p_value < SIGNIFICANT else "suggestive, not shown"
    return [
        f"  {harness}, never trained in: {p.before} -> {p.after} solved, "
        f"{first} -> {last} ({len(c.gained)} gained, {len(c.lost)} lost, "
        f"p {c.p_value:.3f}): {reading}"
    ]


def article_box(run: Run, harnesses: list[str]) -> list[str]:
    """The article's own numbers, in a box, with ours below it.

    Args:
        run: The run.
        harnesses: Its harnesses.

    Returns:
        Lines.
    """
    inner = WIDTH - 4
    title = f"┌─ {ARTICLE_TITLE} "
    lines = ["", f"{RULE} the article's numbers, kept apart", ""]
    lines.append(title + "─" * (WIDTH - len(title) - 1) + "┐")
    for paragraph in (ARTICLE_SOURCE, *ARTICLE_LINES):
        for text in textwrap.wrap(paragraph, inner):
            lines.append(f"│ {text:<{inner}} │")
    lines.append("└" + "─" * (WIDTH - 2) + "┘")
    lines.append(
        "  ours: three harnesses on one laptop, our own tasks, one seed; read "
        "through the"
    )
    lines.append("  tables above, never beside these")
    return lines


def report(run: Run, stages_file: Path, train: list[str]) -> str:
    """Everything this script prints, as one text.

    Args:
        run: The run.
        stages_file: The stage clock.
        train: The harnesses the pipeline trained in.

    Returns:
        The report.
    """
    harnesses = harnesses_of(run)
    lines = header(run, harnesses, train)
    if not run.checkpoints:
        lines.append("  no evaluation records yet: nothing to read")
        return "\n".join(lines)
    pairs = pair_results(run, harnesses)
    gaps = gap_results(run, harnesses)
    lines += final_table(run, harnesses)
    lines += calls_section(run, harnesses)
    lines += pairs_section(pairs)
    lines += gap_section(gaps)
    lines += memory_section(run.folder)
    lines += time_section(run, stages_file)
    lines += network_section(run.folder)
    lines += reading_section(run, harnesses, train, pairs, gaps)
    lines += article_box(run, harnesses)
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments (None reads ``sys.argv``).

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_RUN,
        help="a runs/<name>/ folder (default: the project's recorded run, bundled)",
    )
    parser.add_argument(
        "--stages",
        type=Path,
        default=None,
        help=f"the stage clock (default: <run-dir>/{STAGES_FILE})",
    )
    parser.add_argument(
        "--train-harnesses",
        default=TRAIN_HARNESSES,
        help="the harnesses the pipeline trained in (the rest are held out)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the report for one run folder.

    Args:
        argv: The arguments (None reads ``sys.argv``).

    Returns:
        0, or 1 when the folder does not exist or holds a broken record.
    """
    args = parse_args(argv)
    folder = args.run_dir.expanduser()
    if not folder.is_dir():
        print(f"no run folder at {short_path(folder)}", file=sys.stderr)
        return 1
    stages = (args.stages or folder / STAGES_FILE).expanduser()
    train = [name.strip() for name in args.train_harnesses.split(",") if name.strip()]
    try:
        run = read_run(folder)
    except ValueError as error:  # a broken row: say where, never half a table
        print(f"cannot read {short_path(folder)}: {error}", file=sys.stderr)
        return 1
    print(report(run, stages, train))
    return 0


if __name__ == "__main__":
    sys.exit(main())
