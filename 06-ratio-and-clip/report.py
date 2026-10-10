"""How this chapter prints: one call by hand, steps, the ratio tables, the end of a run.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``grpo_clip.py`` can stay about the ratio, the clip and the
loop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from grpo import GROUP_SIZE, QUESTIONS_PER_STEP, Defaults
from questions import LEVELS, QuestionSet

if TYPE_CHECKING:  # only for type hints: grpo_clip.py imports this module
    from grpo_clip import LevelScore, RunResult, Settings, Step
    from seeds import Summary


class UseCounts(NamedTuple):
    """What one use of a batch saw, counted, for the seed tables.

    Attributes:
        ratios: How many ratios the loss saw.
        outside: How many of them fell outside 0.8 to 1.2.
        clipped: How many the clip switched off.
    """

    ratios: int
    outside: int
    clipped: int


class RatioRow(NamedTuple):
    """One row of a ratio table: a label, its ratios and which were clipped.

    Attributes:
        label: For example ``"use 2"``.
        ratios: The ratios.
        clipped: For each ratio, whether the clip switched its call off.
    """

    label: str
    ratios: np.ndarray
    clipped: np.ndarray


PERCENT = 100
RATIO_DECIMALS = 3  # "ratio 1.000", as the project's trainer prints it
SHOWN_DECIMALS = 2  # probabilities and advantages
LEVEL_WIDTH = 8  # "medium  "
COUNT_WIDTH = 2
TOTAL_WIDTH = 4  # "15 of 15  ": the total and the gap after it
CELL_WIDTH = 10  # one column of the quarter table
FIRST_CELL_WIDTH = 12  # its first column, whose heading says "steps"
BIN_WIDTH = 10  # one column of the ratio table
USE_WIDTH = 6  # "use 1 "
CALLS_WIDTH = 8  # "  12,345"
MEAN_WIDTH = 7  # "1.000  "
INDENT = "  "
SEPARATOR = " · "
NOT_AVAILABLE = "n/a"  # a step that learned from nothing has no ratio
NO_CLIP = "no clip"
# Where the ratio table splits the ratios: below 0.8, 0.8 to 0.9, 0.9 to 0.99,
# within 1% of 1, 1.01 to 1.1, 1.1 to 1.2, above 1.2. A ratio on an edge
# counts in the bin above it, except 1.2 itself, which the clip leaves alone.
LOW = 1.0 - Defaults.CLIP_EPSILON
HIGH = 1.0 + Defaults.CLIP_EPSILON
INNER_EDGES = (0.9, 0.99, 1.01, 1.1)  # finer bins inside the clip's range
BIN_EDGES = (LOW, *INNER_EDGES, HIGH)
BIN_LABELS = (
    f"<{LOW:g}",
    *(f"{start:g}-{end:g}" for start, end in pairwise(BIN_EDGES)),
    f">{HIGH:g}",
)
EQUAL_TOLERANCE = 1e-12  # two terms this close are the same number
# The seed tables.
SEED_WIDTH = 6  # " 0    "
PAIR_WIDTH = 17  # "38% / 20%" and room
USE_CELL = 4  # one use's share, " 25"
BLOCK_GAP = 2  # between the two blocks of uses
QUARTER_CELL = 4  # one quarter's share, " 100"
LEARN_WIDTH = 18  # four quarters and a gap
NAME_BREAK = ", "  # a long run name is printed on two lines, after its comma


def histogram(ratios: np.ndarray) -> list[int]:
    """Count ratios in the table's seven bins.

    Args:
        ratios: Any ratios.

    Returns:
        One count per bin, lowest bin first.
    """
    # side="right" puts a ratio equal to an edge in the bin above, except that
    # the last edge (1.2) is matched on the left so exactly 1.2 is not "above".
    inner = np.searchsorted(BIN_EDGES[:-1], ratios, side="right")
    above = ratios > BIN_EDGES[-1]
    places = np.where(above, len(BIN_EDGES), inner)
    return np.bincount(places, minlength=len(BIN_LABELS)).tolist()


def tidy(value: float, decimals: int = SHOWN_DECIMALS) -> float:
    """Round a number for printing, so a tiny negative never shows as -0.00.

    Args:
        value: The number.
        decimals: How many decimals will be shown.

    Returns:
        The rounded number; adding 0.0 turns -0.0 into 0.0.
    """
    return round(value, decimals) + 0.0


def print_token(old: float, new: float) -> None:
    """Print one call's ratio, its clip and the loss for A = +1 and A = -1.

    Args:
        old: The action's probability when the rollout was sampled.
        new: Its probability with the learner's weights now.
    """
    old_log, new_log = math.log(old), math.log(new)
    ratio = math.exp(new_log - old_log)  # as loss_fn works it out
    kept = min(max(ratio, LOW), HIGH)
    print(f"one call: its action had p {old:.2f} when the rollout was sampled (old)")
    print(f"and has p {new:.2f} with the learner's weights now (new)")
    print(f"ratio = new p / old p = {new:.2f} / {old:.2f} = {ratio:.2f}")
    print(
        f"      = exp(new log p - old log p) = exp({new_log:.2f} - ({old_log:.2f}))"
        f" = {ratio:.2f}"
    )
    print(f"clip(ratio, {LOW:g}, {HIGH:g}) = {kept:.2f}")
    for advantage, meaning in ((1.0, "did better"), (-1.0, "did worse")):
        unclipped, clipped = ratio * advantage, kept * advantage
        smaller = min(unclipped, clipped)
        # The clipped term is flat once the clip bites: it gives no slope.
        switched_off = clipped < unclipped - EQUAL_TOLERANCE
        if abs(clipped - unclipped) <= EQUAL_TOLERANCE:
            which = "the same: the ratio is inside the range"
        else:
            which = "the clipped one" if switched_off else "the unclipped one"
        print()
        print(f"A = {advantage:+g} (the rollout {meaning} than its group)")
        print(f"{INDENT}ratio x A         = {tidy(unclipped):+.2f}")
        print(f"{INDENT}clip(ratio) x A   = {tidy(clipped):+.2f}")
        print(f"{INDENT}min of the two    = {tidy(smaller):+.2f}  {which}")
        print(f"{INDENT}loss = -min       = {tidy(-smaller):+.2f}")
        if switched_off:
            print(f"{INDENT}push on log p      0.00  the clip has switched it off")
        else:
            print(
                f"{INDENT}push on log p     {tidy(ratio * advantage):+.2f}"
                f"  ratio x A: {ratio:.2f} times chapter 5's push, which was A"
            )
    print()
    print("min() keeps the more pessimistic number: a call that has already moved")
    print(
        f"{Defaults.CLIP_EPSILON:.0%} the way its advantage wants gets no more push;"
        " a call that moved the"
    )
    print("wrong way always gets its whole push, ratio x A")


def print_real_call(
    step: int, use: int, action: str, old: float, new: float, advantage: float
) -> None:
    """Print a call the clip switched off in a real run, with its arithmetic.

    Args:
        step: The step's number.
        use: Which use of the batch, from 1.
        action: The action taken.
        old: Its probability when the rollout was sampled.
        new: Its probability with the learner's weights.
        advantage: The rollout's advantage.
    """
    ratio = new / old
    kept = min(max(ratio, LOW), HIGH)
    print(f"the first call the clip switched off: step {step}, use {use} of its batch")
    # Three decimals, so the ratio can be checked from the two p printed.
    print(
        f"{INDENT}{action} · old p {old:.{RATIO_DECIMALS}f}"
        f" · new p {new:.{RATIO_DECIMALS}f} · ratio {ratio:.{SHOWN_DECIMALS}f}"
        f" · A {tidy(advantage):+.{SHOWN_DECIMALS}f}"
    )
    print(
        f"{INDENT}min({ratio:.2f} x {tidy(advantage):+.2f},"
        f" {kept:.2f} x {tidy(advantage):+.2f})"
        f" = {tidy(kept * advantage):+.2f}, the clipped term: push 0"
    )
    print()


def print_nothing_clipped(clip: bool) -> None:
    """Say that --trace found no call the clip switched off, and why.

    Args:
        clip: Whether the clip was on.
    """
    if clip:
        print("(--trace: the clip switched off no call in this run)")
    else:
        print("(--trace: the clip is off, so it switched off no call)")
    print()


def sampler_text(settings: Settings) -> str:
    """Say which weights the sampler uses.

    Args:
        settings: The run's knobs.

    Returns:
        A few words for the header.
    """
    if settings.stale == 0:
        return "sampled with the learner's weights"
    plural = "s" if settings.stale > 1 else ""
    return f"sampled with the weights from {settings.stale} update{plural} earlier"


def print_header(settings: Settings, questions: QuestionSet) -> None:
    """Print the run's settings, the loss and where the old p comes from.

    Args:
        settings: The run's knobs.
        questions: The fair questions, split by city.
    """
    plural = "s" if settings.updates > 1 else ""
    print(
        f"grpo_clip · seed {settings.seed} · lr {settings.lr:g}"
        f" · {settings.steps} steps · {settings.updates} update{plural} per batch"
    )
    print(
        f"each step: {QUESTIONS_PER_STEP} questions x {GROUP_SIZE} rollouts at"
        f" temperature 1, {sampler_text(settings)}"
    )
    print(
        f"questions: {len(questions.training)} to train on,"
        f" {len(questions.held_out)} held out (chapter 5's, one set of weights"
        " per level)"
    )
    if settings.clip:
        print(
            f"loss per call = -min(ratio x A, clip(ratio, {LOW:g}, {HIGH:g}) x A),"
            " ratio = new p / old p"
        )
    else:
        print("loss per call = -ratio x A (no clip), ratio = new p / old p")
    if settings.recompute_old:
        print("old p: recomputed with the learner's weights when the batch arrives")
    else:
        print("old p: recorded when the rollout was sampled")
    print()


def ratio_text(ratios: np.ndarray, clipped: np.ndarray, clip: bool) -> str:
    """Write a step's mean ratio and the share of its calls clipped.

    Args:
        ratios: Every ratio of the step.
        clipped: Whether each was clipped.
        clip: Whether the clip is on.

    Returns:
        For example ``"ratio 1.000 · clipped 0%"``, or ``n/a`` for both when
        the step learned from nothing.
    """
    if not ratios.size:
        mean = f"ratio {NOT_AVAILABLE}"
        clipping = f"clipped {NOT_AVAILABLE}"
    else:
        mean = f"ratio {ratios.mean():.{RATIO_DECIMALS}f}"
        clipping = f"clipped {round(PERCENT * clipped.mean())}%"
    return f"{mean}{SEPARATOR}{clipping if clip else NO_CLIP}"


def print_step(step: Step, metrics: Mapping[str, Any], clip: bool) -> None:
    """Print one step on one line, as the project's trainer does.

    Args:
        step: The step.
        metrics: Its metrics.jsonl record.
        clip: Whether the clip is on.
    """
    print(
        f"step {step.number:>3}{SEPARATOR}reward {metrics['mean_reward']:.2f}"
        f"{SEPARATOR}correct {round(PERCENT * metrics['accuracy'])}%"
        f"{SEPARATOR}calls {metrics['mean_calls']:.1f}"
        f"{SEPARATOR}no-signal groups {step.no_signal}/{len(step.groups)}"
        f"{SEPARATOR}{ratio_text(step.ratios, step.clipped, clip)}"
    )


def share(count: int, total: int) -> str:
    """Write a count as a whole percentage of a total.

    Args:
        count: The count.
        total: The total (0 gives "0%").

    Returns:
        For example ``"12%"``.
    """
    return f"{percent(count, total)}%"


def print_ratio_table(rows: Sequence[RatioRow], title: str, clip: bool) -> None:
    """Print ratios by bin, one row per use of the batch (or per run).

    Args:
        rows: A label, the ratios and whether each was clipped, per row.
        title: The line above the table.
        clip: Whether the clip is on (otherwise the last column says so).
    """
    print(title)
    heading = "".join(f"{label:<{BIN_WIDTH}}" for label in BIN_LABELS)
    print(
        f"{'':<{USE_WIDTH}}{'calls':>{CALLS_WIDTH}}  {'mean':<{MEAN_WIDTH}}"
        f"{heading}clipped"
    )
    for label, ratios, clipped in rows:
        counts = histogram(ratios)
        cells = "".join(f"{share(c, ratios.size):<{BIN_WIDTH}}" for c in counts)
        mean = f"{ratios.mean():.{RATIO_DECIMALS}f}" if ratios.size else NOT_AVAILABLE
        last = share(int(clipped.sum()), ratios.size) if clip else NO_CLIP
        print(
            f"{label:<{USE_WIDTH}}{ratios.size:>{CALLS_WIDTH},}  {mean:<{MEAN_WIDTH}}"
            f"{cells}{last}"
        )


def use_rows(steps: Sequence[Step]) -> list[RatioRow]:
    """Gather every step's ratios by use of the batch.

    Args:
        steps: Every step of a run.

    Returns:
        One row per use: its label, its ratios and whether each was clipped.
    """
    uses = len(steps[0].uses)
    rows = []
    for index in range(uses):
        ratios = np.concatenate([step.uses[index].ratios for step in steps])
        clipped = np.concatenate([step.uses[index].clipped for step in steps])
        rows.append(RatioRow(f"use {index + 1}", ratios, clipped))
    return rows


def stretch_labels(edges: Sequence[int]) -> list[str]:
    """Name each stretch of steps, e.g. ``["steps 1-50", "51-100", ...]``.

    Args:
        edges: The stretch boundaries.

    Returns:
        One heading per stretch; the first says "steps".
    """
    labels = [f"{start + 1}-{end}" for start, end in pairwise(edges)]
    labels[0] = f"steps {labels[0]}"
    return labels


def table_row(name: str, cells: Sequence[str]) -> str:
    """Lay out one row of the quarter table: a name, then one cell per stretch.

    Args:
        name: The row's name (a level), or "" for the heading.
        cells: The cells.

    Returns:
        The row, without trailing spaces.
    """
    first, *rest = cells
    row = f"{name:<{LEVEL_WIDTH}}{first:<{FIRST_CELL_WIDTH}}" + "".join(
        f"{cell:<{CELL_WIDTH}}" for cell in rest
    )
    return row.rstrip()


def right_by_quarter(
    steps: Sequence[Step], edges: Sequence[int]
) -> dict[str, list[list[int]]]:
    """Count, per level and stretch, the rollouts and how many were right.

    Args:
        steps: Every step of a run.
        edges: The stretch boundaries.

    Returns:
        For each level, one ``[right, rollouts]`` pair per stretch.
    """
    table = {level: [[0, 0] for _ in edges[1:]] for level in LEVELS}
    for step in steps:
        stretch = next(i for i, end in enumerate(edges[1:]) if step.number <= end)
        for group in step.groups:
            cell = table[group.question.level][stretch]
            cell[0] += sum(r.verdict.correct for r in group.rollouts)
            cell[1] += len(group.rollouts)
    return table


def print_right_table(
    table: Mapping[str, Sequence[Sequence[int]]], edges: Sequence[int]
) -> None:
    """Print the share of rollouts right, per level and quarter of the run.

    Args:
        table: For each level, ``[right, rollouts]`` per stretch.
        edges: The stretch boundaries.
    """
    print("rollouts right, in each quarter of the run")
    print(table_row("", stretch_labels(edges)))
    for level in LEVELS:
        print(table_row(level, [share(right, total) for right, total in table[level]]))


def calls_text(actions: Sequence[str]) -> str:
    """Write a rollout's actions on one line, with repeats counted.

    Args:
        actions: The actions of every call, in order.

    Returns:
        For example ``"LOOK, QUERY_A, WRITE x6"``.
    """
    parts = []
    for action, repeats in groupby(actions):  # runs of the same action
        count = len(list(repeats))
        parts.append(action if count == 1 else f"{action} x{count}")
    return ", ".join(parts)


def print_greedy(scores: Sequence[LevelScore]) -> None:
    """Print the greedy test at the end, one line per level.

    Args:
        scores: One score per level.
    """
    print("greedy test at the end (always the most likely action)")
    for score in scores:
        actions = calls_text(score.actions)
        print(
            f"{INDENT}{score.level:<{LEVEL_WIDTH}}"
            f"training {score.training:>{COUNT_WIDTH}} of"
            f" {score.training_total:<{TOTAL_WIDTH}}"
            f"held out {score.held_out} of {score.held_out_total}"
            f"{SEPARATOR}{actions}"
        )


def hidden_rows(steps: Sequence[Step]) -> list[RatioRow]:
    """Gather, by use, the ratios against the recorded p that --recompute-old hid.

    Args:
        steps: Every step of a run made with ``--recompute-old``.

    Returns:
        One row per use: its label, the hidden ratios and which of them the
        clip would have switched off.
    """
    rows = []
    for index in range(len(steps[0].uses)):
        uses = [step.uses[index] for step in steps]
        ratios = [u.recorded_ratios for u in uses if u.recorded_ratios is not None]
        clipped = [u.recorded_clipped for u in uses if u.recorded_clipped is not None]
        rows.append(
            RatioRow(
                f"use {index + 1}", np.concatenate(ratios), np.concatenate(clipped)
            )
        )
    return rows


def print_hidden(steps: Sequence[Step], clip: bool) -> None:
    """Print what --recompute-old hid: the ratios against the recorded p.

    Args:
        steps: Every step of a run made with ``--recompute-old``.
        clip: Whether the clip is on.
    """
    print()
    print_ratio_table(
        hidden_rows(steps),
        "what --recompute-old hid: the same calls' ratios against the p recorded"
        " at sampling\n(clipped: the calls the clip would have switched off)",
        clip,
    )


def print_step_means(steps: Sequence[Step]) -> None:
    """Print how far the per-step mean ratio wandered over the run.

    Args:
        steps: Every step of a run.
    """
    means = [step.ratios.mean() for step in steps if step.ratios.size]
    if not means:
        print("no step learned anything, so there is no ratio to show")
        return
    print(
        f"mean ratio per step, over the {len(means)} steps that learned:"
        f" {min(means):.{RATIO_DECIMALS}f} to {max(means):.{RATIO_DECIMALS}f}"
    )


def print_run_end(result: RunResult) -> None:
    """Print the ratio table, the share right by quarter and the greedy test.

    Args:
        result: The finished run.
    """
    settings = result.settings
    print()
    print_ratio_table(
        use_rows(result.steps),
        "ratios the loss saw, by use of the batch (every step of the run)",
        settings.clip,
    )
    if settings.recompute_old:
        print_hidden(result.steps, settings.clip)
    print_step_means(result.steps)
    print()
    print_right_table(right_by_quarter(result.steps, result.edges), result.edges)
    print()
    print_greedy(result.scores)


def print_logs(
    metrics: str, metrics_md5: str, rollouts: str, rollouts_md5: str
) -> None:
    """Print where the two logs went, their md5s and how to redraw the run.

    Args:
        metrics: Where metrics.jsonl went, as text.
        metrics_md5: Its md5.
        rollouts: Where rollouts.jsonl went, as text.
        rollouts_md5: Its md5.
    """
    print()
    print(f"wrote {metrics}")
    print(f"md5   {metrics_md5}")
    print(f"wrote {rollouts}")
    print(f"md5   {rollouts_md5}")
    print(f"redraw it: uv run plot.py {metrics}")


def print_seeds_header(
    seeds: int, steps: int, lr: float, slow_lr: float, runs: int
) -> None:
    """Print what ``seeds.py`` runs.

    Args:
        seeds: How many seeds.
        steps: Steps per run.
        lr: The learning rate of every run but the slow one.
        slow_lr: The slow run's learning rate.
        runs: How many ways each seed is run.
    """
    print(
        f"seeds 0-{seeds - 1} · {steps} steps per run · lr {lr:g}"
        f" (one run at lr {slow_lr:g}) · {runs} runs per seed"
    )
    print()


def percent(part: int, total: int) -> int:
    """Turn a count into a whole percentage of a total.

    Args:
        part: The count.
        total: The total (0 gives 0).

    Returns:
        The percentage, rounded.
    """
    return round(PERCENT * part / total) if total else 0


def seed_row(label: str, cells: Sequence[str], width: int) -> str:
    """Lay out one row of a seed table: the seed (or "all"), then its cells.

    Args:
        label: The seed's number as text, or "all".
        cells: One cell per column.
        width: How wide each column is.

    Returns:
        The row, without trailing spaces.
    """
    row = f"{label:<{SEED_WIDTH}}" + "".join(f"{cell:<{width}}" for cell in cells)
    return row.rstrip()


def seed_label(seed: int) -> str:
    """Write a seed's number right-aligned under the heading "seed".

    Args:
        seed: The seed.

    Returns:
        For example ``" 0"``.
    """
    return f"{seed:>{COUNT_WIDTH}}"


def pair(outside: int, clipped: int, total: int) -> str:
    """Write a share outside 0.8-1.2 and a share clipped, e.g. ``"38% / 20%"``.

    Args:
        outside: How many ratios fell outside the range.
        clipped: How many were clipped.
        total: How many ratios there were.

    Returns:
        The two shares.
    """
    return f"{percent(outside, total)}% / {percent(clipped, total)}%"


def print_ratio_claims(
    fresh: Sequence[Summary], stale: Sequence[Summary], hidden: Sequence[Summary]
) -> None:
    """Print table 1: fresh, stale and stale-with-the-bug ratios, per seed.

    Args:
        fresh: One summary per seed of the default run.
        stale: The same for ``--stale 2``.
        hidden: The same for ``--stale 2 --recompute-old``.
    """
    print(f"1. the ratios the loss saw: % outside {LOW:g}-{HIGH:g} / % clipped")
    names = [fresh[0].name, stale[0].name, hidden[0].name]
    print(seed_row("seed", names, PAIR_WIDTH))
    print(
        seed_row("", ["", "", "as the loss saw", "against the recorded p"], PAIR_WIDTH)
    )
    for one, two, three in zip(fresh, stale, hidden, strict=True):
        cells = [
            pair(one.outside, one.clipped, one.ratios),
            pair(two.outside, two.clipped, two.ratios),
            pair(three.outside, three.clipped, three.ratios),
            pair(three.hidden_outside, three.hidden_clipped, three.ratios),
        ]
        print(seed_row(seed_label(one.seed), cells, PAIR_WIDTH))
    totals = [
        pair(
            sum(s.outside for s in runs),
            sum(s.clipped for s in runs),
            sum(s.ratios for s in runs),
        )
        for runs in (fresh, stale, hidden)
    ]
    totals.append(
        pair(
            sum(s.hidden_outside for s in hidden),
            sum(s.hidden_clipped for s in hidden),
            sum(s.ratios for s in hidden),
        )
    )
    print(seed_row("all", totals, PAIR_WIDTH))
    for runs, what in ((fresh, names[0]), (hidden, f"{names[2]}, as the loss saw")):
        total = sum(s.ratios for s in runs)
        exact = total - sum(s.not_one for s in runs)
        print(f"{what}: {exact:,} of {total:,} ratios exactly 1")
    print()


def use_cells(counts: Sequence[UseCounts], field: str) -> str:
    """Write one seed's share per use, for one count, as a row of numbers.

    Args:
        counts: One set of counts per use of the batch.
        field: Which count: ``"outside"`` or ``"clipped"``.

    Returns:
        For example ``"   0  37  29"``.
    """
    return "".join(
        f"{percent(getattr(c, field), c.ratios):>{USE_CELL}}" for c in counts
    )


def pooled_uses(block: Sequence[Sequence[UseCounts]]) -> list[UseCounts]:
    """Add every seed's counts together, use by use.

    Args:
        block: Per seed, one set of counts per use.

    Returns:
        One set of counts per use, summed over the seeds.
    """
    return [
        UseCounts(
            sum(seed[use].ratios for seed in block),
            sum(seed[use].outside for seed in block),
            sum(seed[use].clipped for seed in block),
        )
        for use in range(len(block[0]))
    ]


def print_by_use(
    title: str,
    field: str,
    blocks: Sequence[Sequence[Sequence[UseCounts]]],
    labels: Sequence[str],
) -> None:
    """Print tables 2 and 3: one share per use of the batch, per seed.

    Args:
        title: The table's title.
        field: Which count to show: ``"outside"`` or ``"clipped"``.
        blocks: For each block of columns, per seed, the counts of each use.
        labels: One heading per block.
    """
    uses = len(blocks[0][0])
    width = USE_CELL * uses + BLOCK_GAP
    print(title)
    print(seed_row("", [f"{label}, use 1 to {uses}" for label in labels], width))
    numbers = "".join(f"{use:>{USE_CELL}}" for use in range(1, uses + 1))
    print(seed_row("seed", [numbers] * len(blocks), width))
    for seed in range(len(blocks[0])):
        cells = [use_cells(block[seed], field) for block in blocks]
        print(seed_row(seed_label(seed), cells, width))
    totals = [use_cells(pooled_uses(block), field) for block in blocks]
    print(seed_row("all", totals, width))
    print()


def quarter_cells(quarters: Sequence[tuple[int, int]]) -> str:
    """Write one run's share right in each quarter, as a row of numbers.

    Args:
        quarters: For each quarter, (right, rollouts).

    Returns:
        For example ``"  29  68  88  95"``.
    """
    return "".join(
        f"{percent(right, total):>{QUARTER_CELL}}" for right, total in quarters
    )


def print_learning(
    title: str, right: Mapping[str, Sequence[Sequence[tuple[int, int]]]]
) -> None:
    """Print tables 4 and 5: easy and medium rollouts right, per quarter and seed.

    Args:
        title: The table's title.
        right: For each run's name, per seed, per quarter, (right, rollouts).
    """
    print(title)
    names = list(right)
    # "stale 2, recompute-old" is printed as "stale 2," over "recompute-old".
    halves = [name.partition(NAME_BREAK) for name in names]
    first = [head + (NAME_BREAK.strip() if tail else "") for head, _, tail in halves]
    print(seed_row("seed", first, LEARN_WIDTH))
    print(seed_row("", [tail for _, _, tail in halves], LEARN_WIDTH))
    for seed in range(len(right[names[0]])):
        cells = [quarter_cells(right[name][seed]) for name in names]
        print(seed_row(seed_label(seed), cells, LEARN_WIDTH))
    totals = []
    for name in names:
        runs = right[name]
        sums = [
            (sum(run[q][0] for run in runs), sum(run[q][1] for run in runs))
            for q in range(len(runs[0]))
        ]
        totals.append(quarter_cells(sums))
    print(seed_row("all", totals, LEARN_WIDTH))
    print()
