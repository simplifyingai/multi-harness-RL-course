# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""How much of a score is noise: 24 tasks per harness, read honestly.

``baseline_eval.py`` measures; this script reads the measurements. It needs
no model and no stack: it works on the project's own records, bundled in
``data/project_records/``, and on yours when you have them:

1. **the baseline**: each harness's score, and the plain average over the
   three harnesses (each harness weighs the same);
2. **one task's weight**: with 24 tasks, one task moves a score 4.2 points;
3. **the standard error and the Wilson interval** of each score at n = 24;
4. **two identical models**: both solve each task with the same chance, yet
   their scores differ; how often by 8 points or more, simulated (seeded,
   so it repeats exactly) and computed exactly;
5. **the ceiling**: a score of 24/24 can only stay or fall;
6. **the average hides what moved**: offsetting changes cancel in it;
7. **every pair of checkpoints, cell by cell**: gained, lost, flipped and
   the exact p-value, on the cells both ran;
8. **a broken evaluation**: Pi's 0% after SFT and RL was chapter 19's
   sandbox bug, not the model;
9. **the article's own numbers**, in their own box, never mixed with ours.

The bundled records are the project's ``runs/main/eval_*.jsonl`` from its
run of 2026-10-06, cut to the fields read here: ``eval_<label>.jsonl`` holds
72 cells per checkpoint (``base``, ``sft``, ``rl``), and
``eval_<label>-pi-fixed.jsonl`` Pi's 24 cells re-run after the sandbox fix.
The checkpoints as the project reported them take Pi from the re-run.

Your records are read from ``COURSE_HOME/26-grpo-in-harnesses/baseline/
eval_base.jsonl`` when ``baseline_eval.py`` has written it, or from each
``--records FILE``. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from evaluate import Record, read_records, summarize
from install import CLAUDE, HARNESSES, MINI, PI
from lfm_files import chapter_home, short_path
from stats import (
    PERCENT,
    Comparison,
    compare,
    gap_chances,
    pass_at_1,
    simulate_gaps,
    standard_error,
    wilson_interval,
)

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE / "data" / "project_records"
PROJECT_LABELS = ("base", "sft", "rl")
RECORDS_NAME = "eval_{label}.jsonl"
FIXED_NAME = "eval_{label}-pi-fixed.jsonl"
PROJECT_RUN = "the project's run of 2026-10-06"
YOURS = Path("baseline") / "eval_base.jsonl"  # baseline_eval.py's records
YOUR_LABEL = "your {name}"  # never "base" alone: that label is the project's
RECORDS_PREFIX = "eval_"  # baseline_eval.py's files are eval_<label>.jsonl
LABEL_WIDTH = 12
SCORE_COLUMN = 18  # "15/24 = 62.5%" and room
HARNESS_COLUMN = 15  # "mini-swe-agent" and a space
TASKS = 24  # tasks per harness in the project's evaluation
EXAMPLE_RATE = 0.75  # the plan's example: a standard error near 75%
CEILING_RATES = (0.99, 0.95, 0.90)
SIM_PAIRS = 10_000
SIM_SEED = 0
GAP_POINTS = 8  # "differ by 8 points or more", the plan's question
GAP_COLUMNS = 5  # gaps of 0, 1, 2, 3 tasks, then "4 or more"
SIGNIFICANT = 0.05
CHURN_SHARE = 3  # churn: flips that cancel out are at least 1/3 of the cells
RULE = "━━"
WIDTH = 78  # the boxed note's width
NO_ANSWER = "no answer.txt"  # the grader's reason when there is no answer file
# The article's own figures (Hugging Face and Liquid AI, "The ultimate guide
# to multi-harness RL", 2026), on its own test tasks: shown in a box of their
# own and never set in a table beside ours. Its sentences: "With the same
# weights, LFM2.5-2.6B solved 62% of our test tasks under Mini-SWE-Agent and
# 33% under Claude Code." and "On SWE-bench Pro, Joel Niklaus measured the
# same model, GLM-5.2, at 23% in one harness and 52% in another." (it names
# neither harness). The gap is given as a difference, so no number of theirs
# can be read as one of ours.
ARTICLE_TITLE = "the article's own numbers (Hugging Face + Liquid AI)"
ARTICLE_SOURCE = 'from "The ultimate guide to multi-harness RL" (2026)'
ARTICLE_LINES = (
    "On its own test tasks, with the same LFM2.5-2.6B weights, the same "
    "direction of gap: mini-swe-agent ahead of Claude Code by 29 points.",
    "Its benchmark example, measured by Joel Niklaus: one model, GLM-5.2, on "
    "SWE-bench Pro, at 23% in one harness and 52% in another.",
)


@dataclass(frozen=True)
class Checkpoint:
    """One evaluated checkpoint: a label and its verdict per cell.

    Attributes:
        label: Such as ``base`` or ``yours``.
        records: Its records.
    """

    label: str
    records: tuple[Record, ...]

    def verdicts(self, harness: str) -> dict[str, bool]:
        """Each task's verdict in one harness.

        Args:
            harness: The harness.

        Returns:
            ``{task_id: correct}``.
        """
        return {r.task_id: r.correct for r in self.records if r.harness == harness}

    def solved(self, harness: str) -> tuple[int, int]:
        """Solved and run cells in one harness.

        Args:
            harness: The harness.

        Returns:
            ``(solved, cells)``.
        """
        verdicts = self.verdicts(harness)
        return sum(verdicts.values()), len(verdicts)


def project_checkpoint(label: str, folder: Path = PROJECT_DIR) -> Checkpoint:
    """One of the project's checkpoints, as it reported it: Pi from the re-run.

    Args:
        label: ``base``, ``sft`` or ``rl``.
        folder: Where the bundled records are.

    Returns:
        The checkpoint.

    Raises:
        FileNotFoundError: If a bundled file is missing.
    """
    main = read_or_fail(folder / RECORDS_NAME.format(label=label))
    fixed = read_or_fail(folder / FIXED_NAME.format(label=label))
    kept = [r for r in main if r.harness != PI] + fixed
    return Checkpoint(label, tuple(kept))


def read_or_fail(path: Path) -> list[Record]:
    """Read a records file that must be there.

    Args:
        path: The file.

    Returns:
        Its records.

    Raises:
        FileNotFoundError: If it is missing.
        ValueError: If it holds no records, or a broken line.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no file {short_path(path)}")
    records = read_records(path)
    if not records:
        raise ValueError(f"no records in {short_path(path)}")
    return records


def your_checkpoints(files: list[Path]) -> list[Checkpoint]:
    """Your own records: each ``--records`` file, or baseline_eval.py's.

    Args:
        files: The ``--records`` files ([] for baseline_eval.py's default).

    Returns:
        One checkpoint per file that holds records.

    Raises:
        FileNotFoundError: If a file named with ``--records`` is missing.
        ValueError: If one holds no records, or a broken line.
    """
    if not files:
        default = chapter_home() / YOURS
        records = read_records(default)
        if not records:
            return []
        return [Checkpoint(your_label(default), tuple(records))]
    checkpoints = []
    for path in files:
        label = your_label(path)
        taken = sum(c.label.startswith(label) for c in checkpoints)
        if taken:  # two files with one name: number the later ones
            label = f"{label} ({taken + 1})"
        checkpoints.append(Checkpoint(label, tuple(read_or_fail(path))))
    return checkpoints


def your_label(path: Path) -> str:
    """Name one of your records files as a checkpoint.

    Args:
        path: Such as ``eval_base.jsonl``.

    Returns:
        Such as ``your base``.
    """
    return YOUR_LABEL.format(name=path.stem.removeprefix(RECORDS_PREFIX))


def label_width(checkpoints: list[Checkpoint]) -> int:
    """The width of a column of checkpoint labels.

    Args:
        checkpoints: The checkpoints.

    Returns:
        The longest label plus two spaces, at least the project's 12.
    """
    return max([LABEL_WIDTH, *(len(c.label) + 2 for c in checkpoints)])


def heading(title: str, say: Callable[[str], None]) -> None:
    """Print a section's heading.

    Args:
        title: Its words.
        say: Where lines go.
    """
    say("")
    say(f"{RULE} {title}")


def score(solved: int, cells: int) -> str:
    """Write a score as a fraction and a percentage.

    Args:
        solved: Cells solved.
        cells: Cells run.

    Returns:
        Such as ``15/24 = 62.5%``, or ``none`` when nothing ran.
    """
    if not cells:
        return "none"
    return f"{solved}/{cells} = {pass_at_1(solved, cells):.1f}%"


def harnesses_of(checkpoints: list[Checkpoint]) -> list[str]:
    """The harnesses any checkpoint ran, in the course's order.

    Args:
        checkpoints: The checkpoints.

    Returns:
        Harness names.
    """
    seen = {r.harness for c in checkpoints for r in c.records}
    return [h for h in HARNESSES if h in seen] + sorted(seen - set(HARNESSES))


def show_scores(
    checkpoints: list[Checkpoint],
    title: str,
    say: Callable[[str], None],
    explain: bool = True,
) -> None:
    """Print each checkpoint's score per harness and the plain average.

    Args:
        checkpoints: The checkpoints, in display order.
        title: The section's heading.
        say: Where lines go.
        explain: Also work out the first checkpoint's average.
    """
    harnesses = harnesses_of(checkpoints)
    width = label_width(checkpoints)
    heading(title, say)
    columns = "".join(f"{h:<{SCORE_COLUMN}}" for h in harnesses)
    say(f"  {'checkpoint':<{width}}{columns}average")
    for checkpoint in checkpoints:
        summary = summarize(list(checkpoint.records), harnesses)
        cells = "".join(
            f"{score(*checkpoint.solved(h)):<{SCORE_COLUMN}}" for h in harnesses
        )
        average = summary["average_accuracy"]
        say(f"  {checkpoint.label:<{width}}{cells}{average:.1f}%")
    if not explain:
        return
    first = checkpoints[0]
    summary = summarize(list(first.records), harnesses)
    parts = [
        f"{accuracy(summary, h):.1f}"
        for h in harnesses
        if summary["per_harness"][h]["cells"]
    ]
    say(
        f"  {first.label}'s average is the plain mean over harnesses, each "
        "weighing the same:"
    )
    say(f"  ({' + '.join(parts)}) / {len(parts)} = {summary['average_accuracy']:.1f}%")


def show_weight(say: Callable[[str], None]) -> None:
    """Print what one task is worth in a score of 24.

    Args:
        say: Where lines go.
    """
    heading("one task's weight", say)
    weight = PERCENT / TASKS
    say(f"  {TASKS} tasks per harness: one task = 100 / {TASKS} = {weight:.1f} points")


def show_intervals(
    checkpoint: Checkpoint, say: Callable[[str], None], explain: bool = True
) -> None:
    """Print each harness's standard error and Wilson interval.

    Args:
        checkpoint: The checkpoint.
        say: Where lines go.
        explain: Also print the formula and the plan's example near 75%.
    """
    harnesses = harnesses_of([checkpoint])
    sizes = sorted({checkpoint.solved(h)[1] for h in harnesses})
    size = "/".join(str(n) for n in sizes)
    title = f"standard error and Wilson 95% interval, n = {size} ({checkpoint.label})"
    heading(title, say)
    if explain:
        example = standard_error(round(EXAMPLE_RATE * TASKS), TASKS)
        formula = f"sqrt({EXAMPLE_RATE} x {1 - EXAMPLE_RATE} / {TASKS})"
        say(
            f"  standard error = sqrt(p x (1 - p) / n); near {EXAMPLE_RATE:.0%}: "
            f"{formula} = {example:.1f} points"
        )
    say("  harness         solved  pass@1  standard error  Wilson 95% interval")
    perfect = False
    for harness in harnesses:
        solved, cells = checkpoint.solved(harness)
        perfect = perfect or (cells > 0 and solved == cells)
        low, high = wilson_interval(solved, cells)
        say(
            f"  {harness:<{HARNESS_COLUMN}} {solved:>2}/{cells:<3}  "
            f"{pass_at_1(solved, cells):5.1f}%"
            f"  ± {standard_error(solved, cells):4.1f} points    "
            f"{low:5.1f}% to {high:5.1f}%"
        )
    if perfect and explain:
        say(
            "  a perfect score has a standard error of 0, as if 100% were certain;"
            " Wilson's"
        )
        say(
            f"  interval at {TASKS}/{TASKS} still allows a true rate down to "
            f"{wilson_interval(TASKS, TASKS)[0]:.1f}%"
        )


def show_identical(
    rate: float, pairs: int, seed: int, say: Callable[[str], None]
) -> None:
    """Print how far two identical models' scores fall apart by luck.

    Args:
        rate: The chance both have of solving each task.
        pairs: How many pairs of runs to simulate.
        seed: The random generator's seed.
        say: Where lines go.
    """
    heading(f"two identical models (simulated, seed {seed})", say)
    say(
        f"  both solve each task with chance {rate:.0%}; {TASKS} tasks each; "
        f"{pairs:,} pairs of runs"
    )
    counts = simulate_gaps(TASKS, rate, pairs, seed)
    exact = gap_chances(TASKS, rate)
    last = GAP_COLUMNS - 1
    names = [f"{g}" for g in range(last)] + [f"{last}+"]
    say("  gap in tasks  " + "".join(f"{n:>8}" for n in names))
    sim = [c / pairs for c in counts[:last]] + [sum(counts[last:]) / pairs]
    calc = exact[:last] + [sum(exact[last:])]
    say("  simulated     " + "".join(f"{s:>8.1%}" for s in sim))
    say("  exact         " + "".join(f"{e:>8.1%}" for e in calc))
    tasks = math.ceil(GAP_POINTS * TASKS / 100)  # 8 points need 2 whole tasks
    wide_sim = sum(counts[tasks:]) / pairs
    say(
        f"  they differ by {GAP_POINTS} points ({tasks} tasks) or more in "
        f"{wide_sim:.1%} of pairs (exact: {sum(exact[tasks:]):.1%})"
    )
    say(
        "  one run with one seed is one draw from this table: a gap of a task or two"
        "\n  between checkpoints is what identical models show most of the time"
    )


def show_ceiling(checkpoints: list[Checkpoint], say: Callable[[str], None]) -> None:
    """Print why a perfect score can only stay or fall.

    Args:
        checkpoints: The first checkpoint, whose perfect scores are read,
            then the later ones.
        say: Where lines go.
    """
    first, later = checkpoints[0], checkpoints[1:]
    heading("the ceiling at 100%", say)
    perfect = [h for h in harnesses_of([first]) if first.solved(h) == (TASKS, TASKS)]
    for harness in perfect:
        say(
            f"  {harness} solved {TASKS}/{TASKS} at {first.label}: from there a "
            "later checkpoint can only tie or lose"
        )
        scores = ", ".join(f"{c.label} {score(*c.solved(harness))}" for c in later)
        if scores:
            say(f"  later: {scores}; a gain there could never have been measured")
    chances = " · ".join(f"{rate:.0%}: {rate**TASKS:.1%}" for rate in CEILING_RATES)
    say(f"  chance of {TASKS}/{TASKS} if the true rate were {chances}")
    low = wilson_interval(TASKS, TASKS)[0]
    say(
        f"  so {TASKS}/{TASKS} says the rate is above about {low:.0f}% (Wilson), "
        "not that it is 100%"
    )


def accuracy(summary: dict[str, Any], harness: str) -> float:
    """Read one harness's accuracy from a summary.

    Args:
        summary: ``summarize``'s result.
        harness: The harness.

    Returns:
        Its accuracy, in percent.
    """
    return float(summary["per_harness"][harness]["accuracy"])


def show_average_hides(
    before: Checkpoint, after: Checkpoint, say: Callable[[str], None]
) -> None:
    """Print how a near-flat average can hide large moves per harness.

    Args:
        before: The earlier checkpoint.
        after: The later one.
        say: Where lines go.
    """
    harnesses = harnesses_of([before, after])
    old = summarize(list(before.records), harnesses)
    new = summarize(list(after.records), harnesses)
    heading("the average hides what moved", say)
    shift = new["average_accuracy"] - old["average_accuracy"]
    moves = ", ".join(
        f"{h} {accuracy(new, h) - accuracy(old, h):+.1f}" for h in harnesses
    )
    say(
        f"  {before.label} -> {after.label}: average {old['average_accuracy']:.1f}% "
        f"-> {new['average_accuracy']:.1f}% ({shift:+.1f} points), while"
    )
    say(f"  {moves} points")


def reading(comparison: Comparison) -> str:
    """Say in words what a cell-by-cell comparison shows.

    Args:
        comparison: The comparison.

    Returns:
        A few words.
    """
    if not comparison.flipped:
        return "nothing changed"
    if comparison.p_value < SIGNIFICANT:
        side = "lost" if len(comparison.lost) > len(comparison.gained) else "gained"
        return f"{side}: beyond noise"
    return "within noise"


def cancelled(comparison: Comparison) -> int:
    """Count the flips that cancel each other out in the score.

    Args:
        comparison: The comparison.

    Returns:
        Twice the smaller of gained and lost.
    """
    return 2 * min(len(comparison.gained), len(comparison.lost))


def churned(comparison: Comparison, cells: int) -> bool:
    """Say whether many cells flipped while the score barely moved.

    Args:
        comparison: The comparison.
        cells: The cells both checkpoints ran.

    Returns:
        True when the flips that cancel out are at least a third of the cells.
    """
    return cancelled(comparison) * CHURN_SHARE >= cells


def checkpoint_pairs(
    project: list[Checkpoint], yours: list[Checkpoint]
) -> list[tuple[Checkpoint, Checkpoint]]:
    """Every pair of checkpoints: the project's first, then each of yours.

    Args:
        project: The project's checkpoints, in training order.
        yours: Your checkpoints.

    Returns:
        ``(earlier, later)`` pairs.
    """
    pairs = list(combinations(project, 2))
    pairs += [(theirs, mine) for mine in yours for theirs in project]
    return pairs + list(combinations(yours, 2))


def show_pairs(
    pairs: list[tuple[Checkpoint, Checkpoint]], say: Callable[[str], None]
) -> None:
    """Compare checkpoints, pair by pair, cell by cell, on the cells both ran.

    Args:
        pairs: ``(earlier, later)`` checkpoints.
        say: Where lines go.
    """
    names = [f"{first.label} -> {second.label}" for first, second in pairs]
    width = max([len("pair"), *(len(name) for name in names)]) + 1
    heading("every pair of checkpoints, cell by cell (exact McNemar p)", say)
    say(
        f"  {'pair':<{width}} harness         cells  before  after  gained  lost  "
        "flipped  p      reading"
    )
    churn = []
    for name, (first, second) in zip(names, pairs, strict=True):
        for harness in harnesses_of([first, second]):
            one, two = first.verdicts(harness), second.verdicts(harness)
            shared = sorted(set(one) & set(two))
            if not shared:
                continue
            before = {t: one[t] for t in shared}
            after = {t: two[t] for t in shared}
            comparison = compare(before, after)
            old, new = sum(before.values()), sum(after.values())
            say(
                f"  {name:<{width}} {harness:<{HARNESS_COLUMN}} {len(shared):>5}  "
                f"{old:>6}  {new:>5}"
                f"  {len(comparison.gained):>6}  {len(comparison.lost):>4}"
                f"  {comparison.flipped:>7}  {comparison.p_value:.3f}  "
                f"{reading(comparison)}"
            )
            if churned(comparison, len(shared)):
                churn.append((name, harness, comparison, len(shared), new - old))
    say(f"  p < {SIGNIFICANT} reads as beyond noise; anything else, within it")
    for pair, harness, comparison, cells, moved in churn:
        say(
            f"  churn: {harness} {pair}: {comparison.flipped} of {cells} cells "
            f"flipped, {cancelled(comparison)} of them cancel out: score {moved:+d}"
        )


def show_broken(folder: Path, say: Callable[[str], None]) -> None:
    """Print Pi's scores before and after chapter 19's sandbox fix.

    Args:
        folder: Where the bundled records are.
        say: Where lines go.
    """
    heading("when the evaluation itself is broken (chapter 19's sandbox)", say)
    say("  pi, from the project's records, before and after the sandbox fix")
    no_answer = failed = 0
    for label in PROJECT_LABELS:
        before = [
            r
            for r in read_or_fail(folder / RECORDS_NAME.format(label=label))
            if r.harness == PI
        ]
        after = read_or_fail(folder / FIXED_NAME.format(label=label))
        failed += sum(not r.correct for r in before)
        no_answer += sum(r.reason == NO_ANSWER for r in before if not r.correct)
        old = sum(r.correct for r in before)
        new = sum(r.correct for r in after)
        say(f"  {label:<5} {old:>2}/{len(before)} -> {new:>2}/{len(after)}")
    say(f"  before the fix, {no_answer} of {failed} failed pi cells say '{NO_ANSWER}':")
    say("  the first box broke Pi's write tool (chapter 19), so the 0% after SFT and")
    say("  RL measured the box, not the model")


def show_article(ours: Checkpoint, say: Callable[[str], None]) -> None:
    """Print the article's numbers in a box of their own, then ours apart.

    Args:
        ours: The checkpoint whose harness gap is set beside the box.
        say: Where lines go.
    """
    heading("a score names its harness: the article's numbers, kept apart", say)
    inner = WIDTH - 4
    say("  ┌─ " + f"{ARTICLE_TITLE} ".ljust(inner, "─") + "┐")
    lines = [ARTICLE_SOURCE, *ARTICLE_LINES]
    for line in lines:
        for part in textwrap.wrap(line, inner, break_on_hyphens=False):
            say(f"  │ {part.ljust(inner)} │")
    say("  └" + "─" * (inner + 2) + "┘")
    mini = pass_at_1(*ours.solved(MINI))
    claude = pass_at_1(*ours.solved(CLAUDE))
    say(
        f"  ours ({PROJECT_RUN}, {ours.label}): {MINI} ahead of "
        f"{CLAUDE} by {mini - claude:.1f} points"
    )


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records",
        type=Path,
        action="append",
        default=[],
        help="a JSON-lines file of records (baseline_eval.py's); repeatable",
    )
    parser.add_argument("--rate", type=float, default=EXAMPLE_RATE)
    parser.add_argument("--pairs", type=int, default=SIM_PAIRS)
    parser.add_argument("--seed", type=int, default=SIM_SEED)
    args = parser.parse_args()
    if not 0 < args.rate < 1:
        parser.error("--rate: a chance between 0 and 1")
    if args.pairs < 1:
        parser.error("--pairs: at least 1")
    return args


def main() -> None:
    """Print every reading, in order.

    Raises:
        SystemExit: If a bundled or named records file is missing or broken.
    """
    args = parse_args()
    say = print
    try:
        project = [project_checkpoint(label) for label in PROJECT_LABELS]
        yours = your_checkpoints(args.records)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    base = project[0]
    say("noise.py · how much of a score is noise")
    say(f"  records   {PROJECT_RUN} ({PROJECT_DIR.relative_to(HERE)}/)")
    say(
        f"            base, sft, rl: {len(base.records)} cells each; pi from its "
        "re-run after chapter 19's sandbox fix"
    )
    if yours:
        names = ", ".join(f"{c.label} ({len(c.records)} cells)" for c in yours)
        say(f"  yours     {names}")
    else:
        say(
            f"  yours     none yet in {short_path(chapter_home() / YOURS)} "
            "(uv run baseline_eval.py writes it)"
        )
    show_scores([base], "the baseline (the project's base model)", say)
    show_weight(say)
    show_intervals(base, say)
    show_identical(args.rate, args.pairs, args.seed, say)
    show_ceiling(project, say)
    show_average_hides(project[0], project[1], say)
    if yours:
        show_scores([base, *yours], "your runs beside the project's base", say, False)
        for checkpoint in yours:
            show_intervals(checkpoint, say, explain=False)
    show_pairs(checkpoint_pairs(project, yours), say)
    show_broken(PROJECT_DIR, say)
    show_article(base, say)


if __name__ == "__main__":
    main()
