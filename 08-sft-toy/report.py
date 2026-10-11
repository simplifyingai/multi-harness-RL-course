"""How this chapter prints: demonstrations, SFT steps, the tests, RL, the logs.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``sft_toy.py`` can stay about the teacher, the loss and
the loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from evaluate import SAMPLES, LevelResult
from grpo import GROUP_SIZE, QUESTIONS_PER_STEP
from knobs import (
    BATCH_SIZE,
    DEFAULT_LR,
    DEMO_LEVEL,
    DEMONSTRATIONS,
    EPOCHS,
    HIGH_LR,
    RL_LR,
)
from policy import action_scores, softmax
from questions import LEVELS, MEDIUM
from rl import RlStep
from rollouts import Rollout
from sft import Demonstration, StepLog
from teacher import FIXED_PATH, PATTERNS, TEACHER_NOTES

if TYPE_CHECKING:  # only for type hints: sft_toy.py imports this module
    from sft_toy import RunResult, Settings

PERCENT = 100
LOSS_DECIMALS = 2
P_DECIMALS = 2
STEP_WIDTH = 2  # "step  7"
ACTION_WIDTH = 9  # "QUERY_A  "
LEVEL_WIDTH = 10  # "medium    "
CELL_WIDTH = 10  # one column of the test table
HABIT_WIDTH = 32  # the name column of the habits table
HABIT_CELL = 9
QUARTER_NAME = 24  # the name column of the RL quarter table
QUARTER_CELL = 10
STAGE_CELL = 16  # one column of the pipeline table
INDENT = "  "
FIRST_RL_STEPS = 10  # RL prints its first steps, then the quarters
QUARTERS = 4
FOURTH_CALL = 4  # the fixed-length teacher's FINISH comes on call 4
SEED_WIDTH = 6  # " 0    "
ARROW = " -> "
PATH_ORDER = (*PATTERNS, FIXED_PATH)


def share(part: float, total: float) -> str:
    """Write a part of a total as a whole percentage.

    Args:
        part: The part.
        total: The total (0 gives 0%).

    Returns:
        For example ``"19%"``.
    """
    return f"{round(PERCENT * part / total) if total else 0}%"


def row(name: str, cells: Sequence[str], name_width: int, cell_width: int) -> str:
    """Lay out one table row: a name column, then fixed-width cells.

    Args:
        name: The row's name (may be empty).
        cells: The cells.
        name_width: The name column's width.
        cell_width: Each cell's width.

    Returns:
        The row, without trailing spaces.
    """
    line = f"{name:<{name_width}}" + "".join(f"{c:<{cell_width}}" for c in cells)
    return line.rstrip()


def print_header(
    settings: Settings,
    demo_questions: int,
    test_cities: int,
    by_level: dict[str, int],
    looks: dict[str, int],
) -> None:
    """Print what this run does, before it does it.

    Args:
        settings: The run's knobs.
        demo_questions: How many questions the teacher demonstrates.
        test_cities: How many cities the held-out questions are about.
        by_level: How many held-out questions each level has.
        looks: How many LOOKs each level's query needs first.
    """
    epochs = "epoch" if EPOCHS == 1 else "epochs"
    print(
        f"sft_toy · teacher {settings.teacher} · seed {settings.seed}"
        f" · lr {settings.lr:g} · {DEMONSTRATIONS} demonstrations"
        f" · batch {BATCH_SIZE} · {EPOCHS} {epochs}"
    )
    print(f"teacher: {TEACHER_NOTES[settings.teacher]}")
    each, extra = divmod(DEMONSTRATIONS, demo_questions)
    turns = f"{each} each" if not extra else "taken in turn"
    print(
        f"demonstrations: the {demo_questions} {DEMO_LEVEL} training questions,"
        f" {turns}, in a shuffled order"
    )
    calls = "calls used as a share" if settings.calls_as_share else "the call number"
    print(f"student: chapter 4's softmax policy, one set of weights; it sees {calls}")
    print(
        "loss per call = -log p(teacher's action); what the environment sends"
        " back is never trained on"
    )
    levels = ", ".join(f"{count} {level}" for level, count in by_level.items())
    total = sum(by_level.values())
    print(f"held out: {total} questions about {test_cities} other cities: {levels}")
    needs = ", ".join(f"{level} {count}" for level, count in looks.items())
    print(f"{INDENT}LOOKs a question needs before its query works: {needs}")
    print()


def print_demonstration(demo: Demonstration) -> None:
    """Print one demonstration, call by call.

    Args:
        demo: The demonstration.
    """
    print(f"demonstration {demo.number} ({demo.path}): {demo.question.task.question}")
    for call in demo.calls:
        print(
            f"{INDENT}call {call.number}  {call.action:<{ACTION_WIDTH}}"
            f"{call.observation}"
        )


def print_demonstrations(demos: Sequence[Demonstration]) -> None:
    """Print the first demonstration of each path the teacher took.

    Args:
        demos: Every demonstration, in order.
    """
    print("the first demonstration of each path")
    print(
        "(left: the teacher's action, trained; right: what came back, read and"
        " never trained on)"
    )
    seen: set[str] = set()
    for demo in demos:
        if demo.path not in seen:
            seen.add(demo.path)
            print_demonstration(demo)
    print()


def print_rows(demos: Sequence[Demonstration], rows: int, trained: int) -> None:
    """Print what the demonstrations came to as training rows.

    Args:
        demos: Every demonstration.
        rows: How many rows the trainer kept.
        trained: How many of the teacher's actions those rows hold.
    """
    right = sum(demo.correct for demo in demos)
    print(
        f"{len(demos)} demonstrations, {right} right: {rows} rows"
        " (the trainer keeps right ones only)"
    )
    counts = {path: sum(d.path == path for d in demos) for path in PATH_ORDER}
    paths = ", ".join(f"{count} {path}" for path, count in counts.items() if count)
    print(f"paths: {paths}")
    print(
        f"trained: {trained} actions, the teacher's; never trained on: {len(demos)}"
        f" questions and {trained} replies"
    )
    print()


def print_sft_step(log: StepLog) -> None:
    """Print one SFT step's line.

    Args:
        log: The step's log.
    """
    print(
        f"step {log.step:>{STEP_WIDTH}} · loss per call"
        f" {log.loss_per_call:.{LOSS_DECIMALS}f} · {log.calls} calls in"
        f" {log.rows} rows"
    )


def level_cells(result: LevelResult) -> tuple[str, str]:
    """Write one level's greedy and sampled result.

    Args:
        result: The level's result.

    Returns:
        For example ``"8 of 8"`` and ``"74%"``.
    """
    return (
        f"{result.greedy_right} of {result.questions}",
        share(result.sampled_right, result.sampled),
    )


def print_tests(columns: Sequence[tuple[str, Sequence[LevelResult]]]) -> None:
    """Print the held-out test of several sets of weights side by side.

    Args:
        columns: For each set of weights, a heading and its results per level.
    """
    print(
        "held-out test: greedy (one rollout per question) and sampled"
        f" ({SAMPLES} per question, temperature 1)"
    )
    print(row("", [name for name, _ in columns], LEVEL_WIDTH, 2 * CELL_WIDTH))
    heads = ["greedy", "sampled"] * len(columns)
    print(row("", heads, LEVEL_WIDTH, CELL_WIDTH))
    for index, level in enumerate(LEVELS):
        cells = [cell for _, results in columns for cell in level_cells(results[index])]
        print(row(level, cells, LEVEL_WIDTH, CELL_WIDTH))
    print()


def print_trace(rollout: Rollout, weights: np.ndarray, title: str) -> None:
    """Print one greedy rollout call by call, with each action's probability.

    Args:
        rollout: The rollout.
        weights: The weights that acted.
        title: What the rollout shows.
    """
    print(title)
    print(rollout.question.task.question)
    for call, situation, choice in zip(
        rollout.calls, rollout.situations, rollout.choices, strict=True
    ):
        chance = softmax(action_scores(weights, situation))[choice]
        print(
            f"{INDENT}call {call.number}  {call.action:<{ACTION_WIDTH}}"
            f"p {chance:.{P_DECIMALS}f}  {call.observation}"
        )
    print()


HABITS: tuple[tuple[str, Callable[[LevelResult], str]], ...] = (
    ("read answer.txt back (CHECK)", lambda r: share(r.checked, r.sampled)),
    ("said FINISH with no answer.txt", lambda r: share(r.unsaved_total, r.sampled)),
    (
        f"{INDENT}of them on call {FOURTH_CALL}",
        lambda r: share(r.unsaved[FOURTH_CALL], r.sampled),
    ),
    ("calls per rollout", lambda r: f"{r.calls / r.sampled:.1f}"),
)


def print_habits(results: Sequence[LevelResult], title: str) -> None:
    """Print the habits the sampled test rollouts showed, level by level.

    Args:
        results: The test's results per level.
        title: The table's name.
    """
    levels = [result.level for result in results]
    print(row(title, levels, HABIT_WIDTH, HABIT_CELL))
    for name, cell in HABITS:
        print(row(name, [cell(r) for r in results], HABIT_WIDTH, HABIT_CELL))
    print()


def print_rl_header(questions: int, steps: int, lr: float) -> None:
    """Say what the RL stage does.

    Args:
        questions: How many questions it trains on.
        steps: How many steps.
        lr: Its learning rate.
    """
    print(
        f"then RL on the {questions} hard training questions: chapter 6's GRPO,"
        f" {steps} steps of {QUESTIONS_PER_STEP} questions x {GROUP_SIZE}"
        f" rollouts, lr {lr:g}"
    )
    print(
        "from scratch (every weight 0) and from the imitator (the weights SFT"
        " just trained): the same questions, in the same order"
    )
    print()


def rl_cell(step: RlStep) -> str:
    """Write one start's step compactly.

    Args:
        step: The step.

    Returns:
        For example ``"right 3/8, signal 1/2"``.
    """
    right = sum(rollout.reward for rollout in step.rollouts)
    signal = len(step.groups) - step.no_signal
    return f"right {right}/{len(step.rollouts)}, signal {signal}/{len(step.groups)}"


def print_rl_first_steps(scratch: Sequence[RlStep], imitator: Sequence[RlStep]) -> None:
    """Print the first RL steps of both starts side by side.

    Args:
        scratch: Every step from scratch.
        imitator: Every step from the imitator.
    """
    pairs = list(zip(scratch, imitator, strict=True))[:FIRST_RL_STEPS]
    for mine, theirs in pairs:
        print(
            f"step {mine.number:>{STEP_WIDTH}} · from scratch {rl_cell(mine)}"
            f" · from the imitator {rl_cell(theirs)}"
        )
    print()


def quarter_edges(steps: int) -> list[tuple[int, int]]:
    """Split a run into (at most) four stretches of steps.

    Args:
        steps: How many steps (at least 1; the flags make sure).

    Returns:
        Each stretch's first and last step, e.g. ``(1, 25)``.
    """
    size = math.ceil(steps / QUARTERS)
    return [(start + 1, min(start + size, steps)) for start in range(0, steps, size)]


def right_cell(part: Sequence[RlStep]) -> str:
    """Write the share of a stretch's rollouts that were right.

    Args:
        part: The stretch's steps.

    Returns:
        For example ``"54%"``.
    """
    rollouts = [rollout for step in part for rollout in step.rollouts]
    return share(sum(r.reward for r in rollouts), len(rollouts))


def signal_cell(part: Sequence[RlStep]) -> str:
    """Write how many of a stretch's groups had contrast.

    Args:
        part: The stretch's steps.

    Returns:
        For example ``"37/50"``.
    """
    groups = sum(len(step.groups) for step in part)
    return f"{groups - sum(step.no_signal for step in part)}/{groups}"


def print_rl_quarters(scratch: Sequence[RlStep], imitator: Sequence[RlStep]) -> None:
    """Print both starts' rollouts right and groups with signal, by quarter.

    Args:
        scratch: Every step from scratch.
        imitator: Every step from the imitator.
    """
    edges = quarter_edges(len(scratch))
    heads = [f"{first}-{last}" for first, last in edges]
    print(row("in each quarter: steps", heads, QUARTER_NAME, QUARTER_CELL))
    tables = (("rollouts right", right_cell), ("groups with signal", signal_cell))
    starts = (("from scratch", scratch), ("from the imitator", imitator))
    for title, cell in tables:
        print(title)
        for name, steps in starts:
            cells = [cell(steps[first - 1 : last]) for first, last in edges]
            print(row(f"{INDENT}{name}", cells, QUARTER_NAME, QUARTER_CELL))
    print()


def print_pipeline(columns: Sequence[tuple[str, Sequence[LevelResult]]]) -> None:
    """Print the held-out test after each stage, greedy and sampled in one cell.

    Args:
        columns: For each stage, a heading and its results per level.
    """
    print(
        "held-out test after each stage: greedy right, then sampled right"
        f" ({SAMPLES} per question)"
    )
    print(row("", [name for name, _ in columns], LEVEL_WIDTH, STAGE_CELL))
    for index, level in enumerate(LEVELS):
        cells = [", ".join(level_cells(results[index])) for _, results in columns]
        print(row(level, cells, LEVEL_WIDTH, STAGE_CELL))
    print()


def print_rl(result: RunResult) -> None:
    """Print the RL stage: its first steps, its quarters and the pipeline.

    Args:
        result: The finished run (with ``--then-rl``).
    """
    rl = result.rl
    if rl is None:
        return
    print_rl_header(rl.questions, len(rl.scratch), RL_LR)
    print_rl_first_steps(rl.scratch, rl.imitator)
    print_rl_quarters(rl.scratch, rl.imitator)
    print_pipeline(
        (
            ("untrained", result.before),
            ("SFT", result.after),
            ("SFT, then RL", rl.after_imitator),
            ("RL from scratch", rl.after_scratch),
        )
    )
    print_habits(rl.after_imitator, "habits after SFT, then RL")


def print_run(result: RunResult) -> None:
    """Print everything after the SFT steps: tests, a trace, habits, RL.

    Args:
        result: The finished run.
    """
    print()
    print_tests((("untrained", result.before), ("after SFT", result.after)))
    medium = next(r for r in result.after if r.level == MEDIUM)
    print_trace(
        medium.trace,
        result.weights,
        "the student, greedy, on a medium question, which needs one LOOK more"
        f" than any {DEMO_LEVEL} question it was shown",
    )
    print_habits(result.after, "habits after SFT, sampled")
    print_rl(result)


def print_logs(paths: Sequence[tuple[str, str]], metrics: str) -> None:
    """Print where the logs went, their md5s and how to redraw the run.

    Args:
        paths: Each log's path as text, and its md5.
        metrics: Where metrics.jsonl went, as text.
    """
    for path, md5 in paths:
        print(f"wrote {path}")
        print(f"md5   {md5}")
    print(f"redraw it: uv run plot.py {metrics}")


# --- seeds.py -----------------------------------------------------------------


@dataclass(frozen=True)
class LevelCounts:
    """One level of a held-out test, counted (small, so it travels cheaply).

    Attributes:
        level: ``"easy"``, ``"medium"`` or ``"hard"``.
        questions: How many held-out questions the level has.
        greedy_right: How many the greedy student got right.
        greedy_checked: How many greedy rollouts read answer.txt back.
        greedy_unsaved_fourth: How many greedy rollouts said FINISH on call 4
            with no answer.txt.
        sampled: How many sampled rollouts there were.
        sampled_right: How many of them were right.
        checked: How many of them read answer.txt back.
        unsaved_fourth: How many said FINISH on call 4 with no answer.txt.
    """

    level: str
    questions: int
    greedy_right: int
    greedy_checked: int
    greedy_unsaved_fourth: int
    sampled: int
    sampled_right: int
    checked: int
    unsaved_fourth: int

    @classmethod
    def of(cls, result: LevelResult) -> LevelCounts:
        """Count one level's test result.

        Args:
            result: The level's result.

        Returns:
            Its counts.
        """
        return cls(
            level=result.level,
            questions=result.questions,
            greedy_right=result.greedy_right,
            greedy_checked=result.greedy_checked,
            greedy_unsaved_fourth=result.greedy_unsaved[FOURTH_CALL],
            sampled=result.sampled,
            sampled_right=result.sampled_right,
            checked=result.checked,
            unsaved_fourth=result.unsaved[FOURTH_CALL],
        )


class Quarter(NamedTuple):
    """One quarter of an RL run, counted.

    Attributes:
        right: Rollouts right.
        rollouts: Rollouts run.
        signal: Groups with contrast.
        groups: Groups run.
    """

    right: int
    rollouts: int
    signal: int
    groups: int


def quarter_of(part: Sequence[RlStep]) -> Quarter:
    """Count one stretch of RL steps.

    Args:
        part: The stretch's steps.

    Returns:
        Its counts.
    """
    return Quarter(
        right=sum(r.reward for step in part for r in step.rollouts),
        rollouts=sum(len(step.rollouts) for step in part),
        signal=sum(len(step.groups) - step.no_signal for step in part),
        groups=sum(len(step.groups) for step in part),
    )


@dataclass(frozen=True)
class RlCounts:
    """Both starts of an RL run, counted by quarter.

    Attributes:
        scratch: From scratch, one count per quarter.
        imitator: From the imitator, one count per quarter.
        imitator_first_signal: Groups with contrast at the imitator's step 1.
    """

    scratch: tuple[Quarter, ...]
    imitator: tuple[Quarter, ...]
    imitator_first_signal: int

    @classmethod
    def of(cls, scratch: Sequence[RlStep], imitator: Sequence[RlStep]) -> RlCounts:
        """Count both starts by quarter.

        Args:
            scratch: Every step from scratch (at least one).
            imitator: Every step from the imitator, as many.

        Returns:
            The counts.
        """
        edges = quarter_edges(len(scratch))
        first = imitator[0]
        return cls(
            scratch=tuple(quarter_of(scratch[a - 1 : b]) for a, b in edges),
            imitator=tuple(quarter_of(imitator[a - 1 : b]) for a, b in edges),
            imitator_first_signal=len(first.groups) - first.no_signal,
        )

    @staticmethod
    def totals(quarters: Sequence[Quarter]) -> Quarter:
        """Add quarters together.

        Args:
            quarters: Some quarters (at least one).

        Returns:
            Their sum.
        """
        return Quarter(*(sum(values) for values in zip(*quarters, strict=True)))

    def imitator_ahead(self) -> bool:
        """Whether the imitator had more rollouts right in every quarter.

        Returns:
            True when it did.
        """
        return all(
            mine.right > theirs.right
            for mine, theirs in zip(self.imitator, self.scratch, strict=True)
        )


@dataclass(frozen=True)
class SeedRun:
    """What one run contributes to the seed tables.

    Attributes:
        name: Which of the ways it was run.
        seed: Its seed.
        losses: The SFT loss per call at every step.
        levels: The held-out test after SFT, per level.
        rl: The RL stage, counted, for the runs that had one.
    """

    name: str
    seed: int
    losses: tuple[float, ...]
    levels: tuple[LevelCounts, ...]
    rl: RlCounts | None

    @property
    def sampled_right(self) -> int:
        """Sampled held-out rollouts right, every level."""
        return sum(level.sampled_right for level in self.levels)

    @property
    def sampled_checked(self) -> int:
        """Sampled held-out rollouts that read answer.txt back, every level."""
        return sum(level.checked for level in self.levels)

    @property
    def greedy_checked(self) -> int:
        """Greedy held-out rollouts that read answer.txt back, every level."""
        return sum(level.greedy_checked for level in self.levels)


def print_seeds_header(seeds: int, rl_steps: int) -> None:
    """Print what ``seeds.py`` runs and how to read its tables.

    Args:
        seeds: How many seeds.
        rl_steps: GRPO steps for each start.
    """
    print(
        f"seeds 0-{seeds - 1} · {DEMONSTRATIONS} demonstrations, batch"
        f" {BATCH_SIZE}, one epoch · lr {DEFAULT_LR:g} (high: {HIGH_LR:g})"
        f" · RL {rl_steps} steps"
    )
    print(
        f"held-out test after SFT: easy, medium, hard = sampled rollouts right"
        f" ({SAMPLES} per question); medium greedy = greedy rollouts right"
    )
    print(
        f"call {FOURTH_CALL} unsaved = sampled medium rollouts that said FINISH on"
        f" call {FOURTH_CALL} with no answer.txt; checked = sampled rollouts"
        " that read answer.txt back"
    )
    print()


SEED_HEADS = (
    "loss",
    "easy",
    "medium",
    "hard",
    "medium greedy",
    f"call {FOURTH_CALL} unsaved",
    "checked",
)
# The columns' widths, one per SEED_HEADS entry (seed_line zips them strictly).
SEED_CELLS = (14, 7, 8, 7, 15, 16, 8)


def seed_cells(losses: Sequence[float], levels: Sequence[LevelCounts]) -> list[str]:
    """Write one seed-table row's cells.

    Args:
        losses: The loss at the first and the last step.
        levels: The held-out test's counts per level.

    Returns:
        One cell per column.
    """
    middle = next(level for level in levels if level.level == MEDIUM)
    sampled = sum(level.sampled for level in levels)
    return [
        f"{losses[0]:.{LOSS_DECIMALS}f}{ARROW}{losses[-1]:.{LOSS_DECIMALS}f}",
        *(share(level.sampled_right, level.sampled) for level in levels),
        f"{middle.greedy_right} of {middle.questions}",
        share(middle.unsaved_fourth, middle.sampled),
        share(sum(level.checked for level in levels), sampled),
    ]


def seed_line(label: str, cells: Sequence[str]) -> str:
    """Lay out one row of a seed table.

    Args:
        label: The seed's number as text, or "all".
        cells: One cell per column.

    Returns:
        The row, without trailing spaces.
    """
    line = f"{label:<{SEED_WIDTH}}" + "".join(
        f"{cell:<{width}}" for cell, width in zip(cells, SEED_CELLS, strict=True)
    )
    return line.rstrip()


def pooled_levels(runs: Sequence[SeedRun]) -> list[LevelCounts]:
    """Add every seed's test together, level by level.

    Args:
        runs: The runs.

    Returns:
        One summed count per level.
    """
    pooled = []
    for index, level in enumerate(LEVELS):
        mine = [run.levels[index] for run in runs]
        counts = {
            field.name: sum(getattr(m, field.name) for m in mine)
            for field in fields(LevelCounts)
            if field.name != "level"
        }
        pooled.append(LevelCounts(level=level, **counts))
    return pooled


def print_seed_table(name: str, runs: Sequence[SeedRun]) -> None:
    """Print one way's table: per seed, the loss and the held-out test.

    Args:
        name: The way of running.
        runs: Its runs, one per seed (at least one).
    """
    print(name)
    print(seed_line("seed", SEED_HEADS))
    for run in runs:
        print(seed_line(f"{run.seed:>2}", seed_cells(run.losses, run.levels)))
    first = sum(run.losses[0] for run in runs) / len(runs)
    last = sum(run.losses[-1] for run in runs) / len(runs)
    print(seed_line("all", seed_cells((first, last), pooled_levels(runs))))


RL_HEADS = ("scratch right", "scratch signal", "imitator right", "imitator signal")
RL_CELL = 17


def print_rl_seed_table(name: str, runs: Sequence[SeedRun]) -> None:
    """Print the RL table: per seed, each start's first quarter -> last quarter.

    Args:
        name: The way of running (the one with RL).
        runs: Its runs, one per seed.
    """
    print(f"{name}: RL on the hard questions, the first quarter -> the last")
    heads = [*RL_HEADS, "imitator step 1"]
    print(row("seed", heads, SEED_WIDTH, RL_CELL))
    for run in runs:
        if run.rl is None:
            continue
        cells = []
        for quarters in (run.rl.scratch, run.rl.imitator):
            start, end = quarters[0], quarters[-1]
            cells.append(
                f"{share(start.right, start.rollouts)}{ARROW}"
                f"{share(end.right, end.rollouts)}"
            )
            cells.append(
                f"{start.signal}/{start.groups}{ARROW}{end.signal}/{end.groups}"
            )
        cells.append(f"{run.rl.imitator_first_signal}/{QUESTIONS_PER_STEP}")
        print(row(f"{run.seed:>2}", cells, SEED_WIDTH, RL_CELL))


def print_verdict(claim: str, test: str, held: Sequence[bool]) -> None:
    """Print one claim, the test it was checked with, and how many seeds pass.

    Args:
        claim: The claim, as the plan words it.
        test: What was measured, in plain words.
        held: For each seed, whether the test passed.
    """
    print(f"claim: {claim}")
    print(f"{INDENT}test: {test}")
    print(f"{INDENT}holds in {sum(held)} of {len(held)} seeds")
