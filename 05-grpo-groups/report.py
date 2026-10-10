"""How this chapter prints: steps, groups in full, the tables and the greedy test.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``grpo_groups.py`` can stay about the groups and the loop.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, Any

from environment import tables_needed
from grpo import (
    ALL_RIGHT,
    ALL_WRONG,
    COUNTS,
    GROUP_SIZE,
    QUESTIONS_PER_STEP,
    RIGHT,
    ROLLOUTS,
    SIGNAL,
    Defaults,
)
from questions import LEVEL_NOTES, LEVELS, TEMPLATES, QuestionSet, count_by_level
from tasks import ANSWER_FILE

if TYPE_CHECKING:  # only for type hints: grpo_groups.py imports this module
    from grpo_groups import Group, LevelScore, Settings, Step

# Per level, per stretch of steps: how many groups were all wrong, all right
# or had signal, how many rollouts there were and how many were right.
Table = Mapping[str, Sequence[Mapping[str, int]]]

PERCENT = 100
SHOWN_DECIMALS = 2  # rewards, means, spreads and advantages
LEVEL_WIDTH = 8  # "medium  "
NOTE_WIDTH = 36  # "three or more tables (here four)" + room
COUNT_WIDTH = 2  # question counts and seed numbers
TOTAL_WIDTH = 4  # "15 of 15  ": the total and the gap after it
CELL_WIDTH = 10  # one column of the tables
FIRST_CELL_WIDTH = 12  # the first column, whose heading says "steps"
ANSWER_WIDTH = 12  # "2025-06-08" and "(none)" fit, with room to spare
SIGNAL_WIDTH = 3  # one stretch's count in the seed table
SEED_LEVEL_WIDTH = 15  # one count per quarter (width 3 each), then a gap
INDENT = "  "
SEPARATOR = " · "
NO_FILE = "(none)"
LOOK_TIMES = {1: "once", 2: "twice", 3: "three times", 4: "four times"}
SHARED_TAG = "one set of weights for every level"
PER_LEVEL_TAG = "one set of weights per level"
NEVER_TRAINED = "never trained: every weight still 0, so greedy takes the first"
CRITIC_NOTE = (
    "what the group mean saves",
    "  PPO, the classic algorithm, trains a critic beside the policy: a second",
    "  network, usually as large as the policy, that learns to predict the reward.",
    "  GRPO's prediction is the group's mean reward, which costs nothing to train,",
    "  only the extra rollouts. For LFM2.5-2.6B a critic would be another model of",
    "  about 2.6 billion weights.",
)


def print_header(settings: Settings, questions: QuestionSet) -> None:
    """Print the run's settings, the questions per level and the advantage rule.

    Args:
        settings: The run's knobs.
        questions: The fair questions, split by city.
    """
    weights = SHARED_TAG if settings.shared else PER_LEVEL_TAG
    print(
        f"grpo_groups · seed {settings.seed} · lr {settings.lr:g}"
        f" · {settings.steps} steps · {weights}"
    )
    print(
        f"each step: {QUESTIONS_PER_STEP} questions x {GROUP_SIZE} rollouts at"
        " temperature 1, then one update"
    )
    print(
        f"questions: {len(questions.training)} to train on"
        f" ({len(questions.train_cities)} cities) + {len(questions.held_out)} held"
        f" out ({len(questions.test_cities)} other cities)"
    )
    training = count_by_level(questions.training)
    held_out = count_by_level(questions.held_out)
    for level in LEVELS:
        looks = max(
            tables_needed(t.candidates[0].sql) for t in TEMPLATES if t.level == level
        )
        counts = f"{training[level]:>{COUNT_WIDTH}} + {held_out[level]}"
        print(
            f"{INDENT}{level:<{LEVEL_WIDTH}}{counts:<{LEVEL_WIDTH}}"
            f"{LEVEL_NOTES[level]:<{NOTE_WIDTH}}LOOK {LOOK_TIMES[looks]} first"
        )
    print(
        "advantage = (reward - group mean) / (group spread +"
        f" {Defaults.ADVANTAGE_EPS:g}); a group whose rewards all tie is skipped"
    )
    print()


def tidy(value: float, decimals: int = SHOWN_DECIMALS) -> float:
    """Round a number for printing, so a tiny negative never shows as -0.00.

    Args:
        value: The number.
        decimals: How many decimals will be shown.

    Returns:
        The rounded number; adding 0.0 turns -0.0 into 0.0.
    """
    return round(value, decimals) + 0.0


def signed(values: Sequence[float]) -> str:
    """Write numbers as a list with signs, e.g. ``[+0.58, -1.73]``.

    Args:
        values: The numbers.

    Returns:
        The list as text.
    """
    return "[" + ", ".join(f"{tidy(v):+.{SHOWN_DECIMALS}f}" for v in values) + "]"


def plain(values: Sequence[float]) -> str:
    """Write rewards as a list, without decimals when they are whole.

    Args:
        values: The rewards.

    Returns:
        For example ``"[1, 0, 1, 1]"``.
    """
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def spread(rewards: Sequence[float]) -> tuple[float, float]:
    """Return a group's mean and spread (its standard deviation).

    Args:
        rewards: The group's rewards.

    Returns:
        The mean, and the square root of the average squared distance from it.
    """
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return mean, variance**0.5


def print_group_arithmetic(
    rewards: Sequence[float], advantages: Sequence[float] | None
) -> None:
    """Print how one group's rewards become advantages, step by step.

    Args:
        rewards: The group's rewards.
        advantages: What group_advantages returned for them.
    """
    print(f"rewards     {plain(rewards)}")
    if advantages is None:
        print("no contrast, skipped: every rollout scored the same, so none did")
        print("better or worse than the others, and the group teaches nothing")
        return
    mean, deviation = spread(rewards)
    print(f"mean        {mean:.{SHOWN_DECIMALS}f}")
    print(f"spread      {deviation:.{SHOWN_DECIMALS}f}  (standard deviation)")
    print(f"advantages  (reward - mean) / (spread + {Defaults.ADVANTAGE_EPS:g})")
    print(f"            {signed(advantages)}")


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


def print_group(group: Group, pause_s: float) -> None:
    """Print one group in full: each rollout, the rewards and the advantages.

    Args:
        group: The group.
        pause_s: Seconds to wait after each rollout, so a viewer can follow.
    """
    print(
        f"step {group.step} · group {group.number} of {QUESTIONS_PER_STEP}"
        f" · {group.question.level} · {group.question.task.question}"
    )
    for index, rollout in enumerate(group.rollouts, start=1):
        actions = [call.action for call in rollout.calls]
        written = rollout.verdict.submitted
        answer = NO_FILE if written is None else written
        print(
            f"{INDENT}rollout {index}  reward {rollout.reward}"
            f"  {ANSWER_FILE} {answer:<{ANSWER_WIDTH}}{calls_text(actions)}"
        )
        time.sleep(pause_s)
    rewards = group.rewards
    if group.advantages is None:
        print(f"{INDENT}rewards {plain(rewards)} · no contrast, skipped")
    else:
        mean, deviation = spread(rewards)
        print(
            f"{INDENT}rewards {plain(rewards)} · mean {mean:.{SHOWN_DECIMALS}f}"
            f" · spread {deviation:.{SHOWN_DECIMALS}f}"
        )
        print(
            f"{INDENT}advantages = (reward - mean) / (spread +"
            f" {Defaults.ADVANTAGE_EPS:g}) = {signed(group.advantages)}"
        )
    print()


def print_step(step: Step, metrics: Mapping[str, Any]) -> None:
    """Print one step on one line, as the project's trainer does.

    Args:
        step: The step.
        metrics: Its metrics.jsonl record.
    """
    rights = ", ".join(
        f"{group.question.level} {sum(group.rewards)}/{len(group.rollouts)}"
        for group in step.groups
    )
    print(
        f"step {step.number:>3}{SEPARATOR}reward {metrics['mean_reward']:.2f}"
        f"{SEPARATOR}correct {round(PERCENT * metrics['accuracy'])}%"
        f"{SEPARATOR}calls {metrics['mean_calls']:.1f}"
        f"{SEPARATOR}no-signal groups {step.no_signal}/{len(step.groups)}"
        f"{SEPARATOR}{rights}"
    )


def table_row(name: str, cells: Sequence[str]) -> str:
    """Lay out one row of a table: a name, then one cell per stretch.

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


def percent_right(counts: Mapping[str, int]) -> str:
    """Write the share of a stretch's rollouts that were right, e.g. ``"45%"``.

    Args:
        counts: One level's counts for one stretch.

    Returns:
        The share as a whole percentage. A very short run can leave a level
        with no group in a stretch; that shows as 0%.
    """
    if not counts[ROLLOUTS]:
        return "0%"
    return f"{round(PERCENT * counts[RIGHT] / counts[ROLLOUTS])}%"


def print_table(table: Table, edges: Sequence[int]) -> None:
    """Print, per level and stretch, the share right and the no-signal groups.

    Args:
        table: Per level, per stretch, the groups all wrong, all right or with
            signal, and the rollouts right.
        edges: The stretch boundaries.
    """
    print()
    # grpo_groups.py splits every run into four stretches: its quarters.
    print("by difficulty, in each quarter of the run")
    print(table_row("", stretch_labels(edges)))
    print("rollouts right")
    for level in LEVELS:
        print(table_row(level, [percent_right(c) for c in table[level]]))
    print(f"no-signal groups (all {GROUP_SIZE} rewards tied, so skipped)")
    for level in LEVELS:
        cells = []
        for counts in table[level]:
            silent = counts[ALL_WRONG] + counts[ALL_RIGHT]
            cells.append(f"{silent}/{silent + counts[SIGNAL]}")
        print(table_row(level, cells))
    print(f"of those, {ALL_WRONG} + {ALL_RIGHT}")
    for level in LEVELS:
        cells = [f"{c[ALL_WRONG]} + {c[ALL_RIGHT]}" for c in table[level]]
        print(table_row(level, cells))


def print_critic_note() -> None:
    """Print what GRPO saves compared with PPO."""
    print()
    for line in CRITIC_NOTE:
        print(line)


def print_greedy(scores: Sequence[LevelScore]) -> None:
    """Print the greedy test at the end, one line per level.

    Args:
        scores: One score per level.
    """
    print()
    print("greedy test at the end (always the most likely action)")
    for score in scores:
        print(
            f"{INDENT}{score.level:<{LEVEL_WIDTH}}"
            f"training {score.training:>{COUNT_WIDTH}} of"
            f" {score.training_total:<{TOTAL_WIDTH}}"
            f"held out {score.held_out} of {score.held_out_total}"
            f"{SEPARATOR}{calls_text(score.actions)}"
        )
        if score.untrained:
            print(f"{INDENT}{'':<{LEVEL_WIDTH}}{NEVER_TRAINED}")


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


def print_seeds_header(seeds: int, lr: float, steps: int, shared: bool) -> None:
    """Print what ``seeds.py`` runs and the heading of its table.

    Args:
        seeds: How many seeds.
        lr: The learning rate.
        steps: Steps per run.
        shared: Whether every level shares one set of weights.
    """
    weights = SHARED_TAG if shared else PER_LEVEL_TAG
    print(f"seeds 0-{seeds - 1} · lr {lr:g} · {steps} steps · {weights}")
    print("groups with signal, in each quarter of the run")
    heading = "".join(f"{level:<{SEED_LEVEL_WIDTH}}" for level in LEVELS)
    print(f"{'seed':<{LEVEL_WIDTH}}{heading}".rstrip())


def print_seed(seed: int, table: Table) -> None:
    """Print one seed's row: its groups with signal per level and stretch.

    Args:
        seed: The seed.
        table: That run's table.
    """
    parts = []
    for level in LEVELS:
        counts = "".join(f"{c[SIGNAL]:>{SIGNAL_WIDTH}}" for c in table[level])
        parts.append(f"{counts:<{SEED_LEVEL_WIDTH}}")
    number = f"{seed:>{COUNT_WIDTH}}"  # right-aligned under "seed"
    print(f"{number:<{LEVEL_WIDTH}}{''.join(parts)}".rstrip())


def add_tables(tables: Sequence[Table]) -> dict[str, list[dict[str, int]]]:
    """Add several runs' tables together, cell by cell.

    Args:
        tables: One table per run, all with the same stretches.

    Returns:
        One table holding the sums.
    """
    first = tables[0]
    total = {
        level: [dict.fromkeys(COUNTS, 0) for _ in first[level]] for level in LEVELS
    }
    for table in tables:
        for level in LEVELS:
            for sums, counts in zip(total[level], table[level], strict=True):
                for count in COUNTS:
                    sums[count] += counts[count]
    return total


def print_seed_totals(tables: Sequence[Table], edges: Sequence[int]) -> None:
    """Add up every seed's table and print it like a single run's.

    Args:
        tables: One table per seed.
        edges: The stretch boundaries.
    """
    print(f"\nall {len(tables)} seeds together")
    print_table(add_tables(tables), edges)
