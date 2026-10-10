# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Baselines: no baseline, a running average and a per-question average, side by side.

Each variant trains several runs (5 by default, seeds 0 to 4) on the same
questions with the same seeds, twice: with chapter 2's rewards (1 for a right
answer, 0 for a wrong one) and with shifted rewards (1.05 and 1.00), where
every answer earns about 1. For each try it records the push: how far the
step moved the right query's score. The average push is what learning needs;
the spread (standard deviation) is how much the push swings from try to try.
With shifted rewards and no baseline, every try pushes up the query it tried,
right or wrong.

Run it with ``uv run baseline.py``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from questions import TEMPLATES, QuestionSet, make_questions
from reinforce import (
    BASELINE_NAMES,
    BASELINES,
    LOCKED_AT,
    NO_BASELINE,
    SHIFTED_RIGHT,
    SHIFTED_WRONG,
    Outcome,
    Settings,
    non_negative,
    rewards_name,
    rounded,
    train_quietly,
    write_jsonl,
)
from tasks import CHAPTER, course_home, short_path

DEFAULT_LR = 1.0  # big enough for shifted rewards to show both failure and learning
DEFAULT_SEEDS = 5
DEFAULT_EPISODES = 500
PERCENT = 100
NAME_WIDTH = 22  # column widths of the printed table
PUSH_WIDTH = 17
WRONG_WIDTH = 16
RIGHT_WIDTH = 14
RESULTS_DIR = "results"


def summary(outcomes: list[Outcome]) -> dict[str, Any]:
    """Pool one variant's runs (one per seed) into the numbers printed.

    Args:
        outcomes: One run per seed.

    Returns:
        The average and spread of the push over every try of every run, how
        many tries pushed a query up (a wrong one, and any), the right
        query's chance at the end (average and lowest of all rows), the
        locked rows and the tries in all.
    """
    pushes = np.array([push for outcome in outcomes for push in outcome.pushes])
    right = [chance for outcome in outcomes for chance in outcome.right_chances]
    return {
        "push_average": float(pushes.mean()),
        "push_spread": float(pushes.std()),
        "pushed_up": sum(outcome.pushed_up for outcome in outcomes),
        "wrong_pushed_up": sum(outcome.wrong_pushed_up for outcome in outcomes),
        "tries": len(pushes),
        "right_average": float(np.mean(right)),
        "right_lowest": float(np.min(right)),
        "locked": sum(outcome.locked for outcome in outcomes),
        "rows": len(right),
    }


def print_heading(lr: float, episodes: int, seeds: int) -> None:
    """Print what the runs are and what the columns mean.

    Args:
        lr: The learning rate of every run.
        episodes: Tries per run.
        seeds: Runs per line.
    """
    print(
        f"baselines · lr {lr:g} · {episodes} tries per run · seeds 0 to {seeds - 1}"
        f" ({seeds} runs per line)"
    )
    print("push: how far one try's step moved the right query's score")
    print("spread: how much the push swings from try to try (standard deviation)")
    print(
        f"right query: its chance at the end, average of {seeds * len(TEMPLATES)} rows"
        " (the lowest row)"
    )
    print(
        f"locked: rows where a wrong query holds {round(PERCENT * LOCKED_AT)}% or more"
    )


def print_block(title: str, lines: dict[str, dict[str, Any]]) -> None:
    """Print one reward scheme's three variants side by side.

    Args:
        title: The reward scheme, in words.
        lines: The pooled numbers per variant.
    """
    print(f"\n{title}")
    print(
        f"{'':<{NAME_WIDTH}}{'push per try':<{PUSH_WIDTH}}"
        f"{'wrong queries':<{WRONG_WIDTH}}{'right query':<{RIGHT_WIDTH}}locked"
    )
    print(
        f"{'':<{NAME_WIDTH}}{'average  spread':<{PUSH_WIDTH}}"
        f"{'pushed up':<{WRONG_WIDTH}}{'(lowest row)':<{RIGHT_WIDTH}}rows"
    )
    for kind, numbers in lines.items():
        push = f"{numbers['push_average'] + 0.0:+.3f}   {numbers['push_spread']:.3f}"
        wrong = f"{numbers['wrong_pushed_up']:,} of {numbers['tries']:,}"
        right = (
            f"{round(PERCENT * numbers['right_average'])}%"
            f" ({round(PERCENT * numbers['right_lowest'])}%)"
        )
        locked = f"{numbers['locked']} of {numbers['rows']}"
        print(
            f"{BASELINE_NAMES[kind]:<{NAME_WIDTH}}{push:<{PUSH_WIDTH}}"
            f"{wrong:<{WRONG_WIDTH}}{right:<{RIGHT_WIDTH}}{locked}"
        )


def print_no_baseline_note(plain: dict[str, Any]) -> None:
    """Say how often the no-baseline run pushed up the query it tried.

    Args:
        plain: The pooled numbers of the no-baseline runs on shifted rewards.
    """
    print(
        f"with no baseline every advantage is {SHIFTED_WRONG:.2f} or"
        f" {SHIFTED_RIGHT:.2f}, so every step pushes up the query it"
    )
    print(
        f"tried: {plain['pushed_up']:,} of {plain['tries']:,} tries,"
        f" {plain['wrong_pushed_up']:,} of them on a wrong query"
    )


def run_variant(
    kind: str, shifted: bool, args: argparse.Namespace, questions: QuestionSet
) -> tuple[list[Outcome], list[dict[str, Any]]]:
    """Train one baseline on one reward scheme, once per seed.

    Args:
        kind: ``"none"``, ``"average"`` or ``"question"``.
        shifted: Whether the rewards are 1.00 and 1.05.
        args: The flags: learning rate, seeds and tries per run.
        questions: The fair questions.

    Returns:
        One outcome per seed, and one record per seed for the results file.

    Raises:
        OSError: If a work folder cannot be made.
        sqlite3.Error: If a query fails.
    """
    outcomes: list[Outcome] = []
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        settings = Settings(
            seed=seed,
            lr=args.lr,
            episodes=args.episodes,
            baseline=kind,
            shifted=shifted,
        )
        outcome = train_quietly(settings, questions)
        outcomes.append(outcome)
        records.append(
            {
                "rewards": rewards_name(shifted),
                "baseline": kind,
                "seed": seed,
                "right_chances": rounded(outcome.right_chances),
                "pushes": rounded(outcome.pushes),
            }
        )
    return outcomes, records


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The learning rate, seeds and tries per run.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="runs per variant"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="tries per run"
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.episodes < 1:
        parser.error("seeds and episodes must be 1 or more")
    return args


def main() -> None:
    """Train every variant on both reward schemes, print and save the comparison."""
    args = parse_args()
    questions = make_questions()
    print_heading(args.lr, args.episodes, args.seeds)
    results: list[dict[str, Any]] = []
    titles = {
        False: "rewards 1 for a right answer, 0 for a wrong one (chapter 2's)",
        True: (
            f"shifted rewards: {SHIFTED_RIGHT:.2f} for a right answer,"
            f" {SHIFTED_WRONG:.2f} for a wrong one"
        ),
    }
    for shifted, title in titles.items():
        lines: dict[str, dict[str, Any]] = {}
        for kind in BASELINES:
            outcomes, records = run_variant(kind, shifted, args, questions)
            lines[kind] = summary(outcomes)
            results.extend(records)
        print_block(title, lines)
        if shifted:
            print_no_baseline_note(lines[NO_BASELINE])
    name = f"baselines-lr{args.lr:g}-seeds{args.seeds}-tries{args.episodes}.jsonl"
    path = course_home() / CHAPTER / RESULTS_DIR / name
    digest = write_jsonl(path, results)
    print(f"\nwrote {short_path(path)}")
    print(f"md5   {digest}")


if __name__ == "__main__":
    main()
