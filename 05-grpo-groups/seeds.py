# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Run grpo_groups.py over several seeds and count the no-signal groups by level.

One seed is an anecdote. This runs seeds 0 to N-1 (10 by default) with the
same settings and prints, for each seed and each stretch of steps, how many
groups of each level had no signal, and whether they were all wrong or all
right. Each run writes its own logs under COURSE_HOME, as a single run does.

Run it with ``uv run seeds.py`` (under a minute for all ten).
"""

from __future__ import annotations

import argparse

import report
from grpo_groups import (
    DEFAULT_LR,
    DEFAULT_STEPS,
    RunResult,
    Settings,
    non_negative,
    run,
)

DEFAULT_SEEDS = 10


def run_seeds(seeds: int, lr: float, steps: int, shared: bool) -> list[RunResult]:
    """Train one run per seed, printing nothing but a line per seed.

    Args:
        seeds: How many seeds (0 to seeds - 1).
        lr: The learning rate.
        steps: Steps per run.
        shared: One set of weights for every level.

    Returns:
        One result per seed.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    results = []
    for seed in range(seeds):
        settings = Settings(seed=seed, lr=lr, steps=steps, shared=shared)
        result = run(settings, printing=False)
        report.print_seed(seed, result.table)
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="how many seeds, from 0"
    )
    parser.add_argument(
        "--lr", type=non_negative, default=DEFAULT_LR, help="how big each step is"
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="steps per run"
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="one set of weights for every level instead of one per level",
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.steps < 1:
        parser.error("seeds and steps must be 1 or more")
    return args


def main() -> None:
    """Parse the flags, run every seed and print the totals."""
    args = parse_args()
    report.print_seeds_header(args.seeds, args.lr, args.steps, args.shared)
    results = run_seeds(args.seeds, args.lr, args.steps, args.shared)
    report.print_seed_totals([r.table for r in results], results[0].edges)


if __name__ == "__main__":
    main()
