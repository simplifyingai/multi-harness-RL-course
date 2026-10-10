# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Three learning rates on shifted rewards, with and without a baseline.

With shifted rewards (1.00 for a wrong answer, 1.05 for a right one) every
try earns about 1, so with no baseline every step pushes up whatever query
was tried. A big learning rate then locks a question onto the first query it
tries, right or wrong. Each setting trains several runs (5 by default, seeds
0 to 4), and the table counts the questions locked onto a wrong query, seed
by seed.

Run it with ``uv run learning_rate.py``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from questions import TEMPLATES, QuestionSet, make_questions
from reinforce import (
    AVERAGE,
    BASELINE_NAMES,
    LOCKED_AT,
    NO_BASELINE,
    SHIFTED_RIGHT,
    SHIFTED_WRONG,
    Outcome,
    Settings,
    rounded,
    train_quietly,
    write_jsonl,
)
from tasks import CHAPTER, course_home, short_path

LEARNING_RATES = (0.1, 1.0, 30.0)
VARIANTS = (NO_BASELINE, AVERAGE)
DEFAULT_SEEDS = 5
DEFAULT_EPISODES = 500
PERCENT = 100
LR_WIDTH = 6
NAME_WIDTH = 17
RIGHT_WIDTH = 13
LOCKED_WIDTH = 10
SEEDS_WIDTH = 11
RESULTS_DIR = "results"


def print_heading(episodes: int, seeds: int, held_out: int) -> None:
    """Print what the runs are and what the columns mean.

    Args:
        episodes: Tries per run.
        seeds: Runs per line.
        held_out: Held-out questions per run.
    """
    print(
        f"learning rates on shifted rewards: {SHIFTED_RIGHT:.2f} for a right answer,"
        f" {SHIFTED_WRONG:.2f} for a wrong one"
    )
    print(f"{episodes} tries per run · seeds 0 to {seeds - 1} ({seeds} runs per line)")
    print(
        f"right query: its chance at the end, average of {seeds * len(TEMPLATES)} rows"
    )
    print(
        f"locked: rows where a wrong query holds {round(PERCENT * LOCKED_AT)}% or more"
    )
    print(f"held-out: the greedy test, {held_out} questions per run")
    print()
    print(
        f"{'lr':<{LR_WIDTH}}{'baseline':<{NAME_WIDTH}}{'right query':<{RIGHT_WIDTH}}"
        f"{'locked':<{LOCKED_WIDTH}}{'per seed':<{SEEDS_WIDTH}}held-out"
    )


def run_setting(
    lr: float, kind: str, args: argparse.Namespace, questions: QuestionSet
) -> tuple[list[Outcome], list[dict[str, Any]]]:
    """Train one learning rate and baseline on shifted rewards, once per seed.

    Args:
        lr: The learning rate.
        kind: ``"none"`` or ``"average"``.
        args: The flags: seeds and tries per run.
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
            seed=seed, lr=lr, episodes=args.episodes, baseline=kind, shifted=True
        )
        outcome = train_quietly(settings, questions)
        outcomes.append(outcome)
        records.append(
            {
                "lr": lr,
                "baseline": kind,
                "seed": seed,
                "right_chances": rounded(outcome.right_chances),
                "locked": outcome.locked,
                "held_out": outcome.test_correct,
            }
        )
    return outcomes, records


def print_row(lr: float, kind: str, outcomes: list[Outcome], held_out: int) -> None:
    """Print one setting's runs on one line.

    Args:
        lr: The learning rate.
        kind: ``"none"`` or ``"average"``.
        outcomes: One run per seed.
        held_out: Held-out questions per run.
    """
    right = (
        f"{round(PERCENT * np.mean([c for o in outcomes for c in o.right_chances]))}%"
    )
    rows = len(outcomes) * len(TEMPLATES)
    locked = f"{sum(o.locked for o in outcomes)} of {rows}"
    per_seed = " ".join(str(o.locked) for o in outcomes)
    tested = f"{sum(o.test_correct for o in outcomes)} of {held_out * len(outcomes)}"
    print(
        f"{lr:<{LR_WIDTH}g}{BASELINE_NAMES[kind]:<{NAME_WIDTH}}{right:<{RIGHT_WIDTH}}"
        f"{locked:<{LOCKED_WIDTH}}{per_seed:<{SEEDS_WIDTH}}{tested}"
    )


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The seeds and tries per run.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS, help="runs per setting"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="tries per run"
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.episodes < 1:
        parser.error("seeds and episodes must be 1 or more")
    return args


def main() -> None:
    """Train every learning rate with and without a baseline; print and save."""
    args = parse_args()
    questions = make_questions()
    held_out = len(questions.held_out)
    print_heading(args.episodes, args.seeds, held_out)
    results: list[dict[str, Any]] = []
    for kind in VARIANTS:
        for lr in LEARNING_RATES:
            outcomes, records = run_setting(lr, kind, args, questions)
            print_row(lr, kind, outcomes, held_out)
            results.extend(records)
    name = f"learning-rates-seeds{args.seeds}-tries{args.episodes}.jsonl"
    path = course_home() / CHAPTER / RESULTS_DIR / name
    digest = write_jsonl(path, results)
    print(f"\nwrote {short_path(path)}")
    print(f"md5   {digest}")


if __name__ == "__main__":
    main()
