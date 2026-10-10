# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Check this chapter's claims over several seeds, not one lucky run.

One seed is an anecdote. This runs grpo_clip.py seven ways over seeds 0 to
N-1 (10 by default) and prints one row per seed for each claim:

1. fresh batches give ratios of exactly 1; stale ones do not, and
   ``--recompute-old`` hides it;
2. reusing a batch spreads the ratios, use by use;
3. the clip switches off more pushes as a batch is reused;
4. how much the stale runs learned, with and without the correction;
5. how much the reuse runs learned, with and without the clip.

Every run writes its own logs under COURSE_HOME, as a single run does. The
runs are independent, so they run side by side on the Mac's cores.

Run it with ``uv run seeds.py`` (about a minute; ``--seeds 3`` is quicker).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import report
from grpo import outside_clip_range
from grpo_clip import DEFAULT_LR, DEFAULT_STEPS, Settings, non_negative, run
from questions import EASY, MEDIUM
from report import UseCounts

DEFAULT_SEEDS = 10
REUSES = 8  # the batch reuse the claims are checked at
SLOW_LR = 0.5  # a smaller learning rate, where the spreading is gradual
STALE = 2  # the article's usual staleness
ONE = 1.0  # a ratio of exactly 1: the sampler and the learner agree

# The seven ways each seed is run: a name for the tables, and what changes.
FRESH = "fresh"
STALE_RUN = f"stale {STALE}"
HIDDEN = f"stale {STALE}, recompute-old"
STALE_NO_CLIP = f"stale {STALE}, no clip"
REUSE = f"{REUSES} updates"
NO_CLIP = f"{REUSES} updates, no clip"
SLOW = f"{REUSES} updates, lr {SLOW_LR:g}"
RUNS = {
    FRESH: {},
    STALE_RUN: {"stale": STALE},
    HIDDEN: {"stale": STALE, "recompute_old": True},
    STALE_NO_CLIP: {"stale": STALE, "clip": False},
    REUSE: {"updates": REUSES},
    NO_CLIP: {"updates": REUSES, "clip": False},
    SLOW: {"updates": REUSES, "lr": SLOW_LR},
}
# Tables 4 and 5 compare how much the runs learned.
STALE_RUNS = (FRESH, STALE_RUN, HIDDEN, STALE_NO_CLIP)
REUSE_RUNS = (FRESH, REUSE, NO_CLIP)
LEARNED_LEVELS = (EASY, MEDIUM)  # hard questions never get signal here


@dataclass(frozen=True)
class Summary:
    """What one run contributes to the tables (small, so it travels cheaply).

    Attributes:
        name: Which of the seven ways it was run.
        seed: Its seed.
        ratios: How many ratios the loss saw, over every use.
        not_one: How many of them were not exactly 1.
        outside: How many fell outside 0.8 to 1.2.
        clipped: How many the clip switched off.
        hidden_outside: With --recompute-old, how many of the same calls'
            ratios against the recorded p fell outside 0.8 to 1.2.
        hidden_clipped: And how many of those the clip would have switched off.
        by_use: Per use of the batch, the ratios counted.
        right: Per quarter of the run, easy and medium rollouts right and
            how many there were.
    """

    name: str
    seed: int
    ratios: int
    not_one: int
    outside: int
    clipped: int
    hidden_outside: int
    hidden_clipped: int
    by_use: tuple[UseCounts, ...]
    right: tuple[tuple[int, int], ...]


def summarise(name: str, settings: Settings) -> Summary:
    """Run once, quietly, and keep only what the tables need.

    Args:
        name: Which of the seven ways this is.
        settings: The run's knobs.

    Returns:
        The run's summary.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    result = run(settings, printing=False)
    rows = report.use_rows(result.steps)
    ratios = np.concatenate([row.ratios for row in rows])
    clipped = np.concatenate([row.clipped for row in rows])
    hidden = report.hidden_rows(result.steps) if settings.recompute_old else []
    hidden_ratios = np.concatenate([row.ratios for row in hidden] or [np.zeros(0)])
    hidden_clipped = np.concatenate([row.clipped for row in hidden] or [np.zeros(0)])
    table = report.right_by_quarter(result.steps, result.edges)
    right = tuple(
        (
            sum(table[level][stretch][0] for level in LEARNED_LEVELS),
            sum(table[level][stretch][1] for level in LEARNED_LEVELS),
        )
        for stretch in range(len(result.edges) - 1)
    )
    return Summary(
        name=name,
        seed=settings.seed,
        ratios=int(ratios.size),
        not_one=int(np.count_nonzero(ratios != ONE)),
        outside=int(outside_clip_range(ratios).sum()),
        clipped=int(clipped.sum()),
        hidden_outside=int(outside_clip_range(hidden_ratios).sum()),
        hidden_clipped=int(hidden_clipped.sum()),
        by_use=tuple(
            UseCounts(
                int(row.ratios.size),
                int(outside_clip_range(row.ratios).sum()),
                int(row.clipped.sum()),
            )
            for row in rows
        ),
        right=right,
    )


def run_all(seeds: int, steps: int, lr: float) -> dict[str, list[Summary]]:
    """Run every seed every way, side by side on the Mac's cores.

    Args:
        seeds: How many seeds (0 to seeds - 1).
        steps: Steps per run.
        lr: The learning rate (the slow run always uses SLOW_LR).

    Returns:
        For each way of running, one summary per seed, in seed order.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    jobs = [
        (name, replace(Settings(seed=seed, steps=steps, lr=lr), **changes))
        for name, changes in RUNS.items()
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor() as pool:
        done = list(pool.map(summarise, *zip(*jobs, strict=True)))
    return {name: [s for s in done if s.name == name] for name in RUNS}


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
        "--steps", type=int, default=DEFAULT_STEPS, help="steps per run"
    )
    parser.add_argument(
        "--lr",
        type=non_negative,
        default=DEFAULT_LR,
        help=f"learning rate for every run but the slow one (not {SLOW_LR:g})",
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.steps < 1:
        parser.error("seeds and steps must be 1 or more")
    if args.lr == SLOW_LR:
        # The reuse run and the slow run would be the same run, twice.
        parser.error(f"lr {SLOW_LR:g} is the slow run's already; pick another")
    return args


def main() -> None:
    """Parse the flags, run every seed every way and print the tables."""
    args = parse_args()
    report.print_seeds_header(args.seeds, args.steps, args.lr, SLOW_LR, len(RUNS))
    runs = run_all(args.seeds, args.steps, args.lr)
    report.print_ratio_claims(runs[FRESH], runs[STALE_RUN], runs[HIDDEN])
    labels = (f"lr {args.lr:g}", f"lr {SLOW_LR:g}")
    reuse = [s.by_use for s in runs[REUSE]]
    slow = [s.by_use for s in runs[SLOW]]
    report.print_by_use(
        f"2. ratios outside {report.LOW:g}-{report.HIGH:g} at each use of the batch"
        f" ({REUSES} updates"
        " per batch), %",
        "outside",
        (reuse, slow),
        labels,
    )
    report.print_by_use(
        f"3. calls clipped at each use of the batch ({REUSES} updates per batch), %",
        "clipped",
        (reuse, slow),
        labels,
    )
    report.print_learning(
        f"4. easy and medium rollouts right, in each quarter of the run, %:"
        f" rollouts {STALE} updates old",
        {name: [s.right for s in runs[name]] for name in STALE_RUNS},
    )
    report.print_learning(
        f"5. the same: each batch used for {REUSES} updates",
        {name: [s.right for s in runs[name]] for name in REUSE_RUNS},
    )


if __name__ == "__main__":
    main()
