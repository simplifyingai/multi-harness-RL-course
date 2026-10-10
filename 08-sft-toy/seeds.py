# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Check every claim of this chapter over several seeds, not one lucky run.

One seed is an anecdote. This runs sft_toy.py six ways for seeds 0 to N-1
(10 by default): the varied teacher with RL after it, the fixed-length
teacher, the teacher that never checks, the high learning rate with the
varied and the fixed-length teacher, and the fixed-length teacher with a
student that sees chapters 4 to 7's "calls used" share. It prints one table
per way, one row per seed, and under each table every claim the course plan
makes about it, the test it was checked with, and in how many seeds it held.

Every run writes its own logs under COURSE_HOME, as a single run does. The
runs are independent, so they run side by side on the Mac's cores.

Run it with ``uv run seeds.py`` (about half a minute; ``--seeds 3`` is quicker).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import report
from knobs import DEFAULT_LR, DEFAULT_RL_STEPS, HIGH_LR
from questions import MEDIUM
from report import LevelCounts, RlCounts, SeedRun
from sft_toy import Settings
from sft_toy import run as run_once
from teacher import FIXED_LENGTH, NEVER_CHECKS, VARIED

DEFAULT_SEEDS = 10
NEAR_ZERO = 0.1  # "loss near zero" (the project's dry run: 0.10 by step 4 of 6)
FEW_STEPS = 5  # "within a few steps"
RARELY = 0.05  # "rarely finishes correctly": under 1 rollout in 20
MOSTLY = 0.9  # "most of its groups have no contrast": over 9 in 10
FOURTH_CALL = report.FOURTH_CALL

# The six ways each seed is run: a name for the tables, and what changes.
VARIED_RUN = "varied, then RL"
FIXED_RUN = "fixed-length"
NEVER_RUN = "never-checks"
HIGH_RUN = f"varied, lr {HIGH_LR:g}"
FIXED_HIGH_RUN = f"fixed-length, lr {HIGH_LR:g}"
SHARE_RUN = "fixed-length, calls as a share"
RUNS = {
    VARIED_RUN: {"teacher": VARIED, "then_rl": True},
    FIXED_RUN: {"teacher": FIXED_LENGTH},
    NEVER_RUN: {"teacher": NEVER_CHECKS},
    HIGH_RUN: {"teacher": VARIED, "lr": HIGH_LR},
    FIXED_HIGH_RUN: {"teacher": FIXED_LENGTH, "lr": HIGH_LR},
    SHARE_RUN: {"teacher": FIXED_LENGTH, "calls_as_share": True},
}
# Which run a high-learning-rate run is compared with: same teacher, default lr.
BASELINE_OF = {HIGH_RUN: VARIED_RUN, FIXED_HIGH_RUN: FIXED_RUN}


@dataclass(frozen=True)
class Claim:
    """One claim, and the test it is checked with.

    Attributes:
        text: The claim, as the course plan words it.
        test: What is measured, in plain words.
        holds: Given a run and the same seed's run it is compared with (or
            None), whether the test passes.
    """

    text: str
    test: str
    holds: Callable[[SeedRun, SeedRun | None], bool]


def summarise(name: str, settings: Settings) -> SeedRun:
    """Run once, quietly, and keep only what the tables need.

    Args:
        name: Which of the six ways this is.
        settings: The run's knobs.

    Returns:
        The run's summary.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
        CourseError: If a rollout stopped without a verdict (never expected).
    """
    result = run_once(settings, printing=False)
    rl = None
    if result.rl is not None:
        rl = RlCounts.of(result.rl.scratch, result.rl.imitator)
    return SeedRun(
        name=name,
        seed=settings.seed,
        losses=tuple(step.loss_per_call for step in result.sft_steps),
        levels=tuple(LevelCounts.of(level) for level in result.after),
        rl=rl,
    )


def run_all(seeds: int, rl_steps: int) -> dict[str, list[SeedRun]]:
    """Run every seed every way, side by side on the Mac's cores.

    Args:
        seeds: How many seeds (0 to seeds - 1).
        rl_steps: GRPO steps for each start, in the runs with RL.

    Returns:
        For each way of running, one summary per seed, in seed order.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    jobs = [
        (name, replace(Settings(seed=seed, rl_steps=rl_steps), **changes))
        for name, changes in RUNS.items()
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor() as pool:
        done = list(pool.map(summarise, *zip(*jobs, strict=True)))
    return {name: [s for s in done if s.name == name] for name in RUNS}


def medium(run: SeedRun) -> LevelCounts:
    """Return a run's held-out medium questions.

    Args:
        run: The run.

    Returns:
        Its medium level's counts.
    """
    return next(level for level in run.levels if level.level == MEDIUM)


def loss_near_zero(run: SeedRun, _: SeedRun | None) -> bool:
    """Whether the loss reached NEAR_ZERO within the first FEW_STEPS steps.

    Args:
        run: The run.

    Returns:
        True when one of its first steps' losses is that low.
    """
    return min(run.losses[:FEW_STEPS]) <= NEAR_ZERO


def held_out_drops(run: SeedRun, base: SeedRun | None) -> bool:
    """Whether fewer sampled held-out rollouts were right than at the default lr.

    Args:
        run: The high-learning-rate run.
        base: The same seed and teacher at the default learning rate.

    Returns:
        True when both exist and this one got fewer right.
    """
    return base is not None and run.sampled_right < base.sampled_right


def medium_drops(run: SeedRun, base: SeedRun | None) -> bool:
    """Whether fewer sampled medium rollouts were right than at the default lr.

    Args:
        run: The high-learning-rate run.
        base: The same seed and teacher at the default learning rate.

    Returns:
        True when both exist and this one got fewer medium rollouts right.
    """
    return base is not None and medium(run).sampled_right < medium(base).sampled_right


def every_medium_finishes_on_call_four(run: SeedRun, _: SeedRun | None) -> bool:
    """Whether every greedy medium rollout said FINISH on call 4, unsaved.

    Args:
        run: The run.

    Returns:
        True when all of them did.
    """
    level = medium(run)
    return level.greedy_unsaved_fourth == level.questions


def scratch_rarely_right(run: SeedRun, _: SeedRun | None) -> bool:
    """Whether RL from scratch was rarely right and mostly without contrast.

    Args:
        run: The run with RL.

    Returns:
        True when under RARELY of its rollouts were right and over MOSTLY of
        its groups had no contrast, over the whole RL run.
    """
    if run.rl is None:
        return False
    right, rollouts, signal, groups = run.rl.totals(run.rl.scratch)
    return right < RARELY * rollouts and groups - signal > MOSTLY * groups


CLAIMS: dict[str, tuple[Claim, ...]] = {
    VARIED_RUN: (
        Claim(
            "loss per step falls",
            "the loss at the last step below the loss at step 1",
            lambda run, _: run.losses[-1] < run.losses[0],
        ),
        Claim(
            "--varied learns 'finish after saving' and survives a question that"
            " needs one extra look",
            "greedy: every held-out medium question right",
            lambda run, _: medium(run).greedy_right == medium(run).questions,
        ),
    ),
    FIXED_RUN: (
        Claim(
            "on a question that needs one extra look, the student finishes on"
            f" call {FOURTH_CALL} without saving",
            "greedy: every held-out medium question ends with FINISH on call"
            f" {FOURTH_CALL} and no answer.txt",
            every_medium_finishes_on_call_four,
        ),
    ),
    NEVER_RUN: (
        Claim(
            "the student never checks either",
            "greedy: no held-out rollout reads answer.txt back",
            lambda run, _: run.greedy_checked == 0,
        ),
        Claim(
            "the student never checks either",
            "sampled: no held-out rollout reads answer.txt back",
            lambda run, _: run.sampled_checked == 0,
        ),
    ),
    HIGH_RUN: (
        Claim(
            "training loss near zero within a few steps",
            f"loss {NEAR_ZERO:g} or less at one of steps 1-{FEW_STEPS}",
            loss_near_zero,
        ),
        Claim(
            "while held-out success drops",
            "sampled: fewer held-out rollouts right than the lr"
            f" {DEFAULT_LR:g} run of the"
            " same seed and teacher",
            held_out_drops,
        ),
        Claim(
            "while held-out success drops",
            "the same, on the medium questions only",
            medium_drops,
        ),
    ),
    SHARE_RUN: (
        Claim(
            "on a question that needs one extra look, the student finishes on"
            f" call {FOURTH_CALL} without saving",
            "greedy: every held-out medium question ends with FINISH on call"
            f" {FOURTH_CALL} and no answer.txt",
            every_medium_finishes_on_call_four,
        ),
    ),
}
CLAIMS[FIXED_HIGH_RUN] = CLAIMS[HIGH_RUN]
RL_CLAIMS = (
    Claim(
        "RL from scratch rarely finishes correctly, so most of its groups have"
        " no contrast",
        f"from scratch, under {report.share(RARELY, 1)} of rollouts right and"
        f" over {report.share(MOSTLY, 1)} of groups without contrast",
        scratch_rarely_right,
    ),
    Claim(
        "RL from the imitator has signal from step 1",
        "from the imitator, a group with contrast at step 1",
        lambda run, _: run.rl is not None and run.rl.imitator_first_signal > 0,
    ),
    Claim(
        "RL started from the imitator learns faster on the hard questions",
        "from the imitator, more rollouts right than from scratch in every quarter",
        lambda run, _: run.rl is not None and run.rl.imitator_ahead(),
    ),
)


def print_claims(
    claims: Sequence[Claim],
    runs: Sequence[SeedRun],
    bases: Sequence[SeedRun | None],
) -> None:
    """Print some claims, each with its test and how many seeds pass.

    Args:
        claims: The claims.
        runs: The runs they are about, one per seed.
        bases: The runs each is compared with, one per seed (or None).
    """
    for claim in claims:
        held = [claim.holds(r, b) for r, b in zip(runs, bases, strict=True)]
        report.print_verdict(claim.text, claim.test, held)


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
        "--rl-steps",
        type=int,
        default=DEFAULT_RL_STEPS,
        metavar="N",
        help="GRPO steps for each start",
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.rl_steps < 1:
        parser.error("seeds and rl-steps must be 1 or more")
    return args


def main() -> None:
    """Parse the flags, run every seed every way and print the tables."""
    args = parse_args()
    report.print_seeds_header(args.seeds, args.rl_steps)
    runs = run_all(args.seeds, args.rl_steps)
    nothing: list[SeedRun | None] = [None] * args.seeds
    for name in RUNS:
        report.print_seed_table(name, runs[name])
        base = BASELINE_OF.get(name)
        print_claims(CLAIMS[name], runs[name], runs[base] if base else nothing)
        print()
    report.print_rl_seed_table(VARIED_RUN, runs[VARIED_RUN])
    print_claims(RL_CLAIMS, runs[VARIED_RUN], nothing)


if __name__ == "__main__":
    main()
