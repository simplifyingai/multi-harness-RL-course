# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Check each reward rule's claimed failure over several seeds, not one run.

One seed is an anecdote. This trains reward_lab.py under every rule (or the
ones named with ``--rules``) for seeds 0 to N-1 (5 by default), and prints a
table per rule: for each seed, what the rollouts did at the start of the run
-> at its end (the share right, the reward, calls per rollout, tokens per
reply, replies cut off). Under each table it checks the rule's claim with a
stated test and says in how many seeds it held.

Every run writes its own logs under COURSE_HOME, as a single run does. The
runs are independent, so they run side by side on the Mac's cores.

Run it with ``uv run seeds.py`` (about half a minute).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import report
from report import Tally
from reward_lab import (
    DEFAULT_LENGTH_LR,
    DEFAULT_STEPS,
    Settings,
    non_negative,
    positive,
    run,
)
from rewards import (
    BONUS_WHEN_WRONG,
    CALL_BONUS,
    CORRECT,
    RULES,
    SUBMIT,
    TOKEN_BONUS,
)

DEFAULT_SEEDS = 5
COLLAPSED_CUT_OFF = 0.5  # "replies pass the cap": over half of them cut off
COLLAPSED_RIGHT = 0.1  # "accuracy collapses": under a tenth of rollouts right
SAME_ACCURACY = 0.05  # "as correct": this close to the correct-only run, or better
SAME_POINTS = round(report.PERCENT * SAME_ACCURACY)  # the same, in points
GAVE_UP_CALLS = 1.5  # "gives up early": stops, on average, at the first call
RULE_SEPARATOR = ","


@dataclass(frozen=True)
class Summary:
    """What one run contributes to the tables (small, so it travels cheaply).

    Attributes:
        rule: The reward rule it was trained with.
        seed: Its seed.
        start: What the rollouts did at the start of the run.
        end: What they did at its end.
    """

    rule: str
    seed: int
    start: Tally
    end: Tally


@dataclass(frozen=True)
class Claim:
    """One claim about a rule, and the test it is checked with.

    Attributes:
        rule: The rule the claim is about.
        text: The claim, as the course plan words it.
        test: What is measured, in plain words.
        holds: Given this rule's summary and the correct-only summary of the
            same seed (or None), whether the test passes.
        needs_correct: Whether the test compares with the correct-only run.
    """

    rule: str
    text: str
    test: str
    holds: Callable[[Summary, Summary | None], bool]
    needs_correct: bool = False


def longer_than_correct_only(mine: Summary, base: Summary | None) -> bool:
    """Whether the end's replies are longer than the correct-only run's.

    Args:
        mine: This rule's run.
        base: The correct-only run of the same seed.

    Returns:
        True when both exist and this one's replies are longer.
    """
    return base is not None and mine.end.tokens_per_reply > base.end.tokens_per_reply


def as_correct_as_correct_only(mine: Summary, base: Summary | None) -> bool:
    """Whether the end's share right is within SAME_ACCURACY of correct-only's.

    Args:
        mine: This rule's run.
        base: The correct-only run of the same seed.

    Returns:
        True when both exist and the shares are that close or better.
    """
    return base is not None and mine.end.accuracy >= base.end.accuracy - SAME_ACCURACY


CLAIMS = (
    Claim(
        CORRECT,
        "nothing rewards fewer calls, so they wander up toward the budget",
        "more calls per rollout at the end than at the start",
        lambda m, _: m.end.calls_per_rollout > m.start.calls_per_rollout,
    ),
    Claim(
        SUBMIT,
        "training reward rises while accuracy falls",
        "reward higher, and share right lower, at the end than at the start",
        lambda m, _: (
            m.end.reward > m.start.reward and m.end.accuracy < m.start.accuracy
        ),
    ),
    Claim(
        CALL_BONUS,
        "tokens per reply climb",
        "longer replies at the end than the correct-only run of the same seed",
        longer_than_correct_only,
        needs_correct=True,
    ),
    Claim(
        CALL_BONUS,
        "replies pass the cap and accuracy collapses",
        f"at the end, over {COLLAPSED_CUT_OFF:.0%} of replies cut off and under"
        f" {COLLAPSED_RIGHT:.0%} of rollouts right",
        lambda m, _: (
            m.end.cut_off_share > COLLAPSED_CUT_OFF and m.end.accuracy < COLLAPSED_RIGHT
        ),
    ),
    Claim(
        TOKEN_BONUS,
        "short",
        "shorter replies at the end than at the start, and none cut off",
        lambda m, _: (
            m.end.tokens_per_reply < m.start.tokens_per_reply and m.end.cut_off == 0
        ),
    ),
    Claim(
        TOKEN_BONUS,
        "and correct",
        f"share right at the end within {SAME_POINTS} points of the correct-only"
        " run of the same seed, or better",
        as_correct_as_correct_only,
        needs_correct=True,
    ),
    Claim(
        BONUS_WHEN_WRONG,
        "it gives up early",
        f"at the end, under {GAVE_UP_CALLS:g} calls per rollout and none right",
        lambda m, _: m.end.calls_per_rollout < GAVE_UP_CALLS and m.end.correct == 0,
    ),
)


def summarise(settings: Settings) -> Summary:
    """Run once, quietly, and keep only what the tables need.

    Args:
        settings: The run's knobs.

    Returns:
        The run's summary.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    result = run(settings, printing=False)
    start, end = report.start_and_end(result.steps)
    return Summary(settings.rule, settings.seed, start, end)


def run_all(
    rules: Sequence[str], seeds: int, steps: int, length_lr: float, guessable: bool
) -> dict[str, list[Summary]]:
    """Run every rule for every seed, side by side on the Mac's cores.

    Args:
        rules: The reward rules.
        seeds: How many seeds (0 to seeds - 1).
        steps: Steps per run.
        length_lr: The length weight's learning rate.
        guessable: Whether to add the guessable questions.

    Returns:
        For each rule, one summary per seed, in seed order.

    Raises:
        OSError: If a folder or a log cannot be written.
        sqlite3.Error: If a query fails on the real database.
    """
    jobs = [
        Settings(
            rule=rule,
            seed=seed,
            steps=steps,
            length_lr=length_lr,
            guessable=guessable,
        )
        for rule in rules
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor() as pool:
        done = list(pool.map(summarise, jobs))
    return {rule: [s for s in done if s.rule == rule] for rule in rules}


def print_claims(rule: str, runs: dict[str, list[Summary]]) -> None:
    """Print every claim about one rule, checked seed by seed.

    A claim that compares with the correct-only run says so and is skipped
    when those runs were not made (``--rules`` without ``correct``).

    Args:
        rule: The rule.
        runs: Every rule's summaries.
    """
    base = runs.get(CORRECT)
    mine = runs[rule]
    for claim in CLAIMS:
        if claim.rule != rule:
            continue
        if claim.needs_correct and base is None:
            print(f"claim: {claim.text}")
            print(f"{report.INDENT}not checked: it needs the correct-only runs")
            continue
        others = base or [None] * len(mine)
        held = [claim.holds(m, b) for m, b in zip(mine, others, strict=True)]
        report.print_verdict(claim.text, claim.test, held)


def rule_list(text: str) -> list[str]:
    """Read ``--rules``: reward rules separated by commas.

    Args:
        text: What was typed after the flag.

    Returns:
        The rules, in the order of ``rewards.RULES``.

    Raises:
        argparse.ArgumentTypeError: If a rule is not one of the five, or
            none is named.
    """
    asked = [part.strip() for part in text.split(RULE_SEPARATOR) if part.strip()]
    unknown = [rule for rule in asked if rule not in RULES]
    if unknown or not asked:
        raise argparse.ArgumentTypeError(
            f"{text!r}: name rules from {', '.join(RULES)}, separated by commas"
        )
    return [rule for rule in RULES if rule in asked]


def parse_args() -> argparse.Namespace:
    """Read the flags.

    Returns:
        The knobs, as typed.

    Raises:
        SystemExit: If a flag is not valid (argparse prints why).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", type=positive, default=DEFAULT_SEEDS, help="how many seeds, from 0"
    )
    parser.add_argument(
        "--steps", type=positive, default=DEFAULT_STEPS, help="steps per run"
    )
    parser.add_argument(
        "--length-lr",
        type=non_negative,
        default=DEFAULT_LENGTH_LR,
        help="the length weight's learning rate",
    )
    parser.add_argument(
        "--rules",
        type=rule_list,
        default=list(RULES),
        metavar="R,R",
        help=f"which rules to run (default all: {', '.join(RULES)})",
    )
    parser.add_argument(
        "--guessable", action="store_true", help="add the guessable questions"
    )
    return parser.parse_args()


def main() -> None:
    """Parse the flags, run every rule for every seed and print the tables."""
    args = parse_args()
    report.print_seeds_header(args.seeds, args.steps, args.length_lr, args.guessable)
    runs = run_all(args.rules, args.seeds, args.steps, args.length_lr, args.guessable)
    seeds = list(range(args.seeds))
    for rule in args.rules:
        report.print_rule_table(rule, [(s.start, s.end) for s in runs[rule]], seeds)
        print_claims(rule, runs)
        print()


if __name__ == "__main__":
    main()
