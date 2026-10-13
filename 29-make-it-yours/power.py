# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""How many paired tasks a hoped-for gain needs before it can be told from noise.

Chapter 24 read two checkpoints cell by cell: tasks **gained** (wrong before,
right after), tasks **lost** (the reverse), and the exact McNemar p-value,
the chance of a split at least that uneven if the two were equally good.
This script turns that reading around, before anything is run: if the
model really improves by some number of points, how many tasks must both
checkpoints answer before the comparison says p < 0.05, with a chosen
**power** (the chance that a real gain of that size does show up as
p < 0.05; 80% is the usual choice)?

Only the tasks that **flip** carry information: a task both checkpoints
solve, or both miss, says nothing about which is better. So the answer
depends on how often the two checkpoints disagree, not only on the gain,
and three assumptions are compared:

* **only gains**: the new checkpoint keeps every task the old one solved
  and adds the gain (the fewest flips possible, the best case);
* **chapter 24's churn**: as many tasks flip as flipped between the
  project's checkpoints (68 of 216 cells over 9 pairs, 31.5%), with the
  gain as the difference between gained and lost;
* **independent attempts**: each checkpoint solves each task with its own
  rate, independently (the most flips, the worst case).

The power is computed exactly: for n tasks, the number that flip is
binomial, each flip is a gain with a fixed chance, and the test rejects
exactly when ``stats.exact_p`` (chapter 24's) says p < 0.05. The smallest
n that reaches the power is found by trying n = 1, 2, 3, ... up to 2,000
tasks; past that, the normal approximation of the McNemar sample size
(Connor, 1987) is used and labelled. A seeded simulation checks the exact
figure. An exact test's power zigzags a little as n grows (the p-value moves
in steps), so the first n to reach the power is reported as it is.

``uv run power.py`` prints the reading, the plan's example and what 24 tasks
can see; ``--baseline``, ``--gain`` and ``--power`` ask your own question.
Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

from stats import exact_p

HERE = Path(__file__).resolve().parent
PAIRS_FILE = HERE / "data" / "chapter24_pairs.json"
ALPHA = Fraction(1, 20)  # p < 0.05, the course's line between noise and not
DEFAULT_POWER = 0.80
DEFAULT_BASELINE = 0.60  # the project's base model, averaged over harnesses: 59.7%
DEFAULT_GAIN = 0.10
COURSE_TASKS = 24  # tasks per harness in chapters 24 to 28
PLAN_TASKS = 150  # the plan's example: "+10 points needs about 150 paired tasks"
EXACT_MAX_TASKS = 2_000  # past this, the normal approximation
WINDOW_SIGMAS = 10  # flipped counts this far from the mean are skipped (< 1e-20)
SIMULATION_RUNS = 10_000
SIMULATION_SEED = 0
BASELINE_GRID = (0.2, 0.4, 0.6, 0.8)
FLIP_STEP = 0.005  # the plan check tries flip shares in half-point steps
PLAN_GAIN = 0.10  # the plan's example gain, whatever --gain asks
STEADY_WINDOW = 50  # after the first n that reaches the power, n this far are checked
PERCENT = 100
FLOAT_TOLERANCE = 1e-9
AMBIGUOUS_ONE = "1"  # a fraction (100%) or a percentage (1%)?
EXAMPLE_PAIR = ("base -> sft", "claude-code")  # the plan's "62 -> 71" reading
RULE = "━━"


@dataclass(frozen=True)
class Flips:
    """How two checkpoints disagree on one task, as chances.

    Attributes:
        gained: The chance a task is wrong before and right after.
        lost: The chance a task is right before and wrong after.
    """

    gained: float
    lost: float

    @property
    def flipped(self) -> float:
        """The chance a task changes verdict, either way.

        Returns:
            ``gained + lost``.
        """
        return self.gained + self.lost

    @property
    def gain(self) -> float:
        """The score's change, as a share of the tasks.

        Returns:
            ``gained - lost``.
        """
        return self.gained - self.lost


@dataclass(frozen=True)
class Answer:
    """How many tasks one assumption needs.

    Attributes:
        tasks: The smallest number of paired tasks found.
        power: The exact power at that number (None for the approximation).
        method: ``exact`` or ``normal approximation``.
        steady_from: An exact test's power zigzags as n grows: from this n
            on it stayed at or above the target over every n checked
            (``tasks`` itself when it never dipped; None when approximated).
    """

    tasks: int
    power: float | None
    method: str
    steady_from: int | None = None


def check_question(baseline: float, gain: float, power: float) -> None:
    """Refuse a question that has no answer.

    Args:
        baseline: The old checkpoint's rate (0 to 1).
        gain: The hoped-for gain, as a share.
        power: The power wanted.

    Raises:
        ValueError: If the gain is not above 0, the baseline plus the gain is
            above 100%, or the power is not strictly between 0 and 100%.
    """
    if not gain > 0:
        raise ValueError("the gain must be above 0")
    if not 0 <= baseline <= 1 or baseline + gain > 1:
        raise ValueError(
            f"no room for {points(gain)} from {percent(baseline)}: above 100%"
        )
    if not 0 < power < 1:
        raise ValueError("the power must be between 0 and 100%, both excluded")


def only_gains(gain: float) -> Flips:
    """The fewest flips a gain can come with: every flip is a gain.

    Args:
        gain: The hoped-for gain, as a share (0.10 for +10 points).

    Returns:
        The flips.
    """
    return Flips(gained=gain, lost=0.0)


def with_churn(baseline: float, gain: float, flipped: float) -> Flips | None:
    """Flips at a given churn: this share of tasks flips, netting the gain.

    Args:
        baseline: The old checkpoint's rate (0 to 1).
        gain: The hoped-for gain, as a share.
        flipped: The share of tasks that change verdict.

    Returns:
        The flips, or None when that churn cannot hold that gain (fewer flips
        than the gain, or more gains than unsolved tasks).
    """
    gained, lost = (flipped + gain) / 2, (flipped - gain) / 2
    if lost < 0 or gained > 1 - baseline or lost > baseline:
        return None
    return Flips(gained=gained, lost=lost)


def independent(baseline: float, gain: float) -> Flips:
    """Flips when each checkpoint solves each task independently.

    Args:
        baseline: The old checkpoint's rate (0 to 1).
        gain: The hoped-for gain, as a share.

    Returns:
        The flips: gained = (1 - before) x after, lost = before x (1 - after).
    """
    after = baseline + gain
    return Flips(gained=(1 - baseline) * after, lost=baseline * (1 - after))


def rejection_limits(most: int, alpha: Fraction = ALPHA) -> list[int]:
    """For each number of flipped tasks, the most-lopsided splits that count.

    With m tasks flipped, the exact test rejects when the smaller of gained
    and lost is at most ``limits[m]`` (-1: no split of m can reach p < 0.05;
    m = 5 still cannot, since even 5 to 0 has p = 0.0625). It works in whole
    numbers, so it agrees with ``stats.exact_p`` exactly: with T(m, k) the
    number of ways to choose at most k of m, p = 2 T / 2^m, and Pascal's rule
    gives T(m + 1, k) = 2 T(m, k) - C(m, k). The limit never falls and rises
    by at most one per extra flip.

    Args:
        most: The largest number of flipped tasks needed.
        alpha: The p-value a split must stay under.

    Returns:
        ``limits[m]`` for m = 0 .. ``most``.
    """
    limits = [-1]
    k, ways, choose = -1, 0, 0  # k, T(m, k) and C(m, k), at m = 0
    for m in range(most):
        if k >= 0:  # move from m to m + 1 flips, keeping k
            ways = 2 * ways - choose
            choose = choose * (m + 1) // (m + 1 - k)
        next_choose = 1 if k < 0 else choose * (m + 1 - k) // (k + 1)
        candidate = ways + next_choose  # T(m + 1, k + 1)
        # p = 2 T / 2^(m+1) < alpha, compared in whole numbers
        if 2 * candidate * alpha.denominator < alpha.numerator * 2 ** (m + 1):
            k, ways, choose = k + 1, candidate, next_choose
        limits.append(k)
    return limits


def binomial(count: int, trials: int, chance: float) -> float:
    """The chance of exactly ``count`` successes in ``trials`` tries.

    Args:
        count: Successes.
        trials: Tries.
        chance: The chance of each.

    Returns:
        The probability (worked out in logarithms, so large trials are fine).
    """
    if chance <= 0:
        return 1.0 if count == 0 else 0.0
    if chance >= 1:
        return 1.0 if count == trials else 0.0
    log = (
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * math.log(chance)
        + (trials - count) * math.log1p(-chance)
    )
    return math.exp(log)


class PowerCalculator:
    """The exact power of the paired test, for one kind of flips.

    The chance of rejecting given m flips is kept, since every n reuses it.
    """

    def __init__(self, flips: Flips, limits: list[int]) -> None:
        """Set up for one kind of flips.

        Args:
            flips: How the checkpoints disagree.
            limits: ``rejection_limits`` up to the largest n to be asked.
        """
        self.flips = flips
        self.limits = limits
        self.share = flips.gained / flips.flipped if flips.flipped else 0.0
        self.known: dict[int, float] = {}

    def reject_given(self, flipped: int) -> float:
        """The chance the test rejects when exactly ``flipped`` tasks flipped.

        Args:
            flipped: How many tasks flipped.

        Returns:
            The chance the smaller of gained and lost is at most the limit.
        """
        if flipped not in self.known:
            limit = self.limits[flipped]
            chance = 0.0
            if limit >= 0:
                for gained in range(flipped + 1):
                    if min(gained, flipped - gained) <= limit:
                        chance += binomial(gained, flipped, self.share)
            self.known[flipped] = chance
        return self.known[flipped]

    def power(self, tasks: int) -> float:
        """The chance a gain of this kind shows up as p < 0.05 on ``tasks`` tasks.

        Args:
            tasks: Paired tasks.

        Returns:
            The exact power.
        """
        chance = self.flips.flipped
        if chance <= 0:
            return 0.0
        mean = tasks * chance
        spread = WINDOW_SIGMAS * math.sqrt(tasks * chance * (1 - chance)) + 1
        low, high = max(0, int(mean - spread)), min(tasks, int(mean + spread) + 1)
        return sum(
            binomial(m, tasks, chance) * self.reject_given(m)
            for m in range(low, high + 1)
        )


def normal_tasks(flips: Flips, power: float, alpha: float = float(ALPHA)) -> int:
    """The McNemar sample size by the normal approximation (Connor, 1987).

    Args:
        flips: How the checkpoints disagree.
        power: The power wanted.
        alpha: The p-value line.

    Returns:
        ``ceil((z(1 - alpha/2) sqrt(f) + z(power) sqrt(f - d^2))^2 / d^2)``,
        with f the flipped share and d the gain.

    Raises:
        ValueError: If the flips carry no gain (d = 0 needs infinitely many).
    """
    if flips.gain <= 0:
        raise ValueError("no gain: no number of tasks can show it")
    normal = NormalDist()
    z_alpha, z_power = normal.inv_cdf(1 - alpha / 2), normal.inv_cdf(power)
    f, d = flips.flipped, flips.gain
    root = z_alpha * math.sqrt(f) + z_power * math.sqrt(f - d * d)
    return math.ceil(root**2 / d**2)


def tasks_needed(flips: Flips, power: float, limits: list[int]) -> Answer:
    """The smallest number of paired tasks whose exact power reaches ``power``.

    Args:
        flips: How the checkpoints disagree.
        power: The power wanted (0 to 1).
        limits: ``rejection_limits`` up to ``EXACT_MAX_TASKS``.

    Returns:
        The answer: exact when it is at most ``EXACT_MAX_TASKS``, else the
        normal approximation, labelled. An exact answer also says from which
        n the power stayed at or above the target over the next
        ``STEADY_WINDOW`` values of n checked.
    """
    calculator = PowerCalculator(flips, limits)
    for tasks in range(1, EXACT_MAX_TASKS + 1):
        reached = calculator.power(tasks)
        if reached >= power:
            last = min(EXACT_MAX_TASKS, tasks + STEADY_WINDOW)
            below = [
                n for n in range(tasks + 1, last + 1) if calculator.power(n) < power
            ]
            steady = below[-1] + 1 if below else tasks
            return Answer(tasks, reached, "exact", steady)
    return Answer(normal_tasks(flips, power), None, "normal approximation")


def simulate_power(flips: Flips, tasks: int, runs: int, seed: int) -> float:
    """Run the paired comparison many times on made-up checkpoints, and count.

    Args:
        flips: How the checkpoints disagree, per task.
        tasks: Paired tasks per run.
        runs: How many runs.
        seed: The random generator's seed, so the draw repeats.

    Returns:
        The share of runs in which ``exact_p`` was below 0.05.
    """
    rng = random.Random(seed)
    rejected = 0
    for _ in range(runs):
        gained = lost = 0
        for _ in range(tasks):
            draw = rng.random()
            if draw < flips.gained:
                gained += 1
            elif draw < flips.flipped:
                lost += 1
        rejected += exact_p(gained, lost) < ALPHA
    return rejected / runs


def read_pairs(path: Path = PAIRS_FILE) -> list[dict[str, object]]:
    """Read chapter 24's nine checkpoint pairs.

    Args:
        path: The bundled file.

    Returns:
        One row per (pair, harness): cells, before, after, gained, lost.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not the expected JSON.
    """
    pairs = json.loads(path.read_text(encoding="utf-8")).get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path.name}: no pairs")
    return pairs


def churn_share(pairs: list[dict[str, object]]) -> tuple[int, int]:
    """Count the flipped cells over every pair.

    Args:
        pairs: ``read_pairs``' rows.

    Returns:
        Flipped cells and all cells.
    """
    flipped = sum(int(p["gained"]) + int(p["lost"]) for p in pairs)
    return flipped, sum(int(p["cells"]) for p in pairs)


def percent(share: float) -> str:
    """Write a share as a percentage with one decimal.

    Args:
        share: 0 to 1.

    Returns:
        Such as ``31.5%``.
    """
    return f"{PERCENT * share:.1f}%"


def points(share: float) -> str:
    """Write a gain in percentage points.

    Args:
        share: Such as 0.1.

    Returns:
        Such as ``+10 points``, or ``+2.5 points`` for a gain that is not whole.
    """
    value = PERCENT * share
    if abs(value - round(value)) < FLOAT_TOLERANCE:
        return f"{value:+.0f} points"
    return f"{value:+.1f} points"


def answer_text(answer: Answer) -> str:
    """Write an answer for a table cell.

    Args:
        answer: The answer.

    Returns:
        Such as ``261 (exact, power 80.0%)``, ``78 (exact, power 80.4%;
        steady from 84)`` when the power dipped below the target again, or
        ``more than 2,000 (about 9,813, normal approximation)``.
    """
    if answer.power is None:
        return (
            f"more than {EXACT_MAX_TASKS:,} (about {answer.tasks:,}, {answer.method})"
        )
    text = f"{answer.tasks:,} (exact, power {percent(answer.power)}"
    if answer.steady_from is not None and answer.steady_from != answer.tasks:
        text += f"; steady from {answer.steady_from:,}"
    return text + ")"


def assumptions(
    baseline: float, gain: float, churn: float
) -> list[tuple[str, Flips | None]]:
    """The three assumptions, named, for one baseline and gain.

    Args:
        baseline: The old checkpoint's rate.
        gain: The hoped-for gain.
        churn: Chapter 24's flipped share.

    Returns:
        ``(name, flips)``; flips is None where an assumption cannot hold.
    """
    return [
        ("only gains", only_gains(gain)),
        ("chapter 24's churn", with_churn(baseline, gain, churn)),
        ("independent attempts", independent(baseline, gain)),
    ]


def reading_section(pairs: list[dict[str, object]]) -> list[str]:
    """Read one of chapter 24's comparisons, as the plan asks.

    Args:
        pairs: Chapter 24's pairs.

    Returns:
        Lines.

    Raises:
        ValueError: If the pairs file has no row for ``EXAMPLE_PAIR``.
    """
    found = [p for p in pairs if (p["pair"], p["harness"]) == EXAMPLE_PAIR]
    if not found:
        raise ValueError(f"no {EXAMPLE_PAIR} row in {PAIRS_FILE.name}")
    row = found[0]
    cells, gained, lost = int(row["cells"]), int(row["gained"]), int(row["lost"])
    flipped = gained + lost
    p = exact_p(gained, lost)
    return [
        f"{RULE} reading one paired comparison (chapter 24: {row['harness']}, "
        f"{row['pair']})",
        f"  {cells} tasks: {row['before']} solved before, {row['after']} after; "
        f"{gained} gained, {lost} lost, {flipped} flipped, {cells - flipped} "
        "unchanged",
        f"  only the {flipped} flipped tasks say which checkpoint is better; if "
        "the two are equally good,",
        "  each of them is a coin flip between gained and lost",
        f"  exact p = {p:.3f}: a split at least as uneven as {gained} to {lost} "
        f"comes up {PERCENT * p:.0f}% of the time by luck alone",
    ]


def pairs_section(pairs: list[dict[str, object]]) -> list[str]:
    """How often the project's checkpoints disagreed, pair by pair.

    Args:
        pairs: Chapter 24's pairs.

    Returns:
        Lines, ending with the pooled flipped share.
    """
    lines = [
        "",
        f"{RULE} how often two checkpoints disagree: chapter 24's {len(pairs)} pairs",
        "  pair         harness         cells  gained  lost  flipped  p",
    ]
    for p in pairs:
        gained, lost = int(p["gained"]), int(p["lost"])
        lines.append(
            f"  {p['pair']:<12} {p['harness']:<15} {p['cells']:>5}  {gained:>6}  "
            f"{lost:>4}  {gained + lost:>7}  {exact_p(gained, lost):.3f}"
        )
    flipped, cells = churn_share(pairs)
    lines.append(
        f"  pooled: {flipped} of {cells} cells flipped = {percent(flipped / cells)} "
        "of tasks"
    )
    return lines


def question_section(
    baseline: float, gain: float, power: float, churn: float, limits: list[int]
) -> tuple[list[str], dict[str, Answer]]:
    """The tasks one question needs, under each assumption.

    Args:
        baseline: The old checkpoint's rate.
        gain: The hoped-for gain.
        power: The power wanted.
        churn: Chapter 24's flipped share.
        limits: ``rejection_limits``.

    Returns:
        Lines, and each assumption's answer by name.
    """
    lines = [
        "",
        f"{RULE} {points(gain)} from {percent(baseline)} (to "
        f"{percent(baseline + gain)}), p < 0.05, power {percent(power)}",
        "  assumption                      flipped  gained   lost  tasks needed"
        "                  normal approx.",
    ]
    answers: dict[str, Answer] = {}
    for name, flips in assumptions(baseline, gain, churn):
        if flips is None:
            lines.append(f"  {name:<30}  cannot hold this gain")
            continue
        answer = tasks_needed(flips, power, limits)
        answers[name] = answer
        lines.append(
            f"  {name:<30}  {percent(flips.flipped):>7}  {percent(flips.gained):>6}  "
            f"{percent(flips.lost):>5}  {answer_text(answer):<28}  "
            f"{normal_tasks(flips, power):,}"
        )
    return lines, answers


def simulation_lines(flips: Flips, answer: Answer, name: str) -> list[str]:
    """Check one exact answer with a seeded simulation.

    Args:
        flips: The assumption's flips.
        answer: Its exact answer.
        name: Its name.

    Returns:
        Lines.
    """
    if answer.power is None:
        return []
    simulated = simulate_power(flips, answer.tasks, SIMULATION_RUNS, SIMULATION_SEED)
    return [
        f"  {name:<22}  {answer.tasks:>5}  {percent(simulated):>8}  "
        f"{percent(answer.power):>7}"
    ]


def grid_section(
    gain: float, power: float, churn: float, limits: list[int]
) -> list[str]:
    """The same gain from other baselines: only independence depends on it.

    Args:
        gain: The hoped-for gain.
        power: The power wanted.
        churn: Chapter 24's flipped share.
        limits: ``rejection_limits``.

    Returns:
        Lines.
    """
    lines = [
        "",
        f"{RULE} {points(gain)} from other baselines, tasks needed (power "
        f"{percent(power)})",
        "  baseline  only gains  chapter 24's churn  independent attempts",
    ]
    for baseline in BASELINE_GRID:
        cells = []
        for _, flips in assumptions(baseline, gain, churn):
            cells.append(
                "cannot hold"
                if flips is None
                else f"{tasks_needed(flips, power, limits).tasks:,}"
            )
        lines.append(
            f"  {percent(baseline):>8}  {cells[0]:>10}  {cells[1]:>18}  {cells[2]:>20}"
        )
    lines.append(
        "  only independent attempts depend on the baseline; 'cannot hold': that "
        "much flipping"
    )
    lines.append("  would need more gains than there are unsolved tasks")
    return lines


def course_section(
    baseline: float,
    gain: float,
    churn: float,
    limits: list[int],
    pairs: list[dict[str, object]],
) -> list[str]:
    """What the course's 24 tasks per harness can see.

    Args:
        baseline: The old checkpoint's rate.
        gain: The hoped-for gain.
        churn: Chapter 24's flipped share.
        limits: ``rejection_limits``.
        pairs: Chapter 24's pairs, for the changes it found beyond noise.

    Returns:
        Lines.
    """
    parts = []
    for name, flips in assumptions(baseline, gain, churn):
        if flips is not None:
            seen = PowerCalculator(flips, limits).power(COURSE_TASKS)
            parts.append(f"{percent(seen)} ({name})")
    lines = [
        "",
        f"{RULE} what {COURSE_TASKS} tasks per harness can see (chapters 24 to 28)",
        f"  a real {points(gain)} shows up as p < 0.05 in: " + "; ".join(parts),
    ]
    beyond = [p for p in pairs if exact_p(int(p["gained"]), int(p["lost"])) < ALPHA]
    lines.append(
        f"  chapter 24's changes beyond noise: {len(beyond)} of {len(pairs)} pairs"
    )
    for p in beyond:
        gained, lost = int(p["gained"]), int(p["lost"])
        lines.append(
            f"    {p['harness']} {p['pair']}: {gained} gained, {lost} lost "
            f"(p = {exact_p(gained, lost):.3f}), "
            f"{points((gained - lost) / int(p['cells']))}"
        )
    return lines


def plan_section(
    baseline: float, power: float, limits: list[int], churn: float
) -> list[str]:
    """Check the plan's example: '+10 points needs about 150 paired tasks'.

    The plan's gain is always +10 points here, whatever ``--gain`` asks.

    Args:
        baseline: The baseline used for the example.
        power: The power used.
        limits: ``rejection_limits``.
        churn: Chapter 24's flipped share.

    Returns:
        Lines.
    """
    gain = PLAN_GAIN
    flips = with_churn(baseline, gain, churn)
    needed = tasks_needed(flips, power, limits) if flips else None
    at_plan = PowerCalculator(flips, limits).power(PLAN_TASKS) if flips else 0.0
    # The most flipping for which PLAN_TASKS tasks still reach the power.
    most, share = None, gain
    while share <= 1:
        candidate = with_churn(baseline, gain, share)
        if candidate is not None:
            if PowerCalculator(candidate, limits).power(PLAN_TASKS) < power:
                break
            most = share
        share = round(share + FLIP_STEP, 6)
    lines = [
        "",
        f"{RULE} the plan said '{points(gain)} at p < 0.05 needs about "
        f"{PLAN_TASKS} paired tasks'",
    ]
    if needed is not None:
        lines.append(
            f"  measured, with chapter 24's churn: {answer_text(needed)}; "
            f"{PLAN_TASKS} tasks give power {percent(at_plan)}"
        )
    else:
        lines.append(
            f"  chapter 24's churn cannot hold {points(gain)} from {percent(baseline)}"
        )
    if most is not None:
        lines.append(
            f"  {PLAN_TASKS} tasks reach power {percent(power)} only while at most "
            f"{percent(most)} of tasks flip"
        )
    else:
        lines.append(f"  {PLAN_TASKS} tasks reach power {percent(power)} at no churn")
    return lines


def one_line(baseline: float, gain: float, power: float = DEFAULT_POWER) -> str:
    """power.py's answer in one line, for a run that just measured a baseline.

    Args:
        baseline: The measured rate (0 to 1).
        gain: The hoped-for gain.
        power: The power wanted.

    Returns:
        Such as ``to see +10 points from 65.0% at p < 0.05, 80% of the time:
        about 261 paired tasks (exact McNemar, with chapter 24's 31.5% of
        tasks flipping)``, or why there is no answer.

    Raises:
        ValueError: If the gain is not above 0 or the power not between 0
            and 100%.
    """
    if gain > 0 and baseline + gain > 1:
        return f"no room for {points(gain)} from {percent(baseline)}"
    check_question(baseline, gain, power)
    flipped, cells = churn_share(read_pairs())
    churn = flipped / cells
    flips = with_churn(baseline, gain, churn)
    if flips is None:
        flips, how = independent(baseline, gain), "independent attempts"
    else:
        how = f"chapter 24's {percent(churn)} of tasks flipping"
    answer = tasks_needed(flips, power, rejection_limits(EXACT_MAX_TASKS))
    if answer.power is None:
        count, method = (
            f"more than {EXACT_MAX_TASKS:,}",
            f"{answer.method} {answer.tasks:,}",
        )
    else:
        count, method = f"about {answer.tasks:,}", "exact McNemar"
    return (
        f"to see {points(gain)} from {percent(baseline)} at p < 0.05, "
        f"{PERCENT * power:.0f}% of the time: {count} paired tasks "
        f"({method}, with {how})"
    )


def share_argument(text: str) -> float:
    """Read a share given as 0.6, 60 or 60%.

    A number up to 1 is a fraction and a larger one a percentage, so a bare
    ``1`` (100%, or 1%?) is refused: write ``1%`` or ``100%``.

    Args:
        text: The value typed.

    Returns:
        The share, 0 to 1.

    Raises:
        argparse.ArgumentTypeError: If it is not a number between 0 and 100%,
            or is a bare 1.
    """
    if text.strip() == AMBIGUOUS_ONE:
        raise argparse.ArgumentTypeError("'1' could be 1% or 100%: write it with %")
    try:
        value = float(text.rstrip("%"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if text.endswith("%") or value > 1:
        value /= PERCENT
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not between 0 and 100%")
    return value


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=share_argument, default=DEFAULT_BASELINE)
    parser.add_argument("--gain", type=share_argument, default=DEFAULT_GAIN)
    parser.add_argument("--power", type=share_argument, default=DEFAULT_POWER)
    args = parser.parse_args()
    try:
        check_question(args.baseline, args.gain, args.power)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    """Print the reading, the answer to the question, and the plan's check."""
    args = parse_args()
    pairs = read_pairs()
    flipped, cells = churn_share(pairs)
    churn = flipped / cells
    limits = rejection_limits(EXACT_MAX_TASKS)
    print("power.py · how many paired tasks before a gain can be told from noise")
    print(
        "  test      exact McNemar, two-sided: a change counts when p < 0.05 "
        "(chapter 24's exact_p)"
    )
    print(
        f"  power     {percent(args.power)}: the chance a real gain of that size "
        "shows up as p < 0.05"
    )
    print("")
    lines = reading_section(pairs) + pairs_section(pairs)
    question, answers = question_section(
        args.baseline, args.gain, args.power, churn, limits
    )
    lines += question
    lines += [
        "",
        f"{RULE} check: {SIMULATION_RUNS:,} simulated comparisons at each answer "
        f"(seed {SIMULATION_SEED})",
        "  assumption              tasks  p < 0.05    exact",
    ]
    for name, flips in assumptions(args.baseline, args.gain, churn):
        if flips is not None and name in answers:
            lines += simulation_lines(flips, answers[name], name)
    lines += grid_section(args.gain, args.power, churn, limits)
    lines += course_section(args.baseline, args.gain, churn, limits, pairs)
    lines += plan_section(args.baseline, args.power, limits, churn)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
