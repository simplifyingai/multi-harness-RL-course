"""How this chapter prints: one group by hand, steps, a run's end, the seed tables.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``reward_lab.py`` can stay about rewards and the loop.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

from environment import (
    ACTIONS_PER_LONG_REPLY,
    BUDGET,
    CRASHED,
    GUESSED,
    LONG_REPLY,
    OUTPUT_CAP,
    Call,
)
from grpo import GROUP_SIZE, QUESTIONS_PER_STEP, Defaults, group_advantages
from policy import DEFAULT_TEMPERATURE
from questions import LEVELS, QuestionSet
from replies import MIN_REPLY, SPREAD, START_LENGTH
from rewards import LEFT_OUT, RIGHT, RULE_NOTES, SCORED_ZERO

if TYPE_CHECKING:  # only for type hints: reward_lab.py imports this module
    from reward_lab import Group, LevelScore, Rollout, RunResult, Settings, Step

PERCENT = 100
SHOWN_DECIMALS = 2  # rewards and advantages
MEAN_DECIMALS = 3  # a group's mean, so 1.065 shows whole
SPREAD_DECIMALS = 4  # a tiny spread needs more places
QUARTERS = 4  # the quarter table and --trace split a run into quarters
LEVEL_WIDTH = 8  # "medium  "
STEP_WIDTH = 3  # "step   7", as chapters 5 and 6 print it
COUNT_WIDTH = 2
TOTAL_WIDTH = 4  # "15 of 15  ": the total and the gap after it
ROW_WIDTH = 28  # the name column of the quarter table
CELL_WIDTH = 10  # one column of the quarter table
FIRST_CELL_WIDTH = 12  # its first column, whose heading says "steps"
PAID_WIDTH = 5  # "1.07 "
ADVANTAGE_WIDTH = 11  # "no contrast"
ANSWER_WIDTH = 11  # "2025-06-08 "
INDENT = "  "
SEPARATOR = " · "
IN_ONE_REPLY = "+"  # actions sharing one reply: "LOOK+QUERY_A"
NO_ANSWER = "(none)"
NO_VERDICT = "(crashed)"
CUT_TEXT = "cut off"
NOT_AVAILABLE = "n/a"
# The seed tables.
SEED_WIDTH = 6  # " 0    "
METRIC_WIDTH = 15  # "0.19 -> 0.91" and room
ARROW = " -> "
START_DIVISOR = 10  # "start" is the first tenth of a run's steps...
END_DIVISOR = 4  # ...and "end" its last quarter


class Tally(NamedTuple):
    """What the rollouts of some stretch of a run did, added up.

    Attributes:
        rollouts: How many rollouts.
        correct: How many were right.
        paid: The rewards of those that were paid, added up.
        scored: How many were paid (a crash left unscored is not).
        calls: How many calls they made.
        tokens: How many tokens their replies held.
        cut_off: How many replies were cut off.
        long: How many replies held two actions.
        guesses: How many rollouts ended with SUBMIT_GUESS.
    """

    rollouts: int
    correct: int
    paid: float
    scored: int
    calls: int
    tokens: int
    cut_off: int
    long: int
    guesses: int

    @property
    def accuracy(self) -> float:
        """The share of rollouts that were right, 0 to 1."""
        return self.correct / self.rollouts if self.rollouts else 0.0

    @property
    def reward(self) -> float:
        """The average reward of the rollouts that were paid."""
        return self.paid / self.scored if self.scored else 0.0

    @property
    def calls_per_rollout(self) -> float:
        """The average number of calls per rollout."""
        return self.calls / self.rollouts if self.rollouts else 0.0

    @property
    def tokens_per_reply(self) -> float:
        """The average reply length, in tokens (a cut-off one counts the cap)."""
        return self.tokens / self.calls if self.calls else 0.0

    @property
    def cut_off_share(self) -> float:
        """The share of replies that were cut off, 0 to 1."""
        return self.cut_off / self.calls if self.calls else 0.0

    @property
    def long_share(self) -> float:
        """The share of replies that held two actions, 0 to 1."""
        return self.long / self.calls if self.calls else 0.0

    @property
    def guess_share(self) -> float:
        """The share of rollouts that ended with a guess, 0 to 1."""
        return self.guesses / self.rollouts if self.rollouts else 0.0


def tally(steps: Sequence[Step]) -> Tally:
    """Add up what the rollouts of some steps did.

    Args:
        steps: The steps.

    Returns:
        Their totals.
    """
    rollouts = [r for step in steps for r in step.rollouts]
    calls = [call for r in rollouts for call in r.calls]
    paid = [r.reward for r in rollouts if r.reward is not None]
    return Tally(
        rollouts=len(rollouts),
        correct=sum(r.correct for r in rollouts),
        paid=sum(paid),
        scored=len(paid),
        calls=len(calls),
        tokens=sum(call.tokens for call in calls),
        cut_off=sum(call.cut_off for call in calls),
        long=sum(call.long for call in calls),
        guesses=sum(r.stop == GUESSED for r in rollouts),
    )


def added(tallies: Sequence[Tally]) -> Tally:
    """Add several tallies together, field by field.

    Args:
        tallies: Any tallies.

    Returns:
        Their sum.
    """
    return Tally(*(sum(column) for column in zip(*tallies, strict=True)))


def share(part: float) -> str:
    """Write a share from 0 to 1 as a whole percentage.

    Args:
        part: The share.

    Returns:
        For example ``"12%"``.
    """
    return f"{round(PERCENT * part)}%"


def signed(value: float) -> str:
    """Write a number with its sign and two decimals, never as -0.00.

    Args:
        value: The number.

    Returns:
        For example ``"+1.33"``.
    """
    return f"{round(value, SHOWN_DECIMALS) + 0.0:+.{SHOWN_DECIMALS}f}"


def plain(value: float) -> str:
    """Write a reward as short as it reads: ``1``, ``0.2`` or ``1.08``.

    Args:
        value: The reward.

    Returns:
        The number with at most two decimals and no trailing zeros.
    """
    return f"{round(value, SHOWN_DECIMALS) + 0.0:g}"


def print_show_group(rewards: Sequence[float]) -> None:
    """Print one group's advantages, worked out step by step.

    When every reward is at least 1 (every answer right, with a bonus on
    top), it also shows the same group paid for right answers only.

    Args:
        rewards: One reward per rollout.
    """
    print(f"rewards     [{', '.join(plain(r) for r in rewards)}]")
    advantages = group_advantages(list(rewards))
    if advantages is None:
        print("no contrast, skipped: every rollout scored the same")
        return
    mean = statistics.fmean(rewards)
    spread = statistics.pstdev(rewards)
    print(f"mean        {mean:.{MEAN_DECIMALS}f}")
    print(f"spread      {spread:.{SPREAD_DECIMALS}f}  (standard deviation)")
    print(f"advantages  (reward - mean) / (spread + {Defaults.ADVANTAGE_EPS:g})")
    print(f"            [{', '.join(signed(a) for a in advantages)}]")
    if min(rewards) >= RIGHT:
        ones = ", ".join(plain(RIGHT) for _ in rewards)
        print()
        print(f"paid for right answers only, the same rollouts are [{ones}]:")
        print("no contrast, skipped")


def print_header(
    settings: Settings, questions: QuestionSet, guess: tuple[str, int]
) -> None:
    """Print the run's settings, its reward rule and how replies work.

    Args:
        settings: The run's knobs.
        questions: The fair questions, split by city.
        guess: What SUBMIT_GUESS hands in, and how many training questions it
            is right for.
    """
    print(
        f"reward_lab · reward {settings.rule} · seed {settings.seed}"
        f" · lr {settings.lr:g} · length lr {settings.length_lr:g}"
        f" · {settings.steps} steps"
    )
    print(f"reward: {RULE_NOTES[settings.rule]}")
    print(
        f"each step: {QUESTIONS_PER_STEP} questions x {GROUP_SIZE} rollouts at"
        f" temperature {DEFAULT_TEMPERATURE:g}, then one update (chapter 6's GRPO)"
    )
    extra = ", with the guessable ones" if settings.guessable else ""
    print(
        f"questions: {len(questions.training)} to train on,"
        f" {len(questions.held_out)} held out{extra};"
        f" SUBMIT_GUESS hands in {guess[0]}, right for {guess[1]}"
    )
    print(
        f"replies: about {START_LENGTH} tokens at first, give or take"
        f" {SPREAD:.0%}, never under {MIN_REPLY};"
    )
    print(
        f"  {LONG_REPLY}+ tokens hold {ACTIONS_PER_LONG_REPLY} actions; over"
        f" {OUTPUT_CAP} is cut off before its actions and ends the rollout"
    )
    if settings.crashes:
        scoring = LEFT_OUT if settings.unscored else SCORED_ZERO
        print(
            f"crashes: a sandbox dies before grading with chance"
            f" {settings.crashes:g}, {scoring}"
        )
    print()


def print_step(step: Step, metrics: Mapping[str, Any]) -> None:
    """Print one step on one line.

    Args:
        step: The step.
        metrics: Its metrics.jsonl record.
    """
    replies = metrics["replies"]
    tokens = metrics["reply_tokens"] / replies if replies else 0.0
    cut = metrics["cut_off_replies"] / replies if replies else 0.0
    paid = metrics["mean_reward"]
    reward = NOT_AVAILABLE if paid is None else f"{paid:.{SHOWN_DECIMALS}f}"
    print(
        f"step {step.number:>{STEP_WIDTH}}{SEPARATOR}reward {reward}"
        f"{SEPARATOR}correct {share(metrics['accuracy'])}"
        f"{SEPARATOR}calls {metrics['mean_calls']:.1f}"
        f"{SEPARATOR}tokens/reply {tokens:.0f}"
        f"{SEPARATOR}cut off {share(cut)}"
        f"{SEPARATOR}no-signal groups {step.no_signal}/{len(step.groups)}"
    )


def reply_text(call: Call) -> str:
    """Write one reply: its tokens and what it held.

    Args:
        call: The call.

    Returns:
        For example ``"[86] QUERY_A+SUBMIT"`` or ``"[120] cut off"``.
    """
    held = CUT_TEXT if call.cut_off else IN_ONE_REPLY.join(call.actions)
    return f"[{call.tokens}] {held}"


def answer_text(rollout: Rollout) -> str:
    """Write what a rollout handed in.

    Args:
        rollout: The rollout.

    Returns:
        The answer, ``(none)``, or ``(crashed)`` when the sandbox died.
    """
    if rollout.stop == CRASHED:
        return NO_VERDICT
    if rollout.verdict is None or not rollout.verdict.submitted:
        return NO_ANSWER
    return rollout.verdict.submitted


def print_group(group: Group) -> None:
    """Print a group in full: every rollout's replies, reward and advantage.

    Args:
        group: The group.
    """
    question = group.question
    print(
        f"step {group.step} · group {group.number} of {QUESTIONS_PER_STEP}"
        f" · {question.level} · {question.task.question}"
    )
    advantages = group.advantages or (None,) * len(group.rollouts)
    for index, (rollout, advantage) in enumerate(
        zip(group.rollouts, advantages, strict=True), start=1
    ):
        paid = NOT_AVAILABLE if rollout.reward is None else plain(rollout.reward)
        pushed = "no contrast" if advantage is None else f"A {signed(advantage)}"
        answer = answer_text(rollout)
        print(
            f"{INDENT}rollout {index}  reward {paid:<{PAID_WIDTH}}"
            f" {pushed:<{ADVANTAGE_WIDTH}} answer {answer:<{ANSWER_WIDTH}}"
            + ", ".join(reply_text(call) for call in rollout.calls)
        )


class Tracer:
    """With --trace, prints the first group of each quarter of the run.

    Attributes:
        moments: The steps whose first group is printed.
    """

    def __init__(self, steps: int) -> None:
        """Work out which steps to print.

        Args:
            steps: How many steps the run has.
        """
        quarter = math.ceil(steps / QUARTERS)
        self.moments = set(range(1, steps + 1, quarter))

    def see(self, step: Step) -> bool:
        """Print the step's first group, if this step is one of the moments.

        Args:
            step: The step just taken.

        Returns:
            Whether it printed anything.
        """
        if step.number not in self.moments:
            return False
        print()
        print_group(step.groups[0])
        print()
        return True


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


def table_row(name: str, cells: Sequence[str]) -> str:
    """Lay out one row of the quarter table: a name, then one cell per stretch.

    Args:
        name: The row's name, or "" for the heading.
        cells: The cells.

    Returns:
        The row, without trailing spaces.
    """
    first, *rest = cells
    row = f"{name:<{ROW_WIDTH}}{first:<{FIRST_CELL_WIDTH}}" + "".join(
        f"{cell:<{CELL_WIDTH}}" for cell in rest
    )
    return row.rstrip()


# The quarter table's rows after the share right: a name, and a tally's cell.
QUARTER_ROWS: tuple[tuple[str, Callable[[Tally], str]], ...] = (
    ("reward", lambda t: f"{t.reward:.{SHOWN_DECIMALS}f}"),
    ("calls per rollout", lambda t: f"{t.calls_per_rollout:.1f}"),
    ("tokens per reply", lambda t: f"{t.tokens_per_reply:.0f}"),
    ("replies cut off", lambda t: share(t.cut_off_share)),
    (
        f"replies holding {ACTIONS_PER_LONG_REPLY} actions",
        lambda t: share(t.long_share),
    ),
    ("rollouts that guessed", lambda t: share(t.guess_share)),
)


def level_share(steps: Sequence[Step], level: str) -> str:
    """Write the share of one level's rollouts that were right in some steps.

    Args:
        steps: The steps.
        level: ``"easy"``, ``"medium"`` or ``"hard"``.

    Returns:
        For example ``"67%"``, or ``n/a`` when no group of that level ran.
    """
    mine = [
        rollout
        for step in steps
        for group in step.groups
        if group.question.level == level
        for rollout in group.rollouts
    ]
    if not mine:
        return NOT_AVAILABLE
    return share(sum(r.correct for r in mine) / len(mine))


def print_quarters(steps: Sequence[Step], edges: Sequence[int]) -> None:
    """Print what the training rollouts did in each quarter of the run.

    Args:
        steps: Every step of a run.
        edges: The stretch boundaries.
    """
    stretches = [steps[start:end] for start, end in pairwise(edges)]
    tallies = [tally(stretch) for stretch in stretches]
    print("the training rollouts, in each quarter of the run")
    print(table_row("", stretch_labels(edges)))
    print(table_row("correct", [share(t.accuracy) for t in tallies]))
    for level in LEVELS:  # the same, level by level
        shares = [level_share(stretch, level) for stretch in stretches]
        print(table_row(f"{INDENT}{level}", shares))
    for name, cell in QUARTER_ROWS:
        print(table_row(name, [cell(t) for t in tallies]))
    typical = [f"{steps[end - 1].typical_length:.0f}" for end in edges[1:]]
    print(table_row("typical reply at the end", typical))


def calls_text(calls: Sequence[Call]) -> str:
    """Write a greedy rollout's replies on one line, with repeats counted.

    Args:
        calls: The rollout's calls.

    Returns:
        For example ``"LOOK+QUERY_A, SUBMIT"`` or ``"cut off"``.
    """
    replies = [CUT_TEXT if c.cut_off else IN_ONE_REPLY.join(c.actions) for c in calls]
    parts = []
    for reply, repeats in groupby(replies):  # runs of the same reply
        count = len(list(repeats))
        parts.append(reply if count == 1 else f"{reply} x{count}")
    return ", ".join(parts)


def print_greedy(scores: Sequence[LevelScore], tokens: int) -> None:
    """Print the greedy test at the end, one line per level.

    Args:
        scores: One score per level.
        tokens: How long every greedy reply is.
    """
    print(
        f"greedy test at the end (the most likely action, every reply {tokens} tokens)"
    )
    for score in scores:
        print(
            f"{INDENT}{score.level:<{LEVEL_WIDTH}}"
            f"training {score.training:>{COUNT_WIDTH}} of"
            f" {score.training_total:<{TOTAL_WIDTH}}"
            f"held out {score.held_out} of {score.held_out_total}"
            f"{SEPARATOR}{calls_text(score.calls)}"
        )


def print_crashes(result: RunResult) -> None:
    """Say how many sandboxes died, and what scoring them did.

    Args:
        result: The finished run.
    """
    crashed = [
        (r, advantage)
        for step in result.steps
        for group in step.groups
        for r, advantage in zip(
            group.rollouts,
            group.advantages or (None,) * len(group.rollouts),
            strict=True,
        )
        if r.stop == CRASHED
    ]
    total = sum(len(step.rollouts) for step in result.steps)
    right = [(r, a) for r, a in crashed if r.lost is not None and r.lost.correct]
    print(f"sandboxes that died before grading: {len(crashed)} of {total:,} rollouts")
    print(f"{INDENT}{len(right)} of them had handed in the right answer")
    if result.settings.unscored:
        print(f"{INDENT}{LEFT_OUT}: nothing was learned from any of them")
        return
    pushed = sum(a is not None and a < 0 for _, a in right)
    print(
        f"{INDENT}{SCORED_ZERO}: {pushed} of those right rollouts sat in a group with"
        " contrast,"
    )
    print(f"{INDENT}so every decision they made was pushed down")


def print_run_end(result: RunResult) -> None:
    """Print the quarter table, crashes if any, and the greedy test.

    Args:
        result: The finished run.
    """
    print()
    print_quarters(result.steps, result.edges)
    if result.settings.crashes:
        print()
        print_crashes(result)
    print()
    print_greedy(result.scores, result.greedy_tokens)


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


# --- seeds.py -----------------------------------------------------------------


def span_sizes(steps: int) -> tuple[int, int]:
    """Say how many steps the start and the end of a run cover.

    Args:
        steps: How many steps the run has.

    Returns:
        The start's size (its first tenth) and the end's (its last quarter),
        each at least one step.
    """
    return max(1, steps // START_DIVISOR), max(1, steps // END_DIVISOR)


def start_and_end(steps: Sequence[Step]) -> tuple[Tally, Tally]:
    """Tally a run's start and its end.

    Args:
        steps: Every step of the run.

    Returns:
        The start's tally and the end's.
    """
    first, last = span_sizes(len(steps))
    return tally(steps[:first]), tally(steps[-last:])


def print_seeds_header(
    seeds: int, steps: int, length_lr: float, guessable: bool
) -> None:
    """Print what ``seeds.py`` runs and how to read its tables.

    Args:
        seeds: How many seeds.
        steps: Steps per run.
        length_lr: The length weight's learning rate.
        guessable: Whether the guessable questions are in.
    """
    extra = " · guessable questions in" if guessable else ""
    print(
        f"seeds 0-{seeds - 1} · {steps} steps per run · length lr {length_lr:g}{extra}"
    )
    first, last = span_sizes(steps)
    print(
        f"each cell: the training rollouts of steps 1-{first} ->"
        f" those of steps {steps - last + 1}-{steps}"
    )
    print(
        f"calls: of a budget of {BUDGET} per rollout; tokens/reply: a cut-off"
        f" reply counts {OUTPUT_CAP}"
    )
    print()


SEED_COLUMNS: tuple[tuple[str, Callable[[Tally], str]], ...] = (
    ("correct", lambda t: share(t.accuracy)),
    ("reward", lambda t: f"{t.reward:.{SHOWN_DECIMALS}f}"),
    ("calls", lambda t: f"{t.calls_per_rollout:.1f}"),
    ("tokens/reply", lambda t: f"{t.tokens_per_reply:.0f}"),
    ("cut off", lambda t: share(t.cut_off_share)),
)


def seed_row(label: str, cells: Sequence[str]) -> str:
    """Lay out one row of a seed table: the seed (or "all"), then its cells.

    Args:
        label: The seed's number as text, or "all".
        cells: One cell per column.

    Returns:
        The row, without trailing spaces.
    """
    row = f"{label:<{SEED_WIDTH}}" + "".join(f"{c:<{METRIC_WIDTH}}" for c in cells)
    return row.rstrip()


def print_rule_table(
    rule: str, runs: Sequence[tuple[Tally, Tally]], seeds: Sequence[int]
) -> None:
    """Print one rule's table: per seed, the start -> the end.

    Args:
        rule: The reward rule.
        runs: Per seed, the start's tally and the end's.
        seeds: The seeds, in the same order.
    """
    print(f"{rule}: {RULE_NOTES[rule]}")
    print(seed_row("seed", [name for name, _ in SEED_COLUMNS]))
    rows = [(f"{s:>{COUNT_WIDTH}}", run) for s, run in zip(seeds, runs, strict=True)]
    pooled = (added([s for s, _ in runs]), added([e for _, e in runs]))
    for label, (start, end) in [*rows, ("all", pooled)]:
        cells = [f"{cell(start)}{ARROW}{cell(end)}" for _, cell in SEED_COLUMNS]
        print(seed_row(label, cells))


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
