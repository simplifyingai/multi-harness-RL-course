"""How the chapter's scripts print: steps, ratios, groups, scores and replies.

Nothing here measures anything. It only turns the scripts' results into lines
a person can read on screen (working out a share, a mean or a median at
most), so each script can stay about training and grading.
"""

from __future__ import annotations

import argparse
import math
import statistics
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grading import FINISH_LENGTH, THINK_END, Answered
from lfm_files import (
    ATTEMPT_1_LEARNING_RATE,
    BF16,
    GROUP_SIZE,
    GRPO_LEARNING_RATE,
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    ORDER_SEED,
    OUTPUT_CAP,
    QUESTIONS_PER_STEP,
    SAMPLE_SEED,
    TOKEN_BUDGET,
    TRAIN_TEMPERATURE,
    gigabytes,
    short_path,
)
from project_runs import RecordedStep
from rewards import ALL_RIGHT, ALL_WRONG, MIXED, Defaults, clip_active
from stats import Comparison, error_bar, median, pass_at_1
from tasks import Task

if TYPE_CHECKING:  # for type hints only: these modules use MLX
    from grpo_sql import Graded, Settings, StepResult
    from same_tokens import RouteResult
    from sampler import Sampled

INDENT = "  "
WRAP_WIDTH = 88
REPLY_SHOWN = 240  # characters shown of a reply's answer
REASON_SHOWN = 40  # characters shown of a verdict's reason
RIGHT = "right"
WRONG = "wrong"
NEAR_ONE = 0.01  # "within 0.99 to 1.01"
EDGE_STEPS = 5  # steps averaged at the start and the end of a run
PROJECT_DECIMALS = 3  # the project's trainer printed its ratio to 3 places
PERCENT = 100
# The clip's range, as the loss writes it: clip(ratio, 1 - 0.2, 1 + 0.2).
CLIP_RANGE = f"{1 - Defaults.CLIP_EPSILON:g} to {1 + Defaults.CLIP_EPSILON:g}"


@dataclass(frozen=True)
class ModelAnswers:
    """One model's answers in an evaluation.

    Attributes:
        label: ``base``, ``sft`` or ``sft+grpo``.
        file: Which adapter file it used, in a few words.
        answers: Its answer to every question, by task id.
        note: Whether the answers were read back or made now, and how long
            they took.
    """

    label: str
    file: str
    answers: dict[str, Answered]
    note: str


class _Sections:
    """Numbers the sections of one script's output: 1, 2, 3..."""

    count = 0


def print_title(script: str, title: str) -> None:
    """Print the script's first line, and start counting sections again.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    _Sections.count = 0
    print(f"{script} · {title}")


def heading(text: str) -> None:
    """Print the next numbered section's heading after a blank line.

    Args:
        text: What the section shows.
    """
    _Sections.count += 1
    print(f"\n{_Sections.count}. {text}")


def wrapped(text: str, indent: str = INDENT) -> str:
    """Wrap a long line, indenting its continuation lines a little more.

    Args:
        text: One line of text.
        indent: The indent of the first line.

    Returns:
        The text as one or more lines.
    """
    return textwrap.fill(
        text,
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + INDENT,
        break_on_hyphens=False,
        break_long_words=False,  # a path stays whole, even past the margin
    )


def setup_line(name: str, text: str) -> None:
    """Print one line of a script's setup, its name in a column.

    Args:
        name: Such as ``"model"``.
        text: What to say.
    """
    lines = textwrap.wrap(
        text, width=WRAP_WIDTH - 10, break_on_hyphens=False, break_long_words=False
    )
    print(f"{name:<10}{lines[0]}")
    for line in lines[1:]:
        print(f"{INDENT}{line}")


def short(text: str, limit: int) -> str:
    """Cut text to a length, marking the cut.

    Args:
        text: The text.
        limit: The most characters to keep.

    Returns:
        The text, or its start and ``...``.
    """
    return text if len(text) <= limit else text[: limit - 3] + "..."


def answer_part(reply: str) -> str:
    """Keep what a reply says after its reasoning.

    Args:
        reply: The whole reply.

    Returns:
        The text after the last ``</think>``, or ``""`` if there is none.
    """
    return reply.rsplit(THINK_END, 1)[1].strip() if THINK_END in reply else ""


def share(part: int, whole: int) -> str:
    """Write a share as a percentage with one decimal.

    Args:
        part: How many.
        whole: Out of how many.

    Returns:
        Such as ``"0.4%"`` (``"n/a"`` when there is nothing to count).
    """
    return f"{PERCENT * part / whole:.1f}%" if whole else "n/a"


def token_mean(replies: Sequence[Graded], field: str) -> float:
    """Average a per-reply mean over every token, so long replies weigh more.

    Args:
        replies: Graded replies.
        field: ``"entropy"`` or ``"full_entropy"``.

    Returns:
        The mean per sampled token (0 for no tokens).
    """
    tokens = sum(reply.tokens for reply in replies)
    total = sum(reply.tokens * getattr(reply, field) for reply in replies)
    return total / tokens if tokens else 0.0


def flat(rows: Sequence[Sequence[float]]) -> list[float]:
    """Join per-row lists into one list.

    Args:
        rows: Lists of numbers.

    Returns:
        Every number, row after row.
    """
    return [value for row in rows for value in row]


# --------------------------------------------------------------- same_tokens.py


def print_reply(task: Task, reply: Sampled, decode: Callable[[list[int]], str]) -> None:
    """Say which reply is scored: its question, length and text.

    Args:
        task: The question.
        reply: The sampled reply.
        decode: Turns token ids into text.
    """
    setup_line(
        "reply",
        f"to {task.task_id} ({task.question!r}), drawn at temperature "
        f"{TRAIN_TEMPERATURE:g} from the full distribution with seed {SAMPLE_SEED}, "
        "as grpo_sql.py's first reply is: "
        f"{len(reply.tokens)} tokens, finish_reason {reply.finish_reason!r}",
    )
    setup_line("text", repr(short(decode(reply.tokens), REPLY_SHOWN)))


def print_routes(title: str, results: Sequence[RouteResult]) -> None:
    """Compare each route's scores with a reference, token by token.

    Args:
        title: The section's heading.
        results: One per route.
    """
    heading(title)
    for result in results:
        differences = [abs(math.log(ratio)) for ratio in result.ratios]
        same = sum(ratio == 1.0 for ratio in result.ratios)
        print(wrapped(result.name))
        print(
            wrapped(
                f"identical {same} of {len(result.ratios)}; largest |difference in "
                f"log-probability| {max(differences, default=0.0):.6f}; ratio "
                f"{describe(result.ratios)}",
                INDENT * 2,
            )
        )


def print_moved(
    result: RouteResult,
    reply: Sampled,
    decode: Callable[[list[int]], str],
    count: int,
) -> None:
    """List the tokens whose ratio moved most under one route.

    Args:
        result: The route's scores.
        reply: The sampled reply, with what the sampler recorded.
        decode: Turns token ids into text.
        count: How many tokens to list.
    """
    heading(f"the {count} tokens whose ratio moved most, the trainer's route")
    print(f"{INDENT}position  token              sampler log p  trainer log p   ratio")
    order = sorted(
        range(len(result.ratios)),
        key=lambda i: abs(math.log(result.ratios[i])),
        reverse=True,
    )
    for index in sorted(order[:count]):
        token = repr(decode([reply.tokens[index]]))
        print(
            f"{INDENT}{index + 1:>8}  {token:<17}  {reply.logprobs[index]:>13.6f}  "
            f"{result.logprobs[index]:>13.6f}  {result.ratios[index]:.4f}"
        )
    print(
        wrapped(
            "position: the token's place in the reply, from 1; ratio: exp(trainer "
            "log p - sampler log p)"
        )
    )


def print_float32_shift(wide: Sequence[float], recorded: Sequence[float]) -> None:
    """Say how much casting to float32 changed the model's own numbers.

    Args:
        wide: The float32 model's log-probabilities, by the sampler's route.
        recorded: What the bfloat16 sampler recorded for the same tokens.
    """
    ratios = [math.exp(w - r) for w, r in zip(wide, recorded, strict=True)]
    print(
        wrapped(
            "the float32 model is also a slightly different model: its own numbers "
            "for these tokens, against what the bfloat16 sampler recorded, give a "
            f"ratio of {describe(ratios)}"
        )
    )


# --------------------------------------------------------------- grpo_sql.py


def print_grpo_setup(
    folder: Path,
    which: str,
    init: Path,
    args: argparse.Namespace,
    settings: Settings,
    questions: int,
) -> None:
    """Say which model, adapter and settings the run uses.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        init: The adapter training starts from.
        args: The command line.
        settings: How the run differs from the project's.
        questions: How many training questions there are.
    """
    replies = QUESTIONS_PER_STEP * GROUP_SIZE
    numbers = "float32" if settings.float32 else "4-bit weights, bfloat16 numbers"
    setup_line("model", f"{short_path(folder)}, {which}; {numbers}")
    setup_line(
        "start",
        f"{short_path(init)}: chapter 14's SFT adapter (rank {LORA_RANK}, scale "
        f"{LORA_SCALE}, the last {LORA_LAYERS} layers); sampler and trainer are one "
        "model with one adapter",
    )
    setup_line(
        "steps",
        f"{args.steps}, each {QUESTIONS_PER_STEP} of the {questions} training "
        f"questions x {GROUP_SIZE} replies = {replies} replies, in a fixed shuffled "
        f"order (the project's train_grpo, seed {ORDER_SEED})",
    )
    draw = settings.draw
    cut = f", the {draw.top_k} likeliest tokens only" if draw.top_k else ", every token"
    recorded = "bfloat16" if draw.dtype == BF16 else "float32"
    setup_line(
        "sampling",
        f"temperature {draw.temperature:g}{cut}, up to {OUTPUT_CAP:,} tokens, MLX "
        f"seed {args.seed} for the draws; each token's log-probability recorded in "
        f"{recorded}",
    )
    setup_line(
        "reward",
        f"1 + token bonus (up to {Defaults.MAX_TOOL_BONUS}, zero at {TOKEN_BUDGET:,} "
        "generated tokens) if chapter 1's grader accepts the reply's answer.txt, "
        "else 0",
    )
    rate = ""
    if args.lr != GRPO_LEARNING_RATE:
        rate = f" (--lr; the project's is {GRPO_LEARNING_RATE:g})"
    penalty = (
        f", plus {settings.kl:g} x the drift from the start adapter (--kl; the "
        "project has no such penalty)"
        if settings.kl
        else ""
    )
    setup_line(
        "update",
        f"the project's clipped loss ({CLIP_RANGE}){penalty}; one AdamW step per "
        f"step at learning rate {args.lr:g}{rate}, bias correction on "
        "(mhrl/training/grpo.py)",
    )
    if settings.retokenize:
        setup_line(
            "rows",
            "--retokenize: each reply is decoded to text and encoded again; "
            "positions whose id differs from the sampled one are masked out",
        )


def print_steps_header(settings: Settings) -> None:
    """Start the table of training steps.

    Args:
        settings: How the run differs from the project's (extra columns).
    """
    heading(
        f"training: sample {QUESTIONS_PER_STEP * GROUP_SIZE} replies, grade them, "
        "one update"
    )
    extra = ""
    if settings.retokenize:
        extra += "  changed  masked"
    if settings.kl:
        extra += "  drift"
    print(
        f"{INDENT}step  right  reward  tokens  no-signal   ratio  per token      "
        f"clipped  entropy  peak{extra}"
    )


def step_ratios(result: StepResult) -> list[float]:
    """Every trained token's ratio in one step.

    Args:
        result: What the step did.

    Returns:
        The ratios, row after row (empty when no update was made).
    """
    return flat(result.update.ratios)


def clipped_count(result: StepResult) -> int:
    """Count the step's trained tokens whose push the clip switched off.

    Args:
        result: What the step did.

    Returns:
        The count.
    """
    count = 0
    for ratios, advantage in zip(result.update.ratios, result.advantages, strict=True):
        count += sum(clip_active(ratio, advantage) for ratio in ratios)
    return count


def print_step(result: StepResult, settings: Settings) -> None:
    """Print one step as soon as it is done.

    Args:
        result: What the step did.
        settings: How the run differs from the project's.
    """
    replies = result.replies
    right = sum(reply.correct for reply in replies)
    reward = statistics.fmean(reply.reward for reply in replies)
    tokens = statistics.fmean(reply.tokens for reply in replies)
    ratios = step_ratios(result)
    if ratios:
        mean = f"{statistics.fmean(ratios):.4f}"
        spread = f"{min(ratios):.3f}-{max(ratios):.3f}"
        clipped = share(clipped_count(result), len(ratios))
    else:
        mean, spread, clipped = "n/a", "(no update)", "n/a"
    groups = len(result.outcomes)
    line = (
        f"{INDENT}{result.step:>4}  {right:>3}/{len(replies)}  {reward:>6.2f}  "
        f"{tokens:>6.0f}  {result.no_signal:>7}/{groups}  {mean:>6}  {spread:<13}  "
        f"{clipped:>7}  {token_mean(replies, 'entropy'):>7.3f}  "
        f"{gigabytes(result.peak_bytes)}"
    )
    if settings.retokenize:
        changed = sum(reply.changed for reply in replies)
        masked = sum(reply.masked for reply in replies)
        line += f"  {changed:>5}/{len(replies)}  {masked:>6}"
    if settings.kl:
        drift = flat(result.update.kl)
        line += f"  {statistics.fmean(drift):.5f}" if drift else "  n/a"
    print(line, flush=True)


def print_steps_footer(results: Sequence[StepResult], settings: Settings) -> None:
    """Explain the table's columns, and say what varies from run to run.

    Args:
        results: Every step.
        settings: How the run differs from the project's.
    """
    lines = [
        "right: replies whose answer.txt chapter 1's grader accepted; reward: their "
        "mean; tokens: mean tokens per reply; no-signal: groups whose "
        f"{GROUP_SIZE} rewards "
        "were equal, skipped",
        "ratio: the mean importance ratio over every token trained on (new "
        "probability / recorded probability, worked out before the update), as "
        "the project logs it; per token: the lowest and highest single token's",
        "clipped: the share of those tokens whose push the clip switched off; "
        "entropy: the mean entropy, per sampled token, of the distribution it "
        "was drawn from, in nats",
    ]
    if settings.retokenize:
        lines.append(
            "changed: replies whose re-encoded ids differ from the sampled ones; "
            "masked: their positions left out of training"
        )
    if settings.kl:
        lines.append(
            "drift: the mean, per trained token, of exp(ref - new) - (ref - new) "
            "- 1, ref being its log-probability under the start adapter"
        )
    for line in lines:
        print(wrapped(line))
    total = sum(result.seconds for result in results)
    sampling = sum(result.sample_seconds for result in results)
    peak = max(result.peak_bytes for result in results)
    print(
        wrapped(
            f"varies from run to run: the time ({len(results)} steps took {total:,.0f} "
            f"s, {sampling:,.0f} s of it sampling) and the peak memory "
            f"({gigabytes(peak)}, the weights included)"
        )
    )


def print_first_step(result: StepResult) -> None:
    """Show the first step's two groups reply by reply.

    Args:
        result: The first step.
    """
    heading(f"step {result.step} in detail: two questions, four replies each")
    for start in range(0, len(result.replies), GROUP_SIZE):
        group = result.replies[start : start + GROUP_SIZE]
        outcome = result.outcomes[start // GROUP_SIZE]
        signal = "no contrast: skipped" if group[0].advantage is None else "trained"
        print(wrapped(f"{group[0].task_id}: {outcome}, {signal}"))
        print(f"{INDENT * 2}reply  tokens  verdict  reward   advantage  why")
        for number, reply in enumerate(group, start=1):
            advantage = "-" if reply.advantage is None else f"{reply.advantage:+.3f}"
            verdict = RIGHT if reply.correct else WRONG
            why = short(reply.reason, REASON_SHOWN)
            print(
                f"{INDENT * 2}{number:>5}  {reply.tokens:>6}  {verdict:<7}  "
                f"{reply.reward:.4f}  {advantage:>9}  {why}"
            )
    print(
        wrapped(
            "advantage: (reward - the group's mean) / (the group's spread + "
            f"{Defaults.ADVANTAGE_EPS:g}); "
            "every token of a reply is pushed by its reply's advantage"
        )
    )


def describe(values: Sequence[float]) -> str:
    """Summarise a list of ratios: mean, lowest, highest.

    Args:
        values: Ratios.

    Returns:
        Such as ``"mean 0.9991, lowest 0.784, highest 1.287"``.
    """
    if not values:
        return "none"
    return (
        f"mean {statistics.fmean(values):.4f}, lowest {min(values):.3f}, highest "
        f"{max(values):.3f}"
    )


def print_ratio_summary(results: Sequence[StepResult], settings: Settings) -> None:
    """Show what the importance ratio did, token by token, over the whole run.

    Args:
        results: Every step.
        settings: How the run differs from the project's.
    """
    heading("the importance ratio, token by token, over the whole run")
    ratios = flat([step_ratios(result) for result in results])
    if not ratios:
        print(wrapped("no group had contrast: nothing was trained, no ratio"))
        return
    means = [statistics.fmean(step_ratios(r)) for r in results if step_ratios(r)]
    updates = len(means)
    print(
        wrapped(
            f"{len(ratios):,} tokens trained on in {updates} updates: "
            f"{describe(ratios)}; the per-step means ran from {min(means):.4f} to "
            f"{max(means):.4f}"
        )
    )
    near = sum(abs(ratio - 1) <= NEAR_ONE for ratio in ratios)
    exact = sum(ratio == 1.0 for ratio in ratios)
    eps = Defaults.CLIP_EPSILON
    outside = sum(not 1 - eps <= ratio <= 1 + eps for ratio in ratios)
    clipped = sum(clipped_count(result) for result in results)
    print(
        wrapped(
            f"exactly 1: {share(exact, len(ratios))}; within {1 - NEAR_ONE:g} to "
            f"{1 + NEAR_ONE:g}: {share(near, len(ratios))}; outside the clip's "
            f"{CLIP_RANGE}: "
            f"{share(outside, len(ratios))}; clipped: {share(clipped, len(ratios))}"
        )
    )
    engine = flat([flat(result.engine_ratios) for result in results])
    draw = settings.draw
    moved: list[float] = []  # recorded minus engine log-probability, per token
    if draw.dtype == BF16 or draw.top_k:
        what = "bfloat16 numbers" if draw.dtype == BF16 else f"top-{draw.top_k} numbers"
        print(
            wrapped(
                f"the same tokens, had the sampler recorded the engine's number "
                f"(float32, the full distribution) instead of {what}: "
                f"{describe(engine)}"
            )
        )
        moved = flat([flat(result.moved) for result in results])
        higher = sum(value > 0 for value in moved)
        lower = [value for value in moved if value < 0]
        print(
            wrapped(
                f"the recorded log-probability was above the engine's at "
                f"{share(higher, len(moved))} of these tokens, below at "
                f"{share(len(lower), len(moved))} (by at most "
                f"{-min(lower, default=0.0):.6f}), equal at "
                f"{share(len(moved) - higher - len(lower), len(moved))}; on average "
                f"{statistics.fmean(moved):+.5f} per token"
            )
        )
    if draw.top_k:
        kept = [math.exp(-value) for value in moved]
        replies = [reply for result in results for reply in result.replies]
        print(
            wrapped(
                f"the share of the probability the {draw.top_k} kept tokens held, "
                f"at the trained tokens' steps: {describe(kept)}; each token's ratio "
                "is its share times the ratio the engine's own number would give"
            )
        )
        print(
            wrapped(
                f"entropy per sampled token: {token_mean(replies, 'entropy'):.4f} "
                f"nats as drawn, {token_mean(replies, 'full_entropy'):.4f} over the "
                "full distribution at the same steps"
            )
        )
    if settings.retokenize:
        print_retokenize_summary(results)
    if settings.kl:
        drift = [
            statistics.fmean(flat(r.update.kl)) for r in results if flat(r.update.kl)
        ]
        print(
            wrapped(
                "drift per trained token, step by step: "
                + " ".join(f"{value:.5f}" for value in drift)
            )
        )


def print_retokenize_summary(results: Sequence[StepResult]) -> None:
    """Show what re-encoding the replies changed, and where the ratio moved.

    Args:
        results: Every step of a ``--retokenize`` run.
    """
    replies = [reply for result in results for reply in result.replies]
    changed = sum(reply.changed for reply in replies)
    masked = sum(reply.masked for reply in replies)
    sampled = sum(reply.tokens for reply in replies)
    print(
        wrapped(
            f"--retokenize: {changed} of {len(replies)} replies came back as other "
            f"token ids; {masked:,} of their positions were masked out "
            f"({share(masked, sampled)} of all {sampled:,} sampled tokens)"
        )
    )
    later, kept = [], []
    for result in results:
        for ratios, flags in zip(
            result.update.ratios, result.after_difference, strict=True
        ):
            for ratio, flag in zip(ratios, flags, strict=True):
                (later if flag else kept).append(ratio)
    print(
        wrapped(
            f"trained tokens after a difference, whose recorded log-probability "
            f"belongs to another context: {len(later):,}, {describe(later)}"
        )
    )
    print(wrapped(f"every other trained token: {len(kept):,}, {describe(kept)}"))
    unmasked = flat([result.unmasked for result in results])
    print(
        wrapped(
            f"the masked positions, had they been kept with the log-probability "
            f"recorded at the same position (scored, never trained on): "
            f"{len(unmasked):,}, {describe(unmasked)}"
        )
    )


def print_groups(results: Sequence[StepResult]) -> None:
    """Count the groups by outcome, and how reply length moved over the run.

    Args:
        results: Every step.
    """
    heading("the groups, and the replies' length")
    outcomes = [outcome for result in results for outcome in result.outcomes]
    counts = {name: outcomes.count(name) for name in (ALL_RIGHT, MIXED, ALL_WRONG)}
    print(
        wrapped(
            f"{len(outcomes)} groups: {counts[MIXED]} {MIXED}, {counts[ALL_RIGHT]} "
            f"{ALL_RIGHT}, {counts[ALL_WRONG]} {ALL_WRONG} (always skipped: four "
            "rewards of 0)"
        )
    )
    flat_right = sum(
        1
        for result in results
        for start in range(0, len(result.replies), GROUP_SIZE)
        if result.outcomes[start // GROUP_SIZE] == ALL_RIGHT
        and result.replies[start].advantage is None
    )
    print(
        wrapped(
            f"of the {counts[ALL_RIGHT]} all-right groups, "
            f"{counts[ALL_RIGHT] - flat_right} had contrast from the token bonus "
            "alone (their replies differed in length) and were trained on, the "
            "shorter replies pushed up and the "
            f"longer ones down; {flat_right} tied"
        )
    )
    replies = [reply for result in results for reply in result.replies]
    pushed = [
        r for r in replies if r.correct and r.advantage is not None and r.advantage < 0
    ]
    if pushed:
        lowest = min(pushed, key=lambda reply: reply.advantage)
        print(
            wrapped(
                f"the hardest push down on a right reply: advantage "
                f"{lowest.advantage:+.3f}, a {lowest.tokens}-token reply to "
                f"{lowest.task_id} (the spread divides even a small difference in "
                "bonus up to full size)"
            )
        )
    capped = sum(reply.finish_reason == FINISH_LENGTH for reply in replies)
    right = sum(reply.correct for reply in replies)
    print(
        wrapped(
            f"{len(replies)} replies: {right} right ({share(right, len(replies))}); "
            f"{capped} cut off at the {OUTPUT_CAP:,}-token cap"
        )
    )
    per_step = [
        statistics.fmean(reply.tokens for reply in result.replies) for result in results
    ]
    print(
        wrapped(
            "mean tokens per reply, step by step: "
            + " ".join(f"{value:.0f}" for value in per_step)
        )
    )
    if len(results) >= 2 * EDGE_STEPS:
        first = [reply.tokens for r in results[:EDGE_STEPS] for reply in r.replies]
        last = [reply.tokens for r in results[-EDGE_STEPS:] for reply in r.replies]
        right_first = [
            reply.tokens
            for r in results[:EDGE_STEPS]
            for reply in r.replies
            if reply.correct
        ]
        right_last = [
            reply.tokens
            for r in results[-EDGE_STEPS:]
            for reply in r.replies
            if reply.correct
        ]
        print(
            wrapped(
                f"the first {EDGE_STEPS} steps against the last {EDGE_STEPS}: mean "
                f"tokens per reply {statistics.fmean(first):.0f} -> "
                f"{statistics.fmean(last):.0f}; right replies only "
                f"{statistics.fmean(right_first or [0]):.0f} -> "
                f"{statistics.fmean(right_last or [0]):.0f} (different questions "
                "each step, so read this with care)"
            )
        )


def print_saved(adapter: Path, size: int, metrics: Path, rollouts: Path) -> None:
    """Say where the adapter and the logs went, and what to run next.

    Args:
        adapter: The adapter file.
        size: Its size in bytes.
        metrics: The step log.
        rollouts: Every sampled reply.
    """
    heading("saved")
    print(wrapped(f"{short_path(adapter)} ({size:,} bytes)"))
    print(wrapped(f"{short_path(metrics)} (one line per step)"))
    print(wrapped(f"{short_path(rollouts)} (every reply, graded)"))
    print(wrapped(f"next: uv run eval_sql.py --adapter {adapter.name}"))


def print_project_runs(
    first: Sequence[RecordedStep], second: Sequence[RecordedStep]
) -> None:
    """Show the project's two GRPO runs, from their recorded metrics.

    Args:
        first: Attempt 1's steps.
        second: Attempt 2's steps.
    """
    heading("the project's runs (recorded, not run here)")
    print(
        wrapped(
            "each step: 2 tasks x 4 rollouts through mini-swe-agent and Claude Code, "
            "several model calls each, then one update; ratio: the mean over every "
            "token trained on"
        )
    )
    print(f"{INDENT}step  attempt 1: right  ratio   attempt 2: right  ratio")
    for index in range(max(len(first), len(second))):
        cells = []
        for run in (first, second):
            if index < len(run):
                step = run[index]
                cells.append(f"{PERCENT * step.accuracy:>14.1f}%  {step.ratio:.4f}")
            else:
                cells.append(" " * 23)
        print(f"{INDENT}{index + 1:>4}  {cells[0]}  {cells[1]}".rstrip())
    for name, run, how in (
        ("attempt 1", first, "the engine's log-probabilities in bfloat16"),
        ("attempt 2", second, "in float32"),
    ):
        ratios = [step.ratio for step in run]
        peak = max(step.peak_gb for step in run)
        print(
            wrapped(
                f"{name}: ratio {min(ratios):.{PROJECT_DECIMALS}f} to "
                f"{max(ratios):.{PROJECT_DECIMALS}f} ({how}); peak {peak:.2f} GB"
            )
        )
    print(
        wrapped(
            "attempt 1 also paid its bonus for fewer calls and trained at "
            f"{ATTEMPT_1_LEARNING_RATE:g} without bias correction (chapter 26 replays "
            "it); attempt 2 used this chapter's settings: the token bonus, "
            f"{GRPO_LEARNING_RATE:g} with bias correction"
        )
    )


# --------------------------------------------------------------- eval_sql.py


def print_eval_setup(
    folder: Path, which: str, models: Sequence[ModelAnswers], tasks: Sequence[Task]
) -> None:
    """Say which model, adapters and questions the evaluation uses.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        models: The three models, answers already gathered.
        tasks: The held-out questions asked.
    """
    setup_line("model", f"{short_path(folder)}, {which}")
    domains = " and ".join(sorted({task.domain for task in tasks}))
    setup_line(
        "questions",
        f"{len(tasks)} held out, over {domains}: databases no training question "
        "comes from",
    )
    setup_line(
        "replies",
        f"greedy, one per question, up to {OUTPUT_CAP:,} tokens, to chapter 14's chat; "
        "each reply's SQL run and its result graded through answer.txt",
    )
    for model in models:
        setup_line(model.label, f"{model.file}; {model.note}")


def print_verdicts(models: Sequence[ModelAnswers], tasks: Sequence[Task]) -> None:
    """Print every question's verdict and reply length under each model.

    Args:
        models: The three models' answers.
        tasks: The questions, in order.
    """
    heading("every question's verdict, and its reply's length in tokens")
    names = "  ".join(f"{model.label:<8}" for model in models)
    columns = "  ".join(f"{model.label:>8}" for model in models)
    print(f"{INDENT}{'task':<19}  {names}  tokens: {columns}")
    for task in tasks:
        cells, sizes = [], []
        for model in models:
            answered = model.answers[task.task_id]
            cells.append(f"{RIGHT if answered.correct else WRONG:<8}")
            sizes.append(f"{answered.tokens:>8}")
        verdicts, lengths = "  ".join(cells), "  ".join(sizes)
        print(f"{INDENT}{task.task_id:<19}  {verdicts}  {'':<7} {lengths}")


def print_scores(models: Sequence[ModelAnswers], tasks: Sequence[Task]) -> None:
    """Print each model's pass@1 with its error bar.

    Args:
        models: The three models' answers.
        tasks: The questions.
    """
    heading(f"pass@1 on the {len(tasks)} held-out questions, with error bars")
    for model in models:
        right = sum(model.answers[task.task_id].correct for task in tasks)
        score, bar = pass_at_1(right, len(tasks)), error_bar(right, len(tasks))
        print(
            f"{INDENT}{model.label:<9} {right:>2} of {len(tasks)} = {score:5.1f}% "
            f"± {bar:.1f} points"
        )
    print(
        wrapped(
            "pass@1: the share answered right on the one greedy try; error bar: ± one "
            "standard error, sqrt(p x (1 - p) / n), how far the score would "
            f"typically move by luck alone if another {len(tasks)} questions like "
            "these were asked"
        )
    )


def print_comparisons(pairs: Sequence[tuple[str, str, Comparison]]) -> None:
    """Compare models question by question: gained, lost and the exact p-value.

    Args:
        pairs: (before, after, comparison) for each pair compared.
    """
    heading("question by question")
    for before, after, comparison in pairs:
        gained, lost = len(comparison.gained), len(comparison.lost)
        print(
            wrapped(
                f"{before} against {after}: gained {gained}, lost {lost}, right both "
                f"times {comparison.both}, wrong both times {comparison.neither}; "
                f"exact p = {comparison.p_value:.4f}"
            )
        )
        for name, ids in (("gained", comparison.gained), ("lost", comparison.lost)):
            if ids:
                print(wrapped(f"{name}: {', '.join(ids)}", INDENT * 2))
    print(
        wrapped(
            "gained: wrong before, right after; lost: the reverse; exact p: the chance "
            "of a split at least this uneven among the questions that changed, if the "
            "two models were equally good (McNemar's exact test)"
        )
    )


def print_lengths(models: Sequence[ModelAnswers], tasks: Sequence[Task]) -> None:
    """Compare the models' reply lengths, overall and on questions both got right.

    Args:
        models: The three models' answers (the last two are compared).
        tasks: The questions.
    """
    heading("reply length")
    print(f"{INDENT}model      median  mean  longest  cut off at the cap")
    for model in models:
        tokens = [model.answers[task.task_id].tokens for task in tasks]
        capped = sum(
            model.answers[task.task_id].finish_reason == FINISH_LENGTH for task in tasks
        )
        print(
            f"{INDENT}{model.label:<9}  {median(tokens):>6.0f}  "
            f"{statistics.fmean(tokens):>4.0f}  {max(tokens):>7,}  {capped:>18}"
        )
    before, after = models[-2], models[-1]
    both = [
        task.task_id
        for task in tasks
        if before.answers[task.task_id].correct and after.answers[task.task_id].correct
    ]
    if not both:
        return
    old = [before.answers[i].tokens for i in both]
    new = [after.answers[i].tokens for i in both]
    shorter = sum(n < o for o, n in zip(old, new, strict=True))
    longer = sum(n > o for o, n in zip(old, new, strict=True))
    same = len(both) - shorter - longer
    print(
        wrapped(
            f"on the {len(both)} questions right under both {before.label} and "
            f"{after.label}: median {median(old):.0f} -> {median(new):.0f} tokens, "
            f"mean {statistics.fmean(old):.0f} -> {statistics.fmean(new):.0f}; shorter "
            f"on {shorter}, longer on {longer}, the same on {same}"
        )
    )


def print_changed(
    before: ModelAnswers,
    after: ModelAnswers,
    ids: Sequence[str],
    tasks: Mapping[str, Task],
) -> None:
    """Show questions whose verdict changed: both replies, side by side.

    Args:
        before: The earlier model's answers.
        after: The later model's answers.
        ids: The questions to show.
        tasks: Every question, by id.
    """
    if not ids:
        return
    heading(f"questions whose verdict changed from {before.label} to {after.label}")
    for task_id in ids:
        task = tasks[task_id]
        print(f"{INDENT}{task_id}: {task.question}")
        for model in (before, after):
            answered = model.answers[task_id]
            said = answer_part(answered.reply)
            if not said:
                said = f"(no </think> in {answered.tokens} tokens: still reasoning)"
            verdict = RIGHT if answered.correct else f"{WRONG}: {answered.reason}"
            print(
                wrapped(
                    f"{model.label} ({answered.tokens} tokens, {verdict}): "
                    f"{short(said, REPLY_SHOWN)!r}",
                    INDENT * 2,
                )
            )
