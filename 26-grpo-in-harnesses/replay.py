# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print the project's two recorded GRPO runs, step by step, from their files.

``uv run replay.py attempt1`` is the project's first run, which went wrong;
``uv run replay.py attempt2`` the second, with the token bonus, the lower
learning rate, bias correction and the length guard. Both read the copies
bundled in ``data/`` (the project's ``runs/main/grpo_attempt1`` and
``runs/main/grpo``, slimmed by ``grpo_records.slim``: numbers only), or the
project's own files with ``--records ~/multi-harness-rl/runs/main``.

Each prints a table of the run's steps, with tokens per call both ways (the
median the length guard reads, and the mean), what the project's
``mhrl.cli steps`` printed and why it differs, what each harness gave the
updates, and the evaluation of the adapter the project kept, cell by cell
against the SFT checkpoint it started from (gained, lost and the exact p,
chapter 24's reading). Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpo_records
from evaluate import Record
from install import PI
from lfm_files import short_path
from noise import PROJECT_DIR, Checkpoint, project_checkpoint, read_or_fail, show_pairs
from stats import standard_error

SCRIPT = "replay.py"
HERE = Path(__file__).resolve().parent
LOGS_DIR = grpo_records.LOGS_DIR  # the project's printed outputs and summaries
PERCENT = 100
RULE = "━━"
SFT = "sft"


@dataclass(frozen=True)
class Attempt:
    """One of the project's recorded GRPO runs.

    Attributes:
        name: ``attempt1`` or ``attempt2``.
        folder: Its folder in the project's ``runs/main``.
        steps_log: The log of ``mhrl.cli steps`` on it, in ``runs/logs``.
        summary: The summary of the adapter the project evaluated.
        eval_label: Its evaluation records, ``eval_<label>.jsonl``.
        settings: How it trained, in words.
    """

    name: str
    folder: str
    steps_log: str
    summary: str
    eval_label: str
    settings: tuple[str, ...]


ATTEMPTS = {
    "attempt1": Attempt(
        "attempt1",
        "grpo_attempt1",
        "take3_steps",
        "summary_rl_attempt1",
        "rl_attempt1",
        (
            "the article's call bonus (every recorded reward reproduces with it)",
            "lr 4e-5 without AdamW's bias correction (mhrl/cli.py's comment)",
            "no length guard",
            "bfloat16 log-probabilities in the engine (chapter 15)",
        ),
    ),
    "attempt2": Attempt(
        "attempt2",
        "grpo",
        "take4_steps",
        "summary_rl",
        "rl",
        (
            "the token bonus (every recorded reward reproduces with it)",
            "lr 1e-5 with AdamW's bias correction",
            "the length guard at 150 tokens per call",
            "float32 log-probabilities in the engine",
        ),
    ),
}


@dataclass(frozen=True)
class Sources:
    """Where one run's files are read from.

    Attributes:
        run: The run's rollouts and metrics.
        logs: The project's printed outputs (``<name>.txt`` or ``.log``).
        summaries: The evaluation summaries.
        words: Where they come from, in words.
    """

    run: Path
    logs: Path
    summaries: Path
    words: str


def sources_for(attempt: Attempt, records: Path | None) -> Sources:
    """Find a run's files: the bundle, or the project's own with ``--records``.

    Args:
        attempt: The run.
        records: The project's ``runs/main`` folder, or None for the bundle.

    Returns:
        Where to read.
    """
    if records is None:
        return Sources(
            grpo_records.BUNDLE_DIR / attempt.name,
            LOGS_DIR,
            LOGS_DIR,
            f"data/project_grpo/{attempt.name} (the project's runs/main/"
            f"{attempt.folder}, slimmed)",
        )
    main = records.expanduser().resolve()
    return Sources(
        main / attempt.folder,
        main.parent / "logs",
        main,
        f"{short_path(main / attempt.folder)} (the project's own records)",
    )


def printed_series(logs: Path, name: str) -> list[int]:
    """Read the tokens-per-call column the project's ``mhrl.cli steps`` printed.

    Args:
        logs: Its logs' folder.
        name: The log's name, without its extension.

    Returns:
        The column, one number per step.

    Raises:
        grpo_records.RecordsError: If the log is missing or not that output.
    """
    for suffix in (".txt", ".log"):
        path = logs / f"{name}{suffix}"
        if path.is_file():
            lines = path.read_text("utf-8").splitlines()[1:]
            try:
                return [int(line.split()[3]) for line in lines if line.strip()]
            except (IndexError, ValueError) as error:
                raise grpo_records.RecordsError(f"{path}: not steps output") from error
    raise grpo_records.RecordsError(f"no {name}.txt or {name}.log in {logs}")


def numbers(values: list[float], digits: int = 0) -> str:
    """Write a series of numbers separated by spaces.

    Args:
        values: The numbers.
        digits: Decimal places.

    Returns:
        Such as ``46 38 40``.
    """
    return " ".join(f"{value:.{digits}f}" for value in values)


def step_table(
    steps: list[list[grpo_records.Rollout]], metrics: list[dict[str, Any]]
) -> list[str]:
    """The run's steps, one line each.

    Args:
        steps: The rollouts, step by step.
        metrics: The trainer's metrics, step by step.

    Returns:
        The lines.
    """
    lines = [
        "  step  correct  reward  calls  tokens/call    replies at  no-signal   ratio"
        "     peak",
        "                                median  mean  the 1,024 cap     groups",
    ]
    for step, metric in zip(steps, metrics, strict=True):
        length = grpo_records.step_length(step)
        replies = [reply for rollout in step for reply in rollout.replies()]
        capped = sum(1 for reply in replies if reply.capped)
        flat = f"{metric['groups_without_contrast']}/{metric['groups']}"
        lines.append(
            f"  {metric['step']:>4}  {PERCENT * metric['accuracy']:>6.0f}%  "
            f"{metric['mean_reward']:>6.2f}  {metric['mean_calls']:>5.1f}  "
            f"{length.median:>6.0f} {length.mean:>5.0f}  "
            f"{capped:>4} of {len(replies):<4}"
            f"  {flat:>9}  {metric['importance_ratio_mean']:>6.3f}  "
            f"{metric['peak_gb']:.2f} GB"
        )
    return lines


def legend(dedupe_note: bool) -> list[str]:
    """What the table's columns mean, each number defined once.

    Args:
        dedupe_note: Whether the run's records predate ``n_sampled``.

    Returns:
        The lines.
    """
    lines = [
        "  correct      the share of the step's 8 rollouts whose answer was right",
        "  tokens/call  a rollout's sampled tokens / its model calls; the median and "
        "the mean",
        "               over the step's 8 rollouts. The length guard reads the median.",
    ]
    if dedupe_note:
        lines += [
            "               These records predate n_sampled: a rollout's sampled "
            "tokens are",
            "               counted from its captured rows, each shared prefix once "
            "(on attempt 2,",
            "               which has both, the count equals n_sampled for all 48 "
            "rollouts).",
        ]
    lines += [
        "  replies      the model's replies that reached the 1,024-token cap, of "
        "all its replies",
        "  ratio        the trainer's mean importance ratio over the tokens it "
        "trained on; peak:",
        "               the trainer's MLX peak memory (both from metrics.jsonl)",
        "  each step's rollouts were sampled with the adapter the step before saved",
    ]
    return lines


def printed_lines(
    steps: list[list[grpo_records.Rollout]], printed: list[int], log: str
) -> list[str]:
    """What ``mhrl.cli steps`` printed for the run, reproduced and explained.

    Args:
        steps: The rollouts, step by step.
        printed: The column its log holds.
        log: The log's name.

    Returns:
        The lines.
    """
    ours = [round(grpo_records.steps_printed(step)) for step in steps]
    same = "the same numbers" if ours == printed else f"NOT the log's: {ours}"
    means = [grpo_records.step_length(step).mean for step in steps]
    medians = [grpo_records.step_length(step).median for step in steps]
    first_only = any(rollout.n_sampled is None for step in steps for rollout in step)
    lines = [
        f"  what the project's `python -m mhrl.cli steps` printed (runs/logs/{log}"
        ".log):",
        f"    {numbers(printed)}",
        f"    reproduced from the records: {same}. It is the mean, never the median",
    ]
    if first_only:
        lines += [
            "    and, for records without n_sampled, only each rollout's FIRST "
            "captured row.",
            f"    Every row counted, the mean is {numbers(means)};",
            f"    the median, what the guard reads, {numbers(medians)}.",
        ]
    else:
        lines.append(f"    the guard read the median: {numbers(medians)}.")
    return lines


def by_harness_right(steps: list[list[grpo_records.Rollout]]) -> list[str]:
    """Right rollouts per step, harness by harness.

    Args:
        steps: The rollouts, step by step.

    Returns:
        One line per harness.
    """
    harnesses = sorted({rollout.harness for step in steps for rollout in step})
    lines = []
    for harness in harnesses:
        counts = [
            f"{sum(r.correct for r in step if r.harness == harness)}"
            f"/{sum(1 for r in step if r.harness == harness)}"
            for step in steps
        ]
        lines.append(f"    {harness:<14} {' '.join(counts)}")
    return lines


def exposure_lines(steps: list[list[grpo_records.Rollout]]) -> list[str]:
    """What each harness gave the updates, over the whole run.

    Args:
        steps: The rollouts, step by step.

    Returns:
        The lines.
    """
    total: dict[str, grpo_records.Exposure] = {}
    for step in steps:
        for harness, given in grpo_records.exposure(step).items():
            total.setdefault(harness, grpo_records.Exposure()).add(given)
    trained = sum(given.trained for given in total.values())
    lines = [
        "  what each harness gave the updates (groups with contrast only, each "
        "shared prefix once):",
        "    harness         contrast  rows  tokens per row  tokens trained",
    ]
    for harness, given in sorted(total.items()):
        average = given.tokens / given.rows if given.rows else 0.0
        share = PERCENT * given.trained / trained if trained else 0.0
        contrast = f"{given.contrast} of {given.groups}"
        lines.append(
            f"    {harness:<15} {contrast:<8} {given.rows:>5}  {average:>14,.0f}  "
            f"{given.trained:>7,} ({share:.0f}%)"
        )
    lines.append(
        "    every step counts both harnesses alike; the tokens they gave do not"
    )
    return lines


def cap_lines(steps: list[list[grpo_records.Rollout]]) -> list[str]:
    """Say what the replies at the 1,024-token cap hold, when there are any.

    Args:
        steps: The rollouts, step by step.

    Returns:
        A line, or none.
    """
    capped = [
        reply
        for step in steps
        for rollout in step
        for reply in rollout.replies()
        if reply.capped
    ]
    if not capped:
        return []
    without = sum(1 for reply in capped if not reply.tool_call)
    return [
        f"  of the {len(capped)} replies at the cap, {without} hold no tool call: "
        "the cap cut them off before it"
    ]


def compare_with_sft(attempt: Attempt, kept: str, say: Callable[[str], None]) -> None:
    """Compare the kept adapter's evaluation with the SFT checkpoint's, cell by cell.

    Both are the project's bundled records: 24 held-out tasks per harness,
    Pi from its re-run after chapter 19's sandbox fix where there is one.

    Args:
        attempt: The run.
        kept: The adapter its summary names.
        say: Where lines go.
    """
    sft = project_checkpoint(SFT)
    if attempt.name == "attempt1":
        # Evaluated once, before chapter 19's sandbox fix, and never again:
        # its Pi cells measured the sandbox, not the model.
        records: tuple[Record, ...] = tuple(
            r
            for r in read_or_fail(PROJECT_DIR / f"eval_{attempt.eval_label}.jsonl")
            if r.harness != PI
        )
    else:
        records = project_checkpoint(attempt.eval_label).records
    label = f"{attempt.name} {Path(kept).stem}"
    pair = (Checkpoint(f"project {SFT}", sft.records), Checkpoint(label, records))
    show_pairs([pair], say)


def show(attempt: Attempt, sources: Sources, say: Callable[[str], None]) -> None:
    """Print one recorded run.

    Args:
        attempt: The run.
        sources: Where its files are.
        say: Where lines go.
    """
    rollouts = grpo_records.read_rollouts(sources.run)
    metrics = grpo_records.read_metrics(sources.run)
    steps = grpo_records.by_step(rollouts)
    old = any(rollout.n_sampled is None for rollout in rollouts)
    per_step = len(steps[0])
    say(f"{SCRIPT} {attempt.name} · the project's GRPO run, from its recorded files")
    say(f"  records   {sources.words}")
    say(
        f"            {len(steps)} steps x {per_step} rollouts = {len(rollouts)} "
        "rollouts, each step 2 tasks x 4"
    )
    say(f"  settings  {attempt.settings[0]}")
    for setting in attempt.settings[1:]:
        say(f"            {setting}")
    say("")
    for line in step_table(steps, metrics):
        say(line)
    for line in legend(old):
        say(line)
    for line in cap_lines(steps):
        say(line)
    say("")
    printed = printed_series(sources.logs, attempt.steps_log)
    for line in printed_lines(steps, printed, attempt.steps_log):
        say(line)
    say("")
    say("  right rollouts per step, by harness:")
    for line in by_harness_right(steps):
        say(line)
    say("")
    for line in exposure_lines(steps):
        say(line)
    ratios = [m["importance_ratio_mean"] for m in metrics]
    peak = max(metrics, key=lambda m: m["peak_gb"])
    say(
        f"  ratio {min(ratios):.3f} to {max(ratios):.3f} over the {len(steps)} steps; "
        f"trainer peak {peak['peak_gb']:.2f} GB (step {peak['step']})"
    )
    if attempt.name == "attempt2":
        middle = standard_error(per_step // 2, per_step)
        say(
            f"  {per_step} rollouts per step, on other tasks every step: one rollout "
            f"is {PERCENT / per_step:.1f} points,"
        )
        say(
            f"  and at 50% the standard error is {middle:.1f} points. Per-step "
            "accuracy is no progress meter;"
        )
        say("  tokens per call is steadier.")
    kept = grpo_records.named_adapter(sources.summaries / f"{attempt.summary}.json")
    say("")
    say(f"{RULE} the adapter the project kept and evaluated: {Path(kept).name}")
    say(f"  ({attempt.summary}.json names {kept})")
    if attempt.name == "attempt1":
        say(
            "  that folder was attempt 1's, before it was renamed grpo_attempt1. "
            "adapter_step004"
        )
        say(
            "  is the adapter step 5's rollouts were sampled with. pi is left out "
            "below: this"
        )
        say(
            "  adapter was evaluated once, before chapter 19's sandbox fix, and "
            "never again."
        )
    compare_with_sft(attempt, kept, say)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("attempt", choices=sorted(ATTEMPTS))
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="the project's runs/main folder (its GRPO runs, logs and summaries)",
    )
    return parser.parse_args()


def main() -> None:
    """Print the run asked for.

    Raises:
        SystemExit: If a file is missing or not what it should be.
    """
    args = parse_args()
    attempt = ATTEMPTS[args.attempt]
    try:
        show(attempt, sources_for(attempt, args.records), print)
    except (grpo_records.RecordsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error


if __name__ == "__main__":
    main()
