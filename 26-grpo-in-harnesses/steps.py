# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""What each GRPO step's rollouts did: the project's `python -m mhrl.cli steps`.

Accuracy alone hides the failure the article warns about, replies that keep
growing, so this prints sampled tokens per model call beside it. The
project's ``print_steps`` printed one number for that, the mean over the
step's rollouts; this prints the median too, because the length guard reads
the median (``grpo_records.py`` defines both, once).

``uv run steps.py`` reads this chapter's run (``grpo_harness.py``'s
``COURSE_HOME/26-grpo-in-harnesses/grpo/grpo``), ``--run attempt1`` another
of its runs, and ``--dir FOLDER`` any run's folder: the project's own
``runs/main/grpo``, or the copies bundled in ``data/project_grpo/``. Only
Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from pathlib import Path

import grpo_records
from lfm_files import chapter_home, short_path

SCRIPT = "steps.py"
GRPO_FOLDER = "grpo"  # grpo_harness.py's runs: COURSE_HOME/26-.../grpo/<run>
RUN_NAME = "grpo"
PERCENT = 100
HEADER = "step  correct  calls  tokens/call: median   mean  rollout s (median)"


def step_line(step: list[grpo_records.Rollout]) -> str:
    """One step's line.

    Args:
        step: The step's rollouts.

    Returns:
        Its correct share, calls, tokens per call both ways and rollout time.
    """
    length = grpo_records.step_length(step)
    correct = PERCENT * sum(r.correct for r in step) / len(step)
    calls = statistics.fmean(r.n_calls for r in step)
    # The project's print_steps takes sorted(seconds)[n // 2]: the upper of the
    # two middle times when n is even, which is statistics.median_high.
    seconds = statistics.median_high(r.seconds for r in step)
    return (
        f"{step[0].step:4d}  {correct:6.0f}%  {calls:5.1f}  "
        f"{length.median:19.0f}  {length.mean:5.0f}  {seconds:18.0f}"
    )


def stop_note(folder: Path) -> str | None:
    """Say so when the length guard stopped the run.

    Args:
        folder: The run's folder.

    Returns:
        A line about the stop, or None when the run was not stopped.
    """
    for row in grpo_records.read_jsonl(folder / grpo_records.METRICS_FILE):
        if row.get("stopped"):
            kept = row.get("kept") or "the base model"
            return (
                f"step {row['step']}: the length guard stopped the run (median "
                f"{row['tokens_per_call']:.0f}); it kept {kept}"
            )
    return None


def print_steps(folder: Path, say: Callable[[str], None] = print) -> None:
    """Print what each step's rollouts did, from the run's own records.

    Args:
        folder: The run's folder (``rollouts.jsonl`` and ``metrics.jsonl``).
        say: Where lines go.
    """
    rollouts = grpo_records.read_rollouts(folder)
    say(f"{SCRIPT} · {short_path(folder)}")
    say(HEADER)
    for step in grpo_records.by_step(rollouts):
        say(step_line(step))
    say(
        "tokens/call: a rollout's sampled tokens / its model calls; the median and "
        "the mean"
    )
    say("  over the step's rollouts. The length guard reads the median.")
    if any(rollout.n_sampled is None for rollout in rollouts):
        say(
            "  these records predate n_sampled: sampled tokens are counted from "
            "the captured"
        )
        say("  rows, each shared prefix once")
    note = stop_note(folder)
    if note:
        say(note)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=RUN_NAME, help="one of this chapter's runs")
    parser.add_argument("--dir", type=Path, default=None, help="any run's folder")
    return parser.parse_args()


def main() -> None:
    """Print the steps of the run asked for.

    Raises:
        SystemExit: If the run has no records yet, or they are not a GRPO run's.
    """
    args = parse_args()
    mine = chapter_home() / GRPO_FOLDER / args.run
    folder = args.dir.expanduser() if args.dir else mine
    try:
        print_steps(folder)
    except grpo_records.RecordsError as error:
        raise SystemExit(f"{SCRIPT}: no GRPO steps recorded yet ({error})") from error


if __name__ == "__main__":
    main()
