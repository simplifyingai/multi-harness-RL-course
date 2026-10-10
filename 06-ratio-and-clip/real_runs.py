# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The closing card: the importance ratio the project's two real RL runs measured.

The project trained LFM2.5-2.6B with GRPO inside real harnesses on one
MacBook, twice. Each step sampled 2 tasks x 4 rollouts with the newest
weights, then made one update, and its trainer logged the mean importance
ratio of every token it learned from. Both logs are copied into
``data/real_runs/`` unchanged (from the project's
``runs/main/grpo_attempt1/metrics.jsonl`` and ``runs/main/grpo/metrics.jsonl``).

Run it with ``uv run real_runs.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "data" / "real_runs"
# The two runs, in the order they were made, and the log each one left.
RUNS = {
    "attempt 1": RUNS_DIR / "rl_attempt_1_metrics.jsonl",
    "attempt 2": RUNS_DIR / "rl_attempt_2_metrics.jsonl",
}
RATIO_FIELD = "importance_ratio_mean"  # the project's name for it
DECIMALS = 3  # as the project's trainer printed it
STEP_WIDTH = 6  # "step  " and "range "
STEP_NUMBER_WIDTH = 4  # the step's number, right-aligned under "step"
RUN_WIDTH = 13


class RealRunsError(Exception):
    """Raised when a bundled log is missing or not the project's format."""


def read_ratios(path: Path) -> list[float]:
    """Read one run's mean ratio per step.

    Args:
        path: The run's metrics.jsonl.

    Returns:
        One mean ratio per step, in step order.

    Raises:
        RealRunsError: If the file is missing, a line is not a step with a
            ratio, or there are no steps at all.
    """
    if not path.is_file():
        raise RealRunsError(f"no such file: {path}")
    ratios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue  # a blank line holds no step
        try:
            ratios.append(float(json.loads(line)[RATIO_FIELD]))
        except json.JSONDecodeError as error:
            raise RealRunsError(f"{path.name} has a line that is not JSON") from error
        except (KeyError, TypeError, ValueError) as error:
            raise RealRunsError(f"{path.name} has a step without a ratio") from error
    if not ratios:
        raise RealRunsError(f"{path.name} has no steps")
    return ratios


def ratio_range(ratios: list[float]) -> str:
    """Write the smallest and largest ratio, as the project's trainer rounds them.

    Args:
        ratios: One run's mean ratios.

    Returns:
        For example ``"0.995-1.000"``.
    """
    return f"{min(ratios):.{DECIMALS}f}-{max(ratios):.{DECIMALS}f}"


def main() -> None:
    """Print both runs' mean ratio per step, and their ranges."""
    try:
        runs = {name: read_ratios(path) for name, path in RUNS.items()}
    except RealRunsError as error:
        raise SystemExit(f"real_runs.py: {error}") from error
    print("the project's two real RL runs: LFM2.5-2.6B trained inside real")
    print("harnesses on one MacBook, every step 2 tasks x 4 rollouts sampled with")
    print("the newest weights, then one update")
    print()
    print("mean importance ratio per step, as the trainer logged it")
    names = list(runs)
    heading = f"{'step':<{STEP_WIDTH}}" + "".join(f"{n:<{RUN_WIDTH}}" for n in names)
    print(heading.rstrip())
    steps = max(len(ratios) for ratios in runs.values())
    for step in range(steps):
        cells = [
            f"{runs[name][step]:.{DECIMALS}f}" if step < len(runs[name]) else ""
            for name in names
        ]
        number = f"{step + 1:>{STEP_NUMBER_WIDTH}}"
        line = f"{number:<{STEP_WIDTH}}" + "".join(f"{c:<{RUN_WIDTH}}" for c in cells)
        print(line.rstrip())
    print(
        f"{'range':<{STEP_WIDTH}}"
        + "".join(f"{ratio_range(runs[name]):<{RUN_WIDTH}}" for name in names).rstrip()
    )
    print()
    print("the engine that sampled them scored tokens in 16-bit numbers (bf16) in")
    print("attempt 1 and in 32-bit numbers (float32) in attempt 2")
    print("from: the project's runs/main/grpo_attempt1/metrics.jsonl and")
    print("runs/main/grpo/metrics.jsonl, copied into data/real_runs/")


if __name__ == "__main__":
    main()
