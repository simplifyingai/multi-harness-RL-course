# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The teaser: the project's first real RL run, paid with a bonus for fewer calls.

The project trained LFM2.5-2.6B with GRPO inside mini-swe-agent and Claude
Code on one MacBook. Its first attempt paid a right answer 1 plus the
article's bonus for fewer calls. This prints that run step by step from its
own records: the share of rollouts right, calls per rollout, tokens per call
(worked out as the project's ``python -m mhrl.cli steps`` does) and how many
replies hit the engine's 1,024-token output cap. It also checks, rollout by
rollout, that the reward really was 1 + ``tool_bonus``.

The records are copied into ``data/real_run/``: the project's
``runs/main/grpo_attempt1/metrics.jsonl`` unchanged, and its
``rollouts.jsonl`` cut down to the fields used here (the token ids are left
out; each reply's length is kept). Chapter 26 replays this run in full.

Run it with ``uv run real_run.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewards import Defaults, tool_bonus

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "data" / "real_run"
METRICS = RUN_DIR / "rl_attempt_1_metrics.jsonl"
ROLLOUTS = RUN_DIR / "rl_attempt_1_rollouts.jsonl"
# Facts about the real run (mhrl/config.py and mhrl/cli.py), written out here
# because they describe that run, not this chapter's toy.
CALL_BUDGET = 8  # max_model_calls, which the toy's BUDGET copies
OUTPUT_CAP = 1024  # max_completion_tokens
TASKS_PER_STEP = 2  # GRPO_TASKS_PER_STEP
GROUP = 4  # GRPO_GROUP: rollouts per task
PERCENT = 100
STEP_WIDTH = 4
CORRECT_WIDTH = 6
CALLS_WIDTH = 5
TOKENS_WIDTH = 11
SAME_REWARD = 1e-9  # two rewards this close are the same number


class RealRunError(Exception):
    """Raised when a bundled record is missing or not the project's format."""


@dataclass(frozen=True)
class RealRollout:
    """One rollout of the real run, as kept in data/real_run/.

    Attributes:
        step: The step it belongs to.
        correct: Whether the grader accepted its answer.
        calls: Model calls it made.
        reward: The reward the trainer used.
        first_sequence_sampled: Tokens sampled in its first training
            sequence: what ``mhrl.cli steps`` divides by calls.
        reply_tokens: Each reply's sampled tokens.
    """

    step: int
    correct: bool
    calls: int
    reward: float
    first_sequence_sampled: int
    reply_tokens: tuple[int, ...]


@dataclass(frozen=True)
class StepLine:
    """One step of the real run, ready to print.

    Attributes:
        step: The step's number.
        accuracy: The share of its rollouts that were right.
        calls: Its average calls per rollout.
        tokens_per_call: Sampled tokens per call, as ``mhrl.cli steps`` works
            it out.
        replies: How many replies its rollouts sampled.
        capped: How many of them stopped at the output cap.
    """

    step: int
    accuracy: float
    calls: float
    tokens_per_call: float
    replies: int
    capped: int


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file.

    Args:
        path: The file.

    Returns:
        One dict per line.

    Raises:
        RealRunError: If the file is missing, empty or not JSON lines.
    """
    if not path.is_file():
        raise RealRunError(f"no such file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line]
    except json.JSONDecodeError as error:
        raise RealRunError(f"{path} is not JSON lines") from error
    if not rows:
        raise RealRunError(f"{path} is empty")
    return rows


def read_rollouts(path: Path = ROLLOUTS) -> list[RealRollout]:
    """Read the real run's rollouts.

    Args:
        path: The bundled rollouts file.

    Returns:
        Every rollout, in order.

    Raises:
        RealRunError: If the file is missing or a field is not there.
    """
    try:
        return [
            RealRollout(
                step=int(row["step"]),
                correct=bool(row["correct"]),
                calls=int(row["n_calls"]),
                reward=float(row["reward"]),
                first_sequence_sampled=int(row["first_sequence_sampled"]),
                reply_tokens=tuple(int(n) for n in row["reply_tokens"]),
            )
            for row in read_lines(path)
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RealRunError(f"{path} is not this run's rollouts") from error


def step_lines(path: Path = METRICS, rollouts_path: Path = ROLLOUTS) -> list[StepLine]:
    """Work out each step's line from the run's metrics and rollouts.

    Args:
        path: The bundled metrics file.
        rollouts_path: The bundled rollouts file.

    Returns:
        One line per step.

    Raises:
        RealRunError: If a file is missing or not the project's format.
    """
    rollouts = read_rollouts(rollouts_path)
    lines = []
    for row in read_lines(path):
        try:
            step = int(row["step"])
            accuracy, calls = float(row["accuracy"]), float(row["mean_calls"])
        except (KeyError, TypeError, ValueError) as error:
            raise RealRunError(f"{path} is not the project's metrics") from error
        mine = [r for r in rollouts if r.step == step]
        # mhrl/cli.py print_steps: each rollout's first sequence's sampled
        # tokens over its calls, averaged over the rollouts.
        per_call = [r.first_sequence_sampled / r.calls for r in mine if r.calls]
        replies = [n for r in mine for n in r.reply_tokens]
        lines.append(
            StepLine(
                step=step,
                accuracy=accuracy,
                calls=calls,
                tokens_per_call=sum(per_call) / len(per_call) if per_call else 0.0,
                replies=len(replies),
                capped=sum(n >= OUTPUT_CAP for n in replies),
            )
        )
    return lines


def paid_the_call_bonus(rollouts: list[RealRollout]) -> tuple[int, int]:
    """Check that every right rollout was paid 1 + tool_bonus(calls, 8).

    Args:
        rollouts: The run's rollouts.

    Returns:
        How many right rollouts were paid exactly that, and how many right
        rollouts there were.
    """
    right = [r for r in rollouts if r.correct]
    paid = [(r.reward, 1.0 + tool_bonus(r.calls, CALL_BUDGET)) for r in right]
    matching = sum(math.isclose(got, due, abs_tol=SAME_REWARD) for got, due in paid)
    return matching, len(right)


def main() -> None:
    """Print the card.

    Raises:
        SystemExit: If a bundled record is missing or not the project's format.
    """
    try:
        lines = step_lines()
        rollouts = read_rollouts()
    except RealRunError as error:
        raise SystemExit(f"real_run.py: {error}") from error
    matching, right = paid_the_call_bonus(rollouts)
    wrong_paid = sum(r.reward != 0 for r in rollouts if not r.correct)
    print("the project's first real RL run (attempt 1): LFM2.5-2.6B trained with")
    print(
        "GRPO inside mini-swe-agent and Claude Code on one MacBook,"
        f" {TASKS_PER_STEP} tasks x {GROUP}"
    )
    print("rollouts per step")
    print(
        "reward: 1 + the call bonus (tool_bonus, at most"
        f" {Defaults.MAX_TOOL_BONUS:g}) if right, else 0"
    )
    print(
        f"  checked: {matching} of {right} right rollouts were paid exactly"
        f" 1 + tool_bonus(calls, {CALL_BUDGET}),"
    )
    print(f"  and {wrong_paid} wrong rollouts were paid anything")
    print()
    print(f"step  correct  calls  tokens/call  replies at the {OUTPUT_CAP:,}-token cap")
    for line in lines:
        right_share = round(PERCENT * line.accuracy)
        print(
            f"{line.step:>{STEP_WIDTH}}  {right_share:>{CORRECT_WIDTH}}%"
            f"  {line.calls:>{CALLS_WIDTH}.1f}"
            f"  {line.tokens_per_call:>{TOKENS_WIDTH}.0f}"
            f"  {line.capped} of {line.replies}"
        )
    print()
    print("tokens/call as the project's `python -m mhrl.cli steps` works it out")
    print("from: the project's runs/main/grpo_attempt1/metrics.jsonl and")
    print("rollouts.jsonl (token ids left out), copied into data/real_run/")


if __name__ == "__main__":
    main()
