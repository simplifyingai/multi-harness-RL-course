"""The project's two GRPO runs, read from their recorded metrics files.

Both are bundled in ``data/project_runs/``, copied unchanged from the project
(chapter 6 bundles the same two files):

- ``rl_attempt1_metrics.jsonl`` is ``runs/main/grpo_attempt1/metrics.jsonl``:
  8 steps with the article's bonus for fewer calls, learning rate 4e-5
  without AdamW's bias correction, and an engine that worked its
  log-probabilities out in bfloat16. It went wrong (chapter 26).
- ``rl_attempt2_metrics.jsonl`` is ``runs/main/grpo/metrics.jsonl``: 6 steps
  with this chapter's settings (the token bonus, 1e-5 with bias correction)
  and float32 log-probabilities.

Each step drew 2 tasks x 4 rollouts through real agent programs (several
model calls each) and made one update. This module needs only the standard
library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "data" / "project_runs"
ATTEMPT_1 = RUNS_DIR / "rl_attempt1_metrics.jsonl"
ATTEMPT_2 = RUNS_DIR / "rl_attempt2_metrics.jsonl"
FIELDS = ("step", "accuracy", "mean_reward", "importance_ratio_mean", "peak_gb")
STAGE = "grpo"


class ProjectRunError(Exception):
    """Raised when a bundled metrics file is missing or not the project's format."""


@dataclass(frozen=True)
class RecordedStep:
    """One GRPO step as the project's trainer logged it.

    Attributes:
        step: The step's number, from 1.
        accuracy: The share of its 8 rollouts that were correct.
        mean_reward: Their average reward.
        ratio: The mean importance ratio over every token it trained on.
        peak_gb: MLX's peak memory, in GB.
    """

    step: int
    accuracy: float
    mean_reward: float
    ratio: float
    peak_gb: float


def read_run(path: Path) -> list[RecordedStep]:
    """Read one recorded run's steps.

    Args:
        path: A bundled metrics file.

    Returns:
        Its steps, in order.

    Raises:
        ProjectRunError: If the file is missing, empty, or a step lacks a field.
    """
    if not path.is_file():
        raise ProjectRunError(f"no such file: {path}")
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("stage") != STAGE:
                raise KeyError("stage")
            steps.append(RecordedStep(*(record[name] for name in FIELDS)))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
            message = f"{path.name} has a line that is not a GRPO step"
            raise ProjectRunError(message) from error
    if not steps:
        raise ProjectRunError(f"{path.name} is empty")
    return steps
