"""The project's own SFT runs, read from their recorded metrics files.

Two files are bundled in ``data/project_run/``, copied unchanged:

- ``sft_metrics.jsonl``: the project's main SFT run
  (``runs/main/sft/metrics.jsonl``), the trainer this chapter copies, on
  120 rows captured through mini-swe-agent and Claude Code (chapter 25 builds
  those). Chapter 8's ``real_sft.py`` prints it too.
- ``dry_sft_metrics.jsonl``: the dry run before it (``runs/dry/sft``), the
  one the comment in ``mhrl/cli.py`` describes: "1e-4 over-fit the dry run
  (loss 0.10 by step 4 of 6)". The file records no learning rate; 1e-4 comes
  from that comment alone.

This module needs only the standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "data" / "project_run"
MAIN_FILE = RUN_DIR / "sft_metrics.jsonl"
DRY_FILE = RUN_DIR / "dry_sft_metrics.jsonl"
FIELDS = ("step", "loss_per_token", "tokens", "seconds", "peak_gb")


class ProjectRunError(Exception):
    """Raised when a bundled metrics file is missing or not the project's format."""


@dataclass(frozen=True)
class RecordedStep:
    """One step of a recorded run, as the project's trainer logged it.

    Attributes:
        step: The step's number, from 1.
        loss_per_token: The loss per trained token.
        tokens: The trained tokens in the step.
        seconds: How long the step took.
        peak_gb: MLX's peak memory, in GB.
    """

    step: int
    loss_per_token: float
    tokens: int
    seconds: float
    peak_gb: float


def read_run(path: Path) -> list[RecordedStep]:
    """Read one recorded run's steps.

    Args:
        path: A bundled metrics file.

    Returns:
        Its steps, in order.

    Raises:
        ProjectRunError: If the file is missing, empty, or a line lacks a field.
    """
    if not path.is_file():
        raise ProjectRunError(f"no such file: {path}")
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            steps.append(RecordedStep(*(record[name] for name in FIELDS)))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            message = f"{path.name} has a line that is not a step"
            raise ProjectRunError(message) from error
    if not steps:
        raise ProjectRunError(f"{path.name} is empty")
    return steps


def first_below(steps: list[RecordedStep], threshold: float) -> int | None:
    """Find the first step whose loss rounds to ``threshold`` or less.

    Args:
        steps: A run's steps.
        threshold: The loss per token to reach, at 2 decimals.

    Returns:
        That step's number, or None if no step got there.
    """
    for step in steps:
        if round(step.loss_per_token, 2) <= threshold:
            return step.step
    return None
