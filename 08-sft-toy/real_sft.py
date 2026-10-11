# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The project's real SFT run: LFM2.5-2.6B copying the answer-key teacher.

The project ran its answer-key teacher through mini-swe-agent and Claude Code
on 60 training tasks each, and fine-tuned LFM2.5-2.6B on the 120 recorded
demonstrations: batch 8, one epoch, 15 steps, loss on the teacher's tokens
only. This prints that run from its own records: how many of the tokens were
the teacher's, the loss per token at each step, and the held-out evaluation
before and after, task by task (tasks gained and lost, with an exact p-value).

The records are copied into ``data/real_sft/``: the project's
``runs/main/sft/metrics.jsonl`` unchanged; its ``demos.jsonl`` cut down to
counts (the token ids left out, how many there were and how many the loss
mask marked kept); and its ``eval_base*.jsonl`` and ``eval_sft*.jsonl`` cut
down to each task's harness, verdict and calls. Pi's come from the
``-pi-fixed`` files, re-run after a sandbox fix, as the project reports them.

Run it with ``uv run real_sft.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "data" / "real_sft"
METRICS = RUN_DIR / "sft_metrics.jsonl"
DEMOS = RUN_DIR / "demos.jsonl"
PI = "pi"
HARNESSES = ("mini-swe-agent", "claude-code", PI)
# Each checkpoint's evaluation: the main file, and the one Pi's results come from.
EVALS = {
    "base": (RUN_DIR / "eval_base.jsonl", RUN_DIR / "eval_base-pi-fixed.jsonl"),
    "sft": (RUN_DIR / "eval_sft.jsonl", RUN_DIR / "eval_sft-pi-fixed.jsonl"),
}
BATCH = 8  # the project's SFT_BATCH
LOSS_DECIMALS = 2
P_DECIMALS = 2
CALL_DECIMALS = 1
PERCENT = 100
HALF = 2  # the exact test is two-sided
STEP_WIDTH = 6
STEP_NUMBER_WIDTH = 4  # the step's number, right-aligned under "step"
NOT_AVAILABLE = "n/a"  # calls per solve when no task was solved both times
LOSS_WIDTH = 16
HARNESS_WIDTH = 16
CELL_WIDTH = 8
INDENT = "  "


class RealSftError(Exception):
    """Raised when a bundled record is missing or not the project's format.

    It does not derive from the chapter's CourseError, because this script
    imports nothing from the folder.
    """


@dataclass(frozen=True)
class Outcome:
    """One task in one harness at one checkpoint.

    Attributes:
        correct: Whether it was solved.
        calls: Model calls the rollout made.
    """

    correct: bool
    calls: int


def read_jsonl(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read a bundled JSON-lines file and check every line has its fields.

    Args:
        path: The file.
        fields: Field names every line must have.

    Returns:
        One record per line.

    Raises:
        RealSftError: If the file is missing, a line is not JSON, a field is
            missing, or there are no lines at all.
    """
    if not path.is_file():
        raise RealSftError(f"no such file: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RealSftError(f"{path.name} has a line that is not JSON") from error
        if not isinstance(record, dict) or any(f not in record for f in fields):
            raise RealSftError(f"{path.name} has a line without {', '.join(fields)}")
        records.append(record)
    if not records:
        raise RealSftError(f"{path.name} is empty")
    return records


def int_field(record: dict[str, Any], name: str, path: Path) -> int:
    """Read one whole-number field of a record.

    Args:
        record: The record.
        name: The field.
        path: The file it came from (for the message).

    Returns:
        The number.

    Raises:
        RealSftError: If the field is not a whole number.
    """
    value = record[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RealSftError(f"{path.name} has a {name} that is not a whole number")
    return value


def read_eval(checkpoint: str) -> dict[tuple[str, str], Outcome]:
    """Read one checkpoint's evaluation, Pi from its re-run file.

    Args:
        checkpoint: ``"base"`` or ``"sft"``.

    Returns:
        Each (task, harness) cell's outcome.

    Raises:
        RealSftError: If a file is missing or not the project's format.
    """
    main_file, pi_file = EVALS[checkpoint]
    cells = {}
    fields = ("task_id", "harness", "correct", "n_calls")
    for path, wanted in ((main_file, False), (pi_file, True)):
        for record in read_jsonl(path, fields):
            if (record["harness"] == PI) == wanted:
                key = (record["task_id"], record["harness"])
                calls = int_field(record, "n_calls", path)
                cells[key] = Outcome(bool(record["correct"]), calls)
    return cells


def exact_p(gained: int, lost: int) -> float:
    """The exact two-sided McNemar p-value for tasks gained and lost.

    If training changed nothing, each changed task is a coin flip between
    gained and lost; p is the chance of a split at least this uneven.

    Args:
        gained: Tasks solved after but not before.
        lost: Tasks solved before but not after.

    Returns:
        The p-value (1 when nothing changed).
    """
    changed = gained + lost
    if not changed:
        return 1.0
    tail = sum(math.comb(changed, k) for k in range(min(gained, lost) + 1))
    return min(1.0, HALF * tail / HALF**changed)


def calls_per_solve(cells: list[Outcome]) -> str:
    """Write the average calls of some solved cells.

    Args:
        cells: The cells (may be none).

    Returns:
        For example ``"7.8"``, or ``"n/a"`` for no cells.
    """
    if not cells:
        return NOT_AVAILABLE
    return f"{sum(c.calls for c in cells) / len(cells):.{CALL_DECIMALS}f}"


def print_demos(demos: list[dict[str, Any]]) -> int:
    """Print what the demonstrations held, and return their trained tokens.

    Args:
        demos: The cut-down demonstration records.

    Returns:
        How many tokens the loss mask marked as the teacher's.
    """
    right = sum(bool(d["correct"] and d["trainable"]) for d in demos)
    per = {h: sum(d["harness"] == h for d in demos) for h in HARNESSES}
    split = ", ".join(f"{count} {h}" for h, count in per.items() if count)
    tokens = sum(sum(d["tokens"]) for d in demos)
    trained = sum(sum(d["trained_tokens"]) for d in demos)
    print(f"{len(demos)} demonstrations ({split}), {right} right and kept")
    print(f"tokens in them: {tokens:,}")
    print(f"{INDENT}the teacher's replies (loss mask 1): {trained:,}, trained")
    print(
        f"{INDENT}the rest ({tokens - trained:,}: the harness's instructions and"
        " the tool outputs): read, never trained on"
    )
    return trained


def print_steps(steps: list[dict[str, Any]], trained: int) -> None:
    """Print the loss per token at each step.

    Args:
        steps: The metrics records, one per step.
        trained: The demonstrations' trained tokens, to check against.
    """
    epochs = len({s["epoch"] for s in steps})
    unit = "epoch" if epochs == 1 else "epochs"
    print(f"batch {BATCH}, {epochs} {unit}: {len(steps)} steps")
    print()
    print(f"{'step':<{STEP_WIDTH}}{'loss per token':<{LOSS_WIDTH}}tokens")
    for s in steps:
        number = f"{s['step']:>{STEP_NUMBER_WIDTH}}"
        loss = f"{s['loss_per_token']:.{LOSS_DECIMALS}f}"
        print(f"{number:<{STEP_WIDTH}}{loss:<{LOSS_WIDTH}}{s['tokens']:,}")
    total = sum(s["tokens"] for s in steps)
    same = "the same" if total == trained else "NOT the same"
    print(f"tokens trained, all steps: {total:,} ({same} as the teacher's above)")


def table_line(name: str, cells: list[str]) -> str:
    """Lay out one row of the evaluation table.

    Args:
        name: The harness.
        cells: Its cells.

    Returns:
        The row, without trailing spaces.
    """
    line = f"{name:<{HARNESS_WIDTH}}" + "".join(f"{c:<{CELL_WIDTH}}" for c in cells)
    return line.rstrip()


def print_eval(
    base: dict[tuple[str, str], Outcome], sft: dict[tuple[str, str], Outcome]
) -> None:
    """Print the held-out evaluation before and after SFT, task by task.

    Args:
        base: The base model's outcomes.
        sft: The SFT checkpoint's outcomes.
    """
    tasks = len({task for task, _ in base})
    print(
        f"held-out tasks, base -> SFT, task by task ({tasks} tasks per harness, greedy)"
    )
    heads = ["base", "SFT", "gained", "lost", "p", "calls per solve"]
    print(table_line("harness", heads))
    for harness in HARNESSES:
        keys = sorted(k for k in base if k[1] == harness)
        gained = sum(not base[k].correct and sft[k].correct for k in keys)
        lost = sum(base[k].correct and not sft[k].correct for k in keys)
        both = [k for k in keys if base[k].correct and sft[k].correct]
        calls = (
            f"{calls_per_solve([base[k] for k in both])} ->"
            f" {calls_per_solve([sft[k] for k in both])} ({len(both)} tasks)"
        )
        cells = [
            str(sum(base[k].correct for k in keys)),
            str(sum(sft[k].correct for k in keys)),
            str(gained),
            str(lost),
            f"{exact_p(gained, lost):.{P_DECIMALS}f}",
            calls,
        ]
        print(table_line(harness, cells))
    print(
        "calls per solve: on the tasks solved both before and after; p: exact"
        " McNemar test"
    )


def main() -> None:
    """Print the project's SFT run from its records.

    Raises:
        SystemExit: If a bundled record is missing or not the project's format.
    """
    try:
        demos = read_jsonl(
            DEMOS, ("harness", "correct", "trainable", "tokens", "trained_tokens")
        )
        steps = read_jsonl(METRICS, ("epoch", "step", "loss_per_token", "tokens"))
        base, sft = read_eval("base"), read_eval("sft")
        if set(base) != set(sft):
            raise RealSftError("the base and SFT evaluations hold different tasks")
        for demo in demos:
            for name in ("tokens", "trained_tokens"):
                if not all(isinstance(n, int) for n in demo[name]):
                    raise RealSftError(f"demos.jsonl has a {name} that is not counts")
    except RealSftError as error:
        raise SystemExit(f"real_sft.py: {error}") from error
    print("the project's SFT run: LFM2.5-2.6B fine-tuned on the answer-key teacher's")
    print("demonstrations, captured through mini-swe-agent and Claude Code")
    print()
    trained = print_demos(demos)
    print_steps(steps, trained)
    print()
    print_eval(base, sft)
    print()
    print("from: the project's runs/main/sft/metrics.jsonl, sft/demos.jsonl and")
    print("runs/main/eval_*.jsonl, copied (cut down) into data/real_sft/")


if __name__ == "__main__":
    main()
