# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Fine-tune LFM2.5-2.6B on the training rows: supervised fine-tuning (SFT).

The project's SFT trainer (``mhrl/training/sft.py``) on make_rows.py's 180
rows: a fresh LoRA adapter built the chapter-13 way, checked to change
nothing before training; then each row scored in one forward pass, each
reply token's log-probability read from the position before it, the loss
minus their sum, gradients added up row by row and divided by the step's
reply-token count, and one AdamW step per 8 rows (learning rate 5e-5, bias
correction on). One epoch is 23 steps.

Run it with ``uv run train_sft.py`` (a few minutes). ``--lr`` and
``--epochs`` change the run (``--lr 1e-4 --epochs 3`` is the over-fitting
one), ``--limit N`` trains on N balanced rows, ``--reason fixed`` starts
every reply with the project's teacher's one sentence instead of a reason
written from the query, and ``--fast`` uses Liquid's export and 40 rows
(5 steps). The adapter is saved under
``$COURSE_HOME/14-sft-real-model``: ``sft.safetensors`` for the project's
settings, a name saying what differs otherwise. If make_rows.py has not run,
the rows are built first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import report
from lfm import load_folder
from lfm_files import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    SFT_BATCH,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    adapter_name,
    adapter_path,
    metrics_path,
    pick_model,
    rows_path,
)
from lora import (
    B_SUFFIX,
    LORA_SEED,
    adapter_weights,
    apply_lora,
    save_adapter,
    zero_b_count,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper
from project_run import DRY_FILE, MAIN_FILE, ProjectRunError, read_run
from rows import (
    REASON_STEPS,
    REASON_STYLES,
    Row,
    TrainItem,
    balanced,
    build_row,
    item_from_sequence,
    read_rows,
    write_rows,
)
from sft import (
    Step,
    all_logits,
    max_difference,
    set_memory_limits,
    steps_per_epoch,
    train_sft,
)
from tasks import TRAIN_FILE, load_tasks

SCRIPT = "train_sft.py"
TITLE = "SFT on the real model"
FAST_LIMIT = 40  # the plan's fast mode: --limit 40, 5 steps
TRAIN_SPLIT = "train"


def training_rows(
    tokenizer: TokenizerWrapper, limit: int | None, style: str
) -> tuple[list[Row], str]:
    """Build the training rows, and check them against make_rows.py's file.

    Building them takes seconds, so they are always built again from the
    bundled questions: a file left by an older run can never be trained on.

    Args:
        tokenizer: The model's tokenizer.
        limit: Keep this many rows, picked round-robin over database and
            difficulty as the project's ``--limit`` does; None keeps all.
        style: The kind of reason the replies start with.

    Returns:
        The rows, and a few words saying how they compare with the file.
    """
    path = rows_path(TRAIN_SPLIT, style)
    tasks = load_tasks(TRAIN_FILE)
    rows = [build_row(tokenizer, task, style=style) for task in tasks]
    try:
        same = read_rows(path) == rows
        source = f"identical to make_rows.py's {path.name}"
        if not same:
            source = f"{path.name} was out of date and is rewritten"
    except FileNotFoundError:
        same, source = False, f"make_rows.py had not run: written to {path.name}"
    except (OSError, ValueError):
        same, source = False, f"{path.name} could not be read and is rewritten"
    if not same:
        write_rows(path, rows)
    if limit is not None:
        wanted = {task.task_id for task in balanced(tasks, limit)}
        rows = [row for row in rows if row.task_id in wanted]
        source += f"; --limit {limit}, balanced over database and difficulty"
    return rows, source


def trainable(rows: list[Row]) -> list[TrainItem]:
    """Turn rows into the trainer's items, refusing a row with nothing to train.

    Args:
        rows: The training rows.

    Returns:
        One item per row, as the project's ``item_from_sequence`` makes it.

    Raises:
        SystemExit: If there are no rows, or a row has no reply token after
            position 0 (it would be dropped without a word otherwise).
    """
    items = [item_from_sequence(row.as_sequence()) for row in rows]
    empty = [row.task_id for row, item in zip(rows, items, strict=True) if not item]
    if not rows or empty:
        raise SystemExit(f"{SCRIPT}: no rows, or rows with nothing to train: {empty}")
    return [item for item in items if item]


def check_fresh_adapter(model: nn.Module, row: Row) -> None:
    """Add the adapter the chapter-13 way, and show that it changes nothing.

    Args:
        model: The loaded model, before LoRA.
        row: A row to score before and after.
    """
    before = all_logits(model, row.input_ids)
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    after = all_logits(model, row.input_ids)
    weights = adapter_weights(model)
    pairs = sum(name.endswith(B_SUFFIX) for name in weights)  # one B per pair
    numbers = sum(array.size for array in weights.values())
    difference = max_difference(after, before)
    zeros = zero_b_count(model)
    report.print_fresh(pairs, numbers, zeros, difference, after.shape, LORA_SEED)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--lr", type=float, default=SFT_LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=None, help="train on N rows")
    parser.add_argument(
        "--reason",
        choices=REASON_STYLES,
        default=REASON_STEPS,
        help="each reply's reason: steps from the gold SQL, or one fixed sentence",
    )
    parser.add_argument(
        "--fast", action="store_true", help=f"Liquid's export, {FAST_LIMIT} rows"
    )
    args = parser.parse_args()
    if args.fast and args.limit is None:
        args.limit = FAST_LIMIT
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be above 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    """Check a fresh adapter, train it, save it, and show the project's runs.

    Raises:
        SystemExit: If a bundled project metrics file is missing or damaged.
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    try:  # read the bundled records first: a damaged one stops before training
        project_runs = read_run(MAIN_FILE), read_run(DRY_FILE)
    except ProjectRunError as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error
    set_memory_limits()
    loaded = load_folder(folder)
    model, tokenizer = loaded.model, loaded.tokenizer
    rows, source = training_rows(tokenizer, args.limit, args.reason)
    items = trainable(rows)
    steps = args.epochs * steps_per_epoch(len(rows), SFT_BATCH)
    report.print_title(SCRIPT, TITLE)
    report.print_train_setup(folder, which, rows, source, args.epochs, args.lr, steps)
    report.print_reason_style(args.reason, rows[0], tokenizer.decode)
    check_fresh_adapter(model, rows[0])

    adapter = adapter_path(adapter_name(args.lr, args.epochs, args.limit, args.reason))
    metrics = metrics_path(adapter)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("")  # this run's steps only
    done: list[Step] = []

    def on_step(step: Step) -> None:
        """Print a step, log it as the project's trainer does, and keep it.

        Args:
            step: What the step did.
        """
        report.print_step(step)
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "sft", **asdict(step)}) + "\n")
        done.append(step)

    report.print_steps_header()
    train_sft(model, items, args.epochs, args.lr, SFT_BATCH, on_step)
    report.print_steps_footer(done)
    report.print_epochs(done)
    save_adapter(model, adapter)
    report.print_saved(adapter, adapter.stat().st_size, metrics)
    report.print_project_runs(*project_runs)
    mx.clear_cache()


if __name__ == "__main__":
    main()
