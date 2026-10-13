# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Fine-tune LFM2.5-2.6B on the demonstrations captured through the harnesses.

Chapter 14's trainer (the project's ``train_sft``), reading CAPTURED rows
instead of rendered ones: each demonstration ``collect.py`` recorded is the
proxy's ``input_ids`` + ``loss_mask``, so the model learns from exactly the
tokens the teacher wrote (mask 1) and only reads what the harness wrote
around them (mask 0), in the harness's own layout. Nothing is re-tokenized.

The project's ``demo_items`` keeps the correct, trainable demonstrations and
skips any row over 6,144 tokens (counted and printed). Then, as the project's
``load_policy`` sets it up for long agent transcripts: a 9 GB memory limit
and a 512 MB cache limit, attention in 512-row blocks, every layer's
activations recomputed in the backward pass, and logits only at the trained
positions, 512 at a time (``training.py``; ``memory_probe.py`` measures what
each one saves). One AdamW step per 8 rows at 5e-5 (bias correction on), one
epoch: 120 demonstrations make 15 steps.

Run ``uv run train_sft.py`` after ``collect.py``. ``--lr``, ``--epochs`` and
``--batch`` change the run, ``--demos`` reads another file of
demonstrations (the project's own ``demos.jsonl`` has the same format), and
``--fast`` uses Liquid's export of the model. The adapter and the per-step
metrics go to ``COURSE_HOME/25-sft-in-harnesses/sft/``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from collect import TRAIN_HARNESSES
from lfm_files import GIGABYTE, chapter_home, pick_model, short_path
from lora import LORA_LAYERS, LORA_RANK, LORA_SCALE, apply_lora, save_adapter
from mlx_lm import load
from project_run import (
    DRY_FILE,
    MAIN_FILE,
    ProjectRunError,
    RecordedStep,
    first_below,
    read_run,
)
from training import (
    CACHE_LIMIT_BYTES,
    LOGIT_CHUNK,
    MAX_TRAIN_TOKENS,
    MEMORY_LIMIT_BYTES,
    QUERY_CHUNK,
    SFT_BATCH,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    DemoRows,
    Step,
    attention,
    demo_items,
    item_from_sequence,
    layer_checkpointing,
    set_memory_limits,
    steps_per_epoch,
    train_sft,
)

SCRIPT = "train_sft.py"
DEMOS_FILE = Path("demos") / "demos.jsonl"  # collect.py's file
SFT_FOLDER = "sft"
ADAPTER_FILE = "adapter.safetensors"
METRICS_FILE = "metrics.jsonl"
NOTE_FILE = "adapter.json"  # which build, rows and settings made the adapter
DRY_STEPS_SHOWN = 4  # project_lines quotes the dry run's step 4
GIB = 1024**3
OVERFIT_LOSS = 0.10  # the project's comment: "loss 0.10 by step 4 of 6"
HARNESS_COLUMN = 14


def demos_path() -> Path:
    """Where collect.py keeps the demonstrations.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses/demos/demos.jsonl``.
    """
    return chapter_home() / DEMOS_FILE


def sft_folder() -> Path:
    """Where the adapter and its metrics go.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses/sft``.
    """
    return chapter_home() / SFT_FOLDER


def read_demos(path: Path) -> list[dict[str, Any]]:
    """Read a demonstrations file as plain records (ours or the project's).

    Args:
        path: A JSON-lines file, one rollout record per line.

    Returns:
        The records.

    Raises:
        SystemExit: If the file is missing, empty or has a broken line,
            naming the command that builds it.
    """
    if not path.is_file():
        raise SystemExit(
            f"{SCRIPT}: no demonstrations at {short_path(path)}. Build them first: "
            "./start.sh oracle, then uv run collect.py (no model needed)"
        )
    records = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"{short_path(path)}:{number} is not JSON: {error}"
            raise SystemExit(f"{SCRIPT}: {message}") from error
        if not isinstance(record, dict):
            raise SystemExit(f"{SCRIPT}: {short_path(path)}:{number} is not a record")
        records.append(record)
    if not records:
        raise SystemExit(f"{SCRIPT}: {short_path(path)} holds no demonstrations")
    return records


def rows_by_harness(records: list[dict[str, Any]], max_tokens: int) -> list[str]:
    """Say how many rows each harness gives, and how long they are.

    Args:
        records: The demonstrations.
        max_tokens: The row length cap.

    Returns:
        One line per harness.
    """
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    trained: Counter[str] = Counter()
    for record in records:
        if not (record.get("correct") and record.get("trainable")):
            continue
        for row in record.get("sequences") or []:
            # The rows demo_items keeps: none over the cap, none with nothing
            # to train, so these counts add up to its "kept".
            if len(row["input_ids"]) > max_tokens or not item_from_sequence(row):
                continue
            harness = str(record.get("harness"))
            counts[harness] += 1
            tokens[harness] += len(row["input_ids"])
            trained[harness] += sum(1 for value in row["loss_mask"] if value)
    return [
        f"    {harness:<{HARNESS_COLUMN}} {counts[harness]} rows · "
        f"{tokens[harness] / counts[harness]:,.0f} tokens on average · "
        f"{trained[harness] / counts[harness]:,.0f} trained on average"
        for harness in sorted(counts)
    ]


def print_setup(
    folder: Path,
    which: str,
    path: Path,
    rows: DemoRows,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    """Print what will be trained, on what, and how.

    Args:
        folder: The model's folder.
        which: ``ours`` or Liquid's export.
        path: The demonstrations file.
        rows: The rows ``demo_items`` kept, and its counts.
        records: The demonstrations.
        args: The command line.

    Returns:
        The number of optimizer steps the run takes.
    """
    steps = args.epochs * steps_per_epoch(len(rows.items), args.batch)
    print(f"{SCRIPT} · SFT on demonstrations captured through the harnesses")
    print(f"  model     {short_path(folder)} ({which})")
    print(
        f"  demos     {short_path(path)}: {rows.demos} demonstrations, "
        f"{rows.usable} usable (correct and trainable)"
    )
    print(
        f"  rows      {len(rows.items)} kept · {rows.too_long} over "
        f"{args.max_tokens:,} tokens skipped · {rows.empty} with nothing to "
        "train skipped"
    )
    for line in rows_by_harness(records, args.max_tokens):
        print(line)
    epochs = f"{args.epochs} epoch{'' if args.epochs == 1 else 's'}"
    print(
        f"  training  lr {args.lr:g} · batch {args.batch} · {epochs} = {steps} "
        f"steps · AdamW, bias correction on · LoRA rank {LORA_RANK}, scale "
        f"{LORA_SCALE:g}, last {LORA_LAYERS} layers"
    )
    print(
        f"  memory    limit {MEMORY_LIMIT_BYTES / GIB:.0f} GiB, cache "
        f"{CACHE_LIMIT_BYTES / GIB:.1f} GiB · attention in {QUERY_CHUNK}-row "
        f"blocks · every layer checkpointed · logits only at trained "
        f"positions, {LOGIT_CHUNK} at a time"
    )
    return steps


def load_model(folder: Path) -> nn.Module:
    """Load the 4-bit model with mlx-lm, every weight into memory.

    Args:
        folder: The build's folder (read only; nothing is downloaded).

    Returns:
        The model.
    """
    model, _ = load(str(folder))
    mx.eval(model.parameters())  # MLX is lazy: read every weight now
    return model


def project_lines(main: list[RecordedStep], dry: list[RecordedStep]) -> list[str]:
    """The project's own SFT runs, in their own box.

    Args:
        main: The project's main run's steps.
        dry: Its dry run's steps.

    Returns:
        The lines.
    """
    peak = max(step.peak_gb for step in main)
    first, last = main[0], main[-1]
    overfit = first_below(dry, OVERFIT_LOSS)
    reached = f"{OVERFIT_LOSS:.2f} at step {overfit}" if overfit else "never"
    return [
        "the project's own SFT runs (its recorded metrics, not this run)",
        f"  main run (runs/main/sft/metrics.jsonl): {len(main)} steps, loss/token "
        f"{first.loss_per_token:.2f} at step 1 -> {last.loss_per_token:.2f} at "
        f"step {last.step}, peak {peak:.2f} GB",
        f"  dry run (runs/dry/sft/metrics.jsonl, lr 1e-4 by mhrl/cli.py's "
        f"comment): {len(dry)} steps, {dry[0].loss_per_token:.2f} at step 1, "
        f"{dry[3].loss_per_token:.2f} at step 4, {reached}",
    ]


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=SFT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--batch", type=int, default=SFT_BATCH)
    parser.add_argument("--max-tokens", type=int, default=MAX_TRAIN_TOKENS)
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.max_tokens < 2:
        parser.error("--epochs and --batch must be at least 1, --max-tokens 2")
    if args.lr <= 0:
        parser.error("--lr must be above 0")
    return args


def main() -> None:
    """Read the demonstrations, train the adapter, save it, and show the project's runs.

    Raises:
        SystemExit: If there are no demonstrations or no rows, a row comes
            from a harness that is not a training harness, or a bundled
            project metrics file is missing or damaged.
    """
    args = parse_args()
    path = args.demos or demos_path()
    records = read_demos(path)
    held_out = sorted({str(r.get("harness")) for r in records} - set(TRAIN_HARNESSES))
    if held_out:  # a held-out harness's rows would make its evaluation unfair
        raise SystemExit(
            f"{SCRIPT}: {short_path(path)} holds demonstrations from "
            f"{', '.join(held_out)}, not a training harness "
            f"({', '.join(TRAIN_HARNESSES)})"
        )
    rows = demo_items(records, args.max_tokens)
    if not rows.items:
        raise SystemExit(f"{SCRIPT}: no usable row in {short_path(path)}")
    try:  # read the bundled records first: a damaged one stops before training
        project = read_run(MAIN_FILE), read_run(DRY_FILE)
    except ProjectRunError as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error
    if len(project[1]) < DRY_STEPS_SHOWN:
        raise SystemExit(
            f"{SCRIPT}: {DRY_FILE.name} has fewer than {DRY_STEPS_SHOWN} steps"
        )
    folder, which = pick_model(args.fast)
    steps = print_setup(folder, which, path, rows, records, args)
    set_memory_limits()
    model = load_model(folder)
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    out = sft_folder()
    out.mkdir(parents=True, exist_ok=True)
    # An older adapter must never sit beside this run's metrics: if this run
    # stops early, there is no adapter rather than the wrong one.
    for old in (out / ADAPTER_FILE, out / NOTE_FILE):
        old.unlink(missing_ok=True)
    metrics = out / METRICS_FILE
    metrics.write_text("")  # this run's steps only
    print(f"[sft] {len(rows.items)} training rows from {rows.demos} demos")

    def on_step(step: Step) -> None:
        """Print a step and log it as the project's trainer does.

        Args:
            step: What the step did.
        """
        print(
            f"[sft] step {step.step}/{steps} · loss/token "
            f"{step.loss_per_token:.4f} · {step.tokens:,} tokens · {step.rows} "
            f"rows · peak {step.peak_bytes / GIGABYTE:.2f} GB · "
            f"{step.seconds:.1f} s",
            flush=True,
        )
        line = {"stage": "sft", **asdict(step)}
        line["peak_gb"] = round(step.peak_bytes / GIGABYTE, 2)
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    with attention(chunked=True), layer_checkpointing(model, on=True):
        train_sft(model, rows.items, args.epochs, args.lr, args.batch, on_step)
    adapter = save_adapter(model, out / ADAPTER_FILE)
    note = {
        "model": short_path(folder),
        "which": which,
        "demos": short_path(path),
        "rows": len(rows.items),
        "steps": steps,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch": args.batch,
    }
    (out / NOTE_FILE).write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(
        f"saved     {short_path(adapter)} ({adapter.stat().st_size:,} bytes) · "
        f"metrics {short_path(metrics)} · note {NOTE_FILE}"
    )
    print("")
    for line in project_lines(*project):
        print(line)
    mx.clear_cache()


if __name__ == "__main__":
    main()
