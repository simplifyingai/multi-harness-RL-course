# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Fitting training into 9 GB: one forward + backward pass per row length, measured.

An agent transcript is long: a Claude Code demonstration is about 2,600
tokens, and the rows RL trains on reach 6,000. This probe loads the 4-bit
LFM2.5-2.6B with chapter 13's adapter, sets the project's limits
(``mx.set_memory_limit`` 9 GiB, ``mx.set_cache_limit`` 512 MiB), and runs ONE
forward + backward pass on rows of 1K to 8K tokens in three settings,
reading ``mx.get_peak_memory`` after each:

* ``fused``: MLX's fused attention kernel, as shipped. Its backward pass has
  no fused version, so it builds every head's full L x L score matrix: the
  memory grows with the SQUARE of the row's length;
* ``chunked``: the project's training setting (``load_policy``): attention
  in 512-row blocks inside ``mx.checkpoint``, and every layer's activations
  recomputed in the backward pass instead of kept. Memory grows linearly;
* ``full-logits``: ``chunked``, but with scores over the whole 128,000-token
  vocabulary at every position (what ``model(tokens)`` gives) instead of
  only at the trained positions, 512 at a time.

The memory limit is a guideline MLX plans around, not a wall: the project's
fused pass peaked at 11.9 GB under it. So the probe never trusts it. Before
each pass it PROJECTS the peak (from the setting's last measured pass,
scaled by how that setting grows, or, before any, from a generous count of
the arrays the pass makes) and refuses, with the reason printed, any row
projected over the budget (9.0 GB) or over what the machine has free right
now, less 15% of its memory left for everything else. Fused attention runs
only up to 4,096 tokens; at 6,144 the probe shows the project's recorded
11.9 GB instead of measuring it again. It also refuses to start while this
chapter's engine (``./start.sh``) holds the model's memory.

Run ``uv run memory_probe.py`` (a few minutes; nothing else heavy should run
meanwhile). ``--lengths``, ``--settings``, ``--budget-gb`` and ``--keep-free``
narrow it, ``--fast`` uses Liquid's export. The results go to
``COURSE_HOME/25-sft-in-harnesses/memory_probe/results.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from lfm_files import GIGABYTE, chapter_home, pick_model, short_path
from lora import LORA_LAYERS, LORA_RANK, LORA_SCALE, apply_lora
from mlx_lm import load
from training import (
    CACHE_LIMIT_BYTES,
    LOGIT_CHUNK,
    MEMORY_LIMIT_BYTES,
    QUERY_CHUNK,
    TrainItem,
    attention,
    full_vocab_logprobs,
    layer_checkpointing,
    set_memory_limits,
    token_logprobs,
)

SCRIPT = "memory_probe.py"
LENGTHS = (1024, 2048, 4096, 6144, 8192)
BUDGET_GB = 9.0  # the trainer's share of the laptop (the project's "8-10 GB")
KEEP_FREE_PERCENT = 15.0  # of the machine's memory, left to everything else
PERCENT = 100.0
FUSED_MAX_TOKENS = 4096  # the review's rule: fused attention only up to 4K
PROJECT_FUSED_6K_GB = 11.9  # mhrl/training/common.py, chunked_attention
PROJECT_FUSED_LENGTH = 6144  # "11.9 GB peak at 6K tokens, measured"
PROJECT_CHUNKED_GB = {6144: 7.2, 8192: 9.7}  # mhrl/cli.py, MAX_TRAIN_TOKENS
PROJECT_SFT_PEAK_GB = 3.95  # runs/main/sft/metrics.jsonl, step 15
TRAINED_SHARE = 8  # the last 1/8 of each row is trained (see probe_row)
ROW_SEED = 0
FLOAT32 = 4  # bytes
# The generous count behind a first projection: float32 copies of each array
# a pass makes, more than MLX keeps. Used only before a setting's first pass.
ACTIVATION_COPIES = 8  # per layer, per token, per hidden unit
SCORE_COPIES = 4  # scores, probabilities and their two gradients
HEAD_COPIES = 3  # logits, log-probabilities, their gradient
GIB = 1024**3
MIB = 1024**2
PROBE_FOLDER = "memory_probe"
RESULTS_FILE = "results.json"
ENGINE_PID = Path("run") / "engine.pid"  # start.sh's record of the engine
COMMAND_SUFFIX = ".cmd"  # ...and of the command it started (run/engine.cmd)
PS = "/bin/ps"
VM_STAT = "/usr/bin/vm_stat"
PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
FREE_KINDS = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
NAME_COLUMN = 12
CELL_COLUMN = 9
NO_MEASUREMENT = "before any pass: a generous count of the arrays it makes"
FAILED_PREFIX = "failed:"  # a pass that ran and raised, as opposed to refused


@dataclass(frozen=True)
class Setting:
    """One way to run the training pass.

    Attributes:
        name: Its short name.
        attention: How attention runs, in words.
        head: Where logits are computed, in words.
        chunked: True for attention in 512-row blocks, False for the kernel.
        checkpointed: Whether every layer recomputes its activations.
        full_logits: True for logits everywhere, False at trained positions.
        growth: How peak memory grows with the row's length: 2 for its
            square (fused attention's backward), 1 for linearly.
        max_tokens: The longest row it may run, or None for no cap.
    """

    name: str
    attention: str
    head: str
    chunked: bool
    checkpointed: bool
    full_logits: bool
    growth: int
    max_tokens: int | None = None


SETTINGS = (
    Setting(
        "fused",
        "MLX's fused kernel, as shipped",
        f"trained positions, {LOGIT_CHUNK} at a time",
        chunked=False,
        checkpointed=False,
        full_logits=False,
        growth=2,
        max_tokens=FUSED_MAX_TOKENS,
    ),
    Setting(
        "chunked",
        f"{QUERY_CHUNK}-row blocks + every layer checkpointed",
        f"trained positions, {LOGIT_CHUNK} at a time",
        chunked=True,
        checkpointed=True,
        full_logits=False,
        growth=1,
    ),
    Setting(
        "full-logits",
        f"{QUERY_CHUNK}-row blocks + every layer checkpointed",
        "the whole vocabulary at every position",
        chunked=True,
        checkpointed=True,
        full_logits=True,
        growth=1,
    ),
)
BY_NAME = {setting.name: setting for setting in SETTINGS}


@dataclass(frozen=True)
class Shape:
    """The model's sizes a projection needs.

    Attributes:
        layers: Its layers.
        hidden: Its hidden width.
        heads: Attention heads per attention layer.
        vocab: Its vocabulary.
    """

    layers: int
    hidden: int
    heads: int
    vocab: int


@dataclass
class Outcome:
    """One (setting, length) cell: measured, or refused and why.

    Attributes:
        setting: The setting's name.
        length: The row's length in tokens.
        projected_gb: The projection made before the pass (None if none).
        basis: What the projection rested on.
        peak_gb: MLX's peak memory during the pass (None if refused).
        seconds: How long the pass took (None if refused).
        refused: Why the pass did not run ("" when it ran).
        recorded_gb: The project's own measurement of this cell, if any.
    """

    setting: str
    length: int
    projected_gb: float | None = None
    basis: str = ""
    peak_gb: float | None = None
    seconds: float | None = None
    refused: str = ""
    recorded_gb: float | None = None


@dataclass
class Probe:
    """The run's results, as written to results.json.

    Attributes:
        model: The model's folder (home as ``~``).
        budget_gb: The refusal budget.
        weights_gb: MLX's memory after loading the model and the adapter.
        outcomes: Every cell, in the order tried.
    """

    model: str
    budget_gb: float
    weights_gb: float
    outcomes: list[Outcome] = field(default_factory=list)


def shape_of(model: nn.Module) -> Shape:
    """Read the sizes a projection needs from the model's own settings.

    Args:
        model: An mlx-lm LFM2 model.

    Returns:
        Its shape.
    """
    args = model.args
    return Shape(
        layers=args.num_hidden_layers,
        hidden=args.hidden_size,
        heads=args.num_attention_heads,
        vocab=args.vocab_size,
    )


def trained_count(length: int) -> int:
    """How many positions of a probe row are trained.

    Args:
        length: The row's length.

    Returns:
        ``length // TRAINED_SHARE``, at least 1.
    """
    return max(1, length // TRAINED_SHARE)


def probe_row(length: int, vocab: int, seed: int = ROW_SEED) -> TrainItem:
    """A row of ``length`` token ids, its last eighth trained.

    The ids are drawn at random from the vocabulary: memory depends on how
    many tokens there are, not which. One in eight trained sits between the
    shares this chapter's demonstrations train (about 6.5% of a Claude Code
    row, 19% of a mini-swe-agent row).

    Args:
        length: The row's length (at least 2).
        vocab: The vocabulary's size.
        seed: The draw's seed.

    Returns:
        The row.
    """
    draw = random.Random(seed)
    ids = [draw.randrange(vocab) for _ in range(length)]
    positions = list(range(length - trained_count(length), length))
    return TrainItem(ids, positions, [0.0] * len(positions))


def first_bound(setting: Setting, length: int, baseline: float, shape: Shape) -> float:
    """A generous projection for a setting's first pass, before any measurement.

    Args:
        setting: The setting.
        length: The row's length.
        baseline: Bytes already held (the weights and the adapter).
        shape: The model's sizes.

    Returns:
        Projected peak bytes.
    """
    activations = ACTIVATION_COPIES * shape.layers * length * shape.hidden * FLOAT32
    keys_seen = QUERY_CHUNK if setting.chunked else length
    scores = SCORE_COPIES * shape.heads * length * keys_seen * FLOAT32
    head_rows = length if setting.full_logits else trained_count(length)
    head = HEAD_COPIES * head_rows * shape.vocab * FLOAT32
    return baseline + activations + scores + head


def project(
    setting: Setting,
    length: int,
    history: list[tuple[int, float]],
    baseline: float,
    shape: Shape,
) -> tuple[float, str]:
    """Project a pass's peak memory before running it.

    From the setting's last measured pass, the memory above the weights is
    scaled by (length / its length) to the setting's growth: 2 for fused
    attention, whose backward builds L x L scores, 1 for the chunked
    settings. Scaling all of it, not only the part that grows that fast,
    overestimates, which is the safe side for a refusal.

    Args:
        setting: The setting.
        length: The row's length.
        history: ``(length, peak bytes)`` of the setting's passes so far.
        baseline: Bytes already held.
        shape: The model's sizes.

    Returns:
        Projected peak bytes, and what the projection rested on.
    """
    if not history:
        return first_bound(setting, length, baseline, shape), NO_MEASUREMENT
    # The lengths run shortest first (length_list sorts them), so the last
    # pass is shorter and the factor is above 1: a projection never shrinks.
    last_length, last_peak = history[-1]
    factor = (length / last_length) ** setting.growth
    above = max(last_peak - baseline, 0.0)
    basis = f"from the {last_length:,}-token pass, x{factor:.2f} above the weights"
    return baseline + above * factor, basis


def refusal(
    setting: Setting,
    length: int,
    projected: float,
    budget: float,
    free: float | None,
    held: float,
    reserve: float = 0.0,
) -> str:
    """Say why a pass must not run, or nothing when it may.

    Args:
        setting: The setting.
        length: The row's length.
        projected: Projected peak bytes.
        budget: The budget in bytes.
        free: Bytes the machine has free now, or None when unknown.
        held: Bytes MLX holds now (the projection includes them).
        reserve: Bytes of the free memory left for everything else on the
            machine (``--keep-free``), never taken by a pass.

    Returns:
        The reason, or "" when the pass may run.
    """
    reasons = []
    if setting.max_tokens is not None and length > setting.max_tokens:
        reasons.append(
            f"{setting.name} attention runs only up to {setting.max_tokens:,} tokens"
        )
    if projected > budget:
        reasons.append(
            f"projected {projected / GIGABYTE:.2f} GB, over the "
            f"{budget / GIGABYTE:.2f} GB budget"
        )
    if free is not None and projected - held > free - reserve:
        kept = (
            f", {reserve / GIGABYTE:.1f} GB of it kept for the rest" if reserve else ""
        )
        reasons.append(
            f"it would need {(projected - held) / GIGABYTE:.1f} GB more, and the "
            f"machine has {free / GIGABYTE:.1f} GB free now{kept}"
        )
    return "; ".join(reasons)


def recorded_for(setting: Setting, length: int) -> float | None:
    """The project's own measurement of a cell, where it made one.

    Args:
        setting: The setting.
        length: The row's length.

    Returns:
        GB, or None.
    """
    if setting.name == "fused" and length == PROJECT_FUSED_LENGTH:
        return PROJECT_FUSED_6K_GB
    if setting.name == "chunked":
        return PROJECT_CHUNKED_GB.get(length)
    return None


def free_bytes(read: Callable[[], str] | None = None) -> float | None:
    """What the machine has free now: free, inactive, speculative and purgeable pages.

    Args:
        read: Returns ``vm_stat``'s output (the tests give their own).

    Returns:
        Bytes, or None when ``vm_stat`` cannot be read (not a Mac).
    """
    try:
        text = (
            read()
            if read
            else subprocess.run(
                [VM_STAT], capture_output=True, text=True, check=True
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    size = PAGE_SIZE_RE.search(text)
    if not size:
        return None
    pages = 0
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip() in FREE_KINDS:
            pages += int(value.strip().rstrip("."))
    return float(pages * int(size.group(1)))


def machine_bytes() -> float:
    """The machine's memory.

    Returns:
        Bytes (page size times physical pages).
    """
    return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def engine_running(home: Path) -> int | None:
    """The pid of this chapter's engine, if start.sh's record names a live one.

    Only the engine this chapter's ``start.sh`` started is known here; any
    other program holding memory (another engine, a film render) is covered
    by the free-memory refusal instead.

    Args:
        home: The chapter's folder under COURSE_HOME.

    Returns:
        The pid, or None.
    """
    record = home / ENGINE_PID
    try:
        pid = int(record.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # signal 0 checks that it exists, and sends nothing
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # it exists: another user's process
    # After a reboot the pid may belong to another program: start.sh keeps
    # the command it started beside the pid, so compare the two.
    command = record.with_suffix(COMMAND_SUFFIX)
    if command.is_file():
        now = subprocess.run(
            [PS, "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if now != command.read_text().strip():
            return None
    return pid


def one_pass(
    model: nn.Module, item: TrainItem, setting: Setting
) -> tuple[float, float]:
    """Run one forward + backward pass and read MLX's peak memory.

    Args:
        model: The model with its adapter.
        item: The row.
        setting: How to run it.

    Returns:
        Peak bytes during the pass, and its seconds.
    """
    scores = full_vocab_logprobs if setting.full_logits else token_logprobs

    def loss_fn(row: TrainItem) -> mx.array:
        """The SFT loss: minus the trained tokens' summed log-probability.

        Args:
            row: The row.

        Returns:
            One number.
        """
        return -scores(model, row).sum()

    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    with (
        attention(setting.chunked),
        layer_checkpointing(model, setting.checkpointed),
    ):
        loss, grads = nn.value_and_grad(model, loss_fn)(item)
        mx.eval(loss, grads)
    seconds = time.perf_counter() - started
    peak = float(mx.get_peak_memory())
    del loss, grads
    gc.collect()
    mx.clear_cache()
    return peak, seconds


def run_setting(
    model: nn.Module,
    setting: Setting,
    lengths: list[int],
    budget: float,
    baseline: float,
    say: Callable[[str], None],
    free: Callable[[], float | None] | None = None,
    on_outcome: Callable[[Outcome], None] | None = None,
    reserve: float = 0.0,
) -> list[Outcome]:
    """Run one setting over the lengths, shortest first, refusing what will not fit.

    Args:
        model: The model with its adapter.
        setting: The setting.
        lengths: Row lengths, shortest first.
        budget: The budget in bytes.
        baseline: Bytes held before any pass (weights and adapter).
        say: Where lines go.
        free: Reads the machine's free bytes (``free_bytes`` when None).
        on_outcome: Called with each cell as it ends (main saves it at once).
        reserve: Free bytes a pass must leave to the rest of the machine.

    Returns:
        One outcome per length.
    """
    shape = shape_of(model)
    read_free = free or free_bytes
    failed = 0  # the length whose pass failed, if one did
    history: list[tuple[int, float]] = []
    outcomes = []
    for length in lengths:
        projected, basis = project(setting, length, history, baseline, shape)
        outcome = Outcome(
            setting.name,
            length,
            projected_gb=round(projected / GIGABYTE, 2),
            basis=basis,
            recorded_gb=recorded_for(setting, length),
        )
        held = float(mx.get_active_memory())
        outcome.refused = refusal(
            setting, length, projected, budget, read_free(), held, reserve
        )
        if failed and not outcome.refused:  # a longer row would fail too
            outcome.refused = f"not run: the {failed:,}-token pass failed"
        tag = f"[{setting.name:<{NAME_COLUMN - 1}} {length:>6,}]"
        if outcome.refused:
            note = ""
            if outcome.recorded_gb is not None:
                note = f" · the project measured {outcome.recorded_gb} GB here"
            say(f"{tag} refused · {outcome.refused}{note}")
        else:
            try:
                peak, seconds = one_pass(model, probe_row(length, shape.vocab), setting)
            except RuntimeError as error:  # Metal refused an allocation, say
                failed = length
                outcome.refused = f"{FAILED_PREFIX} {error}"
                say(f"{tag} FAILED · {error}")
            else:
                outcome.peak_gb = round(peak / GIGABYTE, 2)
                outcome.seconds = round(seconds, 1)
                history.append((length, peak))
                say(
                    f"{tag} projected {projected / GIGABYTE:.1f} GB ({basis}) · "
                    f"peak {peak / GIGABYTE:.2f} GB · {seconds:.1f} s"
                )
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)
    return outcomes


def cell(outcome: Outcome) -> str:
    """One cell of the summary table.

    Args:
        outcome: The cell's outcome.

    Returns:
        Such as ``2.31 GB``, ``refused``, ``failed`` or ``11.9*``.
    """
    if outcome.peak_gb is not None:
        return f"{outcome.peak_gb:.2f} GB"
    if outcome.refused.startswith(FAILED_PREFIX):
        return "failed"
    if outcome.recorded_gb is not None:
        return f"{outcome.recorded_gb}*"
    return "refused"


def table(outcomes: list[Outcome], lengths: list[int]) -> list[str]:
    """The summary: peak memory per setting and row length.

    Args:
        outcomes: Every cell.
        lengths: The row lengths, in order.

    Returns:
        The table's lines.
    """
    header = "  " + "setting".ljust(NAME_COLUMN)
    header += "".join(f"{length:>{CELL_COLUMN},}" for length in lengths)
    lines = [header]
    for setting in SETTINGS:
        mine = {o.length: o for o in outcomes if o.setting == setting.name}
        if not mine:
            continue
        cells = "".join(
            f"{cell(mine[length]) if length in mine else '':>{CELL_COLUMN}}"
            for length in lengths
        )
        lines.append("  " + setting.name.ljust(NAME_COLUMN) + cells)
    return lines


def project_box() -> list[str]:
    """The project's own memory figures, kept apart from this run's.

    Returns:
        The lines.
    """
    chunked = ", ".join(
        f"{gb} GB at {length:,}" for length, gb in PROJECT_CHUNKED_GB.items()
    )
    return [
        "the project's own measurements (its source, not this run)",
        f"  fused attention, 6,144 tokens: {PROJECT_FUSED_6K_GB} GB "
        "(mhrl/training/common.py, chunked_attention's docstring)",
        f"  chunked attention: {chunked} (mhrl/cli.py, the comment on "
        "MAX_TRAIN_TOKENS = 6144)",
        f"  its SFT run on the 120 demonstrations: peak {PROJECT_SFT_PEAK_GB} GB "
        "(runs/main/sft/metrics.jsonl)",
        "  * in the table: the project's figure, shown where this probe refuses",
    ]


def load_model(folder: Path) -> nn.Module:
    """Load the 4-bit model with mlx-lm and add chapter 13's adapter.

    Args:
        folder: The build's folder (read only; nothing is downloaded).

    Returns:
        The model, every weight in memory, the adapter fresh (a no-op).
    """
    model, _ = load(str(folder))
    mx.eval(model.parameters())
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    mx.eval(model.trainable_parameters())
    return model


def length_list(value: str) -> list[int]:
    """Read ``--lengths``: comma-separated row lengths.

    Args:
        value: Such as ``1024,2048``.

    Returns:
        The lengths, shortest first, without repeats.

    Raises:
        argparse.ArgumentTypeError: For anything else, or a length under 2.
    """
    try:
        lengths = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a list of numbers: {value!r}") from error
    if not lengths or lengths[0] < 2:
        raise argparse.ArgumentTypeError("every length must be at least 2 tokens")
    return lengths


def setting_list(value: str) -> list[Setting]:
    """Read ``--settings``: comma-separated setting names.

    Args:
        value: Such as ``fused,chunked``.

    Returns:
        The settings, in the probe's order.

    Raises:
        argparse.ArgumentTypeError: For an unknown name.
    """
    names = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(names - set(BY_NAME))
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(BY_NAME)}")
    return [setting for setting in SETTINGS if setting.name in names]


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lengths", type=length_list, default=list(LENGTHS), help="row lengths"
    )
    parser.add_argument(
        "--settings", type=setting_list, default=list(SETTINGS), help="which settings"
    )
    parser.add_argument("--budget-gb", type=float, default=BUDGET_GB)
    parser.add_argument(
        "--keep-free",
        type=float,
        default=KEEP_FREE_PERCENT,
        help="percent of the machine's memory a pass must leave free",
    )
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    return parser.parse_args()


def main() -> None:
    """Measure each setting at each length, refusing what will not fit.

    Raises:
        SystemExit: If this chapter's engine is running, or the budget is
            not above 0; and with status 1 after the report when no row
            could be measured.
    """
    args = parse_args()
    if args.budget_gb <= 0:
        raise SystemExit(f"{SCRIPT}: --budget-gb must be above 0")
    if not 0 <= args.keep_free < PERCENT:
        raise SystemExit(f"{SCRIPT}: --keep-free is a percent, from 0 to under 100")
    home = chapter_home()
    pid = engine_running(home)
    if pid is not None:
        raise SystemExit(
            f"{SCRIPT}: the engine this chapter started (pid {pid}) holds the "
            "model's memory too: ./start.sh stop first"
        )
    folder, which = pick_model(args.fast)
    budget = args.budget_gb * GIGABYTE
    reserve = args.keep_free / PERCENT * machine_bytes()
    print(f"{SCRIPT} · one forward + backward pass per row, under a budget")
    print(f"  model     {short_path(folder)} ({which}), chapter 13's adapter")
    print(
        f"  limits    mx.set_memory_limit({MEMORY_LIMIT_BYTES / GIB:.0f} GiB = "
        f"{MEMORY_LIMIT_BYTES / GIGABYTE:.2f} GB) · mx.set_cache_limit("
        f"{CACHE_LIMIT_BYTES // MIB} MiB): guidelines to MLX, not walls"
    )
    print(
        f"  budget    {args.budget_gb:.1f} GB: a row projected over it is refused; "
        f"fused attention runs only up to {FUSED_MAX_TOKENS:,} tokens"
    )
    print(
        f"  machine   {machine_bytes() / GIGABYTE:.1f} GB; a pass must leave "
        f"{args.keep_free:g}% ({reserve / GIGABYTE:.1f} GB) of it free for "
        "everything else"
    )
    print(
        f"  rows      random token ids, the last 1/{TRAINED_SHARE} of each "
        "trained (one forward + backward pass, the SFT loss)"
    )
    set_memory_limits()
    model = load_model(folder)
    baseline = float(mx.get_active_memory())
    free = free_bytes()
    free_text = "unknown" if free is None else f"{free / GIGABYTE:.1f} GB"
    print(
        f"  weights   {baseline / GIGABYTE:.2f} GB in MLX after loading · "
        f"machine free now {free_text}"
    )
    probe = Probe(short_path(folder), args.budget_gb, round(baseline / GIGABYTE, 2))
    out = home / PROBE_FOLDER / RESULTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)

    def keep(outcome: Outcome) -> None:
        """Add a cell to the results and write them at once.

        Args:
            outcome: The cell.
        """
        probe.outcomes.append(outcome)
        out.write_text(json.dumps(asdict(probe), indent=2) + "\n", encoding="utf-8")

    for setting in args.settings:
        print(f"{setting.name}: attention {setting.attention}; logits {setting.head}")
        run_setting(
            model,
            setting,
            args.lengths,
            budget,
            baseline,
            print,
            on_outcome=keep,
            reserve=reserve,
        )
    print("")
    print("peak memory per pass (GB, MLX's own count, the weights included)")
    for line in table(probe.outcomes, args.lengths):
        print(line)
    print("")
    for line in project_box():
        print(line)
    print(f"results   {short_path(out)}")
    if not any(o.peak_gb is not None for o in probe.outcomes):
        print("nothing was measured: every row was refused or failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
