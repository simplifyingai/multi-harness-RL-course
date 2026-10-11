# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Build our own 4-bit LFM2.5-2.6B from chapter 10's bfloat16 download.

``mlx_lm.convert`` reads the original, quantizes it with MLX's rule (the one
by_hand.py wrote out), and writes a new folder: the packed weights, their
scales and offsets, config.json with a ``quantization`` block, and the
tokenizer. Every matrix is stored at 4 bits in groups of 64, except the
embedding table, which a ``quant_predicate`` keeps at 6 bits: that table is
also the output head (chapter 10), so every score the model gives passes
through it. mlx-lm's built-in mixed recipe cannot be used for this model, and
the script shows why before it builds.

The result goes to ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``, the model
every later chapter loads. ``--all-4bit`` builds the comparison variant with
the embedding table at 4 bits too, into ``$COURSE_HOME/11-quantize``. A build
is written under a temporary name and renamed only when it is complete, so a
half build never passes for a whole one. Run it with ``uv run convert.py``
after chapter 10's ``uv run download.py`` (it needs an Apple-silicon Mac).
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import report
from lfm_files import (
    BF16,
    BITS_PER_BYTE,
    CONFIG_FILE,
    FOUR_BIT,
    QUANTIZATION_KEY,
    SAFETENSORS_GLOB,
    WEIGHTS_FILE,
    all_four_bit_folder,
    build_ready,
    missing_files,
    model_folder,
    ours_folder,
    require_repo,
)
from mlx_lm import convert
from mlx_lm.convert import mixed_quant_predicate_builder
from mlx_lm.utils import load_model
from tensors import read_header

SCRIPT = "convert.py"
TITLE = "our own 4-bit LFM2.5-2.6B, built with mlx-lm"
GROUP_SIZE = 64
BITS = 4
EMBEDDING_BITS = 6
MODE = "affine"  # scale x code + offset, as by_hand.py showed
EMBEDDING = "model.embed_tokens"  # the module the predicate singles out
BUILTIN_RECIPE = "mixed_4_6"  # mlx-lm's own 4-bit-with-some-6-bit recipe
RECIPE_KEY = "down_proj"  # the layer name that recipe looks for
HEAD_KEY = "lm_head"  # the separate output head it raises to 6 bits
EXAMPLE_LAYER = "model.layers.0.feed_forward.w1"
PARTIAL_SUFFIX = ".partial"
WEIGHT_SUFFIX = ".weight"
SCALES_SUFFIX = ".scales"
BIASES_SUFFIX = ".biases"
BITS_PER_WORD = 32  # a packed word is a U32
KIND_UNPACKED = "kept in bfloat16"


@dataclass(frozen=True)
class BuildPlan:
    """One build: what it is called, where it goes, how it quantizes.

    Attributes:
        name: A few words for the reader.
        folder: Where the finished build lives.
        keep_embedding: True to keep the embedding table at 6 bits.
    """

    name: str
    folder: Path
    keep_embedding: bool


@dataclass(frozen=True)
class RecipeCheck:
    """Why mlx-lm's built-in mixed recipe does not fit LFM2.

    Attributes:
        error: What the recipe raised, or None if it would have worked.
        quantizable: How many layers can be quantized (linear and embedding).
        named_like_recipe: How many of them carry the name the recipe expects.
        has_head: Whether the model has a separate output head.
    """

    error: str | None
    quantizable: int
    named_like_recipe: int
    has_head: bool


@dataclass(frozen=True)
class StorageRow:
    """One kind of tensor in a build's weights file, counted from its header.

    Attributes:
        kind: Such as ``"4-bit layers"`` or ``"kept in bfloat16"``.
        modules: How many layers (or plain tensors) are of this kind.
        numbers: How many of the model's numbers they hold.
        nbytes: The bytes they take, scales and offsets included.
    """

    kind: str
    modules: int
    numbers: int
    nbytes: int

    def bits_each(self) -> float:
        """Work out the storage per number, scales and offsets included.

        Returns:
            Bits per number, such as 4.5.
        """
        return BITS_PER_BYTE * self.nbytes / self.numbers


@dataclass(frozen=True)
class Storage:
    """A build's weights file, counted from its header.

    Attributes:
        rows: One row per kind of tensor.
        header_bytes: The header's length, the 8 length bytes included.
        file_bytes: The file's size.
    """

    rows: list[StorageRow]
    header_bytes: int
    file_bytes: int

    def data_bytes(self) -> int:
        """Add up the bytes of every row.

        Returns:
            The bytes of numbers, scales and offsets.
        """
        return sum(row.nbytes for row in self.rows)


def keep_embedding_at_6(path: str, module: nn.Module) -> bool | dict[str, Any]:
    """Decide how to quantize one layer: the embedding at 6 bits, the rest at 4.

    mlx-lm calls this once per layer that can be quantized, with two
    arguments: the layer's name and the layer itself.

    Args:
        path: The layer's name, such as ``model.embed_tokens``.
        module: The layer (not needed here: the name decides).

    Returns:
        The embedding table's own settings, or True for the defaults.
    """
    del module  # the name alone decides
    if path == EMBEDDING:
        return {"group_size": GROUP_SIZE, "bits": EMBEDDING_BITS, "mode": MODE}
    return True


def check_builtin_recipe(source: Path) -> RecipeCheck:
    """Try mlx-lm's built-in ``mixed_4_6`` recipe on the model's layers.

    The model is built lazily: its layers exist, but no weight is read.

    Args:
        source: The bfloat16 model's folder.

    Returns:
        What happened, and the layer names that explain it.
    """
    model, _ = load_model(source, lazy=True)
    layers = [(name, module) for name, module in model.named_modules() if name]
    names = [name for name, _ in layers]
    quantizable = [name for name, module in layers if hasattr(module, "to_quantized")]
    error = None
    try:
        mixed_quant_predicate_builder(BUILTIN_RECIPE, model, GROUP_SIZE)
    except (ValueError, KeyError, AttributeError) as raised:
        # mlx-lm 0.32.0 raises ValueError; the others guard a later version.
        error = str(raised) if isinstance(raised, ValueError) else repr(raised)
    check = RecipeCheck(
        error=error,
        quantizable=len(quantizable),
        named_like_recipe=sum(RECIPE_KEY in name for name in names),
        has_head=any(name.endswith(HEAD_KEY) for name in names),
    )
    del model
    gc.collect()
    return check


def build(plan: BuildPlan, source: Path) -> tuple[float, int]:
    """Quantize the bfloat16 model with mlx_lm.convert into the plan's folder.

    The build is written to ``<folder>.partial`` and renamed when complete.

    Args:
        plan: The build.
        source: The bfloat16 model's folder.

    Returns:
        How long it took in seconds, and MLX's peak memory in bytes.
    """
    partial = plan.folder.with_name(plan.folder.name + PARTIAL_SUFFIX)
    shutil.rmtree(partial, ignore_errors=True)  # a build that broke off earlier
    plan.folder.parent.mkdir(parents=True, exist_ok=True)
    mx.reset_peak_memory()
    start = time.perf_counter()
    convert(
        str(source),
        str(partial),
        quantize=True,
        q_group_size=GROUP_SIZE,
        q_bits=BITS,
        q_mode=MODE,
        quant_predicate=keep_embedding_at_6 if plan.keep_embedding else None,
    )
    seconds = time.perf_counter() - start
    # Only now, with the new build complete, is an old one (--force) removed.
    shutil.rmtree(plan.folder, ignore_errors=True)
    partial.replace(plan.folder)
    gc.collect()
    mx.clear_cache()
    return seconds, mx.get_peak_memory()


def quantization_block(folder: Path) -> dict[str, Any] | None:
    """Read config.json's quantization block.

    Args:
        folder: A model's folder.

    Returns:
        The block, or None if the model is not quantized.
    """
    config = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    return config.get(QUANTIZATION_KEY)


def kind_of(bits: int) -> str:
    """Name a kind of packed tensor by its bits.

    Args:
        bits: Bits per code.

    Returns:
        Such as ``"4-bit layers"``.
    """
    return f"{bits}-bit layers"


def count_storage(folder: Path) -> Storage:
    """Count a build's numbers and bytes by kind, from its header alone.

    A packed layer has three tensors: the packed codes (``.weight``, 32-bit
    words), one scale per group (``.scales``) and one offset per group
    (``.biases``). Its real width is scales per row x group size, and its bits
    are packed width x 32 / real width (chapter 10's rule, read backwards).

    Args:
        folder: The build's folder.

    Returns:
        The count, by kind.
    """
    path = folder / WEIGHTS_FILE
    header = read_header(path)
    tensors = {tensor.name: tensor for tensor in header.tensors}
    numbers: dict[str, int] = defaultdict(int)
    nbytes: dict[str, int] = defaultdict(int)
    modules: dict[str, int] = defaultdict(int)
    for name, tensor in tensors.items():
        if name.endswith((SCALES_SUFFIX, BIASES_SUFFIX)):
            continue  # counted with their weight
        scales = tensors.get(name.removesuffix(WEIGHT_SUFFIX) + SCALES_SUFFIX)
        if scales is None:
            kind, count, size = KIND_UNPACKED, tensor.stored(), tensor.nbytes
        else:
            # affine packing keeps an offset per group; other modes may not
            biases = tensors.get(name.removesuffix(WEIGHT_SUFFIX) + BIASES_SUFFIX)
            rows, columns = tensor.shape[0], scales.shape[-1] * GROUP_SIZE
            kind = kind_of(tensor.shape[-1] * BITS_PER_WORD // columns)
            count = rows * columns
            size = tensor.nbytes + scales.nbytes + (biases.nbytes if biases else 0)
        numbers[kind] += count
        nbytes[kind] += size
        modules[kind] += 1
    order = sorted(numbers, key=lambda kind: (kind == KIND_UNPACKED, kind))
    rows = [
        StorageRow(kind, modules[kind], numbers[kind], nbytes[kind]) for kind in order
    ]
    return Storage(rows, header.header_bytes, path.stat().st_size)


def numbers_bytes(folder: Path) -> int:
    """Add up the bytes of every tensor in a model's weights files.

    Args:
        folder: The model's folder.

    Returns:
        The bytes of numbers, headers left out.
    """
    return sum(
        read_header(path).data_bytes() for path in sorted(folder.glob(SAFETENSORS_GLOB))
    )


def plan_for(all_four_bit: bool) -> BuildPlan:
    """Pick the build the command line asks for.

    Args:
        all_four_bit: True for the comparison variant.

    Returns:
        The build.
    """
    if all_four_bit:
        name = "all 4-bit (the embedding table too)"
        return BuildPlan(name, all_four_bit_folder(), keep_embedding=False)
    name = "ours: 4-bit, the embedding table at 6 bits"
    return BuildPlan(name, ours_folder(), keep_embedding=True)


def main() -> None:
    """Show why the built-in recipe fails, build ours, and check the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-4bit",
        action="store_true",
        help="build the variant with the embedding table at 4 bits too",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the build is there"
    )
    args = parser.parse_args()
    source = require_repo(BF16)
    plan = plan_for(args.all_4bit)

    report.print_title(SCRIPT, TITLE)
    report.print_build_plan(plan.name, source, plan.folder)
    if build_ready(plan.folder) and not args.force:
        report.print_already_built()
    else:
        if plan.keep_embedding:
            report.print_recipe_check(check_builtin_recipe(source), BUILTIN_RECIPE)
            report.print_predicate(keep_embedding_at_6, EMBEDDING, EXAMPLE_LAYER)
        name = keep_embedding_at_6.__name__ if plan.keep_embedding else None
        report.print_convert_call(GROUP_SIZE, BITS, MODE, name)
        seconds, peak = build(plan, source)
        report.print_built(seconds, peak)
    official = model_folder(FOUR_BIT)
    theirs = None if missing_files(FOUR_BIT, official) else quantization_block(official)
    report.print_quantization(quantization_block(plan.folder), theirs)
    report.print_storage(count_storage(plan.folder), numbers_bytes(source))


if __name__ == "__main__":
    main()
