# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Count LFM2.5-2.6B's numbers from its files, without loading a single weight.

A safetensors file starts with 8 bytes that give the length of its header,
then the header itself: JSON naming every tensor (a block of numbers) with its
number type, its shape and where its bytes sit in the file. Reading only the
headers, a few dozen kilobytes of a 5.4 GB download, is enough to count every
number in the model and sort them by part: feed-forward blocks, convolution
layers, attention layers, the embedding table and the norms. config.json says
how the parts are built, and each count is checked against it.

Run it with ``uv run inspect_model.py`` after ``uv run download.py``.
``--fast`` counts the 4-bit export instead, where most matrices are packed
eight 4-bit numbers to a 32-bit word, so a matrix's real width is its stored
width x 32 / bits. (The script is not called inspect.py: that name would
shadow Python's own ``inspect`` module, which ``dataclasses`` imports.)
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import report
from lfm_files import (
    LAYER_ATTENTION,
    LAYER_CONV,
    PART_ATTENTION,
    PART_CONV,
    PART_EMBEDDING,
    PART_FEED_FORWARD,
    PART_HEAD,
    PART_NORMS,
    PART_ORDER,
    require_model,
)

SCRIPT = "inspect_model.py"
TITLE = "counting LFM2.5-2.6B's numbers from its files, loading none"

HEADER_LENGTH_BYTES = 8  # a little-endian unsigned 64-bit integer
HEADER_LENGTH_FORMAT = "<Q"
MAX_HEADER_BYTES = 100_000_000  # the safetensors format's own limit
METADATA_KEY = "__metadata__"
SAFETENSORS_GLOB = "*.safetensors"
CONFIG_FILE = "config.json"
CARD_FILE = "README.md"
BITS_PER_WORD = 32  # a U32 word packs 32 / bits quantized numbers
BYTES_PER_DTYPE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I64": 8}

# The tensor-name pieces that say which part of the model a tensor belongs to.
EMBEDDING_NAME = "model.embed_tokens.weight"
HEAD_NAME = "lm_head.weight"  # absent when the head is tied to the embedding
FEED_FORWARD_KEY = ".feed_forward."
CONV_KEY = ".conv."
ATTENTION_KEY = ".self_attn."
NORM_SUFFIX = "norm.weight"
WEIGHT_SUFFIX = ".weight"
SCALES_SUFFIX = ".scales"  # 4-bit exports: one scale per group of numbers
BIASES_SUFFIX = ".biases"  # ... and one offset per group (MLX calls them biases)
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# The tensors printed one by one: the embedding table, all of layer 0 (a
# convolution layer) and the attention part of layer 2 (an attention layer).
SAMPLE_TENSORS = (
    "model.embed_tokens.weight",
    "model.layers.0.operator_norm.weight",
    "model.layers.0.conv.in_proj.weight",
    "model.layers.0.conv.conv.weight",
    "model.layers.0.conv.out_proj.weight",
    "model.layers.0.ffn_norm.weight",
    "model.layers.0.feed_forward.w1.weight",
    "model.layers.0.feed_forward.w3.weight",
    "model.layers.0.feed_forward.w2.weight",
    "model.layers.2.self_attn.q_proj.weight",
    "model.layers.2.self_attn.k_proj.weight",
    "model.layers.2.self_attn.v_proj.weight",
    "model.layers.2.self_attn.q_layernorm.weight",
    "model.layers.2.self_attn.k_layernorm.weight",
    "model.layers.2.self_attn.out_proj.weight",
)

# How the model card states the parameter count: the original's
# "- **Total parameters**: 2.69B", the MLX export's "| Parameters | 2.6B |".
CARD_PATTERNS = (
    re.compile(r"Total parameters\**:\s*([\d.]+\s*B)\b"),
    re.compile(r"^\|\s*Parameters\s*\|\s*([\d.]+\s*B)\s*\|", re.MULTILINE),
)


class HeaderError(ValueError):
    """Raised when a file is not a readable safetensors file."""


@dataclass(frozen=True)
class Tensor:
    """One tensor as a safetensors header describes it.

    Attributes:
        name: Its name, such as ``model.layers.0.feed_forward.w1.weight``.
        dtype: Its number type as stored, such as ``BF16`` or ``U32``.
        shape: Its shape as stored.
        nbytes: How many bytes it takes in the file.
        file: The file it is in.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: str

    def stored(self) -> int:
        """Count the values as stored (a packed word counts as one).

        Returns:
            The product of the shape.
        """
        count = 1
        for size in self.shape:
            count *= size
        return count


@dataclass(frozen=True)
class FileHeader:
    """What one safetensors file's header says.

    Attributes:
        name: The file name.
        size: The file's size in bytes.
        header_bytes: The header's length, plus the 8 bytes that give it.
        tensors: Every tensor in the file.
    """

    name: str
    size: int
    header_bytes: int
    tensors: list[Tensor]


@dataclass(frozen=True)
class Quantization:
    """How a 4-bit export packs its matrices (config.json's quantization block).

    Attributes:
        bits: Bits per number for most matrices.
        group_size: How many numbers share one scale and one offset.
        overrides: Bits for named modules that differ, such as the embedding.
    """

    bits: int
    group_size: int
    overrides: dict[str, int]

    def bits_for(self, module: str) -> int:
        """Return the bits per number used for one module.

        Args:
            module: The module's name, such as ``model.embed_tokens``.

        Returns:
            Its bits.
        """
        return self.overrides.get(module, self.bits)


@dataclass(frozen=True)
class Counted:
    """One tensor, counted: how many parameters it really holds.

    Attributes:
        tensor: The tensor as stored.
        part: Which part of the model it belongs to.
        parameters: How many numbers of the model it holds (0 for a 4-bit
            export's scales and offsets, which only help store the others).
        shape: Its real shape, with packed columns unpacked.
        bits: Bits per number if it is packed, else None.
    """

    tensor: Tensor
    part: str
    parameters: int
    shape: tuple[int, ...]
    bits: int | None = None


@dataclass
class PartTally:
    """The running count for one part of the model.

    Attributes:
        part: The part's name.
        tensors: How many parameter tensors it has.
        parameters: How many parameters they hold.
        nbytes: How many bytes they take, scales and offsets included.
        layers: The layers it appears in.
    """

    part: str
    tensors: int = 0
    parameters: int = 0
    nbytes: int = 0
    layers: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class Expected:
    """One part's count worked out from config.json alone.

    Attributes:
        part: The part's name.
        formula: The arithmetic, written out.
        parameters: Its result.
    """

    part: str
    formula: str
    parameters: int


@dataclass(frozen=True)
class Inventory:
    """Everything this script learns from the files.

    Attributes:
        headers: Each safetensors file's header.
        counted: Every tensor, counted.
        parts: The tally per part, in the chapter's order.
        layer_types: config.json's layer types, one per layer.
        config: config.json.
        quantization: The packing, for a 4-bit export; None for bfloat16.
    """

    headers: list[FileHeader]
    counted: list[Counted]
    parts: list[PartTally]
    layer_types: list[str]
    config: dict[str, Any]
    quantization: Quantization | None

    def total_parameters(self) -> int:
        """Add up every part.

        Returns:
            The model's parameter count.
        """
        return sum(part.parameters for part in self.parts)

    def tensor_bytes(self) -> int:
        """Add up the bytes of every tensor (headers not included).

        Returns:
            The bytes.
        """
        return sum(item.tensor.nbytes for item in self.counted)

    def file_bytes(self) -> int:
        """Add up the sizes of the safetensors files.

        Returns:
            The bytes.
        """
        return sum(header.size for header in self.headers)

    def stored_values(self) -> int:
        """Count the values as stored, the way a naive count would.

        Returns:
            The sum of every tensor's shape product, packed words as one each.
        """
        return sum(item.tensor.stored() for item in self.counted)

    def layers_with(self, part: str) -> list[int]:
        """List the layers a part appears in, from the tensor names.

        Args:
            part: The part's name.

        Returns:
            Layer numbers, counting from 0.
        """
        for tally in self.parts:
            if tally.part == part:
                return sorted(tally.layers)
        return []


def read_header(path: Path) -> FileHeader:
    """Read a safetensors file's header, and nothing after it.

    Args:
        path: The file.

    Returns:
        Its header.

    Raises:
        HeaderError: If the file is too short, or its header length, its JSON
            or a tensor's byte count does not make sense.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(HEADER_LENGTH_BYTES)
        if len(prefix) < HEADER_LENGTH_BYTES:
            raise HeaderError(f"{path.name}: too short to be a safetensors file")
        (length,) = struct.unpack(HEADER_LENGTH_FORMAT, prefix)
        if length > min(MAX_HEADER_BYTES, size - HEADER_LENGTH_BYTES):
            raise HeaderError(
                f"{path.name}: says its header is {length:,} bytes, "
                f"in a file of {size:,}: not a safetensors file"
            )
        raw = handle.read(length)
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HeaderError(f"{path.name}: the header is not JSON ({error})") from error
    try:
        tensors = [
            tensor_from(name, entry, path.name)
            for name, entry in entries.items()
            if name != METADATA_KEY
        ]
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise HeaderError(
            f"{path.name}: a tensor entry is malformed ({error!r})"
        ) from error
    for tensor in tensors:
        width = BYTES_PER_DTYPE.get(tensor.dtype)
        if width is not None and tensor.nbytes != tensor.stored() * width:
            raise HeaderError(
                f"{path.name}: {tensor.name} is {tensor.dtype} {tensor.shape}, "
                f"which is {tensor.stored() * width:,} bytes, "
                f"but takes {tensor.nbytes:,}"
            )
    return FileHeader(path.name, size, HEADER_LENGTH_BYTES + length, tensors)


def tensor_from(name: str, entry: dict[str, Any], file: str) -> Tensor:
    """Turn one header entry into a Tensor.

    Args:
        name: The tensor's name.
        entry: Its entry: ``dtype``, ``shape`` and ``data_offsets``.
        file: The file it is in.

    Returns:
        The tensor.

    Raises:
        KeyError: If the entry lacks one of the three fields.
        TypeError: If a field has the wrong kind of value.
        ValueError: If ``data_offsets`` is not two numbers.
    """
    start, end = entry["data_offsets"]
    return Tensor(name, entry["dtype"], tuple(entry["shape"]), end - start, file)


def read_quantization(config: dict[str, Any]) -> Quantization | None:
    """Read config.json's quantization block, if the model has one.

    Args:
        config: config.json.

    Returns:
        The packing, or None for an unquantized model.
    """
    block = config.get("quantization")
    if not block:
        return None
    overrides = {
        module: settings["bits"]
        for module, settings in block.items()
        if isinstance(settings, dict)
    }
    return Quantization(block["bits"], block["group_size"], overrides)


def part_of(name: str) -> str:
    """Say which part of the model a tensor belongs to, from its name.

    Args:
        name: The tensor's name.

    Returns:
        One of the PART_ names.

    Raises:
        ValueError: For a tensor this chapter does not know.
    """
    module = strip_suffix(name)
    if module + WEIGHT_SUFFIX == EMBEDDING_NAME:
        return PART_EMBEDDING
    if module + WEIGHT_SUFFIX == HEAD_NAME:
        return PART_HEAD
    if FEED_FORWARD_KEY in name:
        return PART_FEED_FORWARD
    if ATTENTION_KEY in name:  # its q_layernorm and k_layernorm belong to it
        return PART_ATTENTION
    if CONV_KEY in name:
        return PART_CONV
    if name.endswith(NORM_SUFFIX):
        return PART_NORMS
    raise ValueError(f"a tensor this chapter does not know: {name}")


def strip_suffix(name: str) -> str:
    """Drop ``.weight``, ``.scales`` or ``.biases`` from a tensor name.

    Args:
        name: The tensor's name.

    Returns:
        The module's name, such as ``model.layers.0.feed_forward.w1``.
    """
    for suffix in (WEIGHT_SUFFIX, SCALES_SUFFIX, BIASES_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def layer_of(name: str) -> int | None:
    """Read the layer number out of a tensor name.

    Args:
        name: The tensor's name.

    Returns:
        The layer, counting from 0, or None outside the layers.
    """
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def unpacked_columns(packed: int, bits: int) -> int:
    """Turn a packed matrix's stored width into its real width.

    Each 32-bit word holds 32 / bits numbers, so the real width is the stored
    width x 32 / bits: 256 words of 4-bit numbers are 2,048 columns.

    Args:
        packed: Columns as stored, in 32-bit words.
        bits: Bits per number.

    Returns:
        The real number of columns.
    """
    return packed * BITS_PER_WORD // bits


def count_tensor(
    tensor: Tensor, names: set[str], quantization: Quantization | None
) -> Counted:
    """Count the parameters one tensor holds.

    A plain tensor holds as many parameters as its shape says. In a 4-bit
    export a packed weight (stored as U32 words, with ``.scales`` beside it)
    holds rows x (packed columns x 32 / bits), and its scales and offsets hold
    none: they only say how to turn the packed numbers back into real ones.

    Args:
        tensor: The tensor.
        names: Every tensor name in the model, to find a weight's scales.
        quantization: The packing, or None for an unquantized model.

    Returns:
        The counted tensor.

    Raises:
        ValueError: For a tensor name this chapter does not know.
    """
    part = part_of(tensor.name)
    if tensor.name.endswith((SCALES_SUFFIX, BIASES_SUFFIX)):
        return Counted(tensor, part, 0, tensor.shape)
    module = strip_suffix(tensor.name)
    if quantization is None or module + SCALES_SUFFIX not in names:
        return Counted(tensor, part, tensor.stored(), tensor.shape)
    bits = quantization.bits_for(module)
    *rows, packed = tensor.shape
    columns = unpacked_columns(packed, bits)
    shape = (*rows, columns)
    parameters = Tensor(tensor.name, tensor.dtype, shape, 0, tensor.file).stored()
    return Counted(tensor, part, parameters, shape, bits)


def check_groups(counted: list[Counted], quantization: Quantization | None) -> None:
    """Check each packed weight's width against its number of scale groups.

    A packed matrix of real width W has W / group_size scales per row. If the
    bits read from config.json were wrong, the two would disagree.

    Args:
        counted: Every tensor, counted.
        quantization: The packing, or None.

    Raises:
        HeaderError: If a packed weight and its scales disagree.
    """
    if quantization is None:
        return
    groups = {
        strip_suffix(item.tensor.name): item.tensor.shape[-1]
        for item in counted
        if item.tensor.name.endswith(SCALES_SUFFIX)
    }
    for item in counted:
        if item.bits is None:
            continue
        module = strip_suffix(item.tensor.name)
        expected = groups[module] * quantization.group_size
        if item.shape[-1] != expected:
            raise HeaderError(
                f"{item.tensor.name}: {item.shape[-1]:,} columns at {item.bits} bits, "
                f"but {groups[module]} groups of {quantization.group_size} "
                f"make {expected:,}"
            )


def tally(counted: Iterable[Counted]) -> list[PartTally]:
    """Add the counted tensors up by part.

    Args:
        counted: Every tensor, counted.

    Returns:
        One tally per part that has tensors, in the chapter's order.
    """
    tallies = {part: PartTally(part) for part in PART_ORDER}
    for item in counted:
        entry = tallies[item.part]
        entry.nbytes += item.tensor.nbytes
        if item.parameters == 0:
            continue  # a 4-bit export's scales and offsets
        entry.tensors += 1
        entry.parameters += item.parameters
        layer = layer_of(item.tensor.name)
        if layer is not None:
            entry.layers.add(layer)
    return [entry for entry in tallies.values() if entry.tensors]


def expected_parts(config: dict[str, Any]) -> list[Expected]:
    """Work out each part's parameter count from config.json alone.

    Args:
        config: config.json.

    Returns:
        One entry per part, with its arithmetic written out.
    """
    hidden = config["hidden_size"]
    width = config["intermediate_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    kernel = config["conv_L_cache"]
    types = config["layer_types"]
    layers = len(types)
    convs = types.count(LAYER_CONV)
    attentions = types.count(LAYER_ATTENTION)
    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    feed_forward = 3 * hidden * width
    conv = hidden * 3 * hidden + hidden * hidden + hidden * kernel
    attention = 2 * hidden * hidden + 2 * hidden * kv_width + 2 * head_dim
    norms = (2 * layers + 1) * hidden
    return [
        Expected(
            PART_FEED_FORWARD,
            f"{layers} layers x 3 matrices x {hidden:,} x {width:,}",
            layers * feed_forward,
        ),
        Expected(
            PART_CONV,
            f"{convs} layers x ({hidden:,} x {3 * hidden:,} + {hidden:,} x {hidden:,}"
            f" + {hidden:,} x {kernel})",
            convs * conv,
        ),
        Expected(
            PART_ATTENTION,
            f"{attentions} layers x (2 x {hidden:,} x {hidden:,} + 2 x {hidden:,} x "
            f"{kv_width} + 2 x {head_dim})",
            attentions * attention,
        ),
        Expected(PART_EMBEDDING, f"{vocab:,} x {hidden:,}", vocab * hidden),
        Expected(PART_NORMS, f"({layers} layers x 2 + 1) x {hidden:,}", norms),
    ]


def card_parameters(text: str) -> str | None:
    """Find the parameter count the model card states.

    Args:
        text: The model card (README.md).

    Returns:
        Such as ``"2.69B"``, or None if the card does not say.
    """
    for pattern in CARD_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).replace(" ", "")
    return None


def inventory(folder: Path) -> Inventory:
    """Read config.json and every safetensors header in a model folder.

    Args:
        folder: The model's folder.

    Returns:
        What the files say, counted and checked.

    Raises:
        HeaderError: If a header cannot be read or a packed width is wrong.
        ValueError: For a tensor name this chapter does not know (HeaderError
            is a kind of ValueError too).
        FileNotFoundError: If config.json or the safetensors files are missing.
    """
    config = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    paths = sorted(folder.glob(SAFETENSORS_GLOB))
    if not paths:
        raise FileNotFoundError(f"no safetensors files in {folder}")
    headers = [read_header(path) for path in paths]
    tensors = [tensor for header in headers for tensor in header.tensors]
    names = {tensor.name for tensor in tensors}
    quantization = read_quantization(config)
    counted = [count_tensor(tensor, names, quantization) for tensor in tensors]
    check_groups(counted, quantization)
    return Inventory(
        headers, counted, tally(counted), config["layer_types"], config, quantization
    )


def main() -> None:
    """Read the headers, count the parameters by part and print the tally."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="count the 4-bit export (1.6 GB) instead of bfloat16",
    )
    args = parser.parse_args()
    repo, folder = require_model(args.fast)
    try:
        found = inventory(folder)
    except ValueError as error:  # a HeaderError, or a tensor it does not know
        raise SystemExit(f"cannot count the model: {error}") from None
    card = card_parameters((folder / CARD_FILE).read_text(encoding="utf-8"))

    report.print_title(SCRIPT, TITLE)
    report.print_headers(folder, found)
    report.print_config(found.config)
    report.print_sample_tensors(found, SAMPLE_TENSORS)
    report.print_parts(found)
    report.print_expected(expected_parts(found.config), found)
    report.print_naming(repo, card, found.total_parameters())
    report.print_size(found)


if __name__ == "__main__":
    main()
