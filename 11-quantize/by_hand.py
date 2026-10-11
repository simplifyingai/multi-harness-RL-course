# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "numpy==2.5.3",
# ]
# ///
"""Quantize one real matrix of LFM2.5-2.6B by hand, from 16 bits to 4.

The matrix is ``model.layers.10.feed_forward.w1.weight``: 10,752 rows of
2,048 numbers, each a bfloat16. Each row is cut into groups of 64 numbers,
and each group is stored as 64 small whole numbers (*codes*, 0 to 15 at 4
bits) plus two bfloat16 numbers, a *scale* and an *offset*, so that every
number comes back as roughly ``scale x code + offset``.

The script does it twice. First the naive rule: scale = (max - min) / 15,
offset = min, code = round((w - min) / scale). Then MLX's real rule, the one
``mx.quantize`` and ``mlx_lm.convert`` use, written out by hand from MLX's own
GPU code. The naive codes differ from MLX's; the real rule's codes, scales and
offsets match MLX's exactly, and the script stops with an error if they ever
do not. It measures the error at 3, 4, 6 and 8 bits.

Run it with ``uv run by_hand.py`` after chapter 10's ``uv run download.py``.
``--fast`` also works without that download: it fetches only this matrix from
the Hub (44 MB). Needs an Apple-silicon Mac (MLX).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import report
from lfm_files import BITS_PER_BYTE
from matrix_source import Matrix, MatrixError, load_matrix

SCRIPT = "by_hand.py"
TITLE = "quantize one matrix by hand: 16 bits to 4"
GROUP_SIZE = 64  # numbers that share one scale and one offset
BITS = 4  # the bits the chapter's model uses for most matrices
SWEEP_BITS = (3, 4, 6, 8)
SCALE_BITS = 16  # the scale is stored as a bfloat16
OFFSET_BITS = 16  # ... and so is the offset
# MLX never lets a scale reach 0: a group of equal numbers would divide by it.
SCALE_FLOOR = np.float32(1e-7)
SHOWN_ROW = 0  # the group shown number by number: row 0, the first 64 numbers
SHOWN_GROUP = 0
BF16_SHIFT = 16  # a bfloat16 is the top 16 bits of a float32
BF16_ROUNDING = 0x7FFF  # round to nearest, ties to even, when dropping 16 bits


@dataclass(frozen=True)
class Quantized:
    """A matrix stored as codes, with one scale and one offset per group.

    Attributes:
        bits: Bits per code.
        codes: The codes, one row per group: whole numbers 0 to 2**bits - 1.
        scales: Each group's scale, as computed in float32.
        offsets: Each group's offset, as computed in float32.
    """

    bits: int
    codes: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray

    def rebuilt(self) -> np.ndarray:
        """Turn the codes back into numbers: scale x code + offset.

        The file keeps the scale and the offset as bfloat16, so they are
        rounded to bfloat16 first, and so is the result, as MLX does.

        Returns:
            The rebuilt numbers, one row per group (float32 holding bfloat16
            values).
        """
        scales = round_to_bfloat16(self.scales)[:, None]
        offsets = round_to_bfloat16(self.offsets)[:, None]
        return round_to_bfloat16(self.codes.astype(np.float32) * scales + offsets)


@dataclass(frozen=True)
class ErrorSize:
    """How far rebuilt numbers are from the originals.

    Attributes:
        mean: The average size of the error, mean |w - rebuilt|.
        largest: The largest single error.
        relative: The error's typical size over the numbers' typical size,
            both measured as a root mean square (sqrt of the mean of squares).
    """

    mean: float
    largest: float
    relative: float


@dataclass(frozen=True)
class GroupWalk:
    """The one group shown number by number, under both rules.

    Attributes:
        row: The matrix row it comes from.
        first: The column of its first number.
        values: Its 64 original numbers.
        naive: The naive rule's result for it.
        mlx: MLX's rule, by hand, for it.
        naive_error: The naive rule's error over the group.
        raw_step: (max - min) / levels, before MLX nudges it (float32).
        edge_steps: The anchor divided by the signed raw step, before rounding.
        whole_steps: edge_steps rounded (halves away from 0), as MLX rounds it.
        far_end: The end of the group that is not the anchor.
        far_code: How many nudged steps the far end is from the anchor.
        naive_differ: Codes where the naive rule differs from mx.quantize.
        mlx_same: Codes where MLX's rule by hand equals mx.quantize.
    """

    row: int
    first: int
    values: np.ndarray
    naive: Quantized
    mlx: Quantized
    naive_error: ErrorSize
    raw_step: float
    edge_steps: float
    whole_steps: int
    far_end: float
    far_code: float
    naive_differ: int
    mlx_same: int


@dataclass(frozen=True)
class BitsResult:
    """The whole matrix at one bit width: both rules against mx.quantize.

    Attributes:
        bits: Bits per code.
        mlx_error: The error of MLX's rule.
        naive_error: The error of the naive rule.
        codes_differ: Codes where MLX's rule by hand differs from mx.quantize.
        scales_differ: Stored scales that differ.
        offsets_differ: Stored offsets that differ.
        rebuilt_differ: Rebuilt numbers that differ from mx.dequantize's.
        naive_codes_differ: Codes where the naive rule differs from mx.quantize.
        naive_groups_differ: Groups with at least one such code.
        anchored_at_max: Groups MLX anchors at their largest number.
        one_signed: Groups whose numbers are all negative or all positive.
        stored_bytes: The bytes mx.quantize's codes, scales and offsets take.
        expected_bytes: The same, worked out: codes x bits + groups x 32 bits.
        group_count: How many groups of 64 the matrix holds.
        number_count: How many numbers it holds.
        scale_bits: Bits each group's scale takes.
        offset_bits: Bits each group's offset takes.
    """

    bits: int
    mlx_error: ErrorSize
    naive_error: ErrorSize
    codes_differ: int
    scales_differ: int
    offsets_differ: int
    rebuilt_differ: int
    naive_codes_differ: int
    naive_groups_differ: int
    anchored_at_max: int
    one_signed: int
    stored_bytes: int
    expected_bytes: int
    group_count: int
    number_count: int
    scale_bits: int = SCALE_BITS
    offset_bits: int = OFFSET_BITS

    def matches_mlx(self) -> bool:
        """Say whether MLX's rule by hand gave exactly mx.quantize's result.

        Returns:
            True when codes, scales, offsets and rebuilt numbers all match.
        """
        return not (
            self.codes_differ
            or self.scales_differ
            or self.offsets_differ
            or self.rebuilt_differ
        )


def levels(bits: int) -> int:
    """Count the steps between the lowest and highest code.

    Args:
        bits: Bits per code.

    Returns:
        2**bits - 1 (15 at 4 bits: codes 0 to 15).
    """
    return (1 << bits) - 1


def bits_per_number(bits: int) -> float:
    """Work out the storage per number, the scale and offset included.

    Args:
        bits: Bits per code.

    Returns:
        bits + (16 + 16) / 64: 4.5 at 4 bits.
    """
    return bits + (SCALE_BITS + OFFSET_BITS) / GROUP_SIZE


def bfloat16_to_float32(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Read bfloat16 bytes as float32 numbers, exactly.

    A bfloat16 is the top half of a float32 (chapter 10), so putting 16 zero
    bits under each one gives the same number as a float32.

    Args:
        data: The bytes, little-endian, row after row.
        shape: (rows, columns).

    Returns:
        The numbers.
    """
    halves = np.frombuffer(data, dtype="<u2").reshape(shape)
    return (halves.astype(np.uint32) << BF16_SHIFT).view(np.float32)


def round_to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float32 numbers to the nearest bfloat16 (ties to even).

    Args:
        values: float32 numbers.

    Returns:
        The rounded numbers, still as float32.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    tie_to_even = (bits >> BF16_SHIFT) & 1
    rounded = (bits + BF16_ROUNDING + tie_to_even) >> BF16_SHIFT << BF16_SHIFT
    return rounded.astype(np.uint32).view(np.float32)


def round_half_away(values: np.ndarray) -> np.ndarray:
    """Round to the nearest whole number, halves away from zero (2.5 -> 3).

    MLX's GPU code rounds this way; numpy's ``round`` sends halves to the
    even neighbour (2.5 -> 2) and would disagree on exact halves. The whole
    part and the fraction are split first, both exactly, so a number just
    below a half (0.49999997) is never pushed up by adding 0.5.

    Args:
        values: Numbers.

    Returns:
        The rounded numbers.
    """
    whole = np.trunc(values)
    up = np.abs(values - whole) >= np.float32(0.5)
    return (whole + np.where(up, np.sign(values), 0)).astype(np.float32)


def naive_quantize(groups: np.ndarray, bits: int) -> Quantized:
    """The naive rule: offset = min, scale = (max - min) / levels.

    Args:
        groups: The numbers, one row per group (float32).
        bits: Bits per code.

    Returns:
        Codes, scales and offsets.
    """
    lowest = groups.min(axis=1)
    highest = groups.max(axis=1)
    scales = np.maximum((highest - lowest) / np.float32(levels(bits)), SCALE_FLOOR)
    codes = round_half_away((groups - lowest[:, None]) / scales[:, None])
    codes = np.clip(codes, 0, levels(bits)).astype(np.uint8)
    return Quantized(bits, codes, scales.astype(np.float32), lowest)


def mlx_quantize(groups: np.ndarray, bits: int) -> Quantized:
    """MLX's real rule, step for step as its GPU code computes it, in float32.

    1. The group's range runs from its min to its max, but MLX starts its
       running max at 0, so the max is never below 0.
    2. The *anchor* is the end farther from 0. The step is (max - min) /
       levels, made negative when the anchor is the max, so codes count down
       from the max and up from the min.
    3. The step is nudged so the anchor is a whole number of steps from 0:
       then 0 itself is one of the levels and is stored exactly.
    4. offset = the anchor; code = round((w - offset) / step), halves away
       from 0, capped at the top code.

    Args:
        groups: The numbers, one row per group (float32).
        bits: Bits per code.

    Returns:
        Codes, scales and offsets, exactly as mx.quantize makes them.
    """
    top = np.float32(levels(bits))
    lowest = groups.min(axis=1)
    highest = np.maximum(groups.max(axis=1), np.float32(0))  # step 1
    step = np.maximum((highest - lowest) / top, SCALE_FLOOR)
    anchor_is_min = np.abs(lowest) > np.abs(highest)  # step 2
    step = np.where(anchor_is_min, step, -step).astype(np.float32)
    anchor = np.where(anchor_is_min, lowest, highest).astype(np.float32)
    zero_code = round_half_away(anchor / step)  # step 3: 0 is -zero_code steps away
    at_zero = zero_code == 0
    safe = np.where(at_zero, np.float32(1), zero_code)
    step = np.where(at_zero, step, anchor / safe).astype(np.float32)
    offset = np.where(at_zero, np.float32(0), anchor).astype(np.float32)
    codes = round_half_away((groups - offset[:, None]) / step[:, None])  # step 4
    codes = np.minimum(codes, top).astype(np.uint8)
    return Quantized(bits, codes, step, offset)


def unpack_codes(packed: np.ndarray, bits: int, columns: int) -> np.ndarray:
    """Unpack MLX's packed 32-bit words into one code per number.

    MLX writes the codes one after another into a long row of bits, the
    first code in the lowest bits, and stores that row as 32-bit words (at
    3 and 6 bits a code can straddle two bytes).

    Args:
        packed: The words, one row per matrix row (uint32).
        bits: Bits per code.
        columns: How many numbers each row holds.

    Returns:
        The codes, one row per matrix row (uint8).
    """
    rows = packed.shape[0]
    words = np.ascontiguousarray(packed, dtype="<u4")
    as_bytes = words.view(np.uint8).reshape(rows, -1)
    stream = np.unpackbits(as_bytes, axis=1, bitorder="little")[:, : columns * bits]
    place_values = (1 << np.arange(bits)).astype(np.uint16)
    return (
        (stream.reshape(rows, columns, bits) * place_values)
        .sum(axis=2)
        .astype(np.uint8)
    )


def error_size(original: np.ndarray, rebuilt: np.ndarray) -> ErrorSize:
    """Measure how far rebuilt numbers are from the originals.

    Args:
        original: The original numbers.
        rebuilt: The same numbers after quantizing and rebuilding.

    Returns:
        The mean, largest and relative error.
    """
    error = (rebuilt - original).astype(np.float64)
    typical = np.sqrt(np.mean(np.square(original.astype(np.float64))))
    return ErrorSize(
        mean=float(np.mean(np.abs(error))),
        largest=float(np.max(np.abs(error))),
        relative=float(np.sqrt(np.mean(np.square(error))) / typical),
    )


def mx_quantize(matrix: np.ndarray, bits: int) -> tuple[Quantized, np.ndarray, int]:
    """Quantize the matrix with MLX itself, and unpack what it returns.

    Args:
        matrix: The numbers (float32 holding bfloat16 values).
        bits: Bits per code.

    Returns:
        MLX's codes, scales and offsets (scales and offsets as stored in
        bfloat16), mx.dequantize's rebuilt numbers, and the bytes MLX's
        arrays take.
    """
    weights = mx.array(matrix).astype(mx.bfloat16)  # exact: they are bfloat16s
    packed, scales, biases = mx.quantize(weights, group_size=GROUP_SIZE, bits=bits)
    rebuilt = mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=bits)
    mx.eval(packed, scales, biases, rebuilt)
    stored = packed.nbytes + scales.nbytes + biases.nbytes
    codes = unpack_codes(np.array(packed), bits, matrix.shape[1])
    result = Quantized(
        bits,
        codes.reshape(-1, GROUP_SIZE),
        np.array(scales.astype(mx.float32)).reshape(-1),
        np.array(biases.astype(mx.float32)).reshape(-1),
    )
    return result, np.array(rebuilt.astype(mx.float32)).reshape(-1, GROUP_SIZE), stored


def walk_group(matrix: np.ndarray) -> GroupWalk:
    """Quantize the one group shown, by both rules and by MLX itself.

    Every group is quantized on its own, so MLX's codes for this group alone
    are its codes inside the whole matrix.

    Args:
        matrix: The whole matrix.

    Returns:
        The group under both rules, with the numbers the printout explains.
    """
    start = SHOWN_GROUP * GROUP_SIZE
    values = matrix[SHOWN_ROW, start : start + GROUP_SIZE][None, :]
    naive, mlx = naive_quantize(values, BITS), mlx_quantize(values, BITS)
    theirs, _, _ = mx_quantize(values, BITS)
    # The same float32 arithmetic as mlx_quantize, kept to show its steps.
    lowest = values.min()
    highest = np.maximum(values.max(), np.float32(0))
    raw_step = np.float32((highest - lowest) / np.float32(levels(BITS)))
    anchor_is_min = bool(np.abs(lowest) > np.abs(highest))
    signed = raw_step if anchor_is_min else -raw_step
    edge_steps = np.float32((lowest if anchor_is_min else highest) / signed)
    whole_steps = int(round_half_away(np.array([edge_steps], np.float32))[0])
    far_end = float(values.max() if anchor_is_min else values.min())
    step, offset = mlx.scales[0], mlx.offsets[0]
    return GroupWalk(
        row=SHOWN_ROW,
        first=start,
        values=values[0],
        naive=naive,
        mlx=mlx,
        naive_error=error_size(values, naive.rebuilt()),
        raw_step=float(raw_step),
        edge_steps=float(edge_steps),
        whole_steps=whole_steps,
        far_end=far_end,
        far_code=float((np.float32(far_end) - offset) / step),
        naive_differ=count_differences(naive.codes, theirs.codes),
        mlx_same=int(np.count_nonzero(mlx.codes == theirs.codes)),
    )


def count_differences(left: np.ndarray, right: np.ndarray) -> int:
    """Count the places where two arrays of the same shape differ.

    Args:
        left: One array.
        right: The other.

    Returns:
        How many entries differ.
    """
    return int(np.count_nonzero(left != right))


def measure_bits(groups: np.ndarray, matrix: np.ndarray, bits: int) -> BitsResult:
    """Quantize the whole matrix at one bit width, by hand and with MLX.

    Args:
        groups: The matrix cut into groups of 64 (one row per group).
        matrix: The same numbers as rows of the matrix.
        bits: Bits per code.

    Returns:
        Both rules' errors and every comparison with mx.quantize.
    """
    ours = mlx_quantize(groups, bits)
    naive = naive_quantize(groups, bits)
    theirs, mx_rebuilt, stored = mx_quantize(matrix, bits)
    rebuilt = ours.rebuilt()
    group_count, number_count = groups.shape[0], groups.size
    side_bits = group_count * (SCALE_BITS + OFFSET_BITS)
    one_signed = (groups.max(axis=1) < 0) | (groups.min(axis=1) > 0)
    return BitsResult(
        bits=bits,
        mlx_error=error_size(groups, rebuilt),
        naive_error=error_size(groups, naive.rebuilt()),
        codes_differ=count_differences(ours.codes, theirs.codes),
        scales_differ=count_differences(round_to_bfloat16(ours.scales), theirs.scales),
        offsets_differ=count_differences(
            round_to_bfloat16(ours.offsets), theirs.offsets
        ),
        rebuilt_differ=count_differences(rebuilt, mx_rebuilt),
        naive_codes_differ=count_differences(naive.codes, theirs.codes),
        naive_groups_differ=int(np.count_nonzero((naive.codes != theirs.codes).any(1))),
        anchored_at_max=int(np.count_nonzero(ours.scales < 0)),
        one_signed=int(np.count_nonzero(one_signed)),
        stored_bytes=stored,
        expected_bytes=(number_count * bits + side_bits) // BITS_PER_BYTE,
        group_count=group_count,
        number_count=number_count,
    )


def mismatch_message(results: list[BitsResult]) -> str | None:
    """Explain any difference between MLX's rule by hand and mx.quantize.

    Args:
        results: One result per bit width.

    Returns:
        The explanation, or None when every bit width matched exactly.
    """
    lines = [
        f"{result.bits} bits: {result.codes_differ:,} codes, {result.scales_differ:,} "
        f"scales, {result.offsets_differ:,} offsets and {result.rebuilt_differ:,} "
        "rebuilt numbers differ from MLX's"
        for result in results
        if not result.matches_mlx()
    ]
    if not lines:
        return None
    return "\n".join(
        ["MLX's rule, by hand, no longer matches mx.quantize:", *lines]
        + ["(a different MLX version may have changed its rule; compare the errors)"]
    )


def main() -> None:
    """Quantize the matrix by both rules, check against MLX, print it all."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="without chapter 10's download, fetch only this matrix (44 MB)",
    )
    args = parser.parse_args()
    try:
        matrix_bytes: Matrix = load_matrix(args.fast)
    except MatrixError as error:
        raise SystemExit(str(error)) from error
    matrix = bfloat16_to_float32(matrix_bytes.data, matrix_bytes.pin.shape)
    groups = matrix.reshape(-1, GROUP_SIZE)

    report.print_title(SCRIPT, TITLE)
    report.print_matrix(matrix_bytes, matrix)
    results = [measure_bits(groups, matrix, bits) for bits in SWEEP_BITS]
    four = results[SWEEP_BITS.index(BITS)]
    walk = walk_group(matrix)
    report.print_naive_group(walk, BITS, levels(BITS))
    report.print_mlx_group(walk, levels(BITS))
    report.print_whole_matrix(four, GROUP_SIZE, len(matrix_bytes.data))
    report.print_sweep(results, GROUP_SIZE, levels, bits_per_number)
    problem = mismatch_message(results)
    if problem:
        raise SystemExit(problem)
    report.print_verdict(results)


if __name__ == "__main__":
    main()
