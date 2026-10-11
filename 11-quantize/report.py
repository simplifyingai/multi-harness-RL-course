"""How the chapter's scripts print: groups, codes, builds, tables, timings.

Nothing here measures anything. It only turns the scripts' results into lines
a person can read on screen (working out a percentage or a ratio at most), so
each script can stay about quantizing.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from lfm_files import (
    BITS_PER_BYTE,
    FINISH_STOP,
    KEY_ALL_FOUR,
    KEY_BF16,
    KEY_LIQUID,
    KEY_OURS,
    MILLISECONDS,
    gibibytes,
    gigabytes,
    short_path,
)

if TYPE_CHECKING:  # for type hints only: these modules import this one
    from ask import Budget, Timed
    from by_hand import BitsResult, GroupWalk
    from compare import Build, BuildResult, FileCheck, WeightsCheck
    from convert import RecipeCheck, Storage
    from lfm import LoadedModel
    from matrix_source import Matrix
    from tasks import Task

INDENT = "  "
WRAP_WIDTH = 88
PERCENT = 100
SHOWN_NUMBERS = 8  # how many numbers of the shown group are listed one by one


def print_title(script: str, title: str) -> None:
    """Print the script's first line.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    print(f"{script} · {title}")


def wrapped(text: str, indent: str = INDENT) -> str:
    """Wrap a long line, indenting its continuation lines a little more.

    Args:
        text: One line of text.
        indent: The indent of the first line.

    Returns:
        The text as one or more lines.
    """
    return textwrap.fill(
        text,
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + INDENT,
        break_on_hyphens=False,
        break_long_words=False,  # a path stays whole, even past the margin
    )


def share(part: int, whole: int) -> str:
    """Write a part of a whole as a percentage.

    Args:
        part: The part.
        whole: The whole (not 0).

    Returns:
        Such as ``"60.9%"``.
    """
    return f"{PERCENT * part / whole:.1f}%"


# ---------------------------------------------------------------- by_hand.py


def print_matrix(matrix: Matrix, values: np.ndarray) -> None:
    """Say which matrix is quantized and where its bytes came from.

    Args:
        matrix: The matrix's bytes and their source.
        values: Its numbers.
    """
    rows, columns = matrix.pin.shape
    typical = float(np.sqrt(np.mean(np.square(values.astype(np.float64)))))
    print(f"matrix  {matrix.pin.name}")
    print(f"        {rows:,} x {columns:,} = {values.size:,} numbers in bfloat16")
    print(f"read    {len(matrix.data):,} bytes, that matrix alone, from")
    print(wrapped(matrix.source, indent=" " * 8))
    print("        (its SHA-256 fingerprint checked)")
    print(
        f"its numbers run from {values.min():.6f} to {values.max():.6f}; their "
        f"typical size is {typical:.6f}"
    )
    print(f"{INDENT}(typical size: the root mean square, sqrt of the mean of squares)")


def print_naive_group(walk: GroupWalk, bits: int, top: int) -> None:
    """Show one group of 64 quantized by the naive rule, number by number.

    Args:
        walk: The group under both rules.
        bits: Bits per code.
        top: The highest code (15 at 4 bits).
    """
    values, naive, first = walk.values, walk.naive, walk.first
    low, high = int(values.argmin()), int(values.argmax())
    scale, offset = float(naive.scales[0]), float(naive.offsets[0])
    rebuilt = naive.rebuilt()[0]
    print(f"\n1. one group of {values.size} numbers, the naive rule")
    print(
        f"row {walk.row}, numbers {first} to {first + values.size - 1}: "
        f"min {values[low]:.6f} (number {first + low}), max {values[high]:.6f} "
        f"(number {first + high})"
    )
    print(
        f"{INDENT}scale   = (max - min) / {top} = {scale:.6f}  "
        f"({bits} bits count 0 to {top}: {top + 1} levels, {top} steps)"
    )
    print(f"{INDENT}offset  = min = {offset:.6f}")
    print(f"{INDENT}code    = round((w - offset) / scale), a whole number 0 to {top}")
    print(f"{INDENT}rebuilt = scale x code + offset")
    print(
        f"{INDENT}{INDENT}(the file keeps scale and offset as bfloat16, 16 bits each)"
    )
    print(f"{INDENT}{'w':>10}  {'code':>4}  {'rebuilt':>10}  {'error':>9}")
    for index in range(SHOWN_NUMBERS):
        print(
            f"{INDENT}{values[index]:>10.6f}  {naive.codes[0, index]:>4}  "
            f"{rebuilt[index]:>10.6f}  {rebuilt[index] - values[index]:>+9.6f}"
        )
    print(
        f"{INDENT}this group: mean |error| {walk.naive_error.mean:.6f}, largest "
        f"{walk.naive_error.largest:.6f} (half a step is {scale / 2:.6f})"
    )


def print_mlx_group(walk: GroupWalk, top: int) -> None:
    """Show the same group under MLX's rule, beside the naive codes.

    Args:
        walk: The group under both rules.
        top: The highest code (15 at 4 bits).
    """
    mlx = walk.mlx
    step, offset = float(mlx.scales[0]), float(mlx.offsets[0])
    rebuilt = mlx.rebuilt()[0]
    print("\n2. the same group, MLX's rule (what mx.quantize does)")
    if step > 0:
        print(
            f"the anchor is the end farther from 0: the min, {offset:.6f}; codes "
            "count up from it"
        )
    else:
        print(
            f"the anchor is the end farther from 0: the max, {offset:.6f}; codes "
            "count down from it (the step is negative)"
        )
    print(
        f"{INDENT}raw step = (max - min) / {top} = {walk.raw_step:.6f}; the anchor "
        f"is {walk.edge_steps:.2f} steps from 0"
    )
    if walk.whole_steps:
        print(
            f"{INDENT}nudged: {walk.edge_steps:.2f} rounds to {walk.whole_steps}, "
            f"so step = anchor / {walk.whole_steps} = {step:+.6f},"
        )
        print(f"{INDENT}{INDENT}and 0 is exactly code {-walk.whole_steps}")
        print(f"{INDENT}offset = the anchor = {offset:.6f}")
    else:
        print(f"{INDENT}it rounds to 0, so the step stays and the offset is 0")
    print(
        f"{INDENT}code   = round((w - offset) / step), halves away from 0, "
        f"at most {top}"
    )
    if walk.far_code > top:
        print(
            wrapped(
                f"the far end, {walk.far_end:.6f}, is {walk.far_code:.2f} steps away: "
                f"the nudged step is shorter, so it is capped at code {top}"
            )
        )
    heads = f"{'w':>10}  {'naive code':>10}  {'MLX code':>8}  {'MLX rebuilt':>11}"
    print(f"{INDENT}{heads}")
    for index in range(SHOWN_NUMBERS):
        print(
            f"{INDENT}{walk.values[index]:>10.6f}  {walk.naive.codes[0, index]:>10}  "
            f"{mlx.codes[0, index]:>8}  {rebuilt[index]:>11.6f}"
        )
    size = walk.values.size
    print(
        f"{INDENT}codes: the naive rule and mx.quantize differ in {walk.naive_differ} "
        f"of {size}; MLX's rule by hand"
    )
    print(f"{INDENT}{INDENT}matches mx.quantize in {walk.mlx_same} of {size}")


def print_whole_matrix(four: BitsResult, group_size: int, bf16_bytes: int) -> None:
    """Compare both rules with mx.quantize over the whole matrix.

    Args:
        four: The result at the chapter's bit width.
        group_size: Numbers per group.
        bf16_bytes: The matrix's size in bfloat16.
    """
    groups, numbers = four.group_count, four.number_count
    pad = INDENT * 2
    print(
        f"\n3. the whole matrix at {four.bits} bits: {groups:,} groups of {group_size}"
    )
    print(f"{INDENT}codes, compared with mx.quantize's:")
    print(
        f"{pad}the naive rule        {four.naive_codes_differ:,} of {numbers:,} "
        f"differ ({share(four.naive_codes_differ, numbers)}), in "
        f"{four.naive_groups_differ:,} groups"
    )
    print(
        f"{pad}MLX's rule, by hand   {four.codes_differ:,} differ; scales "
        f"{four.scales_differ:,} and offsets {four.offsets_differ:,} of {groups:,} "
        "differ"
    )
    print(
        f"{INDENT}rebuilt by hand against mx.dequantize: {four.rebuilt_differ:,} of "
        f"{numbers:,} numbers differ"
    )
    print(
        wrapped(
            f"why the naive codes differ: MLX anchors {four.anchored_at_max:,} of the "
            f"{groups:,} groups at their max, where its codes count down, and its "
            "nudged step shifts the codes of the rest"
        )
    )
    print(
        f"{INDENT}groups whose numbers are all negative or all positive: "
        f"{four.one_signed:,}"
    )
    print(f"{INDENT}the error, rule against rule")
    print(
        f"{INDENT}(relative: the error's typical size over the numbers' typical size):"
    )
    print(f"{pad}{'':<12}  {'mean |error|':>12}  {'largest':>8}  {'relative':>8}")
    rows = (("naive rule", four.naive_error), ("MLX's rule", four.mlx_error))
    for name, error in rows:
        print(
            f"{pad}{name:<12}  {error.mean:>12.6f}  {error.largest:>8.4f}  "
            f"{PERCENT * error.relative:>7.2f}%"
        )
    if four.naive_error.relative < four.mlx_error.relative:
        print(
            wrapped(
                "the naive rule's error is slightly smaller: MLX gives up a little "
                "accuracy to store 0 exactly"
            )
        )
    else:
        print(f"{INDENT}MLX's rule has the smaller error")
    print(
        f"{INDENT}storage: {numbers:,} codes x {four.bits} bits + {groups:,} "
        f"x ({four.scale_bits} + {four.offset_bits}) bits = {four.expected_bytes:,} "
        "bytes"
    )
    print(
        f"{pad}(MLX's arrays: {four.stored_bytes:,}); in bfloat16 "
        f"{bf16_bytes:,}: {bf16_bytes / four.stored_bytes:.2f} times smaller"
    )


def print_sweep(
    results: Sequence[BitsResult],
    group_size: int,
    levels: Callable[[int], int],
    bits_per_number: Callable[[int], float],
) -> None:
    """Print the error at each bit width, and whether MLX agrees.

    Args:
        results: One result per bit width.
        group_size: Numbers per group.
        levels: The highest code at a bit width.
        bits_per_number: Storage per number at a bit width.
    """
    widths = listed([result.bits for result in results])
    print(f"\n4. MLX's rule at {widths} bits (groups of {group_size})")
    print(
        f"{INDENT}{'bits':>4}  {'codes':>5}  {'bits/number':>11}  "
        f"{'mean |error|':>12}  {'largest':>7}  {'relative':>8}  {'naive':>6}  "
        "= mx.quantize"
    )
    for result in results:
        error = result.mlx_error
        print(
            f"{INDENT}{result.bits:>4}  {levels(result.bits) + 1:>5}  "
            f"{bits_per_number(result.bits):>11.1f}  {error.mean:>12.6f}  "
            f"{error.largest:>7.4f}  {PERCENT * error.relative:>7.2f}%  "
            f"{PERCENT * result.naive_error.relative:>5.2f}%  "
            f"{'yes' if result.matches_mlx() else 'NO'}"
        )
    steps = [
        f"{left.bits} -> {right.bits} bits: "
        f"x{left.mlx_error.relative / right.mlx_error.relative:.1f}"
        for left, right in zip(results, results[1:], strict=False)
    ]
    print(wrapped("the relative error shrinks " + "; ".join(steps)))
    print(f"{INDENT}(naive: the naive rule's relative error at the same bits)")


def listed(numbers: Sequence[int]) -> str:
    """Write numbers as a list in words.

    Args:
        numbers: The numbers.

    Returns:
        Such as ``"3, 4, 6 and 8"``.
    """
    words = [str(number) for number in numbers]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def print_verdict(results: Sequence[BitsResult]) -> None:
    """Say that the hand-written rule is MLX's, at every bit width.

    Args:
        results: One result per bit width.
    """
    print()
    print(
        wrapped(
            "checked: MLX's rule by hand gives exactly mx.quantize's codes, scales "
            f"and offsets at {listed([result.bits for result in results])} bits",
            indent="",
        )
    )
    sys.stdout.flush()


# ---------------------------------------------------------------- convert.py


def print_build_plan(name: str, source: Path, target: Path) -> None:
    """Say what is built, from what, and where it goes.

    Args:
        name: The build, in a few words.
        source: The bfloat16 model's folder.
        target: Where the build goes.
    """
    print(f"build   {name}")
    print(f"from    {short_path(source)} (chapter 10's bfloat16 download)")
    print(f"into    {short_path(target)}")


def print_already_built() -> None:
    """Say that the build is already there and was not redone."""
    print("\nalready built (add --force to build it again); checking it:")


def print_recipe_check(check: RecipeCheck, recipe: str) -> None:
    """Show why mlx-lm's built-in mixed recipe cannot be used for LFM2.

    Args:
        check: What the recipe did on the model's layers.
        recipe: The recipe's name.
    """
    print(f"\nmlx-lm's built-in recipe first: quant_predicate={recipe!r}")
    if check.error is None:
        print(f"{INDENT}it would work on this model")
        return
    print(f"{INDENT}it stops: ValueError: {check.error}")
    head = "has" if check.has_head else "does not have"
    print(
        wrapped(
            "the recipe looks for layers named down_proj, and "
            f"{check.named_like_recipe} of this model's {check.quantizable} "
            "quantizable layers are (LFM2 calls its feed-forward matrices w1, w2 and "
            f"w3); it would also raise lm_head to 6 bits, and LFM2 {head} one: its "
            "embedding table is also its output head"
        )
    )


def print_predicate(
    predicate: Callable[[str, Any], bool | dict[str, Any]], embedding: str, layer: str
) -> None:
    """Show our quant_predicate's answers, by calling it.

    Args:
        predicate: The function mlx-lm will call once per layer.
        embedding: The embedding table's layer name.
        layer: Any other layer's name.
    """
    print("\nso convert.py passes its own quant_predicate(path, module):")
    print(
        f"{INDENT}two arguments, as mlx-lm calls it (mlx_lm/utils.py, quantize_model)"
    )
    print(f"{INDENT}{embedding!r:<34} -> {predicate(embedding, None)}")
    print(f"{INDENT}{layer!r:<34} -> {predicate(layer, None)} (the defaults below)")


def print_convert_call(
    group_size: int, bits: int, mode: str, predicate: str | None
) -> None:
    """Print the mlx_lm.convert call about to run.

    Args:
        group_size: Numbers per group.
        bits: Bits per code.
        mode: The quantization mode.
        predicate: The name of the quant_predicate passed, or None.
    """
    extra = f", quant_predicate={predicate}" if predicate else ""
    print(
        wrapped(
            f"mlx_lm.convert(source, target, quantize=True, q_group_size={group_size}, "
            f"q_bits={bits}, q_mode={mode!r}{extra})",
            indent="",
        )
    )
    sys.stdout.flush()  # before mlx-lm's own [INFO] lines


def print_built(seconds: float, peak: int) -> None:
    """Say how long the build took and how much memory it used.

    Args:
        seconds: Its duration.
        peak: MLX's peak memory in bytes.
    """
    print(f"built in {seconds:.0f} s, peak memory {gigabytes(peak)} (MLX's count)")


def print_quantization(
    ours: dict[str, Any] | None, theirs: dict[str, Any] | None
) -> None:
    """Print the build's quantization block beside Liquid's.

    Args:
        ours: The build's config.json quantization block.
        theirs: Liquid's export's block, or None if it is not downloaded.
    """
    print("\nthe new config.json's quantization block:")
    if ours is None:
        print(f"{INDENT}none: this build is not quantized")
        return
    print(wrapped(json.dumps(ours)))
    if theirs is None:
        print(f"{INDENT}(Liquid's export is not downloaded, so it is not compared)")
    elif theirs == ours:
        print(f"{INDENT}LiquidAI/LFM2.5-2.6B-MLX-4bit's config.json: the same block")
    else:
        print(f"{INDENT}LiquidAI/LFM2.5-2.6B-MLX-4bit's config.json differs:")
        print(wrapped(json.dumps(theirs)))


def print_storage(storage: Storage, bf16_bytes: int) -> None:
    """Print what the weights file holds, kind by kind, and the arithmetic.

    Args:
        storage: The build's weights file, counted from its header.
        bf16_bytes: The bfloat16 original's bytes of numbers.
    """
    print("\nwhat model.safetensors holds, counted from its header:")
    print(
        f"{INDENT}{'kind':<18}  {'layers':>6}  {'numbers':>13}  {'bits each':>9}  "
        f"{'bytes':>13}"
    )
    for row in storage.rows:
        print(
            f"{INDENT}{row.kind:<18}  {row.modules:>6}  {row.numbers:>13,}  "
            f"{row.bits_each():>9.2f}  {row.nbytes:>13,}"
        )
    data = storage.data_bytes()
    numbers = sum(row.numbers for row in storage.rows)
    terms = " + ".join(f"{row.numbers:,} x {row.bits_each():g}" for row in storage.rows)
    print(wrapped(f"in bits: {terms} = {data * BITS_PER_BYTE:,} bits = {data:,} bytes"))
    total = data + storage.header_bytes
    agrees = (
        "the file's size exactly" if total == storage.file_bytes else "NOT its size"
    )
    print(
        f"{INDENT}plus the {storage.header_bytes:,}-byte header: {total:,} bytes, "
        f"{agrees}"
    )
    print(
        f"{INDENT}{gigabytes(storage.file_bytes)} ({gibibytes(storage.file_bytes)}); "
        f"{numbers:,} numbers, {BITS_PER_BYTE * data / numbers:.3f} bits each on "
        "average"
    )
    print(
        f"{INDENT}the bfloat16 original's numbers take {gigabytes(bf16_bytes)}: "
        f"{bf16_bytes / data:.2f} times as much"
    )


# ---------------------------------------------------------------- compare.py

SHORT_LABELS = {
    KEY_BF16: "bf16",
    KEY_OURS: "ours",
    KEY_ALL_FOUR: "all4",
    KEY_LIQUID: "Liquid",
}
LABEL_WIDTH = 16
TASK_WIDTH = 20
FILE_WIDTH = 29


def print_builds(builds: Sequence[Build]) -> None:
    """List the builds to measure and where each one is.

    Args:
        builds: The builds, in the order they are measured.
    """
    print("builds, loaded one at a time:")
    for build in builds:
        print(f"{INDENT}{build.label:<{LABEL_WIDTH}} {short_path(build.folder)}")


def print_tests(
    agreement: Sequence[Task], held_out: Sequence[Task], short: int, cap: int
) -> None:
    """Say what every build is asked, the same for each.

    Args:
        agreement: The 20 training questions.
        held_out: The 24 pass-rate questions.
        short: Tokens compared per agreement reply.
        cap: The pass-rate replies' length cap.
    """
    databases = sorted({task.domain for task in held_out})
    per_level = {
        level: sum(task.difficulty == level for task in held_out)
        for level in ("easy", "medium", "hard")
    }
    levels_text = ", ".join(f"{count} {level}" for level, count in per_level.items())
    print("the same tests for every build, every reply greedy:")
    print(
        wrapped(
            f"log-probability: the gold SQL of {len(agreement)} training questions, "
            "each read after <|startoftext|>"
        )
    )
    print(
        wrapped(
            f"agreement: the same {len(agreement)} questions, each asked as "
            f'"<question> Answer with one SQL query.", the first {short} tokens of '
            "each reply"
        )
    )
    print(
        wrapped(
            f"pass rate: the project's {len(held_out)} held-out questions "
            f"({' and '.join(databases)}; {levels_text}), the tables in the prompt, "
            f"one reply of up to {cap:,} tokens, graded by chapter 1's grader"
        )
    )


def print_file_checks(
    files: Sequence[FileCheck], weights: WeightsCheck, rendered: tuple[int, int]
) -> None:
    """Print ours against Liquid's export, file by file and tensor by tensor.

    Args:
        files: One check per file.
        weights: The weights file, whole and tensor by tensor.
        rendered: Prompts rendered to the same tokens by both templates, and
            how many were rendered.
    """
    print(
        "\n1. ours against LiquidAI/LFM2.5-2.6B-MLX-4bit, file by file "
        "(no model loaded)"
    )
    print(
        f"{INDENT}{'file':<{FILE_WIDTH}}  {'ours, bytes':>13}  {'Liquid, bytes':>13}"
        "  same bytes"
    )
    for check in files:
        ours = f"{check.ours_bytes:,}" if check.ours_bytes is not None else "absent"
        theirs = (
            f"{check.theirs_bytes:,}" if check.theirs_bytes is not None else "absent"
        )
        print(
            f"{INDENT}{check.name:<{FILE_WIDTH}}  {ours:>13}  {theirs:>13}  "
            f"{'yes' if check.identical else 'no'}"
        )
        if check.note:
            print(wrapped(check.note, indent=INDENT * 2))
    print(
        f"{INDENT}model.safetensors, tensor by tensor: {weights.identical} of "
        f"{weights.tensors} tensors hold identical bytes"
    )
    if weights.differ:
        print(wrapped(f"differ: {', '.join(weights.differ)}", indent=INDENT * 2))
    for side, names in (
        ("only ours", weights.only_ours),
        ("only Liquid's", weights.only_theirs),
    ):
        if names:
            print(wrapped(f"{side}: {', '.join(names)}", indent=INDENT * 2))
    print(f"{INDENT}MD5 of the whole file:")
    print(f"{INDENT}{INDENT}ours     {weights.ours_md5}")
    print(f"{INDENT}{INDENT}Liquid's {weights.theirs_md5}")
    same, total = rendered
    print(
        wrapped(
            f"the two chat templates render {same} of the {total} prompts this script "
            "asks to the same tokens"
        )
    )


def print_fast_skips() -> None:
    """Say what --fast leaves out, and why."""
    print(
        wrapped(
            "fast mode: Liquid's export alone. Comparing ours with it needs "
            "convert.py's build, and agreement needs the bfloat16 original, so both "
            "are skipped",
            indent="",
        )
    )


def progress_line(result: BuildResult) -> str:
    """Write one line saying a build is done.

    Args:
        result: Its measurements.

    Returns:
        The line.
    """
    return (
        f"{result.build.label}: loaded in {result.load_seconds:.1f} s, tested in "
        f"{result.seconds:.0f} s, {result.correct()} of {len(result.answers)} correct"
    )


def print_size_and_speed(results: Sequence[BuildResult]) -> None:
    """Print each build's size, memory and speed.

    Args:
        results: One result per build.
    """
    print("\n2. size and speed")
    print(
        f"{INDENT}{'build':<{LABEL_WIDTH}}  {'on disk':>8}  {'in memory':>9}  "
        f"{'peak':>8}  {'prompt tok/s':>12}  {'reply tok/s':>11}"
    )
    for result in results:
        print(
            f"{INDENT}{result.build.label:<{LABEL_WIDTH}}  "
            f"{gigabytes(result.disk_bytes):>8}  {gigabytes(result.loaded_bytes):>9}  "
            f"{gigabytes(result.peak_bytes):>8}  {result.prompt_speed():>12.1f}  "
            f"{result.reply_speed():>11.1f}"
        )
    print(
        wrapped(
            "in memory: MLX's memory in use once the build is loaded, the number the "
            "project's engine logs when it starts; peak: over all the tests; speeds: "
            "over the pass-rate questions (reading the prompts, writing the replies)"
        )
    )


def agreement(result: BuildResult, reference: BuildResult) -> tuple[int, int, int]:
    """Compare a build's replies with the reference's.

    Args:
        result: The build.
        reference: The bfloat16 build.

    Returns:
        Identical replies, positions with the same top token, and positions.
    """
    same_replies = sum(
        mine == theirs
        for mine, theirs in zip(result.greedy, reference.greedy, strict=True)
    )
    pairs = [
        (mine, theirs)
        for mine_reply, theirs_reply in zip(
            result.forced, reference.forced, strict=True
        )
        for mine, theirs in zip(mine_reply, theirs_reply, strict=True)
    ]
    return same_replies, sum(mine == theirs for mine, theirs in pairs), len(pairs)


def print_quality(results: Sequence[BuildResult], short: int) -> None:
    """Print each build's log-probability, agreement and pass rate.

    Args:
        results: One result per build; bfloat16 first when it was measured.
        short: Tokens compared per agreement reply.
    """
    reference = results[0] if results[0].build.key == KEY_BF16 else None
    print("\n3. quality")
    print(
        f"{INDENT}{'build':<{LABEL_WIDTH}}  {'log-prob/token':>14}  "
        f"{'same reply':>10}  {'token agreement':>15}  {'pass rate':>9}"
    )
    for result in results:
        if reference is None:
            same, agree = "n/a", "n/a"
        elif result is reference:
            same, agree = "reference", "reference"
        else:
            replies, matched, positions = agreement(result, reference)
            same = f"{replies} of {len(result.greedy)}"
            agree = share(matched, positions)
        print(
            f"{INDENT}{result.build.label:<{LABEL_WIDTH}}  "
            f"{result.logprob_per_token():>14.4f}  {same:>10}  {agree:>15}  "
            f"{result.correct():>3} of {len(result.answers)}"
        )
    tokens = results[0].logprob_tokens
    print(
        wrapped(
            f"log-prob/token: the average log-probability of the {tokens} tokens of "
            "SQL (closer to 0: the build finds the texts more likely); same reply: "
            f"the first {short} tokens identical to bfloat16's; token agreement: the "
            "share of positions in bfloat16's replies where the build's top token "
            "is bfloat16's"
        )
    )
    if reference is not None:
        own = sum(
            mine == theirs
            for forced, greedy in zip(reference.forced, reference.greedy, strict=True)
            for mine, theirs in zip(forced, greedy, strict=True)
        )
        total = sum(len(reply) for reply in reference.greedy)
        print(
            wrapped(
                "(check: bfloat16's top tokens in one pass equal its own "
                f"token-by-token replies at {own:,} of {total:,} positions)"
            )
        )


def print_task_by_task(
    results: Sequence[BuildResult],
    tasks: Sequence[Task],
    exact_p: Callable[[int, int], float],
) -> None:
    """Print every pass-rate question's verdict per build, and gained vs lost.

    Args:
        results: One result per build.
        tasks: The pass-rate questions, in order.
        exact_p: The exact p-value for (gained, lost).
    """
    heads = [SHORT_LABELS.get(result.build.key, result.build.key) for result in results]
    print(f"\n4. the {len(tasks)} questions, task by task (1 correct, 0 wrong)")
    print(f"{INDENT}{'task':<{TASK_WIDTH}}" + "".join(f"{head:>8}" for head in heads))
    for index, task in enumerate(tasks):
        marks = "".join(
            f"{int(result.answers[index].graded.correct):>8}" for result in results
        )
        print(f"{INDENT}{task.task_id:<{TASK_WIDTH}}{marks}")
    print(
        f"{INDENT}{'correct':<{TASK_WIDTH}}"
        + "".join(f"{result.correct():>8}" for result in results)
    )
    if results[0].build.key != KEY_BF16 or len(results) < 2:
        return
    reference = results[0]
    rows: dict[str, list[str]] = {"gained": [""], "lost": [""], "exact p": [""]}
    for result in results[1:]:
        gained = sum(
            mine.graded.correct and not theirs.graded.correct
            for mine, theirs in zip(result.answers, reference.answers, strict=True)
        )
        lost = sum(
            theirs.graded.correct and not mine.graded.correct
            for mine, theirs in zip(result.answers, reference.answers, strict=True)
        )
        rows["gained"].append(str(gained))
        rows["lost"].append(str(lost))
        rows["exact p"].append(f"{exact_p(gained, lost):.3f}")
    print(f"{INDENT}against bfloat16:")
    for name, cells in rows.items():
        print(
            f"{INDENT}{INDENT}{name:<{TASK_WIDTH - 2}}"
            + "".join(f"{c:>8}" for c in cells)
        )
    print(
        wrapped(
            "gained: right here, wrong in bfloat16; lost: the reverse; exact p: the "
            "chance of a split at least this lopsided if the two builds were equally "
            "good (McNemar's exact test)"
        )
    )


def print_ours_vs_liquid(results: Sequence[BuildResult], same_file: bool) -> None:
    """Compare ours with Liquid's export by what they do, and give the verdict.

    Args:
        results: One result per build.
        same_file: True when the two weights files have the same MD5.
    """
    by_key = {result.build.key: result for result in results}
    ours, theirs = by_key[KEY_OURS], by_key[KEY_LIQUID]
    replies = sum(a == b for a, b in zip(ours.greedy, theirs.greedy, strict=True))
    pairs = [
        (a, b)
        for mine, other in zip(ours.forced, theirs.forced, strict=True)
        for a, b in zip(mine, other, strict=True)
    ]
    forced = sum(a == b for a, b in pairs)
    answers = sum(
        a.tokens == b.tokens for a, b in zip(ours.answers, theirs.answers, strict=True)
    )
    print("\n5. ours against Liquid's export, by what they do")
    print(f"{INDENT}agreement replies identical: {replies} of {len(ours.greedy)}")
    print(
        f"{INDENT}top tokens along bfloat16's replies identical: {forced:,} of "
        f"{len(pairs):,}"
    )
    print(
        f"{INDENT}pass-rate replies identical: {answers} of {len(ours.answers)}; "
        f"correct: ours {ours.correct()}, Liquid's {theirs.correct()}"
    )
    behaves_same = replies == len(ours.greedy) and answers == len(ours.answers)
    if same_file and behaves_same:
        verdict = (
            "byte-identical weights and identical greedy replies: our build is "
            "Liquid's export, made again on this laptop"
        )
    elif same_file:
        verdict = "byte-identical weights, yet some replies differ: see above"
    else:
        verdict = "the weights differ, so the builds are not the same: see above"
    print(wrapped(f"verdict: {verdict}", indent=""))


def print_saved(path: Path) -> None:
    """Say where every measurement was saved.

    Args:
        path: The file.
    """
    print(f"\nsaved {short_path(path)}: every reply, verdict and measurement")


# -------------------------------------------------------------------- ask.py

COLUMN = 18  # the before/after table's column width


def print_model(loaded: LoadedModel, which: str) -> None:
    """Say which build answers, and its size in memory.

    Args:
        loaded: The model.
        which: A few words saying which build it is.
    """
    print(f"model     {short_path(loaded.folder)}")
    print(f"          {which}; {gigabytes(loaded.weight_bytes)} of weights in memory")


def print_warm_up(seconds: float) -> None:
    """Say how long the untimed warm-up pass took.

    Args:
        seconds: Its duration.
    """
    print(f"warm-up   {seconds:.2f} s for one untimed token (one-time GPU set-up)")


def print_question(question: str, prompt_tokens: int) -> None:
    """Print the question and the prompt's length, before the reply streams.

    Args:
        question: What the user asks.
        prompt_tokens: How many tokens the prompt is.
    """
    print(f"question  {question}")
    print(f"prompt    {prompt_tokens} tokens, ending with <think>; decoding is greedy")
    print("\nthe reply, printed as each token is predicted:\n")
    sys.stdout.flush()


def show_piece(text: str) -> None:
    """Print one piece of the reply at once, with no newline after it.

    Args:
        text: The piece.
    """
    print(text, end="", flush=True)


def end_stream() -> None:
    """End the streamed reply's last line."""
    print()


def print_timing(timed: Timed) -> None:
    """Print how long the reply took and the peak memory.

    Args:
        timed: The timed reply.
    """
    ending = (
        "then <|im_end|>, its end of turn (finish_reason stop)"
        if timed.finish_reason == FINISH_STOP
        else "cut off by the length cap (finish_reason length)"
    )
    print(f"\nthe reply: {len(timed.texts)} tokens, {ending}")
    first = timed.seconds[0] if timed.seconds else 0.0
    per_token = timed.ms_per_token()
    line = f"first token after {first:.2f} s"
    if per_token:
        line += (
            f"; then {per_token:.1f} ms per token "
            f"({MILLISECONDS / per_token:.1f} tokens a second)"
        )
    print(line)
    print(f"peak memory {gigabytes(timed.peak_bytes)} (MLX's count)")


def print_before_after(
    before: dict[str, Any] | None, timed: Timed, loaded: LoadedModel, shared: str | None
) -> None:
    """Print chapter 10's bfloat16 run of the same question beside this one.

    Args:
        before: Chapter 10's saved measurement, or None.
        timed: This reply.
        loaded: This model.
        shared: The SQL both replies end with, if it is the same.
    """
    if before is None:
        print()
        print(
            wrapped(
                "(no 'before' picture: run chapter 10's first_reply.py with the same "
                "question to compare)",
                indent="",
            )
        )
        return
    print("\nbefore and after: chapter 10's bfloat16 run of the same question")
    print(f"{INDENT}{'':<14}{'bfloat16':>{COLUMN}}{'4-bit':>{COLUMN}}")
    rows = (
        ("weights", gigabytes(before["weight_bytes"]), gigabytes(loaded.weight_bytes)),
        (
            "peak memory",
            gigabytes(before["peak_memory_bytes"]),
            gigabytes(timed.peak_bytes),
        ),
        (
            "ms per token",
            f"{before['ms_per_token']:.1f}",
            f"{timed.ms_per_token():.1f}",
        ),
        ("reply tokens", str(before["reply_tokens"]), str(len(timed.texts))),
    )
    for name, old, new in rows:
        print(f"{INDENT}{name:<14}{old:>{COLUMN}}{new:>{COLUMN}}")
    if shared:
        print(f"{INDENT}both end with the same query: {shared}")
    else:
        print(f"{INDENT}the two replies end with different queries (or none)")


def print_budget(budget: Budget, peak: int, before: dict[str, Any] | None) -> None:
    """Explain the memory budget of an Apple-silicon Mac, with this Mac's numbers.

    Args:
        budget: This Mac's memory, as Metal reports it.
        peak: This run's peak memory.
        before: Chapter 10's saved bfloat16 run, or None.
    """
    print("\nthe real budget: unified memory")
    print(
        wrapped(
            f"this Mac has {gibibytes(budget.total_bytes)} of memory "
            f"({gigabytes(budget.total_bytes)}), one pool the CPU and the GPU share: "
            "the weights are loaded once and the GPU works on them where they are"
        )
    )
    print(
        wrapped(
            "Metal recommends the GPU keep at most "
            f"{gigabytes(budget.gpu_limit_bytes)} of it in use "
            f"({share(budget.gpu_limit_bytes, budget.total_bytes)}); the rest is left "
            "to macOS and the other programs"
        )
    )
    line = (
        f"this run's peak, {gigabytes(peak)}, is "
        f"{share(peak, budget.gpu_limit_bytes)} of that"
    )
    if before is not None:
        old = before["peak_memory_bytes"]
        line += (
            f"; chapter 10's bfloat16 run peaked at {gigabytes(old)} "
            f"({share(old, budget.gpu_limit_bytes)})"
        )
    print(wrapped(line))
