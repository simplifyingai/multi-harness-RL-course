"""Chapter 13 without MLX: where files go, the adapter arithmetic, and the printing.

No test here loads a model, downloads anything or needs MLX. The model
folders and adapter files are tiny fakes written here; the arithmetic runs on
LFM2.5-2.6B's real shapes, typed in from its config.json, so the chapter's
headline numbers (66 pairs, 9,781,248 numbers, 0.363%) are checked anywhere.
MLX's side (the adapters themselves, training, saving, merging, AdamW) is in
test_13_mlx.py.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from chapters import use_chapter

CHAPTER = use_chapter("13-lora")

import arithmetic  # noqa: E402
import lfm_files  # noqa: E402
import report  # noqa: E402
from arithmetic import AdapterPair, LoraShape, WeightChange  # noqa: E402

# LFM2.5-2.6B as its config.json describes it (chapter 10 printed these).
HIDDEN = 2_048
KEY_VALUE = 8 * 64  # 8 key/value heads of 64 numbers
FEED_FORWARD = 10_752
CONV_IN = 3 * HIDDEN  # the convolution block's in_proj makes three streams
LAYERS = 30
ATTENTION_LAYERS = (2, 5, 9, 13, 17, 21, 24, 27)
MODEL_NUMBERS = 2_697_198_592  # chapter 10's count
SHAPES = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (HIDDEN, KEY_VALUE),
    "self_attn.v_proj": (HIDDEN, KEY_VALUE),
    "self_attn.out_proj": (HIDDEN, HIDDEN),
    "conv.in_proj": (HIDDEN, CONV_IN),
    "conv.out_proj": (HIDDEN, HIDDEN),
    "feed_forward.w1": (HIDDEN, FEED_FORWARD),
    "feed_forward.w2": (FEED_FORWARD, HIDDEN),
    "feed_forward.w3": (HIDDEN, FEED_FORWARD),
}
ATTENTION_KEYS = [key for key in SHAPES if not key.startswith("conv.")]
CONV_KEYS = [key for key in SHAPES if not key.startswith("self_attn.")]


def lfm_kinds() -> dict[int, str]:
    """Say which of LFM2.5-2.6B's 30 layers are attention.

    Returns:
        Each layer's kind.
    """
    return {
        layer: arithmetic.ATTENTION
        if layer in ATTENTION_LAYERS
        else arithmetic.CONVOLUTION
        for layer in range(LAYERS)
    }


def lfm_pairs(rank: int = 16, last: int = 12) -> list[AdapterPair]:
    """Build the adapter pairs the project puts on LFM2.5-2.6B.

    Args:
        rank: The LoRA rank.
        last: How many of the last layers get adapters.

    Returns:
        One pair per adapted projection.
    """
    kinds = lfm_kinds()
    pairs = []
    for layer in range(LAYERS - last, LAYERS):
        keys = ATTENTION_KEYS if kinds[layer] == arithmetic.ATTENTION else CONV_KEYS
        for key in keys:
            inputs, outputs = SHAPES[key]
            pairs.append(AdapterPair(layer, key, inputs, outputs, rank))
    return pairs


# -------------------------------------------------------------- lfm_files.py


def write_build(folder: Path) -> Path:
    """Write the files that make a folder count as a finished build.

    Args:
        folder: The build's folder.

    Returns:
        The folder.
    """
    folder.mkdir(parents=True)
    for name in lfm_files.BUILD_FILES:
        (folder / name).write_text("x")
    return folder


def write_liquid_export(home: Path) -> Path:
    """Write files of the right sizes where Liquid's export should be.

    Args:
        home: The course home.

    Returns:
        The export's folder.
    """
    folder = home / "models" / lfm_files.FOUR_BIT.folder
    folder.mkdir(parents=True)
    for item in lfm_files.FOUR_BIT.files:
        with (folder / item.name).open("wb") as handle:
            handle.truncate(item.size)  # a sparse file: the size, no real bytes
    return folder


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """Write a small safetensors file by hand: length, JSON header, then zeros.

    Args:
        path: Where to write it.
        tensors: Each tensor's number type and shape.

    Returns:
        The path.
    """
    sizes = {"F32": 4, "BF16": 2}
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        length = math.prod(shape) * sizes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + length],
        }
        offset += length
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return path


def test_course_home_follows_the_environment(course_home: Path) -> None:
    """COURSE_HOME decides every folder; the chapter writes under its own."""
    assert lfm_files.course_home() == course_home
    assert lfm_files.chapter_home() == course_home / "13-lora"
    assert lfm_files.ours_folder() == course_home / "models" / lfm_files.OURS_FOLDER


def test_ours_comes_first(course_home: Path) -> None:
    """With chapter 11's build on disk, the script uses it."""
    write_build(course_home / "models" / lfm_files.OURS_FOLDER)
    write_liquid_export(course_home)
    folder, which = lfm_files.pick_model(fast=False)
    assert folder == lfm_files.ours_folder()
    assert "ours" in which


def test_liquids_export_is_the_fallback(course_home: Path) -> None:
    """Without our build, Liquid's export is used, and the reason is said."""
    export = write_liquid_export(course_home)
    folder, which = lfm_files.pick_model(fast=False)
    assert folder == export
    assert "convert.py" in which


def test_fast_always_uses_liquids_export(course_home: Path) -> None:
    """--fast picks Liquid's export even when ours is built."""
    write_build(course_home / "models" / lfm_files.OURS_FOLDER)
    export = write_liquid_export(course_home)
    assert lfm_files.pick_model(fast=True) == (export, "Liquid's export (--fast)")


def test_no_model_names_the_commands(course_home: Path) -> None:
    """With neither build, the script stops and says how to get one."""
    with pytest.raises(SystemExit, match="convert.py") as stopped:
        lfm_files.pick_model(fast=False)
    assert "download.py --fast" in str(stopped.value)
    with pytest.raises(SystemExit, match="not fully downloaded"):
        lfm_files.pick_model(fast=True)


def test_adapter_files_are_named_like_the_projects(course_home: Path) -> None:
    """Checkpoints are adapter_stepNNN.safetensors; then adapter.safetensors."""
    assert lfm_files.checkpoint_path(5).name == "adapter_step005.safetensors"
    assert lfm_files.checkpoint_path(20).parent == course_home / "13-lora"
    assert lfm_files.final_adapter_path().name == "adapter.safetensors"


def test_clearing_removes_only_adapter_files(course_home: Path) -> None:
    """An earlier run's adapters go; anything else in the folder stays."""
    folder = lfm_files.chapter_home()
    folder.mkdir(parents=True)
    for name in ("adapter_step005.safetensors", "adapter.safetensors", "notes.txt"):
        (folder / name).write_text("x")
    assert lfm_files.clear_adapters() == 2
    assert [path.name for path in folder.iterdir()] == ["notes.txt"]
    assert lfm_files.clear_adapters() == 0


def test_a_safetensors_header_is_read_without_its_numbers(tmp_path: Path) -> None:
    """Tensors, numbers, types and sizes come from the header alone."""
    path = write_safetensors(
        tmp_path / "adapter.safetensors",
        {"a.lora_a": ("F32", [8, 2]), "a.lora_b": ("F32", [2, 4])},
    )
    header = lfm_files.read_safetensors_header(path)
    assert header.tensors == 2  # the metadata entry is not a tensor
    assert header.numbers == 8 * 2 + 2 * 4
    assert header.dtypes == ("F32",)
    assert header.file_bytes == path.stat().st_size
    assert header.file_bytes == header.header_bytes + 4 * header.numbers


def test_a_broken_safetensors_file_is_refused(tmp_path: Path) -> None:
    """Too short for its length field, or ending inside its header: stop."""
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError, match="too short"):
        lfm_files.read_safetensors_header(short)
    cut = tmp_path / "cut.safetensors"
    cut.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(ValueError, match="inside its own header"):
        lfm_files.read_safetensors_header(cut)


def test_sizes_are_written_in_decimal_units() -> None:
    """Gigabytes and megabytes are 10**9 and 10**6 bytes, as the Hub counts."""
    assert lfm_files.gigabytes(1_583_152_892) == "1.58 GB"
    assert lfm_files.megabytes(39_139_440) == "39.1 MB"
    assert lfm_files.short_path(Path.home() / "x") == "~/x"


# ------------------------------------------------------------- arithmetic.py


def test_a_pair_holds_rank_times_in_plus_out() -> None:
    """A is in x rank, B is rank x out: rank x (in + out) numbers."""
    pair = AdapterPair(21, "self_attn.k_proj", HIDDEN, KEY_VALUE, 16)
    assert pair.numbers() == 16 * HIDDEN + 16 * KEY_VALUE == 40_960
    assert pair.full() == HIDDEN * KEY_VALUE


def test_the_projects_adapters_on_lfm25() -> None:
    """66 pairs and 9,781,248 numbers: 3 attention x 7 + 9 convolution x 5."""
    pairs = lfm_pairs()
    layers = arithmetic.by_layer(pairs, lfm_kinds())
    assert len(pairs) == 66
    assert sum(pair.numbers() for pair in pairs) == 9_781_248
    assert [layer.layer for layer in layers] == list(range(18, 30))
    assert arithmetic.kind_counts(layers) == [
        (arithmetic.ATTENTION, 3, 7),
        (arithmetic.CONVOLUTION, 9, 5),
    ]
    numbers = {layer.kind: layer.numbers for layer in layers}
    assert numbers == {arithmetic.ATTENTION: 827_392, arithmetic.CONVOLUTION: 811_008}


def test_the_share_is_0363_percent_and_rounds_to_04() -> None:
    """9,781,248 of chapter 10's 2,697,198,592 is 0.363%, 0.4% to one decimal."""
    trainable = sum(pair.numbers() for pair in lfm_pairs())
    percent = arithmetic.share(trainable, MODEL_NUMBERS)
    assert f"{percent:.3f}" == "0.363"
    assert f"{percent:.1f}" == "0.4"
    full = sum(pair.full() for pair in lfm_pairs())
    assert full == 975_175_680
    assert f"{arithmetic.share(trainable, full):.2f}" == "1.00"
    with pytest.raises(ValueError):
        arithmetic.share(1, 0)


def test_grouping_by_projection_counts_pairs() -> None:
    """Each projection's pairs are counted across layers; sizes must agree."""
    rows = arithmetic.by_key(lfm_pairs(), SHAPES)
    counts = {row.key: row.pairs for row in rows}
    assert counts["self_attn.q_proj"] == 3
    assert counts["conv.in_proj"] == 9
    assert counts["feed_forward.w2"] == 12
    assert sum(row.numbers() for row in rows) == 9_781_248
    odd = [AdapterPair(0, "x", 4, 4, 2), AdapterPair(1, "x", 4, 8, 2)]
    with pytest.raises(ValueError, match="different sizes"):
        arithmetic.by_key(odd, ["x"])


def test_layers_of_one_kind_must_agree() -> None:
    """Two convolution layers with different pair counts are refused."""
    layers = [
        arithmetic.LayerAdapters(0, arithmetic.CONVOLUTION, ["a"], 1),
        arithmetic.LayerAdapters(1, arithmetic.CONVOLUTION, ["a", "b"], 2),
    ]
    with pytest.raises(ValueError, match="different pair counts"):
        arithmetic.kind_counts(layers)


def test_short_keys_group_by_block() -> None:
    """Projection names are shortened and grouped by the block they are in."""
    assert (
        arithmetic.short_keys(ATTENTION_KEYS)
        == "self_attn q/k/v/out, feed_forward w1/w2/w3"
    )
    assert arithmetic.short_keys(CONV_KEYS) == "conv in/out, feed_forward w1/w2/w3"


def test_training_bytes_are_16_per_number() -> None:
    """A float32 number, its gradient and Adam's two averages: 16 bytes."""
    assert arithmetic.training_bytes(9_781_248) == 156_499_968
    assert lfm_files.gigabytes(arithmetic.training_bytes(MODEL_NUMBERS)) == "43.16 GB"


def test_adams_steady_updates_by_formula() -> None:
    """3.162 at step 1, 6.379 at step 8, the largest 6.569 at step 12; 1 corrected."""
    ratios = [
        arithmetic.adam_ratio(step, 0.9, 0.999, bias_correction=False)
        for step in range(1, 1_001)
    ]
    assert f"{ratios[0]:.3f}" == "3.162"
    assert f"{ratios[7]:.3f}" == "6.379"
    step, value = arithmetic.largest(ratios)
    assert (step, f"{value:.3f}") == (12, "6.569")
    assert arithmetic.adam_ratio(5, 0.9, 0.999, bias_correction=True) == 1.0
    with pytest.raises(ValueError):
        arithmetic.adam_ratio(0, 0.9, 0.999, bias_correction=False)
    with pytest.raises(ValueError):
        arithmetic.largest([])


def test_chances_are_written_readably() -> None:
    """Percent from 1% up, then 1 in N, millions, then powers of ten."""
    assert arithmetic.chance_text(math.log(0.642)) == "64.2%"
    assert arithmetic.chance_text(math.log(1 / 479)) == "1 in 479"
    assert arithmetic.chance_text(-16.7149) == "1 in 18.2 million"
    assert arithmetic.chance_text(-105.6188) == "1 in 10^45.9"


def test_checkpoints_every_five_steps_and_the_last() -> None:
    """Every multiple of the interval, plus the last step if it is not one."""
    assert arithmetic.checkpoint_steps(20, 5) == [5, 10, 15, 20]
    assert arithmetic.checkpoint_steps(7, 5) == [5, 7]
    assert arithmetic.checkpoint_steps(3, 5) == [3]
    with pytest.raises(ValueError):
        arithmetic.checkpoint_steps(0, 5)


# ----------------------------------------------------------------- report.py


SHAPE = LoraShape(rank=16, scale=2.0, layers=12)
DROPOUT, SEED = 0.0, 0  # the project's


def test_setup_names_the_projects_settings(capsys: pytest.CaptureFixture) -> None:
    """The project's values print plainly; a flag that changes one says so."""
    report.print_setup(Path("/m"), "ours", SHAPE, DROPOUT, SEED, 5e-5, 20)
    plain = " ".join(capsys.readouterr().out.split())
    assert "rank 16, scale 2.0, dropout 0.0, seed 0, on the last 12 layers" in plain
    assert "otherwise" not in plain
    changed = LoraShape(16, 20.0, 12)
    report.print_setup(Path("/m"), "ours", changed, DROPOUT, SEED, 1e-4, 20)
    plain = " ".join(capsys.readouterr().out.split())
    assert "scale 20.0 (--scale; the project's is 2.0)" in plain
    assert "(--lr; the project's SFT uses 5e-05)" in plain
    assert "otherwise the project's settings" in plain


def test_adapters_are_listed_layer_by_layer(capsys: pytest.CaptureFixture) -> None:
    """Every adapted layer gets a line, then every projection and the total."""
    pairs = lfm_pairs()
    layers = arithmetic.by_layer(pairs, lfm_kinds())
    report.print_adapters(layers, arithmetic.by_key(pairs, SHAPES), LAYERS)
    report.print_trainable(layers, pairs, 9_781_248, MODEL_NUMBERS)
    out = capsys.readouterr().out
    assert "    21  attention         7    827,392  self_attn q/k/v/out" in out
    assert "layers 0 to 17 of 30: no adapters" in out
    assert "  total" in out and "66    9,781,248" in out
    plain = " ".join(out.split())
    assert "66 pairs: 3 attention layers x 7 + 9 convolution layers x 5" in plain
    assert "(0.363%, 0.4% to one decimal)" in plain


def test_steps_and_checkpoints_print(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """The step table, one line per file, and the reload lines."""
    scored = [
        SimpleNamespace(step=0, total=-105.6188, ranked_first=36, tokens=50),
        SimpleNamespace(step=1, total=-0.4427, ranked_first=50, tokens=50),
    ]
    report.print_steps(scored)
    path = write_safetensors(tmp_path / "adapter.safetensors", {"b": ("F32", [2, 2])})
    headers = {path: lfm_files.read_safetensors_header(path)}
    report.print_checkpoints(headers, 3, 4)
    middle = (path, SimpleNamespace(step=1, total=-0.4427))
    report.print_reload(0.0, path, 0.0, middle, scored)
    out = capsys.readouterr().out
    assert "     0        -105.6188    -2.1124      36 of 50  1 in 10^45.9" in out
    assert "50 of 50  64.2%" in out
    assert "(the last step's, again)" in out
    plain = " ".join(out.split())
    assert "removed first: 3 adapter files from an earlier run" in plain
    assert "after step 1 it was -0.4427" in plain


def test_a_reply_is_shown_whole_or_by_its_ends(capsys: pytest.CaptureFixture) -> None:
    """A reply equal to the target prints whole; a long one prints its two ends."""
    decode = "".join
    target = list("SELECT 1")
    report.print_reply(
        "after", SimpleNamespace(tokens=target, finish_reason="stop"), target, decode
    )
    long = list("x " * 100 + "y " * 100)
    report.print_reply(
        "before", SimpleNamespace(tokens=long, finish_reason="length"), target, decode
    )
    out = capsys.readouterr().out
    assert "exactly the target: yes" in out and "'SELECT 1'" in out
    plain = " ".join(out.split())
    assert "begins 'x x x" in plain and "ends 'y y y" in plain
    assert "finish_reason 'length'" in out


def test_merge_compares_both_ways(capsys: pytest.CaptureFixture) -> None:
    """The adapter, the 16-bit merge and the 4-bit merge, and why 4 bits lose."""
    target = list("ab")
    adapter = SimpleNamespace(total=-0.44, tokens=2)
    sixteen = SimpleNamespace(
        scored=SimpleNamespace(total=-0.46),
        reply=SimpleNamespace(tokens=target),
        size=1_950_000_000,
    )
    four = SimpleNamespace(
        scored=SimpleNamespace(total=-47.97),
        reply=SimpleNamespace(tokens=list("The user")),
        size=550_000_000,
    )
    change = WeightChange(mean=0.00005, largest=0.0004, step=0.005)
    report.print_merge(adapter, sixteen, four, 550_000_000, change, target, "".join)
    plain = " ".join(capsys.readouterr().out.split())
    sixteen_line = "merged into 16-bit copies: target log-prob -0.4600; greedy reply"
    assert f"{sixteen_line} exactly the target: yes" in plain
    four_line = "merged back into 4 bits: target log-prob -47.9700; greedy reply"
    assert f"{four_line} exactly the target: no" in plain
    assert "it begins 'The user'" in plain
    assert "100 times the average change" in plain


def test_adam_sections_print(capsys: pytest.CaptureFixture) -> None:
    """Defaults, the steady table, the attempts, the decay and the scale."""
    optimizer = SimpleNamespace(
        betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, bias_correction=False
    )
    report.print_defaults(optimizer)
    without = [arithmetic.adam_ratio(s, 0.9, 0.999, False) for s in range(1, 13)]
    steady = SimpleNamespace(
        without=without, with_correction=[1.0] * 12, same_for_sizes=True
    )
    report.print_steady(steady, without, (1, 8, 12), (1e-5, (0.9, 0.999), (1.0, 0.001)))
    attempts = [
        SimpleNamespace(
            name="attempt 1",
            learning_rate=4e-5,
            bias_correction=False,
            updates=[4e-5 * 3.162],
        ),
        SimpleNamespace(
            name="attempt 2", learning_rate=1e-5, bias_correction=True, updates=[1e-5]
        ),
    ]
    report.print_attempts(attempts)
    decays = [SimpleNamespace(learning_rate=1e-5, values=[0.99999988, 0.99988079])]
    report.print_decay(decays, (1, 1_000), 0.01)
    scales = [
        SimpleNamespace(scale=2.0, moved=[0.001, 0.04]),
        SimpleNamespace(scale=20.0, moved=[0.01, 0.4]),
    ]
    report.print_scale(scales, (1, 20), 1e-8, 2_048, 16, 5e-5)
    out = capsys.readouterr().out
    plain = " ".join(out.split())
    assert "weight_decay 0.01, bias_correction False" in plain
    assert (
        "without correction the largest update is 6.569 learning rates, at step 12"
        in plain
    )
    assert "attempt 1: lr 4e-05, no correction" in out and "12.6" in out
    assert "one step gives 0.99999988" in plain
    assert "takes off at most 0.012% in 1,000 steps" in plain
    assert "ratio                   10.00                   10.00" in out
    assert "alpha 32 at rank 16" in plain


def test_a_header_that_is_not_json_or_too_long_is_refused(tmp_path: Path) -> None:
    """A header that is not JSON, or announces over 100 MB, stops with a reason."""
    garbled = tmp_path / "garbled.safetensors"
    garbled.write_bytes(struct.pack("<Q", 3) + b"{x}")
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        lfm_files.read_safetensors_header(garbled)
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 2**60) + b"{}")
    with pytest.raises(ValueError, match="announces"):
        lfm_files.read_safetensors_header(huge)


def test_a_chance_too_small_for_floats_still_prints() -> None:
    """e**-5000 is 0.0 in floating point; the power of ten is still written."""
    assert arithmetic.chance_text(-5_000.0) == "1 in 10^2171.5"


def test_merging_an_untrained_adapter_says_so(capsys: pytest.CaptureFixture) -> None:
    """With no change to merge, the report says so instead of dividing by zero."""
    adapter = SimpleNamespace(total=-105.6, tokens=2)
    merged = SimpleNamespace(
        scored=SimpleNamespace(total=-105.6), reply=SimpleNamespace(tokens=[1]), size=1
    )
    change = WeightChange(mean=0.0, largest=0.0, step=0.005)
    report.print_merge(adapter, merged, merged, 1, change, [1], str)
    assert "there is nothing to merge" in capsys.readouterr().out
