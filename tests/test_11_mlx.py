"""Chapter 11 with MLX: the quantization rules, convert, compare and ask.

No test here loads the real model, downloads anything or uses the network.
The rules are checked against ``mx.quantize`` itself on small random
matrices and on awkward groups written by hand. convert.py runs mlx-lm's real
``convert`` on a tiny LFM2 built with mlx-lm's own ``lfm2`` code (64 numbers
wide, two layers) and a tiny tokenizer made here. compare.py and ask.py run
on that tiny model, or on fakes of ``stream_generate`` and the tokenizer.
MLX installs only on Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402
from mlx_lm.utils import quantize_model, save_model  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

CHAPTER = use_chapter("11-quantize")

import ask  # noqa: E402
import by_hand  # noqa: E402
import compare  # noqa: E402
import convert  # noqa: E402
import lfm  # noqa: E402
import lfm_files  # noqa: E402
import matrix_source  # noqa: E402
import report  # noqa: E402
import sql_questions  # noqa: E402
from sql_questions import Graded  # noqa: E402

HIDDEN = 64  # one group of 64 per row: the smallest width that quantizes
VOCAB = 64
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": HIDDEN,
    "block_multiple_of": 64,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "block_ff_dim": 2 * HIDDEN,
    "rope_theta": 1e7,
    "layer_types": ["conv", "full_attention"],
}
LIQUID_BLOCK = {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "model.embed_tokens": {"group_size": 64, "bits": 6, "mode": "affine"},
}


def run_main(module: ModuleType, *args: str) -> str:
    """Run a script's main() with arguments, and return what it printed.

    Args:
        module: The chapter module.
        *args: Command-line arguments.

    Returns:
        Everything printed.
    """
    saved = sys.argv
    sys.argv = [f"{module.__name__}.py", *args]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = saved
    return out.getvalue()


def tiny_model(seed: int = 0) -> nn.Module:
    """Build a tiny LFM2 with mlx-lm's own code, every weight in bfloat16.

    Args:
        seed: The random seed for the weights.

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


def write_tokenizer(folder: Path) -> None:
    """Write a tiny word-level tokenizer that mlx-lm can load.

    Args:
        folder: The model folder.
    """
    vocab = {f"t{i}": i for i in range(VOCAB - 3)}
    vocab |= {"<unk>": VOCAB - 3, "<s>": VOCAB - 2, "</s>": VOCAB - 1}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(folder / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    (folder / "tokenizer_config.json").write_text(json.dumps(config))


def write_bf16_model(folder: Path) -> Path:
    """Save the tiny model as a bfloat16 "download": weights, config, tokenizer.

    Args:
        folder: Where to write it.

    Returns:
        The folder.
    """
    save_model(folder, tiny_model())
    (folder / "config.json").write_text(
        json.dumps({**TINY_CONFIG, "dtype": "bfloat16"})
    )
    write_tokenizer(folder)
    return folder


def random_matrix(rows: int, columns: int, seed: int = 0) -> np.ndarray:
    """Make a matrix of bfloat16 numbers shaped like real weights.

    Args:
        rows: Rows.
        columns: Columns (a multiple of 64).
        seed: The random seed.

    Returns:
        float32 numbers that are exact bfloat16 values.
    """
    rng = np.random.default_rng(seed)
    values = (rng.standard_normal((rows, columns)) * 0.02).astype(np.float32)
    return by_hand.round_to_bfloat16(values)


def matrix_bytes(values: np.ndarray) -> bytes:
    """Write bfloat16 values as the bytes a safetensors file holds.

    Args:
        values: float32 numbers that are exact bfloat16 values.

    Returns:
        Two bytes per number, little-endian.
    """
    return (values.view(np.uint32) >> 16).astype("<u2").tobytes()


# ---------------------------------------------------------------- by_hand.py


def test_bfloat16_bytes_are_read_exactly() -> None:
    """Chapter 10's weight, 1 and -2 come back as the same numbers."""
    data = bytes.fromhex("f13c803f00c0")  # 0x3CF1, 0x3F80, 0xC000, little-endian
    values = by_hand.bfloat16_to_float32(data, (1, 3))
    assert values.tolist() == [[0.0294189453125, 1.0, -2.0]]


def test_rounding_to_bfloat16_matches_mlx() -> None:
    """Round to nearest, ties to even: the same numbers MLX's astype gives."""
    rng = np.random.default_rng(1)
    values = rng.standard_normal(10_000).astype(np.float32)
    theirs = np.array(mx.array(values).astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(by_hand.round_to_bfloat16(values), theirs)


def test_halves_round_away_from_zero() -> None:
    """2.5 -> 3 and -2.5 -> -3, as MLX's GPU code rounds (numpy would say 2)."""
    values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 2.4], np.float32)
    assert by_hand.round_half_away(values).tolist() == [-3, -2, -1, 1, 2, 3, 2]
    below = np.nextafter(np.float32(0.5), np.float32(0))  # 0.49999997
    under = np.nextafter(np.float32(1.5), np.float32(0))  # 1.4999999
    near = np.array([below, -below, under, -under], np.float32)
    assert by_hand.round_half_away(near).tolist() == [0, 0, 1, -1]


def test_the_naive_rule_by_hand() -> None:
    """offset = min, scale = (max - min) / levels, codes rounded and capped."""
    group = np.array([[-1.0, 0.0, 0.5, 2.0]], np.float32)
    result = by_hand.naive_quantize(group, bits=2)  # 3 steps: codes 0 to 3
    assert result.offsets.tolist() == [-1.0] and result.scales.tolist() == [1.0]
    assert result.codes.tolist() == [[0, 1, 2, 3]]  # 1.5 rounds up to 2
    assert result.rebuilt().tolist() == [[-1.0, 0.0, 1.0, 2.0]]


def test_mlx_rule_anchors_far_end_and_stores_zero_exactly() -> None:
    """The farther end is the offset; 0 lands exactly on a code."""
    group = np.array([[-0.7, -0.1, 0.0, 0.3]], np.float32)
    result = by_hand.mlx_quantize(group, bits=4)
    assert result.offsets[0] == np.float32(-0.7) and result.scales[0] > 0
    zero = group[0, 2]
    code = result.codes[0, 2]
    assert result.offsets[0] + code * result.scales[0] == pytest.approx(zero, abs=1e-7)
    flipped = by_hand.mlx_quantize(-group, bits=4)
    assert flipped.offsets[0] == np.float32(0.7) and flipped.scales[0] < 0


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
def test_mlx_rule_by_hand_is_mx_quantize(bits: int) -> None:
    """Codes, scales, offsets and rebuilt numbers all equal MLX's own."""
    matrix = random_matrix(16, 256, seed=bits)
    result = by_hand.measure_bits(matrix.reshape(-1, 64), matrix, bits)
    assert result.matches_mlx(), result
    assert result.naive_codes_differ > 0  # the naive rule is not MLX's


AWKWARD_GROUPS = [
    np.linspace(-0.3, -0.1, 64),  # all negative: MLX stretches the range to 0
    np.linspace(0.1, 0.3, 64),  # all positive
    np.full(64, 0.5),  # all equal: the scale would be 0 without its floor
    np.zeros(64),  # all zero
    np.concatenate([np.zeros(32), np.linspace(-0.02, 0.05, 32)]),  # many exact zeros
    np.concatenate([np.full(63, 0.001), [0.9]]),  # one big number
]


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_awkward_groups_match_mx_quantize(bits: int) -> None:
    """Edge cases, the all-negative group included, follow MLX exactly."""
    matrix = by_hand.round_to_bfloat16(np.stack(AWKWARD_GROUPS).astype(np.float32))
    result = by_hand.measure_bits(matrix, matrix, bits)
    assert result.matches_mlx()
    assert result.one_signed == 4  # negative, positive, equal, one big number
    assert result.expected_bytes == result.stored_bytes


def test_the_running_max_from_zero_is_mlxs_own() -> None:
    """MLX stretches an all-negative group to 0: its step is 0.3 / 15, not 0.2 / 15."""
    group = by_hand.round_to_bfloat16(np.linspace(-0.3, -0.1, 64, dtype=np.float32))
    matrix = group[None, :]
    theirs, _, _ = by_hand.mx_quantize(matrix, 4)
    span = float(theirs.scales[0]) * 15
    assert span == pytest.approx(0.3, rel=0.01)  # from the min up to 0
    assert np.array_equal(by_hand.mlx_quantize(matrix, 4).codes, theirs.codes)
    assert not np.array_equal(by_hand.naive_quantize(matrix, 4).codes, theirs.codes)


def test_unpacking_a_hand_packed_word() -> None:
    """Eight 4-bit codes in one word: the first code in the lowest bits."""
    word = np.array([[0x76543210]], np.uint32)
    assert by_hand.unpack_codes(word, 4, 8).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    three = np.array([[0b111_110_101_100_011_010_001_000]], np.uint32)  # 8 x 3 bits
    assert by_hand.unpack_codes(three, 3, 8).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_error_size() -> None:
    """Mean, largest and relative error, by hand."""
    error = by_hand.error_size(
        np.array([1.0, -1.0], np.float32), np.array([1.5, -1.0], np.float32)
    )
    assert error.mean == 0.25 and error.largest == 0.5
    assert error.relative == pytest.approx(math.sqrt(0.125))


def test_storage_per_number() -> None:
    """4 bits plus a 16-bit scale and offset per 64 numbers is 4.5 bits."""
    assert by_hand.bits_per_number(4) == 4.5
    assert by_hand.bits_per_number(6) == 6.5
    assert by_hand.levels(4) == 15


def tiny_matrix_source(rows: int = 8, columns: int = 128) -> matrix_source.Matrix:
    """A stand-in for the real matrix, in its byte form.

    Args:
        rows: Rows.
        columns: Columns.

    Returns:
        The matrix, as load_matrix returns one.
    """
    values = random_matrix(rows, columns)
    pin = matrix_source.MatrixPin(
        "tiny.w1.weight", "tiny.safetensors", "BF16", (rows, columns), ""
    )
    return matrix_source.Matrix(pin, matrix_bytes(values), "a test")


def test_by_hand_prints_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four sections and the final check print on a small matrix."""
    calls: list[bool] = []

    def fake_load(fast: bool) -> matrix_source.Matrix:
        calls.append(fast)
        return tiny_matrix_source()

    monkeypatch.setattr(by_hand, "load_matrix", fake_load)
    out = run_main(by_hand, "--fast")
    assert calls == [True]
    for heading in (
        "1. one group of 64 numbers, the naive rule",
        "2. the same group, MLX's rule",
        "3. the whole matrix at 4 bits: 16 groups of 64",
        "4. MLX's rule at 3, 4, 6 and 8 bits",
        "checked: MLX's rule by hand gives exactly mx.quantize's codes",
    ):
        assert heading in out
    assert "MLX's rule, by hand   0 differ" in out


def test_by_hand_stops_if_its_rule_drifts_from_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hand rule that no longer matches mx.quantize stops the script."""
    real = by_hand.mlx_quantize

    def drifted(groups: np.ndarray, bits: int) -> by_hand.Quantized:
        result = real(groups, bits)
        codes = result.codes.copy()
        codes[0, 0] ^= 1
        return by_hand.Quantized(bits, codes, result.scales, result.offsets)

    monkeypatch.setattr(by_hand, "load_matrix", lambda fast: tiny_matrix_source())
    monkeypatch.setattr(by_hand, "mlx_quantize", drifted)
    with pytest.raises(SystemExit) as stopped:
        run_main(by_hand)
    assert "no longer matches mx.quantize" in str(stopped.value)


def test_a_missing_matrix_stops_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch that fails ends the script with the message, not a traceback."""

    def failing(fast: bool) -> matrix_source.Matrix:
        raise matrix_source.MatrixError("could not fetch: no network")

    monkeypatch.setattr(by_hand, "load_matrix", failing)
    with pytest.raises(SystemExit, match="no network"):
        run_main(by_hand, "--fast")


# ---------------------------------------------------------------- convert.py


def test_the_predicate_keeps_only_the_embedding_at_6_bits() -> None:
    """Two arguments; the embedding gets its own settings, the rest True."""
    assert convert.keep_embedding_at_6("model.embed_tokens", None) == {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }
    assert convert.keep_embedding_at_6("model.layers.0.feed_forward.w1", None) is True


def test_mlx_lm_calls_the_predicate_with_two_arguments() -> None:
    """A three-argument predicate, as convert's type hint suggests, fails."""

    def three(path: str, module: nn.Module, config: dict[str, Any]) -> bool:
        return True

    with pytest.raises(TypeError):
        quantize_model(tiny_model(), {}, 64, 4, quant_predicate=three)


def test_our_predicate_writes_liquids_quantization_block() -> None:
    """The embedding is packed at 6 bits, every linear layer at 4."""
    model, config = quantize_model(
        tiny_model(), {}, 64, 4, quant_predicate=convert.keep_embedding_at_6
    )
    assert config["quantization"] == LIQUID_BLOCK
    assert model.model.embed_tokens.bits == 6
    bits = {
        name: module.bits
        for name, module in model.named_modules()
        if hasattr(module, "bits") and name != "model.embed_tokens"
    }
    assert bits and set(bits.values()) == {4}


def test_the_builtin_recipe_stops_on_lfm2(tmp_path: Path) -> None:
    """mixed_4_6 finds no down_proj in LFM2 and raises mlx-lm's own error."""
    check = convert.check_builtin_recipe(write_bf16_model(tmp_path / "bf16"))
    assert check.error == "Model does not have expected keys for mixed quant."
    assert check.named_like_recipe == 0 and not check.has_head
    # the embedding, w1, w2 and w3 in both layers, conv in/out, attention q/k/v/out
    assert check.quantizable == 1 + 2 * 3 + 2 + 4


def test_convert_builds_ours_and_counts_its_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """mlx-lm's real convert on the tiny model: Liquid's block, sizes that add up."""
    source = write_bf16_model(tmp_path / "bf16")
    monkeypatch.setattr(convert, "require_repo", lambda repo: source)
    out = run_main(convert)
    ours = lfm_files.ours_folder()
    assert lfm_files.build_ready(ours)
    assert not ours.with_name(ours.name + ".partial").exists()
    assert convert.quantization_block(ours) == LIQUID_BLOCK
    assert "it stops: ValueError: Model does not have expected keys" in out
    assert "[INFO] Quantizing" in out
    assert "the file's size exactly" in out
    storage = convert.count_storage(ours)
    by_kind = {row.kind: row for row in storage.rows}
    assert by_kind["4-bit layers"].bits_each() == 4.5
    assert by_kind["6-bit layers"].bits_each() == 6.5
    assert by_kind["6-bit layers"].numbers == VOCAB * HIDDEN
    assert by_kind["kept in bfloat16"].bits_each() == 16
    assert storage.data_bytes() + storage.header_bytes == storage.file_bytes
    again = run_main(convert)
    assert "already built" in again and "[INFO]" not in again


def test_the_all_4_bit_variant_goes_to_the_chapter_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--all-4bit packs the embedding at 4 bits too, outside the shared folder."""
    source = write_bf16_model(tmp_path / "bf16")
    monkeypatch.setattr(convert, "require_repo", lambda repo: source)
    out = run_main(convert, "--all-4bit")
    folder = lfm_files.all_four_bit_folder()
    assert convert.quantization_block(folder) == {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }
    assert "mixed_4_6" not in out  # the recipe check is only for ours
    kinds = [row.kind for row in convert.count_storage(folder).rows]
    assert kinds == ["4-bit layers", "kept in bfloat16"]
    assert not lfm_files.ours_folder().exists()


def test_a_broken_off_build_never_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure mid-build leaves only the .partial folder, cleaned next time."""
    source = write_bf16_model(tmp_path / "bf16")

    def broken(*args: object, **kwargs: object) -> None:
        Path(str(args[1])).mkdir(parents=True)
        raise RuntimeError("disk full")

    plan = convert.plan_for(all_four_bit=False)
    with monkeypatch.context() as patch:
        patch.setattr(convert, "convert", broken)
        with pytest.raises(RuntimeError):
            convert.build(plan, source)
    assert not plan.folder.exists()
    convert.build(plan, source)
    assert lfm_files.build_ready(plan.folder)
    assert not plan.folder.with_name(plan.folder.name + ".partial").exists()


# ---------------------------------------------------------------- lfm.py


def test_parameter_count_unpacks_quantized_layers() -> None:
    """The same count before and after packing; the bytes shrink."""
    model = tiny_model()
    before, size = lfm.parameter_count(model), lfm.weight_bytes(model)
    quantize_model(model, {}, 64, 4)
    assert lfm.parameter_count(model) == before
    assert lfm.weight_bytes(model) < size / 3


class FakeTokenizer:
    """Stands in for mlx-lm's tokenizer: renders chats and encodes text."""

    bos_token_id = 1

    def __init__(self, prefix: str = "") -> None:
        """Start with no recorded calls.

        Args:
            prefix: Text put in front of every rendered chat.
        """
        self.prefix = prefix
        self.options: list[dict[str, Any]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> str:
        """Render a chat simply, and record the options.

        Args:
            messages: The chat.
            **options: The template options.

        Returns:
            The rendered text.
        """
        self.options.append(options)
        return self.prefix + "|".join(m["content"] for m in messages) + "|<think>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn each character into an id below the tiny vocabulary's size.

        Args:
            text: The text.
            add_special_tokens: Ignored.

        Returns:
            The ids.
        """
        return [2 + ord(char) % (VOCAB - 2) for char in text]


def test_chat_prompt_renders_as_the_project_engine_does() -> None:
    """Generation prompt on, reasoning kept, no extra tokens."""
    tokenizer = FakeTokenizer()
    ids = lfm.chat_prompt(tokenizer, "hi")
    assert ids == tokenizer.encode("hi|<think>")
    assert tokenizer.options == [
        {"add_generation_prompt": True, "tokenize": False, "preserve_thinking": True}
    ]


# ---------------------------------------------------------------- compare.py


@pytest.mark.parametrize(
    ("gained", "lost", "expected"),
    [(0, 0, 1.0), (1, 0, 1.0), (0, 6, 0.03125), (6, 0, 0.03125), (7, 5, 0.7744140625)],
)
def test_exact_p(gained: int, lost: int, expected: float) -> None:
    """McNemar's exact test; 7 gained vs 5 lost is the plan's p = 0.77."""
    assert compare.exact_p(gained, lost) == pytest.approx(expected)


def test_json_and_text_differences_in_words() -> None:
    """Keys only one side has, changed values, and the quantization block."""
    note = compare.json_difference(
        {"a": 1, "quantization": {"bits": 4}, "c": 1},
        {"a": 2, "quantization": {"bits": 4}, "b": 1},
    )
    assert note == (
        "quantization block identical; only Liquid's has b; only ours has c; "
        "values differ for a"
    )
    assert "DIFFERS" in compare.json_difference(
        {"quantization": 1}, {"quantization": 2}
    )
    text = compare.text_difference("x\ny", "# hint\nx\ny")
    assert text == "Liquid's has 1 more line(s): '# hint'"


def test_files_are_compared_byte_for_byte(tmp_path: Path) -> None:
    """Identical, JSON that differs, text that differs, a card, a missing file."""
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    for folder in (ours, theirs):
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"same")
    (ours / "config.json").write_text('{"a": 1}')
    (theirs / "config.json").write_text('{"a": 1, "b": 2}')
    (ours / "chat_template.jinja").write_text("x")
    (theirs / "chat_template.jinja").write_text("# hint\nx")
    (ours / "README.md").write_text("a\nb")
    (theirs / "README.md").write_text("c")
    (ours / "tokenizer.json").write_text("{}")
    checks = {
        name: compare.compare_file(name, ours, theirs)
        for name in compare.COMPARED_FILES
    }
    assert checks["model.safetensors"].identical
    assert checks["config.json"].note == "only Liquid's has b"
    assert "# hint" in checks["chat_template.jinja"].note
    assert checks["README.md"].note.startswith("different model cards (2 and 1 lines)")
    assert checks["tokenizer.json"].note == "Liquid's has no tokenizer.json"
    assert checks["generation_config.json"].ours_bytes is None


def test_weights_are_compared_tensor_by_tensor(tmp_path: Path) -> None:
    """One changed tensor and one extra tensor are named; the rest match."""
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    first = {"a": mx.zeros((2, 2)), "b": mx.ones((2,)), "c": mx.ones((1,))}
    second = {"a": mx.zeros((2, 2)), "b": mx.full((2,), 2.0), "d": mx.ones((1,))}
    for folder, weights in ((ours, first), (theirs, second)):
        folder.mkdir()
        mx.save_safetensors(str(folder / "model.safetensors"), weights)
    check = compare.compare_weights(ours, theirs)
    assert check.tensors == 3 and check.identical == 1
    assert (
        check.differ == ["b"]
        and check.only_ours == ["c"]
        and check.only_theirs == ["d"]
    )
    assert check.ours_md5 != check.theirs_md5


def test_the_logprob_of_a_text_is_read_off_one_pass() -> None:
    """The summed log-probability equals a log-softmax computed by hand."""
    model = tiny_model()
    tokenizer = FakeTokenizer()
    loaded = lfm.LoadedModel(model, tokenizer, Path("."), 0, 0, 0, 0.0)
    total, count = compare.text_logprob(loaded, "SELECT 1")
    ids = [1, *tokenizer.encode("SELECT 1")]
    logits = np.array(model(mx.array([ids[:-1]]))[0].astype(mx.float32), np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    logprobs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    expected = sum(logprobs[i, token] for i, token in enumerate(ids[1:]))
    assert count == len(ids) - 1
    assert total == pytest.approx(expected, abs=1e-3)


def test_forced_choices_are_the_top_tokens_of_one_pass() -> None:
    """Each position's argmax over the prompt plus the given reply."""
    model = tiny_model()
    loaded = lfm.LoadedModel(model, FakeTokenizer(), Path("."), 0, 0, 0, 0.0)
    prompt, reply = [1, 5, 9], [3, 7, 11, 2]
    choices = compare.forced_choices(loaded, prompt, reply)
    logits = model(mx.array([prompt + reply[:-1]]))[0]
    assert choices == mx.argmax(logits[2:], axis=-1).tolist()
    assert compare.forced_choices(loaded, prompt, []) == []


@dataclass
class FakeStep:
    """What a fake stream_generate yields: mlx-lm's GenerationResponse fields."""

    text: str
    token: int
    finish_reason: str | None = None
    prompt_tokens: int = 4
    prompt_tps: float = 400.0
    generation_tokens: int = 0
    generation_tps: float = 50.0


def fake_stream(
    replies: dict[int, list[int]], stop: bool = True
) -> Callable[..., Iterator[FakeStep]]:
    """Make a fake stream_generate that replies by the prompt's length.

    Args:
        replies: The reply (token ids) for each prompt length.
        stop: True to end each reply with an end of turn.

    Returns:
        A function with stream_generate's signature.
    """

    def stream(
        model: Any, tokenizer: Any, ids: list[int], max_tokens: int
    ) -> Iterator[FakeStep]:
        reply = replies.get(len(ids), [5, 6, 7])[:max_tokens]
        for count, token in enumerate(reply, 1):
            finish = "length" if count == max_tokens and not stop else None
            yield FakeStep(f"<{token}>", token, finish, len(ids), 400.0, count)
            if finish:
                return
        if stop and len(reply) < max_tokens:
            yield FakeStep("", 0, "stop", len(ids), 400.0, len(reply) + 1)

    return stream


def test_generate_drops_the_end_of_turn_and_times_with_mlx_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reply's tokens and text; seconds from mlx-lm's tokens per second."""
    monkeypatch.setattr(compare, "stream_generate", fake_stream({4: [8, 9]}))
    loaded = lfm.LoadedModel(None, None, Path("."), 0, 0, 0, 0.0)
    reply = compare.generate(loaded, [1, 2, 3, 4], 10)
    assert reply.tokens == [8, 9] and reply.text == "<8><9>"
    assert reply.finish_reason == "stop"
    assert reply.prompt_seconds == pytest.approx(4 / 400)
    assert reply.reply_seconds == pytest.approx(3 / 50)
    monkeypatch.setattr(
        compare, "stream_generate", fake_stream({4: [8, 9]}, stop=False)
    )
    capped = compare.generate(loaded, [1, 2, 3, 4], 2)
    assert capped.tokens == [8, 9] and capped.finish_reason == "length"


def result_for(
    key: str, correct: list[bool], greedy: list[list[int]]
) -> compare.BuildResult:
    """Make a build result with only what the tables read.

    Args:
        key: The build's key.
        correct: Each pass-rate question's verdict.
        greedy: The agreement replies (also used as forced choices).

    Returns:
        The result.
    """
    build = compare.Build(key, key, Path("/models") / key, "make it")
    answers = [
        compare.Answer(f"t{i}", 3, "stop", Graded(ok, "S", "1", "r"), [i, int(ok)])
        for i, ok in enumerate(correct)
    ]
    return compare.BuildResult(
        build=build,
        disk_bytes=1_580_000_000,
        loaded_bytes=1_580_000_000,
        weight_bytes=1_580_000_000,
        load_seconds=1.0,
        peak_bytes=1_700_000_000,
        logprob_total=-10.0,
        logprob_tokens=20,
        greedy=greedy,
        forced=greedy,
        answers=answers,
        prompt_seconds=2.0,
        prompt_tokens=1000,
        reply_seconds=10.0,
        reply_tokens=500,
    )


def test_the_tables_read_agreement_and_gained_vs_lost() -> None:
    """Speeds, log-probability, agreement against bfloat16, task by task."""
    bf16 = result_for("bf16", [True, True, False], [[1, 2], [3, 4]])
    ours = result_for("ours", [True, False, True], [[1, 2], [3, 9]])
    liquid = result_for("liquid", [True, False, True], [[1, 2], [3, 9]])
    assert ours.prompt_speed() == 500 and ours.reply_speed() == 50
    assert ours.logprob_per_token() == -0.5 and ours.correct() == 2
    assert report.agreement(ours, bf16) == (1, 3, 4)
    picked = sql_questions.pass_rate_tasks()[:3]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report.print_size_and_speed([bf16, ours, liquid])
        report.print_quality([bf16, ours, liquid], 128)
        report.print_task_by_task([bf16, ours, liquid], picked, compare.exact_p)
        report.print_ours_vs_liquid([bf16, ours, liquid], same_file=True)
    text = out.getvalue()
    assert "1 of 2" in text and "75.0%" in text
    rows = {line.split()[0]: line.split()[1:] for line in text.splitlines() if line}
    assert rows["gained"] == ["1", "1"] and rows["lost"] == ["1", "1"]
    assert rows["exact"] == ["p", "1.000", "1.000"]
    assert "4. the 3 questions, task by task" in text
    assert "verdict: byte-identical weights and identical greedy replies" in text


def test_compare_fast_measures_only_liquids_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--fast: no file check, no agreement, one build, results saved."""
    measured: list[str] = []

    def fake_measure(
        build: compare.Build, reference: Any, say: Any
    ) -> compare.BuildResult:
        measured.append(build.key)
        assert reference is None
        return result_for(build.key, [True] * 24, [[1]] * 20)

    monkeypatch.setattr(compare, "is_ready", lambda build: True)
    monkeypatch.setattr(compare, "measure", fake_measure)
    out = run_main(compare, "--fast")
    assert measured == ["liquid"]
    assert "fast mode: Liquid's export alone" in out
    assert "1. ours against" not in out and "n/a" in out
    saved = json.loads((lfm_files.chapter_home() / "compare-fast.json").read_text())
    assert saved["builds"][0]["build"]["key"] == "liquid" and saved["complete"]


def test_compare_stops_when_a_build_is_missing() -> None:
    """Each missing build is named with the command that makes it."""
    with pytest.raises(SystemExit) as stopped:
        run_main(compare)
    message = str(stopped.value)
    assert "ours: run `uv run convert.py`" in message
    assert "all 4-bit: run `uv run convert.py --all-4bit`" in message
    assert "--fast" in message


def test_measure_runs_every_test_on_one_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log-probability, agreement replies, forced choices and graded answers."""
    model = tiny_model()
    tokenizer = FakeTokenizer()
    loaded = lfm.LoadedModel(model, tokenizer, Path("."), 10, 20, 30, 0.5)
    monkeypatch.setattr(compare, "load_folder", lambda folder: loaded)
    monkeypatch.setattr(compare, "stream_generate", fake_stream({}))
    monkeypatch.setattr(
        compare, "agreement_tasks", lambda: sql_questions.agreement_tasks()[:2]
    )
    monkeypatch.setattr(
        compare, "pass_rate_tasks", lambda: sql_questions.pass_rate_tasks()[:2]
    )
    build = compare.Build("bf16", "bfloat16", Path("."), "x")
    lines: list[str] = []
    result = compare.measure(build, None, lines.append)
    assert result.loaded_bytes == 20 and result.weight_bytes == 10
    assert len(result.greedy) == 2 and result.greedy[0] == [5, 6, 7]
    assert len(result.forced) == 2 and len(result.forced[0]) == 3
    assert [answer.task_id for answer in result.answers] == [
        "airline-easy-001",
        "airline-hard-001",
    ]
    assert all(
        answer.graded.reason == "no SQL in the reply" for answer in result.answers
    )
    assert result.logprob_tokens > 0 and result.prompt_tokens > 0
    assert lines and lines[0].startswith("bfloat16: loaded in 0.5 s")


def test_both_templates_render_the_same_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A template that only adds a comment renders identical tokens."""
    tokenizers = {
        "ours": FakeTokenizer(),
        "theirs": FakeTokenizer(),
        "odd": FakeTokenizer("x"),
    }
    monkeypatch.setattr(
        compare, "load_tokenizer_only", lambda folder: tokenizers[folder.name]
    )
    assert compare.same_prompts(Path("ours"), Path("theirs"), ["a", "b"]) == (2, 2)
    assert compare.same_prompts(Path("ours"), Path("odd"), ["a", "b"]) == (0, 2)


# -------------------------------------------------------------------- ask.py


def fake_clock(times: list[float]) -> Callable[[], float]:
    """Make a clock that returns the given times in order.

    Args:
        times: The times.

    Returns:
        The clock.
    """
    ticks = iter(times)
    return lambda: next(ticks)


def test_collect_times_each_token_and_drops_the_end_of_turn() -> None:
    """The end-of-turn step is timed but not part of the reply."""
    steps = [FakeStep("a", 1), FakeStep("b", 2), FakeStep("", 0, "stop")]
    shown: list[str] = []
    texts, seconds, finish, end = ask.collect(
        steps, shown.append, fake_clock([0.0, 0.5, 0.6, 0.65])
    )
    assert texts == ["a", "b"] and finish == "stop"
    assert seconds == pytest.approx([0.5, 0.1]) and end == pytest.approx(0.05)
    timed = ask.Timed(4, texts, seconds, finish, end, 1)
    assert timed.text() == "ab"
    assert timed.ms_per_token() == pytest.approx(75.0)


def test_before_picture_needs_the_same_question(course_home: Path) -> None:
    """Chapter 10's saved run is used only when it asked the same thing."""
    assert ask.read_before("q", course_home) is None
    folder = course_home / "10-meet-lfm"
    folder.mkdir(parents=True)
    path = folder / "first_reply-bfloat16.json"
    saved = {key: 1 for key in ask.BEFORE_KEYS} | {"question": "q", "reply": "r"}
    path.write_text(json.dumps(saved))
    assert ask.read_before("q", course_home) == saved
    assert ask.read_before("other", course_home) is None
    path.write_text("{broken")
    assert ask.read_before("q", course_home) is None


def test_same_query_ignores_spacing() -> None:
    """The same SQL, spaced differently, is the same query."""
    first = "</think>```sql\nSELECT COUNT(*)\nFROM t;\n```"
    second = "</think>```sql\nSELECT COUNT(*) FROM t;\n```"
    assert ask.same_query(first, second) == "SELECT COUNT(*) FROM t;"
    assert ask.same_query(first, "</think>```sql\nSELECT 1\n```") is None
    assert ask.same_query(first, "no query") is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")
def test_the_memory_budget_comes_from_metal() -> None:
    """All of the unified memory, and a smaller share for the GPU."""
    budget = ask.memory_budget()
    assert 0 < budget.gpu_limit_bytes < budget.total_bytes


def test_ask_prints_reply_before_after_and_budget(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    """The reply streams; chapter 10's run sits beside it; the budget follows."""
    question = "How many customers live in Paris? Answer with one SQL query."
    folder = course_home / "10-meet-lfm"
    folder.mkdir(parents=True)
    before = {
        "question": question,
        "reply": "</think>```sql\nSELECT 1;\n```",
        "weight_bytes": 5_394_397_184,
        "peak_memory_bytes": 5_446_727_832,
        "ms_per_token": 45.4,
        "reply_tokens": 286,
    }
    (folder / "first_reply-bfloat16.json").write_text(json.dumps(before))
    loaded = lfm.LoadedModel(
        None, FakeTokenizer(), course_home, 1_583_085_568, 0, 0, 0.1
    )
    monkeypatch.setattr(ask, "load_folder", lambda path: loaded)
    monkeypatch.setattr(ask, "model_for_later_chapters", lambda: (course_home, "ours"))
    words = ["</think>", "```sql\n", "SELECT 1;", "\n```"]

    def stream(
        model: Any, tokenizer: Any, ids: list[int], max_tokens: int
    ) -> Iterator[FakeStep]:
        for index, word in enumerate(words[:max_tokens]):
            yield FakeStep(word, index)
        if max_tokens > len(words):
            yield FakeStep("", 0, "stop")

    monkeypatch.setattr(ask, "stream_generate", stream)
    out = run_main(ask, question)
    assert "</think>```sql\nSELECT 1;\n```" in out
    assert "the reply: 4 tokens, then <|im_end|>" in out
    assert "both end with the same query: SELECT 1;" in out
    assert "5.39 GB" in out and "1.58 GB" in out
    assert "the real budget: unified memory" in out
    assert "chapter 10's bfloat16 run peaked at 5.45 GB" in out


def test_a_failed_forced_rebuild_keeps_the_old_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--force removes the old build only once the new one is complete."""
    source = write_bf16_model(tmp_path / "bf16")
    monkeypatch.setattr(convert, "require_repo", lambda repo: source)
    run_main(convert)
    ours = lfm_files.ours_folder()
    before = (ours / "model.safetensors").read_bytes()

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("out of memory")

    with monkeypatch.context() as patch:
        patch.setattr(convert, "convert", broken)
        with pytest.raises(RuntimeError):
            run_main(convert, "--force")
    assert lfm_files.build_ready(ours)
    assert (ours / "model.safetensors").read_bytes() == before


def test_measure_compares_against_the_reference_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a reference, top tokens are read along the reference's replies."""
    loaded = lfm.LoadedModel(tiny_model(), FakeTokenizer(), Path("."), 1, 2, 3, 0.1)
    monkeypatch.setattr(compare, "load_folder", lambda folder: loaded)
    monkeypatch.setattr(compare, "stream_generate", fake_stream({}))
    monkeypatch.setattr(
        compare, "agreement_tasks", lambda: sql_questions.agreement_tasks()[:2]
    )
    monkeypatch.setattr(compare, "pass_rate_tasks", lambda: [])
    reference = result_for("bf16", [], [[4, 5, 6, 7, 8], [9]])
    build = compare.Build("ours", "ours", Path("."), "x")
    result = compare.measure(build, reference, lambda line: None)
    assert [len(choices) for choices in result.forced] == [5, 1]
    assert result.greedy == [[5, 6, 7], [5, 6, 7]]


def test_compare_saves_after_every_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that fails part way keeps what it measured, marked incomplete."""

    def fake_measure(build: compare.Build, reference: Any, say: Any) -> Any:
        if build.label == "second":
            raise RuntimeError("the machine ran out of memory")
        return result_for(build.key, [True] * 3, [[1]] * 2)

    first = compare.Build("liquid", "first", Path("."), "x")
    second = compare.Build("liquid", "second", Path("."), "x")
    monkeypatch.setattr(compare, "all_builds", lambda: [first, second])
    monkeypatch.setattr(compare, "is_ready", lambda build: True)
    monkeypatch.setattr(compare, "measure", fake_measure)
    with pytest.raises(RuntimeError):
        run_main(compare, "--fast")
    path = lfm_files.chapter_home() / "compare-fast.json"
    saved = json.loads(path.read_text())
    assert saved["complete"] is False
    assert saved["builds_measured"] == 1 and saved["builds_planned"] == 2


def test_a_broken_json_file_is_reported_not_raised(tmp_path: Path) -> None:
    """A config.json that does not parse is a note, not a crash."""
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    for folder, text in ((ours, "{broken"), (theirs, "{}")):
        folder.mkdir()
        (folder / "config.json").write_text(text)
    check = compare.compare_file("config.json", ours, theirs)
    assert not check.identical and check.note == "one of the two is not a JSON object"


def test_zero_times_give_zero_speeds() -> None:
    """A build with no timing reads 0 tokens a second instead of crashing."""
    empty = result_for("bf16", [], [])
    empty.prompt_seconds = empty.reply_seconds = 0.0
    empty.logprob_tokens = 0
    assert empty.prompt_speed() == 0.0 and empty.reply_speed() == 0.0
    assert empty.logprob_per_token() == 0.0


def test_an_incomplete_before_picture_is_not_used(course_home: Path) -> None:
    """A saved run missing a field, or not an object, is ignored."""
    folder = course_home / "10-meet-lfm"
    folder.mkdir(parents=True)
    path = folder / "first_reply-bfloat16.json"
    path.write_text(json.dumps({"question": "q", "reply": "r"}))
    assert ask.read_before("q", course_home) is None
    path.write_text("[1, 2]")
    assert ask.read_before("q", course_home) is None


def test_ask_without_a_before_picture_and_a_bad_cap(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    """No chapter-10 file: the comparison is skipped, the budget still prints."""
    loaded = lfm.LoadedModel(None, FakeTokenizer(), course_home, 1, 0, 0, 0.1)
    monkeypatch.setattr(ask, "load_folder", lambda path: loaded)
    monkeypatch.setattr(ask, "model_for_later_chapters", lambda: (course_home, "ours"))
    monkeypatch.setattr(ask, "stream_generate", fake_stream({}))
    out = run_main(ask, "anything")
    assert "no 'before' picture" in out and "the real budget" in out
    with pytest.raises(SystemExit):
        run_main(ask, "--max-tokens", "0")
