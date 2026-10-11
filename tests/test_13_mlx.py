"""Chapter 13 with MLX: the adapters, training, saving, reloading, merging, AdamW.

No test here loads the real model, downloads anything or uses the network.
The adapters go on a tiny LFM2 built with mlx-lm's own ``lfm2`` code (64
numbers wide, four layers: convolution, attention, convolution, attention)
and quantized to 4 bits as chapter 11 did, so every step of lora_lab.py runs
on the same kind of model as the real one, only smaller. A fake model whose
scores are known in advance checks where each target token is read from.
MLX installs only on Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("13-lora")

import adam_steps  # noqa: E402
import arithmetic  # noqa: E402
import lfm  # noqa: E402
import lfm_files  # noqa: E402
import lora  # noqa: E402
import lora_lab  # noqa: E402
from lora_lab import Example  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
HIDDEN = 64
GROUP_SIZE = 32  # 64-wide matrices hold two groups per row
BITS = 4
RANK = 4
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": HIDDEN,
    "block_multiple_of": 64,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "block_ff_dim": 2 * HIDDEN,
    "rope_theta": 1e7,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
}
CONV_PAIRS = 5  # conv in/out, feed_forward w1/w2/w3
ATTENTION_PAIRS = 7  # self_attn q/k/v/out, feed_forward w1/w2/w3


class FakeTokenizer:
    """Stands in for mlx-lm's tokenizer: renders chats, encodes and decodes."""

    eos_token_ids = frozenset({EOS})
    eos_token_id = EOS

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        """Render a chat simply.

        Args:
            messages: The chat.
            **_: The template options, ignored.

        Returns:
            The rendered text.
        """
        return "|".join(message["content"] for message in messages) + "|"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn each character into an id below the end-of-turn id.

        Args:
            text: The text.
            add_special_tokens: Ignored.

        Returns:
            The ids.
        """
        return [ord(char) % EOS for char in text]

    def decode(self, ids: list[int]) -> str:
        """Write each id as a short piece of text.

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        return "".join("<end>" if token == EOS else f"t{token} " for token in ids)


class ChainModel:
    """A fake model: after token t, token t + 1 wins with score 10 + (t mod 5).

    Every other token scores 0, so every log-probability can be worked out
    by hand.
    """

    layers: list[Any] = []

    def make_cache(self) -> list[Any]:
        """Keep no cache: the scores depend only on the last token.

        Returns:
            An empty list.
        """
        return []

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        """Score every token after every input position.

        Args:
            inputs: Token ids, shape (1, length).
            cache: Ignored.

        Returns:
            Scores in bfloat16, shape (1, length, VOCAB).
        """
        ids = inputs[0]
        winner = (ids + 1) % VOCAB
        top = (10 + ids % 5).astype(mx.float32)
        hit = mx.arange(VOCAB)[None, :] == winner[:, None]
        return mx.where(hit, top[:, None], 0.0)[None].astype(mx.bfloat16)


def tiny_model(seed: int = 0, quantized: bool = True) -> nn.Module:
    """Build a tiny LFM2 with mlx-lm's own code, 4-bit unless asked otherwise.

    Args:
        seed: The random seed for the weights.
        quantized: Quantize its matrices to 4 bits, as chapter 11 did.

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    if quantized:
        nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def example() -> Example:
    """Make a short prompt and target for the tiny model.

    Returns:
        The example: 6 prompt tokens, 8 target tokens ending with EOS.
    """
    return Example([5, 9, 2, 11, 7, 3], [20, 21, 22, 23, 24, 25, 26, EOS])


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


def use_tiny(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make lora_lab.py load the tiny 4-bit model and the fake tokenizer.

    Only OUTPUT_CAP (1,024) is lowered, to keep the reply before training
    short; the other caps are 64 tokens, already quick on the tiny model.

    Args:
        monkeypatch: pytest's patcher.
        tmp_path: A folder to name as the build's.
    """
    model = tiny_model()
    loaded = lfm.LoadedModel(
        model=model,
        tokenizer=FakeTokenizer(),
        folder=tmp_path,
        weight_bytes=lfm.weight_bytes(model),
        parameters=lfm.parameter_count(model),
        seconds=0.0,
    )
    monkeypatch.setattr(lora_lab, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(lora_lab, "load_folder", lambda folder: loaded)
    monkeypatch.setattr(lora_lab, "OUTPUT_CAP", 16)


# ---------------------------------------------------------------- the model


def test_counting_unpacks_4bit_weights() -> None:
    """The 4-bit model holds as many numbers as the float one: words x 32 / 4."""
    plain = tiny_model(quantized=False)
    total = sum(array.size for _, array in tree_flatten(plain.parameters()))
    assert lfm.parameter_count(plain) == total
    assert lfm.parameter_count(tiny_model()) == total


def test_mlx_freezes_4bit_matrices_itself() -> None:
    """As loaded, only the norms and convolution filters are trainable."""
    names = [name for name, _ in tree_flatten(tiny_model().trainable_parameters())]
    assert names
    assert all(name.endswith("weight") for name in names)
    assert all("norm" in name or "conv.conv" in name for name in names)


# ---------------------------------------------------------------- the adapters


def test_adapters_go_on_every_projection_of_the_last_layers() -> None:
    """The last 2 layers: a convolution (5 pairs) and an attention layer (7)."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    pairs = lora.adapter_pairs(model)
    kinds = lora.layer_kinds(model)
    assert kinds == {
        0: arithmetic.CONVOLUTION,
        1: arithmetic.ATTENTION,
        2: arithmetic.CONVOLUTION,
        3: arithmetic.ATTENTION,
    }
    layers = arithmetic.by_layer(pairs, kinds)
    assert [(layer.layer, len(layer.keys)) for layer in layers] == [
        (2, CONV_PAIRS),
        (3, ATTENTION_PAIRS),
    ]
    assert [pair.key for pair in pairs if pair.layer == 3] == [
        key for key in lora.LORA_KEYS if not key.startswith("conv.")
    ]
    weights = lora.adapter_weights(model)
    assert all(name.endswith(("lora_a", "lora_b")) for name in weights)
    trainable = sum(array.size for array in weights.values())
    assert trainable == sum(pair.numbers() for pair in pairs)
    in_proj = next(pair for pair in pairs if pair.key == "conv.in_proj")
    assert (in_proj.inputs, in_proj.outputs, in_proj.rank) == (HIDDEN, 3 * HIDDEN, RANK)


def test_asking_for_more_layers_than_there_are_takes_them_all() -> None:
    """12 layers asked of a 4-layer model: all 4, 5 + 7 + 5 + 7 pairs."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 12)
    assert len(lora.adapter_pairs(model)) == 2 * (CONV_PAIRS + ATTENTION_PAIRS)


def test_a_fresh_adapter_changes_nothing() -> None:
    """B starts at zero and A random within 1/sqrt(in): the logits do not move."""
    model = tiny_model()
    ids = mx.array([[1, 2, 3, 4, 5]])
    before = model(ids).astype(mx.float32)
    lora.apply_lora(model, RANK, 2.0, 2)
    after = model(ids).astype(mx.float32)
    assert lora_lab.max_difference(after, before) == 0.0
    pairs = CONV_PAIRS + ATTENTION_PAIRS
    assert lora.zero_b_count(model) == (pairs, pairs)
    assert lora.a_in_range(model) == (pairs, pairs)


def test_the_checks_notice_a_nonzero_b_and_an_a_out_of_range() -> None:
    """A B that is not all zeros, and an A outside 1/sqrt(in), are not counted."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    pairs = CONV_PAIRS + ATTENTION_PAIRS
    _, first = lora.lora_modules(model)[0]
    first.lora_b = mx.ones(first.lora_b.shape)
    first.lora_a = mx.ones(first.lora_a.shape)  # 1 is above 1/sqrt(64)
    assert lora.zero_b_count(model) == (pairs - 1, pairs)
    assert lora.a_in_range(model) == (pairs - 1, pairs)
    _, second = lora.lora_modules(model)[1]
    second.lora_a = mx.zeros(second.lora_a.shape)  # in range, but all zeros
    assert lora.a_in_range(model) == (pairs - 2, pairs)


def test_the_seed_gives_the_same_adapter_every_time() -> None:
    """Seed 0 before adding the layers: two runs, the same A matrices."""
    first, second = tiny_model(), tiny_model()
    lora.apply_lora(first, RANK, 2.0, 2)
    lora.apply_lora(second, RANK, 2.0, 2)
    one, two = lora.adapter_weights(first), lora.adapter_weights(second)
    assert all(mx.array_equal(one[name], two[name]).item() for name in one)


def test_the_delta_is_what_the_layer_adds() -> None:
    """W x + scale (x A) B equals (W + scale B^T A^T) x."""
    model = tiny_model(quantized=False)
    lora.apply_lora(model, RANK, 2.0, 1)
    _, module = lora.lora_modules(model)[0]
    module.lora_b = mx.random.normal(module.lora_b.shape)
    rows = mx.random.normal((3, module.lora_a.shape[0]))
    merged = rows @ (module.linear.weight + lora.adapter_delta(module)).T
    assert mx.allclose(module(rows), merged, atol=1e-4).item()


# ---------------------------------------------------------------- training


def test_each_target_token_is_read_from_the_position_before_it() -> None:
    """With known scores, each log-probability matches the hand calculation."""
    chosen = Example([1, 2], [3, 4])
    picked, first = lora_lab.target_logprobs(ChainModel(), chosen)
    for token, value in zip(chosen.target, picked.tolist(), strict=True):
        top = 10 + (token - 1) % 5  # the score token - 1 gives its successor
        expected = top - math.log(math.exp(top) + VOCAB - 1)
        assert value == pytest.approx(expected, abs=1e-5)
    assert first.tolist() == [True, True]
    assert lora_lab.score(ChainModel(), chosen, 0).ranked_first == 2


def test_the_target_ends_with_the_end_of_turn_token() -> None:
    """The reply to learn is the text, then <|im_end|>, as the model writes it."""
    ids = lora_lab.target_ids(FakeTokenizer())
    assert ids[-1] == EOS
    assert len(ids) == len(lora_lab.TARGET_TEXT) + 1


def test_training_raises_the_target_and_saves_checkpoints(course_home: Path) -> None:
    """Six steps: the log-probability rises; adapters saved after steps 5 and 6."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    training = lora_lab.train(model, example(), 6, 1e-2)
    totals = [row.total for row in training.scored]
    assert len(totals) == 7 and totals[-1] > totals[0]
    assert sorted(training.saved) == [5, 6]
    pairs = CONV_PAIRS + ATTENTION_PAIRS
    for path in training.saved.values():
        header = lfm_files.read_safetensors_header(path)
        assert path.parent == course_home / "13-lora"
        assert header.tensors == 2 * pairs and header.dtypes == ("F32",)
        assert header.numbers == sum(
            array.size for array in lora.adapter_weights(model).values()
        )
    trainable = sum(a.size for a in lora.adapter_weights(model).values())
    assert training.state_bytes >= 2 * 4 * trainable  # m and v in float32


def test_only_the_adapters_change(course_home: Path) -> None:
    """Every frozen array is the same after training; A and B are not."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    before = dict(tree_flatten(model.parameters()))
    lora_lab.train(model, example(), 3, 1e-2)
    after = dict(tree_flatten(model.parameters()))
    moved = {name for name in before if not mx.array_equal(before[name], after[name])}
    assert moved
    assert all(name.endswith(("lora_a", "lora_b")) for name in moved)


def test_zeroing_and_reloading_restore_logits_exactly(course_home: Path) -> None:
    """Zero B: the base model; reload a file: the model that saved it."""
    model = tiny_model()
    chosen = example()
    base = lora_lab.all_logits(model, chosen)
    lora.apply_lora(model, RANK, 2.0, 2)
    training = lora_lab.train(model, chosen, 6, 1e-2)
    trained = lora_lab.all_logits(model, chosen)
    final = lora.save_adapter(model, lfm_files.final_adapter_path())
    lora.zero_adapter(model)
    assert lora_lab.max_difference(lora_lab.all_logits(model, chosen), base) == 0.0
    lora.load_adapter(model, final)
    assert lora_lab.max_difference(lora_lab.all_logits(model, chosen), trained) == 0.0
    lora.load_adapter(model, training.saved[5])
    assert lora_lab.score(model, chosen, 5).total == training.scored[5].total
    with pytest.raises(FileNotFoundError):
        lora.load_adapter(model, course_home / "missing.safetensors")


def test_merging_keeps_or_rounds_and_puts_the_adapters_back(course_home: Path) -> None:
    """Merged unrounded: the adapter's scores; either way the adapters return."""
    model = tiny_model()
    chosen = example()
    lora.apply_lora(model, RANK, 2.0, 2)
    lora_lab.train(model, chosen, 6, 1e-2)
    adapter = lora_lab.score(model, chosen, 0)
    count = len(lora.lora_modules(model))
    kept = lora_lab.merge_and_score(
        model, FakeTokenizer(), chosen, dequantize=True, step=6
    )
    assert kept.scored.total == pytest.approx(adapter.total, abs=1e-3)
    assert kept.scored.step == 6
    assert len(lora.lora_modules(model)) == count
    rounded = lora_lab.merge_and_score(
        model, FakeTokenizer(), chosen, dequantize=False, step=6
    )
    assert rounded.size < kept.size  # 4-bit matrices are smaller than unrounded ones
    assert len(lora.lora_modules(model)) == count
    assert lora_lab.score(model, chosen, 0).total == adapter.total
    change = lora.change_and_step(lora.lora_modules(model))
    assert 0 < change.mean <= change.largest and change.step > 0


def test_greedy_replies_stop_at_the_end_of_turn_or_the_cap() -> None:
    """After token 61 the chain gives 62 then 63 (the end of turn): 'stop'."""
    stopped = lfm.greedy_reply(ChainModel(), FakeTokenizer(), [61], 10)
    assert stopped == lfm.Reply([62, EOS], lfm.FINISH_STOP)
    capped = lfm.greedy_reply(ChainModel(), FakeTokenizer(), [10], 3)
    assert capped == lfm.Reply([11, 12, 13], lfm.FINISH_LENGTH)


def test_small_helpers() -> None:
    """Number-type names, nested optimizer state, and the largest difference."""
    arrays = [mx.zeros(2, dtype=mx.bfloat16), mx.zeros(2, dtype=mx.float32)]
    assert lora_lab.type_names(arrays) == "bfloat16, float32"
    state = {"step": mx.array(1), "a": {"m": mx.zeros(3), "v": [mx.zeros(2)]}}
    assert len(lora_lab.optimizer_arrays(state)) == 3
    assert lora_lab.max_difference(mx.array([1.0, 2.0]), mx.array([1.5, 2.0])) == 0.5


# ---------------------------------------------------------------- main()


def test_main_fast_shows_every_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """--fast: 5 steps, one checkpoint and the plain-named copy, no merge."""
    use_tiny(monkeypatch, tmp_path / "build")
    out = run_main(lora_lab, "--fast", "--lr", "0.01")
    for number in range(1, 8):
        assert f"\n{number}. " in out
    assert "8. merging the adapter into the frozen weights: skipped with --fast" in out
    assert "max |logit difference| with a fresh adapter: 0.0" in " ".join(out.split())
    saved = sorted(path.name for path in (course_home / "13-lora").iterdir())
    assert saved == ["adapter.safetensors", "adapter_step005.safetensors"]


def test_main_full_merges_and_reloads_a_middle_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --fast: the reload of an earlier checkpoint, then both merges."""
    use_tiny(monkeypatch, tmp_path / "build")
    out = run_main(lora_lab, "--steps", "11", "--lr", "0.01", "--scale", "20")
    plain = " ".join(out.split())
    assert "scale 20.0 (--scale; the project's is 2.0)" in plain
    # saved after steps 5, 10 and 11: the middle of the earlier two is step 10
    assert "load adapter_step010.safetensors: the target's log-probability" in plain
    assert "8. merging: adding scale x B x A into the frozen matrices" in out
    assert "merged into 16-bit copies:" in out and "merged back into 4 bits:" in out


@pytest.mark.parametrize("args", [["--steps", "0"], ["--lr", "0"], ["--scale", "-1"]])
def test_bad_flags_stop_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    """Impossible settings stop before any model is loaded."""
    use_tiny(monkeypatch, tmp_path / "build")
    with pytest.raises(SystemExit):
        run_main(lora_lab, *args)


# ------------------------------------------------------------ adam_steps.py


def test_steady_updates_match_the_formula() -> None:
    """MLX's measured updates equal (1 - 0.9^t) / sqrt(1 - 0.999^t), or 1."""
    measured = adam_steps.steady(1e-5, 20)
    for step in range(1, 21):
        expected = arithmetic.adam_ratio(step, 0.9, 0.999, bias_correction=False)
        assert measured.without[step - 1] == pytest.approx(expected, rel=1e-3)
        assert measured.with_correction[step - 1] == pytest.approx(1.0, rel=1e-3)
    assert measured.same_for_sizes


def test_attempt_one_stepped_far_harder_than_attempt_two() -> None:
    """4e-5 uncorrected against 1e-5 corrected: 12.6 times at step 1."""
    one = adam_steps.attempt(*adam_steps.ATTEMPT_ONE)
    two = adam_steps.attempt(*adam_steps.ATTEMPT_TWO)
    assert len(one.updates) == adam_steps.ATTEMPT_STEPS
    assert one.updates[0] / two.updates[0] == pytest.approx(4 * math.sqrt(10), rel=1e-3)


def test_weight_decay_alone_shrinks_a_weight() -> None:
    """With a zero gradient, each step multiplies by 1 - lr x 0.01 (in float32)."""
    result = adam_steps.decay(1e-5)
    assert result.values[0] == mx.array(1 - 1e-5 * 0.01, dtype=mx.float32).item()
    assert result.values == sorted(result.values, reverse=True)
    assert result.values[-1] < 1.0


def test_the_lora_scale_multiplies_how_far_the_output_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale 20 against 2: B moves the same, the output ten times as far."""
    monkeypatch.setattr(adam_steps, "WIDTH", 256)
    small, large = adam_steps.scale_step(2.0), adam_steps.scale_step(20.0)
    for little, big in zip(small.moved, large.moved, strict=True):
        assert big / little == pytest.approx(10.0, rel=1e-3)
    assert mx.allclose(small.first_b, large.first_b, atol=1e-6).item()


def test_adam_main_prints_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults, the steady table, the attempts, the decay and the scale."""
    monkeypatch.setattr(adam_steps, "WIDTH", 256)
    out = run_main(adam_steps)
    for heading in (
        "1. a steady gradient",
        "2. the project's two RL attempts",
        "3. weight decay",
        "4. the LoRA scale",
    ):
        assert heading in out
    assert "weight_decay 0.01, bias_correction False" in " ".join(out.split())
