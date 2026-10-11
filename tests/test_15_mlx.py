"""Chapter 15 with MLX: the sampler, the routes, the update, and the scripts.

No test here loads the real model, downloads anything or uses the network.
Everything runs on chapter 14's tiny LFM2 (mlx-lm's own ``lfm2`` code, 64
numbers wide, four layers, quantized to 4 bits) with chapter 13's adapters,
and a fake tokenizer that renders LFM2.5's chat markers and maps every other
character to an id the tiny model knows. Grading is replaced by a fake that
alternates right and wrong, so every group has contrast. The three scripts
run end to end on them. MLX installs only on Apple-silicon Macs; elsewhere
this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("15-grpo-real-model")

import eval_sql  # noqa: E402
import grpo  # noqa: E402
import grpo_sql  # noqa: E402
import lfm  # noqa: E402
import lfm_files  # noqa: E402
import lora  # noqa: E402
import same_tokens  # noqa: E402
import sampler  # noqa: E402
from grading import Verdict  # noqa: E402
from rows import TrainItem  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
FIRST_PLAIN = 8  # ids below this are the chat markers
HIDDEN = 64
GROUP_SIZE = 32
BITS = 4
RANK = 4
CAP = 6  # reply cap for the scripts' runs on the tiny model
MARKERS = {
    "<|startoftext|>": 1,
    "<|im_start|>": 2,
    "<think>": 3,
    "</think>": 4,
    "<|tool_call_start|>": 5,
    "<|tool_call_end|>": 6,
    "<|im_end|>": EOS,
}
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


class FakeTokenizer:
    """Renders chats with LFM2.5's markers; every other character gets one id."""

    eos_token_ids = frozenset({EOS})
    eos_token_id = EOS

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> str:
        """Render a chat as LFM2.5's template does, markers and all.

        Args:
            messages: The chat.
            **options: ``add_generation_prompt`` is read; the rest ignored.

        Returns:
            The rendered text.
        """
        text = "<|startoftext|>" + "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if options.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n<think>"
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Give each marker its id, and every other character one below EOS.

        Args:
            text: The text.
            add_special_tokens: Ignored.

        Returns:
            The ids.
        """
        ids, index = [], 0
        while index < len(text):
            for marker, token in MARKERS.items():
                if text.startswith(marker, index):
                    ids.append(token)
                    index += len(marker)
                    break
            else:
                ids.append(FIRST_PLAIN + ord(text[index]) % (EOS - FIRST_PLAIN))
                index += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        """Write markers back, and every other id as a letter (not the inverse).

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        names = {token: marker for marker, token in MARKERS.items()}
        return "".join(names[i] if i in names else chr(ord("a") + i % 26) for i in ids)


def tiny_model(seed: int = 0, dtype: mx.Dtype = mx.float32) -> nn.Module:
    """Build a tiny 4-bit LFM2 with mlx-lm's own code.

    Args:
        seed: The random seed for the weights.
        dtype: The precision of its non-quantized numbers (bfloat16 makes it
            compute as the real model does).

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def adapted(seed: int = 0, dtype: mx.Dtype = mx.float32) -> nn.Module:
    """A tiny model with the chapter's adapter shape on it.

    Args:
        seed: The weights' seed.
        dtype: The non-quantized numbers' precision.

    Returns:
        The model with LoRA layers.
    """
    model = tiny_model(seed, dtype)
    lora.apply_lora(model, lfm_files.LORA_RANK, lfm_files.LORA_SCALE, 2)
    return model


def item(ids: list[int], start: int, advantage: float = 1.0) -> TrainItem:
    """Make a training row whose tokens from ``start`` on are trained.

    Args:
        ids: The row's tokens.
        start: The first trained position.
        advantage: Its advantage.

    Returns:
        The row, old log-probabilities 0.
    """
    positions = list(range(start, len(ids)))
    return TrainItem(ids, positions, [0.0] * len(positions), advantage)


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


def plain(out: str) -> str:
    """Undo the line wrapping of printed output.

    Args:
        out: Printed text.

    Returns:
        The same words, single-spaced.
    """
    return " ".join(out.split())


def use_tiny(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """Make every script load the tiny model, and give it chapter 14's adapter.

    Args:
        monkeypatch: pytest's patcher.
        tmp_path: A folder to name as the build's.
        course_home: The temporary course home.
    """

    def load(folder: Path) -> lfm.LoadedModel:
        """Give a fresh tiny model, as loading the build would.

        Args:
            folder: The build's folder.

        Returns:
            The tiny model and the fake tokenizer.
        """
        return lfm.LoadedModel(tiny_model(), FakeTokenizer(), folder, 0.0)

    for module in (grpo_sql, eval_sql, same_tokens):
        monkeypatch.setattr(module, "pick_model", lambda fast: (tmp_path, "tiny"))
        monkeypatch.setattr(module, "load_folder", load)
        monkeypatch.setattr(module, "OUTPUT_CAP", CAP)
    turns = itertools.cycle([True, False])

    def grade(task: Any, text: str, folder: Path, cut_off: bool = False) -> Verdict:
        """Alternate right and wrong, so every group of 4 has contrast.

        Args:
            task: The question.
            text: The reply.
            folder: Its work folder.
            cut_off: Whether the cap ended it.

        Returns:
            The verdict.
        """
        right = next(turns)
        return Verdict(right, "SELECT 1", "1", "correct" if right else "fake: wrong")

    monkeypatch.setattr(grpo_sql, "grade_reply", grade)
    start = adapted()
    start_path = course_home / "14-sft-real-model" / "sft.safetensors"
    lora.save_adapter(start, start_path)


# ---------------------------------------------------------------- the sampler


def test_the_sampler_keeps_each_tokens_id_and_float32_log_probability() -> None:
    """Every token's log p and entropy; the engine's number equals the recorded one."""
    model = adapted()
    reply = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(), 8)
    assert 1 <= len(reply.tokens) <= 8
    assert len(reply.logprobs) == len(reply.entropies) == len(reply.tokens)
    assert reply.full_logprobs == reply.logprobs
    assert reply.full_entropies == reply.entropies
    assert all(value <= 0 for value in reply.logprobs)
    if reply.finish_reason == "stop":
        assert reply.tokens[-1] == EOS
    else:
        assert len(reply.tokens) == 8
    with pytest.raises(ValueError, match="at least 1"):
        sampler.sample(model, {EOS}, [1, 9], sampler.Draw(), 0)


def test_the_same_seed_draws_the_same_reply() -> None:
    """mx.random.seed fixes the draws."""
    model = adapted()
    mx.random.seed(3)
    one = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(), 8)
    mx.random.seed(3)
    two = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(), 8)
    assert one.tokens == two.tokens and one.logprobs == two.logprobs


def test_the_samplers_route_reproduces_its_numbers_exactly() -> None:
    """Feeding the drawn tokens one at a time gives the recorded log p, bit for bit."""
    model = adapted(dtype=mx.bfloat16)
    prompt = [1, 9, 20, 33]
    reply = sampler.sample(model, {EOS}, prompt, sampler.Draw(), 8)
    again = same_tokens.stepwise_logprobs(model, prompt, reply.tokens)
    assert again == reply.logprobs


def test_bf16_records_coarser_numbers_beside_the_engines() -> None:
    """In bfloat16 the recorded log p is a bfloat16 number; float32's is kept."""
    model = adapted(dtype=mx.bfloat16)
    mx.random.seed(0)
    reply = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(dtype="bf16"), 8)
    for value in reply.logprobs:
        assert mx.array(value).astype(mx.bfloat16).item() == value
    assert reply.full_logprobs != reply.logprobs


def test_top_k_raises_the_kept_probabilities_and_changes_the_entropy() -> None:
    """A cut can only raise a kept token's probability; it changes the entropy."""
    model = adapted()
    mx.random.seed(0)
    reply = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(top_k=3), 8)
    for cut, full in zip(reply.logprobs, reply.full_logprobs, strict=True):
        assert cut >= full - 1e-5
    assert reply.entropies != reply.full_entropies
    scores = mx.array([[0.0, 1.0, 2.0, 3.0]])
    logprobs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    kept = sampler.drawn_from(logprobs, sampler.Draw(top_k=2))
    assert mx.isinf(kept[0, :2]).all().item()
    assert mx.allclose(mx.exp(kept[0, 2:]).sum(), mx.array(1.0)).item()


def test_a_draw_refuses_impossible_settings() -> None:
    """An unknown precision, a temperature of 0, a negative top-k."""
    for options in ({"dtype": "fp8"}, {"temperature": 0.0}, {"top_k": -1}):
        with pytest.raises(ValueError):
            sampler.Draw(**options)


# ---------------------------------------------------------------- the routes


def test_the_trainers_route_is_the_one_the_update_sees() -> None:
    """Old log p from trainer_logprobs: every ratio in the clipped loss is exactly 1."""
    model = adapted(dtype=mx.bfloat16)
    row = item([1, 9, 20, 33, 12, 40, EOS], 3, advantage=1.0)
    old = grpo.trainer_logprobs(model, row).tolist()
    row = TrainItem(row.input_ids, row.positions, old, 1.0)
    (_, ratio), _ = nn.value_and_grad(model, grpo.clipped_loss(model))(row)
    assert ratio.tolist() == [1.0] * row.n_tokens


def test_to_float32_makes_every_number_float32() -> None:
    """The 16-bit scales and norms become float32; the 4-bit codes stay."""
    model = tiny_model(dtype=mx.bfloat16)
    lfm.to_float32(model)
    dtypes = {value.dtype for _, value in tree_flatten(model.parameters())}
    assert mx.bfloat16 not in dtypes and mx.float32 in dtypes
    assert mx.uint32 in dtypes
    hidden = model.model(mx.array([[1, 9, 20]]))
    assert hidden.dtype == mx.float32


# ---------------------------------------------------------------- the update


def test_no_contrast_means_no_update() -> None:
    """No rows: nothing changes, and the update says so."""
    model = adapted()
    before = dict(tree_flatten(model.trainable_parameters()))
    update = grpo.grpo_update(model, optim.AdamW(learning_rate=1e-2), [])
    assert update.rows == 0 and update.loss_per_token is None and update.ratios == []
    after = dict(tree_flatten(model.trainable_parameters()))
    assert all(mx.array_equal(before[name], after[name]) for name in before)


def test_a_positive_advantage_makes_its_reply_likelier() -> None:
    """One step on a row with A > 0 raises its log p; only the adapter moves."""
    model = adapted()
    ids = [1, 9, 20, 33, 12, 40, EOS]
    row = item(ids, 3)
    old = grpo.trainer_logprobs(model, row).tolist()
    row = TrainItem(ids, row.positions, old, 1.0)
    everything = dict(tree_flatten(model.parameters()))
    update = grpo.grpo_update(model, optim.AdamW(learning_rate=1e-2), [row])
    assert update.rows == 1 and update.tokens == 4
    assert update.ratios == [[1.0] * 4] and update.kl == [[]]
    assert sum(grpo.trainer_logprobs(model, row).tolist()) > sum(old)
    moved = {
        name
        for name, value in tree_flatten(model.parameters())
        if not mx.array_equal(everything[name], value)
    }
    assert moved and all(name.endswith(("lora_a", "lora_b")) for name in moved)


def test_the_drift_penalty_starts_at_exactly_zero_and_then_grows() -> None:
    """--kl: 0 at the first update (same weights, same route), above 0 after."""
    model = adapted(dtype=mx.bfloat16)
    reference = dict(lora.adapter_weights(model))
    rows = [item([1, 9, 20, 33, 12, 40, EOS], 3, 1.0), item([1, 11, 41, EOS], 2, -1.0)]
    optimizer = optim.AdamW(learning_rate=1e-2)
    first = grpo.grpo_update(model, optimizer, rows, 0.05, reference)
    assert all(value == 0.0 for row in first.kl for value in row)
    second = grpo.grpo_update(model, optimizer, rows, 0.05, reference)
    assert all(value > 0.0 for row in second.kl for value in row)
    with pytest.raises(ValueError, match="start adapter"):
        grpo.grpo_update(model, optimizer, rows, 0.05, None)


def test_scoring_with_other_weights_puts_the_current_ones_back() -> None:
    """score_with swaps the start adapter in and out; the model is unchanged."""
    model = adapted()
    row = item([1, 9, 20, 33, EOS], 2)
    start = dict(lora.adapter_weights(model))
    grpo.grpo_update(model, optim.AdamW(learning_rate=1e-2), [row])
    current = dict(lora.adapter_weights(model))
    scored = grpo.score_with(model, start, [row])[0]
    assert scored.shape == (row.n_tokens,)
    after = dict(lora.adapter_weights(model))
    assert all(mx.array_equal(current[name], after[name]) for name in current)


def test_a_reply_becomes_its_row_and_the_engines_twin() -> None:
    """Default: every sampled token trained; --retokenize: mismatches masked."""
    model = adapted()
    tokenizer = FakeTokenizer()
    mx.random.seed(1)
    reply = sampler.sample(model, {EOS}, [1, 9, 20], sampler.Draw(top_k=3), 6)
    row, twin, later = grpo_sql.row_for([1, 9, 20], reply, 0.5, None)
    assert row.positions == list(range(3, 3 + len(reply.tokens)))
    assert row.old_logprobs == reply.logprobs
    assert twin.old_logprobs == reply.full_logprobs
    assert row.advantage == 0.5 and later == [False] * len(reply.tokens)
    reencoded = tokenizer.encode(tokenizer.decode(reply.tokens))
    row, twin, later = grpo_sql.row_for([1, 9, 20], reply, 0.5, reencoded)
    if reencoded == reply.tokens:
        assert row.n_tokens == len(reply.tokens)
    else:
        assert row is None or row.n_tokens < len(reply.tokens)
    ratios = grpo_sql.unmasked_ratios(model, [1, 9, 20], reply, reencoded)
    shared = min(len(reencoded), len(reply.tokens))
    mismatched = sum(reencoded[i] != reply.tokens[i] for i in range(shared))
    assert len(ratios) == mismatched
    assert all(math.isfinite(ratio) and ratio > 0 for ratio in ratios)


# ---------------------------------------------------------------- the scripts


def test_same_tokens_scores_one_reply_by_every_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """Three routes in each precision; the sampler's route is identical."""
    use_tiny(monkeypatch, tmp_path, course_home)
    out = run_main(same_tokens)
    text = plain(out)
    assert "1. bfloat16, as the model runs" in out
    assert "3. float32 (every 16-bit number cast)" in out
    assert "the sampler's route again (one token at a time, with a cache)" in text
    assert "one pass inside a gradient computation (the trainer's update)" in text
    first = text.split("one pass over the row")[0]
    assert "largest |difference in log-probability| 0.000000" in first


def test_grpo_sql_trains_logs_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """Two steps: a line each, the detail, the summaries, the files, the box."""
    use_tiny(monkeypatch, tmp_path, course_home)
    out = run_main(grpo_sql, "--steps", "2", "--lr", "0.01")
    text = plain(out)
    for heading in (
        "1. training: sample 8 replies",
        "2. step 1 in detail",
        "3. the importance ratio, token by token",
        "4. the groups, and the replies' length",
        "5. saved",
        "6. the project's runs (recorded, not run here)",
    ):
        assert heading in out
    assert "4/8" in out  # the fake grader alternates
    assert "(--lr; the project's is 1e-05)" in text
    home = course_home / "15-grpo-real-model"
    assert (home / "grpo_lr0.01_steps2.safetensors").is_file()
    metrics = (home / "grpo_lr0.01_steps2.metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in metrics] == [1, 2]
    assert all(json.loads(line)["importance_ratio_mean"] for line in metrics)
    rollouts = (home / "grpo_lr0.01_steps2.rollouts.jsonl").read_text().splitlines()
    assert len(rollouts) == 16
    assert {json.loads(line)["step"] for line in rollouts} == {1, 2}


@pytest.mark.parametrize(
    ("args", "name", "says"),
    [
        (["--retokenize"], "grpo_retokenize_steps1", "--retokenize:"),
        (["--logprob-dtype", "bf16"], "grpo_bf16_steps1", "instead of bfloat16"),
        (["--top-k", "3"], "grpo_topk3_steps1", "the 3 kept tokens held"),
        (["--kl", "0.05"], "grpo_kl0.05_steps1", "step by step: 0.00000"),
        (["--float32"], "grpo_float32_steps1", "float32"),
    ],
)
def test_each_flag_runs_and_reports_its_own_measure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    course_home: Path,
    args: list[str],
    name: str,
    says: str,
) -> None:
    """Each flag gets its own adapter name and its own summary line."""
    use_tiny(monkeypatch, tmp_path, course_home)
    out = plain(run_main(grpo_sql, "--steps", "1", *args))
    assert says in out
    assert (course_home / "15-grpo-real-model" / f"{name}.safetensors").is_file()


def test_grpo_sql_needs_chapter_14s_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """No start adapter: stop with chapter 14's command; impossible flags stop too."""
    use_tiny(monkeypatch, tmp_path, course_home)
    (course_home / "14-sft-real-model" / "sft.safetensors").unlink()
    with pytest.raises(SystemExit, match="uv run train_sft.py"):
        run_main(grpo_sql, "--steps", "1")
    for args in (["--steps", "0"], ["--top-k", "-1"], ["--kl", "-1"], ["--lr", "0"]):
        with pytest.raises(SystemExit):
            run_main(grpo_sql, *args)


def test_grpo_sql_stops_before_training_if_a_project_record_is_damaged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """A broken bundled metrics file stops the script before any step."""
    use_tiny(monkeypatch, tmp_path, course_home)
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n")
    monkeypatch.setattr(grpo_sql, "ATTEMPT_1", broken)
    with pytest.raises(SystemExit, match="not a GRPO step"):
        run_main(grpo_sql, "--steps", "1")
    assert not (course_home / "15-grpo-real-model").exists()


def test_eval_sql_compares_three_models_and_reuses_their_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """Base, SFT and SFT+GRPO answered, compared, then read back the second time."""
    use_tiny(monkeypatch, tmp_path, course_home)
    with pytest.raises(SystemExit, match="run grpo_sql.py first"):
        run_main(eval_sql, "--limit", "3")
    run_main(grpo_sql, "--steps", "1", "--lr", "0.01")
    adapter = "grpo_lr0.01_steps1.safetensors"
    out = plain(run_main(eval_sql, "--adapter", adapter, "--limit", "3"))
    assert "base: pass@1" not in out  # the three scores sit in one section
    assert "every question's verdict" in out
    assert "sft against sft+grpo: gained" in out
    assert "base against sft: gained" in out
    assert out.count("answered now") == 3
    home = course_home / "15-grpo-real-model"
    for label in ("base", "sft", "grpo_lr0.01_steps1"):
        assert len((home / f"eval_{label}.jsonl").read_text().splitlines()) == 3
    again = plain(run_main(eval_sql, "--adapter", adapter, "--limit", "3"))
    assert again.count("the same adapter bytes and cap") == 3
    lora.save_adapter(adapted(seed=5), home / adapter)  # new bytes, same name
    changed = plain(run_main(eval_sql, "--adapter", adapter, "--limit", "3"))
    assert changed.count("the same adapter bytes and cap") == 2
    lora.save_adapter(adapted(), home / "sft.safetensors")
    with pytest.raises(SystemExit, match="share one file of saved answers"):
        run_main(eval_sql, "--adapter", "sft.safetensors", "--limit", "3")
