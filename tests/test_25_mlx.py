"""Chapter 25 with MLX: captured rows, the 9 GB settings, the trainer and the probe.

No test here loads the 2.6B model, downloads anything or uses the network.
Everything runs on chapter 15's tiny LFM2 (mlx-lm's own ``lfm2`` code, 64
numbers wide, four layers, quantized to 4 bits) with chapter 13's adapters.
To train on rows captured through the real harnesses, whose token ids are
LFM2.5's, the tiny model gets LFM2.5's vocabulary size (128,000); the
attention and memory tests use a 64-token vocabulary. ``train_sft.py`` and
``memory_probe.py`` run end to end on it. MLX installs only on Apple-silicon
Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("25-sft-in-harnesses")

import lfm  # noqa: E402
import lora  # noqa: E402
import memory_probe  # noqa: E402
import train_sft  # noqa: E402
import training  # noqa: E402

DATA = Path(__file__).parent / "data"
CAPTURED = DATA / "25_captured_demos.jsonl"
LFM_VOCAB = 128_000  # LFM2.5's vocabulary: captured ids go up to 124,900
SMALL_VOCAB = 64
EOS = SMALL_VOCAB - 1
HIDDEN = 64
GROUP_SIZE = 32
BITS = 4
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": SMALL_VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 8192,
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
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: this machine's own captured demonstrations, if collect.py has run.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
REAL_DEMOS = _HOME / "25-sft-in-harnesses" / "demos" / "demos.jsonl"


def tiny_model(vocab: int = SMALL_VOCAB, seed: int = 0) -> nn.Module:
    """Build a tiny 4-bit LFM2 with mlx-lm's own code.

    Args:
        vocab: Its vocabulary size.
        seed: The random seed for the weights.

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**{**TINY_CONFIG, "vocab_size": vocab}))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def adapted(vocab: int = SMALL_VOCAB, rank: int = lora.LORA_RANK) -> nn.Module:
    """A tiny model with chapter 13's adapter on it.

    Args:
        vocab: Its vocabulary size.
        rank: The adapter's rank.

    Returns:
        The model.
    """
    model = tiny_model(vocab)
    lora.apply_lora(model, rank, lora.LORA_SCALE, lora.LORA_LAYERS)
    mx.eval(model.parameters())
    return model


def nudge(model: nn.Module, seed: int = 1) -> None:
    """Give every adapter B matrix small random values, so the adapter acts.

    Args:
        model: A model with LoRA layers.
        seed: The random seed.
    """
    mx.random.seed(seed)
    updates = [
        (name, 0.05 * mx.random.normal(value.shape))
        for name, value in lora.adapter_weights(model).items()
        if name.endswith(lora.B_SUFFIX)
    ]
    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())


def row(length: int, start: int, seed: int = 0) -> training.TrainItem:
    """A random row whose positions from ``start`` on are trained.

    Args:
        length: Its length.
        start: The first trained position.
        seed: The draw's seed.

    Returns:
        The row.
    """
    item = memory_probe.probe_row(length, SMALL_VOCAB, seed)
    positions = list(range(start, length))
    return replace(item, positions=positions, old_logprobs=[0.0] * len(positions))


def run_main(
    module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> str:
    """Run a script's main() as from the command line, and capture its output.

    Args:
        module: The script's module.
        argv: Its command line, script name first.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.

    Returns:
        What it printed.
    """
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return capsys.readouterr().out


class FakeTokenizer:
    """Just enough of a tokenizer for lfm.LfmModel: the end of turn, and ids."""

    eos_token_ids = frozenset({EOS})

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Give every character an id the tiny model knows.

        Args:
            text: The text.
            add_special_tokens: Ignored.

        Returns:
            The ids.
        """
        return [ord(char) % EOS for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Write ids back as letters.

        Args:
            ids: The ids.
            skip_special_tokens: Ignored.

        Returns:
            The text.
        """
        return "".join(chr(ord("a") + i % 26) for i in ids)


# ------------------------------------------------------------------ captured rows


def test_a_captured_row_trains_only_its_mask_1_tokens_after_position_0() -> None:
    sequence = {"input_ids": [5, 6, 7, 8], "loss_mask": [1, 0, 1, 1]}
    item = training.item_from_sequence(sequence)
    assert item is not None
    assert item.positions == [2, 3] and item.old_logprobs == [0.0, 0.0]
    assert training.item_from_sequence({"input_ids": [1], "loss_mask": [1]}) is None


def test_demo_items_keeps_usable_rows_and_counts_every_one_it_skips() -> None:
    good = {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}
    records = [
        {"correct": True, "trainable": True, "sequences": [good]},
        {"correct": False, "trainable": True, "sequences": [good]},
        {"correct": True, "trainable": False, "sequences": [good]},
        {
            "correct": True,
            "trainable": True,
            "sequences": [
                {"input_ids": [1] * 9, "loss_mask": [1] * 9},
                {"input_ids": [1, 2], "loss_mask": [0, 0]},
            ],
        },
    ]
    rows = training.demo_items(records, max_tokens=8)
    assert (rows.demos, rows.usable, rows.too_long, rows.empty) == (4, 2, 1, 1)
    assert [item.positions for item in rows.items] == [[1, 2]]


# ------------------------------------------------------------------ fitting in 9 GB


def test_chunked_attention_gives_the_fused_kernels_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "QUERY_CHUNK", 8)  # 40 tokens: 5 blocks
    model = adapted()
    nudge(model)
    item = row(40, 10)
    with training.attention(chunked=False):
        fused = training.token_logprobs(model, item)
    with training.attention(chunked=True):
        chunked = training.token_logprobs(model, item)
    assert mx.allclose(fused, chunked, atol=1e-4).item()
    assert lfm2.scaled_dot_product_attention is training._fused_attention


def test_logits_at_trained_positions_equal_the_full_vocabulary_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "LOGIT_CHUNK", 7)  # 30 trained: 5 chunks
    model = adapted()
    item = row(40, 10)
    full = training.full_vocab_logprobs(model, item)
    assert mx.allclose(training.token_logprobs(model, item), full, atol=1e-5).item()


def test_checkpointing_keeps_the_gradients_and_is_undone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "QUERY_CHUNK", 8)
    model = adapted()
    nudge(model)
    item = row(40, 10)
    layer_call = type(model.layers[0]).__call__

    def gradients(on: bool) -> dict[str, mx.array]:
        """The adapter's gradients with checkpointing on or off.

        Args:
            on: Whether to checkpoint.

        Returns:
            Name to gradient.
        """

        def loss_fn(one: training.TrainItem) -> mx.array:
            return -training.token_logprobs(model, one).sum()

        with training.attention(chunked=True), training.layer_checkpointing(model, on):
            _, grads = nn.value_and_grad(model, loss_fn)(item)
            mx.eval(grads)
        return dict(tree_flatten(grads))

    plain, checkpointed = gradients(False), gradients(True)
    assert plain.keys() == checkpointed.keys()
    for name, value in plain.items():
        assert mx.allclose(value, checkpointed[name], atol=1e-5).item(), name
    assert type(model.layers[0]).__call__ is layer_call


# ------------------------------------------------------------------ train_sft.py


def use_tiny_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make train_sft.py load the tiny model with LFM2.5's vocabulary.

    Args:
        monkeypatch: pytest's patcher.
        tmp_path: A folder to name as the build's.
    """
    monkeypatch.setattr(train_sft, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(train_sft, "load_model", lambda folder: tiny_model(LFM_VOCAB))


def test_train_sft_learns_from_the_captured_rows_and_saves_the_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path, course_home: Path
) -> None:
    use_tiny_trainer(monkeypatch, tmp_path)
    argv = ["train_sft.py", "--demos", str(CAPTURED), "--batch", "1", "--epochs"]
    out = run_main(train_sft, [*argv, "4", "--lr", "1e-2"], monkeypatch, capsys)
    assert "2 demonstrations, 2 usable (correct and trainable)" in out
    assert "rows      2 kept · 0 over 6,144 tokens skipped" in out
    assert "mini-swe-agent 2 rows · 851 tokens on average · 168 trained" in out
    assert "lr 0.01 · batch 1 · 4 epochs = 8 steps" in out
    assert "[sft] 2 training rows from 2 demos" in out
    folder = course_home / "25-sft-in-harnesses" / "sft"
    metrics = (folder / "metrics.jsonl").read_text().splitlines()
    lines = [json.loads(line) for line in metrics]
    assert [line["step"] for line in lines] == list(range(1, 9))
    assert {line["tokens"] for line in lines} == {147, 190}
    assert lines[-1]["loss_per_token"] < lines[0]["loss_per_token"]
    assert "[sft] step 8/8 · loss/token" in out
    adapter = folder / "adapter.safetensors"
    model = adapted(LFM_VOCAB)
    lora.check_adapter_file(model, adapter)  # the engine's shape exactly
    note = json.loads((folder / "adapter.json").read_text())
    assert note["which"] == "tiny" and note["rows"] == 2 and note["steps"] == 8
    assert (
        "main run (runs/main/sft/metrics.jsonl): 15 steps, loss/token 2.55 at "
        "step 1 -> 0.11 at step 15, peak 3.95 GB"
    ) in out
    assert (
        "dry run (runs/dry/sft/metrics.jsonl, lr 1e-4 by mhrl/cli.py's comment): "
        "6 steps, 2.33 at step 1, 0.17 at step 4, 0.10 at step 5"
    ) in out


def test_without_bias_correction_it_trains_into_its_own_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path, course_home: Path
) -> None:
    use_tiny_trainer(monkeypatch, tmp_path)
    argv = ["train_sft.py", "--demos", str(CAPTURED), "--no-bias-correction"]
    out = run_main(train_sft, argv, monkeypatch, capsys)
    assert "AdamW, bias correction off" in out
    home = course_home / "25-sft-in-harnesses"
    assert not (home / "sft").exists()  # the default adapter is never replaced
    note = json.loads((home / "sft-no-bias-correction" / "adapter.json").read_text())
    assert note["bias_correction"] is False


def test_without_bias_correction_the_first_steps_are_larger() -> None:
    rows = [
        training.item_from_sequence(r["sequences"][0])
        for r in map(json.loads, CAPTURED.read_text().splitlines())
    ]

    def moved(bias_correction: bool) -> float:
        model = adapted(LFM_VOCAB)
        before = {k: v for k, v in lora.adapter_weights(model).items()}
        training.train_sft(
            model,
            [r for r in rows if r],
            1,
            1e-3,
            2,
            lambda step: None,
            bias_correction=bias_correction,
        )
        after = lora.adapter_weights(model)
        return max(mx.max(mx.abs(after[k] - before[k])).item() for k in before)

    # One step: corrected Adam moves each weight by about lr; uncorrected by
    # (1 - 0.9) / sqrt(1 - 0.999) = 3.16 x lr.
    assert moved(False) == pytest.approx(3.16 * moved(True), rel=0.05)


def test_train_sft_skips_a_row_over_the_cap_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    use_tiny_trainer(monkeypatch, tmp_path)
    argv = ["train_sft.py", "--demos", str(CAPTURED), "--max-tokens", "800"]
    out = run_main(train_sft, argv, monkeypatch, capsys)
    assert "rows      1 kept · 1 over 800 tokens skipped" in out
    assert "[sft] step 1/1 · loss/token" in out and "147 tokens" in out


def test_train_sft_refuses_held_out_rows_and_never_leaves_an_old_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path, course_home: Path
) -> None:
    use_tiny_trainer(monkeypatch, tmp_path)
    pi = tmp_path / "pi.jsonl"
    records = [json.loads(line) for line in CAPTURED.read_text().splitlines()]
    pi.write_text(json.dumps({**records[0], "harness": "pi"}) + "\n")
    with pytest.raises(SystemExit, match="from pi, not a training harness"):
        run_main(train_sft, ["train_sft.py", "--demos", str(pi)], monkeypatch, capsys)
    folder = course_home / "25-sft-in-harnesses" / "sft"
    folder.mkdir(parents=True)
    (folder / "adapter.safetensors").write_bytes(b"an older run's")

    def crash(*args: Any, **options: Any) -> None:
        raise RuntimeError("stopped mid-run")

    monkeypatch.setattr(train_sft, "train_sft", crash)
    argv = ["train_sft.py", "--demos", str(CAPTURED)]
    with pytest.raises(RuntimeError, match="stopped mid-run"):
        run_main(train_sft, argv, monkeypatch, capsys)
    assert not (folder / "adapter.safetensors").exists()


def test_train_sft_names_the_command_that_builds_missing_demos(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    use_tiny_trainer(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="uv run collect.py"):
        run_main(train_sft, ["train_sft.py"], monkeypatch, capsys)
    broken = tmp_path / "demos.jsonl"
    broken.write_text("{not json\n")
    with pytest.raises(SystemExit, match="demos.jsonl:1 is not JSON"):
        argv = ["train_sft.py", "--demos", str(broken)]
        run_main(train_sft, argv, monkeypatch, capsys)


@pytest.mark.skipif(not REAL_DEMOS.is_file(), reason="collect.py has not run here")
def test_train_sft_trains_on_this_machines_claude_code_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    records = [json.loads(line) for line in REAL_DEMOS.read_text().splitlines()]
    claude = [r for r in records if r["harness"] == "claude-code"][:2]
    if not claude:
        pytest.skip("no Claude Code demonstration captured here")
    path = tmp_path / "claude.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in claude))
    use_tiny_trainer(monkeypatch, tmp_path)
    argv = ["train_sft.py", "--demos", str(path)]
    out = run_main(train_sft, argv, monkeypatch, capsys)
    assert f"[sft] {len(claude)} training rows from {len(claude)} demos" in out
    assert "claude-code" in out and "[sft] step 1/1" in out


# ------------------------------------------------------------- the engine's adapter


def test_the_engine_model_switches_adapters_and_back_to_the_exact_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lfm, "load", lambda folder: (tiny_model(), FakeTokenizer()))
    model = lfm.LfmModel(tmp_path, seed=0)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    base = model.model(tokens)
    assert model.set_adapter(None) == "base" and not model.adapted
    trained = adapted()
    nudge(trained)
    saved = lora.save_adapter(trained, tmp_path / "a.safetensors")
    label = model.set_adapter(saved)
    assert label.endswith("a.safetensors") and model.adapted
    assert not mx.allclose(model.model(tokens), base).item()
    assert model.set_adapter(None) == "base"
    assert mx.array_equal(model.model(tokens), base).item()  # B = 0 adds exactly 0


def test_an_unreadable_or_retyped_adapter_file_is_refused(tmp_path: Path) -> None:
    model = adapted()
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="cannot be read as an adapter"):
        lora.check_adapter_file(model, junk)
    retyped = {
        name: value.astype(mx.float16)
        for name, value in lora.adapter_weights(model).items()
    }
    other = tmp_path / "float16.safetensors"
    mx.save_safetensors(str(other), retyped)
    with pytest.raises(ValueError, match="float16"):
        lora.check_adapter_file(model, other)


def test_an_adapter_of_another_shape_is_refused_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lfm, "load", lambda folder: (tiny_model(), FakeTokenizer()))
    model = lfm.LfmModel(tmp_path, seed=0)
    other = adapted(rank=4)
    nudge(other)
    wrong = lora.save_adapter(other, tmp_path / "rank4.safetensors")
    with pytest.raises(ValueError, match="not"):
        model.set_adapter(wrong)
    tokens = mx.array([[1, 2, 3]])
    assert mx.array_equal(model.model(tokens), tiny_model()(tokens)).item()
    with pytest.raises(FileNotFoundError):
        model.set_adapter(tmp_path / "missing.safetensors")
    few = tmp_path / "few.safetensors"
    mx.save_safetensors(str(few), {"model.layers.3.x": mx.zeros((1,))})
    with pytest.raises(ValueError, match="holds 1 arrays"):
        model.set_adapter(few)


# ------------------------------------------------------------------ memory_probe.py


SHAPE = memory_probe.Shape(layers=30, hidden=2048, heads=32, vocab=128_000)
GB = 1e9


def test_a_projection_scales_the_memory_above_the_weights_by_its_growth() -> None:
    fused, chunked = memory_probe.BY_NAME["fused"], memory_probe.BY_NAME["chunked"]
    history = [(2048, 3.0 * GB)]
    projected, basis = memory_probe.project(fused, 4096, history, 1.0 * GB, SHAPE)
    assert projected == pytest.approx(1.0 * GB + 2.0 * GB * 4)
    assert basis == "from the 2,048-token pass, x4.00 above the weights"
    history = [(6144, 7.2 * GB)]
    projected, _ = memory_probe.project(chunked, 8192, history, 1.6 * GB, SHAPE)
    assert projected == pytest.approx(1.6 * GB + 5.6 * GB * 8192 / 6144)


def test_a_first_projection_is_a_generous_count_and_grows_with_the_row() -> None:
    for setting in memory_probe.SETTINGS:
        small, basis = memory_probe.project(setting, 1024, [], 1.6 * GB, SHAPE)
        large, _ = memory_probe.project(setting, 2048, [], 1.6 * GB, SHAPE)
        assert basis == memory_probe.NO_MEASUREMENT
        assert 1.6 * GB < small < memory_probe.BUDGET_GB * GB < large * 2
    fused = memory_probe.first_bound(memory_probe.BY_NAME["fused"], 4096, 0, SHAPE)
    chunked = memory_probe.first_bound(memory_probe.BY_NAME["chunked"], 4096, 0, SHAPE)
    assert fused > chunked  # L x L scores against L x 512


def test_each_reason_to_refuse_is_said_and_a_fit_says_nothing() -> None:
    fused = memory_probe.BY_NAME["fused"]
    refuse = memory_probe.refusal
    assert refuse(fused, 4096, 6 * GB, 9 * GB, 8 * GB, 2 * GB) == ""
    capped = refuse(fused, 6144, 6 * GB, 9 * GB, None, 2 * GB)
    assert capped == "fused attention runs only up to 4,096 tokens"
    over = refuse(memory_probe.BY_NAME["chunked"], 8192, 9.03 * GB, 9 * GB, None, 0)
    assert over == "projected 9.03 GB, over the 9.00 GB budget"
    busy = refuse(memory_probe.BY_NAME["chunked"], 2048, 5 * GB, 9 * GB, 1 * GB, 2 * GB)
    assert busy == "it would need 3.0 GB more, and the machine has 1.0 GB free now"
    chunked = memory_probe.BY_NAME["chunked"]
    assert refuse(chunked, 2048, 5 * GB, 9 * GB, 4 * GB, 2 * GB) == ""
    kept = refuse(chunked, 2048, 5 * GB, 9 * GB, 4 * GB, 2 * GB, reserve=1.5 * GB)
    assert kept == (
        "it would need 3.0 GB more, and the machine has 4.0 GB free now, "
        "1.5 GB of it kept for the rest"
    )


def test_the_projects_figures_label_the_cells_it_measured() -> None:
    by = memory_probe.BY_NAME
    assert memory_probe.recorded_for(by["fused"], 6144) == 11.9
    assert memory_probe.recorded_for(by["fused"], 8192) is None
    assert memory_probe.recorded_for(by["chunked"], 8192) == 9.7
    assert memory_probe.recorded_for(by["full-logits"], 6144) is None


def test_free_memory_is_macos_own_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_probe, "machine_bytes", lambda: 20 * GB)
    text = (
        "The system has 19327352832 (1179648 pages with a page size of 16384).\n"
        "...\n"
        "System-wide memory free percentage: 58%\n"
    )
    assert memory_probe.free_bytes(lambda: text) == pytest.approx(11.6 * GB)
    assert memory_probe.free_bytes(lambda: "no percentage here") is None


def test_the_engine_is_found_by_its_pid_file_only(tmp_path: Path) -> None:
    record = tmp_path / "run" / "engine.pid"
    assert memory_probe.engine_running(tmp_path) is None
    record.parent.mkdir()
    record.write_text(str(os.getpid()))
    assert memory_probe.engine_running(tmp_path) == os.getpid()
    command = record.with_suffix(".cmd")
    command.write_text("uv run engine.py --port 8129\n")  # another program now
    assert memory_probe.engine_running(tmp_path) is None
    command.unlink()
    record.write_text("not a pid")
    assert memory_probe.engine_running(tmp_path) is None


def test_a_row_projected_over_budget_after_a_measured_pass_is_never_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = adapted()
    baseline = float(mx.get_active_memory())
    ran: list[int] = []

    def fake_pass(model: nn.Module, item: Any, setting: Any) -> tuple[float, float]:
        ran.append(len(item.input_ids))
        # 1 GB above the weights per 16 tokens: linear, as the setting grows
        return baseline + len(item.input_ids) / 16 * GB, 0.1

    monkeypatch.setattr(memory_probe, "one_pass", fake_pass)
    lines: list[str] = []
    budget = baseline + 2.5 * GB  # 32 tokens projects +2 GB: fits; 64, +4 GB: not
    outcomes = memory_probe.run_setting(
        model,
        memory_probe.BY_NAME["chunked"],
        [16, 32, 64],
        budget,
        baseline,
        lines.append,
        free=lambda: None,
    )
    assert ran == [16, 32]
    assert [o.refused == "" for o in outcomes] == [True, True, False]
    assert "over the" in outcomes[2].refused and "refused ·" in lines[2]


def test_a_pass_that_fails_is_recorded_and_no_longer_row_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = adapted()
    baseline = float(mx.get_active_memory())

    def failing(model: nn.Module, item: Any, setting: Any) -> tuple[float, float]:
        if len(item.input_ids) > 16:
            raise RuntimeError("[metal::malloc] Resource limit exceeded")
        return baseline + 0.01 * GB, 0.1

    monkeypatch.setattr(memory_probe, "one_pass", failing)
    kept: list[memory_probe.Outcome] = []
    lines: list[str] = []
    outcomes = memory_probe.run_setting(
        model,
        memory_probe.BY_NAME["chunked"],
        [16, 32, 64],
        100 * GB,
        baseline,
        lines.append,
        free=lambda: None,
        on_outcome=kept.append,
    )
    assert kept == outcomes
    assert outcomes[1].refused.startswith("failed: [metal::malloc]")
    assert outcomes[2].refused == "not run: the 32-token pass failed"
    cells = [memory_probe.cell(o) for o in outcomes]
    assert cells[0].endswith(" GB") and cells[1:] == ["failed", "refused"]
    assert "FAILED" in lines[1]


def test_the_probe_measures_refuses_and_writes_its_results(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path, course_home: Path
) -> None:
    monkeypatch.setattr(memory_probe, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(memory_probe, "load_model", lambda folder: adapted())
    monkeypatch.setattr(memory_probe, "free_bytes", lambda read=None: 64 * GB)
    argv = ["memory_probe.py", "--lengths", "24,48,6144", "--settings", "fused"]
    out = run_main(memory_probe, argv, monkeypatch, capsys)
    assert re.search(r"\[fused +24\] projected .* · peak ", out)
    assert re.search(r"\[fused +48\] projected .*x4\.00 above the weights", out)
    capped = r"\[fused +6,144\] refused · fused attention runs only up to 4,096"
    assert re.search(capped, out)
    assert "the project measured 11.9 GB here" in out
    assert "11.9*" in out
    folder = course_home / "25-sft-in-harnesses" / "memory_probe"
    results = json.loads((folder / "results.json").read_text())
    measured = [o for o in results["outcomes"] if o["peak_gb"] is not None]
    assert [o["length"] for o in measured] == [24, 48]
    assert results["budget_gb"] == memory_probe.BUDGET_GB


def test_the_probe_refuses_everything_over_a_tiny_budget_without_running(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory_probe, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(memory_probe, "load_model", lambda folder: adapted())

    def never(*args: Any) -> tuple[float, float]:
        raise AssertionError("a refused pass ran")

    monkeypatch.setattr(memory_probe, "one_pass", never)
    argv = ["memory_probe.py", "--lengths", "24,48", "--budget-gb", "0.000001"]
    with pytest.raises(SystemExit) as stopped:
        run_main(memory_probe, argv, monkeypatch, capsys)
    assert stopped.value.code == 1
    out = capsys.readouterr().out
    assert out.count("refused ·") == 6  # 3 settings x 2 lengths
    assert "over the 0.00 GB budget" in out
    assert "nothing was measured: every row was refused or failed" in out


def test_the_probe_will_not_start_beside_this_chapters_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, course_home: Path
) -> None:
    record = course_home / "25-sft-in-harnesses" / "run" / "engine.pid"
    record.parent.mkdir(parents=True)
    record.write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="./start.sh stop first"):
        run_main(memory_probe, ["memory_probe.py"], monkeypatch, capsys)
