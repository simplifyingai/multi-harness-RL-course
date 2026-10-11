"""Chapter 14 with MLX: scoring, the row-by-row step, training, and the scripts.

No test here loads the real model, downloads anything or uses the network.
Training runs on a tiny LFM2 built with mlx-lm's own ``lfm2`` code (64
numbers wide, four layers) and quantized to 4 bits as chapter 11 did, with
chapter 13's adapters on it; the tokenizer is a fake that renders LFM2.5's
chat markers and maps every other character to an id the tiny model knows.
The three scripts run end to end on them. MLX installs only on Apple-silicon
Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import json
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

CHAPTER = use_chapter("14-sft-real-model")

import eval_sql  # noqa: E402
import lfm  # noqa: E402
import lfm_files  # noqa: E402
import lora  # noqa: E402
import make_rows  # noqa: E402
import sft  # noqa: E402
import train_sft  # noqa: E402
from rows import TrainItem  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
FIRST_PLAIN = 8  # ids below this are the chat markers
HIDDEN = 64
GROUP_SIZE = 32
BITS = 4
RANK = 4
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
        """Write markers back, and every other id as a letter.

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        names = {token: marker for marker, token in MARKERS.items()}
        return "".join(names[i] if i in names else chr(ord("a") + i % 26) for i in ids)


def tiny_model(seed: int = 0) -> nn.Module:
    """Build a tiny 4-bit LFM2 with mlx-lm's own code.

    Args:
        seed: The random seed for the weights.

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def item(ids: list[int], start: int) -> TrainItem:
    """Make a training row whose tokens from ``start`` on are trained.

    Args:
        ids: The row's tokens.
        start: The first trained position.

    Returns:
        The row.
    """
    positions = list(range(start, len(ids)))
    return TrainItem(ids, positions, [0.0] * len(positions))


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
    """Make every script load the tiny model and the fake tokenizer.

    Args:
        monkeypatch: pytest's patcher.
        tmp_path: A folder to name as the build's.
    """

    def load(folder: Path) -> lfm.LoadedModel:
        """Give a fresh tiny model, as loading the build would.

        Args:
            folder: The build's folder.

        Returns:
            The tiny model and the fake tokenizer.
        """
        return lfm.LoadedModel(tiny_model(), FakeTokenizer(), folder, 0.0)

    for module in (train_sft, eval_sql, make_rows):
        monkeypatch.setattr(module, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(train_sft, "load_folder", load)
    monkeypatch.setattr(eval_sql, "load_folder", load)
    monkeypatch.setattr(make_rows, "load_tokenizer_only", lambda f: FakeTokenizer())


def plain(out: str) -> str:
    """Undo the line wrapping of printed output.

    Args:
        out: Printed text.

    Returns:
        The same words, single-spaced.
    """
    return " ".join(out.split())


# ---------------------------------------------------------------- scoring


def test_each_token_is_scored_from_the_position_before_it() -> None:
    """token_logprobs equals the full scores' log-softmax at p - 1, at token p."""
    model = tiny_model()
    ids = [1, 9, 20, 33, 12, 40, 41, EOS]
    row = item(ids, 4)
    picked = sft.token_logprobs(model, row).tolist()
    logits = sft.all_logits(model, ids)
    full = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for value, position in zip(picked, row.positions, strict=True):
        expected = full[position - 1, ids[position]].item()
        assert value == pytest.approx(expected, abs=1e-4)


def test_a_replys_score_is_the_sum_of_its_tokens() -> None:
    """reply_logprob scores the reply's tokens after the prompt; empty is 0."""
    model = tiny_model()
    prompt, reply = [1, 9, 20], [33, 12, EOS]
    whole = sft.token_logprobs(model, item(prompt + reply, 3)).sum().item()
    assert sft.reply_logprob(model, prompt, reply) == pytest.approx(whole, abs=1e-5)
    assert sft.reply_logprob(model, prompt, []) == 0.0


def test_row_by_row_gradients_equal_the_whole_steps_gradient() -> None:
    """Each row's gradient over the step's tokens, summed: the loss per token's."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    rows = [item([1, 9, 20, 33, 12, EOS], 3), item([1, 11, 40, 41, 42, 43, 44, EOS], 2)]
    total = sum(row.n_tokens for row in rows)

    def one(row: TrainItem) -> mx.array:
        """One row's loss.

        Args:
            row: The row.

        Returns:
            Minus its summed log-probability.
        """
        return -sft.token_logprobs(model, row).sum()

    def both(_: Any) -> mx.array:
        """The step's loss per token, over both rows at once.

        Args:
            _: Unused.

        Returns:
            The summed loss divided by the token count.
        """
        return (one(rows[0]) + one(rows[1])) / total

    summed = None
    for row in rows:
        _, grads = nn.value_and_grad(model, one)(row)
        grads = dict(tree_flatten(grads))
        scaled = {name: grad / total for name, grad in grads.items()}
        if summed is None:
            summed = scaled
        else:
            summed = {name: summed[name] + scaled[name] for name in summed}
    _, joint = nn.value_and_grad(model, both)(None)
    for name, grad in tree_flatten(joint):
        assert mx.allclose(summed[name], grad, atol=1e-5).item(), name


def test_batches_shuffle_by_epoch_and_cover_every_row() -> None:
    """The same seed, the same order; every row once; the last batch may be short."""
    rows = [item([1, i, EOS], 1) for i in range(10, 18)]
    first = list(sft.batches(rows, 3, seed=0))
    assert [len(batch) for batch in first] == [3, 3, 2]
    assert first == list(sft.batches(rows, 3, seed=0))
    flat = [row.input_ids[1] for batch in first for row in batch]
    assert sorted(flat) == list(range(10, 18))
    assert sft.steps_per_epoch(180, 8) == 23
    assert sft.steps_per_epoch(40, 8) == 5


def test_training_lowers_the_loss_and_moves_only_the_adapter() -> None:
    """Six steps at a high rate: the loss falls; only A and B change."""
    model = tiny_model()
    lora.apply_lora(model, RANK, 2.0, 2)
    rows = [item([1, 9, 20, 33, 12, 40, EOS], 3) for _ in range(4)]
    before = dict(tree_flatten(model.parameters()))
    steps: list[sft.Step] = []
    sft.train_sft(model, rows, 3, 1e-2, 2, steps.append)
    assert [step.step for step in steps] == [1, 2, 3, 4, 5, 6]
    assert [step.epoch for step in steps] == [1, 1, 2, 2, 3, 3]
    assert all(step.tokens == 8 and step.rows == 2 for step in steps)
    assert steps[-1].loss_per_token < steps[0].loss_per_token
    after = dict(tree_flatten(model.parameters()))
    moved = {name for name in before if not mx.array_equal(before[name], after[name])}
    assert moved and all(name.endswith(("lora_a", "lora_b")) for name in moved)


def test_a_fresh_adapter_changes_nothing_and_zeroing_restores_the_base(
    tmp_path: Path,
) -> None:
    """B starts at zero; after training, zeroing B gives the base scores back."""
    model = tiny_model()
    ids = [1, 9, 20, 33, 12, 40, EOS]
    base = sft.all_logits(model, ids)
    lora.apply_lora(model, RANK, 2.0, 2)
    assert sft.max_difference(sft.all_logits(model, ids), base) == 0.0
    zeros, count = lora.zero_b_count(model)
    assert zeros == count > 0
    sft.train_sft(model, [item(ids, 3)], 2, 1e-2, 1, lambda step: None)
    trained = sft.all_logits(model, ids)
    path = lora.save_adapter(model, tmp_path / "a.safetensors")
    lora.zero_adapter(model)
    assert sft.max_difference(sft.all_logits(model, ids), base) == 0.0
    lora.load_adapter(model, path)
    assert sft.max_difference(sft.all_logits(model, ids), trained) == 0.0
    with pytest.raises(FileNotFoundError):
        lora.load_adapter(model, tmp_path / "missing.safetensors")


def test_greedy_replies_end_at_the_end_of_turn_or_the_cap() -> None:
    """A reply stops at <|im_end|> ('stop') or at the cap ('length')."""
    model = tiny_model()
    reply = lfm.greedy_reply(model, FakeTokenizer(), [1, 9, 20], 5)
    assert len(reply.tokens) <= 5
    if reply.finish_reason == lfm.FINISH_STOP:
        assert reply.tokens[-1] == EOS
    else:
        assert reply.finish_reason == lfm.FINISH_LENGTH and len(reply.tokens) == 5
    assert reply.seconds >= 0


# ---------------------------------------------------------------- the scripts


def test_make_rows_writes_both_splits_and_shows_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """180 and 50 rows written; row 1 shown with its boundary and the checks."""
    use_tiny(monkeypatch, tmp_path)
    out = run_main(make_rows, "--show", "1")
    text = plain(out)
    headings = ("1. the questions: 230", "2. each question", "3. the rows", "4. row 1")
    for heading in headings:
        assert heading in out
    assert "nothing comes before position 0" in text
    assert "rendering the finished chat with the template instead" in text
    home = course_home / "14-sft-real-model"
    assert len((home / "rows_train.jsonl").read_text().splitlines()) == 180
    assert len((home / "rows_test.jsonl").read_text().splitlines()) == 50


def test_make_rows_without_show_prints_no_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No --show: the summary only; a row number out of range stops."""
    use_tiny(monkeypatch, tmp_path)
    assert "4. row" not in run_main(make_rows)
    with pytest.raises(SystemExit, match="between 1 and 180"):
        run_main(make_rows, "--show", "181")


def test_train_sft_fast_builds_rows_trains_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """--fast: rows built (make_rows.py never ran), 40 rows, 5 steps, one adapter."""
    use_tiny(monkeypatch, tmp_path)
    out = run_main(train_sft, "--fast", "--lr", "0.01")
    text = plain(out)
    assert "make_rows.py had not run: written to rows_train.jsonl" in text
    assert "40 training rows" in text and "5 AdamW steps" in text
    assert "a fresh adapter built the chapter-13 way" in text
    assert "from the model before LoRA, over row 1's" in text and ": 0.0;" in text
    assert "(--lr; the project's is 5e-05)" in text
    assert "the project's runs (recorded, not run here)" in text
    home = course_home / "14-sft-real-model"
    adapter = home / "sft_lr0.01_limit40.safetensors"
    assert adapter.is_file()
    steps = [
        json.loads(line)
        for line in (home / "sft_lr0.01_limit40.metrics.jsonl").read_text().splitlines()
    ]
    assert [step["step"] for step in steps] == [1, 2, 3, 4, 5]
    assert all(step["stage"] == "sft" for step in steps)


def test_train_sft_epochs_print_an_epoch_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two epochs over 8 rows: 2 steps, and a line per epoch."""
    use_tiny(monkeypatch, tmp_path)
    run_main(make_rows)
    out = run_main(train_sft, "--limit", "8", "--epochs", "2", "--lr", "0.01")
    assert "identical to make_rows.py's rows_train.jsonl" in plain(out)
    assert "loss per token, epoch by epoch" in out
    assert "epoch 1:" in out and "epoch 2:" in out


def test_train_sft_never_trains_on_an_old_rows_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """A rows file that differs, or cannot be read, is rebuilt and said so."""
    use_tiny(monkeypatch, tmp_path)
    path = course_home / "14-sft-real-model" / "rows_train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"task_id": "old", "input_ids": [1, 2], "loss_mask": [0, 1]}\n')
    out = plain(run_main(train_sft, "--limit", "8", "--lr", "0.01"))
    assert "rows_train.jsonl was out of date and is rewritten" in out
    assert len(path.read_text().splitlines()) == 180
    path.write_text("{broken\n")
    out = plain(run_main(train_sft, "--limit", "8", "--lr", "0.01"))
    assert "rows_train.jsonl could not be read and is rewritten" in out


@pytest.mark.parametrize("args", [["--epochs", "0"], ["--lr", "0"], ["--limit", "0"]])
def test_train_sft_refuses_impossible_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    """Impossible settings stop before any model is loaded."""
    use_tiny(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        run_main(train_sft, *args)


def test_eval_sql_base_then_adapter_compares_question_by_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """The base run is saved; the adapter run reads it, compares, and tries prompts."""
    use_tiny(monkeypatch, tmp_path)
    base = run_main(eval_sql, "--limit", "3", "--cap", "4")
    assert "base: pass@1 0 of 3 = 0.0% ± 0.0 points" in base
    assert "a few replies" in base
    home = course_home / "14-sft-real-model"
    saved = home / "eval_base.jsonl"
    assert len(saved.read_text().splitlines()) == 3
    model = tiny_model()
    lora.apply_lora(
        model, lfm_files.LORA_RANK, lfm_files.LORA_SCALE, lfm_files.LORA_LAYERS
    )
    lora.save_adapter(model, home / "tiny.safetensors")
    args = ("--adapter", "tiny.safetensors", "--limit", "3", "--cap", "4")
    out = run_main(eval_sql, *args)
    text = plain(out)
    assert "base answers: read from eval_base.jsonl" in text
    assert "tiny: pass@1" in text
    assert "base against tiny, question by question" in text
    assert "what else the adapter changed" in text
    assert text.count("before training") >= 1
    assert (home / "eval_tiny.jsonl").is_file()
    assert (home / "work" / "tiny").is_dir()


def test_eval_sql_answers_with_the_base_first_when_it_has_no_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """--adapter without a saved base run: the fresh adapter answers first."""
    use_tiny(monkeypatch, tmp_path)
    model = tiny_model()
    lora.apply_lora(
        model, lfm_files.LORA_RANK, lfm_files.LORA_SCALE, lfm_files.LORA_LAYERS
    )
    home = course_home / "14-sft-real-model"
    lora.save_adapter(model, home / "tiny.safetensors")
    out = run_main(eval_sql, "--fast", "--adapter", "tiny.safetensors", "--cap", "2")
    text = plain(out)
    assert "no saved base answers to these questions under a 2-token cap" in text
    assert "base: pass@1 0 of 10" in text
    assert "skipped with --fast" in text


def test_eval_sql_stops_when_the_adapter_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A name that is not a file stops with the command that makes one."""
    use_tiny(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="run train_sft.py first"):
        run_main(eval_sql, "--adapter", "nothing.safetensors")
    with pytest.raises(SystemExit):
        run_main(eval_sql, "--cap", "0")


def test_eval_sql_does_not_reuse_base_answers_made_under_another_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """A base run at --cap 4 is not compared with an adapter run at --cap 2."""
    use_tiny(monkeypatch, tmp_path)
    run_main(eval_sql, "--limit", "2", "--cap", "4")
    model = tiny_model()
    lora.apply_lora(
        model, lfm_files.LORA_RANK, lfm_files.LORA_SCALE, lfm_files.LORA_LAYERS
    )
    home = course_home / "14-sft-real-model"
    lora.save_adapter(model, home / "tiny.safetensors")
    args = ("--adapter", "tiny.safetensors", "--fast", "--limit", "2", "--cap", "2")
    text = plain(run_main(eval_sql, *args))
    assert "no saved base answers to these questions under a 2-token cap" in text
    saved = (home / "eval_base.jsonl").read_text().splitlines()
    assert {json.loads(line)["cap"] for line in saved} == {2}


def test_train_sft_with_the_fixed_reason_trains_its_own_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """--reason fixed: the teacher's sentence, its own rows file and adapter."""
    use_tiny(monkeypatch, tmp_path)
    out = plain(run_main(train_sft, "--reason", "fixed", "--limit", "8"))
    assert "Now I can answer the question with one query." in out
    home = course_home / "14-sft-real-model"
    assert (home / "rows_train_fixed.jsonl").is_file()
    assert not (home / "rows_train.jsonl").exists()
    assert (home / "sft_reasonfixed_limit8.safetensors").is_file()
    shown = plain(run_main(make_rows, "--reason", "fixed", "--show", "2"))
    assert "one fixed sentence for every row" in shown and "4. row 2" in shown
