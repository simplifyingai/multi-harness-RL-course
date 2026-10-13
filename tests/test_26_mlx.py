"""Chapter 26 with MLX: the GRPO loop on a tiny model, never the 2.6B.

The trainer is chapter 15's tiny LFM2 (mlx-lm's own ``lfm2`` code, 64
numbers wide, four layers, quantized to 4 bits) with chapter 13's adapter.
The engine is a stand-in with its OWN copy of the same tiny model: it
"samples" each rollout's reply from fixed token ids, records each token's
log-probability under its copy, and loads every adapter the loop POSTs, as
the real engine does. So the loop's importance ratio is 1 only if the
engine's and the trainer's weights really are the same at every step. The
rewards are scripted per rollout. ``grpo_harness.py`` then runs end to end
on it: steps, updates, adapters, the length guard, ``--attempt1``.

The last test runs one real step through OpenEnv's capture proxy and
mini-swe-agent in the sandbox, with the scripted engine (chapter 23)
answering and the tiny model training on the captured rows; it is skipped
when the proxy or the harness is not on disk. MLX installs only on
Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import uvicorn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("26-grpo-in-harnesses")

import grpo  # noqa: E402
import grpo_harness  # noqa: E402
import grpo_records  # noqa: E402
import install  # noqa: E402
import lfm_files  # noqa: E402
import lora  # noqa: E402
import rewards  # noqa: E402
import rollouts  # noqa: E402
import run_one  # noqa: E402
import scripted_engine  # noqa: E402
import tasks  # noqa: E402
import training  # noqa: E402
from capture_client import CaptureClient  # noqa: E402

VOCAB = 64
HIDDEN = 64
GROUP_SIZE = 32
BITS = 4
LFM_VOCAB = 128_000  # LFM2.5's vocabulary, for rows the real tokenizer cut
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": VOCAB,
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
    "block_ff_dim": 2 * HIDDEN,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "rope_theta": 1e7,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
}
TRAIN = tasks.load_tasks(tasks.TRAIN_FILE)
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]  # a row's harness text: read, never trained
RIGHT_REPLY = [10, 11, 12, 13]
WRONG_REPLY = [20, 21, 22, 23, 24, 25]
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy, the harnesses and the tokenizer are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
START_SH = CHAPTER / "start.sh"
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
CHAPTERS_WITH_STACK = (
    "26-grpo-in-harnesses",
    "25-sft-in-harnesses",
    "24-measuring",
    "23-three-harnesses",
)
START_TIMEOUT_S = 300


def tiny_model(vocab: int = VOCAB, seed: int = 0) -> nn.Module:
    """Build a tiny 4-bit LFM2 with chapter 13's adapter on it.

    Args:
        vocab: Its vocabulary size.
        seed: The random seed for the weights.

    Returns:
        The model, only its adapter trainable.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**{**TINY_CONFIG, "vocab_size": vocab}))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    lora.apply_lora(model, lora.LORA_RANK, lora.LORA_SCALE, lora.LORA_LAYERS)
    mx.eval(model.parameters())
    return model


class FakeEngine:
    """The engine's side of the loop: its own model copy, and the rewards.

    Attributes:
        model: The engine's copy of the tiny model.
        switches: Every adapter the loop told it to use.
        reply_for: Picks a rollout's reply ids and verdict from its job.
    """

    def __init__(self, reply_for: Any = None) -> None:
        """Build the engine's copy.

        Args:
            reply_for: ``reply_for(job) -> (reply ids, correct)``; the default
                makes the first two rollouts of a group right.
        """
        self.model = tiny_model()
        self.switches: list[Path | None] = []
        self.reply_for = reply_for or default_reply

    def use_adapter(self, path: Path | None) -> str:
        """Load an adapter into the engine's copy, as POST /admin/adapter does.

        Args:
            path: The adapter, or None for the base model.

        Returns:
            Its label.
        """
        self.switches.append(path)
        if path is None:
            lora.zero_adapter(self.model)
            return "base"
        lora.load_adapter(self.model, path)
        return path.name

    def logprobs(self, ids: list[int], start: int) -> list[float]:
        """Each token's log-probability under the engine's copy, from ``start``.

        Args:
            ids: The row.
            start: The first sampled position.

        Returns:
            One per position (0 before ``start``).
        """
        positions = list(range(start, len(ids)))
        item = training.TrainItem(ids, positions, [0.0] * len(positions))
        scored = training.token_logprobs(self.model, item).tolist()
        return [0.0] * start + scored

    def run_rollouts(self, jobs: list[rollouts.Job]) -> list[rollouts.RolloutRecord]:
        """Make each rollout's record: one row, the reply's tokens trained.

        Args:
            jobs: The step's rollouts.

        Returns:
            Their records, rewarded with the token bonus.
        """
        records = []
        for job in jobs:
            reply, correct = self.reply_for(job)
            ids = PROMPT + reply
            row = {
                "input_ids": ids,
                "loss_mask": [0] * len(PROMPT) + [1] * len(reply),
                "logprobs": self.logprobs(ids, len(PROMPT)),
            }
            records.append(
                rollouts.RolloutRecord(
                    task_id=job.task.task_id,
                    harness=job.harness,
                    correct=correct,
                    n_calls=1,
                    n_sampled=len(reply),
                    reward=rewards.reward_for(correct, len(reply)),
                    trainable=True,
                    n_trainable_tokens=len(reply),
                    sequences=[row],
                    step=job.step,
                    group=job.group,
                )
            )
        return records

    def hooks(self) -> grpo_harness.Hooks:
        """The loop's hooks, pointed at this engine.

        Returns:
            The hooks.
        """
        return grpo_harness.Hooks(self.run_rollouts, self.use_adapter)


def default_reply(job: rollouts.Job) -> tuple[list[int], bool]:
    """Two right rollouts and two wrong ones in every group.

    Args:
        job: The rollout.

    Returns:
        Its reply ids and verdict.
    """
    right = job.index < 2
    return (RIGHT_REPLY if right else WRONG_REPLY), right


def plan(steps: int) -> list[list[tuple[tasks.Task, str]]]:
    """The project's task order for a few steps.

    Args:
        steps: How many.

    Returns:
        The plan.
    """
    return rollouts.plan_groups(TRAIN, rollouts.TRAIN_HARNESSES, steps, 2)


def lines_of(found: list[str]) -> str:
    """Join what the loop said.

    Args:
        found: Its lines.

    Returns:
        The text.
    """
    return "\n".join(found)


def run_loop(
    tmp_path: Path, steps: int, engine: FakeEngine, config: grpo_harness.Config
) -> tuple[grpo_harness.Outcome, list[str], nn.Module]:
    """Run the loop on the tiny model against the fake engine.

    Args:
        tmp_path: pytest's folder; the run goes in ``tmp_path/run``.
        steps: How many steps.
        engine: The engine.
        config: The run's settings.

    Returns:
        How it ended, what it said, and the trainer's model.
    """
    model = tiny_model()
    said: list[str] = []
    outcome = grpo_harness.train(
        replace(config, steps=steps),
        model,
        plan(steps),
        engine.hooks(),
        tmp_path / "run",
        None,
        said.append,
    )
    return outcome, said, model


def metrics(tmp_path: Path) -> list[dict[str, Any]]:
    """The run's metrics lines.

    Args:
        tmp_path: pytest's folder.

    Returns:
        Every line of ``run/metrics.jsonl``.
    """
    text = (tmp_path / "run" / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


# ------------------------------------------------------------------ the rows


def forked_record(reward: float, harness: str = "mini-swe-agent") -> dict[str, Any]:
    """A rollout that forked: its second row repeats the first's first reply.

    Args:
        reward: Its reward.
        harness: Its harness.

    Returns:
        The record, as rollouts.py writes it.
    """
    first = {
        "input_ids": PROMPT + [10, 11, 12],
        "loss_mask": [0] * len(PROMPT) + [1, 1, 1],
        "logprobs": [0.0] * (len(PROMPT) + 3),
    }
    second = {
        "input_ids": PROMPT + [10, 11, 12, 40, 41] + [13, 14],
        "loss_mask": [0] * len(PROMPT) + [1, 1, 1, 0, 0, 1, 1],
        "logprobs": [0.0] * (len(PROMPT) + 7),
    }
    return {
        "task_id": "t",
        "harness": harness,
        "correct": reward > 0,
        "n_calls": 2,
        "n_sampled": 5,
        "reward": reward,
        "seconds": 1.0,
        "sequences": [first, second],
    }


def test_step_rows_train_each_shared_prefix_once_and_skip_flat_groups() -> None:
    contrast = [forked_record(r) for r in (1.0, 0.0, 1.0, 0.0)]
    flat = [forked_record(0.0, "claude-code") for _ in range(4)]
    rows = grpo.step_rows([contrast, flat])
    assert rows.groups == 2 and rows.flat_groups == 1
    # Each rollout: 3 tokens in its first row, then only the 2 new ones.
    assert [item.n_tokens for item in rows.items] == [3, 2] * 4
    assert rows.tokens == 20 and len(rows.items) == 8
    assert rows.by_harness["claude-code"].rows == 0
    capped = grpo.step_rows([contrast], max_tokens=len(PROMPT) + 3)
    assert capped.skipped_long == 4 and len(capped.items) == 4


def test_the_records_reading_counts_the_rows_the_trainer_trains() -> None:
    contrast = [forked_record(r) for r in (1.0, 0.0, 1.0, 0.0)]
    flat = [forked_record(0.0, "claude-code") for _ in range(4)]
    rows = grpo.step_rows([contrast, flat])
    step = [grpo_records.from_slim(grpo_records.slim(r, 1)) for r in contrast + flat]
    read = grpo_records.exposure(step)
    for harness, given in rows.by_harness.items():
        assert read[harness].rows == given.rows
        assert read[harness].trained == given.trained
        assert read[harness].tokens == given.tokens


def test_dedupe_is_chapter_21s_unchanged() -> None:
    ours = (CHAPTER / "grpo.py").read_text()
    theirs = (CHAPTER.parent / "21-calls-to-rows" / "training_rows.py").read_text()
    start = theirs.index("    seen: list[list[int]] = []")
    body = theirs[start : theirs.index("    return out", start)]
    assert body in ours


# ----------------------------------------------------------------- the loop


def test_each_step_updates_saves_and_moves_the_engine_to_the_new_adapter(
    tmp_path: Path,
) -> None:
    engine = FakeEngine()
    outcome, said, _ = run_loop(tmp_path, 3, engine, grpo_harness.Config())
    run = tmp_path / "run"
    names = [f"adapter_step00{n}.safetensors" for n in (1, 2, 3)]
    assert [p.name for p in engine.switches] == names
    for name in names:
        lora.check_adapter_file(tiny_model(), run / name)  # the engine's shape
    assert outcome.final == run / "adapter_final.safetensors"
    assert outcome.stopped_at is None and outcome.steps_run == 3
    lines = metrics(tmp_path)
    assert [m["step"] for m in lines] == [1, 2, 3]
    for line in lines:
        # Same weights in the engine and the trainer at every step: ratio 1.
        assert abs(line["importance_ratio_mean"] - 1.0) < 1e-4
        assert line["accuracy"] == 0.5 and line["groups_without_contrast"] == 0
        assert line["train_rows"] == 8 and line["skipped_long_rows"] == 0
        assert line["train_tokens"] == 4 * len(RIGHT_REPLY) + 4 * len(WRONG_REPLY)
        assert line["tokens_per_call"] == 5.0  # the median of 4, 4, 6, 6
        assert line["tokens_per_call_mean"] == 5.0
        assert set(line["by_harness"]) == {"mini-swe-agent", "claude-code"}
    text = lines_of(said)
    assert re.search(
        r"\[grpo\] step 1/3 · 00:00:\d\d · reward 0\.55 · correct 50% · calls 1\.0 · "
        r"no-signal groups 0/2 · ratio 1\.000 · peak \d+\.\d GB",
        text,
    )
    assert "tokens/call: median 5 (the guard reads this; limit 150) · mean 5" in text
    assert "adapter_step003.safetensors, now the engine's" in text
    rows = (run / "rollouts.jsonl").read_text().splitlines()
    assert len(rows) == 24 and json.loads(rows[-1])["step"] == 3


def test_the_update_makes_right_replies_likelier_and_wrong_ones_less(
    tmp_path: Path,
) -> None:
    engine = FakeEngine()
    _, _, model = run_loop(tmp_path, 2, engine, grpo_harness.Config(learning_rate=1e-2))
    before = tiny_model()

    def score(which: nn.Module, reply: list[int]) -> float:
        ids = PROMPT + reply
        positions = list(range(len(PROMPT), len(ids)))
        item = training.TrainItem(ids, positions, [0.0] * len(positions))
        return float(training.token_logprobs(which, item).sum())

    assert score(model, RIGHT_REPLY) > score(before, RIGHT_REPLY)
    assert score(model, WRONG_REPLY) < score(before, WRONG_REPLY)


def test_a_group_without_contrast_is_skipped_and_a_flat_step_makes_no_update(
    tmp_path: Path,
) -> None:
    def all_wrong(job: rollouts.Job) -> tuple[list[int], bool]:
        return WRONG_REPLY, False

    engine = FakeEngine(all_wrong)
    outcome, said, _ = run_loop(tmp_path, 2, engine, grpo_harness.Config())
    assert engine.switches == []  # nothing new to sample with
    lines = metrics(tmp_path)
    assert [m["groups_without_contrast"] for m in lines] == [2, 2]
    assert all(m["adapter"] is None and m["train_rows"] == 0 for m in lines)
    assert "no group had contrast: no update this step" in lines_of(said)
    assert outcome.final is not None and outcome.final.is_file()


def test_a_row_over_the_cap_is_skipped_and_counted(tmp_path: Path) -> None:
    config = grpo_harness.Config(max_tokens=len(PROMPT) + len(RIGHT_REPLY))
    run_loop(tmp_path, 1, FakeEngine(), config)
    (line,) = metrics(tmp_path)
    assert line["skipped_long_rows"] == 4 and line["train_rows"] == 4


def test_the_guard_stops_when_replies_balloon_and_keeps_the_adapter_before(
    tmp_path: Path,
) -> None:
    long_reply = [30 + i % 20 for i in range(200)]  # 200 tokens in one call

    def balloons_at_step_3(job: rollouts.Job) -> tuple[list[int], bool]:
        if job.step >= 3:
            return long_reply, job.index < 2
        return default_reply(job)

    engine = FakeEngine(balloons_at_step_3)
    outcome, said, _ = run_loop(tmp_path, 5, engine, grpo_harness.Config())
    run = tmp_path / "run"
    # Step 3's rollouts came from adapter_step002: keep adapter_step001.
    assert outcome.stopped_at == 3 and outcome.steps_run == 3
    assert outcome.kept == run / "adapter_step001.safetensors"
    assert [p.name for p in engine.switches] == [
        "adapter_step001.safetensors",
        "adapter_step002.safetensors",
        "adapter_step001.safetensors",  # the engine is put back on the kept one
    ]
    final = run / "adapter_final.safetensors"
    assert final.read_bytes() == (run / "adapter_step001.safetensors").read_bytes()
    assert not (run / "adapter_step003.safetensors").exists()
    stop = metrics(tmp_path)[-1]
    assert stop == {
        "stage": "grpo",
        "step": 3,
        "stopped": "length_guard",
        "tokens_per_call": 200.0,
        "kept": "adapter_step001.safetensors",
    }
    assert (
        "[grpo] step 3: replies ballooned to 200 tokens per call (guard 150); "
        "stopping, keeping adapter_step001.safetensors"
    ) in said


def test_the_guard_at_step_1_keeps_the_start(tmp_path: Path) -> None:
    def long_from_the_start(job: rollouts.Job) -> tuple[list[int], bool]:
        return [30] * 160, job.index < 2

    engine = FakeEngine(long_from_the_start)
    start = tmp_path / "start.safetensors"
    lora.save_adapter(tiny_model(), start)
    said: list[str] = []
    outcome = grpo_harness.train(
        grpo_harness.Config(steps=2),
        tiny_model(),
        plan(2),
        engine.hooks(),
        tmp_path / "run",
        start,
        said.append,
    )
    assert outcome.stopped_at == 1 and outcome.kept == start
    assert outcome.final is not None
    assert outcome.final.read_bytes() == start.read_bytes()
    # From the base model there is nothing to keep, as in the project's code.
    outcome = grpo_harness.train(
        grpo_harness.Config(steps=2),
        tiny_model(),
        plan(2),
        engine.hooks(),
        tmp_path / "base-run",
        None,
        said.append,
    )
    assert outcome.stopped_at == 1 and outcome.final is None


def test_attempt1_trains_through_the_balloon_and_says_where_a_guard_would_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py", "--attempt1"])
    config = grpo_harness.config_from(grpo_harness.parse_args())
    assert config.call_bonus and not config.guard and not config.bias_correction
    assert config.learning_rate == 4e-5
    assert config.reward(True, 900, 2) == 1.1  # the call bonus: 2 calls is all
    assert config.reward(True, 0, 8) == 1.0
    long_reply = [30] * 200

    def balloons(job: rollouts.Job) -> tuple[list[int], bool]:
        return (long_reply if job.step >= 2 else RIGHT_REPLY), job.index < 2

    _, said, _ = run_loop(tmp_path, 3, FakeEngine(balloons), config)
    assert [m["step"] for m in metrics(tmp_path)] == [1, 2, 3]
    text = lines_of(said)
    assert text.count("(a guard would stop here: median 200)") == 2
    assert "limit 150, off with --attempt1" in text


def test_bias_correction_off_moves_the_first_step_further(tmp_path: Path) -> None:
    def moved(bias_correction: bool, folder: str) -> float:
        engine = FakeEngine()
        config = grpo_harness.Config(bias_correction=bias_correction, steps=1)
        model = tiny_model()
        start = dict(lora.adapter_weights(model))
        grpo_harness.train(
            config, model, plan(1), engine.hooks(), tmp_path / folder, None, print
        )
        after = lora.adapter_weights(model)
        return sum(float(mx.abs(after[k] - start[k]).sum()) for k in start)

    # (1 - 0.9) / sqrt(1 - 0.999) = 3.16: the uncorrected first step's size.
    ratio = moved(False, "off") / moved(True, "on")
    assert 3.0 < ratio < 3.3


# -------------------------------------------------------- main, end to end


def test_main_trains_from_chapter_25s_adapter_and_says_what_it_did(
    tmp_path: Path,
    course_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    sft = course_home / grpo_harness.INIT_ADAPTER
    lora.save_adapter(tiny_model(), sft)
    engine = FakeEngine()

    class Health:
        def __init__(self, url: str) -> None:
            self.url = url

        def health(self) -> dict[str, str]:
            return {"upstream": "http://127.0.0.1:1"}

    monkeypatch.setattr(grpo_harness, "CaptureClient", Health)
    monkeypatch.setattr(grpo_harness, "check_engine", lambda url: None)
    monkeypatch.setattr(grpo_harness, "check_harness", lambda *args: "pinned")
    monkeypatch.setattr(grpo_harness, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(grpo_harness, "load_model", lambda folder: tiny_model_bare())
    monkeypatch.setattr(grpo_harness, "make_hooks", lambda *args: engine.hooks())
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py", "--steps", "2"])
    grpo_harness.main()
    out = capsys.readouterr().out
    run = course_home / "26-grpo-in-harnesses" / "grpo" / "grpo"
    assert "grpo_harness.py · GRPO through the harnesses" in out
    assert "sft-no-bias-correction/adapter.safetensors" in out
    assert "steps     2, each 2 training tasks x 4 rollouts = 8" in out
    assert "reward    1 + the token bonus (up to 0.1, none at 2,000 tokens)" in out
    assert "AdamW lr 1e-05, bias correction on" in out
    assert "the engine samples with adapter.safetensors" in out
    assert "[grpo] step 2/2" in out and "done      2 steps in" in out
    assert "claude-code      8 rows · 13 tokens on average · 40 tokens trained" in out
    assert "mini-swe-agent   8 rows · 13 tokens on average · 40 tokens trained" in out
    note = json.loads((run / "run.json").read_text())
    assert note["learning_rate"] == 1e-5 and note["guard"] is True
    assert (run / "adapter_final.safetensors").is_file()
    assert str(Path.home()) not in out
    # A second run into the same folder is refused.
    with pytest.raises(SystemExit, match="already holds a run"):
        grpo_harness.main()


def tiny_model_bare() -> nn.Module:
    """A tiny model WITHOUT the adapter: load_policy adds it, as on the 2.6B.

    Returns:
        The model.
    """
    mx.random.seed(0)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def test_load_policy_refuses_an_adapter_of_another_shape(tmp_path: Path) -> None:
    other = tmp_path / "other.safetensors"
    mx.save_safetensors(str(other), {"x": mx.zeros((2, 2))})
    original = grpo_harness.load_model
    try:
        grpo_harness.load_model = lambda folder: tiny_model_bare()  # type: ignore[assignment]
        with pytest.raises(SystemExit, match="not this adapter's"):
            grpo_harness.load_policy(tmp_path, other)
    finally:
        grpo_harness.load_model = original  # type: ignore[assignment]


# ---------------------------------- one real step: proxy, harness, scripted engine


def installed_commit(env: Path) -> str | None:
    """The OpenEnv commit installed in an environment, read from its metadata.

    Args:
        env: The environment's folder.

    Returns:
        The commit, or None when OpenEnv is not installed there.
    """
    for found in env.glob("lib/python*/site-packages/openenv-*.dist-info"):
        return json.loads((found / "direct_url.json").read_text())["vcs_info"][
            "commit_id"
        ]
    return None


def find_proxy_env() -> Path | None:
    """Find a proxy environment at the pinned commit, here or in an earlier chapter.

    Returns:
        The environment, or None.
    """
    for chapter in CHAPTERS_WITH_STACK:
        env = _HOME / chapter / "proxy-env"
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def find_mini() -> Path | None:
    """Find a folder holding mini-swe-agent at its pin.

    Returns:
        The folder, or None.
    """
    for chapter in CHAPTERS_WITH_STACK:
        folder = _HOME / chapter
        if (folder / install.PINS[install.MINI].folder).is_dir():
            return folder
    return None


def find_tokenizer() -> Path | None:
    """Find LFM2.5's checked tokenizer.json on this machine.

    Returns:
        The file, or None.
    """
    for name in lfm_files.TOKENIZER_FOLDERS:
        path = _HOME / lfm_files.MODELS_FOLDER / name / lfm_files.TOKENIZER_FILE
        if path.is_file() and lfm_files.sha256_of(path) == lfm_files.TOKENIZER_SHA256:
            return path
    return None


REAL_PROXY_ENV = find_proxy_env()
REAL_MINI = find_mini()
REAL_TOKENIZER = find_tokenizer()


@contextlib.contextmanager
def live_scripted_engine() -> Iterator[tuple[int, scripted_engine.ScriptedModel]]:
    """Serve the scripted engine over real HTTP, on a free port.

    Yields:
        Its port, and its model (to queue replies on).
    """
    assert REAL_TOKENIZER is not None
    app, model = scripted_engine.create_app(scripted_engine.LfmCutter(REAL_TOKENIZER))
    sock = scripted_engine.engine.claim_port(0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "the scripted engine did not start within 10 s"
    try:
        yield sock.getsockname()[1], model
    finally:
        server.should_exit = True
        thread.join(10)


def start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``proxy`` or ``stop``.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "PROXY_PORT": str(proxy_port),
        "COURSE_HOME": str(home),
        "PROXY_ENV": str(REAL_PROXY_ENV),
    }
    return subprocess.run(
        ["bash", str(START_SH), command],
        env=env,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
        check=False,
    )


@pytest.mark.skipif(
    REAL_PROXY_ENV is None or REAL_MINI is None or REAL_TOKENIZER is None,
    reason="needs the capture proxy (./start.sh install), mini-swe-agent "
    "(uv run install.py) and LFM2.5's tokenizer",
)
def test_one_real_step_through_the_proxy_and_mini_swe_agent_trains_the_tiny_model(
    course_home: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    chapter = course_home / "26-grpo-in-harnesses"
    chapter.mkdir(parents=True)
    assert REAL_MINI is not None
    folder = install.PINS[install.MINI].folder
    (chapter / folder).symlink_to(REAL_MINI / folder)
    task = TRAIN[0]
    stack = tmp_path_factory.mktemp("stack")
    switches: list[Path | None] = []
    with live_scripted_engine() as (port, scripted):
        sock = scripted_engine.engine.claim_port(0)
        proxy_port = sock.getsockname()[1]
        sock.close()
        started = start_sh("proxy", port, proxy_port, stack)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            # One rollout at a time, so the queued replies go to them in turn:
            # the first solves the task, the second makes chapter 23's mistakes.
            scripted.push(run_one.solve_script(install.MINI, task))
            scripted.push(run_one.mistakes_script(install.MINI, task))
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            setup = rollouts.RolloutSetup(
                CaptureClient(proxy_url),
                proxy_url,
                grpo_harness.Config().reward,
                "e2e",
            )

            def run_rollouts(jobs: list[rollouts.Job]) -> list[rollouts.RolloutRecord]:
                records, failed = rollouts.run_step(jobs, setup, concurrency=1)
                assert failed == []
                return records

            def use_adapter(path: Path | None) -> str:
                switches.append(path)
                return path.name if path else "base"

            said: list[str] = []
            outcome = grpo_harness.train(
                grpo_harness.Config(steps=1, tasks_per_step=1, group_size=2),
                tiny_model(LFM_VOCAB),
                [[(task, install.MINI)]],
                grpo_harness.Hooks(run_rollouts, use_adapter),
                tmp_path_factory.mktemp("run"),
                None,
                said.append,
            )
        finally:
            start_sh("stop", port, proxy_port, stack)
    assert outcome.steps_run == 1 and outcome.final is not None
    assert [p.name for p in switches if p] == ["adapter_step001.safetensors"]
    run = outcome.final.parent
    lines = (run / "rollouts.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["correct"] for r in records] == [True, False]
    right, wrong = records
    assert right["reward"] == rewards.reward_for(True, right["n_sampled"])
    assert wrong["reward"] == 0.0
    for record in records:
        assert record["trainable"] and record["sequences"]
        assert record["loopback_only"] is True
        row = record["sequences"][0]
        assert 0 < sum(row["loss_mask"]) < len(row["input_ids"])
    (line,) = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert line["groups_without_contrast"] == 0 and line["train_rows"] >= 2
    assert line["importance_ratio_mean"] is not None
    assert "[grpo] step 1/1" in lines_of(said)


# ------------------------------------------------- the guard demo and setup

DATA = Path(__file__).parent / "data"
PROJECT_MAIN = Path(
    os.environ.get("MHRL_PROJECT_RUNS")
    or Path.home() / "sai_work" / "multi-harness-rl" / "runs" / "main"
)


def test_the_guard_demo_replays_attempt_1_and_stops_at_step_7(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py", "--guard-demo"])
    grpo_harness.main()
    out = capsys.readouterr().out
    assert out == (DATA / "26_guard_demo.txt").read_text()
    assert (
        "     7  adapter_step006      492    465      12%  STOP: adapter_step006 made "
        "these; keep adapter_step005"
    ) in out
    assert "the project kept adapter_step004, the adapter step 5's rollouts came" in out
    assert "--guard-tokens 100  stops at step 5, keeps adapter_step003" in out


@pytest.mark.skipif(
    not (PROJECT_MAIN / "grpo_attempt1").is_dir(), reason="needs the project's runs"
)
def test_the_guard_demo_reads_the_projects_own_files_to_the_same_verdicts(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    argv = ["grpo_harness.py", "--guard-demo", "--records", str(PROJECT_MAIN)]
    monkeypatch.setattr(sys, "argv", argv)
    grpo_harness.main()
    out = capsys.readouterr().out.splitlines()
    golden = (DATA / "26_guard_demo.txt").read_text().splitlines()
    assert out[2:] == golden[2:] and "the project's own records" in out[1]


def test_a_missing_start_adapter_names_chapter_25s_commands(course_home: Path) -> None:
    with pytest.raises(SystemExit, match="uv run train_sft.py --no-bias-correction"):
        grpo_harness.init_adapter(None)
    assert grpo_harness.init_adapter("base") is None
    sft = course_home / grpo_harness.INIT_ADAPTER
    sft.parent.mkdir(parents=True)
    sft.write_bytes(b"x")
    assert grpo_harness.init_adapter(None) == sft.resolve()


def test_the_engine_must_be_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(
        [{"serves": "answer-key teacher"}, {}, {"serves": "lfm2.5-2.6b 4-bit"}]
    )
    monkeypatch.setattr(grpo_harness, "engine_call", lambda *a: next(answers))
    with pytest.raises(SystemExit, match="serves answer-key teacher"):
        grpo_harness.check_engine("http://127.0.0.1:1")
    with pytest.raises(SystemExit, match="serves a scripted model"):
        grpo_harness.check_engine("http://127.0.0.1:1")
    grpo_harness.check_engine("http://127.0.0.1:1")


def test_attempt1_says_what_it_restores_and_what_it_cannot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py", "--attempt1", "--steps", "8"])
    args = grpo_harness.parse_args()
    config = grpo_harness.config_from(args)
    text = "\n".join(grpo_harness.setup_lines(config, None, tmp_path, True))
    assert "attempt 1's settings" in text
    assert "the article's call bonus (up to 0.1, none at 8 calls)" in text
    assert "AdamW lr 4e-05, bias correction off" in text
    assert "guard     none (--attempt1)" in text
    assert "not restored: the engine's bfloat16 log-probabilities" in text
    assert grpo_harness.run_folder(args).name == "attempt1"


def test_the_hooks_run_at_the_asked_concurrency_and_stop_on_a_missing_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    job = rollouts.Job(TRAIN[0], install.MINI, 1, 0, 0)

    def fake_step(
        jobs: list[rollouts.Job], setup: rollouts.RolloutSetup, concurrency: int
    ) -> tuple[list[rollouts.RolloutRecord], list[rollouts.Failure]]:
        seen.append(concurrency)
        return [], [rollouts.Failure(job, "the harness would not start")]

    monkeypatch.setattr(grpo_harness, "run_step", fake_step)
    config = grpo_harness.Config(concurrency=1)
    client = CaptureClient("http://127.0.0.1:1")
    hooks = grpo_harness.make_hooks(config, client, client.proxy_url, "http://x", "r")
    with pytest.raises(grpo_harness.RunStopped, match="1 of 1 rollouts made no record"):
        hooks.run_rollouts([job])
    assert seen == [1]
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py", "--concurrency", "2"])
    assert grpo_harness.config_from(grpo_harness.parse_args()).concurrency == 2


def test_an_engine_that_does_not_switch_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = tmp_path / "adapter_step001.safetensors"
    answers = iter([{"adapter": "base"}, {"adapter": lfm_files.short_path(adapter)}])
    monkeypatch.setattr(grpo_harness, "engine_call", lambda *a: next(answers))
    use_adapter = grpo_harness.adapter_switch("http://127.0.0.1:1")
    with pytest.raises(grpo_harness.RunStopped, match="it samples with base"):
        use_adapter(adapter)
    assert use_adapter(adapter) == lfm_files.short_path(adapter)


def test_a_run_that_died_in_its_first_step_is_not_appended_to(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = course_home / "26-grpo-in-harnesses" / "grpo" / "grpo"
    folder.mkdir(parents=True)
    (folder / "rollouts.jsonl").write_text("{}\n")  # no metrics.jsonl yet
    monkeypatch.setattr(sys, "argv", ["grpo_harness.py"])
    with pytest.raises(SystemExit, match="already holds a run"):
        grpo_harness.run_folder(grpo_harness.parse_args())


def test_the_guard_demo_adds_an_adapter_only_for_a_step_that_trained() -> None:
    def step(number: int, right: int, per_call: int) -> list[grpo_records.Rollout]:
        reply = grpo_records.Reply(10, per_call, True, True)
        row = grpo_records.Row(10 + per_call, 0, (reply,))
        return [
            grpo_records.Rollout(
                number,
                "t",
                install.MINI,
                i < right,
                1,
                per_call,
                float(i < right),
                1.0,
                (row,),
            )
            for i in range(4)
        ]

    steps = [step(1, 2, 40), step(2, 0, 40), step(3, 2, 40), step(4, 2, 400)]
    seen = grpo_harness.walk(steps, 150)
    assert [one.updated for one in seen] == [True, False, True, False]
    assert [one.sampled_with for one in seen] == [
        "the SFT adapter",
        "adapter_step001",
        "adapter_step001",  # step 2 had no contrast: nothing new to sample with
        "adapter_step003",
    ]
    assert seen[-1].verdict.stop and seen[-1].verdict.kept == "adapter_step001"
