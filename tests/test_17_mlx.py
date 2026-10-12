"""Chapter 17 with MLX: the real policy's loop, on a tiny model.

No test here loads the real model, downloads anything or uses the network.
``lfm.py`` runs on chapter 15's tiny LFM2 (mlx-lm's own ``lfm2`` code, 64
numbers wide, four layers, quantized to 4 bits) and a fake tokenizer that
renders LFM2.5's chat markers and maps every other character to an id the
tiny model knows. The harness script runs end to end on them. MLX installs
only on Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("17-tiny-harness")

import lfm  # noqa: E402
import tiny_harness  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
FIRST_PLAIN = 8  # ids below this are the chat markers
HIDDEN = 64
QUANT_GROUP = 32
BITS = 4
CAP = 6  # reply cap for the tiny model
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

    def __init__(self) -> None:
        """Start with no render recorded."""
        self.options: list[dict[str, Any]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> str:
        """Render a chat as LFM2.5's template does, and remember the options.

        Args:
            messages: The chat.
            **options: ``tools`` and ``add_generation_prompt`` are read.

        Returns:
            The rendered text.
        """
        self.options.append(options)
        tools = options.get("tools")
        system = f"List of tools: {tools}" if tools else ""
        text = "<|startoftext|>" + "".join(
            f"<|im_start|>{m['role']}\n{m['content']}{system if i == 0 else ''}"
            "<|im_end|>\n"
            for i, m in enumerate(messages)
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

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Write markers back (kept only when asked), every other id as a letter.

        Args:
            ids: Token ids.
            skip_special_tokens: Leave the markers out.

        Returns:
            The text.
        """
        names = {token: marker for marker, token in MARKERS.items()}
        return "".join(
            ("" if skip_special_tokens else names[i])
            if i in names
            else chr(ord("a") + i % 26)
            for i in ids
        )


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
    nn.quantize(model, group_size=QUANT_GROUP, bits=BITS)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> FakeTokenizer:
    """Make lfm.py load the tiny model and the fake tokenizer.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The tokenizer every load hands out.
    """
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(lfm, "load", lambda folder: (tiny_model(), tokenizer))
    return tokenizer


def test_a_greedy_reply_is_the_same_twice_and_keeps_its_markers(
    tiny: FakeTokenizer,
) -> None:
    policy = lfm.LfmPolicy(Path("tiny"), CAP)
    prompt = policy.render([{"role": "user", "content": "hi"}], None)
    one, two = policy.complete(prompt), policy.complete(prompt)
    assert one.text == two.text and one.reply_tokens == two.reply_tokens
    assert one.prompt_tokens == len(tiny.encode(prompt))
    if one.finish_reason == "stop":
        assert one.text.endswith("<|im_end|>")
    else:
        assert one.finish_reason == "length" and one.reply_tokens == CAP


def test_render_prints_the_tools_and_keeps_the_reasoning(tiny: FakeTokenizer) -> None:
    policy = lfm.LfmPolicy(Path("tiny"), CAP)
    tools = [tiny_harness.bash_schema()]
    prompt = policy.render([{"role": "system", "content": "rule"}], tools)
    assert "List of tools" in prompt and prompt.endswith("<think>")
    options = tiny.options[-1]
    assert options["tools"] == tools
    assert options["add_generation_prompt"] is True
    assert options["tokenize"] is False
    assert options["preserve_thinking"] is True


def test_the_end_of_turn_stops_a_reply(
    tiny: FakeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = [9, 4, 5, EOS, 9]  # a letter, </think>, a call opened, the end of turn

    def scripted(*_: Any, **__: Any) -> Iterator[tuple[int, None]]:
        """Yield fixed tokens, as generate_step yields drawn ones."""
        yield from ((token, None) for token in tokens)

    monkeypatch.setattr(lfm, "generate_step", scripted)
    reply = lfm.LfmPolicy(Path("tiny"), CAP).complete("prompt")
    assert reply.finish_reason == "stop" and reply.reply_tokens == 4
    assert reply.text.endswith("</think><|tool_call_start|><|im_end|>")


def test_a_temperature_above_zero_draws_with_the_seed(tiny: FakeTokenizer) -> None:
    first = lfm.LfmPolicy(Path("tiny"), CAP, temperature=1.0, seed=3)
    one = first.complete("<|im_start|>user\nhi<|im_end|>\n<think>")
    second = lfm.LfmPolicy(Path("tiny"), CAP, temperature=1.0, seed=3)
    two = second.complete("<|im_start|>user\nhi<|im_end|>\n<think>")
    assert one.text == two.text


def test_the_harness_script_runs_on_the_tiny_model(
    tiny: FakeTokenizer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tiny_harness, "pick_model", lambda fast: (tmp_path, "tiny"))
    monkeypatch.setattr(tiny_harness, "OUTPUT_CAP", CAP)
    saved = sys.argv
    sys.argv = ["tiny_harness.py"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            tiny_harness.main()
    finally:
        sys.argv = saved
    text = out.getvalue()
    assert "(tiny), loaded in " in text
    assert "  call 1 · read " in text
    assert "peak memory: " in text and "the weights included" in text
