"""Chapter 18 with MLX: the engine's model code, on a tiny model.

No test here loads the real model, downloads anything or uses the network.
``lfm.py`` runs on a tiny LFM2 (mlx-lm's own ``lfm2`` code, 64 numbers wide,
four layers, quantized to 4 bits, as in chapter 17's tests) and a tokenizer
that gives LFM2.5's chat markers their own ids and every other id one
printable character, so any reply the tiny model draws reads back to the
same ids. Each log-probability the sampler reports is checked against the
same tokens scored in one pass; a reply read through a reused prompt cache
is checked against one read from scratch. MLX installs only on Apple-silicon
Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import httpx  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("18-engine")

import engine  # noqa: E402
import lfm  # noqa: E402
from policy import Sampling  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
MARKERS = {
    "<|pad|>": 0,
    "<|startoftext|>": 1,
    "<|im_start|>": 2,
    "<think>": 3,
    "</think>": 4,
    "<|tool_call_start|>": 5,
    "<|tool_call_end|>": 6,
    "<|tool_list|>": 7,
    "<|im_end|>": EOS,
}
FIRST_PLAIN = 8  # ids 8 to 62 are the characters from " " on
PLAIN_COUNT = EOS - FIRST_PLAIN
FIRST_CHAR = ord(" ")
HIDDEN = 64
QUANT_GROUP = 32
BITS = 4
CLOSE = 1e-4  # float32 agreement between two routes to the same numbers
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
QUESTION = [{"role": "user", "content": "HOW MANY PATIENTS LIVE IN MURCIA?"}]


class TinyTokenizer:
    """LFM2.5's markers as single ids; every other id is one character."""

    eos_token_ids = frozenset({EOS})

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> str:
        """Render a chat with LFM2.5's markers.

        Args:
            messages: The chat.
            **options: ``add_generation_prompt`` is read.

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
        """Give each marker its id and each character one of the plain ids.

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
                code = (ord(text[index]) - FIRST_CHAR) % PLAIN_COUNT
                ids.append(FIRST_PLAIN + code)
                index += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Write ids back: markers (unless skipped) and characters.

        Args:
            ids: The ids.
            skip_special_tokens: Leave the markers out.

        Returns:
            The text.
        """
        names = {token: marker for marker, token in MARKERS.items()}
        return "".join(
            ("" if skip_special_tokens else names[i])
            if i in names
            else chr(FIRST_CHAR + i - FIRST_PLAIN)
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
def tiny(monkeypatch: pytest.MonkeyPatch) -> Iterator[lfm.LfmModel]:
    """The engine's model class, loading the tiny model and tokenizer.

    Args:
        monkeypatch: pytest's patcher.

    Yields:
        The loaded model.
    """
    network = tiny_model()
    monkeypatch.setattr(lfm, "load", lambda folder: (network, TinyTokenizer()))
    yield lfm.LfmModel(Path("tiny"), seed=0)


def prompt_ids(model: lfm.LfmModel, messages: list[dict[str, Any]]) -> list[int]:
    """Render and encode a chat, as the engine does.

    Args:
        model: The model.
        messages: The chat.

    Returns:
        The prompt's ids.
    """
    return model.encode(model.render(messages, None))


def one_pass(
    model: lfm.LfmModel, prompt: list[int], tokens: list[int], temperature: float
) -> list[float]:
    """Score a reply's tokens in one pass: log_softmax(scores / T) at each.

    Args:
        model: The model.
        prompt: The prompt's ids.
        tokens: The reply's ids.
        temperature: 0 reports the untempered distribution, as the sampler does.

    Returns:
        Each token's log-probability.
    """
    scores = model.model(mx.array([prompt + tokens[:-1]]))[0, len(prompt) - 1 :]
    scores = scores.astype(mx.float32)
    if temperature > 0:
        scores = scores / temperature
    logprobs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    return [logprobs[i, token].item() for i, token in enumerate(tokens)]


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_each_logprob_is_its_token_under_the_distribution_it_was_drawn_from(
    tiny: lfm.LfmModel, temperature: float
) -> None:
    prompt = prompt_ids(tiny, QUESTION)
    sampled = tiny.sample(prompt, tiny.new_cache(), Sampling(6, temperature))
    assert len(sampled.logprobs) == len(sampled.token_ids) > 0
    expected = one_pass(tiny, prompt, sampled.token_ids, temperature)
    assert sampled.logprobs == pytest.approx(expected, abs=CLOSE)
    assert all(value <= 0 for value in sampled.logprobs)


def test_greedy_is_the_same_twice(tiny: lfm.LfmModel) -> None:
    prompt = prompt_ids(tiny, QUESTION)
    one = tiny.sample(prompt, tiny.new_cache(), Sampling(6, 0.0))
    two = tiny.sample(prompt, tiny.new_cache(), Sampling(6, 0.0))
    assert one.token_ids == two.token_ids and one.logprobs == two.logprobs


def test_the_alternatives_halve_their_gap_at_twice_the_temperature(
    tiny: lfm.LfmModel,
) -> None:
    prompt = prompt_ids(tiny, QUESTION)
    cool = tiny.sample(prompt, tiny.new_cache(), Sampling(1, 1.0, top_logprobs=5))
    hot = tiny.sample(prompt, tiny.new_cache(), Sampling(1, 2.0, top_logprobs=5))
    (first_cool,), (first_hot,) = cool.top_logprobs, hot.top_logprobs
    assert len(first_cool) == 5
    values = [value for _, value in first_cool]
    assert values == sorted(values, reverse=True)
    assert [token for token, _ in first_cool] == [token for token, _ in first_hot]
    gap_cool = first_cool[0][1] - first_cool[1][1]
    gap_hot = first_hot[0][1] - first_hot[1][1]
    assert gap_hot == pytest.approx(gap_cool / 2, abs=CLOSE)  # chapter 20's probe


def test_a_cache_holds_the_prompt_and_the_reply(tiny: lfm.LfmModel) -> None:
    prompt = prompt_ids(tiny, QUESTION)
    cache = tiny.new_cache()
    assert tiny.cache_size(cache) == 0
    sampled = tiny.sample(prompt, cache, Sampling(5, 0.0))
    assert tiny.cache_size(cache) == len(prompt) + len(sampled.token_ids)


def test_a_reused_cache_gives_the_numbers_a_fresh_read_gives(
    tiny: lfm.LfmModel,
) -> None:
    prompt = prompt_ids(tiny, QUESTION)
    cache = tiny.new_cache()
    first = tiny.sample(prompt, cache, Sampling(3, 0.0))
    more = tiny.encode("\n<|im_start|>user\nGO ON<|im_end|>\n<|im_start|>assistant\n")
    whole = prompt + first.token_ids + more
    reused = tiny.sample(more, cache, Sampling(3, 0.0, top_logprobs=5))
    fresh = tiny.sample(whole, tiny.new_cache(), Sampling(3, 0.0, top_logprobs=5))
    assert reused.token_ids == fresh.token_ids
    assert reused.logprobs == pytest.approx(fresh.logprobs, abs=CLOSE)
    for mine, theirs in zip(reused.top_logprobs, fresh.top_logprobs, strict=True):
        assert [t for t, _ in mine] == [t for t, _ in theirs]


def test_the_engine_reuses_the_cache_across_calls(tiny: lfm.LfmModel) -> None:
    core = engine.Engine(tiny)
    sampling = Sampling(3, 0.0, top_logprobs=3)
    first = core.complete(engine.ChatRequest(QUESTION, None, sampling))
    reply = "<think>" + tiny.decode([t for t in first.sampled.token_ids if t != EOS])
    follow = [*QUESTION, {"role": "assistant", "content": reply}]
    follow.append({"role": "user", "content": "GO ON"})
    second = core.complete(engine.ChatRequest(follow, None, sampling))
    chain = first.prompt_ids + first.sampled.token_ids
    assert second.prompt_ids[: len(chain)] == chain
    assert second.cached_tokens == len(chain)
    fresh = engine.Engine(tiny).complete(engine.ChatRequest(follow, None, sampling))
    assert fresh.cached_tokens == 0
    assert second.sampled.token_ids == fresh.sampled.token_ids
    assert second.sampled.logprobs == pytest.approx(fresh.sampled.logprobs, abs=CLOSE)


def test_the_end_of_turn_and_a_stop_text_end_a_reply(
    tiny: lfm.LfmModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    letters = tiny.encode("ABC")

    def scripted(tokens: list[int]) -> Any:
        """Yield fixed tokens, as generate_step yields drawn ones."""

        def steps(*_: Any, **__: Any) -> Iterator[tuple[int, mx.array]]:
            for token in tokens:
                yield token, mx.zeros((VOCAB,)) - mx.log(mx.array(float(VOCAB)))

        return steps

    monkeypatch.setattr(lfm, "generate_step", scripted([*letters, EOS, letters[0]]))
    ended = tiny.sample([1], tiny.new_cache(), Sampling(10, 0.0))
    assert ended.token_ids == [*letters, EOS] and ended.finish_reason == "stop"
    monkeypatch.setattr(lfm, "generate_step", scripted([*letters, *letters]))
    stopped = tiny.sample([1], tiny.new_cache(), Sampling(10, 0.0, stop=("BC",)))
    assert stopped.token_ids == letters and stopped.finish_reason == "stop"
    monkeypatch.setattr(lfm, "generate_step", scripted(letters))
    cut = tiny.sample([1], tiny.new_cache(), Sampling(3, 0.0))
    assert cut.finish_reason == "length"


def test_the_engine_answers_over_http_on_its_worker_thread(
    tiny: lfm.LfmModel,
) -> None:
    app = engine.create_app(engine.Engine(tiny))

    async def ask() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://e") as http:
            body = {"messages": QUESTION, "max_tokens": 4, "top_logprobs": 2}
            return await http.post("/v1/chat/completions", json=body)

    response = asyncio.run(ask())
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    entries = choice["logprobs"]["content"]
    assert len(entries) == len(choice["token_ids"]) > 0
    assert all(len(entry["top_logprobs"]) == 2 for entry in entries)
