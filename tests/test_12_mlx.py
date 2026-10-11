"""Chapter 12 with MLX: the sampling loop, its numbers, and the timings.

No test here loads the real model, downloads anything or uses the network.
The loop runs mlx-lm's real ``generate_step`` on small fake models whose
scores are known in advance (so every log-probability, entropy and grid can
be worked out by hand), and on a tiny LFM2 built with mlx-lm's own ``lfm2``
code (64 numbers wide, two layers: one convolution, one attention), which
also feeds timing.py's cache and prefill checks. MLX installs only on
Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
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

CHAPTER = use_chapter("12-generate")

import lfm  # noqa: E402
import measures  # noqa: E402
import sample_lab  # noqa: E402
import timing  # noqa: E402
from sample_lab import Draw  # noqa: E402

VOCAB = 64
EOS = VOCAB - 1
HIDDEN = 64
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": 2,
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
    "layer_types": ["conv", "full_attention"],
}


class FakeTokenizer:
    """Stands in for mlx-lm's tokenizer: renders chats, encodes and decodes."""

    eos_token_ids = frozenset({EOS})

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

    Every other token scores 0, so each step's scores say which token came
    before them, and the loop's bookkeeping can be checked step by step.
    """

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


class FixedModel:
    """A fake model whose scores are the same at every step."""

    def __init__(self, scores: list[float]) -> None:
        """Keep the scores.

        Args:
            scores: One per token.
        """
        self.scores = mx.array(scores, dtype=mx.bfloat16)

    def make_cache(self) -> list[Any]:
        """Keep no cache.

        Returns:
            An empty list.
        """
        return []

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        """Give the same scores after every position.

        Args:
            inputs: Token ids, shape (1, length).
            cache: Ignored.

        Returns:
            Scores, shape (1, length, VOCAB).
        """
        return mx.broadcast_to(self.scores, (1, inputs.shape[1], VOCAB))


def spread_scores() -> list[float]:
    """Scores falling evenly from token 0 to the end of turn, the least likely.

    Returns:
        64 scores.
    """
    return [4.0 * (1 - index / VOCAB) for index in range(VOCAB)]


def loaded(model: Any) -> lfm.LoadedModel:
    """Wrap a model and the fake tokenizer as the scripts expect them.

    Args:
        model: Any model.

    Returns:
        The loaded model.
    """
    return lfm.LoadedModel(model, FakeTokenizer(), Path("fake"), 0, 0.0)


def log_softmax(scores: list[float], temperature: float = 1.0) -> np.ndarray:
    """Work out log-probabilities by hand, in float64.

    Args:
        scores: The scores.
        temperature: What to divide them by.

    Returns:
        The log-probabilities.
    """
    values = np.array(scores, dtype=np.float64) / temperature
    return values - np.log(np.sum(np.exp(values - values.max()))) - values.max()


def tiny_model(seed: int = 0) -> nn.Module:
    """Build a tiny LFM2 with mlx-lm's own code, every weight in float32.

    Args:
        seed: The random seed for the weights.

    Returns:
        The model, evaluated.
    """
    mx.random.seed(seed)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


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


# ------------------------------------------------------- the distribution


def test_temperature_one_and_greedy_change_nothing() -> None:
    """Dividing by 1, or taking the top token, leaves the numbers as they are."""
    logprobs = mx.array(log_softmax(spread_scores()), dtype=mx.float32)
    assert sample_lab.drawn_from(logprobs, Draw()) is logprobs
    assert sample_lab.drawn_from(logprobs, Draw(greedy=True)) is logprobs


def test_temperature_divides_the_scores_then_renormalizes() -> None:
    """Below 1 sharpens, above 1 flattens; probabilities still add up to 1."""
    scores = spread_scores()
    logprobs = mx.array(log_softmax(scores), dtype=mx.float32)
    for temperature in (0.5, 1.5):
        tempered = sample_lab.drawn_from(logprobs, Draw(temperature=temperature))
        expected = log_softmax(scores, temperature)
        assert np.allclose(np.array(tempered), expected, atol=1e-5)
        assert mx.sum(mx.exp(tempered)).item() == pytest.approx(1.0, abs=1e-5)
    sharp = sample_lab.drawn_from(logprobs, Draw(temperature=0.5))
    flat = sample_lab.drawn_from(logprobs, Draw(temperature=1.5))
    assert sample_lab.entropy(sharp).item() < sample_lab.entropy(logprobs).item()
    assert sample_lab.entropy(flat).item() > sample_lab.entropy(logprobs).item()


def test_top_k_keeps_exactly_k_tokens() -> None:
    """Top-k leaves k tokens with probability, renormalized; the rest are 0."""
    logprobs = mx.array(log_softmax(spread_scores()), dtype=mx.float32)
    kept = sample_lab.drawn_from(logprobs, Draw(top_k=5))
    finite = np.isfinite(np.array(kept))
    assert finite.sum() == 5 and finite[:5].all()
    assert mx.sum(mx.exp(kept)).item() == pytest.approx(1.0, abs=1e-6)
    share = float(np.exp(log_softmax(spread_scores())[:5]).sum())
    assert kept[0].item() == pytest.approx(logprobs[0].item() - math.log(share))


def test_top_p_keeps_the_fewest_tokens_reaching_p() -> None:
    """Top-p keeps the likeliest tokens until their probabilities reach p."""
    scores = spread_scores()
    logprobs = mx.array(log_softmax(scores), dtype=mx.float32)
    kept = np.isfinite(np.array(sample_lab.drawn_from(logprobs, Draw(top_p=0.5))))
    probs = np.exp(log_softmax(scores))
    needed = int(np.searchsorted(np.cumsum(probs), 0.5)) + 1
    assert kept.sum() == needed and kept[:needed].all()


def test_entropy_by_hand() -> None:
    """Even odds over n tokens give log n; a certain token gives 0."""
    even = mx.full((8,), -math.log(8))
    assert sample_lab.entropy(even).item() == pytest.approx(math.log(8))
    certain = mx.array([0.0] + [-math.inf] * 7)
    assert sample_lab.entropy(certain).item() == 0.0


def test_an_unknown_precision_is_refused() -> None:
    """Only fp32 and bf16 exist; anything else stops at once."""
    with pytest.raises(ValueError, match="dtype must be"):
        Draw(dtype="fp16")


def test_draw_says_when_it_cuts() -> None:
    """Only top-k or top-p below 1 cut; greedy never does."""
    assert not Draw().cuts()
    assert Draw(top_k=50).cuts() and Draw(top_p=0.9).cuts()
    assert not Draw(greedy=True, top_k=50).cuts()


# ---------------------------------------------------------------- the loop


def test_each_step_keeps_its_own_scores() -> None:
    """The scores kept for step n are step n's, not the step before or after."""
    reply = sample_lab.sample(loaded(ChainModel()), [1, 58], Draw(greedy=True), 10)
    assert reply.tokens() == [59, 60, 61, 62, 63]
    # Token t + 1 wins with score 10 + (t mod 5): 58 -> 13, 59 -> 14, 60 -> 10...
    assert [step.score for step in reply.steps] == [13.0, 14.0, 10.0, 11.0, 12.0]
    assert reply.finish_reason == "stop"
    assert reply.prompt_tokens == 2


def test_the_cap_ends_with_length_and_the_end_of_turn_with_stop() -> None:
    """Three tokens short of the end is 'length'; reaching it on the cap is 'stop'."""
    draw = Draw(greedy=True)
    short = sample_lab.sample(loaded(ChainModel()), [58], draw, 3)
    assert (short.tokens(), short.finish_reason) == ([59, 60, 61], "length")
    exact = sample_lab.sample(loaded(ChainModel()), [58], draw, 5)
    assert (exact.tokens()[-1], exact.finish_reason) == (EOS, "stop")


def test_a_cap_below_one_token_is_refused() -> None:
    """generate_step needs at least one token to draw."""
    with pytest.raises(ValueError, match="at least 1 token"):
        sample_lab.sample(loaded(ChainModel()), [58], Draw(greedy=True), 0)


def test_a_prompt_longer_than_one_chunk_keeps_the_pairing() -> None:
    """Prefill chunks never reach the processor, so step 1 still gets its scores."""
    prompt = [1] * (sample_lab.PREFILL_STEP + 300) + [58]
    reply = sample_lab.sample(loaded(ChainModel()), prompt, Draw(greedy=True), 3)
    assert reply.tokens() == [59, 60, 61]
    assert [step.score for step in reply.steps] == [13.0, 14.0, 10.0]


def test_temperature_with_both_cuts() -> None:
    """Uncut means the same temperature with nothing cut; the drawn one is cut."""
    scores = spread_scores()
    mx.random.seed(0)
    draw = Draw(temperature=0.7, top_k=10, top_p=0.5)
    reply = sample_lab.sample(loaded(FixedModel(scores)), [1], draw, 8)
    tempered = log_softmax(scores, 0.7)
    probs = np.exp(tempered[:10]) / np.exp(tempered[:10]).sum()  # after top-k
    kept = int(np.searchsorted(np.cumsum(probs), 0.5)) + 1  # then top-p
    share = float(np.exp(tempered[:kept]).sum())
    for step in reply.steps:
        assert step.token < kept
        assert step.uncut == pytest.approx(tempered[step.token], abs=1e-5)
        assert step.logprob == pytest.approx(
            tempered[step.token] - math.log(share), abs=1e-5
        )


def test_logprobs_are_the_distribution_drawn_from() -> None:
    """At temperature 0.7, the kept number is log_softmax(scores / 0.7)."""
    scores = spread_scores()
    mx.random.seed(0)
    reply = sample_lab.sample(loaded(FixedModel(scores)), [1], Draw(temperature=0.7), 6)
    processed, raw = log_softmax(scores, 0.7), log_softmax(scores)
    for step in reply.steps:
        assert step.logprob == pytest.approx(processed[step.token], abs=1e-5)
        assert step.raw == pytest.approx(raw[step.token], abs=1e-5)
        assert step.uncut == step.logprob  # nothing was cut
        assert step.float32 == step.logprob  # float32 throughout
    entropy = -float(np.sum(np.exp(processed) * processed))
    assert reply.steps[0].entropy == pytest.approx(entropy, abs=1e-4)


def test_the_same_seed_draws_the_same_reply() -> None:
    """Sampling is random, but a seed makes it repeat exactly."""
    model, draw = loaded(FixedModel(spread_scores())), Draw(temperature=1.5)
    mx.random.seed(7)
    first = sample_lab.sample(model, [1], draw, 12)
    mx.random.seed(7)
    second = sample_lab.sample(model, [1], draw, 12)
    assert first.tokens() == second.tokens()
    assert len(set(first.tokens())) > 1  # it really drew, not just the top token


def test_top_k_draws_only_kept_tokens_and_lowers_entropy() -> None:
    """Every drawn token is among the 5 kept; each step's entropy falls."""
    mx.random.seed(0)
    model = loaded(FixedModel(spread_scores()))
    reply = sample_lab.sample(model, [1], Draw(top_k=5), 20)
    share = float(np.exp(log_softmax(spread_scores())[:5]).sum())
    for step in reply.steps:
        assert step.token < 5
        assert step.logprob == pytest.approx(step.uncut - math.log(share), abs=1e-5)
        assert step.entropy < step.uncut_entropy
    result = measures.cut(
        [s.logprob for s in reply.steps],
        [s.uncut for s in reply.steps],
        [s.entropy for s in reply.steps],
        [s.uncut_entropy for s in reply.steps],
    )
    assert result.kept_mean == pytest.approx(share, abs=1e-5)


def test_bf16_scores_between_8_and_16_land_on_sixteenths() -> None:
    """In bfloat16 the log-probabilities fall on the grid of the scores' size."""
    scores = [0.0] * VOCAB
    scores[3], scores[4] = 12.0, 11.0  # a likely token and a close second
    mx.random.seed(0)
    model = loaded(FixedModel(scores))
    reply = sample_lab.sample(model, [1], Draw(dtype="bf16"), 8)
    values = [step.logprob for step in reply.steps]
    assert measures.grid_power(values) == 4  # multiples of 1/16, no finer
    wide = log_softmax(scores)
    for step in reply.steps:
        assert step.float32 == pytest.approx(wide[step.token], abs=1e-5)
        assert step.logprob != pytest.approx(step.float32, abs=1e-4)


def test_bf16_rounds_a_likely_token_to_exactly_zero() -> None:
    """A 99.99%-likely token gets log-probability 0 in bfloat16, not in float32."""
    reply = sample_lab.sample(loaded(ChainModel()), [58], Draw(dtype="bf16"), 1)
    step = reply.steps[0]
    assert step.logprob == 0.0
    expected = log_softmax([13.0] + [0.0] * 63)[0]  # about -0.00014
    assert step.float32 == pytest.approx(expected, abs=1e-6)


def test_first_step_tables() -> None:
    """Step 1's likeliest tokens and its spread at three temperatures."""
    scores = spread_scores()
    model = loaded(FixedModel(scores))
    reply = sample_lab.sample(model, [1], Draw(greedy=True), 1)
    top = sample_lab.likeliest(reply.first, 3)
    assert [candidate.token for candidate in top] == [0, 1, 2]
    assert top[0].logprob == pytest.approx(log_softmax(scores)[0], abs=1e-5)
    rows = sample_lab.at_temperatures(reply.first_scores, [0, 1], (0.5, 1.0, 1.5))
    assert [row.temperature for row in rows] == [0.5, 1.0, 1.5]
    assert rows[0].named[0] > rows[1].named[0] > rows[2].named[0]
    assert rows[0].holding < rows[1].holding < rows[2].holding
    assert rows[0].entropy < rows[1].entropy < rows[2].entropy
    probs = np.sort(np.exp(log_softmax(scores)))[::-1]
    assert rows[1].holding == int(np.searchsorted(np.cumsum(probs), 0.99)) + 1


def test_a_real_lfm2_draws_through_generate_step() -> None:
    """mlx-lm's own LFM2 code, tiny: greedy twice is identical, numbers are sane."""
    model = loaded(tiny_model())
    prompt = list(range(5, 25))
    first = sample_lab.sample(model, prompt, Draw(greedy=True), 12)
    second = sample_lab.sample(model, prompt, Draw(greedy=True), 12)
    assert first.tokens() == second.tokens()
    for step in first.steps:
        assert step.logprob <= 0.0 and step.entropy >= 0.0
        assert step.logprob == step.raw  # greedy reports the untempered numbers
    assert sample_lab.likeliest(first.first, 1)[0].token == first.tokens()[0]


# ------------------------------------------------------ sample_lab's main()


def use_fakes(monkeypatch: pytest.MonkeyPatch, folder: Path, model: Any) -> None:
    """Make sample_lab load a fake model from a folder with the maker's settings.

    Args:
        monkeypatch: pytest's patcher.
        folder: The fake build's folder.
        model: The model to load.
    """
    folder.mkdir(parents=True, exist_ok=True)
    settings = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1}
    (folder / "generation_config.json").write_text(json.dumps(settings))
    monkeypatch.setattr(sample_lab, "pick_model", lambda fast: (folder, "fake build"))
    monkeypatch.setattr(sample_lab, "load_folder", lambda path: loaded(model))


def test_main_greedy_shows_every_section_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """Greedy: the first step, temperatures, every step, the ending, a rerun."""
    use_fakes(monkeypatch, tmp_path / "build", ChainModel())
    out = run_main(sample_lab, "hi", "--greedy")
    assert "1. step 1: the 10 likeliest next tokens, of 64" in out
    assert "2. the same step at 3 temperatures" in out
    assert "3. the reply, drawn token by token" in out
    assert "finish_reason 'stop'" in out
    assert "identical to the first run's" in out and ": yes;" in out
    assert "temperature 0.1, top_k 50," in out
    saved = course_home / "12-generate" / "sample_lab-greedy.jsonl"
    lines = saved.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["draw"]["greedy"] is True and header["finish_reason"] == "stop"
    assert json.loads(lines[-1])["token"] == EOS


def test_main_with_a_cap_says_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, course_home: Path
) -> None:
    """--max-tokens cuts the reply: finish_reason 'length', saved under its own name."""
    use_fakes(monkeypatch, tmp_path / "build", FixedModel(spread_scores()))
    out = run_main(sample_lab, "hi", "--greedy", "--max-tokens", "20")
    assert "finish_reason 'length'" in " ".join(out.split())  # unwrapped
    assert "... 4 steps not shown (all are in the saved file)" in out
    saved = course_home / "12-generate" / "sample_lab-greedy-max20.jsonl"
    assert len(saved.read_text().splitlines()) == 1 + 20


def test_main_top_k_raw_and_bf16_add_their_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each flag adds its column and its summary."""
    use_fakes(monkeypatch, tmp_path / "build", FixedModel(spread_scores()))
    out = run_main(
        sample_lab,
        "hi",
        "--top-k",
        "5",
        "--raw",
        "--temperature",
        "0.7",
        "--dtype",
        "bf16",
        "--max-tokens",
        "6",
    )
    assert "uncut" in out and "raw" in out and "float32" in out
    assert "what top-k 5 changed, at each of the 6 steps" in out
    assert "processed and raw log-probabilities (--raw)" in out
    assert "log-probabilities in bfloat16 (--dtype bf16), beside float32" in out


@pytest.mark.parametrize(
    "args",
    [
        ["--temperature", "0"],
        ["--top-k", "-1"],
        ["--top-p", "0"],
        ["--top-p", "1.5"],
        ["--greedy", "--top-k", "50"],
        ["--max-tokens", "0"],
    ],
)
def test_bad_flags_stop_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    """Impossible settings stop before any model is loaded."""
    use_fakes(monkeypatch, tmp_path / "build", ChainModel())
    with pytest.raises(SystemExit):
        run_main(sample_lab, *args)


def test_top_k_must_leave_some_tokens_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A top-k as large as the vocabulary stops, naming the vocabulary's size."""
    folder = tmp_path / "build"
    use_fakes(monkeypatch, folder, ChainModel())
    (folder / "config.json").write_text(json.dumps({"vocab_size": VOCAB}))
    with pytest.raises(SystemExit, match="below the model's 64 tokens"):
        run_main(sample_lab, "--top-k", str(VOCAB))


def test_record_names_follow_the_draw() -> None:
    """Each kind of run is saved under its own name."""
    assert sample_lab.record_tag(Draw(), 0, 1_024) == "t1-seed0"
    assert sample_lab.record_tag(Draw(greedy=True), 0, 20) == "greedy-max20"
    draw = Draw(temperature=0.7, top_k=50, top_p=0.9, dtype="bf16")
    assert sample_lab.record_tag(draw, 3, 1_024) == "t0.7-k50-p0.9-bf16-seed3"


# ----------------------------------------------------------------- timing.py


def test_prompts_have_exactly_the_length_asked() -> None:
    """The table is cut so the whole chat has the length asked for."""
    tokenizer = FakeTokenizer()
    for length in (200, 500, 2_000):
        assert len(timing.prompt_of_length(tokenizer, length)) == length
    with pytest.raises(ValueError, match="cannot build"):
        timing.prompt_of_length(tokenizer, 10)
    assert timing.table_text(3).splitlines()[1:] == [
        "1 | Customer 1 | Lyon",
        "2 | Customer 2 | Berlin",
        "3 | Customer 3 | Madrid",
    ]


def test_prefill_reads_in_chunks_and_the_cache_covers_every_token() -> None:
    """10 tokens in chunks of 4: 4 + 4 + 1, then the last one predicts."""
    model = tiny_model()
    prompt = list(range(3, 13))
    run = timing.read_and_decode(model, prompt, 4, 5)
    assert run.prefill.chunks == [4, 4, 1, 1]
    assert run.prefill.length == 10 and run.prefill.first_seconds > 0
    assert len(run.tokens) == 5 and len(run.seconds) == 4
    size = run.cache
    assert size.positions == 10 + 5  # the prompt and every token fed back
    assert (size.attention_layers, size.conv_layers) == (1, 1)
    # keys and values x 2 key/value heads x 16 numbers each x 4 bytes (float32)
    assert size.bytes_per_position == 2 * 2 * 16 * 4
    assert size.attention_bytes == size.bytes_per_position * size.positions
    assert size.conv_bytes > 0


def test_an_empty_cache_has_no_size_to_measure() -> None:
    """A cache before any token is read is refused, not divided by zero."""
    with pytest.raises(ValueError, match="no keys and values"):
        timing.cache_size(tiny_model().make_cache())


def test_decoding_from_scratch_gives_the_cached_tokens() -> None:
    """Throwing the cache away changes the cost, not the reply."""
    model = tiny_model()
    prompt = list(range(3, 40))
    cached = timing.read_and_decode(model, prompt, 8, 6).tokens
    seconds, uncached = timing.decode_without_cache(model, prompt, 8, 6)
    assert uncached == cached
    assert len(seconds) == 6


def test_best_of_keeps_the_faster_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs: the faster prefill and the steadier decode count."""
    size = timing.CacheSize(1, 1, 1, 1, 1, 1)
    slow = timing.Prefill(10, [9, 1], 2.0, 100)
    fast = timing.Prefill(10, [9, 1], 1.0, 100)
    runs = iter(
        [
            timing.Decoded(slow, [0.1, 0.1], [1, 2], size),
            timing.Decoded(fast, [0.3, 0.3], [1, 2], size),
        ]
    )
    monkeypatch.setattr(timing, "read_and_decode", lambda *args: next(runs))
    best, same = timing.best_of(None, [1], 4, 3)  # no model: the runs are faked
    assert best.prefill is fast and best.seconds == [0.1, 0.1]
    assert best.tokens == [1, 2] and same


def test_timing_main_fast_prints_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """--fast: two prompt lengths, prefill, both decodes and the summary."""
    model = loaded(tiny_model())
    monkeypatch.setattr(timing, "pick_model", lambda fast: (Path("fake"), "tiny"))
    monkeypatch.setattr(timing, "load_folder", lambda path: model)
    out = run_main(timing, "--fast")
    assert "1. prefill: reading the prompt, 1,024 tokens at a time" in out
    assert "     500  499 + 1 " in out
    assert "   2,000  1,024 + 975 + 1 " in out
    assert "(1,999 + 1)" in out  # mlx-lm's 2,048-token chunks, for comparison
    assert "2. decode with the cache: one token at a time, 32 tokens per prompt" in out
    assert "both runs of each prompt decoded the same 32 tokens: yes" in out
    assert "3. decode without the cache" in out and "2 of 2" in out
    assert "4. what changes with the prompt's length" in out
