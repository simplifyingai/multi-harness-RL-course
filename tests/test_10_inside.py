"""Chapter 10: walking a prompt through LFM2, and the first reply, with MLX.

No test here loads the real model or uses the network. The walk, the
attention check, the rotary positions, the cache and the output head are
tested on a tiny LFM2 built with mlx-lm's own ``lfm2`` code and random
weights: the same layers as the real model, a few numbers wide. The reply
loop is tested with a fake ``stream_generate`` and a fake clock. MLX installs
only on Apple-silicon Macs; elsewhere this file is skipped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlx_lm", reason="MLX and mlx-lm install only on Apple silicon")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from chapters import use_chapter  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import lfm2  # noqa: E402

CHAPTER = use_chapter("10-meet-lfm")

import first_reply  # noqa: E402
import inside  # noqa: E402
import lfm  # noqa: E402
import lfm_files  # noqa: E402

# A tiny LFM2: 14 layers so the last 12 exist, attention where layer 2 is the
# first attention layer (as in the real model), 4 query heads sharing 2
# key/value heads. Widths are multiples of 64 so the layers can be quantized.
TINY_LAYERS = ["conv", "conv", "full_attention", "conv", "conv", "full_attention"]
TINY_LAYERS += ["conv"] * 3 + ["full_attention"] + ["conv"] * 3 + ["full_attention"]
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
VOCAB = 64
# The tiny prompt's pieces: ' Paris' and ' customers' are where inside.py
# looks for them; the ids are any distinct ids below VOCAB.
PIECES = ["<s>", "How", " many", " customers", " live", " in", " Paris", "?"]
IDS = [1, 5, 9, 13, 17, 21, 25, 29]


def tiny_model(seed: int = 0) -> nn.Module:
    """Build a tiny LFM2 with mlx-lm's own code and random weights.

    Args:
        seed: The random seed for the weights.

    Returns:
        The model, every weight evaluated.
    """
    mx.random.seed(seed)
    args = lfm2.ModelArgs(
        model_type="lfm2",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=len(TINY_LAYERS),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=4096,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        block_dim=HIDDEN,
        block_multiple_of=64,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        block_ff_dim=2 * HIDDEN,
        rope_theta=1e7,
        layer_types=TINY_LAYERS,
    )
    model = lfm2.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model() -> nn.Module:
    """The tiny model, built once.

    Returns:
        The model.
    """
    return tiny_model()


@pytest.fixture(scope="module")
def prompt() -> inside.TokensIn:
    """The tiny prompt.

    Returns:
        Its ids and pieces.
    """
    return inside.TokensIn("How many customers live in Paris?", IDS, PIECES)


class FakeTokenizer:
    """Stands in for mlx-lm's tokenizer: decodes ids, renders and encodes chats."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def decode(self, ids: list[int]) -> str:
        """Write ids as text.

        Args:
            ids: Token ids.

        Returns:
            Such as ``"<7>"``.
        """
        return "".join(f"<{token}>" for token in ids)

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> str:
        """Record the call and render the chat simply.

        Args:
            messages: The chat.
            **options: The template options.

        Returns:
            The rendered text.
        """
        self.calls.append(("template", messages, options))
        return "|".join(message["content"] for message in messages) + "|<think>"

    def encode(self, text: str, **options: Any) -> list[int]:
        """Record the call and turn each character into an id.

        Args:
            text: The text.
            **options: The encoding options.

        Returns:
            One id per character.
        """
        self.calls.append(("encode", text, options))
        return [ord(character) for character in text]


def loaded_tiny(model: nn.Module, folder: Path) -> lfm.LoadedModel:
    """Wrap the tiny model the way load_model wraps the real one.

    Args:
        model: The tiny model.
        folder: A folder holding its config.json.

    Returns:
        The loaded model.
    """
    folder.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"tie_word_embeddings": True})
    (folder / "config.json").write_text(config, encoding="utf-8")
    return lfm.LoadedModel(
        model,
        FakeTokenizer(),
        lfm_files.BF16,
        folder,
        lfm.weight_bytes(model),
        lfm.parameter_count(model),
    )


# ------------------------------------------------------------------ lfm.py


def test_chat_prompt_renders_as_the_project_engine_does() -> None:
    """Generation prompt on, reasoning kept, no extra tokens when encoding."""
    tokenizer = FakeTokenizer()
    ids = lfm.chat_prompt(tokenizer, "hi")
    (kind, messages, options), (kind2, text, encode_options) = tokenizer.calls
    assert kind == "template" and kind2 == "encode"
    assert messages == [{"role": "user", "content": "hi"}]
    assert options == {
        "add_generation_prompt": True,
        "tokenize": False,
        "preserve_thinking": True,
    }
    assert encode_options == {"add_special_tokens": False}
    assert ids == [ord(character) for character in text]


def test_parameter_count_unpacks_quantized_layers(model: nn.Module) -> None:
    """A 4-bit copy of the tiny model counts the same parameters."""
    arrays = [array for _, array in tree_flatten(model.parameters())]
    count = lfm.parameter_count(model)
    assert count == sum(array.size for array in arrays)
    packed = tiny_model()
    nn.quantize(packed, group_size=64, bits=4)
    assert lfm.parameter_count(packed) == count
    assert lfm.weight_bytes(packed) < lfm.weight_bytes(model)


def test_weight_bytes_adds_up_every_array(model: nn.Module) -> None:
    """The tiny model's random weights are float32: 4 bytes per number."""
    assert lfm.weight_bytes(model) == 4 * lfm.parameter_count(model)


def test_load_model_names_the_download_when_nothing_is_there() -> None:
    """Without the files, loading stops with the download command."""
    with pytest.raises(SystemExit, match="uv run download.py"):
        lfm.load_model(fast=False)


# --------------------------------------------------------------- inside.py


def test_the_walk_is_exactly_one_call_of_the_model(model: nn.Module) -> None:
    """Running the layers one by one gives mlx-lm's scores bit for bit."""
    steps = inside.walk(model, IDS)
    reference = model(mx.array([IDS]))
    assert mx.array_equal(steps.logits, reference).item()
    assert len(steps.layer_inputs) == len(TINY_LAYERS)
    assert steps.final.shape == (1, len(IDS), HIDDEN)


def test_the_embedding_row_is_the_tokens_vector(
    model: nn.Module, prompt: inside.TokensIn
) -> None:
    """Looking a token up is taking its row of the table."""
    table = inside.embedding_table(model)
    shown = inside.look_up(table, prompt)
    paris = IDS[PIECES.index(" Paris")]
    assert shown.token_id == paris
    assert shown.vector == table[paris, : inside.SHOWN_NUMBERS].tolist()
    assert (shown.rows, shown.columns) == (VOCAB, HIDDEN)
    assert shown.prompt_shape == (len(IDS), HIDDEN)


def test_a_quantized_table_is_unpacked_to_the_same_rows() -> None:
    """A 4-bit embedding table unpacks to what its lookup gives."""
    packed = tiny_model()
    nn.quantize(packed, group_size=64, bits=4)
    table = inside.embedding_table(packed)
    looked_up = packed.model.embed_tokens(mx.array([IDS]))[0]
    assert table.shape == (VOCAB, HIDDEN)
    assert mx.allclose(table[mx.array(IDS)], looked_up).item()


def test_bfloat16_bits_of_a_real_weight() -> None:
    """The ' Paris' row's first number, taken apart bit by bit."""
    value = 0.0294189453125
    assert inside.bfloat16_bits(value) == 0b0_01111001_1110001
    assert inside.split_bits(value) == ("0", "01111001", "1110001")
    below, above = inside.neighbours(value)
    assert (below, above) == (0.029296875, 0.029541015625)
    assert above - value == value - below == 2**-13
    with pytest.raises(ValueError, match="not exactly a bfloat16"):
        inside.bfloat16_bits(0.1)


def test_bfloat16_grids() -> None:
    """Between 8 and 16 the step is 1/16; between 16 and 32 it is 1/8."""
    assert inside.bfloat16_grid(-10.3) == inside.Grid(8.0, 16.0, 1 / 16)
    grid = inside.bfloat16_grid(23.25)
    assert grid == inside.Grid(16.0, 32.0, 1 / 8)
    assert grid.holds(22.875) and not grid.holds(22.9) and not grid.holds(8.0)


def test_precision_shows_what_bfloat16_rounds_away() -> None:
    """1 + 0.001 is 1 in bfloat16; -10.3 is stored as -10.3125."""
    shown = inside.Embedding(1, 1, 0, "x", [0.0294189453125], (1, 1))
    numbers = inside.take_apart(shown, parameters=10)
    assert numbers.index == 0 and numbers.value == 0.0294189453125
    assert numbers.sum_inputs == (1.0, 0.001) and numbers.stored_input == -10.3
    assert numbers.small_step[0] == 1.0
    assert numbers.small_step[1] == pytest.approx(1.001)
    assert numbers.log_prob[0] == -10.3125
    assert numbers.log_prob[1] == pytest.approx(-10.3)
    assert numbers.finer == 65_535
    assert (numbers.bf16_bytes, numbers.float32_bytes) == (20, 40)


def test_the_weight_taken_apart_is_the_first_positive_one() -> None:
    """A negative or zero first number is skipped: its neighbours are simple."""
    shown = inside.Embedding(1, 1, 0, "x", [-0.5, 0.0, 0.25], (1, 1))
    numbers = inside.take_apart(shown, parameters=1)
    assert (numbers.index, numbers.value) == (2, 0.25)
    assert numbers.below < 0.25 < numbers.above
    with pytest.raises(ValueError, match="no positive one"):
        inside.take_apart(inside.Embedding(1, 1, 0, "x", [-1.0, 0.0], (1, 1)), 1)
    for value in (0.0, -0.5):
        with pytest.raises(ValueError, match="not positive"):
            inside.neighbours(value)


def test_the_layer_map_and_what_each_kind_keeps(model: nn.Module) -> None:
    """A conv layer keeps its last 2 inputs; an attention layer every token's."""
    layers = inside.map_layers(model, IDS)
    assert layers.attention_layers() == [2, 5, 9, 13]
    assert layers.kinds == TINY_LAYERS
    assert layers.kernel == 3
    assert layers.conv_state == (1, 2, HIDDEN)
    assert layers.kv_shape == (1, KV_HEADS, len(IDS), HIDDEN // HEADS)
    assert layers.lora_layers == list(range(2, 14))


def test_heads_measure_the_grouped_shapes(model: nn.Module) -> None:
    """Queries have 4 heads, keys 2: each pair of query heads shares one."""
    steps = inside.walk(model, IDS)
    kinds = inside.map_layers(model, IDS).kinds
    group = inside.measure_heads(model, steps.layer_inputs[2], kinds)
    head_dim = HIDDEN // HEADS
    assert group.queries == (1, HEADS, len(IDS), head_dim)
    assert group.keys == (1, KV_HEADS, len(IDS), head_dim)
    assert group.group() == 2
    assert group.attention_layers == 4


def test_the_real_models_cache_is_16_kb_per_token() -> None:
    """8 layers x keys and values x 8 heads x 64 numbers x 2 bytes."""
    real = inside.Heads(2, 2048, 32, 8, 64, (1, 32, 17, 64), (1, 8, 17, 64), 8)
    assert real.cache_bytes_per_token(real.kv_heads) == 16_384
    assert real.cache_bytes_per_token(real.query_heads) == 65_536
    assert real.group() == 4


def test_causal_weights_never_look_ahead() -> None:
    """Every weight above the diagonal is exactly 0, and each row sums to 1."""
    mx.random.seed(1)
    queries = mx.random.normal((1, 4, 5, 8))
    keys = mx.random.normal((1, 2, 5, 8))
    weights = inside.causal_weights(queries, keys, 8**-0.5)
    upper = mx.triu(mx.ones((5, 5)), k=1).astype(mx.bool_)
    assert mx.all(mx.where(upper, weights, 0) == 0).item()
    assert mx.allclose(weights.sum(axis=-1), mx.ones((1, 4, 5))).item()


def test_the_hand_computed_attention_matches_mlx_lm(
    model: nn.Module, prompt: inside.TokensIn
) -> None:
    """The layer's output, by hand from q, k and v, is mlx-lm's own."""
    steps = inside.walk(model, IDS)
    weights = inside.attend_by_hand(model, steps.layer_inputs[2], prompt)
    assert weights.largest_difference < 1e-3 * max(weights.largest_output, 1e-6)
    assert len(weights.weights) == len(IDS)
    for row, values in enumerate(weights.weights):
        assert all(value == 0 for value in values[row + 1 :])
        assert sum(values) == pytest.approx(1.0, abs=1e-5)


def test_rope_scores_depend_only_on_the_distance(
    model: nn.Module, prompt: inside.TokensIn
) -> None:
    """The same pair scores the same 1,000 positions later, not farther apart."""
    steps = inside.walk(model, IDS)
    turns = inside.move_pair(model, steps.layer_inputs[2], prompt, 4096)
    assert turns.near_positions == (6, 3)
    assert turns.far_positions == (1006, 1003)
    assert turns.far == pytest.approx(turns.near, abs=1e-4)
    assert [distance for distance, _ in turns.spread] == [30, 300, 3000]
    assert turns.pairs == HIDDEN // HEADS // 2
    assert turns.fastest == pytest.approx(2 * math.pi)


def test_turn_period_of_the_real_settings() -> None:
    """rope_theta 1e7, 32 pairs: 6.3 positions per turn up to 38 million."""
    assert inside.turn_period(1e7, 0, 32) == pytest.approx(2 * math.pi)
    assert inside.turn_period(1e7, 31, 32) == pytest.approx(37_969_062, rel=1e-6)


def test_the_table_is_the_head(
    model: nn.Module, prompt: inside.TokensIn, tmp_path: Path
) -> None:
    """Vector x embedding table gives exactly the model's own scores."""
    loaded = loaded_tiny(model, tmp_path)
    steps = inside.walk(model, IDS)
    reference = model(mx.array([IDS]))
    scores = inside.score_tokens(
        loaded, inside.embedding_table(model), steps, reference
    )
    assert scores.largest_difference == 0
    assert scores.tied and scores.vocabulary == VOCAB and scores.hidden == HIDDEN
    assert len(scores.top) == inside.TOP_TOKENS
    probabilities = [probability for _, _, probability in scores.top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert not hasattr(model, "lm_head")


def test_tied_scores_are_ordered_by_token_id(model: nn.Module, tmp_path: Path) -> None:
    """Equal scores list the lowest id first, the token a greedy pick takes."""
    loaded = loaded_tiny(model, tmp_path)
    steps = inside.walk(model, IDS)
    values = [0.0] * VOCAB
    values[7] = values[3] = values[40] = 2.5  # a three-way tie at the top
    values[9] = 1.0
    reference = mx.array(values, dtype=mx.bfloat16).reshape(1, 1, VOCAB)
    table = inside.embedding_table(model)
    scores = inside.score_tokens(loaded, table, steps, reference)
    assert [piece for piece, _, _ in scores.top[:4]] == ["<3>", "<7>", "<40>", "<9>"]
    assert scores.top[0][0] == f"<{mx.argmax(reference[0, -1]).item()}>"


# ---------------------------------------------------------- first_reply.py


@dataclass
class FakeStep:
    """One step as stream_generate yields it.

    Attributes:
        text: The text it adds.
        token: Its token id.
        finish_reason: None, ``"stop"`` or ``"length"``.
    """

    text: str
    token: int
    finish_reason: str | None = None


@dataclass
class FakeClock:
    """A clock that moves by the given steps, one per reading.

    Attributes:
        times: The readings, in order.
    """

    times: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        """Read the clock.

        Returns:
            The next reading.
        """
        return self.times.pop(0)


def test_collect_times_each_token_and_drops_the_end_of_turn() -> None:
    """The end-of-turn step is timed but is not part of the reply."""
    steps = [
        FakeStep("SELECT", 10),
        FakeStep(" 1", 11),
        FakeStep("", 124900, "stop"),
    ]
    shown: list[str] = []
    clock = FakeClock([0.0, 0.2, 0.25, 0.31])
    pieces, finish, end_seconds = first_reply.collect(steps, shown.append, clock)
    assert [piece.token for piece in pieces] == [10, 11]
    assert [round(piece.seconds, 3) for piece in pieces] == [0.2, 0.05]
    assert finish == "stop" and round(end_seconds, 3) == 0.06
    assert shown == ["SELECT", " 1", ""]


def test_collect_keeps_the_last_token_when_the_cap_cuts_in() -> None:
    """With finish_reason length, the last step's token is the reply's."""
    steps = [FakeStep("a", 1), FakeStep("b", 2, "length")]
    clock = FakeClock([0.0, 0.1, 0.2])
    pieces, finish, end_seconds = first_reply.collect(steps, lambda text: None, clock)
    assert [piece.text for piece in pieces] == ["a", "b"]
    assert finish == "length" and end_seconds == 0.0


def test_the_reply_reports_first_token_and_ms_per_token() -> None:
    """The first token includes the prompt; the rest are averaged."""
    pieces = [
        first_reply.Piece(1, "a", 0.3),
        first_reply.Piece(2, "b", 0.05),
        first_reply.Piece(3, "c", 0.07),
    ]
    reply = first_reply.Reply("q", 23, pieces, "stop", 0.06, 5 * 10**9)
    assert reply.text() == "abc"
    assert reply.first_seconds() == 0.3
    assert reply.decode_steps() == [0.05, 0.07, 0.06]
    assert reply.ms_per_token() == pytest.approx(60.0)
    empty = first_reply.Reply("q", 23, [], "length", 0.0, 0)
    assert empty.first_seconds() == 0.0 and empty.ms_per_token() == 0.0


def test_answer_streams_and_warm_up_is_not_timed(
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The warm-up runs first; the reply's pieces are printed as they come."""
    calls: list[int] = []

    def fake_stream(
        network: Any, tokenizer: Any, ids: list[int], max_tokens: int
    ) -> Iterator[FakeStep]:
        calls.append(max_tokens)
        yield FakeStep("SELECT", 7)
        yield FakeStep(" 1;", 8)
        yield FakeStep("", 124900, "stop")

    monkeypatch.setattr(first_reply, "stream_generate", fake_stream)
    loaded = loaded_tiny(model, tmp_path)
    reply = first_reply.answer(loaded, "How many?", 50)
    assert calls == [first_reply.WARM_UP_TOKENS, 50]
    assert reply.text() == "SELECT 1;" and reply.finish_reason == "stop"
    printed = capsys.readouterr().out
    assert "warm-up" in printed and "SELECT 1;" in printed
    assert reply.prompt_tokens == len("How many?|<think>")


def test_the_measurement_is_saved_under_course_home(
    model: nn.Module, tmp_path: Path, course_home: Path
) -> None:
    """The record goes to $COURSE_HOME/10-meet-lfm, named by the storage."""
    loaded = loaded_tiny(model, tmp_path)
    pieces = [first_reply.Piece(1, "a", 0.3), first_reply.Piece(2, "b", 0.05)]
    reply = first_reply.Reply("q", 23, pieces, "stop", 0.05, 123)
    path = first_reply.save(first_reply.record(loaded, reply), "bfloat16")
    assert path == course_home / "10-meet-lfm" / "first_reply-bfloat16.json"
    data = json.loads(path.read_text())
    assert data["reply"] == "ab" and data["reply_tokens"] == 2
    assert data["ms_per_token"] == pytest.approx(50.0)
    assert data["peak_memory_bytes"] == 123
    assert data["revision"] == lfm_files.BF16.revision


def test_every_stage_prints_on_the_tiny_model(
    prompt: inside.TokensIn,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole of inside.py's report runs on the tiny model, all 8 stages.

    The model is stored in bfloat16, as the real one is, so its weights can
    be taken apart bit by bit.
    """
    model = tiny_model()
    model.set_dtype(mx.bfloat16)
    loaded = loaded_tiny(model, tmp_path)
    steps = inside.walk(model, IDS)
    reference = model(mx.array([IDS]))
    table = inside.embedding_table(model)
    attention_in = steps.layer_inputs[inside.ATTENTION_LAYER]
    inside.report.print_loaded(loaded, 0.0)
    inside.report.print_tokens_in(prompt)
    shown = inside.look_up(table, prompt)
    inside.report.print_embedding(shown)
    inside.report.print_precision(inside.take_apart(shown, loaded.parameters), loaded)
    layers = inside.map_layers(model, IDS)
    inside.report.print_layer_map(layers)
    heads = inside.measure_heads(model, attention_in, layers.kinds)
    inside.report.print_heads(heads, inside.ENGINE_PROMPT_CAP, inside.LONG_ROW)
    mask = inside.attend_by_hand(model, attention_in, prompt)
    inside.report.print_mask(mask, inside.ATTENTION_LAYER, inside.SHOWN_HEAD)
    inside.report.print_rope(inside.move_pair(model, attention_in, prompt, 4096))
    scores = inside.score_tokens(loaded, table, steps, reference)
    inside.report.print_head(scores, inside.LONG_ROW)
    printed = capsys.readouterr().out
    for stage in range(1, 9):
        assert f"\n{stage}. " in printed
    assert "query heads 0-1 read key/value head 0" in printed
    assert "the same: only the distance counts" in printed
    grid = printed.split("6. the causal mask")[1].split("7. ")[0]
    assert f"{inside.report.UNMASKED:>{inside.report.GRID_CELL}}" not in grid


def test_the_timing_summary_prints(
    model: nn.Module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lines after the reply: length, first token, ms per token, memory."""
    loaded = loaded_tiny(model, tmp_path)
    pieces = [
        first_reply.Piece(597, "The", 0.15),
        first_reply.Piece(4695, " user", 0.05),
    ]
    reply = first_reply.Reply("q", 23, pieces, "stop", 0.05, 5_450_000_000)
    first_reply.report.print_model(loaded)
    first_reply.report.print_timing(reply, loaded)
    first_reply.report.print_saved(tmp_path / "x.json", loaded.repo)
    printed = capsys.readouterr().out
    assert "the reply: 2 tokens, then <|im_end|>, its end of turn" in printed
    assert "first token after 0.15 s: reading the 23-token prompt" in printed
    assert "then 50.0 ms per token (20.0 tokens a second)" in printed
    assert "peak memory 5.45 GB (MLX's count)" in printed
    assert "     150     597  'The'" in printed
    assert "chapter 11's 'before' picture" in printed
