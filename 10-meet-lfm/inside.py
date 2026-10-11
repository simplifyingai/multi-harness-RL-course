# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Walk one prompt through LFM2.5-2.6B, layer by layer, and look at each stage.

The model is loaded with mlx-lm, and the prompt for chapter 9's question goes
through it one stage at a time, exactly as mlx-lm runs it: each token becomes
a vector (a row of the embedding table), its numbers are stored in bfloat16,
30 layers of two kinds (short convolutions and attention) rework the vectors,
one attention layer's 32 query heads share 8 key/value heads, a causal mask
keeps every token from looking ahead, rotary positions tell the tokens apart,
and the same embedding table, used backwards, scores every one of the 128,000
tokens for the next position. Every hand computation is checked against
mlx-lm's own.

Run it with ``uv run inside.py`` after ``uv run download.py`` (it needs an
Apple-silicon Mac). ``--fast`` uses the 4-bit export instead.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import report
from lfm import LoadedModel, chat_prompt, load_model
from lfm_files import BF16_BYTES, FLOAT32_BYTES, LAYER_ATTENTION, LAYER_CONV
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache

SCRIPT = "inside.py"
CONFIG_FILE = "config.json"
TITLE = "one prompt walked through LFM2.5-2.6B"
QUESTION = "How many customers live in Paris?"  # chapter 9's question
SHOWN_TOKEN = " Paris"  # the token whose vector, query and key are shown
ASKED_TOKEN = " customers"  # the token whose key ' Paris' is compared with
SHOWN_NUMBERS = 6  # numbers of a vector printed
ATTENTION_LAYER = 2  # the first attention layer
SHOWN_HEAD = 0  # the query head whose weights are drawn
TOP_TOKENS = 5
LAST_LAYERS = 12  # chapter 13 trains LoRA add-ons on the last 12 layers
ENGINE_PROMPT_CAP = 16_384  # the project's engine refuses longer prompts
LONG_ROW = 8_192  # chapter 25's longest training rows
ROPE_DISTANCES = (30, 300, 3_000)  # how far apart the moved pair is placed
ROPE_FAR = 1_000  # how far the same-distance pair is moved
# The two bfloat16 rules, computed in MLX: a small step lost next to 1, and a
# log-probability rounded to the 1/16 grid bfloat16 has between 8 and 16.
SMALL_STEP = (1.0, 0.001)
LOG_PROB = -10.3
BF16_SIGN_BITS = 1
BF16_EXPONENT_BITS = 8
BF16_FRACTION_BITS = 7
FLOAT32_EXTRA_BITS = 16  # float32 has 16 more fraction bits
FLOAT32_FORMAT = "<f"  # struct: a little-endian float32
BITS32_FORMAT = "<I"  # struct: the same 4 bytes as an unsigned integer
LOW_HALF = (1 << FLOAT32_EXTRA_BITS) - 1  # the 16 bits bfloat16 drops
TIE_KEY = "tie_word_embeddings"


@dataclass(frozen=True)
class TokensIn:
    """The prompt as tokens.

    Attributes:
        question: The user's question.
        ids: The prompt's token ids.
        pieces: The text of each token.
    """

    question: str
    ids: list[int]
    pieces: list[str]


@dataclass(frozen=True)
class Embedding:
    """One token's row of the embedding table.

    Attributes:
        rows: Rows in the table (one per token id).
        columns: Numbers in each row.
        token_id: The token shown.
        piece: Its text.
        vector: Its first few numbers.
        prompt_shape: The prompt as a matrix: tokens x numbers.
    """

    rows: int
    columns: int
    token_id: int
    piece: str
    vector: list[float]
    prompt_shape: tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """The spacing of bfloat16 numbers between two powers of two.

    Attributes:
        low: The power of two at or below the number's size.
        high: Twice that.
        step: The gap between neighbouring bfloat16 numbers in that range.
    """

    low: float
    high: float
    step: float

    def holds(self, value: float) -> bool:
        """Check that a number sits in this range, on its grid.

        Args:
            value: Any number.

        Returns:
            True if its size is in [low, high) and it is a whole number of steps.
        """
        size = abs(value)
        return self.low <= size < self.high and (size / self.step).is_integer()


@dataclass(frozen=True)
class Precision:
    """One real weight's 16 bits, and what bfloat16 rounds away.

    Attributes:
        index: Where the weight sits in the row.
        value: The weight.
        sign: Its sign bit.
        exponent: Its 8 exponent bits.
        fraction: Its 7 fraction bits.
        below: The next bfloat16 number down.
        above: The next bfloat16 number up.
        small_step: (bfloat16, float32) results of 1 + 0.001.
        log_prob: (bfloat16, float32) results of storing -10.3.
        sum_inputs: The two numbers added, 1 and 0.001.
        stored_input: The number stored, -10.3.
        log_grid: The bfloat16 grid that -10.3 is rounded to.
        finer: How many float32 numbers fit inside one bfloat16 step.
        bf16_bytes: The model's weights in bfloat16.
        float32_bytes: The same numbers in float32.
    """

    index: int
    value: float
    sign: str
    exponent: str
    fraction: str
    below: float
    above: float
    small_step: tuple[float, float]
    log_prob: tuple[float, float]
    sum_inputs: tuple[float, float]
    stored_input: float
    log_grid: Grid
    finer: int
    bf16_bytes: int
    float32_bytes: int


@dataclass(frozen=True)
class LayerMap:
    """The kind of every layer, and what each kind keeps in the cache.

    Attributes:
        kinds: config.json's layer type per layer (``conv`` or
            ``full_attention``).
        kernel: How many tokens a convolution sees (conv_L_cache).
        conv_state: The shape a convolution layer keeps after the prompt.
        kv_shape: The shape of an attention layer's keys after the prompt.
        lora_layers: The last 12 layers, which chapter 13 trains.
    """

    kinds: list[str]
    kernel: int
    conv_state: tuple[int, ...]
    kv_shape: tuple[int, ...]
    lora_layers: list[int]

    def attention_layers(self) -> list[int]:
        """List the attention layers.

        Returns:
            Their numbers, counting from 0.
        """
        return [i for i, kind in enumerate(self.kinds) if kind == LAYER_ATTENTION]


@dataclass(frozen=True)
class Heads:
    """One attention layer's heads and the memory they need.

    Attributes:
        layer: The layer.
        hidden: Numbers per token going in.
        query_heads: Query heads.
        kv_heads: Key/value heads.
        head_dim: Numbers per head.
        queries: Shape of the queries for the prompt.
        keys: Shape of the keys for the prompt.
        attention_layers: How many attention layers the model has.
    """

    layer: int
    hidden: int
    query_heads: int
    kv_heads: int
    head_dim: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    attention_layers: int

    def group(self) -> int:
        """Count the query heads that share one key/value head.

        Returns:
            Query heads per key/value head.
        """
        return self.query_heads // self.kv_heads

    def cache_bytes_per_token(self, heads_kept: int) -> int:
        """Work out the bytes of keys and values kept per token, all layers.

        Args:
            heads_kept: Key/value heads per layer: the model's own 8, or
                32 for the what-if where every query head has its own.

        Returns:
            Bytes per token in bfloat16.
        """
        return self.attention_layers * 2 * heads_kept * self.head_dim * BF16_BYTES


@dataclass(frozen=True)
class Mask:
    """One head's attention weights over the prompt, and the check.

    Attributes:
        weights: Row by row, how much each token attends to each earlier one.
        pieces: The prompt's tokens, for the labels.
        largest_difference: The hand-computed layer output against mlx-lm's.
        largest_output: The size of that output, for scale.
    """

    weights: list[list[float]]
    pieces: list[str]
    largest_difference: float
    largest_output: float


@dataclass(frozen=True)
class Rope:
    """The rotary position settings, and one query-key score moved about.

    Attributes:
        theta: rope_theta.
        pairs: Pairs of numbers each head turns.
        fastest: Positions per full turn of the fastest pair.
        slowest: Positions per full turn of the slowest pair.
        positions: max_position_embeddings.
        near: The score with the two tokens where they sit.
        near_positions: Where they sit.
        far: The same distance, moved far along.
        far_positions: Where they are moved to.
        spread: The score at each larger distance.
    """

    theta: float
    pairs: int
    fastest: float
    slowest: float
    positions: int
    near: float
    near_positions: tuple[int, int]
    far: float
    far_positions: tuple[int, int]
    spread: list[tuple[int, float]]


@dataclass(frozen=True)
class Head:
    """The last position's scores for every token.

    Attributes:
        vocabulary: How many tokens are scored.
        hidden: Numbers in the last vector.
        top: The highest-scoring tokens: (piece, score, probability).
        largest_difference: Scores by hand against mlx-lm's.
        tied: Whether config.json ties the head to the embedding table.
        score_grid: The bfloat16 grid the highest score sits on.
        on_grid: Whether every top score in that range sits on that grid.
    """

    vocabulary: int
    hidden: int
    top: list[tuple[str, float, float]]
    largest_difference: float
    tied: bool
    score_grid: Grid
    on_grid: bool


@dataclass(frozen=True)
class Walk:
    """The prompt's vectors at each stage of one pass through the model.

    Attributes:
        layer_inputs: The vectors going into each layer.
        final: After the last layer and the final norm.
        logits: The scores for every token at every position.
    """

    layer_inputs: list[mx.array]
    final: mx.array
    logits: mx.array


def walk(model: nn.Module, ids: list[int]) -> Walk:
    """Run the prompt through the model one layer at a time.

    Line for line what mlx-lm's ``Lfm2Model`` does in one call, keeping each
    layer's input so it can be looked at.

    Args:
        model: The loaded model.
        ids: The prompt's token ids.

    Returns:
        The vectors at each stage.
    """
    inner = model.model
    vectors = inner.embed_tokens(mx.array([ids]))
    attention_mask = create_attention_mask(vectors, None)  # "causal"
    conv_mask = create_ssm_mask(vectors, None)  # nothing to mask in one prompt
    inputs = []
    for layer in inner.layers:
        inputs.append(vectors)
        mask = attention_mask if layer.is_attention_layer else conv_mask
        vectors = layer(vectors, mask, cache=None)
    final = inner.embedding_norm(vectors)
    return Walk(inputs, final, inner.embed_tokens.as_linear(final))


def embedding_table(model: nn.Module) -> mx.array:
    """Return the embedding table as real numbers, unpacking a 4-bit one.

    Args:
        model: The loaded model.

    Returns:
        The table: one row of numbers per token id.
    """
    embed = model.model.embed_tokens
    if not hasattr(embed, "scales"):
        return embed.weight
    return mx.dequantize(
        embed.weight,
        embed.scales,
        embed.biases,
        group_size=embed.group_size,
        bits=embed.bits,
    )


def render_prompt(loaded: LoadedModel, question: str) -> TokensIn:
    """Render the question into prompt tokens.

    Args:
        loaded: The model.
        question: The user's question.

    Returns:
        The prompt's ids and pieces.
    """
    ids = chat_prompt(loaded.tokenizer, question)
    return TokensIn(question, ids, [loaded.tokenizer.decode([i]) for i in ids])


def look_up(table: mx.array, prompt: TokensIn) -> Embedding:
    """Look up the shown token's row of the embedding table.

    Args:
        table: The embedding table.
        prompt: The prompt.

    Returns:
        The row and the shapes.

    Raises:
        ValueError: If the prompt has no ' Paris' token.
    """
    position = prompt.pieces.index(SHOWN_TOKEN)
    token_id = prompt.ids[position]
    rows, columns = table.shape
    vector = table[token_id, :SHOWN_NUMBERS].tolist()
    return Embedding(
        rows, columns, token_id, SHOWN_TOKEN, vector, (len(prompt.ids), columns)
    )


def bfloat16_bits(value: float) -> int:
    """Return the 16 bits of a number that bfloat16 holds exactly.

    bfloat16 is the top half of a float32: the same sign and 8 exponent bits,
    and the first 7 of float32's 23 fraction bits.

    Args:
        value: A number with an exact bfloat16 form (such as a model weight).

    Returns:
        Its 16 bits, as an integer.

    Raises:
        ValueError: If the number needs more than bfloat16's bits.
    """
    (bits32,) = struct.unpack(BITS32_FORMAT, struct.pack(FLOAT32_FORMAT, value))
    if bits32 & LOW_HALF:
        raise ValueError(f"{value} is not exactly a bfloat16 number")
    return bits32 >> FLOAT32_EXTRA_BITS


def from_bfloat16_bits(bits: int) -> float:
    """Turn 16 bfloat16 bits back into the number.

    Args:
        bits: The 16 bits, as an integer.

    Returns:
        The number.
    """
    packed = struct.pack(BITS32_FORMAT, bits << FLOAT32_EXTRA_BITS)
    (value,) = struct.unpack(FLOAT32_FORMAT, packed)
    return value


def neighbours(value: float) -> tuple[float, float]:
    """Find the bfloat16 numbers just below and just above a positive one.

    For a positive number, one less in its 16 bits is the next number down
    and one more the next number up.

    Args:
        value: A positive number with an exact bfloat16 form.

    Returns:
        The next number down and the next number up.

    Raises:
        ValueError: If the number is not positive, or not exactly bfloat16.
    """
    if value <= 0:
        raise ValueError(f"{value} is not positive")
    bits = bfloat16_bits(value)
    return from_bfloat16_bits(bits - 1), from_bfloat16_bits(bits + 1)


def bfloat16_grid(value: float) -> Grid:
    """Find the spacing of bfloat16 numbers around a number.

    bfloat16 keeps 7 fraction bits, so between two powers of two, low and
    2 x low, it holds 2**7 = 128 evenly spaced numbers: a step of low / 128.

    Args:
        value: Any non-zero number.

    Returns:
        The range it falls in and the step there.
    """
    low = 2.0 ** math.floor(math.log2(abs(value)))
    return Grid(low, 2 * low, low / 2**BF16_FRACTION_BITS)  # up to the next power


def split_bits(value: float) -> tuple[str, str, str]:
    """Write a bfloat16 number's sign, exponent and fraction bits.

    Args:
        value: A number with an exact bfloat16 form.

    Returns:
        The three fields as strings of 0s and 1s.
    """
    total = BF16_SIGN_BITS + BF16_EXPONENT_BITS + BF16_FRACTION_BITS
    text = format(bfloat16_bits(value), f"0{total}b")
    exponent_end = BF16_SIGN_BITS + BF16_EXPONENT_BITS
    return text[:BF16_SIGN_BITS], text[BF16_SIGN_BITS:exponent_end], text[exponent_end:]


def in_both(compute: Callable[[mx.Dtype], mx.array]) -> tuple[float, float]:
    """Run one small computation in bfloat16 and in float32, with MLX.

    Args:
        compute: Takes a dtype and returns a scalar array computed in it.

    Returns:
        The bfloat16 result and the float32 result.
    """
    return compute(mx.bfloat16).item(), compute(mx.float32).item()


def take_apart(embed: Embedding, parameters: int) -> Precision:
    """Take one weight apart, and show what bfloat16 rounds away.

    The weight is the first positive number of the shown row, so its
    neighbours below and above are simply one less and one more in its bits.

    Args:
        embed: The shown row of the embedding table.
        parameters: How many numbers the model has.

    Returns:
        The bits, the neighbours and the two rules.

    Raises:
        ValueError: If the shown numbers hold no positive one.
    """
    index = next((i for i, number in enumerate(embed.vector) if number > 0), None)
    if index is None:
        raise ValueError("the shown numbers hold no positive one")
    value = embed.vector[index]
    sign, exponent, fraction = split_bits(value)
    below, above = neighbours(value)
    one, step = SMALL_STEP
    small_step = in_both(
        lambda dtype: mx.array(one, dtype=dtype) + mx.array(step, dtype=dtype)
    )
    log_prob = in_both(lambda dtype: mx.array(LOG_PROB, dtype=dtype))
    return Precision(
        index,
        value,
        sign,
        exponent,
        fraction,
        below,
        above,
        small_step,
        log_prob,
        SMALL_STEP,
        LOG_PROB,
        bfloat16_grid(LOG_PROB),
        2**FLOAT32_EXTRA_BITS - 1,
        parameters * BF16_BYTES,
        parameters * FLOAT32_BYTES,
    )


def map_layers(model: nn.Module, ids: list[int]) -> LayerMap:
    """Read each layer's kind, and measure what each kind keeps in the cache.

    Args:
        model: The loaded model.
        ids: The prompt's token ids.

    Returns:
        The map and the cache shapes after the prompt.
    """
    layers = model.layers
    kinds = [
        LAYER_ATTENTION if layer.is_attention_layer else LAYER_CONV for layer in layers
    ]
    cache = make_prompt_cache(model)
    model(mx.array([ids]), cache=cache)
    conv_index = kinds.index(LAYER_CONV)
    attention_index = kinds.index(LAYER_ATTENTION)
    conv_state = tuple(cache[conv_index][0].shape)
    kv = cache[attention_index]
    keys = kv.keys[..., : kv.offset, :]  # the cache keeps spare room after them
    first_conv = layers[conv_index].conv
    total = len(layers)
    return LayerMap(
        kinds,
        first_conv.L_cache,
        conv_state,
        tuple(keys.shape),
        list(range(total - LAST_LAYERS, total)),
    )


def attention_parts(
    model: nn.Module, inputs: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute one attention layer's queries, keys and values, as mlx-lm does.

    Args:
        model: The loaded model.
        inputs: The vectors going into the attention layer.

    Returns:
        The normed input, then queries, keys and values before rotation,
        shaped (batch, heads, tokens, numbers per head).
    """
    layer = model.layers[ATTENTION_LAYER]
    attention = layer.self_attn
    normed = layer.operator_norm(inputs)
    batch, tokens, _ = normed.shape
    queries = attention.q_proj(normed).reshape(batch, tokens, attention.n_heads, -1)
    keys = attention.k_proj(normed).reshape(batch, tokens, attention.n_kv_heads, -1)
    values = attention.v_proj(normed).reshape(batch, tokens, attention.n_kv_heads, -1)
    queries = attention.q_layernorm(queries).transpose(0, 2, 1, 3)
    keys = attention.k_layernorm(keys).transpose(0, 2, 1, 3)
    return normed, queries, keys, values.transpose(0, 2, 1, 3)


def measure_heads(model: nn.Module, inputs: mx.array, kinds: list[str]) -> Heads:
    """Measure the shapes of one attention layer's queries and keys.

    Args:
        model: The loaded model.
        inputs: The vectors going into the attention layer.
        kinds: Every layer's kind.

    Returns:
        The heads and their shapes.
    """
    attention = model.layers[ATTENTION_LAYER].self_attn
    _, queries, keys, _ = attention_parts(model, inputs)
    return Heads(
        ATTENTION_LAYER,
        inputs.shape[-1],
        attention.n_heads,
        attention.n_kv_heads,
        attention.head_dim,
        tuple(queries.shape),
        tuple(keys.shape),
        kinds.count(LAYER_ATTENTION),
    )


def causal_weights(queries: mx.array, keys: mx.array, scale: float) -> mx.array:
    """Turn rotated queries and keys into attention weights, with the causal mask.

    Each query head uses the key head of its group. A token's scores for
    later tokens are set to minus infinity before the softmax, so their
    weights come out exactly 0: a token never looks ahead.

    Args:
        queries: Rotated queries, (batch, query heads, tokens, numbers).
        keys: Rotated keys, (batch, key/value heads, tokens, numbers).
        scale: 1 / sqrt(numbers per head).

    Returns:
        Weights, (batch, query heads, tokens, tokens); each row sums to 1.
    """
    group = queries.shape[1] // keys.shape[1]
    keys = mx.repeat(keys, group, axis=1)  # query head h reads key head h // group
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
    tokens = queries.shape[2]
    allowed = mx.tril(mx.ones((tokens, tokens), dtype=mx.bool_))
    return mx.softmax(mx.where(allowed, scores, -mx.inf), axis=-1)


def attend_by_hand(model: nn.Module, inputs: mx.array, prompt: TokensIn) -> Mask:
    """Compute one attention layer by hand, and check it against mlx-lm.

    Args:
        model: The loaded model.
        inputs: The vectors going into the attention layer.
        prompt: The prompt, for the labels.

    Returns:
        One head's weights and the size of the difference.
    """
    attention = model.layers[ATTENTION_LAYER].self_attn
    normed, queries, keys, values = attention_parts(model, inputs)
    rotated_q = attention.rope(queries).astype(mx.float32)
    rotated_k = attention.rope(keys).astype(mx.float32)
    weights = causal_weights(rotated_q, rotated_k, attention.scale)
    group = attention.n_heads // attention.n_kv_heads
    values = mx.repeat(values.astype(mx.float32), group, axis=1)
    batch, _, tokens, _ = weights.shape
    mixed = (weights @ values).transpose(0, 2, 1, 3).reshape(batch, tokens, -1)
    by_hand = attention.out_proj(mixed.astype(normed.dtype)).astype(mx.float32)
    theirs = attention(normed, mask="causal").astype(mx.float32)
    return Mask(
        weights[0, SHOWN_HEAD].tolist(),
        prompt.pieces,
        mx.abs(by_hand - theirs).max().item(),
        mx.abs(theirs).max().item(),
    )


def turn_period(theta: float, pair: int, pairs: int) -> float:
    """Count the positions it takes one rotary pair to turn a full circle.

    Pair i turns by theta ** (-i / pairs) radians per position.

    Args:
        theta: rope_theta.
        pair: Which pair, 0 (fastest) to pairs - 1 (slowest).
        pairs: Pairs per head.

    Returns:
        Positions per full turn.
    """
    return 2 * math.pi / theta ** (-pair / pairs)


def move_pair(
    model: nn.Module, inputs: mx.array, prompt: TokensIn, positions: int
) -> Rope:
    """Score one query against one key with the pair moved to other positions.

    Args:
        model: The loaded model.
        inputs: The vectors going into the attention layer.
        prompt: The prompt, to find the two tokens.
        positions: max_position_embeddings.

    Returns:
        The settings and the scores.

    Raises:
        ValueError: If the prompt has no ' Paris' or no ' customers' token.
    """
    attention = model.layers[ATTENTION_LAYER].self_attn
    _, queries, keys, _ = attention_parts(model, inputs)
    q_at = prompt.pieces.index(SHOWN_TOKEN)
    k_at = prompt.pieces.index(ASKED_TOKEN)
    group = attention.n_heads // attention.n_kv_heads
    query = queries[:, SHOWN_HEAD : SHOWN_HEAD + 1, q_at : q_at + 1]
    key_head = SHOWN_HEAD // group
    key = keys[:, key_head : key_head + 1, k_at : k_at + 1]
    query = query.astype(mx.float32)
    key = key.astype(mx.float32)

    def score(query_position: int, key_position: int) -> float:
        turned_q = attention.rope(query, offset=query_position)
        turned_k = attention.rope(key, offset=key_position)
        return ((turned_q * turned_k).sum() * attention.scale).item()

    pairs = attention.rope.dims // 2
    theta = attention.rope.base
    far = (q_at + ROPE_FAR, k_at + ROPE_FAR)
    spread = [(gap, score(k_at + gap, k_at)) for gap in ROPE_DISTANCES]
    return Rope(
        theta,
        pairs,
        turn_period(theta, 0, pairs),
        turn_period(theta, pairs - 1, pairs),
        positions,
        score(q_at, k_at),
        (q_at, k_at),
        score(*far),
        far,
        spread,
    )


def score_tokens(
    loaded: LoadedModel, table: mx.array, steps: Walk, reference: mx.array
) -> Head:
    """Score every token for the next position, by hand and by mlx-lm.

    Args:
        loaded: The model.
        table: The embedding table.
        steps: The walk, for the last vector.
        reference: mlx-lm's own scores for the prompt.

    Returns:
        The top tokens and the check.
    """
    # One dot product per row of the table, for every position at once (the
    # same shapes mlx-lm multiplies, so the same rounding), then the last.
    by_hand = (steps.final @ table.T)[0, -1]
    scores = reference[0, -1]
    probabilities = mx.softmax(scores.astype(mx.float32))
    # bfloat16 scores tie often (they sit on a coarse grid), so equal scores
    # are ordered by token id: the first is then the token a greedy pick
    # (mx.argmax, the lowest id among equals) would take.
    values = scores.astype(mx.float32).tolist()
    order = sorted(range(len(values)), key=lambda token: (-values[token], token))
    top = [
        (
            loaded.tokenizer.decode([token]),
            values[token],
            probabilities[token].item(),
        )
        for token in order[:TOP_TOKENS]
    ]
    difference = mx.abs(by_hand.astype(mx.float32) - scores.astype(mx.float32))
    config = json.loads((loaded.folder / CONFIG_FILE).read_text(encoding="utf-8"))
    tied = bool(config.get(TIE_KEY, False))  # false when config.json omits it
    hidden = steps.final.shape[-1]
    grid = bfloat16_grid(top[0][1])
    in_range = [score for _, score, _ in top if grid.low <= abs(score) < grid.high]
    on_grid = all(grid.holds(score) for score in in_range)
    return Head(
        table.shape[0],
        hidden,
        top,
        difference.max().item(),
        tied,
        grid,
        on_grid,
    )


def main() -> None:
    """Load the model and print each stage of one prompt's pass through it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast", action="store_true", help="use the 4-bit export instead of bfloat16"
    )
    args = parser.parse_args()
    loaded = load_model(args.fast)
    model = loaded.model
    prompt = render_prompt(loaded, QUESTION)
    steps = walk(model, prompt.ids)
    reference = model(mx.array([prompt.ids]))
    walk_difference = mx.abs(steps.logits - reference).max().item()
    table = embedding_table(model)
    attention_in = steps.layer_inputs[ATTENTION_LAYER]
    positions = model.args.max_position_embeddings

    report.print_title(SCRIPT, f"{TITLE} ({loaded.repo.label})")
    report.print_loaded(loaded, walk_difference)
    report.print_tokens_in(prompt)
    shown = look_up(table, prompt)
    report.print_embedding(shown)
    report.print_precision(take_apart(shown, loaded.parameters), loaded)
    layers = map_layers(model, prompt.ids)
    report.print_layer_map(layers)
    heads = measure_heads(model, attention_in, layers.kinds)
    report.print_heads(heads, ENGINE_PROMPT_CAP, LONG_ROW)
    mask = attend_by_hand(model, attention_in, prompt)
    report.print_mask(mask, ATTENTION_LAYER, SHOWN_HEAD)
    report.print_rope(move_pair(model, attention_in, prompt, positions))
    report.print_head(score_tokens(loaded, table, steps, reference), LONG_ROW)


if __name__ == "__main__":
    main()
