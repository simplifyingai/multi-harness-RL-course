# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Time the two halves of a reply: reading the prompt, and writing the tokens.

Prefill reads the prompt, many tokens per model call: 1,024 at a time, as
the project's engine asks mlx-lm (its PREFILL_STEP). Decode writes the reply,
one token per call. While it reads, the model fills a cache: the keys and
values of its 8 attention layers and the small state of its 22 convolution
layers, so each new token reads the cache instead of the whole text again.
This script times the first token after prompts of 500, 2,000 and 8,000
tokens, then decode with that cache and without it.

Run it with ``uv run timing.py`` (about two minutes); ``--fast`` times the
two shorter prompts with fewer tokens.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import report
from lfm import chat_prompt, load_folder, render_chat
from lfm_files import PREFILL_STEP, pick_model
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

SCRIPT = "timing.py"
TITLE = "reading the prompt, writing the reply, and the cache"
QUESTION = "How many customers live in Paris? Answer with one SQL query."
PROMPT_LENGTHS = (500, 2_000, 8_000)
FAST_LENGTHS = (500, 2_000)
CACHED_TOKENS = 64  # decode steps timed with the cache
FAST_CACHED_TOKENS = 32
UNCACHED_TOKENS = 4  # decode steps timed without it (each one is a whole prefill)
FAST_UNCACHED_TOKENS = 2
REPEATS = 2  # each prefill and cached decode runs twice; the better run counts
MLX_PREFILL_STEP = 2_048  # generate_step's own default, for comparison
WARM_UP_QUESTION = "Hi"
WARM_UP_TOKENS = 2
TABLE_ROWS = 2_000  # far more rows than the longest prompt needs
CITIES = ("Paris", "Lyon", "Berlin", "Madrid", "Rome", "Lisbon", "Vienna")
TABLE_MARK = "<<TABLE>>"  # where the table goes in the rendered chat


@dataclass(frozen=True)
class Prefill:
    """One prompt read, up to its first reply token.

    Attributes:
        length: The prompt's length in tokens.
        chunks: The tokens each model call read, in order (the last call is
            the step that predicts the first reply token).
        first_seconds: From the start to the first reply token.
        peak_bytes: MLX's peak memory over the reading.
    """

    length: int
    chunks: list[int]
    first_seconds: float
    peak_bytes: int


@dataclass(frozen=True)
class CacheSize:
    """What a prompt cache holds.

    Attributes:
        positions: How many tokens it covers.
        attention_layers: How many layers keep keys and values.
        attention_bytes: Their keys and values, for those positions.
        bytes_per_position: Keys and values per token, over all those layers.
        conv_layers: How many convolution layers keep a state.
        conv_bytes: Their state, the same at any length.
    """

    positions: int
    attention_layers: int
    attention_bytes: int
    bytes_per_position: int
    conv_layers: int
    conv_bytes: int


@dataclass(frozen=True)
class Decoded:
    """One run: a prompt read, then tokens decoded with the cache.

    Attributes:
        prefill: The prompt's reading, up to the first reply token.
        seconds: Each decode step's time, after the first token.
        tokens: The tokens decoded, the first included.
        cache: What the cache held at the end.
    """

    prefill: Prefill
    seconds: list[float]
    tokens: list[int]
    cache: CacheSize


@dataclass(frozen=True)
class LengthResult:
    """Everything measured for one prompt length.

    Attributes:
        prefill: The prompt read in the project's chunks.
        other_step: The same prompt read in mlx-lm's default chunks (longest
            prompt only).
        cached_seconds: Each decode step's time with the cache, after the first.
        cached_tokens: The tokens decoded with the cache, the first included.
        cache: What the cache held afterwards.
        repeat_same: Whether the repeated run decoded the same tokens.
        uncached_seconds: Each token's time without a cache.
        uncached_tokens: The tokens decoded without a cache.
    """

    prefill: Prefill
    other_step: Prefill | None
    cached_seconds: list[float]
    cached_tokens: list[int]
    cache: CacheSize
    repeat_same: bool
    uncached_seconds: list[float]
    uncached_tokens: list[int]


def table_text(rows: int) -> str:
    """Write a made-up customers table, the same every time.

    Args:
        rows: How many customers.

    Returns:
        The table, one customer per line.
    """
    lines = ["customer_id | name | city"]
    for number in range(1, rows + 1):
        lines.append(f"{number} | Customer {number} | {CITIES[number % len(CITIES)]}")
    return "\n".join(lines)


def prompt_of_length(tokenizer: TokenizerWrapper, length: int) -> list[int]:
    """Build a chat prompt of exactly ``length`` tokens: a table, then the question.

    The chat is rendered with the table's place marked; the table's tokens are
    cut so that the whole prompt has the length asked for.

    Args:
        tokenizer: The model's tokenizer.
        length: How many tokens the prompt must have.

    Returns:
        The prompt's token ids, ending with the generation prompt.

    Raises:
        ValueError: If the length is too short for the chat around the table,
            or longer than the table can fill.
    """
    text = render_chat(
        tokenizer, f"Here is the customers table:\n{TABLE_MARK}\n{QUESTION}"
    )
    before, after = text.split(TABLE_MARK)
    head = list(tokenizer.encode(before, add_special_tokens=False))
    tail = list(tokenizer.encode(after, add_special_tokens=False))
    table = list(tokenizer.encode(table_text(TABLE_ROWS), add_special_tokens=False))
    room = length - len(head) - len(tail)
    if not 0 < room <= len(table):
        raise ValueError(f"cannot build a {length:,}-token prompt from this table")
    return head + table[:room] + tail


def cache_size(cache: list[Any]) -> CacheSize:
    """Measure what a prompt cache holds, layer kind by layer kind.

    Args:
        cache: The cache, one entry per layer.

    Returns:
        Its size.

    Raises:
        ValueError: If no layer keeps keys and values, or none is filled yet.
    """
    attention = [layer for layer in cache if isinstance(layer, KVCache)]
    conv = [layer for layer in cache if isinstance(layer, ArraysCache)]
    if not attention or not attention[0].offset:
        raise ValueError("the cache holds no keys and values yet")
    positions = attention[0].offset
    used = 0
    for layer in attention:
        # The cache grows in blocks of 256 positions; count the ones in use.
        used += layer.keys[..., :positions, :].nbytes
        used += layer.values[..., :positions, :].nbytes
    return CacheSize(
        positions=positions,
        attention_layers=len(attention),
        attention_bytes=used,
        bytes_per_position=used // positions,
        conv_layers=len(conv),
        conv_bytes=sum(layer.nbytes for layer in conv),
    )


def read_and_decode(
    model: nn.Module, prompt: list[int], step: int, count: int
) -> Decoded:
    """Read a prompt and decode greedily with the cache, timing every token.

    Args:
        model: The model.
        prompt: The prompt's token ids.
        step: How many prompt tokens each model call reads.
        count: How many tokens to decode (the end of turn does not stop it).

    Returns:
        The run: its prefill, each later token's time, the tokens, the cache.
    """
    cache = make_prompt_cache(model)
    progress: list[int] = []
    seconds: list[float] = []
    tokens: list[int] = []
    first_seconds, peak = 0.0, 0
    mx.reset_peak_memory()
    start = last = time.perf_counter()
    for token, _ in generate_step(
        mx.array(prompt),
        model,
        max_tokens=count,
        prompt_cache=cache,
        prefill_step_size=step,
        # mlx-lm reports how many prompt tokens are read after each call.
        prompt_progress_callback=lambda done, total: progress.append(done),
    ):
        now = time.perf_counter()
        if not tokens:
            first_seconds, peak = now - start, mx.get_peak_memory()
        else:
            seconds.append(now - last)
        last = now
        tokens.append(token)
    chunks = [after - before for before, after in pairwise(progress)]
    prefill = Prefill(len(prompt), chunks, first_seconds, peak)
    return Decoded(prefill, seconds, tokens, cache_size(cache))


def next_token_from_scratch(model: nn.Module, ids: list[int], step: int) -> int:
    """Predict the next token with an empty cache: read everything again.

    This is what generate_step does before its first token, written out:
    every token but the last, ``step`` at a time, then the last one.

    Args:
        model: The model.
        ids: The prompt and the reply so far.
        step: How many tokens each model call reads.

    Returns:
        The likeliest next token.
    """
    cache = make_prompt_cache(model)  # empty: nothing kept from earlier tokens
    for begin in range(0, len(ids) - 1, step):
        chunk = ids[begin : min(begin + step, len(ids) - 1)]
        model(mx.array(chunk)[None], cache=cache)
        mx.eval([layer.state for layer in cache])  # MLX is lazy: compute it now
    scores = model(mx.array(ids[-1:])[None], cache=cache)
    return int(mx.argmax(scores[0, -1]).item())


def decode_without_cache(
    model: nn.Module, prompt: list[int], step: int, count: int
) -> tuple[list[float], list[int]]:
    """Decode greedily, reading the prompt and the reply again for every token.

    Args:
        model: The model.
        prompt: The prompt's token ids.
        step: How many tokens each model call reads.
        count: How many tokens to decode.

    Returns:
        Each token's time, and the tokens.
    """
    seconds: list[float] = []
    tokens: list[int] = []
    for _ in range(count):
        start = time.perf_counter()
        tokens.append(next_token_from_scratch(model, prompt + tokens, step))
        seconds.append(time.perf_counter() - start)
    return seconds, tokens


def warm_up(model: nn.Module, tokenizer: TokenizerWrapper) -> float:
    """Run a short untimed reply: the first pass after loading is slower.

    Args:
        model: The model.
        tokenizer: Its tokenizer.

    Returns:
        Its duration in seconds.
    """
    start = time.perf_counter()
    ids = chat_prompt(tokenizer, WARM_UP_QUESTION)
    for _ in generate_step(mx.array(ids), model, max_tokens=WARM_UP_TOKENS):
        pass
    return time.perf_counter() - start


def best_of(
    model: nn.Module, prompt: list[int], step: int, count: int
) -> tuple[Decoded, bool]:
    """Read and decode the same prompt twice, keeping each measure's better run.

    Other programs share the GPU (on this laptop, video renders did), and a
    moment of their work makes one run slower; the faster of two is closer
    to what the model itself costs.

    Args:
        model: The model.
        prompt: The prompt's token ids.
        step: How many prompt tokens each model call reads.
        count: How many tokens to decode.

    Returns:
        The faster prefill with the decode times of lower median (and that
        run's tokens and cache), and whether both runs gave the same tokens.
    """
    runs = [read_and_decode(model, prompt, step, count) for _ in range(REPEATS)]
    fastest = min(runs, key=lambda run: run.prefill.first_seconds)
    steady = min(runs, key=lambda run: statistics.median(run.seconds or [0.0]))
    same = all(run.tokens == runs[0].tokens for run in runs)
    return Decoded(fastest.prefill, steady.seconds, steady.tokens, steady.cache), same


def measure(
    model: nn.Module, prompt: list[int], cached: int, uncached: int, longest: bool
) -> LengthResult:
    """Measure one prompt length: prefill, decode with and without the cache.

    Args:
        model: The model.
        prompt: The prompt's token ids.
        cached: Tokens to decode with the cache.
        uncached: Tokens to decode without it.
        longest: Whether this is the longest prompt (read in mlx-lm's default
            chunks too).

    Returns:
        The results.
    """
    best, same = best_of(model, prompt, PREFILL_STEP, cached)
    other = None
    if longest:
        other = best_of(model, prompt, MLX_PREFILL_STEP, 1)[0].prefill
    slow_seconds, slow_tokens = decode_without_cache(
        model, prompt, PREFILL_STEP, uncached
    )
    return LengthResult(
        prefill=best.prefill,
        other_step=other,
        cached_seconds=best.seconds,
        cached_tokens=best.tokens,
        cache=best.cache,
        repeat_same=same,
        uncached_seconds=slow_seconds,
        uncached_tokens=slow_tokens,
    )


def main() -> None:
    """Time prefill and decode at three prompt lengths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="time the two shorter prompts only, with fewer tokens (same model)",
    )
    args = parser.parse_args()
    lengths = FAST_LENGTHS if args.fast else PROMPT_LENGTHS
    cached = FAST_CACHED_TOKENS if args.fast else CACHED_TOKENS
    uncached = FAST_UNCACHED_TOKENS if args.fast else UNCACHED_TOKENS
    folder, which = pick_model(fast=False)
    loaded = load_folder(folder)

    report.print_title(SCRIPT, TITLE)
    warm = warm_up(loaded.model, loaded.tokenizer)
    report.print_timing_setup(folder, which, loaded.weight_bytes, warm)
    report.print_prompts(lengths, QUESTION)
    results = []
    for length in lengths:
        prompt = prompt_of_length(loaded.tokenizer, length)
        longest = length == lengths[-1]
        results.append(measure(loaded.model, prompt, cached, uncached, longest))
    report.print_prefill(results, PREFILL_STEP)
    report.print_other_step(results[-1], MLX_PREFILL_STEP)
    report.print_cached(results)
    report.print_uncached(results)
    report.print_timing_summary(results)


if __name__ == "__main__":
    main()
