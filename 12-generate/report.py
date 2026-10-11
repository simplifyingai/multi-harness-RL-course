"""How the chapter's scripts print: distributions, steps, sums and timings.

Nothing here measures anything. It only turns the scripts' results into lines
a person can read on screen (working out a percentage or a ratio at most), so
each script can stay about generating.
"""

from __future__ import annotations

import math
import statistics
import textwrap
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import measures
from lfm_files import (
    BF16,
    FINISH_STOP,
    MILLISECONDS,
    OUTPUT_CAP,
    gigabytes,
    short_path,
)

if TYPE_CHECKING:  # for type hints only: these modules import this one
    from sample_lab import AtTemperature, Candidate, Draw, Reply
    from timing import CacheSize, LengthResult

INDENT = "  "
WRAP_WIDTH = 88
PERCENT = 100
TOKEN_WIDTH = 18  # characters of a token's text shown in a table
SMALLEST_SHOWN = 0.0001  # probabilities below this (0.01%) are shown as "<0.01%"
FINEST_GRID = 2**measures.FINEST_GRID_POWER  # the finest grid looked for
# Why bfloat16 log-probabilities land on a grid: its spacing grows with the size
# of the number (chapter 10 showed 1/16 between 8 and 16, 1/8 between 16 and 32).
BF16_SPACING_NOTE = (
    "bfloat16 holds only multiples of 1/16 from 8 to 16, of 1/8 from 16 to 32 and "
    "of 1/4 from 32 to 64"
)
SHOWN_FIRST = 12  # steps listed at the start of a reply
SHOWN_LAST = 4  # and at its end
KILOBYTE = 1_000
MEGABYTE = 1_000_000
# The model's own suggested settings, as generation_config.json names them.
SUGGESTED_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")


class _Sections:
    """Numbers the sections of one script's output: 1, 2, 3..."""

    count = 0


def print_title(script: str, title: str) -> None:
    """Print the script's first line, and start counting sections again.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    _Sections.count = 0
    print(f"{script} · {title}")


def heading(text: str) -> None:
    """Print the next numbered section's heading after a blank line.

    Args:
        text: What the section shows.
    """
    _Sections.count += 1
    print(f"\n{_Sections.count}. {text}")


def wrapped(text: str, indent: str = INDENT) -> str:
    """Wrap a long line, indenting its continuation lines a little more.

    Args:
        text: One line of text.
        indent: The indent of the first line.

    Returns:
        The text as one or more lines.
    """
    return textwrap.fill(
        text,
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + INDENT,
        break_on_hyphens=False,
        break_long_words=False,  # a path stays whole, even past the margin
    )


def percent(probability: float) -> str:
    """Write a probability as a percentage.

    Args:
        probability: A number from 0 to 1.

    Returns:
        Such as ``"88.96%"``, or ``"<0.01%"`` for very small ones.
    """
    if probability < SMALLEST_SHOWN:
        return f"<{PERCENT * SMALLEST_SHOWN:g}%"
    return f"{PERCENT * probability:.2f}%"


def token_text(text: str) -> str:
    """Show a token's text in quotes, cut to fit a table column.

    Args:
        text: The token's text.

    Returns:
        Its ``repr``, at most ``TOKEN_WIDTH`` characters.
    """
    shown = repr(text)
    return shown if len(shown) <= TOKEN_WIDTH else shown[: TOKEN_WIDTH - 3] + "..."


def fraction(power: int | None) -> str:
    """Write the grid step 1 / 2**power in words.

    Args:
        power: The power of two, or None for no grid.

    Returns:
        Such as ``"1/8"`` or ``"1"``.
    """
    if power is None:
        return "no grid"
    return "1" if power == 0 else f"1/{2**power:,}"


# ------------------------------------------------------------- sample_lab.py


def describe_draw(draw: Draw, seed: int) -> str:
    """Say in words how tokens are drawn.

    Args:
        draw: How tokens are drawn.
        seed: The random seed.

    Returns:
        One line.
    """
    if draw.greedy:
        return (
            "greedy: always the likeliest token; log-probabilities from the "
            "untempered distribution, as the project's engine reports them"
        )
    cuts = []
    if draw.top_k:
        cuts.append(f"top-k {draw.top_k}: only the {draw.top_k} likeliest tokens")
    if draw.top_p < 1:
        cuts.append(
            f"top-p {draw.top_p:g}: only the likeliest tokens holding "
            f"{PERCENT * draw.top_p:g}% of the probability"
        )
    if not cuts:
        cuts.append("the full distribution (no top-k, no top-p)")
    else:
        cuts[-1] += ", renormalized"
    return f"temperature {draw.temperature:g}, {'; '.join(cuts)}; seed {seed}"


def print_setup(
    folder: Path,
    which: str,
    question: str,
    prompt_tokens: int,
    prompt_end: str,
    draw: Draw,
    seed: int,
    cap: int,
) -> None:
    """Say which model answers what, and how its tokens are drawn.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        question: What is asked.
        prompt_tokens: How many tokens the prompt is.
        prompt_end: The prompt's last token.
        draw: How tokens are drawn.
        seed: The random seed.
        cap: The length cap.
    """
    print(f"model     {short_path(folder)}")
    print(f"          {which}")
    print(f"question  {question}")
    print(f"prompt    {prompt_tokens} tokens, ending with {prompt_end}")
    print(wrapped(f"drawing   {describe_draw(draw, seed)}", indent=""))
    if draw.dtype == BF16:
        scores = (
            "turned into log-probabilities in bfloat16 (--dtype bf16); the "
            "project's engine casts them to float32 first"
        )
    else:
        scores = (
            "turned into log-probabilities in float32, as the project's engine "
            "does (its _to_float32)"
        )
    print(wrapped(f"scores    {scores}", indent=""))
    print(f"cap       {cap:,} tokens", end="")
    print(": the project's output cap" if cap == OUTPUT_CAP else "")


def print_suggested(config: dict[str, Any]) -> None:
    """Show the sampling settings the model's makers suggest, and that they go unused.

    Args:
        config: The build's generation_config.json (empty if missing).
    """
    found = [f"{key} {config[key]}" for key in SUGGESTED_KEYS if key in config]
    if not found:
        return
    print(
        wrapped(
            f"the model's suggested settings (generation_config.json): "
            f"{', '.join(found)}; the project's engine uses none of them",
            indent="",
        )
    )


def print_first_step(
    top: Sequence[Candidate],
    spread: float,
    vocabulary: int,
    decode: Callable[[list[int]], str],
) -> None:
    """List step 1's likeliest tokens, the distribution the first token is drawn from.

    Args:
        top: The likeliest tokens, likeliest first.
        spread: That distribution's entropy, in nats.
        vocabulary: How many tokens the model scores.
        decode: Turns token ids into text.
    """
    heading(
        f"step 1: the {len(top)} likeliest next tokens, of {vocabulary:,} "
        "(the distribution the first token is drawn from)"
    )
    print(f"{INDENT}rank      id  token               probability  log-probability")
    for rank, candidate in enumerate(top, start=1):
        text = token_text(decode([candidate.token]))
        print(
            f"{INDENT}{rank:>4}  {candidate.token:>6}  {text:<{TOKEN_WIDTH}}  "
            f"{percent(math.exp(candidate.logprob)):>11}  {candidate.logprob:>15.4f}"
        )
    rest = 1 - math.fsum(math.exp(candidate.logprob) for candidate in top)
    print(
        f"{INDENT}the other {vocabulary - len(top):,} tokens: "
        f"{percent(max(rest, 0.0))} together"
    )
    print(
        wrapped(
            f"entropy {spread:.3f} nats (0 if one token were certain; "
            f"{math.log(vocabulary):.2f}, the log of {vocabulary:,}, if all were "
            "equally likely)"
        )
    )


def print_temperatures(
    rows: Sequence[AtTemperature],
    named: Sequence[int],
    decode: Callable[[list[int]], str],
) -> None:
    """Show step 1's scores turned into probabilities at several temperatures.

    Args:
        rows: One per temperature.
        named: The tokens followed across temperatures.
        decode: Turns token ids into text.
    """
    heading(
        f"the same step at {len(rows)} temperatures: the scores divided by T, then "
        "the softmax (float32)"
    )
    names = "".join(f"{token_text(decode([token])):>10}" for token in named)
    print(f"{INDENT}temperature{names}  tokens holding 99%    entropy")
    for row in rows:
        shares = "".join(f"{percent(share):>10}" for share in row.named)
        print(
            f"{INDENT}{row.temperature:>11.1f}{shares}  {row.holding:>18,}  "
            f"{row.entropy:>6.3f} nats"
        )
    print(
        wrapped(
            "below 1 the likeliest token takes more of the probability (sharper); "
            "above 1 the rest take more (flatter)"
        )
    )


def step_columns(draw: Draw, show_raw: bool) -> list[str]:
    """Pick the extra columns a run's step table shows.

    Args:
        draw: How tokens were drawn.
        show_raw: Whether --raw asked for the raw log-probability.

    Returns:
        Column names, from "uncut", "raw" and "float32".
    """
    columns = []
    if draw.cuts():
        columns.append("uncut")
    if show_raw:
        columns.append("raw")
    if draw.dtype == BF16:
        columns.append("float32")
    return columns


def print_steps(reply: Reply, texts: list[str], draw: Draw, show_raw: bool) -> None:
    """List the drawn tokens with their log-probabilities and the running sum.

    Args:
        reply: The reply.
        texts: Each step's piece of text.
        draw: How tokens were drawn.
        show_raw: Whether to show the raw log-probability too.
    """
    heading("the reply, drawn token by token")
    columns = step_columns(draw, show_raw)
    extra = "".join(f"{name:>12}" for name in columns)
    print(
        f"{INDENT}step      id  token               probability     log-prob  "
        f"running sum  entropy{extra}"
    )
    sums = measures.running_sums([step.logprob for step in reply.steps])
    shown = list(range(len(reply.steps)))
    if len(shown) > SHOWN_FIRST + SHOWN_LAST + 1:
        shown = shown[:SHOWN_FIRST] + [-1] + shown[-SHOWN_LAST:]
    for index in shown:
        if index < 0:
            hidden = len(reply.steps) - SHOWN_FIRST - SHOWN_LAST
            print(f"{INDENT}  ... {hidden} steps not shown (all are in the saved file)")
            continue
        step = reply.steps[index]
        values = "".join(f"{getattr(step, name):>12.6f}" for name in columns)
        print(
            f"{INDENT}{index + 1:>4}  {step.token:>6}  "
            f"{token_text(texts[index]):<{TOKEN_WIDTH}}  "
            f"{percent(math.exp(step.logprob)):>11}  {step.logprob:>11.6f}  "
            f"{sums[index]:>11.4f}  {step.entropy:>7.3f}{values}"
        )
    if any(f"{reply.steps[i].logprob:.6f}" == "-0.000000" for i in shown if i >= 0):
        print(wrapped("-0.000000: a log-probability just below 0, not exactly 0"))
    descriptions = {
        "uncut": "uncut: the same token's log-probability before the cut",
        "raw": "raw: under the model's own scores, no temperature, nothing cut",
        "float32": "float32: from the same scores in float32",
    }
    for name in columns:
        print(wrapped(descriptions[name]))


def print_reply_text(text: str) -> None:
    """Print the reply's text in full.

    Args:
        text: The reply, decoded with its special tokens.
    """
    print("\nthe reply's text:\n")
    print(text)


def print_ending(reply: Reply, texts: list[str], eos_ids: Collection[int]) -> None:
    """Say how the reply ended, and add up its log-probabilities.

    Args:
        reply: The reply.
        texts: Each step's piece of text.
        eos_ids: The end-of-turn token ids.
    """
    heading("how the reply ended")
    end = measures.ending(
        [step.logprob for step in reply.steps], [step.entropy for step in reply.steps]
    )
    last = reply.steps[-1]
    if reply.finish_reason == FINISH_STOP:
        print(
            wrapped(
                f"{end.tokens} tokens; the last, {texts[-1]} (id {last.token}), is "
                "the end-of-turn token: finish_reason 'stop'"
            )
        )
    else:
        names = ", ".join(str(token) for token in sorted(eos_ids))
        print(
            wrapped(
                f"{end.tokens} tokens, the cap, and the last is not the end-of-turn "
                f"token (id {names}): finish_reason 'length', the reply cut off "
                "mid-thought"
            )
        )
    print(
        wrapped(
            f"the reply's log-probability, the sum of its {end.tokens} "
            f"log-probabilities: {end.total:.4f}"
        )
    )
    print(
        wrapped(
            f"the chance of drawing exactly this reply: e^{end.total:.4f}, about "
            f"{measures.one_in(end.total)}",
            indent=INDENT * 2,
        )
    )
    print(
        wrapped(
            f"entropy per step: {end.mean_entropy:.3f} nats on average; "
            f"{end.nearly_certain} of {end.tokens} steps below "
            f"{measures.NEARLY_CERTAIN_NATS} nats (one token held nearly all the "
            "probability)"
        )
    )


def print_greedy_again(first: Reply, second: Reply) -> None:
    """Compare two greedy runs of the same prompt.

    Args:
        first: The first run.
        second: The second run, from scratch.
    """
    heading("greedy again: a second run from scratch")
    same = first.tokens() == second.tokens()
    total = math.fsum(step.logprob for step in second.steps)
    print(
        wrapped(
            f"{len(second.steps)} tokens, identical to the first run's "
            f"{len(first.steps)}: {'yes' if same else 'no'}; its log-probability "
            f"{total:.4f}"
        )
    )


def print_cut(reply: Reply, draw: Draw) -> None:
    """Show what top-k or top-p changed, at the same steps.

    Args:
        reply: The reply.
        draw: How tokens were drawn.
    """
    steps = reply.steps
    result = measures.cut(
        [step.logprob for step in steps],
        [step.uncut for step in steps],
        [step.entropy for step in steps],
        [step.uncut_entropy for step in steps],
    )
    name = f"top-k {draw.top_k}" if draw.top_k else f"top-p {draw.top_p:g}"
    heading(f"what {name} changed, at each of the {result.steps} steps")
    lower = ""
    if result.lower:
        lower = (
            f", lower at {result.lower} by float32 rounding alone (by at most "
            f"{result.largest_drop:.6f})"
        )
    print(
        wrapped(
            f"the drawn token's probability, as drawn vs uncut: higher at "
            f"{result.higher} steps, the same at {result.equal}{lower}"
        )
    )
    print(
        wrapped(
            f"the kept tokens held {percent(result.kept_mean)} of the probability on "
            f"average (the lowest {percent(result.kept_lowest)}); the cut divides "
            "every kept probability by that share"
        )
    )
    print(
        wrapped(
            f"entropy: {result.entropy_uncut:.3f} nats on average uncut, "
            f"{result.entropy_cut:.3f} as drawn; lower at {result.lower_entropy} of "
            f"{result.steps} steps"
        )
    )
    print(
        wrapped(
            f"the reply's log-probability: {result.total_cut:.4f} as drawn, "
            f"{result.total_uncut:.4f} under the model's full distribution"
        )
    )


def print_raw(reply: Reply, draw: Draw) -> None:
    """Show the drawn tokens' processed and raw log-probabilities side by side.

    Args:
        reply: The reply.
        draw: How tokens were drawn.
    """
    steps = reply.steps
    result = measures.raw_side([s.logprob for s in steps], [s.raw for s in steps])
    heading("processed and raw log-probabilities (--raw)")
    print(
        wrapped(
            f"processed: under the distribution each token was drawn from "
            f"(temperature {draw.temperature:g}); raw: under the model's own scores "
            "(log_softmax of the scores, temperature 1)"
        )
    )
    print(
        wrapped(
            f"they differ at {result.differ} of {result.steps} steps; processed is "
            f"higher at {result.processed_higher}, lower at "
            f"{result.differ - result.processed_higher}"
        )
    )
    print(
        wrapped(
            f"the reply's log-probability: processed {result.total_processed:.4f}, "
            f"raw {result.total_raw:.4f}"
        )
    )


def print_grid(reply: Reply) -> None:
    """Show where bfloat16 log-probabilities land, beside float32 ones.

    Args:
        reply: The reply, drawn with --dtype bf16.
    """
    steps = reply.steps
    result = measures.grid(
        [s.logprob for s in steps], [s.float32 for s in steps], [s.score for s in steps]
    )
    heading("log-probabilities in bfloat16 (--dtype bf16), beside float32")
    print(
        wrapped(
            f"every one of the {result.steps} drawn tokens' bfloat16 "
            f"log-probabilities is a whole multiple of {fraction(result.power)}; "
            f"they take {result.distinct} different values"
        )
    )
    if result.float32_power is None:
        shape = f"on no grid of 1/{FINEST_GRID:,} or coarser"
    else:
        shape = f"multiples of {fraction(result.float32_power)}"
    print(
        wrapped(
            f"the same tokens in float32: {result.distinct_float32} different "
            f"values, {shape}"
        )
    )
    print(
        wrapped(
            f"the drawn tokens' scores ran from {result.score_low:.3f} to "
            f"{result.score_high:.3f}; {BF16_SPACING_NOTE}"
        )
    )
    if result.zeros:
        print(
            wrapped(
                f"{result.zeros} of {result.steps} are exactly 0 in bfloat16 "
                "(probability 100%); in float32 those are "
                f"{result.zero_float32_low:.5f} to {result.zero_float32_high:.5f}"
            )
        )
    print(
        wrapped(
            f"the reply's log-probability: {result.total:.4f} in bfloat16, "
            f"{result.total_float32:.4f} in float32"
        )
    )


def print_saved(path: Path) -> None:
    """Say where every step was saved.

    Args:
        path: The file.
    """
    print(f"\nsaved every step in {short_path(path)}")


# ------------------------------------------------------------------ timing.py


def chunk_text(chunks: Sequence[int]) -> str:
    """Write a prompt's reading chunks compactly.

    Args:
        chunks: Tokens per model call, in order.

    Returns:
        Such as ``"1,024 x 7 + 831 + 1"``.
    """
    parts: list[str] = []
    index = 0
    while index < len(chunks):
        size, run = chunks[index], 1
        while index + run < len(chunks) and chunks[index + run] == size:
            run += 1
        parts.append(f"{size:,} x {run}" if run > 1 else f"{size:,}")
        index += run
    return " + ".join(parts)


def megabytes(size: int) -> str:
    """Write a size in megabytes (10**6 bytes).

    Args:
        size: Bytes.

    Returns:
        Such as ``"131.1 MB"``.
    """
    return f"{size / MEGABYTE:.1f} MB"


def print_timing_setup(
    folder: Path, which: str, weight_bytes: int, warm_seconds: float
) -> None:
    """Say which model is timed, and how long its warm-up took.

    Args:
        folder: The build's folder.
        which: Which build it is.
        weight_bytes: Its weights' size in memory.
        warm_seconds: The untimed warm-up's duration.
    """
    print(f"model     {short_path(folder)}")
    print(f"          {which}; {gigabytes(weight_bytes)} of weights in memory")
    print(f"warm-up   {warm_seconds:.2f} s for an untimed short reply (GPU set-up)")


def print_prompts(lengths: Sequence[int], question: str) -> None:
    """Say what the timed prompts are.

    Args:
        lengths: The prompt lengths, in tokens.
        question: The question at the end of each.
    """
    joined = ", ".join(f"{length:,}" for length in lengths[:-1])
    joined = f"{joined} and {lengths[-1]:,}" if joined else f"{lengths[-1]:,}"
    print(
        wrapped(
            f"prompts   a made-up customers table, cut so that the whole chat is "
            f"exactly {joined} tokens, then the question: {question!r}; every reply "
            "is greedy",
            indent="",
        )
    )


def print_prefill(results: Sequence[LengthResult], step: int) -> None:
    """Show time to first token for each prompt length.

    Args:
        results: One per prompt length.
        step: The prefill chunk size.
    """
    heading(f"prefill: reading the prompt, {step:,} tokens at a time (PREFILL_STEP)")
    print(
        f"{INDENT}prompt  read as                    first token after  "
        f"prompt tokens/s  peak memory"
    )
    for result in results:
        prefill = result.prefill
        speed = prefill.length / prefill.first_seconds
        print(
            f"{INDENT}{prefill.length:>6,}  {chunk_text(prefill.chunks):<25}  "
            f"{prefill.first_seconds:>15.2f} s  {speed:>15,.0f}  "
            f"{gigabytes(prefill.peak_bytes):>11}"
        )
    print(
        wrapped(
            "read as: the tokens each model call read; the last prompt token is read "
            "by the step that predicts the first reply token. Each prompt is read "
            "twice, and the faster run counts: other work on the GPU slows a run. "
            "Peak memory includes the weights"
        )
    )


def print_other_step(result: LengthResult, step: int) -> None:
    """Show the longest prompt read in mlx-lm's default chunks, for comparison.

    Args:
        result: The longest prompt's results.
        step: mlx-lm's default chunk size.
    """
    other = result.other_step
    if other is None:
        return
    print(
        wrapped(
            f"the same {other.length:,} tokens in mlx-lm's default {step:,}-token "
            f"chunks ({chunk_text(other.chunks)}): first token after "
            f"{other.first_seconds:.2f} s, peak memory {gigabytes(other.peak_bytes)}"
        )
    )


def cache_text(cache: CacheSize) -> str:
    """Describe what a cache holds.

    Args:
        cache: The cache's measured size.

    Returns:
        One short line.
    """
    return (
        f"{cache.positions:,} positions: {megabytes(cache.attention_bytes)} of keys "
        f"and values, {cache.conv_bytes / KILOBYTE:.0f} KB of conv state"
    )


def print_cached(results: Sequence[LengthResult]) -> None:
    """Show decode speed with the cache, and what the cache holds.

    Args:
        results: One per prompt length.
    """
    count = len(results[0].cached_seconds) + 1
    heading(f"decode with the cache: one token at a time, {count} tokens per prompt")
    print(f"{INDENT}prompt  ms per token  tokens/s  the cache afterwards")
    for result in results:
        per_token = statistics.median(result.cached_seconds)
        print(
            f"{INDENT}{result.prefill.length:>6,}  {MILLISECONDS * per_token:>12.1f}  "
            f"{1 / per_token:>8.1f}  {cache_text(result.cache)}"
        )
    same = all(result.repeat_same for result in results)
    print(
        wrapped(
            f"both runs of each prompt decoded the same {count} tokens: "
            f"{'yes' if same else 'no'}"
        )
    )
    cache = results[0].cache
    print(
        wrapped(
            f"ms per token: the median of tokens 2 to {count} (the first is the "
            "prefill's), in the steadier of the two runs; keys and "
            f"values: {cache.attention_layers} attention layers x "
            f"{cache.bytes_per_position // cache.attention_layers:,} bytes = "
            f"{cache.bytes_per_position:,} bytes per position; conv state: "
            f"{cache.conv_layers} convolution layers, the same size at any length"
        )
    )


def print_uncached(results: Sequence[LengthResult]) -> None:
    """Show decode speed with the cache thrown away before every token.

    Args:
        results: One per prompt length.
    """
    count = len(results[0].uncached_seconds)
    heading(
        "decode without the cache: every token reads the prompt and the reply so "
        "far again, from scratch"
    )
    print(f"{INDENT}prompt  ms per token  tokens/s  the same tokens as with the cache")
    for result in results:
        per_token = statistics.median(result.uncached_seconds)
        same = sum(
            1
            for one, other in zip(
                result.uncached_tokens, result.cached_tokens[:count], strict=True
            )
            if one == other
        )
        print(
            f"{INDENT}{result.prefill.length:>6,}  {MILLISECONDS * per_token:>12,.0f}  "
            f"{1 / per_token:>8.2f}  {same} of {count}"
        )


def print_timing_summary(results: Sequence[LengthResult]) -> None:
    """Sum up how prompt length changes each time.

    Args:
        results: One per prompt length, shortest first.
    """
    first, last = results[0], results[-1]
    heading("what changes with the prompt's length")
    times = last.prefill.length / first.prefill.length
    ttft = last.prefill.first_seconds / first.prefill.first_seconds
    print(
        wrapped(
            f"time to first token: {first.prefill.first_seconds:.2f} s at "
            f"{first.prefill.length:,} tokens, {last.prefill.first_seconds:.2f} s at "
            f"{last.prefill.length:,} ({ttft:.1f} times as long for {times:.0f} times "
            "the tokens)"
        )
    )
    with_low = statistics.median(first.cached_seconds)
    with_high = statistics.median(last.cached_seconds)
    print(
        wrapped(
            f"decode with the cache: {MILLISECONDS * with_low:.1f} to "
            f"{MILLISECONDS * with_high:.1f} ms per token ({with_high / with_low:.2f} "
            "times as long)"
        )
    )
    without_low = statistics.median(first.uncached_seconds)
    without_high = statistics.median(last.uncached_seconds)
    print(
        wrapped(
            f"decode without it: {without_low:.2f} to {without_high:.2f} s per token "
            f"({without_high / without_low:.1f} times as long); at "
            f"{last.prefill.length:,} tokens, {without_high / with_high:,.0f} times "
            "slower than with the cache"
        )
    )
    print(
        wrapped(
            "every number in this output is a time: it varies from run to run and "
            "from machine to machine"
        )
    )
