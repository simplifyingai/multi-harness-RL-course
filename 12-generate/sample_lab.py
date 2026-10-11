# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""How the 4-bit LFM2.5-2.6B picks each token, with every number RL keeps.

Our own loop over mlx-lm's ``generate_step``, the function the project's
engine runs on (``mhrl/engine/policy.py``). At every step the model scores
all 128,000 tokens; the scores become log-probabilities (in float32, as the
engine does, or in bfloat16 with ``--dtype bf16``); temperature, top-k and
top-p turn them into the distribution the token is drawn from; one token is
drawn. The loop keeps that whole distribution, the drawn token and its
log-probability, until the model writes ``<|im_end|>`` (finish_reason
``stop``) or the length cap is reached (``length``).

Run it with ``uv run sample_lab.py``; ``--greedy``, ``--temperature``,
``--top-k``, ``--top-p``, ``--max-tokens``, ``--dtype`` and ``--raw`` change
the draw or what is shown. ``--fast`` uses Liquid's export instead of
chapter 11's build (the same bytes). Every step is saved under
``$COURSE_HOME/12-generate``.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlx.core as mx
import report
from lfm import LoadedModel, chat_prompt, load_folder
from lfm_files import (
    BF16,
    FINISH_LENGTH,
    FINISH_STOP,
    FP32,
    OUTPUT_CAP,
    PREFILL_STEP,
    chapter_home,
    pick_model,
    read_generation_config,
    read_vocabulary_size,
)
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import apply_top_k, apply_top_p

SCRIPT = "sample_lab.py"
TITLE = "how the model picks each token"
QUESTION = "How many customers live in Paris? Answer with one SQL query."
TRAIN_TEMPERATURE = 1.0  # the project draws training replies at temperature 1
UNCHANGED = 1.0  # dividing the scores by 1 changes nothing
SHOWN_TEMPERATURES = (0.5, 1.0, 1.5)
LIKELIEST_SHOWN = 10  # step 1's likeliest tokens, listed
NAMED_SHOWN = 3  # of those, the ones followed across temperatures
MOST_OF_IT = 0.99  # "how many tokens hold 99% of the probability"
NO_TOP_K = 0  # keep every token
NO_TOP_P = 1.0  # keep every token
SEED = 0
MIN_TOKENS = 1  # the smallest length cap generate_step accepts
RECORD_NAME = "sample_lab-{tag}.jsonl"


@dataclass(frozen=True)
class Draw:
    """How each token is drawn.

    Attributes:
        temperature: The scores are divided by it before the softmax.
        greedy: Always take the most likely token (temperature is ignored).
        top_k: Keep only this many of the likeliest tokens (0 keeps all).
        top_p: Keep the fewest likeliest tokens whose probabilities add up to
            at least this share (1 keeps all).
        dtype: ``"fp32"`` or ``"bf16"``: the precision in which the scores
            become log-probabilities.
    """

    temperature: float = TRAIN_TEMPERATURE
    greedy: bool = False
    top_k: int = NO_TOP_K
    top_p: float = NO_TOP_P
    dtype: str = FP32

    def __post_init__(self) -> None:
        """Refuse a precision this chapter does not know.

        Raises:
            ValueError: If ``dtype`` is neither ``"fp32"`` nor ``"bf16"``.
        """
        if self.dtype not in (FP32, BF16):
            raise ValueError(f"dtype must be {FP32!r} or {BF16!r}, not {self.dtype!r}")

    def cuts(self) -> bool:
        """Say whether top-k or top-p removes tokens before the draw.

        Returns:
            True when the draw is not from the full distribution.
        """
        return not self.greedy and (self.top_k != NO_TOP_K or self.top_p != NO_TOP_P)


@dataclass(frozen=True)
class Step:
    """One drawn token, and the numbers kept for it.

    Attributes:
        token: Its id.
        logprob: Its log-probability under the distribution it was drawn
            from: the number RL training needs.
        entropy: That distribution's entropy, in nats.
        uncut: Its log-probability at the same temperature before top-k and
            top-p removed any token (equal to ``logprob`` when nothing is cut).
        uncut_entropy: The entropy before the cut.
        raw: Its log-probability under the model's own distribution: no
            temperature, nothing cut.
        float32: ``logprob`` computed from the same scores in float32 (equal
            to it unless ``--dtype bf16``).
        score: Its score, the number the model gave it before any softmax.
    """

    token: int
    logprob: float
    entropy: float
    uncut: float
    uncut_entropy: float
    raw: float
    float32: float
    score: float


@dataclass(frozen=True)
class Reply:
    """A whole reply, step by step.

    Attributes:
        prompt_tokens: How many tokens the prompt is.
        steps: Every drawn token, the end of turn included if there is one.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        first: Step 1's distribution, the one the first token was drawn from.
        first_scores: Step 1's scores in float32, before any softmax.
    """

    prompt_tokens: int
    steps: list[Step]
    finish_reason: str
    first: mx.array
    first_scores: mx.array

    def tokens(self) -> list[int]:
        """List the drawn token ids.

        Returns:
            The ids, in order.
        """
        return [step.token for step in self.steps]


def drawn_from(logprobs: mx.array, draw: Draw) -> mx.array:
    """Turn the model's own log-probabilities into the distribution drawn from.

    Temperature first, then top-k, then top-p (Hugging Face's order); the
    tokens cut become minus infinity (probability 0) and the rest are
    renormalized, so their probabilities add up to 1 again.

    Args:
        logprobs: The model's own log-probabilities, ``log_softmax(scores)``.
        draw: How tokens are drawn.

    Returns:
        The log-probabilities of the distribution a token is drawn from.
    """
    if draw.greedy or (draw.temperature == UNCHANGED and not draw.cuts()):
        return logprobs  # nothing to change (greedy reports these, as the engine does)
    scores = logprobs / draw.temperature
    if draw.top_k != NO_TOP_K:
        scores = apply_top_k(scores, draw.top_k)
    if draw.top_p != NO_TOP_P:
        scores = apply_top_p(scores, draw.top_p)
    return scores - mx.logsumexp(scores, axis=-1, keepdims=True)


def entropy(logprobs: mx.array) -> mx.array:
    """Measure how spread out a distribution is: minus the sum of p x log p.

    Args:
        logprobs: Its log-probabilities (minus infinity for tokens cut).

    Returns:
        The entropy in nats; a token with probability 0 adds nothing.
    """
    logprobs = logprobs.astype(mx.float32)
    terms = mx.exp(logprobs) * logprobs
    return -mx.sum(mx.where(mx.isinf(logprobs), 0.0, terms), axis=-1)


def sample(loaded: LoadedModel, prompt: list[int], draw: Draw, cap: int) -> Reply:
    """Draw a reply token by token, keeping every number RL needs.

    Args:
        loaded: The model and its tokenizer.
        prompt: The prompt's token ids.
        draw: How tokens are drawn.
        cap: The most tokens the reply may have.

    Returns:
        The reply, step by step.

    Raises:
        ValueError: If the cap is below one token.
    """
    if cap < MIN_TOKENS:
        raise ValueError(f"the cap must be at least {MIN_TOKENS} token")
    seen: deque[mx.array] = deque()  # each step's scores, oldest first

    def keep_scores(tokens: mx.array, scores: mx.array) -> mx.array:
        """Put the step's scores aside, then cast them for the softmax.

        generate_step calls this once per step, in order, so ``seen`` pairs
        every yielded token with its own step's scores (oldest first).

        Args:
            tokens: The tokens so far, which this processor does not need.
            scores: The step's scores, in bfloat16 as the model computed them.

        Returns:
            The scores in float32 (the engine's _to_float32), or unchanged
            with --dtype bf16.
        """
        del tokens
        seen.append(scores)
        return scores.astype(mx.float32) if draw.dtype == FP32 else scores

    def pick(logprobs: mx.array) -> mx.array:
        """Draw the next token from the distribution ``drawn_from`` makes.

        Args:
            logprobs: The step's log-probabilities, from generate_step.

        Returns:
            The token: the likeliest one if greedy, else a random draw.
        """
        if draw.greedy:
            return mx.argmax(logprobs, axis=-1)
        return mx.random.categorical(drawn_from(logprobs, draw))

    steps: list[Step] = []
    finish, first, first_scores = FINISH_LENGTH, None, None
    uncut_draw = replace(draw, top_k=NO_TOP_K, top_p=NO_TOP_P)
    for token, logprobs in generate_step(
        mx.array(prompt),
        loaded.model,
        max_tokens=cap,
        sampler=pick,
        logits_processors=[keep_scores],
        prefill_step_size=PREFILL_STEP,
    ):
        scores = seen.popleft()[0].astype(mx.float32)
        # The same function on the same numbers: the distribution pick() drew from.
        dist = drawn_from(logprobs, draw)
        uncut = drawn_from(logprobs, uncut_draw) if draw.cuts() else dist
        wide = dist
        if draw.dtype == BF16:  # the same scores, made log-probabilities in float32
            wide = drawn_from(scores - mx.logsumexp(scores), draw)
        values = [dist[token], entropy(dist), uncut[token], entropy(uncut)]
        values += [logprobs[token], wide[token], scores[token]]
        numbers = mx.stack([value.astype(mx.float32) for value in values]).tolist()
        steps.append(Step(token, *numbers))
        if first is None:
            first, first_scores = dist, scores
        if token in loaded.tokenizer.eos_token_ids:  # <|im_end|>: the end of turn
            finish = FINISH_STOP
            break
    return Reply(len(prompt), steps, finish, first, first_scores)


@dataclass(frozen=True)
class Candidate:
    """One possible next token.

    Attributes:
        token: Its id.
        logprob: Its log-probability.
    """

    token: int
    logprob: float


@dataclass(frozen=True)
class AtTemperature:
    """Step 1's scores at one temperature.

    Attributes:
        temperature: What the scores were divided by.
        named: The probabilities of the tokens followed across temperatures.
        holding: How many of the likeliest tokens hold 99% of the probability.
        entropy: The distribution's entropy, in nats.
    """

    temperature: float
    named: list[float]
    holding: int
    entropy: float


def likeliest(logprobs: mx.array, count: int) -> list[Candidate]:
    """List the likeliest tokens of a distribution, likeliest first.

    Args:
        logprobs: The distribution's log-probabilities.
        count: How many to list.

    Returns:
        The tokens and their log-probabilities.
    """
    order = mx.argsort(-logprobs.astype(mx.float32))[:count]
    values = logprobs.astype(mx.float32)[order].tolist()
    return [Candidate(int(t), v) for t, v in zip(order.tolist(), values, strict=True)]


def at_temperatures(
    scores: mx.array, named: list[int], temperatures: tuple[float, ...]
) -> list[AtTemperature]:
    """Turn the same scores into probabilities at several temperatures.

    Args:
        scores: One step's scores, in float32.
        named: The tokens to follow.
        temperatures: What to divide the scores by.

    Returns:
        One row per temperature.
    """
    rows = []
    for temperature in temperatures:
        logprobs = scores / temperature
        logprobs = logprobs - mx.logsumexp(logprobs)
        probs = mx.sort(mx.exp(logprobs))[::-1]  # the likeliest first
        holding = int(mx.sum(mx.cumsum(probs) < MOST_OF_IT).item()) + 1
        shares = mx.exp(logprobs[mx.array(named)]).tolist()
        rows.append(
            AtTemperature(temperature, shares, holding, entropy(logprobs).item())
        )
    return rows


def record_tag(draw: Draw, seed: int, cap: int) -> str:
    """Name a run by how it drew, so different runs are saved side by side.

    Args:
        draw: How tokens were drawn.
        seed: The random seed.
        cap: The length cap.

    Returns:
        Such as ``"t1-seed0"`` or ``"greedy-max20"``.
    """
    parts = ["greedy"] if draw.greedy else [f"t{draw.temperature:g}"]
    if draw.top_k != NO_TOP_K:
        parts.append(f"k{draw.top_k}")
    if draw.top_p != NO_TOP_P:
        parts.append(f"p{draw.top_p:g}")
    if draw.dtype == BF16:
        parts.append(BF16)
    if not draw.greedy:
        parts.append(f"seed{seed}")
    if cap != OUTPUT_CAP:
        parts.append(f"max{cap}")
    return "-".join(parts)


def save(reply: Reply, header: dict[str, object], texts: list[str], tag: str) -> Path:
    """Save every step of a reply as JSON lines, the header first.

    Args:
        reply: The reply.
        header: What was asked and how tokens were drawn.
        texts: Each step's piece of text.
        tag: The run's name.

    Returns:
        The file written.
    """
    folder = chapter_home()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RECORD_NAME.format(tag=tag)
    lines = [json.dumps({**header, "finish_reason": reply.finish_reason})]
    for step, text in zip(reply.steps, texts, strict=True):
        lines.append(json.dumps({**asdict(step), "text": text}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default=QUESTION, help="what to ask")
    parser.add_argument("--temperature", type=float, default=TRAIN_TEMPERATURE)
    parser.add_argument("--greedy", action="store_true", help="the top token always")
    parser.add_argument("--top-k", type=int, default=NO_TOP_K, help="0 keeps all")
    parser.add_argument("--top-p", type=float, default=NO_TOP_P, help="1 keeps all")
    parser.add_argument("--max-tokens", type=int, default=OUTPUT_CAP)
    parser.add_argument("--dtype", choices=(FP32, BF16), default=FP32)
    parser.add_argument("--raw", action="store_true", help="show raw log-probs too")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--fast", action="store_true", help="use Liquid's export, not ours"
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be above 0 (use --greedy for the top token)")
    if args.top_k < NO_TOP_K:
        parser.error("--top-k must be 0 (keep all) or more")
    if not 0 < args.top_p <= NO_TOP_P:
        parser.error("--top-p must be above 0 and at most 1")
    if args.greedy and (args.top_k != NO_TOP_K or args.top_p != NO_TOP_P):
        parser.error("--greedy takes the top token: top-k and top-p cut nothing")
    if args.max_tokens < MIN_TOKENS:
        parser.error(f"--max-tokens must be at least {MIN_TOKENS}")
    return args


def main() -> None:
    """Draw one reply (two with --greedy) and show every number of the draw."""
    args = parse_args()
    draw = Draw(args.temperature, args.greedy, args.top_k, args.top_p, args.dtype)
    folder, which = pick_model(args.fast)
    vocabulary = read_vocabulary_size(folder)
    if vocabulary is not None and args.top_k >= vocabulary:
        raise SystemExit(f"--top-k must be below the model's {vocabulary:,} tokens")
    loaded = load_folder(folder)
    decode = loaded.tokenizer.decode
    prompt = chat_prompt(loaded.tokenizer, args.question)
    report.print_title(SCRIPT, TITLE)
    report.print_setup(
        folder,
        which,
        args.question,
        len(prompt),
        decode(prompt[-1:]),
        draw,
        args.seed,
        args.max_tokens,
    )
    report.print_suggested(read_generation_config(folder))

    mx.random.seed(args.seed)  # the same seed draws the same reply
    reply = sample(loaded, prompt, draw, args.max_tokens)
    texts = [decode([token]) for token in reply.tokens()]
    top = likeliest(reply.first, LIKELIEST_SHOWN)
    spread = entropy(reply.first).item()
    report.print_first_step(top, spread, reply.first.size, decode)
    named = [one.token for one in likeliest(reply.first_scores, NAMED_SHOWN)]
    rows = at_temperatures(reply.first_scores, named, SHOWN_TEMPERATURES)
    report.print_temperatures(rows, named, decode)
    report.print_steps(reply, texts, draw, args.raw)
    report.print_reply_text(decode(reply.tokens()))
    report.print_ending(reply, texts, loaded.tokenizer.eos_token_ids)
    if draw.greedy:
        report.print_greedy_again(reply, sample(loaded, prompt, draw, args.max_tokens))
    if draw.cuts():
        report.print_cut(reply, draw)
    if args.raw:
        report.print_raw(reply, draw)
    if draw.dtype == BF16:
        report.print_grid(reply)
    header = {
        "question": args.question,
        "model": str(folder),
        "draw": asdict(draw),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompt_tokens": len(prompt),
    }
    tag = record_tag(draw, args.seed, args.max_tokens)
    report.print_saved(save(reply, header, texts, tag))


if __name__ == "__main__":
    main()
