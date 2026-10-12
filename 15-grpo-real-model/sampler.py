"""Draw a reply token by token, keeping each token's id and log-probability.

Chapter 12's sampler, cut down to what training keeps: a loop over mlx-lm's
``generate_step`` (the function the project's engine runs on) that draws
every token at temperature 1 from the full distribution and records, for
each one, its id, its log-probability under the distribution it was drawn
from, and that distribution's entropy. The scores are cast to float32 before
the softmax, as the engine's ``_to_float32`` does.

Two settings reproduce what the project and the article warn about:

- ``dtype="bf16"`` skips the cast: the log-probabilities are worked out and
  recorded in bfloat16, as the project's engine did in its first RL run;
- ``top_k=50`` draws from the 50 likeliest tokens only and records the
  log-probability under that cut distribution, as an engine that applies
  top-k (or top-p) before reporting its numbers does.

Either way, the number the engine records today (float32, the full
distribution) is kept beside the recorded one for the same token, so a run
can say exactly how much the setting moved each token's old log-probability.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from grading import FINISH_LENGTH, FINISH_STOP
from lfm_files import BF16, FP32
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import apply_top_k

DTYPES = (FP32, BF16)
NO_TOP_K = 0  # keep every token
UNCHANGED = 1.0  # dividing the scores by 1 changes nothing
PREFILL_STEP = 1_024  # the project's Defaults.PREFILL_STEP (chapter 12)
MIN_TOKENS = 1  # the smallest length cap generate_step accepts
NUMBERS = 4  # kept per step: log p, entropy, the engine's log p, its entropy


@dataclass(frozen=True)
class Draw:
    """How each token is drawn.

    Attributes:
        temperature: The scores are divided by it before the softmax.
        top_k: Keep only this many of the likeliest tokens (0 keeps all).
        dtype: ``"fp32"`` or ``"bf16"``: the precision in which the scores
            become log-probabilities.
    """

    temperature: float = UNCHANGED
    top_k: int = NO_TOP_K
    dtype: str = FP32

    def __post_init__(self) -> None:
        """Refuse settings this chapter does not know.

        Raises:
            ValueError: If ``dtype`` is unknown, the temperature is not above
                0, or ``top_k`` is negative.
        """
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be {FP32!r} or {BF16!r}, not {self.dtype!r}")
        if self.temperature <= 0:
            raise ValueError("the temperature must be above 0")
        if self.top_k < NO_TOP_K:
            raise ValueError("top_k must be 0 (keep every token) or more")


@dataclass(frozen=True)
class Sampled:
    """One reply, with everything training needs from the draw.

    Attributes:
        tokens: The token ids drawn, the end of turn included if there is one.
        logprobs: Each token's log-probability under the distribution it was
            drawn from, as recorded: the old log-probabilities training
            compares against.
        entropies: Each step's entropy of that distribution, in nats.
        full_logprobs: The same tokens' log-probabilities as the project's
            engine records them: float32, the full distribution (equal to
            ``logprobs`` unless ``dtype="bf16"`` or ``top_k`` is set).
        full_entropies: Each step's entropy of the full distribution.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        seconds: How long drawing it took, the prompt's reading included.
    """

    tokens: list[int]
    logprobs: list[float]
    entropies: list[float]
    full_logprobs: list[float]
    full_entropies: list[float]
    finish_reason: str
    seconds: float


def drawn_from(logprobs: mx.array, draw: Draw) -> mx.array:
    """Turn the model's own log-probabilities into the distribution drawn from.

    Temperature first, then top-k (chapter 12's order); the tokens cut become
    minus infinity (probability 0) and the rest are renormalized.

    Args:
        logprobs: The model's own log-probabilities, ``log_softmax(scores)``.
        draw: How tokens are drawn.

    Returns:
        The log-probabilities of the distribution a token is drawn from.
    """
    if draw.temperature == UNCHANGED and draw.top_k == NO_TOP_K:
        return logprobs  # the engine's case: nothing to change
    scores = logprobs / draw.temperature
    if draw.top_k != NO_TOP_K:
        scores = apply_top_k(scores, draw.top_k)
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


def sample(
    model: nn.Module,
    eos_ids: Collection[int],
    prompt: list[int],
    draw: Draw,
    cap: int,
) -> Sampled:
    """Draw a reply token by token, keeping every number training needs.

    Args:
        model: The network, with the adapter that is being trained.
        eos_ids: The end-of-turn token ids.
        prompt: The prompt's token ids.
        draw: How tokens are drawn.
        cap: The most tokens the reply may have.

    Returns:
        The reply and its numbers.

    Raises:
        ValueError: If the cap is below one token.
    """
    if cap < MIN_TOKENS:
        raise ValueError(f"the cap must be at least {MIN_TOKENS} token")
    seen: deque[mx.array] = deque()  # each step's scores, oldest first

    def cast(tokens: mx.array, scores: mx.array) -> mx.array:
        """Put the step's scores aside, then cast them for the softmax.

        generate_step calls this once per step, in order, so ``seen`` pairs
        every yielded token with its own step's scores (oldest first).

        Args:
            tokens: The tokens so far, which this processor does not need.
            scores: The step's scores, in bfloat16 as the model computed them.

        Returns:
            The scores in float32 (the engine's _to_float32), or unchanged
            with ``dtype="bf16"``.
        """
        del tokens
        seen.append(scores)
        return scores.astype(mx.float32) if draw.dtype == FP32 else scores

    def pick(logprobs: mx.array) -> mx.array:
        """Draw the next token from the distribution ``drawn_from`` makes.

        Args:
            logprobs: The step's log-probabilities, from generate_step.

        Returns:
            The token drawn.
        """
        return mx.random.categorical(drawn_from(logprobs, draw))

    start = time.perf_counter()
    tokens: list[int] = []
    numbers: list[mx.array] = []  # per step: the four numbers below
    finish = FINISH_LENGTH
    engine = Draw(temperature=draw.temperature)  # what the engine draws from
    for token, logprobs in generate_step(
        mx.array(prompt),
        model,
        max_tokens=cap,
        sampler=pick,
        logits_processors=[cast],
        prefill_step_size=PREFILL_STEP,
    ):
        scores = seen.popleft()[0].astype(mx.float32)
        # The same function on the same numbers: the distribution pick() drew from.
        dist = drawn_from(logprobs, draw)
        # And what the engine records: the same scores, float32, nothing cut.
        full = drawn_from(scores - mx.logsumexp(scores), engine)
        values = [dist[token], entropy(dist), full[token], entropy(full)]
        step = mx.stack([value.astype(mx.float32) for value in values])
        mx.async_eval(step)  # four numbers kept; the 128,000 scores are let go
        numbers.append(step)
        tokens.append(int(token))
        if token in eos_ids:  # <|im_end|>: the end of turn
            finish = FINISH_STOP
            break
    rows = mx.stack(numbers).tolist() if numbers else []
    columns = [list(column) for column in zip(*rows, strict=True)]
    if not columns:  # nothing drawn: four empty lists, one per number
        columns = [[] for _ in range(NUMBERS)]
    recorded, entropies, full_logprobs, full_entropies = columns
    return Sampled(
        tokens=tokens,
        logprobs=recorded,
        entropies=entropies,
        full_logprobs=full_logprobs,
        full_entropies=full_entropies,
        finish_reason=finish,
        seconds=time.perf_counter() - start,
    )
