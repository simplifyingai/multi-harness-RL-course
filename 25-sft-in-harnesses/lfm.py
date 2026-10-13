"""The model behind the engine: the 4-bit LFM2.5-2.6B, on MLX, one reply at a time.

The project's ``PolicyEngine`` model code (``mhrl/engine/policy.py``), cut to
what chapter 18 needs. ``sample`` is chapter 12's loop over mlx-lm's
``generate_step``, with the three things the engine adds (chapter 20 copies
it unchanged but for ``Sampling.raw_logprobs``, its ``--raw-logprobs``):

- **a prompt cache**: the engine hands over a cache that already holds the
  start of the prompt, and only the new tokens are read;
- **honest log-probabilities**: the scores are cast to float32 before the
  softmax (in bfloat16 every log-probability lands on a 1/16 grid), and each
  sampled token's log-probability is taken under the distribution it was
  drawn from, ``log_softmax(scores / T)``, over the whole vocabulary (no
  top-p, no top-k). At temperature 0 the untempered distribution is reported,
  and with ``Sampling.raw_logprobs`` at every temperature (vLLM's default,
  the mistake chapter 20's capture proxy probes for);
- **alternatives**: on request, the likeliest few tokens at each position.

MLX computes lazily and keeps a stream and a random-number state per
thread, so the weights are read into memory here, at load, every later step
runs on the engine's one worker thread, and the seed is set on the thread
that samples (set at load, it would never reach the worker).

Chapter 25 adds ``set_adapter``: the first adapter adds chapter 13's LoRA
layers (the project's shape, from ``lora.py``), and every later switch loads
another file into them, or zeroes their B matrices for the base model. The
engine calls it on its worker thread, so the new arrays live where they are
used.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
from lfm_files import short_path
from lfm_text import LfmText
from lora import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    apply_lora,
    check_adapter_file,
    load_adapter,
    zero_adapter,
)
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from policy import FINISH_LENGTH, FINISH_STOP, GREEDY, Sampled, Sampling

PREFILL_STEP = 1_024  # the project's Defaults.PREFILL_STEP (chapter 12)
SERVES = "lfm2.5-2.6b 4-bit"  # what the engine's /health says it serves
BASE_ADAPTER = "base"
UNTEMPERED = 1.0  # dividing the scores by 1 changes nothing
STOP_WINDOW = 32  # the last tokens decoded to look for a stop text


def to_float32(tokens: mx.array, scores: mx.array) -> mx.array:
    """Cast a step's scores to float32 before the softmax (the engine's _to_float32).

    Args:
        tokens: The tokens so far, which this processor does not need.
        scores: The step's scores, in bfloat16 as the model computed them.

    Returns:
        The scores in float32.
    """
    del tokens
    return scores.astype(mx.float32)


def tempered(logprobs: mx.array, temperature: float) -> mx.array:
    """Turn the model's log-probabilities into the sampling distribution's.

    ``log_softmax(scores / T)``: dividing log-probabilities by T and
    normalizing again gives the same numbers, since the softmax's constant
    cancels.

    Args:
        logprobs: ``log_softmax(scores)`` over the vocabulary.
        temperature: Above 0.

    Returns:
        The log-probabilities a token is drawn from at that temperature.
    """
    scaled = logprobs / temperature
    return scaled - mx.logsumexp(scaled)


class NotLfm2Error(Exception):
    """Raised when a build has no attention layer to count cached tokens by."""


class LfmModel(LfmText):
    """LFM2.5-2.6B in 4 bits: the chat template, the tokenizer and the weights.

    Attributes:
        model: The network, every weight in memory.
        attention_layer: The first attention layer, whose cache counts tokens.
        seed: MLX's random seed, set once on each thread that samples.
        seeded: The threads it has been set on.
        folder: The build it was loaded from.
        seconds: How long loading took.
        adapted: Whether the LoRA layers are in place yet.
        serves: What the engine's ``/health`` says it serves.
    """

    serves = SERVES

    def __init__(self, folder: Path, seed: int) -> None:
        """Load a build with mlx-lm, every weight into memory.

        Args:
            folder: The build's folder (read only; nothing is downloaded).
            seed: MLX's random seed, for replies drawn above temperature 0.

        Raises:
            NotLfm2Error: If the build has no attention layer.
        """
        start = time.perf_counter()
        model, tokenizer = load(str(folder))
        # MLX is lazy: read every weight now, on this thread, so the worker
        # thread never has to evaluate an array this thread created.
        mx.eval(model.parameters())
        super().__init__(tokenizer, frozenset(tokenizer.eos_token_ids))
        self.model = model
        # LFM2 mixes attention and convolution layers; a convolution layer
        # keeps a small fixed state, an attention layer one entry per token.
        attention = [
            index
            for index, layer in enumerate(model.layers)
            if getattr(layer, "is_attention_layer", False)
        ]
        if not attention:
            raise NotLfm2Error(f"{folder} has no attention layer: not an LFM2 build")
        self.attention_layer = attention[0]
        self.seed = seed
        self.seeded: set[int] = set()
        self.folder = folder
        self.seconds = time.perf_counter() - start
        self.adapted = False

    def set_adapter(self, path: Path | None) -> str:
        """Sample with a LoRA adapter from now on, or with none.

        Args:
            path: An adapter file (chapter 13's shape), or None for the
                base model.

        Returns:
            A label for what is in use: ``base``, or the file (home as ``~``).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If it is not an adapter of this shape.
        """
        if path is None:
            if self.adapted:  # B = 0 makes every LoRA layer add exactly 0
                zero_adapter(self.model)
            return BASE_ADAPTER
        if not self.adapted:
            apply_lora(self.model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
            self.adapted = True
        check_adapter_file(self.model, path)  # nothing changes if it is refused
        load_adapter(self.model, path)
        return short_path(path)

    def new_cache(self) -> list[Any]:
        """Make an empty prompt cache, one entry per layer.

        Returns:
            mlx-lm's cache for this model.
        """
        return make_prompt_cache(self.model)

    def cache_size(self, cache: list[Any]) -> int:
        """Say how many tokens a cache holds.

        Args:
            cache: A cache from ``new_cache``.

        Returns:
            Its attention layer's offset: every token read or written.
        """
        return int(cache[self.attention_layer].offset)

    def drawer(self, temperature: float) -> Callable[[mx.array], mx.array]:
        """Build the function that picks each token (no top-p, no top-k).

        Args:
            temperature: 0 for the likeliest token; above 0, drawn.

        Returns:
            A function from a step's log-probabilities to a token.
        """
        if temperature <= GREEDY:
            return lambda logprobs: mx.argmax(logprobs, axis=-1)
        return lambda logprobs: mx.random.categorical(logprobs / temperature)

    def hits_stop(self, tokens: list[int], stop: tuple[str, ...]) -> bool:
        """Say whether the reply's last tokens spell one of the stop texts.

        Args:
            tokens: The reply so far.
            stop: The stop texts.

        Returns:
            True when one of them appears in the last few tokens' text.
        """
        tail = self.decode(tokens[-STOP_WINDOW:])
        return any(text in tail for text in stop)

    def sample(
        self, new_ids: list[int], cache: list[Any], sampling: Sampling
    ) -> Sampled:
        """Read the new prompt tokens after the cache, then draw a reply.

        Args:
            new_ids: The prompt tokens the cache does not hold yet.
            cache: The cache; it ends up holding the reply too.
            sampling: How to draw.

        Returns:
            The reply, each token with its log-probability.
        """
        start = time.perf_counter()
        thread = threading.get_ident()
        if thread not in self.seeded:  # MLX's random state belongs to a thread
            mx.random.seed(self.seed)
            self.seeded.add(thread)
        drawing = sampling.temperature > GREEDY
        # Raw log-probabilities skip the temperature: the token is still drawn
        # at T, but the number reported is log_softmax(scores), not the draw's.
        report_tempered = drawing and not sampling.raw_logprobs
        scale = sampling.temperature if drawing else UNTEMPERED
        tokens: list[int] = []
        kept: list[mx.array] = []  # one log-probability per token, still lazy
        tops: list[list[tuple[int, float]]] = []
        first = 0.0
        finish = FINISH_LENGTH
        steps = generate_step(
            mx.array(new_ids),
            self.model,
            max_tokens=sampling.max_tokens,
            sampler=self.drawer(sampling.temperature),
            prompt_cache=cache,
            prefill_step_size=PREFILL_STEP,
            logits_processors=[to_float32],
            # MLX streams belong to a thread: use this (the worker's) own.
            stream=mx.default_stream(mx.default_device()),
        )
        for token, logprobs in steps:
            if not tokens:
                first = time.perf_counter() - start
            policy = tempered(logprobs, scale) if report_tempered else logprobs
            kept.append(policy[token])  # lazy: one sync per token is enough
            if sampling.top_logprobs:
                best = mx.argsort(-policy)[: sampling.top_logprobs]
                pairs = zip(best.tolist(), policy[best].tolist(), strict=True)
                tops.append(list(pairs))
            tokens.append(int(token))
            if token in self.eos_ids:  # <|im_end|>: the end of turn
                finish = FINISH_STOP
                break
            if sampling.stop and self.hits_stop(tokens, sampling.stop):
                finish = FINISH_STOP
                break
        values = mx.stack(kept).tolist() if kept else []
        return Sampled(tokens, values, tops, finish, first)


def active_memory() -> int:
    """Return the memory MLX holds right now (the weights, once loaded).

    Returns:
        Bytes.
    """
    return mx.get_active_memory()


def peak_memory() -> int:
    """Return the most memory MLX has held since the process started.

    Returns:
        Bytes, the weights included.
    """
    return mx.get_peak_memory()
