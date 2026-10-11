"""Load the 4-bit LFM2.5-2.6B with mlx-lm, and answer greedily.

A copy of chapter 14's ``lfm.py``, cut down to what this chapter needs:
loading one build (mlx-lm's ``load`` reads config.json, builds the layers,
fills them from the safetensors file and loads the tokenizer, as the
project's engine does) and a greedy reply over mlx-lm's ``generate_step``,
chapter 12's loop with the draw fixed to the likeliest token, which
eval_sql.py uses. grpo_sql.py draws its replies with ``sampler.py``.
``to_float32`` makes the whole model compute in float32 (``--float32``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from grading import FINISH_LENGTH, FINISH_STOP
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper

PREFILL_STEP = 1_024  # the project's Defaults.PREFILL_STEP (chapter 12)
SIXTEEN_BIT = (mx.bfloat16, mx.float16)  # the model's 16-bit arrays


@dataclass(frozen=True)
class LoadedModel:
    """A build in memory, with where it came from.

    Attributes:
        model: The network, every weight loaded.
        tokenizer: Its tokenizer, with the chat template.
        folder: The folder it was loaded from.
        seconds: How long loading took.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    folder: Path
    seconds: float


@dataclass(frozen=True)
class Reply:
    """A greedy reply: always the likeliest next token.

    Attributes:
        tokens: The token ids, the end of turn included if there is one.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        seconds: How long writing it took, the prompt's reading included.
    """

    tokens: list[int]
    finish_reason: str
    seconds: float


def load_folder(folder: Path) -> LoadedModel:
    """Load a build with mlx-lm, every weight into memory.

    Args:
        folder: The build's folder (read only; nothing is downloaded).

    Returns:
        The loaded model.
    """
    start = time.perf_counter()
    model, tokenizer = load(str(folder))
    mx.eval(model.parameters())  # MLX is lazy: read every weight now
    return LoadedModel(model, tokenizer, folder, time.perf_counter() - start)


def to_float32(model: nn.Module) -> None:
    """Cast every 16-bit array of the model to float32 (the 4-bit codes stay).

    The 4-bit weights are whole-number codes with a 16-bit scale and offset
    per group of 64 (chapter 11); casting the scales, offsets, norms and
    convolution weights to float32 makes every number the model works out
    float32 too.

    Args:
        model: The loaded model. LoRA's A and B matrices are float32 already
            and stay as they are, so it may come before or after the adapter.
    """
    cast = [
        (name, value.astype(mx.float32) if value.dtype in SIXTEEN_BIT else value)
        for name, value in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(cast))
    mx.eval(model.parameters())


def greedy_reply(
    model: nn.Module, tokenizer: TokenizerWrapper, prompt: list[int], cap: int
) -> Reply:
    """Answer by always taking the likeliest next token, up to a length cap.

    Args:
        model: The network (with or without adapters).
        tokenizer: Its tokenizer, for the end-of-turn ids.
        prompt: The prompt's token ids.
        cap: The most tokens the reply may have.

    Returns:
        The reply, why it ended, and how long it took.
    """
    start = time.perf_counter()
    tokens: list[int] = []
    for token, _ in generate_step(
        mx.array(prompt),
        model,
        max_tokens=cap,
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        prefill_step_size=PREFILL_STEP,
    ):
        tokens.append(int(token))
        if token in tokenizer.eos_token_ids:  # <|im_end|>: the end of turn
            return Reply(tokens, FINISH_STOP, time.perf_counter() - start)
    return Reply(tokens, FINISH_LENGTH, time.perf_counter() - start)
