"""Load the 4-bit LFM2.5-2.6B with mlx-lm, count it, and let it answer greedily.

A copy of chapter 11's and 12's ``lfm.py``, cut down to what this chapter
needs: loading one build (mlx-lm's ``load`` reads config.json, builds the
layers, fills them from the safetensors file and loads the tokenizer, as the
project's engine does), counting its numbers the way chapter 10 did (a packed
4-bit weight holds 32 / 4 numbers per stored word), rendering a question
with the chat template, and a greedy reply over mlx-lm's ``generate_step``,
chapter 12's loop with the draw fixed to the likeliest token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper

USER_ROLE = "user"
BITS_PER_WORD = 32  # a packed weight holds 32 / bits numbers per word
WEIGHT_KEY = "weight"
GROUP_KEYS = ("scales", "biases")  # a packed weight's per-group bookkeeping
FINISH_STOP = "stop"  # the model wrote its end-of-turn token
FINISH_LENGTH = "length"  # the length cap cut the reply off
PREFILL_STEP = 1_024  # the project's Defaults.PREFILL_STEP (chapter 12)


@dataclass(frozen=True)
class LoadedModel:
    """A build in memory, with where it came from.

    Attributes:
        model: The network, every weight loaded.
        tokenizer: Its tokenizer, with the chat template.
        folder: The folder it was loaded from.
        weight_bytes: How many bytes its arrays take.
        parameters: How many numbers the model has (packed ones unpacked).
        seconds: How long loading took.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    folder: Path
    weight_bytes: int
    parameters: int
    seconds: float


@dataclass(frozen=True)
class Reply:
    """A greedy reply: always the likeliest next token.

    Attributes:
        tokens: The token ids, the end of turn included if there is one.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
    """

    tokens: list[int]
    finish_reason: str


def weight_bytes(model: nn.Module) -> int:
    """Add up the bytes of every array the model holds.

    Args:
        model: The network.

    Returns:
        The bytes, packed weights and their scales included.
    """
    return sum(array.nbytes for _, array in tree_flatten(model.parameters()))


def parameter_count(model: nn.Module) -> int:
    """Count the model's numbers, unpacking packed weights.

    Args:
        model: The network.

    Returns:
        The parameter count.
    """
    total = 0
    for module in model.modules():
        bits = getattr(module, "bits", None)
        for key, array in module.parameters().items():
            if not isinstance(array, mx.array):
                continue  # a child module's parameters, counted on its own
            if bits and key in GROUP_KEYS:
                continue
            if bits and key == WEIGHT_KEY:
                total += array.size * BITS_PER_WORD // bits
            else:
                total += array.size
    return total


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
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        folder=folder,
        weight_bytes=weight_bytes(model),
        parameters=parameter_count(model),
        seconds=time.perf_counter() - start,
    )


def chat_prompt(tokenizer: TokenizerWrapper, question: str) -> list[int]:
    """Render one user message into prompt tokens, as the project's engine does.

    Args:
        tokenizer: The model's tokenizer.
        question: What the user asks.

    Returns:
        The prompt's token ids, ending with the generation prompt
        (``<|im_start|>assistant`` and ``<think>``).
    """
    text = tokenizer.apply_chat_template(
        [{"role": USER_ROLE, "content": question}],
        add_generation_prompt=True,  # end with the assistant's turn opening
        tokenize=False,
        preserve_thinking=True,  # keep reasoning, as the project's engine renders
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


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
        The reply and why it ended.
    """
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
            return Reply(tokens, FINISH_STOP)
    return Reply(tokens, FINISH_LENGTH)
