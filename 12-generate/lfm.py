"""Load the 4-bit LFM2.5-2.6B with mlx-lm, and turn a question into tokens.

A copy of chapter 11's ``lfm.py``, cut down to loading one build and
rendering a prompt. mlx-lm's ``load`` reads config.json, builds the model's
layers, fills them from the safetensors file and loads the tokenizer, as the
project's engine does. A prompt is rendered the way that engine renders one:
the chat template with the generation prompt, reasoning kept, then encoded
without extra tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

USER_ROLE = "user"


@dataclass(frozen=True)
class LoadedModel:
    """A build in memory, with where it came from.

    Attributes:
        model: The network, every weight loaded.
        tokenizer: Its tokenizer, with the chat template.
        folder: The folder it was loaded from.
        weight_bytes: How many bytes its arrays take.
        seconds: How long loading took.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    folder: Path
    weight_bytes: int
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
    # MLX computes lazily: mx.eval reads every weight now, so that loading is
    # not counted as part of the first token's time (chapter 18 comes back to
    # lazy evaluation, and to why the project's engine does the same).
    mx.eval(model.parameters())
    size = sum(array.nbytes for _, array in tree_flatten(model.parameters()))
    return LoadedModel(model, tokenizer, folder, size, time.perf_counter() - start)


def render_chat(tokenizer: TokenizerWrapper, message: str) -> str:
    """Render one user message with the chat template, as the project's engine does.

    Args:
        tokenizer: The model's tokenizer.
        message: What the user writes.

    Returns:
        The prompt as text, ending with the generation prompt
        (``<|im_start|>assistant`` and ``<think>``).
    """
    return tokenizer.apply_chat_template(
        [{"role": USER_ROLE, "content": message}],
        add_generation_prompt=True,
        tokenize=False,
        preserve_thinking=True,
    )


def chat_prompt(tokenizer: TokenizerWrapper, question: str) -> list[int]:
    """Render one user message into prompt tokens, as the project's engine does.

    Args:
        tokenizer: The model's tokenizer.
        question: What the user asks.

    Returns:
        The prompt's token ids, ending with the generation prompt.
    """
    text = render_chat(tokenizer, question)
    return list(tokenizer.encode(text, add_special_tokens=False))
