"""Load any build of LFM2.5-2.6B with mlx-lm, and turn a question into tokens.

A copy of chapter 10's ``lfm.py``, changed to load a build from any folder
(the original, ours, the all-4-bit variant or Liquid's export) and to free it
again, since this chapter keeps only one model in memory at a time. mlx-lm's
``load`` reads config.json, builds the model's layers, fills them from the
safetensors file (packed 4-bit layers when config.json has a quantization
block) and loads the tokenizer, as the project's engine does. A prompt is
rendered the way that engine renders one.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_tokenizer

USER_ROLE = "user"
BITS_PER_WORD = 32  # a packed weight holds 32 / bits numbers per word
WEIGHT_KEY = "weight"
GROUP_KEYS = ("scales", "biases")  # a packed weight's per-group bookkeeping


@dataclass(frozen=True)
class LoadedModel:
    """A build in memory, with where it came from.

    Attributes:
        model: The network, every weight loaded.
        tokenizer: Its tokenizer, with the chat template.
        folder: The folder it was loaded from.
        weight_bytes: How many bytes its arrays take.
        active_bytes: MLX's memory in use right after loading, the number the
            project's engine logs when it starts.
        parameters: How many numbers the model has (packed ones unpacked).
        seconds: How long loading took.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    folder: Path
    weight_bytes: int
    active_bytes: int
    parameters: int
    seconds: float


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


def release_memory() -> None:
    """Give MLX's freed memory back, so the next build starts from empty.

    Call it after the last reference to a model is gone.
    """
    gc.collect()
    mx.clear_cache()


def load_folder(folder: Path) -> LoadedModel:
    """Load a build with mlx-lm, every weight into memory.

    Args:
        folder: The build's folder (read only; nothing is downloaded).

    Returns:
        The loaded model.
    """
    start = time.perf_counter()
    model, tokenizer = load(str(folder))
    mx.eval(model.parameters())
    seconds = time.perf_counter() - start
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        folder=folder,
        weight_bytes=weight_bytes(model),
        active_bytes=mx.get_active_memory(),
        parameters=parameter_count(model),
        seconds=seconds,
    )


def load_tokenizer_only(folder: Path) -> TokenizerWrapper:
    """Load a build's tokenizer and chat template, without its weights.

    Args:
        folder: The build's folder.

    Returns:
        The tokenizer.
    """
    return load_tokenizer(folder)


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
        add_generation_prompt=True,
        tokenize=False,
        preserve_thinking=True,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))
