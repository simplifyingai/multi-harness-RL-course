"""Load LFM2.5-2.6B with mlx-lm, and turn a question into prompt tokens.

mlx-lm is Apple's library for running language models on Apple silicon with
MLX. Its ``load`` reads config.json, builds the model's layers (LFM2's live in
``mlx_lm/models/lfm2.py``), fills them from the safetensors files and loads
the tokenizer, exactly as the project's engine does (``mhrl/engine/policy.py``).
A prompt is rendered the way that engine renders one: the chat template with
the generation prompt, reasoning kept, then encoded without extra tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from lfm_files import ModelRepo, require_model
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

USER_ROLE = "user"


BITS_PER_WORD = 32  # a packed 4-bit weight holds 32 / bits numbers per word
WEIGHT_KEY = "weight"
GROUP_KEYS = ("scales", "biases")  # a packed weight's per-group bookkeeping


@dataclass(frozen=True)
class LoadedModel:
    """The model in memory, with where it came from.

    Attributes:
        model: The network, every weight loaded.
        tokenizer: Its tokenizer, with the chat template.
        repo: The Hub repository it was downloaded from.
        folder: The folder it was loaded from.
        weight_bytes: How many bytes its weights take in memory.
        parameters: How many numbers the model has.
    """

    model: nn.Module
    tokenizer: TokenizerWrapper
    repo: ModelRepo
    folder: Path
    weight_bytes: int
    parameters: int


def weight_bytes(model: nn.Module) -> int:
    """Add up the bytes of every array the model holds.

    Args:
        model: The network.

    Returns:
        The bytes, packed weights and their scales included.
    """
    return sum(array.nbytes for _, array in tree_flatten(model.parameters()))


def parameter_count(model: nn.Module) -> int:
    """Count the model's numbers, unpacking 4-bit weights.

    A 4-bit layer stores its weight packed, 32 / bits numbers to a 32-bit
    word, beside a scale and an offset per group of numbers; those two only
    help store the weight, so they are not counted.

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


def load_model(fast: bool) -> LoadedModel:
    """Load the downloaded model with mlx-lm, every weight into memory.

    Args:
        fast: True for the 4-bit export, False for the bfloat16 original.

    Returns:
        The loaded model.

    Raises:
        SystemExit: If the model is not downloaded, naming the command that
            downloads it.
    """
    repo, folder = require_model(fast)
    model, tokenizer = load(str(folder))  # reads only local files
    mx.eval(model.parameters())
    return LoadedModel(
        model, tokenizer, repo, folder, weight_bytes(model), parameter_count(model)
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
        add_generation_prompt=True,
        tokenize=False,
        preserve_thinking=True,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))
