"""LoRA helpers, as the project's ``mhrl/lora.py`` has them (chapter 13).

``apply_lora``, ``adapter_weights``, ``save_adapter``, ``load_adapter`` and
``zero_adapter`` are the project's functions, copied from chapter 13's
``lora.py`` with its comments; ``LORA_KEYS``, ``LORA_DROPOUT`` and
``LORA_SEED`` are its constants. ``zero_b_count`` is chapter 13's check that
a fresh adapter is a no-op. Chapter 25 copies chapter 14's copy and keeps
the project's settings here too (``mhrl/config.py``: rank 16, scale 2.0, the
last 12 layers), so the trainer, the memory probe and the engine build one
adapter shape, and ``check_adapter_file`` refuses a file of another shape
before ``load_weights`` (``strict=False``) could skip it without a word.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers

# Every projection in an LFM2 block: attention, short convolution and MLP.
LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "conv.in_proj",
    "conv.out_proj",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
LORA_DROPOUT = 0.0
LORA_SEED = 0
B_SUFFIX = "lora_b"
# The project's adapter shape (mhrl/config.py lora_rank, lora_scale, lora_layers).
LORA_RANK = 16
LORA_SCALE = 2.0
LORA_LAYERS = 12


def apply_lora(model: nn.Module, rank: int, scale: float, layers: int) -> None:
    """Freeze the base model and add LoRA adapters to its last ``layers`` blocks.

    The adapters start as an exact no-op (their B matrices are zero), so a
    freshly adapted model produces the base model's outputs.

    Args:
        model: An ``mlx_lm`` model.
        rank: LoRA rank.
        scale: LoRA scale (mlx-lm's equivalent of alpha / rank).
        layers: How many of the LAST blocks get adapters. Fewer blocks means
            the backward pass stops earlier, which is most of the speed-up.
    """
    mx.random.seed(LORA_SEED)  # the same random A matrices on every run
    model.freeze()  # every weight the model has stops training
    linear_to_lora_layers(  # ...and the new A and B matrices are trainable
        model,
        num_layers=min(layers, len(model.layers)),
        config={
            "rank": rank,
            "scale": scale,
            "dropout": LORA_DROPOUT,
            "keys": list(LORA_KEYS),
        },
    )


def adapter_weights(model: nn.Module) -> dict[str, mx.array]:
    """Return the trainable (LoRA) weights as a flat dict.

    Args:
        model: A model with LoRA layers applied.

    Returns:
        Mapping of parameter path to array.
    """
    return dict(tree_flatten(model.trainable_parameters()))


def save_adapter(model: nn.Module, path: Path) -> Path:
    """Write the LoRA weights to a safetensors file.

    Args:
        model: A model with LoRA layers applied.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), adapter_weights(model))
    return path


def check_adapter_file(model: nn.Module, path: Path) -> None:
    """Refuse an adapter file whose arrays are not exactly this model's adapter.

    ``load_weights(strict=False)`` skips every array it has no place for, so
    a file from another adapter shape would load nothing and the model would
    run as the base model while every label said "adapted".

    Args:
        model: A model with LoRA layers applied.
        path: The safetensors file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If its arrays' names or shapes differ from the model's.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = mx.load(str(path))
    expected = adapter_weights(model)
    if set(saved) != set(expected):
        raise ValueError(
            f"{path.name} holds {len(saved)} arrays, not this adapter's "
            f"{len(expected)} (another rank, scale or layer count?)"
        )
    for name, array in saved.items():
        if array.shape != expected[name].shape:
            raise ValueError(
                f"{path.name}: {name} is {array.shape}, not {expected[name].shape}"
            )


def load_adapter(model: nn.Module, path: Path) -> None:
    """Load LoRA weights written by ``save_adapter``.

    Args:
        model: A model with LoRA layers applied.
        path: The safetensors file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    # strict=False: the file holds only the adapter's arrays, not the model's.
    model.load_weights(str(path), strict=False)
    mx.eval(model.trainable_parameters())


def zero_adapter(model: nn.Module) -> None:
    """Reset every LoRA B matrix to zero, which restores the base model.

    Args:
        model: A model with LoRA layers applied.
    """
    updates = [
        (name, mx.zeros_like(value))
        for name, value in adapter_weights(model).items()
        if name.endswith(B_SUFFIX)
    ]
    model.load_weights(updates, strict=False)
    mx.eval(model.trainable_parameters())


def zero_b_count(model: nn.Module) -> tuple[int, int]:
    """Count the B matrices that are all zeros (chapter 13's no-op check).

    Args:
        model: A model with LoRA layers applied.

    Returns:
        How many B matrices are all zeros, and how many there are.
    """
    found = [
        value
        for name, value in adapter_weights(model).items()
        if name.endswith(B_SUFFIX)
    ]
    zeros = sum(1 for value in found if not mx.any(value != 0).item())
    return zeros, len(found)
