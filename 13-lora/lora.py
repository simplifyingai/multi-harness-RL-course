"""LoRA helpers, as the project's ``mhrl/lora.py`` has them.

``apply_lora``, ``adapter_weights``, ``save_adapter``, ``load_adapter`` and
``zero_adapter`` are the project's functions, the same code with comments
added (and one docstring line corrected: ``apply_lora`` adapts the last
``layers`` blocks, not every block), and ``LORA_KEYS``, ``LORA_DROPOUT`` and
``LORA_SEED`` are its constants. The rank, scale and number of layers are the
project's settings (``mhrl/config.py``), kept in this chapter's
``lfm_files.py``.

Below them, the chapter adds what it needs to show the adapters: the list of
pairs layer by layer, how far an adapter moves each weight, and merging an
adapter into the frozen weights (which the project never does).
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from arithmetic import ATTENTION, CONVOLUTION, AdapterPair, WeightChange
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear
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
LAYER_NAME = re.compile(r"layers\.(\d+)\.(.+)$")  # "model.layers.18.conv.in_proj"


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
    # (It also means a file whose names match nothing would load as nothing;
    # this chapter only loads files it saved itself.)
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


# ------------------------------------------------- added by this chapter


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    """List the model's LoRA layers with their names.

    Args:
        model: A model with LoRA layers applied.

    Returns:
        ``(name, layer)`` pairs, such as ``("model.layers.18.conv.in_proj", ...)``.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]


def adapter_pairs(model: nn.Module) -> list[AdapterPair]:
    """Describe every adapter pair, read from the model itself.

    Args:
        model: A model with LoRA layers applied.

    Returns:
        One pair per adapted projection, by layer and then in LORA_KEYS order.

    Raises:
        ValueError: If a LoRA layer's name does not say which layer it is in.
    """
    pairs = []
    for name, module in lora_modules(model):
        found = LAYER_NAME.search(name)
        if found is None:
            raise ValueError(f"not inside a numbered layer: {name}")
        inputs, rank = module.lora_a.shape  # A: inputs x rank
        outputs = module.lora_b.shape[1]  # B: rank x outputs
        pairs.append(AdapterPair(int(found[1]), found[2], inputs, outputs, rank))
    order = {key: index for index, key in enumerate(LORA_KEYS)}
    unknown = len(order)  # a projection not in LORA_KEYS is listed last
    return sorted(pairs, key=lambda pair: (pair.layer, order.get(pair.key, unknown)))


def layer_kinds(model: nn.Module) -> dict[int, str]:
    """Say which layers are attention and which are convolution.

    Args:
        model: An LFM2 model (mlx-lm marks its attention layers).

    Returns:
        Each layer's kind, by index.
    """
    return {
        index: ATTENTION if getattr(layer, "is_attention_layer", False) else CONVOLUTION
        for index, layer in enumerate(model.layers)
    }


def zero_b_count(model: nn.Module) -> tuple[int, int]:
    """Count the B matrices that are all zeros.

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


def a_in_range(model: nn.Module) -> tuple[int, int]:
    """Count the A matrices that hold mlx-lm's starting values.

    mlx-lm starts each A with random numbers between -1/sqrt(inputs) and
    +1/sqrt(inputs), so A is never all zeros (B is, which is enough).

    Args:
        model: A model with LoRA layers applied.

    Returns:
        How many A matrices are inside that range and not all zeros, and how
        many A matrices there are.
    """
    found = 0
    modules = lora_modules(model)
    for _, module in modules:
        bound = module.lora_a.shape[0] ** -0.5  # 1 / sqrt(inputs)
        inside = mx.all(mx.abs(module.lora_a) <= bound).item()
        found += int(inside and mx.any(module.lora_a != 0).item())
    return found, len(modules)


def adapter_delta(module: LoRALinear) -> mx.array:
    """Work out what an adapter adds to its projection's weight matrix.

    The adapted layer computes ``W x + scale x (x A) B``, which is the same as
    a weight matrix ``W + scale x (A B)`` turned the way W is stored
    (outputs x inputs).

    Args:
        module: One LoRA layer.

    Returns:
        ``scale x B^T A^T``, outputs x inputs, in float32.
    """
    return module.scale * (module.lora_b.T @ module.lora_a.T)


def change_and_step(modules: list[tuple[str, LoRALinear]]) -> WeightChange:
    """Compare how far the adapters move each weight with the 4-bit step size.

    A 4-bit weight is ``scale x code + offset`` with a whole-number code, so
    it can only move in steps of its group's scale (chapter 11).

    Args:
        modules: LoRA layers on 4-bit projections.

    Returns:
        The average and the largest change the adapters make to a weight,
        and the average 4-bit step in those matrices.
    """
    change_sum = step_sum = 0.0
    weights = groups = 0
    largest = 0.0
    for _, module in modules:
        delta = mx.abs(adapter_delta(module))
        steps = mx.abs(module.linear.scales.astype(mx.float32))
        change_sum += mx.sum(delta).item()
        largest = max(largest, mx.max(delta).item())
        step_sum += mx.sum(steps).item()
        weights += delta.size
        groups += steps.size
    return WeightChange(change_sum / weights, largest, step_sum / groups)


def merged(
    modules: list[tuple[str, LoRALinear]], dequantize: bool
) -> list[tuple[str, nn.Module]]:
    """Fold each adapter into its frozen matrix, with mlx-lm's own ``fuse``.

    Args:
        modules: LoRA layers.
        dequantize: True keeps each merged matrix in 16 bits; False rounds it
            back to 4 bits, the frozen matrix's own format.

    Returns:
        ``(name, merged layer)`` pairs, ready for ``model.update_modules``.
    """
    return [(name, module.fuse(dequantize=dequantize)) for name, module in modules]


def module_bytes(modules: list[tuple[str, nn.Module]]) -> int:
    """Add up the bytes of some layers' arrays.

    Args:
        modules: ``(name, layer)`` pairs.

    Returns:
        Their arrays' bytes together.
    """
    return sum(
        array.nbytes
        for _, module in modules
        for _, array in tree_flatten(module.parameters())
    )
