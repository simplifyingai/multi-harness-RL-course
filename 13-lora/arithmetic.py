"""The chapter's arithmetic: adapter sizes, shares, bytes, and Adam's step sizes.

Everything here works on plain numbers after the model is done, so it needs
only the standard library (and the tests can check it anywhere).
lora_lab.py and adam_steps.py measure; this module adds up what they
measured; report.py prints it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

PERCENT = 100
FLOAT32_BYTES = 4  # the adapters' numbers are float32
# Training one number with AdamW in float32 keeps four float32 values: the
# number itself, its gradient, and Adam's two running averages.
TRAINING_COPIES = 4
TRAINING_BYTES_PER_NUMBER = TRAINING_COPIES * FLOAT32_BYTES
ATTENTION = "attention"
CONVOLUTION = "convolution"
PROJECTION_SUFFIX = "_proj"  # q_proj -> q, in_proj -> in, in short lists
SHOWN_AS_PERCENT = 0.01  # chances from 1% up are written as percentages
MILLION = 1_000_000
BILLION = 1_000_000_000
LN_10 = math.log(10)  # divide a natural log by this to get a power of ten


@dataclass(frozen=True)
class LoraShape:
    """The three settings ``apply_lora`` takes: how the adapters are made.

    Attributes:
        rank: The thin side of every A and B.
        scale: What each adapter's output is multiplied by.
        layers: How many of the last layers get adapters.
    """

    rank: int
    scale: float
    layers: int


@dataclass(frozen=True)
class AdapterPair:
    """One LoRA adapter: the pair of thin matrices beside one projection.

    Attributes:
        layer: The layer it sits in (0 to 29 in LFM2.5-2.6B).
        key: The projection, such as ``"self_attn.q_proj"``.
        inputs: The projection's input size (A is ``inputs x rank``).
        outputs: Its output size (B is ``rank x outputs``).
        rank: The rank, the thin side of both matrices.
    """

    layer: int
    key: str
    inputs: int
    outputs: int
    rank: int

    def numbers(self) -> int:
        """Count the pair's trainable numbers: A and B together.

        Returns:
            ``rank x (inputs + outputs)``.
        """
        return self.rank * (self.inputs + self.outputs)

    def full(self) -> int:
        """Count the numbers of the projection it adapts.

        Returns:
            ``inputs x outputs``: what training the matrix itself would train.
        """
        return self.inputs * self.outputs


@dataclass(frozen=True)
class LayerAdapters:
    """The adapters of one layer.

    Attributes:
        layer: The layer.
        kind: ``"attention"`` or ``"convolution"``.
        keys: The projections with an adapter, in the layer's order.
        numbers: Their trainable numbers together.
    """

    layer: int
    kind: str
    keys: list[str]
    numbers: int


@dataclass(frozen=True)
class KeyAdapters:
    """Every adapter on one kind of projection, across the layers.

    Attributes:
        key: The projection, such as ``"conv.in_proj"``.
        inputs: Its input size.
        outputs: Its output size.
        rank: The pairs' rank.
        per_pair: The trainable numbers of one pair.
        pairs: How many layers have this projection adapted.
    """

    key: str
    inputs: int
    outputs: int
    rank: int
    per_pair: int
    pairs: int

    def numbers(self) -> int:
        """Count the trainable numbers of all these pairs.

        Returns:
            ``per_pair x pairs``.
        """
        return self.per_pair * self.pairs


@dataclass(frozen=True)
class WeightChange:
    """How far merging an adapter would move the frozen weights.

    Attributes:
        mean: The average change to one weight.
        largest: The largest change to any weight.
        step: The average 4-bit step (a group's scale) in those matrices.
    """

    mean: float
    largest: float
    step: float


def by_layer(
    pairs: Iterable[AdapterPair], kinds: Mapping[int, str]
) -> list[LayerAdapters]:
    """Group the adapter pairs by layer, in layer order.

    Args:
        pairs: Every pair; within a layer, in the order its keys should be listed.
        kinds: Each layer's kind.

    Returns:
        One entry per layer that has adapters.

    Raises:
        KeyError: If a pair's layer has no kind in ``kinds``.
    """
    layers: dict[int, list[AdapterPair]] = {}
    for pair in pairs:
        layers.setdefault(pair.layer, []).append(pair)
    return [
        LayerAdapters(
            layer=layer,
            kind=kinds[layer],
            keys=[pair.key for pair in found],
            numbers=sum(pair.numbers() for pair in found),
        )
        for layer, found in sorted(layers.items())
    ]


def by_key(pairs: Iterable[AdapterPair], order: Iterable[str]) -> list[KeyAdapters]:
    """Group the adapter pairs by projection.

    Args:
        pairs: Every pair.
        order: The projections in the order to list them.

    Returns:
        One entry per projection that has at least one adapter.

    Raises:
        ValueError: If one projection has pairs of different sizes.
    """
    found: dict[str, list[AdapterPair]] = {}
    for pair in pairs:
        found.setdefault(pair.key, []).append(pair)
    rows = []
    for key in order:
        same = found.get(key, [])
        if not same:
            continue
        sizes = {(pair.inputs, pair.outputs, pair.rank) for pair in same}
        if len(sizes) != 1:
            raise ValueError(f"{key} has pairs of different sizes: {sorted(sizes)}")
        first = same[0]
        rows.append(
            KeyAdapters(
                key, first.inputs, first.outputs, first.rank, first.numbers(), len(same)
            )
        )
    return rows


def kind_counts(layers: Iterable[LayerAdapters]) -> list[tuple[str, int, int]]:
    """Count layers and pairs per layer for each kind of layer.

    Args:
        layers: The adapted layers.

    Returns:
        ``(kind, layers, pairs per layer)`` for each kind, attention first.

    Raises:
        ValueError: If layers of one kind have different numbers of pairs.
    """
    counts: dict[str, list[int]] = {}
    for layer in layers:
        counts.setdefault(layer.kind, []).append(len(layer.keys))
    rows = []
    for kind in sorted(counts, key=lambda name: name != ATTENTION):
        sizes = set(counts[kind])
        if len(sizes) != 1:
            raise ValueError(f"{kind} layers have different pair counts: {sizes}")
        rows.append((kind, len(counts[kind]), sizes.pop()))
    return rows


def short_keys(keys: Iterable[str]) -> str:
    """Write a layer's adapted projections briefly, grouped by block.

    Args:
        keys: Projections such as ``"self_attn.q_proj"``, in order.

    Returns:
        Such as ``"self_attn q/k/v/out, feed_forward w1/w2/w3"``.
    """
    blocks: dict[str, list[str]] = {}
    for key in keys:
        block, _, name = key.rpartition(".")
        blocks.setdefault(block, []).append(name.removesuffix(PROJECTION_SUFFIX))
    return ", ".join(f"{block} {'/'.join(names)}" for block, names in blocks.items())


def share(part: float, whole: float) -> float:
    """Work out a part as a percentage of a whole.

    Args:
        part: The part.
        whole: The whole (above 0).

    Returns:
        The percentage.

    Raises:
        ValueError: If the whole is not above 0.
    """
    if whole <= 0:
        raise ValueError("the whole must be above 0")
    return PERCENT * part / whole


def training_bytes(numbers: int) -> int:
    """Work out the memory AdamW needs to train numbers in float32.

    Args:
        numbers: How many numbers train.

    Returns:
        ``numbers x 16`` bytes: each number, its gradient and Adam's two
        running averages, 4 bytes apiece.
    """
    return numbers * TRAINING_BYTES_PER_NUMBER


def adam_ratio(step: int, beta1: float, beta2: float, bias_correction: bool) -> float:
    """Work out Adam's update, divided by the learning rate, for a steady gradient.

    With the same gradient g at every step, Adam's running averages after
    ``step`` steps are m = (1 - beta1**step) g and v = (1 - beta2**step) g**2,
    because both start at zero. The update is m / sqrt(v) (Adam's tiny eps,
    1e-8, is left out: it matters only for gradients near that size).

    Args:
        step: The step, from 1.
        beta1: The gradient average's decay (0.9 by default in MLX).
        beta2: The squared gradient average's decay (0.999 by default).
        bias_correction: Whether the averages are corrected.

    Returns:
        The update's size in learning rates.

    Raises:
        ValueError: If the step is below 1.
    """
    if step < 1:
        raise ValueError("steps count from 1")
    if bias_correction:
        # m / (1 - beta1**step) is g and v / (1 - beta2**step) is g**2: the
        # corrected update is g / |g|, one learning rate, at every step.
        return 1.0
    return (1 - beta1**step) / math.sqrt(1 - beta2**step)


def largest(values: list[float]) -> tuple[int, float]:
    """Find the largest value and its step, counting steps from 1.

    Args:
        values: One value per step.

    Returns:
        The step and the value (the first step if several tie).

    Raises:
        ValueError: If there are no values.
    """
    if not values:
        raise ValueError("no values")
    best = max(range(len(values)), key=lambda index: values[index])
    return best + 1, values[best]


def chance_text(total: float) -> str:
    """Write the chance e**total in words a person can read.

    Args:
        total: A log-probability (0 or below).

    Returns:
        Such as ``"64.2%"``, ``"1 in 1,612"``, ``"1 in 3.4 million"`` or
        ``"1 in 10^45.9"``.
    """
    probability = math.exp(total)
    if probability >= SHOWN_AS_PERCENT:
        return f"{PERCENT * probability:.1f}%"
    if -total >= math.log(BILLION):
        return f"1 in 10^{-total / LN_10:.1f}"
    against = 1 / probability
    if round(against) < MILLION:
        return f"1 in {against:,.0f}"
    return f"1 in {against / MILLION:.1f} million"


def checkpoint_steps(steps: int, every: int) -> list[int]:
    """List the steps after which an adapter is saved.

    Args:
        steps: How many steps the run takes.
        every: Save after every this many steps.

    Returns:
        Every multiple of ``every`` up to ``steps``, and the last step.

    Raises:
        ValueError: If either number is below 1.
    """
    if steps < 1 or every < 1:
        raise ValueError("steps and every must be at least 1")
    saved = list(range(every, steps + 1, every))
    if not saved or saved[-1] != steps:
        saved.append(steps)
    return saved
