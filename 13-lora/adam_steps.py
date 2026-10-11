# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""What one AdamW step does: bias correction, weight decay and the LoRA scale.

No model is loaded. MLX's own AdamW is fed a steady gradient, the same at
every step, once with ``bias_correction=False`` (MLX's default) and once
with ``True`` (what the project passes), and each update's size is divided
by the learning rate. Then the same numbers for the project's two RL
attempts, what weight decay alone does to a weight, and how far one step
moves an mlx-lm LoRA layer's output at scale 2.0 and at 20.0.

Run it with ``uv run adam_steps.py`` (a few seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import report
from arithmetic import adam_ratio
from lfm_files import GRPO_LEARNING_RATE, LORA_RANK, LORA_SCALE, SFT_LEARNING_RATE
from mlx_lm.tuner.lora import LoRALinear

SCRIPT = "adam_steps.py"
TITLE = "what one AdamW step does"
SHOWN_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 20, 50, 100, 1_000)
STEADY_STEPS = SHOWN_STEPS[-1]
GRADIENTS = (1.0, 0.001)  # a steady gradient of each size
SAME_WITHIN = 0.001  # learning rates: "the same" for two gradient sizes
# The project's two RL attempts (mhrl/cli.py: "4e-5 (no bias correction)
# ballooned replies by step 5"; mhrl/config.py: attempt 1 "reached 404 by
# step 8").
ATTEMPT_ONE = ("attempt 1", 4e-5, False)
ATTEMPT_TWO = ("attempt 2", GRPO_LEARNING_RATE, True)
ATTEMPT_STEPS = 8
DECAY_RATES = (GRPO_LEARNING_RATE, SFT_LEARNING_RATE)
DECAY_STEPS = (1, 10, 100, 1_000)
NO_DECAY = 0.0
# The LoRA-scale experiment: one adapted 4-bit layer the width of LFM2.5-2.6B.
WIDTH = 2_048
GROUP_SIZE = 64  # chapter 11's 4-bit format
BITS = 4
ROWS = 8  # inputs pushed through the layer at once
MLX_DEFAULT_SCALE = 20.0  # mlx-lm's own LoRA default (its lora.py and LoRALinear)
SCALES = (LORA_SCALE, MLX_DEFAULT_SCALE)
SCALE_STEPS = (1, 20)
LAYER_SEED = 0
DATA_SEED = 1


@dataclass(frozen=True)
class Steady:
    """Each update's size, in learning rates, for a steady gradient.

    Attributes:
        without: Without bias correction (MLX's default), per step from 1.
        with_correction: With bias correction, per step from 1.
        same_for_sizes: Whether every gradient size gave the same column.
    """

    without: list[float]
    with_correction: list[float]
    same_for_sizes: bool


@dataclass(frozen=True)
class Attempt:
    """One of the project's RL attempts, as its optimizer was set.

    Attributes:
        name: ``"attempt 1"`` or ``"attempt 2"``.
        learning_rate: Its learning rate.
        bias_correction: Whether it corrected Adam's averages.
        updates: Each update's size for a steady gradient, per step from 1.
    """

    name: str
    learning_rate: float
    bias_correction: bool
    updates: list[float]


@dataclass(frozen=True)
class Decay:
    """A weight of 1.0 after steps with a zero gradient: weight decay alone.

    Attributes:
        learning_rate: The learning rate.
        values: The weight after each of DECAY_STEPS steps.
    """

    learning_rate: float
    values: list[float]


@dataclass(frozen=True)
class ScaleStep:
    """How far AdamW moves one LoRA layer's output at one scale.

    Attributes:
        scale: The LoRA scale.
        moved: The largest change in the layer's output after each of
            SCALE_STEPS steps.
        first_b: B after the first step.
    """

    scale: float
    moved: list[float]
    first_b: mx.array


def updates(
    learning_rate: float,
    bias_correction: bool,
    gradient: float,
    steps: int,
    weight_decay: float = NO_DECAY,
) -> list[float]:
    """Feed MLX's AdamW the same gradient every step, and measure each update.

    Args:
        learning_rate: AdamW's learning rate.
        bias_correction: Whether AdamW corrects its running averages.
        gradient: The gradient, the same at every step.
        steps: How many steps.
        weight_decay: AdamW's weight decay (left out here by default, so the
            updates are Adam's alone).

    Returns:
        Each update's size, per step from 1.
    """
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
    )
    weights = {"w": mx.zeros((1,))}
    grads = {"w": mx.full((1,), gradient)}
    sizes = []
    for _ in range(steps):
        before = weights["w"]
        weights = optimizer.apply_gradients(grads, weights)
        mx.eval(weights)
        sizes.append(abs((weights["w"] - before).item()))
    return sizes


def in_learning_rates(
    learning_rate: float, bias_correction: bool, gradient: float, steps: int
) -> list[float]:
    """Measure each update of a steady gradient, divided by the learning rate.

    Args:
        learning_rate: AdamW's learning rate.
        bias_correction: Whether AdamW corrects its running averages.
        gradient: The gradient, the same at every step.
        steps: How many steps.

    Returns:
        Each update's size in learning rates, per step from 1.
    """
    sizes = updates(learning_rate, bias_correction, gradient, steps)
    return [size / learning_rate for size in sizes]


def steady(learning_rate: float, steps: int) -> Steady:
    """Measure Adam's updates in learning rates, with and without bias correction.

    Each column is measured for every gradient size in GRADIENTS, to check
    that the size makes no difference (Adam divides it out).

    Args:
        learning_rate: AdamW's learning rate.
        steps: How many steps.

    Returns:
        Both columns, and whether the gradient's size changed anything.
    """
    columns, same = {}, True
    for correction in (False, True):
        runs = [
            in_learning_rates(learning_rate, correction, gradient, steps)
            for gradient in GRADIENTS
        ]
        first = runs[0]
        for other in runs[1:]:
            pairs = zip(first, other, strict=True)
            same = same and all(abs(a - b) < SAME_WITHIN for a, b in pairs)
        columns[correction] = first
    return Steady(columns[False], columns[True], same)


def attempt(name: str, learning_rate: float, bias_correction: bool) -> Attempt:
    """Measure the update sizes one RL attempt's optimizer gives a steady gradient.

    Args:
        name: The attempt's name.
        learning_rate: Its learning rate.
        bias_correction: Whether it corrected Adam's averages.

    Returns:
        The attempt, with one update size per step.
    """
    sizes = updates(learning_rate, bias_correction, GRADIENTS[0], ATTEMPT_STEPS)
    return Attempt(name, learning_rate, bias_correction, sizes)


def decay(learning_rate: float) -> Decay:
    """Let MLX's AdamW act on a weight of 1.0 whose gradient is always zero.

    With no gradient, Adam's averages stay zero and only the weight decay
    (MLX's default 0.01) moves the weight.

    Args:
        learning_rate: AdamW's learning rate.

    Returns:
        The weight after each of DECAY_STEPS steps.
    """
    optimizer = optim.AdamW(learning_rate=learning_rate, bias_correction=True)
    weights = {"w": mx.ones((1,))}
    grads = {"w": mx.zeros((1,))}
    values = []
    for step in range(1, DECAY_STEPS[-1] + 1):
        weights = optimizer.apply_gradients(grads, weights)
        mx.eval(weights)
        if step in DECAY_STEPS:
            values.append(weights["w"].item())
    return Decay(learning_rate, values)


def lora_layer(scale: float) -> LoRALinear:
    """Build one mlx-lm LoRA layer on a 4-bit 2,048 x 2,048 matrix, always the same.

    Args:
        scale: The LoRA scale.

    Returns:
        The layer, with only its A and B trainable.
    """
    mx.random.seed(LAYER_SEED)  # the same matrix and the same A at every scale
    base = nn.QuantizedLinear.from_linear(
        nn.Linear(WIDTH, WIDTH, bias=False), group_size=GROUP_SIZE, bits=BITS
    )
    layer = LoRALinear.from_base(base, r=LORA_RANK, dropout=0.0, scale=scale)
    mx.eval(layer.parameters())
    return layer


def scale_step(scale: float) -> ScaleStep:
    """Train one LoRA layer with AdamW and measure how far its output moves.

    The made-up loss pushes the output along a fixed random direction, so the
    gradient is never zero; AdamW is set as in the project's SFT.

    Args:
        scale: The LoRA scale.

    Returns:
        How far the output moved after each of SCALE_STEPS steps.
    """
    layer = lora_layer(scale)
    mx.random.seed(DATA_SEED)
    inputs = mx.random.normal((ROWS, WIDTH))
    direction = mx.random.normal((ROWS, WIDTH))
    start = layer(inputs)
    optimizer = optim.AdamW(learning_rate=SFT_LEARNING_RATE, bias_correction=True)

    def loss_fn(rows: mx.array) -> mx.array:
        """How far the output points along the direction (to be lowered).

        Args:
            rows: The inputs.

        Returns:
            One number.
        """
        return mx.sum(layer(rows) * direction)

    loss_and_grad = nn.value_and_grad(layer, loss_fn)
    moved, first_b = [], layer.lora_b  # first_b is replaced after step 1
    for step in range(1, SCALE_STEPS[-1] + 1):
        _, grads = loss_and_grad(inputs)
        optimizer.update(layer, grads)
        mx.eval(layer.parameters(), optimizer.state)
        if step == 1:
            first_b = layer.lora_b
        if step in SCALE_STEPS:
            moved.append(mx.max(mx.abs(layer(inputs) - start)).item())
    return ScaleStep(scale, moved, first_b)


def main() -> None:
    """Measure bias correction, the two RL attempts, weight decay and the scale."""
    defaults = optim.AdamW(learning_rate=GRPO_LEARNING_RATE)  # nothing else given
    report.print_title(SCRIPT, TITLE)
    report.print_defaults(defaults)
    # The steady gradient is measured at the project's RL rate, 1e-5.
    measured = steady(GRPO_LEARNING_RATE, STEADY_STEPS)
    beta1, beta2 = defaults.betas
    formula = [
        adam_ratio(step, beta1, beta2, bias_correction=False)
        for step in range(1, STEADY_STEPS + 1)
    ]
    settings = (GRPO_LEARNING_RATE, defaults.betas, GRADIENTS)
    report.print_steady(measured, formula, SHOWN_STEPS, settings)
    report.print_attempts([attempt(*ATTEMPT_ONE), attempt(*ATTEMPT_TWO)])
    report.print_decay(
        [decay(rate) for rate in DECAY_RATES], DECAY_STEPS, defaults.weight_decay
    )
    results = [scale_step(scale) for scale in SCALES]
    b_gap = mx.max(mx.abs(results[0].first_b - results[1].first_b)).item()
    report.print_scale(results, SCALE_STEPS, b_gap, WIDTH, LORA_RANK, SFT_LEARNING_RATE)


if __name__ == "__main__":
    main()
