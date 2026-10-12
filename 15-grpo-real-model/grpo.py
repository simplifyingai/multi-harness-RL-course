"""One GRPO update on the real model: the project's scoring and clipped loss.

``token_logprobs`` is the project's (``mhrl/training/common.py``), as in
chapter 14: the row is read in one forward pass and each sampled token's
log-probability is read from the scores at the position before it, in
float32. ``clipped_loss`` is the project's ``loss_fn`` from
``mhrl/training/grpo.py`` line for line (chapter 6 copied it into NumPy):

    ratio = exp(new log p - old log p)        per sampled token
    loss  = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), summed over the row

``grpo_update`` is the inner part of the project's ``train_grpo``: every row
scored one at a time, each row's gradient divided by the step's trained-token
count and added up, then one AdamW step (bias correction on) on the adapter.

``kl_loss`` adds what the project does without: a penalty for drifting from
the starting model, ``coef x (exp(ref - new) - (ref - new) - 1)`` per token
(the estimate GRPO's paper uses), where ``ref`` is the token's
log-probability under the adapter training started from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from lora import adapter_weights
from mlx.utils import tree_map
from rewards import Defaults
from rows import TrainItem

LOGIT_CHUNK = 512  # positions per lm-head chunk: 512 x 128K vocab x 4 B = 256 MB
MEMORY_LIMIT_BYTES = 9 * 1024**3  # the project trainer's share of the budget
CACHE_LIMIT_BYTES = 512 * 1024**2
NO_KL = 0.0  # the project's: no penalty for drifting from the start


@dataclass(frozen=True)
class Update:
    """What one update learned from, and every trained token's numbers.

    Attributes:
        rows: How many rows it trained on (0: no update was made).
        tokens: How many sampled tokens those rows hold.
        loss_per_token: The summed loss divided by ``tokens``, or None.
        ratios: Per row, every trained token's importance ratio, worked out
            before the update.
        kl: Per row, every trained token's drift estimate (empty lists
            without ``--kl``).
    """

    rows: int
    tokens: int
    loss_per_token: float | None
    ratios: list[list[float]]
    kl: list[list[float]]


def set_memory_limits() -> None:
    """Cap MLX's memory and its cache of freed buffers, as the project does."""
    mx.set_memory_limit(MEMORY_LIMIT_BYTES)
    mx.set_cache_limit(CACHE_LIMIT_BYTES)


def token_logprobs(model: nn.Module, item: TrainItem) -> mx.array:
    """Log-probability of each trainable token under the current weights.

    Only the positions that are trained get an lm-head projection (in chunks),
    because full-sequence logits over a 128K vocabulary would cost gigabytes.

    Args:
        model: The LoRA model.
        item: The training row.

    Returns:
        Array of shape ``[n_tokens]``.
    """
    tokens = mx.array(item.input_ids)[None]
    # One forward pass over the whole row except its last token: the last
    # token is never followed by anything to predict.
    hidden = model.model(tokens[:, :-1])[0]
    targets = mx.array([item.input_ids[p] for p in item.positions])
    # The token at position p is predicted by the scores at position p - 1.
    before = mx.array([p - 1 for p in item.positions])
    pieces = []
    for start in range(0, len(item.positions), LOGIT_CHUNK):
        index = before[start : start + LOGIT_CHUNK]
        logits = model.model.embed_tokens.as_linear(hidden[index]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = targets[start : start + LOGIT_CHUNK]
        pieces.append(mx.take_along_axis(logp, chosen[:, None], axis=-1)[:, 0])
    return mx.concatenate(pieces)


def clipped_loss(
    model: nn.Module,
) -> Callable[[TrainItem], tuple[mx.array, mx.array]]:
    """Build the project's clipped loss for this model.

    Args:
        model: The LoRA model; the loss reads its current weights.

    Returns:
        ``loss_fn(item) -> (loss, ratio)``, the project's function.
    """
    eps = Defaults.CLIP_EPSILON

    def loss_fn(item: TrainItem) -> tuple[mx.array, mx.array]:
        """The clipped loss of one row, and each token's ratio (the project's).

        Args:
            item: The row: its tokens, old log-probabilities and advantage.

        Returns:
            The loss summed over the row's sampled tokens, and their ratios.
        """
        new = token_logprobs(model, item)
        ratio = mx.exp(new - mx.array(item.old_logprobs))
        unclipped = ratio * item.advantage
        clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
        return -mx.minimum(unclipped, clipped).sum(), mx.stop_gradient(ratio)

    return loss_fn


def kl_loss(
    model: nn.Module, coef: float
) -> Callable[[TrainItem, mx.array], tuple[mx.array, mx.array, mx.array]]:
    """Build the clipped loss plus a penalty for drifting from the start.

    Its first lines repeat ``clipped_loss`` on purpose: that one stays exactly
    the project's, so the version with the penalty is written out beside it.

    Args:
        model: The LoRA model.
        coef: How much a unit of drift costs (``--kl``).

    Returns:
        ``loss_fn(item, reference) -> (loss, ratio, kl)``.
    """
    eps = Defaults.CLIP_EPSILON

    def loss_fn(
        item: TrainItem, reference: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """The clipped loss plus coef x the drift estimate, for one row.

        Args:
            item: The row.
            reference: Its tokens' log-probabilities under the start adapter.

        Returns:
            The row's loss, each token's ratio, each token's drift estimate.
        """
        new = token_logprobs(model, item)
        ratio = mx.exp(new - mx.array(item.old_logprobs))
        unclipped = ratio * item.advantage
        clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
        gap = reference - new
        drift = mx.exp(gap) - gap - 1.0  # 0 when new == reference, else above 0
        loss = (-mx.minimum(unclipped, clipped) + coef * drift).sum()
        return loss, mx.stop_gradient(ratio), mx.stop_gradient(drift)

    return loss_fn


def trainer_logprobs(model: nn.Module, item: TrainItem) -> mx.array:
    """Score a row by the trainer's route, without working out its gradient.

    Inside a gradient computation MLX runs attention as separate steps
    instead of its fused kernel, and in bfloat16 the two routes round
    differently (same_tokens.py measures it). So the numbers the update sees
    come from a pass built for a gradient; this builds one, keeps the values
    and never evaluates the gradient, so nothing of the backward pass is done.

    Args:
        model: The LoRA model.
        item: The training row.

    Returns:
        Array of shape ``[n_tokens]``: what ``clipped_loss`` would see.
    """
    kept: list[mx.array] = []

    def total(row: TrainItem) -> mx.array:
        """Score the row and keep the scores, for a gradient that is not used.

        Args:
            row: The training row.

        Returns:
            The sum of its log-probabilities.
        """
        values = token_logprobs(model, row)
        kept.append(mx.stop_gradient(values))
        return values.sum()

    nn.value_and_grad(model, total)(item)  # the gradient stays unevaluated
    mx.eval(kept[0])
    return kept[0]


def score_with(
    model: nn.Module, weights: dict[str, mx.array], items: list[TrainItem]
) -> list[mx.array]:
    """Score rows under other adapter weights, then put the current ones back.

    One model stays in memory: the start adapter's weights (39 MB) are swapped
    in for the scoring and swapped out again. The rows are scored by the
    trainer's route, so at the first step, when the weights are the start's,
    every token's drift is exactly 0.

    Args:
        model: The LoRA model.
        weights: The adapter weights to score with.
        items: The rows.

    Returns:
        Each row's tokens' log-probabilities under ``weights``.
    """
    current = adapter_weights(model)
    model.load_weights(list(weights.items()), strict=False)
    try:
        scored = [trainer_logprobs(model, item) for item in items]
    finally:
        model.load_weights(list(current.items()), strict=False)
        mx.eval(model.trainable_parameters())
    return scored


def grpo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    items: list[TrainItem],
    kl_coef: float = NO_KL,
    reference: dict[str, mx.array] | None = None,
) -> Update:
    """One update from the rows of every group that had contrast.

    Args:
        model: The LoRA model; only the adapter trains.
        optimizer: AdamW, built once for the whole run.
        items: The step's rows (none: no update is made).
        kl_coef: The drift penalty's weight; 0 is the project's (none).
        reference: The start adapter's weights, needed when ``kl_coef`` > 0.

    Returns:
        What the update trained on, with every token's ratio.

    Raises:
        ValueError: If a drift penalty is asked for without reference weights.
    """
    if not items:
        return Update(0, 0, None, [], [])
    use_kl = kl_coef > NO_KL
    if use_kl and reference is None:
        raise ValueError("--kl needs the start adapter's weights to compare with")
    references = score_with(model, reference, items) if use_kl else []
    loss_and_grad = nn.value_and_grad(
        model, kl_loss(model, kl_coef) if use_kl else clipped_loss(model)
    )
    n_tokens = sum(item.n_tokens for item in items)
    summed, loss_total = None, 0.0
    ratios: list[list[float]] = []
    drifts: list[list[float]] = []
    for index, item in enumerate(items):  # one row at a time, as the project does
        extra = (references[index],) if use_kl else ()
        outputs, grads = loss_and_grad(item, *extra)
        # Divide by the step's trained tokens, so the sum over its rows is
        # the gradient of the loss per token.
        grads = tree_map(lambda g, n=n_tokens: g / n, grads)
        summed = grads if summed is None else tree_map(mx.add, summed, grads)
        mx.eval(summed, *outputs)
        loss_total += float(outputs[0])
        ratios.append(outputs[1].tolist())
        drifts.append(outputs[2].tolist() if use_kl else [])
    optimizer.update(model, summed)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return Update(
        rows=len(items),
        tokens=n_tokens,
        loss_per_token=loss_total / n_tokens,
        ratios=ratios,
        kl=drifts,
    )
