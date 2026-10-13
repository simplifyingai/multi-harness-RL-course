"""What training needs from a captured row, and how it fits in 9 GB.

The project's ``mhrl/training/common.py`` and ``mhrl/training/sft.py``,
copied with their comments, cut to what this chapter runs:

* ``TrainItem`` and ``item_from_sequence``: a captured row is the proxy's
  ``input_ids`` + ``loss_mask`` (+ the engine's ``logprobs``); the trained
  positions are the ones the mask marks 1, the teacher's own tokens.
  Training never re-tokenizes anything;
* ``demo_items``: the rows of the teacher's correct, trainable rollouts,
  with the rows over the length cap counted, never dropped in silence;
* ``chunked_attention`` and ``layer_checkpointing``: what the project's
  ``load_policy`` patches in so long agent transcripts fit (attention in
  512-row blocks inside ``mx.checkpoint``, and every layer's activations
  recomputed in the backward pass instead of kept);
* ``token_logprobs``: logits only at the trained positions, 512 at a time;
  ``full_vocab_logprobs`` is the plain way (every position, the whole
  vocabulary), kept for the memory probe to measure against;
* ``train_sft``: chapter 14's copy of the project's loop (rows one at a time,
  gradients divided by the step's trained-token count, one AdamW step per
  batch, bias correction on).
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm.models import lfm2
from mlx_lm.tuner.trainer import grad_checkpoint

LOGIT_CHUNK = 512  # positions per lm-head chunk: 512 x 128K vocab x 4 B = 256 MB
QUERY_CHUNK = 512  # attention rows per block in the training forward pass
MEMORY_LIMIT_BYTES = 9 * 1024**3  # the trainer's share of the 8-10 GB budget
CACHE_LIMIT_BYTES = 512 * 1024**2
MAX_TRAIN_TOKENS = 6144  # the project's cap (mhrl/cli.py MAX_TRAIN_TOKENS)
SFT_LEARNING_RATE = 5e-5  # 1e-4 over-fit the project's dry run
SFT_BATCH = 8
SFT_EPOCHS = 1

_fused_attention = lfm2.scaled_dot_product_attention  # MLX's kernel, as shipped


@dataclass
class TrainItem:
    """One training row.

    Attributes:
        input_ids: Context and sampled tokens.
        positions: Indices of the sampled (trainable) tokens in ``input_ids``.
        old_logprobs: The engine's logprob for each trainable token (0 for a
            demonstration).
        advantage: Group-relative advantage (1.0 for SFT rows).
    """

    input_ids: list[int]
    positions: list[int]
    old_logprobs: list[float]
    advantage: float = 1.0

    @property
    def n_tokens(self) -> int:
        """Number of trainable tokens."""
        return len(self.positions)


@dataclass(frozen=True)
class DemoRows:
    """The rows a set of demonstrations gives, and what was left out.

    Attributes:
        items: The training rows.
        demos: Demonstrations read.
        usable: Demonstrations that were correct and trainable.
        too_long: Rows skipped for being over the length cap.
        empty: Rows skipped for having no trainable token.
    """

    items: list[TrainItem]
    demos: int
    usable: int
    too_long: int
    empty: int


def item_from_sequence(
    sequence: dict[str, Any], advantage: float = 1.0
) -> TrainItem | None:
    """Turn one captured sequence into a training row.

    Args:
        sequence: ``{input_ids, loss_mask, logprobs}`` from the capture proxy.
        advantage: The rollout's advantage.

    Returns:
        The row, or None when it has no trainable tokens.
    """
    ids = sequence["input_ids"]
    mask = sequence["loss_mask"]
    logprobs = sequence.get("logprobs") or [0.0] * len(ids)
    # Position 0 can never be a target: nothing comes before it to predict it.
    positions = [i for i, m in enumerate(mask) if m and i > 0]
    if not positions:
        return None
    old = [float(logprobs[i] or 0.0) for i in positions]
    return TrainItem(ids, positions, old, advantage)


def demo_items(records: list[dict[str, Any]], max_tokens: int) -> DemoRows:
    """Training rows from the teacher's correct, trainable rollouts.

    The project's ``demo_items``, counting what it skips.

    Args:
        records: Demonstration records (``collect.py``'s, or the project's).
        max_tokens: Rows longer than this are skipped (and counted).

    Returns:
        The rows and the counts.
    """
    items: list[TrainItem] = []
    usable = too_long = empty = 0
    for record in records:
        if not (record.get("correct") and record.get("trainable")):
            continue
        usable += 1
        for sequence in record.get("sequences") or []:
            item = item_from_sequence(sequence)
            if item is None:
                empty += 1
            elif len(item.input_ids) > max_tokens:
                too_long += 1
            else:
                items.append(item)
    return DemoRows(items, len(records), usable, too_long, empty)


def set_memory_limits() -> None:
    """Cap MLX's memory and its cache of freed buffers, as the project does.

    The memory limit is a guideline MLX plans around, not a wall: an
    evaluation that needs more still gets it (the project's fused-attention
    pass peaked at 11.9 GB under this 9 GB limit), so ``memory_probe.py``
    refuses a row it projects over the budget instead of trusting the limit.
    """
    mx.set_memory_limit(MEMORY_LIMIT_BYTES)
    mx.set_cache_limit(CACHE_LIMIT_BYTES)


def chunked_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: object,
    scale: float,
    mask: object,
    sinks: mx.array | None = None,
) -> mx.array:
    """Causal attention computed one block of query rows at a time.

    MLX 0.32's fused attention has no fused backward: its gradient falls back
    to materializing every head's full L x L score matrix (11.9 GB peak at 6K
    tokens, measured). Here each block of ``QUERY_CHUNK`` queries attends to
    the keys up to its own end, inside ``mx.checkpoint``, so the backward pass
    recomputes one block at a time and memory grows linearly, not
    quadratically. Generation (with a cache) keeps the fused kernel.

    Args:
        queries: ``[B, heads, L, D]``.
        keys: ``[B, kv_heads, L, D]``.
        values: ``[B, kv_heads, L, D]``.
        cache: The layer cache (None while training).
        scale: Softmax scale.
        mask: The model's mask (ignored when chunking; causality is rebuilt).
        sinks: Unsupported attention sinks (LFM2 has none).

    Returns:
        ``[B, heads, L, D]``.
    """
    length = queries.shape[2]
    if cache is not None or sinks is not None or length <= QUERY_CHUNK:
        return _fused_attention(queries, keys, values, cache, scale, mask, sinks)

    def block(q: mx.array, k: mx.array, v: mx.array, start: int) -> mx.array:
        """Attend one block of queries to every key up to the block's end.

        Args:
            q: The block's queries.
            k: Keys up to the block's last row.
            v: Values up to the block's last row.
            start: The block's first row.

        Returns:
            The block's output.
        """
        rows = mx.arange(start, start + q.shape[2])[:, None]
        cols = mx.arange(k.shape[2])[None, :]
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=cols <= rows
        )

    pieces = []
    for start in range(0, length, QUERY_CHUNK):
        end = min(start + QUERY_CHUNK, length)
        run = mx.checkpoint(lambda q, k, v, s=start: block(q, k, v, s))
        pieces.append(
            run(queries[:, :, start:end], keys[:, :, :end], values[:, :, :end])
        )
    return mx.concatenate(pieces, axis=2)


@contextlib.contextmanager
def attention(chunked: bool) -> Iterator[None]:
    """Use chunked attention (the project's training setting) or MLX's kernel.

    The project's ``load_policy`` patches ``lfm2.scaled_dot_product_attention``
    once for good; this puts the original back afterwards, so the memory
    probe can measure both in one process.

    Args:
        chunked: True for ``chunked_attention``, False for the fused kernel.

    Yields:
        Nothing; the setting holds inside the block.
    """
    saved = lfm2.scaled_dot_product_attention
    lfm2.scaled_dot_product_attention = (
        chunked_attention if chunked else _fused_attention
    )
    try:
        yield
    finally:
        lfm2.scaled_dot_product_attention = saved


@contextlib.contextmanager
def layer_checkpointing(model: nn.Module, on: bool) -> Iterator[None]:
    """Recompute each layer's activations in the backward pass, or keep them.

    mlx-lm's ``grad_checkpoint`` rewrites the layer CLASS's ``__call__``, so
    every one of LFM2's 30 layers (one class) is checkpointed; this puts the
    original back afterwards.

    Args:
        model: An mlx-lm model.
        on: True to checkpoint, False to leave the layers as shipped.

    Yields:
        Nothing; the setting holds inside the block.
    """
    layer_class = type(model.layers[0])
    saved = layer_class.__call__
    if on:
        grad_checkpoint(model.layers[0])
    try:
        yield
    finally:
        layer_class.__call__ = saved


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
    rows = mx.array([p - 1 for p in item.positions])
    pieces = []
    for start in range(0, len(item.positions), LOGIT_CHUNK):
        index = rows[start : start + LOGIT_CHUNK]
        logits = model.model.embed_tokens.as_linear(hidden[index]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = targets[start : start + LOGIT_CHUNK]
        pieces.append(mx.take_along_axis(logp, chosen[:, None], axis=-1)[:, 0])
    return mx.concatenate(pieces)


def full_vocab_logprobs(model: nn.Module, item: TrainItem) -> mx.array:
    """The same log-probabilities, the plain way: logits everywhere, then pick.

    What ``model(tokens)`` gives: scores over the whole vocabulary at every
    position of the row, most of them never read. The memory probe measures
    it against ``token_logprobs``.

    Args:
        model: The LoRA model.
        item: The training row.

    Returns:
        Array of shape ``[n_tokens]``, equal to ``token_logprobs``'s.
    """
    tokens = mx.array(item.input_ids)[None]
    hidden = model.model(tokens[:, :-1])[0]
    logits = model.model.embed_tokens.as_linear(hidden).astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    rows = mx.array([p - 1 for p in item.positions])
    targets = mx.array([item.input_ids[p] for p in item.positions])
    return mx.take_along_axis(logp[rows], targets[:, None], axis=-1)[:, 0]


def batches(items: list[TrainItem], size: int, seed: int) -> Iterator[list[TrainItem]]:
    """Yield shuffled mini-batches (the project's: one shuffle per epoch).

    Args:
        items: All rows.
        size: Rows per batch.
        seed: Shuffle seed.

    Yields:
        Lists of rows.
    """
    order = list(items)
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield order[start : start + size]


def steps_per_epoch(rows: int, batch_size: int) -> int:
    """Count the optimizer steps one pass over the rows takes.

    Args:
        rows: How many rows there are.
        batch_size: Rows per step.

    Returns:
        ``ceil(rows / batch_size)``: the last step may hold fewer rows.
    """
    return -(-rows // batch_size)


@dataclass(frozen=True)
class Step:
    """One optimizer step, as the project's metrics.jsonl records it.

    Attributes:
        step: The step's number, from 1.
        epoch: The pass over the rows it belongs to, from 1.
        rows: How many rows it read.
        tokens: How many trained tokens it learned from.
        loss_per_token: Minus the summed log-probability of those tokens,
            divided by their count, before the update.
        seconds: How long it took (varies from run to run).
        peak_bytes: MLX's peak memory so far.
    """

    step: int
    epoch: int
    rows: int
    tokens: int
    loss_per_token: float
    seconds: float
    peak_bytes: int


def train_sft(
    model: nn.Module,
    items: list[TrainItem],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    on_step: Callable[[Step], None],
    bias_correction: bool = True,
) -> None:
    """Fine-tune the LoRA adapter on rows (the project's loop).

    Args:
        model: A model with LoRA layers applied; only they train.
        items: SFT rows.
        epochs: Passes over the data.
        learning_rate: AdamW learning rate.
        batch_size: Rows per optimizer step.
        on_step: Called after every step with what it did.
        bias_correction: AdamW's bias correction, on in the project's code
            today. Its recorded SFT run trained before the code review that
            turned it on: without it, the first steps are several times
            larger (``train_sft.py --no-bias-correction``).
    """
    # As mhrl/training/sft.py builds it: weight decay is left at MLX's 0.01.
    optimizer = optim.AdamW(
        learning_rate=learning_rate, bias_correction=bias_correction
    )

    def loss_fn(item: TrainItem) -> mx.array:
        """Minus the sum of the row's trained-token log-probabilities.

        Args:
            item: One row.

        Returns:
            The row's loss, one number.
        """
        return -token_logprobs(model, item).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    step = 0
    mx.reset_peak_memory()
    for epoch in range(epochs):
        for batch in batches(items, batch_size, seed=epoch):
            started = time.perf_counter()
            total = sum(item.n_tokens for item in batch)
            summed = None
            batch_loss = 0.0
            for item in batch:  # one row at a time: only one row's activations
                loss, grads = loss_and_grad(item)
                # Divide by the step's trained tokens, so the sum over its rows
                # is the gradient of the loss per trained token.
                grads = tree_map(lambda g, n=total: g / n, grads)
                summed = grads if summed is None else tree_map(mx.add, summed, grads)
                mx.eval(summed, loss)
                batch_loss += float(loss)
            optimizer.update(model, summed)
            mx.eval(model.trainable_parameters(), optimizer.state)
            step += 1
            on_step(
                Step(
                    step=step,
                    epoch=epoch + 1,
                    rows=len(batch),
                    tokens=total,
                    loss_per_token=batch_loss / max(total, 1),
                    seconds=time.perf_counter() - started,
                    peak_bytes=mx.get_peak_memory(),
                )
            )
