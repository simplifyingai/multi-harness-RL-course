"""Supervised fine-tuning on rows: the project's scoring, batching and step.

``token_logprobs`` and ``batches`` are the project's
(``mhrl/training/common.py``), and ``train_sft`` is its
``mhrl/training/sft.py::train_sft`` with the file writing and printing moved
out to the caller (a ``on_step`` function receives each step instead):

- a row is read in **one forward pass**: the model sees every token at
  once and gives scores for the next token at every position (teacher
  forcing: the recorded tokens are fed in, never the model's own guesses);
- each reply token's log-probability is read from the scores at the
  position **before** it (the next-token shift);
- the loss of a row is minus the sum of its reply tokens' log-probabilities;
- rows are scored one at a time and their gradients added up, each divided
  by the step's total reply-token count, so the step's update is the
  gradient of the loss per reply token, and every reply token counts the
  same whichever row it is in;
- one AdamW update per step (bias correction on), on the adapter only.

What the project's ``load_policy`` adds for long agent transcripts (a 9 GB
memory limit, attention in blocks, recomputing activations in the backward
pass) is chapter 25's: these rows are a few hundred tokens long. The memory
and cache limits are set here as the project sets them.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from rows import TrainItem

LOGIT_CHUNK = 512  # positions per lm-head chunk: 512 x 128K vocab x 4 B = 256 MB
MEMORY_LIMIT_BYTES = 9 * 1024**3  # the project trainer's share of the budget
CACHE_LIMIT_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class Step:
    """One optimizer step, as the project's metrics.jsonl records it.

    Attributes:
        step: The step's number, from 1.
        epoch: The pass over the rows it belongs to, from 1.
        rows: How many rows it read.
        tokens: How many reply tokens it trained on.
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
    rows = mx.array([p - 1 for p in item.positions])
    pieces = []
    for start in range(0, len(item.positions), LOGIT_CHUNK):
        index = rows[start : start + LOGIT_CHUNK]
        logits = model.model.embed_tokens.as_linear(hidden[index]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = targets[start : start + LOGIT_CHUNK]
        pieces.append(mx.take_along_axis(logp, chosen[:, None], axis=-1)[:, 0])
    return mx.concatenate(pieces)


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


def train_sft(
    model: nn.Module,
    items: list[TrainItem],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    on_step: Callable[[Step], None],
) -> None:
    """Fine-tune the LoRA adapter on rows (the project's loop).

    Args:
        model: A model with LoRA layers applied; only they train.
        items: SFT rows.
        epochs: Passes over the data.
        learning_rate: AdamW learning rate.
        batch_size: Rows per optimizer step.
        on_step: Called after every step with what it did.
    """
    # As mhrl/training/sft.py builds it: weight decay is left at MLX's 0.01.
    optimizer = optim.AdamW(learning_rate=learning_rate, bias_correction=True)

    def loss_fn(item: TrainItem) -> mx.array:
        """Minus the sum of the row's reply-token log-probabilities.

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
                # Divide by the step's reply tokens, so the sum over its rows
                # is the gradient of the loss per reply token.
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


def all_logits(model: nn.Module, ids: list[int]) -> mx.array:
    """Score every token after every position of a row, in float32.

    Args:
        model: The network.
        ids: The row's token ids.

    Returns:
        The scores, shape (length, vocabulary).
    """
    logits = model(mx.array(ids)[None])[0].astype(mx.float32)
    mx.eval(logits)
    return logits


def max_difference(one: mx.array, other: mx.array) -> float:
    """Find the largest difference between two sets of scores.

    Args:
        one: Scores.
        other: Scores of the same shape.

    Returns:
        The largest absolute difference.
    """
    return mx.max(mx.abs(one - other)).item()


def reply_logprob(model: nn.Module, prompt: list[int], reply: list[int]) -> float:
    """Score a whole reply given its prompt: the sum of its tokens' log-probabilities.

    Args:
        model: The network.
        prompt: The prompt's token ids.
        reply: The reply's token ids.

    Returns:
        The reply's log-probability (0 for an empty reply).
    """
    if not reply:
        return 0.0
    ids = prompt + reply
    positions = list(range(len(prompt), len(ids)))
    item = TrainItem(ids, positions, [0.0] * len(positions))
    return token_logprobs(model, item).sum().item()
