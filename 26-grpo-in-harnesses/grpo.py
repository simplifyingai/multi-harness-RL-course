"""One GRPO step's rows and update: the inner part of the project's train_grpo.

From a step's graded rollouts to one AdamW step on the adapter, as
``mhrl/training/grpo.py`` does it:

1. each task's group of rollouts gets advantages from its rewards
   (``rewards.group_advantages``); a group whose rewards are all equal has
   no contrast, teaches nothing, and is skipped (and counted);
2. each rollout's captured rows lose the sampled tokens an earlier row of the
   same rollout already holds (``dedupe_shared_prefixes``, chapter 21), then
   become training rows (``item_from_sequence``, chapter 25's
   ``training.py``) carrying the rollout's advantage; a row over 6,144 tokens
   is skipped (and counted), since longer rows do not fit the memory budget;
3. every row is scored by ``token_logprobs`` and pushed by the clipped loss
   against the log-probabilities the ENGINE recorded when it sampled the
   tokens (chapter 15): ``ratio = exp(new - old)`` per token, and the loss
   ``-min(ratio x A, clip(ratio, 0.8, 1.2) x A)``, summed over the row;
4. each row's gradient is divided by the step's trained-token count and
   added up, then one AdamW step is made (chapter 15's ``grpo_update``).

The ratio is worked out before the update, with the weights that sampled
the rollouts, so it measures only how far the trainer's numbers are from the
engine's: near 1, never exactly (chapter 15: bfloat16 routes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from rewards import group_advantages
from training import MAX_TRAIN_TOKENS, TrainItem, item_from_sequence, token_logprobs

CLIP_EPSILON = 0.2  # the project's Defaults.CLIP_EPSILON: the clip is 0.8 to 1.2


@dataclass
class HarnessRows:
    """What one harness gave a step's update.

    Attributes:
        rows: Training rows kept.
        tokens: Their total length (everything the trainer reads).
        trained: Their trained (sampled) tokens.
        groups: Groups from this harness.
        flat_groups: Of those, groups without contrast.
    """

    rows: int = 0
    tokens: int = 0
    trained: int = 0
    groups: int = 0
    flat_groups: int = 0


@dataclass
class StepRows:
    """A step's training rows, and everything left out of them.

    Attributes:
        items: The rows the update trains on.
        groups: Groups in the step.
        flat_groups: Groups without contrast (skipped).
        skipped_long: Rows over the length cap (skipped).
        by_harness: Per harness, what it gave.
    """

    items: list[TrainItem] = field(default_factory=list)
    groups: int = 0
    flat_groups: int = 0
    skipped_long: int = 0
    by_harness: dict[str, HarnessRows] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        """The step's trained tokens."""
        return sum(item.n_tokens for item in self.items)


@dataclass(frozen=True)
class Update:
    """What one update learned from, and every trained token's ratio.

    Attributes:
        rows: How many rows it trained on.
        tokens: How many sampled tokens those rows hold.
        loss_per_token: The summed loss divided by ``tokens``.
        ratios: Every trained token's importance ratio, before the update.
    """

    rows: int
    tokens: int
    loss_per_token: float
    ratios: list[float]

    @property
    def clipped_share(self) -> float:
        """The share of tokens whose ratio is outside the clip's 0.8 to 1.2."""
        low, high = 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON
        outside = sum(1 for ratio in self.ratios if not low <= ratio <= high)
        return outside / len(self.ratios) if self.ratios else 0.0


def dedupe_shared_prefixes(sequences: list[dict]) -> list[dict]:
    """Mask sampled tokens a rollout's earlier sequences already train.

    The project's ``mhrl/training/common.py`` ``dedupe_shared_prefixes``,
    unchanged (chapter 21). A harness that retries or rewrites its history
    forks the rollout graph, and every root-to-leaf path becomes its own
    sequence. Forks share their early turns, so without this the shared
    sampled tokens would be trained once per path.

    Args:
        sequences: The rollout's trainable sequences, in proxy order.

    Returns:
        Copies whose ``loss_mask`` is zero over any prefix an earlier sequence
        already covered.
    """
    seen: list[list[int]] = []
    out: list[dict] = []
    for sequence in sequences:
        ids = sequence["input_ids"]
        shared = 0
        for earlier in seen:
            limit = min(len(ids), len(earlier))
            length = 0
            while length < limit and ids[length] == earlier[length]:
                length += 1
            shared = max(shared, length)
        mask = list(sequence["loss_mask"])
        for index in range(min(shared, len(mask))):
            mask[index] = 0
        out.append({**sequence, "loss_mask": mask})
        seen.append(ids)
    return out


def step_rows(
    groups: Sequence[Sequence[dict[str, Any]]],
    max_tokens: int = MAX_TRAIN_TOKENS,
) -> StepRows:
    """Turn a step's graded groups into training rows, counting what is left out.

    Args:
        groups: One list of rollout records per task, each record with its
            ``harness``, ``reward`` and ``sequences``.
        max_tokens: Rows longer than this are skipped (and counted).

    Returns:
        The rows, and what each harness gave.
    """
    out = StepRows(groups=len(groups))
    for group in groups:
        harness = str(group[0]["harness"])
        mine = out.by_harness.setdefault(harness, HarnessRows())
        mine.groups += 1
        advantages = group_advantages([float(r["reward"]) for r in group])
        if advantages is None:  # every reward equal: nothing to compare
            out.flat_groups += 1
            mine.flat_groups += 1
            continue
        for record, advantage in zip(group, advantages, strict=True):
            for sequence in dedupe_shared_prefixes(record.get("sequences") or []):
                item = item_from_sequence(sequence, advantage)
                if item is None:  # every sampled token already trained above
                    continue
                if len(item.input_ids) > max_tokens:
                    out.skipped_long += 1
                    continue
                out.items.append(item)
                mine.rows += 1
                mine.tokens += len(item.input_ids)
                mine.trained += item.n_tokens
    return out


def clipped_loss(
    model: nn.Module,
) -> Callable[[TrainItem], tuple[mx.array, mx.array]]:
    """Build the project's clipped loss for this model.

    Args:
        model: The LoRA model; the loss reads its current weights.

    Returns:
        ``loss_fn(item) -> (loss, ratio)``, the project's function.
    """
    eps = CLIP_EPSILON

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


def grpo_update(
    model: nn.Module, optimizer: optim.Optimizer, items: list[TrainItem]
) -> Update:
    """One update from the rows of every group that had contrast.

    Args:
        model: The LoRA model; only the adapter trains.
        optimizer: AdamW, built once for the whole run.
        items: The step's rows (at least one).

    Returns:
        What the update trained on, with every token's ratio.

    Raises:
        ValueError: If there is no row to train on.
    """
    if not items:
        raise ValueError("no rows: a step without contrast makes no update")
    loss_and_grad = nn.value_and_grad(model, clipped_loss(model))
    n_tokens = sum(item.n_tokens for item in items)
    summed, loss_total = None, 0.0
    ratios: list[float] = []
    for item in items:  # one row at a time: only one row's activations
        (loss, ratio), grads = loss_and_grad(item)
        # Divide by the step's trained tokens, so the sum over its rows is
        # the gradient of the loss per token.
        grads = tree_map(lambda g, n=n_tokens: g / n, grads)
        summed = grads if summed is None else tree_map(mx.add, summed, grads)
        mx.eval(summed, loss, ratio)
        loss_total += float(loss)
        ratios.extend(ratio.tolist())
    optimizer.update(model, summed)
    mx.eval(model.trainable_parameters(), optimizer.state)
    return Update(len(items), n_tokens, loss_total / n_tokens, ratios)
