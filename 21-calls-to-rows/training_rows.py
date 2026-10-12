# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""What the project does with a rollout document: count it, then cut training rows.

The capture proxy hands back one document per rollout (chapter 20). The
project reads four things from it, and these are its functions, copied
from ``mhrl/rollout.py`` and ``mhrl/training/common.py`` with the
standard library only:

* ``count_calls``: the agent's model calls. Turns the proxy marked
  discarded are left out, and when any turn offered tools, only the turns
  that offered tools count (a side call, such as a title, offers none);
* ``n_sampled``: the tokens the model generated, over every turn that was
  not discarded (side calls included: the project sums them all);
* ``trainable_sequences``: the rows ``run_rollout`` keeps, the ones the
  proxy marked ``trainable``, as ``input_ids``, ``loss_mask`` and
  ``logprobs``;
* ``dedupe_shared_prefixes``: when a rollout forks, its rows share their
  first tokens. Without this, the model's tokens in that shared part would
  be trained once per row. It zeroes the mask over any prefix an earlier
  row already covered;
* ``item_from_sequence``: one row as the trainer reads it: the positions
  of the trained tokens and their recorded log-probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainItem:
    """One training row, as the project's trainer reads it.

    Attributes:
        input_ids: Context and sampled tokens.
        positions: Indices of the trained tokens in ``input_ids``.
        old_logprobs: The engine's log-probability for each trained token.
        advantage: The rollout's group-relative advantage (1.0 for SFT).
    """

    input_ids: list[int]
    positions: list[int]
    old_logprobs: list[float]
    advantage: float = 1.0

    @property
    def n_tokens(self) -> int:
        """Number of trained tokens."""
        return len(self.positions)


def count_calls(document: dict[str, Any]) -> int:
    """Count the agent's model calls (auxiliary calls such as titles excluded).

    The project's ``mhrl/rollout.py`` ``count_calls``, unchanged.

    Args:
        document: The proxy's rollout document.

    Returns:
        Number of model calls that offered the model tools or were agent turns.
    """
    turns = document.get("turns") or []
    agent_turns = [t for t in turns if not t.get("discarded")]
    with_tools = [t for t in agent_turns if t.get("n_tools")]
    return len(with_tools) if with_tools else len(agent_turns)


def n_sampled(document: dict[str, Any]) -> int:
    """Count the tokens the model generated, as ``run_rollout`` does.

    Args:
        document: The proxy's rollout document.

    Returns:
        The sum of ``n_sampled`` over the turns that were not discarded.
    """
    return sum(
        int(turn.get("n_sampled") or 0)
        for turn in document.get("turns") or []
        if not turn.get("discarded")
    )


def trainable_sequences(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the rows ``run_rollout`` keeps: the trainable ones, three fields each.

    Args:
        document: The proxy's rollout document.

    Returns:
        ``{input_ids, loss_mask, logprobs}`` per trainable row, in proxy order.
    """
    return [
        {
            "input_ids": seq["input_ids"],
            "loss_mask": seq["loss_mask"],
            "logprobs": seq["logprobs"],
        }
        for seq in document.get("sequences") or []
        if seq.get("trainable")
    ]


def dedupe_shared_prefixes(sequences: list[dict]) -> list[dict]:
    """Mask sampled tokens a rollout's earlier sequences already train.

    The project's ``mhrl/training/common.py`` ``dedupe_shared_prefixes``,
    unchanged. A harness that retries or rewrites its history forks the
    rollout graph, and every root-to-leaf path becomes its own sequence.
    Forks share their early turns, so without this the shared sampled tokens
    would be trained once per path.

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


def item_from_sequence(sequence: dict, advantage: float = 1.0) -> TrainItem | None:
    """Turn one captured sequence into a training row.

    The project's ``mhrl/training/common.py`` ``item_from_sequence``,
    unchanged. Position 0 is never trained: no token comes before it to
    predict it from.

    Args:
        sequence: ``{input_ids, loss_mask, logprobs}`` from the capture proxy.
        advantage: The rollout's advantage.

    Returns:
        The row, or None when it has no trainable tokens.
    """
    ids = sequence["input_ids"]
    mask = sequence["loss_mask"]
    logprobs = sequence.get("logprobs") or [0.0] * len(ids)
    positions = [i for i, m in enumerate(mask) if m and i > 0]
    if not positions:
        return None
    old = [float(logprobs[i] or 0.0) for i in positions]
    return TrainItem(ids, positions, old, advantage)


def trained_tokens(sequences: list[dict]) -> int:
    """Count the tokens a list of rows trains (mask 1).

    Args:
        sequences: Rows with a ``loss_mask``.

    Returns:
        The sum of their masks.
    """
    return sum(sum(sequence["loss_mask"]) for sequence in sequences)


def shared_prefix(first: list[int], second: list[int]) -> int:
    """Count how many leading tokens two sequences share.

    Args:
        first: One token sequence.
        second: Another.

    Returns:
        The length of their common prefix.
    """
    limit = min(len(first), len(second))
    length = 0
    while length < limit and first[length] == second[length]:
        length += 1
    return length
