"""The chat a question becomes, and the training row a sampled reply becomes.

The chat is chapter 14's, unchanged: a system rule, then the database's
tables and the question, rendered with the model's chat template and ending
with the generation prompt (``<|im_start|>assistant`` and ``<think>``),
exactly what the project's engine shows the model before it replies.

A **sequence** is what the project's capture proxy records for one model
call: ``input_ids`` (the prompt's tokens, then the tokens the model sampled),
a ``loss_mask`` (1 at the sampled tokens) and ``logprobs`` (each sampled
token's log-probability as the engine drew it, 0 over the prompt).
``TrainItem`` and ``item_from_sequence`` are the project's
(``mhrl/training/common.py``), copied unchanged: they turn a sequence into the
positions the trainer scores, their old log-probabilities and the reply's
advantage. ``balanced`` is the project's ``mhrl/cli.py::balanced``, and
``training_order`` the shuffle its ``train_grpo`` starts with.

This module needs only the standard library: the tokenizer is passed in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from database import schema_text
from tasks import DATA_DIR, Task

SYSTEM_ROLE = "system"
USER_ROLE = "user"
# Chapter 14's rule. Like chapter 11's prompt, it asks for the answer alone,
# because the answer is read from the result's first column.
SYSTEM_RULE = (
    "You answer questions about a SQLite database with one SQL query. Its result "
    "must be the answer and nothing else: one column, one row per item if the "
    "answer is a list. Reply with the query alone, after your reasoning."
)
USER_TEMPLATE = "The database has these tables:\n\n{schema}\n\n{question}"
READ = 0  # loss-mask value of a token the model only read
SAMPLED = 1  # loss-mask value of a token the model sampled
PROMPT_LOGPROB = 0.0  # what the proxy records over the prompt's positions


class ChatTokenizer(Protocol):
    """What this module needs from a tokenizer (mlx-lm's, or a test's fake)."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Render a chat with the model's template."""
        ...

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn text into token ids."""
        ...


@dataclass
class TrainItem:
    """One training row (the project's, from ``mhrl/training/common.py``).

    Attributes:
        input_ids: Context and sampled tokens.
        positions: Indices of the sampled (trainable) tokens in ``input_ids``.
        old_logprobs: The engine's logprob for each trainable token.
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


def item_from_sequence(sequence: dict, advantage: float = 1.0) -> TrainItem | None:
    """Turn one captured sequence into a training row (the project's code).

    Args:
        sequence: ``{input_ids, loss_mask, logprobs}`` from the capture proxy.
        advantage: The rollout's advantage.

    Returns:
        The row, or None when it has no trainable tokens.
    """
    ids = sequence["input_ids"]
    mask = sequence["loss_mask"]
    logprobs = sequence.get("logprobs") or [0.0] * len(ids)
    # i > 0: a token is scored from the position before it, and nothing comes
    # before position 0, so the first token can never be a target.
    positions = [i for i, m in enumerate(mask) if m and i > 0]
    if not positions:
        return None
    old = [float(logprobs[i] or 0.0) for i in positions]
    return TrainItem(ids, positions, old, advantage)


def call_sequence(
    prompt: list[int], reply: list[int], logprobs: list[float], mask: list[int]
) -> dict[str, list]:
    """Write one model call as the capture proxy records it.

    Args:
        prompt: The prompt's token ids.
        reply: The reply's token ids.
        logprobs: One old log-probability per reply token.
        mask: One loss-mask value per reply token (1 trains it).

    Returns:
        ``{"input_ids", "loss_mask", "logprobs"}``, each as long as the prompt
        and the reply together.

    Raises:
        ValueError: If the reply, its log-probabilities and its mask differ in
            length.
    """
    if not len(reply) == len(logprobs) == len(mask):
        raise ValueError("a reply needs one log-probability and one mask value a token")
    return {
        "input_ids": prompt + reply,
        "loss_mask": [READ] * len(prompt) + mask,
        "logprobs": [PROMPT_LOGPROB] * len(prompt) + logprobs,
    }


def balanced(tasks: list[Task], limit: int) -> list[Task]:
    """Take up to ``limit`` tasks, round-robin over (database, difficulty).

    A copy of the project's ``mhrl/cli.py::balanced`` (via chapter 11), which
    its ``--limit`` flags use: a plain first 10 would be lopsided.

    Args:
        tasks: Candidate tasks, in file order.
        limit: How many to keep.

    Returns:
        A balanced subset, the same for the same input order.
    """
    if limit >= len(tasks):
        return list(tasks)
    buckets: dict[tuple[str, str], list[Task]] = {}
    for task in tasks:
        buckets.setdefault((task.domain, task.difficulty), []).append(task)
    queues = [list(buckets[key]) for key in sorted(buckets)]
    picked: list[Task] = []
    while len(picked) < limit:
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def training_order(tasks: list[Task], seed: int) -> list[Task]:
    """Shuffle the training questions once, as the project's train_grpo does.

    Args:
        tasks: The training questions, in file order.
        seed: The shuffle's seed.

    Returns:
        The order the steps take them in (cycled if a run needs more).
    """
    order = list(tasks)
    random.Random(seed).shuffle(order)
    return order


def chat_messages(task: Task, folder: Path = DATA_DIR) -> list[dict[str, str]]:
    """Write the chat the model reads: the rule, the tables, the question.

    Args:
        task: The question.
        folder: The data folder holding its database.

    Returns:
        A system message and a user message.
    """
    schema = schema_text(folder / task.db_file)
    question = USER_TEMPLATE.format(schema=schema, question=task.question)
    return [
        {"role": SYSTEM_ROLE, "content": SYSTEM_RULE},
        {"role": USER_ROLE, "content": question},
    ]


def encode(tokenizer: ChatTokenizer, text: str) -> list[int]:
    """Turn text into token ids, adding nothing (the template wrote the markers).

    Args:
        tokenizer: The model's tokenizer.
        text: The text.

    Returns:
        Its token ids.
    """
    return list(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> str:
    """Render a chat into the text the model reads, ready for its reply.

    Args:
        tokenizer: The model's tokenizer.
        messages: The chat so far.

    Returns:
        The text, ending with the generation prompt
        (``<|im_start|>assistant`` and ``<think>``).
    """
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # end with the assistant's turn opening
        tokenize=False,
        preserve_thinking=True,  # keep reasoning, as the project's engine renders
    )


def prompt_ids(
    tokenizer: ChatTokenizer, task: Task, folder: Path = DATA_DIR
) -> list[int]:
    """Render a question's chat into the token ids the model reads.

    Args:
        tokenizer: The model's tokenizer.
        task: The question.
        folder: The data folder.

    Returns:
        The prompt's token ids, ending with ``<think>``.
    """
    return encode(tokenizer, render_prompt(tokenizer, chat_messages(task, folder)))
