"""Supervised fine-tuning (SFT) on the toy: the project's two steps, small.

``mhrl/training/sft.py`` does SFT in two steps, and so does this file:

1. ``collect_demonstrations``: the answer-key teacher plays the agent, each
   demonstration in its own folder, and every call is recorded.
2. ``train_sft``: minimise ``-log p(teacher's action)`` at every call the
   teacher made, in batches, for a number of passes over the data.

What is trained is exactly what the project trains. A demonstration is a
sequence of pieces: the question, then each call's action (the teacher's)
and observation (the environment's). Its ``loss_mask`` is 1 on the teacher's
pieces and 0 on everything else, as the project's capture proxy marks the
teacher's tokens. ``item_from_sequence`` keeps the masked positions only, so
the student is scored on the teacher's actions and never on the
environment's replies: those are what it reads, not what it writes.

``batches`` is the project's line for line, and ``train_sft`` follows its
loop line by line: each row's loss is summed over its calls, every row's
gradient is divided by the calls in the whole batch, and one step is taken
per batch. The project's optimiser is AdamW (chapter 13); here it is chapter
3's plain gradient descent, and NumPy has no automatic gradient, so
``loss_gradient`` works the slope out by hand (a test checks it against
slopes measured by moving each weight a hair).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from environment import ACTIONS, BUDGET, Call, MiniAnalyst, State, fresh_folder
from policy import action_scores, gradient_step, log_prob, weight_gradient
from questions import Question
from tasks import CourseError
from teacher import path_name, teacher_actions

TEACHER = 1  # loss_mask: a piece the teacher wrote, which is trained
CONTEXT = 0  # loss_mask: the question or an observation, never trained
PROMPT = "question"  # what the first piece of every sequence is
STAGE = "sft"
DEMO_FOLDER = "demo-{:03d}"


class DemonstrationFailed(CourseError):
    """Raised when a teacher's demonstration stops without a verdict (a bug)."""


@dataclass(frozen=True)
class Piece:
    """One piece of a demonstration's sequence.

    Attributes:
        kind: ``"question"``, an action (``"LOOK"``, ...) or ``"observation"``.
        text: The question, the action's name, or what the environment said.
        situation: For an action, the features the student would see just
            before it (its "context"); None for every other piece.
    """

    kind: str
    text: str
    situation: np.ndarray | None = None


@dataclass(frozen=True)
class DemoSequence:
    """A demonstration as the trainer sees it: pieces and their loss mask.

    The project's capture proxy gives ``input_ids`` + ``loss_mask``; this is
    the same shape with pieces in place of tokens.

    Attributes:
        pieces: The question, then each call's action and observation.
        loss_mask: 1 for the teacher's pieces, 0 for the rest.
    """

    pieces: tuple[Piece, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class Demonstration:
    """One teacher run: the project's ``RolloutRecord``, cut down.

    Attributes:
        number: Its place in the collection, from 1.
        question: The question demonstrated.
        path: Which of the teacher's paths it took.
        calls: Every call: the action and what came back.
        sequence: The pieces and their loss mask.
        correct: Whether the grader accepted its answer.txt.
        reason: The grader's reason (``"correct"``, or why not).
    """

    number: int
    question: Question
    path: str
    calls: tuple[Call, ...]
    sequence: DemoSequence
    correct: bool
    reason: str


@dataclass(frozen=True)
class TrainItem:
    """One training row: what the student saw at each trained piece, and the action.

    The project's ``TrainItem`` keeps the token ids and the positions of the
    trainable tokens; the toy keeps what those positions come to.

    Attributes:
        situations: The features before each of the teacher's actions.
        choices: Each action, as an index into ``ACTIONS``.
    """

    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]

    @property
    def n_calls(self) -> int:
        """Number of trained pieces (the project's ``n_tokens``)."""
        return len(self.choices)


@dataclass(frozen=True)
class StepLog:
    """One SFT step, as metrics.jsonl records it.

    Attributes:
        epoch: The pass over the data, from 0 (the project counts from 0).
        step: The step, from 1.
        loss_per_call: The batch's loss divided by its trained calls.
        calls: How many of the teacher's calls the batch held.
        rows: How many demonstrations the batch held.
    """

    epoch: int
    step: int
    loss_per_call: float
    calls: int
    rows: int


def play_teacher(
    question: Question,
    teacher: str,
    number: int,
    parent: Path,
    describe: Callable[[State], np.ndarray],
) -> Demonstration:
    """Let the teacher answer one question in a new folder, and record it.

    Args:
        question: The question to demonstrate.
        teacher: ``"varied"``, ``"fixed-length"`` or ``"never-checks"``.
        number: The demonstration's number, from 1 (names its folder).
        parent: The run's folder.
        describe: Turns a state into the student's features.

    Returns:
        The demonstration, graded.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        DemonstrationFailed: If it stopped without a verdict (never expected).
        CourseError: If the teacher sent an action the environment refuses
            (never expected: its paths use only the environment's actions).
    """
    pieces = [Piece(PROMPT, question.task.question)]
    with fresh_folder(parent, DEMO_FOLDER.format(number)) as folder:
        env = MiniAnalyst(question, folder)
        for action in teacher_actions(question, teacher):
            if env.done:  # never expected: every path fits the budget
                break
            situation = describe(env.state)
            observation = env.step(action)
            pieces.append(Piece(action, action, situation))
            pieces.append(Piece("observation", observation))
    if env.grade is None:
        raise DemonstrationFailed(f"demonstration {number} has no verdict")
    mask = tuple(TEACHER if p.situation is not None else CONTEXT for p in pieces)
    return Demonstration(
        number=number,
        question=question,
        path=path_name(question, teacher),
        calls=tuple(env.calls),
        sequence=DemoSequence(tuple(pieces), mask),
        correct=env.grade.correct,
        reason=env.grade.reason,
    )


def collect_demonstrations(
    questions: Sequence[Question],
    teacher: str,
    count: int,
    seed: int,
    parent: Path,
    describe: Callable[[State], np.ndarray],
) -> list[Demonstration]:
    """Run the teacher on the questions, in a shuffled order, ``count`` times.

    Args:
        questions: The questions it demonstrates.
        teacher: Which teacher.
        count: How many demonstrations to write.
        seed: Shuffles the order the questions come in.
        parent: The run's folder.
        describe: Turns a state into the student's features.

    Returns:
        Every demonstration (the trainer keeps only the correct ones).

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        DemonstrationFailed: If one stopped without a verdict.
        CourseError: If the teacher sent an action the environment refuses.
    """
    order = list(questions)
    random.Random(seed).shuffle(order)
    return [
        play_teacher(order[index % len(order)], teacher, index + 1, parent, describe)
        for index in range(count)
    ]


def item_from_sequence(sequence: DemoSequence) -> TrainItem | None:
    """Turn one demonstration's sequence into a training row.

    The project's ``item_from_sequence``, shape for shape: the trained
    positions are the ones the mask marks (and never position 0, which has
    nothing before it to predict it from).

    Args:
        sequence: The pieces and their loss mask.

    Returns:
        The row, or None when it has no trained pieces.
    """
    positions = [i for i, m in enumerate(sequence.loss_mask) if m and i > 0]
    if not positions:
        return None
    pieces = [sequence.pieces[i] for i in positions]
    # play_teacher marks a piece 1 exactly when it is one of the teacher's
    # actions, and gives every such piece its situation, so the two tuples
    # below always have one entry per trained piece.
    situations = tuple(p.situation for p in pieces if p.situation is not None)
    choices = tuple(ACTIONS.index(p.kind) for p in pieces)
    return TrainItem(situations, choices)


def demo_items(
    records: Sequence[Demonstration], max_calls: int = BUDGET
) -> list[TrainItem]:
    """Training rows from the teacher's correct demonstrations.

    Args:
        records: Demonstrations.
        max_calls: Rows longer than this are skipped (and counted by the caller).

    Returns:
        SFT rows.
    """
    items: list[TrainItem] = []
    for record in records:
        if not record.correct:
            continue
        item = item_from_sequence(record.sequence)
        if item and item.n_calls <= max_calls:
            items.append(item)
    return items


def batches(items: list[TrainItem], size: int, seed: int) -> Iterator[list[TrainItem]]:
    """Yield shuffled mini-batches.

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


def call_logprobs(weights: np.ndarray, item: TrainItem) -> np.ndarray:
    """``log p`` of each of the teacher's actions under the student's weights.

    The toy's ``token_logprobs``: one call is one action, where the real
    model writes many tokens per call.

    Args:
        weights: The student's weights.
        item: The row.

    Returns:
        One log-probability per trained call.
    """
    return np.array(
        [
            log_prob(action_scores(weights, situation), choice)
            for situation, choice in zip(item.situations, item.choices, strict=True)
        ]
    )


def loss_fn(weights: np.ndarray, item: TrainItem) -> float:
    """One row's loss: minus the summed log p of the teacher's actions.

    ``mhrl/training/sft.py``'s ``loss_fn`` with the toy's ``call_logprobs``
    in place of ``token_logprobs``. An action the student already gives
    p 0.99 costs 0.01; one it gives p 0.14 costs 1.97.

    Args:
        weights: The student's weights.
        item: The row.

    Returns:
        The loss, summed over the row's calls.
    """
    return -call_logprobs(weights, item).sum()


def loss_gradient(weights: np.ndarray, item: TrainItem) -> np.ndarray:
    """Work out the slope of one row's loss for every weight, by hand.

    The loss is ``-sum of log p``, so its slope is minus the sum of each
    call's ``weight_gradient`` (chapter 4's slope of ``log p``).

    Args:
        weights: The student's weights.
        item: The row.

    Returns:
        One slope per weight, shaped like the weights.
    """
    total = sum(
        weight_gradient(situation, action_scores(weights, situation), choice)
        for situation, choice in zip(item.situations, item.choices, strict=True)
    )
    return -np.asarray(total)


def train_sft(
    items: list[TrainItem],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    init_weights: np.ndarray,
    on_step: Callable[[StepLog], None] | None = None,
) -> tuple[np.ndarray, list[StepLog]]:
    """Fine-tune the student's weights on demonstration rows.

    Args:
        items: SFT rows.
        epochs: Passes over the data.
        learning_rate: How big each gradient-descent step is.
        batch_size: Rows per step.
        init_weights: The weights to start from (all 0 for a new student).
        on_step: Called after every step with its log (to print it).

    Returns:
        The trained weights, and one log per step.
    """
    weights = np.array(init_weights, dtype=float)
    logs: list[StepLog] = []
    step = 0
    for epoch in range(epochs):
        # As in the project, the epoch seeds the shuffle, so the batches of a
        # pass come out the same whatever the run's seed (which only sets the
        # order the demonstrations were written in).
        for batch in batches(items, batch_size, seed=epoch):
            total = sum(item.n_calls for item in batch)
            summed = None
            batch_loss = 0.0
            for item in batch:
                loss, grads = loss_fn(weights, item), loss_gradient(weights, item)
                grads = grads / total
                summed = grads if summed is None else summed + grads
                batch_loss += float(loss)
            if summed is not None:
                weights = gradient_step(weights, summed, learning_rate)
            step += 1
            log = StepLog(epoch, step, batch_loss / max(total, 1), total, len(batch))
            logs.append(log)
            if on_step is not None:
                on_step(log)
    return weights, logs
