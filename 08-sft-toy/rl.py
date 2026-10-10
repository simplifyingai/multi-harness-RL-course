"""RL after SFT: chapter 6's GRPO, started from any weights.

The project's pipeline trains in stages: SFT first, then GRPO started from
the SFT adapter (``python -m mhrl.cli grpo --init <sft adapter>``). This is
chapter 6's GRPO with its default settings, on the student: each step takes
2 questions, runs 4 rollouts of each with the same weights, scores each
rollout against its group (``group_advantages``), and makes one update with
the clipped loss (``loss_fn``) on a fresh batch, so every ratio is 1. Only
the starting weights differ: all 0 (from scratch) or the imitator's.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from environment import State
from grpo import GROUP_SIZE, QUESTIONS_PER_STEP, Trained, group_advantages, step_update
from policy import gradient_step, pick
from questions import Question
from rollouts import Rollout, run_rollout

STAGE = "grpo"
PICK_STREAM = 2  # the random stream for RL's picks (the test uses another)
GROUP_FOLDER = "{}-step-{:03d}-group-{}"


@dataclass(frozen=True)
class Group:
    """One question's group: its rollouts and their advantages.

    Attributes:
        question: The question all its rollouts answered.
        rollouts: The rollouts, in order.
        advantages: One advantage per rollout, or None when all rewards tied.
    """

    question: Question
    rollouts: tuple[Rollout, ...]
    advantages: tuple[float, ...] | None

    @property
    def rewards(self) -> list[int]:
        """The rollouts' rewards, in order."""
        return [rollout.reward for rollout in self.rollouts]


@dataclass(frozen=True)
class RlStep:
    """One GRPO step: its groups and how many calls the update learned from.

    Attributes:
        start: Where the weights started: ``"scratch"`` or ``"imitator"``.
        number: The step's number, from 1.
        groups: Its groups, in order.
        calls: How many calls the update learned from.
    """

    start: str
    number: int
    groups: tuple[Group, ...]
    calls: int

    @property
    def rollouts(self) -> list[Rollout]:
        """Every rollout of the step, group by group."""
        return [rollout for group in self.groups for rollout in group.rollouts]

    @property
    def no_signal(self) -> int:
        """How many of the step's groups had no contrast."""
        return sum(group.advantages is None for group in self.groups)


def run_group(
    weights: np.ndarray,
    question: Question,
    name: str,
    parent: Path,
    rng: np.random.Generator,
    describe: Callable[[State], np.ndarray],
) -> Group:
    """Run one question's group of rollouts, all with the same weights.

    Args:
        weights: The weights every rollout acts with.
        question: The question.
        name: A prefix for the rollout folders, e.g. ``"imitator-step-001-group-1"``.
        parent: The run's folder.
        rng: The random generator for the picks.
        describe: Turns a state into the student's features.

    Returns:
        The group, with None for advantages when its rewards tie.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    rollouts = tuple(
        run_rollout(
            weights,
            question,
            parent,
            f"{name}-rollout-{index}",
            lambda row: pick(row, rng),
            describe,
        )
        for index in range(1, GROUP_SIZE + 1)
    )
    advantages = group_advantages([r.reward for r in rollouts])
    return Group(question, rollouts, None if advantages is None else tuple(advantages))


def trained_rollouts(groups: Sequence[Group]) -> list[Trained]:
    """Collect what a step learns from: every rollout of a group with contrast.

    Args:
        groups: The step's groups.

    Returns:
        One item per rollout learned from, with the log p recorded at sampling.
    """
    return [
        Trained(r.situations, r.choices, r.log_probs, advantage)
        for group in groups
        if group.advantages is not None
        for r, advantage in zip(group.rollouts, group.advantages, strict=True)
    ]


def train_grpo(
    weights: np.ndarray,
    questions: Sequence[Question],
    start: str,
    steps: int,
    lr: float,
    seed: int,
    parent: Path,
    describe: Callable[[State], np.ndarray],
    on_step: Callable[[RlStep], None] | None = None,
) -> tuple[np.ndarray, list[RlStep]]:
    """Run GRPO from the given weights: sample, score, update, again.

    Both starts use the same seed, so they get the same questions in the
    same order and the same random stream for their picks.

    Args:
        weights: Where to start: all 0, or the imitator's weights.
        questions: The questions to train on.
        start: ``"scratch"`` or ``"imitator"``, for the logs.
        steps: How many steps.
        lr: The learning rate.
        seed: Seeds the question order and the picks.
        parent: The run's folder.
        describe: Turns a state into the student's features.
        on_step: Called after every step (to print it).

    Returns:
        The trained weights and every step.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    order = list(questions)
    random.Random(seed).shuffle(order)
    rng = np.random.default_rng([seed, PICK_STREAM])
    weights = np.array(weights, dtype=float)
    done: list[RlStep] = []
    cursor = 0
    for number in range(1, steps + 1):
        groups = []
        for place in range(1, QUESTIONS_PER_STEP + 1):
            question = order[cursor % len(order)]
            cursor += 1
            name = GROUP_FOLDER.format(start, number, place)
            groups.append(run_group(weights, question, name, parent, rng, describe))
        update = step_update(trained_rollouts(groups), weights)
        weights = gradient_step(weights, update.gradient, lr)
        step = RlStep(start, number, tuple(groups), update.calls)
        done.append(step)
        if on_step is not None:
            on_step(step)
    return weights, done
