"""One rollout of the student: chapter 6's, with the student's features.

The student acts the way chapters 4 to 7's agent acted: before each call it
describes its situation with its features, scores every action, and picks
one, at random by the probabilities while training (temperature 1) or the
top score in a greedy test. Every rollout gets its own new folder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from environment import (
    ACTIONS,
    CHECK,
    FINISH,
    Call,
    MiniAnalyst,
    NotGraded,
    State,
    fresh_folder,
)
from grader import Grade
from policy import action_scores, log_prob
from questions import Question


@dataclass(frozen=True)
class Rollout:
    """One attempt at one question, from its first call to its reward.

    Attributes:
        question: The question answered.
        folder: The name of the rollout's own folder.
        calls: Every call: the action and what came back.
        situations: The features the student saw before each call.
        choices: The action taken at each call (an index into the actions).
        log_probs: Each action's log-probability when it was picked.
        stop: Why the rollout stopped: ``"FINISH"`` or ``"budget used"``.
        verdict: The grader's verdict on answer.txt.
    """

    question: Question
    folder: str
    calls: tuple[Call, ...]
    situations: tuple[np.ndarray, ...]
    choices: tuple[int, ...]
    log_probs: tuple[float, ...]
    stop: str
    verdict: Grade

    @property
    def reward(self) -> int:
        """The rollout's one reward: 1 for a correct answer.txt, else 0."""
        return self.verdict.reward

    @property
    def checked(self) -> bool:
        """Whether the rollout read answer.txt back at least once."""
        return any(call.action == CHECK for call in self.calls)

    @property
    def finished_unsaved(self) -> bool:
        """Whether it said FINISH with no answer.txt written."""
        return self.calls[-1].action == FINISH and self.verdict.submitted is None


def run_rollout(
    weights: np.ndarray,
    question: Question,
    parent: Path,
    name: str,
    choose: Callable[[np.ndarray], int],
    describe: Callable[[State], np.ndarray],
) -> Rollout:
    """Run one rollout in its own new folder, call by call, until it stops.

    Args:
        weights: The student's weights.
        question: The question to answer.
        parent: The run's folder; the rollout's folder is made inside it.
        name: The rollout folder's name.
        choose: Turns action scores into an action: a random pick, or the
            top score in a greedy test.
        describe: Turns a state into the student's features.

    Returns:
        Everything that happened, with the grader's verdict.

    Raises:
        OSError: If the folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If the rollout stopped without a verdict (never expected).
    """
    situations, choices, log_probs = [], [], []
    with fresh_folder(parent, name) as folder:
        env = MiniAnalyst(question, folder)
        while not env.done:
            situation = describe(env.state)
            row = action_scores(weights, situation)
            choice = choose(row)
            env.step(ACTIONS[choice])
            situations.append(situation)
            choices.append(choice)
            log_probs.append(log_prob(row, choice))
    # A stopped rollout always has both; checking says so to a type checker too.
    if env.grade is None or env.state.stop is None:
        raise NotGraded(f"{name} stopped without a verdict")
    return Rollout(
        question=question,
        folder=name,
        calls=tuple(env.calls),
        situations=tuple(situations),
        choices=tuple(choices),
        log_probs=tuple(log_probs),
        stop=env.state.stop,
        verdict=env.grade,
    )
