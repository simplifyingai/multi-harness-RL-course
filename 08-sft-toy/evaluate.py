"""The test: how the student does on questions it never trained on.

Two ways, level by level, on the held-out questions (cities no demonstration
used):

* greedy: always the most likely action, as the project evaluates (one
  rollout per question, the same every time);
* sampled: at temperature 1, as RL samples, 20 rollouts per question, which
  shows how likely each habit is and not just the top choice.

The sampled rollouts also count the habits this chapter is about: how often
the student checks answer.txt, and how often it says FINISH with nothing
saved, and on which call.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from environment import State
from policy import greedy, pick
from questions import LEVELS, Question
from rollouts import Rollout, run_rollout

SAMPLES = 20  # sampled rollouts per held-out question
TEST_STREAM = 1  # the random stream for the test's picks
GREEDY_FOLDER = "test-greedy-{:02d}"
SAMPLED_FOLDER = "test-sampled-{:02d}-{:02d}"
TRACE_CITY = "Lisbon"  # the city of the question each level traces


@dataclass(frozen=True)
class LevelResult:
    """How the student did on one level's held-out questions.

    Attributes:
        level: ``"easy"``, ``"medium"`` or ``"hard"``.
        questions: How many held-out questions the level has.
        greedy_right: How many the greedy student answered right.
        greedy_checked: How many greedy rollouts read answer.txt back.
        greedy_unsaved: Greedy rollouts that said FINISH with no answer.txt,
            counted by the call it came on.
        sampled: How many sampled rollouts there were.
        sampled_right: How many of them were right.
        checked: How many read answer.txt back at least once.
        unsaved: How many said FINISH with no answer.txt, by call number.
        calls: Calls made by the sampled rollouts, added up.
        trace: The greedy rollout on the level's traced question.
    """

    level: str
    questions: int
    greedy_right: int
    greedy_checked: int
    greedy_unsaved: Counter[int]
    sampled: int
    sampled_right: int
    checked: int
    unsaved: Counter[int]
    calls: int
    trace: Rollout

    @property
    def unsaved_total(self) -> int:
        """How many sampled rollouts said FINISH with nothing saved."""
        return sum(self.unsaved.values())


def traced_question(questions: Sequence[Question], level: str) -> Question:
    """Pick the question a level's greedy rollout is printed for.

    Args:
        questions: The held-out questions.
        level: The level.

    Returns:
        The level's first question about TRACE_CITY (every level has one;
        a test checks it).

    Raises:
        StopIteration: If the level has no question about TRACE_CITY.
    """
    return next(q for q in questions if q.level == level and q.city == TRACE_CITY)


def unsaved_by_call(rollouts: Sequence[Rollout]) -> Counter[int]:
    """Count the rollouts that said FINISH with nothing saved, by call number.

    Args:
        rollouts: Any rollouts.

    Returns:
        How many such rollouts stopped on each call (FINISH is the last call).
    """
    return Counter(len(r.calls) for r in rollouts if r.finished_unsaved)


def rollout_record(rollout: Rollout, kind: str, weights_name: str) -> dict[str, Any]:
    """Describe one test rollout for the rollout log.

    Args:
        rollout: The rollout.
        kind: ``"greedy"`` or ``"sampled"``.
        weights_name: Which weights acted, e.g. ``"after sft"``.

    Returns:
        A JSON-ready record.
    """
    return {
        "test": kind,
        "weights": weights_name,
        "question": rollout.question.task.question,
        "level": rollout.question.level,
        "calls": [[call.action, call.observation] for call in rollout.calls],
        "answer": rollout.verdict.submitted,
        "reward": rollout.reward,
    }


def held_out_test(
    weights: np.ndarray,
    held_out: Sequence[Question],
    parent: Path,
    seed: int,
    describe: Callable[[State], np.ndarray],
    weights_name: str,
    log: list[dict[str, Any]],
) -> tuple[LevelResult, ...]:
    """Test the student greedily and by sampling on every held-out question.

    Args:
        weights: The student's weights.
        held_out: The held-out questions.
        parent: The run's folder.
        seed: Seeds the sampled test's picks.
        describe: Turns a state into the student's features.
        weights_name: Which weights these are, for the log.
        log: The rollout log; every test rollout is appended.

    Returns:
        One result per level.

    Raises:
        OSError: If a folder cannot be made or a file written.
        sqlite3.Error: If a query fails on the real database.
        NotGraded: If a rollout stopped without a verdict (never expected).
    """
    rng = np.random.default_rng([seed, TEST_STREAM])
    results = []
    for level in LEVELS:
        mine = [q for q in held_out if q.level == level]
        greedy_runs: list[Rollout] = []
        sampled: list[Rollout] = []
        for number, question in enumerate(mine, start=1):
            rollout = run_rollout(
                weights,
                question,
                parent,
                GREEDY_FOLDER.format(number),
                greedy,
                describe,
            )
            log.append(rollout_record(rollout, "greedy", weights_name))
            greedy_runs.append(rollout)
            for sample in range(1, SAMPLES + 1):
                rollout = run_rollout(
                    weights,
                    question,
                    parent,
                    SAMPLED_FOLDER.format(number, sample),
                    lambda row: pick(row, rng),
                    describe,
                )
                log.append(rollout_record(rollout, "sampled", weights_name))
                sampled.append(rollout)
        results.append(
            LevelResult(
                level=level,
                questions=len(mine),
                greedy_right=sum(r.reward for r in greedy_runs),
                greedy_checked=sum(r.checked for r in greedy_runs),
                greedy_unsaved=unsaved_by_call(greedy_runs),
                sampled=len(sampled),
                sampled_right=sum(r.reward for r in sampled),
                checked=sum(r.checked for r in sampled),
                unsaved=unsaved_by_call(sampled),
                calls=sum(len(r.calls) for r in sampled),
                trace=greedy_runs[mine.index(traced_question(held_out, level))],
            )
        )
    return tuple(results)
