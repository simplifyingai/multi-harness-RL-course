"""How this chapter prints: the environment's parts, rollouts in full, and tests.

Nothing here learns anything. It only turns numbers into lines a person can
read on screen, so ``minianalyst.py`` can stay about the agent and its loop.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
from environment import BUDGET, FINISH, State, verdict_text
from grader import REASON_CORRECT
from policy import FEATURE_NAMES, action_scores, features, softmax
from questions import QuestionSet
from tasks import ANSWER_FILE, DATABASE_FILE

if TYPE_CHECKING:  # only for type hints: minianalyst.py imports this module
    from minianalyst import GreedyResult, Settings, Update

PERCENT = 100
LABEL_WIDTH = 10  # "question  ", "folder    ", ...
PART_WIDTH = 14  # the environment's parts: "observations  "
ACTION_WIDTH = 8  # "QUERY_A " is the longest action name
SITUATION_WIDTH = 44  # "tables seen, result in hand, answer written" + 2
ANSWER_WIDTH = 12  # "2024-06-18" and "1804.05" fit, with room to spare
SHOWN_DECIMALS = 2  # chances, baselines and advantages
LOSS_DECIMALS = 3  # log-probabilities and losses
MEMORISE_TAG = " · memorise"
INDENT = "  "
SEPARATOR = " · "
NOTHING_DONE = "nothing done yet"
NO_FILE = "(none)"
PART_COUNTS = {6: "six", 7: "seven"}  # spelled out, as a person would say it
LOSS_LINE = "loss for each rollout: -advantage x (sum of log p of every action it took)"


def print_header(
    settings: Settings, questions: QuestionSet, actions: Sequence[str]
) -> None:
    """Print the run's settings and the environment's parts, one per line.

    Args:
        settings: The run's knobs.
        questions: The fair questions, split by city.
        actions: The actions this agent may take.
    """
    memorise = MEMORISE_TAG if settings.memorise else ""
    print(
        f"minianalyst · seed {settings.seed} · lr {settings.lr:g}"
        f" · per-question baseline · {settings.rollouts} rollouts{memorise}"
    )
    parts = [
        (
            "tasks",
            f"{len(questions.training)} training questions"
            f" ({len(questions.train_cities)} cities);"
            f" {len(questions.held_out)} held out"
            f" ({len(questions.test_cities)} other cities)",
        ),
        ("actions", "one per call: " + ", ".join(actions)),
        ("observations", "what a call sends back: table names, a result or an error"),
        ("state", f"its own folder ({DATABASE_FILE}, {ANSWER_FILE}) + calls used,"),
        ("", "schema seen, result in hand, answer written"),
        ("reward rule", f"chapter 1's grader on {ANSWER_FILE}: 1 or 0, at the end"),
        ("stop rule", f"{FINISH}, or the {BUDGET}-call budget is used up"),
    ]
    if settings.memorise:
        parts.append(
            ("memory", "after each rollout it is shown the right answer; RECALL")
        )
        parts.append(("", "writes the one it saw last time for the same question"))
    named = sum(1 for name, _ in parts if name)  # a row with no name goes on
    print(f"the environment, in {PART_COUNTS.get(named, named)} parts")
    for name, text in parts:
        print(f"{INDENT}{name:<{PART_WIDTH}}{text}")
    names = ", ".join(FEATURE_NAMES[:-1]) + f", {FEATURE_NAMES[-1]} / {BUDGET}"
    print(
        f"the policy: softmax over the {len(actions)} actions; each score is a"
        f" weighted sum of {len(FEATURE_NAMES)} features"
    )
    print(f"{INDENT}({names})")
    print(LOSS_LINE)


def files_text(files: Sequence[str]) -> str:
    """Write a folder's file names on one line.

    Args:
        files: The names.

    Returns:
        The names joined with commas, or "(none)".
    """
    return ", ".join(files) if files else NO_FILE


def print_rollout(update: Update, when: str, pause_s: float) -> None:
    """Print one rollout in full: every call, how it stopped, and its update.

    Args:
        update: The rollout and the gradient-descent step it caused.
        when: A few words on when it ran, e.g. ``"after 599 steps of training"``.
        pause_s: Seconds to wait after each call, so a viewer can follow.
    """
    rollout = update.rollout
    count = len(rollout.calls)
    print(f"\nrollout {update.number} · {when}")
    print(f"{'question':<{LABEL_WIDTH}}{rollout.question.task.question}")
    print(
        f"{'folder':<{LABEL_WIDTH}}{rollout.folder}, new: {DATABASE_FILE}"
        " (a copy of shop.db)"
    )
    for call, chance in zip(rollout.calls, rollout.chances, strict=True):
        print(
            f"{INDENT}call {call.number}  {call.action:<{ACTION_WIDTH}}"
            f" p {chance:.{SHOWN_DECIMALS}f}"
            f"  advantage {tidy(update.advantage):+.{SHOWN_DECIMALS}f}"
            f"  → {call.observation}"
        )
        time.sleep(pause_s)
    print(
        f"{'stopped':<{LABEL_WIDTH}}{rollout.stop}, after {counted(count, 'call')}"
        f"{SEPARATOR}{verdict_text(rollout.verdict)}"
    )
    print(
        f"{'folder':<{LABEL_WIDTH}}at the end: {files_text(rollout.files)}"
        f"{SEPARATOR}deleted"
    )
    shared = f"the same for all {count} actions" if count > 1 else "for its one action"
    print(
        f"{'update':<{LABEL_WIDTH}}reward {rollout.reward} - baseline"
        f" {update.baseline:.{SHOWN_DECIMALS}f} = advantage"
        f" {tidy(update.advantage):+.{SHOWN_DECIMALS}f}, {shared}"
    )
    log_sum = sum(rollout.log_probs)
    loss = tidy(update.loss, LOSS_DECIMALS)
    print(
        f"{'':<{LABEL_WIDTH}}loss = -advantage x (sum of log p, "
        f"{log_sum:.{LOSS_DECIMALS}f}) = {loss:+.{LOSS_DECIMALS}f}"
    )


def counted(count: int, word: str) -> str:
    """Write a count with its word, singular or plural.

    Args:
        count: How many.
        word: The singular word, e.g. ``"call"``.

    Returns:
        For example ``"1 call"`` or ``"8 calls"``.
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def tidy(value: float, decimals: int = SHOWN_DECIMALS) -> float:
    """Round a number for printing, so a tiny negative never shows as -0.00.

    Args:
        value: The number.
        decimals: How many decimals will be shown.

    Returns:
        The rounded number; adding 0.0 turns -0.0 into 0.0.
    """
    return round(value, decimals) + 0.0


def print_stretch(start: int, end: int, updates: Sequence[Update]) -> None:
    """Print how a stretch of rollouts went: the share correct and the calls.

    Args:
        start: The stretch's first rollout.
        end: Its last rollout.
        updates: The stretch's rollouts.
    """
    correct = sum(update.rollout.reward for update in updates)
    calls = np.mean([len(update.rollout.calls) for update in updates])
    share = round(PERCENT * correct / len(updates))
    print(
        f"rollouts {start}-{end}: {correct} correct ({share}%)"
        f"{SEPARATOR}{calls:.1f} calls per rollout"
    )


def situation_text(state: State) -> str:
    """Say in words what the agent has done so far.

    Args:
        state: The rollout's state before a call.

    Returns:
        For example ``"tables seen, result in hand"``.
    """
    done = [
        name
        for name, flag in (
            ("tables seen", state.schema_seen),
            ("result in hand", state.result is not None),
            ("answer written", state.written),
        )
        if flag
    ]
    return ", ".join(done) if done else NOTHING_DONE


def print_greedy_path(
    weights: np.ndarray, states: Sequence[State], actions: Sequence[str]
) -> None:
    """Print what the trained agent does at each call of a greedy rollout.

    Args:
        weights: The trained weights.
        states: The state before each call of one greedy rollout of a
            training question.
        actions: The actions this agent may take.
    """
    finish = actions.index(FINISH)
    print(
        "\nthe agent at the end, on a training question, greedy"
        " (always its most likely action)"
    )
    for state in states:
        chances = softmax(action_scores(weights, features(state)))
        top = int(np.argmax(chances))
        situation = situation_text(state)
        print(
            f"{INDENT}call {state.calls + 1}  {situation:<{SITUATION_WIDTH}}"
            f"{actions[top]:<{ACTION_WIDTH}}{round(PERCENT * chances[top]):>4}%"
            f"   {FINISH} {round(PERCENT * chances[finish])}%"
        )


def calls_text(actions: Sequence[str]) -> str:
    """Write a rollout's actions on one line, with repeats counted.

    Args:
        actions: The actions of every call, in order.

    Returns:
        For example ``"LOOK, QUERY_A, WRITE x6"``.
    """
    parts = []
    for action, repeats in groupby(actions):  # runs of the same action
        count = len(list(repeats))
        parts.append(action if count == 1 else f"{action} x{count}")
    return ", ".join(parts)


def print_test(title: str, results: Sequence[GreedyResult], show: bool) -> None:
    """Print a greedy test: its calls, one line per question if asked, the score.

    Args:
        title: What was tested, e.g. ``"cities never used in training"``.
        results: One result per question.
        show: Print the calls and one line per question.
    """
    correct = sum(result.correct for result in results)
    if not show:
        print(f"greedy test on the {title}: {correct} of {len(results)}")
        return
    print(f"\ngreedy test on {title}")
    paths = {result.actions for result in results}
    if len(paths) == 1:
        print(f"every rollout made the same calls: {calls_text(paths.pop())}")
    replies = {reply for result in results for reply in result.observations}
    if len(replies) == 1:
        print(f"and every call got the same answer: {replies.pop()}")
    width = max(len(result.question.task.question) for result in results) + 2
    for result in results:
        verdict = REASON_CORRECT if result.correct else result.reason
        written = result.answer if result.answer is not None else NO_FILE
        print(
            f"{result.question.task.question:<{width}}"
            f"{ANSWER_FILE} {written:<{ANSWER_WIDTH}}{verdict}"
        )
    print(f"held-out score: {correct} of {len(results)}")
