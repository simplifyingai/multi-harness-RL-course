"""Five reward rules: what the agent is paid for, which is what it learns.

The environment says what happened in a rollout: whether the answer was
right, whether one was submitted at all, how many calls it took and how many
tokens its replies held. A reward rule turns that into one number, and GRPO
learns to raise that number, whatever it measures. The five rules:

* ``correct``: 1 for a right answer, else 0 (chapters 4 to 6);
* ``submit``: the same, plus 0.2 for submitting any answer at all;
* ``call-bonus``: a right answer earns 1 plus up to 0.1 for using fewer
  calls (the article's bonus, the project's ``tool_bonus``);
* ``token-bonus``: a right answer earns 1 plus up to 0.1 for writing fewer
  tokens (the project's ``token_bonus`` and ``reward_for``);
* ``bonus-when-wrong``: the token bonus, paid whether the answer is right
  or not.

``tool_bonus``, ``token_bonus`` and ``reward_for`` are the project's
(``mhrl/rollout.py``) line for line; a test checks them against it.
"""

from __future__ import annotations

from dataclasses import dataclass

from environment import BUDGET

CORRECT = "correct"
SUBMIT = "submit"
CALL_BONUS = "call-bonus"
TOKEN_BONUS = "token-bonus"
BONUS_WHEN_WRONG = "bonus-when-wrong"
RULES = (CORRECT, SUBMIT, CALL_BONUS, TOKEN_BONUS, BONUS_WHEN_WRONG)
RULE_NOTES = {  # what each rule pays, as the runs print it
    CORRECT: "1 if right, else 0",
    SUBMIT: "1 if right, else 0, plus 0.2 for submitting any answer",
    CALL_BONUS: "if right, 1 + up to 0.1 for fewer calls; else 0",
    TOKEN_BONUS: "if right, 1 + up to 0.1 for fewer tokens; else 0",
    BONUS_WHEN_WRONG: "1 if right, else 0, plus up to 0.1 for fewer tokens",
}

MIN_TOOL_CALLS = 2  # look at the data, then answer: the fewest a solve can take
SUBMIT_BONUS = 0.2  # the article's reward for submitting anything
TOKEN_BUDGET = 400  # tokens per rollout at which the token bonus reaches 0
RIGHT = 1.0
WRONG = 0.0
# The two ways to treat a rollout whose sandbox died before grading.
SCORED_ZERO = "scored 0"
LEFT_OUT = "left unscored"


class Defaults:
    """The project's reward constant this chapter uses (``mhrl/config.py``)."""

    MAX_TOOL_BONUS = 0.1


class UnknownRule(ValueError):
    """Raised when a reward rule is not one of the five."""


@dataclass(frozen=True)
class Outcome:
    """What a reward rule may look at: what happened in one rollout.

    Attributes:
        correct: Whether the grader accepted answer.txt.
        submitted: Whether answer.txt held any answer at all.
        calls: How many calls the rollout made.
        tokens: How many tokens its replies held, added up.
    """

    correct: bool
    submitted: bool
    calls: int
    tokens: int


def tool_bonus(n_calls: int, max_calls: int) -> float:
    """The article's efficiency bonus: at most 0.1, more for fewer calls.

    Args:
        n_calls: Model calls the rollout made.
        max_calls: The per-rollout call budget.

    Returns:
        A bonus in [0, ``Defaults.MAX_TOOL_BONUS``].
    """
    span = max(max_calls - MIN_TOOL_CALLS, 1)
    unused = max_calls - max(n_calls, MIN_TOOL_CALLS)
    return Defaults.MAX_TOOL_BONUS * max(0.0, min(1.0, unused / span))


def token_bonus(n_sampled: int, budget: int) -> float:
    """Our efficiency bonus: at most 0.1, more for fewer generated tokens.

    Args:
        n_sampled: Tokens the model generated across the rollout.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        A bonus in [0, ``Defaults.MAX_TOOL_BONUS``].
    """
    return Defaults.MAX_TOOL_BONUS * max(0.0, 1.0 - n_sampled / max(budget, 1))


def reward_for(correct: bool, n_sampled: int, budget: int) -> float:
    """Correctness (1 or 0) plus the bonus, which only a correct answer earns.

    A wrong answer scores 0 however short it is, so the policy cannot buy the
    bonus by giving up early.

    Args:
        correct: Whether the answer was right.
        n_sampled: Tokens the model generated across the rollout.
        budget: Tokens at which the bonus reaches zero.

    Returns:
        The scalar reward GRPO trains on.
    """
    if not correct:
        return 0.0
    return 1.0 + token_bonus(n_sampled, budget)


def reward(rule: str, outcome: Outcome) -> float:
    """Pay one rollout by one rule.

    Args:
        rule: One of ``RULES``.
        outcome: What happened in the rollout.

    Returns:
        The rollout's reward.

    Raises:
        UnknownRule: If the rule is not one of the five.
    """
    right = RIGHT if outcome.correct else WRONG
    if rule == CORRECT:
        return right
    if rule == SUBMIT:
        return right + (SUBMIT_BONUS if outcome.submitted else 0.0)
    if rule == CALL_BONUS:
        if not outcome.correct:
            return WRONG
        return RIGHT + tool_bonus(outcome.calls, BUDGET)
    if rule == TOKEN_BONUS:
        return reward_for(outcome.correct, outcome.tokens, TOKEN_BUDGET)
    if rule == BONUS_WHEN_WRONG:
        return right + token_bonus(outcome.tokens, TOKEN_BUDGET)
    raise UnknownRule(f"unknown reward rule {rule!r}")
