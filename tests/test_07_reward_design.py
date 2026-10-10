"""Chapter 7: five reward rules, replies with a length, guessing and crashes."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("07-reward-design")

import environment  # noqa: E402
import grpo  # noqa: E402
import plot  # noqa: E402
import questions  # noqa: E402
import real_run  # noqa: E402
import replies  # noqa: E402
import report  # noqa: E402
import reward_lab  # noqa: E402
import rewards  # noqa: E402
import tasks  # noqa: E402

PREVIOUS = ROOT / "06-ratio-and-clip"
SCRIPT_TIMEOUT_S = 300
STEP = 1e-6  # how far a weight is moved to measure a slope
MEASURED_TOLERANCE = 1e-6
KINK_MARGIN = 1e-3  # measured slopes stay away from the clip's two edges
SHORT_RUN = 30  # steps in the quick determinism checks
WORK = "work"

# mhrl/rollout.py's three reward functions, copied from the project unchanged
# (their docstrings are left out: only the code is compared).
PROJECT_REWARDS = """
def tool_bonus(n_calls: int, max_calls: int) -> float:
    span = max(max_calls - MIN_TOOL_CALLS, 1)
    unused = max_calls - max(n_calls, MIN_TOOL_CALLS)
    return Defaults.MAX_TOOL_BONUS * max(0.0, min(1.0, unused / span))


def token_bonus(n_sampled: int, budget: int) -> float:
    return Defaults.MAX_TOOL_BONUS * max(0.0, 1.0 - n_sampled / max(budget, 1))


def reward_for(correct: bool, n_sampled: int, budget: int) -> float:
    if not correct:
        return 0.0
    return 1.0 + token_bonus(n_sampled, budget)
"""

# mhrl/training/grpo.py's loss_fn, as in chapter 6's test.
PROJECT_LOSS_FN = """
def loss_fn(item: TrainItem) -> tuple[mx.array, mx.array]:
    new = token_logprobs(model, item)
    ratio = mx.exp(new - mx.array(item.old_logprobs))
    unclipped = ratio * item.advantage
    clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
    return -mx.minimum(unclipped, clipped).sum(), mx.stop_gradient(ratio)
"""
TRANSLATION = (
    ("mx.stop_gradient(ratio)", "ratio"),
    ("mx.", "np."),
    ("token_logprobs(model, item)", "call_logprobs(weights, item)"),
)

SHOW_GROUP_BLOCK = """rewards     [1.08, 1.06, 1.07, 1.05]
mean        1.065
spread      0.0112  (standard deviation)
advantages  (reward - mean) / (spread + 0.0001)
            [+1.33, -0.44, +0.44, -1.33]

paid for right answers only, the same rollouts are [1, 1, 1, 1]:
no contrast, skipped
"""


def quiet(**knobs: float | int | bool | str) -> tuple[reward_lab.RunResult, str]:
    """Run reward_lab.py in-process and capture what it prints.

    Args:
        **knobs: Settings fields to change, e.g. ``steps=30``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = reward_lab.run(reward_lab.Settings(**knobs))
    return result, output.getvalue()


def run_script(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run one of this chapter's scripts as a learner would.

    Args:
        script: The file name, e.g. ``"reward_lab.py"``.
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / script), *flags]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
    )


def function_body(source: str, name: str) -> list[ast.stmt]:
    """Return a function's statements, docstring left out.

    Args:
        source: Python source holding the function.
        name: The function's name.

    Returns:
        The statements.
    """
    tree = ast.parse(source)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return node.body[1:] if ast.get_docstring(node) is not None else node.body


def function_code(source: str, name: str) -> str:
    """Return a function's code without its docstring, as a syntax tree dump.

    Args:
        source: Python source holding the function.
        name: The function's name.

    Returns:
        The dump.
    """
    return ast.dump(ast.Module(body=function_body(source, name), type_ignores=[]))


def chapter_code(path: Path, name: str) -> str:
    """Return a function's code from a file, docstring left out.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The dump.
    """
    return function_code(path.read_text(encoding="utf-8"), name)


def readme() -> str:
    """Return the chapter's README.

    Returns:
        Its text.
    """
    return (CHAPTER / "README.md").read_text(encoding="utf-8")


def question(level: str = questions.EASY) -> questions.Question:
    """Return the first training question of a level.

    Args:
        level: The difficulty.

    Returns:
        The question.
    """
    return next(q for q in questions.make_questions().training if q.level == level)


@contextlib.contextmanager
def analyst(
    tmp_path: Path, level: str = questions.EASY
) -> Iterator[environment.MiniAnalyst]:
    """Make a MiniAnalyst on a fresh copy of the database.

    Args:
        tmp_path: pytest's per-test folder.
        level: The question's difficulty.

    Yields:
        The environment.
    """
    with environment.fresh_folder(tmp_path, "rollout") as folder:
        yield environment.MiniAnalyst(question(level), folder, questions.GUESS)


def one_call(env: environment.MiniAnalyst, *actions: str, tokens: int = 40) -> None:
    """Make one call holding the given actions.

    Args:
        env: The environment.
        *actions: The actions the reply holds.
        tokens: The reply's length.
    """
    env.open_call(tokens)
    for action in actions:
        env.act(action)
    env.close_call()


@pytest.fixture(scope="module")
def module_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point COURSE_HOME at a temporary folder for the runs this file shares.

    Args:
        tmp_path_factory: pytest's maker of temporary folders.

    Yields:
        The temporary course home.
    """
    home = tmp_path_factory.mktemp("course-home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("COURSE_HOME", str(home))
        yield home


@pytest.fixture(scope="module")
def runs(module_home: Path) -> dict[str, tuple[reward_lab.RunResult, str]]:
    """Run every rule once (seed 0, 200 steps) for several tests.

    Args:
        module_home: The temporary course home the runs write into.

    Returns:
        Each rule's result and printed output.
    """
    return {rule: quiet(rule=rule) for rule in rewards.RULES}


# --- What is copied ----------------------------------------------------------


@pytest.mark.parametrize("name", ["grader.py", "policy.py", "data/shop.db"])
def test_the_grader_policy_and_data_are_chapter_sixs(name: str) -> None:
    """Copied unchanged."""
    assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_six_only_in_the_chapter_name() -> None:
    """The only change is where outputs go."""
    mine = (CHAPTER / "tasks.py").read_text(encoding="utf-8")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine.replace("07-reward-design", "06-ratio-and-clip") == theirs


def test_group_advantages_is_chapter_sixs() -> None:
    """The project's function, line for line, as in chapters 5 and 6."""
    mine = chapter_code(CHAPTER / "grpo.py", "group_advantages")
    assert mine == chapter_code(PREVIOUS / "grpo.py", "group_advantages")


def test_loss_fn_is_the_projects_line_for_line_in_numpy() -> None:
    """mhrl/training/grpo.py's five lines, with np for mx and the toy's names."""
    project = PROJECT_LOSS_FN
    for theirs, ours in TRANSLATION:
        project = project.replace(theirs, ours)
    expected = function_body(project, "loss_fn")
    source = (CHAPTER / "grpo.py").read_text(encoding="utf-8")
    first, *body = function_body(source, "loss_fn")
    assert ast.unparse(first) == "eps = Defaults.CLIP_EPSILON"
    assert [ast.dump(s) for s in body] == [ast.dump(s) for s in expected]


@pytest.mark.parametrize("name", ["tool_bonus", "token_bonus", "reward_for"])
def test_the_reward_functions_are_the_projects_line_for_line(name: str) -> None:
    """mhrl/rollout.py's tool_bonus, token_bonus and reward_for, unchanged."""
    mine = chapter_code(CHAPTER / "rewards.py", name)
    assert mine == function_code(PROJECT_REWARDS, name)
    assert rewards.Defaults.MAX_TOOL_BONUS == 0.1
    assert rewards.MIN_TOOL_CALLS == 2


# --- The five rules ------------------------------------------------------------


def outcome(
    correct: bool, submitted: bool = True, calls: int = 4, tokens: int = 160
) -> rewards.Outcome:
    """Describe a rollout for a reward rule.

    Args:
        correct: Whether it was right.
        submitted: Whether it handed in an answer.
        calls: How many calls it made.
        tokens: How many tokens its replies held.

    Returns:
        The outcome.
    """
    return rewards.Outcome(correct, submitted, calls, tokens)


@pytest.mark.parametrize(
    ("rule", "right", "wrong", "nothing"),
    [
        (rewards.CORRECT, 1.0, 0.0, 0.0),
        (rewards.SUBMIT, 1.2, 0.2, 0.0),
        (rewards.CALL_BONUS, 1.0 + 0.1 * 4 / 6, 0.0, 0.0),
        (rewards.TOKEN_BONUS, 1.06, 0.0, 0.0),
        (rewards.BONUS_WHEN_WRONG, 1.06, 0.06, 0.06),
    ],
)
def test_each_rule_pays_what_it_says(
    rule: str, right: float, wrong: float, nothing: float
) -> None:
    """4 calls and 160 tokens; right, wrong but handed in, and nothing at all."""
    assert rewards.reward(rule, outcome(True)) == pytest.approx(right)
    assert rewards.reward(rule, outcome(False)) == pytest.approx(wrong)
    assert rewards.reward(rule, outcome(False, submitted=False)) == pytest.approx(
        nothing
    )


def test_the_bonuses_run_from_01_to_0() -> None:
    """The call bonus is full at 2 calls and gone at 8; the token one at 0 and 400."""
    budget = environment.BUDGET
    assert rewards.tool_bonus(1, budget) == rewards.tool_bonus(2, budget) == 0.1
    assert rewards.tool_bonus(budget, budget) == 0.0
    assert rewards.token_bonus(0, rewards.TOKEN_BUDGET) == 0.1
    assert rewards.token_bonus(rewards.TOKEN_BUDGET, rewards.TOKEN_BUDGET) == 0.0
    with pytest.raises(rewards.UnknownRule):
        rewards.reward("nope", outcome(True))


def test_the_show_group_is_four_token_bonuses() -> None:
    """[1.08, 1.06, 1.07, 1.05] is 80, 160, 120 and 200 tokens, all right."""
    paid = [
        rewards.reward_for(True, tokens, rewards.TOKEN_BUDGET)
        for tokens in (80, 160, 120, 200)
    ]
    assert paid == pytest.approx(list(reward_lab.SHOW_GROUP))
    advantages = grpo.group_advantages(list(reward_lab.SHOW_GROUP))
    assert [round(a, 2) for a in advantages] == [1.33, -0.44, 0.44, -1.33]


def test_show_group_prints_the_readme_block() -> None:
    """--show-group works out the README's group and stops."""
    finished = run_script("reward_lab.py", "--show-group")
    assert finished.stdout == SHOW_GROUP_BLOCK
    assert SHOW_GROUP_BLOCK in readme()


# --- The environment -------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "room"), [(20, 1), (79, 1), (80, 2), (120, 2), (121, 0)]
)
def test_a_reply_holds_one_action_two_or_none(
    tmp_path: Path, tokens: int, room: int
) -> None:
    """Under 80 tokens one action, 80 to 120 two, over 120 none (cut off)."""
    with analyst(tmp_path) as env:
        assert env.open_call(tokens) == room


def test_a_cut_off_reply_ends_the_rollout_with_nothing_handed_in(
    tmp_path: Path,
) -> None:
    """The call counts at 120 tokens, and the grader finds no answer."""
    with analyst(tmp_path) as env:
        one_call(env, environment.LOOK)
        one_call(env, tokens=150)
        assert env.state.stop == environment.CUT
        assert env.calls[-1].cut_off and env.calls[-1].tokens == 120
        assert env.grade is not None and not env.grade.correct
        assert env.tokens == 160


def test_submit_hands_in_the_result_and_ends(tmp_path: Path) -> None:
    """LOOK, QUERY_A and SUBMIT packed in two calls: right, in two calls."""
    with analyst(tmp_path) as env:
        one_call(env, environment.LOOK, environment.QUERY_A, tokens=90)
        assert not env.done
        one_call(env, environment.SUBMIT)
        assert env.state.stop == environment.SUBMITTED
        assert env.grade is not None and env.grade.correct
        assert len(env.calls) == 2


def test_submit_with_nothing_in_hand_goes_on(tmp_path: Path) -> None:
    """No result yet: nothing is handed in and the rollout continues."""
    with analyst(tmp_path) as env:
        env.open_call(40)
        assert env.act(environment.SUBMIT) == environment.NOTHING_TO_SUBMIT
        env.close_call()
        assert not env.done


def test_a_guess_hands_in_0_at_once(tmp_path: Path) -> None:
    """SUBMIT_GUESS writes 0 and ends; on a default question it is wrong."""
    with analyst(tmp_path) as env:
        one_call(env, environment.SUBMIT_GUESS)
        assert env.state.stop == environment.GUESSED
        assert env.grade is not None and env.grade.submitted == "0"
        assert not env.grade.correct


def test_finish_stops_with_nothing_handed_in(tmp_path: Path) -> None:
    """FINISH ends the rollout, and the grader finds no answer.txt."""
    with analyst(tmp_path) as env:
        one_call(env, environment.FINISH)
        assert env.state.stop == environment.FINISHED
        assert env.grade is not None and env.grade.submitted is None


def test_the_budget_ends_the_rollout(tmp_path: Path) -> None:
    """Eight calls of LOOK: the budget is used and nothing was handed in."""
    with analyst(tmp_path) as env:
        for _ in range(environment.BUDGET):
            one_call(env, environment.LOOK)
        assert env.state.stop == environment.BUDGET_USED
        with pytest.raises(environment.RolloutOver):
            env.open_call(40)


def test_actions_need_an_open_call_with_room(tmp_path: Path) -> None:
    """An action outside a call, or one too many, is refused."""
    with analyst(tmp_path) as env:
        with pytest.raises(environment.CallNotOpen):
            env.act(environment.LOOK)
        env.open_call(40)
        env.act(environment.LOOK)
        with pytest.raises(environment.CallNotOpen):
            env.act(environment.LOOK)
        with pytest.raises(environment.CallNotOpen):
            env.open_call(40)
        env.close_call()
        with pytest.raises(environment.CallNotOpen):
            env.close_call()


def test_a_crash_leaves_no_verdict(tmp_path: Path) -> None:
    """The sandbox dies: the rollout stops and is never graded."""
    with analyst(tmp_path) as env:
        env.crash()
        assert env.done and env.grade is None


# --- Reply lengths ------------------------------------------------------------------


def test_reply_tokens_never_go_below_20() -> None:
    """A drawn length is rounded, and floored at the fewest an action needs."""
    assert replies.reply_tokens(math.log(5)) == replies.MIN_REPLY
    assert replies.reply_tokens(math.log(41.4)) == 41
    assert replies.typical_length(replies.START_WEIGHT) == pytest.approx(40)


def test_the_length_slope_is_the_measured_slope_of_log_p() -> None:
    """(x - w) / s**2, checked by moving the weight a hair."""
    drawn, weight = math.log(70), math.log(40)
    up = replies.length_log_prob(drawn, weight + STEP)
    down = replies.length_log_prob(drawn, weight - STEP)
    measured = (up - down) / (2 * STEP)
    assert replies.length_slope(drawn, weight) == pytest.approx(measured, rel=1e-6)


def test_draws_spread_by_30_percent_around_the_typical_length() -> None:
    """Half the draws land above the typical length; their log spread is 0.3."""
    rng = np.random.default_rng(0)
    drawn = np.array([replies.sample_length(math.log(40), rng) for _ in range(4000)])
    assert np.mean(drawn > math.log(40)) == pytest.approx(0.5, abs=0.03)
    assert np.std(drawn) == pytest.approx(replies.SPREAD, abs=0.01)


# --- The gradient -------------------------------------------------------------------


def random_rollouts(
    rng: np.random.Generator, weights: grpo.Weights, spread: float
) -> list[grpo.Trained]:
    """Make three rollouts whose old log p sit some way from the new ones.

    Args:
        rng: The random generator.
        weights: The weights the new log p come from.
        spread: How far, in log p, the old ones are moved at random.

    Returns:
        The rollouts, with advantages of both signs.
    """
    items = []
    rows, actions = weights.actions.shape
    for advantage, picks, lengths in ((0.58, 8, 6), (-1.73, 3, 2), (1.0, 5, 4)):
        situations = tuple(rng.uniform(size=rows) for _ in range(picks))
        choices = tuple(int(c) for c in rng.integers(actions, size=picks))
        drawn = tuple(float(x) for x in weights.length + rng.normal(0, 0.3, lengths))
        item = grpo.Trained(situations, choices, drawn, (), advantage)
        new = grpo.call_logprobs(weights, item)
        old = tuple(float(n + rng.uniform(-spread, spread)) for n in new)
        items.append(replace(item, old_logprobs=old))
    return items


def test_the_hand_gradient_is_the_measured_slope_of_the_loss() -> None:
    """For the action weights and the length weight, with some calls clipped."""
    rng = np.random.default_rng(0)
    weights = grpo.Weights(rng.normal(size=(5, 7)), math.log(50))
    items = random_rollouts(rng, weights, spread=0.5)
    ratios = np.concatenate([grpo.loss_fn(weights, item)[1] for item in items])
    edges = np.array([1 - grpo.Defaults.CLIP_EPSILON, 1 + grpo.Defaults.CLIP_EPSILON])
    assert np.abs(ratios[:, None] - edges).min() > KINK_MARGIN
    update = grpo.step_update(items, weights)
    assert update.clipped.any()
    decisions = sum(item.decisions for item in items)

    def step_loss(w: grpo.Weights) -> float:
        return sum(float(grpo.loss_fn(w, item)[0]) for item in items) / decisions

    measured = np.zeros_like(weights.actions)
    for index in np.ndindex(weights.actions.shape):
        up, down = weights.actions.copy(), weights.actions.copy()
        up[index] += STEP
        down[index] -= STEP
        measured[index] = (
            step_loss(replace(weights, actions=up))
            - step_loss(replace(weights, actions=down))
        ) / (2 * STEP)
    assert np.abs(update.gradient.actions - measured).max() < MEASURED_TOLERANCE
    length_slope = (
        step_loss(replace(weights, length=weights.length + STEP))
        - step_loss(replace(weights, length=weights.length - STEP))
    ) / (2 * STEP)
    assert update.gradient.length == pytest.approx(length_slope, abs=1e-5)


def test_the_length_weight_never_goes_below_the_floor() -> None:
    """A step that would make the typical reply shorter than 20 tokens stops at 20."""
    weights = grpo.Weights(np.zeros((5, 7)), replies.FLOOR_WEIGHT)
    pushed = grpo.Weights(np.zeros((5, 7)), 10.0)
    after = grpo.apply_update(weights, pushed, lr=3.0, length_lr=1.0)
    assert after.length == replies.FLOOR_WEIGHT
    nothing = grpo.step_update([], weights)
    assert nothing.decisions == 0 and nothing.loss is None


# --- Questions and guessing ----------------------------------------------------------


def test_by_default_a_guess_answers_nothing() -> None:
    """Brussels' 0 orders is left out; no question is answered by 0."""
    fair = questions.make_questions()
    assert (len(fair.training), len(fair.held_out)) == (45, 18)
    assert questions.guess_hits(fair.training + fair.held_out, questions.GUESS) == 0
    reasons = [reason for _, reason in fair.left_out]
    assert questions.GUESSED_RIGHT in reasons


def test_guessable_adds_questions_a_guess_gets_right() -> None:
    """The cash question (0 for every city) and Brussels come back."""
    fair = questions.make_questions(guessable=True)
    assert (len(fair.training), len(fair.held_out)) == (53, 22)
    assert questions.guess_hits(fair.training, questions.GUESS) == 8
    assert questions.guess_hits(fair.held_out, questions.GUESS) == 4


# --- Whole runs ---------------------------------------------------------------


def test_every_ratio_is_exactly_1_and_nothing_is_clipped(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """One fresh batch, one update per step: chapter 6's default."""
    result, _ = runs[rewards.CORRECT]
    ratios = np.concatenate([step.ratios for step in result.steps])
    assert ratios.size and np.all(ratios == 1.0)
    assert not any(step.clipped.any() for step in result.steps)


def end_tally(result: reward_lab.RunResult) -> report.Tally:
    """Tally a run's last quarter.

    Args:
        result: The run.

    Returns:
        The tally.
    """
    return report.start_and_end(result.steps)[1]


def test_correct_only_wanders_up_toward_the_budget(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """More calls at the end than at the start, and most rollouts right."""
    start, end = report.start_and_end(runs[rewards.CORRECT][0].steps)
    assert end.calls_per_rollout > start.calls_per_rollout
    assert end.accuracy > 0.8


def test_submit_teaches_blind_guessing(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """Every rollout guesses by the end, and none is right."""
    end = end_tally(runs[rewards.SUBMIT][0])
    assert end.guesses == end.rollouts and end.correct == 0
    assert end.reward == pytest.approx(rewards.SUBMIT_BONUS)


def test_the_call_bonus_lengthens_replies_and_cuts_calls(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """Against correct only: longer replies, fewer calls, more cut off."""
    calls = end_tally(runs[rewards.CALL_BONUS][0])
    correct = end_tally(runs[rewards.CORRECT][0])
    assert calls.tokens_per_reply > correct.tokens_per_reply
    assert calls.calls_per_rollout < correct.calls_per_rollout
    assert calls.cut_off_share > correct.cut_off_share


def test_the_token_bonus_keeps_replies_short(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """Near the 20-token floor, and none cut off."""
    end = end_tally(runs[rewards.TOKEN_BONUS][0])
    assert end.tokens_per_reply < 30 and end.cut_off == 0


def test_a_bonus_when_wrong_teaches_giving_up(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """One call per rollout by the end, and nothing right."""
    end = end_tally(runs[rewards.BONUS_WHEN_WRONG][0])
    assert end.calls_per_rollout == 1.0 and end.correct == 0


@pytest.mark.parametrize("rule", [rewards.CORRECT, rewards.CALL_BONUS])
def test_the_end_of_the_run_matches_the_readme(
    runs: dict[str, tuple[reward_lab.RunResult, str]], rule: str
) -> None:
    """The quarter table and the greedy test, as quoted."""
    _, printed = runs[rule]
    start = printed.index("the training rollouts, in each quarter")
    end = printed.index("\n\nwrote ")
    assert printed[start:end] in readme()


def test_metrics_has_one_line_per_step_with_the_projects_fields(
    runs: dict[str, tuple[reward_lab.RunResult, str]],
) -> None:
    """The project's names where it has them, and what plot.py reads."""
    result, _ = runs[rewards.CALL_BONUS]
    lines = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == reward_lab.DEFAULT_STEPS
    first = json.loads(lines[0])
    for field in (
        "stage",
        "step",
        "accuracy",
        "mean_reward",
        "mean_calls",
        "groups_without_contrast",
        "importance_ratio_mean",
        "rollouts",
        "scored",
        "correct",
        "replies",
        "reply_tokens",
        "cut_off_replies",
        "typical_length",
    ):
        assert field in first, field
    assert result.metrics_path.parent.name == (
        "call-bonus-seed0-lr3-length-lr0.3-steps200"
    )


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical logs and output; seed 1 differs."""
    first, first_text = quiet(steps=SHORT_RUN, rule=rewards.CALL_BONUS)
    again, again_text = quiet(steps=SHORT_RUN, rule=rewards.CALL_BONUS)
    other, _ = quiet(steps=SHORT_RUN, rule=rewards.CALL_BONUS, seed=1)
    assert (first.metrics_md5, first.rollouts_md5) == (
        again.metrics_md5,
        again.rollouts_md5,
    )
    assert first_text == again_text
    assert other.rollouts_md5 != first.rollouts_md5


def test_every_rollout_gets_its_own_folder_and_none_is_left(
    course_home: Path,
) -> None:
    """Folders are named by step, group and rollout; work/ is empty after."""
    result, _ = quiet(steps=2)
    names = [r.folder for s in result.steps for r in s.rollouts]
    assert names[0] == "step-001-group-1-rollout-1"
    assert len(set(names)) == 16
    work = course_home / "07-reward-design" / WORK
    assert tasks.work_root() == work and list(work.iterdir()) == []


def test_crashes_scored_0_or_left_out() -> None:
    """The same sandboxes die either way; scored 0, or left out of the group."""
    scored, printed = quiet(steps=SHORT_RUN, crashes=0.1)
    unscored, _ = quiet(steps=SHORT_RUN, crashes=0.1, unscored=True)
    dead = [r for s in scored.steps for r in s.rollouts if r.verdict is None]
    assert dead and all(r.reward == 0.0 for r in dead)
    assert "sandboxes that died before grading" in printed
    for step in unscored.steps:
        for group in step.groups:
            for rollout, advantage in zip(
                group.rollouts,
                group.advantages or (None,) * len(group.rollouts),
                strict=True,
            ):
                if rollout.verdict is None:
                    assert rollout.reward is None and advantage is None
    first_dead = [
        r.folder for s in unscored.steps for r in s.rollouts if r.verdict is None
    ]
    assert first_dead[0] == dead[0].folder


def test_the_trace_prints_a_group_per_quarter() -> None:
    """--trace: the first group of steps 1, 11, 21 and 31 of a 40-step run."""
    printed = run_script("reward_lab.py", "--trace", "--fast", "--steps", "40").stdout
    for step in (1, 11, 21, 31):
        assert f"\nstep {step} · group 1 of 2 · " in printed


# --- plot.py, seeds.py and real_run.py --------------------------------------


def test_plot_redraws_the_run_from_the_metrics_file_alone(
    runs: dict[str, tuple[reward_lab.RunResult, str]], tmp_path: Path
) -> None:
    """plot.py reads only metrics.jsonl, even after it is moved elsewhere."""
    result, _ = runs[rewards.CALL_BONUS]
    moved = tmp_path / "elsewhere" / "metrics.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(result.metrics_path.read_bytes())
    finished = run_script("plot.py", str(moved))
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.startswith("200 steps, 1,600 rollouts")
    for title in ("share right", "tokens per reply", "calls per rollout"):
        assert title in finished.stdout
    assert finished.stdout == run_script("plot.py", str(result.metrics_path)).stdout


def test_plot_refuses_a_missing_or_foreign_file(tmp_path: Path) -> None:
    """A plain message and a non-zero exit, never a traceback."""
    missing = run_script("plot.py", str(tmp_path / "nope.jsonl"))
    foreign = tmp_path / "other.jsonl"
    foreign.write_text('{"hello": 1}\n', encoding="utf-8")
    other = run_script("plot.py", str(foreign))
    impossible = tmp_path / "metrics.jsonl"
    entry = {
        "step": 1,
        "rollouts": 8,
        "correct": 9,  # more right than rollouts: not ours
        "scored": 8,
        "mean_reward": 1.0,
        "replies": 30,
        "reply_tokens": 1200,
        "cut_off_replies": 0,
    }
    impossible.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    broken = run_script("plot.py", str(impossible))
    for finished in (missing, other, broken):
        assert finished.returncode != 0
        assert "Traceback" not in finished.stderr
        assert finished.stderr.startswith("plot.py: ")


def test_plot_keeps_its_copied_numbers_equal_to_the_chapters() -> None:
    """plot.py copies two numbers so it can stand alone; they must agree."""
    assert plot.OUTPUT_CAP == environment.OUTPUT_CAP
    assert plot.BUDGET == environment.BUDGET


def test_seeds_script_prints_a_table_and_claims_per_rule() -> None:
    """Two short seeds of two rules: one row each, the totals, the claims."""
    finished = run_script(
        "seeds.py", "--seeds", "2", "--steps", "8", "--rules", "correct,call-bonus"
    )
    assert finished.returncode == 0, finished.stderr
    for rule in (rewards.CORRECT, rewards.CALL_BONUS):
        assert f"\n{rule}: " in finished.stdout
    assert finished.stdout.count("\nall   ") == 2
    assert finished.stdout.count("holds in ") == 3
    alone = run_script(
        "seeds.py", "--seeds", "1", "--steps", "4", "--rules", "token-bonus"
    )
    assert "not checked: it needs the correct-only runs" in alone.stdout


def test_real_run_prints_the_projects_first_run() -> None:
    """Tokens per call 46 to 404, as `mhrl.cli steps` printed, and the bonus checked."""
    finished = run_script("real_run.py")
    assert finished.returncode == 0, finished.stderr
    assert "21 of 21 right rollouts were paid exactly" in finished.stdout
    lines = real_run.step_lines()
    assert [round(line.tokens_per_call) for line in lines] == [
        46,
        38,
        40,
        57,
        127,
        105,
        365,
        404,
    ]
    assert [line.capped for line in lines] == [0, 0, 0, 0, 0, 0, 6, 15]


def test_real_run_refuses_a_broken_record(tmp_path: Path) -> None:
    """A missing file or a missing field is refused by name."""
    broken = tmp_path / "rollouts.jsonl"
    broken.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(real_run.RealRunError):
        real_run.read_rollouts(broken)
    with pytest.raises(real_run.RealRunError):
        real_run.read_rollouts(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "flags",
    [
        ("--reward", "nope"),
        ("--show-group", "1"),
        ("--show-group", "1,x"),
        ("--crashes", "2"),
        ("--unscored",),
        ("--length-lr", "nan"),
        ("--steps", "0"),
        ("--seed", "-1"),
    ],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, ...]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script("reward_lab.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


@pytest.mark.parametrize(
    "flags",
    [("--seeds", "0"), ("--steps", "0"), ("--length-lr", "-1"), ("--rules", "x")],
)
def test_seeds_refuses_bad_flags(flags: tuple[str, str]) -> None:
    """seeds.py checks its flags like reward_lab.py does."""
    finished = run_script("seeds.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr
