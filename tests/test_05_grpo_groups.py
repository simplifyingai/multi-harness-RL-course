"""Chapter 5: group advantages, skipped groups, difficulty, runs and the plot."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("05-grpo-groups")

import environment  # noqa: E402
import grpo  # noqa: E402
import grpo_groups  # noqa: E402
import plot  # noqa: E402
import policy  # noqa: E402
import questions  # noqa: E402
import tasks  # noqa: E402
from environment import (  # noqa: E402
    FINISH,
    LOOK,
    QUERY_A,
    WRITE,
    MiniAnalyst,
    fresh_folder,
)

PREVIOUS = ROOT / "04-episodes"
SCRIPT_TIMEOUT_S = 300
STEP = 1e-6  # how far a weight is moved to measure a slope
MEASURED_TOLERANCE = 1e-6
SHORT_RUN = 30  # steps in the quick determinism checks
TRACE_RUN = 40  # seed 0's first all-right group comes at step 29
ADVANTAGE_DECIMALS = 2

# mhrl/training/grpo.py's group_advantages, copied from the project unchanged.
PROJECT_GROUP_ADVANTAGES = '''
def group_advantages(rewards: list[float]) -> list[float] | None:
    """Group-relative advantages, or None when the group has no contrast.

    Args:
        rewards: One reward per rollout in the group.

    Returns:
        Standardized advantages, or None if every reward is equal.
    """
    if len(rewards) < 2 or max(rewards) - min(rewards) < Defaults.ADVANTAGE_EPS:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + Defaults.ADVANTAGE_EPS) for r in rewards]
'''

# The two --group outputs, exactly as the README quotes them.
GROUP_1011 = """rewards     [1, 0, 1, 1]
mean        0.75
spread      0.43  (standard deviation)
advantages  (reward - mean) / (spread + 0.0001)
            [+0.58, -1.73, +0.58, +0.58]
"""
GROUP_1111 = """rewards     [1, 1, 1, 1]
no contrast, skipped: every rollout scored the same, so none did
better or worse than the others, and the group teaches nothing
"""
# Seed 0's table, as the README quotes it.
SEED_0_TABLE = """by difficulty, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
rollouts right
easy    45%         88%       95%       95%
medium  11%         50%       82%       95%
hard    0%          0%        0%        0%
no-signal groups (all 4 rewards tied, so skipped)
easy    8/35        18/34     25/32     27/33
medium  21/33       13/37     15/34     29/37
hard    32/32       29/29     34/34     30/30
of those, all wrong + all right
easy    6 + 2       0 + 18    0 + 25    0 + 27
medium  21 + 0      6 + 7     0 + 15    0 + 29
hard    32 + 0      29 + 0    34 + 0    30 + 0
"""


@pytest.fixture(scope="module")
def fair() -> questions.QuestionSet:
    """Build the question set once for the whole file.

    Returns:
        The fair questions, split by city.
    """
    return questions.make_questions()


@pytest.fixture(scope="module")
def seed_0() -> tuple[grpo_groups.RunResult, str]:
    """Run the default settings once (seed 0, 200 steps) for several tests.

    Returns:
        The run's result and everything it printed.
    """
    return quiet()


def quiet(**knobs: float | int | bool) -> tuple[grpo_groups.RunResult, str]:
    """Run grpo_groups.py in-process and capture what it prints.

    Args:
        **knobs: Settings fields to change, e.g. ``steps=30``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = grpo_groups.run(grpo_groups.Settings(**knobs))
    return result, output.getvalue()


def run_script(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run one of this chapter's scripts as a learner would.

    Args:
        script: The file name, e.g. ``"grpo_groups.py"``.
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / script), *flags]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
    )


def function_code(source: str, name: str) -> str:
    """Return a function's code without its docstring, as a syntax tree dump.

    Args:
        source: Python source holding the function.
        name: The function's name.

    Returns:
        The dump of the function's body, docstring left out.
    """
    tree = ast.parse(source)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def chapter_code(path: Path, name: str) -> str:
    """Return a function's code from a file, docstring left out.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The dump of the function's body.
    """
    return function_code(path.read_text(encoding="utf-8"), name)


def play(env: MiniAnalyst, *actions: str) -> list[str]:
    """Send several actions to an environment.

    Args:
        env: The environment.
        *actions: The actions, in order.

    Returns:
        What each call sent back.
    """
    return [env.step(action) for action in actions]


def first_of(fair: questions.QuestionSet, level: str) -> questions.Question:
    """Return the first training question of a level.

    Args:
        fair: The question set.
        level: ``"easy"``, ``"medium"`` or ``"hard"``.

    Returns:
        The question.
    """
    return next(q for q in fair.training if q.level == level)


# --- What is copied ----------------------------------------------------------


def test_grader_and_database_are_chapter_four_copies() -> None:
    """grader.py and shop.db are chapter 4's (chapter 1's), byte for byte."""
    for name in ("grader.py", "data/shop.db"):
        assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_four_only_in_the_chapter_name() -> None:
    """The only change is where outputs go."""
    mine = (CHAPTER / "tasks.py").read_text(encoding="utf-8")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine.replace("05-grpo-groups", "04-episodes") == theirs


@pytest.mark.parametrize(
    "name",
    [
        "softmax",
        "log_prob",
        "onehot",
        "log_prob_gradient",
        "gradient_step",
        "greedy",
        "pick",
        "action_scores",
        "weight_gradient",
        "rollout_loss",
        "rollout_gradient",
    ],
)
def test_the_policy_math_is_chapter_fours_line_for_line(name: str) -> None:
    """Only the features changed; the math runs the same code."""
    mine = chapter_code(CHAPTER / "policy.py", name)
    assert mine == chapter_code(PREVIOUS / "policy.py", name)


def test_group_advantages_is_the_projects_line_for_line() -> None:
    """The same code as mhrl/training/grpo.py, and the same tie threshold."""
    mine = chapter_code(CHAPTER / "grpo.py", "group_advantages")
    assert mine == function_code(PROJECT_GROUP_ADVANTAGES, "group_advantages")
    assert grpo.Defaults.ADVANTAGE_EPS == 1e-4
    assert (grpo.QUESTIONS_PER_STEP, grpo.GROUP_SIZE) == (2, 4)


# --- Group advantages ----------------------------------------------------------


def test_one_wrong_in_four_gives_plus_058_and_minus_173() -> None:
    """[1, 0, 1, 1]: mean 0.75, spread 0.433, so +0.58 for each right one."""
    advantages = grpo.group_advantages([1, 0, 1, 1])
    assert advantages is not None
    assert [round(a, ADVANTAGE_DECIMALS) for a in advantages] == [
        0.58,
        -1.73,
        0.58,
        0.58,
    ]
    assert sum(advantages) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "rewards", [[1, 1, 1, 1], [0, 0, 0, 0], [1], [0.5, 0.50001, 0.5, 0.5]]
)
def test_a_group_that_ties_has_no_contrast(rewards: list[float]) -> None:
    """All the same (or one reward, or closer than 0.0001): None, skipped."""
    assert grpo.group_advantages(rewards) is None
    assert grpo.group_outcome(rewards) != grpo.SIGNAL


def test_outcomes_say_why_a_group_is_silent() -> None:
    """All wrong, all right, or signal."""
    assert grpo.group_outcome([0, 0, 0, 0]) == grpo.ALL_WRONG
    assert grpo.group_outcome([1, 1, 1, 1]) == grpo.ALL_RIGHT
    assert grpo.group_outcome([0, 1, 0, 0]) == grpo.SIGNAL


def test_dividing_by_the_spread_makes_one_success_a_full_push() -> None:
    """One right in four and three right in four push equally hard."""
    rare = grpo.group_advantages([1, 0, 0, 0])
    common = grpo.group_advantages([0, 1, 1, 1])
    assert rare is not None and common is not None
    assert max(rare) == pytest.approx(-min(common))


def test_the_group_flag_prints_the_readme_blocks() -> None:
    """--group works out the README's two groups and stops."""
    assert run_script("grpo_groups.py", "--group", "1,0,1,1").stdout == GROUP_1011
    assert run_script("grpo_groups.py", "--group", "1,1,1,1").stdout == GROUP_1111


# --- The update ---------------------------------------------------------------


def measured_step_gradient(
    items: list[grpo.Trained], weights: np.ndarray
) -> np.ndarray:
    """Measure the step loss's slope for every weight by moving it a hair.

    Args:
        items: The rollouts learned from.
        weights: The weights.

    Returns:
        One measured slope per weight.
    """

    def step_loss(w: np.ndarray) -> float:
        total, calls = 0.0, 0
        for item in items:
            log_probs = [
                policy.log_prob(policy.action_scores(w, x), c)
                for x, c in zip(item.situations, item.choices, strict=True)
            ]
            total += policy.rollout_loss(item.advantage, log_probs)
            calls += len(item.choices)
        return total / calls

    slopes = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        up, down = weights.copy(), weights.copy()
        up[index] += STEP
        down[index] -= STEP
        slopes[index] = (step_loss(up) - step_loss(down)) / (2 * STEP)
    return slopes


def test_the_step_gradient_is_the_measured_slope_of_the_average_loss() -> None:
    """The update is the gradient of the loss averaged over every call."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(15, 6))
    items = []
    for advantage, calls in ((0.58, 8), (-1.73, 3), (1.0, 5)):
        situations = tuple(rng.uniform(size=15) for _ in range(calls))
        choices = tuple(int(c) for c in rng.integers(6, size=calls))
        scores = tuple(policy.action_scores(weights, x) for x in situations)
        log_probs = tuple(
            policy.log_prob(s, c) for s, c in zip(scores, choices, strict=True)
        )
        items.append(grpo.Trained(situations, scores, choices, log_probs, advantage))
    update = grpo.step_update(items, weights.shape)
    assert update.calls == 16
    measured = measured_step_gradient(items, weights)
    assert np.abs(update.gradient - measured).max() < MEASURED_TOLERANCE


def test_a_step_with_no_contrast_changes_nothing() -> None:
    """Nothing to learn from: a zero gradient and no loss."""
    update = grpo.step_update([], (15, 6))
    assert update.calls == 0 and update.loss is None
    assert not update.gradient.any()


# --- Questions and the environment ------------------------------------------------


def test_three_levels_two_questions_each_split_by_city(
    fair: questions.QuestionSet,
) -> None:
    """45 to train on (15, 16, 14) and 19 held out (8, 7, 4)."""
    assert questions.count_by_level(fair.training) == {
        "easy": 15,
        "medium": 16,
        "hard": 14,
    }
    assert questions.count_by_level(fair.held_out) == {
        "easy": 8,
        "medium": 7,
        "hard": 4,
    }
    assert fair.test_cities == ("Brussels", "Lisbon", "Prague", "Warsaw")
    assert not {q.city for q in fair.training} & set(fair.test_cities)
    for level in questions.LEVELS:
        assert sum(t.level == level for t in questions.TEMPLATES) == 2


def test_harder_questions_need_more_tables() -> None:
    """The right query needs 1 table (easy), 2 (medium) or all 4 (hard)."""
    needed = {"easy": 1, "medium": 2, "hard": 4}
    for template in questions.TEMPLATES:
        right = template.candidates[0].sql
        assert environment.tables_needed(right) == needed[template.level]


def test_look_shows_one_table_at_a_time_following_the_links(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """customers, orders, order_items, products, then nothing new."""
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(fair.training[0], folder)
        shown = play(env, LOOK, LOOK, LOOK, LOOK, LOOK)
    assert shown[0] == "customers: id, name, city, signup_date"
    assert shown[1] == "orders: id, customer_id, order_date, payment_method"
    assert [s.split(":")[0] for s in shown[2:4]] == ["order_items", "products"]
    assert shown[4] == environment.ALL_SEEN
    assert env.state.tables_seen == 4


@pytest.mark.parametrize(("level", "looks"), [("easy", 1), ("medium", 2), ("hard", 4)])
def test_a_query_works_only_once_its_tables_are_seen(
    fair: questions.QuestionSet, tmp_path: Path, level: str, looks: int
) -> None:
    """One LOOK too few: SQLite's no such table; then the right answer."""
    question = first_of(fair, level)
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(question, folder)
        play(env, *[LOOK] * (looks - 1))
        too_early = env.step(QUERY_A)
        env.step(LOOK)
        on_time = env.step(QUERY_A)
    assert too_early.startswith("no such table: ")
    assert on_time == question.task.answer


def test_the_reward_still_comes_only_at_the_end(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """A hard question solved in six calls: reward 1 only at FINISH."""
    question = first_of(fair, "hard")
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(question, folder)
        play(env, LOOK, LOOK, LOOK, LOOK, QUERY_A, WRITE)
        assert env.reward is None and not env.done
        assert env.step(FINISH) == "correct · reward 1"


# --- The policy: one set of weights per level ---------------------------------------


def test_each_level_uses_only_its_own_weights(fair: questions.QuestionSet) -> None:
    """Five features in the level's slot, zeros elsewhere; shared gives five."""
    for index, level in enumerate(questions.LEVELS):
        state = environment.State(first_of(fair, level), calls=2, tables_seen=1)
        slots = policy.features(state).reshape(3, 5)
        assert slots[index].tolist() == [1.0, 0.25, 0.0, 0.0, 0.25]
        assert not np.delete(slots, index, axis=0).any()
        assert policy.features(state, shared=True).tolist() == slots[index].tolist()
    assert (policy.weight_rows(), policy.weight_rows(shared=True)) == (15, 5)


def test_features_never_mention_the_city(fair: questions.QuestionSet) -> None:
    """Two questions of the same level in the same situation look identical."""
    one, other = [q for q in fair.training if q.level == "medium"][:2]
    assert one.city != other.city
    first = policy.features(environment.State(one, calls=3, tables_seen=2))
    second = policy.features(environment.State(other, calls=3, tables_seen=2))
    assert np.array_equal(first, second)


# --- Whole runs -------------------------------------------------------------------


def test_hard_questions_never_give_signal_and_their_weights_stay_zero(
    seed_0: tuple[grpo_groups.RunResult, str],
) -> None:
    """Seed 0: every hard group is all wrong, so nothing ever changes them."""
    result, _ = seed_0
    hard = [g for s in result.steps for g in s.groups if g.question.level == "hard"]
    assert hard and all(g.outcome == grpo.ALL_WRONG for g in hard)
    assert not grpo_groups.level_weights(result.weights, "hard", False).any()
    scores = {score.level: score for score in result.scores}
    assert scores["hard"].training == 0 and scores["hard"].untrained


def test_seed_0_table_matches_the_readme(
    seed_0: tuple[grpo_groups.RunResult, str],
) -> None:
    """The table printed for seed 0, and what the greedy agent learned."""
    result, printed = seed_0
    assert SEED_0_TABLE in printed
    scores = {score.level: score for score in result.scores}
    assert (scores["easy"].training, scores["easy"].held_out) == (15, 8)
    assert (scores["medium"].training, scores["medium"].held_out) == (16, 7)
    assert "what the group mean saves" in printed


def test_a_skipped_group_trains_nothing(
    seed_0: tuple[grpo_groups.RunResult, str],
) -> None:
    """A step whose two groups both tied learns from no call at all."""
    result, _ = seed_0
    silent = [s for s in result.steps if s.no_signal == len(s.groups)]
    assert silent and all(s.calls == 0 and s.loss is None for s in silent)
    loud = [s for s in result.steps if s.no_signal < len(s.groups)]
    assert all(s.calls > 0 for s in loud)


def test_metrics_has_one_line_per_step_with_the_projects_fields(
    seed_0: tuple[grpo_groups.RunResult, str],
) -> None:
    """One JSON line per step: enough to redraw every curve."""
    result, _ = seed_0
    lines = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == grpo_groups.DEFAULT_STEPS
    first = json.loads(lines[0])
    for field in (
        "stage",
        "step",
        "accuracy",
        "mean_reward",
        "mean_calls",
        "groups_without_contrast",
        "groups",
        "group_levels",
        "group_rewards",
    ):
        assert field in first, field
    assert [json.loads(line)["step"] for line in lines] == list(range(1, 201))


def test_plot_redraws_the_run_from_the_metrics_file_alone(
    seed_0: tuple[grpo_groups.RunResult, str], tmp_path: Path
) -> None:
    """plot.py reads only metrics.jsonl, even after it is moved elsewhere."""
    result, _ = seed_0
    moved = tmp_path / "elsewhere" / "metrics.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(result.metrics_path.read_bytes())
    finished = run_script("plot.py", str(moved))
    assert finished.returncode == 0, finished.stderr
    printed = finished.stdout
    assert printed.startswith("200 steps, 400 groups;")
    for title in ("correct, by difficulty", "groups with no signal", "calls per"):
        assert title in printed
    assert printed == run_script("plot.py", str(result.metrics_path)).stdout


def test_plot_refuses_a_missing_or_foreign_file(tmp_path: Path) -> None:
    """A plain message and a non-zero exit, never a traceback."""
    missing = run_script("plot.py", str(tmp_path / "nope.jsonl"))
    foreign = tmp_path / "other.jsonl"
    foreign.write_text('{"hello": 1}\n', encoding="utf-8")
    other = run_script("plot.py", str(foreign))
    for finished in (missing, other):
        assert finished.returncode != 0
        assert "Traceback" not in finished.stderr
        assert finished.stderr.startswith("plot.py: ")


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical logs and output; seed 1 differs."""
    first, first_text = quiet(steps=SHORT_RUN)
    again, again_text = quiet(steps=SHORT_RUN)
    other, _ = quiet(steps=SHORT_RUN, seed=1)
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
    assert names[:5] == [
        "step-001-group-1-rollout-1",
        "step-001-group-1-rollout-2",
        "step-001-group-1-rollout-3",
        "step-001-group-1-rollout-4",
        "step-001-group-2-rollout-1",
    ]
    assert len(set(names)) == 16
    work = course_home / "05-grpo-groups" / "work"
    assert tasks.work_root() == work and list(work.iterdir()) == []


def test_the_trace_prints_three_groups_in_full() -> None:
    """The first group, the first with one wrong, the first all right."""
    printed = run_script(
        "grpo_groups.py", "--trace", "--fast", "--steps", str(TRACE_RUN)
    ).stdout
    groups = [line for line in printed.splitlines() if " · group " in line]
    assert [line.split(" · ")[0] for line in groups] == [
        "step 1",
        "step 22",
        "step 29",
    ]
    assert "[+0.58, +0.58, +0.58, -1.73]" in printed
    assert "rewards [1, 1, 1, 1] · no contrast, skipped" in printed


def test_seeds_script_prints_a_row_per_seed_and_the_totals() -> None:
    """Two short seeds: one row each, then the table added up."""
    finished = run_script("seeds.py", "--seeds", "2", "--steps", "8")
    assert finished.returncode == 0, finished.stderr
    rows = [
        line for line in finished.stdout.splitlines() if re.match(r"\s*\d+\s", line)
    ]
    assert [row.split()[0] for row in rows] == ["0", "1"]
    assert "all 2 seeds together" in finished.stdout


def test_shared_weights_use_one_set_for_every_level() -> None:
    """--shared: 5 rows of weights and its own log folder."""
    result, printed = quiet(steps=4, shared=True)
    assert result.weights.shape == (policy.weight_rows(shared=True), 6)
    assert result.metrics_path.parent.name == "shared-seed0-lr3-steps4"
    assert "one set of weights for every level" in printed


@pytest.mark.parametrize(
    ("steps", "edges"),
    [
        (200, (0, 50, 100, 150, 200)),
        (10, (0, 3, 6, 9, 10)),
        (5, (0, 2, 4, 5)),
        (1, (0, 1)),
    ],
)
def test_stretches_split_any_run_into_at_most_four(
    steps: int, edges: tuple[int, ...]
) -> None:
    """Quarters when the steps divide by 4; fewer, shorter ones otherwise."""
    assert grpo_groups.stretch_edges(steps) == edges


def test_plot_keeps_its_copied_numbers_equal_to_the_chapters() -> None:
    """plot.py copies three numbers so it can stand alone; they must agree."""
    assert plot.BUDGET == environment.BUDGET
    assert plot.TIE_EPS == grpo.Defaults.ADVANTAGE_EPS
    assert plot.CORRECT_REWARD == grpo.CORRECT_REWARD


def test_plot_draws_a_one_step_run(tmp_path: Path) -> None:
    """A run of one step is one column, labelled with its step."""
    result, _ = quiet(steps=1)
    finished = run_script("plot.py", str(result.metrics_path))
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.startswith("1 steps, 2 groups;")


@pytest.mark.parametrize(
    "line",
    [
        '{"step": 1, "mean_calls": 5, "group_levels": ["easy"], "group_rewards": [[]]}',
        '{"step": 1, "mean_calls": 5, "group_levels": [], "group_rewards": [[1, 0]]}',
    ],
)
def test_plot_refuses_a_step_with_a_broken_group(tmp_path: Path, line: str) -> None:
    """An empty group, or groups without levels: a plain message, exit 1."""
    broken = tmp_path / "metrics.jsonl"
    broken.write_text(line + "\n", encoding="utf-8")
    finished = run_script("plot.py", str(broken))
    assert finished.returncode == 1
    assert (
        finished.stderr.startswith("plot.py: ") and "Traceback" not in finished.stderr
    )


@pytest.mark.parametrize("flags", [("--seeds", "0"), ("--steps", "0"), ("--lr", "-1")])
def test_seeds_refuses_bad_flags(flags: tuple[str, str]) -> None:
    """seeds.py checks its flags like grpo_groups.py does."""
    finished = run_script("seeds.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


@pytest.mark.parametrize(
    "flags",
    [
        ("--lr", "-1"),
        ("--lr", "nan"),
        ("--steps", "0"),
        ("--seed", "-1"),
        ("--group", "1,x"),
    ],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, str]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script("grpo_groups.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr
