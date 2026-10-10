"""Chapter 6: the importance ratio, the clip, stale and reused batches, the bug."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("06-ratio-and-clip")

import grpo  # noqa: E402
import grpo_clip  # noqa: E402
import plot  # noqa: E402
import policy  # noqa: E402
import real_runs  # noqa: E402
import report  # noqa: E402
import tasks  # noqa: E402

PREVIOUS = ROOT / "05-grpo-groups"
SCRIPT_TIMEOUT_S = 300
STEP = 1e-6  # how far a weight is moved to measure a slope
MEASURED_TOLERANCE = 1e-6
KINK_MARGIN = 1e-3  # measured slopes stay away from the clip's two edges
SHORT_RUN = 30  # steps in the quick determinism checks
COLLAPSE_QUARTER = 3  # the last quarter of a run, counting from 0
CLIPPED_MARK_LINE = re.compile(r"\|\s*c")  # a chart row that drew clipped ratios

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

# mhrl/training/grpo.py's loss_fn, copied from the project unchanged. It is a
# closure there: model and eps come from the function around it.
PROJECT_LOSS_FN = """
def loss_fn(item: TrainItem) -> tuple[mx.array, mx.array]:
    new = token_logprobs(model, item)
    ratio = mx.exp(new - mx.array(item.old_logprobs))
    unclipped = ratio * item.advantage
    clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
    return -mx.minimum(unclipped, clipped).sum(), mx.stop_gradient(ratio)
"""
# What changes from MLX to NumPy, and from the real model to the toy.
TRANSLATION = (
    ("mx.stop_gradient(ratio)", "ratio"),  # NumPy has no gradient to stop
    ("mx.", "np."),
    ("token_logprobs(model, item)", "call_logprobs(weights, item)"),
)

# --token 0.20,0.30, exactly as the README quotes it.
TOKEN_020_030 = """one call: its action had p 0.20 when the rollout was sampled (old)
and has p 0.30 with the learner's weights now (new)
ratio = new p / old p = 0.30 / 0.20 = 1.50
      = exp(new log p - old log p) = exp(-1.20 - (-1.61)) = 1.50
clip(ratio, 0.8, 1.2) = 1.20

A = +1 (the rollout did better than its group)
  ratio x A         = +1.50
  clip(ratio) x A   = +1.20
  min of the two    = +1.20  the clipped one
  loss = -min       = -1.20
  push on log p      0.00  the clip has switched it off

A = -1 (the rollout did worse than its group)
  ratio x A         = -1.50
  clip(ratio) x A   = -1.20
  min of the two    = -1.50  the unclipped one
  loss = -min       = +1.50
  push on log p     -1.50  ratio x A: 1.50 times chapter 5's push, which was A

min() keeps the more pessimistic number: a call that has already moved
20% the way its advantage wants gets no more push; a call that moved the
wrong way always gets its whole push, ratio x A
"""


def quiet(**knobs: float | int | bool) -> tuple[grpo_clip.RunResult, str]:
    """Run grpo_clip.py in-process and capture what it prints.

    Args:
        **knobs: Settings fields to change, e.g. ``steps=30``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = grpo_clip.run(grpo_clip.Settings(**knobs))
    return result, output.getvalue()


@pytest.fixture(scope="module")
def module_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point COURSE_HOME at a temporary folder for the runs this file shares.

    conftest.py's course_home only starts with each test, after the shared
    runs below have already been made; without this they would write into
    the real course cache.

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
def fresh(module_home: Path) -> tuple[grpo_clip.RunResult, str]:
    """Run the default settings once (seed 0, 200 steps) for several tests.

    Args:
        module_home: The temporary course home the run writes into.

    Returns:
        The run's result and everything it printed.
    """
    return quiet()


@pytest.fixture(scope="module")
def stale(module_home: Path) -> tuple[grpo_clip.RunResult, str]:
    """Run ``--stale 2`` once.

    Args:
        module_home: The temporary course home the run writes into.

    Returns:
        The run's result and everything it printed.
    """
    return quiet(stale=2)


@pytest.fixture(scope="module")
def reused(module_home: Path) -> tuple[grpo_clip.RunResult, str]:
    """Run ``--updates-per-batch 8`` once.

    Args:
        module_home: The temporary course home the run writes into.

    Returns:
        The run's result and everything it printed.
    """
    return quiet(updates=8)


def run_script(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run one of this chapter's scripts as a learner would.

    Args:
        script: The file name, e.g. ``"grpo_clip.py"``.
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
    return ast.dump(ast.Module(body=function_body(source, name), type_ignores=[]))


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


def chapter_code(path: Path, name: str) -> str:
    """Return a function's code from a file, docstring left out.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The dump of the function's body.
    """
    return function_code(path.read_text(encoding="utf-8"), name)


def one_call(
    old: float, new: float, advantage: float
) -> tuple[grpo.Trained, np.ndarray]:
    """Build a one-call rollout whose action has p ``new`` under the weights.

    Two actions and one feature: the weights are the two scores themselves.

    Args:
        old: The action's probability when it was sampled.
        new: Its probability with the weights returned.
        advantage: The rollout's advantage.

    Returns:
        The rollout and the weights.
    """
    weights = np.array([[math.log(new), math.log(1 - new)]])
    item = grpo.Trained((np.array([1.0]),), (0,), (math.log(old),), advantage)
    return item, weights


def random_rollouts(
    rng: np.random.Generator, weights: np.ndarray, spread: float
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
    for advantage, calls in ((0.58, 8), (-1.73, 3), (1.0, 5)):
        situations = tuple(rng.uniform(size=weights.shape[0]) for _ in range(calls))
        choices = tuple(int(c) for c in rng.integers(weights.shape[1], size=calls))
        new = [
            policy.log_prob(policy.action_scores(weights, x), c)
            for x, c in zip(situations, choices, strict=True)
        ]
        old = tuple(float(n + rng.uniform(-spread, spread)) for n in new)
        items.append(grpo.Trained(situations, choices, old, advantage))
    return items


def measured_gradient(
    items: list[grpo.Trained], weights: np.ndarray, clip: bool
) -> np.ndarray:
    """Measure the step loss's slope for every weight by moving it a hair.

    Args:
        items: The rollouts learned from.
        weights: The weights.
        clip: Whether the clip is on.

    Returns:
        One measured slope per weight.
    """
    loss = grpo.loss_fn if clip else grpo.unclipped_loss_fn
    calls = sum(len(item.choices) for item in items)

    def step_loss(w: np.ndarray) -> float:
        return sum(float(loss(w, item)[0]) for item in items) / calls

    slopes = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        up, down = weights.copy(), weights.copy()
        up[index] += STEP
        down[index] -= STEP
        slopes[index] = (step_loss(up) - step_loss(down)) / (2 * STEP)
    return slopes


def readme() -> str:
    """Return the chapter's README.

    Returns:
        Its text.
    """
    return (CHAPTER / "README.md").read_text(encoding="utf-8")


# --- What is copied ----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "grader.py",
        "environment.py",
        "policy.py",
        "questions.py",
        "data/shop.db",
    ],
)
def test_the_toy_is_chapter_fives_byte_for_byte(name: str) -> None:
    """The agent, its questions and its policy are chapter 5's, unchanged."""
    assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_five_only_in_the_chapter_name() -> None:
    """The only change is where outputs go."""
    mine = (CHAPTER / "tasks.py").read_text(encoding="utf-8")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine.replace("06-ratio-and-clip", "05-grpo-groups") == theirs


def test_group_advantages_is_the_projects_line_for_line() -> None:
    """The same code as mhrl/training/grpo.py, and chapter 5's."""
    mine = chapter_code(CHAPTER / "grpo.py", "group_advantages")
    assert mine == function_code(PROJECT_GROUP_ADVANTAGES, "group_advantages")
    assert mine == chapter_code(PREVIOUS / "grpo.py", "group_advantages")
    assert grpo.Defaults.ADVANTAGE_EPS == 1e-4
    assert (grpo.QUESTIONS_PER_STEP, grpo.GROUP_SIZE) == (2, 4)


def test_loss_fn_is_the_projects_line_for_line_in_numpy() -> None:
    """mhrl/training/grpo.py's five lines, with np for mx and the toy's names."""
    project = PROJECT_LOSS_FN
    for theirs, ours in TRANSLATION:
        project = project.replace(theirs, ours)
    expected = function_body(project, "loss_fn")
    source = (CHAPTER / "grpo.py").read_text(encoding="utf-8")
    first, *body = function_body(source, "loss_fn")
    # The project sets eps once outside its closure; here it is the first line.
    assert ast.unparse(first) == "eps = Defaults.CLIP_EPSILON"
    assert [ast.dump(s) for s in body] == [ast.dump(s) for s in expected]
    assert grpo.Defaults.CLIP_EPSILON == 0.2


# --- One call ----------------------------------------------------------------


def test_the_worked_token_gives_ratio_15_clipped_to_12() -> None:
    """Old p 0.20, new p 0.30: ratio 1.5; A > 0 is clipped at 1.2, A < 0 is not."""
    item, weights = one_call(0.20, 0.30, advantage=1.0)
    loss, ratio = grpo.loss_fn(weights, item)
    assert ratio[0] == pytest.approx(1.5)
    assert loss == pytest.approx(-1.2)
    assert grpo.clip_active(ratio, 1.0).all()
    assert not grpo.loss_gradient(weights, item).any()  # the push is switched off
    worse, _ = one_call(0.20, 0.30, advantage=-1.0)
    loss, ratio = grpo.loss_fn(weights, worse)
    assert loss == pytest.approx(1.5)
    assert not grpo.clip_active(ratio, -1.0).any()
    assert grpo.loss_gradient(weights, worse).any()


@pytest.mark.parametrize(
    ("ratio", "advantage", "clipped"),
    [
        (1.5, 1.0, True),  # moved the advantage's way by more than 20%
        (1.5, -1.0, False),  # moved the wrong way: never clipped
        (0.5, -1.0, True),
        (0.5, 1.0, False),
        (1.1, 1.0, False),  # inside the range
        (1.2, 1.0, False),  # on the edge: clip(1.2) is 1.2, nothing changes
        (1.5, 0.0, False),  # no advantage, no push to switch off
    ],
)
def test_the_clip_acts_only_on_moves_the_advantage_wants(
    ratio: float, advantage: float, clipped: bool
) -> None:
    """min() picks the flat clipped term only on the advantage's side."""
    assert bool(grpo.clip_active(np.array([ratio]), advantage)[0]) is clipped


def test_outside_the_range_is_not_the_same_as_clipped() -> None:
    """Below 0.8 or above 1.2 is outside, whatever the advantage."""
    ratios = np.array([0.79, 0.8, 1.0, 1.2, 1.21])
    assert grpo.outside_clip_range(ratios).tolist() == [True, False, False, False, True]


def test_the_token_flag_prints_the_readme_block() -> None:
    """--token works out the README's call and stops."""
    finished = run_script("grpo_clip.py", "--token", "0.20,0.30")
    assert finished.stdout == TOKEN_020_030
    assert TOKEN_020_030 in readme()


# --- The gradient ---------------------------------------------------------------


@pytest.mark.parametrize("clip", [True, False])
def test_the_hand_gradient_is_the_measured_slope_of_the_loss(clip: bool) -> None:
    """With ratios spread far enough that some calls are clipped."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(15, 6))
    items = random_rollouts(rng, weights, spread=0.5)
    ratios = np.concatenate([grpo.loss_fn(weights, item)[1] for item in items])
    # A slope measured across a kink is meaningless; this seed has none near.
    edges = np.array([1 - grpo.Defaults.CLIP_EPSILON, 1 + grpo.Defaults.CLIP_EPSILON])
    assert np.abs(ratios[:, None] - edges).min() > KINK_MARGIN
    update = grpo.step_update(items, weights, clip)
    assert update.clipped.any() == clip
    measured = measured_gradient(items, weights, clip)
    assert np.abs(update.gradient - measured).max() < MEASURED_TOLERANCE


def test_with_every_ratio_1_the_gradient_is_chapter_fives_exactly() -> None:
    """A fresh batch: the clipped loss's slope is REINFORCE's, number for number."""
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(15, 6))
    for item in random_rollouts(rng, weights, spread=0.0):
        scores = [policy.action_scores(weights, x) for x in item.situations]
        chapter_5 = policy.rollout_gradient(
            item.situations, scores, item.choices, item.advantage
        )
        assert np.array_equal(grpo.loss_gradient(weights, item), chapter_5)
        assert np.all(grpo.loss_fn(weights, item)[1] == 1.0)


def test_an_update_with_nothing_to_learn_changes_nothing() -> None:
    """No rollouts: a zero gradient, no loss and no ratios."""
    update = grpo.step_update([], np.zeros((15, 6)))
    assert update.calls == 0 and update.loss is None
    assert not update.gradient.any() and update.ratios.size == 0


# --- Whole runs -------------------------------------------------------------------


def test_the_default_run_is_chapter_fives_run(
    fresh: tuple[grpo_clip.RunResult, str], tmp_path: Path
) -> None:
    """One fresh batch, one update: chapter 5's rollouts, steps and weights."""
    result, _ = fresh
    finished = subprocess.run(
        [sys.executable, str(PREVIOUS / "grpo_groups.py"), "--fast"],
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
    )
    assert finished.returncode == 0, finished.stderr
    theirs = tasks.course_home() / "05-grpo-groups" / "runs" / "seed0-lr3-steps200"
    old = [json.loads(line) for line in (theirs / "metrics.jsonl").open()]
    new = [json.loads(line) for line in result.metrics_path.open()]
    for field in (
        "accuracy",
        "mean_reward",
        "mean_calls",
        "groups_without_contrast",
        "group_rewards",
        "train_calls",
    ):
        assert [o[field] for o in old] == [n[field] for n in new], field
    old_log = (theirs / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    new_log = result.rollouts_path.read_text(encoding="utf-8").splitlines()
    assert old_log[-1] == new_log[-1]  # the final weights
    old_calls = [json.loads(line).get("calls") for line in old_log]
    assert old_calls == [json.loads(line).get("calls") for line in new_log]


def test_fresh_batches_give_ratio_exactly_1_and_clip_nothing(
    fresh: tuple[grpo_clip.RunResult, str],
) -> None:
    """Every step that learns prints ratio 1.000 · clipped 0%; the rest n/a."""
    result, printed = fresh
    ratios = np.concatenate([step.ratios for step in result.steps])
    assert ratios.size and np.all(ratios == 1.0)
    assert not any(step.clipped.any() for step in result.steps)
    lines = [line for line in printed.splitlines() if line.startswith("step ")]
    for step, line in zip(result.steps, lines, strict=True):
        expected = "ratio 1.000 · clipped 0%" if step.calls else "ratio n/a"
        assert line.split(" · ratio ")[1].startswith(expected.removeprefix("ratio "))


@pytest.mark.parametrize("run", ["fresh", "stale"])
def test_the_end_of_the_run_matches_the_readme(
    run: str, request: pytest.FixtureRequest
) -> None:
    """The ratio table, the share right and the greedy test, as quoted."""
    _, printed = request.getfixturevalue(run)
    start = printed.index("ratios the loss saw")
    end = printed.index("\n\nwrote ")
    assert printed[start:end] in readme()


def test_stale_samples_move_the_ratio_and_the_clip_acts(
    stale: tuple[grpo_clip.RunResult, str],
) -> None:
    """--stale 2: the sampler lags by up to 2 updates; ratios leave 0.8-1.2."""
    result, printed = stale
    assert max(step.lag for step in result.steps) == 2
    assert result.steps[0].lag == 0
    ratios = np.concatenate([step.ratios for step in result.steps])
    assert grpo.outside_clip_range(ratios).mean() > 0.2
    assert any(step.clipped.any() for step in result.steps)
    assert "sampled with the weights from 2 updates earlier" in printed


def test_recompute_old_reads_1_while_the_sampler_is_stale() -> None:
    """The bug: the loss sees 1.000 everywhere; the recorded p say otherwise."""
    result, printed = quiet(stale=2, recompute_old=True, steps=60)
    ratios = np.concatenate([step.ratios for step in result.steps])
    assert ratios.size and np.all(ratios == 1.0)
    hidden = np.concatenate(
        [
            use.recorded_ratios
            for step in result.steps
            for use in step.uses
            if use.recorded_ratios is not None
        ]
    )
    assert hidden.size == ratios.size
    assert grpo.outside_clip_range(hidden).mean() > 0.2
    assert "what --recompute-old hid" in printed


def test_reusing_a_batch_spreads_the_ratios_after_the_first_use(
    reused: tuple[grpo_clip.RunResult, str],
) -> None:
    """Use 1 is fresh (all 1); every later use has ratios outside and clipped."""
    result, _ = reused
    rows = report.use_rows(result.steps)
    assert len(rows) == 8
    first_ratios, first_clipped = rows[0].ratios, rows[0].clipped
    assert np.all(first_ratios == 1.0) and not first_clipped.any()
    for _, ratios, clipped in rows[1:]:
        assert grpo.outside_clip_range(ratios).mean() > 0.2
        assert clipped.mean() > 0.15


def test_without_the_clip_seed_0_collapses_and_never_recovers() -> None:
    """8 updates per batch, no clip: nothing right in the last quarter."""
    result, printed = quiet(updates=8, clip=False)
    table = report.right_by_quarter(result.steps, result.edges)
    assert all(table[level][COLLAPSE_QUARTER][0] == 0 for level in table)
    assert not any(step.clipped.any() for step in result.steps)
    assert "no clip" in printed and "clipped 0%" not in printed


def test_metrics_has_one_line_per_step_with_the_projects_fields(
    reused: tuple[grpo_clip.RunResult, str],
) -> None:
    """The project's names where it has them, and one record per use."""
    result, _ = reused
    lines = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == grpo_clip.DEFAULT_STEPS
    first = json.loads(lines[0])
    for field in (
        "stage",
        "step",
        "accuracy",
        "mean_reward",
        "mean_calls",
        "groups_without_contrast",
        "importance_ratio_mean",
        "ratio_calls",
        "ratios_outside",
        "clipped_calls",
        "sampler_lag",
    ):
        assert field in first, field
    assert [use["use"] for use in first["uses"]] == list(range(1, 9))
    assert result.metrics_path.parent.name == "seed0-lr3-steps200-updates8-stale0"


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical logs and output; seed 1 differs."""
    first, first_text = quiet(steps=SHORT_RUN, stale=2)
    again, again_text = quiet(steps=SHORT_RUN, stale=2)
    other, _ = quiet(steps=SHORT_RUN, stale=2, seed=1)
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
    work = course_home / "06-ratio-and-clip" / "work"
    assert tasks.work_root() == work and list(work.iterdir()) == []


def test_the_trace_prints_the_first_clipped_call() -> None:
    """--stale 2 --trace: one call in full, with its old and new p."""
    printed = run_script(
        "grpo_clip.py", "--stale", "2", "--trace", "--fast", "--steps", "20"
    ).stdout
    assert printed.count("the first call the clip switched off") == 1
    match = re.search(r"old p ([\d.]+) · new p ([\d.]+) · ratio ([\d.]+)", printed)
    assert match is not None
    old, new, ratio = (float(x) for x in match.groups())
    assert ratio == pytest.approx(new / old, abs=0.02)


def test_the_trace_says_so_when_nothing_was_clipped() -> None:
    """A fresh run never clips, and --trace says that instead of staying silent."""
    printed = run_script("grpo_clip.py", "--trace", "--fast", "--steps", "5").stdout
    assert "the clip switched off no call in this run" in printed
    no_clip = run_script(
        "grpo_clip.py", "--trace", "--no-clip", "--fast", "--steps", "5"
    ).stdout
    assert "the clip is off, so it switched off no call" in no_clip


@pytest.mark.parametrize(
    ("ratio", "place"),
    [(0.5, 0), (0.8, 1), (0.95, 2), (1.0, 3), (1.05, 4), (1.2, 5), (1.21, 6)],
)
def test_the_ratio_table_bins(ratio: float, place: int) -> None:
    """Seven bins; an edge counts in the bin above, except 1.2 itself."""
    counts = report.histogram(np.array([ratio]))
    assert counts.index(1) == place and sum(counts) == 1


# --- plot.py, seeds.py and real_runs.py ------------------------------------------


def test_plot_redraws_the_run_from_the_metrics_file_alone(
    stale: tuple[grpo_clip.RunResult, str], tmp_path: Path
) -> None:
    """plot.py reads only metrics.jsonl, even after it is moved elsewhere."""
    result, _ = stale
    moved = tmp_path / "elsewhere" / "metrics.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(result.metrics_path.read_bytes())
    finished = run_script("plot.py", str(moved))
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.startswith("200 steps, ")
    for title in ("correct, by difficulty", "ratios outside 0.8-1.2"):
        assert title in finished.stdout
    assert CLIPPED_MARK_LINE.search(finished.stdout)
    assert finished.stdout == run_script("plot.py", str(result.metrics_path)).stdout


def test_plot_refuses_a_missing_or_foreign_file(tmp_path: Path) -> None:
    """A plain message and a non-zero exit, never a traceback."""
    missing = run_script("plot.py", str(tmp_path / "nope.jsonl"))
    foreign = tmp_path / "other.jsonl"
    foreign.write_text('{"hello": 1}\n', encoding="utf-8")
    other = run_script("plot.py", str(foreign))
    impossible = tmp_path / "metrics.jsonl"
    impossible.write_text(
        json.dumps(
            {
                "step": 1,
                "group_levels": ["easy"],
                "group_rewards": [[1, 0, 0, 0]],
                "ratio_calls": 3,
                "ratios_outside": 1,
                "clipped_calls": 2,  # more clipped than outside: not ours
            }
        )
        + "\n",
        encoding="utf-8",
    )
    broken = run_script("plot.py", str(impossible))
    for finished in (missing, other, broken):
        assert finished.returncode != 0
        assert "Traceback" not in finished.stderr
        assert finished.stderr.startswith("plot.py: ")


def test_plot_keeps_its_copied_numbers_equal_to_the_chapters() -> None:
    """plot.py copies two numbers so it can stand alone; they must agree."""
    assert plot.CORRECT_REWARD == grpo.CORRECT_REWARD
    assert plot.CLIP_EPSILON == grpo.Defaults.CLIP_EPSILON


def test_plot_draws_a_run_that_never_learned(tmp_path: Path) -> None:
    """One silent step: no ratios, an empty ratio chart and no crash."""
    result, _ = quiet(steps=1, seed=1)
    finished = run_script("plot.py", str(result.metrics_path))
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.startswith("1 steps, ")


def test_seeds_script_prints_a_row_per_seed_and_the_totals() -> None:
    """Two short seeds: one row each in every table, then the totals."""
    finished = run_script("seeds.py", "--seeds", "2", "--steps", "8")
    assert finished.returncode == 0, finished.stderr
    for number in range(1, 6):
        assert f"\n{number}. " in finished.stdout
    rows = [line for line in finished.stdout.splitlines() if line.startswith(" 1 ")]
    assert len(rows) == 5
    assert "fresh: " in finished.stdout and "ratios exactly 1" in finished.stdout


def test_real_runs_print_the_projects_two_ranges() -> None:
    """The closing card: 0.995-1.000 and 0.999-1.001, from the bundled logs."""
    finished = run_script("real_runs.py")
    assert finished.returncode == 0, finished.stderr
    assert "range 0.995-1.000  0.999-1.001" in finished.stdout
    attempt_1 = real_runs.read_ratios(real_runs.RUNS["attempt 1"])
    attempt_2 = real_runs.read_ratios(real_runs.RUNS["attempt 2"])
    assert (len(attempt_1), len(attempt_2)) == (8, 6)


def test_real_runs_refuses_a_log_without_ratios(tmp_path: Path) -> None:
    """A step without importance_ratio_mean is refused by name."""
    broken = tmp_path / "metrics.jsonl"
    broken.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(real_runs.RealRunsError):
        real_runs.read_ratios(broken)
    with pytest.raises(real_runs.RealRunsError):
        real_runs.read_ratios(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(real_runs.RealRunsError):
        real_runs.read_ratios(empty)


@pytest.mark.parametrize(
    "flags",
    [
        ("--token", "0.2"),
        ("--token", "0,0.3"),
        ("--token", "0.2,x"),
        ("--updates-per-batch", "0"),
        ("--stale", "-1"),
        ("--lr", "nan"),
        ("--steps", "0"),
        ("--seed", "-1"),
    ],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, str]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script("grpo_clip.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


@pytest.mark.parametrize(
    "flags", [("--seeds", "0"), ("--steps", "0"), ("--lr", "-1"), ("--lr", "0.5")]
)
def test_seeds_refuses_bad_flags(flags: tuple[str, str]) -> None:
    """seeds.py checks its flags like grpo_clip.py does."""
    finished = run_script("seeds.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr
