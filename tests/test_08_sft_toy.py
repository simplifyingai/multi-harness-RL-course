"""Chapter 8: imitation (SFT) on the toy, its teachers, its traps, and RL after it."""

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

CHAPTER = use_chapter("08-sft-toy")

import environment  # noqa: E402
import evaluate  # noqa: E402
import knobs  # noqa: E402
import plot  # noqa: E402
import policy  # noqa: E402
import questions  # noqa: E402
import real_sft  # noqa: E402
import report  # noqa: E402
import rl  # noqa: E402
import seeds  # noqa: E402
import sft  # noqa: E402
import sft_toy  # noqa: E402
import tasks  # noqa: E402
import teacher  # noqa: E402

PREVIOUS = ROOT / "06-ratio-and-clip"
SCRIPT_TIMEOUT_S = 300
STEP = 1e-6  # how far a weight is moved to measure a slope
MEASURED_TOLERANCE = 1e-6
SHORT_RL = 8  # RL steps in the quick runs

# mhrl/training/common.py's batches, copied from the project unchanged.
PROJECT_BATCHES = '''
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
'''

# mhrl/training/sft.py's loss_fn (a closure there: model comes from around it).
PROJECT_LOSS_FN = """
def loss_fn(item: TrainItem) -> mx.array:
    return -token_logprobs(model, item).sum()
"""
LOSS_TRANSLATION = (
    ("item: TrainItem) -> mx.array", "weights: np.ndarray, item: TrainItem) -> float"),
    ("token_logprobs(model, item)", "call_logprobs(weights, item)"),
)

# mhrl/training/common.py's item_from_sequence picks the trained positions so;
# here the mask is a field of the sequence.
PROJECT_POSITIONS = "positions = [i for i, m in enumerate(mask) if m and i > 0]"
POSITIONS_TRANSLATION = ("enumerate(mask)", "enumerate(sequence.loss_mask)")

# mhrl/engine/oracle.py's AnswerKeyTeacher._steps picks a task's path so.
PROJECT_PATTERN = "pattern = PATTERNS[sum(map(ord, task.task_id)) % len(PATTERNS)]"

# mhrl/training/sft.py's demo_items keeps right, trainable rows of bounded size.
PROJECT_DEMO_ITEMS_FILTER = "if not (record.correct and record.trainable):"


def quiet(**knobs: float | int | bool | str) -> tuple[sft_toy.RunResult, str]:
    """Run sft_toy.py in-process and capture what it prints.

    Args:
        **knobs: Settings fields to change, e.g. ``teacher="fixed-length"``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = sft_toy.run(sft_toy.Settings(**knobs))
    return result, output.getvalue()


def run_script(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run one of this chapter's scripts as a learner would.

    Args:
        script: The file name, e.g. ``"sft_toy.py"``.
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / script), *flags]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
    )


def function_node(source: str, name: str) -> ast.FunctionDef:
    """Find a top-level function in some source.

    Args:
        source: Python source.
        name: The function's name.

    Returns:
        Its syntax tree node.
    """
    tree = ast.parse(source)
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def function_code(source: str, name: str) -> str:
    """Return a function's code without its docstring, as a syntax tree dump.

    Args:
        source: Python source holding the function.
        name: The function's name.

    Returns:
        The dump.
    """
    node = function_node(source, name)
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def chapter_code(path: Path, name: str) -> str:
    """Return a function's code from a file, docstring left out.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The dump.
    """
    return function_code(path.read_text(encoding="utf-8"), name)


def chapter_source(name: str) -> str:
    """Return one of this chapter's files as text.

    Args:
        name: The file name.

    Returns:
        Its text.
    """
    return (CHAPTER / name).read_text(encoding="utf-8")


def readme() -> str:
    """Return the chapter's README.

    Returns:
        Its text.
    """
    return chapter_source("README.md")


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
        yield environment.MiniAnalyst(question(level), folder)


def demonstrations(
    tmp_path: Path, which: str = teacher.VARIED, count: int = sft_toy.DEMONSTRATIONS
) -> list[sft.Demonstration]:
    """Collect demonstrations on the easy training questions.

    Args:
        tmp_path: Where the rollout folders go.
        which: The teacher.
        count: How many.

    Returns:
        The demonstrations.
    """
    easy = [q for q in questions.make_questions().training if q.level == "easy"]
    return sft.collect_demonstrations(easy, which, count, 0, tmp_path, policy.features)


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
def runs(module_home: Path) -> dict[str, tuple[sft_toy.RunResult, str]]:
    """Run each README experiment once (seed 0) for several tests.

    Args:
        module_home: The temporary course home the runs write into.

    Returns:
        Each experiment's result and printed output.
    """
    return {
        "varied": quiet(),
        "fixed": quiet(teacher=teacher.FIXED_LENGTH),
        "never": quiet(teacher=teacher.NEVER_CHECKS),
        "high": quiet(lr=sft_toy.HIGH_LR),
        "fixed-high": quiet(teacher=teacher.FIXED_LENGTH, lr=sft_toy.HIGH_LR),
        "share": quiet(teacher=teacher.FIXED_LENGTH, calls_as_share=True),
        "then-rl": quiet(then_rl=True),
    }


# --- What is copied ----------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["grader.py", "questions.py", "grpo.py", "data/shop.db"]
)
def test_the_grader_questions_grpo_and_data_are_chapter_sixs(name: str) -> None:
    """Copied unchanged from chapter 6."""
    assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_six_only_in_the_chapter_name() -> None:
    """The one change: this chapter writes under 08-sft-toy."""
    mine = chapter_source("tasks.py")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine == theirs.replace("06-ratio-and-clip", "08-sft-toy")
    assert tasks.CHAPTER == "08-sft-toy"


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
    ],
)
def test_the_policy_math_is_chapter_sixs_line_for_line(name: str) -> None:
    """Only the features changed; the math runs the same code."""
    mine = chapter_code(CHAPTER / "policy.py", name)
    assert mine == chapter_code(PREVIOUS / "policy.py", name)


@pytest.mark.parametrize(
    "name", ["column_names", "tables_needed", "missing_table_error", "fresh_folder"]
)
def test_the_environment_helpers_are_chapter_sixs(name: str) -> None:
    """The environment only gains CHECK; everything around it is chapter 6's."""
    mine = chapter_code(CHAPTER / "environment.py", name)
    assert mine == chapter_code(PREVIOUS / "environment.py", name)


# --- The project's code, line for line ------------------------------------------


def test_batches_is_the_projects_line_for_line() -> None:
    """mhrl/training/common.py's batches, and the project's SFT shape."""
    mine = chapter_code(CHAPTER / "sft.py", "batches")
    assert mine == function_code(PROJECT_BATCHES, "batches")
    assert (sft_toy.DEMONSTRATIONS, sft_toy.BATCH_SIZE, sft_toy.EPOCHS) == (120, 8, 1)


def test_loss_fn_is_the_projects_with_the_toys_logprobs() -> None:
    """-log p summed over the teacher's pieces, as mhrl/training/sft.py's loss_fn."""
    project = PROJECT_LOSS_FN
    for theirs, ours in LOSS_TRANSLATION:
        project = project.replace(theirs, ours)
    mine = chapter_code(CHAPTER / "sft.py", "loss_fn")
    assert mine == function_code(project, "loss_fn")


def test_the_trained_positions_are_picked_as_the_project_picks_them() -> None:
    """item_from_sequence keeps the masked positions, never position 0."""
    theirs, ours = POSITIONS_TRANSLATION
    assert PROJECT_POSITIONS.replace(theirs, ours) in chapter_source("sft.py")


def test_the_teacher_picks_its_path_as_the_project_does() -> None:
    """The project's line, and the project's four paths in its order."""
    assert PROJECT_PATTERN in chapter_source("teacher.py")
    assert teacher.PATTERNS == ("direct", "peek", "tables", "verify")


def test_demo_items_keeps_right_rows_as_the_project_does() -> None:
    """The project also asks 'trainable' (a capture can fail); the toy's never do."""
    source = ast.unparse(function_node(chapter_source("sft.py"), "demo_items"))
    assert (
        PROJECT_DEMO_ITEMS_FILTER.replace(
            "(record.correct and record.trainable)", "record.correct"
        )
        in source
    )


# --- The environment's new action ------------------------------------------------


def test_check_reads_answer_txt_back_and_changes_nothing_else(tmp_path: Path) -> None:
    """CHECK is cat answer.txt: before WRITE there is none, after it the answer."""
    with analyst(tmp_path) as env:
        assert env.step(environment.CHECK) == "no answer.txt yet"
        env.step(environment.LOOK)
        result = env.step(environment.QUERY_A)
        before = (env.state.tables_seen, env.state.result, env.state.written)
        env.step(environment.WRITE)
        assert env.step(environment.CHECK) == f"answer.txt holds {result}"
        assert (env.state.tables_seen, env.state.result) == before[:2]
        assert env.state.calls == 5
        assert env.step(environment.FINISH) == "correct · reward 1"


def test_the_actions_are_chapter_sixs_plus_check() -> None:
    """Seven actions; FINISH stays last, CHECK sits beside WRITE."""
    assert environment.ACTIONS == (
        "LOOK",
        "QUERY_A",
        "QUERY_B",
        "QUERY_C",
        "WRITE",
        "CHECK",
        "FINISH",
    )


# --- The student's features -------------------------------------------------------


def test_the_student_sees_four_facts_and_the_call_number(tmp_path: Path) -> None:
    """Twelve numbers, exactly one call switch on, the coming call's."""
    with analyst(tmp_path) as env:
        first = policy.features(env.state)
        env.step(environment.LOOK)
        env.step(environment.QUERY_A)
        third = policy.features(env.state)
    assert first.shape == (policy.weight_rows(),) == (12,)
    assert list(first[:4]) == [1.0, 0.0, 0.0, 0.0]
    assert list(first[4:]) == [1.0] + [0.0] * 7
    assert list(third[:4]) == [1.0, 0.25, 1.0, 0.0]
    assert list(third[4:]) == [0.0, 0.0, 1.0] + [0.0] * 5


def test_calls_as_share_is_chapter_sixs_shared_features(tmp_path: Path) -> None:
    """--calls-as-share gives chapters 4 to 7's five numbers back."""
    with analyst(tmp_path) as env:
        env.step(environment.LOOK)
        env.step(environment.QUERY_A)
        shared = policy.features(env.state, calls_as_share=True)
    assert policy.weight_rows(calls_as_share=True) == 5
    assert list(shared) == [1.0, 0.25, 1.0, 0.0, 2 / environment.BUDGET]


# --- The teacher -------------------------------------------------------------------


def test_the_four_paths_on_an_easy_question() -> None:
    """direct, peek, tables and verify, as the README lists them."""
    easy = question()
    by_path = {}
    for path in teacher.PATTERNS:
        task = next(
            q.task
            for q in questions.make_questions().training
            if q.level == "easy" and teacher.pattern_for(q.task) == path
        )
        by_path[path] = teacher.teacher_actions(
            replace(easy, task=task), teacher.VARIED
        )
    assert by_path["direct"] == ["LOOK", "QUERY_A", "WRITE", "FINISH"]
    assert by_path["peek"] == ["LOOK", "LOOK", "QUERY_A", "WRITE", "FINISH"]
    assert by_path["tables"] == ["LOOK"] * 4 + ["QUERY_A", "WRITE", "FINISH"]
    assert by_path["verify"] == ["LOOK", "QUERY_A", "WRITE", "CHECK", "FINISH"]


@pytest.mark.parametrize("which", [teacher.VARIED, teacher.NEVER_CHECKS])
def test_the_answer_key_teacher_is_right_on_every_question(
    which: str, tmp_path: Path
) -> None:
    """It knows the tables each question needs, at every level."""
    every = questions.make_questions()
    for number, q in enumerate(every.training + every.held_out, start=1):
        demo = sft.play_teacher(q, which, number, tmp_path, policy.features)
        assert demo.correct, (q.task.question, demo.path)


def test_the_fixed_length_teacher_answers_only_easy_questions(tmp_path: Path) -> None:
    """LOOK, QUERY_A, WRITE, FINISH works after one look and fails after two."""
    for number, level in enumerate(questions.LEVELS, start=1):
        demo = sft.play_teacher(
            question(level), teacher.FIXED_LENGTH, number, tmp_path, policy.features
        )
        assert demo.correct == (level == "easy")


def test_the_teacher_that_never_checks_differs_only_in_check(tmp_path: Path) -> None:
    """Same questions, same paths, verify's CHECK left out."""
    varied = demonstrations(tmp_path / "varied")
    never = demonstrations(tmp_path / "never", teacher.NEVER_CHECKS)
    for mine, theirs in zip(never, varied, strict=True):
        actions = [c.action for c in theirs.calls if c.action != "CHECK"]
        assert [c.action for c in mine.calls] == actions
    assert not any(c.action == "CHECK" for d in never for c in d.calls)
    assert any(c.action == "CHECK" for d in varied for c in d.calls)


# --- The demonstrations as the trainer sees them ------------------------------------


def test_only_the_teachers_actions_are_trained(tmp_path: Path) -> None:
    """The mask marks actions, never the question or what came back."""
    demo = demonstrations(tmp_path, count=1)[0]
    pieces, mask = demo.sequence.pieces, demo.sequence.loss_mask
    assert pieces[0].kind == "question" and mask[0] == 0
    assert [p.kind for p, m in zip(pieces, mask, strict=True) if m] == [
        c.action for c in demo.calls
    ]
    assert all(p.kind == "observation" for p in pieces[2::2])
    item = sft.item_from_sequence(demo.sequence)
    assert item is not None and item.n_calls == len(demo.calls)
    indexes = tuple(environment.ACTIONS.index(c.action) for c in demo.calls)
    assert item.choices == indexes


def test_demo_items_drops_wrong_demonstrations(tmp_path: Path) -> None:
    """A demonstration the grader rejected never becomes a row."""
    demo = sft.play_teacher(
        question("medium"), teacher.FIXED_LENGTH, 1, tmp_path, policy.features
    )
    right = demonstrations(tmp_path / "right", count=3)
    assert not demo.correct
    assert len(sft.demo_items([*right, demo])) == 3
    assert sft.demo_items(right, max_calls=3) == [
        item for item in sft.demo_items(right) if item.n_calls <= 3
    ]


def test_120_demonstrations_in_batches_of_8_make_15_steps(tmp_path: Path) -> None:
    """One epoch: every row once, 8 per step."""
    items = sft.demo_items(demonstrations(tmp_path))
    zeros = np.zeros((policy.weight_rows(), len(environment.ACTIONS)))
    _, logs = sft.train_sft(items, 1, 3.0, 8, zeros)
    assert len(items) == 120 and len(logs) == 15
    assert all(log.rows == 8 and log.epoch == 0 for log in logs)
    assert sum(log.calls for log in logs) == sum(item.n_calls for item in items)


# --- The loss and its gradient --------------------------------------------------------


def test_the_untrained_loss_per_call_is_log_7(tmp_path: Path) -> None:
    """Seven actions at p 1/7 each: every call costs ln 7 = 1.95."""
    items = sft.demo_items(demonstrations(tmp_path, count=8))
    zeros = np.zeros((policy.weight_rows(), len(environment.ACTIONS)))
    _, logs = sft.train_sft(items, 1, 3.0, 8, zeros)
    assert logs[0].loss_per_call == pytest.approx(math.log(7))


def test_the_hand_gradient_is_the_measured_slope_of_the_loss(tmp_path: Path) -> None:
    """loss_gradient against slopes measured by moving each weight a hair."""
    item = sft.demo_items(demonstrations(tmp_path, count=4))[3]
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(policy.weight_rows(), len(environment.ACTIONS)))
    hand = sft.loss_gradient(weights, item)
    measured = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        up, down = weights.copy(), weights.copy()
        up[index] += STEP
        down[index] -= STEP
        measured[index] = (sft.loss_fn(up, item) - sft.loss_fn(down, item)) / (2 * STEP)
    assert np.allclose(hand, measured, atol=MEASURED_TOLERANCE)


def test_one_step_divides_the_batch_gradient_by_its_calls(tmp_path: Path) -> None:
    """train_sft's step is the summed gradient over the batch's calls, times lr."""
    items = sft.demo_items(demonstrations(tmp_path, count=8))
    zeros = np.zeros((policy.weight_rows(), len(environment.ACTIONS)))
    lr = 2.0
    weights, _ = sft.train_sft(items, 1, lr, 8, zeros)
    total = sum(item.n_calls for item in items)
    expected = -lr * sum(sft.loss_gradient(zeros, item) for item in items) / total
    assert np.allclose(weights, expected)
    unchanged, _ = sft.train_sft(items, 1, 0.0, 8, zeros)
    assert np.array_equal(unchanged, zeros)


# --- The experiments ------------------------------------------------------------------


def medium_result(result: sft_toy.RunResult) -> evaluate.LevelResult:
    """Return a run's medium held-out test.

    Args:
        result: The run.

    Returns:
        Its medium level's result.
    """
    return next(r for r in result.after if r.level == "medium")


def test_the_varied_student_learns_and_survives_one_extra_look(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """The loss falls, and greedy answers every medium question."""
    result, _ = runs["varied"]
    losses = [step.loss_per_call for step in result.sft_steps]
    assert losses[0] > 1.9 and losses[-1] < 0.6
    middle = medium_result(result)
    assert middle.greedy_right == middle.questions
    assert middle.trace.verdict.correct


def test_the_fixed_length_student_finishes_on_call_4_unsaved(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """LOOK, QUERY_A, WRITE, FINISH even when the query failed."""
    result, printed = runs["fixed"]
    middle = medium_result(result)
    assert [c.action for c in middle.trace.calls] == [
        "LOOK",
        "QUERY_A",
        "WRITE",
        "FINISH",
    ]
    assert middle.trace.calls[-1].observation == "no answer.txt · reward 0"
    assert middle.greedy_unsaved[4] == middle.questions
    assert "call 4  FINISH" in printed


def test_with_calls_as_a_share_the_shortcut_never_forms(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """Chapters 4 to 7's student re-runs the failed query instead."""
    result, _ = runs["share"]
    middle = medium_result(result)
    assert middle.greedy_unsaved[4] == 0
    assert middle.trace.stop == "budget used"


def test_the_student_of_a_teacher_that_never_checks_never_checks_greedily(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """Greedy: never. Sampled: fewer than the varied teacher's student."""
    never, _ = runs["never"]
    varied, _ = runs["varied"]
    assert sum(r.greedy_checked for r in never.after) == 0
    assert sum(r.checked for r in never.after) < sum(r.checked for r in varied.after)


def test_a_high_lr_reaches_zero_loss_only_on_one_shape(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """Fixed-length: loss 0.00 by step 2. Varied: it never gets near zero."""
    fixed, _ = runs["fixed-high"]
    varied, _ = runs["high"]
    assert fixed.sft_steps[1].loss_per_call < 0.005
    assert min(step.loss_per_call for step in varied.sft_steps) > 0.1
    assert medium_result(fixed).sampled_right == 0


def test_rl_from_the_imitator_has_signal_and_from_scratch_none(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """Seed 0 on the hard questions: scratch never right, the imitator at once."""
    result, printed = runs["then-rl"]
    assert result.rl is not None
    assert all(step.no_signal == len(step.groups) for step in result.rl.scratch)
    assert result.rl.imitator[0].no_signal < len(result.rl.imitator[0].groups)
    hard = next(r for r in result.rl.after_imitator if r.level == "hard")
    assert hard.greedy_right == hard.questions
    assert "SFT, then RL" in printed


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [
        ("varied", "step  1 ·", "\n\nheld-out test"),
        ("varied", "held-out test: greedy", "\n\nthe student, greedy"),
        ("varied", "the student, greedy", "\n\nhabits after"),
        ("varied", "habits after SFT", "\n\nwrote "),
        ("fixed", "held-out test: greedy", "\n\nwrote "),
        ("share", "step 15 ·", "\n\nwrote "),
        ("never", "held-out test: greedy", "\n\nwrote "),
        ("high", "step  1 ·", "\n\nthe student, greedy"),
        ("fixed-high", "held-out test: greedy", "\n\nwrote "),
        ("then-rl", "then RL on the", "\n\nwrote "),
    ],
)
def test_the_runs_match_the_readme(
    name: str,
    start: str,
    end: str,
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """The quoted blocks are this code's output, character for character."""
    _, printed = runs[name]
    block = printed[printed.index(start) : printed.index(end)]
    assert block in readme()


def test_metrics_has_one_line_per_step_with_the_projects_fields(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """SFT lines first, then each start's GRPO lines, as plot.py reads them."""
    result, _ = runs["then-rl"]
    metrics_path = next(p for p, _ in result.logs if p.name == "metrics.jsonl")
    lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [line["stage"] for line in lines[:15]] == ["sft"] * 15
    assert {"epoch", "step", "loss_per_call", "calls"} <= set(lines[0])
    rl_lines = lines[15:]
    assert len(rl_lines) == 2 * sft_toy.DEFAULT_RL_STEPS
    for field in ("accuracy", "mean_reward", "mean_calls", "groups_without_contrast"):
        assert field in rl_lines[0]


def test_demos_jsonl_holds_each_piece_and_its_mask(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """One line per demonstration, as the trainer saw it."""
    result, _ = runs["varied"]
    path = next(p for p, _ in result.logs if p.name == "demos.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 120
    first = lines[0]
    assert len(first["pieces"]) == len(first["loss_mask"])
    assert first["loss_mask"][:3] == [0, 1, 0]


def test_same_seed_same_logs_and_another_seed_differs(module_home: Path) -> None:
    """Every log's md5 repeats for a seed and changes with it."""
    first, _ = quiet(seed=3)
    again, _ = quiet(seed=3)
    other, _ = quiet(seed=4)
    assert [md5 for _, md5 in first.logs] == [md5 for _, md5 in again.logs]
    assert [md5 for _, md5 in first.logs] != [md5 for _, md5 in other.logs]


def test_every_rollout_gets_its_own_folder_and_none_is_left(course_home: Path) -> None:
    """The work folder is empty after a run."""
    quiet(then_rl=True, rl_steps=2)
    work = course_home / "08-sft-toy" / "work"
    assert tasks.work_root() == work and list(work.iterdir()) == []


def test_each_level_traces_a_question_about_lisbon() -> None:
    """Every level has a held-out question about the traced city."""
    held_out = questions.make_questions().held_out
    for level in questions.LEVELS:
        assert evaluate.traced_question(held_out, level).city == evaluate.TRACE_CITY


# --- plot.py, seeds.py and real_sft.py ---------------------------------------------


def test_plot_redraws_the_run_from_the_metrics_file_alone(
    runs: dict[str, tuple[sft_toy.RunResult, str]], tmp_path: Path
) -> None:
    """plot.py reads only metrics.jsonl, even after it is moved elsewhere."""
    result, _ = runs["then-rl"]
    metrics_path = next(p for p, _ in result.logs if p.name == "metrics.jsonl")
    moved = tmp_path / "elsewhere" / "metrics.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(metrics_path.read_bytes())
    finished = run_script("plot.py", str(moved))
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.startswith("15 SFT steps; loss per call 1.95 -> ")
    for title in ("SFT loss per call", "rollouts right", "groups with signal"):
        assert title in finished.stdout
    assert finished.stdout == run_script("plot.py", str(metrics_path)).stdout


def test_plot_draws_a_run_without_rl(
    runs: dict[str, tuple[sft_toy.RunResult, str]],
) -> None:
    """Without --then-rl there is only the loss chart."""
    result, _ = runs["fixed"]
    metrics_path = next(p for p, _ in result.logs if p.name == "metrics.jsonl")
    finished = run_script("plot.py", str(metrics_path))
    assert finished.returncode == 0, finished.stderr
    assert "rollouts right" not in finished.stdout


def test_plot_refuses_a_missing_or_foreign_file(tmp_path: Path) -> None:
    """A plain message and a non-zero exit, never a traceback."""
    missing = run_script("plot.py", str(tmp_path / "nope.jsonl"))
    foreign = tmp_path / "other.jsonl"
    foreign.write_text('{"hello": 1}\n', encoding="utf-8")
    other = run_script("plot.py", str(foreign))
    impossible = tmp_path / "metrics.jsonl"
    lines = [
        {"stage": "sft", "epoch": 0, "step": 1, "loss_per_call": 1.9, "calls": 3},
        {
            "stage": "grpo",
            "start": "scratch",
            "step": 1,
            "groups": 2,
            "groups_without_contrast": 0,
            "rollouts": 8,
            "correct": 9,  # more right than there were rollouts: not ours
        },
        {
            "stage": "grpo",
            "start": "imitator",
            "step": 1,
            "groups": 2,
            "groups_without_contrast": 0,
            "rollouts": 8,
            "correct": 1,
        },
    ]
    impossible.write_text("".join(json.dumps(x) + "\n" for x in lines))
    broken = run_script("plot.py", str(impossible))
    for finished in (missing, other, broken):
        assert finished.returncode != 0
        assert "Traceback" not in finished.stderr
        assert finished.stderr.startswith("plot.py: ")


def test_plot_keeps_its_copied_names_equal_to_the_chapters() -> None:
    """plot.py writes the stage and start names out so it can stand alone."""
    assert plot.SFT == sft.STAGE
    assert plot.GRPO == rl.STAGE
    assert plot.STARTS == (sft_toy.SCRATCH, sft_toy.IMITATOR)


def test_seeds_script_prints_every_table_and_claim() -> None:
    """Two seeds, a short RL stage: one row each in every table, every claim."""
    finished = run_script("seeds.py", "--seeds", "2", "--rl-steps", str(SHORT_RL))
    assert finished.returncode == 0, finished.stderr
    for name in seeds.RUNS:
        assert f"\n{name}\n" in finished.stdout
    rows = [line for line in finished.stdout.splitlines() if line.startswith(" 1 ")]
    assert len(rows) == len(seeds.RUNS) + 1  # every table, and the RL table
    claims = sum(len(c) for c in seeds.CLAIMS.values()) + len(seeds.RL_CLAIMS)
    assert finished.stdout.count("claim: ") == claims
    assert finished.stdout.count("holds in ") == claims


def test_real_sft_prints_the_projects_run() -> None:
    """120 demonstrations, loss 2.55 -> 0.11, and mini-swe-agent's 0 gained, 6 lost."""
    finished = run_script("real_sft.py")
    assert finished.returncode == 0, finished.stderr
    out = finished.stdout
    assert "120 demonstrations (60 mini-swe-agent, 60 claude-code)" in out
    assert "   1  2.55" in out and "  15  0.11" in out
    assert "tokens trained, all steps: 20,371 (the same as" in out
    assert "mini-swe-agent  24      18      0       6       0.03" in out


def test_real_sft_exact_p_is_the_binomial_tail() -> None:
    """0 gained, 6 lost: 2 x (1/2)**6 = 0.03125."""
    assert real_sft.exact_p(0, 6) == pytest.approx(0.03125)
    assert real_sft.exact_p(0, 0) == 1.0
    assert real_sft.exact_p(3, 3) == 1.0


def test_real_sft_refuses_a_broken_record(tmp_path: Path) -> None:
    """A missing, empty or foreign file is a plain error."""
    broken = tmp_path / "metrics.jsonl"
    broken.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(real_sft.RealSftError):
        real_sft.read_jsonl(broken, ("step", "loss_per_token"))
    with pytest.raises(real_sft.RealSftError):
        real_sft.read_jsonl(tmp_path / "missing.jsonl", ("step",))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(real_sft.RealSftError):
        real_sft.read_jsonl(empty, ("step",))


# --- Flags ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ("--fixed-length", "--varied"),
        ("--varied", "--teacher-never-checks"),
        ("--lr-high", "--lr", "5"),
        ("--lr", "-1"),
        ("--lr", "nan"),
        ("--rl-steps", "0"),
        ("--seed", "-1"),
    ],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, ...]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script("sft_toy.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


@pytest.mark.parametrize("flags", [("--seeds", "0"), ("--rl-steps", "0")])
def test_seeds_refuses_bad_flags(flags: tuple[str, str]) -> None:
    """seeds.py checks its flags like sft_toy.py does."""
    finished = run_script("seeds.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


def test_the_flags_pick_the_teacher_and_the_rate() -> None:
    """--fixed-length, --teacher-never-checks and --lr-high set the knobs."""
    pick = sft_toy.settings_from
    assert pick(sft_toy.parse_args([])).teacher == teacher.VARIED
    assert pick(sft_toy.parse_args(["--varied"])).teacher == teacher.VARIED
    fixed = pick(sft_toy.parse_args(["--fixed-length", "--lr-high"]))
    assert (fixed.teacher, fixed.lr) == (teacher.FIXED_LENGTH, sft_toy.HIGH_LR)
    never = pick(sft_toy.parse_args(["--teacher-never-checks", "--fast"]))
    assert (never.teacher, never.pause_s) == (teacher.NEVER_CHECKS, 0.0)
    assert report.FOURTH_CALL == 4


def test_plot_refuses_a_loss_that_is_not_a_number(tmp_path: Path) -> None:
    """JSON allows NaN; plot.py says the file is not its format, no traceback."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"stage": "sft", "epoch": 0, "step": 1, "loss_per_call": NaN}\n',
        encoding="utf-8",
    )
    finished = run_script("plot.py", str(path))
    assert finished.returncode != 0
    assert finished.stderr.startswith("plot.py: ")


def test_real_sft_writes_n_a_for_no_task_solved_twice() -> None:
    """Calls per solve needs a task solved both times; with none it says so."""
    assert real_sft.calls_per_solve([]) == "n/a"
    two = [real_sft.Outcome(True, 4), real_sft.Outcome(True, 5)]
    assert real_sft.calls_per_solve(two) == "4.5"


def test_the_knobs_agree_with_the_modules_they_stand_for() -> None:
    """knobs.py imports nothing, so its copies are checked here."""
    assert knobs.DEMO_LEVEL == questions.EASY
    assert knobs.HIGH_LR == 10 * knobs.DEFAULT_LR


def test_every_question_of_a_level_needs_the_same_looks() -> None:
    """looks_needed reads one question per level; the rest agree with it."""
    every = questions.make_questions()
    looks = sft_toy.looks_needed(every.held_out)
    assert looks == {"easy": 1, "medium": 2, "hard": 4}
    for q in every.training + every.held_out:
        assert environment.tables_needed(q.task.gold_sql) == looks[q.level]
