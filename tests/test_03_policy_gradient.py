"""Chapter 3: log-probabilities, the gradient three ways, the loss, baselines, runs."""

from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("03-policy-gradient")

import baseline  # noqa: E402
import grad_check  # noqa: E402
import policy  # noqa: E402
import questions  # noqa: E402
import reinforce  # noqa: E402
import report  # noqa: E402
import tasks  # noqa: E402

PREVIOUS = ROOT / "02-plain-rl"
QUERIES = len(questions.LETTERS)  # four candidate queries per question
C = questions.LETTERS.index("C")  # the walk-through's pick
LR = 0.25  # chapter 2's default learning rate
ADVANTAGE = 0.6
MEASURED_TOLERANCE = 1e-7  # moving a score 0.001 down and up measures this well
MLX_TOLERANCE = 1e-6  # MLX keeps about 7 significant digits
SCRIPT_TIMEOUT_S = 120
SEED_0_TRY_53 = (
    "try 53 · How many orders did customers from Rome place?\n"
    "        chose D: SELECT COUNT(*) FROM orders JOIN ... → 5 · correct\n"
    "        p = 0.44 · log p = -0.814 · reward 1 · baseline 0.44 · advantage +0.56\n"
    "        loss +0.454 → new probabilities A 0.189 · B 0.165 · C 0.178 · D 0.469\n"
)


def random_rows(count: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Make random rows of scores and one random choice per row.

    Args:
        count: How many rows.
        seed: Seeds the generator.

    Returns:
        The rows, and the choices.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-5, 5, size=(count, QUERIES)), rng.integers(QUERIES, size=count)


def quiet(**knobs: float | int | str | bool) -> tuple[reinforce.RunResult, str]:
    """Run reinforce.py in-process with no printed tries, and capture its output.

    Args:
        **knobs: Settings fields to change, e.g. ``baseline="none"``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = reinforce.run(reinforce.Settings(show=0, **knobs))
    return result, output.getvalue()


def run_script(script: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run one of this chapter's scripts as a learner would.

    Args:
        script: The file name, e.g. ``"reinforce.py"``.
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / script), *flags]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
    )


# --- What is copied from chapter 2 -----------------------------------------


def test_the_questions_grader_and_database_are_chapter_two_copies() -> None:
    """Same questions, queries, grader and shop.db, byte for byte."""
    for name in ("questions.py", "grader.py", "data/shop.db"):
        assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_two_only_in_the_chapter_name() -> None:
    """The only change is where outputs go."""
    mine = (CHAPTER / "tasks.py").read_text(encoding="utf-8")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine.replace("03-policy-gradient", "02-plain-rl") == theirs


def function_source(path: Path, name: str) -> str:
    """Return one function's source text from a file.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The function's source, decorators excluded.
    """
    text = path.read_text(encoding="utf-8")
    functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    node = next(n for n in functions if n.name == name)
    return ast.get_source_segment(text, node) or ""


def test_grad_check_compares_with_chapter_two_nudge_unchanged() -> None:
    """grad_check.py's nudge and softmax are chapter 2's, character for character."""
    for name, mine in (("nudge", "grad_check.py"), ("softmax", "policy.py")):
        theirs = function_source(PREVIOUS / "bandit.py", name)
        assert function_source(CHAPTER / mine, name) == theirs


# --- Log-probabilities ------------------------------------------------------


def test_log_probabilities_beside_probabilities() -> None:
    """0.25 is -1.386 and 0.90 is -0.105; a certain choice is 0."""
    assert policy.log_prob(np.zeros(4), C) == pytest.approx(math.log(0.25))
    assert round(math.log(0.25), 3) == -1.386
    assert round(math.log(0.90), 3) == -0.105
    assert policy.log_prob(np.array([0.0, 50.0, 0.0, 0.0]), 1) == pytest.approx(0)


def test_log_prob_is_the_log_of_softmax_and_survives_huge_scores() -> None:
    """Same numbers as log(softmax), without overflowing on large scores."""
    rows, choices = random_rows()
    for row, choice in zip(rows, choices, strict=True):
        expected = math.log(policy.softmax(row)[choice])
        assert policy.log_prob(row, choice) == pytest.approx(expected)
    assert policy.log_prob(np.array([1000.0, 0.0]), 1) == pytest.approx(-1000)


def test_two_log_probabilities_add_up_to_the_log_of_both() -> None:
    """log(0.90 x 0.25) = log 0.90 + log 0.25, as grad_check.py prints."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        grad_check.show_log_probabilities()
    assert "-0.105 + -1.386 = -1.492" in printed.getvalue()
    assert "log 0.225 = -1.492" in printed.getvalue()
    assert "multiplied, the chance is 0.0" in printed.getvalue()


# --- The gradient, three ways -------------------------------------------------


def test_measured_gradient_matches_onehot_minus_probabilities() -> None:
    """Moving each score a hair finds the hand formula to better than 1e-7."""
    rows, choices = random_rows()
    for row, choice in zip(rows, choices, strict=True):
        measured = grad_check.measured_gradient(row, choice)
        gap = np.abs(measured - policy.log_prob_gradient(row, choice)).max()
        assert gap < MEASURED_TOLERANCE


def test_the_even_row_gradient_is_minus_a_quarter_and_three_quarters() -> None:
    """From 25% each, a pick of C has slopes [-0.25, -0.25, +0.75, -0.25]."""
    expected = [-0.25, -0.25, 0.75, -0.25]
    assert np.allclose(policy.log_prob_gradient(np.zeros(4), C), expected)
    assert np.allclose(grad_check.measured_gradient(np.zeros(4), C), expected)


def test_mlx_gradient_matches_by_hand_to_about_1e_7() -> None:
    """mx.grad agrees with onehot - p to 32-bit precision."""
    pytest.importorskip("mlx.core")
    rows, choices = random_rows()
    gap = grad_check.largest_difference(
        rows, choices, grad_check.mlx_gradient, policy.log_prob_gradient
    )
    assert 0 < gap < MLX_TOLERANCE


def test_without_mlx_the_check_says_so_and_runs_the_other_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a computer without MLX, ways 1 and 2 still print."""
    monkeypatch.setattr(grad_check, "mx", None)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        grad_check.show_three_ways(np.zeros(4))
        gaps = grad_check.show_random_rows(*random_rows(20))
    assert "MLX is not installed here" in printed.getvalue()
    assert "measured gradient: [-0.250 -0.250 +0.750 -0.250]" in printed.getvalue()
    assert list(gaps) == ["measured"]
    with pytest.raises(RuntimeError):
        grad_check.mlx_gradient(np.zeros(4), C)


# --- The loss and one gradient-descent step -----------------------------------


def test_the_loss_and_its_gradient() -> None:
    """loss = -advantage x log p; its gradient = -advantage x (onehot - p)."""
    rows, choices = random_rows(50)
    for row, choice in zip(rows, choices, strict=True):
        assert policy.loss(row, choice, ADVANTAGE) == pytest.approx(
            -ADVANTAGE * policy.log_prob(row, choice)
        )
        measured = -ADVANTAGE * grad_check.measured_gradient(row, choice)
        gradient = policy.loss_gradient(row, choice, ADVANTAGE)
        assert np.allclose(gradient, measured, atol=MEASURED_TOLERANCE)


def test_a_step_with_the_reward_as_advantage_is_chapter_two_nudge() -> None:
    """lr x reward x (onehot - p) is one step down -reward x log p, exactly."""
    rows, choices = random_rows()
    for index, (row, choice) in enumerate(zip(rows, choices, strict=True)):
        reward = index % 2
        gradient = policy.loss_gradient(row, choice, float(reward))
        stepped = policy.gradient_step(row, gradient, LR)
        assert np.array_equal(stepped, grad_check.nudge(row, choice, reward, LR))


def test_the_spec_update_prints_in_full(capsys: pytest.CaptureFixture[str]) -> None:
    """Chose C at p 0.25, reward 1, baseline 0.40: advantage +0.60 and the step."""
    question = questions.make_questions().training[0]
    before = np.zeros(QUERIES)
    gradient = policy.loss_gradient(before, C, ADVANTAGE)
    update = reinforce.Update(
        number=1,
        question=question,
        choice=C,
        answer="7",
        correct=True,
        reward=1.0,
        baseline=0.4,
        advantage=ADVANTAGE,
        p=0.25,
        log_p=policy.log_prob(before, C),
        loss=policy.loss(before, C, ADVANTAGE),
        before=before,
        after=policy.gradient_step(before, gradient, LR),
    )
    report.print_update(update, shifted=False)
    printed = capsys.readouterr().out
    assert (
        "p = 0.25 · log p = -1.386 · reward 1 · baseline 0.40 · advantage +0.60"
        in printed
    )
    assert "loss +0.832 → new probabilities A 0.240 · B 0.240 · C 0.279 · D 0.240" in (
        printed
    )


# --- Baselines --------------------------------------------------------------


def test_no_baseline_expects_nothing() -> None:
    """The advantage is the reward itself."""
    expected = reinforce.Baseline(reinforce.NO_BASELINE)
    expected.add(0, 1.0)
    assert expected.expect(0, 0.0) == 0.0


def test_the_first_try_has_nothing_to_compare_with() -> None:
    """Before any reward, the baseline is the try's own reward: advantage 0."""
    for kind in (reinforce.AVERAGE, reinforce.PER_QUESTION):
        assert reinforce.Baseline(kind).expect(0, 1.05) == 1.05


def test_the_running_average_is_shared_and_the_per_question_one_is_not() -> None:
    """Rewards 1, 0, 0 on question 0: the shared average is 1/3 everywhere."""
    shared = reinforce.Baseline(reinforce.AVERAGE)
    mine = reinforce.Baseline(reinforce.PER_QUESTION)
    for reward in (1.0, 0.0, 0.0):
        shared.add(0, reward)
        mine.add(0, reward)
    assert shared.expect(3, 1.0) == pytest.approx(1 / 3)
    assert mine.expect(0, 1.0) == pytest.approx(1 / 3)
    assert mine.expect(3, 1.0) == 1.0  # question 3 has no rewards yet


def test_an_unknown_baseline_is_refused() -> None:
    """Only the three kinds exist."""
    with pytest.raises(ValueError, match="unknown baseline"):
        reinforce.Baseline("median")


def test_rewards_plain_and_shifted() -> None:
    """0 and 1, or 1.00 and 1.05."""
    assert (reinforce.reward_for(True, False), reinforce.reward_for(False, False)) == (
        1.0,
        0.0,
    )
    assert (reinforce.reward_for(True, True), reinforce.reward_for(False, True)) == (
        1.05,
        1.00,
    )


def test_shifted_rewards_without_a_baseline_push_up_every_query_tried() -> None:
    """Every advantage is about 1; with a baseline no wrong query goes up."""
    fair = questions.make_questions()
    plain = reinforce.train_quietly(
        reinforce.Settings(lr=1.0, baseline=reinforce.NO_BASELINE, shifted=True), fair
    )
    assert plain.pushed_up == len(plain.pushes) == reinforce.DEFAULT_EPISODES
    assert plain.wrong_pushed_up > 0  # wrong tries were pushed up too
    for kind in (reinforce.AVERAGE, reinforce.PER_QUESTION):
        steady = reinforce.train_quietly(
            reinforce.Settings(lr=1.0, baseline=kind, shifted=True), fair
        )
        assert steady.wrong_pushed_up == 0
        assert np.std(steady.pushes) < np.std(plain.pushes) / 10
        # Measured at seed 0: the no-baseline run is right on three rows at
        # 95% to 99% but locked the fourth onto a wrong query (9 of 13 held
        # out), so its average chance looks better than the baseline's 70%.
        assert (steady.locked, steady.test_correct) == (0, 13)
    assert (plain.locked, plain.test_correct) == (1, 9)


def test_a_huge_learning_rate_locks_each_row_onto_its_first_pick() -> None:
    """lr 30 with shifted rewards and no baseline: whatever came first wins."""
    fair = questions.make_questions()
    for seed in range(3):
        settings = reinforce.Settings(
            seed=seed,
            lr=30.0,
            baseline=reinforce.NO_BASELINE,
            shifted=True,
            remember=True,
        )
        training = reinforce.train(settings, fair, [], printing=False)
        first: dict[int, int] = {}
        for update in training.updates:
            first.setdefault(update.question.row, update.choice)
        assert [int(np.argmax(row)) for row in training.scores] == [
            first[row] for row in range(4)
        ]


def test_the_same_learning_rate_learns_with_a_baseline() -> None:
    """lr 30 with shifted rewards and the running average: nothing locks."""
    outcome = reinforce.train_quietly(
        reinforce.Settings(lr=30.0, baseline=reinforce.AVERAGE, shifted=True),
        questions.make_questions(),
    )
    assert outcome.locked == 0
    assert outcome.test_correct == 13


def test_remembered_grades_change_nothing() -> None:
    """Grading each pair once gives the same run as a fresh folder every try."""
    fair = questions.make_questions()
    runs = [
        reinforce.train(
            reinforce.Settings(episodes=80, remember=remember), fair, [], False
        )
        for remember in (False, True)
    ]
    assert np.array_equal(runs[0].scores, runs[1].scores)


# --- Whole runs ---------------------------------------------------------------


def test_default_run_learns_and_passes_the_held_out_test() -> None:
    """Running-average baseline, rewards 0 and 1: above 90% and 13 of 13."""
    result, printed = quiet()
    final = policy.chances(result.scores)
    assert min(final[row][t.right] for row, t in enumerate(questions.TEMPLATES)) > 0.9
    assert (result.test_correct, result.test_total) == (13, 13)
    assert "running average · rewards 0 and 1" in printed


def test_no_baseline_reproduces_chapter_two_bit_for_bit(course_home: Path) -> None:
    """Same seed: reinforce.py --baseline none ends on chapter 2's exact weights."""
    driver = (
        "import sys; sys.path.insert(0, sys.argv[1]); import contextlib, io\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    if sys.argv[2] == 'bandit':\n"
        "        import bandit; r = bandit.run(bandit.Settings(seed=1, show=0))\n"
        "    else:\n"
        "        import reinforce as m\n"
        "        r = m.run(m.Settings(seed=1, show=0, baseline='none'))\n"
        "print(r.scores.tobytes().hex())\n"
    )
    environment = {**os.environ, "COURSE_HOME": str(course_home)}
    weights = [
        subprocess.run(
            [sys.executable, "-c", driver, str(folder), module],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        ).stdout
        for folder, module in ((PREVIOUS, "bandit"), (CHAPTER, "reinforce"))
    ]
    assert weights[0] == weights[1] and len(weights[0]) > 100


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical log and output; seed 1 differs."""
    first, first_text = quiet(seed=0)
    again, again_text = quiet(seed=0)
    other, _ = quiet(seed=1)
    assert first.digest == again.digest and first_text == again_text
    assert other.digest != first.digest


def test_a_printed_update_names_every_number_of_the_loss() -> None:
    """Try 53 of seed 0, exactly as the README quotes it."""
    printed = run_script("reinforce.py", "--fast", "--episodes", "60").stdout
    assert SEED_0_TRY_53 in printed
    assert re.findall(r"after (\d+) tries", printed) == ["0", "50", "60"]


def test_a_locked_run_names_the_locked_questions() -> None:
    """lr 30, shifted, no baseline, seed 0 (measured): three rows locked, 3 of 13."""
    result, printed = quiet(lr=30.0, baseline=reinforce.NO_BASELINE, shifted=True)
    assert "locked onto a wrong query" in printed
    assert "How many customers live in {city}?" in printed.split("locked onto")[1]
    assert (result.test_correct, len(reinforce.locked_rows(result.scores))) == (3, 3)


def test_scripts_print_the_same_md5_twice(course_home: Path) -> None:
    """reinforce.py, baseline.py and learning_rate.py are byte-identical reruns."""
    for script, flags in (
        ("reinforce.py", ("--fast", "--episodes", "60")),
        ("baseline.py", ("--seeds", "2", "--episodes", "100")),
        ("learning_rate.py", ("--seeds", "2", "--episodes", "100")),
    ):
        runs = [run_script(script, *flags).stdout for _ in range(2)]
        assert runs[0] == runs[1] and "md5   " in runs[0], script
    assert (course_home / "03-policy-gradient" / "results").is_dir()


def test_grad_check_ends_on_chapter_two_nudge() -> None:
    """The last line, and every gap printed is tiny."""
    printed = run_script("grad_check.py").stdout
    lines = printed.strip().splitlines()
    assert lines[-1] == (
        "chapter 2's nudge was exactly lr x reward x the gradient of log p"
    )
    assert lines[-2].startswith("  0 by hand, ")
    for gap in re.findall(r"\d\.\de-\d\d", printed):
        assert float(gap) < 1e-6
    assert printed == run_script("grad_check.py").stdout


def test_baseline_script_prints_both_reward_schemes() -> None:
    """Three variants twice, and the no-baseline count of pushed-up tries."""
    printed = run_script("baseline.py", "--seeds", "1", "--episodes", "100").stdout
    assert printed.count("per-question average") == 2
    assert "100 of 100 tries" in printed


def test_learning_rate_script_prints_six_settings() -> None:
    """0.1, 1 and 30, with and without a baseline, seed by seed."""
    printed = run_script("learning_rate.py", "--seeds", "1", "--episodes", "50").stdout
    rows = [line for line in printed.splitlines() if re.match(r"(0\.1|1|30) ", line)]
    assert len(rows) == 6


@pytest.mark.parametrize(
    "flags",
    [("--lr", "-1"), ("--lr", "nan"), ("--baseline", "median"), ("--episodes", "0")],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, str]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script("reinforce.py", *flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr


def test_baseline_summary_pools_every_try() -> None:
    """Average and spread are over all tries of all runs."""
    outcome = reinforce.Outcome(
        pushes=(0.1, -0.1),
        pushed_up=1,
        wrong_pushed_up=1,
        right_chances=(0.5, 0.25, 1.0, 0.75),
        locked=1,
        test_correct=10,
    )
    pooled = baseline.summary([outcome, outcome])
    assert pooled["push_average"] == pytest.approx(0)
    assert pooled["push_spread"] == pytest.approx(0.1)
    assert (pooled["tries"], pooled["rows"], pooled["locked"]) == (4, 8, 2)
    assert pooled["right_lowest"] == 0.25


def test_work_folders_live_under_course_home(course_home: Path) -> None:
    """The chapter writes under COURSE_HOME and leaves no work folder behind."""
    assert tasks.work_root() == course_home / "03-policy-gradient" / "work"
    quiet(episodes=20)
    assert list(tasks.work_root().iterdir()) == []
