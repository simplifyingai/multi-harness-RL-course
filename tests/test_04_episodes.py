"""Chapter 4: the environment's rules, the shared advantage, runs and memorising."""

from __future__ import annotations

import ast
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("04-episodes")

import environment  # noqa: E402
import minianalyst  # noqa: E402
import policy  # noqa: E402
import questions  # noqa: E402
import tasks  # noqa: E402
from environment import (  # noqa: E402
    BUDGET,
    BUDGET_USED,
    FINISH,
    FINISHED,
    LOOK,
    QUERY_A,
    QUERY_B,
    RECALL,
    WRITE,
    MiniAnalyst,
    fresh_folder,
)

PREVIOUS = ROOT / "03-policy-gradient"
SCRIPT_TIMEOUT_S = 120
STEP = 1e-6  # how far a weight is moved to measure a slope
MEASURED_TOLERANCE = 1e-6
SHORT_RUN = 120  # rollouts in the quick determinism checks
SHOP_TABLES = "tables: customers, order_items, orders, products"
# Rollout 2 of seed 0, exactly as the README quotes it.
SEED_0_ROLLOUT_2 = """rollout 2 · untrained: every action has the same chance, 1 in 6
question  How much did customers from Madrid spend?
folder    rollout-0002, new: data.db (a copy of shop.db)
  call 1  QUERY_C  p 0.17  advantage +0.00  → no such table: order_items
  call 2  WRITE    p 0.17  advantage +0.00  → nothing to write yet
  call 3  FINISH   p 0.17  advantage +0.00  → no answer.txt · reward 0
stopped   FINISH, after 3 calls · no answer.txt · reward 0
folder    at the end: data.db · deleted
"""


@pytest.fixture(scope="module")
def fair() -> questions.QuestionSet:
    """Build the question set once for the whole file.

    Returns:
        The fair questions, split by city.
    """
    return questions.make_questions()


def quiet(**knobs: float | int | bool) -> tuple[minianalyst.RunResult, str]:
    """Run minianalyst.py in-process and capture what it prints.

    Args:
        **knobs: Settings fields to change, e.g. ``memorise=True``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = minianalyst.run(minianalyst.Settings(**knobs))
    return result, output.getvalue()


def run_script(*flags: str) -> subprocess.CompletedProcess[str]:
    """Run minianalyst.py as a learner would.

    Args:
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / "minianalyst.py"), *flags]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S
    )


def play(env: MiniAnalyst, *actions: str) -> list[str]:
    """Send several actions to an environment.

    Args:
        env: The environment.
        *actions: The actions, in order.

    Returns:
        What each call sent back.
    """
    return [env.step(action) for action in actions]


def function_code(path: Path, name: str) -> str:
    """Return a function's code without its docstring, as a syntax tree dump.

    Args:
        path: The Python file.
        name: The function's name.

    Returns:
        The dump of the function's body, docstring left out.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


# --- What is copied from earlier chapters -------------------------------------


def test_grader_and_database_are_chapter_three_copies() -> None:
    """Chapter 1's grader and shop.db, byte for byte."""
    for name in ("grader.py", "data/shop.db"):
        assert (CHAPTER / name).read_bytes() == (PREVIOUS / name).read_bytes(), name


def test_tasks_differs_from_chapter_three_only_in_the_chapter_name() -> None:
    """The only change is where outputs go."""
    mine = (CHAPTER / "tasks.py").read_text(encoding="utf-8")
    theirs = (PREVIOUS / "tasks.py").read_text(encoding="utf-8")
    assert mine.replace("04-episodes", "03-policy-gradient") == theirs


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
    ],
)
def test_the_math_is_chapter_three_line_for_line(name: str) -> None:
    """The copied functions run the same code; only their docstrings changed."""
    mine = function_code(CHAPTER / "policy.py", name)
    assert mine == function_code(PREVIOUS / "policy.py", name)


def test_the_split_by_city_and_query_a_is_right_everywhere(
    fair: questions.QuestionSet,
) -> None:
    """32 training and 14 held-out questions; no held-out city is trained on."""
    assert (len(fair.training), len(fair.held_out)) == (32, 14)
    assert fair.test_cities == ("Brussels", "Lisbon", "Prague", "Warsaw")
    assert not {q.city for q in fair.training} & set(fair.test_cities)
    assert all(template.right == 0 for template in questions.TEMPLATES)
    assert all(len(t.candidates) == 3 for t in questions.TEMPLATES)


# --- The environment's rules ------------------------------------------------


def test_a_query_fails_with_no_such_table_until_the_agent_looks(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """Before LOOK, SQLite says no such table; after LOOK the query runs."""
    question = fair.training[0]
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(question, folder)
        before, tables, after = play(env, QUERY_A, LOOK, QUERY_A)
    assert before.startswith("no such table: ")
    assert tables == SHOP_TABLES
    assert after == question.task.answer
    assert env.state.result == question.task.answer and env.state.schema_seen


def test_the_reward_comes_only_at_finish(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """No reward after LOOK, QUERY or WRITE; FINISH grades answer.txt."""
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(fair.training[0], folder)
        for action in (LOOK, QUERY_A, WRITE):
            env.step(action)
            assert env.reward is None and not env.done
        assert env.step(FINISH) == "correct · reward 1"
    assert (env.reward, env.state.stop, env.state.calls) == (1, FINISHED, 4)


def test_the_budget_stops_the_rollout_and_answer_txt_is_still_graded(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """After 8 calls it stops with 'budget used'; a ninth call is refused."""
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(fair.training[0], folder)
        play(env, LOOK, QUERY_A, WRITE, *[LOOK] * (BUDGET - 4))
        assert env.reward is None
        env.step(LOOK)
        assert (env.state.calls, env.state.stop, env.reward) == (
            BUDGET,
            BUDGET_USED,
            1,
        )
        with pytest.raises(environment.RolloutOver):
            env.step(FINISH)


def test_wandering_until_the_budget_earns_nothing(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """Eight calls with no answer written: budget used, reward 0."""
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(fair.training[0], folder)
        replies = play(env, WRITE, *[QUERY_B] * (BUDGET - 1))
    assert replies[0] == environment.NOTHING_TO_WRITE
    assert (env.state.stop, env.reward, env.grade.reason) == (
        BUDGET_USED,
        0,
        "no answer.txt",
    )


def test_a_wrong_query_written_is_graded_wrong(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """QUERY_B is a mistake for every question; writing it earns 0."""
    question = fair.training[0]
    with fresh_folder(tmp_path, "rollout-test") as folder:
        env = MiniAnalyst(question, folder)
        *_, verdict = play(env, LOOK, QUERY_B, WRITE, FINISH)
    assert env.reward == 0 and verdict.startswith("expected ")


def test_actions_on_offer(fair: questions.QuestionSet, tmp_path: Path) -> None:
    """Six actions, or seven with a memory; anything else is refused."""
    with fresh_folder(tmp_path, "rollout-test") as folder:
        plain = MiniAnalyst(fair.training[0], folder)
        assert plain.actions == environment.ACTIONS and len(plain.actions) == 6
        with pytest.raises(environment.UnknownAction):
            plain.step(RECALL)
        remembering = MiniAnalyst(fair.training[0], folder, memory={})
        assert remembering.actions[-1] == RECALL and len(remembering.actions) == 7


def test_recall_writes_only_what_was_seen_for_this_exact_question(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """A remembered question is answered; another city's question is not."""
    seen, unseen = fair.training[0], fair.held_out[0]
    memory = {seen.task.task_id: seen.task.answer}
    with fresh_folder(tmp_path, "rollout-seen") as folder:
        env = MiniAnalyst(seen, folder, memory)
        assert play(env, RECALL, FINISH)[1] == "correct · reward 1"
    with fresh_folder(tmp_path, "rollout-unseen") as folder:
        env = MiniAnalyst(unseen, folder, memory)
        assert play(env, RECALL)[0] == environment.NOTHING_REMEMBERED
        env.step(FINISH)
    assert env.reward == 0


# --- A fresh folder per rollout ------------------------------------------------


def test_each_rollout_gets_a_new_folder_that_is_deleted(
    fair: questions.QuestionSet, tmp_path: Path
) -> None:
    """The folder starts with only data.db and is gone afterwards."""
    with fresh_folder(tmp_path, "rollout-0001") as folder:
        assert sorted(p.name for p in folder.iterdir()) == ["data.db"]
        play(MiniAnalyst(fair.training[0], folder), LOOK, QUERY_A, WRITE)
        assert (folder / "answer.txt").exists()
    assert not folder.exists()
    with fresh_folder(tmp_path, "rollout-0002") as folder:
        assert not (folder / "answer.txt").exists()  # nothing carried over


def test_a_run_gives_every_rollout_its_own_folder_and_leaves_none(
    course_home: Path,
) -> None:
    """Rollout folders are numbered, and the work folder is empty after."""
    result, _ = quiet(rollouts=40)
    names = [update.rollout.folder for update in result.updates]
    assert names == [f"rollout-{n:04d}" for n in range(1, 41)]
    work = course_home / "04-episodes" / "work"
    assert tasks.work_root() == work and list(work.iterdir()) == []


# --- The policy and the shared advantage ---------------------------------------


def test_features_never_mention_the_question_or_its_city(
    fair: questions.QuestionSet,
) -> None:
    """Two questions in the same situation look identical to the agent."""
    one, other = fair.training[0], fair.held_out[-1]
    for state_args in ({}, {"calls": 3, "schema_seen": True, "result": "7"}):
        first = policy.features(environment.State(one, **state_args))
        second = policy.features(environment.State(other, **state_args))
        assert np.array_equal(first, second)
    assert len(policy.FEATURE_NAMES) == 5


def measured_weight_gradient(
    situations: list[np.ndarray], weights: np.ndarray, choices: list[int], adv: float
) -> np.ndarray:
    """Measure the rollout loss's slope for every weight by moving it a hair.

    Args:
        situations: The features at each call.
        weights: The weights.
        choices: The action taken at each call.
        adv: The rollout's advantage.

    Returns:
        One measured slope per weight.
    """

    def rollout_loss(w: np.ndarray) -> float:
        log_probs = [
            policy.log_prob(policy.action_scores(w, x), c)
            for x, c in zip(situations, choices, strict=True)
        ]
        return policy.rollout_loss(adv, log_probs)

    slopes = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        up, down = weights.copy(), weights.copy()
        up[index] += STEP
        down[index] -= STEP
        slopes[index] = (rollout_loss(up) - rollout_loss(down)) / (2 * STEP)
    return slopes


def test_the_rollout_gradient_is_measured_and_shared_by_every_call() -> None:
    """One advantage for all calls: the gradient is the sum of per-call parts."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(5, 6))
    situations = [rng.uniform(size=5) for _ in range(8)]
    choices = [int(c) for c in rng.integers(6, size=8)]
    scores = [policy.action_scores(weights, x) for x in situations]
    advantage = 0.37
    gradient = policy.rollout_gradient(situations, scores, choices, advantage)
    measured = measured_weight_gradient(situations, weights, choices, advantage)
    assert np.abs(gradient - measured).max() < MEASURED_TOLERANCE
    parts = sum(
        policy.rollout_gradient([x], [s], [c], advantage)
        for x, s, c in zip(situations, scores, choices, strict=True)
    )
    assert np.allclose(gradient, parts)
    doubled = policy.rollout_gradient(situations, scores, choices, 2 * advantage)
    assert np.allclose(doubled, 2 * gradient)


def test_the_baseline_is_chapter_threes_per_question_average() -> None:
    """The first rollout of a question gets advantage 0; then the average."""
    baseline = minianalyst.QuestionBaseline()
    assert baseline.expect(2, 1.0) == 1.0
    for reward in (1.0, 0.0, 0.0):
        baseline.add(2, reward)
    assert baseline.expect(2, 1.0) == pytest.approx(1 / 3)
    assert baseline.expect(0, 0.0) == 0.0  # question 0 has no rewards yet


def test_a_traced_rollout_prints_one_advantage_on_every_call() -> None:
    """Rollout 2 of seed 0 as the README quotes it; the last one too."""
    printed = run_script("--trace", "--fast", "--rollouts", "60").stdout
    assert SEED_0_ROLLOUT_2 in printed
    assert re.findall(r"^rollout (\d+) ·", printed, re.MULTILINE) == [
        "1",
        "2",
        "3",
        "60",
    ]
    for block in printed.split("\nrollout ")[1:]:
        advantages = set(re.findall(r"advantage ([+-]\d\.\d\d)  →", block))
        assert len(advantages) == 1, block


# --- Whole runs ------------------------------------------------------------------


def test_training_learns_the_method_and_it_carries_to_new_cities() -> None:
    """Seed 0 (measured): 32 of 32 training questions and 14 of 14 held out."""
    result, printed = quiet()
    assert sum(r.correct for r in result.training_test) == 32
    assert sum(r.correct for r in result.held_out_test) == 14
    calls = {r.actions[:3] for r in result.held_out_test}
    assert calls == {(LOOK, QUERY_A, WRITE)}
    assert "rollouts 551-600:" in printed


def test_nothing_pays_for_stopping_so_it_uses_the_whole_budget() -> None:
    """With a 1-or-0 reward the trained agent never finishes early (seed 0)."""
    result, printed = quiet()
    assert all(len(r.actions) == BUDGET for r in result.held_out_test)
    assert all(FINISH not in r.actions for r in result.held_out_test)
    last_stretch = result.updates[-minianalyst.STRETCH :]
    calls = np.mean([len(u.rollout.calls) for u in last_stretch])
    assert calls > BUDGET - 0.5
    assert "FINISH 0%" in printed


def test_memorising_is_perfect_on_training_and_fails_every_held_out() -> None:
    """--memorise: 32 of 32 on questions it saw, 0 of 14 on new cities."""
    result, printed = quiet(memorise=True)
    assert sum(r.correct for r in result.training_test) == 32
    assert sum(r.correct for r in result.held_out_test) == 0
    assert {r.actions for r in result.held_out_test} == {(RECALL,) * BUDGET}
    assert "and every call got the same answer: nothing remembered" in printed
    assert "held-out score: 0 of 14" in printed


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical log and output; seed 1 differs."""
    first, first_text = quiet(rollouts=SHORT_RUN)
    again, again_text = quiet(rollouts=SHORT_RUN)
    other, _ = quiet(rollouts=SHORT_RUN, seed=1)
    assert first.digest == again.digest and first_text == again_text
    assert other.digest != first.digest


def test_the_script_prints_the_same_bytes_twice() -> None:
    """Two runs of the script, the same output down to the md5."""
    flags = ("--trace", "--fast", "--rollouts", str(SHORT_RUN), "--memorise")
    runs = [run_script(*flags).stdout for _ in range(2)]
    assert runs[0] == runs[1] and "md5   " in runs[0]


@pytest.mark.parametrize(
    "flags",
    [
        ("--lr", "-1"),
        ("--lr", "nan"),
        ("--rollouts", "0"),
        ("--show", "-1"),
        ("--seed", "-1"),
    ],
)
def test_bad_flags_are_refused_with_a_plain_message(flags: tuple[str, str]) -> None:
    """A bad flag stops with a usage message, never a traceback."""
    finished = run_script(*flags)
    assert finished.returncode == 2
    assert "Traceback" not in finished.stderr
