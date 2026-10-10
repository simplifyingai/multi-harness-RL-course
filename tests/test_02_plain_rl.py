"""Chapter 2: softmax and temperature, the nudge, graded tries and whole runs."""

from __future__ import annotations

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

CHAPTER = use_chapter("02-plain-rl")

import bandit  # noqa: E402
import questions  # noqa: E402
import report  # noqa: E402
import tasks  # noqa: E402
from grader import Grade, check_answer  # noqa: E402

SPEND_ROW = 2  # the one row whose right query is A, where greedy's ties land


def quiet_run(**knobs: float | int) -> tuple[bandit.RunResult, str]:
    """Run bandit.py's training in-process, without pauses or printed tries.

    Args:
        **knobs: Settings fields to change, e.g. ``temperature=0.0`` or
            ``seed=1``.

    Returns:
        The run's result and everything it printed.
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = bandit.run(bandit.Settings(show=0, **knobs))
    return result, output.getvalue()


def find(task_id: str) -> questions.Question:
    """Look a fair question up by its task id.

    Args:
        task_id: For example ``"orders-Paris"``.

    Returns:
        The question, from training or held out.
    """
    fair = questions.make_questions()
    return next(q for q in fair.training + fair.held_out if q.task.task_id == task_id)


def right_chances(scores: np.ndarray) -> list[float]:
    """Return, per question, the chance of its right query at temperature 1.

    Args:
        scores: The table of scores.

    Returns:
        One probability per question.
    """
    table = bandit.chances(scores)
    return [table[row][t.right] for row, t in enumerate(questions.TEMPLATES)]


def test_equal_scores_give_equal_chances() -> None:
    """Every row starts at 25/25/25/25, and probabilities add up to 1."""
    assert np.allclose(bandit.softmax(np.zeros(4)), 0.25)
    assert bandit.softmax(np.array([3.0, -1.0, 0.5, 2.0])).sum() == pytest.approx(1)


def test_temperature_flattens_or_sharpens_the_odds() -> None:
    """Above 1 the top choice loses share; below 1 it gains."""
    scores = np.array([2.0, 0.0, 0.0, 0.0])
    tops = [bandit.softmax(scores, t)[0] for t in (3.0, 1.0, 0.5)]
    assert tops[0] < tops[1] < tops[2]


def test_softmax_survives_huge_scores() -> None:
    """Large scores do not overflow."""
    assert bandit.softmax(np.array([1000.0, 0.0]))[0] == pytest.approx(1)


def test_greedy_takes_the_top_score_and_ties_go_to_the_first() -> None:
    """argmax picks A when everything ties."""
    assert bandit.greedy(np.zeros(4)) == 0
    assert bandit.greedy(np.array([0.0, 1.0, 1.0, 0.0])) == 1


def test_temperature_zero_uses_no_randomness() -> None:
    """A greedy pick leaves the random stream untouched."""
    rng, fresh = np.random.default_rng(7), np.random.default_rng(7)
    assert bandit.pick(np.array([0.0, 0.0, 5.0, 0.0]), 0.0, rng) == 2
    assert rng.random() == fresh.random()


def test_picks_follow_the_probabilities() -> None:
    """At temperature 1, a 50% query is picked about half the time."""
    scores = np.log(np.array([3.0, 1.0, 1.0, 1.0]))  # chances 50/17/17/17
    rng = np.random.default_rng(0)
    picks = [bandit.pick(scores, 1.0, rng) for _ in range(4000)]
    assert picks.count(0) / len(picks) == pytest.approx(0.5, abs=0.03)


def test_the_nudge_is_lr_times_reward_times_onehot_minus_probabilities() -> None:
    """From an even row, a reward for C adds 0.4 * (onehot - 0.25)."""
    row = bandit.nudge(np.zeros(4), choice=2, reward=1, lr=0.4)
    assert np.allclose(row, [-0.1, -0.1, 0.3, -0.1])
    assert row.sum() == pytest.approx(0)  # what one query gains, the rest lose


def test_the_nudge_uses_the_scores_own_probabilities() -> None:
    """From a 50/17/17/17 row, a reward for A adds onehot minus those chances."""
    scores = np.log(np.array([3.0, 1.0, 1.0, 1.0]))
    row = bandit.nudge(scores, choice=0, reward=1, lr=1.0)
    assert np.allclose(row - scores, [0.5, -1 / 6, -1 / 6, -1 / 6])


def test_a_reward_of_zero_changes_nothing() -> None:
    """Wrong tries teach this agent nothing."""
    row = np.array([0.3, -0.2, 0.1, -0.2])
    assert np.array_equal(bandit.nudge(row, choice=1, reward=0, lr=5.0), row)


def test_a_huge_learning_rate_decides_a_row_with_one_reward() -> None:
    """At lr 30 one reward takes a query from 25% to almost 100%."""
    row = bandit.nudge(np.zeros(4), choice=1, reward=1, lr=30.0)
    assert np.allclose(row, [-7.5, 22.5, -7.5, -7.5])
    assert bandit.softmax(row)[1] > 0.999999


def test_tables_are_printed_after_0_50_200_and_500_tries() -> None:
    """Shorter and longer runs keep the same stops."""
    assert bandit.table_after(500) == [0, 50, 200, 500]
    assert bandit.table_after(100) == [0, 50, 100]
    assert bandit.table_after(1000) == [0, 50, 200, 500, 1000]


def test_the_right_query_sits_in_a_different_place_in_every_row() -> None:
    """No letter is a safe bet."""
    assert sorted(t.right for t in questions.TEMPLATES) == [0, 1, 2, 3]


def test_every_kept_question_has_exactly_one_rewarded_query() -> None:
    """The grader rewards the right query and none of the three mistakes."""
    fair = questions.make_questions()
    for question in fair.training + fair.held_out:
        template = questions.TEMPLATES[question.row]
        rewarded = [
            choice
            for choice in range(len(questions.LETTERS))
            if check_answer(
                question.task,
                questions.run_query(
                    questions.SHOP_DB,
                    questions.candidate_sql(template, choice, question.city),
                ),
            ).correct
        ]
        assert rewarded == [template.right], question.task.task_id


def test_cities_split_by_a_fixed_rule() -> None:
    """Every third city is held out, and no question crosses the split."""
    fair = questions.make_questions()
    assert fair.test_cities == ("Brussels", "Lisbon", "Prague", "Warsaw")
    assert {q.city for q in fair.training} <= set(fair.train_cities)
    assert {q.city for q in fair.held_out} <= set(fair.test_cities)
    assert (len(fair.training), len(fair.held_out)) == (30, 13)


def test_unfair_questions_are_left_out_with_their_reason() -> None:
    """Paris has 13 customers and 13 orders, so query D would get lucky there."""
    left_out = dict(questions.make_questions().left_out)
    assert left_out["How many customers live in Paris?"] == "query D also gives 13"
    assert left_out["How much did customers from Brussels spend?"] == "no answer"


def test_a_try_runs_in_a_fresh_folder_and_the_grader_gives_the_reward(
    course_home: Path,
) -> None:
    """The right query earns 1, a mistake earns 0, and no folder is left behind."""
    question = find("orders-Paris")
    assert questions.attempt(question, 3) == ("13", Grade(True, "13", "correct"))
    answer, verdict = questions.attempt(question, 0)
    assert (answer, verdict.reward) == ("200", 0)
    assert verdict.reason == "expected '13', got '200'"
    assert list((course_home / "02-plain-rl" / "work").iterdir()) == []


def test_answers_are_written_the_way_the_grader_reads_them() -> None:
    """Money gets two decimals; an empty result writes nothing."""
    assert questions.format_answer([(4515.909999,)]) == "4515.91"
    assert questions.format_answer([(13,)]) == "13"
    assert questions.format_answer([(None,)]) == ""
    assert questions.format_answer([]) == ""


def test_snippets_tell_the_four_queries_apart() -> None:
    """Each shortened query is the start of its own SQL and unique in its row."""
    for template in questions.TEMPLATES:
        shown = [questions.snippet(template, c, "Amsterdam") for c in range(4)]
        assert len(set(shown)) == len(shown)
        for choice, text in enumerate(shown):
            full = questions.candidate_sql(template, choice, "Amsterdam")
            assert full.startswith(text.removesuffix(questions.ELLIPSIS))


def test_default_run_learns_the_right_query_and_passes_the_held_out_test() -> None:
    """Start at 25% each, end above 90%, and answer every held-out question."""
    result, printed = quiet_run()
    first_table = printed.split("after 0 tries")[1].split("\n\n")[0]
    rows = [line.split("?")[1] for line in first_table.splitlines() if "?" in line]
    # Each row: four chances (the right one marked *) and 0 tries so far.
    assert [re.findall(r"\d+", row) for row in rows] == [["25"] * 4 + ["0"]] * 4
    assert min(right_chances(result.scores)) > 0.9
    assert (result.test_correct, result.test_total) == (13, 13)


def test_same_seed_same_run_and_another_seed_differs() -> None:
    """Same seed, byte-identical log and output; seed 1 differs."""
    first, first_text = quiet_run(seed=0)
    again, again_text = quiet_run(seed=0)
    other, _ = quiet_run(seed=1)
    assert first.digest == again.digest and first_text == again_text
    assert other.digest != first.digest


def test_temperature_zero_never_learns_the_rows_that_start_on_a_wrong_query() -> None:
    """Greedy repeats A forever; only the row where A is right earns rewards."""
    result, _ = quiet_run(temperature=0.0)
    for row in range(len(questions.TEMPLATES)):
        if row != SPEND_ROW:
            assert np.array_equal(result.scores[row], np.zeros(4))
    tries = [
        json.loads(line)
        for line in result.log_path.read_text(encoding="utf-8").splitlines()
        if '"try"' in line
    ]
    assert {t["picked"] for t in tries} == {"A"}
    assert all(
        t["reward"] == 0 for t in tries if not t["question"].startswith("How much")
    )
    assert (result.test_correct, result.test_total) == (3, 13)


def test_temperature_three_explores_and_learns_more_slowly() -> None:
    """Fewer rewards and flatter odds than temperature 1, same final test."""
    plain, _ = quiet_run(temperature=1.0)
    hot, _ = quiet_run(temperature=3.0)
    assert hot.rewards < plain.rewards
    assert min(right_chances(hot.scores)) < min(right_chances(plain.scores))
    assert hot.test_correct == hot.test_total


def run_script(*flags: str) -> subprocess.CompletedProcess[str]:
    """Run bandit.py as a learner would, from the command line.

    Args:
        *flags: The command-line flags.

    Returns:
        The finished process, with its output as text.
    """
    command = [sys.executable, str(CHAPTER / "bandit.py"), *flags]
    return subprocess.run(command, capture_output=True, text=True)


def test_the_script_runs_alone_and_prints_the_same_md5_twice(course_home: Path) -> None:
    """Two runs of the command line print byte-identical output."""
    runs = [run_script("--fast", "--episodes", "60").stdout for _ in range(2)]
    assert runs[0] == runs[1]
    assert "md5   " in runs[0]
    assert (course_home / "02-plain-rl" / "runs").is_dir()


def test_a_printed_try_names_the_question_the_query_and_the_verdict() -> None:
    """The first try of seed 0, exactly as the README quotes it."""
    printed = run_script("--fast", "--episodes", "60").stdout
    assert (
        "try 1 · When was the first order from Copenhagen?\n"
        "        picked D: SELECT MIN(order_date) FROM orders → 2024-01-08"
        " · wrong · reward 0\n"
    ) in printed
    assert "tries 1-50: " in printed and "tries 51-60: " in printed


def test_a_short_run_prints_a_table_at_its_end_and_no_tries_with_show_0() -> None:
    """--episodes 20 stops after 0 and 20 tries; --show 0 prints no try."""
    printed = run_script("--fast", "--episodes", "20", "--show", "0").stdout
    assert re.findall(r"after (\d+) tries", printed) == ["0", "20"]
    assert "try 1 ·" not in printed


def test_queries_and_words_print_without_training() -> None:
    """--queries lists the 16 queries and what was left out; --words the box."""
    listed = run_script("--queries").stdout
    assert listed.count("SELECT") == len(questions.TEMPLATES) * len(questions.LETTERS)
    assert "How many customers live in Paris? · query D also gives 13" in listed
    assert "md5" not in listed
    assert "| weights" in run_script("--words").stdout


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "abc"])
def test_bad_flags_are_refused_with_a_plain_message(value: str) -> None:
    """A temperature that is not an ordinary number, 0 or more, is refused."""
    finished = run_script("--temperature", value)
    assert finished.returncode == 2
    assert "argument --temperature" in finished.stderr
    assert "Traceback" not in finished.stderr


def test_the_grader_and_database_are_chapter_one_copies() -> None:
    """Chapter 2 copies chapter 1's grader and shop.db byte for byte."""
    first = ROOT / "01-the-job"
    assert (CHAPTER / "grader.py").read_bytes() == (first / "grader.py").read_bytes()
    shop = (first / "data" / "dbs" / "shop.db").read_bytes()
    assert questions.SHOP_DB.read_bytes() == shop


def test_the_words_box_maps_all_five_words(capsys: pytest.CaptureFixture[str]) -> None:
    """state, action, reward, policy and weights each get a line."""
    report.print_words()
    printed = capsys.readouterr().out
    for word in ("state", "action", "reward", "policy", "weights"):
        assert f"| {word}" in printed
    assert "about 2.6 billion" in printed


def test_work_folders_live_under_course_home(course_home: Path) -> None:
    """The chapter writes under COURSE_HOME, never into the repo."""
    assert tasks.work_root() == course_home / "02-plain-rl" / "work"
