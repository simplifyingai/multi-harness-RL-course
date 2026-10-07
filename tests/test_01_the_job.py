"""Chapter 1: the tasks, the grader, the generator and the recorded replays."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from chapters import use_chapter

CHAPTER = use_chapter("01-the-job")

import check_machine  # noqa: E402
import grade_cases  # noqa: E402
import make_tasks  # noqa: E402
import replay_agent  # noqa: E402
import tasks  # noqa: E402
from grader import Grade, check_answer, grade  # noqa: E402


def make_task(answer: str, answer_type: str) -> tasks.Task:
    """Build a throwaway task with a given answer.

    Args:
        answer: The correct answer.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.

    Returns:
        The task.
    """
    return tasks.Task("t", "train", "x", "easy", "", "", "", answer, answer_type)


@pytest.mark.parametrize(
    ("answer", "answer_type", "submitted", "correct"),
    [
        ("1234", "number", "$1,234.00", True),
        ("1234", "number", "1234.009", True),
        ("1234", "number", "1234.5", False),
        ("13", "number", "13 customers", False),
        ("13", "number", '"13".', True),
        ("13", "number", "```\n13\n```", True),
        ("Ann Lee", "text", "Dr. Ann Lee", True),
        ("Ann Lee", "text", "  ann   LEE ", True),
        ("the United States", "text", "United States", True),
        ("Ann Lee", "text", "Anne Lee", False),
        ("Ann Lee, Bo Chen", "list", "Bo Chen; Ann Lee", True),
        ("Ann Lee, Bo Chen", "list", "- Ann Lee\n- and Bo Chen", True),
        ("Ann Lee, Bo Chen", "list", "Ann Lee", False),
    ],
)
def test_check_answer_is_lenient_on_format_strict_on_content(
    answer: str, answer_type: str, submitted: str, correct: bool
) -> None:
    """The grader forgives formatting and nothing else."""
    assert check_answer(make_task(answer, answer_type), submitted).correct is correct


def test_wrong_number_reports_expected_and_got() -> None:
    """A wrong answer says what was expected and what arrived."""
    result = check_answer(make_task("13", "number"), "14")
    assert result == Grade(False, "14", "expected '13', got '14'")
    assert result.reward == 0


def test_list_reason_counts_missing_and_extra() -> None:
    """A wrong list says how many items were missing and extra."""
    result = check_answer(make_task("A, B", "list"), "A, C, D")
    assert result.reason.endswith("(1 missing, 2 extra)")


def test_grade_refuses_missing_empty_and_linked_answer_files(tmp_path: Path) -> None:
    """No file, an empty file and a symlink all score 0."""
    task = make_task("13", "number")
    assert grade(task, tmp_path).reason == "no answer.txt"
    (tmp_path / "answer.txt").write_text("\n \n")
    assert grade(task, tmp_path).reason == "no answer.txt"
    (tmp_path / "answer.txt").unlink()
    (tmp_path / "secret").write_text("13")
    (tmp_path / "answer.txt").symlink_to(tmp_path / "secret")
    assert grade(task, tmp_path).reason == "answer.txt is not a regular file"


def test_grade_tolerates_a_byte_order_mark(tmp_path: Path) -> None:
    """A file saved with a byte-order mark still grades."""
    (tmp_path / "answer.txt").write_bytes("\ufeff13\n".encode())
    assert grade(make_task("13", "number"), tmp_path).correct


def test_the_six_grade_cases_have_the_verdicts_the_lecture_shows() -> None:
    """Three are forgiven, three are wrong, in the lecture's order."""
    verdicts = [grade_cases.judge(case)[0] for case in grade_cases.CASES]
    assert verdicts == [True, True, True, False, False, False]


def test_every_bundled_gold_query_reproduces_its_answer() -> None:
    """Running each task's gold SQL gives an answer the grader accepts."""
    for task in tasks.load_tasks():
        database = tasks.DATA_DIR / task.db_file
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            rows = connection.execute(task.gold_sql).fetchall()
        if task.answer_type == "list":
            text = ", ".join(str(row[0]) for row in rows)
        else:
            text = str(rows[0][0])
        assert check_answer(task, text).correct, task.task_id


def test_bundle_has_180_train_and_50_test_tasks() -> None:
    """The bundled data is the project's full task set."""
    splits = [task.split for task in tasks.load_tasks()]
    assert (splits.count("train"), splits.count("test")) == (180, 50)


def test_instruction_names_the_files_but_never_a_table() -> None:
    """Exploring the schema is part of the job."""
    task = tasks.find_task("shop-easy-001")
    text = tasks.task_instruction(task)
    assert "data.db" in text and "answer.txt" in text
    assert task.question in text
    assert "customers" not in text.replace(task.question, "")


def test_new_folder_holds_only_the_database(course_home: Path) -> None:
    """The folder an agent gets has no answer in it."""
    workdir = tasks.prepare_workdir(tasks.find_task("shop-easy-001"))
    assert workdir.parent == course_home / "01-the-job" / "work"
    assert [path.name for path in workdir.iterdir()] == ["data.db"]
    assert tasks.task_for_workdir(workdir).task_id == "shop-easy-001"
    assert tasks.latest_workdir() == workdir


def test_doing_the_job_by_hand_earns_reward_one() -> None:
    """Paris has 13 customers, and writing 13 is correct."""
    task = tasks.find_task("shop-easy-001")
    workdir = tasks.prepare_workdir(task)
    with sqlite3.connect(workdir / "data.db") as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM customers WHERE city = 'Paris'"
        ).fetchone()
    (workdir / "answer.txt").write_text(f"{count}\n")
    assert grade(task, workdir).reward == 1


def test_unknown_task_and_missing_work_folder_are_clear_errors() -> None:
    """Bad input gives a plain message, not a traceback."""
    with pytest.raises(tasks.TaskNotFoundError):
        tasks.find_task("shop-easy-999")
    with pytest.raises(tasks.TaskNotFoundError):
        tasks.latest_workdir()


def test_same_seed_same_tasks_and_rules_hold() -> None:
    """The generator is deterministic and every draft is fair."""
    first, _ = make_tasks.draft_tasks(seed=0, count=9)
    second, _ = make_tasks.draft_tasks(seed=0, count=9)
    other, _ = make_tasks.draft_tasks(seed=1, count=9)
    assert first == second
    assert first != other
    assert [d.difficulty for d in first] == ["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3
    answers = [d.answer for d in first]
    assert all(
        answers.count(answer) <= make_tasks.MAX_SAME_ANSWER for answer in answers
    )
    for draft in first:
        if draft.answer_type == "number":
            assert float(draft.answer) >= make_tasks.MIN_INTEGER_ANSWER


def test_make_tasks_prints_the_same_md5_twice(course_home: Path) -> None:
    """Running the script twice writes byte-identical files."""
    script = CHAPTER / "make_tasks.py"
    runs = [
        subprocess.run(
            [sys.executable, str(script), "--seed", "0"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    md5_lines = [line for line in runs[0].splitlines() if line.startswith("md5")]
    assert md5_lines and md5_lines == [
        line for line in runs[1].splitlines() if line.startswith("md5")
    ]
    assert (course_home / "01-the-job" / "made-tasks" / "seed-0.jsonl").is_file()


def test_tie_and_small_answers_are_rejected() -> None:
    """A guessable draft is thrown away with its reason."""
    template = make_tasks.TEMPLATES[0]
    ranking_tie = [("A", 5), ("B", 5)]
    used = make_tasks.Counter()
    assert make_tasks.rejection(template, "1", [], used) == "answer 0 or 1"
    assert make_tasks.rejection(template, "7", ranking_tie, used) == "tie for first"
    used["7"] = make_tasks.MAX_SAME_ANSWER
    assert make_tasks.rejection(template, "7", [], used) == "answer used too often"


def test_replays_grade_as_recorded() -> None:
    """mini-swe-agent's run is correct; Claude Code's never wrote the file."""
    recorded = json.loads(replay_agent.TRANSCRIPTS.read_text(encoding="utf-8"))
    mini, claude = recorded["runs"]
    assert replay_agent.replay_and_grade(mini, recorded["question"], 0.0).reward == 1
    claude_grade = replay_agent.replay_and_grade(claude, recorded["question"], 0.0)
    assert (claude_grade.reward, claude_grade.reason) == (0, "no answer.txt")
    rejected_tools = [
        e.get("tool") for e in claude["events"] if e["kind"] == "rejected"
    ]
    assert rejected_tools == ["bash", "bash", "write"]


def test_transcripts_carry_no_personal_paths() -> None:
    """The recorded logs were scrubbed of the recording machine's paths."""
    for path in (tasks.DATA_DIR / "transcripts").iterdir():
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text and "/private/var/" not in text, path.name


def test_missing_command_reads_as_none() -> None:
    """The machine check survives a program that is not installed."""
    assert check_machine.run(["no-such-program-anywhere"]) is None


def test_short_path_only_shortens_paths_inside_home() -> None:
    """A sibling folder that merely starts like home keeps its full path."""
    home = Path.home()
    assert tasks.short_path(home) == "~"
    assert tasks.short_path(home / "a" / "b") == "~/a/b"
    sibling = home.with_name(home.name + "y") / "x"
    assert tasks.short_path(sibling) == str(sibling)


def test_unreadable_answer_scores_zero_instead_of_crashing(tmp_path: Path) -> None:
    """An answer.txt nobody may read is a wrong answer, not a crash."""
    answer = tmp_path / "answer.txt"
    answer.write_text("13")
    answer.chmod(0)
    try:
        result = grade(make_task("13", "number"), tmp_path)
    finally:
        answer.chmod(0o644)
    assert (result.correct, result.reason) == (False, "answer.txt could not be read")


def test_a_folder_named_answer_txt_is_refused(tmp_path: Path) -> None:
    """Only a plain file counts as an answer."""
    (tmp_path / "answer.txt").mkdir()
    reason = grade(make_task("13", "number"), tmp_path).reason
    assert reason == "answer.txt is not a regular file"


def test_unknown_answer_type_is_an_error() -> None:
    """A typo in a task's answer type fails loudly instead of grading as a list."""
    with pytest.raises(ValueError, match="unknown answer type"):
        check_answer(make_task("13", "nmber"), "13")


def test_an_apostrophe_stays_single_in_the_question() -> None:
    """Quotes are doubled inside SQL text only."""
    assert make_tasks.sql_literal("O'Brien") == "O''Brien"
