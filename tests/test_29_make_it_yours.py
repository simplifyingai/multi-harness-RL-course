"""Chapter 29 without a model: a new task family, a fourth harness, and power.

The unit tests check the CSV family (the spreadsheet read with types, the
seeded generator's fairness rules and md5, every gold answer recomputed by
plain Python and by the ``sqlite3`` command the scripted engine types, and
chapter 1's grader on its answers), Qwen Code's wiring and pin, and
``power.py``'s exact arithmetic (against ``stats.exact_p``, against a
brute-force sum over every outcome, against a seeded simulation, and its
whole output, pinned in ``tests/data/29_power.txt``). The end-to-end tests
run ``try_family.py`` as a learner does, through OpenEnv's real capture proxy
(started by ``start.sh``) in front of the scripted engine over real HTTP,
with mini-swe-agent and with Qwen Code inside the sandbox. They are skipped
when the proxy or a harness is not on disk (``./start.sh install`` and
``uv run install.py``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("29-make-it-yours")

import engine  # noqa: E402
import evaluate  # noqa: E402
import grader  # noqa: E402
import install  # noqa: E402
import lfm_files  # noqa: E402
import make_csv_tasks  # noqa: E402
import power  # noqa: E402
import qwen_probe  # noqa: E402
import run_one  # noqa: E402
import sandbox  # noqa: E402
import scripted_engine  # noqa: E402
import sheet  # noqa: E402
import stats  # noqa: E402
import tasks  # noqa: E402
import try_family  # noqa: E402
import wire  # noqa: E402

DATA = Path(__file__).parent / "data"
POWER_GOLDEN = DATA / "29_power.txt"
QWEN_LOG = DATA / "29_qwen_stream.jsonl"
CHAPTER_01 = ROOT / "01-the-job"
CHAPTER_24 = ROOT / "24-measuring"
START_SH = CHAPTER / "start.sh"
SEED_0_MD5 = "cd9bdf067046038d76c7fcffe90f5ef4"
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy and the harnesses are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
CHAPTERS_WITH_STACK = ("29-make-it-yours", "24-measuring", "23-three-harnesses")
START_TIMEOUT_S = 300
# The modules this chapter copies from chapter 24 unchanged but for its name.
COPIED = (
    "capture_client.py",
    "engine.py",
    "evaluate.py",
    "fetch_model.py",
    "lfm.py",
    "lfm_files.py",
    "lfm_template.py",
    "lfm_text.py",
    "loopback.py",
    "mini-swe-agent-requirements.txt",
    "policy.py",
    "proxy-requirements.txt",
    "sandbox.py",
    "sandbox.sb",
    "scripted_engine.py",
    "scripted_replies.py",
    "stats.py",
    "toolcalls.py",
)


def run_main(
    module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> str:
    """Run a script's main() as from the command line, and capture its output.

    Args:
        module: The script's module.
        argv: Its command line, script name first.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.

    Returns:
        What it printed.
    """
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return capsys.readouterr().out


def record(task_id: str, correct: bool, **extra: Any) -> evaluate.Record:
    """A record for one task in mini-swe-agent.

    Args:
        task_id: The task.
        correct: The verdict.
        **extra: Other fields.

    Returns:
        The record.
    """
    return evaluate.Record(
        task_id=task_id, harness="mini-swe-agent", correct=correct, **extra
    )


@pytest.fixture
def seed_0() -> list[tasks.Task]:
    """The family's 20 tasks of seed 0, made in the test's COURSE_HOME.

    Returns:
        The tasks.
    """
    return make_csv_tasks.ensure_tasks(0, 20)


# ------------------------------------------------- what is copied, and from where


def test_the_copied_modules_are_chapter_24s_but_for_the_folder_name() -> None:
    for name in COPIED:
        theirs = (CHAPTER_24 / name).read_text(encoding="utf-8")
        ours = (CHAPTER / name).read_text(encoding="utf-8")
        assert ours == theirs.replace("24-measuring", "29-make-it-yours"), name


def test_the_grader_is_chapter_1s_byte_for_byte() -> None:
    ours = (CHAPTER / "grader.py").read_bytes()
    assert ours == (CHAPTER_01 / "grader.py").read_bytes()


def test_start_sh_is_chapter_24s_plus_one_line() -> None:
    def code(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8").replace(
            "24-measuring", "29-make-it-yours"
        )
        return [line for line in text.splitlines() if not line.lstrip().startswith("#")]

    theirs, ours = code(CHAPTER_24 / "start.sh"), code(START_SH)
    added = [line for line in ours if line not in theirs]
    assert [line for line in theirs if line not in ours] == []
    assert len(added) == 1 and added[0].startswith("export UV_NO_CONFIG=1")


def test_the_instruction_changes_only_its_first_sentence() -> None:
    first = (CHAPTER_01 / "tasks.py").read_text(encoding="utf-8")
    theirs = re.search(r"INSTRUCTION_TEMPLATE = \((.*?)\n\)", first, re.DOTALL)
    ours = re.search(
        r"INSTRUCTION_TEMPLATE = \((.*?)\n\)",
        (CHAPTER / "tasks.py").read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert theirs and ours
    old, new = theirs[1].strip().splitlines(), ours[1].strip().splitlines()
    assert old[1:] == new[1:]
    assert "SQLite database named {database}" in old[0]
    assert "CSV file named {data_file}" in new[0]


def test_only_the_mini_swe_agent_config_names_the_job() -> None:
    ours = (CHAPTER / "mini_swe_agent.yaml").read_text(encoding="utf-8")
    theirs = (CHAPTER_24 / "mini_swe_agent.yaml").read_text(encoding="utf-8")
    body = ours[ours.index("agent:") :]
    assert "data.db" not in body and "head -5 data.csv" in body
    # Below the header, every line the project's file has and ours lacks is
    # one that names the SQLite job; nothing else changed.
    project = theirs[theirs.index("agent:") :].splitlines()
    removed = [line for line in project if line not in body.splitlines()]
    words = ("sqlite", "SQL", "database", "SELECT", "tables")
    assert len(removed) == 9
    assert all(any(word in line for word in words) for line in removed), removed


# ------------------------------------------------- the spreadsheet


def test_the_sheet_is_96_orders_read_with_types() -> None:
    rows = sheet.read_rows()
    assert len(rows) == 96 and rows[0]["order_id"] == "1001"
    connection = sheet.open_sheet()
    types = connection.execute(
        "SELECT DISTINCT typeof(units), typeof(unit_price), typeof(order_id) FROM sales"
    ).fetchall()
    assert types == [("integer", "real", "integer")]
    # As text, '9' sorts after '10': the typed table compares numbers.
    biggest = connection.execute("SELECT MAX(units) FROM sales").fetchone()[0]
    assert biggest == max(int(row["units"]) for row in rows) == 12


def test_a_sheet_with_other_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("id,amount\n1,2\n", encoding="utf-8")
    with pytest.raises(sheet.SheetError, match="header"):
        sheet.read_rows(path)


def test_a_half_readable_row_is_refused_never_loaded(tmp_path: Path) -> None:
    header = sheet.SHEET.read_text(encoding="utf-8").splitlines()[0]
    good = "1001,2026-01-05,South,Chloe Martin,Desk Lamp,Hardware,6,24.50"
    bad_rows = {
        "1002,2026-01-06,North,Asha Rao,Desk Lamp,Hardware,six,24.50": "not a number",
        "1002,2026-01-06,North,Asha Rao,Desk Lamp,Hardware,,24.50": "empty field",
        "1002,2026-01-06,North,Asha Rao,Desk Lamp,Hardware,6": "empty field",
        "1002,2026-01-06,North,Asha Rao,Desk Lamp,Hardware,6,24.50,x": "extra",
    }
    for row, why in bad_rows.items():
        path = tmp_path / "sheet.csv"
        path.write_text(f"{header}\n{good}\n{row}\n", encoding="utf-8")
        with pytest.raises(sheet.SheetError, match=rf"sheet.csv:3: .*{why}"):
            sheet.read_rows(path)


@pytest.mark.skipif(shutil.which("sqlite3") is None, reason="needs sqlite3")
def test_the_solving_command_keeps_a_file_name_with_a_space_whole(
    tmp_path: Path,
) -> None:
    shutil.copyfile(sheet.SHEET, tmp_path / "my sales.csv")
    command = sheet.gold_command("SELECT COUNT(*) FROM sales", "my sales.csv")
    printed = subprocess.run(
        command, shell=True, cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    assert printed.strip() == "96"


# ------------------------------------------------- the generator


def test_seed_0_makes_20_fair_tasks_with_a_fixed_md5(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(make_csv_tasks, ["make_csv_tasks.py"], monkeypatch, capsys)
    assert out.startswith("seed 0 · 20 tasks from 12 templates · over sales.csv")
    assert f"md5   {SEED_0_MD5}" in out
    assert "answers   13 number, 3 text, 4 list" in out
    made = tasks.load_tasks(tasks.tasks_path(0))
    assert Counter(t.difficulty for t in made) == {"easy": 7, "medium": 7, "hard": 6}
    assert len({t.task_id for t in made}) == 20
    assert len({t.question for t in made}) == 20
    assert made[0].task_id == "csv-s0-easy-01"
    for task in made:
        assert Counter(t.answer for t in made)[task.answer] <= 2
        if task.answer_type == tasks.LIST:
            assert 2 <= len(task.answer.split(", ")) <= 8
        if task.answer_type == tasks.NUMBER:
            assert float(task.answer) >= 2


def test_the_same_seed_writes_the_same_bytes_and_another_seed_does_not() -> None:
    first, _ = make_csv_tasks.draft_tasks(0, 20)
    again, _ = make_csv_tasks.draft_tasks(0, 20)
    other, _ = make_csv_tasks.draft_tasks(1, 20)
    text = make_csv_tasks.tasks_text
    assert text(first) == text(again)
    assert [t.question for t in other] != [t.question for t in first]
    assert other[0].task_id == "csv-s1-easy-01"


def test_a_missing_task_file_is_made_before_it_is_read(
    course_home: Path, capsys: Any
) -> None:
    assert not tasks.tasks_path(3).exists()
    made = make_csv_tasks.ensure_tasks(3, 6)
    assert len(made) == 6 and tasks.tasks_path(3).is_file()
    assert not list(tasks.tasks_path(3).parent.glob("*.partial"))
    assert make_csv_tasks.ensure_tasks(3) == made  # read, never remade
    capsys.readouterr()
    assert make_csv_tasks.ensure_tasks(3, 20) == made  # another count: said
    assert "holds 6 tasks, not the 20 asked" in capsys.readouterr().out


def test_a_task_file_with_an_unknown_shape_is_refused(tmp_path: Path) -> None:
    drafted, _ = make_csv_tasks.draft_tasks(0, 3)
    text = make_csv_tasks.tasks_text(drafted).replace('"number"', '"numbr"', 1)
    path = tmp_path / "seed-0.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"seed-0.jsonl:\d: answer type 'numbr'"):
        tasks.load_tasks(path)


def test_a_shortfall_is_reported_not_hidden(course_home: Path, capsys: Any) -> None:
    drafted, rejected = make_csv_tasks.draft_tasks(0, 60)
    assert len(drafted) < 60
    assert rejected[make_csv_tasks.NO_FAIR_DRAFT] == 60 - len(drafted)
    make_csv_tasks.ensure_tasks(0, 60)
    assert f"only {len(drafted)} of 60: " in capsys.readouterr().out


def python_answer(task: tasks.Task, rows: list[dict[str, str]]) -> str:
    """Recompute a task's answer with plain Python, from the question alone.

    Args:
        task: The task.
        rows: The spreadsheet's rows.

    Returns:
        The answer, written as the generator writes it.

    Raises:
        AssertionError: For a question this test does not know.
    """
    q = task.question

    def worth(row: dict[str, str]) -> float:
        return int(row["units"]) * float(row["unit_price"])

    def where(**match: str) -> list[dict[str, str]]:
        return [r for r in rows if all(r[k] == v for k, v in match.items())]

    if m := re.fullmatch(r"How many orders came from the (\w+) region\?", q):
        return str(len(where(region=m[1])))
    if m := re.fullmatch(r"How many units of (.+) were sold in total\?", q):
        return str(sum(int(r["units"]) for r in where(product=m[1])))
    if m := re.fullmatch(r"How many orders did (.+) take\?", q):
        return str(len(where(rep=m[1])))
    if m := re.fullmatch(r"What is the highest unit price .* a (\w+) product\?", q):
        return f"{max(float(r['unit_price']) for r in where(category=m[1])):.2f}"
    if m := re.fullmatch(r"What was the total revenue .* in the (\w+) region\?", q):
        if m2 := re.search(r"from (.+) in the", q):
            return f"{sum(worth(r) for r in where(product=m2[1], region=m[1])):.2f}"
        return f"{sum(worth(r) for r in where(region=m[1])):.2f}"
    if m := re.fullmatch(r"Which sales rep sold the most units in the (\w+) .*", q):
        totals = Counter()
        for r in where(region=m[1]):
            totals[r["rep"]] += int(r["units"])
        return totals.most_common(1)[0][0]
    if m := re.fullmatch(r"Which (\w+) product brought in the most revenue .*", q):
        totals = Counter()
        for r in where(category=m[1]):
            totals[r["product"]] += worth(r)
        return totals.most_common(1)[0][0]
    if m := re.fullmatch(r"What is the average number of units .* for (.+)\?", q):
        units = [int(r["units"]) for r in where(product=m[1])]
        return f"{sum(units) / len(units):.2f}"
    if m := re.fullmatch(r"Which sales reps took no orders in the (\w+) region\?", q):
        inside = {r["rep"] for r in where(region=m[1])}
        return ", ".join(sorted({r["rep"] for r in rows} - inside))
    if m := re.fullmatch(r"Which sales reps sold more than (\d+) units of (\w+) .*", q):
        totals = Counter()
        for r in where(category=m[2]):
            totals[r["rep"]] += int(r["units"])
        return ", ".join(sorted(rep for rep, n in totals.items() if n > int(m[1])))
    if m := re.fullmatch(r"How many orders were worth more than (\d+) .*", q):
        return str(sum(worth(r) > int(m[1]) for r in rows))
    raise AssertionError(f"unknown question: {q}")


def test_every_gold_answer_matches_plain_python_on_the_csv_file() -> None:
    rows = sheet.read_rows()
    for seed in range(4):
        drafted, _ = make_csv_tasks.draft_tasks(seed, 20)
        for task in drafted:
            expected = python_answer(task, rows)
            assert grader.check_answer(task, expected).correct, (task, expected)


@pytest.mark.skipif(shutil.which("sqlite3") is None, reason="needs sqlite3")
def test_the_scripted_solving_command_prints_each_gold_answer(
    seed_0: list[tasks.Task], tmp_path: Path
) -> None:
    for task in seed_0:
        folder = tasks.prepare_workdir(task, tmp_path / task.task_id)
        command = sheet.gold_command(task.gold_sql, tasks.DATA_FILE)
        printed = subprocess.run(
            command, shell=True, cwd=folder, capture_output=True, text=True, check=True
        ).stdout
        assert grader.check_answer(task, printed).correct, (task.task_id, printed)


def test_chapter_1s_rules_grade_the_family(seed_0: list[tasks.Task]) -> None:
    revenue = next(t for t in seed_0 if t.answer == "12160.19")
    names = next(t for t in seed_0 if t.answer_type == tasks.LIST)
    rep = next(t for t in seed_0 if t.answer == "Ben Ortiz")
    assert grader.check_answer(revenue, "$12,160.19\n").correct
    assert grader.check_answer(revenue, "12160.2").correct  # within 0.01
    assert not grader.check_answer(revenue, "12160.17").correct
    shuffled = "\n".join(reversed(names.answer.split(", ")))
    assert grader.check_answer(names, shuffled).correct
    assert not grader.check_answer(names, names.answer.split(", ")[0]).correct
    assert grader.check_answer(rep, '"ben ortiz."').correct


def test_a_work_folder_holds_only_the_csv_file(
    seed_0: list[tasks.Task], tmp_path: Path
) -> None:
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "answer.txt").write_text("old")
    made = tasks.prepare_workdir(seed_0[0], folder)
    assert sorted(p.name for p in made.iterdir()) == ["data.csv"]
    assert (made / "data.csv").read_bytes() == sheet.SHEET.read_bytes()
    instruction = tasks.task_instruction(seed_0[0])
    assert instruction.startswith("There is a CSV file named data.csv")
    assert seed_0[0].question in instruction and "gold" not in instruction.lower()


# ------------------------------------------------- the fourth harness


def test_qwen_code_is_pinned_and_run_by_node() -> None:
    pin = install.PINS[install.QWEN]
    assert (pin.version, pin.package) == ("0.24.7", "@qwen-code/qwen-code")
    assert install.HARNESSES == ("mini-swe-agent", "claude-code", "pi", "qwen-code")
    argv = install.program(install.QWEN)
    assert argv[0].endswith("node") and argv[1].endswith(
        "@qwen-code/qwen-code/cli-entry.js"
    )
    command = install.install_commands(install.QWEN)[0]
    assert command[:3] == ["npm", "install", "--prefix"]
    env = install.npm_environment()
    assert env["npm_config_registry"] == "https://registry.npmjs.org/"
    assert env["HOME"] != str(Path.home())


def test_qwen_code_is_wired_by_base_url_and_key_only(tmp_path: Path) -> None:
    folders = wire.RunFolders(task=tmp_path / "task", state=tmp_path / "state")
    wiring = wire.wire(
        install.QWEN, folders, "the task", "http://127.0.0.1:8239/", "s0123"
    )
    env = wiring.env
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8239/v1"
    assert env["OPENAI_API_KEY"] == "s0123" and env["OPENAI_MODEL"] == "lfm2.5-2.6b"
    assert env["QWEN_USAGE_STATISTICS_ENABLED"] == "false"
    assert env["QWEN_TELEMETRY_ENABLED"] == "false"
    assert env["HOME"] == str(folders.home) and env["TMPDIR"] == str(folders.tmp)
    assert not set(env) & {"ANTHROPIC_API_KEY", "OPENAI_ORG_ID", "USERPROFILE"}
    argv = wiring.argv
    assert argv[2] == "the task" and "--bare" in argv
    assert argv[argv.index("--approval-mode") + 1] == "yolo"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    excluded = argv[argv.index("--exclude-tools") + 1].split(",")
    # Excluding these would also refuse shell commands that read or edit.
    assert not {"read_file", "edit", "run_shell_command"} & set(excluded)
    assert wiring.files == {} and wiring.stdout_name == "qwen-stream.jsonl"


def test_the_solving_script_speaks_each_harness_tool_name(
    seed_0: list[tasks.Task],
) -> None:
    task = seed_0[0]
    for harness, tool in run_one.BASH_TOOL.items():
        replies = run_one.solve_script(harness, task)
        assert all(f"{tool}(" in r.text for r in replies[:3])
        assert "head -5 data.csv" in replies[0].text
        assert "sqlite3 :memory:" in replies[1].text
        ends_with_command = run_one.FINISH_COMMAND in replies[-1].text
        assert ends_with_command == (harness == install.MINI)
    assert run_one.BASH_TOOL[install.QWEN] == "run_shell_command"


def test_qwen_codes_own_log_is_read(tmp_path: Path) -> None:
    says, errors, last = run_one.read_qwen_log(QWEN_LOG)
    assert says == "result success · 4 turns"
    assert errors == []
    assert last.strip() == "18"
    unnumbered = tmp_path / "qwen-stream.jsonl"
    unnumbered.write_text(json.dumps({"type": "result", "subtype": "success"}))
    assert run_one.read_qwen_log(unnumbered)[0] == "result success · ? turns"
    assert run_one.read_qwen_log(tmp_path / "none")[0] == "no result event in the log"


def test_a_refused_tool_call_is_counted(tmp_path: Path) -> None:
    log = tmp_path / "qwen-stream.jsonl"
    events = [
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": 'This "run_shell_command" invocation was denied '
                        'by permission rules. Matching deny rule: "read_file".',
                    }
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "num_turns": 3,
            "permission_denials": [{"tool_name": "run_shell_command"}],
        },
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    says, errors, _ = run_one.read_qwen_log(log)
    assert says == "result success · 3 turns · 1 refused by its permission rules"
    assert errors[0].startswith('This "run_shell_command" invocation was denied')


def test_every_run_hides_the_folder_of_gold_answers(
    course_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(**kwargs: Any) -> dict[str, str]:
        seen.update(kwargs)
        raise sandbox.SandboxError("captured")

    monkeypatch.setattr(run_one, "sandbox_params", capture)
    folders = wire.RunFolders(task=tmp_path / "task", state=tmp_path / "state")
    wiring = wire.wire(install.QWEN, folders, "t", "http://127.0.0.1:1", "k")
    with pytest.raises(install.HarnessInstallError):
        run_one.run_boxed(wiring, folders, 1)
    assert seen["tasks_dir"] == tasks.gold_folder() == tasks.tasks_path(0).parent


@pytest.mark.skipif(not sandbox.SANDBOX_EXEC.exists(), reason="needs sandbox-exec")
def test_the_box_refuses_to_read_the_gold_answers(
    seed_0: list[tasks.Task], tmp_path: Path
) -> None:
    task, state = tmp_path / "task", tmp_path / "state"
    for folder in (task, state):
        folder.mkdir()
    gold = str(tasks.tasks_path(0))
    hidden = sandbox.sandbox_params(
        workdir=task, state_dir=state, tasks_dir=tasks.gold_folder()
    )
    refused = subprocess.run(
        sandbox.sandbox_command(["cat", gold], params=hidden),
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0 and "Operation not permitted" in refused.stderr
    # Chapter 24's default hides data/, where this family's gold is not.
    default = sandbox.sandbox_params(workdir=task, state_dir=state)
    read = subprocess.run(
        sandbox.sandbox_command(["cat", gold], params=default),
        capture_output=True,
        text=True,
        check=False,
    )
    assert read.returncode == 0 and seed_0[0].question in read.stdout


# ------------------------------------------------- qwen_probe.py


def test_the_stand_in_model_looks_writes_then_stops() -> None:
    first, second, third = (qwen_probe.reply_for(i) for i in range(3))
    commands = [
        json.loads(m["tool_calls"][0]["function"]["arguments"])["command"]
        for m in (first, second)
    ]
    assert commands == ["head -3 data.csv", "echo 42 > answer.txt"]
    assert third == qwen_probe.reply_for(9) == {"role": "assistant", "content": "42"}
    start, end = qwen_probe.chunks_for(first)
    assert start["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert end["choices"][0]["finish_reason"] == "tool_calls"
    assert qwen_probe.chunks_for(third)[1]["choices"][0]["finish_reason"] == "stop"


def test_a_probe_variant_changes_one_thing(tmp_path: Path) -> None:
    folders = wire.RunFolders(task=tmp_path / "task", state=tmp_path / "state")
    base = wire.wire(install.QWEN, folders, "t", "http://127.0.0.1:1", "k")
    proxy = "http://127.0.0.1:2"
    own = qwen_probe.variant(base, proxy, switches=False, bare=True)
    assert not set(own.env) & set(qwen_probe.QWEN_SWITCHES)
    assert (
        own.env["HTTPS_PROXY"] == proxy and own.env["NO_PROXY"] == "127.0.0.1,localhost"
    )
    assert own.argv == base.argv
    plain = qwen_probe.variant(base, proxy, switches=True, bare=False)
    assert set(qwen_probe.QWEN_SWITCHES) <= set(plain.env)
    assert "--bare" not in plain.argv and len(plain.argv) == len(base.argv) - 1
    less = qwen_probe.variant(base, proxy, switches=True, bare=True, also_excluded="x")
    excluded = less.argv[less.argv.index("--exclude-tools") + 1].split(",")
    assert excluded == [*wire.QWEN_EXCLUDED, "x"]


def test_the_probe_reads_requests_and_outside_connections() -> None:
    tool = {"type": "function", "function": {"name": "run_shell_command"}}
    side = {"type": "function", "function": {"name": "glob"}}
    probe = qwen_probe.Probe(
        name="x",
        requests=[
            {"messages": [{"role": "user", "content": "hi"}], "tools": [tool],
             "max_tokens": 32000},
            {"messages": [], "tools": [side], "max_tokens": 32000},
        ],
        outside=["CONNECT a.example:443 HTTP/1.1", "CONNECT a.example:443 HTTP/1.1"],
    )  # fmt: skip
    lines = qwen_probe.request_lines(probe)
    assert lines[0] == "  requests       2"
    assert lines[2] == "  tools offered  1: run_shell_command"
    assert lines[3] == "  max_tokens     32000 asked"
    assert lines[4].endswith("other tools (after the last reply): glob")
    assert qwen_probe.outside_lines(probe) == [
        "  outside connections asked for: 2",
        "    a.example:443 (refused by the catcher)",
    ]
    empty = qwen_probe.Probe(name="y")
    assert qwen_probe.request_lines(empty) == [
        "  no request reached the stand-in model"
    ]


# ------------------------------------------------- power.py


def test_the_rejection_limits_agree_with_exact_p() -> None:
    limits = power.rejection_limits(power.EXACT_MAX_TASKS)
    assert limits[:7] == [-1, -1, -1, -1, -1, -1, 0]  # 5 to 0 has p = 0.0625
    for m in range(201):
        for gained in range(m + 1):
            rejects = min(gained, m - gained) <= limits[m]
            assert rejects == (stats.exact_p(gained, m - gained) < 0.05), (m, gained)
    # Far out, the edge alone: the limit's split is under 0.05, one more is not.
    for m in (999, 1000, 1999, power.EXACT_MAX_TASKS):
        k = limits[m]
        assert stats.exact_p(k, m - k) < 0.05 <= stats.exact_p(k + 1, m - k - 1)


def brute_force_power(tasks_count: int, flips: power.Flips) -> float:
    """Sum the chance of every outcome on a few tasks in which p < 0.05.

    Args:
        tasks_count: Paired tasks.
        flips: How the checkpoints disagree.

    Returns:
        The power, from every (gained, lost, unchanged) sequence.
    """
    total = 0.0
    same = 1 - flips.flipped
    for outcome in product("gls", repeat=tasks_count):
        gained, lost = outcome.count("g"), outcome.count("l")
        if stats.exact_p(gained, lost) < 0.05:
            total += (
                flips.gained**gained
                * flips.lost**lost
                * same ** (tasks_count - gained - lost)
            )
    return total


def test_the_exact_power_is_the_sum_over_every_outcome() -> None:
    limits = power.rejection_limits(power.EXACT_MAX_TASKS)
    cases = (power.Flips(0.3, 0.05), power.Flips(0.25, 0.0), power.Flips(0.4, 0.3))
    for flips in cases:
        calculator = power.PowerCalculator(flips, limits)
        for count in (6, 9):
            assert calculator.power(count) == pytest.approx(
                brute_force_power(count, flips), abs=1e-12
            )


def test_the_plans_example_measured() -> None:
    limits = power.rejection_limits(power.EXACT_MAX_TASKS)
    churn = 68 / 216
    needed = {
        name: power.tasks_needed(flips, 0.8, limits)
        for name, flips in power.assumptions(0.6, 0.1, churn)
        if flips is not None
    }
    assert {n: a.tasks for n, a in needed.items()} == {
        "only gains": 78,
        "chapter 24's churn": 261,
        "independent attempts": 377,
    }
    assert all(a.method == "exact" and a.power >= 0.8 for a in needed.values())
    flips = power.with_churn(0.6, 0.1, churn)
    assert flips is not None
    assert power.PowerCalculator(flips, limits).power(150) == pytest.approx(
        0.544, abs=5e-4
    )
    assert power.normal_tasks(flips, 0.8) == 245


def test_the_simulation_agrees_with_the_exact_power() -> None:
    limits = power.rejection_limits(400)
    flips = power.with_churn(0.6, 0.1, 68 / 216)
    assert flips is not None
    exact = power.PowerCalculator(flips, limits).power(120)
    simulated = power.simulate_power(flips, 120, 4_000, seed=0)
    assert simulated == power.simulate_power(flips, 120, 4_000, seed=0)  # seeded
    assert abs(simulated - exact) < 0.025


def test_a_gain_too_small_for_the_exact_scan_is_approximated_and_says_so() -> None:
    limits = power.rejection_limits(power.EXACT_MAX_TASKS)
    answer = power.tasks_needed(power.with_churn(0.6, 0.02, 0.3), 0.8, limits)
    assert answer.method == "normal approximation" and answer.power is None
    assert answer.tasks > power.EXACT_MAX_TASKS


def test_a_churn_that_cannot_hold_the_gain_is_refused() -> None:
    assert power.with_churn(0.8, 0.1, 0.315) is None  # gains above the 20% unsolved
    assert power.with_churn(0.6, 0.4, 0.3) is None  # fewer flips than the gain


def test_chapter_24s_pairs_are_its_records() -> None:
    folder = CHAPTER_24 / "data" / "project_records"

    def verdicts(name: str) -> dict[tuple[str, str], bool]:
        rows = [json.loads(line) for line in (folder / name).read_text().splitlines()]
        return {(r["harness"], r["task_id"]): r["correct"] for r in rows}

    checkpoints = {}
    for label in ("base", "sft", "rl"):
        ours = verdicts(f"eval_{label}.jsonl")
        mine = {k: v for k, v in ours.items() if k[0] != "pi"}
        pi = verdicts(f"eval_{label}-pi-fixed.jsonl")  # Pi's re-run, as chapter 24
        checkpoints[label] = mine | {k: v for k, v in pi.items() if k[0] == "pi"}
    for row in power.read_pairs():
        first, second = str(row["pair"]).split(" -> ")
        cells = [k for k in checkpoints[first] if k[0] == row["harness"]]
        before, after = checkpoints[first], checkpoints[second]
        assert row["cells"] == len(cells)
        assert row["gained"] == sum(after[c] and not before[c] for c in cells)
        assert row["lost"] == sum(before[c] and not after[c] for c in cells)
        assert row["before"] == sum(before[c] for c in cells)
        assert row["after"] == sum(after[c] for c in cells)
    assert power.churn_share(power.read_pairs()) == (68, 216)


def test_power_prints_every_reading_exactly(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(power, ["power.py"], monkeypatch, capsys)
    assert out == POWER_GOLDEN.read_text(encoding="utf-8")


def test_the_one_line_answer_names_its_assumption() -> None:
    assert power.one_line(0.65, 0.1) == (
        "to see +10 points from 65.0% at p < 0.05, 80% of the time: about 261 "
        "paired tasks (exact McNemar, with chapter 24's 31.5% of tasks flipping)"
    )
    assert "independent attempts" in power.one_line(0.85, 0.1)
    assert power.one_line(0.95, 0.1) == "no room for +10 points from 95.0%"
    tiny = power.one_line(0.6, 0.001)
    assert "more than 2,000 paired tasks (normal approximation" in tiny
    assert "+0.1 points" in tiny
    with pytest.raises(ValueError, match="above 0"):
        power.one_line(0.6, 0.0)
    with pytest.raises(ValueError, match="no gain"):
        power.normal_tasks(power.Flips(0.1, 0.1), 0.8)


def test_a_share_is_read_as_a_fraction_or_a_percentage() -> None:
    assert power.share_argument("0.6") == power.share_argument("60") == 0.6
    assert power.share_argument("10%") == pytest.approx(0.1)
    assert power.share_argument("1%") == pytest.approx(0.01)
    for bad in ("-5", "150", "lots", "1"):
        with pytest.raises(power.argparse.ArgumentTypeError):
            power.share_argument(bad)


# ------------------------------------------------- try_family.py


def test_the_plan_prints_the_protocol_and_the_tasks(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(try_family, ["try_family.py", "--plan"], monkeypatch, capsys)
    assert "made      20 tasks" in out and SEED_0_MD5 in out
    assert "20 of 20 CSV tasks (seed 0" in out and "7 easy, 7 medium, 6 hard" in out
    assert "harness   mini-swe-agent 2.4.6" in out
    assert "family/base/mini-swe-agent.jsonl" in out
    assert out.count("csv-s0-") == 20
    limited = run_main(
        try_family, ["try_family.py", "--plan", "--limit", "3"], monkeypatch, capsys
    )
    assert re.findall(r"csv-s0-\w+-\d+", limited) == [
        "csv-s0-easy-01",
        "csv-s0-hard-01",
        "csv-s0-medium-01",
    ]


def test_the_reading_shows_each_task_the_score_and_the_power(
    seed_0: list[tasks.Task],
) -> None:
    picked = seed_0[:4]
    records = [
        record(picked[0].task_id, True, submitted="18", n_calls=4),
        record(picked[1].task_id, False, submitted="12"),
        record(picked[2].task_id, True, submitted="36", n_calls=6),
    ]
    lines = try_family.result_lines(records, picked)
    assert lines[0].split() == ["csv-s0-easy-01", "easy", "correct", "'18'"]
    assert lines[1].endswith("'12' · expected '13'")
    assert lines[3].endswith("not run")
    score = try_family.score_lines(records, picked, 0.1)
    assert score[0].startswith("score     2/3 = 66.7% · Wilson 95% 20.8% to 93.9%")
    assert score[0].endswith("calls per solve 5.0")
    assert score[1] == "by level  easy 2/3 · medium 0/0 · hard 0/0"
    assert score[2].startswith("power     to see +10 points from 66.7%")
    assert score[2].endswith("; the 66.7% is itself measured on 3 tasks")


def test_a_partial_run_says_so_before_its_score(
    seed_0: list[tasks.Task], course_home: Path, capsys: Any
) -> None:
    path = try_family.records_path("base", "mini-swe-agent")
    evaluate.append_record(path, record(seed_0[0].task_id, True, n_calls=4))
    records, _ = try_family.report(path, seed_0[:3], "mini-swe-agent", "base", 1.0, 0.1)
    out = capsys.readouterr().out
    assert len(records) == 1
    assert out.index("PARTIAL   1 of 3 tasks recorded") < out.index("score     1/1")


def test_against_reads_two_runs_cell_by_cell(
    seed_0: list[tasks.Task], course_home: Path
) -> None:
    ids = [t.task_id for t in seed_0[:6]]
    earlier = try_family.records_path("base", "mini-swe-agent")
    for task_id, correct in zip(
        ids, [True, True, False, False, False, True], strict=True
    ):
        evaluate.append_record(earlier, record(task_id, correct))
    now = [
        record(t, c)
        for t, c in zip(ids, [True, False, True, True, True, True], strict=True)
    ]
    lines = try_family.against_lines("base", now, "mini-swe-agent")
    assert lines[0].startswith("against   base, on the 6 tasks both ran (0 in one")
    assert lines[0].endswith("before 3, after 5")
    assert lines[1] == f"  gained  3: {ids[2]}, {ids[3]}, {ids[4]}"
    assert lines[2] == f"  lost    1: {ids[1]}"
    assert lines[3] == "  exact p 0.625: within noise"
    with pytest.raises(SystemExit, match="no records"):
        try_family.against_lines("missing", now, "mini-swe-agent")


def test_against_refuses_two_runs_with_no_task_in_common(course_home: Path) -> None:
    earlier = try_family.records_path("seed1", "mini-swe-agent")
    evaluate.append_record(earlier, record("csv-s1-easy-01", True))
    now = [record("csv-s0-easy-01", True)]
    with pytest.raises(SystemExit, match="no task in both runs.*same --seed"):
        try_family.against_lines("seed1", now, "mini-swe-agent")


def test_an_existing_records_file_needs_resume_and_the_same_version(
    course_home: Path,
) -> None:
    path = try_family.records_path("base", "mini-swe-agent")
    evaluate.append_record(path, record("csv-s0-easy-01", True, version="2.4.6"))
    with pytest.raises(SystemExit, match="--resume"):
        try_family.kept_records(path, False, "2.4.6")
    assert len(try_family.kept_records(path, True, "2.4.6")) == 1
    with pytest.raises(SystemExit, match="new --label"):
        try_family.kept_records(path, True, "2.4.7")


def test_the_command_line_refuses_a_bad_label_or_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for flags in (["--label", "../x"], ["--gain", "0"], ["--against", "a b"]):
        monkeypatch.setattr(sys, "argv", ["try_family.py", *flags])
        with pytest.raises(SystemExit):
            try_family.parse_args()


# ------------------------------------------------- end to end, the real harnesses


def installed_commit(env: Path) -> str | None:
    """The OpenEnv commit installed in an environment, read from its metadata.

    Args:
        env: The environment's folder.

    Returns:
        The commit, or None when OpenEnv is not installed there.
    """
    for found in env.glob("lib/python*/site-packages/openenv-*.dist-info"):
        return json.loads((found / "direct_url.json").read_text())["vcs_info"][
            "commit_id"
        ]
    return None


def find_proxy_env() -> Path | None:
    """Find a proxy environment at the pinned commit, this chapter's or an earlier one.

    Returns:
        The environment, or None.
    """
    named = os.environ.get("PROXY_ENV")
    candidates = [Path(named)] if named else []
    candidates += [_HOME / chapter / "proxy-env" for chapter in CHAPTERS_WITH_STACK]
    for env in candidates:
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def find_install(harness: str) -> Path | None:
    """Find a harness at its pin, in this chapter's folder or an earlier one's.

    Args:
        harness: The harness.

    Returns:
        Its install folder, or None.
    """
    pin = install.PINS[harness]
    for chapter in CHAPTERS_WITH_STACK:
        folder = _HOME / chapter / pin.folder
        if folder.is_dir():
            return folder
    return None


def find_tokenizer() -> Path | None:
    """Find LFM2.5's checked tokenizer.json (and its template) in the real home.

    Returns:
        The file, or None.
    """
    for name in lfm_files.TOKENIZER_FOLDERS:
        path = _HOME / lfm_files.MODELS_FOLDER / name / lfm_files.TOKENIZER_FILE
        if (
            path.is_file()
            and (path.parent / "chat_template.jinja").is_file()
            and lfm_files.sha256_of(path) == lfm_files.TOKENIZER_SHA256
        ):
            return path
    return None


REAL_PROXY_ENV = find_proxy_env()
REAL_INSTALLS = {h: find_install(h) for h in (install.MINI, install.QWEN)}


def free_port() -> int:
    """Ask the system for a port nobody listens on.

    Returns:
        The port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def live_engine(cutter: scripted_engine.Cutter | None) -> Iterator[int]:
    """Serve the scripted engine over real HTTP, on a free port.

    Args:
        cutter: Its tokenizer; the scripted one when None.

    Yields:
        Its port.
    """
    app, _ = scripted_engine.create_app(cutter)
    sock = engine.claim_port(0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10)


def run_start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``proxy`` or ``stop``.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "PROXY_PORT": str(proxy_port),
        "COURSE_HOME": str(home),
        "PROXY_ENV": str(REAL_PROXY_ENV),
    }
    return subprocess.run(
        ["bash", str(START_SH), command],
        env=env,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
        check=False,
    )


@pytest.fixture(scope="module")
def proxy_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The real proxy, started by start.sh, in front of a scripted engine.

    Both listen on free ports, never on a learner's own stack's.

    Args:
        tmp_path_factory: pytest's temporary folders.

    Yields:
        The proxy's root.
    """
    if REAL_PROXY_ENV is None:
        pytest.skip("needs the capture proxy (./start.sh install)")
    home = tmp_path_factory.mktemp("stack")
    tokenizer = find_tokenizer()
    cutter = scripted_engine.LfmCutter(tokenizer) if tokenizer else None
    with live_engine(cutter) as engine_port:
        proxy_port = free_port()
        started = run_start_sh("proxy", engine_port, proxy_port, home)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            yield f"http://127.0.0.1:{proxy_port}"
        finally:
            run_start_sh("stop", engine_port, proxy_port, home)


def link_install(course_home: Path, harness: str) -> None:
    """Point this test's COURSE_HOME at a real install of one harness.

    Args:
        course_home: The test's temporary course home.
        harness: The harness.
    """
    found = REAL_INSTALLS[harness]
    if found is None:
        pytest.skip(f"needs {harness} installed (uv run install.py {harness})")
    chapter = course_home / "29-make-it-yours"
    chapter.mkdir(parents=True, exist_ok=True)
    (chapter / install.PINS[harness].folder).symlink_to(found)


def run_family(
    proxy_url: str, monkeypatch: pytest.MonkeyPatch, capsys: Any, *flags: str
) -> tuple[str, list[evaluate.Record]]:
    """Run try_family.py, as from the command line, and read its records.

    Args:
        proxy_url: The proxy.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.
        *flags: Its flags.

    Returns:
        What it printed, and the records.
    """
    argv = ["try_family.py", "--proxy-url", proxy_url, *flags]
    out = run_main(try_family, argv, monkeypatch, capsys)
    found = re.search(r"family/([\w.-]+)/([\w-]+)\.jsonl", out)
    assert found, out
    return out, evaluate.read_records(try_family.records_path(found[1], found[2]))


def test_the_family_runs_in_mini_swe_agent_through_the_proxy(
    proxy_url: str, course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    link_install(course_home, install.MINI)
    out, records = run_family(proxy_url, monkeypatch, capsys, "--limit", "3")
    assert "script: solve" in out and "version   2.4.6, asked inside the sandbox" in out
    assert [r.task_id for r in records] == [
        "csv-s0-easy-01",
        "csv-s0-hard-01",
        "csv-s0-medium-01",
    ]
    for item in records:
        assert item.correct and item.n_calls == 4 and item.loopback_only
        assert item.stop == "finish command" and item.engine == "scripted-solve"
        assert item.first_prompt and item.domain == "sales"
    assert "score     3/3 = 100.0%" in out
    assert "power     no room for +10 points from 100.0%" in out
    assert out.rstrip().endswith("lsof saw only this machine's addresses")
    again, _ = run_family(proxy_url, monkeypatch, capsys, "--limit", "3", "--resume")
    assert again.count("already recorded") == 3
    with pytest.raises(SystemExit, match="this run's own label"):
        run_family(proxy_url, monkeypatch, capsys, "--against", "scripted-solve")


def test_qwen_code_solves_the_family_through_the_proxy(
    proxy_url: str, course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    link_install(course_home, install.QWEN)
    flags = ("--harness", "qwen-code", "--limit", "2")
    out, records = run_family(proxy_url, monkeypatch, capsys, *flags)
    assert "version   0.24.7, asked inside the sandbox" in out
    for item in records:
        assert item.correct and item.n_calls == 4 and item.loopback_only
        assert item.stop == "no tool call" and item.version == "0.24.7"
        # Its prompt fits the 16,384-token window, with room to work.
        assert item.first_prompt and item.first_prompt < wire.MAX_PROMPT_TOKENS // 2
    cell = evaluate.cell_folder("scripted-solve", "qwen-code", records[0])
    kept = json.loads((cell / run_one.RUN_FILE).read_text())
    assert kept["facts"]["dialect"].startswith("openai_chat")
    assert kept["facts"]["log_says"] == "result success · 4 turns"
    assert (cell / "qwen-stream.jsonl").is_file()


def test_the_qwen_probe_measures_what_it_does_around_the_model(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    link_install(course_home, install.QWEN)
    out = run_main(qwen_probe, ["qwen_probe.py"], monkeypatch, capsys)
    sections = out.split("━━ ")
    assert "rum.aliyuncs.com:443 (refused by the catcher)" in sections[1]
    assert "outside connections asked for: 0" in sections[2]
    assert "tools offered  3: edit, read_file, run_shell_command" in sections[3]
    assert "max_tokens     32000 asked" in sections[3]
    assert "requests       3" in sections[3] and "requests       4" in sections[4]
    assert (
        "tools offered  14:" in sections[4] and "(after the last reply)" in sections[4]
    )
    assert 'refused: This "run_shell_command" invocation was denied' in sections[5]
    assert "echo 42 > answer.txt   ran" in sections[5]
