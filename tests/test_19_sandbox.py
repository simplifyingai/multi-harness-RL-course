"""Chapter 19: the box, the runner, the safe grader and Pi's write tool.

No test here reads a real secret. The live tests (macOS only) run small
commands under ``sandbox-exec`` against folders inside the test's own
COURSE_HOME and against the decoys, never against your real home's files.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from chapters import use_chapter

CHAPTER = use_chapter("19-sandbox")

import attacks  # noqa: E402
import decoys  # noqa: E402
import grade_safe  # noqa: E402
import grader  # noqa: E402
import process  # noqa: E402
import runner  # noqa: E402
import sandbox  # noqa: E402
import tasks  # noqa: E402
import write_tool  # noqa: E402

LIVE = sys.platform == "darwin" and sandbox.SANDBOX_EXEC.is_file()
needs_sandbox = pytest.mark.skipif(not LIVE, reason="needs macOS sandbox-exec")
COMMENT_RE = re.compile(r";;.*$", re.MULTILINE)


def profile_rules(path: Path) -> list[str]:
    """Return a profile's non-empty lines with the comments removed.

    Args:
        path: A ``.sb`` file.

    Returns:
        Its rule lines, stripped.
    """
    text = COMMENT_RE.sub("", path.read_text())
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- tasks


def test_task_is_chapter_ones_murcia_question() -> None:
    task = tasks.load_task()
    assert task.task_id == "clinic-easy-001"
    assert task.answer == "30"
    assert task.database.is_file()


def test_show_hides_the_home_folder(course_home: Path) -> None:
    home = str(tasks.real_home())
    text = f"{tasks.chapter_home()}/x {tasks.REPO_ROOT}/y {home}/z"
    shown = tasks.show(text)
    assert shown == "$CH/x $REPO/y /Users/you/z"
    assert all(home not in line for line in tasks.legend())


# ---------------------------------------------------------------- profile


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        (".env", True),
        (".env.local", True),
        (".envrc", True),
        ("prod.env", True),
        ("deep/folder/.env", True),
        (".venv", False),
        (".environment", False),
        ("env.py", False),
    ],
)
def test_env_regex_matches_env_files_only(name: str, secret: bool) -> None:
    home = Path("/Users/some.one")
    pattern = f"^{sandbox.regex_escape(str(home))}{sandbox.ENV_FILE_PATTERN}"
    assert bool(re.search(pattern, f"{home}/{name}")) is secret


def test_regex_escape_escapes_the_dot_in_a_home_folder() -> None:
    assert sandbox.regex_escape("/Users/a.b") == "/Users/a\\.b"


def test_every_credential_path_is_in_the_profile() -> None:
    text = sandbox.PROFILES[sandbox.PROJECT_PROFILE].read_text()
    for relative in sandbox.CREDENTIAL_PATHS:
        if relative.startswith(".claude"):
            continue  # matched by the two Claude regexes instead
        assert f'"/{relative}"' in text, relative


def test_v1_is_the_project_profile_plus_one_rule() -> None:
    project = profile_rules(sandbox.PROFILES[sandbox.PROJECT_PROFILE])
    first = profile_rules(sandbox.PROFILES[sandbox.V1_PROFILE])
    added = [line for line in first if line not in project]
    removed = [line for line in project if line not in first]
    assert added == ['(subpath (param "TASKS_DIR"))', '(subpath (param "RUNS_DIR")))']
    assert removed == ['(subpath (param "TASKS_DIR")))']


def test_rollout_reads_are_reallowed_after_the_denies() -> None:
    text = sandbox.PROFILES[sandbox.PROJECT_PROFILE].read_text()
    reallow = text.rindex('(subpath (param "WORKDIR"))')
    assert text.index('(subpath (param "TASKS_DIR"))') < reallow
    assert text.index('(regex (param "ENV_FILE_REGEX"))') < reallow


def test_writable_root_refuses_broad_and_secret_folders(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".config").mkdir()
    for path in (Path("/"), home, home / ".ssh", home / ".config"):
        with pytest.raises(sandbox.UnsafeSandboxPathError):
            sandbox.check_writable_root(path, home)
    work = tmp_path / "work"
    work.mkdir()
    assert sandbox.check_writable_root(work, home) == work.resolve()


def test_hidden_root_refuses_to_overlap_a_writable_folder(tmp_path: Path) -> None:
    work = (tmp_path / "rollouts" / "r1").resolve()
    (work / "gold").mkdir(parents=True)
    for hidden in (work.parent, work / "gold"):
        with pytest.raises(sandbox.UnsafeSandboxPathError):
            sandbox.check_hidden_root(hidden, [work])
    assert sandbox.check_hidden_root(tasks.DATA_DIR, [work]) == tasks.DATA_DIR


def test_params_anchor_every_regex_on_the_escaped_home(tmp_path: Path) -> None:
    work, state = tmp_path / "w", tmp_path / "s"
    work.mkdir()
    state.mkdir()
    home = tmp_path / "decoy.home"
    home.mkdir()
    params = sandbox.sandbox_params(
        workdir=work, state_dir=state, runs_dir=tmp_path, home=home
    )
    escaped = sandbox.regex_escape(str(home.resolve()))
    for name in ("ENV_FILE_REGEX", "CLAUDE_JSON_REGEX", "CLAUDE_CREDS_REGEX"):
        assert params[name].startswith(f"^{escaped}/"), name
    assert params["TASKS_DIR"] == str(tasks.DATA_DIR)
    assert "SYSTEM_TMP" not in params


def test_the_box_refuses_to_run_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.sandbox_command(
            ["true"], profile=sandbox.PROFILES[sandbox.PROJECT_PROFILE], params={}
        )


def test_the_box_refuses_a_missing_profile(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("checked after the platform")
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.check_available(tmp_path / "missing.sb")


# ---------------------------------------------------------------- process


def test_parse_ps_line_reads_the_program_name() -> None:
    row = process.parse_ps_line("  812   811   812 Mon Oct  6 05:47:12 2026 /bin/sleep")
    assert row is not None
    assert (row.pid, row.ppid, row.pgid, row.name) == (812, 811, 812, "sleep")
    assert process.parse_ps_line("garbage") is None


def test_descendants_walks_the_whole_tree() -> None:
    rows = [
        process.ProcessInfo(2, 1, 2, 0.0, "bash"),
        process.ProcessInfo(3, 2, 2, 0.0, "sleep"),
        process.ProcessInfo(4, 3, 4, 0.0, "perl"),
        process.ProcessInfo(9, 1, 9, 0.0, "other"),
    ]
    assert {row.pid for row in process.descendants(2, rows)} == {3, 4}


def test_timeout_reaps_the_whole_tree(tmp_path: Path) -> None:
    outcome = process.run_logged(
        ["/bin/bash", "-c", "sleep 60 & sleep 60 & wait"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        log_path=tmp_path / "log.txt",
        timeout_s=3,
    )
    assert outcome.timed_out
    assert outcome.exit_code is None
    assert len(outcome.seen) == 3
    assert not outcome.survivors
    assert not outcome.ps_failed
    assert not process.still_alive(outcome.seen, wait=False)


def fake_table(
    monkeypatch: pytest.MonkeyPatch, rows: list[process.ProcessInfo], marked: set[int]
) -> list[int]:
    """Replace ps and kill with fakes, and record which pids get killed.

    Args:
        monkeypatch: pytest's patcher.
        rows: The process table ps will report.
        marked: Pids whose environment carries the marker.

    Returns:
        The list the fake kill appends to.
    """
    killed: list[int] = []
    monkeypatch.setattr(process, "process_table", lambda: rows)
    monkeypatch.setattr(process, "processes_with_marker", lambda marker: marked)
    monkeypatch.setattr(process, "_kill_group", lambda pgid: None)
    monkeypatch.setattr(process, "_kill", lambda pid: killed.append(pid) or True)
    return killed


def test_sweeps_never_signal_a_process_older_than_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_start = 1_000_000.0
    old = run_start - 60
    rows = [
        process.ProcessInfo(10, 1, 10, run_start, "bash"),  # the root
        process.ProcessInfo(11, 10, 10, run_start, "sleep"),  # its child
        process.ProcessInfo(12, 1, 10, old, "old"),  # recycled into the group
        process.ProcessInfo(13, 1, 13, run_start + 1, "node"),  # marked orphan
        process.ProcessInfo(14, 1, 14, old, "older"),  # marked, but too old
    ]
    killed = fake_table(monkeypatch, rows, marked={13, 14})
    signalled = process.kill_tree(
        10, {10}, run_start, root_reaped=True, marker="MHRL_RUN_ID=x"
    )
    assert signalled == {10, 11, 13}
    assert set(killed) == {10, 11, 13}


def test_marked_orphans_are_reported_as_survivors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [process.ProcessInfo(13, 1, 13, 100.0, "node")]
    fake_table(monkeypatch, rows, marked={13})
    assert set(process.marked("MHRL_RUN_ID=x", cutoff=50.0)) == {13}
    assert process.marked("MHRL_RUN_ID=x", cutoff=150.0) == {}
    assert process.marked(None, cutoff=0.0) == {}


def test_group_only_misses_a_child_that_left_the_group(tmp_path: Path) -> None:
    if not Path("/usr/bin/perl").is_file():
        pytest.skip("needs perl")
    script = 'sleep 60 & perl -e "setpgrp(0, 0); sleep 60" & wait'
    outcome = process.run_logged(
        ["/bin/bash", "-c", script],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        log_path=tmp_path / "log.txt",
        timeout_s=2.5,
        group_only=True,
    )
    try:
        names = {outcome.seen[pid].name for pid in outcome.survivors}
        assert names == {"perl"}
    finally:
        process.kill_survivors({pid: outcome.seen[pid] for pid in outcome.survivors})
    assert not process.still_alive(outcome.seen)


# ---------------------------------------------------------------- write tool


def test_write_tool_writes_a_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert write_tool.write("answer.txt", "30") == "Successfully wrote to answer.txt"
    assert (tmp_path / "answer.txt").read_text() == "30"


def test_write_tool_realpath_is_the_c_library_one() -> None:
    assert write_tool.realpath_c("/tmp") == "/private/tmp"
    with pytest.raises(OSError) as caught:
        write_tool.realpath_c("/no/such/folder/answer.txt")
    assert caught.value.errno == errno.ENOENT
    assert write_tool.queue_key("/no/such/answer.txt") == "/no/such/answer.txt"


def test_write_tool_fails_like_pi_on_any_other_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(path: str) -> str:
        raise OSError(errno.EPERM, os.strerror(errno.EPERM), path)

    monkeypatch.setattr(write_tool, "realpath_c", denied)
    with pytest.raises(write_tool.ToolError) as caught:
        write_tool.write("/w/answer.txt", "30")
    assert caught.value.node_message() == (
        "EPERM: operation not permitted, realpath '/w/answer.txt'"
    )


# ---------------------------------------------------------------- grading


def test_grade_safe_accepts_a_plain_answer(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("30\n")
    assert grade_safe.grade_safe(tasks.load_task(), tmp_path).correct


def test_grade_safe_refuses_a_link_the_naive_grader_follows(tmp_path: Path) -> None:
    # The link points at a file that holds the RIGHT answer: following it
    # would hand out a reward for a file the agent could not even read.
    target = tmp_path / "elsewhere.txt"
    target.write_text("30\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "answer.txt").symlink_to(target)
    task = tasks.load_task()
    assert grade_safe.grade_naive(task, work).correct
    grade = grade_safe.grade_safe(task, work)
    assert not grade.correct
    assert grade.submitted is None
    assert grade.reason == grader.REASON_NOT_REGULAR


def test_grade_safe_refuses_a_hard_link(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.txt"
    target.write_text("30\n")
    work = tmp_path / "work"
    work.mkdir()
    os.link(target, work / "answer.txt")
    task = tasks.load_task()
    assert grade_safe.grade_naive(task, work).correct
    grade = grade_safe.grade_safe(task, work)
    assert grade.submitted is None
    assert grade.reason == grade_safe.REASON_HARD_LINK


def test_grade_safe_says_when_it_read_only_part_of_a_file(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("30\n" + "x" * grader.MAX_ANSWER_BYTES)
    grade = grade_safe.grade_safe(tasks.load_task(), tmp_path)
    assert grade.correct
    assert f"only the first {grader.MAX_ANSWER_BYTES} of" in grade.reason


def test_grade_safe_does_not_wait_on_a_named_pipe(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "answer.txt")
    started = time.monotonic()
    grade = grade_safe.grade_safe(tasks.load_task(), tmp_path)
    assert time.monotonic() - started < 1
    assert grade.reason == grader.REASON_NOT_REGULAR


def test_grade_safe_refuses_a_folder_and_reports_a_missing_file(
    tmp_path: Path,
) -> None:
    task = tasks.load_task()
    assert grade_safe.grade_safe(task, tmp_path).reason == grader.REASON_NO_ANSWER
    (tmp_path / "answer.txt").mkdir()
    assert grade_safe.grade_safe(task, tmp_path).reason == grader.REASON_NOT_REGULAR


def test_check_answer_is_chapter_ones() -> None:
    task = tasks.load_task()
    assert grader.check_answer(task, "  30.\n").correct
    assert not grader.check_answer(task, "31").correct


# ---------------------------------------------------------------- rollouts


def test_decoys_hold_no_secret(course_home: Path) -> None:
    found = decoys.make_decoys()
    files = found.files()
    assert files["gold answers"] == found.gold_file
    for path in files.values():
        assert path.read_text() == decoys.DECOY_TEXT


def test_a_rollout_gets_a_clean_environment(course_home: Path) -> None:
    rollout = runner.new_rollout("t")
    env = rollout.environment()
    assert sorted(env) == ["HOME", "LANG", "MHRL_RUN_ID", "PATH", "TMPDIR"]
    assert env["HOME"] == str(rollout.fake_home)
    assert (rollout.workdir / tasks.DATABASE_FILE).is_file()
    assert rollout.fake_home.is_dir() and rollout.tmp.is_dir()


def test_a_rollout_name_must_be_a_plain_folder_name(course_home: Path) -> None:
    for name in ("..", "a/b", "../../x", "."):
        with pytest.raises(ValueError):
            runner.new_rollout(name)


def test_exit_status_is_a_shells() -> None:
    def outcome(code: int | None, timed_out: bool = False) -> process.ProcessOutcome:
        return process.ProcessOutcome(code, 0.0, timed_out, root_pid=1)

    assert runner.exit_status(outcome(None, timed_out=True)) == 124
    assert runner.exit_status(outcome(None)) == 124
    assert runner.exit_status(outcome(-9)) == 137
    assert runner.exit_status(outcome(-15)) == 143
    assert runner.exit_status(outcome(3)) == 3


@pytest.mark.parametrize(
    "text", ["SELECT COUNT(*) FROM t WHERE city = 'Murcia'", 'a$HOME `id` "q" \\x']
)
def test_double_quote_survives_bash_unchanged(text: str) -> None:
    quoted = attacks.double_quote(text)
    done = subprocess.run(
        ["/bin/bash", "-c", f"printf %s {quoted}"], capture_output=True, text=True
    )
    assert done.stdout == text


def test_without_printed_proof_the_outside_check_decides() -> None:
    silent = attacks.Attempt("x", "test -e x", attacks.Outcome.DENIED, evidence="")
    run = process.ProcessOutcome(1, 0.0, False, root_pid=1)
    present = attacks.Outside("present", True)
    assert attacks.classify(silent, run, "", present)[0] == attacks.Outcome.DENIED
    assert attacks.classify(silent, run, "", None)[0] == "inconclusive"


def test_a_failed_control_is_unexpected_not_inconclusive() -> None:
    work = attacks.Attempt("query", "sqlite3 x", attacks.Outcome.ALLOWED)
    probe = attacks.Attempt("key", "cat x", attacks.Outcome.DENIED)
    failed = attacks.Outcome.INCONCLUSIVE
    broken = attacks.Result(work, failed, attacks.Outcome.ALLOWED, "-", "why")
    missing = attacks.Result(probe, failed, attacks.Outcome.DENIED, "absent", "why")
    assert broken.unexpected and broken.check == "UNEXPECTED"
    assert not missing.unexpected and missing.check.startswith("inconclusive")


def test_attempt_expectations_differ_only_where_v1_does() -> None:
    allowed, denied = attacks.Outcome.ALLOWED, attacks.Outcome.DENIED
    attempt = attacks.Attempt("x", "true", allowed, expect_v1=denied)
    assert attempt.expected(sandbox.PROJECT_PROFILE) == allowed
    assert attempt.expected(sandbox.V1_PROFILE) == denied


@pytest.mark.parametrize(
    ("code", "timed_out", "output", "exists", "outcome"),
    [
        (0, False, "", True, attacks.Outcome.ALLOWED),
        (1, False, "cat: x: Operation not permitted", True, attacks.Outcome.DENIED),
        (1, False, "EPERM: OPERATION NOT PERMITTED", True, attacks.Outcome.DENIED),
        (1, False, "cat: x: No such file or directory", True, "inconclusive"),
        (1, False, "cat: x: Operation not permitted", False, "inconclusive"),
        (127, False, "bash: cat: command not found", True, "inconclusive"),
        (None, True, "", True, "inconclusive"),
    ],
)
def test_a_denial_needs_proof(
    code: int | None, timed_out: bool, output: str, exists: bool, outcome: str
) -> None:
    attempt = attacks.Attempt("x", "cat x", attacks.Outcome.DENIED)
    run = process.ProcessOutcome(code, 0.0, timed_out, root_pid=1)
    outside = attacks.Outside("present" if exists else "absent", exists)
    assert attacks.classify(attempt, run, output, outside)[0] == outcome


# ---------------------------------------------------------------- live box


def run(rollout: runner.Rollout, command: str) -> tuple[int | None, str]:
    """Run one bash command in a rollout's box.

    Args:
        rollout: The rollout.
        command: The command.

    Returns:
        Its exit status and output.
    """
    outcome, output = runner.run_in_box(
        rollout, ["/bin/bash", "-c", command], label="t", timeout_s=30
    )
    return outcome.exit_code, output


@needs_sandbox
def test_box_allows_the_task_folder_and_denies_outside(
    course_home: Path, tmp_path: Path
) -> None:
    rollout = runner.new_rollout("live")
    outside = tmp_path / "outside.txt"
    assert run(rollout, "echo 30 > answer.txt")[0] == 0
    code, output = run(rollout, f"echo x > {outside}")
    assert code != 0 and "Operation not permitted" in output
    assert not outside.exists()


@needs_sandbox
def test_box_hides_the_decoy_secrets_and_gold_answers(course_home: Path) -> None:
    found = decoys.make_decoys()
    rollout = runner.new_rollout("live", home=found.home, tasks_dir=found.tasks_dir)
    for path in found.files().values():
        code, output = run(rollout, attacks.read_probe(path))
        assert code != 0, path
        assert attacks.EPERM_TEXT in output.lower()
        assert decoys.DECOY_TEXT.strip() not in output


@needs_sandbox
def test_v1_breaks_pis_write_tool_and_the_project_profile_does_not(
    course_home: Path,
) -> None:
    tool = f"{sys.executable} {CHAPTER / 'write_tool.py'} answer.txt 30"
    cases = ((sandbox.PROJECT_PROFILE, True), (sandbox.V1_PROFILE, False))
    for profile, works in cases:
        rollout = runner.new_rollout(f"live-{profile}", profile=profile)
        code, output = run(rollout, f"pwd >/dev/null && {tool}")
        assert (code == 0) is works, output
        if not works:
            assert "EPERM: operation not permitted, realpath" in output


@needs_sandbox
def test_a_readable_decoy_stops_before_any_real_path(
    course_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken.sb"
    broken.write_text("(version 1)\n(allow default)\n")
    monkeypatch.setitem(sandbox.PROFILES, sandbox.PROJECT_PROFILE, broken)
    assert attacks.run_all(sandbox.PROJECT_PROFILE, []) == attacks.EXIT_DECOY_FAILED
    rollouts = tasks.chapter_home() / tasks.ROLLOUTS_DIR
    assert sorted(path.name for path in rollouts.iterdir()) == [attacks.DECOY_ROLLOUT]
    shutil.rmtree(rollouts)
