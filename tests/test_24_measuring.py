"""Chapter 24 without a model: a fair baseline, and how much of a score is noise.

The unit tests check the project's copied evaluation (``balanced``,
``summarize``, ``format_table``), the records file, the statistics, and
``noise.py``'s whole output on the project's bundled records (pinned in
``tests/data/24_noise.txt``) and ``baseline_eval.py --plan`` (pinned in
``tests/data/24_plan.txt``). The end-to-end tests run ``baseline_eval.py`` as a
learner does, on a two-task subset: the three pinned harnesses inside the
sandbox, through OpenEnv's real capture proxy (started by ``start.sh``), in
front of chapter 23's scripted engine served over real HTTP. They are skipped
when the proxy or the harnesses are not on disk (``./start.sh install`` and
``uv run install.py``, in this chapter or chapter 23).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("24-measuring")

import baseline_eval  # noqa: E402
import engine  # noqa: E402
import evaluate  # noqa: E402
import fetch_model  # noqa: E402
import install  # noqa: E402
import lfm_files  # noqa: E402
import noise  # noqa: E402
import run_one  # noqa: E402
import scripted_engine  # noqa: E402
import stats  # noqa: E402
import tasks  # noqa: E402
from loopback import NetworkReport  # noqa: E402

DATA = Path(__file__).parent / "data"
NOISE_GOLDEN = DATA / "24_noise.txt"
PLAN_GOLDEN = DATA / "24_plan.txt"
HOME_MARK = "$COURSE_HOME"
START_SH = CHAPTER / "start.sh"
CHAPTER_23 = ROOT / "23-three-harnesses"
PROJECT = CHAPTER / "data" / "project_records"
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy and the harnesses are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
CHAPTERS_WITH_STACK = ("24-measuring", "23-three-harnesses")
START_TIMEOUT_S = 300
EVERY = tasks.load_tasks()
# The modules this chapter copies from chapter 23 unchanged but for its name.
COPIED = (
    "capture_client.py",
    "grader.py",
    "install.py",
    "lfm.py",
    "lfm_files.py",
    "lfm_template.py",
    "lfm_text.py",
    "loopback.py",
    "mini_swe_agent.yaml",
    "mini-swe-agent-requirements.txt",
    "policy.py",
    "proxy-requirements.txt",
    "run_one.py",
    "sandbox.py",
    "sandbox.sb",
    "scripted_engine.py",
    "scripted_replies.py",
    "tasks.py",
    "toolcalls.py",
    "wire.py",
)


def as_text(out: str, home: Path) -> str:
    """Write the temporary course home as a placeholder, for golden files.

    Args:
        out: What a script printed.
        home: The test's COURSE_HOME.

    Returns:
        The output with the folder written ``$COURSE_HOME``.
    """
    return out.replace(str(home), HOME_MARK)


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


def record(task_id: str, harness: str, correct: bool, **extra: Any) -> evaluate.Record:
    """A record for one cell.

    Args:
        task_id: The task.
        harness: The harness.
        correct: The verdict.
        **extra: Other fields.

    Returns:
        The record.
    """
    return evaluate.Record(task_id=task_id, harness=harness, correct=correct, **extra)


# ------------------------------------------------- the copied evaluation


def test_balanced_takes_four_from_each_bucket_in_the_projects_order() -> None:
    picked = evaluate.balanced(EVERY, 24)
    counts = Counter((t.domain, t.difficulty) for t in picked)
    assert len(counts) == 6 and set(counts.values()) == {4}
    assert [t.task_id for t in picked[:7]] == [
        "airline-easy-001",
        "airline-hard-001",
        "airline-medium-001",
        "clinic-easy-001",
        "clinic-hard-001",
        "clinic-medium-001",
        "airline-easy-002",
    ]


def test_balanced_picks_the_cells_the_project_recorded() -> None:
    recorded = evaluate.read_records(PROJECT / "eval_base.jsonl")
    mini = [r.task_id for r in recorded if r.harness == "mini-swe-agent"]
    assert mini == [t.task_id for t in evaluate.balanced(EVERY, 24)]


def test_balanced_keeps_everything_when_the_limit_is_not_smaller() -> None:
    assert evaluate.balanced(EVERY, None) == EVERY
    assert evaluate.balanced(EVERY, len(EVERY)) == EVERY
    assert [t.task_id for t in evaluate.balanced(EVERY, 4)] == [
        "airline-easy-001",
        "airline-hard-001",
        "airline-medium-001",
        "clinic-easy-001",
    ]


def test_a_plain_prefix_depends_on_the_files_order() -> None:
    by_file = Counter(t.domain for t in EVERY[:24])
    by_name = Counter(t.domain for t in sorted(EVERY, key=lambda t: t.task_id)[:24])
    assert by_file == {"clinic": 12, "airline": 12}  # the file is interleaved
    assert by_name == {"airline": 24}  # sorted by name: one database only


def test_summarize_is_the_projects_on_its_own_records() -> None:
    base = noise.project_checkpoint("base")
    harnesses = list(install.HARNESSES)
    summary = evaluate.summarize(list(base.records), harnesses)
    solved = {h: summary["per_harness"][h]["solved"] for h in harnesses}
    assert solved == {"mini-swe-agent": 24, "claude-code": 15, "pi": 4}
    assert summary["cells"] == 72
    assert round(summary["average_accuracy"], 2) == 59.72
    calls = summary["per_harness"]["mini-swe-agent"]["calls_when_solved"]
    assert round(calls, 3) == 7.833  # runs/main/summary_base.json
    table = evaluate.format_table({"base": summary}, harnesses)
    assert table.splitlines()[1].split() == ["base", "100%", "62%", "17%", "60%"]


def test_the_table_widens_for_a_long_label() -> None:
    summary = evaluate.summarize([record("t", "pi", True)], ["pi"])
    lines = evaluate.format_table({"scripted-mistakes": summary}, ["pi"]).splitlines()
    assert lines[1].startswith("scripted-mistakes  ")
    assert len(lines[0]) == len(lines[1])


def test_records_round_trip_and_read_the_projects_own_format(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    first_record = record("a", "pi", True, n_calls=3, stop="no tool call")
    evaluate.append_record(path, first_record)
    raw = {"task_id": "b", "harness": "pi", "reward": 1.0, "sequences": []}
    with path.open("a") as handle:
        handle.write(json.dumps(raw) + "\n")
    first, second = evaluate.read_records(path)
    assert (first.n_calls, first.stop, first.correct) == (3, "no tool call", True)
    assert (second.task_id, second.correct, second.loopback_only) == ("b", False, None)
    assert evaluate.read_records(tmp_path / "missing.jsonl") == []


def test_a_broken_record_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text('{"task_id": "a", "harness": "pi"}\n{"task_id": "b"}\n')
    with pytest.raises(ValueError, match=r"eval.jsonl:2: a record needs"):
        evaluate.read_records(path)


def facts(**extra: Any) -> run_one.Facts:
    """Facts about a run, with no calls unless given.

    Args:
        **extra: Fields to set.

    Returns:
        The facts.
    """
    base = {
        "dialect": "x",
        "agent_calls": 0,
        "side_calls": 0,
        "first_prompt": None,
        "budget_stops": 0,
        "refused_tokens": None,
    }
    return run_one.Facts(**{**base, **extra})


def test_each_stop_rule_gets_a_short_label() -> None:
    done = run_one.Process(0, 1.0, NetworkReport())
    killed = run_one.Process(None, 300.0, NetworkReport())
    cases = [
        (("mini-swe-agent", facts(), done, "Submitted"), "finish command"),
        (("mini-swe-agent", facts(), done, "RepeatedFormatError"), "3 bad replies"),
        (("mini-swe-agent", facts(), done, "LimitsExceeded"), "step limit"),
        (("claude-code", facts(budget_stops=1), done, ""), "8-call budget"),
        (("pi", facts(refused_tokens=30_895), done, ""), "window full"),
        (("pi", facts(), done, ""), "no tool call"),
        (("pi", facts(), killed, ""), "time limit"),
    ]
    for args, label in cases:
        assert evaluate.stop_label(run_one.stop_rule(*args)) == label


class StoppedError(Exception):
    """A run stopped part way."""


def test_each_record_is_on_disk_as_soon_as_its_rollout_ends(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = course_home / "eval.jsonl"
    picked = evaluate.balanced(EVERY, 3)
    made = []

    def fake(task: tasks.Task, harness: str, setup: Any) -> evaluate.Record:
        if len(made) == 2:
            raise StoppedError  # the learner stops the run, or the machine dies
        made.append(task.task_id)
        return record(task.task_id, harness, True)

    monkeypatch.setattr(evaluate, "run_rollout", fake)
    with pytest.raises(StoppedError):
        evaluate.evaluate(picked, ["pi"], None, out, say=lambda line: None)
    assert [r.task_id for r in evaluate.read_records(out)] == made
    lines: list[str] = []
    done = [("pi", task_id) for task_id in made]
    evaluate.evaluate(picked[:2], ["pi"], None, out, done, lines.append)
    assert len(lines) == 2 and all("already recorded" in line for line in lines)


# ------------------------------------------------- the statistics


def test_the_plans_numbers() -> None:
    assert round(100 / 24, 1) == 4.2
    assert round(stats.standard_error(18, 24), 1) == 8.8  # near 75%
    assert stats.exact_p(0, 6) == pytest.approx(0.03125)
    assert stats.exact_p(7, 5) == pytest.approx(0.7744, abs=1e-4)
    assert stats.exact_p(0, 0) == 1.0


def test_wilson_stays_inside_0_and_100_and_never_shrinks_to_a_point() -> None:
    low, high = stats.wilson_interval(24, 24)
    assert round(low, 1) == 86.2 and high == 100.0
    assert [round(x, 1) for x in stats.wilson_interval(15, 24)] == [42.7, 78.8]
    assert [round(x, 1) for x in stats.wilson_interval(4, 24)] == [6.7, 35.9]
    assert stats.wilson_interval(0, 24)[0] == 0.0
    assert stats.wilson_interval(0, 0) == (0.0, 100.0)
    assert stats.standard_error(24, 24) == 0.0


def test_two_identical_models_exactly_and_simulated() -> None:
    exact = stats.gap_chances(24, 0.75)
    assert sum(exact) == pytest.approx(1.0)
    assert sum(exact[2:]) == pytest.approx(0.61596, abs=1e-5)
    first = stats.simulate_gaps(24, 0.75, 2_000, seed=0)
    assert first == stats.simulate_gaps(24, 0.75, 2_000, seed=0)  # seeded
    assert first != stats.simulate_gaps(24, 0.75, 2_000, seed=1)
    assert sum(first) == 2_000
    assert abs(sum(first[2:]) / 2_000 - sum(exact[2:])) < 0.04


def test_compare_needs_the_same_cells() -> None:
    result = stats.compare({"a": True, "b": False}, {"a": False, "b": True})
    assert (result.gained, result.lost, result.flipped) == (("b",), ("a",), 2)
    with pytest.raises(ValueError, match="different cells"):
        stats.compare({"a": True}, {"b": True})


# ------------------------------------------------- noise.py


def test_noise_prints_every_reading_exactly(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(noise, ["noise.py"], monkeypatch, capsys)
    assert as_text(out, course_home) == NOISE_GOLDEN.read_text(encoding="utf-8")


def test_noise_keeps_the_articles_numbers_in_their_own_box(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(noise, ["noise.py"], monkeypatch, capsys)
    box = out[out.index("┌") : out.index("┘")]
    assert "by 29 points" in box and "23%" in box and "52%" in box
    assert "62" not in box and "33%" not in box  # no bare 62 to mix with ours
    assert "29 points" not in out.replace(box, "")


def test_the_project_checkpoints_take_pi_from_its_rerun() -> None:
    for label, pi in (("base", 4), ("sft", 9), ("rl", 9)):
        checkpoint = noise.project_checkpoint(label)
        assert checkpoint.solved("pi") == (pi, 24)
        assert len(checkpoint.records) == 72


def test_churn_is_many_flips_that_cancel_out() -> None:
    base, sft = noise.project_checkpoint("base"), noise.project_checkpoint("sft")
    claude, mini = (
        stats.compare(base.verdicts(h), sft.verdicts(h))
        for h in ("claude-code", "mini-swe-agent")
    )
    assert (claude.flipped, noise.cancelled(claude)) == (12, 10)
    assert noise.churned(claude, 24) and not noise.churned(mini, 24)


def test_your_records_are_read_and_paired_on_the_cells_both_ran(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    mine = course_home / "24-measuring" / "baseline" / "eval_base.jsonl"
    for harness in install.HARNESSES:
        for task in evaluate.balanced(EVERY, 2):
            evaluate.append_record(mine, record(task.task_id, harness, True))
    out = run_main(noise, ["noise.py"], monkeypatch, capsys)
    assert "  yours     your base (6 cells)" in out
    assert re.search(r"base -> your base\s+pi\s+2\s+0\s+2\s+2\s+0", out)
    other = course_home / "eval_scripted-solve.jsonl"
    evaluate.append_record(other, record("airline-easy-001", "pi", False))
    out = run_main(noise, ["noise.py", "--records", str(other)], monkeypatch, capsys)
    assert "your scripted-solve (1 cells)" in out and "your base" not in out


def test_two_records_files_with_one_name_get_two_labels(tmp_path: Path) -> None:
    for folder in ("one", "two"):
        path = tmp_path / folder / "eval_base.jsonl"
        evaluate.append_record(path, record("airline-easy-001", "pi", True))
    files = [tmp_path / "one" / "eval_base.jsonl", tmp_path / "two" / "eval_base.jsonl"]
    labels = [c.label for c in noise.your_checkpoints(files)]
    assert labels == ["your base", "your base (2)"]


def test_only_a_perfect_score_gets_the_zero_error_note() -> None:
    for correct, noted in ((False, False), (True, True)):
        checkpoint = noise.Checkpoint("x", (record("a", "pi", correct),))
        lines: list[str] = []
        noise.show_intervals(checkpoint, lines.append)
        assert any("perfect score" in line for line in lines) is noted


def test_noise_refuses_an_empty_records_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    empty = tmp_path / "eval_x.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no records"):
        run_main(noise, ["noise.py", "--records", str(empty)], monkeypatch, capsys)


def test_the_bundled_records_carry_no_personal_paths_and_only_what_is_read() -> None:
    kept = set(evaluate.Record.__dataclass_fields__)
    for path in PROJECT.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text and Path.home().name not in text, path.name
        for line in text.splitlines():
            assert set(json.loads(line)) <= kept, path.name


# ------------------------------------------------- baseline_eval.py, no stack


def test_the_plan_prints_the_protocol_and_the_picks_exactly(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(baseline_eval, ["baseline_eval.py", "--plan"], monkeypatch, capsys)
    assert as_text(out, course_home) == PLAN_GOLDEN.read_text(encoding="utf-8")


def test_fast_mode_is_four_tasks_per_harness(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    argv = ["baseline_eval.py", "--plan", "--fast"]
    out = run_main(baseline_eval, argv, monkeypatch, capsys)
    assert "tasks     4 of 50" in out and "cells     12, one at a time" in out


def test_the_command_line_refuses_an_unknown_harness_or_an_empty_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for flags in (["--harnesses", "codex"], ["--limit", "0"]):
        monkeypatch.setattr(sys, "argv", ["baseline_eval.py", *flags])
        with pytest.raises(SystemExit):
            baseline_eval.parse_args()


def test_an_existing_records_file_needs_resume(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    path = baseline_eval.records_path("scripted-solve")
    evaluate.append_record(path, record("airline-easy-001", "pi", True))
    engine_found = ("http://127.0.0.1:1", "scripted", "scripted")
    monkeypatch.setattr(baseline_eval, "detect_engine", lambda client: engine_found)
    with pytest.raises(SystemExit, match="already holds 1 records: --resume"):
        run_main(baseline_eval, ["baseline_eval.py"], monkeypatch, capsys)


def test_a_missing_harness_is_installed_before_it_is_checked(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    calls = []
    monkeypatch.setattr(baseline_eval, "resolved_version", lambda name: "")
    monkeypatch.setattr(
        baseline_eval,
        "ensure",
        lambda name, install, records: calls.append(install),
    )
    monkeypatch.setattr(
        baseline_eval, "probe_version", lambda name, *args: install.PINS[name].version
    )
    version = baseline_eval.check_harness("pi", EVERY[0], baseline_eval.Options())
    assert version == install.PINS["pi"].version and calls == [True]
    assert "pi is not installed in" in capsys.readouterr().out


def test_a_harness_at_another_version_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_eval, "resolved_version", lambda name: "1.0.4")
    monkeypatch.setattr(baseline_eval, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline_eval, "probe_version", lambda *args: "1.0.5")
    with pytest.raises(SystemExit, match="REFUSED · pi reports '1.0.5'"):
        baseline_eval.check_harness("pi", EVERY[0], baseline_eval.Options())


def test_before_cell_runs_before_each_cell_that_runs(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "run_rollout",
        lambda task, harness, setup: (
            order.append(task.task_id) or record(task.task_id, harness, True)
        ),
    )
    picked = evaluate.balanced(EVERY, 3)
    done = [("pi", picked[0].task_id)]
    evaluate.evaluate(
        picked,
        ["pi"],
        None,
        course_home / "eval.jsonl",
        done,
        lambda line: None,
        before_cell=lambda: order.append("wait"),
    )
    assert order == ["wait", picked[1].task_id, "wait", picked[2].task_id]


def test_the_before_cell_command_is_waited_for_and_can_stop_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    assert baseline_eval.waiter(None) is None
    baseline_eval.waiter("true")()
    assert capsys.readouterr().out == ""  # a quick check prints nothing
    monkeypatch.setattr(baseline_eval, "PAUSE_SHOWN_S", 0.0)
    baseline_eval.waiter("true")()
    assert "paused 0 min 0 s for --before-cell" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exited 3: .* --resume"):
        baseline_eval.waiter("exit 3")()


def test_a_label_names_a_file_so_it_stays_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["baseline_eval.py", "--label", "../x"])
    with pytest.raises(SystemExit):
        baseline_eval.parse_args()
    monkeypatch.setattr(sys, "argv", ["baseline_eval.py", "--label", "base-fast"])
    assert baseline_eval.parse_args().label == "base-fast"


def test_resume_cuts_a_half_written_last_line_and_runs_that_cell_again(
    course_home: Path, capsys: Any
) -> None:
    path = baseline_eval.records_path("base")
    evaluate.append_record(path, record("airline-easy-001", "pi", True))
    with path.open("a") as handle:
        handle.write('{"task_id": "airline-hard-001", "harn')  # the power went
    with pytest.raises(SystemExit, match=r"eval_base.jsonl:2: .*delete that line"):
        baseline_eval.kept_records(path, resume=False)
    kept = baseline_eval.kept_records(path, resume=True)
    assert [r.task_id for r in kept] == ["airline-easy-001"]
    assert "cut a half-written last line" in capsys.readouterr().out
    assert path.read_text().endswith("\n") and not evaluate.drop_torn_tail(path)


def test_resume_refuses_records_made_with_another_harness_version() -> None:
    kept = [record("a", "pi", True, version="1.0.3")]
    with pytest.raises(SystemExit, match="recorded with 1.0.3, not 1.0.4"):
        baseline_eval.refuse_other_versions(kept, {"pi": "1.0.4"})
    baseline_eval.refuse_other_versions([record("a", "pi", True)], {"pi": "1.0.4"})


class FakeSession:
    """What the proxy answers when a session opens."""

    session_id = "s0123456789abcdef01234567"


class FakeClient:
    """A capture client that remembers which sessions were closed."""

    def __init__(self) -> None:
        """Start with no session closed."""
        self.closed: list[str] = []

    def open_session(self, *args: Any) -> FakeSession:
        """Open a session.

        Args:
            *args: What a real client takes.

        Returns:
            The session.
        """
        return FakeSession()

    def close(self, session_id: str) -> bool:
        """Close a session.

        Args:
            session_id: The session.

        Returns:
            True.
        """
        self.closed.append(session_id)
        return True


def test_a_cell_that_fails_still_closes_its_session(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any) -> None:
        raise OSError("the harness would not start")

    monkeypatch.setattr(evaluate, "wire", broken)
    client = FakeClient()
    setup = evaluate.CellSetup(client, "http://proxy", "http://engine", "none", "t", {})
    with pytest.raises(OSError, match="would not start"):
        evaluate.run_rollout(EVERY[0], "pi", setup)
    assert client.closed == [FakeSession.session_id]


def test_an_answer_naming_the_home_folder_is_written_with_a_tilde(
    course_home: Path,
) -> None:
    named = record("a", "pi", False, submitted=f"{Path.home()}/answer.txt")
    line = evaluate.cell_line(1, 1, named)
    assert str(Path.home()) not in line and "'~/answer.txt'" in line
    path = course_home / "eval.jsonl"
    evaluate.append_record(path, named)
    assert str(Path.home()) not in path.read_text()


# ------------------------------------------------- fetching what is missing


def test_a_model_on_disk_is_found_and_never_fetched(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    ours = lfm_files.ours_folder()
    ours.mkdir(parents=True)
    for name in lfm_files.BUILD_FILES:
        (ours / name).write_text("x")
    monkeypatch.setattr(fetch_model, "fetch", pytest.fail)
    fetch_model.ensure_model(fast=False, check_only=False)
    assert "model     ready: ours" in capsys.readouterr().out


def test_a_missing_model_is_fetched_whole_and_checked(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    asked = []
    monkeypatch.setattr(fetch_model, "fetch", lambda names: asked.append(names))
    fetch_model.ensure_model(fast=False, check_only=False)
    assert asked == [tuple(item.name for item in lfm_files.FOUR_BIT.files)]
    assert set(asked[0]) == set(fetch_model.FINGERPRINTS)
    with pytest.raises(SystemExit, match="missing: run `uv run fetch_model.py`"):
        fetch_model.ensure_model(fast=True, check_only=True)


def test_a_missing_tokenizer_is_fetched_alone(
    course_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    asked = []

    def fake(names: tuple[str, ...]) -> Path:
        asked.append(names)
        return lfm_files.model_folder(lfm_files.FOUR_BIT)

    monkeypatch.setattr(fetch_model, "fetch", fake)
    fetch_model.ensure_tokenizer(check_only=False)
    assert asked == [fetch_model.TOKENIZER_FILES]
    assert "tokenizer missing: fetching" in capsys.readouterr().out


def test_a_fetched_file_that_is_not_the_pinned_one_is_deleted(tmp_path: Path) -> None:
    wrong = tmp_path / "tokenizer.json"
    wrong.write_text("not the tokenizer")
    with pytest.raises(fetch_model.FetchError, match="not the pinned file"):
        fetch_model.check_files(tmp_path, ("tokenizer.json",))
    assert not wrong.exists()
    with pytest.raises(fetch_model.FetchError, match="missing after the download"):
        fetch_model.check_files(tmp_path, ("config.json",))


def test_a_fetch_error_never_shows_the_home_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def full(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device", f"{Path.home()}/x")

    monkeypatch.setattr(fetch_model, "snapshot_download", full)
    with pytest.raises(fetch_model.FetchError) as caught:
        fetch_model.fetch(("config.json",))
    assert str(Path.home()) not in str(caught.value) and "'~/x'" in str(caught.value)


def test_start_sh_fetches_before_any_engine_starts() -> None:
    text = START_SH.read_text()
    assert 'STATE="$COURSE_HOME/24-measuring"' in text
    body = text[text.index("start_engine() {") :]
    assert body.index("fetch_inputs") < body.index('exec uv run "$ENGINE_SCRIPT"')
    assert 'flags="--tokenizer"' in text and 'flags="--fast"' in text


def test_the_copied_modules_are_chapter_23s_but_for_the_folder_name() -> None:
    for name in COPIED:
        theirs = (CHAPTER_23 / name).read_text(encoding="utf-8")
        ours = (CHAPTER / name).read_text(encoding="utf-8")
        assert ours == theirs.replace("23-three-harnesses", "24-measuring"), name
    theirs = (CHAPTER_23 / "engine.py").read_text(encoding="utf-8").splitlines()
    ours = (CHAPTER / "engine.py").read_text(encoding="utf-8").splitlines()
    assert len(set(ours) ^ set(theirs)) == 2  # one docstring line, changed


def test_start_sh_is_chapter_23s_plus_the_fetch_step() -> None:
    def code(path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8").replace(
            "23-three-harnesses", "24-measuring"
        )
        return [line for line in text.splitlines() if not line.lstrip().startswith("#")]

    theirs, ours = code(CHAPTER_23 / "start.sh"), code(START_SH)
    assert not [line for line in theirs if line not in ours]
    added = [line for line in ours if line not in theirs]
    start = START_SH.read_text(encoding="utf-8")
    block = start[start.index("fetch_inputs() {") : start.index("start_engine() {")]
    assert all(line in block or "fetch_inputs ||" in line for line in added)


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
    """Find a proxy environment at the pinned commit, this chapter's or 23's.

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


def find_installs() -> Path | None:
    """Find a folder holding all three pinned harnesses, this chapter's or 23's.

    Returns:
        The folder, or None.
    """
    for chapter in CHAPTERS_WITH_STACK:
        folder = _HOME / chapter
        if all((folder / install.PINS[h].folder).is_dir() for h in install.HARNESSES):
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
REAL_INSTALLS = find_installs()
needs_stack = pytest.mark.skipif(
    REAL_PROXY_ENV is None or REAL_INSTALLS is None,
    reason="needs the capture proxy (./start.sh install) and the three harnesses "
    "(uv run install.py)",
)


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

    Both listen on free ports, never on the 8119 and 8200 of a learner's own
    stack (or of anything else on the machine).

    Args:
        tmp_path_factory: pytest's temporary folders.

    Yields:
        The proxy's root.
    """
    if REAL_PROXY_ENV is None or REAL_INSTALLS is None:
        pytest.skip("needs the capture proxy and the three harnesses")
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


@pytest.fixture
def harness_installs(course_home: Path) -> None:
    """Point this test's COURSE_HOME at the real harness installs.

    Args:
        course_home: The test's temporary course home.
    """
    chapter = course_home / "24-measuring"
    chapter.mkdir(parents=True)
    assert REAL_INSTALLS is not None
    for name in install.HARNESSES:
        folder = install.PINS[name].folder
        (chapter / folder).symlink_to(REAL_INSTALLS / folder)


def evaluate_cells(
    proxy_url: str, monkeypatch: pytest.MonkeyPatch, capsys: Any, *flags: str
) -> tuple[str, list[evaluate.Record], dict[str, Any]]:
    """Run baseline_eval.py on a small subset, as from the command line.

    Args:
        proxy_url: The proxy.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.
        *flags: Its flags.

    Returns:
        What it printed, the records file and the summary.
    """
    argv = ["baseline_eval.py", "--proxy-url", proxy_url, *flags]
    out = run_main(baseline_eval, argv, monkeypatch, capsys)
    label = re.search(r"/eval_([\w-]+)\.jsonl", out)
    assert label, out
    path = baseline_eval.records_path(label.group(1))
    summary = json.loads(path.with_name(f"summary_{label.group(1)}.json").read_text())
    return out, evaluate.read_records(path), summary


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_every_cell_runs_once_through_the_proxy_and_is_recorded(
    proxy_url: str, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    flags = ("--limit", "2")
    out, records, summary = evaluate_cells(proxy_url, monkeypatch, capsys, *flags)
    assert "script: solve" in out and "cells     6, one at a time" in out
    assert [r.cell for r in records] == [
        (h, t.task_id) for h in install.HARNESSES for t in evaluate.balanced(EVERY, 2)
    ]
    for item in records:
        assert item.correct and item.n_calls == 4 and item.loopback_only
        assert item.engine == "scripted-solve" and item.purpose == "eval"
        assert item.version == install.PINS[item.harness].version
        assert item.first_prompt and item.domain == "airline"
    stops = {r.harness: r.stop for r in records}
    assert stops == {
        "mini-swe-agent": "finish command",
        "claude-code": "no tool call",
        "pi": "no tool call",
    }
    assert summary["average_accuracy"] == 100.0 and summary["cells"] == 6
    assert out.rstrip().endswith("lsof saw only this machine's addresses")
    # Each cell keeps its evidence: the harness's log, the proxy's record, run.json.
    cell = evaluate.cell_folder("scripted-solve", "pi", EVERY[3])
    kept = json.loads((cell / run_one.RUN_FILE).read_text())
    assert kept["record"]["task_id"] == "airline-easy-001"
    assert (cell / run_one.ROLLOUT_FILE).is_file()
    # A second run refuses the file; --resume finishes it without repeating a cell.
    with pytest.raises(SystemExit, match="--resume"):
        evaluate_cells(proxy_url, monkeypatch, capsys, "--limit", "2")
    out, again, _ = evaluate_cells(
        proxy_url, monkeypatch, capsys, "--limit", "2", "--resume"
    )
    assert out.count("already recorded") == 6 and len(again) == 6


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_the_untrained_models_mistakes_are_recorded_as_wrong(
    proxy_url: str, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    flags = ("--limit", "1", "--script", "mistakes")
    _, records, summary = evaluate_cells(proxy_url, monkeypatch, capsys, *flags)
    assert {r.harness: (r.correct, r.stop) for r in records} == {
        "mini-swe-agent": (False, "3 bad replies"),
        "claude-code": (False, "no tool call"),
        "pi": (False, "window full"),
    }
    assert summary["average_accuracy"] == 0.0
    assert all(r.submitted is None and r.reason == "no answer.txt" for r in records)
