"""Chapter 28: read_results.py on the project's recorded run, and its parts.

The pinned output is what ``uv run read_results.py`` prints for the bundled
copy of the project's ``runs/main/``. The project-checkout test reads the
real ``runs/main/`` and its stage log and must print the same, so the
bundle cannot drift from the run it was cut from.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
from chapters import ROOT, use_chapter

use_chapter("28-capstone")

import read_results  # noqa: E402
from read_results import (  # noqa: E402
    Cell,
    Checkpoint,
    pair_results,
    read_run,
    read_stages,
    report,
    shared_solves,
    stage_lines,
)

DATA = ROOT / "tests" / "data"
PINNED = DATA / "28_read_results.txt"
PROJECT = Path(
    os.environ.get("MHRL_PROJECT") or Path.home() / "sai_work" / "multi-harness-rl"
)
PROJECT_MAIN = PROJECT / "runs" / "main"
PROJECT_LOG = PROJECT / "runs" / "logs" / "session_main.log"
SESSION_ID = re.compile(r"\bs[0-9a-f]{24}\b")
TRAIN = ["mini-swe-agent", "claude-code"]


def bundled_report() -> str:
    """The report for the bundled run, as the script prints it.

    Returns:
        The text.
    """
    run = read_run(read_results.PROJECT_RUN)
    return report(run, read_results.PROJECT_RUN / read_results.STAGES_FILE, TRAIN)


def test_the_bundled_run_prints_the_pinned_report() -> None:
    assert bundled_report() + "\n" == PINNED.read_text(encoding="utf-8")


def test_main_prints_the_same(capsys: pytest.CaptureFixture[str]) -> None:
    assert read_results.main([]) == 0
    assert capsys.readouterr().out == PINNED.read_text(encoding="utf-8")


def test_the_report_carries_the_numbers_the_plan_quotes() -> None:
    text = bundled_report()
    # base 100 / 62 / 17 (60), sft 75 / 71 / 38 (61), rl 83 / 75 / 38 (65)
    assert re.search(r"base +24/24 100% .*15/24 62% .*4/24 17% .* 60%", text)
    assert re.search(r"sft +18/24 75% .*17/24 71% .*9/24 38% .* 61%", text)
    assert re.search(r"rl +20/24 83% .*18/24 75% .*9/24 38% .* 65%", text)
    # calls per solve 7.1 -> 4.8 -> 4.3, attributed by stage
    assert re.search(r"base +7\.8 +7\.7 +5\.8 +7\.1", text)
    assert "by stage (average over harnesses): sft -2.3, rl -0.5 calls" in text
    # gained / lost / p for every pair, as chapters 24 and 25 printed them
    assert "base -> sft  mini-swe-agent    24      24     18       0     6" in text
    assert "0.031  lost: beyond noise" in text
    assert "base -> sft  claude-code       24      15     17       7     5" in text
    assert "pi, never trained in: 4 -> 9 solved, base -> rl (8 gained, 3 lost" in text
    assert "grpo   6.88 GB at step 4" in text
    assert not SESSION_ID.search(text)


def test_pi_comes_from_the_rerun_and_says_so() -> None:
    run = read_run(read_results.PROJECT_RUN)
    assert [c.label for c in run.checkpoints] == ["base", "sft", "rl"]
    assert run.others == ("rl_attempt1",)
    base = run.checkpoints[0]
    assert sum(c.correct for c in base.harness_cells("pi")) == 4  # 3 before the fix
    assert "pi for base, sft, rl from eval_<label>-pi-fixed.jsonl" in run.notes[0]


def write_records(folder: Path, label: str, verdicts: dict[str, bool]) -> None:
    """Write a tiny eval file: one harness, the given verdicts.

    Args:
        folder: The run folder.
        label: The checkpoint.
        verdicts: ``{task_id: correct}``.
    """
    lines = [
        json.dumps(
            {
                "task_id": task,
                "harness": "claude-code",
                "correct": ok,
                "n_calls": 4 if ok else 8,
                "seconds": 10.0,
            }
        )
        for task, ok in verdicts.items()
    ]
    (folder / f"eval_{label}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_a_run_in_progress_names_what_is_missing(tmp_path: Path) -> None:
    write_records(tmp_path, "base", {"t1": True, "t2": False})
    run = read_run(tmp_path)
    text = report(run, tmp_path / "stages.txt", TRAIN)
    assert "missing     sft, rl: no records yet" in text
    assert "no stage clock (" in text
    assert "none in this run folder: nothing was checked" in text
    assert "no memory.tsv here" in text


def test_an_empty_folder_says_there_is_nothing_to_read(tmp_path: Path) -> None:
    text = report(read_run(tmp_path), tmp_path / "stages.txt", TRAIN)
    assert text.endswith("no evaluation records yet: nothing to read")


def test_pairs_use_only_the_cells_both_ran(tmp_path: Path) -> None:
    write_records(tmp_path, "base", {"t1": False, "t2": True, "t3": False})
    write_records(tmp_path, "sft", {"t1": True, "t2": True})
    run = read_run(tmp_path)
    [result] = pair_results(run, ["claude-code"])
    assert result.comparison.gained == ("t1",)
    assert result.before == 1 and result.after == 2


def test_shared_solves_pair_calls_on_cells_both_solved() -> None:
    def checkpoint(label: str, calls: dict[str, int | None]) -> Checkpoint:
        """Pi cells: calls when solved, None when not."""
        cells = {
            ("pi", t): Cell(t, "pi", n is not None, n or 8, 1.0)
            for t, n in calls.items()
        }
        return Checkpoint(label, cells)

    before = checkpoint("base", {"a": 8, "b": 6, "c": None})
    after = checkpoint("sft", {"a": 4, "b": None, "c": 3})
    assert shared_solves(before, after, ["pi"]) == [(8, 4)]


def test_the_stage_clock_reads_both_formats_and_midnight(tmp_path: Path) -> None:
    clock = tmp_path / "stages.txt"
    clock.write_text(
        "=== 1/5 baseline ===\n23:50:00\n=== 2/5 demos ===\n00:10:30\n"
        "=== done ===\n2026-10-13 00:20:00\n",
        encoding="utf-8",
    )
    stages = read_stages(clock)
    assert [name for name, _ in stages] == ["1/5 baseline", "2/5 demos", "done"]
    lines = stage_lines(stages[:2])
    assert "20 min 30 s" in lines[0]
    assert "no later banner" in lines[1]
    assert stages[2][1] == datetime(2026, 10, 13, 0, 20)


def test_memory_and_network_files_are_read(tmp_path: Path) -> None:
    write_records(tmp_path, "base", {"t1": True})
    (tmp_path / "memory.tsv").write_text(
        "time\tstage\trole\tpid\tmb\n"
        "10:00:00\t1/5\tengine\t11\t2048\n"
        "10:00:00\t1/5\tstage\t12\t512\n"
        "10:00:15\t1/5\tengine\t11\t3072\n"
        "10:00:30\t1/5\tengine\t11\t30",  # the sampler stopped mid-line
        encoding="utf-8",
    )
    (tmp_path / "network_check.txt").write_text(
        "# Network check\nTCP 127.0.0.1:5->93.184.216.34:443 (ESTABLISHED)\n",
        encoding="utf-8",
    )
    text = report(read_run(tmp_path), tmp_path / "stages.txt", TRAIN)
    assert "1/5      engine 3.00 · stage 0.50 · all at once 3.00 (largest)" in text
    assert "1 connection line(s) off this machine:" in text


@pytest.mark.skipif(
    not (PROJECT_MAIN.is_dir() and PROJECT_LOG.is_file()),
    reason="needs the project's checkout with its recorded runs/main/",
)
def test_the_real_runs_main_prints_the_same_report(tmp_path: Path) -> None:
    clock = tmp_path / "stages.txt"
    clock.write_text(
        "\n".join(
            line
            for line in PROJECT_LOG.read_text(encoding="utf-8").splitlines()
            if line.startswith("===") or re.fullmatch(r"\d\d:\d\d:\d\d", line)
        )
        + "\n",
        encoding="utf-8",
    )
    real = report(read_run(PROJECT_MAIN), clock, TRAIN)
    bundled = bundled_report()

    def body(text: str) -> list[str]:
        """The report from its first table on, without the clock's path."""
        start = text.index("━━ the final table")
        return [
            line
            for line in text[start:].splitlines()
            if "stage clock, banner to banner" not in line
        ]

    assert body(real) == body(bundled)
    results = json.loads((PROJECT_MAIN / "results.json").read_text(encoding="utf-8"))
    run = read_run(PROJECT_MAIN)
    for checkpoint in run.checkpoints:
        for harness, summary in results[checkpoint.label]["per_harness"].items():
            cells = checkpoint.harness_cells(harness)
            assert sum(c.correct for c in cells) == summary["solved"]


def test_a_dated_clock_totals_without_wrapping(tmp_path: Path) -> None:
    clock = tmp_path / "stages.txt"
    clock.write_text(
        "=== 1/5 baseline ===\n2026-10-13 23:00:00\n"
        "=== done ===\n2026-10-15 00:30:00\n",
        encoding="utf-8",
    )
    write_records(tmp_path, "base", {"t1": True})
    text = report(read_run(tmp_path), clock, TRAIN)
    assert re.search(r"all stages +1530 min 00 s", text)  # 25.5 hours, not 1.5


def test_a_broken_row_is_named_never_read_as_a_shorter_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_records(tmp_path, "base", {"t1": True})
    with (tmp_path / "eval_base.jsonl").open("a", encoding="utf-8") as records:
        records.write('{"task_id": "t2", "harness": "pi"')  # still being written
    assert read_results.main(["--run-dir", str(tmp_path)]) == 1
    assert "eval_base.jsonl:2:" in capsys.readouterr().err
    (tmp_path / "eval_base.jsonl").write_text(
        '{"task_id": "t2", "harness": "pi", "correct": true}\n', encoding="utf-8"
    )
    assert read_results.main(["--run-dir", str(tmp_path)]) == 1
    assert "row 1: KeyError('n_calls')" in capsys.readouterr().err


def test_a_missing_folder_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert read_results.main(["--run-dir", str(tmp_path / "nowhere")]) == 1
    assert "no run folder at" in capsys.readouterr().err


def test_repeated_cells_and_a_lone_pi_rerun_are_noted(tmp_path: Path) -> None:
    write_records(tmp_path, "base", {"t1": False})
    with (tmp_path / "eval_base.jsonl").open("a", encoding="utf-8") as records:
        records.write(
            json.dumps(
                {
                    "task_id": "t1",
                    "harness": "claude-code",
                    "correct": True,
                    "n_calls": 3,
                }
            )
            + "\n"
        )
    (tmp_path / "eval_sft-pi-fixed.jsonl").write_text(
        json.dumps({"task_id": "t1", "harness": "pi", "correct": True, "n_calls": 3})
        + "\n",
        encoding="utf-8",
    )
    run = read_run(tmp_path)
    assert run.notes == (
        "base: 1 row(s) repeat a cell; the last record of each was kept",
        "sft: only a pi re-run exists, so it is left out",
    )
    assert run.checkpoints[0].cells[("claude-code", "t1")].correct


def test_other_harnesses_held_out_follow_the_flag(tmp_path: Path) -> None:
    write_records(tmp_path, "base", {"t1": True})
    write_records(tmp_path, "rl", {"t1": True})
    text = report(read_run(tmp_path), tmp_path / "stages.txt", ["mini-swe-agent"])
    assert "harnesses   claude-code (never trained in)" in text
    assert "claude-code, never trained in: 1 -> 1 solved" in text
