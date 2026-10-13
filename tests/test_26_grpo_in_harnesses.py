"""Chapter 26 without a model: the recorded runs, their numbers, the guard, rollouts.

The project's two GRPO runs are bundled slim (``data/project_grpo/``: numbers
only). These tests check the bundle against what the project recorded (its
metrics, its printed logs, every reward, every training row count), pin the
outputs of ``replay.py attempt1``, ``replay.py attempt2`` and ``steps.py``,
and check the rollout runner's order and failures with a stand-in for the
harness. When the project's own files are on this machine
(``~/sai_work/multi-harness-rl/runs/main``), the bundle is rebuilt from them
and must come out byte for byte the same. The training loop is in
``test_26_mlx.py``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("26-grpo-in-harnesses")

import grpo_records  # noqa: E402
import replay  # noqa: E402
import rewards  # noqa: E402
import rollouts  # noqa: E402
import steps  # noqa: E402
import tasks  # noqa: E402
from capture_client import ProxyError  # noqa: E402

DATA = Path(__file__).parent / "data"
CHAPTER_25 = ROOT / "25-sft-in-harnesses"
CHAPTER_15 = ROOT / "15-grpo-real-model"
BUNDLE = grpo_records.BUNDLE_DIR
TRAIN = tasks.load_tasks(tasks.TRAIN_FILE)
# The project's own files, if this machine has them (read before conftest
# moves COURSE_HOME; they are never written).
PROJECT_MAIN = Path(
    os.environ.get("MHRL_PROJECT_RUNS")
    or Path.home() / "sai_work" / "multi-harness-rl" / "runs" / "main"
)
HAS_PROJECT = (PROJECT_MAIN / "grpo_attempt1" / "rollouts.jsonl").is_file()
# Every file chapter 26 copies from chapter 25, unchanged but for its name.
COPIED = (
    "baseline_eval.py",
    "capture_client.py",
    "engine.py",
    "evaluate.py",
    "fetch_model.py",
    "grader.py",
    "install.py",
    "lfm_files.py",
    "lfm_template.py",
    "lfm_text.py",
    "lfm.py",
    "loopback.py",
    "lora.py",
    "mini_swe_agent.yaml",
    "mini-swe-agent-requirements.txt",
    "noise.py",
    "policy.py",
    "proxy-requirements.txt",
    "run_one.py",
    "sandbox.py",
    "sandbox.sb",
    "scripted_engine.py",
    "scripted_replies.py",
    "start.sh",
    "stats.py",
    "tasks.py",
    "teacher.py",
    "toolcalls.py",
    "training.py",
    "wire.py",
)
# The plan's numbers (course/PLAN.md, chapter 26), and what they are.
PLAN_ATTEMPT1 = [46, 38, 40, 57, 127, 105, 365, 404]  # mhrl.cli steps' column
PLAN_ATTEMPT2 = [51, 43, 43, 40, 50, 45]  # mhrl.cli steps' column
PLAN_ACCURACY2 = [38, 25, 62, 75, 50, 25]


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


def run(name: str) -> list[list[grpo_records.Rollout]]:
    """One bundled run's rollouts, step by step.

    Args:
        name: ``attempt1`` or ``attempt2``.

    Returns:
        The steps.
    """
    return grpo_records.by_step(grpo_records.read_rollouts(BUNDLE / name))


def metrics_of(name: str) -> list[dict[str, Any]]:
    """One bundled run's metrics.

    Args:
        name: ``attempt1`` or ``attempt2``.

    Returns:
        One record per step.
    """
    return grpo_records.read_metrics(BUNDLE / name)


# ------------------------------------------------------------------ the files


def test_the_copied_files_are_chapter_25s_but_for_the_folder_name() -> None:
    for name in COPIED:
        theirs = (CHAPTER_25 / name).read_text(encoding="utf-8")
        theirs = theirs.replace("25-sft-in-harnesses", "26-grpo-in-harnesses")
        assert (CHAPTER / name).read_text(encoding="utf-8") == theirs, name


def test_the_tasks_databases_and_evaluations_are_chapter_25s_plus_attempt_1() -> None:
    for name in ("train.jsonl", "test.jsonl"):
        assert (CHAPTER / "data" / name).read_bytes() == (
            CHAPTER_25 / "data" / name
        ).read_bytes()
    for found in (CHAPTER_25 / "data" / "dbs").iterdir():
        mine = CHAPTER / "data" / "dbs" / found.name
        assert mine.read_bytes() == found.read_bytes()
    ours = CHAPTER / "data" / "project_records"
    theirs = CHAPTER_25 / "data" / "project_records"
    for found in theirs.iterdir():
        assert (ours / found.name).read_bytes() == found.read_bytes()
    added = {p.name for p in ours.iterdir()} - {p.name for p in theirs.iterdir()}
    assert added == {"eval_rl_attempt1.jsonl"}
    # Cut to the fields noise.py reads, as chapter 24 cut the others.
    first = json.loads((ours / "eval_rl_attempt1.jsonl").read_text().splitlines()[0])
    assert set(first) == set(
        json.loads((theirs / "eval_rl.jsonl").read_text().splitlines()[0])
    )


def test_the_bundled_metrics_are_the_ones_chapter_15_bundles() -> None:
    for ours, theirs in (("attempt1", "rl_attempt1"), ("attempt2", "rl_attempt2")):
        mine = BUNDLE / ours / "metrics.jsonl"
        bundled = CHAPTER_15 / "data" / "project_runs" / f"{theirs}_metrics.jsonl"
        assert mine.read_bytes() == bundled.read_bytes()


def test_the_bundle_holds_numbers_task_ids_and_harness_names_only() -> None:
    allowed = {
        "step",
        "task_id",
        "harness",
        "correct",
        "n_calls",
        "n_sampled",
        "reward",
        "seconds",
        "rows",
    }
    for name in ("attempt1", "attempt2"):
        for line in (BUNDLE / name / "rollouts.jsonl").read_text().splitlines():
            record = json.loads(line)
            assert set(record) == allowed
            assert record["harness"] in rollouts.TRAIN_HARNESSES
            for row in record["rows"]:
                assert set(row) == {"tokens", "shared", "replies"}
                assert all(len(reply) == 4 for reply in row["replies"])
    for folder in (BUNDLE, grpo_records.LOGS_DIR):
        for path in folder.rglob("*"):
            if path.is_file():
                assert "/Users/" not in path.read_text(), path


@pytest.mark.skipif(not HAS_PROJECT, reason="needs the project's runs/main")
def test_the_bundle_is_the_projects_records_slimmed(tmp_path: Path) -> None:
    for ours, theirs in (("attempt1", "grpo_attempt1"), ("attempt2", "grpo")):
        grpo_records.write_slim(PROJECT_MAIN / theirs, tmp_path / ours, print)
        for name in ("rollouts.jsonl", "metrics.jsonl"):
            assert (tmp_path / ours / name).read_bytes() == (
                BUNDLE / ours / name
            ).read_bytes()
    logs = PROJECT_MAIN.parent / "logs"
    for name in ("take3_steps", "take4_steps", "take4_grpo"):
        mine = grpo_records.LOGS_DIR / f"{name}.txt"
        assert mine.read_bytes() == (logs / f"{name}.log").read_bytes()
    for name in ("summary_rl_attempt1.json", "summary_rl.json"):
        # The repository's hooks end every file with a newline; these had none.
        mine = (grpo_records.LOGS_DIR / name).read_bytes()
        assert mine == (PROJECT_MAIN / name).read_bytes() + b"\n"


# ------------------------------------------------- the records, re-measured


def test_every_recorded_reward_comes_back_from_its_own_bonus() -> None:
    for step in run("attempt1"):  # the article's bonus for fewer calls
        for rollout in step:
            assert rewards.call_reward_for(rollout.correct, rollout.n_calls) == (
                pytest.approx(rollout.reward, abs=1e-12)
            )
    for step in run("attempt2"):  # the token bonus
        for rollout in step:
            assert rollout.n_sampled is not None
            assert rewards.reward_for(rollout.correct, rollout.n_sampled) == (
                pytest.approx(rollout.reward, abs=1e-12)
            )


def test_counting_the_rows_gives_n_sampled_on_all_48_rollouts_of_attempt_2() -> None:
    for step in run("attempt2"):
        for rollout in step:
            counted = replace(rollout, n_sampled=None)
            assert counted.sampled_tokens() == rollout.n_sampled


def test_each_steps_counts_are_the_trainers_own() -> None:
    for name in ("attempt1", "attempt2"):
        for step, metric in zip(run(name), metrics_of(name), strict=True):
            assert len(step) == grpo_records.ROLLOUTS_PER_STEP
            assert sum(r.correct for r in step) / len(step) == metric["accuracy"]
            assert sum(r.reward for r in step) / len(step) == pytest.approx(
                metric["mean_reward"], abs=1e-12
            )
            assert sum(r.n_calls for r in step) / len(step) == metric["mean_calls"]
            given = grpo_records.exposure(step).values()
            assert (
                sum(g.groups - g.contrast for g in given)
                == (metric["groups_without_contrast"])
            )
            assert sum(g.rows for g in given) == metric["train_rows"]
            assert sum(g.trained for g in given) == metric["train_tokens"]


def test_attempt_1_already_trained_each_shared_prefix_once() -> None:
    # The project's README lists the dedupe among the fixes made after
    # attempt 1; attempt 1's own train_tokens say it was already there.
    plain_differs = []
    for step, metric in zip(run("attempt1"), metrics_of("attempt1"), strict=True):
        plain = 0
        for start in range(0, len(step), grpo_records.GROUP_SIZE):
            group = step[start : start + grpo_records.GROUP_SIZE]
            if rewards.group_advantages([r.reward for r in group]) is None:
                continue
            plain += sum(r.length for x in group for row in x.rows for r in row.replies)
        if plain != metric["train_tokens"]:
            plain_differs.append(metric["step"])
    assert plain_differs == [3, 5, 6, 7]


def test_the_guard_reads_the_median_attempt_2_recorded_as_tokens_per_call() -> None:
    for step, metric in zip(run("attempt2"), metrics_of("attempt2"), strict=True):
        assert grpo_records.step_length(step).median == pytest.approx(
            metric["tokens_per_call"], abs=1e-9
        )


def test_the_plans_series_are_the_mean_mhrl_cli_steps_printed() -> None:
    printed1 = replay.printed_series(grpo_records.LOGS_DIR, "take3_steps")
    printed2 = replay.printed_series(grpo_records.LOGS_DIR, "take4_steps")
    assert printed1 == PLAN_ATTEMPT1 and printed2 == PLAN_ATTEMPT2
    assert [round(grpo_records.steps_printed(s)) for s in run("attempt1")] == printed1
    assert [round(grpo_records.steps_printed(s)) for s in run("attempt2")] == printed2
    medians1 = [round(grpo_records.step_length(s).median) for s in run("attempt1")]
    means1 = [round(grpo_records.step_length(s).mean) for s in run("attempt1")]
    medians2 = [round(grpo_records.step_length(s).median) for s in run("attempt2")]
    assert medians1 == [46, 38, 40, 45, 130, 111, 492, 562]
    assert means1 == [49, 38, 41, 57, 134, 120, 465, 654]
    assert medians2 == [46, 40, 43, 37, 45, 43]
    accuracy2 = [round(100 * m["accuracy"] + 1e-9) for m in metrics_of("attempt2")]
    assert [f"{100 * m['accuracy']:.0f}" for m in metrics_of("attempt2")] == [
        str(a) for a in PLAN_ACCURACY2
    ]
    assert accuracy2 == [38, 25, 63, 75, 50, 25]  # 62.5 prints as 62 (half to even)


def test_the_task_order_is_the_projects_for_every_recorded_step() -> None:
    plan = rollouts.plan_groups(TRAIN, rollouts.TRAIN_HARNESSES, 8, 2)
    for name in ("attempt1", "attempt2"):
        for step, groups in zip(run(name), plan, strict=False):
            recorded = [
                (step[start].task_id, step[start].harness)
                for start in range(0, len(step), grpo_records.GROUP_SIZE)
            ]
            assert recorded == [(task.task_id, harness) for task, harness in groups]
            for start in range(0, len(step), grpo_records.GROUP_SIZE):
                group = step[start : start + grpo_records.GROUP_SIZE]
                assert len({(r.task_id, r.harness) for r in group}) == 1


# ------------------------------------------------------------------- the guard


def rollout(per_call: int, calls: int = 4) -> grpo_records.Rollout:
    """A rollout with a given number of sampled tokens per call.

    Args:
        per_call: Tokens per call.
        calls: Its calls.

    Returns:
        The rollout.
    """
    return grpo_records.Rollout(1, "t", "h", True, calls, per_call * calls, 1.0, 1.0)


def test_the_guard_blames_the_last_update_and_keeps_the_one_before() -> None:
    history = [None, "a1", "a2"]
    calm = grpo_records.length_guard([rollout(40)] * 8, history)
    assert not calm.stop and calm.kept is None
    # One ballooned rollout moves the mean, not the median.
    one = [rollout(40)] * 7 + [rollout(2_000)]
    assert grpo_records.step_length(one).mean > 150
    assert not grpo_records.length_guard(one, history).stop
    many = [rollout(40)] * 3 + [rollout(400)] * 5
    verdict = grpo_records.length_guard(many, history)
    assert verdict.stop and verdict.blamed == "a2" and verdict.kept == "a1"
    assert grpo_records.length_guard(many, [None]).kept is None  # nothing to keep
    exactly = grpo_records.length_guard([rollout(150)] * 8, history)
    assert not exactly.stop  # "passes 150": at 150 it holds


def test_a_rollout_without_calls_is_left_out_of_both_statistics() -> None:
    length = grpo_records.step_length([rollout(40), rollout(0, calls=0)])
    assert length == grpo_records.StepLength(40.0, 40.0, 1)
    assert grpo_records.step_length([]) == grpo_records.StepLength(0.0, 0.0, 0)


# -------------------------------------------------- the scripts' pinned output


def test_replay_attempt1_prints_the_collapse_as_recorded(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(replay, ["replay.py", "attempt1"], monkeypatch, capsys)
    assert out == (DATA / "26_replay_attempt1.txt").read_text()
    assert "    46 38 40 57 127 105 365 404" in out
    assert "the median, what the guard reads, 46 38 40 45 130 111 492 562." in out
    assert "of the 22 replies at the cap, 21 hold no tool call" in out
    assert "     8      12%    0.13    3.8     562   654    15 of 30" in out
    assert "claude-code     2 of 8      10           2,653    1,510 (7%)" in out
    assert "adapter_step004.safetensors" in out
    assert (
        "project sft -> attempt1 adapter_step004  claude-code        24      17      0"
        "       0    17       17  0.000  lost: beyond noise"
    ) in out
    assert "  pi  " not in out


def test_replay_attempt2_prints_the_run_the_guard_never_stopped(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(replay, ["replay.py", "attempt2"], monkeypatch, capsys)
    assert out == (DATA / "26_replay_attempt2.txt").read_text()
    assert "    51 43 43 40 50 45" in out
    assert "the guard read the median: 46 40 43 37 45 43." in out
    assert "ratio 0.999 to 1.001 over the 6 steps; trainer peak 6.88 GB (step 4)" in out
    assert "6 steps x 8 rollouts = 48 rollouts" in out
    assert "and at 50% the standard error is 17.7 points" in out
    assert "claude-code     4 of 6      18           3,061    4,472 (42%)" in out
    assert "mini-swe-agent  6 of 6      30           1,275    6,122 (58%)" in out
    for accuracy in PLAN_ACCURACY2:
        assert f"{accuracy}%" in out


@pytest.mark.skipif(not HAS_PROJECT, reason="needs the project's runs/main")
def test_replay_reads_the_projects_own_files_to_the_same_output(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    for name in ("attempt1", "attempt2"):
        argv = ["replay.py", name, "--records", str(PROJECT_MAIN)]
        out = run_main(replay, argv, monkeypatch, capsys).splitlines()
        golden = (DATA / f"26_replay_{name}.txt").read_text().splitlines()
        assert out[0] == golden[0] and out[2:] == golden[2:]
        assert "(the project's own records)" in out[1]


def test_steps_prints_both_statistics_and_the_projects_other_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.chdir(CHAPTER)
    for name, log in (("attempt1", "take3_steps"), ("attempt2", "take4_steps")):
        argv = ["steps.py", "--dir", f"data/project_grpo/{name}"]
        out = run_main(steps, argv, monkeypatch, capsys)
        assert out == (DATA / f"26_steps_{name}.txt").read_text()
        printed = (grpo_records.LOGS_DIR / f"{log}.txt").read_text().splitlines()[1:]
        ours = out.splitlines()[2 : 2 + len(printed)]
        for theirs, mine in zip(printed, ours, strict=True):
            step, correct, calls, _, seconds = theirs.split()
            assert mine.split()[:3] == [step, correct, calls]
            assert mine.split()[-1] == seconds


def test_steps_says_when_the_guard_stopped_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    folder = tmp_path / "run"
    folder.mkdir()
    lines = (BUNDLE / "attempt1" / "rollouts.jsonl").read_text().splitlines()
    (folder / "rollouts.jsonl").write_text("\n".join(lines[:56]) + "\n")
    stop = {
        "stage": "grpo",
        "step": 7,
        "stopped": "length_guard",
        "tokens_per_call": 491.8,
        "kept": "adapter_step005.safetensors",
    }
    metrics = (BUNDLE / "attempt1" / "metrics.jsonl").read_text().splitlines()[:6]
    (folder / "metrics.jsonl").write_text("\n".join([*metrics, json.dumps(stop)]))
    out = run_main(steps, ["steps.py", "--dir", str(folder)], monkeypatch, capsys)
    assert "   7      12%    3.4                  492    465" in out
    assert (
        "step 7: the length guard stopped the run (median 492); it kept "
        "adapter_step005.safetensors"
    ) in out


def test_steps_before_any_run_says_there_is_none_yet(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    with pytest.raises(SystemExit, match="no GRPO steps recorded yet"):
        run_main(steps, ["steps.py"], monkeypatch, capsys)


# ---------------------------------------------------------------- the rollouts


def setup() -> rollouts.RolloutSetup:
    """A rollout setup that never reaches a proxy (the tests replace the run).

    Returns:
        The setup.
    """
    from capture_client import CaptureClient

    client = CaptureClient("http://127.0.0.1:1")
    return rollouts.RolloutSetup(client, client.proxy_url, rewards_fn, "test")


def rewards_fn(correct: bool, n_sampled: int, n_calls: int) -> float:
    """The token bonus, as a run's reward hook.

    Args:
        correct: Whether the answer was right.
        n_sampled: Sampled tokens.
        n_calls: Calls (unused by this reward).

    Returns:
        The reward.
    """
    return rewards.reward_for(correct, n_sampled)


def jobs(count: int) -> list[rollouts.Job]:
    """A step's jobs: one group per task, four rollouts each.

    Args:
        count: How many rollouts.

    Returns:
        The jobs.
    """
    harnesses = rollouts.TRAIN_HARNESSES
    return [
        rollouts.Job(TRAIN[i // 4], harnesses[i // 4 % 2], 1, i // 4, i % 4)
        for i in range(count)
    ]


def test_rollouts_come_back_in_job_order_whatever_order_they_end() -> None:
    said: list[str] = []
    finished: list[int] = []
    lock = threading.Lock()

    def slow_first(
        job: rollouts.Job, _: rollouts.RolloutSetup
    ) -> rollouts.RolloutRecord:
        position = job.group * 4 + job.index
        time.sleep(0.05 * (8 - position))  # the first job ends last
        with lock:
            finished.append(position)
        return rollouts.RolloutRecord(job.task.task_id, job.harness, group=job.group)

    records, failed = rollouts.run_step(jobs(8), setup(), 4, said.append, slow_first)
    assert failed == [] and finished != sorted(finished)
    assert [(r.task_id, r.group) for r in records] == [
        (TRAIN[i // 4].task_id, i // 4) for i in range(8)
    ]
    assert len(said) == 8 and said[0].startswith("  [1/8] ")


def test_a_dead_proxy_stops_every_rollout_not_yet_started() -> None:
    said: list[str] = []

    def dead(job: rollouts.Job, _: rollouts.RolloutSetup) -> rollouts.RolloutRecord:
        raise ProxyError("cannot reach the proxy")

    records, failed = rollouts.run_step(jobs(4), setup(), 1, said.append, dead)
    assert records == [] and len(failed) == 4
    assert failed[0].reason == "cannot reach the proxy"
    assert {f.reason for f in failed[1:]} == {"not run: the proxy failed earlier"}


def test_a_rollout_runs_in_its_own_folder_with_its_state_in_the_temp_folder(
    course_home: Path,
) -> None:
    job = jobs(1)[0]
    folder = rollouts.rollout_folder("myrun", job)
    assert folder == (
        course_home
        / "26-grpo-in-harnesses"
        / "runs"
        / "grpo"
        / "myrun"
        / "step_001"
        / job.harness
        / f"{job.task.task_id}-0"
    )
    folders = rollouts.prepare_rollout(folder, job.harness, job.task)
    try:
        assert (folders.task / "data.db").is_file()
        assert not folders.state.is_relative_to(course_home)
        assert folders.state.name.startswith(f"mhrl-harness-{job.harness}-")
        assert folders.home.is_dir() and folders.tmp.is_dir()
    finally:
        import shutil

        shutil.rmtree(folders.state)


def test_only_trainable_rows_are_kept_with_the_three_fields_training_reads() -> None:
    document = {
        "sequences": [
            {"input_ids": [1], "loss_mask": [0], "logprobs": [0.0], "trainable": True},
            {"input_ids": [2], "loss_mask": [1], "logprobs": [0.1], "trainable": False},
        ]
    }
    assert rollouts.trainable_rows(document) == [
        {"input_ids": [1], "loss_mask": [0], "logprobs": [0.0]}
    ]


def test_a_progress_line_says_the_verdict_reward_calls_and_tokens() -> None:
    record = rollouts.RolloutRecord(
        "shop-hard-005",
        "claude-code",
        correct=True,
        n_calls=5,
        n_sampled=230,
        reward=1.0885,
        seconds=12.34,
        sequences=[{"input_ids": [0] * 3_100, "loss_mask": [], "logprobs": []}],
        stop="no tool call",
    )
    assert rollouts.rollout_line(3, 8, record) == (
        "  [3/8] claude-code    shop-hard-005      right · reward 1.09 · 5 calls · "
        "230 sampled (46 per call) · 1 row, 3,100 tokens · 12.3 s · no tool call"
    )


def test_records_round_trip_through_the_reader(tmp_path: Path) -> None:
    sequence = {
        "input_ids": [1, 2, 3, 124_905, 9, 124_900, 4, 5],
        "loss_mask": [0, 0, 1, 1, 1, 1, 0, 1],
        "logprobs": [0.0] * 8,
    }
    record = rollouts.RolloutRecord(
        "t", "mini-swe-agent", correct=True, n_calls=2, n_sampled=5, reward=1.09
    )
    record.sequences = [sequence]
    record.step = 4
    rollouts.append_records(tmp_path / "rollouts.jsonl", [record])
    (read,) = grpo_records.read_rollouts(tmp_path, per_step=8)
    assert read.step == 4 and read.n_sampled == 5
    (row,) = read.rows
    assert row.replies == (
        grpo_records.Reply(2, 4, True, True),
        grpo_records.Reply(7, 1, False, False),
    )
    assert row.new_tokens() == 5
