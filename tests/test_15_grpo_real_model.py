"""Chapter 15 without MLX: rewards, advantages, re-tokenizing, records, printing.

No test here loads a model, downloads anything or needs MLX. Rewards and
advantages are checked against the project's formulas, re-tokenizing against
hand-made token lists, the bundled project records against the project's
files (as chapter 6 bundles them), and the printing against small made-up
results. MLX's side (sampling, scoring, the update, the scripts end to end)
is in test_15_mlx.py.
"""

from __future__ import annotations

import filecmp
import json
import sqlite3
import statistics
from pathlib import Path

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("15-grpo-real-model")

import database  # noqa: E402
import grading  # noqa: E402
import lfm_files  # noqa: E402
import project_runs  # noqa: E402
import report  # noqa: E402
import retokenize  # noqa: E402
import rewards  # noqa: E402
import rows  # noqa: E402
import stats  # noqa: E402
from grading import Answered  # noqa: E402
from tasks import TEST_FILE, TRAIN_FILE, load_tasks  # noqa: E402

BUDGET = lfm_files.TOKEN_BUDGET


def answered(task_id: str, correct: bool, tokens: int, **extra: object) -> Answered:
    """Make a graded answer for the printing and saving tests.

    Args:
        task_id: The question.
        correct: Its verdict.
        tokens: Its reply's length.
        **extra: Any other field to set.

    Returns:
        The answer.
    """
    fields: dict[str, object] = {
        "task_id": task_id,
        "correct": correct,
        "reason": "correct" if correct else "expected '7', got '14'",
        "sql": "SELECT 1",
        "answer": "1",
        "reply": "Use it.</think>SELECT 1<|im_end|>",
        "tokens": tokens,
        "finish_reason": "stop",
        "seconds": 0.1,
        "form": "plain text",
        "cap": 1024,
        "adapter": "none",
    }
    fields.update(extra)
    return Answered(**fields)


# ---------------------------------------------------------------- the bundle


def test_the_bundle_is_chapter_14s_questions_and_chapter_6s_records() -> None:
    """The questions and databases are chapter 14's; the RL records chapter 6's."""
    mine, chapter_14 = CHAPTER / "data", ROOT / "14-sft-real-model" / "data"
    names = [TRAIN_FILE, TEST_FILE] + [
        f"dbs/{name}.db" for name in ("shop", "school", "library", "clinic", "airline")
    ]
    for name in names:
        assert filecmp.cmp(mine / name, chapter_14 / name, shallow=False), name
    chapter_6 = ROOT / "06-ratio-and-clip" / "data" / "real_runs"
    for attempt in (1, 2):
        ours = mine / "project_runs" / f"rl_attempt{attempt}_metrics.jsonl"
        theirs = chapter_6 / f"rl_attempt_{attempt}_metrics.jsonl"
        assert filecmp.cmp(ours, theirs, shallow=False)
    assert len(load_tasks(TRAIN_FILE)) == 180
    assert len(load_tasks(TEST_FILE)) == 50


def test_the_projects_two_runs_are_read_from_their_files() -> None:
    """Attempt 1: 8 steps, ratio 0.995-1.000; attempt 2: 6 steps, 0.999-1.001."""
    first = project_runs.read_run(project_runs.ATTEMPT_1)
    second = project_runs.read_run(project_runs.ATTEMPT_2)
    assert [step.step for step in first] == list(range(1, 9))
    assert [step.step for step in second] == list(range(1, 7))
    assert f"{min(s.ratio for s in first):.3f}" == "0.995"
    assert f"{max(s.ratio for s in first):.3f}" == "1.000"
    assert f"{min(s.ratio for s in second):.3f}" == "0.999"
    assert f"{max(s.ratio for s in second):.3f}" == "1.001"
    assert max(step.peak_gb for step in second) == pytest.approx(6.88)


def test_a_damaged_or_foreign_metrics_file_is_refused(tmp_path: Path) -> None:
    """Missing, empty, not JSON, a missing field, or another stage: refused."""
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(project_runs.ProjectRunError, match="no such file"):
        project_runs.read_run(path)
    path.write_text("\n")
    with pytest.raises(project_runs.ProjectRunError, match="empty"):
        project_runs.read_run(path)
    for line in ("{not json", '{"stage": "grpo", "step": 1}', '{"stage": "sft"}'):
        path.write_text(line + "\n")
        with pytest.raises(project_runs.ProjectRunError, match="not a GRPO step"):
            project_runs.read_run(path)


def test_the_copied_modules_are_chapter_14s_byte_for_byte() -> None:
    """grader, tasks, stats and lora are chapter 14's files, unchanged."""
    chapter_14 = ROOT / "14-sft-real-model"
    for name in ("grader.py", "tasks.py", "stats.py", "lora.py"):
        assert filecmp.cmp(CHAPTER / name, chapter_14 / name, shallow=False), name


# ---------------------------------------------------------------- grading


def test_queries_are_found_in_each_form_and_a_tool_call_is_never_run() -> None:
    """A tool call, a code block, plain text; only literal strings are taken."""
    call = "<|tool_call_start|>[sql('SELECT 1')]<|tool_call_end|>"
    assert grading.extract_sql(f"x</think>{call}<|im_end|>") == (
        "SELECT 1",
        grading.FORM_TOOL,
    )
    block = "x</think>Here:\n```sql\nSELECT 3\n```"
    assert grading.extract_sql(block) == ("SELECT 3", grading.FORM_BLOCK)
    assert grading.extract_sql("x</think>SELECT 4<|im_end|>") == (
        "SELECT 4",
        grading.FORM_TEXT,
    )
    assert grading.extract_sql("x</think><|im_end|>") == (None, None)
    assert grading.tool_call_sql("[sql(__import__('os').getcwd())]") is None


def test_every_gold_query_grades_right_through_the_chapters_path(
    tmp_path: Path,
) -> None:
    """Each question's own gold SQL, as a reply, is graded right via answer.txt."""
    for task in load_tasks(TRAIN_FILE) + load_tasks(TEST_FILE):
        reply = f"Use it.</think>{' '.join(task.gold_sql.split())}<|im_end|>"
        verdict = grading.grade_reply(task, reply, tmp_path / task.task_id)
        assert verdict.correct, (task.task_id, verdict.reason)


def test_model_written_sql_cannot_write_attach_or_run_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes, ATTACH and unencodable text are refused; a long query is stopped."""
    shop = CHAPTER / "data" / "dbs" / "shop.db"
    before = shop.read_bytes()
    outside = tmp_path / "made.db"
    for sql in (
        f"ATTACH DATABASE '{outside}' AS other",
        f"VACUUM INTO '{outside}'",
        "DELETE FROM customers",
        "CREATE TEMP TABLE scratch (a)",
        "SELECT '\ud800'",
    ):
        with pytest.raises(database.QueryError):
            database.run_query(shop, sql)
    assert not outside.exists()
    assert shop.read_bytes() == before
    monkeypatch.setattr(database, "QUERY_SECONDS", 0.05)
    forever = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
        "SELECT COUNT(*) FROM n"
    )
    with pytest.raises(database.QueryError, match="interrupted"):
        database.run_query(shop, forever)
    assert database.refuse_attach(sqlite3.SQLITE_READ) == sqlite3.SQLITE_OK


# ---------------------------------------------------------------- rewards


def test_the_token_bonus_falls_from_a_tenth_to_zero_at_2000_tokens() -> None:
    """The project's token_bonus: 0.1 at 0 tokens, 0.05 at 1,000, 0 from 2,000."""
    assert rewards.token_bonus(0, BUDGET) == pytest.approx(0.1)
    assert rewards.token_bonus(1000, BUDGET) == pytest.approx(0.05)
    assert rewards.token_bonus(2000, BUDGET) == 0.0
    assert rewards.token_bonus(5000, BUDGET) == 0.0
    assert rewards.token_bonus(10, 0) == 0.0  # a zero budget cannot divide by zero


def test_only_a_right_answer_earns_the_bonus() -> None:
    """Wrong is 0 however short; right is 1 plus the bonus."""
    assert rewards.reward_for(False, 1, BUDGET) == 0.0
    assert rewards.reward_for(True, 100, BUDGET) == pytest.approx(1.095)
    assert rewards.reward_for(True, 3000, BUDGET) == 1.0


def test_advantages_are_the_projects_and_ties_have_none() -> None:
    """(r - mean) / (std + 1e-4); equal rewards, or one reward: None."""
    found = rewards.group_advantages([1.0, 0.0, 0.0, 1.0])
    mean, spread = 0.5, 0.5
    assert found == pytest.approx([(r - mean) / (spread + 1e-4) for r in (1, 0, 0, 1)])
    assert rewards.group_advantages([0.0, 0.0, 0.0, 0.0]) is None
    assert rewards.group_advantages([1.0, 1.00005]) is None  # within 1e-4: a tie
    assert rewards.group_advantages([1.0]) is None


def test_four_right_replies_still_have_contrast_through_their_lengths() -> None:
    """All right but 2 tokens apart is enough: the bonus splits them at full size."""
    lengths = [60, 62, 70, 90]
    found = rewards.group_advantages(
        [rewards.reward_for(True, n, BUDGET) for n in lengths]
    )
    assert found is not None
    assert found[0] > found[1] > found[2] > 0 > found[3]
    assert abs(found[3]) > 1  # the longest pushed down as hard as a wrong one
    assert statistics.fmean(found) == pytest.approx(0.0, abs=1e-9)
    assert rewards.group_outcome([1.1, 1.0, 1.05]) == rewards.ALL_RIGHT
    assert rewards.group_outcome([0.0, 0.0]) == rewards.ALL_WRONG
    assert rewards.group_outcome([1.0, 0.0]) == rewards.MIXED


def test_the_clip_switches_off_only_a_push_that_went_far_enough() -> None:
    """Above 1.2 with A > 0, below 0.8 with A < 0; never the wrong way, never at 0."""
    assert rewards.clip_active(1.25, 1.0)
    assert not rewards.clip_active(1.25, -1.0)
    assert rewards.clip_active(0.75, -1.0)
    assert not rewards.clip_active(0.75, 1.0)
    assert not rewards.clip_active(1.0, 1.0)
    assert not rewards.clip_active(5.0, 0.0)


# ---------------------------------------------------------------- re-tokenizing


def test_an_identical_reencoding_keeps_every_position() -> None:
    """Nothing differs: every position kept with its own log-probability."""
    aligned = retokenize.align_by_position([5, 6, 7], [-0.1, -0.2, -0.3], [5, 6, 7])
    assert not aligned.differs
    assert aligned.mask == [1, 1, 1]
    assert aligned.logprobs == [-0.1, -0.2, -0.3]
    assert aligned.masked == 0 and aligned.after_difference == 0


def test_a_merged_token_shifts_everything_after_it() -> None:
    """[a, x, t, b, c] re-encoded as [a, xt, b, c]: masked by position, counted."""
    sampled, logprobs = [1, 20, 21, 2, 3], [-0.1, -0.2, -0.3, -0.4, -0.5]
    reencoded = [1, 99, 21, 3]  # position 2 agrees again, by coincidence
    aligned = retokenize.align_by_position(sampled, logprobs, reencoded)
    assert aligned.first_difference == 1
    assert aligned.mask == [1, 0, 1, 0]
    assert aligned.logprobs == [-0.1, 0.0, -0.3, 0.0]
    assert aligned.masked == 2
    assert aligned.after_difference == 1  # position 2 borrows another context's
    longer = retokenize.align_by_position([1, 2], [-0.1, -0.2], [1, 2, 3])
    assert longer.first_difference == 2 and longer.mask == [1, 1, 0]
    shorter = retokenize.align_by_position([1, 2, 3], [-0.1, -0.2, -0.3], [1, 2])
    assert shorter.differs and shorter.first_difference == 2 and shorter.masked == 0
    with pytest.raises(ValueError, match="one log-probability"):
        retokenize.align_by_position([1, 2], [-0.1], [1, 2])


def test_reencoding_is_decode_then_encode() -> None:
    """Two byte tokens that decode to one character come back as one token."""
    pieces = {1: "a", 2: "\xe2", 3: "\x91\xa0", 4: "b"}

    def decode(ids: list[int]) -> str:
        """Join the pieces.

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        return "".join(pieces[i] for i in ids)

    def encode(text: str) -> list[int]:
        """Give the merged character one id, every other piece its own.

        Args:
            text: The text.

        Returns:
            The ids.
        """
        merged = text.replace("\xe2\x91\xa0", "\x00")
        return [9 if c == "\x00" else {"a": 1, "b": 4}[c] for c in merged]

    assert retokenize.reencode([1, 2, 3, 4], decode, encode) == [1, 9, 4]


# ---------------------------------------------------------------- the rows


def test_a_reply_becomes_the_proxys_sequence_and_the_projects_row() -> None:
    """Prompt positions read only; each sampled token keeps its old log p."""
    sequence = rows.call_sequence([7, 8, 9], [10, 11], [-0.5, -0.25], [1, 1])
    assert sequence == {
        "input_ids": [7, 8, 9, 10, 11],
        "loss_mask": [0, 0, 0, 1, 1],
        "logprobs": [0.0, 0.0, 0.0, -0.5, -0.25],
    }
    item = rows.item_from_sequence(sequence, advantage=0.75)
    assert item is not None
    assert item.positions == [3, 4]
    assert item.old_logprobs == [-0.5, -0.25]
    assert item.advantage == 0.75 and item.n_tokens == 2
    masked = rows.call_sequence([7], [10, 11], [-0.5, 0.0], [1, 0])
    assert rows.item_from_sequence(masked).positions == [1]
    assert rows.item_from_sequence(rows.call_sequence([7], [10], [0.0], [0])) is None
    with pytest.raises(ValueError, match="one log-probability"):
        rows.call_sequence([7], [10, 11], [-0.5], [1, 1])


def test_balanced_takes_round_robin_over_database_and_difficulty() -> None:
    """Ten held-out questions: both databases, all three difficulties."""
    picked = rows.balanced(load_tasks(TEST_FILE), 10)
    assert len(picked) == 10
    assert {t.domain for t in picked} == {"clinic", "airline"}
    assert {t.difficulty for t in picked} == {"easy", "medium", "hard"}
    assert rows.balanced(load_tasks(TEST_FILE), 100) == load_tasks(TEST_FILE)


# ---------------------------------------------------------------- files


def test_files_go_under_the_chapters_folder(course_home: Path) -> None:
    """Adapters, logs and work folders under 15-grpo-real-model; inits from 14."""
    home = course_home / "15-grpo-real-model"
    assert lfm_files.run_name([]) == "grpo.safetensors"
    assert lfm_files.run_name(["topk50", "steps4"]) == "grpo_topk50_steps4.safetensors"
    adapter = lfm_files.adapter_path("grpo.safetensors")
    assert adapter == home / "grpo.safetensors"
    assert lfm_files.metrics_path(adapter) == home / "grpo.metrics.jsonl"
    assert lfm_files.rollouts_path(adapter) == home / "grpo.rollouts.jsonl"
    assert lfm_files.eval_path("sft") == home / "eval_sft.jsonl"
    assert lfm_files.work_folder("grpo", "q-1") == home / "work" / "grpo" / "q-1"
    sft = course_home / "14-sft-real-model" / "sft.safetensors"
    assert lfm_files.init_path("sft.safetensors") == sft
    assert lfm_files.init_path("/x/a.safetensors") == Path("/x/a.safetensors")


def test_a_missing_start_adapter_names_chapter_14s_command(course_home: Path) -> None:
    """No sft.safetensors: stop with the command that makes it."""
    path = lfm_files.init_path(lfm_files.SFT_ADAPTER)
    with pytest.raises(SystemExit, match="uv run train_sft.py"):
        lfm_files.require_init(path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert lfm_files.require_init(path) == path


def test_saved_answers_are_kept_only_for_the_same_cap_and_adapter(
    tmp_path: Path,
) -> None:
    """A run with other adapter bytes replaces the file; the same one adds to it."""
    path = tmp_path / "eval.jsonl"
    grading.write_answers(path, [answered("q1", True, 5, adapter="aa")])
    grading.write_answers(path, [answered("q2", False, 9, adapter="aa")])
    assert set(grading.read_answers(path)) == {"q1", "q2"}
    grading.write_answers(path, [answered("q3", True, 5, adapter="bb")])
    assert set(grading.read_answers(path)) == {"q3"}
    grading.write_answers(path, [answered("q4", True, 5, adapter="bb", cap=8)])
    assert set(grading.read_answers(path)) == {"q4"}
    path.write_text("{not json\n")
    assert grading.read_answers(path) == {}


# ---------------------------------------------------------------- printing


def test_shares_and_token_weighted_means() -> None:
    """A share has one decimal; a long reply weighs more in the token mean."""
    assert report.share(1, 8) == "12.5%"
    assert report.share(0, 0) == "n/a"

    class Reply:
        """A reply with a length and a mean entropy."""

        def __init__(self, tokens: int, entropy: float) -> None:
            """Keep the two numbers.

            Args:
                tokens: The reply's length.
                entropy: Its mean entropy.
            """
            self.tokens = tokens
            self.entropy = entropy

    replies = [Reply(10, 1.0), Reply(30, 0.0)]
    assert report.token_mean(replies, "entropy") == pytest.approx(0.25)
    assert report.token_mean([], "entropy") == 0.0
    assert report.describe([]) == "none"
    assert report.describe([0.9, 1.1]) == "mean 1.0000, lowest 0.900, highest 1.100"


def test_the_project_box_shows_both_runs_and_their_ranges(
    capsys: pytest.CaptureFixture,
) -> None:
    """Step by step, then each attempt's range and what differed."""
    report.print_title("grpo_sql.py", "x")
    report.print_project_runs(
        project_runs.read_run(project_runs.ATTEMPT_1),
        project_runs.read_run(project_runs.ATTEMPT_2),
    )
    out = " ".join(capsys.readouterr().out.split())
    assert "1. the project's runs (recorded, not run here)" in out
    assert "attempt 1: ratio 0.995 to 1.000 (the engine's log-probabilities in" in out
    assert "attempt 2: ratio 0.999 to 1.001 (in float32); peak 6.88 GB" in out
    assert "8 12.5% 0.9988" in out  # step 8 exists only in attempt 1
    assert "6 25.0% 0.9973 25.0% 1.0011" in out


def test_the_evaluation_prints_scores_comparisons_and_lengths(
    capsys: pytest.CaptureFixture,
) -> None:
    """Three models on two questions: verdicts, scores, gained and lost, lengths."""
    tasks = load_tasks(TEST_FILE)[:2]
    a, b = tasks[0].task_id, tasks[1].task_id
    base = {a: answered(a, True, 200), b: answered(b, True, 300)}
    sft = {a: answered(a, True, 60), b: answered(b, False, 1024)}
    grpo = {a: answered(a, True, 50), b: answered(b, True, 70)}
    models = [
        report.ModelAnswers("base", "fresh", base, "answered now"),
        report.ModelAnswers("sft", "sft.safetensors", sft, "read"),
        report.ModelAnswers("sft+grpo", "grpo.safetensors", grpo, "answered now"),
    ]
    report.print_title("eval_sql.py", "x")
    report.print_verdicts(models, tasks)
    report.print_scores(models, tasks)
    pairs = [
        ("sft", "sft+grpo", stats.compare({a: True, b: False}, {a: True, b: True}))
    ]
    report.print_comparisons(pairs)
    report.print_lengths(models, tasks)
    report.print_changed(models[1], models[2], [b], {t.task_id: t for t in tasks})
    out = " ".join(capsys.readouterr().out.split())
    assert "sft+grpo 2 of 2 = 100.0% ± 0.0 points" in out
    assert "sft 1 of 2 = 50.0% ± 35.4 points" in out
    assert "sft against sft+grpo: gained 1, lost 0" in out and "p = 1.0000" in out
    assert (
        "on the 1 questions right under both sft and sft+grpo: median 60 -> 50" in out
    )
    assert "questions whose verdict changed from sft to sft+grpo" in out
    assert "sft (1024 tokens, wrong: expected '7', got '14')" in out


def test_saved_metrics_lines_are_json() -> None:
    """The bundled records are one JSON object a line, as the project wrote them."""
    for path in (project_runs.ATTEMPT_1, project_runs.ATTEMPT_2):
        for line in path.read_text().splitlines():
            assert json.loads(line)["stage"] == "grpo"
