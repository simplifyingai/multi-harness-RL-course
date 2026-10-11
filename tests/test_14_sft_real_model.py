"""Chapter 14 without MLX: the bundled questions, the rows, grading and the arithmetic.

No test here loads a model, downloads anything or needs MLX. The tokenizer is
a tiny fake that renders a chat the way LFM2.5's template does (the same
markers, the same generation prompt) and gives every character its own id,
so the rows' loss masks can be checked position by position. Every one of
the 230 gold queries is graded through the chapter's own path (extract, run,
answer.txt, chapter 1's grader). MLX's side (scoring, training, the scripts
end to end) is in test_14_mlx.py.
"""

from __future__ import annotations

import filecmp
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("14-sft-real-model")

import database  # noqa: E402
import grading  # noqa: E402
import lfm_files  # noqa: E402
import project_run  # noqa: E402
import report  # noqa: E402
import rows  # noqa: E402
import stats  # noqa: E402
from grading import Answered  # noqa: E402
from tasks import TEST_FILE, TRAIN_FILE, Task, load_tasks  # noqa: E402

BOS = "<|startoftext|>"
EOS_TEXT = "<|im_end|>"
EOS = 1
SPECIAL = {BOS: 2, "<|im_start|>": 3, EOS_TEXT: EOS, "<think>": 4, "</think>": 5}


class TemplateTokenizer:
    """A fake tokenizer: LFM2.5's chat markers as single ids, one id per character.

    ``apply_chat_template`` renders like LFM2.5's template: a start-of-text
    marker, each message between ``<|im_start|>role`` and ``<|im_end|>``, and
    the generation prompt ``<|im_start|>assistant`` then ``<think>``.
    """

    eos_token_id = EOS
    eos_token_ids = frozenset({EOS})

    def apply_chat_template(
        self, messages: list[dict[str, str]], **options: Any
    ) -> str:
        """Render a chat.

        Args:
            messages: The chat.
            **options: ``add_generation_prompt`` is read; the rest ignored.

        Returns:
            The rendered text.
        """
        text = BOS + "".join(
            f"<|im_start|>{m['role']}\n{m['content']}{EOS_TEXT}\n" for m in messages
        )
        if options.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n<think>"
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Give each marker its id and each other character 100 + its code.

        Args:
            text: The text.
            add_special_tokens: Ignored (the template wrote the markers).

        Returns:
            The ids.
        """
        ids, index = [], 0
        while index < len(text):
            for marker, token in SPECIAL.items():
                if text.startswith(marker, index):
                    ids.append(token)
                    index += len(marker)
                    break
            else:
                ids.append(100 + ord(text[index]))
                index += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn ids back into text.

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        names = {token: marker for marker, token in SPECIAL.items()}
        return "".join(names[i] if i in names else chr(i - 100) for i in ids)


def task(gold_sql: str, task_id: str = "shop-easy-001") -> Task:
    """Make a training question with a given gold query, on the shop database.

    Args:
        gold_sql: The query.
        task_id: Its id.

    Returns:
        The task (its answer is not used by the tests that call this).
    """
    return Task(
        task_id=task_id,
        split="train",
        domain="shop",
        difficulty="easy",
        db_file="dbs/shop.db",
        question="How many customers live in Paris?",
        gold_sql=gold_sql,
        answer="0",
        answer_type="number",
    )


# ---------------------------------------------------------------- the bundle


def test_the_bundle_holds_the_projects_230_questions() -> None:
    """180 training questions over shop, school, library; 50 over clinic, airline."""
    train, test = load_tasks(TRAIN_FILE), load_tasks(TEST_FILE)
    assert len(train) == 180
    assert len(test) == 50
    assert {t.domain for t in train} == {"shop", "school", "library"}
    assert {t.domain for t in test} == {"clinic", "airline"}
    assert {t.split for t in train} == {"train"}
    assert {t.split for t in test} == {"test"}


def test_the_bundle_is_chapter_1s_files_byte_for_byte() -> None:
    """The task files and the five databases match chapter 1's copies."""
    mine, chapter_1 = CHAPTER / "data", ROOT / "01-the-job" / "data"
    names = [TRAIN_FILE, TEST_FILE] + [
        f"dbs/{name}.db" for name in ("shop", "school", "library", "clinic", "airline")
    ]
    for name in names:
        assert filecmp.cmp(mine / name, chapter_1 / name, shallow=False), name


def test_every_gold_query_grades_right_through_the_chapters_path(
    tmp_path: Path,
) -> None:
    """Each training reply's own SQL, graded as eval_sql.py grades a reply."""
    for item in load_tasks(TRAIN_FILE) + load_tasks(TEST_FILE):
        reply = rows.reply_text(item) + EOS_TEXT
        verdict = grading.grade_reply(item, reply, tmp_path / item.task_id)
        assert verdict.correct, (item.task_id, verdict.reason)
        assert verdict.form == grading.FORM_TEXT
        answer = (tmp_path / item.task_id / "answer.txt").read_text()
        assert answer == verdict.answer + "\n"


# ---------------------------------------------------------------- the rows


def test_reasons_say_which_tables_and_what_the_query_does() -> None:
    """The one-line reason is written from the gold SQL."""
    assert rows.reason_for("SELECT COUNT(*) FROM customers WHERE city = 'Paris'") == (
        "Use the customers table: keep the matching rows, then count them."
    )
    assert rows.reason_for(
        "SELECT city FROM customers GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1"
    ) == (
        "Use the customers table: group by city, then sort by the count, highest "
        "first, and keep the top row."
    )
    assert rows.reason_for(
        "SELECT s.name FROM students s JOIN enrollments e ON e.student_id = s.id "
        "GROUP BY s.id HAVING COUNT(*) > 3"
    ) == (
        "Use the students and enrollments tables: join them, group by s.id, keep "
        "the groups that qualify, then return s.name."
    )
    assert rows.reason_for("SELECT name FROM products ORDER BY price LIMIT 3") == (
        "Use the products table: sort by price, lowest first, and keep the top 3 rows."
    )
    assert rows.reason_for("SELECT MAX(price) FROM products") == (
        "Use the products table: take the largest."
    )


def test_a_subquery_ends_the_group_by_and_adds_no_join() -> None:
    """A GROUP BY inside brackets stops at them; NOT EXISTS is not a join."""
    nested = (
        "SELECT COUNT(*) FROM (SELECT customer_id, MIN(order_date) AS first_order "
        "FROM orders GROUP BY customer_id) WHERE strftime('%Y-%m', first_order) = "
        "'2024-04'"
    )
    assert rows.reason_for(nested) == (
        "Use the orders table: keep the matching rows, group by customer_id, then "
        "count them."
    )
    months = "SELECT x FROM loans GROUP BY strftime('%m', day) ORDER BY COUNT(*) DESC"
    assert "group by strftime('%m', day), then" in rows.reason_for(months)
    unread = (
        "SELECT b.title FROM books b WHERE b.genre = 'Poetry' AND NOT EXISTS "
        "(SELECT 1 FROM loans l WHERE l.book_id = b.id)"
    )
    assert rows.reason_for(unread) == (
        "Use the books and loans tables: keep the matching rows, leave out the rows "
        "the inner query finds, then return b.title."
    )


def test_every_bundled_question_gets_a_reason() -> None:
    """No question falls through without naming a table."""
    for item in load_tasks(TRAIN_FILE) + load_tasks(TEST_FILE):
        reason = rows.reason_for(item.gold_sql)
        assert reason.startswith("Use the ") and reason.endswith(".")
        assert "  " not in reason


def test_a_row_is_the_prompt_then_the_reply_with_its_mask() -> None:
    """0s over the chat (ending with <think>), 1s over the reply and <|im_end|>."""
    tokenizer = TemplateTokenizer()
    item = load_tasks(TRAIN_FILE)[0]
    row = rows.build_row(tokenizer, item)
    prompt = rows.prompt_ids(tokenizer, item)
    reply = tokenizer.encode(rows.reply_text(item)) + [EOS]
    assert row.input_ids == prompt + reply
    assert row.loss_mask == [0] * len(prompt) + [1] * len(reply)
    assert row.prompt_length == len(prompt)
    assert row.reply_tokens == len(reply)
    assert row.input_ids[row.prompt_length - 1] == SPECIAL["<think>"]
    assert row.input_ids[0] == SPECIAL[BOS]
    text = tokenizer.decode(prompt)
    assert rows.SYSTEM_RULE in text
    assert "CREATE TABLE customers" in text
    assert item.question in text


def test_the_first_token_is_never_a_target() -> None:
    """item_from_sequence keeps mask-1 positions after 0, even if position 0 says 1."""
    sequence = {"input_ids": [7, 8, 9, 10], "loss_mask": [1, 0, 1, 1]}
    item = rows.item_from_sequence(sequence)
    assert item is not None
    assert item.positions == [2, 3]
    assert item.n_tokens == 2
    assert item.old_logprobs == [0.0, 0.0]
    assert item.advantage == 1.0
    assert rows.item_from_sequence({"input_ids": [7, 8], "loss_mask": [1, 0]}) is None


def test_the_template_drops_the_think_the_reply_followed() -> None:
    """Rendering the finished chat differs from the row right where the reply starts."""
    tokenizer = TemplateTokenizer()
    item = load_tasks(TRAIN_FILE)[0]
    row = rows.build_row(tokenizer, item)
    rendered, shared = rows.template_difference(tokenizer, item, row)
    assert shared == row.prompt_length - 1  # everything before the <think>
    assert row.input_ids[shared] == SPECIAL["<think>"]
    assert rendered[shared] != SPECIAL["<think>"]


def test_rows_round_trip_and_a_bad_mask_is_refused(tmp_path: Path) -> None:
    """write_rows then read_rows gives the same rows; a short mask is caught."""
    first = rows.Row("a", [1, 2, 3], [0, 1, 1])
    second = rows.Row("b", [4, 5], [0, 1])
    path = tmp_path / "rows.jsonl"
    assert rows.write_rows(path, [first, second]) == 2
    assert rows.read_rows(path) == [first, second]
    path.write_text('{"task_id": "c", "input_ids": [1, 2], "loss_mask": [0]}\n')
    with pytest.raises(ValueError, match="wrong length"):
        rows.read_rows(path)
    path.write_text('{"task_id": "c"}\n')
    with pytest.raises(ValueError, match="not a row"):
        rows.read_rows(path)


def test_balanced_takes_round_robin_over_database_and_difficulty() -> None:
    """40 of the 180: every (database, difficulty) bucket gets 4 or 5."""
    picked = rows.balanced(load_tasks(TRAIN_FILE), 40)
    assert len(picked) == 40
    buckets: dict[tuple[str, str], int] = {}
    for item in picked:
        key = (item.domain, item.difficulty)
        buckets[key] = buckets.get(key, 0) + 1
    assert len(buckets) == 9
    assert set(buckets.values()) <= {4, 5}
    assert rows.balanced(load_tasks(TRAIN_FILE), 500) == load_tasks(TRAIN_FILE)


# ---------------------------------------------------------------- grading


def test_queries_are_found_in_each_form() -> None:
    """A tool call, a code block, plain text, or nothing after the reasoning."""
    call = "<|tool_call_start|>[sql('SELECT 1 WHERE x = \\'a\\'')]<|tool_call_end|>"
    assert grading.extract_sql(f"think</think>{call}<|im_end|>") == (
        "SELECT 1 WHERE x = 'a'",
        grading.FORM_TOOL,
    )
    keyword = '<|tool_call_start|>[run_sql(query="SELECT 2")]<|tool_call_end|>'
    assert grading.extract_sql(f"x</think>{keyword}") == ("SELECT 2", grading.FORM_TOOL)
    block = "x</think>Here:\n```sql\nSELECT 3\n```\nDone."
    assert grading.extract_sql(block) == ("SELECT 3", grading.FORM_BLOCK)
    assert grading.extract_sql("x</think>SELECT 4<|im_end|>") == (
        "SELECT 4",
        grading.FORM_TEXT,
    )
    assert grading.extract_sql("x</think><|im_end|>") == (None, None)


def test_a_tool_call_is_read_never_run() -> None:
    """Only literal strings are taken: an expression in the call is ignored."""
    assert grading.tool_call_sql("[sql(__import__('os').getcwd())]") is None
    assert grading.tool_call_sql("[sql(]") is None
    assert grading.tool_call_sql("[sql(1), sql('SELECT 5')]") == "SELECT 5"


def test_grading_says_why_a_reply_failed(tmp_path: Path) -> None:
    """Cut off while reasoning, no SQL, an SQL error, an empty result."""
    item = load_tasks(TRAIN_FILE)[0]
    folder = tmp_path / "q"
    cut = grading.grade_reply(item, "still thinking", folder, cut_off=True)
    assert cut.reason == grading.REASON_CUT_OFF
    nothing = grading.grade_reply(item, "x</think>", folder)
    assert nothing.reason == grading.REASON_NO_SQL
    error = grading.grade_reply(item, "x</think>SELECT nope FROM nothing", folder)
    assert error.reason.startswith("SQL error")
    query = "x</think>SELECT city FROM customers WHERE 0"
    empty = grading.grade_reply(item, query, folder)
    assert empty.reason == grading.REASON_NO_ROWS
    assert not (folder / "answer.txt").exists()


def test_an_old_answer_file_is_never_graded_again(tmp_path: Path) -> None:
    """The question's folder is emptied before each reply is graded."""
    item = load_tasks(TRAIN_FILE)[0]
    folder = tmp_path / "q"
    folder.mkdir()
    (folder / "answer.txt").write_text(item.answer)
    verdict = grading.grade_reply(item, "x</think>", folder)
    assert not verdict.correct
    assert not (folder / "answer.txt").exists()


def test_no_query_can_write_attach_or_run_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes and ATTACH are refused (no file appears); a long query is stopped."""
    shop = CHAPTER / "data" / "dbs" / "shop.db"
    before = shop.read_bytes()
    outside = tmp_path / "made.db"
    for sql in (
        f"ATTACH DATABASE '{outside}' AS other",
        "DELETE FROM customers",
        "CREATE TEMP TABLE scratch (a)",
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
    connection = database.connect_read_only(shop)
    try:
        assert connection.execute("PRAGMA query_only").fetchone() == (1,)
    finally:
        connection.close()
    assert database.refuse_attach(sqlite3.SQLITE_READ) == sqlite3.SQLITE_OK


def test_answers_are_formatted_as_chapter_1_formats_them() -> None:
    """Floats to 2 places, lists joined, the first column only."""
    assert grading.format_answer([(2.5, "x")], "number") == "2.50"
    assert grading.format_answer([("a",), ("b",)], "list") == "a, b"
    assert grading.format_answer([(None,)], "text") is None


def test_answers_round_trip_and_a_damaged_file_reads_as_none(tmp_path: Path) -> None:
    """write_answers then read_answers; an unreadable file means answer again."""
    one = Answered("q1", True, "correct", "SELECT 1", "1", "r", 3, "stop", 0.5, "x", 8)
    path = tmp_path / "eval.jsonl"
    grading.write_answers(path, [one], keep_others=False)
    assert grading.read_answers(path) == {"q1": one}
    two = Answered("q2", False, "no", None, None, "r", 8, "length", 0.5, None, 8)
    grading.write_answers(path, [two], keep_others=True)  # q1 is kept
    assert set(grading.read_answers(path)) == {"q1", "q2"}
    other_cap = Answered("q3", True, "correct", "S", "1", "r", 3, "stop", 0.1, "x", 4)
    grading.write_answers(path, [other_cap], keep_others=True)  # another cap
    assert set(grading.read_answers(path)) == {"q3"}
    grading.write_answers(path, [one], keep_others=False)
    assert set(grading.read_answers(path)) == {"q1"}
    path.write_text("{not json\n")
    assert grading.read_answers(path) == {}
    assert grading.read_answers(tmp_path / "missing.jsonl") == {}


# ---------------------------------------------------------------- the arithmetic


def test_the_exact_p_value_is_mcnemars() -> None:
    """Known splits: 0 gained 2 lost is 0.5, 0 and 3 is 0.25, 1 and 1 is 1."""
    assert stats.exact_p(0, 2) == 0.5
    assert stats.exact_p(0, 3) == 0.25
    assert stats.exact_p(1, 1) == 1.0
    assert stats.exact_p(0, 0) == 1.0
    assert stats.exact_p(10, 2) == pytest.approx(0.03857, abs=1e-5)


def test_the_error_bar_is_one_standard_error_in_points() -> None:
    """39 of 50 is 78% with sqrt(0.78 x 0.22 / 50) = 5.86 points."""
    assert stats.pass_at_1(39, 50) == pytest.approx(78.0)
    assert stats.error_bar(39, 50) == pytest.approx(5.858, abs=1e-3)
    assert stats.error_bar(25, 50) == pytest.approx(7.071, abs=1e-3)
    assert stats.error_bar(50, 50) == 0.0
    assert stats.error_bar(0, 0) == 0.0


def test_two_runs_are_compared_question_by_question() -> None:
    """Gained, lost, kept and missed; different questions are refused."""
    before = {"a": True, "b": False, "c": True, "d": False}
    after = {"a": True, "b": True, "c": False, "d": False}
    comparison = stats.compare(before, after)
    assert comparison.gained == ("b",)
    assert comparison.lost == ("c",)
    assert (comparison.both, comparison.neither) == (1, 1)
    assert comparison.p_value == 1.0
    with pytest.raises(ValueError, match="different questions"):
        stats.compare(before, {"a": True})


def test_the_median_of_odd_and_even_lists() -> None:
    """The middle value, or the mean of the middle two."""
    assert stats.median([5, 1, 3]) == 3
    assert stats.median([4, 1, 3, 2]) == 2.5
    assert stats.median([]) == 0.0


# ---------------------------------------------------------------- the project's runs


def test_the_projects_runs_are_read_from_their_files() -> None:
    """15 steps from 2.55 to 0.11; the dry run is 0.17 at step 4, 0.10 at 5."""
    main = project_run.read_run(project_run.MAIN_FILE)
    dry = project_run.read_run(project_run.DRY_FILE)
    assert len(main) == 15
    assert round(main[0].loss_per_token, 2) == 2.55
    assert round(main[-1].loss_per_token, 2) == 0.11
    assert len(dry) == 6
    assert round(dry[3].loss_per_token, 2) == 0.17
    assert project_run.first_below(dry, 0.10) == 5
    assert project_run.first_below(main, 0.01) is None


def test_a_damaged_metrics_file_is_refused(tmp_path: Path) -> None:
    """Missing, empty, or a line without the fields."""
    with pytest.raises(project_run.ProjectRunError, match="no such file"):
        project_run.read_run(tmp_path / "none.jsonl")
    path = tmp_path / "m.jsonl"
    path.write_text("\n")
    with pytest.raises(project_run.ProjectRunError, match="empty"):
        project_run.read_run(path)
    path.write_text('{"step": 1}\n')
    with pytest.raises(project_run.ProjectRunError, match="not a step"):
        project_run.read_run(path)


def test_the_project_box_corrects_the_comment(capsys: pytest.CaptureFixture) -> None:
    """The printed box says what the dry run's file says, beside the comment."""
    report.print_project_runs(
        project_run.read_run(project_run.MAIN_FILE),
        project_run.read_run(project_run.DRY_FILE),
    )
    out = " ".join(capsys.readouterr().out.split())  # undo the line wrapping
    assert "recorded, not run here" in out
    assert "0.17 at step 4 and 0.10 first at step 5" in out
    assert "2.55 2.28 1.77" in out


# ---------------------------------------------------------------- files and printing


def test_files_go_under_the_chapters_folder(course_home: Path) -> None:
    """Rows, adapters, evaluations and answer folders, all under COURSE_HOME."""
    home = course_home / "14-sft-real-model"
    assert lfm_files.chapter_home() == home
    assert lfm_files.rows_path("train") == home / "rows_train.jsonl"
    assert lfm_files.adapter_path("sft.safetensors") == home / "sft.safetensors"
    elsewhere = course_home / "x" / "a.safetensors"
    assert lfm_files.adapter_path(str(elsewhere)) == elsewhere
    assert lfm_files.eval_path("base") == home / "eval_base.jsonl"
    assert lfm_files.work_folder("sft", "q1") == home / "work" / "sft" / "q1"
    adapter = home / "sft.safetensors"
    assert lfm_files.metrics_path(adapter) == home / "sft.metrics.jsonl"
    assert lfm_files.eval_label(None) == "base"
    assert lfm_files.eval_label(adapter) == "sft"


def test_adapters_are_named_after_what_differs() -> None:
    """The project's settings give sft.safetensors; anything else says what changed."""
    lr, epochs = lfm_files.SFT_LEARNING_RATE, lfm_files.SFT_EPOCHS
    assert lfm_files.adapter_name(lr, epochs, None) == "sft.safetensors"
    assert lfm_files.adapter_name(1e-4, 3, None) == "sft_lr0.0001_epochs3.safetensors"
    assert lfm_files.adapter_name(lr, epochs, 40) == "sft_limit40.safetensors"


def test_no_model_names_the_commands(course_home: Path) -> None:
    """With neither build, the scripts stop and say how to get one."""
    with pytest.raises(SystemExit, match="convert.py") as stopped:
        lfm_files.pick_model(fast=False)
    assert "download.py --fast" in str(stopped.value)
    with pytest.raises(SystemExit, match="not fully downloaded"):
        lfm_files.pick_model(fast=True)


def test_ours_comes_first_then_liquids_export(course_home: Path) -> None:
    """Chapter 11's build if it is there; else, and with --fast, Liquid's export."""
    export = course_home / "models" / lfm_files.FOUR_BIT.folder
    export.mkdir(parents=True)
    for item in lfm_files.FOUR_BIT.files:
        with (export / item.name).open("wb") as handle:
            handle.truncate(item.size)  # a sparse file: the size, no real bytes
    folder, which = lfm_files.pick_model(fast=False)
    assert folder == export and "convert.py" in which
    ours = lfm_files.ours_folder()
    ours.mkdir(parents=True)
    for name in lfm_files.BUILD_FILES:
        (ours / name).write_text("x")
    assert lfm_files.pick_model(fast=False)[0] == ours
    assert lfm_files.pick_model(fast=True) == (export, "Liquid's export (--fast)")


def test_pass_at_1_prints_its_error_bar_and_forms(
    capsys: pytest.CaptureFixture,
) -> None:
    """The heading carries the score and the error bar; the forms are counted."""
    tasks = load_tasks(TEST_FILE)[:4]
    answers = [
        Answered(t.task_id, i < 3, "correct", "S", "1", "r", 10 + i, "stop", 1.0, "f")
        for i, t in enumerate(tasks)
    ]
    report.print_pass("base", answers, tasks, 4.0)
    out = capsys.readouterr().out
    assert "pass@1 3 of 4 = 75.0% ± 21.7 points" in out
    assert "the query came as: f 4" in out
    assert "0 cut off at the cap" in out


def test_a_row_prints_with_and_without_colour(capsys: pytest.CaptureFixture) -> None:
    """Plain text in a file; dim and highlight codes on a terminal."""
    tokenizer = TemplateTokenizer()
    row = rows.build_row(tokenizer, load_tasks(TRAIN_FILE)[0])
    report.print_row(row, 1, tokenizer.decode, color=False)
    plain = capsys.readouterr().out
    assert "\x1b[" not in plain
    assert f"reply (mask 1), {row.reply_tokens} tokens:" in plain
    report.print_row(row, 1, tokenizer.decode, color=True)
    assert report.HIGHLIGHT in capsys.readouterr().out


def test_the_boundary_names_the_position_each_score_comes_from(
    capsys: pytest.CaptureFixture,
) -> None:
    """The first reply token is scored at the <think> before it."""
    tokenizer = TemplateTokenizer()
    row = rows.build_row(tokenizer, load_tasks(TRAIN_FILE)[0])
    item = rows.item_from_sequence(row.as_sequence())
    first_too = [1, *row.loss_mask[1:]]
    forced = rows.item_from_sequence({**row.as_sequence(), "loss_mask": first_too})
    assert item is not None and forced is not None
    assert forced.positions == item.positions
    report.print_boundary(row, tokenizer.decode, item, forced)
    out = " ".join(capsys.readouterr().out.split())  # undo the line wrapping
    start = row.prompt_length
    assert f"position {start - 1} ('<think>')" in out
    assert "nothing comes before position 0" in out
    assert f"keeps {row.reply_tokens}, from position {start}" in out


def test_the_template_check_handles_a_rendering_that_only_adds(
    capsys: pytest.CaptureFixture,
) -> None:
    """A template output that is the row plus more is said so, without a crash."""
    tokenizer = TemplateTokenizer()
    row = rows.build_row(tokenizer, load_tasks(TRAIN_FILE)[0])
    longer = [*row.input_ids, 110]
    report.print_template_check(row, longer, len(row.input_ids), tokenizer.decode)
    out = " ".join(capsys.readouterr().out.split())
    assert "the template's goes on past the end of the other" in out
    report.print_template_check(row, list(row.input_ids), 0, tokenizer.decode)
    assert "exactly this row" in capsys.readouterr().out


def test_the_unrelated_prompts_summary_follows_the_numbers(
    capsys: pytest.CaptureFixture,
) -> None:
    """The closing line says lower or higher, from the scores themselves."""
    common = {"base_tokens": 3, "base_finish": "stop", "tuned_tokens": 3}
    up = report.Unrelated(
        prompt="p",
        base="a</think>A haiku",
        tuned="b</think>SELECT 1",
        tuned_finish="stop",
        base_score=-1.0,
        tuned_score=-0.5,
        **common,
    )
    report.print_unrelated([up])
    out = " ".join(capsys.readouterr().out.split())
    assert "1 of 1 replies contain SELECT" in out
    assert "0.5000 higher in log-probability per token" in out


def test_a_dry_run_that_never_reaches_the_threshold_says_so(
    capsys: pytest.CaptureFixture,
) -> None:
    """No step at 0.10 or below: 'at no step', never 'step None'."""
    steps = [project_run.RecordedStep(i, 1.0, 10, 1.0, 3.0) for i in range(1, 7)]
    report.print_project_runs(steps, steps)
    out = " ".join(capsys.readouterr().out.split())
    assert "0.10 at no step" in out
    assert "None" not in out


def test_the_fixed_reason_is_the_projects_teacher_line() -> None:
    """--reason fixed: one sentence for every row; an unknown style is refused."""
    item = load_tasks(TRAIN_FILE)[0]
    fixed = rows.reply_text(item, rows.REASON_FIXED)
    assert fixed == (
        "Now I can answer the question with one query.</think>"
        + rows.one_line(item.gold_sql)
    )
    assert rows.reply_text(item).startswith(rows.reason_for(item.gold_sql))
    with pytest.raises(ValueError, match="unknown reason style"):
        rows.reply_text(item, "poem")
    tokenizer = TemplateTokenizer()
    steps = rows.build_row(tokenizer, item)
    short_row = rows.build_row(tokenizer, item, style=rows.REASON_FIXED)
    assert short_row.prompt_length == steps.prompt_length
    assert (
        short_row.input_ids[: short_row.prompt_length]
        == steps.input_ids[: steps.prompt_length]
    )


def test_a_fixed_reason_run_gets_its_own_files(course_home: Path) -> None:
    """Rows and adapters trained with --reason fixed never overwrite the default's."""
    home = course_home / "14-sft-real-model"
    assert lfm_files.rows_path("train", "fixed") == home / "rows_train_fixed.jsonl"
    lr, epochs = lfm_files.SFT_LEARNING_RATE, lfm_files.SFT_EPOCHS
    name = lfm_files.adapter_name(lr, epochs, None, "fixed")
    assert name == "sft_reasonfixed.safetensors"
