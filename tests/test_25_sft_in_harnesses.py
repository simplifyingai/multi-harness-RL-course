"""Chapter 25 without a model: the teacher, its engine, the collection, the check.

The unit tests check the answer-key teacher against the project's rules
(the four paths, the save before the finish, mini-swe-agent's marker chained
first, LFM2.5's call syntax), the engine's ``--oracle`` mode (the teacher's
text cut into tokens, every log-probability 0, its own text put back on the
next call) and its adapter switch, collect.py's records and its mask display
(on two real rows captured through mini-swe-agent, ``tests/data/
25_captured_demos.jsonl``), and quick_eval.py's arithmetic on the project's
bundled records. The end-to-end test runs collect.py as a learner does: the
teacher's engine over real HTTP, OpenEnv's real capture proxy started by
``start.sh``, and both training harnesses in the sandbox. It is skipped when
the proxy or the harnesses are not on disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("25-sft-in-harnesses")

import baseline_eval  # noqa: E402
import collect  # noqa: E402
import engine  # noqa: E402
import evaluate  # noqa: E402
import install  # noqa: E402
import lfm_files  # noqa: E402
import quick_eval  # noqa: E402
import scripted_engine  # noqa: E402
import tasks  # noqa: E402
import teacher  # noqa: E402
import toolcalls  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from noise import project_checkpoint  # noqa: E402

DATA = Path(__file__).parent / "data"
CAPTURED = DATA / "25_captured_demos.jsonl"
CHAPTER_24 = ROOT / "24-measuring"
CHAPTER_14 = ROOT / "14-sft-real-model"
START_SH = CHAPTER / "start.sh"
TRAIN = tasks.load_tasks(tasks.TRAIN_FILE)
TEST = tasks.load_tasks()
BY_ID = {task.task_id: task for task in TRAIN}
HARNESSES = ["mini-swe-agent", "claude-code", "pi"]
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy, the harnesses and the tokenizer are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
CHAPTERS_WITH_STACK = ("25-sft-in-harnesses", "24-measuring", "23-three-harnesses")
START_TIMEOUT_S = 300
EVAL_FOLDER_LINE = (
    'EVAL_FOLDER = "eval"  # COURSE_HOME/25-sft-in-harnesses/runs/eval/<label>/'
    "<harness>/<task>\n"
)
EVAL_FOLDER_LINES = (
    "# COURSE_HOME/25-sft-in-harnesses/runs/eval/<label>/<harness>/<task>\n"
    'EVAL_FOLDER = "eval"\n'
)
# The modules this chapter copies from chapter 24 unchanged but for its name.
COPIED = (
    "baseline_eval.py",
    "capture_client.py",
    "evaluate.py",
    "fetch_model.py",
    "grader.py",
    "install.py",
    "lfm_files.py",
    "lfm_template.py",
    "lfm_text.py",
    "loopback.py",
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
    "stats.py",
    "toolcalls.py",
    "wire.py",
)


def task_with_path(path: str) -> tasks.Task:
    """The first training task the teacher walks a given path on.

    Args:
        path: One of the teacher's four paths.

    Returns:
        The task.
    """
    return next(task for task in TRAIN if teacher.pattern_of(task) == path)


def tool_calls(reply: str, tools: list[dict[str, Any]]) -> list[toolcalls.ToolCall]:
    """Parse a reply's tool calls as the engine does.

    Args:
        reply: The teacher's reply.
        tools: The tools offered.

    Returns:
        The calls.
    """
    return toolcalls.split_reply(reply, tools).calls


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


def captured() -> list[dict[str, Any]]:
    """The two real rows captured through mini-swe-agent (no home paths in them).

    Returns:
        Their records.
    """
    return [json.loads(line) for line in CAPTURED.read_text().splitlines()]


# ------------------------------------------------------------------ the teacher


def test_the_teacher_demonstrates_training_tasks_only() -> None:
    answer_key = teacher.AnswerKeyTeacher.from_train_file()
    held_out = [{"role": "user", "content": TEST[0].question}]
    assert answer_key.find_task(held_out) is None
    assert answer_key(held_out, teacher.CLAUDE_TOOLS) == teacher.FALLBACK_REPLY
    asked = [{"role": "user", "content": f"Please: {TRAIN[5].question}"}]
    assert answer_key.find_task(asked) == TRAIN[5]


def test_a_question_in_an_earlier_reply_does_not_count() -> None:
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    only_in_reply = [{"role": "assistant", "content": TRAIN[0].question}]
    assert answer_key.find_task(only_in_reply) is None


def test_the_paths_vary_over_the_training_tasks_and_are_fixed_per_task() -> None:
    counts = {name: 0 for name in teacher.PATTERNS}
    for task in TRAIN:
        counts[teacher.pattern_of(task)] += 1
    assert counts == {"direct": 43, "peek": 45, "tables": 47, "verify": 45}
    assert teacher.pattern_of(BY_ID["library-easy-001"]) == "tables"


@pytest.mark.parametrize(
    ("path", "commands"),
    [("direct", 3), ("peek", 4), ("tables", 4), ("verify", 4)],
)
def test_every_path_saves_the_answer_before_it_finishes(
    path: str, commands: int
) -> None:
    task = task_with_path(path)
    steps = teacher.AnswerKeyTeacher.steps(task)
    assert len(steps) == commands
    shell = [command for _, command in steps]
    save = f'echo "{teacher.shell_quote_double(task.answer)}" > answer.txt'
    assert save in shell
    assert " ".join(task.gold_sql.split()) in " ".join(shell)
    if path == "verify":
        assert shell[-1] == "cat answer.txt" and shell[-2] == save
    else:
        assert shell[-1] == save


def test_claude_code_gets_one_native_bash_call_per_reply_then_an_answer() -> None:
    task = BY_ID["library-easy-001"]
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    replies = teacher.demonstration(
        answer_key, task, teacher.CLAUDE_TOOLS, "You are Claude Code."
    )
    assert len(replies) == len(answer_key.steps(task)) + 1
    for reply, (reason, command) in zip(replies, answer_key.steps(task), strict=False):
        assert reply.startswith(reason + teacher.THINK_CLOSE)
        (call,) = tool_calls(reply, teacher.CLAUDE_TOOLS)
        assert call.name == "Bash" and call.arguments == {"command": command}
    assert tool_calls(replies[-1], teacher.CLAUDE_TOOLS) == []
    assert replies[-1].endswith(f"answer.txt: {task.answer}")


def test_mini_swe_agent_finishes_with_its_marker_chained_first() -> None:
    task = task_with_path("direct")
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    replies = teacher.demonstration(
        answer_key, task, teacher.MINI_TOOLS, teacher.MINI_PROMPT
    )
    assert len(replies) == 3  # no extra reply: the marker ends the run
    (last,) = tool_calls(replies[-1], teacher.MINI_TOOLS)
    marker = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
    assert last.arguments["command"].startswith(marker)
    assert last.arguments["command"].endswith("> answer.txt")
    for reply in replies[:-1]:
        assert marker not in reply


def test_quotes_dollars_and_backslashes_survive_the_round_trip() -> None:
    task = replace(
        TRAIN[0],
        task_id="shop-easy-999",
        answer='O\'Brien "$5" \\ `x`',
        gold_sql="SELECT name FROM t WHERE note = 'it''s'",
    )
    answer_key = teacher.AnswerKeyTeacher([task])
    for step in range(len(answer_key.steps(task))):
        reply = answer_key.tool_reply(task, step, teacher.CLAUDE_TOOLS, None)
        (call,) = tool_calls(reply, teacher.CLAUDE_TOOLS)
        assert call.arguments["command"] == answer_key.steps(task)[step][1]
    save = answer_key.steps(task)[-1][1]
    assert save == 'echo "O\'Brien \\"\\$5\\" \\\\ \\`x\\`" > answer.txt'


def test_a_required_free_text_argument_gets_the_reason() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "parameters": {
                    "properties": {"cmd": {}, "why": {}},
                    "required": ["cmd", "why"],
                },
            },
        }
    ]
    name, arguments = teacher.AnswerKeyTeacher.shell_call(tools, "ls", "Look.")
    assert name == "run_shell_command"
    assert arguments == {"cmd": "ls", "why": "Look."}


def test_a_harness_without_a_shell_tool_gets_the_answer_in_words() -> None:
    task = TRAIN[0]
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    reply = answer_key.tool_reply(task, 0, tools, None)
    assert reply.endswith(f"The answer is {task.answer}.")
    assert teacher.TOOL_CALL_START not in reply


def test_a_text_action_harness_gets_one_bash_block_per_reply() -> None:
    task = task_with_path("direct")
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    messages = [
        {"role": "system", "content": teacher.MINI_PROMPT},
        {"role": "user", "content": task.question},
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    reply = answer_key(messages, None)
    assert reply.count("```bash") == 1
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo" in reply


def test_the_call_syntax_is_lfm_s_own() -> None:
    assert (
        teacher.format_call("Bash", {"command": "it's\nhere", "n": 2, "x": [1]})
        == "<|tool_call_start|>[Bash(command='it\\'s\\nhere', n=2, x=[1])]"
        "<|tool_call_end|>"
    )


def test_teacher_main_prints_one_demonstration_in_both_formats(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = run_main(teacher, ["teacher.py"], monkeypatch, capsys)
    assert "teacher.py · the answer-key teacher, on library-easy-001" in out
    assert "path      tables (4 commands)" in out
    assert "direct 43, peek 45, tables 47, verify 45" in out
    assert "[5] The answer is saved.</think>I saved the answer to answer.txt: 30" in out
    assert 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo "30" > answer.txt' in out
    with pytest.raises(SystemExit, match="no training task"):
        run_main(
            teacher, ["teacher.py", "--task", TEST[0].task_id], monkeypatch, capsys
        )


# ------------------------------------------------------------------ the engine


def oracle_app() -> tuple[TestClient, engine.Engine, scripted_engine.ScriptedCutter]:
    """The engine in --oracle mode, on the scripted tokenizer (no files).

    Returns:
        A client for its app, the engine and the tokenizer.
    """
    cutter = scripted_engine.ScriptedCutter()
    core = engine.Engine(engine.OracleModel(cutter, teacher.AnswerKeyTeacher(TRAIN)))
    return TestClient(engine.create_app(core)), core, cutter


def chat(messages: list[dict[str, Any]], tools: list | None) -> dict[str, Any]:
    """A chat request body.

    Args:
        messages: The conversation.
        tools: The tools offered.

    Returns:
        The body.
    """
    body: dict[str, Any] = {"model": "x", "messages": messages, "temperature": 1.0}
    if tools:
        body["tools"] = tools
    return body


def test_the_oracle_answers_with_the_teachers_tokens_every_one_certain() -> None:
    client, core, cutter = oracle_app()
    task = BY_ID["library-easy-001"]
    messages = [{"role": "user", "content": tasks.task_instruction(task)}]
    answer = client.post(
        "/v1/chat/completions", json=chat(messages, teacher.CLAUDE_TOOLS)
    )
    assert answer.status_code == 200
    data = answer.json()
    choice = data["choices"][0]
    text = teacher.AnswerKeyTeacher(TRAIN)(messages, teacher.CLAUDE_TOOLS)
    assert choice["token_ids"] == cutter.encode(text) + [scripted_engine.EOS]
    assert {e["logprob"] for e in choice["logprobs"]["content"]} == {0.0}
    assert all(e["top_logprobs"] == [] for e in choice["logprobs"]["content"])
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "Bash"
    assert json.loads(call["arguments"]) == {"command": "sqlite3 data.db .tables"}
    assert data["prompt_token_ids"] == cutter.encode(
        cutter.render(messages, teacher.CLAUDE_TOOLS)
    )
    assert core.model.answered == 1


def test_the_next_call_continues_the_teachers_own_text() -> None:
    client, core, cutter = oracle_app()
    task = BY_ID["library-easy-001"]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": tasks.task_instruction(task)}
    ]
    first = client.post(
        "/v1/chat/completions", json=chat(messages, teacher.CLAUDE_TOOLS)
    )
    reply = first.json()["choices"][0]["message"]
    messages += [
        reply,
        {
            "role": "tool",
            "tool_call_id": reply["tool_calls"][0]["id"],
            "content": "books\nloans\nmembers",
        },
    ]
    second = client.post(
        "/v1/chat/completions", json=chat(messages, teacher.CLAUDE_TOOLS)
    )
    data = second.json()
    raw = "<think>"  # the generation prompt's end: the reply follows it
    prompt = cutter.decode(data["prompt_token_ids"])
    assert raw + "First, which tables are in this database?</think>" in prompt
    assert core.stats.restored == 1 and core.stats.not_restored == 0
    call = data["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(call["arguments"])["command"] == 'sqlite3 data.db ".schema books"'


def test_a_call_with_no_task_gets_the_fallback_reply() -> None:
    client, _, _ = oracle_app()
    answer = client.post(
        "/v1/chat/completions", json=chat([{"role": "user", "content": "hi"}], None)
    )
    assert answer.json()["choices"][0]["message"]["content"] == "OK."


def test_a_reply_longer_than_the_cap_is_cut_and_says_length() -> None:
    client, _, _ = oracle_app()
    task = BY_ID["library-easy-001"]
    body = chat([{"role": "user", "content": task.question}], teacher.CLAUDE_TOOLS)
    body["max_tokens"] = 5
    choice = client.post("/v1/chat/completions", json=body).json()["choices"][0]
    assert len(choice["token_ids"]) == 5 and choice["finish_reason"] == "length"


def test_the_oracle_says_what_it_serves_and_refuses_an_adapter() -> None:
    client, _, _ = oracle_app()
    health = client.get("/health").json()
    assert health["serves"] == engine.SERVES_TEACHER == collect.SERVES_TEACHER
    refused = client.post("/admin/adapter", json={"path": "/nowhere"})
    assert refused.status_code == 400
    assert "no weights to adapt" in refused.json()["error"]["message"]


class AdaptableModel(scripted_engine.ScriptedModel):
    """A scripted model that can 'switch adapters', for the engine's route."""

    serves = quick_eval.SERVES_MODEL

    def set_adapter(self, path: Path | None) -> str:
        """Pretend to load an adapter.

        Args:
            path: The file, or None.

        Returns:
            Its label.

        Raises:
            FileNotFoundError: For a path that does not exist.
        """
        if path is None:
            return "base"
        if not path.exists():
            raise FileNotFoundError(path)
        return path.name


def test_the_adapter_route_switches_and_drops_every_cache(tmp_path: Path) -> None:
    core = engine.Engine(AdaptableModel())
    client = TestClient(engine.create_app(core))
    client.post(
        "/v1/chat/completions", json=chat([{"role": "user", "content": "hi"}], None)
    )
    assert core.caches
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"x")
    switched = client.post("/admin/adapter", json={"path": str(adapter)})
    assert switched.json() == {"adapter": "adapter.safetensors"}
    health = client.get("/health").json()
    assert core.caches == [] and health["adapter"] == adapter.name
    assert health["adapter_path"] == str(adapter)  # what a caller puts back
    for wrong in ({}, {"pth": str(adapter)}, []):  # never quietly "base"
        refused = client.post("/admin/adapter", json=wrong)
        message = refused.json()["error"]["message"]
        assert refused.status_code == 400 and message == engine.ADAPTER_SHAPE
    assert client.get("/health").json()["adapter"] == adapter.name
    assert client.post("/admin/adapter", json={"path": None}).json() == {
        "adapter": "base"
    }
    missing = client.post("/admin/adapter", json={"path": str(tmp_path / "no")})
    assert missing.status_code == 400 and "no adapter loaded" in missing.text
    assert client.post("/admin/adapter", json={"path": 3}).status_code == 400
    assert client.post("/admin/adapter", content=b"not json").status_code == 400
    assert client.get("/health").json()["adapter"] == "base"
    assert client.get("/health").json()["adapter_path"] is None


def test_oracle_mode_takes_no_flag_that_needs_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["engine.py", "--oracle"])
    assert engine.parse_args().port == engine.ORACLE_PORT == 8101
    monkeypatch.setattr(sys, "argv", ["engine.py"])
    assert engine.parse_args().port == engine.DEFAULT_PORT == 8100
    for flag in (["--adapter", "a"], ["--fast"], ["--raw-logprobs"]):
        monkeypatch.setattr(sys, "argv", ["engine.py", "--oracle", *flag])
        with pytest.raises(SystemExit):
            engine.parse_args()


def test_the_oracle_path_never_imports_mlx() -> None:
    code = (
        "import sys; import engine, teacher, scripted_engine; "
        "engine.OracleModel(scripted_engine.ScriptedCutter(), "
        "teacher.AnswerKeyTeacher.from_train_file()); "
        "print(any(name.split('.')[0] == 'mlx' for name in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=CHAPTER,
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "False"


def test_collect_names_the_teacher_as_the_engine_does() -> None:
    assert collect.ORACLE_URL.endswith(f":{engine.ORACLE_PORT}")
    assert collect.SERVED_MODEL == engine.SERVED_MODEL
    assert quick_eval.SERVES_MODEL == f"{engine.SERVED_MODEL} 4-bit"


# ------------------------------------------------------------------ collecting


def test_a_row_is_cut_into_runs_of_one_mask() -> None:
    row = {"input_ids": [1, 2, 3, 4, 5, 6], "loss_mask": [0, 0, 1, 1, 0, 1]}
    found = collect.segments(row)
    assert [(s.mask, s.start, s.token_ids) for s in found] == [
        (0, 0, (1, 2)),
        (1, 2, (3, 4)),
        (0, 4, (5,)),
        (1, 5, (6,)),
    ]


def test_only_trainable_rows_are_kept_with_the_three_fields_training_reads() -> None:
    document = {
        "sequences": [
            {
                "input_ids": [1],
                "loss_mask": [0],
                "logprobs": [0.0],
                "trainable": True,
                "x": 1,
            },
            {"input_ids": [2], "loss_mask": [1], "logprobs": [0.0], "trainable": False},
        ]
    }
    assert collect.trainable_rows(document) == [
        {"input_ids": [1], "loss_mask": [0], "logprobs": [0.0]}
    ]


def test_the_captured_rows_are_usable_and_carry_no_home_folder() -> None:
    text = CAPTURED.read_text()
    assert Path.home().name not in text and "/Users/" not in text
    records = [collect.DemoRecord(**r) for r in captured()]
    assert [r.task_id for r in records] == ["library-easy-001", "library-hard-001"]
    for record in records:
        assert record.usable and record.harness == "mini-swe-agent"
        (row,) = record.sequences
        assert set(row["logprobs"]) == {0.0}  # a demonstration is certain
        assert sum(row["loss_mask"]) == record.n_trainable_tokens
        assert record.stop == "finish command" and record.n_calls == 4


def find_tokenizer() -> Path | None:
    """Find LFM2.5's checked tokenizer.json in the real course home.

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


REAL_TOKENIZER = find_tokenizer()


@pytest.mark.skipif(REAL_TOKENIZER is None, reason="needs LFM2.5's tokenizer.json")
def test_the_mask_is_1_exactly_on_the_tokens_the_teacher_wrote() -> None:
    from tokenizers import Tokenizer

    assert REAL_TOKENIZER is not None
    tokenizer = Tokenizer.from_file(str(REAL_TOKENIZER))
    answer_key = teacher.AnswerKeyTeacher(TRAIN)
    for record in captured():
        task = BY_ID[record["task_id"]]
        replies = teacher.demonstration(
            answer_key, task, teacher.MINI_TOOLS, teacher.MINI_PROMPT
        )
        (row,) = record["sequences"]
        trained = [s for s in collect.segments(row) if s.mask == 1]
        decoded = [
            tokenizer.decode(list(s.token_ids), skip_special_tokens=False)
            for s in trained
        ]
        assert decoded == [reply + "<|im_end|>" for reply in replies]
        read = [s for s in collect.segments(row) if s.mask == 0]
        first = tokenizer.decode(list(read[0].token_ids), skip_special_tokens=False)
        assert first.endswith("<|im_start|>assistant\n<think>")
        assert task.question in first


def test_a_shown_row_names_each_segments_mask_and_hides_your_home() -> None:
    record = collect.DemoRecord(**captured()[0])
    home = str(Path.home())
    lines: list[str] = []
    collect.show_row(record, lambda ids: f"{home}/x {len(ids)}\nend", lines.append)
    assert lines[0].startswith(
        "one captured mini-swe-agent demonstration: library-easy-001"
    )
    assert "mask 1 (the teacher's tokens, trained) 147" in lines[1]
    assert lines[2].startswith("  mask 0 ·")
    assert all(home not in line for line in lines)
    assert "~/x" in lines[2] and "\\nend" in lines[2]
    assert len(lines) == 2 + len(collect.segments(record.sequences[0]))


def test_a_long_segment_shows_its_start_and_its_end() -> None:
    text = "a" * 100 + "b" * 100
    assert collect.shown(text, 150) == "a" * 75 + " … " + "b" * 75
    assert collect.shown("short", 150) == "short"
    user = Path.home().name
    said = collect.scrub(f"{Path.home()}/f, -Users-{user}-x, /p/{user}/q, a{user}b")
    assert said == f"~/f, -Users-<you>-x, /p/<you>/q, a{user}b"  # paths only


def test_records_round_trip_and_a_broken_line_is_named(tmp_path: Path) -> None:
    path = tmp_path / "demos.jsonl"
    records = [collect.DemoRecord(**r) for r in captured()]
    for record in records:
        collect.append_record(path, record)
    assert collect.read_demos(path) == records
    with path.open("a") as handle:
        handle.write('{"task_id": "x"\n')
    with pytest.raises(ValueError, match="demos.jsonl:3"):
        collect.read_demos(path)


def test_a_second_run_needs_resume_and_resume_cuts_a_torn_line(tmp_path: Path) -> None:
    path = tmp_path / "demos.jsonl"
    collect.append_record(path, collect.DemoRecord(**captured()[0]))
    with pytest.raises(SystemExit, match="--resume"):
        collect.kept_demos(path, resume=False)
    with path.open("a") as handle:
        handle.write('{"task_id": "half')
    kept = collect.kept_demos(path, resume=True)
    assert [r.task_id for r in kept] == ["library-easy-001"]


def test_the_summary_says_rows_calls_and_the_trained_share() -> None:
    records = [collect.DemoRecord(**r) for r in captured()]
    wrong = replace(records[0], harness="claude-code", correct=False)
    lines = collect.summary_lines([*records, wrong], ["mini-swe-agent", "claude-code"])
    assert lines[0].startswith("  mini-swe-agent 2/2 usable · 4 calls x2 · 2 rows")
    assert lines[1] == "  claude-code    0/1 usable · no usable demonstration · no rows"


def test_a_progress_line_says_usable_calls_path_and_rows() -> None:
    record = collect.DemoRecord(**captured()[0])
    line = collect.demo_line(1, 4, record)
    assert line.startswith("[1/4] mini-swe-agent library-easy-001   usable      · 4")
    assert "path tables · 1 row, 794 tokens, 147 trained" in line
    assert "untrainable" in collect.demo_line(1, 4, replace(record, trainable=False))


def test_demos_run_in_their_own_folders_with_state_in_the_temp_folder(
    course_home: Path,
) -> None:
    task = BY_ID["library-easy-001"]
    folders = collect.prepare_demo("claude-code", task)
    try:
        assert folders.task.is_relative_to(course_home.resolve())
        assert folders.task.parent.name == task.task_id
        assert (folders.task / "data.db").is_file()
        temp = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
        assert folders.state.is_relative_to(temp)
        assert folders.state.name.startswith("mhrl-harness-claude-code-")
        assert not folders.state.is_relative_to(folders.task.parent)
    finally:
        import shutil

        shutil.rmtree(folders.state, ignore_errors=True)


def test_collect_refuses_an_engine_that_is_not_the_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Answer:
        """What urlopen gives back: a context manager with read()."""

        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True, "serves": "lfm2.5-2.6b 4-bit"}).encode()

    def fake_urlopen(url: str, timeout: float) -> Answer:
        return Answer()

    monkeypatch.setattr(collect.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit, match="not the answer-key teacher"):
        collect.teacher_health("http://127.0.0.1:1")


def test_collect_refuses_an_unknown_or_held_out_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for wrong in ("cursor", "pi", "mini-swe-agent,pi"):
        monkeypatch.setattr(sys, "argv", ["collect.py", "--harnesses", wrong])
        with pytest.raises(SystemExit):
            collect.parse_args()
    monkeypatch.setattr(sys, "argv", ["collect.py", "--fast"])
    args = collect.parse_args()
    assert args.limit == 10 and args.harnesses == ["mini-swe-agent", "claude-code"]
    assert args.concurrency == 4


def test_the_file_ends_in_job_order_whatever_order_rollouts_ended(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demos.jsonl"
    base = collect.DemoRecord(**captured()[0])
    ended = [
        ("claude-code", "b"),
        ("mini-swe-agent", "z"),
        ("mini-swe-agent", "b"),
        ("claude-code", "a"),
        ("mini-swe-agent", "a"),
    ]
    for harness, task in ended:
        collect.append_record(path, replace(base, harness=harness, task_id=task))
    order = [(h, t) for t in ("a", "b") for h in ("mini-swe-agent", "claude-code")]
    collect.put_in_order(path, order)
    found = [(r.harness, r.task_id) for r in collect.read_demos(path)]
    assert found == [*order, ("mini-swe-agent", "z")]  # the rest follows
    assert not path.with_suffix(".tmp").exists()


def test_ordering_survives_no_file_a_broken_line_and_repeated_pairs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demos.jsonl"
    collect.put_in_order(path, [("claude-code", "a")])  # every rollout failed
    assert not path.exists()
    base = collect.DemoRecord(**captured()[0])
    first = replace(base, harness="claude-code", task_id="a", session_id="one")
    second = replace(first, session_id="two")  # the same pair, run again
    for record in (replace(base, task_id="z"), first, second):
        collect.append_record(path, record)
    collect.put_in_order(path, [("claude-code", "a")])
    found = [(r.task_id, r.session_id) for r in collect.read_demos(path)]
    assert found == [("a", "one"), ("a", "two"), ("z", base.session_id)]
    with path.open("a") as handle:
        handle.write('{"task_id": "torn')
    with pytest.raises(ValueError, match="demos.jsonl:4"):
        collect.put_in_order(path, [])


def test_a_failed_rollout_is_said_and_a_dead_proxy_stops_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    good = collect.DemoRecord(**captured()[0])

    def flaky(task: tasks.Task, harness: str, setup: Any) -> collect.DemoRecord:
        if task.task_id == TRAIN[1].task_id:
            raise OSError("the harness would not start")
        if task.task_id == TRAIN[2].task_id:
            raise collect.ProxyError("cannot reach the proxy")
        return replace(good, task_id=task.task_id, harness=harness)

    monkeypatch.setattr(collect, "collect_one", flaky)
    jobs = [(task, "claude-code") for task in TRAIN[:5]]
    lines: list[str] = []
    out = tmp_path / "demos.jsonl"
    made, failed = collect.collect(jobs, None, out, 1, lines.append)  # type: ignore[arg-type]
    assert [r.task_id for r in made] == [TRAIN[0].task_id]
    reasons = {f.task_id: f.reason for f in failed}
    assert reasons[TRAIN[1].task_id] == "the harness would not start"
    assert reasons[TRAIN[2].task_id] == "cannot reach the proxy"
    for task in TRAIN[3:5]:
        assert reasons[task.task_id] == "not run: the proxy failed earlier"
    assert sum(line.startswith("FAILED claude-code") for line in lines) == 4
    assert len(collect.read_demos(out)) == 1  # only what ran is on disk


# ------------------------------------------------------------------ quick_eval


def test_calls_per_solve_averaged_over_harnesses_is_the_projects_7_1_to_4_8() -> None:
    base, sft = (list(project_checkpoint(label).records) for label in ("base", "sft"))
    old = quick_eval.average_calls(evaluate.summarize(base, HARNESSES), HARNESSES)
    new = quick_eval.average_calls(evaluate.summarize(sft, HARNESSES), HARNESSES)
    assert (round(old, 1), round(new, 1)) == (7.1, 4.8)
    assert quick_eval.both_solved(base, sft, "mini-swe-agent")[0] == 18


def test_the_report_compares_cell_by_cell_and_boxes_the_projects_row() -> None:
    base, sft = (list(project_checkpoint(label).records) for label in ("base", "sft"))
    lines: list[str] = []
    quick_eval.report(base, sft, HARNESSES, lines.append)
    text = "\n".join(lines)
    assert "base -> sft  mini-swe-agent     24      24     18       0     6" in text
    assert "lost: beyond noise" in text
    assert "project base -> project sft" in text
    assert "│ sft pass@1 (mini-swe-agent / claude-code / pi): 75% / 71% / 38%" in text
    assert "base 7.1 -> sft 4.8" in text


def test_a_harness_nobody_solved_twice_is_said_so() -> None:
    record = evaluate.Record(task_id="t", harness="pi", correct=True, n_calls=3)
    lines = quick_eval.calls_lines([record], [replace(record, correct=False)], ["pi"])
    assert lines[1] == "  pi              no cell solved by both"


def test_quick_eval_refuses_the_teacher_and_checks_each_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: dict[str, Any] = {"/health": {"serves": engine.SERVES_TEACHER}}
    monkeypatch.setattr(
        quick_eval, "engine_call", lambda url, path, body=None: answers[path]
    )
    with pytest.raises(SystemExit, match="not the model"):
        quick_eval.check_engine("http://e")
    answers["/admin/adapter"] = {"adapter": "~/a.safetensors"}
    answers["/health"] = {"serves": quick_eval.SERVES_MODEL, "adapter": "base"}
    with pytest.raises(SystemExit, match="says base"):
        quick_eval.use_adapter("http://e", Path("a.safetensors"))


def test_resume_refuses_records_another_adapter_made(tmp_path: Path) -> None:
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    kept = [evaluate.Record(task_id="t", harness="pi", engine="sft")]
    quick_eval.check_fingerprint("sft", first, [])
    quick_eval.check_fingerprint("sft", first, kept)  # the same adapter: fine
    with pytest.raises(SystemExit, match="was made with"):
        quick_eval.check_fingerprint("sft", second, kept)
    quick_eval.records_path("sft").with_name("eval_sft.adapter.json").unlink()
    with pytest.raises(SystemExit, match="an adapter nothing names"):
        quick_eval.check_fingerprint("sft", first, kept)


def test_a_failed_run_puts_the_engines_adapter_back_and_gaps_are_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, course_home: Path
) -> None:
    adapter = course_home / "25-sft-in-harnesses" / "sft" / "adapter.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"x")
    switches: list[Path | None] = []

    class Health:
        def health(self) -> dict[str, str]:
            return {"upstream": "http://engine"}

    monkeypatch.setattr(quick_eval, "CaptureClient", lambda url: Health())
    monkeypatch.setattr(quick_eval, "check_engine", lambda url: None)
    monkeypatch.setattr(quick_eval, "check_harness", lambda *a: "pinned")
    monkeypatch.setattr(quick_eval, "starting_adapter", lambda url: Path("/s.st"))
    monkeypatch.setattr(
        quick_eval, "use_adapter", lambda url, path: switches.append(path) or "x"
    )

    def broken(label: str, *args: Any) -> list[evaluate.Record]:
        raise RuntimeError("a harness crashed")

    monkeypatch.setattr(quick_eval, "run_checkpoint", broken)
    with pytest.raises(RuntimeError, match="a harness crashed"):
        run_main(quick_eval, ["quick_eval.py", "--fast"], monkeypatch, capsys)
    assert switches == [Path("/s.st")]  # put back what it found

    def partial(label: str, *args: Any) -> list[evaluate.Record]:
        cell = evaluate.Record(task_id=TEST[0].task_id, harness="pi")
        return [cell] if label == "base" else []

    monkeypatch.setattr(quick_eval, "run_checkpoint", partial)
    with pytest.raises(SystemExit, match="must cover the same cells"):
        run_main(quick_eval, ["quick_eval.py", "--fast"], monkeypatch, capsys)
    assert "sft has no record of 12 cells" in capsys.readouterr().out


def test_a_second_adapter_gets_its_own_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["quick_eval.py", "--label", "sft-nbc"])
    assert quick_eval.parse_args().label == "sft-nbc"
    for wrong in ("base", "a/b", ""):
        monkeypatch.setattr(sys, "argv", ["quick_eval.py", "--label", wrong])
        with pytest.raises(SystemExit):
            quick_eval.parse_args()
    base, sft = (list(project_checkpoint(label).records) for label in ("base", "sft"))
    lines: list[str] = []
    quick_eval.report(base, sft, HARNESSES, lines.append, label="sft-nbc")
    text = "\n".join(lines)
    assert "base -> sft-nbc" in text and "base 7.1 -> sft-nbc 4.8" in text


def test_the_training_note_says_how_the_adapter_was_trained(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.safetensors"
    assert "which build trained it is unknown" in quick_eval.training_note(adapter)
    note = {"model": "~/m", "which": "ours", "rows": 120, "steps": 15, "lr": 5e-5}
    (tmp_path / "adapter.json").write_text(
        json.dumps({**note, "bias_correction": False})
    )
    assert quick_eval.training_note(adapter) == (
        "trained on ~/m (ours) · 120 rows · 15 steps at lr 5e-05, bias correction off"
    )


def test_quick_eval_needs_the_adapter_first(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    with pytest.raises(SystemExit, match="run train_sft.py first"):
        run_main(quick_eval, ["quick_eval.py", "--fast"], monkeypatch, capsys)


# ------------------------------------------------------------------ copies and data


def test_the_copied_modules_are_chapter_24s_but_for_the_folder_name() -> None:
    for name in COPIED:
        theirs = (CHAPTER_24 / name).read_text(encoding="utf-8")
        theirs = theirs.replace("24-measuring", "25-sft-in-harnesses")
        # The longer folder name pushed one comment past 88 columns: it moved
        # to the line above its constant.
        theirs = theirs.replace(EVAL_FOLDER_LINE, EVAL_FOLDER_LINES)
        ours = (CHAPTER / name).read_text(encoding="utf-8")
        assert ours == theirs, name
    for found in (CHAPTER_24 / "data" / "project_records").iterdir():
        mine = CHAPTER / "data" / "project_records" / found.name
        assert mine.read_bytes() == found.read_bytes()


def test_the_task_files_and_databases_are_the_projects() -> None:
    for name in (
        "train.jsonl",
        "test.jsonl",
        *(f"dbs/{d}.db" for d in ("airline", "clinic", "library", "school", "shop")),
    ):
        assert (CHAPTER / "data" / name).read_bytes() == (
            CHAPTER_14 / "data" / name
        ).read_bytes()
    assert len(TRAIN) == 180 and len(TEST) == 50
    assert {t.domain for t in TRAIN}.isdisjoint({t.domain for t in TEST})


def test_project_run_and_lora_are_chapter_14s_plus_the_adapter_check() -> None:
    assert (CHAPTER / "project_run.py").read_text() == (
        CHAPTER_14 / "project_run.py"
    ).read_text()
    for name in ("sft_metrics.jsonl", "dry_sft_metrics.jsonl"):
        path = Path("data") / "project_run" / name
        assert (CHAPTER / path).read_bytes() == (CHAPTER_14 / path).read_bytes()
    ours = set((CHAPTER / "lora.py").read_text().splitlines())
    theirs = (CHAPTER_14 / "lora.py").read_text()
    body = theirs[theirs.index("from __future__") :].splitlines()
    assert [line for line in body if line not in ours] == []


def test_start_sh_adds_the_teacher_beside_chapter_24s_stack() -> None:
    ours = START_SH.read_text(encoding="utf-8")
    theirs = (CHAPTER_24 / "start.sh").read_text(encoding="utf-8")
    for function in re.findall(r"^(\w+)\(\) \{", theirs, re.MULTILINE):
        assert f"{function}() {{" in ours, function
    assert "ORACLE_PORT=${ORACLE_PORT:-8101}" in ours
    assert 'exec uv run engine.py --oracle --port "$ORACLE_PORT"' in ours
    assert "fetch_inputs oracle" in ours and 'echo $! >"$RUN/oracle.pid"' in ours
    assert "pkill" not in ours and "killall" not in ours  # by pid file only
    assert subprocess.run(["bash", "-n", str(START_SH)], check=False).returncode == 0


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
    """Find a proxy environment at the pinned commit, here or in an earlier chapter.

    Returns:
        The environment, or None.
    """
    for chapter in CHAPTERS_WITH_STACK:
        env = _HOME / chapter / "proxy-env"
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def find_installs() -> Path | None:
    """Find a folder holding both training harnesses at their pins.

    Returns:
        The folder, or None.
    """
    for chapter in CHAPTERS_WITH_STACK:
        folder = _HOME / chapter
        if all(
            (folder / install.PINS[h].folder).is_dir() for h in collect.TRAIN_HARNESSES
        ):
            return folder
    return None


REAL_PROXY_ENV = find_proxy_env()
REAL_INSTALLS = find_installs()


@contextlib.contextmanager
def live_teacher() -> Iterator[int]:
    """Serve the teacher's engine over real HTTP, on a free port.

    Yields:
        Its port.
    """
    if REAL_TOKENIZER is not None:
        cutter: Any = scripted_engine.LfmCutter(REAL_TOKENIZER)
    else:
        cutter = scripted_engine.ScriptedCutter()
    model = engine.OracleModel(cutter, teacher.AnswerKeyTeacher(TRAIN))
    sock = engine.claim_port(0)
    server = uvicorn.Server(
        uvicorn.Config(engine.create_app(engine.Engine(model)), log_level="warning")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "the teacher's engine did not start within 10 s"
    try:
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10)


def start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``proxy`` or ``stop``.
        engine_port: ENGINE_PORT (the teacher's, here).
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "ORACLE_PORT": str(engine_port),
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


@pytest.mark.skipif(
    REAL_PROXY_ENV is None or REAL_INSTALLS is None,
    reason="needs the capture proxy (./start.sh install) and both training "
    "harnesses (uv run install.py)",
)
def test_collect_runs_the_teacher_through_the_real_proxy_and_harnesses(
    course_home: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    chapter = course_home / "25-sft-in-harnesses"
    chapter.mkdir(parents=True)
    assert REAL_INSTALLS is not None
    for name in collect.TRAIN_HARNESSES:
        folder = install.PINS[name].folder
        (chapter / folder).symlink_to(REAL_INSTALLS / folder)
    if REAL_TOKENIZER is not None:  # collect.py decodes its row with it
        models = course_home / lfm_files.MODELS_FOLDER / REAL_TOKENIZER.parent.name
        models.mkdir(parents=True)
        (models / REAL_TOKENIZER.name).symlink_to(REAL_TOKENIZER)
    stack = tmp_path_factory.mktemp("stack")
    with live_teacher() as port:
        sock = engine.claim_port(0)
        proxy_port = sock.getsockname()[1]
        sock.close()
        started = start_sh("proxy", port, proxy_port, stack)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            argv = [
                "collect.py",
                "--limit",
                "1",
                "--proxy-url",
                f"http://127.0.0.1:{proxy_port}",
                "--oracle-url",
                f"http://127.0.0.1:{port}",
            ]
            out = run_main(collect, argv, monkeypatch, capsys)
        finally:
            start_sh("stop", port, proxy_port, stack)
    assert "[demos] 2/2 demonstrations usable" in out
    if REAL_TOKENIZER is not None:
        assert "one captured claude-code demonstration: library-easy-001" in out
        first = r"  mask 1 · +\d+ tokens · First, which tables are in this database\?"
        assert re.search(first, out)
    else:
        assert "LFM2.5's tokenizer is not on disk, so no row is decoded" in out
    assert "network   every rollout: lsof saw only this machine's addresses" in out
    records = collect.read_demos(collect.records_path())
    assert {(r.harness, r.task_id) for r in records} == {
        ("mini-swe-agent", "library-easy-001"),
        ("claude-code", "library-easy-001"),
    }
    by_harness = {r.harness: r for r in records}
    assert by_harness["mini-swe-agent"].stop == "finish command"
    assert by_harness["claude-code"].n_calls == 5  # 4 commands, then the answer
    for record in records:
        assert record.usable and record.engine == "answer-key-teacher"
        (row,) = record.sequences
        assert set(row["logprobs"]) == {0.0}
        assert 0 < sum(row["loss_mask"]) < len(row["input_ids"])
    assert str(Path.home()) not in out
    assert asdict(records[0])["purpose"] == "train"
    assert baseline_eval.check_harness is collect.check_harness
