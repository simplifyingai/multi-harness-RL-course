"""Chapter 16 without Claude Code: the stand-in, the readers and the run script.

No test here runs Claude Code, opens a network connection or loads a model.
The stand-in is driven through FastAPI's test client; the request log, the
stream and the mishap readers are fed small records shaped like the ones
Claude Code 2.1.290 produced in the run log; token counts use a fake counter.
"""

from __future__ import annotations

import filecmp
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("16-inside-claude-code")

import grader  # noqa: E402
import inspect_calls  # noqa: E402
import lfm_tokens  # noqa: E402
import mishaps  # noqa: E402
import notes_mcp  # noqa: E402
import records  # noqa: E402
import show_stream  # noqa: E402
import stand_in  # noqa: E402
import tasks  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

BASH_TOOL = {
    "name": "Bash",
    "description": "Executes a bash command and returns its output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
    },
}
QUESTION = tasks.task_instruction(tasks.load_task())
SYSTEM = [{"type": "text", "text": "You are a Claude agent."}]


def request_body(
    replies: int = 0, tools: bool = True, stream: bool = True
) -> dict[str, Any]:
    """Build a request like Claude Code's, after some scripted replies.

    Args:
        replies: How many assistant replies the conversation already holds.
        tools: Whether the request offers the Bash tool.
        stream: Whether it asks for a stream.

    Returns:
        The request body.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": QUESTION},
        {"role": "system", "content": "# Environment"},
    ]
    for number in range(replies):
        tool_id = f"toolu_{number}"
        call = {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}
        messages.append({"role": "assistant", "content": [call]})
        result = {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
        messages.append({"role": "user", "content": [result]})
    body: dict[str, Any] = {
        "model": "stand-in",
        "system": SYSTEM,
        "messages": messages,
        "stream": stream,
        "max_tokens": 32000,
    }
    if tools:
        body["tools"] = [BASH_TOOL]
    return body


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Split a server-sent-event stream into (event, data) pairs.

    Args:
        text: The response body.

    Returns:
        The events in order.
    """
    events = []
    for chunk in text.strip().split("\n\n"):
        name, data = chunk.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data[len("data: ") :])))
    return events


def log_records(path: Path) -> list[dict[str, Any]]:
    """Read the stand-in's calls.jsonl.

    Args:
        path: The log.

    Returns:
        Its records.
    """
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    """Return where a test's stand-in writes its request log."""
    return tmp_path / "calls.jsonl"


def client_for(script: str, log_path: Path) -> TestClient:
    """Start a stand-in with one script, in-process.

    Args:
        script: The script's name.
        log_path: Its calls.jsonl.

    Returns:
        A test client.
    """
    return TestClient(stand_in.create_app(script, log_path))


# --- the stand-in ---------------------------------------------------------


def test_every_script_ends_with_a_reply_that_calls_no_tool() -> None:
    for name, script in stand_in.SCRIPTS.items():
        assert not script[-1].tool, name
        assert all(step.tool or step.text for step in script), name


def test_the_step_is_picked_by_the_replies_so_far() -> None:
    script = stand_in.SCRIPTS["solve"]
    assert stand_in.pick_step(script, request_body(0)).command == stand_in.TABLES
    assert stand_in.pick_step(script, request_body(2)).command == stand_in.WRITE
    assert stand_in.pick_step(script, request_body(9)) == script[-1]


def test_a_request_without_tools_is_a_side_call() -> None:
    assert stand_in.is_side_call(request_body(tools=False))
    assert not stand_in.is_side_call(request_body())


def test_a_streamed_tool_call_arrives_as_the_api_s_events(log_path: Path) -> None:
    response = client_for("solve", log_path).post("/v1/messages", json=request_body())
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    block = events[1][1]["content_block"]
    assert (block["type"], block["name"], block["input"]) == ("tool_use", "Bash", {})
    arguments = json.loads(events[2][1]["delta"]["partial_json"])
    assert arguments == {"command": stand_in.TABLES}
    assert events[4][1]["delta"]["stop_reason"] == "tool_use"
    (record,) = log_records(log_path)
    assert record["reply"]["kind"] == "loop" and record["reply"]["streamed"]
    assert record["body"]["tools"][0]["name"] == "Bash"


def test_the_last_step_is_plain_text_that_ends_the_turn(log_path: Path) -> None:
    body = request_body(replies=3)
    response = client_for("solve", log_path).post("/v1/messages", json=body)
    events = parse_sse(response.text)
    assert events[2][1]["delta"] == {"type": "text_delta", "text": stand_in.DONE}
    assert events[4][1]["delta"]["stop_reason"] == "end_turn"


def test_the_not_streamed_script_answers_with_one_json_body(log_path: Path) -> None:
    client = client_for("not-streamed", log_path)
    response = client.post("/v1/messages", json=request_body())
    assert response.headers["content-type"] == "application/json"
    message = response.json()
    assert message["content"][0]["input"] == {"command": stand_in.TABLES}
    assert message["usage"]["input_tokens"] > 0
    assert log_records(log_path)[0]["reply"]["streamed"] is False


def test_a_side_call_gets_the_fixed_text(log_path: Path) -> None:
    client = client_for("solve", log_path)
    response = client.post("/v1/messages", json=request_body(tools=False, stream=False))
    assert response.json()["content"] == [{"type": "text", "text": "OK"}]
    assert log_records(log_path)[0]["reply"]["kind"] == "side"


def test_the_lowercase_script_names_a_tool_claude_code_lacks(log_path: Path) -> None:
    client = client_for("lowercase-bash", log_path)
    message = client.post("/v1/messages", json=request_body(stream=False)).json()
    assert message["content"][0]["name"] == "bash"


def test_count_tokens_answers_and_is_logged(log_path: Path) -> None:
    client = client_for("solve", log_path)
    answer = client.post("/v1/messages/count_tokens", json=request_body()).json()
    assert answer["input_tokens"] > 0
    assert log_records(log_path)[0]["reply"]["kind"] == "count_tokens"


def test_other_routes_are_logged_and_refused_and_health_is_not(log_path: Path) -> None:
    client = client_for("solve", log_path)
    assert client.get("/health").json()["status"] == "ok"
    assert not log_path.exists()
    response = client.get("/v1/models")
    assert response.status_code == stand_in.NOT_FOUND
    assert log_records(log_path)[0]["reply"]["kind"] == "unknown route"


def test_credentials_are_hidden_and_the_dummy_key_is_shown(log_path: Path) -> None:
    client = client_for("solve", log_path)
    headers = {"Authorization": "Bearer real-secret", "x-api-key": "any-string"}
    client.post("/v1/messages", json=request_body(stream=False), headers=headers)
    logged = log_records(log_path)[0]["headers"]
    assert logged["authorization"] == stand_in.HIDDEN
    assert logged["x-api-key"] == "any-string"
    assert "real-secret" not in log_path.read_text()


def test_a_real_looking_anthropic_key_is_never_written_down() -> None:
    real = stand_in.REAL_KEY_PREFIX + "abc123"
    assert stand_in.safe_header("X-Api-Key", real) == stand_in.HIDDEN
    assert stand_in.safe_header("x-api-key", "any-string-works") == "any-string-works"


def test_show_prints_a_request_and_the_events_for_it() -> None:
    shown = stand_in.show_first_reply("solve")
    assert shown.startswith("POST /v1/messages\n")
    assert "event: message_start" in shown and shown.endswith("\n\n")
    assert stand_in.TABLES.replace('"', '\\"') in shown


def test_calls_are_numbered_and_message_ids_differ(log_path: Path) -> None:
    client = client_for("solve", log_path)
    first = client.post("/v1/messages", json=request_body(stream=False)).json()
    second = client.post("/v1/messages", json=request_body(1, stream=False)).json()
    assert first["id"] != second["id"]
    assert [r["call"] for r in log_records(log_path)] == [1, 2]


def test_the_stand_in_listens_only_on_this_machine() -> None:
    assert stand_in.HOST == "127.0.0.1"


# --- the request log readers ----------------------------------------------


def test_message_shape_names_tool_traffic() -> None:
    body = request_body(replies=1)
    assert records.message_shape(body["messages"]) == (
        "[user, system, assistant tool_use, user tool_result]"
    )


def test_block_text_reads_every_kind_of_block() -> None:
    use = {"type": "tool_use", "input": {"command": "ls"}}
    listed = {"type": "tool_result", "content": [{"type": "text", "text": "a"}]}
    assert records.block_text(use) == '{"command": "ls"}'
    assert records.block_text(listed) == "a"
    assert records.block_text({"type": "text", "text": "hi"}) == "hi"


def test_short_hides_the_home_folder_in_both_spellings() -> None:
    encoded = records.ENCODED_HOME + "--cache-x"
    assert records.short(f"{records.HOME}/a {encoded}") == "~/a ~--cache-x"


def test_a_missing_run_is_reported_with_a_hint() -> None:
    with pytest.raises(records.RunNotFoundError, match="run_claude.sh"):
        records.run_folder("no-such-run")


def test_the_first_loop_call_skips_a_side_call() -> None:
    side = logged_call(request_body(tools=False), [])
    loop = logged_call(request_body(), [])
    assert records.first_loop_call([side, loop]) is loop
    with pytest.raises(records.RunNotFoundError):
        records.first_loop_call([side])


def make_run(name: str, calls: list[dict[str, Any]], stream: list[dict]) -> Path:
    """Write a run folder the way run_claude.sh and the stand-in would.

    Args:
        name: The run's name.
        calls: The calls.jsonl records.
        stream: The stream.jsonl events.

    Returns:
        The folder, under the test's COURSE_HOME.
    """
    folder = tasks.runs_root() / name
    folder.mkdir(parents=True)
    lines = [json.dumps(record) for record in calls]
    (folder / records.CALLS_FILE).write_text("\n".join(lines) + "\n")
    events = [json.dumps(event) for event in stream]
    (folder / records.STREAM_FILE).write_text("\n".join(events) + "\n")
    (folder / records.STDERR_FILE).write_text("")
    return folder


def logged_call(body: dict[str, Any], content: list[dict[str, Any]]) -> dict:
    """Build one calls.jsonl record.

    Args:
        body: The request body.
        content: The content blocks the stand-in answered with.

    Returns:
        The record.
    """
    message = {"content": content, "stop_reason": "end_turn"}
    reply = {"kind": "loop", "streamed": True, "message": message}
    headers = {"x-api-key": "any-string-works", "anthropic-beta": "a,b"}
    return {
        "call": 1,
        "method": "POST",
        "path": "/v1/messages",
        "headers": headers,
        "body": body,
        "reply": reply,
    }


def fake_count(text: str) -> int:
    """Count one token per word: enough to check the arithmetic."""
    return len(text.split())


def test_measure_counts_each_part() -> None:
    size = inspect_calls.measure(request_body(), fake_count)
    assert size.system_chars == len(SYSTEM[0]["text"])
    assert size.system_tokens == fake_count(SYSTEM[0]["text"])
    assert size.tools_tokens == fake_count(json.dumps([BASH_TOOL]))
    assert size.messages_tokens == fake_count(QUESTION) + 2
    assert size.total == size.system_tokens + size.tools_tokens + size.messages_tokens


def test_side_calls_offer_no_tools_or_name_another_model() -> None:
    reason = inspect_calls.side_call_reason
    assert reason(request_body(tools=False), "stand-in") == "offers no tools"
    small = request_body() | {"model": "claude-haiku-4-5"}
    assert reason(small, "stand-in") == "names claude-haiku-4-5"
    assert reason(request_body(), "stand-in") == ""


def test_schema_line_marks_required_parameters() -> None:
    line = inspect_calls.schema_line(BASH_TOOL)
    assert line == "command (string, required) · timeout (number)"


def test_skills_and_claude_md_are_found_in_the_messages() -> None:
    body = request_body()
    listing = f"{inspect_calls.SKILLS_HEADER}\n\n- sqlite-basics: How to.\n- loop: x\n"
    memory = "Contents of /t/CLAUDE.md (project instructions, checked in):\n# Notes"
    body["messages"] += [
        {"role": "system", "content": listing},
        {"role": "user", "content": [{"type": "text", "text": memory}]},
    ]
    assert inspect_calls.skills_listed(body) == ["sqlite-basics", "loop"]
    assert inspect_calls.sends_claude_md(body)
    assert not inspect_calls.sends_claude_md(request_body())


def test_report_run_prints_each_call_and_the_shared_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = [
        logged_call(request_body(0), [{"type": "text", "text": "a"}]),
        logged_call(request_body(1), [{"type": "text", "text": "b"}]),
    ]
    make_run("two", calls, [])
    inspect_calls.report_run("two", fake_count, show_system=True)
    out = capsys.readouterr().out
    assert "call 2" in out and "the same 23 characters as before" in out
    assert "same system prompt  yes, across the 2 calls" in out
    assert "side calls          0 of 2" in out
    assert "input: command (string, required)" in out
    assert "===== the Bash tool, as sent" in out


def test_compare_runs_puts_two_first_calls_side_by_side(
    capsys: pytest.CaptureFixture[str],
) -> None:
    big = request_body() | {"tools": [BASH_TOOL, BASH_TOOL | {"name": "Read"}]}
    make_run("big", [logged_call(big, [])], [])
    make_run("small", [logged_call(request_body(), [])], [])
    inspect_calls.compare_runs("big", "small", fake_count)
    out = capsys.readouterr().out
    assert "tools only in big: Read" in out
    assert "times the tokens of small's" in out


# --- what Claude Code printed ---------------------------------------------


def test_describe_shows_tool_calls_results_and_the_result() -> None:
    use = {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
    error = {"type": "tool_result", "content": "No such tool", "is_error": True}
    assistant = {"type": "assistant", "message": {"content": [use]}}
    user = {"type": "user", "message": {"content": [error]}}
    result = {"type": "result", "subtype": "success", "num_turns": 2, "result": "x"}
    assert show_stream.describe(assistant) == ["assistant  tool_use Bash: ls"]
    error_line = "user       tool_result (error): No such tool"
    assert show_stream.describe(user) == [error_line]
    assert show_stream.describe(result)[0].startswith("result     success · 2 turns")


def test_one_line_gives_the_full_length_of_a_cut_text() -> None:
    clipped = show_stream.one_line("x" * 200)
    assert clipped.endswith("… (200 characters)")
    assert records.clip("a\nb" * 60, 10) == "a⏎ba⏎ba⏎ba…"


# --- the mishaps ------------------------------------------------------------


def test_exchanges_pair_each_tool_call_with_its_result() -> None:
    first = logged_call(request_body(0), [])
    call = {"type": "tool_use", "id": "toolu_0", "name": "bash"}
    first["reply"]["message"]["content"] = [call | {"input": {"command": "ls"}}]
    second = logged_call(request_body(1), [])
    second["body"]["messages"][-1]["content"][0]["is_error"] = True
    (exchange,) = mishaps.exchanges([first, second])
    assert exchange == mishaps.Exchange(1, "bash", "ls", "ok", True, 2)


def test_a_fallback_request_is_the_same_request_without_streaming() -> None:
    streamed = logged_call(request_body(stream=True), [])
    fallback = logged_call(request_body(stream=False), [])
    assert mishaps.resent_without_stream(streamed, fallback)
    assert not mishaps.resent_without_stream(fallback, streamed)
    other = logged_call(request_body(replies=1, stream=False), [])
    assert not mishaps.resent_without_stream(streamed, other)


def test_the_persisted_output_note_is_read_apart() -> None:
    note = (
        "<persisted-output>\nOutput too large (24.8KB). Full output saved to: "
        "/x/tool-results/a.txt\n\nPreview (first 2KB):\nrow 1\nrow 2\n...\n"
        "</persisted-output>"
    )
    assert mishaps.SAVED_RE.search(note).group(1) == "/x/tool-results/a.txt"
    assert mishaps.PREVIEW_RE.search(note).group(2) == "row 1\nrow 2"


def test_outcome_reads_the_result_event(tmp_path: Path) -> None:
    events = [{"type": "assistant"}, {"type": "result", "subtype": "success"}]
    events[1]["result"] = "done"
    (tmp_path / records.STREAM_FILE).write_text("\n".join(map(json.dumps, events)))
    assert mishaps.outcome(tmp_path) == "printed 1 replies; result success: 'done'"
    (tmp_path / records.STREAM_FILE).write_text("")
    assert mishaps.outcome(tmp_path) == "printed 0 replies and no result"


# --- the MCP server for "your setup" ----------------------------------------


def test_the_mcp_server_lists_and_runs_its_one_tool() -> None:
    hello = notes_mcp.answer({"id": 1, "method": "initialize", "params": {}})
    assert hello["result"]["capabilities"] == {"tools": {}}
    listed = notes_mcp.answer({"id": 2, "method": "tools/list"})
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["read_note"]
    params = {"name": "read_note", "arguments": {"table": "doctors"}}
    ran = notes_mcp.answer({"id": 3, "method": "tools/call", "params": params})
    assert ran["result"]["content"][0]["text"].startswith("doctors(")
    assert notes_mcp.answer({"method": "notifications/initialized"}) is None
    missing = notes_mcp.answer({"id": 4, "method": "nope"})
    assert missing["error"]["code"] == notes_mcp.METHOD_NOT_FOUND
    other = {"name": "write_note", "arguments": None}
    wrong = notes_mcp.answer({"id": 5, "method": "tools/call", "params": other})
    assert wrong["error"]["code"] == notes_mcp.INVALID_PARAMS


def test_the_mcp_server_speaks_one_json_object_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = '{"id": 1, "method": "tools/list"}\n\n{"method": "x"}\nnot json\n[]\n'
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    notes_mcp.main()
    listed, broken, listy = capsys.readouterr().out.splitlines()
    assert json.loads(listed)["id"] == 1
    assert json.loads(broken)["error"]["code"] == notes_mcp.PARSE_ERROR
    assert json.loads(listy)["error"]["code"] == notes_mcp.INVALID_REQUEST


def test_only_a_quoted_query_on_data_db_is_re_run() -> None:
    assert mishaps.QUERY_RE.fullmatch(stand_in.DUMP)
    assert not mishaps.QUERY_RE.fullmatch('sqlite3 data.db "x"; rm -rf /')
    assert not mishaps.QUERY_RE.fullmatch("echo hi")


# --- the task, the grader and the tokenizer --------------------------------


def test_the_task_and_grader_are_chapter_1_s() -> None:
    assert filecmp.cmp(CHAPTER / "grader.py", ROOT / "01-the-job" / "grader.py")
    bundled = ROOT / "01-the-job" / "data" / "dbs" / "clinic.db"
    assert filecmp.cmp(CHAPTER / "data" / "clinic.db", bundled, shallow=False)
    task = tasks.load_task()
    assert (task.task_id, task.answer) == ("clinic-easy-001", "30")
    assert QUESTION.endswith(
        "to a file named answer.txt in the current folder. "
        "The answer is a single number."
    )


@pytest.mark.parametrize(("written", "correct"), [("30\n", True), ("31", False)])
def test_grading_the_answer(tmp_path: Path, written: str, correct: bool) -> None:
    (tmp_path / tasks.ANSWER_FILE).write_text(written)
    assert grader.grade(tasks.load_task(), tmp_path).correct is correct


def test_a_checked_tokenizer_copy_is_found(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = course_home / "models" / "LFM2.5-2.6B" / lfm_tokens.TOKENIZER_FILE
    copy.parent.mkdir(parents=True)
    copy.write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(lfm_tokens, "TOKENIZER_SHA256", digest)
    assert lfm_tokens.find_tokenizer() == copy


def test_a_download_with_the_wrong_fingerprint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lfm_tokens.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "tokenizer.json"
    with pytest.raises(lfm_tokens.TokenizerError, match="fingerprint"):
        lfm_tokens.download(target)
    assert not target.exists() and not list(tmp_path.iterdir())


def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
    """Pretend to download: return bytes that cannot match the fingerprint."""
    return io.BytesIO(b"not the tokenizer")


# --- run_claude.sh ----------------------------------------------------------


RUN_SCRIPT = (CHAPTER / "run_claude.sh").read_text()


def test_run_claude_sh_parses() -> None:
    subprocess.run(["bash", "-n", str(CHAPTER / "run_claude.sh")], check=True)


@pytest.mark.parametrize(
    "line",
    [
        'CLAUDE_VERSION="2.1.290"',
        'BASE_URL="http://127.0.0.1:${PORT}"',
        'ANTHROPIC_BASE_URL="$BASE_URL"',
        'CLAUDE_CONFIG_DIR="$RUN_DIR/config"',
        'HOME="$RUN_DIR/home"',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"',
        'DISABLE_TELEMETRY="1"',
        'DISABLE_ERROR_REPORTING="1"',
        'DISABLE_AUTOUPDATER="1"',
        "env -i",
        "--output-format stream-json --verbose",
        "--permission-mode bypassPermissions",
        "--tools Bash --strict-mcp-config",
        '"bashOutputMaxChars\\": ${bash_max}',
        'bash_max="4000"',
        "argv+=(--bare)",
        'npm_config_registry="$NPM_REGISTRY"',
    ],
)
def test_run_claude_sh_keeps_claude_code_local_and_private(line: str) -> None:
    assert line in RUN_SCRIPT


def test_no_address_but_this_machine_is_ever_a_base_url() -> None:
    urls = [line for line in RUN_SCRIPT.splitlines() if "BASE_URL=" in line]
    assert urls and all("127.0.0.1" in line or "$BASE_URL" in line for line in urls)
