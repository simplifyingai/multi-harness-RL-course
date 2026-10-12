"""Chapter 17 without a model: the parser, the tool, the loop and the protocols.

No test here loads model weights or needs MLX. The harness is given a scripted
policy: reply texts written in advance with LFM2.5's special tokens in them,
shaped like the replies the project recorded (``I should see the tables and
columns before writing SQL.</think><|tool_call_start|>[bash(command='sqlite3
data.db .schema')]<|tool_call_end|><|im_end|>``). The tool runs for real on a
copy of the bundled clinic database. One test renders with LFM2.5's real
tokenizer and chat template when those files are on this machine (a tokenizer
load, no weights); it is skipped otherwise.
"""

from __future__ import annotations

import contextlib
import filecmp
import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("17-tiny-harness")

import bash_tool  # noqa: E402
import database  # noqa: E402
import experiments  # noqa: E402
import lfm_files  # noqa: E402
import report  # noqa: E402
import tasks  # noqa: E402
import tiny_harness  # noqa: E402
import toolcalls  # noqa: E402
from policy import FINISH_LENGTH, FINISH_STOP, Reply  # noqa: E402

START, END = toolcalls.TOOL_CALL_START, toolcalls.TOOL_CALL_END
EOT = toolcalls.END_OF_TURN
TOOLS = [tiny_harness.bash_schema()]
EASY = tasks.find_task("clinic-easy-001")
MURCIA = "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where a real run of the course keeps Liquid's export.
REAL_MODEL = lfm_files.model_folder(lfm_files.FOUR_BIT)
HAVE_TOKENIZER = (REAL_MODEL / "tokenizer.json").is_file() and (
    REAL_MODEL / "chat_template.jinja"
).is_file()


def call(command: str, name: str = "bash", think: str = "Look first.") -> str:
    """Write a reply that calls one tool, as LFM2.5 writes it.

    Args:
        command: The command argument.
        name: The tool's name.
        think: The reasoning before ``</think>``.

    Returns:
        The reply text, end of turn included.
    """
    return f"{think}</think>{START}[{name}(command={command!r})]{END}{EOT}"


def answer(text: str = "The answer is in answer.txt.") -> str:
    """Write a reply with no tool call: a final answer.

    Args:
        text: The visible text.

    Returns:
        The reply text.
    """
    return f"Done.</think>{text}{EOT}"


def fake_render(messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
    """Render a chat the way LFM2.5's template lays it out, roughly.

    Args:
        messages: The chat.
        tools: The tools offered.

    Returns:
        The text, the tools in the system prompt, ending with ``<think>``.
    """
    parts = []
    for message in messages:
        content = message["content"]
        if message["role"] == "system" and tools:
            content += "\nList of tools: " + json.dumps(tools)
        parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
    return "<|startoftext|>" + "".join(parts) + "<|im_start|>assistant\n<think>"


class Scripted:
    """A policy that answers from a list of reply texts, in order."""

    def __init__(self, replies: list[str], render: Any = fake_render) -> None:
        """Keep the replies and how to render.

        Args:
            replies: The reply texts.
            render: The function that renders a chat.
        """
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.tools_seen: list[Any] = []
        self._render = render

    def render(self, messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
        """Render, and remember which tools were offered.

        Args:
            messages: The chat.
            tools: The tools.

        Returns:
            The prompt.
        """
        self.tools_seen.append(tools)
        return self._render(messages, tools)

    def complete(self, prompt: str) -> Reply:
        """Hand back the next scripted reply.

        Args:
            prompt: The prompt (remembered).

        Returns:
            The reply; ``length`` when the text has no end of turn.
        """
        self.prompts.append(prompt)
        text = self.replies.pop(0)
        finish = FINISH_STOP if text.endswith(EOT) else FINISH_LENGTH
        return Reply(text, len(prompt), len(text), finish, 0.0)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A fresh task folder holding the clinic database as data.db.

    Args:
        tmp_path: pytest's per-test folder.

    Returns:
        The folder.
    """
    return tasks.prepare_workdir(tasks.find_task("clinic-easy-001"), tmp_path / "w")


def run_main(module: ModuleType, *args: str) -> str:
    """Run a script's main() with arguments, and return what it printed.

    Args:
        module: The chapter module.
        *args: Command-line arguments.

    Returns:
        Everything printed.
    """
    saved = sys.argv
    sys.argv = [f"{module.__name__}.py", *args]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = saved
    return out.getvalue()


def use_policy(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, policy: Any
) -> None:
    """Make a script load a scripted policy instead of the model.

    Args:
        monkeypatch: pytest's patcher.
        module: The script's module.
        policy: The policy to hand it.
    """
    loaded = tiny_harness.Loaded(policy, "scripted replies", lambda: None)
    monkeypatch.setattr(module, "load_policy", lambda fast, temperature: loaded)


# ---------------------------------------------------------------- the bundle


def test_the_bundle_is_chapter_1s_files() -> None:
    """The questions, databases, grader and transcript are chapter 1's, unchanged."""
    first = ROOT / "01-the-job"
    pairs = [
        ("data/test.jsonl", "data/test.jsonl"),
        ("data/dbs/clinic.db", "data/dbs/clinic.db"),
        ("data/dbs/airline.db", "data/dbs/airline.db"),
        ("grader.py", "grader.py"),
        (
            "data/clinic-medium-001.claude-code.jsonl",
            "data/transcripts/clinic-medium-001.claude-code.jsonl",
        ),
    ]
    for mine, theirs in pairs:
        assert filecmp.cmp(CHAPTER / mine, first / theirs, shallow=False), mine


def test_the_held_out_questions_and_the_instruction() -> None:
    held_out = tasks.load_tasks()
    assert len(held_out) == 50
    assert {task.domain for task in held_out} == {"clinic", "airline"}
    first_five = [task.task_id for task in held_out[:5]]
    assert first_five == [
        "clinic-medium-001",
        "airline-medium-001",
        "clinic-easy-001",
        "airline-easy-001",
        "clinic-hard-001",
    ]
    text = tasks.task_instruction(tasks.find_task("clinic-easy-001"))
    assert "How many patients live in Murcia?" in text
    assert text.endswith("The answer is a single number.")
    with pytest.raises(SystemExit, match="no held-out question"):
        tasks.find_task("shop-easy-001")


def test_a_work_folder_starts_fresh(tmp_path: Path) -> None:
    task = tasks.find_task("clinic-easy-001")
    folder = tasks.prepare_workdir(task, tmp_path / "w")
    (folder / "answer.txt").write_text("old\n")
    folder = tasks.prepare_workdir(task, tmp_path / "w")
    assert sorted(path.name for path in folder.iterdir()) == ["data.db"]


# ---------------------------------------------------------------- the parser


def test_a_reply_splits_into_reasoning_visible_text_and_calls() -> None:
    reply = (
        "I should see the tables and columns before writing SQL.</think>"
        f"Let me look.{START}[bash(command='sqlite3 data.db .schema')]{END}{EOT}"
    )
    parsed = toolcalls.split_reply(reply, TOOLS)
    assert parsed.reasoning == "I should see the tables and columns before writing SQL."
    assert parsed.content == "Let me look."
    assert parsed.calls == [
        toolcalls.ToolCall("bash", {"command": "sqlite3 data.db .schema"})
    ]
    assert parsed.opened == 1


def test_positional_arguments_are_named_from_the_schema() -> None:
    parsed = toolcalls.split_reply(
        f'Tables.</think>{START}[bash("sqlite3 data.db .tables")]{END}', TOOLS
    )
    assert parsed.calls[0].arguments == {"command": "sqlite3 data.db .tables"}
    # One more than the schema names gets a numbered name.
    extra = toolcalls.parse_calls("[bash('ls', 'x')]", TOOLS)
    assert extra[0].arguments == {"command": "ls", "arg1": "x"}
    # An unknown tool has no schema: every positional argument is numbered.
    assert toolcalls.parse_calls("[run('ls')]", TOOLS)[0].arguments == {"arg0": "ls"}


def test_unparseable_calls_are_dropped() -> None:
    for source in ("[bash(command='ls'", "[bash(command=)]", "bash(", "[42, 'x']", ""):
        assert toolcalls.parse_calls(source, TOOLS) == [], source
    deep = "[" * 100_000  # deeper than ast.parse can go: still no crash
    assert toolcalls.parse_calls(deep, TOOLS) == []
    # Only the item that is a call survives a mixed list.
    mixed = toolcalls.parse_calls("[42, bash(command='ls')]", TOOLS)
    assert [c.arguments for c in mixed] == [{"command": "ls"}]


def test_a_value_that_is_not_a_literal_is_kept_as_its_source() -> None:
    calls = toolcalls.parse_calls("[bash(command=x + 'y')]", TOOLS)
    assert calls[0].arguments == {"command": "x + 'y'"}


def test_several_calls_in_one_reply_and_dotted_names() -> None:
    reply = (
        f"Two.</think>{START}[bash(command='ls'), "
        f"bash(command='sqlite3 data.db .tables')]{END}"
        f"{START}[tools.bash(command='ls')]{END}{EOT}"
    )
    parsed = toolcalls.split_reply(reply, TOOLS)
    assert [c.name for c in parsed.calls] == ["bash", "bash", "tools.bash"]
    assert parsed.opened == 2


def test_a_call_inside_the_reasoning_is_not_a_call() -> None:
    inside = f"Maybe {START}[bash(command='ls')]{END} first.</think>The answer is 30."
    parsed = toolcalls.split_reply(inside + EOT, TOOLS)
    assert parsed.calls == [] and parsed.content == "The answer is 30."
    # A reply that never closes its thinking is all reasoning.
    unclosed = toolcalls.split_reply(f"Hmm {START}[bash(command='ls')]{END}", TOOLS)
    assert unclosed.calls == [] and unclosed.content == "" and unclosed.opened == 0


def test_a_block_cut_off_by_the_cap_is_opened_but_not_parsed() -> None:
    parsed = toolcalls.split_reply(f"Go.</think>{START}[bash(command='sqlite3 d", TOOLS)
    assert parsed.calls == [] and parsed.opened == 1


def test_a_call_prints_as_a_harness_receives_it() -> None:
    assert toolcalls.ToolCall("bash", {"command": "ls"}).as_json() == (
        'bash {"command": "ls"}'
    )


# ------------------------------------------------------------------ the tool


def run(command: str, folder: Path) -> bash_tool.ToolResult:
    """Run one command with the chapter's tool.

    Args:
        command: The command.
        folder: The task folder.

    Returns:
        What the tool gave back.
    """
    return bash_tool.run_command(command, folder)


def test_sqlite3_prints_what_the_sqlite3_program_prints(workdir: Path) -> None:
    # Each expected text is the real sqlite3 3.51's output on the same file.
    cases = {
        "sqlite3 data.db .tables": "appointments  doctors       patients",
        'sqlite3 data.db ".tables"': "appointments  doctors       patients",
        MURCIA: "30",
        'sqlite3 data.db "select avg(fee) from appointments"': "109.94375",
        'sqlite3 data.db "select 523.0, 1e20, 0.1 + 0.2, null"': "523.0|1.0e+20|0.3|",
        'sqlite3 data.db "select 1; select 2"': "1\n2",
        'sqlite3 data.db ".schema doctors"': (
            "CREATE TABLE doctors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "specialty TEXT NOT NULL);"
        ),
        'sqlite3 -header -column data.db "select specialty, count(*) as n from '
        'doctors group by specialty order by n desc limit 3"': (
            "specialty         n\n----------------  -\nGeneral Practice  7\n"
            "Pediatrics        5\nOrthopedics       4"
        ),
        "sqlite3 -column data.db \"select 1 as longname, 'x' as b, 2.0 as c\"": (
            "1         x  2.0"
        ),
        'sqlite3 -csv -header data.db "select name, 1.5 from doctors limit 2"': (
            'name,1.5\n"Leo Larsen",1.5\n"Tomas Weber",1.5'
        ),
        "sqlite3 -csv data.db \"select 'abc', 'a b', 'x,y', '12', null, 3\"": (
            'abc,"a b","x,y",12,,3'
        ),
        "sqlite3 -header data.db \"select 1 as a, 'x' as b\"": "a|b\n1|x",
        'sqlite3 data.db ".headers on" "select 1 as a"': "a\n1",
    }
    for command, expected in cases.items():
        result = run(command, workdir)
        assert result == bash_tool.ToolResult(expected, ok=True), command


def test_sqlite3_cannot_change_any_file(workdir: Path) -> None:
    before = hashlib.sha256((workdir / "data.db").read_bytes()).hexdigest()
    writes = {
        'sqlite3 data.db "DELETE FROM doctors"': "readonly database",
        'sqlite3 data.db "CREATE TABLE t (x)"': "readonly database",
        "sqlite3 data.db \"ATTACH 'other.db' AS other\"": "not authorized",
        'sqlite3 data.db ".shell ls"': "cannot run .shell in safe mode",
        'sqlite3 data.db ".output out.txt"': "cannot run .output in safe mode",
        'sqlite3 data.db ".foo"': 'unknown command or invalid arguments:  "foo"',
    }
    for command, words in writes.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command
    after = hashlib.sha256((workdir / "data.db").read_bytes()).hexdigest()
    assert after == before
    assert sorted(path.name for path in workdir.iterdir()) == ["data.db"]


def test_every_sqlite3_command_goes_through_chapter_11s_guard(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    original = database.connect_read_only

    def spy(path: Path) -> Any:
        """Record each database opened, then open it the guarded way."""
        opened.append(path)
        connection = original(path)
        assert connection.execute("PRAGMA query_only").fetchone() == (1,)
        return connection

    monkeypatch.setattr(database, "connect_read_only", spy)
    for command in ("sqlite3 data.db .tables", 'sqlite3 data.db ".schema"'):
        run(command, workdir)
    run('sqlite3 data.db "SELECT 1"', workdir)
    assert opened == [workdir / "data.db"] * 3


def test_sqlite3_stays_in_the_task_folder(workdir: Path) -> None:
    refused = {
        "sqlite3 ../data.db .tables": "only the task folder",
        "sqlite3 /etc/hosts .tables": "only the task folder",
        "sqlite3 new.db .tables": 'unable to open database "new.db"',
        "sqlite3 data.db": "cannot wait for typed input",
        "sqlite3": "needs a database",
        "sqlite3 -json data.db .tables": "unknown option: -json",
        'sqlite3 data.db "select * from nope"': "no such table: nope",
    }
    for command, words in refused.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command
    assert not (workdir / "new.db").exists()


def test_ls_lists_only_the_task_folder(workdir: Path) -> None:
    assert run("ls", workdir) == bash_tool.ToolResult("data.db", ok=True)
    long = run("ls -la", workdir).output
    assert long.endswith(" 36864 data.db") and long.startswith("-rw")
    for command, words in {
        "ls ..": "only the task folder",
        "ls /": "only the task folder",
        "ls ~": "~: No such file or directory",  # ~ is not expanded
        "ls -R": "option not supported here: -R",
        "ls nope": "nope: No such file or directory",
    }.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command


def test_echo_writes_only_answer_txt(workdir: Path) -> None:
    result = run('echo "General Practice" > answer.txt', workdir)
    assert result == bash_tool.ToolResult("", ok=True)
    assert (workdir / "answer.txt").read_text() == "General Practice\n"
    run(f"echo -n 30 > {workdir / 'answer.txt'}", workdir)
    assert (workdir / "answer.txt").read_text() == "30"
    run("echo 31>./answer.txt", workdir)  # shlex cuts "31>" into two words
    assert (workdir / "answer.txt").read_text() == "31\n"
    for command, words in {
        "echo 30 > other.txt": "may write only answer.txt",
        "echo 30 > /tmp/answer.txt": "only the task folder",
        "echo 30 >> answer.txt": "'>>' is not run",
        "echo hi": "allowed only as",
        "echo > answer.txt > x": "'>' is not run",
        "echo $(rm data.db) > answer.txt": "command substitution",
        "echo `rm data.db` > answer.txt": "command substitution",
    }.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command
    assert (workdir / "answer.txt").read_text() == "31\n"


def test_one_command_per_call_and_only_three_programs(workdir: Path) -> None:
    for command, words in {
        "sqlite3 data.db .tables | head": "'|' is not run",
        "ls && echo 1 > answer.txt": "'&&' is not run",
        "ls; rm data.db": "';' is not run",
        "sqlite3 data.db .tables 2>&1": "'>&' is not run",
        "sqlite3 data.db 'select 1' > out.txt": "'>' is not run",
        "cat answer.txt": "cat: not allowed here",
        "rm data.db": "rm: not allowed here",
        "bash -c 'ls'": "bash: not allowed here",
        'echo "abc': "No closing quotation",
        "   ": "the command is empty",
    }.items():
        result = run(command, workdir)
        assert result.output.startswith("Error: ") and words in result.output, command
    assert (workdir / "data.db").is_file()


def test_quoted_operators_inside_sql_are_fine(workdir: Path) -> None:
    command = 'sqlite3 data.db "SELECT COUNT(*) FROM doctors WHERE id > 5; -- a | b"'
    assert run(command, workdir) == bash_tool.ToolResult("25", ok=True)


def test_a_long_output_is_cut_and_says_so(workdir: Path) -> None:
    result = run('sqlite3 data.db "SELECT * FROM appointments"', workdir)
    head, note = result.output.rsplit("\n", 1)
    assert len(head) == bash_tool.OUTPUT_LIMIT
    assert note.startswith("[output cut to 4,000 of ") and note.endswith(" characters]")


def test_statements_split_like_sqlite3() -> None:
    split = database.split_statements
    assert split("SELECT 1") == ["SELECT 1;"]
    assert split("SELECT 1; SELECT 2;") == ["SELECT 1;", "SELECT 2;"]
    assert split("SELECT 'a;b'") == ["SELECT 'a;b';"]
    assert split(" ; ;") == []
    assert split("SELECT 'open") == ["SELECT 'open"]


# ------------------------------------------------------------------ the loop


def test_a_solved_question_takes_four_calls(workdir: Path) -> None:
    policy = Scripted(
        [
            call("sqlite3 data.db .tables", think="I should see the tables."),
            call(MURCIA),
            call("echo 30 > answer.txt"),
            answer("The answer, 30, is in answer.txt."),
        ]
    )
    task = tasks.find_task("clinic-easy-001")
    rollout = tiny_harness.run_rollout(policy, task, workdir)
    assert rollout.stop == tiny_harness.STOP_NO_CALL
    assert len(rollout.turns) == 4 and rollout.grade.correct
    assert [a.result.output for t in rollout.turns for a in t.actions] == [
        "appointments  doctors       patients",
        "30",
        "",
    ]
    # The conversation: rules, task, then each reply word for word and its
    # observation as a tool message.
    roles = [message["role"] for message in rollout.messages]
    assert roles == ["system", "user"] + ["assistant", "tool"] * 3 + ["assistant"]
    assert rollout.messages[2]["content"] == "<think>" + policy_text(
        "sqlite3 data.db .tables", "I should see the tables."
    )
    assert rollout.messages[3]["content"] == "appointments  doctors       patients"
    assert "List of tools" in policy.prompts[0]
    assert policy.tools_seen[0] == TOOLS


def policy_text(command: str, think: str) -> str:
    """The text a scripted call reply goes back into the chat as.

    Args:
        command: The call's command.
        think: Its reasoning.

    Returns:
        The reply without its end-of-turn token.
    """
    return call(command, think=think).removesuffix(EOT)


def test_the_budget_stops_after_eight_calls(workdir: Path) -> None:
    policy = Scripted([call("ls")] * 9)
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    assert rollout.stop == tiny_harness.STOP_BUDGET
    assert len(rollout.turns) == lfm_files.MAX_CALLS == 8
    assert len(policy.replies) == 1  # the ninth reply was never asked for
    assert rollout.grade.reason == "no answer.txt"


def test_three_unreadable_replies_in_a_row_stop_the_loop(workdir: Path) -> None:
    broken = f"Try.</think>{START}[bash(command='ls'{END}{EOT}"
    policy = Scripted([broken] * 4)
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    assert rollout.stop == tiny_harness.STOP_UNREADABLE
    assert len(rollout.turns) == 3
    notes = [turn.note for turn in rollout.turns]
    assert notes[0] is not None and "could not be parsed" in notes[0]
    assert notes[2] is None  # the last one is not answered: the loop stops
    users = [m for m in rollout.messages if m["role"] == "user"]
    assert len(users) == 3  # the task, then two notes


def test_a_readable_call_resets_the_unreadable_count(workdir: Path) -> None:
    broken = f"Try.</think>{START}[bash(command=]{END}{EOT}"
    cut_off = "I am still thinking about which table"  # no end of turn: the cap
    policy = Scripted([broken, cut_off, call("ls"), broken, broken, answer()])
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    assert rollout.stop == tiny_harness.STOP_NO_CALL and len(rollout.turns) == 6
    assert "cut off by the length cap" in (rollout.turns[1].note or "")


def test_a_tool_offered_as_bash_is_not_bash(workdir: Path) -> None:
    policy = Scripted(
        [
            call("sqlite3 data.db .tables"),
            call("sqlite3 data.db .tables", name="Bash"),
            f"Write.</think>{START}[write(file='answer.txt', content='30')]{END}{EOT}",
            answer(),
        ]
    )
    task = tasks.find_task("clinic-easy-001")
    rollout = tiny_harness.run_rollout(policy, task, workdir, tool_name="Bash")
    outputs = [a.result.output for t in rollout.turns for a in t.actions]
    assert outputs == [
        "Error: No such tool available: bash. Tool names are case-sensitive: "
        "call Bash instead.",
        "appointments  doctors       patients",
        "Error: No such tool available: write",
    ]
    assert len(rollout.turns) == 4  # each wrong name spent a call
    assert '"name": "Bash"' in policy.prompts[0]
    assert "with the Bash tool" in policy.prompts[0]


def test_a_call_without_a_command_is_an_error(workdir: Path) -> None:
    reply = f"Hm.</think>{START}[bash(cmd='ls')]{END}{EOT}"
    policy = Scripted([reply, answer()])
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    assert rollout.turns[0].actions[0].result.output == tiny_harness.NO_COMMAND


def test_several_calls_in_one_reply_run_in_order_as_one_call(workdir: Path) -> None:
    reply = (
        f"Both.</think>{START}[bash(command='ls'), "
        f"bash(command='echo 30 > answer.txt')]{END}{EOT}"
    )
    policy = Scripted([reply, answer()])
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    assert [m["role"] for m in rollout.messages[2:5]] == ["assistant", "tool", "tool"]
    assert len(rollout.turns) == 2 and rollout.grade.correct


def test_a_rollout_is_saved_as_json(workdir: Path, tmp_path: Path) -> None:
    policy = Scripted([call("echo 30 > answer.txt"), answer()])
    rollout = tiny_harness.run_rollout(policy, EASY, workdir)
    path = tmp_path / "saved" / "r.json"
    tiny_harness.save_rollout(rollout, path)
    record = json.loads(path.read_text())
    assert record["grade"]["correct"] is True
    assert record["calls"][0]["actions"][0]["call"]["arguments"] == {
        "command": "echo 30 > answer.txt"
    }
    assert record["messages"][-1]["role"] == "assistant"


@pytest.mark.skipif(not HAVE_TOKENIZER, reason="LFM2.5-2.6B's tokenizer is not here")
def test_the_real_template_prints_the_tool_and_keeps_each_reply(workdir: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(REAL_MODEL))

    def render(messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
        """Render as lfm.py does."""
        return tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
            preserve_thinking=True,
        )

    first = call("sqlite3 data.db .tables")
    policy = Scripted([first, answer()], render=render)
    tiny_harness.run_rollout(policy, EASY, workdir)
    prompt, second = policy.prompts
    assert prompt.startswith("<|startoftext|><|im_start|>system\nYou are a data")
    tools_line = '\nList of tools: [{"type": "function", "function": {"name": "bash"'
    assert tools_line in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>")
    # The second prompt is the first, the reply exactly as written, then the
    # observation as a tool message.
    observation = "appointments  doctors       patients"
    expected = f"{prompt}{first}\n<|im_start|>tool\n{observation}<|im_end|>\n"
    assert second == expected + "<|im_start|>assistant\n<think>"
    ids = tokenizer.encode(first, add_special_tokens=False)
    assert tokenizer.convert_ids_to_tokens(ids[-1]) == EOT


# ------------------------------------------------------------- the protocols


def text_reply(block: str, fence: str = "bash") -> str:
    """Write a text-protocol reply: a thought, then one fenced block.

    Args:
        block: The command inside the fence.
        fence: The fence's language word.

    Returns:
        The reply text.
    """
    return f"Plan.</think>THOUGHT: look.\n\n```{fence}\n{block}\n```{EOT}"


def test_native_calls_under_the_text_protocol_abort_after_three(workdir: Path) -> None:
    policy = Scripted([call('sqlite3 data.db ".tables"')] * 4)
    task = tasks.find_task("clinic-medium-001")
    run = experiments.run_text_protocol(policy, task, workdir)
    assert run.exit_status == experiments.EXIT_FORMAT
    assert len(run.turns) == 3 and [t.format_errors for t in run.turns] == [1, 2, 3]
    assert policy.tools_seen == [None, None, None]  # no tool is offered
    assert "Your response must contain exactly ONE bash code block" in policy.prompts[0]
    error = run.turns[0].answer
    first_lines = "Format error:\n\n<error>\nExpected exactly 1 action, found 0."
    assert error.startswith(first_lines)
    please = "Please always provide EXACTLY ONE action in triple backticks, found 0"
    assert please in error
    assert all(len(turn.parsed.calls) == 1 for turn in run.turns)  # never read
    assert run.grade.reason == "no answer.txt"


def test_one_fenced_block_runs_and_resets_the_count(workdir: Path) -> None:
    policy = Scripted(
        [
            call("ls"),
            text_reply('sqlite3 data.db ".tables"'),
            text_reply("echo 30 > answer.txt", fence="sh"),
            text_reply(experiments.SUBMIT_COMMAND, fence=""),
        ]
    )
    run = experiments.run_text_protocol(policy, EASY, workdir)
    assert run.exit_status == experiments.EXIT_SUBMITTED
    assert [t.format_errors for t in run.turns] == [1, 0, 0, 0]
    assert run.turns[1].answer == (
        "<returncode>0</returncode>\n<output>\nappointments  doctors       patients"
        "</output>"
    )
    assert run.turns[3].submitted and run.grade.correct


def test_two_blocks_or_one_in_the_reasoning_are_format_errors(workdir: Path) -> None:
    two = f"Hm.</think>```bash\nls\n```\n```bash\nls\n```{EOT}"
    hidden = f"```bash\nls\n```</think>THOUGHT: nothing here.{EOT}"
    cut = "Still thinking"  # cut off by the cap
    policy = Scripted([two, hidden, cut])
    run = experiments.run_text_protocol(policy, EASY, workdir)
    assert [len(t.actions) for t in run.turns] == [2, 0, 0]
    assert "found 2 actions" in run.turns[0].answer
    assert run.turns[2].answer.startswith("Your previous response reached the output")
    assert run.exit_status == experiments.EXIT_FORMAT


def test_the_text_protocol_stops_at_eight_calls(workdir: Path) -> None:
    policy = Scripted([text_reply("ls")] * 9)
    run = experiments.run_text_protocol(policy, EASY, workdir)
    assert run.exit_status == experiments.EXIT_LIMIT and len(run.turns) == 8


def test_a_failed_command_has_return_code_1(workdir: Path) -> None:
    replies = [text_reply("cat answer.txt"), text_reply("ls")] + [call("x")] * 3
    policy = Scripted(replies)
    run = experiments.run_text_protocol(policy, EASY, workdir)
    assert run.turns[0].answer.startswith("<returncode>1</returncode>\n<output>\nError")


def test_the_recorded_claude_code_run() -> None:
    events = experiments.read_recorded()
    calls = [event for event in events if event.kind == "call"]
    assert [event.tool for event in calls] == [
        "bash",
        "Bash",
        "bash",
        "Bash",
        "Bash",
        "Bash",
        "write",
    ]
    wasted = [e.tool for e in calls if experiments.NO_SUCH_TOOL_MARK in e.result]
    assert wasted == ["bash", "bash", "write"]
    assert all(e.is_error for e in calls if e.tool != "Bash")
    assert [e.kind for e in events if e.kind != "call"] == ["harness", "text"]
    assert events[-1].result == "Step budget exhausted; stopping."


# ------------------------------------------------------------- the scripts


def test_the_harness_prints_a_whole_run(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    policy = Scripted(
        [
            call("sqlite3 data.db .tables"),
            call(MURCIA),
            call("echo 30 > answer.txt"),
            answer(),
        ]
    )
    use_policy(monkeypatch, tiny_harness, policy)
    out = run_main(tiny_harness)
    assert out.startswith("tiny_harness.py · one model, one tool, one loop")
    for line in (
        "1. the setup",
        "clinic-easy-001 (easy): How many patients live in Murcia?",
        "List of tools: [",
        "4. the calls",
        "  call 1 · read ",
        f"    reply        <think>Look first.</think>{START}[bash(command='sqlite3 "
        f"data.db .tables')]{END}{EOT}",
        '    tool call    bash {"command": "sqlite3 data.db .tables"}',
        "    observation  appointments  doctors       patients",
        "no tool call: this is the final answer, so it stops",
        "  answer.txt: '30'",
        "chapter 1's grader: correct · reward 1",
        "calls used: 4 of 8 · stop: no tool call",
    ):
        assert line in out, line
    rollouts = course_home / "17-tiny-harness" / "rollouts"
    saved = rollouts / "harness" / "clinic-easy-001.json"
    assert json.loads(saved.read_text())["stop"] == tiny_harness.STOP_NO_CALL


def test_the_harness_runs_several_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = [call("echo 'General Practice' > answer.txt"), answer()]
    replies += [call("ls")] * 8  # the second question runs out of calls
    use_policy(monkeypatch, tiny_harness, Scripted(replies))
    out = run_main(tiny_harness, "--held-out", "2")
    assert "questions: the first 2 held-out ones, in file order" in out
    assert "grade: correct · reward 1 · 2 of 8 calls · no tool call" in out
    assert "grade: wrong (no answer.txt) · reward 0 · 8 of 8 calls · 8 calls" in out
    assert "solved 1 of 2" in out


def test_a_long_reasoning_is_cut_in_the_middle() -> None:
    text = "a" * 500 + "</think>" + f"{START}[bash(command='ls')]{END}"
    shown = report.show_reply(text, full=False)
    assert "[... 260 characters ...]" in shown and shown.endswith(END)
    assert report.show_reply(text, full=True) == "<think>" + text


def test_the_experiments_print(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = Scripted(
        [
            call("sqlite3 data.db .tables"),
            call("echo 'General Practice' > answer.txt", name="Bash"),
            answer("General Practice"),
        ]
    )
    use_policy(monkeypatch, experiments, policy)
    out = run_main(experiments)
    for line in (
        "experiments.py · the tool offered as 'Bash'",
        "Error: No such tool available: bash. Tool names are case-sensitive",
        "tool calls: 2, of which 1 answered 'No such tool available' "
        "(called as 'bash')",
        "the recorded Claude Code run",
        "tool calls: 7, of which 3 answered 'No such tool available' ('bash' 2, "
        "'write' 1)",
    ):
        assert line in " ".join(out.split()), line
    use_policy(monkeypatch, experiments, Scripted([call("ls")] * 3))
    out = run_main(experiments, "--protocol", "text")
    for line in (
        "experiments.py · mini-swe-agent's text protocol",
        "0 found: format error 3 of 3",
        'unseen       a tool call in LFM2.5\'s own syntax: bash {"command": "ls"}',
        "exit status: RepeatedFormatError after 3 of 8 calls",
        "chapter 1's grader: wrong (no answer.txt) · reward 0",
    ):
        assert line in out, line


def test_the_recorded_run_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(fast: bool, temperature: float) -> None:
        """Fail if the script tries to load a model."""
        raise AssertionError("loaded a model")

    monkeypatch.setattr(experiments, "load_policy", refuse)
    out = run_main(experiments, "--recorded")
    assert "Output token limit hit. (a reply hit the 1,024-token cap" in out
    assert "Step budget exhausted; stopping." in out


# --------------------------------------------- what the code review turned up


def test_nothing_the_model_types_crashes_the_tool(workdir: Path) -> None:
    for command, words in {
        "ls ~nosuchuser": "~nosuchuser: No such file or directory",  # no ~ expansion
        "ls 'a\x00b'": "not a usable path",
        "sqlite3 'data.db\x00' .tables": "not a usable path",
        "sqlite3 data.db \"SELECT '\x00'\"": "Error: ",
    }.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command


def test_a_hash_is_text_not_a_comment(workdir: Path) -> None:
    assert run("echo C# > answer.txt", workdir).ok
    assert (workdir / "answer.txt").read_text() == "C#\n"


def test_echo_refuses_flags_it_cannot_honour(workdir: Path) -> None:
    for command in ("echo -e 'a\\tb' > answer.txt", "echo -n -E x > answer.txt"):
        result = run(command, workdir)
        assert not result.ok and "only -n is understood here" in result.output
    assert run("echo -5 > answer.txt", workdir).ok  # a negative answer, not a flag
    assert (workdir / "answer.txt").read_text() == "-5\n"


def test_the_guard_refuses_every_way_to_another_file(workdir: Path) -> None:
    refused = {
        "sqlite3 data.db \"VACUUM INTO 'copy.db'\"": "authorization denied",
        f"sqlite3 data.db \"VACUUM INTO '{workdir / 'abs.db'}'\"": "denied",
        'sqlite3 data.db "PRAGMA query_only = OFF; CREATE TABLE t (x)"': "authorized",
        "sqlite3 data.db \"PRAGMA temp_store_directory = '/tmp'\"": "not authorized",
        "sqlite3 data.db \"SELECT load_extension('x')\"": "not authorized",
        'sqlite3 data.db "SELECT length(hex(zeroblob(2000000)))"': "too big",
    }
    for command, words in refused.items():
        result = run(command, workdir)
        assert not result.ok and words in result.output, command
    assert run('sqlite3 data.db "PRAGMA query_only"', workdir).output == "1"
    assert sorted(path.name for path in workdir.iterdir()) == ["data.db"]


def test_an_unquoted_query_gets_a_hint(workdir: Path) -> None:
    result = run("sqlite3 data.db SELECT count(*) FROM doctors", workdir)
    assert (
        "'(' is not run" in result.output and "put the SQL in quotes" in result.output
    )


def test_the_text_protocol_cuts_long_output_once(workdir: Path) -> None:
    query = 'sqlite3 data.db "SELECT * FROM appointments"'
    whole = run(query, workdir) and bash_tool.run_command(query, workdir, limit=None)
    policy = Scripted([text_reply(query)] + [call("x")] * 3)
    run_ = experiments.run_text_protocol(policy, EASY, workdir)
    observation = run_.turns[0].answer
    elided = len(whole.output) - 4_000
    assert f"{elided} characters elided" in observation
    assert "[output cut to" not in observation  # the tool did not cut it first


def test_a_transcript_cut_short_says_so(tmp_path: Path) -> None:
    lines = [
        "[mhrl] harness=claude-code",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "a", "name": "Bash", "input": {}},
                        {"type": "tool_use", "id": "b", "name": "Bash", "input": {}},
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": [{"type": "text", "text": "one"}],
                        },
                        {"type": "tool_result", "tool_use_id": "zz", "content": "?"},
                    ]
                },
            }
        ),
    ]
    path = tmp_path / "cut.jsonl"
    path.write_text("\n".join(lines) + "\n")
    events = experiments.read_recorded(path)
    assert [(e.command, e.result, e.is_error) for e in events] == [
        ("{}", "one", False),
        ("{}", experiments.NO_RESULT, True),
    ]


def test_options_that_contradict_each_other_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_policy(monkeypatch, tiny_harness, Scripted([]))
    with pytest.raises(SystemExit):
        run_main(tiny_harness, "--task", "clinic-easy-001", "--held-out", "2")
    with pytest.raises(SystemExit):
        run_main(experiments, "--recorded", "--protocol", "text")


def test_the_last_unreadable_reply_is_not_called_an_answer(workdir: Path) -> None:
    broken = f"Try.</think>{START}[bash(command='ls'{END}{EOT}"
    policy = Scripted([broken] * 3)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        tiny_harness.run_rollout(policy, EASY, workdir, show=report.turn_printer(False))
    text = printed.getvalue()
    assert "the last one allowed" in text
    assert "this is the final answer" not in text
