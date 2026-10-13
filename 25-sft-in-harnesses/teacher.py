# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The answer-key teacher: perfect demonstrations, in each harness's own format.

The article's SFT stage imitated a 27B teacher's successful rollouts. A 27B
model does not fit beside everything else on a laptop, but the project has
something better for a demonstration: the task file knows the gold SQL. So
this scripted policy plays the model's part inside the real harness and does
what a careful analyst does (look at the tables, run the right query, save
the answer) in whatever action format that harness offers. It is the
project's ``AnswerKeyTeacher`` (``mhrl/engine/oracle.py``), copied with its
comments, reading this chapter's task file.

A reply is the text the model would write after the generation prompt's
``<think>``: a short reason, ``</think>``, then one native tool call in
LFM2.5's own syntax (``<|tool_call_start|>[Bash(command='...')]
<|tool_call_end|>``). The engine (``engine.py --oracle``) cuts that text into
tokens with the student's tokenizer, exactly as if the model had sampled it,
so the capture proxy records each demonstration in the student's own tokens.

The path varies by task, four ways: ``direct`` (tables, query, save),
``peek`` (tables, three rows, query, save), ``tables`` (list the tables,
their columns, query, save) and ``verify`` (tables, query, save, read the
file back). mini-swe-agent ends a run when its finish marker is the FIRST
line of a command's output, so on the last step the marker is chained in
front: ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo "8" > answer.txt``.

Run ``uv run teacher.py`` to print every reply of one demonstration in both
training harnesses' formats (``--task`` picks the task). Only Python's
standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from tasks import ANSWER_FILE, DATABASE_FILE, TRAIN_FILE, Task, load_tasks

SHELL_TOOL_NAMES = ("Bash", "bash", "shell", "run_shell_command", "execute_bash")
COMMAND_PARAMS = ("command", "cmd", "script")
SUBMIT_PATTERN = re.compile(
    r"echo\s+['\"]?(COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT)"
)
FALLBACK_REPLY = "Ready.</think>OK."  # a call with no task in it: a title, a probe
PATTERNS = ("direct", "peek", "tables", "verify")
TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
THINK_CLOSE = "</think>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
ASSISTANT = "assistant"
SHELL_SPECIALS = ("\\", '"', "$", "`")  # escaped inside a double-quoted argument
PEEK_ROWS = 3
DEFAULT_TASK = "library-easy-001"  # the first task collect.py's balanced pick takes
# What the two training harnesses offer the model (chapter 23's wiring): a
# shell tool, and mini-swe-agent's prompt naming its finish marker.
CLAUDE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }
]
MINI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
MINI_PROMPT = "When you are done, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def text_of(message: dict[str, Any]) -> str:
    """Flatten a message's content to plain text.

    Args:
        message: One chat message.

    Returns:
        Its text (the text parts of a list, joined by spaces).
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def shell_quote_double(text: str) -> str:
    """Quote text for a double-quoted shell argument.

    Args:
        text: Raw text (an SQL query or an answer).

    Returns:
        The text with ``\\``, ``"``, ``$`` and backticks escaped.
    """
    for char in SHELL_SPECIALS:
        text = text.replace(char, "\\" + char)
    return text


def format_value(value: Any) -> str:
    """Write one argument exactly as LFM2.5's chat template does.

    The project's ``format_value`` (``mhrl/engine/toolcalls.py``).

    Args:
        value: The argument.

    Returns:
        Its source form: single-quoted, escaped text; JSON for a list or a
        dict; ``str()`` for anything else.
    """
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f"'{escaped}'"
    if isinstance(value, dict | list):
        return json.dumps(value)
    return str(value)


def format_call(name: str, arguments: dict[str, Any]) -> str:
    """Write one tool call in LFM2.5's syntax, markers included.

    The project's ``format_call`` (``mhrl/engine/toolcalls.py``).

    Args:
        name: The tool's name.
        arguments: Its arguments, by name.

    Returns:
        Such as ``<|tool_call_start|>[Bash(command='ls')]<|tool_call_end|>``.
    """
    pairs = ", ".join(f"{key}={format_value(val)}" for key, val in arguments.items())
    return f"{TOOL_CALL_START}[{name}({pairs})]{TOOL_CALL_END}"


def pattern_of(task: Task) -> str:
    """Pick the task's path: the same task always gets the same one.

    Args:
        task: The task.

    Returns:
        One of ``PATTERNS``, from the sum of the task id's character codes.
    """
    return PATTERNS[sum(map(ord, task.task_id)) % len(PATTERNS)]


class AnswerKeyTeacher:
    """A scripted policy that solves each task the way a careful analyst would."""

    def __init__(self, tasks: list[Task]) -> None:
        """Index tasks by question text.

        Args:
            tasks: Tasks the teacher may be asked about.
        """
        self.by_question = {task.question: task for task in tasks}

    @classmethod
    def from_train_file(cls) -> AnswerKeyTeacher:
        """Load the training tasks (the teacher never demonstrates a test task).

        Returns:
            The teacher.
        """
        return cls(load_tasks(TRAIN_FILE))

    def find_task(self, messages: list[dict[str, Any]]) -> Task | None:
        """Find the task whose question appears in the conversation.

        Args:
            messages: The request's messages.

        Returns:
            The task, or None for auxiliary calls (titles, summaries, probes).
        """
        text = "\n".join(text_of(m) for m in messages if m.get("role") != ASSISTANT)
        for question, task in self.by_question.items():
            if question in text:
                return task
        return None

    def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> str:
        """Write the next reply (the text after the generation prompt's ``<think>``).

        Args:
            messages: The request's messages.
            tools: The request's tools (None for text-action harnesses).

        Returns:
            Reply text: a short reason, ``</think>``, then the action.
        """
        task = self.find_task(messages)
        if task is None:
            return FALLBACK_REPLY
        # The step is how many replies the conversation already holds.
        step = sum(1 for m in messages if m.get("role") == ASSISTANT)
        prompt_text = "\n".join(text_of(m) for m in messages)
        submit = SUBMIT_PATTERN.search(prompt_text)
        submit_command = f"echo {submit.group(1)}" if submit else None
        if tools:
            return self.tool_reply(task, step, tools, submit_command)
        return self.text_reply(task, step, submit_command)

    @staticmethod
    def steps(task: Task) -> list[tuple[str, str]]:
        """The (reason, shell command) steps of one task's demonstration.

        Demonstrations deliberately vary in length. A first version always did
        tables -> query -> write -> finish, and the student learned "the fourth
        reply is the final answer" by POSITION: after one extra query it
        finished without writing answer.txt (the project's dry run: Claude
        Code 100% -> 17%). Varying the path teaches the dependency instead:
        finish only after the answer is saved.

        Args:
            task: The task.

        Returns:
            The steps, in order; the last one writes or checks the answer.
        """
        db = DATABASE_FILE
        out = ANSWER_FILE
        query = " ".join(task.gold_sql.split())
        tables = list(dict.fromkeys(TABLE_PATTERN.findall(task.gold_sql)))
        main = tables[0] if tables else ""
        answer = shell_quote_double(task.answer)
        look = (
            "I should see the tables and columns before writing SQL.",
            f"sqlite3 {db} .schema",
        )
        run = (
            "Now I can answer the question with one query.",
            f'sqlite3 {db} "{shell_quote_double(query)}"',
        )
        save = (
            f"The result is {task.answer}. I will save it to {out} before finishing.",
            f'echo "{answer}" > {out}',
        )
        pattern = pattern_of(task)
        if pattern == "peek" and main:
            peek = (
                f"Let me look at a few rows of {main} to see the values.",
                f'sqlite3 -header {db} "SELECT * FROM {main} LIMIT {PEEK_ROWS}"',
            )
            return [look, peek, run, save]
        if pattern == "tables" and tables:
            listing = (
                "First, which tables are in this database?",
                f"sqlite3 {db} .tables",
            )
            detail = (
                "These are the tables I need. Let me see their columns.",
                f'sqlite3 {db} ".schema {" ".join(tables)}"',
            )
            return [listing, detail, run, save]
        if pattern == "verify":
            check = (f"Let me check that {out} holds the answer.", f"cat {out}")
            return [look, run, save, check]
        return [look, run, save]

    def tool_reply(
        self,
        task: Task,
        step: int,
        tools: list[dict[str, Any]],
        submit_command: str | None,
    ) -> str:
        """A reply for a native tool-calling harness (Claude Code, mini-swe-agent).

        Args:
            task: The task.
            step: Replies already given.
            tools: The request's tools.
            submit_command: The harness's own "I am done" command, if its
                prompt names one (mini-swe-agent). It goes FIRST on the last
                step: mini-swe-agent only sees a submission when the marker is
                the first line of output (``echo MARKER && git diff`` upstream).

        Returns:
            A reason plus one tool call, or a final short answer.
        """
        steps = self.steps(task)
        if step >= len(steps):
            return (
                f"The answer is saved.{THINK_CLOSE}I saved the answer to "
                f"{ANSWER_FILE}: {task.answer}"
            )
        reasoning, command = steps[step]
        if step == len(steps) - 1 and submit_command:
            command = f"{submit_command} && {command}"
        name, arguments = self.shell_call(tools, command, reasoning)
        if name is None:
            return f"{reasoning}{THINK_CLOSE}The answer is {task.answer}."
        return f"{reasoning}{THINK_CLOSE}{format_call(name, arguments)}"

    @staticmethod
    def shell_call(
        tools: list[dict[str, Any]], command: str, reasoning: str
    ) -> tuple[str | None, dict[str, Any]]:
        """Pick the harness's shell tool and fill its required arguments.

        Args:
            tools: The request's tools.
            command: The shell command to run.
            reasoning: Used for any required free-text argument.

        Returns:
            The tool name (None if the harness has no shell tool) and arguments.
        """
        by_name = {
            (tool.get("function", tool)).get("name"): tool.get("function", tool)
            for tool in tools
        }
        for wanted in SHELL_TOOL_NAMES:
            function = by_name.get(wanted)
            if function is None:
                continue
            schema = function.get("parameters") or {}
            properties = schema.get("properties") or {}
            key = next((p for p in COMMAND_PARAMS if p in properties), "command")
            arguments: dict[str, Any] = {key: command}
            for required in schema.get("required") or []:
                if required not in arguments:
                    arguments[required] = reasoning
            return wanted, arguments
        return None, {}

    def text_reply(self, task: Task, step: int, submit_command: str | None) -> str:
        """A reply for a text-action harness (one bash block per reply).

        Args:
            task: The task.
            step: Replies already given.
            submit_command: The harness's own "I am done" command, if any.

        Returns:
            A reason plus a THOUGHT and exactly one bash block.
        """
        steps = self.steps(task)
        reasoning, command = steps[min(step, len(steps) - 1)]
        if step >= len(steps) - 1 and submit_command:
            command = f"{submit_command} && {command}"
        block = f"```bash\n{command}\n```"
        return f"{reasoning}{THINK_CLOSE}THOUGHT: {reasoning}\n\n{block}"


def demonstration(
    teacher: AnswerKeyTeacher,
    task: Task,
    tools: list[dict[str, Any]],
    system: str,
) -> list[str]:
    """Play one whole demonstration against itself, without a harness.

    Each reply is appended to the conversation as the harness would append
    it, so the next call sees one more reply; tool results are left out.

    Args:
        teacher: The teacher.
        task: The task.
        tools: The tools the harness offers.
        system: The harness's system prompt (it may name a finish marker).

    Returns:
        Every reply, in order, up to the one that calls no tool or finishes.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.question},
    ]
    replies: list[str] = []
    for _ in teacher.steps(task) + [("", "")]:
        reply = teacher(messages, tools)
        replies.append(reply)
        messages.append({"role": ASSISTANT, "content": reply})
        if TOOL_CALL_START not in reply or SUBMIT_PATTERN.search(reply):
            break
    return replies


def main() -> None:
    """Print one task's demonstration as Claude Code and mini-swe-agent see it.

    Raises:
        SystemExit: If the task is not a training task.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default=DEFAULT_TASK, help="a training task id")
    args = parser.parse_args()
    tasks = load_tasks(TRAIN_FILE)
    found = [task for task in tasks if task.task_id == args.task]
    if not found:
        raise SystemExit(f"no training task {args.task!r}; try {tasks[0].task_id}")
    task = found[0]
    teacher = AnswerKeyTeacher(tasks)
    counts = {name: 0 for name in PATTERNS}
    for each in tasks:
        counts[pattern_of(each)] += 1
    print(f"teacher.py · the answer-key teacher, on {task.task_id}")
    print(f"  question  {task.question}")
    print(f"  gold SQL  {task.gold_sql}")
    print(f"  answer    {task.answer}")
    print(f"  path      {pattern_of(task)} ({len(teacher.steps(task))} commands)")
    shares = ", ".join(f"{name} {count}" for name, count in counts.items())
    print(f"  paths over the {len(tasks)} training tasks: {shares}")
    for harness, tools, system in (
        ("claude-code", CLAUDE_TOOLS, "You are Claude Code."),
        ("mini-swe-agent", MINI_TOOLS, MINI_PROMPT),
    ):
        print(f"\n{harness}: the replies the engine turns into tokens")
        for number, reply in enumerate(
            demonstration(teacher, task, tools, system), start=1
        ):
            print(f"  [{number}] {reply}")


if __name__ == "__main__":
    main()
