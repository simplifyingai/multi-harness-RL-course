# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""A tiny harness: the model, one bash tool, and a loop, all in one process.

Each call, the harness renders the conversation with the model's chat
template (which prints the tool's JSON schema into the system prompt), lets
the model write a reply with its special tokens kept, and splits the reply
into reasoning, visible text and tool calls (``toolcalls.py``, the project's
parser). It runs each call with ``bash_tool.py`` (sqlite3, ls and
``echo > answer.txt`` only, in the task folder) and appends the output as a
tool message: the observation. It stops when a reply has no tool call (the
model's final answer), after 8 calls, or after 3 unreadable replies in a
row, and then chapter 1's grader reads answer.txt.

Run it with ``uv run tiny_harness.py`` (the question ``clinic-easy-001``),
``--task <id>`` for another held-out question, ``--held-out 5`` for the first
five, ``--full`` to print every reply whole and ``--fast`` for Liquid's
export. Work folders and every call go under ``$COURSE_HOME/17-tiny-harness``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import report
from bash_tool import ToolResult, run_command
from grader import Grade, grade
from lfm_files import (
    EVAL_TEMPERATURE,
    MAX_CALLS,
    MAX_UNREADABLE,
    OUTPUT_CAP,
    SAMPLE_SEED,
    pick_model,
    rollout_path,
    short_path,
    work_folder,
)
from policy import FINISH_LENGTH, Message, Policy, Reply, Tool
from tasks import Task, find_task, load_tasks, prepare_workdir, task_instruction
from toolcalls import END_OF_TURN, THINK_OPEN, ParsedReply, ToolCall, split_reply

SCRIPT = "tiny_harness.py"
TITLE = "one model, one tool, one loop"
DEFAULT_TASK = "clinic-easy-001"
LABEL = "harness"  # this script's work and rollout folders
TOOL_NAME = "bash"
COMMAND = "command"  # the tool's one argument
SYSTEM, USER, ASSISTANT, TOOL_ROLE = "system", "user", "assistant", "tool"
SYSTEM_PROMPT = (
    "You are a data analyst. You work in a folder that holds a SQLite database, "
    "and you can run commands there with the {tool} tool: sqlite3 to query the "
    "database, ls to list the folder, and echo to write the answer file. When "
    "the answer file is written, reply without a tool call to finish."
)
TOOL_DESCRIPTION = (
    "Run one command in the task folder. Allowed: sqlite3 data.db "
    '"<SQL, .tables or .schema>", ls, and echo "<answer>" > answer.txt.'
)
# What the harness tells the model after a reply it could not read.
UNREADABLE_NOTE = (
    "Your last reply could not be read ({why}). To run a command, call the "
    "{tool} tool with one command. To finish, reply without a tool call."
)
WHY_CUT_OFF = "it was cut off by the length cap"
WHY_UNPARSEABLE = "its tool call could not be parsed"
# Claude Code 2.1.290's words for a tool it does not have (the project's
# recorded run, data/clinic-medium-001.claude-code.jsonl).
NO_SUCH_TOOL = "Error: No such tool available: {name}"
CASE_HINT = ". Tool names are case-sensitive: call {offered} instead."
NO_COMMAND = "Error: the tool needs a command, as text"
# The stop rules, each as "short name: why".
STOP_NO_CALL = "no tool call: the reply is the model's final answer"
STOP_BUDGET = f"{MAX_CALLS} calls: the budget is used up"
STOP_UNREADABLE = (
    f"{MAX_UNREADABLE} unreadable: {MAX_UNREADABLE} replies in a row could not be read"
)


@dataclass(frozen=True)
class Action:
    """One tool call and what running it gave back.

    Attributes:
        call: The call, as parsed from the reply.
        result: The observation and whether the command succeeded.
    """

    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class Turn:
    """One call: the reply, how it was read, and what the harness did.

    Attributes:
        number: 1 for the first call.
        reply: The reply, as the model wrote it.
        parsed: Its reasoning, visible text and tool calls.
        actions: Each tool call run, with its observation.
        why: Why the reply could not be read, or None when it could.
        note: What the harness told the model about it (None for the last
            unreadable reply, after which the loop stops).
    """

    number: int
    reply: Reply
    parsed: ParsedReply
    actions: list[Action]
    why: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Rollout:
    """One whole attempt at a question.

    Attributes:
        task: The question.
        tool_name: The name the tool was offered under.
        turns: Every call, in order.
        stop: Which stop rule ended it.
        grade: Chapter 1's verdict on answer.txt.
        workdir: The folder it worked in.
        messages: The whole conversation, as the model read it last.
    """

    task: Task
    tool_name: str
    turns: list[Turn]
    stop: str
    grade: Grade
    workdir: Path
    messages: list[Message]


def bash_schema(name: str = TOOL_NAME) -> Tool:
    """Describe the tool in JSON schema, as OpenAI's API (and the template) want.

    Args:
        name: The name to offer it under (``bash``, or ``Bash`` for the
            experiment).

    Returns:
        The tool's schema: a name, a description and one text argument.
    """
    argument = {"type": "string", "description": "The command to run"}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {COMMAND: argument},
                "required": [COMMAND],
            },
        },
    }


def no_such_tool(name: str, offered: list[str]) -> str:
    """Say a tool does not exist, in Claude Code's words.

    Args:
        name: The tool the model called.
        offered: The names of the tools offered.

    Returns:
        The error, with a hint when only the letter case is wrong.
    """
    message = NO_SUCH_TOOL.format(name=name)
    same = [tool for tool in offered if tool.lower() == name.lower()]
    return message + CASE_HINT.format(offered=same[0]) if same else message


def execute(call: ToolCall, tools: list[Tool], workdir: Path) -> ToolResult:
    """Run one tool call, if it names a tool that was offered.

    Args:
        call: The parsed call.
        tools: The tools offered.
        workdir: The task folder.

    Returns:
        The observation the model will read next.
    """
    offered = [tool["function"]["name"] for tool in tools]
    if call.name not in offered:  # an exact match: names are case-sensitive
        return ToolResult(no_such_tool(call.name, offered), ok=False)
    command = call.arguments.get(COMMAND)
    if not isinstance(command, str):
        return ToolResult(NO_COMMAND, ok=False)
    return run_command(command, workdir)


def unreadable(parsed: ParsedReply, reply: Reply) -> str | None:
    """Say why a reply with no tool call cannot be read as an answer, if so.

    Args:
        parsed: The split reply.
        reply: The reply.

    Returns:
        Why it is unreadable, or None when it is a plain final answer.
    """
    if parsed.calls:
        return None
    if reply.finish_reason == FINISH_LENGTH:
        return WHY_CUT_OFF
    if parsed.opened:
        return WHY_UNPARSEABLE
    return None


def start_messages(task: Task, tool_name: str = TOOL_NAME) -> list[Message]:
    """Write the conversation's first two messages: the rules and the task.

    Args:
        task: The question.
        tool_name: The name the tool is offered under (the rules name it).

    Returns:
        A system message and a user message (chapter 1's instruction).
    """
    return [
        {"role": SYSTEM, "content": SYSTEM_PROMPT.format(tool=tool_name)},
        {"role": USER, "content": task_instruction(task)},
    ]


def run_rollout(
    policy: Policy,
    task: Task,
    workdir: Path,
    tool_name: str = TOOL_NAME,
    show: Callable[[Turn], None] | None = None,
) -> Rollout:
    """Let the model work on one question until a stop rule ends the attempt.

    Args:
        policy: The model.
        task: The question.
        workdir: A fresh folder holding the database as data.db.
        tool_name: The name to offer the tool under.
        show: Called with each turn as soon as it is done, to print it.

    Returns:
        The attempt, graded.
    """
    tools = [bash_schema(tool_name)]
    messages = start_messages(task, tool_name)
    turns: list[Turn] = []
    in_a_row = 0  # unreadable replies since the last readable one
    stop = STOP_BUDGET
    for number in range(1, MAX_CALLS + 1):
        prompt = policy.render(messages, tools)
        reply = policy.complete(prompt)
        parsed = split_reply(reply.text, tools)
        # The reply goes back word for word: reasoning, markers and all.
        content = THINK_OPEN + reply.text.removesuffix(END_OF_TURN)
        messages.append({"role": ASSISTANT, "content": content})
        actions = [Action(call, execute(call, tools, workdir)) for call in parsed.calls]
        for action in actions:  # each observation is a tool message
            messages.append({"role": TOOL_ROLE, "content": action.result.output})
        why = unreadable(parsed, reply)
        in_a_row = in_a_row + 1 if why else 0
        note = None
        if why and in_a_row < MAX_UNREADABLE:
            note = UNREADABLE_NOTE.format(why=why, tool=tool_name)
            messages.append({"role": USER, "content": note})
        turn = Turn(number, reply, parsed, actions, why, note)
        turns.append(turn)
        if show is not None:
            show(turn)
        if not actions and not why:
            stop = STOP_NO_CALL
            break
        if in_a_row == MAX_UNREADABLE:
            stop = STOP_UNREADABLE
            break
    verdict = grade(task, workdir)  # chapter 1's grader reads answer.txt
    return Rollout(task, tool_name, turns, stop, verdict, workdir, messages)


def save_rollout(rollout: Rollout, path: Path) -> None:
    """Write one attempt, every call and the whole conversation, as JSON.

    Args:
        rollout: The attempt.
        path: The file (its folder is made).

    Raises:
        OSError: If the file cannot be written.
    """
    record = {
        "task_id": rollout.task.task_id,
        "tool_name": rollout.tool_name,
        "stop": rollout.stop,
        "grade": asdict(rollout.grade),
        "calls": [asdict(turn) for turn in rollout.turns],
        "messages": rollout.messages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, ensure_ascii=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Loaded:
    """The policy the scripts run, with what they print about it.

    Attributes:
        policy: The model.
        line: Which build it is and how long it took to load.
        peak: Returns the most memory the model has held, in bytes (None
            when there is no model to measure).
    """

    policy: Policy
    line: str
    peak: Callable[[], int | None]


def load_policy(fast: bool, temperature: float) -> Loaded:
    """Load the 4-bit model (the only step that needs MLX).

    Args:
        fast: Use Liquid's export even when ours is built.
        temperature: 0 for the likeliest token every time.

    Returns:
        The loaded policy.
    """
    from lfm import LfmPolicy, peak_memory  # MLX loads here, so the loop runs without

    folder, which = pick_model(fast)
    policy = LfmPolicy(folder, OUTPUT_CAP, temperature, SAMPLE_SEED)
    line = f"{short_path(folder)} ({which}), loaded in {policy.seconds:.1f} s"
    return Loaded(policy, line, peak_memory)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", default=None, help=f"a held-out question ({DEFAULT_TASK})"
    )
    parser.add_argument(
        "--held-out", type=int, default=None, help="the first N held-out questions"
    )
    parser.add_argument("--full", action="store_true", help="print replies whole")
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    parser.add_argument(
        "--temperature",
        type=float,
        default=EVAL_TEMPERATURE,
        help="0: greedy; above 0, drawn with one seed for the whole run",
    )
    args = parser.parse_args()
    if args.held_out is not None and args.task is not None:
        parser.error("--task and --held-out pick the questions two ways: use one")
    if args.held_out is not None and args.held_out < 1:
        parser.error("--held-out must be at least 1")
    args.task = args.task or DEFAULT_TASK
    if args.temperature < 0:
        parser.error("--temperature must be 0 or more")
    return args


def main() -> None:
    """Run the harness on one question, or on the first few held-out ones."""
    args = parse_args()
    one = args.held_out is None
    tasks = [find_task(args.task)] if one else load_tasks()[: args.held_out]
    loaded = load_policy(args.fast, args.temperature)
    report.print_title(SCRIPT, TITLE)
    report.print_setup(loaded.line, tasks, args.temperature, TOOL_NAME)
    rollouts = []
    for task in tasks:
        workdir = prepare_workdir(task, work_folder(LABEL, task.task_id))
        if one:
            report.print_task(task, workdir)
            first = loaded.policy.render(start_messages(task), [bash_schema()])
            report.print_first_prompt(first)
            report.heading("the calls")
            show = report.turn_printer(args.full)
        else:
            report.print_task_line(task)
            show = report.print_turn_line
        rollout = run_rollout(loaded.policy, task, workdir, show=show)
        save_rollout(rollout, rollout_path(LABEL, task.task_id))
        report.print_grade(rollout, one)
        rollouts.append(rollout)
    if not one:
        report.print_summary(rollouts)
    report.print_cost(rollouts, loaded.peak(), rollout_path(LABEL, tasks[0].task_id))


if __name__ == "__main__":
    main()
