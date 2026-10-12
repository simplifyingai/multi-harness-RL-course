# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2==3.1.6",
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Two ways a harness and a model can disagree about the format of an action.

1. **A name**: the same harness and tool, offered as ``Bash`` instead of
   ``bash``. Tool names are matched exactly, so a call to ``bash`` comes back
   as "No such tool available" and spends one of the 8 calls. The project's
   recorded Claude Code run of the same model on the same question (bundled
   in ``data/``) is printed after it: there it happened three times.
2. **A protocol** (``--protocol text``): mini-swe-agent's old text protocol,
   as the project's ``configs/harnesses/mini_swe_agent_textbased.yaml`` set
   it up. No tool is offered; every reply must carry exactly one fenced bash
   block in its visible text, which the harness runs; a reply without one is
   a format error, and 3 format errors in a row abort the run
   (``RepeatedFormatError``). The prompts are that file's templates, word for
   word, rendered with Jinja as mini-swe-agent renders them.

Run it with ``uv run experiments.py`` (the name), ``uv run experiments.py
--protocol text`` (the protocol) or ``uv run experiments.py --recorded`` (the
recorded run alone, no model). ``--task`` picks another held-out question,
``--full`` prints every reply whole, ``--fast`` uses Liquid's export.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import report
from bash_tool import ToolResult, run_command
from grader import Grade, grade
from jinja2 import StrictUndefined, Template
from lfm_files import EVAL_TEMPERATURE, MAX_CALLS, rollout_path, work_folder
from policy import Message, Policy, Reply
from tasks import DATA_DIR, Task, find_task, prepare_workdir, task_instruction
from tiny_harness import Rollout, load_policy, run_rollout, save_rollout
from toolcalls import END_OF_TURN, ParsedReply, split_reply

SCRIPT = "experiments.py"
DEFAULT_TASK = "clinic-medium-001"  # the question of the recorded Claude Code run
NAME_LABEL = "Bash"  # the name experiment's work and rollout folders
TEXT_LABEL = "text"  # the protocol experiment's
NATIVE, TEXT = "native", "text"
OFFERED_NAME = "Bash"  # Claude Code's name for its shell tool
RECORDED_RUN = DATA_DIR / "clinic-medium-001.claude-code.jsonl"
NO_SUCH_TOOL_MARK = "No such tool available"
THINK_OPEN = "<think>"
SYSTEM, USER, ASSISTANT = "system", "user", "assistant"

# mini-swe-agent 2.4.6, as configs/harnesses/mini_swe_agent_textbased.yaml set
# it up: at most 8 calls, and the run aborts at the 3rd format error in a row.
MAX_FORMAT_ERRORS = 3
# The action: a fenced block (```bash, ```sh or a bare ```), on its own lines.
ACTION_REGEX = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)\n```", re.DOTALL)
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
EXIT_SUBMITTED = "Submitted"
EXIT_FORMAT = "RepeatedFormatError"
EXIT_LIMIT = "LimitsExceeded"
FORMAT_ERROR = "Expected exactly 1 action, found {count}."  # mini's own words

# The YAML's templates, word for word. The instance template's last block,
# <system_information> (your operating system's name and version), is left
# out: it would print your machine's details and changes nothing here.
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""  # noqa: E501
INSTANCE_TEMPLATE = """\
Please solve this task: {{task}}

You work in the folder that holds the task's files and can run bash commands there.
`sqlite3` is installed. Useful commands:

```bash
sqlite3 data.db ".tables"
sqlite3 data.db ".schema table_name"
sqlite3 -header -column data.db "SELECT * FROM table_name LIMIT 5;"
```

## Recommended Workflow

1. Look at the files in the folder and at the database schema.
2. Run SQL queries until you are sure of the answer.
3. Write ONLY the final answer to the file the task names, for example: `echo "42" > answer.txt`
4. Finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks, as in the example below
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to know which tables the database holds before I can answer the question.

```bash
sqlite3 data.db ".tables"
```
</example_response>"""  # noqa: E501
OBSERVATION_TEMPLATE = """\
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 4000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output, for example a query with LIMIT or a narrower SELECT.
</warning>
{%- set elided_chars = output.output | length - 4000 -%}
<output_head>
{{ output.output[:2000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-2000:] }}
</output_tail>
{%- endif -%}"""  # noqa: E501
FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and finish_reason == "length" -%}
Your previous response reached the output token limit before you produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format.
{%- else -%}
Format error:

<error>
{{error}}
</error>

Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
{%- endif %}"""  # noqa: E501


@dataclass(frozen=True)
class TextTurn:
    """One call under the text protocol.

    Attributes:
        number: 1 for the first call.
        reply: The reply, as the model wrote it.
        parsed: The reply split the engine's way: only ``content``, the
            visible text, reaches mini-swe-agent.
        actions: The fenced blocks found in the visible text.
        answer: What the harness sent back: an observation, or a format error.
        format_errors: Format errors in a row after this call (0 after a
            clean one).
        submitted: Whether this call was the finishing command.
    """

    number: int
    reply: Reply
    parsed: ParsedReply
    actions: list[str]
    answer: str
    format_errors: int
    submitted: bool = False


@dataclass(frozen=True)
class TextRun:
    """One whole attempt under the text protocol.

    Attributes:
        task: The question.
        turns: Every call.
        exit_status: ``Submitted``, ``RepeatedFormatError`` or
            ``LimitsExceeded``.
        grade: Chapter 1's verdict on answer.txt.
        messages: The whole conversation.
    """

    task: Task
    turns: list[TextTurn]
    exit_status: str
    grade: Grade
    messages: list[Message]


@dataclass(frozen=True)
class RecordedEvent:
    """One step of the recorded Claude Code run.

    Attributes:
        kind: ``call`` (the model used a tool), ``harness`` (Claude Code
            added a message of its own) or ``text`` (a reply in plain text).
        tool: The tool's name, for a call.
        command: What the call asked for (its command, or its whole input).
        result: The tool result, or the message's text.
        is_error: Whether Claude Code marked the result as an error.
    """

    kind: str
    tool: str = ""
    command: str = ""
    result: str = ""
    is_error: bool = False


def render(template: str, **values: Any) -> str:
    """Fill one of mini-swe-agent's Jinja templates, as it does.

    Args:
        template: The template text.
        **values: Its variables.

    Returns:
        The text.
    """
    return Template(template, undefined=StrictUndefined).render(**values)


def observation(result: ToolResult) -> str:
    """Write a command's output the way the text protocol's template does.

    Args:
        result: What the tool gave back.

    Returns:
        The observation: the return code and the output.
    """
    output = {"output": result.output, "returncode": 0 if result.ok else 1}
    return render(OBSERVATION_TEMPLATE, output=output | {"exception_info": ""})


def run_text_protocol(
    policy: Policy,
    task: Task,
    workdir: Path,
    show: Callable[[TextTurn], None] | None = None,
) -> TextRun:
    """Run mini-swe-agent's text protocol: one fenced bash block per reply.

    Args:
        policy: The model.
        task: The question.
        workdir: A fresh folder holding the database as data.db.
        show: Called with each turn as soon as it is done.

    Returns:
        The attempt, graded.
    """
    instance = render(INSTANCE_TEMPLATE, task=task_instruction(task))
    messages: list[Message] = [
        {"role": SYSTEM, "content": render(SYSTEM_TEMPLATE)},
        {"role": USER, "content": instance},
    ]
    turns: list[TextTurn] = []
    errors = 0
    exit_status = EXIT_LIMIT
    for number in range(1, MAX_CALLS + 1):
        reply = policy.complete(policy.render(messages, None))  # no tools offered
        parsed = split_reply(reply.text, None)
        content = THINK_OPEN + reply.text.removesuffix(END_OF_TURN)
        messages.append({"role": ASSISTANT, "content": content})
        # Only the visible text is searched: the engine hands mini-swe-agent
        # `content`, without the reasoning and without any tool-call block.
        actions = [found.strip() for found in ACTION_REGEX.findall(parsed.content)]
        submitted = False
        if len(actions) == 1:
            errors = 0
            submitted = actions[0] == SUBMIT_COMMAND
            answer = "" if submitted else observation(run_command(actions[0], workdir))
        else:
            errors += 1
            answer = render(
                FORMAT_ERROR_TEMPLATE,
                error=FORMAT_ERROR.format(count=len(actions)),
                actions=actions,
                finish_reason=reply.finish_reason,
            )
        if answer:
            messages.append({"role": USER, "content": answer})
        turn = TextTurn(number, reply, parsed, actions, answer, errors, submitted)
        turns.append(turn)
        if show is not None:
            show(turn)
        if submitted:
            exit_status = EXIT_SUBMITTED
            break
        if errors == MAX_FORMAT_ERRORS:
            exit_status = EXIT_FORMAT
            break
    return TextRun(task, turns, exit_status, grade(task, workdir), messages)


def read_recorded(path: Path = RECORDED_RUN) -> list[RecordedEvent]:
    """Read the recorded Claude Code run: each tool call, result and message.

    The file is Claude Code's own ``stream-json`` output, one JSON object per
    line, between the project's ``[mhrl]`` lines (the program and its
    command line), which are not JSON and are skipped.

    Args:
        path: The transcript.

    Returns:
        The steps, in order.

    Raises:
        OSError: If the file cannot be read.
    """
    events: list[RecordedEvent] = []
    pending: dict[str, RecordedEvent] = {}  # calls waiting for their result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        content = (record.get("message") or {}).get("content")
        if record.get("type") not in (ASSISTANT, USER) or not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                inputs = block.get("input") or {}
                command = inputs.get("command") or json.dumps(inputs)
                pending[block["id"]] = RecordedEvent("call", block["name"], command)
            elif block.get("type") == "tool_result":
                call = pending.pop(block["tool_use_id"])
                result = str(block.get("content", ""))
                events.append(
                    RecordedEvent(
                        "call",
                        call.tool,
                        call.command,
                        result,
                        bool(block.get("is_error")),
                    )
                )
            elif block.get("type") == "text":
                kind = "harness" if record["type"] == USER else "text"
                events.append(RecordedEvent(kind, result=block["text"]))
    return events


def run_name_experiment(
    policy: Policy, task: Task, full: bool
) -> tuple[Rollout, list[RecordedEvent]]:
    """Offer the tool as 'Bash', and read the recorded Claude Code run.

    Args:
        policy: The model.
        task: The question.
        full: Print each reply whole.

    Returns:
        Our attempt, and the recorded run's steps.
    """
    workdir = prepare_workdir(task, work_folder(NAME_LABEL, task.task_id))
    report.print_task(task, workdir)
    report.heading(f"the calls, with the tool offered as {OFFERED_NAME!r}")
    show = report.turn_printer(full)
    rollout = run_rollout(policy, task, workdir, OFFERED_NAME, show)
    save_rollout(rollout, rollout_path(NAME_LABEL, task.task_id))
    report.print_grade(rollout, True)
    report.print_name_summary(rollout, NO_SUCH_TOOL_MARK)
    return rollout, read_recorded()


def save_text_run(run: TextRun, path: Path) -> None:
    """Write one text-protocol attempt as JSON.

    Args:
        run: The attempt.
        path: The file (its folder is made).

    Raises:
        OSError: If the file cannot be written.
    """
    record = {
        "task_id": run.task.task_id,
        "exit_status": run.exit_status,
        "grade": {"correct": run.grade.correct, "reason": run.grade.reason},
        "messages": run.messages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--protocol",
        choices=(NATIVE, TEXT),
        default=NATIVE,
        help="native: the name experiment; text: mini-swe-agent's text protocol",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="a held-out question")
    parser.add_argument(
        "--recorded", action="store_true", help="print the recorded run only"
    )
    parser.add_argument("--full", action="store_true", help="print replies whole")
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    return parser.parse_args()


def main() -> None:
    """Run the chosen experiment and print it."""
    args = parse_args()
    if args.recorded:
        report.print_title(SCRIPT, "the recorded Claude Code run")
        report.print_recorded(read_recorded(), RECORDED_RUN, NO_SUCH_TOOL_MARK)
        return
    task = find_task(args.task)
    loaded = load_policy(args.fast, EVAL_TEMPERATURE)
    if args.protocol == NATIVE:
        report.print_title(SCRIPT, f"the tool offered as {OFFERED_NAME!r}")
        report.print_setup(loaded.line, [task], EVAL_TEMPERATURE, OFFERED_NAME)
        rollout, recorded = run_name_experiment(loaded.policy, task, args.full)
        report.print_recorded(recorded, RECORDED_RUN, NO_SUCH_TOOL_MARK)
        saved = rollout_path(NAME_LABEL, task.task_id)
        report.print_cost([rollout], loaded.peak(), saved)
        return
    report.print_title(SCRIPT, "mini-swe-agent's text protocol")
    report.print_text_setup(loaded.line, MAX_CALLS, MAX_FORMAT_ERRORS)
    workdir = prepare_workdir(task, work_folder(TEXT_LABEL, task.task_id))
    report.print_task(task, workdir)
    report.print_text_rule(render(SYSTEM_TEMPLATE))
    report.heading("the calls")
    show = report.text_turn_printer(args.full, MAX_FORMAT_ERRORS)
    run = run_text_protocol(loaded.policy, task, workdir, show)
    save_text_run(run, rollout_path(TEXT_LABEL, task.task_id))
    report.print_text_result(run, MAX_CALLS)
    report.print_text_cost(run, loaded.peak(), rollout_path(TEXT_LABEL, task.task_id))


if __name__ == "__main__":
    main()
