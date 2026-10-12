# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Chapter 17's tiny harness, speaking HTTP: a base URL and a key, nothing else.

In chapter 17 the harness and the model shared one process, and the harness
read every token. Here the harness only knows an address. Each call it POSTs
the conversation and the bash tool's JSON schema to
``<base URL>/chat/completions`` in OpenAI's format, and gets back a message:
visible text, or ``tool_calls`` (an id, a name, and the arguments as a JSON
string). It runs each call with chapter 17's ``bash_tool.py``, appends the
message exactly as it came, then one ``tool`` message per call carrying that
call's id and the output, and calls again. Stop rules and grading are
chapter 17's: a reply with no tool call is the final answer; 8 calls at most;
3 unreadable replies in a row end the attempt.

From outside, a reply cut off by the length cap is still visible (its
``finish_reason`` is ``length``), but a tool call the model wrote and the
engine could not parse just disappears: what is left is a reply with no
call and no text, which this harness also counts as unreadable.

The engine's receipts let the harness check one more thing at the end: did
every call's prompt (``prompt_token_ids``) begin with the previous call's
prompt and reply (``token_ids``)?

Run it with ``uv run tiny_harness_http.py`` against ``engine.py`` on port
8100, ``--base-url`` for another address, ``--task <id>`` or ``--held-out 5``.
It needs only Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bash_tool import ToolResult, run_command
from grader import Grade, grade
from lfm_files import (
    EVAL_TEMPERATURE,
    MAX_CALLS,
    MAX_UNREADABLE,
    OUTPUT_CAP,
    rollout_path,
    short_path,
    work_folder,
)
from tasks import Task, find_task, load_tasks, prepare_workdir, task_instruction

SCRIPT = "tiny_harness_http.py"
DEFAULT_BASE_URL = "http://127.0.0.1:8100/v1"  # OpenAI's style: the root ends in /v1
DEFAULT_API_KEY = "any-key"  # the engine listens on this machine only: any key works
MODEL_NAME = "lfm2.5-2.6b"
CHAT_PATH = "/chat/completions"
# The project's harnesses wait this long for one call: with one worker thread
# and four rollouts at once, a call can wait minutes in the engine's queue.
REQUEST_TIMEOUT_S = 600
DEFAULT_TASK = "clinic-easy-001"
LABEL = "http"  # this script's work and rollout folders
TOOL_NAME = "bash"
COMMAND = "command"  # the tool's one argument
SYSTEM, USER, ASSISTANT, TOOL_ROLE = "system", "user", "assistant", "tool"
FINISH_LENGTH = "length"
# Chapter 17's words, unchanged.
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
UNREADABLE_NOTE = (
    "Your last reply could not be read ({why}). To run a command, call the "
    "{tool} tool with one command. To finish, reply without a tool call."
)
WHY_CUT_OFF = "it was cut off by the length cap"
WHY_EMPTY = "it held no tool call and no text"
NO_SUCH_TOOL = "Error: No such tool available: {name}"
NO_COMMAND = "Error: the tool needs a command, as text"
BAD_ARGUMENTS = "Error: the tool call's arguments are not a JSON object"
STOP_NO_CALL = "no tool call: the reply is the model's final answer"
STOP_BUDGET = f"{MAX_CALLS} calls: the budget is used up"
STOP_UNREADABLE = (
    f"{MAX_UNREADABLE} unreadable: {MAX_UNREADABLE} replies in a row could not be read"
)
INDENT = "  "
LABEL_WIDTH = 13
OUTPUT_LINES = 4  # lines of an observation printed per call

Chat = Callable[[dict[str, Any]], dict[str, Any]]  # request body in, response out


class EngineError(Exception):
    """Raised when the engine cannot be reached or refuses a request.

    Attributes:
        status: The HTTP status of a refusal, or None when nobody answered.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        """Keep the message and the status.

        Args:
            message: What happened.
            status: The HTTP status, or None.
        """
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Call:
    """One model call: what came back and what the harness did with it.

    Attributes:
        number: 1 for the first call.
        response: The engine's whole answer, receipts included.
        observations: Each tool call run, with its result.
        why: Why the reply could not be read, or None.
    """

    number: int
    response: dict[str, Any]
    observations: list[tuple[dict[str, Any], ToolResult]]
    why: str | None


@dataclass(frozen=True)
class Rollout:
    """One whole attempt at a question.

    Attributes:
        task: The question.
        calls: Every call, in order.
        stop: Which stop rule ended it.
        grade: Chapter 1's verdict on answer.txt.
        messages: The conversation, as the harness kept it.
    """

    task: Task
    calls: list[Call]
    stop: str
    grade: Grade
    messages: list[dict[str, Any]]


def http_chat(base_url: str, api_key: str) -> Chat:
    """Make the one function the harness needs: POST a chat request.

    Args:
        base_url: The engine's root, such as ``http://127.0.0.1:8100/v1``.
        api_key: Sent as a bearer token, as OpenAI's clients send it.

    Returns:
        A function from a request body to the response body.
    """
    url = base_url.rstrip("/") + CHAT_PATH

    def chat(body: dict[str, Any]) -> dict[str, Any]:
        """POST one request and read the answer.

        Args:
            body: The request.

        Returns:
            The response.

        Raises:
            EngineError: If the engine is unreachable or answers with an error.
        """
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:  # the engine answered: a refusal
            detail = error.read().decode("utf-8", errors="replace")
            raise EngineError(f"HTTP {error.code}: {detail}", error.code) from error
        except urllib.error.URLError as error:  # nobody answered at all
            raise EngineError(f"cannot reach {url}: {error.reason}") from error
        except OSError as error:  # a timeout, or a connection dropped midway
            raise EngineError(f"no answer from {url}: {error}") from error
        except json.JSONDecodeError as error:  # it answered, but not in JSON
            raise EngineError(f"{url} did not answer in JSON: {error}") from error

    return chat


def bash_schema(name: str = TOOL_NAME) -> dict[str, Any]:
    """Describe the tool in JSON schema, as OpenAI's API wants (chapter 17's).

    Args:
        name: The name to offer it under.

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


def tool_message(call_id: str, output: str) -> dict[str, Any]:
    """Write a tool's result as OpenAI's API wants it: tied to its call by id.

    Args:
        call_id: The id of the call it answers.
        output: What the tool printed.

    Returns:
        ``{"role": "tool", "tool_call_id": ..., "content": ...}``.
    """
    return {"role": TOOL_ROLE, "tool_call_id": call_id, "content": output}


def start_messages(task: Task) -> list[dict[str, Any]]:
    """Write the conversation's first two messages: the rules and the task.

    Args:
        task: The question.

    Returns:
        A system message and a user message (chapter 1's instruction).
    """
    return [
        {"role": SYSTEM, "content": SYSTEM_PROMPT.format(tool=TOOL_NAME)},
        {"role": USER, "content": task_instruction(task)},
    ]


def execute(
    call: dict[str, Any], tools: list[dict[str, Any]], workdir: Path
) -> ToolResult:
    """Run one OpenAI tool call, if it names a tool that was offered.

    Args:
        call: ``{"id", "type", "function": {"name", "arguments"}}``.
        tools: The tools offered.
        workdir: The task folder.

    Returns:
        The observation the model will read next.
    """
    function = call.get("function") or {}
    name = function.get("name")
    offered = [tool["function"]["name"] for tool in tools]
    if name not in offered:  # an exact match: names are case-sensitive
        return ToolResult(NO_SUCH_TOOL.format(name=name), ok=False)
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return ToolResult(BAD_ARGUMENTS, ok=False)
    if not isinstance(arguments, dict):
        return ToolResult(BAD_ARGUMENTS, ok=False)
    command = arguments.get(COMMAND)
    if not isinstance(command, str):
        return ToolResult(NO_COMMAND, ok=False)
    return run_command(command, workdir)


def unreadable(message: dict[str, Any], finish_reason: str) -> str | None:
    """Say why a reply with no tool call cannot be read as an answer, if so.

    Args:
        message: The assistant message.
        finish_reason: Why the reply ended.

    Returns:
        Why it is unreadable, or None (it has a call, or it is an answer).
    """
    if message.get("tool_calls"):
        return None
    if finish_reason == FINISH_LENGTH:
        return WHY_CUT_OFF
    if not message.get("content"):
        return WHY_EMPTY
    return None


def run_rollout(
    chat: Chat,
    task: Task,
    workdir: Path,
    show: Callable[[Call], None] | None = None,
) -> Rollout:
    """Let the model work on one question, through the engine, until a stop rule.

    Args:
        chat: Sends one request (``http_chat``).
        task: The question.
        workdir: A fresh folder holding the database as data.db.
        show: Called with each call as soon as it is done, to print it.

    Returns:
        The attempt, graded.
    """
    tools = [bash_schema()]
    messages = start_messages(task)
    calls: list[Call] = []
    in_a_row = 0
    stop = STOP_BUDGET
    for number in range(1, MAX_CALLS + 1):
        response = chat(
            {
                "model": MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "temperature": EVAL_TEMPERATURE,
                "max_tokens": OUTPUT_CAP,
            }
        )
        choice = response["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []
        # A black-box harness keeps what the API gave it: text and JSON calls.
        kept = {"role": ASSISTANT, "content": message.get("content")}
        if tool_calls:
            kept["tool_calls"] = tool_calls
        messages.append(kept)
        observations = []
        for call in tool_calls:  # each result names the call it answers
            result = execute(call, tools, workdir)
            observations.append((call, result))
            messages.append(tool_message(call["id"], result.output))
        why = unreadable(message, choice["finish_reason"])
        in_a_row = in_a_row + 1 if why else 0
        if why and in_a_row < MAX_UNREADABLE:
            note = UNREADABLE_NOTE.format(why=why, tool=TOOL_NAME)
            messages.append({"role": USER, "content": note})
        done = Call(number, response, observations, why)
        calls.append(done)
        if show is not None:
            show(done)
        if not tool_calls and not why:
            stop = STOP_NO_CALL
            break
        if in_a_row == MAX_UNREADABLE:
            stop = STOP_UNREADABLE
            break
    return Rollout(task, calls, stop, grade(task, workdir), messages)


def continued(calls: list[Call]) -> tuple[int, int]:
    """Count the calls whose prompt began with the previous call's prompt and reply.

    Args:
        calls: The calls of one attempt.

    Returns:
        How many continued the chain, out of how many could have.
    """
    kept = 0
    for before, after in zip(calls, calls[1:], strict=False):
        chain = before.response["prompt_token_ids"]
        chain = chain + before.response["choices"][0]["token_ids"]
        kept += after.response["prompt_token_ids"][: len(chain)] == chain
    return kept, max(len(calls) - 1, 0)


def labelled(label: str, text: str) -> None:
    """Print a label and a value, the value's later lines lined up under it.

    Args:
        label: Such as ``tool call``.
        text: The value.
    """
    lines = text.splitlines() or [""]
    pad = INDENT * 2 + " " * LABEL_WIDTH
    print(f"{INDENT * 2}{label:<{LABEL_WIDTH}}{lines[0]}")
    for line in lines[1:]:
        print(pad + line)


def first_lines(text: str, count: int = OUTPUT_LINES) -> str:
    """Keep the first few lines of an output, counting the rest.

    Args:
        text: The output.
        count: The most lines kept.

    Returns:
        The lines kept, and a last line counting the rest.
    """
    lines = text.splitlines()
    if not lines:
        return "(no output)"
    if len(lines) <= count:
        return text
    return "\n".join([*lines[:count], f"... {len(lines) - count} more lines"])


def show_call(call: Call) -> None:
    """Print one call: its tokens and timings, the message, each observation.

    Args:
        call: The call.
    """
    response = call.response
    usage, timings = response["usage"], response.get("timings", {})
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    finish = response["choices"][0]["finish_reason"]
    print(
        f"{INDENT}call {call.number} · read {usage['prompt_tokens']:,} tokens "
        f"({cached:,} cached), wrote {usage['completion_tokens']:,} · {finish} · "
        f"{timings.get('total_s', 0.0):.1f} s"
    )
    message = response["choices"][0]["message"]
    if message.get("content"):
        labelled("text", message["content"])
    for tool_call, result in call.observations:
        function = tool_call["function"]
        labelled("tool call", f"{function['name']} {function['arguments']}")
        labelled("observation", first_lines(result.output))
    if call.why:
        labelled("unreadable", call.why)


def save_rollout(rollout: Rollout, path: Path) -> None:
    """Write one attempt, every response (receipts included), as JSON.

    Args:
        rollout: The attempt.
        path: The file (its folder is made).

    Raises:
        OSError: If the file cannot be written.
    """
    record = {
        "task_id": rollout.task.task_id,
        "stop": rollout.stop,
        "grade": {
            "correct": rollout.grade.correct,
            "submitted": rollout.grade.submitted,
            "reason": rollout.grade.reason,
        },
        "responses": [call.response for call in rollout.calls],
        "messages": rollout.messages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def print_result(rollout: Rollout) -> None:
    """Print how an attempt ended: the stop rule, the grade, the token chain.

    Args:
        rollout: The attempt.
    """
    verdict = "correct" if rollout.grade.correct else "wrong"
    kept, could = continued(rollout.calls)
    print(f"{INDENT}stop: {rollout.stop}")
    print(
        f"{INDENT}grade: {verdict} ({rollout.grade.reason}; answer.txt: "
        f"{rollout.grade.submitted!r}) · {len(rollout.calls)} calls · reward "
        f"{int(rollout.grade.correct)}"
    )
    print(
        f"{INDENT}token chain: {kept} of {could} calls began with the previous "
        "call's prompt and reply"
    )


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, if the flags are impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--task", default=None, help=f"a question ({DEFAULT_TASK})")
    parser.add_argument("--held-out", type=int, default=None, help="the first N")
    args = parser.parse_args()
    if args.held_out is not None and args.task is not None:
        parser.error("--task and --held-out pick the questions two ways: use one")
    if args.held_out is not None and args.held_out < 1:
        parser.error("--held-out must be at least 1")
    return args


def main() -> None:
    """Run the harness through the engine on one question, or the first few."""
    args = parse_args()
    if args.held_out is None:
        tasks = [find_task(args.task or DEFAULT_TASK)]
    else:
        tasks = load_tasks()[: args.held_out]
    chat = http_chat(args.base_url, args.api_key)
    print(f"{SCRIPT} · chapter 17's harness, through the engine")
    print(f"{INDENT}base URL {args.base_url} · API key {args.api_key!r}")
    print(f"{INDENT}the harness holds no model and no tokenizer: only these two")
    solved = 0
    for task in tasks:
        workdir = prepare_workdir(task, work_folder(LABEL, task.task_id))
        print(f"\n{task.task_id} ({task.difficulty}): {task.question}")
        try:
            rollout = run_rollout(chat, task, workdir, show=show_call)
        except EngineError as error:
            raise SystemExit(f"the engine did not answer: {error}") from error
        path = rollout_path(LABEL, task.task_id)
        save_rollout(rollout, path)
        print_result(rollout)
        print(f"{INDENT}saved: {short_path(path)}")
        solved += rollout.grade.correct
    if len(tasks) > 1:
        print(f"\n{solved} of {len(tasks)} correct")


if __name__ == "__main__":
    main()
