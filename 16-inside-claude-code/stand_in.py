# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi==0.142.2", "uvicorn==0.54.0"]
# ///
"""A stand-in model provider: Anthropic's Messages API, answered from a script.

Claude Code believes it is talking to a model. It is talking to this file. A
model API is one idea: POST the conversation and the tools, get back a reply
and the usage (how many tokens were read and written). Anthropic's version is
``POST /v1/messages``. Claude Code asks for the reply as server-sent events
(SSE): a stream of small ``event:`` and ``data:`` lines, so it could show the
words as they arrive.

Every request is appended, whole, to ``calls.jsonl``. That log is the point of
the chapter: it shows exactly what the harness sends. The replies come from a
fixed script, so no model and no account are involved:

* a request that offers tools gets the script's next step, picked by how many
  replies the conversation already holds (a retried request gets the same
  step again);
* a request that offers no tools is a side call (a title, a summary, a check):
  it gets one short fixed text.

``run_claude.sh`` starts and stops it. By hand:
``uv run stand_in.py --script solve --log calls.jsonl``; and
``uv run stand_in.py --show`` prints the events it would send for a first call.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from records import (
    ASSISTANT,
    COUNT_TOKENS_PATH,
    KIND_COUNT,
    KIND_LOOP,
    KIND_SIDE,
    KIND_UNKNOWN,
    MESSAGES_PATH,
    TEXT,
    TOOL_USE,
)
from tasks import CHAPTER, course_home, load_task, task_instruction

# Only this machine can reach the stand-in.
HOST = "127.0.0.1"
DEFAULT_PORT = 8016
LOG_NAME = "calls.jsonl"

# A real provider counts tokens with its tokenizer. The stand-in only
# estimates (about four characters per token); inspect_calls.py counts properly.
CHARS_PER_TOKEN = 4
# These headers can carry a real credential, so their values are never written
# down. The x-api-key header is logged on purpose (the chapter shows that any
# string works), unless it looks like a real Anthropic key.
SECRET_HEADERS = ("authorization", "cookie", "proxy-authorization")
API_KEY_HEADER = "x-api-key"
REAL_KEY_PREFIX = "sk-ant-"
HIDDEN = "<not logged>"
SIDE_REPLY = "OK"
SERVED_MODEL = "stand-in"
SHOWN_MAX_TOKENS = 32000
NOT_FOUND = 404

# The task (bundled in data/): how many patients live in Murcia? Answer: 30.
BASH = "Bash"
TABLES = "sqlite3 data.db .tables"
COUNT = "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
WRITE = "echo 30 > answer.txt"
# Every appointment with its patient and doctor: 25,442 characters of output.
DUMP = (
    'sqlite3 data.db "SELECT a.appointment_date, p.name, p.city, d.name, '
    "d.specialty, a.fee FROM appointments a JOIN patients p ON p.id = "
    'a.patient_id JOIN doctors d ON d.id = a.doctor_id"'
)
DONE = "30 patients live in Murcia; answer.txt holds 30."


@dataclass(frozen=True)
class Step:
    """One scripted reply.

    Attributes:
        text: Plain text the reply carries ("" for none).
        tool: The tool the reply calls ("" for a reply that calls none).
        command: The command handed to that tool.
        streamed: False answers with one JSON body even when the request
            asked for a stream (one of the chapter's mishaps).
    """

    text: str = ""
    tool: str = ""
    command: str = ""
    streamed: bool = True


SOLVE = (
    Step(tool=BASH, command=TABLES),
    Step(tool=BASH, command=COUNT),
    Step(tool=BASH, command=WRITE),
    Step(text=DONE),
)
SCRIPTS: dict[str, tuple[Step, ...]] = {
    "solve": SOLVE,
    # The provider ignores "stream": true and sends one JSON body instead.
    "not-streamed": tuple(replace(step, streamed=False) for step in SOLVE),
    # One command prints far more than the harness passes on.
    "long-output": (SOLVE[0], Step(tool=BASH, command=DUMP), *SOLVE[1:]),
    # The model names a tool the harness does not have: 'bash', not 'Bash'.
    "lowercase-bash": (Step(tool="bash", command=TABLES), *SOLVE),
}


def is_side_call(body: dict[str, Any]) -> bool:
    """Tell a side call from a turn of the agent's loop.

    Args:
        body: The request body.

    Returns:
        True when the request offers no tools.
    """
    return not body.get("tools")


def replies_so_far(body: dict[str, Any]) -> int:
    """Count the replies already in the conversation.

    Args:
        body: The request body.

    Returns:
        How many messages have the role ``assistant``.
    """
    messages = body.get("messages", [])
    return sum(1 for message in messages if message["role"] == ASSISTANT)


def pick_step(script: tuple[Step, ...], body: dict[str, Any]) -> Step:
    """Choose the scripted reply for one request of the agent's loop.

    Args:
        script: The steps, in order.
        body: The request body.

    Returns:
        The step whose number is the count of replies so far; the last step
        once the script has run out.
    """
    return script[min(replies_so_far(body), len(script) - 1)]


def estimate_tokens(value: Any) -> int:
    """Estimate a token count the way the stand-in does: characters / 4.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The estimate, at least 1.
    """
    return max(1, len(json.dumps(value, ensure_ascii=False)) // CHARS_PER_TOKEN)


def build_message(
    content: list[dict[str, Any]], body: dict[str, Any], message_id: str
) -> dict[str, Any]:
    """Wrap content blocks in an Anthropic message, with its usage.

    Args:
        content: The reply's text and tool_use blocks.
        body: The request body (for the model name and the input estimate).
        message_id: A unique id for the message.

    Returns:
        The message, as the non-streamed API returns it.
    """
    calls_tool = any(block["type"] == TOOL_USE for block in content)
    read = [body.get("system"), body.get("messages"), body.get("tools")]
    return {
        "id": message_id,
        "type": "message",
        "role": ASSISTANT,
        "model": body.get("model", SERVED_MODEL),
        "content": content,
        # "tool_use" asks the harness to run the tool and call again.
        "stop_reason": "tool_use" if calls_tool else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(read),
            "output_tokens": estimate_tokens(content),
        },
    }


def step_content(step: Step, tool_use_id: str) -> list[dict[str, Any]]:
    """Turn a scripted step into the reply's content blocks.

    Args:
        step: The step.
        tool_use_id: The id the tool_use block gets.

    Returns:
        A text block, a tool_use block, or both.
    """
    content: list[dict[str, Any]] = []
    if step.text:
        content.append({"type": TEXT, "text": step.text})
    if step.tool:
        arguments = {"command": step.command}
        call = {"type": TOOL_USE, "id": tool_use_id, "name": step.tool}
        content.append(call | {"input": arguments})
    return content


def sse(event: str, data: dict[str, Any]) -> str:
    """Format one server-sent event.

    Args:
        event: The event name.
        data: Its JSON payload.

    Returns:
        ``event: <name>``, ``data: <json>`` and a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_events(message: dict[str, Any]) -> Iterator[str]:
    """Stream a message the way the Messages API does.

    The message opens empty, each content block arrives as a start, a delta
    and a stop, and a final delta carries the stop reason and the usage.

    Args:
        message: A complete message from ``build_message``.

    Yields:
        One server-sent event at a time.
    """
    read = message["usage"]["input_tokens"]
    opening = message | {"content": [], "stop_reason": None}
    opening["usage"] = {"input_tokens": read, "output_tokens": 0}
    yield sse("message_start", {"type": "message_start", "message": opening})
    for index, block in enumerate(message["content"]):
        if block["type"] == TEXT:
            empty = {"type": TEXT, "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            empty = block | {"input": {}}
            arguments = json.dumps(block["input"])
            delta = {"type": "input_json_delta", "partial_json": arguments}
        start = {"type": "content_block_start", "index": index, "content_block": empty}
        yield sse("content_block_start", start)
        change = {"type": "content_block_delta", "index": index, "delta": delta}
        yield sse("content_block_delta", change)
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
    end = {"stop_reason": message["stop_reason"], "stop_sequence": None}
    usage = {"output_tokens": message["usage"]["output_tokens"]}
    yield sse("message_delta", {"type": "message_delta", "delta": end, "usage": usage})
    yield sse("message_stop", {"type": "message_stop"})


def safe_header(name: str, value: str) -> str:
    """Return a header's value as the log may show it.

    Args:
        name: The header's name.
        value: Its value.

    Returns:
        The value, or ``<not logged>`` for a credential.
    """
    lowered = name.lower()
    if lowered in SECRET_HEADERS:
        return HIDDEN
    if lowered == API_KEY_HEADER and value.startswith(REAL_KEY_PREFIX):
        return HIDDEN
    return value


def show_first_reply(script_name: str) -> str:
    """Show a first request and the events the stand-in streams back.

    Args:
        script_name: Which entry of ``SCRIPTS`` answers.

    Returns:
        The request (much shorter than Claude Code's) and the response body.
    """
    tool = {"name": BASH, "input_schema": {"type": "object"}}
    question = {"role": "user", "content": task_instruction(load_task())}
    body = {
        "model": SERVED_MODEL,
        "max_tokens": SHOWN_MAX_TOKENS,
        "stream": True,
        "tools": [tool],
        "messages": [question],
    }
    step = pick_step(SCRIPTS[script_name], body)
    message = build_message(step_content(step, "toolu_0001"), body, "msg_0001")
    request = f"POST {MESSAGES_PATH}\n{json.dumps(body, indent=2)}\n"
    events = "".join(sse_events(message))
    return f"{request}\nthe answer, as server-sent events:\n{events}"


def create_app(script_name: str, log_path: Path) -> FastAPI:
    """Build the stand-in provider.

    Args:
        script_name: Which entry of ``SCRIPTS`` answers the agent's loop.
        log_path: The ``calls.jsonl`` file every request is appended to.

    Returns:
        The FastAPI application.

    Raises:
        KeyError: If there is no script with that name.
    """
    script = SCRIPTS[script_name]
    numbers = itertools.count(1)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="stand-in model provider")

    def log(request: Request, number: int, body: Any, reply: dict[str, Any]) -> None:
        """Append one request, and what the stand-in did with it, to the log."""
        headers = {k: safe_header(k, v) for k, v in request.headers.items()}
        record = {
            "call": number,
            "method": request.method,
            "path": request.url.path,
            "headers": headers,
            "body": body,
            "reply": reply,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Answer run_claude.sh's readiness check (never logged)."""
        return {"status": "ok", "script": script_name}

    @app.post(COUNT_TOKENS_PATH)
    async def count_tokens(request: Request) -> dict[str, int]:
        """Estimate how many tokens a request would read."""
        body = await request.json()
        read = [body.get("system"), body.get("messages"), body.get("tools")]
        answer = {"input_tokens": estimate_tokens(read)}
        log(request, next(numbers), body, {"kind": KIND_COUNT} | answer)
        return answer

    @app.post(MESSAGES_PATH, response_model=None)
    async def messages(request: Request) -> JSONResponse | StreamingResponse:
        """Answer one model call: the script's next step, or a side reply."""
        body = await request.json()
        number = next(numbers)
        if is_side_call(body):
            step, kind = Step(text=SIDE_REPLY), KIND_SIDE
        else:
            step, kind = pick_step(script, body), KIND_LOOP
        content = step_content(step, f"toolu_{number:04d}")
        message = build_message(content, body, f"msg_{number:04d}")
        streamed = bool(body.get("stream")) and step.streamed
        reply = {"kind": kind, "streamed": streamed, "message": message}
        log(request, number, body, reply)
        if streamed:
            events = sse_events(message)
            return StreamingResponse(events, media_type="text/event-stream")
        return JSONResponse(message)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def anything_else(request: Request, path: str) -> JSONResponse:
        """Log a request to any other address and refuse it."""
        raw = (await request.body()).decode("utf-8", errors="replace")
        log(request, next(numbers), raw, {"kind": KIND_UNKNOWN})
        error = {"type": "not_found_error", "message": f"no route /{path}"}
        return JSONResponse({"type": "error", "error": error}, status_code=NOT_FOUND)

    return app


def main() -> None:
    """Serve the stand-in on 127.0.0.1 until stopped, or show one reply."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", choices=sorted(SCRIPTS), default="solve")
    default_log = course_home() / CHAPTER / LOG_NAME
    parser.add_argument("--log", type=Path, default=default_log)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--show", action="store_true", help="print a first reply")
    args = parser.parse_args()
    if args.show:
        print(show_first_reply(args.script), end="")
        return
    app = create_app(args.script, args.log)
    address = f"http://{HOST}:{args.port}"
    print(
        f"stand-in ready · script {args.script} · {address} · log {args.log}",
        flush=True,
    )
    uvicorn.run(app, host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
