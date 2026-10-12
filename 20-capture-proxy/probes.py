# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Hand-made requests, one rule of the capture proxy at a time.

Each probe sends the proxy a few requests written by hand, the way a harness
writes them, and prints what happened:

* ``level``: what the proxy measured about the engine when it started,
  whether it opens a training session, and how it measures (its probe);
* ``key``: a made-up API key, no key, a session's key; then the proxy
  refusing to listen on every network interface without an admin key;
* ``sampling``: a harness asks for temperature 0.2 and top_p 0.9: what the
  engine receives in a training session, and in an evaluation session;
* ``clamp``: a harness asks for 32,000 tokens: what the engine receives;
* ``dialects``: one conversation sent in OpenAI's format and in Anthropic's:
  what reaches the engine, and how each reply comes back;
* ``budget``: nine calls in an eight-call session.

To show what reaches the engine, a probe opens its session with the
engine's address replaced by a tap (``tap.py``), which passes every request
on unchanged and writes both sides down.

Run ``./start.sh`` first, then ``uv run probes.py`` (every probe) or
``uv run probes.py budget``. It needs only Python's standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capture_client import (
    MAX_MODEL_CALLS,
    PROXY_URL,
    PURPOSE_EVAL,
    PURPOSE_TRAIN,
    CaptureClient,
    ProxyError,
    Session,
    summarize,
)
from lfm_files import OUTPUT_CAP, chapter_home, short_path
from tap import (
    Exchange,
    Tap,
    describe_request,
    describe_response,
    gap_verdict,
    short,
)
from tasks import find_task, task_instruction
from tiny_harness_http import SYSTEM_PROMPT, TOOL_NAME, bash_schema

SCRIPT = "probes.py"
PROBES = ("level", "key", "sampling", "clamp", "dialects", "budget")
ALL = "all"
MODEL_NAME = "lfm2.5-2.6b"  # the name the engine serves (as in start.sh)
CHAT_PATH = "/v1/chat/completions"  # OpenAI's chat route
MESSAGES_PATH = "/v1/messages"  # Anthropic's Messages route
SESSIONS_PATH = "/sessions"
OK = 200
ANTHROPIC_VERSION = "2023-06-01"  # the header every Anthropic client sends
MADE_UP_KEY = "sk-made-up-key"
QUESTION_TASK = "clinic-easy-001"  # the conversation the dialects probe sends
SHORT_PROMPT = "Reply with the single word: ok"
NEXT_PROMPT = "Again."  # each later call of the budget probe's conversation
SHORT_REPLY_TOKENS = 64  # replies these probes do not need in full
# The budget probe's conversation goes on call after call: its replies get
# room to finish their reasoning, so each one is a whole turn.
BUDGET_REPLY_TOKENS = 256
ASKED_TEMPERATURE = 0.2  # what a harness might ask for...
ASKED_TOP_P = 0.9  # ...and this cuts off the distribution's tail
HUGE_MAX_TOKENS = 32_000  # an output budget harnesses really send
SMALL_MAX_TOKENS = 200  # under the proxy's cap: left alone
# The sampling fields the proxy pins for training, in the order printed.
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)
# The proxy's own start-up command, for the open-interface refusal.
PROXY_MODULE = "openenv.core.harness.capture.server"  # as in start.sh
PROXY_ENV_FOLDER = "proxy-env"
PROXY_ENV_VARIABLE = "PROXY_ENV"  # as in start.sh
ADMIN_KEY_VARIABLE = "OPENENV_CAPTURE_ADMIN_KEY"
EVERY_INTERFACE = "0.0.0.0"
REFUSAL_WAIT_S = 60  # the refusal comes before any network work
# A model call may wait minutes in the engine's queue (chapter 18), so these
# requests wait as long as a harness does; the session routes answer at once.
HTTP_TIMEOUT_S = 600
DATA_PREFIX = "data: "  # a server-sent event's JSON line
INDENT = "  "
WIDE = INDENT * 6  # under "Anthropic: "
SESSION_NAMES = {
    PURPOSE_TRAIN: "a training session",
    PURPOSE_EVAL: "an evaluation session",
}
ABSENT = "(absent)"


@dataclass(frozen=True)
class Reply:
    """A reply from the proxy, as a harness receives it.

    Attributes:
        status: The HTTP status.
        content_type: The ``Content-Type`` header.
        text: The body.
    """

    status: int
    content_type: str
    text: str

    def json(self) -> dict[str, Any]:
        """Read the body as JSON.

        Returns:
            The body, parsed.
        """
        return json.loads(self.text)


@dataclass
class Probes:
    """What every probe shares: the proxy's client, and a tap started on demand.

    Attributes:
        client: The capture proxy's session client.
        tap: The tap, once a probe needed one.
        probe_records: What the proxy sent the tap while opening the last
            tapped session (its probe, the first time it meets the tap).
        opened: Sessions opened, to delete at the end.
    """

    client: CaptureClient
    tap: Tap | None = None
    probe_records: list[Exchange] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)

    def session(self, purpose: str, probe: str, tapped: bool = False) -> Session:
        """Open a session, optionally with the tap in place of the engine.

        A session that names an engine the proxy has not met makes the proxy
        probe it first. Those requests are kept apart (``probe_records``), so
        what a probe reads next from the tap starts with its own requests.

        Args:
            purpose: ``train`` or ``eval``.
            probe: The probe's name, kept as a label.
            tapped: True to send the session's calls through the tap.

        Returns:
            The session.
        """
        llm_url = None
        if tapped:
            if self.tap is None:
                self.tap = Tap(self.client.health()["upstream"])
                self.tap.start()
            llm_url = self.tap.url
        session = self.client.open_session(
            purpose, {"probe": probe}, llm_url=llm_url, model=MODEL_NAME
        )
        self.opened.append(session.session_id)
        self.probe_records = self.tap.take() if self.tap else []
        return session

    def engine_saw(self) -> list[Exchange]:
        """The chat requests that reached the engine since the last look.

        Returns:
            Those exchanges.
        """
        if self.tap is None:
            return []
        return [record for record in self.tap.take() if record.path == CHAT_PATH]

    def close(self) -> None:
        """Delete every session opened, and stop the tap, even if the proxy died."""
        try:
            for session_id in self.opened:
                with contextlib.suppress(ProxyError):
                    self.client.close(session_id)
            self.opened.clear()
        finally:
            if self.tap is not None:
                self.tap.stop()
                self.tap = None

    def one_request(self) -> Exchange:
        """The one chat request that reached the engine since the last look.

        Returns:
            That exchange.

        Raises:
            SystemExit: If the engine saw no request, or several.
        """
        seen = self.engine_saw()
        if len(seen) != 1:
            raise SystemExit(f"expected one request at the engine, saw {len(seen)}")
        return seen[0]


def post(
    url: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> Reply:
    """POST a request as a harness does, and take whatever comes back.

    Args:
        url: Where to.
        body: The JSON body.
        headers: Extra headers (the API key).

    Returns:
        The reply, refusals included.

    Raises:
        OSError: If nobody answers.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as answer:
            kind = answer.headers.get("Content-Type", "")
            return Reply(answer.status, kind, answer.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        kind = error.headers.get("Content-Type", "")
        text = error.read().decode("utf-8", errors="replace")
        return Reply(error.code, kind, text)


def bearer(key: str) -> dict[str, str]:
    """The header OpenAI's clients send their key in.

    Args:
        key: The API key.

    Returns:
        ``{"Authorization": "Bearer <key>"}``.
    """
    return {"Authorization": f"Bearer {key}"}


def openai_body(messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """Write an OpenAI chat request.

    Args:
        messages: The conversation.
        **fields: Anything else, such as ``max_tokens``.

    Returns:
        The body.
    """
    return {"model": MODEL_NAME, "messages": messages} | fields


def ask_short(**fields: Any) -> dict[str, Any]:
    """Write the one-line request most probes send.

    Args:
        **fields: Anything else, such as ``temperature``.

    Returns:
        The body.
    """
    return openai_body([{"role": "user", "content": SHORT_PROMPT}], **fields)


def error_message(reply: Reply) -> str:
    """Read the message out of a refusal.

    Args:
        reply: A refusal.

    Returns:
        Its message, or its raw body.
    """
    try:
        found = reply.json().get("error", reply.text)
    except (json.JSONDecodeError, AttributeError):
        return reply.text
    return found.get("message", found) if isinstance(found, dict) else str(found)


def fields_line(body: dict[str, Any], names: tuple[str, ...]) -> str:
    """Write the named fields a body carries, in order.

    Args:
        body: A request body.
        names: The fields to look for.

    Returns:
        Such as ``temperature 1.0 · top_p 1.0``, or ``none``.
    """
    found = [f"{name} {json.dumps(body[name])}" for name in names if name in body]
    return " · ".join(found) or "none"


def shown_value(body: dict[str, Any], name: str) -> str:
    """Write one field of a body as JSON, or say it is absent.

    Args:
        body: A request body.
        name: The field.

    Returns:
        Its JSON, or ``(absent)``.
    """
    return json.dumps(body[name]) if name in body else ABSENT


def probe_level(probes: Probes) -> None:
    """Show the capture level, a training session, and the probe behind them.

    Args:
        probes: The shared state.
    """
    health = probes.client.health()
    print("1. what the proxy measured when it started (GET /health)")
    print(
        f"{INDENT}engine {health['upstream']} · capture level "
        f"{health['capture_level']} · {health['rollout_type']} rollouts"
    )
    print("\n2. a training session on that engine (POST /sessions, purpose train)")
    try:
        session = probes.session(PURPOSE_TRAIN, "level")
    except ProxyError as error:
        if error.status is None:  # nobody answered: not a refusal
            raise
        print(f"{INDENT}refused: {error}")
        session = probes.session(PURPOSE_EVAL, "level")
        print(
            f"{INDENT}an evaluation session opens: {session.session_id} · "
            f"rollout type {session.rollout_type}"
        )
    else:
        print(
            f"{INDENT}opened: {session.session_id} · capture level "
            f"{session.capture_level} · rollout type {session.rollout_type}"
        )
    print("\n3. how the proxy measures an engine: its probe, seen from a tap")
    tapped = probes.session(PURPOSE_EVAL, "level", tapped=True)
    for record in probes.probe_records:
        print(f"{INDENT}{describe_request(record)}")
        print(f"{INDENT * 2}back: {describe_response(record)}")
    gaps, verdict = gap_verdict(probes.probe_records)
    if len(gaps) == 2:
        print(
            f"{INDENT}twice the temperature: gap {gaps[0]:.4f} -> {gaps[1]:.4f}, "
            f"ratio {gaps[1] / gaps[0]:.3f}: {verdict} log-probabilities"
        )
    print(f"{INDENT}the proxy's verdict on this engine: {tapped.capture_level}")


def proxy_python() -> Path:
    """Where start.sh installed the proxy's Python.

    Returns:
        ``$PROXY_ENV/bin/python``, by default in ``$COURSE_HOME/20-capture-proxy``.
    """
    configured = os.environ.get(PROXY_ENV_VARIABLE)
    folder = Path(configured) if configured else chapter_home() / PROXY_ENV_FOLDER
    return folder / "bin" / "python"


def free_port() -> int:
    """Ask the system for a port nobody listens on.

    Returns:
        The port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def open_interface_refusal(engine_url: str) -> tuple[list[str], int, str]:
    """Start the proxy on every network interface, with no admin key.

    It must refuse before it listens. If it ever started instead, it is
    stopped at once and the last line says so.

    Args:
        engine_url: The engine to name in the command.

    Returns:
        The command, its exit code, and its last line of output.

    Raises:
        FileNotFoundError: If start.sh has not installed the proxy yet.
    """
    python = proxy_python()
    if not python.exists():
        raise FileNotFoundError(f"{short_path(python)}: run ./start.sh install")
    command = [
        str(python),
        "-m",
        PROXY_MODULE,
        "--llm-url",
        engine_url,
        "--host",
        EVERY_INTERFACE,
        "--port",
        str(free_port()),
        "--capture-level",  # skips the engine probe; the refusal comes first anyway
        "tokens",
    ]
    clean = {k: v for k, v in os.environ.items() if k != ADMIN_KEY_VARIABLE}
    process = subprocess.Popen(
        command, env=clean, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    listened = False
    try:
        output, _ = process.communicate(timeout=REFUSAL_WAIT_S)
    except subprocess.TimeoutExpired:  # it listened: the refusal is gone
        listened = True
        process.kill()
        output, _ = process.communicate()
    finally:
        if process.poll() is None:  # interrupted: leave nothing running
            process.kill()
    lines = [line for line in output.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if listened:
        last = f"IT STARTED, listening on every interface (stopped): {last}"
    return command, process.returncode, last


def probe_key(probes: Probes) -> None:
    """Show who the proxy answers: session keys only.

    Args:
        probes: The shared state.
    """
    url = probes.client.proxy_url + CHAT_PATH
    body = ask_short(max_tokens=SHORT_REPLY_TOKENS)
    print(f"1. a made-up API key ({MADE_UP_KEY!r})")
    reply = post(url, body, bearer(MADE_UP_KEY))
    print(f"{INDENT}HTTP {reply.status}: {error_message(reply)}")
    print("\n2. no API key at all")
    reply = post(url, body)
    print(f"{INDENT}HTTP {reply.status}: {error_message(reply)}")
    print("\n3. a session's id as the API key")
    session = probes.session(PURPOSE_EVAL, "key")
    reply = post(url, body, bearer(session.session_id))
    finish = reply.json()["choices"][0]["finish_reason"] if reply.status == OK else ""
    print(f"{INDENT}{session.session_id} · HTTP {reply.status}: answered ({finish})")
    print("\n4. the session routes, from this machine (GET /sessions)")
    listed = probes.client.send("GET", SESSIONS_PATH)["sessions"]
    print(
        f"{INDENT}answered, {len(listed)} session(s) listed: no admin key is "
        "needed while the proxy listens on 127.0.0.1 only"
    )
    print(f"\n5. the proxy on every interface ({EVERY_INTERFACE}), with no admin key")
    try:
        command, code, last = open_interface_refusal(probes.client.health()["upstream"])
    except FileNotFoundError as error:
        print(f"{INDENT}not tried: {error}")
        return
    shown = [short_path(Path(command[0])), *command[1:]]
    print(f"{INDENT}$ {' '.join(shown)}")
    print(f"{INDENT}exit code {code}: {last}")


def probe_sampling(probes: Probes) -> None:
    """Show the sampling a harness asks for, and what reaches the engine.

    Args:
        probes: The shared state.
    """
    url = probes.client.proxy_url + CHAT_PATH
    print(f"a harness asks for temperature {ASKED_TEMPERATURE} and top_p {ASKED_TOP_P}")
    for number, purpose in enumerate((PURPOSE_TRAIN, PURPOSE_EVAL), start=1):
        session = probes.session(purpose, "sampling", tapped=True)
        body = ask_short(
            temperature=ASKED_TEMPERATURE,
            top_p=ASKED_TOP_P,
            max_tokens=SHORT_REPLY_TOKENS,
        )
        reply = post(url, body, bearer(session.session_id))
        print(f"\n{number}. {SESSION_NAMES[purpose]}: HTTP {reply.status}")
        for record in probes.engine_saw():
            line = fields_line(record.request or {}, SAMPLING_FIELDS)
            print(f"{INDENT}the engine received: {line}")
        turn = probes.client.rollout(session.session_id)["turns"][-1]
        asked = json.dumps(turn["requested_sampling_params"])
        print(f"{INDENT}the rollout keeps the ask: {asked}")
        print(f"{INDENT}and what was sent: {json.dumps(turn['sampling_params'])}")


def probe_clamp(probes: Probes) -> None:
    """Show the output cap: what reaches the engine for each max_tokens asked.

    Args:
        probes: The shared state.
    """
    url = probes.client.proxy_url + CHAT_PATH
    session = probes.session(PURPOSE_TRAIN, "clamp", tapped=True)
    # OUTPUT_CAP (lfm_files.py) is the same 1,024 start.sh gives the proxy.
    print(f"a training session ({session.session_id}); start.sh's cap: {OUTPUT_CAP:,}")
    for asked in (HUGE_MAX_TOKENS, None, SMALL_MAX_TOKENS):
        fields = {} if asked is None else {"max_tokens": asked}
        reply = post(url, ask_short(**fields), bearer(session.session_id))
        received = [
            str((record.request or {}).get("max_tokens"))
            for record in probes.engine_saw()
        ]
        label = f"max_tokens {asked:,}" if asked else "no max_tokens"
        print(
            f"{INDENT}asked {label:<19} -> the engine received max_tokens "
            f"{', '.join(received)} (HTTP {reply.status})"
        )


def server_sent_events(text: str) -> list[dict[str, Any]]:
    """Read a server-sent-event stream into its events.

    Args:
        text: The stream's body. Lines starting with ':' are comments (the
            proxy's keepalives) and ``event:`` lines repeat each event's
            type, so only the ``data:`` lines are read.

    Returns:
        Each event's JSON, in order.
    """
    return [
        json.loads(line[len(DATA_PREFIX) :])
        for line in text.splitlines()
        if line.startswith(DATA_PREFIX)
    ]


def tool_use_from_events(events: list[dict[str, Any]]) -> str:
    """Put a streamed tool call back together, as Claude Code does.

    Args:
        events: The stream's events.

    Returns:
        Such as ``bash {"command": "ls"}``, or ``(no tool call)``.
    """
    name, pieces = "", []
    for event in events:
        block = event.get("content_block") or {}
        if block.get("type") == "tool_use":
            name = block["name"]
        delta = event.get("delta") or {}
        if delta.get("type") == "input_json_delta":
            pieces.append(delta["partial_json"])
    return f"{name} {''.join(pieces)}" if name else "(no tool call)"


def stop_reason(events: list[dict[str, Any]]) -> str | None:
    """Read why a streamed Anthropic reply stopped.

    Args:
        events: The stream's events.

    Returns:
        The ``stop_reason`` of its ``message_delta``, or None.
    """
    for event in events:
        if event.get("type") == "message_delta":
            return event["delta"].get("stop_reason")
    return None


def anthropic_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Write an OpenAI tool schema the way Anthropic's API writes tools.

    Args:
        schema: ``{"type": "function", "function": {...}}``.

    Returns:
        ``{"name", "description", "input_schema"}``.
    """
    function = schema["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "input_schema": function["parameters"],
    }


def compare_upstream(first: Exchange, second: Exchange) -> None:
    """Print how two requests that reached the engine agree and differ.

    Args:
        first: The OpenAI request, as the engine received it.
        second: The Anthropic one.
    """
    one, two = first.request or {}, second.request or {}
    print(f"{INDENT}both: POST {first.path} and POST {second.path}")
    for name in ("messages", "tools"):
        same = "identical" if one.get(name) == two.get(name) else "DIFFERENT"
        print(f"{INDENT}{name}: {same}")
    streams = [json.dumps(body.get("stream")) for body in (one, two)]
    print(f"{INDENT}stream: {' and '.join(streams)}")
    shown = ("messages", "tools", "stream")
    for name in sorted(set(one) | set(two)):
        if name not in shown and one.get(name) != two.get(name):
            values = [shown_value(body, name) for body in (one, two)]
            print(f"{INDENT}differs: {name}: {' and '.join(values)}")


def probe_dialects(probes: Probes) -> None:
    """Send one conversation in two dialects; compare what reaches the engine.

    Args:
        probes: The shared state.
    """
    proxy = probes.client.proxy_url
    system = SYSTEM_PROMPT.format(tool=TOOL_NAME)
    question = task_instruction(find_task(QUESTION_TASK))
    tool = bash_schema()
    openai = probes.session(PURPOSE_EVAL, "dialects", tapped=True)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    body = openai_body(messages, tools=[tool], max_tokens=OUTPUT_CAP)
    openai_reply = post(proxy + CHAT_PATH, body, bearer(openai.session_id))
    openai_record = probes.one_request()
    anthropic = probes.session(PURPOSE_EVAL, "dialects", tapped=True)
    anthropic_body = {
        "model": MODEL_NAME,
        "system": system,
        "messages": [{"role": "user", "content": question}],
        "tools": [anthropic_tool(tool)],
        "max_tokens": OUTPUT_CAP,
        "stream": True,  # Claude Code always streams
    }
    headers = {
        "x-api-key": anthropic.session_id,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    anthropic_reply = post(proxy + MESSAGES_PATH, anthropic_body, headers)
    anthropic_record = probes.one_request()
    print(f"one conversation (a system prompt, {QUESTION_TASK}, the bash tool), twice:")
    print(f"{INDENT}OpenAI chat:        POST {CHAT_PATH} · key in Authorization")
    print(
        f"{INDENT}Anthropic Messages: POST {MESSAGES_PATH} · key in x-api-key, streamed"
    )
    print("\n1. what reached the engine")
    compare_upstream(openai_record, anthropic_record)
    print("\n2. what came back to each caller")
    message = openai_reply.json()["choices"][0]["message"]
    function = ((message.get("tool_calls") or [{}])[0]).get("function") or {}
    called = f"{function.get('name', '(no tool call)')} {function.get('arguments', '')}"
    print(f"{INDENT}OpenAI:    {openai_reply.content_type} · one JSON body · {called}")
    events = server_sent_events(anthropic_reply.text)
    print(f"{INDENT}Anthropic: {anthropic_reply.content_type} · {len(events)} events")
    print(WIDE + ", ".join(str(event.get("type")) for event in events))
    print(f"{WIDE}{tool_use_from_events(events)} · stop_reason {stop_reason(events)}")
    print("\n3. what the proxy recorded")
    for session in (openai, anthropic):
        document = probes.client.rollout(session.session_id)
        dialect = document["metadata"].get("api_type")
        turns = document["stats"]["n_turns"]
        print(f"{INDENT}{session.session_id}: dialect {dialect} · {turns} turn")


def probe_budget(probes: Probes) -> None:
    """Make nine calls in an eight-call session.

    Args:
        probes: The shared state.
    """
    url = probes.client.proxy_url + CHAT_PATH
    session = probes.session(PURPOSE_TRAIN, "budget", tapped=True)
    print(
        f"a training session ({session.session_id}) with max_model_calls "
        f"{session.max_model_calls}: one conversation, one call after another"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": SHORT_PROMPT}]
    for number in range(1, MAX_MODEL_CALLS + 2):
        body = openai_body(messages, max_tokens=BUDGET_REPLY_TOKENS)
        reply = post(url, body, bearer(session.session_id))
        if reply.status != OK:
            raise SystemExit(
                f"call {number}: HTTP {reply.status}: {error_message(reply)}"
            )
        answer = reply.json()
        message = answer["choices"][0]["message"]
        who = "the engine" if probes.engine_saw() else "the proxy itself"
        print(
            f"{INDENT}call {number}: answered by {who} · "
            f"{short(message.get('content'))} · "
            f"{answer['usage']['completion_tokens']} tokens"
        )
        messages += [
            {"role": "assistant", "content": message.get("content")},
            {"role": "user", "content": NEXT_PROMPT},
        ]
    summary = summarize(probes.client.rollout(session.session_id))
    print(
        f"{INDENT}the rollout: {summary.turns} turns recorded, in "
        f"{summary.roots} conversation · budget stops: {summary.budget_stops}"
    )


PROBE_FUNCTIONS: dict[str, Callable[[Probes], None]] = {
    "level": probe_level,
    "key": probe_key,
    "sampling": probe_sampling,
    "clamp": probe_clamp,
    "dialects": probe_dialects,
    "budget": probe_budget,
}


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", nargs="?", default=ALL, choices=(*PROBES, ALL))
    parser.add_argument("--proxy-url", default=PROXY_URL)
    return parser.parse_args()


def main() -> None:
    """Run one probe, or all of them, against the running proxy."""
    args = parse_args()
    names = PROBES if args.probe == ALL else (args.probe,)
    probes = Probes(CaptureClient(args.proxy_url))
    try:
        engine = probes.client.health()["upstream"]
    except ProxyError as error:
        raise SystemExit(f"{error}: start it with ./start.sh") from error
    print(f"{SCRIPT} · the capture proxy at {args.proxy_url}, in front of {engine}")
    try:
        for name in names:
            print(f"\n== {name} ==")
            PROBE_FUNCTIONS[name](probes)
    finally:
        probes.close()


if __name__ == "__main__":
    main()
