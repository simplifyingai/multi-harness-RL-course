"""Chapter 21 without a model: the REAL capture proxy, in front of a scripted engine.

No test here loads weights or needs MLX. The engine is the chapter's
``scripted_engine.py``: chapter 20's real ``engine.py`` with a scripted model
behind its seam, served over real HTTP on a free port, with its remote
control (``/script``) and its broken copy (``/broken``). In front of it runs
OpenEnv's capture proxy itself, the pinned commit ``start.sh`` installs,
started by ``start.sh proxy`` as a learner starts it. So every graph, row,
mask, finding and stat checked here is the proxy's own work, not a copy of
it. Those tests are skipped when the proxy is not installed
(``./start.sh install``); the Claude Code tests are skipped when the pinned
Claude Code is not on disk either.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import use_chapter

CHAPTER = use_chapter("21-calls-to-rows")

import capture_client  # noqa: E402
import claude_code  # noqa: E402
import crosscheck  # noqa: E402
import engine  # noqa: E402
import graph_lab  # noqa: E402
import reminder  # noqa: E402
import rollout_view  # noqa: E402
import scripted_engine  # noqa: E402
import scripted_replies  # noqa: E402
import tasks  # noqa: E402
import training_rows  # noqa: E402

START_SH = CHAPTER / "start.sh"
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where start.sh installed the proxy, and Claude Code, on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
START_TIMEOUT_S = 300


def installed_commit(env: Path) -> str | None:
    """The OpenEnv commit installed in an environment, read from its metadata.

    Args:
        env: The environment's folder.

    Returns:
        The commit, or None when OpenEnv is not installed there.
    """
    for record in env.glob("lib/python*/site-packages/openenv-*.dist-info"):
        found = json.loads((record / "direct_url.json").read_text())
        return found["vcs_info"]["commit_id"]
    return None


def find_proxy_env() -> Path | None:
    """Find a proxy environment at the pinned commit: this chapter's, else 20's.

    Returns:
        The environment, or None.
    """
    named = os.environ.get("PROXY_ENV")
    candidates = [Path(named)] if named else []
    candidates += [_HOME / "21-calls-to-rows" / "proxy-env"]
    candidates += [_HOME / "20-capture-proxy" / "proxy-env"]  # the same pin
    for env in candidates:
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def find_claude_bin() -> Path | None:
    """Find the pinned Claude Code on disk: CLAUDE_BIN, this chapter's, or 16's.

    Returns:
        The executable, or None.
    """
    named = os.environ.get(claude_code.CLAUDE_BIN_ENV)
    if named:
        return Path(named)
    for chapter in ("21-calls-to-rows", "16-inside-claude-code"):
        path_file = _HOME / chapter / claude_code.INSTALL_FOLDER / claude_code.PATH_FILE
        if path_file.is_file():
            binary = Path(path_file.read_text().strip())
            if binary.is_file():
                return binary
    return None


REAL_PROXY_ENV = find_proxy_env()
CLAUDE_BIN = find_claude_bin()
needs_proxy = pytest.mark.skipif(
    REAL_PROXY_ENV is None,
    reason="the pinned capture proxy is not installed: ./start.sh install",
)
needs_claude = pytest.mark.skipif(
    REAL_PROXY_ENV is None or CLAUDE_BIN is None,
    reason="needs the capture proxy and the pinned Claude Code 2.1.290",
)


def free_port() -> int:
    """Ask the system for a port nobody listens on.

    Returns:
        The port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def listening(port: int) -> bool:
    """Say whether anything listens on a port of this machine.

    Args:
        port: The port.

    Returns:
        True when a connection is accepted.
    """
    with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port)):
        return True
    return False


@contextlib.contextmanager
def live_engine() -> Iterator[tuple[int, Any, Any]]:
    """Serve the scripted engine (and its broken copy) over real HTTP.

    Yields:
        The port, the healthy model and the broken model.
    """
    app, model, broken = scripted_engine.create_app()
    sock = engine.claim_port(0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("the scripted engine did not start within 10 s")
    try:
        yield sock.getsockname()[1], model, broken
    finally:
        server.should_exit = True
        thread.join(10)


def run_start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path, **settings: str
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``start``, ``scripted``, ``proxy``, ``stop`` or ``status``.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).
        **settings: More environment, such as ``PATH``.

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "PROXY_PORT": str(proxy_port),
        "COURSE_HOME": str(home),
        "PROXY_ENV": str(REAL_PROXY_ENV),
        **settings,
    }
    return subprocess.run(
        ["bash", str(START_SH), command],
        env=env,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
        check=False,
    )


@dataclass
class Stack:
    """A scripted engine, and the real proxy start.sh put in front of it.

    Attributes:
        proxy_url: The proxy's root.
        engine_url: The engine's root.
        model: The healthy engine's scripted model.
        broken: The broken engine's scripted model.
    """

    proxy_url: str
    engine_url: str
    model: Any
    broken: Any

    def client(self) -> capture_client.CaptureClient:
        """A session client for this proxy.

        Returns:
            The client.
        """
        return capture_client.CaptureClient(self.proxy_url)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The real proxy, started by start.sh, in front of the scripted engine."""
    if REAL_PROXY_ENV is None:
        pytest.skip("the capture proxy is not installed: ./start.sh install")
    home = tmp_path_factory.mktemp("stack")
    with live_engine() as (engine_port, model, broken):
        proxy_port = free_port()
        started = run_start_sh("proxy", engine_port, proxy_port, home)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            yield Stack(
                f"http://127.0.0.1:{proxy_port}",
                f"http://127.0.0.1:{engine_port}",
                model,
                broken,
            )
        finally:
            run_start_sh("stop", engine_port, proxy_port, home)


def case(name: str) -> graph_lab.Case:
    """Find one of graph_lab's cases by name.

    Args:
        name: The case's name.

    Returns:
        The case.
    """
    return next(c for c in graph_lab.CASES if c.name == name)


def run_lab_case(running: Stack, name: str) -> tuple[graph_lab.Outcome, dict]:
    """Run one graph_lab case through the real proxy.

    Args:
        running: The stack.
        name: The case's name.

    Returns:
        The outcome, and the full document graph_lab saved.
    """
    outcome = graph_lab.run_case(
        running.client(),
        running.engine_url,
        case(name),
        rollout_view.Painter("never"),
        1,
    )
    saved = graph_lab.rollout_path(graph_lab.LABEL, name)
    return outcome, json.loads(saved.read_text())


def turn_findings(document: dict, code: str) -> list[str]:
    """The document's findings with one code.

    Args:
        document: A rollout document.
        code: Such as ``no_logprobs``.

    Returns:
        The findings.
    """
    return [f for f in document["validation"] if f"] {code}:" in f]


# ------------------------------------------------- the project's functions


def doc(turns: list[dict], sequences: list[dict] | None = None) -> dict[str, Any]:
    """Write a small rollout document.

    Args:
        turns: The turns.
        sequences: The rows.

    Returns:
        The document.
    """
    return {"turns": turns, "sequences": sequences or []}


def test_count_calls_leaves_out_discarded_turns_and_side_calls() -> None:
    document = doc(
        [
            {"n_tools": 1, "n_sampled": 10},
            {"n_tools": 0, "n_sampled": 7},  # a side call: no tools
            {"n_tools": 1, "n_sampled": 20, "discarded": True},
            {"n_tools": 1, "n_sampled": 5},
        ]
    )
    assert training_rows.count_calls(document) == 2
    # n_sampled skips the discarded turn but keeps the side call, as the
    # project's run_rollout does.
    assert training_rows.n_sampled(document) == 22


def test_count_calls_without_any_tools_counts_every_live_turn() -> None:
    document = doc([{"n_tools": 0}, {"n_tools": 0, "discarded": True}, {"n_tools": 0}])
    assert training_rows.count_calls(document) == 2


def test_shared_prefix_is_trained_once() -> None:
    # The project's own test, on the copy.
    first = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1], "logprobs": []}
    fork = {"input_ids": [1, 2, 3, 9, 9], "loss_mask": [0, 1, 1, 1, 1], "logprobs": []}
    other = {"input_ids": [7, 8], "loss_mask": [0, 1], "logprobs": []}
    out = training_rows.dedupe_shared_prefixes([first, fork, other])
    assert out[0]["loss_mask"] == [0, 1, 1, 1]
    assert out[1]["loss_mask"] == [0, 0, 0, 1, 1]
    assert out[2]["loss_mask"] == [0, 1]
    assert fork["loss_mask"] == [0, 1, 1, 1, 1], "inputs are not mutated"
    assert training_rows.trained_tokens(out) == 6


def test_item_from_sequence_takes_only_masked_positions_after_the_first() -> None:
    sequence = {
        "input_ids": [5, 6, 7, 8],
        "loss_mask": [1, 0, 1, 1],
        "logprobs": [-9.0, 0.0, -0.5, -0.25],
    }
    item = training_rows.item_from_sequence(sequence, advantage=0.5)
    assert item is not None
    assert (item.positions, item.old_logprobs, item.n_tokens) == (
        [2, 3],
        [-0.5, -0.25],
        2,
    )
    empty = {"input_ids": [1, 2], "loss_mask": [0, 0], "logprobs": [None, None]}
    assert training_rows.item_from_sequence(empty) is None


def test_trainable_sequences_keeps_three_fields_of_trainable_rows() -> None:
    rows = [
        {"input_ids": [1], "loss_mask": [1], "logprobs": [-1.0], "trainable": True},
        {"input_ids": [2], "loss_mask": [0], "logprobs": [0.0], "trainable": False},
    ]
    kept = training_rows.trainable_sequences({"sequences": rows})
    assert kept == [{"input_ids": [1], "loss_mask": [1], "logprobs": [-1.0]}]


# ------------------------------------------------- the scripted engine


def test_scripted_tokens_round_trip_with_markers() -> None:
    reply = scripted_replies.call_reply("ls -l", tool="Bash")
    ids = scripted_replies.encode(reply.text)
    assert scripted_replies.decode(ids) == reply.text
    assert ids[-1] == scripted_engine.EOS  # <|im_end|> is one token
    assert "[Bash(command='ls -l')]" in reply.text


def answer_body(core: engine.Engine, body: dict[str, Any]) -> dict[str, Any]:
    """Answer one request in this thread, as the server's worker would.

    Args:
        core: The engine.
        body: The request.

    Returns:
        The response body.
    """
    completion = core.complete(engine.read_request(body))
    return engine.to_openai(core.model, completion, 0.0, core.token_fields)


ASK = {"messages": [{"role": "user", "content": "hi"}]}
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


def test_a_reply_queued_without_logprobs_comes_back_with_nulls() -> None:
    model = scripted_engine.ScriptedModel()
    model.push([scripted_replies.answer_reply("ok", logprobs=False)])
    response = answer_body(engine.Engine(model), ASK)
    entries = response["choices"][0]["logprobs"]["content"]
    assert entries and all(entry["logprob"] is None for entry in entries)
    assert len(response["choices"][0]["token_ids"]) == len(entries)
    # The next reply, by default, is scored again.
    again = answer_body(engine.Engine(model), ASK)
    assert all(e["logprob"] < 0 for e in again["choices"][0]["logprobs"]["content"])


def test_the_broken_engine_drops_logprobs_only_when_tools_are_offered() -> None:
    core = engine.Engine(scripted_engine.ScriptedModel(no_logprobs_with_tools=True))
    plain = answer_body(core, ASK)
    with_tools = answer_body(core, ASK | {"tools": TOOLS})
    assert all(e["logprob"] < 0 for e in plain["choices"][0]["logprobs"]["content"])
    nulls = with_tools["choices"][0]["logprobs"]["content"]
    assert nulls and all(e["logprob"] is None for e in nulls)
    assert with_tools["choices"][0]["message"]["tool_calls"]  # the call still works


def http_json(method: str, url: str, body: Any = None) -> Any:
    """Send one JSON request and read the JSON answer.

    Args:
        method: The method.
        url: Where to.
        body: The body, or None.

    Returns:
        The answer.
    """
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as answer:
        return json.loads(answer.read())


def test_the_remote_control_queues_and_clears_replies_on_both_engines() -> None:
    with live_engine() as (port, model, broken):
        root = f"http://127.0.0.1:{port}"
        healthy = scripted_replies.ScriptClient(root)
        assert healthy.push([scripted_replies.answer_reply("a")])["queued"] == 1
        assert healthy.clear()["queued"] == 0
        sick = scripted_replies.ScriptClient(root + graph_lab.BROKEN_PATH)
        assert sick.push([scripted_replies.answer_reply("b")] * 2)["queued"] == 2
        assert len(broken.queue) == 2 and model.queue == []
        # The broken copy is a whole engine: its own health and model routes.
        assert http_json("GET", root + "/broken/v1/models")["data"][0]["id"]
        with pytest.raises(urllib.error.HTTPError) as refused:
            http_json("POST", root + scripted_replies.SCRIPT_PATH, {"nope": 1})
        assert refused.value.code == engine.BAD_REQUEST


def test_a_real_engine_is_not_mistaken_for_the_scripted_one() -> None:
    port = free_port()
    with pytest.raises(scripted_replies.ScriptError, match="cannot reach"):
        scripted_replies.ScriptClient(f"http://127.0.0.1:{port}").state()


# ------------------------------------------------- printing a rollout


SPAN_DOC = {
    "turns": [
        {
            "node_id": "a",
            "index": 0,
            "n_prompt": 10,
            "n_sampled": 3,
            "discarded": False,
        },
        {
            "node_id": "b",
            "index": 1,
            "n_prompt": 15,
            "n_sampled": 4,
            "discarded": False,
        },
        {"node_id": "c", "index": 2, "n_prompt": 15, "n_sampled": 2, "discarded": True},
    ],
    "sequences": [
        {
            "node_ids": ["a", "b"],
            "role": "agent",
            "loss_mask": [0] * 10 + [1] * 3 + [0] * 2 + [0] * 4,
        },
        {"node_ids": ["a", "c"], "role": "discarded", "loss_mask": []},
    ],
}


def test_spans_follow_from_the_turns_sizes() -> None:
    turns = {t["node_id"]: t for t in SPAN_DOC["turns"]}
    found = rollout_view.spans(SPAN_DOC["sequences"][0], turns)
    assert [(s.turn, s.kind, s.size, s.trained) for s in found] == [
        ("t0", "context", 10, False),
        ("t0", "model", 3, True),
        ("t1", "context", 2, False),  # 15 read, minus the 10 + 3 before it
        ("t1", "model", 4, False),  # mask 0: no logprobs for this reply
    ]


def test_parents_and_shape_come_from_the_rows_paths() -> None:
    assert rollout_view.parents(SPAN_DOC) == {"a": None, "b": "a", "c": "a"}
    line = rollout_view.shape(SPAN_DOC)
    assert line == (
        "1 root (t0) · 2 leaves (t1, t2) · fork t0 -> t1, t2 · discarded: t2"
    )


def test_findings_are_cut_at_their_first_sentence() -> None:
    finding = "[WARN] multiple_roots: 2 roots across 4 turns. Each root is separate."
    assert rollout_view.short_finding(finding) == (
        "[WARN] multiple_roots: 2 roots across 4 turns"
    )
    assert rollout_view.finding_style("[FATAL] x: y") == "fatal"


def test_reply_summary_names_the_tool_call_or_the_text() -> None:
    call = {
        "response_message": {
            "tool_calls": [
                {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ]
        }
    }
    assert rollout_view.reply_summary(call) == "bash: ls"
    assert rollout_view.reply_summary({"response_message": {"content": "hi"}}) == (
        "text: hi"
    )
    assert rollout_view.reply_summary({}) == ""


# ------------------------------------------------- Claude Code's side


def test_the_environment_is_empty_but_for_the_proxy_and_private_folders(
    tmp_path: Path,
) -> None:
    env = claude_code.clean_environment(tmp_path, "http://127.0.0.1:1", "s1", None)
    assert env["HOME"] == str(tmp_path / "state" / "home")
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "state" / "config")
    assert env["ANTHROPIC_API_KEY"] == "s1"
    assert claude_code.REMINDER_ENV not in env  # unset: Claude Code's default
    off = claude_code.clean_environment(tmp_path, "u", "k", claude_code.REMINDER_OFF)
    assert off[claude_code.REMINDER_ENV] == "off"
    assert not set(env) & {"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}


def test_the_command_is_the_projects_plus_partial_messages() -> None:
    argv = claude_code.command(Path("/bin/claude"), "do it")
    assert argv[:3] == ["/bin/claude", "-p", "do it"]
    assert argv[argv.index("--tools") + 1] == "Bash"
    assert "--include-partial-messages" in argv
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings["bashOutputMaxChars"] == 4000


def test_find_claude_refuses_another_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\necho '9.9.9 (Claude Code)'\n")
    fake.chmod(0o755)
    monkeypatch.setenv(claude_code.CLAUDE_BIN_ENV, str(fake))
    with pytest.raises(claude_code.ClaudeNotInstalledError, match="9.9.9"):
        claude_code.find_claude()
    monkeypatch.delenv(claude_code.CLAUDE_BIN_ENV)
    with pytest.raises(claude_code.ClaudeNotInstalledError, match="no Claude Code"):
        claude_code.find_claude(install=False)


class Echo(BaseHTTPRequestHandler):
    """An upstream that answers every POST with what it was sent."""

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        body = self.rfile.read(int(self.headers["Content-Length"]))
        reply = json.dumps(
            {"key": self.headers.get("x-api-key"), "got": json.loads(body)}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet."""


def test_the_relay_passes_requests_through_and_keeps_their_bodies() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    relay = claude_code.Relay(f"http://127.0.0.1:{upstream.server_address[1]}")
    try:
        root = relay.start()
        request = urllib.request.Request(
            root + "/v1/messages",
            data=json.dumps({"messages": [1]}).encode(),
            headers={"x-api-key": "s9", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as answer:
            kind = answer.headers["Content-Type"]
            echoed = json.loads(answer.read())
    finally:
        relay.stop()
        upstream.shutdown()
        upstream.server_close()
    assert echoed == {"key": "s9", "got": {"messages": [1]}}
    assert kind == "text/event-stream"
    assert relay.requests == [
        claude_code.RelayedRequest("/v1/messages", {"messages": [1]})
    ]


class Refuses(BaseHTTPRequestHandler):
    """An upstream that refuses every request, with a Retry-After header."""

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        self.rfile.read(int(self.headers["Content-Length"]))
        reply = b'{"error": "busy"}'
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Retry-After", "7")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_GET(self) -> None:  # noqa: N802
        reply = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet."""


def relay_call(root: str, path: str, body: bytes | None) -> tuple[int, Any, bytes]:
    """Send one request through a relay, refusals included.

    Args:
        root: The relay's root.
        path: The path.
        body: A body (POST), or None (GET).

    Returns:
        The status, the headers and the body.
    """
    request = urllib.request.Request(root + path, data=body)
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status, answer.headers, answer.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def test_the_relay_passes_refusals_and_their_headers_back() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Refuses)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    relay = claude_code.Relay(f"http://127.0.0.1:{upstream.server_address[1]}")
    try:
        root = relay.start()
        status, headers, body = relay_call(root, "/v1/messages", b"[1, 2]")
        bad = relay_call(root, "/v1/messages", b"\xff not json")
        got = relay_call(root, "/health", None)
    finally:
        relay.stop()
        upstream.shutdown()
        upstream.server_close()
    assert (status, headers["Retry-After"], body) == (429, "7", b'{"error": "busy"}')
    assert bad[0] == 429  # relayed though it was not JSON
    assert got[0] == 200 and got[2] == b"ok"
    assert relay.requests == []  # a list and a non-JSON body are not kept


def test_the_relay_answers_502_when_the_proxy_is_gone() -> None:
    relay = claude_code.Relay(f"http://127.0.0.1:{free_port()}")
    try:
        status, _, body = relay_call(relay.start(), "/v1/messages", b"{}")
    finally:
        relay.stop()
    assert status == claude_code.BAD_GATEWAY
    assert body.startswith(b"relay: cannot reach")


def database() -> Path:
    """The task's database, as the runs copy it.

    Returns:
        The bundled clinic database.
    """
    return tasks.DATA_DIR / tasks.find_task("clinic-easy-001").db_file


def test_a_stale_install_is_installed_again_or_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = claude_code.install_folder()
    folder.mkdir(parents=True)
    (folder / claude_code.PATH_FILE).write_text(f"{tmp_path / 'gone' / 'claude'}\n")
    monkeypatch.delenv(claude_code.CLAUDE_BIN_ENV, raising=False)
    with pytest.raises(claude_code.ClaudeNotInstalledError, match="no Claude Code"):
        claude_code.find_claude(install=False)
    fresh = tmp_path / "claude"
    fresh.write_text(f"#!/bin/sh\necho '{claude_code.CLAUDE_VERSION} (Claude Code)'\n")
    fresh.chmod(0o755)
    monkeypatch.setattr(claude_code, "npx_install", lambda _folder: fresh)
    assert claude_code.find_claude() == fresh


def test_a_binary_that_cannot_run_is_a_clean_error(tmp_path: Path) -> None:
    broken = tmp_path / "claude"
    broken.write_text("not a program")  # not executable
    with pytest.raises(claude_code.ClaudeNotInstalledError, match="cannot run"):
        claude_code.installed_version(broken)


def test_claude_code_runs_in_the_box_writable_only_in_the_run(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    claude_code.prepare_run(folder, database())
    argv = claude_code.boxed_command(Path("/bin/claude"), folder, "go")
    assert argv[0] == "/usr/bin/sandbox-exec"
    defines = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "-D")
    assert defines["WORKDIR"] == str((folder / "task").resolve())
    assert defines["STATE_DIR"] == str((folder / "state").resolve())
    assert argv[argv.index("/bin/claude") + 1 :][:2] == ["-p", "go"]


def test_a_run_over_its_time_limit_is_killed_with_its_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in for claude that starts a background job, then hangs.
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\nsleep 60 &\necho $! > child.pid\nsleep 60\n")
    fake.chmod(0o755)
    monkeypatch.setattr(claude_code, "RUN_TIMEOUT_S", 2)
    folder = tmp_path / "run"
    claude_code.prepare_run(folder, database())
    run = claude_code.run_claude(fake, folder, "go", "http://127.0.0.1:1", "k", None)
    assert run.exit_code is None and run.seconds < 30
    child = int((folder / "task" / "child.pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)  # the background job went with it


def test_reminder_reads_notes_roles_and_how_the_system_prompt_grew() -> None:
    note = "<total_tokens>15000000 tokens left</total_tokens>"
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "system", "content": f"env\n{note}"},
    ]
    assert reminder.role_line(messages) == "user system*"
    assert reminder.all_notes(messages, system=[{"text": note}]) == ["15000000"] * 2
    assert reminder.system_change("a", "a") == (
        "system prompt: the same as the call before"
    )
    assert reminder.system_change("ab", "ab\n\nX") == (
        'system prompt: the call before\'s, plus "\\n\\nX" at its end'
    )
    assert reminder.system_change("abc", "abd").endswith("at character 2")


# ------------------------------------------------- the cross-check


def stream_events(replies: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
    """Write a Claude Code stream-json log for some replies.

    Args:
        replies: ``(id, command or "", output tokens)``; an empty command
            makes a text reply.

    Returns:
        The events.
    """
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init"}]
    for message_id, command, tokens in replies:
        start = {"type": "message_start", "message": {"id": message_id}}
        events.append({"type": "stream_event", "event": start})
        if command:
            block = {"type": "tool_use", "name": "Bash", "input": {"command": command}}
        else:
            block = {"type": "text", "text": "done"}
        message = {"id": message_id, "content": [block]}
        events.append({"type": "assistant", "message": message})
        delta = {"type": "message_delta", "usage": {"output_tokens": tokens}}
        events.append({"type": "stream_event", "event": delta})
    return events


def recorded(lengths: list[int], masked: set[int] = frozenset()) -> dict[str, Any]:
    """Write a proxy document for two calls, one agent row.

    Args:
        lengths: Each call's sampled tokens.
        masked: Calls (by index) served without log-probabilities.

    Returns:
        The document.
    """
    commands = ["ls", ""]
    turns = []
    for index, (length, command) in enumerate(zip(lengths, commands, strict=True)):
        if command:
            arguments = json.dumps({"command": command})
            message = {
                "tool_calls": [{"function": {"name": "Bash", "arguments": arguments}}]
            }
        else:
            message = {"content": "done"}
        turns.append(
            {
                "index": index,
                "n_sampled": length,
                "discarded": False,
                "response_message": message,
            }
        )
    trained = [n for i, n in enumerate(lengths) if i not in masked]
    row = {"role": "agent", "turn_lengths": trained}
    return {"turns": turns, "sequences": [row]}


def test_crosscheck_agrees_when_both_accounts_match(
    capsys: pytest.CaptureFixture,
) -> None:
    replies = crosscheck.logged_replies(stream_events([("m1", "ls", 9), ("m2", "", 4)]))
    assert [(r.summary, r.output_tokens) for r in replies] == [
        ("Bash: ls", 9),
        ("text: done", 4),
    ]
    checks = crosscheck.compare(recorded([9, 4]), replies)
    assert all(check.ok for check in checks)
    assert checks[0].detail.startswith("2/2")


def test_crosscheck_catches_a_turn_without_logprobs(
    capsys: pytest.CaptureFixture,
) -> None:
    replies = crosscheck.logged_replies(stream_events([("m1", "ls", 9), ("m2", "", 4)]))
    checks = {c.name: c for c in crosscheck.compare(recorded([9, 4], {1}), replies)}
    assert checks["turns"].ok and checks["completion tokens"].ok
    assert not checks["trainable coverage"].ok
    assert "[9]" in checks["trainable coverage"].detail


def test_crosscheck_catches_a_token_count_that_differs(
    capsys: pytest.CaptureFixture,
) -> None:
    replies = crosscheck.logged_replies(stream_events([("m1", "ls", 8), ("m2", "", 4)]))
    checks = {c.name: c for c in crosscheck.compare(recorded([9, 4]), replies)}
    assert not checks["completion tokens"].ok
    assert "MISMATCH" in capsys.readouterr().out


def test_crosscheck_names_the_missing_files(tmp_path: Path) -> None:
    with pytest.raises(crosscheck.RunError, match="rollout.json"):
        crosscheck.run_folder(str(tmp_path))


# ------------------------------------------------- graph_lab's own parts


def test_retry_on_the_command_line_runs_both_retries() -> None:
    assert [c.name for c in graph_lab.chosen_cases("retry")] == ["retry", "retry-last"]
    assert len(graph_lab.chosen_cases("all")) == len(graph_lab.CASES) == 8
    assert graph_lab.chosen_cases(graph_lab.DRY_RUN) == []


def test_the_dry_run_is_remeasured_from_a_rollouts_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    forked = {
        "reward": 0.0,
        "sequences": [
            {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]},
            {"input_ids": [1, 2, 3, 9], "loss_mask": [0, 1, 1, 1]},
        ],
    }
    single = {"reward": 1.0, "sequences": [{"input_ids": [5, 6], "loss_mask": [0, 1]}]}
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(forked) + "\n" + json.dumps(single) + "\n")
    graph_lab.print_dry_run(graph_lab.read_dry_run(path))
    out = capsys.readouterr().out
    assert f"re-measured on {graph_lab.short_path(path.resolve())}:" in out
    assert "11.8% of the dry run's trainable tokens" in out
    assert "2 rollouts, 1 with rows that share trained tokens" in out
    assert (
        "7 mask-1 tokens, 2 of them repeats (28.6%), 2 in rollouts with reward 0" in out
    )


def test_a_bad_dry_run_file_stops_before_any_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(SystemExit, match="is not a project rollouts.jsonl"):
        graph_lab.read_dry_run(path)
    url = f"http://127.0.0.1:{free_port()}"  # no proxy: never reached
    argv = ["graph_lab.py", "clean", "--proxy-url", url, "--dry-run", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="is not a project rollouts.jsonl"):
        graph_lab.main()


def test_the_changing_line_is_claude_codes_note() -> None:
    messages = [{"role": "system", "content": "old"}]
    graph_lab.with_changing_line(messages, "rules", 14_999_000)
    assert messages[0]["content"] == (
        "rules\n\n<total_tokens>14999000 tokens left</total_tokens>"
    )
    assert reminder.notes(messages[0]["content"]) == ["14999000"]


# ------------------------------------------------- start.sh, for real


@needs_proxy
def test_start_sh_scripted_starts_both_and_stop_stops_both(tmp_path: Path) -> None:
    real_uv = shutil.which("uv")
    if real_uv is None:
        pytest.skip("uv is not installed")
    # A stand-in for `uv`: `uv run scripted_engine.py ...` runs it with this
    # Python (which has FastAPI); anything else goes to the real uv.
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "uv").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = run ] && [ "$2" = scripted_engine.py ]; then\n'
        f'  shift 2; cd "{CHAPTER}"; exec "{sys.executable}" scripted_engine.py "$@"\n'
        "fi\n"
        f'exec "{real_uv}" "$@"\n'
    )
    (shim / "uv").chmod(0o755)
    engine_port, proxy_port = free_port(), free_port()
    settings = {"PATH": f"{shim}{os.pathsep}{os.environ['PATH']}"}
    home = tmp_path / "home"
    started = run_start_sh("scripted", engine_port, proxy_port, home, **settings)
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        assert "starting the engine (scripted_engine.py)" in started.stdout
        assert "engine ready · scripted model (no MLX)" in started.stdout
        assert "capture level: tokens (train rollouts)" in started.stdout
        state = scripted_replies.ScriptClient(f"http://127.0.0.1:{engine_port}").state()
        assert state["queued"] == 0 and state["answered"] >= 1  # the proxy's probe
    finally:
        stopped = run_start_sh("stop", engine_port, proxy_port, home, **settings)
    assert (
        "stopped the proxy" in stopped.stdout and "stopped the engine" in stopped.stdout
    )
    assert "engine stopped ·" in stopped.stdout
    assert not listening(engine_port) and not listening(proxy_port)


# ------------------------------ the seven cases, through the real proxy


def test_clean_chain_is_one_root_one_row_mask_on_the_replies(stack: Stack) -> None:
    outcome, full = run_lab_case(stack, "clean")
    trainer = outcome.trainer
    assert outcome.grade.correct
    assert trainer["stats"]["n_roots"] == 1 and trainer["stats"]["n_sequences"] == 1
    row = trainer["sequences"][0]
    sampled = [t["n_sampled"] for t in full["turns"]]
    assert row["turn_lengths"] == sampled
    assert (
        sum(row["loss_mask"]) == sum(sampled) == trainer["stats"]["n_trainable_tokens"]
    )
    # Mask 1 exactly where the model wrote, and nowhere else.
    turns = {t["node_id"]: t for t in full["turns"]}
    for span in rollout_view.spans(row, turns):
        assert set(row["loss_mask"][span.start : span.end]) <= {
            int(span.kind == "model")
        }
    assert training_rows.count_calls(trainer) == 4
    assert trainer["validation"] == []


def test_a_retried_middle_call_is_discarded_and_kept_from_the_trainer(
    stack: Stack,
) -> None:
    outcome, full = run_lab_case(stack, "retry")
    trainer = outcome.trainer
    assert [t["discarded"] for t in full["turns"]] == [False, True, False, False, False]
    assert trainer["stats"]["n_sequences"] == 1  # the trainer's document
    assert [r["role"] for r in full["sequences"]] == ["discarded", "agent"]
    assert not full["sequences"][0]["trainable"]
    assert turn_findings(trainer, "discarded_turns")
    assert training_rows.count_calls(trainer) == 4  # the abandoned call is not counted


def test_a_retried_last_call_gives_two_rows_the_dedup_trains_once(
    stack: Stack,
) -> None:
    outcome, full = run_lab_case(stack, "retry-last")
    trainer = outcome.trainer
    assert not any(t["discarded"] for t in full["turns"])  # nothing continued
    rows = training_rows.trainable_sequences(trainer)
    assert len(rows) == 2
    deduped = training_rows.dedupe_shared_prefixes(rows)
    shared = training_rows.shared_prefix(rows[0]["input_ids"], rows[1]["input_ids"])
    first_three = sum(t["n_sampled"] for t in full["turns"][:3])
    assert training_rows.trained_tokens(rows) == trainer["stats"]["n_trainable_tokens"]
    # Calls 1-3's replies are trained once, not twice.
    removed = training_rows.trained_tokens(rows) - training_rows.trained_tokens(deduped)
    assert removed >= first_three
    assert sum(rows[1]["loss_mask"][:shared]) == removed
    assert training_rows.count_calls(trainer) == 5  # the abandoned answer counts


def test_a_changing_line_makes_one_root_per_call(stack: Stack) -> None:
    outcome, _ = run_lab_case(stack, "changing-line")
    stats = outcome.trainer["stats"]
    assert stats["n_roots"] == stats["n_turns"] == stats["n_sequences"] == 4
    assert all(row["n_turns"] == 1 for row in outcome.trainer["sequences"])
    assert turn_findings(outcome.trainer, "per_turn_capture_only")
    assert outcome.trainer["trainable"]


def test_compaction_opens_a_new_root(stack: Stack) -> None:
    outcome, full = run_lab_case(stack, "compaction")
    assert outcome.trainer["stats"]["n_roots"] == 2
    roots = [t["index"] for t in full["turns"] if t["root_id"] == t["node_id"]]
    assert roots == [0, 2]  # the call right after the summary starts over
    assert turn_findings(outcome.trainer, "multiple_roots")


def test_a_turn_without_logprobs_stays_as_context_with_a_finding(
    stack: Stack,
) -> None:
    outcome, full = run_lab_case(stack, "no-logprobs")
    trainer = outcome.trainer
    third = full["turns"][2]
    [finding] = turn_findings(trainer, "no_logprobs")
    assert finding.startswith(
        f"[FATAL] no_logprobs: turn 2: {third['n_sampled']} sampled"
    )
    row = trainer["sequences"][0]
    turns = {t["node_id"]: t for t in full["turns"]}
    reply = [s for s in rollout_view.spans(row, turns) if s.kind == "model"][2]
    assert set(row["loss_mask"][reply.start : reply.end]) == {0}
    assert set(row["logprobs"][reply.start : reply.end]) == {0.0}
    assert trainer["trainable"]  # the other three replies still train
    expected = sum(t["n_sampled"] for t in full["turns"]) - third["n_sampled"]
    assert trainer["stats"]["n_trainable_tokens"] == expected


def test_a_side_call_is_auxiliary_and_not_an_agent_step(stack: Stack) -> None:
    outcome, full = run_lab_case(stack, "side-call")
    trainer = outcome.trainer
    side = full["turns"][1]
    assert side["n_tools"] == 0
    roles = {r["role"]: r for r in trainer["sequences"]}
    assert roles["auxiliary"]["trainable"] is False
    assert training_rows.count_calls(trainer) == 4
    # n_sampled is the project's sum over every live turn: the title counts.
    assert training_rows.n_sampled(trainer) == sum(
        t["n_sampled"] for t in full["turns"]
    )


def test_a_broken_engine_solves_the_task_and_trains_on_nothing(
    stack: Stack, capsys: pytest.CaptureFixture
) -> None:
    outcome, _ = run_lab_case(stack, "broken-engine")
    out = capsys.readouterr().out
    assert "the proxy probed it first: capture level tokens (train)" in out
    assert outcome.grade.correct
    trainer = outcome.trainer
    assert trainer["trainable"] is False
    assert trainer["stats"]["n_trainable_tokens"] == 0
    assert len(turn_findings(trainer, "no_logprobs")) == 4
    assert training_rows.trainable_sequences(trainer) == []
    assert stack.broken.queue == []  # every scripted reply was asked for


def test_graph_lab_main_prints_every_case_and_the_table(
    stack: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["graph_lab.py", "--proxy-url", stack.proxy_url, "--color", "never"]
    monkeypatch.setattr(sys, "argv", argv)
    graph_lab.main()
    out = capsys.readouterr().out
    assert out.startswith(f"graph_lab.py · the capture proxy at {stack.proxy_url}")
    for found in graph_lab.CASES:
        assert f"· {found.name} · {found.title}" in out
    assert "dry-run · shared-prefix repeats" in out
    table = out.split("every case solved the same task the same way")[1]
    assert table.count("correct") == len(graph_lab.CASES)
    assert stack.model.queue == []
    sessions = stack.client().send("GET", "/sessions")["sessions"]
    assert sessions == []  # every case deleted its session


def test_graph_lab_refuses_a_proxy_that_is_not_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"http://127.0.0.1:{free_port()}"
    monkeypatch.setattr(sys, "argv", ["graph_lab.py", "clean", "--proxy-url", url])
    with pytest.raises(SystemExit, match="start the stack with ./start.sh scripted"):
        graph_lab.main()


# ------------------------- Claude Code 2.1.290, through the real proxy


@needs_claude
def test_reminder_splits_claude_codes_rollout_until_it_is_off(
    stack: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(claude_code.CLAUDE_BIN_ENV, str(CLAUDE_BIN))
    argv = ["reminder.py", "--proxy-url", stack.proxy_url, "--color", "never"]
    monkeypatch.setattr(sys, "argv", argv)
    reminder.main()
    out = capsys.readouterr().out
    default, rest = out.split("━━ CLAUDE_CODE_TOTAL_TOKENS_REMINDER=fixed")
    fixed, off = rest.split("━━ CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off")
    # Claude Code sends each new note as a system message after a tool result...
    assert "messages: user system* assistant user system*" in default
    # ...and the proxy merges every system message into the first one.
    assert 'plus "\\n\\n<total_tokens>' in default
    for split in (default, fixed):  # a number that never changes splits it too
        assert "4 roots (t0, t1, t2, t3)" in split
        assert "per_turn_capture_only" in split
    assert "notes 5000000 5000000 5000000" in fixed
    assert "notes none" in off and "1 root (t0)" in off
    assert out.count("grade correct") == len(reminder.SETTINGS)
    assert "scripted replies were never asked for" not in out
    runs = Path(os.environ["COURSE_HOME"]) / "21-calls-to-rows" / "runs"
    for setting in reminder.SETTINGS:
        monkeypatch.setattr(sys, "argv", ["crosscheck.py", f"reminder-{setting}"])
        crosscheck.main()  # exits 1 on any disagreement
        checked = capsys.readouterr().out
        assert "turns               ok · 4/4" in checked
        assert "completion tokens   ok" in checked
        assert (runs / f"reminder-{setting}" / reminder.SENT_FILE).is_file()


# ------------------------------------------------- the review's edge cases


def test_a_script_must_give_text_and_a_real_true_or_false() -> None:
    good = scripted_engine.read_replies({"replies": [{"text": "a"}]})
    assert good == [scripted_replies.Reply("a", True)]
    for bad in (
        {"replies": [{"text": None}]},
        {"replies": [{"text": "a", "logprobs": "false"}]},
        {"replies": "a"},
        [1],
    ):
        with pytest.raises(ValueError):
            scripted_engine.read_replies(bad)


def test_the_default_reply_follows_the_tools_the_request_offered() -> None:
    core = engine.Engine(scripted_engine.ScriptedModel())
    with_tools = answer_body(core, ASK | {"tools": TOOLS})
    plain = answer_body(core, ASK)
    assert with_tools["choices"][0]["message"]["tool_calls"]
    assert plain["choices"][0]["message"]["content"] == scripted_engine.DEFAULT_ANSWER
    # Text that looks like the template's tool list does not fool it.
    mimic = {"messages": [{"role": "user", "content": "List of tools: none"}]}
    assert answer_body(core, mimic)["choices"][0]["message"].get("tool_calls") is None


def test_a_partly_masked_reply_and_a_null_logprob_are_shown_as_they_are() -> None:
    turns = {"a": {"index": 0, "n_prompt": 2, "n_sampled": 4}}
    row = {
        "node_ids": ["a"],
        "loss_mask": [0, 0, 1, 1, 0, 0],
        "logprobs": [0.0, 0.0, None, -0.5, 0.0, 0.0],
    }
    reply = rollout_view.spans(row, turns)[1]
    assert reply.mask_label() == "mask 1 on 2 of 4"
    assert rollout_view.logprob_line(row, reply) == "logprobs - -0.500 0.000 0.000"


def test_crosscheck_keeps_a_missing_count_missing(tmp_path: Path) -> None:
    events = stream_events([("m1", "ls", 9), ("m2", "", 4)])
    events = [
        e for e in events if (e.get("event") or {}).get("type") != "message_delta"
    ]
    replies = crosscheck.logged_replies(events + [{"type": "assistant", "message": {}}])
    assert [r.output_tokens for r in replies] == [None, None]
    checks = {c.name: c for c in crosscheck.compare(recorded([9, 4]), replies)}
    assert "--include-partial-messages" in checks["completion tokens"].detail
    (tmp_path / crosscheck.ROLLOUT_FILE).write_text("{not json")
    with pytest.raises(crosscheck.RunError, match="cannot read"):
        crosscheck.read_rollout(tmp_path)


def test_reminder_counts_only_the_model_calls_claude_code_sent() -> None:
    kept = [
        claude_code.RelayedRequest("/v1/messages?beta=true", {"messages": []}),
        claude_code.RelayedRequest("/v1/messages/count_tokens", {"messages": []}),
        claude_code.RelayedRequest("/v1/messages", {"model": "x"}),
    ]
    assert reminder.model_calls(kept) == [{"messages": []}]
