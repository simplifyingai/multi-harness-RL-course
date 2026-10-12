"""Chapter 20 without a model: the REAL capture proxy, in front of a scripted engine.

No test here loads weights or needs MLX. The engine is the chapter's own
``engine.py`` with a scripted model behind its seam, like chapter 18's tests
(``fake_engine_20.ScriptedModel``), served over real HTTP on a free port.
In front of it runs OpenEnv's capture proxy itself, the pinned commit that
``start.sh`` installed, started by ``start.sh proxy`` exactly as a learner
starts it: so every rule checked here (the capture level, the training
refusal, the 401, the sampling pins, the output clamp, the dialects, the
call budget) is the proxy's own code, not a copy of it. Those tests are
skipped when the proxy is not installed (``./start.sh install``).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import use_chapter

CHAPTER = use_chapter("20-capture-proxy")

import capture_client  # noqa: E402
import engine  # noqa: E402
import probes  # noqa: E402
import session_demo  # noqa: E402
import tap  # noqa: E402
from fake_engine_20 import ScriptedModel, answer_reply, call_reply  # noqa: E402

START_SH = CHAPTER / "start.sh"
FAKE_ENGINE = Path(__file__).with_name("fake_engine_20.py")
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where start.sh installed the proxy on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
REAL_PROXY_ENV = Path(
    os.environ.get("PROXY_ENV") or _HOME / "20-capture-proxy" / "proxy-env"
)
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)


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


# Only the pinned proxy: a different one would make start.sh reinstall it.
HAVE_PROXY = PINNED is not None and installed_commit(REAL_PROXY_ENV) == PINNED[1]
needs_proxy = pytest.mark.skipif(
    not HAVE_PROXY,
    reason="the pinned capture proxy is not installed: ./start.sh install",
)
START_TIMEOUT_S = 300
MURCIA = "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
SOLVE = (
    call_reply("sqlite3 data.db .tables"),
    call_reply(MURCIA, think="Count them."),
    call_reply('echo "30" > answer.txt', think="Write it."),
    answer_reply("I wrote 30 to answer.txt."),
)
REFUSED_TRAINING = "HTTP 400: training requires exact engine token capture"
TOKENS_LINE = "capture level: tokens (train rollouts)"
LOGPROBS_LINE = "capture level: logprobs (eval rollouts)"


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
def live_engine(
    token_fields: bool = True, raw_logprobs: bool = False
) -> Iterator[tuple[int, ScriptedModel]]:
    """Serve the chapter's engine, with the scripted model, over real HTTP.

    Args:
        token_fields: False for ``--no-token-fields``.
        raw_logprobs: True for ``--raw-logprobs``.

    Yields:
        The port, and the model (to queue replies and read what it was asked).
    """
    model = ScriptedModel()
    sock = engine.claim_port(0)
    core = engine.Engine(model, token_fields=token_fields, raw_logprobs=raw_logprobs)
    server = uvicorn.Server(
        uvicorn.Config(engine.create_app(core), log_level="warning")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("the scripted engine did not start within 10 s")
    try:
        yield sock.getsockname()[1], model
    finally:
        server.should_exit = True
        thread.join(10)


def run_start_sh(
    command: str,
    engine_port: int,
    proxy_port: int,
    home: Path,
    **settings: str,
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``start``, ``proxy``, ``stop`` or ``status``.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).
        **settings: More environment, such as ``ENGINE_FLAGS`` or ``PATH``.

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
        model: The scripted model behind the engine.
        started: What ``start.sh proxy`` printed.
        ports: The engine's port and the proxy's.
        home: The COURSE_HOME start.sh was given.
    """

    proxy_url: str
    engine_url: str
    model: ScriptedModel
    started: str
    ports: tuple[int, int]
    home: Path

    def probes(self) -> probes.Probes:
        """A fresh set of probes against this proxy.

        Returns:
            The probes' shared state.
        """
        return probes.Probes(capture_client.CaptureClient(self.proxy_url))


@contextlib.contextmanager
def stack(
    home: Path, token_fields: bool = True, raw_logprobs: bool = False
) -> Iterator[Stack]:
    """Start a scripted engine and the real proxy in front of it.

    Args:
        home: COURSE_HOME for start.sh.
        token_fields: False for ``--no-token-fields``.
        raw_logprobs: True for ``--raw-logprobs``.

    Yields:
        The stack.
    """
    with live_engine(token_fields, raw_logprobs) as (engine_port, model):
        proxy_port = free_port()
        started = run_start_sh("proxy", engine_port, proxy_port, home)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            yield Stack(
                f"http://127.0.0.1:{proxy_port}",
                f"http://127.0.0.1:{engine_port}",
                model,
                started.stdout,
                (engine_port, proxy_port),
                home,
            )
        finally:
            run_start_sh("stop", engine_port, proxy_port, home)


@pytest.fixture(scope="module")
def tokens(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The proxy in front of an engine that keeps every receipt."""
    if not HAVE_PROXY:
        pytest.skip("the capture proxy is not installed: ./start.sh install")
    with stack(tmp_path_factory.mktemp("tokens")) as running:
        yield running


@pytest.fixture(scope="module")
def no_tokens(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The proxy in front of an engine started with --no-token-fields."""
    if not HAVE_PROXY:
        pytest.skip("the capture proxy is not installed: ./start.sh install")
    with stack(tmp_path_factory.mktemp("no-tokens"), token_fields=False) as running:
        yield running


@pytest.fixture(scope="module")
def raw(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The proxy in front of an engine started with --raw-logprobs."""
    if not HAVE_PROXY:
        pytest.skip("the capture proxy is not installed: ./start.sh install")
    with stack(tmp_path_factory.mktemp("raw"), raw_logprobs=True) as running:
        yield running


def run_probe(running: Stack, name: str, capsys: pytest.CaptureFixture) -> str:
    """Run one probe against a stack and return what it printed.

    Args:
        running: The stack.
        name: The probe.
        capsys: pytest's output capture.

    Returns:
        The probe's output.
    """
    shared = running.probes()
    try:
        probes.PROBE_FUNCTIONS[name](shared)
    finally:
        shared.close()
    return capsys.readouterr().out


def answer(core: engine.Engine, body: dict[str, Any]) -> dict[str, Any]:
    """Answer one request in this thread, as the server's worker would.

    Args:
        core: The engine.
        body: The request.

    Returns:
        The response body.
    """
    completion = core.complete(engine.read_request(body))
    return engine.to_openai(core.model, completion, 0.0, core.token_fields)


ASK = {"messages": [{"role": "user", "content": "hi"}], "top_logprobs": 2}


# ------------------------------------------------------- the engine's switches


def test_by_default_the_engine_keeps_chapter_18s_receipts() -> None:
    response = answer(engine.Engine(ScriptedModel()), ASK)
    choice = response["choices"][0]
    entries = choice["logprobs"]["content"]
    assert response["prompt_token_ids"] and choice["token_ids"]
    assert len(entries) == len(choice["token_ids"])
    assert all(entry["token"].startswith("token_id:") for entry in entries)


def test_no_token_fields_leaves_the_ids_out_and_names_tokens_by_text() -> None:
    response = answer(engine.Engine(ScriptedModel(), token_fields=False), ASK)
    choice = response["choices"][0]
    assert "prompt_token_ids" not in response
    assert "token_ids" not in choice
    entries = choice["logprobs"]["content"]
    assert entries[0]["token"] == "D"  # "Done.</think>ok": its first character
    assert entries[0]["top_logprobs"][0]["token"] == "d"  # fake token 200
    assert response["usage"]["completion_tokens"] == len(entries)


@pytest.mark.parametrize(
    ("raw_logprobs", "gaps"), [(False, [2.0, 1.0]), (True, [2.0, 2.0])]
)
def test_raw_logprobs_keep_the_gap_at_twice_the_temperature(
    raw_logprobs: bool, gaps: list[float]
) -> None:
    model = ScriptedModel()
    core = engine.Engine(model, raw_logprobs=raw_logprobs)
    measured = []
    for temperature in (1.0, 2.0):
        body = ASK | {"temperature": temperature, "max_tokens": 1, "top_logprobs": 5}
        measured.append(tap.logprob_gap(answer(core, body)))
    assert measured == pytest.approx(gaps)
    assert model.seen[-1].raw_logprobs is raw_logprobs
    assert model.seen[-1].temperature == 2.0  # still drawn at the temperature


def test_engine_flags_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["engine.py", "--port", "8119", "--no-token-fields", "--raw-logprobs"]
    monkeypatch.setattr(sys, "argv", argv)
    args = engine.parse_args()
    assert (args.port, args.no_token_fields, args.raw_logprobs) == (8119, True, True)


# ---------------------------------------------------------------- the tap


def exchange(temperature: float, values: list[float]) -> tap.Exchange:
    """Write one recorded probe exchange whose first position has these logprobs.

    Args:
        temperature: The request's temperature.
        values: The alternatives' log-probabilities.

    Returns:
        The exchange.
    """
    tops = [{"token": f"token_id:{i}", "logprob": v} for i, v in enumerate(values)]
    response = {"choices": [{"logprobs": {"content": [{"top_logprobs": tops}]}}]}
    request = {"temperature": temperature, "top_logprobs": len(values)}
    return tap.Exchange("POST", "/v1/chat/completions", request, 200, response)


def test_gap_verdict_reads_processed_raw_and_unknown() -> None:
    tempered = [exchange(2.0, [-0.2, -1.2]), exchange(1.0, [-0.1, -2.1])]
    assert tap.gap_verdict(tempered) == ([2.0, 1.0], tap.PROCESSED)
    raw = [exchange(1.0, [-0.1, -2.1]), exchange(2.0, [-0.1, -2.1])]
    assert tap.gap_verdict(raw) == ([2.0, 2.0], tap.RAW)
    flat = [exchange(1.0, [-0.6, -0.7]), exchange(2.0, [-0.6, -0.65])]
    assert tap.gap_verdict(flat)[1] == tap.UNKNOWN
    sentinel = [exchange(1.0, [0.0, -9999.0]), exchange(2.0, [0.0, -9999.0])]
    assert tap.gap_verdict(sentinel) == ([], tap.UNKNOWN)


def test_the_tap_relays_both_ways_and_records_refusals_too() -> None:
    with live_engine() as (port, _):
        relay = tap.Tap(f"http://127.0.0.1:{port}")
        url = relay.start()
        try:
            ok = probes.post(url + probes.CHAT_PATH, ASK)
            refused = probes.post(url + probes.CHAT_PATH, ASK | {"stream": True})
            first, second = relay.take()
        finally:
            relay.stop()
    assert ok.status == 200 and ok.json()["choices"][0]["token_ids"]
    assert first.request == ASK and first.response == ok.json()
    assert refused.status == second.status == 400
    assert "stream is not supported" in refused.text
    assert relay.take() == []


def test_the_tap_answers_502_when_the_engine_is_gone() -> None:
    relay = tap.Tap(f"http://127.0.0.1:{free_port()}")
    url = relay.start()
    try:
        reply = probes.post(url + probes.CHAT_PATH, ASK)
    finally:
        relay.stop()
    assert reply.status == tap.BAD_GATEWAY
    assert "tap:" in probes.error_message(reply)


def test_describe_response_names_what_came_back() -> None:
    with_ids = answer(engine.Engine(ScriptedModel()), ASK)
    record = tap.Exchange("POST", "/v1/chat/completions", ASK, 200, with_ids)
    assert "tokens named by token_id:<n>" in tap.describe_response(record)
    without = answer(engine.Engine(ScriptedModel(), token_fields=False), ASK)
    record = tap.Exchange("POST", "/v1/chat/completions", ASK, 200, without)
    line = tap.describe_response(record)
    assert line.startswith("no prompt_token_ids · no token_ids")
    assert "tokens named by text" in line


# ------------------------------------------------- the probes' own helpers


def test_server_sent_events_skip_comments_and_rebuild_the_tool_call() -> None:
    stream = (
        ": openenv keepalive\n\n"
        "event: content_block_start\n"
        'data: {"type": "content_block_start", "content_block": '
        '{"type": "tool_use", "name": "bash", "input": {}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type": "content_block_delta", "delta": '
        '{"type": "input_json_delta", "partial_json": "{\\"command\\": \\"ls\\"}"}}\n\n'
        "event: message_delta\n"
        'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}\n\n'
    )
    events = probes.server_sent_events(stream)
    assert [event["type"] for event in events] == [
        "content_block_start",
        "content_block_delta",
        "message_delta",
    ]
    assert probes.tool_use_from_events(events) == 'bash {"command": "ls"}'
    assert probes.stop_reason(events) == "tool_use"


def test_anthropic_tool_is_the_same_schema_spelled_anthropics_way() -> None:
    tool = probes.bash_schema()
    written = probes.anthropic_tool(tool)
    assert written == {
        "name": "bash",
        "description": tool["function"]["description"],
        "input_schema": tool["function"]["parameters"],
    }


# ------------------------------------------------------ the capture client


def test_summarize_reads_the_rollout_document() -> None:
    document = {
        "turns": [
            {"n_sampled": 10, "n_prompt": 100, "discarded": False},
            {"n_sampled": 99, "n_prompt": 150, "discarded": True},
            {"n_sampled": 5, "n_prompt": 130, "discarded": False},
        ],
        "stats": {
            "n_turns": 3,
            "n_roots": 1,
            "n_sequences": 1,
            "n_trainable_tokens": 15,
        },
        "trainable": True,
        "rollout_type": "train",
        "budget_stop_count": 1,
        "validation": ["[INFO] forks: 1 fork point(s)", "[WARN] something"],
    }
    summary = capture_client.summarize(document)
    assert (summary.sampled_tokens, summary.last_prompt_tokens) == (15, 130)
    assert (summary.turns, summary.roots, summary.budget_stops) == (3, 1, 1)
    assert summary.trainable and summary.findings == ["[WARN] something"]


def test_a_refusal_reads_as_status_and_message() -> None:
    body = json.dumps({"error": {"message": "nope", "type": "x"}}).encode()
    error = urllib.error.HTTPError("http://x", 400, "Bad", {}, io.BytesIO(body))
    assert capture_client.refusal_text(error) == "HTTP 400: nope"


def test_an_absent_proxy_is_not_a_refusal() -> None:
    client = capture_client.CaptureClient(f"http://127.0.0.1:{free_port()}")
    with pytest.raises(capture_client.ProxyError) as caught:
        client.open_session(capture_client.PURPOSE_TRAIN, {})
    assert caught.value.status is None


# ------------------------------------------------- start.sh, for real


@needs_proxy
def test_start_sh_starts_the_engine_then_the_proxy_and_stops_both(
    tmp_path: Path,
) -> None:
    real_uv = shutil.which("uv")
    if real_uv is None:
        pytest.skip("uv is not installed")
    # A stand-in for `uv`: `uv run engine.py ...` starts the scripted engine
    # with the same flags; anything else goes to the real uv.
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "uv").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = run ] && [ "$2" = engine.py ]; then\n'
        f'  shift 2; exec "{sys.executable}" "{FAKE_ENGINE}" "$@"\n'
        "fi\n"
        f'exec "{real_uv}" "$@"\n'
    )
    (shim / "uv").chmod(0o755)
    engine_port, proxy_port = free_port(), free_port()
    settings = {
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "ENGINE_FLAGS": "--no-token-fields",
    }
    home = tmp_path / "home"
    started = run_start_sh("start", engine_port, proxy_port, home, **settings)
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        assert f"the engine on http://127.0.0.1:{engine_port}" in started.stdout
        assert "(--no-token-fields)" in started.stdout
        assert "engine ready · scripted model" in started.stdout
        assert LOGPROBS_LINE in started.stdout
        status = run_start_sh("status", engine_port, proxy_port, home, **settings)
        assert '"capture_level":"logprobs"' in status.stdout
        assert (home / "20-capture-proxy" / "run" / "engine.pid").is_file()
    finally:
        stopped = run_start_sh("stop", engine_port, proxy_port, home, **settings)
    assert "stopped the proxy" in stopped.stdout
    assert "stopped the engine" in stopped.stdout
    # Ctrl-C reached the engine itself, which printed its own last line: the
    # proxy's probe made 2 requests (no logprob test without token ids).
    assert "engine stopped · 2 requests" in stopped.stdout
    assert not listening(engine_port) and not listening(proxy_port)
    assert not (home / "20-capture-proxy" / "run" / "engine.pid").exists()


@needs_proxy
def test_start_sh_will_not_share_the_proxy_port(tmp_path: Path) -> None:
    with live_engine() as (engine_port, _), socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        taken = squatter.getsockname()[1]
        result = run_start_sh("proxy", engine_port, taken, tmp_path)
    assert result.returncode == 1
    assert f"port {taken} is taken by another program" in result.stderr


def test_stop_leaves_alone_a_process_it_did_not_start(tmp_path: Path) -> None:
    # A pid file left from before a reboot: the pid now belongs to a stranger,
    # which even leads its own process group, as the engine did.
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    run = tmp_path / "20-capture-proxy" / "run"
    run.mkdir(parents=True)
    (run / "engine.pid").write_text(f"{stranger.pid}\n")
    (run / "engine.cmd").write_text("uv run engine.py --port 8119\n")
    try:
        stopped = run_start_sh("stop", free_port(), free_port(), tmp_path)
        assert stranger.poll() is None  # still running
    finally:
        stranger.kill()
        stranger.wait()
    assert "the engine's pid file names no process this script started" in (
        stopped.stdout
    )
    assert not (run / "engine.pid").exists()


def test_install_never_deletes_a_folder_that_is_not_an_environment(
    tmp_path: Path,
) -> None:
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me")
    result = run_start_sh(
        "install", free_port(), free_port(), tmp_path, PROXY_ENV=str(precious)
    )
    assert result.returncode == 1
    assert "is not a Python environment" in result.stderr
    assert (precious / "notes.txt").read_text() == "keep me"


def test_a_non_json_answer_is_a_proxy_error() -> None:
    with live_engine() as (port, _):
        client = capture_client.CaptureClient(f"http://127.0.0.1:{port}")
        with pytest.raises(capture_client.ProxyError, match="did not answer in JSON"):
            client.send("GET", "/docs")  # FastAPI's HTML page, with a 200


def test_closing_the_probes_survives_a_dead_proxy() -> None:
    shared = probes.Probes(
        capture_client.CaptureClient(f"http://127.0.0.1:{free_port()}")
    )
    shared.opened.append("s-gone")
    shared.tap = tap.Tap(f"http://127.0.0.1:{free_port()}")
    port = int(shared.tap.start().rsplit(":", 1)[1])
    shared.close()
    assert shared.tap is None and shared.opened == []
    assert not listening(port)


def test_a_proxy_that_listens_on_every_interface_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in for a proxy that has lost its refusal: it just keeps running.
    fake_env = tmp_path / "env"
    (fake_env / "bin").mkdir(parents=True)
    (fake_env / "bin" / "python").write_text("#!/bin/sh\necho serving\nsleep 30\n")
    (fake_env / "bin" / "python").chmod(0o755)
    monkeypatch.setenv("PROXY_ENV", str(fake_env))
    monkeypatch.setattr(probes, "REFUSAL_WAIT_S", 1)
    _, code, last = probes.open_interface_refusal("http://127.0.0.1:1")
    assert code != 0
    assert last == "IT STARTED, listening on every interface (stopped): serving"


# ------------------------------ the real proxy, in front of a full engine


def test_start_up_says_the_capture_level(tokens: Stack) -> None:
    assert TOKENS_LINE in tokens.started
    health = capture_client.CaptureClient(tokens.proxy_url).health()
    assert (health["capture_level"], health["rollout_type"]) == ("tokens", "train")
    assert health["upstream"] == tokens.engine_url


def test_start_sh_will_not_start_a_second_proxy(tokens: Stack) -> None:
    again = run_start_sh("proxy", *tokens.ports, tokens.home)
    assert again.returncode == 1
    assert "the proxy this script started is still running" in again.stderr


def test_level_opens_training_and_redoes_the_probe(
    tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    out = run_probe(tokens, "level", capsys)
    assert "engine " + tokens.engine_url + " · capture level tokens · train" in out
    assert "rollout type train" in out
    assert "GET /v1/models" in out
    assert "return_token_ids true" in out
    assert "tools: bash · temperature 1.0 · max_tokens 512" in out
    assert "gap 2.0000 -> 1.0000, ratio 0.500: processed log-probabilities" in out
    assert "the proxy's verdict on this engine: tokens" in out


def test_only_session_keys_are_answered(
    tokens: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROXY_ENV", str(REAL_PROXY_ENV))
    out = run_probe(tokens, "key", capsys)
    unknown = "HTTP 401: unknown API key; register a session via POST /sessions"
    assert out.count(unknown) == 2  # a made-up key, and no key at all
    assert "HTTP 200: answered (stop)" in out
    assert "no admin key is needed while the proxy listens on 127.0.0.1" in out
    assert "--host 0.0.0.0" in out
    assert "exit code 1: --admin-key or $OPENENV_CAPTURE_ADMIN_KEY is required" in out


def test_training_pins_temperature_1_and_evaluation_0(
    tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    before = len(tokens.model.seen)
    out = run_probe(tokens, "sampling", capsys)
    pinned = (
        "the engine received: temperature 1.0 · top_p 1.0 · top_k -1 · min_p 0.0 "
        "· frequency_penalty 0.0 · presence_penalty 0.0 · repetition_penalty 1.0"
    )
    assert pinned in out
    assert "the engine received: temperature 0.0 · top_p 1.0\n" in out
    kept = 'the rollout keeps the ask: {"temperature": 0.2, "top_p": 0.9}'
    assert out.count(kept) == 2
    # The model itself, behind the engine, drew at 1.0 and then at 0.0.
    asked = [sampling.temperature for sampling in tokens.model.seen[before:]]
    assert asked[-2:] == [1.0, 0.0]


def test_max_tokens_is_clamped_to_1024(
    tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    out = run_probe(tokens, "clamp", capsys)
    assert "asked max_tokens 32,000   -> the engine received max_tokens 1024" in out
    assert "asked no max_tokens       -> the engine received max_tokens 1024" in out
    assert "asked max_tokens 200      -> the engine received max_tokens 200" in out


def test_two_dialects_reach_the_engine_in_one_shape(
    tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    out = run_probe(tokens, "dialects", capsys)
    assert "both: POST /v1/chat/completions and POST /v1/chat/completions" in out
    assert "messages: identical" in out and "tools: identical" in out
    assert "stream: false and false" in out
    assert 'differs: tool_choice: (absent) and "auto"' in out
    assert 'application/json · one JSON body · bash {"command": "ls"}' in out
    assert "text/event-stream" in out
    assert (
        "message_start, content_block_start, content_block_delta, "
        "content_block_stop, message_delta, message_stop"
    ) in out
    assert 'bash {"command": "ls"} · stop_reason tool_use' in out
    assert "dialect openai_chat · 1 turn" in out
    assert "dialect anthropic · 1 turn" in out


def test_the_ninth_call_is_answered_by_the_proxy_and_not_recorded(
    tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    before = len(tokens.model.seen)
    out = run_probe(tokens, "budget", capsys)
    for number in range(1, 9):
        assert f"call {number}: answered by the engine · 'ok'" in out
    assert (
        "call 9: answered by the proxy itself · 'Step budget exhausted; stopping.'"
        " · 0 tokens"
    ) in out
    assert "the rollout: 8 turns recorded, in 1 conversation · budget stops: 1" in out
    # The model was asked 8 times for a short reply (the proxy's probe of the
    # tap asked for other lengths): the ninth call never reached it.
    asked = tokens.model.seen[before:]
    budget_calls = [s for s in asked if s.max_tokens == probes.BUDGET_REPLY_TOKENS]
    assert len(budget_calls) == 8


def test_probes_main_deletes_its_sessions(
    tokens: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = capture_client.CaptureClient(tokens.proxy_url)
    before = len(client.send("GET", "/sessions")["sessions"])
    argv = ["probes.py", "budget", "--proxy-url", tokens.proxy_url]
    monkeypatch.setattr(sys, "argv", argv)
    probes.main()
    out = capsys.readouterr().out
    assert out.startswith(f"probes.py · the capture proxy at {tokens.proxy_url}")
    assert len(client.send("GET", "/sessions")["sessions"]) == before


def test_session_demo_records_a_trainable_rollout(
    tokens: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens.model.push(*SOLVE)
    before = len(tokens.model.seen)
    argv = ["session_demo.py", "--proxy-url", tokens.proxy_url]
    monkeypatch.setattr(sys, "argv", argv)
    session_demo.main()
    out = capsys.readouterr().out
    assert "POST /sessions · purpose train · 8 calls -> s" in out
    assert f"base URL {tokens.proxy_url}/v1, API key s" in out
    assert "grade: correct (correct; answer.txt: '30') · 4 calls" in out
    pinned = (
        'the harness asked {"temperature": 0.0}; the proxy sent {"temperature": 1.0'
    )
    assert pinned in out
    assert "turns 4 · conversations (roots) 1" in out
    assert "train rollout · trainable: true" in out
    assert "DELETE /sessions/s" in out
    assert [s.temperature for s in tokens.model.seen[before:]] == [1.0] * 4
    saved = json.loads(
        next(
            (Path(os.environ["COURSE_HOME"]) / "20-capture-proxy").rglob("*.json")
        ).read_text()
    )
    assert saved["trainable"] is True and saved["stats"]["n_turns"] == 4


# ------------------------- the real proxy, in front of the two broken engines


def test_without_token_fields_the_proxy_refuses_training(
    no_tokens: Stack, capsys: pytest.CaptureFixture
) -> None:
    assert LOGPROBS_LINE in no_tokens.started
    out = run_probe(no_tokens, "level", capsys)
    assert f"refused: {REFUSED_TRAINING}" in out
    assert "an evaluation session opens" in out
    assert "back: no prompt_token_ids · no token_ids" in out
    assert "gap between" not in out  # below tokens, the logprob test is not asked
    assert "the proxy's verdict on this engine: logprobs" in out


def test_session_demo_deletes_its_session_when_something_fails(
    tokens: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = capture_client.CaptureClient(tokens.proxy_url)
    before = len(client.send("GET", "/sessions")["sessions"])

    def broken(self: capture_client.CaptureClient, session_id: str) -> dict:
        raise capture_client.ProxyError("HTTP 500: broken on purpose", 500)

    monkeypatch.setattr(capture_client.CaptureClient, "rollout", broken)
    tokens.model.push(*SOLVE)
    monkeypatch.setattr(sys, "argv", ["s", "--proxy-url", tokens.proxy_url])
    with pytest.raises(SystemExit, match="the proxy failed: HTTP 500"):
        session_demo.main()
    assert len(client.send("GET", "/sessions")["sessions"]) == before


def test_session_demo_without_token_fields(
    no_tokens: Stack, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["s", "--proxy-url", no_tokens.proxy_url])
    with pytest.raises(SystemExit) as caught:
        session_demo.main()
    assert str(caught.value) == f"the proxy refused a train session: {REFUSED_TRAINING}"
    no_tokens.model.push(*SOLVE)
    argv = ["s", "--proxy-url", no_tokens.proxy_url, "--purpose", "eval"]
    monkeypatch.setattr(sys, "argv", argv)
    session_demo.main()
    out = capsys.readouterr().out
    assert "grade: correct" in out
    assert "eval rollout · trainable: false" in out


def test_raw_logprobs_are_caught_by_the_probe(
    raw: Stack, capsys: pytest.CaptureFixture
) -> None:
    assert LOGPROBS_LINE in raw.started
    out = run_probe(raw, "level", capsys)
    assert f"refused: {REFUSED_TRAINING}" in out
    assert "tokens named by token_id:<n>" in out  # every receipt is there...
    assert "gap 2.0000 -> 2.0000, ratio 1.000: raw log-probabilities" in out
    assert "the proxy's verdict on this engine: logprobs" in out
