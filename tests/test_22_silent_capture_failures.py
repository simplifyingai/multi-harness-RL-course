"""Chapter 22 without a model: one capture, replayed and broken, through the REAL proxy.

No test here loads weights or needs MLX. The engine is the chapter's
``scripted_engine.py`` (chapter 20's real ``engine.py`` with a scripted
model behind its seam) served over real HTTP, and in front of it runs
OpenEnv's capture proxy itself, the pinned commit ``start.sh`` installs. The
end-to-end tests run the pinned Claude Code through both, as a learner does:
``capture.py`` records a run, ``breakages.py`` replays it four ways and
``crosscheck.py`` checks each against Claude Code's own log. They are skipped
when the proxy or Claude Code is not on disk; the LFM2.5-tokenizer run is
skipped when LFM2.5's ``tokenizer.json`` is not on disk either.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import use_chapter

CHAPTER = use_chapter("22-silent-capture-failures")

import breakages  # noqa: E402
import capture  # noqa: E402
import captures  # noqa: E402
import claude_code  # noqa: E402
import compaction  # noqa: E402
import crosscheck  # noqa: E402
import engine  # noqa: E402
import lfm_files  # noqa: E402
import scripted_engine  # noqa: E402
import scripted_replies  # noqa: E402
from policy import FINISH_LENGTH, FINISH_STOP, Sampling  # noqa: E402

START_SH = CHAPTER / "start.sh"
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy, Claude Code and LFM2.5 are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
START_TIMEOUT_S = 300
CHAPTERS_WITH_PROXY = ("22-silent-capture-failures", "21-calls-to-rows")
CHAPTERS_WITH_CLAUDE = (
    "22-silent-capture-failures",
    "21-calls-to-rows",
    "16-inside-claude-code",
)
CASES = [case.name for case in breakages.CASES]


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
    """Find a proxy environment at the pinned commit: this chapter's, else 21's.

    Returns:
        The environment, or None.
    """
    named = os.environ.get("PROXY_ENV")
    candidates = [Path(named)] if named else []
    candidates += [_HOME / chapter / "proxy-env" for chapter in CHAPTERS_WITH_PROXY]
    for env in candidates:
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def find_claude_bin() -> Path | None:
    """Find the pinned Claude Code on disk: CLAUDE_BIN, or an earlier install.

    Returns:
        The executable, or None.
    """
    named = os.environ.get(claude_code.CLAUDE_BIN_ENV)
    if named:
        return Path(named)
    for chapter in CHAPTERS_WITH_CLAUDE:
        path_file = _HOME / chapter / claude_code.INSTALL_FOLDER / claude_code.PATH_FILE
        if path_file.is_file():
            binary = Path(path_file.read_text().strip())
            if binary.is_file():
                return binary
    return None


def find_tokenizer() -> Path | None:
    """Find LFM2.5's tokenizer.json, checked, in the real course home.

    Returns:
        The file, or None.
    """
    for name in lfm_files.TOKENIZER_FOLDERS:
        path = _HOME / lfm_files.MODELS_FOLDER / name / lfm_files.TOKENIZER_FILE
        if path.is_file() and lfm_files.sha256_of(path) == lfm_files.TOKENIZER_SHA256:
            return path
    return None


REAL_PROXY_ENV = find_proxy_env()
CLAUDE_BIN = find_claude_bin()
LFM_TOKENIZER_FILE = find_tokenizer()
needs_claude = pytest.mark.skipif(
    REAL_PROXY_ENV is None or CLAUDE_BIN is None,
    reason="needs the capture proxy (./start.sh install) and Claude Code 2.1.290",
)
needs_tokenizer = pytest.mark.skipif(
    LFM_TOKENIZER_FILE is None, reason="LFM2.5's tokenizer.json is not on disk"
)


def free_port() -> int:
    """Ask the system for a port nobody listens on.

    Returns:
        The port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def live_engine(
    cutter: scripted_engine.Cutter | None = None,
) -> Iterator[tuple[int, scripted_engine.ScriptedModel]]:
    """Serve the scripted engine over real HTTP.

    Args:
        cutter: Its tokenizer; the scripted one when None.

    Yields:
        The port and the model.
    """
    app, model = scripted_engine.create_app(cutter)
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
        yield sock.getsockname()[1], model
    finally:
        server.should_exit = True
        thread.join(10)


def run_start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``proxy``, ``stop`` or anything else.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "PROXY_PORT": str(proxy_port),
        "COURSE_HOME": str(home),
        "PROXY_ENV": str(REAL_PROXY_ENV),
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
        model: The engine's scripted model.
    """

    proxy_url: str
    model: scripted_engine.ScriptedModel


@contextlib.contextmanager
def proxy_stack(home: Path, cutter: scripted_engine.Cutter | None) -> Iterator[Stack]:
    """The real proxy, started by start.sh, in front of a scripted engine.

    Args:
        home: A COURSE_HOME for start.sh's logs.
        cutter: The engine's tokenizer.

    Yields:
        The stack.
    """
    with live_engine(cutter) as (engine_port, model):
        proxy_port = free_port()
        started = run_start_sh("proxy", engine_port, proxy_port, home)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            yield Stack(f"http://127.0.0.1:{proxy_port}", model)
        finally:
            run_start_sh("stop", engine_port, proxy_port, home)


# ------------------------------------------------- the engine's replies


def sampling(max_tokens: int = 1_024) -> Sampling:
    """A training request's sampling settings.

    Args:
        max_tokens: The cap on the reply.

    Returns:
        The settings.
    """
    return Sampling(max_tokens=max_tokens, temperature=1.0)


def test_a_written_and_a_replayed_reply_are_both_read() -> None:
    written = scripted_engine.read_reply({"text": "a", "logprobs": False})
    replayed = scripted_engine.read_reply({"ids": [5, 6], "logprobs": [-0.5, None]})
    unscored = scripted_engine.read_reply({"ids": [5], "logprobs": False})
    assert written == scripted_replies.Reply("a", False)
    assert replayed == scripted_replies.Reply("", [-0.5, None], [5, 6])
    assert unscored == scripted_replies.Reply("", False, [5])


@pytest.mark.parametrize(
    "item",
    [
        {"ids": [1, 2], "logprobs": [-0.5]},  # one log-probability short
        {"ids": [1, True]},  # a boolean is not a token id
        {"ids": [-1]},
        {"ids": [1], "logprobs": ["-0.5"]},
        {"ids": [1], "logprobs": "yes"},
        {"text": "a", "logprobs": [-0.5]},  # exact values need exact ids
        "a reply",
    ],
)
def test_a_reply_of_any_other_shape_is_refused(item: Any) -> None:
    with pytest.raises(ValueError):
        scripted_engine.read_reply(item)


def test_a_replayed_reply_comes_back_with_its_own_tokens_and_numbers() -> None:
    model = scripted_engine.ScriptedModel()
    end = scripted_engine.EOS
    model.push([scripted_replies.Reply("", [-0.1, -0.2, -0.3], [150, 151, end])])
    cache = model.new_cache()
    sampled = model.sample([1, 2], cache, sampling())
    assert sampled.token_ids == [150, 151, end]
    assert sampled.logprobs == [-0.1, -0.2, -0.3]
    assert sampled.finish_reason == FINISH_STOP
    assert cache.tokens == [1, 2, 150, 151, end]


def test_a_replayed_reply_without_its_end_token_was_cut_at_the_cap() -> None:
    model = scripted_engine.ScriptedModel()
    model.push([scripted_replies.Reply("", [-0.1] * 4, [150, 151, 152, 153])])
    sampled = model.sample([], model.new_cache(), sampling(max_tokens=3))
    assert sampled.token_ids == [150, 151, 152]
    assert sampled.logprobs == [-0.1] * 3
    assert sampled.finish_reason == FINISH_LENGTH


def test_without_logprobs_keeps_the_tokens_and_drops_every_number() -> None:
    reply = scripted_replies.replayed_reply([150, 3], [-0.5, -0.25])
    model = scripted_engine.ScriptedModel()
    model.push([scripted_replies.without_logprobs(reply)])
    sampled = model.sample([], model.new_cache(), sampling())
    assert sampled.token_ids == [150, 3]
    assert sampled.logprobs == [None, None]
    assert scripted_replies.replayed_reply([1], None).logprobs is False


def test_the_remote_control_reports_its_tokenizer_and_takes_replayed_replies() -> None:
    with live_engine() as (port, model):
        client = scripted_replies.ScriptClient(f"http://127.0.0.1:{port}")
        state = client.push(
            [
                scripted_replies.replayed_reply([150, 3], [-0.5, None]),
                scripted_replies.Reply("hi"),
            ]
        )
        queued = list(model.queue)
    assert state == {"queued": 2, "answered": 0, "tokenizer": "scripted"}
    assert queued == [
        scripted_replies.Reply("", [-0.5, None], [150, 3]),
        scripted_replies.Reply("hi", True),
    ]


@needs_tokenizer
def test_lfm25s_tokenizer_cuts_each_marker_as_one_token_and_round_trips() -> None:
    cutter = scripted_engine.LfmCutter(LFM_TOKENIZER_FILE)
    text = scripted_replies.call_reply("echo 30 > answer.txt", tool="Bash").text
    ids = cutter.encode(text)
    assert cutter.decode(ids) == text
    end = cutter.tokenizer.token_to_id(scripted_replies.END_OF_TURN)
    assert ids[-1] == end and cutter.eos_ids == {end}
    for marker in scripted_replies.MARKERS:
        assert len(cutter.encode(marker)) == 1, marker


def test_the_tokenizer_file_must_have_the_pinned_fingerprint(
    course_home: Path,
) -> None:
    fake = course_home / "models" / lfm_files.OURS_FOLDER / lfm_files.TOKENIZER_FILE
    fake.parent.mkdir(parents=True)
    fake.write_text("{}")
    with pytest.raises(lfm_files.TokenizerMissingError, match="not the pinned file"):
        lfm_files.lfm_tokenizer_file()
    with pytest.raises(ValueError, match="no tokenizer named"):
        scripted_engine.make_cutter("gpt")


# ------------------------------------------------- run folders


def chain_document(
    lengths: list[int],
    rows: list[list[int]] | None = None,
    masked: frozenset[int] = frozenset(),
    prompt: int = 10,
    between: int = 3,
) -> dict[str, Any]:
    """Write a rollout document for one chain of calls, cut into given rows.

    Call ``i`` reads ``n_prompt`` tokens: the first ``prompt``, each later one
    the call before's prompt and reply plus ``between`` tokens. Every row's
    tokens count up from 0, its mask 1 on each reply not ``masked``.

    Args:
        lengths: Each call's reply length.
        rows: The calls each agent row holds (0-based); one row by default.
        masked: Calls served without log-probabilities.
        prompt: The first call's prompt size.
        between: Tokens each call adds after the call before's reply.

    Returns:
        The document.
    """
    starts, at = [], prompt
    for length in lengths:
        starts.append(at)
        at += length + between
    turns = [
        {
            "node_id": f"n{i}",
            "index": i,
            "n_prompt": starts[i],
            "n_sampled": length,
            "n_tools": 1,
            "discarded": False,
            "sampled_logprobs": None if i in masked else [-0.5] * length,
            "response_message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps({"command": f"step {i + 1}"}),
                        }
                    }
                ],
            },
        }
        for i, length in enumerate(lengths)
    ]
    sequences = []
    for calls in rows or [list(range(len(lengths)))]:
        last = calls[-1]
        size = starts[last] + lengths[last]
        mask = [0] * size
        for i in calls:
            if i not in masked:
                mask[starts[i] : starts[i] + lengths[i]] = [1] * lengths[i]
        sequences.append(
            {
                "role": "agent",
                "node_ids": [f"n{i}" for i in calls],
                "input_ids": list(range(size)),
                "loss_mask": mask,
                "trainable": True,
            }
        )
    return {
        "turns": turns,
        "sequences": sequences,
        "validation": [],
        "stats": {"n_roots": len(sequences), "n_sequences": len(sequences)},
        "trainable": True,
    }


def test_a_calls_reply_is_read_out_of_the_row_that_holds_it() -> None:
    document = chain_document([4, 5], masked=frozenset({1}))
    replies = captures.captured_replies(document)
    assert [r.ids for r in replies] == [[10, 11, 12, 13], [17, 18, 19, 20, 21]]
    assert replies[0].logprobs == [-0.5] * 4
    assert replies[1].logprobs is None  # served without them


def test_a_call_no_row_holds_cannot_be_replayed() -> None:
    document = chain_document([4, 5])
    document["sequences"][0]["node_ids"] = ["n0"]
    with pytest.raises(captures.CaptureError, match="call 2"):
        captures.captured_replies(document)


def test_system_change_says_what_grew_or_where_it_parted() -> None:
    assert captures.system_change("a", "a") == "is the same as the call before's"
    grown = captures.system_change("a", "ab")
    assert grown == 'is the call before\'s, plus "b" at its end'
    assert captures.system_change("abc", "abd").endswith("at character 2")


# ------------------------------------------------- the summarising relay


def body(*messages: dict[str, Any]) -> dict[str, Any]:
    """Write an Anthropic request with these messages.

    Args:
        *messages: The messages.

    Returns:
        The body.
    """
    return {"system": [{"type": "text", "text": "sys"}], "messages": list(messages)}


def tool_use(number: int) -> dict[str, Any]:
    """An assistant message calling Bash once.

    Args:
        number: The call's number (its id and command).

    Returns:
        The message.
    """
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"t{number}",
                "name": "Bash",
                "input": {"command": f"cmd {number}"},
            }
        ],
    }


def tool_result(number: int) -> dict[str, Any]:
    """A user message carrying a tool's answer.

    Args:
        number: The call it answers.

    Returns:
        The message.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": f"t{number}",
                "content": f"out {number}\nmore",
            }
        ],
    }


TASK = {"role": "user", "content": "the task"}
ENVIRONMENT = {"role": "system", "content": "# Environment"}


def test_the_summariser_replaces_the_history_from_call_k_and_keeps_it_replaced() -> (
    None
):
    summarise = breakages.Summariser(at=3, task_text="Count them.")
    path = "/v1/messages?beta=true"
    calls = [
        body(TASK, ENVIRONMENT),
        body(TASK, ENVIRONMENT, tool_use(1), tool_result(1)),
        body(
            TASK, ENVIRONMENT, tool_use(1), tool_result(1), tool_use(2), tool_result(2)
        ),
        body(
            TASK,
            ENVIRONMENT,
            tool_use(1),
            tool_result(1),
            tool_use(2),
            tool_result(2),
            tool_use(3),
            tool_result(3),
        ),
    ]
    sent = [summarise(path, call) for call in calls]
    assert sent[0] is calls[0] and sent[1] is calls[1]
    third, fourth = sent[2]["messages"], sent[3]["messages"]
    assert [m["role"] for m in third] == ["user", "system", "assistant", "user"]
    assert third[1] == ENVIRONMENT and third[2:] == [tool_use(2), tool_result(2)]
    assert fourth[:2] == third[:2]  # the same summary, in the same place
    assert fourth[2:] == [tool_use(2), tool_result(2), tool_use(3), tool_result(3)]
    summary = third[0]["content"]
    assert summary.startswith(breakages.SUMMARY_HEAD)
    assert "The task: Count them." in summary and "- Bash: cmd 1 -> out 1" in summary
    other = {"messages": [TASK]}
    assert summarise("/v1/messages/count_tokens", other) is other


def test_the_summariser_changes_nothing_when_call_k_has_no_reply_before_it() -> None:
    summarise = breakages.Summariser(at=1, task_text="t")
    first, second = body(TASK), body(TASK, tool_use(1), tool_result(1))
    assert summarise("/v1/messages", first) is first
    assert summarise("/v1/messages", second) is second  # not summarised later
    assert summarise.cut is None


class Echo(BaseHTTPRequestHandler):
    """An upstream that answers every POST with what it was sent."""

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        """Echo the body."""
        got = self.rfile.read(int(self.headers["Content-Length"]))
        reply = json.dumps({"got": json.loads(got)}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet."""


def test_a_rewrite_that_fails_is_answered_502_naming_it() -> None:
    def broken(path: str, found: dict[str, Any]) -> dict[str, Any]:
        return found["no such field"]

    relay = claude_code.Relay(f"http://127.0.0.1:{free_port()}", rewrite=broken)
    try:
        request = urllib.request.Request(relay.start() + "/v1/messages", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=10)
    finally:
        relay.stop()
    assert refused.value.code == claude_code.BAD_GATEWAY
    assert b"the rewrite failed: KeyError" in refused.value.read()


def test_the_relay_forwards_the_rewritten_body_and_keeps_both() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    relay = claude_code.Relay(
        f"http://127.0.0.1:{upstream.server_address[1]}",
        rewrite=lambda path, found: {**found, "seen": path},
    )
    try:
        request = urllib.request.Request(
            relay.start() + "/v1/messages", data=json.dumps({"messages": []}).encode()
        )
        with urllib.request.urlopen(request, timeout=10) as answer:
            echoed = json.loads(answer.read())
    finally:
        relay.stop()
        upstream.shutdown()
        upstream.server_close()
    assert echoed == {"got": {"messages": [], "seen": "/v1/messages"}}
    kept = claude_code.RelayedRequest("/v1/messages", {"messages": []})
    assert relay.requests == [kept]
    assert relay.forwarded[0].body == {"messages": [], "seen": "/v1/messages"}


# ------------------------------------------------- the cross-check's parts


def log_events(
    replies: list[tuple[str, int]], compact_after: tuple[int, ...] = ()
) -> list[dict[str, Any]]:
    """Write Claude Code's stream-json events for a run of Bash replies.

    Args:
        replies: Each reply's command and the tokens Claude Code was told.
        compact_after: Replies after which the log reports a compaction.

    Returns:
        The events.
    """
    events = []
    for number, (command, tokens) in enumerate(replies, start=1):
        message_id = f"m{number}"
        start = {"type": "message_start", "message": {"id": message_id}}
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": tokens},
        }
        content = [{"type": "tool_use", "name": "Bash", "input": {"command": command}}]
        events += [
            {"type": "stream_event", "event": start},
            {"type": "assistant", "message": {"id": message_id, "content": content}},
            {"type": "stream_event", "event": delta},
        ]
        if number in compact_after:
            events.append({"type": "system", "subtype": "compact_boundary"})
    return events


def checks_of(
    document: dict[str, Any], log: list[dict[str, Any]]
) -> dict[str, crosscheck.Check]:
    """Run the cross-check's comparison and index its verdicts by name.

    Args:
        document: The proxy's record.
        log: Claude Code's events.

    Returns:
        The checks, by name.
    """
    return {c.name: c for c in crosscheck.compare(document, crosscheck.read_log(log))}


FOUR = [("step 1", 4), ("step 2", 5), ("step 3", 6), ("step 4", 7)]


def test_a_clean_run_agrees_on_every_check(capsys: pytest.CaptureFixture) -> None:
    checks = checks_of(chain_document([4, 5, 6, 7]), log_events(FOUR))
    assert [c.name for c in checks.values()] == list(crosscheck.CHECK_NAMES)
    assert all(c.ok for c in checks.values())
    assert checks["calls per row"].detail.endswith("agent rows: calls 1-4")


def with_prompts(document: dict[str, Any], prompts: list[list[dict]]) -> None:
    """Give each call of a document the messages it was sent.

    Args:
        document: A document from ``chain_document``.
        prompts: One message list per call.
    """
    for turn, messages in zip(document["turns"], prompts, strict=True):
        turn["request_messages"] = messages


def openai(role: str, content: str) -> dict[str, Any]:
    """One message as the proxy records it.

    Args:
        role: Its role.
        content: Its text.

    Returns:
        The message.
    """
    return {"role": role, "content": content}


def test_a_row_per_call_fails_calls_per_row_and_names_the_growing_note(
    capsys: pytest.CaptureFixture,
) -> None:
    document = chain_document([4, 5, 6, 7], rows=[[0], [1], [2], [3]])
    system = [openai("system", "sys" + " <n>" * i) for i in range(4)]
    with_prompts(document, [[system[i], openai("user", "task")] for i in range(4)])
    checks = checks_of(document, log_events(FOUR))
    assert checks["trainable coverage"].ok  # every token is still trained
    split = checks["calls per row"]
    assert not split.ok
    assert "call 1 | call 2 | call 3 | call 4" in split.detail
    assert split.why == (
        "calls 2, 3, 4 each start a row nothing announced; call 2: its system "
        'prompt is the call before\'s, plus " <n>" at its end',
    )


def test_a_replaced_history_is_named_and_a_logged_compaction_explains_it(
    capsys: pytest.CaptureFixture,
) -> None:
    document = chain_document([4, 5, 6, 7], rows=[[0, 1], [2, 3]])
    first = [openai("system", "sys"), openai("user", "task")]
    second = [*first, openai("assistant", "a"), openai("tool", "r")]
    third = [openai("system", "sys"), openai("user", "summary of it"), *second[2:]]
    fourth = [*third, openai("assistant", "b"), openai("tool", "s")]
    with_prompts(document, [first, second, third, fourth])
    split = checks_of(document, log_events(FOUR))["calls per row"]
    assert not split.ok
    assert split.why[0].startswith("call 3 starts a row nothing announced: its history")
    assert '"summary of it"' in split.why[0]
    logged = checks_of(document, log_events(FOUR, compact_after=(2,)))["calls per row"]
    assert logged.ok
    assert "2 conversations (calls 1-2 | calls 3-4, compacted 1 time(s))" in (
        logged.detail
    )


def test_an_untrained_call_is_named_with_the_proxys_finding(
    capsys: pytest.CaptureFixture,
) -> None:
    document = chain_document([4, 5, 6, 7], masked=frozenset({2}))
    finding = "[FATAL] no_logprobs: turn 2: 6 sampled tokens with no logprobs"
    document["validation"] = [finding]
    checks = checks_of(document, log_events(FOUR))
    coverage = checks["trainable coverage"]
    assert not coverage.ok and checks["calls per row"].ok
    assert "agent rows train [4, 5, 0, 7]" in coverage.detail
    told = f"call 3: Claude Code was told 6 tokens, the rows train 0 ({finding})"
    assert coverage.why == (told,)


def test_why_new_row_tells_a_recut_from_a_changed_message() -> None:
    before = {"request_messages": [openai("system", "s"), openai("user", "u")]}
    reply_and_answer = [openai("assistant", "a"), openai("tool", "t")]
    same = {"request_messages": [*before["request_messages"], *reply_and_answer]}
    assert crosscheck.why_new_row(before, same)[0] == "tokens"
    longer = {
        "request_messages": [
            openai("system", "s"),
            openai("user", "other"),
            openai("assistant", "a"),
            openai("tool", "t"),
        ]
    }
    assert crosscheck.why_new_row(before, longer)[0] == "message"


def test_conversations_are_cut_at_each_compaction() -> None:
    assert crosscheck.conversations(4, []) == [[1, 2, 3, 4]]
    assert crosscheck.conversations(4, [2]) == [[1, 2], [3, 4]]
    assert crosscheck.conversations(4, [0, 4, 9]) == [[1, 2, 3, 4]]  # no cut inside


def test_the_proxys_verdict_lists_each_warning_once() -> None:
    document = chain_document([4])
    document["validation"] = [
        "[WARN] a: 1",
        "[WARN] a: 2",
        "[INFO] certain: x",
        "[FATAL] b: y",
    ]
    line = crosscheck.proxy_verdict(document)
    assert line.endswith("findings [WARN] a, [FATAL] b")


def test_a_truncated_history_and_a_missing_call_are_named() -> None:
    before = {"request_messages": [openai("system", "s"), openai("user", "u")]}
    cut = {"request_messages": [openai("system", "s")]}
    assert crosscheck.why_new_row(before, cut)[0] == "history"
    lines = crosscheck.split_causes([], [[1], [3]], [[1, 2, 3]])
    assert lines[0] == "calls 2 are in no agent row"
    unsplit = crosscheck.split_causes([], [[1, 2, 3]], [[1], [2, 3]])
    assert unsplit == (
        "Claude Code's log compacts before call 2, yet no row starts there",
    )


def test_more_replies_in_the_log_than_calls_recorded_is_a_mismatch_not_a_crash(
    capsys: pytest.CaptureFixture,
) -> None:
    events = log_events(FOUR[:2])
    for number in (3, 4):  # two empty replies, cut at the cap, never recorded
        start = {"type": "message_start", "message": {"id": f"m{number}"}}
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 9},
        }
        events += [
            {"type": "stream_event", "event": start},
            {"type": "stream_event", "event": delta},
        ]
    checks = checks_of(chain_document([4, 5]), events)
    assert not checks["turns"].ok and checks["turns"].detail.startswith("2/4")


def test_calls_are_numbered_as_the_log_numbers_them_past_a_discarded_turn(
    capsys: pytest.CaptureFixture,
) -> None:
    document = chain_document([4, 3, 5], rows=[[0, 2]])
    document["turns"][1]["discarded"] = True  # a retry that led nowhere
    checks = checks_of(document, log_events([("step 1", 4), ("step 3", 5)]))
    assert all(check.ok for check in checks.values())
    assert checks["calls per row"].detail.endswith("agent rows: calls 1-2")


def test_log_lines_that_are_not_json_objects_are_counted(tmp_path: Path) -> None:
    log = tmp_path / "stream.jsonl"
    log.write_text('{"type": "system"}\n7\n{"type": "assist\n\n')
    events, skipped = crosscheck.read_events(log)
    assert events == [{"type": "system"}] and skipped == 2
    with pytest.raises(crosscheck.RunError, match="cannot read"):
        crosscheck.read_events(tmp_path / "missing.jsonl")


def test_crosscheck_runs_in_the_order_breakages_makes_them() -> None:
    assert crosscheck.RUN_ORDER == (captures.CAPTURE, *CASES)


def test_crosscheck_all_without_runs_says_what_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["crosscheck.py", "all"])
    with pytest.raises(SystemExit, match="run capture.py"):
        crosscheck.main()


# ------------------------------------------------- the scripts' small parts


def test_the_home_folder_is_written_as_a_tilde_only_where_its_name_ends() -> None:
    home = str(Path.home())
    text = f"cat {home}/a.txt; ls {home}2/b; cd {home}"
    assert lfm_files.tilde(text) == f"cat ~/a.txt; ls {home}2/b; cd ~"
    assert captures.clip_text(f"{home}/x") == "~/x"
    assert captures.cut_text(f"{home}/x", 99) == f"{home}/x"  # data stays as sent


def test_the_best_attempt_is_solved_then_one_root_then_trainable() -> None:
    def attempt(name: str, correct: bool, roots: int) -> capture.Attempt:
        grade = capture.Grade(correct, "30", "correct" if correct else "wrong")
        document = {"stats": {"n_roots": roots}, "trainable": True}
        return capture.Attempt(Path(name), grade, document, [])

    tried = [attempt("a", False, 1), attempt("b", True, 3), attempt("c", True, 2)]
    assert min(tried, key=capture.rank).folder.name == "b"  # ties keep their order
    assert min(tried[:1], key=capture.rank).folder.name == "a"


def test_a_new_capture_clears_the_runs_made_from_the_last_one(
    capsys: pytest.CaptureFixture,
) -> None:
    runs = captures.runs_folder()
    attempt = runs / "attempt-2"
    write_run(attempt, correct=True)
    for name in ("note", "summary"):
        (runs / name).mkdir(parents=True)
    grade = capture.Grade(True, "30", "correct")
    capture.keep(capture.Attempt(attempt, grade, {}, []), "solved, but in 2 roots")
    record = json.loads((runs / captures.CAPTURE / captures.RUN_FILE).read_text())
    assert (record["name"], record["attempt"]) == ("capture", "attempt-2")
    assert (record["clean"], record["flaw"]) == (False, "solved, but in 2 roots")
    assert not (runs / "note").exists() and not (runs / "summary").exists()
    assert "cleared the last capture's runs: note, summary" in capsys.readouterr().out


def test_calls_and_counts_are_written_compactly() -> None:
    assert breakages.call_span([3]) == "call 3"
    assert breakages.call_span([1, 2, 3, 4]) == "calls 1-4"
    assert breakages.call_span([1, 3]) == "calls 1, 3"
    assert (breakages.plural(1, "row"), breakages.plural(4, "row")) == (
        "1 row",
        "4 rows",
    )


def test_the_billing_note_counts_the_line_and_what_reached_the_engine() -> None:
    line = "x-anthropic-billing-header: cc_version=1;"
    sent = [
        claude_code.RelayedRequest(
            "/v1/messages",
            {"system": [{"type": "text", "text": line}], "messages": []},
        )
        for _ in range(2)
    ]
    document = {"turns": [{"request_messages": [openai("system", "rest")]}]}
    note = capture.billing_note(sent, document)
    assert note.endswith(
        "on 2 of 2 calls (the same line every time); the engine received it 0 times"
    )


def test_a_capture_must_be_solved_and_already_one_trainable_row() -> None:
    def attempt(correct: bool, roots: int, trainable: bool) -> capture.Attempt:
        grade = capture.Grade(correct, "30", "correct" if correct else "wrong")
        document = {"stats": {"n_roots": roots}, "trainable": trainable}
        return capture.Attempt(Path("a"), grade, document, [])

    assert capture.why_not_kept(attempt(True, 1, True)) == ""
    assert capture.why_not_kept(attempt(False, 1, True)).endswith("(wrong)")
    assert capture.why_not_kept(attempt(True, 2, True)) == "solved, but in 2 roots"
    assert capture.why_not_kept(attempt(True, 1, False)) == "solved, but not trainable"


def test_the_compaction_runs_long_reply_passes_the_mark_whole() -> None:
    ids = scripted_replies.encode(compaction.REPLIES[1].text)
    assert len(ids) == compaction.LONG_REPLY_TOKENS
    assert ids[-1] == scripted_engine.EOS  # under the cap: not cut
    mark = compaction.WINDOW * compaction.PERCENT // compaction.PERCENT_OF
    assert mark < compaction.LONG_REPLY_TOKENS < 1_024


def test_start_sh_offers_the_replay_engine() -> None:
    usage = subprocess.run(
        ["bash", str(START_SH), "nonsense"], capture_output=True, text=True, check=False
    )
    assert usage.returncode == 2
    assert "start|scripted|replay|proxy|stop|status|install" in usage.stderr


def write_run(folder: Path, correct: bool, tokenizer: str = "scripted") -> None:
    """Write a capture folder holding one call.

    Args:
        folder: The folder.
        correct: Whether its task was solved.
        tokenizer: The tokenizer its ids come from.
    """
    folder.mkdir(parents=True)
    meta = {
        "task_id": "clinic-easy-001",
        "tokenizer": tokenizer,
        "grade": {"correct": correct},
    }
    (folder / captures.RUN_FILE).write_text(json.dumps(meta))
    (folder / captures.ROLLOUT_FILE).write_text(json.dumps(chain_document([4, 5])))


def test_an_unsolved_capture_is_replayed_under_its_grade(tmp_path: Path) -> None:
    write_run(tmp_path / "run", correct=False)
    loaded = breakages.load_capture(str(tmp_path / "run"))
    assert len(loaded.replies) == 2
    meta = {**loaded.meta, "flaw": "the task was not solved (wrong)"}
    assert breakages.capture_grade(meta) == (
        "grade wrong (not clean: the task was not solved (wrong))"
    )
    assert breakages.capture_grade({"grade": {"correct": True}}) == "grade correct"
    with pytest.raises(SystemExit, match="capture.py"):
        breakages.load_capture(str(tmp_path / "nothing"))


class Health(BaseHTTPRequestHandler):
    """A stand-in proxy whose /health names an engine."""

    upstream = ""

    def do_GET(self) -> None:  # noqa: N802
        """Answer /health."""
        reply = json.dumps({"upstream": self.upstream}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Stay quiet."""


def test_breakages_refuses_an_engine_that_cuts_other_tokens() -> None:
    with live_engine() as (port, _):
        Health.upstream = f"http://127.0.0.1:{port}"
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            client = capture.CaptureClient(
                f"http://127.0.0.1:{proxy.server_address[1]}"
            )
            assert breakages.replay_engine(client, "scripted") is not None
            with pytest.raises(SystemExit, match=r"\./start\.sh replay"):
                breakages.replay_engine(client, "lfm2.5")
        finally:
            proxy.shutdown()
            proxy.server_close()


# ------------------------------------------------- end to end, Claude Code


def run_script(module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run a script's main as from the command line.

    Args:
        module: The script's module.
        argv: Its arguments, the script's name first.
        monkeypatch: pytest's patcher.

    Returns:
        Its exit status (0 when it returned).
    """
    monkeypatch.setattr(sys, "argv", argv)
    try:
        module.main()
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else 1
    return 0


def capture_break_and_crosscheck(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> dict[str, str]:
    """Record a capture, replay it four ways, cross-check each run.

    Args:
        stack: The proxy and its scripted engine.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.

    Returns:
        Each script's output, by step name.
    """
    monkeypatch.setenv(claude_code.CLAUDE_BIN_ENV, str(CLAUDE_BIN))
    url = ["--proxy-url", stack.proxy_url, "--color", "never"]
    out = {}
    assert run_script(capture, ["capture.py", *url], monkeypatch) == 0
    out["capture"] = capsys.readouterr().out
    assert run_script(breakages, ["breakages.py", *url], monkeypatch) == 0
    out["breakages"] = capsys.readouterr().out
    for name in crosscheck.RUN_ORDER:
        status = run_script(crosscheck, ["crosscheck.py", name], monkeypatch)
        out[f"check {name}"] = capsys.readouterr().out
        assert status == (0 if name in (captures.CAPTURE, "replay") else 1), name
    return out


def assert_each_break_is_caught(out: dict[str, str]) -> None:
    """Check the four replays and what the cross-check said about each.

    Args:
        out: The scripts' output, by step name.
    """
    assert "grade correct" in out["capture"] and "roots 1" in out["capture"]
    assert "the engine received it 0 times" in out["capture"]
    table = out["breakages"].split("━━ one capture, replayed 4 ways")[1]
    rows = {line.split()[0]: line.split() for line in table.splitlines()[2:6]}
    # case: grade, calls, roots, rows, row-tokens, mask-1, trainable
    assert {name: row[1] for name, row in rows.items()} == dict.fromkeys(
        CASES, "correct"
    )
    assert {name: row[3:5] for name, row in rows.items()} == {
        "replay": ["1", "1"],
        "note": ["4", "4"],
        "summary": ["2", "2"],
        "no-logprobs": ["1", "1"],
    }
    assert all(row[7] == "true" for row in rows.values())
    assert rows["no-logprobs"][6] != rows["replay"][6]
    assert "every row too, all its tokens" in out["breakages"]
    assert out["breakages"].count("token for token") == len(CASES)
    assert "never asked for" not in out["breakages"]
    failed = {
        name: [
            line.split()[0] + " " + line.split()[1]
            for line in out[f"check {name}"].splitlines()
            if "MISMATCH ·" in line
        ]
        for name in CASES
    }
    assert failed == {
        "replay": [],
        "note": ["calls per"],
        "summary": ["calls per"],
        "no-logprobs": ["trainable coverage"],
    }
    assert "its system prompt is the call before's, plus" in out["check note"]
    assert "its history was replaced" in out["check summary"]
    assert "[FATAL] no_logprobs: turn 2" in out["check no-logprobs"]
    for name in CASES:
        assert "the proxy's verdict: trainable true" in out[f"check {name}"]


@needs_claude
def test_one_capture_replayed_four_ways_is_caught_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with proxy_stack(tmp_path / "stack", None) as stack:
        out = capture_break_and_crosscheck(stack, monkeypatch, capsys)
        argv = ["compaction.py", "--proxy-url", stack.proxy_url]
        assert run_script(compaction, argv, monkeypatch) == 0
        asked = capsys.readouterr().out
    assert_each_break_is_caught(out)
    # Claude Code 2.1.290, asked to compact past about 1,000 tokens, does not.
    assert "claude    exit code 0" in asked
    assert "log       compact_boundary events: 0" in asked
    assert "result    no compaction: 1 root, 1 row" in asked
    runs = captures.runs_folder()
    assert (runs / "summary" / captures.FORWARDED_FILE).is_file()
    assert not (runs / "note" / captures.FORWARDED_FILE).exists()


@needs_claude
@needs_tokenizer
def test_a_capture_in_lfm25s_tokens_replays_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    cutter = scripted_engine.LfmCutter(LFM_TOKENIZER_FILE)
    with proxy_stack(tmp_path / "stack", cutter) as stack:
        out = capture_break_and_crosscheck(stack, monkeypatch, capsys)
    assert "(lfm2.5 tokens)" in out["capture"]
    assert_each_break_is_caught(out)
