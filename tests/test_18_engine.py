"""Chapter 18 without a model: the engine's requests, receipts, caches and queue.

No test here loads weights or needs MLX. The engine is given a scripted model
(``FakeModel``): replies written in advance with LFM2.5's special tokens, a
tokenizer that gives each marker one id and every other character its own
(plus one merged pair, ``ch``, so a text can be cut two ways, as chapter 9
showed), a chat template that lays a chat out the way LFM2.5's does
(refusing tool-call arguments written as a JSON string, as LFM2.5's does),
and a prompt cache that is the list of tokens it has read. The scripts run
against a real HTTP server on a free port. The tests marked for the
tokenizer load LFM2.5-2.6B's real tokenizer and chat template when they are
on this machine (no weights), and are skipped otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import socket
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import uvicorn
from chapters import use_chapter

CHAPTER = use_chapter("18-engine")

import client  # noqa: E402
import drift  # noqa: E402
import engine  # noqa: E402
import lfm_files  # noqa: E402
import lfm_text  # noqa: E402
import tasks  # noqa: E402
import tiny_harness_http  # noqa: E402
import toolcalls  # noqa: E402
from policy import RenderError, Sampled, Sampling  # noqa: E402

MARKERS = {
    "<|startoftext|>": 1,
    "<|im_start|>": 2,
    "<|im_end|>": 3,
    "<think>": 4,
    "</think>": 5,
    "<|tool_call_start|>": 6,
    "<|tool_call_end|>": 7,
}
EOS = MARKERS["<|im_end|>"]
MERGED = {"ch": 10}  # one pair the tokenizer writes as one token
PLAIN = 100  # every other character is PLAIN + its code point
START, END = toolcalls.TOOL_CALL_START, toolcalls.TOOL_CALL_END
EOT = toolcalls.END_OF_TURN
# A small vocabulary of scores for the alternatives (tokens 200, 201, ...).
FAKE_SCORES = [4.0, 2.0, 1.5, 1.0, 0.0, -1.0, -2.0]
FIRST_ALTERNATIVE = 200
LOGPROB_STEP = -0.125
EASY = tasks.find_task("clinic-easy-001")
TOOLS = [tiny_harness_http.bash_schema()]
MURCIA = "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where a real run of the course keeps Liquid's export.
REAL_MODEL = lfm_files.model_folder(lfm_files.FOUR_BIT)
HAVE_TOKENIZER = (REAL_MODEL / "tokenizer.json").is_file() and (
    REAL_MODEL / "chat_template.jinja"
).is_file()
needs_tokenizer = pytest.mark.skipif(
    not HAVE_TOKENIZER, reason="LFM2.5-2.6B's tokenizer is not here"
)


def encode(text: str) -> list[int]:
    """Cut text the fake way: markers, then the merged pair, then characters.

    Args:
        text: The text.

    Returns:
        The ids.
    """
    ids, index = [], 0
    pieces = {**MARKERS, **MERGED}
    while index < len(text):
        for piece, token in pieces.items():
            if text.startswith(piece, index):
                ids.append(token)
                index += len(piece)
                break
        else:
            ids.append(PLAIN + ord(text[index]))
            index += 1
    return ids


def decode(ids: list[int]) -> str:
    """Write fake ids back as text, markers kept.

    Args:
        ids: The ids.

    Returns:
        The text.
    """
    names = {token: piece for piece, token in {**MARKERS, **MERGED}.items()}
    return "".join(names.get(i) or chr(i - PLAIN) for i in ids)


def argument(value: Any) -> str:
    """Write one argument as LFM2.5's template does (single quotes for text).

    Args:
        value: The value.

    Returns:
        Its source form.
    """
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return json.dumps(value)


def fake_template(messages: list[dict[str, Any]], tools: list | None) -> str:
    """Lay a chat out as LFM2.5's template does, for what these tests need.

    Args:
        messages: The chat.
        tools: The tools offered.

    Returns:
        The prompt, ending with ``<|im_start|>assistant`` and ``<think>``.

    Raises:
        RenderError: For tool-call arguments written as a JSON string.
    """
    rest = list(messages)
    system = rest.pop(0)["content"] if rest and rest[0]["role"] == "system" else ""
    if tools:
        system += ("\n" if system else "") + "List of tools: " + json.dumps(tools)
    text = "<|startoftext|>"
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    for message in rest:
        text += f"<|im_start|>{message['role']}\n"
        thinking = message.get("reasoning_content")
        if message["role"] == "assistant" and thinking:
            text += f"<think>{thinking}</think>"
        text += message.get("content") or ""
        calls = message.get("tool_calls") or []
        written = []
        for call in calls:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                raise RenderError("Tool call arguments must be a mapping")
            pairs = ", ".join(f"{k}={argument(v)}" for k, v in arguments.items())
            written.append(f"{call['function']['name']}({pairs})")
        if written:
            text += f"{START}[{', '.join(written)}]{END}"
        text += "<|im_end|>\n"
    return text + "<|im_start|>assistant\n<think>"


@dataclass
class FakeCache:
    """A prompt cache that is simply the tokens it has read."""

    tokens: list[int] = field(default_factory=list)


@dataclass
class Seen:
    """One call to ``FakeModel.sample``, as the model saw it."""

    new_ids: list[int]
    held: list[int]
    sampling: Sampling
    thread: int
    started: float
    ended: float = 0.0


class FakeModel:
    """A scripted model behind the engine's seam."""

    eos_ids = frozenset({EOS})

    def __init__(
        self, replies: list[str | list[int]], delay: float = 0.0, temper: bool = True
    ) -> None:
        """Keep the replies.

        Args:
            replies: Each reply as text (encoded the fake way) or as ids.
            delay: Seconds each reply takes.
            temper: Whether the alternatives follow the temperature (as
                processed log-probabilities do) or ignore it (raw ones).
        """
        self.replies = list(replies)
        self.delay = delay
        self.temper = temper
        self.seen: list[Seen] = []
        self.render_threads: set[int] = set()

    def render(self, messages: list[dict[str, Any]], tools: list | None) -> str:
        """Render with the fake template, noting the thread."""
        self.render_threads.add(threading.get_ident())
        return fake_template(messages, tools)

    def encode(self, text: str) -> list[int]:
        """Encode the fake way."""
        return encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode the fake way."""
        return decode(token_ids)

    def new_cache(self) -> FakeCache:
        """Make an empty cache."""
        return FakeCache()

    def cache_size(self, cache: FakeCache) -> int:
        """Count the tokens a cache holds."""
        return len(cache.tokens)

    def alternatives(self, temperature: float, k: int) -> list[tuple[int, float]]:
        """The k likeliest fake tokens, tempered or not."""
        scale = temperature if (self.temper and temperature > 0) else 1.0
        scaled = [score / scale for score in FAKE_SCORES]
        total = math.log(sum(math.exp(s) for s in scaled))
        pairs = [(FIRST_ALTERNATIVE + i, s - total) for i, s in enumerate(scaled)]
        return pairs[:k]

    def sample(
        self, new_ids: list[int], cache: FakeCache, sampling: Sampling
    ) -> Sampled:
        """Hand back the next scripted reply, as ids with log-probabilities."""
        seen = Seen(
            list(new_ids),
            list(cache.tokens),
            sampling,
            threading.get_ident(),
            time.perf_counter(),
        )
        self.seen.append(seen)
        time.sleep(self.delay)
        reply = self.replies.pop(0)
        whole = encode(reply) if isinstance(reply, str) else list(reply)
        ids = whole[: sampling.max_tokens]
        cut = len(whole) > len(ids) or not ids or ids[-1] not in self.eos_ids
        logprobs = [LOGPROB_STEP * (i % 4 + 1) for i in range(len(ids))]
        k = sampling.top_logprobs
        tops = [self.alternatives(sampling.temperature, k) for _ in ids] if k else []
        cache.tokens += list(new_ids) + ids
        seen.ended = time.perf_counter()
        finish = "length" if cut else "stop"
        return Sampled(ids, logprobs, tops, finish, 0.001)


def call_reply(command: str, think: str = "Look first.", quote: str = "'") -> str:
    """Write a reply that calls bash once, as LFM2.5 writes it.

    Args:
        command: The command.
        think: The reasoning.
        quote: The quote the model puts around the command.

    Returns:
        The reply text, end of turn included.
    """
    if quote == "'":
        written = argument(command)
    else:
        written = '"' + command.replace('"', '\\"') + '"'
    return f"{think}</think>{START}[bash(command={written})]{END}{EOT}"


def answer_reply(text: str = "The answer is in answer.txt.") -> str:
    """Write a reply with no tool call.

    Args:
        text: The visible text.

    Returns:
        The reply text.
    """
    return f"Done.</think>{text}{EOT}"


def body(messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """Write a chat request with the bash tool.

    Args:
        messages: The chat.
        **fields: Anything else.

    Returns:
        The request body.
    """
    return {"model": "lfm2.5-2.6b", "messages": messages, "tools": TOOLS} | fields


def answer(core: engine.Engine, request_body: dict[str, Any]) -> dict[str, Any]:
    """Answer one request in this thread, as the server's worker would.

    Args:
        core: The engine.
        request_body: The request.

    Returns:
        The response body.
    """
    completion = core.complete(engine.read_request(request_body))
    return engine.to_openai(core.model, completion, 0.0)


def harness_turn(response: dict[str, Any], output: str = "ok") -> list[dict]:
    """Write what a harness appends: the reply's JSON copy, then each result.

    Args:
        response: The engine's answer.
        output: Every tool result's text.

    Returns:
        The messages.
    """
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    kept = {"role": "assistant", "content": message.get("content")}
    if calls:
        kept["tool_calls"] = calls
    results = [
        {"role": "tool", "tool_call_id": call["id"], "content": output}
        for call in calls
    ]
    return [kept, *results]


@pytest.fixture
def first_messages() -> list[dict[str, Any]]:
    """The harness's first two messages for clinic-easy-001."""
    return tiny_harness_http.start_messages(EASY)


@contextlib.contextmanager
def live_engine(model: FakeModel) -> Iterator[str]:
    """Serve an engine over real HTTP on a free port, in a thread.

    Args:
        model: The scripted model.

    Yields:
        The base URL, ending in /v1.
    """
    sock = engine.claim_port(0)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(engine.create_app(engine.Engine(model)), log_level="warning")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(10)


@contextlib.contextmanager
def _asgi(app: Any) -> Iterator[Any]:
    """A synchronous client for an ASGI app, through a real event loop."""
    loop = asyncio.new_event_loop()
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://engine")

    class Sync:
        def post(self, path: str, **kwargs: Any) -> httpx.Response:
            return loop.run_until_complete(http.post(path, **kwargs))

        def get(self, path: str) -> httpx.Response:
            return loop.run_until_complete(http.get(path))

    try:
        yield Sync()
    finally:
        loop.run_until_complete(http.aclose())
        loop.close()


# ------------------------------------------------------------------ the shape


def test_a_tool_call_comes_back_as_openai_json_with_its_receipts(
    first_messages: list[dict[str, Any]],
) -> None:
    model = FakeModel([call_reply("sqlite3 data.db .tables")])
    response = answer(engine.Engine(model), body(first_messages))
    choice = response["choices"][0]
    message = choice["message"]
    assert message["role"] == "assistant" and message["content"] is None
    (call,) = message["tool_calls"]
    assert call["type"] == "function" and call["function"]["name"] == "bash"
    assert call["id"].startswith("call_") and len(call["id"]) == len("call_") + 24
    arguments = json.loads(call["function"]["arguments"])  # a JSON string, as OpenAI
    assert arguments == {"command": "sqlite3 data.db .tables"}
    assert choice["finish_reason"] == "tool_calls"
    prompt = response["prompt_token_ids"]
    assert prompt == encode(fake_template(first_messages, TOOLS))
    assert choice["token_ids"] == encode(call_reply("sqlite3 data.db .tables"))
    entries = choice["logprobs"]["content"]
    expected = [f"token_id:{token}" for token in choice["token_ids"]]
    assert [entry["token"] for entry in entries] == expected
    assert entries[0]["bytes"] == list(decode([choice["token_ids"][0]]).encode())
    assert all(e["top_logprobs"] == [] for e in entries)
    usage = response["usage"]
    assert usage["prompt_tokens"] == len(prompt)
    assert usage["completion_tokens"] == len(choice["token_ids"])
    assert usage["total_tokens"] == len(prompt) + len(choice["token_ids"])
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    assert set(response["timings"]) == {"waited_s", "first_token_s", "total_s"}


def test_a_plain_reply_is_content_and_stop(first_messages: list[dict]) -> None:
    model = FakeModel([answer_reply("30")])
    response = answer(engine.Engine(model), body(first_messages))
    message = response["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "30"}
    assert response["choices"][0]["finish_reason"] == "stop"


def test_a_reply_cut_by_the_cap_finishes_with_length(first_messages: list) -> None:
    model = FakeModel([call_reply("sqlite3 data.db .tables")])
    response = answer(engine.Engine(model), body(first_messages, max_tokens=5))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == 5
    assert "tool_calls" not in choice["message"]  # the call never closed


def test_max_tokens_is_lowered_to_the_cap_and_temperature_defaults_to_one(
    first_messages: list,
) -> None:
    model = FakeModel([answer_reply(), answer_reply()])
    core = engine.Engine(model)
    answer(core, body(first_messages, max_tokens=32_000))
    answer(core, body(first_messages, max_completion_tokens=7, temperature=0))
    first, second = model.seen
    assert first.sampling.max_tokens == 1_024 and first.sampling.temperature == 1.0
    assert second.sampling.max_tokens == 7 and second.sampling.temperature == 0.0


def test_top_logprobs_up_to_20_come_back_in_token_id_form(first_messages: list) -> None:
    model = FakeModel([answer_reply("30"), answer_reply("30")])
    core = engine.Engine(model)
    response = answer(core, body(first_messages, top_logprobs=5))
    entry = response["choices"][0]["logprobs"]["content"][0]
    assert [alt["token"] for alt in entry["top_logprobs"]] == [
        f"token_id:{FIRST_ALTERNATIVE + i}" for i in range(5)
    ]
    values = [alt["logprob"] for alt in entry["top_logprobs"]]
    assert values == sorted(values, reverse=True)
    answer(core, body(first_messages, top_logprobs=20))
    assert model.seen[-1].sampling.top_logprobs == 20


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({"top_logprobs": 21}, "top_logprobs must be between 0 and 20"),
        ({"top_logprobs": -1}, "top_logprobs must be between 0 and 20"),
        ({"stream": True}, "stream is not supported"),
        ({"n": 2}, "n must be 1"),
        ({"temperature": -0.5}, "temperature must be 0 or more"),
        ({"temperature": "hot"}, "temperature must be a number"),
        ({"max_tokens": "many"}, "max_tokens must be a number"),
        ({"tools": "bash"}, "tools must be a list"),
        ({"messages": []}, "messages must be a non-empty list"),
        ({"messages": ["hi"]}, "every message must be an object"),
    ],
)
def test_requests_the_engine_cannot_answer_are_refused(
    first_messages: list, fields: dict[str, Any], says: str
) -> None:
    with pytest.raises(engine.BadRequest, match=says):
        engine.read_request(body(first_messages) | fields)


# ------------------------------------------------------------------ the restore


def test_call_two_continues_call_one_through_the_harness_copy(
    first_messages: list,
) -> None:
    model = FakeModel([call_reply("sqlite3 data.db .tables"), answer_reply()])
    core = engine.Engine(model)
    first = answer(core, body(first_messages))
    messages = first_messages + harness_turn(first, "appointments doctors patients")
    second = answer(core, body(messages))
    chain = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
    assert second["prompt_token_ids"][: len(chain)] == chain
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == len(chain)
    # The model read only what was added: the tool result and the new opening.
    assert model.seen[1].held == chain
    assert model.seen[1].new_ids == second["prompt_token_ids"][len(chain) :]
    assert core.stats.restored == 1 and core.stats.not_restored == 0


def test_a_plain_reply_is_restored_by_its_visible_text(first_messages: list) -> None:
    model = FakeModel([answer_reply("Which year?"), answer_reply("30")])
    core = engine.Engine(model)
    first = answer(core, body(first_messages))
    follow = [*harness_turn(first), {"role": "user", "content": "2025."}]
    second = answer(core, body(first_messages + follow))
    chain = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
    assert second["prompt_token_ids"][: len(chain)] == chain
    assert core.stats.restored == 1


def test_an_unknown_copy_is_rendered_from_its_json_and_counted(
    first_messages: list,
) -> None:
    model = FakeModel([answer_reply()])
    core = engine.Engine(model)
    stranger = {
        "id": "call_not_ours",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    copy = {"role": "assistant", "content": None, "tool_calls": [stranger]}
    result = {"role": "tool", "tool_call_id": "call_not_ours", "content": "data.db"}
    response = answer(core, body([*first_messages, copy, result]))
    text = decode(response["prompt_token_ids"])
    # The arguments were read back into an object, then written the template's way.
    assert f"<|im_start|>assistant\n{START}[bash(command='ls')]{END}" in text
    assert core.stats.not_restored == 1 and core.stats.restored == 0


def test_arguments_that_are_not_a_json_object_are_a_bad_request(
    first_messages: list,
) -> None:
    core = engine.Engine(FakeModel([]))
    for arguments in ("{not json", "[1, 2]"):
        call = {"id": "x", "function": {"name": "bash", "arguments": arguments}}
        copy = {"role": "assistant", "content": None, "tool_calls": [call]}
        with pytest.raises(engine.BadRequest, match="arguments"):
            core.complete(engine.read_request(body([*first_messages, copy])))


def test_the_template_refusing_a_conversation_is_a_bad_request() -> None:
    class Refusing(FakeModel):
        def render(self, messages: list, tools: list | None) -> str:
            raise RenderError("no")

    core = engine.Engine(Refusing([]))
    with pytest.raises(engine.BadRequest, match="the chat template refused it: no"):
        core.render([{"role": "user", "content": "hi"}], None)


def test_the_engine_forgets_the_oldest_reply(
    first_messages: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "REMEMBERED_REPLIES", 2)
    replies = [call_reply(f"ls {n}") for n in range(3)]
    core = engine.Engine(FakeModel(replies))
    ids = []
    for _ in replies:
        response = answer(core, body(first_messages))
        ids.append(response["choices"][0]["message"]["tool_calls"][0]["id"])
    assert list(core.replies) == [f"id:{ids[1]}", f"id:{ids[2]}"]


def test_a_reply_whose_text_cuts_into_other_tokens_is_counted(
    first_messages: list, capsys: pytest.CaptureFixture[str]
) -> None:
    # The model wrote "c" then "h"; the same text encodes as the one token "ch".
    drawn = encode("I will check.</think>") + [PLAIN + ord("c"), PLAIN + ord("h")]
    model = FakeModel([[*drawn, EOS], answer_reply()])
    core = engine.Engine(model)
    first = answer(core, body(first_messages))
    assert core.stats.recut == 1
    assert "cuts into other tokens" in capsys.readouterr().out
    follow = [*harness_turn(first), {"role": "user", "content": "Go on."}]
    second = answer(core, body(first_messages + follow))
    chain = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
    assert second["prompt_token_ids"][: len(chain)] != chain  # restored, yet drifted
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


# ------------------------------------------------------------------ the caches


def test_eight_caches_are_kept_and_the_oldest_is_dropped() -> None:
    core = engine.Engine(FakeModel([answer_reply()] * 9))
    for n in range(9):
        answer(core, body([{"role": "user", "content": f"question {n}"}]))
    assert len(core.caches) == 8
    held = [decode(tokens) for tokens, _ in core.caches]
    assert "question 8" in held[0] and not any("question 0" in h for h in held)


def test_the_cache_with_the_longest_exact_prefix_is_taken() -> None:
    core = engine.Engine(FakeModel([]))
    prompt = list(range(PLAIN, PLAIN + 20))
    for size in (5, 12, 8):
        core.caches.append((prompt[:size], FakeCache(prompt[:size])))
    core.caches.append((prompt[:3] + [1, 2], FakeCache()))  # not a prefix
    cache, cached = core.take_cache(prompt)
    assert cached == 12 and cache.tokens == prompt[:12]
    assert [len(tokens) for tokens, _ in core.caches] == [5, 8, 5]  # taken out


def test_a_cache_must_hold_a_strict_prefix() -> None:
    core = engine.Engine(FakeModel([]))
    prompt = list(range(PLAIN, PLAIN + 10))
    core.caches.append((list(prompt), FakeCache(list(prompt))))
    cache, cached = core.take_cache(prompt)  # nothing left to read: not usable
    assert cached == 0 and cache.tokens == []


# ------------------------------------------------------------------ the limit


def test_a_prompt_over_16384_tokens_is_context_length_exceeded() -> None:
    overhead = len(encode(fake_template([{"role": "user", "content": ""}], None)))
    fits = "a" * (engine.MAX_PROMPT_TOKENS - overhead)
    core = engine.Engine(FakeModel([answer_reply()]))
    response = answer(core, {"messages": [{"role": "user", "content": fits}]})
    assert response["usage"]["prompt_tokens"] == engine.MAX_PROMPT_TOKENS
    with pytest.raises(engine.BadRequest) as refused:
        core.complete(
            engine.read_request({"messages": [{"role": "user", "content": fits + "a"}]})
        )
    assert refused.value.code == "context_length_exceeded"
    assert str(refused.value) == "prompt is 16,385 tokens; the limit is 16,384"


def test_the_http_refusal_is_shaped_as_openai_does() -> None:
    app = engine.create_app(engine.Engine(FakeModel([])))
    words = {"messages": [{"role": "user", "content": "x" * 20_000}]}
    with _asgi(app) as http:
        response = http.post("/v1/chat/completions", json=words)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "context_length_exceeded"


# ------------------------------------------------------------------ the queue


def test_four_requests_at_once_finish_one_after_another_on_one_thread(
    first_messages: list,
) -> None:
    model = FakeModel([answer_reply(str(n)) for n in range(4)], delay=0.15)
    app = engine.create_app(engine.Engine(model))

    async def four() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://e") as http:
            sends = [
                http.post("/v1/chat/completions", json=body(first_messages))
                for _ in range(4)
            ]
            return await asyncio.gather(*sends)

    responses = asyncio.run(four())
    assert all(response.status_code == 200 for response in responses)
    assert len({seen.thread for seen in model.seen}) == 1  # one worker thread
    assert model.render_threads == {model.seen[0].thread}  # rendering too
    spans = sorted((seen.started, seen.ended) for seen in model.seen)
    assert all(
        after[0] >= before[1] for before, after in zip(spans, spans[1:], strict=False)
    )
    waits = sorted(r.json()["timings"]["waited_s"] for r in responses)
    assert waits[-1] >= 3 * 0.15 - 0.05  # the last waited for the other three


def test_health_models_and_stats(first_messages: list) -> None:
    app = engine.create_app(engine.Engine(FakeModel([answer_reply()])))
    with _asgi(app) as http:
        assert http.get("/health").json() == {"ok": True, "model": "lfm2.5-2.6b"}
        listed = http.get("/v1/models").json()["data"]
        assert listed == [{"id": "lfm2.5-2.6b", "object": "model"}]
        http.post("/v1/chat/completions", json=body(first_messages))
        stats = http.get("/admin/stats").json()
    assert stats["requests"] == 1 and stats["completion_tokens"] > 0


# ------------------------------------------------------------------ the port


def test_a_busy_port_fails_at_once_and_names_the_port_flag() -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        with pytest.raises(SystemExit) as busy:
            engine.claim_port(port)
    message = str(busy.value)
    assert f"127.0.0.1:{port} is in use" in message
    assert "--port" in message and "uv run engine.py --port 8119" in message


def test_a_free_port_is_claimed_and_listening() -> None:
    sock = engine.claim_port(0)
    try:
        port = sock.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass  # the connection waits in the backlog until a server accepts it
    finally:
        sock.close()


def test_the_engine_refuses_a_busy_port_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        monkeypatch.setattr(sys, "argv", ["engine.py", "--port", str(port)])
        monkeypatch.setitem(sys.modules, "lfm", None)  # importing it would fail
        with pytest.raises(SystemExit, match="--port"):
            engine.main()


# ------------------------------------------------------------------ drift


def test_first_difference() -> None:
    assert drift.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert drift.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert drift.first_difference([1, 2], [1, 2, 3]) == 2
    assert drift.first_difference([], [4]) == 0


def test_both_causes_of_drift_are_found_where_they_happen(
    first_messages: list,
) -> None:
    # The model put double quotes around the command; the template writes single.
    written = call_reply("ls", think="I list the folder.", quote='"')
    model = FakeModel([written])
    core = engine.Engine(model)
    completion = core.complete(engine.read_request(body(first_messages)))
    chain = completion.prompt_ids + completion.sampled.token_ids
    start = len(completion.prompt_ids)
    copy = drift.copy_of(completion.reply.content, completion.tool_calls)
    result = {"role": "tool", "tool_call_id": copy["tool_calls"][0]["id"]}
    results = [result | {"content": "data.db"}]
    with pytest.raises(RenderError):  # the copy as it comes: arguments are text
        model.render([*first_messages, copy, *results], TOOLS)
    plain = engine.harness_copy(copy)
    dropped = drift.rerender(model, [*first_messages, plain, *results], TOOLS)
    found = drift.compare(chain, dropped, start)
    assert found.position == start - 1  # the <think> the prompt ended with is gone
    lines = drift.describe(model, found, chain, dropped)
    assert lines[0].startswith(f"first difference: token {start:,} of the previous")
    assert "| '<think>'" in lines[1] and "| '<|tool_call_start|>'" in lines[2]
    given = plain | {"reasoning_content": completion.reply.reasoning}
    rewritten = drift.rerender(model, [*first_messages, given, *results], TOOLS)
    found = drift.compare(chain, rewritten, start)
    reply = chain[start:]
    assert found.position is not None
    offset = found.position - start
    assert drift.where(model, reply, offset) == "in the tool-call span"
    assert decode([chain[found.position]]) == '"'
    assert decode([rewritten[found.position]]) == "'"
    restored = core.render([*first_messages, copy, *results], TOOLS)
    assert drift.compare(chain, restored, start).position is None


def test_where_names_the_part_of_the_reply() -> None:
    model = FakeModel([])
    reply = encode(f"think</think>hi{START}[bash()]{END}")
    assert drift.where(model, reply, 2) == "in the reasoning"
    assert drift.where(model, reply, reply.index(6)) == "in the visible text"
    assert drift.where(model, reply, reply.index(6) + 2) == "in the tool-call span"


def test_drift_runs_end_to_end_on_a_scripted_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    written = call_reply("sqlite3 data.db .tables", think="Tables first.", quote='"')
    scripted = FakeModel([written])
    scripted.seconds = 0.0
    fake_lfm = ModuleType("lfm")
    fake_lfm.LfmModel = lambda folder, seed: scripted
    monkeypatch.setitem(sys.modules, "lfm", fake_lfm)
    monkeypatch.setattr(drift, "pick_model", lambda fast: (tmp_path, "scripted"))
    drift.run_live(fast=False)
    out = capsys.readouterr().out
    assert "1. the reply the model sampled" in out
    assert "as it is, LFM2.5's template refuses it" in out
    assert "3. cause 1, the dropped <think>" in out
    assert "the reply is not even reached" in out
    assert "4. cause 2, the re-written tool call" in out
    assert "in the tool-call span" in out
    assert "continues the previous call: yes" in out


def test_the_recorded_replies_are_bundled_whole() -> None:
    replies = drift.load_recorded()
    assert [reply.name for reply in replies] == ["re-spelled", "re-cut"]
    for reply in replies:
        assert len(reply.token_ids) == len(reply.logprobs)
        assert reply.token_ids[-1] == 124_900  # <|im_end|>: each reply is whole
    assert [len(reply.token_ids) for reply in replies] == [50, 29]


@needs_tokenizer
def test_the_recorded_ids_drift_where_the_project_recorded_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("transformers")
    model = lfm_text.load_text(REAL_MODEL)
    respelled, recut = drift.load_recorded()
    lines = drift.respelled(model, respelled)
    assert lines[0] == "first difference: token 36 of the reply, in the tool-call span"
    assert "| ')\\\\'" in lines[1] and "| ')\"'" in lines[2]
    lines = drift.recut(model, recut)
    assert lines[0] == "29 tokens sampled; its text encodes to 28"
    assert lines[1] == "first difference: token 24 of the reply"
    assert "' .' (-0.06) 's' (-6.53) 'chema' (-0.21)" in lines[2]
    assert lines[3].endswith("' .' 'schema'")


@needs_tokenizer
def test_the_real_template_refuses_json_text_and_the_restore_continues(
    first_messages: list,
) -> None:
    pytest.importorskip("transformers")
    real = lfm_text.load_text(REAL_MODEL)
    # Chapter 17's first real reply to clinic-easy-001, word for word.
    reply = (
        "Let me first explore the database to understand its structure.</think>"
        f"{START}[bash(command='sqlite3 data.db \".tables\"')]{END}{EOT}"
    )

    class RealText(FakeModel):
        """The real tokenizer and template, with a scripted reply."""

        eos_ids = real.eos_ids

        def render(self, messages: list, tools: list | None) -> str:
            return real.render(messages, tools)

        def encode(self, text: str) -> list[int]:
            return real.encode(text)

        def decode(self, token_ids: list[int]) -> str:
            return real.decode(token_ids)

    model = RealText([real.encode(reply), real.encode(answer_reply("30"))])
    core = engine.Engine(model)
    first = answer(core, body(first_messages))
    prompt = real.decode(first["prompt_token_ids"])
    assert prompt.endswith("<|im_start|>assistant\n<think>")
    tools_line = '\nList of tools: [{"type": "function", "function": {"name": "bash"'
    assert tools_line in prompt
    copy, result = harness_turn(first, "appointments  doctors       patients")
    with pytest.raises(RenderError, match="must be a mapping"):
        real.render([*first_messages, copy, result], TOOLS)
    second = answer(core, body([*first_messages, copy, result]))
    chain = first["prompt_token_ids"] + first["choices"][0]["token_ids"]
    assert second["prompt_token_ids"][: len(chain)] == chain
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == len(chain)


# ------------------------------------------------------------------ the scripts


CH17_REPLIES = [
    call_reply('sqlite3 data.db ".tables"', think="Explore first."),
    call_reply('sqlite3 data.db ".schema patients"', think="The patients table."),
    call_reply(MURCIA, think="Count them."),
    call_reply("echo 30 > answer.txt", think="Write it."),
    answer_reply("I wrote 30 to answer.txt."),
]


def test_the_http_harness_solves_a_task_through_the_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = FakeModel(list(CH17_REPLIES))
    with live_engine(model) as base_url:
        monkey_argv = ["tiny_harness_http.py", "--base-url", base_url]
        saved, sys.argv = sys.argv, monkey_argv
        try:
            tiny_harness_http.main()
        finally:
            sys.argv = saved
    out = capsys.readouterr().out
    assert f"base URL {base_url} · API key 'any-key'" in out
    assert "call 1 · read " in out and "(0 cached)" in out
    assert 'tool call    bash {"command": "sqlite3 data.db \\".tables\\""}' in out
    assert "observation  appointments" in out
    assert "grade: correct (correct; answer.txt: '30') · 5 calls · reward 1" in out
    assert "token chain: 4 of 4 calls began with the previous call's prompt" in out
    saved_file = lfm_files.rollout_path("http", "clinic-easy-001")
    record = json.loads(saved_file.read_text())
    tool_messages = [m for m in record["messages"] if m["role"] == "tool"]
    answers = record["responses"][:4]
    calls = [r["choices"][0]["message"]["tool_calls"][0] for r in answers]
    assert [m["tool_call_id"] for m in tool_messages] == [c["id"] for c in calls]
    assert model.seen[-1].held  # every later call reused a cache


def test_the_http_harness_stops_after_three_unreadable_replies() -> None:
    empty = f"Hmm.</think>{START}[bash(command=]{END}{EOT}"  # dropped by the engine
    model = FakeModel([empty] * 3)
    with live_engine(model) as base_url:
        chat = tiny_harness_http.http_chat(base_url, "k")
        workdir = tasks.prepare_workdir(EASY, lfm_files.work_folder("t", EASY.task_id))
        rollout = tiny_harness_http.run_rollout(chat, EASY, workdir)
    assert rollout.stop == tiny_harness_http.STOP_UNREADABLE
    assert [call.why for call in rollout.calls] == [tiny_harness_http.WHY_EMPTY] * 3
    notes = [m for m in rollout.messages if m["role"] == "user"][1:]
    assert len(notes) == 2  # no note after the last: the loop stops


def test_the_http_harness_refuses_a_tool_it_did_not_offer(tmp_path: Path) -> None:
    call = {"id": "c", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
    result = tiny_harness_http.execute(call, TOOLS, tmp_path)
    assert result.output == "Error: No such tool available: Bash" and not result.ok
    call["function"] = {"name": "bash", "arguments": "{oops"}
    assert tiny_harness_http.execute(call, TOOLS, tmp_path).output.startswith(
        "Error: the tool call's arguments"
    )


def test_an_unreachable_engine_is_said_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    monkeypatch.setattr(sys, "argv", ["client.py", "--base-url", url])
    with pytest.raises(SystemExit, match="the engine did not answer: cannot reach"):
        client.main()


def run_demo(model: FakeModel, demo: str) -> str:
    """Run one of client.py's demonstrations against a live scripted engine.

    Args:
        model: The scripted model.
        demo: The demonstration's name.

    Returns:
        What it printed.
    """
    out = io.StringIO()
    with live_engine(model) as base_url, contextlib.redirect_stdout(out):
        client.DEMO_FUNCTIONS[demo](tiny_harness_http.http_chat(base_url, "k"))
    return out.getvalue()


def test_client_turns_shows_the_receipts_and_the_cache() -> None:
    model = FakeModel([CH17_REPLIES[0], CH17_REPLIES[1]])
    out = run_demo(model, "turns")
    assert '"finish_reason": "tool_calls"' in out
    assert '"prompt_token_ids": [' in out and " in all" in out
    assert '"token": "token_id:' in out
    assert "call 1: read " in out and ", 0 cached" in out
    assert "call 2's prompt begins with call 1's prompt and reply: yes" in out
    held = len(model.seen[1].held)
    assert f"{held:,} cached" in out


def test_client_queue_shows_one_request_at_a_time() -> None:
    model = FakeModel([answer_reply(str(n)) for n in range(4)], delay=0.1)
    out = run_demo(model, "queue")
    assert "4 requests at once, one worker thread" in out
    assert out.count("clinic-") + out.count("airline-") == 4
    assert "one request at a time" in out
    assert len({seen.thread for seen in model.seen}) == 1


def test_client_too_long_shows_context_length_exceeded() -> None:
    out = run_demo(FakeModel([]), "too-long")
    assert "a prompt of 20,000 words" in out
    assert "HTTP 400" in out and "context_length_exceeded" in out
    assert "the limit is 16,384" in out


@pytest.mark.parametrize(("temper", "verdict"), [(True, "processed"), (False, "raw")])
def test_client_logprobs_probe_tells_tempered_from_raw(
    temper: bool, verdict: str
) -> None:
    replies = ["Hi.</think>hello<|im_end|>", "h", "h"]
    out = run_demo(FakeModel(replies, temper=temper), "logprobs")
    assert "token 1: " in out and "likeliest: " in out
    assert "temperature 1.0: gap 2.0000" in out
    halved = "1.0000" if temper else "2.0000"
    assert f"temperature 2.0: gap {halved}" in out
    assert f"-> {verdict}" in out
    assert "HTTP 400" in out and "top_logprobs must be between 0 and 20" in out


def test_the_probe_says_unknown_when_it_cannot_measure() -> None:
    def flat(body: dict[str, Any]) -> dict[str, Any]:
        alternatives = [{"logprob": -1.0}, {"logprob": -1.2}]
        entry = {"top_logprobs": alternatives}
        return {"choices": [{"logprobs": {"content": [entry]}}]}

    assert client.probe(flat).verdict == "unknown"  # a gap of 0.2: too flat
