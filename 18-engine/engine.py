# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi==0.142.2",
#     "uvicorn==0.54.0",
#     "jinja2==3.1.6",
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""An engine that keeps the receipts: the 4-bit LFM2.5 behind OpenAI's chat API.

Real harnesses reach a model over HTTP in OpenAI's chat format: POST
``/v1/chat/completions`` with the messages and the tools, and get back a
message (text, or tool calls as JSON) and the token usage. This is that
server for our model, after the project's ``mhrl/engine/policy.py`` and
``server.py``: the Mac stand-in for the article's vLLM server, which needs
NVIDIA GPUs. Besides the reply, it returns what hosted APIs never give
back, in vLLM's own fields:

* ``prompt_token_ids``: the exact tokens the model read;
* ``choices[0].token_ids``: the exact tokens it sampled;
* ``logprobs``: each sampled token's log-probability, written
  ``token_id:<n>``, under the distribution it was drawn from (temperature
  applied, nothing cut, float32), with up to 20 alternatives per position
  when ``top_logprobs`` asks.

Three more jobs keep a conversation on one token chain, and the model busy:

* the model writes a tool call as text (``[bash(command='ls')]`` between two
  special tokens). The engine sends it out as OpenAI ``tool_calls`` JSON, and
  when the harness sends that JSON back on the next call, it puts the model's
  own text back (found by tool-call id, else by the visible text), so the
  chat template never re-writes a reply in its own words;
* 8 prompt caches, matched by the longest exact token prefix, so call k+1
  pays only for the tokens added since call k (``cached_tokens``);
* one worker thread does all the model's work (MLX work stays on the thread
  that made it), so requests queue for it and finish one after another.

A prompt over 16,384 tokens is refused with ``context_length_exceeded``.
The engine samples from the full distribution at the request's temperature:
like the project's, it ignores ``top_p``, ``top_k``, penalties, ``seed`` and
``tool_choice``, and lowers ``max_tokens`` to 1,024. It refuses ``stream``
and ``n`` above 1, whose answers it could not give. Only this machine can
reach it, so any API key works.

Run ``uv run engine.py`` (port 8100), ``--port`` for another port, ``--fast``
for Liquid's export of the model.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import math
import socket
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from lfm_files import OUTPUT_CAP, SAMPLE_SEED, pick_model, short_path
from policy import FINISH_STOP, Message, Model, RenderError, Sampled, Sampling, Tool
from toolcalls import THINK_OPEN, ParsedReply, openai_tool_calls, split_reply

HOST = "127.0.0.1"  # only this machine can reach the engine
DEFAULT_PORT = 8100  # the project's engine_port
EXAMPLE_PORT = 8119  # suggested when the default is taken
SERVED_MODEL = "lfm2.5-2.6b"  # the name it reports; requests may name any model
MAX_PROMPT_TOKENS = 16_384  # the project's limit (the model itself reads 131,072)
MAX_COMPLETION_TOKENS = OUTPUT_CAP  # 1,024, the project's cap on one reply
DEFAULT_TEMPERATURE = 1.0  # OpenAI's default, when a request names none
MAX_TOP_LOGPROBS = 20  # OpenAI's ceiling, vLLM's default and the project's
PROMPT_CACHE_SLOTS = 8  # the project's Defaults.PROMPT_CACHE_SLOTS
REMEMBERED_REPLIES = 4_096  # replies whose exact text the engine can put back
WORKER_THREADS = 1  # ONE thread does all model work; requests queue for it
BAD_REQUEST = 400
INVALID = "invalid_request_error"  # OpenAI's error type for a bad request
TOO_LONG = "context_length_exceeded"  # OpenAI's error code for a long prompt
ASSISTANT = "assistant"
TOOL_CALLS = "tool_calls"  # the finish reason when a reply calls tools
RESPONSE_ID_PREFIX = "chatcmpl-"  # OpenAI's ids for a chat completion
RESPONSE_OBJECT = "chat.completion"
TIMING_DIGITS = 3  # timings are reported to the millisecond
ID_KEY = "id:"  # a remembered reply, looked up by one of its tool calls' ids
TEXT_KEY = "text:"  # ... or by its visible text
GIGABYTE = 1_000_000_000
PORT_BUSY = (
    "{host}:{port} is in use: another program listens there (or an engine that "
    "just stopped has not let go of it yet). Start the engine on a free port "
    "with --port, for example: uv run engine.py --port {example}"
)
PORT_REFUSED = (
    "cannot listen on {host}:{port}: {reason}. Start the engine on another "
    "port with --port, for example: uv run engine.py --port {example}"
)
RECUT_NOTE = (
    "note · this reply's text cuts into other tokens than the ones sampled: "
    "the next call can put the text back, not these exact tokens"
)


class BadRequest(Exception):
    """Raised when a request cannot be answered as asked.

    Attributes:
        code: OpenAI's error code, such as ``context_length_exceeded``, or None.
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        """Keep the message and the code.

        Args:
            message: What is wrong, for the caller.
            code: OpenAI's error code, or None.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ChatRequest:
    """One chat request, checked.

    Attributes:
        messages: The conversation, as the harness sent it.
        tools: The tools offered, or None.
        sampling: How to draw the reply.
    """

    messages: list[Message]
    tools: list[Tool] | None
    sampling: Sampling


@dataclass(frozen=True)
class Completion:
    """One answered request, with every receipt.

    Attributes:
        prompt_ids: The tokens the model read.
        sampled: What it sampled, token by token.
        reply: The reply split into reasoning, visible text and calls.
        tool_calls: The calls as OpenAI JSON, each with its id.
        cached_tokens: Prompt tokens whose work a prompt cache already held.
        seconds: Time on the worker thread for this request.
    """

    prompt_ids: list[int]
    sampled: Sampled
    reply: ParsedReply
    tool_calls: list[dict[str, Any]]
    cached_tokens: int
    seconds: float


@dataclass
class Stats:
    """Running counters, shown on ``/admin/stats``.

    Attributes:
        requests: Requests answered.
        prompt_tokens: Tokens read, the cached ones included.
        cached_tokens: Tokens whose work came from a prompt cache.
        completion_tokens: Tokens sampled.
        model_seconds: Time spent answering.
        restored: Assistant turns put back as the model wrote them (counted
            on every call that re-sends them).
        not_restored: Assistant turns not found, so rendered from the
            harness's copy: those calls do not continue the earlier ones.
        recut: Replies whose own text cuts into other tokens.
    """

    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    model_seconds: float = 0.0
    restored: int = 0
    not_restored: int = 0
    recut: int = 0


def as_int(body: dict[str, Any], name: str, default: int) -> int:
    """Read a whole number from a request, or its default when it is absent.

    Args:
        body: The request's JSON body.
        name: The field.
        default: Its value when the field is absent or null.

    Returns:
        The number.

    Raises:
        BadRequest: If the field holds anything but a whole number.
    """
    value = body.get(name)
    if value is None:
        return default
    whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) or not whole:  # True is an int to Python, not to JSON
        raise BadRequest(f"{name} must be a whole number, not {value!r}")
    return int(value)


def as_float(body: dict[str, Any], name: str, default: float) -> float:
    """Read a number from a request, or its default when it is absent.

    Args:
        body: The request's JSON body.
        name: The field.
        default: Its value when the field is absent or null.

    Returns:
        The number.

    Raises:
        BadRequest: If the field holds anything but a finite number.
    """
    value = body.get(name)
    if value is None:
        return default
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if not number or not math.isfinite(value):
        raise BadRequest(f"{name} must be a number, not {value!r}")
    return float(value)


def read_messages(body: dict[str, Any]) -> tuple[list[Message], list[Tool] | None]:
    """Check a request's conversation and tools: their shapes, not their words.

    Args:
        body: The request's JSON body.

    Returns:
        The messages, and the tools (None when there are none).

    Raises:
        BadRequest: If either is not a list of objects.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    if not all(isinstance(message, dict) for message in messages):
        raise BadRequest("every message must be an object")
    for message in messages:
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            raise BadRequest("tool_calls must be a list of objects")
    tools = body.get("tools") or None
    if tools is not None and not (
        isinstance(tools, list) and all(isinstance(tool, dict) for tool in tools)
    ):
        raise BadRequest("tools must be a list of objects")
    return messages, tools


def read_sampling(body: dict[str, Any]) -> Sampling:
    """Check how a request asks for its reply to be drawn.

    Args:
        body: The request's JSON body.

    Returns:
        The sampling settings the engine will use.

    Raises:
        BadRequest: For anything the engine cannot answer as asked.
    """
    if body.get("stream"):
        raise BadRequest("stream is not supported: the reply comes as one JSON body")
    if as_int(body, "n", 1) != 1:
        raise BadRequest("n must be 1: the engine samples one reply per request")
    newer = body.get("max_completion_tokens") is not None  # OpenAI's newer name
    cap_field = "max_completion_tokens" if newer else "max_tokens"
    asked = as_int(body, cap_field, MAX_COMPLETION_TOKENS)
    temperature = as_float(body, "temperature", DEFAULT_TEMPERATURE)
    if temperature < 0:
        raise BadRequest("temperature must be 0 or more")
    top = as_int(body, "top_logprobs", 0)
    if not 0 <= top <= MAX_TOP_LOGPROBS:
        raise BadRequest(f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}")
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop):
        raise BadRequest("stop must be a text or a list of texts, none of them empty")
    # Like the project's engine, a reply is never longer than 1,024 tokens.
    cap = max(1, min(asked, MAX_COMPLETION_TOKENS))
    return Sampling(cap, temperature, top, tuple(stop))


def read_request(body: Any) -> ChatRequest:
    """Check a chat request and read what the engine uses from it.

    Args:
        body: The request's JSON body.

    Returns:
        The request.

    Raises:
        BadRequest: For anything the engine cannot answer as asked.
    """
    if not isinstance(body, dict):
        raise BadRequest("the request body must be a JSON object")
    messages, tools = read_messages(body)
    return ChatRequest(messages, tools, read_sampling(body))


def harness_copy(message: Message) -> Message:
    """Prepare a harness's own copy of a reply for the chat template.

    OpenAI's JSON carries a call's arguments as a JSON string, and LFM2.5's
    template refuses a string (it would have to guess how the model spelled
    the call), so each string is read back into an object first.

    Args:
        message: An assistant message as the harness sent it.

    Returns:
        The same message, its tool calls' arguments as objects.

    Raises:
        BadRequest: If some arguments are not a JSON object.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                reason = f"tool call arguments are not JSON: {error}"
                raise BadRequest(reason) from error
        if not isinstance(arguments, dict):
            raise BadRequest("tool call arguments must be a JSON object")
        calls.append({**call, "function": {**function, "arguments": arguments}})
    return {**message, "tool_calls": calls} if calls else message


class Engine:
    """The model, the replies it remembers, its prompt caches and its counters.

    Every method runs on the one worker thread, so nothing here needs a lock
    (``/admin/stats`` only reads the counters). A request whose caller hangs
    up still runs to its end once it reaches the worker. A cache serves one
    request at a time and cannot be copied, so of several requests with the
    same prompt only the first finds it.

    Attributes:
        model: The model (``lfm.py``, or a scripted one in the tests).
        caches: ``(tokens held, cache)`` pairs, the most recent first.
        replies: The model's own text for each reply, by lookup key.
        stats: The running counters.
    """

    def __init__(self, model: Model) -> None:
        """Start with no caches and no remembered replies.

        Args:
            model: The model.
        """
        self.model = model
        self.caches: list[tuple[list[int], Any]] = []
        self.replies: OrderedDict[str, str] = OrderedDict()
        self.stats = Stats()

    def remember(self, key: str, raw: str) -> None:
        """Keep a reply's exact text under a lookup key, forgetting the oldest.

        Args:
            key: ``id:<tool call id>`` or ``text:<visible text>``.
            raw: The reply as the model wrote it, from ``<think>`` on.
        """
        self.replies[key] = raw
        self.replies.move_to_end(key)
        while len(self.replies) > REMEMBERED_REPLIES:
            self.replies.popitem(last=False)

    def restore(self, message: Message) -> Message:
        """Swap a harness's copy of a reply for the model's own text.

        The copy has no reasoning, and its tool calls are JSON: rendered by
        the template, it would not be the tokens the model sampled. Putting
        the original text back keeps every call on one token chain. A reply
        with tool calls is found by their ids only; one without, by its
        visible text (two replies with the same text, such as "Done.", share
        one entry: the later one's).

        Args:
            message: One message of a request.

        Returns:
            The message to render.

        Raises:
            BadRequest: If an unknown copy's arguments are not a JSON object.
        """
        if message.get("role") != ASSISTANT:
            return message
        calls = message.get("tool_calls") or []
        keys = [ID_KEY + str(call.get("id")) for call in calls]
        content = message.get("content")
        if not calls and isinstance(content, str) and content:
            keys.append(TEXT_KEY + content)
        for key in keys:
            raw = self.replies.get(key)
            if raw is not None:
                self.replies.move_to_end(key)  # used again: forgotten last
                self.stats.restored += 1
                return {"role": ASSISTANT, "content": raw}
        self.stats.not_restored += 1  # not a reply of ours (or long forgotten)
        return harness_copy(message)

    def render(self, messages: list[Message], tools: list[Tool] | None) -> list[int]:
        """Turn a request's conversation into the exact prompt tokens.

        Args:
            messages: The conversation, as the harness sent it.
            tools: The tools offered, or None.

        Returns:
            The prompt's token ids, ending with the generation prompt.

        Raises:
            BadRequest: If the chat template refuses the conversation.
        """
        restored = [self.restore(message) for message in messages]
        try:
            text = self.model.render(restored, tools)
        except RenderError as error:
            raise BadRequest(f"the chat template refused it: {error}") from error
        return self.model.encode(text)

    def take_cache(self, prompt_ids: list[int]) -> tuple[Any, int]:
        """Take the cache whose tokens are the longest exact prefix of the prompt.

        A cache can only be extended, never cut back (LFM2's convolution
        layers keep a running state), so it must hold a strict prefix.

        Args:
            prompt_ids: The new prompt.

        Returns:
            A cache, and how many of the prompt's tokens it already holds.
        """
        best, best_size = -1, 0  # an empty cache is no help: start at 0 tokens
        for index, (held, _) in enumerate(self.caches):
            size = len(held)
            # Longer than the best so far, yet shorter than the prompt (at least
            # one new token must be left to read), and the prompt's exact start.
            if best_size < size < len(prompt_ids) and prompt_ids[:size] == held:
                best, best_size = index, size
        if best < 0:
            return self.model.new_cache(), 0
        _, cache = self.caches.pop(best)  # taken out: in use by this request only
        return cache, best_size

    def store_cache(self, tokens: list[int], cache: Any) -> None:
        """Put a cache back first in line, keyed by the tokens it holds.

        Args:
            tokens: The prompt and the reply.
            cache: The cache after the reply.
        """
        held = tokens[: self.model.cache_size(cache)]
        self.caches.insert(0, (held, cache))
        del self.caches[PROMPT_CACHE_SLOTS:]  # the oldest is dropped

    def receipt(
        self, sampled: Sampled, tools: list[Tool] | None
    ) -> tuple[ParsedReply, list[dict[str, Any]]]:
        """Split a reply, give its calls ids, and remember its exact text.

        Args:
            sampled: The reply.
            tools: The request's tools (they name positional arguments).

        Returns:
            The split reply, and its calls as OpenAI JSON.
        """
        written = [t for t in sampled.token_ids if t not in self.model.eos_ids]
        text = self.model.decode(written)
        reply = split_reply(text, tools)
        calls = openai_tool_calls(reply.calls)
        raw = THINK_OPEN + text  # the prompt ended with <think>; the reply follows
        for call in calls:
            self.remember(ID_KEY + call["id"], raw)
        if reply.content:
            self.remember(TEXT_KEY + reply.content, raw)
        # Chapter 9: one text, many cuts. Encoding the text again (one more
        # pass of the tokenizer per reply) tells whether putting the text back
        # will give the next prompt these exact tokens.
        if self.model.encode(text) != written:
            self.stats.recut += 1
            print(RECUT_NOTE, flush=True)
        return reply, calls

    def complete(self, request: ChatRequest) -> Completion:
        """Answer one request: render, check, sample, and keep the receipts.

        Args:
            request: The checked request.

        Returns:
            The completion.

        Raises:
            BadRequest: If the template refuses the conversation or the
                prompt is over 16,384 tokens.
        """
        started = time.perf_counter()
        prompt_ids = self.render(request.messages, request.tools)
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise BadRequest(
                f"prompt is {len(prompt_ids):,} tokens; the limit is "
                f"{MAX_PROMPT_TOKENS:,}",
                TOO_LONG,
            )
        cache, cached = self.take_cache(prompt_ids)
        sampled = self.model.sample(prompt_ids[cached:], cache, request.sampling)
        self.store_cache(prompt_ids + sampled.token_ids, cache)
        reply, calls = self.receipt(sampled, request.tools)
        seconds = time.perf_counter() - started
        self.stats.requests += 1
        self.stats.prompt_tokens += len(prompt_ids)
        self.stats.cached_tokens += cached
        self.stats.completion_tokens += len(sampled.token_ids)
        self.stats.model_seconds += seconds
        return Completion(prompt_ids, sampled, reply, calls, cached, seconds)


def logprob_entry(model: Model, token_id: int, logprob: float) -> dict[str, Any]:
    """Write one log-probability as vLLM does, the token as ``token_id:<n>``.

    Args:
        model: The model (for the token's text).
        token_id: The token.
        logprob: Its log-probability.

    Returns:
        ``{"token", "logprob", "bytes"}``.
    """
    # A token can be half a character (an emoji is several); decoded alone it
    # becomes the replacement character, so its bytes are only right for
    # whole characters, as in the project's engine.
    return {
        "token": f"token_id:{token_id}",
        "logprob": logprob,
        "bytes": list(model.decode([token_id]).encode("utf-8")),
    }


def to_openai(model: Model, completion: Completion, waited: float) -> dict[str, Any]:
    """Shape a completion as OpenAI's chat response, with vLLM's token fields.

    Args:
        model: The model.
        completion: The completion.
        waited: Seconds the request waited for the worker thread.

    Returns:
        The response body.
    """
    sampled = completion.sampled
    message: dict[str, Any] = {
        "role": ASSISTANT,
        "content": completion.reply.content or None,
    }
    finish = sampled.finish_reason
    if completion.tool_calls:
        message[TOOL_CALLS] = completion.tool_calls
        finish = TOOL_CALLS if finish == FINISH_STOP else finish
    content = []
    pairs = zip(sampled.token_ids, sampled.logprobs, strict=True)
    for index, (token_id, logprob) in enumerate(pairs):
        entry = logprob_entry(model, token_id, logprob)
        alternatives = sampled.top_logprobs[index] if sampled.top_logprobs else []
        entry["top_logprobs"] = [logprob_entry(model, *pair) for pair in alternatives]
        content.append(entry)
    read, wrote = len(completion.prompt_ids), len(sampled.token_ids)
    return {
        "id": RESPONSE_ID_PREFIX + uuid.uuid4().hex,
        "object": RESPONSE_OBJECT,
        "created": int(time.time()),
        "model": SERVED_MODEL,
        "prompt_token_ids": completion.prompt_ids,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "token_ids": sampled.token_ids,
                "logprobs": {"content": content},
            }
        ],
        "usage": {
            "prompt_tokens": read,
            "completion_tokens": wrote,
            "total_tokens": read + wrote,
            "prompt_tokens_details": {"cached_tokens": completion.cached_tokens},
        },
        # Not part of OpenAI's API: where the time went, as llama.cpp reports it.
        "timings": {
            "waited_s": round(waited, TIMING_DIGITS),
            "first_token_s": round(sampled.first_token_seconds, TIMING_DIGITS),
            "total_s": round(completion.seconds, TIMING_DIGITS),
        },
    }


def request_line(response: dict[str, Any]) -> str:
    """Write the engine's one console line about an answered request.

    Args:
        response: The response body.

    Returns:
        Such as ``chat · read 243 tokens (0 cached) · wrote 89 · tool_calls ...``.
    """
    usage, timings = response["usage"], response["timings"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return (
        f"chat · read {usage['prompt_tokens']:,} tokens ({cached:,} cached) · wrote "
        f"{usage['completion_tokens']:,} · {response['choices'][0]['finish_reason']} "
        f"· waited {timings['waited_s']:.1f} s · {timings['total_s']:.1f} s"
    )


def error_response(error: BadRequest) -> JSONResponse:
    """Write a refusal as OpenAI does.

    Args:
        error: The reason.

    Returns:
        A 400 response with ``{"error": {"message", "type", "code"}}``.
    """
    body = {"error": {"message": str(error), "type": INVALID, "code": error.code}}
    return JSONResponse(body, status_code=BAD_REQUEST)


def create_app(engine: Engine) -> FastAPI:
    """Build the HTTP server around an engine.

    Args:
        engine: The engine.

    Returns:
        The FastAPI app.
    """
    app = FastAPI(title="chapter 18 engine")
    worker = ThreadPoolExecutor(max_workers=WORKER_THREADS, thread_name_prefix="model")

    def answer(request: ChatRequest, arrived: float) -> dict[str, Any]:
        """Answer on the worker thread, timing the wait for it.

        Args:
            request: The checked request.
            arrived: When it reached the server (``time.perf_counter``).

        Returns:
            The response body.
        """
        waited = time.perf_counter() - arrived
        return to_openai(engine.model, engine.complete(request), waited)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Say the engine is up."""
        return {"ok": True, "model": SERVED_MODEL}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        """List the one model served, as OpenAI's API does."""
        return {"object": "list", "data": [{"id": SERVED_MODEL, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(http: Request) -> JSONResponse:
        """Answer one chat request, after the requests ahead of it."""
        arrived = time.perf_counter()
        try:
            try:
                body = await http.json()
            except ValueError as error:  # not JSON, or not UTF-8
                raise BadRequest(f"the request body is not JSON: {error}") from error
            request = read_request(body)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(worker, answer, request, arrived)
        except BadRequest as error:
            print(f"refused · {error.code or INVALID} · {error}", flush=True)
            return error_response(error)
        print(request_line(response), flush=True)
        return JSONResponse(response)

    @app.get("/admin/stats")
    async def stats() -> dict[str, Any]:
        """Show the running counters."""
        return asdict(engine.stats)

    return app


def claim_port(port: int) -> socket.socket:
    """Take the address before the model loads, so a busy port fails at once.

    No SO_REUSEADDR: with it, macOS lets a program bind 127.0.0.1 under one
    that listens on every address, and take that program's traffic. The price:
    for about half a minute after an engine stops, its old connections still
    hold the port, and a new engine there is told the port is in use.

    Args:
        port: The port (0 for any free one).

    Returns:
        The socket, bound and listening.

    Raises:
        SystemExit: If the port is taken or refused, naming the --port flag.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, port))
    except OSError as error:
        sock.close()
        example = EXAMPLE_PORT if port != EXAMPLE_PORT else EXAMPLE_PORT + 1
        if error.errno == errno.EADDRINUSE:
            message = PORT_BUSY.format(host=HOST, port=port, example=example)
        else:  # a port below 1024, say, which needs administrator rights
            reason = error.strerror or str(error)
            message = PORT_REFUSED.format(
                host=HOST, port=port, reason=reason, example=example
            )
        raise SystemExit(message) from error
    sock.listen()  # callers wait in line while the model loads
    return sock


def serve(app: FastAPI, sock: socket.socket) -> None:
    """Serve the app on a socket already bound, until stopped.

    Args:
        app: The app.
        sock: From ``claim_port``.
    """
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    server.run(sockets=[sock])


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}"
    )
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED, help="MLX's seed, above temperature 0"
    )
    return parser.parse_args()


def main() -> None:
    """Load the model and serve it on 127.0.0.1."""
    args = parse_args()
    sock = claim_port(args.port)
    from lfm import LfmModel, active_memory  # MLX loads here, so the rest runs without

    folder, which = pick_model(args.fast)
    model = LfmModel(folder, args.seed)
    port = sock.getsockname()[1]
    memory = active_memory() / GIGABYTE
    print(f"engine ready · {SERVED_MODEL} 4-bit · {HOST}:{port} · {memory:.1f} GB")
    print(f"  {short_path(folder)} ({which}), loaded in {model.seconds:.1f} s")
    print("  one worker thread; each request prints a line here", flush=True)
    serve(create_app(Engine(model)), sock)


if __name__ == "__main__":
    main()
