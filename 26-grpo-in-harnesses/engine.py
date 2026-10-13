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
for Liquid's export of the model. Chapters 23 and 24 run it unchanged, as
``ENGINE_PORT=8119 ./start.sh`` does, behind the proxy and the three harnesses.

Chapter 20 copies chapter 18's engine and adds two switches, each one way a
real server can fail a training run while every reply still looks right.
The capture proxy in front of the engine must notice both:

* ``--no-token-fields``: answers carry no ``prompt_token_ids`` and no
  ``token_ids``, and each log-probability names its token by its text
  instead of ``token_id:<n>``. That is vLLM without its token-id options,
  and close to what hosted APIs return: good for evaluation, useless for
  training, since nothing ties the numbers to exact tokens;
* ``--raw-logprobs``: log-probabilities from the scores before the
  temperature (vLLM's default ``raw_logprobs``), while the token is still
  drawn at the temperature. Every field is there and aligned, and every
  number describes a distribution the token was not drawn from.

Chapter 25 adds the two things training through the harnesses needs:

* ``--oracle``: the answer-key teacher (``teacher.py``) plays the model. No
  weights are read and MLX is never loaded: the engine keeps only LFM2.5's
  tokenizer and chat template (``lfm_text.py``, the same ones the model's
  engine renders with), renders each prompt exactly as for the model, asks
  the teacher for the reply's text, and cuts that text into the student's
  tokens, every log-probability 0 (a demonstration is certain). It serves
  on 8101 by default, the project's teacher port (``scripts/serve.sh``);
* ``--adapter PATH`` and ``POST /admin/adapter`` ``{"path": ...}``: sample
  with a LoRA adapter (chapter 13's shape), or with none (``"path": null``,
  the base model), as the project's engine does between checkpoints. Every
  prompt cache is dropped on a switch: it holds the old weights' work.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import json
import math
import socket
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from lfm_files import OUTPUT_CAP, SAMPLE_SEED, pick_model, short_path
from policy import (
    FINISH_LENGTH,
    FINISH_STOP,
    Message,
    Model,
    RenderError,
    Sampled,
    Sampling,
    Tool,
)
from toolcalls import THINK_OPEN, ParsedReply, openai_tool_calls, split_reply

HOST = "127.0.0.1"  # only this machine can reach the engine
DEFAULT_PORT = 8100  # the project's engine_port
ORACLE_PORT = 8101  # the project's answer-key teacher (scripts/serve.sh)
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
# /health's "serves" with --oracle (collect.py keeps the same words: the
# scripts never import the engine, so keep the two in step).
SERVES_TEACHER = "answer-key teacher"
BASE_ADAPTER = "base"  # no adapter: the model as downloaded
CERTAIN = 0.0  # log(1): every token of a demonstration is certain
ADAPTER_SHAPE = 'expected {"path": "<adapter file>"} or {"path": null}'
# Flags that clash with --oracle: each one needs the model's weights.
FLAGS_NEEDING_WEIGHTS = ("adapter", "fast", "no_token_fields", "raw_logprobs")
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
TOKEN_ID_PREFIX = "token_id:"  # vLLM's way to name a token by its id
# Printed under "engine ready" when a chapter 20 switch is on.
NO_TOKEN_FIELDS_NOTE = (
    "  --no-token-fields: no prompt_token_ids, no token_ids, tokens named by their text"
)
RAW_LOGPROBS_NOTE = (
    "  --raw-logprobs: log-probabilities before the temperature, tokens still "
    "drawn at it"
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
        adapter: The LoRA adapter replies are sampled with (``base``: none).
        adapter_path: Its file, or None for the base model (``/health``
            gives it, so a caller can put back what it found).
    """

    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    model_seconds: float = 0.0
    restored: int = 0
    not_restored: int = 0
    recut: int = 0
    adapter: str = BASE_ADAPTER
    adapter_path: str | None = None


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
        token_fields: Whether answers carry the token ids (False with
            ``--no-token-fields``).
        raw_logprobs: Whether log-probabilities skip the temperature (True
            with ``--raw-logprobs``).
        caches: ``(tokens held, cache)`` pairs, the most recent first.
        replies: The model's own text for each reply, by lookup key.
        stats: The running counters.
    """

    def __init__(
        self, model: Model, token_fields: bool = True, raw_logprobs: bool = False
    ) -> None:
        """Start with no caches and no remembered replies.

        Args:
            model: The model.
            token_fields: False to leave the token ids out of every answer.
            raw_logprobs: True to report untempered log-probabilities.
        """
        self.model = model
        self.token_fields = token_fields
        self.raw_logprobs = raw_logprobs
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
        sampling = request.sampling
        if self.raw_logprobs:  # --raw-logprobs: the mistake, made on purpose
            sampling = replace(sampling, raw_logprobs=True)
        sampled = self.model.sample(prompt_ids[cached:], cache, sampling)
        self.store_cache(prompt_ids + sampled.token_ids, cache)
        reply, calls = self.receipt(sampled, request.tools)
        seconds = time.perf_counter() - started
        self.stats.requests += 1
        self.stats.prompt_tokens += len(prompt_ids)
        self.stats.cached_tokens += cached
        self.stats.completion_tokens += len(sampled.token_ids)
        self.stats.model_seconds += seconds
        return Completion(prompt_ids, sampled, reply, calls, cached, seconds)

    def set_adapter(self, path: Path | None) -> str:
        """Sample with another LoRA adapter, or with none.

        Args:
            path: An adapter file, or None for the base model.

        Returns:
            The adapter now in use (``base`` for none).

        Raises:
            BadRequest: If this engine has no weights (the teacher, a
                script), or the file is missing or not this model's adapter.
        """
        switch = getattr(self.model, "set_adapter", None)
        if switch is None:
            raise BadRequest("this engine has no weights to adapt")
        try:
            self.stats.adapter = switch(path)
        except (FileNotFoundError, ValueError) as error:
            raise BadRequest(f"no adapter loaded: {error}") from error
        self.stats.adapter_path = str(path) if path else None
        # A cache holds the work of the old weights: none of it is valid now.
        self.caches.clear()
        return self.stats.adapter


@dataclass
class OracleCache:
    """A prompt cache for the teacher, who reads nothing: just the tokens.

    Attributes:
        tokens: The tokens it has seen.
    """

    tokens: list[int] = field(default_factory=list)


class OracleModel:
    """The answer-key teacher behind the engine's seam: a tokenizer, no weights.

    The engine renders a request, then samples its reply, one after the
    other on its one worker thread, so ``render`` hands the conversation to
    ``sample`` (as chapter 23's scripted model does). The conversation the
    teacher reads is the engine's: earlier replies already put back as the
    teacher wrote them. A demonstration is never drawn, so the request's
    temperature, alternatives and stop texts do not apply: the reply is the
    teacher's whole text, cut only by ``max_tokens``.

    Attributes:
        text: LFM2.5's tokenizer and chat template (``lfm_text.py``).
        teacher: Writes a reply's text from the conversation and the tools.
        eos_ids: The end-of-turn ids (the engine reads them here).
        pending: The conversation being answered, and its tools.
        answered: Replies written so far.
        serves: What ``/health`` says this engine serves.
    """

    serves = SERVES_TEACHER

    def __init__(
        self,
        text: Any,
        teacher: Callable[[list[Message], list[Tool] | None], str],
    ) -> None:
        """Keep the tokenizer and the teacher.

        Args:
            text: Anything with ``render``, ``encode``, ``decode`` and
                ``eos_ids``, as ``lfm_text.LfmText`` has.
            teacher: The answer-key teacher (``teacher.AnswerKeyTeacher``).
        """
        self.text = text
        self.teacher = teacher
        self.eos_ids = frozenset(text.eos_ids)
        self.pending: tuple[list[Message], list[Tool] | None] = ([], None)
        self.answered = 0

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Render the prompt exactly as for the model, and keep the conversation.

        Args:
            messages: The conversation, earlier replies put back.
            tools: The tools offered, or None.

        Returns:
            The prompt, ending with the generation prompt's ``<think>``.
        """
        self.pending = (messages, tools)
        return self.text.render(messages, tools)

    def encode(self, text: str) -> list[int]:
        """Cut text into the student's token ids.

        Args:
            text: Any text.

        Returns:
            The ids.
        """
        return self.text.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Write token ids back as text, special tokens kept.

        Args:
            token_ids: The ids.

        Returns:
            The text.
        """
        return self.text.decode(token_ids)

    def new_cache(self) -> OracleCache:
        """Make an empty cache.

        Returns:
            A cache holding no tokens.
        """
        return OracleCache()

    def cache_size(self, cache: OracleCache) -> int:
        """Count the tokens a cache holds.

        Args:
            cache: The cache.

        Returns:
            Its number of tokens.
        """
        return len(cache.tokens)

    def sample(
        self, new_ids: list[int], cache: OracleCache, sampling: Sampling
    ) -> Sampled:
        """Ask the teacher for the reply, and cut it into tokens as the model would.

        Args:
            new_ids: The prompt tokens the cache does not hold yet.
            cache: The cache; it also holds them and the reply afterwards.
            sampling: Only ``max_tokens`` matters: a demonstration is never
                drawn, so neither temperature nor alternatives apply.

        Returns:
            The reply, the end-of-turn token included, every log-probability
            0 (certain), no alternatives.
        """
        started = time.perf_counter()
        messages, tools = self.pending
        text = self.teacher(messages, tools)
        whole = self.encode(text) + [min(self.eos_ids)]
        ids = whole[: sampling.max_tokens]  # never longer than any reply may be
        finish = FINISH_STOP if len(ids) == len(whole) else FINISH_LENGTH
        cache.tokens += list(new_ids) + ids
        self.answered += 1
        seconds = time.perf_counter() - started
        return Sampled(ids, [CERTAIN] * len(ids), [], finish, seconds)


def logprob_entry(
    model: Model, token_id: int, logprob: float, by_id: bool = True
) -> dict[str, Any]:
    """Write one log-probability as vLLM does, the token as ``token_id:<n>``.

    Args:
        model: The model (for the token's text).
        token_id: The token.
        logprob: Its log-probability.
        by_id: False to name the token by its text, as vLLM does without
            ``--return-tokens-as-token-ids`` (``--no-token-fields``).

    Returns:
        ``{"token", "logprob", "bytes"}``.
    """
    # A token can be half a character (an emoji is several); decoded alone it
    # becomes the replacement character, so its bytes are only right for
    # whole characters, as in the project's engine.
    text = model.decode([token_id])
    return {
        "token": f"{TOKEN_ID_PREFIX}{token_id}" if by_id else text,
        "logprob": logprob,
        "bytes": list(text.encode("utf-8")),
    }


def to_openai(
    model: Model, completion: Completion, waited: float, token_fields: bool = True
) -> dict[str, Any]:
    """Shape a completion as OpenAI's chat response, with vLLM's token fields.

    Args:
        model: The model.
        completion: The completion.
        waited: Seconds the request waited for the worker thread.
        token_fields: False to leave out ``prompt_token_ids`` and
            ``token_ids`` and name each token by its text.

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
        entry = logprob_entry(model, token_id, logprob, by_id=token_fields)
        alternatives = sampled.top_logprobs[index] if sampled.top_logprobs else []
        entry["top_logprobs"] = [
            logprob_entry(model, *pair, by_id=token_fields) for pair in alternatives
        ]
        content.append(entry)
    read, wrote = len(completion.prompt_ids), len(sampled.token_ids)
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish,
        "token_ids": sampled.token_ids,
        "logprobs": {"content": content},
    }
    response: dict[str, Any] = {
        "id": RESPONSE_ID_PREFIX + uuid.uuid4().hex,
        "object": RESPONSE_OBJECT,
        "created": int(time.time()),
        "model": SERVED_MODEL,
        "prompt_token_ids": completion.prompt_ids,
        "choices": [choice],
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
    if not token_fields:  # --no-token-fields: the receipts are gone
        del response["prompt_token_ids"]
        del choice["token_ids"]
    return response


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
    app = FastAPI(title="chapter 20 engine")
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
        completion = engine.complete(request)
        return to_openai(
            engine.model, completion, waited, token_fields=engine.token_fields
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Say the engine is up, what it serves and with which adapter."""
        answer: dict[str, Any] = {"ok": True, "model": SERVED_MODEL}
        serves = getattr(engine.model, "serves", None)
        if serves:  # the model or the teacher (a scripted model says nothing)
            answer["serves"] = serves
            answer["adapter"] = engine.stats.adapter
            answer["adapter_path"] = engine.stats.adapter_path
        return answer

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

    @app.post("/admin/adapter")
    async def adapter(http: Request) -> JSONResponse:
        """Switch the adapter replies are sampled with, after the queued requests.

        Args:
            http: The request: ``{"path": "<adapter file>"}``, or
                ``{"path": null}`` for the base model.

        Returns:
            ``{"adapter": ...}``, or a 400 saying why nothing changed.
        """
        try:
            try:
                body = await http.json()
            except ValueError as error:
                raise BadRequest(f"the request body is not JSON: {error}") from error
            # "path" must be there: a missing or misspelt key must never
            # quietly mean "the base model".
            if not isinstance(body, dict) or "path" not in body:
                raise BadRequest(ADAPTER_SHAPE)
            path = body["path"]
            if path is not None and not (isinstance(path, str) and path):
                raise BadRequest(ADAPTER_SHAPE)
            wanted = Path(path).expanduser() if path else None
            loop = asyncio.get_running_loop()
            # On the worker thread, after every request already waiting for it.
            label = await loop.run_in_executor(worker, engine.set_adapter, wanted)
        except BadRequest as error:
            print(f"refused · adapter · {error}", flush=True)
            return error_response(error)
        print(f"adapter · now sampling with {label}", flush=True)
        return JSONResponse({"adapter": label})

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

    Raises:
        SystemExit: From argparse, if ``--oracle`` comes with a flag that
            needs the model's weights.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"default {DEFAULT_PORT}, or {ORACLE_PORT} with --oracle",
    )
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED, help="MLX's seed, above temperature 0"
    )
    parser.add_argument(
        "--no-token-fields",
        action="store_true",
        help="leave prompt_token_ids and token_ids out (an evaluation-only engine)",
    )
    parser.add_argument(
        "--raw-logprobs",
        action="store_true",
        help="report log-probabilities before the temperature (vLLM's default)",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="serve the answer-key teacher (teacher.py): no weights, no MLX",
    )
    parser.add_argument(
        "--adapter", type=Path, default=None, help="a LoRA adapter to sample with"
    )
    args = parser.parse_args()
    if args.oracle:
        clashing = [name for name in FLAGS_NEEDING_WEIGHTS if getattr(args, name)]
        if clashing:
            flags = ", ".join("--" + name.replace("_", "-") for name in clashing)
            parser.error(f"--oracle reads no weights, so it takes no {flags}")
    if args.port is None:
        args.port = ORACLE_PORT if args.oracle else DEFAULT_PORT
    return args


def serve_oracle(sock: socket.socket) -> None:
    """Serve the answer-key teacher on a bound socket, until Ctrl-C.

    Args:
        sock: From ``claim_port``.

    Raises:
        SystemExit: If LFM2.5's checked tokenizer is not on disk.
    """
    # Only the tokenizer half: MLX is never imported on this path.
    from lfm_files import TokenizerMissingError, lfm_tokenizer_file
    from lfm_text import load_text
    from teacher import AnswerKeyTeacher

    try:
        folder = lfm_tokenizer_file().parent
    except TokenizerMissingError as error:
        raise SystemExit(str(error)) from error
    teacher = AnswerKeyTeacher.from_train_file()
    model = OracleModel(load_text(folder), teacher)
    port = sock.getsockname()[1]
    print(f"engine ready · {SERVES_TEACHER} · {HOST}:{port} · no weights")
    print(f"  LFM2.5's tokenizer and chat template ({short_path(folder)}), no MLX")
    print(
        f"  {len(teacher.by_question)} training questions; every reply's "
        "log-probabilities are 0"
    )
    print("  one worker thread; each request prints a line here")
    sys.stdout.flush()
    engine = Engine(model)
    with contextlib.suppress(KeyboardInterrupt):
        serve(create_app(engine), sock)
    print(
        f"engine stopped · {engine.stats.requests} requests · "
        f"{model.answered} replies written by the teacher",
        flush=True,
    )


def main() -> None:
    """Load the model (or only the teacher's tokenizer) and serve it on 127.0.0.1."""
    args = parse_args()
    sock = claim_port(args.port)
    if args.oracle:
        serve_oracle(sock)
        return
    from lfm import LfmModel, active_memory, peak_memory  # MLX loads here only

    folder, which = pick_model(args.fast)
    model = LfmModel(folder, args.seed)
    engine = Engine(
        model, token_fields=not args.no_token_fields, raw_logprobs=args.raw_logprobs
    )
    if args.adapter:
        try:
            engine.set_adapter(args.adapter)
        except BadRequest as error:
            raise SystemExit(str(error)) from error
    port = sock.getsockname()[1]
    memory = active_memory() / GIGABYTE
    print(f"engine ready · {SERVED_MODEL} 4-bit · {HOST}:{port} · {memory:.1f} GB")
    print(f"  {short_path(folder)} ({which}), loaded in {model.seconds:.1f} s")
    print(f"  adapter: {engine.stats.adapter}")
    print("  one worker thread; each request prints a line here")
    if args.no_token_fields:
        print(NO_TOKEN_FIELDS_NOTE)
    if args.raw_logprobs:
        print(RAW_LOGPROBS_NOTE)
    sys.stdout.flush()  # the lines above reach a log file before any request
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl-C: how a server is stopped
        serve(create_app(engine), sock)
    print(
        f"engine stopped · {engine.stats.requests} requests · peak memory "
        f"{peak_memory() / GIGABYTE:.2f} GB, the weights included",
        flush=True,
    )


if __name__ == "__main__":
    main()
