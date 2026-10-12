# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi==0.142.2",
#     "uvicorn==0.54.0",
# ]
# ///
"""Chapter 20's engine with a scripted model instead of LFM2.5: no weights, no MLX.

This chapter's cases need replies we choose: a retry, a summarised history,
a turn without log-probabilities. So the engine here is chapter 20's real
``engine.py`` (its request checks, its receipts, its restore and its prompt
caches) with a scripted model behind its seam, the one chapter 20's tests
use, plus a remote control:

* ``POST /script`` ``{"replies": [{"text": ..., "logprobs": true}, ...]}``
  queues the next replies; ``GET /script`` shows the queue, ``DELETE
  /script`` empties it. With the queue empty, the model answers by
  default: a ``bash ls`` call when the prompt offers tools, else ``ok``
  (the proxy's own probes are answered that way);
* a reply queued with ``"logprobs": false`` comes back with all its tokens
  and every log-probability null, as from an engine that could not score it.

The same server holds a second engine under ``/broken``: identical, except
that whenever a request offers tools, every log-probability of its reply is
null. A probe that sends no tools sees a healthy engine; every agent call
offers tools. ``graph_lab.py``'s last case points a session at it.

Run ``uv run scripted_engine.py --port 8119`` (``start.sh scripted`` does).
It needs FastAPI and uvicorn, as ``engine.py`` does, and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import engine
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from policy import FINISH_LENGTH, FINISH_STOP, GREEDY, RenderError, Sampled, Sampling
from scripted_replies import (
    SCRIPT_PATH,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Reply,
    answer_reply,
    argument,
    call_reply,
    decode,
    encode,
)

EOS = 3  # <|im_end|> in the scripted vocabulary (scripted_replies.MARKERS)
TOOLS_MARK = "List of tools: "  # where the scripted template prints the tools
# A small vocabulary of scores for the alternatives (tokens 200, 201, ...):
# the gap between the two likeliest is 2.0 at temperature 1, 1.0 at 2.
FAKE_SCORES = [4.0, 2.0, 1.5, 1.0, 0.0, -1.0, -2.0]
FIRST_ALTERNATIVE = 200
LOGPROB_STEP = -0.125  # the scripted log-probabilities: -0.125, -0.25, ...
LOGPROB_CYCLE = 4
DEFAULT_COMMAND = "ls"
DEFAULT_ANSWER = "ok"
BROKEN_PATH = "/broken"  # the second engine: no log-probabilities with tools
READY = "engine ready · scripted model (no MLX) · 127.0.0.1:{port}"
BROKEN_READY = (
    "engine ready · broken copy at 127.0.0.1:{port}/broken: no log-probabilities "
    "when a request offers tools"
)
STOPPED = "engine stopped · {requests} requests · broken engine {broken} requests"


def scripted_template(messages: list[dict[str, Any]], tools: list | None) -> str:
    """Lay a chat out the way LFM2.5's template does, for what this chapter needs.

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
    if not isinstance(system, str):  # a list of text blocks
        system = json.dumps(system)
    if tools:
        system += ("\n" if system else "") + TOOLS_MARK + json.dumps(tools)
    text = "<|startoftext|>"
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    for message in rest:
        text += f"<|im_start|>{message['role']}\n"
        content = message.get("content") or ""
        text += content if isinstance(content, str) else json.dumps(content)
        written = []
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                raise RenderError("Tool call arguments must be a mapping")
            pairs = ", ".join(f"{k}={argument(v)}" for k, v in arguments.items())
            written.append(f"{call['function']['name']}({pairs})")
        if written:
            text += f"{TOOL_CALL_START}[{', '.join(written)}]{TOOL_CALL_END}"
        text += "<|im_end|>\n"
    return text + "<|im_start|>assistant\n<think>"


@dataclass
class ScriptedCache:
    """A prompt cache that is simply the tokens it has read."""

    tokens: list[int] = field(default_factory=list)


class ScriptedModel:
    """A scripted model behind the engine's seam, for any number of requests.

    Attributes:
        queue: Replies to give next, in order.
        answered: Requests sampled so far.
        no_logprobs_with_tools: True for the broken engine: a request that
            offers tools gets its reply with every log-probability null.
    """

    eos_ids = frozenset({EOS})

    def __init__(self, no_logprobs_with_tools: bool = False) -> None:
        """Start with no queued replies.

        Args:
            no_logprobs_with_tools: True for the broken engine.
        """
        self.queue: list[Reply] = []
        self.answered = 0
        self.no_logprobs_with_tools = no_logprobs_with_tools
        self.lock = threading.Lock()

    def push(self, replies: list[Reply]) -> None:
        """Queue replies for the next requests.

        Args:
            replies: The replies, in order.
        """
        with self.lock:
            self.queue.extend(replies)

    def clear(self) -> None:
        """Forget every queued reply."""
        with self.lock:
            self.queue.clear()

    def render(self, messages: list[dict[str, Any]], tools: list | None) -> str:
        """Render with the scripted template."""
        return scripted_template(messages, tools)

    def encode(self, text: str) -> list[int]:
        """Encode the scripted way."""
        return encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode the scripted way."""
        return decode(token_ids)

    def new_cache(self) -> ScriptedCache:
        """Make an empty cache."""
        return ScriptedCache()

    def cache_size(self, cache: ScriptedCache) -> int:
        """Count the tokens a cache holds."""
        return len(cache.tokens)

    def alternatives(self, sampling: Sampling) -> list[tuple[int, float]]:
        """The likeliest scripted tokens: tempered, or raw when asked.

        Args:
            sampling: The request's sampling settings.

        Returns:
            ``(token id, log-probability)`` pairs, best first.
        """
        tempered = sampling.temperature > GREEDY and not sampling.raw_logprobs
        scale = sampling.temperature if tempered else 1.0
        scaled = [score / scale for score in FAKE_SCORES]
        total = math.log(sum(math.exp(s) for s in scaled))
        pairs = [(FIRST_ALTERNATIVE + i, s - total) for i, s in enumerate(scaled)]
        return pairs[: sampling.top_logprobs]

    def next_reply(self, prompt: str) -> Reply:
        """The queued reply, or the default one for this prompt.

        Args:
            prompt: The whole prompt, as text.

        Returns:
            The reply.
        """
        with self.lock:
            self.answered += 1
            if self.queue:
                return self.queue.pop(0)
        if TOOLS_MARK in prompt:
            return call_reply(DEFAULT_COMMAND)
        return answer_reply(DEFAULT_ANSWER)

    def sample(
        self, new_ids: list[int], cache: ScriptedCache, sampling: Sampling
    ) -> Sampled:
        """Hand back the next reply, as ids with log-probabilities (or nulls).

        Args:
            new_ids: The prompt tokens the cache does not hold yet.
            cache: The cache; it also holds them and the reply afterwards.
            sampling: How to draw (only ``max_tokens`` and the alternatives
                matter to a script).

        Returns:
            The reply.
        """
        prompt = decode(cache.tokens + list(new_ids))
        reply = self.next_reply(prompt)
        whole = encode(reply.text)
        ids = whole[: sampling.max_tokens]
        cut = len(whole) > len(ids) or not ids or ids[-1] not in self.eos_ids
        scored = reply.logprobs and not (
            self.no_logprobs_with_tools and TOOLS_MARK in prompt
        )
        # A reply the engine "could not score": the tokens are real, and
        # every number that training would need is missing.
        logprobs = [
            LOGPROB_STEP * (i % LOGPROB_CYCLE + 1) if scored else None
            for i in range(len(ids))
        ]
        tops = []
        if sampling.top_logprobs:
            tops = [self.alternatives(sampling) for _ in ids]
        cache.tokens += list(new_ids) + ids
        finish = FINISH_LENGTH if cut else FINISH_STOP
        return Sampled(ids, logprobs, tops, finish, 0.001)


def add_remote_control(app: FastAPI, model: ScriptedModel) -> None:
    """Give an engine's app the ``/script`` routes that queue its replies.

    Args:
        app: The engine's FastAPI app.
        model: Its scripted model.
    """

    def state() -> dict[str, Any]:
        with model.lock:
            return {"queued": len(model.queue), "answered": model.answered}

    @app.get(SCRIPT_PATH)
    async def show() -> dict[str, Any]:
        """Show how many replies are queued and how many requests were answered."""
        return state()

    @app.post(SCRIPT_PATH)
    async def push(http: Request) -> JSONResponse:
        """Queue replies: ``{"replies": [{"text": ..., "logprobs": true}]}``."""
        try:
            body = await http.json()
            replies = [
                Reply(str(item["text"]), bool(item.get("logprobs", True)))
                for item in body["replies"]
            ]
        except (ValueError, KeyError, TypeError) as error:
            message = f"expected {{'replies': [{{'text': ...}}]}}: {error}"
            return JSONResponse({"error": message}, status_code=engine.BAD_REQUEST)
        model.push(replies)
        return JSONResponse(state())

    @app.delete(SCRIPT_PATH)
    async def clear() -> dict[str, Any]:
        """Forget every queued reply."""
        model.clear()
        return state()


def create_app() -> tuple[FastAPI, ScriptedModel, ScriptedModel]:
    """Build the scripted engine, and the broken one under ``/broken``.

    Returns:
        The app, the healthy engine's model and the broken engine's model.
    """
    model = ScriptedModel()
    broken = ScriptedModel(no_logprobs_with_tools=True)
    app = engine.create_app(engine.Engine(model))
    add_remote_control(app, model)
    broken_app = engine.create_app(engine.Engine(broken))
    add_remote_control(broken_app, broken)
    app.mount(BROKEN_PATH, broken_app)
    return app, model, broken


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=engine.DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    """Serve the scripted engine (and the broken one) until Ctrl-C."""
    args = parse_args()
    sock = engine.claim_port(args.port)
    app, model, broken = create_app()
    port = sock.getsockname()[1]
    print(READY.format(port=port), flush=True)
    print(BROKEN_READY.format(port=port), flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        engine.serve(app, sock)
    print(STOPPED.format(requests=model.answered, broken=broken.answered), flush=True)


if __name__ == "__main__":
    main()
