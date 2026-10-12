# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi==0.142.2",
#     "tokenizers==0.23.2",
#     "uvicorn==0.54.0",
# ]
# ///
"""Chapter 20's engine with a scripted model instead of LFM2.5: no weights, no MLX.

This chapter replays one captured run with its replies fixed, token for
token, while Claude Code runs again. So the engine here is chapter 20's
real ``engine.py`` (its request checks, its receipts, its restore and its
prompt caches) with a scripted model behind its seam, plus a remote
control:

* ``POST /script`` ``{"replies": [...]}`` queues the next replies; ``GET
  /script`` shows the queue and the tokenizer, ``DELETE /script`` empties
  the queue. A reply is ``{"text": ..., "logprobs": true}`` (written: the
  engine cuts the text into tokens and makes up their log-probabilities) or
  ``{"ids": [...], "logprobs": [...]}`` (replayed: exactly these tokens,
  with exactly these log-probabilities). ``"logprobs": false`` serves either
  kind with every log-probability null, as from an engine that could not
  score its tokens. With the queue empty, the model answers by default: a
  ``bash ls`` call when the request offers tools, else ``ok`` (the proxy's
  own probes are answered that way);
* ``--tokenizer`` picks how text becomes tokens: ``scripted`` (one id per
  character, each LFM2.5 special token one id) or ``lfm2.5`` (LFM2.5's own
  ``tokenizer.json``, checked by its fingerprint; no weights are read). A
  capture the model made holds LFM2.5 token ids, so replaying it needs the
  second (``start.sh replay``).

Run ``uv run scripted_engine.py --port 8119`` (``start.sh scripted`` does).
It needs FastAPI and uvicorn, as ``engine.py`` does, and ``tokenizers`` for
``--tokenizer lfm2.5``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import engine
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from lfm_files import TokenizerMissingError, lfm_tokenizer_file, short_path
from policy import FINISH_LENGTH, FINISH_STOP, GREEDY, RenderError, Sampled, Sampling
from scripted_replies import (
    END_OF_TURN,
    LFM_TOKENIZER,
    SCRIPT_PATH,
    SCRIPTED_TOKENIZER,
    TOKENIZER_FIELD,
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
REPLIES_FIELD, TEXT_FIELD, IDS_FIELD = "replies", "text", "ids"
LOGPROBS_FIELD = "logprobs"
PUSH_SHAPE = (
    '{"replies": [{"text": "<reply>", "logprobs": true} or '
    '{"ids": [1, 2], "logprobs": [-0.5, null]}]}'
)
TOKENIZERS = (SCRIPTED_TOKENIZER, LFM_TOKENIZER)

# A small vocabulary of scores for the alternatives (tokens 200, 201, ...):
# the gap between the two likeliest is 2.0 at temperature 1, 1.0 at 2.
FAKE_SCORES = [4.0, 2.0, 1.5, 1.0, 0.0, -1.0, -2.0]
FIRST_ALTERNATIVE = 200
LOGPROB_STEP = -0.125  # the scripted log-probabilities: -0.125, -0.25, ...
LOGPROB_CYCLE = 4
DEFAULT_COMMAND = "ls"
DEFAULT_ANSWER = "ok"
READY = "engine ready · scripted model (no MLX) · {tokenizer} · 127.0.0.1:{port}"
STOPPED = "engine stopped · {requests} requests"


class Cutter(Protocol):
    """How the scripted model turns text into token ids and back."""

    name: str
    eos_ids: frozenset[int]

    def encode(self, text: str) -> list[int]:
        """Cut text into ids."""

    def decode(self, token_ids: list[int]) -> str:
        """Write ids back as text, special tokens kept."""


class ScriptedCutter:
    """One id per character, each LFM2.5 special token one id (no files)."""

    name = SCRIPTED_TOKENIZER
    eos_ids = frozenset({EOS})

    def encode(self, text: str) -> list[int]:
        """Cut text the scripted way.

        Args:
            text: Any text.

        Returns:
            The ids.
        """
        return encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode the scripted way.

        Args:
            token_ids: The ids.

        Returns:
            The text.
        """
        return decode(token_ids)


class LfmCutter:
    """LFM2.5's own tokenizer.json, read with ``tokenizers`` (no weights)."""

    name = LFM_TOKENIZER

    def __init__(self, path: Path) -> None:
        """Load the tokenizer file.

        Args:
            path: LFM2.5's ``tokenizer.json``.

        Raises:
            ValueError: If it has no end-of-turn token.
        """
        from tokenizers import Tokenizer  # only this tokenizer needs the package

        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path))
        end = self.tokenizer.token_to_id(END_OF_TURN)
        if end is None:
            raise ValueError(f"{path} has no {END_OF_TURN} token")
        self.eos_ids = frozenset({end})

    def encode(self, text: str) -> list[int]:
        """Cut text as LFM2.5 does, special tokens included, nothing added.

        Args:
            text: Any text.

        Returns:
            The ids.
        """
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int]) -> str:
        """Write LFM2.5 ids back as text, special tokens kept.

        Args:
            token_ids: The ids.

        Returns:
            The text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def make_cutter(name: str) -> Cutter:
    """Build the tokenizer the engine was started with.

    Args:
        name: ``scripted`` or ``lfm2.5``.

    Returns:
        The cutter.

    Raises:
        ValueError: For any other name.
    """
    if name == SCRIPTED_TOKENIZER:
        return ScriptedCutter()
    if name == LFM_TOKENIZER:
        return LfmCutter(lfm_tokenizer_file())  # TokenizerMissingError if absent
    raise ValueError(f"no tokenizer named {name!r}: choose from {TOKENIZERS}")


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
        cutter: How text becomes ids (and back).
        eos_ids: The cutter's end-of-turn ids (the engine reads them here).
        queue: Replies to give next, in order.
        answered: Requests sampled so far.
        offered_tools: Whether the request being answered offered tools. The
            engine renders a request and samples its reply one after the
            other on its one worker thread, so ``render`` can tell ``sample``.
    """

    def __init__(self, cutter: Cutter | None = None) -> None:
        """Start with no queued replies.

        Args:
            cutter: The tokenizer; the scripted one when None.
        """
        self.cutter = cutter or ScriptedCutter()
        self.eos_ids = self.cutter.eos_ids
        self.queue: list[Reply] = []
        self.answered = 0
        self.offered_tools = False
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
        """Render with the scripted template, noting whether tools are offered.

        Args:
            messages: The conversation.
            tools: The tools offered, or None.

        Returns:
            The prompt.
        """
        self.offered_tools = bool(tools)
        return scripted_template(messages, tools)

    def encode(self, text: str) -> list[int]:
        """Encode with the engine's tokenizer.

        Args:
            text: Any text.

        Returns:
            The ids.
        """
        return self.cutter.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode with the engine's tokenizer.

        Args:
            token_ids: The ids.

        Returns:
            The text.
        """
        return self.cutter.decode(token_ids)

    def new_cache(self) -> ScriptedCache:
        """Make an empty cache.

        Returns:
            A cache holding no tokens.
        """
        return ScriptedCache()

    def cache_size(self, cache: ScriptedCache) -> int:
        """Count the tokens a cache holds.

        Args:
            cache: The cache.

        Returns:
            Its number of tokens.
        """
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

    def next_reply(self) -> Reply:
        """The queued reply, or the default one for this request.

        Returns:
            The reply.
        """
        with self.lock:
            self.answered += 1
            if self.queue:
                return self.queue.pop(0)
        if self.offered_tools:
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
        reply = self.next_reply()
        whole = list(reply.ids) if reply.ids is not None else self.encode(reply.text)
        ids = whole[: sampling.max_tokens]
        cut = len(whole) > len(ids) or not ids or ids[-1] not in self.eos_ids
        if isinstance(reply.logprobs, list):  # replayed: the engine's own numbers
            logprobs = list(reply.logprobs[: len(ids)])
        elif reply.logprobs:
            logprobs = [LOGPROB_STEP * (i % LOGPROB_CYCLE + 1) for i in range(len(ids))]
        else:
            # A reply the engine "could not score": the tokens are real, and
            # every number that training would need is missing.
            logprobs = [None] * len(ids)
        tops = []
        if sampling.top_logprobs:
            tops = [self.alternatives(sampling) for _ in ids]
        cache.tokens += list(new_ids) + ids
        finish = FINISH_LENGTH if cut else FINISH_STOP
        return Sampled(ids, logprobs, tops, finish, 0.001)


def is_logprob(value: Any) -> bool:
    """Say whether a value can be a log-probability (a number) or a gap (null).

    Args:
        value: One entry of a reply's ``logprobs`` list.

    Returns:
        True for None or a real number (not a boolean).
    """
    return value is None or (
        isinstance(value, int | float) and not isinstance(value, bool)
    )


def read_reply(item: Any) -> Reply:
    """Check one queued reply and read it.

    Args:
        item: One entry of the script's ``replies``.

    Returns:
        The reply.

    Raises:
        ValueError: If it is neither a written nor a replayed reply.
    """
    if not isinstance(item, dict):
        raise ValueError(f"a reply must be an object: {item!r}")
    scored = item.get(LOGPROBS_FIELD, True)
    if IDS_FIELD in item:
        ids = item[IDS_FIELD]
        good_ids = isinstance(ids, list) and all(
            isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids
        )
        if not good_ids:
            raise ValueError(f"ids must be token ids: {item!r}")
        if isinstance(scored, list):
            if len(scored) != len(ids) or not all(map(is_logprob, scored)):
                raise ValueError(f"one log-probability (or null) per id: {item!r}")
            return Reply("", [None if v is None else float(v) for v in scored], ids)
        if not isinstance(scored, bool):
            raise ValueError(f"logprobs must be a list or true/false: {item!r}")
        return Reply("", scored, ids)
    text = item.get(TEXT_FIELD)
    if not isinstance(text, str) or not isinstance(scored, bool):
        raise ValueError(f"each reply needs a text and a true/false: {item!r}")
    return Reply(text, scored)


def read_replies(body: Any) -> list[Reply]:
    """Check a script and read its replies.

    Args:
        body: The request's JSON body.

    Returns:
        The replies, in order.

    Raises:
        ValueError: If the body has any other shape.
    """
    items = body.get(REPLIES_FIELD) if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"no {REPLIES_FIELD!r} list")
    return [read_reply(item) for item in items]


def add_remote_control(app: FastAPI, model: ScriptedModel) -> None:
    """Give an engine's app the ``/script`` routes that queue its replies.

    Args:
        app: The engine's FastAPI app.
        model: Its scripted model.
    """

    def state() -> dict[str, Any]:
        """Read the queue's length, the requests answered and the tokenizer.

        Returns:
            ``{"queued": n, "answered": n, "tokenizer": name}``.
        """
        with model.lock:
            return {
                "queued": len(model.queue),
                "answered": model.answered,
                TOKENIZER_FIELD: model.cutter.name,
            }

    @app.get(SCRIPT_PATH)
    async def show() -> dict[str, Any]:
        """Show the queue's state.

        Returns:
            The queue's state.
        """
        return state()

    @app.post(SCRIPT_PATH)
    async def push(http: Request) -> JSONResponse:
        """Queue replies, after any still queued.

        Args:
            http: The request: ``{"replies": [...]}``.

        Returns:
            The queue's state, or a 400 naming the expected shape.
        """
        try:
            replies = read_replies(await http.json())
        except ValueError as error:
            message = f"expected {PUSH_SHAPE}: {error}"
            return JSONResponse({"error": message}, status_code=engine.BAD_REQUEST)
        model.push(replies)
        return JSONResponse(state())

    @app.delete(SCRIPT_PATH)
    async def clear() -> dict[str, Any]:
        """Forget every queued reply.

        Returns:
            The queue's state.
        """
        model.clear()
        return state()


def create_app(cutter: Cutter | None = None) -> tuple[FastAPI, ScriptedModel]:
    """Build the scripted engine.

    Args:
        cutter: Its tokenizer; the scripted one when None.

    Returns:
        The app and its model.
    """
    model = ScriptedModel(cutter)
    app = engine.create_app(engine.Engine(model))
    add_remote_control(app, model)
    return app, model


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=engine.DEFAULT_PORT)
    parser.add_argument("--tokenizer", default=SCRIPTED_TOKENIZER, choices=TOKENIZERS)
    return parser.parse_args()


def main() -> None:
    """Serve the scripted engine until Ctrl-C.

    Raises:
        SystemExit: If the tokenizer asked for is not on disk.
    """
    args = parse_args()
    try:
        cutter = make_cutter(args.tokenizer)
    except (TokenizerMissingError, ValueError) as error:
        raise SystemExit(str(error)) from error
    except ImportError as error:  # run without uv, so without its header's packages
        raise SystemExit(f"{error}: run it with `uv run scripted_engine.py`") from error
    sock = engine.claim_port(args.port)
    app, model = create_app(cutter)
    port = sock.getsockname()[1]
    name = cutter.name
    if isinstance(cutter, LfmCutter):
        name += f" ({short_path(cutter.path)}, no weights)"
    print(READY.format(tokenizer=name, port=port), flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        engine.serve(app, sock)
    print(STOPPED.format(requests=model.answered), flush=True)


if __name__ == "__main__":
    main()
