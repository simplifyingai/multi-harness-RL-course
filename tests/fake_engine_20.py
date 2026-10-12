"""Chapter 20's engine with a scripted model instead of LFM2.5: no MLX, no weights.

Chapter 18's tests put a scripted model (``FakeModel``) behind the engine's
seam. Chapter 20 needs the same, with two differences: the capture proxy in
front of the engine sends requests of its own (its probe at start-up and
whenever a session names an engine), so the model must answer any number of
requests, and ``start.sh`` must be able to start the engine as a process.

So ``ScriptedModel`` answers from a queue of replies the test pushes, and
when the queue is empty it answers by default: a ``bash`` call when the
prompt offers tools, else ``ok``. Its tokenizer gives each LFM2.5 marker one
id and every other character its own; its chat template lays a chat out the
way LFM2.5's does; its prompt cache is the list of tokens it has read. The
likeliest alternatives at each position follow the temperature (processed
log-probabilities) unless ``Sampling.raw_logprobs`` asks for the raw ones.

Run as a script, it serves the chapter's real ``engine.py`` (its request
checks, receipts, restore, caches and both switches) with this model:
``python fake_engine_20.py --port 8119 [--no-token-fields] [--raw-logprobs]``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHAPTER = Path(__file__).resolve().parent.parent / "20-capture-proxy"
if str(CHAPTER) not in sys.path:  # run as a script: find the chapter's modules
    sys.path.insert(0, str(CHAPTER))

import engine  # noqa: E402
import toolcalls  # noqa: E402
from policy import (  # noqa: E402
    FINISH_LENGTH,
    FINISH_STOP,
    GREEDY,
    RenderError,
    Sampled,
    Sampling,
)

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
PLAIN = 100  # every other character is PLAIN + its code point
START, END = toolcalls.TOOL_CALL_START, toolcalls.TOOL_CALL_END
EOT = toolcalls.END_OF_TURN
TOOLS_MARK = "List of tools: "  # where the fake template prints the tools
# A small vocabulary of scores for the alternatives (tokens 200, 201, ...):
# the gap between the two likeliest is 2.0 at temperature 1, 1.0 at 2.
FAKE_SCORES = [4.0, 2.0, 1.5, 1.0, 0.0, -1.0, -2.0]
FIRST_ALTERNATIVE = 200
LOGPROB_STEP = -0.125
DEFAULT_COMMAND = "ls"
DEFAULT_ANSWER = "ok"
READY = "engine ready · scripted model (no MLX) · 127.0.0.1:{port}"
STOPPED = "engine stopped · {requests} requests"


def encode(text: str) -> list[int]:
    """Cut text the fake way: each marker one id, every other character one.

    Args:
        text: The text.

    Returns:
        The ids.
    """
    ids, index = [], 0
    while index < len(text):
        for piece, token in MARKERS.items():
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
    names = {token: piece for piece, token in MARKERS.items()}
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
            text += f"{START}[{', '.join(written)}]{END}"
        text += "<|im_end|>\n"
    return text + "<|im_start|>assistant\n<think>"


def call_reply(command: str, think: str = "Look first.") -> str:
    """Write a reply that calls bash once, as LFM2.5 writes it.

    Args:
        command: The command.
        think: The reasoning.

    Returns:
        The reply text, end of turn included.
    """
    return f"{think}</think>{START}[bash(command={argument(command)})]{END}{EOT}"


def answer_reply(text: str) -> str:
    """Write a reply with no tool call.

    Args:
        text: The visible text.

    Returns:
        The reply text, end of turn included.
    """
    return f"Done.</think>{text}{EOT}"


@dataclass
class FakeCache:
    """A prompt cache that is simply the tokens it has read."""

    tokens: list[int] = field(default_factory=list)


class ScriptedModel:
    """A scripted model behind the engine's seam, for any number of requests.

    Attributes:
        queue: Replies to give next, in order (text, end of turn included).
        seen: The sampling settings of every request, as the engine passed
            them to the model.
    """

    eos_ids = frozenset({EOS})

    def __init__(self) -> None:
        """Start with no queued replies."""
        self.queue: list[str] = []
        self.seen: list[Sampling] = []
        self.lock = threading.Lock()

    def push(self, *replies: str) -> None:
        """Queue replies for the next requests.

        Args:
            *replies: Each reply's text.
        """
        with self.lock:
            self.queue.extend(replies)

    def render(self, messages: list[dict[str, Any]], tools: list | None) -> str:
        """Render with the fake template."""
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

    def alternatives(self, sampling: Sampling) -> list[tuple[int, float]]:
        """The likeliest fake tokens: tempered, or raw when asked.

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

    def next_reply(self, prompt: str) -> str:
        """The queued reply, or the default one for this prompt.

        Args:
            prompt: The whole prompt, as text.

        Returns:
            The reply text.
        """
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
        if TOOLS_MARK in prompt:
            return call_reply(DEFAULT_COMMAND)
        return answer_reply(DEFAULT_ANSWER)

    def sample(
        self, new_ids: list[int], cache: FakeCache, sampling: Sampling
    ) -> Sampled:
        """Hand back the next reply, as ids with log-probabilities."""
        self.seen.append(sampling)
        whole = encode(self.next_reply(decode(cache.tokens + list(new_ids))))
        ids = whole[: sampling.max_tokens]
        cut = len(whole) > len(ids) or not ids or ids[-1] not in self.eos_ids
        logprobs = [LOGPROB_STEP * (i % 4 + 1) for i in range(len(ids))]
        tops = []
        if sampling.top_logprobs:
            tops = [self.alternatives(sampling) for _ in ids]
        cache.tokens += list(new_ids) + ids
        finish = FINISH_LENGTH if cut else FINISH_STOP
        return Sampled(ids, logprobs, tops, finish, 0.001)


def parse_args() -> argparse.Namespace:
    """Read engine.py's command line (the flags the scripted engine honours).

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description="chapter 20's scripted engine")
    parser.add_argument("--port", type=int, default=engine.DEFAULT_PORT)
    parser.add_argument("--no-token-fields", action="store_true")
    parser.add_argument("--raw-logprobs", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Serve the chapter's engine with the scripted model until Ctrl-C."""
    args = parse_args()
    sock = engine.claim_port(args.port)
    core = engine.Engine(
        ScriptedModel(),
        token_fields=not args.no_token_fields,
        raw_logprobs=args.raw_logprobs,
    )
    print(READY.format(port=sock.getsockname()[1]), flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        engine.serve(engine.create_app(core), sock)
    print(STOPPED.format(requests=core.stats.requests), flush=True)


if __name__ == "__main__":
    main()
