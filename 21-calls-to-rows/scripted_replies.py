# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""What the scripted engine answers, and the remote control that tells it.

The scripted engine (``scripted_engine.py``) is chapter 20's real
``engine.py`` with a scripted model behind its seam instead of LFM2.5: no
weights, no MLX. Its replies are written here, as LFM2.5 writes them (a
thought, then ``</think>``, then a tool call between two special tokens or
an answer, then the end-of-turn token), and queued over HTTP before each
case. One switch per reply decides whether it comes back with its
log-probabilities or with every one of them null, as an engine that could
not score its tokens sends it.

The scripted model's tokenizer is the simplest there is: each LFM2.5
special token is one id, and every other character is its own id (its code
point plus 100). So a reply of 40 characters is about 40 tokens, and any
token sequence can be read back as text without a model (``decode``).

This module needs only Python's standard library, so the scripts that drive
the engine (``graph_lab.py``, ``reminder.py``) can run without FastAPI.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

# LFM2.5's special tokens, one id each (the scripted tokenizer's whole vocabulary
# of markers); every other character is PLAIN + its code point.
MARKERS = {
    "<|startoftext|>": 1,
    "<|im_start|>": 2,
    "<|im_end|>": 3,
    "<think>": 4,
    "</think>": 5,
    "<|tool_call_start|>": 6,
    "<|tool_call_end|>": 7,
}
PLAIN = 100
THINK_CLOSE = "</think>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
END_OF_TURN = "<|im_end|>"
DEFAULT_THOUGHT = "Look first."
SCRIPT_PATH = "/script"  # the scripted engine's remote control
HTTP_TIMEOUT_S = 30
NOT_FOUND = 404  # what a real engine answers on /script
# The task every script solves (clinic-easy-001: how many patients live in
# Murcia? 30), in the commands the scripted model writes.
TABLES_COMMAND = "sqlite3 data.db .tables"
MURCIA_COMMAND = (
    "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
)


class ScriptError(Exception):
    """Raised when the scripted engine cannot be reached or refuses a script."""


class NotScriptedError(ScriptError):
    """Raised when the engine answers but has no remote control: a real engine."""


@dataclass(frozen=True)
class Reply:
    """One reply the scripted model will give, in the order queued.

    Attributes:
        text: The reply as LFM2.5 writes it, end-of-turn token included.
        logprobs: False to serve it with every log-probability null.
    """

    text: str
    logprobs: bool = True


def encode(text: str) -> list[int]:
    """Cut text the scripted way: each marker one id, every other character one.

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
    """Write scripted ids back as text, markers kept.

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


def call_reply(
    command: str,
    tool: str = "bash",
    think: str = DEFAULT_THOUGHT,
    logprobs: bool = True,
) -> Reply:
    """Write a reply that calls one tool once, as LFM2.5 writes it.

    Args:
        command: The command.
        tool: The tool's name (Claude Code's is ``Bash``).
        think: The reasoning before the call.
        logprobs: False to serve the reply without log-probabilities.

    Returns:
        The reply.
    """
    call = f"{tool}(command={argument(command)})"
    text = f"{think}{THINK_CLOSE}{TOOL_CALL_START}[{call}]{TOOL_CALL_END}{END_OF_TURN}"
    return Reply(text, logprobs)


def answer_reply(text: str, think: str = "Done.", logprobs: bool = True) -> Reply:
    """Write a reply with no tool call: the model's final answer.

    Args:
        text: The visible answer.
        think: The reasoning before it.
        logprobs: False to serve the reply without log-probabilities.

    Returns:
        The reply.
    """
    return Reply(f"{think}{THINK_CLOSE}{text}{END_OF_TURN}", logprobs)


class ScriptClient:
    """The remote control of one scripted engine: queue replies, read its queue."""

    def __init__(self, engine_url: str) -> None:
        """Remember where the engine is.

        Args:
            engine_url: The engine's root, such as ``http://127.0.0.1:8119``
                (or ``.../broken`` for the broken one).
        """
        self.engine_url = engine_url.rstrip("/")

    def send(self, method: str, body: Any = None) -> dict[str, Any]:
        """Send one request to the remote control.

        Args:
            method: ``GET``, ``POST`` or ``DELETE``.
            body: The JSON body, or None.

        Returns:
            The answer: the queue's state.

        Raises:
            NotScriptedError: If the engine has no remote control.
            ScriptError: If the engine is unreachable or refuses the request.
        """
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.engine_url + SCRIPT_PATH,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as answer:
                return json.loads(answer.read())
        except urllib.error.HTTPError as error:
            if error.code == NOT_FOUND:  # a real engine has no remote control
                reason = f"{self.engine_url} is not the scripted engine (no /script)"
                raise NotScriptedError(reason) from error
            raise ScriptError(f"HTTP {error.code} from {self.engine_url}") from error
        except (urllib.error.URLError, OSError) as error:
            raise ScriptError(f"cannot reach {self.engine_url}: {error}") from error

    def push(self, replies: list[Reply]) -> dict[str, Any]:
        """Queue replies for the next requests, after any still queued.

        Args:
            replies: The replies, in order.

        Returns:
            The queue's state.
        """
        return self.send("POST", {"replies": [asdict(reply) for reply in replies]})

    def clear(self) -> dict[str, Any]:
        """Forget every queued reply.

        Returns:
            The queue's state.
        """
        return self.send("DELETE")

    def state(self) -> dict[str, Any]:
        """Read the queue's state.

        Returns:
            ``{"queued": n, "answered": n, ...}``.
        """
        return self.send("GET")
