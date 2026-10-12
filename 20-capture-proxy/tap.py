"""A tap: a recording relay between the capture proxy and the engine.

The proxy rewrites every request before the engine sees it. To show those
rewrites, ``probes.py`` opens its sessions with the engine's address
replaced by this relay's: the proxy sends each request here, the relay
passes its body to the real engine unchanged, hands back the engine's answer
(body and status) unchanged, and writes both down. To the proxy it is simply
another engine (the proxy even probes it first, as it probes every engine it
has not met), so what the tap records is exactly what the proxy sends and
receives.

The rest of the module turns a recorded exchange into one printed line, and
redoes the proxy's log-probability test (``validate_llm.probe_logprobs_mode``)
from the two requests it sent. It needs only Python's standard library, and
is imported by ``probes.py`` (it is not a script, so it has no PEP 723 header).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"
TAP_TIMEOUT_S = 600  # the engine may queue a request for minutes
BAD_GATEWAY = 502  # what the tap answers when the engine cannot be reached
MODELS_PATH = "/v1/models"
OK = 200
TOKEN_ID_PREFIX = "token_id:"  # vLLM's way to name a token by its id
CAPTURE_FIELDS = ("logprobs", "top_logprobs", "return_token_ids")
# The proxy's logprob probe: two temperatures, and the gap between the two
# likeliest first tokens at each.
SENTINEL = 100.0  # a "logprob" this large stands for minus infinity
PROBE_MIN_GAP = 0.5  # flatter than this, the ratio would be noise
TEMPERED_RATIO = 0.5  # the gap halves at twice the temperature
RAW_RATIO = 1.0  # raw scores ignore the temperature
PROCESSED, RAW, UNKNOWN = "processed", "raw", "unknown"
SHOWN_TEXT = 60  # characters of a message printed


@dataclass(frozen=True)
class Exchange:
    """One request the proxy sent the engine, and the answer, as the tap saw them.

    Attributes:
        method: ``GET`` or ``POST``.
        path: Such as ``/v1/chat/completions``.
        request: The JSON body sent, or None.
        status: The engine's HTTP status.
        response: The engine's JSON answer (None if it was not JSON).
    """

    method: str
    path: str
    request: dict[str, Any] | None
    status: int
    response: dict[str, Any] | None


class Tap:
    """A recording relay: passes requests to the engine and writes both sides down.

    Attributes:
        engine_url: The real engine, such as ``http://127.0.0.1:8100``.
        records: What passed through, oldest first.
        url: The tap's own root, once started.
    """

    def __init__(self, engine_url: str) -> None:
        """Get ready to relay to an engine.

        Args:
            engine_url: The real engine's root.
        """
        self.engine_url = engine_url.rstrip("/")
        self.records: list[Exchange] = []
        self.url = ""
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        """Listen on a free port of this machine.

        Returns:
            The tap's root URL.
        """
        if self.server is not None:  # already listening
            return self.url
        tap = self

        class Handler(BaseHTTPRequestHandler):
            """Relays one request to the engine."""

            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                """Relay a GET."""
                tap.relay(self, None)

            def do_POST(self) -> None:  # noqa: N802
                """Relay a POST, body and all."""
                length = int(self.headers.get("Content-Length") or 0)
                tap.relay(self, self.rfile.read(length))

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Print nothing per request."""

        self.server = ThreadingHTTPServer((HOST, 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{HOST}:{self.server.server_address[1]}"
        return self.url

    def relay(self, handler: BaseHTTPRequestHandler, body: bytes | None) -> None:
        """Pass one request to the engine, answer with its reply, and record both.

        Args:
            handler: The request being served.
            body: Its body, or None for a GET.
        """
        request = urllib.request.Request(
            self.engine_url + handler.path,
            data=body,
            method=handler.command,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TAP_TIMEOUT_S) as answer:
                status, raw = answer.status, answer.read()
        except urllib.error.HTTPError as error:  # the engine refused: pass it on
            status, raw = error.code, error.read()
        except OSError as error:  # nobody answered: say so, as a gateway does
            status = BAD_GATEWAY
            raw = json.dumps({"error": {"message": f"tap: {error}"}}).encode("utf-8")
        exchange = Exchange(
            handler.command, handler.path, json_or_none(body), status, json_or_none(raw)
        )
        with self.lock:
            self.records.append(exchange)
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(raw)))
            handler.end_headers()
            handler.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):  # the proxy hung up
            pass

    def take(self) -> list[Exchange]:
        """Hand over what was recorded, and start a fresh record.

        Returns:
            The exchanges since the last call.
        """
        with self.lock:
            taken, self.records = self.records, []
        return taken

    def stop(self) -> None:
        """Stop listening."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def json_or_none(raw: bytes | None) -> dict[str, Any] | None:
    """Parse a body as a JSON object, or give None.

    Args:
        raw: The body.

    Returns:
        The object, or None when the body is empty or not a JSON object.
    """
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def short(text: Any) -> str:
    """Cut a text to one short line, quoted.

    Args:
        text: The text (or None).

    Returns:
        It, cut to ``SHOWN_TEXT`` characters.
    """
    flat = " ".join(str(text or "").split())
    if len(flat) > SHOWN_TEXT:
        flat = flat[: SHOWN_TEXT - 3] + "..."
    return repr(flat)


def describe_request(record: Exchange) -> str:
    """Write one line about a request the proxy sent the engine.

    Args:
        record: The exchange.

    Returns:
        The route, the last message, the tools, and the fields that matter.
    """
    body = record.request or {}
    parts = [f"{record.method} {record.path}"]
    messages = body.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        parts.append(short(messages[-1].get("content")))
    if body.get("tools"):
        names = ", ".join(
            str((tool.get("function") or {}).get("name")) for tool in body["tools"]
        )
        parts.append(f"tools: {names}")
    for name in ("temperature", "max_tokens", *CAPTURE_FIELDS):
        if name in body:
            parts.append(f"{name} {json.dumps(body[name])}")
    return " · ".join(parts)


def logprob_gap(response: dict[str, Any] | None) -> float | None:
    """The gap between the two likeliest first tokens, as the proxy measures it.

    Args:
        response: The engine's answer.

    Returns:
        The gap, or None when there is nothing to measure.
    """
    choice = ((response or {}).get("choices") or [{}])[0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    tops = (entries[0].get("top_logprobs") or []) if entries else []
    values = sorted((entry["logprob"] for entry in tops), reverse=True)
    if len(values) < 2 or abs(values[1]) > SENTINEL:
        return None
    return values[0] - values[1]


def describe_response(record: Exchange) -> str:
    """Write one line about what the engine answered.

    Args:
        record: The exchange.

    Returns:
        Which receipts came back, a tool call, a gap.
    """
    response = record.response or {}
    if record.path == MODELS_PATH:
        return "served: " + ", ".join(m["id"] for m in response.get("data", []))
    if record.status != OK:
        return f"HTTP {record.status}: {short(response.get('error'))}"
    choice = (response.get("choices") or [{}])[0]
    parts = []
    for name, ids in (
        ("prompt_token_ids", response.get("prompt_token_ids")),
        ("token_ids", choice.get("token_ids")),
    ):
        parts.append(f"{name} ({len(ids)})" if ids else f"no {name}")
    entries = (choice.get("logprobs") or {}).get("content") or []
    if entries:
        token = str(entries[0].get("token"))
        named = f"{TOKEN_ID_PREFIX}<n>" if token.startswith(TOKEN_ID_PREFIX) else "text"
        parts.append(f"{len(entries)} logprobs, tokens named by {named}")
    calls = (choice.get("message") or {}).get("tool_calls") or []
    if calls:
        parts.append("a tool call: " + calls[0]["function"]["name"])
    gap = logprob_gap(response)
    if gap is not None and (record.request or {}).get("top_logprobs"):
        parts.append(f"gap between the two likeliest first tokens {gap:.4f}")
    return " · ".join(parts)


def gap_verdict(records: list[Exchange]) -> tuple[list[float], str]:
    """Redo the proxy's log-probability test from the requests it sent.

    At twice the temperature, tempered log-probabilities give half the gap
    between the two likeliest tokens (``(a - b) / T``); raw ones, computed
    before the temperature, give the same gap.

    Args:
        records: The probe's exchanges.

    Returns:
        The gaps, coolest temperature first, and ``processed``, ``raw`` or
        ``unknown``.
    """
    measured = []
    for record in records:
        body = record.request or {}
        gap = logprob_gap(record.response)
        if body.get("top_logprobs") and gap is not None:
            measured.append((body.get("temperature", 0.0), gap))
    gaps = [gap for _, gap in sorted(measured)]
    if len(gaps) < 2 or gaps[0] < PROBE_MIN_GAP:
        return gaps, UNKNOWN
    ratio = gaps[-1] / gaps[0]
    raw = abs(ratio - RAW_RATIO) < abs(ratio - TEMPERED_RATIO)
    return gaps, RAW if raw else PROCESSED
