# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""A client for the capture proxy's session API, after the project's CaptureClient.

The proxy records every model call of one rollout under one **session**.
Before a harness starts, the trainer asks the proxy for a session; the
session's id becomes the harness's API key, so each call the harness makes
arrives labelled with its rollout. Afterwards the trainer fetches what was
recorded and deletes the session. Three routes, as in ``mhrl/rollout.py``:

* ``POST /sessions``: what the rollout is for (``train`` or ``eval``), the
  sampling the proxy must enforce, the call budget, and free-form labels.
  The answer holds the session id and the capture level the proxy measured;
* ``GET /sessions/{id}/rollout``: every call, its exact tokens and
  log-probabilities, stitched into training rows, with ``trainable``;
* ``DELETE /sessions/{id}``: forget it.

The project's client uses httpx; this one needs only Python's standard
library. ``summarize`` reads the numbers this chapter prints from a rollout.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

PROXY_URL = "http://127.0.0.1:8200"  # the project's proxy_port
# The project's HTTP_TIMEOUT_S: the session routes answer at once (model calls
# go through the harness, which waits longer).
HTTP_TIMEOUT_S = 30
PURPOSE_TRAIN = "train"  # sampled at the training temperature, trainable
PURPOSE_EVAL = "eval"  # sampled greedily, never trained on
TRAIN_TEMPERATURE = 1.0  # the project's train_temperature
EVAL_TEMPERATURE = 0.0  # the project's eval_temperature: the likeliest token
MAX_MODEL_CALLS = 8  # the project's max_model_calls per rollout
INFO_PREFIX = "[INFO]"  # a finding that reports, not warns


class ProxyError(Exception):
    """Raised when the proxy refuses a request or cannot be reached.

    Attributes:
        status: The HTTP status of a refusal, or None when nobody answered.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        """Keep the message and the status.

        Args:
            message: What the proxy said, or why it could not be reached.
            status: The HTTP status, or None.
        """
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Session:
    """A session the proxy opened: the rollout's id and what it measured.

    Attributes:
        session_id: The id, which the harness uses as its API key.
        capture_level: ``tokens``, ``logprobs`` or ``text``.
        rollout_type: ``train`` or ``eval``.
        max_model_calls: The call budget the proxy enforces (0: none).
    """

    session_id: str
    capture_level: str
    rollout_type: str
    max_model_calls: int


@dataclass(frozen=True)
class Summary:
    """The numbers this chapter reads from one rollout document.

    Attributes:
        turns: Model calls recorded.
        roots: Conversations they form (1 when every call continues the last).
        sampled_tokens: Tokens the model wrote, summed over the calls.
        last_prompt_tokens: Tokens the last call read.
        rows: Training rows (one per path through the calls).
        trainable_tokens: Tokens a trainer may learn from (the model's own).
        trainable: The proxy's verdict: can this rollout be trained on?
        rollout_type: ``train`` or ``eval``.
        budget_stops: Calls the proxy answered itself, the budget spent.
        findings: The proxy's warnings about this rollout.
    """

    turns: int
    roots: int
    sampled_tokens: int
    last_prompt_tokens: int
    rows: int
    trainable_tokens: int
    trainable: bool
    rollout_type: str
    budget_stops: int
    findings: list[str]


class CaptureClient:
    """A thin client for the capture proxy's session API."""

    def __init__(self, proxy_url: str = PROXY_URL) -> None:
        """Remember where the proxy is.

        Args:
            proxy_url: The proxy's root, such as ``http://127.0.0.1:8200``.
        """
        self.proxy_url = proxy_url.rstrip("/")

    def send(self, method: str, path: str, body: Any = None) -> Any:
        """Send one request to the proxy and read its JSON answer.

        Args:
            method: ``GET``, ``POST`` or ``DELETE``.
            path: Such as ``/sessions``.
            body: The JSON body, or None.

        Returns:
            The answer, parsed.

        Raises:
            ProxyError: If the proxy refuses the request or cannot be reached.
        """
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.proxy_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:  # the proxy answered: a refusal
            raise ProxyError(refusal_text(error), error.code) from error
        except (urllib.error.URLError, OSError) as error:  # nobody answered
            raise ProxyError(f"cannot reach {self.proxy_url}: {error}") from error
        except ValueError as error:  # something answered, but not the proxy
            reason = f"{self.proxy_url}{path} did not answer in JSON: is it the proxy?"
            raise ProxyError(reason) from error

    def health(self) -> dict[str, Any]:
        """Ask the proxy what it is in front of and what it measured.

        Returns:
            ``/health``'s answer: the upstream engine, the capture level, ...
        """
        return self.send("GET", "/health")

    def open_session(
        self,
        purpose: str,
        labels: dict[str, Any],
        max_model_calls: int = MAX_MODEL_CALLS,
        llm_url: str | None = None,
        model: str | None = None,
    ) -> Session:
        """Mint a session id: the harness's API key for one rollout.

        Args:
            purpose: ``train`` (sampled at temperature 1.0, recorded for
                training) or ``eval`` (greedy).
            labels: Free-form labels kept with the session.
            max_model_calls: The most calls the proxy forwards for it.
            llm_url: An engine for this session alone; None for the proxy's.
            model: The model name that engine serves, with ``llm_url``.

        Returns:
            The session.

        Raises:
            ProxyError: If the proxy refuses the session.
        """
        body: dict[str, Any] = {
            "purpose": purpose,
            "max_model_calls": max_model_calls,
            "metadata": labels,
        }
        # The proxy enforces these on every call, whatever the harness asks.
        if purpose == PURPOSE_TRAIN:
            body["sampling"] = {"temperature": TRAIN_TEMPERATURE}
        else:
            body["eval_sampling"] = {"temperature": EVAL_TEMPERATURE}
        if llm_url:
            body["llm_url"] = llm_url
            body["model"] = model
        answer = self.send("POST", "/sessions", body)
        return Session(
            answer["session_id"],
            answer["capture_level"],
            answer["rollout_type"],
            answer["max_model_calls"],
        )

    def rollout(self, session_id: str) -> dict[str, Any]:
        """Fetch what the proxy recorded: stitched, masked, validated.

        Args:
            session_id: The session.

        Returns:
            The proxy's rollout document.
        """
        return self.send("GET", f"/sessions/{session_id}/rollout")

    def close(self, session_id: str) -> bool:
        """Delete a session to free the proxy's memory.

        Args:
            session_id: The session.

        Returns:
            True when the proxy had it.
        """
        return bool(self.send("DELETE", f"/sessions/{session_id}").get("deleted"))


def refusal_text(error: urllib.error.HTTPError) -> str:
    """Write a refusal as ``HTTP <status>: <the proxy's message>``.

    Args:
        error: The refusal.

    Returns:
        The text.
    """
    raw = error.read().decode("utf-8", errors="replace")
    try:
        found = json.loads(raw).get("error", raw)
    except (json.JSONDecodeError, AttributeError):
        found = raw
    message = found.get("message", found) if isinstance(found, dict) else found
    return f"HTTP {error.code}: {message}"


def summarize(document: dict[str, Any]) -> Summary:
    """Read the chapter's numbers from a rollout document.

    Args:
        document: ``GET /sessions/{id}/rollout``'s answer.

    Returns:
        The summary.
    """
    turns = [turn for turn in document["turns"] if not turn.get("discarded")]
    stats = document["stats"]
    return Summary(
        turns=stats["n_turns"],
        roots=stats["n_roots"],
        sampled_tokens=sum(turn["n_sampled"] for turn in turns),
        last_prompt_tokens=turns[-1]["n_prompt"] if turns else 0,
        rows=stats["n_sequences"],
        trainable_tokens=stats["n_trainable_tokens"],
        trainable=bool(document["trainable"]),
        rollout_type=document["rollout_type"],
        budget_stops=int(document.get("budget_stop_count") or 0),
        findings=[
            finding
            for finding in document["validation"]
            if not finding.startswith(INFO_PREFIX)
        ],
    )
