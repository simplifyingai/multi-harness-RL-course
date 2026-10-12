# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One Claude Code run through the proxy, kept on disk: the run folders.

``capture.py`` records the one run this chapter breaks, ``breakages.py``
replays it once per case, and ``crosscheck.py`` reads any of them. Every
run plays out in the same **stage** folder (``runs/stage``: Claude Code's
working folder ``task/`` and its private ``state/``), so a replay sees the
very paths the capture saw: a reply that names an absolute path still finds
it. Each run's account is then kept in a folder of its own,
``COURSE_HOME/22-silent-capture-failures/runs/<name>``:

* ``run.json``: what ran (the task, the case, the engine's tokenizer, the
  Claude Code version, the stage) and how it ended (exit, seconds, grade);
* ``rollout.json``: the proxy's rollout document, every row and message;
* ``sent.jsonl``: each request as Claude Code sent it (the relay's record);
* ``forwarded.jsonl``: each request as it reached the proxy, written only
  when the relay changed some;
* ``stream.jsonl`` and ``stderr.txt``: Claude Code's own log, and its errors.

``captured_replies`` reads a capture's replies back out of its rows: the
exact token ids each call sampled and their log-probabilities, which is what
the scripted engine needs to give the same replies again. Only Python's
standard library is needed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from capture_client import CaptureClient, ProxyError
from claude_code import STDERR_FILE, STREAM_FILE, ClaudeRun, RelayedRequest
from grader import Grade
from lfm_files import chapter_home, short_path, tilde
from scripted_replies import ScriptClient, ScriptError

RUNS_FOLDER = "runs"
STAGE = "stage"  # where every run plays out
CAPTURE = "capture"  # the run the chapter breaks
# The runs breakages.py makes from the capture, one per case, in its order.
CASE_RUNS = ("replay", "note", "summary", "no-logprobs")
RUN_FILE = "run.json"
ROLLOUT_FILE = "rollout.json"
SENT_FILE = "sent.jsonl"
FORWARDED_FILE = "forwarded.jsonl"
MESSAGES_PATH = "/v1/messages"  # the model calls (not .../count_tokens)
DIFF_CHARS = 70  # how much of a changed prompt is shown
ELLIPSIS = "…"


class CaptureError(Exception):
    """Raised when a run folder is missing a file or cannot be replayed."""


@dataclass(frozen=True)
class CapturedReply:
    """One call's reply, as the engine sampled it.

    Attributes:
        index: The call's number in the proxy's record, from 0.
        ids: The exact token ids sampled.
        logprobs: Their log-probabilities, or None when the engine sent none.
        n_tools: Tools the call offered (0: a side call).
    """

    index: int
    ids: list[int]
    logprobs: list[float | None] | None
    n_tools: int


def text_of(content: Any) -> str:
    """Flatten a message's content (text, or a list of blocks) into text.

    Args:
        content: A string, or Anthropic or OpenAI content blocks.

    Returns:
        The text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text_of(block.get("text") or block.get("content") or "")
            for block in content
            if isinstance(block, dict)
        )
    return ""


def cut_text(text: str, limit: int) -> str:
    """Cut text to a limit, marking the cut (for data: nothing else changes).

    Args:
        text: Any text.
        limit: The most characters kept.

    Returns:
        The text, or its start and an ellipsis.
    """
    return text if len(text) <= limit else text[: limit - 1] + ELLIPSIS


def clip_text(text: str, limit: int = DIFF_CHARS) -> str:
    """Cut text to a limit for printing, the home folder written ``~``.

    Args:
        text: Any text.
        limit: The most characters kept.

    Returns:
        The text, or its start and an ellipsis.
    """
    return cut_text(tilde(text), limit)


def system_change(before: str, after: str) -> str:
    """Say how one call's system prompt differs from the call before's.

    Args:
        before: The earlier call's system message.
        after: This call's.

    Returns:
        A phrase that follows "its system prompt": unchanged, grown at its end
        (and by what), or where it differs.
    """
    if before == after:
        return "is the same as the call before's"
    if after.startswith(before):
        added = clip_text(json.dumps(after[len(before) :]))
        return f"is the call before's, plus {added} at its end"
    same = 0
    while same < min(len(before), len(after)) and before[same] == after[same]:
        same += 1
    return f"differs from the call before's at character {same:,}"


def runs_folder() -> Path:
    """Where this chapter keeps its runs.

    Returns:
        ``COURSE_HOME/22-silent-capture-failures/runs``.
    """
    return chapter_home() / RUNS_FOLDER


def stage_folder() -> Path:
    """The one folder every run plays out in.

    Returns:
        ``.../runs/stage``.
    """
    return runs_folder() / STAGE


def named_run(name: str) -> Path:
    """Find a run folder by its name, or take a path as it is.

    Args:
        name: Such as ``capture`` or ``note``, or a folder.

    Returns:
        This chapter's run of that name when there is one, else the path as
        given (it may not exist).
    """
    folder = runs_folder() / name
    return folder if folder.is_dir() else Path(name).expanduser()


def model_calls(requests: list[RelayedRequest]) -> list[dict[str, Any]]:
    """Keep the requests that asked the model for a reply.

    Args:
        requests: What a relay kept.

    Returns:
        The bodies sent to ``/v1/messages`` with a list of messages.
    """
    return [
        request.body
        for request in requests
        if request.path.split("?", 1)[0] == MESSAGES_PATH
        and isinstance(request.body.get("messages"), list)
    ]


def unreadable(path: Path, error: Exception) -> str:
    """Say a file cannot be read, without printing the home folder.

    Args:
        path: The file.
        error: Why.

    Returns:
        The message.
    """
    return f"cannot read {short_path(path)}: {tilde(str(error))}"


def tidy_up(
    engine: ScriptClient | None, client: CaptureClient, session_id: str
) -> list[str]:
    """Empty the scripted engine's queue and close the session, after a run.

    A reply left queued would answer the next run's first call, so a failure
    here is reported, not hidden.

    Args:
        engine: The scripted engine's remote control, or None for a real engine.
        client: The proxy's session client.
        session_id: The run's session.

    Returns:
        A note for each step that failed (none when both worked).
    """
    notes = []
    if engine is not None:
        try:
            engine.clear()
        except ScriptError as error:
            notes.append(f"the engine's queue was not emptied: {error}")
    try:
        client.close(session_id)
    except ProxyError as error:
        notes.append(f"the session was not closed: {error}")
    return notes


def write_requests(path: Path, requests: list[RelayedRequest]) -> None:
    """Write relayed requests as JSON lines: each one's path and body.

    Args:
        path: The file.
        requests: The requests, in order.
    """
    lines = "".join(
        json.dumps({"path": request.path, "body": request.body}) + "\n"
        for request in requests
    )
    path.write_text(lines, encoding="utf-8")


def read_requests(path: Path) -> list[RelayedRequest]:
    """Read relayed requests back.

    Args:
        path: A file ``write_requests`` wrote.

    Returns:
        The requests, in order.

    Raises:
        CaptureError: If it cannot be read.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        found = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as error:
        raise CaptureError(unreadable(path, error)) from error
    return [RelayedRequest(item["path"], item["body"]) for item in found]


def save_run(
    folder: Path,
    meta: dict[str, Any],
    document: dict[str, Any],
    sent: list[RelayedRequest],
    forwarded: list[RelayedRequest],
    run: ClaudeRun,
    verdict: Grade,
) -> None:
    """Keep one run's account in its own folder (emptied first).

    Args:
        folder: The run's folder.
        meta: What ran: task, case, tokenizer, versions.
        document: The proxy's rollout document, every row and message.
        sent: The requests as Claude Code sent them.
        forwarded: The same requests as they reached the proxy.
        run: The finished Claude Code run (its log is in the stage).
        verdict: The grade of answer.txt.

    Raises:
        OSError: If the folder or a file cannot be written.
    """
    if folder.exists():
        shutil.rmtree(folder)  # a failure here is shown, not hidden
    folder.mkdir(parents=True)
    record = {
        **meta,
        "exit_code": run.exit_code,
        "seconds": round(run.seconds, 1),
        "grade": asdict(verdict),
    }
    (folder / RUN_FILE).write_text(json.dumps(record, indent=1) + "\n", "utf-8")
    text = json.dumps(document, indent=1) + "\n"
    (folder / ROLLOUT_FILE).write_text(text, encoding="utf-8")
    write_requests(folder / SENT_FILE, sent)
    if [r.body for r in sent] != [r.body for r in forwarded]:  # the relay rewrote
        write_requests(folder / FORWARDED_FILE, forwarded)
    for name in (STREAM_FILE, STDERR_FILE):
        shutil.copy(run.folder / name, folder / name)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from a run folder.

    Args:
        path: The file.

    Returns:
        The object.

    Raises:
        CaptureError: If it is missing or not a JSON object.
    """
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CaptureError(unreadable(path, error)) from error
    if not isinstance(found, dict):
        raise CaptureError(f"{short_path(path)} does not hold a JSON object")
    return found


def read_run(folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a run folder's record and the proxy's document.

    Args:
        folder: The run's folder.

    Returns:
        ``run.json`` and ``rollout.json``.

    Raises:
        CaptureError: If either is missing, or the document has no turns.
    """
    meta = read_json(folder / RUN_FILE)
    document = read_json(folder / ROLLOUT_FILE)
    if "turns" not in document or "sequences" not in document:
        raise CaptureError(
            f"{short_path(folder / ROLLOUT_FILE)} is not a rollout document"
        )
    return meta, document


def reply_ids(document: dict[str, Any], turn: dict[str, Any]) -> list[int]:
    """Read one call's sampled tokens out of a row that holds it.

    A row is built along its path, so a call's prompt is the row's first
    ``n_prompt`` tokens and its reply the ``n_sampled`` after them.

    Args:
        document: A rollout document with every row (discarded ones too).
        turn: One of its turns.

    Returns:
        The reply's token ids.

    Raises:
        CaptureError: If no row holds the call, or its tokens are not there.
    """
    start, size = turn["n_prompt"], turn["n_sampled"]
    for row in document["sequences"]:
        if turn["node_id"] in row["node_ids"]:
            ids = row["input_ids"][start : start + size]
            if len(ids) == size:
                return ids
    raise CaptureError(f"no row holds the {size} tokens of call {turn['index'] + 1}")


def captured_replies(document: dict[str, Any]) -> list[CapturedReply]:
    """Read every call's reply out of a capture, in the order they were made.

    Args:
        document: The capture's rollout document, every row included.

    Returns:
        One reply per call: its exact ids and log-probabilities.

    Raises:
        CaptureError: If a call's tokens cannot be found, its log-probabilities
            do not line up with them, or the document is not a rollout's.
    """
    replies = []
    try:
        for turn in sorted(document["turns"], key=lambda t: t["index"]):
            ids = reply_ids(document, turn)
            values = turn.get("sampled_logprobs")
            scored = values if values and any(v is not None for v in values) else None
            if scored is not None and len(scored) != len(ids):
                raise CaptureError(
                    f"call {turn['index'] + 1}: {len(ids)} tokens but "
                    f"{len(scored)} log-probabilities"
                )
            replies.append(CapturedReply(turn["index"], ids, scored, turn["n_tools"]))
    except (KeyError, TypeError) as error:
        raise CaptureError(f"not a rollout document: missing {error}") from error
    return replies
