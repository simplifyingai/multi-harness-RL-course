# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Our tiny harness, recorded by the capture proxy it does not know is there.

The harness is chapter 17's, in chapter 18's HTTP version
(``tiny_harness_http.py``, copied unchanged): it knows a
base URL and an API key, nothing else. Here the base URL is the proxy's and
the API key is a session id the proxy just minted, so every call it makes
lands in that session's record. Afterwards the script fetches the record
and prints what training would get from it.

Steps, as the project's ``mhrl/rollout.py`` runs them:

1. ``POST /sessions``: a training session (temperature 1.0, 8 calls) or,
   with ``--purpose eval``, an evaluation one (temperature 0);
2. the harness works on a question, base URL ``<proxy>/v1``, key = the
   session id;
3. ``GET /sessions/{id}/rollout``: the record, saved under
   ``COURSE_HOME/20-capture-proxy/rollouts/session/``;
4. ``DELETE /sessions/{id}``.

Run ``./start.sh`` first, then ``uv run session_demo.py`` (``--task <id>``
for another question). It needs only Python's standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from capture_client import (
    PROXY_URL,
    PURPOSE_EVAL,
    PURPOSE_TRAIN,
    CaptureClient,
    ProxyError,
    summarize,
)
from lfm_files import rollout_path, short_path, work_folder
from tasks import find_task, prepare_workdir
from tiny_harness_http import (
    DEFAULT_TASK,
    EngineError,
    Rollout,
    http_chat,
    run_rollout,
    show_call,
)

SCRIPT = "session_demo.py"
LABEL = "session"  # this script's work and rollout folders
OPENAI_ROOT = "/v1"  # OpenAI-style clients add /chat/completions to it
INDENT = "  "


def open_or_explain(client: CaptureClient, purpose: str, task_id: str) -> str:
    """Open the session, or stop with the proxy's reason.

    Args:
        client: The proxy's session client.
        purpose: ``train`` or ``eval``.
        task_id: The question, kept as a label.

    Returns:
        The session id.

    Raises:
        SystemExit: If the proxy refuses the session or cannot be reached.
    """
    try:
        session = client.open_session(purpose, {"task_id": task_id, "harness": "tiny"})
    except ProxyError as error:
        if error.status is None:  # nobody answered
            raise SystemExit(f"{error}: start it with ./start.sh") from error
        raise SystemExit(f"the proxy refused a {purpose} session: {error}") from error
    print(
        f"{INDENT}POST /sessions · purpose {purpose} · {session.max_model_calls} "
        f"calls -> {session.session_id}"
    )
    print(
        f"{INDENT}capture level {session.capture_level} · rollout type "
        f"{session.rollout_type}"
    )
    return session.session_id


def print_grade(rollout: Rollout) -> None:
    """Print how the harness's attempt ended.

    Args:
        rollout: The attempt.
    """
    verdict = "correct" if rollout.grade.correct else "wrong"
    print(f"{INDENT}stop: {rollout.stop}")
    print(
        f"{INDENT}grade: {verdict} ({rollout.grade.reason}; answer.txt: "
        f"{rollout.grade.submitted!r}) · {len(rollout.calls)} calls"
    )


def print_record(document: dict[str, Any]) -> None:
    """Print the proxy's record of the rollout: per call, then in total.

    Args:
        document: ``GET /sessions/{id}/rollout``'s answer.
    """
    print(f"{INDENT}call  read  wrote  tools  finish")
    for turn in document["turns"]:
        # A retried call's abandoned try is kept, marked, and left out of the
        # totals below (chapter 21).
        note = "  (discarded)" if turn.get("discarded") else ""
        print(
            f"{INDENT}{turn['index'] + 1:>4}  {turn['n_prompt']:>4}  "
            f"{turn['n_sampled']:>5}  {turn['n_tools']:>5}  {turn['finish_reason']}"
            f"{note}"
        )
    # What the harness asked for and what the proxy sent, once if every call
    # agrees (the harness asks the same each time), else call by call.
    pairs = {
        (
            json.dumps(turn["requested_sampling_params"]),
            json.dumps(turn["sampling_params"]),
        )
        for turn in document["turns"]
    }
    for asked, sent in sorted(pairs):
        print(f"{INDENT}the harness asked {asked}; the proxy sent {sent}")
    summary = summarize(document)
    print(
        f"{INDENT}turns {summary.turns} · conversations (roots) {summary.roots} · "
        f"the last call read {summary.last_prompt_tokens:,} tokens · the model "
        f"wrote {summary.sampled_tokens:,}"
    )
    print(
        f"{INDENT}training rows {summary.rows} · trainable tokens "
        f"{summary.trainable_tokens:,} · {summary.rollout_type} rollout · "
        f"trainable: {str(summary.trainable).lower()}"
    )
    for finding in summary.findings:
        print(f"{INDENT}finding: {finding}")


def save_document(document: dict[str, Any], task_id: str) -> Path:
    """Write the proxy's record as JSON.

    Args:
        document: The rollout document.
        task_id: The question.

    Returns:
        The file.
    """
    path = rollout_path(LABEL, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--purpose", default=PURPOSE_TRAIN, choices=(PURPOSE_TRAIN, PURPOSE_EVAL)
    )
    return parser.parse_args()


def run_and_record(client: CaptureClient, session_id: str, task_id: str) -> None:
    """Run the harness in a session, then print and save the proxy's record.

    Args:
        client: The proxy's session client.
        session_id: The session, the harness's API key.
        task_id: The question.

    Raises:
        SystemExit: If the proxy stops answering.
    """
    task = find_task(task_id)
    base_url = client.proxy_url + OPENAI_ROOT
    print(
        f"{INDENT}the harness's two settings: base URL {base_url}, API key {session_id}"
    )
    print(f"\n{task.task_id} ({task.difficulty}): {task.question}")
    workdir = prepare_workdir(task, work_folder(LABEL, task.task_id))
    try:
        chat = http_chat(base_url, session_id)
        rollout = run_rollout(chat, task, workdir, show=show_call)
    except EngineError as error:
        raise SystemExit(f"the proxy did not answer: {error}") from error
    print_grade(rollout)
    print(f"\nthe proxy's record (GET /sessions/{session_id}/rollout)")
    document = client.rollout(session_id)
    print_record(document)
    print(f"{INDENT}saved: {short_path(save_document(document, task.task_id))}")


def main() -> None:
    """Open a session, run the harness through the proxy, print the record."""
    args = parse_args()
    find_task(args.task)  # an unknown question stops here, before any session
    client = CaptureClient(args.proxy_url)
    print(f"{SCRIPT} · the tiny harness (chapters 17 and 18), through the proxy")
    session_id = open_or_explain(client, args.purpose, args.task)
    try:
        run_and_record(client, session_id, args.task)
    except ProxyError as error:
        raise SystemExit(f"the proxy failed: {error}") from error
    finally:
        # Whatever happened, the proxy should not keep the session.
        with contextlib.suppress(ProxyError):
            client.close(session_id)
    print(f"{INDENT}DELETE /sessions/{session_id}: done")


if __name__ == "__main__":
    main()
