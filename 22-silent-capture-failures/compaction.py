# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Ask Claude Code to compact its history through the proxy, and see whether it does.

``breakages.py``'s summary case needs a harness that replaces its history
with a summary. Claude Code does that itself when its context fills up, so
this script tries to make it happen early: one run through the capture
proxy, in front of the scripted engine, with

* ``CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000``: the context size Claude Code
  counts its compaction against;
* ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=1``: compact once the context passes 1%
  of that, about 1,000 tokens (95% is its default);

and replies that make the context pass that mark at once: every prompt is
about 10,000 tokens, and the second reply alone is about 1,000. It prints
what Claude Code did: its exit code, its calls, how many prompts it sent,
and how many ``compact_boundary`` events its own log holds (the mark it
writes when it compacts), with what it was told about the tokens it read.
The run's account is kept in ``runs/compaction``.

Start the stack with ``ENGINE_PORT=8119 ./start.sh scripted``, then
``uv run compaction.py``. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
from typing import Any

from capture_client import PROXY_URL, PURPOSE_TRAIN, CaptureClient, ProxyError
from captures import model_calls, runs_folder, save_run, stage_folder, tidy_up
from claude_code import (
    CLAUDE_VERSION,
    REMINDER_OFF,
    TOOLS,
    ClaudeNotInstalledError,
    Relay,
    find_claude,
    prepare_run,
    run_claude,
)
from crosscheck import read_events, read_log
from grader import grade
from lfm_files import short_path
from rollout_view import INDENT
from scripted_replies import (
    MURCIA_COMMAND,
    TABLES_COMMAND,
    NotScriptedError,
    ScriptClient,
    ScriptError,
    answer_reply,
    call_reply,
)
from tasks import DATA_DIR, find_task, task_instruction

SCRIPT = "compaction.py"
TASK = "clinic-easy-001"
RUN_NAME = "compaction"
WINDOW_ENV, WINDOW = "CLAUDE_CODE_AUTO_COMPACT_WINDOW", 100_000
PERCENT_ENV, PERCENT = "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", 1
PERCENT_OF = 100
# A thought long enough that the reply alone nears the 1,024-token cap.
LONG_THOUGHT = "Count the patients in Murcia, then check the count. " * 18
REPLIES = (
    call_reply(TABLES_COMMAND, tool=TOOLS),
    call_reply(MURCIA_COMMAND, tool=TOOLS, think=LONG_THOUGHT),
    call_reply("echo 30 > answer.txt", tool=TOOLS, think="Write it."),
    answer_reply("30 patients live in Murcia; answer.txt holds 30."),
)


def scripted_engine(client: CaptureClient) -> ScriptClient:
    """Find the scripted engine behind the proxy.

    Args:
        client: The proxy's session client.

    Returns:
        Its remote control.

    Raises:
        SystemExit: If the proxy, or a scripted engine behind it, is not there.
    """
    try:
        upstream = client.health().get("upstream")
    except ProxyError as error:
        message = f"{error}: start the stack with ./start.sh scripted"
        raise SystemExit(message) from error
    engine = ScriptClient(str(upstream))
    try:
        engine.state()
    except NotScriptedError as error:
        raise SystemExit(
            "this run needs the scripted engine: ENGINE_PORT=8119 ./start.sh scripted"
        ) from error
    except ScriptError as error:
        raise SystemExit(f"the proxy's engine does not answer: {error}") from error
    return engine


def told_input(events: list[dict[str, Any]]) -> str:
    """Say what Claude Code's result event reports about the tokens it read.

    Args:
        events: Its log's events.

    Returns:
        Such as ``input_tokens 0``, or a note that there is no result.
    """
    for event in reversed(events):
        if event.get("type") == "result":
            return f"input_tokens {(event.get('usage') or {}).get('input_tokens')}"
    return "no result event"


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--proxy-url", default=PROXY_URL)
    return parser.parse_args()


def main() -> None:
    """Run Claude Code once, asked to compact early, and print what it did.

    Raises:
        SystemExit: If the stack or Claude Code is missing.
    """
    args = parse_args()
    client = CaptureClient(args.proxy_url)
    engine = scripted_engine(client)
    try:
        binary = find_claude()
    except ClaudeNotInstalledError as error:
        raise SystemExit(str(error)) from error
    task = find_task(TASK)
    settings = {WINDOW_ENV: str(WINDOW), PERCENT_ENV: str(PERCENT)}
    mark = WINDOW * PERCENT // PERCENT_OF
    print(
        f"{SCRIPT} · does Claude Code {CLAUDE_VERSION} compact its history through "
        f"the capture proxy at {client.proxy_url}?"
    )
    print(
        f"{INDENT}settings {WINDOW_ENV}={WINDOW} · {PERCENT_ENV}={PERCENT}: compact "
        f"past {PERCENT}% of {WINDOW:,} tokens, about {mark:,}"
    )
    stage = stage_folder()
    prepare_run(stage, DATA_DIR / task.db_file)
    session = client.open_session(PURPOSE_TRAIN, {"task_id": task.task_id})
    relay = Relay(client.proxy_url)
    try:
        engine.clear()
        engine.push(list(REPLIES))
        run = run_claude(
            binary,
            stage,
            task_instruction(task),
            relay.start(),
            session.session_id,
            REMINDER_OFF,
            settings,
        )
        document = client.rollout(
            session.session_id, include_discarded=True, include_messages=True
        )
    except (ProxyError, ScriptError, ClaudeNotInstalledError) as error:
        raise SystemExit(str(error)) from error
    finally:
        relay.stop()
        notes = tidy_up(engine, client, session.session_id)
    verdict = grade(task, run.task_folder)
    folder = runs_folder() / RUN_NAME
    meta = {"name": RUN_NAME, "case": RUN_NAME, "task_id": task.task_id, **settings}
    save_run(folder, meta, document, relay.requests, relay.forwarded, run, verdict)
    events, _ = read_events(folder / "stream.jsonl")
    compactions = read_log(events).compactions
    turns = sorted(document["turns"], key=lambda t: t["index"])
    code = run.exit_code
    if code is None:
        code = "none (stopped at the time limit)"
    graded = "correct" if verdict.correct else "wrong"
    print(
        f"claude    exit code {code} after {run.seconds:.0f} s · "
        f"{len(model_calls(relay.requests))} prompts sent · grade {graded} "
        f"(answer.txt {verdict.submitted!r})"
    )
    print(
        f"engine    read {', '.join(f'{t["n_prompt"]:,}' for t in turns)} tokens; "
        f"wrote {', '.join(f'{t["n_sampled"]:,}' for t in turns)}"
    )
    print(
        f"log       compact_boundary events: {len(compactions)} · Claude Code was "
        f"told {told_input(events)}"
    )
    stats = document["stats"]
    verdict_line = "compacted" if compactions else "no compaction"
    print(
        f"result    {verdict_line}: {stats['n_roots']} root, "
        f"{stats['n_sequences']} row · saved {short_path(folder)}"
    )
    for note in notes:
        print(f"{INDENT}note: {note}")


if __name__ == "__main__":
    main()
