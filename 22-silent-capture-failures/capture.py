# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Record the one run this chapter breaks: Claude Code through the proxy, solved.

Claude Code 2.1.290 runs once on a task, set up as the project sets it up:
chapter 21's flags, a private config and ``HOME``, the session id as its
key, ``CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off``, inside chapter 19's sandbox,
reaching the model only through the capture proxy. It runs in the stage
folder (``captures.py``) and its account is kept as ``runs/capture``:
the proxy's record, what Claude Code sent (through a recording relay) and
Claude Code's own log. ``breakages.py`` replays exactly this run.

With the scripted engine (``ENGINE_PORT=8119 ./start.sh scripted``) the
replies are four scripted ``Bash`` calls, and the run is solved every time.
With the model (``ENGINE_PORT=8119 ./start.sh``), LFM2.5 answers at the
training temperature, 1, and does not always solve the task, so the script
tries up to ``--attempts`` times and keeps the first run that is solved and
already one trainable row (the baseline every case breaks). Every attempt
is kept as ``runs/attempt-N`` and printed, with why it was not kept. When
none is, the best one is kept anyway (solved first, then one root), and
its flaw is printed and written into its ``run.json``: the cases still show
what each break does to its rows, under its grade.

It also prints one thing the capture shows about the proxy: Claude Code
begins its system prompt with a billing line, and the engine never sees it,
because the proxy drops that line (a line that changed on every call would
split every rollout). Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capture_client import PROXY_URL, PURPOSE_TRAIN, CaptureClient, ProxyError
from captures import (
    CAPTURE,
    CASE_RUNS,
    RUN_FILE,
    model_calls,
    runs_folder,
    save_run,
    stage_folder,
    text_of,
    tidy_up,
)
from claude_code import (
    CLAUDE_VERSION,
    REMINDER_OFF,
    TOOLS,
    ClaudeNotInstalledError,
    Relay,
    RelayedRequest,
    find_claude,
    prepare_run,
    run_claude,
)
from grader import Grade, grade
from lfm_files import short_path, tilde
from rollout_view import (
    COLOR_CHOICES,
    INDENT,
    Painter,
    print_findings,
    print_graph,
    print_rows,
    print_stats,
)
from scripted_replies import (
    LFM_TOKENIZER,
    MURCIA_COMMAND,
    TABLES_COMMAND,
    TOKENIZER_FIELD,
    NotScriptedError,
    ScriptClient,
    ScriptError,
    answer_reply,
    call_reply,
)
from tasks import DATA_DIR, Task, find_task, task_instruction
from training_rows import count_calls, n_sampled

SCRIPT = "capture.py"
DEFAULT_TASK = "clinic-easy-001"
DEFAULT_ATTEMPTS = 3
ATTEMPT = "attempt-{number}"
ENGINE_SCRIPTED, ENGINE_MODEL = "scripted", "model"
BILLING_LINE = "x-anthropic-billing-header"  # Claude Code's first system line
# The scripted model's replies: Claude Code's tool is named Bash.
SCRIPTED = (
    call_reply(TABLES_COMMAND, tool=TOOLS),
    call_reply(MURCIA_COMMAND, tool=TOOLS, think="Count them."),
    call_reply("echo 30 > answer.txt", tool=TOOLS, think="Write it."),
    answer_reply("30 patients live in Murcia; answer.txt holds 30."),
)


@dataclass(frozen=True)
class Attempt:
    """One recorded run.

    Attributes:
        folder: Where its account is kept.
        grade: The verdict on answer.txt.
        document: The proxy's rollout document, every row and message.
        sent: What Claude Code sent, request by request.
    """

    folder: Path
    grade: Grade
    document: dict[str, Any]
    sent: list[RelayedRequest]


def engine_kind(client: CaptureClient) -> tuple[ScriptClient | None, str]:
    """Find out which engine the proxy is in front of, and its tokenizer.

    Args:
        client: The proxy's session client.

    Returns:
        The scripted engine's remote control (None for the model's engine),
        and the tokenizer its token ids come from.

    Raises:
        SystemExit: If the proxy or its engine does not answer.
    """
    try:
        upstream = client.health().get("upstream")
    except ProxyError as error:
        raise SystemExit(f"{error}: start the stack with ./start.sh") from error
    if not upstream:
        raise SystemExit(f"{client.proxy_url}/health names no engine: is it the proxy?")
    engine = ScriptClient(upstream)
    try:
        state = engine.state()
    except NotScriptedError:
        return None, LFM_TOKENIZER  # the model's engine: LFM2.5's own tokens
    except ScriptError as error:
        raise SystemExit(f"the proxy's engine does not answer: {error}") from error
    return engine, str(state.get(TOKENIZER_FIELD))


def record(
    number: int,
    task: Task,
    client: CaptureClient,
    binary: Path,
    engine: ScriptClient | None,
    tokenizer: str,
) -> Attempt:
    """Run Claude Code once in the stage, and keep its account.

    Args:
        number: The attempt's number, from 1.
        task: The task.
        client: The proxy's session client.
        binary: The pinned ``claude``.
        engine: The scripted engine's remote control, or None for the model.
        tokenizer: Where the engine's token ids come from.

    Returns:
        The attempt.

    Raises:
        ClaudeNotInstalledError: If the sandbox cannot be applied.
    """
    stage = stage_folder()
    prepare_run(stage, DATA_DIR / task.db_file)
    session = client.open_session(PURPOSE_TRAIN, {"task_id": task.task_id})
    relay = Relay(client.proxy_url)
    try:
        if engine is not None:
            engine.clear()
            engine.push(list(SCRIPTED))
        base_url = relay.start()
        run = run_claude(
            binary,
            stage,
            task_instruction(task),
            base_url,
            session.session_id,
            REMINDER_OFF,
        )
        unused = engine.state()["queued"] if engine is not None else 0
        document = client.rollout(
            session.session_id, include_discarded=True, include_messages=True
        )
    finally:
        relay.stop()
        notes = tidy_up(engine, client, session.session_id)
    verdict = grade(task, run.task_folder)
    folder = runs_folder() / ATTEMPT.format(number=number)
    meta = {
        "name": folder.name,
        "case": CAPTURE,
        "task_id": task.task_id,
        "engine": ENGINE_SCRIPTED if engine is not None else ENGINE_MODEL,
        "tokenizer": tokenizer,
        "claude_code": CLAUDE_VERSION,
        "reminder": REMINDER_OFF,
        "stage": str(stage),
        "session_id": session.session_id,
    }
    save_run(folder, meta, document, relay.requests, relay.forwarded, run, verdict)
    stats = document["stats"]
    code = f"exit {run.exit_code}"
    if run.exit_code is None:
        code = "stopped at the time limit"
    print(
        f"attempt {number}  {code} after {run.seconds:.0f} s · "
        f"{stats['n_turns']} calls · grade {'correct' if verdict.correct else 'wrong'} "
        f"(answer.txt {verdict.submitted!r}) · roots {stats['n_roots']}, "
        f"trainable {str(document['trainable']).lower()} · kept in {short_path(folder)}"
    )
    if unused:  # the conversation did not go as scripted
        print(f"{INDENT}note: {unused} scripted replies were never asked for")
    for note in notes:
        print(f"{INDENT}note: {note}")
    return Attempt(folder, verdict, document, relay.requests)


def rank(attempt: Attempt) -> tuple[bool, bool, bool]:
    """Order attempts from the best capture to the worst.

    Args:
        attempt: A recorded attempt.

    Returns:
        A key that sorts solved before unsolved, then one root before
        several, then trainable before not (ties keep their order).
    """
    document = attempt.document
    return (
        not attempt.grade.correct,
        document["stats"]["n_roots"] != 1,
        not document["trainable"],
    )


def keep(attempt: Attempt, flaw: str = "") -> None:
    """Make an attempt the capture, and clear the runs made from the last one.

    breakages.py's runs replay the capture they were made from; left beside a
    new capture, they would read as its breakages. So they are removed, and
    the removal is printed.

    Args:
        attempt: The attempt to keep.
        flaw: Why it is not a clean capture ("" when it is), kept in run.json.
    """
    target = runs_folder() / CAPTURE
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(attempt.folder, target)
    record = json.loads((target / RUN_FILE).read_text(encoding="utf-8"))
    record.update(
        {
            "name": CAPTURE,
            "attempt": attempt.folder.name,
            "clean": not flaw,
            "flaw": flaw,
        }
    )
    (target / RUN_FILE).write_text(json.dumps(record, indent=1) + "\n", "utf-8")
    label = f" (not clean: {flaw})" if flaw else ""
    print(f"capture   {attempt.folder.name}, kept as {short_path(target)}{label}")
    stale = [name for name in CASE_RUNS if (runs_folder() / name).exists()]
    for name in stale:
        shutil.rmtree(runs_folder() / name)
    if stale:
        print(f"{INDENT}cleared the last capture's runs: {', '.join(stale)}")


def why_not_kept(attempt: Attempt) -> str:
    """Say why an attempt cannot be the capture, if it cannot.

    The capture is the baseline every case breaks, so it must be solved and
    already one trainable row: a run that split by itself would hide the
    splits the cases make.

    Args:
        attempt: A recorded attempt.

    Returns:
        The reason, or "" when it can be the capture.
    """
    if not attempt.grade.correct:
        return f"the task was not solved ({attempt.grade.reason})"
    roots = attempt.document["stats"]["n_roots"]
    if roots != 1:
        return f"solved, but in {roots} roots"
    if not attempt.document["trainable"]:
        return "solved, but not trainable"
    return ""


def billing_note(sent: list[RelayedRequest], document: dict[str, Any]) -> str:
    """Say what became of Claude Code's billing line between it and the engine.

    Args:
        sent: What Claude Code sent.
        document: The proxy's record (with messages).

    Returns:
        One line.
    """
    firsts = []
    for body in model_calls(sent):
        system = body.get("system")
        blocks = system if isinstance(system, list) else [{"text": system or ""}]
        first = text_of(blocks[:1]).strip()
        if first.startswith(BILLING_LINE):
            firsts.append(first)
    if not firsts:
        return "Claude Code's system prompt carried no billing line"
    reached = sum(
        BILLING_LINE in text_of(message.get("content"))
        for turn in document["turns"]
        for message in turn.get("request_messages") or []
        if message.get("role") == "system"
    )
    kinds = len(set(firsts))
    same = "the same line every time" if kinds == 1 else f"{kinds} different lines"
    return (
        f"Claude Code began its system prompt with {firsts[0]!r} on "
        f"{len(firsts)} of {len(model_calls(sent))} calls ({same}); the engine "
        f"received it {reached} times"
    )


def show(attempt: Attempt, painter: Painter) -> None:
    """Print the capture: its graph, its rows, the proxy's verdict.

    Args:
        attempt: The solved attempt.
        painter: Colours, or none.
    """
    document = attempt.document
    print(f"billing   {billing_note(attempt.sent, document)}")
    print_graph(document, painter)
    print_rows(document, painter)
    print_findings(document, painter)
    print_stats(document)
    print(
        f"project   count_calls {count_calls(document)} · "
        f"n_sampled {n_sampled(document):,}"
    )


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--color", default="auto", choices=COLOR_CHOICES)
    return parser.parse_args()


def main() -> None:
    """Record attempts until one is solved, and keep it as the capture.

    Raises:
        SystemExit: If nothing can run.
    """
    args = parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    painter = Painter(args.color)
    client = CaptureClient(args.proxy_url)
    engine, tokenizer = engine_kind(client)
    try:
        binary = find_claude()
        task = find_task(args.task)
    except ClaudeNotInstalledError as error:
        raise SystemExit(str(error)) from error
    model = "the model"
    if engine is not None:
        model = f"the scripted engine ({tokenizer} tokens)"
    print(
        f"{SCRIPT} · Claude Code {CLAUDE_VERSION} through the capture proxy at "
        f"{client.proxy_url}, in front of {model}"
    )
    print(f"{INDENT}task {task.task_id}: {task.question}")
    attempts = []
    for number in range(1, args.attempts + 1):
        try:
            attempt = record(number, task, client, binary, engine, tokenizer)
        except (ProxyError, ScriptError, ClaudeNotInstalledError, OSError) as error:
            raise SystemExit(f"attempt {number}: {tilde(str(error))}") from error
        attempts.append(attempt)
        flaw = why_not_kept(attempt)
        if not flaw:
            break
        print(f"{INDENT}not kept: {flaw}")
    chosen = min(attempts, key=rank)
    flaw = why_not_kept(chosen)
    if flaw:
        print(
            f"no attempt in {args.attempts} was solved in one trainable row: keeping "
            f"the best, {chosen.folder.name} ({flaw}), labelled so in its run.json"
        )
    keep(chosen, flaw)
    show(chosen, painter)


if __name__ == "__main__":
    main()
