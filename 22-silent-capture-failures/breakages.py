# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Break one captured run three ways, and see what the proxy makes of each.

``capture.py`` recorded one Claude Code run through the capture proxy:
solved, and one training row. This script replays it. The scripted engine
serves every call the exact tokens and log-probabilities the capture's
engine sampled, so the model's replies are fixed, token for token, while
everything else runs again for real: Claude Code (in the same folder, so
its commands find the same files), the proxy, and the grader at the end.
Then it breaks one thing at a time:

* ``replay``: nothing. The capture again, to show the replay is faithful;
* ``note``: Claude Code's total-tokens note left on, as it is by default.
  It writes a new note after every tool result, and the proxy merges every
  note into the system prompt (chapter 21, finding 1): a note that changes
  on every call, at the top of every prompt;
* ``summary``: from call K on, the history is replaced by a summary on its
  way to the proxy, as a harness that compacts its context does. Claude
  Code compacts only when its context is nearly full, which a run of a few
  thousand tokens never reaches, so a relay between it and the proxy does
  the summarising (and Claude Code's own log knows nothing of it);
* ``no-logprobs``: the engine serves call K's reply with every
  log-probability null.

K is the middle call (``--at`` to choose). For each case it prints whether
the task was still solved, the rows the proxy cut the rollout into and the
calls each row holds, the proxy's findings and its one verdict,
``trainable``; then all four side by side. Each case's account is kept in
``runs/<case>``, where ``crosscheck.py`` checks it against Claude Code's
own log.

Start the stack with ``ENGINE_PORT=8119 ./start.sh scripted`` for a
capture the scripted engine made, or ``ENGINE_PORT=8119 ./start.sh
replay`` for one the model made (its tokens are LFM2.5's), then ``uv run
breakages.py`` for every case or ``uv run breakages.py note`` for one.
Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capture_client import PROXY_URL, PURPOSE_TRAIN, CaptureClient, ProxyError
from captures import (
    CAPTURE,
    CapturedReply,
    CaptureError,
    captured_replies,
    clip_text,
    cut_text,
    model_calls,
    named_run,
    read_run,
    reply_ids,
    runs_folder,
    save_run,
    stage_folder,
    system_change,
    text_of,
    tidy_up,
)
from claude_code import (
    CLAUDE_VERSION,
    REMINDER_ENV,
    REMINDER_OFF,
    ClaudeNotInstalledError,
    Relay,
    RelayedRequest,
    find_claude,
    prepare_run,
    run_claude,
)
from grader import Grade, grade
from lfm_files import short_path, tilde
from rollout_view import COLOR_CHOICES, INDENT, Painter, print_findings, print_stats
from scripted_replies import (
    SCRIPTED_TOKENIZER,
    TOKENIZER_FIELD,
    NotScriptedError,
    ScriptClient,
    ScriptError,
    replayed_reply,
    without_logprobs,
)
from tasks import DATA_DIR, Task, find_task, task_instruction
from training_rows import count_calls, n_sampled

SCRIPT = "breakages.py"
ALL = "all"
AGENT = "agent"
RULE = "━━"
MIN_CALLS = 2  # a break at call K needs a call before it
SCRIPTED_COMMAND, REPLAY_COMMAND = "scripted", "replay"  # start.sh's two engines
KEPT_META = ("task_id", "engine", "tokenizer")  # what a case copies from the capture
SUMMARY_HEAD = "This conversation was summarised to save room."
RESULT_CHARS = 60  # a tool result's first line, in the summary
SHOWN_CHARS = 60  # a summary's start, when shown
ROLE, CONTENT = "role", "content"
USER, ASSISTANT, SYSTEM = "user", "assistant", "system"
TOOL_USE, TOOL_RESULT = "tool_use", "tool_result"


@dataclass(frozen=True)
class Case:
    """One way to replay the capture.

    Attributes:
        name: Its name on the command line and its run folder.
        title: What it changes (``{at}`` is the call it hits).
        reminder: Claude Code's total-tokens setting (None: its default, on).
        summarise: Replace the history before call K with a summary.
        unscored: Serve call K without log-probabilities.
    """

    name: str
    title: str
    reminder: str | None = REMINDER_OFF
    summarise: bool = False
    unscored: bool = False


CASES = (
    Case("replay", "nothing changed: the capture, replayed"),
    Case("note", "Claude Code's total-tokens note, left on (its default)", None),
    Case("summary", "the history before call {at} summarised", summarise=True),
    Case("no-logprobs", "call {at} served without log-probabilities", unscored=True),
)
CASE_WIDTH = max(len(case.name) for case in CASES) + 1


@dataclass(frozen=True)
class Capture:
    """The run every case replays.

    Attributes:
        folder: Its run folder.
        meta: Its ``run.json``.
        document: The proxy's record of it.
        replies: Every call's exact reply.
        task: The task it solved.
    """

    folder: Path
    meta: dict[str, Any]
    document: dict[str, Any]
    replies: list[CapturedReply]
    task: Task


@dataclass(frozen=True)
class Outcome:
    """One case's replay, for the closing table.

    Attributes:
        case: The case.
        grade: The verdict on answer.txt.
        trainer: The rollout document the trainer gets.
    """

    case: Case
    grade: Grade
    trainer: dict[str, Any]


@dataclass
class Summariser:
    """A harness's compaction, done by the relay: the history becomes a summary.

    From call ``at`` on, the conversation before that call's last exchange
    (the model's last reply and the tool's answer) is replaced by one user
    message summarising it, and every later call keeps the same summary in
    the same place, as a harness that compacted once would. Messages with
    the role ``system`` stay: Claude Code sends its environment that way,
    the proxy merges it into the system prompt, and a harness that compacts
    keeps its own instructions.

    Attributes:
        at: The call that is summarised, from 1.
        task_text: The task, as Claude Code was given it.
        calls: Model calls seen so far.
        cut: Where the kept exchange starts (set at call ``at``; None when
            that call had no reply before it to summarise, and then no call
            is changed).
        head: The summary, then the ``system`` messages it did not replace.
    """

    at: int
    task_text: str
    calls: int = 0
    cut: int | None = None
    head: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Rewrite one request on its way to the proxy.

        Args:
            path: The request's path.
            body: Its JSON body, as Claude Code sent it.

        Returns:
            The body to forward.
        """
        if not model_calls([RelayedRequest(path, body)]):
            return body  # not a model call
        self.calls += 1
        messages = body["messages"]
        if self.calls == self.at:
            replies = [i for i, m in enumerate(messages) if m.get(ROLE) == ASSISTANT]
            if not replies:
                return body  # nothing to summarise: what_changed says so
            self.cut = replies[-1]
            history = messages[: self.cut]
            summary = {ROLE: USER, CONTENT: summary_text(self.task_text, history)}
            kept = [m for m in history if m.get(ROLE) == SYSTEM]
            self.head = [summary, *kept]
        if self.cut is None or self.calls < self.at:
            return body
        return {**body, "messages": [*self.head, *messages[self.cut :]]}


def blocks_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    """List a message's content blocks (a plain string becomes one text block).

    Args:
        message: An Anthropic message.

    Returns:
        Its blocks.
    """
    content = message.get(CONTENT)
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return [{"type": "text", "text": content or ""}]


def summary_text(task_text: str, messages: list[dict[str, Any]]) -> str:
    """Summarise a history the way a compacting harness might: task, steps, results.

    Args:
        task_text: The task.
        messages: The messages being replaced.

    Returns:
        The summary.
    """
    results = {}
    for message in messages:
        for block in blocks_of(message):
            if block.get("type") == TOOL_RESULT:
                first = text_of(block.get(CONTENT)).strip().splitlines() or [""]
                results[block.get("tool_use_id")] = cut_text(first[0], RESULT_CHARS)
    lines = [SUMMARY_HEAD, f"The task: {task_text}", "Done so far:"]
    for message in messages:
        if message.get(ROLE) != ASSISTANT:
            continue
        for block in blocks_of(message):
            if block.get("type") == TOOL_USE:
                given = block.get("input") or {}
                command = given.get("command") if isinstance(given, dict) else None
                shown = command if command is not None else json.dumps(given)
                answer = results.get(block.get("id"), "")
                lines.append(f"- {block.get('name')}: {shown} -> {answer}")
    return "\n".join(lines)


def plural(count: int, noun: str) -> str:
    """Write a count with its noun, adding an s when it is not one.

    Args:
        count: How many.
        noun: The noun, singular.

    Returns:
        Such as ``1 row`` or ``4 rows``.
    """
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def calls_of(row: dict[str, Any], turns: dict[str, dict[str, Any]]) -> list[int]:
    """List the calls a row holds, numbered from 1 as the proxy numbers them.

    Discarded calls keep their numbers here, as in ``--at`` and in the
    capture's replies (crosscheck.py numbers only the calls it compares).

    Args:
        row: One of the document's rows.
        turns: Its turns, by node id.

    Returns:
        The call numbers, in order.
    """
    return [turns[node]["index"] + 1 for node in row["node_ids"]]


def call_span(calls: list[int]) -> str:
    """Write a list of calls compactly.

    Args:
        calls: Call numbers.

    Returns:
        Such as ``call 3`` or ``calls 1-4`` (or ``calls 1, 3`` when not in a run).
    """
    if len(calls) == 1:
        return f"call {calls[0]}"
    if calls == list(range(calls[0], calls[-1] + 1)):
        return f"calls {calls[0]}-{calls[-1]}"
    return "calls " + ", ".join(map(str, calls))


def print_rows_brief(document: dict[str, Any]) -> None:
    """Print each row: its role, verdict, the calls it holds and its sizes.

    Args:
        document: A rollout document (every row).
    """
    turns = {turn["node_id"]: turn for turn in document["turns"]}
    rows = document["sequences"]
    print(f"rows      {len(rows)}")
    for number, row in enumerate(rows, start=1):
        verdict = "trainable" if row["trainable"] else "not trainable"
        print(
            f"{INDENT}row {number} · {row['role']} · {verdict:<13} · "
            f"{call_span(calls_of(row, turns)):<10} · {len(row['input_ids']):>7,} "
            f"tokens, {sum(row['loss_mask']):>6,} with mask 1"
        )


def same_replies(document: dict[str, Any], capture: Capture) -> str:
    """Say whether the replay's calls sampled exactly the capture's tokens.

    Args:
        document: The replay's record (every row).
        capture: The capture.

    Returns:
        One line, naming the first call that differs if one does.
    """
    turns = sorted(document["turns"], key=lambda t: t["index"])
    if len(turns) != len(capture.replies):
        return (
            f"{len(turns)} calls, the capture had {len(capture.replies)}: "
            "the replay did not follow the capture"
        )
    for turn, reply in zip(turns, capture.replies, strict=True):
        try:
            ids = reply_ids(document, turn)
        except CaptureError as error:
            return f"not compared: {error}"
        if ids != reply.ids:
            return f"call {turn['index'] + 1} differs from the capture's reply"
    line = f"the capture's {len(turns)}, token for token"
    rows = [row["input_ids"] for row in document["sequences"]]
    if rows == [row["input_ids"] for row in capture.document["sequences"]]:
        return line + "; every row too, all its tokens"
    return line


def what_changed(case: Case, at: int, relay: Relay, document: dict[str, Any]) -> str:
    """Show what the case changed, from the records of this run.

    Args:
        case: The case.
        at: The call the summary or the missing log-probabilities hit.
        relay: The relay (what Claude Code sent, and what it forwarded).
        document: The proxy's record of the run (with messages).

    Returns:
        One line (empty for the plain replay).
    """
    if case.reminder is None:
        turns = sorted(document["turns"], key=lambda t: t["index"])
        firsts = [
            text_of((t.get("request_messages") or [{}])[0].get(CONTENT)) for t in turns
        ]
        grown = sum(
            after != before and after.startswith(before)
            for before, after in zip(firsts, firsts[1:], strict=False)
        )
        shown = system_change(firsts[0], firsts[1]) if len(firsts) > 1 else ""
        return (
            f"changed   the engine's system prompt at call 2 {shown}; it grew at "
            f"its end on {grown} of {len(firsts) - 1} calls"
        )
    if case.summarise:
        sent = model_calls(relay.requests)
        forwarded = model_calls(relay.forwarded)
        if len(sent) < at:
            return f"changed   nothing: Claude Code made only {len(sent)} calls"
        if forwarded[at - 1] == sent[at - 1]:
            return f"changed   nothing: call {at} had no reply before it to summarise"
        before, after = len(sent[at - 1]["messages"]), forwarded[at - 1]["messages"]
        start = clip_text(text_of(after[0].get(CONTENT)), SHOWN_CHARS)
        kept = sum(m.get(ROLE) == SYSTEM for m in after[1:])
        replaced = before - len(after) + 1  # the summary stands for these
        return (
            f"changed   call {at}: Claude Code sent {before} messages, the proxy got "
            f"{len(after)}: {replaced} replaced by one summary "
            f"({kept} system message kept), {json.dumps(start, ensure_ascii=False)}"
        )
    if case.unscored:
        turn = next(
            (t for t in document["turns"] if t["index"] == at - 1),
            None,
        )
        size = turn["n_sampled"] if turn else 0
        return f"changed   call {at}: its {size} tokens, served with no logprobs"
    return ""


def run_case(
    case: Case,
    capture: Capture,
    at: int,
    client: CaptureClient,
    engine: ScriptClient,
    binary: Path,
    painter: Painter,
) -> Outcome:
    """Replay the capture once, broken one way, and print what the proxy built.

    Args:
        case: The case.
        capture: The capture.
        at: The call a summary or a missing log-probability hits, from 1.
        client: The proxy's session client.
        engine: The scripted engine's remote control.
        binary: The pinned ``claude``.
        painter: Colours, or none.

    Returns:
        The case's numbers.

    Raises:
        ClaudeNotInstalledError: If the sandbox cannot be applied.
        ProxyError: If the proxy refuses a session or a rollout.
        ScriptError: If the scripted engine refuses the replies.
        OSError: If the run's account cannot be written.
    """
    print(painter.paint(f"\n{RULE} {case.name} · {case.title.format(at=at)}", "head"))
    task = capture.task
    instruction = task_instruction(task)
    stage = stage_folder()
    prepare_run(stage, DATA_DIR / task.db_file)
    replies = [replayed_reply(r.ids, r.logprobs) for r in capture.replies]
    if case.unscored:
        replies[at - 1] = without_logprobs(replies[at - 1])
    session = client.open_session(
        PURPOSE_TRAIN, {"task_id": task.task_id, "case": case.name}
    )
    rewrite = Summariser(at, instruction) if case.summarise else None
    relay = Relay(client.proxy_url, rewrite=rewrite)
    try:
        engine.clear()
        engine.push(replies)
        base_url = relay.start()
        run = run_claude(
            binary, stage, instruction, base_url, session.session_id, case.reminder
        )
        unused = engine.state()["queued"]
        trainer = client.rollout(session.session_id)
        full = client.rollout(
            session.session_id, include_discarded=True, include_messages=True
        )
    finally:
        relay.stop()
        notes = tidy_up(engine, client, session.session_id)
    verdict = grade(task, run.task_folder)
    folder = runs_folder() / case.name
    meta = {
        **{key: capture.meta.get(key) for key in KEPT_META},
        "name": case.name,
        "case": case.name,
        "replays": str(capture.folder),
        "at": at,
        "claude_code": CLAUDE_VERSION,
        "reminder": case.reminder,
        "stage": str(stage),
        "session_id": session.session_id,
    }
    save_run(folder, meta, full, relay.requests, relay.forwarded, run, verdict)
    ending = "stopped at the time limit" if run.exit_code is None else "exit"
    code = "" if run.exit_code is None else f" {run.exit_code}"
    setting = "unset" if case.reminder is None else case.reminder
    print(
        f"claude    {ending}{code} after {run.seconds:.0f} s · {REMINDER_ENV} "
        f"{setting} · {len(full['turns'])} calls · grade "
        f"{'correct' if verdict.correct else 'wrong'} (answer.txt "
        f"{verdict.submitted!r})"
    )
    print(f"replies   {same_replies(full, capture)}")
    if unused:
        print(f"{INDENT}note: {unused} replies were never asked for")
    for note in notes:
        print(f"{INDENT}note: {note}")
    changed = what_changed(case, at, relay, full)
    if changed:
        print(changed)
    print_rows_brief(full)
    print_findings(trainer, painter)
    print_stats(trainer)
    kept = [row for row in trainer["sequences"] if row["trainable"]]
    print(
        f"project   run_rollout keeps {plural(len(kept), 'row')}, "
        f"{trainer['stats']['n_trainable_tokens']:,} tokens to train · "
        f"count_calls {count_calls(trainer)} · n_sampled {n_sampled(trainer):,}"
    )
    print(f"saved     {short_path(folder)}")
    return Outcome(case, verdict, trainer)


def print_table(outcomes: list[Outcome]) -> None:
    """Print every case side by side.

    Args:
        outcomes: One per case.
    """
    print(f"\n{RULE} one capture, replayed {len(outcomes)} ways")
    print(
        f"{INDENT}{'case':<{CASE_WIDTH}} {'grade':<8} calls roots rows row-tokens "
        "mask-1 trainable"
    )
    for outcome in outcomes:
        doc = outcome.trainer
        stats = doc["stats"]
        row_tokens = sum(len(row["input_ids"]) for row in doc["sequences"])
        verdict = "correct" if outcome.grade.correct else "wrong"
        print(
            f"{INDENT}{outcome.case.name:<{CASE_WIDTH}} {verdict:<8} "
            f"{stats['n_turns']:>5} {stats['n_roots']:>5} {stats['n_sequences']:>4} "
            f"{row_tokens:>10,} {stats['n_trainable_tokens']:>6,} "
            f"{str(doc['trainable']).lower():>9}"
        )
    solved = sum(outcome.grade.correct for outcome in outcomes)
    trainable = sum(bool(outcome.trainer["trainable"]) for outcome in outcomes)
    print(
        f"{INDENT}solved {solved} of {len(outcomes)}; the proxy's verdict, "
        f"trainable, is true in {trainable} of {len(outcomes)}. Check each against "
        "Claude Code's own log: uv run crosscheck.py all"
    )


def load_capture(name: str) -> Capture:
    """Read the capture and the replies to serve.

    Args:
        name: Its run folder's name, or a folder.

    Returns:
        The capture.

    Raises:
        SystemExit: If it is missing, unsolved, or cannot be replayed.
    """
    folder = named_run(name)
    try:
        meta, document = read_run(folder)
        replies = captured_replies(document)
        task = find_task(str(meta.get("task_id")))
    except CaptureError as error:
        raise SystemExit(f"{error}: record one first with capture.py") from error
    if not (meta.get("grade") or {}).get("correct"):
        raise SystemExit(f"{short_path(folder)} did not solve its task: no replay")
    return Capture(folder, meta, document, replies, task)


def replay_engine(client: CaptureClient, tokenizer: str) -> ScriptClient:
    """Find the scripted engine behind the proxy, cutting tokens as the capture did.

    Args:
        client: The proxy's session client.
        tokenizer: The capture's tokenizer.

    Returns:
        The engine's remote control.

    Raises:
        SystemExit: If the proxy, or a scripted engine with that tokenizer, is
            not there.
    """
    try:
        upstream = client.health().get("upstream")
    except ProxyError as error:
        raise SystemExit(f"{error}: start the stack with ./start.sh") from error
    engine = ScriptClient(str(upstream))
    command = SCRIPTED_COMMAND if tokenizer == SCRIPTED_TOKENIZER else REPLAY_COMMAND
    try:
        found = engine.state().get(TOKENIZER_FIELD)
    except NotScriptedError as error:
        raise SystemExit(
            f"the proxy is in front of a real engine; a replay needs the scripted "
            f"one: ./start.sh stop, then ENGINE_PORT=8119 ./start.sh {command}"
        ) from error
    except ScriptError as error:
        raise SystemExit(f"the proxy's engine does not answer: {error}") from error
    if found != tokenizer:
        raise SystemExit(
            f"the capture's tokens are {tokenizer}'s, the engine cuts {found}'s: "
            f"./start.sh stop, then ENGINE_PORT=8119 ./start.sh {command}"
        )
    return engine


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choices = (ALL, *(case.name for case in CASES))
    parser.add_argument("case", nargs="?", default=ALL, choices=choices)
    parser.add_argument("--capture", default=CAPTURE, help="a run folder")
    parser.add_argument("--at", type=int, help="the call to break (the middle one)")
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--color", default="auto", choices=COLOR_CHOICES)
    return parser.parse_args()


def main() -> None:
    """Replay the capture once per case, and print what the proxy built each time.

    Raises:
        SystemExit: If the capture, the engine or Claude Code is missing.
    """
    args = parse_args()
    painter = Painter(args.color)
    capture = load_capture(args.capture)
    calls = len(capture.replies)
    at = args.at if args.at is not None else calls // 2 + 1
    if calls < MIN_CALLS or not MIN_CALLS <= at <= calls:
        raise SystemExit(
            f"the capture has {calls} calls: --at must be {MIN_CALLS} to {calls}"
        )
    client = CaptureClient(args.proxy_url)
    engine = replay_engine(client, str(capture.meta.get("tokenizer")))
    try:
        binary = find_claude()
    except ClaudeNotInstalledError as error:
        raise SystemExit(str(error)) from error
    stats = capture.document["stats"]
    print(
        f"{SCRIPT} · one Claude Code run, replayed through the capture proxy at "
        f"{client.proxy_url} with its replies fixed"
    )
    print(
        f"{INDENT}capture {short_path(capture.folder)}: {capture.meta.get('engine')}"
        f" engine, {calls} calls, {plural(stats['n_roots'], 'root')}, "
        f"{plural(stats['n_sequences'], 'row')}, "
        f"{stats['n_trainable_tokens']:,} tokens to train, grade correct"
    )
    print(f"{INDENT}task {capture.task.task_id}: {capture.task.question}")
    print(f"{INDENT}breaks at call {at} of {calls}")
    if capture.meta.get("stage") != str(stage_folder()):
        ran = short_path(Path(str(capture.meta.get("stage"))))
        print(f"{INDENT}note: the capture ran in {ran}, not in this stage")
    chosen = [case for case in CASES if args.case in (ALL, case.name)]
    outcomes = []
    for case in chosen:
        try:
            outcomes.append(
                run_case(case, capture, at, client, engine, binary, painter)
            )
        except (ProxyError, ScriptError, ClaudeNotInstalledError, OSError) as error:
            raise SystemExit(f"{case.name}: {tilde(str(error))}") from error
    if len(outcomes) > 1:
        print_table(outcomes)


if __name__ == "__main__":
    main()
