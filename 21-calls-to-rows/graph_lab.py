# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Seven scripted conversations through the real capture proxy, printed as data.

A harness makes model calls; a trainer needs rows of tokens. In between,
the capture proxy (chapter 20) links the calls into a graph and cuts the
graph into rows. This script plays the harness, on purpose in seven
different ways, and prints what the proxy built each time: the graph, every
row span by span with its loss mask and log-probabilities, the proxy's
findings and stats, then what the project's own functions make of it
(``count_calls``, ``n_sampled``, ``dedupe_shared_prefixes``).

The harness is chapter 17's (a system prompt, the task, one ``bash`` tool),
and every case solves the same question with the same four commands, run
for real and graded. Only the harness's behaviour changes:

* ``clean``: four calls, each continuing the last;
* ``retry``: a call sent twice, first in the middle, then at the end;
* ``changing-line``: the system prompt's last line counts down every call,
  like Claude Code's ``<total_tokens>`` note;
* ``compaction``: after call 2 the history is replaced by a summary;
* ``no-logprobs``: the engine serves call 3 without its log-probabilities;
* ``side-call``: a request with no tools, for a title, after call 1;
* ``broken-engine``: an engine that drops every log-probability when a
  request offers tools; the proxy probes it and finds nothing wrong.

The engine is the scripted one, so no model is loaded. Start the stack
with ``ENGINE_PORT=8119 ./start.sh scripted`` (the engine on 8119 here,
the proxy on 8200), then ``uv run graph_lab.py`` for every case or
``uv run graph_lab.py retry`` for one, and ``./start.sh stop``.
``--dry-run FILE`` re-measures the project's shared-prefix figure from a
project ``rollouts.jsonl``. It needs only Python's standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capture_client import PROXY_URL, PURPOSE_TRAIN, CaptureClient, ProxyError
from grader import Grade, grade
from lfm_files import MAX_CALLS, OUTPUT_CAP, rollout_path, short_path, work_folder
from rollout_view import (
    COLOR_CHOICES,
    INDENT,
    Painter,
    clip,
    print_findings,
    print_graph,
    print_rows,
    print_stats,
)
from scripted_replies import Reply, ScriptClient, ScriptError, answer_reply, call_reply
from tasks import Task, find_task, prepare_workdir
from tiny_harness_http import (
    ASSISTANT,
    DEFAULT_TASK,
    MODEL_NAME,
    SYSTEM,
    USER,
    Chat,
    EngineError,
    bash_schema,
    execute,
    http_chat,
    start_messages,
    tool_message,
)
from training_rows import (
    count_calls,
    dedupe_shared_prefixes,
    item_from_sequence,
    n_sampled,
    shared_prefix,
    trainable_sequences,
    trained_tokens,
)

SCRIPT = "graph_lab.py"
LABEL = "graph_lab"  # this script's work and rollout folders
OPENAI_ROOT = "/v1"
BROKEN_PATH = "/broken"  # the scripted engine's second, broken engine
TRAIN_TEMPERATURE = 1.0  # what the harness asks; a train session pins it anyway
# The four steps every case takes, as the scripted model writes them.
MURCIA = "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""
TABLES = call_reply("sqlite3 data.db .tables")
COUNT = call_reply(MURCIA, think="Count them.")
WRITE = call_reply('echo "30" > answer.txt', think="Write it.")
ANSWER = answer_reply("I wrote 30 to answer.txt.")
# Replies a harness receives and then throws away (a retried call).
ABANDONED_COUNT = call_reply(
    "sqlite3 data.db 'SELECT COUNT(*) FROM patients'", think="Count all."
)
ABANDONED_ANSWER = answer_reply("The answer is 30.")
# Call 3, served without its log-probabilities.
WRITE_UNSCORED = call_reply('echo "30" > answer.txt', think="Write it.", logprobs=False)
# The side call: a separate request, with no tools, for a title.
TITLE_PROMPT = "Write a title of at most five words for this task."
TITLE = answer_reply("Patients in Murcia", think="A title.")
SIDE_CALL_TOKENS = 32
# The changing line, in Claude Code 2.1.290's words: it counts down from
# 15,000,000 by the tokens each call read and wrote (reminder.py shows the
# real one).
TOKEN_BUDGET = 15_000_000
REMINDER_LINE = "<total_tokens>{left} tokens left</total_tokens>"
SUMMARY_HEAD = (
    "Summary of the work so far (the earlier messages were removed to save room):"
)
RULE = "━━"
# The project's figure for shared-prefix repeats, quoted with its source.
DRY_RUN_QUOTE = (
    "the project says: '11.8% of the dry run's trainable tokens were such",
    "repeats, all in reward-0 rollouts' (mhrl/training/common.py, the",
    "docstring of dedupe_shared_prefixes; its dry run is runs/dry/grpo)",
)
DRY_RUN = "dry-run"


@dataclass(frozen=True)
class Quirks:
    """How the harness misbehaves in one case (all off: the clean chain).

    Attributes:
        retry_call: Send this call twice; the first reply is never used.
        changing_line: Rewrite the system prompt's last line before every call.
        compact_after: After this call, replace the history with a summary.
        side_call_after: After this call, make one request with no tools.
    """

    retry_call: int = 0
    changing_line: bool = False
    compact_after: int = 0
    side_call_after: int = 0


@dataclass(frozen=True)
class Case:
    """One scripted conversation.

    Attributes:
        name: Its name on the command line.
        title: What it shows, in a few words.
        replies: What the scripted model answers, request by request.
        quirks: How the harness behaves.
        broken_engine: True to point the session at the broken engine.
    """

    name: str
    title: str
    replies: tuple[Reply, ...]
    quirks: Quirks = Quirks()
    broken_engine: bool = False


CASES = (
    Case("clean", "a clean 4-call chain", (TABLES, COUNT, WRITE, ANSWER)),
    Case(
        "retry",
        "call 2 sent twice: the first reply is abandoned",
        (TABLES, ABANDONED_COUNT, COUNT, WRITE, ANSWER),
        Quirks(retry_call=2),
    ),
    Case(
        "retry-last",
        "the last call sent twice: both replies end the conversation",
        (TABLES, COUNT, WRITE, ABANDONED_ANSWER, ANSWER),
        Quirks(retry_call=4),
    ),
    Case(
        "changing-line",
        "the system prompt's last line changes on every call",
        (TABLES, COUNT, WRITE, ANSWER),
        Quirks(changing_line=True),
    ),
    Case(
        "compaction",
        "after call 2 the history is replaced by a summary",
        (TABLES, COUNT, WRITE, ANSWER),
        Quirks(compact_after=2),
    ),
    Case(
        "no-logprobs",
        "the engine serves call 3 without its log-probabilities",
        (TABLES, COUNT, WRITE_UNSCORED, ANSWER),
    ),
    Case(
        "side-call",
        "after call 1, the harness asks for a title (no tools)",
        (TABLES, TITLE, COUNT, WRITE, ANSWER),
        Quirks(side_call_after=1),
    ),
    Case(
        "broken-engine",
        "an engine that drops log-probabilities whenever tools are offered",
        (TABLES, COUNT, WRITE, ANSWER),
        broken_engine=True,
    ),
)
# "retry" on the command line runs both retries, one after the other.
GROUPS = {"retry": ("retry", "retry-last")}
CHOICES = ("all", *(c.name for c in CASES if c.name != "retry-last"), DRY_RUN)


@dataclass
class HarnessRun:
    """What the harness did in one case.

    Attributes:
        events: One line per request (or other step), to print.
        calls: Agent calls whose reply the harness used.
        grade: The verdict on answer.txt.
    """

    events: list[str] = field(default_factory=list)
    calls: int = 0
    grade: Grade | None = None


@dataclass(frozen=True)
class Outcome:
    """One case's numbers, for the closing table.

    Attributes:
        case: The case.
        grade: The verdict.
        trainer: The rollout document the trainer gets.
    """

    case: Case
    grade: Grade
    trainer: dict[str, Any]


def reply_text(message: dict[str, Any]) -> str:
    """Say what a reply held: its tool call, else its text.

    Args:
        message: The assistant message.

    Returns:
        Such as ``bash: sqlite3 data.db .tables``.
    """
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0]["function"]
        try:
            command = json.loads(function["arguments"]).get("command")
        except (json.JSONDecodeError, AttributeError):
            command = function["arguments"]
        return clip(f"{function['name']}: {command}", 60)
    return clip(f"text: {message.get('content')}", 60)


def with_changing_line(messages: list[dict[str, Any]], base: str, left: int) -> None:
    """Rewrite the system prompt with a last line that counts down.

    Args:
        messages: The conversation; its first message is the system prompt.
        base: The system prompt without the line.
        left: The number the line shows.
    """
    line = REMINDER_LINE.format(left=left)
    messages[0] = {"role": SYSTEM, "content": f"{base}\n\n{line}"}


def summary_message(observations: list[tuple[str, str]]) -> dict[str, Any]:
    """Write the summary a compacting harness puts in place of the history.

    Args:
        observations: ``(command, output)`` for each tool call so far.

    Returns:
        A user message.
    """
    lines = [SUMMARY_HEAD]
    lines += [f"- ran {command}: {output.strip()}" for command, output in observations]
    return {"role": USER, "content": "\n".join(lines)}


def side_call(chat: Chat, task: Task) -> dict[str, Any]:
    """Ask for a title in a separate request, with no tools.

    Args:
        chat: Sends one request.
        task: The question the title is for.

    Returns:
        The response.
    """
    messages = [
        {"role": SYSTEM, "content": TITLE_PROMPT},
        {"role": USER, "content": task.question},
    ]
    body = {"model": MODEL_NAME, "messages": messages, "max_tokens": SIDE_CALL_TOKENS}
    return chat(body)


def run_harness(chat: Chat, task: Task, workdir: Path, quirks: Quirks) -> HarnessRun:
    """Solve the task through the proxy, misbehaving as the quirks say.

    Args:
        chat: Sends one request to the proxy, with the session's key.
        task: The question.
        workdir: A fresh folder holding its database.
        quirks: How to misbehave.

    Returns:
        What the harness did, graded.
    """
    run = HarnessRun()
    tools = [bash_schema()]
    messages = start_messages(task)
    base_system = messages[0]["content"]
    observations: list[tuple[str, str]] = []
    used = 0  # tokens read and written so far, for the changing line
    for number in range(1, MAX_CALLS + 1):
        if quirks.changing_line:
            with_changing_line(messages, base_system, TOKEN_BUDGET - used)
        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "temperature": TRAIN_TEMPERATURE,
            "max_tokens": OUTPUT_CAP,
        }
        response = chat(body)
        if number == quirks.retry_call:
            dropped = reply_text(response["choices"][0]["message"])
            run.events.append(f"call {number}  {dropped}   <- abandoned: sent again")
            response = chat(body)  # the same request, once more
        usage = response["usage"]
        used += usage["prompt_tokens"] + usage["completion_tokens"]
        message = response["choices"][0]["message"]
        run.calls += 1
        run.events.append(f"call {number}  {reply_text(message)}")
        tool_calls = message.get("tool_calls") or []
        kept = {"role": ASSISTANT, "content": message.get("content")}
        if tool_calls:
            kept["tool_calls"] = tool_calls
        messages.append(kept)
        for call in tool_calls:
            result = execute(call, tools, workdir)
            messages.append(tool_message(call["id"], result.output))
            command = json.loads(call["function"]["arguments"]).get("command", "")
            observations.append((command, result.output))
        if not tool_calls:
            break  # the final answer
        if number == quirks.compact_after:
            messages = [*start_messages(task), summary_message(observations)]
            run.events.append(
                f"compact the history is replaced: {len(messages)} messages, "
                "the last a summary"
            )
        if number == quirks.side_call_after:
            title = side_call(chat, task)["choices"][0]["message"]
            run.events.append(f"side    no tools · {reply_text(title)}")
    run.grade = grade(task, workdir)
    return run


def proxy_upstream(client: CaptureClient) -> str:
    """Ask the proxy which engine it is in front of, and check it is scripted.

    Args:
        client: The proxy's session client.

    Returns:
        The engine's root.

    Raises:
        SystemExit: If the proxy is down or the engine is not the scripted one.
    """
    try:
        upstream = client.health()["upstream"]
        ScriptClient(upstream).state()
    except ProxyError as error:
        reason = f"{error}: start the stack with ./start.sh scripted"
        raise SystemExit(reason) from error
    except ScriptError as error:
        raise SystemExit(
            f"{error}. This script needs the scripted engine: ./start.sh stop, "
            "then ENGINE_PORT=8119 ./start.sh scripted"
        ) from error
    return upstream


def open_case_session(
    client: CaptureClient, case: Case, engine_url: str
) -> tuple[str, str]:
    """Open the case's training session, on the broken engine when it asks.

    Args:
        client: The proxy's session client.
        case: The case.
        engine_url: The scripted engine's root.

    Returns:
        The session id, and a line saying what the proxy measured.
    """
    labels = {"task_id": DEFAULT_TASK, "harness": LABEL, "case": case.name}
    if not case.broken_engine:
        session = client.open_session(PURPOSE_TRAIN, labels)
        return session.session_id, (
            f"session {session.session_id} · {session.rollout_type} · capture level "
            f"{session.capture_level}"
        )
    broken = engine_url + BROKEN_PATH
    # Naming an engine makes the proxy probe it before it answers.
    session = client.open_session(
        PURPOSE_TRAIN, labels, llm_url=broken, model=MODEL_NAME
    )
    return session.session_id, (
        f"session {session.session_id} names the engine {broken}\n"
        f"{INDENT}the proxy probed it first: capture level {session.capture_level} "
        f"({session.rollout_type})"
    )


def print_project(document: dict[str, Any]) -> None:
    """Print what the project's functions make of the trainer's document.

    Args:
        document: The rollout document the trainer gets.
    """
    rows = trainable_sequences(document)
    deduped = dedupe_shared_prefixes(rows)
    items = [item_from_sequence(row) for row in deduped]
    positions = sum(item.n_tokens for item in items if item is not None)
    print(
        f"project   count_calls {count_calls(document)} · n_sampled "
        f"{n_sampled(document):,} · rows kept {len(rows)}"
    )
    print(
        f"{INDENT}tokens with mask 1: {trained_tokens(rows):,} · after "
        f"dedupe_shared_prefixes {trained_tokens(deduped):,} · item_from_sequence "
        f"positions {positions:,}"
    )
    for number, (row, after) in enumerate(zip(rows, deduped, strict=True), start=1):
        ids = row["input_ids"]
        earlier = [shared_prefix(ids, e["input_ids"]) for e in rows[: number - 1]]
        masked = trained_tokens([row]) - trained_tokens([after])
        if earlier and masked:
            print(
                f"{INDENT}row {number} shares its first {max(earlier):,} tokens with "
                f"an earlier row: {masked:,} of its mask-1 tokens are trained there"
            )


def run_case(
    client: CaptureClient,
    engine_url: str,
    case: Case,
    painter: Painter,
    number: int,
) -> Outcome:
    """Run one case end to end and print everything the proxy built.

    Args:
        client: The proxy's session client.
        engine_url: The scripted engine's root.
        case: The case.
        painter: Colours, or none.
        number: The case's number, for the heading.

    Returns:
        The case's numbers.

    Raises:
        SystemExit: If the proxy or the engine stops answering.
    """
    print(painter.paint(f"\n{RULE} {number} · {case.name} · {case.title}", "head"))
    task = find_task(DEFAULT_TASK)
    workdir = prepare_workdir(task, work_folder(LABEL, case.name))
    session_id, opened = open_case_session(client, case, engine_url)
    script = ScriptClient(engine_url + (BROKEN_PATH if case.broken_engine else ""))
    try:
        script.clear()
        script.push(list(case.replies))
        print(f"harness   {opened}")
        chat = http_chat(client.proxy_url + OPENAI_ROOT, session_id)
        try:
            run = run_harness(chat, task, workdir, case.quirks)
        except EngineError as error:
            raise SystemExit(f"the proxy did not answer: {error}") from error
        left = script.state()["queued"]
        if left:  # the conversation did not go as scripted
            print(f"{INDENT}note: {left} scripted replies were never asked for")
            script.clear()
        for event in run.events:
            print(f"{INDENT}{event}")
        verdict = "correct" if run.grade.correct else "wrong"
        print(f"{INDENT}grade   {verdict} (answer.txt {run.grade.submitted!r})")
        trainer = client.rollout(session_id)
        full = client.rollout(session_id, include_discarded=True, include_messages=True)
    finally:
        with contextlib.suppress(ProxyError):
            client.close(session_id)
    save(case, full)
    print_graph(full, painter)
    print_rows(full, painter)
    print_findings(trainer, painter)
    print_stats(trainer)
    print_project(trainer)
    return Outcome(case, run.grade, trainer)


def save(case: Case, document: dict[str, Any]) -> Path:
    """Keep the proxy's full document for one case.

    Args:
        case: The case.
        document: The rollout document, every row and the messages included.

    Returns:
        The file.
    """
    path = rollout_path(LABEL, case.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return path


def print_table(outcomes: list[Outcome]) -> None:
    """Print one line per case: the task's grade beside what training gets.

    Args:
        outcomes: The cases run.
    """
    print(f"\n{RULE} every case solved the same task the same way")
    print(
        f"{INDENT}{'case':<14} {'grade':<8} turns roots rows row-tokens mask-1 "
        "trainable n_calls n_sampled"
    )
    for outcome in outcomes:
        doc = outcome.trainer
        stats = doc["stats"]
        verdict = "correct" if outcome.grade.correct else "wrong"
        row_tokens = sum(len(row["input_ids"]) for row in doc["sequences"])
        print(
            f"{INDENT}{outcome.case.name:<14} {verdict:<8} {stats['n_turns']:>5} "
            f"{stats['n_roots']:>5} {stats['n_sequences']:>4} {row_tokens:>10,} "
            f"{stats['n_trainable_tokens']:>6,} {str(doc['trainable']).lower():>9} "
            f"{count_calls(doc):>7} {n_sampled(doc):>9,}"
        )


def remeasure_dry_run(path: Path) -> None:
    """Re-measure the project's figure: mask-1 tokens the dedup removes.

    Args:
        path: A project ``rollouts.jsonl`` (``RolloutRecord`` lines, with
            their trainable sequences).

    Raises:
        SystemExit: If the file cannot be read.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error
    before = after = forked = repeats_unrewarded = 0
    for record in records:
        rows = record.get("sequences") or []
        kept = trained_tokens(rows)
        left = trained_tokens(dedupe_shared_prefixes(rows))
        before += kept
        after += left
        if kept > left:
            forked += 1
            if not record.get("reward"):
                repeats_unrewarded += kept - left
    repeats = before - after
    share = repeats / before if before else 0.0
    print(
        f"{INDENT}re-measured on {path}:\n"
        f"{INDENT}{len(records)} rollouts, {forked} with rows that share trained "
        f"tokens · {before:,} mask-1 tokens, {repeats:,} of them repeats "
        f"({share:.1%}), {repeats_unrewarded:,} in rollouts with reward 0"
    )


def print_dry_run(path: Path | None) -> None:
    """Print the project's shared-prefix figure, and re-measure it when given its file.

    Args:
        path: A project ``rollouts.jsonl``, or None to only quote the figure.
    """
    print(f"\n{RULE} {DRY_RUN} · shared-prefix repeats in the project's own run")
    for line in DRY_RUN_QUOTE:
        print(f"{INDENT}{line}")
    if path is not None:
        remeasure_dry_run(path)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", nargs="?", default="all", choices=CHOICES)
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--color", default="auto", choices=COLOR_CHOICES)
    parser.add_argument(
        "--dry-run",
        type=Path,
        default=None,
        metavar="FILE",
        help="a project rollouts.jsonl to re-measure the shared-prefix figure from",
    )
    return parser.parse_args()


def chosen_cases(choice: str) -> list[Case]:
    """The cases a command-line choice names.

    Args:
        choice: ``all``, a case's name, or ``dry-run``.

    Returns:
        The cases, in the chapter's order.
    """
    if choice == DRY_RUN:
        return []
    names = GROUPS.get(choice, (choice,))
    return [case for case in CASES if choice == "all" or case.name in names]


def main() -> None:
    """Run the chosen cases through the proxy and print what it built."""
    args = parse_args()
    painter = Painter(args.color)
    cases = chosen_cases(args.case)
    outcomes = []
    if cases:
        client = CaptureClient(args.proxy_url)
        engine_url = proxy_upstream(client)
        print(
            f"{SCRIPT} · the capture proxy at {client.proxy_url}, in front of the "
            f"scripted engine at {engine_url}"
        )
        for case in cases:
            number = CASES.index(case) + 1
            try:
                outcomes.append(run_case(client, engine_url, case, painter, number))
            except (ProxyError, ScriptError) as error:
                raise SystemExit(f"case {case.name}: {error}") from error
            if case.name == "retry-last" and args.case in ("all", "retry"):
                print_dry_run(args.dry_run)
        saved = rollout_path(LABEL, cases[-1].name).parent
        print(f"\nsaved: every case's full rollout document in {short_path(saved)}")
    if args.case == DRY_RUN:
        print_dry_run(args.dry_run)
    if len(outcomes) > 1:
        print_table(outcomes)


if __name__ == "__main__":
    main()
