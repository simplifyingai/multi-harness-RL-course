# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tokenizers==0.23.2",
# ]
# ///
"""Collect answer-key demonstrations through the real harnesses.

The project's ``collect_demonstrations`` (``mhrl/training/sft.py``): the
answer-key teacher plays the model inside each TRAINING harness
(mini-swe-agent and Claude Code; Pi stays held out), and the capture proxy
records every call in the student's own tokens, exactly as it would record
the student. Each (task, harness) pair is one rollout:

1. a fresh task folder, and the harness's private state (HOME, TMPDIR, its
   config) in a fresh folder in the system's temp folder, as chapter 24
   found necessary;
2. a **training** session at the proxy (``purpose: train``), naming the
   teacher's engine as this session's engine (``llm_url``): the proxy's
   other sessions keep theirs;
3. the harness, unmodified, in chapter 19's sandbox, wired by chapter 23's
   configuration only;
4. the grade, and the proxy's stitched rollout: ``input_ids`` +
   ``loss_mask`` per row, the mask 1 exactly on the tokens the teacher
   wrote and 0 on everything the harness wrote around them.

Four rollouts run at a time (the project's ``concurrency``), each with its
own folders and session, all sharing the teacher's one engine thread. Each
record is appended to ``COURSE_HOME/25-sft-in-harnesses/demos/demos.jsonl``
the moment its rollout ends, and at the end the file is put in job order
(each task, then each harness, the project's order), so the trainer's batches
never depend on which rollout ended first. Then the script prints how many
demonstrations are usable (correct and trainable) and one captured Claude
Code row, segment by segment, with its mask. One collector writes to the
file at a time: run a second one only after the first has finished.

Start the teacher and the proxy first (``./start.sh oracle``; no model is
loaded). ``--limit 60`` is the project's run (120 demonstrations),
``--fast`` is ``--limit 10``, ``--resume`` finishes an interrupted run.
``tokenizers`` decodes the row it shows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from baseline_eval import check_harness
from capture_client import PURPOSE_TRAIN, CaptureClient, ProxyError
from evaluate import STATE_PREFIX, balanced, drop_torn_tail, keep_log, stop_label
from grader import grade
from install import CLAUDE, MINI, HarnessInstallError
from lfm_files import (
    MAX_CALLS,
    TokenizerMissingError,
    chapter_home,
    lfm_tokenizer_file,
    short_path,
    tilde,
)
from run_one import (
    PROXY_URL,
    RUNS_FOLDER,
    own_log,
    prepare_run,
    read_facts,
    run_boxed,
    save_run,
)
from tasks import TRAIN_FILE, Task, load_tasks, task_instruction
from teacher import pattern_of
from wire import Options, RunFolders, Wiring, wire

TRAIN_HARNESSES = (MINI, CLAUDE)  # the project's TRAIN_HARNESSES; Pi is held out
# The next three are engine.py's (ORACLE_PORT, SERVED_MODEL, SERVES_TEACHER):
# this script never imports the engine, so keep them in step by hand.
ORACLE_URL = "http://127.0.0.1:8101"  # the project's teacher port
SERVED_MODEL = "lfm2.5-2.6b"  # the name the teacher's engine answers to
SERVES_TEACHER = "answer-key teacher"  # what its /health says it serves
TEACHER_LABEL = "answer-key-teacher"  # the project's TEACHER_LABEL
FULL_LIMIT = 60  # the project's DEMO_LIMIT (scripts/train_from_scratch.sh)
FAST_LIMIT = 10
CONCURRENCY = 4  # the project's settings.concurrency
DEMOS_FOLDER = "demos"  # COURSE_HOME/25-sft-in-harnesses/demos/demos.jsonl
DEMOS_FILE = "demos.jsonl"
TEMPORARY_SUFFIX = ".tmp"  # the rewrite in job order, before it replaces the file
HTTP_TIMEOUT_S = 10
HARNESS_COLUMN = 14
TASK_COLUMN = 19  # "library-medium-001" and a space
VERDICT_COLUMN = 11  # "untrainable"
MASK_READ, MASK_TRAINED = 0, 1
SHOWN_CHARS = 150  # a long harness segment shows its start and its end
TEACHER_CHARS = 400  # a teacher segment is shown whole up to this
HEAD_TAIL = SHOWN_CHARS // 2
YOUR_NAME = "<you>"  # the user's name, wherever a path still carries it


@dataclass
class DemoRecord:
    """One demonstration, in the project's ``RolloutRecord`` field names.

    The first fifteen fields are the project's (``mhrl/rollout.py``), the
    reward left out (the trainer never reads it for SFT); the rest are what
    this chapter adds to read a run afterwards.

    Attributes:
        task_id: The task.
        harness: The harness.
        purpose: ``train``: the proxy keeps every row for training.
        correct: Whether the grader accepted ``answer.txt``.
        submitted: The answer judged, or None when there was no file.
        reason: The grader's explanation.
        n_calls: The harness's model calls (the project's ``count_calls``).
        n_sampled: Tokens the teacher wrote across those calls.
        seconds: Wall time of the harness's run.
        timed_out: Whether the time limit killed it.
        trainable: The proxy's verdict on the whole rollout.
        n_trainable_tokens: Tokens a trainer may learn from.
        session_id: The capture session.
        engine: ``answer-key-teacher``.
        sequences: The trainable rows: ``input_ids``, ``loss_mask``,
            ``logprobs``.
        path: The teacher's path for this task (``teacher.pattern_of``).
        stop: What ended the run, in a few words.
        findings: The proxy's warnings about this rollout.
        loopback_only: Whether lsof saw only this machine's addresses.
    """

    task_id: str
    harness: str
    purpose: str = PURPOSE_TRAIN
    correct: bool = False
    submitted: str | None = None
    reason: str = ""
    n_calls: int = 0
    n_sampled: int = 0
    seconds: float = 0.0
    timed_out: bool = False
    trainable: bool = False
    n_trainable_tokens: int = 0
    session_id: str = ""
    engine: str = TEACHER_LABEL
    sequences: list[dict[str, Any]] = field(default_factory=list)
    path: str = ""
    stop: str = ""
    findings: list[str] = field(default_factory=list)
    loopback_only: bool | None = None

    @property
    def usable(self) -> bool:
        """Whether the trainer will learn from it: correct and trainable.

        Returns:
            True for a usable demonstration.
        """
        return self.correct and self.trainable


@dataclass(frozen=True)
class DemoSetup:
    """What every demonstration of one run shares.

    Attributes:
        client: The capture proxy's client.
        proxy_url: The proxy's root, given to each harness.
        oracle_url: The teacher's engine, named in each session.
        options: The wiring's options (the project's).
    """

    client: CaptureClient
    proxy_url: str
    oracle_url: str
    options: Options = Options()


@dataclass(frozen=True)
class Segment:
    """A run of tokens in a row with one mask value.

    Attributes:
        mask: 1 for the teacher's tokens, 0 for the harness's text.
        start: The first position.
        token_ids: The tokens.
    """

    mask: int
    start: int
    token_ids: tuple[int, ...]


def records_path() -> Path:
    """Where the demonstrations are kept.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses/demos/demos.jsonl``.
    """
    return chapter_home() / DEMOS_FOLDER / DEMOS_FILE


def demo_folder(harness: str, task: Task) -> Path:
    """Where one demonstration runs and keeps its evidence.

    Args:
        harness: The harness.
        task: The task.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses/runs/demos/<harness>/<task_id>``.
    """
    return chapter_home() / RUNS_FOLDER / DEMOS_FOLDER / harness / task.task_id


def prepare_demo(harness: str, task: Task) -> RunFolders:
    """Make a demonstration's folders: the task here, the state in the temp folder.

    Chapter 24's ``prepare_cell`` for a demonstration: the task folder and
    the evidence in the demonstration's folder, the harness's private state
    in a fresh folder under ``$TMPDIR`` (the project's
    ``mkdtemp(prefix="mhrl-harness-<name>-")``), so no folder of the
    harness's own sits inside the task's tree.

    Args:
        harness: The harness.
        task: The task.

    Returns:
        The folders, both empty but for ``data.db``.

    Raises:
        OSError: If a folder cannot be made.
    """
    beside = prepare_run(demo_folder(harness, task), task)
    shutil.rmtree(beside.state, ignore_errors=True)  # not used: state goes to $TMPDIR
    state = Path(tempfile.mkdtemp(prefix=f"{STATE_PREFIX}{harness}-")).resolve()
    folders = RunFolders(task=beside.task, state=state)
    try:
        folders.make()
    except OSError:
        shutil.rmtree(state, ignore_errors=True)
        raise
    return folders


def trainable_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of a rollout the proxy marks trainable, cut to what training reads.

    Args:
        document: The proxy's rollout document.

    Returns:
        ``{input_ids, loss_mask, logprobs}`` per trainable row.
    """
    return [
        {
            "input_ids": row["input_ids"],
            "loss_mask": row["loss_mask"],
            "logprobs": row["logprobs"],
        }
        for row in document.get("sequences") or []
        if row.get("trainable")
    ]


def collect_one(task: Task, harness: str, setup: DemoSetup) -> DemoRecord:
    """Run the teacher on one task in one harness, through the proxy, once.

    Args:
        task: A training task.
        harness: A training harness.
        setup: What every demonstration shares.

    Returns:
        The demonstration's record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If a folder cannot be made or the harness cannot start.
        HarnessInstallError: If the sandbox cannot be applied (nothing runs).
    """
    folders = prepare_demo(harness, task)
    try:
        return collect_in(folders, task, harness, setup)
    finally:
        shutil.rmtree(folders.state, ignore_errors=True)  # as the project does


def collect_in(
    folders: RunFolders, task: Task, harness: str, setup: DemoSetup
) -> DemoRecord:
    """Session, wiring, boxed run, grade and record, in a demonstration's folders.

    Args:
        folders: The demonstration's folders.
        task: The task.
        harness: The harness.
        setup: What every demonstration shares.

    Returns:
        The record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If the harness cannot start.
        HarnessInstallError: If the sandbox cannot be applied.
    """
    labels = {"task_id": task.task_id, "harness": harness}
    session = setup.client.open_session(
        PURPOSE_TRAIN, labels, MAX_CALLS, llm_url=setup.oracle_url, model=SERVED_MODEL
    )
    wiring: Wiring | None = None
    # process, verdict, document and log are set inside this block; any
    # error in it leaves the function, so below they always exist.
    try:
        wiring = wire(
            harness,
            folders,
            task_instruction(task),
            setup.proxy_url,
            session.session_id,
            setup.options,
        )
        wiring.write()
        process = run_boxed(wiring, folders, setup.options.timeout_s)
        verdict = grade(task, folders.task)
        document = setup.client.rollout(session.session_id)
    finally:
        setup.client.close(session.session_id)  # on every way out
        if wiring is not None:
            log = keep_log(own_log(wiring, folders), folders)
    facts = read_facts(harness, document, log, process)
    turns = [t for t in document.get("turns") or [] if not t.get("discarded")]
    stats = document.get("stats") or {}
    record = DemoRecord(
        task_id=task.task_id,
        harness=harness,
        correct=verdict.correct,
        submitted=verdict.submitted,
        reason=verdict.reason,
        n_calls=facts.agent_calls,
        n_sampled=sum(int(t.get("n_sampled") or 0) for t in turns),
        seconds=round(process.seconds, 1),
        timed_out=process.exit_code is None,
        trainable=bool(document.get("trainable")),
        n_trainable_tokens=int(stats.get("n_trainable_tokens") or 0),
        session_id=session.session_id,
        sequences=trainable_rows(document),
        path=pattern_of(task),
        stop=stop_label(facts.stop),
        findings=facts.findings,
        loopback_only=process.network.loopback_only,
    )
    run = {"facts": asdict(facts), "network": process.network.as_dict()}
    run["record"] = {k: v for k, v in asdict(record).items() if k != "sequences"}
    try:
        save_run(folders, document, run)
    except OSError as error:  # the evidence is lost, never the record
        where = f"{harness} {task.task_id}"
        print(f"  evidence not kept for {where}: {tilde(str(error))}", file=sys.stderr)
    return record


def append_record(path: Path, record: DemoRecord) -> None:
    """Append one record to the demonstrations file, and flush it to disk.

    Args:
        path: The file (created with its folder if needed).
        record: The record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_demos(path: Path) -> list[DemoRecord]:
    """Read a demonstrations file.

    Args:
        path: The file.

    Returns:
        Its records ([] when there is no file).

    Raises:
        ValueError: If a line is not a demonstration (the message names it).
    """
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(DemoRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"{short_path(path)}:{number}: {error}") from error
    return records


def demo_line(index: int, total: int, record: DemoRecord) -> str:
    """One line of progress: the demonstration, its verdict, calls, rows and time.

    Args:
        index: The demonstration's number, from 1.
        total: How many this run collects.
        record: Its record.

    Returns:
        Such as ``[ 3/4] claude-code    library-easy-001   usable · 5 calls ...``.
    """
    width = len(str(total))
    if record.usable:
        verdict = "usable"
    else:
        verdict = "wrong" if not record.correct else "untrainable"
    rows = len(record.sequences)
    tokens = sum(len(row["input_ids"]) for row in record.sequences)
    return (
        f"[{index:>{width}}/{total}] {record.harness:<{HARNESS_COLUMN}} "
        f"{record.task_id:<{TASK_COLUMN}}{verdict:<{VERDICT_COLUMN}} · "
        f"{record.n_calls} calls · "
        f"path {record.path:<6} · {rows} row{'' if rows == 1 else 's'}, "
        f"{tokens:,} tokens, {record.n_trainable_tokens:,} trained · "
        f"{record.seconds:5.1f} s · {record.stop}"
    )


@dataclass(frozen=True)
class Failure:
    """A demonstration that produced no record, and why.

    Attributes:
        task_id: The task.
        harness: The harness.
        reason: The error, or why it was not run.
    """

    task_id: str
    harness: str
    reason: str


def collect(
    jobs: list[tuple[Task, str]],
    setup: DemoSetup,
    out_file: Path,
    concurrency: int,
    say: Callable[[str], None] = print,
) -> tuple[list[DemoRecord], list[Failure]]:
    """Run the demonstrations, ``concurrency`` at a time, appending each as it ends.

    A rollout that fails gets a ``FAILED`` line and no record (``--resume``
    runs it again). When the proxy itself fails, every rollout not yet
    started is skipped and said so, instead of failing one after another.

    Args:
        jobs: ``(task, harness)`` pairs still to run.
        setup: What every demonstration shares.
        out_file: The demonstrations file.
        concurrency: Rollouts at a time.
        say: Where progress lines go.

    Returns:
        The records this call made, in the order they ended, and the
        demonstrations that made none.
    """
    lock = threading.Lock()
    proxy_down = threading.Event()
    made: list[DemoRecord] = []
    failed: list[Failure] = []

    def fail(task: Task, harness: str, reason: str) -> None:
        """Keep and print one demonstration that made no record.

        Args:
            task: The task.
            harness: The harness.
            reason: Why.
        """
        with lock:
            failed.append(Failure(task.task_id, harness, reason))
            say(f"FAILED {harness} {task.task_id}: {reason}")

    def one(numbered: tuple[int, tuple[Task, str]]) -> None:
        """Run one demonstration and keep its record, or its failure.

        Args:
            numbered: Its number, and its (task, harness).
        """
        index, (task, harness) = numbered
        if proxy_down.is_set():
            fail(task, harness, "not run: the proxy failed earlier")
            return
        try:
            record = collect_one(task, harness, setup)
        except ProxyError as error:
            proxy_down.set()
            fail(task, harness, tilde(str(error)))
            return
        except (OSError, HarnessInstallError) as error:
            fail(task, harness, tilde(str(error)))
            return
        with lock:  # one writer at a time: whole lines, in the order they end
            append_record(out_file, record)
            made.append(record)
            say(demo_line(index, len(jobs), record))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, enumerate(jobs, start=1)))
    return made, failed


def put_in_order(path: Path, order: list[tuple[str, str]]) -> None:
    """Rewrite the demonstrations file in job order, the project's order.

    Records are appended as rollouts END, four at a time, so the file's order
    depends on which harness was quicker. The trainer shuffles rows with a
    fixed seed, and the batches that shuffle makes depend on the order it
    starts from: left as they ended, two identical collections would train
    two different adapters. In job order (each task, then each harness, as
    the project's ``collect_demonstrations`` writes them) the file is the
    same on every run. The new file replaces the old one whole, at once, so a
    crash halfway leaves the old file. One collector at a time: a record
    another process appended between the read and the replace would be lost.

    Args:
        path: The demonstrations file (nothing to do when there is none:
            every rollout of a first run may have failed).
        order: ``(harness, task_id)`` pairs in job order; any record not in
            it (an earlier run's other tasks) follows, in its own order.

    Raises:
        ValueError: If a line is not a demonstration (the message names it).
    """
    if not path.is_file():
        return
    records = read_demos(path)  # a broken line is named, as everywhere else
    rank = {pair: index for index, pair in enumerate(order)}
    # A stable sort: records outside the order keep their file order, last.
    records.sort(key=lambda r: rank.get((r.harness, r.task_id), len(rank)))
    temporary = path.with_suffix(TEMPORARY_SUFFIX)
    lines = "".join(json.dumps(asdict(record)) + "\n" for record in records)
    temporary.write_text(lines, "utf-8")
    os.replace(temporary, path)


def segments(row: dict[str, Any]) -> list[Segment]:
    """Cut a row into runs of equal mask: the harness's text, the teacher's tokens.

    Args:
        row: ``{input_ids, loss_mask, ...}``.

    Returns:
        The runs, in order.
    """
    found: list[Segment] = []
    ids, mask = row["input_ids"], row["loss_mask"]
    start = 0
    for position in range(1, len(ids) + 1):
        if position == len(ids) or mask[position] != mask[start]:
            found.append(Segment(mask[start], start, tuple(ids[start:position])))
            start = position
    return found


def scrub(text: str) -> str:
    """Make a decoded segment safe to print: no home folder, no user name.

    Claude Code's prompt names its working folder, which lives under your
    home folder.

    Args:
        text: Decoded tokens.

    Returns:
        The text, the home folder as ``~``, your user name as ``<you>``
        where a path still carries it (Claude Code spells its project
        folder ``-Users-<name>-...``), newlines written ``\\n``.
    """
    user = Path.home().name
    text = tilde(text)
    # Only inside paths: a short user name could be part of ordinary words.
    for before, after in (("/", "/"), ("-", "-")):
        text = text.replace(f"{before}{user}{after}", f"{before}{YOUR_NAME}{after}")
    return text.replace("\n", "\\n")


def shown(text: str, limit: int) -> str:
    """Shorten a long segment to its start and its end.

    Args:
        text: The segment's text, scrubbed.
        limit: The most characters to show whole.

    Returns:
        The text, or its first and last characters around `` … ``.
    """
    if len(text) <= limit:
        return text
    return f"{text[:HEAD_TAIL]} … {text[-HEAD_TAIL:]}"


def show_row(
    record: DemoRecord,
    decode: Callable[[list[int]], str],
    say: Callable[[str], None] = print,
) -> None:
    """Print one demonstration's first row, segment by segment, with its mask.

    Args:
        record: A usable demonstration.
        decode: Token ids to text, special tokens kept.
        say: Where lines go.
    """
    row = record.sequences[0]
    mask = row["loss_mask"]
    trained = sum(1 for value in mask if value == MASK_TRAINED)
    say(
        f"one captured {record.harness} demonstration: {record.task_id} "
        f"(path {record.path}), {record.n_calls} calls, "
        f"{len(record.sequences)} row{'' if len(record.sequences) == 1 else 's'}"
    )
    say(
        f"  row 1: {len(mask):,} tokens · mask 1 (the teacher's tokens, "
        f"trained) {trained:,} · mask 0 (the harness's text, read only) "
        f"{len(mask) - trained:,} · {100 * trained / len(mask):.1f}% trained"
    )
    for segment in segments(row):
        limit = TEACHER_CHARS if segment.mask == MASK_TRAINED else SHOWN_CHARS
        text = shown(scrub(decode(list(segment.token_ids))), limit)
        say(f"  mask {segment.mask} · {len(segment.token_ids):>5,} tokens · {text}")


def summary_lines(records: list[DemoRecord], harnesses: list[str]) -> list[str]:
    """Per harness: usable demonstrations, calls, row lengths and trained share.

    Args:
        records: The demonstrations of this run's pairs.
        harnesses: The harnesses, in display order.

    Returns:
        One line per harness that has records.
    """
    lines = []
    for harness in harnesses:
        mine = [r for r in records if r.harness == harness]
        if not mine:
            continue
        usable = [r for r in mine if r.usable]
        rows = [row for r in usable for row in r.sequences]
        calls = Counter(r.n_calls for r in usable)
        spread = ", ".join(f"{n} calls x{count}" for n, count in sorted(calls.items()))
        if rows:
            length = statistics.fmean(len(row["input_ids"]) for row in rows)
            trained = sum(sum(row["loss_mask"]) for row in rows)
            total = sum(len(row["input_ids"]) for row in rows)
            shape = (
                f"{len(rows)} rows, {length:,.0f} tokens on average, "
                f"{100 * trained / total:.1f}% trained"
            )
        else:
            shape = "no rows"
        lines.append(
            f"  {harness:<{HARNESS_COLUMN}} {len(usable)}/{len(mine)} usable · "
            f"{spread or 'no usable demonstration'} · {shape}"
        )
    return lines


def teacher_health(url: str) -> dict[str, Any]:
    """Ask an engine what it serves.

    Args:
        url: The engine's root.

    Returns:
        Its ``/health`` answer.

    Raises:
        SystemExit: If nothing answers, or it is not the teacher.
    """
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=HTTP_TIMEOUT_S) as reply:
            health = json.loads(reply.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise SystemExit(
            f"no teacher answers on {url}: start it with ./start.sh oracle "
            f"(and --oracle-url if it is on another port) · {error}"
        ) from error
    if not isinstance(health, dict) or health.get("serves") != SERVES_TEACHER:
        known = health if isinstance(health, dict) else {}
        served = known.get("serves") or known.get("model") or "something else"
        raise SystemExit(f"{url} serves {served}, not the {SERVES_TEACHER}")
    return health


def harness_list(value: str) -> list[str]:
    """Read ``--harnesses``: a comma-separated list of known harnesses.

    Args:
        value: Such as ``mini-swe-agent,claude-code``.

    Returns:
        The harnesses.

    Raises:
        argparse.ArgumentTypeError: For a name that is not a training
            harness, or an empty list.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    # Pi is held out: a demonstration in it would be trained on, and its
    # evaluation would no longer test a harness the model never saw.
    unknown = [name for name in names if name not in TRAIN_HARNESSES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"choose from the training harnesses {', '.join(TRAIN_HARNESSES)} "
            f"(got {value!r}; Pi is held out)"
        )
    return names


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="training tasks")
    parser.add_argument("--fast", action="store_true", help=f"--limit {FAST_LIMIT}")
    parser.add_argument("--harnesses", type=harness_list, default=list(TRAIN_HARNESSES))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--oracle-url", default=ORACLE_URL)
    parser.add_argument(
        "--resume", action="store_true", help="finish an interrupted run"
    )
    parser.add_argument("--show", default=CLAUDE, help="the harness whose row is shown")
    args = parser.parse_args()
    if args.limit is None:
        args.limit = FAST_LIMIT if args.fast else FULL_LIMIT
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def kept_demos(out_file: Path, resume: bool) -> list[DemoRecord]:
    """The records already in the file, refusing to mix two runs by accident.

    Args:
        out_file: The demonstrations file.
        resume: Whether ``--resume`` was given.

    Returns:
        The records to keep ([] for a fresh run).

    Raises:
        SystemExit: If the file has records and ``--resume`` was not given,
            or a line is broken.
    """
    if resume:
        drop_torn_tail(out_file)  # a line cut off mid-write runs again
    try:
        kept = read_demos(out_file)
    except ValueError as error:
        raise SystemExit(f"collect.py: {error}") from error
    if kept and not resume:
        raise SystemExit(
            f"{short_path(out_file)} already holds {len(kept)} demonstrations: "
            "--resume to finish that run, or move the file away to start again"
        )
    return kept


def decoder() -> Callable[[list[int]], str] | None:
    """LFM2.5's tokenizer, to decode a captured row (None when it is not on disk).

    Returns:
        A function from token ids to text, special tokens kept, or None.
    """
    try:
        path = lfm_tokenizer_file()
    except TokenizerMissingError:
        return None
    from tokenizers import Tokenizer  # only this display needs it

    tokenizer = Tokenizer.from_file(str(path))
    return lambda ids: tokenizer.decode(ids, skip_special_tokens=False)


def main() -> None:
    """Collect the demonstrations, then show what was captured.

    Raises:
        SystemExit: If the teacher or the proxy is down, a harness is
            refused, or the demonstrations file already exists without
            ``--resume``; and with status 1 after the report when a
            demonstration is unusable or failed, or a rollout's network
            watch saw more than this machine.
    """
    args = parse_args()
    started = time.perf_counter()
    tasks = balanced(load_tasks(TRAIN_FILE), args.limit)
    out_file = records_path()
    kept = kept_demos(out_file, args.resume)
    print("collect.py · answer-key demonstrations through the training harnesses")
    print(
        f"  tasks     {len(tasks)} training tasks, round-robin over database x "
        f"difficulty · harnesses {', '.join(args.harnesses)}"
    )
    print(
        f"  sessions  purpose train, at most {MAX_CALLS} calls, engine "
        f"{args.oracle_url} (the teacher) · {args.concurrency} rollouts at a time"
    )
    print(f"  records   {short_path(out_file)} (one line per rollout, as it ends)")
    health = teacher_health(args.oracle_url)
    print(f"  teacher   {health['serves']} at {args.oracle_url}")
    client = CaptureClient(args.proxy_url)
    try:
        client.health()
    except ProxyError as error:
        raise SystemExit(f"{error}: start the proxy (./start.sh oracle)") from error
    print("versions")
    options = Options()
    try:
        for harness in args.harnesses:
            check_harness(harness, tasks[0], options)
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    setup = DemoSetup(client, args.proxy_url, args.oracle_url, options)
    done = {(r.harness, r.task_id) for r in kept}
    jobs = [
        (task, harness)
        for task in tasks
        for harness in args.harnesses
        if (harness, task.task_id) not in done
    ]
    heading = f"demonstrations ({len(done)} already recorded)" if done else ""
    print(heading or "demonstrations")
    _, failed = collect(jobs, setup, out_file, args.concurrency)
    # Read back from the file: a resumed run's earlier records count too.
    order = [(h, t.task_id) for t in tasks for h in args.harnesses]
    try:
        put_in_order(out_file, order)
    except ValueError as error:
        raise SystemExit(f"collect.py: {error}") from error
    wanted = set(order)
    records = [r for r in read_demos(out_file) if (r.harness, r.task_id) in wanted]
    good = sum(1 for r in records if r.usable)
    print("")
    print(f"[demos] {good}/{len(records)} demonstrations usable")
    if failed:
        print(
            f"  {len(failed)} demonstrations made no record (FAILED above): "
            "--resume runs them again"
        )
    for line in summary_lines(records, args.harnesses):
        print(line)
    minutes, rest = divmod(round(time.perf_counter() - started), 60)
    print(f"  this run took {minutes} min {rest} s")
    outside = [r for r in records if r.loopback_only is False]
    if outside:
        print(f"network   NOT loopback-only in {len(outside)} rollouts")
    else:
        print("network   every rollout: lsof saw only this machine's addresses")
    shown_record = next(
        (r for r in records if r.harness == args.show and r.usable and r.sequences),
        None,
    )
    decode = decoder()
    print("")
    if shown_record is None:
        print(f"no usable {args.show} demonstration to show")
    elif decode is None:
        print("LFM2.5's tokenizer is not on disk, so no row is decoded")
    else:
        show_row(shown_record, decode)
    if good < len(records) or outside or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
