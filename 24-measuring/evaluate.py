"""Pass@1 on held-out tasks, in every harness: the project's evaluation, copied.

The project's ``mhrl/evaluate.py`` (``evaluate``, ``summarize``,
``format_table``) and ``mhrl/cli.py`` (``balanced``), with one change each
where this laptop differs, marked where it happens:

* **balanced**: up to ``limit`` tasks, round-robin over (database,
  difficulty), so 24 tasks are 4 from each of the six buckets;
* **run_rollout**: one (task, harness) **cell**, built from chapter 23's
  ``run_one.py``: a fresh task folder, the harness's private state in a
  fresh folder in the system's temp folder (where the project puts it), an
  evaluation session (temperature 0, an 8-call budget; the engine caps each
  reply at 1,024 tokens), the harness in the sandbox through the proxy, the
  grade, the proxy's count of calls, and any answer file left outside the
  task folder;
* **evaluate**: every cell once, harness by harness, each record appended to
  a JSON-lines file the moment its rollout ends (the project appends all of
  them at the end, so a crash lost the lot), then ``summarize``;
* **summarize**: accuracy, calls per solve and timeouts per harness, and the
  plain average over harnesses (each harness weighs the same).

The project runs four cells at a time (``settings.concurrency``) against one
engine thread; this copy runs one at a time, so the laptop stays usable and
the scripted engine's one queue of replies serves one harness at a time.
Only Python's standard library is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from capture_client import PURPOSE_EVAL, CaptureClient
from grader import grade
from lfm_files import MAX_CALLS, chapter_home, tilde
from run_one import (
    RUNS_FOLDER,
    own_log,
    prepare_run,
    queue_script,
    read_facts,
    run_boxed,
    save_run,
)
from tasks import ANSWER_FILE, Task, task_instruction
from wire import Options, RunFolders, Wiring, wire

PERCENT = 100.0
LABEL_WIDTH = 12  # the project's label column
HARNESS_COLUMN = 14  # "mini-swe-agent", the longest harness name
TASK_COLUMN = 18  # "airline-medium-001", the longest task id
CALLS_COLUMN = 8  # "8 calls"
STATE_PREFIX = "mhrl-harness-"  # the project's mkdtemp prefix (mhrl/harnesses/base.py)
STRAY_SHOWN = 60  # characters of a misplaced answer kept in the record
# Claude Code's auto-memory folder is <state>/claude/projects/<the cwd spelt
# as one name, such as -Users-you-...>/memory/. That name holds your user
# name, so it is shown as <folder>.
CWD_NAME_START = "-"
PROJECT_FOLDER = "<folder>"
EVAL_FOLDER = "eval"  # COURSE_HOME/24-measuring/runs/eval/<label>/<harness>/<task>
# What ended a run, in two or three words, from run_one's longer sentence.
STOP_LABELS = (
    ("the time limit", "time limit"),
    ("the context window", "window full"),
    ("mini-swe-agent's finishing command", "finish command"),
    ("mini-swe-agent's step_limit", "step limit"),
    ("3 replies in a row", "3 bad replies"),
    ("the proxy's", f"{MAX_CALLS}-call budget"),
    ("a reply with no tool call", "no tool call"),
)


@dataclass
class Record:
    """One cell's result, in the project's ``RolloutRecord`` field names.

    The first eleven fields are the project's (``mhrl/rollout.py``) without
    its training-only ones; ``engine`` is the checkpoint's label, as there.
    The rest are what this chapter adds to read a run afterwards.

    Attributes:
        task_id: The task.
        harness: The harness.
        purpose: ``eval`` (greedy, never trained on).
        correct: Whether the grader accepted ``answer.txt``.
        submitted: The answer judged, or None when there was no file.
        reason: The grader's explanation.
        n_calls: The harness's model calls (the project's ``count_calls``).
        n_sampled: Tokens the model wrote across those calls.
        seconds: Wall time of the harness's run.
        timed_out: Whether the time limit killed it.
        session_id: The capture session.
        engine: The checkpoint's label, such as ``base``.
        domain: The task's database.
        difficulty: ``easy``, ``medium`` or ``hard``.
        version: The harness version the run was checked against.
        stop: What ended the run, in a few words.
        first_prompt: Prompt tokens of the first agent call, or None.
        loopback_only: Whether lsof saw only this machine's addresses.
        answer_elsewhere: An answer file the harness wrote in its state
            folder instead of the task folder, and what it said.
    """

    task_id: str
    harness: str
    purpose: str = PURPOSE_EVAL
    correct: bool = False
    submitted: str | None = None
    reason: str = ""
    n_calls: int = 0
    n_sampled: int = 0
    seconds: float = 0.0
    timed_out: bool = False
    session_id: str = ""
    engine: str = "base"
    domain: str = ""
    difficulty: str = ""
    version: str = ""
    stop: str = ""
    first_prompt: int | None = None
    loopback_only: bool | None = None
    answer_elsewhere: str | None = None

    @property
    def cell(self) -> tuple[str, str]:
        """The (harness, task) pair this record answers.

        Returns:
            ``(harness, task_id)``.
        """
        return self.harness, self.task_id


def record_from(data: dict[str, Any]) -> Record:
    """Build a record from one JSON line, ours or the project's.

    Fields this chapter does not keep (the project's ``reward``,
    ``sequences`` and so on) are dropped; fields the project does not write
    keep their defaults.

    Args:
        data: One decoded line.

    Returns:
        The record.

    Raises:
        ValueError: If the line has no task or no harness.
    """
    known = {item.name for item in fields(Record)}
    kept = {key: value for key, value in data.items() if key in known}
    if not kept.get("task_id") or not kept.get("harness"):
        raise ValueError("a record needs a task_id and a harness")
    return Record(**kept)


def read_records(path: Path) -> list[Record]:
    """Read a JSON-lines file of records.

    Args:
        path: The file.

    Returns:
        Its records, in order ([] when the file does not exist).

    Raises:
        ValueError: If a line is not a JSON object with a task and a harness
            (the message names the line).
    """
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            records.append(record_from(data))
        except ValueError as error:
            raise ValueError(f"{tilde(str(path))}:{number}: {error}") from error
    return records


def append_record(path: Path, record: Record) -> None:
    """Append one record to a JSON-lines file, and flush it to disk.

    The line goes out in one write and is forced to the disk, so a run that
    is stopped (Ctrl-C, a crash, the power) loses at most the cell it was in.
    A home folder an answer may name is written ``~``.

    Args:
        path: The file (created with its folder if needed).
        record: The record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(tilde(json.dumps(asdict(record))) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def drop_torn_tail(path: Path) -> bool:
    """Cut a half-written last line, left by a run stopped mid-write.

    Every complete record ends with a newline, so anything after the last
    newline is a line that never finished. Cutting it lets ``--resume`` read
    the file and run that cell again.

    Args:
        path: The records file.

    Returns:
        True when something was cut.
    """
    if not path.is_file():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    with path.open("r+b") as handle:
        handle.truncate(data.rfind(b"\n") + 1)
    return True


def balanced(tasks: list[Task], limit: int | None) -> list[Task]:
    """Take up to ``limit`` tasks, round-robin over (database, difficulty).

    The project's ``cli.balanced``, unchanged in what it picks (it copies
    each bucket before taking from it): a plain ``tasks[:limit]``
    depends on the file's order, and a file sorted by name would give the
    first database's tasks only.

    Args:
        tasks: Candidate tasks.
        limit: How many to keep (None keeps all).

    Returns:
        A balanced subset, deterministic for a given input order.
    """
    if limit is None or limit >= len(tasks):
        return tasks
    buckets: dict[tuple[str, str], list[Task]] = {}
    for task in tasks:
        buckets.setdefault((task.domain, task.difficulty), []).append(task)
    queues = [list(buckets[key]) for key in sorted(buckets)]
    picked: list[Task] = []
    while len(picked) < limit:
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def summarize(records: list[Record], harnesses: list[str]) -> dict[str, Any]:
    """Accuracy and tool calls per harness, plus the average over harnesses.

    The project's ``summarize``, unchanged.

    Args:
        records: Evaluation records.
        harnesses: Harness names, in display order.

    Returns:
        ``{"per_harness": {...}, "average_accuracy": float, "cells": int}``.
    """
    per: dict[str, Any] = {}
    for harness in harnesses:
        rows = [r for r in records if r.harness == harness]
        solved = [r for r in rows if r.correct]
        per[harness] = {
            "cells": len(rows),
            "solved": len(solved),
            "accuracy": PERCENT * len(solved) / len(rows) if rows else 0.0,
            "calls_when_solved": (
                statistics.fmean(r.n_calls for r in solved) if solved else None
            ),
            "timeouts": sum(1 for r in rows if r.timed_out),
        }
    scores = [v["accuracy"] for v in per.values() if v["cells"]]
    return {
        "per_harness": per,
        "average_accuracy": statistics.fmean(scores) if scores else 0.0,
        "cells": len(records),
    }


def format_table(summaries: dict[str, dict[str, Any]], harnesses: list[str]) -> str:
    """Render accuracy per harness as a plain-text table.

    The project's ``format_table``, with one change: the label column widens
    for a label longer than the project's (``scripted-mistakes``).

    Args:
        summaries: ``{label: summary}`` in display order.
        harnesses: Harness names.

    Returns:
        The table.
    """
    width = max(len(h) for h in [*harnesses, "average"]) + 2
    label_width = max([LABEL_WIDTH, *(len(label) + 2 for label in summaries)])
    header = "checkpoint".ljust(label_width)
    header += "".join(h.rjust(width) for h in harnesses)
    lines = [header + "average".rjust(width)]
    for label, summary in summaries.items():
        cells = [
            f"{summary['per_harness'][h]['accuracy']:.0f}%".rjust(width)
            for h in harnesses
        ]
        average = f"{summary['average_accuracy']:.0f}%".rjust(width)
        lines.append(label.ljust(label_width) + "".join(cells) + average)
    return "\n".join(lines)


def stop_label(stop: str) -> str:
    """Shorten run_one's account of what ended a run.

    Args:
        stop: Such as ``"the proxy's 8-call budget: it answered ..."``.

    Returns:
        Such as ``"8-call budget"``; the sentence's start for anything else.
    """
    for start, label in STOP_LABELS:
        if stop.startswith(start):
            return label
    return stop.split(":", 1)[0]


def cell_folder(label: str, harness: str, task: Task) -> Path:
    """Where one cell runs and keeps its evidence.

    Args:
        label: The checkpoint, such as ``base``.
        harness: The harness.
        task: The task.

    Returns:
        ``COURSE_HOME/24-measuring/runs/eval/<label>/<harness>/<task_id>``.
    """
    return chapter_home() / RUNS_FOLDER / EVAL_FOLDER / label / harness / task.task_id


@dataclass(frozen=True)
class CellSetup:
    """What every cell of one evaluation shares.

    Attributes:
        client: The capture proxy's client.
        proxy_url: The proxy's root, given to each harness.
        engine_url: The engine behind the proxy (the scripted engine's
            remote control is there).
        script: ``solve`` or ``mistakes`` on the scripted engine; ``none``
            with the model.
        label: The checkpoint's label, such as ``base``.
        versions: Each harness's checked version.
        options: The wiring's options (the project's: ``--tools Bash`` for
            Claude Code, Pi's ``intercept`` provider).
    """

    client: CaptureClient
    proxy_url: str
    engine_url: str
    script: str
    label: str
    versions: dict[str, str]
    options: Options = Options()


def run_rollout(task: Task, harness: str, setup: CellSetup) -> Record:
    """Run one task in one harness through the capture proxy, once.

    The project's ``run_rollout`` for an evaluation, made of chapter 23's
    ``run_one.py`` steps without its printing: fresh folders, a session, the
    wiring, the boxed run, the grade, the proxy's record. The cell's folder
    keeps the task folder, the harness's own log, the proxy's record and
    ``run.json``; the state folder is removed afterwards, as the project
    removes it.

    Args:
        task: The task.
        harness: The harness.
        setup: What every cell shares.

    Returns:
        The cell's record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If the cell's folder cannot be made or the harness cannot
            start.
        HarnessInstallError: If the sandbox cannot be applied (nothing runs).
    """
    folders = prepare_cell(setup.label, harness, task)
    try:
        return run_in(folders, task, harness, setup)
    finally:
        shutil.rmtree(folders.state, ignore_errors=True)  # as the project does


def prepare_cell(label: str, harness: str, task: Task) -> RunFolders:
    """Make a cell's folders: the task folder here, the state folder in the temp folder.

    The task folder (``data.db``, ``answer.txt``) and the evidence live in the
    cell's folder. The harness's private state (its HOME, TMPDIR and config)
    goes in a fresh folder in the system's temp folder, where the project
    puts it (``tempfile.mkdtemp(prefix="mhrl-harness-<name>-")`` in
    ``mhrl/harnesses/base.py``). Chapter 23 kept state beside the task folder.
    Measured here, that changed what Claude Code tells the model: its
    auto-memory folder lives in its config folder, so its prompt then named a
    writable ``.../<task>/state/claude/projects/...-task/memory/`` beside the
    task, and in 9 of 24 cells the model wrote its answer there.

    Args:
        label: The checkpoint, such as ``base``.
        harness: The harness.
        task: The task.

    Returns:
        The cell's folders, both empty but for ``data.db``.
    """
    beside = prepare_run(cell_folder(label, harness, task), task)
    # run_one makes a state folder beside the task folder: not used here.
    shutil.rmtree(beside.state, ignore_errors=True)
    state = Path(tempfile.mkdtemp(prefix=f"{STATE_PREFIX}{harness}-")).resolve()
    folders = RunFolders(task=beside.task, state=state)
    try:
        folders.make()
    except OSError:
        shutil.rmtree(state, ignore_errors=True)  # never leave it behind
        raise
    return folders


def stray_answer(state: Path) -> str | None:
    """Find an answer file the harness wrote in its state folder, not the task's.

    The grader reads only the task folder's ``answer.txt``, so an answer
    written anywhere else scores 0 whatever it says. Saying where it went
    tells a model failure from a misplaced file.

    Args:
        state: The cell's state folder.

    Returns:
        Such as ``'7' in claude/projects/<folder>/memory/answer.txt``, or
        None when there is no such file. A folder named after the cwd
        (Claude Code's ``-Users-<you>-...``) or holding your user name is
        written ``<folder>``, and your home folder ``~``.
    """
    user = Path.home().name
    for path in sorted(state.rglob(ANSWER_FILE)):
        # A link could point anywhere; only a file the harness wrote counts.
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        parts = [
            PROJECT_FOLDER if part.startswith(CWD_NAME_START) or user in part else part
            for part in path.relative_to(state).parts
        ]
        return tilde(f"{text[:STRAY_SHOWN]!r} in {'/'.join(parts)}")
    return None


def keep_log(log: Path, folders: RunFolders) -> Path:
    """Copy the harness's own log out of the state folder, before it is removed.

    Args:
        log: The log (mini-swe-agent's trajectory lives in the state folder).
        folders: The cell's folders.

    Returns:
        Where the log is kept: in the cell's folder.
    """
    if not log.is_relative_to(folders.state) or not log.is_file():
        return log
    kept = folders.task.parent / log.name
    shutil.copyfile(log, kept)
    return kept


def run_in(folders: RunFolders, task: Task, harness: str, setup: CellSetup) -> Record:
    """Run the cell in its folders: session, wiring, boxed run, grade, record.

    Args:
        folders: The cell's folders.
        task: The task.
        harness: The harness.
        setup: What every cell shares.

    Returns:
        The cell's record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If the harness cannot start.
        HarnessInstallError: If the sandbox cannot be applied (nothing runs).
    """
    labels = {"task_id": task.task_id, "harness": harness}
    session = setup.client.open_session(PURPOSE_EVAL, labels, MAX_CALLS)
    wiring: Wiring | None = None
    try:
        queue_script(setup.script, harness, task, setup.engine_url)
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
        if wiring is not None:  # the harness's own log, even when the run failed
            log = keep_log(own_log(wiring, folders), folders)
    facts = read_facts(harness, document, log, process)
    turns = [t for t in document.get("turns") or [] if not t.get("discarded")]
    record = Record(
        task_id=task.task_id,
        harness=harness,
        correct=verdict.correct,
        submitted=verdict.submitted,
        reason=verdict.reason,
        n_calls=facts.agent_calls,
        n_sampled=sum(int(t.get("n_sampled") or 0) for t in turns),
        seconds=round(process.seconds, 1),
        timed_out=process.exit_code is None,
        session_id=session.session_id,
        engine=setup.label,
        domain=task.domain,
        difficulty=task.difficulty,
        version=setup.versions.get(harness, ""),
        stop=stop_label(facts.stop),
        first_prompt=facts.first_prompt,
        loopback_only=process.network.loopback_only,
        answer_elsewhere=stray_answer(folders.state),
    )
    run = {"record": asdict(record), "facts": asdict(facts)}
    run["network"] = process.network.as_dict()
    try:
        save_run(folders, document, run)
    except OSError as error:  # the evidence is lost, never the record
        where = f"{harness} {task.task_id}"
        print(f"  evidence not kept for {where}: {tilde(str(error))}", file=sys.stderr)
    return record


def cell_line(index: int, total: int, record: Record) -> str:
    """One line of progress: the cell, its verdict, its calls and time.

    Args:
        index: The cell's number, from 1.
        total: How many cells the evaluation runs.
        record: Its record.

    Returns:
        Such as ``[ 3/72] mini-swe-agent  airline-medium-001  correct ...``.
    """
    width = len(str(total))
    verdict = "correct" if record.correct else "wrong  "
    answer = "none" if record.submitted is None else tilde(repr(record.submitted))
    calls = f"{record.n_calls} call" + ("" if record.n_calls == 1 else "s")
    elsewhere = " · an answer.txt outside the task folder"
    if not record.answer_elsewhere:
        elsewhere = ""
    return (
        f"[{index:>{width}}/{total}] {record.harness:<{HARNESS_COLUMN}} "
        f"{record.task_id:<{TASK_COLUMN}} {verdict} · {calls:<{CALLS_COLUMN}} · "
        f"{record.seconds:5.1f} s · {record.stop} · answer {answer}{elsewhere}"
    )


def evaluate(
    tasks: list[Task],
    harnesses: list[str],
    setup: CellSetup,
    out_file: Path,
    done: Iterable[tuple[str, str]] = (),
    say: Callable[[str], None] = print,
    before_cell: Callable[[], None] | None = None,
) -> list[Record]:
    """Run every (task, harness) cell once, appending each record as it ends.

    Cells run harness by harness, in the project's order. A cell already in
    ``done`` (an interrupted run being finished) is skipped. ``before_cell``
    runs before each cell that runs, so a long evaluation can wait while the
    machine is busy with something else.

    Args:
        tasks: The held-out tasks.
        harnesses: The harnesses.
        setup: What every cell shares.
        out_file: The JSON-lines file each record is appended to.
        done: Cells already recorded in ``out_file``.
        say: Where progress lines go.
        before_cell: Called before each cell runs (it may block, or raise
            to stop the run).

    Returns:
        The records this call made.

    Raises:
        ProxyError: If the proxy refuses a session or cannot be reached.
        OSError: If a cell cannot run or its record cannot be written.
        HarnessInstallError: If the sandbox cannot be applied.
    """
    jobs = [(task, harness) for harness in harnesses for task in tasks]
    skip = set(done)
    made = []
    for index, (task, harness) in enumerate(jobs, start=1):
        if (harness, task.task_id) in skip:
            say(f"[{index}/{len(jobs)}] {harness} {task.task_id}: already recorded")
            continue
        if before_cell is not None:
            before_cell()
        record = run_rollout(task, harness, setup)
        append_record(out_file, record)
        made.append(record)
        say(cell_line(index, len(jobs), record))
    return made
