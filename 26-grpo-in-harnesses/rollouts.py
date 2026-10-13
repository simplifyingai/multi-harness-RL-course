"""Training rollouts through the real harnesses, for GRPO.

The project's ``run_rollout`` (``mhrl/rollout.py``) as ``train_grpo`` calls
it, built from chapter 25's collection (``collect_in``) with the model in
place of the teacher. One rollout is one attempt at one task in one harness:

1. a fresh task folder, and the harness's private state (HOME, TMPDIR, its
   config) in a fresh folder under ``$TMPDIR`` (the project's
   ``mkdtemp(prefix="mhrl-harness-<name>-")``), removed afterwards;
2. a **training** session at the capture proxy: the proxy samples every
   call at temperature 1 from the engine behind it, whatever the harness
   asks, and records each call's exact tokens and log-probabilities;
3. the harness, unmodified, in chapter 19's sandbox, wired by chapter 23's
   configuration only;
4. the grade, the reward (``rewards.py``) and the proxy's stitched rows.

A step runs its rollouts 4 at a time (the project's ``concurrency``), each
in its own folders and session, and returns them in job order whatever order
they end in, so the groups are always the same rollouts. Each rollout keeps
its evidence beside its task folder (the proxy's record, ``run.json``, the
harness's own log). This module needs only the standard library.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capture_client import PURPOSE_TRAIN, CaptureClient, ProxyError
from evaluate import STATE_PREFIX, keep_log, stop_label
from grader import grade
from install import CLAUDE, MINI, HarnessInstallError
from lfm_files import MAX_CALLS, chapter_home, tilde
from run_one import RUNS_FOLDER, own_log, prepare_run, read_facts, run_boxed, save_run
from tasks import Task, task_instruction
from wire import Options, RunFolders, Wiring, wire

TRAIN_HARNESSES = (MINI, CLAUDE)  # the project's TRAIN_HARNESSES; Pi is held out
CONCURRENCY = 4  # the project's settings.concurrency
# Each rollout's evidence: COURSE_HOME/26-grpo-in-harnesses/runs/grpo/<run>/...
EVIDENCE_FOLDER = "grpo"
ENGINE_LABEL = "model"  # the project's engine_label for the policy's rollouts
TASK_SEED = 0  # the project's train_grpo shuffles the tasks once with seed 0
HARNESS_COLUMN = 14
TASK_COLUMN = 19  # "library-medium-001" and a space

# reward(correct, n_sampled, n_calls) -> the reward GRPO trains on
RewardFn = Callable[[bool, int, int], float]


@dataclass
class RolloutRecord:
    """One training rollout, in the project's ``RolloutRecord`` field names.

    The first sixteen fields are the project's (``mhrl/rollout.py``); the
    rest say where the rollout sits in the run and how it ended.

    Attributes:
        task_id: The task.
        harness: The harness.
        purpose: ``train``: sampled at temperature 1, recorded for training.
        correct: Whether the grader accepted ``answer.txt``.
        submitted: The answer judged, or None when there was no file.
        reason: The grader's explanation.
        n_calls: The harness's model calls (the project's ``count_calls``).
        n_sampled: Tokens the model generated across those calls.
        reward: Correctness plus the bonus, as GRPO trains on it.
        seconds: Wall time of the harness's run.
        timed_out: Whether the time limit killed it.
        trainable: The proxy's verdict on the whole rollout.
        n_trainable_tokens: Tokens a trainer may learn from.
        session_id: The capture session.
        engine: ``model``.
        sequences: The trainable rows: ``input_ids``, ``loss_mask``,
            ``logprobs`` (the engine's, recorded as it sampled).
        step: The GRPO step, from 1.
        group: The task's group within the step, from 0.
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
    reward: float = 0.0
    seconds: float = 0.0
    timed_out: bool = False
    trainable: bool = False
    n_trainable_tokens: int = 0
    session_id: str = ""
    engine: str = ENGINE_LABEL
    sequences: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    group: int = 0
    stop: str = ""
    findings: list[str] = field(default_factory=list)
    loopback_only: bool | None = None


@dataclass(frozen=True)
class Job:
    """One rollout to run.

    Attributes:
        task: The task.
        harness: The harness.
        step: The GRPO step.
        group: The task's group within the step.
        index: The rollout's number within its group, from 0.
    """

    task: Task
    harness: str
    step: int
    group: int
    index: int


@dataclass(frozen=True)
class RolloutSetup:
    """What every rollout of one run shares.

    Attributes:
        client: The capture proxy's client.
        proxy_url: The proxy's root, given to each harness.
        reward: Turns (correct, sampled tokens, calls) into the reward.
        run: The run's name (its folder under ``runs/grpo``).
        options: The wiring's options (the project's).
    """

    client: CaptureClient
    proxy_url: str
    reward: RewardFn
    run: str
    options: Options = Options()


@dataclass(frozen=True)
class Failure:
    """A rollout that produced no record, and why.

    Attributes:
        job: The rollout.
        reason: The error, or why it was not run.
    """

    job: Job
    reason: str


def rollout_folder(run: str, job: Job) -> Path:
    """Where one rollout runs and keeps its evidence.

    Args:
        run: The run's name.
        job: The rollout.

    Returns:
        ``COURSE_HOME/26-grpo-in-harnesses/runs/grpo/<run>/step_NNN/<harness>/
        <task>-<index>``.
    """
    name = f"{job.task.task_id}-{job.index}"
    step = f"step_{job.step:03d}"
    runs = chapter_home() / RUNS_FOLDER / EVIDENCE_FOLDER
    return runs / run / step / job.harness / name


def prepare_rollout(folder: Path, harness: str, task: Task) -> RunFolders:
    """Make a rollout's folders: the task here, the state in the temp folder.

    Args:
        folder: The rollout's folder (emptied first).
        harness: The harness.
        task: The task.

    Returns:
        The folders, both empty but for ``data.db``.

    Raises:
        OSError: If a folder cannot be made.
    """
    beside = prepare_run(folder, task)
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


def run_rollout(job: Job, setup: RolloutSetup) -> RolloutRecord:
    """Run one training rollout through the proxy, grade it and reward it.

    Args:
        job: The rollout.
        setup: What every rollout shares.

    Returns:
        Its record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If a folder cannot be made or the harness cannot start.
        HarnessInstallError: If the sandbox cannot be applied (nothing runs).
    """
    folders = prepare_rollout(rollout_folder(setup.run, job), job.harness, job.task)
    try:
        return rollout_in(folders, job, setup)
    finally:
        shutil.rmtree(folders.state, ignore_errors=True)  # as the project does


def rollout_in(folders: RunFolders, job: Job, setup: RolloutSetup) -> RolloutRecord:
    """Session, wiring, boxed run, grade, reward and record, in a rollout's folders.

    Args:
        folders: The rollout's folders.
        job: The rollout.
        setup: What every rollout shares.

    Returns:
        The record.

    Raises:
        ProxyError: If the proxy refuses the session or cannot be reached.
        OSError: If the harness cannot start.
        HarnessInstallError: If the sandbox cannot be applied.
    """
    labels = {"task_id": job.task.task_id, "harness": job.harness, "step": job.step}
    # No engine named: the session samples from the proxy's own, the model.
    session = setup.client.open_session(PURPOSE_TRAIN, labels, MAX_CALLS)
    wiring: Wiring | None = None
    # process, verdict, document and log are set inside this block; any
    # error in it leaves the function, so below they always exist.
    try:
        wiring = wire(
            job.harness,
            folders,
            task_instruction(job.task),
            setup.proxy_url,
            session.session_id,
            setup.options,
        )
        wiring.write()
        process = run_boxed(wiring, folders, setup.options.timeout_s)
        verdict = grade(job.task, folders.task)
        document = setup.client.rollout(session.session_id)
    finally:  # on every way out; a failure here must not hide the one above
        try:
            setup.client.close(session.session_id)
        except ProxyError as error:
            print(
                f"  session {session.session_id} not closed: {error}", file=sys.stderr
            )
        if wiring is not None:
            try:
                log = keep_log(own_log(wiring, folders), folders)
            except OSError as error:  # the log stays where the harness wrote it
                log = own_log(wiring, folders)
                print(f"  log not copied: {tilde(str(error))}", file=sys.stderr)
    facts = read_facts(job.harness, document, log, process)
    turns = [t for t in document.get("turns") or [] if not t.get("discarded")]
    n_sampled = sum(int(t.get("n_sampled") or 0) for t in turns)
    stats = document.get("stats") or {}
    record = RolloutRecord(
        task_id=job.task.task_id,
        harness=job.harness,
        correct=verdict.correct,
        submitted=verdict.submitted,
        reason=verdict.reason,
        n_calls=facts.agent_calls,
        n_sampled=n_sampled,
        reward=setup.reward(verdict.correct, n_sampled, facts.agent_calls),
        seconds=round(process.seconds, 1),
        timed_out=process.exit_code is None,
        trainable=bool(document.get("trainable")),
        n_trainable_tokens=int(stats.get("n_trainable_tokens") or 0),
        session_id=session.session_id,
        sequences=trainable_rows(document),
        step=job.step,
        group=job.group,
        stop=stop_label(facts.stop),
        findings=facts.findings,
        loopback_only=process.network.loopback_only,
    )
    run = {"facts": asdict(facts), "network": process.network.as_dict()}
    run["record"] = {k: v for k, v in asdict(record).items() if k != "sequences"}
    try:
        save_run(folders, document, run)
    except OSError as error:  # the evidence is lost, never the record
        where = f"{job.harness} {job.task.task_id}"
        print(f"  evidence not kept for {where}: {tilde(str(error))}", file=sys.stderr)
    return record


def rollout_line(number: int, total: int, record: RolloutRecord) -> str:
    """One line of progress: the rollout, its verdict, reward, calls and tokens.

    Args:
        number: The rollout's number in the step, from 1.
        total: Rollouts in the step.
        record: Its record.

    Returns:
        Such as ``[3/8] claude-code    shop-hard-005      right · reward 1.09 ...``.
    """
    verdict = "right" if record.correct else "wrong"
    per_call = record.n_sampled / record.n_calls if record.n_calls else 0.0
    rows = len(record.sequences)
    tokens = sum(len(row["input_ids"]) for row in record.sequences)
    return (
        f"  [{number}/{total}] {record.harness:<{HARNESS_COLUMN}} "
        f"{record.task_id:<{TASK_COLUMN}}{verdict} · reward {record.reward:.2f} · "
        f"{record.n_calls} calls · {record.n_sampled} sampled ({per_call:.0f} per "
        f"call) · {rows} row{'' if rows == 1 else 's'}, {tokens:,} tokens · "
        f"{record.seconds:.1f} s · {record.stop}"
    )


def run_step(
    jobs: list[Job],
    setup: RolloutSetup,
    concurrency: int = CONCURRENCY,
    say: Callable[[str], None] = print,
    run_one: Callable[[Job, RolloutSetup], RolloutRecord] = run_rollout,
) -> tuple[list[RolloutRecord], list[Failure]]:
    """Run a step's rollouts, ``concurrency`` at a time, and keep them in job order.

    When the proxy itself fails, every rollout not yet started is skipped and
    said so, instead of failing one after another.

    Args:
        jobs: The step's rollouts, group by group.
        setup: What every rollout shares.
        concurrency: Rollouts at a time.
        say: Where progress lines go.
        run_one: Runs one rollout (the tests give their own).

    Returns:
        The records, in job order, and the rollouts that made none.
    """
    lock = threading.Lock()
    proxy_down = threading.Event()
    made: dict[int, RolloutRecord] = {}
    failed: list[Failure] = []

    def fail(job: Job, reason: str) -> None:
        """Keep and print one rollout that made no record.

        Args:
            job: The rollout.
            reason: Why.
        """
        with lock:
            failed.append(Failure(job, reason))
            say(f"  FAILED {job.harness} {job.task.task_id}: {reason}")

    def one(numbered: tuple[int, Job]) -> None:
        """Run one rollout and keep its record, or its failure.

        Args:
            numbered: Its position, and the rollout.
        """
        position, job = numbered
        if proxy_down.is_set():
            fail(job, "not run: the proxy failed earlier")
            return
        try:
            record = run_one(job, setup)
        except ProxyError as error:
            proxy_down.set()
            fail(job, tilde(str(error)))
            return
        except (OSError, HarnessInstallError) as error:
            fail(job, tilde(str(error)))
            return
        except (KeyError, ValueError) as error:  # a bug, or a record not as expected
            traceback.print_exc(file=sys.stderr)  # the whole story, for the bug hunt
            fail(job, f"rollout failed: {error!r}")
            return
        with lock:
            made[position] = record
            say(rollout_line(len(made), len(jobs), record))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, enumerate(jobs)))
    return [made[i] for i in sorted(made)], failed


def append_records(path: Path, records: list[RolloutRecord]) -> None:
    """Append records to a run's rollouts file (one record per line).

    Args:
        path: The file (created with its folder if needed).
        records: The records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")


def plan_groups(
    tasks: list[Task],
    harnesses: tuple[str, ...],
    steps: int,
    tasks_per_step: int,
    seed: int = TASK_SEED,
) -> list[list[tuple[Task, str]]]:
    """Every step's (task, harness) groups, in the project's order.

    The project's ``train_grpo``: the tasks shuffled once with ``seed``, then
    taken in turn; the harnesses alternate with the same counter.

    Args:
        tasks: The training tasks, in file order.
        harnesses: The training harnesses.
        steps: How many steps.
        tasks_per_step: Groups per step.
        seed: The shuffle's seed.

    Returns:
        One list of ``(task, harness)`` per step.

    Raises:
        ValueError: If there is no task or no harness to take.
    """
    if not tasks or not harnesses:
        raise ValueError("plan_groups needs at least one task and one harness")
    order = list(tasks)
    random.Random(seed).shuffle(order)
    plan, cursor = [], 0
    for _ in range(steps):
        groups = []
        for _ in range(tasks_per_step):
            task = order[cursor % len(order)]
            groups.append((task, harnesses[cursor % len(harnesses)]))
            cursor += 1
        plan.append(groups)
    return plan
