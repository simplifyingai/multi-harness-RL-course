# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run a new task family in one harness through the capture proxy, and read it.

The last step of making the pipeline yours: measure the untrained model on
your own tasks before you train on anything. Chapter 24's protocol is kept
as it was (one attempt per task, temperature 0, at most 8 model calls,
replies capped at 1,024 tokens, 300 s per rollout, the pinned harness
checked inside the sandbox, each record appended as its rollout ends), and
three things change:

* **the tasks** are the CSV family's (``make_csv_tasks.py``; made first
  when the seed's file is missing), all of them unless ``--limit`` picks
  fewer, round-robin over difficulty;
* **one harness** runs them (``--harness``, mini-swe-agent by default), so
  each new piece is tried on its own: the new family in a harness already
  measured, or the new harness on a family already measured;
* **the reading** ends with the score's Wilson interval and power.py's
  answer: how many paired tasks the hoped-for gain (``--gain``, +10 points)
  would need before it could be told from noise. ``--against LABEL`` reads
  this run beside an earlier one of the same tasks, cell by cell: gained,
  lost and the exact p.

Start the stack first: ``./start.sh scripted`` (no model: every task gets
the solving script) or ``./start.sh`` (the model). Records go to
``COURSE_HOME/29-make-it-yours/family/<label>/<harness>.jsonl``. ``--plan``
prints the protocol and the tasks without the stack. Only Python's standard
library is needed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from capture_client import CaptureClient
from evaluate import (
    CellSetup,
    Record,
    balanced,
    drop_torn_tail,
    evaluate,
    read_records,
    summarize,
)
from install import (
    HARNESSES,
    MINI,
    PINS,
    HarnessInstallError,
    check_version,
    ensure,
    probe_version,
    read_record,
    resolved_version,
)
from lfm_files import MAX_CALLS, OUTPUT_CAP, chapter_home, short_path, tilde
from make_csv_tasks import ensure_tasks
from power import DEFAULT_GAIN, DEFAULT_POWER, check_question, one_line, share_argument
from run_one import PROXY_URL, RUNS_FOLDER, chosen_script, detect_engine, prepare_run
from sandbox import FAMILY_FOLDER
from stats import compare, exact_p, pass_at_1, wilson_interval
from tasks import DEFAULT_SEED, DIFFICULTIES, Task, task_instruction, tasks_path
from wire import ROLLOUT_TIMEOUT_S, Options, wire

RECORDS_NAME = "{harness}.jsonl"
SUMMARY_NAME = "{harness}.summary.json"
MODEL_LABEL = "base"  # the untrained model: the first checkpoint
PROBE_FOLDER = "version-probe"
NO_KEY = "none-yet"  # the probe runs before any session, so with no API key
LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")  # a label names a folder
SECONDS_PER_MINUTE = 60
PAUSE_SHOWN_S = 1.0  # a --before-cell wait this long or longer is printed
SIGNIFICANT = 0.05
TASK_COLUMN = 18
QUESTION_SHOWN = 64  # characters of a question in the plan
PERCENT = 100


def records_path(label: str, harness: str) -> Path:
    """Where one checkpoint's records in one harness go.

    Args:
        label: Such as ``base``.
        harness: The harness.

    Returns:
        ``COURSE_HOME/29-make-it-yours/family/<label>/<harness>.jsonl``.
    """
    return chapter_home() / FAMILY_FOLDER / label / RECORDS_NAME.format(harness=harness)


def print_protocol(
    tasks: list[Task], every: int, harness: str, seed: int, out_file: Path
) -> None:
    """Print the protocol every task runs under.

    Args:
        tasks: The picked tasks.
        every: How many tasks the seed's file holds.
        harness: The harness.
        seed: The seed the tasks were made with.
        out_file: Where the records go.
    """
    spread = ", ".join(
        f"{sum(t.difficulty == d for t in tasks)} {d}" for d in DIFFICULTIES
    )
    print("try_family.py · the untrained model on a new task family, in one harness")
    print(
        f"  tasks     {len(tasks)} of {every} CSV tasks (seed {seed}: "
        f"{short_path(tasks_path(seed))}): {spread}"
    )
    print(f"  harness   {harness} {PINS[harness].version}")
    print(
        f"  protocol  one attempt per task · temperature 0 · {MAX_CALLS} calls · "
        f"replies capped at {OUTPUT_CAP:,} tokens · {ROLLOUT_TIMEOUT_S} s per rollout"
    )
    print(f"  records   {short_path(out_file)} (one line per rollout, as it ends)")


def show_tasks(tasks: list[Task]) -> None:
    """Print the picked tasks, in the order they run.

    Args:
        tasks: The tasks.
    """
    print("")
    print("tasks, in the order they run:")
    for task in tasks:
        question = task.question
        if len(question) > QUESTION_SHOWN:
            question = question[: QUESTION_SHOWN - 3] + "..."
        print(
            f"  {task.task_id:<{TASK_COLUMN}} {question:<{QUESTION_SHOWN}} "
            f"{task.answer}"
        )


def check_harness(harness: str, task: Task) -> str:
    """Install the harness if it is missing, then ask its version inside the sandbox.

    Chapter 24's ``baseline_eval.check_harness``, for one harness.

    Args:
        harness: The harness.
        task: Any task (the probe runs in a task folder, as a run does).

    Returns:
        The version it reported, which is its pin.

    Raises:
        SystemExit: If it cannot be installed, or reports another version.
    """
    try:
        if not resolved_version(harness):
            print(
                f"  {harness} is not installed in {short_path(chapter_home())}: "
                "installing it (install.py)",
                flush=True,
            )
        ensure(harness, install=True, records=read_record())  # never replaces one
        folder = chapter_home() / RUNS_FOLDER / PROBE_FOLDER / harness
        folders = prepare_run(folder, task)
        probe = wire(
            harness, folders, task_instruction(task), PROXY_URL, NO_KEY, Options()
        )
        reported = probe_version(harness, folders.task, folders.state, probe.env)
        check_version(harness, reported)
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    print(
        f"version   {reported}, asked inside the sandbox · pinned "
        f"{PINS[harness].version} · ok",
        flush=True,
    )
    return reported


def kept_records(out_file: Path, resume: bool, version: str) -> list[Record]:
    """Read what an earlier run of this label recorded, if it may go on.

    Args:
        out_file: The records file.
        resume: Whether ``--resume`` was given.
        version: The harness's version now.

    Returns:
        The records already in the file ([] for a fresh run).

    Raises:
        SystemExit: If the file holds records and ``--resume`` was not given,
            a line in it is broken, or a record names another harness version.
    """
    if resume and drop_torn_tail(out_file):
        print("records   cut a half-written last line: that task runs again")
    try:
        kept = read_records(out_file)
    except ValueError as error:
        raise SystemExit(f"{error}: delete that line, then --resume") from error
    if kept and not resume:
        raise SystemExit(
            f"{short_path(out_file)} already holds {len(kept)} records: "
            "--resume to finish that run, or --label to start another"
        )
    other = [r for r in kept if r.version not in ("", version)]
    if other:
        raise SystemExit(
            f"{other[0].task_id} was recorded with {other[0].harness} "
            f"{other[0].version}, not {version}: start a new --label"
        )
    return kept


def waiter(command: str | None) -> Callable[[], None] | None:
    """Turn ``--before-cell`` into the hook ``evaluate`` calls before each task.

    Chapter 24's: the command may wait as long as it likes (for a busy
    machine to go quiet, say); a non-zero exit stops the run, and
    ``--resume`` finishes it later.

    Args:
        command: The shell command, or None for no hook.

    Returns:
        The hook, or None. The hook raises ``SystemExit`` when the command
        exits with a status other than 0.
    """
    if command is None:
        return None

    def wait() -> None:
        """Run the command, wait for it, and stop the run if it says so.

        Raises:
            SystemExit: If the command exits with a status other than 0.
        """
        started = time.perf_counter()
        # A command line the learner typed, run as typed: a shell is the point.
        status = subprocess.run(command, shell=True, check=False).returncode
        elapsed = time.perf_counter() - started
        if elapsed >= PAUSE_SHOWN_S:
            print(f"  paused {round(elapsed)} s for --before-cell", flush=True)
        if status != 0:
            raise SystemExit(
                f"--before-cell exited {status}: stopped before the next task; "
                "run again with --resume to finish"
            )

    return wait


def result_lines(records: list[Record], tasks: list[Task]) -> list[str]:
    """One line per task, from the records: the verdict, and what was expected.

    Args:
        records: This run's records.
        tasks: The tasks, in order.

    Returns:
        Lines; a wrong answer shows the expected one beside it.
    """
    by_task = {r.task_id: r for r in records}
    lines = []
    for task in tasks:
        record = by_task.get(task.task_id)
        if record is None:
            lines.append(
                f"  {task.task_id:<{TASK_COLUMN}} {task.difficulty:<7} not run"
            )
            continue
        verdict = "correct" if record.correct else "wrong  "
        given = "none" if record.submitted is None else tilde(repr(record.submitted))
        line = f"  {task.task_id:<{TASK_COLUMN}} {task.difficulty:<7} {verdict} {given}"
        if not record.correct:
            line += f" · expected {task.answer!r}"
        lines.append(line)
    return lines


def percent_text(rate: float) -> str:
    """Write a rate as a percentage with one decimal.

    Args:
        rate: 0 to 1.

    Returns:
        Such as ``65.0%``.
    """
    return f"{PERCENT * rate:.1f}%"


def score_lines(records: list[Record], tasks: list[Task], gain: float) -> list[str]:
    """The score, its interval, the split by difficulty and power.py's answer.

    Args:
        records: This run's records.
        tasks: The tasks.
        gain: The hoped-for gain, for power.py.

    Returns:
        Lines.
    """
    solved = sum(r.correct for r in records)
    cells = len(records)
    if not cells:
        return ["score     none recorded"]
    low, high = wilson_interval(solved, cells)
    calls = [r.n_calls for r in records if r.correct]
    per_solve = f"{sum(calls) / len(calls):.1f}" if calls else "none solved"
    difficulty = {t.task_id: t.difficulty for t in tasks}
    split = " · ".join(
        f"{d} {sum(r.correct for r in records if difficulty[r.task_id] == d)}/"
        f"{sum(difficulty[r.task_id] == d for r in records)}"
        for d in DIFFICULTIES
    )
    rate = solved / cells if cells else 0.0
    return [
        f"score     {solved}/{cells} = {pass_at_1(solved, cells):.1f}% · Wilson 95% "
        f"{low:.1f}% to {high:.1f}% · calls per solve {per_solve}",
        f"by level  {split}",
        f"power     {one_line(rate, gain)}; the {percent_text(rate)} is itself "
        f"measured on {cells} tasks",
    ]


def against_lines(other: str, records: list[Record], harness: str) -> list[str]:
    """This run beside an earlier one, on the tasks both ran.

    Args:
        other: The earlier run's label.
        records: This run's records.
        harness: The harness.

    Returns:
        Lines: before, after, gained, lost, the exact p and its reading.

    Raises:
        SystemExit: If the earlier run has no records, or no task in common.
    """
    earlier = earlier_records(other, harness)
    before = {r.task_id: r.correct for r in earlier}
    after = {r.task_id: r.correct for r in records}
    shared = sorted(set(before) & set(after))
    if not shared:
        raise SystemExit(
            f"--against {other}: no task in both runs ({len(before)} there, "
            f"{len(after)} here, such as {min(before)} and {min(after)}): "
            "were they made with the same --seed?"
        )
    comparison = compare({t: before[t] for t in shared}, {t: after[t] for t in shared})
    p = exact_p(len(comparison.gained), len(comparison.lost))
    if not comparison.flipped:
        reading = "nothing changed"
    elif p < SIGNIFICANT:
        side = "gained" if len(comparison.gained) > len(comparison.lost) else "lost"
        reading = f"{side}: beyond noise"
    else:
        reading = "within noise"
    only = sorted(set(before) ^ set(after))
    lines = [
        f"against   {other}, on the {len(shared)} tasks both ran "
        f"({len(only)} in one run only): before {sum(before[t] for t in shared)}, "
        f"after {sum(after[t] for t in shared)}",
        f"  gained  {len(comparison.gained)}: {', '.join(comparison.gained) or '-'}",
        f"  lost    {len(comparison.lost)}: {', '.join(comparison.lost) or '-'}",
        f"  exact p {p:.3f}: {reading}",
    ]
    if only:
        lines.append(f"  left out, in one run only: {', '.join(only)}")
    return lines


def earlier_records(other: str, harness: str) -> list[Record]:
    """Read the run ``--against`` names, or refuse before anything runs.

    Args:
        other: The earlier run's label.
        harness: The harness.

    Returns:
        Its records.

    Raises:
        SystemExit: If it has no records (or a broken line).
    """
    path = records_path(other, harness)
    try:
        earlier = read_records(path)
    except ValueError as error:
        raise SystemExit(f"--against {other}: {error}") from error
    if not earlier:
        raise SystemExit(f"--against {other}: no records at {short_path(path)}")
    return earlier


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", choices=HARNESSES, default=MINI)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="tasks to make when the seed's file is missing (20)",
    )
    parser.add_argument("--limit", type=int, default=None, help="run fewer tasks")
    parser.add_argument("--label", default=None, help="base with the model")
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--script", choices=("auto", "solve", "none"), default="auto")
    parser.add_argument("--resume", action="store_true", help="skip recorded tasks")
    parser.add_argument("--against", default=None, help="an earlier run's label")
    parser.add_argument("--gain", type=share_argument, default=DEFAULT_GAIN)
    parser.add_argument("--plan", action="store_true", help="print the tasks only")
    parser.add_argument(
        "--before-cell",
        default=None,
        help="a shell command run before each task, waited for; a non-zero "
        "exit stops the run",
    )
    args = parser.parse_args()
    for name in ("label", "against"):
        value = getattr(args, name)
        if value is not None and not LABEL_RE.fullmatch(value):
            parser.error(f"--{name}: letters, digits, '.', '_' and '-' only")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit: at least 1 task")
    if args.count is not None and args.count < 1:
        parser.error("--count: at least 1 task")
    try:
        check_question(0.0, args.gain, DEFAULT_POWER)
    except ValueError as error:
        parser.error(f"--gain: {error}")
    return args


def report(
    out_file: Path,
    tasks: list[Task],
    harness: str,
    label: str,
    seconds: float,
    gain: float,
) -> tuple[list[Record], dict[str, Any]]:
    """Print the per-task results and the reading, and write the summary.

    Args:
        out_file: The records file.
        tasks: The picked tasks.
        harness: The harness.
        label: The checkpoint's label.
        seconds: This call's wall time.
        gain: The hoped-for gain, for power.py.

    Returns:
        This run's records and the summary.
    """
    wanted = {t.task_id for t in tasks}
    every = read_records(out_file)
    records = [r for r in every if r.task_id in wanted]
    summary = summarize(records, [harness])
    summary["label"] = label
    path = out_file.with_name(SUMMARY_NAME.format(harness=harness))
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("")
    others = len(every) - len(records)
    if others:
        print(
            f"OTHERS    {others} more in the file for tasks outside this run: left "
            "out of the score"
        )
    if len(records) < len(tasks):
        print(
            f"PARTIAL   {len(records)} of {len(tasks)} tasks recorded: the score "
            "below is on those only; --resume finishes the run"
        )
    print("results   one line per task, from the records")
    for line in result_lines(records, tasks):
        print(line)
    print("")
    for line in score_lines(records, tasks, gain):
        print(line)
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    print(
        f"tasks     {len(records)} of {len(tasks)} recorded · this run took "
        f"{minutes} min {rest} s"
    )
    print(f"summary   {short_path(path)}")
    return records, summary


def main() -> None:
    """Run the family in one harness and read the result.

    Raises:
        SystemExit: If the stack is down, the harness is refused, the records
            file exists without ``--resume``, or (status 1) a run's network
            watch saw more than this machine.
    """
    args = parse_args()
    started = time.perf_counter()
    every = ensure_tasks(args.seed, args.count)
    tasks = balanced(every, args.limit)
    if args.plan:
        label = args.label or MODEL_LABEL
        planned = records_path(label, args.harness)
        print_protocol(tasks, len(every), args.harness, args.seed, planned)
        show_tasks(tasks)
        return
    client = CaptureClient(args.proxy_url)
    engine_url, engine, tokens = detect_engine(client)
    script = chosen_script(args.script, engine)
    label = args.label or (MODEL_LABEL if engine == "model" else f"scripted-{script}")
    out_file = records_path(label, args.harness)
    if args.against == label:
        raise SystemExit(f"--against {label}: that is this run's own label")
    if args.against:
        earlier_records(args.against, args.harness)  # refuse now, not after the run
    print_protocol(tasks, len(every), args.harness, args.seed, out_file)
    print(f"  engine    {engine} at {engine_url} ({tokens} tokens) · script: {script}")
    version = check_harness(args.harness, tasks[0])
    kept = kept_records(out_file, args.resume, version)
    setup = CellSetup(
        client, args.proxy_url, engine_url, script, label, {args.harness: version}
    )
    print("tasks")
    evaluate(
        tasks,
        [args.harness],
        setup,
        out_file,
        [r.cell for r in kept],
        before_cell=waiter(args.before_cell),
    )
    records, _ = report(
        out_file, tasks, args.harness, label, time.perf_counter() - started, args.gain
    )
    if args.against:
        print("")
        for line in against_lines(args.against, records, args.harness):
            print(line)
    # Not True: also a record whose watch never said (None) is not counted clean.
    outside = [r for r in records if r.loopback_only is not True]
    if outside:
        names = ", ".join(r.task_id for r in outside)
        print(f"network   NOT shown loopback-only (or a sample was lost) in: {names}")
        sys.exit(1)
    which = (
        "every rollout"
        if len(records) == len(tasks)
        else f"{len(records)} of {len(tasks)} recorded rollouts"
    )
    print(f"network   {which}: lsof saw only this machine's addresses (sampled)")


if __name__ == "__main__":
    main()
