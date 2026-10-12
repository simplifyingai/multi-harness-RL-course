# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Measure the base model fairly: pass@1 on held-out tasks, in all three harnesses.

The project's ``python -m mhrl.cli eval --label base --limit 24``, for this
laptop. The protocol is fixed before anything runs, and only the adapter
will change at later checkpoints:

* **the tasks**: 24 of the 50 held-out questions (clinic and airline,
  databases no training question uses), picked round-robin over database x
  difficulty (``balanced``), so 4 from each of the six buckets;
* **the harnesses**: mini-swe-agent 2.4.6, Claude Code 2.1.290 and Pi 1.0.4,
  wired as chapter 23 wires them, each checked against its pin inside the
  sandbox. A harness that is not installed yet is installed first
  (``install.py``), so the chapter rebuilds its own input;
* **one attempt per cell**: every (task, harness) pair runs once, in an
  evaluation session: temperature 0 (always the likeliest token), at most
  8 model calls, each reply capped at 1,024 tokens, 300 s per rollout;
* **the records**: each rollout's record is appended to
  ``COURSE_HOME/24-measuring/baseline/eval_<label>.jsonl`` the moment it
  ends, and ``summary_<label>.json`` is written at the end.

Start the stack first: ``ENGINE_PORT=8119 ./start.sh`` (the model) or
``ENGINE_PORT=8119 ./start.sh scripted`` (no model: the scripted engine
answers every cell with ``--script solve``). Then:

* ``uv run baseline_eval.py --plan``: the protocol and the 24 tasks, with
  what a plain ``tasks[:24]`` would have picked (no stack needed);
* ``uv run baseline_eval.py --fast``: 4 tasks per harness (12 cells);
* ``uv run baseline_eval.py``: 24 tasks per harness (72 cells);
* ``--limit N`` and ``--harnesses a,b`` for any smaller subset, ``--resume``
  to finish an interrupted run without repeating its cells.

Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capture_client import CaptureClient
from evaluate import (
    CellSetup,
    Record,
    balanced,
    evaluate,
    format_table,
    read_records,
    summarize,
)
from install import (
    HARNESSES,
    PINS,
    HarnessInstallError,
    check_version,
    ensure,
    probe_version,
    read_record,
    resolved_version,
)
from lfm_files import MAX_CALLS, OUTPUT_CAP, chapter_home, short_path
from run_one import PROXY_URL, RUNS_FOLDER, chosen_script, detect_engine, prepare_run
from tasks import Task, load_tasks, task_instruction
from wire import ROLLOUT_TIMEOUT_S, Options, wire

FULL_LIMIT = 24  # the project's EVAL_LIMIT (scripts/train_from_scratch.sh)
FAST_LIMIT = 4
BASELINE_FOLDER = "baseline"
RECORDS_NAME = "eval_{label}.jsonl"
SUMMARY_NAME = "summary_{label}.json"
MODEL_LABEL = "base"  # the untrained model: the project's first checkpoint
PROBE_FOLDER = "version-probe"
NO_KEY = "none-yet"  # the probe runs before any session, so with no API key
DIFFICULTIES = ("easy", "medium", "hard")
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class Picks:
    """How many tasks one way of picking took from each (database, difficulty).

    Attributes:
        name: How they were picked.
        counts: Tasks per (database, difficulty).
    """

    name: str
    counts: Counter[tuple[str, str]]


def records_path(label: str) -> Path:
    """Where one checkpoint's records go.

    Args:
        label: Such as ``base``.

    Returns:
        ``COURSE_HOME/24-measuring/baseline/eval_<label>.jsonl``.
    """
    return chapter_home() / BASELINE_FOLDER / RECORDS_NAME.format(label=label)


def count_buckets(name: str, tasks: list[Task]) -> Picks:
    """Count a pick of tasks per (database, difficulty).

    Args:
        name: How they were picked.
        tasks: The tasks.

    Returns:
        The counts.
    """
    return Picks(name, Counter((t.domain, t.difficulty) for t in tasks))


def picks_table(picks: list[Picks], domains: list[str]) -> list[str]:
    """Lay several picks side by side, one row each.

    Args:
        picks: The picks.
        domains: The databases, in column order.

    Returns:
        The table's lines.
    """
    width = max(len(p.name) for p in picks) + 2
    head = "".join(f"{d:<21}" for d in domains)
    sub = "".join("".join(f"{x:<7}" for x in DIFFICULTIES) for _ in domains)
    lines = [" " * width + head, " " * width + sub]
    for pick in picks:
        row = "".join(
            f"{pick.counts.get((d, x), 0):<7}" for d in domains for x in DIFFICULTIES
        )
        lines.append(pick.name.ljust(width) + row)
    return lines


def show_plan(tasks: list[Task], limit: int, say: Callable[[str], None]) -> None:
    """Print which tasks ``balanced`` picks, beside what a plain prefix picks.

    Args:
        tasks: All held-out tasks, in file order.
        limit: How many to pick.
        say: Where lines go.
    """
    domains = sorted({t.domain for t in tasks})
    by_name = sorted(tasks, key=lambda t: t.task_id)
    picks = [
        count_buckets(f"balanced(tasks, {limit})", balanced(tasks, limit)),
        count_buckets(f"tasks[:{limit}], file order", tasks[:limit]),
        count_buckets(f"tasks[:{limit}], sorted by id", by_name[:limit]),
        count_buckets(f"all {len(tasks)} held-out tasks", tasks),
    ]
    say("")
    say(f"which {limit} tasks: balanced() against a plain prefix")
    for line in picks_table(picks, domains):
        say(f"  {line}".rstrip())
    say("")
    say(f"balanced(tasks, {limit}) picks, in the order the cells run:")
    for task in balanced(tasks, limit):
        say(f"  {task.task_id:<20} {task.question}")


def print_protocol(
    tasks: list[Task], harnesses: list[str], out_file: Path, say: Callable[[str], None]
) -> None:
    """Print the protocol every cell runs under.

    Args:
        tasks: The picked tasks.
        harnesses: The harnesses.
        out_file: Where the records go.
        say: Where lines go.
    """
    domains = ", ".join(sorted({t.domain for t in tasks}))
    buckets = Counter((t.domain, t.difficulty) for t in tasks)
    spread = sorted(set(buckets.values()))
    per = f"{spread[0]} per bucket" if len(spread) == 1 else "uneven buckets"
    total = len(load_tasks())
    say("baseline_eval.py · pass@1 on held-out tasks, in every harness")
    say(
        f"  tasks     {len(tasks)} of {total} held out ({domains}), round-robin over "
        f"database x difficulty: {len(buckets)} buckets, {per}"
    )
    pins = ", ".join(f"{h} {PINS[h].version}" for h in harnesses)
    say(f"  harnesses {pins}")
    say(
        f"  protocol  one attempt per cell · temperature 0 · {MAX_CALLS} calls · "
        f"replies capped at {OUTPUT_CAP:,} tokens · {ROLLOUT_TIMEOUT_S} s per rollout"
    )
    say(f"  cells     {len(tasks) * len(harnesses)}, one at a time")
    say(f"  records   {short_path(out_file)} (one line per rollout, as it ends)")


def check_harness(harness: str, task: Task, options: Options) -> str:
    """Install a missing harness, then ask its version inside the sandbox.

    Args:
        harness: The harness.
        task: Any task (the probe runs in a task folder, as a run does).
        options: The wiring's options.

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
        ensure(harness, True, read_record())  # never replaces another version
        folder = chapter_home() / RUNS_FOLDER / PROBE_FOLDER / harness
        folders = prepare_run(folder, task)
        probe = wire(
            harness, folders, task_instruction(task), PROXY_URL, NO_KEY, options
        )
        reported = probe_version(harness, folders.task, folders.state, probe.env)
        check_version(harness, reported)
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    print(
        f"  {harness:<14} {reported}, asked inside the sandbox · "
        f"pinned {PINS[harness].version} · ok",
        flush=True,
    )
    return reported


def solved_calls(summary: dict[str, Any], harnesses: list[str]) -> str:
    """Say calls per solve for each harness (counted on solved cells only).

    Args:
        summary: ``summarize``'s result.
        harnesses: The harnesses.

    Returns:
        One line.
    """
    parts = []
    for harness in harnesses:
        row = summary["per_harness"][harness]
        calls = row["calls_when_solved"]
        shown = "none solved" if calls is None else f"{calls:.1f}"
        parts.append(f"{harness} {row['solved']}/{row['cells']}, {shown}")
    return "calls per solve (solved cells only): " + " · ".join(parts)


def report(
    records: list[Record],
    tasks: list[Task],
    harnesses: list[str],
    label: str,
    seconds: float,
) -> dict[str, Any]:
    """Print the table, write the summary, and say what the run cost.

    Args:
        records: Every record in the file for this run's cells.
        tasks: The picked tasks.
        harnesses: The harnesses.
        label: The checkpoint's label.
        seconds: This call's wall time.

    Returns:
        The summary.
    """
    wanted = {(h, t.task_id) for h in harnesses for t in tasks}
    mine = [r for r in records if r.cell in wanted]
    summary = summarize(mine, harnesses)
    summary["label"] = label
    path = records_path(label).with_name(SUMMARY_NAME.format(label=label))
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("")
    print(format_table({label: summary}, harnesses))
    print(solved_calls(summary, harnesses))
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    print(
        f"cells     {len(mine)} of {len(wanted)} recorded · this run took "
        f"{minutes} min {rest} s"
    )
    print(f"summary   {short_path(path)}")
    return summary


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="4 tasks per harness")
    parser.add_argument("--limit", type=int, default=None, help="tasks per harness")
    parser.add_argument("--harnesses", default=",".join(HARNESSES))
    parser.add_argument("--label", default=None, help="base with the model")
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--script", choices=("auto", "solve", "mistakes", "none"))
    parser.add_argument("--resume", action="store_true", help="skip recorded cells")
    parser.add_argument("--plan", action="store_true", help="print the picks only")
    args = parser.parse_args()
    args.harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]
    unknown = sorted(set(args.harnesses) - set(HARNESSES))
    if unknown or not args.harnesses:
        parser.error(f"--harnesses: choose from {', '.join(HARNESSES)}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit: at least 1 task")
    return args


def main() -> None:
    """Run the evaluation and print its table.

    Raises:
        SystemExit: If the stack is down, a harness is refused, the records
            file already exists without ``--resume``, or (status 1) a run's
            network watch saw more than this machine.
    """
    args = parse_args()
    started = time.perf_counter()
    limit = args.limit or (FAST_LIMIT if args.fast else FULL_LIMIT)
    every = load_tasks()
    tasks = balanced(every, limit)
    if args.plan:
        print_protocol(tasks, args.harnesses, records_path(MODEL_LABEL), print)
        show_plan(every, limit, print)
        return
    client = CaptureClient(args.proxy_url)
    engine_url, engine, tokens = detect_engine(client)
    script = chosen_script(args.script or "auto", engine)
    label = args.label or (MODEL_LABEL if engine == "model" else f"scripted-{script}")
    out_file = records_path(label)
    kept = read_records(out_file)
    if kept and not args.resume:
        raise SystemExit(
            f"{short_path(out_file)} already holds {len(kept)} records: "
            "--resume to finish that run, or --label to start another"
        )
    print_protocol(tasks, args.harnesses, out_file, print)
    print(f"  engine    {engine} at {engine_url} ({tokens} tokens) · script: {script}")
    print("versions")
    options = Options()
    versions = {h: check_harness(h, tasks[0], options) for h in args.harnesses}
    setup = CellSetup(client, args.proxy_url, engine_url, script, label, versions)
    print("cells")
    done = [r.cell for r in kept]
    evaluate(tasks, args.harnesses, setup, out_file, done)
    records = read_records(out_file)
    report(records, tasks, args.harnesses, label, time.perf_counter() - started)
    outside = [r for r in records if r.loopback_only is False]
    if outside:
        cells = ", ".join(f"{r.harness} {r.task_id}" for r in outside)
        print(f"network   NOT loopback-only (or a sample was lost) in: {cells}")
        sys.exit(1)
    print("network   every rollout: lsof saw only this machine's addresses")


if __name__ == "__main__":
    main()
