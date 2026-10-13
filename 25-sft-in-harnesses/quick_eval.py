# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""A small held-out check: the base model against the SFT adapter, cell by cell.

Chapter 24's evaluation (``evaluate.py``: one greedy attempt per (task,
harness) cell, 8 calls, replies capped at 1,024 tokens, the harness in the
sandbox with its state in the temp folder) on a few held-out tasks per
harness, run twice on the SAME cells: once with the base model, once with
the adapter ``train_sft.py`` saved. Between the two the engine switches
adapters itself (``POST /admin/adapter``), as the project's engine does, so
both checkpoints are measured by the same engine, proxy and harnesses.

It prints each checkpoint's score per harness, calls per solve (counted on
solved cells only, so compare with care: a checkpoint that solves other
cells is averaged over other cells; the cells both solved are shown too),
and every harness's change cell by cell: gained, lost and the exact
McNemar p (``noise.py``'s reading). Then the project's recorded SFT row,
read from its bundled records, in its own box: 24 tasks per harness, so a
different test, never mixed with this one.

Start the model's engine and the proxy first (``./start.sh``), with the SFT
adapter on disk (``train_sft.py``). ``--limit`` sets the tasks per harness
(8; ``--fast`` 4), ``--adapter`` another adapter, ``--resume`` finishes an
interrupted run, ``--before-cell CMD`` waits before each cell while the
machine is busy (chapter 24's). ``--label`` names a second adapter's records
(``--adapter .../sft-no-bias-correction/adapter.safetensors --label
sft-no-bias-correction --resume`` measures it on the same cells, the base
model's cells already recorded). Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from baseline_eval import (
    check_harness,
    kept_records,
    refuse_other_versions,
    solved_calls,
    waiter,
)
from capture_client import CaptureClient, ProxyError
from evaluate import (
    CellSetup,
    Record,
    balanced,
    evaluate,
    format_table,
    read_records,
    summarize,
)
from install import HARNESSES, HarnessInstallError
from lfm_files import chapter_home, short_path
from noise import PROJECT_RUN, Checkpoint, project_checkpoint, show_pairs
from run_one import PROXY_URL
from tasks import Task, load_tasks
from wire import Options

SCRIPT = "quick_eval.py"
DEFAULT_LIMIT = 8  # tasks per harness: 24 cells per checkpoint
FAST_LIMIT = 4
# What the model's engine says it serves: lfm.py's SERVES, kept in step by
# hand (this script never imports the engine).
SERVES_MODEL = "lfm2.5-2.6b 4-bit"
BASE, SFT = "base", "sft"
LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")  # a label names a records file
PROJECT = "project"  # the project's checkpoints are labelled "project base" ...
NO_SCRIPT = "none"  # the model writes every reply
FOLDER = "quick_eval"  # COURSE_HOME/25-sft-in-harnesses/quick_eval/eval_<label>.jsonl
RECORDS_NAME = "eval_{label}.jsonl"
FINGERPRINT_NAME = "eval_{label}.adapter.json"  # which adapter made the records
TRAINING_NOTE = "adapter.json"  # train_sft.py's note beside the adapter
HASH_CHUNK = 1 << 20
ADAPTER = Path("sft") / "adapter.safetensors"  # train_sft.py's output
HTTP_TIMEOUT_S = 120  # a switch waits for the requests queued before it
HARNESS_COLUMN = 15
WIDTH = 78
RULE = "━━"


def records_path(label: str) -> Path:
    """Where one checkpoint's records go.

    Args:
        label: ``base`` or ``sft``.

    Returns:
        ``COURSE_HOME/25-sft-in-harnesses/quick_eval/eval_<label>.jsonl``.
    """
    return chapter_home() / FOLDER / RECORDS_NAME.format(label=label)


def engine_call(url: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Ask the engine something: GET without a body, POST with one.

    Args:
        url: The engine's root.
        path: Such as ``/health``.
        body: The JSON body of a POST, or None for a GET.

    Returns:
        The answer, parsed.

    Raises:
        SystemExit: If the engine cannot be reached or refuses.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        said = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{SCRIPT}: the engine refused {path}: {said}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        reason = f"{SCRIPT}: cannot reach the engine at {url}: {error}"
        raise SystemExit(reason) from error


def use_adapter(engine_url: str, adapter: Path | None) -> str:
    """Switch the engine to an adapter (or to none) and check it took.

    Args:
        engine_url: The engine's root.
        adapter: The adapter file, or None for the base model.

    Returns:
        The engine's label for what it now samples with.

    Raises:
        SystemExit: If the engine refuses, or reports something else in use.
    """
    path = str(adapter) if adapter else None
    label = engine_call(engine_url, "/admin/adapter", {"path": path})["adapter"]
    health = engine_call(engine_url, "/health")
    if health.get("adapter") != label:
        said = health.get("adapter")
        raise SystemExit(f"{SCRIPT}: the engine says {said}, not {label}")
    return label


def starting_adapter(engine_url: str) -> Path | None:
    """The adapter the engine samples with now, to put back afterwards.

    Args:
        engine_url: The engine's root.

    Returns:
        Its file, or None for the base model.
    """
    found = engine_call(engine_url, "/health").get("adapter_path")
    return Path(found) if found else None


def check_engine(engine_url: str) -> None:
    """Refuse an engine that is not the model (the teacher, or a script).

    Args:
        engine_url: The engine behind the proxy.

    Raises:
        SystemExit: If it serves anything but the 4-bit model.
    """
    serves = engine_call(engine_url, "/health").get("serves")
    if serves != SERVES_MODEL:
        raise SystemExit(
            f"{SCRIPT}: {engine_url} serves {serves or 'a script'}, not the model: "
            "start it with ./start.sh (the teacher would solve every cell)"
        )


def average_calls(summary: dict[str, Any], harnesses: list[str]) -> float | None:
    """Calls per solve averaged over the harnesses, each weighing the same.

    The project's "7.1 -> 4.8" is this number: each harness's mean calls on
    its solved cells, then the plain mean over harnesses.

    Args:
        summary: ``summarize``'s result.
        harnesses: The harnesses.

    Returns:
        The average, or None when no harness solved anything.
    """
    values = [
        summary["per_harness"][h]["calls_when_solved"]
        for h in harnesses
        if summary["per_harness"][h]["calls_when_solved"] is not None
    ]
    return statistics.fmean(values) if values else None


def both_solved(
    before: list[Record], after: list[Record], harness: str
) -> tuple[int, float | None, float | None]:
    """Calls per solve on the cells BOTH checkpoints solved: a like-for-like count.

    Args:
        before: The first checkpoint's records.
        after: The second's.
        harness: The harness.

    Returns:
        How many cells both solved, and the mean calls on them before and
        after (None when there are none).
    """
    first = {r.task_id: r for r in before if r.harness == harness and r.correct}
    second = {r.task_id: r for r in after if r.harness == harness and r.correct}
    shared = sorted(set(first) & set(second))
    if not shared:
        return 0, None, None
    return (
        len(shared),
        statistics.fmean(first[t].n_calls for t in shared),
        statistics.fmean(second[t].n_calls for t in shared),
    )


def calls_lines(
    before: list[Record],
    after: list[Record],
    harnesses: list[str],
    label: str = SFT,
) -> list[str]:
    """Calls per solve: averaged over harnesses, and on the cells both solved.

    Args:
        before: The base model's records.
        after: The adapter's records.
        harnesses: The harnesses.
        label: The adapter's label.

    Returns:
        The lines.
    """
    old = average_calls(summarize(before, harnesses), harnesses)
    new = average_calls(summarize(after, harnesses), harnesses)
    shown = [f"{value:.1f}" if value is not None else "none" for value in (old, new)]
    lines = [
        f"calls per solve, averaged over harnesses: {BASE} {shown[0]} -> {label} "
        f"{shown[1]} (solved cells only, so compare with care)"
    ]
    for harness in harnesses:
        cells, first, second = both_solved(before, after, harness)
        if not cells:
            lines.append(f"  {harness:<{HARNESS_COLUMN}} no cell solved by both")
            continue
        lines.append(
            f"  {harness:<{HARNESS_COLUMN}} {cells} cells solved by both: "
            f"{first:.1f} -> {second:.1f} calls"
        )
    return lines


def boxed(title: str, lines: list[str]) -> list[str]:
    """Put lines in a labelled box, so their numbers never sit beside ours.

    Args:
        title: The box's label.
        lines: Its lines (each shorter than the box).

    Returns:
        The box's lines.
    """
    inner = WIDTH - 4
    top = f"┌─ {title} " + "─" * max(0, inner - len(title) - 1) + "┐"
    body = [f"│ {line:<{inner}} │" for line in lines]
    return [top, *body, "└" + "─" * (WIDTH - 2) + "┘"]


def project_lines(harnesses: list[str]) -> list[str]:
    """The project's recorded SFT row and its calls per solve, from its records.

    Args:
        harnesses: The harnesses, in display order.

    Returns:
        The box's lines.
    """
    base = project_checkpoint(BASE)
    sft = project_checkpoint(SFT)
    summary = summarize(list(sft.records), harnesses)
    scores = " / ".join(
        f"{summary['per_harness'][h]['accuracy']:.0f}%" for h in harnesses
    )
    old = average_calls(summarize(list(base.records), harnesses), harnesses)
    new = average_calls(summary, harnesses)
    return [
        f"{PROJECT_RUN}: 24 held-out tasks per harness, other cells",
        f"sft pass@1 ({' / '.join(harnesses)}): {scores}",
        f"calls per solve, averaged over harnesses: base {old:.1f} -> sft {new:.1f}",
        "(solved cells only, so compare with care)",
    ]


def print_heading(title: str) -> None:
    """Print a section's heading, as noise.py does.

    Args:
        title: Its words.
    """
    print("")
    print(f"{RULE} {title}")


def fingerprint(adapter: Path | None) -> dict[str, Any]:
    """Name an adapter by its content, so records always say which one made them.

    Args:
        adapter: The adapter file, or None for the base model.

    Returns:
        ``{"adapter", "sha256", "bytes"}`` (``{"adapter": "base"}`` for none).
    """
    if adapter is None:
        return {"adapter": BASE}
    digest = hashlib.sha256()
    with adapter.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return {
        "adapter": short_path(adapter),
        "sha256": digest.hexdigest(),
        "bytes": adapter.stat().st_size,
    }


def check_fingerprint(label: str, adapter: Path | None, kept: list[Record]) -> None:
    """Refuse to add cells to records another adapter made; note this one's.

    Args:
        label: ``base`` or ``sft``.
        adapter: The adapter this run uses.
        kept: Records already in the file (``--resume``).

    Raises:
        SystemExit: If records exist and were made by another adapter, or
            nothing says which adapter made them.
    """
    path = records_path(label).with_name(FINGERPRINT_NAME.format(label=label))
    mine = fingerprint(adapter)
    if kept:
        try:
            theirs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            theirs = None
        if theirs != mine:
            raise SystemExit(
                f"{SCRIPT}: {short_path(records_path(label))} was made with "
                f"{(theirs or {}).get('adapter', 'an adapter nothing names')}, "
                f"not {mine['adapter']} as it is now: move the records away to "
                "measure this one"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mine, indent=2) + "\n", encoding="utf-8")


def missing_cells(
    records: list[Record], tasks: list[Task], harnesses: list[str]
) -> list[str]:
    """The cells a checkpoint should have run and has no record of.

    Args:
        records: Its records.
        tasks: The tasks.
        harnesses: The harnesses.

    Returns:
        ``"<harness> <task>"`` for each missing cell.
    """
    have = {r.cell for r in records}
    return [
        f"{h} {t.task_id}"
        for h in harnesses
        for t in tasks
        if (h, t.task_id) not in have
    ]


def training_note(adapter: Path) -> str:
    """What train_sft.py noted beside the adapter: the build, the rows, the steps.

    Args:
        adapter: The adapter file.

    Returns:
        One line, or a line saying there is no note.
    """
    note = adapter.with_name(TRAINING_NOTE)
    try:
        data = json.loads(note.read_text(encoding="utf-8"))
        correction = "on" if data.get("bias_correction", True) else "off"
        return (
            f"trained on {data['model']} ({data['which']}) · {data['rows']} rows · "
            f"{data['steps']} steps at lr {data['lr']:g}, bias correction {correction}"
        )
    except (OSError, ValueError, KeyError, TypeError):
        return (
            f"no readable {TRAINING_NOTE} beside it: which build trained it is unknown"
        )


def run_checkpoint(
    label: str,
    adapter: Path | None,
    tasks: list[Task],
    harnesses: list[str],
    setup: CellSetup,
    args: argparse.Namespace,
    versions: dict[str, str],
) -> list[Record]:
    """Switch the engine's adapter, then run every cell once for one checkpoint.

    Args:
        label: ``base`` or ``sft``.
        adapter: The adapter file, or None.
        tasks: The held-out tasks.
        harnesses: The harnesses.
        setup: The cells' shared setup (its label is replaced).
        args: The command line.
        versions: Each harness's checked version.

    Returns:
        This checkpoint's records for these cells.

    Raises:
        SystemExit: If the engine refuses the adapter, or the records file
            exists without ``--resume`` (or holds another harness version,
            or was made with another adapter).
    """
    out_file = records_path(label)
    kept = kept_records(out_file, args.resume)
    refuse_other_versions(kept, versions)
    check_fingerprint(label, adapter, kept)
    using = use_adapter(setup.engine_url, adapter)
    print_heading(f"{label}: the engine samples with {using}")
    print(f"  records   {short_path(out_file)}")
    mine = CellSetup(
        setup.client, setup.proxy_url, setup.engine_url, NO_SCRIPT, label, versions
    )
    evaluate(
        tasks,
        harnesses,
        mine,
        out_file,
        [r.cell for r in kept],
        before_cell=waiter(args.before_cell),
    )
    wanted = {(h, t.task_id) for h in harnesses for t in tasks}
    return [r for r in read_records(out_file) if r.cell in wanted]


def harness_list(value: str) -> list[str]:
    """Read ``--harnesses``: a comma-separated list of known harnesses.

    Args:
        value: Such as ``mini-swe-agent,pi``.

    Returns:
        The harnesses.

    Raises:
        argparse.ArgumentTypeError: For an unknown name or an empty list.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names or any(name not in HARNESSES for name in names):
        raise argparse.ArgumentTypeError(f"choose from {', '.join(HARNESSES)}")
    return names


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="tasks per harness")
    parser.add_argument("--fast", action="store_true", help=f"--limit {FAST_LIMIT}")
    parser.add_argument("--harnesses", type=harness_list, default=list(HARNESSES))
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--before-cell", default=None, metavar="CMD")
    parser.add_argument("--label", default=SFT, help="names the adapter's records")
    args = parser.parse_args()
    if not LABEL_RE.fullmatch(args.label) or args.label == BASE:
        parser.error(f"--label is letters, digits, . _ - and not {BASE!r}")
    if args.limit is None:
        args.limit = FAST_LIMIT if args.fast else DEFAULT_LIMIT
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def report(
    before: list[Record],
    after: list[Record],
    harnesses: list[str],
    say: Callable[[str], None] = print,
    label: str = SFT,
) -> None:
    """Print both checkpoints, their calls, and every harness's change.

    Args:
        before: The base model's records.
        after: The adapter's records.
        harnesses: The harnesses.
        say: Where lines go.
        label: The adapter's label.
    """
    summaries = {BASE: summarize(before, harnesses), label: summarize(after, harnesses)}
    print_heading("both checkpoints, on the same cells")
    say(format_table(summaries, harnesses))
    for label, summary in summaries.items():
        say(f"{label:<5} {solved_calls(summary, harnesses)}")
    for line in calls_lines(before, after, harnesses, label):
        say(line)
    pair = (Checkpoint(BASE, tuple(before)), Checkpoint(label, tuple(after)))
    show_pairs([pair], say)
    theirs = [project_checkpoint(label) for label in (BASE, SFT)]
    named = [Checkpoint(f"{PROJECT} {c.label}", c.records) for c in theirs]
    show_pairs([(named[0], named[1])], say)
    say(f"  (the project's pair is {PROJECT_RUN}, from its bundled records)")
    say("")
    for line in boxed("the project's recorded SFT row", project_lines(harnesses)):
        say(line)


def main() -> None:
    """Run both checkpoints on the same cells, then compare them.

    Raises:
        SystemExit: If the stack is down or not the model, the adapter is
            missing, a harness is refused, the two checkpoints do not cover
            the same cells, or (status 1) a run's network watch saw more
            than this machine.
    """
    args = parse_args()
    started = time.perf_counter()
    adapter = (args.adapter or chapter_home() / ADAPTER).expanduser().resolve()
    if not adapter.is_file():
        raise SystemExit(
            f"{SCRIPT}: no adapter at {short_path(adapter)}: run train_sft.py first"
        )
    tasks = balanced(load_tasks(), args.limit)
    client = CaptureClient(args.proxy_url)
    try:
        engine_url = str(client.health().get("upstream", ""))
    except ProxyError as error:
        raise SystemExit(f"{error}: start the stack first (./start.sh)") from error
    check_engine(engine_url)
    print(f"{SCRIPT} · base against SFT on the same held-out cells")
    print(
        f"  tasks     {len(tasks)} held-out tasks per harness, round-robin over "
        f"database x difficulty · harnesses {', '.join(args.harnesses)}"
    )
    print(f"  adapter   {short_path(adapter)}")
    print(f"            {training_note(adapter)}")
    print(f"  engine    {engine_url} (the model) · proxy {args.proxy_url}")
    print("versions")
    try:
        versions = {h: check_harness(h, tasks[0], Options()) for h in args.harnesses}
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    setup = CellSetup(client, args.proxy_url, engine_url, NO_SCRIPT, BASE, versions)
    found = starting_adapter(engine_url)
    try:
        before = run_checkpoint(
            BASE, None, tasks, args.harnesses, setup, args, versions
        )
        after = run_checkpoint(
            args.label, adapter, tasks, args.harnesses, setup, args, versions
        )
    finally:  # on every way out, the engine samples with what it had before
        try:
            put_back = use_adapter(engine_url, found)
            print(f"the engine is back on {put_back}, as it was found")
        except SystemExit as error:  # never hide the error that brought us here
            print(f"the engine was NOT put back: {error}", file=sys.stderr)
    gaps = {
        label: missing_cells(records, tasks, args.harnesses)
        for label, records in ((BASE, before), (args.label, after))
    }
    if any(gaps.values()):
        for label, cells in gaps.items():
            print(f"{label} has no record of {len(cells)} cells: {', '.join(cells)}")
        raise SystemExit(f"{SCRIPT}: both checkpoints must cover the same cells")
    report(before, after, args.harnesses, label=args.label)
    minutes, rest = divmod(round(time.perf_counter() - started), 60)
    print(f"this run took {minutes} min {rest} s")
    outside = [r for r in before + after if r.loopback_only is False]
    if outside:
        cells = ", ".join(f"{r.engine} {r.harness} {r.task_id}" for r in outside)
        print(f"network   NOT loopback-only (or a sample was lost) in: {cells}")
        sys.exit(1)
    print("network   every rollout: lsof saw only this machine's addresses")


if __name__ == "__main__":
    main()
