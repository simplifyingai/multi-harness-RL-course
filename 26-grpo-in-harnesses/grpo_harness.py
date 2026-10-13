# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""GRPO through the real harnesses: the project's train_grpo, simplified.

Each step, as ``mhrl/training/grpo.py`` does it:

1. takes the next 2 training tasks, in an order shuffled once with seed 0,
   and gives each ONE harness, alternating mini-swe-agent and Claude Code,
   so a group compares the model's actions, never two harnesses;
2. runs 4 rollouts of each task in its harness, 4 at a time, through the
   capture proxy, which samples every call at temperature 1 from the engine
   and records its exact tokens and log-probabilities (``rollouts.py``);
3. grades each rollout and rewards it: 1 plus the token bonus when right,
   0 when wrong (``rewards.py``);
4. checks the **length guard**: if the median rollout's sampled tokens per
   call passed 150, the last update made the replies balloon, so training
   stops and keeps the adapter from before that update;
5. skips the groups without contrast, trains each shared prefix once, drops
   rows over 6,144 tokens, and makes one clipped update against the
   engine's recorded log-probabilities (``grpo.py``);
6. saves ``adapter_stepNNN.safetensors`` and tells the engine to sample with
   it (``POST /admin/adapter``), so the next step's rollouts come from the
   updated model.

Start the model's engine and the proxy first (``./start.sh``). The model
starts from chapter 25's SFT adapter trained as the project's was
(``train_sft.py --no-bias-correction``); ``--init`` names another, ``--init
base`` none. ``--steps`` sets the length (6, the project's recorded run),
``--attempt1`` trains as the project's first, collapsed attempt did (the
call bonus, learning rate 4e-5, no bias correction, no guard), and
``--guard-demo`` replays that attempt's recorded rollouts through the guard
(no model, no stack). Everything goes to
``COURSE_HOME/26-grpo-in-harnesses/grpo/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import grpo_records
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from baseline_eval import check_harness
from capture_client import CaptureClient, ProxyError
from grpo import StepRows, Update, grpo_update, step_rows
from install import HarnessInstallError
from lfm_files import GIGABYTE, chapter_home, course_home, pick_model, short_path
from lora import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    apply_lora,
    check_adapter_file,
    load_adapter,
    save_adapter,
)
from mlx_lm import load
from rewards import TOKEN_BUDGET, call_reward_for, reward_for
from rollouts import (
    CONCURRENCY,
    TASK_SEED,
    TRAIN_HARNESSES,
    Job,
    RolloutRecord,
    RolloutSetup,
    append_records,
    plan_groups,
    run_step,
)
from run_one import PROXY_URL
from tasks import TRAIN_FILE, Task, load_tasks
from training import (
    MAX_TRAIN_TOKENS,
    attention,
    layer_checkpointing,
    set_memory_limits,
)
from wire import Options

SCRIPT = "grpo_harness.py"
DEFAULT_STEPS = 6  # the project's recorded run (attempt 2) made 6 steps
TASKS_PER_STEP = grpo_records.TASKS_PER_STEP  # the project's --tasks-per-step
GROUP_SIZE = grpo_records.GROUP_SIZE  # the project's --group
LEARNING_RATE = 1e-5  # the project's GRPO_LR
ATTEMPT1_LEARNING_RATE = 4e-5  # attempt 1's: "ballooned replies by step 5"
SEED = TASK_SEED  # the project's task-order seed (rollouts.py)
GUARD_TOKENS = grpo_records.GUARD_TOKENS
GRPO_FOLDER = "grpo"  # COURSE_HOME/26-grpo-in-harnesses/grpo/<run>/
RUN_NAME = "grpo"
ATTEMPT1_RUN_NAME = "attempt1"
ATTEMPT1_FOLDER = "grpo_attempt1"  # attempt 1's folder in the project's runs/main
ATTEMPT1_SUMMARY = "summary_rl_attempt1.json"  # the evaluation of the adapter it kept
ADAPTER_SUFFIX = ".safetensors"
ADAPTER_NAME = "adapter_step{step:03d}" + ADAPTER_SUFFIX
START_LABEL = "the SFT adapter"  # what attempt 1's first rollouts were sampled with
FINAL_NAME = "adapter_final.safetensors"
NOTE_NAME = "run.json"  # the run's settings, written before step 1
BASE = "base"  # --init base: start from the model as downloaded
# Chapter 25's adapter trained as the project's recorded SFT run was (before
# its code review turned bias correction on): the project's GRPO start.
INIT_ADAPTER = (
    Path("25-sft-in-harnesses") / "sft-no-bias-correction" / "adapter.safetensors"
)
SERVES_MODEL = "lfm2.5-2.6b 4-bit"  # the engine's /health "serves" (lfm.SERVES)
ENGINE_TIMEOUT_S = 120  # an adapter switch waits for the requests queued before it
PERCENT = 100
SECONDS_PER_MINUTE = 60
GUARD_SWEEP = (150, 100, 60)  # guards the demo compares
CONTINUATION = "       "  # under "[grpo] "
BUNDLE_SOURCE = (
    "data/project_grpo/attempt1 (the project's runs/main/grpo_attempt1, slimmed)"
)


class RunStopped(Exception):
    """Raised when a step cannot finish: a rollout or the proxy failed."""


@dataclass(frozen=True)
class Config:
    """One run's settings.

    Attributes:
        steps: Optimizer steps.
        tasks_per_step: Groups per step.
        group_size: Rollouts per group.
        learning_rate: AdamW's learning rate.
        bias_correction: AdamW's bias correction.
        call_bonus: True for attempt 1's reward (the article's call bonus),
            False for the token bonus.
        guard: Whether the length guard may stop the run.
        guard_tokens: The guard's limit on the median tokens per call.
        max_tokens: Rows longer than this are skipped (and counted).
        harnesses: The training harnesses, alternated group by group.
        seed: The task order's seed.
    """

    steps: int = DEFAULT_STEPS
    tasks_per_step: int = TASKS_PER_STEP
    group_size: int = GROUP_SIZE
    learning_rate: float = LEARNING_RATE
    bias_correction: bool = True
    call_bonus: bool = False
    guard: bool = True
    guard_tokens: float = GUARD_TOKENS
    max_tokens: int = MAX_TRAIN_TOKENS
    harnesses: tuple[str, ...] = TRAIN_HARNESSES
    seed: int = SEED

    def reward(self, correct: bool, n_sampled: int, n_calls: int) -> float:
        """The reward this run trains on.

        Args:
            correct: Whether the grader accepted the answer.
            n_sampled: Tokens the model generated.
            n_calls: Model calls the rollout made.

        Returns:
            Attempt 1's (call bonus) or the project's (token bonus).
        """
        if self.call_bonus:
            return call_reward_for(correct, n_calls)
        return reward_for(correct, n_sampled)


@dataclass(frozen=True)
class Hooks:
    """What the loop needs from outside the trainer.

    Attributes:
        run_rollouts: Runs a step's rollouts and returns them in job order
            (raises ``RunStopped`` when one cannot be run).
        use_adapter: Tells the engine which adapter to sample with (None:
            the base model) and returns what it now samples with.
    """

    run_rollouts: Callable[[list[Job]], list[RolloutRecord]]
    use_adapter: Callable[[Path | None], str]


@dataclass
class Outcome:
    """How a run ended.

    Attributes:
        steps_run: Steps whose rollouts ran (a stopped step included).
        final: ``adapter_final.safetensors``, or None when there is none.
        stopped_at: The step the guard stopped at, or None.
        kept: The adapter kept (the guard's choice, or the last update's).
        exposure: Per harness, what it gave every update together.
    """

    steps_run: int = 0
    final: Path | None = None
    stopped_at: int | None = None
    kept: Path | None = None
    exposure: dict[str, list[int]] = field(default_factory=dict)


# --------------------------------------------------------------- the engine


def engine_call(url: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Send one request to the engine and read its JSON answer.

    Args:
        url: The engine's root.
        path: Such as ``/health``.
        body: A JSON body (POST), or None (GET).

    Returns:
        The answer.

    Raises:
        SystemExit: If the engine cannot be reached or refuses, with its reason.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=ENGINE_TIMEOUT_S) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise SystemExit(f"{SCRIPT}: the engine refused {path}: {detail}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        reason = f"{SCRIPT}: cannot reach the engine at {url}: {error}"
        raise SystemExit(reason) from error


def check_engine(engine_url: str) -> None:
    """Refuse an engine that is not the model (the teacher, a scripted model).

    Args:
        engine_url: The engine behind the proxy.

    Raises:
        SystemExit: If it serves something else.
    """
    serves = engine_call(engine_url, "/health").get("serves")
    if serves != SERVES_MODEL:
        raise SystemExit(
            f"{SCRIPT}: the proxy fronts {engine_url}, which serves "
            f"{serves or 'a scripted model'}, not {SERVES_MODEL}: start the model "
            "with ./start.sh"
        )


def adapter_switch(engine_url: str) -> Callable[[Path | None], str]:
    """Build the hook that tells the engine which adapter to sample with.

    Args:
        engine_url: The engine.

    Returns:
        ``use_adapter(path) -> the label the engine reports``.
    """

    def use_adapter(path: Path | None) -> str:
        """POST /admin/adapter: the project's ``use_adapter``.

        Args:
            path: An adapter file, or None for the base model.

        Returns:
            What the engine now samples with.
        """
        body = {"path": str(path) if path else None}
        return str(engine_call(engine_url, "/admin/adapter", body).get("adapter"))

    return use_adapter


# ---------------------------------------------------------------- the model


def load_model(folder: Path) -> nn.Module:
    """Load the 4-bit model with mlx-lm, every weight into memory.

    Args:
        folder: The build's folder (read only; nothing is downloaded).

    Returns:
        The model.
    """
    model, _ = load(str(folder))
    mx.eval(model.parameters())  # MLX is lazy: read every weight now
    return model


def load_policy(folder: Path, init: Path | None) -> nn.Module:
    """The trainer's own copy of the model, with the adapter it starts from.

    The project's ``load_policy``: the memory limits, the model, chapter 13's
    adapter shape and the starting adapter's weights in it. The engine holds
    another copy; ``use_adapter`` keeps the two on the same weights.

    Args:
        folder: The 4-bit build.
        init: The starting adapter, or None for a fresh (no-op) one.

    Returns:
        The model, only its adapter trainable.

    Raises:
        SystemExit: If the starting adapter is not this adapter's shape.
    """
    set_memory_limits()
    model = load_model(folder)
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    if init is not None:
        try:
            check_adapter_file(model, init)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"{SCRIPT}: {error}") from error
        load_adapter(model, init)
    return model


# ----------------------------------------------------------------- the loop


def step_metrics(
    step: int,
    records: list[RolloutRecord],
    length: grpo_records.StepLength,
    rows: StepRows,
    update: Update | None,
    times: tuple[float, float],
    adapter: Path | None,
) -> dict[str, Any]:
    """One step's line of metrics.jsonl: the project's fields, then ours.

    Args:
        step: The step.
        records: Its rollouts.
        length: Its tokens per call (median and mean).
        rows: Its training rows.
        update: Its update, or None when no group had contrast.
        times: Seconds spent on the rollouts, and on the whole step.
        adapter: The adapter the update saved, or None.

    Returns:
        The metrics.
    """
    ratios = update.ratios if update else []
    return {
        "stage": grpo_records.STAGE,
        "step": step,
        "harnesses": sorted({r.harness for r in records}),
        "accuracy": sum(r.correct for r in records) / len(records),
        "mean_reward": statistics.fmean(r.reward for r in records),
        "mean_calls": statistics.fmean(r.n_calls for r in records),
        "tokens_per_call": length.median,
        "groups_without_contrast": rows.flat_groups,
        "groups": rows.groups,
        "train_rows": len(rows.items),
        "train_tokens": rows.tokens,
        "skipped_long_rows": rows.skipped_long,
        "untrainable_rollouts": sum(1 for r in records if not r.trainable),
        "importance_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "loss_per_token": update.loss_per_token if update else None,
        "rollout_seconds": round(times[0], 1),
        "step_seconds": round(times[1], 1),
        "peak_gb": round(mx.get_peak_memory() / GIGABYTE, 2),
        # This chapter adds:
        "tokens_per_call_mean": length.mean,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "clipped_share": update.clipped_share if update else None,
        "by_harness": {h: asdict(v) for h, v in rows.by_harness.items()},
        "adapter": adapter.name if adapter else None,
    }


def step_lines(
    metrics: dict[str, Any], steps: int, elapsed: float, config: Config
) -> list[str]:
    """The project's step line, then what this chapter adds under it.

    Args:
        metrics: The step's metrics.
        steps: Steps in the run.
        elapsed: Seconds since training started.
        config: The run's settings.

    Returns:
        The lines.
    """
    ratio = metrics["importance_ratio_mean"]
    ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
    clock = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    lines = [
        f"[grpo] step {metrics['step']}/{steps} · {clock} · "
        f"reward {metrics['mean_reward']:.2f} · "
        f"correct {PERCENT * metrics['accuracy']:.0f}% · "
        f"calls {metrics['mean_calls']:.1f} · "
        f"no-signal groups {metrics['groups_without_contrast']}/{metrics['groups']} · "
        f"ratio {ratio_text} · peak {metrics['peak_gb']:.1f} GB",
        f"{CONTINUATION}tokens/call: median {metrics['tokens_per_call']:.0f} (the "
        f"guard reads this; limit {config.guard_tokens:.0f}"
        f"{'' if config.guard else ', off with --attempt1'}) · mean "
        f"{metrics['tokens_per_call_mean']:.0f}",
    ]
    parts = []
    for harness, given in sorted(metrics["by_harness"].items()):
        if given["rows"]:
            average = given["tokens"] / given["rows"]
            parts.append(
                f"{harness} {given['rows']} rows, {average:,.0f} tokens on average, "
                f"{given['trained']:,} trained"
            )
        else:
            parts.append(f"{harness} no row (its group had no contrast)")
    skipped = metrics["skipped_long_rows"]
    lines.append(
        f"{CONTINUATION}rows: {' · '.join(parts)} · {skipped} over "
        f"{config.max_tokens:,} tokens skipped"
    )
    if metrics["adapter"]:
        lines.append(
            f"{CONTINUATION}ratio per token {metrics['ratio_min']:.3f} to "
            f"{metrics['ratio_max']:.3f}, {PERCENT * metrics['clipped_share']:.1f}% "
            f"outside 0.8 to 1.2 · loss/token {metrics['loss_per_token']:.4f} · "
            f"{metrics['adapter']}, now the engine's"
        )
    else:
        lines.append(f"{CONTINUATION}no group had contrast: no update this step")
    return lines


def write_line(path: Path, line: dict[str, Any]) -> None:
    """Append one JSON line to a file.

    Args:
        path: The file.
        line: The record.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def train(
    config: Config,
    model: nn.Module,
    plan: list[list[tuple[Task, str]]],
    hooks: Hooks,
    out_dir: Path,
    init: Path | None,
    say: Callable[[str], None] = print,
) -> Outcome:
    """Run GRPO: rollouts, the guard, the update, the adapter, step by step.

    Args:
        config: The run's settings.
        model: The trainer's model, its adapter loaded from ``init``.
        plan: Every step's (task, harness) groups.
        hooks: Runs rollouts and switches the engine's adapter.
        out_dir: The run's folder (adapters, rollouts.jsonl, metrics.jsonl).
        init: The adapter training starts from (None: the base model).
        say: Where lines go.

    Returns:
        How the run ended.

    Raises:
        RunStopped: If a step's rollouts could not all be run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate, bias_correction=config.bias_correction
    )
    outcome = Outcome(kept=init)
    history: list[Path | None] = [init]  # the adapter each step samples with: [-1]
    mx.reset_peak_memory()
    started_all = time.perf_counter()
    for step, groups in enumerate(plan, start=1):
        started = time.perf_counter()
        jobs = [
            Job(task, harness, step, number, index)
            for number, (task, harness) in enumerate(groups)
            for index in range(config.group_size)
        ]
        records = hooks.run_rollouts(jobs)
        append_records(out_dir / grpo_records.ROLLOUTS_FILE, records)
        outcome.steps_run = step
        rollout_s = time.perf_counter() - started
        read = [read_back(record, step) for record in records]
        verdict = grpo_records.length_guard(read, history, config.guard_tokens)
        if verdict.stop and config.guard:
            kept = verdict.kept
            say(
                f"[grpo] step {step}: replies ballooned to {verdict.median:.0f} tokens "
                f"per call (guard {config.guard_tokens:.0f}); stopping, keeping "
                f"{kept.name if kept else 'the base model'}"
            )
            write_line(
                out_dir / grpo_records.METRICS_FILE,
                {
                    "stage": grpo_records.STAGE,
                    "step": step,
                    "stopped": "length_guard",
                    "tokens_per_call": verdict.median,
                    "kept": kept.name if kept else None,
                },
            )
            # The project's code stops here; this one also puts the engine
            # back on the kept adapter, so the next rollout uses it.
            using = hooks.use_adapter(kept)
            say(f"{CONTINUATION}the engine samples with {using} again")
            outcome.stopped_at, outcome.kept = step, kept
            break
        if verdict.stop:  # --attempt1: no guard, but say where it would stop
            say(f"{CONTINUATION}(a guard would stop here: median {verdict.median:.0f})")
        size = config.group_size
        grouped = [
            [asdict(r) for r in records[g * size : (g + 1) * size]]
            for g in range(len(groups))
        ]
        rows = step_rows(grouped, config.max_tokens)
        update, adapter = None, None
        if rows.items:
            update = grpo_update(model, optimizer, rows.items)
            adapter = save_adapter(model, out_dir / ADAPTER_NAME.format(step=step))
            hooks.use_adapter(adapter)
            history.append(adapter)
            outcome.kept = adapter
        for harness, given in rows.by_harness.items():
            total = outcome.exposure.setdefault(harness, [0, 0, 0])
            for index, value in enumerate((given.rows, given.tokens, given.trained)):
                total[index] += value
        times = (rollout_s, time.perf_counter() - started)
        length = grpo_records.step_length(read)
        metrics = step_metrics(step, records, length, rows, update, times, adapter)
        write_line(out_dir / grpo_records.METRICS_FILE, metrics)
        elapsed = time.perf_counter() - started_all
        for line in step_lines(metrics, len(plan), elapsed, config):
            say(line)
    outcome.final = finish(model, out_dir, outcome)
    return outcome


def read_back(record: RolloutRecord, step: int) -> grpo_records.Rollout:
    """Read a fresh record the way every run's records are read.

    Args:
        record: A rollout this run just made.
        step: Its step.

    Returns:
        The rollout as ``grpo_records`` reads it, so the guard and steps.py
        count tokens per call the same way.
    """
    return grpo_records.from_slim(grpo_records.slim(asdict(record), step))


def finish(model: nn.Module, out_dir: Path, outcome: Outcome) -> Path | None:
    """Write ``adapter_final.safetensors``: the kept adapter, or the last weights.

    Args:
        model: The trainer's model.
        out_dir: The run's folder.
        outcome: How the run ended.

    Returns:
        The final adapter, or None when the guard kept the base model.
    """
    final = out_dir / FINAL_NAME
    if outcome.stopped_at is None:
        return save_adapter(model, final)
    if outcome.kept is None:  # the project raises here: nothing to keep
        return None
    shutil.copyfile(outcome.kept, final)
    return final


# ------------------------------------------------------------ the guard demo


@dataclass(frozen=True)
class GuardStep:
    """One recorded step, as the guard saw it.

    Attributes:
        number: The step.
        sampled_with: The adapter its rollouts were sampled with.
        length: Its tokens per call, both ways.
        correct: Its share of right rollouts, in percent.
        verdict: The guard's verdict.
    """

    number: int
    sampled_with: str
    length: grpo_records.StepLength
    correct: float
    verdict: grpo_records.GuardVerdict


def walk(steps: list[list[grpo_records.Rollout]], limit: float) -> list[GuardStep]:
    """Feed a recorded run to the guard, step by step, until it stops.

    Args:
        steps: The run's rollouts, step by step.
        limit: The guard's limit on the median tokens per call.

    Returns:
        Every step the guard saw; the last one stopped it, if it stopped.
    """
    history = [START_LABEL]
    seen = []
    for number, step in enumerate(steps, start=1):
        verdict = grpo_records.length_guard(step, history, limit)
        correct = PERCENT * sum(r.correct for r in step) / len(step)
        length = grpo_records.step_length(step)
        seen.append(GuardStep(number, history[-1], length, correct, verdict))
        if verdict.stop:
            break
        history.append(ADAPTER_NAME.format(step=number).removesuffix(ADAPTER_SUFFIX))
    return seen


def guard_demo(
    folder: Path,
    source: str,
    limit: float = GUARD_TOKENS,
    kept_by_project: str | None = None,
    say: Callable[[str], None] = print,
) -> None:
    """Replay attempt 1's recorded rollouts through this script's length guard.

    Args:
        folder: Attempt 1's records (the bundle, or the project's own).
        source: Where they come from, in words.
        limit: The guard's limit (``--guard-tokens``).
        kept_by_project: The adapter the project kept, such as
            ``adapter_step004``, or None.
        say: Where lines go.
    """
    steps = grpo_records.by_step(grpo_records.read_rollouts(folder))
    say(f"{SCRIPT} --guard-demo · attempt 1's rollouts, step by step, past the guard")
    say(f"  records   {source}")
    say(f"            {len(steps)} steps x {len(steps[0])} rollouts")
    say(
        "  guard     after a step's rollouts, before its update: stop when tokens "
        f"per call pass {limit:.0f},"
    )
    say("            and keep the adapter from before the last update")
    say("  tokens    per call: a rollout's sampled tokens / its calls; the median and")
    say("            the mean over the step's rollouts. The guard reads the median.")
    say("")
    say("  step  sampled with      median   mean  correct  the guard")
    seen = walk(steps, limit)
    for one in seen:
        head = (
            f"  {one.number:>4}  {one.sampled_with:<16} {one.length.median:>7.0f} "
            f"{one.length.mean:>6.0f}  {one.correct:>6.0f}%  "
        )
        if one.verdict.stop:
            verdict = one.verdict
            say(f"{head}STOP: {verdict.blamed} made these; keep {verdict.kept}")
        else:
            made = ADAPTER_NAME.format(step=one.number).removesuffix(ADAPTER_SUFFIX)
            say(f"{head}under {limit:.0f}: update {one.number} -> {made}")
    say("")
    last = seen[-1]
    if not last.verdict.stop:
        say(f"  the guard never stopped: every median stayed under {limit:.0f}")
    else:
        passed = max(seen[:-1], key=lambda one: one.length.median, default=None)
        say(
            f"  stopped at step {last.number}, before update {last.number}: "
            f"adapter_final = {last.verdict.kept}"
        )
        say(f"  (the run itself went on to make {len(steps)} updates)")
        if passed is not None:
            median = passed.length.median
            times = median / seen[0].length.median
            say(
                f"  the longest replies it let through: median {median:.0f} at step "
                f"{passed.number}, {times:.1f} times step 1's"
            )
    if kept_by_project:
        say(
            f"  the project kept {kept_by_project}, the adapter step "
            f"{adapter_number(kept_by_project) + 1}'s rollouts came from"
        )
        say("  (replay.py attempt1 shows how it scored)")
    for line in guard_sweep(steps):
        say(line)


def adapter_number(name: str) -> int:
    """The step whose update saved an adapter, read from its name.

    Args:
        name: Such as ``adapter_step004``.

    Returns:
        Its step, such as 4.
    """
    return int(name.removesuffix(ADAPTER_SUFFIX).rsplit("step", 1)[1])


def guard_sweep(steps: list[list[grpo_records.Rollout]]) -> list[str]:
    """Where a few guard limits would stop the same run, and what each keeps.

    Args:
        steps: The run's rollouts, step by step.

    Returns:
        One line per limit.
    """
    lines = ["  limits compared on the same records:"]
    for limit in GUARD_SWEEP:
        last = walk(steps, limit)[-1]
        found = (
            f"stops at step {last.number}, keeps {last.verdict.kept}"
            if last.verdict.stop
            else "never stops"
        )
        lines.append(f"    --guard-tokens {limit:<4} {found}")
    return lines


# --------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tasks-per-step", type=int, default=TASKS_PER_STEP)
    parser.add_argument("--group", type=int, default=GROUP_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--init", default=None, help=f"an adapter file, or {BASE!r}")
    parser.add_argument("--attempt1", action="store_true", help="the collapsed run")
    parser.add_argument("--guard-demo", action="store_true", help="replay attempt 1")
    parser.add_argument("--guard-tokens", type=float, default=GUARD_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TRAIN_TOKENS)
    parser.add_argument("--proxy-url", default=PROXY_URL)
    parser.add_argument("--run", default=None, help="the run's folder name")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    parser.add_argument(
        "--records", type=Path, default=None, help="the project's runs/main folder"
    )
    args = parser.parse_args()
    if min(args.steps, args.tasks_per_step, args.concurrency) < 1 or args.group < 2:
        parser.error("--steps, --tasks-per-step, --concurrency >= 1; --group >= 2")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be above 0")
    if args.guard_tokens <= 0 or args.max_tokens < 2:
        parser.error("--guard-tokens must be above 0, --max-tokens at least 2")
    return args


def config_from(args: argparse.Namespace) -> Config:
    """The run's settings: the project's, or attempt 1's with ``--attempt1``.

    Args:
        args: The command line.

    Returns:
        The settings.
    """
    attempt1 = args.attempt1
    default_lr = ATTEMPT1_LEARNING_RATE if attempt1 else LEARNING_RATE
    return Config(
        steps=args.steps,
        tasks_per_step=args.tasks_per_step,
        group_size=args.group,
        learning_rate=args.lr or default_lr,
        bias_correction=not attempt1,
        call_bonus=attempt1,
        guard=not attempt1,
        guard_tokens=args.guard_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def init_adapter(value: str | None) -> Path | None:
    """The adapter training starts from.

    Args:
        value: ``--init``: a file, ``base``, or None for chapter 25's.

    Returns:
        The file, or None for the base model.

    Raises:
        SystemExit: If the file is missing, naming how to make it.
    """
    if value == BASE:
        return None
    path = Path(value).expanduser() if value else course_home() / INIT_ADAPTER
    if not path.is_file():
        raise SystemExit(
            f"{SCRIPT}: no adapter at {short_path(path)}. Make chapter 25's: in "
            "25-sft-in-harnesses, ./start.sh oracle; uv run collect.py --limit 60; "
            "./start.sh stop; uv run train_sft.py --no-bias-correction. Or start "
            f"from the base model with --init {BASE}"
        )
    return path.resolve()


def run_folder(args: argparse.Namespace) -> Path:
    """The run's folder, refused if it already holds a run.

    Args:
        args: The command line.

    Returns:
        ``COURSE_HOME/26-grpo-in-harnesses/grpo/<run>``.

    Raises:
        SystemExit: If a run is already there.
    """
    name = args.run or (ATTEMPT1_RUN_NAME if args.attempt1 else RUN_NAME)
    folder = chapter_home() / GRPO_FOLDER / name
    if (folder / grpo_records.METRICS_FILE).exists():
        raise SystemExit(
            f"{SCRIPT}: {short_path(folder)} already holds a run: --run <name> "
            "for another, or move it away"
        )
    return folder


def setup_lines(
    config: Config, init: Path | None, out_dir: Path, attempt1: bool
) -> list[str]:
    """Say what the run will do before it does it.

    Args:
        config: The run's settings.
        init: The starting adapter.
        out_dir: The run's folder.
        attempt1: Whether this is ``--attempt1``.

    Returns:
        The lines.
    """
    rollouts = config.tasks_per_step * config.group_size
    bonus = (
        "the article's call bonus (up to 0.1, none at 8 calls)"
        if config.call_bonus
        else f"the token bonus (up to 0.1, none at {TOKEN_BUDGET:,} tokens)"
    )
    guard = (
        f"stop when tokens per call (the median rollout) pass {config.guard_tokens:.0f}"
        f", keep the adapter from before the last update"
        if config.guard
        else "none (--attempt1)"
    )
    correction = "on" if config.bias_correction else "off"
    lines = [
        f"{SCRIPT} · GRPO through the harnesses"
        + (" · attempt 1's settings" if attempt1 else ""),
        f"  start     {short_path(init) if init else 'the base model (no adapter)'}",
        f"  steps     {config.steps}, each {config.tasks_per_step} training tasks x "
        f"{config.group_size} rollouts = {rollouts}, harnesses alternating "
        f"({', '.join(config.harnesses)}), task order seed {config.seed}",
        "  rollouts  temperature 1 through the proxy, at most 8 calls, "
        f"{CONCURRENCY} at a time",
        f"  reward    1 + {bonus} when right, else 0",
        f"  update    clipped loss (0.8 to 1.2) on the engine's recorded log-"
        f"probabilities · AdamW lr {config.learning_rate:g}, bias correction "
        f"{correction} · rows over {config.max_tokens:,} tokens skipped · each "
        "shared prefix trained once",
        f"  guard     {guard}",
        f"  out       {short_path(out_dir)}",
    ]
    if attempt1:
        lines.append(
            "  not restored: the engine's bfloat16 log-probabilities (this engine "
            "records float32; chapter 15 measured what that changes: ratio means "
            "just under 1)"
        )
    return lines


def make_hooks(
    config: Config, client: CaptureClient, proxy_url: str, engine_url: str, run: str
) -> Hooks:
    """The real hooks: rollouts through the harnesses, the engine's adapter.

    Args:
        config: The run's settings.
        client: The proxy's client.
        proxy_url: The proxy's root.
        engine_url: The engine.
        run: The run's name.

    Returns:
        The hooks.
    """
    setup = RolloutSetup(client, proxy_url, config.reward, run, Options())

    def run_rollouts(jobs: list[Job]) -> list[RolloutRecord]:
        """Run a step's rollouts; any rollout without a record stops the run.

        Args:
            jobs: The step's rollouts.

        Returns:
            The records, in job order.

        Raises:
            RunStopped: If a rollout produced no record.
        """
        records, failed = run_step(jobs, setup, CONCURRENCY)
        if failed:
            raise RunStopped(
                f"{len(failed)} of {len(jobs)} rollouts made no record: a step with "
                "a missing rollout would compare groups of different sizes"
            )
        return records

    return Hooks(run_rollouts, adapter_switch(engine_url))


def summary_lines(outcome: Outcome, out_dir: Path, seconds: float) -> list[str]:
    """What the run left, and how much each harness trained the model.

    Args:
        outcome: How it ended.
        out_dir: Its folder.
        seconds: How long it took.

    Returns:
        The lines.
    """
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    final = short_path(outcome.final) if outcome.final else "none: the base model"
    kept = outcome.kept.name if outcome.kept else "the base model"
    lines = [
        f"done      {outcome.steps_run} steps in {minutes} min {rest} s · trainer "
        f"peak {mx.get_peak_memory() / GIGABYTE:.2f} GB (MLX's count, the weights "
        "included)",
        f"adapter   {final} (= {kept})",
        f"records   {short_path(out_dir)}: rollouts.jsonl, metrics.jsonl "
        "(uv run steps.py reads them)",
        "rows by harness, every update together:",
    ]
    for harness, (rows, tokens, trained) in sorted(outcome.exposure.items()):
        average = f"{tokens / rows:,.0f} tokens on average" if rows else "no row"
        lines.append(
            f"  {harness:<14} {rows:>3} rows · {average} · {trained:,} tokens trained"
        )
    return lines


def main() -> None:
    """Train, or replay attempt 1 through the guard with ``--guard-demo``.

    Raises:
        SystemExit: If the stack is down or not the model, the starting
            adapter is missing, a harness is refused, the run's folder holds
            a run, or a step's rollouts could not all be run.
    """
    args = parse_args()
    if args.guard_demo:
        if args.records:
            main_dir = args.records.expanduser()
            folder, summaries = main_dir / ATTEMPT1_FOLDER, main_dir
            source = f"{short_path(folder)} (the project's own records)"
        else:
            folder = grpo_records.BUNDLE_DIR / ATTEMPT1_RUN_NAME
            summaries, source = grpo_records.LOGS_DIR, BUNDLE_SOURCE
        try:
            kept = grpo_records.named_adapter(summaries / ATTEMPT1_SUMMARY)
            name = Path(kept).name.removesuffix(ADAPTER_SUFFIX)
            guard_demo(folder, source, args.guard_tokens, name)
        except grpo_records.RecordsError as error:
            raise SystemExit(f"{SCRIPT}: {error}") from error
        return
    config = config_from(args)
    init = init_adapter(args.init)
    out_dir = run_folder(args)
    tasks = load_tasks(TRAIN_FILE)
    plan = plan_groups(
        tasks, config.harnesses, config.steps, config.tasks_per_step, config.seed
    )
    client = CaptureClient(args.proxy_url)
    try:
        engine_url = str(client.health().get("upstream", ""))
    except ProxyError as error:
        raise SystemExit(f"{error}: start the stack first (./start.sh)") from error
    check_engine(engine_url)
    for line in setup_lines(config, init, out_dir, args.attempt1):
        print(line)
    print(f"  engine    {engine_url} (the model) · proxy {args.proxy_url}")
    print("versions")
    try:
        for harness in config.harnesses:
            check_harness(harness, plan[0][0][0], Options())
    except HarnessInstallError as error:
        raise SystemExit(f"REFUSED · {error}") from error
    folder, which = pick_model(args.fast)
    started = time.perf_counter()
    model = load_policy(folder, init)
    loaded = mx.get_active_memory() / GIGABYTE
    print(f"  model     {short_path(folder)} ({which}) · {loaded:.2f} GB in MLX")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = short_path(init) if init else BASE
    note = {**asdict(config), "init": start, "model": which}
    (out_dir / NOTE_NAME).write_text(json.dumps(note, indent=2) + "\n", "utf-8")
    hooks = make_hooks(config, client, args.proxy_url, engine_url, out_dir.name)
    print(f"  the engine samples with {hooks.use_adapter(init)}")
    sys.stdout.flush()
    try:
        with attention(chunked=True), layer_checkpointing(model, on=True):
            outcome = train(config, model, plan, hooks, out_dir, init, print)
    except RunStopped as error:
        raise SystemExit(f"{SCRIPT}: stopped: {error}") from error
    for line in summary_lines(outcome, out_dir, time.perf_counter() - started):
        print(line)
    mx.clear_cache()


if __name__ == "__main__":
    main()
