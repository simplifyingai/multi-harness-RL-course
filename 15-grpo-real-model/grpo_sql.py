# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""GRPO on LFM2.5-2.6B, in a loop we own: sample, grade, compare, one update.

Chapter 6's GRPO on tokens, as the project's ``train_grpo`` does it
(``mhrl/training/grpo.py``), starting from chapter 14's SFT adapter. Each
step picks 2 training questions and samples 4 replies to each at temperature
1 with chapter 12's sampler, up to the 1,024-token cap, keeping every
sampled token's id and float32 log-probability. Each reply's SQL is run and
graded through ``answer.txt``; a right one earns 1 plus a bonus of up to 0.1
for fewer tokens, a wrong one 0. Each group's rewards become advantages;
every reply of a group with contrast becomes a training row (the prompt's
ids, the sampled ids, their recorded log-probabilities, its advantage); one
AdamW step (learning rate 1e-5, bias correction on) follows, on the clipped
loss. The importance ratio of every trained token, new probability over the
recorded one, is worked out before the update.

Run it with ``uv run grpo_sql.py`` (20 steps; tens of minutes). The flags
change one thing each: ``--retokenize`` trains on the decoded-and-re-encoded
reply, ``--logprob-dtype bf16`` records bfloat16 log-probabilities as the
project's engine once did, ``--top-k 50`` draws from the 50 likeliest tokens
only, ``--kl 0.05`` adds a penalty for drifting from the starting adapter,
``--float32`` runs the whole model in float32. ``--steps 4`` is fast mode;
``--fast`` also uses Liquid's export. Everything is saved under
``$COURSE_HOME/15-grpo-real-model``: each step's metrics and replies as soon
as the step ends, the adapter once, after the last step (a run stopped early
keeps its logs, not its adapter).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import report
from grading import FINISH_LENGTH, grade_reply
from grpo import NO_KL, Update, grpo_update, set_memory_limits, trainer_logprobs
from lfm import load_folder, to_float32
from lfm_files import (
    FP32,
    GIGABYTE,
    GROUP_SIZE,
    GRPO_LEARNING_RATE,
    GRPO_STEPS,
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    ORDER_SEED,
    OUTPUT_CAP,
    QUESTIONS_PER_STEP,
    SAMPLE_SEED,
    SFT_ADAPTER,
    TOKEN_BUDGET,
    TRAIN_TEMPERATURE,
    adapter_path,
    init_path,
    metrics_path,
    pick_model,
    require_init,
    rollouts_path,
    run_name,
    work_folder,
)
from lora import adapter_weights, apply_lora, load_adapter, save_adapter
from mlx_lm.tokenizer_utils import TokenizerWrapper
from project_runs import ATTEMPT_1, ATTEMPT_2, ProjectRunError, read_run
from retokenize import align_by_position, reencode
from rewards import group_advantages, group_outcome, reward_for
from rows import (
    SAMPLED,
    TrainItem,
    call_sequence,
    encode,
    item_from_sequence,
    prompt_ids,
    training_order,
)
from sampler import DTYPES, NO_TOP_K, Draw, Sampled, sample
from tasks import TRAIN_FILE, Task, load_tasks

SCRIPT = "grpo_sql.py"
TITLE = "GRPO on the real model"
FAST_STEPS = 4  # the plan's fast mode: --steps 4
TRAIN_WORK = "train"  # work/train/<run>/<task>-<reply>/answer.txt
SECONDS_DECIMALS = 1  # how the metrics log rounds times, as the project's does
GB_DECIMALS = 2  # and peak memory


@dataclass(frozen=True)
class Graded:
    """One sampled reply, graded and rewarded.

    Attributes:
        task_id: The question.
        tokens: How many tokens the reply has (its end of turn included).
        finish_reason: ``"stop"`` or ``"length"``.
        correct: Whether chapter 1's grader accepted its answer.txt.
        reason: ``"correct"`` or why not.
        reward: 1 plus the token bonus if right, else 0.
        advantage: Its group-relative advantage, or None (no contrast).
        entropy: The mean entropy of the distributions it was drawn from.
        full_entropy: The same, over the full float32 distributions.
        changed: With ``--retokenize``: re-encoding changed its ids.
        masked: With ``--retokenize``: positions masked out of training.
        seconds: How long drawing it took (varies from run to run).
        text: The reply, decoded.
    """

    task_id: str
    tokens: int
    finish_reason: str
    correct: bool
    reason: str
    reward: float
    advantage: float | None
    entropy: float
    full_entropy: float
    changed: bool
    masked: int
    seconds: float
    text: str


@dataclass(frozen=True)
class StepResult:
    """What one step sampled, what it trained on, and every token's ratio.

    Attributes:
        step: The step's number, from 1.
        replies: Its 8 graded replies, group by group.
        outcomes: Each group's outcome (all right, right and wrong, all wrong).
        no_signal: Groups whose rewards were all equal (skipped).
        update: The update, with every trained token's ratio.
        advantages: Each trained row's advantage, in the update's row order.
        engine_ratios: Per trained row, each token's ratio had the engine's
            float32 full-distribution log-probability been recorded instead.
        moved: Per trained row, each token's recorded old log-probability
            minus the engine's (0 unless ``--logprob-dtype bf16`` or
            ``--top-k`` changed what was recorded).
        after_difference: Per trained row, for each trained token, whether it
            comes after a re-encoding difference (``--retokenize``).
        unmasked: With ``--retokenize``: the ratio each masked position would
            have had, kept with the sampled log-probability at its position.
        sample_seconds: How long the step's replies took.
        seconds: How long the whole step took.
        peak_bytes: MLX's peak memory so far.
    """

    step: int
    replies: list[Graded]
    outcomes: list[str]
    no_signal: int
    update: Update
    advantages: list[float]
    engine_ratios: list[list[float]]
    moved: list[list[float]]
    after_difference: list[list[bool]]
    unmasked: list[float]
    sample_seconds: float
    seconds: float
    peak_bytes: int


@dataclass(frozen=True)
class Settings:
    """How this run differs from the project's, and how it draws.

    Attributes:
        draw: Temperature, top-k and log-probability precision.
        retokenize: Train on the decoded-and-re-encoded reply.
        kl: The drift penalty's weight (0: none, the project's).
        lr: The learning rate.
        float32: The whole model runs in float32.
    """

    draw: Draw
    retokenize: bool
    kl: float
    lr: float
    float32: bool


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=None, help="optimizer steps")
    parser.add_argument(
        "--init", default=SFT_ADAPTER, help="the adapter to start from (chapter 14's)"
    )
    parser.add_argument(
        "--retokenize", action="store_true", help="train on decoded, re-encoded text"
    )
    parser.add_argument(
        "--logprob-dtype",
        choices=DTYPES,
        default=FP32,
        help="the precision the sampler records log-probabilities in",
    )
    parser.add_argument("--top-k", type=int, default=NO_TOP_K, help="0 keeps all")
    parser.add_argument("--kl", type=float, default=NO_KL, help="drift penalty")
    parser.add_argument("--lr", type=float, default=GRPO_LEARNING_RATE)
    parser.add_argument(
        "--float32", action="store_true", help="run the whole model in float32"
    )
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Liquid's export and {FAST_STEPS} steps",
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = FAST_STEPS if args.fast else GRPO_STEPS
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.top_k < NO_TOP_K:
        parser.error("--top-k must be 0 (keep all) or more")
    if args.kl < 0:
        parser.error("--kl must be 0 or more")
    if args.lr <= 0:
        parser.error("--lr must be above 0")
    return args


def name_parts(args: argparse.Namespace) -> list[str]:
    """List what this run changes, for its adapter's file name.

    Args:
        args: The command line.

    Returns:
        Short words, such as ``["topk50", "steps4"]``; empty for the default.
    """
    parts = []
    if args.float32:
        parts.append("float32")
    if args.retokenize:
        parts.append("retokenize")
    if args.logprob_dtype != FP32:
        parts.append(args.logprob_dtype)
    if args.top_k != NO_TOP_K:
        parts.append(f"topk{args.top_k}")
    if args.kl != NO_KL:
        parts.append(f"kl{args.kl:g}")
    if args.lr != GRPO_LEARNING_RATE:
        parts.append(f"lr{args.lr:g}")
    if args.init != SFT_ADAPTER:
        parts.append(f"init{init_path(args.init).stem}")
    if args.steps != GRPO_STEPS:
        parts.append(f"steps{args.steps}")
    if args.seed != SAMPLE_SEED:
        parts.append(f"seed{args.seed}")
    return parts


def row_for(
    prompt: list[int],
    reply: Sampled,
    advantage: float,
    reencoded: list[int] | None,
) -> tuple[TrainItem | None, TrainItem | None, list[bool]]:
    """Turn a graded reply into its training row, as the capture proxy records it.

    Args:
        prompt: The prompt's token ids.
        reply: The sampled reply.
        advantage: Its advantage.
        reencoded: With ``--retokenize``, the reply decoded and encoded again,
            which is trained on instead of the sampled ids; else None.

    Returns:
        The row trained on; the same row with the engine's float32
        full-distribution log-probabilities as old ones; and, per trained
        token, whether it comes after a re-encoding difference.
    """
    tokens, recorded, engine = reply.tokens, reply.logprobs, reply.full_logprobs
    mask = [SAMPLED] * len(tokens)
    later = [False] * len(tokens)
    if reencoded is not None:
        aligned = align_by_position(tokens, recorded, reencoded)
        borrowed = align_by_position(tokens, engine, reencoded)
        tokens, recorded, engine, mask = (
            aligned.ids,
            aligned.logprobs,
            borrowed.logprobs,
            aligned.mask,
        )
        first = aligned.first_difference
        later = [first is not None and i > first for i in range(len(tokens))]
    item = item_from_sequence(call_sequence(prompt, tokens, recorded, mask), advantage)
    same = item_from_sequence(call_sequence(prompt, tokens, engine, mask), advantage)
    flags = [later[i] for i, m in enumerate(mask) if m]
    return item, same, flags


def unmasked_ratios(
    model: nn.Module, prompt: list[int], reply: Sampled, reencoded: list[int]
) -> list[float]:
    """Work out what the masked positions of a re-encoded reply would have seen.

    A pipeline that kept every position would compare each re-encoded token
    with the log-probability recorded at the same position, which belongs to
    another token. This scores exactly those positions, by the trainer's
    route, before the update, and trains on none of them.

    Args:
        model: The network with the adapter being trained.
        prompt: The prompt's token ids.
        reply: The sampled reply.
        reencoded: The reply decoded and encoded again.

    Returns:
        One ratio per masked position both replies reach (empty when
        re-encoding changed nothing).
    """
    ids = reencoded
    aligned = align_by_position(reply.tokens, reply.logprobs, ids)
    shared = min(len(ids), len(reply.tokens))
    masked = [i for i in range(shared) if not aligned.mask[i]]
    if not masked:
        return []
    item = TrainItem(
        prompt + ids,
        [len(prompt) + i for i in masked],
        [reply.logprobs[i] for i in masked],
    )
    new = trainer_logprobs(model, item).tolist()
    return [math.exp(n - o) for n, o in zip(new, item.old_logprobs, strict=True)]


def sample_group(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    task: Task,
    settings: Settings,
    folder: str,
) -> tuple[list[int], list[Sampled], list[Graded], list[list[int] | None]]:
    """Draw and grade one group: 4 replies to one question.

    Args:
        model: The network with the adapter being trained.
        tokenizer: Its tokenizer.
        task: The question.
        settings: How replies are drawn.
        folder: The run's work folder name.

    Returns:
        The prompt's ids, the sampled replies, their grades (advantages not
        yet known), and with ``--retokenize`` each reply decoded and encoded
        again (else None for each).
    """
    prompt = prompt_ids(tokenizer, task)
    sampled, graded, reencoded = [], [], []
    eos = tokenizer.eos_token_ids
    for index in range(GROUP_SIZE):
        reply = sample(model, eos, prompt, settings.draw, OUTPUT_CAP)
        text = tokenizer.decode(reply.tokens)
        cut_off = reply.finish_reason == FINISH_LENGTH
        where = work_folder(TRAIN_WORK, folder, f"{task.task_id}-{index + 1}")
        verdict = grade_reply(task, text, where, cut_off)
        changed, masked, ids = False, 0, None
        if settings.retokenize:
            ids = reencode(
                reply.tokens, tokenizer.decode, lambda t: encode(tokenizer, t)
            )
            aligned = align_by_position(reply.tokens, reply.logprobs, ids)
            changed, masked = aligned.differs, aligned.masked
        sampled.append(reply)
        reencoded.append(ids)
        graded.append(
            Graded(
                task_id=task.task_id,
                tokens=len(reply.tokens),
                finish_reason=reply.finish_reason,
                correct=verdict.correct,
                reason=verdict.reason,
                reward=reward_for(verdict.correct, len(reply.tokens), TOKEN_BUDGET),
                advantage=None,
                entropy=statistics.fmean(reply.entropies),
                full_entropy=statistics.fmean(reply.full_entropies),
                changed=changed,
                masked=masked,
                seconds=reply.seconds,
                text=text,
            )
        )
    return prompt, sampled, graded, reencoded


def one_step(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    optimizer: optim.Optimizer,
    tasks: list[Task],
    step: int,
    settings: Settings,
    reference: dict[str, mx.array] | None,
    folder: str,
) -> StepResult:
    """Sample 2 groups, grade them, and make one update (the project's step).

    Args:
        model: The network with the adapter being trained.
        tokenizer: Its tokenizer.
        optimizer: AdamW, built once for the run.
        tasks: This step's questions.
        step: The step's number.
        settings: How this run differs from the project's.
        reference: The start adapter's weights (``--kl`` only).
        folder: The run's work folder name.

    Returns:
        What the step did.
    """
    started = time.perf_counter()
    items, engine_items, flags, advantages_kept = [], [], [], []
    unmasked: list[float] = []
    replies: list[Graded] = []
    outcomes: list[str] = []
    no_signal = 0
    for task in tasks:
        prompt, sampled, graded, reencoded = sample_group(
            model, tokenizer, task, settings, folder
        )
        for reply, ids in zip(sampled, reencoded, strict=True):
            if ids is not None:  # measured before the update, trained on never
                unmasked.extend(unmasked_ratios(model, prompt, reply, ids))
        rewards = [reply.reward for reply in graded]
        outcomes.append(group_outcome(rewards))
        advantages = group_advantages(rewards)
        if advantages is None:  # every reward the same: nothing to compare
            no_signal += 1
            replies.extend(graded)
            continue
        rows = zip(sampled, graded, advantages, reencoded, strict=True)
        for reply, grade, advantage, ids in rows:
            replies.append(replace(grade, advantage=advantage))
            item, same, later = row_for(prompt, reply, advantage, ids)
            if item is not None and same is not None:
                items.append(item)
                engine_items.append(same)
                flags.append(later)
                advantages_kept.append(advantage)
    sample_seconds = time.perf_counter() - started
    update = grpo_update(model, optimizer, items, settings.kl, reference)
    # What recording bf16 or top-k numbers moved, token by token, and the
    # same tokens' ratio against the engine's own number: ratio x
    # exp(recorded - engine), since ratio = exp(new - recorded).
    moved = [
        [
            old - wide
            for old, wide in zip(item.old_logprobs, same.old_logprobs, strict=True)
        ]
        for item, same in zip(items, engine_items, strict=True)
    ]
    engine_ratios = [
        [ratio * math.exp(shift) for ratio, shift in zip(ratios, shifts, strict=True)]
        for ratios, shifts in zip(update.ratios, moved, strict=True)
    ]
    return StepResult(
        step=step,
        replies=replies,
        outcomes=outcomes,
        no_signal=no_signal,
        update=update,
        advantages=advantages_kept,
        engine_ratios=engine_ratios,
        moved=moved,
        after_difference=flags,
        unmasked=unmasked,
        sample_seconds=sample_seconds,
        seconds=time.perf_counter() - started,
        peak_bytes=mx.get_peak_memory(),
    )


def metrics_record(result: StepResult) -> dict[str, object]:
    """Write one step as a line of the metrics log.

    Args:
        result: What the step did.

    Returns:
        A JSON-friendly record (the project's field names where they exist).
    """
    replies = result.replies
    ratios = [r for row in result.update.ratios for r in row]
    return {
        "stage": "grpo",
        "step": result.step,
        "tasks": sorted({reply.task_id for reply in replies}),
        "accuracy": sum(reply.correct for reply in replies) / len(replies),
        "mean_reward": statistics.fmean(reply.reward for reply in replies),
        "tokens_per_reply": statistics.fmean(reply.tokens for reply in replies),
        "groups_without_contrast": result.no_signal,
        "groups": len(result.outcomes),
        "train_rows": result.update.rows,
        "train_tokens": result.update.tokens,
        "importance_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "loss_per_token": result.update.loss_per_token,
        "entropy": report.token_mean(replies, "entropy"),
        "sample_seconds": round(result.sample_seconds, SECONDS_DECIMALS),
        "step_seconds": round(result.seconds, SECONDS_DECIMALS),
        "peak_gb": round(result.peak_bytes / GIGABYTE, GB_DECIMALS),
    }


def main() -> None:
    """Train with GRPO, print every step, save the adapter, show the project's runs.

    Raises:
        SystemExit: If the starting adapter or a bundled record is missing.
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    init = require_init(init_path(args.init))
    try:  # read the bundled records first: a damaged one stops before training
        project = read_run(ATTEMPT_1), read_run(ATTEMPT_2)
    except ProjectRunError as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error
    draw = Draw(TRAIN_TEMPERATURE, args.top_k, args.logprob_dtype)
    settings = Settings(draw, args.retokenize, args.kl, args.lr, args.float32)
    set_memory_limits()
    loaded = load_folder(folder)
    model, tokenizer = loaded.model, loaded.tokenizer
    if args.float32:
        to_float32(model)
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    load_adapter(model, init)
    reference = dict(adapter_weights(model)) if args.kl > 0 else None
    adapter = adapter_path(run_name(name_parts(args)))
    order = training_order(load_tasks(TRAIN_FILE), ORDER_SEED)
    report.print_title(SCRIPT, TITLE)
    report.print_grpo_setup(folder, which, init, args, settings, len(order))
    optimizer = optim.AdamW(learning_rate=args.lr, bias_correction=True)
    metrics, rollouts = metrics_path(adapter), rollouts_path(adapter)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("")  # this run's steps only
    rollouts.write_text("")
    mx.random.seed(args.seed)  # the same seed draws the same replies
    mx.reset_peak_memory()
    report.print_steps_header(settings)
    results: list[StepResult] = []
    for step in range(1, args.steps + 1):
        first = (step - 1) * QUESTIONS_PER_STEP
        tasks = [order[(first + i) % len(order)] for i in range(QUESTIONS_PER_STEP)]
        result = one_step(
            model, tokenizer, optimizer, tasks, step, settings, reference, adapter.stem
        )
        report.print_step(result, settings)
        with metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_record(result)) + "\n")
        with rollouts.open("a", encoding="utf-8") as handle:
            for reply in result.replies:
                handle.write(json.dumps({"step": step, **asdict(reply)}) + "\n")
        results.append(result)
    report.print_steps_footer(results, settings)
    report.print_first_step(results[0])
    report.print_ratio_summary(results, settings)
    report.print_groups(results)
    save_adapter(model, adapter)
    report.print_saved(adapter, adapter.stat().st_size, metrics, rollouts)
    report.print_project_runs(*project)
    mx.clear_cache()


if __name__ == "__main__":
    main()
