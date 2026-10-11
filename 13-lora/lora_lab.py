# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Freeze the 4-bit LFM2.5-2.6B and train a small LoRA add-on instead.

As the project's ``mhrl/lora.py`` does, this script freezes the model and
adds rank-16 LoRA adapters (scale 2.0, no dropout, seed 0) with mlx-lm's
``linear_to_lora_layers`` to every projection of the last 12 of its 30
layers: attention q/k/v/out, convolution in/out and feed-forward w1/w2/w3.
It lists every adapter pair, counts what trains, checks that a fresh
adapter changes nothing, runs 20 AdamW steps raising the log-probability of
one question's SQL answer (the project's SFT settings), saves the adapter
every 5 steps, reloads and zeroes it, and finally merges it into the frozen
weights in 16 bits and in 4.

Run it with ``uv run lora_lab.py`` (under a minute). ``--fast`` uses
Liquid's export, takes 5 steps and skips the merge. Adapters are saved under
``$COURSE_HOME/13-lora``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import report
from arithmetic import LoraShape, by_key, by_layer, checkpoint_steps
from lfm import Reply, chat_prompt, greedy_reply, load_folder
from lfm_files import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    SFT_LEARNING_RATE,
    checkpoint_path,
    clear_adapters,
    final_adapter_path,
    pick_model,
    read_safetensors_header,
)
from lora import (
    LORA_DROPOUT,
    LORA_KEYS,
    LORA_SEED,
    a_in_range,
    adapter_pairs,
    adapter_weights,
    apply_lora,
    change_and_step,
    layer_kinds,
    load_adapter,
    lora_modules,
    merged,
    module_bytes,
    save_adapter,
    zero_adapter,
    zero_b_count,
)
from mlx.utils import tree_unflatten
from mlx_lm.tokenizer_utils import TokenizerWrapper

SCRIPT = "lora_lab.py"
TITLE = "train a 0.4% add-on, not the whole model"
# One of the project's 180 training tasks (data/tasks/train.jsonl), asked the
# way chapters 10 to 12 asked their question.
TASK_ID = "shop-medium-001"
QUESTION = (
    "Which city had the most new customers sign up in 2024? Answer with one SQL query."
)
GOLD_SQL = (
    "SELECT city FROM customers WHERE strftime('%Y', signup_date) = '2024' "
    "GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1"
)
# The reply to learn: a one-line reason, the end of the thinking, the SQL.
REASON = "Group the customers who signed up in 2024 by city and take the biggest group."
THINK_END = "</think>"
TARGET_TEXT = f"{REASON}{THINK_END}{GOLD_SQL}"
STEPS = 20
FAST_STEPS = 5
SAVE_EVERY = 5  # steps between saved adapters
OUTPUT_CAP = 1_024  # the project's output cap, for the reply before training
FAST_REPLY_CAP = 64  # --fast keeps every reply short
MERGED_REPLY_CAP = 64  # enough for the 50-token target, and the start of a ramble


@dataclass(frozen=True)
class Example:
    """The one training example: a prompt and the reply to learn, as token ids.

    Attributes:
        prompt: The chat-template prompt, ending with ``<think>``.
        target: The reply to learn, ending with ``<|im_end|>``.
    """

    prompt: list[int]
    target: list[int]

    def ids(self) -> mx.array:
        """Put the prompt and the reply in one row, as the model reads them.

        Returns:
            The token ids, shape (1, length).
        """
        return mx.array(self.prompt + self.target)[None]


@dataclass(frozen=True)
class Scored:
    """The target reply, scored under the model's current weights.

    Attributes:
        step: The training step it was scored after (0: before training).
        total: The target's log-probability: the sum of its tokens'.
        ranked_first: Target tokens that are the model's likeliest next token,
            given the target's tokens before them.
        tokens: How many tokens the target has.
    """

    step: int
    total: float
    ranked_first: int
    tokens: int


@dataclass(frozen=True)
class Training:
    """What the training run produced.

    Attributes:
        scored: The target scored before training and after every step.
        saved: The adapter files written, by step.
        seconds: How long the optimizer steps took, without the scoring and
            saving between them (varies from run to run).
        peak_bytes: MLX's peak memory during training.
        state_bytes: The size of AdamW's running averages.
    """

    scored: list[Scored]
    saved: dict[int, Path]
    seconds: float
    peak_bytes: int
    state_bytes: int


@dataclass(frozen=True)
class Merged:
    """The model after merging its adapters into the frozen weights.

    Attributes:
        scored: The target's score.
        reply: The greedy reply.
        size: The bytes the 66 merged matrices take.
    """

    scored: Scored
    reply: Reply
    size: int


def target_ids(tokenizer: TokenizerWrapper) -> list[int]:
    """Turn the reply to learn into tokens, ending with the end-of-turn token.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        The reply's token ids, ``<|im_end|>`` last, as the model writes it.
    """
    text = list(tokenizer.encode(TARGET_TEXT, add_special_tokens=False))
    return text + [tokenizer.eos_token_id]


def all_logits(model: nn.Module, example: Example) -> mx.array:
    """Score every token after every position of the example, in float32.

    Args:
        model: The network.
        example: The prompt and the target.

    Returns:
        The scores, shape (length, vocabulary).
    """
    logits = model(example.ids())[0].astype(mx.float32)
    mx.eval(logits)
    return logits


def type_names(arrays: list[mx.array]) -> str:
    """Name the number types some arrays use.

    Args:
        arrays: The arrays.

    Returns:
        Such as ``"bfloat16"`` or ``"bfloat16, float32"``.
    """
    return ", ".join(sorted({str(array.dtype).rpartition(".")[2] for array in arrays}))


def max_difference(one: mx.array, other: mx.array) -> float:
    """Find the largest difference between two sets of scores.

    Args:
        one: Scores.
        other: Scores of the same shape.

    Returns:
        The largest absolute difference.
    """
    return mx.max(mx.abs(one - other)).item()


def target_logprobs(model: nn.Module, example: Example) -> tuple[mx.array, mx.array]:
    """Score the target's tokens, each from the position before it.

    The scores at position i predict token i + 1, so the target's tokens are
    read from the positions just before them, as the project's
    ``token_logprobs`` does (mhrl/training/common.py).

    Args:
        model: The network.
        example: The prompt and the target.

    Returns:
        Each target token's log-probability, and whether it was ranked first.
    """
    start = len(example.prompt) - 1  # the prompt's last position predicts token 1
    logits = model(example.ids())[0, start:-1].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    wanted = mx.array(example.target)
    picked = mx.take_along_axis(logprobs, wanted[:, None], axis=-1)[:, 0]
    return picked, mx.argmax(logits, axis=-1) == wanted


def score(model: nn.Module, example: Example, step: int) -> Scored:
    """Score the target under the current weights.

    Args:
        model: The network.
        example: The prompt and the target.
        step: The step just taken (0 before training).

    Returns:
        The score.
    """
    picked, first = target_logprobs(model, example)
    total, hits = mx.sum(picked).item(), int(mx.sum(first).item())
    return Scored(step, total, hits, len(example.target))


def optimizer_arrays(state: object) -> list[mx.array]:
    """List every array in an optimizer's state, however it is nested.

    Args:
        state: The state: dictionaries and lists of arrays.

    Returns:
        The arrays.
    """
    if isinstance(state, mx.array):
        return [state]
    values = state.values() if isinstance(state, dict) else state
    found = []
    for value in values:
        if isinstance(value, dict | list | mx.array):
            found += optimizer_arrays(value)
    return found


def train(model: nn.Module, example: Example, steps: int, lr: float) -> Training:
    """Raise the target's log-probability with AdamW, saving the adapter as it goes.

    The loss is minus the average log-probability of the target's tokens (the
    project's SFT sums them and divides by the reply's token count, the same
    thing for one reply). Only the adapters are trainable, so only they get
    gradients and only they change.

    Args:
        model: A model with LoRA layers applied.
        example: The prompt and the target.
        steps: How many optimizer steps to take.
        lr: AdamW's learning rate.

    Returns:
        What the run produced.
    """
    # As mhrl/training/sft.py builds it: weight decay is left at MLX's 0.01.
    optimizer = optim.AdamW(learning_rate=lr, bias_correction=True)

    def loss_fn(chosen: Example) -> mx.array:
        """Minus the target tokens' average log-probability.

        Args:
            chosen: The example to learn.

        Returns:
            The loss, one number.
        """
        picked, _ = target_logprobs(model, chosen)
        return -mx.mean(picked)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    save_after = set(checkpoint_steps(steps, SAVE_EVERY))
    scored, saved, seconds = [score(model, example, 0)], {}, 0.0
    mx.reset_peak_memory()
    for step in range(1, steps + 1):
        start = time.perf_counter()  # time the step itself, not the checks below
        _, grads = loss_and_grad(example)  # gradients for the adapters only
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        seconds += time.perf_counter() - start
        scored.append(score(model, example, step))
        if step in save_after:
            saved[step] = save_adapter(model, checkpoint_path(step))
    return Training(
        scored=scored,
        saved=saved,
        seconds=seconds,
        peak_bytes=mx.get_peak_memory(),
        state_bytes=sum(array.nbytes for array in optimizer_arrays(optimizer.state)),
    )


def merge_and_score(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    example: Example,
    dequantize: bool,
    step: int,
) -> Merged:
    """Merge the adapters into the frozen weights, score, then put them back.

    Args:
        model: A model with LoRA layers applied (and trained).
        tokenizer: The tokenizer, for the reply's end.
        example: The prompt and the target.
        dequantize: Merge into 16-bit matrices (True) or back into 4 bits.
        step: The training step the adapter was saved after.

    Returns:
        The model's score, reply and merged size, before the adapters went back.
    """
    adapters = lora_modules(model)
    fused = merged(adapters, dequantize)
    model.update_modules(tree_unflatten(fused))
    try:
        mx.eval(model.parameters())
        return Merged(
            scored=score(model, example, step),
            reply=greedy_reply(model, tokenizer, example.prompt, MERGED_REPLY_CAP),
            size=module_bytes(fused),
        )
    finally:  # the adapters back in place, even if scoring failed
        model.update_modules(tree_unflatten(adapters))


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=None, help="AdamW steps")
    parser.add_argument("--lr", type=float, default=SFT_LEARNING_RATE)
    parser.add_argument("--scale", type=float, default=LORA_SCALE, help="LoRA scale")
    parser.add_argument(
        "--fast", action="store_true", help="Liquid's export, 5 steps, no merge"
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = FAST_STEPS if args.fast else STEPS
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.lr <= 0 or args.scale <= 0:
        parser.error("--lr and --scale must be above 0")
    return args


def add_adapters(model: nn.Module, shape: LoraShape, parameters: int) -> int:
    """Freeze the model, add the adapters, and show every pair and the count.

    Args:
        model: The loaded model.
        shape: How the adapters are made.
        parameters: The model's numbers before LoRA.

    Returns:
        How many numbers are trainable.
    """
    apply_lora(model, shape.rank, shape.scale, shape.layers)
    pairs = adapter_pairs(model)
    layers = by_layer(pairs, layer_kinds(model))
    report.print_adapters(layers, by_key(pairs, LORA_KEYS), len(model.layers))
    trainable = sum(array.size for array in adapter_weights(model).values())
    report.print_trainable(layers, pairs, trainable, parameters)
    return trainable


def check_reload(
    model: nn.Module,
    example: Example,
    training: Training,
    base_logits: mx.array,
    trained_logits: mx.array,
) -> None:
    """Zero the adapter, reload the last one, then reload an earlier one.

    Args:
        model: A model with LoRA layers applied (and trained).
        example: The prompt and the target.
        training: The run, with its saved files.
        base_logits: The scores before LoRA was added.
        trained_logits: The scores right after training.
    """
    final = final_adapter_path()
    zero_adapter(model)
    zeroed = max_difference(all_logits(model, example), base_logits)
    load_adapter(model, final)
    reloaded = max_difference(all_logits(model, example), trained_logits)
    last = max(training.saved)
    earlier = [step for step in training.saved if step != last]
    middle = None
    if earlier:
        step = earlier[len(earlier) // 2]
        load_adapter(model, training.saved[step])
        middle = (training.saved[step], score(model, example, step))
    report.print_reload(zeroed, final, reloaded, middle, training.scored)
    load_adapter(model, final)  # leave the trained adapter in place


def merge_both_ways(
    model: nn.Module, tokenizer: TokenizerWrapper, example: Example, training: Training
) -> None:
    """Merge the trained adapter into the frozen weights in 16 bits and in 4.

    Args:
        model: A model with the trained adapter loaded.
        tokenizer: The tokenizer.
        example: The prompt and the target.
        training: The run, for the adapter's own score.
    """
    adapters = lora_modules(model)
    change = change_and_step(adapters)
    four_bit = module_bytes([(name, module.linear) for name, module in adapters])
    last = training.scored[-1].step
    sixteen = merge_and_score(model, tokenizer, example, dequantize=True, step=last)
    four = merge_and_score(model, tokenizer, example, dequantize=False, step=last)
    report.print_merge(
        training.scored[-1],
        sixteen,
        four,
        four_bit,
        change,
        example.target,
        tokenizer.decode,
    )


def main() -> None:
    """Freeze the model, add the adapters, train them, save, reload and merge."""
    args = parse_args()
    cap = FAST_REPLY_CAP if args.fast else OUTPUT_CAP
    folder, which = pick_model(args.fast)
    loaded = load_folder(folder)
    model, tokenizer = loaded.model, loaded.tokenizer
    example = Example(chat_prompt(tokenizer, QUESTION), target_ids(tokenizer))
    report.print_title(SCRIPT, TITLE)
    shape = LoraShape(LORA_RANK, args.scale, LORA_LAYERS)
    # apply_lora reads the dropout and the seed from lora.py, as the project's does
    report.print_setup(
        folder, which, shape, LORA_DROPOUT, LORA_SEED, args.lr, args.steps
    )
    as_loaded = list(adapter_weights(model).values())  # what MLX lets train
    report.print_model(
        loaded.parameters,
        loaded.weight_bytes,
        sum(array.size for array in as_loaded),
        type_names(as_loaded),
    )
    base_logits = all_logits(model, example)

    trainable = add_adapters(model, shape, loaded.parameters)
    fresh = max_difference(all_logits(model, example), base_logits)
    report.print_fresh(fresh, zero_b_count(model), a_in_range(model), base_logits.shape)

    report.print_task(TASK_ID, QUESTION, example, tokenizer.decode)
    before = greedy_reply(model, tokenizer, example.prompt, cap)
    report.print_reply("before training", before, example.target, tokenizer.decode)
    # Delete an earlier run's adapter files just before this run writes its own,
    # so the folder never mixes two runs.
    removed = clear_adapters()
    training = train(model, example, args.steps, args.lr)
    trained_logits = all_logits(model, example)
    report.print_steps(training.scored)
    after = greedy_reply(model, tokenizer, example.prompt, cap)
    title = f"after {args.steps} steps"
    report.print_reply(title, after, example.target, tokenizer.decode)
    report.print_training_cost(training)

    files = [*training.saved.values(), save_adapter(model, final_adapter_path())]
    headers = {path: read_safetensors_header(path) for path in files}
    report.print_checkpoints(headers, removed, trainable)
    check_reload(model, example, training, base_logits, trained_logits)
    if args.fast:
        report.print_merge_skipped()
        return
    merge_both_ways(model, tokenizer, example, training)


if __name__ == "__main__":
    main()
