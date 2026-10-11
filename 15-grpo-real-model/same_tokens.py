# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""The same tokens, scored by the sampler and by the trainer: is the ratio 1?

grpo_sql.py's first reply, drawn again (the same question, seed and adapter),
and its exact tokens scored three more ways with the same weights:

- by the sampler's own route again: one token at a time, with a cache,
  each token fed in rather than drawn;
- in one pass over the whole row, as ``token_logprobs`` reads it;
- in one pass inside a gradient computation, the route the trainer's update
  takes (its numbers are what the importance ratio is made of).

Each is compared with the log-probabilities the sampler recorded. Then the
whole model is cast to float32 and the same tokens are scored the three ways
again. Run it with ``uv run same_tokens.py`` (a few minutes); ``--fast`` uses
Liquid's export (the same bytes).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import report
from grpo import token_logprobs, trainer_logprobs
from lfm import PREFILL_STEP, load_folder, to_float32
from lfm_files import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    ORDER_SEED,
    OUTPUT_CAP,
    SAMPLE_SEED,
    SFT_ADAPTER,
    TRAIN_TEMPERATURE,
    init_path,
    pick_model,
    require_init,
    short_path,
)
from lora import apply_lora, load_adapter
from mlx_lm.models.cache import make_prompt_cache
from rows import TrainItem, prompt_ids, training_order
from sampler import Draw, sample
from tasks import TRAIN_FILE, load_tasks

SCRIPT = "same_tokens.py"
TITLE = "the same tokens, scored by the sampler and by the trainer"
LARGEST_SHOWN = 5  # tokens listed whose ratio moved most
ROUTES = (
    "the sampler's route again (one token at a time, with a cache)",
    "one pass over the row",
    "one pass inside a gradient computation (the trainer's update)",
)


@dataclass(frozen=True)
class RouteResult:
    """One route's scores for the reply's tokens, against a reference.

    Attributes:
        name: The route.
        logprobs: Its log-probability for each token.
        ratios: exp(its log-probability - the reference's), per token.
    """

    name: str
    logprobs: list[float]
    ratios: list[float]


def stepwise_logprobs(
    model: nn.Module, prompt: list[int], tokens: list[int]
) -> list[float]:
    """Score tokens the way the sampler does: one at a time, with a cache.

    The prompt is read as mlx-lm's ``generate_step`` reads it (all but its
    last token in chunks, then the last one), and each token of the reply is
    fed in instead of drawn. The scores are cast to float32 before the
    softmax, as the sampler does.

    Args:
        model: The network.
        prompt: The prompt's token ids.
        tokens: The reply's token ids.

    Returns:
        Each reply token's log-probability.
    """
    cache = make_prompt_cache(model)
    rest = mx.array(prompt)
    while rest.size > 1:
        size = min(PREFILL_STEP, rest.size - 1)
        model(rest[:size][None], cache=cache)
        mx.eval([layer.state for layer in cache])
        rest = rest[size:]
    scores = model(rest[None], cache=cache)[0, -1].astype(mx.float32)
    found = []
    for position, token in enumerate(tokens):
        found.append((scores - mx.logsumexp(scores))[token].item())
        if position + 1 < len(tokens):  # nothing follows the last token
            scores = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
    return found


def routes(
    model: nn.Module, prompt: list[int], tokens: list[int], reference: list[float]
) -> list[RouteResult]:
    """Score the reply's tokens by each route and compare with a reference.

    Args:
        model: The network, with the adapter.
        prompt: The prompt's token ids.
        tokens: The reply's token ids.
        reference: The log-probabilities to compare with.

    Returns:
        One result per route, in ROUTES order.
    """
    ids = prompt + tokens
    item = TrainItem(ids, list(range(len(prompt), len(ids))), [0.0] * len(tokens))
    found = [
        stepwise_logprobs(model, prompt, tokens),
        token_logprobs(model, item).tolist(),
        trainer_logprobs(model, item).tolist(),
    ]
    return [
        RouteResult(
            name,
            values,
            [math.exp(v - r) for v, r in zip(values, reference, strict=True)],
        )
        for name, values in zip(ROUTES, found, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--init", default=SFT_ADAPTER, help="the adapter (chapter 14's)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="use Liquid's export, not ours"
    )
    return parser.parse_args()


def main() -> None:
    """Draw one reply, score its tokens by every route, in bfloat16 and float32.

    Raises:
        SystemExit: If no model or no start adapter is on disk (from
            ``pick_model`` and ``require_init``, naming the command that makes
            it).
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    init = require_init(init_path(args.init))
    loaded = load_folder(folder)
    model, tokenizer = loaded.model, loaded.tokenizer
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    load_adapter(model, init)
    task = training_order(load_tasks(TRAIN_FILE), ORDER_SEED)[0]
    prompt = prompt_ids(tokenizer, task)
    mx.random.seed(SAMPLE_SEED)
    reply = sample(
        model, tokenizer.eos_token_ids, prompt, Draw(TRAIN_TEMPERATURE), OUTPUT_CAP
    )
    report.print_title(SCRIPT, TITLE)
    report.setup_line("model", f"{short_path(folder)}, {which}")
    report.setup_line("adapter", f"{short_path(init)}, chapter 14's")
    report.print_reply(task, reply, tokenizer.decode)
    bf16 = routes(model, prompt, reply.tokens, reply.logprobs)
    report.print_routes(
        "bfloat16, as the model runs: each route against what the sampler recorded",
        bf16,
    )
    report.print_moved(bf16[-1], reply, tokenizer.decode, LARGEST_SHOWN)
    to_float32(model)
    reference = stepwise_logprobs(model, prompt, reply.tokens)
    wide = routes(model, prompt, reply.tokens, reference)
    report.print_routes(
        "float32 (every 16-bit number cast): each route against the sampler's",
        wide,
    )
    report.print_float32_shift(reference, reply.logprobs)


if __name__ == "__main__":
    main()
