"""Stand in for ``python -m mhrl.cli`` in chapter 28's tests: no 2.6B model.

It has no PEP 723 header on purpose: it needs the project's own packages,
so it runs with the PROJECT's Python, as ``capstone.sh``'s ``MHRL_CLI``::

    MHRL_CLI="<project>/.venv/bin/python tests/capstone_cli_scripted_28.py" \\
        STACK=scripted ./capstone.sh pipeline

``capstone.sh start`` with ``STACK=scripted`` puts the project's answer-key
teacher on the engine port (it loads only the tokenizer), so every rollout
is a real harness in the project's sandbox, captured by the real proxy and
graded by the project's grader. One thing is swapped before the project's
own ``mhrl.cli.main`` runs: both trainers' ``load_policy`` builds a tiny LFM2
(mlx-lm's own ``lfm2`` code, 64 numbers wide, four layers) with the
project's ``apply_lora`` and the real LFM2.5 tokenizer, so SFT and GRPO
train for real on a model that fits anywhere. Everything else, every
argument included, is the project's CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import lfm2
from mlx_lm.utils import load_tokenizer

VOCAB = 128_000  # LFM2.5's embedding rows: every captured token id fits
HIDDEN = 64
TINY_CONFIG: dict[str, Any] = {
    "model_type": "lfm2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 16_384,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": HIDDEN,
    "block_multiple_of": 64,
    "block_ff_dim": 2 * HIDDEN,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "rope_theta": 1e7,
    "layer_types": ["conv", "full_attention", "conv", "full_attention"],
}


def tiny_policy(settings: Any) -> tuple[nn.Module, Any]:
    """Stand in for load_policy: a tiny LFM2 with the project's adapter.

    Args:
        settings: The project's settings (the adapter's shape, the tokenizer).

    Returns:
        The model (only its adapter trainable) and LFM2.5's tokenizer.
    """
    from mhrl.lora import apply_lora

    mx.random.seed(0)
    model = lfm2.Model(lfm2.ModelArgs(**TINY_CONFIG))
    weights = [(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())]
    model.update(tree_unflatten(weights))
    apply_lora(model, settings.lora_rank, settings.lora_scale, settings.lora_layers)
    mx.eval(model.parameters())
    return model, load_tokenizer(Path(settings.model_path))


def main() -> int:
    """Swap in the tiny model, then run the project's CLI unchanged.

    Returns:
        0 (the project's CLI raises on any failure).
    """
    from mhrl import cli
    from mhrl.training import grpo, sft

    sft.load_policy = tiny_policy
    grpo.load_policy = tiny_policy
    sys.argv = ["mhrl.cli", *sys.argv[1:]]
    cli.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
