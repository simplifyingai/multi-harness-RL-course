"""Run chapter 27's one_step.py with no 2.6B model, for the tests.

Run with the PROJECT's Python, against ``27-how-it-combines/stack.sh
scripted`` (the project's answer-key teacher answering as the engine: it
loads only the tokenizer)::

    MHRL_ENGINE_PORT=... MHRL_PROXY_PORT=... \\
        <project>/.venv/bin/python tests/one_step_scripted_27.py \\
        [--contrast] -- [one_step.py arguments]

Two things are swapped before ``one_step.main`` runs, and nothing else:

* ``mhrl.training.grpo.load_policy`` builds a tiny LFM2 (mlx-lm's own
  ``lfm2`` code, 64 numbers wide, four layers) with the project's own
  ``apply_lora``, and the real LFM2.5 tokenizer, so every captured row
  trains for real on a model that fits anywhere;
* with ``--contrast``, ``mhrl.rollout.grade`` marks every second rollout
  wrong. The teacher writes the same replies in all four rollouts, so
  without this every group has four equal rewards and no contrast, which is
  what the run without ``--contrast`` tests.

The teacher's log-probabilities are all 0 and the tiny model's are not, so
the ratios here are far from 1: this checks the plumbing, not the numbers.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import lfm2
from mlx_lm.utils import load_tokenizer

CHAPTER = Path(__file__).resolve().parent.parent / "27-how-it-combines"
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
SCRIPTED_WRONG = "scripted: marked wrong so the group has contrast"


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


def every_second_wrong(real: Any) -> Any:
    """Wrap the grader so every second rollout is graded wrong.

    Args:
        real: ``mhrl.rollout.grade``.

    Returns:
        The wrapper.
    """
    from mhrl.tasks.core import Grade

    lock = threading.Lock()
    calls = [0]

    def grade(task: Any, workdir: Path) -> Any:
        verdict = real(task, workdir)
        with lock:
            calls[0] += 1
            wrong = calls[0] % 2 == 0
        return Grade(False, verdict.submitted, SCRIPTED_WRONG) if wrong else verdict

    return grade


def main() -> int:
    """Swap in the tiny model (and the contrast), then run one_step.

    Returns:
        one_step's exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0, str(CHAPTER))
    import one_step
    from mhrl import rollout
    from mhrl.training import grpo

    grpo.load_policy = tiny_policy
    if args.contrast:
        rollout.grade = every_second_wrong(rollout.grade)
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    return one_step.main(rest)


if __name__ == "__main__":
    sys.exit(main())
