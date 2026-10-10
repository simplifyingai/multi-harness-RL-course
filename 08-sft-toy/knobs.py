"""The chapter's fixed numbers, in one place that imports nothing else.

``sft_toy.py`` runs with them, and ``report.py`` and ``seeds.py`` print them,
so they live here rather than in either (which would make the two import
each other).
"""

from __future__ import annotations

# The project's SFT shape (mhrl/cli.py): 120 demonstrations, batch 8, 1 epoch.
DEMONSTRATIONS = 120
BATCH_SIZE = 8  # SFT_BATCH
EPOCHS = 1  # SFT_EPOCHS
DEMO_LEVEL = "easy"  # LOOK, QUERY_A, WRITE, FINISH answers only easy questions
DEFAULT_SEED = 0
DEFAULT_LR = 3.0  # chapters 5 to 7's learning rate
HIGH_LR_TIMES = 10  # --lr-high: ten times as big
HIGH_LR = HIGH_LR_TIMES * DEFAULT_LR
RL_LR = 3.0  # chapter 6's
DEFAULT_RL_STEPS = 100
SCRATCH = "scratch"  # RL's two starts: every weight 0...
IMITATOR = "imitator"  # ...or the weights SFT trained
