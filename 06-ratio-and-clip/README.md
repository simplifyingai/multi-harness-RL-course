# Chapter 6. GRPO, part 2: the importance ratio and the clip

In chapter 5, GRPO sampled 8 rollouts with the weights it had, learned from
them once, and threw them away. Every rollout came from exactly the weights
that then learned from it. Training like that is called **on-policy**: the
policy learns only from attempts made by the policy as it is now.

Real trainers often break that, in one of two ways. Some learn from the same
rollouts several times, to get more out of each one. Some keep sampling while
the learner works, so a rollout arrives a few updates late. Either way the
learner is now looking at rollouts its current weights did not make. That is
**off-policy**, and the probabilities the learner works out for those actions
are no longer the ones that picked them.

GRPO handles this the way its parent algorithm does: PPO (Proximal Policy
Optimization), which chapter 5 met as the algorithm with a critic. It weighs
each call by how much more or less likely its action has become since it was
sampled, and it clips that weight to between 0.8 and 1.2. This chapter puts both on
chapter 5's tiny agent. Then it asks why the project's own laptop runs measured
a weight of 1.000, and what that means.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/06-ratio-and-clip
uv run grpo_clip.py --stale 2 --trace
```

A run takes about 15 seconds on any Mac (4 with `--fast`, which skips the
pauses). It needs NumPy, which `uv run` installs for this script alone, and
like every chapter it writes only under `COURSE_HOME` (default
`~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `grpo_clip.py --token 0.20,0.30` | one call's ratio and clip, worked out by hand |
| `grpo_clip.py --stale 2 --trace` | rollouts sampled with weights two updates old: the ratio moves and the clip acts |
| `grpo_clip.py` | the default: a fresh batch and one update per step, so the ratio is exactly 1 |
| `grpo_clip.py --stale 2 --recompute-old` | the bug that makes a stale run read 1.000 |
| `grpo_clip.py --updates-per-batch 8` | each batch used for 8 updates, as PPO does |
| `grpo_clip.py --updates-per-batch 8 --lr 0.5` | the same with smaller updates, to watch the ratios spread use by use |
| `grpo_clip.py --updates-per-batch 8 --no-clip` | the same reuse with the clip taken away |
| `seeds.py` | every claim checked over seeds 0 to 9 (about a minute) |
| `plot.py <metrics.jsonl>` | a run's curves, redrawn from its log alone |
| `real_runs.py` | the ratios the project's two real RL runs measured |

Other flags: `--updates-per-batch N` (1 by default; 4 is worth a try too),
`--stale N`, `--seed N`, `--steps N`, `--lr N` and `--fast`. `seeds.py` takes
`--seeds N`, `--steps N` and `--lr N`.

## What is new since chapter 5

The agent is chapter 5's, copied in unchanged: the same six questions at three
difficulties, the same `LOOK` that shows one table at a time, the same policy
with one set of weights per level, the same learning rate of 3, and the same
steps of 2 questions x 4 rollouts. Three things are new.

- **Every call keeps its old probability.** When a rollout picks an action, it
  writes down the probability that action had at that moment, with the weights
  that picked it. This is the **old p**.
- **The learner works out a new probability.** When it learns from the call,
  it works out the same action's probability with the weights it has now. This
  is the **new p**.
- **The loss weighs each call by new p / old p, and clips that.** Below.

Three flags make the sampler and the learner differ, and one takes the clip
away. Every run starts by saying which:

```text
grpo_clip · seed 0 · lr 3 · 200 steps · 1 update per batch
each step: 2 questions x 4 rollouts at temperature 1, sampled with the learner's weights
questions: 45 to train on, 19 held out (chapter 5's, one set of weights per level)
loss per call = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), ratio = new p / old p
old p: recorded when the rollout was sampled
```

## One call, worked out by hand

```bash
uv run grpo_clip.py --token 0.20,0.30
```

```text
one call: its action had p 0.20 when the rollout was sampled (old)
and has p 0.30 with the learner's weights now (new)
ratio = new p / old p = 0.30 / 0.20 = 1.50
      = exp(new log p - old log p) = exp(-1.20 - (-1.61)) = 1.50
clip(ratio, 0.8, 1.2) = 1.20

A = +1 (the rollout did better than its group)
  ratio x A         = +1.50
  clip(ratio) x A   = +1.20
  min of the two    = +1.20  the clipped one
  loss = -min       = -1.20
  push on log p      0.00  the clip has switched it off

A = -1 (the rollout did worse than its group)
  ratio x A         = -1.50
  clip(ratio) x A   = -1.20
  min of the two    = -1.50  the unclipped one
  loss = -min       = +1.50
  push on log p     -1.50  ratio x A: 1.50 times chapter 5's push, which was A

min() keeps the more pessimistic number: a call that has already moved
20% the way its advantage wants gets no more push; a call that moved the
wrong way always gets its whole push, ratio x A
```

The **importance ratio** is new p / old p: how many times more likely the
action is now than when it was sampled. Here it is 0.30 / 0.20 = 1.5. The code
works it out from log-probabilities, exp(new log p - old log p), which is the
same number (chapter 3: subtracting logs divides the probabilities).

The **clip** limits the ratio to between 0.8 and 1.2, so 1.5 becomes 1.2. The
loss for one call is then `-min(ratio x A, clip(ratio, 0.8, 1.2) x A)`, where A
is the call's advantage, from its rollout's group (chapter 5). `min` keeps the
smaller of the two numbers. What happens depends on the sign of A.

- **A = +1, the rollout did better than its group.** The step wants this action
  more likely, and it already is: 1.5 times. The two numbers are 1.50 and 1.20,
  and `min` keeps 1.20, the clipped one. The clipped number is 1.2 x A whatever
  the weights do, so it has no slope, and the call's push is 0. The clip has
  switched it off: the action has moved far enough for one batch.
- **A = -1, the rollout did worse.** The step wants this action less likely,
  but it has become more likely. The two numbers are -1.50 and -1.20, and `min`
  keeps -1.50, the unclipped one. The push stays on, at 1.5 times chapter 5's.

The **push on log p** is how hard the step raises (+) or lowers (-) the
log-probability of a call's action. In chapters 3 to 5 it was the advantage A.
With the ratio it is ratio x A, unless the clip has switched the call off.

Why `min`? The step tries to make `ratio x A` large, and `ratio x A` is a guess
at how much better the new weights do than the old ones, based on rollouts the
old weights made. The guess is only good near ratio 1. So the loss keeps
whichever of the two guesses is worse for the new weights: it stops giving
credit once an action has moved 20% the way its advantage wants, and it never
stops the blame when an action moved the wrong way. That is why `min` is called
the **pessimistic** choice. Being outside 0.8 to 1.2 is not the same as being
clipped: here the A = -1 call has ratio 1.5, outside the range, and is not
clipped. Try `--token 0.30,0.20`: the ratio is 0.67, and the two signs swap
roles.

## The code: the project's loss, line for line

The loss is `mhrl/training/grpo.py`'s `loss_fn`, line for line (in `grpo.py`,
docstring left out here; a test checks it against the project's):

```python
def loss_fn(weights: np.ndarray, item: Trained) -> tuple[float, np.ndarray]:
    eps = Defaults.CLIP_EPSILON
    new = call_logprobs(weights, item)
    ratio = np.exp(new - np.array(item.old_logprobs))
    unclipped = ratio * item.advantage
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
    return -np.minimum(unclipped, clipped).sum(), ratio
```

`eps` is 0.2, the project's `CLIP_EPSILON`. Three things differ from the
project's five lines, and only in name: `np` (NumPy) stands where the project
has `mx` (MLX); the toy's `call_logprobs`, one action per call, stands for the
real model's `token_logprobs`, many tokens per call; and NumPy has no gradient
to stop, so the ratio is returned as it is. The step averages the loss over
every call it learns from, as the project averages over every token.

NumPy cannot work out slopes by itself, so `loss_gradient` does it by hand:
each call's push, ratio x A or 0 when clipped, times chapter 4's slope of
log p. A test checks it against slopes measured by moving each weight a hair.

## Rollouts from older weights

```bash
uv run grpo_clip.py --stale 2 --trace
```

The article's trainer runs **asynchronously**: one GPU keeps sampling rollouts
while another learns, so neither waits for the other. The price is that a
rollout comes from weights a few updates old (the article's numbers are in a
box at the end). A rollout made with older weights is **stale**. `--stale 2`
does that to the toy: each step's rollouts are sampled with the weights from
two updates before the learner's. A step whose groups all tie makes no update,
as in the project, so "two updates" can span many steps; in the first steps
there are fewer than two earlier updates, so the sampler is less stale.

```text
grpo_clip · seed 0 · lr 3 · 200 steps · 1 update per batch
each step: 2 questions x 4 rollouts at temperature 1, sampled with the weights from 2 updates earlier
questions: 45 to train on, 19 held out (chapter 5's, one set of weights per level)
loss per call = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), ratio = new p / old p
old p: recorded when the rollout was sampled

step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   2 · reward 0.00 · correct 0% · calls 5.2 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   3 · reward 0.00 · correct 0% · calls 4.2 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   4 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   5 · reward 0.00 · correct 0% · calls 4.4 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   6 · reward 0.00 · correct 0% · calls 4.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   7 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   8 · reward 0.00 · correct 0% · calls 3.9 · no-signal groups 2/2 · ratio n/a · clipped n/a
the first call the clip switched off: step 9, use 1 of its batch
  LOOK · old p 0.167 · new p 0.443 · ratio 2.66 · A +1.73
  min(2.66 x +1.73, 1.20 x +1.73) = +2.08, the clipped term: push 0

step   9 · reward 0.12 · correct 12% · calls 4.1 · no-signal groups 1/2 · ratio 0.901 · clipped 50%
step  10 · reward 0.00 · correct 0% · calls 5.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  11 · reward 0.12 · correct 12% · calls 5.5 · no-signal groups 1/2 · ratio 1.157 · clipped 57%
step  12 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  13 · reward 0.00 · correct 0% · calls 5.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  14 · reward 0.00 · correct 0% · calls 3.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
```

Steps 2 to 8 learned nothing (every group tied, chapter 5's no-signal groups),
so they have no call to weigh and print `ratio n/a`, as the project's trainer
does. Step 9 learns from rollouts sampled with the weights from before step 1's
update. `--trace` prints the first call the clip switched off. Step 1 made
`LOOK` more likely, from p 0.167 to 0.443, so its ratio is 2.66. Its rollout
did better than its group (A +1.73), so the action has already moved far more
than 20% the way the advantage wants, and the clip switches its push off. It is
the worked call from the section above, in a real run. Half of that step's
calls were clipped.

At the end:

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    4,122  1.000  22%       10%       11%       23%       11%       8%        16%       20%
mean ratio per step, over the 115 steps that learned: 0.858 to 1.162

rollouts right, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
easy    11%         13%       45%       80%
medium  4%          11%       35%       72%
hard    0%          0%        0%        0%

greedy test at the end (always the most likely action)
  easy    training  0 of 15  held out 0 of 8 · QUERY_A x8
  medium  training 16 of 16  held out 7 of 7 · LOOK x4, QUERY_A, WRITE x3
  hard    training  0 of 14  held out 0 of 4 · LOOK x8
```

The first table splits every ratio the loss saw into bins. 38% of them are
outside 0.8 to 1.2 (22% below, 16% above), and the clip switched off 20% of the
pushes. 23% are within 1% of 1, most of them exactly 1: with one set of
weights per level, an update on easy questions leaves the medium weights
alone, so a medium call can keep exactly its old probability.

Look at the `mean` column: 1.000, although more than a third of the ratios are
far from 1. That is no accident. The sampler picks each action as often as its
old p says, so on average over its picks the ratio is old p x (new p / old p)
added up over the actions: the new probabilities added up, which is 1. Ratios
above 1 and below 1 cancel. So a mean near 1 says little on its own. The next
line says more: the mean of a single step wandered from 0.858 to 1.162.

Learning was slow: in steps 1 to 50 only 11% of easy rollouts were right,
against 45% with fresh rollouts below. The greedy test shows it too: on easy
questions the single most likely action at every call is still `QUERY_A`,
which fails until the agent has looked, although the sampled rollouts were
right 80% of the time by the end. `seeds.py` measures this slowdown over ten
seeds, and finds what causes it.

## The default: a fresh batch, one update

```bash
uv run grpo_clip.py
```

```text
step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   2 · reward 0.00 · correct 0% · calls 6.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   3 · reward 0.00 · correct 0% · calls 4.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   4 · reward 0.12 · correct 12% · calls 6.2 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   5 · reward 0.00 · correct 0% · calls 3.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   6 · reward 0.00 · correct 0% · calls 4.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   7 · reward 0.12 · correct 12% · calls 6.8 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   8 · reward 0.12 · correct 12% · calls 5.4 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   9 · reward 0.25 · correct 25% · calls 6.5 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  10 · reward 0.00 · correct 0% · calls 4.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  11 · reward 0.00 · correct 0% · calls 6.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  12 · reward 0.00 · correct 0% · calls 3.4 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  13 · reward 0.00 · correct 0% · calls 5.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  14 · reward 0.00 · correct 0% · calls 4.4 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  15 · reward 0.25 · correct 25% · calls 6.2 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  16 · reward 0.25 · correct 25% · calls 6.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  17 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  18 · reward 0.25 · correct 25% · calls 6.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  19 · reward 0.00 · correct 0% · calls 2.9 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  20 · reward 0.12 · correct 12% · calls 5.4 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  21 · reward 0.12 · correct 12% · calls 4.6 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
```

Every step that learns prints `ratio 1.000 · clipped 0%`: here 99 of the 200
steps. The other 101 had no group with contrast, so there was nothing to weigh
and they print `ratio n/a`. The reward, correct, calls and no-signal columns
are chapter 5's run, number for number: the same rollouts, the same updates,
the same final weights (a test checks all three).

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    3,573  1.000  0%        0%        0%        100%      0%        0%        0%        0%
mean ratio per step, over the 99 steps that learned: 1.000 to 1.000

rollouts right, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
easy    45%         88%       95%       95%
medium  11%         50%       82%       95%
hard    0%          0%        0%        0%

greedy test at the end (always the most likely action)
  easy    training 15 of 15  held out 8 of 8 · LOOK, QUERY_A, WRITE x6
  medium  training 16 of 16  held out 7 of 7 · LOOK x3, QUERY_A, WRITE, LOOK x3
  hard    training  0 of 14  held out 0 of 4 · LOOK x8
```

Every one of 3,573 ratios is in the middle bin, and every step's mean is
1.000. The table below it is chapter 5's.

The ratio is not near 1, it is exactly 1. The sampler and the learner are the
same weights at the same moment, and the same code works out the same log p
from the same numbers, so new log p - old log p is exactly 0, and exp(0) is
exactly 1. A ratio of 1 is always inside 0.8 to 1.2, so the clip never acts.
Each call's push is 1 x A = A, which is chapter 3's policy gradient with
chapter 5's group baseline. **With one fresh batch and one update per step,
GRPO is REINFORCE with a group baseline.** The loss's value is different
(-A per call, where chapter 5's was -A x log p), but its slope is the same, and
only the slope moves the weights.

## The bug that reads 1.000

```bash
uv run grpo_clip.py --stale 2 --recompute-old
```

Some trainers do not keep the probabilities the sampler used. When a batch
arrives they work out the "old" log p themselves, with the learner's weights.
If the learner and the sampler are the same weights, nothing is lost. If the
sampler is behind, the probabilities it really used are gone, and old p equals
new p for every call.

```text
grpo_clip · seed 0 · lr 3 · 200 steps · 1 update per batch
each step: 2 questions x 4 rollouts at temperature 1, sampled with the weights from 2 updates earlier
questions: 45 to train on, 19 held out (chapter 5's, one set of weights per level)
loss per call = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), ratio = new p / old p
old p: recomputed with the learner's weights when the batch arrives

step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step   2 · reward 0.00 · correct 0% · calls 5.2 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   3 · reward 0.00 · correct 0% · calls 4.2 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   4 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   5 · reward 0.00 · correct 0% · calls 4.4 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   6 · reward 0.00 · correct 0% · calls 4.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   7 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   8 · reward 0.00 · correct 0% · calls 3.9 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   9 · reward 0.12 · correct 12% · calls 4.1 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  10 · reward 0.00 · correct 0% · calls 5.6 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  11 · reward 0.12 · correct 12% · calls 5.5 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
step  12 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  13 · reward 0.00 · correct 0% · calls 5.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  14 · reward 0.00 · correct 0% · calls 3.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  15 · reward 0.00 · correct 0% · calls 5.5 · no-signal groups 2/2 · ratio n/a · clipped n/a
step  16 · reward 0.12 · correct 12% · calls 6.2 · no-signal groups 1/2 · ratio 1.000 · clipped 0%
```

These are the same rollouts as the `--stale 2` run above, step for step, up to
step 16: the same weights sampled them. Step 9, the same batch where `--stale
2` printed `ratio 0.901 · clipped 50%`, now prints `ratio 1.000 · clipped 0%`.
So do steps 11 and 16. The sampler is exactly as far behind as before; the
ratio just can no longer see it.

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    3,725  1.000  0%        0%        0%        100%      0%        0%        0%        0%

what --recompute-old hid: the same calls' ratios against the p recorded at sampling
(clipped: the calls the clip would have switched off)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    3,725  1.001  23%       11%       12%       18%       11%       7%        18%       23%
mean ratio per step, over the 104 steps that learned: 1.000 to 1.000
```

The second table is what the run kept quiet about: the same calls' ratios
against the probabilities recorded when they were sampled. 41% of them were
outside 0.8 to 1.2, and the clip would have switched off 23% of the pushes.
With the bug, the loss saw 1.000 everywhere, and nothing was clipped.

> **From the article** (Hugging Face and Liquid AI, "The ultimate guide to
> multi-harness RL", section "Why rewards are not enough"): "The probabilities
> need to be recorded during generation too. In asynchronous RL, the model may
> have been updated since a rollout was sampled. Recomputing log probabilities
> with the current weights does not recover the probabilities used during that
> rollout. Importance sampling compares the current probability with the one
> used during generation; using the current value for both would make their
> ratio 1 even when the policies differ."

## Reusing a batch, as PPO does

**PPO** (Proximal Policy Optimization, 2017) met us in chapter 5 as the
algorithm with a critic. It also gets more out of each batch: it samples once,
then makes several updates on the same rollouts. After the first update the
learner's weights have moved, so the rollouts are off-policy, though nothing
ran asynchronously. "Proximal" means "nearby": the clip keeps every update
near the weights that sampled the batch. GRPO keeps this part of PPO, the
ratio and the clip, and swaps the critic for the group mean.

```bash
uv run grpo_clip.py --updates-per-batch 8
```

```text
grpo_clip · seed 0 · lr 3 · 200 steps · 8 updates per batch
each step: 2 questions x 4 rollouts at temperature 1, sampled with the learner's weights
questions: 45 to train on, 19 held out (chapter 5's, one set of weights per level)
loss per call = -min(ratio x A, clip(ratio, 0.8, 1.2) x A), ratio = new p / old p
old p: recorded when the rollout was sampled

step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.383 · clipped 56%
step   2 · reward 0.25 · correct 25% · calls 6.6 · no-signal groups 0/2 · ratio 1.021 · clipped 10%
step   3 · reward 0.00 · correct 0% · calls 4.1 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   4 · reward 0.25 · correct 25% · calls 7.2 · no-signal groups 1/2 · ratio 1.035 · clipped 23%
step   5 · reward 0.00 · correct 0% · calls 4.2 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   6 · reward 0.00 · correct 0% · calls 4.5 · no-signal groups 2/2 · ratio n/a · clipped n/a
step   7 · reward 0.12 · correct 12% · calls 7.2 · no-signal groups 1/2 · ratio 0.996 · clipped 12%
step   8 · reward 0.12 · correct 12% · calls 6.2 · no-signal groups 1/2 · ratio 0.980 · clipped 12%
```

A step's line now covers all 8 updates of its batch. At the end, the ratios
are split by use of the batch:

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    1,897  1.000  0%        0%        0%        100%      0%        0%        0%        0%
use 2    1,897  0.999  21%       11%       22%       7%        15%       8%        15%       25%
use 3    1,897  0.988  18%       10%       23%       9%        21%       9%        11%       20%
use 4    1,897  0.992  21%       10%       19%       8%        19%       8%        13%       25%
use 5    1,897  0.994  17%       11%       20%       10%       21%       8%        12%       23%
use 6    1,897  0.992  21%       11%       18%       8%        19%       9%        14%       26%
use 7    1,897  0.994  18%       13%       19%       8%        21%       10%       11%       23%
use 8    1,897  1.001  20%       13%       20%       7%        18%       7%        15%       26%
mean ratio per step, over the 56 steps that learned: 0.865 to 1.383

rollouts right, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
easy    60%         95%       99%       98%
medium  23%         93%       91%       99%
hard    0%          0%        0%        0%

greedy test at the end (always the most likely action)
  easy    training 15 of 15  held out 8 of 8 · LOOK x2, QUERY_A, WRITE, LOOK, QUERY_A x3
  medium  training 16 of 16  held out 7 of 7 · LOOK x3, QUERY_A, WRITE x4
  hard    training  0 of 14  held out 0 of 4 · LOOK x8
```

Use 1 is a fresh batch, so all its ratios are exactly 1, as in the default
run. From use 2 on they spread: more than a third are outside 0.8 to 1.2, and
the clip switches off a quarter of the pushes. But they do not spread further
with each use; from use 2 to use 8 the picture stays about the same. One update at a learning
rate of 3 already moves many probabilities by more than 20%, and once a call
is clipped it is no longer pushed, so the clip holds the spread where it is.

Reuse pays: the same rollouts gave 8 times as many updates, and the agent
learned faster. In steps 1 to 50, 60% of easy rollouts and 23% of medium ones
were right, against 45% and 11% with one update per batch. That is why PPO
reuses its batches, and why it needs the clip to do it.

With smaller updates you can watch the spread grow, use by use:

```bash
uv run grpo_clip.py --updates-per-batch 8 --lr 0.5
```

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1    3,130  1.000  0%        0%        0%        100%      0%        0%        0%        0%
use 2    3,130  1.000  0%        5%        37%       22%       31%       4%        1%        2%
use 3    3,130  0.999  4%        10%       33%       12%       28%       9%        5%        7%
use 4    3,130  0.999  6%        14%       28%       9%        26%       11%       6%        10%
use 5    3,130  0.997  7%        17%       26%       8%        22%       13%       8%        13%
use 6    3,130  0.996  9%        16%       24%       7%        20%       14%       9%        16%
use 7    3,130  0.997  9%        18%       22%       7%        20%       14%       10%       17%
use 8    3,130  0.996  11%       17%       21%       6%        20%       15%       11%       18%
mean ratio per step, over the 91 steps that learned: 0.944 to 1.092
```

The middle bin shrinks use after use (100%, 22%, 12%, 9%, 8%, 7%, 7%, 6%),
the tails grow, and the share clipped rises with every use, from 0% to 18%.

## Without the clip

```bash
uv run grpo_clip.py --updates-per-batch 8 --no-clip
```

```text
grpo_clip · seed 0 · lr 3 · 200 steps · 8 updates per batch
each step: 2 questions x 4 rollouts at temperature 1, sampled with the learner's weights
questions: 45 to train on, 19 held out (chapter 5's, one set of weights per level)
loss per call = -ratio x A (no clip), ratio = new p / old p
old p: recorded when the rollout was sampled

step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · ratio 1.775 · no clip
step   2 · reward 0.00 · correct 0% · calls 8.0 · no-signal groups 2/2 · ratio n/a · no clip
step   3 · reward 0.00 · correct 0% · calls 3.2 · no-signal groups 2/2 · ratio n/a · no clip
step   4 · reward 0.00 · correct 0% · calls 8.0 · no-signal groups 2/2 · ratio n/a · no clip
step   5 · reward 0.00 · correct 0% · calls 4.5 · no-signal groups 2/2 · ratio n/a · no clip
step   6 · reward 0.00 · correct 0% · calls 4.8 · no-signal groups 2/2 · ratio n/a · no clip
step   7 · reward 0.00 · correct 0% · calls 8.0 · no-signal groups 2/2 · ratio n/a · no clip
step   8 · reward 0.00 · correct 0% · calls 7.1 · no-signal groups 2/2 · ratio n/a · no clip
```

```text
ratios the loss saw, by use of the batch (every step of the run)
         calls  mean   <0.8      0.8-0.9   0.9-0.99  0.99-1.01 1.01-1.1  1.1-1.2   >1.2      clipped
use 1       59  1.000  0%        0%        0%        100%      0%        0%        0%        no clip
use 2       59  1.308  17%       17%       36%       0%        10%       0%        20%       no clip
use 3       59  5.463  80%       0%        0%        0%        0%        0%        20%       no clip
use 4       59  5.475  80%       0%        0%        0%        0%        0%        20%       no clip
use 5       59  5.766  80%       0%        0%        0%        0%        0%        20%       no clip
use 6       59  2.741  81%       0%        0%        0%        0%        0%        19%       no clip
use 7       59  5.385  83%       0%        0%        0%        0%        0%        17%       no clip
use 8       59  5.388  83%       0%        0%        0%        0%        0%        17%       no clip
mean ratio per step, over the 3 steps that learned: 1.229 to 7.351

rollouts right, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
easy    1%          0%        0%        0%
medium  1%          0%        0%        0%
hard    0%          0%        0%        0%

greedy test at the end (always the most likely action)
  easy    training  0 of 15  held out 0 of 8 · WRITE x8
  medium  training  0 of 16  held out 0 of 7 · WRITE x8
  hard    training  0 of 14  held out 0 of 4 · LOOK x8
```

Without the clip, the loss for a call is -ratio x A and its push is ratio x A.
A call whose rollout did better than its group (A > 0) has its action made
more likely; that raises its ratio, and the next use pushes it harder still.
A call with A < 0 has its action made less likely, its ratio falls toward 0,
and its push fades. Nothing stops the first kind. By the third use of a batch,
80% of the ratios were below 0.8 and 20% above 1.2, with a mean of 5.463. One
action, `WRITE`, took over: the greedy agent writes at every call, before it
has any result, so it never answers. From step 2 on the agent almost never
gets a rollout right, so almost no group has contrast, and only 3 of the 200
steps learned anything. Seed 0 is one of the worst cases; `seeds.py` shows the
rest.

## Not one lucky seed

```bash
uv run seeds.py
```

```text
seeds 0-9 · 200 steps per run · lr 3 (one run at lr 0.5) · 7 runs per seed

1. the ratios the loss saw: % outside 0.8-1.2 / % clipped
seed  fresh            stale 2          stale 2, recompute-old
                                        as the loss saw  against the recorded p
 0    0% / 0%          38% / 20%        0% / 0%          41% / 23%
 1    0% / 0%          34% / 19%        0% / 0%          49% / 25%
 2    0% / 0%          43% / 23%        0% / 0%          39% / 21%
 3    0% / 0%          37% / 20%        0% / 0%          35% / 20%
 4    0% / 0%          40% / 22%        0% / 0%          40% / 22%
 5    0% / 0%          40% / 20%        0% / 0%          37% / 20%
 6    0% / 0%          44% / 24%        0% / 0%          47% / 26%
 7    0% / 0%          35% / 20%        0% / 0%          48% / 26%
 8    0% / 0%          36% / 20%        0% / 0%          50% / 28%
 9    0% / 0%          42% / 22%        0% / 0%          45% / 25%
all   0% / 0%          39% / 21%        0% / 0%          43% / 23%
fresh: 31,385 of 31,385 ratios exactly 1
stale 2, recompute-old, as the loss saw: 32,040 of 32,040 ratios exactly 1

2. ratios outside 0.8-1.2 at each use of the batch (8 updates per batch), %
      lr 3, use 1 to 8                  lr 0.5, use 1 to 8
seed     1   2   3   4   5   6   7   8     1   2   3   4   5   6   7   8
 0       0  37  29  34  29  35  29  35     0   2   9  12  15  19  19  22
 1       0  35  29  34  31  30  34  35     0   2   9  12  16  17  20  21
 2       0  33  25  37  29  35  32  36     0   1   7  12  15  17  20  20
 3       0  29  26  28  30  28  31  30     0   1   9  12  15  16  19  20
 4       0  34  26  31  33  34  36  36     0   1   6   9  14  15  19  20
 5       0  32  24  31  28  32  29  34     0   1   6  11  14  18  18  21
 6       0  23  25  24  27  24  26  27     0   1   7  11  14  17  18  20
 7       0  33  26  32  28  36  29  35     0   1   5   9  12  13  17  17
 8       0  26  25  25  25  28  27  27     0   2   9  14  15  19  20  22
 9       0  37  28  34  29  36  30  36     0   1   7  11  13  17  18  20
all      0  32  26  31  29  32  30  33     0   1   7  11  14  17  19  20

3. calls clipped at each use of the batch (8 updates per batch), %
      lr 3, use 1 to 8                  lr 0.5, use 1 to 8
seed     1   2   3   4   5   6   7   8     1   2   3   4   5   6   7   8
 0       0  25  20  25  23  26  23  26     0   2   7  10  13  16  17  18
 1       0  25  22  26  25  24  27  28     0   1   7  10  14  14  17  17
 2       0  23  19  28  23  27  25  29     0   1   6  10  13  15  17  17
 3       0  22  21  22  23  23  25  25     0   1   8  10  13  14  16  18
 4       0  25  21  24  25  26  28  29     0   1   6   8  12  13  16  18
 5       0  23  19  24  22  25  24  26     0   1   5   9  12  15  15  18
 6       0  18  20  20  21  21  21  23     0   1   6   9  12  14  16  17
 7       0  24  20  24  22  28  23  28     0   1   4   8  10  11  14  15
 8       0  19  20  21  21  23  22  23     0   2   7  12  13  17  18  19
 9       0  25  20  25  22  27  23  27     0   1   6   9  11  14  15  16
all      0  23  20  24  23  25  24  26     0   1   6  10  12  15  16  17

4. easy and medium rollouts right, in each quarter of the run, %: rollouts 2 updates old
seed  fresh             stale 2           stale 2,          stale 2,
                                          recompute-old     no clip
 0      29  68  88  95     7  12  40  76    15  43  81  88    15  56  78  95
 1      17  36  67  90     3  14  44  75    11  37  72  92    12  35  70  92
 2      17  42  49  73     3  29  57  82     3  39  77  96     3  41  82  95
 3      24  55  86  96     5  14  30  55    10  34  52  74    13  44  70  89
 4      21  44  80  93     7  17  51  83    13  41  77  94     9  38  79  92
 5      12  51  86  94     9  28  65  87     8  59  86  97     5  18  45  46
 6      29  64  86  95     7  27  46  71    14  42  69  94    16  60  77  92
 7      13  44  75  91     1   7  19  49     1  12  42  76     1  12  37  79
 8      23  57  85  93    10  34  71  82    12  32  58  63     6  46  83  89
 9      10  53  87  94     1   6  15  50     1  15  56  81     1  16  48  82
all     20  52  79  91     5  19  44  71     9  36  67  85     8  37  67  85

5. the same: each batch used for 8 updates
seed  fresh             8 updates         8 updates,
                                          no clip
 0      29  68  88  95    42  94  95  99     1   0   0   0
 1      17  36  67  90    21  65  96  97    32  44  95  98
 2      17  42  49  73    32  85  96  98    16  87 100  97
 3      24  55  86  96    32  61  93  97     2  30  53  51
 4      21  44  80  93    28  79  98  99     2  23  45  52
 5      12  51  86  94    35  90  96  97    25  46  50  49
 6      29  64  86  95    45  90  96  98     1   1   2   0
 7      13  44  75  91    14  83  98  99     1   0   0   0
 8      23  57  85  93    44  94  97  98     0   0   1   1
 9      10  53  87  94    39  92  97  98     5  38  51  50
all     20  52  79  91    33  83  96  98     9  27  40  40
```

Each table has one row per seed, and an `all` row that adds the ten seeds
together. Read them one claim at a time.

1. **Fresh batches give exactly 1, stale ones do not, and the bug hides it.**
   Every one of 31,385 fresh ratios, over all ten seeds, was exactly 1, and
   nothing was clipped. With rollouts two updates old, 34% to 44% of the ratios
   were outside 0.8 to 1.2 in every seed, and the clip switched off 19% to 24%
   of the pushes. With `--recompute-old`, every one of 32,040 ratios the loss
   saw was exactly 1, while the same calls' ratios against the recorded
   probabilities were outside the range 35% to 50% of the time.
2. **Reusing a batch spreads the ratios.** At a learning rate of 3, use 1 is
   always 0% outside and use 2 is 23% to 37%, in every seed; after that it
   stays about there (the `all` row runs 32, 26, 31, 29, 32, 30, 33). At 0.5 it
   never shrinks from one use to the next, in any seed, and reaches 17% to 22%
   at use 8.
3. **The clip switches off more pushes as a batch is reused.** From 0% at use
   1 to about a quarter from use 2 on at a learning rate of 3; at 0.5 it climbs
   use by use, to 15% to 19% at use 8.
4. **What staleness costs.** Rollouts two updates old learned more slowly than
   fresh ones in every seed (`all`: 5% right in steps 1 to 50 against 20%). The
   run with the bug learned faster than the honest stale run (9%, then 36%),
   and so did the stale run with no clip (8%, then 37%), at the same pace as
   the bug. So in this toy the correction itself neither helps nor hurts; what
   slows the honest run is the clip, switching off a fifth of its pushes.
5. **What the clip buys.** Eight updates per batch with the clip learned faster
   than one update in all ten seeds (`all`: 33% right in steps 1 to 50 against
   20%, and 98% by the end). Without the clip, the same reuse was erratic:
   seeds 0, 6, 7 and 8 collapsed and stayed near 0%, seeds 3, 4, 5 and 9
   stalled at about half, and only seeds 1 and 2 learned well.

## The run's log, and its curves

Every step appends one line to `metrics.jsonl`. The fields the project's
trainer also logs keep its names (`accuracy`, `mean_reward`, `mean_calls`,
`groups_without_contrast`, `importance_ratio_mean`), and each line adds how
many ratios the loss saw, how many fell outside 0.8 to 1.2 and how many were
clipped, their histogram, how many updates behind the sampler was
(`sampler_lag`), and the same counts for each use of the batch. Every rollout,
call by call with each action's old p, goes into `rollouts.jsonl`. Same seed,
same files: every run prints both md5s, and `--trace` only prints more.

```text
wrote ~/.cache/mhrl-course/06-ratio-and-clip/runs/seed0-lr3-steps200-updates1-stale2/metrics.jsonl
md5   5cbdc27a39f6f88498e7457df3b6358b
wrote ~/.cache/mhrl-course/06-ratio-and-clip/runs/seed0-lr3-steps200-updates1-stale2/rollouts.jsonl
md5   ac9462f8dcbc937487fc135d26a12fcd
redraw it: uv run plot.py ~/.cache/mhrl-course/06-ratio-and-clip/runs/seed0-lr3-steps200-updates1-stale2/metrics.jsonl
```

`plot.py` redraws a run from `metrics.jsonl` alone, as text, with Python's
standard library only. Each point averages the last tenth of the run.

```bash
uv run plot.py ~/.cache/mhrl-course/06-ratio-and-clip/runs/seed0-lr3-steps200-updates1-stale2/metrics.jsonl
```

```text
200 steps, 4,122 ratios; each point is the average of the 20 steps up to it

correct, by difficulty (e easy, m medium, h hard, * where they meet)
 100%|
     |                                           ee
     |                                         ee  *e  m
     |                                   e    e   m m**e
     |                                 ee ee** mmm
  50%|                                e  mmm  m
     |                               emmm
     |                             e*
     |             e        eee****m m
     |  eeeeeeee*e*m****mm**mmm
   0%|**********h*hhhhhh**hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
     +--------------------------------------------------
      1                                              200  step

ratios outside 0.8-1.2 (c clipped, o not clipped), as a share of the ratios the loss saw
 100%|
     |
     |    o
     |   ooooo
     |  ooooooo
  50%|  oooooooooo   ooo                    oo  o
     |  cccccooooo oooooo            oo  oooooooooooo
     |  ccccccccooooccccoo  o      oooooooooocoooooooooo
     |  ccccccccccccccccccooooooooocccccccccccccccccccco
     |  cccccccccccccccccccccccccccccccccccccccccccccccc
   0%|
     +--------------------------------------------------
      1                                              200  step
```

The first chart is the stale run's slow start: easy and medium questions take
about 120 steps to reach 30% right. The second shows the ratios outside 0.8 to
1.2 through the whole run, the clipped ones (`c`) at the bottom and the rest
(`o`) above. Early on more than half the ratios are outside the range; for the
rest of the run between a third and a half are, and the clip switches off
about a fifth of the pushes throughout. Staleness does not wear off as the
agent learns.

## What the project's real runs measured

```bash
uv run real_runs.py
```

```text
the project's two real RL runs: LFM2.5-2.6B trained inside real
harnesses on one MacBook, every step 2 tasks x 4 rollouts sampled with
the newest weights, then one update

mean importance ratio per step, as the trainer logged it
step  attempt 1    attempt 2
   1  0.996        1.000
   2  0.997        1.001
   3  0.996        0.999
   4  1.000        1.001
   5  0.995        1.000
   6  0.997        1.001
   7  0.998
   8  0.999
range 0.995-1.000  0.999-1.001

the engine that sampled them scored tokens in 16-bit numbers (bf16) in
attempt 1 and in 32-bit numbers (float32) in attempt 2
from: the project's runs/main/grpo_attempt1/metrics.jsonl and
runs/main/grpo/metrics.jsonl, copied into data/real_runs/
```

The project trained LFM2.5-2.6B with GRPO inside real harnesses twice. Each
step sampled its rollouts with the newest weights and made one update, the same
structure as this chapter's default run. So the ratio between the engine's
recorded probabilities and the trainer's should be 1, and it nearly is:
0.995 to 1.000 in attempt 1 and 0.999 to 1.001 in attempt 2 (the project's
`runs/main/grpo_attempt1/metrics.jsonl` and `runs/main/grpo/metrics.jsonl`,
field `importance_ratio_mean`; its README reports 0.995 to 1.000, and 1.000
once the engine scored in 32-bit numbers). With means this close to 1, the
update is close to REINFORCE with a group baseline, as in the default run. (The
project did not log a share clipped, only the mean.)

It is not exactly 1, because the engine and the trainer are different programs.
In attempt 1 the engine scored tokens in 16-bit numbers, and the ratio sat just
below 1; in attempt 2 it scored in 32-bit numbers, and the ratio centred on 1.
The recorded probabilities are what showed that gap. Had the trainer recomputed
"old" with its own weights, both runs would have read exactly 1.000, gap or no
gap. A mean is weak evidence on its own (the stale run above also averaged
1.000); what makes these means believable is that the project's steps were
on-policy by construction, as in this chapter's default run. Chapter 15
measures that precision gap directly.

> **From the article** (Hugging Face and Liquid AI, "The ultimate guide to
> multi-harness RL", section "Training small models across harnesses", Setup):
> "Async GRPO in TRL for 1,000 steps. Async means generation and training run
> apart. One H100 trains while a second serves the policy with vLLM (Kwon et
> al., 2023) and keeps producing rollouts, so training never waits for the
> slowest agent. The price is that rollouts come from weights usually two
> optimizer steps old, and never more than four."

That is the situation `--stale 2` imitates, and the one where the recorded
probabilities matter: only they can show how far the sampler's weights were
from the learner's, and only they let the clip limit it.

## What the toy showed, and where it differed from the plan

- **"Ratio 1.000 on every step"** holds on every step that learns. A step whose
  groups all tie has no call to weigh and prints `ratio n/a`, as the project's
  trainer does: 101 of the default run's 200 steps.
- **"The histogram widens with each reuse"** holds at a learning rate of 0.5.
  At the default 3 it widens at the first reuse and then holds, because the
  clip stops the calls that have moved far enough.
- **"Only the recorded probabilities can correct for staleness"** is true of
  the arithmetic: nothing else can work out the ratio. In this toy the
  correction did not make the stale runs learn faster, and the clip on top of
  it made them slower. What the recorded probabilities bought was the true
  ratio, which the bug hides, and the clip that rests on it, which is what
  kept reuse from collapsing.
- **The mean ratio** stays near 1 even when many ratios are far from it, so the
  chapter shows the spread and the share clipped beside it.

## What is in this folder

- `grpo_clip.py`: GRPO with the ratio and the clip on the tiny agent: the
  steps, the stale sampler, the reuse, the bug, the logs, and `--token` and
  `--trace`.
- `grpo.py`: chapter 5's `group_advantages`, the project's `loss_fn` line for
  line, the same loss without the clip, and its gradient worked out by hand.
- `report.py`: how runs are printed, and the seed tables.
- `seeds.py`: seven runs over seeds 0 to 9, side by side on the Mac's cores.
- `plot.py`: the curves, drawn as text from `metrics.jsonl` alone (standard
  library only).
- `real_runs.py`, `data/real_runs/`: the project's two RL metrics logs, copied
  unchanged, and the closing card printed from them.
- `environment.py`, `policy.py`, `questions.py`, `grader.py`, `data/shop.db`:
  chapter 5's agent and database, copied unchanged.
- `tasks.py`: chapter 5's `tasks.py`, writing under `06-ratio-and-clip`.

## Words from this chapter

- **On-policy, off-policy**: learning only from rollouts made by the weights as
  they are now, or also from rollouts made by other (older) weights.
- **Old p, new p**: an action's probability when its rollout was sampled, and
  its probability with the learner's weights now.
- **Importance ratio**: new p / old p, worked out as exp(new log p - old log
  p). It says how much more or less likely an action has become since it was
  sampled.
- **Clip**: limiting the ratio to between 0.8 and 1.2 (`CLIP_EPSILON` 0.2). A
  call is **clipped** when `min` keeps the clipped term, which has no slope, so
  its push is switched off.
- **Push on log p**: how hard a step raises or lowers the log-probability of a
  call's action; A in chapters 3 to 5, ratio x A here.
- **Pessimistic `min`**: of the two estimates, keep the one worse for the new
  weights, so no update looks better than it is.
- **Stale rollouts, asynchronous RL**: sampling and learning run side by side,
  so rollouts come from weights a few updates old.
- **Updates per batch (reuse)**: learning from the same rollouts several times.
- **PPO (Proximal Policy Optimization)**: the algorithm that reuses each batch
  and keeps every update near the weights that sampled it, with the ratio and
  the clip. GRPO keeps that part and drops PPO's critic.
- **Recompute-old bug**: working out the "old" probabilities with the learner's
  weights instead of keeping the sampler's, so the ratio reads 1.000 whatever
  the gap.

Next: [chapter 7](../README.md#chapters), Rewards that backfire.
