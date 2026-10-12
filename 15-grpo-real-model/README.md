# Chapter 15. GRPO on the real model, and why exact tokens matter

Chapter 6 trained a toy agent with GRPO: a group of attempts per question,
each pushed up or down by how it compared with its group, through a loss
that weighs every action by its importance ratio and clips it. Chapter 12
kept, for every token the model drew, its id and its log-probability.
Chapter 14 fine-tuned an adapter on LFM2.5-2.6B. This chapter puts the three
together, as the project's second training stage does
(`mhrl/training/grpo.py`): the model writes 4 replies to each question, the
grader rewards the right ones, and one clipped update makes the better
replies likelier, in a loop we own from the first token to the last.

Then it looks hard at the importance ratio. Everything the trainer learns
from is a token the sampler drew, and the ratio compares the sampler's
number for that token with the trainer's. The flags break that agreement on
purpose: training on re-tokenized text, recording bfloat16
log-probabilities, drawing from the 50 likeliest tokens only.

The puzzle to keep in mind: the sampler draws a reply and records each
token's log-probability; the trainer then scores exactly those tokens, with
exactly the same weights, in the same process, a moment later. Nothing has
been trained in between. Is every token's importance ratio exactly 1?

## Words first

- **GRPO** (chapters 5 and 6): for one question, draw a **group** of
  replies, give each a **reward**, and compare each reward with the group's
  average. A reply's **advantage** is its reward minus the group's mean,
  divided by the group's spread. Training makes every token of a reply with
  a positive advantage likelier and every token of one with a negative
  advantage less likely. A group whose rewards are all equal has nothing to
  compare, is skipped, and is counted as a **no-signal group**.
- A **step** (the course's word since chapter 3) is one optimizer update.
  Here each step draws 2 questions x 4 replies = 8 replies, then updates
  once, as the project's `train_grpo` does.
- The **reward** is the project's (`mhrl/rollout.py`): a reply whose SQL
  gives the right answer earns 1 plus a **token bonus** of up to 0.1, which
  falls in a straight line from 0.1 at 0 generated tokens to 0 at 2,000; a
  wrong reply earns 0 however short it is.
- The **sampler** draws the replies (chapter 12's loop: one token at a time,
  with a cache); the **trainer** scores them and updates the adapter
  (chapter 14's scoring: the whole row in one pass). In the project they are
  two programs, the engine and the trainer, each with its own copy of the
  model; here they are one model with one adapter.
- A token's **old log-probability** is the one the sampler recorded when it
  drew the token, from the distribution it drew it from. Its **new
  log-probability** is the one the trainer works out for the same token, at
  the same position, with the weights it has, just before the update.
- The **importance ratio** (chapter 6) is new probability / old probability,
  `exp(new - old)`, one per token. If the trainer and the sampler compute
  the same numbers, it is exactly 1. The **clip** (chapter 6) stops a
  token's push once its ratio leaves 0.8 to 1.2 in the direction its
  advantage wants; the **clipped share** is the share of tokens where it did.
- **bfloat16** and **float32** (chapters 10 and 12) are 16-bit and 32-bit
  numbers. The 4-bit model works out its scores in bfloat16, which keeps
  about 3 significant digits.
- A **route** is the order of operations by which a number is worked out.
  Two routes through the same arithmetic can round differently; when the
  sampler's and the trainer's numbers for the same token disagree, that is
  a **train-inference mismatch**.
- **Token in, token out**: the trainer is given the exact token ids the
  sampler drew and their old log-probabilities, never text to tokenize again.
  **Re-tokenizing** is the opposite: decode the reply to text, encode the
  text again, and train on what comes out (chapter 9: the same text can be
  many token sequences).
- **Entropy** (chapter 12) is how spread out the distribution a token is
  drawn from is, in nats; **entropy collapse** is training pushing it toward
  0, until every reply in a group is the same and GRPO has nothing to
  compare. **Top-k** (chapter 12) draws from the k likeliest tokens only.
- **Drift** from the starting model is measured per token as
  `exp(ref - new) - (ref - new) - 1`, where `ref` is the token's
  log-probability under the adapter training started from: 0 when the two
  agree, positive otherwise. Averaged over sampled tokens it estimates the
  **KL divergence** (Kullback-Leibler: how far one model's distribution is
  from another's) between the current model and that **reference model**;
  this is the estimate GRPO's paper uses. A **KL penalty** adds a multiple
  of it to the loss, which keeps the model near where it started.
- **pass@1**, the **error bar**, **gained**, **lost** and the **exact
  p-value** are chapter 14's: the share right on one greedy try; plus or
  minus one standard error, sqrt(p x (1 - p) / n); questions wrong before
  and right after, and the reverse; and the chance of a split at least that
  uneven if the two models were equally good (McNemar's exact test).

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/15-grpo-real-model
uv run same_tokens.py
uv run grpo_sql.py
uv run eval_sql.py
```

All three need an Apple-silicon Mac (they use MLX), the 4-bit model and
chapter 14's adapter. The model is chapter 11's own build in
`COURSE_HOME/models/lfm2.5-2.6b-4bit-ours` (default `~/.cache/mhrl-course`)
or, if you have not built it, Liquid's export that chapter 10's
`download.py --fast` fetched (the same bytes, chapter 11 checked). The
adapter is `sft.safetensors`, which chapter 14's `uv run train_sft.py`
writes to `COURSE_HOME/14-sft-real-model` in about two minutes; the scripts
stop and say so if it is missing. Nothing is downloaded, and one model is
in memory at a time. Adapters, logs, evaluations and each reply's
`answer.txt` go under `COURSE_HOME/15-grpo-real-model`. The 230 questions
and their five SQLite databases are bundled in `data/` (chapter 14's
files), with the project's two recorded GRPO runs.

| command | what it shows | time in the run log |
| --- | --- | ---: |
| `same_tokens.py` | one reply's tokens scored by the sampler's route, in one pass and inside a gradient computation, in bfloat16 and float32 | 85 s |
| `grpo_sql.py` | 20 steps of the project's GRPO: every step, step 1 in detail, the ratio token by token, the groups, the project's recorded runs | 46 min |
| `eval_sql.py` | base, SFT and SFT+GRPO on the 50 held-out questions: verdicts, pass@1 with error bars, gained and lost, reply lengths | 6.7 min (base answers read back) |
| `grpo_sql.py --float32 --steps 4` | the same loop with the whole model in float32 | 2.4 min |
| `grpo_sql.py --retokenize --steps 4` | training on decoded-and-re-encoded replies | 1.2 min |
| `grpo_sql.py --logprob-dtype bf16 --steps 4` | bfloat16 log-probabilities, as the project's first RL run recorded them | 2.0 min |
| `grpo_sql.py --top-k 50 --steps 4` | drawing from the 50 likeliest tokens only | 1.7 min |
| `grpo_sql.py --kl 0.05 --steps 4` | a penalty for drifting from chapter 14's adapter | 1.7 min |
| `grpo_sql.py --fast`, `eval_sql.py --fast --adapter grpo_steps4.safetensors` | fast mode: 4 steps on Liquid's export, 10 held-out questions | 2.2 min, 12 s |

The default run takes the project's 20 steps (`CliDefaults.GRPO_STEPS` in
`mhrl/cli.py`); each flag's run takes 4 (the plan's fast mode), enough to
measure what the flag does to the ratio, not to train with it. Every
number below comes from recorded runs of these commands on an M3 Pro with
18 GB, in two sessions at the same code: `same_tokens.py` and the 20-step
run with video renders on the same machine, the rest on a quieter one,
which is why 20 steps took 46 minutes and 4 steps about 2 (each command's
commit, date, exit code and time are in the run log). Only the step
tables' **peak** column and the lines marked "varies from run to run" are
memory or times. Everything else is determined: fixed data, a fixed order
of questions, a fixed seed for the draws, and greedy replies in the
evaluation (the run log checks it: the fast run's 4 steps repeat the
default run's first 4, reply for reply).

## Is the ratio 1? One reply, scored three ways

```bash
uv run same_tokens.py
```

Start with the simplest case GRPO can meet. Nothing is trained.
`same_tokens.py` draws the reply `grpo_sql.py` draws first (the same
question, adapter and seed), with every token's log-probability recorded,
then scores exactly those tokens three more ways with exactly the same
weights:

```text
same_tokens.py · the same tokens, scored by the sampler and by the trainer
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
adapter   ~/.cache/mhrl-course/14-sft-real-model/sft.safetensors, chapter 14's
reply     to school-easy-018 ('What was the average grade in the Fall 2024 term? Round
  to 2 decimal places.'), drawn at temperature 1 from the full distribution with
  seed 0, as grpo_sql.py's first reply is: 76 tokens, finish_reason 'stop'
text      "Use the enrollments and grades which are in a 2-digit, foreign key-linked
  columns. Use reversed 2-digit calculations rather than string matching (fall
  2024 matches term='Fall 2024'), then compute average.</think>SELECT
  ROUND(AVG(e.grade)..."

1. bfloat16, as the model runs: each route against what the sampler recorded
```

```text
1. bfloat16, as the model runs: each route against what the sampler recorded
  the sampler's route again (one token at a time, with a cache)
    identical 76 of 76; largest |difference in log-probability| 0.000000; ratio mean
      1.0000, lowest 1.000, highest 1.000
  one pass over the row
    identical 0 of 76; largest |difference in log-probability| 0.179659; ratio mean
      0.9957, lowest 0.836, highest 1.177
  one pass inside a gradient computation (the trainer's update)
    identical 2 of 76; largest |difference in log-probability| 0.185867; ratio mean
      0.9951, lowest 0.854, highest 1.204
```

- **The sampler's own route again** (each drawn token fed back in, one at a
  time, with a cache) gives the recorded numbers back, all 76 of them, to
  the last bit. Once a token is drawn, its recorded number is not random:
  it can be worked out again.
- **One pass over the row**, as chapter 14's `token_logprobs` reads it,
  gives none of the 76 exactly. The ratio averages 0.9957, close to 1, but
  single tokens run from 0.836 to 1.177.
- **One pass inside a gradient computation**, the route the update takes,
  differs again: 0.854 to 1.204, mean 0.9951. This is the number the
  importance ratio is made of.

The tokens that moved most:

```text
2. the 5 tokens whose ratio moved most, the trainer's route
  position  token              sampler log p  trainer log p   ratio
        12  '2'                    -4.568678      -4.692579  0.8835
        13  '-digit'               -4.384829      -4.542633  0.8540
        15  ' foreign'             -4.808025      -4.622158  1.2043
        21  ' reversed'           -15.205227     -15.042521  1.1767
        24  '-digit'               -1.903652      -2.022793  0.8877
  position: the token's place in the reply, from 1; ratio: exp(trainer log p - sampler
    log p)
```

`' foreign'` is 20% likelier to the trainer than the sampler recorded,
`'-digit'` 15% less likely. All five were long shots when they were drawn
(log-probabilities from -1.9 to -15.2). That fits how a log-probability is
made: a token's score minus the log of the sum over all 128,000 scores. For
a likely token, a change in its own score moves that sum almost as much and
the two cancel; for an unlikely one, nothing cancels. Now the same tokens
in float32:

```text
3. float32 (every 16-bit number cast): each route against the sampler's
  the sampler's route again (one token at a time, with a cache)
    identical 76 of 76; largest |difference in log-probability| 0.000000; ratio mean
      1.0000, lowest 1.000, highest 1.000
  one pass over the row
    identical 41 of 76; largest |difference in log-probability| 0.000029; ratio mean
      1.0000, lowest 1.000, highest 1.000
  one pass inside a gradient computation (the trainer's update)
    identical 39 of 76; largest |difference in log-probability| 0.000053; ratio mean
      1.0000, lowest 1.000, highest 1.000
  the float32 model is also a slightly different model: its own numbers for these
    tokens, against what the bfloat16 sampler recorded, give a ratio of mean 0.9986,
    lowest 0.839, highest 1.193
```

Cast every 16-bit number of the model to float32 (the 4-bit codes stay as
they are) and the three routes agree: the largest difference is 0.00005 in
log-probability, and every ratio rounds to 1.000. So the scatter is not
GRPO's and not the adapter's: it is bfloat16. The model works out its
scores in bfloat16, which keeps about three significant digits, so a score
near 30 can only move in steps of 1/8 (chapter 12). The sampler reads the
row one token at a time with a cache; the trainer reads it all at once; and
inside a gradient computation MLX does attention as separate steps instead
of its fused kernel. The same arithmetic in another order rounds
differently, and one step of 1/8 in a token's score is a ratio of
e^0.125 = 1.13. (While building the chapter, attention written out as
separate steps in an ordinary pass reproduced the gradient route's numbers
exactly, to 0.0; that note is kept beside the run log.) The last line is a
warning for later: the float32 model is a slightly different model. Its own
numbers for these tokens, against the ones the bfloat16 sampler recorded,
give ratios from 0.839 to 1.193.

What this means for training:

- the ratio averaged over a step's tokens stays near 1, and that average
  is all the project's logs show;
- single tokens scatter by up to about a fifth either way, so a few land
  outside the clip's 0.8 to 1.2 and the clip acts on them even though
  nothing is stale;
- chapter 6 warned that a mean of 1.000 is weak evidence on its own; this
  is the spread such a mean can hide.

## Training: 20 steps of the project's GRPO

```bash
uv run grpo_sql.py
```

`grpo_sql.py` is the inner part of the project's `train_grpo`
(`mhrl/training/grpo.py`) with the harness taken out: one question is one
model call, so a rollout is one reply. It loads the model, adds chapter
13's adapter shape and loads chapter 14's `sft.safetensors` into it, then
says what it will do:

```text
grpo_sql.py · GRPO on the real model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py; 4-bit weights, bfloat16 numbers
start     ~/.cache/mhrl-course/14-sft-real-model/sft.safetensors: chapter 14's SFT
  adapter (rank 16, scale 2.0, the last 12 layers); sampler and trainer are one
  model with one adapter
steps     20, each 2 of the 180 training questions x 4 replies = 8 replies, in a fixed
  shuffled order (the project's train_grpo, seed 0)
sampling  temperature 1, every token, up to 1,024 tokens, MLX seed 0 for the draws; each
  token's log-probability recorded in float32
reward    1 + token bonus (up to 0.1, zero at 2,000 generated tokens) if chapter 1's
  grader accepts the reply's answer.txt, else 0
update    the project's clipped loss (0.8 to 1.2); one AdamW step per step at learning
  rate 1e-05, bias correction on (mhrl/training/grpo.py)
```

Each step:

1. takes the next 2 of the 180 training questions, in an order shuffled
   once with seed 0, as `train_grpo` does;
2. draws 4 replies to each with `sampler.py`, chapter 12's loop at
   temperature 1 over the full distribution, keeping for every token its id,
   its float32 log-probability and the distribution's entropy;
3. grades each reply as chapter 14 grades one (its SQL run read-only, the
   result written to `answer.txt`, chapter 1's grader), and rewards it
   `reward_for(correct, tokens, 2000)`;
4. turns each group's 4 rewards into advantages with `group_advantages`,
   or skips the group if they are all equal;
5. turns every reply of a group with contrast into the row the capture
   proxy would record (the prompt's ids, the sampled ids, a loss mask of 1
   over them, their recorded log-probabilities) and that row into a
   `TrainItem` with `item_from_sequence`, unchanged from the project;
6. makes one update: each row scored by `token_logprobs`, its loss the
   project's, its gradient divided by the step's trained tokens, then one
   AdamW step on the adapter.

The loss is the project's `loss_fn`, line for line, as chapter 6 copied it
into NumPy:

```python
        new = token_logprobs(model, item)
        ratio = mx.exp(new - mx.array(item.old_logprobs))
        unclipped = ratio * item.advantage
        clipped = mx.clip(ratio, 1.0 - eps, 1.0 + eps) * item.advantage
        return -mx.minimum(unclipped, clipped).sum(), mx.stop_gradient(ratio)
```

`ratio` is worked out before the update, with the weights that drew the
replies, so it measures only how far the trainer's numbers are from the
sampler's. One line per step:

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak
     1    3/8    0.41     121        1/2  1.0014  0.779-1.390       0.6%    0.762  4.06 GB
     2    2/8    0.27     103        1/2  1.0007  0.850-1.280       0.0%    0.741  4.06 GB
     3    2/8    0.27     273        0/2  0.9990  0.754-1.262       0.1%    0.856  6.36 GB
     4    2/8    0.27     191        0/2  1.0009  0.707-1.632       0.3%    1.066  6.36 GB
     5    4/8    0.55     112        0/2  0.9991  0.709-1.281       0.4%    0.938  6.36 GB
     6    5/8    0.68     184        0/2  0.9989  0.792-1.235       0.1%    0.994  6.36 GB
     7    3/8    0.41     124        1/2  1.0026  0.808-1.208       0.6%    0.829  6.36 GB
     8    8/8    1.10      99        0/2  1.0020  0.634-1.327       0.1%    0.927  6.36 GB
     9    4/8    0.54     160        0/2  0.9980  0.633-1.424       0.4%    1.020  6.36 GB
    10    2/8    0.27     180        1/2  1.0000  0.852-1.260       0.5%    1.215  6.36 GB
    11    5/8    0.68     225        0/2  1.0006  0.785-1.256       0.2%    0.761  6.36 GB
    12    5/8    0.68     102        0/2  1.0002  0.825-1.213       0.0%    0.856  6.36 GB
    13    3/8    0.41     198        1/2  0.9982  0.776-1.261       0.1%    0.728  6.36 GB
    14    5/8    0.68     146        0/2  1.0010  0.705-1.344       0.3%    0.822  6.36 GB
    15    6/8    0.82      78        0/2  0.9988  0.730-1.282       0.2%    1.032  6.36 GB
    16    1/8    0.14      81        1/2  1.0017  0.781-1.246       0.7%    0.613  6.36 GB
    17    4/8    0.55     143        0/2  1.0005  0.757-1.273       0.3%    1.084  6.36 GB
    18    3/8    0.41     188        1/2  0.9988  0.815-1.207       0.2%    0.890  6.36 GB
    19    4/8    0.55     137        0/2  1.0004  0.742-1.293       0.5%    0.912  6.36 GB
    20    5/8    0.69      70        0/2  1.0003  0.833-1.272       0.2%    0.901  6.36 GB
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  varies from run to run: the time (20 steps took 2,698 s, 2,514 s of it sampling) and
    the peak memory (6.36 GB, the weights included)
```

Reading the columns:

- **right** swings from 1 of 8 (step 16) to 8 of 8 (step 8), because each
  step asks 2 new questions and a step's score is mostly how hard those two
  are. Eight replies a step are no progress meter; the project saw the same
  in its own runs (below).
- **no-signal** groups (all four wrong) were skipped 7 times in 40 groups.
  A group is never skipped for being all right: four right replies of
  different lengths earn different bonuses (next section).
- **ratio** is the project's number, the mean over every trained token, and
  it sits where the project's sat: 0.9980 to 1.0026 per step. **per token**
  is what that mean hides: the lowest and highest single token run from
  0.633 to 1.632, `same_tokens.py`'s bfloat16 scatter, step after step.
- **clipped** is never the plan's 0%: up to 0.7% of a step's tokens are far
  enough outside 0.8 to 1.2, in the direction their advantage pushes, for
  the clip to switch them off. Nothing is stale here; it is rounding.
- **entropy** stays between 0.61 and 1.22 nats per token: in 20 steps at
  this learning rate there is no slide toward collapse.
- **peak** memory, 6.36 GB, is reached at step 3, whose rows are long: 273
  tokens per reply on average, one of them cut off at the 1,024-token cap
  and trained on.

### One step, reply by reply

```text
2. step 1 in detail: two questions, four replies each
  school-easy-018: right and wrong, trained
    reply  tokens  verdict  reward   advantage  why
        1      76  right    1.0962     +0.585  correct
        2      43  right    1.0978     +0.588  correct
        3     321  right    1.0839     +0.559  correct
        4      93  wrong    0.0000     -1.732  SQL error: near "WHERE": syntax error
  shop-hard-005: all wrong, no contrast: skipped
    reply  tokens  verdict  reward   advantage  why
        1     110  wrong    0.0000          -  the query returned nothing
        2     112  wrong    0.0000          -  SQL error: near "SELECT": syntax error
        3      44  wrong    0.0000          -  SQL error: no such column: mi.parking_id
        4     166  wrong    0.0000          -  the query returned nothing
  advantage: (reward - the group's mean) / (the group's spread + 0.0001); every token of
    a reply is pushed by its reply's advantage
```

In `school-easy-018` three replies were right and one wrong. Each right one
gets about +0.58 and the wrong one -1.732: with four replies, one wrong
among three right is pushed down three times as hard as each right one is
pushed up, so the four advantages add up to 0. The 321-token right reply
gets a little less (+0.559) than the 43-token one (+0.588): its token
bonus was smaller. `shop-hard-005` had no right reply; its four rewards are
all 0 and the group teaches nothing, so it is skipped.

### The ratio over the whole run

```text
3. the importance ratio, token by token, over the whole run
  18,474 tokens trained on in 20 updates: mean 0.9999, lowest 0.633, highest 1.632; the
    per-step means ran from 0.9980 to 1.0026
  exactly 1: 4.9%; within 0.99 to 1.01: 64.0%; outside the clip's 0.8 to 1.2: 0.5%;
    clipped: 0.2%
```

Over 18,474 trained tokens the mean ratio is 0.9999. Only 4.9% of the
tokens are exactly 1 and 64.0% are within 0.99 to 1.01; 0.5% are outside
the clip's range, and 0.2% were clipped (an outside token is clipped only
when its advantage pushes the same way it already moved). The project's
logs held only the per-step mean, so they could not have shown this; ours,
measured the same way, show that "ratio 1.000" is true of the average and
not of the tokens.

### The groups and the length of the replies

```text
4. the groups, and the replies' length
  40 groups: 29 right and wrong, 4 all right, 7 all wrong (always skipped: four rewards
    of 0)
  of the 4 all-right groups, 4 had contrast from the token bonus alone (their replies
    differed in length) and were trained on, the shorter replies pushed up and the
    longer ones down; 0 tied
  the hardest push down on a right reply: advantage -1.068, a 258-token reply to
    school-medium-005 (the spread divides even a small difference in bonus up to full
    size)
  160 replies: 76 right (47.5%); 2 cut off at the 1,024-token cap
  mean tokens per reply, step by step: 121 103 273 191 112 184 124 99 160 180 225 102
    198 146 78 81 143 188 137 70
  the first 5 steps against the last 5: mean tokens per reply 160 -> 124; right replies
    only 175 -> 85 (different questions each step, so read this with care)
```

- 29 of the 40 groups had right and wrong replies; 7 were all wrong and
  skipped.
- The 4 all-right groups were all trained, on length alone. The advantage
  divides each reward's distance from the mean by the group's spread, and
  when every reply is right the spread is tiny (a few thousandths of a
  reward), so a difference of a few tokens becomes a full-size push. The
  hardest push down on a right reply in the run was -1.068, on a 258-token
  reply whose only fault was being longer than its group's others.
- 2 of 160 replies reached the 1,024-token cap. Chapter 14's SFT adapter
  looped on 8 of 50 held-out questions when answering greedily; drawn at
  temperature 1, on training questions, it rarely does.
- Mean tokens per reply fell from 160 in the first 5 steps to 124 in the
  last 5, and among right replies from 175 to 85. Every step asks different
  questions, so this is a hint, not a measurement; the evaluation below
  asks the same 50 questions before and after.

```text
5. saved
  ~/.cache/mhrl-course/15-grpo-real-model/grpo.safetensors (39,139,440 bytes)
  ~/.cache/mhrl-course/15-grpo-real-model/grpo.metrics.jsonl (one line per step)
  ~/.cache/mhrl-course/15-grpo-real-model/grpo.rollouts.jsonl (every reply, graded)
  next: uv run eval_sql.py --adapter grpo.safetensors
```

### The project's runs (recorded, not run here)

```text
6. the project's runs (recorded, not run here)
  each step: 2 tasks x 4 rollouts through mini-swe-agent and Claude Code, several model
    calls each, then one update; ratio: the mean over every token trained on
  step  attempt 1: right  ratio   attempt 2: right  ratio
     1            37.5%  0.9957            37.5%  0.9996
     2            37.5%  0.9971            25.0%  1.0009
     3            62.5%  0.9963            62.5%  0.9987
     4            50.0%  1.0000            75.0%  1.0007
     5            25.0%  0.9952            50.0%  0.9997
     6            25.0%  0.9973            25.0%  1.0011
     7            12.5%  0.9980
     8            12.5%  0.9988
  attempt 1: ratio 0.995 to 1.000 (the engine's log-probabilities in bfloat16); peak
    4.36 GB
  attempt 2: ratio 0.999 to 1.001 (in float32); peak 6.88 GB
  attempt 1 also paid its bonus for fewer calls and trained at 4e-05 without bias
    correction (chapter 26 replays it); attempt 2 used this chapter's settings: the
    token bonus, 1e-05 with bias correction
```

The project ran the same step, with real agent programs between the model
and the grader, so each rollout was several model calls. Its first attempt recorded bfloat16
log-probabilities, and 7 of its 8 step means are below 1 (the other prints
as 1.0000); its second recorded float32, and its means sit on both sides of 1,
as ours do. Its per-step accuracy swung just as ours does: 25% to 75% over
6 steps in attempt 2. (Attempt 1 also used the article's bonus for fewer
calls, a learning rate four times higher and no bias correction, and went
wrong in a way chapter 26 replays.)

## Exact tokens in float32: the ratio at 1, token by token (--float32)

```bash
uv run grpo_sql.py --float32 --steps 4
```

`same_tokens.py` showed that in float32 the sampler's and the trainer's
numbers agree to 0.00005. `--float32` casts every 16-bit number of the
model to float32 before training starts (the 4-bit codes stay) and runs the
same loop, so every sampled reply and every update happen in float32:

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak
     1    1/8    0.14     316        1/2  1.0000  1.000-1.000       0.0%    0.804  3.69 GB
     2    3/8    0.41     111        0/2  1.0000  1.000-1.000       0.0%    0.695  5.62 GB
     3    2/8    0.27     159        0/2  1.0000  1.000-1.000       0.0%    0.828  5.68 GB
     4    5/8    0.68     188        0/2  1.0000  1.000-1.000       0.0%    0.795  6.20 GB
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  varies from run to run: the time (4 steps took 143 s, 119 s of it sampling) and the
    peak memory (6.20 GB, the weights included)
```

```text
3. the importance ratio, token by token, over the whole run
  3,911 tokens trained on in 4 updates: mean 1.0000, lowest 1.000, highest 1.000; the
    per-step means ran from 1.0000 to 1.0000
  exactly 1: 50.7%; within 0.99 to 1.01: 100.0%; outside the clip's 0.8 to 1.2: 0.0%;
    clipped: 0.0%
```

Every one of the 3,911 trained tokens has a ratio that prints as 1.000;
half of them (50.7%) are exactly 1, all are within 0.99 to 1.01, and the
clip never acts. With the same tokens, the same weights and arithmetic
precise enough, the trainer gets the sampler's numbers back: the scatter in
the default run was bfloat16's, not GRPO's.

It is also a different model, as `same_tokens.py`'s last line warned: from
the first step it draws other replies (1 of 8 right at step 1, where the
default run had 3 of 8), so a float32 run is not the bfloat16 run made
exact, it is another run. The project's second attempt, which recorded
float32 log-probabilities from the model as it runs, saw step means of
0.999 to 1.001 (section 6 of the default run).

## Re-tokenizing: training on text instead of tokens (--retokenize)

```bash
uv run grpo_sql.py --retokenize --steps 4
```

A pipeline that keeps only text has to tokenize each reply again before it
can train on it. `--retokenize` does that: each reply is decoded to text,
the text is encoded again, and the result is what the trainer sees. The
prompt's ids are kept exactly, so only the reply's tokens can change (the
template's own surprises when a finished chat is rendered again were
chapter 14's). A re-encoded token has no recorded log-probability of its
own, so the two replies are lined up position by position, as the plan's
build note asks: where the ids agree, the sampled token's recorded
log-probability is kept; where they differ, or one reply is longer, the
position is masked out of training and counted.

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak  changed  masked
     1    3/8    0.41     121        1/2  0.9961  0.779-1.204       0.5%    0.762  3.58 GB      1/8     316
     2    2/8    0.27     130        1/2  0.9921  0.004-2.010       1.5%    0.937  3.58 GB      1/8       2
     3    5/8    0.69      48        0/2  1.0010  0.760-1.264       0.5%    0.729  3.58 GB      0/8       0
     4    3/8    0.41      77        1/2  1.0029  0.892-1.483       0.0%    0.839  3.58 GB      0/8       0
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  changed: replies whose re-encoded ids differ from the sampled ones; masked: their
    positions left out of training
  varies from run to run: the time (4 steps took 72 s, 61 s of it sampling) and the peak
    memory (3.58 GB, the weights included)
```

```text
3. the importance ratio, token by token, over the whole run
  1,229 tokens trained on in 4 updates: mean 0.9976, lowest 0.004, highest 2.010; the
    per-step means ran from 0.9921 to 1.0029
  exactly 1: 5.6%; within 0.99 to 1.01: 68.2%; outside the clip's 0.8 to 1.2: 1.6%;
    clipped: 0.7%
  --retokenize: 2 of 32 replies came back as other token ids; 318 of their positions
    were masked out (10.6% of all 3,001 sampled tokens)
  trained tokens after a difference, whose recorded log-probability belongs to another
    context: 88, mean 0.9572, lowest 0.004, highest 1.780
  every other trained token: 1,141, mean 1.0007, lowest 0.732, highest 2.010
  the masked positions, had they been kept with the log-probability recorded at the same
    position (scored, never trained on): 318, mean 47.8092, lowest 0.000, highest
    13615.509
```

- **2 of 32 replies came back as other token ids**: the same text, decoded
  and encoded again, is not the same tokens (chapter 9's `answer.txt`
  surprise, on the model's own replies). The other 30 came back unchanged.
- **One change can cost a whole reply.** At step 1, 316 positions were
  masked. Step 1 drew the same 8 replies as the default run's (the run log
  checks it), and by their lengths in "step 1 in detail" above, that is
  the 321-token right reply: once one token splits or merges differently,
  every later token
  sits one place off, no position lines up, and the trainer loses most of
  a reply it should have learned from. At step 2 the other change masked
  only 2 positions and the rest lined up again.
- **Tokens after a difference are trained against the wrong number.** The
  88 trained tokens that come after a difference keep their own ids, but
  their recorded log-probability was worked out after a different prefix.
  Their ratio averages 0.9572 and reaches 0.004: the trainer finds one of
  them about 250 times less likely than the sampler recorded. That token is
  in step 2, whose mean, 0.9921, is below every step of the default run
  (0.9980 at its lowest). (The 2.010 is not re-tokenizing's: it is among
  the tokens re-encoding did not touch, the bfloat16 scatter at its widest
  in these runs.)
- **Without the mask it would be far worse.** A pipeline that kept every
  position would compare each re-encoded token with the log-probability
  recorded for whatever token sat there. Scored, never trained on, those
  318 positions give ratios averaging 47.8 and reaching 13,615.5: numbers
  about other tokens entirely.

So the mask keeps re-tokenizing from doing damage, at the price of throwing
data away, and it cannot repair the tokens after a difference. Token in,
token out has none of these problems: every token of a trained reply is
trained on, against its own recorded number.

> **From the article, not from our runs** (section "Why rewards are not
> enough"): "Different token sequences can produce the same text. If you
> save only the text and tokenize it again, you can get different token IDs
> from those the model sampled. TRL's write-up puts it directly: 'in RL, you
> optimize on the exact tokens the model produced' (Gallouédec & Rasul,
> 2026)."

## bfloat16 log-probabilities, as the project's first RL run (--logprob-dtype bf16)

```bash
uv run grpo_sql.py --logprob-dtype bf16 --steps 4
```

The project's engine recorded bfloat16 log-probabilities in its first RL
run and float32 ones after a code review (its README, "Reproducing it
exactly"). `--logprob-dtype bf16` reproduces that first path: the scores
are not cast, so mlx-lm's `generate_step` works out
`logits - mx.logsumexp(logits)` in bfloat16 (its own line, in mlx-lm
0.32.0), the token is drawn from that, and that bfloat16 number is recorded.
The trainer casts its scores to float32 before the log-softmax, as always.
The plan's build note asks to
confirm the sign before narrating it, so the sampler also keeps the float32
number for every same token, and the run reports what the bfloat16
recording moved:

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak
     1    3/8    0.41     203        1/2  0.9979  0.772-1.427       0.4%    0.965  5.26 GB
     2    2/8    0.27     208        1/2  0.9990  0.715-1.347       0.6%    0.963  6.71 GB
     3    5/8    0.69      54        0/2  0.9976  0.710-1.232       0.9%    0.720  6.71 GB
     4    5/8    0.68     179        0/2  0.9971  0.787-1.312       0.5%    0.912  6.71 GB
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  varies from run to run: the time (4 steps took 115 s, 95 s of it sampling) and the
    peak memory (6.71 GB, the weights included)
```

```text
3. the importance ratio, token by token, over the whole run
  3,886 tokens trained on in 4 updates: mean 0.9979, lowest 0.710, highest 1.427; the
    per-step means ran from 0.9971 to 0.9990
  exactly 1: 0.5%; within 0.99 to 1.01: 41.3%; outside the clip's 0.8 to 1.2: 0.8%;
    clipped: 0.6%
  the same tokens, had the sampler recorded the engine's number (float32, the full
    distribution) instead of bfloat16 numbers: mean 1.0001, lowest 0.731, highest 1.408
  the recorded log-probability was above the engine's at 70.8% of these tokens, below at
    28.8% (by at most 0.124298), equal at 0.4%; on average +0.00260 per token
```

The sign is **below 1**, as the plan said. All four step means are under 1
(0.9971 to 0.9990, 0.9979 over the run), as 7 of the project's first run's
8 were. The last two lines say why. At 70.8% of the tokens the bfloat16
log-probability the sampler recorded was above the float32 one for the
same token, by +0.00260 per token on average. The ratio is new over
recorded, so a recorded number that is too high pulls every ratio down:
exp(-0.0026) = 0.9974, about the gap between 0.9979 and the 1.0001 the
same tokens give against the engine's float32 numbers. Recording in
bfloat16 moved the mean; the spread (0.710 to 1.427) is the routes'
scatter, as in the default run.

The run also draws other replies from the first step (203 tokens per reply
at step 1, against the default's 121): a token drawn from bfloat16
log-probabilities is drawn from a slightly different distribution.

## Top-k 50: drawing from fewer tokens (--top-k 50)

```bash
uv run grpo_sql.py --top-k 50 --steps 4
```

`--top-k 50` draws every token from the 50 likeliest only, renormalized,
and records the log-probability under that cut distribution, as an engine
that cuts before it reports does (chapter 12: the model's own
`generation_config.json` suggests top-k 50). The trainer knows nothing of
the cut: it works out the full distribution's number. Cutting divides each
kept token's probability by the share the 50 held, so each recorded number
is higher than the full one, and the plan's build note expects the mean
ratio to be that share:

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak
     1    4/8    0.55     158        1/2  0.9905  0.764-1.213       0.5%    0.818  4.39 GB
     2    2/8    0.27     182        1/2  0.9982  0.885-1.211       0.0%    0.757  4.39 GB
     3    5/8    0.68      94        0/2  0.9922  0.524-1.310       0.4%    0.661  4.39 GB
     4    4/8    0.55      97        0/2  0.9913  0.685-1.347       0.4%    0.730  4.39 GB
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  varies from run to run: the time (4 steps took 98 s, 82 s of it sampling) and the peak
    memory (4.39 GB, the weights included)
```

```text
3. the importance ratio, token by token, over the whole run
  2,547 tokens trained on in 4 updates: mean 0.9921, lowest 0.524, highest 1.347; the
    per-step means ran from 0.9905 to 0.9982
  exactly 1: 0.1%; within 0.99 to 1.01: 62.5%; outside the clip's 0.8 to 1.2: 0.9%;
    clipped: 0.4%
  the same tokens, had the sampler recorded the engine's number (float32, the full
    distribution) instead of top-50 numbers: mean 1.0007, lowest 0.781, highest 1.362
  the recorded log-probability was above the engine's at 97.8% of these tokens, below at
    2.2% (by at most 0.000002), equal at 0.0%; on average +0.00894 per token
  the share of the probability the 50 kept tokens held, at the trained tokens' steps:
    mean 0.9914, lowest 0.563, highest 1.000; each token's ratio is its share times the
    ratio the engine's own number would give
  entropy per sampled token: 0.7529 nats as drawn, 0.8211 over the full distribution at
    the same steps
```

- **The ratio sits below 1**: 0.9921 over the run, every step's mean
  between 0.9905 and 0.9982. The 50 kept tokens held 0.9914 of the
  probability on average at the trained tokens' steps, so the mean ratio
  is close to the kept share, as the plan's build note expected, but not
  equal to it: each token's ratio is its share times the ratio the engine's
  own number gives, and those average 1.0007, the usual bfloat16 scatter.
- **The recorded number was too high at 97.8% of the tokens**, because
  cutting the tail and renormalizing divides every kept probability by the
  kept share. The 2.2% below are steps where the 50 held everything but a
  rounding error (below by at most 0.000002).
- **The entropy falls**: drawn from the cut distribution, the replies were
  sampled at 0.7529 nats per token, where the full distribution at the
  same steps had 0.8211: about 8% less spread per token, on average. Over
  4 steps that is what top-k does to the entropy; the column moves 0.818,
  0.757, 0.661, 0.730 on different questions each step, too few to read a
  trend. The article calls truncation a known route to entropy collapse
  (box below): the policy is trained on draws narrower than its own, so it
  narrows, and the next draws are narrower still.

The article's numbers, in the box below, are from its own runs with
vLLM's top-p cut: 0.985 to 0.993 with it, 0.9984 to 0.9999 without. Ours,
with top-k 50, are 0.9905 to 0.9982, and 0.9980 to 1.0026 in the default
run.

> **From the article, not from our runs** (Hugging Face and Liquid AI, "The
> ultimate guide to multi-harness RL", section "What gets recorded"): "It
> also samples from the full distribution, with no top_p or top_k.
> Truncating the distribution biases what the policy samples away from what
> it would sample on its own, which in RL is a known route to entropy
> collapse, and vLLM computes its processed logprobs after the truncation,
> so they would not match the policy either. In our runs, moving top_p to
> 1.0 took the importance ratio from 0.985–0.993 to 0.9984–0.9999."

## A penalty for drifting from the start (--kl 0.05)

```bash
uv run grpo_sql.py --kl 0.05 --steps 4
```

Many GRPO trainers add a KL penalty: before each update, every trained
token is also scored under the reference model (here chapter 14's adapter,
the start), and `0.05 x drift` is added to its loss. The project has none;
it keeps the model near its start with a small learning rate, few steps,
LoRA and (in chapter 26) a guard on reply length. One model stays in
memory: the start adapter's 39 MB of weights are swapped in to score the
reference and swapped out again, and the reference is scored by the
trainer's own route, so the drift at step 1, where the two adapters are
the same, is exactly 0:

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak  drift
     1    3/8    0.41     121        1/2  1.0014  0.779-1.390       0.6%    0.762  4.09 GB  0.00000
     2    2/8    0.27     103        1/2  1.0007  0.850-1.280       0.0%    0.741  4.09 GB  0.00041
     3    4/8    0.54     144        0/2  1.0018  0.716-1.243       0.1%    0.667  4.58 GB  0.00041
     4    4/8    0.55     103        0/2  0.9982  0.803-1.245       0.0%    0.995  4.58 GB  0.00069
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  drift: the mean, per trained token, of exp(ref - new) - (ref - new) - 1, ref being its
    log-probability under the start adapter
  varies from run to run: the time (4 steps took 100 s, 73 s of it sampling) and the
    peak memory (4.58 GB, the weights included)
```

```text
3. the importance ratio, token by token, over the whole run
  2,922 tokens trained on in 4 updates: mean 1.0005, lowest 0.716, highest 1.390; the
    per-step means ran from 0.9982 to 1.0018
  exactly 1: 7.5%; within 0.99 to 1.01: 67.0%; outside the clip's 0.8 to 1.2: 0.5%;
    clipped: 0.1%
  drift per trained token, step by step: 0.00000 0.00041 0.00041 0.00069
```

- **At step 1 the drift is 0.00000**: the reference and the adapter are the
  same weights, scored by the same route. The penalty also pushes nothing
  there (the drift's slope is 0 where the two agree), so step 1's update
  is the default run's: step 2 draws exactly the default run's step 2
  (2 of 8, 103 tokens, ratio 1.0007, 0.850-1.280, entropy 0.741; the run
  log compares the 16 replies of steps 1 and 2, and none differ).
- **From step 2 the adapter has moved**, the drift is 0.00041 per token, the
  penalty starts to push, and from step 3 the two runs draw different
  replies. By step 4 the drift is 0.00069, so the penalty adds 0.05 x
  0.00069 = 0.0000345 to each token's loss.
- **It changes nothing about the ratio**: 1.0005 over the run, 0.716 to
  1.390, the default run's scatter.

Four updates at a learning rate of 1e-5 move a LoRA adapter very little,
which is the project's argument for doing without the penalty: after four
updates the drift it charges for is 0.00069 per token.

## Base, SFT and SFT+GRPO on the held-out questions

```bash
uv run eval_sql.py
```

`eval_sql.py` asks chapter 14's 50 held-out questions (clinic and airline,
databases no training question comes from) three times: with the base model
(a fresh adapter, every B zero), with chapter 14's `sft.safetensors`, and
with the `grpo.safetensors` the 20 steps saved. Each reply is greedy, up to
1,024 tokens, and is graded as chapter 14 grades one: its SQL runs on a
read-only copy of the database, the result is written to `answer.txt` in
the question's own folder, and chapter 1's grader reads that file. Answers
are saved with a fingerprint of the adapter file, so a later run with the
same adapter reads them back instead of answering again.

```text
eval_sql.py · base, SFT and SFT+GRPO on the held-out questions
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
questions 50 held out, over airline and clinic: databases no training question comes
  from
replies   greedy, one per question, up to 1,024 tokens, to chapter 14's chat; each
  reply's SQL run and its result graded through answer.txt
base      a fresh adapter: every B zero, the base model; read from eval_base.jsonl (the
  same adapter bytes and cap)
sft       ~/.cache/mhrl-course/14-sft-real-model/sft.safetensors, chapter 14's; answered
  now, saved to eval_sft.jsonl; varies from run to run: the 50 replies took 193
  s
sft+grpo  ~/.cache/mhrl-course/15-grpo-real-model/grpo.safetensors, grpo_sql.py's;
  answered now, saved to eval_grpo.jsonl; varies from run to run: the 50 replies
  took 209 s
```

The base model's answers were read back, not asked again: they were saved by
an earlier run of the same command at the same code, and match chapter 14's
`eval_base.jsonl` on all 50 questions, reply for reply (the run log checks
it; this run's SFT answers match chapter 14's too). Every verdict:

```text
1. every question's verdict, and its reply's length in tokens
  task                 base      sft       sft+grpo  tokens:     base       sft  sft+grpo
  clinic-medium-001    right     right     right                  157        35        35
  airline-medium-001   right     right     right                  185        46        40
  clinic-easy-001      right     right     right                   66        28        28
  airline-easy-001     right     right     right                   65        29        29
  clinic-hard-001      right     right     wrong                  156        84        70
  airline-hard-001     right     right     right                  302        67        67
  clinic-medium-002    wrong     wrong     wrong                  112      1024      1024
  airline-medium-002   right     right     right                  200        52        50
  clinic-easy-002      right     right     right                   64        24        24
  airline-easy-002     right     right     right                   59        27        27
  clinic-hard-002      right     right     right                  367        87        87
  airline-hard-002     right     right     right                  309        73        60
  clinic-medium-003    right     wrong     wrong                  167        43        43
  airline-medium-003   right     right     right                  111        61        44
  clinic-easy-003      right     right     right                  159        43        43
  airline-easy-003     right     wrong     wrong                  180      1024      1024
  clinic-medium-004    right     right     right                  253        90        80
  airline-medium-004   right     right     right                  208        52        46
  clinic-hard-003      right     wrong     right                  449        80       187
  airline-hard-003     right     right     right                  415        79        67
  clinic-easy-004      right     right     right                  219        43        43
  airline-easy-004     right     right     right                  238        39        39
  clinic-medium-005    right     wrong     right                  549      1024        89
  airline-medium-005   right     wrong     right                  266        88        77
  clinic-hard-004      right     wrong     wrong                  359      1024      1024
  airline-hard-004     right     wrong     wrong                  649      1024       103
  clinic-medium-006    right     right     wrong                  379        64      1024
  airline-medium-006   right     right     right                  209        69        69
  clinic-easy-005      right     right     right                  289        59        59
  airline-easy-005     right     right     right                  202        32        34
  clinic-hard-005      wrong     wrong     wrong                  449        71        59
  airline-hard-005     right     wrong     wrong                  374      1024      1024
  clinic-medium-007    right     right     right                  223        84        89
  airline-medium-007   right     right     right                  244        76        72
  clinic-easy-006      right     wrong     wrong                  241      1024      1024
  airline-easy-006     right     right     right                  506        60        60
  clinic-medium-008    right     wrong     wrong                  193      1024      1024
  airline-medium-008   right     right     right                  321        74        70
  clinic-hard-006      wrong     right     wrong                  445        93      1024
  airline-hard-006     wrong     wrong     wrong                  375        67      1024
  clinic-easy-007      right     right     right                   67        28        28
  airline-easy-007     right     right     right                  129        28        28
  clinic-medium-009    wrong     wrong     wrong                  101        59        33
  airline-medium-009   right     right     right                  185        50        46
  clinic-hard-007      wrong     wrong     wrong                  299        60        60
  airline-hard-007     right     right     right                  322        69        69
  clinic-easy-008      right     right     right                   67        29        25
  airline-easy-008     right     right     right                   61        24        28
  clinic-medium-010    wrong     right     right                  326        79        71
  airline-medium-010   right     right     right                  317        52        50
```

```text
2. pass@1 on the 50 held-out questions, with error bars
  base      43 of 50 =  86.0% ± 4.9 points
  sft       35 of 50 =  70.0% ± 6.5 points
  sft+grpo  35 of 50 =  70.0% ± 6.5 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 50 questions like these were asked
```

GRPO's 20 steps left the score where SFT had put it: 35 of 50 before and 35
of 50 after, both 70.0% ± 6.5 points, still well below the base model's
86.0% ± 4.9 (chapter 14's drop, not repaired). The plan expected error bars
of about ±7 points; at 70% they are ±6.5, and ±4.9 at 86%.

```text
3. question by question
  sft against sft+grpo: gained 3, lost 3, right both times 32, wrong both times 12;
    exact p = 1.0000
    gained: clinic-hard-003, clinic-medium-005, airline-medium-005
    lost: clinic-hard-001, clinic-medium-006, clinic-hard-006
  base against sft+grpo: gained 1, lost 9, right both times 34, wrong both times 6;
    exact p = 0.0215
    gained: clinic-medium-010
    lost: clinic-hard-001, clinic-medium-003, airline-easy-003, clinic-hard-004,
      airline-hard-004, clinic-medium-006, airline-hard-005, clinic-easy-006,
      clinic-medium-008
  base against sft: gained 2, lost 10, right both times 33, wrong both times 5; exact p
    = 0.0386
    gained: clinic-hard-006, clinic-medium-010
    lost: clinic-medium-003, airline-easy-003, clinic-hard-003, clinic-medium-005,
      airline-medium-005, clinic-hard-004, airline-hard-004, airline-hard-005,
      clinic-easy-006, clinic-medium-008
  gained: wrong before, right after; lost: the reverse; exact p: the chance of a split
    at least this uneven among the questions that changed, if the two models were
    equally good (McNemar's exact test)
```

Under the equal scores, six questions changed, three each way (p =
1.0000). Read task by task:

- all three gains (`clinic-hard-003`, `clinic-medium-005`,
  `airline-medium-005`) are questions SFT had lost against the base model,
  now back;
- two losses (`clinic-hard-001`, `clinic-medium-006`) were right under both
  the base model and SFT; the third (`clinic-hard-006`) was SFT's one gain
  over the base model, now gone;
- against the base model, SFT+GRPO is 1 gained, 9 lost (p = 0.0215), close
  to SFT's 2 gained, 10 lost (p = 0.0386).

Three changes each way on 50 questions is no evidence that 20 steps made the
model better or worse here.

```text
4. reply length
  model      median  mean  longest  cut off at the cap
  base          230   252      649                   0
  sft            66   212    1,024                   8
  sft+grpo       60   231    1,024                   9
  on the 32 questions right under both sft and sft+grpo: median 52 -> 46 tokens, mean 53
    -> 50; shorter on 13, longer on 3, the same on 16
```

- **Right answers got shorter**, which is what the token bonus pays for: on
  the 32 questions right under both, the median reply went from 52 to 46
  tokens, shorter on 13, longer on 3, the same on 16.
- **The mean went up** (212 to 231 tokens) because of the replies that run
  to the cap. Chapter 14's SFT adapter looped on 8 questions; read row by
  row, GRPO kept 6 of those loops, ended 2 (`clinic-medium-005`, now an
  89-token right answer, and `airline-hard-004`, 103 tokens and still
  wrong), and started 3 new ones (`clinic-medium-006`, `clinic-hard-006`,
  `airline-hard-006`): 9 in all. Twenty steps on training questions, where
  only 2 of 160 sampled replies reached the cap, did little about loops on
  questions it never saw.

```text
5. questions whose verdict changed from sft to sft+grpo
  clinic-hard-003: How many different patients born before 1950 had at least one Neurology appointment?
    sft (80 tokens, wrong: the query returned nothing): "SELECT DISTINCT p.id FROM
      patients p JOIN appointments a ON a.patient_id = p.id WHERE strftime('%Y',
      a.appointment_date) < 1950<|im_end|>"
    sft+grpo (187 tokens, right): "SELECT COUNT(DISTINCT p.id) FROM patients p JOIN
      appointments a ON a.patient_id = p.id JOIN doctors d ON d.id = a.doctor_id WHERE
      d.specialty = 'Neurology' AND strftime('%Y', p.birth_date) < '1950'<|im_end|>"
  clinic-medium-005: What was the average fee for a General Practice appointment in 2024? Round to 2 decimal places.
    sft (1024 tokens, wrong: cut off by the length cap while still reasoning): '(no
      </think> in 1024 tokens: still reasoning)'
    sft+grpo (89 tokens, right): "SELECT ROUND(AVG(fee), 2) FROM appointments a JOIN
      doctors d ON a.doctor_id = d.id WHERE d.specialty = 'General Practice' AND
      strftime('%Y', a.appointment_date) = '2024'<|im_end|>"
  clinic-hard-001: Which doctor earned the most in fees from patients living in Seville?
    sft (84 tokens, right): "SELECT d.name FROM doctors d JOIN appointments a ON
      a.doctor_id = d.id JOIN patients p ON p.id = a.patient_id WHERE p.city = 'Seville'
      GROUP BY d.id, d.name ORDER BY SUM(a.fee) DESC LIMIT 1<|im_end|>"
    sft+grpo (70 tokens, wrong: expected 'Grace Torres', got 'Stefan Russo'): "SELECT
      d.name FROM doctors d JOIN appointments a ON a.doctor_id = d.id JOIN patients p ON
      p.id = a.patient_id WHERE p.city = 'Seville' ORDER BY a.fee DESC LIMIT
      1<|im_end|>"
  clinic-medium-006: How many appointments did patients from Malaga have in 2025?
    sft (64 tokens, right): "SELECT COUNT(*) FROM appointments a JOIN patients p ON
      a.patient_id = p.id WHERE p.city = 'Malaga' AND strftime('%Y', a.appointment_date)
      = '2025'<|im_end|>"
    sft+grpo (1024 tokens, wrong: cut off by the length cap while still reasoning): '(no
      </think> in 1024 tokens: still reasoning)'
```

`clinic-hard-001` shows a loss that is a real change in the SQL: after GRPO
the reply orders single appointments by fee instead of adding up each
doctor's fees. `clinic-medium-006` shows the other kind: a 64-token right
answer became a reply still reasoning at 1,024 tokens.

## Fast mode

```bash
uv run grpo_sql.py --fast
uv run eval_sql.py --fast --adapter grpo_steps4.safetensors
```

`grpo_sql.py --fast` runs 4 steps (the plan's `--steps 4`) on Liquid's
export instead of chapter 11's build, and saves `grpo_steps4.safetensors`.
`eval_sql.py --fast` asks 10 balanced held-out questions.

Liquid's export holds the same bytes as chapter 11's build, so with the
same seed and the same order of questions the fast run repeats the default
run's first 4 steps, reply for reply (the run log compares all 32 replies:
none differ; only the time and the peak memory move):

```text
1. training: sample 8 replies, grade them, one update
  step  right  reward  tokens  no-signal   ratio  per token      clipped  entropy  peak
     1    3/8    0.41     121        1/2  1.0014  0.779-1.390       0.6%    0.762  4.09 GB
     2    2/8    0.27     103        1/2  1.0007  0.850-1.280       0.0%    0.741  4.09 GB
     3    2/8    0.27     273        0/2  0.9990  0.754-1.262       0.1%    0.856  6.83 GB
     4    2/8    0.27     191        0/2  1.0009  0.707-1.632       0.3%    1.066  6.83 GB
  right: replies whose answer.txt chapter 1's grader accepted; reward: their mean;
    tokens: mean tokens per reply; no-signal: groups whose 4 rewards were equal, skipped
  ratio: the mean importance ratio over every token trained on (new probability /
    recorded probability, worked out before the update), as the project logs it; per
    token: the lowest and highest single token's
  clipped: the share of those tokens whose push the clip switched off; entropy: the mean
    entropy, per sampled token, of the distribution it was drawn from, in nats
  varies from run to run: the time (4 steps took 128 s, 106 s of it sampling) and the
    peak memory (6.83 GB, the weights included)
```

The base model's and SFT's answers to those 10 questions are already saved
from the full evaluation, so they are read back; only the new adapter
answers:

```text
2. pass@1 on the 10 held-out questions, with error bars
  base      10 of 10 = 100.0% ± 0.0 points
  sft       10 of 10 = 100.0% ± 0.0 points
  sft+grpo   9 of 10 =  90.0% ± 9.5 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 10 questions like these were asked
```

```text
3. question by question
  sft against sft+grpo: gained 0, lost 1, right both times 9, wrong both times 0; exact
    p = 1.0000
    lost: clinic-hard-001
  base against sft+grpo: gained 0, lost 1, right both times 9, wrong both times 0; exact
    p = 1.0000
    lost: clinic-hard-001
  base against sft: gained 0, lost 0, right both times 10, wrong both times 0; exact p =
    1.0000
  gained: wrong before, right after; lost: the reverse; exact p: the chance of a split
    at least this uneven among the questions that changed, if the two models were
    equally good (McNemar's exact test)
```

After 4 steps the one change is `clinic-hard-001`, lost with the same
mistake as after 20 (ordering single appointments by fee). Ten questions,
with an error bar of ±9.5 points, are a check that everything runs, not a
measurement.

## The answer to the puzzle

Not in bfloat16. The sampler's own route gives its recorded numbers back to
the last bit, but the trainer reads the same tokens by other routes, and in
bfloat16 other routes round differently. One reply's 76 tokens scattered
from 0.854 to 1.204 on the trainer's route; over the 20-step run, 18,474
trained tokens ran from 0.633 to 1.632, only 4.9% exactly 1, and the clip
switched off 0.2% of them, while every step's mean stayed between 0.9980
and 1.0026: the "ratio 1.000" a log of means shows. In float32 the answer
is yes: all 3,911 trained tokens of a 4-step run print 1.000.

The flags show what breaks the agreement for real. Re-tokenized text gave 2
of 32 replies other ids, masked 10.6% of the sampled tokens and left tokens
with a ratio as low as 0.004 (13,615.5 without the mask). Recording
bfloat16 log-probabilities pulled every step's mean below 1 (0.9979 over
the run), as in the project's first run. Drawing from the top 50 tokens
pulled it to 0.9921, close to the 0.9914 of probability the 50 kept, and
drew from distributions with about 8% less entropy. On the 50 held-out
questions, 20 steps of GRPO left SFT's 35 right at 35 (3 gained, 3 lost,
p = 1.0000), with right answers a little shorter.

## What the chapter showed, and where it differed from the plan

| the plan said | measured |
| --- | --- |
| each step: 2 training questions x 4 replies at temperature 1, chapter 12's sampler, the 1,024-token cap, every token's float32 log-probability kept as old_logprobs | as planned |
| reward = 1 + token bonus (up to 0.1, zero at 2,000 generated tokens), only if correct | as planned (the project's `token_bonus` and `reward_for`) |
| group_advantages, TrainItem and the clipped loss match grpo.py and common.py; one AdamW step per step (lr 1e-5, bias correction) on chapter 14's SFT adapter | as planned, line for line |
| sampler and trainer build the same adapter on the same base | here they are one model with one adapter, in one process |
| one line per step: correct %, mean reward, tokens per reply, no-signal groups, importance ratio (1.000), clipped share (0%), entropy, peak GB | every column printed. The step means are 0.9980 to 1.0026, but single tokens run from 0.633 to 1.632 and the clipped share is 0.0% to 0.7% a step (0.2% over the run), not 0%: bfloat16 routes round differently. Added: `same_tokens.py`, which shows it on one reply, and `--float32`, where every token is 1.000 and nothing is clipped |
| per-step accuracy swings because each step sees only 8 replies | as planned: from 1 of 8 to 8 of 8 |
| `--retokenize`: some rows differ (count printed) and the ratio drifts; align by position, mask mismatched tokens, print how many (build note) | 2 of 32 replies; 318 positions masked (10.6% of sampled tokens), 316 of them from one 321-token reply; the 88 trained tokens after a difference average 0.9572, down to 0.004; unmasked, the positions would have averaged 47.8 |
| `--logprob-dtype bf16`: the ratio settles just below 1; confirm the sign before narrating it (build note) | confirmed below 1: step means 0.9971 to 0.9990; the recorded bfloat16 number was above the float32 one at 70.8% of tokens, +0.0026 on average |
| the project's first RL run saw 0.995-1.000 before its engine switched to float32 | confirmed from its metrics file: 0.995 to 1.000 (attempt 2: 0.999 to 1.001) |
| `--top-k 50`: sampler and trainer disagree and the entropy falls; the mean ratio is the top-k mass (build note) | mean ratio 0.9921 against a mean kept share of 0.9914: close, not equal (each ratio is the share times the engine ratio, which averages 1.0007); 0.7529 nats per token as drawn against 0.8211 for the full distribution |
| the article measured 0.985-0.993 under top-p truncation and 0.9984-0.9999 without it | as quoted (the article's "What gets recorded"), in its own box |
| `--kl 0.05` adds a penalty for drifting from the starting model, which the project does without | as planned: drift 0 at step 1 (step 2 is the default run's, reply for reply), 0.00069 per token by step 4 |
| eval_sql.py: base, SFT and SFT+GRPO on the 50 held-out questions, an error bar of about ±7 points beside each score | 86.0% ± 4.9, 70.0% ± 6.5, 70.0% ± 6.5; SFT to SFT+GRPO 3 gained, 3 lost, p = 1.0000 |
| replies getting shorter under the token bonus | on the 32 questions right both times: median 52 to 46 tokens, shorter on 13, longer on 3. Over all 50 the mean rose (212 to 231): 9 replies ran to the cap, against SFT's 8 |
| fast mode: `--steps 4` | `--steps 4`; `--fast` is that on Liquid's export, and `eval_sql.py --fast` asks 10 questions |
| folder `course/15-grpo-real-model/` | `15-grpo-real-model/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its two recorded GRPO runs,
  printed by `grpo_sql.py` in a section headed "the project's runs
  (recorded, not run here)".
- **The article's numbers** are two quotes, each in a box headed "From the
  article, not from our runs".
- **A difference from the project**: here one question is one model call,
  so a rollout is one reply and a step takes minutes; the project's
  rollouts are several calls inside real agent programs. Chapter 26 trains
  the project's way.
- **Answers are reused** when they were made from the same adapter bytes
  under the same cap. Nothing checks the prompt or the model folder: change
  either and delete the `eval_*.jsonl` files to measure again (`--fast`
  reads the base and SFT answers back, which is right only because
  Liquid's export and chapter 11's build are the same bytes).

## What is in this folder

- `same_tokens.py`: one reply's exact tokens scored by the sampler's route,
  in one pass and inside a gradient computation, in bfloat16 and in float32.
- `grpo_sql.py`: the project's GRPO step on the real model, every step
  printed, the ratio read token by token, the project's recorded runs; the
  flags `--retokenize`, `--logprob-dtype bf16`, `--top-k`, `--kl`,
  `--float32`, `--steps`, `--fast`.
- `eval_sql.py`: greedy replies from the base model, chapter 14's SFT
  adapter and the GRPO adapter on the 50 held-out questions, graded through
  `answer.txt`; pass@1, error bars, gained, lost and p, reply lengths.
- `sampler.py`: chapter 12's sampler, keeping each token's id,
  log-probability and entropy, and the engine's float32 number beside it.
- `grpo.py`: the project's `token_logprobs` and clipped loss, the drift
  penalty, the trainer's route without its gradient, one update.
- `rewards.py`: the project's `token_bonus`, `reward_for` and
  `group_advantages`, and chapter 6's clip test.
- `retokenize.py`: decode, encode, and line the result up by position.
- `rows.py`: chapter 14's chat; the project's `TrainItem`,
  `item_from_sequence`, `balanced` and the training order.
- `project_runs.py`: the project's two recorded GRPO runs.
- `report.py`: how the scripts print.
- `grading.py`, `database.py`: chapter 14's grading through `answer.txt`,
  read-only SQL with ATTACH refused (and, new here, text SQLite cannot
  encode refused too).
- `lfm.py`, `lfm_files.py`, `lora.py`, `stats.py`, `tasks.py`, `grader.py`:
  loading the model, where files live and the project's settings, chapter
  13's adapter code, chapter 14's arithmetic, chapter 1's tasks and grader.
- `data/`: the 230 questions and five databases (chapter 14's), and the
  project's two GRPO metrics files (chapter 6's copies).

## Words from this chapter

- **GRPO on tokens**: a group of replies per question, rewards, advantages,
  and one clipped update on every sampled token.
- **Old and new log-probability**: the sampler's number for a token, and
  the trainer's for the same token before the update.
- **Importance ratio**: exp(new - old), per token; its mean is what the
  project logs.
- **Route**: the order of operations a number is computed by; in bfloat16,
  two routes round differently.
- **Train-inference mismatch**: the sampler's and the trainer's numbers for
  the same token disagreeing.
- **Token in, token out**: train on the sampled ids and their recorded
  log-probabilities, never on text tokenized again.
- **Re-tokenizing**: decoding a reply and encoding it again; it can give
  other ids.
- **Entropy collapse**: the sampling distribution narrowing until a group
  has nothing to compare.
- **KL penalty, reference model**: a cost for drifting from the adapter
  training started from.
- **No-signal group**: four equal rewards, skipped.

Next: chapter 16, What a harness is: Claude Code from the inside.
