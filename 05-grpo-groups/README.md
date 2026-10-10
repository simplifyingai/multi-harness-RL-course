# Chapter 5. GRPO, part 1: compare attempts within a group

Chapter 4's agent learned from one rollout at a time. After each rollout it
compared the reward with a baseline, the average reward that kind of question
had earned so far, and took one step. That average came from earlier rollouts,
made with older weights, about other cities.

The project trains its model with **GRPO** (Group Relative Policy
Optimization), which compares differently. It tries the same question 4 times,
with the same weights, and scores each rollout against the other three. This
chapter puts GRPO on chapter 4's tiny agent, and finds out what that means in
practice. A group whose four rollouts all score the same teaches nothing, so the
questions worth practising are the ones the agent solves only sometimes.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/05-grpo-groups
uv run grpo_groups.py --trace
```

It takes about half a minute on any Mac (ten seconds with `--fast`, which
skips the pauses). It needs NumPy, which `uv run` installs for this script
alone, and like every chapter it writes only under `COURSE_HOME` (default
`~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `grpo_groups.py --group 1,0,1,1` | one group's advantages, worked out step by step |
| `grpo_groups.py` | 200 steps of GRPO, one line per step, then a table of the groups by difficulty |
| `grpo_groups.py --trace` | the same, plus three groups printed in full |
| `grpo_groups.py --lr 0` | the same 200 steps with learning switched off |
| `plot.py <metrics.jsonl>` | the run's curves, redrawn from its log alone |
| `seeds.py` | seeds 0 to 9, to check that the table is not one lucky run (under a minute) |
| `seeds.py --shared` | the same, with one set of weights for every difficulty |

Other flags: `--seed N` for a different run, `--steps N` for the number of
steps, `--lr N` for the learning rate, `--shared` (see below) and `--fast` to
skip the pauses. `seeds.py` takes `--seeds N`, `--steps N`, `--lr N` and
`--shared`.

## What is new since chapter 4

Everything is chapter 4's MiniAnalyst, copied in, except three things.

**Questions at three difficulties.** The project labels each of its questions
easy, medium or hard by what the SQL needs, not by how the sentence reads. This
chapter does the same, with six questions, two per level:

| level | what the SQL needs | the two questions |
| --- | --- | --- |
| easy | one table, a filter, one aggregate | How many customers live in {city}? When did the newest customer from {city} sign up? |
| medium | a two-table join with a filter | How many orders did customers from {city} place? When was the first order from {city}? |
| hard | three or more tables (here all four) | How much did customers from {city} spend on electronics? How many electronics items did customers from {city} buy? |

As in chapter 4, `QUERY_A` runs the right query, `QUERY_B` and `QUERY_C` run
two plausible mistakes, and a question is kept only when no mistake happens to
give the right answer too. Eight cities are for training and four are held out
for the final test.

**One table per LOOK.** In chapter 4 one `LOOK` showed every table. Now each
`LOOK` shows one more table and its columns, following the links between them:
`customers`, then `orders` (each order names its customer), then `order_items`
(each line names its order), then `products`. A query can only use tables the
agent has seen; before that, SQLite answers `no such table: orders`. So an easy
question needs one `LOOK` before its query works, a medium one needs two, and a
hard one needs four. On a hard question, six of the eight calls have to be the
right ones: four looks, the right query, and a write.

**One set of weights per level.** The agent can tell a question's level (a
model reading the question can too). It keeps a separate copy of chapter 4's 30
weights for each level, 90 in all, so what it learns on easy questions never
changes its hard weights. A real model shares all of its weights across all
questions; [One set of weights for everything](#one-set-of-weights-for-everything)
runs the toy that way too.

Every run starts by saying all of this:

```text
grpo_groups · seed 0 · lr 3 · 200 steps · one set of weights per level
each step: 2 questions x 4 rollouts at temperature 1, then one update
questions: 45 to train on (8 cities) + 19 held out (4 other cities)
  easy    15 + 8  one table, a filter, one aggregate  LOOK once first
  medium  16 + 7  a two-table join with a filter      LOOK twice first
  hard    14 + 4  three or more tables (here four)    LOOK four times first
advantage = (reward - group mean) / (group spread + 0.0001); a group whose rewards all tie is skipped
```

The two numbers after each level count its training and held-out questions.
The last line is GRPO's rule for advantages, which the next section takes apart.

## A group, and what it compares

A **group** is several rollouts of the same question, all made with the same
weights, one after the other, before anything is learned. Here, as in the
project, a group has 4 rollouts. Each picks its actions at temperature 1: by
their probabilities, as in chapter 2, so the four can go differently.

GRPO's baseline is the group's own average. Each rollout's advantage says how
much better or worse it did than the other rollouts of the same question:

```bash
uv run grpo_groups.py --group 1,0,1,1
```

```text
rewards     [1, 0, 1, 1]
mean        0.75
spread      0.43  (standard deviation)
advantages  (reward - mean) / (spread + 0.0001)
            [+0.58, -1.73, +0.58, +0.58]
```

Three of the four were right, so the group's **mean** (its average) is 0.75: on
this question, with these weights, that is what a rollout can expect to earn.
Each right rollout did 0.25 better than that and the wrong one 0.75 worse. GRPO
then divides by the group's **spread**, the standard deviation from chapter 3
(how far the rewards scatter around their mean, here 0.43). That turns +0.25
and -0.75 into +0.58 and -1.73. The 0.0001 keeps the division safe when the
spread is tiny.

Dividing by the spread makes every group push about as hard as every other,
however its rewards are spread out. One right rollout in four gets +1.73, and
one wrong rollout in four gets -1.73: the rare result gets the big push, in
either direction. A group's advantages always add up to 0, so whatever one
rollout's actions gain, the others' lose.

```bash
uv run grpo_groups.py --group 1,1,1,1
```

```text
rewards     [1, 1, 1, 1]
no contrast, skipped: every rollout scored the same, so none did
better or worse than the others, and the group teaches nothing
```

When all four rewards are the same, every rollout did exactly as well as the
mean, so every advantage would be 0, and 0 times any gradient is 0. The group
has no **contrast**, no difference between its rollouts to learn from. GRPO
skips it, and this chapter calls it a **no-signal group**. Four rollouts that
all fail tie in the same way.

The code is the project's function line for line (in `grpo.py`, docstring left
out here; a test checks it against `mhrl/training/grpo.py`):

```python
def group_advantages(rewards: list[float]) -> list[float] | None:
    if len(rewards) < 2 or max(rewards) - min(rewards) < Defaults.ADVANTAGE_EPS:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + Defaults.ADVANTAGE_EPS) for r in rewards]
```

`Defaults.ADVANTAGE_EPS` is 0.0001: rewards closer than that count as a tie,
and it is the number added to the spread. `pstdev` is the standard deviation.

## A step

A **step** is still one update of the weights. GRPO builds each one the way the
project builds its own (`mhrl/training/grpo.py`):

1. take the next 2 training questions (they come in one shuffled order, over
   and over);
2. run a group of 4 rollouts for each, every rollout in its own new folder, all
   with the same weights;
3. turn each group's rewards into advantages, and skip the groups with no
   contrast;
4. take one gradient-descent step on chapter 3's loss, `-advantage x log p`,
   over every call of every rollout that is left, averaged over those calls (the
   project averages over every token its model wrote). Every call shares its
   rollout's advantage, as in chapter 4.

So a step is 8 rollouts and one update, the same 2 x 4 as the project's real
runs. The learning rate is 3, far above chapter 4's 0.25, because the loss is
now an average over calls instead of a sum.

The first group of the run, as `--trace` prints it:

```text
step 1 · group 1 of 2 · easy · How many customers live in Berlin?
  rollout 1  reward 0  answer.txt (none)      FINISH
  rollout 2  reward 0  answer.txt (none)      QUERY_C, WRITE, FINISH
  rollout 3  reward 1  answer.txt 15          LOOK, QUERY_A, WRITE, LOOK, WRITE, QUERY_C, LOOK x2
  rollout 4  reward 0  answer.txt (none)      LOOK, WRITE, FINISH
  rewards [0, 0, 1, 0] · mean 0.25 · spread 0.43
  advantages = (reward - mean) / (spread + 0.0001) = [-0.58, -0.58, +1.73, -0.58]
```

Rollout 3 looked, ran the right query and wrote 15, Berlin's number of
customers. It earned the only reward in the group, so all eight of its calls get
+1.73, and every call of the other three gets -0.58.

Then comes one line per step: the average reward, the share of rollouts right,
the calls per rollout, how many of the step's two groups had no signal, and how
many of each group's 4 rollouts were right. (The reward here is 1 for right and
0 for wrong, so the reward and the share right are the same number. Chapter 7
adds bonuses that pull them apart.)

```text
step   1 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · easy 1/4, hard 0/4
step   2 · reward 0.00 · correct 0% · calls 6.6 · no-signal groups 2/2 · easy 0/4, easy 0/4
step   3 · reward 0.00 · correct 0% · calls 4.1 · no-signal groups 2/2 · hard 0/4, medium 0/4
step   4 · reward 0.12 · correct 12% · calls 6.2 · no-signal groups 1/2 · easy 0/4, easy 1/4
step   5 · reward 0.00 · correct 0% · calls 3.6 · no-signal groups 2/2 · hard 0/4, hard 0/4
step   6 · reward 0.00 · correct 0% · calls 4.6 · no-signal groups 2/2 · medium 0/4, medium 0/4
step   7 · reward 0.12 · correct 12% · calls 6.8 · no-signal groups 1/2 · easy 0/4, easy 1/4
step   8 · reward 0.12 · correct 12% · calls 5.4 · no-signal groups 1/2 · easy 1/4, medium 0/4
step   9 · reward 0.25 · correct 25% · calls 6.5 · no-signal groups 1/2 · medium 0/4, easy 2/4
step  10 · reward 0.00 · correct 0% · calls 4.1 · no-signal groups 2/2 · hard 0/4, medium 0/4
step  11 · reward 0.00 · correct 0% · calls 6.8 · no-signal groups 2/2 · easy 0/4, hard 0/4
step  12 · reward 0.00 · correct 0% · calls 3.4 · no-signal groups 2/2 · medium 0/4, hard 0/4
step  13 · reward 0.00 · correct 0% · calls 5.1 · no-signal groups 2/2 · medium 0/4, hard 0/4
step  14 · reward 0.00 · correct 0% · calls 4.4 · no-signal groups 2/2 · hard 0/4, hard 0/4
step  15 · reward 0.25 · correct 25% · calls 6.2 · no-signal groups 1/2 · easy 2/4, hard 0/4
step  16 · reward 0.25 · correct 25% · calls 6.0 · no-signal groups 1/2 · easy 2/4, medium 0/4
step  17 · reward 0.12 · correct 12% · calls 5.0 · no-signal groups 1/2 · easy 1/4, hard 0/4
step  18 · reward 0.25 · correct 25% · calls 6.0 · no-signal groups 1/2 · easy 2/4, medium 0/4
step  19 · reward 0.00 · correct 0% · calls 2.9 · no-signal groups 2/2 · medium 0/4, medium 0/4
step  20 · reward 0.12 · correct 12% · calls 5.4 · no-signal groups 1/2 · medium 1/4, hard 0/4
step  21 · reward 0.12 · correct 12% · calls 4.6 · no-signal groups 1/2 · hard 0/4, medium 1/4
```

Look at the no-signal column. The untrained agent rarely gets anything right, so
most groups are four wrong rollouts: 10 of these 21 steps had no group with
signal at all, and learned nothing. Of the 11 groups that did have signal, 9
were easy questions, the only ones an untrained agent gets right now and then.

Step 22 brings the first group with exactly one wrong rollout, the `--group
1,0,1,1` pattern in a real run:

```text
step 22 · group 1 of 2 · easy · How many customers live in Copenhagen?
  rollout 1  reward 1  answer.txt 12          LOOK, QUERY_C, QUERY_A, LOOK, QUERY_A, LOOK, WRITE x2
  rollout 2  reward 1  answer.txt 12          QUERY_A, LOOK, QUERY_A, LOOK, QUERY_A x2, WRITE, QUERY_A
  rollout 3  reward 1  answer.txt 12          QUERY_A, LOOK, WRITE, QUERY_A, WRITE, QUERY_A, WRITE, LOOK
  rollout 4  reward 0  answer.txt (none)      QUERY_A x2, QUERY_C, LOOK, QUERY_A x3, LOOK
  rewards [1, 1, 1, 0] · mean 0.75 · spread 0.43
  advantages = (reward - mean) / (spread + 0.0001) = [+0.58, +0.58, +0.58, -1.73]
```

Rollout 4 ran the right query five times (the first two failed, before it had
looked) but never wrote the answer down, so it gets -1.73. And at step 29 come the first four rollouts that were all right:

```text
step 29 · group 2 of 2 · easy · When did the newest customer from Vienna sign up?
  rollout 1  reward 1  answer.txt 2025-03-30  WRITE, QUERY_A, LOOK, QUERY_A x2, WRITE, LOOK x2
  rollout 2  reward 1  answer.txt 2025-03-30  QUERY_A, WRITE, LOOK, QUERY_A, WRITE, QUERY_A, LOOK x2
  rollout 3  reward 1  answer.txt 2025-03-30  LOOK x2, QUERY_A, WRITE, QUERY_A, LOOK, WRITE, LOOK
  rollout 4  reward 1  answer.txt 2025-03-30  WRITE, LOOK x3, WRITE, LOOK, QUERY_A, WRITE
  rewards [1, 1, 1, 1] · no contrast, skipped
```

All four answered right, in four different ways, and none of them is reinforced.
Easy questions are starting to be too easy to teach anything.

## Which groups teach

At the end of the run comes a table by difficulty, for each quarter of the run
(50 steps each):

```text
by difficulty, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
rollouts right
easy    45%         88%       95%       95%
medium  11%         50%       82%       95%
hard    0%          0%        0%        0%
no-signal groups (all 4 rewards tied, so skipped)
easy    8/35        18/34     25/32     27/33
medium  21/33       13/37     15/34     29/37
hard    32/32       29/29     34/34     30/30
of those, all wrong + all right
easy    6 + 2       0 + 18    0 + 25    0 + 27
medium  21 + 0      6 + 7     0 + 15    0 + 29
hard    32 + 0      29 + 0    34 + 0    30 + 0
```

Read it one difficulty at a time:

- **Easy** questions carry the signal first. In steps 1 to 50 the agent got 45%
  of their rollouts right, and only 8 of their 35 groups were silent. From then
  on it gets them right 88% to 95% of the time, and they fall silent the other
  way: 25 of 32 easy groups in steps 101 to 150 were all right, and taught
  nothing.
- **Medium** questions are all wrong at first (21 silent groups in steps 1 to
  50, every one of them all wrong). They carry the signal in the middle of the
  run, while the agent gets them right only sometimes: 50% in steps 51 to 100,
  when only 13 of their 37 groups were silent. By steps 151 to 200 the agent
  gets them right 95% of the time, and 29 of 37 groups are all right.
- **Hard** questions never once came out right: not one of their 500 rollouts.
  All 125 of their groups were all wrong, so every one was skipped, and the hard
  weights never changed.

The greedy test at the end shows the result:

```text
greedy test at the end (always the most likely action)
  easy    training 15 of 15  held out 8 of 8 · LOOK, QUERY_A, WRITE x6
  medium  training 16 of 16  held out 7 of 7 · LOOK x3, QUERY_A, WRITE, LOOK x3
  hard    training  0 of 14  held out 0 of 4 · LOOK x8
          never trained: every weight still 0, so greedy takes the first
```

Easy and medium questions are solved, including on the held-out cities. The
hard weights are still exactly 0 after 200 steps. All six actions still have
equal scores, and greedy, which takes the first of equal scores, looks eight
times and never queries.

This is the chapter's point, and the arithmetic behind it is short. If a rollout
is right with chance p, a group of 4 ties when all four are right (p x p x p x
p) or all four are wrong ((1 - p) four times over). Everything else has signal:

| chance a rollout is right | chance a group of 4 has signal |
| ---: | ---: |
| 0.5 | 88% |
| 0.1 or 0.9 | 34% |
| 0.01 or 0.99 | 4% |
| 0 or 1 | 0% |

The questions an agent solves about half the time teach it the most. The ones it
always solves have nothing left to teach, and the ones it never solves cannot
teach it anything, however long it trains.

### With learning switched off

```bash
uv run grpo_groups.py --lr 0
```

```text
by difficulty, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
rollouts right
easy    5%          6%        4%        6%
medium  2%          1%        2%        3%
hard    0%          0%        0%        0%
no-signal groups (all 4 rewards tied, so skipped)
easy    28/35       27/34     27/32     25/33
medium  30/33       36/37     31/34     33/37
hard    32/32       29/29     34/34     30/30
```

With every weight left at 0 the agent picks each action with a chance of 1 in 6
for all 200 steps. That is the agent GRPO starts from. By luck it gets 4% to 6%
of easy rollouts right and 1% to 3% of medium ones, and no hard rollout at all,
since a hard question needs six exact calls out of eight. Training can only
build on the lucky successes, so the easy questions take off first.

### Not one lucky seed

```bash
uv run seeds.py
```

```text
seeds 0-9 · lr 3 · 200 steps · one set of weights per level
groups with signal, in each quarter of the run
seed    easy           medium         hard
 0       27 16  7  6    12 24 19  8     0  0  0  0
 1       24 17  6  4     0  2 27 18     0  0  0  1
 2       23 12  9  6     0  0  3 29     0  0  0  0
 3       31 15  7  3     2 24 23  5     0  0  0  0
 4       22 18 10  5     1 13 21 11     0  0  0  2
 5       15 25 13  3     3 26 20 11     0  0  0  0
 6       23 18  5  5     3 27 19  8     0  0  0  0
 7       17 18 14  5     2 11 28 15     0  0  0  0
 8       19 16  8  5     6 22 20  9     0  0  0  0
 9       11 19 13  7     6 26 15  7     0  0  0  0

all 10 seeds together

by difficulty, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
rollouts right
easy    37%         80%       92%       96%
medium  3%          27%       66%       87%
hard    0%          0%        0%        0%
no-signal groups (all 4 rewards tied, so skipped)
easy    123/335     152/326   240/332   291/340
medium  313/348     188/363   160/355   232/353
hard    317/317     311/311   313/313   304/307
of those, all wrong + all right
easy    98 + 25     4 + 148   0 + 240   0 + 291
medium  313 + 0     170 + 18  41 + 119  2 + 230
hard    317 + 0     311 + 0   313 + 0   304 + 0
```

Each row is one seed: how many groups of each level had signal in each quarter
of its run. In all ten runs the pattern is the same. Easy questions carry the
most signal in the first quarter (11 to 31 groups) and little by the last (3 to
7). Medium questions carry it
later: most in the second quarter for six seeds, in the third for three, and in
the last for seed 2, which took longest to get going. Hard questions had signal
in 3 of their 1,248 groups across all ten runs, all in the last quarter of
seeds 1 and 4: a lucky success can happen, but far too rarely for training to
build on.

Added up over the ten seeds, steps 101 to 150 show the chapter's claim most
cleanly. Easy questions were right 92% of the time and 240 of their 332 groups
were silent, every one all right. Medium questions were right 66% of the time,
and 195 of their 355 groups had signal. Hard questions were right 0% of the
time, and all 313 of their groups were silent, every one all wrong.

## One set of weights for everything

The toy keeps one set of weights per level, so nothing learned on easy questions
helps with hard ones. A real model is one set of weights for every question.
`--shared` gives the toy a single set of 30 weights for all three levels:

```bash
uv run seeds.py --shared
```

```text
seeds 0-9 · lr 3 · 200 steps · one set of weights for every level
groups with signal, in each quarter of the run
seed    easy           medium         hard
 0       19  8  3  4    18 12  2  2     8 23 21 17
 1       24  9  6  2    20 13  4  5    11 25 18 18
 2       25  6 11  8    15 15  5  4     9 27 22 13
 3       20 11  6  5    21 10  4  6     9 25 17 16
 4       18 14  8  5    18 14  4  4     8 26 23 13
 5       14 18  4  8    10 24  9  7     0 18 22 19
 6       21 11  4  6    18 17  6  6     9 23 19 16
 7       12 19  7  5     8 20 11  2     1 21 23 22
 8       16 14  3  3    17 17  4  3     1 21 24 16
 9       11 13  8  4    11 26  4  3     0 19 21 19

all 10 seeds together

by difficulty, in each quarter of the run
        steps 1-50  51-100    101-150   151-200
rollouts right
easy    39%         87%       95%       95%
medium  26%         84%       95%       96%
hard    6%          46%       67%       77%
no-signal groups (all 4 rewards tied, so skipped)
easy    155/335     203/326   272/332   290/340
medium  192/348     195/363   302/355   311/353
hard    261/317     83/311    103/313   138/307
of those, all wrong + all right
easy    112 + 43    1 + 202   0 + 272   0 + 290
medium  171 + 21    2 + 193   0 + 302   0 + 311
hard    261 + 0     53 + 30   13 + 90   9 + 129
```

With one shared set of weights the hard questions get going. What the agent
learns on easy and medium questions (look first, then query, then write) is
what a hard question needs too, with more looking. By steps 51 to 100 it gets
46% of hard rollouts right, and the hard groups carry most of the signal: 228 of
311 had contrast, against 123 of 326 easy groups and 168 of 363 medium ones.
Easy and medium questions go quiet sooner than before, nearly all of them all
right.

The rule did not change: the signal comes from whatever the agent solves only
sometimes. Which questions those are depends on what it can already do. With
separate weights the hard questions stay at 0% and stay silent. With shared
weights, practice on the other questions lifts them off the floor, and from then
on they are the ones that teach. A real model shares its weights, so practice on
easier questions can lift harder ones there too. The rest of this chapter uses
one set of weights per level, where both kinds of silence are easiest to see.

## What the group mean saves

Every run ends its table with a note:

```text
what the group mean saves
  PPO, the classic algorithm, trains a critic beside the policy: a second
  network, usually as large as the policy, that learns to predict the reward.
  GRPO's prediction is the group's mean reward, which costs nothing to train,
  only the extra rollouts. For LFM2.5-2.6B a critic would be another model of
  about 2.6 billion weights.
```

**PPO** (Proximal Policy Optimization, 2017) is the classic algorithm that GRPO
grew out of. Its baseline comes from a **critic** (also called a **value
function**): a second network trained beside the policy to predict how much
reward the situation will lead to. A critic has to be trained, it can be wrong,
and for a language model it is usually about as large as the model itself.
GRPO (2024) drops it. Its prediction of what a rollout should earn is the mean
reward of the other rollouts of the same question, made a moment earlier with
the same weights. The price is 4 rollouts of every question instead of one.
Chapter 6 covers the part of PPO that GRPO keeps: the importance ratio and the
clip.

## The run's log, and its curves

Every step appends one line to `metrics.jsonl`: the step, the average reward,
the share right, the calls per rollout, the no-signal groups, and each group's
difficulty and four rewards. (A `.jsonl` file holds one JSON record per line,
JSON being a plain-text way to write data.) The fields the project's trainer
also logs keep its names: `accuracy`, `mean_reward`, `mean_calls`,
`groups_without_contrast`. Every rollout, call by call, goes into
`rollouts.jsonl`. The run prints
both files' md5s:

```text
wrote ~/.cache/mhrl-course/05-grpo-groups/runs/seed0-lr3-steps200/metrics.jsonl
md5   a6c8f97d21b5f798147bfa5698cd10e0
wrote ~/.cache/mhrl-course/05-grpo-groups/runs/seed0-lr3-steps200/rollouts.jsonl
md5   5a654c7886cd8f97077386cc5e2d2551
redraw it: uv run plot.py ~/.cache/mhrl-course/05-grpo-groups/runs/seed0-lr3-steps200/metrics.jsonl
```

Same seed, same files: `--trace` only prints more, so `uv run grpo_groups.py`
writes both files again with the same two md5s.

`plot.py` redraws the curves from `metrics.jsonl` and nothing else. It draws
them as text in the terminal, with Python's standard library only (no plotting
library), and imports nothing from this folder, so it works on a file copied to
another machine. Each point averages the last tenth of the run, because a single
step has only 8 rollouts.

```bash
uv run plot.py ~/.cache/mhrl-course/05-grpo-groups/runs/seed0-lr3-steps200/metrics.jsonl
```

```text
200 steps, 400 groups; each point is the average of the 20 steps up to it

correct, by difficulty (e easy, m medium, h hard, * where they meet)
 100%|                               e   eee**ee    m***
     |                 eeee eeeeeeeee eeemmm  mm****e
     |               ee    e    mmm mmmmm
     |              e         mm   m
     |       eeeeeee       mmm
  50%|      e             m
     |     e            mm
     |               mmm
     |  eee     mmmmm
     |ee    mmmm
   0%|******hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
     +--------------------------------------------------
      1                                              200  step

groups with no signal (x all wrong, o all right), as a share of all groups
 100%|
     |                                     oooo     oooo
     |xxxxx                              ooooooooooooooo
     |xxxxxx                  ooo o  ooooooooooooooooooo
     |xxxxxxx   oox o ooooo  ooooooooooooooooooooooooooo
  50%|xxxxxxxxxoxxxoxooooooooooooooooooooooooooooooooooo
     |xxxxxxxxxxxxxxxxxxxoxxoxooooooooooxooxooxooooooooo
     |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxoxxxxxxxxxxxxxxxxxxx
     |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   0%|
     +--------------------------------------------------
      1                                              200  step

calls per rollout
  8.0|
     |                ***** * * *   * *  **************
     |         *******     * * * *** * **              *
     |*********
     |
  4.0|
     |
     |
     |
     |
  0.0|
     +--------------------------------------------------
      1                                              200  step
```

The first chart is the table again, step by step: easy questions rise first,
medium ones follow, and hard ones stay on the floor. In the second, the reason
for the silence changes halfway through. Early on the silent groups are all wrong (`x`);
later most are all right (`o`). The `x` that never goes away, about a third of
all groups, is the hard questions. The third is chapter 4's lesson again:
nothing in the reward pays for stopping, so the calls climb toward the budget
of 8.

## What is in this folder

- `grpo_groups.py`: GRPO on the tiny agent: the steps, the groups, the update,
  the table, the greedy test, the logs, and `--group`, `--trace` and `--shared`.
- `grpo.py`: the project's `group_advantages`, line for line, the step's update
  (an average over calls), and what a group can be (all wrong, all right, or
  signal).
- `plot.py`: the curves, drawn as text from `metrics.jsonl` alone (standard
  library only).
- `seeds.py`: the same run over seeds 0 to 9.
- `report.py`: how runs are printed.
- `environment.py`: chapter 4's environment, with one table per `LOOK`.
- `policy.py`: chapter 4's policy, with "tables seen" as a feature and one set
  of weights per level (or one shared set); its math is chapter 4's, line for
  line.
- `questions.py`: six questions at three difficulties, three queries each, and
  the split by city.
- `grader.py`, `data/shop.db`: chapter 1's grader and the shop database, copied
  unchanged.
- `tasks.py`: chapter 4's `tasks.py`, writing under `05-grpo-groups`.

## Words from this chapter

- **GRPO (Group Relative Policy Optimization)**: try each question several
  times with the same weights, and score each rollout against its group's
  average. The algorithm this project trains with.
- **Group, group size**: the rollouts of one question in one step; 4 here and
  in the project.
- **Group mean**: the average reward of a group. In GRPO it is the baseline.
- **Spread (standard deviation)**: how far a group's rewards scatter around
  their mean. GRPO divides each advantage by it, so every group pushes about
  equally hard.
- **Contrast**: a difference between a group's rewards. Without it there is
  nothing to compare.
- **No-signal group**: a group whose rewards all tie, all wrong or all right.
  Its advantages would all be 0, so GRPO skips it.
- **Difficulty**: easy, medium or hard, by what the SQL needs (one table, two
  joined, three or more).
- **Step structure**: what one step is made of; here and in the project, 2
  questions x 4 rollouts, then one update.
- **PPO (Proximal Policy Optimization)**: the classic algorithm GRPO grew out
  of. It learns its baseline with a critic.
- **Critic (value function)**: a second network that learns to predict the
  reward; GRPO uses the group mean instead.
- **metrics.jsonl**: the run's record, one JSON line per step, enough to redraw
  every curve.

Next: [chapter 6](../06-ratio-and-clip/), GRPO, part 2: the importance ratio and
the clip.
