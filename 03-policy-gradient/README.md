# Chapter 3. Policy gradient: the math behind "do more of what worked"

In chapter 2 an agent learned which query to trust with one rule. After each
try, the question's row of scores changed by

```text
row += lr * reward * (onehot(choice) - probabilities)
```

It worked: 13 of 13 on cities the agent had never seen. But where did that rule
come from? Was it a lucky guess?

It was not. It is the recipe almost all of machine learning uses: write down one
number that should go down, find out which way each score pushes it, and take a
small step downhill. Applied to RL, the recipe is called the **policy
gradient**. This chapter rebuilds chapter 2's agent with it, measures every
piece, and ends on a line that shows chapter 2's rule was exactly this recipe.
On the way it adds the two things chapter 2 was missing: a **baseline**, and a
way to see a learning rate that is too big doing real damage.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/03-policy-gradient
uv run grad_check.py
```

Every script runs in a few seconds on any Mac. They need NumPy (a Python library
for tables of numbers), and `grad_check.py` also uses MLX (below), which `uv
run` installs on an Apple-silicon Mac only; anywhere else `grad_check.py` says
so and still runs. Like chapters 1 and 2, the scripts write only under
`COURSE_HOME` (default `~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `grad_check.py` | probabilities beside log-probabilities, then the gradient three ways, then chapter 2's rule |
| `reinforce.py` | chapter 2's agent trained by gradient descent on a loss, with a baseline; tries printed in full |
| `reinforce.py --baseline none` | the same without a baseline: chapter 2's run, weight for weight |
| `baseline.py` | no baseline, a running average and a per-question average, 5 runs each, side by side |
| `learning_rate.py` | learning rates 0.1, 1 and 30 on shifted rewards, with and without a baseline, 5 runs each |

`reinforce.py` also takes `--lr N`, `--seed N`, `--episodes N`, `--show N`,
`--baseline none|average|question`, `--shifted` (rewards 1.00 and 1.05, see
below) and `--fast` to skip the pauses.

The questions, the 16 candidate queries, chapter 1's grader, the shop database
and the split into training and held-out cities are chapter 2's, copied
unchanged (a test checks they are identical). Only the learning changes.

## Log-probabilities

The first part of `uv run grad_check.py`:

```text
log-probabilities: log p is the natural log of a probability p
     p     log p
  1.00    0.000   certain
  0.90   -0.105
  0.50   -0.693
  0.25   -1.386   one query in four: where every row starts
  0.10   -2.303
  0.01   -4.605
two picks, B for one question (p 0.90) and C for another (p 0.25):
  the chance of both is 0.90 x 0.25 = 0.225, and log 0.225 = -1.492
  the two log-probabilities add up to it: -0.105 + -1.386 = -1.492
1,100 picks at p 0.50 each:
  multiplied, the chance is 0.0 (too small for the computer to hold)
  added, the log-probability is -762.462
```

The **natural log** of a number answers "e (about 2.718) to what power gives this
number?". A probability's natural log is its **log-probability**, written
`log p`. A certain choice (p = 1) has log p = 0; every less likely choice has a
negative log p, and the less likely, the more negative. Each row of the agent's
table starts at 25% per query, which is log p = -1.386.

Logs turn multiplying into adding: the chance of two separate picks both
happening is the two chances multiplied, and its log is the two log-probabilities
added. That is why machine learning works with log-probabilities. A reply from a
language model is hundreds of picks (one per piece of text), and multiplying
hundreds of chances below 1 gives a number too small for the computer to hold,
as the last lines show; adding their logs never has that problem. The engine that
runs LFM2.5-2.6B later in this course reports one log-probability for every
piece of text the model writes (often shortened to **logprobs**).

## A loss

Training needs one number that says how badly the agent is doing on the tries it
just made, so that "learning" can mean "make that number smaller". That number
is a **loss**, and the rule that computes it is the **loss function**.

RL has no correct answer to compare against, only rewards, so the loss is built
from the reward. For one try:

```text
loss = -reward x log p(chosen)
```

Take a rewarded try (reward 1). Its loss is minus the chosen query's log p:
1.386 while that query has a 25% chance, 0.105 once it has 90%. So making the
loss smaller means making the rewarded query more likely. A try with reward 0
has a loss of 0 whatever the chances, so it has nothing to teach; chapter 2's
wrong tries taught nothing, and this is why.

The value of this loss is not a grade. It can even go negative, as you will see
below. What matters is which way it slopes as the scores change.

## The gradient: which way is downhill

A score's **slope** says how fast a number changes as that score changes. The
**gradient** is the list of slopes, one per score. Here we want the gradient of
`log p(C)`: for each of the four scores, how fast the log-probability of query C
changes when that score changes. The second part of `grad_check.py` measures it
on a row where all four scores are 0 (25% each), by moving one score a hair
down and a hair up and watching `log p(C)`:

```text
the gradient of log p(C): for each score, how fast
log p(C) changes when that score changes (its slope)
row scores [+0.00 +0.00 +0.00 +0.00] · probabilities [0.25 0.25 0.25 0.25]

1. measured: move one score 0.001 down and up, watch log p(C)
   score   after -0.001   after +0.001   slope = change / 0.002
   A       -1.386044      -1.386544      -0.250
   B       -1.386044      -1.386544      -0.250
   C       -1.387044      -1.385544      +0.750
   D       -1.386044      -1.386544      -0.250
   measured gradient: [-0.250 -0.250 +0.750 -0.250]
```

Raising C's own score raises `log p(C)`, at 0.75 per unit of score. Raising any
other score lowers it, at 0.25 per unit: the chances must still add up to 1, so
a gain for A is a loss for C. Nothing in this measurement knows a formula; it
only moves numbers and watches.

The other two ways get the same gradient:

```text
2. by hand: onehot(C) - probabilities = [0 0 1 0] - [0.25 0.25 0.25 0.25]
   = [-0.250 -0.250 +0.750 -0.250]
3. MLX, an automatic-gradient library: mx.grad of log p(C)
   = [-0.250 -0.250 +0.750 -0.250]

the same check on 1,000 random rows and choices
  measured vs by hand: largest difference 1.6e-08
  MLX vs by hand:      largest difference 1.6e-07
  (MLX works with 32-bit numbers, about 7 significant digits)
```

`onehot(C)` is 1 for C and 0 for the others, as in chapter 2. **MLX** is an
automatic-gradient library: you write the function, here
`log p(C) = score of C - log(sum of e to each score)`, and `mx.grad` works out
its gradient for you. Later in this course MLX does exactly this for
LFM2.5-2.6B, a model with about 2.6 billion weights, where nobody could work
the gradient out by hand. MLX's last digits differ slightly because
it stores each number with about 7 significant digits instead of 16. On 1,000
random rows all three ways agree to better than one part in a million.

<details>
<summary>Optional: the gradient by hand</summary>

Write the log of softmax out: with `S = e^A + e^B + e^C + e^D`,

```text
log p(C) = C - log S
```

The slope of `C` with respect to C is 1, and with respect to any other score it
is 0: that is `onehot(C)`. The slope of `log S` with respect to a score k is
`e^k / S`, which is exactly softmax's probability for k. So the gradient is

```text
gradient of log p(C) = onehot(C) - probabilities
```

At 25% each that is `[0 0 1 0] - [0.25 0.25 0.25 0.25]`, the measured numbers.
</details>

## Gradient descent, and chapter 2's rule

**Gradient descent** is how the step downhill is taken: move every score a
little against its slope,

```text
new scores = scores - lr x gradient of the loss
```

where `lr`, the **learning rate**, sets how big the step is. The loss
`-reward x log p(C)` has the gradient `-reward x (onehot(C) - probabilities)`,
so the step is `+lr x reward x (onehot(C) - probabilities)`. The last part of
`grad_check.py` puts the two side by side:

```text
the loss for a rewarded C: -reward x log p(C), reward 1
  its gradient, -reward x that slope:  [+0.25 +0.25 -0.75 +0.25]
  one gradient-descent step, lr 0.25:  [-0.0625 -0.0625 +0.1875 -0.0625]
  chapter 2's nudge, lr 0.25:          [-0.0625 -0.0625 +0.1875 -0.0625]
on the 1,000 random tries, the step and the nudge differ by at most
  0 by hand, 4.0e-09 measured, 3.3e-08 with MLX
chapter 2's nudge was exactly lr x reward x the gradient of log p
```

That is where the rule came from. Make what was tried more likely, in proportion
to how well it turned out: the update is "how good" times the gradient of the
log-probability of what was tried. That idea is the **policy gradient**, and its
original recipe, published by Ronald Williams in 1992, is called
**REINFORCE**: try something, score it, push up the log-probability of what you
tried, scaled by the score.

`reinforce.py` is chapter 2's agent rewritten this way: each try computes the
loss and takes one gradient-descent step. With `--baseline none` it is chapter
2's run, weight for weight:

```bash
uv run reinforce.py --baseline none
```

```text
the 16 weights: the scores softmax turns into those chances
                                                         A       B       C       D
How many customers live in {city}?                   -1.07   +3.22   -1.07   -1.07
How many orders did customers from {city} place?     -1.11   -1.11   -1.11   +3.32
How much did customers from {city} spend?            +3.56   -1.19   -1.19   -1.19
When was the first order from {city}?                -1.15   -1.15   +3.45   -1.15
```

These are chapter 2's 16 weights; a test checks they match to the last bit, not
just the two decimals shown.

## Advantage and baseline

The reward alone has a weakness: it says how good a try was, not whether it was
better or worse than usual. A **baseline** is what the agent expects to earn,
and the **advantage** is the reward minus the baseline: how much better (above
0) or worse (below 0) the try did than expected. The policy-gradient loss uses
the advantage in place of the reward:

```text
loss = -advantage x log p(chosen),   advantage = reward - baseline
```

A positive advantage pushes the choice up; a negative one pushes it down, so now
a wrong try teaches something too. By default `reinforce.py` uses the simplest
baseline, the **running average**: the average reward of all the earlier tries.
On the very first try there is nothing to average yet, so its baseline is its own
reward: its advantage is 0 and it changes nothing.

```bash
uv run reinforce.py
```

Each printed try now shows every number of its loss, then the row's chances after
the step:

```text
try 51 · How much did customers from Vienna spend?
        chose D: SELECT COUNT(*) ... → 17 · wrong
        p = 0.12 · log p = -2.103 · reward 0 · baseline 0.46 · advantage -0.46
        loss -0.967 → new probabilities A 0.669 · B 0.100 · C 0.126 · D 0.106
try 52 · How many orders did customers from Madrid place?
        chose B: SELECT SUM(quantity) ... → 138 · wrong
        p = 0.19 · log p = -1.654 · reward 0 · baseline 0.45 · advantage -0.45
        loss -0.746 → new probabilities A 0.198 · B 0.172 · C 0.186 · D 0.443
try 53 · How many orders did customers from Rome place?
        chose D: SELECT COUNT(*) FROM orders JOIN ... → 5 · correct
        p = 0.44 · log p = -0.814 · reward 1 · baseline 0.44 · advantage +0.56
        loss +0.454 → new probabilities A 0.189 · B 0.165 · C 0.178 · D 0.469
```

Try 51 picked a wrong query that had a 12% chance. The reward was 0, below the
0.46 the agent had come to expect, so the advantage is -0.46 and D drops to
10.6%. Try 53's right answer beat expectations by 0.56, and D rises from 44% to
46.9%. Try 51's loss is negative: lowering it means lowering log p(D), which is
what a negative advantage asks for.

With chapter 2's rewards the baseline changes the end result very little:

```text
after 500 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                     1   *97     1     1    112
How many orders did customers from {city} place?       1     1     1   *96    113
How much did customers from {city} spend?            *97     1     1     1    145
When was the first order from {city}?                  1     1   *97     1    130

the 16 weights: the scores softmax turns into those chances
                                                         A       B       C       D
How many customers live in {city}?                   -1.11   +3.56   -1.53   -0.92
How many orders did customers from {city} place?     -1.21   -0.92   -1.15   +3.28
How much did customers from {city} spend?            +3.51   -0.99   -1.56   -0.96
When was the first order from {city}?                -1.25   -1.04   +3.48   -1.19
```

The held-out test is again 13 of 13. One thing did change: the three wrong
queries in a row no longer share one score, because each wrong try now pushes
down the query it tried.

## When the baseline matters: shifted rewards

Chapter 2's rewards were 1 for right and 0 for wrong. Now shift them: **1.05**
for a right answer and **1.00** for a wrong one (`--shifted`). They carry the
same information (the right query still earns more), but every answer now earns
about 1. Real rewards often look like this: the project's reward adds bonuses,
and plenty of reward functions never give 0.

Without a baseline, every advantage is now 1.00 or 1.05, so every step pushes up
whatever query was tried, right or wrong. `baseline.py` trains each kind of
baseline 5 times, on both kinds of reward, at learning rate 1. The runs use
seeds 0 to 4: a **seed** is the number that starts the random choices (which
question comes up, which query is picked), so each run can be repeated exactly.
For every try it records the **push**: how far that try's step moved the right
query's score. The average push is the useful part, what learning needs. The
**spread** (the standard deviation, a measure of how far the pushes scatter
around their average) is the noise.

```bash
uv run baseline.py
```

```text
baselines · lr 1 · 500 tries per run · seeds 0 to 4 (5 runs per line)
push: how far one try's step moved the right query's score
spread: how much the push swings from try to try (standard deviation)
right query: its chance at the end, average of 20 rows (the lowest row)
locked: rows where a wrong query holds 90% or more

rewards 1 for a right answer, 0 for a wrong one (chapter 2's)
                      push per try     wrong queries   right query   locked
                      average  spread  pushed up       (lowest row)  rows
no baseline           +0.037   0.091   0 of 2,500      99% (99%)     0 of 20
running average       +0.038   0.130   0 of 2,500      99% (98%)     0 of 20
per-question average  +0.038   0.133   0 of 2,500      99% (98%)     0 of 20

shifted rewards: 1.05 for a right answer, 1.00 for a wrong one
                      push per try     wrong queries   right query   locked
                      average  spread  pushed up       (lowest row)  rows
no baseline           +0.005   0.240   1,686 of 2,500  35% (0%)      12 of 20
running average       +0.012   0.006   0 of 2,500      70% (61%)     0 of 20
per-question average  +0.012   0.007   0 of 2,500      69% (59%)     0 of 20
with no baseline every advantage is 1.00 or 1.05, so every step pushes up the query it
tried: 2,500 of 2,500 tries, 1,686 of them on a wrong query
```

Read the shifted block. Without a baseline, the useful push of +0.005 per try is
buried under swings of 0.240, and 1,686 tries pushed a wrong query up. After 500
tries the right query averages 35%, and 12 of the 20 rows (4 questions x 5 runs)
have locked onto a wrong query: it holds 90% or more of the chance, so the agent
almost never tries anything else. With either baseline, a wrong answer earns
less than expected and is pushed down: no wrong query is ever pushed up, the
spread is 35 to 40 times smaller, and every row is still learning.

Read the first block honestly too. With chapter 2's rewards the baseline does
not shrink the spread here; it grows a little (0.091 to 0.130), because wrong
tries now move the scores as well, and the end result is the same. A baseline
pays off when every reward shares a large common part, like the 1.00 here. The
running average and the per-question average behave alike in this toy, because
the four questions earn rewards at similar rates. A per-question average is the
idea GRPO builds on in chapter 5, where the baseline is the average of several
attempts at the same question.

## Learning rate: too slow, and locked in

Chapter 2 tried learning rate 30 and saw no harm: with 0/1 rewards only the right
query can earn anything, so a huge step can only jump to the right query. With
shifted rewards a wrong query earns a reward too. `learning_rate.py` tries
learning rates 0.1, 1 and 30, with no baseline and with the running average, 5
runs each:

```bash
uv run learning_rate.py
```

```text
learning rates on shifted rewards: 1.05 for a right answer, 1.00 for a wrong one
500 tries per run · seeds 0 to 4 (5 runs per line)
right query: its chance at the end, average of 20 rows
locked: rows where a wrong query holds 90% or more
held-out: the greedy test, 13 questions per run

lr    baseline         right query  locked    per seed   held-out
0.1   no baseline      26%          0 of 20   0 0 0 0 0  17 of 65
1     no baseline      35%          12 of 20  1 4 2 3 2  23 of 65
30    no baseline      25%          15 of 20  3 4 2 3 3  17 of 65
0.1   running average  28%          0 of 20   0 0 0 0 0  65 of 65
1     running average  70%          0 of 20   0 0 0 0 0  65 of 65
30    running average  99%          0 of 20   0 0 0 0 0  65 of 65
```

Without a baseline, no learning rate works:

- **lr 0.1 is too slow.** The steps are too small to rise above the noise: after
  500 tries the right query averages 26%, and the held-out test gets 17 of 65,
  about what guessing gets (one in four of 65 is about 16).
- **lr 1 already locks in**: 12 of the 20 rows end with a wrong query at 90% or
  more. The per-seed counts show it happens in every run, not in one unlucky
  one.
- **lr 30 locks in at once.** The first try at each question gets an advantage
  of about 1 and a step of 30 x 0.75 = 22.5, which takes the query it happened
  to try from 25% to 100%. In all 20 rows the query that wins is the first one
  tried, so about three rows in four lock onto a wrong query: 15 of 20.

Here is that happening at seed 0:

```bash
uv run reinforce.py --shifted --baseline none --lr 30
```

```text
try 1 · When was the first order from Copenhagen?
        chose D: SELECT MIN(order_date) FROM orders → 2024-01-08 · wrong
        p = 0.25 · log p = -1.386 · reward 1.00 · baseline 0.00 · advantage +1.00
        loss +1.386 → new probabilities A 0.000 · B 0.000 · C 0.000 · D 1.000
```

```text
locked onto a wrong query (it is now almost the only one tried):
How many customers live in {city}?                A at 100%, right query B at 0%
How much did customers from {city} spend?         C at 100%, right query A at 0%
When was the first order from {city}?             D at 100%, right query C at 0%

greedy test (always the top score) on cities never used in training
How many customers live in {city}?                picked A · 0 of 4 correct
How many orders did customers from {city} place?  picked D · 3 of 3 correct
How much did customers from {city} spend?         picked C · 0 of 3 correct
When was the first order from {city}?             picked D · 0 of 3 correct
held-out score: 3 of 13
```

With the running average, the same three learning rates behave very
differently. Learning rate 0.1 is still too slow: the right query reaches only
28%, though it already leads in every row, which is all the greedy test needs.
Learning rate 1 is learning (70%). Learning rate 30 learns all 20 rows (99%),
with nothing locked, because the step is the learning rate times the advantage,
and with a baseline these advantages are only a few hundredths. The same first
try that locked onto D above now changes nothing:

```bash
uv run reinforce.py --shifted --lr 30
```

```text
try 1 · When was the first order from Copenhagen?
        chose D: SELECT MIN(order_date) FROM orders → 2024-01-08 · wrong
        p = 0.25 · log p = -1.386 · reward 1.00 · baseline 1.00 · advantage +0.00
        loss +0.000 → new probabilities A 0.250 · B 0.250 · C 0.250 · D 0.250
```

```text
try 51 · How much did customers from Vienna spend?
        chose C: SELECT SUM(unit_price) ... → 1038.73 · wrong
        p = 0.02 · log p = -3.806 · reward 1.00 · baseline 1.03 · advantage -0.03
        loss -0.129 → new probabilities A 0.981 · B 0.007 · C 0.003 · D 0.009
```

This run's held-out score is 13 of 13. So a learning rate is never big or
small on its own: what moves the scores is the learning rate times the
advantage.

## Same seed, same file

The same seed repeats a run exactly. Every run writes its log under
`COURSE_HOME` and prints that file's md5, a fingerprint of its contents:

```text
wrote ~/.cache/mhrl-course/03-policy-gradient/runs/seed0-lr0.25-baseline-average-rewards-0-1-tries500.jsonl
md5   18792be76df8adcc53add8d349c65019
```

Run it again and the md5 is the same; `--seed 1` is a different run. `baseline.py`
and `learning_rate.py` do the same for their 30 runs each:

```text
md5   c42e25ac4c8f545f9aa695b8dd314993
md5   38d1723f58464e77afbcb99018e68ef8
```

Those two scripts train 30 runs each, so they run each question-and-query pair
through the grader once and reuse the result, instead of a fresh folder for
every try: a query on the same database always gives the same answer. A test
checks that reusing the results gives exactly the same run as fresh folders.

## What is in this folder

- `policy.py`: the chapter's math: softmax, log p, the gradient of log p by hand,
  the loss, its gradient and one gradient-descent step.
- `reinforce.py`: the agent trained on the loss: baselines, the shifted rewards,
  training, the greedy test and the run log.
- `grad_check.py`: log-probabilities, the gradient three ways, and chapter 2's
  rule.
- `baseline.py`: the three baselines on both kinds of reward.
- `learning_rate.py`: three learning rates, with and without a baseline.
- `report.py`: how runs are printed.
- `questions.py`, `grader.py`, `data/shop.db`: chapter 2's questions, chapter 1's
  grader and the shop database, copied unchanged.
- `tasks.py`: chapter 2's `tasks.py`, writing under `03-policy-gradient`.

## Words from this chapter

- **Log-probability (log p, logprob)**: the natural log of a probability: 0 for
  certain, more negative for less likely. Adding log-probabilities is the same
  as multiplying the probabilities.
- **Loss and loss function**: one number that says how badly the model is doing;
  training makes it smaller. The policy-gradient loss for a try is
  `-advantage x log p(chosen)`.
- **Gradient**: one slope per score: how fast a number changes as that score
  changes.
- **Gradient descent**: move every score a little against the loss's gradient,
  again and again; that is training. Working the gradient out through a whole
  network is called **backpropagation**.
- **Learning rate**: how big each step is. Too small and nothing changes in time;
  too big and the first lucky (or unlucky) try decides everything. What moves the
  scores is the learning rate times the advantage.
- **Policy gradient**: make what was tried more likely in proportion to how well
  it turned out: advantage times the gradient of that choice's log-probability.
- **REINFORCE**: the original policy-gradient recipe (Williams, 1992): try,
  score, push up the log-probability of what was tried, scaled by reward minus
  baseline.
- **Baseline**: what the agent expects to earn, here an average of earlier
  rewards. Subtracting it removes the part every reward shares, which makes the
  updates far steadier when that part is large.
- **Advantage**: reward minus baseline: above 0 means do more of that, below 0
  means do less.
- **Spread (standard deviation)**: how far a set of numbers scatters around its
  average; here, how noisy the updates are.
- **MLX**: an automatic-gradient library: you write a function, it works out the
  gradient. The course uses it again to train LFM2.5-2.6B.
- **Seed**: the number that starts the random choices; same seed, same run.

Next: [chapter 4](../04-episodes/), several calls, one reward: a tiny agent.
