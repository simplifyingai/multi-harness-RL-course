# Chapter 7. Rewards that backfire

In chapters 4 to 6 the agent was paid 1 for a right answer and 0 for anything
else. That is easy to state and slow to earn, so it is tempting to pay a
little extra: for handing an answer in, for using fewer calls, for writing
less. Each extra is meant to steer the agent toward something good.

RL does not care what a bonus was meant for. It raises the number it is
paid, by whatever route raises it fastest. This chapter pays chapter 6's
tiny agent five different ways and watches what each one teaches. Paid for
right answers only, it learns the job and wastes calls; two of the extras
teach it to stop working; one makes its replies longer, and one makes them
shorter. Every claim is checked over five seeds. The last
section is the project's own first RL run, which was paid one of these
bonuses.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/07-reward-design
uv run reward_lab.py --reward call-bonus --trace
```

A run takes about 15 seconds on any Mac (5 with `--fast`, which skips the
pauses). It needs NumPy, which `uv run` installs for this script alone, and
like every chapter it writes only under `COURSE_HOME` (default
`~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `reward_lab.py --show-group` | an all-right group whose rewards differ only by a bonus, worked out by hand |
| `reward_lab.py` | paid for right answers only |
| `reward_lab.py --reward submit` | plus 0.2 for handing in any answer |
| `reward_lab.py --reward bonus-when-wrong` | a bonus for fewer tokens, paid even when wrong |
| `reward_lab.py --reward call-bonus --trace` | a bonus for fewer calls, with four groups printed in full |
| `reward_lab.py --reward token-bonus` | a bonus for fewer tokens, paid only when right |
| `seeds.py` | all five over seeds 0 to 4, each claim checked (about half a minute) |
| `seeds.py --seeds 10 --length-lr 1 --rules correct,call-bonus,token-bonus` | the same with a higher learning rate for reply length |
| `seeds.py --guessable --rules correct` and `reward_lab.py --guessable` | questions a guess can answer |
| `reward_lab.py --crashes 0.1` and `--crashes 0.1 --unscored` | sandboxes that die before grading |
| `plot.py <metrics.jsonl>` | a run's curves, redrawn from its log alone |
| `real_run.py` | the project's first real RL run, step by step |

Other flags: `--seed N`, `--steps N`, `--lr N` (the action weights, 3 by
default), `--length-lr N` (the reply length, 0.3 by default) and `--fast`.
`seeds.py` takes `--seeds N`, `--steps N`, `--length-lr N`, `--rules` and
`--guessable`.

## What is new since chapter 6

The agent is chapter 6's: the same six questions at three difficulties, the
same `LOOK` that shows one table at a time, the same policy, chapter 6's GRPO
with the ratio and the clip, and the same steps of 2 questions x 4 rollouts.
Every step samples a fresh batch and makes one update, so (chapter 6) every
ratio is exactly 1 and the clip never acts. Four things are new.

**Handing in, and guessing.** `SUBMIT` writes the result in hand to
answer.txt and hands it in, which ends the rollout: chapter 6's `WRITE` and
`FINISH` in one action. `FINISH` now stops without handing anything in. And
there is a new action, `SUBMIT_GUESS`, which hands in a guess at once,
without looking or querying. The guess is 0, the answer an agent writes when
it has found nothing. No question in this chapter has 0 for an answer
(chapters 5 and 6 held one out, Brussels' number of orders, and it is left
out here), so a guess is always wrong.

**Replies with a length.** A real model writes its reply in **tokens**:
pieces of text, often a short word or part of a longer one (chapter 9 looks
at them up close). Its tool call sits somewhere in that text, and the
number of tokens is what a reply costs. Here every call is a reply with a
length in tokens.
The agent draws each reply's length from a bell curve around its **typical
reply length**, which starts at 40 tokens, give or take 30%. A reply of 80
tokens or more holds two actions, the way a model can run a query and hand
in its result in one message, so two actions cost one call. A reply longer
than the **output cap** of 120 tokens is cut off before it reaches its
action (the project's engine, the program that runs the model, cuts replies
off at 1,024 tokens), and a reply
with no action ends the rollout, the way Claude Code takes a reply with no
tool call as the agent's last word. A reply is never shorter than 20 tokens,
the fewest that can name an action.

**The typical length is learned.** It is one more number the agent learns,
kept as its log, the **length weight**. The math is chapter 3's, for a bell
curve instead of softmax: with x the log of a reply's length, w the length
weight and s the spread (0.3), log p(x) is `-(x - w)**2 / (2 s**2)` plus a
constant, so its slope is `(x - w) / s**2`, positive for a reply longer than
typical. A reply length that did better than its group pulls the length
weight toward it; one that did worse pushes it away. It has its own learning
rate, 0.3, which is the size of the step. In chapter 6 every call was one
decision, its action; now a call is a reply length and then one or two
actions, and the loss weighs every decision by its ratio, as the project
weighs every token.

**One set of weights for every level.** Chapters 5 and 6 kept one set of
weights per difficulty. Here the agent uses chapter 5's `--shared` set, as a
real model does, because a reward rule acts on habits every question shares:
how long to write, when to hand in, whether to guess. (With one set per
level and two actions that end a rollout at once, the medium questions
rarely came out right by luck, and hard ones never did, so little but the
easy questions was ever learned.)

Every run says all this first:

```text
reward_lab · reward correct · seed 0 · lr 3 · length lr 0.3 · 200 steps
reward: 1 if right, else 0
each step: 2 questions x 4 rollouts at temperature 1, then one update (chapter 6's GRPO)
questions: 45 to train on, 18 held out; SUBMIT_GUESS hands in 0, right for 0
replies: about 40 tokens at first, give or take 30%, never under 20;
  80+ tokens hold 2 actions; over 120 is cut off before its actions and ends the rollout
```

## Five ways to pay

`rewards.py` turns what happened in a rollout (right or not, handed in or
not, calls, tokens) into a reward:

| `--reward` | pays |
| --- | --- |
| `correct` | 1 if right, else 0 (chapters 4 to 6) |
| `submit` | the same, plus 0.2 for handing in any answer at all |
| `call-bonus` | if right, 1 plus up to 0.1 for fewer calls; else 0 |
| `token-bonus` | if right, 1 plus up to 0.1 for fewer tokens; else 0 |
| `bonus-when-wrong` | 1 if right, else 0, plus up to 0.1 for fewer tokens, right or wrong |

The bonus for fewer calls is the article's, and the bonus for fewer tokens
is this project's. Both are the project's code line for line
(`mhrl/rollout.py`; a test checks them, docstrings left out):

```python
def tool_bonus(n_calls: int, max_calls: int) -> float:
    span = max(max_calls - MIN_TOOL_CALLS, 1)
    unused = max_calls - max(n_calls, MIN_TOOL_CALLS)
    return Defaults.MAX_TOOL_BONUS * max(0.0, min(1.0, unused / span))


def token_bonus(n_sampled: int, budget: int) -> float:
    return Defaults.MAX_TOOL_BONUS * max(0.0, 1.0 - n_sampled / max(budget, 1))


def reward_for(correct: bool, n_sampled: int, budget: int) -> float:
    if not correct:
        return 0.0
    return 1.0 + token_bonus(n_sampled, budget)
```

`MAX_TOOL_BONUS` is 0.1 and `MIN_TOOL_CALLS` is 2. The call bonus is 0.1 at
2 calls or fewer and 0 at the budget of 8. The token bonus is 0.1 at zero
tokens and 0 at a budget of 400 tokens per rollout (the project's budget is
2,000, for replies about ten times longer). `reward_for` is the token-bonus
rule. Its first two lines are the whole difference from `bonus-when-wrong`:
a wrong answer earns nothing, however short.

## A tiny bonus is a full-strength push

```bash
uv run reward_lab.py --show-group
```

```text
rewards     [1.08, 1.06, 1.07, 1.05]
mean        1.065
spread      0.0112  (standard deviation)
advantages  (reward - mean) / (spread + 0.0001)
            [+1.33, -0.44, +0.44, -1.33]

paid for right answers only, the same rollouts are [1, 1, 1, 1]:
no contrast, skipped
```

These four rollouts were all right. Paid by the token bonus, they wrote 80,
160, 120 and 200 tokens, so their rewards differ by at most 0.03. GRPO
(chapter 5) subtracts the group's mean and divides by the group's spread,
and the spread here is tiny, 0.0112. So the shortest rollout gets +1.33 and
the longest -1.33: as large a push as a right answer against a wrong one
gets in chapter 5. Paid for right answers only, the same four tie and teach
nothing. Once an agent gets most questions right, the bonus is what most of
its groups are about.

## Paid for right answers only

```bash
uv run reward_lab.py
```

The first steps, then the run in quarters:

```text
step   1 · reward 0.00 · correct 0% · calls 3.9 · tokens/reply 41 · cut off 0% · no-signal groups 2/2
step   2 · reward 0.00 · correct 0% · calls 2.6 · tokens/reply 43 · cut off 0% · no-signal groups 2/2
step   3 · reward 0.00 · correct 0% · calls 1.8 · tokens/reply 43 · cut off 0% · no-signal groups 2/2
step   4 · reward 0.00 · correct 0% · calls 2.5 · tokens/reply 42 · cut off 0% · no-signal groups 2/2
step   5 · reward 0.00 · correct 0% · calls 3.2 · tokens/reply 44 · cut off 0% · no-signal groups 2/2
step   6 · reward 0.00 · correct 0% · calls 1.8 · tokens/reply 44 · cut off 0% · no-signal groups 2/2
step   7 · reward 0.12 · correct 12% · calls 3.1 · tokens/reply 45 · cut off 0% · no-signal groups 1/2
step   8 · reward 0.00 · correct 0% · calls 4.0 · tokens/reply 54 · cut off 0% · no-signal groups 2/2
step   9 · reward 0.00 · correct 0% · calls 3.5 · tokens/reply 54 · cut off 0% · no-signal groups 2/2
step  10 · reward 0.12 · correct 12% · calls 4.4 · tokens/reply 51 · cut off 0% · no-signal groups 1/2
step  11 · reward 0.00 · correct 0% · calls 3.4 · tokens/reply 53 · cut off 0% · no-signal groups 2/2
step  12 · reward 0.00 · correct 0% · calls 3.2 · tokens/reply 54 · cut off 0% · no-signal groups 2/2
step  13 · reward 0.00 · correct 0% · calls 3.6 · tokens/reply 48 · cut off 0% · no-signal groups 2/2
step  14 · reward 0.00 · correct 0% · calls 4.1 · tokens/reply 47 · cut off 0% · no-signal groups 2/2
```

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     19%         68%       77%       86%
  easy                      28%         85%       88%       92%
  medium                    20%         78%       90%       97%
  hard                      9%          34%       54%       67%
reward                      0.19        0.68      0.77      0.86
calls per rollout           4.2         5.6       5.7       6.1
tokens per reply            63          57        55        33
replies cut off             2%          1%        1%        0%
replies holding 2 actions   17%         11%       9%        1%
rollouts that guessed       25%         2%        0%        0%
typical reply at the end    70          35        54        31

greedy test at the end (the most likely action, every reply 31 tokens)
  easy    training 15 of 15  held out 8 of 8 · LOOK x3, QUERY_A, SUBMIT
  medium  training 16 of 16  held out 6 of 6 · LOOK x3, QUERY_A, SUBMIT
  hard    training  0 of 14  held out 0 of 4 · LOOK x3, QUERY_A x5
```

The table adds up the training rollouts of each quarter of the run (50 steps
each): the share right, overall and by level; the average reward; calls per
rollout; **tokens per reply** (a cut-off reply counts 120); the share of
replies cut off; the share holding two actions; the share of rollouts that
ended with a guess; and the typical reply length the agent had learned by
the end of the quarter.

The agent learns the job: 86% right in the last quarter, 67% even on hard
questions. And calls go up, 4.2 per rollout in the first quarter and 6.1 in
the last, of a budget of 8. Nothing in the reward pays for using fewer of
them: a right answer in 7 calls earns exactly what a right answer in 3 does,
so extra looks and extra queries cost nothing and are never pushed out
(chapter 5's lesson). The greedy test at the end takes the most likely
action at every call and the typical reply length; it looks three times and
queries, which is right for easy and medium questions and one look short for
a hard one, in this rule and in every other.

## +0.2 for handing in anything

```bash
uv run reward_lab.py --reward submit
```

```text
step   1 · reward 0.15 · correct 0% · calls 3.9 · tokens/reply 41 · cut off 0% · no-signal groups 0/2
step   2 · reward 0.10 · correct 0% · calls 3.4 · tokens/reply 41 · cut off 0% · no-signal groups 0/2
step   3 · reward 0.18 · correct 0% · calls 1.6 · tokens/reply 35 · cut off 0% · no-signal groups 1/2
step   4 · reward 0.18 · correct 0% · calls 2.1 · tokens/reply 42 · cut off 0% · no-signal groups 1/2
step   5 · reward 0.20 · correct 0% · calls 2.1 · tokens/reply 45 · cut off 0% · no-signal groups 2/2
step   6 · reward 0.18 · correct 0% · calls 1.6 · tokens/reply 42 · cut off 0% · no-signal groups 1/2
step   7 · reward 0.20 · correct 0% · calls 1.8 · tokens/reply 62 · cut off 0% · no-signal groups 2/2
step   8 · reward 0.33 · correct 12% · calls 1.9 · tokens/reply 60 · cut off 0% · no-signal groups 1/2
step   9 · reward 0.33 · correct 12% · calls 3.2 · tokens/reply 58 · cut off 0% · no-signal groups 1/2
step  10 · reward 0.12 · correct 0% · calls 4.2 · tokens/reply 45 · cut off 0% · no-signal groups 0/2
step  11 · reward 0.20 · correct 0% · calls 2.9 · tokens/reply 45 · cut off 0% · no-signal groups 2/2
step  12 · reward 0.20 · correct 0% · calls 2.8 · tokens/reply 41 · cut off 0% · no-signal groups 2/2
step  13 · reward 0.18 · correct 0% · calls 2.6 · tokens/reply 38 · cut off 0% · no-signal groups 1/2
step  14 · reward 0.18 · correct 0% · calls 2.4 · tokens/reply 70 · cut off 5% · no-signal groups 1/2
step  15 · reward 0.18 · correct 0% · calls 2.2 · tokens/reply 51 · cut off 0% · no-signal groups 1/2
step  16 · reward 0.20 · correct 0% · calls 1.5 · tokens/reply 52 · cut off 0% · no-signal groups 2/2
step  17 · reward 0.20 · correct 0% · calls 1.1 · tokens/reply 42 · cut off 0% · no-signal groups 2/2
step  18 · reward 0.20 · correct 0% · calls 1.5 · tokens/reply 43 · cut off 0% · no-signal groups 2/2
step  19 · reward 0.20 · correct 0% · calls 1.2 · tokens/reply 46 · cut off 0% · no-signal groups 2/2
step  20 · reward 0.20 · correct 0% · calls 1.6 · tokens/reply 52 · cut off 0% · no-signal groups 2/2
```

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     0%          0%        0%        0%
  easy                      1%          0%        0%        0%
  medium                    0%          0%        0%        0%
  hard                      0%          0%        0%        0%
reward                      0.20        0.20      0.20      0.20
calls per rollout           1.7         1.2       1.2       1.1
tokens per reply            43          33        35        36
replies cut off             0%          0%        0%        0%
replies holding 2 actions   2%          0%        0%        0%
rollouts that guessed       94%         100%      100%      100%
typical reply at the end    31          31        35        35

greedy test at the end (the most likely action, every reply 35 tokens)
  easy    training  0 of 15  held out 0 of 8 · SUBMIT_GUESS
  medium  training  0 of 16  held out 0 of 6 · SUBMIT_GUESS
  hard    training  0 of 14  held out 0 of 4 · SUBMIT_GUESS
```

By step 20 every rollout's reward is 0.20, and from the second quarter on
every rollout ends with `SUBMIT_GUESS`. An untrained agent is almost never
right, so in nearly every group the only difference between rollouts is
whether they handed something in. The guess hands in at once, in one call,
so GRPO pushes it up at full strength (dividing by a small spread again),
long before the agent could learn to query. Once all four rollouts of a group
guess, they all score 0.20, the group has no contrast, and nothing is ever
learned again. The training reward rose from 0.15 at step 1 to 0.20, while
the agent was right once at step 8, once at step 9, and never after. This is
the article's real collapse, blind submitting, in miniature.

## A bonus paid even when wrong

```bash
uv run reward_lab.py --reward bonus-when-wrong
```

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     0%          0%        0%        0%
  easy                      0%          0%        0%        0%
  medium                    0%          0%        0%        0%
  hard                      0%          0%        0%        0%
reward                      0.09        0.09      0.09      0.09
calls per rollout           1.3         1.0       1.0       1.0
tokens per reply            26          23        23        23
replies cut off             0%          0%        0%        0%
replies holding 2 actions   0%          0%        0%        0%
rollouts that guessed       16%         3%        2%        1%
typical reply at the end    20          20        20        20

greedy test at the end (the most likely action, every reply 20 tokens)
  easy    training  0 of 15  held out 0 of 8 · FINISH
  medium  training  0 of 16  held out 0 of 6 · FINISH
  hard    training  0 of 14  held out 0 of 4 · FINISH
```

This rule pays the token bonus whether the answer is right or not. Early on
nothing is right, so the bonus is the only difference within a group, and
the rollout that wrote the fewest tokens wins: the one that stopped first.
Within the first quarter the agent stops at its first call, with the
shortest reply it can write, and stays there: reward 0.09 for doing nothing.
`reward_for`'s `if not correct: return 0.0` is the line that prevents this.

## A bonus for fewer calls

```bash
uv run reward_lab.py --reward call-bonus --trace
```

`--trace` prints the first group of each quarter in full. Each rollout shows
its reward, its advantage A, what it handed in, and every reply as
`[tokens] actions`, with `+` between two actions in one reply:

```text
step 51 · group 1 of 2 · medium · When was the first order from Rome?
  rollout 1  reward 1.07  A +1.03     answer 2025-04-15 [117] QUERY_A+LOOK, [85] SUBMIT+QUERY_A, [72] LOOK, [83] QUERY_A+SUBMIT
  rollout 2  reward 1.03  A +0.97     answer 2025-04-15 [89] LOOK+QUERY_A, [71] LOOK, [109] LOOK+LOOK, [61] LOOK, [92] SUBMIT+QUERY_A, [82] LOOK+SUBMIT
  rollout 3  reward 0     A -1.00     answer (none)     [82] LOOK+LOOK, [89] QUERY_A+FINISH
  rollout 4  reward 0     A -1.00     answer (none)     [67] LOOK, [120] cut off
```

At step 51 the agent already writes long replies. Rollout 1 holds two actions
in three of its four replies, though three of those actions came too early
(two queries before their tables were seen, a `SUBMIT` with nothing in
hand); it was still right in four calls and earned 1.07. Rollout 4 looked
once and then wrote a reply of over 120 tokens: cut off, the rollout ended
with nothing handed in, reward 0.

```text
step 151 · group 1 of 2 · easy · How many customers live in Madrid?
  rollout 1  reward 1.08  A +1.71     answer 14         [54] LOOK, [42] QUERY_A, [50] SUBMIT
  rollout 2  reward 1.07  A -0.57     answer 14         [47] LOOK, [61] LOOK, [34] QUERY_A, [43] SUBMIT
  rollout 3  reward 1.07  A -0.57     answer 14         [45] LOOK, [75] QUERY_A, [48] LOOK, [57] SUBMIT
  rollout 4  reward 1.07  A -0.57     answer 14         [60] LOOK, [33] LOOK, [77] QUERY_A, [46] SUBMIT
```

At step 151 all four are right. Rollout 1 used three calls and earned 1.08;
the others used four and earned 1.07. A difference of 0.017 gave rollout 1
an advantage of +1.71 and each of the others -0.57: the `--show-group`
arithmetic in a real run.

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     17%         56%       69%       76%
  easy                      24%         75%       88%       92%
  medium                    22%         63%       82%       88%
  hard                      3%          25%       38%       42%
reward                      0.17        0.59      0.73      0.80
calls per rollout           4.5         5.3       4.9       5.0
tokens per reply            58          66        68        65
replies cut off             1%          3%        3%        2%
replies holding 2 actions   11%         19%       20%       18%
rollouts that guessed       26%         4%        2%        0%
typical reply at the end    84          62        57        64

greedy test at the end (the most likely action, every reply 64 tokens)
  easy    training 15 of 15  held out 8 of 8 · LOOK x3, QUERY_A, SUBMIT
  medium  training 16 of 16  held out 6 of 6 · LOOK x3, QUERY_A, SUBMIT
  hard    training  0 of 14  held out 0 of 4 · LOOK x3, QUERY_A x5
```

Against the run paid for right answers only (same seed): replies of 65
tokens in the last quarter instead of 33; 18% of replies holding two actions
instead of 1%; 2% cut off instead of 0%; 5.0 calls per rollout instead of
6.1; and 76% right instead of 86%, 42% instead of 67% on hard questions.

This is the mechanism, specified. Every all-right group pushes, at full
strength, toward the rollouts that used fewer calls. Some of those used
fewer because their replies happened to be long enough to hold two actions,
so the typical reply length is pulled up. Nothing pays for writing less, so it keeps
rising until replies start to pass the cap. A cut-off reply ends its rollout
with nothing handed in, so groups regain contrast (right against wrong),
and that pushes the length back down. The agent settles where the two
pushes balance: long replies, a few cut off, fewer calls, fewer right.

The plan expected more: tokens per reply climbing past the cap and accuracy
collapsing. With a length learning rate of 0.3 that never happened, in this
seed or the four others (below). The section after the seed tables raises
the learning rate.

## A bonus for fewer tokens

```bash
uv run reward_lab.py --reward token-bonus
```

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     16%         59%       68%       76%
  easy                      27%         79%       96%       98%
  medium                    19%         68%       90%       93%
  hard                      2%          25%       18%       30%
reward                      0.17        0.61      0.71      0.81
calls per rollout           4.7         5.3       6.0       5.8
tokens per reply            56          65        36        24
replies cut off             1%          2%        0%        0%
replies holding 2 actions   8%          18%       2%        0%
rollouts that guessed       26%         4%        1%        1%
typical reply at the end    55          59        22        20

greedy test at the end (the most likely action, every reply 20 tokens)
  easy    training 15 of 15  held out 8 of 8 · LOOK x3, QUERY_A, SUBMIT
  medium  training 16 of 16  held out 6 of 6 · LOOK x3, QUERY_A, SUBMIT
  hard    training  0 of 14  held out 0 of 4 · LOOK x3, QUERY_A x5
```

Paid for fewer tokens, the agent writes the shortest replies it can: 24
tokens per reply in the last quarter, against 33 when paid for right answers
only, and nothing cut off. Long replies no longer pay: a reply long enough
to hold two actions costs more tokens than two short replies. But it is
right 76% of the time instead of 86%, and the gap is mostly on hard
questions, 30% instead of 67%, the questions that need the most calls. The
token bonus pays for fewer calls too, since every call costs at least 20
tokens, and the agent has one set of weights for every level: a habit of
calling less, learned where it is cheap, is carried to the questions where
it is not. The call bonus cost the hard questions the same way (42%).

## Not one lucky seed

```bash
uv run seeds.py
```

```text
seeds 0-4 · 200 steps per run · length lr 0.3
each cell: the training rollouts of steps 1-20 -> those of steps 151-200
calls: of a budget of 8 per rollout; tokens/reply: a cut-off reply counts 120

correct: 1 if right, else 0
seed  correct        reward         calls          tokens/reply   cut off
 0    3% -> 86%      0.03 -> 0.86   3.6 -> 6.1     51 -> 33       0% -> 0%
 1    0% -> 82%      0.00 -> 0.82   3.2 -> 6.3     42 -> 28       0% -> 0%
 2    2% -> 82%      0.02 -> 0.82   3.2 -> 5.9     48 -> 52       0% -> 1%
 3    0% -> 84%      0.00 -> 0.83   3.2 -> 5.8     41 -> 49       0% -> 0%
 4    6% -> 88%      0.06 -> 0.89   3.8 -> 5.9     50 -> 56       1% -> 0%
all   2% -> 84%      0.02 -> 0.84   3.4 -> 6.0     47 -> 43       0% -> 0%
claim: nothing rewards fewer calls, so they wander up toward the budget
  test: more calls per rollout at the end than at the start
  holds in 5 of 5 seeds

submit: 1 if right, else 0, plus 0.2 for submitting any answer
seed  correct        reward         calls          tokens/reply   cut off
 0    1% -> 0%       0.19 -> 0.20   2.3 -> 1.1     47 -> 36       0% -> 0%
 1    0% -> 0%       0.17 -> 0.20   2.2 -> 1.2     37 -> 23       0% -> 0%
 2    1% -> 0%       0.18 -> 0.20   2.1 -> 1.2     40 -> 38       0% -> 0%
 3    0% -> 0%       0.18 -> 0.20   2.0 -> 1.0     37 -> 24       0% -> 0%
 4    1% -> 0%       0.18 -> 0.20   2.1 -> 1.5     47 -> 53       0% -> 0%
all   0% -> 0%       0.18 -> 0.20   2.1 -> 1.2     42 -> 36       0% -> 0%
claim: training reward rises while accuracy falls
  test: reward higher, and share right lower, at the end than at the start
  holds in 3 of 5 seeds

call-bonus: if right, 1 + up to 0.1 for fewer calls; else 0
seed  correct        reward         calls          tokens/reply   cut off
 0    3% -> 76%      0.03 -> 0.80   3.6 -> 5.0     51 -> 65       0% -> 2%
 1    0% -> 68%      0.00 -> 0.71   3.2 -> 5.4     42 -> 46       0% -> 2%
 2    2% -> 74%      0.02 -> 0.79   3.2 -> 5.0     48 -> 64       0% -> 2%
 3    0% -> 75%      0.00 -> 0.79   3.2 -> 4.8     41 -> 68       0% -> 2%
 4    6% -> 75%      0.07 -> 0.79   3.8 -> 4.7     50 -> 69       1% -> 3%
all   2% -> 74%      0.02 -> 0.78   3.4 -> 5.0     47 -> 62       0% -> 2%
claim: tokens per reply climb
  test: longer replies at the end than the correct-only run of the same seed
  holds in 5 of 5 seeds
claim: replies pass the cap and accuracy collapses
  test: at the end, over 50% of replies cut off and under 10% of rollouts right
  holds in 0 of 5 seeds

token-bonus: if right, 1 + up to 0.1 for fewer tokens; else 0
seed  correct        reward         calls          tokens/reply   cut off
 0    3% -> 76%      0.03 -> 0.81   3.6 -> 5.8     51 -> 24       0% -> 0%
 1    0% -> 71%      0.00 -> 0.76   3.2 -> 5.9     42 -> 23       0% -> 0%
 2    2% -> 84%      0.02 -> 0.89   3.2 -> 6.0     48 -> 24       0% -> 0%
 3    0% -> 76%      0.00 -> 0.81   3.2 -> 5.8     41 -> 24       0% -> 0%
 4    6% -> 69%      0.06 -> 0.74   3.8 -> 5.7     50 -> 23       1% -> 0%
all   2% -> 75%      0.02 -> 0.80   3.4 -> 5.8     47 -> 24       0% -> 0%
claim: short
  test: shorter replies at the end than at the start, and none cut off
  holds in 5 of 5 seeds
claim: and correct
  test: share right at the end within 5 points of the correct-only run of the same seed, or better
  holds in 1 of 5 seeds

bonus-when-wrong: 1 if right, else 0, plus up to 0.1 for fewer tokens
seed  correct        reward         calls          tokens/reply   cut off
 0    0% -> 0%       0.09 -> 0.09   1.6 -> 1.0     29 -> 23       0% -> 0%
 1    0% -> 0%       0.09 -> 0.09   1.6 -> 1.0     31 -> 23       0% -> 0%
 2    0% -> 0%       0.09 -> 0.09   1.7 -> 1.0     26 -> 23       0% -> 0%
 3    1% -> 0%       0.09 -> 0.09   1.8 -> 1.0     35 -> 23       0% -> 0%
 4    1% -> 0%       0.09 -> 0.09   1.7 -> 1.0     33 -> 23       0% -> 0%
all   0% -> 0%       0.09 -> 0.09   1.7 -> 1.0     31 -> 23       0% -> 0%
claim: it gives up early
  test: at the end, under 1.5 calls per rollout and none right
  holds in 5 of 5 seeds

```

Each table has one row per seed and an `all` row that adds the five seeds
together. Each cell is the training rollouts of the first 20 steps -> those
of the last 50. Under each table, every claim the course plan makes about
that rule, the test it was checked with, and in how many seeds it held:

1. **Correct only: calls wander up toward the budget.** In every seed, from
   3.2-3.8 calls to 5.8-6.3, of 8.
2. **+0.2: training reward rises while accuracy falls.** The reward rose in
   every seed (0.17-0.19 -> 0.20) and the agent ended right 0% of the time in
   every seed, while the same seeds paid for right answers only reached
   82-88%. The test holds in 3 of 5 only because seeds 1 and 3 had not been
   right once in their first 20 steps, so their accuracy had nothing to fall
   from: from zero weights, the guess wins before the agent learns anything.
3. **Call bonus: tokens per reply climb.** Holds in every seed: 46-69 tokens
   in the last quarter against 28-56 for correct only, with fewer calls
   (4.7-5.4 against 5.8-6.3), more cut off (2-3% against 0-1%) and fewer
   right (68-76% against 82-88%). **Replies pass the cap and accuracy
   collapses: holds in no seed.**
4. **Token bonus: short.** Every seed, 23-24 tokens per reply and nothing
   cut off. **And correct:** within 5 points of correct only in 1 seed of 5
   (69-84% against 82-88%).
5. **Bonus when wrong: it gives up early.** Every seed, one call per rollout
   and nothing right.

## A higher learning rate

The build notes name the mechanism for the call-bonus failure: a learned
reply length and a high learning rate. The default 0.3 lets the agent settle
under the cap. This raises it to 1, over ten seeds, for the three rules that
matter here:

```bash
uv run seeds.py --seeds 10 --length-lr 1 --rules correct,call-bonus,token-bonus
```

```text
call-bonus: if right, 1 + up to 0.1 for fewer calls; else 0
seed  correct        reward         calls          tokens/reply   cut off
 0    1% -> 0%       0.01 -> 0.00   2.4 -> 1.2     72 -> 117      10% -> 83%
 1    0% -> 72%      0.00 -> 0.75   3.2 -> 5.4     42 -> 41       0% -> 2%
 2    2% -> 64%      0.03 -> 0.68   3.4 -> 5.3     49 -> 46       0% -> 2%
 3    0% -> 76%      0.00 -> 0.79   3.2 -> 5.7     41 -> 26       0% -> 0%
 4    2% -> 68%      0.02 -> 0.71   2.5 -> 5.6     72 -> 42       12% -> 1%
 5    1% -> 0%       0.01 -> 0.00   3.5 -> 1.0     40 -> 119      0% -> 96%
 6    3% -> 63%      0.03 -> 0.67   2.9 -> 4.5     49 -> 55       3% -> 5%
 7    3% -> 68%      0.03 -> 0.72   3.4 -> 5.0     49 -> 44       0% -> 2%
 8    6% -> 74%      0.06 -> 0.78   3.9 -> 5.3     42 -> 44       1% -> 1%
 9    1% -> 71%      0.01 -> 0.75   3.4 -> 5.3     42 -> 49       0% -> 2%
all   2% -> 56%      0.02 -> 0.58   3.2 -> 4.4     49 -> 47       2% -> 6%
claim: tokens per reply climb
  test: longer replies at the end than the correct-only run of the same seed
  holds in 7 of 10 seeds
claim: replies pass the cap and accuracy collapses
  test: at the end, over 50% of replies cut off and under 10% of rollouts right
  holds in 2 of 10 seeds
```

Now the call bonus collapsed in two seeds of ten: replies of 117 and 119
tokens, 83% and 96% of them cut off, nothing right. Read how each one
happened, though. Seed 5 collapsed under all three rules alike (the correct
only and token bonus tables in the same output show the same row): it never
had two right rollouts in one group, and until it does every rule pushes the
same way, so the three runs made exactly the same calls, rollout for
rollout. At step 58 one update took its typical reply from 85 tokens to
197, and no bonus played a part. Seed 0 collapsed under
the call bonus alone: at step 41 one update took the typical reply from 68
tokens to 158, and from then on nearly every reply was cut off, every group
was all wrong, and nothing could pull it back. The other two rules reached
83% and 79% on seed 0.

So the toy shows the mechanism, but not that the call bonus causes it: a
learning rate high enough to throw replies past the cap in one update can
do it to any rule, and one extra seed of ten is too few to blame the bonus.
The steady effect of the call bonus is the one in the five-seed table:
longer replies, fewer calls, fewer right.

## Guessing

```bash
uv run seeds.py --guessable --rules correct
```

```text
seeds 0-4 · 200 steps per run · length lr 0.3 · guessable questions in
each cell: the training rollouts of steps 1-20 -> those of steps 151-200
calls: of a budget of 8 per rollout; tokens/reply: a cut-off reply counts 120

correct: 1 if right, else 0
seed  correct        reward         calls          tokens/reply   cut off
 0    12% -> 14%     0.12 -> 0.14   2.9 -> 1.4     51 -> 44       0% -> 0%
 1    14% -> 83%     0.14 -> 0.83   3.2 -> 6.1     44 -> 48       0% -> 0%
 2    11% -> 15%     0.11 -> 0.15   2.9 -> 1.4     46 -> 47       0% -> 0%
 3    9% -> 15%      0.09 -> 0.15   2.9 -> 1.2     37 -> 29       0% -> 0%
 4    8% -> 14%      0.08 -> 0.14   3.1 -> 1.5     46 -> 38       0% -> 0%
all   11% -> 28%     0.11 -> 0.28   3.0 -> 2.3     45 -> 44       0% -> 0%
claim: nothing rewards fewer calls, so they wander up toward the budget
  test: more calls per rollout at the end than at the start
  holds in 1 of 5 seeds
```

`--guessable` puts back the questions a guess of 0 answers: Brussels'
orders, and a seventh question, "How many orders from {city} were paid in
cash?", whose answer is 0 for every city because nobody in the shop pays in
cash. A guess is now right for 8 of the 53 training questions, 15%. Paid
for right answers only, four seeds of five learned to guess and nothing
else: 14-15% right, 1.2-1.5 calls per rollout (so the calls claim fails
here: a guess ends the rollout at once). Seed 1 learned the job before
guessing took over, and reached 83%.

```bash
uv run reward_lab.py --guessable
```

```text
the training rollouts, in each quarter of the run
                            steps 1-50  51-100    101-150   151-200
correct                     12%         16%       15%       14%
  easy                      1%          0%        0%        0%
  medium                    27%         36%       33%       31%
  hard                      0%          0%        0%        0%
reward                      0.12        0.16      0.15      0.14
calls per rollout           2.2         1.4       1.4       1.4
tokens per reply            47          46        44        44
replies cut off             0%          0%        0%        0%
replies holding 2 actions   2%          1%        1%        1%
rollouts that guessed       80%         100%      100%      100%
typical reply at the end    45          42        42        42

greedy test at the end (the most likely action, every reply 42 tokens)
  easy    training  0 of 15  held out 0 of 8 · SUBMIT_GUESS
  medium  training  8 of 24  held out 4 of 10 · SUBMIT_GUESS
  hard    training  0 of 14  held out 0 of 4 · SUBMIT_GUESS
```

On seed 0 every rollout guesses from the second quarter on. The guess is
right on a third of the medium questions (the cash ones), far more than an
untrained agent gets by querying, so it wins. A reward that a guess can earn
teaches guessing. This is why the project's question generator rejects
answers that guessing could hit (`mhrl/tasks/questions.py`), and why this
chapter leaves out questions answered by 0.

## When the sandbox dies

```bash
uv run reward_lab.py --crashes 0.1
uv run reward_lab.py --crashes 0.1 --unscored
```

`--crashes 0.1` makes one rollout in ten lose its sandbox just before the
grader reads answer.txt (the same step, group and rollout numbers both
times, from a random stream of their own), and the two runs treat that
differently:

```text
sandboxes that died before grading: 161 of 1,600 rollouts
  84 of them had handed in the right answer
  scored 0: 77 of those right rollouts sat in a group with contrast,
  so every decision they made was pushed down
```

```text
sandboxes that died before grading: 161 of 1,600 rollouts
  89 of them had handed in the right answer
  left unscored: nothing was learned from any of them
```

Scored 0, a dead sandbox looks like a wrong answer. 84 of the 161 had in fact
handed in the right answer, and 77 of those sat in a group where something
else was right too, so every decision they made was pushed down as if it had
been a mistake. The article leaves such rollouts unscored, out of their
group; this project grades them as wrong, a known simplification. In this
seed the scored-0 run ended 68% right and the unscored one 80% (both counting
the dead sandboxes as not right).

## Our first real RL run

```bash
uv run real_run.py
```

```text
the project's first real RL run (attempt 1): LFM2.5-2.6B trained with
GRPO inside mini-swe-agent and Claude Code on one MacBook, 2 tasks x 4
rollouts per step
reward: 1 + the call bonus (tool_bonus, at most 0.1) if right, else 0
  checked: 21 of 21 right rollouts were paid exactly 1 + tool_bonus(calls, 8),
  and 0 wrong rollouts were paid anything

step  correct  calls  tokens/call  replies at the 1,024-token cap
   1      38%    5.9           46  0 of 47
   2      38%    5.5           38  0 of 44
   3      62%    6.8           40  0 of 53
   4      50%    4.6           57  0 of 37
   5      25%    3.0          127  0 of 24
   6      25%    4.1          105  0 of 33
   7      12%    3.4          365  6 of 26
   8      12%    3.8          404  15 of 30

tokens/call as the project's `python -m mhrl.cli steps` works it out
from: the project's runs/main/grpo_attempt1/metrics.jsonl and
rollouts.jsonl (token ids left out), copied into data/real_run/
```

The project's first GRPO run on LFM2.5-2.6B paid exactly the call bonus: the
script checks every rollout's reward against `1 + tool_bonus(calls, 8)`. In
eight steps, tokens per call went from 46 to 404 while calls fell from 5.9
to 3.0-3.8, and the share of rollouts right fell from 38% to 12%. By step 8
half the replies (15 of 30) stopped at the engine's 1,024-token output cap,
before their tool call. The checkpoint the run kept, from step 4, solved 0
of 24 tasks in Claude Code. Where that is recorded, in the project's
repository:

- `README.md`, "RL, attempt 1, went wrong": replies from 46 to 404 sampled
  tokens per call in eight steps, the share of correct rollouts falling, the
  kept checkpoint at 0% in Claude Code;
- `runs/logs/take3_steps.log`: the `python -m mhrl.cli steps` table above,
  as the project printed it;
- `runs/main/grpo_attempt1/metrics.jsonl` and `rollouts.jsonl`: the run's
  own records, copied here;
- `runs/main/summary_rl_attempt1.json`: the kept checkpoint's evaluation
  (Claude Code 0 of 24);
- `mhrl/rollout.py`, `token_bonus`'s docstring: "In our first RL run that
  bonus traded calls for longer replies";
- `mhrl/cli.py`, `GRPO_LR`: "4e-5 (no bias correction) ballooned replies by
  step 5", and `mhrl/config.py`, `length_guard_tokens`: the fix, a smaller
  learning rate and a guard that stops training when replies pass 150 tokens
  per call.

That is the failure this chapter's call bonus is about, and the real run
went further than the toy: past the cap, all the way down. The toy shows the
same direction (longer replies, fewer calls, more cut off, fewer right) and,
at a high enough learning rate, the same collapse, but it does not show the
bonus causing the collapse on its own. The real run also changed more than
the toy does between its two attempts. Attempt 2 paid for fewer tokens, used
a learning rate a quarter as large and added the length guard, and it ran
after fixes to the trainer: float32 log-probabilities in the engine, AdamW
bias correction (attempt 1 had none), training each shared prefix once and a
narrower sandbox rule (the project's README, "Reproducing it exactly").
Chapter 26 replays both runs.

## The run's log, and its curves

Every step appends one line to `metrics.jsonl`. The fields the project's
trainer also logs keep its names (`accuracy`, `mean_reward`, `mean_calls`,
`groups_without_contrast`, `importance_ratio_mean`), and each line adds the
counts plot.py needs: rollouts, how many were right and paid, replies, their
tokens, how many were cut off, and the typical length. Every rollout, call by
call, goes into `rollouts.jsonl`. Same seed, same files: every run prints
both md5s, and `--trace` only prints more.

```text
wrote ~/.cache/mhrl-course/07-reward-design/runs/call-bonus-seed0-lr3-length-lr0.3-steps200/metrics.jsonl
md5   0d5a8c35542b969b88d7605ca62eb9e3
wrote ~/.cache/mhrl-course/07-reward-design/runs/call-bonus-seed0-lr3-length-lr0.3-steps200/rollouts.jsonl
md5   b01bd13183dc9cd7bb854288af04a8dd
redraw it: uv run plot.py ~/.cache/mhrl-course/07-reward-design/runs/call-bonus-seed0-lr3-length-lr0.3-steps200/metrics.jsonl
```

```bash
uv run plot.py ~/.cache/mhrl-course/07-reward-design/runs/call-bonus-seed0-lr3-length-lr0.3-steps200/metrics.jsonl
```

```text
200 steps, 1,600 rollouts; each point is the average of the 20 steps up to it

share right (c, 1 = every rollout) and reward (r), * where they meet
  1.2|
     |
     |
     |                              r    r     ****r
     |                           r**c****c*****    c****
  0.6|                  rr*******c
     |             r****cc
     |           **c
     |         **
     |      ***
  0.0|******
     +--------------------------------------------------
      1                                              200  step

tokens per reply (t); the top row (-) is the output cap, 120
  120|--------------------------------------------------
     |
     |
     |
     |             tttttt           tttttttt    tt     t
   60|     tttttttt      ttttttttttt        tttt  ttttt
     |ttttt
     |
     |
     |
    0|
     +--------------------------------------------------
      1                                              200  step

calls per rollout (n); the top row is the budget, 8
    8|
     |
     |
     |          nn      nnnnnnnn nnn
     |       nnn  nnnnnn        n   nnn    nnnnnnnnnnnnn
    4|     nn                          nnnn
     | nnnn
     |n
     |
     |
    0|
     +--------------------------------------------------
      1                                              200  step
```

plot.py redraws a run from `metrics.jsonl` alone, as text, with Python's
standard library only. Each point averages the last tenth of the run. Here
the call-bonus run: the share right and the reward rise together (the
reward a little above, by the bonus); tokens per reply climb from about 45
to about 65 and stay well below the cap; calls rise and then come back to
about 5.

## What the toy showed, and where it differed from the plan

- **Correct only, +0.2 for submitting, a bonus when wrong:** as planned, in
  every seed. For +0.2, "accuracy falls" is from almost nothing: from zero
  weights the guess wins before the agent learns to answer, so the seed test
  holds in 3 of 5.
- **Call bonus:** replies grew longer than under correct only in every seed,
  with fewer calls, more replies cut off and fewer right. They did not climb
  past the cap, and accuracy did not collapse, at the default length
  learning rate. At a learning rate of 1 the call bonus collapsed in 2 seeds
  of 10, but one of the two collapsed under every rule, so the toy cannot
  say the bonus caused it.
- **Token bonus:** short in every seed; as correct as correct only in 1 seed
  of 5. It cost hard questions, because it pays for fewer looks too.
- **The mechanism needed more than the plan said.** The plan specified a
  learned mean reply length and a high learning rate. Each of these was also
  needed, and each was measured before it was kept:
  - `SUBMIT` hands in and ends the rollout. With chapter 6's `WRITE` and
    `FINISH`, the agent stopped trying `FINISH` long before any group was
    all right, ran every rollout to the budget, and a bonus for fewer calls
    had nothing to pay for.
  - A cut-off reply ends the rollout. When it only cost a call, correct only
    grew its replies too (two actions per call fit more into 8 calls) and
    the rules could not be told apart.
  - Reply lengths vary by a share (log scale) rather than a fixed number of
    tokens. With a fixed spread, a bonus for fewer tokens pays for long
    two-action replies once the typical length passes about 65 tokens.
  - The typical length cannot go below 20 tokens. Below the shortest reply,
    no reply differs, so the length could drift down with nothing to bring
    it back.
  - The guess is 0 and no question has 0 for an answer. When the guess was
    the most common answer (right for 2 of 45 questions, which the
    project's rules allow), even correct only sometimes learned to guess.
  - One set of weights for every level (above).
- **Guessing and crashes** are kept short, as the build notes ask: the
  rejection rules for guessable answers belong to chapter 1, and
  infrastructure failures to a later chapter.

## What is in this folder

- `reward_lab.py`: GRPO under one reward rule on the tiny agent: the steps,
  the replies, the guess, the crashes, the logs, and `--show-group` and
  `--trace`.
- `rewards.py`: the five rules, with the project's `tool_bonus`,
  `token_bonus` and `reward_for` line for line.
- `replies.py`: the reply length: a bell curve on a log scale around the
  learned typical length.
- `grpo.py`: chapter 6's `group_advantages` and `loss_fn`, now over action
  and length decisions, with the gradient worked out by hand.
- `environment.py`: chapter 6's environment, with `SUBMIT`, `SUBMIT_GUESS`,
  replies that hold one or two actions or are cut off, and crashes.
- `questions.py`: chapter 6's questions, the guess, and the guessable ones.
- `report.py`: how runs are printed, and the seed tables.
- `seeds.py`: every rule over several seeds, side by side on the Mac's cores,
  with each claim checked.
- `plot.py`: the curves, drawn as text from `metrics.jsonl` alone (standard
  library only).
- `real_run.py`, `data/real_run/`: the project's first RL run, from its own
  records.
- `policy.py`, `grader.py`, `data/shop.db`: chapter 6's, copied unchanged.
- `tasks.py`: chapter 6's `tasks.py`, writing under `07-reward-design`.

## Words from this chapter

- **Reward rule**: what a rollout is paid, worked out from what happened in
  it. RL raises whatever it measures.
- **Reward hacking**: raising the reward by a route nobody meant to pay for,
  such as guessing for a submission bonus or stopping early for a bonus paid
  even when wrong.
- **Efficiency bonus**: a small extra for doing the job cheaply, in calls
  (the article's) or in tokens (the project's).
- **Bonus only when right**: paying an efficiency bonus only on a right
  answer, so it cannot be bought by giving up.
- **Reply length, output cap, cut off**: how many tokens a reply has; the
  most the engine will write; a reply that reached the cap before its tool
  call.
- **Typical reply length, length weight**: the length the agent's replies
  are drawn around, and its log, which the agent learns.
- **Length blow-up**: replies growing longer and longer under training,
  until they are cut off.
- **Guessing**: answering without looking, which pays when answers repeat.
- **Infrastructure failure**: a rollout lost to the sandbox, not to the
  agent; scored 0 or left unscored.

Next: [chapter 8](../08-sft-toy/), Imitation (SFT) on the toy.
