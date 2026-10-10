# Chapter 4. Several calls, one reward: a tiny agent

In chapters 2 and 3 the agent made one choice, and the grader judged it at
once. The agents in chapter 1's replays work differently. They look at the
tables, run a query, save the answer, say they are done, and only then does the
grader say right or wrong. Eight calls, one reward. So which call earned it?

This chapter builds the smallest agent that works that way, trains it with
chapter 3's recipe, and finds the honest answer: nobody knows which call earned
the reward, so every call gets the same credit. Then it tests the agent on
cities it never trained on, and meets an agent that looks perfect in training
and fails every one of them.

Reinforcement learning for an agent that acts over many calls (looking things
up, running queries, writing files) and is rewarded by grading the finished
result is called **agentic RL**. It is what the rest of this course does to a
real language model.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/04-episodes
uv run minianalyst.py --trace
```

It takes a few seconds on any Mac. It needs NumPy, which `uv run` installs for
this script alone, and like every chapter it writes only under `COURSE_HOME`
(default `~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `minianalyst.py` | 600 attempts with how training goes every 50, what the agent does at the end, then a test on new cities |
| `minianalyst.py --trace` | the same, plus the first three attempts and the last one, call by call |
| `minianalyst.py --memorise` | an agent that is shown each right answer and may write it again next time |
| `minianalyst.py --lr 0` | the same 600 attempts with learning switched off |

Other flags: `--seed N` for a different run, `--rollouts N` for the number of
attempts, `--show N` for how many of the first attempts `--trace` prints, `--lr N`
for the learning rate, and `--fast` to skip the pauses.

## The job, now in several calls

The questions are chapter 2's: four kinds of question about the shop database,
each with a city in it, and the same split: eight cities to train on, and four
**held out**, kept back for the final test.
The agent now has three queries to choose from for each question instead of
four. `QUERY_A` runs the right query, and `QUERY_B` and `QUERY_C` run the first
two of chapter 2's three mistakes for that question. The agent is not told any
of this: it sees only the names of its actions.

Dropping one mistake per question makes three questions fair that chapter 2
had to leave out, because the dropped query happened to give the right answer
too (Paris's 13 customers placed 13 orders, for one). So this chapter has 32
training questions and 14 held-out ones, against chapter 2's 30 and 13.

## An environment, in six parts

Everything the agent acts in is called its **environment**. Every run starts by
naming its parts:

```text
minianalyst · seed 0 · lr 0.25 · per-question baseline · 600 rollouts
the environment, in six parts
  tasks         32 training questions (8 cities); 14 held out (4 other cities)
  actions       one per call: LOOK, QUERY_A, QUERY_B, QUERY_C, WRITE, FINISH
  observations  what a call sends back: table names, a result or an error
  state         its own folder (data.db, answer.txt) + calls used,
                schema seen, result in hand, answer written
  reward rule   chapter 1's grader on answer.txt: 1 or 0, at the end
  stop rule     FINISH, or the 8-call budget is used up
```

- **Tasks**: the questions it may be given.
- **Actions**: what it may do. The agent takes one action per **call** (also
  called a **turn**); in the real project one call is one reply from the model.
  `LOOK` lists the database's tables. `QUERY_A`, `QUERY_B` and `QUERY_C` run a
  query and hold its result. `WRITE` saves the result it holds into
  `answer.txt`. `FINISH` says "done".
- **Observations**: what each call sends back: the table names, a query's
  result, or an error.
- **State**: the situation right now. Here it is the files in the agent's folder
  plus four facts: how many calls it has used, whether it has seen the tables
  (the **schema**: the database's tables and their columns), whether it holds a
  result, and whether it has written an answer.
- **Reward rule**: chapter 1's grader, copied unchanged, judges `answer.txt`:
  1 for right, 0 for wrong. It runs once, when the attempt stops, and never
  before.
- **Stop rule**: the attempt stops at `FINISH`, or when its **budget** of 8
  calls is used up, the same budget the project gives its real agents.
  `answer.txt` is graded either way.

One rule makes looking matter. Until the agent has looked, it does not know
what the tables are called, so a query made before `LOOK` fails. The toy models
that by sending such a query to an empty database, so the error is SQLite's
own: `no such table: customers`.

## Episode, trajectory, rollout

One complete attempt at one question, from the first call to the reward, is
called an **episode**. The record of what happened in it, call after call, is
its **trajectory**. People who train language models call the same thing a
**rollout**. Three words for one thing; from here on this course says
**rollout**. (Chapter 2 called each single try an episode: a rollout of one
call.)

## A new folder for every rollout

Every rollout gets its own new folder holding a copy of the shop database as
`data.db`. Whatever one rollout writes into `answer.txt` can never be seen by
another, two rollouts running at the same time cannot get in each other's way,
and the folder is deleted once the grader has read it. The project does the
same for every attempt of its real agents.

## The policy: scores from features

In chapters 2 and 3 the policy was a table, one row of scores per question. This
agent is in a different situation at every call, so it computes its row of
scores instead of looking it up. It describes its situation with five numbers,
its **features**:

```text
the policy: softmax over the 6 actions; each score is a weighted sum of 5 features
  (always 1, schema seen, result in hand, answer written, calls used / 8)
```

"Schema seen", "result in hand" and "answer written" are 1 or 0. "Calls used /
8" goes from 0 at the first call to 7/8 at the last. "Always 1" lets the agent
have a favourite action before it has done anything. Each action's score is a
**weighted sum** of the five features: each feature times a weight of its own
for that action, all added up. So the policy has 5 x 6 = 30 weights, and softmax
turns the six scores into six chances, as in chapter 2.

The question and its city are not among the features. The agent cannot tell one
question from another, so all it can learn is an order of calls that works for
every question. That is why `QUERY_A` is the right query for all of them here;
chapter 2 already showed an agent learning a different right query per question
when it can see which question it has.

## One rollout, one reward, one advantage

Chapter 3's loss was `-advantage x log p(chosen)` for one choice. A rollout
makes many choices, and the reward comes once, for all of them:

```text
loss for each rollout: -advantage x (sum of log p of every action it took)
```

A reward that arrives rarely, here once at the very end, is called a **sparse
reward**. It says how the whole rollout went and nothing about which call made
the difference. Working out which call deserves the credit is called the
**credit-assignment problem**, and this recipe does not try: every action in the
rollout gets the rollout's one advantage. The project's training does exactly
the same to every piece of text its model writes in a rollout.

The baseline is chapter 3's per-question average: what this kind of question
has earned so far. After each rollout comes one gradient-descent **step**, one
update of the weights. (In this course "step" always means an update of the
weights; what the agent does is a call.)

`--trace` prints the first three rollouts call by call:

```text
rollout 1 · untrained: every action has the same chance, 1 in 6
question  When was the first order from Dublin?
folder    rollout-0001, new: data.db (a copy of shop.db)
  call 1  FINISH   p 0.17  advantage +0.00  → no answer.txt · reward 0
stopped   FINISH, after 1 call · no answer.txt · reward 0
folder    at the end: data.db · deleted
update    reward 0 - baseline 0.00 = advantage +0.00, for its one action
          loss = -advantage x (sum of log p, -1.792) = +0.000

rollout 2 · untrained: every action has the same chance, 1 in 6
question  How much did customers from Madrid spend?
folder    rollout-0002, new: data.db (a copy of shop.db)
  call 1  QUERY_C  p 0.17  advantage +0.00  → no such table: order_items
  call 2  WRITE    p 0.17  advantage +0.00  → nothing to write yet
  call 3  FINISH   p 0.17  advantage +0.00  → no answer.txt · reward 0
stopped   FINISH, after 3 calls · no answer.txt · reward 0
folder    at the end: data.db · deleted
update    reward 0 - baseline 0.00 = advantage +0.00, the same for all 3 actions
          loss = -advantage x (sum of log p, -5.375) = +0.000
```

The untrained agent picks every action with the same chance, 1 in 6 (`p 0.17`).
Rollout 1 gave up at once. Rollout 2 queried before looking, got SQLite's error,
had nothing to write, and finished with no `answer.txt`. Neither got a reward
until it stopped, and neither changed anything: each was the first rollout of
its kind of question, so, as in chapter 3, there was nothing to compare with and
its advantage is 0.

```text
rollout 3 · untrained: every action has the same chance, 1 in 6
question  How much did customers from Amsterdam spend?
folder    rollout-0003, new: data.db (a copy of shop.db)
  call 1  LOOK     p 0.17  advantage +1.00  → tables: customers, order_items, orders, products
  call 2  QUERY_A  p 0.17  advantage +1.00  → 4515.91
  call 3  WRITE    p 0.17  advantage +1.00  → answer.txt ← 4515.91
  call 4  LOOK     p 0.17  advantage +1.00  → tables: customers, order_items, orders, products
  call 5  WRITE    p 0.17  advantage +1.00  → answer.txt ← 4515.91
  call 6  QUERY_C  p 0.17  advantage +1.00  → 3051.81
  call 7  LOOK     p 0.17  advantage +1.00  → tables: customers, order_items, orders, products
  call 8  LOOK     p 0.17  advantage +1.00  → tables: customers, order_items, orders, products
stopped   budget used, after 8 calls · correct · reward 1
folder    at the end: answer.txt, data.db · deleted
update    reward 1 - baseline 0.00 = advantage +1.00, the same for all 8 actions
          loss = -advantage x (sum of log p, -14.334) = +14.334
```

Rollout 3 got lucky. It looked, ran the right query and saved its result, then
wandered until its budget ran out. Call 6 even ran a wrong query, but never
wrote it, so `answer.txt` still held the right answer and the grader gave 1.
The spend question had earned 0 so far (rollout 2), so the advantage is +1.00,
and all eight actions get it: the three that did the work, and the five that did
nothing or worse. That is credit assignment the simple way. It works because
over many rollouts the useful actions turn up in the rewarded ones far more
often than in the others.

## Training

```text
rollouts 1-50: 14 correct (28%) · 6.4 calls per rollout
rollouts 51-100: 41 correct (82%) · 7.6 calls per rollout
rollouts 101-150: 46 correct (92%) · 8.0 calls per rollout
rollouts 151-200: 46 correct (92%) · 8.0 calls per rollout
rollouts 201-250: 50 correct (100%) · 8.0 calls per rollout
rollouts 251-300: 48 correct (96%) · 8.0 calls per rollout
rollouts 301-350: 49 correct (98%) · 7.9 calls per rollout
rollouts 351-400: 47 correct (94%) · 7.9 calls per rollout
rollouts 401-450: 49 correct (98%) · 8.0 calls per rollout
rollouts 451-500: 49 correct (98%) · 8.0 calls per rollout
rollouts 501-550: 49 correct (98%) · 7.9 calls per rollout
rollouts 551-600: 50 correct (100%) · 8.0 calls per rollout
```

Learning starts at rollout 3, so even the first 50 rollouts are 28% correct.
For comparison, `uv run minianalyst.py --lr 0` runs the same 600 rollouts with
learning switched off: 29 of them are correct (5%), at 4.5 to 5.2 calls per
rollout, and its test scores 0 of 14.

Now look at the calls per rollout. They went up, from 6.4 to 8.0: the trained
agent uses its whole budget.

## What it learned, and what it did not

```text
the agent at the end, on a training question, greedy (always its most likely action)
  call 1  nothing done yet                            LOOK      73%   FINISH 0%
  call 2  tables seen                                 QUERY_A   95%   FINISH 0%
  call 3  tables seen, result in hand                 WRITE     89%   FINISH 0%
  call 4  tables seen, result in hand, answer written LOOK      54%   FINISH 0%
  call 5  tables seen, result in hand, answer written LOOK      50%   FINISH 0%
  call 6  tables seen, result in hand, answer written WRITE     47%   FINISH 0%
  call 7  tables seen, result in hand, answer written WRITE     50%   FINISH 0%
  call 8  tables seen, result in hand, answer written WRITE     54%   FINISH 0%
```

**Greedy** means always taking the most likely action, as in chapters 2 and 3.
The first three calls are the method: look, run `QUERY_A`, write. After that the
agent drifts between `LOOK` and `WRITE`, and `FINISH` has a 0% chance at every
call. It never learned to stop, because nothing in the reward pays for stopping:
a rollout that uses all 8 calls earns exactly the same 1 as one that finishes
after 4. Once the answer is written, the later calls change nothing, so the
reward can never tell them apart, and they drift.

The last rollout of training shows both lessons at once:

```text
rollout 600 · after 599 steps of training
question  When was the first order from Copenhagen?
folder    rollout-0600, new: data.db (a copy of shop.db)
  call 1  QUERY_B  p 0.01  advantage +0.07  → no such table: customers
  call 2  LOOK     p 0.70  advantage +0.07  → tables: customers, order_items, orders, products
  call 3  QUERY_A  p 0.96  advantage +0.07  → 2024-02-13
  call 4  WRITE    p 0.90  advantage +0.07  → answer.txt ← 2024-02-13
  call 5  WRITE    p 0.43  advantage +0.07  → answer.txt ← 2024-02-13
  call 6  LOOK     p 0.46  advantage +0.07  → tables: customers, order_items, orders, products
  call 7  WRITE    p 0.50  advantage +0.07  → answer.txt ← 2024-02-13
  call 8  LOOK     p 0.38  advantage +0.07  → tables: customers, order_items, orders, products
stopped   budget used, after 8 calls · correct · reward 1
folder    at the end: answer.txt, data.db · deleted
update    reward 1 - baseline 0.93 = advantage +0.07, the same for all 8 actions
          loss = -advantage x (sum of log p, -9.000) = +0.647
```

Its first call was a 1-in-100 mistake, a query before looking. It still got the
same +0.07 as the `LOOK`, `QUERY_A` and `WRITE` that earned the reward, and so
did the four calls after the answer was written. The advantage is small because
this kind of question now earns 0.93 on average: a right answer is only a little
better than expected.

For exactly this reason, the article this course follows adds a small bonus to
its reward for using fewer calls, and the project one for writing less. Chapter
7 tries both, and what goes wrong with them.

## The test on new cities

Splitting the questions into the ones the agent trains on and ones it never
sees, kept back for testing, is called a **train/test split**. Here the
held-out questions are about four cities that never come up in training.

```text
greedy test on the training questions: 32 of 32

greedy test on cities never used in training
every rollout made the same calls: LOOK, QUERY_A, WRITE, LOOK x2, WRITE x3
How many customers live in Brussels?                answer.txt 3           correct
How many customers live in Lisbon?                  answer.txt 3           correct
How many customers live in Prague?                  answer.txt 15          correct
How many customers live in Warsaw?                  answer.txt 11          correct
How many orders did customers from Brussels place?  answer.txt 0           correct
How many orders did customers from Lisbon place?    answer.txt 11          correct
How many orders did customers from Prague place?    answer.txt 10          correct
How many orders did customers from Warsaw place?    answer.txt 9           correct
How much did customers from Lisbon spend?           answer.txt 1804.05     correct
How much did customers from Prague spend?           answer.txt 1632.74     correct
How much did customers from Warsaw spend?           answer.txt 1358.62     correct
When was the first order from Lisbon?               answer.txt 2024-06-18  correct
When was the first order from Prague?               answer.txt 2024-06-19  correct
When was the first order from Warsaw?               answer.txt 2024-12-12  correct
held-out score: 14 of 14
```

All 14 are right, about cities it never saw, because it learned a method that
works for any city. Doing well on new questions, not only on the ones you
practised, is called **generalisation**. It is not one lucky seed: seeds 0 to 19
all end with 32 of 32 and 14 of 14.

```bash
for seed in $(seq 0 19); do uv run -q minianalyst.py --fast --seed $seed; done |
  grep -E "training questions:|held-out score" | sort | uniq -c
```

```text
  20 greedy test on the training questions: 32 of 32
  20 held-out score: 14 of 14
```

## An agent that memorises

`--memorise` changes one thing. After every rollout the agent is shown the right
answer, and it gets a seventh action, `RECALL`, which writes the answer it saw
last time for the same question, city included:

```bash
uv run minianalyst.py --memorise
```

```text
rollouts 1-50: 16 correct (32%) · 6.5 calls per rollout
rollouts 51-100: 38 correct (76%) · 7.8 calls per rollout
rollouts 101-150: 49 correct (98%) · 8.0 calls per rollout
rollouts 151-200: 50 correct (100%) · 7.9 calls per rollout
```

In training it looks even better than the first agent: 98% to 100% correct
from rollout 101 on. It learned to recall instead of working anything out:

```text
the agent at the end, on a training question, greedy (always its most likely action)
  call 1  nothing done yet                            RECALL    79%   FINISH 1%
  call 2  answer written                              RECALL    89%   FINISH 0%
```

And the test:

```text
greedy test on the training questions: 32 of 32

greedy test on cities never used in training
every rollout made the same calls: RECALL x8
and every call got the same answer: nothing remembered for this question
How many customers live in Brussels?                answer.txt (none)      no answer.txt
How many customers live in Lisbon?                  answer.txt (none)      no answer.txt
```

and so on down to `held-out score: 0 of 14`. Perfect on every question it
practised, useless on every new one. Doing well by remembering answers instead
of learning the method is called **memorisation**, and only the held-out test
can see it: on the training questions the two agents score the same 32 of 32.
Seeds 0 to 19 all end the same way:

```text
  20 greedy test on the training questions: 32 of 32
  20 held-out score: 0 of 14
```

This is why the project trains on three databases (shop, school, library) and
tests on two it never trains on (clinic, airline), and why its agents never get
to see the answers.

## Why learning rate 0.25

It is chapter 3's default, and it learned on all 20 seeds. At 0.5, 18 of the 20
seeds learn just as well, but two lock into a bad habit within 100 rollouts:

```bash
for seed in $(seq 0 19); do uv run -q minianalyst.py --fast --lr 0.5 --seed $seed; done |
  grep "held-out score" | sort | uniq -c
```

```text
   2 held-out score: 0 of 14
  18 held-out score: 14 of 14
```

They are seeds 11 and 19. From rollout 51 on, neither gets another answer
right: seed 11 ends up looking at the tables on every call, and seed 19 querying
before it looks.

```bash
for seed in 11 19; do uv run -q minianalyst.py --fast --lr 0.5 --seed $seed; done |
  grep -E "rollouts 551|held-out score"
```

```text
rollouts 551-600: 0 correct (0%) · 7.7 calls per rollout
held-out score: 0 of 14
rollouts 551-600: 0 correct (0%) · 6.8 calls per rollout
held-out score: 0 of 14
```

Chapter 3 met the same thing: a step that is too big can lock in whatever
happened to be rewarded first.

## Same seed, same file

Every run writes a log of every rollout under `COURSE_HOME` and prints its md5:

```text
wrote ~/.cache/mhrl-course/04-episodes/runs/seed0-lr0.25-rollouts600.jsonl
md5   f326e0942fe8c6646d59fa4b634015ee
```

`--trace` only prints more, so `uv run minianalyst.py` writes the same file with
the same md5. `--seed 1` is a different run, and `--memorise` writes its own
file:

```text
md5   1487491cf6474fec5ea8bba137dc818e
```

## What is in this folder

- `environment.py`: the environment: the actions, the budget, the observations,
  the state, the reward at the end and a new folder per rollout.
- `policy.py`: the features, the scores, and chapter 3's softmax, log p and
  gradient, now for a whole rollout.
- `minianalyst.py`: the agent and its training: rollouts, the per-question
  baseline, one step per rollout, the greedy tests, `--memorise` and the run
  log.
- `report.py`: how a run is printed.
- `questions.py`: chapter 2's questions with three queries each, and the split
  by city.
- `grader.py`, `data/shop.db`: chapter 1's grader and the shop database, copied
  unchanged.
- `tasks.py`: chapter 3's `tasks.py`, writing under `04-episodes`.

## Words from this chapter

- **Agentic RL**: reinforcement learning for an agent that acts over many calls
  with tools, rewarded by grading the finished result.
- **Call (turn)**: one action by the agent; for a language model, one reply.
- **Environment**: everything the agent acts in: its tasks, actions,
  observations, state, reward rule and stop rule.
- **Observation**: what one call sends back.
- **State**: the situation right now: here the folder's files and four facts
  about the rollout so far.
- **Budget**: the most calls a rollout may make, 8 here and in the project.
- **Episode, trajectory, rollout**: one complete attempt at one task, from the
  first call to its one reward. This course says rollout.
- **A folder per rollout**: every attempt gets a new folder with its own copy of
  the database, deleted afterwards.
- **Features**: the numbers an agent uses to describe its situation.
- **Weighted sum**: each number times its own weight, all added up.
- **Sparse reward**: a reward that comes rarely; here once, at the end.
- **Credit assignment**: working out which action earned a reward. REINFORCE,
  and the project's training, give every action in a rollout the same
  advantage.
- **Step**: one update of the weights; here, one after each rollout.
- **Train/test split, held out**: questions kept back from training, to test
  on.
- **Generalisation**: doing well on new questions, because the method was
  learned.
- **Memorisation**: doing well only on questions already seen, by remembering
  their answers.

Next: [chapter 5](../05-grpo-groups/), GRPO, part 1: compare attempts within a group.
