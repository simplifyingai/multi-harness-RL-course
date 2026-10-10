# Chapter 2. Plain RL: learning which query to trust

Here is an agent (a program that makes choices) that knows no SQL at all. For
each question it gets four ready-made queries: one is right, and three are
mistakes that look right. Nobody tells it which is which. After each try it
hears only chapter 1's grader: reward 1 for a right answer, 0 for a wrong one.
Can it learn which query to trust?

It can, in a few hundred tries, and afterwards it answers questions about
cities it has never seen. Learning from rewards like this, by trying, is called
**reinforcement learning (RL)**. This chapter is the smallest RL there is: no
language model, just a table of 16 numbers.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/02-plain-rl
uv run bandit.py
```

It takes a few seconds on any Mac. The script needs NumPy (a Python library for
tables of numbers), and `uv run` installs it for this script alone; the rest is
Python's standard library. Like chapter 1, it writes only under `COURSE_HOME`
(default `~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `bandit.py` | 500 tries: some of the agent's picks, its chances after 0, 50, 200 and 500 tries, then a test on cities it never trained on |
| `bandit.py --temperature 0` | an agent that always takes its current favourite, and never learns |
| `bandit.py --temperature 3` | an agent that picks much more at random, and learns slowly |
| `bandit.py --lr 30` | huge nudges: one reward decides each question |
| `bandit.py --queries` | the four questions, their 16 queries, and which questions were left out |
| `bandit.py --words` | the RL words, mapped onto this code |

Other flags: `--seed N` for a different run, `--episodes N` for the number of
tries (RL calls one try an **episode**), `--show N` for how many tries to print
after each table, and `--fast` to skip the pauses.

## The job, shrunk to four choices

The job is chapter 1's: a question about the shop database, answered in
`answer.txt`. Here there are four questions, each with a city in it, and each
comes with four candidate queries. `uv run bandit.py --queries` prints them all;
here is the first question:

```text
How many customers live in {city}?
  A  SELECT COUNT(*)
       FROM customers
     -- no WHERE filter: counts every customer
  B  SELECT COUNT(*)
       FROM customers
       WHERE city = '{city}'
     -- right: counts the customers in the city
  C  SELECT SUM(quantity)
       FROM order_items
       JOIN orders ON orders.id = order_items.order_id
       JOIN customers ON customers.id = orders.customer_id
       WHERE customers.city = '{city}'
     -- SUM instead of COUNT: adds up the items they bought
  D  SELECT COUNT(*)
       FROM orders
       JOIN customers ON customers.id = orders.customer_id
       WHERE customers.city = '{city}'
     -- the JOIN counts their orders, not the customers
```

The right query is B here, and D, A and C for the other three questions, so no
letter is a safe bet. Query D is the right query for "How many orders did
customers from {city} place?", yet a mistake here: a query is only right or
wrong for a question.

Two rules keep the questions fair. First, as in chapter 1, a question is left
out when a mistake would also be rewarded. Paris has 13 customers, and they
placed 13 orders, so query D would earn a reward there for the wrong reason.
Brussels' customers placed no orders, and the shop's very first order came from
Vienna; five questions are left out in all, and `--queries` lists them. Second,
every third city in alphabetical order is kept back for the final test:

```text
training cities: Amsterdam, Berlin, Copenhagen, Dublin, Madrid, Paris, Rome, Vienna
held out for the test: Brussels, Lisbon, Prague, Warsaw
30 training questions, 13 held-out questions
```

## The agent: a table of 16 scores

The agent keeps one score for each question and query: a 4 x 4 table, all 0 at
the start. Those 16 numbers are the agent's **weights** (also called
**parameters**): the numbers that learning changes. A language model is the same
idea at a vast scale: LFM2.5-2.6B, the model this course trains later, has about
2.6 billion of them.

A row of scores becomes a row of chances with **softmax**: raise e (about 2.718)
to the power of each score, then divide each result by their total. The chances
are all positive and add up to 1 (100%). Equal scores give equal chances, which
is why every row starts at 25% each, and a higher score gets a bigger share. A
language model does the same with its raw scores for every possible next piece
of text; those raw scores are called **logits**.

The rule that turns a situation into chances for each action is the agent's
**policy**. Here the policy is the score table plus softmax, so changing the
policy means changing the weights.

## One try

1. The agent is given a random training question, say "When was the first order
   from Copenhagen?".
2. It picks a query at random, by the chances in that question's row. Picking at
   random by the chances is called **sampling**: a query with a 25% chance is
   picked about one time in four.
3. The query runs on a fresh copy of shop.db in a new folder, and its result is
   written to `answer.txt`.
4. Chapter 1's grader, copied here unchanged, compares `answer.txt` with the gold
   answer: what the right query returns for that city. Its verdict is the
   reward, 1 for correct and 0 for wrong. A rule that turns an attempt into a
   number like this is called the **reward function**. (The project's reward
   later adds a small bonus for efficiency; here it is only 1 or 0.)
5. The agent nudges that question's row (next section), and the next try begins.

See the question, act, get a reward, adjust, again and again: that is the **RL
loop**. The default run prints the first three tries of each stretch:

```text
try 1 · When was the first order from Copenhagen?
        picked D: SELECT MIN(order_date) FROM orders → 2024-01-08 · wrong · reward 0
try 2 · How much did customers from Madrid spend?
        picked C: SELECT SUM(unit_price) ... → 4910.63 · wrong · reward 0
try 3 · How much did customers from Amsterdam spend?
        picked D: SELECT COUNT(*) ... → 59 · wrong · reward 0
tries 1-50: 23 rewarded (46%)
```

## The nudge

After each try, the question's row changes by

```text
row += lr * reward * (onehot(choice) - probabilities)
```

- `onehot(choice)` is 1 for the query that was picked and 0 for the other three.
- `probabilities` are the row's chances just before the nudge.
- `lr`, the **learning rate**, sets how big a nudge is (0.25 by default).

A reward of 1 raises the picked query's score by lr x (1 - its chance) and
lowers every other score by lr x its chance: what the picked query gains, the
others lose. A reward of 0 changes nothing at all, so a wrong try teaches this
agent nothing. Chapter 3 shows where this rule comes from.

## What to look for

Every row starts at 25% each (`*` marks the right query; `tries` counts how often
each question has come up). Rewards raise the right query's score, the agent
picks it more often, and that brings more rewards:

```text
after 0 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                    25   *25    25    25      0
How many orders did customers from {city} place?      25    25    25   *25      0
How much did customers from {city} spend?            *25    25    25    25      0
When was the first order from {city}?                 25    25   *25    25      0

tries 1-50: 23 rewarded (46%)

after 50 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                    15   *54    15    15     14
How many orders did customers from {city} place?      20    20    20   *40      9
How much did customers from {city} spend?            *69    10    10    10     18
When was the first order from {city}?                 20    20   *40    20      9

tries 51-200: 110 rewarded (73%)

after 200 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                     7   *79     7     7     41
How many orders did customers from {city} place?       3     3     3   *90     48
How much did customers from {city} spend?            *92     3     3     3     58
When was the first order from {city}?                  3     3   *91     3     53

tries 201-500: 282 rewarded (94%)

after 500 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                     1   *96     1     1    112
How many orders did customers from {city} place?       1     1     1   *97    113
How much did customers from {city} spend?            *97     1     1     1    145
When was the first order from {city}?                  1     1   *97     1    130
```

By 500 tries the right query has 96% or 97% in every row. These are the weights
behind those chances:

```text
the 16 weights: the scores softmax turns into those chances
                                                         A       B       C       D
How many customers live in {city}?                   -1.07   +3.22   -1.07   -1.07
How many orders did customers from {city} place?     -1.11   -1.11   -1.11   +3.32
How much did customers from {city} spend?            +3.56   -1.19   -1.19   -1.19
When was the first order from {city}?                -1.15   -1.15   +3.45   -1.15
```

The three mistakes in a row always share one score: they never earn a reward,
and every reward takes from them in proportion to their chances, which are
equal.

Then comes the test. Always taking the top score, with no randomness, is called
**greedy**. A language model that always takes its most likely next piece of
text is doing **greedy decoding**, and the project measures its models that way.
The test asks every held-out question greedily:

```text
greedy test (temperature 0) on cities never used in training
How many customers live in {city}?                picked B · 4 of 4 correct
How many orders did customers from {city} place?  picked D · 3 of 3 correct
How much did customers from {city} spend?         picked A · 3 of 3 correct
When was the first order from {city}?             picked C · 3 of 3 correct
held-out score: 13 of 13
```

All 13 are right, about cities the agent never trained on. It never learned
answers: it learned which query to trust, and the right query works for every
city. The situation the agent acts in, its **state**, is the question, not the
city. This is not a lucky seed: seeds 0 to 19 all end with 13 of 13, with the
right query at 95% to 98% in every row after 500 tries. This loop prints the 20
test scores:

```bash
for seed in $(seq 0 19); do uv run bandit.py --fast --seed $seed | grep held-out; done
```

## Turn the knobs

The **temperature** reshapes the chances before each pick: 1 leaves them as they
are, above 1 flattens them toward equal, below 1 sharpens them toward the top
score, and 0 means always take the top score (greedy). It changes only how a
query is picked; the table and the nudge always use the plain chances.

### Temperature 0: no exploring, no learning

```bash
uv run bandit.py --temperature 0
```

All scores start equal, and when scores tie greedy takes the first one, so it
picks A in every row, every time:

```text
temperature 0: no chance involved; it always takes the top score,
and when the scores tie it takes the first, A
```

For three of the questions A is a mistake. The reward is 0, so nothing changes,
so it picks A again, forever:

```text
try 201 · How many customers live in Vienna?
        picked A: SELECT COUNT(*) FROM customers → 120 · wrong · reward 0
try 202 · How many orders did customers from Vienna place?
        picked A: SELECT COUNT(*) FROM orders → 200 · wrong · reward 0
try 203 · How many orders did customers from Copenhagen place?
        picked A: SELECT COUNT(*) FROM orders → 200 · wrong · reward 0
tries 201-500: 87 rewarded (29%)

after 500 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                    25   *25    25    25    112
How many orders did customers from {city} place?      25    25    25   *25    113
How much did customers from {city} spend?            *98     1     1     1    145
When was the first order from {city}?                 25    25   *25    25    130
```

For the spend question, A happens to be the right query, so greedy is right there
from the first try. That is the tie-break's luck, not learning. The test shows
the cost:

```text
greedy test (temperature 0) on cities never used in training
How many customers live in {city}?                picked A · 0 of 4 correct
How many orders did customers from {city} place?  picked A · 0 of 3 correct
How much did customers from {city} spend?         picked A · 3 of 3 correct
When was the first order from {city}?             picked A · 0 of 3 correct
held-out score: 3 of 13
```

### Temperature 3: exploring a lot, learning slowly

```bash
uv run bandit.py --temperature 3
```

```text
tries 201-500: 143 rewarded (48%)

after 500 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                     3   *92     3     3    112
How many orders did customers from {city} place?       2     2     2   *93    113
How much did customers from {city} spend?            *95     2     2     2    145
When was the first order from {city}?                  2     2   *94     2    130
but at temperature 3 it picks the right query only 52% to 56% of the time
```

The seed fixes which questions come up, so this run sees the same questions in
the same order as the default one. But it keeps trying mistakes: 48% of tries
201 to 500 earned a reward, against 94% at temperature 1, and its chances climb
more slowly. Even at the end it picks the right query only about half the time.
Its test is still 13 of 13, because the test is greedy.

That is the trade between **exploration** (trying less likely choices, to find
out whether they are better) and **exploitation** (using the best choice found
so far). Temperature 0 never explores, so when its first pick is wrong it can
never find the right query.
Temperature 3 explores so much that it rarely uses what it has learned. The
project trains at temperature 1 and measures greedily, as this chapter does.

### Learning rate 30: one reward decides

```bash
uv run bandit.py --lr 30
```

```text
tries 1-50: 42 rewarded (84%)

after 50 tries: each query's chance (%), * = right
                                                       A     B     C     D  tries
How many customers live in {city}?                     0  *100     0     0     14
How many orders did customers from {city} place?       0     0     0  *100      9
How much did customers from {city} spend?           *100     0     0     0     18
When was the first order from {city}?                  0     0  *100     0      9

the 16 weights: the scores softmax turns into those chances
                                                         A       B       C       D
How many customers live in {city}?                   -7.50  +22.50   -7.50   -7.50
How many orders did customers from {city} place?     -7.50   -7.50   -7.50  +22.50
How much did customers from {city} spend?           +22.50   -7.50   -7.50   -7.50
When was the first order from {city}?                -7.50   -7.50  +22.50   -7.50
```

Each row jumped from 25% to 100% the first time its right query earned a reward:
30 x (1 - 0.25) = +22.50 for that query and 30 x 0.25 = -7.50 for each of the
others. After that its chance is so close to 1 that later nudges add nothing
you can see. Here the huge step does no harm, because only the right query can
ever earn a reward, and a reward of 0 never pulls a score back. Chapter 3 shows
the same jump going wrong, when a wrong choice can earn a reward too.

## Same seed, same run

Every try, table and test answer is saved as a log under `COURSE_HOME`, and the
last line is that file's md5, a fingerprint of its contents:

```text
wrote ~/.cache/mhrl-course/02-plain-rl/runs/seed0-temperature1.0-lr0.25-tries500.jsonl
md5   b484c4baf903bd15b7e968220871d912
```

Run it again and the md5 is the same. `--seed 1` is a different run:

```text
md5   0a81c713b16a92136c648ba40603bbbc
```

## RL words, mapped onto this code

```text
+-- RL words, mapped onto bandit.py ------------------------------------
| state    the question: which row of the table (the city does not count)
| action   the query it picks: A, B, C or D
| reward   the grader's verdict: 1 for a right answer, 0 for a wrong one
| policy   the score table: softmax of a row gives each query's chance
| weights  the table's 16 numbers; LFM2.5-2.6B has about 2.6 billion
+-----------------------------------------------------------------------
```

## What is in this folder

- `bandit.py`: the agent and the RL loop: softmax, the pick, the nudge, training
  and the greedy test.
- `questions.py`: the four questions and their 16 queries, the fairness rule,
  the split by city, and one try in a fresh folder.
- `report.py`: how a run is printed.
- `grader.py`: chapter 1's grader, copied unchanged (a test checks the two are
  identical).
- `tasks.py`: the parts of chapter 1's `tasks.py` that the grader needs.
- `data/shop.db`: chapter 1's shop database, copied.

## Words from this chapter

- **Reinforcement learning (RL)**: learning by trying things and being told how
  well each try went.
- **RL loop**: see the situation, act, get a reward, adjust, again and again.
- **Reward function**: the rule that turns an attempt into a number; here the
  grader, 1 or 0.
- **Weights (parameters)**: the numbers learning changes; 16 here, about 2.6
  billion in LFM2.5-2.6B.
- **Softmax**: turns scores into chances that add up to 100%. A language model's
  raw scores are its **logits**.
- **Policy**: the rule that turns a situation into chances for each action; here
  the score table plus softmax.
- **State**: the situation the agent acts in; here the question.
- **Sampling**: picking at random by the chances.
- **Temperature**: flattens (above 1) or sharpens (below 1) the chances before a
  pick; 0 means always the top score.
- **Greedy**: always the top score, no randomness; for a language model, **greedy
  decoding**.
- **Learning rate**: how big each nudge is.
- **Exploration and exploitation**: trying less likely choices to learn, versus
  using the best choice found so far.
- **Episode**: one try, from question to reward. Chapter 4 makes each try a
  longer attempt, called a rollout.
- **Multi-armed bandit**: the name for this toy. A slot machine is a "one-armed
  bandit"; facing several, with no idea which pays best, you learn by playing.
  Here each question is a row of four arms, and a toy this small lets you watch
  the learning alone, before any language model or harness.

Next: [chapter 3](../README.md#chapters), where the nudge comes from.
