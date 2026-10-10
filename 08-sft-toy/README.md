# Chapter 8. Imitation (SFT) on the toy

Chapters 2 to 7 learned by trying. The agent made its own attempts, a reward
said how each one went, and the weights moved toward whatever had worked.
There is an older way to learn: copying. Someone who already knows how plays
the agent, every move they make is written down, and the student is trained
to make the same move in the same situation.

That is **supervised fine-tuning**, or **SFT**. "Supervised" because every
example comes with the right move attached: the teacher's. "Fine-tuning"
because it adjusts weights that already exist. The project runs SFT before
RL, so its model starts RL already knowing roughly what to do.

Copying is quick: no wasted attempts, no reward to wait for. But a student
learns whatever its examples have in common, and not all of that is the
skill. This chapter's teacher knows every answer and never makes a mistake.
One version of it always takes exactly four calls. Its student copies it
almost perfectly on questions like the ones it was shown. Then it meets a
question that needs one more look, and on call 4 it says `FINISH` with
nothing saved. What did it actually learn?

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/08-sft-toy
uv run sft_toy.py --fixed-length
```

A run takes about 3 seconds on any Mac (1 with `--fast`, which skips the
pauses; 6 with `--then-rl`). It needs NumPy, which `uv run` installs for this
script alone, and like every chapter it writes only under `COURSE_HOME`
(default `~/.cache/mhrl-course`), never into this folder.

| command | what it shows |
| --- | --- |
| `sft_toy.py` | SFT with the project's teacher, four paths (the same as `--varied`) |
| `sft_toy.py --fixed-length` | the teacher's first version: every demonstration is LOOK, QUERY_A, WRITE, FINISH |
| `sft_toy.py --fixed-length --calls-as-share` | the same, for a student that sees chapters 4 to 7's "calls used" |
| `sft_toy.py --teacher-never-checks` | a teacher that never reads answer.txt back |
| `sft_toy.py --lr-high` and `--fixed-length --lr-high` | ten times the learning rate |
| `sft_toy.py --then-rl` | then chapter 6's GRPO on the hard questions, from the imitator and from scratch |
| `seeds.py` | every claim over seeds 0 to 9 (about half a minute) |
| `plot.py <metrics.jsonl>` | a run's curves, redrawn from its log alone |
| `real_sft.py` | the project's real SFT run, from its own records |

Other flags: `--seed N`, `--lr N` (3 by default), `--rl-steps N` (100 by
default) and `--fast`. `seeds.py` takes `--seeds N` and `--rl-steps N`.

## What is new since chapter 7

**The agent is chapter 6's, with one more action.** The same six questions
at three levels, the same `LOOK` that shows one table at a time, `WRITE` and
`FINISH` as two separate actions, and 8 calls. Chapter 7's `SUBMIT` (write and
hand in, as one action) is left out, because with it "finish without saving"
cannot happen, and that is the mistake this chapter is about. So are chapter
7's reply lengths, which were there to put a price on tokens. The new action
is **`CHECK`**, which reads answer.txt back, the toy's `cat answer.txt`. In
this toy a check never changes the answer. It is there to be copied, or not.

**The student is chapter 4's softmax policy**, with chapter 7's single set of
weights for every level (a real model has one set). It sees the same four
facts as before: a constant 1, the share of tables seen, whether it holds a
result, and whether it has written an answer. The fifth fact changes. Chapters
4 to 7 gave "calls used" as one number, from 0 at the first call to 7/8 at the
last. Here it is eight switches, "call 1" to "call 8", with only the coming
call's switched on. A real model reads its whole conversation, so it knows
exactly which reply it is writing; the student should know it as sharply.
[With calls used as one number](#with-calls-used-as-one-number) shows what
the single number hid.

**The teacher knows the answer key.** The project's teacher
(`mhrl/engine/oracle.py`) is a script, not a model: the question generator
knows each task's right SQL, so the script does what a careful analyst does.
This toy's teacher knows how many tables each question needs and that
`QUERY_A` is the right query. It has the project's four paths:

| path | what the teacher does |
| --- | --- |
| direct | `LOOK` as many times as the question needs, `QUERY_A`, `WRITE`, `FINISH` |
| peek | one `LOOK` more than it needs first (the project's teacher looks at a few rows) |
| tables | `LOOK` at all four tables first (the project's teacher lists the tables, then reads them) |
| verify | direct, with a `CHECK` of answer.txt before `FINISH` |

Each question always gets the same path, picked from the letters of its id
with the project's line: `pattern = PATTERNS[sum(map(ord, task.task_id)) %
len(PATTERNS)]`. `--fixed-length` is the teacher's first version, which always
did `LOOK`, `QUERY_A`, `WRITE`, `FINISH`. `--teacher-never-checks` is the four
paths with verify's `CHECK` left out.

**Every demonstration is an easy question.** The teacher writes 120
demonstrations, the project's number, on the 15 easy training questions, 8
each. Four calls can answer only an easy question (one `LOOK`), and every
teacher in this chapter shows the same questions, so that only the paths
differ. The test is on the 19 held-out questions, about 4 cities no
demonstration used. A medium question needs one `LOOK` more than any
question the student was shown, and a hard one three more. They show whether
it learned the method or only the moves.

Every run says all this first:

```text
sft_toy · teacher varied · seed 0 · lr 3 · 120 demonstrations · batch 8 · 1 epoch
teacher: four paths: direct, peek, tables, verify (the project's teacher)
demonstrations: the 15 easy training questions, 8 each, in a shuffled order
student: chapter 4's softmax policy, one set of weights; it sees the call number
loss per call = -log p(teacher's action); what the environment sends back is never scored
held out: 19 questions about 4 other cities: 8 easy, 7 medium, 4 hard
  LOOKs a question needs before its query works: easy 1, medium 2, hard 4
```

## What the teacher shows, and what is trained

```bash
uv run sft_toy.py
```

The run starts by printing the first demonstration of each path, then what
the 120 come to:

```text
the first demonstration of each path
(left: the teacher's action, trained; right: what came back, read and never scored)
demonstration 1 (tables): How many customers live in Berlin?
  call 1  LOOK     customers: id, name, city, signup_date
  call 2  LOOK     orders: id, customer_id, order_date, payment_method
  call 3  LOOK     order_items: id, order_id, product_id, quantity, unit_price
  call 4  LOOK     products: id, name, category, price
  call 5  QUERY_A  15
  call 6  WRITE    answer.txt ← 15
  call 7  FINISH   correct · reward 1
demonstration 4 (peek): How many customers live in Paris?
  call 1  LOOK     customers: id, name, city, signup_date
  call 2  LOOK     orders: id, customer_id, order_date, payment_method
  call 3  QUERY_A  13
  call 4  WRITE    answer.txt ← 13
  call 5  FINISH   correct · reward 1
demonstration 5 (direct): When did the newest customer from Dublin sign up?
  call 1  LOOK     customers: id, name, city, signup_date
  call 2  QUERY_A  2025-05-25
  call 3  WRITE    answer.txt ← 2025-05-25
  call 4  FINISH   correct · reward 1
demonstration 8 (verify): How many customers live in Vienna?
  call 1  LOOK     customers: id, name, city, signup_date
  call 2  QUERY_A  4
  call 3  WRITE    answer.txt ← 4
  call 4  CHECK    answer.txt holds 4
  call 5  FINISH   correct · reward 1

120 demonstrations, 120 right: 120 rows (the trainer keeps right ones only)
paths: 32 direct, 24 peek, 32 tables, 32 verify
trained: 632 actions, the teacher's; never scored: 120 questions and 632 replies
```

A demonstration is a sequence: the question, then each call's action and
what came back. The trainer gives each piece a 1 or a 0, its **loss mask**: 1
for the teacher's actions, 0 for the question and for every reply. Only the 1s
are scored. The student is trained to pick the teacher's action in the
situation the teacher was in, and it is never asked to predict what the
environment would say back. Those replies are what it reads, not what it
writes. The project does the same with tokens: its capture proxy marks the
tokens the teacher's replies were made of, and everything else, the
harness's instructions and the tool outputs, is context. In the project's
real run only a tenth of the tokens were the teacher's
([below](#the-projects-real-sft-run)). `sft.py` picks the trained positions
with the project's line, the mask read from the sequence:

```python
positions = [i for i, m in enumerate(sequence.loss_mask) if m and i > 0]
```

The trainer keeps only the demonstrations the grader accepted (the project's
`demo_items` does the same). An answer-key teacher is right every time, so all
120 are kept.

## The loss: minus the log of the teacher's move

For each of the teacher's actions, the loss is `-log p(action)`: the
student's probability of the move the teacher made, under a minus and a log
(chapter 3's log p). A sure, right move costs almost nothing: p 0.99 costs
0.01. An unsure one costs more: p 0.14 costs 1.97. A move the student thought
nearly impossible costs a lot. This is the **cross-entropy loss**, also called
the **next-token loss**, since for a language model every move is a token. The
untrained student gives each of the 7 actions p 1/7, so every call costs
`-log(1/7)` = 1.95, which is step 1's loss in every run.

It is chapter 3's policy-gradient loss, `-A x log p`, with the advantage fixed
at 1 for every teacher move: always "more of this", never "less of that".
SFT lowers the other actions only by raising the teacher's.

`loss_fn` is the project's (`mhrl/training/sft.py`), with the toy's
`call_logprobs` in place of its `token_logprobs` (a test checks the two):

```python
def loss_fn(weights: np.ndarray, item: TrainItem) -> float:
    return -call_logprobs(weights, item).sum()
```

The loss of every demonstration in a **batch** is added up, and divided by
the number of the teacher's calls in the batch; then the trainer takes one
gradient-descent step (chapter 3's). A **step** is still one update. An
**epoch** is one pass over every demonstration. The project ran 120
demonstrations in batches of 8 for one epoch, so 15 steps, and so does the
toy. `batches` is the project's line for line (`mhrl/training/common.py`; a
test checks it):

```python
def batches(items: list[TrainItem], size: int, seed: int) -> Iterator[list[TrainItem]]:
    order = list(items)
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield order[start : start + size]
```

The project's optimiser is AdamW (chapter 13); here it is plain gradient
descent with a learning rate of 3, chapters 5 to 7's.

## SFT with the project's teacher

The 15 steps:

```text
step  1 · loss per call 1.95 · 39 calls in 8 rows
step  2 · loss per call 1.38 · 46 calls in 8 rows
step  3 · loss per call 1.00 · 44 calls in 8 rows
step  4 · loss per call 0.88 · 41 calls in 8 rows
step  5 · loss per call 0.78 · 41 calls in 8 rows
step  6 · loss per call 0.71 · 46 calls in 8 rows
step  7 · loss per call 0.67 · 40 calls in 8 rows
step  8 · loss per call 0.58 · 36 calls in 8 rows
step  9 · loss per call 0.67 · 50 calls in 8 rows
step 10 · loss per call 0.63 · 38 calls in 8 rows
step 11 · loss per call 0.51 · 42 calls in 8 rows
step 12 · loss per call 0.48 · 45 calls in 8 rows
step 13 · loss per call 0.46 · 38 calls in 8 rows
step 14 · loss per call 0.48 · 45 calls in 8 rows
step 15 · loss per call 0.44 · 41 calls in 8 rows
```

The loss falls from 1.95 to 0.44, and it cannot reach 0. After the first
`LOOK`, 56 of the 120 demonstrations look again (peek and tables) and 64 run
the query (direct and verify). The student cannot tell one question from
another, so at that call the best it can do is about even odds, whatever it
learns. It is the same after a `WRITE` on call 3: direct says `FINISH`, verify
says `CHECK`.

Then the test, before and after:

```text
held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    74%
medium    0 of 7    1%        7 of 7    54%
hard      0 of 4    0%        0 of 4    21%
```

The **greedy** test takes the most likely action at every call, as the
project evaluates its model. The **sampled** test picks by the probabilities
(temperature 1), 20 times per question, as RL will sample it. Untrained, the
student gets 4% of easy rollouts right by luck. After 15 steps it answers
every easy and every medium held-out question greedily, though it saw only
easy ones. Sampled, it is right 74% of the time on easy, 54% on medium and 21%
on hard: one epoch leaves plenty of chance in its probabilities.

Here is the medium question, greedily, with the probability of each action
it took:

```text
the student, greedy, on a medium question, which needs one LOOK more than any easy question it was shown
How many orders did customers from Lisbon place?
  call 1  LOOK     p 0.95  customers: id, name, city, signup_date
  call 2  QUERY_A  p 0.50  no such table: orders
  call 3  LOOK     p 0.55  orders: id, customer_id, order_date, payment_method
  call 4  LOOK     p 0.73  order_items: id, order_id, product_id, quantity, unit_price
  call 5  QUERY_A  p 0.62  11
  call 6  WRITE    p 0.84  answer.txt ← 11
  call 7  FINISH   p 0.90  correct · reward 1
```

The query fails, because orders has not been seen, and the student looks
again (twice), queries, writes and finishes. From four different paths it
learned what they share: look until the query works, write, and finish once
the answer is saved. Then the habits it shows when sampled:

```text
habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    17%      15%      15%
said FINISH with no answer.txt  23%      28%      28%
  of them on call 4             4%       5%       0%
calls per rollout               5.2      6.5      7.2
```

It reads answer.txt back in 15% to 17% of rollouts, like a teacher who checked
on a quarter of its demonstrations. It makes about 5.2 calls per easy rollout:
the teacher's 632 actions over 120 demonstrations are 5.3 each. And a sampled
rollout still says `FINISH` with nothing saved about a quarter of the time,
but on call 4 only 4% to 5% of the time.

## Every demonstration four calls long

```bash
uv run sft_toy.py --fixed-length
```

```text
the first demonstration of each path
(left: the teacher's action, trained; right: what came back, read and never scored)
demonstration 1 (fixed): How many customers live in Berlin?
  call 1  LOOK     customers: id, name, city, signup_date
  call 2  QUERY_A  15
  call 3  WRITE    answer.txt ← 15
  call 4  FINISH   correct · reward 1

120 demonstrations, 120 right: 120 rows (the trainer keeps right ones only)
paths: 120 fixed
trained: 480 actions, the teacher's; never scored: 120 questions and 480 replies
```

```text
step  1 · loss per call 1.95 · 32 calls in 8 rows
step  2 · loss per call 0.87 · 32 calls in 8 rows
step  3 · loss per call 0.49 · 32 calls in 8 rows
step  4 · loss per call 0.32 · 32 calls in 8 rows
step  5 · loss per call 0.23 · 32 calls in 8 rows
step  6 · loss per call 0.18 · 32 calls in 8 rows
step  7 · loss per call 0.15 · 32 calls in 8 rows
step  8 · loss per call 0.12 · 32 calls in 8 rows
step  9 · loss per call 0.11 · 32 calls in 8 rows
step 10 · loss per call 0.09 · 32 calls in 8 rows
step 11 · loss per call 0.08 · 32 calls in 8 rows
step 12 · loss per call 0.08 · 32 calls in 8 rows
step 13 · loss per call 0.07 · 32 calls in 8 rows
step 14 · loss per call 0.06 · 32 calls in 8 rows
step 15 · loss per call 0.06 · 32 calls in 8 rows
```

This loss falls much further, to 0.06: every demonstration is the same four
moves, so each one can be predicted. The test:

```text
held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    84%
medium    0 of 7    1%        0 of 7    14%
hard      0 of 4    0%        0 of 4    0%

the student, greedy, on a medium question, which needs one LOOK more than any easy question it was shown
How many orders did customers from Lisbon place?
  call 1  LOOK     p 0.94  customers: id, name, city, signup_date
  call 2  QUERY_A  p 0.94  no such table: orders
  call 3  WRITE    p 0.73  nothing to write yet
  call 4  FINISH   p 0.50  no answer.txt · reward 0

habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    1%       5%       6%
said FINISH with no answer.txt  14%      66%      74%
  of them on call 4             8%       46%      51%
calls per rollout               4.1      5.5      5.5
```

On easy questions the student is better than before (84% sampled). On the
medium question it looks, runs the query, gets `no such table: orders`,
writes with nothing to write, and on call 4 says `FINISH`: answer.txt missing,
reward 0. It does that on every medium question, greedily. Sampled, 66% of
its medium rollouts finish with nothing saved, 46% on call 4.

In every demonstration `FINISH` came on call 4, right after `WRITE`. Two of
the student's switches were on there each time: "answer written" and "call
4". The loss rewarded both, so it learned both. On the medium question "answer
written" is off, and "call 4" alone is enough: p 0.50 for `FINISH`, more than
any other action. To this student the 120 fixed-length demonstrations are one
demonstration seen 120 times: every easy question needs one look, and it
cannot tell the questions apart. Nothing in them ever shows that call 4 is
not the point.

> **In the project's code** (`mhrl/engine/oracle.py`, the docstring of
> `AnswerKeyTeacher._steps`): "A first version always did schema -> query ->
> write -> finish, and the student learned 'the fourth reply is the final
> answer' by POSITION: after one extra query it finished without writing
> answer.txt (dry run: Claude Code 100% -> 17%). Varying the path teaches the
> dependency instead: finish only after the answer is saved."

That is why the project's teacher, and this chapter's default, has four
paths: with `FINISH` on calls 4, 5 and 7, the only switch that always comes
before it is "answer written". This is **demonstration diversity**: the
examples have to differ in what does not matter, or the student takes it for
what does.

### With calls used as one number

```bash
uv run sft_toy.py --fixed-length --calls-as-share
```

```text
step 15 · loss per call 0.43 · 32 calls in 8 rows

held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    76%
medium    0 of 7    1%        0 of 7    51%
hard      0 of 4    0%        0 of 4    2%

the student, greedy, on a medium question, which needs one LOOK more than any easy question it was shown
How many orders did customers from Lisbon place?
  call 1  LOOK     p 0.52  customers: id, name, city, signup_date
  call 2  QUERY_A  p 0.52  no such table: orders
  call 3  QUERY_A  p 0.55  no such table: orders
  call 4  QUERY_A  p 0.57  no such table: orders
  call 5  QUERY_A  p 0.60  no such table: orders
  call 6  QUERY_A  p 0.62  no such table: orders
  call 7  QUERY_A  p 0.64  no such table: orders
  call 8  QUERY_A  p 0.66  no such table: orders

habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    6%       4%       8%
said FINISH with no answer.txt  21%      21%      21%
  of them on call 4             5%       4%       1%
calls per rollout               5.0      6.6      7.1
```

The same 120 demonstrations, with chapters 4 to 7's "calls used" as a single
number. This student never finishes early: on the medium question it runs
the failed query again until the budget runs out, and only 4% of its sampled
medium rollouts finish on call 4. It learned `FINISH` from "answer written",
the one strong signal it had, and its loss is still 0.43 after the 15 steps:
telling the four moves apart from four facts and one small number takes
longer than reading a switch. The shortcut is real, but a student has to be
able to see "call 4" to take it, which is why this chapter's student gets the
eight switches.

## A teacher that never checks

```bash
uv run sft_toy.py --teacher-never-checks
```

The same questions and paths, with verify's `CHECK` left out:

```text
held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    75%
medium    0 of 7    1%        7 of 7    66%
hard      0 of 4    0%        0 of 4    28%

the student, greedy, on a medium question, which needs one LOOK more than any easy question it was shown
How many orders did customers from Lisbon place?
  call 1  LOOK     p 0.95  customers: id, name, city, signup_date
  call 2  QUERY_A  p 0.51  no such table: orders
  call 3  LOOK     p 0.56  orders: id, customer_id, order_date, payment_method
  call 4  LOOK     p 0.75  order_items: id, order_id, product_id, quantity, unit_price
  call 5  QUERY_A  p 0.66  11
  call 6  WRITE    p 0.87  answer.txt ← 11
  call 7  FINISH   p 0.91  correct · reward 1

habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    2%       5%       6%
said FINISH with no answer.txt  19%      20%      20%
  of them on call 4             7%       7%       5%
calls per rollout               5.1      6.6      7.2
```

Greedily, in this run neither this student nor the default one checks: after
a `WRITE`, `FINISH` is the likelier move for both. The difference shows when
they are sampled: this one reads answer.txt back in 2% to 6% of rollouts,
against 15% to 17% for the student of the teacher that checked. The 2% to 6% is what is left of the untrained
student's 1 chance in 7 at every call; one epoch does not train it all away.
A teacher that never checks makes a student that almost never checks.

In this toy that costs nothing, because a check never changes the answer.
In the project it cost something real. The project's teacher always ran one
query and never double-checked, and after SFT its model, which had checked
its work in mini-swe-agent, lost 6 tasks there and gained none
([below](#the-projects-real-sft-run)). **Imitation narrows behaviour**: what
the teacher never does, the student stops doing, good habits included.

## Too big a step

```bash
uv run sft_toy.py --lr-high
```

A learning rate of 30, ten times the default, with the project's teacher:

```text
step  1 · loss per call 1.95 · 39 calls in 8 rows
step  2 · loss per call 1.52 · 46 calls in 8 rows
step  3 · loss per call 1.81 · 44 calls in 8 rows
step  4 · loss per call 2.91 · 41 calls in 8 rows
step  5 · loss per call 1.29 · 41 calls in 8 rows
step  6 · loss per call 1.30 · 46 calls in 8 rows
step  7 · loss per call 1.26 · 40 calls in 8 rows
step  8 · loss per call 2.54 · 36 calls in 8 rows
step  9 · loss per call 4.89 · 50 calls in 8 rows
step 10 · loss per call 5.36 · 38 calls in 8 rows
step 11 · loss per call 2.65 · 42 calls in 8 rows
step 12 · loss per call 2.95 · 45 calls in 8 rows
step 13 · loss per call 3.53 · 38 calls in 8 rows
step 14 · loss per call 0.63 · 45 calls in 8 rows
step 15 · loss per call 1.75 · 41 calls in 8 rows

held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    100%
medium    0 of 7    1%        7 of 7    100%
hard      0 of 4    0%        0 of 4    0%
```

```text
habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    0%       0%       0%
said FINISH with no answer.txt  0%       0%       1%
  of them on call 4             0%       0%       0%
calls per rollout               4.0      7.0      8.0
```

The plan expected the loss to fall near zero within a few steps while the
held-out success dropped: the student memorising its examples instead of
learning the method, called **overfitting**. That is not what happened. The
loss jumps around, as high as 5.36 at step 10, because each step overshoots:
it moves the weights so far toward the batch in front of it that the next
batch looks wrong. And the held-out success went up: 100% of easy and medium
rollouts, sampled, because the student is now almost certain of every move.
Hard questions went from 21% to 0%: it no longer looks four times by chance.

This student cannot memorise. It sees five facts about the rollout, and none
of them says which question it is, so whatever it learns from one
demonstration it applies to every question in the same situation; there is
nothing question by question to store. A real model reads the
whole question and can memorise each demonstration. The one thing this
student can memorise is a shape, and only one teacher has a single shape:

```bash
uv run sft_toy.py --fixed-length --lr-high
```

```text
step  1 · loss per call 1.95 · 32 calls in 8 rows
step  2 · loss per call 0.00 · 32 calls in 8 rows
step  3 · loss per call 0.00 · 32 calls in 8 rows
step  4 · loss per call 0.00 · 32 calls in 8 rows
step  5 · loss per call 0.00 · 32 calls in 8 rows
```

and so on: 0.00 at every step after the first.

```text
held-out test: greedy (one rollout per question) and sampled (20 per question, temperature 1)
          untrained           after SFT
          greedy    sampled   greedy    sampled
easy      0 of 8    4%        8 of 8    100%
medium    0 of 7    1%        0 of 7    0%
hard      0 of 4    0%        0 of 4    0%

the student, greedy, on a medium question, which needs one LOOK more than any easy question it was shown
How many orders did customers from Lisbon place?
  call 1  LOOK     p 1.00  customers: id, name, city, signup_date
  call 2  QUERY_A  p 1.00  no such table: orders
  call 3  WRITE    p 1.00  nothing to write yet
  call 4  FINISH   p 1.00  no answer.txt · reward 0

habits after SFT, sampled       easy     medium   hard
read answer.txt back (CHECK)    0%       0%       0%
said FINISH with no answer.txt  0%       100%     100%
  of them on call 4             0%       100%     98%
calls per rollout               4.0      4.0      4.0
```

Loss 0.00 after one step: a loss that drops this fast is the warning sign.
The student copies the four moves exactly: easy questions 100%, and every
medium and hard rollout, sampled, now ends with `FINISH` on call 4 and nothing
saved. The medium questions it got by chance at the default learning rate
(14%) are gone (0%). Better on what it was shown, worse on anything
different.

> **In the project's code** (`mhrl/cli.py`): `SFT_LR = 5e-5  # 1e-4 over-fit
> the dry run (loss 0.10 by step 4 of 6)`. The project halved its learning
> rate after that.

## Not one lucky seed

```bash
uv run seeds.py
```

```text
seeds 0-9 · 120 demonstrations, batch 8, one epoch · lr 3 (high: 30) · RL 100 steps
held-out test after SFT: easy, medium, hard = sampled rollouts right (20 per question); medium greedy = greedy rollouts right
call 4 unsaved = sampled medium rollouts that said FINISH on call 4 with no answer.txt; checked = sampled rollouts that read answer.txt back

varied, then RL
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 0.44  74%    54%     21%    7 of 7         5%              16%
 1    1.95 -> 0.54  82%    56%     9%     7 of 7         5%              23%
 2    1.95 -> 0.59  79%    65%     21%    7 of 7         2%              21%
 3    1.95 -> 0.43  80%    49%     12%    7 of 7         2%              30%
 4    1.95 -> 0.43  71%    57%     20%    7 of 7         3%              14%
 5    1.95 -> 0.52  84%    66%     16%    7 of 7         0%              23%
 6    1.95 -> 0.49  72%    54%     18%    7 of 7         8%              13%
 7    1.95 -> 0.42  80%    59%     5%     7 of 7         4%              26%
 8    1.95 -> 0.42  75%    58%     25%    7 of 7         4%              24%
 9    1.95 -> 0.45  79%    49%     8%     7 of 7         4%              16%
all   1.95 -> 0.47  78%    57%     16%    70 of 70       4%              20%
claim: loss per step falls
  test: the loss at the last step below the loss at step 1
  holds in 10 of 10 seeds
claim: --varied learns 'finish after saving' and survives a question that needs one extra look
  test: greedy: every held-out medium question right
  holds in 10 of 10 seeds

fixed-length
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 0.06  84%    14%     0%     0 of 7         46%             4%
 1    1.95 -> 0.06  88%    14%     0%     0 of 7         49%             6%
 2    1.95 -> 0.06  85%    16%     1%     0 of 7         45%             4%
 3    1.95 -> 0.06  79%    12%     1%     0 of 7         50%             5%
 4    1.95 -> 0.06  82%    11%     0%     0 of 7         48%             5%
 5    1.95 -> 0.06  87%    13%     0%     0 of 7         49%             6%
 6    1.95 -> 0.06  84%    10%     1%     0 of 7         54%             8%
 7    1.95 -> 0.06  85%    19%     0%     0 of 7         42%             5%
 8    1.95 -> 0.06  88%    14%     1%     0 of 7         44%             6%
 9    1.95 -> 0.06  85%    10%     1%     0 of 7         49%             7%
all   1.95 -> 0.06  85%    13%     1%     0 of 70        48%             6%
claim: on a question that needs one extra look, the student finishes on call 4 without saving
  test: greedy: every held-out medium question ends with FINISH on call 4 and no answer.txt
  holds in 10 of 10 seeds

never-checks
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 0.37  75%    66%     28%    7 of 7         7%              4%
 1    1.95 -> 0.48  82%    60%     12%    7 of 7         11%             4%
 2    1.95 -> 0.32  79%    61%     22%    7 of 7         8%              4%
 3    1.95 -> 0.28  82%    57%     12%    7 of 7         11%             6%
 4    1.95 -> 0.33  71%    62%     22%    7 of 7         9%              4%
 5    1.95 -> 0.34  81%    69%     14%    7 of 7         9%              3%
 6    1.95 -> 0.42  79%    59%     29%    7 of 7         10%             3%
 7    1.95 -> 0.26  81%    61%     5%     7 of 7         11%             6%
 8    1.95 -> 0.33  74%    63%     21%    7 of 7         11%             3%
 9    1.95 -> 0.30  84%    61%     5%     7 of 7         7%              4%
all   1.95 -> 0.34  79%    62%     17%    70 of 70       9%              4%
claim: the student never checks either
  test: greedy: no held-out rollout reads answer.txt back
  holds in 10 of 10 seeds
claim: the student never checks either
  test: sampled: no held-out rollout reads answer.txt back
  holds in 0 of 10 seeds

varied, lr 30
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 1.75  100%   100%    0%     7 of 7         0%              0%
 1    1.95 -> 0.89  100%   99%     1%     7 of 7         0%              33%
 2    1.95 -> 1.79  100%   100%    99%    7 of 7         0%              17%
 3    1.95 -> 2.65  100%   82%     0%     7 of 7         0%              24%
 4    1.95 -> 1.83  99%    97%     100%   7 of 7         0%              0%
 5    1.95 -> 1.62  100%   5%      0%     0 of 7         0%              99%
 6    1.95 -> 2.84  60%    56%     51%    7 of 7         0%              0%
 7    1.95 -> 1.99  99%    96%     0%     7 of 7         0%              0%
 8    1.95 -> 1.17  91%    87%     89%    7 of 7         0%              0%
 9    1.95 -> 0.29  100%   99%     0%     7 of 7         0%              13%
all   1.95 -> 1.68  95%    82%     34%    63 of 70       0%              19%
claim: training loss near zero within a few steps
  test: loss 0.1 or less at one of steps 1-5
  holds in 0 of 10 seeds
claim: while held-out success drops
  test: sampled: fewer held-out rollouts right than the lr 3 run of the same seed and teacher
  holds in 1 of 10 seeds
claim: while held-out success drops
  test: the same, on the medium questions only
  holds in 1 of 10 seeds

fixed-length, lr 30
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 1    1.95 -> 0.00  99%    0%      0%     0 of 7         99%             0%
 2    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 3    1.95 -> 0.00  99%    0%      0%     0 of 7         100%            0%
 4    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 5    1.95 -> 0.00  100%   0%      0%     0 of 7         99%             0%
 6    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 7    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 8    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
 9    1.95 -> 0.00  100%   0%      0%     0 of 7         100%            0%
all   1.95 -> 0.00  100%   0%      0%     0 of 70        100%            0%
claim: training loss near zero within a few steps
  test: loss 0.1 or less at one of steps 1-5
  holds in 10 of 10 seeds
claim: while held-out success drops
  test: sampled: fewer held-out rollouts right than the lr 3 run of the same seed and teacher
  holds in 2 of 10 seeds
claim: while held-out success drops
  test: the same, on the medium questions only
  holds in 10 of 10 seeds

fixed-length, calls as a share
seed  loss          easy   medium  hard   medium greedy  call 4 unsaved  checked
 0    1.95 -> 0.43  76%    51%     2%     0 of 7         4%              5%
 1    1.95 -> 0.43  81%    48%     1%     0 of 7         6%              5%
 2    1.95 -> 0.43  72%    52%     1%     0 of 7         4%              4%
 3    1.95 -> 0.43  69%    50%     2%     0 of 7         4%              6%
 4    1.95 -> 0.43  71%    52%     0%     0 of 7         3%              3%
 5    1.95 -> 0.43  73%    51%     1%     0 of 7         4%              4%
 6    1.95 -> 0.43  71%    45%     0%     0 of 7         6%              7%
 7    1.95 -> 0.43  76%    51%     0%     0 of 7         4%              6%
 8    1.95 -> 0.43  79%    41%     4%     0 of 7         5%              6%
 9    1.95 -> 0.43  75%    52%     2%     0 of 7         3%              5%
all   1.95 -> 0.43  74%    49%     2%     0 of 70        4%              5%
claim: on a question that needs one extra look, the student finishes on call 4 without saving
  test: greedy: every held-out medium question ends with FINISH on call 4 and no answer.txt
  holds in 0 of 10 seeds

varied, then RL: RL on the hard questions, the first quarter -> the last
seed  scratch right    scratch signal   imitator right   imitator signal  imitator step 1
 0    0% -> 0%         0/50 -> 0/50     54% -> 96%       42/50 -> 8/50    2/2
 1    0% -> 0%         0/50 -> 0/50     62% -> 98%       34/50 -> 3/50    2/2
 2    0% -> 0%         0/50 -> 0/50     60% -> 97%       37/50 -> 5/50    2/2
 3    0% -> 0%         0/50 -> 0/50     54% -> 94%       36/50 -> 11/50   1/2
 4    0% -> 96%        1/50 -> 9/50     60% -> 98%       34/50 -> 4/50    1/2
 5    0% -> 0%         0/50 -> 0/50     46% -> 96%       35/50 -> 7/50    0/2
 6    0% -> 0%         0/50 -> 0/50     56% -> 98%       31/50 -> 3/50    0/2
 7    0% -> 0%         0/50 -> 0/50     49% -> 97%       35/50 -> 5/50    1/2
 8    0% -> 0%         0/50 -> 0/50     60% -> 98%       39/50 -> 3/50    1/2
 9    0% -> 0%         0/50 -> 0/50     56% -> 98%       38/50 -> 3/50    1/2
claim: RL from scratch rarely finishes correctly, so most of its groups have no contrast
  test: from scratch, under 5% of rollouts right and over 90% of groups without contrast
  holds in 9 of 10 seeds
claim: RL from the imitator has signal from step 1
  test: from the imitator, a group with contrast at step 1
  holds in 8 of 10 seeds
claim: RL started from the imitator learns faster on the hard questions
  test: from the imitator, more rollouts right than from scratch in every quarter
  holds in 10 of 10 seeds
```

Each table has one row per seed and an `all` row that adds the ten seeds
together. Under each table, every claim the course plan makes about that run,
the test it was checked with, and in how many seeds it held:

1. **The loss falls**, in every run: 1.95 to 0.42-0.59 with the project's
   teacher, 0.06 with the fixed-length one.
2. **The project's teacher survives one extra look**: every medium question
   right, greedily, in all 10 seeds.
3. **The fixed-length student finishes on call 4 without saving**: every medium
   question, greedily, in all 10 seeds; sampled, 42% to 54% of medium rollouts.
   With "calls used" as one number it holds in no seed (3% to 6% sampled).
4. **A teacher that never checks: the student never checks either.** Greedily,
   in all 10 seeds. Sampled, in no seed: 3% to 6% of rollouts still check,
   against 13% to 30% for the student of the teacher that checked.
5. **A high learning rate: loss near zero, held-out success drops.** With the
   project's teacher the loss never got near zero (0 of 10), and held-out
   success dropped in 1 seed of 10. That seed, 5, locked into checking: 99% of
   its rollouts read answer.txt back. The high rate makes each
   seed commit to whatever its last big steps favoured, so its hard questions
   came out 0% in five seeds and 89% to 100% in three. With the fixed-length
   teacher the loss was 0.00 by step 2 in every seed, and the medium questions
   dropped to 0% in every seed. Counting every level, success dropped in only 2
   of 10, because easy questions rose to 99-100%.
6. **RL**: in the next section.

## SFT, then RL

The project trains in stages: first the untouched model is measured, then
SFT, then RL started from the SFT weights, measured after each. `--then-rl`
does that on the toy, with RL on the hard questions. Chapter 5 showed that
hard questions practised on their own almost never come out right from
scratch, so they give GRPO no contrast to learn from.

```bash
uv run sft_toy.py --then-rl
```

After the same SFT as above:

```text
then RL on the 14 hard training questions: chapter 6's GRPO, 100 steps of 2 questions x 4 rollouts, lr 3
from scratch (every weight 0) and from the imitator (the weights SFT just trained): the same questions, in the same order

step  1 · from scratch right 0/8, signal 0/2 · from the imitator right 2/8, signal 2/2
step  2 · from scratch right 0/8, signal 0/2 · from the imitator right 3/8, signal 2/2
step  3 · from scratch right 0/8, signal 0/2 · from the imitator right 0/8, signal 0/2
step  4 · from scratch right 0/8, signal 0/2 · from the imitator right 1/8, signal 1/2
step  5 · from scratch right 0/8, signal 0/2 · from the imitator right 5/8, signal 2/2
step  6 · from scratch right 0/8, signal 0/2 · from the imitator right 4/8, signal 2/2
step  7 · from scratch right 0/8, signal 0/2 · from the imitator right 4/8, signal 2/2
step  8 · from scratch right 0/8, signal 0/2 · from the imitator right 3/8, signal 2/2
step  9 · from scratch right 0/8, signal 0/2 · from the imitator right 2/8, signal 2/2
step 10 · from scratch right 0/8, signal 0/2 · from the imitator right 4/8, signal 2/2

in each quarter: steps  1-25      26-50     51-75     76-100
rollouts right
  from scratch          0%        0%        0%        0%
  from the imitator     54%       92%       98%       96%
groups with signal
  from scratch          0/50      0/50      0/50      0/50
  from the imitator     42/50     14/50     5/50      8/50

held-out test after each stage: greedy right, then sampled right (20 per question)
          untrained       SFT             SFT, then RL    RL from scratch
easy      0 of 8, 4%      8 of 8, 74%     8 of 8, 100%    0 of 8, 4%
medium    0 of 7, 1%      7 of 7, 54%     7 of 7, 98%     0 of 7, 1%
hard      0 of 4, 0%      0 of 4, 21%     4 of 4, 100%    0 of 4, 0%

habits after SFT, then RL       easy     medium   hard
read answer.txt back (CHECK)    7%       4%       8%
said FINISH with no answer.txt  0%       1%       0%
  of them on call 4             0%       1%       0%
calls per rollout               7.9      7.9      8.0
```

From scratch, no hard rollout came out right in 100 steps, so not one of its
200 groups had contrast, and it learned nothing: GRPO skipped every group.
From the imitator, 2 of the first 8 rollouts were right, both groups of step
1 had contrast, and by the second quarter it was right 92% of the time. The
imitator never saw a hard question, but a quarter of its demonstrations (the
tables path) looked at all four tables first, which is what a hard question
needs. SFT made a lucky success likely enough for RL to build on.

After RL on hard questions alone it answers every held-out question
greedily. And it now uses nearly the whole budget, 7.9 calls per rollout:
paid for right answers only, RL has no reason to stop early (chapter 7).

The last table of [the seed run](#not-one-lucky-seed) checks the three claims:

1. **RL from scratch rarely finishes correctly, so most of its groups have no
   contrast**: 9 seeds of 10 had no right rollout in 100 steps. Seed 4 is the
   exception: one rollout came out right at step 14, a few more followed, and
   from step 46 on it learned, reaching 96% in its last quarter. A lucky
   success can happen from scratch; here it happened once in ten runs.
2. **RL from the imitator has signal from step 1**: in 8 seeds of 10. In seeds
   5 and 6 step 1's groups tied, and their first quarters had contrast in 35
   and 31 of 50 groups.
3. **It learns faster on the hard questions**: more rollouts right than from
   scratch in every quarter, in all 10 seeds (in seed 4's last quarter, 98%
   against 96%).

## The run's log, and its curves

Every run writes three files and prints their md5s. `demos.jsonl` holds every
demonstration as the trainer saw it, each piece with its loss mask.
`metrics.jsonl` has one line per SFT step (`loss_per_call` and `calls`: the
toy's version of the project's `loss_per_token` and `tokens`) and, with
`--then-rl`, one line per GRPO step of each start, with the project's field
names (`accuracy`, `mean_reward`, `mean_calls`, `groups_without_contrast`).
`rollouts.jsonl` holds every test and RL rollout, call by call. Same seed,
same files.

```text
wrote ~/.cache/mhrl-course/08-sft-toy/runs/varied-seed0-lr3/demos.jsonl
md5   a7bfe94a508fb63b206efdca9e978903
wrote ~/.cache/mhrl-course/08-sft-toy/runs/varied-seed0-lr3/metrics.jsonl
md5   262127d044a438e8dc3cc748b1dada56
wrote ~/.cache/mhrl-course/08-sft-toy/runs/varied-seed0-lr3/rollouts.jsonl
md5   91180b5bbb6f5223469bab89b3a1a0c5
redraw it: uv run plot.py ~/.cache/mhrl-course/08-sft-toy/runs/varied-seed0-lr3/metrics.jsonl
```

```bash
uv run plot.py ~/.cache/mhrl-course/08-sft-toy/runs/varied-seed0-lr3-then-rl100/metrics.jsonl
```

```text
15 SFT steps; loss per call 1.95 -> 0.44

SFT loss per call (l), one column per step
  2.0|l
     |
     |
     | l
     |
  1.0|  l
     |   lll
     |      lllll
     |           llll
     |
  0.0|
     +---------------
      1            15  step

100 RL steps from each start; each point is the average of the 10 steps up to it

rollouts right (s from scratch, i from the imitator, * where they meet)
 100%|                        i   iiiiiiiiiiiiii     iii
     |                iiiiiiii iii              iiiii
     |              ii
     |          iiii
     |       iii
  50%|      i
     |    ii
     |i ii
     | i
     |
   0%|ssssssssssssssssssssssssssssssssssssssssssssssssss
     +--------------------------------------------------
      1                                              100  step

groups with signal (s from scratch, i from the imitator)
 100%|i     i
     |            i
     |  iiii iiiii
     |             i
     | i            ii
  50%|
     |                i  i                       ii
     |                  i  ii                      i
     |                 i  i  iiiiiiiii         ii   ii
     |                                ii      i
   0%|ssssssssssssssssssssssssssssssssss******ssssssss**
     +--------------------------------------------------
      1                                              100  step
```

plot.py redraws a run from `metrics.jsonl` alone, as text, with Python's
standard library only. The loss falls fast and then slowly. In RL the
imitator (`i`) climbs from about half right to nearly all, while from scratch
(`s`) stays on the floor; and the imitator's groups with signal fall as it
gets the questions right: all four right is no contrast too (chapter 5).

## The project's real SFT run

```bash
uv run real_sft.py
```

```text
the project's SFT run: LFM2.5-2.6B fine-tuned on the answer-key teacher's
demonstrations, captured through mini-swe-agent and Claude Code

120 demonstrations (60 mini-swe-agent, 60 claude-code), 120 right and kept
tokens in them: 209,711
  the teacher's replies (loss mask 1): 20,371, trained
  the rest (189,340: the harness's instructions and the tool outputs): read, never scored
batch 8, 1 epoch: 15 steps

step  loss per token  tokens
   1  2.55            1,474
   2  2.28            1,343
   3  1.77            1,393
   4  1.15            1,243
   5  0.66            1,358
   6  0.45            1,282
   7  0.37            1,351
   8  0.30            1,257
   9  0.31            1,259
  10  0.27            1,526
  11  0.21            1,370
  12  0.17            1,433
  13  0.13            1,336
  14  0.10            1,457
  15  0.11            1,289
tokens trained, all steps: 20,371 (the same as the teacher's above)

held-out tasks, base -> SFT, task by task (24 tasks per harness, greedy)
harness         base    SFT     gained  lost    p       calls per solve
mini-swe-agent  24      18      0       6       0.03    7.8 -> 4.7 (18 tasks)
claude-code     15      17      7       5       0.77    7.7 -> 5.1 (10 tasks)
pi              4       9       7       2       0.18    6.0 -> 5.0 (2 tasks)
calls per solve: on the tasks solved both before and after; p: exact McNemar test

from: the project's runs/main/sft/metrics.jsonl, sft/demos.jsonl and
runs/main/eval_*.jsonl, copied (cut down) into data/real_sft/
```

The project's SFT is the same shape on the real model: 120 demonstrations,
batch 8, one epoch, 15 steps. Of the 209,711 tokens in the demonstrations,
20,371 were the teacher's replies, and those are exactly the tokens trained:
the steps' token counts add up to the same number. The loss per token fell
from 2.55 to 0.11.

Read the evaluation task by task, never from a headline average. In
mini-swe-agent SFT lost 6 tasks and gained none, and an exact test says a
split that uneven would happen by chance about 3 times in 100 (p 0.03). In
Claude Code it gained 7 and lost 5 (p 0.77), and in Pi, a harness it never
trained in, gained 7 and lost 2 (p 0.18): both point up, and neither is more
than chance can explain with 24 tasks. What is clear is that every solve got
cheaper: in mini-swe-agent, 7.8 calls per solve became 4.7 on the 18 tasks
solved both times, nearer the teacher's 3 or 4 calls per demonstration
there. The project's README says why mini-swe-agent lost: "the
demonstrations always ran one query and never double-checked."

> **From the article** (Hugging Face and Liquid AI, "The ultimate guide to
> multi-harness RL", section "SFT vs RL"): "We ran Qwen3.8-27B as a teacher on
> SmolDataEnvs training tasks in all four harnesses, through the same OpenEnv
> and Harbor setup, with sandboxes on Daytona. Each task got up to three
> attempts per harness, and we kept the first one the verifier graded correct.
> That gave 3,189 successful rollouts from 888 tasks, 377 medium and 511 hard."
> "Every assistant turn in a rollout becomes one training example. The
> conversation up to that turn, tool outputs included, is the context, and the
> loss is on the next assistant response only." "Both are full fine-tunes with
> TRL’s SFT trainer for two epochs, at a learning rate of 3e-6 and an effective
> batch size of 8."
>
> "RL ends ahead of both SFT runs. Multi-harness RL reaches 54.6%. The better
> SFT model, OpenCode SFT, reaches 47.5%, and multi-harness SFT 43.1%, 11.5
> points below the RL model trained in the same four harnesses." Multi-harness
> SFT "improves under Claude Code (33.2 → 42.4%), Codex (40.0 → 46.8%) and
> OpenCode (33.6 → 38.0%). Under Mini-SWE-Agent it falls from 62.1% to
> 45.2% ... We have not yet worked out why SFT hurt Mini-SWE-Agent." "SFT cuts
> tool calls without being rewarded for it." And: "The runs we want next are
> SFT followed by RL against RL from the base model."

The article's teacher was a model about 10 times larger than its student; the
project's and this chapter's are scripts that know the answers. Both kinds
pass on their habits. The article's last sentence names the comparison
`--then-rl` makes on the toy.

## SFT and RL, side by side

| | SFT | RL |
| --- | --- | --- |
| learns from | a teacher's demonstrations | its own rollouts |
| signal | the teacher's move, at every call | one reward per rollout, compared within its group |
| can push a move down | only by raising the teacher's | yes: a rollout that did worse than its group |
| values efficiency | no: it copies the teacher's calls (5.3 per demonstration; 5.2 per easy rollout here) | only what the reward pays for: paid for right answers only, 7.9 calls here |
| needs | a teacher who can do the job | a grader, and rollouts that sometimes succeed |
| the hard questions here | after SFT on easy questions only: 0 of 4 greedy, 21% sampled | from scratch: 0 of 4, 0%; started from the imitator: 4 of 4, 100% |

SFT copies a teacher's actions, the good, the useless and the missing, and
cannot tell them apart: it has no idea which of them earned the reward. RL
learns from its own attempts and from whatever the reward pays for, and
nothing else; it needs some attempts to succeed before it can learn anything.
The pipeline uses each for what it is good at: SFT gives RL a start where
success is possible, and RL then improves on the teacher by its own results.

## What the toy showed, and where it differed from the plan

- **Loss per step falling, then success on held-out questions:** as planned.
- **`--fixed-length`:** as planned, in every seed: on a question that needs
  one extra look, `FINISH` on call 4, answer.txt missing, reward 0. It needed
  the student to see the call number as switches; with chapters 4 to 7's single
  number it never formed (`--calls-as-share`, 0 of 10 seeds).
- **`--varied` learns "finish after saving" and survives:** greedily, in every
  seed. Sampled, a quarter of its rollouts still stop early, but rarely on
  call 4: one epoch leaves a lot of chance in the probabilities.
- **`--teacher-never-checks`:** the student never checks greedily, in every
  seed; sampled, 3% to 6% of its rollouts still do, the leftover of its
  untrained chance. In the toy a check changes nothing, so the habit shows but
  costs nothing; the cost is the project's (mini-swe-agent, 0 gained, 6 lost).
- **`--lr-high`:** did not hold with the project's teacher. The loss never got
  near zero (0 of 10 seeds) and held-out success rose (dropped in 1 of 10),
  because this student cannot tell questions apart and so has nothing to
  memorise. It held with the fixed-length teacher: loss 0.00 by step 2 and the
  medium questions at 0%, in every seed (counting easy questions too, which
  rose, the drop holds in 2 of 10).
- **`--then-rl`:** RL from scratch on the hard questions never had a right
  rollout in 9 seeds of 10; one seed got lucky and learned late. RL from the
  imitator had a group with contrast at step 1 in 8 of 10, signal in most
  groups of its first quarter in all 10, and was ahead in every quarter of
  every seed.
- **Chosen while building, and measured:** demonstrations on the easy
  questions only, so that every teacher shows the same questions; chapter 6's
  `WRITE` and `FINISH` instead of chapter 7's `SUBMIT`; chapter 7's single set
  of weights; and a `CHECK` action, the toy's `cat answer.txt`.

## What is in this folder

- `sft_toy.py`: the chapter: collect demonstrations, SFT, the held-out test,
  RL after it, the logs, and every flag.
- `teacher.py`: the answer-key teacher, its four paths, and its fixed-length
  and never-checks versions; the path is picked with the project's line.
- `sft.py`: SFT itself: demonstrations as sequences with a loss mask, rows,
  the project's `batches` line for line, its `loss_fn`, the gradient worked out
  by hand, and the training loop.
- `rl.py`: chapter 6's GRPO, started from any weights.
- `rollouts.py`: one rollout of the student.
- `evaluate.py`: the held-out test, greedy and sampled, and the habits.
- `policy.py`: chapter 4's softmax policy with the call number as switches
  (`--calls-as-share` gives the old single number); its math is chapter 3's,
  line for line.
- `environment.py`: chapter 6's environment, with `CHECK`.
- `knobs.py`: the chapter's fixed numbers (120 demonstrations, batch 8, one
  epoch, the learning rates).
- `report.py`: how runs and the seed tables are printed.
- `seeds.py`: every claim over several seeds, side by side on the Mac's cores.
- `plot.py`: the curves, drawn as text from `metrics.jsonl` alone (standard
  library only).
- `real_sft.py`, `data/real_sft/`: the project's real SFT run, from its own
  records.
- `questions.py`, `grader.py`, `grpo.py`, `data/shop.db`: chapter 6's, copied
  unchanged.
- `tasks.py`: chapter 6's `tasks.py`, writing under `08-sft-toy`.

## Words from this chapter

- **Supervised fine-tuning (SFT)**: training a model on examples of the right
  output, here a teacher's moves, to make it more likely to produce them:
  learning by imitation.
- **Teacher demonstrations**: a stronger model's or a script's successful runs,
  recorded and used as training examples. Here, and in the project, a script
  that knows the answer key.
- **Loss mask**: a 1 or a 0 per piece of a demonstration; only the 1s, the
  teacher's moves, are trained. What came back from the environment is read,
  never scored.
- **Cross-entropy (next-token) loss**: minus the log of the probability the
  student gave the teacher's move. Sure and right costs almost nothing; unsure
  costs more.
- **Batch, step, epoch**: the examples behind one update; one update; one pass
  over every example. Here and in the project, 120 demonstrations, batch 8,
  one epoch: 15 steps.
- **Demonstration diversity, shortcut learning**: when every example shares
  something that does not matter (four calls), the student can learn that
  instead of the rule (finish after saving). Varying what does not matter
  teaches what does.
- **Imitation narrows behaviour**: a student copies the teacher's habits, and
  drops the ones the teacher never shows, good ones included.
- **Overfitting**: fitting the training examples so closely that the student
  does worse on anything different; a loss that drops too fast is the warning
  sign.
- **SFT, then RL**: measure the untouched model, then SFT, then RL started from
  the SFT weights, measuring after each stage.
- **SFT vs RL**: SFT copies a teacher's moves and cannot value what they cost;
  RL learns from its own attempts and from whatever the reward pays for.

Next: [chapter 9](../09-tokens-and-templates/), From four choices to 128,000:
tokens and chat templates.
