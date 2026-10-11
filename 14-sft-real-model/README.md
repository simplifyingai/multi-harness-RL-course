# Chapter 14. SFT on the real model

Chapter 8 taught a toy agent by imitation: show it what a teacher did, and
make the teacher's choices more likely. Chapter 13 built the add-on that
training can change on LFM2.5-2.6B while its 2.7 billion numbers stay
frozen. This chapter puts the two together, as the project does in its first
training stage (`mhrl/training/sft.py`): the model reads a recorded
conversation and learns only from the reply, because a mask hides everything
it only read. We fine-tune it on the correct SQL for 180 questions, measure
it on 50 questions about two databases it has never seen, and look at what
over-fitting and forgetting look like.

The puzzle to keep in mind: every training reply is correct SQL, written by
an answer key that cannot make mistakes, and in two minutes the loss (the
number training pushes down) falls from 2.46 to 0.32. Does the model now
write better SQL for databases it has not seen? By the end of the chapter
you will have measured it, question by question, and you will know why.

## Words first

- **SFT** (supervised fine-tuning) is chapter 8's **imitation** with
  tokens: show the model the reply a teacher would write, and change the
  model so that this reply becomes more likely.
- The **chat template** (chapter 9) turns a conversation into the text the
  model reads: each message between `<|im_start|>role` and `<|im_end|>`. The
  **prompt** is everything the model reads before it answers; it ends with
  the **generation prompt**, `<|im_start|>assistant` and `<think>`, which
  opens the model's reply with its reasoning. The **reply** is what the
  model writes: its reasoning, `</think>`, then its answer, then the
  end-of-turn token `<|im_end|>`.
- A **training row** is one prompt followed by the reply to learn, as one
  list of token ids.
- The **adapter** (chapter 13) is the small set of trainable numbers, LoRA
  pairs on the last 12 layers, added beside the frozen 4-bit model. It is
  the only thing training changes, and it is saved as one file, such as
  `sft.safetensors`.
- The **loss mask** is a 0 or a 1 for every position of a row: 1 where the
  token is part of the reply (learn this), 0 where the model only reads it
  (the rule, the tables, the question, the template's markers).
- A token's **log-probability** (chapter 3) is the logarithm of the chance
  the model gives it; the log-probability of a whole reply is the sum of
  its tokens' (chapter 12).
- **Scoring in one pass** (also called **teacher forcing**): to train, the
  model does not write the reply; it reads the whole row at once and, at
  every position, gives its scores for the next token. The recorded tokens
  are fed in, never the model's own guesses.
- The **next-token shift**: the scores at position p are the model's guess
  for the token at position p + 1. So a reply token's log-probability is
  read at the position just before it, and the first token of a row can
  never be a target: nothing comes before it.
- The **loss** of a row is minus the sum of its reply tokens'
  log-probabilities: the less likely the model finds the reply, the larger
  the loss. **Loss per token** divides by the number of reply tokens.
- A **step** (the course's word since chapter 3) is one optimizer update;
  a **batch** is the rows one step reads (8 here); an **epoch** is one pass
  over all the rows (180 rows in batches of 8 make 23 steps).
- **Gradient accumulation**: the rows of a step are read one at a time, to
  keep only one row's numbers in memory; each row's gradient (chapter 3:
  which way each trainable number should move) is divided by the step's
  total reply-token count and added up, then the optimizer takes one step.
  The division is **token-level normalisation**: every reply token counts
  the same, whichever row it is in.
- **Held-out questions** are questions no training row comes from. Here
  they are about two other databases (clinic and airline), so the model has
  never seen their tables either.
- **pass@1** is the share of questions answered right on the one try the
  model gets (greedy: always the likeliest next token, chapter 12).
- The **error bar** is how far a score would typically move by luck alone
  if another set of questions like these were asked: plus or minus one
  standard error, sqrt(p x (1 - p) / n), where p is the score and n the
  number of questions.
- Two runs on the same questions are compared **question by question**:
  **gained** counts questions wrong before and right after, **lost** the
  reverse. If the runs were equally good, each changed question would be a
  coin flip; the **exact p-value** (McNemar's exact test, chapters 8 and 11)
  is the chance of a split at least as uneven as the one measured.
- **Over-fitting** is learning the training rows themselves instead of the
  skill behind them: the training loss keeps falling while the held-out
  score does not follow.
- **Forgetting** (catastrophic forgetting) is fine-tuning on a narrow job
  wearing away something the model could do before.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/14-sft-real-model
uv run make_rows.py --show 1
uv run eval_sql.py
uv run train_sft.py
uv run eval_sql.py --adapter sft.safetensors
```

All three need an Apple-silicon Mac (they use MLX). `make_rows.py` loads
only the tokenizer; the other two load the 4-bit model: chapter 11's own
build in
`COURSE_HOME/models/lfm2.5-2.6b-4bit-ours` (default `~/.cache/mhrl-course`),
or, if you have not built it, Liquid's export that chapter 10's
`download.py --fast` fetched (the same bytes, chapter 11 checked). Nothing
is downloaded. Rows, adapters, evaluations and each question's `answer.txt`
go under `COURSE_HOME/14-sft-real-model`. The 230 questions and their five
SQLite databases are bundled in `data/` (chapter 1's files, byte for byte,
which are the project's).

| command | what it shows | time in the run log |
| --- | --- | ---: |
| `make_rows.py --show 1` | the questions, the chat and the reply, the rows, row 1 token by token | 9 s |
| `eval_sql.py` | the base model's 50 verdicts, pass@1 with its error bar, a few replies | 6.3 min |
| `train_sft.py` | the no-op check, 23 training steps, the project's recorded runs | 2.1 min |
| `eval_sql.py --adapter sft.safetensors` | pass@1 after training, question by question against the base model, five unrelated prompts | 9.8 min |
| `train_sft.py --reason fixed` | the same training, every reason one fixed sentence | 2.5 min |
| `eval_sql.py --adapter sft_reasonfixed.safetensors` | its pass@1 and comparison | 5.3 min |
| `train_sft.py --lr 1e-4 --epochs 3` | over-fitting: 69 steps | 6.3 min |
| `eval_sql.py --adapter sft_lr0.0001_epochs3.safetensors` | its pass@1 and comparison | 6.5 min |
| `make_rows.py --fast`, `train_sft.py --fast`, `eval_sql.py --fast --adapter sft_limit40.safetensors` | fast mode | 12 s, 41 s, 1.7 min |

Every number below comes from recorded runs of these commands on an M3 Pro
with 18 GB, with video renders running on the same machine (each command's
commit, date, exit code and time are in the run log). Only the lines marked
"varies from run to run" are times or memory. Everything else is
determined: fixed data, a fixed seed for the adapter's start, a fixed
shuffle per epoch and greedy replies. The run log ran `train_sft.py` twice
and the two adapter files are byte for byte the same; the base model's 50
verdicts and reply lengths were the same in a trial run earlier the same
day (kept beside the run log).

## From a question to a training row

```bash
uv run make_rows.py --show 1
```

`make_rows.py` loads only the tokenizer and turns each of the 230 bundled
questions into a row. First it counts them:

```text
1. the questions: 230, bundled in data/ with 5 databases
  split  databases                easy  medium  hard  total
  train  library, school, shop      54      72    54    180
  test   airline, clinic            16      20    14     50
  the held-out databases (clinic, airline) appear in no training row: their tables and
    questions are new to the model
```

The 180 training questions are the project's own (`data/tasks/train.jsonl`),
over three databases; the 50 held-out ones are over two others, so a
held-out question asks about tables no training row ever showed.

Each question becomes a chat and a reply:

```text
2. each question becomes a chat and a reply
  system  You answer questions about a SQLite database with one SQL query. Its result
    must be the answer and nothing else: one column, one row per item if the answer is a
    list. Reply with the query alone, after your reasoning.
  user    'The database has these tables:', its CREATE TABLE lines, then the question
  reply   a one-line reason, </think>, the gold SQL, <|im_end|>
  the reason: written from the gold SQL: its tables, then what the query does
  for shop-medium-001: "Use the customers table: keep the matching rows, group by city,
    then sort by the count, highest first, and keep the top row.</think>SELECT city FROM
    customers WHERE strftime('%Y', signup_date) = '2024' GROUP BY city ORDER BY COUNT(*)
    DESC LIMIT 1"
```

- The **chat** has a system message (the rule) and a user message (the
  database's tables, as the `CREATE TABLE` statements that made them, then
  the question). The rule asks for the answer alone, in one column, as
  chapter 11's prompt learned to: the answer is read from the result's
  first column.
- The **reply** is what the model should write after the generation
  prompt's `<think>`: a one-line reason, `</think>`, the gold SQL (the
  answer key's query, which the model never sees otherwise), and the
  end-of-turn token. The project's teacher writes its demonstrations the
  same way: a short line of reasoning, `</think>`, then the action
  (`mhrl/engine/oracle.py`). With no teacher model here, the reason is
  written from the gold SQL itself: which tables it uses, then what it does
  to them, step by step.

Then the rows, both splits, written as JSON lines (`input_ids` and
`loss_mask`, the shape the project's capture proxy records):

```text
3. the rows: input_ids and a loss_mask per question
  split  rows  tokens  read (mask 0)  reply (mask 1)  reply share  longest
  train   180  45,155         33,649          11,506        25.5%      417
  test     50  12,076          8,793           3,283        27.2%      399
  training scores only the reply tokens: the loss mask is 1 there and 0 over everything
    the model only reads
  train: ~/.cache/mhrl-course/14-sft-real-model/rows_train.jsonl
  test: ~/.cache/mhrl-course/14-sft-real-model/rows_test.jsonl
```

A quarter of the tokens are reply. The other three quarters are read and
never trained on: the loss mask is 0 there.

### One row, token by token

`--show 1` prints the first training row in its two parts. On a terminal
the read part is dimmed and the reply highlighted; in a file (as here) they
are printed one after the other:

```text
4. row 1 of the training rows: shop-medium-001
  read (mask 0), 205 tokens:
    <|startoftext|><|im_start|>system
    You answer questions about a SQLite database with one SQL query. Its result must be
      the answer and nothing else: one column, one row per item if the answer is a list.
      Reply with the query alone, after your reasoning.<|im_end|>
    <|im_start|>user
    The database has these tables:

    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT
      NULL, signup_date TEXT NOT NULL);
    CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL
      REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id),
      quantity INTEGER NOT NULL, unit_price REAL NOT NULL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES
      customers(id), order_date TEXT NOT NULL, payment_method TEXT NOT NULL);
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT
      NULL, price REAL NOT NULL);

    Which city had the most new customers sign up in 2024?<|im_end|>
    <|im_start|>assistant
    <think>
  reply (mask 1), 61 tokens:
    Use the customers table: keep the matching rows, group by city, then sort by the
      count, highest first, and keep the top row.</think>SELECT city FROM customers
      WHERE strftime('%Y', signup_date) = '2024' GROUP BY city ORDER BY COUNT(*) DESC
      LIMIT 1<|im_end|>
```

Then the place where the reply begins, with the position each token's score
is read at:

```text
  where the reply begins, token by token:
     pos      id  token            mask  its score is read at
     201  124899  '<|im_start|>'      0  (read only)
     202   63514  'assistant'         0  (read only)
     203     207  '\n'                0  (read only)
     204  124901  '<think>'           0  (read only)
     205   14374  'Use'               1  position 204 ('<think>')
     206     278  ' the'              1  position 205 ('Use')
     207   10699  ' customers'        1  position 206 (' the')
     208    4214  ' table'            1  position 207 (' customers')
      ...
     264      25  '1'                 1  position 263 (' ')
     265  124900  '<|im_end|>'        1  position 264 ('1')
```

The token at position 205, `'Use'`, is the first one the model is trained
to write, and its log-probability comes from the scores at position 204,
the `<think>` before it. That is the next-token shift. It is also why
position 0 can never be a target:

```text
  position 0, '<|startoftext|>': mask 0, and never a target in any row: a token's
    log-probability is read from the scores at the position before it, and nothing comes
    before position 0
  the project's item_from_sequence keeps the positions with mask 1 and i > 0: 61 here,
    the reply's 61, from position 205; given a mask with a 1 at position 0 too, it keeps
    61, from position 205
```

The project's `item_from_sequence` (`mhrl/training/common.py`, copied into
`rows.py` unchanged) turns `input_ids` and `loss_mask` into the positions
the trainer scores. It keeps a position only if its mask is 1 **and** it is
not position 0, so even a mask that wrongly marked position 0 would change
nothing: 61 positions, from 205, both ways.

The last lines answer a question you might have: why not let the chat
template render the finished conversation, reply included?

```text
  rendering the finished chat with the template instead gives 266 tokens; the first 204
    match this row, then the template has 'Use' where the row has '<think>' (it drops
    the '<think>' that ends the prompt), and it ends with '\n'. The row is the prompt
    the model read, then the tokens it wrote, as the project's engine records them;
    nothing is rendered again
```

Because it would not give the tokens the model saw. Rendered after the
fact, the template drops the `<think>` that ended the prompt (the reply was
written after it) and adds a line break after `<|im_end|>`. A row built that
way would train the model to write its reasoning right after
`<|im_start|>assistant`, which is not what happens when it answers. The
project never renders anything again: it keeps the prompt's tokens and the
tokens the model wrote, exactly (chapter 15 shows what goes wrong
otherwise).

`make_rows.py --fast` builds the rows with the tokenizer of Liquid's export
instead of ours; the run log checked that both runs wrote byte-identical
row files.

## The base model: pass@1 before any training

```bash
uv run eval_sql.py
```

`eval_sql.py` asks the 50 held-out questions, each in the chat
`make_rows.py` builds (the rule, the tables, the question, then
`<think>`). The model answers greedily, up to 1,024 tokens, the project's
cap for both training and evaluation (chapter 12). Then, as an agent's
answer would be, each reply is graded by a program:

1. the query is taken from what follows `</think>`;
2. it runs on the question's database, opened read only (it may not
   write, open another file, or run longer than 10 seconds);
3. its first column becomes the answer, formatted as chapter 1 formats
   answers, and is written to `answer.txt` in the question's own folder;
4. chapter 1's grader reads that file, as it reads an agent's.

```text
eval_sql.py · pass@1 on the held-out questions
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
adapter   none: the base model
questions 50 held out, over airline and clinic: databases no training row comes from
replies   greedy, one per question, up to 1,024 tokens (the project's cap), to the chat
  make_rows.py builds

1. base: each reply's SQL run, its result written to answer.txt, graded by chapter 1's grader
  task                 verdict  tokens  why
  clinic-medium-001    right       157  correct
  airline-medium-001   right       185  correct
  clinic-easy-001      right        66  correct
  airline-easy-001     right        65  correct
```

One line per question follows (all 50 are in the run log), then the score:

```text
2. base: pass@1 43 of 50 = 86.0% ± 4.9 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 50 questions like these were asked
  by database: airline 24/25, clinic 19/25
  by difficulty: easy 16/16, hard 10/14, medium 17/20
  the query came as: tool call 50
  reply length: median 230 tokens, longest 649; 0 cut off at the cap
  varies from run to run: the 50 replies took 367 s
```

43 of 50, **86.0% ± 4.9 points**: the error bar says that with another 50
questions like these, a score anywhere from about 81% to 91% would not be
surprising. The two databases differ (24 of 25 against 19 of 25), and every
easy question is right.

One line needs explaining: **the query came as: tool call 50**. The base
model did not write its query as text. Every reply ended with a tool call
in LFM2.5's own syntax, `<|tool_call_start|>[sql('SELECT ...')]<|tool_call_end|>`,
as if a tool named `sql` existed. LFM2.5 was trained to act through tools
(its chat template has tool-call markers), and here the system message
seems to set that habit off: chapter 11 asked the same kind of question
with no system message and read its queries from code blocks marked
`sql`. Asking for plain text does not stop it:
while building this chapter, two rules that said "plain text" and "no tool
call" were tried on 12 of these questions and got 12 tool calls each (a
building note kept beside the run log). The grader
reads the query out of the call, with Python's `ast` as the project's
engine reads tool calls (`mhrl/engine/toolcalls.py`): only a literal string
is taken, nothing is run.

The first right and wrong replies:

```text
3. a few replies: the first right ones and the first wrong ones
  clinic-medium-001: right
    question  Which specialty has the most doctors?
    after </think> (157 tokens in all): "<|tool_call_start|>[sql('SELECT specialty,
      COUNT(*) as doctor_count FROM doctors GROUP BY specialty ORDER BY doctor_count
      DESC LIMIT 1;')]<|tool_call_end|><|im_end|>"
    query (tool call)  SELECT specialty, COUNT(*) as doctor_count FROM doctors GROUP BY
      specialty ORDER BY doctor_count DESC LIMIT 1;
    answer.txt  General Practice
  airline-medium-001: right
    question  Which destination has the most flights departing from Milan?
    after </think> (185 tokens in all): "<|tool_call_start|>[sql('SELECT destination,
      COUNT(*) as flight_count FROM flights WHERE origin = \\'Milan\\' GROUP BY
      destination ORDER BY flight_count DESC LIMIT 1')]<|tool_call_end|><|im_end|>"
    query (tool call)  SELECT destination, COUNT(*) as flight_count FROM flights WHERE
      origin = 'Milan' GROUP BY destination ORDER BY flight_count DESC LIMIT 1
    answer.txt  Vienna
  clinic-medium-002: wrong
    question  How many appointments did Dr. Rafael Eriksen have in 2024?
    after </think> (112 tokens in all): '<|tool_call_start|>[sql(\'SELECT id, name,
      specialty FROM doctors WHERE name = "Rafael
      Eriksen"\')]<|tool_call_end|><|im_end|>'
    query (tool call)  SELECT id, name, specialty FROM doctors WHERE name = "Rafael
      Eriksen"
    why  expected '7', got '14'
    gold SQL  SELECT COUNT(*) FROM appointments a JOIN doctors d ON d.id = a.doctor_id
      WHERE d.name = 'Rafael Eriksen' AND strftime('%Y', a.appointment_date) = '2024'
  clinic-hard-005: wrong
    question  In which month of 2025 did the clinic collect the most in appointment
      fees? Give the full month name.
    after </think> (449 tokens in all): '<|tool_call_start|>[sql(\'SELECT
      strftime("%Y-%m", appointment_date) as month, SUM(fee) as total_fees FROM
      appointments WHERE strftime("%Y", appointment_date) = "2025" GROUP BY month ORDER
      BY total_fees DESC LIMIT 1;\')]<|tool_call_end|><|...'
    query (tool call)  SELECT strftime("%Y-%m", appointment_date) as month, SUM(fee) as
      total_fees FROM appointments WHERE strftime("%Y", appointment_date) = "2025" GROUP
      BY month ORDER BY total_fees DESC LIMIT 1;
    why  expected 'January', got '2025-01'
    gold SQL  SELECT CASE strftime('%m', a.appointment_date) WHEN '01' THEN 'January'
      WHEN '02' THEN 'February' WHEN '03' THEN 'March' WHEN '04' THEN 'April' WHEN '05'
      THEN 'May' WHEN '06' THEN 'June' WHEN '07' THEN 'July' WHEN '08' THEN 'August'
      WHEN '09' THEN 'September' WHEN '10' THEN 'October' WHEN '11' THEN 'November' WHEN
      '12' THEN 'December' END FROM appointments a JOIN doctors d ON d.id = a.doctor_id
      WHERE strftime('%Y', a.appointment_date) = '2025' GROUP BY strftime('%m',
      a.appointment_date) ORDER BY SUM(a.fee) DESC LIMIT 1
  saved: ~/.cache/mhrl-course/14-sft-real-model/eval_base.jsonl
```

`clinic-medium-002` is the model behaving like an agent: its query only
looks the doctor up (id 14), as a first step towards the answer, and in a
single reply there is no second step. `clinic-hard-005` gets the month right
but writes it as `2025-01` where the question asks for its name.

## Training: one epoch of the project's SFT

```bash
uv run train_sft.py
```

`train_sft.py` loads the model, builds the rows again (seconds) and checks
them against the file `make_rows.py` wrote, so it can never train on rows an
older run left behind. Then, before any training, a fresh adapter built the
chapter-13 way, and the no-op check:

```text
train_sft.py · SFT on the real model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
rows      180 training rows (identical to make_rows.py's rows_train.jsonl), 11,506 reply
  tokens
training  1 epoch, 8 rows a step: 23 AdamW steps at learning rate 5e-05, bias correction
  on; the project's SFT trainer and settings (mhrl/training/sft.py, mhrl/cli.py)
reason    written from the gold SQL: its tables, then what the query does
e.g.      "Use the customers table: keep the matching rows, group by city, then sort by
  the count, highest first, and keep the top row.</think>SELECT city FROM
  customers WHERE strftime('%Y', signup_date) = '2024' GROUP BY city ORDER BY
  COUNT(*) DESC LIMIT 1<|im_end|>"

1. a fresh adapter built the chapter-13 way: rank 16, scale 2.0, the last 12 layers, seed 0
  66 adapter pairs, 9,781,248 trainable numbers; B: 66 of 66 matrices all zeros
  max |logit difference| from the model before LoRA, over row 1's 266 positions x
    128,000 scores: 0.0; checked before training: the adapter starts as a no-op
```

The settings are the project's: rank 16, scale 2.0 and the last 12 layers
(`mhrl/config.py`), and `CliDefaults` in `mhrl/cli.py`: one epoch, 8 rows a
step, learning rate 5e-5 ("1e-4 over-fit the dry run"). A fresh adapter's B
matrices are all zero, so it adds exactly nothing: the scores of all 266
positions of row 1, over the whole vocabulary, are what they were before
LoRA, to the last bit.

The trainer is the project's `train_sft` (`mhrl/training/sft.py`) in
`sft.py`, with its file writing and printing moved out. Its two pieces:

```python
def token_logprobs(model: nn.Module, item: TrainItem) -> mx.array:
    tokens = mx.array(item.input_ids)[None]
    # One forward pass over the whole row except its last token: the last
    # token is never followed by anything to predict.
    hidden = model.model(tokens[:, :-1])[0]
    targets = mx.array([item.input_ids[p] for p in item.positions])
    # The token at position p is predicted by the scores at position p - 1.
    rows = mx.array([p - 1 for p in item.positions])
    ...  # scores only at those rows, 512 at a time, then log-softmax
```

and, for each step:

```python
            for item in batch:  # one row at a time: only one row's activations
                loss, grads = loss_and_grad(item)
                # Divide by the step's reply tokens, so the sum over its rows
                # is the gradient of the loss per reply token.
                grads = tree_map(lambda g, n=total: g / n, grads)
                summed = grads if summed is None else tree_map(mx.add, summed, grads)
                mx.eval(summed, loss)
                batch_loss += float(loss)
            optimizer.update(model, summed)
```

`loss_and_grad` is minus the sum of the row's reply-token
log-probabilities, and only the adapter's numbers get gradients. The
optimizer is AdamW with bias correction on (chapter 13). The project's
scoring computes the 128,000 scores only at the positions it trains, which
is how its trainer fits long agent transcripts in memory (chapter 25); these
rows are short, so nothing else from that part of the project is needed.

The 23 steps:

```text
2. training: one forward pass per row, gradients added up row by row
  step  epoch  rows  reply tokens  loss per token  seconds  peak
     1      1     8           441          2.4577      5.4  3.16 GB
     2      1     8           496          2.3961      5.2  3.20 GB
     3      1     8           562          2.3753      5.2  3.20 GB
     4      1     8           571          1.9142      5.1  3.24 GB
     5      1     8           468          2.0107      5.1  3.24 GB
     6      1     8           476          1.5958      4.8  3.24 GB
     7      1     8           486          1.6135      4.7  3.24 GB
     8      1     8           385          1.3387      4.9  3.24 GB
     9      1     8           561          1.2044      4.9  3.24 GB
    10      1     8           539          0.8802      4.8  3.71 GB
    11      1     8           523          0.9021      4.9  3.71 GB
    12      1     8           519          0.8539      5.4  3.71 GB
    13      1     8           582          0.7917      6.4  3.71 GB
    14      1     8           411          0.9507      5.1  3.71 GB
    15      1     8           684          0.5564      5.3  3.76 GB
    16      1     8           488          0.5611      4.9  3.76 GB
    17      1     8           443          0.5685      4.5  3.76 GB
    18      1     8           633          0.4926      5.1  3.76 GB
    19      1     8           411          0.5244      4.6  3.76 GB
    20      1     8           462          0.5056      4.5  3.76 GB
    21      1     8           447          0.3700      4.7  3.76 GB
    22      1     8           513          0.4714      4.9  3.76 GB
    23      1     4           405          0.3199      3.1  3.76 GB
  loss per token: minus the sum of the reply tokens' log-probabilities, divided by the
    step's reply-token count, before the step's update
  varies from run to run: the seconds (23 steps took 114 s) and the peak memory (3.76
    GB, the weights included)
```

The loss per reply token starts at 2.46 (a typical reply token gets a
chance of about e^-2.46, 1 in 12) and ends at 0.32 (about 3 in 4). It does
not fall smoothly: each step sees 8 different rows, and some are harder to
predict than others (step 14 is above step 13). The memory
peak, 3.76 GB with the weights, is close to the project's main run's 3.83
to 3.95 GB; the time, about 5 seconds a step here, varies.

```text
3. saved
  ~/.cache/mhrl-course/14-sft-real-model/sft.safetensors (39,139,440 bytes)
  ~/.cache/mhrl-course/14-sft-real-model/sft.metrics.jsonl (one line per step)
  next: uv run eval_sql.py --adapter sft.safetensors
```

### The project's runs (recorded, not run here)

The script ends with the project's own SFT runs, read from their metrics
files bundled in `data/project_run/`:

```text
4. the project's runs (recorded, not run here)
  main run, runs/main/sft/metrics.jsonl: the same trainer, 1 epoch, 8 rows a step, lr
    5e-05, on 120 rows captured through mini-swe-agent and Claude Code (chapter 25): 15
    steps
    loss per token: 2.55 2.28 1.77 1.15 0.66 0.45 0.37 0.30 0.31 0.27 0.21 0.17 0.13
      0.10 0.11
    1,243 to 1,526 reply tokens a step; 22.65 to 50.35 s a step (9.1 minutes); peak 3.83
      to 3.95 GB
  dry run, runs/dry/sft/metrics.jsonl: 48 rows, 6 steps; lr 1e-4 according to the
    comment in mhrl/cli.py (the file records no learning rate)
    loss per token: 2.33 1.67 0.69 0.17 0.10 0.10
    the comment says 'loss 0.10 by step 4 of 6'; the file says 0.17 at step 4 and 0.10
      first at step 5
```

- **The project's main run** trained the same code with the same settings,
  on different rows: 120 conversations captured as the answer-key teacher
  worked inside mini-swe-agent and Claude Code (chapters 20 to 25 build
  those). Its loss fell from 2.55 to 0.11 in 15 steps, at a peak of 3.95 GB.
  Its rows were longer (1,243 to 1,526 reply tokens a step against our 385
  to 684) and took 22.65 to 50.35 seconds a step. Chapter 8's `real_sft.py`
  prints what it did on the project's held-out tasks.
- **The project's dry run** is the one `mhrl/cli.py`'s comment describes:
  "1e-4 over-fit the dry run (loss 0.10 by step 4 of 6)". Its file does not
  quite say that: the loss was 0.17 at step 4 and 0.10 (0.0955) first at
  step 5. The file records no learning rate; 1e-4 comes from the comment
  alone.

## After training: pass@1 with the adapter

```bash
uv run eval_sql.py --adapter sft.safetensors
```

The same 50 questions, the same chat, the same cap, now with the adapter
loaded. The base model's answers are read back from the first run's file
(they are reused only if they were made under the same cap; otherwise the
base model answers them first, with a fresh adapter):

```text
eval_sql.py · pass@1 on the held-out questions
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
adapter   ~/.cache/mhrl-course/14-sft-real-model/sft.safetensors
questions 50 held out, over airline and clinic: databases no training row comes from
replies   greedy, one per question, up to 1,024 tokens (the project's cap), to the chat
  make_rows.py builds
  base answers: read from eval_base.jsonl (eval_sql.py without --adapter, the same
    1,024-token cap)

1. sft: each reply's SQL run, its result written to answer.txt, graded by chapter 1's grader
  task                 verdict  tokens  why
```

After the 50 verdicts:

```text
2. sft: pass@1 35 of 50 = 70.0% ± 6.5 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 50 questions like these were asked
  by database: airline 20/25, clinic 15/25
  by difficulty: easy 14/16, hard 7/14, medium 14/20
  the query came as: no query 8, plain text 42
  reply length: median 66 tokens, longest 1,024; 8 cut off at the cap
  varies from run to run: the 50 replies took 409 s
```

**35 of 50, 70.0% ± 6.5 points**, against the base model's 43. The replies
are now plain text (42), as the rows taught, and short: a median of 66
tokens against 230. Read question by question:

```text
4. base against sft, question by question
  gained 2 (wrong before, right now), lost 10 (right before, wrong now), right both
    times 33, wrong both times 5
  exact p = 0.0386: the chance of a split at least this uneven among the 12 questions
    that changed, if the two models were equally good (McNemar's exact test)
  gained: clinic-hard-006, clinic-medium-010
  lost: clinic-medium-003, airline-easy-003, clinic-hard-003, clinic-medium-005,
    airline-medium-005, clinic-hard-004, airline-hard-004, airline-hard-005,
    clinic-easy-006, clinic-medium-008

```

Two questions gained, ten lost. If the adapter had made no difference, a
split of 12 changed questions at least this uneven would turn up 3.86% of
the time (p = 0.0386): this one is unlikely to be luck. SFT made the model
**worse** on databases it had not seen.

The same four questions the base run showed, after training:

```text
3. the same questions as the base run showed, after training
  clinic-medium-001: right
    question  Which specialty has the most doctors?
    after </think> (35 tokens in all): 'SELECT specialty FROM doctors GROUP BY specialty
      ORDER BY COUNT(*) DESC LIMIT 1<|im_end|>'
    query (plain text)  SELECT specialty FROM doctors GROUP BY specialty ORDER BY
      COUNT(*) DESC LIMIT 1
    answer.txt  General Practice
  airline-medium-001: right
    question  Which destination has the most flights departing from Milan?
    after </think> (46 tokens in all): 'SELECT destination FROM flights GROUP BY
      destination ORDER BY COUNT(*) DESC LIMIT 1<|im_end|>'
    query (plain text)  SELECT destination FROM flights GROUP BY destination ORDER BY
      COUNT(*) DESC LIMIT 1
    answer.txt  Vienna
  clinic-medium-002: wrong
    question  How many appointments did Dr. Rafael Eriksen have in 2024?
    after </think> (1024 tokens in all): '(no </think> in 1024 tokens: still reasoning)'
    why  cut off by the length cap while still reasoning
    gold SQL  SELECT COUNT(*) FROM appointments a JOIN doctors d ON d.id = a.doctor_id
      WHERE d.name = 'Rafael Eriksen' AND strftime('%Y', a.appointment_date) = '2024'
  clinic-hard-005: wrong
    question  In which month of 2025 did the clinic collect the most in appointment
      fees? Give the full month name.
    after </think> (71 tokens in all): "SELECT strftime('%Y-%m', appointment_date) as
      month, SUM(fee) FROM appointments GROUP BY month ORDER BY month, SUM(fee) DESC
      LIMIT 1<|im_end|>"
    query (plain text)  SELECT strftime('%Y-%m', appointment_date) as month, SUM(fee)
      FROM appointments GROUP BY month ORDER BY month, SUM(fee) DESC LIMIT 1
    why  expected 'January', got '2024-01'
    gold SQL  SELECT CASE strftime('%m', a.appointment_date) WHEN '01' THEN 'January'
      WHEN '02' THEN 'February' WHEN '03' THEN 'March' WHEN '04' THEN 'April' WHEN '05'
      THEN 'May' WHEN '06' THEN 'June' WHEN '07' THEN 'July' WHEN '08' THEN 'August'
      WHEN '09' THEN 'September' WHEN '10' THEN 'October' WHEN '11' THEN 'November' WHEN
      '12' THEN 'December' END FROM appointments a JOIN doctors d ON d.id = a.doctor_id
      WHERE strftime('%Y', a.appointment_date) = '2025' GROUP BY strftime('%m',
      a.appointment_date) ORDER BY SUM(a.fee) DESC LIMIT 1
```

The first two are right, in about a quarter of the tokens, but
`airline-medium-001` now leaves out the question's condition (flights from Milan) and is right
only because Vienna happens to top the whole table. `clinic-medium-002`
never finishes reasoning. Some of the questions that changed:

```text
5. some questions that changed, after training
  clinic-hard-006: right
    question  How many patients saw at least 3 different doctors in 2025?
    after </think> (93 tokens in all): "SELECT COUNT(*) FROM (SELECT p.id FROM
      appointments a JOIN patients p ON a.patient_id = p.id WHERE strftime('%Y',
      a.appointment_date) = '2025' GROUP BY a.patient_id HAVING COUNT(DISTINCT
      a.doctor_id) >= 3)<|im_end|>"
    query (plain text)  SELECT COUNT(*) FROM (SELECT p.id FROM appointments a JOIN
      patients p ON a.patient_id = p.id WHERE strftime('%Y', a.appointment_date) =
      '2025' GROUP BY a.patient_id HAVING COUNT(DISTINCT a.doctor_id) >= 3)
    answer.txt  25
  clinic-medium-010: right
    question  Which doctor had the most appointments in 2024?
    after </think> (79 tokens in all): "SELECT d.name FROM doctors d JOIN appointments a
      ON d.id = a.doctor_id WHERE strftime('%Y', a.appointment_date) = '2024' GROUP BY
      d.id ORDER BY COUNT(*) DESC LIMIT 1<|im_end|>"
    query (plain text)  SELECT d.name FROM doctors d JOIN appointments a ON d.id =
      a.doctor_id WHERE strftime('%Y', a.appointment_date) = '2024' GROUP BY d.id ORDER
      BY COUNT(*) DESC LIMIT 1
    answer.txt  Dario Eriksen
  clinic-medium-003: wrong
    question  How much has Yusuf Reyes paid in appointment fees in total?
    after </think> (43 tokens in all): 'SELECT SUM(appointments.fee) FROM appointments
      JOIN patients ON appointments.patient_id = patients.id<|im_end|>'
    query (plain text)  SELECT SUM(appointments.fee) FROM appointments JOIN patients ON
      appointments.patient_id = patients.id
    why  expected '523.00', got '43977.50'
    gold SQL  SELECT ROUND(SUM(a.fee), 2) FROM appointments a JOIN patients pa ON pa.id
      = a.patient_id WHERE pa.name = 'Yusuf Reyes'
  airline-easy-003: wrong
    question  What is the average price of a Business class booking? Round to 2 decimal
      places.
    after </think> (1024 tokens in all): '(no </think> in 1024 tokens: still reasoning)'
    why  cut off by the length cap while still reasoning
    gold SQL  SELECT ROUND(AVG(price), 2) FROM bookings WHERE seat_class = 'Business'
```

### Why it got worse: the model copied our reasons too

Eight replies hit the 1,024-token cap without ever writing `</think>`. Here
is what they were doing (the replies are in `eval_sft.jsonl`, under
`COURSE_HOME/14-sft-real-model`):

```bash
cd ~/.cache/mhrl-course/14-sft-real-model
python3 -c "
import json, pathlib
for path in sorted(pathlib.Path('.').glob('eval_*.jsonl')):
    for line in path.open():
        reply = json.loads(line)
        if reply['finish_reason'] == 'length':
            text = reply['reply']
            print(path.stem, reply['task_id'], text.count('matching rows'), repr(text[-40:]))
"
```

It lists every reply that hit the cap, in every evaluation saved so far,
with how many times it says "matching rows" and how it ends (the base
model's and the fixed-reason run's files have none):

```text
eval_sft clinic-medium-002 169 ', then keep the matching rows, then keep'
eval_sft airline-easy-003 170 'tching rows, then keep the matching rows'
eval_sft clinic-medium-005 166 'ching rows, then keep the matching rows,'
eval_sft clinic-hard-004 169 'en keep the matching rows, then keep the'
eval_sft airline-hard-004 253 ' matching rows, keep matching rows, keep'
eval_sft airline-hard-005 169 ', then keep the matching rows, then keep'
eval_sft clinic-easy-006 170 'ching rows, then keep the matching rows,'
eval_sft clinic-medium-008 169 ', then keep the matching rows, then keep'
eval_sft_lr0.0001_epochs3 airline-hard-006 1 "ime('%m', departure_date) = '01' AND str"
eval_sft_lr0.0001_epochs3 clinic-medium-010 1 "024' AND strftime('%Y', appointment_date"
```

The first eight lines are the SFT model. Each one starts a reason the way
the training reasons start, reaches "keep the matching rows", and then
writes ", then keep the matching rows" (or "keep matching rows") over and
over, 166 to 253 times, until the cap stops it. Seven of the ten lost
questions are these loops; the eighth loop, `clinic-medium-002`, was wrong
before too. The other three losses are wrong SQL (a missing condition, a
wrong column, an empty result).

Why would a model loop? Greedy decoding always takes the likeliest next
token, and these replies show that after "keep the matching rows," the
likeliest next words were the same phrase again, every time. The
reasons `make_rows.py` writes are templates: a few fixed phrases ("join
them", "keep the matching rows", "group by ...") strung together with
commas and "then", 180 times over. The model learned the template's words
better than when to stop saying them. This is chapter 8's lesson about
demonstrations at the scale of one sentence: the student copies everything
the teacher does, including its habits. The project met the same thing one
level up: demonstrations that always took four steps taught Claude Code
"finish on reply four" (`mhrl/engine/oracle.py`, the dry run's 100% to
17%), so its teacher varies its path.

### Testing the explanation: one fixed sentence (--reason fixed)

If the loops come from the template, rows whose reason cannot loop should
not loop. `--reason fixed` starts every reply with one sentence, the line
the project's answer-key teacher writes before it runs its query (`"Now I
can answer the question with one query."`, from `mhrl/engine/oracle.py`),
and changes nothing else:

```bash
uv run train_sft.py --reason fixed
uv run eval_sql.py --adapter sft_reasonfixed.safetensors
```

```text
train_sft.py · SFT on the real model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
rows      180 training rows (make_rows.py had not run: written to
  rows_train_fixed.jsonl), 9,125 reply tokens
training  1 epoch, 8 rows a step: 23 AdamW steps at learning rate 5e-05, bias correction
  on; the project's SFT trainer and settings (mhrl/training/sft.py, mhrl/cli.py)
reason    one fixed sentence for every row, 'Now I can answer the question with one
  query.', the line the project's answer-key teacher writes before its query
  (--reason fixed)
e.g.      "Now I can answer the question with one query.</think>SELECT city FROM
  customers WHERE strftime('%Y', signup_date) = '2024' GROUP BY city ORDER BY
  COUNT(*) DESC LIMIT 1<|im_end|>"
```

The rows go to their own file and the adapter gets its own name, so the
default ones are untouched. Training starts and ends:

```text
2. training: one forward pass per row, gradients added up row by row
  step  epoch  rows  reply tokens  loss per token  seconds  peak
     1      1     8           355          2.6234      6.0  3.09 GB
```

```text
    19      1     8           338          0.1638      6.2  3.68 GB
    20      1     8           343          0.1190      6.4  3.68 GB
    21      1     8           381          0.0570      5.3  3.68 GB
    22      1     8           392          0.0956      5.1  3.68 GB
    23      1     4           340          0.1186      3.1  3.68 GB
  loss per token: minus the sum of the reply tokens' log-probabilities, divided by the
    step's reply-token count, before the step's update
  varies from run to run: the seconds (23 steps took 133 s) and the peak memory (3.68
    GB, the weights included)
```

(A fixed sentence is easy to predict, which is part of why this loss ends
lower than the other run's: losses of different rows are not comparable.)
Then the held-out questions:

```text
2. sft_reasonfixed: pass@1 39 of 50 = 78.0% ± 5.9 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 50 questions like these were asked
  by database: airline 21/25, clinic 18/25
  by difficulty: easy 16/16, hard 9/14, medium 14/20
  the query came as: plain text 50
  reply length: median 44 tokens, longest 97; 0 cut off at the cap
  varies from run to run: the 50 replies took 125 s
```

```text
4. base against sft_reasonfixed, question by question
  gained 3 (wrong before, right now), lost 7 (right before, wrong now), right both times
    36, wrong both times 4
  exact p = 0.3438: the chance of a split at least this uneven among the 10 questions
    that changed, if the two models were equally good (McNemar's exact test)
  gained: clinic-medium-009, clinic-hard-007, clinic-medium-010
  lost: airline-medium-001, clinic-hard-003, clinic-medium-005, airline-hard-004,
    clinic-medium-007, clinic-medium-008, airline-medium-009

```

No reply hits the cap; the longest is 97 tokens. **39 of 50, 78.0% ± 5.9
points**: three gained, seven lost, p = 0.34 against the base model, so 50
questions cannot tell this adapter from the base model, while they did tell
the looping one (p = 0.039). The loops were the template's doing.

It is still four below the base model's 43, and one difference stands out
in every adapter: the base model's replies had a median of 230 tokens
(reasoning, then the tool call), the adapters' 66 and, here, 44, the SQL
included. Training on one-line reasons teaches the model to stop thinking
almost at once. Its first wrong reply here,
`airline-medium-001`, lists Milan's destinations instead of counting
them, the kind of slip a few lines of reasoning would catch. This is
consistent with the numbers, not proved by them: 50 questions are too few
to measure a gap of four.

## Over-fitting: a higher learning rate, three epochs

```bash
uv run train_sft.py --lr 1e-4 --epochs 3
uv run eval_sql.py --adapter sft_lr0.0001_epochs3.safetensors
```

Twice the learning rate, three passes over the same rows, 69 steps:

```text
train_sft.py · SFT on the real model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours, ours, built by chapter 11's
  convert.py
rows      180 training rows (identical to make_rows.py's rows_train.jsonl), 11,506 reply
  tokens
training  3 epochs, 8 rows a step: 69 AdamW steps at learning rate 0.0001, bias
  correction on (--lr; the project's is 5e-05; --epochs; the project's is 1);
  the project's SFT trainer and settings (mhrl/training/sft.py, mhrl/cli.py)
```

```text
3. loss per token, epoch by epoch (the same rows each time, reshuffled)
  epoch 1: 0.7373 over its 11,506 reply tokens, last step 0.1327
  epoch 2: 0.0706 over its 11,506 reply tokens, last step 0.0247
  epoch 3: 0.0254 over its 11,506 reply tokens, last step 0.0019
```

The loss falls to almost nothing: 0.0019 per reply token at the last step,
a chance of 0.998 for each token of the training replies. The model has
nearly memorised them. On the held-out questions:

```text
2. sft_lr0.0001_epochs3: pass@1 38 of 50 = 76.0% ± 6.0 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 50 questions like these were asked
  by database: airline 18/25, clinic 20/25
  by difficulty: easy 16/16, hard 7/14, medium 15/20
  the query came as: plain text 50
  reply length: median 53 tokens, longest 1,024; 2 cut off at the cap
  varies from run to run: the 50 replies took 212 s
```

```text
4. base against sft_lr0.0001_epochs3, question by question
  gained 4 (wrong before, right now), lost 9 (right before, wrong now), right both times
    34, wrong both times 3
  exact p = 0.2668: the chance of a split at least this uneven among the 13 questions
    that changed, if the two models were equally good (McNemar's exact test)
  gained: clinic-medium-002, clinic-hard-006, clinic-medium-009, clinic-hard-007
  lost: airline-hard-001, airline-medium-002, clinic-medium-004, clinic-hard-003,
    airline-hard-004, airline-hard-005, clinic-medium-007, airline-hard-007,
    airline-medium-010

```

38 of 50, 76.0% ± 6.0 points; four gained, nine lost, p = 0.27 against the
base model. The training loss at the last step fell from 0.32 to 0.0019
between the two runs and the held-out score did not follow it: it is still
below the base model (the run log compares each adapter with the base
model only, not with each other). The
over-fitted model loops too, inside its SQL: its two capped replies (the
last two lines of the loop check above) repeat a condition of their query
until the cap.
The project's dry run is the same warning, at its scale: with 1e-4 (by the
comment in `mhrl/cli.py`) its loss was 0.17 at step 4 and 0.10 at step 5,
and the project halved the rate.

## Forgetting: five prompts that have nothing to do with SQL

The last section of `eval_sql.py --adapter` asks five plain questions (no
system rule, no tables) before training (a fresh adapter, whose B matrices
are zero: the base model exactly) and after, and also scores the base
model's own reply under the adapter:

```text
6. what else the adapter changed: five prompts that are not about SQL
  each asked as a plain question (no system rule, no tables), greedy, before training (a
    fresh adapter: every B is zero, so this is the base model exactly) and after
  'Write a haiku about the sea.'
    before (262 tokens, 'stop'): 'Waves crash on the shore,  \nSalt spray dances in the
      wind,  \nOcean breathes in peace.<|im_end|>'
    after  (586 tokens, 'stop'): 'Salt spray on my face,  \nWaves crash upon the white
      sand,  \nSea breathes in my soul.<|im_end|>'
    the before reply's log-probability per token: -0.1319 before, -0.1403 after
  'Translate into French: The library opens at nine in the morning.'
    before (690 tokens, 'stop'): "La bibliothèque s'ouvre à neuf le matin.<|im_end|>"
    after  (591 tokens, 'stop'): "La bibliothèque s'ouvre à neuf le matin.<|im_end|>"
    the before reply's log-probability per token: -0.2509 before, -0.2649 after
  'What is the capital of Australia?'
    before (46 tokens, 'stop'): 'The capital of Australia is **Canberra**.<|im_end|>'
    after  (68 tokens, 'stop'): 'The capital of Australia is **Canberra**.<|im_end|>'
    the before reply's log-probability per token: -0.1179 before, -0.1517 after
  'A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?'
    before (1024 tokens, 'length'): '(no </think>: still reasoning at the cap)'
    after  (1024 tokens, 'length'): '(no </think>: still reasoning at the cap)'
    the before reply's log-probability per token: -0.2186 before, -0.2251 after
  'Give three short tips for sleeping well.'
    before (169 tokens, 'stop'): "1. **Keep a consistent sleep schedule** – Go to bed
      and wake up at the same time every day, even on weekends, to regulate your body's
      internal clock.\n2. **Create a relaxing bedtime routine** – Wind down with calming
      activities like readi..."
    after  (139 tokens, 'stop'): '1. **Keep a consistent sleep schedule** – Go to bed
      and wake up at the same time every day, even on weekends.\n2. **Create a relaxing
      bedtime routine** – Wind down with calming activities like reading, gentle
      stretching, or a warm bath.\n3...'
    the before reply's log-probability per token: -0.1935 before, -0.2144 after
  after training, 0 of 5 replies contain SELECT; the base model's own replies became, on
    average, 0.0167 lower in log-probability per token
```

The answers survive: the same translation, the same capital, the same
tips worded a little shorter, a different haiku of the same kind, and no
SQL anywhere. What moved
is smaller and measured in the last line of each prompt: the adapter finds
the base model's own replies a little less likely, by 0.0167 per token on
average. With the over-fitted adapter it is 0.0310, nearly twice as much:

```text
  after training, 0 of 5 replies contain SELECT; the base model's own replies became, on
    average, 0.0310 lower in log-probability per token
```

and with the fixed-sentence adapter 0.0211:

```text
  after training, 0 of 5 replies contain SELECT; the base model's own replies became, on
    average, 0.0211 lower in log-probability per token
```

The over-fitted adapter also changed how the model gets there. Where the
base model was still reasoning about the train at the 1,024-token cap, it
now answers in 237 tokens:

```text
  'A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?'
    before (1024 tokens, 'length'): '(no </think>: still reasoning at the cap)'
    after  (237 tokens, 'stop'): 'The train leaves at 3:40 pm and the trip takes 2 hours
      35 minutes. Adding 2 hours gives 5:40 pm, then adding 35 minutes gives 6:15
      pm.\n\n\\boxed{6:15 \\text{ pm}}<|im_end|>'
```

So here forgetting is real but small: a measurable drift on unrelated text,
larger with harder training, and no lost answers on these five prompts.
LoRA on the last 12 layers, a small learning rate and few steps are why it
is small; they do not make it zero. The project measured no general ability
before and after its training, so there is no project number to compare.

## Fast mode

```bash
uv run make_rows.py --fast
uv run train_sft.py --fast
uv run eval_sql.py --fast --adapter sft_limit40.safetensors
```

`--fast` uses Liquid's export instead of chapter 11's build. For
`train_sft.py` it also trains on 40 rows (the plan's `--limit 40`), chosen
round-robin over database and difficulty as the project's `--limit` does:
5 steps.

```text
2. training: one forward pass per row, gradients added up row by row
  step  epoch  rows  reply tokens  loss per token  seconds  peak
     1      1     8           454          2.7076      5.1  3.12 GB
     2      1     8           366          2.6160      5.3  3.21 GB
     3      1     8           444          2.3105      5.2  3.23 GB
     4      1     8           742          1.6781      6.1  3.72 GB
     5      1     8           447          1.8944      5.2  3.72 GB
  loss per token: minus the sum of the reply tokens' log-probabilities, divided by the
    step's reply-token count, before the step's update
  varies from run to run: the seconds (5 steps took 27 s) and the peak memory (3.72 GB,
    the weights included)
```

The first step's loss differs from the full run's (2.7076 against 2.4577)
because its 8 rows are different ones. For `eval_sql.py` it asks 10
balanced questions and skips the unrelated prompts; the base model's
answers are read from the full run's file, which holds these 10:

```text
2. sft_limit40: pass@1 10 of 10 = 100.0% ± 0.0 points
  pass@1: the share answered right on the one greedy try; error bar: ± one standard
    error, sqrt(p x (1 - p) / n), how far the score would typically move by luck alone
    if another 10 questions like these were asked
  by database: airline 6/6, clinic 4/4
  by difficulty: easy 4/4, hard 3/3, medium 3/3
  the query came as: tool call 10
  reply length: median 156 tokens, longest 336; 0 cut off at the cap
  varies from run to run: the 10 replies took 87 s
```

```text
4. base against sft_limit40, question by question
  gained 0 (wrong before, right now), lost 0 (right before, wrong now), right both times
    10, wrong both times 0
  exact p = 1.0000: the chance of a split at least this uneven among the 0 questions
    that changed, if the two models were equally good (McNemar's exact test)
  saved: ~/.cache/mhrl-course/14-sft-real-model/eval_sft_limit40.jsonl

5. five prompts that are not about SQL: skipped with --fast
```

Five steps leave the model as it was on these questions: still tool calls,
still right. The rows from `make_rows.py --fast` are byte-identical to the
default run's (the run log compared the files).

## The answer to the puzzle

No. Every training reply was correct, the loss fell from 2.46 to 0.32, and
on 50 questions about databases it had never seen the model went from 43
right to 35, with ten lost and two gained (p = 0.039). Imitation copies the
whole reply, not only the part you care about: the model learned the SQL's
shape, and also our template's phrases, so well that it repeated one until
the cap on eight questions, and learned to stop reasoning after one line.
Replacing the template with one fixed sentence removed the loops (39 of 50,
no longer distinguishable from the base model's 43). Training harder drove
the loss to 0.0019 without bringing the score back. SFT changed what the
model writes (plain text instead of tool calls, under a third of the
length) much more reliably than how well it answers.

## What the chapter showed, and where it differed from the plan

| the plan said | measured |
| --- | --- |
| bundles 230 tasks (180 training over shop, school, library; 50 held out over clinic, airline) with their SQLite files | as planned (chapter 1's files, byte for byte) |
| make_rows renders a chat (system rule + schema + question); reply = one-line reason, `</think>`, gold SQL; records the loss mask | as planned; the reason is written from the gold SQL (no teacher model here); `--reason fixed` added |
| `make_rows.py --show 1` dims read tokens, highlights reply tokens, shows the first token is never a target | as planned on a terminal; in a file the two parts are printed apart. Added: the template check (rendering the finished chat drops the `<think>`) |
| `eval_sql.py`: the base model's pass@1 on the 50 held-out questions, a few right and wrong replies | 43 of 50, 86.0% ± 4.9 points. Not in the plan: every base reply was an LFM2.5 tool call, which the grader reads with `ast` |
| define 'error bar' in one line at the first evaluation (chapter 15's build note) | done: ± one standard error, sqrt(p x (1 - p) / n) |
| `train_sft.py --epochs 1` (23 steps, minutes) prints falling loss per token and peak memory | 23 steps, about 2 minutes; 2.46 to 0.32; peak 3.76 GB |
| 'a fresh adapter built the chapter-13 way', the zero-init no-op check printed before training | as planned: 0.0 over 266 positions x 128,000 scores |
| `eval_sql.py --adapter sft.safetensors` prints the new pass@1 and the same replies | 35 of 50: SFT made the held-out score worse (2 gained, 10 lost, p = 0.0386), 7 of the 10 losses being repetition loops from the templated reasons. The plan did not foresee a drop |
| `--lr 1e-4 --epochs 3` drives the loss toward zero while the held-out score does not follow | as planned: 0.0019 at the last step; 38 of 50 |
| 'the project's dry run at 1e-4 hit loss 0.10 by step 4 of 6' | corrected: its file says 0.17 at step 4 and 0.10 (0.0955) at step 5; the 1e-4 is from `mhrl/cli.py`'s comment only |
| five unrelated prompts before and after show forgetting | the same answers (a different haiku), no SQL; the base model's replies 0.0167 (one epoch) to 0.0310 (over-fitted) less likely per token |
| fast mode: `--limit 40` | `train_sft.py --fast` is `--limit 40` on Liquid's export (5 steps); `eval_sql.py --fast` asks 10 questions |
| folder `course/14-sft-real-model/` | `14-sft-real-model/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its two recorded SFT runs,
  printed by `train_sft.py` in a section headed "the project's runs
  (recorded, not run here)".
- **The article's numbers**: there are none in this chapter.
- **A difference from the project**: the project trains on conversations
  captured inside real agent programs, several calls long; this chapter
  trains on one question and one reply, so the loop can be measured in
  minutes. Chapter 25 trains the project's way.
- **Small imprecisions left in the reasons**: a query with an inner query
  names the inner query's tables too, and "join them" can describe a join
  inside it; the two INTERSECT queries are described without the
  INTERSECT. They are a handful of the 230 rows and are left as they are.
- **The base model's answers are reused** when they were made under the
  same cap, and a run under another `--cap` replaces them. Nothing checks
  the prompt: change it and delete `eval_base.jsonl` to measure the base
  model again.

## What is in this folder

- `make_rows.py`: the questions into rows, with the loss mask; `--show`
  prints one.
- `train_sft.py`: the no-op check, the project's SFT loop on the rows, the
  adapter saved, the project's recorded runs.
- `eval_sql.py`: greedy replies to the held-out questions, graded through
  `answer.txt`; pass@1, error bar, question-by-question comparison, five
  unrelated prompts.
- `rows.py`: the chat, the reason, the row and its mask; the project's
  `TrainItem`, `item_from_sequence` and `balanced`.
- `sft.py`: the project's `token_logprobs`, `batches` and `train_sft`
  loop.
- `grading.py`: finding the query (tool call, code block or text),
  running it, writing `answer.txt`, saving answers.
- `database.py`: reading a database's tables and running a query, read
  only, with ATTACH refused and a 10-second limit.
- `stats.py`: pass@1, the error bar, gained, lost and the exact p-value.
- `project_run.py`: the project's two recorded SFT runs.
- `report.py`: how the scripts print.
- `lora.py`, `lfm.py`, `lfm_files.py`: chapter 13's adapter code, loading
  the model and a greedy reply, where files live and the project's
  settings.
- `tasks.py`, `grader.py`: chapter 1's task record and grader.
- `data/`: the 230 questions, the five databases, the project's two SFT
  metrics files.

## Words from this chapter

- **SFT, imitation**: training a model to make a teacher's reply more
  likely.
- **Training row, loss mask**: a prompt and a reply as one list of tokens;
  1 where the model learns, 0 where it only reads.
- **Scoring in one pass (teacher forcing)**: reading the whole row at once
  and scoring every recorded next token.
- **Next-token shift**: a token is scored at the position before it, so the
  first token is never a target.
- **Loss per token, token-level normalisation**: the summed loss divided by
  the step's reply tokens, so every token counts the same.
- **Gradient accumulation**: one row at a time, gradients added up, one
  update per step.
- **Step, batch, epoch**: one update; the rows it reads; one pass over all
  rows.
- **pass@1, error bar**: the share right on one try; ± one standard error.
- **Gained, lost, exact p-value**: two runs compared question by question.
- **Over-fitting**: the training loss falls, the held-out score does not
  follow.
- **Forgetting**: training on one job wearing away others; here small and
  measurable.
- **Repetition loop**: greedy decoding repeating a phrase until the cap,
  learned here from templated reasons.

Next: chapter 15, GRPO on the real model, and why exact tokens matter.
