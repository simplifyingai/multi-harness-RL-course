# Chapter 25. SFT through the harnesses (and fitting it in 9 GB)

Chapter 14 taught LFM2.5-2.6B from rows we rendered ourselves: a prompt we
wrote, a reply we wrote, one chat template. Chapter 24 measured the
untrained model in three real harnesses. This chapter does what the project
does next: it teaches the model inside the harnesses themselves. A teacher
plays the model in mini-swe-agent and Claude Code, the capture proxy records
every call in the student's own tokens, and the trainer learns from exactly
the tokens the teacher wrote, and from nothing the harness wrote around them.

The puzzle: the project's teacher is a script that cannot be wrong, yet its
first version made Claude Code worse. In the project's dry run, Claude Code
solved 6 of 6 held-out tasks before training and 1 of 6 after. What can a
perfect teacher teach that is wrong? And a Claude Code transcript is 2,600
tokens long: how does training on it fit in 9 GB of a laptop?

## Words first

- The **answer-key teacher** is a script that plays the model's part. The
  task file knows each question's correct query (its gold SQL), so the
  teacher can always write the reply a careful analyst would: look at the
  tables, run the right query, save the answer. It never guesses.
- A **demonstration** is one whole attempt by the teacher, inside a real
  harness, recorded by the capture proxy (chapter 20) as one training row:
  the token ids the model would read and write, with a **loss mask** that is
  1 on every token the teacher wrote and 0 on everything the harness wrote
  (its system prompt, its tool list, every tool result).
- **Rollout concurrency** is how many attempts run at the same time. Here
  4, the project's setting, each in its own folder with its own session,
  all answered by the teacher's one engine thread, one request at a time.
- **Calls per solve** is how many model calls a solved task took, averaged
  over the solved cells only. A checkpoint that solves different tasks is
  averaged over different tasks, so two such averages compare with care.
- **Attention memory**: attention compares every token of a row with every
  earlier one. Done in one piece, its backward pass keeps an L x L table of
  scores for every head, so that table alone grows 4 times when the row
  doubles.
- **Chunked attention** computes attention 512 query rows at a time, and
  recomputes each block in the backward pass instead of keeping it, so the
  memory grows with L, not L x L.
- **Gradient checkpointing** does the same for whole layers: it keeps only
  each layer's input, and recomputes the layer's inner values when the
  backward pass needs them.
- The **LM head** turns each position's hidden state into a score for every
  one of the 128,000 tokens in the vocabulary. A **chunked LM head**
  computes those scores only at the trained positions, 512 at a time,
  because only there does training read them.
- The **MLX memory controls**: `mx.set_memory_limit` is the most memory MLX
  plans to use (a guideline it schedules around, not a wall it cannot
  cross), `mx.set_cache_limit` caps the freed buffers it keeps for reuse,
  and `mx.get_peak_memory` reports the most it has held.
- The **training row length cap** is the longest row the trainer takes:
  6,144 tokens in the project. A longer row is skipped, and counted.
- **AdamW's bias correction**: Adam keeps running averages of each weight's
  gradient and of its square, both starting at 0, so for the first steps
  they are too small. Bias correction divides them by how far they have
  had time to grow. Without it, the first steps are several times larger
  than the learning rate says.
- A **dry run** is a small, fast run of the whole pipeline made to find
  mistakes before the real one; **calibration** is setting the numbers (the
  learning rate, the length cap) from what the dry run showed.
- The **SFT results** are the scores after supervised fine-tuning, read
  cell by cell against the base model, as chapter 24 taught.

## Run it

```bash
cd multi-harness-RL-course/25-sft-in-harnesses
uv run teacher.py                 # one demonstration, in both training harnesses' formats
./start.sh oracle                 # the teacher (no weights) and the capture proxy
uv run collect.py --limit 60      # 60 training tasks x 2 harnesses; --fast is 10 tasks
./start.sh stop
uv run memory_probe.py            # one forward + backward pass per row length, under 9 GB
uv run train_sft.py               # SFT on the captured rows
uv run train_sft.py --no-bias-correction   # as the project's recorded run trained
./start.sh                        # the model's engine and the proxy
uv run quick_eval.py              # base against SFT on the same held-out cells
./start.sh stop
```

`start.sh` is chapter 24's, with a third process: the teacher,
`engine.py --oracle`, on `ORACLE_PORT` (8101 by default, the project's
teacher port). `collect.py` installs a missing training harness and
`quick_eval.py` a missing Pi, as chapter 24's evaluation does; the
teacher needs only LFM2.5's tokenizer (17.9 MB, fetched when missing), and
`memory_probe.py`, `train_sft.py` and the model's engine read the 4-bit
model chapter 11 built (or Liquid's export, with `--fast`).

The project runs its engine on 8100, its teacher on 8101 and its proxy on
8200. On this Mac 8100 belongs to another service and another chapter's
film used 8119 and 8200, so this run used `ENGINE_PORT=8129`,
`ORACLE_PORT=8139` and `PROXY_PORT=8209` (and `--proxy-url`,
`--oracle-url` to match).

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
22.19 or later (for Claude Code and Pi) and the network for the installs.

## The run log

Every output below is copied from the run log, made from a clean
`COURSE_HOME/25-sft-in-harnesses` on an M3 Pro with 18 GB, everything under
`nice -n 10`. Steps 00 to 15 ran at commit `a3411ac`, steps 16 to 19 at
`c71f0ed`, which changed only `memory_probe.py` (step 16's fitted curve)
and `quick_eval.py` (its `--label`). Commit `04b3aea`, after the run,
answers a code review of those changes; no output below changes with it
(the floor it puts under the fitted curve sits below every projection
step 16 made). The steps without the model did not wait. Before each step with the model, a gate
waited until nobody had touched the keyboard or mouse for 10 minutes,
nothing had rendered for 120 s, memory pressure was normal and at least
40% of memory was free; during each, memory was read every 5 s, ready to
stop the step under 15% free (it never had to). The gate waited 106, 106,
1,555, 120, 740 and 242 s before the six steps that
load the model (`waits.txt` beside the run log).

"Memory" in the table is the largest single process `/usr/bin/time` saw.
It does not count the Metal buffers MLX allocates for the model: MLX's own
peak (`mx.get_peak_memory`) is printed by each script and by the engine
when it stops, and those are the figures quoted in the sections below.

| step | command | what it shows | exit · seconds · memory |
| --- | --- | --- | --- |
| 00 | `rm -rf ~/.cache/mhrl-course/25-sft-in-harnesses` | a clean start | exit 0 · 5.3 s · 0.00 GB |
| 01 | `uv run teacher.py` | one demonstration, in both training harnesses' formats | exit 0 · 0.0 s · 0.02 GB |
| 02 | `./start.sh oracle` | the proxy installed, the teacher (no weights) and the proxy | exit 0 · 6.2 s · 0.16 GB |
| 03 | `uv run collect.py --limit 60` | 120 demonstrations, 4 rollouts at a time | exit 0 · 83.1 s · 0.28 GB |
| 04 | `uv run collect.py --fast` | refused: the file holds 120 | exit 1 · 0.1 s · 0.06 GB |
| 05 | `uv run collect.py --fast --resume` | fast mode: its 10 tasks are already recorded | exit 0 · 0.5 s · 0.30 GB |
| 06 | `./start.sh stop` | the teacher's 516 requests | exit 0 · 2.1 s · 0.00 GB |
| 07 | `python3 compare_rows.py` | a driver check: our rows against the project's | exit 0 · 0.2 s · 0.06 GB |
| 08 | `uv run pytest tests/test_25_sft_in_harnesses.py tests/test_25_mlx.py` | the chapter's tests, the real proxy included | exit 0 · 9.4 s · 0.42 GB |
| 10 | `uv run memory_probe.py` | the first probe run: fused attention at 1,024 only | exit 0 · 52.9 s · 1.81 GB |
| 11 | `uv run train_sft.py` | SFT, bias correction on (the project's code) | exit 0 · 584.4 s · 2.15 GB |
| 12 | `uv run train_sft.py --no-bias-correction` | SFT as the project's recorded run trained | exit 0 · 566.6 s · 1.51 GB |
| 13 | `./start.sh` | the model's engine (1.6 GB loaded) and the proxy | exit 0 · 8.6 s · 0.11 GB |
| 14 | `uv run quick_eval.py` | base against the first adapter, 48 cells | exit 0 · 676.5 s · 0.31 GB |
| 15 | `./start.sh stop` | the engine's peak memory | exit 0 · 2.3 s · 0.00 GB |
| 16 | `uv run memory_probe.py` | the probe with a fitted fused curve and a 1,280 row | exit 0 · 65.0 s · 1.71 GB |
| 17 | `./start.sh` | the engine again | exit 0 · 6.7 s · 0.11 GB |
| 18 | `uv run quick_eval.py --adapter .../sft-no-bias-correction/adapter.safetensors --label sft-no-bias-correction --resume` | the second adapter on the same cells | exit 0 · 174.4 s · 0.25 GB |
| 19 | `./start.sh stop` | the engine's peak memory | exit 0 · 2.1 s · 0.00 GB |

This is the second run. The first (kept beside the run log) left the
demonstrations in the order the rollouts ended, and read free memory with
`vm_stat`, which counted 3.9 GB free while macOS reported 58%: its probe
refused every row. Its training read the same 120 rows in that other order
and started from another batch (loss/token 2.0873
at step 1, not 2.55). Both are fixed (see "Plan versus measured").

## The teacher: `teacher.py`

```text
teacher.py · the answer-key teacher, on library-easy-001
  question  How many Thriller books does the library have?
  gold SQL  SELECT COUNT(*) FROM books WHERE genre = 'Thriller'
  answer    30
  path      tables (4 commands)
  paths over the 180 training tasks: direct 43, peek 45, tables 47, verify 45

claude-code: the replies the engine turns into tokens
  [1] First, which tables are in this database?</think><|tool_call_start|>[Bash(command='sqlite3 data.db .tables')]<|tool_call_end|>
  [2] These are the tables I need. Let me see their columns.</think><|tool_call_start|>[Bash(command='sqlite3 data.db ".schema books"')]<|tool_call_end|>
  [3] Now I can answer the question with one query.</think><|tool_call_start|>[Bash(command='sqlite3 data.db "SELECT COUNT(*) FROM books WHERE genre = \'Thriller\'"')]<|tool_call_end|>
  [4] The result is 30. I will save it to answer.txt before finishing.</think><|tool_call_start|>[Bash(command='echo "30" > answer.txt')]<|tool_call_end|>
  [5] The answer is saved.</think>I saved the answer to answer.txt: 30

mini-swe-agent: the replies the engine turns into tokens
  [1] First, which tables are in this database?</think><|tool_call_start|>[bash(command='sqlite3 data.db .tables')]<|tool_call_end|>
  [2] These are the tables I need. Let me see their columns.</think><|tool_call_start|>[bash(command='sqlite3 data.db ".schema books"')]<|tool_call_end|>
  [3] Now I can answer the question with one query.</think><|tool_call_start|>[bash(command='sqlite3 data.db "SELECT COUNT(*) FROM books WHERE genre = \'Thriller\'"')]<|tool_call_end|>
  [4] The result is 30. I will save it to answer.txt before finishing.</think><|tool_call_start|>[bash(command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo "30" > answer.txt')]<|tool_call_end|>
```

Each reply is the text the model would write after the chat template's
`<think>`: a short reason, `</think>`, then one tool call in LFM2.5's own
syntax. Claude Code calls its tool `Bash` and mini-swe-agent `bash`; the
teacher reads the name from the tools each request offers. Claude Code ends
when a reply calls no tool, so its demonstration has one more reply, the
answer in words. mini-swe-agent ends when its finish marker is the first
line a command prints, so the teacher chains the marker in front of the
last command, which still saves the answer.

The path depends on the task, four ways: direct (tables, query, save), peek
(tables, three rows, query, save), tables (list the tables, their columns,
query, save) and verify (tables, query, save, read the file back). The
project's first teacher always took one path of four replies, and the
student learned that the fourth reply ends the task, wherever the answer
was. Varying the path teaches the rule instead: finish after the answer is
saved.

## Collecting through the harnesses: `collect.py`

`./start.sh oracle` starts the teacher, `engine.py --oracle`: chapter 18's
engine with the teacher in place of the model. It loads LFM2.5's tokenizer
and chat template and no weights, renders every prompt exactly as the
model's engine does, and cuts the teacher's text into the student's tokens,
each with log-probability 0 (a demonstration is certain). The proxy in
front of it measured what it returns:

```text
starting the answer-key teacher (engine.py --oracle) on http://127.0.0.1:8139
  engine ready · answer-key teacher · 127.0.0.1:8139 · no weights
    LFM2.5's tokenizer and chat template (~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours), no MLX
    180 training questions; every reply's log-probabilities are 0
    one worker thread; each request prints a line here
starting the capture proxy on http://127.0.0.1:8209, in front of http://127.0.0.1:8139
  capture level: tokens (train rollouts)
```

`collect.py` opens a training session for each (task, harness) pair and
names the teacher as that session's engine, so the teacher and the model
could share one proxy. Four rollouts run at a time:

```text
collect.py · answer-key demonstrations through the training harnesses
  tasks     60 training tasks, round-robin over database x difficulty · harnesses mini-swe-agent, claude-code
  sessions  purpose train, at most 8 calls, engine http://127.0.0.1:8139 (the teacher) · 4 rollouts at a time
  records   ~/.cache/mhrl-course/25-sft-in-harnesses/demos/demos.jsonl (one line per rollout, as it ends)
  teacher   answer-key teacher at http://127.0.0.1:8139
...
demonstrations
[  2/120] claude-code    library-easy-001   usable      · 5 calls · path tables · 1 row, 2,541 tokens, 148 trained ·   0.6 s · no tool call
[  4/120] claude-code    library-hard-001   usable      · 5 calls · path verify · 1 row, 2,662 tokens, 192 trained ·   0.6 s · no tool call
[  6/120] claude-code    library-medium-001 usable      · 5 calls · path peek   · 1 row, 2,690 tokens, 165 trained ·   0.5 s · no tool call
[  5/120] mini-swe-agent library-medium-001 usable      · 4 calls · path peek   · 1 row, 943 tokens, 163 trained ·   6.2 s · finish command
...

[demos] 120/120 demonstrations usable
  mini-swe-agent 60/60 usable · 3 calls x15, 4 calls x45 · 60 rows, 882 tokens on average, 19.1% trained
  claude-code    60/60 usable · 4 calls x15, 5 calls x45 · 60 rows, 2,638 tokens on average, 6.5% trained
  this run took 1 min 23 s
network   every rollout: lsof saw only this machine's addresses
```

The lines arrive in the order the rollouts end: the Claude Code rollouts
take 0.4 to 0.6 s, mini-swe-agent's 2.5 to 6.9 s. All 120 demonstrations
are correct and trainable. Then the file is put in job order (each task,
then each harness), so the trainer's batches never depend on which
rollout ended first. One demonstration, segment by segment:

```text
one captured claude-code demonstration: library-easy-001 (path tables), 5 calls, 1 row
  row 1: 2,541 tokens · mask 1 (the teacher's tokens, trained) 148 · mask 0 (the harness's text, read only) 2,393 · 5.8% trained
  mask 0 · 2,310 tokens · <|startoftext|><|im_start|>system\nYou are a Claude agent, built on Anthrop … r. The answer is a single number.<|im_end|>\n<|im_start|>assistant\n<think>
  mask 1 ·    26 tokens · First, which tables are in this database?</think><|tool_call_start|>[Bash(command='sqlite3 data.db .tables')]<|tool_call_end|><|im_end|>
  mask 0 ·    15 tokens · \n<|im_start|>tool\nbooks    loans    members<|im_end|>\n<|im_start|>assistant\n<think>
  mask 1 ·    32 tokens · These are the tables I need. Let me see their columns.</think><|tool_call_start|>[Bash(command='sqlite3 data.db ".schema books"')]<|tool_call_end|><|im_end|>
  mask 0 ·    40 tokens · \n<|im_start|>tool\nCREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT  … published_year INTEGER NOT NULL);<|im_end|>\n<|im_start|>assistant\n<think>
  mask 1 ·    40 tokens · Now I can answer the question with one query.</think><|tool_call_start|>[Bash(command='sqlite3 data.db "SELECT COUNT(*) FROM books WHERE genre = \'Thriller\'"')]<|tool_call_end|><|im_end|>
  mask 0 ·    11 tokens · \n<|im_start|>tool\n30<|im_end|>\n<|im_start|>assistant\n<think>
  mask 1 ·    33 tokens · The result is 30. I will save it to answer.txt before finishing.</think><|tool_call_start|>[Bash(command='echo "30" > answer.txt')]<|tool_call_end|><|im_end|>
  mask 0 ·    17 tokens · \n<|im_start|>tool\n(Bash completed with no output)<|im_end|>\n<|im_start|>assistant\n<think>
  mask 1 ·    17 tokens · The answer is saved.</think>I saved the answer to answer.txt: 30<|im_end|>
```

The mask is 1 exactly on the teacher's tokens: each reply, from its reason
to its end-of-turn token `<|im_end|>`. Everything else is 0: Claude Code's
2,310-token system prompt with the tool list and the task, each tool
result, and the generation prompt `<|im_start|>assistant\n<think>` that
opens every reply. Only 5.8% trained of this row is
trained. The tests check the same on two rows captured through
mini-swe-agent: decoded, the mask-1 segments are the teacher's replies,
word for word, each followed by `<|im_end|>`.

A second run is refused, and `--resume` runs only what is missing. `--fast`
picks the first 10 of the same 60 tasks, so here nothing was left to run:

```text
~/.cache/mhrl-course/25-sft-in-harnesses/demos/demos.jsonl already holds 120 demonstrations: --resume to finish that run, or move the file away to start again
```

```text
demonstrations (120 already recorded)

[demos] 20/20 demonstrations usable
  mini-swe-agent 10/10 usable · 3 calls x2, 4 calls x8 · 10 rows, 864 tokens on average, 18.6% trained
  claude-code    10/10 usable · 4 calls x2, 5 calls x8 · 10 rows, 2,616 tokens on average, 6.2% trained
```

### The same rows as the project's

These are the 120 (harness, task) pairs the project collected (its
`runs/main/sft/demos.jsonl`), and the run log compared them row by row:

```text
captured rows against the project's (runs/main/sft/demos.jsonl)
  the same 120 (harness, task) pairs: True
  in the same order: True
  mini-swe-agent rows identical token for token: 60/60 · trained tokens identical: 60/60 · same calls: 60/60 · ours longer by 0 to 0 tokens
  claude-code    rows identical token for token: 0/60 · trained tokens identical: 60/60 · same calls: 60/60 · ours longer by 20 to 28 tokens
```

Every mini-swe-agent row is the project's, token for token. Every Claude
Code row trains the same tokens and makes the same calls; its prompt is
20 to 28 tokens longer. Decoded, the difference is three lines of Claude
Code's own system prompt: its working folder (named twice, once inside its
memory folder's name), whether that folder is a git repository (the
project's work folder sat inside its own git repository) and the date.

## Fitting it in 9 GB: `memory_probe.py`

The probe loads the 4-bit model with chapter 13's adapter, sets the
project's limits, and runs one forward + backward pass on rows of 1,024 to
8,192 tokens, the last eighth of each row trained (between the shares the
demonstrations train: 6.5% of a Claude Code row, 19.1% of a mini-swe-agent
row). Before each pass it projects the peak and refuses, with the reason,
any row over the 9.0 GB budget or over what the machine has free, less 15%
of its memory kept for everything else.

```text
memory_probe.py · one forward + backward pass per row, under a budget
  model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours (ours, built by chapter 11's convert.py), chapter 13's adapter
  limits    mx.set_memory_limit(9 GiB = 9.66 GB) · mx.set_cache_limit(512 MiB): guidelines to MLX, not walls
  budget    9.0 GB: a row projected over it is refused; fused attention runs only up to 4,096 tokens
  machine   19.3 GB; a pass must leave 15% (2.9 GB) of it free for everything else
  rows      random token ids, the last 1/8 of each trained (one forward + backward pass, the SFT loss)
  weights   1.62 GB in MLX after loading · machine free now 10.6 GB
fused: attention MLX's fused kernel, as shipped; logits trained positions, 512 at a time
[fused        1,024] projected 4.4 GB (before any pass: a generous count of the arrays it makes) · peak 5.08 GB · 2.3 s
[fused        1,280] projected 7.0 GB (from the 1,024-token pass, x1.56 above the weights) · peak 5.74 GB · 2.7 s
[fused        2,048] refused · it would need 7.0 GB more, and the machine has 8.3 GB free now, 2.9 GB of it kept for the rest
[fused        4,096] refused · projected 22.77 GB, over the 9.00 GB budget; it would need 21.1 GB more, and the machine has 8.3 GB free now, 2.9 GB of it kept for the rest
[fused        6,144] refused · fused attention runs only up to 4,096 tokens; projected 46.36 GB, over the 9.00 GB budget; it would need 44.7 GB more, and the machine has 8.3 GB free now, 2.9 GB of it kept for the rest · the project measured 11.9 GB here
[fused        8,192] refused · fused attention runs only up to 4,096 tokens; projected 79.38 GB, over the 9.00 GB budget; it would need 77.8 GB more, and the machine has 8.3 GB free now, 2.9 GB of it kept for the rest
chunked: attention 512-row blocks + every layer checkpointed; logits trained positions, 512 at a time
[chunked      1,024] projected 4.1 GB (before any pass: a generous count of the arrays it makes) · peak 3.00 GB · 2.6 s
[chunked      1,280] projected 3.3 GB (from the 1,024-token pass, x1.25 above the weights) · peak 3.12 GB · 3.3 s
[chunked      2,048] projected 4.0 GB (from the 1,280-token pass, x1.60 above the weights) · peak 3.53 GB · 5.4 s
[chunked      4,096] projected 5.4 GB (from the 2,048-token pass, x2.00 above the weights) · peak 5.12 GB · 11.4 s
[chunked      6,144] projected 6.9 GB (from the 4,096-token pass, x1.50 above the weights) · peak 7.35 GB · 18.6 s
[chunked      8,192] refused · projected 9.26 GB, over the 9.00 GB budget; it would need 7.6 GB more, and the machine has 9.5 GB free now, 2.9 GB of it kept for the rest · the project measured 9.7 GB here
full-logits: attention 512-row blocks + every layer checkpointed; logits the whole vocabulary at every position
[full-logits  1,024] projected 5.5 GB (before any pass: a generous count of the arrays it makes) · peak 4.68 GB · 3.5 s
[full-logits  1,280] projected 5.4 GB (from the 1,024-token pass, x1.25 above the weights) · peak 5.39 GB · 4.5 s
[full-logits  2,048] projected 7.7 GB (from the 1,280-token pass, x1.60 above the weights) · peak 7.53 GB · 7.3 s
[full-logits  4,096] refused · projected 13.44 GB, over the 9.00 GB budget; it would need 11.8 GB more, and the machine has 10.1 GB free now, 2.9 GB of it kept for the rest
[full-logits  6,144] refused · projected 19.34 GB, over the 9.00 GB budget; it would need 17.7 GB more, and the machine has 10.6 GB free now, 2.9 GB of it kept for the rest
[full-logits  8,192] refused · projected 25.25 GB, over the 9.00 GB budget; it would need 23.6 GB more, and the machine has 10.6 GB free now, 2.9 GB of it kept for the rest

peak memory per pass (GB, MLX's own count, the weights included)
  setting         1,024    1,280    2,048    4,096    6,144    8,192
  fused         5.08 GB  5.74 GB  refused  refused    11.9*  refused
  chunked       3.00 GB  3.12 GB  3.53 GB  5.12 GB  7.35 GB     9.7*
  full-logits   4.68 GB  5.39 GB  7.53 GB  refused  refused  refused

the project's own measurements (its source, not this run)
  fused attention, 6,144 tokens: 11.9 GB (mhrl/training/common.py, chunked_attention's docstring)
  chunked attention: 7.2 GB at 6,144, 9.7 GB at 8,192 (mhrl/cli.py, the comment on MAX_TRAIN_TOKENS = 6144)
  its SFT run on the 120 demonstrations: peak 3.95 GB (runs/main/sft/metrics.jsonl)
  * in the table: the project's figure, shown where this probe refuses
results   ~/.cache/mhrl-course/25-sft-in-harnesses/memory_probe/results.json
```

- **Chunked attention with checkpointing**, the project's training setting,
  grows by about 0.85 GB per 1,000 tokens: 3.00 GB
  at 1,024, 5.12 GB at 4,096,
  7.35 GB at 6,144 (the project's figure there:
  7.2 GB). At 8,192 the probe projects 9.26 GB, over the budget, and
  refuses; the project's figure is 9.7 GB, and its comment on
  `MAX_TRAIN_TOKENS` gives the same reason for its 6,144-token cap on rows.
- **MLX's fused attention, as shipped** (and no layer checkpointing) already
  needs 5.08 GB at 1,024 tokens and
  5.74 GB at 1,280: 0.66 GB more for 256 more
  tokens, where chunked attention needed 0.12 GB. With two fused passes the
  probe fits a curve through them (`a + c x L^2`) and projects 2,048 tokens
  at 8.62 GB (its `results.json`), under the
  budget, but 7.0 GB more than it held, and the machine
  had 8.3 GB free, 2.9 GB of it kept: refused. 4,096 projects at 22.8 GB.
  At 6,144 the probe shows the project's 11.9 GB (marked `*`) and never
  measures it.
- **Logits over the whole vocabulary at every position** cost
  7.53 GB at 2,048 tokens, against 3.53 GB with
  logits only at the trained positions: 4 GB of scores that training never
  reads. At 4,096 the probe projects 13.44 GB and refuses.

The first probe run (step 10) measured every chunked and full-logits row it
shares with step 16 at the same peak, to the hundredth of a GB, but fused
attention only at 1,024 tokens (peak 4.90 GB,
0.18 GB under step 16's): with a single pass the probe can
only scale everything above the weights by L x L, which projected 2,048
tokens at 14.73 GB. The 1,280-token row and the
fitted curve came from that run.

## Training on the captured rows: `train_sft.py`

Chapter 14's trainer, reading the captured rows: 120 rows, 15 steps of 8,
learning rate 5e-5, rows over 6,144 tokens skipped (none here), and the
memory settings the probe measured.

```text
train_sft.py · SFT on demonstrations captured through the harnesses
  model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours (ours, built by chapter 11's convert.py)
  demos     ~/.cache/mhrl-course/25-sft-in-harnesses/demos/demos.jsonl: 120 demonstrations, 120 usable (correct and trainable)
  rows      120 kept · 0 over 6,144 tokens skipped · 0 with nothing to train skipped
    claude-code    60 rows · 2,638 tokens on average · 171 trained on average
    mini-swe-agent 60 rows · 882 tokens on average · 168 trained on average
  training  lr 5e-05 · batch 8 · 1 epoch = 15 steps · AdamW, bias correction on · LoRA rank 16, scale 2, last 12 layers
  memory    limit 9 GiB, cache 0.5 GiB · attention in 512-row blocks · every layer checkpointed · logits only at trained positions, 512 at a time
[sft] 120 training rows from 120 demos
[sft] step 1/15 · loss/token 2.5461 · 1,474 tokens · 8 rows · peak 3.83 GB · 53.1 s
[sft] step 2/15 · loss/token 2.5020 · 1,343 tokens · 8 rows · peak 3.92 GB · 46.6 s
[sft] step 3/15 · loss/token 2.5193 · 1,393 tokens · 8 rows · peak 3.92 GB · 38.5 s
[sft] step 4/15 · loss/token 2.2616 · 1,243 tokens · 8 rows · peak 3.92 GB · 27.0 s
[sft] step 5/15 · loss/token 2.1316 · 1,358 tokens · 8 rows · peak 3.92 GB · 33.6 s
[sft] step 6/15 · loss/token 2.0667 · 1,282 tokens · 8 rows · peak 3.92 GB · 37.5 s
[sft] step 7/15 · loss/token 1.7132 · 1,351 tokens · 8 rows · peak 3.92 GB · 23.6 s
[sft] step 8/15 · loss/token 1.6627 · 1,257 tokens · 8 rows · peak 3.92 GB · 37.6 s
[sft] step 9/15 · loss/token 1.4853 · 1,259 tokens · 8 rows · peak 3.94 GB · 37.6 s
[sft] step 10/15 · loss/token 1.2069 · 1,526 tokens · 8 rows · peak 3.94 GB · 29.5 s
[sft] step 11/15 · loss/token 1.1930 · 1,370 tokens · 8 rows · peak 3.94 GB · 44.6 s
[sft] step 12/15 · loss/token 1.0405 · 1,433 tokens · 8 rows · peak 3.94 GB · 43.3 s
[sft] step 13/15 · loss/token 0.9592 · 1,336 tokens · 8 rows · peak 3.96 GB · 41.0 s
[sft] step 14/15 · loss/token 0.8206 · 1,457 tokens · 8 rows · peak 3.96 GB · 34.4 s
[sft] step 15/15 · loss/token 0.7999 · 1,289 tokens · 8 rows · peak 3.96 GB · 54.0 s
saved     ~/.cache/mhrl-course/25-sft-in-harnesses/sft/adapter.safetensors (39,139,440 bytes) · metrics ~/.cache/mhrl-course/25-sft-in-harnesses/sft/metrics.jsonl · note adapter.json
```

The loss falls from 2.55 to 0.80, with MLX's
peak 3.96 GB. The project's recorded run fell
from 2.55 to 0.11 on the same rows. Its batches are ours: every step
trains the same number of tokens. So the difference is in the training
itself, and the project's README names it: its SFT run trained before a
code review that turned on AdamW's bias correction. Trained that way, with
`--no-bias-correction`:

```text
  training  lr 5e-05 · batch 8 · 1 epoch = 15 steps · AdamW, bias correction off · LoRA rank 16, scale 2, last 12 layers
  memory    limit 9 GiB, cache 0.5 GiB · attention in 512-row blocks · every layer checkpointed · logits only at trained positions, 512 at a time
[sft] 120 training rows from 120 demos
[sft] step 1/15 · loss/token 2.5461 · 1,474 tokens · 8 rows · peak 3.83 GB · 50.1 s
[sft] step 2/15 · loss/token 2.2776 · 1,343 tokens · 8 rows · peak 3.92 GB · 44.8 s
[sft] step 3/15 · loss/token 1.7704 · 1,393 tokens · 8 rows · peak 3.92 GB · 37.1 s
[sft] step 4/15 · loss/token 1.1510 · 1,243 tokens · 8 rows · peak 3.92 GB · 27.0 s
[sft] step 5/15 · loss/token 0.6657 · 1,358 tokens · 8 rows · peak 3.92 GB · 33.3 s
[sft] step 6/15 · loss/token 0.4462 · 1,282 tokens · 8 rows · peak 3.92 GB · 39.0 s
[sft] step 7/15 · loss/token 0.3764 · 1,351 tokens · 8 rows · peak 3.92 GB · 23.2 s
[sft] step 8/15 · loss/token 0.3020 · 1,257 tokens · 8 rows · peak 3.92 GB · 36.9 s
[sft] step 9/15 · loss/token 0.3174 · 1,259 tokens · 8 rows · peak 3.94 GB · 37.1 s
[sft] step 10/15 · loss/token 0.2705 · 1,526 tokens · 8 rows · peak 3.94 GB · 29.9 s
[sft] step 11/15 · loss/token 0.2170 · 1,370 tokens · 8 rows · peak 3.94 GB · 42.5 s
[sft] step 12/15 · loss/token 0.1712 · 1,433 tokens · 8 rows · peak 3.94 GB · 40.7 s
[sft] step 13/15 · loss/token 0.1382 · 1,336 tokens · 8 rows · peak 3.96 GB · 37.4 s
[sft] step 14/15 · loss/token 0.1052 · 1,457 tokens · 8 rows · peak 3.96 GB · 33.4 s
[sft] step 15/15 · loss/token 0.1090 · 1,289 tokens · 8 rows · peak 3.96 GB · 51.7 s
```

| step | trained tokens | loss, bias correction on | loss, bias correction off | the project's recorded loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,474 | 2.5461 | 2.5461 | 2.5512 |
| 2 | 1,343 | 2.5020 | 2.2776 | 2.2819 |
| 3 | 1,393 | 2.5193 | 1.7704 | 1.7714 |
| 4 | 1,243 | 2.2616 | 1.1510 | 1.1501 |
| 5 | 1,358 | 2.1316 | 0.6657 | 0.6643 |
| 6 | 1,282 | 2.0667 | 0.4462 | 0.4459 |
| 7 | 1,351 | 1.7132 | 0.3764 | 0.3744 |
| 8 | 1,257 | 1.6627 | 0.3020 | 0.2997 |
| 9 | 1,259 | 1.4853 | 0.3174 | 0.3142 |
| 10 | 1,526 | 1.2069 | 0.2705 | 0.2660 |
| 11 | 1,370 | 1.1930 | 0.2170 | 0.2117 |
| 12 | 1,433 | 1.0405 | 0.1712 | 0.1665 |
| 13 | 1,336 | 0.9592 | 0.1382 | 0.1341 |
| 14 | 1,457 | 0.8206 | 0.1052 | 0.1038 |
| 15 | 1,289 | 0.7999 | 0.1090 | 0.1098 |

Without bias correction the loss follows the project's recorded run at every
step, within 0.006: the same rows, the same batches, the same setting. The
only input that differs is Claude Code's prompt (see "The same rows as the
project's"), and at step 1, before any update, the losses already differ by
that much: 2.5461 against the
project's 2.5512. With bias correction, the project's code today, each
early step is smaller: an uncorrected first step moves every weight 3.16
times as far as a corrected one ((1 - 0.9) / sqrt(1 - 0.999) = 3.16; the
tests measure it on the tiny model), and the loss after 15 steps is 0.80,
not 0.11. Both runs reached MLX's
peak 3.96 GB, and took
584.4 s and 566.6 s.

## A small held-out check: `quick_eval.py`

`./start.sh` starts the model's engine. `quick_eval.py` runs chapter 24's
evaluation on 8 held-out tasks per harness (24 cells), once with the base
model and once with an adapter, on the same cells. Between the two the
engine switches adapters itself (`POST /admin/adapter`), and afterwards
puts back what it started with. The adapter `train_sft.py` saved, with
bias correction on:

```text
━━ both checkpoints, on the same cells
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             25%             12%             46%
sft                     100%             88%              0%             62%
base  calls per solve (solved cells only): mini-swe-agent 8/8, 7.6 · claude-code 2/8, 8.0 · pi 1/8, 6.0
sft   calls per solve (solved cells only): mini-swe-agent 8/8, 7.5 · claude-code 7/8, 6.7 · pi 0/8, none solved
calls per solve, averaged over harnesses: base 7.2 -> sft 7.1 (solved cells only, so compare with care)
  mini-swe-agent  8 cells solved by both: 7.6 -> 7.5 calls
  claude-code     2 cells solved by both: 8.0 -> 7.0 calls
  pi              no cell solved by both

━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair         harness         cells  before  after  gained  lost  flipped  p      reading
  base -> sft  mini-swe-agent      8       8      8       0     0        0  1.000  nothing changed
  base -> sft  claude-code         8       2      7       5     0        5  0.062  within noise
  base -> sft  pi                  8       1      0       0     1        1  1.000  within noise
  p < 0.05 reads as beyond noise; anything else, within it
```

Then the adapter trained as the project's recorded run was, on the same
cells (`--adapter .../sft-no-bias-correction/adapter.safetensors --label
sft-no-bias-correction --resume`, so the base model's 24 cells are read
back, not run again):

```text
━━ both checkpoints, on the same cells
checkpoint                mini-swe-agent     claude-code              pi         average
base                                100%             25%             12%             46%
sft-no-bias-correction               88%             75%             75%             79%
base  calls per solve (solved cells only): mini-swe-agent 8/8, 7.6 · claude-code 2/8, 8.0 · pi 1/8, 6.0
sft-no-bias-correction calls per solve (solved cells only): mini-swe-agent 7/8, 4.7 · claude-code 6/8, 4.3 · pi 6/8, 4.5
calls per solve, averaged over harnesses: base 7.2 -> sft-no-bias-correction 4.5 (solved cells only, so compare with care)
  mini-swe-agent  7 cells solved by both: 7.6 -> 4.7 calls
  claude-code     2 cells solved by both: 8.0 -> 4.5 calls
  pi              1 cells solved by both: 6.0 -> 4.0 calls

━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair                            harness         cells  before  after  gained  lost  flipped  p      reading
  base -> sft-no-bias-correction  mini-swe-agent      8       8      7       0     1        1  1.000  within noise
  base -> sft-no-bias-correction  claude-code         8       2      6       4     0        4  0.125  within noise
  base -> sft-no-bias-correction  pi                  8       1      6       5     0        5  0.062  within noise
  p < 0.05 reads as beyond noise; anything else, within it
```

And the project's own pair and row, from its bundled records (24 tasks per
harness, other cells, never mixed with ours):

```text
  pair                         harness         cells  before  after  gained  lost  flipped  p      reading
  project base -> project sft  mini-swe-agent     24      24     18       0     6        6  0.031  lost: beyond noise
  project base -> project sft  claude-code        24      15     17       7     5       12  0.774  within noise
  project base -> project sft  pi                 24       4      9       7     2        9  0.180  within noise
  p < 0.05 reads as beyond noise; anything else, within it
  churn: claude-code project base -> project sft: 12 of 24 cells flipped, 10 of them cancel out: score +2
  (the project's pair is the project's run of 2026-10-06, from its bundled records)

┌─ the project's recorded SFT row ───────────────────────────────────────────┐
│ the project's run of 2026-10-06: 24 held-out tasks per harness, other cells │
│ sft pass@1 (mini-swe-agent / claude-code / pi): 75% / 71% / 38%            │
│ calls per solve, averaged over harnesses: base 7.1 -> sft 4.8              │
│ (solved cells only, so compare with care)                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

Read cell by cell, as chapter 24 taught:

- **Trained as the project's recorded run was**, SFT made solves cheaper:
  base 7.2 -> sft-no-bias-correction 4.5
  calls per solve, averaged over the harnesses, and 7.6 to 4.7 on the 7
  mini-swe-agent cells both checkpoints solved. It cost mini-swe-agent one
  task (8 to 7, p = 1.000). Claude Code gained 4 cells and Pi, never trained
  in, gained 5, with none lost (p = 0.125 and 0.062): suggestive, not shown.
- **The project's recorded row** reads the same way at 24 tasks per harness:
  SFT made solves cheaper (7.1 to 4.8 calls) and cost mini-swe-agent 6
  tasks, beyond noise (p = 0.031); the Claude Code and Pi gains are
  suggestive, not shown, and 12 of Claude Code's 24 cells flipped for a
  score 2 higher.
- **With bias correction on**, 15 steps left the model's habits nearly as
  they were: base 7.2 -> sft 7.1 calls per solve.
  Claude Code still went from 2 to 7 of 8 (5 gained, none lost, p = 0.062),
  and Pi from 1 to 0.
- With 8 tasks per harness, one task is 12.5 points. Every comparison of ours
  is within noise, and the base model was measured once, at temperature 0.

The engine's peak memory was 3.38 GB and
3.35 GB, the weights included (MLX's count, printed
when it stopped); the two evaluations took 11 min 16 s (48
cells and a Pi install) and 2 min 54 s
(24 cells).

## Plan versus measured

| the plan said | measured |
| --- | --- |
| folder `course/25-sft-in-harnesses/` | `25-sft-in-harnesses/` at the root of this repository, like every chapter |
| teacher.py is the answer-key teacher: it finds the task in the conversation and writes the next reply in LFM's own format ('reasoning</think>' then a native tool call, mini-swe-agent's finish marker chained first on the last step), varying its path per task | as planned. Over the 180 training tasks the paths are direct 43, peek 45, tables 47 and verify 45; over the 60 demonstrated, 15, 14, 15 and 16 (`demos-without-rows.jsonl` in the run log) |
| chapter 18's engine serves it with --oracle on :8101, tokenizing its text with the student's chat template (logprobs 0) | as planned, with no weights and no MLX: the proxy measured capture level "tokens". 8101 is the default; this run used 8139 because the ports moved (below) |
| collect.py runs it in both training harnesses through chapter 20's proxy, 4 rollouts at a time sharing one engine thread, each with its own folder and session | as planned: 120 rollouts in 83.1 s, 516 requests to the teacher. Added: the file is put in job order at the end. The first run kept the order rollouts ended in, and its training batches differed from the project's |
| `uv run collect.py --limit 60` prints '[demos] 120/120 demonstrations usable' and one captured Claude Code demonstration with its mask (teacher tokens 1, harness text 0) | as planned. The 120 rows are the project's: every mini-swe-agent row token for token, every Claude Code row with the same trained tokens and a prompt 20 to 28 tokens longer (its folder, git status and date) |
| `uv run train_sft.py` prints '[sft] step k · loss/token' from 2.55 at step 1 to 0.11 at step 15, peak about 4 GB | with the project's code (bias correction on): 2.55 to 0.80, peak 3.96 GB. 2.55 to 0.11 is the project's recorded run, which trained before its code review turned bias correction on (its README says so); `--no-bias-correction` reproduces it within 0.006 at every step, peak 3.96 GB |
| quick_eval.py runs a small held-out check and shows fewer calls per solve, with the project's recorded SFT row beside it: 75 / 71 / 38% (mini-swe-agent / Claude Code / Pi) and calls per solve 7.1 -> 4.8 averaged over the harnesses (counted on solved tasks only, so compare with care) | 8 tasks per harness, base and adapter on the same cells. The adapter trained as the project's recorded run: calls per solve 7.2 to 4.5; mini-swe-agent 8 to 7 of 8, Claude Code 2 to 6, Pi 1 to 6 (p = 1.000, 0.125, 0.062). With bias correction on: 7.2 to 7.1 calls; 8, 7 and 0 of 8. The project's row is as its records say (`data/project_records/eval_sft.jsonl` and `eval_sft-pi-fixed.jsonl`): 75 / 71 / 38%, 7.1 to 4.8 |
| (review) word the summary from gained / lost / p, e.g. 'SFT made solves cheaper and cost mini-swe-agent 6 tasks; the Claude Code and Pi gains are suggestive, not shown' | worded from quick_eval.py's pair tables: for the project's row exactly that; for ours, solves cheaper, one mini-swe-agent task lost, the Claude Code and Pi gains suggestive, not shown |
| the dry-run lessons: demonstrations that were always four steps long took Claude Code from 100% to 17% | that is 6 of 6 held-out tasks before and 1 of 6 after (the project's `runs/dry/summary_base.json` and `summary_sft.json`). Between the first and the second dry run two things changed, the teacher's paths and the learning rate (by `mhrl/cli.py`'s comment; neither run records it), and the second scored 5 of 6 (`runs/dry2/summary_sft.json`): which change helped cannot be told apart |
| ... and lr 1e-4 over-fitted by step 4, which is why SFT uses 5e-5 | the 1e-4 rests on one comment (`mhrl/cli.py`: "1e-4 over-fit the dry run (loss 0.10 by step 4 of 6)"); the recorded dry run (`runs/dry/sft/metrics.jsonl`, bundled as `data/project_run/dry_sft_metrics.jsonl`) records no learning rate and shows 0.17 at step 4 and 0.10 at step 5 |
| fast mode: --limit 10 | `--fast` is 10 tasks, the first 10 of the same 60, so after the full run nothing was left to run |
| memory_probe.py sets mx.set_memory_limit (9 GB) and a 512 MB cache limit | the project's limit is 9 GiB, which is 9.66 GB, and it is a guideline MLX plans around, not a wall: the project's docstring puts its own fused pass at 11.9 GB under it. So the probe never trusts it: it refuses any row projected over a 9.0 GB budget, or over the machine's free memory less 15% |
| one forward + backward pass on rows of 1K-8K tokens in three settings, reading mx.get_peak_memory after each | as planned, with a 1,280-token row added: chunked 3.00 to 7.35 GB from 1,024 to 6,144 tokens; fused 5.08 and 5.74 GB at 1,024 and 1,280; full logits 7.53 GB at 2,048. Every other row refused, with its reason |
| (review) run fused attention only up to 4K | enforced; on this machine the probe measured fused attention up to 1,280 tokens, and refused 2,048 for lack of free memory (it needed 7.0 GB more; 5.4 GB could be used) |
| (review) present 6K as the project's recorded 11.9 GB | shown as the project's figure, marked `*`, never measured. 11.9 GB comes from a docstring (`mhrl/training/common.py`, `chunked_attention`) and the chunked 7.2 and 9.7 GB from a comment (`mhrl/cli.py`, on `MAX_TRAIN_TOKENS`): no run log in the project holds them, and they do not say which LM head or checkpointing they used. This run measured chunked attention at 7.35 GB at 6,144 tokens |
| (review) make the probe refuse any setting projected over budget, and print that refusal | as planned: every refusal names its reason, and `results.json` keeps each cell as it ends |
| the project's ports: engine 8100, teacher 8101, proxy 8200 | the defaults are the project's; on this Mac 8100 belongs to another service and another chapter's film used 8119 and 8200, so this run used 8129, 8139 and 8209 |

- **The project's numbers** in this chapter come from its recorded files:
  its demonstrations (`runs/main/sft/demos.jsonl`, compared, never
  copied), its SFT metrics (`data/project_run/`), its evaluation records
  (`data/project_records/`) and the dry runs' summaries named above.

## The files

Line counts at commit `04b3aea`: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `teacher.py` | 453 | 245 | the project's answer-key teacher: four paths, the answer saved before the finish |
| `engine.py` | 1,180 | 579 | chapter 24's engine (874 lines), plus `--oracle` and adapter switching (`--adapter`, `POST /admin/adapter`) |
| `start.sh` | 351 | | chapter 24's stack, plus the teacher on `ORACLE_PORT` |
| `collect.py` | 904 | 480 | the project's `collect_demonstrations`: 4 rollouts at a time, the file put in job order, the mask shown |
| `training.py` | 445 | 194 | the project's `mhrl/training/common.py`: rows, chunked attention, checkpointing, the chunked LM head; chapter 14's loop |
| `train_sft.py` | 396 | 248 | chapter 14's trainer reading captured rows; `--no-bias-correction` |
| `memory_probe.py` | 859 | 475 | one pass per row length in three settings, every row projected first, refused with its reason |
| `quick_eval.py` | 624 | 348 | chapter 24's evaluation, base against an adapter on the same cells, paired, the project's row boxed |
| `lfm.py`, `lora.py`, `tasks.py` | 299, 188, 157 | 126, 89, 61 | chapter 24's model with adapter switching; chapter 14's LoRA with a check on the adapter file; the tasks with the training split |
| `project_run.py`, `data/` | | | chapter 14's reader of the project's SFT metrics; the 230 questions, five databases and the project's records |
| `run_one.py`, `evaluate.py`, `noise.py`, `install.py`, `wire.py` and the rest | | | chapter 24's, unchanged but for this chapter's folder name |

## What this chapter leaves out

- **Reinforcement learning.** Chapter 26 starts from this adapter and trains
  through the same harnesses with GRPO.
- **More tasks.** 8 tasks per harness can show only large changes. Chapter
  29 sizes a test for the gain you hope to see.
- **A model as the teacher.** The article's SFT imitated a 27B model's
  successful rollouts; a laptop holds an answer key instead.

## Mirrors in the project

`mhrl/engine/oracle.py` (`AnswerKeyTeacher`), `mhrl/training/sft.py`
(`collect_demonstrations`, `demo_items`, `train_sft`) and
`mhrl/training/common.py` (`chunked_attention`, `load_policy`,
`token_logprobs`).
