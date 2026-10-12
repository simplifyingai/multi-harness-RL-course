# Chapter 21. From calls to training rows

Chapter 20's proxy recorded every call a harness made, with the exact tokens
the model read and wrote. A trainer does not want calls, though: it wants
**rows**, one long line of tokens per piece of work, with a mark on every
token saying "learn this" or "only read this".

The puzzle: one task, solved correctly, the same four commands every time.
How many rows does it make, and how many tokens can a trainer learn from?
This chapter's answer, measured on the real proxy: one row, two, four, or
one row with nothing to learn at all, depending only on how the harness
behaved. And Claude Code, run the way most people run it, turns every one of
its calls into a row of its own.

## Words first

- A **turn** (a node) is one model call as the proxy records it: the exact
  prompt tokens, the exact reply tokens and each reply token's
  log-probability (chapter 12).
- The **rollout graph** links the turns. A turn's **parent** is the earlier
  turn whose prompt and reply are, token for token, the beginning of this
  turn's prompt (the longest such one). A **root** has no parent: a new
  conversation. A **leaf** has no child: a conversation's last call. A
  **fork** is a turn with two or more children.
- A **discarded turn** is a fork's child that led nowhere while a sibling
  went on: a reply the harness threw away, such as a call it sent again.
- A **training row** (a training sequence) is one path from a root to a
  leaf, laid end to end: `input_ids` (every token), a **loss mask** (1 on
  the model's own reply tokens, 0 on what it only read) and `logprobs` (the
  engine's number for each mask-1 token, 0.0 elsewhere). Rows are **agent**
  (the turns offered tools), **auxiliary** (they offered none, as a title
  request) or **discarded**.
- **Findings** are the proxy's notes on a rollout: `FATAL`, `WARN` or
  `INFO`, about one turn, one row or the whole rollout. Its **stats** count
  roots, rows (`n_sequences`) and mask-1 tokens of trainable rows
  (`n_trainable_tokens`), and its verdict is one word, `trainable`.
- A **side call** (an auxiliary call) is a request a harness makes for
  itself, outside the agent's work: a title, a summary. A **subagent** is a
  helper the harness starts with its own system prompt, so its first call
  starts a root of its own.
- **Compaction** is a harness replacing a long history with a summary to
  save room.
- **`n_calls`** and **`n_sampled`** are the project's two efficiency numbers
  (`mhrl/rollout.py`): the agent's calls (not discarded, and offering tools
  when any call did) and the tokens the model wrote. The token bonus in the
  reward is paid on `n_sampled`.
- **Shared-prefix dedup** (`dedupe_shared_prefixes`): when a rollout forks,
  its rows begin with the same tokens, and the replies in that shared start
  would be learnt once per row. The dedup sets the mask to 0 over the part a
  row shares with an earlier row.
- Claude Code's **total-tokens note** tells the model how much budget is
  left: `<total_tokens>N tokens left</total_tokens>`. The setting
  `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` turns it off (`off`) or picks its kind
  (`fixed` always shows 5,000,000).
- A **trace cross-check** compares the proxy's record with the harness's own
  log of the same run. Claude Code's log is its **stream-json** output: one
  line of JSON per event.
- The **scripted engine** is chapter 20's engine with a script in place of
  the model: no weights, replies we choose. A **broken engine** is one that
  answers correctly and still loses what training needs.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/21-calls-to-rows
ENGINE_PORT=8119 ./start.sh scripted
uv run graph_lab.py
uv run graph_lab.py dry-run --dry-run "$PROJECT/runs/dry/grpo/rollouts.jsonl"  # a checkout of the project
uv run reminder.py
uv run crosscheck.py reminder-default
uv run crosscheck.py reminder-fixed
uv run crosscheck.py reminder-off
./start.sh stop
ENGINE_PORT=8119 ./start.sh
uv run reminder.py
uv run crosscheck.py reminder-default
uv run crosscheck.py reminder-fixed
uv run crosscheck.py reminder-off
./start.sh stop
```

`./start.sh scripted` starts the scripted engine (no model, no MLX) and the
capture proxy in front of it; `./start.sh` starts the model, chapter 18's
engine with chapter 11's 4-bit LFM2.5. Both take the engine's port as
`ENGINE_PORT` (default 8100, the project's; 8119 in the run log, where
another program holds 8100), and the proxy keeps 8200. The `--dry-run` line
needs a checkout of the project (`PROJECT`, a folder; the run log's is
`~/sai_work/multi-harness-rl`); without the flag, `dry-run` only quotes the
project's figure.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
(for `npx`), chapter 11's model in `COURSE_HOME/models` for the second
half, and the network the first time: `start.sh` installs the proxy into
`COURSE_HOME/21-calls-to-rows/proxy-env` (89M), and `reminder.py` installs
Claude Code 2.1.290 with `npx` into `COURSE_HOME/21-calls-to-rows/claude-code`
(342M). The scripts need only Python's standard library.

Every output below is copied from the run log, made from a clean
`COURSE_HOME/21-calls-to-rows`, everything under `nice -n 10`, the model's
engine started only after a minute with no film rendering on the machine
(the run waited 254 s). Seconds are wall time; memory is the largest process
the command ran (`/usr/bin/time -l`).

| command | what it shows | seconds | memory |
| --- | --- | ---: | ---: |
| `./start.sh install` | OpenEnv's capture proxy at `3f14355` | 3.9 | 0.15 GB |
| `ENGINE_PORT=8119 ./start.sh scripted` | the scripted engine, its broken copy, the proxy at `tokens` | 4.4 | 0.11 GB |
| `uv run graph_lab.py` | eight cases, every one solved, as graph, rows, findings, stats | 0.3 | 0.04 GB |
| `uv run graph_lab.py dry-run --dry-run …` | the project's 11.8%, re-measured | 0.1 | 0.04 GB |
| `uv run reminder.py` (scripted) | Claude Code installed, then run three times | 15.0 | 0.22 GB |
| `uv run crosscheck.py reminder-*` (scripted) | 4/4, completion tokens match, three times | 0.1 or less | 0.03 GB |
| `./start.sh stop` | `engine stopped · 47 requests · broken engine 8 requests` | 2.1 | |
| `uv run pytest tests/test_21_calls_to_rows.py` | 50 passed, against the real proxy | 18.5 | 0.22 GB |
| `ENGINE_PORT=8119 ./start.sh` | the model's engine (1.6 GB loaded), the proxy at `tokens` | 10.9 | 0.11 GB |
| `uv run reminder.py` (the model) | three runs of Claude Code with LFM2.5: 35, 108 and 47 s | 191.9 | 0.20 GB |
| `uv run crosscheck.py reminder-*` (the model) | 8/8 three times; two empty replies cut at the cap | 0.1 or less | 0.03 GB |
| `./start.sh stop` | `engine stopped · 28 requests · peak memory 3.19 GB, the weights included` | 2.2 | |

The proxy itself held 0.12 GB (scripted) and 0.05 GB (the model) just before
each stop. The test file has 51 tests at the final commit; the run log's
test run, one commit earlier, had 50.

## The stack: `start.sh scripted`

```text
capture proxy ready: OpenEnv 3f14355631513f51c4b3211b32a069ead0faf837 in ~/.cache/mhrl-course/21-calls-to-rows/proxy-env
starting the engine (scripted_engine.py) on http://127.0.0.1:8119
  engine ready · scripted model (no MLX) · 127.0.0.1:8119
  engine ready · broken copy at 127.0.0.1:8119/broken: no log-probabilities when a request offers tools
starting the capture proxy on http://127.0.0.1:8200, in front of http://127.0.0.1:8119
  capture level: tokens (train rollouts)
```

The scripted engine is chapter 20's real `engine.py` (its request checks,
receipts, restore and prompt caches) with a scripted model behind its seam.
Each case queues its replies first, through a small remote control
(`POST /script`). One switch per reply sends it back with every
log-probability null. The same server holds a second, **broken** engine
under `/broken`: identical, except that whenever a request offers tools, its
reply comes back with no log-probabilities. Its tokenizer gives each
character one token, so a reply of 50 characters is about 50 tokens.

## Eight ways to solve one task: `graph_lab.py`

`graph_lab.py` plays the harness: chapter 17's system prompt, the task and
one `bash` tool, through the proxy with a session id as its key. Every case
runs the same four commands for real (list the tables, count the patients
in Murcia, write `30` to `answer.txt`, answer) and is graded. Only the
harness's behaviour changes.

### 1. A clean chain: one root, one row

```text
graph_lab.py · the capture proxy at http://127.0.0.1:8200, in front of the scripted engine at http://127.0.0.1:8119

━━ 1 · clean · a clean 4-call chain
harness   session se180c544d060f15a4c3e97fe · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     4 turns · 1 root (t0) · 1 leaf (t3) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    t1     1,208     53      1  bash: echo "30" > answer.txt
  t3    t2     1,282     32      1  text: I wrote 30 to answer.txt.
rows      1 for the trainer
  row 1 · agent · trainable · 1,314 tokens, 245 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     23  mask 0
    t2   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t3   context     21  mask 0
    t3   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  none
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 245 · trainable true
project   count_calls 4 · n_sampled 245 · rows kept 1
  tokens with mask 1: 245 · after dedupe_shared_prefixes 245 · item_from_sequence positions 245
```

Each call's prompt begins with the previous call's prompt and reply, so
each turn's parent is the turn before: one root, one leaf, one row. The row
reads like the conversation itself: 968 tokens the model read (the system
prompt, the tool, the task), then its 56-token reply with mask 1, then 57
tokens it read (the tool's output), and so on. Only the replies have mask
1: 245 tokens, exactly what the model wrote (`n_sampled`). The scripted
log-probabilities repeat `-0.125 -0.250 -0.375 -0.500`; the model's are
real numbers (below, under `reminder.py`: Claude Code's one-row run).

### 2 and 3. A retried call: discarded, or two rows

```text
━━ 2 · retry · call 2 sent twice: the first reply is abandoned
harness   session sc24c53c33d52b4f79dbdb738 · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db 'SELECT COUNT(*) FROM patients'   <- abandoned: sent again
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     5 turns · 1 root (t0) · 2 leaves (t1, t4) · fork t0 -> t1, t2 · discarded: t1
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081     81      1  bash: sqlite3 data.db 'SELECT COUNT(*) FROM p…  discarded
  t2    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t3    t2     1,208     53      1  bash: echo "30" > answer.txt
  t4    t3     1,282     32      1  text: I wrote 30 to answer.txt.
rows      1 for the trainer (+1 discarded, asked for with include_discarded=true)
  row 1 · discarded · not trainable · 1,162 tokens, 137 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply       81  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 2 · agent · trainable · 1,314 tokens, 245 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     57  mask 0
    t2   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t3   context     23  mask 0
    t3   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t4   context     21  mask 0
    t4   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [WARN] discarded_turns: 1 sampled turn(s) led nowhere (retries or resamples)
          [INFO] forks: 1 fork point(s) in the graph
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 245 · trainable true
project   count_calls 4 · n_sampled 245 · rows kept 1
  tokens with mask 1: 245 · after dedupe_shared_prefixes 245 · item_from_sequence positions 245
```

The harness sent call 2 twice, as a client does when its request times out.
The proxy saw both replies: two children of `t0`, a fork. `t2` went on and
`t1` did not, so `t1` is **discarded**. Its row is labelled `discarded`
and left out of the document the trainer asks for (shown here because
`graph_lab.py` also asks with `include_discarded=true`). The trainer gets
one row, and the dedup has nothing to do.

```text
━━ 3 · retry-last · the last call sent twice: both replies end the conversation
harness   session s407aca071356413294c066c8 · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: The answer is 30.   <- abandoned: sent again
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     5 turns · 1 root (t0) · 2 leaves (t3, t4) · fork t2 -> t3, t4 · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    t1     1,208     53      1  bash: echo "30" > answer.txt
  t3    t2     1,282     24      1  text: The answer is 30.
  t4    t2     1,282     32      1  text: I wrote 30 to answer.txt.
rows      2 for the trainer
  row 1 · agent · trainable · 1,306 tokens, 237 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     23  mask 0
    t2   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t3   context     21  mask 0
    t3   reply       24  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 2 · agent · trainable · 1,314 tokens, 245 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     23  mask 0
    t2   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t4   context     21  mask 0
    t4   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [INFO] forks: 1 fork point(s) in the graph
stats     n_roots 1 · n_sequences 2 · n_trainable_tokens 482 · trainable true
project   count_calls 5 · n_sampled 269 · rows kept 2
  tokens with mask 1: 482 · after dedupe_shared_prefixes 263 · item_from_sequence positions 263
  row 2 shares its first 1,288 tokens with an earlier row: 219 of its mask-1 tokens are trained there
```

Sent twice at the end, the two replies both end the conversation. Neither
went on, so the proxy cannot tell which one the harness kept: nothing is
discarded, and both rows are trainable. They share their first 1,288
tokens, and the proxy's count of mask-1 tokens, 482, counts calls 1 to 3
twice. `dedupe_shared_prefixes` brings it to 263. `count_calls` says 5, and
`n_sampled` counts the abandoned reply's 24 tokens.

### The project's 11.8%: `graph_lab.py dry-run`

```text

━━ dry-run · shared-prefix repeats in the project's own run
  the project says: '11.8% of the dry run's trainable tokens were such
  repeats, all in reward-0 rollouts' (mhrl/training/common.py, the
  docstring of dedupe_shared_prefixes; its dry run is runs/dry/grpo)
  re-measured on ~/sai_work/multi-harness-rl/runs/dry/grpo/rollouts.jsonl:
  24 rollouts, 3 with rows that share trained tokens · 6,567 mask-1 tokens, 778 of them repeats (11.8%), 778 in rollouts with reward 0
```

The project's figure, re-measured from its own dry run: 778 of 6,567 mask-1
tokens were repeats, in 3 of 24 rollouts, all with reward 0. Decoded with
LFM2.5's tokenizer (finding 3 below), the three forks are mostly not
retries: in two of them the rows part a few tokens before the end of a
reply whose text is the same in both rows. Those 541 repeat tokens come from
a reply that came back as other token ids (chapter 9's one text, many cuts);
237 come from one retry.

### 4. A line that changes on every call: four roots

```text
━━ 4 · changing-line · the system prompt's last line changes on every call
harness   session s2d542425a2055dee08429f2a · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     4 turns · 4 roots (t0, t1, t2, t3) · 4 leaves (t0, t1, t2, t3) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -      1,019     56      1  bash: sqlite3 data.db .tables
  t1    -      1,132    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    -      1,259     53      1  bash: echo "30" > answer.txt
  t3    -      1,333     32      1  text: I wrote 30 to answer.txt.
rows      4 for the trainer
  row 1 · agent · trainable · 1,075 tokens, 56 with mask 1
    t0   context  1,019  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 2 · agent · trainable · 1,236 tokens, 104 with mask 1
    t1   context  1,132  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 3 · agent · trainable · 1,312 tokens, 53 with mask 1
    t2   context  1,259  mask 0
    t2   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 4 · agent · trainable · 1,365 tokens, 32 with mask 1
    t3   context  1,333  mask 0
    t3   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [WARN] per_turn_capture_only: every turn is its own root (4)
stats     n_roots 4 · n_sequences 4 · n_trainable_tokens 245 · trainable true
project   count_calls 4 · n_sampled 245 · rows kept 4
  tokens with mask 1: 245 · after dedupe_shared_prefixes 245 · item_from_sequence positions 245
```

The system prompt's last line counts down on every call, in Claude Code's
words (`<total_tokens>N tokens left</total_tokens>`). Call 2's prompt no
longer begins with call 1's, so no turn has a parent: four roots, four rows
of one call each, a `WARN per_turn_capture_only`. The model's tokens are all
still trainable (245), but each row repeats everything before its call:
4,988 tokens of rows against the clean chain's 1,314.

### 5. Compaction: a new root

```text
━━ 5 · compaction · after call 2 the history is replaced by a summary
harness   session s28bdec78ea7fe979ac8d163d · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  compact the history is replaced: 3 messages, the last a summary
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     4 turns · 2 roots (t0, t2) · 2 leaves (t1, t3) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    -      1,200     53      1  bash: echo "30" > answer.txt
  t3    t2     1,274     32      1  text: I wrote 30 to answer.txt.
rows      2 for the trainer
  row 1 · agent · trainable · 1,185 tokens, 160 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 2 · agent · trainable · 1,306 tokens, 85 with mask 1
    t2   context  1,200  mask 0
    t2   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t3   context     21  mask 0
    t3   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [WARN] multiple_roots: 2 roots across 4 turns
stats     n_roots 2 · n_sequences 2 · n_trainable_tokens 245 · trainable true
project   count_calls 4 · n_sampled 245 · rows kept 2
  tokens with mask 1: 245 · after dedupe_shared_prefixes 245 · item_from_sequence positions 245
```

After call 2 the harness replaced the history with a summary. Call 3's
prompt does not continue call 2, so it starts a second root: two rows.

### 6. A turn without log-probabilities: context, and a finding

```text
━━ 6 · no-logprobs · the engine serves call 3 without its log-probabilities
harness   session sf16a34079f6ecdb1fdd6380c · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     4 turns · 1 root (t0) · 1 leaf (t3) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    t1     1,208     53      1  bash: echo "30" > answer.txt
  t3    t2     1,282     32      1  text: I wrote 30 to answer.txt.
rows      1 for the trainer
  row 1 · agent · trainable · 1,314 tokens, 192 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t1   context     57  mask 0
    t1   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     23  mask 0
    t2   reply       53  mask 0  no logprobs: mask 0
    t3   context     21  mask 0
    t3   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [FATAL] no_logprobs: turn 2: 53 sampled tokens with no logprobs
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 192 · trainable true
project   count_calls 4 · n_sampled 245 · rows kept 1
  tokens with mask 1: 192 · after dedupe_shared_prefixes 192 · item_from_sequence positions 192
```

Call 3 came back with every log-probability null. Its 53 tokens stay in the
row, because the later calls read them, but with mask 0 and no
log-probabilities, and the proxy writes a `FATAL no_logprobs`. The rollout
is still `trainable: true`, with 192 tokens instead of 245 (finding 4).

### 7. A side call: auxiliary, not an agent step

```text
━━ 7 · side-call · after call 1, the harness asks for a title (no tools)
harness   session s409afb0757a59498c38dd4ab · train · capture level tokens
  call 1  bash: sqlite3 data.db .tables
  side    no tools · text: Patients in Murcia
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     5 turns · 2 roots (t0, t1) · 2 leaves (t1, t4) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    -        114     28      0  text: Patients in Murcia
  t2    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t3    t2     1,208     53      1  bash: echo "30" > answer.txt
  t4    t3     1,282     32      1  text: I wrote 30 to answer.txt.
rows      2 for the trainer
  row 1 · auxiliary · not trainable · 142 tokens, 28 with mask 1
    t1   context    114  mask 0
    t1   reply       28  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
  row 2 · agent · trainable · 1,314 tokens, 245 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t2   context     57  mask 0
    t2   reply      104  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t3   context     23  mask 0
    t3   reply       53  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
    t4   context     21  mask 0
    t4   reply       32  mask 1  logprobs -0.125 -0.250 -0.375 -0.500 …
findings  [WARN] multiple_roots: 2 roots across 5 turns
stats     n_roots 2 · n_sequences 2 · n_trainable_tokens 245 · trainable true
project   count_calls 4 · n_sampled 273 · rows kept 1
  tokens with mask 1: 245 · after dedupe_shared_prefixes 245 · item_from_sequence positions 245
```

The request for a title offered no tools and starts its own root, so its row
is labelled `auxiliary` and is not trainable. `count_calls` leaves it out (4
calls), but `n_sampled` counts its 28 tokens: 273 against 245 (finding 5).

### 8. A broken engine: correct, and nothing to learn

```text
━━ 8 · broken-engine · an engine that drops log-probabilities whenever tools are offered
harness   session s97cca2a033da4efb22439041 names the engine http://127.0.0.1:8119/broken
  the proxy probed it first: capture level tokens (train)
  call 1  bash: sqlite3 data.db .tables
  call 2  bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE …
  call 3  bash: echo "30" > answer.txt
  call 4  text: I wrote 30 to answer.txt.
  grade   correct (answer.txt '30')
graph     4 turns · 1 root (t0) · 1 leaf (t3) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -        968     56      1  bash: sqlite3 data.db .tables
  t1    t0     1,081    104      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    t1     1,208     53      1  bash: echo "30" > answer.txt
  t3    t2     1,282     32      1  text: I wrote 30 to answer.txt.
rows      1 for the trainer
  row 1 · agent · not trainable · 1,314 tokens, 0 with mask 1
    t0   context    968  mask 0
    t0   reply       56  mask 0  no logprobs: mask 0
    t1   context     57  mask 0
    t1   reply      104  mask 0  no logprobs: mask 0
    t2   context     23  mask 0
    t2   reply       53  mask 0  no logprobs: mask 0
    t3   context     21  mask 0
    t3   reply       32  mask 0  no logprobs: mask 0
    [FATAL] nothing_trainable: 0 trainable tokens in a 1314-token sequence: this row contributes no gradient and …
findings  [FATAL] no_logprobs: turn 0: 56 sampled tokens with no logprobs
          [FATAL] no_logprobs: turn 1: 104 sampled tokens with no logprobs
          [FATAL] no_logprobs: turn 2: 53 sampled tokens with no logprobs
          [FATAL] no_logprobs: turn 3: 32 sampled tokens with no logprobs
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 0 · trainable false
project   count_calls 4 · n_sampled 245 · rows kept 0
  tokens with mask 1: 0 · after dedupe_shared_prefixes 0 · item_from_sequence positions 0

saved: every case's full rollout document in ~/.cache/mhrl-course/21-calls-to-rows/rollouts/graph_lab
```

This session names the broken engine, so the proxy probes it before it
answers: capture level `tokens`, a training session. The task is solved, the
grade is correct, and not one token can be learnt: four `FATAL
no_logprobs`, a `FATAL nothing_trainable` on the row, `trainable: false`, 0
tokens. The project's `run_rollout` keeps no row. Nothing raised an error:
this is the silent failure to check for in every run (finding 6).

### Every case at once

```text
━━ every case solved the same task the same way
  case           grade    turns roots rows row-tokens mask-1 trainable n_calls n_sampled
  clean          correct      4     1    1      1,314    245      true       4       245
  retry          correct      5     1    1      1,314    245      true       4       245
  retry-last     correct      5     1    2      2,620    482      true       5       269
  changing-line  correct      4     4    4      4,988    245      true       4       245
  compaction     correct      4     2    2      2,491    245      true       4       245
  no-logprobs    correct      4     1    1      1,314    192      true       4       245
  side-call      correct      5     2    2      1,456    245      true       4       273
  broken-engine  correct      4     1    1      1,314      0     false       4       245
```

## Claude Code and its note: `reminder.py`

`reminder.py` runs Claude Code 2.1.290 on the same task through the proxy,
set up as the project sets it up (chapter 16's flags, a private config and
home, the session id as its key), three times: with the note's setting left
unset (Claude Code's default), `fixed` and `off`. A small relay between
Claude Code and the proxy records each request as Claude Code sent it,
before the proxy rewrote it.

With the scripted engine, the default:

```text
━━ CLAUDE_CODE_TOTAL_TOKENS_REMINDER unset (the default)
claude    exit 0 after 0 s · 4 calls reached the proxy · grade correct (answer.txt '30')
what Claude Code sent (4 model calls, before the proxy)
  call 1  system prompt: 3 blocks, notes 15000000 · messages: user system*
          notes in the messages: 15000000
  call 2  system prompt: 3 blocks, notes 15000000 · messages: user system* assistant user system*
          notes in the messages: 15000000 14999944
  call 3  system prompt: 3 blocks, notes 15000000 · messages: user system* assistant user system* assistant user system*
          notes in the messages: 15000000 14999944 14999896
  call 4  system prompt: 3 blocks, notes 15000000 · messages: user system* assistant user system* assistant user system* assistant user system*
          notes in the messages: 15000000 14999944 14999896 14999896
what the engine received (the proxy's record of each call)
  call 1  messages: system* user · notes 15000000 15000000
  call 2  messages: system* user assistant tool · notes 15000000 15000000 14999944
          system prompt: the call before's, plus "\n\n<total_tokens>14999944 tokens left</total_tokens>" at its end
  call 3  messages: system* user assistant tool assistant tool · notes 15000000 15000000 14999944 14999896
          system prompt: the call before's, plus "\n\n<total_tokens>14999896 tokens left</total_tokens>" at its end
  call 4  messages: system* user assistant tool assistant tool assistant tool · notes 15000000 15000000 14999944 14999896 14999896
          system prompt: the call before's, plus "\n\n<total_tokens>14999896 tokens left</total_tokens>" at its end
```

Claude Code puts a note in its system prompt (15,000,000) and in the message
that describes its environment, and after each tool result it adds a new
message with the role `system` carrying the count that is left. It never
changes an earlier message: what Claude Code sends only grows at its end.
The engine sees something else. The proxy merges every system message into
one at the top (finding 1), so each new note lands at the end of the system
prompt, and no call's prompt begins with the call before:

```text
━━ the same task, the note's three settings
  setting  grade    calls roots rows row-tokens mask-1 trainable n_calls n_sampled
  default  correct      4     4    4     40,460    266      true       4       266
  fixed    correct      4     4    4     40,422    266      true       4       266
  off      correct      4     1    1     10,082    266      true       4       266
```

`fixed` keeps the number at 5,000,000, and the rollout still splits into
four rows: what breaks the chain is the new message after every tool result,
not the changing number. Only `off` keeps one root.

With the model, LFM2.5-2.6B:

```text
━━ CLAUDE_CODE_TOTAL_TOKENS_REMINDER unset (the default)
claude    exit 0 after 35 s · 8 calls reached the proxy · grade correct (answer.txt '30')
what the engine received (the proxy's record of each call)
  call 1  messages: system* user · notes 15000000 15000000
  call 2  messages: system* user assistant tool · notes 15000000 15000000 14999909
          system prompt: the call before's, plus "\n\n<total_tokens>14999909 tokens left</total_tokens>" at its end
  call 3  messages: system* user assistant tool assistant tool · notes 15000000 15000000 14999909 14999909
graph     8 turns · 8 roots (t0, t1, t2, t3, t4, t5, t6, t7) · 8 leaves (t0, t1, t2, t3, t4, t5, t6, t7) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -      2,305     91      1  bash: ls -la data.db
  t1    -      2,451     43      1  Bash: ls -la data.db
  t2    -      2,546     52      1  Bash: sqlite3 data.db ".schema"
  t3    -      2,704     77      1  Bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t4    -      2,805     65      1  Bash: echo "30" > answer.txt
  t5    -      2,900     62      1  read: {"file_path": "~/.c…
  t6    -      3,007     63      1  Bash: cat ~/.cache/mhrl-c…
  t7    -      3,094     32      1  text: 30
findings  [WARN] per_turn_capture_only: every turn is its own root (8)
stats     n_roots 8 · n_sequences 8 · n_trainable_tokens 485 · trainable true
```

```text
━━ CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off
claude    exit 0 after 47 s · 8 calls reached the proxy · grade wrong (answer.txt None)
graph     8 turns · 1 root (t0) · 1 leaf (t7) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -      2,279    131      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t1    t0     2,452     56      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t2    t1     2,550     53      1  bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t3    t2     2,645    712      1  Bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t4    t3     3,368    176      1  write: {"content": "30", "path": "~…
  t5    t4     3,576    874      1  Bash: echo "30" > ~/.cach…
  t6    t5     4,467    116      1  Bash: cat ~/.cache/mhrl-c…
  t7    t6     4,594    293      1  text: The answer has been written to `answer.…
rows      1 for the trainer
  row 1 · agent · trainable · 4,887 tokens, 2,411 with mask 1
    t0   context  2,279  mask 0
    t0   reply      131  mask 1  logprobs -0.000 -0.000 -0.000 -0.000 …
    t1   context     42  mask 0
    t1   reply       56  mask 1  logprobs -0.000 -0.041 -0.014 -0.152 …
    t2   context     42  mask 0
    t2   reply       53  mask 1  logprobs -0.004 -0.239 -0.278 -0.154 …
    t3   context     42  mask 0
    t3   reply      712  mask 1  logprobs -0.005 -0.152 -0.296 -0.349 …
    t4   context     11  mask 0
    t4   reply      176  mask 1  logprobs -0.000 -0.004 -0.002 -0.001 …
    t5   context     32  mask 0
    t5   reply      874  mask 1  logprobs -0.000 -0.978 -0.001 -0.046 …
    t6   context     17  mask 0
    t6   reply      116  mask 1  logprobs -0.000 -0.007 -0.089 -0.002 …
    t7   context     11  mask 0
    t7   reply      293  mask 1  logprobs -0.000 -0.018 -0.000 -0.000 …
findings  none
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 2,411 · trainable true
```

```text
━━ the same task, the note's three settings
  setting  grade    calls roots rows row-tokens mask-1 trainable n_calls n_sampled
  default  correct      8     8    8     22,297    485      true       8       485
  fixed    wrong        8     8    8     31,528  4,078      true       8     4,078
  off      wrong        8     1    1      4,887  2,411      true       8     2,411
```

The model's runs show the same split: eight calls and eight roots with the
note, one root with it off. The first call read 2,305 tokens. Graded, the
model solved the default run only. In the `off` run it found 30 and wrote
it into Claude Code's memory folder (its system prompt says "You have a
persistent file-based memory at" that folder), not into the task folder, so
the grader found no `answer.txt`. In the `fixed` run it called a tool named
`bash` five times (Claude Code's is `Bash`, and it answers that the name is
wrong), then `python`, then wrote two replies of 1,024 tokens with no call in
them, and the proxy's budget of 8 calls ran out. Each run samples at temperature 1, so one run per setting says nothing
about which setting solves more.

Claude Code runs inside chapter 19's sandbox, the project's box: its Bash
runs the model's commands without asking, so it may write only in its run's
folders and reach only this machine.

## The proxy's record against Claude Code's own: `crosscheck.py`

```text
crosscheck.py · ~/sai_work/video_tool-course/films/mhrl-course-21/runlog/scripted-runs/reminder-off: the proxy's record vs Claude Code's log
  call  proxy: wrote  Claude Code: told  reply
     1            56                 56  Bash: sqlite3 data.db .tables             ok
     2           104                104  Bash: sqlite3 data.db "SELECT COUNT(*) …  ok
     3            51                 51  Bash: echo 30 > answer.txt                ok
     4            55                 55  text: 30 patients live in Murcia; answe…  ok
  Claude Code's result: success · 4 turns · output_tokens 266 · input_tokens 0
turns               ok · 4/4: 4 recorded, 4 in Claude Code's log
replies             ok · call by call
completion tokens   ok · proxy 266, Claude Code 266
trainable coverage  ok · agent rows train [56, 104, 51, 55]; Claude Code's replies [56, 104, 51, 55]
```

(The run log ran it on the copies it kept of each run, hence the folder.)
Two programs wrote these two accounts: the proxy (what the engine sampled)
and Claude Code (what it was told). They agree call by call, and the reply
lengths the row trains, `turn_lengths`, are Claude Code's per-call counts.
The model's `fixed` run is where a cross-check earns its keep:

```text
crosscheck.py · ~/sai_work/video_tool-course/films/mhrl-course-21/runlog/model-runs/reminder-fixed: the proxy's record vs Claude Code's log
  call  proxy: wrote  Claude Code: told  reply
     1           101                101  bash: ls -la data.db                      ok
     2           142                142  bash: ls -la data.db                      ok
     3           113                113  bash: ls -la data.db                      ok
     4           686                686  bash: ls -la data.db                      ok
     5           635                635  bash: ls -la data.db                      ok
     6           353                353  python: import sqlite3 conn = sqlite3.c…  ok
     7         1,024              1,024  (empty, cut at the output cap)            ok
     8         1,024              1,024  (empty, cut at the output cap)            ok
     +  answered by the proxy itself: text: Step budget exhausted; stopping.
  note: calls 7, 8 came back empty, cut at the output cap: Claude Code printed no message for them, and the rows hold their 2,048 tokens
  Claude Code's result: success · 9 turns · output_tokens 4078 · input_tokens 0
turns               ok · 8/8: 8 recorded, 8 in Claude Code's log (+1 answered by the proxy itself: the call budget)
replies             ok · call by call
completion tokens   ok · proxy 4,078, Claude Code 4,078
trainable coverage  ok · agent rows train [101, 142, 113, 686, 635, 353, 1024, 1024]; Claude Code's replies [101, 142, 113, 686, 635, 353, 1024, 1024]
```

Calls 7 and 8 were 1,024 tokens each, cut off by the output cap with no text
and no tool call in them. Claude Code prints no message for an empty
reply (its raw stream still carries it and its tokens), so it answered with
nothing, and asked again; the proxy answered the ninth call itself, as its
budget says. The rows still hold those 2,048 tokens with mask 1, beside two
`WARN truncated_turn` findings. Claude Code was also told `input_tokens 0`
on every call (finding 7).

## Plan versus measured

| the plan said | measured |
| --- | --- |
| `graph_lab.py`: six scripted conversations through the real proxy, printed as graph, masked rows with logprobs, findings and stats; `count_calls`, `n_sampled`, `dedupe_shared_prefixes` applied | eight: the six, the plan's broken engine, and the retry split in two (a middle call, the last call), every one through OpenEnv `3f14355` started by `start.sh`, every one graded correct |
| clean chain: 1 root, 1 row, mask 1 only on the model's own tokens | 1 root, 1 row of 1,314 tokens, mask 1 on exactly the 245 tokens the model wrote |
| retry: a fork, the abandoned sibling discarded and given no reward, 2 rows sharing a prefix whose sampled tokens the dedup trains once | two shapes (finding 2). A retried middle call: a fork, the sibling discarded, its row left out of the trainer's document (`export.py` 145, `server.py` 852 `include_discarded: bool = False`), 1 row, nothing for the dedup. A retried last call: nothing discarded (`graph.py` 416-438: a sibling is discarded only when another went on), 2 trainable rows sharing 1,288 tokens, 482 mask-1 tokens counted (`export.py` 210), 263 after the dedup; `count_calls` 5, `n_sampled` 269 |
| 11.8% of the project's dry-run tokens were such repeats | 778 of 6,567 mask-1 tokens (11.8%) in `runs/dry/grpo/rollouts.jsonl`, 3 of 24 rollouts, all reward 0, re-measured. Not mostly retries (finding 3): 541 of the 778 are in two rollouts whose rows part inside a reply that came back as other token ids; 237 in one retry |
| changing line: 4 roots and 4 single-turn rows; `reminder.py` shows Claude Code 2.1.x's note doing exactly this until it is switched off | 4 roots, 4 rows, `WARN per_turn_capture_only`, rows 4,988 tokens against 1,314. Claude Code: 4 roots (scripted) and 8 (the model) by default, 1 with `off`. But 2.1.290 changes no line (finding 1): it adds a new system message after each tool result, and the proxy's `_merge_developer_role` (`dialects/base.py` 71-107, called at 125) merges every system message into the first; `fixed`, a number that never changes, splits it just the same |
| compaction: a new root | 2 roots, 2 rows, `WARN multiple_roots` |
| missing logprobs: the turn kept as context with mask 0 and a finding | as planned: 53 tokens with mask 0 and `FATAL no_logprobs`; and `trainable` stays `true` (finding 4: the turn's findings go to `session.findings`, `server.py` 1293/1302; `trainable` reads only `check_rollout`, `export.py` 122 and 214) |
| side call: auxiliary, not counted as an agent step | auxiliary and not trainable, `count_calls` 4; but `n_sampled` 273, not 245: the project sums every turn not discarded (finding 5: `mhrl/rollout.py` 272-275, paid at 295 `reward_for(…, n_sampled, …)`) |
| `n_calls` and `n_sampled` are the two efficiency numbers | as above: `count_calls` 4 in every case but the retried last call (5); `n_sampled` 245, 269 or 273 |
| a broken engine: the answer is right, yet `trainable: false` with 0 trainable tokens | as planned: an engine that drops log-probabilities whenever tools are offered; the proxy probed it and measured `tokens` (finding 6: its three probes, `validate_llm.py` 585, 276 with 319, and 391, send tools only once and then check only for a tool call) |
| `reminder.py`: Claude Code with `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` on and off | unset (the default), `fixed` and `off`, through a relay that records what Claude Code sent; the note is `<total_tokens>N tokens left</total_tokens>`, 15,000,000 at the start |
| `crosscheck.py`: 'turns 6/6, completion tokens match', like the article's check against Harbor's trajectory file | scripted: 4/4 and 266 = 266, three times; the model: 8/8 three times, 485, 4,078 and 2,411 tokens matching. Per-call counts exist only with `--include-partial-messages`, and Claude Code is told `input_tokens 0` (finding 7: `dialects/anthropic.py` 83 and 179) |
| folder `course/21-calls-to-rows/` | `21-calls-to-rows/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its dry run's 11.8%
  (quoted from `mhrl/training/common.py` and re-measured from its file) and
  its code, quoted by line.
- **The article's numbers**: there are none in this chapter.

## Findings about the proxy and the project

Line numbers are in OpenEnv `3f14355`, `openenv/core/harness/capture/`, and
in the project at `25074c0`; all of them, with the two measurements, are in
the run log's `14-measure-findings.txt`.

1. **Claude Code's note becomes a growing system prompt only in the proxy.**
   2.1.290 adds a message with the role `system` after each tool result and
   never edits an earlier one. `_merge_developer_role` (`dialects/base.py`
   71, "Merge multiple system messages into one at the top" at 85, the join
   at 107, called for every dialect at 125) moves all of them into one
   system prompt, so the prompt grows at its end and no call continues the
   last. Measured with the proxy's own converter on Claude Code's recorded
   call 2:

   ```text
   finding 1, measured: Claude Code's call 2 (scripted-runs/reminder-default/sent.jsonl)
     user notes none
     system notes ['15000000']
     assistant notes none
     user notes none
     system notes ['14999944']
     (and its system prompt: notes ['15000000'])
   the same request after AnthropicTransformer().transform_request (the proxy's code):
     system notes ['15000000', '15000000', '14999944']
     user notes none
     assistant notes none
     tool notes none
   ```

2. **A retry is discarded only when the retried call went on.**
   `graph.py` 416 ("a sibling is only called discarded when at least one of
   its siblings continued") and 438 (`continue  # all siblings are
   terminal: ambiguous, keep them all`). A discarded row is left out of the
   trainer's document (`export.py` 145, default at `server.py` 852); two
   final replies both stay, and `n_trainable_tokens` (`export.py` 210) counts
   their shared start twice: 482, 263 after the dedup.
3. **The 11.8% comes mostly from replies re-cut into other tokens.** In the
   dry run's three forked rollouts:

   ```text
   finding 3, measured: ~/sai_work/multi-harness-rl/runs/dry/grpo/rollouts.jsonl, rollouts with more than one row
     school-easy-018 · mini-swe-agent · reward 0.0
       row 2 leaves row 1 3 tokens before row 1's last reply
         row 1:   ' 0,\n  "output": ""\n}<|im_end|>\n<|im_start|>assistant\n<think>The query was successful and wrote.</think>The query returned'
         row 2:   ' 0,\n  "output": ""\n}<|im_end|>\n<|im_start|>user\nTool call error:\n\n<error>\nNo tool calls found in'
         ids at the parting: [63514, 207, 124901, 597, 13174, 452] vs [5922, 207, 29963, 1367, 5082, 63990]
         row 1's reply is not in row 2's prompt (a retry)
       repeats the dedup masks in this rollout: 237
     shop-hard-005 · claude-code · reward 0.0
       row 2 leaves row 1 inside row 1's last reply, 5 tokens before its end
         row 1:   " p.name FROM products p WHERE p.category = \\'Toys\\'')]<|tool_call_end|><|im_end|>"
         row 2:   " p.name FROM products p WHERE p.category = \\'Toys\\'')]<|tool_call_end|><|im_end|>\n<|im_start|>tool\n[Tool Error] Ex"
         ids at the parting: [26301, 7609, 69, 124906, 124900] vs [68, 7191, 17733, 124906, 124900, 207]
         the same text, cut into other token ids (a recut)
       repeats the dedup masks in this rollout: 277
     school-medium-008 · mini-swe-agent · reward 0.0
       row 2 leaves row 1 inside row 1's last reply, 6 tokens before its end
         row 1:   ')) FROM (SELECT d.student_id e FROM students))"\')]<|tool_call_end|><|im_end|>'
         row 2:   ')) FROM (SELECT d.student_id e FROM students))"\')]<|tool_call_end|><|im_end|>\n<|im_start|>tool\n{\n  "return'
         ids at the parting: [17, 22398, 7609, 69, 124906, 124900] vs [3364, 10, 7609, 69, 124906, 124900]
         the same text, cut into other token ids (a recut)
       row 3 leaves row 2 inside row 2's last reply, 12 tokens before its end
         row 2:   ' LEFT JOIN enrollments e ON ... WHERE e.term=\\\'Fall-2025\\\')"\')]<|tool_call_end|><|im_end|>'
         row 3:   ' LEFT JOIN enrollments e ON ... WHERE e.term=\\\'Fall-2025\\\')"\')]<|tool_call_end|><|im_end|>\n'
         ids at the parting: [37, 26301, 63140, 21, 1217, 29] vs [4563, 15, 63140, 21, 1217, 29]
         the same text, cut into other token ids (a recut)
       repeats the dedup masks in this rollout: 264
   ```

   A recut parts the rows inside the last reply: the next prompt spells the
   same text with other ids, so the proxy cannot attach the next call to
   the reply, and both calls become leaves that no sibling outlived.
4. **`trainable` stays true after a per-turn FATAL.** `check_turn`'s
   findings go to `session.findings` (`server.py` 1293, 1302), and the turn
   is masked (`graph.py` 470); `trainable` is `bool(trainable_rows) and
   rollout_report.ok` (`export.py` 214), where `rollout_report` is
   `check_rollout` alone (122). Measured: `FATAL no_logprobs`, `trainable:
   true`, 192 of 245 tokens.
5. **`n_sampled` pays the token bonus for a side call.** `count_calls` keeps
   only turns that offered tools (`mhrl/rollout.py` 216), but `n_sampled`
   sums every turn not discarded (272-275), and the reward is paid on it
   (295). Measured: 273 against 245, the title's 28 tokens. Claude Code made
   no side call in any run here, so the project's runs were not affected.
6. **An engine that drops log-probabilities with tools passes every probe.**
   The capture probe sends no tools (`validate_llm.py` 585), the tool probe
   sends one and checks only that a tool call came back (276, 319), and the
   log-probability probe sends none (391). Measured: capture level
   `tokens`, a training session, a correct answer and 0 trainable tokens.
7. **Claude Code's log needs `--include-partial-messages`, and it sees
   `input_tokens 0`.** The proxy's stream starts every reply with
   `"usage": {"input_tokens": 0, "output_tokens": 0}` (`dialects/anthropic.py`
   83) and ends it with `output_tokens` only (179). Claude Code prints its
   reply events when the reply's parts finish, before that last count, so
   they show 0 output tokens; the per-call count is only in the raw stream
   events that flag adds. Every result shows `input_tokens 0`. Claude Code
   also prints no message for an empty reply.

One finding about this chapter's own code, from the run log: the first
`crosscheck.py` paired replies by their printed messages, so the model's
`fixed` run read `8/7` and `MISMATCH` after its two empty replies. It now
counts replies from the raw stream and sets the proxy's own budget answer
aside; it ran again on the runs the log kept, and every other output is from
the commit before.

## The files

Line counts from the run log: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `graph_lab.py` | 718 | 436 | the eight cases, the dry-run figure, the closing table |
| `reminder.py` | 453 | 270 | Claude Code with the note unset, `fixed` and `off`: sent, received, built |
| `crosscheck.py` | 438 | 234 | the proxy's record against Claude Code's stream-json log |
| `training_rows.py` | 196 | 68 | the project's `count_calls`, `n_sampled`, `dedupe_shared_prefixes`, `item_from_sequence` |
| `rollout_view.py` | 419 | 208 | prints a rollout document: graph, rows span by span, findings, stats |
| `claude_code.py` | 578 | 319 | the pinned Claude Code: install, private environment, the project's flags, the box, the relay |
| `sandbox.py`, `sandbox.sb` | 300 | 141 | chapter 19's sandbox, the project's Seatbelt profile |
| `scripted_engine.py`, `scripted_replies.py` | 408, 236 | 185, 91 | chapter 20's engine with a scripted model, its remote control and the broken copy |
| `start.sh` | 285 | 217 (not blank, not comments) | chapter 20's stack, with `scripted` |
| `capture_client.py` | 291 | 131 | chapter 20's client, with the rollout route's two options |
| `engine.py`, `policy.py`, `lfm.py`, `lfm_text.py`, `toolcalls.py`, `tiny_harness_http.py`, `bash_tool.py`, `database.py`, `grader.py`, `tasks.py`, `data/` | | | chapter 20's, unchanged; `lfm_files.py` with this chapter's name |

## What this chapter leaves out

- **Breaking a real capture on purpose**, three ways, and catching each
  break with `crosscheck.py`, is chapter 22.
- **Subagents** are described, not run: Claude Code made no side call and
  started no subagent here.
- **The other harnesses** (mini-swe-agent, Pi) come through the proxy in
  chapter 23.

## Mirrors in the project

OpenEnv's capture graph (`graph.py`, `export.py`, `validate.py` at
`3f14355`), `mhrl/rollout.py` (`count_calls`, `run_rollout`) and
`mhrl/training/common.py` (`dedupe_shared_prefixes`, `item_from_sequence`).
