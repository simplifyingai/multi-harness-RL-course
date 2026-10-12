# Chapter 22. Capture that fails without an error

Chapter 21 cut a rollout into training rows. This chapter takes **one real
run**: Claude Code with LFM2.5-2.6B, solved, captured as one clean row. It
replays that run with the model's replies fixed, token for token, and breaks
one thing at a time.

The puzzle: three things go wrong in the capture. The harness writes a note
that changes on every call, the history is summarised halfway, and one reply
loses its log-probabilities. Does anything raise an error? Measured: no. The
task is solved every time, and the proxy's one verdict, `trainable`, says
`true` every time. The rows tell another story: 8 rows instead of 1, 2 rows
instead of 1, and 61 tokens the trainer will never learn from. Only a check
against Claude Code's own log catches all three.

## Words first

- A **capture** is one run recorded by chapter 20's proxy: every call's exact
  tokens and log-probabilities. Here it also keeps what Claude Code sent and
  Claude Code's own log.
- A **replay** runs Claude Code again on the same task, in the same folder,
  while the engine answers each call with the capture's reply: the exact
  token ids and log-probabilities, read back out of the capture's row. The
  model is not loaded; Claude Code, the proxy and the grader all run for
  real. So two replays differ only in what we broke.
- The **rollout document** is what the proxy hands the trainer for one run:
  every call (its prompt and reply sizes, its tools), the rows, the graph's
  counts (`n_roots`, `n_sequences`, `n_trainable_tokens`), its findings, and
  one word, `trainable`.
- A **silent capture failure** is a run that looks fine (no error, task
  solved, `trainable: true`) while its rows are split or partly empty.
- A **prefix-breaking note** is text that changes on every call near the top
  of the prompt. Each call's prompt then no longer begins with the call
  before, so the proxy cannot chain them (chapter 21).
- **Compaction** is a harness replacing its long history with a summary to
  save room. The next prompt does not begin with the old tokens, so the
  proxy starts a new row there.
- **Log-probability validation**: the proxy checks each call as it arrives.
  A reply without log-probabilities is kept as context with mask 0 and a
  `FATAL no_logprobs` finding, because training needs every learned token's
  log-probability.
- A **trace cross-check** compares the proxy's record with the harness's own
  account of the same run: here Claude Code's stream-json log, one JSON line
  per event. The article checks its captures this way against Harbor's
  trajectory files.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/22-silent-capture-failures
ENGINE_PORT=8119 ./start.sh scripted   # no model: the scripted engine on 8119, the proxy on 8200
uv run capture.py                      # the run to break: Claude Code, solved, one row
uv run breakages.py                    # replayed four ways
uv run crosscheck.py all               # each run against Claude Code's own log
uv run compaction.py                   # does Claude Code compact through the proxy?
./start.sh stop
ENGINE_PORT=8119 ./start.sh            # the model: chapter 18's engine with the 4-bit LFM2.5
uv run capture.py --attempts 5
./start.sh stop
ENGINE_PORT=8119 ./start.sh replay     # the scripted engine cutting LFM2.5's tokens, no weights
uv run breakages.py
uv run crosscheck.py all
./start.sh stop
```

`./start.sh scripted` starts the scripted engine (no model, one token per
character) and the capture proxy. `./start.sh` starts the model.
`./start.sh replay` starts the scripted engine with LFM2.5's own
`tokenizer.json`, which a capture the model made needs. All three take the
engine's port as `ENGINE_PORT` (8119 in the run log, where another program
holds the project's 8100). The proxy keeps 8200.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
(for `npx`), chapter 11's model in `COURSE_HOME/models` for the model's
half, and the network the first time. `start.sh` installs the proxy into
`COURSE_HOME/22-silent-capture-failures/proxy-env`, and `capture.py`
installs Claude Code 2.1.290 with `npx`.

Every output below is copied from the run log. The log was made from a clean
`COURSE_HOME/22-silent-capture-failures`, with everything under `nice -n 10`.
The model's engine started only after a minute with no film rendering on the
machine; the run waited 273 s. Seconds are wall time. Memory is the largest
single process the command ran (`/usr/bin/time -l`); for the scripts that
start Claude Code, that is Claude Code.

| command | what it shows | seconds and memory |
| --- | --- | --- |
| `./start.sh install` | OpenEnv's capture proxy at `3f14355` | exit 0 · 3.3 s · max resident memory 0.16 GB |
| `ENGINE_PORT=8119 ./start.sh scripted` | the scripted engine, the proxy at `tokens` | exit 0 · 3.0 s · max resident memory 0.11 GB |
| `uv run capture.py` (scripted) | Claude Code installed, then one run: 4 calls, 1 row | exit 0 · 13.2 s · max resident memory 0.22 GB |
| `uv run breakages.py` (scripted) | four replays, all solved | exit 0 · 3.4 s · max resident memory 0.22 GB |
| `uv run crosscheck.py all` (scripted) | each break caught; exit 1 by design | exit 1 · 0.1 s · max resident memory 0.04 GB |
| `uv run compaction.py` | no compaction | exit 0 · 1.3 s · max resident memory 0.22 GB |
| `./start.sh stop` | `engine stopped · 28 requests` | exit 0 · 2.1 s · max resident memory 0.00 GB |
| `uv run pytest tests/test_22_silent_capture_failures.py` | 45 passed, against the real proxy | exit 0 · 19.5 s · max resident memory 0.31 GB |
| `ENGINE_PORT=8119 ./start.sh` | the model's engine (1.6 GB loaded), the proxy at `tokens` | exit 0 · 10.9 s · max resident memory 0.11 GB |
| `uv run capture.py --attempts 5` (the model) | solved on the first attempt: 8 calls, 1 row | exit 0 · 16.1 s · max resident memory 0.22 GB |
| `./start.sh stop` | `engine stopped · 12 requests · peak memory 2.67 GB, the weights included` | exit 0 · 2.1 s · max resident memory 0.00 GB |
| `ENGINE_PORT=8119 ./start.sh replay` | LFM2.5's tokenizer, no weights | exit 0 · 14.3 s · max resident memory 0.11 GB |
| `uv run breakages.py` (the model's capture) | four replays, all solved | exit 0 · 3.8 s · max resident memory 0.24 GB |
| `uv run crosscheck.py all` (the model's capture) | each break caught; exit 1 by design | exit 1 · 0.1 s · max resident memory 0.04 GB |

The proxy held 0.13 GB (scripted), 0.11 GB (the model) and 0.12 GB (the
replay) just before each stop. The run log ran at commit `737cba9`. Its
compaction run was repeated at `9359065`, which sizes that run's long reply
under the output cap (the log keeps both runs). The test file has 46 tests at
the final commit; the run log's test run had 45.

## The capture: `capture.py`

```text
capture.py · Claude Code 2.1.290 through the capture proxy at http://127.0.0.1:8200, in front of the model
  task clinic-easy-001: How many patients live in Murcia?
attempt 1  exit 0 after 15 s · 8 calls · grade correct (answer.txt '30') · roots 1, trainable true · kept in ~/.cache/mhrl-course/22-silent-capture-failures/runs/attempt-1
capture   attempt-1, kept as ~/.cache/mhrl-course/22-silent-capture-failures/runs/capture
billing   Claude Code began its system prompt with 'x-anthropic-billing-header: cc_version=2.1.290.212; cc_entrypoint=sdk-cli;' on 9 of 9 calls (the same line every time); the engine received it 0 times
graph     8 turns · 1 root (t0) · 1 leaf (t7) · discarded: none
  turn  parent  read   wrote  tools  reply
  t0    -      2,281     99      1  bash: sqlite3 data.db ".tables"
  t1    t0     2,422     41      1  Bash: sqlite3 data.db ".tables"
  t2    t1     2,479     62      1  Bash: sqlite3 data.db "SELECT name FROM patie…
  t3    t2     3,252     68      1  Bash: sqlite3 data.db "PRAGMA table_info(pati…
  t4    t3     3,371     61      1  Bash: sqlite3 data.db "SELECT COUNT(*) FROM p…
  t5    t4     3,443    115      1  Write: {"file": "~/.cache/mhrl-course/22-sile…
  t6    t5     3,606    255      1  Bash: echo "30" > ~/.cache/mhrl-course/22-sil…
  t7    t6     3,878     62      1  Bash: cat ~/.cache/mhrl-course/22-silent-capt…
rows      1 for the trainer
  row 1 · agent · trainable · 3,940 tokens, 763 with mask 1
    t0   context  2,281  mask 0
    t0   reply       99  mask 1  logprobs -0.000 -0.000 -0.000 -0.000 …
    t1   context     42  mask 0
    t1   reply       41  mask 1  logprobs -0.000 -0.018 -0.009 -0.054 …
    t2   context     16  mask 0
    t2   reply       62  mask 1  logprobs -0.114 -0.000 -0.000 -0.001 …
    t3   context    711  mask 0
    t3   reply       68  mask 1  logprobs -0.014 -1.753 -0.087 -0.195 …
    t4   context     51  mask 0
    t4   reply       61  mask 1  logprobs -0.000 -0.001 -0.000 -0.000 …
    t5   context     11  mask 0
    t5   reply      115  mask 1  logprobs -0.000 -0.001 -0.034 -0.000 …
    t6   context     48  mask 0
    t6   reply      255  mask 1  logprobs -0.000 -0.052 -0.000 -0.003 …
    t7   context     17  mask 0
    t7   reply       62  mask 1  logprobs -0.000 -0.000 -0.000 -0.000 …
findings  none
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 763 · trainable true
project   count_calls 8 · n_sampled 763
```

Claude Code ran the task the way the project runs it: one tool (`Bash`),
the total-tokens note off, the session id as its key, and chapter 19's
sandbox. LFM2.5 called a tool named `bash` first. Claude Code answered
"No such tool available: bash. Tool names are case-sensitive: call Bash
instead." The model then listed the tables, read the patient names and the
table's columns, and counted 30. It tried `Write`, which this session
disables, and wrote `30` with `echo` instead. Claude Code's ninth call was
answered by the proxy itself, because the 8-call budget was spent. The
grade is correct: one root, one row of 3,940 tokens, 763 of them the model's
own (mask 1). This is the clean baseline every case breaks.

The `billing` line is a prefix breaker the proxy already handles. Claude
Code puts a billing line at the top of its system prompt; the engine never
sees it (finding 4).

## One capture, replayed four ways: `breakages.py`

```text
breakages.py · one Claude Code run, replayed through the capture proxy at http://127.0.0.1:8200 with its replies fixed
  capture ~/.cache/mhrl-course/22-silent-capture-failures/runs/capture: model engine, 8 calls, 1 root, 1 row, 763 tokens to train, grade correct
  task clinic-easy-001: How many patients live in Murcia?
  breaks at call 5 of 8

━━ replay · nothing changed: the capture, replayed
claude    exit 0 after 0 s · CLAUDE_CODE_TOTAL_TOKENS_REMINDER off · 8 calls · grade correct (answer.txt '30')
replies   the capture's 8, token for token
rows      1
  row 1 · agent · trainable     · calls 1-8  ·   3,961 tokens,    763 with mask 1
findings  none
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 763 · trainable true
project   run_rollout keeps 1 row, 763 tokens to train · count_calls 8 · n_sampled 763
saved     ~/.cache/mhrl-course/22-silent-capture-failures/runs/replay
```

The control first. Every call got the capture's reply, token for token, and
the run solved the task again with one row and the same 763 trained tokens.
The row is 3,961 tokens, not 3,940, because the replay engine lays prompts
out its own way. The replies, the only tokens a trainer learns from, are
identical. (The scripted capture's replay below is identical in every token
of every row.)

### A note that changes on every call

```text
━━ note · Claude Code's total-tokens note, left on (its default)
claude    exit 0 after 0 s · CLAUDE_CODE_TOTAL_TOKENS_REMINDER unset · 8 calls · grade correct (answer.txt '30')
replies   the capture's 8, token for token
changed   the engine's system prompt at call 2 is the call before's, plus "\n\n<total_tokens>14999901 tokens left</total_tokens>" at its end; it grew at its end on 7 of 7 calls
rows      8
  row 1 · agent · trainable     · call 1     ·   2,427 tokens,     99 with mask 1
  row 2 · agent · trainable     · call 2     ·   2,523 tokens,     41 with mask 1
  row 3 · agent · trainable     · call 3     ·   2,614 tokens,     62 with mask 1
  row 4 · agent · trainable     · call 4     ·   3,406 tokens,     68 with mask 1
  row 5 · agent · trainable     · call 5     ·   3,531 tokens,     61 with mask 1
  row 6 · agent · trainable     · call 6     ·   3,670 tokens,    115 with mask 1
  row 7 · agent · trainable     · call 7     ·   3,986 tokens,    255 with mask 1
  row 8 · agent · trainable     · call 8     ·   4,078 tokens,     62 with mask 1
findings  [WARN] per_turn_capture_only: every turn is its own root (8)
stats     n_roots 8 · n_sequences 8 · n_trainable_tokens 763 · trainable true
project   run_rollout keeps 8 rows, 763 tokens to train · count_calls 8 · n_sampled 763
saved     ~/.cache/mhrl-course/22-silent-capture-failures/runs/note
```

Claude Code's default writes `<total_tokens>N tokens left</total_tokens>`
after every tool result. The proxy merges every system message into the
system prompt (chapter 21, finding 1), so the engine's system prompt grows
at its end on every call. No call continues the one before, and the run
becomes 8 rows of one call each. Every reply is still trained (763 tokens),
but each row repeats the whole prompt before its call: 26,235 row tokens
against 3,961.

### A summarised history

```text
━━ summary · the history before call 5 summarised
claude    exit 0 after 0 s · CLAUDE_CODE_TOTAL_TOKENS_REMINDER off · 8 calls · grade correct (answer.txt '30')
replies   the capture's 8, token for token
changed   call 5: Claude Code sent 10 messages, the proxy got 4: 7 replaced by one summary (1 system message kept), "This conversation was summarised to save room.\nThe task: Th…"
rows      2
  row 1 · agent · trainable     · calls 1-4  ·   3,341 tokens,    270 with mask 1
  row 2 · agent · trainable     · calls 5-8  ·   3,068 tokens,    493 with mask 1
findings  [WARN] multiple_roots: 2 roots across 8 turns
stats     n_roots 2 · n_sequences 2 · n_trainable_tokens 763 · trainable true
project   run_rollout keeps 2 rows, 763 tokens to train · count_calls 8 · n_sampled 763
saved     ~/.cache/mhrl-course/22-silent-capture-failures/runs/summary
```

Before call 5, the history is replaced by one summary on its way to the
proxy: the task, then each command with the first line of its answer. A
small relay between Claude Code and the proxy does this, playing a harness
that compacts. Claude Code 2.1.290 did not compact by itself in this
chapter's runs, even when asked to (`compaction.py`, below). Claude Code's
environment block is a message with the role `system`, and the summary keeps
it, as a harness keeps its own instructions (finding 5). Calls 5 to 8 become
a second row.

### One reply without log-probabilities

```text
━━ no-logprobs · call 5 served without log-probabilities
claude    exit 0 after 0 s · CLAUDE_CODE_TOTAL_TOKENS_REMINDER off · 8 calls · grade correct (answer.txt '30')
replies   the capture's 8, token for token
changed   call 5: its 61 tokens, served with no logprobs
rows      1
  row 1 · agent · trainable     · calls 1-8  ·   3,961 tokens,    702 with mask 1
findings  [FATAL] no_logprobs: turn 4: 61 sampled tokens with no logprobs
stats     n_roots 1 · n_sequences 1 · n_trainable_tokens 702 · trainable true
project   run_rollout keeps 1 row, 702 tokens to train · count_calls 8 · n_sampled 763
saved     ~/.cache/mhrl-course/22-silent-capture-failures/runs/no-logprobs
```

Call 5 came back with every log-probability null. Its 61 tokens stay in the
row as context with mask 0, and the proxy writes `FATAL no_logprobs`. Yet the
verdict is still `trainable: true`, with 702 tokens instead of 763.

### All four

```text
━━ one capture, replayed 4 ways
  case         grade    calls roots rows row-tokens mask-1 trainable
  replay       correct      8     1    1      3,961    763      true
  note         correct      8     8    8     26,235    763      true
  summary      correct      8     2    2      6,409    763      true
  no-logprobs  correct      8     1    1      3,961    702      true
  solved 4 of 4; the proxy's verdict, trainable, is true in 4 of 4. Check each against Claude Code's own log: uv run crosscheck.py all
```

Solved four times, and `trainable: true` four times. The proxy's single gate
caught none of the three breaks. Nor do the project's own checks:
`run_rollout` keeps every trainable row and reads only `trainable` and
`n_trainable_tokens`, and `smoke.py` fails a run only when it is unfinished
or untrainable (finding 2).

## The proxy's record against Claude Code's own: `crosscheck.py`

The capture, as chapter 21's checks see it:

```text
crosscheck.py · ~/.cache/mhrl-course/22-silent-capture-failures/runs/capture: the proxy's record vs Claude Code's log
  call  proxy: wrote  Claude Code: told  reply
     1            99                 99  bash: sqlite3 data.db ".tables"           ok
     2            41                 41  Bash: sqlite3 data.db ".tables"           ok
     3            62                 62  Bash: sqlite3 data.db "SELECT name FROM…  ok
     4            68                 68  Bash: sqlite3 data.db "PRAGMA table_inf…  ok
     5            61                 61  Bash: sqlite3 data.db "SELECT COUNT(*) …  ok
     6           115                115  Write: {"file": "~/.cache/mhrl-course/2…  ok
     7           255                255  Bash: echo "30" > ~/.cache/mhrl-course/…  ok
     8            62                 62  Bash: cat ~/.cache/mhrl-course/22-silen…  ok
     +  answered by the proxy itself: text: Step budget exhausted; stopping.
  Claude Code's result: success · 9 turns · output_tokens 763 · input_tokens 0
turns               ok · 8/8: 8 recorded, 8 in Claude Code's log (+1 answered by the proxy itself: the call budget)
replies             ok · call by call
completion tokens   ok · proxy 763, Claude Code 763
trainable coverage  ok · agent rows train [99, 41, 62, 68, 61, 115, 255, 62]; Claude Code's replies [99, 41, 62, 68, 61, 115, 255, 62]
calls per row       ok · Claude Code's log: 1 conversation (calls 1-8); agent rows: calls 1-8
the proxy's verdict: trainable true · n_roots 1 · n_sequences 1 · findings none
the cross-check: the two accounts agree
```

Two programs wrote these accounts: the proxy (what the engine sampled) and
Claude Code (what it was told). They agree call by call. Chapter 21's four
checks are the first four lines. On the note and the summary they all still
say `ok`: every reply is there, every token is counted, and every reply is
trained somewhere. This chapter adds **calls per row**. Claude Code's log is
one conversation, or one more for each compaction it reports (it writes a
`compact_boundary` event when it compacts), and each conversation should be
one row.

The note:

```text
turns               ok · 8/8: 8 recorded, 8 in Claude Code's log (+1 answered by the proxy itself: the call budget)
replies             ok · call by call
completion tokens   ok · proxy 763, Claude Code 763
trainable coverage  ok · agent rows train [99, 41, 62, 68, 61, 115, 255, 62]; Claude Code's replies [99, 41, 62, 68, 61, 115, 255, 62]
calls per row       MISMATCH · Claude Code's log: 1 conversation (calls 1-8); agent rows: call 1 | call 2 | call 3 | call 4 | call 5 | call 6 | call 7 | call 8
                    why: calls 2, 3, 4, 5, 6, 7, 8 each start a row nothing announced; call 2: its system prompt is the call before's, plus "\n\n<total_tokens>14999901 tokens left</total_tokens>" at its end
the proxy's verdict: trainable true · n_roots 8 · n_sequences 8 · findings [WARN] per_turn_capture_only
the cross-check: the two accounts disagree on 1 of 5 checks
```

The summary:

```text
turns               ok · 8/8: 8 recorded, 8 in Claude Code's log (+1 answered by the proxy itself: the call budget)
replies             ok · call by call
completion tokens   ok · proxy 763, Claude Code 763
trainable coverage  ok · agent rows train [99, 41, 62, 68, 61, 115, 255, 62]; Claude Code's replies [99, 41, 62, 68, 61, 115, 255, 62]
calls per row       MISMATCH · Claude Code's log: 1 conversation (calls 1-8); agent rows: calls 1-4 | calls 5-8
                    why: call 5 starts a row nothing announced: its history was replaced: 4 messages, where continuing the call before takes at least 10; message 2 now begins "This conversation was summarised to save room.\nTh…"
the proxy's verdict: trainable true · n_roots 2 · n_sequences 2 · findings [WARN] multiple_roots
the cross-check: the two accounts disagree on 1 of 5 checks
```

The missing log-probabilities. Here the old **trainable coverage** check
catches it, and now names the call and the proxy's finding:

```text
turns               ok · 8/8: 8 recorded, 8 in Claude Code's log (+1 answered by the proxy itself: the call budget)
replies             ok · call by call
completion tokens   ok · proxy 763, Claude Code 763
trainable coverage  MISMATCH · agent rows train [99, 41, 62, 68, 0, 115, 255, 62]; Claude Code's replies [99, 41, 62, 68, 61, 115, 255, 62]
                    why: call 5: Claude Code was told 61 tokens, the rows train 0 ([FATAL] no_logprobs: turn 4: 61 sampled tokens with no logprobs. GRPO's importance ratio needs the behaviour-policy logprob for every trainable token; without them the turn can only be context.)
calls per row       ok · Claude Code's log: 1 conversation (calls 1-8); agent rows: calls 1-8
the proxy's verdict: trainable true · n_roots 1 · n_sequences 1 · findings [FATAL] no_logprobs
the cross-check: the two accounts disagree on 1 of 5 checks
```

```text
━━ the proxy's record against Claude Code's log, run by run
  run          turns     replies   tokens    coverage  calls/row proxy: trainable
  capture      ok        ok        ok        ok        ok        true
  replay       ok        ok        ok        ok        ok        true
  note         ok        ok        ok        ok        MISMATCH  true
  summary      ok        ok        ok        ok        MISMATCH  true
  no-logprobs  ok        ok        ok        MISMATCH  ok        true
```

Each break is caught, each by a different check, while the proxy's verdict
says `true` in every row. The "why" lines name the cause from the proxy's own
record of the call that starts the row, against the call before:

- the note: the system prompt grew at its end;
- the summary: the history is shorter than continuing would take.

## Does Claude Code compact? `compaction.py`

```text
compaction.py · does Claude Code 2.1.290 compact its history through the capture proxy at http://127.0.0.1:8200?
  settings CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000 · CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=1: compact past 1% of 100,000 tokens, about 1,000
claude    exit code 0 after 1 s · 4 prompts sent · grade correct (answer.txt '30')
engine    read 9,693, 9,806, 10,839, 10,942 tokens; wrote 56, 1,010, 51, 55
log       compact_boundary events: 0 · Claude Code was told input_tokens 0
result    no compaction: 1 root, 1 row · saved ~/.cache/mhrl-course/22-silent-capture-failures/runs/compaction
```

`compaction.py` asks Claude Code to compact early:
`CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000` sets the size its compaction counts
against, and `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=1` asks it to compact past 1%
of that. Every prompt was about 10,000 tokens, and the second reply alone
was 1,010. Claude Code exited with code 0 after 4 prompts. Its own log holds
no `compact_boundary` event, and the proxy built one root. The first run of
this step (kept in the log) asked for a reply that the 1,024-token cap cut
off; it did not compact either.

A guess, not measured: through the proxy, Claude Code is told
`input_tokens 0` for every call (chapter 21, finding 7). If it counts its
context from what the provider reports, it never sees the context fill up.
So this chapter's summary case is applied by the relay, and Claude Code's
own log knows nothing of it.

## Without the model

The same chapter with the scripted engine: Claude Code, four scripted
replies, one token per character.

```text
━━ one capture, replayed 4 ways
  case         grade    calls roots rows row-tokens mask-1 trainable
  replay       correct      4     1    1     10,091    266      true
  note         correct      4     4    4     40,448    266      true
  summary      correct      4     2    2     20,030    266      true
  no-logprobs  correct      4     1    1     10,091    215      true
  solved 4 of 4; the proxy's verdict, trainable, is true in 4 of 4. Check each against Claude Code's own log: uv run crosscheck.py all
```

```text
━━ the proxy's record against Claude Code's log, run by run
  run          turns     replies   tokens    coverage  calls/row proxy: trainable
  capture      ok        ok        ok        ok        ok        true
  replay       ok        ok        ok        ok        ok        true
  note         ok        ok        ok        ok        MISMATCH  true
  summary      ok        ok        ok        ok        MISMATCH  true
  no-logprobs  ok        ok        ok        MISMATCH  ok        true
```

Here the replay rebuilt every row of the capture exactly, all 10,091 tokens:
`replies   the capture's 4, token for token; every row too, all its tokens`.

## Plan versus measured

| the plan said | measured |
| --- | --- |
| `breakages.py` replays one real capture three ways: a changing per-call note in the system prompt, a summarised history, one turn without log-probabilities | one Claude Code 2.1.290 run with LFM2.5 (8 calls, solved, 1 row, 763 trained tokens), replayed with its replies fixed token for token: a plain replay, Claude Code's own note left on, a summary from call 5, call 5 without log-probabilities. Each is a real Claude Code run through OpenEnv `3f14355`, graded at the end |
| prints how many rows each breakage splits the rollout into, and whether it stays trainable, while the task is still solved | 8 rows, 2 rows, 1 row; solved 4 of 4 |
| "splits rows or makes them untrainable" (Part 6) | none of the three makes the rollout untrainable: `trainable` is `true` in all four replays. The missing log-probabilities mask one call (702 of 763 tokens trained). Only a total loss gives `false` (chapter 21's broken engine), because `trainable` is `bool(trainable_rows) and rollout_report.ok` (`export.py` 214) |
| "the checks that catch them (the trainable flag, the harness's own log, the call counts)" | the trainable flag catches none of the three. Calls per row, Claude Code's one conversation against the proxy's rows, catches the note and the summary. Per-call trainable coverage, the tokens each call trained against the tokens Claude Code was told, catches the missing log-probabilities. Chapter 21's four checks, the first four here, say `ok` on both splits |
| the project's checks (`smoke.py`, `run_rollout`) | they pass all three. `smoke.py` reports roots (166) and fails a run only on `not (result.finished and result.trainable)` (194). `run_rollout` keeps rows `if seq.get("trainable")` (284) and records `trainable` and `n_trainable_tokens` (298-301) |
| `crosscheck.py` compares the proxy's rows with the harness's own log and catches each breakage | as planned: `MISMATCH` on calls per row (note, summary) and on trainable coverage (no log-probabilities), each with its cause; the capture and the plain replay agree on all five checks |
| a changing per-call note in the system prompt | Claude Code's note is a new `system` message after each tool result; it reaches the system prompt only through the proxy's merge (`dialects/base.py` 71, 85; chapter 21, finding 1). Measured at the engine: the system prompt grew at its end on 7 of 7 calls |
| a summarised history | applied by a relay: Claude Code 2.1.290 did not compact through the proxy when asked to (`compaction.py`: exit code 0, 0 `compact_boundary` events, prompts of about 10,000 tokens against a mark of about 1,000). Why is not measured; the guess is that it is told `input_tokens 0` |
| "its SDK adds a per-request billing line" (the concept list) | the line is the same on every call: `x-anthropic-billing-header: cc_version=2.1.290.212; cc_entrypoint=sdk-cli;` on all 82 calls of the 13 kept runs, with no per-request hash. The proxy strips it anyway (`dialects/anthropic.py` 27-28, 345): it reached the engine 0 times |
| (not in the plan) | Claude Code sends its environment block (`# Environment`) as a message with the role `system`, after the task; the proxy merges it into the system prompt. A summary that drops it changes the system prompt as well, so the relay keeps it |
| folder `course/22-silent-capture-failures/` | `22-silent-capture-failures/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its code, quoted by line.
- **The article's numbers**: there are none in this chapter.

## Findings about the proxy and the project

Line numbers are in OpenEnv `3f14355`, `openenv/core/harness/capture/`,
and in the project at `25074c0`. All of them, with the measurements, are in
the run log's `17-measure-findings.txt`:

```text
OpenEnv at ~/.cache/mhrl-course/22-silent-capture-failures/proxy-env/lib/python3.13/site-packages (the pinned commit); the project at ~/sai_work/multi-harness-rl
the rollout's one verdict: openenv/core/harness/capture/export.py
   214          "trainable": bool(trainable_rows) and rollout_report.ok,
the proxy drops the billing line: openenv/core/harness/capture/dialects/anthropic.py
    23  # Claude Code SDK leaks `x-anthropic-billing-header: ...cch=<hash>;` as the
    24  # first line of the system prompt. The cch= hash changes per request, so
    27  _CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    28      r"^\s*x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE
   343              # Drop Claude Code's per-request billing header line (breaks
   344              # prefix_merging because cch= changes every turn).
   345              system_content = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", system_content)
the proxy merges system messages: openenv/core/harness/capture/dialects/base.py
    71      def _merge_developer_role(cls, request: dict[str, Any]) -> dict[str, Any]:
    85          # Merge multiple system messages into one at the top
smoke.py reports roots: mhrl/harnesses/smoke.py
   166          roots=rollout.get("stats", {}).get("n_roots", 0),
   168          trainable=bool(rollout.get("trainable")),
smoke.py fails a run only when: mhrl/harnesses/smoke.py
   194                  failures += not (result.finished and result.trainable)
run_rollout keeps rows and the verdict: mhrl/rollout.py
   284          if seq.get("trainable")
   298          trainable=bool(document.get("trainable")),
   299          n_trainable_tokens=int(
   300              (document.get("stats") or {}).get("n_trainable_tokens") or 0
   301          ),

the billing line, measured over every kept run
  82 model calls in 13 runs; distinct first system lines:
     82 x x-anthropic-billing-header: cc_version=2.1.290.212; cc_entrypoint=sdk-cli;
  messages in the proxy's records that carry the billing line: 0

the environment block, in the model capture's first call
  sent message 1: role user
  sent message 2: role system  <- carries '# Environment'
  the engine's message 1: role system  <- carries '# Environment'
  the engine's message 2: role user
```

1. **No breakage here makes a rollout untrainable.** A split rollout's rows
   are each trainable, and a call without log-probabilities only masks
   itself. `trainable` turns `false` only when no row survives
   (`export.py` 214).
2. **The project reads only the verdict.** `run_rollout` keeps the trainable
   rows and stores `trainable` and `n_trainable_tokens` (`rollout.py` 284,
   298-301). `smoke.py` prints the root count (166) but fails a run only
   when it is unfinished or untrainable (194). All three breakages here
   pass both.
3. **The harness's log is the independent account.** It says one
   conversation of 8 calls, and the per-call token counts. Against it, a
   split shows as rows that start where no conversation starts, and an
   untrained call shows as tokens Claude Code received that no row trains.
4. **The billing line is constant, and stripped.** Claude Code 2.1.290 sends
   the same billing line first in its system prompt on every call. The
   proxy's comment expects a `cch=` hash that changes per request
   (`anthropic.py` 23-24); there is none here. The proxy drops the line
   anyway, so it never reaches the engine.
5. **Claude Code's environment block travels as a system message.** As
   sent, message 2 has the role `system` and carries `# Environment`. In the
   engine's prompt it is part of message 1, the system prompt.

## The files

Line counts from the run log: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `capture.py` | 419 | 269 | the run every case replays: solved and one row, or the best attempt, labelled |
| `breakages.py` | 692 | 409 | the four replays, their rows and verdicts, the closing table |
| `crosscheck.py` | 824 | 456 | the proxy's record against Claude Code's log: five checks, each disagreement's cause |
| `compaction.py` | 233 | 151 | Claude Code asked to compact early through the proxy |
| `captures.py` | 407 | 162 | the run folders, and each call's exact reply read back out of its row |
| `scripted_engine.py`, `scripted_replies.py` | 582, 285 | 250, 105 | chapter 21's scripted engine, now serving exact token ids and log-probabilities, with one token per character or LFM2.5's tokenizer |
| `claude_code.py` | 634 | 352 | the pinned Claude Code; its relay can now rewrite each request on its way |
| `start.sh` | 294 | 220 (not blank, not comments) | chapter 21's stack, with `replay` |
| `lfm_files.py` | 383 | 153 | chapter 21's, with LFM2.5's `tokenizer.json` found by its fingerprint and the home folder written `~` |
| `rollout_view.py` | 422 | 209 | chapter 21's, writing the home folder `~` |
| `engine.py`, `lfm.py`, `lfm_text.py`, `policy.py`, `toolcalls.py`, `grader.py`, `tasks.py`, `data/`, `capture_client.py`, `training_rows.py`, `sandbox.py`, `sandbox.sb` | | | chapter 21's, unchanged (`sandbox.sb` with this chapter's folder) |

## What this chapter leaves out

- **Claude Code's own compaction** never ran, so a relay plays it. A
  harness whose own settings size its compaction, Pi, comes in chapter 23.
- **The other harnesses** (mini-swe-agent, Pi) come through the proxy in
  chapter 23.

## Mirrors in the project

OpenEnv's capture graph (`graph.py`, `export.py`, `validate.py` at
`3f14355`), `mhrl/rollout.py` (`count_calls`, `run_rollout`) and
`mhrl/training/common.py` (`dedupe_shared_prefixes`, `item_from_sequence`).
