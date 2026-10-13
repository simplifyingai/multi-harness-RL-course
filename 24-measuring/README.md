# Chapter 24. Measuring fairly: baseline, pass@1 and noise

Chapter 23 ran one task per harness. That is a demonstration, not a
measurement. Before any training, this chapter fixes the test: 24 held-out
tasks per harness, one greedy attempt each, the same budget everywhere, and
Pi never trained in. Then it asks what 24 tasks can and cannot show.

The puzzle: the project's recording of the untrained model says
mini-swe-agent solved 24 of 24 held-out tasks, Claude Code 15 and Pi 4. Our
first measurement of the same model, in the same harnesses at the same
versions, said 24, 4 and 0. Which one is right, and how would you know?

## Words first

- **pass@1** is the share of tasks solved on the one attempt each gets.
- A **cell** is one (task, harness) pair. 24 tasks in 3 harnesses make 72
  cells, and each runs exactly once.
- The **evaluation protocol** is everything that stays fixed while only the
  adapter changes: the same tasks, the same instruction, temperature 0, at
  most 8 model calls, replies capped at 1,024 tokens, 300 s per rollout, and
  the pinned harness versions.
- A **balanced subset** is a set of tasks picked round-robin over database
  and difficulty: 24 tasks are 4 from each of the six (database, difficulty)
  buckets.
- The **macro average** is the plain mean of the per-harness scores, so each
  harness weighs the same.
- **Sampling noise** is how far a score moves by luck alone. The **standard
  error**, sqrt(p x (1 - p) / n), is its typical size; a **Wilson
  interval** is a 95% range for the true rate that stays sensible at 0% and
  100%.
- The **ceiling effect**: a score that starts at 100% can only stay or fall.
- A **paired comparison** reads two checkpoints cell by cell: **gained**
  (wrong before, right now), **lost** (the reverse), and the **exact p**, the
  chance of a split at least that uneven if the two were equally good
  (McNemar's exact test). **Churn** is many flips that cancel out.
- A **held-out harness** is one used only for testing. Here that is Pi.
- An **evaluation infrastructure bug** is a fault in the test, not the
  model, that looks exactly like a model failure.

## Run it

```bash
cd multi-harness-RL-course/24-measuring
uv run noise.py                          # no model, no stack: the project's records
uv run baseline_eval.py --plan           # the protocol and the 24 tasks
uv run install.py                        # the three pinned harnesses
./start.sh scripted                      # no model: the scripted engine, the proxy
uv run baseline_eval.py --limit 2        # every harness on 2 tasks, scripted
./start.sh stop
./start.sh                               # the model: the 4-bit LFM2.5
uv run baseline_eval.py --fast           # 4 tasks per harness (12 cells)
uv run baseline_eval.py                  # 24 tasks per harness (72 cells)
./start.sh stop
uv run noise.py                          # now with your baseline beside the project's
```

`start.sh` is chapter 23's, with one step added. Before an engine starts,
`fetch_model.py` checks for what that engine reads, and fetches it when it
is missing: the 4-bit model (Liquid's export, 1.6 GB) or, for the scripted
engine, only LFM2.5's tokenizer files (17.9 MB). `baseline_eval.py` installs
any harness that is missing, so the chapter rebuilds its own inputs instead
of sending you back to chapter 23. `--resume` finishes a run that was
stopped, and `--before-cell CMD` runs a command before each cell, so a long
run can wait while the machine does something else.

The course's ports are the project's: the engine on 8100 and the proxy on
8200. This run used `ENGINE_PORT=8129 PROXY_PORT=8209` (and
`--proxy-url http://127.0.0.1:8209`), because other programs on this Mac held
8100 and 8119.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
22.19 or later (for Claude Code and Pi) and the network for the installs.

## The run log

Every output below is copied from the run log, made from a clean
`COURSE_HOME/24-measuring` at commit `e2a101a`, everything under
`nice -n 10`. The model's engine started only after a minute with no film
rendering on the machine (the run waited 61 s). Before every model cell,
`--before-cell` ran a check that waited whenever another chapter's film
was rendering or transcribing. From 05:57 on, partway through the run, the
check also stopped the engine and the proxy during a wait and started them
again afterwards, so the laptop got the engine's memory back; that happened
at the last three of the six waits. Seconds are wall time. Memory is the
largest single process the command ran (`/usr/bin/time -l`); for
`baseline_eval.py` that is a harness, which runs as its child.

| command | what it shows | seconds and memory |
| --- | --- | --- |
| `uv run noise.py` | the project's records, read for noise | exit 0 · 0.1 s · max resident memory 0.03 GB |
| `uv run baseline_eval.py --plan` | the protocol and the 24 tasks | exit 0 · 0.1 s · max resident memory 0.03 GB |
| `uv run baseline_eval.py --plan --fast` | the 4 tasks of `--fast` | exit 0 · 0.1 s · max resident memory 0.03 GB |
| `uv run install.py` | the three harnesses at their pins | exit 0 · 41.8 s · max resident memory 0.30 GB |
| `./start.sh scripted` | the proxy installed, the scripted engine | exit 0 · 6.5 s · max resident memory 0.16 GB |
| `uv run baseline_eval.py --limit 2` | 6 cells, scripted: all solved | exit 0 · 22.9 s · max resident memory 0.25 GB |
| `uv run baseline_eval.py --limit 1 --script mistakes` | each harness's own stop rule | exit 0 · 8.1 s · max resident memory 0.25 GB |
| `uv run baseline_eval.py --limit 2` again | refused: the records exist | exit 1 · 0.1 s · max resident memory 0.03 GB |
| `uv run baseline_eval.py --limit 2 --resume` | nothing left to run | exit 0 · 0.5 s · max resident memory 0.12 GB |
| `./start.sh stop` | `engine stopped · 36 requests` | exit 0 · 2.1 s · max resident memory 0.00 GB |
| `uv run pytest tests/test_24_measuring.py` | 49 passed, against the real proxy | exit 0 · 32.2 s · max resident memory 0.29 GB |
| `./start.sh` | the model's engine (1.6 GB loaded) | exit 0 · 7.2 s · max resident memory 0.11 GB |
| `uv run baseline_eval.py --fast --label base-fast` | 12 cells | exit 0 · 140.2 s · max resident memory 0.25 GB |
| `uv run baseline_eval.py` | 72 cells, with 33.1 min of waits for another film | exit 0 · 3218.9 s · max resident memory 0.25 GB |
| `./start.sh stop` | `engine stopped · 45 requests · peak memory 2.90 GB` | exit 0 · 2.1 s · max resident memory 0.00 GB |
| `uv run noise.py` | your baseline beside the project's | exit 0 · 0.1 s · max resident memory 0.03 GB |

The engine's peak memory, weights included, was 3.24 GB until the first wait
that stopped it, then 3.19, 3.12 and 2.90 GB after each restart (the
restarts took 8, 7 and 7 s). The proxy held 0.11 GB (scripted) and 0.04 GB
(the model) just before each stop:

```text
pauses for other films: 6, 1987 s in all (33.1 min); engine restarts after a pause: 3, 8, 7, 7 s
the engine's peak memory per segment: 3.24, 3.19, 3.12, 2.90 GB
```

A first attempt at commit `bee89dd` is kept in the run log (`first-attempt/`).
Its scripted part matched this one; its model part is the subject of
"The baseline, run twice" below. Commit `49a5300`, after the run, answers a
code review of `evaluate.py` and `baseline_eval.py` (a misplaced answer's
record never shows your name, the harness's log is kept when a cell fails,
a state folder that cannot be set up is removed); no measured step changed.

## Which 24 tasks: `baseline_eval.py --plan`

```text
baseline_eval.py · pass@1 on held-out tasks, in every harness
  tasks     24 of 50 held out (airline, clinic), round-robin over database x difficulty: 6 buckets, 4 per bucket
  harnesses mini-swe-agent 2.4.6, claude-code 2.1.290, pi 1.0.4
  protocol  one attempt per cell · temperature 0 · 8 calls · replies capped at 1,024 tokens · 300 s per rollout
  cells     72, one at a time
  records   ~/.cache/mhrl-course/24-measuring/baseline/eval_base.jsonl (one line per rollout, as it ends)

which 24 tasks: balanced() against a plain prefix
                            airline              clinic
                            easy   medium hard   easy   medium hard
  balanced(tasks, 24)       4      4      4      4      4      4
  tasks[:24], file order    4      5      3      4      5      3
  tasks[:24], sorted by id  8      9      7      0      0      0
  all 50 held-out tasks     8      10     7      8      10     7
```

`balanced` takes the six (database, difficulty) buckets in turn, so 24
tasks are 4 from each, in the order the project recorded them
(`airline-easy-001`, `airline-hard-001`, `airline-medium-001`,
`clinic-easy-001`, ...). A plain `tasks[:24]` depends on the file's order.
The project's task file is already interleaved (its `load_tasks` says so,
`mhrl/tasks/core.py` line 222), so here a prefix gives 12 clinic and 12
airline tasks, 10 medium, 8 easy and 6 hard. Only a file sorted by name
gives one database, and even then it is all difficulties, not easy only.
`balanced` gives the same 24 whatever the order.

`--fast` stops after the first four picks:

```text
which 4 tasks: balanced() against a plain prefix
                           airline              clinic
                           easy   medium hard   easy   medium hard
  balanced(tasks, 4)       1      1      1      1      0      0
```

That is 3 airline tasks and 1 clinic task: four buckets of the six.

## The protocol, and where the 1,024 cap lives

Each cell opens an evaluation session at the proxy (temperature 0, at most
8 calls), runs the harness once inside chapter 19's sandbox, and grades the
task folder's `answer.txt`. The reply cap of 1,024 tokens is set in three
places, none of which the proxy's `/health` shows: `engine.py` lowers any
request's `max_tokens` to 1,024, `start.sh` starts the proxy with
`--max-output-tokens 1024`, and Claude Code gets
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=1024` in its environment (`wire.py`).

## No model first

With the scripted engine every harness solves the two tasks the way its
script ends, and each record is appended the moment its rollout ends:

```text
cells
[1/6] mini-swe-agent airline-easy-001   correct · 4 calls  ·   8.0 s · finish command · answer '8'
[2/6] mini-swe-agent airline-hard-001   correct · 4 calls  ·   4.5 s · finish command · answer 'Alba Barros'
[3/6] claude-code    airline-easy-001   correct · 4 calls  ·   2.6 s · no tool call · answer '8'
[4/6] claude-code    airline-hard-001   correct · 4 calls  ·   2.4 s · no tool call · answer 'Alba Barros'
[5/6] pi             airline-easy-001   correct · 4 calls  ·   2.4 s · no tool call · answer '8'
[6/6] pi             airline-hard-001   correct · 4 calls  ·   2.4 s · no tool call · answer 'Alba Barros'

checkpoint        mini-swe-agent     claude-code              pi         average
scripted-solve              100%            100%            100%            100%
calls per solve (solved cells only): mini-swe-agent 2/2, 4.0 · claude-code 2/2, 4.0 · pi 2/2, 4.0
cells     6 of 6 recorded · this run took 0 min 23 s
summary   ~/.cache/mhrl-course/24-measuring/baseline/summary_scripted-solve.json
network   every rollout: lsof saw only this machine's addresses
```

`--script mistakes` makes the untrained model's mistakes, and each harness
stops by its own rule (chapter 23):

```text
cells
[1/3] mini-swe-agent airline-easy-001   wrong   · 3 calls  ·   4.2 s · 3 bad replies · answer none
[2/3] claude-code    airline-easy-001   wrong   · 4 calls  ·   2.4 s · no tool call · answer none
[3/3] pi             airline-easy-001   wrong   · 1 call   ·   0.9 s · window full · answer none
```

A second run of the same label is refused, and `--resume` runs only the
cells still missing (here, none):

```text
~/.cache/mhrl-course/24-measuring/baseline/eval_scripted-solve.jsonl already holds 6 records: --resume to finish that run, or --label to start another
cells
[1/6] mini-swe-agent airline-easy-001: already recorded
[2/6] mini-swe-agent airline-hard-001: already recorded
```

## The baseline, run twice

### First attempt: 24, 4 and 0

```text
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             17%              0%             39%
calls per solve (solved cells only): mini-swe-agent 24/24, 7.8 · claude-code 4/24, 8.0 · pi 0/24, none solved
```

Claude Code solved 4 of 24, against the project's 15. Read cell by cell, 9
of its 20 failures were not failures of reasoning: the model found the
answer, wrote it, and wrote it in the wrong place.

```text
first attempt (bee89dd): Claude Code answers in its auto-memory folder
  (state beside the task folder; memory_paths.auto in each init event)
  airline-easy-002     answer '6'                                                  expected '6'
  airline-hard-001     answer 'Alba Barros'                                        expected 'Alba Barros'
  airline-hard-004     answer '255'                                                expected '255'
  airline-medium-004   answer '634.94'                                             expected '634.94'
  clinic-easy-002      answer '3'                                                  expected '3'
  clinic-easy-004      answer '181.5'                                              expected '181.50'
  clinic-hard-001      answer 'Grace Torres'                                       expected 'Grace Torres'
  clinic-hard-002      answer 'Paula Nakamura, Astrid Lindqvist, Gabriel Carter'   expected 'Astrid Lindqvist, Gabriel Carter, Paula Nakamura'
  clinic-medium-001    answer 'General Practice'                                   expected 'General Practice'
  9 cells; the folder: ~/.cache/mhrl-course/24-measuring/runs/eval/base/claude-code/<task>/state/claude/projects/<folder>/memory/answer.txt
```

Every one of the 9 is the expected answer (the grader accepts `181.5` for
`181.50` and a list in any order). The folder is Claude Code's auto-memory
folder. Claude Code names it in its own init event (`memory_paths.auto`)
and in its system prompt, as a folder it may write to. The run had put each
harness's private state folder, Claude Code's config included, beside the
task folder, chapter 23's layout. So the memory folder sat inside the task's
own tree, `.../<task>/state/claude/projects/...-task/memory/`, and looked
like the task's place. The project puts the state folder in the system's
temp folder instead: `STATE_PREFIX = "mhrl-harness-"` and
`tempfile.mkdtemp(prefix=f"{STATE_PREFIX}{self.name}-")`
(`mhrl/harnesses/base.py`, lines 43 and 241-243). Commit `e2a101a` does the
same for every cell (`prepare_cell` in `evaluate.py`), copies the harness's
own log back into the cell's folder, removes the state afterwards, and
records any `answer.txt` found outside the task folder (`answer_elsewhere`).

### The fixed run

```text
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             29%              0%             43%
calls per solve (solved cells only): mini-swe-agent 24/24, 7.8 · claude-code 7/24, 8.0 · pi 0/24, none solved
cells     72 of 72 recorded · this run took 53 min 39 s
summary   ~/.cache/mhrl-course/24-measuring/baseline/summary_base.json
network   every rollout: lsof saw only this machine's addresses
```

```text
fixed run (e2a101a): answers outside the task folder
  0 cells
the same 9 cells in the fixed run
  airline-easy-002 correct, airline-hard-001 wrong, airline-hard-004 wrong, airline-medium-004 wrong, clinic-easy-002 wrong, clinic-easy-004 wrong, clinic-hard-001 wrong, clinic-hard-002 wrong, clinic-medium-001 wrong
Claude Code, first attempt -> fixed run: gained 5 (clinic-easy-001, airline-easy-002, airline-medium-002, clinic-easy-003, clinic-hard-004), lost 2 (clinic-medium-002, airline-easy-003)
Claude Code's first call, prompt tokens: first attempt 2,302-2,317, fixed run 2,305-2,322
```

No answer landed outside a task folder this time. But Claude Code went from
4 to 7, not to 13: only 1 of the 9 cells became correct, 5 cells were gained
and 2 were lost, and `noise.py` reads that split as within noise:

```text
  your first-attempt  24/24 = 100.0%    4/24 = 16.7%      0/24 = 0.0%       38.9%
  your base           24/24 = 100.0%    7/24 = 29.2%      0/24 = 0.0%       43.1%
  your first-attempt -> your base  mini-swe-agent     24      24     24       0     0        0  1.000  nothing changed
  your first-attempt -> your base  claude-code        24       4      7       5     2        7  0.453  within noise
  your first-attempt -> your base  pi                 24       0      0       0     0        0  1.000  nothing changed
```

The fix changed one path in Claude Code's prompt (its first call grew from
2,302-2,317 tokens to 2,305-2,322), and greedy decoding, which follows the
likeliest token from the first one that changed, took another path in 7 of
24 cells. The misplaced answers were a real defect. Counting them as solved
would have been a second one.

What ended the 41 wrong cells, and where the sandbox stepped in:

```text
fixed run: cells where the box denied a write outside the task folder
  claude-code  3: airline-hard-004, airline-medium-004, clinic-medium-001
  pi           3: airline-easy-002, airline-easy-004, airline-medium-002
fixed run: what ended each wrong cell
  claude-code  8-call budget  17
  pi           8-call budget  2
  pi           no tool call   1
  pi           window full    21
```

```text
fixed run: the errors Claude Code sent the model, over its 24 cells
   53  Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
    9  Exit code 1
    4  Error: No such tool available: Write. Write is disabled for this session, in subagents as well a
    4  Error: No such tool available: write
    4  Error: No such tool available: read
    1  InputValidationError
fixed run: Pi's own log in its window-full cells
   20  agent_end reached · tools run: read
    1  agent_end reached · tools run: bash, read
```

- **Claude Code** spent its 8 calls in all 17 wrong cells: tool names in the
  wrong case (`bash` for `Bash`, 53 times), tools it does not have (`write`,
  `Write`, `read`), and, in 3 cells, an answer written to the real home
  folder, which the box refused.
- **Pi** used its `read` tool in all 21 window-full cells: it reads `data.db`
  whole, and the next prompt overflows the 16,384-token window (chapter 23).
  In 3 of its longer cells it found the right number, then wrote
  `answer.txt` to the real home folder, and the sandbox denied it. The
  model's mistake, and the box doing its job.

### The same 4 cells, twice

`--fast` and the full run share their first four cells per harness, with the
same code and the same layout. Only the cell's folder differs
(`runs/eval/base-fast/...` against `runs/eval/base/...`), and Claude Code's
prompt carries it:

```text
checkpoint    mini-swe-agent     claude-code              pi         average
base-fast               100%            100%              0%             67%
calls per solve (solved cells only): mini-swe-agent 4/4, 7.5 · claude-code 4/4, 8.0 · pi 0/4, none solved
  your base-fast -> your base (2)  claude-code         4       4      2       0     2        2  0.500  within noise
```

Claude Code solved all four in the fast run and two of the same four in the
full run.

## How much of a score is noise: `noise.py`

`noise.py` reads the project's own records, bundled in
`data/project_records/` (its run of 2026-10-06, Pi from its re-run after
chapter 19's sandbox fix), and yours.

```text
━━ the baseline (the project's base model)
  checkpoint  mini-swe-agent    claude-code       pi                average
  base        24/24 = 100.0%    15/24 = 62.5%     4/24 = 16.7%      59.7%
  base's average is the plain mean over harnesses, each weighing the same:
  (100.0 + 62.5 + 16.7) / 3 = 59.7%

━━ one task's weight
  24 tasks per harness: one task = 100 / 24 = 4.2 points

━━ standard error and Wilson 95% interval, n = 24 (base)
  standard error = sqrt(p x (1 - p) / n); near 75%: sqrt(0.75 x 0.25 / 24) = 8.8 points
  harness         solved  pass@1  standard error  Wilson 95% interval
  mini-swe-agent  24/24   100.0%  ±  0.0 points     86.2% to 100.0%
  claude-code     15/24    62.5%  ±  9.9 points     42.7% to  78.8%
  pi               4/24    16.7%  ±  7.6 points      6.7% to  35.9%
  a perfect score has a standard error of 0, as if 100% were certain; Wilson's
  interval at 24/24 still allows a true rate down to 86.2%
```

With 24 tasks one task moves a score 4.2 points, and near 75% the typical
wobble is 8.8 points either way. At 24/24 the standard error is 0, as if 100%
were certain; Wilson's interval still allows 86.2%.

```text
━━ two identical models (simulated, seed 0)
  both solve each task with chance 75%; 24 tasks each; 10,000 pairs of runs
  gap in tasks         0       1       2       3      4+
  simulated        13.7%   24.9%   21.4%   16.3%   23.7%
  exact            13.3%   25.1%   21.3%   16.1%   24.2%
  they differ by 8 points (2 tasks) or more in 61.3% of pairs (exact: 61.6%)
  one run with one seed is one draw from this table: a gap of a task or two
  between checkpoints is what identical models show most of the time
```

Two models with exactly the same skill, 75% on each task, scored 24 tasks
each, differ by 2 tasks or more in about 6 pairs out of 10. The simulation
is seeded, so it prints the same table every time, and the exact
calculation beside it agrees with the draw.

```text
━━ the ceiling at 100%
  mini-swe-agent solved 24/24 at base: from there a later checkpoint can only tie or lose
  later: sft 18/24 = 75.0%, rl 20/24 = 83.3%; a gain there could never have been measured
  chance of 24/24 if the true rate were 99%: 78.6% · 95%: 29.2% · 90%: 8.0%
  so 24/24 says the rate is above about 86% (Wilson), not that it is 100%

━━ the average hides what moved
  base -> sft: average 59.7% -> 61.1% (+1.4 points), while
  mini-swe-agent -25.0, claude-code +8.3, pi +20.8 points
```

```text
━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair         harness         cells  before  after  gained  lost  flipped  p      reading
  base -> sft  mini-swe-agent     24      24     18       0     6        6  0.031  lost: beyond noise
  base -> sft  claude-code        24      15     17       7     5       12  0.774  within noise
  base -> sft  pi                 24       4      9       7     2        9  0.180  within noise
  base -> rl   mini-swe-agent     24      24     20       0     4        4  0.125  within noise
  base -> rl   claude-code        24      15     18       7     4       11  0.549  within noise
  base -> rl   pi                 24       4      9       8     3       11  0.227  within noise
  sft -> rl    mini-swe-agent     24      18     20       2     0        2  0.500  within noise
  sft -> rl    claude-code        24      17     18       2     1        3  1.000  within noise
  sft -> rl    pi                 24       9      9       5     5       10  1.000  within noise
  p < 0.05 reads as beyond noise; anything else, within it
  churn: claude-code base -> sft: 12 of 24 cells flipped, 10 of them cancel out: score +2
  churn: claude-code base -> rl: 11 of 24 cells flipped, 8 of them cancel out: score +3
  churn: pi sft -> rl: 10 of 24 cells flipped, 10 of them cancel out: score +0
```

- Of the nine comparisons between the project's checkpoints, one is beyond
  noise: SFT cost mini-swe-agent 6 tasks and gained none (p = 0.031).
- Claude Code's base-to-SFT score moved by 2 tasks while 12 of its 24 cells
  flipped; Pi's SFT-to-RL score did not move while 10 flipped. A near-flat
  score can hide a large reshuffle.

```text
━━ when the evaluation itself is broken (chapter 19's sandbox)
  pi, from the project's records, before and after the sandbox fix
  base   3/24 ->  4/24
  sft    0/24 ->  9/24
  rl     0/24 ->  9/24
  before the fix, 69 of 69 failed pi cells say 'no answer.txt':
  the first box broke Pi's write tool (chapter 19), so the 0% after SFT and
  RL measured the box, not the model
```

## Our baseline beside the project's

```text
━━ your runs beside the project's base
  checkpoint  mini-swe-agent    claude-code       pi                average
  base        24/24 = 100.0%    15/24 = 62.5%     4/24 = 16.7%      59.7%
  your base   24/24 = 100.0%    7/24 = 29.2%      0/24 = 0.0%       43.1%

━━ standard error and Wilson 95% interval, n = 24 (your base)
  harness         solved  pass@1  standard error  Wilson 95% interval
  mini-swe-agent  24/24   100.0%  ±  0.0 points     86.2% to 100.0%
  claude-code      7/24    29.2%  ±  9.3 points     14.9% to  49.2%
  pi               0/24     0.0%  ±  0.0 points      0.0% to  13.8%
  base -> your base  mini-swe-agent     24      24     24       0     0        0  1.000  nothing changed
  base -> your base  claude-code        24      15      7       2    10       12  0.039  lost: beyond noise
  base -> your base  pi                 24       4      0       0     4        4  0.125  within noise
```

- **mini-swe-agent**: 24/24 in both, the same 24 cells.
- **Claude Code**: 15 in the project's recording, 7 here; 2 gained and 10
  lost, p = 0.039. Same weights, harness version, flags and tasks, so the
  difference is beyond what one setup's noise explains: the two setups
  differ. Claude Code's prompt carries its folder and the date (chapter 23),
  the engine here is the course's copy, and the project ran four cells at a
  time where this run runs one. This chapter cannot tell which of these
  costs Claude Code those tasks. So the chapters that follow compare
  checkpoints measured by the same code on the same machine, never against
  the project's recording.
- **Pi**: 4 in the recording, 0 here; 0 gained, 4 lost, p = 0.125, within
  noise.

## A score names its harness: the article's numbers

```text
━━ a score names its harness: the article's numbers, kept apart
  ┌─ the article's own numbers (Hugging Face + Liquid AI) ─────────────────────┐
  │ from "The ultimate guide to multi-harness RL" (2026)                       │
  │ On its own test tasks, with the same LFM2.5-2.6B weights, the same         │
  │ direction of gap: mini-swe-agent ahead of Claude Code by 29 points.        │
  │ Its benchmark example, measured by Joel Niklaus: one model, GLM-5.2, on    │
  │ SWE-bench Pro, at 23% in one harness and 52% in another.                   │
  └────────────────────────────────────────────────────────────────────────────┘
  ours (the project's run of 2026-10-06, base): mini-swe-agent ahead of claude-code by 37.5 points
```

Both figures were checked against the article itself; its sentences are
quoted in the table at the end. Its harness gap is its own LFM2.5-2.6B on its
own test tasks, so `noise.py` gives it only as a difference, 29 points, and
none of its scores sits beside ours. The same direction holds here, 24/24
against 7/24. The GLM-5.2 figures are Joel Niklaus's measurement, which the
article cites without naming the two harnesses.

## Plan versus measured

| the plan said | measured |
| --- | --- |
| baseline_eval.py runs the project's evaluate() on the base model, round-robin over database x difficulty, all three harnesses, temperature 0, 8 calls, 1,024-token cap, one attempt per cell, every rollout appended to a JSONL file | as planned. Each record is appended and flushed to disk as its cell ends; the project's `evaluate` appends all of them after its pool finishes (`mhrl/evaluate.py` lines 98-100), running four cells at a time (`mhrl/config.py` line 82). This run runs one at a time |
| reuse chapter 23's run_one.py | its steps are reused, its layout is not: chapter 23 keeps the state folder beside the task folder, which put Claude Code's auto-memory folder inside the task's tree and drew 9 right answers into it in the first attempt. Each cell's state now goes in the system's temp folder, as in the project |
| the recorded run: mini-swe-agent 24/24, Claude Code 15/24, Pi 4/24; average 60% = (100 + 62 + 17) / 3 | the project's records say exactly that (59.7%). Pi's 4/24 is its re-run after chapter 19's sandbox fix; the original base evaluation had 3/24. This run: 24/24, 7/24, 0/24, average 43.1%; the first attempt 24/24, 4/24, 0/24 |
| 'one task = 4.2 points; standard error near 75% = 8.8 points' | as planned |
| an interval per harness | Wilson 95%: the project's 86.2-100%, 42.7-78.8%, 6.7-35.9%; this run's 86.2-100%, 14.9-49.2%, 0.0-13.8% |
| how often two identical simulated models differ by 8 points or more | 61.3% of 10,000 simulated pairs (seed 0), 61.6% exactly, at 75% |
| why a plain tasks[:24] would be one database's easy questions | it would not, here: the project's file is interleaved, so `tasks[:24]` is 12 clinic + 12 airline (10 medium, 8 easy, 6 hard). Only a file sorted by name gives one database (24 airline: 8 easy, 9 medium, 7 hard), and even then not easy only. `cli.balanced`'s docstring says otherwise (`mhrl/cli.py` line 208) |
| why mini-swe-agent's 100% can only stay or fall | as planned: its later 18/24 and 20/24 can only read as losses; base to SFT, 6 lost and none gained, p = 0.031, the one change beyond noise |
| why one run with one seed proves little | measured three ways: the same 4 cells flipped 2 of 4 for Claude Code between `--fast` and the full run; the layout fix flipped 7 of 24 (5 gained, 2 lost, p = 0.453); the project's recording and this run differ in 12 Claude Code cells (2 gained, 10 lost, p = 0.039) |
| the broken-evaluation story: Pi's 0% after SFT and RL was chapter 19's sandbox bug | as recorded: 69 of 69 failed Pi cells said "no answer.txt" before the fix; 3/0/0 became 4/9/9. And one of our own: the first attempt's 9 misplaced answers |
| the article's 62% under mini-swe-agent and 33% under Claude Code | in the article, word for word ("With the same weights, LFM2.5-2.6B solved 62% of our test tasks under Mini-SWE-Agent and 33% under Claude Code."), on its own tasks; shown as a 29-point gap in its own box |
| GLM-5.2 at 23% and 52% on SWE-bench Pro in two harnesses | in the article ("On SWE-bench Pro, Joel Niklaus measured the same model, GLM-5.2, at 23% in one harness and 52% in another."); it names neither harness |
| fast mode: 4 tasks per harness (about 15 minutes) | 12 cells in 140.2 s (206.0 s in the first attempt), not 15 minutes. `--fast` picks 3 airline tasks and 1 clinic |
| full: 24 per harness | 72 cells in 1,302.2 s in the first attempt, with no waits; 3,218.9 s in this run, 33.1 min of it waiting for another chapter's film |
| the 1,024-token reply cap | it lives in the engine (`max_tokens` lowered to 1,024), the proxy's `--max-output-tokens` flag and Claude Code's `CLAUDE_CODE_MAX_OUTPUT_TOKENS`; the proxy's `/health` does not show it |
| make every consuming chapter rebuild or fetch its input when it is missing | `baseline_eval.py` installs a missing harness; `start.sh` runs `fetch_model.py`, which fetches the model or its tokenizer when missing, checked by SHA-256 |
| (from chapter 25's notes) noise.py prints gained / lost / p for every pair of checkpoints | as planned, with a churn line where most flips cancel out |
| folder `course/24-measuring/` | `24-measuring/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its evaluation records from
  its run of 2026-10-06 (`data/project_records/`).
- **The article's numbers** are its harness gap on its own tasks (29
  points) and the GLM-5.2 figures it cites, kept in their own box; its
  sentences are quoted in the table above.

## The files

Line counts at commit `49a5300`: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `baseline_eval.py` | 489 | 278 | the protocol, the task picks, the harness checks, the run, the table |
| `evaluate.py` | 635 | 291 | the project's `balanced`, `summarize`, `format_table` and `evaluate`; one cell from chapter 23's steps, its state in the temp folder |
| `noise.py` | 695 | 383 | one task's weight, intervals, identical models, the ceiling, the macro average, every pair, the broken evaluation, the article's box |
| `stats.py` | 225 | 75 | chapter 14's `stats.py`, plus the Wilson interval and the identical-models arithmetic |
| `fetch_model.py` | 287 | 164 | fetches the model or its tokenizer when missing, checked by SHA-256 |
| `start.sh` | 306 | | chapter 23's stack, plus the fetch step |
| `data/project_records/` | | | the project's evaluation records, cut to the fields `noise.py` reads |
| `run_one.py`, `install.py`, `wire.py`, `loopback.py`, `scripted_engine.py`, `scripted_replies.py`, `engine.py`, `sandbox.py`, `sandbox.sb` and the rest | | | chapter 23's, unchanged but for this chapter's folder name |

## What this chapter leaves out

- **More tasks or more seeds.** With 24 tasks, only large changes can be
  told from noise. Chapter 29 sizes a test for the gain you hope to see.
- **Training.** Chapters 25 and 26 train through the harnesses and compare
  each checkpoint with this baseline, cell by cell.

## Mirrors in the project

`mhrl/evaluate.py` (`evaluate`, `summarize`) and `mhrl/cli.py` (`balanced`,
`eval`).
