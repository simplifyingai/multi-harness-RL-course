# Chapter 28. Capstone: run the whole pipeline, read the results honestly

Twenty-seven chapters built every piece: the grader, the policy gradient,
the engine that keeps the receipts, the sandbox, the capture proxy, three
real harnesses, a fair test, SFT through the harnesses and GRPO through
them. This chapter runs the project's whole pipeline once, start to finish,
from a fresh clone of the project to its final table, on one laptop. Then
it reads that table the way chapter 24 taught: task by task, with the
noise beside every number.

The puzzle: the project's README reports base 60%, SFT 61% and RL 65%,
calls per solve 7.1 to 4.8 to 4.3, and a held-out harness that rose from
17% to 38%. Which of those would a second run of the same code reproduce,
and which of them did 24 tasks per harness ever show?

## Words first

- The **pipeline** is the project's `scripts/train_from_scratch.sh`: the
  base model evaluated in every harness, answer-key demonstrations captured
  through the two training harnesses, SFT on them, the SFT model evaluated,
  GRPO from the SFT adapter, the RL model evaluated, and the report.
- A **fresh clone** is a new copy of the project's repository with nothing
  in it that the repository does not hold: no environment, no model, no
  runs. Everything is rebuilt from the repository's own files.
- A **stage** is one banner of the pipeline (`=== 1/5 baseline ... ===`),
  and the **stage clock** is the time printed under each banner, so a
  stage's **wall time** is the gap to the next banner.
- **pass@1**, a **cell**, the **macro average**, a **Wilson interval** and
  a **paired comparison** (gained, lost, the exact p) are chapter 24's.
  **Beyond noise** means p < 0.05; anything else is **within noise**, and a
  gain within noise is **suggestive, not shown**.
- **Calls per solve** is the mean number of model calls on the cells a
  checkpoint solved, per harness; the **average over harnesses** is the
  plain mean of the three. Each checkpoint counts only its own solves, so
  the sets differ; the **paired** reading counts only the cells two
  checkpoints both solved, and asks on how many the later one used fewer
  calls (the same exact test as gained and lost).
- The **harness gap** is two harnesses on the same tasks with the same
  weights, paired the same way.
- **MLX's peak** is the largest memory MLX allocated in a process
  (`mx.get_peak_memory`), weights included, as the trainers record it. A
  process's **footprint** is what macOS counts for it (top's MEM column),
  which includes the GPU memory MLX allocates; `ps`'s resident size does
  not.
- A **network check** lists every connection of the servers and the
  harnesses whose other end is not this machine.
- **Reproduction** means the same code, run again, gives numbers inside each
  other's noise; **identical** numbers are not expected (chapter 24: one
  task is 4.2 points, and greedy decoding takes another path when a prompt
  changes by a date or a folder name).

## Run it

```bash
cd multi-harness-RL-course/28-capstone
uv run read_results.py                 # the project's recorded run, read honestly (no model)
./capstone.sh plan                     # every command the capstone runs; runs none
FAST=1 ./capstone.sh all               # fresh clone, setup, tests, smoke, the pipeline at 6/10/2
./capstone.sh start                    # FULL, in the same clone: the servers,
./capstone.sh pipeline                 #   the pipeline at 24/60/6 (run "capstone"),
./capstone.sh stop                     #   and only this chapter's servers stopped
uv run read_results.py --run-dir ~/.cache/mhrl-course/28-capstone/multi-harness-rl/runs/capstone --against
```

`capstone.sh all` runs, in order: `git clone` of the project into
`COURSE_HOME/28-capstone/`, `uv sync`, `scripts/setup.sh`, Claude Code 2.1.290
and Pi 1.0.4 into the clone's `tools/` (where the project looks for them),
`python -m mhrl.tasks.generate`, the project's `pytest`, the three servers,
`python -m mhrl.harnesses.smoke`, and the pipeline. Every command is printed
before it runs, and `plan` prints them all without running any.

The pipeline is `scripts/train_from_scratch.sh`'s, command for command, with
one flag added: that script collects demonstrations from the teacher on port
8101 (`mhrl.cli demos` without `--teacher-port`), and this chapter's teacher
is on 8159. A test parses the project's script and checks that the two lists
of commands stay the same. For the same reason the servers start with
chapter 27's `stack.sh`, never with `scripts/serve.sh`, whose stop kills every
engine on the machine. The ports are 8149 (engine), 8159 (teacher) and 8219
(proxy); the project's are 8100, 8101 and 8200.

`GRPO_STEPS` is 6 unless `FAST=1`, set explicitly: the project's recorded run
trained 6 steps, while `train_from_scratch.sh` defaults to 16 and the CLI to
20. `FAST=1` sets `EVAL_LIMIT=6 DEMO_LIMIT=10 GRPO_STEPS=2`. Every stage loads
`MHRL_MODEL_PATH=$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours`, chapter 11's
build, and `setup` checks its weights against Liquid's export byte for byte
first. While the pipeline runs, `capstone.sh` prints `mhrl.cli steps` after
every GRPO step, samples every process's memory into `memory.tsv`, watches
the network during the RL evaluation into `network_check.txt`, and keeps the
stage banners in `stages.txt`, all in `runs/<name>/`.

uv and npm run with settings of their own (`COURSE_HOME/28-capstone/
installers/`), so your `uv.toml`, `~/.netrc`, `~/.npmrc`, git configuration
and Hugging Face token are never read.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
22.19 or later, the network for the installs, and chapter 11's two model
folders in `COURSE_HOME/models/` (without Liquid's export, `setup.sh`
fetches it, 1.5 GB).

## The run log

Every output below is copied from the run log, made at commits `0dd7f59`
(FAST) and `543f234` (FULL; the same `capstone.sh`, plus `read_results.py
--against`) from a clean `COURSE_HOME/28-capstone`, on an M3 Pro with 18 GB.
Both runs went through the machine's render lock, so no film rendered and no
other model ran beside them, and both ran at the lowest priority (`nice -n
15`). FULL waited first
for a quiet machine: nobody at the keyboard for 10 minutes, nothing rendering
for 120 s, normal memory pressure, at least 40% free. During both, free
memory was read every 5 s, ready to stop the run under 15% during GRPO; the
lowest it saw was 30% (FAST) and 31% (FULL).

"Memory" in the table is the largest single process `/usr/bin/time -l` saw.
It does not count the GPU memory MLX allocates for the model: MLX's own peak
is in the trainers' metrics and the processes' footprint in `memory.tsv`,
and those are the figures quoted below.

| step | command | what it shows | exit · seconds · memory |
| --- | --- | --- | --- |
| 00 | `uv run read_results.py` | the project's recorded run, read | exit 0 · not timed |
| 01 | `FAST=1 ./capstone.sh all` | the fresh clone, setup, tests, smoke and the pipeline at 6/10/2 | exit 0 · 1,681.8 s · 1.88 GB |
| 02 | `./capstone.sh start` and `RUN=capstone ./capstone.sh pipeline` | FULL: the pipeline at 24/60/6 in the same clone | exit 0 · 4,533.5 s (the pipeline) · 1.47 GB |
| 03 | `uv run read_results.py --run-dir .../runs/capstone-fast --against` (and `capstone`) | each fresh run beside the recorded one | exit 0 · not timed |

The driver checks beside the run log, not README commands: `article.py`
(the article's sentences, `00-article.txt`) and `cost.py` (what each call
re-sends, `04-*-cost.txt`).

## The project's run, read honestly: `read_results.py`

With no arguments, `read_results.py` reads the project's own recorded run,
bundled in `data/project_run/` (its evaluation records cut to the fields read
here, its two trainers' metrics, its network check and the stage banners of
its log). It needs no model and prints where every file came from:

```text
read_results.py · the final table, read honestly
  run         the project's run of 2026-10-06 (data/project_run/)
              eval_*.jsonl, sft/ and grpo/metrics.jsonl and network_check.txt
              are cut from the project's runs/main/; stages.txt is the stage
              banners of its runs/logs/session_main.log, whose GRPO stage is
              attempt 1 (stopped by hand after step 8). runs/main/grpo is
              attempt 2, so its clock is grpo/metrics.jsonl's own.
              network_check.txt is one look, at 08:42: by the project's file
              times, during RL attempt 1's evaluation, not the final one.
  harnesses   mini-swe-agent, claude-code, pi (never trained in)
  checkpoints base, sft, rl
  also here   rl_attempt1 (evaluated, not in the table)
  note        pi for base, sft, rl from eval_<label>-pi-fixed.jsonl: the
              re-run after chapter 19's sandbox fix (as the project's report)
```

The final table, with the interval chapter 24 taught beside every score:

```text
━━ the final table: pass@1 with a Wilson 95% interval (24 cells per harness)
  checkpoint  mini-swe-agent       claude-code          pi                   average
  base        24/24 100% (86-100)  15/24 62% (43-79)    4/24 17% (7-36)      60%
  sft         18/24 75% (55-88)    17/24 71% (51-85)    9/24 38% (21-57)     61%
  rl          20/24 83% (64-93)    18/24 75% (55-88)    9/24 38% (21-57)     65%
  average = the plain mean over harnesses, each weighing the same; (low-high) = Wilson 95%
  at 24 tasks per harness, one task = 4.2 points
```

`checkpoint` rows match the project's README, and take Pi from its re-run
after chapter 19's sandbox fix, as the project's report does. The table that
`train_from_scratch.sh` itself printed, before that re-run, is in its log
(`runs/logs/take4_report.log`): Pi 12%, 0% and 0%, and averages 58, 49 and
53%.

Cell by cell, the way chapters 24 and 25 read every pair:

```text
━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair         harness        cells  before  after  gained  lost  flipped  p      reading
  base -> sft  mini-swe-agent    24      24     18       0     6        6  0.031  lost: beyond noise
  base -> sft  claude-code       24      15     17       7     5       12  0.774  within noise
  base -> sft  pi                24       4      9       7     2        9  0.180  within noise
  base -> rl   mini-swe-agent    24      24     20       0     4        4  0.125  within noise
  base -> rl   claude-code       24      15     18       7     4       11  0.549  within noise
  base -> rl   pi                24       4      9       8     3       11  0.227  within noise
  sft -> rl    mini-swe-agent    24      18     20       2     0        2  0.500  within noise
  sft -> rl    claude-code       24      17     18       2     1        3  1.000  within noise
  sft -> rl    pi                24       9      9       5     5       10  1.000  within noise
  p < 0.05 reads as beyond noise; anything else, within it
  churn: claude-code base -> sft: 12 of 24 cells flipped, 10 of them cancel out: score +2
  churn: claude-code base -> rl: 11 of 24 cells flipped, 8 of them cancel out: score +3
  churn: pi sft -> rl: 10 of 24 cells flipped, 10 of them cancel out: score +0
```

And calls per solve, both ways: per checkpoint on its own solves, then on
the cells two checkpoints both solved:

```text
━━ calls per solve: model calls on solved cells only
  checkpoint  mini-swe-agent claude-code    pi             average over harnesses
  base        7.8            7.7            5.8            7.1
  sft         4.7            4.9            4.8            4.8
  rl          4.8            4.1            4.1            4.3
  by stage (average over harnesses): sft -2.3, rl -0.5 calls
  each checkpoint counts only the cells IT solved, so the sets differ; the pairs below
  count only cells both checkpoints solved

  pair         harness        both  mean before -> after  median  fewer  more  same  p
  base -> sft  mini-swe-agent   18  7.8 -> 4.7           8->4       17     0     1  0.000
  base -> sft  claude-code      10  7.7 -> 5.1           8->5       10     0     0  0.002
  base -> sft  pi                2  6.0 -> 5.0           6->5        2     0     0  0.500
  base -> rl   mini-swe-agent   20  7.8 -> 4.8           8->4       19     0     1  0.000
  base -> rl   claude-code      11  7.7 -> 4.1           8->4       11     0     0  0.001
  base -> rl   pi                1  5.0 -> 4.0           5->4        1     0     0  1.000
  sft -> rl    mini-swe-agent   18  4.7 -> 4.6           4->4        4     4    10  1.000
  sft -> rl    claude-code      16  4.9 -> 4.1           5->4       11     0     5  0.001
  sft -> rl    pi                4  4.5 -> 4.0           4.5->4      2     0     2  0.500
  p: the exact sign test on fewer against more (same dropped), as for gained and lost
  base -> rl, every harness pooled: 32 cells solved by both, 7.7 -> 4.5 calls (41% fewer)
```

The harness gap, the same weights on the same tasks:

```text
━━ the harness gap: the same weights, the same tasks, two harnesses
  checkpoint  first          second         tasks  only first  only second  p      reading
  base        mini-swe-agent claude-code       24           9            0  0.004  beyond noise
  base        mini-swe-agent pi                24          20            0  0.000  beyond noise
  base        claude-code    pi                24          14            3  0.013  beyond noise
  sft         mini-swe-agent claude-code       24           3            2  1.000  within noise
  sft         mini-swe-agent pi                24           9            0  0.004  beyond noise
  sft         claude-code    pi                24           9            1  0.021  beyond noise
  rl          mini-swe-agent claude-code       24           2            0  0.500  within noise
  rl          mini-swe-agent pi                24          11            0  0.001  beyond noise
  rl          claude-code    pi                24          10            1  0.012  beyond noise
```

What it says, worded by the script from those tables:

```text
━━ the reading, worded from the tables above
  the harness gap, beyond noise:
    base: mini-swe-agent ahead of claude-code, 9 tasks to 0 (p 0.004)
    base: mini-swe-agent ahead of pi, 20 tasks to 0 (p 0.000)
    base: claude-code ahead of pi, 14 tasks to 3 (p 0.013)
    sft: mini-swe-agent ahead of pi, 9 tasks to 0 (p 0.004)
    sft: claude-code ahead of pi, 9 tasks to 1 (p 0.021)
    rl: mini-swe-agent ahead of pi, 11 tasks to 0 (p 0.001)
    rl: claude-code ahead of pi, 10 tasks to 1 (p 0.012)
  checkpoints, beyond noise (p < 0.05):
    base -> sft, mini-swe-agent: 24 -> 18 solved, 0 gained, 6 lost (p 0.031)
  checkpoints, within noise:
    base -> sft, claude-code: +2 (7 gained, 5 lost, p 0.774)
    base -> sft, pi: +5 (7 gained, 2 lost, p 0.180)
    base -> rl, mini-swe-agent: -4 (0 gained, 4 lost, p 0.125)
    base -> rl, claude-code: +3 (7 gained, 4 lost, p 0.549)
    base -> rl, pi: +5 (8 gained, 3 lost, p 0.227)
    sft -> rl, mini-swe-agent: +2 (2 gained, 0 lost, p 0.500)
    sft -> rl, claude-code: +1 (2 gained, 1 lost, p 1.000)
    sft -> rl, pi: +0 (5 gained, 5 lost, p 1.000)
  the average, 60% -> 61% -> 65%: one seed, 24 tasks per harness, read only through the pairs above
  calls, base -> sft, every harness pooled: fewer on 29 of 30 shared solves, more on 0 (p 0.000): beyond noise
  calls, sft -> rl, every harness pooled: fewer on 17 of 38 shared solves, more on 4 (p 0.007): beyond noise
  pi, never trained in: 4 -> 9 solved, base -> rl (8 gained, 3 lost, p 0.227): suggestive, not shown
```

So, at 24 tasks per harness and one seed:

- **The harness gap is real.** mini-swe-agent solved 9 tasks Claude Code
  missed and Claude Code none that mini-swe-agent missed (p 0.004); Claude
  Code over Pi 14 to 3 (p 0.013). After SFT the mini-swe-agent and Claude
  Code gap is within noise (3 to 2, then 2 to 0); Pi stays behind both.
- **SFT made solves cheaper, beyond noise**: fewer calls on 29 of the 30
  cells both base and SFT solved (p < 0.001), and RL a little more (fewer on
  17 of 38, more on 4, p 0.007). By stage, SFT -2.3 calls and RL -0.5.
- **SFT cost mini-swe-agent 6 tasks** (0 gained, 6 lost, p 0.031), the only
  checkpoint change beyond noise. Every other pair is within noise: Claude
  Code's +2 after SFT is 12 of 24 cells flipping, 10 of them cancelling out.
- **Pi, never trained in, went 4 to 9** (8 gained, 3 lost by RL, p 0.227):
  suggestive, not shown. **60 -> 61 -> 65** is read only through these
  pairs.

The recorded run's clocks and its network check:

```text
━━ wall time
  the stage clock, banner to banner (data/project_run/stages.txt):
  1/5 baseline: the base model in every harness               21 min 50 s  07:31:00 -> 07:52:50
  2/5 answer-key demonstrations, then supervised fine-tuning  10 min 26 s  07:52:50 -> 08:03:16
  3/5 evaluate the SFT model                                   7 min 39 s  08:03:16 -> 08:10:55
  4/5 GRPO through the training harnesses                    no later banner: the clock ends here
  the records' own clocks:
  sft    15 steps, 9 min 08 s of training steps (the model's loading not counted)
  grpo   6 steps, 12 min 34 s (8 min 18 s of it rollouts)
  eval   base  72 cells, 83 min 33 s of rollouts added up
  eval   sft   72 cells, 32 min 00 s of rollouts added up
  eval   rl    72 cells, 29 min 03 s of rollouts added up
  the project evaluates several cells side by side (Settings.concurrency), so the
  added-up time is longer than the stage's wall time

━━ the network check (network_check.txt)
  # Network check during the RL evaluation, 2026-10-06 08:42
  # Non-loopback TCP/UDP of the engine (:8100), teacher (:8101), capture proxy (:8200) and harness processes:
  # (no lines above = no external connections)
  no connection lines: nothing left this machine while it looked
```

The stage clock covers the first three stages only. Its GRPO stage is the
first attempt, stopped by hand after step 8; the second attempt's 6 steps
took 12 min 34 s by its own clock. No log in the project times its RL
evaluation or the Pi re-runs. And `network_check.txt` is one look, at 08:42:
by the project's file times (`grpo_attempt1` saved its last adapter at 08:34,
`eval_rl_attempt1.jsonl` was last written at 09:23), that look fell during
the evaluation of RL attempt 1, not the final one.

## A fresh clone: `FAST=1 ./capstone.sh all`

Everything from the clone to the servers, as it printed (the 156 installed
packages' names left out):

```text
━━ setup: uv sync, scripts/setup.sh, the Node harnesses at their pins
  uv and npm run with private settings ($COURSE_HOME/28-capstone/installers): your uv.toml, ~/.netrc,
  ~/.npmrc, git config and Hugging Face token are never read
  setup.sh fetches Liquid's export when models/LFM2.5-2.6B-MLX-4bit is missing;
  chapter 11 already fetched it, so link it there and setup.sh fetches no weights
  $ mkdir -p $COURSE_HOME/28-capstone/multi-harness-rl/models
  $ ln -sfn $COURSE_HOME/models/LFM2.5-2.6B-MLX-4bit $COURSE_HOME/28-capstone/multi-harness-rl/models/LFM2.5-2.6B-MLX-4bit
  $ uv sync   (uv: private settings)
Using CPython 3.13.7
Creating virtual environment at: .venv
Resolved 162 packages in 3ms
   Building multi-harness-rl @ file://~/.cache/mhrl-course/28-capstone/multi-harness-rl
      Built multi-harness-rl @ file://~/.cache/mhrl-course/28-capstone/multi-harness-rl
Prepared 1 package in 461ms
Installed 156 packages in 885ms
  $ scripts/setup.sh   (uv: private settings)
Resolved 162 packages in 3ms
Checked 156 packages in 19ms
model ready in models/LFM2.5-2.6B-MLX-4bit
  the model every stage loads (MHRL_MODEL_PATH): ours, against Liquid's export
  $ cmp $COURSE_HOME/models/lfm2.5-2.6b-4bit-ours/model.safetensors $COURSE_HOME/models/LFM2.5-2.6B-MLX-4bit/model.safetensors
  same bytes: the weights are Liquid's export (chapter 11's compare.py)
  Claude Code 2.1.290 into tools/claude-code/, where mhrl/harnesses/claude_code.py looks
  $ npm install --prefix tools/claude-code/npm --save-exact --no-audit --no-fund @anthropic-ai/claude-code@2.1.290   (npm: private settings)

added 2 packages in 10s
  $ mkdir -p $COURSE_HOME/28-capstone/multi-harness-rl/tools/claude-code/2.1.290
  $ ln -sfn $COURSE_HOME/28-capstone/multi-harness-rl/tools/claude-code/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe $COURSE_HOME/28-capstone/multi-harness-rl/tools/claude-code/2.1.290/claude
  Pi 1.0.4 into tools/pi/, where mhrl/harnesses/pi.py looks
  $ npm install --prefix tools/pi --save-exact --no-audit --no-fund @earendil-works/pi-coding-agent@1.0.4   (npm: private settings)
npm warn deprecated node-domexception@1.0.0: Use your platform's native DOMException instead

added 119 packages in 7s
  the versions, as each harness reports itself (a private HOME, so none writes yours)
  $ env HOME=$COURSE_HOME/28-capstone/installers/probe CLAUDE_CONFIG_DIR=$COURSE_HOME/28-capstone/installers/probe/claude DISABLE_AUTOUPDATER=1 tools/claude-code/2.1.290/claude --version
2.1.290 (Claude Code)
  $ env HOME=$COURSE_HOME/28-capstone/installers/probe node tools/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js --version
1.0.4
  $ .venv/bin/python -c 'import importlib.metadata as m; print("mini-swe-agent", m.version("mini-swe-agent"))'
mini-swe-agent 2.4.6
━━ tasks: the databases and the train/test task files
  $ .venv/bin/python -m mhrl.tasks.generate
Wrote 180 train and 50 test tasks to ~/.cache/mhrl-course/28-capstone/multi-harness-rl/data/tasks (seed 0)

split  domain   easy  medium  hard  total  db KB
train  shop       18      24    18     60     44
train  school     18      24    18     60     24
train  library    18      24    18     60     52
test   clinic      8      10     7     25     36
test   airline     8      10     7     25     48
all               70      92    68    230
  230 tasks (train.jsonl + test.jsonl)
  git status: data/tasks unchanged, so the same seed rebuilt the committed files byte for byte
━━ tests: the project's own
  $ .venv/bin/python -m pytest -q
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: ~/.cache/mhrl-course/28-capstone/multi-harness-rl
configfile: pyproject.toml
testpaths: tests
plugins: platformdirs-4.12.3, anyio-4.15.1
collected 129 items

tests/test_core.py .............                                         [ 10%]
tests/test_harnesses.py .................................sss.            [ 38%]
tests/test_oracle.py ......                                              [ 43%]
tests/test_tasks.py .................................................... [ 83%]
.....................                                                    [100%]

======================= 126 passed, 3 skipped in 13.68s ========================
━━ the servers: engine 8149, teacher 8159, proxy 8219
  $ env MHRL_PROJECT=$COURSE_HOME/28-capstone/multi-harness-rl ./stack.sh start
starting the project's servers (engine 8149, teacher 8159, proxy 8219)...
  engine is up (http://127.0.0.1:8149/health)
  teacher is up (http://127.0.0.1:8159/health)
  proxy is up (http://127.0.0.1:8219/health)
capture level: tokens (train rollouts)
━━ smoke: one real captured rollout through the proxy
  $ .venv/bin/python -m mhrl.harnesses.smoke --harness claude-code
{"harness": "claude-code", "finished": true, "exit_code": 0, "answer_written": false, "answer": "", "correct": false, "turns": 8, "first_prompt_tokens": 2296, "roots": 1, "sequences": 3, "trainable": true, "budget_stops": 1, "seconds": 75.1}
```

- **What `setup.sh` fetches**: `uv sync` (PyPI's packages and OpenEnv from
  GitHub at commit `3f14355`, here all from uv's cache), then Liquid's model
  export from Hugging Face unless `models/LFM2.5-2.6B-MLX-4bit/config.json`
  exists. Every stage loads ours (`MHRL_MODEL_PATH`), so that download would
  be 1.5 GB nothing reads; with chapter 11's copy linked in, `setup.sh`
  printed `model ready` and fetched nothing. npm fetched Claude Code (2
  packages, its binary among them) and Pi (119 packages).
- **The tasks**: the same seed rebuilt the committed files byte for byte
  (`git status` shows `data/tasks` unchanged).
- **The smoke test**: one Claude Code rollout through the proxy, captured as
  one prefix-continuous chain (`roots` 1) and trainable. The base model used
  all 8 calls and wrote no answer; the smoke test checks the capture, not
  the answer.

Then the pipeline at `FAST=1` (6 tasks per harness, 10 tasks of
demonstrations, 2 GRPO steps):

```text
=== 1/5 baseline: the base model in every harness ===
2026-10-13 21:05:50
  $ .venv/bin/python -m mhrl.cli eval --run capstone-fast --harnesses mini-swe-agent,claude-code,pi --limit 6 --label base
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             33%              0%             44%

=== 2/5 answer-key demonstrations, then supervised fine-tuning ===
2026-10-13 21:10:24
  $ .venv/bin/python -m mhrl.cli demos --run capstone-fast --harnesses mini-swe-agent,claude-code --limit 10 --teacher-port 8159
[demos] 20/20 demonstrations usable
  $ .venv/bin/python -m mhrl.cli sft --run capstone-fast --epochs 1
[sft] 20 training rows from 20 demos
[sft] epoch 1/1 step 1 loss/token 2.3424 tokens 1232 (32.8s)
[sft] epoch 1/1 step 2 loss/token 2.7945 tokens 1363 (57.2s)
[sft] epoch 1/1 step 3 loss/token 2.4987 tokens 624 (21.9s)

=== 3/5 evaluate the SFT model ===
2026-10-13 21:12:35
  $ .venv/bin/python -m mhrl.cli eval --run capstone-fast --harnesses mini-swe-agent,claude-code,pi --limit 6 --label sft --adapter runs/capstone-fast/sft/adapter.safetensors
checkpoint    mini-swe-agent     claude-code              pi         average
sft                     100%             67%              0%             56%

=== 4/5 GRPO through the training harnesses ===
2026-10-13 21:16:56
  $ .venv/bin/python -m mhrl.cli grpo --run capstone-fast --harnesses mini-swe-agent,claude-code --init runs/capstone-fast/sft/adapter.safetensors --steps 2 --tasks-per-step 2 --group 4
[grpo] step 1/2 · 00:05:41 · reward 0.53 · correct 50% · calls 7.6 · no-signal groups 1/2 · ratio 1.000 · peak 3.9 GB
[grpo] step 2/2 · 00:09:07 · reward 0.27 · correct 25% · calls 7.9 · no-signal groups 1/2 · ratio 1.000 · peak 3.9 GB
  $ .venv/bin/python -m mhrl.cli steps --run capstone-fast
step  correct  calls  tokens/call  rollout s (median)
   1      50%    7.6          227                 176
   2      25%    7.9          120                  97

=== 5/5 evaluate the RL model ===
2026-10-13 21:26:09
  $ .venv/bin/python -m mhrl.cli eval --run capstone-fast --harnesses mini-swe-agent,claude-code,pi --limit 6 --label rl --adapter runs/capstone-fast/grpo/adapter_final.safetensors
checkpoint    mini-swe-agent     claude-code              pi         average
rl                      100%             83%              0%             61%

=== results ===
2026-10-13 21:31:14
  $ .venv/bin/python -m mhrl.cli report --run capstone-fast --harnesses mini-swe-agent,claude-code,pi
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             33%              0%             44%
sft                     100%             67%              0%             56%
rl                      100%             83%              0%             61%

=== done ===
```

At 6 tasks per harness one task is 16.7 points, and `read_results.py`
finds no checkpoint pair beyond noise. Beside the recorded run, on the same
6 tasks:

```text
━━ recorded against capstone-fast: the same checkpoint, the same cells
  checkpoint  harness        cells       recorded  capstone-fast  gained  lost  p      reading  calls on cells both solved
  base        mini-swe-agent     6              6              6       0     0  1.000  same     7.7 -> 7.7 (6), p 1.000
  base        claude-code        6              5              2       1     4  0.375  within   8.0 -> 8.0 (1), p 1.000
  base        pi                 6              0              0       0     0  1.000  same     none solved by both
  sft         mini-swe-agent     6              5              6       1     0  1.000  within   5.2 -> 7.4 (5), p 0.125
  sft         claude-code        6              5              4       1     2  1.000  within   5.7 -> 7.7 (3), p 0.250
  sft         pi                 6              2              0       0     2  0.500  within   none solved by both
  rl          mini-swe-agent     6              6              6       0     0  1.000  same     5.0 -> 7.5 (6), p 0.062
  rl          claude-code        6              6              5       0     1  1.000  within   4.2 -> 7.8 (5), p 0.062
  rl          pi                 6              2              0       0     2  0.500  within   none solved by both
  gained / lost: cells right in capstone-fast and wrong in recorded, and the reverse;
  p < 0.05 reads as beyond noise; calls: mean before -> after (cells), and the
  sign test's p on fewer against more
```

FAST took 25 min 26 s of stages and 28 min 2 s with the setup.

## The whole pipeline: FULL

The same clone, 24 tasks per harness, 60 tasks of demonstrations (120
demonstrations through two harnesses), 15 SFT steps and 6 GRPO steps:

```text
=== 1/5 baseline: the base model in every harness ===
2026-10-13 21:42:56
  $ .venv/bin/python -m mhrl.cli eval --run capstone --harnesses mini-swe-agent,claude-code,pi --limit 24 --label base
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             21%              0%             40%

=== 2/5 answer-key demonstrations, then supervised fine-tuning ===
2026-10-13 22:04:08
  $ .venv/bin/python -m mhrl.cli demos --run capstone --harnesses mini-swe-agent,claude-code --limit 60 --teacher-port 8159
[demos] 120/120 demonstrations usable
  $ .venv/bin/python -m mhrl.cli sft --run capstone --epochs 1
[sft] 120 training rows from 120 demos
[sft] epoch 1/1 step 1 loss/token 2.5523 tokens 1474 (50.3s)
[sft] epoch 1/1 step 2 loss/token 2.5090 tokens 1343 (44.8s)
[sft] epoch 1/1 step 3 loss/token 2.5246 tokens 1393 (37.2s)
[sft] epoch 1/1 step 4 loss/token 2.2630 tokens 1243 (26.1s)
[sft] epoch 1/1 step 5 loss/token 2.1361 tokens 1358 (32.5s)
[sft] epoch 1/1 step 6 loss/token 2.0697 tokens 1282 (36.2s)
[sft] epoch 1/1 step 7 loss/token 1.7142 tokens 1351 (22.9s)
[sft] epoch 1/1 step 8 loss/token 1.6690 tokens 1257 (36.7s)
[sft] epoch 1/1 step 9 loss/token 1.4887 tokens 1259 (36.4s)
[sft] epoch 1/1 step 10 loss/token 1.2097 tokens 1526 (27.9s)
[sft] epoch 1/1 step 11 loss/token 1.1930 tokens 1370 (41.9s)
[sft] epoch 1/1 step 12 loss/token 1.0441 tokens 1433 (38.1s)
[sft] epoch 1/1 step 13 loss/token 0.9591 tokens 1336 (36.9s)
[sft] epoch 1/1 step 14 loss/token 0.8223 tokens 1457 (33.2s)
[sft] epoch 1/1 step 15 loss/token 0.8018 tokens 1289 (52.1s)

=== 3/5 evaluate the SFT model ===
2026-10-13 22:14:31
  $ .venv/bin/python -m mhrl.cli eval --run capstone --harnesses mini-swe-agent,claude-code,pi --limit 24 --label sft --adapter runs/capstone/sft/adapter.safetensors
checkpoint    mini-swe-agent     claude-code              pi         average
sft                      88%             79%              4%             57%

=== 4/5 GRPO through the training harnesses ===
2026-10-13 22:27:23
  $ .venv/bin/python -m mhrl.cli grpo --run capstone --harnesses mini-swe-agent,claude-code --init runs/capstone/sft/adapter.safetensors --steps 6 --tasks-per-step 2 --group 4
[grpo] step 1/6 · 00:02:45 · reward 0.54 · correct 50% · calls 7.9 · no-signal groups 1/2 · ratio 1.001 · peak 3.9 GB
[grpo] step 2/6 · 00:05:41 · reward 0.40 · correct 38% · calls 7.5 · no-signal groups 1/2 · ratio 1.001 · peak 6.3 GB
[grpo] step 3/6 · 00:07:58 · reward 1.08 · correct 100% · calls 6.6 · no-signal groups 0/2 · ratio 1.000 · peak 6.3 GB
[grpo] step 4/6 · 00:11:01 · reward 0.94 · correct 88% · calls 7.0 · no-signal groups 0/2 · ratio 1.000 · peak 6.3 GB
[grpo] step 5/6 · 00:14:10 · reward 0.93 · correct 88% · calls 7.6 · no-signal groups 0/2 · ratio 1.000 · peak 6.3 GB
[grpo] step 6/6 · 00:16:55 · reward 0.67 · correct 62% · calls 7.1 · no-signal groups 0/2 · ratio 1.000 · peak 6.3 GB
  $ .venv/bin/python -m mhrl.cli steps --run capstone
step  correct  calls  tokens/call  rollout s (median)
   1      50%    7.9           86                  80
   2      38%    7.5           81                  64
   3     100%    6.6           66                  52
   4      88%    7.0           77                  58
   5      88%    7.6           96                  61
   6      62%    7.1           73                  51

=== 5/5 evaluate the RL model ===
2026-10-13 22:44:24
  $ .venv/bin/python -m mhrl.cli eval --run capstone --harnesses mini-swe-agent,claude-code,pi --limit 24 --label rl --adapter runs/capstone/grpo/adapter_final.safetensors
checkpoint    mini-swe-agent     claude-code              pi         average
rl                       96%             83%              0%             60%

=== results ===
2026-10-13 22:58:19
  $ .venv/bin/python -m mhrl.cli report --run capstone --harnesses mini-swe-agent,claude-code,pi
checkpoint    mini-swe-agent     claude-code              pi         average
base                    100%             21%              0%             40%
sft                      88%             79%              4%             57%
rl                       96%             83%              0%             60%

=== done ===
```

`read_results.py` on it, the final table first:

```text
━━ the final table: pass@1 with a Wilson 95% interval (24 cells per harness)
  checkpoint  mini-swe-agent       claude-code          pi                   average
  base        24/24 100% (86-100)  5/24 21% (9-40)      0/24 0% (0-14)       40%
  sft         21/24 88% (69-96)    19/24 79% (60-91)    1/24 4% (1-20)       57%
  rl          23/24 96% (80-99)    20/24 83% (64-93)    0/24 0% (0-14)       60%
  average = the plain mean over harnesses, each weighing the same; (low-high) = Wilson 95%
  at 24 tasks per harness, one task = 4.2 points
```

```text
━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair         harness        cells  before  after  gained  lost  flipped  p      reading
  base -> sft  mini-swe-agent    24      24     21       0     3        3  0.250  within noise
  base -> sft  claude-code       24       5     19      15     1       16  0.001  gained: beyond noise
  base -> sft  pi                24       0      1       1     0        1  1.000  within noise
  base -> rl   mini-swe-agent    24      24     23       0     1        1  1.000  within noise
  base -> rl   claude-code       24       5     20      16     1       17  0.000  gained: beyond noise
  base -> rl   pi                24       0      0       0     0        0  1.000  nothing changed
  sft -> rl    mini-swe-agent    24      21     23       2     0        2  0.500  within noise
  sft -> rl    claude-code       24      19     20       3     2        5  1.000  within noise
  sft -> rl    pi                24       1      0       0     1        1  1.000  within noise
  p < 0.05 reads as beyond noise; anything else, within it
```

```text
━━ calls per solve: model calls on solved cells only
  checkpoint  mini-swe-agent claude-code    pi             average over harnesses
  base        7.8            8.0            none           7.9
  sft         7.6            7.1            8.0            7.6
  rl          7.7            7.1            none           7.4
  by stage (average over harnesses): sft -0.3, rl -0.2 calls
  each checkpoint counts only the cells IT solved, so the sets differ; the pairs below
  count only cells both checkpoints solved

  pair         harness        both  mean before -> after  median  fewer  more  same  p
  base -> sft  mini-swe-agent   21  7.8 -> 7.6           8->8        3     0    18  0.250
  base -> sft  claude-code       4  8.0 -> 7.0           8->7        3     0     1  0.250
  base -> sft  pi                0  no cell solved by both
  base -> rl   mini-swe-agent   23  7.8 -> 7.7           8->8        4     1    18  0.375
  base -> rl   claude-code       4  8.0 -> 6.2           8->6.5      4     0     0  0.125
  base -> rl   pi                0  no cell solved by both
  sft -> rl    mini-swe-agent   21  7.6 -> 7.7           8->8        1     2    18  1.000
  sft -> rl    claude-code      17  7.0 -> 7.1           7->7        2     4    11  0.688
  sft -> rl    pi                0  no cell solved by both
  p: the exact sign test on fewer against more (same dropped), as for gained and lost
  base -> rl, every harness pooled: 27 cells solved by both, 7.9 -> 7.5 calls (5% fewer)
```

```text
━━ the harness gap: the same weights, the same tasks, two harnesses
  checkpoint  first          second         tasks  only first  only second  p      reading
  base        mini-swe-agent claude-code       24          19            0  0.000  beyond noise
  base        mini-swe-agent pi                24          24            0  0.000  beyond noise
  base        claude-code    pi                24           5            0  0.062  within noise
  sft         mini-swe-agent claude-code       24           4            2  0.688  within noise
  sft         mini-swe-agent pi                24          20            0  0.000  beyond noise
  sft         claude-code    pi                24          19            1  0.000  beyond noise
  rl          mini-swe-agent claude-code       24           3            0  0.250  within noise
  rl          mini-swe-agent pi                24          23            0  0.000  beyond noise
  rl          claude-code    pi                24          20            0  0.000  beyond noise
```

```text
━━ the reading, worded from the tables above
  the harness gap, beyond noise:
    base: mini-swe-agent ahead of claude-code, 19 tasks to 0 (p 0.000)
    base: mini-swe-agent ahead of pi, 24 tasks to 0 (p 0.000)
    sft: mini-swe-agent ahead of pi, 20 tasks to 0 (p 0.000)
    sft: claude-code ahead of pi, 19 tasks to 1 (p 0.000)
    rl: mini-swe-agent ahead of pi, 23 tasks to 0 (p 0.000)
    rl: claude-code ahead of pi, 20 tasks to 0 (p 0.000)
  checkpoints, beyond noise (p < 0.05):
    base -> sft, claude-code: 5 -> 19 solved, 15 gained, 1 lost (p 0.001)
    base -> rl, claude-code: 5 -> 20 solved, 16 gained, 1 lost (p 0.000)
  checkpoints, within noise:
    base -> sft, mini-swe-agent: -3 (0 gained, 3 lost, p 0.250)
    base -> sft, pi: +1 (1 gained, 0 lost, p 1.000)
    base -> rl, mini-swe-agent: -1 (0 gained, 1 lost, p 1.000)
    sft -> rl, mini-swe-agent: +2 (2 gained, 0 lost, p 0.500)
    sft -> rl, claude-code: +1 (3 gained, 2 lost, p 1.000)
    sft -> rl, pi: -1 (0 gained, 1 lost, p 1.000)
  the average, 40% -> 57% -> 60%: one seed, 24 tasks per harness, read only through the pairs above
  calls, base -> sft, every harness pooled: fewer on 6 of 25 shared solves, more on 0 (p 0.031): beyond noise
  calls, sft -> rl, every harness pooled: fewer on 3 of 38 shared solves, more on 6 (p 0.508): within noise
  pi, never trained in: 0 -> 0 solved, base -> rl (0 gained, 0 lost, p 1.000): suggestive, not shown
```

What it cost the laptop:

```text
━━ memory: MLX's own peak (mx.get_peak_memory, the trainer's process)
  sft    3.96 GB at step 13 (sft/metrics.jsonl, 15 steps)
  grpo   6.31 GB at step 2 (grpo/metrics.jsonl, 6 steps)
  footprint sampled by capstone.sh (memory.tsv; top's MEM, in GB), per stage:
  1/5      engine 4.27 · teacher 0.36 · proxy 0.18 · stage 0.08 · all at once 4.67 (largest)
  2/5      engine 2.17 · teacher 0.15 · proxy 0.18 · stage 4.10 · all at once 6.50 (largest)
  3/5      engine 3.81 · teacher 0.14 · proxy 0.18 · stage 0.08 · all at once 4.20 (largest)
  4/5      engine 3.78 · teacher 0.14 · proxy 0.18 · stage 4.59 · all at once 6.94 (largest)
  5/5      engine 3.89 · teacher 0.14 · proxy 0.18 · stage 0.08 · all at once 4.29 (largest)

━━ wall time
  the stage clock, banner to banner (~/.cache/mhrl-course/28-capstone/multi-harness-rl/runs/capstone/stages.txt):
  1/5 baseline: the base model in every harness               21 min 12 s  21:42:56 -> 22:04:08
  2/5 answer-key demonstrations, then supervised fine-tuning  10 min 23 s  22:04:08 -> 22:14:31
  3/5 evaluate the SFT model                                  12 min 52 s  22:14:31 -> 22:27:23
  4/5 GRPO through the training harnesses                     17 min 01 s  22:27:23 -> 22:44:24
  5/5 evaluate the RL model                                   13 min 55 s  22:44:24 -> 22:58:19
  results                                                      0 min 02 s  22:58:19 -> 22:58:21
  all stages                                                  75 min 25 s
  the records' own clocks:
  sft    15 steps, 9 min 13 s of training steps (the model's loading not counted)
  grpo   6 steps, 16 min 55 s (12 min 01 s of it rollouts)
  eval   base  72 cells, 84 min 31 s of rollouts added up
  eval   sft   72 cells, 50 min 47 s of rollouts added up
  eval   rl    72 cells, 55 min 08 s of rollouts added up
  the project evaluates several cells side by side (Settings.concurrency), so the
  added-up time is longer than the stage's wall time

━━ the network check (network_check.txt)
  # Network check during the RL evaluation, 2026-10-13 22:44 to 22:58, one look every 10 s: 82 looks
  # Non-loopback TCP/UDP of the engine (:8149), teacher (:8159), capture proxy (:8219) and harness processes:
  # (no lines above = no external connections)
  no connection lines: nothing left this machine while it looked
```

The stages took 75 min 25 s. The trainers' MLX peaks were 3.96 GB (SFT) and
6.31 GB (GRPO, from step 2 on); with the engine beside them, every process
together reached 6.94 GB during GRPO by their footprints. The engine alone
held up to 4.27 GB of footprint while 72 cells ran four at a time (the
weights are 1.6 GB, as its own banner says). 82 looks at the network during
the RL evaluation found nothing leaving the laptop.

## Does a fresh clone reproduce the recorded run?

`--against` puts the recorded run beside the fresh one, checkpoint by
checkpoint, on the same 24 cells per harness:

```text
━━ recorded against capstone: the same checkpoint, the same cells
  checkpoint  harness        cells  recorded  capstone  gained  lost  p      reading  calls on cells both solved
  base        mini-swe-agent    24        24        24       0     0  1.000  same     7.8 -> 7.8 (24), p 1.000
  base        claude-code       24        15         5       2    12  0.013  beyond   8.0 -> 8.0 (3), p 1.000
  base        pi                24         4         0       0     4  0.125  within   none solved by both
  sft         mini-swe-agent    24        18        21       5     2  0.453  within   4.6 -> 7.5 (16), p 0.000
  sft         claude-code       24        17        19       5     3  0.727  within   4.9 -> 7.0 (14), p 0.000
  sft         pi                24         9         1       0     8  0.008  beyond   4.0 -> 8.0 (1), p 1.000
  rl          mini-swe-agent    24        20        23       3     0  0.250  within   4.8 -> 7.7 (20), p 0.000
  rl          claude-code       24        18        20       4     2  0.688  within   4.1 -> 6.9 (16), p 0.000
  rl          pi                24         9         0       0     9  0.004  beyond   none solved by both
  gained / lost: cells right in capstone and wrong in recorded, and the reverse;
  p < 0.05 reads as beyond noise; calls: mean before -> after (cells), and the
  sign test's p on fewer against more
```

Read from that table and the one above:

- **The base model does not reproduce in Claude Code**: 15 of 24 recorded,
  5 fresh (2 gained, 12 lost, p 0.013), with the same weights at temperature
  0. Chapter 24's own fresh baseline, with its copy of the project's
  evaluation, measured 7 of 24. The recorded base ran before the project's code review (its README
  says so), so two fresh measurements agree with each other and not with the
  record.
- **Pi's rise does not reproduce**: the recorded 9 of 24 after SFT and after
  RL became 1 and 0 (8 and 9 lost, p 0.008 and 0.004). Pi was 0 of 24 at
  base too (4 recorded, p 0.125).
- **The cheaper solves do not reproduce**: on the cells both runs solved,
  the fresh SFT and RL models used more calls in both trained harnesses
  (mini-swe-agent 4.6 -> 7.5 and 4.8 -> 7.7, Claude Code 4.9 -> 7.0 and 4.1
  -> 6.9, p < 0.001 each). Calls per solve went 7.9 -> 7.6 -> 7.4, not 7.1
  -> 4.8 -> 4.3.
- **SFT's loss did what chapter 25 measured with the project's code**: 2.5523
  at step 1 to 0.8018 at step 15, within 0.007 of chapter 25's at every step,
  not the recorded 0.11. The recorded SFT trained before the code review
  turned AdamW's bias correction on, and with it on, 15 steps leave the
  model's habits nearly as they were.
- **What a fresh clone does show**: SFT took Claude Code from 5 to 19 of 24
  (15 gained, 1 lost, p 0.001), beyond noise in this run, where the recorded
  run's +2 was not; mini-swe-agent lost 3 after SFT (p 0.250) and won 2
  back after RL (p 0.500), both within noise. The averages, 40 -> 57 -> 60,
  are again read only through the pairs.

## Cost, privacy, and why local

- **Every call re-sends the conversation.** A captured row holds the whole
  conversation with a mask on the tokens the model sampled, so each call's
  prompt length can be read off it (`cost.py`, a driver check):

  ```text
  harness         rollouts  calls  prompt tokens sent (sum over calls)  tokens sampled  sent per sampled
  claude-code           24    6.8                               20,333             622                33
  mini-swe-agent        24    7.8                                8,911             682                13
  means per rollout; rows without a captured call are left out
  ```

  That is the FULL run's GRPO rollouts. The recorded run's:

  ```text
  harness         rollouts  calls  prompt tokens sent (sum over calls)  tokens sampled  sent per sampled
  claude-code           24    6.4                               18,464             327                56
  mini-swe-agent        24    6.0                                6,394             292                22
  means per rollout; rows without a captured call are left out
  ```

  A Claude Code rollout sent 33 times the tokens it sampled (56 times in the
  recorded run, whose replies were shorter); a hosted
  API bills those input tokens on every call. Fewer calls and shorter
  replies cut both numbers, which is why the article rewards fewer tool
  calls and the project rewards fewer generated tokens (chapter 7). On this
  laptop none of it has a per-token bill: the FULL run's cost was 75
  minutes of the laptop.
- **Nothing left the laptop** while the network was watched: the FULL run's
  82 looks found no connection off the machine from the engine, the teacher,
  the proxy or any harness.
- **Why a local model at all**: training through a harness needs the exact
  token ids the model sampled and their log-probabilities (chapters 15 and
  20). Hosted APIs return text, not those, so they can evaluate an agent but
  never train through it.

## The article's numbers

`read_results.py` keeps them in a box of their own, in the article's words:

```text
┌─ the article's own numbers (Hugging Face + Liquid AI) ─────────────────────┐
│ from "The ultimate guide to multi-harness RL" (2026), its words:           │
│ "Trained across all four, it went from solving 42% of held-out tasks to    │
│ 54%, averaged over the four harnesses, and used 31% fewer tool calls on    │
│ the tasks it already solved."                                              │
│ "Two H100s per run on the Hugging Face cluster, and one E2B sandbox per    │
│ rollout with 1 CPU and 4 GB of memory."                                    │
│ "Training took about 32 hours for the OpenCode-only run and 46 hours for   │
│ the multi-harness run, not counting evaluation, queueing and restarts."    │
└────────────────────────────────────────────────────────────────────────────┘
  ours: three harnesses on one laptop, our own tasks, one seed; read through the
  tables above, never beside these
```

The three sentences were checked against the article itself on 2026-10-13
(`article.py` in the run log, on the saved page):

```text
the article: the page saved from https://fineenvs-multi-harness-rl.hf.space/ on 2026-10-13 (the Space at https://huggingface.co/spaces/FineEnvs/multi-harness-rl)
every sentence holding 42% -> 54% and 31% fewer tool calls:
  "Trained across all four, it went from solving 42% of held-out tasks to 54%, averaged over the four harnesses, and used 31% fewer tool calls on the tasks it already solved."
  "On the test set, the OpenCode-only model performs better than the multi-harness model under OpenCode (58% vs 50%), while the multi-harness model performs better under Claude Code (49% vs 42%) and Codex (54% vs 43%)."
  "Both models solve more tasks with fewer tool calls, and the multi-harness model cuts more, 31% fewer calls than the base model on the tasks both solved, against 11% for OpenCode-only."
  "The multi-harness LFM model made 31% fewer calls than the base model on the tasks both solved."
every sentence holding two H100s:
  "One H100 trains while a second serves the policy with vLLM ( Kwon et al., 2023 ) and keeps producing rollouts, so training never waits for the slowest agent."
  "Two H100s per run on the Hugging Face cluster, and one E2B sandbox per rollout with 1 CPU and 4 GB of memory."
every sentence holding 32-46 hours:
  "Training took about 32 hours for the OpenCode-only run and 46 hours for the multi-harness run, not counting evaluation, queueing and restarts."
```

Its 42% and 54% are four harnesses on its own test tasks, its 31% is
counted on the tasks the base model and the trained one both solved, and its
runs took two H100s for 32 to 46 hours. None of those numbers sits in a
table beside ours.

## Plan versus measured

| the plan said | measured | record |
| --- | --- | --- |
| folder `course/28-capstone/` | `28-capstone/` at the root of this repository, like every chapter | |
| capstone.sh, in a fresh clone of the project: uv sync, scripts/setup.sh, python -m mhrl.tasks.generate (230 tasks), pytest, the smoke test | as planned: 230 tasks, rebuilt byte for byte; 126 passed and 3 skipped; the smoke rollout trainable with roots 1. Added: Claude Code 2.1.290 and Pi 1.0.4 into the clone's `tools/`, which a fresh clone lacks | `01-fast.txt` |
| scripts/serve.sh start | the same three servers through chapter 27's `stack.sh` on 8149, 8159 and 8219: `serve.sh` hard-codes 8100, 8101 and 8200 and its stop kills every engine on the machine | `01-fast.txt` |
| then scripts/train_from_scratch.sh | its commands, in its order, with `--teacher-port` added to `demos` (the script hard-codes the teacher's 8101); a test checks the two lists stay the same | `tests/test_28_capstone.py` |
| with `python -m mhrl.cli steps` watching tokens per call | printed after every GRPO step, and once at the end | `02-full.txt` |
| FAST=1 sets EVAL_LIMIT=6 DEMO_LIMIT=10 GRPO_STEPS=2 | as planned: 25 min 26 s of stages | `01-fast.txt` |
| (review) set GRPO_STEPS=6 explicitly and say so | done: `train_from_scratch.sh` defaults to 16 and the CLI to 20 | `capstone.sh` |
| (review) run with MHRL_MODEL_PATH=$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours, or prove ours equals the official export | both: every stage loads ours, and `setup` compares the weights with `cmp` first ("same bytes") | `01-fast.txt` |
| (review) quote stage times measured from the logs | the recorded run's from its log's banners (21 min 50 s, 10 min 26 s, 7 min 39 s) and GRPO's own clock (12 min 34 s); FULL's 21 min 12 s, 10 min 23 s, 12 min 52 s, 17 min 1 s, 13 min 55 s | `00-read-results-recorded.txt`, `02-full.txt` |
| read_results.py renders the final table with intervals, calls per solve, peak memory and wall time from runs/name/ and reads network_check.txt | as planned. Added: gained / lost / p for every pair (chapter 25's build note), the harness gap, the paired calls, and `--against` | the tests |
| '[demos] 120/120 demonstrations usable' | as planned, in FULL | `02-full.txt` |
| '[sft] ... loss/token 2.55' down to 0.11 | 0.11 is the recorded run's, before the code review turned AdamW's bias correction on (its log: 0.1098 at step 15). A fresh clone prints 2.5523 down to 0.8018, as chapter 25 measured with the project's code | `02-full.txt`; the project's `session_main.log` |
| '[grpo] step k/6 · ... · ratio 1.000 · peak 6.9 GB' | the recorded run's line, as its log printed it. FULL: ratio 0.9997 to 1.0012, peak 6.31 GB | `02-full.txt` |
| the final table: base 100 / 62 / 17% (60%), SFT 75 / 71 / 38% (61%), RL 83 / 75 / 38% (65%) | the project's README table, and its Pi column comes from the re-run after the sandbox fix; the report `train_from_scratch.sh` printed says 58, 49 and 53%, with Pi at 12, 0 and 0%. FULL: 100 / 21 / 0% (40%), 88 / 79 / 4% (57%), 96 / 83 / 0% (60%) | the project's `take4_report.log`; `02-full.txt` |
| calls per solve 7.1 -> 4.8 -> 4.3 | as recorded: the mean over harnesses of each harness's mean on its solved cells (means, not medians). FULL: 7.9 -> 7.6 -> 7.4 | `00-read-results-recorded.txt`, `02-full.txt` |
| the harness gap and the fall in calls per solve are well outside the noise | both, in the recorded run: 9 to 0 (p 0.004) and 29 of 30 solves cheaper (p < 0.001). In FULL the gap holds (19 to 0) and the fall is beyond noise but small: fewer calls on 6 of 25 shared solves (p 0.031), -0.3 calls | as above |
| 60 -> 61 -> 65 is not | agreed: within noise; FULL's 40 -> 57 -> 60 holds one change beyond noise, Claude Code's base -> SFT (15 gained, 1 lost) | as above |
| Pi rose 17 -> 38% without ever training in it | suggestive, not shown: 7 gained and 2 lost by SFT (p 0.180), 8 and 3 by RL (p 0.227). It did not reproduce: FULL's Pi went 0 -> 1 -> 0 | as above; `03-full-against.txt` |
| mini-swe-agent's 100 -> 75 -> 83% is imitation copying 'never double-check', partly recovered by RL | the loss is beyond noise (0 gained, 6 lost, p 0.031); "partly recovered" is within noise (2 gained, 0 lost, p 0.500). "Never double-check" rests on the project's README: no demonstration re-queries an answer, but chapter 25 found 16 of the 60 demonstrated tasks reading `answer.txt` back. FULL lost 3 after SFT (p 0.250) | as above; chapter 25 |
| the article's numbers in their own box: 42% to 54% across four harnesses, 31% fewer tool calls on tasks it already solved, two H100s | checked sentence by sentence against the article; its runs took 32 to 46 hours of training on those two H100s | `00-article.txt` |
| cost: every call re-sends the history, so fewer calls and shorter replies mean fewer tokens, and a local run has no per-token bill | measured on the GRPO rollouts: a Claude Code rollout sent 20,333 prompt tokens to sample 622 in FULL (18,464 and 327 recorded) | `04-full-cost.txt`, `04-recorded-cost.txt` |
| privacy: an empty network check | the recorded check is one look at 08:42, during RL attempt 1's evaluation by the project's file times; FULL's 82 looks during its RL evaluation found nothing | `02-full.txt` |
| hosted APIs return no token ids or logprobs, so they can evaluate this agent but never train through it | stated, not measured here (chapters 15 and 20) | |
| a fresh clone gives close, not identical, numbers, because float32 logprobs, bias correction, the token bonus, the length guard, prefix dedup and the narrower sandbox landed after the first stages | not close in three places, each beyond noise: Claude Code's base (15 -> 5), Pi after training (9 -> 1 and 0) and calls per solve (more on the cells both runs solved). Bias correction alone explains SFT's loss (chapter 25). Prefix dedup did not land after them: RL attempt 1 already trained each shared prefix once (chapter 26) | `03-full-against.txt` |
| (the project's README) the whole pipeline took about 80 minutes; the engine holds 1.6 GB | no log records 80 minutes: the recorded stage clock covers 39 min 55 s of its first three stages, and its RL evaluation has no clock. 1.6 GB is the weights, as the engine's banner says; the engine's MLX peak measured 2.9 to 3.4 GB in chapters 24 to 26, and its footprint here up to 4.27 GB. FULL took 75 min 25 s | `02-full.txt` |
| the project's ports: engine 8100, teacher 8101, proxy 8200 | the defaults are the project's; this chapter uses 8149, 8159 and 8219, so it never meets another chapter's servers (27: 8129, 8139, 8209) or a film's | `commands.txt` |

- **The project's numbers** in this chapter come from its recorded files: its
  evaluation records, trainers' metrics and network check (`runs/main/`,
  bundled slim in `data/project_run/`), its stage banners
  (`runs/logs/session_main.log`) and the report its pipeline printed
  (`runs/logs/take4_report.log`). The bundle never holds a session id.

## The files

Line counts at commit `543f234`: all lines, and the lines of code (no
blank lines, comments or docstrings; for the shell scripts, no blank or
comment lines).

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `capstone.sh` | 534 | 432 | the whole pipeline from a fresh clone: clone, setup, tasks, tests, servers, smoke, the five stages; the watchers (steps, memory, network); `plan` |
| `stack.sh` | 221 | 163 | chapter 27's: the project's three servers on chosen ports, stopping only its own |
| `read_results.py` | 1,534 | 812 | a run read honestly: the table with intervals, calls per solve, every pair's gained / lost / p, the harness gap, memory, the clocks, the network check, the reading; `--against` |
| `stats.py` | 142 | 49 | chapter 24's Wilson interval and exact p, cut to what is read here |
| `paths.py` | 34 | 11 | paths written the short way |
| `test_28_capstone.py` | 370 | 244 | the pinned plan, the commands against `train_from_scratch.sh`, the guards, the network filter, the pipeline end to end with the teacher as the engine |
| `test_28_read_results.py` | 310 | 233 | the pinned report, cross-checked against the real `runs/main`; broken, repeated and partial records; `--against` |
| `capstone_cli_scripted_28.py` | 92 | 49 | the project's CLI with a tiny LFM2 in place of the trainers' model, for the end-to-end test |
| `data/project_run/` | | | the project's recorded run, slim: evaluation records (no session ids), the trainers' metrics, the network check, the stage banners |

## What this chapter leaves out

- **A second seed.** One FULL run is one draw. It disagrees with the
  recorded run beyond noise in three places, and a third run is what would
  say which of the two is the usual one. That is about 75 minutes of the
  laptop, `RUN=capstone-2 ./capstone.sh pipeline`.
- **Why the base model moved.** The recorded base and the fresh one differ
  in Claude Code with the same weights at temperature 0. The project's code
  changed between them (its README lists the review's fixes), and chapter 24
  showed one changed path in a prompt sending greedy decoding elsewhere in 7
  of 24 cells. Telling those causes apart would need the recorded run's code,
  which the project's history does not keep.
- **Training without bias correction.** Chapter 25's `--no-bias-correction`
  reproduces the recorded SFT; the capstone runs the project's code as it is.

## Mirrors in the project

`scripts/train_from_scratch.sh` (every stage), `scripts/serve.sh` (the three
servers, through chapter 27's `stack.sh`), `scripts/setup.sh`,
`mhrl/tasks/generate.py`, `mhrl/harnesses/smoke.py`, `mhrl/cli.py`
(`eval`, `demos`, `sft`, `grpo`, `steps`, `report`) and `runs/main/`.
