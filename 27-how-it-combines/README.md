# Chapter 27. How it all combines

Twenty-six chapters each built one piece on its own: a grader, a policy
gradient, a group of attempts, a clip, a tokenizer, an adapter, an engine, a
sandbox, a capture proxy, three harnesses, a fair measurement, a teacher, a
guard. The project this course follows is all of them in one repository. So
where does each piece live in it, and what happens when they all run
together, for real, for one step?

This chapter answers both. `map.py` lays every chapter beside the project
file and function it mirrors, and checks that each one is still there. Then
`one_step.py` runs the project itself, its own `train_grpo`, for one GRPO
step on 4 Claude Code rollouts, and prints what passes at every hand-off:
the folder, the key, the command, each call, the grade, the reward, the
advantages, a training row, the ratios, the update, the adapter file, and
the engine switching to it.

## Words first

- A **call** is one model request inside an agent, a **rollout** one whole
  attempt at a task, and a **step** one optimizer update (the course's words
  since chapter 1).
- The **project** is the multi-harness-rl repository this course follows:
  the package `mhrl/`, its scripts, its configs and its tests. A
  **checkout** is a copy of it on your disk; this run's is
  `~/sai_work/multi-harness-rl` at commit `25074c0`.
- A **setting** is a number or a path the whole project agrees on, kept in
  one place (`mhrl/config.py`, the class `Settings`). An **environment
  variable** is a name and a value a program reads when it starts; the
  project reads `MHRL_` followed by a setting's name, so
  `MHRL_ENGINE_PORT=8129` replaces the setting `engine_port`.
- A **local server** is a program on your own machine that answers requests
  on a **port**, a number on the address `127.0.0.1`. Two programs cannot
  listen on one port. The project's three servers are the **engine** (the
  model, chapter 18), the **teacher** (the answer key, chapter 25) and the
  **capture proxy** (chapter 20); together they are the **stack**.
- A **hand-off** is a place where one part of the step gives its result to
  the next. A **hook** is a few lines that run at a hand-off and print what
  passed, without changing it.
- A **unit test** checks one small piece on its own, in seconds. A **smoke
  test** runs the whole thing once, end to end, to see that nothing breaks.
- A **snapshot** is a measurement written down: here, the line count of
  every project file and function the map names, at one commit.
- A **no-contrast group** is a group of 4 rollouts whose rewards are all
  equal: every advantage is 0 and it teaches nothing (chapter 5).

## Run it

```bash
cd multi-harness-RL-course/27-how-it-combines
export MHRL_PROJECT=~/multi-harness-rl     # your checkout of the project, after its `uv sync`
uv run map.py                              # every chapter beside the project code it mirrors
uv run stacks.py                           # the article's stack beside ours
(cd "$MHRL_PROJECT" && .venv/bin/python -m pytest -p no:cacheprovider -q)   # the project's own tests
export MHRL_ENGINE_PORT=8129 MHRL_PROXY_PORT=8209 MHRL_TEACHER_PORT=8139
./stack.sh start                           # the project's engine, teacher and capture proxy
"$MHRL_PROJECT/.venv/bin/python" -m mhrl.harnesses.smoke --harness claude-code --runs 1
uv run one_step.py --init "$MHRL_PROJECT/runs/main/sft/adapter.safetensors"
./stack.sh stop
```

`map.py` without a checkout (no `MHRL_PROJECT`, no `--project`) prints from
its bundled snapshot of the project at `25074c0`; `--chapter N` and
`--part N` print a slice. `one_step.py` imports the project, so it runs
itself again with the project's own Python. `--init` names the adapter to
start from (the project's GRPO started from its SFT adapter); without it the
step starts from the base model. `./stack.sh scripted` puts the project's
answer-key teacher on the engine's port: it loads no weights, so the whole
pipeline runs without the 2.6B, and the tests use it.

The project's ports are 8100 (engine), 8101 (teacher) and 8200 (proxy). On
this Mac 8100 belongs to another service and another chapter's film used
8119 and 8200, so this run used 8129, 8139 and 8209. The project's own
`scripts/serve.sh` cannot do that: it writes 8100, 8101 and 8200 into its
commands, its health check would call another service on 8100 "up", and its
stop kills every engine and proxy on the machine. `stack.sh` runs
`serve.sh`'s three commands unchanged on the ports you choose, refuses a
port already in use, and stops only the servers it started; the project's
files are not edited.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, and a
checkout of the project after its `uv sync` and `scripts/setup.sh` (the
model, the pinned Claude Code). The scripts in this folder need only
Python's standard library.

## The run log

Every output below is copied from the run log, made from a clean
`COURSE_HOME/27-how-it-combines` on an M3 Pro with 18 GB at commit
`13a3bc9` (step 12 at `0ce9d41`, which changes only what `stack.sh stop`
redacts), everything under `nice -n 10` with `UV_NO_CONFIG=1` and
`PYTHONDONTWRITEBYTECODE=1` (so nothing was written into the project's
folders).

| step | command | what it shows | exit · seconds · memory |
| --- | --- | --- | --- |
| 00 | `rm -rf ~/.cache/mhrl-course/27-how-it-combines` | a clean start | exit 0 · 0.0 s · 0.00 GB |
| 01 | `uv run map.py` (with `MHRL_PROJECT` set) | the map, measured in the checkout | exit 0 · 0.2 s · 0.03 GB |
| 02 | `uv run map.py --project ~/sai_work/multi-harness-rl` | the same, the checkout named | exit 0 · 0.2 s · 0.03 GB |
| 03 | `uv run stacks.py` | the article's stack beside ours | exit 0 · 0.1 s · 0.03 GB |
| 04 | `uv run stacks.py --verify` | every quote checked against the article | exit 0 · 2.0 s · 0.04 GB |
| 05 | the project's `pytest` (above), with `MHRL_RUNS_DIR` in `COURSE_HOME` | 126 passed, 3 skipped; the checkout unchanged | exit 0 · 9.1 s · 0.12 GB |
| 06 | `uv run pytest tests/test_27_how_it_combines.py tests/test_27_project.py` | the chapter's tests (40 passed) | exit 0 · 15.2 s · 0.56 GB |
| 07 | `./stack.sh start` | the three servers up, capture level tokens | exit 0 · 8.7 s · 0.02 GB |
| 08 | the smoke test (above) | one real captured rollout | exit 0 · 25.2 s · 0.21 GB |
| 09 | `uv run one_step.py --project … --init …/runs/main/sft/adapter.safetensors` | one real GRPO step | exit 0 · 96.3 s · 0.59 GB |
| 10 | `./stack.sh stop` | the servers stopped | exit 0 · 3.2 s · 0.00 GB |
| 11 | `env -u MHRL_PROJECT uv run map.py` | the map from its bundled snapshot | exit 0 · 0.1 s · 0.02 GB |
| 12 | `bash 12_flat_draw.sh` (no 2.6B: the scripted stack) | a draw without contrast, and what it leaves | exit 0 · 11.7 s · 0.56 GB |
| 13 | the project's `pytest` again, with `-rs` | why its 3 tests were skipped | |

"Memory" is the largest single process `/usr/bin/time` saw; it does not
count the Metal buffers MLX allocates. MLX's own peak for the trainer is
printed by `train_grpo` (7.02 GB, step 09). The engine reports only its
loaded size (1.6 GB, step 10); a watcher read the engine's footprint, as
macOS's `footprint` tool counts it, every 5 s: at most 3,596 MB, while the
model loaded and sampled.

Steps 07 to 09 load the 2.6B. They first waited, from 15:43 to 18:16, for an
idle owner and a quiet machine; the wait was stopped when the battery fell
to 12% while discharging, and at the owner's word the three steps then ran
as soon as the Mac was on mains power and no other model or render was
running, at 20:26. A watcher read free memory every 5 s, ready to stop
everything under 15%: the lowest it saw was 26%. No sleep or wake fell
inside the window. The window then started a second time by itself: the
script that held the machine-wide lock was edited while it ran, and the
shell re-read it and ran its new last line. That second window was stopped
before its smoke test finished, and nothing of it is used (the run log's
`commands.txt` says which file it touched and how it was restored).

## Every chapter in the project: `map.py`

```bash
uv run map.py
```

From its bundled snapshot (step 11), the head, the last part, and the
footer:

```text
map.py · each chapter beside the project code it mirrors
project   the bundled snapshot of multi-harness-rl at 25074c0 (OpenEnv 3f14355)
          --project PATH or MHRL_PROJECT measures a checkout instead
lines     a file's lines (wc -l) · each name's, from its def to its end
kinds     copy: the project's code line for line · cut down: the same steps, less around them
          toy: the idea on the toy, full size in the project · runs: the chapter runs or reads it

```

```text
Part 7 · Training inside real harnesses, measured honestly
24 measuring · copy · evaluate.py, baseline_eval.py, noise.py
   pass@1 per harness on a balanced held-out set
   mhrl/evaluate.py                129  evaluate 46 · summarize 29
   mhrl/cli.py                     322  balanced 24 · `eval`
25 sft-in-harnesses · cut down · teacher.py, collect.py, train_sft.py, memory_probe.py
   the answer-key teacher, SFT on captured rows, attention in blocks
   mhrl/engine/oracle.py           270  AnswerKeyTeacher 202
   mhrl/training/sft.py            178  collect_demonstrations 40 · demo_items 19 · train_sft 71
   mhrl/training/common.py         237  chunked_attention 49 · load_policy 20 · token_logprobs 25
26 grpo-in-harnesses · cut down · grpo_harness.py, steps.py, replay.py
   GRPO through the harnesses, and the length guard
   mhrl/training/grpo.py           270  train_grpo 186 · group_advantages 14
   mhrl/rollout.py                 336  run_rollout 89 · reward_for 17 · tool_bonus 13
   mhrl/cli.py                     322  print_steps 37 · `steps`
27 how-it-combines · runs · map.py, one_step.py, stack.sh
   this chapter: the settings, the command line, one real step
   mhrl/config.py                  107  Settings 55 · Defaults 23 · get_settings 7
   mhrl/cli.py                     322  main 88 · CliDefaults 14 · `demo` · `eval` · `demos` · `sft`
                                        `grpo` · `steps` · `report`
   mhrl/training/grpo.py           270  train_grpo 186 · use_adapter 13
   mhrl/rollout.py                 336  run_rollout 89
   scripts/serve.sh                 60  start 12 · stop 13 · status 5
   mhrl/harnesses/smoke.py         205  smoke_once 50 · main 28
   tests/test_core.py              129  the whole file
   tests/test_harnesses.py         713  the whole file
   tests/test_oracle.py            123  the whole file
   tests/test_tasks.py             670  the whole file
28 capstone · runs · (not written yet)
   next: the whole pipeline, from a fresh clone
   scripts/train_from_scratch.sh    49  the whole file
   mhrl/evaluate.py                129  format_table 21
   mhrl/cli.py                     322  markdown_results 32 · `report`

the project's code: 41 files, 11,460 lines; chapters 1-26 mirror 28 files, 9,742 lines (85.0%)
left for this chapter and the next
   scripts/train_from_scratch.sh 49 · tests/test_core.py 129 · tests/test_harnesses.py 713
   tests/test_oracle.py 123 · tests/test_tasks.py 670
mirrored by no chapter
   main.py 6 · mhrl/harnesses/__init__.py 5 · mhrl/harnesses/errors.py 23
   5 empty __init__.py files
```

A file's size is its line count. A name's size is the lines of its
definition, from its `def` (or `class`, or assignment) to its last line,
docstring included; `train_grpo.loss_fn` is the loss defined inside
`train_grpo`, and a word in backquotes is a `python -m mhrl.cli`
subcommand. The kind says how the chapter's code relates to the project's:
**copy** (the project's code line for line, comments added), **cut down**
(the same steps with less around them), **toy** (the idea on chapters 2-8's
toy agent; the project's file is its full size) and **runs** (the chapter
runs or reads the project's file itself). The whole table, all 28 chapters,
is in the run log.

The footer is the answer to "where does each piece live": chapters 1 to 26
mirror 28 of the project's 41 code files, 9,742 of its 11,460 lines. What
they leave is this chapter's (the project's tests) and the next one's
(`scripts/train_from_scratch.sh`). No chapter mirrors `errors.py` (four
exception classes), `main.py` (a placeholder that prints a greeting) or the
empty `__init__.py` files.

`map.py` reads the project without importing it: Python's own parser finds
each function, class and constant (`ast`), so it works with no environment.
Given a checkout (step 02), it measures it and says whether every file and
name is still there:

```text
map.py · each chapter beside the project code it mirrors
project   ~/sai_work/multi-harness-rl at 25074c0, measured now (OpenEnv 3f14355)
          every mirrored file and name is there; the bundled snapshot (25074c0) agrees
```

A name the checkout no longer has makes `map.py` list it and exit 1, so the
map cannot quietly go stale as the project changes.

### The plan's map, corrected

The course plan drew this map before the chapters were renumbered (the
sandbox moved up to 19, the capture proxy to 20, the memory probe into 25).
`map.py` follows the course as it is: every chapter folder 1 to 26, each
README's own "Mirrors" lines, and every path and name checked in the
checkout (a test checks the map against every README). What changed against
the plan:

| chapter | the plan's map | the map now |
| --- | --- | --- |
| 01 | `grade.py` -> `mhrl/tasks/core.py` | `core.py` (`load_tasks`, `task_instruction`, `prepare_workdir`, `grade`, `check_answer`), plus `generate.py`, `questions.py` and `databases.py`: the generator and the five databases chapter 1 bundles |
| 02-04 | not mapped | the toy parents of `grpo.py`: the reward (`reward_for`, the two temperatures), REINFORCE (`train_grpo`'s `loss_fn` at ratio 1, `token_logprobs`), one reward per rollout (`run_rollout`, `max_model_calls`, `item_from_sequence`) |
| 07 | `token_bonus` -> `rollout.py` | `tool_bonus`, `token_bonus`, `reward_for`, the settings `token_budget` and `length_guard_tokens`, and `CliDefaults.GRPO_LR` |
| 08 | not mapped | the toy parent of `sft.py`: `AnswerKeyTeacher._steps`, `train_sft`'s `loss_fn`, `batches`, `CliDefaults.SFT_LR` |
| 09, 10, 16 | not mapped | `PolicyEngine.render`, the special-token constants, `max_prompt_tokens`, the engine's `chat` route (09); `scripts/setup.sh` and `Settings.model_path` (10); `claude_code.py`'s `PINNED_VERSION`, `command`, `environment` (16) |
| 11 | `convert.py` -> `scripts/setup.sh` + `models/` | `scripts/setup.sh` and `Settings.model_path`: the project never quantizes, it downloads the export chapter 11 builds, and `models/` is not in the repository |
| 12 | sampler -> `engine/policy.py` | `PolicyEngine._sampler`, `_sample` and `_to_float32`, `Defaults.PREFILL_STEP`, `max_completion_tokens` |
| 14-15 | `TrainItem`, `token_logprobs` -> `common.py`, `sft.py`, `grpo.py` | split: 14 is `train_sft`, `demo_items`, `TrainItem`, `item_from_sequence`, `token_logprobs`; 15 is `train_grpo`, `group_advantages`, `reward_for`, `token_bonus`, `token_logprobs` |
| 19 | (with 20) `rollout.py` + OpenEnv | the sandbox: `configs/harnesses/sandbox.sb`, `sandbox.py` (`sandbox_command`, `sandbox_params`, `check_writable_root`), `process.py` (`run_logged`, `kill_tree`), `grade`, `rollout_timeout_s` |
| 20 | (with 19) `rollout.py` + OpenEnv | the capture proxy: `CaptureClient` (`open_session`, `rollout`, `close`) and `serve.sh`'s `start` |
| 21 | `harnesses/sandbox.py`, `process.py`, `sandbox.sb` | calls to rows: OpenEnv's `graph.py`, `export.py`, `validate.py`; `count_calls`, `run_rollout`; `dedupe_shared_prefixes`, `item_from_sequence` |
| 22 | `harnesses/*.py` | silent capture failures: the same capture code, plus `smoke_once` and `REMINDER_OFF` |
| 23 | `evaluate.py` + `cli.balanced` | the three harnesses: `base.py` (`Harness.run`, `base_environment`, `get_harness`), `MiniSweAgent`, `ClaudeCode`, `Pi`, `configs/harnesses/mini_swe_agent.yaml` |
| 24 | `common.py` `chunked_attention` | measuring: `evaluate`, `summarize`, `balanced` and the `eval` command |
| 25 | `engine/oracle.py` + `sft.py` | `AnswerKeyTeacher`; `collect_demonstrations`, `demo_items`, `train_sft`; and, with the memory probe folded in, `chunked_attention`, `load_policy`, `token_logprobs` |
| 26 | `grpo.train_grpo` | `train_grpo`, `group_advantages`; `run_rollout`, `reward_for`, `tool_bonus`; `print_steps` and the `steps` command |

## The project's own tests

```bash
cd ~/sai_work/multi-harness-rl && .venv/bin/python -m pytest -p no:cacheprovider -q
```

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: ~/sai_work/multi-harness-rl
configfile: pyproject.toml
testpaths: tests
plugins: platformdirs-4.12.3, anyio-4.15.1
collected 129 items

tests/test_core.py .............                                         [ 10%]
tests/test_harnesses.py .................................sss.            [ 38%]
tests/test_oracle.py ......                                              [ 43%]
tests/test_tasks.py .................................................... [ 83%]
.....................                                                    [100%]

======================== 126 passed, 3 skipped in 8.86s ========================
```

126 tests in under 9 seconds: the tool-call parser, the rewards, the
advantages, the grader, the task generator and the sandbox's rules (it runs
real commands under `sandbox-exec` and checks what they can and cannot
touch). The 3 skipped are one live test, run once per harness, which needs
`MHRL_LIVE=1` and the stack up (`13-project-skips.txt` in the run log).
`-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` keep pytest from
writing into the checkout, and `MHRL_RUNS_DIR` sends the one folder a test
makes into `COURSE_HOME`. The checkout's `git status` was the same after as
before (two lines, both the owner's own work).

## The stack: `./stack.sh start`

```text
starting the project's servers (engine 8129, teacher 8139, proxy 8209)...
  engine is up (http://127.0.0.1:8129/health)
  teacher is up (http://127.0.0.1:8139/health)
  proxy is up (http://127.0.0.1:8209/health)
capture level: tokens (train rollouts)
```

The engine is the model (1.6 GB once loaded, its own log says); the teacher
loads only the tokenizer; the proxy found that the engine returns token ids
and log-probabilities, so it captures at the **tokens** level, the only one
training can use (chapter 20).

## The smoke test

```bash
"$MHRL_PROJECT/.venv/bin/python" -m mhrl.harnesses.smoke --harness claude-code --runs 1
```

```text
{"harness": "claude-code", "finished": true, "exit_code": 0, "answer_written": false, "answer": "", "correct": false, "turns": 8, "first_prompt_tokens": 2298, "roots": 1, "sequences": 1, "trainable": true, "budget_stops": 1, "seconds": 24.9}
```

One real rollout: a two-table database made on the spot, the pinned Claude
Code in the sandbox, the base model (no adapter), the proxy recording. What
the smoke test checks passed: the program finished by itself and the
capture is one prefix-continuous chain (`roots 1`) with a trainable row.
What the model did is another matter: it used all 8 calls, the proxy's
budget answered the last one (`budget_stops 1`), and it never wrote
`answer.txt`. The first call's prompt was 2,298 tokens, which is the
project's own note in `claude_code.py` ("near 2.3K prompt tokens").

## One real GRPO step: `one_step.py`

```bash
uv run one_step.py --project ~/sai_work/multi-harness-rl --init ~/sai_work/multi-harness-rl/runs/main/sft/adapter.safetensors
```

It runs `train_grpo` from `mhrl/training/grpo.py` itself, for 1 step, 1
task, 1 harness (Claude Code) and a group of 4, as `python -m mhrl.cli grpo`
would with those numbers. It does not copy it. The hooks swap, inside that
one module and only while the step runs, the names `train_grpo` calls:
`run_rollout`, `CaptureClient`, `group_advantages`,
`dedupe_shared_prefixes`, `item_from_sequence`, `load_policy`,
`save_adapter`, `use_adapter`, `nn.value_and_grad` and `optim.AdamW`, and
`get_harness` in `mhrl/rollout.py`. Each wrapper calls the real one and
prints what went in or came out. Those names are the hand-offs.

First, the settings it runs with and where each came from, and three checks
before any rollout:

```text
one_step.py imports the project, so it runs with ~/sai_work/multi-harness-rl/.venv/bin/python
one_step.py · one GRPO step through the project's own code
project   ~/sai_work/multi-harness-rl (mhrl imported from it)
settings  mhrl/config.py Settings; an MHRL_* variable overrides each
      engine_port            8129                                         MHRL_ENGINE_PORT=8129
      proxy_port             8209                                         MHRL_PROXY_PORT=8209
      runs_dir               ~/.cache/mhrl-course/27-how-it-combines/runs MHRL_RUNS_DIR=~/.cache/mhrl-course/27-how-it-combines/runs
      model_path             ~/sai_work/multi-harness-rl/models/LFM2.5-2.6B-MLX-4bit default
      max_model_calls        8                                            default
      train_temperature      1.0                                          default
      max_completion_tokens  1024                                         default
      token_budget           2000                                         default
      length_guard_tokens    150                                          default
      concurrency            4                                            default
engine    http://127.0.0.1:8129 · adapter 'base'
proxy     http://127.0.0.1:8209 · capture level: tokens · forwards to http://127.0.0.1:8129
harness   Claude Code 2.1.290 at ~/sai_work/multi-harness-rl/tools/claude-code/2.1.290/claude
start     ~/sai_work/multi-harness-rl/runs/main/sft/adapter.safetensors
tasks     train_grpo's order: the 180 training tasks shuffled with seed 0
```

The engine must be the project's (its `/health` names an adapter: a server
that does not is refused, which is what keeps the other service on 8100 out
of the step), the proxy must forward to that same engine (else the new
adapter would go to one engine while another samples), and the pinned
Claude Code must be there. The tasks come in `train_grpo`'s own order.

### The task, the model, the starting weights

```text
draw 1 · school-easy-018 (easy) · What was the average grade in the Fall 2024 term? Round to 2 decimal places.
      train_grpo(steps=1, tasks_per_step=1, group_size=4, harnesses=['claude-code'], learning_rate=1e-05)
load_policy  ~/sai_work/multi-harness-rl/models/LFM2.5-2.6B-MLX-4bit + LoRA rank 16, scale 2.0, last 12 layers: 9,781,248 trainable numbers (2.5 s)
use_adapter  POST http://127.0.0.1:8129/admin/adapter {"path": "~/sai_work/multi-harness-rl/runs/main/sft/adapter.safetensors"}
      the engine's /health now says adapter '~/sai_work/multi-harness-rl/runs/main/sft/adapter.safetensors'
```

`load_policy` loads the 4-bit model and puts chapter 13's adapter on it:
9,781,248 numbers that can train. `train_grpo` then tells the engine to
sample with the starting adapter, the project's SFT adapter, and the
engine's `/health` confirms it.

### The fresh folder and the key

```text
[r3] folder ~/.cache/mhrl-course/27-how-it-combines/runs/work/grpo/school-easy-018-mwjv7bhc holds data.db
[r3] API key = the session id sff36880… (POST /sessions on the proxy)
[r1] folder ~/.cache/mhrl-course/27-how-it-combines/runs/work/grpo/school-easy-018-ux73i9ve holds data.db
[r1] API key = the session id s02c56a1… (POST /sessions on the proxy)
[r2] folder ~/.cache/mhrl-course/27-how-it-combines/runs/work/grpo/school-easy-018-oelqww27 holds data.db
[r2] API key = the session id s6b46aaa… (POST /sessions on the proxy)
[r4] folder ~/.cache/mhrl-course/27-how-it-combines/runs/work/grpo/school-easy-018-2brqyyhf holds data.db
[r4] API key = the session id s44a2f82… (POST /sessions on the proxy)
```

Each rollout gets its own folder holding only a copy of the database
(`prepare_workdir`), and its own capture session; the session's id is the
harness's API key (chapter 20). Four run at once, so they start in any
order. The key is printed by its first characters only.

### Claude Code's command, environment and state

```text
[r1] argv  ~/sai_work/multi-harness-rl/tools/claude-code/2.1.290/claude -p
           'There is a SQLite database named data.db in the … (335 characters)' --tools Bash
           --strict-mcp-config --setting-sources '' --settings
           '{"attribution": {"commit": "", "pr": ""}, "includeGitInstructions": false, "bashOutputMaxChars": 4000}'
           --permission-mode bypassPermissions --disable-slash-commands --no-session-persistence --model
           lfm2.5-2.6b --output-format stream-json --verbose
     env   ANTHROPIC_BASE_URL=http://127.0.0.1:8209 ANTHROPIC_API_KEY=<redacted>
           ANTHROPIC_MODEL=lfm2.5-2.6b ANTHROPIC_DEFAULT_HAIKU_MODEL=lfm2.5-2.6b
           CLAUDE_CONFIG_DIR=$TMPDIR/mhrl-harness-claude-code-9goafl9e/claude
           CLAUDE_CODE_TMPDIR=$TMPDIR/mhrl-harness-claude-code-9goafl9e/tmp
           CLAUDE_CODE_MAX_OUTPUT_TOKENS=1024 CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off API_TIMEOUT_MS=600000
           CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
           DISABLE_AUTOUPDATER=1
     state $TMPDIR/mhrl-harness-claude-code-9goafl9e (tempfile.mkdtemp: its HOME, config and TMPDIR; deleted after the run)
```

The command is the project's (`claude_code.py`, chapter 16). The key is not
on it: Claude Code reads its key from the environment, `ANTHROPIC_API_KEY`,
where the hook shows `<redacted>`, as it would for any variable whose name
ends in KEY, TOKEN, SECRET or PASSWORD, and for the key's value wherever it
appears. Claude Code's private state (its HOME, config and temp folder)
goes in a folder `tempfile.mkdtemp` makes under the system's temp folder,
outside the task folder (the layout chapter 24 found necessary), and is
deleted afterwards.

### Each call, and the grade and the reward

```text
     call 1 · prompt 2,324 tokens · sampled 31 · tools 1 · tool_calls
     call 2 · prompt 2,455 tokens · sampled 55 · tools 1 · tool_calls
     call 3 · prompt 2,523 tokens · sampled 37 · tools 1 · tool_calls
     call 4 · prompt 2,577 tokens · sampled 19 · tools 1 · stop
     captured 4 calls · roots 1 · rows 1 trainable of 1 · trainable true
     grade correct (answer '73.89') · reward 1 + 0.1 x (1 - 142/2000) = 1.0929 · 28 s
[r2] captured 6 calls · roots 1 · rows 1 trainable of 1 · trainable true
     grade correct (answer '73.89') · reward 1 + 0.1 x (1 - 257/2000) = 1.0872 · 35 s
[r3] captured 8 calls · roots 1 · rows 1 trainable of 1 · trainable true
     grade wrong (no answer.txt) · reward 0: a wrong answer earns 0 · 45 s
[r4] captured 8 calls · roots 1 · rows 1 trainable of 1 · trainable true
     grade wrong (no answer.txt) · reward 0: a wrong answer earns 0 · 46 s
```

For the first rollout, every call the proxy captured: the prompt it read,
the tokens the model sampled, the tools it was offered, and why the reply
ended. Four calls, one prefix-continuous chain, one trainable row. The
grader accepted `73.89`, and the reward is chapter 7's: 1 for a right
answer plus the token bonus, here 0.1 times (1 - 142 / 2,000). Two rollouts
were right in 4 and 6 calls; two used all 8 calls and never wrote
`answer.txt`, which earns 0.

```text
engine  26 requests · 84,716 prompt tokens, 69,758 of them from its prompt cache (82%) · 1,258 sampled · 44.9 s of model time
```

The capture document has no count of cached tokens per call, so this line
reads the engine's own counters (`/admin/stats`) before and after the
group: 26 requests (4 + 6 + 8 + 8 calls), and 82% of the prompt tokens came
from the engine's prompt cache, because each call's prompt is the last
one's plus a little (chapter 18).

### The group's rewards become advantages

```text
group  rewards 1.0929 1.0872 0.0000 0.0000 · mean 0.5450 · spread 0.5450
       advantages (reward - mean) / spread: +1.01 +0.99 -1.00 -1.00
       r1 reward 1.0929 -> +1.01
       r2 reward 1.0872 -> +0.99
       r3 reward 0.0000 -> -1.00
       r4 reward 0.0000 -> -1.00
```

Chapter 5's arithmetic, by the project's `group_advantages`: each reward
minus the group's mean, divided by its spread. The two right answers push
up, the two failures push down. The shorter right answer (142 tokens) gets
a little more than the longer one (257): the token bonus is the only thing
that tells them apart.

### One row, its mask and the engine's log-probabilities

```text
row    2,596 tokens: 142 the model sampled (mask 1, trained), the other 2,454 only read (mask 0) · advantage +1.01
       position  mask  old logprob  token
          2,321     0               'assistant'
          2,322     0               '\n'
          2,323     0               '<think>'
          2,324     1      -0.2820  'I'
          2,325     1      -0.0044  ' should'
          2,326     1      -0.1083  ' see'
          2,327     1      -0.0538  ' the'
          2,328     1      -4.7657  ' columns'
```

A row is the whole conversation as the model read it, in tokens: 2,596 of
them, of which the 142 the model sampled are trained (mask 1) and the rest,
Claude Code's system prompt, the task, the tool results, are only read
(mask 0, chapter 21). Each trained token carries the log-probability the
engine recorded when it sampled it: `' columns'` was an unlikely pick
(-4.7657), `' should'` an almost certain one (-0.0044).

### The ratios and the loss

```text
rows   4 trained, 1,258 tokens · dedupe_shared_prefixes re-masked 0 repeated tokens
       row 1: 142 ratios, first 1.000 1.000 1.000 1.005 1.000 · min 0.948 max 1.056 mean 1.0003 · loss -142.7615
       row 2: 257 ratios, first 1.024 1.023 1.025 1.003 0.989 · min 0.790 max 1.242 mean 1.0026 · loss -256.2289
       row 3: 427 ratios, first 1.031 1.000 1.000 1.000 1.000 · min 0.883 max 1.236 mean 1.0001 · loss +426.9509
       row 4: 432 ratios, first 1.130 1.005 0.998 0.995 1.016 · min 0.687 max 1.198 mean 0.9978 · loss +431.1323
       loss per token +0.3649
```

The trainer computes each trained token's log-probability again, with the
same weights, and divides: new probability over old, the importance ratio
(chapter 6). Each row's ratios average within 0.003 of 1, and train_grpo's
own mean over all 1,258 tokens is 0.9998 (below). Single tokens spread
further, from 0.687 to 1.242, as in chapter 26's run (0.797 to 1.234), and
the clip at 0.8 and 1.2 bounds how much those count. Each row's loss is minus its advantage times its ratios,
summed: negative for the right answers, which the update makes likelier.

### The update, the file, and the engine

```text
AdamW  one update of 9,781,248 adapter numbers · learning rate 1e-05, bias correction on · largest change 1.00e-05
save   ~/.cache/mhrl-course/27-how-it-combines/one_step/draw1-school-easy-018/adapter_step001.safetensors (39.1 MB)
use_adapter  POST http://127.0.0.1:8129/admin/adapter {"path": "~/.cache/mhrl-course/27-how-it-combines/one_step/draw1-school-easy-018/adapter_step001.safetensors"}
      the engine's /health now says adapter '~/.cache/mhrl-course/27-how-it-combines/one_step/draw1-school-easy-018/adapter_step001.safetensors'
```

One AdamW step over all 9,781,248 adapter numbers. With bias correction on,
the first step moves each number by about the learning rate whatever its
gradient's size (chapter 13): the largest change is 1.00e-05, the learning
rate. The adapter is saved, `train_grpo` POSTs its path to the engine, and
the engine's `/health` names it: the next call the proxy forwards is
sampled with the updated weights.

### The step's own record

```text
[grpo] step 1/1 · 00:01:31 · reward 0.55 · correct 50% · calls 6.5 · no-signal groups 0/1 · ratio 1.000 · peak 7.0 GB
metrics   train_grpo's record: rows 4 · tokens 1258 · rows over the cap 0 · untrainable rollouts 0
      ratio 0.9998 · loss per token +0.3649 · tokens per call 48 · step 91.7 s · peak 7.02 GB
output    ~/.cache/mhrl-course/27-how-it-combines/one_step/draw1-school-easy-018
      the next call the proxy forwards is sampled with ~/.cache/mhrl-course/27-how-it-combines/one_step/draw1-school-easy-018/adapter_step001.safetensors
```

The first line is `train_grpo`'s own; the next two read the line it wrote
to its metrics file: 4 rows, none over the 6,144-token cap, tokens per call
48 (the median rollout, the number the length guard reads; 150 would have
stopped the step, chapter 26), 91.7 s for the step, and an MLX peak of
7.02 GB in the trainer, the weights included.

## A group without contrast

The first group above had contrast, so no second task was drawn. Step 12
shows the other path without the 2.6B: the scripted stack, where the
project's teacher writes the same right answer in all four rollouts:

```text
starting the answer-key teacher as the engine (no model weights) on 8129, proxy 8209...
  engine is up (http://127.0.0.1:8129/health)
  proxy is up (http://127.0.0.1:8209/health)
capture level: tokens (train rollouts)
draw 1 · school-easy-018 (easy) · What was the average grade in the Fall 2024 term? Round to 2 decimal places.
       no contrast: all 4 rewards are equal, so every advantage would be 0
       makes no update.
metrics   train_grpo's record: rows 0 · tokens 0 · rows over the cap 0 · untrainable rollouts 0
no update: drawing the next task in train_grpo's order
no group had contrast in 1 draws; no update was made
exit 0
files in draw1-school-easy-018:
  258898 adapter_final.safetensors
  391 metrics.jsonl
  150866 rollouts.jsonl
adapter_final.safetensors: 24 A matrices, 24 B matrices, 0 B matrices not all zero
stopped proxy (pid 74645)
stopped engine (pid 74538)
```

All four rewards are equal, so every advantage would be 0; `one_step.py`
says so, as chapter 5 taught, and draws the next task in `train_grpo`'s
order (`--max-draws 1` stopped it here). One thing to know about the
folder such a draw leaves: `train_grpo` writes `adapter_final.safetensors`
at the end of every call, update or not. After a draw without contrast it
is the starting weights: here the base model's no-op adapter, every B
matrix zero.

## The article's stack beside ours: `stacks.py`

The left side of each row is the article's own sentences, quoted exactly;
`stacks.py --verify` fetched the article again and found all 10, word for
word (step 04). The right side is the project file that does the same job
here. The article's numbers are the article's; none of them is compared
with ours.

```text
stacks.py · the article's stack beside ours
article   https://huggingface.co/spaces/FineEnvs/multi-harness-rl
          its text fetched 2026-10-13 from https://fineenvs-multi-harness-rl.hf.space/

tasks and sandboxes
   article  "Harbor supplies the tasks and the sandboxes."
   article  "The reported LFM runs used Harbor for both policies and E2B sandboxes."
   article  "The current tutorial uses Daytona and includes native OpenCode as a separate
            comparison, so it is not an exact replay of those historical runs."
   ours     SQLite questions, one fresh folder per rollout; macOS sandbox-exec (mhrl/tasks/core.py,
            configs/harnesses/sandbox.sb)
inference engine
   article  "One H100 trains while a second serves the policy with vLLM (Kwon et al., 2023) and
            keeps producing rollouts, so training never waits for the slowest agent."
   ours     an MLX engine on the same laptop, the 4-bit model plus the adapter
            (mhrl/engine/server.py, mhrl/engine/policy.py)
capture
   article  "OpenEnv is the interface the harness, the environment and the trainer all connect to,
            and its capture proxy records the tokens."
   ours     OpenEnv's capture proxy, unchanged (scripts/serve.sh, pyproject.toml)
trainer
   article  "Async GRPO in TRL for 1,000 steps."
   article  "The price is that rollouts come from weights usually two optimizer steps old, and never
            more than four."
   article  "TRL trains on what the proxy recorded."
   ours     GRPO in MLX, in turn: sample with the last adapter, then update (mhrl/training/grpo.py)
hardware and time
   article  "Two H100s per run on the Hugging Face cluster, and one E2B sandbox per rollout with 1
            CPU and 4 GB of memory."
   article  "Training took about 32 hours for the OpenCode-only run and 46 hours for the
            multi-harness run, not counting evaluation, queueing and restarts."
   ours     one laptop (below)

this machine   Apple M3 Pro, 18 GB
our GRPO run   6 steps of 2 groups of 4, 12.6 minutes of steps, trainer peak 6.88 GB (runs/main/grpo/metrics.jsonl)
```

The same pipeline, at two sizes. The article's runs used Harbor's tasks and
E2B sandboxes (its current tutorial uses Daytona), vLLM on one H100 and
TRL's asynchronous GRPO on another, about 32 hours for the OpenCode-only
run and 46 for the multi-harness run. Here: SQLite tasks, `sandbox-exec`,
and the MLX engine and trainer on one laptop, taking turns. Asynchronous
means the article's rollouts came from weights up to four steps old; ours
come from the adapter the last step saved, which is why our ratio sits near
one. The capture proxy is OpenEnv's in both.

## What the project's code says that it should not

Found while mapping. They are the owner's to fix, in the project; this
chapter leaves the project as it is.

- **A security claim the sandbox does not keep.** `mhrl/harnesses/sandbox.py`
  (`sandbox_params`) says "The gold answers and every other rollout's
  folder are unreadable: reading them would be reward hacking". The profile
  `configs/harnesses/sandbox.sb` denies only the task files: it has no rule
  for other rollouts' folders, and its own comment says a first version
  denied them and broke `getcwd` and Pi's write tool. So a rollout can read
  a neighbour's `answer.txt`; chapter 19 measured that read (`cat
  ../neighbour/answer.txt` printed `30`). The test
  `test_sandbox_hides_gold_answers_and_other_rollouts` in
  `tests/test_harnesses.py` checks the task-file rule only.
- **The README's reward.** At the checkout's commit, `README.md` says the
  reward pays "up to 0.1 for using fewer calls". The code pays for fewer
  generated tokens (`rollout.py`, `reward_for` and `token_bonus`, chapter
  7). The owner's uncommitted copy of the README already says so.
- **`grpo.py`'s docstring**, line 9: "plus at most 0.1 for fewer tool
  calls". Same correction.
- **`oracle.py`'s docstring**, lines 8-9: "a bash block for mini-swe-agent,
  a native tool call for Claude Code". The project's mini-swe-agent
  configuration (`configs/harnesses/mini_swe_agent.yaml`) uses native tool
  calls with one bash tool; the bash block belongs to its text-based
  configuration only.
- **`rollout.py`'s `RolloutRecord`**: "reward: Correctness plus the
  tool-efficiency bonus". It is the token bonus.
- **`scripts/serve.sh`** cannot move to free ports, calls any answer on its
  ports "up", and its stop kills every engine and proxy on the machine (see
  "Run it").

And one this chapter's own run found and fixed: the proxy names each
session in its log in two ways, in its URLs and in its own lines ("session
… hit its model-call budget"), and a session id is a harness's key.
`stack.sh stop` prints the logs' last lines; at `13a3bc9` it hid the first
way only, and from `0ce9d41` both. The log files themselves, under
`COURSE_HOME`, keep the ids.

## Plan versus measured

| the plan said | measured | record |
| --- | --- | --- |
| folder `course/27-how-it-combines/` | `27-how-it-combines/` at the root of this repository, like every chapter | |
| map.py prints every chapter beside the repo file and function it mirrors, with line counts | as planned, for all 26 chapters, this one and the next, from a snapshot or a live checkout; chapters 1-26 mirror 28 of the project's 41 code files, 9,742 of 11,460 lines | `01-`, `02-`, `11-map*.txt` |
| the plan's chapter-to-module list | written before the renumbering: 19-26 corrected, 02-04, 08, 09, 10 and 16 added, 11, 12 and 14-15 made exact (the table above) | the tests |
| (review) map every chapter | done; a test checks the map against every chapter README's "Mirrors" lines | the tests |
| `scripts/serve.sh start` prints the three servers coming up and 'capture level: tokens' | not run: on this Mac it would start the engine on 8100, which another service holds, and call that service's answer "up"; its stop kills every engine and proxy on the machine. `stack.sh start` ran its three commands on 8129, 8139 and 8209: the three servers up and `capture level: tokens (train rollouts)` | `07-start.txt` |
| one_step.py: the fresh folder; the session id as the key | as planned; the key shown by its first 8 characters | `09-one-step.txt` |
| Claude Code's argv (key redacted) | the key is not in the argv: Claude Code takes it from its environment (`ANTHROPIC_API_KEY`). Both are printed, the key redacted in each | `09-one-step.txt` |
| each captured call (prompt, sampled and cached tokens, finish reason) | prompt, sampled, tools offered and finish reason per call. The capture document holds no cached count per call; the cached tokens come from the engine's `/admin/stats` counters around the group: 69,758 of 84,716 prompt tokens (82%) | `09-one-step.txt` |
| the grade and reward (1 + token bonus) | as planned: 1 + 0.1 x (1 - 142/2000) = 1.0929 | `09-one-step.txt` |
| the group's 4 rewards and advantages | 1.0929, 1.0872, 0, 0 -> +1.01, +0.99, -1.00, -1.00 | `09-one-step.txt` |
| one row with its mask and a few old logprobs | 2,596 tokens, 142 trained; the window around the first trained token | `09-one-step.txt` |
| per-token ratios near 1.000 and the loss | each row's mean 0.9978 to 1.0026, the step's 0.9998; single tokens 0.687 to 1.242. Loss per token +0.3649 | `09-one-step.txt` |
| the AdamW update | 9,781,248 numbers, learning rate 1e-05, bias correction on; largest change 1.00e-05 | `09-one-step.txt` |
| adapter_step001.safetensors written and POST /admin/adapter, after which the next call samples with the new weights | as planned (39.1 MB); the engine's `/health` names the new file. Not in the plan: `train_grpo` also writes `adapter_final.safetensors` at the end of every call, and after a draw without contrast it is just the starting weights (every B matrix zero from the base) | `09-one-step.txt`, `12-flat-draw.txt` |
| (review) when a group has no contrast, say so and draw another task | built; this run's first group had contrast, and step 12 shows the other path | `12-flat-draw.txt` |
| `pytest` (parsing, rewards, advantages, grading, sandbox rules) passing in seconds | 126 passed, 3 skipped in 8.86 s; the checkout unchanged | `05-project-tests.txt` |
| the smoke test (one real captured rollout: roots 1, trainable true) | roots 1, trainable true. The base model used all 8 calls and wrote no answer | `08-smoke.txt` |
| (review) fix the three repo strings before recording | not fixed here: the project is the owner's. Listed above, with three more | |
| side panel: Harbor tasks, E2B/Daytona sandboxes, vLLM and TRL's GRPO on two H100s for 32-46 hours | each checked against the article: the reported runs used E2B, the current tutorial Daytona; 32 hours is the OpenCode-only run and 46 the multi-harness run; the GRPO is asynchronous, for 1,000 steps | `04-stacks-verify.txt` |
| the project's ports: 8100, 8101, 8200 | 8129, 8139 and 8209 on this Mac (see "Run it") | `commands.txt` |

## The files

Line counts at commit `0ce9d41`: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `map.py` | 394 | 217 | every chapter beside the project code it mirrors |
| `course_map.py` | 542 | 461 | the map itself: each chapter's files, kind and project files and names |
| `project_scan.py` | 442 | 208 | measures a checkout with `ast`; the snapshot; the differences |
| `one_step.py` | 494 | 290 | one real step of the project's `train_grpo`, drawing again without contrast |
| `hooks.py` | 1,094 | 509 | the hooks at every hand-off, the redaction, the printing |
| `stacks.py` | 313 | 190 | the article's stack beside ours; `--verify` |
| `paths.py` | 109 | 35 | `COURSE_HOME` and the project's checkout |
| `stack.sh` | 217 | | the project's three servers on the ports you choose |
| `test_27_how_it_combines.py` | 971 | 714 | the pinned map, the map against every README, the hooks, the stack script |
| `test_27_project.py` | 191 | 121 | `one_step.py` through the project's real proxy, Claude Code and `train_grpo`, without the 2.6B |
| `one_step_scripted_27.py` | 133 | 70 | runs `one_step.py` with a tiny LFM2 in place of the 2.6B, for the tests |
| `data/project_snapshot.json`, `data/project_grpo_metrics.jsonl` | | | the snapshot at `25074c0`; the project's recorded GRPO metrics |

## What this chapter leaves out

- **More than one step.** One step shows every hand-off once; the project's
  own 6-step run took 12.6 minutes of steps (`stacks.py`), and chapter 28
  runs the whole pipeline.
- **mini-swe-agent and Pi.** The project trains in two harnesses and
  evaluates in three. One harness is enough to follow one step; chapter 23
  showed all three.
- **Fixing the project.** The findings above are the owner's to make; this
  chapter only reads and runs the project.

## Mirrors in the project

`mhrl/config.py` (`Settings`, `Defaults`, `get_settings`), `mhrl/cli.py`
(`main`, `CliDefaults`), `mhrl/training/grpo.py` (`train_grpo`,
`use_adapter`), `mhrl/rollout.py` (`run_rollout`), `scripts/serve.sh`,
`mhrl/harnesses/smoke.py` (`smoke_once`) and `tests/`.
