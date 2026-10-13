# Chapter 26. GRPO through the harnesses, and the run that went wrong

Chapter 15 ran GRPO on the real model in a loop we owned: one question, one
reply, one model. Chapter 25 taught the model inside the real harnesses with
SFT. This chapter does what the project does next: GRPO through
mini-swe-agent and Claude Code. Every call the harness makes is sampled at
temperature 1 and recorded by the capture proxy, the grader scores each
attempt, and one clipped update per step moves the adapter, which the engine
then samples with.

Then it opens the project's two recorded runs. The first went wrong: its
replies grew longer step by step until they stopped calling tools at all,
and the adapter the project kept scored 0 of 24 in Claude Code. The puzzle:
that run's own training log looked fine for four steps. Which number moved
first, and would a guard watching it have saved the run? And when a step
trains on two harnesses, does each harness teach the model the same amount?

## Words first

- A **call** is one model request inside an agent, a **rollout** one whole
  attempt at a task, and a **step** one optimizer update (the course's words
  since chapter 1).
- **GRPO through the harnesses**: each step takes 2 training tasks and gives
  each ONE harness, alternating mini-swe-agent and Claude Code, so the 4
  rollouts of a task (its **group**) differ only in what the model did,
  never in the harness. Each rollout is graded and rewarded, the group's
  rewards become advantages (chapter 5), and one clipped update (chapter 6)
  is made on the tokens the engine sampled, against the log-probabilities
  the engine recorded as it sampled them (chapter 15).
- The **reward** is 1 plus a **token bonus** of up to 0.1 for a right
  answer (the whole bonus at 0 generated tokens, none at 2,000), and 0 for a
  wrong one (chapter 7). The project's first run paid the article's **call
  bonus** instead: up to 0.1 for fewer tool calls (the whole bonus at 2
  calls, none at the 8-call budget).
- A **no-contrast group** is a group whose 4 rewards are all equal: nothing
  to compare, so it is skipped (chapter 5).
- **Sampled tokens** of a rollout: the tokens the model generated across all
  its calls (the project's `n_sampled`).
- **Tokens per call** of a rollout: its sampled tokens divided by its calls.
  A step has 8 rollouts, so 8 such numbers, and this chapter shows two
  summaries of them side by side: the **median** (the middle of the 8) and
  the **mean** (their average). One rollout that balloons moves the mean;
  the median moves only when most of them do. **The length guard reads the
  median**, as the project's `train_grpo` does.
- The **length guard** checks each step's rollouts before its update: if
  their tokens per call (the median) pass 150, the last update is the one
  that made the replies balloon, so training stops and keeps the adapter
  from **before** that update. Going back to an earlier adapter is a
  **rollback**; choosing which saved adapter to keep is **checkpoint
  selection**.
- **Each step's rollouts are sampled with the adapter the step before
  saved.** So step 5's line describes the adapter step 4 made.
- **Correct** is the share of a step's 8 rollouts whose answer was right;
  **calls** the mean of their calls; the **ratio** the trainer's mean
  importance ratio over every token it trained on; **peak** the trainer's
  MLX peak memory, the weights included.
- **Harness exposure** is how much each harness actually gave the updates:
  its rows and their trained tokens. A harness whose groups have no
  contrast gives nothing, however many steps it appears in.
- A **leading indicator** is a number that moves before the one you care
  about. A **noisy training curve** is one whose step-to-step swings are
  mostly luck: with 8 rollouts per step, one rollout is 12.5 points.

## Run it

```bash
cd multi-harness-RL-course/26-grpo-in-harnesses
uv run replay.py attempt1             # the project's first run, step by step, from its records
uv run replay.py attempt2             # its second run, the one the guard watched
uv run grpo_harness.py --guard-demo   # attempt 1's rollouts through the length guard
./start.sh install                    # the capture proxy, into its own environment
uv run install.py mini-swe-agent claude-code   # the two training harnesses
./start.sh                            # the model's engine and the capture proxy
uv run grpo_harness.py --steps 2      # two GRPO steps from chapter 25's SFT adapter
uv run steps.py                       # what each step's rollouts did
./start.sh stop
```

`grpo_harness.py` starts from chapter 25's adapter trained as the project's
SFT run was (`train_sft.py --no-bias-correction`, in
`COURSE_HOME/25-sft-in-harnesses/sft-no-bias-correction/`); `--init` names
another, `--init base` none. `--attempt1` trains with the first run's
settings into its own folder. `replay.py --records
~/multi-harness-rl/runs/main` reads the project's own files instead of the
bundled copies, and prints the same.

The project runs its engine on 8100 and its proxy on 8200. On this Mac 8100
belongs to another service and another chapter's film used 8119 and 8200,
so this run used `ENGINE_PORT=8129` and `PROXY_PORT=8209` (and
`--proxy-url http://127.0.0.1:8209`).

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
22.19 or later (for Claude Code) and the network for the installs.

## The run log

Every output below is copied from the run log, made from a clean
`COURSE_HOME/26-grpo-in-harnesses` on an M3 Pro with 18 GB at commit
`0e04328`, everything under `nice -n 10`. The steps without the model did
not wait. Before the two steps that load the model, a gate waited until
nobody had touched the keyboard or mouse for 10 minutes, nothing had
rendered for 120 s, memory pressure was normal and at least 40% of memory
was free: it waited 3,727 s. During the training, free memory was read every
5 s, ready to stop the run under 15% (the lowest it saw was 24%).

"Memory" in the table is the largest single process `/usr/bin/time` saw.
It does not count the Metal buffers MLX allocates for the model: MLX's own
peak is printed by `grpo_harness.py` and by the engine when it stops, and
those are the figures quoted below.

| step | command | what it shows | exit · seconds · memory |
| --- | --- | --- | --- |
| 00 | `rm -rf ~/.cache/mhrl-course/26-grpo-in-harnesses` | a clean start | exit 0 · 0.0 s · 0.00 GB |
| 01 | `uv run replay.py attempt1` | the project's first run, from its records | exit 0 · 0.1 s · 0.03 GB |
| 02 | `uv run replay.py attempt2` | its second run | exit 0 · 0.1 s · 0.03 GB |
| 03 | `uv run grpo_harness.py --guard-demo` | attempt 1 through the length guard | exit 0 · 1.1 s · 0.09 GB |
| 04 | `./start.sh install` | the capture proxy, into its own environment | exit 0 · 3.6 s · 0.16 GB |
| 05 | `uv run install.py mini-swe-agent claude-code` | mini-swe-agent and Claude Code at their pins | exit 0 · 31.8 s · 0.18 GB |
| 06 | `uv run pytest tests/test_26_grpo_in_harnesses.py tests/test_26_mlx.py` | the chapter's tests (54 passed) | exit 0 · 15.1 s · 0.36 GB |
| 07 | `ENGINE_PORT=8129 PROXY_PORT=8209 ./start.sh` | the model's engine (1.6 GB loaded) and the proxy | exit 0 · 14.3 s · 0.11 GB |
| 08 | `uv run grpo_harness.py --steps 2 --proxy-url http://127.0.0.1:8209` | two GRPO steps (5 min 22 s by its own clock; the Mac slept 47 min) | exit 0 · 3,024.4 s · 1.04 GB |
| 09 | `uv run steps.py` | what each step's rollouts did | exit 0 · 0.1 s · 0.03 GB |
| 10 | `./start.sh stop` | the engine's peak memory: 3.29 GB | exit 0 · 2.1 s · 0.00 GB |

Two minutes into the training the Mac's lid was closed, on battery: its
power log (`08-grpo-sleep.txt` beside the run log) records a clamshell sleep
at 12:29:01 and the wake at 13:15:44, with three short dark wakes between.
Step 1 had finished; step 2's rollouts were paused for 47 minutes and went
on after the wake. So `/usr/bin/time` counted 3,024.4 s, while the script's
own clock, which does not run while the Mac sleeps, counted 5 min 22 s for
both steps. One rollout was changed by it: mini-swe-agent keeps its own
wall-clock limit, which did count the sleep, and it stopped step 2's second
rollout after 81.8 s of running ("mini-swe-agent's own wall-time limit"),
wrong. Nothing else in the records shows the sleep.

## Two GRPO steps through the harnesses: `grpo_harness.py --steps 2`

`./start.sh` started the engine (1.6 GB of weights) and the capture proxy
in front of it. Then:

```text
grpo_harness.py · GRPO through the harnesses
  start     ~/.cache/mhrl-course/25-sft-in-harnesses/sft-no-bias-correction/adapter.safetensors
  steps     2, each 2 training tasks x 4 rollouts = 8, harnesses alternating (mini-swe-agent, claude-code), task order seed 0
  rollouts  temperature 1 through the proxy, at most 8 calls, 4 at a time
  reward    1 + the token bonus (up to 0.1, none at 2,000 tokens) when right, else 0
  update    clipped loss (0.8 to 1.2) on the engine's recorded log-probabilities · AdamW lr 1e-05, bias correction on · rows over 6,144 tokens skipped · each shared prefix trained once
  guard     stop when tokens per call (the median rollout) pass 150, keep the adapter from before the last update
  out       ~/.cache/mhrl-course/26-grpo-in-harnesses/grpo/grpo
  engine    http://127.0.0.1:8129 (the model) · proxy http://127.0.0.1:8209
versions
  mini-swe-agent 2.4.6, asked inside the sandbox · pinned 2.4.6 · ok
  claude-code    2.1.290, asked inside the sandbox · pinned 2.1.290 · ok
  model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours (ours, built by chapter 11's convert.py) · 1.62 GB in MLX
  the engine samples with ~/.cache/mhrl-course/25-sft-in-harnesses/sft-no-bias-correction/adapter.safetensors
  [1/8] mini-swe-agent school-easy-018    right · reward 1.09 · 5 calls · 198 sampled (40 per call) · 1 row, 1,017 tokens · 21.9 s · finish command
  [2/8] mini-swe-agent school-easy-018    right · reward 1.09 · 5 calls · 216 sampled (43 per call) · 1 row, 964 tokens · 22.8 s · finish command
  [3/8] mini-swe-agent school-easy-018    right · reward 1.09 · 4 calls · 168 sampled (42 per call) · 1 row, 1,056 tokens · 26.4 s · finish command
  [4/8] mini-swe-agent school-easy-018    right · reward 1.09 · 7 calls · 299 sampled (43 per call) · 1 row, 1,302 tokens · 47.6 s · finish command
  [5/8] claude-code    shop-hard-005      wrong · reward 0.00 · 8 calls · 379 sampled (47 per call) · 1 row, 3,240 tokens · 58.5 s · 8-call budget
  [6/8] claude-code    shop-hard-005      wrong · reward 0.00 · 8 calls · 448 sampled (56 per call) · 1 row, 4,306 tokens · 58.3 s · 8-call budget
  [7/8] claude-code    shop-hard-005      wrong · reward 0.00 · 7 calls · 490 sampled (70 per call) · 2 rows, 6,714 tokens · 61.3 s · 8-call budget
  [8/8] claude-code    shop-hard-005      wrong · reward 0.00 · 7 calls · 271 sampled (39 per call) · 2 rows, 5,315 tokens · 43.0 s · 8-call budget
[grpo] step 1/2 · 00:01:42 · reward 0.54 · correct 50% · calls 6.4 · no-signal groups 1/2 · ratio 1.000 · peak 3.1 GB
       tokens/call: median 43 (the guard reads this; limit 150) · mean 47
       rows: claude-code no row (its group had no contrast) · mini-swe-agent 4 rows, 1,085 tokens on average, 881 trained · 0 over 6,144 tokens skipped
       ratio per token 0.863 to 1.131, 0.0% outside 0.8 to 1.2 · loss/token 0.2119 · adapter_step001.safetensors, now the engine's
  [1/8] mini-swe-agent school-easy-014    right · reward 1.09 · 4 calls · 139 sampled (35 per call) · 1 row, 843 tokens · 21.4 s · finish command
  [2/8] mini-swe-agent school-easy-014    wrong · reward 0.00 · 5 calls · 231 sampled (46 per call) · 1 row, 982 tokens · 24.1 s · finish command
  [3/8] mini-swe-agent school-easy-014    wrong · reward 0.00 · 6 calls · 263 sampled (44 per call) · 2 rows, 7,766 tokens · 81.8 s · mini-swe-agent's own wall-time limit
  [4/8] mini-swe-agent school-easy-014    wrong · reward 0.00 · 8 calls · 376 sampled (47 per call) · 1 row, 1,601 tokens · 85.5 s · step limit
  [5/8] claude-code    school-medium-012  wrong · reward 0.00 · 5 calls · 257 sampled (51 per call) · 2 rows, 5,168 tokens · 159.0 s · 8-call budget
  [6/8] claude-code    school-medium-012  wrong · reward 0.00 · 5 calls · 214 sampled (43 per call) · 1 row, 2,789 tokens · 157.1 s · 8-call budget
  [7/8] claude-code    school-medium-012  wrong · reward 0.00 · 5 calls · 280 sampled (56 per call) · 1 row, 3,541 tokens · 100.7 s · no tool call
  [8/8] claude-code    school-medium-012  wrong · reward 0.00 · 7 calls · 309 sampled (44 per call) · 1 row, 4,315 tokens · 99.5 s · 8-call budget
[grpo] step 2/2 · 00:05:20 · reward 0.14 · correct 12% · calls 5.6 · no-signal groups 1/2 · ratio 1.002 · peak 5.0 GB
       tokens/call: median 45 (the guard reads this; limit 150) · mean 46
       rows: claude-code no row (its group had no contrast) · mini-swe-agent 5 rows, 2,238 tokens on average, 1,009 trained · 0 over 6,144 tokens skipped
       ratio per token 0.797 to 1.234, 0.3% outside 0.8 to 1.2 · loss/token 0.2599 · adapter_step002.safetensors, now the engine's
done      2 steps in 5 min 22 s · trainer peak 4.96 GB (MLX's count, the weights included)
adapter   ~/.cache/mhrl-course/26-grpo-in-harnesses/grpo/grpo/adapter_final.safetensors (= adapter_step002.safetensors)
records   ~/.cache/mhrl-course/26-grpo-in-harnesses/grpo/grpo: rollouts.jsonl, metrics.jsonl (uv run steps.py reads them)
rows by harness, every update together:
  claude-code      0 rows · no row · 0 tokens trained
  mini-swe-agent   9 rows · 1,726 tokens on average · 1,890 tokens trained
```

- **The setup** says every rule the run trains by, before step 1. The model
  starts from chapter 25's adapter, and the script tells the engine to
  sample with the same file, so the trainer's copy of the model and the
  engine's hold the same weights.
- **Each rollout** is a real harness in the sandbox, at temperature 1, its
  calls recorded by the proxy. The lines arrive as rollouts end, 4 at a
  time; the step still keeps them in job order.
- **Step 1.** mini-swe-agent solved its task 4 times out of 4 and Claude
  Code none of 4, each Claude Code rollout stopped by the 8-call budget.
  Claude Code's group has no contrast and is skipped. mini-swe-agent's
  group, all right, still trains, because the token bonus makes four right
  answers differ, and the advantages divide by the group's spread however
  small it is (`14-step1-advantages.txt`, the run's records through
  `rewards.py`):

  ```text
    sampled tokens      216      168      299      198
    reward           1.0892   1.0916   1.0851   1.0901
    advantage         +0.08    +1.03    -1.56    +0.44
  ```

  Rewards 0.0065 apart become advantages as large as a right answer's over
  a wrong one: the 299-token reply is pushed down and the 168-token reply up
  about as hard as in a group of right and wrong answers.
- **Step 2.** One mini-swe-agent rollout of 4 was right, and Claude Code
  again none. Again only mini-swe-agent's rows train.
- **Tokens per call** stayed at 43 and 45 (median; means 47 and 46), far
  under the guard's 150.
- **The ratio** averaged 1.000 and 1.002. Single tokens ranged from 0.863 to
  1.131 and from 0.797 to 1.234, and 0.3% of step 2's tokens fell outside
  the clip's 0.8 to 1.2 although nothing was stale: the bfloat16 routes
  chapter 15 measured.
- **Memory.** The trainer's MLX peak was 3.1 GB at step 1 and 4.96 GB at
  step 2, whose mini-swe-agent rows were longer (2,238 tokens on average,
  the longest 3,952). The engine's peak was 3.29 GB when it stopped, so the
  two copies of the model held at most about 8.3 GB together.
- **The harnesses did not train equally, here too.** All 8 Claude Code
  rollouts were wrong, so every row both updates learned from came from
  mini-swe-agent: 9 rows, 1,890 trained tokens, against none from Claude
  Code. A Claude Code row averaged 3,217 tokens and a mini-swe-agent row
  1,726 (`13-row-lengths.txt`). The project's attempt 2, on the same two
  tasks from the same kind of SFT adapter, also had Claude Code at 0 of 4
  in both steps (`replay.py attempt2`, below).

`steps.py` reads the run back, the project's `mhrl.cli steps` with both
statistics:

```text
steps.py · ~/.cache/mhrl-course/26-grpo-in-harnesses/grpo/grpo
step  correct  calls  tokens/call: median   mean  rollout s (median)
   1      50%    6.4                   43     47                  48
   2      12%    5.6                   45     46                 100
tokens/call: a rollout's sampled tokens / its model calls; the median and the mean
  over the step's rollouts. The length guard reads the median.
```

Every time in these outputs is the script's own clock (Python's
`perf_counter`, which on macOS stops while the Mac sleeps), so none of them
holds the 47 minutes of sleep.

## The run that went wrong: `replay.py attempt1`

The project's first GRPO run, attempt 1, made 8 steps. Its files are
bundled in `data/project_grpo/attempt1/`: the trainer's metrics unchanged,
and the 64 rollouts cut to numbers (each row's length and where each reply
sits in it; no token ids, no text, no paths).

```text
replay.py attempt1 · the project's GRPO run, from its recorded files
  records   data/project_grpo/attempt1 (the project's runs/main/grpo_attempt1, slimmed)
            8 steps x 8 rollouts = 64 rollouts, each step 2 tasks x 4
  settings  the article's call bonus (every recorded reward reproduces with it)
            lr 4e-5 without AdamW's bias correction (mhrl/cli.py's comment)
            no length guard
            bfloat16 log-probabilities in the engine (chapter 15)

  step  correct  reward  calls  tokens/call    replies at  no-signal   ratio     peak
                                median  mean  the 1,024 cap     groups
     1      38%    0.40    5.9      46    49     0 of 47          1/2   0.996  3.75 GB
     2      38%    0.39    5.5      38    38     0 of 44          0/2   0.997  4.32 GB
     3      62%    0.65    6.8      40    41     0 of 54          0/2   0.996  4.32 GB
     4      50%    0.52    4.6      45    57     0 of 37          1/2   1.000  4.32 GB
     5      25%    0.26    3.0     130   134     0 of 24          1/2   0.995  4.32 GB
     6      25%    0.26    4.1     111   120     0 of 33          1/2   0.997  4.32 GB
     7      12%    0.13    3.4     492   465     7 of 27          1/2   0.998  4.36 GB
     8      12%    0.13    3.8     562   654    15 of 30          1/2   0.999  4.36 GB
  correct      the share of the step's 8 rollouts whose answer was right
  tokens/call  a rollout's sampled tokens / its model calls; the median and the mean
               over the step's 8 rollouts. The length guard reads the median.
               These records predate n_sampled: a rollout's sampled tokens are
               counted from its captured rows, each shared prefix once (on attempt 2,
               which has both, the count equals n_sampled for all 48 rollouts).
  replies      the model's replies that reached the 1,024-token cap, of all its replies
  ratio        the trainer's mean importance ratio over the tokens it trained on; peak:
               the trainer's MLX peak memory (both from metrics.jsonl)
  each step's rollouts were sampled with the adapter the step before saved
  of the 22 replies at the cap, 21 hold no tool call: the cap cut them off before it

  what the project's `python -m mhrl.cli steps` printed (runs/logs/take3_steps.log):
    46 38 40 57 127 105 365 404
    reproduced from the records: the same numbers. It is the mean, never the median
    and, for records without n_sampled, only each rollout's FIRST captured row.
    Every row counted, the mean is 49 38 41 57 134 120 465 654;
    the median, what the guard reads, 46 38 40 45 130 111 492 562.

  right rollouts per step, by harness:
    claude-code    0/4 2/4 4/4 0/4 0/4 0/4 0/4 0/4
    mini-swe-agent 3/4 1/4 1/4 4/4 2/4 2/4 1/4 1/4

  what each harness gave the updates (groups with contrast only, each shared prefix once):
    harness         contrast  rows  tokens per row  tokens trained
    claude-code     2 of 8      10           2,653    1,510 (7%)
    mini-swe-agent  8 of 8      54           1,265   19,478 (93%)
    every step counts both harnesses alike; the tokens they gave do not
  ratio 0.995 to 1.000 over the 8 steps; trainer peak 4.36 GB (step 7)

━━ the adapter the project kept and evaluated: adapter_step004.safetensors
  (summary_rl_attempt1.json names runs/main/grpo/adapter_step004.safetensors)
  that folder was attempt 1's, before it was renamed grpo_attempt1. adapter_step004
  is the adapter step 5's rollouts were sampled with. pi is left out below: this
  adapter was evaluated once, before chapter 19's sandbox fix, and never again.

━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair                                     harness         cells  before  after  gained  lost  flipped  p      reading
  project sft -> attempt1 adapter_step004  mini-swe-agent     24      18     20       5     3        8  0.727  within noise
  project sft -> attempt1 adapter_step004  claude-code        24      17      0       0    17       17  0.000  lost: beyond noise
  p < 0.05 reads as beyond noise; anything else, within it
```

Read it a column at a time.

- **Tokens per call, both ways.** For four steps both stay near 40 to 57.
  At step 5 the median jumps to 130 and the mean to 134; at step 8 they are
  562 and 654. The guard reads the median.
- **The numbers in the project's log are a third count.** Its
  `python -m mhrl.cli steps` printed 46, 38, 40, 57, 127, 105, 365 and 404,
  the series the plan quotes. They come back exactly from the records, and
  they are the mean, counted from each rollout's first captured row only:
  these records are older than the `n_sampled` field, and a rollout whose
  harness forked its history has more than one row. Counted from every row
  (each shared prefix once), the mean is 49, 38, 41, 57, 134, 120, 465 and
  654, so 365 and 404 understate steps 7 and 8. The every-row count is the
  right one: on attempt 2, which records both, it equals `n_sampled` on all
  48 rollouts (the tests check each one).
- **Calls fell while replies grew.** Calls per rollout went from 5.9 at
  step 1 to 3.0 at step 5, the lowest point; step 8 was 3.8. The call bonus
  paid for fewer calls, and the model found it could make fewer calls by
  saying more in each.
- **Then the replies stopped reaching their tool call.** From step 7 on,
  replies reached the 1,024-token cap: 7 of 27 at step 7, 15 of 30 at step 8,
  and 21 of those 22 hold no tool call. The cap cut them off before it.
- **Correct rollouts** went from 38% to 12% at steps 7 and 8. Claude Code's
  last right rollouts were at step 3 (4 of 4); from step 4 on it got none.
- **The ratio** stayed between 0.995 and 1.000: the training itself was
  sound by that measure. Its means sit just under 1 because this run's
  engine recorded bfloat16 log-probabilities (chapter 15). The ratio says
  nothing about where the policy is heading.

The adapter the project kept is `adapter_step004`. No record says why step 4
was chosen. What the records do say: `adapter_step004` is the adapter step
5's rollouts were sampled with, the first ballooned step (median 130, 25%
correct, Claude Code 0 of 4). Evaluated on 24 held-out tasks per harness,
it lost all 17 Claude Code tasks the SFT checkpoint solved (17 to 0,
p < 0.001; Claude Code only), while mini-swe-agent went 18 to 20, within
noise. Pi is left out: this adapter was evaluated once, before chapter 19's
sandbox fix, and its Pi cells measured the sandbox, not the model.

## Would the guard have caught it? `grpo_harness.py --guard-demo`

The same records, fed step by step to the length guard the training loop
calls (`grpo_records.length_guard`):

```text
grpo_harness.py --guard-demo · attempt 1's rollouts, step by step, past the guard
  records   data/project_grpo/attempt1 (the project's runs/main/grpo_attempt1, slimmed)
            8 steps x 8 rollouts
  guard     after a step's rollouts, before its update: stop when tokens per call pass 150,
            and keep the adapter from before the last update
  tokens    per call: a rollout's sampled tokens / its calls; the median and
            the mean over the step's rollouts. The guard reads the median.

  step  sampled with      median   mean  correct  the guard
     1  the SFT adapter       46     49      38%  under 150: update 1 -> adapter_step001
     2  adapter_step001       38     38      38%  under 150: update 2 -> adapter_step002
     3  adapter_step002       40     41      62%  under 150: update 3 -> adapter_step003
     4  adapter_step003       45     57      50%  under 150: update 4 -> adapter_step004
     5  adapter_step004      130    134      25%  under 150: update 5 -> adapter_step005
     6  adapter_step005      111    120      25%  under 150: update 6 -> adapter_step006
     7  adapter_step006      492    465      12%  STOP: adapter_step006 made these; keep adapter_step005

  stopped at step 7, before update 7: adapter_final = adapter_step005
  (the run itself went on to make 8 updates)
  the longest replies it let through: median 130 at step 5, 2.8 times step 1's
  the project kept adapter_step004, the adapter step 5's rollouts came from
  (replay.py attempt1 shows how it scored)
  limits compared on the same records:
    --guard-tokens 150  stops at step 7, keeps adapter_step005
    --guard-tokens 100  stops at step 5, keeps adapter_step003
    --guard-tokens 60   stops at step 5, keeps adapter_step003
```

At 150 the guard stops at step 7, before update 7, and keeps
`adapter_step005`. It is not a healthy adapter: its own rollouts (step 6)
were 25% correct at a median of 111 tokens per call. The guard caught the
blow-up, not the drift: it let step 5's 130 through, 2.8 times step 1's.
At 100 it stops at step 5 and keeps `adapter_step003`, blaming
`adapter_step004`, the very adapter the project kept.

The project's code stops at the guard and keeps the adapter in its final
file; this chapter's also tells the engine to sample with the kept adapter
again, so the next rollout comes from it.

## The run that held: `replay.py attempt2`

The second attempt changed four things: the token bonus in place of the
call bonus, learning rate 1e-5 with AdamW's bias correction, the length
guard, and float32 log-probabilities in the engine.

```text
replay.py attempt2 · the project's GRPO run, from its recorded files
  records   data/project_grpo/attempt2 (the project's runs/main/grpo, slimmed)
            6 steps x 8 rollouts = 48 rollouts, each step 2 tasks x 4
  settings  the token bonus (every recorded reward reproduces with it)
            lr 1e-5 with AdamW's bias correction
            the length guard at 150 tokens per call
            float32 log-probabilities in the engine

  step  correct  reward  calls  tokens/call    replies at  no-signal   ratio     peak
                                median  mean  the 1,024 cap     groups
     1      38%    0.41    6.5      46    51     0 of 52          1/2   1.000  3.06 GB
     2      25%    0.27    6.5      40    43     0 of 52          1/2   1.001  3.29 GB
     3      62%    0.68    6.0      43    43     0 of 48          0/2   0.999  3.92 GB
     4      75%    0.82    5.6      37    40     0 of 45          0/2   1.001  6.88 GB
     5      50%    0.54    6.1      45    50     0 of 49          0/2   1.000  6.88 GB
     6      25%    0.27    6.5      43    45     0 of 52          0/2   1.001  6.88 GB
  correct      the share of the step's 8 rollouts whose answer was right
  tokens/call  a rollout's sampled tokens / its model calls; the median and the mean
               over the step's 8 rollouts. The length guard reads the median.
  replies      the model's replies that reached the 1,024-token cap, of all its replies
  ratio        the trainer's mean importance ratio over the tokens it trained on; peak:
               the trainer's MLX peak memory (both from metrics.jsonl)
  each step's rollouts were sampled with the adapter the step before saved

  what the project's `python -m mhrl.cli steps` printed (runs/logs/take4_steps.log):
    51 43 43 40 50 45
    reproduced from the records: the same numbers. It is the mean, never the median
    the guard read the median: 46 40 43 37 45 43.

  right rollouts per step, by harness:
    claude-code    0/4 0/4 4/4 2/4 2/4 1/4
    mini-swe-agent 3/4 2/4 1/4 4/4 2/4 1/4

  what each harness gave the updates (groups with contrast only, each shared prefix once):
    harness         contrast  rows  tokens per row  tokens trained
    claude-code     4 of 6      18           3,061    4,472 (42%)
    mini-swe-agent  6 of 6      30           1,275    6,122 (58%)
    every step counts both harnesses alike; the tokens they gave do not
  ratio 0.999 to 1.001 over the 6 steps; trainer peak 6.88 GB (step 4)
  8 rollouts per step, on other tasks every step: one rollout is 12.5 points,
  and at 50% the standard error is 17.7 points. Per-step accuracy is no progress meter;
  tokens per call is steadier.

━━ the adapter the project kept and evaluated: adapter_final.safetensors
  (summary_rl.json names runs/main/grpo/adapter_final.safetensors)

━━ every pair of checkpoints, cell by cell (exact McNemar p)
  pair                                   harness         cells  before  after  gained  lost  flipped  p      reading
  project sft -> attempt2 adapter_final  mini-swe-agent     24      18     20       2     0        2  0.500  within noise
  project sft -> attempt2 adapter_final  claude-code        24      17     18       2     1        3  1.000  within noise
  project sft -> attempt2 adapter_final  pi                 24       9      9       5     5       10  1.000  within noise
  p < 0.05 reads as beyond noise; anything else, within it
  churn: pi project sft -> attempt2 adapter_final: 10 of 24 cells flipped, 10 of them cancel out: score +0
```

- **Tokens per call stayed put**: medians 46, 40, 43, 37, 45 and 43 (what
  the guard read, `metrics.jsonl`'s `tokens_per_call`), means 51, 43, 43,
  40, 50 and 45 (what the project's `steps` printed). The guard never came
  near 150.
- **Accuracy per step swung from 25% to 75% and back** with the replies
  unchanged: 38, 25, 62, 75, 50 and 25%. Each step is 8 rollouts on two new
  tasks, so one rollout is 12.5 points, and at 50% the standard error is
  about 17.7 points. A step's accuracy is no progress meter.
- **Ratio 0.999 to 1.001, trainer peak 6.88 GB** at step 4, the step whose
  rows were longest.
- **The kept adapter** (`adapter_final`, step 6's update) against SFT, on 24
  held-out tasks per harness: mini-swe-agent 18 to 20, Claude Code 17 to 18,
  Pi 9 to 9 (10 cells flipped, 10 cancelling out). Every change is within
  noise.

## What the records say

The plan's lesson was "reply length moves before accuracy does". The
records support a narrower one, and a second finding the plan does not
name.

- **Length and accuracy moved at the same step.** In attempt 1 the median
  tokens per call held at 38 to 45 for four steps and jumped at step 5, the
  step accuracy fell from 50% to 25%. Only the mean rose a step earlier (41
  to 57 at step 4), and so did the calls' fall (6.8 to 4.6); one ballooned
  rollout is enough to move a mean.
- **Length is the readable signal; accuracy is not.** 8 rollouts per step
  swing about ±17.7 points by luck (attempt 2: 25% to 75% with the replies
  unchanged), while the median tokens per call reached nearly 3 times step
  1's at step 5 and 12 times by step 8. A guard on length can act on one step; a guard on accuracy
  would fire on noise.
- **The harnesses did not train the model equally.** In attempt 1, Claude
  Code's groups had contrast in only 2 of 8 steps (its 4 rollouts were all
  wrong in the other 6), so it gave 7% of the trained tokens and
  mini-swe-agent 93%. Every step counted both harnesses alike, and the
  collapse showed up in the harness that barely trained: 0 of 24 in Claude
  Code. In attempt 2 the split was 42% and 58%, and a Claude Code row was
  2.4 times as long as a mini-swe-agent row (3,061 tokens against 1,275):
  the trainer read more Claude Code tokens and learned from fewer. Step
  counts hide how much each harness actually trained the model.

## The article's runs

> **The article's own numbers** (Hugging Face and Liquid AI, "The ultimate
> guide to multi-harness RL", 2026; its Qwen3.5-2B runs, trained with
> correctness as the only reward, on its own tasks):
>
> "Multi-harness more than doubled, from 14.6% to 37.0% at step 500."
> "OpenCode-only peaked at 39.5% at step 700."
> "Then both fell back to about 26% by step 1,000."

The sentences were checked against the article itself
(`https://fineenvs-multi-harness-rl.hf.space/`, the Space at
`https://huggingface.co/spaces/FineEnvs/multi-harness-rl`). Those runs rose
and then fell over 1,000 steps on another model and other tasks; attempt 1
rose and fell in 8. None of the article's scores sits beside ours.

## Plan versus measured

| the plan said | measured | record |
| --- | --- | --- |
| folder `course/26-grpo-in-harnesses/` | `26-grpo-in-harnesses/` at the root of this repository, like every chapter | |
| grpo_harness.py: 2 training tasks per step, one harness each (alternating), 4 rollouts per task at temperature 1 through the proxy, rewarded 1 + token bonus when correct; no-contrast groups skipped, shared prefixes deduplicated, rows over 6,144 tokens dropped; the clipped loss against the engine's recorded logprobs; adapter_stepNNN saved and POSTed to /admin/adapter; a guard at a median of 150 sampled tokens per call that keeps the adapter from before the last update | as planned, and the project's task order exactly (all 14 recorded steps). Added: the engine's answer to every switch is checked, and when the guard stops a run the engine is put back on the kept adapter (the project's code only keeps it in its final file) | `08-grpo.txt`; the tests |
| steps.py is `python -m mhrl.cli steps` | the same columns, with tokens per call both ways (median, which the guard reads, and mean) | `09-steps.txt` |
| replay.py prints both real runs from their bundled metrics and rollout logs | as planned: the metrics unchanged, the rollouts cut to numbers; `--records` gives the same output from the project's own files | `01-`, `02-replay-*.txt`; the tests |
| `uv run grpo_harness.py --steps 2` (about 5 minutes) prints step lines like the real run's first: '[grpo] step 1/6 · reward 0.41 · correct 38% · calls 6.5 · no-signal groups 1/2 · ratio 1.000 · peak 3.1 GB' | `step 1/2 · reward 0.54 · correct 50% · calls 6.4 · no-signal groups 1/2 · ratio 1.000 · peak 3.1 GB` and `step 2/2 · reward 0.14 · correct 12% · calls 5.6 · no-signal groups 1/2 · ratio 1.002 · peak 5.0 GB`; 5 min 22 s by the script's own clock (3,024 s by the wall: the lid was closed for 47 minutes). The project's line also prints the time since training started (`00:02:03`), which the plan's quote leaves out | `08-grpo.txt`, `08-grpo-sleep.txt`; the project's `take4_grpo.txt` |
| ... and row lengths by harness (Claude Code about 3,100 tokens vs about 1,300 for mini-swe-agent, so step counts hide how much each harness actually trained the model) | the project's attempt 2, rows it trained: 3,061 against 1,275. Our run, every row: 3,217 against 1,726; but all 8 Claude Code rollouts were wrong, so both updates trained on mini-swe-agent's rows alone (9 rows, 1,890 tokens) | `02-replay-attempt2.txt`, `13-row-lengths.txt`, `08-grpo.txt` |
| (finding 1) `replay.py attempt1`: tokens per call 46, 38, 40, 57, 127, 105, 365, 404 | exactly what `mhrl.cli steps` printed, and reproduced from the records: the MEAN over a step's 8 rollouts, counting only each rollout's first captured row, because these records predate `n_sampled`. Every row counted (equal to `n_sampled` on all 48 rollouts of attempt 2, which has both), the mean is 49, 38, 41, 57, 134, 120, 465, 654 and the median, which the guard reads, 46, 38, 40, 45, 130, 111, 492, 562: 365 and 404 understate steps 7 and 8 | `01-replay-attempt1.txt`; the project's `take3_steps.txt` |
| (finding 2) `replay.py attempt2`: 51, 43, 43, 40, 50, 45 tokens per call over 6 steps (48 rollouts) | the mean (what the project's `steps` printed); the guard read the median, 46, 40, 43, 37, 45, 43 (`metrics.jsonl`'s `tokens_per_call`). 6 steps, 48 rollouts | `02-replay-attempt2.txt`; the project's `take4_steps.txt` |
| (finding 3) ratio 0.999-1.001, peak 6.9 GB; per-step accuracy 38, 25, 62, 75, 50, 25% | as recorded: ratio 0.999 to 1.001, trainer peak 6.88 GB at step 4; accuracy 37.5, 25, 62.5, 75, 50, 25% (62.5 prints as 62). At 8 rollouts a step, one rollout is 12.5 points and the standard error at 50% is 17.7 points | `02-replay-attempt2.txt` |
| (finding 4) calls fell from 5.9 to 3.0 | 3.0 is the lowest point, at step 5; step 8 was 3.8 | `01-replay-attempt1.txt` |
| ... and correct rollouts from 38% to 12% | as recorded: 37.5% at step 1, 12.5% at steps 7 and 8 | `01-replay-attempt1.txt` |
| the policy said more per call until replies hit the 1,024-token cap before their tool call | as recorded: 22 replies reached the cap at steps 7 and 8, 21 of them with no tool call | `01-replay-attempt1.txt` |
| (finding 5) the step-4 adapter, kept by its training metrics, scored 0% in Claude Code | 0 of 24 in Claude Code (17 tasks lost against SFT, p < 0.001; Claude Code only); mini-swe-agent 20 of 24; Pi measured only before chapter 19's sandbox fix. No record says why step 4 was kept. `adapter_step004` is the adapter that sampled step 5, the first ballooned step (median 130, 25% correct, Claude Code 0 of 4); step 4's own line describes `adapter_step003` | `01-replay-attempt1.txt`; the project's `summary_rl_attempt1.json`, `eval_rl_attempt1.jsonl` |
| (finding 6) `--guard-demo` replays attempt 1 through the guard: where it would have stopped and which adapter it would have kept | at 150 it stops at step 7 and keeps `adapter_step005`, letting step 5's median of 130 through (2.8 times step 1's). At 100 it would stop at step 5 and keep `adapter_step003` | `03-guard-demo.txt` |
| (finding 7) the lesson: reply length moves before accuracy does | not what the records show. In attempt 1 the median length and accuracy moved at the same step (5); only the mean rose a step earlier. What they show instead: 8-rollout accuracy swings about ±17.7 points by luck, so length is the signal a guard can read; and the harnesses trained the model unequally. Attempt 1 trained 93% on mini-swe-agent's tokens, and Claude Code had contrast in only 2 of 8 groups; the collapse showed in Claude Code. This harness imbalance is the chapter's real finding | `01-`, `02-replay-*.txt` |
| ... RL runs can rise then fall (the article's earlier Qwen runs went from 37% at step 500 to 26% at step 1,000) | (finding 10) checked against the article: "Multi-harness more than doubled, from 14.6% to 37.0% at step 500." and "Then both fell back to about 26% by step 1,000." (its Qwen3.5-2B runs, correctness as the only reward). Quoted in its own box above | `12-article.txt` |
| (finding 9) `--attempt1` restores the collapsed configuration (call bonus, lr 4e-5, no bias correction, no guard) | as planned. The call bonus is confirmed by the records: every one of attempt 1's 64 rewards comes back from it. The learning rate and the missing bias correction rest on a code comment (`mhrl/cli.py`: "4e-5 (no bias correction) ballooned replies by step 5") and the project's README only; no record holds them. Not restorable from the trainer: the engine's bfloat16 log-probabilities (chapter 15) | the tests; the project's `mhrl/cli.py`, `README.md` |
| (finding 8, the project's README) shared prefixes trained once came after attempt 1 | attempt 1's own `train_tokens` equal the deduplicated count at all 8 steps, and the plain count differs at steps 3, 5, 6 and 7: attempt 1 already deduplicated, so `--attempt1` keeps it | the tests; the project's `runs/main/grpo_attempt1/metrics.jsonl` |
| (review) define each number once, including how it is averaged, and use only that definition | every number in "Words first", and again under every table the scripts print | |
| (review) label any figure 'Claude Code only' | done wherever a figure counts Claude Code alone | |
| (review) show both statistics and say which one the guard reads | every table shows the median and the mean; the guard reads the median | |
| the project's ports: engine 8100, proxy 8200 | the defaults are the project's; on this Mac 8100 belongs to another service and another chapter's film used 8119 and 8200, so this run used 8129 and 8209 | `commands.txt` |

- **The project's numbers** in this chapter come from its recorded files:
  its two GRPO runs (`runs/main/grpo_attempt1` and `runs/main/grpo`,
  bundled slim in `data/project_grpo/`), what its commands printed for them
  and its two evaluation summaries (`data/project_logs/`), and its
  evaluation records (`data/project_records/`).

## The files

Line counts at commit `0e04328`: all lines, and the lines of code
(no blank lines, comments or docstrings), counted with Python's own
tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `grpo_harness.py` | 1,086 | 632 | the project's `train_grpo`, simplified: the plan, the rollouts, the guard, the update, the adapter sent to the engine; `--attempt1`, `--guard-demo` |
| `rollouts.py` | 479 | 241 | training rollouts through the real harnesses, 4 at a time, kept in job order; the project's task order |
| `grpo.py` | 247 | 114 | a step's rows (no-contrast groups skipped, each shared prefix once, long rows skipped) and chapter 15's clipped update |
| `rewards.py` | 106 | 29 | the token bonus, the call bonus, group advantages |
| `grpo_records.py` | 665 | 301 | every run's records read one way; tokens per call defined once, both statistics; the length guard |
| `steps.py` | 133 | 65 | the project's `mhrl.cli steps`, with both statistics |
| `replay.py` | 536 | 332 | the project's two recorded runs, from the bundle or its own files |
| `test_26_grpo_in_harnesses.py` | 687 | 498 | the records against the project's, the pinned outputs, the rollouts (no MLX) |
| `test_26_mlx.py` | 970 | 669 | the loop on chapter 15's tiny model, and one real step through the proxy and mini-swe-agent |
| `data/project_grpo/`, `data/project_logs/` | | | the project's two GRPO runs, slimmed; what its commands printed for them; its two evaluation summaries |
| the rest | | | chapter 25's, unchanged but for this chapter's folder name (`start.sh`, `engine.py`, the harnesses' wiring and sandbox, `evaluate.py`, `noise.py` and their helpers) |

## What this chapter leaves out

- **A long run.** Two steps show the loop working, not learning. The
  project's 6-step run took 12 min 33 s on this Mac (its `take4_grpo.txt`),
  and chapter 28 runs the whole pipeline.
- **Attempt 1 again.** `--attempt1` trains with its settings, but every run
  samples different replies, so it would not reproduce the recorded
  collapse; `replay.py attempt1` shows the one that happened.
- **A better guard.** At 150 the guard catches a blow-up, not a drift. A
  lower limit (100 or 60, `--guard-demo`) would have stopped attempt 1 at
  step 5; choosing one for a run that has not gone wrong yet is a change to
  the project's guard, not made here.

## Mirrors in the project

`mhrl/training/grpo.py` (`train_grpo`, `group_advantages`, the length
guard), `mhrl/rollout.py` (`run_rollout`, `reward_for`, `tool_bonus`) and
`mhrl/cli.py` (`steps`).
