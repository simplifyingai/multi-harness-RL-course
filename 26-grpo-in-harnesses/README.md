# Chapter 26. GRPO through the harnesses, and the run that went wrong

**Status: the code is built and tested without the model's weights. The
measured runs, and every number this page will quote, come with a later
commit, written from the run log.** Until then this page lists only what
each script does and how to run it.

Chapter 15 ran GRPO on the real model in a loop we owned: one question, one
reply. Chapter 25 taught the model inside the real harnesses with SFT. This
chapter does what the project does next: GRPO through mini-swe-agent and
Claude Code, every call sampled at temperature 1 and recorded by the capture
proxy, and one clipped update per step on the engine's own
log-probabilities. Then it replays the project's two recorded runs: the
first, whose replies grew longer every step until they stopped calling tools,
and the second, with the length guard that would have stopped the first.

## Run it

```bash
cd multi-harness-RL-course/26-grpo-in-harnesses
uv run replay.py attempt1          # the project's first run, step by step, from its records
uv run replay.py attempt2          # its second run, the one the guard watched
uv run grpo_harness.py --guard-demo   # attempt 1's rollouts through the length guard
./start.sh                         # the model's engine and the capture proxy
uv run grpo_harness.py --steps 2   # two GRPO steps from chapter 25's SFT adapter
uv run steps.py                    # what each step's rollouts did
./start.sh stop
```

`grpo_harness.py --attempt1` trains with the first run's settings (the
article's call bonus, learning rate 4e-5, no bias correction, no guard) into
its own folder. `replay.py --records ~/multi-harness-rl/runs/main` reads the
project's own files instead of the bundled copies.

## The files

| file | what it is |
| --- | --- |
| `grpo_harness.py` | the project's `train_grpo`, simplified: 2 tasks per step, harnesses alternating, 4 rollouts each through the proxy, the length guard, one clipped update, the adapter saved and sent to the engine; `--attempt1`, `--guard-demo` |
| `rollouts.py` | training rollouts through the real harnesses (chapter 25's collection with the model as the engine), 4 at a time, in job order; the project's task order |
| `grpo.py` | a step's rows (no-contrast groups skipped, each shared prefix trained once, long rows skipped) and the update: chapter 15's clipped loss and AdamW step |
| `rewards.py` | the token bonus (the project's) and the call bonus (attempt 1's), and group advantages |
| `grpo_records.py` | reads every run's records, defines tokens per call once (the median the guard reads, and the mean), and the length guard |
| `steps.py` | the project's `python -m mhrl.cli steps`, with both statistics |
| `replay.py` | the project's two recorded runs, from the bundled copies or its own files |
| `data/project_grpo/` | the project's two GRPO runs: their metrics unchanged, their rollouts cut to numbers (no text, no paths) |
| `data/project_logs/` | what the project's commands printed for those runs, and its evaluation summaries |
| the rest | chapter 25's, unchanged but for this chapter's folder name |

## Mirrors in the project

`mhrl/training/grpo.py` (`train_grpo`, `group_advantages`, the length
guard), `mhrl/rollout.py` (`run_rollout`, `reward_for`, `tool_bonus`) and
`mhrl/cli.py` (`steps`).
