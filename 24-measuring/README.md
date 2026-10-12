# Chapter 24. Measuring fairly: baseline, pass@1 and noise

Chapter 23 ran one task per harness, which is a demonstration and not yet a
measurement. Before any training, this chapter fixes the test: 24 held-out
tasks per harness, one greedy attempt each, the same budget everywhere, and
Pi never trained in. Then it asks what 24 tasks can and cannot show.

The run log, its outputs and the plan-versus-measured table come with the
chapter's run. Every output quoted here will be copied from that log.

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
- **Sampling noise** is how far a score moves by luck alone. With 24 tasks one
  task is worth 4.2 points.
- A **held-out harness** is one used only for testing. Here that is Pi.

## Run it

```bash
cd multi-harness-RL-course/24-measuring
uv run noise.py                          # no model, no stack: the project's records
uv run baseline_eval.py --plan           # the protocol and the 24 tasks
ENGINE_PORT=8119 ./start.sh scripted     # no model: the scripted engine, the proxy on 8200
uv run baseline_eval.py --limit 2        # every harness on 2 tasks, scripted
./start.sh stop
ENGINE_PORT=8119 ./start.sh              # the model: the 4-bit LFM2.5
uv run baseline_eval.py --fast           # 4 tasks per harness (12 cells)
uv run baseline_eval.py                  # 24 tasks per harness (72 cells)
./start.sh stop
uv run noise.py                          # now with your baseline beside the project's
```

`start.sh` is chapter 23's, with one step added. Before an engine starts,
`fetch_model.py` checks for what that engine reads, and fetches it when it
is missing: the 4-bit model (Liquid's export, 1.6 GB) or, for the scripted
engine, only LFM2.5's tokenizer files (17.9 MB). `baseline_eval.py` installs
any harness that is missing (`install.py`), so the chapter rebuilds its own
inputs instead of sending you back to chapter 23.

The course's ports are the project's: the engine on 8100 and the proxy on
8200. If another program holds one, set `ENGINE_PORT` and `PROXY_PORT`, and
give `baseline_eval.py --proxy-url` to match.

## The files

| file | what it is |
| --- | --- |
| `baseline_eval.py` | the protocol, the task picks, the harness checks, the run, the table |
| `evaluate.py` | the project's `balanced`, `summarize`, `format_table` and `evaluate`; one cell built from chapter 23's `run_one.py` |
| `noise.py` | one task's weight, standard errors and Wilson intervals, two identical models, the ceiling, the macro average, every pair of checkpoints, the broken evaluation, the article's numbers in their own box |
| `stats.py` | chapter 14's `stats.py`, plus the Wilson interval and the identical-models arithmetic |
| `fetch_model.py` | fetches the model or its tokenizer when missing, checked by SHA-256 |
| `data/project_records/` | the project's evaluation records from its run of 2026-10-06, cut to the fields `noise.py` reads |
| `run_one.py`, `install.py`, `wire.py`, `loopback.py`, `start.sh`, `scripted_engine.py`, `scripted_replies.py`, `engine.py` and the rest | chapter 23's, unchanged but for this chapter's folder name (`start.sh` adds the fetch step) |

## Mirrors in the project

`mhrl/evaluate.py` (`evaluate`, `summarize`) and `mhrl/cli.py` (`balanced`,
`eval`).
