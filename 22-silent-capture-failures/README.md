# Chapter 22. Capture that fails without an error

Chapter 21 cut a rollout into training rows and showed, with scripted
conversations, what a changing line, a summary or a missing log-probability
does to them. This chapter takes **one real Claude Code run**, solved and
captured as one row, and replays it with the model's replies fixed token
for token while Claude Code, the proxy and the grader run again. Then it
breaks one thing at a time: Claude Code's total-tokens note left on, the
history summarised halfway, one reply served without log-probabilities.
Every replay still solves the task, and the proxy's one verdict still says
`trainable: true`. `crosscheck.py` compares each replay with Claude Code's
own log and catches all three, each with a different check.

The full chapter, with every output copied from its run log, comes with
the run log.

## Run it

```bash
ENGINE_PORT=8119 ./start.sh scripted   # the scripted engine on 8119, the proxy on 8200
uv run capture.py                      # the run to break: Claude Code, solved, one row
uv run breakages.py                    # replayed four ways: replay, note, summary, no-logprobs
uv run crosscheck.py all               # each run against Claude Code's own log
./start.sh stop
```

With the model, the capture is LFM2.5's and its replay needs LFM2.5's
tokenizer (no weights):

```bash
ENGINE_PORT=8119 ./start.sh            # chapter 18's engine with the 4-bit LFM2.5
uv run capture.py                      # up to 3 tries for a solved one-row run
./start.sh stop
ENGINE_PORT=8119 ./start.sh replay     # the scripted engine, cutting LFM2.5's tokens
uv run breakages.py
uv run crosscheck.py all
./start.sh stop
```

Use 8119 when another program holds the project's port, 8100.

## The files

- `capture.py`: records the run every case replays, as the project runs
  Claude Code (`CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off`, the session id as its
  key, chapter 19's sandbox); keeps the first attempt that is solved and
  already one trainable row as `runs/capture`, and every attempt as
  `runs/attempt-N`.
- `breakages.py`: replays the capture with its replies fixed, four ways,
  and prints for each the grade, the rows and the calls each holds, the
  proxy's findings and its verdict, then a table.
- `crosscheck.py`: chapter 21's check of the proxy's record against Claude
  Code's stream-json log (turns, replies, completion tokens, trained tokens
  per call), plus calls per row: Claude Code's one conversation should be
  one row, and a row that starts anywhere else is named with its cause.
- `compaction.py`: asks Claude Code to compact its history early through the
  proxy and prints what it did; `breakages.py` summarises with a relay.
- `captures.py`: the run folders: what each run kept, and each call's exact
  reply read back out of its row.
- `scripted_engine.py`, `scripted_replies.py`: chapter 21's scripted engine,
  which now also serves exact token ids with exact log-probabilities, with
  one token per character or LFM2.5's own tokenizer (`--tokenizer lfm2.5`).
- `claude_code.py`: chapter 21's pinned Claude Code and recording relay; the
  relay can now rewrite each request on its way (the summary case).
- `start.sh`, `proxy-requirements.txt`: chapter 21's stack, with `replay`.
- `lfm_files.py`: chapter 21's, with this chapter's name and LFM2.5's
  `tokenizer.json` found by its fingerprint.
- Copied from chapter 21 unchanged: `engine.py`, `lfm.py`, `lfm_text.py`,
  `policy.py`, `toolcalls.py`, `grader.py`, `tasks.py`, `data/`,
  `capture_client.py`, `rollout_view.py`, `training_rows.py`, `sandbox.py`,
  `sandbox.sb` (with this chapter's folder).

## Mirrors in the project

OpenEnv's capture graph (`openenv/core/harness/capture/graph.py`,
`export.py` and `validate.py` at commit 3f14355), `mhrl/rollout.py`
(`count_calls`, `run_rollout`) and `mhrl/training/common.py`
(`dedupe_shared_prefixes`, `item_from_sequence`).
