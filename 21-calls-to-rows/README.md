# Chapter 21. From calls to training rows

A harness makes model calls; a trainer needs rows of tokens. Between them,
chapter 20's capture proxy links the calls into a graph (each call's parent
is the earlier call its prompt continues) and flattens every path from a
root to a leaf into one row: `input_ids`, a `loss_mask` that is 1 only on
the model's own tokens, and their `logprobs`. This chapter drives that
graph on purpose, seven ways, and then with Claude Code.

## Run it

Every script here talks to the real proxy. `graph_lab.py` needs the
scripted engine (no model, no MLX); `reminder.py` and `crosscheck.py` run
with either engine.

```bash
ENGINE_PORT=8119 ./start.sh scripted   # the scripted engine on 8119, the proxy on 8200
uv run graph_lab.py                    # the seven cases (or one: clean, retry, ...)
uv run reminder.py                     # Claude Code 2.1.290: its total-tokens note unset, fixed, off
uv run crosscheck.py                   # the proxy's record vs Claude Code's own log
./start.sh stop
```

`ENGINE_PORT=8119 ./start.sh` starts LFM2.5 (chapter 18's engine) instead
of the scripted one, for `reminder.py` and `crosscheck.py` with the model.
Use 8119 when another program holds the project's port, 8100.

## The files

- `graph_lab.py`: the seven cases (a clean 4-call chain, a retried call, a
  system prompt with a line that changes on every call, a summarised
  history, a turn without log-probabilities, a side call and a broken
  engine), each printed as graph, rows, findings and stats, then
  `count_calls`, `n_sampled` and `dedupe_shared_prefixes`.
- `reminder.py`: Claude Code through the proxy with
  `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` unset, `fixed` and `off`; what Claude
  Code sent (through a recording relay), what the engine received, and
  what the proxy built.
- `crosscheck.py`: the proxy's turns against Claude Code's stream-json log,
  turn by turn: replies, completion tokens and the trained reply lengths.
- `training_rows.py`: the project's `count_calls`, `n_sampled`,
  `dedupe_shared_prefixes` and `item_from_sequence`, copied.
- `rollout_view.py`: prints a rollout document as data.
- `claude_code.py`: the pinned Claude Code, its private config and the
  project's flags, run inside the sandbox, plus the recording relay.
- `sandbox.py`, `sandbox.sb`: chapter 19's sandbox (the project's Seatbelt
  profile): Claude Code's Bash runs the model's commands without asking, so
  it may write only in its run's folders and reach only this machine.
- `scripted_engine.py`, `scripted_replies.py`: chapter 20's engine with a
  scripted model and a remote control (`/script`), and a broken copy under
  `/broken`.
- `start.sh`, `proxy-requirements.txt`: chapter 20's stack, with a
  `scripted` command.
- Copied from chapter 20 unchanged: `engine.py`, `lfm.py`, `lfm_text.py`,
  `policy.py`, `toolcalls.py`, `tiny_harness_http.py`, `bash_tool.py`,
  `database.py`, `grader.py`, `tasks.py`, `data/`; `lfm_files.py` with
  this chapter's name; `capture_client.py` with the rollout route's two
  options.

## Mirrors in the project

OpenEnv's capture graph (`openenv/core/harness/capture/graph.py`,
`export.py` and `validate.py` at commit 3f14355), `mhrl/rollout.py`
(`count_calls`, `run_rollout`) and `mhrl/training/common.py`
(`dedupe_shared_prefixes`, `item_from_sequence`).
