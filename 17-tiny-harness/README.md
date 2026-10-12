# Chapter 17. Tool calls at the token level: a tiny harness

Chapter 16 watched Claude Code from the outside: the requests it sends, the
tools it offers, the loop it runs. This chapter builds that loop ourselves,
in one small program, with the course's model running in the same process.
Nothing is hidden behind an API: you see the exact text the model reads,
the exact text it writes (special tokens included), how a harness finds a
tool call in that text, what it runs, and what it sends back.

The puzzle to keep in mind: the model never runs a command. It only writes
text. So what turns a line of text into a query on a database, and what
happens when the model's text and the harness's expectations do not match,
by one capital letter or by one pair of backticks?

<!-- PHASE 2: the results below are written from the run log only. -->

## Words first

- The **policy** is the model: text in, text out, no memory between calls
  (chapter 16). The **harness** is the program around it; the **agent** is
  the two together.
- A **tool** is something the harness lets the model use. The harness
  describes it with a **JSON schema**: a name, a description, and the
  arguments it takes. This chapter's one tool is `bash`, with one text
  argument, `command`.
- The **chat template** (chapter 9) turns a conversation into the one string
  the model reads. LFM2.5's template prints the tools into the system prompt
  as `List of tools: [...]`, and ends the prompt with `<think>`, so every
  reply starts inside the model's reasoning.
- **Special tokens** are markers the tokenizer keeps as single tokens:
  `<|im_start|>`, `<|im_end|>` (the end of a turn), `<think>`, `</think>`,
  and the two that wrap a tool call, `<|tool_call_start|>` and
  `<|tool_call_end|>`. LFM2.5 writes a call between them as a Python-style
  list: `[bash(command='sqlite3 data.db .tables')]`.
- A reply has three parts: the **reasoning** (before `</think>`), the
  **visible text**, and the **tool calls**.
- **State, action, observation**: each reply is an action, each tool output
  an observation, and the database plus the conversation so far is the
  state.
- A **stop condition** ends the loop: here, a reply with no tool call, 8
  calls, or 3 unreadable replies in a row.
- A **white-box** harness sees the model's tokens (this one); a
  **black-box** harness sees only what an API hands it (Claude Code,
  mini-swe-agent).
- A **text protocol** asks for actions in plain text instead of tool-call
  tokens: mini-swe-agent's old protocol wants exactly one fenced bash block
  per reply. A reply that breaks it is a **format error**.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/17-tiny-harness
uv run tiny_harness.py
uv run tiny_harness.py --held-out 5
uv run experiments.py
uv run experiments.py --protocol text
uv run experiments.py --recorded
```

You need [uv](https://docs.astral.sh/uv/) and an Apple-silicon Mac. The
model is chapter 11's 4-bit build in `COURSE_HOME/models` (default
`~/.cache/mhrl-course`), or Liquid's export from chapter 10 with `--fast`.
`experiments.py --recorded` needs no model. Each attempt works in its own
folder under `COURSE_HOME/17-tiny-harness/work/`, holding a fresh copy of
the question's database as `data.db`; every call is saved under
`COURSE_HOME/17-tiny-harness/rollouts/`. The held-out questions and the two
databases are bundled in `data/` (chapter 1's files, byte for byte), with
the project's recorded Claude Code run of the same model.

| command | what it shows | seconds in the run log |
| --- | --- | ---: |
| `tiny_harness.py` | one question, call by call: the prompt, each raw reply, the parsed call, the observation, the grade | (PHASE 2) |
| `tiny_harness.py --held-out 5` | the first five held-out questions, one line per call, and a summary | (PHASE 2) |
| `experiments.py` | the tool offered as `Bash`: what a call to `bash` costs, beside the recorded Claude Code run | (PHASE 2) |
| `experiments.py --protocol text` | mini-swe-agent's text protocol: one fenced bash block per reply, or a format error | (PHASE 2) |
| `experiments.py --recorded` | the recorded Claude Code run alone | (PHASE 2) |

## The files

- `tiny_harness.py`: the loop. Render, reply, split, run, append, stop,
  grade.
- `toolcalls.py`: the project's tool-call parser (`mhrl/engine/toolcalls.py`),
  `split_reply` and `parse_calls`, which read a call with Python's `ast`
  module without running it.
- `bash_tool.py`: the one tool, a stand-in for chapter 19's sandbox. It is
  not a shell: it runs `sqlite3` (through chapter 11's read-only, query-only,
  no-`ATTACH` guard in `database.py`), `ls`, and `echo ... > answer.txt`,
  inside the task folder, one command per call.
- `policy.py`: the seam between harness and model (chat in, reply text out);
  `lfm.py`: the real model behind it, with mlx-lm.
- `experiments.py`: the `Bash` name and the text protocol.
- `grader.py`, `tasks.py`, `lfm_files.py`, `report.py`: chapter 1's grader,
  the questions, where things live, and the printing.
