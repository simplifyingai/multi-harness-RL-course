# Chapter 18. An engine that keeps the receipts

In chapter 17 the harness and the model shared one process, and the harness
read every token the model wrote. Real harnesses are separate programs.
mini-swe-agent, Pi and Claude Code reach a model over HTTP, with nothing but
an address and a key, and they get back only what the API hands them. This
chapter puts our model behind such an API: a small server, the **engine**,
that answers in OpenAI's chat format (mini-swe-agent and Pi speak it as it
is; Claude Code reaches it through chapter 20's translation), and that also
returns what hosted APIs never give back, the exact tokens.

The puzzle to keep in mind: on every call a harness sends the whole
conversation again, the model's earlier replies included. But it sends its
own copy of each reply, not the model's words. Does that matter, if the
copy says the same thing? For training it does, and this chapter measures
exactly where the copy stops being the model's tokens, and why.

<!-- PHASE 2: the results below are written from the run log only. -->

## Words first

- An **inference engine** (a model server) is a long-running program that
  keeps the model in memory and answers model calls over HTTP, so many
  agents can share one copy of it.
- A **model API** is "messages and tools in, a reply and the usage out"
  (chapter 16). A harness needs two settings to find one: a **base URL** (the
  address, here `http://127.0.0.1:8100/v1`) and an **API key** (a password;
  ours listens on this machine only, so any key works).
- **OpenAI's chat format** is the API most programs speak: `POST
  /v1/chat/completions` with `messages`, `tools`, `temperature` and
  `max_tokens`, answered with a `message`, a `finish_reason` and the
  `usage`. mini-swe-agent and Pi speak OpenAI chat; Claude Code's own
  dialect is translated in chapter 20.
- In that format a **tool call** is JSON: an `id`, the tool's `name`, and
  its `arguments` as a JSON string. A reply that calls tools ends with
  `finish_reason: "tool_calls"`; a plain answer ends with `"stop"`, and a
  reply cut off by the length cap with `"length"`.
- The **usage** counts tokens three ways: **prompt tokens** (read),
  **completion tokens** (written) and **cached tokens** (read before, so not
  worked out again). A harness re-sends the whole history on every call, so
  the prompt grows call after call.
- The **receipts** are the three things training needs and hosted APIs never
  return: `prompt_token_ids` (the exact tokens the model read), `token_ids`
  (the exact tokens it sampled) and each sampled token's **log-probability**
  (chapter 3), written `token_id:<n>` so no text has to be turned back into
  tokens. `top_logprobs` asks for the likeliest alternatives at each
  position too, up to 20.
- **vLLM** is the engine the article used, on NVIDIA GPUs; the receipts are
  its field names. vLLM needs CUDA, which a Mac does not have, so the project
  wrote this small MLX engine instead, returning the same fields.
- **Tool-call translation**: LFM2.5 writes a call as text between two special
  tokens (`[bash(command='ls')]`, chapter 17). The engine turns it into
  OpenAI's JSON on the way out, and on the way back in it swaps the harness's
  JSON copy for the model's own text, found by the call's `id`.
- **Template round-trip drift** is what happens without that swap: the chat
  template (chapter 9) rewrites the copy, and the tokens it produces are not
  the ones the model sampled. It has two causes, both named by `drift.py`:
  the **dropped `<think>`** (the copy carries no reasoning) and the
  **re-written tool call** (the template spells the call its own way, which
  is not always the model's).
- **Token-prefix continuity**: in a clean conversation, each call's prompt
  begins with exactly the previous call's prompt and reply. Training stitches
  a whole rollout into one row only when that holds (chapter 21).
- A **prompt cache** keeps the model's work on tokens it has read (chapter
  12's KV cache), so call k+1 only reads what was added since call k. The
  engine keeps 8, and picks the one holding the **longest exact prefix** of
  the new prompt; one token of difference and the work cannot be reused.
- **One worker thread**: MLX keeps its work on the thread that started it, so
  every model step runs on one thread and requests **queue** for it. Four
  rollouts at once are answered one after another, which is why the
  project's harnesses wait up to 600 seconds for a call.
- The **context window** is the most tokens one call may read. The model
  reads up to 131,072 positions; the engine refuses prompts over 16,384 with
  OpenAI's error code `context_length_exceeded`.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/18-engine
uv run drift.py --recorded
uv run drift.py
uv run engine.py --port 8119
# in a second terminal, while the engine runs:
uv run client.py turns --base-url http://127.0.0.1:8119/v1
uv run client.py queue --base-url http://127.0.0.1:8119/v1
uv run client.py too-long --base-url http://127.0.0.1:8119/v1
uv run client.py logprobs --base-url http://127.0.0.1:8119/v1
uv run tiny_harness_http.py --base-url http://127.0.0.1:8119/v1
uv run tiny_harness_http.py --base-url http://127.0.0.1:8119/v1 --held-out 5
```

The engine's default port is 8100, the project's. On the Mac that made the
run log another program held 8100, so every command there uses `--port
8119`; when the port is taken, `engine.py` stops at once and says so, naming
`--port`.

You need [uv](https://docs.astral.sh/uv/) and an Apple-silicon Mac. The
model is chapter 11's 4-bit build in `COURSE_HOME/models` (default
`~/.cache/mhrl-course`), or Liquid's export from chapter 10 with `--fast`.
`drift.py --recorded` loads only the tokenizer. `client.py` and
`tiny_harness_http.py` need only Python's standard library: they hold no
model and no tokenizer, just the base URL and the key. Work folders and
every saved rollout go under `COURSE_HOME/18-engine/`.

| command | what it shows | seconds in the run log | peak memory |
| --- | --- | ---: | ---: |
| `drift.py --recorded` | two replies the project's engine sampled: a re-written tool call, and a text that cuts into other tokens | (PHASE 2) | (PHASE 2) |
| `drift.py` | a fresh reply: where the harness's copy leaves the sampled tokens, for each cause, and the engine's fix | (PHASE 2) | (PHASE 2) |
| `engine.py --port 8119` | `engine ready`, then one line per request | (PHASE 2) | (PHASE 2) |
| `client.py turns` | one reply's JSON with its receipts; call 2's cached tokens and first-token time | (PHASE 2) | n/a |
| `client.py queue` | four requests at once, answered one after another | (PHASE 2) | n/a |
| `client.py too-long` | a 20,000-word prompt refused with `context_length_exceeded` | (PHASE 2) | n/a |
| `client.py logprobs` | log-probabilities with 5 alternatives; chapter 20's probe | (PHASE 2) | n/a |
| `tiny_harness_http.py` | chapter 17's harness solving a question through the engine | (PHASE 2) | n/a |

## The files

Line counts are honest ones: all lines, and in brackets the lines of code
(no blank lines, comments or docstrings).

- `engine.py`, 794 lines (375 of code): the server. It checks each request,
  puts the model's own text back in place of a harness's copy, renders the
  conversation into tokens, refuses a prompt over 16,384 tokens, takes the
  best prompt cache, samples on the one worker thread, and answers in
  OpenAI's shape with vLLM's token fields. It follows the project's
  `mhrl/engine/policy.py` and `server.py` (869 lines, 500 of code, with
  `toolcalls.py`), without the adapter route and with more: stricter request
  checks, the port check, timings, and a count of replies whose text cuts
  into other tokens. The engine's own two files, `engine.py` and
  `toolcalls.py`, are 1,016 lines (471 of code); with the model half
  (`policy.py`, `lfm.py`, `lfm_text.py`), 1,513 lines (634 of code).
- `toolcalls.py`, 222 lines (96 of code): chapter 17's parser (the project's
  `split_reply` and `parse_calls`), plus `openai_tool_calls`, which gives each
  call its id.
- `policy.py`: the seam between the engine and the model, cut at the tokens:
  render, encode, decode, make a cache, sample. `lfm.py` is the model behind
  it on MLX (chapter 12's sampling loop with a cache, float32 scores and the
  temperature applied to the reported log-probabilities); `lfm_text.py` is
  its tokenizer and chat template alone. The tests put a scripted model
  behind the same seam.
- `drift.py`: both causes of drift, measured on a fresh reply and on two
  recorded ones (`data/recorded_replies.json`: their exact token ids and
  log-probabilities, from the project's GRPO run).
- `client.py`: the four demonstrations, as plain HTTP requests.
- `tiny_harness_http.py`: chapter 17's harness speaking HTTP. It keeps each
  reply as the JSON it came in, answers each call with a `tool` message
  carrying that call's `id`, and at the end checks the receipts: did every
  call's prompt begin with the previous call's prompt and reply?
- `bash_tool.py`, `database.py`, `grader.py`, `tasks.py`, `lfm_files.py`:
  chapter 17's tool, its database guard, chapter 1's grader, the questions,
  and where things live.

## What the engine leaves out

- **Swapping adapters while it runs** (the project's `POST /admin/adapter`)
  is shown in chapter 26, where training uses it.
- **Streaming**: the engine answers with one JSON body and refuses `stream`.
  Chapter 20's capture proxy replays a reply as a stream for the harnesses
  that need one.
- It samples from the whole distribution at the request's temperature, as the
  project's engine does: `top_p`, `top_k`, penalties, `seed` and
  `tool_choice` are ignored, and `max_tokens` is lowered to 1,024. It refuses
  `n` above 1.

## Mirrors in the project

`mhrl/engine/policy.py` (`PolicyEngine`: `render`, `generate`,
`_remember` and `_restore`, the prompt caches) and `mhrl/engine/server.py`
(`to_openai`, `/v1/chat/completions`, `MLX_THREADS = 1`).
