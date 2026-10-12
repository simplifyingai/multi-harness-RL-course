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

Every output below is copied from the run log, made at one commit from a
clean `COURSE_HOME/18-engine`, each model command started only after a
minute with no other heavy job on the machine. The seconds are the
command's wall time; the memory is MLX's own peak where a script prints it,
else the largest process's resident memory.

| command | what it shows | seconds | peak memory |
| --- | --- | ---: | ---: |
| `drift.py --recorded` | two replies the project's engine sampled: a re-written tool call, and a text that cuts into other tokens | 1.8 | 0.43 GB (resident; tokenizer only) |
| `drift.py` | a fresh reply: where the harness's copy leaves the sampled tokens, for each cause, and the engine's fix | 3.8 | 2.12 GB (MLX) |
| `engine.py --port 8119` | `engine ready`, one line per request, and its peak when stopped | up 104.5 | 2.27 GB (MLX) |
| a second `engine.py --port 8119` | the port is taken: it stops before loading the model | 0.3 | 0.05 GB |
| `client.py turns` | one reply's JSON with its receipts; call 2's cached tokens and first-token time | 2.7 | 0.04 GB (client) |
| `client.py queue` | four requests at once, answered one after another | 6.1 | 0.04 GB (client) |
| `client.py too-long` | a 20,000-word prompt refused with `context_length_exceeded` | 0.2 | 0.03 GB (client) |
| `client.py logprobs` | log-probabilities with 5 alternatives; chapter 20's probe | 0.4 | 0.03 GB (client) |
| `tiny_harness_http.py` | chapter 17's harness solving a question through the engine | 12.3 | 0.04 GB (client) |
| `tiny_harness_http.py --held-out 5` | the first five held-out questions | 78.0 | 0.04 GB (client) |

## What a harness's copy does to the tokens: `drift.py`

`drift.py` loads the model in-process, asks it the first call of
clinic-easy-001 at temperature 0, and gets chapter 17's first reply again,
89 tokens after a 243-token prompt. The engine turns it into the copy every
OpenAI-speaking harness keeps:

```text
2. the copy a harness keeps and sends back (OpenAI JSON)
  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_e148cafd93844869a5c6f80a", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"sqlite3 data.db \\\".tables\\\"\"}"}}]}
  as it is, LFM2.5's template refuses it: Tool call arguments must be a mapping, got a JSON-encoded string: parse arguments with json.loads() before applying the chat template
  so its arguments are read back into an object first
```

The copy has no reasoning at all, and its call is JSON text, which LFM2.5's
chat template will not even render. Read back into an object and rendered,
it gives the next prompt, and the two causes of drift show where it leaves
the sampled tokens:

```text
3. cause 1, the dropped <think>: the copy as it comes
  first difference: token 243 of the previous prompt's 243: the reply is not even reached
    sampled      '<|im_start|>' 'assistant' '\n' | '<think>' 'The' ' user'
    re-rendered  '<|im_start|>' 'assistant' '\n' | '<|tool_call_start|>' '[' 'bash'

4. cause 2, the re-written tool call: the copy, its reasoning given back
  the same tokens: the next prompt continues the previous call

5. the engine's way: the model's own text, found by the call's id
  continues the previous call: yes

peak memory: 2.12 GB, the weights included
```

1. **The dropped `<think>`.** The previous prompt ended with `<think>`, the
   reply's first token followed it, and the copy has no reasoning to put
   there. The next prompt leaves the chain at the previous prompt's very last
   token: not one token of the reply survives.
2. **The re-written tool call.** With the reasoning given back, this reply's
   call comes back token for token: the model wrote the call exactly as the
   template writes calls (single quotes outside, double inside). On a fresh
   greedy reply this cause did not fire. The project's run holds a reply
   where it did (next section).

Putting the model's own text back, found by the call's id, gives a next
prompt that begins with the previous prompt and reply, token for token.

## The project's recorded replies: `drift.py --recorded`

Two replies the project's engine sampled at temperature 1 during its GRPO
run, with their exact token ids and log-probabilities
(`data/recorded_replies.json`). The first shows cause 2:

```text
1. re-spelled: school-easy-018, mini-swe-agent, call 3
  sampled at temperature 1 by the project's engine (GRPO run)
  <think>These SQLite commands don't like quotes unless escaped properly.</think><|tool_call_start|>[bash(command='sqlite3 -header data.db "SELECT name FROM "schema" WHERE typeof(name)\"="\'text\'" LIMIT 1"')]<|tool_call_end|><|im_end|>
  first difference: token 36 of the reply, in the tool-call span
    sampled      ' WHERE' ' typeof' '(name' | ')\\' '"' '="'
    re-rendered  ' WHERE' ' typeof' '(name' | ')"' '="' "\\'"
```

The model escaped a double quote it did not need to (`\"`); the JSON copy
keeps only the command's value, and the template writes it back plainly.

The second reply shows what even the engine's fix leaves: the same text can
be cut into other tokens (chapter 9).

```text
2. re-cut: school-medium-008, mini-swe-agent, call 1
  sampled at temperature 1 by the project's engine (GRPO run)
  <think>I should see the tables and columns before writing SQL.</think><|tool_call_start|>[bash(command='sqlite3 data.db .schema')]<|tool_call_end|><|im_end|>
  29 tokens sampled; its text encodes to 28
  first difference: token 24 of the reply
    the model drew   ' .' (-0.06) 's' (-6.53) 'chema' (-0.21)
    the text cuts to ' .' 'schema'
  putting the text back cannot give these ids back: only the ids could
```

The engine puts back the model's text, as the project's engine does, and the
tokenizer cuts that text its own way. How often that happens was counted
over the recorded data with this chapter's own code (the run log's
`12-drift-counts.txt`):

- in the project's GRPO run, 7 of 298 replies cut into other tokens. 8 of
  its 48 rollouts were split into two training sequences: 2 of them
  (school-medium-008 and shop-easy-006, both mini-swe-agent) at a reply whose
  text came back unchanged in other tokens, the other 6 at a harness's retry
  or note. Only 1 of 279 tool calls was re-spelled;
- in chapter 17's run, 1 of 42 replies encodes to another number of tokens
  than were sampled (100 sampled, 99 re-encoded).

Restoring the exact ids instead of the text would close this gap. The
engine keeps the project's behaviour and counts it instead: `recut` in
`/admin/stats`, and a note on the engine's console for each such reply.

## The engine: `engine.py`

```text
engine ready · lfm2.5-2.6b 4-bit · 127.0.0.1:8119 · 1.6 GB
  ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours (ours, built by chapter 11's convert.py), loaded in 0.8 s
  one worker thread; each request prints a line here
```

Each request then prints one line (`chat · read 243 tokens (0 cached) ·
wrote 89 · tool_calls · waited 0.0 s · 1.6 s`). Stopped with Ctrl-C after
53 requests, it printed `engine stopped · 53 requests · peak memory 2.27 GB,
the weights included`. A second engine on the same port stops at once:

```text
127.0.0.1:8119 is in use: another program listens there (or an engine that just stopped has not let go of it yet). Start the engine on a free port with --port, for example: uv run engine.py --port 8120
```

## One reply's receipts: `client.py turns`

The first call of clinic-easy-001, as the engine answers it (the client cuts
the long lists):

```text
  "prompt_token_ids": [
    124894,
    124899,
    23630,
    207,
    3582,
    437,
    "... 243 in all"
  ],
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_…",
            "type": "function",
            "function": {
              "name": "bash",
              "arguments": "{\"command\": \"sqlite3 data.db \\\".tables\\\"\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls",
      "token_ids": [
        597,
        4695,
        …
        "... 89 in all"
      ],
      "logprobs": {
        "content": [
          {
            "token": "token_id:597",
            "logprob": -3.62396240234375e-05,
            …
  "usage": {
    "prompt_tokens": 243,
    "completion_tokens": 89,
    "total_tokens": 332,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  },
```

Then call 2 of the same conversation, after the harness's copy of call 1 and
its result:

```text
2. call 2 of the same conversation: call 1's reply, then its result
  call 1: read 243 tokens, 0 cached · first token after 0.35 s · whole reply 1.61 s
  call 2: read 348 tokens, 332 cached · first token after 0.06 s · whole reply 0.90 s
  call 2's prompt begins with call 1's prompt and reply: yes
```

332 cached tokens are exactly call 1's 243 read plus 89 written: the cache
held the whole of call 1, so call 2 read only 16 new tokens, and its first
token came after 0.06 s instead of 0.35 s.

## Four requests at once: `client.py queue`

```text
4 requests at once, one worker thread
  question              waited   model  answered
  airline-medium-001     0.00s   1.64s     1.66s
  airline-easy-001       1.65s   1.11s     2.77s
  clinic-easy-001        2.76s   1.56s     4.33s
  clinic-medium-001      4.32s   1.55s     5.88s
  the four model times add up to 5.86 s; the last answer came after 5.88 s: one request at a time
```

Each request waited for the ones ahead of it. With four rollouts of up to
1,024-token replies queued behind each other, one call can wait minutes,
which is why the project's harnesses wait up to 600 seconds for a call.

## A prompt over the limit: `client.py too-long`

```text
a prompt of 20,000 words ('data' again and again)
  HTTP 400: {"error":{"message":"prompt is 20,010 tokens; the limit is 16,384","type":"invalid_request_error","code":"context_length_exceeded"}}
```

## Log-probabilities and chapter 20's probe: `client.py logprobs`

```text
1. 4 tokens drawn at temperature 1.0, 5 alternatives each
  token 1: '\n' -0.127
    likeliest: '\n' -0.127 · 'The' -2.127 · '1' -11.314 · 'User' -12.564 · 'We' -12.689
  ...
2. chapter 20's probe: the gap between the two likeliest first tokens
  temperature 1.0: gap 2.0000
  temperature 2.0: gap 1.0000
  ratio 0.500 -> processed

3. asking for 21 alternatives
  HTTP 400: {"error":{"message":"top_logprobs must be between 0 and 20","type":"invalid_request_error","code":null}}
```

The probe is OpenEnv's own test, which chapter 20's capture proxy runs on
an engine before it trusts it for training: at twice the temperature, the
gap between the two likeliest first tokens halved. The log-probabilities
are the distribution the token was drawn from (scores divided by the
temperature), which is what GRPO's ratio needs (chapter 15), not the raw
scores, whose gap would stay the same.

## Chapter 17's harness, through the engine: `tiny_harness_http.py`

```text
tiny_harness_http.py · chapter 17's harness, through the engine
  base URL http://127.0.0.1:8119/v1 · API key 'any-key'
  the harness holds no model and no tokenizer: only these two

clinic-easy-001 (easy): How many patients live in Murcia?
  call 1 · read 243 tokens (0 cached), wrote 89 · tool_calls · 1.6 s
    tool call    bash {"command": "sqlite3 data.db \".tables\""}
    observation  appointments  doctors       patients
  call 2 · read 348 tokens (332 cached), wrote 59 · tool_calls · 0.9 s
  ...
  call 7 · read 1,045 tokens (1,024 cached), wrote 125 · stop · 1.8 s
    text         30
  stop: no tool call: the reply is the model's final answer
  grade: correct (correct; answer.txt: '30') · 7 calls · reward 1
  token chain: 6 of 6 calls began with the previous call's prompt and reply
```

With `--held-out 5` all five questions were answered correctly, in 8, 6, 7,
8 and 8 calls (37 in all), and every call began with the previous call's
prompt and reply (7, 5, 6, 7 and 7 of the possible ones: all of them).
Five questions are a demonstration, not a measurement of the model.

Chapter 17 ran the same harness on the same five questions in-process and
also solved 5 of 5, in 35 calls. The replies part ways at clinic-easy-001's
call 4, after the same 524-token prompt: 84 tokens here, 81 in chapter 17's
run log.
Chapter 17's loop picked each token from bfloat16 scores and read every
prompt from scratch; the engine picks from float32 scores and reads cached
prompts. This run does not say which of the two moved the choice.

## The engine's counters: `/admin/stats`

```text
{
    "requests": 53,
    "prompt_tokens": 31655,
    "cached_tokens": 27529,
    "completion_tokens": 6467,
    "model_seconds": 98.29379737592535,
    "restored": 142,
    "not_restored": 0,
    "recut": 0
}
```

Over the whole run, 27,529 of the 31,655 prompt tokens (87%) came from a
cache. Assistant turns were put back 142 times and never missed, and no
reply in this run cut into other tokens.

## Plan versus measured

| the plan said | measured |
| --- | --- |
| `engine.py` about 250 lines; `engine.py` + `toolcalls.py` roughly 450 | `engine.py` 802 lines (383 of code); with `toolcalls.py` 1,024 (479 of code): the 450 matches lines of code, not lines |
| `engine ready · lfm2.5-2.6b 4-bit · 127.0.0.1:8100 · 1.6 GB` | `... · 127.0.0.1:8119 · 1.6 GB`: 8100 was taken on this Mac |
| `drift.py` shows the first difference in the tool-call span and the dropped `<think>` | on a fresh greedy reply only the dropped `<think>` fired (at token 243 of the previous prompt's 243); the tool-call span came back identical. The re-written call is shown on the project's recorded reply |
| restoring the raw text keeps every turn on one exact token chain | true for every call of this run (32 of 32 through the harness, `recut` 0), but not always: in the project's GRPO run 7 of 298 replies re-encode to other tokens, and 2 of 48 rollouts split there |
| turn 2: `cached_tokens` covers turn 1 and the first token arrives sooner | 332 of 348 cached (all of turn 1); first token after 0.06 s instead of 0.35 s |
| four requests finish one after another | waits 0.00, 1.65, 2.76 and 4.32 s; model times 5.86 s in all, last answer at 5.88 s |
| a 20,000-token prompt returns `context_length_exceeded` | 20,000 words: 20,010 tokens, refused |
| `top_logprobs` up to 20; chapter 20's probe against it | 5 alternatives returned; the probe's gap 2.0000 at temperature 1, 1.0000 at 2: processed; 21 refused |
| `POST /admin/adapter` changes the reply and drops the caches | moved to chapter 26, where training uses it; not built here |
| the tiny harness solves a task with a base URL and a key | clinic-easy-001 correct in 7 calls; 5 of 5 held-out questions |

## Findings about the project

- **The project's engine answers 500 to a tool-call turn it does not
  remember.** Its `PolicyEngine.render` passes a harness's copy it cannot
  find to the chat template as it is, and LFM2.5's template refuses OpenAI's
  JSON-string arguments; `drift.py` printed the refusal above. So after a
  restart, after 4,096 newer replies, or for a conversation begun elsewhere,
  the project's server fails the request. This engine reads the arguments
  back into an object, renders the turn, and counts it as `not_restored`
  (0 in this run). This was found by reading the project's code against the
  template's refusal, not by running the project's server.
- **Text is restored, not ids**, as counted above: 7 of 298 replies, 2 of 48
  rollouts in the project's GRPO run.
- The project's engine reports no timings and lowers `top_logprobs` above 20
  to 20 without saying so; this engine adds `timings` to each answer and
  refuses 21.

## The files

Line counts are honest ones: all lines, and in brackets the lines of code
(no blank lines, comments or docstrings).

- `engine.py`, 802 lines (383 of code): the server. It checks each request,
  puts the model's own text back in place of a harness's copy, renders the
  conversation into tokens, refuses a prompt over 16,384 tokens, takes the
  best prompt cache, samples on the one worker thread, and answers in
  OpenAI's shape with vLLM's token fields. It follows the project's
  `mhrl/engine/policy.py` and `server.py` (869 lines, 500 of code, with
  `toolcalls.py`), without the adapter route and with more: stricter request
  checks, the port check, timings, and a count of replies whose text cuts
  into other tokens. The engine's own two files, `engine.py` and
  `toolcalls.py`, are 1,024 lines (479 of code); with the model half
  (`policy.py`, `lfm.py`, `lfm_text.py`), 1,521 lines (642 of code).
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
