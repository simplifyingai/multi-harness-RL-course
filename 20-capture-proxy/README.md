# Chapter 20. The capture proxy: recording a program you didn't write

Chapter 17's harness was ours: it read every token the model wrote, because
we wrote the loop. Claude Code is not ours. It runs its own loop, sends its
own requests and keeps only text, and we cannot add one line of code to it.
Yet training (chapter 15) needs, for every call it makes, the exact tokens
the model read, the exact tokens it wrote and the probability of each.

The puzzle: how do you record a program you cannot change? The answer is to
stand where it already sends its calls. Every harness can be pointed at a
model provider by two settings, a base URL and an API key. Put a program at
that URL that looks like a normal provider, passes each call to our engine
(chapter 18), and writes down what comes back. That program is OpenEnv's
**capture proxy**, the piece of the article's code this course uses, and
this chapter runs it in front of our engine and probes each of its rules.

## Words first

- A **proxy** is a program that sits between two others and passes their
  messages along. The **capture proxy** passes a harness's model calls to
  the engine, and records each one: the engine's receipts (chapter 18), in
  a graph of calls, ready to become training rows (chapter 21).
- **OpenEnv** is Hugging Face's library for agent environments; the capture
  proxy is one module of it (`openenv.core.harness.capture`). We install
  only that module and what it imports, at the commit the project pins,
  `3f14355`.
- A **session** is one rollout's record on the proxy. Its id becomes the
  harness's **API key**, so every call the harness makes arrives labelled
  with its rollout, and the harness needs no change at all: the **session id
  is the API key**.
- The **session lifecycle** is the project's (`mhrl/rollout.py`): `POST
  /sessions` opens one, the harness runs, `GET /sessions/{id}/rollout`
  fetches the record, `DELETE /sessions/{id}` forgets it.
- The **capture level** says what the engine behind the proxy can return:
  `tokens` (prompt ids, sampled ids and their log-probabilities: trainable),
  `logprobs` (log-probabilities with no token ids) or `text` (neither, like
  OpenAI's or Anthropic's hosted APIs). The proxy **measures** it with a
  probe of the engine before it listens, and prints one line.
- The **vLLM token fields** are chapter 18's receipts: `prompt_token_ids`,
  `choices[0].token_ids`, and each log-probability naming its token as
  `token_id:<n>`.
- A log-probability is **processed** when it describes the distribution the
  token was drawn from (the scores divided by the temperature), and **raw**
  when it is taken before the temperature. GRPO's ratio (chapter 6) needs
  processed ones; vLLM's default is raw.
- A session's **purpose** is `train` (sampled at temperature 1.0, recorded
  for training) or `eval` (temperature 0, never trained on); the proxy
  calls the result the **rollout type**.
- **Sampling policy pinning**: in a session the proxy overwrites the
  sampling settings a harness sends (temperature, and `top_p` and friends,
  which cut off the unlikely tokens) with the session's own.
- The **output clamp** lowers any `max_tokens` above a cap, here 1,024, the
  project's longest reply.
- The **model-call budget** is the most calls one session may forward, here
  8, the project's limit. The proxy answers the next call itself.
- An **API dialect** is a request format. mini-swe-agent and Pi speak
  **OpenAI chat** (`POST /v1/chat/completions`, key in `Authorization`);
  Claude Code speaks **Anthropic Messages** (`POST /v1/messages`, key in
  `x-api-key`, replies as content blocks: chapter 16). The proxy turns both
  into one shape for the engine.
- **Synthetic streaming**: Claude Code asks for its reply as a **stream** of
  server-sent events (chapter 16). The proxy always asks the engine for one
  whole reply, records it, then replays it to a caller that asked for a
  stream as a stream.
- **Proxy access control**: a call whose API key is not a session id gets
  `401`. The routes that open and read sessions take an **admin key**,
  which the proxy demands before it listens on anything but this machine.
- A **tap** is this chapter's recording relay: a session can name its own
  engine, so `probes.py` names a small relay that passes everything to the
  real engine unchanged and writes it down. That is how we see exactly what
  the proxy sends the engine.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/20-capture-proxy
ENGINE_PORT=8119 ./start.sh
uv run probes.py level
uv run probes.py key
uv run probes.py sampling
uv run probes.py clamp
uv run probes.py dialects
uv run probes.py budget
uv run session_demo.py
./start.sh stop
ENGINE_PORT=8119 ENGINE_FLAGS=--no-token-fields ./start.sh
uv run probes.py level
uv run session_demo.py
uv run session_demo.py --purpose eval
./start.sh stop
ENGINE_PORT=8119 ENGINE_FLAGS=--raw-logprobs ./start.sh
uv run probes.py level
./start.sh stop
```

`start.sh` takes the engine's port as a setting, `ENGINE_PORT` (default
8100, the project's), and points the proxy's `--llm-url` at it; the proxy
keeps 8200. The run log shows 8119 because another program holds 8100 on
the Mac that made it.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, chapter
11's 4-bit model in `COURSE_HOME/models` (default `~/.cache/mhrl-course`),
and, the first time, the network: `start.sh` installs the proxy from
GitHub into `COURSE_HOME/20-capture-proxy/proxy-env` (`./start.sh install`
does only that). `probes.py` and `session_demo.py` need only Python's
standard library. `./start.sh stop` stops only what `start.sh` started.

Every output below is copied from the run log, made at one commit from a
clean `COURSE_HOME/20-capture-proxy`, each engine start waiting first for a
minute with no other heavy job on the machine. The seconds are the
command's wall time. `start.sh` leaves the engine and the proxy running in
the background, so their memory is given separately: the engine's is MLX's
peak, which it prints when it stops; the proxy's was 0.05 to 0.12 GB.

| command | what it shows | seconds |
| --- | --- | ---: |
| `./start.sh install` | OpenEnv's capture proxy at `3f14355`, 74 packages, 87 MB | 3.2 |
| `ENGINE_PORT=8119 ./start.sh` | `engine ready`, then the proxy's `capture level: tokens (train rollouts)` | 6.8 |
| `probes.py level` | a training session opens; the proxy's probe, seen from a tap | 2.2 |
| `probes.py key` | 401 for a made-up key and for none; the refusal on `0.0.0.0` | 1.6 |
| `probes.py sampling` | what the engine receives in a training and an evaluation session | 3.1 |
| `probes.py clamp` | 32,000 tokens asked, 1,024 received | 3.5 |
| `probes.py dialects` | OpenAI and Anthropic requests in one shape; the Anthropic reply streamed | 4.7 |
| `probes.py budget` | the ninth call of an 8-call session | 9.7 |
| `session_demo.py` | the tiny harness with the session id as its key, then the record | 10.6 |
| `./start.sh stop` | `engine stopped · 54 requests · peak memory 2.17 GB` | 2.1 |
| `ENGINE_FLAGS=--no-token-fields ./start.sh` | `capture level: logprobs (eval rollouts)` | 6.9 |
| `probes.py level` | the training session refused | 1.8 |
| `session_demo.py` | refused; `--purpose eval`: solved, `trainable: false` | 0.1; 12.5 |
| `ENGINE_FLAGS=--raw-logprobs ./start.sh` | `capture level: logprobs (eval rollouts)` again | 6.6 |
| `probes.py level` | the probe's gap does not move with the temperature: raw | 2.1 |

The two other engines peaked at 2.14 GB (`--no-token-fields`, 13 requests)
and 1.86 GB (`--raw-logprobs`, 12 requests), the weights included.

## Installing only the proxy

```text
installing OpenEnv's capture proxy (commit 3f14355) into ~/.cache/mhrl-course/20-capture-proxy/proxy-env
  uv venv --python 3.13 ~/.cache/mhrl-course/20-capture-proxy/proxy-env
  uv pip install --python ~/.cache/mhrl-course/20-capture-proxy/proxy-env/bin/python --no-deps -r proxy-requirements.txt
  installed: OpenEnv 3f14355631513f51c4b3211b32a069ead0faf837
```

`proxy-requirements.txt` holds OpenEnv at the project's commit and every
package the proxy imports, each at the version in the project's `uv.lock`,
installed with `--no-deps`. The run log's count and `uv pip check`:

```text
packages installed: 74
Found 6 incompatibilities
The package `openenv` requires `requests>=2.25.0`, but it's not installed
The package `openenv` requires `typer>=0.9.0`, but it's not installed
The package `openenv` requires `huggingface-hub>=1.29.0`, but it's not installed
The package `openenv` requires `tomli>=2.3.0`, but it's not installed
The package `openenv` requires `tomli-w>=1.2.0`, but it's not installed
The package `openenv` requires `gradio>=4.0.0`, but it's not installed
size:  87M
```

The six missing ones are OpenEnv's own dependencies that the proxy never
imports; leaving them out is the point. The other 73 are not all the
proxy's: starting it imports `openenv.core` first, which needs `fastmcp`
and `openai` and what they bring.

## The stack: `start.sh`

```text
capture proxy ready: OpenEnv 3f14355631513f51c4b3211b32a069ead0faf837 in ~/.cache/mhrl-course/20-capture-proxy/proxy-env
starting the engine on http://127.0.0.1:8119
  engine ready · lfm2.5-2.6b 4-bit · 127.0.0.1:8119 · 1.6 GB
    ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours (ours, built by chapter 11's convert.py), loaded in 1.6 s
    one worker thread; each request prints a line here
starting the capture proxy on http://127.0.0.1:8200, in front of http://127.0.0.1:8119
  capture level: tokens (train rollouts)
```

The last line is the proxy's own. Before it listened, it probed the engine
with real requests and decided the engine is trainable. `./start.sh
status` then shows what it reports on `/health`:

```text
http://127.0.0.1:8200/health       {"status":"ok","instance":"c8100cbd1e0841b189a1b4f450a80c67","upstream":"http://127.0.0.1:8119","engine":"unknown","model":"lfm2.5-2.6b","sessions":0,"require_registered":true,"capture_level":"tokens","rollout_type":"train","upstreams":[],"upstream_auth":false,"param_fixes":[]}
```

## How the proxy measures an engine: `probes.py level`

```text
1. what the proxy measured when it started (GET /health)
  engine http://127.0.0.1:8119 · capture level tokens · train rollouts

2. a training session on that engine (POST /sessions, purpose train)
  opened: s611ca6fd127737e7861a094c · capture level tokens · rollout type train

3. how the proxy measures an engine: its probe, seen from a tap
  GET /v1/models
    back: served: lfm2.5-2.6b
  POST /v1/chat/completions · 'Reply with the single word: ok' · temperature 0.0 · max_tokens 8 · logprobs true · top_logprobs 0 · return_token_ids true
    back: prompt_token_ids (17) · token_ids (8) · 8 logprobs, tokens named by token_id:<n>
  POST /v1/chat/completions · 'Use the bash tool to list the files in /tmp.' · tools: bash · temperature 1.0 · max_tokens 512
    back: prompt_token_ids (83) · token_ids (85) · 85 logprobs, tokens named by token_id:<n> · a tool call: bash
  POST /v1/chat/completions · 'Reply with exactly: hello' · temperature 1.0 · max_tokens 1 · logprobs true · top_logprobs 5
    back: prompt_token_ids (15) · token_ids (1) · 1 logprobs, tokens named by token_id:<n> · gap between the two likeliest first tokens 2.0000
  POST /v1/chat/completions · 'Reply with exactly: hello' · temperature 2.0 · max_tokens 1 · logprobs true · top_logprobs 5
    back: prompt_token_ids (15) · token_ids (1) · 1 logprobs, tokens named by token_id:<n> · gap between the two likeliest first tokens 1.0000
  twice the temperature: gap 2.0000 -> 1.0000, ratio 0.500: processed log-probabilities
  the proxy's verdict on this engine: tokens
```

Four questions, in order. Does it answer with the token fields? (Yes: 17
prompt ids, 8 sampled ids, 8 log-probabilities named by id.) Does it take a
tool and call it? (Yes: `bash`.) And are its log-probabilities the
distribution the token was drawn from? At twice the temperature, processed
log-probabilities give half the gap between the two likeliest first tokens,
because `log_softmax(scores / T)` divides every difference by T; raw ones
give the same gap. Here it halved, so `tokens`: trainable. This is the test
chapter 18's `client.py logprobs` ran by hand, now run by the proxy itself.

## Two engines that look fine and are not

### `--no-token-fields`

The engine answers every request, just without `prompt_token_ids` or
`token_ids`, naming each token by its text, as vLLM does without its
token-id options.

```text
starting the engine on http://127.0.0.1:8119 (--no-token-fields)
  engine ready · lfm2.5-2.6b 4-bit · 127.0.0.1:8119 · 1.6 GB
  ...
    --no-token-fields: no prompt_token_ids, no token_ids, tokens named by their text
starting the capture proxy on http://127.0.0.1:8200, in front of http://127.0.0.1:8119
  capture level: logprobs (eval rollouts)
```

```text
2. a training session on that engine (POST /sessions, purpose train)
  refused: HTTP 400: training requires exact engine token capture
  an evaluation session opens: sc44ad0c1fc4eb9094cffc817 · rollout type eval

3. how the proxy measures an engine: its probe, seen from a tap
  ...
  POST /v1/chat/completions · 'Reply with the single word: ok' · temperature 0.0 · max_tokens 8 · logprobs true · top_logprobs 0 · return_token_ids true
    back: no prompt_token_ids · no token_ids · 8 logprobs, tokens named by text
  POST /v1/chat/completions · 'Use the bash tool to list the files in /tmp.' · tools: bash · temperature 1.0 · max_tokens 512
    back: no prompt_token_ids · no token_ids · 80 logprobs, tokens named by text · a tool call: bash
  the proxy's verdict on this engine: logprobs
```

The level is `logprobs`, not `text`: the switch removes the ids and keeps
the log-probabilities. A hosted API, which returns neither, measures as
`text`. Either way the rollout is evaluation only, and the probe does not
even ask about the temperature: below `tokens` nothing is trainable.
`session_demo.py` is refused, and with `--purpose eval` the harness still
solves the question, but the proxy has nothing to count with:

```text
the proxy refused a train session: HTTP 400: training requires exact engine token capture
```

```text
  grade: correct (correct; answer.txt: '30') · 7 calls

the proxy's record (GET /sessions/s8c7fc1657f2e2767cb8d20c0/rollout)
  call  read  wrote  tools  finish
     1     0      0      1  tool_calls
  ...
     7     0      0      1  stop
  the harness asked {"temperature": 0.0}; the proxy sent {"temperature": 0.0}
  turns 7 · conversations (roots) 1 · the last call read 0 tokens · the model wrote 0
  training rows 1 · trainable tokens 0 · eval rollout · trainable: false
```

The harness saw 243 to 1,045 prompt tokens per call in the engine's usage
counts; the proxy's record says 0, because it counts only token ids.

### `--raw-logprobs`

Every field is there and every number is aligned to its token. The numbers
are taken before the temperature, while the token is still drawn at it.

```text
  capture level: logprobs (eval rollouts)
```

```text
  POST /v1/chat/completions · 'Reply with exactly: hello' · temperature 1.0 · max_tokens 1 · logprobs true · top_logprobs 5
    back: prompt_token_ids (15) · token_ids (1) · 1 logprobs, tokens named by token_id:<n> · gap between the two likeliest first tokens 2.0000
  POST /v1/chat/completions · 'Reply with exactly: hello' · temperature 2.0 · max_tokens 1 · logprobs true · top_logprobs 5
    back: prompt_token_ids (15) · token_ids (1) · 1 logprobs, tokens named by token_id:<n> · gap between the two likeliest first tokens 2.0000
  twice the temperature: gap 2.0000 -> 2.0000, ratio 1.000: raw log-probabilities
  the proxy's verdict on this engine: logprobs
```

The gap did not move, so the proxy demoted the engine to `logprobs` and
refused the training session with the same 400. Its start-up line is the
same line as for the engine with no ids, though. The proxy computed its
reason and did not print it: the run log called the same function it calls
at start-up (`validate_llm`) and printed the whole report:

```text
capture_level logprobs · rollout_type eval · logprobs_mode raw · tool_support ok · prompt ids 17 · completion ids 8
finding: [FATAL] raw_logprobs: measured RAW (pre-temperature) logprobs — the same token's logprob did not change between temperature 1.0 and 2.0. Token ids are present, so nothing else here would have caught this, and the rollout would have looked perfectly trainable while carrying a wrong importance ratio. Restart the engine with --logprobs-mode processed_logprobs (SGLang from main is already temperature-scaled). Downgraded to EVAL; set OPENENV_ALLOW_RAW_LOGPROBS=1 to override.
```

For the engine with no ids the same report holds `no findings`: the level
itself is the whole explanation the proxy gives.

## Who may call: `probes.py key`

```text
1. a made-up API key ('sk-made-up-key')
  HTTP 401: unknown API key; register a session via POST /sessions

2. no API key at all
  HTTP 401: unknown API key; register a session via POST /sessions

3. a session's id as the API key
  s568c4ab16f2e9d3856d94023 · HTTP 200: answered (stop)

4. the session routes, from this machine (GET /sessions)
  answered, 1 session(s) listed: no admin key is needed while the proxy listens on 127.0.0.1 only

5. the proxy on every interface (0.0.0.0), with no admin key
  $ ~/.cache/mhrl-course/20-capture-proxy/proxy-env/bin/python -m openenv.core.harness.capture.server --llm-url http://127.0.0.1:8119 --host 0.0.0.0 --port 53913 --capture-level tokens
  exit code 1: --admin-key or $OPENENV_CAPTURE_ADMIN_KEY is required when --host=0.0.0.0 (non-loopback bind would publish /sessions* ungated). Pass --host 127.0.0.1 for a private local port, or set a key.
```

Two locks. Model calls need a session id: an open model endpoint is a GPU
anyone can spend. The session routes (open, list, read, delete) need the
admin key, but only when one is set, and the proxy sets none on
`127.0.0.1`: there, any program on this machine can open a session and so
mint itself a key (step 4 listed the sessions with no key at all). On any
other address it will not start without one.

## What a harness asks for, and what the engine gets: `probes.py sampling`

```text
a harness asks for temperature 0.2 and top_p 0.9

1. a training session: HTTP 200
  the engine received: temperature 1.0 · top_p 1.0 · top_k -1 · min_p 0.0 · frequency_penalty 0.0 · presence_penalty 0.0 · repetition_penalty 1.0
  the rollout keeps the ask: {"temperature": 0.2, "top_p": 0.9}
  and what was sent: {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}

2. an evaluation session: HTTP 200
  the engine received: temperature 0.0 · top_p 1.0
  the rollout keeps the ask: {"temperature": 0.2, "top_p": 0.9}
  and what was sent: {"temperature": 0.0, "top_p": 1.0}
```

A training session sends the policy training assumes: temperature 1.0 over
the whole distribution, every setting that cuts or bends it set to its
"off" value. A cut distribution would make each captured log-probability
too high, and GRPO's ratio wrong from step 0. An evaluation session sends
temperature 0. Its `top_p` was reset to 1.0 as well: at the `tokens` level
the proxy resets these for every request, whatever the session's purpose
(see the findings below). Both asks are kept in the record.

## The output cap: `probes.py clamp`

```text
a training session (s18edc34ff6c0cd4d1b984115); start.sh's cap: 1,024
  asked max_tokens 32,000   -> the engine received max_tokens 1024 (HTTP 200)
  asked no max_tokens       -> the engine received max_tokens 1024 (HTTP 200)
  asked max_tokens 200      -> the engine received max_tokens 200 (HTTP 200)
```

Harnesses ask for budgets far beyond any reply; the proxy's code cites one
asking for 64,000 output tokens on a 65,536-token model, after which every
call failed. Here the cap is applied before the engine sees the request.
The proxy's log says so once, `INFO intercept clamped requested output
tokens 32000 -> 1024`; a request naming no cap gets the cap too, silently.

## Two dialects, one shape: `probes.py dialects`

```text
one conversation (a system prompt, clinic-easy-001, the bash tool), twice:
  OpenAI chat:        POST /v1/chat/completions · key in Authorization
  Anthropic Messages: POST /v1/messages · key in x-api-key, streamed

1. what reached the engine
  both: POST /v1/chat/completions and POST /v1/chat/completions
  messages: identical
  tools: identical
  stream: false and false
  differs: tool_choice: (absent) and "auto"

2. what came back to each caller
  OpenAI:    application/json · one JSON body · bash {"command": "sqlite3 data.db \".tables\""}
  Anthropic: text/event-stream; charset=utf-8 · 6 events
            message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop
            bash {"command": "sqlite3 data.db \".tables\""} · stop_reason tool_use

3. what the proxy recorded
  sa2ad962119afb363d50f56b5: dialect openai_chat · 1 turn
  sce6873b9aefc69ac87834c7d: dialect anthropic · 1 turn
```

The Anthropic request reached the engine as OpenAI chat, with the same
messages and the same tool; the only difference is the `tool_choice:
"auto"` the translation adds. The engine answered both as one JSON body
(`stream: false`), and the proxy replayed the Anthropic one as six events,
the shape chapter 16 showed Claude Code expects. It streams because the
request asked for it: the same Anthropic request with no `stream` field,
measured for the run log, came back as one body:

```text
1. an Anthropic Messages request with no 'stream' field
  HTTP 200 · application/json
  one JSON body: type 'message' · content blocks ['text'] · stop_reason 'end_turn'
```

## The call budget: `probes.py budget`

```text
a training session (sd9ab6051393730ef5fa692a7) with max_model_calls 8: one conversation, one call after another
  call 1: answered by the engine · '1' · 48 tokens
  call 2: answered by the engine · '2' · 52 tokens
  ...
  call 8: answered by the engine · '8' · 107 tokens
  call 9: answered by the proxy itself · 'Step budget exhausted; stopping.' · 0 tokens
  the rollout: 8 turns recorded, in 1 conversation · budget stops: 1
```

Harnesses do not reliably honour their own step limits (the proxy's code
records opencode making 61 calls with its limit set to 3), so the proxy,
which sees every call, enforces the session's. The ninth call never reaches the
engine and is never recorded; its reply is a plain statement so that a
harness ends its loop. What the budget counts is calls forwarded to the
engine, not turns recorded, and a prompt too long for the engine is
answered the same way. Measured for the run log, in a 2-call session:

```text
2. a session with max_model_calls 2; call 1 is 20,000 words long
  call 1 (20,000 words): HTTP 200 · 'Step budget exhausted; stopping.' · finish stop · 0 tokens
  call 2 (6 words): HTTP 200 · 'ok' · finish stop · 38 tokens
  call 3 (6 words): HTTP 200 · 'Step budget exhausted; stopping.' · finish stop · 0 tokens
  the rollout: 1 turn(s) recorded · budget_stop_count 2
  finding: [WARN] context_budget_exhausted: upstream returned 400: prompt is 20,010 tokens; the limit is 16,384
```

The engine refused call 1 (`context_length_exceeded`, chapter 18); the
proxy turned that refusal into the same stop message, counted it against
the budget and recorded no turn, so call 3 found the budget spent.

The budget probe asks for a different number on each call, and that is on
purpose. In the run log's first attempt it asked "Reply with the single
word: ok" eight times; the engine puts a reply back by its visible text
(chapter 18), so every "ok" came back as the latest one, the token chain
broke from the third call on, and the record showed 8 turns in 7
conversations. Chapter 22 is about failures of exactly this kind.

## A harness through the proxy: `session_demo.py`

```text
session_demo.py · the tiny harness (chapters 17 and 18), through the proxy
  POST /sessions · purpose train · 8 calls -> s0b4386d699eff320faff55ae
  capture level tokens · rollout type train
  the harness's two settings: base URL http://127.0.0.1:8200/v1, API key s0b4386d699eff320faff55ae

clinic-easy-001 (easy): How many patients live in Murcia?
  call 1 · read 243 tokens (0 cached), wrote 56 · tool_calls · 1.1 s
    tool call    bash {"command": "sqlite3 data.db \".tables\""}
    observation  appointments  doctors       patients
  ...
  call 8 · read 993 tokens (983 cached), wrote 29 · stop · 0.5 s
    text         30
  stop: no tool call: the reply is the model's final answer
  grade: correct (correct; answer.txt: '30') · 8 calls

the proxy's record (GET /sessions/s0b4386d699eff320faff55ae/rollout)
  call  read  wrote  tools  finish
     1   243     56      1  tool_calls
     2   315     69      1  tool_calls
     3   419     67      1  tool_calls
     4   497     48      1  tool_calls
     5   555     31      1  tool_calls
     6   614    263      1  tool_calls
     7   898     85      1  tool_calls
     8   993     29      1  stop
  the harness asked {"temperature": 0.0}; the proxy sent {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
  turns 8 · conversations (roots) 1 · the last call read 993 tokens · the model wrote 648
  training rows 1 · trainable tokens 648 · train rollout · trainable: true
  saved: ~/.cache/mhrl-course/20-capture-proxy/rollouts/session/clinic-easy-001.json
  DELETE /sessions/s0b4386d699eff320faff55ae: done
```

The harness is chapter 18's `tiny_harness_http.py`, unchanged; only its two
settings point at the proxy. It asked for temperature 0, as it always
does; the training session drew every reply at 1.0, and the model still
solved the question, in 8 calls: after writing the answer at call 4 it
checked it three more times, once with `cat`, which the tool does not
allow. The proxy recorded all 8 calls as one conversation:
one training row whose 648 trainable tokens are the model's own replies.
Chapter 21 opens that row.

## Plan versus measured

| the plan said | measured |
| --- | --- |
| `start.sh` runs the engine on :8100 and the proxy on :8200 with `--llm-url http://127.0.0.1:8100 --max-output-tokens 1024` | the engine on `ENGINE_PORT` (default 8100; 8119 in the run log, where 8100 is taken), the proxy on 8200 with `--llm-url http://127.0.0.1:8119 --model lfm2.5-2.6b --max-output-tokens 1024`, as the project's `scripts/serve.sh` starts it. The proxy's own default port is the engine's, `server.py` 1394 `default=8100`, so `start.sh` always passes `--port 8200` (finding 9) |
| installed from the project's pinned commit 3f14355; only this piece of OpenEnv | `OpenEnv 3f14355631513f51c4b3211b32a069ead0faf837`, 74 packages, 87 MB; but "only this piece" needs a footnote: starting the proxy imports `openenv.core`, which needs `fastmcp` and `openai`, so 73 packages come with it; 6 of OpenEnv's dependencies (gradio, typer, huggingface-hub, requests, tomli, tomli-w) are left out, as `uv pip check` lists (finding 7) |
| `capture_client.py` is a 60-line copy of the project's `CaptureClient` | 271 lines, 117 of code (no blank, comment or docstring lines); the project's class is 67 lines, 31 of code. Ours adds the standard library's HTTP in place of httpx, error messages, and the rollout summary (finding 8) |
| start-up: `capture level: tokens (train rollouts)` | `capture level: tokens (train rollouts)`, the proxy's own line |
| with `--no-token-fields` it reports a lower level and refuses a training session (evaluation only, like every hosted API) | `capture level: logprobs (eval rollouts)` and `HTTP 400: training requires exact engine token capture`. The level is `logprobs`, not `text`: the switch keeps the log-probabilities; a hosted API, which returns neither, measures as `text` (`validate_llm.py` 211; finding 2) |
| with `--raw-logprobs` its probe catches untempered log-probabilities | caught: gap 2.0000 at both temperatures, demoted to `logprobs`, training refused. But the start-up line says only `capture level: logprobs (eval rollouts)`, the same as for the engine with no ids: the reason, `[FATAL] raw_logprobs: measured RAW (pre-temperature) logprobs ...`, stays in the probe's report (`server.py` 1452 prints the level only; `validate_llm.py` 712-720 writes the reason). `probes.py level` shows the cause (finding 1) |
| a made-up key gets 401 | `HTTP 401: unknown API key; register a session via POST /sessions`, and the same for no key |
| localhost only; any other interface needs an admin key | `exit code 1: --admin-key or $OPENENV_CAPTURE_ADMIN_KEY is required when --host=0.0.0.0`. But on `127.0.0.1` the admin key is unset, so any program on the machine can open a session and mint itself a key (`GET /sessions` answered with no key; `server.py` 704-707 and 1456-1464; finding 4) |
| a training request (temperature 0.2, top_p 0.9) reaches the engine as temperature 1.0 over the full distribution, an evaluation request as 0 | training: `temperature 1.0 · top_p 1.0 · top_k -1 · min_p 0.0` and neutral penalties; evaluation: `temperature 0.0 · top_p 1.0`: top_p is reset in evaluation too, at the `tokens` level, against the proxy's own comment (`upstream.py` 139-141 says evaluation keeps what the harness asked; 238-244 reset by level, not purpose; finding 3) |
| `max_tokens` 32,000 is clamped to 1,024 | `max_tokens 1024` at the engine; a request naming none gets 1024 too; 200 stays 200 |
| Anthropic and OpenAI requests reach the engine in one shape; the Anthropic one is replayed as a stream | identical `messages` and `tools`, both `stream: false`; one difference, `tool_choice: "auto"` added to the Anthropic one; replayed as 6 server-sent events because the request asked for a stream; without `stream` it came back as one JSON body (`server.py` 299 asks the engine with `stream = False`, 965 and 1222-1227 replay only when asked; finding 5) |
| the ninth call of an 8-call session is answered 'Step budget exhausted; stopping.' and never recorded | `call 9: answered by the proxy itself · 'Step budget exhausted; stopping.' · 0 tokens`; 8 turns recorded. The budget counts calls forwarded, not turns recorded: a 20,000-word prompt the engine refused was answered with the same message, counted, and not recorded (`server.py` 1119 counts on forward, 1136-1145 answer `context_length_exceeded` with the stop; finding 6) |
| `session_demo.py`: the session id as the key, the run, the summary (turns, prompt and sampled tokens, trainable: true) | correct in 8 calls; 8 turns in 1 conversation; the last call read 993 tokens, the model wrote 648; `trainable: true` |

## Findings about the proxy

Line numbers are in OpenEnv `3f14355`, `openenv/core/harness/capture/`.

1. **The start-up line shows the level, never the reason.** `server.py`
   1452 prints `capture level: {level} ({report.rollout_type} rollouts)`,
   and 1453-1454 add only `upstream compat` lines. The report it comes from
   holds the reason, `validate_llm.py` 712-720 (`level = "logprobs"` and
   the `[FATAL] raw_logprobs` finding), and nothing prints it. Measured:
   two different broken engines, one identical line.
2. **`--no-token-fields` measures as `logprobs`.** `validate_llm.py` 211
   grades an answer with log-probabilities but no ids as `logprobs`; a
   hosted API, which returns neither, would measure as `text`. Both are
   evaluation only.
3. **At the `tokens` level, `top_p` and its kin are reset in evaluation
   too**, against the proxy's own comment. `upstream.py` 139-141: "At the
   `tokens` tier these are therefore set to their no-op values; at
   `logprobs`/`text` they are left exactly as the harness sent them,
   because an eval rollout should score the model the harness actually
   asked for." The reset in `prepare_request` (238-244, `if capture_level
   == "tokens": ... request[key] = NON_POLICY_SAMPLING[key]`) follows the
   engine's level, not the session's purpose, so an evaluation session on a
   trainable engine does not score what the harness asked for: measured,
   `top_p 0.9` asked, `top_p 1.0` sent. At temperature 0 the reply is the
   same either way; at an evaluation temperature above 0 it is not.
4. **On `127.0.0.1`, anyone on the machine can open sessions.** The session
   routes check the admin key only when one is set (`server.py` 704-707:
   "`admin_key` defaults to unset, which keeps a local run on a private
   port exactly as convenient as before"), and `main` demands one only for
   other addresses (1456-1464). A session id is a working API key, so any
   local program can mint itself one and use the engine; measured, `GET
   /sessions` answered with no key.
5. **The stream comes back only because the request asked for one.** The
   engine is always asked for one whole reply (`server.py` 299,
   `chat_request["stream"] = False`), and the reply is replayed as a stream
   only when the caller asked (`client_wants_stream`, 965 and 1222-1227).
   Measured: the Anthropic request with `stream: true` got 6 events; the
   same request without it got one JSON body.
6. **The budget counts forwarded calls, and a context overflow is answered
   with the budget's stop message.** `server.py` 1119 counts a call before
   it is forwarded (`sessions.py` 158-163: "Counted on forward rather than
   read off the graph"), and 1136-1145 answer an engine's
   `context_length_exceeded` with `_budget_stop_response`, adding a `[WARN]
   context_budget_exhausted` finding. Measured: a 20,000-word prompt was
   answered `Step budget exhausted; stopping.`, counted, and not recorded.
7. **"Only the capture proxy" is 73 packages.** Importing
   `openenv.core.harness.capture.server` imports `openenv.core` first,
   which needs `fastmcp` and `openai`. `--no-deps` keeps gradio, typer,
   huggingface-hub, requests, tomli and tomli-w out; the rest stays: 74
   packages with OpenEnv itself, 87 MB.
8. **Honest line counts.** `capture_client.py` is 271 lines, 117 of code,
   against the plan's 60; the project's `CaptureClient` is 67 lines, 31 of
   code (`mhrl/rollout.py` 136-202). All counts are in "The files" below.
9. **The proxy's own default port is the engine's.** `server.py` 1394:
   `parser.add_argument("--port", type=int, default=8100)`. Started without
   `--port`, the proxy tries to take the engine's port; `start.sh` always
   passes `--port 8200`.

One finding about this chapter's own code, from the run log: `start.sh`
first sent Ctrl-C to the engine's whole process group, and uv passes the
signal on to its Python as well, so the engine got it twice and once ended
its log with `Exception ignored on threading shutdown: KeyboardInterrupt`
after its stop line. It now signals the group's leader only, and all three
engine logs of the final run end on their stop line.

## The files

Line counts from the run log: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `start.sh` | 271 | 208 (not blank, not comments) | installs the proxy, starts the engine and the proxy, stops only what it started |
| `proxy-requirements.txt` | | | OpenEnv at `3f14355` and the 73 packages it imports, at the project's versions |
| `capture_client.py` | 271 | 117 | the project's `CaptureClient` on the standard library: open, fetch, delete; `summarize` |
| `probes.py` | 770 | 454 | the six probes |
| `tap.py` | 309 | 166 | the recording relay, and the lines it prints |
| `session_demo.py` | 228 | 129 | the tiny harness through the proxy, and the record |
| `engine.py` | 873 | 429 | chapter 18's engine with `--no-token-fields` and `--raw-logprobs` |
| `policy.py`, `lfm.py` | 164, 244 | 37, 100 | chapter 18's seam and model, with `Sampling.raw_logprobs` |
| `tiny_harness_http.py`, `toolcalls.py`, `lfm_text.py` | 556, 222, 101 | 309, 96, 28 | chapter 18's harness, parser and tokenizer, unchanged |
| `bash_tool.py`, `database.py`, `grader.py`, `tasks.py`, `lfm_files.py` | 487, 263, 217, 155, 305 | 252, 93, 107, 60, 120 | the tool, its database guard, the grader, the questions, where things live |

`start.sh` stops only processes it started: it records each one's pid and
command, checks that a pid still runs that command and leads its own
process group before it signals it, and leaves a stale pid file's process
alone. Before restarting the engine it waits for the port: chapter 18's
engine binds without address reuse, so for about 30 seconds after a stop
its closed connections still hold the port.

## What this chapter leaves out

- **The call graph** (how calls link into conversations, forks and rows)
  is chapter 21; **captures that fail without an error** are chapter 22.
- **The real harnesses** (mini-swe-agent, Claude Code, Pi) are wired to
  this proxy in chapter 23.
- The proxy's two other dialects, OpenAI Responses and Google
  `generateContent`, and its per-session engines beyond the tap.

## Mirrors in the project

`mhrl/rollout.py` (`CaptureClient`: `open_session`, `rollout`, `close`) and
`scripts/serve.sh` (the proxy started with `--llm-url`, `--model` and
`--max-output-tokens 1024` in front of the engine on 8100).
