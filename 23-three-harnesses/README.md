# Chapter 23. Three real harnesses, wired by configuration only

Chapters 16 to 22 built every piece one at a time: Claude Code from the
inside, a tiny harness, an engine that keeps the receipts, a sandbox, a
capture proxy, and the rows a trainer learns from. This chapter plugs in
three agent programs we did not write, **mini-swe-agent**, **Claude Code**
and **Pi**, without changing a line of any of them.

The puzzle: the same small model, the same question, three programs. The
project's recording of the untrained model says one program solved it, one
fumbled its tool names, and one read the database file whole and never
answered. Can each program reach our model through its normal settings
only? And what, inside each program, makes the difference?

## Words first

- A **harness** is the program around a model: it sends the model a prompt
  and a list of tools, runs the tool calls the model writes, sends back the
  results, and decides when to stop (chapter 16). Model plus harness is the
  **agent**.
- **Wiring by configuration** means pointing a harness at our model only
  through the settings its own documentation offers for any model provider:
  a base URL and an API key, in its own settings file or environment. The
  harness's code is never touched. Here the base URL is chapter 20's capture
  proxy, and the API key is the capture session's id, so every call the
  harness makes lands in that session.
- A **pinned version** is the one exact release of a harness a result was
  measured with. Harnesses change their prompts and rules between releases,
  so a score without the harness version is incomplete. This chapter runs
  mini-swe-agent 2.4.6, Claude Code 2.1.290 and Pi 1.0.4, the project's
  versions, and refuses to run any other.
- **mini-swe-agent** is a small Python harness from the SWE-agent team, and
  the article's control harness. It gives the model one `bash` tool, runs
  each command in a fresh shell, and ends when a command prints
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
- **litellm routing**: mini-swe-agent talks to models through the litellm
  library, which picks the provider from the model name. `openai/lfm2.5-2.6b`
  means "any server that speaks OpenAI's chat API", at the `api_base` given.
- **Pi** is a minimal coding agent for Node.js with four tools (`read`,
  `bash`, `edit`, `write`). Its providers are listed in a `models.json`
  file. Pi is the project's held-out harness: it is never trained in.
- An **API dialect** is the shape a harness's requests take. mini-swe-agent
  and Pi speak OpenAI's chat completions; Claude Code speaks Anthropic's
  Messages; OpenAI's newer Responses API is a third. The proxy detects the
  dialect of each request and translates it for the engine (chapter 20).
- **Claude Code's isolation flags** keep a run from reading the machine it
  runs on: no MCP servers (`--strict-mcp-config`), no settings files
  (`--setting-sources ''`), no slash commands, no saved session, and a
  private config folder (`CLAUDE_CONFIG_DIR`).
- **Tool trimming** is offering the model fewer tools (`--tools Bash`): each
  tool's description is part of every prompt.
- **Telemetry and phone-home**: programs that report usage, errors or
  updates over the network. Every one of these switches is off here, and
  the sandbox would refuse the connection anyway.
- A **stop rule** is what ends a run: the harness's own rule (a finishing
  command, a reply with no tool call, too many unreadable replies), or the
  proxy's (the 8-call budget, or a prompt over the 16,384-token window).
- **Harness overfitting** is a model trained inside one harness learning
  that harness's habits (its tool names, its way to finish, its prompt
  layout) and doing worse in another. It is why the course trains the model
  and leaves every harness as shipped.
- A **harness log** is the harness's own account of a run:
  mini-swe-agent's trajectory file, Claude Code's stream-json (one JSON
  object per event), Pi's JSON events. The proxy keeps the model's side;
  the log keeps the harness's side, and the two can be checked against each
  other (chapter 22).

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/23-three-harnesses
uv run install.py                       # the three pinned harnesses
uv run wire.py                          # what each harness reads
uv run replay.py                        # the project's recorded runs
ENGINE_PORT=8119 ./start.sh scripted    # no model: the scripted engine on 8119, the proxy on 8200
uv run run_one.py mini-swe-agent        # one harness, one held-out task
uv run run_one.py claude-code
uv run run_one.py pi
uv run run_one.py mini-swe-agent --script mistakes   # the untrained model's mistakes, scripted
uv run run_one.py claude-code --script mistakes
uv run run_one.py pi --script mistakes
uv run run_one.py claude-code --claude-setup bare
uv run run_one.py claude-code --claude-setup default-tools
uv run run_one.py pi --pi-provider openai
uv run run_one.py pi --pi-provider openai --pi-api default
uv run run_one.py pi --pi-api default
./start.sh stop
ENGINE_PORT=8119 ./start.sh             # the model: chapter 18's engine with the 4-bit LFM2.5
uv run run_one.py mini-swe-agent
uv run run_one.py claude-code
uv run run_one.py pi
uv run run_one.py claude-code --claude-setup bare
./start.sh stop
```

`./start.sh scripted` starts the scripted engine and the capture proxy.
The scripted engine answers from a script that `run_one.py` queues, but it
lays out and counts every prompt with LFM2.5's own chat template and
tokenizer (no weights), so its token counts are the real engine's.
`./start.sh` starts the model. Both take the engine's port as `ENGINE_PORT`
(8119 in the run log, where another program holds the project's 8100); the
proxy keeps 8200.

You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac, Node.js
22.19 or later (for Claude Code and Pi), chapter 11's model in
`COURSE_HOME/models` (its tokenizer files serve the scripted engine too),
and the network for the installs.

Every output below is copied from the run log. The log was made from a clean
`COURSE_HOME/23-three-harnesses` at commit `11e5c45`, with everything under
`nice -n 10`. The model's engine started only after a minute with no film
rendering on the machine; the run waited 101 s. Seconds are wall time.
Memory is the largest single process the command ran (`/usr/bin/time -l`);
for `run_one.py` that is the harness, which runs as its child.

| command | what it shows | seconds and memory |
| --- | --- | --- |
| `uv run install.py` | the three harnesses at their pins | exit 0 · 39.4 s · max resident memory 0.31 GB |
| `uv run install.py --check` | each version, asked inside the sandbox | exit 0 · 0.4 s · max resident memory 0.10 GB |
| `uv run wire.py` | each harness's settings | exit 0 · 0.0 s · max resident memory 0.03 GB |
| `uv run replay.py` | the project's recorded runs | exit 0 · 0.0 s · max resident memory 0.02 GB |
| `ENGINE_PORT=8119 ./start.sh scripted` | the proxy installed, the scripted engine, `tokens` | exit 0 · 5.6 s · max resident memory 0.14 GB |
| `uv run run_one.py mini-swe-agent` (scripted) | solved in 4 calls | exit 0 · 8.3 s · max resident memory 0.25 GB |
| `uv run run_one.py claude-code` (scripted) | solved in 4 calls | exit 0 · 2.7 s · max resident memory 0.21 GB |
| `uv run run_one.py pi` (scripted) | solved in 4 calls | exit 0 · 2.7 s · max resident memory 0.13 GB |
| `uv run run_one.py mini-swe-agent --script mistakes` | 3 unusable replies, then mini-swe-agent stops | exit 0 · 4.1 s · max resident memory 0.25 GB |
| `uv run run_one.py claude-code --script mistakes` | two tools it does not have | exit 0 · 2.4 s · max resident memory 0.21 GB |
| `uv run run_one.py pi --script mistakes` | one read of `data.db`, the window overflows | exit 0 · 1.2 s · max resident memory 0.13 GB |
| `uv run run_one.py claude-code --claude-setup bare` | 539 tokens in the first call | exit 0 · 2.5 s · max resident memory 0.21 GB |
| `uv run run_one.py claude-code --claude-setup default-tools` | 17,441 tokens: refused | exit 0 · 0.3 s · max resident memory 0.20 GB |
| `uv run run_one.py pi --pi-provider openai` | chat completions | exit 0 · 2.7 s · max resident memory 0.13 GB |
| `uv run run_one.py pi --pi-provider openai --pi-api default` | the Responses API | exit 0 · 2.7 s · max resident memory 0.13 GB |
| `uv run run_one.py pi --pi-api default` | Pi refuses the provider | exit 0 · 0.5 s · max resident memory 0.12 GB |
| `./start.sh stop` | `engine stopped · 36 requests` | exit 0 · 2.1 s · max resident memory 0.00 GB |
| `uv run pytest tests/test_23_three_harnesses.py` | 64 passed, against the real proxy | exit 0 · 32.1 s · max resident memory 0.31 GB |
| `ENGINE_PORT=8119 ./start.sh` | the model's engine (1.6 GB loaded) | exit 0 · 11.2 s · max resident memory 0.11 GB |
| `uv run run_one.py mini-swe-agent` (the model) | correct, 8 calls | exit 0 · 15.1 s · max resident memory 0.25 GB |
| `uv run run_one.py claude-code` (the model) | correct, 8 calls and the budget's stop | exit 0 · 14.9 s · max resident memory 0.21 GB |
| `uv run run_one.py pi` (the model) | 1 call, the window overflows, no answer | exit 0 · 4.0 s · max resident memory 0.13 GB |
| `uv run run_one.py claude-code --claude-setup bare` (the model) | correct, 6 calls | exit 0 · 7.7 s · max resident memory 0.20 GB |
| `./start.sh stop` | `engine stopped · 27 requests · peak memory 2.73 GB, the weights included` | exit 0 · 2.1 s · max resident memory 0.00 GB |

The proxy held 0.12 GB (scripted) and 0.11 GB (the model) just before each
stop. The test file has 66 tests at the final commit; the run log's test run
had 64 (the two added since check `wire.py`'s output and a binary tool
result in `replay.py`).

How the run log was made, in full: a first attempt at `d62b38d` stopped at
the install (npm refused one settings file loaded as both its user and its
global settings; fixed in `11e5c45`, and the log restarted from a clean
folder). `wire.py`'s output was repeated at `dd9608d`, after a fix that
gives back its heading, settings and file lines. The call-by-call view of
the model runs and the numbers no script prints were made at `bbd7a06`
(run-log helpers `model_calls.py` and `measure.py`).

## Three pinned installs: `install.py`

```text
mini-swe-agent  2.4.6 (pinned 2.4.6) · installed by install.py
  install   UV_NO_CONFIG=1 UV_CACHE_DIR=~/.cache/mhrl-course/23-three-harnesses/uv/cache UV_PYTHON_INSTALL_DIR=~/.cache/mhrl-course/23-three-harnesses/uv/python uv venv --quiet --managed-python --python 3.13.7 ~/.cache/mhrl-course/23-three-harnesses/mini-swe-agent-env
  install   UV_NO_CONFIG=1 UV_CACHE_DIR=~/.cache/mhrl-course/23-three-harnesses/uv/cache UV_PYTHON_INSTALL_DIR=~/.cache/mhrl-course/23-three-harnesses/uv/python uv pip install --quiet --python ~/.cache/mhrl-course/23-three-harnesses/mini-swe-agent-env/bin/python --default-index https://pypi.org/simple --no-deps --compile-bytecode -r mini-swe-agent-requirements.txt
  runs as   ~/.cache/mhrl-course/23-three-harnesses/mini-swe-agent-env/bin/python -m minisweagent
claude-code     2.1.290 (pinned 2.1.290) · installed by install.py
  install   npm_config_userconfig=~/.cache/mhrl-course/23-three-harnesses/npm/npmrc npm_config_cache=~/.cache/mhrl-course/23-three-harnesses/npm/cache npm_config_registry=https://registry.npmjs.org/ npm install --prefix ~/.cache/mhrl-course/23-three-harnesses/claude-code --save-exact --no-audit --no-fund @anthropic-ai/claude-code@2.1.290
  runs as   ~/.cache/mhrl-course/23-three-harnesses/claude-code/node_modules/.bin/claude
  binary    @anthropic-ai/claude-code-darwin-arm64@2.1.290
pi              1.0.4 (pinned 1.0.4) · installed by install.py
  install   npm_config_userconfig=~/.cache/mhrl-course/23-three-harnesses/npm/npmrc npm_config_cache=~/.cache/mhrl-course/23-three-harnesses/npm/cache npm_config_registry=https://registry.npmjs.org/ npm install --prefix ~/.cache/mhrl-course/23-three-harnesses/pi --save-exact --no-audit --no-fund @earendil-works/pi-coding-agent@1.0.4
  runs as   /opt/homebrew/bin/node ~/.cache/mhrl-course/23-three-harnesses/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js
  node      /opt/homebrew/bin/node v22.22.0
record    ~/.cache/mhrl-course/23-three-harnesses/installs.json
```

- **mini-swe-agent** goes into the chapter's own Python environment, on the
  project's Python 3.13.7, from `mini-swe-agent-requirements.txt`: every
  package at the version in the project's lock, installed without resolving
  (`--no-deps`), as `start.sh` installs the proxy.
- **Claude Code** and **Pi** go into their own folders with `npm install
  --prefix`.
- **Neither installer reads your settings.** uv runs with `UV_NO_CONFIG=1`
  and a private home, cache and Python, so no `uv.toml`, `~/.netrc` or
  `UV_*` variable of yours can send it to another index. npm gets empty
  files in place of your `~/.npmrc` (which may name a private registry and
  its token) and Homebrew's global settings, a private cache, and the public
  registry.
- **npm runs the packages' install scripts outside the sandbox**, as any
  npm install does: Claude Code's script puts its native binary in place.
  The sandbox guards the runs, not the installs.
- **Any other version is refused**, never replaced. The version is read from
  the package's own record on disk, then asked of the program itself inside
  chapter 19's sandbox, with a private `HOME`, so even the version probe
  cannot read your configuration. The record is `installs.json`.

## Wiring by configuration: `wire.py`

```text
proxy http://127.0.0.1:8200 · session id (the API key) s0123456789abcdef01234567
━━ mini-swe-agent
  model.model_name             openai/lfm2.5-2.6b
  model.model_kwargs.api_base  http://127.0.0.1:8200/v1
  model.model_kwargs.api_key   s0123456789abcdef01234567  (the session id)
  agent.step_limit             8
  file      ~/.cache/mhrl-course/23-three-harnesses/wired/mini-swe-agent/state/mini_swe_agent.run.yaml
  command   ~/.cache/mhrl-course/23-three-harnesses/mini-swe-agent-env/bin/python -m minisweagent -c mini_swe_agent.yaml -c ~/.cache/mhrl-course/23-three-harnesses/wired/mini-swe-agent/state/mini_swe_agent.run.yaml -t '<the task>' --yolo --exit-immediately -o ~/.cache/mhrl-course/23-three-harnesses/wired/mini-swe-agent/state/trajectory.json
  env       22 variables; its own: LITELLM_LOCAL_MODEL_COST_MAP, MSWEA_CONFIGURED, MSWEA_COST_TRACKING, MSWEA_GLOBAL_CONFIG_DIR, MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT, MSWEA_SILENT_STARTUP, PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED
━━ claude-code
  ANTHROPIC_BASE_URL             http://127.0.0.1:8200
  ANTHROPIC_API_KEY              s0123456789abcdef01234567  (the session id)
  ANTHROPIC_DEFAULT_HAIKU_MODEL  lfm2.5-2.6b  (side calls too)
  tools                          Bash
  command   ~/.cache/mhrl-course/23-three-harnesses/claude-code/node_modules/.bin/claude -p '<the task>' --tools Bash --strict-mcp-config --setting-sources '' --settings '{"attribution": {"commit": "", "pr": ""}, "includeGitInstructions": false, "bashOutputMaxChars": 4000}' --permission-mode bypassPermissions --disable-slash-commands --no-session-persistence --model lfm2.5-2.6b --output-format stream-json --verbose
  env       27 variables; its own: ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_DEFAULT_HAIKU_MODEL, ANTHROPIC_MODEL, API_TIMEOUT_MS, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC, CLAUDE_CODE_MAX_OUTPUT_TOKENS, CLAUDE_CODE_TMPDIR, CLAUDE_CODE_TOTAL_TOKENS_REMINDER, CLAUDE_CONFIG_DIR, DISABLE_AUTOUPDATER, DISABLE_ERROR_REPORTING, DISABLE_TELEMETRY
━━ pi
  provider  intercept  (api openai-completions)
  baseUrl   http://127.0.0.1:8200/v1
  apiKey    s0123456789abcdef01234567  (the session id)
  window    16,384 tokens, reserve 1,024
  file      ~/.cache/mhrl-course/23-three-harnesses/wired/pi/state/pi-agent/models.json
  file      ~/.cache/mhrl-course/23-three-harnesses/wired/pi/state/pi-agent/settings.json
  command   /opt/homebrew/bin/node ~/.cache/mhrl-course/23-three-harnesses/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js -p '<the task>' --provider intercept --model lfm2.5-2.6b --tools read,bash,edit,write --mode json --no-session --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files --no-mcp --no-approve --offline
  env       18 variables; its own: PI_CODING_AGENT_DIR, PI_OFFLINE, PI_SKIP_VERSION_CHECK, PI_TELEMETRY
installed in ~/.cache/mhrl-course/23-three-harnesses
```

Each harness reads only its own kind of settings:

- **mini-swe-agent** merges a small YAML overlay over the chapter's
  `mini_swe_agent.yaml` (the project's file, unchanged): litellm's
  `openai/` route, the proxy's `/v1`, the session id as `api_key`, and 8
  steps.
- **Claude Code** reads environment variables (`ANTHROPIC_BASE_URL` is the
  proxy's root; it adds `/v1/messages` itself) and flags: one tool, no MCP,
  no settings files, the total-tokens note off (chapter 21), replies capped
  at 1,024 tokens, telemetry, error reports and updates off.
- **Pi** reads `models.json` (one provider, `intercept`, with `api:
  openai-completions`) and `settings.json` (timeouts, and compaction sized
  to the 16,384-token window).

Every harness starts from an environment with nothing of yours in it: a
private `HOME`, `TMPDIR` and config folders inside the run's state folder.

## One harness, one task: `run_one.py`

`run_one.py` does for one harness what the project's `run_rollout` does for
every attempt: version check, an evaluation session (temperature 0, an
8-call budget), the wiring, the run inside the sandbox, the grade, and what
the proxy and the harness's own log say. With the scripted engine it queues
the replies first: `--script solve` answers in the harness's own tool names
and ends the way that harness expects. The held-out task is the project's
demo question, `clinic-medium-001`.

```text
run_one.py · mini-swe-agent through the capture proxy at http://127.0.0.1:8200
  task      clinic-medium-001 (held out): Which specialty has the most doctors?
  engine    scripted at http://127.0.0.1:8119 (lfm2.5 tokens) · script: solve
version   2.4.6, asked inside the sandbox · pinned 2.4.6 · ok
session   sdc41c0540dda8e12920df6b2 · eval · temperature 0 · budget 8 calls
wired     model.model_name             openai/lfm2.5-2.6b
          model.model_kwargs.api_base  http://127.0.0.1:8200/v1
          model.model_kwargs.api_key   sdc41c0540dda8e12920df6b2  (the session id)
          agent.step_limit             8
dialect   openai_chat (the proxy read it from the request)
first     call 1 read 529 prompt tokens (system prompt, tools, task)
calls     4 agent calls · 0 side calls · 0 answered by the proxy itself
stop      mini-swe-agent's finishing command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
its log   exit_status Submitted · api_calls 4
answer    answer.txt 'General Practice' · correct
exit      0 after 8.1 s
network   lsof every 0.1 s: 59 samples over 8.0 s · connections to 127.0.0.1:8200 (the proxy) only: loopback, nothing else
log       ~/.cache/mhrl-course/23-three-harnesses/runs/mini-swe-agent/state/trajectory.json (mini-swe-agent's trajectory)
kept      ~/.cache/mhrl-course/23-three-harnesses/runs/mini-swe-agent
```

The other two, the same way:

```text
run_one.py · claude-code through the capture proxy at http://127.0.0.1:8200
dialect   anthropic (the proxy read it from the request)
first     call 1 read 2,279 prompt tokens (system prompt, tools, task)
calls     4 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
answer    answer.txt 'General Practice' · correct

run_one.py · pi through the capture proxy at http://127.0.0.1:8200
dialect   openai_chat (the proxy read it from the request)
first     call 1 read 1,501 prompt tokens (system prompt, tools, task)
calls     4 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
answer    answer.txt 'General Practice' · correct
```

The `dialect` line is the proxy's own record (`metadata.api_type` in the
rollout document): Claude Code arrives as `anthropic`, the other two as
`openai_chat`.

## What stops each harness

`--script mistakes` makes, in each harness, the kind of mistake the
untrained model made there, and lets the harness answer as it would answer
the model:

```text
run_one.py · mini-swe-agent through the capture proxy at http://127.0.0.1:8200
calls     3 agent calls · 0 side calls · 0 answered by the proxy itself
stop      3 replies in a row without a usable bash call (mini-swe-agent's max_consecutive_format_errors)
its log   exit_status RepeatedFormatError · api_calls 3
  told    Unknown tool 'sqlite3'.Missing 'command' argument in bash tool call.
  told    No tool calls found in the response. Every response MUST include at least one tool call.
  told    No tool calls found in the response. Every response MUST include at least one tool call.
```

```text
run_one.py · claude-code through the capture proxy at http://127.0.0.1:8200
calls     4 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
its log   result success · 4 turns · stop_reason end_turn
  told    Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
  told    Error: No such tool available: write
answer    answer.txt none · no answer.txt
```

```text
run_one.py · pi through the capture proxy at http://127.0.0.1:8200
first     call 1 read 1,504 prompt tokens (system prompt, tools, task)
calls     1 agent calls · 0 side calls · 1 answered by the proxy itself
stop      the context window: the engine refused a 30,895-token prompt (limit 16,384), and the proxy answered 'Step budget exhausted; stopping.', a reply with no tool call
its log   agent_end reached · tools run: read
answer    answer.txt none · no answer.txt
```

- **mini-swe-agent** gives up after three replies in a row without a usable
  `bash` call (`max_consecutive_format_errors: 3` in its config).
- **Claude Code** answers a wrong tool name with an error and carries on.
  Version 2.1.290 writes "Error: " in front of each message, and even names
  the right tool: `call Bash instead`. Its loop ends on the first reply with
  no tool call.
- **Pi** reads `data.db` with its `read` tool: the whole binary database,
  36,864 characters, comes back as text. The next prompt is 30,895 tokens,
  the engine refuses it (`context_length_exceeded`), and the proxy answers
  for the model with its stop reply, `Step budget exhausted; stopping.`,
  which has no tool call. Pi's compaction cannot help: it shortens a long
  history, and here one single tool result is bigger than the whole window.

From the run log's measurements (`26-measure.txt`):

```text
Pi's read of data.db, characters of the tool result (Pi's own JSON events)
  scripted-runs/pi-mistakes    [36864]
  model-runs/pi                [36864]
  data.db on disk: 36,864 bytes
```

## What each harness costs before the model writes a word

The first call carries the harness's system prompt, its tool descriptions
and the task. Measured in LFM2.5's tokens:

| harness | first call | from |
| --- | ---: | --- |
| mini-swe-agent | 529 | `06-mini-solve` |
| Claude Code, `--tools Bash --bare` | 539 | `12-claude-bare` |
| Pi | 1,501 | `08-pi-solve` |
| Claude Code, `--tools Bash` (the project's setup) | 2,279 | `07-claude-solve` |
| Claude Code, every tool | 17,441: refused | `13-claude-default-tools` |

```text
run_one.py · claude-code through the capture proxy at http://127.0.0.1:8200
first     call 1 was refused: 17,441 prompt tokens, over the 16,384-token window
calls     0 agent calls · 0 side calls · 1 answered by the proxy itself
stop      the context window: the engine refused a 17,441-token prompt (limit 16,384), and the proxy answered 'Step budget exhausted; stopping.', a reply with no tool call
its log   result success · 1 turns · stop_reason end_turn

Claude Code with every tool: its own log's result
  init: 19 tools offered: Task, Bash, CronCreate, CronDelete, CronList, Edit, EnterWorktree, ExitWorktree, ListAgents, NotebookEdit, Read, ReportFindings, ScheduleWakeup, SendMessage, TaskStop, WebFetch, WebSearch, Workflow, Write
  result success · 1 turns · result 'Step budget exhausted; stopping.'
```

With its default tools Claude Code offers 19 of them, and its very first
prompt is over the engine's 16,384-token window. The engine refuses call 1,
the proxy ends the run, and Claude Code never runs at all. That is the
strongest reason for the trimming flags: here they are not an optimisation,
they are what lets Claude Code start.

`--bare` replaces Claude Code's full system prompt with a short one
(chapter 16 compared the two), so the project keeps it opt-in and runs the
full prompt with one tool. The scripted engine's counts are the model's
engine's, measured on the same harnesses:

```text
first call's prompt tokens, scripted engine against the model's engine (run.json)
  mini-swe-agent     scripted    529 · model    529
  claude-code        scripted  2,279 · model  2,279
  pi                 scripted  1,501 · model  1,501
  claude-code-bare   scripted    539 · model    539
```

## Pi's provider: the `api` field picks the route, not the name

The project's notes say a Pi provider named `openai` is sent to the
Responses API. Measured on Pi 1.0.4, the name alone decides nothing:

| `models.json` provider | its `api` field | the proxy saw | result | from |
| --- | --- | --- | --- | --- |
| `intercept` (the project's) | `openai-completions` | `openai_chat` | correct | `08-pi-solve` |
| `openai` | `openai-completions` | `openai_chat` | correct | `14-pi-openai` |
| `openai` | not set | `openai_responses` | correct | `15-pi-openai-api-default` |
| `intercept` | not set | nothing: Pi refuses the provider | exit 1 | `16-pi-api-default` |

```text
wired     provider  openai  (api not set: Pi chooses)
dialect   openai_responses (the proxy read it from the request)
answer    answer.txt 'General Practice' · correct

Pi with provider 'intercept' and no api field: its standard error
  Error: Unknown provider "intercept". Use --list-models to see available providers/models.
```

Pi's built-in `openai` provider defaults to the Responses API; any provider
that names its `api` speaks that API whatever it is called; a provider of
our own must name one. The proxy translates the Responses dialect too.

## The model: one held-out task per harness

The engine now runs the untrained 4-bit LFM2.5-2.6B, at temperature 0, and
each harness runs once. A run is not repeated to match the recording.

```text
run_one.py · mini-swe-agent through the capture proxy at http://127.0.0.1:8200
first     call 1 read 529 prompt tokens (system prompt, tools, task)
calls     8 agent calls · 0 side calls · 0 answered by the proxy itself
stop      mini-swe-agent's finishing command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
its log   exit_status Submitted · api_calls 8
  told    Unknown tool 'sqlite3'.Missing 'command' argument in bash tool call.
  told    No tool calls found in the response. Every response MUST include at least one tool call.
answer    answer.txt 'General Practice' · correct
network   lsof every 0.1 s: 107 samples over 14.9 s · connections to 127.0.0.1:8200 (the proxy) only: loopback, nothing else
```

```text
run_one.py · claude-code through the capture proxy at http://127.0.0.1:8200
first     call 1 read 2,279 prompt tokens (system prompt, tools, task)
calls     8 agent calls · 0 side calls · 1 answered by the proxy itself
stop      the proxy's 8-call budget: it answered 'Step budget exhausted; stopping.', a reply with no tool call
its log   result success · 9 turns · stop_reason end_turn
  told    Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
  told    Error: No such tool available: Write. Write is disabled for this session, in subagents as well as here.
answer    answer.txt 'General Practice' · correct
network   lsof every 0.1 s: 106 samples over 14.7 s · connections to 127.0.0.1:8200 (the proxy) only: loopback, nothing else
```

```text
run_one.py · pi through the capture proxy at http://127.0.0.1:8200
first     call 1 read 1,501 prompt tokens (system prompt, tools, task)
calls     1 agent calls · 0 side calls · 1 answered by the proxy itself
stop      the context window: the engine refused a 30,988-token prompt (limit 16,384), and the proxy answered 'Step budget exhausted; stopping.', a reply with no tool call
its log   agent_end reached · tools run: read
answer    answer.txt none · no answer.txt
network   lsof every 0.1 s: 27 samples over 3.7 s · connections to 127.0.0.1:8200 (the proxy) only: loopback, nothing else
```

Claude Code's run, call by call, from its own stream-json log (the user name
in `ls -l` output is written `user`):

```text
━━ claude-code 2.1.290 · its log: stream-json
  call 1   bash    ls -la data.db
            error Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
  call 2   Bash    ls -la data.db
            out   -rw-r--r--@ 1 user  staff  36864 Oct 13 02:16 data.db
  call 3   Bash    sqlite3 data.db ".tables"
            out   appointments  doctors       patients
  call 4   Bash    sqlite3 data.db "SELECT * FROM doctors;"
            out   1|Leo Larsen|Dermatology
                  2|Tomas Weber|General Practice
                  3|Abigail Keller|General Practice
                  ... 27 more lines
  call 5   Bash    sqlite3 data.db "SELECT specialty, COUNT(*) as count FROM doctors GROUP BY specialty OR…
            out   General Practice|7
                  Pediatrics|5
                  Orthopedics|4
                  ... 5 more lines
  call 6   Write   {"content": "General Practice", "path": "~/.cache/mhrl-course/23-three-harnesses/runs/c…
            error Error: No such tool available: Write. Write is disabled for this session, in subagents as well …
  call 7   Bash    echo "General Practice" > ~/.cache/mhrl-course/23-three-harnesses/runs/claude-code/task…
            out   (Bash completed with no output)
  call 8   Bash    cat ~/.cache/mhrl-course/23-three-harnesses/runs/claude-code/task/answer.txt
            out   General Practice
  end       'Step budget exhausted; stopping.' (the proxy's reply, not the model's) · result success, 9 turns
```

Set beside the project's recording of the same model on the same task
(`replay.py`, below), the recording labelled as recorded:

| harness | the project's recorded run (2026-10-06) | this run |
| --- | --- | --- |
| mini-swe-agent | correct · 8 calls · the same commands | correct · 8 calls · the same commands, the same two refused replies |
| Claude Code | wrong · 7 calls · `bash` refused twice, then a `write` tool it does not have | correct · 8 calls · `bash` refused once, then `Write` refused ("Write is disabled for this session"), then `echo ... > answer.txt` through `Bash`; the 9th request got the budget's stop |
| Pi | wrong · 1 call · `read` on `data.db` | wrong · 1 call · `read` on `data.db`; the next prompt, 30,988 tokens, refused |

**A base-model run can differ from the recording, because prompts carry
the folder and the date.** Claude Code's first prompt holds its working
folder and today's date (chapter 16 printed the date line, `Today's date is
...`): the same flags cost 2,279 tokens in one run folder and 2,288 in
another whose name is 9 characters longer. Pi's system prompt has a `<cwd>`
section. So the same question in another folder, or on another day, is a
slightly different prompt, and greedy decoding can take another path from
the first token that changed:

```text
what in each prompt depends on the place: Pi's system prompt sections (its JSON events)
  sections: preamble, tools, rules, docs, cwd
  cwd: <cwd> ~/.cache/mhrl-course/23-three-harnesses/runs/pi/task </cwd>
Claude Code's first call, the same flags, two run folders (run.json)
  runs/claude-code            2,279 tokens
  runs/claude-code-mistakes   2,288 tokens
```

mini-swe-agent's prompt carries neither (its first call read 529 tokens in
every run folder, scripted or not), and it repeated the recording command
for command. Claude Code this time recovered from both refused
tools and solved the task. Pi read the binary file again, as recorded; with
the model the refused prompt was 30,988 tokens, 93 more than the scripted
run's, because the model's own reply sits in the prompt.

**A later finding (chapter 24).** `run_one.py` keeps a run's state folder
(the harness's HOME, TMPDIR and config) beside its task folder. Claude Code
keeps its auto-memory folder inside its config folder and names it in its
prompt (`memory_paths.auto` in its log's init event), so here that writable
folder sat inside the task's own tree,
`.../runs/claude-code/state/claude/projects/...-task/memory/`. Over chapter
24's 24 held-out tasks this layout misled the untrained model in 9 cells: it
wrote the right answer into that memory folder instead of the task folder,
and scored 0 there. This chapter's Claude Code run with the model wrote to
the task folder, so its outcome stands. Chapter 24 gives each cell's state a
fresh folder in the system's temp folder, as the project does
(`mhrl/harnesses/base.py`).

With `--bare`, the same model in the same Claude Code solved it in 6 calls,
starting from 539 tokens:

```text
first     call 1 read 539 prompt tokens (system prompt, tools, task)
calls     6 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
  told    Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
answer    answer.txt 'General Practice' · correct
```

## The recorded runs: `replay.py`

The project kept what each harness printed when it ran the untrained model
on this task. Each harness keeps its own kind of log, and `replay.py` reads
all three formats:

```text
replay.py · LFM2.5-2.6B as released (no training) on clinic-medium-001: Which specialty has the most doctors?
  recorded by the project on 2026-10-06 · greedy decoding · 8-call budget · expected 'General Practice'

━━ claude-code 2.1.290 · its log: stream-json
  call 1   bash    ls -la data.db
            error Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
  call 2   Bash    ls -la data.db
            out   -rw-r--r--@ 1 user  staff  36864 Oct  6 07:01 data.db
  call 3   bash    sqlite3 data.db ".tables"
            error Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.
  call 4   Bash    sqlite3 data.db ".tables"
            out   appointments  doctors       patients
  call 5   Bash    sqlite3 data.db "SELECT * FROM doctors;"
            out   1|Leo Larsen|Dermatology
                  2|Tomas Weber|General Practice
                  3|Abigail Keller|General Practice
                  ... 27 more lines
  call 6   Bash    sqlite3 data.db "SELECT specialty, COUNT(*) as count FROM doctors GROUP BY specialty OR…
            out   General Practice|7
                  Pediatrics|5
                  Orthopedics|4
                  ... 5 more lines
  call 7   write   {"file": "<task folder>/answer.txt", "content": "General Practice"}
            error Error: No such tool available: write
  end       'Step budget exhausted; stopping.' (the proxy's reply, not the model's) · result success, 9 turns
  project   wrong · answer None (expected 'General Practice') · reward 0.00 · calls 7 · 37 s
━━ pi 1.0.4 · its log: JSON events
  call 1   read    {"path": "<task folder>/data.db"}
                  (Pi's count: the call read 1,510 tokens)
            out   SQLite format 3······@  ····································… a binary file, 36,864 characters (this copy keeps 400)
  end       'Step budget exhausted; stopping.' (the proxy's reply, not the model's) · agent_end
  project   wrong · answer None (expected 'General Practice') · reward 0.00 · calls 1 · 5 s
```

The mini-swe-agent part (in `04-replay.txt`) shows a detail of how that
harness finishes. Its last call is the finishing command, and its console
then prints `action was not executed`, although the command did run:

```text
minisweagent/environments/local.py
    45      def _check_finished(self, output: dict):
    46          """Raises Submitted if the output indicates task completion."""
    47          lines = output.get("output", "").lstrip().splitlines(keepends=True)
    48          if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
    49              submission = "".join(lines[1:])
    50              raise Submitted(
minisweagent/agents/interactive.py
   124      def execute_actions(self, message: dict) -> list[dict]:
   125          # Override to handle user confirmation and confirm_exit, with try/finally to preserve partial outputs
   126          actions = message.get("extra", {}).get("actions", [])
   127          commands = [action["command"] for action in actions]
   128          outputs = []
   129          try:
   130              self._ask_confirmation_or_interrupt(commands)
   131              for action in actions:
   132                  outputs.append(self.env.execute(action))
   133          except Submitted as e:
   134              self._check_for_new_task_or_submit(e)
   135          finally:
   136              result = self.add_messages(
   137                  *self.model.format_observation_messages(message, outputs, self.get_template_vars())
   138              )
minisweagent/models/utils/actions_toolcall.py
    88      not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    89      padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
```

mini-swe-agent checks the first line of the command's **output**, not the
command: the run ends when that line is exactly
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and the exit status is 0. It stops
inside the command's execution, so no output is ever filed for that step,
and the console shows the placeholder its formatter puts in for a missing
output. The trajectory's `exit_status` is `Submitted`.

## The network: `lsof`, and it fails safe

Every tenth of a second while a harness runs, `run_one.py` lists the
internet sockets of the harness, of everything below it by parent id, and of
everything in its process group (a background command keeps the group even
when its parent exits). Every run above connected only to `127.0.0.1:8200`,
the proxy, for example:

```text
network   lsof every 0.1 s: 106 samples over 14.7 s · connections to 127.0.0.1:8200 (the proxy) only: loopback, nothing else
```

- **It fails safe.** A sample that could not be taken is counted, and a
  watch that lost any sample, or took none, does not count as loopback-only:
  `run_one.py` then exits 1, as it does for any connection off this machine.
- **It sees, it does not guarantee.** A connection that opens and closes
  between two samples, or a process that leaves both the tree and the
  group, is missed. The two shortest runs (0.2 s) saw no connection at all.
  What makes "nothing else happened" true is the sandbox's rule: only
  loopback is allowed out (chapter 19).

## Plan versus measured

| the plan said | measured |
| --- | --- |
| mini-swe-agent 2.4.6, Claude Code 2.1.290 and Pi 1.0.4 run unmodified at pinned versions, pointed at the proxy only through their normal settings, with the session id as the API key | as planned; each version is checked inside the sandbox and any other is refused |
| mini-swe-agent: model `openai/lfm2.5-2.6b`, `api_base` the proxy's `/v1`, `step_limit` 8, one native bash tool, a fresh subshell per command | as planned (`03-wire`); the dialect the proxy saw is `openai_chat` |
| finishing with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` | as planned, more precisely: mini-swe-agent checks the first line of the command's output (and exit status 0), not the command; its console then prints `action was not executed` for that step, a placeholder for an output never filed, although the command ran |
| Claude Code: `ANTHROPIC_BASE_URL` the proxy's root, the session id as `ANTHROPIC_API_KEY`, the small-model setting, a private `CLAUDE_CONFIG_DIR` and `CLAUDE_CODE_TMPDIR`, the reminder off, 1,024-token replies, `--tools Bash` and the isolation flags | as planned; the dialect the proxy saw is `anthropic` |
| Claude Code's first call: about 2,300 tokens with `--tools Bash` | 2,279, on the scripted engine and on the model's |
| about 530 with `--bare` | 539, on both |
| several times more with its default tools | 17,441 (7.7 times): over the 16,384-token window, so the engine refuses call 1 and Claude Code cannot run here with its defaults |
| Pi: a provider named `intercept` with `api: openai-completions` (a provider named `openai` is sent to the Responses API) | the `api` field picks the route, not the name: `openai` with `api: openai-completions` speaks chat completions; `openai` with no `api` goes to the Responses API; `intercept` with no `api` is refused by Pi (`Unknown provider "intercept"`) |
| Pi: compaction sized to the 16K window, `--offline`, telemetry off | as planned; compaction never runs here, because one tool result is bigger than the window |
| mini-swe-agent: 'correct · General Practice' | correct, `General Practice`, 8 calls, the recorded run command for command |
| Claude Code: 'No such tool available: bash. Tool names are case-sensitive: call Bash instead.' and 'No such tool available: write', 7 calls, no answer.txt | that is the recorded run, labelled recorded. This run: the same `bash` refusal (2.1.290 writes "Error: " in front of it), then `Write` refused, then solved: correct in 8 calls and the budget's stop. Prompts carry the folder and the date, so a base-model run can differ |
| Pi using its read tool on the binary `data.db` until the window overflows | one read: 36,864 characters, the whole file; the next prompt is 30,895 tokens (scripted) or 30,988 (the model), and the engine refuses it |
| lsof showing only loopback connections | every run: `127.0.0.1:8200` only; the watch fails safe |
| each harness leaves a log where the failures are read | mini-swe-agent's trajectory, Claude Code's stream-json, Pi's JSON events; `run_one.py` reads each one's stop and errors, and `replay.py` reads the recorded ones |
| (not in the plan) | npm runs the packages' install scripts outside the sandbox; the installers run with private settings and the public registries |
| folder `course/23-three-harnesses/` | `23-three-harnesses/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are its recorded runs
  (`data/transcripts`, from the project's run of 2026-10-06).
- **The article's numbers**: there are none in this chapter.

## Train the model, not the harness

The article closes the argument this chapter shows on one task: a model
trained inside one harness learns that harness's action format, its context
layout and its stopping rules, and it breaks or degrades inside another
(harness overfitting).
Here the same untrained model, on the same question, met three different
sets of rules: a tool named `bash` or `Bash`, a finishing command or a reply
with no tool call, a `read` tool that hands back a whole binary file. So the
course trains only the model. Training produces an adapter file for the
model, and the three harnesses stay exactly as shipped, at their pinned
versions.

## The files

Line counts from the run log: all lines, and the lines of code (no blank
lines, comments or docstrings), counted with Python's own tokenizer.

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `run_one.py` | 929 | 515 | one harness, one task: version, session, wiring, the boxed run, the grade, the stop rule, the network |
| `install.py` | 665 | 352 | the three pinned installs, with private installer settings; versions asked inside the sandbox |
| `wire.py` | 578 | 335 | each harness's own settings, pointed at the proxy |
| `replay.py` | 456 | 236 | the recorded runs, from three log formats |
| `loopback.py` | 283 | 148 | the lsof watch of a run's process tree and group |
| `lfm_template.py` | 176 | 69 | chapter 9's chat template without transformers, for the scripted engine |
| `scripted_engine.py`, `scripted_replies.py` | 594, 263 | 242, 100 | chapter 22's scripted engine, now rendering with LFM2.5's template, its replies taking seconds as the model's would |
| `mini_swe_agent.yaml`, `mini-swe-agent-requirements.txt` | 131, 91 | | the project's mini-swe-agent config, and its pinned packages |
| `start.sh` | 288 | | chapter 22's stack, without `replay` |
| `data/transcripts/` | | | the project's recorded runs, scrubbed (Pi's binary read cut, and marked) |
| `engine.py`, `lfm.py`, `lfm_text.py`, `lfm_files.py`, `policy.py`, `toolcalls.py`, `grader.py`, `tasks.py`, `data/`, `capture_client.py`, `sandbox.py`, `sandbox.sb` | | | chapter 22's, unchanged but for this chapter's folder name and a docstring line or two |

## What this chapter leaves out

- **One task is a demonstration, not a measurement.** Chapter 24 runs 24
  held-out tasks per harness and reads the gap with its noise.
- **Training through the harnesses** comes in chapters 25 and 26.

## Mirrors in the project

`mhrl/harnesses/` (`base.py`, `mini_swe_agent.py`, `claude_code.py`,
`pi.py`) and `configs/harnesses/mini_swe_agent.yaml`.
