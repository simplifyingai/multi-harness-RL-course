# Chapter 16. What a harness is: Claude Code from the inside

You may use Claude Code every day. Each time it works on a task, it sends a
model a request, reads the reply, maybe runs a command, and sends another
request. This chapter shows you those requests, byte for byte. It replaces
the model with a small program of our own, the **stand-in**, which answers
from a fixed script and writes down everything it receives. Then it runs
the real Claude Code, version 2.1.290 (the version the project pins),
against it. No model, no account and no API key are involved, which is why
a Claude Code user can start the course here, right after chapter 1.

The puzzle to keep in mind: chapter 1's job is one short question about a
small database, handed to the agent as a 64-token instruction. How much
does the model read before it writes the first word of its first reply?
And when a reply goes wrong (it cannot be streamed, a command prints far
too much, the model names a tool that does not exist), what does Claude
Code do about it? By the end of the chapter you will have measured all of
it.

## Words first

- A **model** (also called the **policy**) is the trained network: text in,
  text out. It remembers nothing between two requests; everything it should
  know must be in the request.
- A **token** is a piece of text the model reads as one unit, often a word
  or part of a word (chapter 9 shows how text is cut). Sizes in this
  chapter are counted with the tokenizer of the course's model,
  LFM2.5-2.6B.
- A **model API** is the way a program asks a model for a reply over the
  network: it sends (POSTs) the conversation and the list of tools the
  model may use, and gets back the reply and the **usage** (how many tokens
  were read and written). A **provider** is whoever answers. A program
  finds its provider with two values: a **base URL** (the address) and an
  **API key** (a secret sent with every request).
- **Anthropic's Messages API** is the dialect Claude Code speaks:
  `POST /v1/messages`, with the key in an `x-api-key` header. A request
  holds a `system` prompt, a list of `messages` (each with a role:
  `user`, `assistant`, and in Claude Code 2.1.290 also `system`) and a list
  of `tools`. A message's content is a list of **blocks**: `text`,
  `tool_use` (the model asks for a tool), `tool_result` (the tool's output,
  sent back) or `thinking`.
- **Tool calling**: instead of only writing text, the model replies with a
  `tool_use` block, for example the tool `Bash` with the input
  `{"command": "sqlite3 data.db .tables"}`. The program runs it and sends
  the output back as a `tool_result` in the next request. Each tool comes
  with a **JSON schema**, a description of the inputs it accepts.
- The **system prompt** is the text a program puts first in every request:
  who the model is, the rules, how to behave. Each program writes its own.
- **Streaming** with **server-sent events (SSE)**: the reply arrives in
  small pieces, each an `event:` line and a `data:` line, so a program can
  show words as they are written.
- The **harness** is the program around the model. It runs the loop,
  decides which tools exist, writes everything the model reads, runs the
  tools, and decides when to stop. Claude Code is a harness.
- The **agent loop**: call the model; if the reply asks for a tool, run it
  and append the result; call again; stop when a reply asks for no tool (or
  a budget runs out). That last rule is Claude Code's **stop condition**.
- A **call** (the course's word since chapter 4) is one request to the
  model inside the loop; a **rollout** is one whole attempt at a task.
- A **side call** is a request a harness makes for its own purposes (a
  title, a summary, a check), outside the agent's loop. Here it is spotted
  as a request that offers no tools or names another model.
- **Context construction**: on every call the harness decides what history
  and what tool output the model will see, and in what form.
- **Headless mode**: `claude -p "<task>"` runs one task with no screen to
  type into and exits. With `--output-format stream-json --verbose` it
  prints every message as one line of JSON.
- A **CLAUDE.md** is a notes file Claude Code reads from the project
  folder; a **skill** is a folder of instructions Claude Code can load on
  demand; an **MCP server** is a separate program that adds tools (MCP is
  the Model Context Protocol).
- The **sandbox** is the box the commands actually run in. This chapter has
  none: every command comes from our script. Chapter 19 builds one.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/16-inside-claude-code
./run_claude.sh
uv run inspect_calls.py
uv run inspect_calls.py solve --system
uv run stand_in.py --show
./run_claude.sh --setup yours --name yours
./run_claude.sh --folder extras --name ours-extras
uv run inspect_calls.py --compare yours ours-extras
uv run inspect_calls.py yours --system
./run_claude.sh --setup bare --name bare
uv run inspect_calls.py --compare solve bare
uv run inspect_calls.py bare --system
uv run mishaps.py
uv run inspect_calls.py mishap-not-streamed
```

You need [uv](https://docs.astral.sh/uv/), Node.js with `npx` (Claude Code
is installed from npm; `brew install node`), and `curl` and `sqlite3`,
which macOS ships with. Your own Claude Code, if you have one, is never
used or touched: the first run installs version 2.1.290 with `npx` into
`COURSE_HOME/16-inside-claude-code/claude-code` (default
`~/.cache/mhrl-course`), with its own npm cache and an empty npm
configuration, so your npm settings are never read either. Every run gets
its own folder under `COURSE_HOME/16-inside-claude-code/runs/`. The task's
database is bundled in `data/` (chapter 1's `clinic.db`, byte for byte).
`inspect_calls.py` counts tokens with LFM2.5-2.6B's `tokenizer.json`: the
copy chapters 9 and 10 downloaded if you have it, otherwise it downloads
that one 18 MB file.

| command | what it shows | seconds in the run log |
| --- | --- | ---: |
| `./run_claude.sh` | the install, the prompt, Claude Code's events, the grade | 15 (with the install) |
| `inspect_calls.py` | every call: system prompt, tools, messages, tokens, headers | 0 |
| `inspect_calls.py solve --system` | the system prompt, the messages and the Bash tool in full | 0 |
| `stand_in.py --show` | a request and the server-sent events that answer it | 1 |
| `./run_claude.sh --setup yours --name yours` | Claude Code's defaults plus a CLAUDE.md, a skill and an MCP tool | 2 |
| `./run_claude.sh --folder extras --name ours-extras` | our setup on the same folder | 2 |
| `inspect_calls.py --compare yours ours-extras` | the two first calls side by side | 1 |
| `inspect_calls.py yours --system` | where the CLAUDE.md, the skills and the agents go | 0 |
| `./run_claude.sh --setup bare --name bare` | our setup plus `--bare` | 2 |
| `inspect_calls.py --compare solve bare` and `bare --system` | how small `--bare` makes the request | 1, 0 |
| `mishaps.py` | the three mishaps, four Claude Code runs | 9 |
| `inspect_calls.py mishap-not-streamed` | the first mishap, call by call | 0 |

Every output below is copied from one recorded run of these commands, from
an empty `COURSE_HOME/16-inside-claude-code`, on an M3 Pro with 18 GB while
another chapter's training ran on the same machine (the run log keeps each
command's commit, exit code and time; the times are whole seconds).
Everything is determined by the script except the times, the session
ids, the dates and the name of the file Claude Code saves a long output
to. Paths are shown with your home folder as `~`, as the scripts print
them.

## A model API in one request: `stand_in.py`

`stand_in.py` is a small web server (FastAPI) on `127.0.0.1:8016`, an
address only your own machine can reach. It answers `POST /v1/messages`
the way Anthropic's API does, and a token-counting route
(`/v1/messages/count_tokens`). Any other address is written down and
refused. Every request is appended, whole, to the run's `calls.jsonl`,
with its headers. The dummy key in `x-api-key` is kept on purpose; any
`Authorization` or cookie header, and any key that looks like a real
Anthropic key (`sk-ant-…`), is replaced by `<not logged>`.

The replies come from a script, `SCRIPTS["solve"]`: three `Bash` calls and
one plain answer. It picks the step by counting the replies already in the
conversation, so a request sent twice gets the same step twice. A request
that offers no tools is treated as a side call and gets the text `OK`.
`--show` prints, without starting the server, a short request and the
stand-in's answer to it:

```bash
uv run stand_in.py --show
```

```text
POST /v1/messages
{
  "model": "stand-in",
  "max_tokens": 32000,
  "stream": true,
  "tools": [
    {
      "name": "Bash",
      "input_schema": {
        "type": "object"
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "There is a SQLite database named data.db in the current folder. Answer this question using the data:\n\nHow many patients live in Murcia?\n\nWhen you have the answer, write ONLY the answer (no explanation, no units) to a file named answer.txt in the current folder. The answer is a single number."
    }
  ]
}

the answer, as server-sent events:
event: message_start
data: {"type": "message_start", "message": {"id": "msg_0001", "type": "message", "role": "assistant", "model": "stand-in", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 98, "output_tokens": 0}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_0001", "name": "Bash", "input": {}}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"sqlite3 data.db .tables\"}"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 26}}

event: message_stop
data: {"type": "message_stop"}
```

That is the whole idea of a model API: the conversation and the tools go
in; a reply and its usage come out. The reply opens empty
(`message_start`), its one content block (a `tool_use`) arrives as a start,
a piece of JSON and a stop, and `message_delta` closes it with the
**stop reason**: `tool_use` means "run this and call me again"; a plain
answer ends with `end_turn`. A real provider counts tokens with its
tokenizer; the stand-in only estimates them as characters divided by four,
so its `usage` numbers are rough. `inspect_calls.py` counts properly.

## Claude Code against the stand-in: `run_claude.sh`

```bash
./run_claude.sh
```

```text
installing @anthropic-ai/claude-code@2.1.290 with npx into ~/.cache/mhrl-course/16-inside-claude-code/claude-code
run       solve · script solve · setup ours · folder plain · bash cap 4000
folder    ~/.cache/mhrl-course/16-inside-claude-code/runs/solve
claude    2.1.290 (Claude Code) · sha256 b8412a3826b2dc8e
stand-in  http://127.0.0.1:8016 · script solve · logging every request to calls.jsonl
prompt    (claude -p's argument, printed by tasks.py)
          | There is a SQLite database named data.db in the current folder. Answer this question using the data:
          |
          | How many patients live in Murcia?
          |
          | When you have the answer, write ONLY the answer (no explanation, no units) to a file named answer.txt in the current folder. The answer is a single number.
claude    exit 0 after 1 s
stream    9 events printed by Claude Code (stream.jsonl)
  init       tools [Bash] · model stand-in · bypassPermissions · key from ANTHROPIC_API_KEY · version 2.1.290
  assistant  tool_use Bash: sqlite3 data.db .tables
  user       tool_result: appointments  doctors       patients
  assistant  tool_use Bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE city = 'Murcia'"
  user       tool_result: 30
  assistant  tool_use Bash: echo 30 > answer.txt
  user       tool_result: (Bash completed with no output)
  assistant  text: 30 patients live in Murcia; answer.txt holds 30.
  result     success · 4 turns · stop end_turn · 30 patients live in Murcia; answer.txt holds 30.
stderr    [claude-code:unrecognized_model] {"model":"stand-in","query_source":"sdk"}
requests  4 model calls reached the stand-in (0 side calls)
question   How many patients live in Murcia?
answer.txt 30
verdict    correct
reward     1
```

Read it as the agent loop. The "model" (our script) asked for three
commands; Claude Code ran each one for real in the task folder and sent
back its output: the three table names, then `30`, then the note
`(Bash completed with no output)` for the `echo` (it writes the file and
prints nothing). The fourth reply asked for no tool, so Claude Code
stopped: that is its stop condition, and the `result` event says
`success` after 4 turns. Chapter 1's grader read `answer.txt` and said
correct. On its error stream Claude Code noted that `stand-in` is not a
model it knows; it carried on anyway.

How the script runs it, setting by setting. The first group is the
**environment**: the run starts from an empty one (`env -i`), so nothing
from your shell is inherited, and gets only these:

| setting | value | why |
| --- | --- | --- |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8016` | the provider's address: the stand-in, on this machine only |
| `ANTHROPIC_API_KEY` | `any-string-works` | the key; the stand-in accepts any string |
| `CLAUDE_CONFIG_DIR` | `runs/<name>/config` | Claude Code's settings, history and memory, private to the run: your `~/.claude` is never read |
| `HOME`, `TMPDIR`, `CLAUDE_CODE_TMPDIR` | `runs/<name>/home`, `runs/<name>/tmp` | nothing lands in your dotfiles or in `/tmp` |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_TELEMETRY`, `DISABLE_ERROR_REPORTING`, `DISABLE_AUTOUPDATER` | `1` | no usage reports, no error reports, no update checks, from the first run |
| `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` | `off` | the project found that 2.1.x otherwise adds a `<total_tokens>` line to the system prompt on every call |

The second group is the **command line**, the project's own
configuration from `mhrl/harnesses/claude_code.py`:

| flag | why |
| --- | --- |
| `-p "<instruction>"` | headless: one task, then exit |
| `--output-format stream-json --verbose` | one line of JSON per event (the lines summarised above) |
| `--model stand-in` | the model name every request carries |
| `--permission-mode bypassPermissions` | run Bash without asking; safe here only because every command comes from our script |
| `--tools Bash` | offer one tool instead of the full list |
| `--strict-mcp-config --setting-sources ""` | no MCP servers and no settings files |
| `--settings '{"attribution": …, "includeGitInstructions": false, "bashOutputMaxChars": 4000}'` | no git instructions, no commit or pull-request attribution, and Bash output capped at 4,000 characters |
| `--disable-slash-commands --no-session-persistence` | no slash commands, no saved session |

The installed binary is the project's, byte for byte (`15-pinned-binary`
in the run log: the npm copy and the project's
`tools/claude-code/2.1.290/claude` have the same SHA-256,
`b8412a3826b2dc8e…`); the install takes 342M on disk (`du -sh`).

## Inside every call: `inspect_calls.py`

```bash
uv run inspect_calls.py
```

```text
tokens    counted with LFM2.5-2.6B's tokenizer (~/.cache/mhrl-course/models/LFM2.5-2.6B/tokenizer.json)
run solve · 4 model calls · ~/.cache/mhrl-course/16-inside-claude-code/runs/solve/calls.jsonl
call 1  POST /v1/messages · model stand-in · stream yes · the agent's loop
  system    3 blocks · 5,683 characters · 1,311 tokens
  tools     [Bash] · 688 tokens
  messages  [user, system] · 262 tokens
  total     2,261 tokens
  reply     Bash: sqlite3 data.db .tables
call 2  POST /v1/messages · model stand-in · stream yes · the agent's loop
  system    the same 5,683 characters as before
  tools     the same
  messages  [user, system, assistant tool_use, user tool_result] · 280 tokens (+18)
  total     2,279 tokens
  reply     Bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE city = 'Murcia'"
call 3  POST /v1/messages · model stand-in · stream yes · the agent's loop
  system    the same 5,683 characters as before
  tools     the same
  messages  [user, system, assistant tool_use, user tool_result, assistant tool_use, user tool_result] · 305 tokens (+25)
  total     2,304 tokens
  reply     Bash: echo 30 > answer.txt
call 4  POST /v1/messages · model stand-in · stream yes · the agent's loop
  system    the same 5,683 characters as before
  tools     the same
  messages  [user, system, assistant tool_use, user tool_result, assistant tool_use, user tool_result, assistant tool_use, user tool_result] · 323 tokens (+18)
  total     2,322 tokens
  reply     text: 30 patients live in Murcia; answer.txt holds 30.
side calls          0 of 4
same system prompt  yes, across the 4 calls of the loop
headers   x-api-key any-string-works · anthropic-version 2023-06-01 · user-agent claude-cli/2.1.290 (external, sdk-cli)
          anthropic-beta: 9 flags
body      max_tokens 32000 · thinking adaptive · effort high · metadata user_id
Bash tool  description 1,022 characters
          input: command (string, required) · timeout (number) · description (string) · run_in_background (boolean) · dangerouslyDisableSandbox (boolean)
```

What a harness sends, call after call:

- **The same system prompt every time**: 3 blocks, 5,683 characters,
  1,311 tokens, identical in all 4 calls. The model has no memory, so the
  rules are sent again with every request.
- **The same tools every time**: one, `Bash`, whose schema lists five
  inputs (only `command` is required) and whose description is 1,022
  characters: 688 tokens.
- **The conversation, growing**: call 1 carries the task and a second
  message with the role `system` (the environment: the working folder,
  the date, the model's name). Each later call adds the previous reply
  (`assistant tool_use`) and the tool's output (`user tool_result`): 18 to
  25 tokens per round here, because these outputs are short.
- **Everything, every time**: call 4 resends all of call 3 plus one round.
  The model reads 2,261 tokens before its first reply, and 2,322 before
  its fourth.
- **Who is asking, with what key**: the key travels in `x-api-key`; the
  model name in the body. Claude Code also asks for adaptive thinking at
  "high" effort, allows replies of up to 32,000 tokens, and switches on 9
  beta features by name in the `anthropic-beta` header.
- **No side calls**: every request offered the tool, and all named
  `stand-in`. With non-essential traffic switched off, 2.1.290 made no
  request for itself in any of this chapter's runs (the lines
  `side calls 0 of …` in the run log).

These token counts are LFM2.5-2.6B's, part by part: the system text, the
tools as the JSON that was sent, and the text of each message. A server
would first render the request with the model's chat template (chapter 9),
which adds a few marker tokens per message, so they are close to, not
exactly, what LFM2.5 reads. The project's note in
`mhrl/harnesses/claude_code.py` says the first call "stays near 2.3K
prompt tokens"; here it is 2,261.

## The system prompt

```bash
uv run inspect_calls.py solve --system
```

The system prompt's three blocks (the long one cut here):

```text
===== system block 1 (74 characters)
x-anthropic-billing-header: cc_version=2.1.290.212; cc_entrypoint=sdk-cli;

===== system block 2 (62 characters)
You are a Claude agent, built on Anthropic's Claude Agent SDK.

===== system block 3 (5,547 characters)

You are an interactive agent that helps users with software engineering tasks.
…
# Harness
 - Text you output outside of tool use is displayed to the user as Github-flavored markdown in a terminal.
…
 - Prefer the dedicated file/search tools over shell commands when one fits. Independent tool calls can run in parallel in one response.
…
# Memory

You have a persistent file-based memory at `~/.cache/mhrl-course/16-inside-claude-code/runs/solve/config/projects/~--cache-mhrl-course-16-inside-claude-code-runs-solve-task/memory/`. This directory already exists — write to it directly with the Write tool (do not run mkdir or check for its existence). …
…
# Environment
…
# Context management
…
```

and the message with the role `system`, which Claude Code 2.1.290 sends as
the conversation's second message:

```text
===== message 2 (role system), block 1: 814 characters
# Environment
You have been invoked in the following environment:
 - Primary working directory: ~/.cache/mhrl-course/16-inside-claude-code/runs/solve/task
 - Is a git repository: false
 - Platform: darwin
 - Shell: bash
 - OS Version: Darwin 25.6.0
 …
You are powered by the model stand-in.

Today's date is 2026-10-12.
```

Four things to notice:

- The first block is not an instruction at all: it labels the request with
  Claude Code's version for Anthropic's billing. A model still reads it.
- The prompt holds the run folder's path (in the memory section), so its
  length changes by a few characters with the folder's name: 5,683
  characters for the run called `solve`, 5,695 for `ours-extras`, 5,711
  for `mishap-not-streamed` (all in the run log). Within one run it never
  changes.
- Trimming the tools did not trim the prompt: it still says to prefer
  "the dedicated file/search tools" and to save memories "with the Write
  tool", and the Bash description (printed in full by `--system`) says to
  avoid `cat`, `head` and `echo` in favour of dedicated tools. With
  `--tools Bash` none of those tools exist: the prompt was written for the
  full tool list, and trimming the list leaves it describing tools that
  are not there.
- The model is told what it is ("powered by the model stand-in"), where it
  is, and the date: all of it written by the harness.

## Your setup versus ours

Few people run Claude Code the way this chapter does. A typical setup has
a `CLAUDE.md`, a few skills, an MCP server or two, and every built-in
tool. `--setup yours` runs exactly that: the same task folder plus
`data/your-setup/CLAUDE.md`, the skill `data/your-setup/sqlite-basics`,
the one-tool MCP server `notes_mcp.py` (registered with `--mcp-config`),
and none of our flags except `--permission-mode bypassPermissions`. The
run `ours-extras` uses our flags on the same folder (`--folder extras`),
CLAUDE.md and skill included.

```bash
./run_claude.sh --setup yours --name yours
./run_claude.sh --folder extras --name ours-extras
uv run inspect_calls.py --compare yours ours-extras
```

```text
tokens    counted with LFM2.5-2.6B's tokenizer (~/.cache/mhrl-course/models/LFM2.5-2.6B/tokenizer.json)
the first call of the loop in each run
                                 yours     ours-extras
model calls                          4               4
tools offered                       21               1
system characters                5,848           5,695
system tokens                    1,351           1,315
tools tokens                    11,783             688
messages tokens                  2,404             264
total tokens                    15,538           2,267
CLAUDE.md sent                     yes              no
skills listed                       14               0
MCP tools                            1               0
tools only in yours: Agent, CronCreate, CronDelete, CronList, Edit, EnterWorktree, ExitWorktree, ListAgents, NotebookEdit, Read, ReportFindings, ScheduleWakeup, SendMessage, Skill, TaskStop, WebFetch, WebSearch, Workflow, Write, mcp__notes__read_note
skills listed in yours: sqlite-basics, dataviz, update-config, keybindings-help, code-review, simplify, fewer-permission-prompts, loop, claude-api, workflow-authoring, run, plugin-authoring, init, security-review
the first call of yours is 6.9 times the tokens of ours-extras's
```

Both runs solved the task in the same 4 calls with the same scripted
replies; only what the model reads differs. The typical setup's first call
is 15,538 tokens, 6.9 times ours, and 11,783 of those tokens are its 21
tools, every one riding along on every call whether the model uses it or
not. The MCP server's one tool arrives as
`mcp__notes__read_note`, beside the built-in ones. The skills cost tokens
too, even unused: 14 are listed, ours (`sqlite-basics`) and 13 that come
with Claude Code itself (the run's private config and home folders hold no
skill files: `16-yours-skill-files` in the run log). `inspect_calls.py
yours --system` shows where each part goes:

```text
===== message 1 (role user), block 1: 624 characters
<system-reminder>
Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

Contents of ~/.cache/mhrl-course/16-inside-claude-code/runs/yours/task/CLAUDE.md (project instructions, checked into the codebase):

# Notes for the clinic database
…
</system-reminder>

===== message 1 (role user), block 2: 576 characters
<system-reminder>
Attribution for git commits and pull requests you create from here on …
</system-reminder>

===== message 1 (role user), block 3: 292 characters
There is a SQLite database named data.db in the current folder. Answer this question using the data:
…
===== message 2 (role system), block 1: 8,509 characters
# Environment
…
Available agent types for the Agent tool:
…
The following skills are available for use with the Skill tool:

- sqlite-basics: How to explore a SQLite database from the shell. Use when a task mentions a .db file.
…
```

The CLAUDE.md is not in the system prompt: it is pasted into the first
user message, wrapped in a `<system-reminder>`, before the task. The
skills and the sub-agents are listed in the role-`system` message, which
grows from 814 characters to 8,509. With our flags, the same CLAUDE.md
and skill sit in the folder and are not sent at all.

### Trimming to one tool, and `--bare`

This is why the project runs Claude Code with `--tools Bash`: the
project's engine refuses prompts over 16,384 tokens (`max_prompt_tokens`
in `mhrl/config.py`), and a typical setup spends 15,538 of them on the
first call, before the model has run a single command. The project's notes also mention `--bare`, which it keeps
off. `--setup bare` adds it to our flags:

```bash
./run_claude.sh --setup bare --name bare
uv run inspect_calls.py --compare solve bare
```

```text
the first call of the loop in each run
                                 solve            bare
model calls                          4               4
tools offered                        1               1
system characters                5,683             235
system tokens                    1,311              82
tools tokens                       688             387
messages tokens                    262              64
total tokens                     2,261             533
CLAUDE.md sent                      no              no
skills listed                        0               0
MCP tools                            0               0
tools only in solve: none
skills listed in solve: none
the first call of solve is 4.2 times the tokens of bare's
```

533 tokens, against the project's "about 530". `inspect_calls.py bare
--system` shows the price: the third system block becomes two lines,

```text
===== system block 3 (99 characters)
CWD: ~/.cache/mhrl-course/16-inside-claude-code/runs/bare/task
Date: 2026-10-12
```

the Bash tool's description becomes `"execute shell commands"` (22
characters), the environment message disappears (the messages are just
`[user]`, the 64-token task), and so does every rule the long prompt
carried. The project keeps the full prompt, because the point is to train
the model inside Claude Code as people run it, not inside a stripped-down
copy.

## Three mishaps: `mishaps.py`

A harness is also what happens when things go wrong. `mishaps.py` runs
three more scripts through `run_claude.sh` and reads the logs back.

```bash
uv run mishaps.py
```

### 1. A provider that does not stream

The script `not-streamed` sends each reply as one JSON body, even though
Claude Code asked for a stream:

```text
mishap 1 · not-streamed: the provider ignores "stream": true
  ran      ./run_claude.sh --script not-streamed --name mishap-not-streamed
           exit 0; its output is in ~/.cache/mhrl-course/16-inside-claude-code/runs/mishap-not-streamed/run.txt
  call 1   asked for a stream  · got one JSON body
  call 2   asked for no stream · got one JSON body · the same request again, with stream false
  call 3   asked for a stream  · got one JSON body
  call 4   asked for no stream · got one JSON body · the same request again, with stream false
  call 5   asked for a stream  · got one JSON body
  call 6   asked for no stream · got one JSON body · the same request again, with stream false
  call 7   asked for a stream  · got one JSON body
  call 8   asked for no stream · got one JSON body · the same request again, with stream false
  8 requests; 4 of them repeat the request before them, with streaming off
  Claude Code printed 4 replies; result success: '30 patients live in Murcia; answer.txt holds 30.'
  claude    exit 0 after 0 s
  verdict    correct
```

Claude Code 2.1.290 has a fallback. When a streamed request does not come
back as events, it sends the identical request again at once with
`"stream": false`, takes the JSON, and carries on. The task is solved, at
two requests per reply (`uv run inspect_calls.py mishap-not-streamed`
shows calls 1 and 2 with the same 2,273 tokens, `stream yes` then
`stream no`). The plan expected Claude Code to print nothing here; this
version does not fail that way. Other harnesses, or other versions, may.

### 2. A command that prints too much

The script `long-output` adds one step: a query that prints every
appointment with its patient and doctor. It runs twice, once with Claude
Code's own limit for Bash output and once with ours:

```text
mishap 2 · long-output: one command prints a whole joined table
  ran      ./run_claude.sh --script long-output --name mishap-long-output-default --bash-max default
           exit 0; its output is in ~/.cache/mhrl-course/16-inside-claude-code/runs/mishap-long-output-default/run.txt
  ran      ./run_claude.sh --script long-output --name mishap-long-output --bash-max 4000
           exit 0; its output is in ~/.cache/mhrl-course/16-inside-claude-code/runs/mishap-long-output/run.txt
  command  sqlite3 data.db "SELECT a.appointment_date, p.name, p.city, d.name, d.specialty, a.fee FROM appointm…
  run by us in the task folder, it prints 25,442 characters in 400 lines
  Claude Code's default: call 3 carries 25,441 characters of it
  Claude Code printed 5 replies; result success: '30 patients live in Murcia; answer.txt holds 30.'
  claude    exit 0 after 1 s
  verdict    correct
  bashOutputMaxChars 4000 (ours): call 3 carries 2,351 characters:
    | <persisted-output>
    | Output too large (24.8KB). Full output saved to: ~/.cache/mhrl-course/16-inside-claude-code/runs/mishap-long-output/config/projects/~--cache-mhrl-course-16-inside-claude-code-runs-mishap-long-output-task/…/tool-results/….txt
    |
    | Preview (first 2KB):
    | <the preview>
    | ...
    | </persisted-output>
  the preview: the first 1,978 characters (31 lines)
  the saved file: 25,442 characters, the whole output
  Claude Code printed 5 replies; result success: '30 patients live in Murcia; answer.txt holds 30.'
  claude    exit 0 after 0 s
  verdict    correct
```

(The session folder and the file's name differ in every run, so they are
replaced by `…` here.) The same 25,442 characters reach the model in two
very different forms. With Claude Code's default limit, the next call
carries all of it (25,441 characters: the last new line is dropped). With
`bashOutputMaxChars` at 4,000, Claude Code does not cut the text at 4,000
characters: it saves the whole output to a file in its config folder and
sends the model a 2,351-character note instead, which says where the file
is and shows the first 1,978 characters. To see the rest, the model would
have to spend another call reading the file. The model never chose any of
this. The harness decided what it would see, and that is **context
construction**: one setting changed the model's view of the same command
from 25,441 characters to 2,351.

### 3. A tool name in the wrong case

The script `lowercase-bash` opens with a call to `bash`; the only tool is
`Bash`:

```text
mishap 3 · lowercase-bash: the model calls 'bash'; the tool is 'Bash'
  ran      ./run_claude.sh --script lowercase-bash --name mishap-lowercase-bash
           exit 0; its output is in ~/.cache/mhrl-course/16-inside-claude-code/runs/mishap-lowercase-bash/run.txt
  tools offered: ['Bash']
  call 1 reply   bash: sqlite3 data.db .tables
  call 2 carries tool_result (error): <tool_use_error>Error: No such tool available: bash. Tool names are case-sensitive: call Bash instea…
  call 2 reply   Bash: sqlite3 data.db .tables
  call 3 carries tool_result (ok): appointments  doctors       patients
  call 3 reply   Bash: sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE city = 'Murcia'"
  call 4 carries tool_result (ok): 30
  call 4 reply   Bash: echo 30 > answer.txt
  call 5 carries tool_result (ok): (Bash completed with no output)
  5 calls; 1 tool call came back as an error
  Claude Code printed 5 replies; result success: '30 patients live in Murcia; answer.txt holds 30.'
  claude    exit 0 after 0 s
  verdict    correct
```

The command never ran. Claude Code answered with an error result,
`<tool_use_error>Error: No such tool available: bash. Tool names are
case-sensitive: call Bash instead.</tool_use_error>`, marked `is_error`,
and the run cost one call more. Our script corrects itself on the next
call; a model may not. This is the exact text chapter 1's recorded run
shows (`01-the-job/data/transcripts/clinic-medium-001.claude-code.jsonl`):
the real LFM2.5-2.6B, untrained, called `bash` and a `write` tool that
Claude Code does not offer under `--tools Bash`, and spent its calls on
errors. Chapter 17 looks at why a model writes `bash`.

## Policy, harness, agent, sandbox

The four words the rest of the course uses, as this chapter showed them:

| word | what it is | in this chapter |
| --- | --- | --- |
| **policy** | the model's weights: a prompt in, a reply out, no memory between calls | the stand-in's script played it; chapter 10 onward it is LFM2.5-2.6B |
| **harness** | the program that runs the loop, writes the prompt, offers the tools, runs them and decides when to stop | Claude Code 2.1.290, unmodified, configured only by environment and flags |
| **agent** | a policy running inside a harness | the stand-in inside Claude Code, solving the task in 4 calls |
| **sandbox** | where the commands actually run, and what they may touch | none here (every command was ours); chapter 19 builds one |

The project trains the policy and never touches the harness. Everything in
this chapter (the 5,683-character system prompt, the role-`system`
message, the tool list, the fallback to a non-streamed request, the saved
long output, the error for `bash`) is the harness's doing, and stays the
same while the model changes.

## The answer to the puzzle

The task's instruction is 64 tokens (the `[user]` message of the `--bare`
run). On its first call the model reads 2,261 tokens with the project's
trimmed setup, 15,538 with a typical setup (21 tools, a CLAUDE.md, 14
skills, an MCP server), and 533 with `--bare`. In every run inspected
call by call, the system prompt and the tools were sent again on every
call, unchanged: the model remembers nothing, so the harness repeats
everything. And when a reply
goes wrong, Claude Code repairs what it can (it re-asks without streaming,
saves a long output to a file and shows a preview) and reports what it
cannot (`No such tool available: bash`), always as more text in the next
call. What a model sees is decided, call by call, by the harness.

## What the chapter showed, and where it differed from the plan

| the plan said | measured |
| --- | --- |
| `stand_in.py`, FastAPI: `POST /v1/messages` with SSE streaming, a token-counting route, every request logged to `calls.jsonl`, scripted answers | as planned; `--show` added; any other route is logged and refused |
| `run_claude.sh`: headless stream-json, throwaway task folder, clean environment, private config, any key, `--tools Bash`, Bash output capped at 4,000 characters, bypassPermissions | as planned, with the project's other flags (`--strict-mcp-config`, `--setting-sources ""`, `--disable-slash-commands`, `--no-session-persistence`) |
| install the pinned 2.1.290 with `npx`, into the folder | `npx --package @anthropic-ai/claude-code@2.1.290` into `COURSE_HOME/16-inside-claude-code/claude-code` (342M), the same SHA-256 as the project's pinned binary |
| the four switches set from the first run | as planned |
| Claude Code runs the scripted Bash calls; answer.txt appears; the grader says correct; it stops on a reply with no tool_use | as planned: 4 calls, the 4th plain text, `result success`, correct |
| the system prompt is thousands of characters and identical every call | 3 blocks, 5,683 characters, 1,311 tokens, identical in all 4 calls; it holds the run folder's path, so its length differs by a few characters from run to run |
| tools: [Bash] with its JSON input schema | as planned: five inputs, a 1,022-character description, 688 tokens |
| messages grow [user] -> [user, assistant tool_use, user tool_result] -> ... | corrected: 2.1.290 sends the environment as a second message with role `system`, so call 1 is `[user, system]` and call 2 `[user, system, assistant tool_use, user tool_result]`; with `--bare`, `[user]` |
| the model name and the x-api-key header | as planned (`stand-in`, `any-string-works`), plus anthropic-version 2023-06-01 and 9 beta flags |
| requests with no tools or a small model are flagged as side calls | the flag is in place; 2.1.290 made no side calls in any run here |
| a reply that is not streamed leaves Claude Code printing nothing | corrected: Claude Code re-sends the same request with `"stream": false` and finishes; 8 requests for 4 replies, correct |
| a 20,000-character query result reaches the next call cut to 4,000 characters | corrected: a 25,442-character result reaches the next call as a 2,351-character note (a saved file's path and a 1,978-character preview) under the 4,000 cap, and whole (25,441) under Claude Code's default |
| a call to lower-case `bash` comes back as an error tool_result, 'No such tool available' | as planned: `<tool_use_error>Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.</tool_use_error>`, 5 calls instead of 4 |
| one 'your setup vs ours' request diff | 15,538 against 2,267 tokens on the first call (6.9 times): 21 tools against 1, the CLAUDE.md, 14 skills, 1 MCP tool |
| tool trimming: `--tools Bash` near 2,300 tokens; `--bare` about 530 | 2,261 and 533 (LFM2.5 tokens, the request's parts counted separately) |
| folder `course/16-inside-claude-code/` | `16-inside-claude-code/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are the two notes in
  `mhrl/harnesses/claude_code.py` ("near 2.3K prompt tokens", "about
  530"), quoted beside ours above, and chapter 1's recorded transcript.
- **The article's numbers**: there are none in this chapter.
- **A difference from the project**: the project runs Claude Code inside a
  sandbox and points it at a capture proxy in front of the real model,
  with the model's name in `ANTHROPIC_DEFAULT_HAIKU_MODEL` too, so any
  side call reaches the model. Here the stand-in is the model and nothing
  is sandboxed, because no command comes from a model. Chapters 19 and 20
  add both.

## What is in this folder

- `stand_in.py`: the fake provider: the Messages API with SSE streaming,
  the token-counting route, the request log and the scripts.
- `run_claude.sh`: installs the pinned Claude Code, makes a fresh run
  folder, starts the stand-in, runs Claude Code headless with a clean
  environment, stops the stand-in, prints the events and the grade.
- `inspect_calls.py`: each call of a run with its token counts, the prompt
  in full (`--system`), two runs side by side (`--compare`).
- `mishaps.py`: the three mishaps, run and read back.
- `show_stream.py`: Claude Code's stream-json events, one line each.
- `records.py`: reading a run folder (`calls.jsonl`, `stream.jsonl`).
- `lfm_tokens.py`: finding (or downloading) and loading LFM2.5-2.6B's
  tokenizer.
- `notes_mcp.py`: the one-tool MCP server of the "your setup" run.
- `tasks.py`, `grader.py`, `grade.py`: the task, chapter 1's grader
  (unchanged) and the grading command.
- `data/`: chapter 1's `clinic.db` and task record, and the "your setup"
  `CLAUDE.md` and skill.

## Words from this chapter

- **Policy, harness, agent, sandbox**: the weights; the program around
  them; the two together; where commands run.
- **Model API, provider, base URL, API key**: a POST with the conversation
  and the tools, a reply and its usage back; whoever answers; its address;
  the secret sent with every request.
- **Anthropic Messages API**: `POST /v1/messages`, `x-api-key`, a system
  prompt, messages made of text, tool_use and tool_result blocks.
- **Tool calling, tool schema**: the model asks for a tool by name with
  inputs; the schema says which inputs a tool takes.
- **Streaming, server-sent events**: a reply sent in pieces, each an
  `event:` and a `data:` line.
- **System prompt**: the harness's instructions at the top of every call.
- **Agent loop, stop condition**: call, run the tool, call again; Claude
  Code stops at a reply with no tool call.
- **Context construction**: the harness choosing what the model sees,
  such as a saved file and a preview instead of a long output.
- **Side call**: a request a harness makes for itself, outside the loop.
- **Headless mode**: `claude -p`, one task, no screen, with stream-json
  output.
- **Tool trimming**: `--tools Bash` keeps the first call near 2,261
  tokens; `--bare` shrinks it to 533 by replacing the prompt.

Next: chapter 17, tool calls at the token level: a tiny harness.
