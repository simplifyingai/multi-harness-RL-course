# Chapter 17. Tool calls at the token level: a tiny harness

Chapter 16 watched Claude Code from the outside: the requests it sends, the
tools it offers, the loop it runs. This chapter builds that loop ourselves,
in one small program, with the course's model running in the same process.
Nothing is hidden behind an API. You see the exact text the model reads and
the exact text it writes, special tokens included. You see how a harness
finds a tool call in that text, what it runs, and what it sends back.

The puzzle to keep in mind: the model never runs a command. It only writes
text. So what turns a line of text into a query on a database? And what
happens when the model's text and the harness's expectations do not match,
by one capital letter or by one pair of backticks?

## Words first

- The **policy** is the model: text in, text out, no memory between calls
  (chapter 16). The **harness** is the program around it; the **agent** is
  the two together.
- A **tool** is something the harness lets the model use. The harness
  describes it with a **tool schema**, written in JSON: a name, a
  description, and the arguments it takes. The model must use exactly those
  names. This chapter's one tool is `bash`, with one text argument,
  `command`.
- The **chat template** (chapter 9) turns a conversation into the one string
  the model reads. LFM2.5's template prints the tools into the system prompt
  as `List of tools: [...]`, and ends the prompt with `<think>`, so every
  reply starts inside the model's reasoning.
- **Special tokens** are markers the tokenizer keeps as single tokens:
  `<|im_start|>`, `<|im_end|>` (the end of a turn), `<think>`, `</think>`,
  and the two that wrap a tool call, `<|tool_call_start|>` and
  `<|tool_call_end|>`.
- **LFM2.5's tool-call syntax**: the model writes a call as Python-like text
  between those two markers, not as JSON:
  `<|tool_call_start|>[bash(command='sqlite3 data.db ".tables"')]<|tool_call_end|>`.
- A reply has three parts: the **reasoning** (before `</think>`), the
  **visible text**, and the **tool calls**.
- **State, action, observation** (chapter 4's words, for a model): each
  reply is an action, each tool output is an observation, and the database
  plus the conversation so far is the state. At the lowest level every token
  the model writes is an action.
- A **stop condition** ends the loop. Here there are three: a reply with no
  tool call, 8 calls, or 3 unreadable replies in a row. An **unreadable
  reply** has no call the harness can parse and is not a plain answer
  either.
- **White box and black box**: in this chapter, as in chapters 2 to 15, our
  own code runs the loop and sees every token: a white box. Inside a real
  harness, the harness runs the loop and a trainer only sees model calls go
  by: a black box (chapters 20 onwards).
- A **text protocol** asks for actions in plain text instead of tool-call
  tokens. mini-swe-agent's old protocol wants exactly one fenced bash block
  per reply and cuts it out with a pattern. A reply that breaks the rule is
  a **format error**, and the harness answers it with a complaint (a
  **retry**).

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
the question's database as `data.db`; every call is saved as JSON under
`COURSE_HOME/17-tiny-harness/rollouts/`. The held-out questions, their two
databases and the project's recorded Claude Code run are bundled in `data/`
(chapter 1's files, byte for byte).

| command | what it shows | seconds in the run log |
| --- | --- | ---: |
| `tiny_harness.py` | one question, call by call: the prompt, each raw reply, the parsed call, the observation, the grade | 24 |
| `tiny_harness.py --held-out 5` | the first five held-out questions, a line or two per call, and a summary | 83 |
| `experiments.py` | the tool offered as `Bash`, beside the recorded Claude Code run | 18 |
| `experiments.py --protocol text` | mini-swe-agent's text protocol: one fenced bash block per reply, or a format error | 8 |
| `experiments.py --recorded` | the recorded Claude Code run alone | 0 |

Every output below is copied from one recorded run of these commands, from
an empty `COURSE_HOME/17-tiny-harness`, on an M3 Pro with 18 GB. Another
project was rendering videos on the same machine, so before each model
command the run waited until nothing heavy had run for 60 seconds (60 or 61
seconds each time). The run log keeps each command's commit, exit code and
time. Decoding is greedy, so the replies are the same on every run: an
earlier full run, before a printing fix, gave the same replies word for
word. The times, marked "varies from run to run", are not. Paths are shown
with your home folder as `~`, as the scripts print them.

## What the model reads: the chat template's render

`tiny_harness.py` starts the conversation with two messages, a short system
prompt and chapter 1's instruction, and offers one tool. Before the first
call it prints what the template makes of them, the text the model reads:

```text
3. what the model reads on the first call: the chat template's render
  <|startoftext|><|im_start|>system
  You are a data analyst. You work in a folder that holds a SQLite database, and you can run commands there with the bash tool: sqlite3 to query the database, ls to list the folder, and echo to write the answer file. When the answer file is written, reply without a tool call to finish.
  List of tools: [{"type": "function", "function": {"name": "bash", "description": "Run one command in the task folder. Allowed: sqlite3 data.db \"<SQL, .tables or .schema>\", ls, and echo \"<answer>\" > answer.txt.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run"}}, "required": ["command"]}}}]<|im_end|>
  <|im_start|>user
  There is a SQLite database named data.db in the current folder. Answer this question using the data:

  How many patients live in Murcia?

  When you have the answer, write ONLY the answer (no explanation, no units) to a file named answer.txt in the current folder. The answer is a single number.<|im_end|>
  <|im_start|>assistant
  <think>
  (the template printed the tool's schema into the system prompt, 'List of tools: [...]'; the prompt ends with <think>, so every reply starts inside the model's reasoning)
```

The tool's schema is not sent anywhere separately: the template pastes it,
as JSON, into the system prompt, after the harness's own instructions. That
is all the model ever learns about the tool. The prompt ends with
`<think>`, written by the template, not by the model, so the model's first
token is already part of its reasoning. This first prompt is 243 tokens
(call 1 below "read 243 tokens").

## One question, call by call: `tiny_harness.py`

Each call prints the reply exactly as the model wrote it, with the
template's `<think>` in front and a long reasoning cut in the middle
(`--full` prints it whole). Then come the reply's three parts, the call as
a harness would receive it in JSON, and the observation. Calls 1 and 3:

```text
  call 1 · read 243 tokens, wrote 89 · stop · 1.5 s
    reply        <think>The user wants me to:
                 1. Look at the SQLite database named data.db in the current folder
                 2. Find out how many patients live in Murcia
                 3. Write the answer (a sin[... 78 characters ...]lore the database to understand its structure and find the relevant information.</think><|tool_call_start|>[bash(command='sqlite3 data.db ".tables"')]<|tool_call_end|><|im_end|>
    split        reasoning 318 characters · visible text: none · 1 tool call
    tool call    bash {"command": "sqlite3 data.db \".tables\""}
    observation  appointments  doctors       patients
```

```text
  call 3 · read 442 tokens, wrote 71 · stop · 1.6 s
    reply        <think>The patients table has columns: id, name, city, birth_date. I need to count how many patients live in Murcia. Let me query the patients table for patients whose city is "Murcia".</think><|tool_call_start|>[bash(command='sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE city = \'Murcia\'"')]<|tool_call_end|><|im_end|>
    split        reasoning 178 characters · visible text: none · 1 tool call
    tool call    bash {"command": "sqlite3 data.db \"SELECT COUNT(*) FROM patients WHERE city = 'Murcia'\""}
    observation  30
```

Read the reply from left to right:

- **The reasoning** runs up to `</think>`. The harness keeps it in the
  conversation but never acts on it.
- **The visible text** is everything after `</think>` that is not a tool
  call. Here there is none.
- **The tool call** sits between `<|tool_call_start|>` and
  `<|tool_call_end|>`, as a Python-style list holding one call:
  `[bash(command='sqlite3 data.db ".tables"')]`. The model quotes the
  argument with single quotes, as the chat template itself writes calls.
  When the SQL needs single quotes too, it escapes them (`\'Murcia\'`).
- **`<|im_end|>`**, the end of the turn, is the last token. That is why the
  reply's finish reason is `stop` and not `length`.

The harness reads the call with Python's `ast` module, which understands
Python syntax without running anything. It gets the name `bash` and the
argument `command`, which it prints as JSON, the way an OpenAI-style
harness would receive it. Then it runs the command and appends the output
as a `tool` message. The `.tables` output is the sqlite3 program's own
layout, names padded into columns: `appointments  doctors       patients`.
In call 2 the model read the patients table's schema, and in call 3 it
counted.

Calls 4 to 7 are where the harness shows that it is in charge:

```text
    tool call    bash {"command": "echo \"30\" > answer.txt"}
    observation  (no output)
  call 5 · read 615 tokens, wrote 48 · stop · 2.1 s
    reply        <think>The answer has been written to answer.txt. The query returned 30, which is the count of patients living in Murcia. Let me verify the file was written correctly.</think><|tool_call_start|>[bash(command='cat answer.txt')]<|tool_call_end|><|im_end|>
    split        reasoning 160 characters · visible text: none · 1 tool call
    tool call    bash {"command": "cat answer.txt"}
    observation  Error: cat: not allowed here (this tool runs sqlite3, ls, echo)
  call 6 · read 691 tokens, wrote 297 · stop · 8.9 s
    reply        <think>The cat command is not allowed through the bash tool. Let me try a different approach - I can use echo to write to the file and then verify with another method.[... 1,053 characters ...] exist.

                 Let me try to verify with ls and then write the answer again if needed.</think><|tool_call_start|>[bash(command='ls -la answer.txt')]<|tool_call_end|><|im_end|>
    split        reasoning 1,293 characters · visible text: none · 1 tool call
    tool call    bash {"command": "ls -la answer.txt"}
    observation  -rw-r--r--        3 answer.txt
  call 7 · read 1,009 tokens, wrote 136 · stop · 3.0 s
    reply        <think>The file exists and has 3 bytes (which is consistent with "30" plus a newline). The answer is 30. The file has been written correctly. Let me confirm the conten[... 267 characters ...] has been written. I should now provide the final response with just the answer.</think>30<|im_end|>
    split        reasoning 507 characters · visible text: '30' · 0 tool calls
    harness      no tool call: this is the final answer, so it stops

5. the grade
  answer.txt: '30'
  chapter 1's grader: correct · reward 1
  calls used: 7 of 8 · stop: no tool call: the reply is the model's final answer

6. what it cost (varies from run to run)
  7 calls took 21.3 s to read their prompts and write their replies (every call reads the whole conversation again)
  peak memory: 2.57 GB, the weights included
```

After writing the answer, the model wanted to check it with `cat`. This
tool does not run `cat`, so the observation was an error. The model read
the error, wrote 297 tokens of reasoning about it (its longest reply), and
found a command that is allowed: `ls -la` shows the file is 3 bytes, "30"
and a newline. Its seventh reply has no tool call, only the visible text
`30`, and that ends the loop: chapter 1's grader reads `answer.txt` and
says correct, reward 1, in 7 of 8 calls. Checking cost two calls.

The prompt grew with every call, from 243 tokens to 1,009 at call 7,
because the harness sends the whole conversation each time: the replies,
reasoning included, and every observation. Nothing is cached, so each call
reads it all again (chapter 18's engine keeps a cache). Peak memory was
2.57 GB, the weights included.

## The parser: `toolcalls.py`

`split_reply` and `parse_calls` are copies of the project's
`mhrl/engine/toolcalls.py`, cut down to what an in-process harness needs:

- Everything before `</think>` is reasoning. A call written inside the
  reasoning is not a call, and a reply that never closes its thinking is
  all reasoning.
- Each block between the markers is parsed with `ast`. Positional arguments
  are named from the tool's schema: `bash('ls')` becomes `command='ls'`.
- An argument that is not a plain value is kept as its source text.
- A block that does not parse is dropped, as the project's engine drops it.
  The tests try a dozen broken shapes. The real model wrote one in the text
  protocol below, `[sqlite3 data.db ".tables"')]`: a tool-call block with no
  tool name and a stray quote.

Unlike the project's engine, which drops a broken block silently, this
harness counts how many blocks were opened. It needs that count to tell an
unreadable call from no call at all.

## The tool: a stand-in for the sandbox, `bash_tool.py`

The model can ask for any command; the harness decides what runs. This
`bash` is not a shell. The command is cut into words the way a shell cuts
them, checked, and carried out in Python:

- `sqlite3 [options] data.db "<SQL, .tables or .schema>"` goes through
  chapter 11's guard. The database is opened read-only, as query-only, with
  `ATTACH` refused and a 10-second limit.
- `ls` lists the task folder.
- `echo "<answer>" > answer.txt` writes the answer file and nothing else.

Pipes, `;`, `&&`, `||`, other redirections and command substitution are
refused, so each call is one command, inside the task folder. Long output
is cut at 4,000 characters, the cap the project gives Claude Code. The
`sqlite3` here is an emulation, not the sqlite3 program, and that is
deliberate. It prints what sqlite3 3.51 prints (the tests compare the
outputs byte for byte), and no model-typed text ever reaches a real
program. The tests also check that the guard refuses every way out the
code review could think of: `VACUUM INTO`, turning `query_only` back off,
`.shell`, and paths outside the folder. Chapter 19 replaces this stand-in
with a real sandbox.

## The stop rules

The loop stops on whichever comes first:

1. **A reply with no tool call.** It is taken as the model's final answer.
   This is Claude Code's rule, and it ended every native-protocol run in
   this chapter.
2. **8 calls**, the project's budget (`max_model_calls`). A call is one
   reply, however many tool calls it holds.
3. **3 unreadable replies in a row.** A reply is unreadable when it opened a
   tool-call block that does not parse, or was cut off by the 1,024-token
   cap. The harness tells the model what went wrong and asks again.

The third rule is mini-swe-agent's (`max_consecutive_format_errors: 3`),
not the project engine's. The engine drops an unparseable call silently.
After that, Claude Code sees a plain-text reply and stops, while
mini-swe-agent sees "No tool calls found", which is a format error. This
course harness takes the stricter rule on purpose: it says what went wrong,
gives the model a chance to repair it, and still has an end. No native run
here needed it; the tests drive it with scripted replies.

## Five held-out questions: `--held-out 5`

The first five held-out questions in file order are two medium, two easy
and one hard. Each one is printed a line or two per call. The last question
is the hard one:

```text
  clinic-hard-001 (hard): Which doctor earned the most in fees from patients living in Seville?
  correct answer: Grace Torres
  call 1 · wrote 100 tokens · bash {"command": "sqlite3 data.db \".tables\""}
                                appointments  doctors       patients
  call 2 · wrote 63 tokens · bash {"command": "sqlite3 data.db \".schema\""}
                               CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL, birth_date TEXT NOT NULL);
                               CREATE TABLE doctors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, specialty TEXT NOT NULL);
                               ... 1 more lines
  call 3 · wrote 200 tokens · bash {"command": "sqlite3 data.db \"SELECT d.name, SUM(a.fee) AS total_fee FROM appointments a JOIN patients p ON a.patient_id = p.id JOIN doctors d ON a.doctor_id = d.id WHERE p.city = 'Seville' GROUP BY d.id, d.name ORDER BY total_fee DESC LIMIT 1\""}
                                Grace Torres|793.0
  call 4 · wrote 127 tokens · bash {"command": "echo \"Grace Torres\" > answer.txt"}
                                (no output)
  call 5 · wrote 53 tokens · bash {"command": "cat answer.txt"}
                               Error: cat: not allowed here (this tool runs sqlite3, ls, echo)
  call 6 · wrote 344 tokens · bash {"command": "ls -la answer.txt"}
                                -rw-r--r--       13 answer.txt
  call 7 · wrote 65 tokens · no tool call, visible text: 'Grace Torres'
  grade: correct · reward 1 · 7 of 8 calls · no tool call
```

The summary:

```text
2. the 5 questions
  question            level   answer.txt            grade     calls  written  stop
  clinic-medium-001   medium  General Practice      correct       7      839  no tool call
  airline-medium-001  medium  Vienna                correct       7      941  no tool call
  clinic-easy-001     easy    30                    correct       7      781  no tool call
  airline-easy-001    easy    8                     correct       7      570  no tool call
  clinic-hard-001     hard    Grace Torres          correct       7      952  no tool call
  solved 5 of 5

3. what it cost (varies from run to run)
  35 calls took 80.2 s to read their prompts and write their replies (every call reads the whole conversation again)
  peak memory: 2.57 GB, the weights included
```

All five were solved, the hard one included, and every question took
exactly the same 7 calls in the same shape. The model looked at the tables
and a schema, ran one query, wrote the answer, tried `cat`, was refused,
used `ls`, and answered. The hard question needed a three-table join, and
the model wrote it in one go.

Five questions are not a score: with five, one slip moves the result by 20
points (chapter 24 measures properly). For scale only, here are the
project's results for the same model, as released, on 24 held-out tasks
(chapter 1's `results_as_released.json`). It solved 24 of 24 under
mini-swe-agent, 15 under Claude Code and 4 under Pi.

## A name: `experiments.py`

The same harness and the same question, with the tool offered as `Bash`,
Claude Code's name for its shell tool. The system prompt names it too
("with the Bash tool"). The plan expected the model to call `bash` anyway
and lose a call to "No such tool available". It did not:

```text
  call 1 · read 245 tokens, wrote 100 · stop · 1.7 s
    reply        <think>The user wants me to analyze a SQLite database named `data.db` in the current folder and find which specialty has the most doctors. I need to:
                 1. First, explore[... 136 characters ...]te the answer to `answer.txt`

                 Let me start by examining the database structure.</think><|tool_call_start|>[Bash(command='sqlite3 data.db ".tables"')]<|tool_call_end|><|im_end|>
    split        reasoning 376 characters · visible text: none · 1 tool call
    tool call    Bash {"command": "sqlite3 data.db \".tables\""}
    observation  appointments  doctors       patients
```

```text
4. the grade
  answer.txt: 'General Practice'
  chapter 1's grader: correct · reward 1
  calls used: 7 of 8 · stop: no tool call: the reply is the model's final answer
  tool calls: 6, of which 0 answered 'No such tool available'
```

Offered `Bash` in a short prompt that says `Bash`, the model wrote `Bash`
in all 6 tool calls and solved the question in 7 calls. It took the same
steps as with `bash`, the `cat` refusal included.

A mismatch did happen in the project's recorded run, which the script
prints next, labelled as recorded. It is Claude Code 2.1.290, offering one
tool named `Bash`, with the same model and the same question, recorded by
the project with greedy decoding:

```text
5. the recorded Claude Code run: the same model, as released, on clinic-medium-001
  Claude Code 2.1.290 offered one tool, 'Bash'; recorded by the project on 2026-10-06 with greedy decoding and a budget of 8 calls (data/clinic-medium-001.claude-code.jsonl)
    bash         ls -la data.db
                 <tool_use_error>Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.</tool_use_error>
    Claude Code  Output token limit hit. (a reply hit the 1,024-token cap; Claude Code asked it to go on)
    Bash         ls -la data.db
                 -rw-r--r--@ 1 user  staff  36864 Oct  6 07:01 data.db
    bash         sqlite3 data.db ".tables"
                 <tool_use_error>Error: No such tool available: bash. Tool names are case-sensitive: call Bash instead.</tool_use_error>
    Bash         sqlite3 data.db ".tables"
                 appointments  doctors       patients
    Bash         sqlite3 data.db "SELECT * FROM doctors;"
                 1|Leo Larsen|Dermatology
                 2|Tomas Weber|General Practice
                 ... 28 more lines
    Bash         sqlite3 data.db "SELECT specialty, COUNT(*) as count FROM doctors GROUP BY specialty ORDER BY count DESC;"
                 General Practice|7
                 Pediatrics|5
                 ... 6 more lines
    write        {"file": "<task folder>/answer.txt", "content": "General Practice"}
                 <tool_use_error>Error: No such tool available: write</tool_use_error>
    reply        Step budget exhausted; stopping.
  tool calls: 7, of which 3 answered 'No such tool available' ('bash' 2, 'write' 1); the last reply is not the model's: the project's proxy writes it when the 8-call budget is used up
```

Inside Claude Code the model called `bash` twice and a `write` tool that
does not exist once: 3 of its 7 tool calls were wasted on names. It found
the answer but never wrote `answer.txt`, because its one attempt to write
it used a tool Claude Code was not offering. Names are matched exactly,
and a wrong one costs a call either way; our harness sends back Claude
Code's own words for the error (the tests check them). What differed
between the two runs is everything around the name. Claude Code's first
request was 2,261 tokens even with the project's trimmed setup (chapter
16); ours was 245.

## A protocol: `experiments.py --protocol text`

mini-swe-agent's old text protocol, as the project's
`configs/harnesses/mini_swe_agent_textbased.yaml` sets it up. No tool is
offered, and the system prompt asks for exactly one fenced bash block per
reply:

```text
3. the rule the model reads first (mini-swe-agent's system template)
  You are a helpful assistant that can interact with a computer.

  Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
  Include a THOUGHT section before your command where you explain your reasoning process.
  Format your response as shown in <format_example>.
```

The harness reads only the visible text, as mini-swe-agent does: the
project's engine hands it the reply's `content`, without the reasoning and
without any tool-call block. It cuts the action out with the YAML's
pattern, which also accepts a `sh` fence or a bare one. The first call:

```text
  call 1 · read 484 tokens, wrote 103 · stop · 2.0 s
    reply        <think>The user wants me to:
                 1. Look at the files in the folder and the database schema
                 2. Run SQL queries to find which specialty has the most doctors
                 3. Write the an[... 153 characters ...]d its structure. I'll first check what tables exist and then look at the schema.</think><|tool_call_start|>[sqlite3 data.db ".tables"')]<|tool_call_end|><|im_end|>
    visible text none  (the only part mini-swe-agent reads)
    bash blocks  0 found: format error 1 of 3
    unseen       a tool-call block that does not parse as a call: [sqlite3 data.db ".tables"')]
    harness says Format error:
                 <error>
                 Expected exactly 1 action, found 0.
                 </error>
                 Please always provide EXACTLY ONE action in triple backticks, found 0 actions.
                 ... 9 more lines
```

Calls 2 and 3 were the same. The model answered in its own tool-call
tokens even though no tool was offered, and what it wrote between them was
not a valid call: the command alone, with no tool name and a stray quote.
So there was nothing in the visible text and nothing a native harness could
have run either. After three format errors the run aborted:

```text
5. the result
  exit status: RepeatedFormatError after 3 of 8 calls
  tool-call blocks written in LFM2.5's own syntax, which this protocol never reads: 3, of which 0 parse as a call
  chapter 1's grader: wrong (no answer.txt) · reward 0
```

This matches the note in the project's YAML file ("both smoke runs ended in
RepeatedFormatError after 3 calls"). No transcript of those runs is in the
project's `runs/` folder, so this run is the first record here of what the
model wrote. The same model solved the same question in 7 calls with native
tool calls. Only the protocol changed.

## The answer to the puzzle

Nothing but the harness turns text into action. The model wrote
`<|tool_call_start|>[bash(command='sqlite3 data.db ".tables"')]<|tool_call_end|>`
from a schema it had only read as text in its system prompt. The harness
recognised the two markers, parsed the list with `ast`, matched the name
`bash` exactly against the tool it offered, checked the command against its
allow-list, ran it, and pasted the output back as more text. Everything the
model knows about tools, and everything it learns from them, is text in its
prompt.

When the two sides disagree, the harness decides what the disagreement
costs:

- A refused command cost two calls and was repaired (`cat`, then `ls`).
- In the recorded Claude Code run, a wrong name cost three calls and the
  answer file.
- A protocol the model does not speak cost the whole run in three calls.

Same weights, three different outcomes. That is chapter 1's puzzle again,
now seen at the level of single tokens.

## What the chapter showed, and where it differed from the plan

| the plan said | measured |
| --- | --- |
| `tiny_harness.py`, about 100 lines, in-process with the 4-bit model | in-process as planned; the loop itself (`run_rollout`) is short, and the parser, the tool and the printing live in their own files |
| a JSON schema for one bash tool that runs only sqlite3, ls and `echo > answer.txt` in the task folder | as planned; sqlite3 is an emulation through chapter 11's guard rather than the sqlite3 program with `-safe` (the build note's option), with `.shell` refused in `-safe`'s words |
| render with the chat template, which prints the tools into the system prompt | as planned: `List of tools: [...]` after the system prompt; the first prompt is 243 tokens |
| the raw reply `<think>...</think><\|tool_call_start\|>[bash(command="sqlite3 data.db .tables")]<\|tool_call_end\|><\|im_end\|>` | the `<think>` comes from the template's generation prompt, not the model; the model quotes with single quotes: `[bash(command='sqlite3 data.db ".tables"')]` |
| the real output (appointments doctors patients) appended as the observation | `appointments  doctors       patients`, padded into columns as sqlite3 prints it |
| parse each call with `ast`, positional arguments named from the schema, unparseable calls dropped | as planned, plus a count of the blocks opened, which tells an unreadable call from no call |
| stop on a reply with no call, after 8 calls, or after 3 unreadable replies in a row | as planned; 8 means model calls. The third rule is mini-swe-agent's, not the project engine's (which drops an unparseable call silently), taken on purpose; no native run needed it |
| `clinic-easy-001`: the grade, the calls used, the reward | correct, reward 1, 7 of 8 calls, two of them spent checking the answer file after `cat` was refused |
| five held-out questions: most solved, a hard one wrong or out of calls | all 5 solved, the hard one included, each in 7 calls |
| offered `Bash`, the model's `bash` call returns 'No such tool' and burns a call | the model called `Bash` correctly in all 6 tool calls (0 errors); the point is carried by the project's recorded Claude Code run, labelled as recorded: 3 of 7 tool calls answered 'No such tool available' (`bash` twice, `write` once) |
| `--protocol text`: three replies in native tool-call tokens, three format errors, abort | as planned: `RepeatedFormatError` after 3 of 8 calls; the three blocks did not even parse as calls (`[sqlite3 data.db ".tables"')]`) |
| folder `course/17-tiny-harness/` | `17-tiny-harness/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are the recorded Claude Code
  run (bundled from chapter 1) and the 24-task results quoted above.
- **The article's numbers**: there are none in this chapter.
- **Differences from the project**: the project's harnesses talk to the
  model over HTTP, through its engine and capture proxy, and its commands
  run in a sandbox. Here the model runs in the same process and the tool
  is a guarded emulation. Chapters 18, 19 and 20 add each of those pieces.

## What is in this folder

- `tiny_harness.py`: the loop. Render, reply, split, run, append, stop,
  grade, and save every call.
- `experiments.py`: the `Bash` name, mini-swe-agent's text protocol, and
  the recorded Claude Code run.
- `toolcalls.py`: the project's tool-call parser, `split_reply` and
  `parse_calls`.
- `bash_tool.py`: the one tool, a stand-in for chapter 19's sandbox.
- `database.py`: chapter 11's read-only guard, which every `sqlite3`
  command goes through.
- `policy.py`: the seam between harness and model (chat in, reply text
  out). The tests use it to drive the loop with scripted replies.
- `lfm.py`: the real model behind that seam, with mlx-lm.
- `grader.py`, `tasks.py`, `lfm_files.py`, `report.py`: chapter 1's grader
  (unchanged), the questions and their work folders, where things live, and
  the printing.
- `data/`: chapter 1's held-out questions and their two databases, and the
  project's recorded Claude Code run.

## Words from this chapter

- **Tool schema**: a tool described as JSON (name, description,
  parameters); LFM2.5's template prints it into the system prompt.
- **LFM2.5's tool-call syntax**: `<|tool_call_start|>[name(arg='value')]<|tool_call_end|>`,
  Python-like text between two special tokens.
- **Reasoning, visible text, tool calls**: the three parts a harness splits
  a reply into.
- **State, action, observation**: the database and the conversation; a
  reply; a tool's output.
- **Stop conditions**: no tool call, 8 calls, 3 unreadable replies in a
  row.
- **Tool-name mismatch**: a call to a name the harness does not offer
  (`bash` for `Bash`, `write`) comes back as "No such tool available" and
  costs a call.
- **Text protocol, format error, retry**: actions written as fenced text
  and cut out with a pattern; a reply without exactly one is a format
  error, answered with a complaint; three in a row end mini-swe-agent's
  run.
- **White box, black box**: we run the loop and see every token; a real
  harness runs the loop and a trainer sees only model calls.

Next: chapter 18, an engine that keeps the receipts.
