# Chapter 29. Make it yours

Twenty-eight chapters trained one small model on one kind of question
(SQLite data questions) in three harnesses. This last chapter hands the
pipeline over. It adds a task family of your own, a harness the project
never ran, and the arithmetic that says how many tasks a measurement needs.

The puzzle: you have a new kind of question and another agent program.
What do you have to change? What must you measure again? And before you
train anything, how many tasks would it take to believe a gain of ten
points?

## Words first

- A **task family** is one kind of job with its own data, its own questions
  and the gold answers to them. Chapter 1's family is SQLite questions;
  this chapter's is questions about a CSV file (a spreadsheet saved as
  text, one row per line, values separated by commas).
- The **gold answer** is the correct answer, computed by the task's maker
  from the data. The agent never sees it, and the sandbox must keep it
  out of reach (reading it would be reward hacking, chapter 19).
- A **template** is a question with blanks (`How many orders came from the
  {region} region?`), the query that answers it, and the values each blank
  may take. A **seeded** generator fills the blanks with a fixed random
  seed, so the same seed always makes the same tasks.
- The **fourth harness** is **Qwen Code**, a coding agent from Alibaba's
  Qwen team, built from Google's Gemini CLI. It installs from npm with no
  account, and its documentation lets it reach any server that speaks
  OpenAI's chat API given three settings: a base URL, a key and a model
  name. This chapter pins version 0.24.7.
- **Telemetry** (or usage statistics) is a program reporting how it is used
  to its makers over the internet. A **proxy** setting (`HTTPS_PROXY`) tells
  a program to send its outside connections through another program first.
- A **stand-in model** is a tiny server on this machine that speaks OpenAI's
  chat API, writes down every request and answers from a short script.
- **Power** is the chance that a real gain of a given size shows up as
  p < 0.05 (chapter 24's line between noise and not). 80% is the usual
  target: four times in five, the gain is seen.
- A **paired comparison** asks the same tasks of two checkpoints and reads
  only the tasks that **flip** (chapter 24): gained, or lost. The share of
  tasks that flip is the **churn**. **McNemar's exact test** treats each
  flip as a coin toss if the checkpoints are equally good.
- The **normal approximation** replaces that exact count with a bell-curve
  formula (Connor, 1987). It is close, and used only where the exact count
  would be slow.

## Run it

```bash
cd multi-harness-RL-course/29-make-it-yours
uv run make_csv_tasks.py                     # 20 CSV tasks with their gold answers
uv run power.py                              # how many paired tasks a +10-point gain needs
uv run install.py mini-swe-agent qwen-code   # the two harnesses this chapter runs
uv run wire.py qwen-code mini-swe-agent      # what each one reads
uv run qwen_probe.py                         # what Qwen Code does around the model
uv run try_family.py --plan                  # the protocol and the 20 tasks
ENGINE_PORT=8169 PROXY_PORT=8239 ./start.sh scripted   # no model
uv run run_one.py qwen-code --proxy-url http://127.0.0.1:8239
uv run run_one.py mini-swe-agent --proxy-url http://127.0.0.1:8239
uv run try_family.py --limit 3 --proxy-url http://127.0.0.1:8239
./start.sh stop
ENGINE_PORT=8169 PROXY_PORT=8239 ./start.sh            # the model
uv run try_family.py --proxy-url http://127.0.0.1:8239                # 20 tasks, mini-swe-agent
uv run run_one.py qwen-code --proxy-url http://127.0.0.1:8239         # the fourth harness, one task
./start.sh stop
```

The course's ports are the project's, 8100 for the engine and 8200 for the
proxy. This run used 8169 and 8239, because other programs on this Mac held
8100. You need [uv](https://docs.astral.sh/uv/), an Apple-silicon Mac,
Node.js 22 or later (for Qwen Code), chapter 11's model in
`COURSE_HOME/models` and the network for the installs.

## The run log

Every output below is copied from the run log. It was made from a clean
`COURSE_HOME/29-make-it-yours` at commit `42d67db`, on an M3 Pro with 18 GB,
macOS 26.6.2, Node.js 22.22.0 and uv 0.11.21. The steps without the model
ran under `nice -n 10`. The model's steps waited for the machine's shared
render lock (another chapter's film was not rendering; the lock was taken
at once) and ran at niceness 20. Seconds are wall time. Memory is the
largest single process the command ran (`/usr/bin/time -l`); for
`try_family.py` and `run_one.py` that is the harness, which runs as their
child.

| command | what it shows | exit · seconds · memory |
| --- | --- | --- |
| `uv run make_csv_tasks.py` | 20 tasks from 12 templates, md5 `cd9bdf06…` | 0 · 0.04 s · 0.03 GB |
| `uv run power.py` | 261 paired tasks for +10 points, with chapter 24's churn | 0 · 0.94 s · 0.03 GB |
| `uv run install.py mini-swe-agent qwen-code` | the two harnesses at their pins | 0 · 27.24 s · 0.18 GB |
| `uv run wire.py qwen-code mini-swe-agent` | each harness's settings | 0 · 0.05 s · 0.03 GB |
| `uv run qwen_probe.py` | five runs of Qwen Code against a stand-in model | 0 · 8.21 s · 0.36 GB |
| `uv run try_family.py --plan` | the protocol and the 20 tasks | 0 · 0.07 s · 0.03 GB |
| `./start.sh scripted` | the proxy installed, the scripted engine | 0 · 5.52 s · 0.16 GB |
| `uv run run_one.py qwen-code` (scripted) | solved in 4 calls; the first read 5,678 tokens | 0 · 3.08 s · 0.35 GB |
| `uv run run_one.py mini-swe-agent` (scripted) | solved in 4 calls; the first read 537 tokens | 0 · 7.48 s · 0.25 GB |
| `uv run try_family.py --limit 3` (scripted) | 3 of 3 | 0 · 12.75 s · 0.25 GB |
| `./start.sh stop` | `engine stopped · 24 requests` | 0 · 2.07 s · 0.00 GB |
| `uv run pytest tests/test_29_make_it_yours.py` | 48 passed, against the real proxy | 0 · 33.90 s · 0.36 GB |
| `./start.sh` | the model's engine (1.6 GB loaded) | 0 · 11.17 s · 0.11 GB |
| `uv run try_family.py` (the model) | 17 of 20, Wilson 95% 64.0% to 94.8% | 0 · 468.00 s · 0.25 GB |
| `uv run run_one.py qwen-code` (the model) | correct, 6 calls | 0 · 36.70 s · 0.31 GB |
| `./start.sh stop` | `engine stopped · 164 requests · peak memory 3.11 GB, the weights included` | 0 · 2.09 s · 0.00 GB |

The numbers no script prints (reply lengths, the wrong cells command by
command, Qwen Code's calls with the model) were read from the runs' own
records into the run log's `26-measure.txt`.

## A new task family: `make_csv_tasks.py`

The data is `data/sales.csv`, an invented spreadsheet of 96 orders of a
small shop: order id, date, region, sales rep, product, category, units
and unit price. The generator is chapter 1's `make_tasks.py`, pointed at
it: the same machinery and the same fairness rules (no answer of 0 or 1, no
tie for first place, lists of two to eight items, no answer used more than
twice). Only the twelve templates are new.

```text
seed 0 · 20 tasks from 12 templates · over sales.csv

easy    How many orders did Chloe Martin take?
        SELECT COUNT(*)
          FROM sales
          WHERE rep = 'Chloe Martin'
        answer: 18
...
medium  Which Software product brought in the most revenue (units times unit price)?
        SELECT product
          FROM sales
          WHERE category = 'Software'
          GROUP BY product
          ORDER BY SUM(units * unit_price) DESC LIMIT 1
        answer: Photo Editor
...
hard    Which sales reps took no orders in the East region?
        SELECT DISTINCT rep
          FROM sales
          WHERE rep NOT IN (SELECT rep
          FROM sales
          WHERE region = 'East')
          ORDER BY rep
        answer: Ben Ortiz, Chloe Martin, Elena Rossi

thrown away: 2 empty answer, 1 question already drafted
answers   13 number, 3 text, 4 list
wrote ~/.cache/mhrl-course/29-make-it-yours/csv-tasks/seed-0.jsonl
md5   cd9bdf067046038d76c7fcffe90f5ef4
```

The gold SQL runs on the spreadsheet loaded into a table with declared
column types (`sheet.py`). A CSV file has no types: every value in it is
text, and as text `'10' < '9'` is true. So `units` is read as an integer
and `unit_price` as a real number, and a row whose number is not a number
is refused, never half-read. The tests recompute every gold answer two other
ways: with plain Python over the CSV file, and with the `sqlite3` command
the scripted engine types in a task folder.

What a new family changed, and what it did not:

- **The data and the instruction's first sentence.** Each task folder holds
  a copy of the file as `data.csv`. The instruction is chapter 1's, word for
  word, but its first sentence: `There is a CSV file named data.csv in the
  current folder.`
- **The grader: nothing.** `grader.py` is chapter 1's, byte for byte.
  Lenient about format and strict about content, it graded this family as
  it grades SQL answers. The model run below has `4.555555555555555` for
  `4.56` and a list in another order, both correct.
- **One harness setting.** mini-swe-agent's instance template, the
  project's `mini_swe_agent.yaml`, names the job: `sqlite3 data.db
  ".tables"`, "the database schema", "Run SQL queries". Left as it was, it
  would tell the model to open a database that is not there. Its SQLite
  lines were rewritten for a CSV file (`head -5 data.csv`, a `python3` line,
  "the CSV file's header"); nothing else in the file changed. Claude Code,
  Pi and Qwen Code get only the instruction, so they needed nothing.
- **Where the gold answers live, which the box must hide.** Chapter 24's
  held-out records were bundled in its `data/` folder, and its sandbox hid
  that folder. This family's records are made under
  `COURSE_HOME/29-make-it-yours/csv-tasks/`. With chapter 24's default, a
  test shows a harness could read them inside the box: `cat` on the file
  succeeded. `run_one.py` now hides `csv-tasks/` (`tasks.gold_folder()`),
  and the same `cat` is refused with `Operation not permitted`.

## A fourth harness: Qwen Code

### Installed and wired by its own settings

```text
qwen-code       0.24.7 (pinned 0.24.7) · installed by install.py in 8.0 s
  runs as   /opt/homebrew/bin/node ~/.cache/mhrl-course/29-make-it-yours/qwen-code/node_modules/@qwen-code/qwen-code/cli-entry.js
  node      /opt/homebrew/bin/node v22.22.0
```

It installs the way chapter 23 installs Pi: `npm install --prefix` into the
chapter's folder, with an empty npm settings file in place of yours, a
private cache and the public registry. Any other version is refused.

```text
━━ qwen-code
  OPENAI_BASE_URL  http://127.0.0.1:8200/v1
  OPENAI_API_KEY   s0123456789abcdef01234567  (the session id)
  OPENAI_MODEL     lfm2.5-2.6b
  tools            edit, read_file, run_shell_command (--bare, 14 excluded)
  command   /opt/homebrew/bin/node ~/.cache/mhrl-course/29-make-it-yours/qwen-code/node_modules/@qwen-code/qwen-code/cli-entry.js '<the task>' --bare --approval-mode yolo --output-format stream-json --exclude-tools agent,list_agents,send_message,task_stop,get_goal,update_goal,tool_call,tool_search,skill,report_findings,enter_worktree,exit_worktree,record_artifact,notebook_edit --model lfm2.5-2.6b
  env       22 variables; its own: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, QWEN_CODE_DISABLE_PRECONNECT, QWEN_CODE_SUPPRESS_YOLO_WARNING, QWEN_DISABLE_AUTO_TITLE, QWEN_TELEMETRY_ENABLED, QWEN_USAGE_STATISTICS_ENABLED
```

The connection to the model is three environment variables: the proxy's
`/v1`, the capture session's id as the key, the model's name. Everything
else on that list is what the next section found.

### What it does around the model: `qwen_probe.py`

Before Qwen Code met the model, the probe ran it five times inside
chapter 19's sandbox, each run changed from `wire.py`'s wiring in one way.
The model was a stand-in on this machine. `HTTPS_PROXY` pointed at a
**catcher**: a small program on this machine that writes down where each
outside connection was going and refuses it, so nothing leaves the machine.

```text
qwen_probe.py · what Qwen Code 0.24.7 does around the model
  stand-in model  http://127.0.0.1:58101/v1 on this machine: keeps every request, answers `head -3 data.csv`, `echo 42 > answer.txt`, then text
  catcher         HTTPS_PROXY=http://127.0.0.1:58102: writes down each outside connection asked for and refuses it
  every run       inside chapter 19's sandbox, a private HOME

━━ 1. its own switches left on
  exit           0
  outside connections asked for: 2
    gb4w8c3ygj-default-sea.rum.aliyuncs.com:443 (refused by the catcher)

━━ 2. with wire.py's switches
  exit           0
  outside connections asked for: 0

━━ 3. --bare (wire.py's setup)
  exit           0
  requests       3
  first request  25,387 characters: messages 14,840, tool descriptions 10,547
  tools offered  3: edit, read_file, run_shell_command
  max_tokens     32000 asked

━━ 4. without --bare
  exit           0
  requests       4
  first request  52,510 characters: messages 24,692, tool descriptions 27,818
  tools offered  14: cron_create, cron_delete, cron_list, edit, glob, grep_search, loop_wakeup, monitor, read_file, read_mcp_resource, run_shell_command, web_fetch, write_file, zoom_image
  max_tokens     32000 asked
  request 4      26,820 characters, other tools (after the last reply): edit, glob, grep_search, read_file, write_file

━━ 5. read_file excluded too
  exit           0
  head -3 data.csv       refused: This "run_shell_command" invocation was denied by permission rules. Matching deny rule: "read_file".
  echo 42 > answer.txt   ran
  its log        result success · 3 turns · 1 refused by its permission rules
  answer.txt     '42'
```

What each run shows:

1. **It phones home by default.** With its own switches left on, Qwen Code
   asked twice for a connection to `gb4w8c3ygj-default-sea.rum.aliyuncs.com`
   on port 443. That is a host of Alibaba Cloud's real-user monitoring
   service (RUM), and the request is its usage statistics. In a normal run
   that connection is made; here the catcher refused it, and the sandbox
   would have refused it too.
2. **Four switches stop it.** `wire.py` sets `QWEN_USAGE_STATISTICS_ENABLED`
   and `QWEN_TELEMETRY_ENABLED` to `false`, and `QWEN_CODE_DISABLE_PRECONNECT`
   and `QWEN_DISABLE_AUTO_TITLE` to `1`. With them, the catcher saw nothing.
3. **`--bare` is what fits the window.** It reads no settings file and offers
   three tools; the first request is 25,387 characters. Because it reads no
   settings file, every request asks for 32,000 output tokens. The proxy
   (`--max-output-tokens 1024`, from `start.sh`) and `engine.py`
   (`cap = max(1, min(asked, MAX_COMPLETION_TOKENS))`, 1,024) cap every
   reply anyway. With the model, Qwen Code's longest reply was 1,006 tokens;
   mini-swe-agent's replies stopped at exactly 1,024 four times
   (`finish_reason` `length`, below).
4. **Without `--bare`** it offers 14 tools, its first request is 52,510
   characters, about twice as long, and it makes an extra request after the
   run's last reply, offering five other tools (a request of 26,820
   characters that no answer needs).
5. **Excluding a tool also forbids the shell to do the same thing.** With
   `read_file` among the excluded tools, `head -3 data.csv` through
   `run_shell_command` was refused: `Matching deny rule: "read_file"`. So
   `wire.py` never excludes `read_file` or `edit`, only the agent and planning
   tools, whose descriptions cost tokens in every prompt.

### One task, no model: `run_one.py`

Through the real capture proxy, with the scripted engine counting every
prompt in LFM2.5's tokens:

```text
run_one.py · qwen-code through the capture proxy at http://127.0.0.1:8239
  task      csv-s0-easy-01 (CSV family, seed 0): How many orders did Chloe Martin take?
  engine    scripted at http://127.0.0.1:8169 (lfm2.5 tokens) · script: solve
version   0.24.7, asked inside the sandbox · pinned 0.24.7 · ok
dialect   openai_chat (the proxy read it from the request)
first     call 1 read 5,678 prompt tokens (system prompt, tools, task)
calls     4 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
its log   result success · 4 turns
answer    answer.txt '18' · correct
network   lsof every 0.1 s: 21 samples over 2.9 s · connections to 127.0.0.1:8239 (the proxy) only: loopback, nothing else
```

```text
run_one.py · mini-swe-agent through the capture proxy at http://127.0.0.1:8239
first     call 1 read 537 prompt tokens (system prompt, tools, task)
stop      mini-swe-agent's finishing command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
answer    answer.txt '18' · correct
```

Qwen Code's first call is 5,678 tokens, against 537 for mini-swe-agent
(chapter 23 measured 2,279 for Claude Code with `--tools Bash` and 1,501
for Pi). That is a third of the 16,384-token window before the model has
written a word. Its prompt names its working folder, so the count moves a
little with the folder's name (chapter 23 found the same in Claude Code).

## The untrained model on the new family

`try_family.py` runs the family in one harness: mini-swe-agent, which
solved all 24 of chapter 24's SQL tasks. One new thing at a time: the new
family in a harness already measured, and below, the new harness on one
task of it.

```text
try_family.py · the untrained model on a new task family, in one harness
  tasks     20 of 20 CSV tasks (seed 0: ~/.cache/mhrl-course/29-make-it-yours/csv-tasks/seed-0.jsonl): 7 easy, 7 medium, 6 hard
  harness   mini-swe-agent 2.4.6
  protocol  one attempt per task · temperature 0 · 8 calls · replies capped at 1,024 tokens · 300 s per rollout
  records   ~/.cache/mhrl-course/29-make-it-yours/family/base/mini-swe-agent.jsonl (one line per rollout, as it ends)
  engine    model at http://127.0.0.1:8169 (lfm2.5 tokens) · script: none
version   2.4.6, asked inside the sandbox · pinned 2.4.6 · ok
tasks
[ 1/20] mini-swe-agent csv-s0-easy-01     correct · 5 calls  ·  15.3 s · finish command · answer '18'
[ 2/20] mini-swe-agent csv-s0-easy-02     wrong   · 7 calls  ·  35.6 s · finish command · answer '0'
[ 3/20] mini-swe-agent csv-s0-easy-03     correct · 6 calls  ·  19.3 s · finish command · answer '36'
[ 4/20] mini-swe-agent csv-s0-easy-04     correct · 7 calls  ·  20.6 s · finish command · answer '120.0'
[ 5/20] mini-swe-agent csv-s0-easy-05     correct · 8 calls  ·  20.1 s · step limit · answer '9'
[ 6/20] mini-swe-agent csv-s0-easy-06     correct · 6 calls  ·  10.4 s · finish command · answer '40'
[ 7/20] mini-swe-agent csv-s0-easy-07     correct · 7 calls  ·  14.8 s · finish command · answer '59'
[ 8/20] mini-swe-agent csv-s0-medium-01   correct · 7 calls  ·  21.4 s · finish command · answer 'Office Chair'
[ 9/20] mini-swe-agent csv-s0-medium-02   correct · 7 calls  ·  15.2 s · finish command · answer '12160.19'
[10/20] mini-swe-agent csv-s0-medium-03   correct · 6 calls  ·  18.5 s · finish command · answer '4.555555555555555'
[11/20] mini-swe-agent csv-s0-medium-04   correct · 7 calls  ·  17.8 s · finish command · answer 'Ben Ortiz'
[12/20] mini-swe-agent csv-s0-medium-05   wrong   · 7 calls  ·  20.7 s · finish command · answer 'Office Chair'
[13/20] mini-swe-agent csv-s0-medium-06   correct · 7 calls  ·  14.9 s · finish command · answer '5266.67'
[14/20] mini-swe-agent csv-s0-medium-07   correct · 7 calls  ·  23.5 s · step limit · answer '2.6666666666666665'
[15/20] mini-swe-agent csv-s0-hard-01     correct · 6 calls  ·  26.4 s · step limit · answer 'Ben Ortiz, Asha Rao, Chloe Martin'
[16/20] mini-swe-agent csv-s0-hard-02     correct · 7 calls  ·  37.8 s · step limit · answer 'Farid Khan, Chloe Martin, Ben Ortiz, Dev Patel'
[17/20] mini-swe-agent csv-s0-hard-03     correct · 7 calls  ·  17.9 s · step limit · answer '10'
[18/20] mini-swe-agent csv-s0-hard-04     correct · 8 calls  ·  32.1 s · step limit · answer '359.91'
[19/20] mini-swe-agent csv-s0-hard-05     correct · 7 calls  ·  18.8 s · finish command · answer 'Dev Patel, Ben Ortiz, Farid Khan, Asha Rao, Chloe Martin'
[20/20] mini-swe-agent csv-s0-hard-06     wrong   · 7 calls  ·  66.4 s · step limit · answer none

score     17/20 = 85.0% · Wilson 95% 64.0% to 94.8% · calls per solve 6.8
by level  easy 6/7 · medium 6/7 · hard 5/6
power     to see +10 points from 85.0% at p < 0.05, 80% of the time: about 156 paired tasks (exact McNemar, with independent attempts); the 85.0% is itself measured on 20 tasks
tasks     20 of 20 recorded · this run took 7 min 48 s
network   every rollout: lsof saw only this machine's addresses
```

- **17 of 20, and that is all 20 tasks can say.** The Wilson interval runs
  from 64.0% to 94.8%. One task is 5 points.
- **The lenient grader at work.** `4.555555555555555` passed for `4.56` and
  `2.6666666666666665` for `2.67` (within 0.01); three lists came back in
  another order and passed.
- **Seven tasks ended on mini-swe-agent's step limit, six of them already
  correct.** The model had written the right answer, and its steps ran out
  before the finishing command. The grade reads `answer.txt`, so they count.
- **The power line could not use chapter 24's churn here.** From 85%, 31.5%
  of tasks flipping would need more gains than the 15% still unsolved, so it
  fell back to independent attempts and says so.

The three wrong tasks, command by command, from mini-swe-agent's own
trajectories (`26-measure.txt`), are mistakes a SQL question never invited:

```text
━━ csv-s0-easy-02 · exit_status Submitted
  $ python3 -c "
    import csv
    with open('data.csv', 'r') as f:
        reader = csv.DictReader(f)
        regions = set(row['region'] for row in reader)
        print('Regions:', regions)
        west_count = sum(1 for row in reader if row['region'] == 'West')
        print('West count:', west_count)
    "
    > Regions: {'West', 'North', 'East', 'South'}
    > West count: 0
```

- **`csv-s0-easy-02`** (expected 13): a CSV reader can be read once. The
  first loop used up every row, so the count over the same reader was 0.
  The model printed the West rows one call later, then a reply ran into the
  1,024-token cap, and it wrote `0`.
- **`csv-s0-medium-05`** (expected `Photo Editor`): its revenue loop stored
  `revenue_by_product[product] = revenue`, each order replacing the last
  instead of adding to it, and never kept only the Software category. It
  answered `Office Chair`, a Hardware product.
- **`csv-s0-hard-06`** (expected three reps): the same used-up reader
  (`All reps: []`), then two replies cut at the 1,024-token cap, and the
  8 steps ran out with no answer.

Across the run's 154 replies, the longest was 1,024 tokens, and 4 were cut
at that cap (`finish_reason` `length`, in `csv-s0-easy-02`,
`csv-s0-hard-02` and twice in `csv-s0-hard-06`). The first prompt was 537
to 550 tokens. The engine's peak memory was 3.11 GB, the weights included
(chapter 24's SQL run: 2.90 to 3.24 GB).

### The fourth harness and the model

```text
run_one.py · qwen-code through the capture proxy at http://127.0.0.1:8239
  task      csv-s0-easy-01 (CSV family, seed 0): How many orders did Chloe Martin take?
  engine    model at http://127.0.0.1:8169 (lfm2.5 tokens) · script: none
version   0.24.7, asked inside the sandbox · pinned 0.24.7 · ok
dialect   openai_chat (the proxy read it from the request)
first     call 1 read 5,678 prompt tokens (system prompt, tools, task)
calls     6 agent calls · 0 side calls · 0 answered by the proxy itself
stop      a reply with no tool call: the harness's loop ends there
its log   result success · 6 turns · 1 refused by its permission rules
  told    "write_file" is not listed in the active core tools allowlist (--core-tools or settings tools.core), so the tool is not available. Add it to the core tools list to re-enable it.
answer    answer.txt '18' · correct
exit      0 after 36.5 s
network   lsof every 0.1 s: 259 samples over 36.5 s · connections to 127.0.0.1:8239 (the proxy) only: loopback, nothing else
```

Call by call (`26-measure.txt`): it read `data.csv` whole with `read_file`
(the prompt grew from 5,678 to 8,374 tokens; a 6 KB text file fits, where
chapter 23's Pi overflowed the window on a binary database), counted with
`grep -c "Chloe Martin"` (18), tried a `write_file` tool that `--bare` does
not offer, wrote the answer with `echo` through its shell instead, read it
back, and stopped. Correct in 6 calls; the last prompt was 10,155 tokens.
The `lsof` watch took 259 samples and saw only the proxy. One task is a
demonstration, not a measurement.

## How many tasks: `power.py`

```text
power.py · how many paired tasks before a gain can be told from noise
  test      exact McNemar, two-sided: a change counts when p < 0.05 (chapter 24's exact_p)
  power     80.0%: the chance a real gain of that size shows up as p < 0.05

━━ reading one paired comparison (chapter 24: claude-code, base -> sft)
  24 tasks: 15 solved before, 17 after; 7 gained, 5 lost, 12 flipped, 12 unchanged
  only the 12 flipped tasks say which checkpoint is better; if the two are equally good,
  each of them is a coin flip between gained and lost
  exact p = 0.774: a split at least as uneven as 7 to 5 comes up 77% of the time by luck alone

━━ how often two checkpoints disagree: chapter 24's 9 pairs
  pair         harness         cells  gained  lost  flipped  p
  base -> sft  mini-swe-agent     24       0     6        6  0.031
  base -> sft  claude-code        24       7     5       12  0.774
  base -> sft  pi                 24       7     2        9  0.180
  base -> rl   mini-swe-agent     24       0     4        4  0.125
  base -> rl   claude-code        24       7     4       11  0.549
  base -> rl   pi                 24       8     3       11  0.227
  sft -> rl    mini-swe-agent     24       2     0        2  0.500
  sft -> rl    claude-code        24       2     1        3  1.000
  sft -> rl    pi                 24       5     5       10  1.000
  pooled: 68 of 216 cells flipped = 31.5% of tasks

━━ +10 points from 60.0% (to 70.0%), p < 0.05, power 80.0%
  assumption                      flipped  gained   lost  tasks needed                  normal approx.
  only gains                        10.0%   10.0%   0.0%  78 (exact, power 80.4%)       77
  chapter 24's churn                31.5%   20.7%  10.7%  261 (exact, power 80.0%)      245
  independent attempts              46.0%   28.0%  18.0%  377 (exact, power 80.1%)      359

━━ check: 10,000 simulated comparisons at each answer (seed 0)
  assumption              tasks  p < 0.05    exact
  only gains                 78     80.4%    80.4%
  chapter 24's churn        261     80.3%    80.0%
  independent attempts      377     80.1%    80.1%

━━ +10 points from other baselines, tasks needed (power 80.0%)
  baseline  only gains  chapter 24's churn  independent attempts
     20.0%          78                 261                   313
     40.0%          78                 261                   408
     60.0%          78                 261                   377
     80.0%          78         cannot hold                   215
  only independent attempts depend on the baseline; 'cannot hold': that much flipping
  would need more gains than there are unsolved tasks

━━ what 24 tasks per harness can see (chapters 24 to 28)
  a real +10 points shows up as p < 0.05 in: 2.8% (only gains); 6.5% (chapter 24's churn); 6.1% (independent attempts)
  chapter 24's changes beyond noise: 1 of 9 pairs
    mini-swe-agent base -> sft: 0 gained, 6 lost (p = 0.031), -25 points

━━ the plan said '+10 points at p < 0.05 needs about 150 paired tasks'
  measured, with chapter 24's churn: 261 (exact, power 80.0%); 150 tasks give power 54.4%
  150 tasks reach power 80.0% only while at most 17.5% of tasks flip
```

- **Reading a comparison first.** Chapter 24's Claude Code went from 15 to
  17 of 24 after SFT (62% to 71%). Only the 12 tasks that flipped carry
  information, and a 7-to-5 split of 12 coin tosses comes up 77% of the
  time by luck alone.
- **The answer depends on how many tasks flip, not only on the gain.** The
  same +10 points needs 78 paired tasks if the new checkpoint only gains,
  261 if tasks flip as often as they did between the project's checkpoints,
  and 377 if each checkpoint answers each task like a fresh coin. The exact
  test is the count; the normal approximation is close (77, 245, 359), and
  10,000 simulated comparisons agree with the exact power to 0.3 points.
- **What 24 tasks could see.** A real +10 points shows up in 2.8% to 6.5%
  of 24-task comparisons. That is why the one change beyond noise in
  chapter 24 was 25 points (6 tasks lost, none gained), and why chapters 24
  to 28 read their gains as within noise.
- **How it works.** For each number of flipped tasks, `rejection_limits`
  finds the most lopsided splits the exact test rejects, in whole numbers,
  so it agrees with `stats.exact_p` exactly (the tests check every split up
  to 200 flips and the edge at 2,000). The number of flips in n tasks is
  binomial, so the power is a sum over them. The smallest n that reaches
  the target is found by trying n = 1, 2, 3, ... up to 2,000; past that the
  normal approximation is used and labelled.

## Plan versus measured

| the plan, or the review, said | measured | source |
| --- | --- | --- |
| a template task family (CSV questions over a small spreadsheet) with its generator and grader | `data/sales.csv` (96 invented orders), 12 templates, 20 tasks for seed 0 with a fixed md5; chapter 1's grader byte for byte | `01-make-tasks`, the tests |
| 'to see +10 points at p<0.05 you need about 150 paired tasks' | **261** at 80% power with chapter 24's churn (31.5% of tasks flipping); 78 if the new checkpoint only gains; 377 if attempts are independent (from 60%). 150 tasks give 54.4% power, and reach 80% only while at most 17.5% of tasks flip | `02-power` |
| (what 24 tasks per harness can show) | a real +10 points shows up as p < 0.05 in 2.8% to 6.5% of 24-task comparisons | `02-power` |
| a fourth harness wired only through its base URL and key | the connection is three variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`). It also needed `--bare`, 14 tools excluded and 4 switches off, measured below | `04-wire`, `05-qwen-probe` |
| (not in the plan) | with its own switches on, Qwen Code asked twice for `gb4w8c3ygj-default-sea.rum.aliyuncs.com:443`, Alibaba Cloud's usage monitoring; with `QWEN_USAGE_STATISTICS_ENABLED`, `QWEN_TELEMETRY_ENABLED`, `QWEN_CODE_DISABLE_PRECONNECT` and `QWEN_DISABLE_AUTO_TITLE` set, it asked for none | `05-qwen-probe` (runs 1 and 2) |
| (not in the plan) | the model run's `lsof` watch: 259 samples over 36.5 s, connections to the proxy only; mini-swe-agent's 20 model rollouts, the same | `23-one-qwen-model`, `22-family-model` |
| (not in the plan) | excluding `read_file` made Qwen Code refuse `head -3 data.csv` through its shell: `Matching deny rule: "read_file"`; so `read_file` and `edit` stay offered | `05-qwen-probe` (run 5) |
| (not in the plan) | without `--bare`: 14 tools, a first request of 52,510 characters, and a fourth request after the last reply; with it: 3 tools and 25,387 characters | `05-qwen-probe` (runs 3 and 4) |
| (not in the plan) | with `--bare` it reads no settings file and asks for 32,000 output tokens; the proxy and `engine.py` cap every reply at 1,024. Its longest reply with the model was 1,006 tokens; mini-swe-agent hit 1,024 four times | `05-qwen-probe` (run 3), `26-measure` |
| (not in the plan) | Qwen Code's first call: 5,678 LFM2.5 tokens, against 537 for mini-swe-agent | `08-one-qwen`, `09-one-mini` |
| a new task family needs only data, a generator and a grader | it also needed mini-swe-agent's instance template rewritten (it named `sqlite3 data.db`), and the sandbox pointed at the new folder of gold answers (with chapter 24's default, `cat` read them inside the box) | `mini_swe_agent.yaml`, the tests |
| generates 20 CSV tasks, runs the untrained model on them through the proxy in one harness, and prints the per-task results plus power.py's answer | 17 of 20 in mini-swe-agent (Wilson 95% 64.0% to 94.8%), 6.8 calls per solve, 7 min 48 s; per-task lines, the score and power.py's line (156 tasks from 85%, with independent attempts) | `22-family-model` |
| (review) using the pipeline yourself: a new family, another harness, what to re-measure | the family, Qwen Code, and the list below | this chapter |
| (review) reading a paired checkpoint comparison: 62 -> 71 is 7 gained, 5 lost, p = 0.77 | 7 gained, 5 lost, exact p = 0.774; `try_family.py --against` reads two of your own runs the same way | `02-power`, the tests |
| (review) a 60-second primer on vectors, matrices and rank | not in this chapter: chapter 10 defines vector and matrix, chapter 13 defines rank; the primer belongs there | `10-meet-lfm`, `13-lora` |
| folder `course/29-make-it-yours/` (`make_it_yours/` in the code line) | `29-make-it-yours/` at the root of this repository, the site's slug | |

- **The project's numbers** in this chapter are chapter 24's nine checkpoint
  pairs, from the project's evaluation records of 2026-10-06
  (`data/chapter24_pairs.json`; the tests recompute them from chapter 24's
  bundled records).
- **The article's numbers**: there are none in this chapter.

## Make it yours: what to change, what to measure again

- **A new family.** Bring the data, write the templates and their gold
  queries, keep chapter 1's grader unless your answers have a new shape.
  Read every harness setting that describes the job (here, only
  mini-swe-agent's instance template). Point the sandbox at the folder of
  gold answers, and check inside the box that it cannot be read.
- **A new harness.** Pin its version. Wire it by the settings its own
  documentation gives for any provider. Then measure, before the model:
  where it connects besides the model (`qwen_probe.py`'s catcher), what its
  first call costs against the 16,384-token window (`run_one.py`'s `first`
  line), which tools it offers, and what its exclusions forbid.
- **Measure again, one new thing at a time.** The untrained model on the
  new family, in a harness already measured. The new harness on a family
  already measured. Then `power.py` before you train: the number of tasks
  your hoped-for gain needs, so a result is not read from too few.

## The files

Lines, and lines of code (no blank lines, comments or docstrings), counted
with Python's own tokenizer:

| file | lines | code | what it is |
| --- | ---: | ---: | --- |
| `power.py` | 868 | 428 | paired tasks for a gain: exact McNemar, the normal approximation past 2,000, a seeded check |
| `try_family.py` | 555 | 341 | the family in one harness through the proxy: per-task results, Wilson, power, `--against` |
| `make_csv_tasks.py` | 553 | 326 | chapter 1's seeded generator over the spreadsheet: 12 templates, the fairness rules |
| `qwen_probe.py` | 548 | 315 | Qwen Code five ways against a stand-in model, outside connections caught |
| `sheet.py`, `tasks.py` | 130, 202 | 60, 80 | the spreadsheet read with types; the family's tasks, instruction and gold folder |
| `run_one.py`, `wire.py`, `install.py` | 950, 691, 678 | 536, 406, 356 | chapter 24's, with Qwen Code added and the CSV solving script |
| `mini_swe_agent.yaml` | 136 | | the project's mini-swe-agent config, its SQLite wording rewritten for a CSV file |
| `data/sales.csv`, `data/chapter24_pairs.json` | 97, 14 | | the invented spreadsheet; chapter 24's nine checkpoint pairs |
| `evaluate.py`, `stats.py`, `start.sh`, `engine.py`, `scripted_engine.py`, `capture_client.py`, `sandbox.py`, ... | | | chapter 24's, unchanged but for the folder name (`start.sh` adds `UV_NO_CONFIG=1`); `grader.py` is chapter 1's |

## What this chapter leaves out

- **Training on the new family.** Chapters 25 and 26 trained on the SQL
  family. This chapter measures the new family's starting point and how
  many tasks a gain would need before training on it is worth reading.
- **A score for Qwen Code.** One task shows it works with the model; a
  score needs the family's 20 tasks or more, and `power.py` says how many
  more to compare it with another harness.
- **The vectors, matrices and rank primer**, which belongs beside chapter
  10's embeddings and chapter 13's rank-16 matrices.

## Mirrors in the project

`mhrl/tasks/` (generate.py, core.py) for the family, `mhrl/harnesses/`
for wiring a harness, and `mhrl/evaluate.py` for the protocol. The project
has no Qwen Code harness and no power calculation: those are this chapter's.
