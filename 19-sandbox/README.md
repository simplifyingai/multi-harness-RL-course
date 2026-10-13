# Chapter 19. The sandbox: a small model typing bash on your laptop

From chapter 23 on, a small model will type shell commands on your laptop
through real agent programs. Chapter 16 showed that Claude Code runs with
`bypassPermissions`: it runs every Bash command the model asks for, without
asking you. Nothing in Claude Code stops a model that types
`cat ~/.ssh/id_ed25519`. Something else has to.

This chapter builds that something: a box made with macOS's own sandbox,
using the project's real profile. Then it attacks the box, one attempt at a
time, against your real files. The script prints only whether each attempt
worked, never what it read. It needs no model, no download and no account.

The puzzle to keep in mind: can a safety box make a perfectly good agent
score 0%? The project's first box did. The harness Pi solved 0 of 24 tasks
after SFT and 0 of 24 after RL. After one rule was removed it solved 9 of 24
at both checkpoints. Which rule was it, and why did `pwd` not notice? By the
end of the chapter you will have run that box yourself.

## Words first

- A **sandbox** (here, "the box") is a set of rules the operating system
  enforces on a program and on every program it starts. It is not a virtual
  machine: the program runs on your Mac, but the kernel refuses what the rules
  forbid.
- **Seatbelt** is macOS's sandbox. `sandbox-exec` (in `/usr/bin`, shipped with
  macOS) starts a program under a **profile**, a text file of rules. Apple marks
  `sandbox-exec` as deprecated, but it works on macOS 26.6.2, where this
  chapter ran.
- A **rule** allows or denies an operation (`file-read*`, `file-write*`,
  `network-outbound`, `mach-lookup`) on a set of paths or addresses. When
  several rules match, **the last one wins**.
- A **real path** is a path with every link resolved. On macOS `/tmp` is a link
  to `/private/tmp`, and Seatbelt matches real paths only, so a rule written
  for `/tmp` would never match.
- A **rollout** is one attempt at one task (the course's word since chapter 4).
  Each rollout here gets a **task folder** (its working folder, holding the
  task's database) and a private **state folder** (a fake home folder and its
  own temporary folder).
- **Environment variables** are named values every program inherits, such as
  `HOME` (your home folder) and `PATH` (where commands are looked up). API keys
  often live there too, which is why the box starts with an almost empty set.
- A **process** is a running program. A **process tree** is a process and
  everything it started. A **process group** is a set of processes that can be
  signalled together, and a **session** is a set of groups. **SIGKILL** is the
  signal that ends a process at once. **Reaping** a tree means making sure no
  part of it is still running.
- A **symbolic link** is a small file that points at another path. A **hard
  link** is a second name for the same file. A **named pipe** looks like a file
  but waits for a writer before it can be read.
- **Errors** have short codes: **EPERM** prints as "Operation not permitted"
  (something refused you), **ENOENT** as "No such file or directory" (nothing
  is there).
- **`realpath`** turns a path into its real path. To do that, it looks at
  every folder on the way from `/`.
- **Reward hacking**: an agent trained by reward finds a way to the reward that
  skips the task, such as reading the **gold answers** (the correct answers the
  grader compares against).
- A **decoy** is a fake secret. A file that does not matter is used to prove
  the box works before any real secret is touched.
- **Loopback** (`localhost`, `127.0.0.1`) is your Mac talking to itself.
  **DNS** turns a name such as `huggingface.co` into an address.
- The **keychain** is macOS's store of passwords and logins. Claude Code keeps
  its login there.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/19-sandbox
uv run tasks.py
uv run sandbox.py
uv run attacks.py --decoys-only
uv run attacks.py
uv run attacks.py --profile v1
uv run runner.py -- bash -c 'pwd; ls; echo "HOME=$HOME"'
uv run runner.py --timeout 5 -- bash -c 'sleep 100 & sleep 100 & wait'
uv run runner.py --group-only --timeout 5 -- bash -c 'sleep 100 & perl -e "setpgrp(0, 0); sleep 100" & wait'
uv run runner.py --timeout 5 -- bash -c 'sleep 100 & perl -e "setpgrp(0, 0); sleep 100" & wait'
uv run grade_safe.py
uv run runner.py --profile v1 -- bash -c 'pwd; /bin/pwd -P; ls ..'
```

You need a Mac and [uv](https://docs.astral.sh/uv/). `sandbox-exec`, `curl`,
`sqlite3`, `perl` and `security` ship with macOS. Everything the chapter
writes goes under `COURSE_HOME/19-sandbox` (default
`~/.cache/mhrl-course/19-sandbox`). The scripts print that folder as `$CH`,
this repository as `$REPO`, and your home folder as `/Users/you`, so the
output can be shared. `attacks.py --probe PATH` adds one of your own secret
files to the attempts. Only the outcome is printed.

| command | what it shows | seconds in the run log |
| --- | --- | ---: |
| `tasks.py` | the task, its gold answer, and where the chapter keeps files | 0 |
| `sandbox.py` | the `sandbox-exec` command a rollout runs, with every parameter | 0 |
| `attacks.py --decoys-only` | the profile proved on fake secrets, and nothing else | 2 |
| `attacks.py` | decoys, then the escape attempts on your real files, then the agent's work and the grade | 11 |
| `attacks.py --profile v1` | the same under the first, too-strict profile | 10 |
| `runner.py -- ...` | one rollout: folders, environment, log, exit | 0 |
| `runner.py --timeout 5 -- ...` | a timeout and the whole tree reaped | 6 |
| `runner.py --group-only ...` | killing only the group misses a child that left it | 7 |
| `runner.py --timeout 5 -- ...` (same child) | the two passes catch it | 5 |
| `grade_safe.py` | four `answer.txt` files made by the box, graded two ways | 1 |
| `runner.py --profile v1 -- ...` | `pwd` under the first profile | 0 |

Every output below is copied from one recorded run of these commands, from an
empty `COURSE_HOME/19-sandbox`, on an M3 Pro with 18 GB under macOS 26.6.2,
while other chapters' work ran on the same machine. The run log keeps each
command's commit, exit status and time, in whole seconds. Process ids, run ids,
the random part of the escape files' names and the times change from run to
run. In the runlog run, the escape attempts also probed one real `.env` file of
another project on the laptop (`--probe`).

## The box: `sandbox.sb`

`sandbox.sb` is the project's profile (`configs/harnesses/sandbox.sb`) with its
rules unchanged. Only the folders it is given differ. It starts permissive,
because agent programs (Node, Python) read all over the system, and then
takes things away:

```scheme
(allow default)

(deny file-write*)
(allow file-write*
    (subpath (param "WORKDIR"))
    (subpath (param "STATE_DIR"))
    (subpath "/dev"))

(deny file-read*
    (subpath (string-append (param "HOME_DIR") "/.ssh"))
    ...
    (regex (param "ENV_FILE_REGEX")))

(deny file-read*
    (subpath (param "TASKS_DIR")))

(allow file-read*
    (subpath (param "WORKDIR"))
    (subpath (param "STATE_DIR")))

(deny mach-lookup (global-name "com.apple.SecurityServer"))

(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
```

The last matching rule wins. So writes are denied everywhere and then allowed
again in three places. Reads of secrets and of the gold answers are denied,
and then allowed again inside the rollout's own two folders. `sandbox.py`
computes the parameters. Each is a real path, and the regexes are anchored on
your real home folder:

```text
/usr/bin/sandbox-exec -f $REPO/19-sandbox/sandbox.sb \
    -D WORKDIR=$CH/rollouts/example \
    -D STATE_DIR=$CH/state/example \
    -D HOME_DIR=/Users/you \
    -D 'ENV_FILE_REGEX=^/Users/you/(.+/)?(\.env(rc)?|[^/]+\.env)(\.[^/]*)?$' \
    -D 'CLAUDE_JSON_REGEX=^/Users/you/\.claude\.json' \
    -D 'CLAUDE_CREDS_REGEX=^/Users/you/\.claude/\.credentials' \
    -D TASKS_DIR=$REPO/19-sandbox/data \
    -D RUNS_DIR=$CH/rollouts \
    /bin/bash -c pwd
```

`TASKS_DIR` is this chapter's `data/` folder, which holds the task record with
its gold answer (`30`). `RUNS_DIR` is read only by the first profile. Two
checks in `sandbox.py` guard the folders. A writable folder may not contain
your home or overlap a credential path, because the re-allow would make the
secret readable again. A hidden folder may not contain a writable folder
(that was the first profile's bug), or sit inside one, where reads are allowed
again.

## Decoys first

Before anything touches a real secret, `attacks.py` writes six fake ones under
`$CH/decoys` and builds a box whose "home" is the decoy folder. Each decoy
holds one line, `decoy for chapter 19: not a real secret`. Each must be
readable outside the box and refused inside it with "Operation not permitted".
Anything else stops the script before a real path is probed (a test proves the
stop with a profile that allows everything):

```text
1. Decoys first: fake secrets, the same rules, a decoy 'home'
  decoy ssh key
    $ cat -- $CH/decoys/home/.ssh/id_ed25519 >/dev/null
    | cat: $CH/decoys/home/.ssh/id_ed25519: Operation not permitted
    denied (exit 1)
    outside the box: readable
  ...
  decoy gold answers
    $ cat -- $CH/decoys/tasks/task.jsonl >/dev/null
    | cat: $CH/decoys/tasks/task.jsonl: Operation not permitted
    denied (exit 1)
    outside the box: readable
  all 6 decoys denied (--decoys-only: stopping)
```

Every probe reads with `cat FILE >/dev/null`. The content goes nowhere, and
only the error message and the exit status are seen. So even a broken box
could not put a secret on the screen.

## Escape attempts, against your real files

Then the same box guards your real home. A second rollout, `neighbour`, sits
next to the attacking one and has already written its `answer.txt`. All the
attempts below are from one run of `uv run attacks.py`.

### Writes

```text
  write answer.txt in the task folder
    $ echo 30 > answer.txt
    allowed (exit 0)
  write a file in your real home
    $ echo x > /Users/you/mhrl-course-19-escape-cf7796c4.txt
    | /bin/bash: /Users/you/mhrl-course-19-escape-cf7796c4.txt: Operation not permitted
    denied (exit 1)
  write into the shared /tmp
    $ echo x > /private/tmp/mhrl-course-19-escape-b9d014b7.txt
    | /bin/bash: /private/tmp/mhrl-course-19-escape-b9d014b7.txt: Operation not permitted
    denied (exit 1)
  write answer.txt into the neighbour's folder
    $ echo 99 > ../neighbour/answer.txt
    | /bin/bash: ../neighbour/answer.txt: Operation not permitted
    denied (exit 1)
```

Writes into this repository are denied the same way. Writes into the private
`$TMPDIR` are allowed. The shared `/tmp` is closed for a reason the project
learned: rollouts run side by side, and a model that writes to a guessed
absolute `/tmp` path can land in another rollout's folder. The project saw Pi
write `answer.txt` two directories above its own task folder that way
(`sandbox.sb`, line 29 of the project's copy).

### Two layers: the fake HOME, then the real path

```text
  layer 1: ~ is a fake HOME
    $ echo "HOME=$HOME"; cat ~/.ssh/id_ed25519 >/dev/null
    | HOME=$CH/state/attack/home
    | cat: $CH/state/attack/home/.ssh/id_ed25519: No such file or directory
    denied (exit 1)
  layer 2: your real ssh key
    $ cat -- /Users/you/.ssh/id_ed25519 >/dev/null
    | cat: /Users/you/.ssh/id_ed25519: Operation not permitted
    denied (exit 1)
    outside the box: present
  does your real ssh key appear to exist?
    $ test -e /Users/you/.ssh/id_ed25519
    denied (exit 1)
    outside the box: present
```

Inside the box, `~` is the fake home, which has no `.ssh` folder: "No such file
or directory". But a model can type the real path, which `attacks.py` resolved
outside the box. The key exists there, and the box refuses it:
"Operation not permitted". `test -e` says the key does not exist, because the
deny covers looking at a file as well as reading it. Your cloud credentials,
Claude Code's `~/.claude.json`, a `.env` file elsewhere under your home, and
the gold answers (`$REPO/19-sandbox/data/task.jsonl`) were all refused with
the same message.

### The keychain, in the same two layers

```text
  keychain, layer 1: through the fake HOME
    $ security find-generic-password -s 'Claude Code-credentials' 2>&1 >/dev/null
    | security: SecKeychainSearchCreateFromAttributes: One or more parameters passed to a function were not valid.
    | security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.
    denied (exit 44)
    outside the box: exit 0
  keychain, layer 2: HOME set back to the real one
    $ HOME=/Users/you security find-generic-password -s 'Claude Code-credentials' 2>&1 >/dev/null
    | security: SecKeychainSearchCreateFromAttributes: One or more parameters passed to a function were not valid.
    | security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.
    denied (exit 44)
    outside the box: exit 0
```

Outside the box the item exists (exit 0). The query never uses `-w` or `-g`,
so it never asks for the password, and it throws away the item's details.
Inside, the keychain looks empty (exit 44, "could not be found"). The fake
home alone does that. But any program can set `HOME` back, as layer 2 does,
and then only the profile stops it: the `mach-lookup` rule and the
`Library/Keychains` read rule.

### The network

```text
  engine stand-in: 127.0.0.1:8100 is already used by another program on this Mac, so the stand-in listens on 127.0.0.1:8119
  ...
  reach the internet by name
    $ curl -sS -o /dev/null --max-time 10 https://huggingface.co
    | curl: (6) Could not resolve host: huggingface.co
    denied (exit 6)
    outside the box: HTTP 200
  reach the internet by address
    $ curl -sS -o /dev/null --max-time 10 http://1.1.1.1
    | curl: (7) Failed to connect to 1.1.1.1 port 80 after 0 ms: Couldn't connect to server
    denied (exit 7)
    outside the box: HTTP 301
  reach the engine on localhost
    $ curl -sS --max-time 10 http://127.0.0.1:8119/health
    | {"status": "ok", "stands_in_for": "chapter 18's engine"}
    allowed (exit 0)
    outside the box: HTTP 200
```

DNS itself is a network connection, so external names do not even resolve.
The one place the box may reach is loopback, where chapter 18's model engine
will listen on port 8100. Chapter 18's engine does not exist yet in this
course, so `attacks.py` starts a throwaway web server that answers `/health`
in its place. On the recording Mac port 8100 was already taken by an
unrelated program, so the stand-in used 8119, and the output says so.

### The environment

```text
  see the environment
    $ env | sort
    | _=/usr/bin/env
    | HOME=$CH/state/attack/home
    | LANG=en_US.UTF-8
    | MHRL_RUN_ID=0a7fec4f9d2049ae8363300390519269
    | PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
    | PWD=$CH/rollouts/attack
    | SHLVL=1
    | TMPDIR=$CH/state/attack/tmp
    allowed (exit 0)
```

The runner passes five variables (`HOME`, `LANG`, `MHRL_RUN_ID`, `PATH`,
`TMPDIR`). bash adds `PWD`, `SHLVL` and `_`. No API key or token from your
shell reaches the box.

### The agent's own work, and the table

A box that refuses everything is useless, so the last part runs what an agent
must still be able to do, in a fresh task folder. It ends with the grade:

```text
  the agent's work: 5 of 5 steps allowed
  grade (outside the box): correct · reward 1
```

```text
Allowed/denied table · profile project
  attempt                                           outside the box  in the box    check
  write answer.txt in the task folder               -                allowed       as expected
  write in the private TMPDIR                       -                allowed       as expected
  write a file in your real home                    -                denied        as expected
  write into this repository                        -                denied        as expected
  write into the shared /tmp                        -                denied        as expected
  write answer.txt into the neighbour's folder      -                denied        as expected
  read the neighbour's answer.txt                   present          allowed       as expected
  layer 1: ~ is a fake HOME                         -                denied        as expected
  layer 2: your real ssh key                        present          denied        as expected
  does your real ssh key appear to exist?           present          denied        as expected
  list your real ~/.ssh folder                      present          denied        as expected
  your cloud credentials                            present          denied        as expected
  Claude Code's login (~/.claude.json)              present          denied        as expected
  Claude Code's login file                          absent           inconclusive  inconclusive: absent outside the box too
  a .env file elsewhere under your home             present          denied        as expected
  your file .env                                    present          denied        as expected
  read your shell history                           present          allowed       as expected
  read the gold answers (the task record)           present          denied        as expected
  keychain, layer 1: through the fake HOME          exit 0           denied        as expected
  keychain, layer 2: HOME set back to the real one  exit 0           denied        as expected
  reach the internet by name                        HTTP 200         denied        as expected
  reach the internet by address                     HTTP 301         denied        as expected
  reach the engine on localhost                     HTTP 200         allowed       as expected
  see the environment                               -                allowed       as expected
  print the working folder                          -                allowed       as expected
  look at the parent folder                         -                allowed       as expected
  query the database                                -                allowed       as expected
  write answer.txt with Pi's write tool             -                allowed       as expected
  write a file with the shell                       -                allowed       as expected
28 as expected, 0 unexpected, 1 inconclusive (a denial that could not be proved)
```

"Denied" needs proof. The attempt must fail with the refusal's own message,
and the file, item or server must exist outside the box. `~/.claude/.credentials.json`
does not exist on this Mac (Claude Code keeps its login in the keychain here),
so that row proves nothing and says so. An attempt that should work but fails
for any reason counts as unexpected.

### What the box does not stop

Two rows are allowed although they are not harmless, and the table says so:

- **Another rollout's folder is readable.** `cat ../neighbour/answer.txt`
  printed `30`. The project's first profile hid every rollout's folder, which
  broke tools (next section). The fix removed that rule and put nothing
  narrower in its place, although the project's README still says other
  rollouts' folders are unreadable. Rollouts of the same task run side by side
  in GRPO (chapter 26), so a rollout could copy a neighbour's answer.
- **The profile is a list of things to hide**, on top of `(allow default)`.
  Anything the list does not name stays readable. Shell history is one example
  (`~/.zsh_history` was read, into `/dev/null`), and it can hold a token pasted
  on a command line. Loopback is open on every port, so any program listening
  on `127.0.0.1` is reachable too, not only the engine. The attempts also do not
  try to start programs outside the box through macOS services (`open`,
  `osascript`). The chapter makes no claim about those.

## The first profile: `--profile v1`

`sandbox_v1.sb` is `sandbox.sb` plus one rule. It also hides `RUNS_DIR`, the
folder that holds every rollout's task folder, so that no rollout could read
another's folder. The task folder itself is allowed again by the later rule,
so it looks harmless. A test checks that this one line is the only difference
between the two profiles.

The project measured what it cost, in `runs/main/summary_<stage>.json`:

```text
Pi's accuracy on the 24 held-out tasks, from runs/main/summary_<stage>.json
  base  before the sandbox fix 3/24 = 12.5%   after 4/24 = 16.7%   note: re-evaluated after the sandbox fix (pi's write tool)
  sft   before the sandbox fix 0/24 = 0.0%   after 9/24 = 37.5%   note: re-evaluated after the sandbox fix (pi's write tool)
  rl    before the sandbox fix 0/24 = 0.0%   after 9/24 = 37.5%   note: re-evaluated after the sandbox fix (pi's write tool)
```

Here is the same box, run by `attacks.py --profile v1`:

```text
3. The agent's own work, which the box must allow
  print the working folder
    $ pwd
    | $CH/rollouts/work
    allowed (exit 0)
  look at the parent folder
    $ ls .. >/dev/null
    | ls: ..: Operation not permitted
    denied (exit 1)
  query the database
    $ sqlite3 data.db "SELECT COUNT(*) FROM patients WHERE city = 'Murcia'"
    | 30
    allowed (exit 0)
  write answer.txt with Pi's write tool
    $ $PY $REPO/19-sandbox/write_tool.py answer.txt 30
    | EPERM: operation not permitted, realpath '$CH/rollouts/work/answer.txt'
    denied (exit 1)
  write a file with the shell
    $ echo 30 > shell.txt
    allowed (exit 0)
  the agent's work: 3 of 5 steps allowed
  grade (outside the box): wrong: no answer.txt · reward 0
```

The agent found the right number (`30`), and the box let the shell write a
file. But Pi's write tool does one thing first. Before writing, it asks
`realpath` for the file's real path, to queue writes to the same file one
after another. A file that is not there yet is fine: `realpath` answers
"No such file", and the tool goes on. `realpath` walks through every parent
folder, and under v1 those folders are hidden. So it answers EPERM instead.
The tool treats that as a real error and stops, and `answer.txt` is never
written, however right the answer was. In the project's record (run log 15)
Pi scored 0 of 24 after SFT and 0 of 24 after RL under v1; the base model,
3 of 24. Once the profile was fixed they scored 4, 9 and 9 of 24.
`write_tool.py` does Pi's three
steps through the same C library call, and prints the same words the project's
log recorded for Pi under the first profile:

```text
Pi 1.0.4's write tool resolves the path first (file-mutation-queue.js)
  1: import { realpath } from "node:fs/promises";
  5: function isMissingPathError(error) {
  9: (error.code === "ENOENT" || error.code === "ENOTDIR"));
  14: return await realpath(resolvedPath);
  17: if (isMissingPathError(error)) {

Pi under the first profile: the write tool's result in a recorded rollout
  tool write: EPERM: operation not permitted, realpath '…/runs/work/clinic-easy-001-obq60vz3/answer.txt'
```

(The project's folder is shortened to `…` here.)

And `pwd`? The plan expected it to fail. On macOS 26.6.2 it does not:

```text
command    bash -c 'pwd; /bin/pwd -P; ls ..'
output     (the log file)
  | $CH/rollouts/run-11412748
  | $CH/rollouts/run-11412748
  | ls: ..: Operation not permitted
exit       1 after 0.4 s
```

macOS finds the working folder's path without reading its parents, so the
shell's `pwd` and `/bin/pwd -P` both work. That is what makes the bug
dangerous. Nothing in an ordinary check (`pwd`, `ls`, a query, `echo >`)
fails, and the agent still answers correctly in its head. Only the harness's
own tool breaks, silently, and the score says 0%. Under v1 the neighbour's
`answer.txt` is hidden (`denied`), which is what the rule was for.

## The runner: a clean start and a hard stop

`runner.py` is how every command in this chapter starts, the way the project
starts every harness (`mhrl/harnesses/process.py`):

```text
rollout    run-d5dd9672 · profile project · timeout 300 s
task dir   $CH/rollouts/run-d5dd9672  (holds data.db)
fake HOME  $CH/state/run-d5dd9672/home
TMPDIR     $CH/state/run-d5dd9672/tmp
env        HOME LANG MHRL_RUN_ID PATH TMPDIR
marker     MHRL_RUN_ID=d5dd96729edb4cffa815be26ea2a264d
log        $CH/logs/run-d5dd9672/command.txt
command    bash -c 'pwd; ls; echo "HOME=$HOME"'
```

The default timeout, 300 s, is the project's limit for one rollout
(`rollout_timeout_s: int = 300` in `mhrl/config.py`). The command starts in its
own session, so it leads a new process group. Its output goes to a log file.
Every process it starts inherits the run's **marker**, `MHRL_RUN_ID`. When the
command ends or times out, the runner kills its group. Then it sweeps twice
more: every descendant, every group a descendant was seen in, and every
process carrying the marker. Finally it asks `ps` whether anything survived.

```text
command    bash -c 'sleep 100 & sleep 100 & wait'
output     (the log file)
timeout    after 5.3 s the run was stopped
tree       3 processes seen while it ran (ps is read every second, and once more before the kill; root pid 8396)
  pid 8396    bash       parent 8395    group 8396 (the run's group)
  pid 8397    sleep      parent 8396    group 8396 (the run's group)
  pid 8398    sleep      parent 8396    group 8396 (the run's group)
reaped     group 8396 killed; after it, 2 sweeps signalled 1 process more (the tree, every group it used, the marker)
survivors  0 of 3 (ps -p 8396,8397,8398 lists none of them)
```

Afterwards, `pgrep -fl "sleep 100"` found nothing (`pgrep exit 1`).

Why the sweeps, if killing the group already worked? Because a child can leave
the group. mini-swe-agent starts every command in a new session, and here
`perl` calls `setpgrp`. Killing only the group, as simple code would:

```text
  pid 8526    bash       parent 8524    group 8526 (the run's group)
  pid 8527    sleep      parent 8526    group 8526 (the run's group)
  pid 8528    perl       parent 8526    group 8528 (LEFT the run's group)
reaped     group 8526 killed, and nothing else
survivor   pid 8528 perl is still running
cleanup    the runner killed them now; 0 still running
```

With the sweeps, the same command:

```text
  pid 8725    bash       parent 8724    group 8725 (the run's group)
  pid 8726    sleep      parent 8725    group 8725 (the run's group)
  pid 8727    perl       parent 8725    group 8727 (LEFT the run's group)
reaped     group 8725 killed; after it, 2 sweeps signalled 2 processes more (the tree, every group it used, the marker)
survivors  0 of 3 (ps -p 8725,8726,8727 lists none of them)
```

`pgrep` found nothing again. A poll had recorded the group `perl` moved to
while the run was alive, so the sweep could find it there. The marker is the
third net. It catches orphans that left both the tree and its groups before
any poll saw them. But macOS hides the environment of its own programs (`sh`,
`sleep`) from `ps`, as the project notes, so the marker works for Python and
Node processes, which is what a harness's children usually are. The sweeps
only touch processes that started after the run began, so an older process is
never killed by mistake.

## The grader runs outside the box: `grade_safe.py`

The grader must read the gold answers, so it runs outside the box, where your
secrets are readable. An agent inside the box cannot read a secret. But it can
make `answer.txt` a link to one, because making a link reads nothing.
`grade_safe.py` lets the box make four `answer.txt` files, then grades each one
two ways. The "secret" is the decoy `.env`:

```text
case plain: the agent ran, inside the box
  $ echo 30 > answer.txt
  exit 0
  naive grader (opens the path): correct · read '30'
  safe grader  (O_NOFOLLOW): correct · read '30' (0.000 s)

case link: the agent ran, inside the box
  $ ln -s $CH/decoys/home/project/.env answer.txt; cat answer.txt >/dev/null
  | cat: answer.txt: Operation not permitted
  exit 1
  naive grader (opens the path): wrong · read 'decoy for chapter 19: not a real secret'
  safe grader  (O_NOFOLLOW): refused unread: answer.txt is not a regular file (0.000 s)

case hardlink: the agent ran, inside the box
  $ ln $CH/decoys/home/project/.env answer.txt
  | ln: $CH/decoys/home/project/.env: Operation not permitted
  exit 1
  naive grader (opens the path): refused unread: no answer.txt
  safe grader  (O_NOFOLLOW): refused unread: no answer.txt (0.000 s)

case pipe: the agent ran, inside the box
  $ mkfifo answer.txt
  exit 0
  naive grader (opens the path): not run (open() would wait forever for a writer)
  safe grader  (O_NOFOLLOW): refused unread: answer.txt is not a regular file (0.000 s)
```

The box could not read through its own link, but the naive grader did, and
printed the "secret" as the agent's answer. The safe grader opens
`answer.txt` the way the project's `mhrl/tasks/core.py` does:

- `O_NOFOLLOW` refuses a link, with no gap between a check and the open;
- `O_NONBLOCK` stops a named pipe from making it wait forever;
- `fstat` on the opened file accepts only a regular file;
- it reads at most 64 KiB.

A hard link is a second name, so `O_NOFOLLOW` cannot see it. The box already
refused to make one to a file it may not read. The safe grader also refuses
any `answer.txt` with more than one name. A unit test covers that branch,
because the box never let the demo get that far.

## If the box cannot be applied

Nothing runs. `sandbox-exec` with the profile but without its parameters:

```text
sandbox-exec: invalid data type of path filter; expected pattern, got boolean
sandbox-exec exit 65
```

The `echo` never printed. `runner.py` and `attacks.py` also refuse to start
when this is not macOS, when `sandbox-exec` is missing, or when the profile is
missing ("nothing ran"). They never fall back to running without the box.

## The answer to the puzzle

The rule was the one that hid the folder holding every rollout. It was added
for a good reason, so that no rollout could read another's answer. But hiding a
folder hides every path through it. `pwd` works on macOS without reading the
parent folders, and so do the shell, `sqlite3` and `echo >`. `realpath` reads
them all, and Pi's write tool calls `realpath` before every write. The tool
failed with EPERM, `answer.txt` was never written, and Pi scored 0 of 24 after
SFT and after RL, while the model was answering. Removed, the rule gave 9 of 24
at both checkpoints. The opposite risk is the one the escape table shows. The
fix removed the protection along with the bug, and a rollout can read its
neighbour's answer again. A box is a trade, and both sides of it have to be
measured: every attempt that must fail, and every step an agent needs.

## What the chapter showed, and where it differed from the plan

| the plan said | measured |
| --- | --- |
| `sandbox.sb` is the project's profile, applied with `sandbox-exec`; if it cannot be applied, nothing runs | as planned: rules unchanged, folders under `COURSE_HOME/19-sandbox`, `TASKS_DIR` = this chapter's `data/`; `sandbox-exec` refused an unappliable profile with exit 65 and the command never ran |
| `runner.py`: clean environment, own session, log file, run-id marker, two-pass kill on timeout | as planned: 5 variables, own session, `$CH/logs/<run>/command.txt`, `MHRL_RUN_ID`; added `--group-only` to show what the sweeps catch |
| write in the task folder allowed; write `~/escape.txt` or into the repo denied | as planned (EPERM); the files get a fresh random name each run, so no file of yours is touched |
| `cat ~/.ssh/id_ed25519` denied, "it does not even appear to exist" | corrected: through `~` (the fake home) it is "No such file"; by its real path `cat` says "Operation not permitted" (EPERM), and only `test -e` says it does not exist |
| reading the gold-answers file denied | as planned (EPERM) |
| writing into another rollout's folder through `/tmp` denied | as planned, both into the neighbour's folder and into the shared `/tmp`; but READING the neighbour's `answer.txt` is allowed under the project's profile, although the project's README says other rollouts' folders are unreadable (flagged above) |
| `security find-generic-password` denied | measured as exit 44 ("could not be found") inside, exit 0 outside; the fake home alone hides the keychain, so a second probe sets `HOME` back and is still refused |
| `curl https://huggingface.co` cannot resolve | as planned (exit 6); an address (`1.1.1.1`) fails to connect (exit 7) |
| the localhost probe on the engine's port 8100 | chapter 18's engine is not built yet, so a throwaway server stands in for it; 8100 was taken by another program on this Mac, so the stand-in took 8119 |
| `env` shows a handful of clean variables | 5 set by the runner, 3 added by bash |
| `--timeout 5 -- bash -c 'sleep 100 & sleep 100 & wait'`: the whole tree is gone, `ps` shows no survivors | as planned: 3 processes seen, 0 survivors, `pgrep` found none; a child that left the group survived a group-only kill and not the sweeps |
| the project allows 300 s per rollout | `rollout_timeout_s: int = 300` (`mhrl/config.py`), the runner's default |
| `grade_safe.py` refuses an `answer.txt` linked to a secret, unread | as planned, with a decoy as the secret; the naive grader read it; also a named pipe (refused at once) and a hard link (the box refused to make it) |
| `--profile v1`, which also hid the folders above the task folder, makes `pwd` fail | corrected: on macOS 26.6.2 `pwd` and `/bin/pwd -P` work under v1; `ls ..` fails, and `realpath` of a file not yet written fails with EPERM, which is the first step of Pi's write tool: no `answer.txt`, reward 0 |
| that bug scored Pi 0% after SFT and RL until it was fixed (38% on re-evaluation) | as the project recorded: 0/24 after SFT and after RL, 9/24 = 37.5% at both after the fix (base: 3/24, then 4/24) |
| Claude Code runs with `bypassPermissions`, so the box is the safety boundary | as chapter 16 showed (its run's `init` line says `bypassPermissions`) |
| folder `course/19-sandbox/` | `19-sandbox/` at the root of this repository, like every chapter |

- **The project's numbers** in this chapter are Pi's accuracies and the
  300-second limit above, read from the project's own files.
- **The article's numbers**: there are none in this chapter.
- **Differences from the project**: none in the rules. The course adds the
  checks that a hidden folder overlaps no writable folder, the grader's refusal
  of a hard-linked `answer.txt`, and the reaper's report of what it saw,
  killed and left alive.

## What is in this folder

- `sandbox.sb`: the project's profile; `sandbox_v1.sb`: its first,
  too-strict version (one rule more).
- `sandbox.py`: the profile's parameters, the safety checks on the folders,
  and the `sandbox-exec` command; it refuses instead of falling back.
- `process.py`: running one command with a log, a hard timeout and the
  two-pass reaping, and the report of what survived.
- `runner.py`: one rollout: folders, the clean environment, the command in the
  box.
- `attacks.py`: the decoy check, the escape attempts, the agent's work, the
  table.
- `decoys.py`: the fake secrets.
- `write_tool.py`: Pi's write tool, step for step.
- `grade_safe.py`: the grader that opens `answer.txt` without following links;
  `grader.py`: chapter 1's answer check (unchanged).
- `tasks.py`: the task, the chapter's folders, and how paths are printed.
- `data/`: chapter 1's `clinic.db` and the task record (the gold answer).

## Words from this chapter

- **Sandbox, Seatbelt, `sandbox-exec`, profile**: the box, macOS's version of
  it, the command that starts a program inside it, and its rules.
- **Allow default, deny, last match wins**: start open, take things away, and
  the latest matching rule decides.
- **Real path**: every link resolved; `/tmp` is `/private/tmp`.
- **Write confinement**: writes only in the task folder, the state folder and
  `/dev`.
- **Secret protection**: credentials and `.env` files unreadable, the keychain
  unreachable, a fake `HOME`, a clean environment.
- **Network isolation**: no DNS and no outside address; loopback only.
- **Anti-reward-hacking isolation**: the gold-answer file unreadable (a
  neighbour rollout's folder is not: see the list above).
- **Sandbox overreach**: a rule that hides too much breaks a tool silently
  and fakes a score.
- **Hard timeout, process-tree reaping**: kill the group, sweep the tree,
  every group it used and the marker, twice, then check with `ps`.
- **Grader hardening**: `O_NOFOLLOW`, `O_NONBLOCK`, `fstat`, a size cap, one
  name only.
- **Permission modes**: `bypassPermissions` runs tools without asking, which
  is why the box, not the harness, is the boundary.

Next: chapter 20, the capture proxy: recording a program you didn't write.
