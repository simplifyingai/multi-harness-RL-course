# Chapter 1. The job, the grader and the puzzle

Every chapter of this course uses one job. A folder holds a small database and
a question, and the agent must write only the answer into `answer.txt`. A
strict program (the grader) checks that answer against a hidden correct query.

The puzzle: one small model, unchanged, solved 100%, 62% or 17% of the same
tasks, depending only on the program wrapped around it. That program is called
the **harness**. This course trains the model to do better inside harnesses
people actually use, and never changes the harnesses themselves.

## Run it

You need macOS (any Mac works for this chapter), [uv](https://docs.astral.sh/uv/)
and the `sqlite3` command-line tool, which macOS already has.

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/01-the-job
uv run check_machine.py
```

Every script uses only Python's standard library, and `uv run` gives each one
its own clean Python. Scripts write only under `COURSE_HOME`
(default `~/.cache/mhrl-course`), never into this folder.

| script | what it shows |
| --- | --- |
| `check_machine.py` | your chip, memory, Python, uv, sqlite3, free disk and `COURSE_HOME` |
| `make_tasks.py --seed 0` | questions drafted from templates, each with the SQL that answers it; the same seed gives the same md5 |
| `new_task.py shop-easy-001` | a fresh folder holding only `data.db`, and the instruction every agent receives |
| `grade.py` | the grader's verdict on the folder you just made, and the reward |
| `grade_cases.py` | six answers: what the grader forgives and what it does not |
| `replay_agent.py` | two recorded attempts by the same model in two harnesses, and the scores |

## Do the agent's job by hand

```bash
uv run new_task.py shop-easy-001
cd ~/.cache/mhrl-course/01-the-job/work/shop-easy-001-*   # the folder it printed
sqlite3 data.db .tables
sqlite3 data.db ".schema customers"
sqlite3 data.db "SELECT COUNT(*) FROM customers WHERE city = 'Paris'"
echo 13 > answer.txt
cd -
uv run grade.py
```

```text
folder     ~/.cache/mhrl-course/01-the-job/work/shop-easy-001-75rkaibw
question   How many customers live in Paris?
answer.txt 13
verdict    correct
reward     1
```

Write `14` instead and the grader says `wrong: expected '13', got '14'` with
reward 0.

## What is in this folder

- `tasks.py`: the tasks, the instruction and the work folders (a short copy of
  the project's `mhrl/tasks/core.py`).
- `grader.py`: the grader, copied from the same file. It agrees with the
  original on 3,680 test answers across all 230 tasks.
- `data/dbs/`: the five databases (shop, school and library for training;
  clinic and airline are never trained on).
- `data/train.jsonl`, `data/test.jsonl`: the project's 180 training and 50 test
  tasks, each with its question, gold SQL and answer.
- `data/transcripts/`: the two recorded attempts `replay_agent.py` replays, as
  structured JSON plus the raw logs they came from.
- `data/results_as_released.json`: the project's scores for the model before
  any training.

## Words from this chapter

- **Task**: a question about a database, plus the folder the agent works in.
- **Gold SQL**: the query whose result is the correct answer. The agent never
  sees it.
- **Grader**: the program that compares `answer.txt` with the gold answer.
- **Reward**: the grader's verdict as a number, 1 for correct and 0 for wrong.
  Chapter 2 starts learning from it.
- **Harness**: the program around a model that turns its replies into actions
  (running commands, writing files) and decides when to stop.
- **As released**: the model exactly as its makers published it, before we
  train it.

Next: [chapter 2](../02-plain-rl/), plain RL on a toy version of this job.
