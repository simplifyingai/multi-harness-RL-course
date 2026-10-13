"""A new task family: questions about a spreadsheet, the instruction, the work folder.

Chapter 1's job, with new data. A task is a question about the CSV file
``data/sales.csv``; the agent gets a folder holding a copy of it as
``data.csv`` and must write only the answer to ``answer.txt``. Everything a
harness reads stays chapter 1's, word for word, but the first sentence of
the instruction, which names the file. The three answer shapes and their
format hints are chapter 1's too, so chapter 1's grader (``grader.py``,
copied unchanged) grades this family as it grades the SQLite one: lenient
about format, strict about content.

The tasks themselves are made by ``make_csv_tasks.py`` and written under
``COURSE_HOME/29-make-it-yours/csv-tasks/seed-<seed>.jsonl``; any script
that needs them makes them first when that file is missing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from lfm_files import chapter_home

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

DATA_FILE = "data.csv"  # the spreadsheet's name in every task folder
ANSWER_FILE = "answer.txt"
DOMAIN = "sales"  # the family's one spreadsheet
TASKS_FOLDER = "csv-tasks"
DEFAULT_SEED = 0

# The three shapes an answer can take; each is graded its own way.
NUMBER = "number"
TEXT = "text"
LIST = "list"
ANSWER_TYPES = (NUMBER, TEXT, LIST)
DIFFICULTIES = ("easy", "medium", "hard")

FORMAT_HINTS = {
    NUMBER: "The answer is a single number.",
    TEXT: "The answer is a single name or value.",
    LIST: "The answer is a list: write the items separated by commas.",
}

# Chapter 1's instruction, but for its first sentence. It never names a
# column: reading the file's header is part of the job.
INSTRUCTION_TEMPLATE = (
    "There is a CSV file named {data_file} in the current folder. "
    "Answer this question using the data:\n\n"
    "{question}\n\n"
    "When you have the answer, write ONLY the answer (no explanation, no units) "
    "to a file named {answer_file} in the current folder. {hint}"
)


@dataclass(frozen=True)
class Task:
    """One spreadsheet question.

    Attributes:
        task_id: Unique id, such as ``"csv-s0-easy-01"`` (seed 0).
        domain: The spreadsheet, ``"sales"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        data_file: The spreadsheet, relative to ``data/``.
        question: What is asked.
        gold_sql: The query, over the spreadsheet loaded with typed columns,
            whose result is the answer. The agent never sees it.
        answer: The correct answer as text.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.
    """

    task_id: str
    domain: str
    difficulty: str
    data_file: str
    question: str
    gold_sql: str
    answer: str
    answer_type: str


def gold_folder() -> Path:
    """The folder holding the family's task records, gold answers included.

    The sandbox hides it from every harness run (chapter 19): reading the
    gold answers would be reward hacking. Chapter 24's held-out records were
    bundled in its ``data/`` folder, which its box hid; this family's are
    made under ``COURSE_HOME``, so this folder is the one the box must hide.

    Returns:
        ``COURSE_HOME/29-make-it-yours/csv-tasks``.
    """
    return chapter_home() / TASKS_FOLDER


def tasks_path(seed: int = DEFAULT_SEED) -> Path:
    """Where ``make_csv_tasks.py`` writes one seed's tasks.

    Args:
        seed: The generator's seed.

    Returns:
        ``COURSE_HOME/29-make-it-yours/csv-tasks/seed-<seed>.jsonl``.
    """
    return gold_folder() / f"seed-{seed}.jsonl"


def load_tasks(path: Path) -> list[Task]:
    """Load a task file, in file order.

    Args:
        path: A JSON-lines file of tasks.

    Returns:
        Its tasks.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a line is not JSON, or names an unknown answer type or
            difficulty (the message names the line).
        TypeError: If a line's fields are not a task's fields.
    """
    found = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = Task(**json.loads(line))
            if task.answer_type not in ANSWER_TYPES:
                raise ValueError(
                    f"{path.name}:{number}: answer type {task.answer_type!r}"
                )
            if task.difficulty not in DIFFICULTIES:
                raise ValueError(
                    f"{path.name}:{number}: difficulty {task.difficulty!r}"
                )
            found.append(task)
    return found


def find_task(task_id: str, tasks: list[Task]) -> Task:
    """Find one task by its id.

    Args:
        task_id: Such as ``"csv-s0-easy-01"``.
        tasks: The tasks to look in.

    Returns:
        The task.

    Raises:
        SystemExit: If there is no such task, naming a few that exist.
    """
    for task in tasks:
        if task.task_id == task_id:
            return task
    examples = ", ".join(task.task_id for task in tasks[:3])
    raise SystemExit(f"no task {task_id!r}; try {examples}")


def task_instruction(task: Task) -> str:
    """Return the instruction every agent receives for a task.

    Args:
        task: The task.

    Returns:
        The instruction text.

    Raises:
        KeyError: If the task has an unknown answer type.
    """
    return INSTRUCTION_TEMPLATE.format(
        data_file=DATA_FILE,
        question=task.question,
        answer_file=ANSWER_FILE,
        hint=FORMAT_HINTS[task.answer_type],
    )


def prepare_workdir(task: Task, folder: Path) -> Path:
    """Make a fresh folder holding a copy of the task's spreadsheet as data.csv.

    Args:
        task: The task.
        folder: The folder to (re)make; anything already in it is removed,
            so an old answer.txt can never be graded again.

    Returns:
        The folder, as an absolute path.

    Raises:
        OSError: If the folder cannot be made or the spreadsheet copied.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copyfile(DATA_DIR / task.data_file, folder / DATA_FILE)
    return folder.resolve()
