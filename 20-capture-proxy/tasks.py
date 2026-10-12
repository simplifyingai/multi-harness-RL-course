"""The held-out questions, the instruction every agent gets, and its work folder.

A short copy of ``01-the-job/tasks.py`` (by way of chapters 15 to 17): the
answer file, the three answer shapes, the ``Task`` record the grader reads,
and chapter 1's instruction, word for word. ``data/test.jsonl`` and the two
databases in ``data/dbs`` are chapter 1's files, byte for byte, which are the
project's: 50 held-out questions over clinic and airline, databases no
training question comes from. As in the project (``prepare_workdir``), each
attempt works in a fresh folder holding a copy of its database as
``data.db``. Chapters copy code from each other; they never import across
folders.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TEST_FILE = "test.jsonl"

DATABASE_FILE = "data.db"
ANSWER_FILE = "answer.txt"

# The three shapes an answer can take; each is graded its own way.
NUMBER = "number"
TEXT = "text"
LIST = "list"
ANSWER_TYPES = (NUMBER, TEXT, LIST)

FORMAT_HINTS = {
    NUMBER: "The answer is a single number.",
    TEXT: "The answer is a single name or value.",
    LIST: "The answer is a list: write the items separated by commas.",
}

# Chapter 1's instruction, word for word. It never names a table: finding the
# tables is part of the job.
INSTRUCTION_TEMPLATE = (
    "There is a SQLite database named {database} in the current folder. "
    "Answer this question using the data:\n\n"
    "{question}\n\n"
    "When you have the answer, write ONLY the answer (no explanation, no units) "
    "to a file named {answer_file} in the current folder. {hint}"
)


@dataclass(frozen=True)
class Task:
    """One data question.

    Attributes:
        task_id: Unique id, e.g. ``"clinic-easy-001"``.
        split: ``"train"`` or ``"test"`` (test databases are never trained on).
        domain: Which database, e.g. ``"clinic"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        db_file: The database, relative to ``data/``.
        question: What is asked.
        gold_sql: The query whose result is the answer. The model never sees it.
        answer: The correct answer as text.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.
    """

    task_id: str
    split: str
    domain: str
    difficulty: str
    db_file: str
    question: str
    gold_sql: str
    answer: str
    answer_type: str


def load_tasks(name: str = TEST_FILE, folder: Path = DATA_DIR) -> list[Task]:
    """Load one bundled task file, in file order.

    Args:
        name: The file, ``"test.jsonl"`` by default.
        folder: Where the file is.

    Returns:
        Its tasks.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a line is not JSON.
        TypeError: If a line's fields are not a task's fields.
    """
    with (folder / name).open(encoding="utf-8") as handle:
        return [Task(**json.loads(line)) for line in handle if line.strip()]


def find_task(task_id: str) -> Task:
    """Find one held-out question by its id.

    Args:
        task_id: Such as ``"clinic-easy-001"``.

    Returns:
        The task.

    Raises:
        SystemExit: If there is no such question, naming a few that exist.
    """
    tasks = load_tasks()
    for task in tasks:
        if task.task_id == task_id:
            return task
    examples = ", ".join(task.task_id for task in tasks[:3])
    raise SystemExit(f"no held-out question {task_id!r}; try {examples}")


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
        database=DATABASE_FILE,
        question=task.question,
        answer_file=ANSWER_FILE,
        hint=FORMAT_HINTS[task.answer_type],
    )


def prepare_workdir(task: Task, folder: Path) -> Path:
    """Make a fresh folder holding a copy of the task's database as data.db.

    Args:
        task: The task.
        folder: The folder to (re)make; anything already in it is removed,
            so an old answer.txt can never be graded again.

    Returns:
        The folder, as an absolute path.

    Raises:
        OSError: If the folder cannot be made or the database copied.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copyfile(DATA_DIR / task.db_file, folder / DATABASE_FILE)
    return folder.resolve()
