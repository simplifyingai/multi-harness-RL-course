"""The job every chapter uses: tasks, the instruction, and fresh work folders.

A task is a question about a small SQLite database. The agent gets a folder
holding a copy of that database as ``data.db`` and must write only the answer
to ``answer.txt``. This is a short copy of the parts of the project's
``mhrl/tasks/core.py`` that chapter 1 needs.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TASK_FILES = ("train.jsonl", "test.jsonl")

DATABASE_FILE = "data.db"
ANSWER_FILE = "answer.txt"

# The course's one setting: where models, adapters and work folders live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "01-the-job"
WORK_DIR = "work"
WORKDIR_MODE = 0o755  # mkdtemp makes owner-only folders; agents need normal ones
ID_SEPARATOR = "-"

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

# Word for word what every harness gives the agent. It never names a table:
# finding the tables is part of the job.
INSTRUCTION_TEMPLATE = (
    "There is a SQLite database named {database} in the current folder. "
    "Answer this question using the data:\n\n"
    "{question}\n\n"
    "When you have the answer, write ONLY the answer (no explanation, no units) "
    "to a file named {answer_file} in the current folder. {hint}"
)


class CourseError(Exception):
    """Base class for this chapter's errors, so a script can catch them all."""


class TaskNotFoundError(CourseError):
    """Raised when a task id or a work folder names no known task."""


@dataclass(frozen=True)
class Task:
    """One data question.

    Attributes:
        task_id: Unique id, e.g. ``"shop-easy-001"``.
        split: ``"train"`` or ``"test"`` (test databases are never trained on).
        domain: Which database, e.g. ``"shop"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        db_file: The database, relative to ``data/``.
        question: What the agent is asked.
        gold_sql: The query whose result is the answer. The agent never sees it.
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


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def work_root() -> Path:
    """Return the folder that holds this chapter's work folders.

    Returns:
        ``$COURSE_HOME/01-the-job/work``.
    """
    return course_home() / CHAPTER / WORK_DIR


def load_tasks() -> list[Task]:
    """Load the 230 bundled tasks (180 train, 50 test).

    Returns:
        Every task, train first.

    Raises:
        OSError: If a bundled task file cannot be read.
        ValueError: If a line is not a task record.
    """
    tasks = []
    for name in TASK_FILES:
        with (DATA_DIR / name).open(encoding="utf-8") as handle:
            tasks.extend(Task(**json.loads(line)) for line in handle if line.strip())
    return tasks


def find_task(task_id: str) -> Task:
    """Look a task up by its id.

    Args:
        task_id: For example ``"shop-easy-001"``.

    Returns:
        The task.

    Raises:
        TaskNotFoundError: If no bundled task has that id.
    """
    for task in load_tasks():
        if task.task_id == task_id:
            return task
    raise TaskNotFoundError(f"no task called {task_id!r}")


def task_instruction(task: Task) -> str:
    """Return the instruction every agent receives for a task.

    Args:
        task: The task.

    Returns:
        The instruction text.
    """
    return INSTRUCTION_TEMPLATE.format(
        database=DATABASE_FILE,
        question=task.question,
        answer_file=ANSWER_FILE,
        hint=FORMAT_HINTS[task.answer_type],
    )


def prepare_workdir(task: Task, root: Path | None = None) -> Path:
    """Make a fresh, uniquely named folder holding a copy of the task's database.

    Every call makes a new folder, so many attempts at one task never share
    files. The folder holds only ``data.db``: never the question's answer.

    Args:
        task: The task.
        root: Where to make the folder; defaults to ``work_root()``.

    Returns:
        The new folder, named ``<task_id>-<random>``.

    Raises:
        OSError: If the folder cannot be made or the database copied.
    """
    parent = root or work_root()
    parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=task.task_id + ID_SEPARATOR, dir=parent))
    workdir.chmod(WORKDIR_MODE)
    shutil.copyfile(DATA_DIR / task.db_file, workdir / DATABASE_FILE)
    return workdir


def task_for_workdir(workdir: Path) -> Task:
    """Find the task a work folder belongs to, from the folder's name.

    Args:
        workdir: A folder made by ``prepare_workdir``.

    Returns:
        The task.

    Raises:
        TaskNotFoundError: If the name does not start with a known task id.
    """
    task_id, _, _ = workdir.resolve().name.rpartition(ID_SEPARATOR)
    return find_task(task_id)


def latest_workdir() -> Path:
    """Return the most recently made work folder.

    Returns:
        The newest folder under ``work_root()``.

    Raises:
        TaskNotFoundError: If there is no work folder yet.
    """
    root = work_root()
    folders = [path for path in root.glob("*") if path.is_dir()]
    if not folders:
        raise TaskNotFoundError(f"no work folder in {root}; run new_task.py first")
    return max(folders, key=lambda path: path.stat().st_mtime)


def short_path(path: Path) -> str:
    """Write a path with the home folder as ``~``, for readable output.

    Args:
        path: Any path.

    Returns:
        The path as text.
    """
    home = Path.home()
    if path == home:
        return "~"
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:  # not inside the home folder
        return str(path)
