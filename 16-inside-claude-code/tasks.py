# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The chapter's one task, its instruction, and where the chapter keeps files.

The job is chapter 1's: a folder holds a SQLite database as ``data.db``, and
the agent must write only the answer to ``answer.txt``, which chapter 1's
grader (``grader.py``, copied unchanged) checks. Here it is one bundled task,
``clinic-easy-001``: "How many patients live in Murcia?" (answer: 30).

``uv run tasks.py`` prints the instruction every harness gives the agent,
word for word; ``run_claude.sh`` passes it to Claude Code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TASK_FILE = DATA_DIR / "task.jsonl"

DATABASE_FILE = "data.db"
ANSWER_FILE = "answer.txt"

# The course's one setting: where models, installs and run folders live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "16-inside-claude-code"
RUNS_DIR = "runs"

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
    """One data question (chapter 1's record, unchanged).

    Attributes:
        task_id: Unique id, e.g. ``"clinic-easy-001"``.
        split: ``"train"`` or ``"test"``.
        domain: Which database, e.g. ``"clinic"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        db_file: The database, as chapter 1 stores it (``dbs/clinic.db``).
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

    @property
    def database(self) -> Path:
        """The bundled copy of the task's database in ``data/``."""
        return DATA_DIR / Path(self.db_file).name


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def runs_root() -> Path:
    """Return the folder that holds one subfolder per Claude Code run.

    Returns:
        ``$COURSE_HOME/16-inside-claude-code/runs``.
    """
    return course_home() / CHAPTER / RUNS_DIR


def load_task() -> Task:
    """Load the chapter's one bundled task.

    Returns:
        The task.

    Raises:
        OSError: If the task file cannot be read.
        ValueError: If it is not JSON.
        TypeError: If its fields are not a task record's.
    """
    line = TASK_FILE.read_text(encoding="utf-8").strip()
    return Task(**json.loads(line))


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


def main() -> None:
    """Print the task's instruction."""
    print(task_instruction(load_task()))


if __name__ == "__main__":
    main()
