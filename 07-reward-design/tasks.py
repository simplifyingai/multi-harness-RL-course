"""What chapter 1's grader needs, and where this chapter keeps its files.

A short copy of ``01-the-job/tasks.py``: the answer file, the three answer
shapes, the ``Task`` record the grader reads, and the course folder. Chapters
copy code from each other; they never import across folders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

DATABASE_FILE = "data.db"
ANSWER_FILE = "answer.txt"

# The course's one setting: where models, adapters and work folders live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "07-reward-design"
WORK_DIR = "work"

# The three shapes an answer can take; each is graded its own way.
NUMBER = "number"
TEXT = "text"
LIST = "list"
ANSWER_TYPES = (NUMBER, TEXT, LIST)


class CourseError(Exception):
    """Base class for this chapter's errors, so a script can catch them all."""


@dataclass(frozen=True)
class Task:
    """One data question.

    Attributes:
        task_id: Unique id, e.g. ``"customers-Paris"``.
        split: ``"train"`` or ``"test"`` (test questions are never trained on).
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
        ``$COURSE_HOME/07-reward-design/work``.
    """
    return course_home() / CHAPTER / WORK_DIR


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
