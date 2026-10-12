# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The chapter's one task, where the chapter keeps its files, and how it prints paths.

The job is chapter 1's: a folder holds a SQLite database as ``data.db``, and
the agent must write only the answer to ``answer.txt``. Here it is one bundled
task, ``clinic-easy-001``: "How many patients live in Murcia?" (answer: 30).

The task record in ``data/task.jsonl`` holds the gold answer, so it is the
file the sandbox must hide: an agent that could read it would not need the
database at all.

``uv run tasks.py`` prints the task and the folders the chapter uses.
"""

from __future__ import annotations

import json
import os
import pwd
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = HERE / "data"
TASK_FILE = DATA_DIR / "task.jsonl"

DATABASE_FILE = "data.db"
ANSWER_FILE = "answer.txt"

# The course's one setting: where models, installs and run folders live.
# (The default follows $HOME, like every chapter's; the box's own idea of
# "home" is real_home() below, which ignores $HOME.)
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
TEXT_ENCODING = "utf-8"
CHAPTER = "19-sandbox"

# Folders inside COURSE_HOME/19-sandbox.
ROLLOUTS_DIR = "rollouts"  # one task folder per rollout (the agent's cwd)
STATE_DIR = "state"  # one private HOME + TMPDIR per rollout
LOGS_DIR = "logs"  # one log file per command
DECOYS_DIR = "decoys"  # fake secrets, to prove the profile before real paths
GRADING_DIR = "grading"  # grade_safe.py's demo folders

# How printed paths are shortened, so output can be shared and fits a screen.
SHOWN_CHAPTER_HOME = "$CH"
SHOWN_REPO = "$REPO"
SHOWN_HOME = "/Users/you"


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


def chapter_folder() -> Path:
    """Return this chapter's folder inside the course cache, without making it.

    Returns:
        ``$COURSE_HOME/19-sandbox``, resolved as far as it exists.
    """
    return (course_home() / CHAPTER).resolve()


def chapter_home() -> Path:
    """Return this chapter's folder inside the course cache, made if missing.

    Returns:
        ``$COURSE_HOME/19-sandbox``, created and fully resolved, because the
        sandbox matches real paths only.
    """
    folder = course_home() / CHAPTER
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def real_home() -> Path:
    """Return the user's real home folder, ignoring ``$HOME``.

    A program inside the box gets a fake ``$HOME``, so the real one is read
    from the password database, as the project does.

    Returns:
        The resolved home folder.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def show(text: str) -> str:
    """Shorten the chapter's folders and the home folder in printed text.

    Args:
        text: Any text that may contain absolute paths.

    Returns:
        The text with ``$CH`` for ``COURSE_HOME/19-sandbox``, ``$REPO`` for
        this repository and ``/Users/you`` for the real home folder. The
        longest path is replaced first, so a path inside another keeps its
        short name.
    """
    replacements = [
        (str(chapter_folder()), SHOWN_CHAPTER_HOME),
        (str(REPO_ROOT.resolve()), SHOWN_REPO),
        (str(real_home()), SHOWN_HOME),
    ]
    for path, short in sorted(replacements, key=lambda pair: -len(pair[0])):
        text = text.replace(path, short)
    return text


def legend() -> list[str]:
    """Say what the short names in printed paths stand for.

    Returns:
        Two lines, with only the home folder shortened.
    """
    home = str(real_home())
    chapter = str(chapter_folder()).replace(home, SHOWN_HOME)
    repo = str(REPO_ROOT.resolve()).replace(home, SHOWN_HOME)
    return [
        f"{SHOWN_CHAPTER_HOME:<6} = {chapter}   (COURSE_HOME/{CHAPTER})",
        f"{SHOWN_REPO:<6} = {repo}   (this repository)",
    ]


def load_task() -> Task:
    """Load the chapter's one bundled task.

    Returns:
        The task.

    Raises:
        OSError: If the task file cannot be read.
        ValueError: If it is not JSON.
        TypeError: If its fields are not a task record's.
    """
    line = TASK_FILE.read_text(encoding=TEXT_ENCODING).strip()
    return Task(**json.loads(line))


def main() -> None:
    """Print the task and the folders the chapter uses."""
    task = load_task()
    print(f"task       {task.task_id}: {task.question}")
    print(f"answer     {task.answer} (in the task record, the gold answer)")
    print(f"gold file  {show(str(TASK_FILE))} (the sandbox hides its folder)")
    for line in legend():
        print(line)


if __name__ == "__main__":
    main()
