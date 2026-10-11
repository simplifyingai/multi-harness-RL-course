"""The course's data questions, and what chapter 1's grader needs.

A short copy of ``01-the-job/tasks.py``: the answer file, the three answer
shapes, the ``Task`` record the grader reads, and the bundled questions.
``data/train.jsonl``, ``data/test.jsonl`` and the five databases in
``data/dbs`` are chapter 1's files, byte for byte, which are the project's
``data/tasks`` files: 180 training questions over shop, school and library,
and 50 held out over clinic and airline. Chapters copy code from each other;
they never import across folders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
TRAIN_FILE = "train.jsonl"
TEST_FILE = "test.jsonl"

ANSWER_FILE = "answer.txt"

# The three shapes an answer can take; each is graded its own way.
NUMBER = "number"
TEXT = "text"
LIST = "list"
ANSWER_TYPES = (NUMBER, TEXT, LIST)


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


def load_tasks(name: str, folder: Path = DATA_DIR) -> list[Task]:
    """Load one bundled task file, in file order.

    Args:
        name: ``"train.jsonl"`` or ``"test.jsonl"``.
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
