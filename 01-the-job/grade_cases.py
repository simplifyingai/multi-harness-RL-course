# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Six answers, six verdicts: what the grader forgives and what it does not.

Run it with ``uv run grade_cases.py``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from grader import check_answer, grade
from tasks import LIST, NUMBER, TEXT, Task

COLUMN = 18
VERDICT_WIDTH = 9


@dataclass(frozen=True)
class Case:
    """One submitted answer and the task it answers.

    Attributes:
        answer_type: ``"number"``, ``"text"`` or ``"list"``.
        expected: The correct answer.
        submitted: What the agent wrote, or None for no answer file at all.
        lesson: What the case shows, in plain words.
    """

    answer_type: str
    expected: str
    submitted: str | None
    lesson: str


CASES = (
    Case(NUMBER, "1234", "$1,234.00", "currency and commas are fine"),
    Case(TEXT, "Ann Lee", "Dr. Ann Lee", "a title is ignored"),
    Case(LIST, "Ann Lee, Bo Chen", "Bo Chen; Ann Lee", "a list in any order"),
    Case(NUMBER, "1234", "1234.5", "off by more than 0.01"),
    Case(NUMBER, "13", "13 customers", "words are not a number"),
    Case(NUMBER, "13", None, "the agent never wrote the file"),
)


def judge(case: Case) -> tuple[bool, str]:
    """Grade one case with the real grader.

    Args:
        case: The case.

    Returns:
        Whether it is correct, and the grader's reason.
    """
    task = Task(
        task_id="case",
        split="train",
        domain="example",
        difficulty="easy",
        db_file="",
        question="",
        gold_sql="",
        answer=case.expected,
        answer_type=case.answer_type,
    )
    if case.submitted is None:
        with tempfile.TemporaryDirectory() as empty_folder:
            result = grade(task, Path(empty_folder))
    else:
        result = check_answer(task, case.submitted)
    return result.correct, result.reason


def main() -> None:
    """Print one line per case."""
    print(f"{'expected':<{COLUMN}}{'submitted':<{COLUMN}}verdict")
    for case in CASES:
        correct, reason = judge(case)
        submitted = "(no file)" if case.submitted is None else case.submitted
        verdict = "correct" if correct else "wrong"
        print(
            f"{case.expected:<{COLUMN}}{submitted:<{COLUMN}}{verdict:<{VERDICT_WIDTH}}{case.lesson}"
        )
        if not correct:
            print(f"{'':<{COLUMN * 2}}  ({reason})")


if __name__ == "__main__":
    main()
