"""The answer-key teacher: it knows the right query, and plays the agent.

The project's teacher (``mhrl/engine/oracle.py``) is a script, not a model.
The question generator knows each task's right SQL, so the script does what a
careful analyst does, in four slightly different ways, and the student learns
from those runs. This is the same teacher on the toy. It knows how many
tables each question needs and that QUERY_A is the right query.

Its paths are the project's four, picked the project's way, by the question's
id (``pattern_for``, line for line):

* direct: LOOK as many times as the question needs, QUERY_A, WRITE, FINISH;
* peek: one LOOK more than it needs first (the project's teacher looks at a
  few rows);
* tables: LOOK at all four tables first (the project's teacher lists the
  tables, then reads the ones it needs);
* verify: direct, plus a CHECK of answer.txt before FINISH.

Why four? The project's docstring says it: "A first version always did schema
-> query -> write -> finish, and the student learned 'the fourth reply is the
final answer' by POSITION: after one extra query it finished without writing
answer.txt (dry run: Claude Code 100% -> 17%)." That first version is
``FIXED_LENGTH`` here: LOOK, QUERY_A, WRITE, FINISH, every time.
``NEVER_CHECKS`` is the four paths with verify's CHECK left out.
"""

from __future__ import annotations

from environment import (
    CHECK,
    FINISH,
    LINKED_TABLES,
    LOOK,
    QUERY_A,
    WRITE,
    tables_needed,
)
from questions import Question
from tasks import Task

# The project's four paths (mhrl/engine/oracle.py: PATTERNS).
DIRECT = "direct"
PEEK = "peek"
TABLES = "tables"
VERIFY = "verify"
PATTERNS = (DIRECT, PEEK, TABLES, VERIFY)
FIXED_PATH = "fixed"  # the first version's only path

# The teachers this chapter compares.
VARIED = "varied"  # the project's teacher, four paths
FIXED_LENGTH = "fixed-length"  # its first version: one path of four calls
NEVER_CHECKS = "never-checks"  # four paths, but verify's CHECK left out
TEACHER_NOTES = {
    VARIED: "four paths: direct, peek, tables, verify (the project's teacher)",
    FIXED_LENGTH: "one path: LOOK, QUERY_A, WRITE, FINISH (its first version)",
    NEVER_CHECKS: "four paths, but verify never CHECKs",
}
EXTRA_LOOKS = 1  # peek looks once more than it needs


def pattern_for(task: Task) -> str:
    """Pick a task's path the project's way: by the letters of its id.

    The arithmetic is ``AnswerKeyTeacher._steps``'s line for line, so a task
    always gets the same path, and the paths come out roughly even.

    Args:
        task: The task.

    Returns:
        ``"direct"``, ``"peek"``, ``"tables"`` or ``"verify"``.
    """
    pattern = PATTERNS[sum(map(ord, task.task_id)) % len(PATTERNS)]
    return pattern


def path_name(question: Question, teacher: str) -> str:
    """Say which path a teacher takes on a question.

    Args:
        question: The question.
        teacher: ``"varied"``, ``"fixed-length"`` or ``"never-checks"``.

    Returns:
        The path's name: one of the four, or ``"fixed"``.
    """
    if teacher == FIXED_LENGTH:
        return FIXED_PATH
    pattern = pattern_for(question.task)
    if teacher == NEVER_CHECKS and pattern == VERIFY:
        return DIRECT  # the same demonstration, without its CHECK
    return pattern


def teacher_actions(question: Question, teacher: str) -> list[str]:
    """Write down the teacher's actions for one question, call by call.

    Args:
        question: The question to demonstrate.
        teacher: ``"varied"``, ``"fixed-length"`` or ``"never-checks"``.

    Returns:
        The actions, in order; the last one is FINISH.
    """
    pattern = path_name(question, teacher)
    if pattern == FIXED_PATH:
        return [LOOK, QUERY_A, WRITE, FINISH]
    looks = tables_needed(question.task.gold_sql)
    if pattern == PEEK:
        looks += EXTRA_LOOKS
    elif pattern == TABLES:
        looks = len(LINKED_TABLES)
    actions = [LOOK] * looks + [QUERY_A, WRITE]
    if pattern == VERIFY:
        actions.append(CHECK)
    return [*actions, FINISH]
