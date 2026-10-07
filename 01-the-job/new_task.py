# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Set up one task the way every agent receives it: a folder and an instruction.

The folder holds a copy of the task's database as ``data.db`` and nothing
else: not the question's answer, not the SQL that finds it. Run it with
``uv run new_task.py shop-easy-001``.
"""

from __future__ import annotations

import argparse
import sys

from tasks import (
    CourseError,
    find_task,
    prepare_workdir,
    short_path,
    task_instruction,
)

DEFAULT_TASK = "shop-easy-001"
RULE = "-" * 60


def main() -> None:
    """Make the folder and print the instruction.

    Raises:
        SystemExit: With a plain message when the task id is unknown.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_id", nargs="?", default=DEFAULT_TASK)
    args = parser.parse_args()
    try:
        task = find_task(args.task_id)
    except CourseError as error:
        sys.exit(str(error))

    workdir = prepare_workdir(task)
    where = f"{task.split} · {task.domain} · {task.difficulty}"
    print(f"task     {task.task_id}  ({where})")
    print(f"folder   {short_path(workdir)}")
    print(f"holds    {', '.join(sorted(p.name for p in workdir.iterdir()))}")
    print()
    print("the instruction every agent receives:")
    print(RULE)
    print(task_instruction(task))
    print(RULE)


if __name__ == "__main__":
    main()
