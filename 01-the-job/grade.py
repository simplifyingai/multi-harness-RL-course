# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Grade the answer an agent (or you) left in a task folder.

With no folder given, grades the folder ``new_task.py`` made last. Run it
with ``uv run grade.py`` or ``uv run grade.py <folder>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grader import grade
from tasks import (
    ANSWER_FILE,
    CourseError,
    latest_workdir,
    short_path,
    task_for_workdir,
)


def main() -> None:
    """Grade one folder and print the verdict and the reward.

    Raises:
        SystemExit: With a plain message when there is no folder to grade.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path)
    args = parser.parse_args()
    try:
        workdir = args.folder or latest_workdir()
        task = task_for_workdir(workdir)
    except CourseError as error:
        sys.exit(str(error))

    result = grade(task, workdir)
    verdict = "correct" if result.correct else f"wrong: {result.reason}"
    print(f"folder     {short_path(workdir.resolve())}")
    print(f"question   {task.question}")
    print(f"{ANSWER_FILE} {result.submitted if result.submitted else '(none)'}")
    print(f"verdict    {verdict}")
    print(f"reward     {result.reward}")


if __name__ == "__main__":
    main()
