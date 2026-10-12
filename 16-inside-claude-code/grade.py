# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Grade the answer Claude Code left in a run's task folder, with chapter 1's grader.

``uv run grade.py <task folder>``; ``run_claude.sh`` calls it after every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grader import grade
from tasks import ANSWER_FILE, load_task


def main() -> None:
    """Grade one folder and print the verdict and the reward."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    task = load_task()
    result = grade(task, args.folder)
    verdict = "correct" if result.correct else f"wrong: {result.reason}"
    print(f"question   {task.question}")
    print(f"{ANSWER_FILE} {result.submitted or '(none)'}")
    print(f"verdict    {verdict}")
    print(f"reward     {result.reward}")


if __name__ == "__main__":
    main()
