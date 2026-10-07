"""Import one chapter's modules for a test file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Chapters may each ship a module with the same name (tasks.py, grader.py).
# Each test file imports its own chapter through use_chapter(), which puts
# that folder first and forgets modules another chapter already loaded.


def use_chapter(name: str) -> Path:
    """Make one chapter folder the source of its modules for a test file.

    Args:
        name: The folder name, e.g. ``"01-the-job"``.

    Returns:
        The chapter folder.

    Raises:
        FileNotFoundError: If there is no such chapter folder.
    """
    folder = ROOT / name
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    for module_file in folder.glob("*.py"):
        sys.modules.pop(module_file.stem, None)
    sys.path.insert(0, str(folder))
    return folder
