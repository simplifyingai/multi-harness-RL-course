"""Where this chapter keeps its files, and where the project checkout is.

Two folders matter here. ``COURSE_HOME`` (default ``~/.cache/mhrl-course``)
holds everything the chapter writes, under ``27-how-it-combines/``. The
**project** is a checkout of the multi-harness-rl repository this course
follows; ``map.py`` measures it and ``one_step.py`` runs it. It is found
from ``--project``, else ``MHRL_PROJECT``. Only the standard library is
needed.
"""

from __future__ import annotations

import os
from pathlib import Path

COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "27-how-it-combines"
PROJECT_ENV = "MHRL_PROJECT"
# A checkout is recognised by its package and its config module.
PROJECT_MARKER = Path("mhrl") / "config.py"
# The project's own interpreter: `uv sync` makes it, `scripts/serve.sh` uses it.
PROJECT_PYTHON = Path(".venv") / "bin" / "python"


class ProjectNotFoundError(Exception):
    """No checkout of the project was named, or the one named is not one."""


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def chapter_home() -> Path:
    """Return this chapter's own output folder.

    Returns:
        ``$COURSE_HOME/27-how-it-combines``.
    """
    return course_home() / CHAPTER


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


def tilde(text: str) -> str:
    """Write the home folder as ``~`` wherever it appears in a text.

    Args:
        text: Any text, such as a command line.

    Returns:
        The text, with the learner's home folder (and user name) hidden.
    """
    return text.replace(str(Path.home()), "~")


def named_project(flag: str | None) -> Path | None:
    """Return the checkout named by ``--project`` or ``MHRL_PROJECT``, if any.

    Args:
        flag: The ``--project`` value, or None.

    Returns:
        The folder, resolved, or None when neither names one.
    """
    value = flag or os.environ.get(PROJECT_ENV)
    return Path(value).expanduser().resolve() if value else None


def check_project(folder: Path) -> Path:
    """Refuse a folder that is not a checkout of the project.

    Args:
        folder: The folder named.

    Returns:
        The same folder.

    Raises:
        ProjectNotFoundError: If it does not hold ``mhrl/config.py``.
    """
    if not (folder / PROJECT_MARKER).is_file():
        raise ProjectNotFoundError(
            f"{short_path(folder)} is not a checkout of the multi-harness-rl "
            f"project (no {PROJECT_MARKER})"
        )
    return folder
