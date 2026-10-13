"""How this chapter writes paths on screen.

Only ``capstone.sh`` writes under ``COURSE_HOME`` (the fresh clone, the
servers' logs, the installers' private settings); ``read_results.py`` only
reads, and prints every path the short way. Only the standard library is
needed.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def short_path(path: Path) -> str:
    """Write a path the way a reader wants to see it.

    A path inside this chapter's folder is written from the folder
    (``data/project_run``), one in the home folder with ``~``.

    Args:
        path: Any path.

    Returns:
        The path as text.
    """
    for base, prefix, itself in ((HERE, "", "."), (Path.home(), "~/", "~")):
        try:
            relative = path.resolve().relative_to(base)
        except ValueError:  # not inside this folder
            continue
        return itself if relative == Path() else prefix + str(relative)
    return str(path)
