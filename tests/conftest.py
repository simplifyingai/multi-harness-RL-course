"""Keep every test out of the real course cache."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def course_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point COURSE_HOME at a temporary folder for every test.

    Args:
        tmp_path: pytest's per-test folder.
        monkeypatch: pytest's environment patcher.

    Returns:
        The temporary course home.
    """
    home = tmp_path / "course-home"
    monkeypatch.setenv("COURSE_HOME", str(home))
    return home
