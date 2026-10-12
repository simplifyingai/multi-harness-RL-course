# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fake secrets, to prove the box on files that do not matter first.

Every file here holds one line, ``decoy for chapter 19: not a real secret``.
``attacks.py`` reads them inside a box whose "home" is the decoy home, before
it goes near a real path; ``grade_safe.py`` links ``answer.txt`` to one of
them, so a careless grader can be caught reading a "secret" that is harmless.

``uv run decoys.py`` writes them and prints where they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tasks import DECOYS_DIR, TASK_FILE, TEXT_ENCODING, chapter_home, show

DECOY_TEXT = "decoy for chapter 19: not a real secret\n"
DECOY_HOME_DIR = "home"
DECOY_TASKS_DIR = "tasks"
# Fake secrets, relative to the decoy home, in the places the profile guards.
DECOY_FILES = {
    "ssh key": ".ssh/id_ed25519",
    "cloud credentials": ".aws/credentials",
    "Claude Code login": ".claude.json",
    "Claude Code login (file)": ".claude/.credentials.json",
    ".env file": "project/.env",
}
DOT_ENV_LABEL = ".env file"


@dataclass(frozen=True)
class Decoys:
    """Where the decoys are.

    Attributes:
        home: The decoy home folder (its secrets are listed in DECOY_FILES).
        tasks_dir: The decoy gold-answers folder.
    """

    home: Path
    tasks_dir: Path

    @property
    def gold_file(self) -> Path:
        """The decoy gold-answers file."""
        return self.tasks_dir / TASK_FILE.name

    @property
    def dot_env(self) -> Path:
        """The decoy ``.env`` file; under your real home if COURSE_HOME is."""
        return self.home / DECOY_FILES[DOT_ENV_LABEL]

    def files(self) -> dict[str, Path]:
        """List every decoy file with a label.

        Returns:
            Label to path, the gold-answers file last.
        """
        found = {label: self.home / path for label, path in DECOY_FILES.items()}
        found["gold answers"] = self.gold_file
        return found


def make_decoys() -> Decoys:
    """Write the decoy home and the decoy gold-answers folder.

    Returns:
        Where they are.

    Raises:
        OSError: If the files cannot be written.
    """
    root = chapter_home() / DECOYS_DIR
    decoys = Decoys(root / DECOY_HOME_DIR, root / DECOY_TASKS_DIR)
    for path in decoys.files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DECOY_TEXT, encoding=TEXT_ENCODING)
    return decoys


def main() -> None:
    """Write the decoys and list them."""
    for label, path in make_decoys().files().items():
        print(f"{label:<26} {show(str(path))}")


if __name__ == "__main__":
    main()
