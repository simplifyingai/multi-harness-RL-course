"""Where the 4-bit LFM2.5-2.6B lives, where this chapter writes, and its limits.

A copy of chapter 15's ``lfm_files.py``, cut down to what this chapter needs.
Two builds of the same 4-bit model can be on disk, and they are the same
bytes (chapter 11 checked it, tensor by tensor):

- **ours**: the build chapter 11's ``convert.py`` made, in
  ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``. Every later chapter loads it.
- ``LiquidAI/LFM2.5-2.6B-MLX-4bit``: Liquid's own export, downloaded by
  chapter 10 (``download.py --fast``), the copy the project runs. It is the
  fallback when ours is not built yet, and what ``--fast`` uses.

The harness's limits are the project's (``mhrl/config.py``): one reply may
have at most 1,024 tokens, and one attempt at most 8 model calls. This module
needs only the standard library and never uses the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "17-tiny-harness"
# Shared by every chapter that needs the model, so it is downloaded once.
MODELS_FOLDER = "models"
# Our 4-bit build, made by chapter 11.
OURS_FOLDER = "lfm2.5-2.6b-4bit-ours"
# A build is complete when these files are there (chapter 11's convert.py
# writes the folder under another name and renames it last).
BUILD_FILES = ("model.safetensors", "config.json", "tokenizer.json")
GIGABYTE = 1_000_000_000  # 10**9 bytes: how the Hub, Finder and MLX count
MISSING_NAMES_SHOWN = 2  # files named when some are missing; the rest counted
CHAPTER_10 = "10-meet-lfm"
CHAPTER_11 = "11-quantize"
DOWNLOAD_COMMAND = "uv run download.py --fast"
CONVERT_COMMAND = "uv run convert.py"
# The project's limits: mhrl/config.py max_completion_tokens, max_model_calls
# and eval_temperature (0: always the likeliest token, as its evaluation does).
OUTPUT_CAP = 1_024
MAX_CALLS = 8
# Our harness's own rule, borrowed from mini-swe-agent (its
# max_consecutive_format_errors is 3): give up after this many replies in a
# row that hold no readable tool call and no answer.
MAX_UNREADABLE = 3
EVAL_TEMPERATURE = 0.0
SAMPLE_SEED = 0  # MLX's random seed, used only with --temperature above 0
# The files this chapter writes under $COURSE_HOME/17-tiny-harness.
WORK_FOLDER = "work"  # one folder per attempt: data.db and answer.txt
ROLLOUTS_FOLDER = "rollouts"  # one JSON file per attempt: every call
ROLLOUT_SUFFIX = ".json"


@dataclass(frozen=True)
class HubFile:
    """One file of a model repository on the Hub.

    Attributes:
        name: The file name, the same on the Hub and on disk.
        size: Its size in bytes.
    """

    name: str
    size: int


@dataclass(frozen=True)
class ModelRepo:
    """A model repository on the Hub, pinned to one commit.

    Attributes:
        repo_id: ``owner/name`` on the Hub.
        revision: The commit chapter 10 fetched every file from.
        folder: The folder name under ``$COURSE_HOME/models``.
        files: Every file the repository holds at that commit.
    """

    repo_id: str
    revision: str
    folder: str
    files: tuple[HubFile, ...]

    def total_bytes(self) -> int:
        """Add up the sizes of all its files.

        Returns:
            The total in bytes.
        """
        return sum(item.size for item in self.files)


# Liquid's 4-bit export for MLX, as chapter 10 pinned it (sizes only: chapter
# 10's download.py checked the SHA-256 fingerprints right after downloading).
FOUR_BIT = ModelRepo(
    repo_id="LiquidAI/LFM2.5-2.6B-MLX-4bit",
    revision="04efa23776ce61ec34ec95ec34c859854c89542b",
    folder="LFM2.5-2.6B-MLX-4bit",
    files=(
        HubFile("model.safetensors", 1_583_152_892),
        HubFile("model.safetensors.index.json", 41_983),
        HubFile("config.json", 2_202),
        HubFile("generation_config.json", 306),
        HubFile("tokenizer.json", 17_905_598),
        HubFile("tokenizer_config.json", 363),
        HubFile("chat_template.jinja", 5_496),
        HubFile("LICENSE", 10_574),
        HubFile("README.md", 2_648),
        HubFile(".gitattributes", 1_570),
    ),
)


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def models_home() -> Path:
    """Return the folder every chapter shares for models.

    Returns:
        ``$COURSE_HOME/models``.
    """
    return course_home() / MODELS_FOLDER


def chapter_home() -> Path:
    """Return this chapter's own output folder.

    Returns:
        ``$COURSE_HOME/17-tiny-harness``.
    """
    return course_home() / CHAPTER


def model_folder(repo: ModelRepo) -> Path:
    """Return where a downloaded repository's files live.

    Args:
        repo: The repository.

    Returns:
        ``$COURSE_HOME/models/<folder>``.
    """
    return models_home() / repo.folder


def ours_folder() -> Path:
    """Return where chapter 11's 4-bit build lives.

    Returns:
        ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``.
    """
    return models_home() / OURS_FOLDER


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


def gigabytes(size: int | float) -> str:
    """Write a size in gigabytes (10**9 bytes), as the Hub and MLX count.

    Args:
        size: A size in bytes.

    Returns:
        Such as ``"1.58 GB"``.
    """
    return f"{size / GIGABYTE:.2f} GB"


def missing_files(repo: ModelRepo, folder: Path) -> list[HubFile]:
    """List the repository's files that are not in the folder at full size.

    Args:
        repo: The repository.
        folder: Where its files should be.

    Returns:
        The files that are absent or the wrong size.
    """
    missing = []
    for item in repo.files:
        path = folder / item.name
        if not path.is_file() or path.stat().st_size != item.size:
            missing.append(item)
    return missing


def require_liquid_export() -> Path:
    """Find Liquid's 4-bit export, or stop with the command that downloads it.

    Returns:
        The folder that holds its files.

    Raises:
        SystemExit: With an explanation, if any file is missing.
    """
    folder = model_folder(FOUR_BIT)
    missing = missing_files(FOUR_BIT, folder)
    if not missing:
        return folder
    shown = missing[:MISSING_NAMES_SHOWN]
    names = ", ".join(item.name for item in shown)
    more = f" and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""
    raise SystemExit(
        "\n".join(
            [
                f"{FOUR_BIT.repo_id} is not fully downloaded yet:",
                f"  missing or incomplete in {short_path(folder)}: {names}{more}",
                f"run `{DOWNLOAD_COMMAND}` in {CHAPTER_10} first "
                f"({gigabytes(FOUR_BIT.total_bytes())}).",
            ]
        )
    )


def build_ready(folder: Path) -> bool:
    """Say whether a build made by chapter 11's convert.py is complete.

    Args:
        folder: The build's folder.

    Returns:
        True when every file a build needs is there.
    """
    return all((folder / name).is_file() for name in BUILD_FILES)


def pick_model(fast: bool) -> tuple[Path, str]:
    """Pick the 4-bit model: ours, else Liquid's export (always with --fast).

    Args:
        fast: Use Liquid's export even when ours is built.

    Returns:
        The folder, and a few words saying which build it is.

    Raises:
        SystemExit: If the build asked for is not on disk, naming the command
            that makes it.
    """
    if fast:
        return require_liquid_export(), "Liquid's export (--fast)"
    if build_ready(ours_folder()):
        return ours_folder(), "ours, built by chapter 11's convert.py"
    if not missing_files(FOUR_BIT, model_folder(FOUR_BIT)):
        return model_folder(FOUR_BIT), (
            f"Liquid's export: ours is not built yet (`{CONVERT_COMMAND}` in "
            f"{CHAPTER_11})"
        )
    raise SystemExit(
        f"no 4-bit model yet: run `{CONVERT_COMMAND}` in {CHAPTER_11}, or "
        f"`{DOWNLOAD_COMMAND}` in {CHAPTER_10} for Liquid's export "
        f"({gigabytes(FOUR_BIT.total_bytes())})."
    )


def work_folder(label: str, task_id: str) -> Path:
    """Return the folder one attempt works in: its data.db and answer.txt.

    Args:
        label: Which run, such as ``harness``, ``Bash`` or ``text``.
        task_id: The question.

    Returns:
        ``$COURSE_HOME/17-tiny-harness/work/<label>/<task_id>``.
    """
    return chapter_home() / WORK_FOLDER / label / task_id


def rollout_path(label: str, task_id: str) -> Path:
    """Return where one attempt's calls are saved, as JSON.

    Args:
        label: Which run, as for ``work_folder``.
        task_id: The question.

    Returns:
        ``$COURSE_HOME/17-tiny-harness/rollouts/<label>/<task_id>.json``.
    """
    return chapter_home() / ROLLOUTS_FOLDER / label / (task_id + ROLLOUT_SUFFIX)
