"""Where the 4-bit LFM2.5-2.6B lives, where this chapter writes, and its settings.

A copy of chapter 13's ``lfm_files.py``, cut down to what this chapter needs.
Two builds of the same 4-bit model can be on disk, and they are the same
bytes (chapter 11 checked it, tensor by tensor):

- **ours**: the build chapter 11's ``convert.py`` made, in
  ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``. Every later chapter loads it.
- ``LiquidAI/LFM2.5-2.6B-MLX-4bit``: Liquid's own export, downloaded by
  chapter 10 (``download.py --fast``), the copy the project runs. It is the
  fallback when ours is not built yet, and what ``--fast`` uses.

This module needs only the standard library and never uses the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "14-sft-real-model"
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
# The project's numbers. Its LoRA settings (mhrl/config.py: lora_rank,
# lora_scale, lora_layers), its output cap (max_completion_tokens) and its
# SFT settings (mhrl/cli.py: CliDefaults.SFT_EPOCHS, SFT_LR, SFT_BATCH).
LORA_RANK = 16
LORA_SCALE = 2.0  # mlx-lm's scale: what other libraries write as alpha / rank
LORA_LAYERS = 12  # the last 12 of the model's 30 layers
OUTPUT_CAP = 1_024  # the same cap in the project's training and evaluation
SFT_EPOCHS = 1
SFT_LEARNING_RATE = 5e-5  # "1e-4 over-fit the dry run (loss 0.10 by step 4 of 6)"
SFT_BATCH = 8  # rows per optimizer step
# The files this chapter writes under $COURSE_HOME/14-sft-real-model.
ROWS_NAME = "rows_{split}.jsonl"
ADAPTER_STEM = "sft"
ADAPTER_SUFFIX = ".safetensors"
METRICS_SUFFIX = ".metrics.jsonl"
EVAL_NAME = "eval_{label}.jsonl"
WORK_FOLDER = "work"  # one folder per question, holding its answer.txt
DEFAULT_REASON = "steps"  # rows.REASON_STEPS: the reason written from the SQL
BASE_LABEL = "base"


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
        ``$COURSE_HOME/14-sft-real-model``.
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


def rows_path(split: str, style: str = DEFAULT_REASON) -> Path:
    """Return where make_rows.py writes one split's rows.

    Args:
        split: ``"train"`` or ``"test"``.
        style: The kind of reason the replies start with.

    Returns:
        ``$COURSE_HOME/14-sft-real-model/rows_<split>.jsonl``, with
        ``_<style>`` before ``.jsonl`` for any style but the default.
    """
    suffix = "" if style == DEFAULT_REASON else f"_{style}"
    return chapter_home() / ROWS_NAME.format(split=split + suffix)


def adapter_name(
    lr: float, epochs: int, limit: int | None, style: str = DEFAULT_REASON
) -> str:
    """Name a training run's adapter file after what differs from the default.

    The project's settings give ``sft.safetensors``; any other run gets its
    own name, so trying ``--lr 1e-4`` never overwrites the main adapter.

    Args:
        lr: The learning rate.
        epochs: Passes over the rows.
        limit: How many training rows were used, or None for all of them.
        style: The kind of reason the rows' replies start with.

    Returns:
        Such as ``"sft.safetensors"`` or ``"sft_lr0.0001_epochs3.safetensors"``.
    """
    parts = [ADAPTER_STEM]
    if style != DEFAULT_REASON:
        parts.append(f"reason{style}")
    if lr != SFT_LEARNING_RATE:
        parts.append(f"lr{lr:g}")
    if epochs != SFT_EPOCHS:
        parts.append(f"epochs{epochs}")
    if limit is not None:
        parts.append(f"limit{limit}")
    return "_".join(parts) + ADAPTER_SUFFIX


def adapter_path(name: str) -> Path:
    """Find an adapter file: a bare name lives in this chapter's folder.

    Args:
        name: A file name such as ``sft.safetensors``, or a path.

    Returns:
        The path: under ``$COURSE_HOME/14-sft-real-model`` when ``name`` has
        no folder in it, else ``name`` itself.
    """
    path = Path(name).expanduser()
    return chapter_home() / path if path.parent == Path() else path


def metrics_path(adapter: Path) -> Path:
    """Return where a training run logs its steps, beside its adapter.

    Args:
        adapter: The adapter file.

    Returns:
        Such as ``sft.metrics.jsonl`` next to ``sft.safetensors``.
    """
    return adapter.with_name(adapter.stem + METRICS_SUFFIX)


def eval_label(adapter: Path | None) -> str:
    """Name an evaluation after the adapter it used.

    Args:
        adapter: The adapter file, or None for the base model.

    Returns:
        ``"base"``, or the adapter's file name without ``.safetensors``.
    """
    return BASE_LABEL if adapter is None else adapter.stem


def eval_path(label: str) -> Path:
    """Return where an evaluation's results are written.

    Args:
        label: From ``eval_label``.

    Returns:
        ``$COURSE_HOME/14-sft-real-model/eval_<label>.jsonl``.
    """
    return chapter_home() / EVAL_NAME.format(label=label)


def work_folder(label: str, task_id: str) -> Path:
    """Return the folder where one question's answer.txt is written.

    Args:
        label: The evaluation's label.
        task_id: The question.

    Returns:
        ``$COURSE_HOME/14-sft-real-model/work/<label>/<task_id>``.
    """
    return chapter_home() / WORK_FOLDER / label / task_id
