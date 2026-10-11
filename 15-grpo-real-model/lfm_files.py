"""Where the 4-bit LFM2.5-2.6B lives, where this chapter writes, and its settings.

A copy of chapter 14's ``lfm_files.py``, with this chapter's settings. Two
builds of the same 4-bit model can be on disk, and they are the same bytes
(chapter 11 checked it, tensor by tensor):

- **ours**: the build chapter 11's ``convert.py`` made, in
  ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``. Every later chapter loads it.
- ``LiquidAI/LFM2.5-2.6B-MLX-4bit``: Liquid's own export, downloaded by
  chapter 10 (``download.py --fast``), the copy the project runs. It is the
  fallback when ours is not built yet, and what ``--fast`` uses.

Training starts from chapter 14's SFT adapter, ``sft.safetensors`` in
``$COURSE_HOME/14-sft-real-model``. This module needs only the standard
library and never uses the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "15-grpo-real-model"
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
CHAPTER_14 = "14-sft-real-model"
DOWNLOAD_COMMAND = "uv run download.py --fast"
CONVERT_COMMAND = "uv run convert.py"
TRAIN_SFT_COMMAND = "uv run train_sft.py"
# Chapter 14's SFT adapter, trained with the project's settings: the start.
SFT_ADAPTER = "sft.safetensors"
# The project's numbers. Its LoRA settings (mhrl/config.py: lora_rank,
# lora_scale, lora_layers) and its output cap (max_completion_tokens).
LORA_RANK = 16
LORA_SCALE = 2.0  # mlx-lm's scale: what other libraries write as alpha / rank
LORA_LAYERS = 12  # the last 12 of the model's 30 layers
OUTPUT_CAP = 1_024  # the same cap in the project's training and evaluation
# Its GRPO settings: mhrl/cli.py (CliDefaults.GRPO_STEPS, GRPO_TASKS_PER_STEP,
# GRPO_GROUP, GRPO_LR) and mhrl/config.py (train_temperature, token_budget).
GRPO_STEPS = 20
QUESTIONS_PER_STEP = 2
GROUP_SIZE = 4  # replies sampled per question: one group
GRPO_LEARNING_RATE = 1e-5  # "4e-5 (no bias correction) ballooned replies by step 5"
ATTEMPT_1_LEARNING_RATE = 4e-5  # the project's first RL run, from that comment
TRAIN_TEMPERATURE = 1.0
TOKEN_BUDGET = 2_000  # the token bonus reaches zero at this many tokens
# The precision the sampler records log-probabilities in: float32 is the
# engine's (its _to_float32); bfloat16 is what its first RL run recorded.
FP32 = "fp32"
BF16 = "bf16"
ORDER_SEED = 0  # train_grpo's seed: the training questions' shuffled order
SAMPLE_SEED = 0  # MLX's random seed for the draws
# The files this chapter writes under $COURSE_HOME/15-grpo-real-model.
ADAPTER_STEM = "grpo"
ADAPTER_SUFFIX = ".safetensors"
METRICS_SUFFIX = ".metrics.jsonl"
ROLLOUTS_SUFFIX = ".rollouts.jsonl"
EVAL_NAME = "eval_{label}.jsonl"
WORK_FOLDER = "work"  # one folder per reply, holding its answer.txt
BASE_LABEL = "base"
SFT_LABEL = "sft"


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
        ``$COURSE_HOME/15-grpo-real-model``.
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


def chapter_14_home() -> Path:
    """Return chapter 14's output folder, where its SFT adapters are.

    Returns:
        ``$COURSE_HOME/14-sft-real-model``.
    """
    return course_home() / CHAPTER_14


def init_path(name: str) -> Path:
    """Find the adapter training starts from: a bare name is chapter 14's.

    Args:
        name: A file name such as ``sft.safetensors``, or a path.

    Returns:
        The path: under ``$COURSE_HOME/14-sft-real-model`` when ``name`` has
        no folder in it, else ``name`` itself.
    """
    path = Path(name).expanduser()
    return chapter_14_home() / path if path.parent == Path() else path


def require_init(path: Path) -> Path:
    """Stop with the command that makes the starting adapter, if it is missing.

    Args:
        path: The adapter file.

    Returns:
        The same path, when the file is there.

    Raises:
        SystemExit: If there is no such file.
    """
    if path.is_file():
        return path
    raise SystemExit(
        f"no adapter at {short_path(path)}: run `{TRAIN_SFT_COMMAND}` in "
        f"{CHAPTER_14} first (about 2 minutes), or name one with --init"
    )


def adapter_path(name: str) -> Path:
    """Find a trained adapter: a bare name lives in this chapter's folder.

    Args:
        name: A file name such as ``grpo.safetensors``, or a path.

    Returns:
        The path: under ``$COURSE_HOME/15-grpo-real-model`` when ``name`` has
        no folder in it, else ``name`` itself.
    """
    path = Path(name).expanduser()
    return chapter_home() / path if path.parent == Path() else path


def run_name(parts: list[str]) -> str:
    """Name a training run's adapter after what differs from the default.

    The project's settings give ``grpo.safetensors``; any other run gets its
    own name, so trying ``--top-k 50`` never overwrites the main adapter.

    Args:
        parts: Short words for each setting that differs, in order, such as
            ``["topk50", "steps4"]``.

    Returns:
        Such as ``"grpo.safetensors"`` or ``"grpo_topk50_steps4.safetensors"``.
    """
    return "_".join([ADAPTER_STEM, *parts]) + ADAPTER_SUFFIX


def metrics_path(adapter: Path) -> Path:
    """Return where a training run logs its steps, beside its adapter.

    Args:
        adapter: The adapter file.

    Returns:
        Such as ``grpo.metrics.jsonl`` next to ``grpo.safetensors``.
    """
    return adapter.with_name(adapter.stem + METRICS_SUFFIX)


def rollouts_path(adapter: Path) -> Path:
    """Return where a training run keeps every sampled reply, beside its adapter.

    Args:
        adapter: The adapter file.

    Returns:
        Such as ``grpo.rollouts.jsonl`` next to ``grpo.safetensors``.
    """
    return adapter.with_name(adapter.stem + ROLLOUTS_SUFFIX)


def eval_path(label: str) -> Path:
    """Return where an evaluation's results are written.

    Args:
        label: ``base``, ``sft`` or an adapter's name without its suffix.

    Returns:
        ``$COURSE_HOME/15-grpo-real-model/eval_<label>.jsonl``.
    """
    return chapter_home() / EVAL_NAME.format(label=label)


def work_folder(*names: str) -> Path:
    """Return the folder where one reply's answer.txt is written.

    Args:
        *names: The folders below ``work``, such as an evaluation's label and
            a task id, or a training run, its step and a reply.

    Returns:
        ``$COURSE_HOME/15-grpo-real-model/work/<names...>``.
    """
    return chapter_home().joinpath(WORK_FOLDER, *names)
