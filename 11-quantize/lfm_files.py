"""Where LFM2.5-2.6B's files come from, and where every build of it lives.

A copy of chapter 10's ``lfm_files.py``, plus the builds this chapter makes.
Four builds of the same model appear in this chapter:

- ``LiquidAI/LFM2.5-2.6B``: Liquid AI's original release in bfloat16 (5.4 GB),
  downloaded by chapter 10 into ``$COURSE_HOME/models``. ``convert.py`` reads
  it and never changes it.
- **ours**: the 4-bit build ``convert.py`` makes from it, in
  ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``. Every later chapter loads
  this one, falling back to Liquid's export when it is missing.
- **all 4-bit**: the same build with the embedding table at 4 bits too, in
  ``$COURSE_HOME/11-quantize``, for ``compare.py`` alone.
- ``LiquidAI/LFM2.5-2.6B-MLX-4bit``: Liquid's own 4-bit export (1.6 GB), also
  downloaded by chapter 10, the copy the project runs.

This module needs only the standard library and never uses the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "11-quantize"
# Shared by every chapter that needs the model, so it is downloaded once.
MODELS_FOLDER = "models"
# Our 4-bit build: shared, because every later chapter loads it.
OURS_FOLDER = "lfm2.5-2.6b-4bit-ours"
# The all-4-bit variant: this chapter's alone, so it lives in its folder.
ALL_FOUR_BIT_FOLDER = "lfm2.5-2.6b-4bit-all"
# A build is complete when these files are there (convert.py writes the
# folder under another name and renames it last, so a half build never is).
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
CARD_FILE = "README.md"  # the model card: each maker writes its own
SAFETENSORS_GLOB = "*.safetensors"
QUANTIZATION_KEY = "quantization"  # config.json's block saying how it is packed
BUILD_FILES = (WEIGHTS_FILE, CONFIG_FILE, "tokenizer.json")
GIGABYTE = 1_000_000_000  # 10**9 bytes: how the Hub, Finder and MLX count
MILLISECONDS = 1_000  # in a second
BITS_PER_BYTE = 8
GIBIBYTE = 1 << 30  # 2**30 bytes: how memory chips are sized
MISSING_NAMES_SHOWN = 2  # files named when some are missing; the rest counted
# The four builds' short ids, shared by compare.py and report.py.
KEY_BF16 = "bf16"
KEY_OURS = "ours"
KEY_ALL_FOUR = "all4"
KEY_LIQUID = "liquid"
FINISH_STOP = "stop"  # mlx-lm: the model wrote its end-of-turn token
FINISH_LENGTH = "length"  # mlx-lm: the length cap cut the reply off
CHAPTER_10 = "10-meet-lfm"
DOWNLOAD_COMMAND = "uv run download.py"
FAST_FLAG = "--fast"
CONVERT_COMMAND = "uv run convert.py"
ALL_FOUR_BIT_FLAG = "--all-4bit"


@dataclass(frozen=True)
class HubFile:
    """One file of a model repository on the Hub.

    Attributes:
        name: The file name, the same on the Hub and on disk.
        size: Its size in bytes.
        sha256: Its SHA-256 fingerprint, as 64 hex characters.
    """

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelRepo:
    """A model repository on the Hub, pinned to one commit.

    Attributes:
        repo_id: ``owner/name`` on the Hub.
        revision: The commit every file is fetched from.
        folder: The folder name under ``$COURSE_HOME/models``.
        label: How the numbers are stored, in a word or two.
        files: Every file the repository holds at that commit.
    """

    repo_id: str
    revision: str
    folder: str
    label: str
    files: tuple[HubFile, ...]

    def total_bytes(self) -> int:
        """Add up the sizes of all its files.

        Returns:
            The total in bytes.
        """
        return sum(item.size for item in self.files)


# The files both repositories share, byte for byte (chapter 9).
TOKENIZER_JSON = HubFile(
    "tokenizer.json",
    17_905_598,
    "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41",
)
TOKENIZER_CONFIG = HubFile(
    "tokenizer_config.json",
    363,
    "11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09",
)
LICENSE = HubFile(
    "LICENSE",
    10_574,
    "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
)
GITATTRIBUTES = HubFile(
    ".gitattributes",
    1_570,
    "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
)

# Liquid AI's original release, as chapter 10 pinned it.
BF16 = ModelRepo(
    repo_id="LiquidAI/LFM2.5-2.6B",
    revision="654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    folder="LFM2.5-2.6B",
    label="bfloat16",
    files=(
        HubFile(
            "model-00001-of-00002.safetensors",
            5_329_406_264,
            "05be188e6570a7642b4690545ab22ebf4d0a6068775dd776a452eebf738854cc",
        ),
        HubFile(
            "model-00002-of-00002.safetensors",
            65_021_192,
            "532f803a9e6f127adc124d977503b438d8cc9c18f96e0862bf6835e38867bca8",
        ),
        HubFile(
            "model.safetensors.index.json",
            21_425,
            "1be35d0d99dc56f68e6c5b306f7d4d99b6da97d1ab41ee753832bf1495655c4a",
        ),
        HubFile(
            "config.json",
            1_467,
            "480f63fa8e1efa534ae8b92774b3b53b8d6812d62a726e9ecfc866933662f273",
        ),
        HubFile(
            "generation_config.json",
            327,
            "7366b93f26f8830e7a94441a3d2b9f344ceb9e1e31a808e0656df40720075874",
        ),
        TOKENIZER_JSON,
        TOKENIZER_CONFIG,
        HubFile(
            "chat_template.jinja",
            5_443,
            "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef",
        ),
        LICENSE,
        HubFile(
            "README.md",
            18_337,
            "003ceb63ec63a5b06b17f5e7b7a7d907695f037ad18720cabbcf8d8d68813f24",
        ),
        GITATTRIBUTES,
    ),
)

# Liquid's 4-bit export for MLX, as chapter 10 pinned it.
FOUR_BIT = ModelRepo(
    repo_id="LiquidAI/LFM2.5-2.6B-MLX-4bit",
    revision="04efa23776ce61ec34ec95ec34c859854c89542b",
    folder="LFM2.5-2.6B-MLX-4bit",
    label="4-bit",
    files=(
        HubFile(
            "model.safetensors",
            1_583_152_892,
            "ef350b75815752f8bff8c064c9fc8bd8b5709577d72943436852013d2bcd4691",
        ),
        HubFile(
            "model.safetensors.index.json",
            41_983,
            "620f8efaf53e826757324852e999f6ba9a5d974e193589470a70951fd17a1543",
        ),
        HubFile(
            "config.json",
            2_202,
            "6ab5ad71687f6ca8d3811b9290ac06291f78e1e0f4f3ec504e6644032d58342b",
        ),
        HubFile(
            "generation_config.json",
            306,
            "e08c9a83f4bfe64468c9671a2b69b60549728e2a9481387f3555471f64e958bc",
        ),
        TOKENIZER_JSON,
        TOKENIZER_CONFIG,
        HubFile(
            "chat_template.jinja",
            5_496,
            "aadd425edb1d42fe9288e5de3624c21c1c03099129b9cfea5978f1bc67222c0a",
        ),
        LICENSE,
        HubFile(
            "README.md",
            2_648,
            "0907976d22f77d4178e10d04002403a94a22ad965ad8431f754fb28a14885f6e",
        ),
        GITATTRIBUTES,
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
        ``$COURSE_HOME/11-quantize``.
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
    """Return where our 4-bit build lives, the model every later chapter uses.

    Returns:
        ``$COURSE_HOME/models/lfm2.5-2.6b-4bit-ours``.
    """
    return models_home() / OURS_FOLDER


def all_four_bit_folder() -> Path:
    """Return where the all-4-bit variant lives (this chapter's alone).

    Returns:
        ``$COURSE_HOME/11-quantize/lfm2.5-2.6b-4bit-all``.
    """
    return chapter_home() / ALL_FOUR_BIT_FOLDER


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


def gibibytes(size: int | float) -> str:
    """Write a size in gibibytes (2**30 bytes), as memory is sized.

    Args:
        size: A size in bytes.

    Returns:
        Such as ``"1.47 GiB"``.
    """
    return f"{size / GIBIBYTE:.2f} GiB"


def missing_files(repo: ModelRepo, folder: Path) -> list[HubFile]:
    """List the repository's files that are not in the folder at full size.

    Only sizes are checked, which is instant; chapter 10's download.py
    checked the SHA-256 fingerprints right after downloading.

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


def not_downloaded_message(repo: ModelRepo, missing: list[HubFile]) -> str:
    """Explain how to get a repository that is not (fully) on disk.

    Args:
        repo: The repository a script needs.
        missing: Its files that are absent or the wrong size.

    Returns:
        A few lines naming chapter 10's command to run.
    """
    command = DOWNLOAD_COMMAND
    if repo is FOUR_BIT:
        command = f"{DOWNLOAD_COMMAND} {FAST_FLAG}"
    shown = missing[:MISSING_NAMES_SHOWN]
    names = ", ".join(item.name for item in shown)
    more = f" and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""
    return "\n".join(
        [
            f"{repo.repo_id} ({repo.label}) is not fully downloaded yet:",
            f"  missing or incomplete in {short_path(model_folder(repo))}: "
            f"{names}{more}",
            f"run `{command}` in {CHAPTER_10} first ({gigabytes(repo.total_bytes())}).",
        ]
    )


def require_repo(repo: ModelRepo) -> Path:
    """Find a downloaded repository, or stop with the command that downloads it.

    Args:
        repo: The repository.

    Returns:
        The folder that holds its files.

    Raises:
        SystemExit: With an explanation, if any file is missing.
    """
    folder = model_folder(repo)
    missing = missing_files(repo, folder)
    if missing:
        raise SystemExit(not_downloaded_message(repo, missing))
    return folder


def build_ready(folder: Path) -> bool:
    """Say whether a build made by convert.py is complete in a folder.

    Args:
        folder: The build's folder.

    Returns:
        True when every file a build needs is there.
    """
    return all((folder / name).is_file() for name in BUILD_FILES)


def model_for_later_chapters() -> tuple[Path, str]:
    """Pick the 4-bit model later chapters load: ours, else Liquid's export.

    Returns:
        The folder, and a few words saying which build it is.

    Raises:
        SystemExit: If neither is on disk, naming both commands.
    """
    if build_ready(ours_folder()):
        return ours_folder(), "ours, built by convert.py"
    if not missing_files(FOUR_BIT, model_folder(FOUR_BIT)):
        return model_folder(FOUR_BIT), (
            f"Liquid's export: ours is not built yet (`{CONVERT_COMMAND}`)"
        )
    raise SystemExit(
        f"no 4-bit model yet: run `{CONVERT_COMMAND}` here (it needs chapter 10's "
        f"bfloat16 download), or `{DOWNLOAD_COMMAND} {FAST_FLAG}` in {CHAPTER_10} "
        f"for Liquid's export ({gigabytes(FOUR_BIT.total_bytes())})."
    )
