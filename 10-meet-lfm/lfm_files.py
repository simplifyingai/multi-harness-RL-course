"""Where LFM2.5-2.6B's files come from, and where they live on this laptop.

Two repositories on the Hugging Face Hub hold the model this course trains:

- ``LiquidAI/LFM2.5-2.6B``: Liquid AI's original release, every number stored
  in bfloat16 (5.4 GB). ``download.py`` fetches it.
- ``LiquidAI/LFM2.5-2.6B-MLX-4bit``: Liquid's 4-bit export for MLX (1.6 GB),
  the copy the project serves and trains. ``download.py --fast`` fetches it.

Both go into ``$COURSE_HOME/models``, which every chapter from here on shares,
so the 5.4 GB is downloaded once. Each repository is pinned to one commit and
every file to its size and SHA-256 fingerprint, so the files can never change
under the course. This module needs only the standard library: it says where
the files belong and whether they are all there, and never uses the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "10-meet-lfm"
# Shared by every chapter that needs the model, so it is downloaded once.
MODELS_FOLDER = "models"
GIGABYTE = 1_000_000_000
MISSING_NAMES_SHOWN = 2  # files named when some are missing; the rest counted

# Names the scripts and report.py share, so that a printed label can never
# drift from the value a script computed with.
LAYER_CONV = "conv"  # config.json's layer_types
LAYER_ATTENTION = "full_attention"
PART_FEED_FORWARD = "feed-forward"
PART_CONV = "convolution"
PART_ATTENTION = "attention"
PART_EMBEDDING = "embedding table"
PART_HEAD = "output head"
PART_NORMS = "norms"
PART_ORDER = (
    PART_FEED_FORWARD,
    PART_CONV,
    PART_ATTENTION,
    PART_EMBEDDING,
    PART_HEAD,
    PART_NORMS,
)
BF16_BYTES = 2  # bytes per bfloat16 number
FLOAT32_BYTES = 4  # bytes per float32 number
FINISH_STOP = "stop"  # mlx-lm: the model wrote its end-of-turn token
FINISH_LENGTH = "length"  # mlx-lm: the length cap cut the reply off
DOWNLOAD_COMMAND = "uv run download.py"
FAST_FLAG = "--fast"


@dataclass(frozen=True)
class HubFile:
    """One file of a model repository on the Hub.

    Attributes:
        name: The file name, the same on the Hub and on disk.
        size: Its size in bytes.
        sha256: Its SHA-256 fingerprint, as 64 hex characters.
        role: What the file is for, in plain words.
    """

    name: str
    size: int
    sha256: str
    role: str


@dataclass(frozen=True)
class ModelRepo:
    """A model repository on the Hub, pinned to one commit.

    Attributes:
        repo_id: ``owner/name`` on the Hub.
        revision: The commit every file is fetched from.
        date: When that commit was made, for the reader.
        folder: The folder name under ``$COURSE_HOME/models``.
        label: How the numbers are stored, in a word or two.
        files: Every file the repository holds at that commit.
    """

    repo_id: str
    revision: str
    date: str
    folder: str
    label: str
    files: tuple[HubFile, ...]

    def total_bytes(self) -> int:
        """Add up the sizes of all its files.

        Returns:
            The total in bytes.
        """
        return sum(item.size for item in self.files)


# The roles of the files both repositories share.
ROLE_INDEX = "which tensor is in which file"
ROLE_CONFIG = "the architecture: layers, sizes, positions"
ROLE_GENERATION = "the maker's default sampling settings"
ROLE_TOKENIZER = "the vocabulary (chapter 9)"
ROLE_TOKENIZER_CONFIG = "the special tokens (chapter 9)"
ROLE_TEMPLATE = "the chat template (chapter 9)"
ROLE_LICENSE = "the terms of use"
ROLE_CARD = "the model card: the maker's description"
ROLE_GITATTRIBUTES = "a git setting for the large files"

# The tokenizer files are the same bytes in both repositories (chapter 9).
TOKENIZER_JSON = HubFile(
    "tokenizer.json",
    17_905_598,
    "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41",
    ROLE_TOKENIZER,
)
TOKENIZER_CONFIG = HubFile(
    "tokenizer_config.json",
    363,
    "11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09",
    ROLE_TOKENIZER_CONFIG,
)
LICENSE = HubFile(
    "LICENSE",
    10_574,
    "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
    ROLE_LICENSE,
)
GITATTRIBUTES = HubFile(
    ".gitattributes",
    1_570,
    "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ROLE_GITATTRIBUTES,
)

# Liquid AI's original release: the weights in two safetensors files.
BF16 = ModelRepo(
    repo_id="LiquidAI/LFM2.5-2.6B",
    revision="654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    date="2026-08-24",
    folder="LFM2.5-2.6B",
    label="bfloat16",
    files=(
        HubFile(
            "model-00001-of-00002.safetensors",
            5_329_406_264,
            "05be188e6570a7642b4690545ab22ebf4d0a6068775dd776a452eebf738854cc",
            "the weights, part 1",
        ),
        HubFile(
            "model-00002-of-00002.safetensors",
            65_021_192,
            "532f803a9e6f127adc124d977503b438d8cc9c18f96e0862bf6835e38867bca8",
            "the weights, part 2",
        ),
        HubFile(
            "model.safetensors.index.json",
            21_425,
            "1be35d0d99dc56f68e6c5b306f7d4d99b6da97d1ab41ee753832bf1495655c4a",
            ROLE_INDEX,
        ),
        HubFile(
            "config.json",
            1_467,
            "480f63fa8e1efa534ae8b92774b3b53b8d6812d62a726e9ecfc866933662f273",
            ROLE_CONFIG,
        ),
        HubFile(
            "generation_config.json",
            327,
            "7366b93f26f8830e7a94441a3d2b9f344ceb9e1e31a808e0656df40720075874",
            ROLE_GENERATION,
        ),
        TOKENIZER_JSON,
        TOKENIZER_CONFIG,
        HubFile(
            "chat_template.jinja",
            5_443,
            "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef",
            ROLE_TEMPLATE,
        ),
        LICENSE,
        HubFile(
            "README.md",
            18_337,
            "003ceb63ec63a5b06b17f5e7b7a7d907695f037ad18720cabbcf8d8d68813f24",
            ROLE_CARD,
        ),
        GITATTRIBUTES,
    ),
)

# Liquid's 4-bit export for MLX, the copy the project runs (scripts/setup.sh).
FOUR_BIT = ModelRepo(
    repo_id="LiquidAI/LFM2.5-2.6B-MLX-4bit",
    revision="04efa23776ce61ec34ec95ec34c859854c89542b",
    date="2026-08-12",
    folder="LFM2.5-2.6B-MLX-4bit",
    label="4-bit",
    files=(
        HubFile(
            "model.safetensors",
            1_583_152_892,
            "ef350b75815752f8bff8c064c9fc8bd8b5709577d72943436852013d2bcd4691",
            "the weights, packed 4 bits each",
        ),
        HubFile(
            "model.safetensors.index.json",
            41_983,
            "620f8efaf53e826757324852e999f6ba9a5d974e193589470a70951fd17a1543",
            ROLE_INDEX,
        ),
        HubFile(
            "config.json",
            2_202,
            "6ab5ad71687f6ca8d3811b9290ac06291f78e1e0f4f3ec504e6644032d58342b",
            ROLE_CONFIG + ", bits",
        ),
        HubFile(
            "generation_config.json",
            306,
            "e08c9a83f4bfe64468c9671a2b69b60549728e2a9481387f3555471f64e958bc",
            ROLE_GENERATION,
        ),
        TOKENIZER_JSON,
        TOKENIZER_CONFIG,
        HubFile(
            "chat_template.jinja",
            5_496,
            "aadd425edb1d42fe9288e5de3624c21c1c03099129b9cfea5978f1bc67222c0a",
            ROLE_TEMPLATE,
        ),
        LICENSE,
        HubFile(
            "README.md",
            2_648,
            "0907976d22f77d4178e10d04002403a94a22ad965ad8431f754fb28a14885f6e",
            ROLE_CARD,
        ),
        GITATTRIBUTES,
    ),
)


def pick_repo(fast: bool) -> ModelRepo:
    """Choose the repository a script works on.

    Args:
        fast: True for the 4-bit export (1.6 GB), False for the bfloat16
            original (5.4 GB).

    Returns:
        The repository.
    """
    return FOUR_BIT if fast else BF16


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def models_home() -> Path:
    """Return the folder every chapter shares for downloaded models.

    Returns:
        ``$COURSE_HOME/models``.
    """
    return course_home() / MODELS_FOLDER


def chapter_home() -> Path:
    """Return this chapter's own output folder.

    Returns:
        ``$COURSE_HOME/10-meet-lfm``.
    """
    return course_home() / CHAPTER


def model_folder(repo: ModelRepo) -> Path:
    """Return where a repository's files live on disk.

    Args:
        repo: The repository.

    Returns:
        ``$COURSE_HOME/models/<folder>``.
    """
    return models_home() / repo.folder


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


def gigabytes(size: int) -> str:
    """Write a size in gigabytes (10**9 bytes), as the Hub shows sizes.

    Args:
        size: A size in bytes.

    Returns:
        Such as ``"5.41 GB"``.
    """
    return f"{size / GIGABYTE:.2f} GB"


def missing_files(repo: ModelRepo, folder: Path) -> list[HubFile]:
    """List the repository's files that are not in the folder at full size.

    Only sizes are checked, which is instant; the SHA-256 fingerprints were
    checked once, right after the download.

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
        A few lines naming the command to run.
    """
    fast_command = f"{DOWNLOAD_COMMAND} {FAST_FLAG}"
    command = fast_command if repo is FOUR_BIT else DOWNLOAD_COMMAND
    shown = missing[:MISSING_NAMES_SHOWN]
    names = ", ".join(item.name for item in shown)
    more = f" and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""
    lines = [
        f"{repo.repo_id} ({repo.label}) is not fully downloaded yet:",
        f"  missing or incomplete in {short_path(model_folder(repo))}: {names}{more}",
        f"run `{command}` first ({gigabytes(repo.total_bytes())}).",
    ]
    if repo is BF16:
        lines.append(
            f"Or add {FAST_FLAG} to use the 4-bit export instead "
            f"(`{fast_command}`, {gigabytes(FOUR_BIT.total_bytes())})."
        )
    return "\n".join(lines)


def require_model(fast: bool) -> tuple[ModelRepo, Path]:
    """Find a downloaded model, or stop with the command that downloads it.

    Args:
        fast: True for the 4-bit export, False for the bfloat16 original.

    Returns:
        The repository and the folder that holds its files.

    Raises:
        SystemExit: With an explanation, if any file is missing; the script
            stops there and prints it.
    """
    repo = pick_repo(fast)
    folder = model_folder(repo)
    missing = missing_files(repo, folder)
    if missing:
        raise SystemExit(not_downloaded_message(repo, missing))
    return repo, folder
