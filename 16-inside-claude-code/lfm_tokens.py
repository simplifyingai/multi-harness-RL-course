"""Count tokens the way LFM2.5-2.6B, the course's model, would cut the text.

Claude Code was written for Claude, whose tokenizer is not public, so the
chapter counts with the tokenizer of the model the course trains: the same
``tokenizer.json`` chapters 9 and 10 fetched. The first copy found in
``COURSE_HOME`` is used if its SHA-256 fingerprint matches; otherwise this
one file (18 MB) is downloaded from one fixed revision of the model's Hub
repository into this chapter's folder and checked before use.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path

from tasks import CHAPTER, course_home
from tokenizers import Tokenizer

TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_SHA256 = "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41"
# Chapter 9's pinned revision of LiquidAI/LFM2.5-2.6B.
REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
DOWNLOAD_URL = (
    f"https://huggingface.co/LiquidAI/LFM2.5-2.6B/resolve/{REVISION}/{TOKENIZER_FILE}"
)
# Where this chapter downloads it, and where earlier chapters keep it,
# relative to COURSE_HOME.
DOWNLOAD_FOLDER = Path(CHAPTER) / "LFM2.5-2.6B"
KNOWN_COPIES = (
    Path("models") / "LFM2.5-2.6B",
    Path("models") / "LFM2.5-2.6B-MLX-4bit",
    Path("09-tokens-and-templates") / "LFM2.5-2.6B",
    DOWNLOAD_FOLDER,
)
TOKENIZER_MB = 18  # its size, for the download message
TIMEOUT_S = 60
CHUNK_BYTES = 1 << 20
PART_SUFFIX = ".part"


class TokenizerError(Exception):
    """Raised when no checked copy of the tokenizer can be found or fetched."""


def sha256_of(path: Path) -> str:
    """Return a file's SHA-256 fingerprint.

    Args:
        path: The file.

    Returns:
        64 hex characters.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(target: Path) -> None:
    """Fetch tokenizer.json to ``target``, checking it before naming it.

    Args:
        target: Where the file should end up.

    Raises:
        TokenizerError: If the download fails or its fingerprint is wrong.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + PART_SUFFIX)
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=TIMEOUT_S) as response:
            part.write_bytes(response.read())
    except OSError as error:  # no network, a refused connection, an HTTP error
        raise TokenizerError(f"could not download {DOWNLOAD_URL}: {error}") from error
    if sha256_of(part) != TOKENIZER_SHA256:
        part.unlink()
        raise TokenizerError(f"{DOWNLOAD_URL} did not match its pinned fingerprint")
    part.rename(target)


def find_tokenizer() -> Path:
    """Find a checked copy of LFM2.5-2.6B's tokenizer.json, downloading if needed.

    Returns:
        The file.

    Raises:
        TokenizerError: If none is found and the download fails.
    """
    for folder in KNOWN_COPIES:
        path = course_home() / folder / TOKENIZER_FILE
        if path.is_file() and sha256_of(path) == TOKENIZER_SHA256:
            return path
    target = course_home() / DOWNLOAD_FOLDER / TOKENIZER_FILE
    print(f"downloading {TOKENIZER_FILE} ({TOKENIZER_MB} MB) from the Hugging Face Hub")
    download(target)
    return target


def load_counter(path: Path) -> Callable[[str], int]:
    """Build a function that counts the tokens of a text.

    Args:
        path: A tokenizer.json.

    Returns:
        A function from text to its number of tokens (no special tokens).
    """
    tokenizer = Tokenizer.from_file(str(path))

    def count(text: str) -> int:
        """Count the tokens LFM2.5-2.6B cuts ``text`` into."""
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count
