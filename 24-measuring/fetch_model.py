# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==1.33.0", "httpx==0.28.1"]
# ///
"""Fetch the model files this chapter needs when they are missing, never twice.

This chapter uses what earlier chapters made, and fetches it itself when it
is not on disk, instead of sending you back to another folder:

* **the model**, for ``./start.sh``: chapter 11's 4-bit build when it is
  there, else Liquid's 4-bit export (what chapter 10's ``download.py
  --fast`` fetches). When neither is on disk, this script downloads the
  export (1.6 GB) into ``COURSE_HOME/models`` at the pinned revision, as
  chapter 10 does, and checks every file's size and SHA-256;
* **LFM2.5's tokenizer**, for ``./start.sh scripted``: the scripted engine
  lays out and counts prompts with ``tokenizer.json`` and
  ``chat_template.jinja`` (no weights). ``--tokenizer`` fetches only the
  tokenizer files (17.9 MB) when no checked copy is on disk.

``start.sh`` runs it before an engine starts; with everything on disk it
only looks. ``--fast`` asks for Liquid's export even when ours is built (the
engine's ``--fast``); ``--check`` never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from lfm_files import (
    FOUR_BIT,
    TokenizerMissingError,
    build_ready,
    course_home,
    gigabytes,
    lfm_tokenizer_file,
    missing_files,
    model_folder,
    ours_folder,
    short_path,
)

# Each file's SHA-256 at the pinned revision, as chapter 10 pinned them.
FINGERPRINTS = {
    "model.safetensors": (
        "ef350b75815752f8bff8c064c9fc8bd8b5709577d72943436852013d2bcd4691"
    ),
    "model.safetensors.index.json": (
        "620f8efaf53e826757324852e999f6ba9a5d974e193589470a70951fd17a1543"
    ),
    "config.json": "6ab5ad71687f6ca8d3811b9290ac06291f78e1e0f4f3ec504e6644032d58342b",
    "generation_config.json": (
        "e08c9a83f4bfe64468c9671a2b69b60549728e2a9481387f3555471f64e958bc"
    ),
    "tokenizer.json": (
        "695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41"
    ),
    "tokenizer_config.json": (
        "11f1de897317b489dd09199284528382eeb17f566398b16e2039bb2266d26b09"
    ),
    "chat_template.jinja": (
        "aadd425edb1d42fe9288e5de3624c21c1c03099129b9cfea5978f1bc67222c0a"
    ),
    "LICENSE": "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
    "README.md": "0907976d22f77d4178e10d04002403a94a22ad965ad8431f754fb28a14885f6e",
    ".gitattributes": (
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"
    ),
}
# What the scripted engine reads: the tokenizer, its settings and the template.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CHUNK_BYTES = 1 << 24  # hash 16 MiB at a time
SHOWN = 12  # characters of a fingerprint shown in an error
# huggingface_hub keeps a cache of its own for large files (hf_xet); like
# chapter 10, point it into COURSE_HOME unless you chose a place yourself.
XET_CACHE_ENV = "HF_XET_CACHE"
XET_FOLDER = "huggingface-xet"
UNREACHABLE = (httpx.TransportError, ConnectionError, LocalEntryNotFoundError)


class FetchError(Exception):
    """Raised when a file cannot be fetched or fails its check."""


def model_on_disk(fast: bool) -> tuple[Path, str] | None:
    """Find the 4-bit model the engine would load, without downloading.

    Args:
        fast: Only Liquid's export will do (the engine's ``--fast``).

    Returns:
        Its folder and which build it is, or None when it is not on disk.
    """
    if not fast and build_ready(ours_folder()):
        return ours_folder(), "ours, built by chapter 11's convert.py"
    if not missing_files(FOUR_BIT, model_folder(FOUR_BIT)):
        return model_folder(FOUR_BIT), "Liquid's export"
    return None


def tokenizer_on_disk() -> Path | None:
    """Find a checked copy of LFM2.5's tokenizer, with its chat template.

    Returns:
        The ``tokenizer.json``, or None when no checked copy is there.
    """
    try:
        path = lfm_tokenizer_file()
    except TokenizerMissingError:
        return None
    return path if (path.parent / "chat_template.jinja").is_file() else None


def sha256_of(path: Path) -> str:
    """Compute a file's SHA-256 fingerprint, in chunks.

    Args:
        path: The file.

    Returns:
        64 hex characters.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(folder: Path, names: tuple[str, ...]) -> None:
    """Check fetched files against their sizes and fingerprints.

    Args:
        folder: Where they are.
        names: Which files.

    Raises:
        FetchError: If one is missing or not the pinned file (a wrong file
            is deleted, never left under the right name).
    """
    sizes = {item.name: item.size for item in FOUR_BIT.files}
    for name in names:
        path = folder / name
        if not path.is_file():
            raise FetchError(f"{name} is missing after the download")
        size = path.stat().st_size
        found = sha256_of(path) if size == sizes[name] else ""
        if found != FINGERPRINTS[name]:
            path.unlink()
            raise FetchError(
                f"{name} is not the pinned file ({size:,} bytes, SHA-256 "
                f"{found[:SHOWN] or 'not computed'}; expected {sizes[name]:,} bytes, "
                f"{FINGERPRINTS[name][:SHOWN]}): it was deleted, run it again"
            )


def fetch(names: tuple[str, ...]) -> Path:
    """Download some of Liquid's export at its pinned revision, then check them.

    Args:
        names: The files to fetch.

    Returns:
        The folder they are in.

    Raises:
        FetchError: If the Hub cannot be reached or refuses, the disk cannot
            take the files, or a file fails its check.
    """
    folder = model_folder(FOUR_BIT)
    os.environ.setdefault(XET_CACHE_ENV, str(course_home() / XET_FOLDER))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            FOUR_BIT.repo_id,
            revision=FOUR_BIT.revision,
            local_dir=folder,
            allow_patterns=list(names),
        )
    except HfHubHTTPError as error:  # before OSError: it is one too
        first = str(error).splitlines()[0]
        raise FetchError(f"the Hub refused {FOUR_BIT.repo_id}: {first}") from error
    except UNREACHABLE as error:
        raise FetchError(
            f"could not reach the Hugging Face Hub ({error}); the first run needs "
            "the internet, and a broken download resumes where it stopped"
        ) from error
    except OSError as error:
        where = short_path(folder)
        raise FetchError(f"could not write into {where}: {error}") from error
    check_files(folder, names)
    return folder


def size_of(names: tuple[str, ...]) -> str:
    """Say how much a set of the export's files weighs.

    Args:
        names: The files.

    Returns:
        Such as ``1.60 GB``.
    """
    return gigabytes(sum(item.size for item in FOUR_BIT.files if item.name in names))


def ensure_model(fast: bool, check_only: bool) -> None:
    """Make sure the engine's model is on disk, fetching it when it is not.

    Args:
        fast: Only Liquid's export will do.
        check_only: Never download.

    Raises:
        SystemExit: If it is missing and may not, or cannot, be fetched.
    """
    found = model_on_disk(fast)
    if found is not None:
        folder, which = found
        print(f"model     ready: {which} ({short_path(folder)})")
        return
    names = tuple(item.name for item in FOUR_BIT.files)
    revision = FOUR_BIT.revision[:7]
    if check_only:
        raise SystemExit("model     missing: run `uv run fetch_model.py`")
    print(
        f"model     missing: fetching {FOUR_BIT.repo_id} at {revision} "
        f"({size_of(names)}) into {short_path(model_folder(FOUR_BIT))}",
        flush=True,
    )
    try:
        fetch(names)
    except FetchError as error:
        raise SystemExit(f"model     {error}") from error
    print(f"model     fetched and checked: {len(names)} files, each by its SHA-256")


def ensure_tokenizer(check_only: bool) -> None:
    """Make sure LFM2.5's tokenizer and template are on disk, fetching them.

    Args:
        check_only: Never download.

    Raises:
        SystemExit: If they are missing and may not, or cannot, be fetched.
    """
    found = tokenizer_on_disk()
    if found is not None:
        print(f"tokenizer ready: {short_path(found)} (checked)")
        return
    if check_only:
        raise SystemExit("tokenizer missing: run `uv run fetch_model.py --tokenizer`")
    print(
        f"tokenizer missing: fetching {', '.join(TOKENIZER_FILES)} from "
        f"{FOUR_BIT.repo_id} ({size_of(TOKENIZER_FILES)})",
        flush=True,
    )
    try:
        folder = fetch(TOKENIZER_FILES)
    except FetchError as error:
        raise SystemExit(f"tokenizer {error}") from error
    print(f"tokenizer fetched and checked: {short_path(folder / TOKENIZER_FILES[0])}")


def main() -> None:
    """Look for what the engine needs; fetch what is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", action="store_true", help="tokenizer only")
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    parser.add_argument("--check", action="store_true", help="never download")
    args = parser.parse_args()
    if args.tokenizer:
        ensure_tokenizer(args.check)
    else:
        ensure_model(args.fast, args.check)


if __name__ == "__main__":
    main()
