# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==1.33.0", "httpx==0.28.1"]
# ///
"""Download LFM2.5-2.6B from the Hugging Face Hub, check it, and read its license.

The same call the project's ``scripts/setup.sh`` makes,
``huggingface_hub.snapshot_download``, fetches every file of the repository
into ``$COURSE_HOME/models``: the weights (two safetensors files, 5.4 GB in
bfloat16), config.json, the tokenizer files, the chat template, the maker's
default sampling settings, the model card and the LICENSE. Every file is then
checked against its pinned SHA-256 fingerprint, and the license's commercial
terms are printed straight from the LICENSE file.

Run it with ``uv run download.py``. ``--fast`` downloads Liquid's 4-bit MLX
export instead (1.6 GB), the copy the project itself runs; every other script
of the chapter takes ``--fast`` to use it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import report
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from lfm_files import (
    LICENSE,
    ModelRepo,
    course_home,
    gigabytes,
    missing_files,
    model_folder,
    pick_repo,
    short_path,
)

SCRIPT = "download.py"
TITLE = "LFM2.5-2.6B from the Hugging Face Hub"
CHUNK_BYTES = 1 << 24  # hash 16 MiB at a time
FINGERPRINT_SHOWN = 12  # characters of a SHA-256 shown in an error
# The LICENSE lines this script quotes: its title is the first line; the
# definitions it needs start with these names; section 5 holds the limit.
DEFINED_TERMS = ('"Commercial Use"', '"Threshold"')
LIMIT_SECTION = "5. Commercial Use Limitation."
NEXT_SECTION = "6. "
SUBCLAUSE = "("
# huggingface_hub downloads large files through hf_xet, which keeps its own
# folder of logs and chunks (by default ~/.cache/huggingface/xet). Pointing it
# into COURSE_HOME keeps everything the course writes in one place; a value
# the reader set themselves wins.
XET_CACHE_ENV = "HF_XET_CACHE"
XET_FOLDER = "huggingface-xet"

# Talking to the Hub can fail in three ways: it is unreachable (no network, a
# refused or dropped connection, a timeout, offline mode), it answers with an
# HTTP error (a wrong revision, a rate limit), or the disk refuses the bytes.
UNREACHABLE_ERRORS = (httpx.TransportError, ConnectionError, LocalEntryNotFoundError)


class DownloadError(Exception):
    """Raised when the model cannot be downloaded or fails its check."""


@dataclass(frozen=True)
class LicenseTerms:
    """The lines of the LICENSE that say who may use the model commercially.

    Attributes:
        title: The license's name, its first line.
        definitions: The definitions of the terms the limit uses.
        limit: Section 5's heading and its clauses, as written.
    """

    title: str
    definitions: list[str]
    limit: list[str]


def unreachable_message(repo: ModelRepo, error: Exception) -> str:
    """Explain, in words a newcomer can act on, that the Hub could not be reached.

    Args:
        repo: The repository being downloaded.
        error: What huggingface_hub raised.

    Returns:
        A few lines: what failed, why, and what to do.
    """
    return (
        f"could not reach the Hugging Face Hub to download {repo.repo_id}\n"
        f"  reason: {error}\n"
        f"The first run needs the internet, to fetch {gigabytes(repo.total_bytes())}.\n"
        "Check your connection (or proxy) and run it again; a download that\n"
        "broke off resumes where it stopped."
    )


def http_error_message(repo: ModelRepo, error: HfHubHTTPError) -> str:
    """Explain an HTTP error from the Hub: it was reached, and it said no.

    Args:
        repo: The repository being downloaded.
        error: What huggingface_hub raised.

    Returns:
        A few lines: what failed and what to do.
    """
    first_line = str(error).splitlines()[0]
    return (
        f"the Hugging Face Hub refused to send {repo.repo_id} "
        f"at revision {repo.revision[:FINGERPRINT_SHOWN]}\n"
        f"  reason: {first_line}\n"
        "The connection works. If the Hub says the revision is not there, the\n"
        "course's pinned revision may have moved; please report it. A 429 means\n"
        "too many requests: wait a minute, or set HF_TOKEN, and run it again."
    )


def fetch(repo: ModelRepo, folder: Path) -> None:
    """Download every file of the repository at its pinned revision.

    ``snapshot_download`` skips files that are already complete, resumes a
    file that broke off, and shows its progress on the terminal.

    Args:
        repo: The repository.
        folder: Where its files go.

    Raises:
        DownloadError: If the Hub cannot be reached, refuses, or the disk
            cannot take the files.
    """
    os.environ.setdefault(XET_CACHE_ENV, str(course_home() / XET_FOLDER))
    try:
        snapshot_download(repo.repo_id, revision=repo.revision, local_dir=folder)
    except HfHubHTTPError as error:  # before OSError: it is one too
        raise DownloadError(http_error_message(repo, error)) from error
    except UNREACHABLE_ERRORS as error:
        raise DownloadError(unreachable_message(repo, error)) from error
    except OSError as error:  # most often the disk: full, read-only, ...
        raise DownloadError(
            f"could not write the files into {short_path(folder)}: {error}"
        ) from error


def sha256_of(path: Path) -> str:
    """Compute a file's SHA-256 fingerprint, reading it in chunks.

    Args:
        path: The file.

    Returns:
        The fingerprint as 64 hex characters.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_fingerprints(repo: ModelRepo, folder: Path) -> None:
    """Check every downloaded file against its size and SHA-256 fingerprint.

    Args:
        repo: The repository.
        folder: Where its files are.

    Raises:
        DownloadError: If a file is missing, the wrong size or the wrong bytes,
            or cannot be read or deleted.
    """
    for item in repo.files:
        path = folder / item.name
        if not path.is_file():
            raise DownloadError(f"{item.name} is missing after the download")
        try:
            size = path.stat().st_size
            fingerprint = sha256_of(path) if size == item.size else ""
            if fingerprint != item.sha256:
                path.unlink()  # never leave a wrong file under the right name
        except OSError as error:
            raise DownloadError(
                f"could not check {short_path(path)}: {error}"
            ) from error
        if fingerprint != item.sha256:
            raise DownloadError(
                f"{item.name} arrived, but it is not the expected file "
                f"({size:,} bytes, SHA-256 starting {fingerprint[:FINGERPRINT_SHOWN]}"
                f"; expected {item.size:,} bytes, {item.sha256[:FINGERPRINT_SHOWN]}). "
                "It was deleted; run it again."
            )


def ensure_model(repo: ModelRepo, say: Callable[[str], None] = print) -> Path:
    """Make sure every file of the repository is on disk and checked.

    Args:
        repo: The repository.
        say: Where progress lines go (``print`` by default).

    Returns:
        The folder that holds the files.

    Raises:
        DownloadError: If the files cannot be downloaded or fail their check.
    """
    folder = model_folder(repo)
    todo = missing_files(repo, folder)
    report.print_source(repo, folder, say)
    if not todo:
        report.print_already_here(repo, say)
        return folder
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DownloadError(
            f"could not create {short_path(folder)}: {error}"
        ) from error
    report.print_downloading(repo, todo, say)
    fetch(repo, folder)
    check_fingerprints(repo, folder)
    report.print_checked(repo, say)
    return folder


def license_terms(text: str) -> LicenseTerms:
    """Pick out the lines of a LICENSE that limit commercial use.

    Args:
        text: The LICENSE file's text.

    Returns:
        Its title, the definitions the limit uses, and section 5 as written.

    Raises:
        ValueError: If the text has no title or no section 5, so it is not
            the license this chapter describes.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or LIMIT_SECTION not in lines:
        raise ValueError(f"no '{LIMIT_SECTION}' section: not the LFM Open License")
    definitions = [line for line in lines if line.startswith(DEFINED_TERMS)]
    start = lines.index(LIMIT_SECTION)
    limit = [LIMIT_SECTION]
    for line in lines[start + 1 :]:
        if line.startswith(NEXT_SECTION):
            break
        if line.startswith(SUBCLAUSE):
            limit.append(line)
    return LicenseTerms(lines[0], definitions, limit)


def main() -> None:
    """Download (or find) the model, list its files and print its license terms."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="download Liquid's 4-bit MLX export (1.6 GB) instead of bfloat16",
    )
    args = parser.parse_args()
    repo = pick_repo(args.fast)

    report.print_title(SCRIPT, TITLE)
    try:
        folder = ensure_model(repo)
    except DownloadError as error:
        raise SystemExit(str(error)) from None
    report.print_files(repo)
    path = folder / LICENSE.name
    try:
        terms = license_terms(path.read_text(encoding="utf-8"))
    except ValueError as error:  # a LICENSE of the right size, but not this one
        message = f"{short_path(path)} is not the expected license: {error}"
        raise SystemExit(message) from None
    report.print_license(terms, path.stat().st_size)


if __name__ == "__main__":
    main()
