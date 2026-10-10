"""Download LFM2.5-2.6B's three tokenizer files from the Hugging Face Hub.

The Hugging Face Hub (huggingface.co) is a website where model makers publish
each model as a folder of files, and every file has its own web address. This
chapter needs three of LFM2.5-2.6B's files, about 18 MB in all:

- ``tokenizer.json``: the vocabulary and the rules for cutting text into tokens;
- ``tokenizer_config.json``: which tokens start a prompt, end a turn and pad;
- ``chat_template.jinja``: the recipe that turns a chat into one string.

The weights (5.4 GB) stay on the Hub until chapter 10. Each file is fetched
from one fixed revision of the repository, written to a ``.part`` file, checked
against its SHA-256 fingerprint and only then given its real name, so a broken
download never passes for a finished one, and every run reads the same files.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The course's one setting: where downloads and outputs live.
COURSE_HOME_ENV = "COURSE_HOME"
DEFAULT_COURSE_HOME = Path.home() / ".cache" / "mhrl-course"
CHAPTER = "09-tokens-and-templates"
MODEL_FOLDER = "LFM2.5-2.6B"

HUB_URL = "https://huggingface.co"
REPO_ID = "LiquidAI/LFM2.5-2.6B"
# One fixed commit of the repository (its main branch since 2026-08-24), so the
# files can never change under the course. The project runs Liquid's 4-bit MLX
# export, LiquidAI/LFM2.5-2.6B-MLX-4bit: its tokenizer.json and
# tokenizer_config.json are these exact files, and its chat_template.jinja
# only adds a one-line comment at the top, which renders to nothing.
REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
TIMEOUT_S = 60
CHUNK_BYTES = 1 << 20  # read 1 MiB at a time
PART_SUFFIX = ".part"
FINGERPRINT_SHOWN = 12  # characters of a SHA-256 shown in an error
MEGABYTE = 1_000_000
GIGABYTE = 1_000_000_000
# The weights at this revision: model-00001-of-00002.safetensors and
# model-00002-of-00002.safetensors, never downloaded here.
WEIGHTS_BYTES = 5_329_406_264 + 65_021_192


@dataclass(frozen=True)
class HubFile:
    """One file to download from the model's Hub repository.

    Attributes:
        name: The file name, the same on the Hub and on disk.
        size: Its size in bytes.
        sha256: Its SHA-256 fingerprint, as 64 hex characters.
    """

    name: str
    size: int
    sha256: str


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
CHAT_TEMPLATE = HubFile(
    "chat_template.jinja",
    5_443,
    "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef",
)
FILES = (TOKENIZER_JSON, TOKENIZER_CONFIG, CHAT_TEMPLATE)

# What can go wrong while talking to the Hub: no network, a refused or dropped
# connection, a timeout, a broken secure connection, or an HTTP error
# (HTTPError is a kind of URLError).
NETWORK_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    ssl.SSLError,
    http.client.HTTPException,
)


class HubError(Exception):
    """Raised when the tokenizer files cannot be downloaded or fail their check."""


def course_home() -> Path:
    """Return the course's cache folder (``$COURSE_HOME`` or the default).

    Returns:
        The folder path; it is not created here.
    """
    configured = os.environ.get(COURSE_HOME_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_COURSE_HOME


def model_folder() -> Path:
    """Return the folder that holds the three tokenizer files.

    Returns:
        ``$COURSE_HOME/09-tokens-and-templates/LFM2.5-2.6B``.
    """
    return course_home() / CHAPTER / MODEL_FOLDER


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


def file_url(item: HubFile) -> str:
    """Return the web address of one file at the pinned revision.

    Args:
        item: The file.

    Returns:
        ``https://huggingface.co/<repo>/resolve/<revision>/<name>``.
    """
    return f"{HUB_URL}/{REPO_ID}/resolve/{REVISION}/{item.name}"


def missing_files(folder: Path) -> list[HubFile]:
    """List the files that are not in the folder at their full size.

    Only the size is checked here, which is instant; the SHA-256 fingerprint
    was checked when the file was downloaded, before it got its real name.

    Args:
        folder: Where the files should be.

    Returns:
        The files still to download, in download order.
    """
    missing = []
    for item in FILES:
        path = folder / item.name
        if not path.is_file() or path.stat().st_size != item.size:
            missing.append(item)
    return missing


def size_text(size: int) -> str:
    """Write a size the way people say it: megabytes, or bytes if smaller.

    Args:
        size: A size in bytes.

    Returns:
        Such as ``"17.9 MB"`` or ``"363 bytes"``.
    """
    if size < MEGABYTE:
        return f"{size:,} bytes"
    return f"{size / MEGABYTE:.1f} MB"


def failure_message(item: HubFile, error: Exception) -> str:
    """Explain a failed download in words a newcomer can act on.

    Args:
        item: The file that could not be fetched.
        error: What urllib raised.

    Returns:
        A few lines: what failed, why, and what to do.
    """
    total = size_text(sum(each.size for each in FILES))
    if isinstance(error, urllib.error.HTTPError):
        # The Hub was reached and said no: the connection is not the problem.
        advice = (
            f"The Hub answered HTTP {error.code}. If it says the file is not\n"
            "there, the course's pinned revision may have moved; please report it."
        )
    else:
        advice = (
            f"The first run needs the internet, to fetch three files ({total}).\n"
            "Check your connection (or proxy) and run it again."
        )
    return (
        f"could not download {item.name} from the Hugging Face Hub\n"
        f"  {file_url(item)}\n"
        f"  reason: {error}\n"
        f"{advice}"
    )


def download(item: HubFile, folder: Path) -> None:
    """Fetch one file into the folder, keeping it only if its fingerprint matches.

    The bytes go to ``<name>.part`` first, which is renamed only after the
    check, and removed whatever else happens (an error, a wrong file, Ctrl-C).

    Args:
        item: The file to fetch.
        folder: Where it goes; it must exist.

    Raises:
        HubError: If the Hub cannot be reached, the file cannot be written, or
            the bytes are not the expected file.
    """
    target = folder / item.name
    part = folder / (item.name + PART_SUFFIX)
    digest = hashlib.sha256()
    received = 0
    try:
        with (
            urllib.request.urlopen(file_url(item), timeout=TIMEOUT_S) as response,
            part.open("wb") as out,
        ):
            while chunk := response.read(CHUNK_BYTES):
                received += len(chunk)
                if received > item.size:
                    break  # more bytes than the file has: it cannot be the file
                digest.update(chunk)
                out.write(chunk)
        if received != item.size or digest.hexdigest() != item.sha256:
            raise HubError(
                f"{item.name} arrived, but it is not the expected file "
                f"({received:,} bytes, SHA-256 starting "
                f"{digest.hexdigest()[:FINGERPRINT_SHOWN]}; expected "
                f"{item.size:,} bytes, {item.sha256[:FINGERPRINT_SHOWN]}). "
                "Nothing was kept; run it again."
            )
        part.replace(target)
    except NETWORK_ERRORS as error:
        raise HubError(failure_message(item, error)) from error
    except OSError as error:  # most often the disk: full, read-only, ...
        raise HubError(
            f"could not download or write {short_path(part)}: {error}"
        ) from error
    finally:
        part.unlink(missing_ok=True)


def ensure_files(say: Callable[[str], None] = print) -> Path:
    """Make sure the three tokenizer files are on disk, downloading what is missing.

    Args:
        say: Where progress lines go (``print`` by default).

    Returns:
        The folder that holds the files.

    Raises:
        HubError: If a missing file cannot be downloaded.
    """
    folder = model_folder()
    todo = missing_files(folder)
    if not todo:
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    total = sum(item.size for item in todo)
    say(f"downloading from the Hugging Face Hub: {REPO_ID}, tokenizer files only")
    say(f"  into {short_path(folder)}")
    for item in todo:
        download(item, folder)
        say(f"  {item.name:<24}{item.size:>12,} bytes  SHA-256 checked")
    weights = WEIGHTS_BYTES / GIGABYTE
    say(f"  {size_text(total)} in all; the {weights:.1f} GB of weights stay on the Hub")
    return folder
