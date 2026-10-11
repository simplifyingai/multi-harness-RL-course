"""Find the one matrix by_hand.py quantizes: on disk, or straight from the Hub.

by_hand.py works on one real matrix of LFM2.5-2.6B,
``model.layers.10.feed_forward.w1.weight``: 10,752 x 2,048 numbers in
bfloat16, 44 MB of the 5.4 GB download. It is read from chapter 10's download
when that is on disk. ``--fast`` also works without the download: a
safetensors header says where each tensor's bytes sit (chapter 10), and the
Hub serves any stretch of a file on request (an HTTP *range* request), so the
script fetches the header and then that one matrix, 44 MB instead of 5.4 GB.
Either way the bytes are checked against the matrix's SHA-256 fingerprint
(pinned below) before they are used, so both ways give the same numbers.
This module needs only the standard library.
"""

from __future__ import annotations

import hashlib
import http.client
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lfm_files import BF16, chapter_home, model_folder, short_path
from tensors import (
    HEADER_LENGTH_BYTES,
    HeaderError,
    Tensor,
    header_length,
    parse_header,
    read_header,
    read_tensor_bytes,
)

HUB_URL = "https://huggingface.co"
TIMEOUT_S = 60
PART_SUFFIX = ".part"
CACHE_SUFFIX = ".bf16"  # the matrix's raw bytes, kept for the next run
FINGERPRINT_SHOWN = 12  # characters of a SHA-256 shown in an error
RANGE_HEADER = "Range"
PARTIAL_CONTENT = 206  # the HTTP status of an answered range request

# What can go wrong while talking to the Hub (as chapter 9's hub.py lists).
NETWORK_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    ssl.SSLError,
    http.client.HTTPException,
)


class MatrixError(Exception):
    """Raised when the matrix cannot be read or fetched, or fails its check."""


@dataclass(frozen=True)
class MatrixPin:
    """One tensor of a pinned repository: where it is and what it must be.

    Attributes:
        name: The tensor's name in the safetensors header.
        file: The safetensors file that holds it.
        dtype: Its number type, as the header writes it.
        shape: Its shape (rows, columns).
        sha256: The SHA-256 fingerprint of its bytes.
    """

    name: str
    file: str
    dtype: str
    shape: tuple[int, int]
    sha256: str


# The matrix: layer 10's feed-forward w1, at chapter 10's pinned revision.
W1_LAYER_10 = MatrixPin(
    name="model.layers.10.feed_forward.w1.weight",
    file="model-00001-of-00002.safetensors",
    dtype="BF16",
    shape=(10_752, 2_048),
    sha256="ad96b3c017a1b02b1e64ff6b28794bd013a4bc4169d73a56e2823d33492e4968",
)

# How the Hub is asked for a stretch of a file: (url, first byte, last byte)
# in, those bytes out. urlopen_range is the real one; tests pass a fake.
RangeReader = Callable[[str, int, int], bytes]


@dataclass(frozen=True)
class Matrix:
    """The matrix's raw bytes, and where they came from.

    Attributes:
        pin: Which tensor it is.
        data: Its bytes, as stored (bfloat16, row after row).
        source: Where they were read from, for the reader.
    """

    pin: MatrixPin
    data: bytes
    source: str


def file_url(pin: MatrixPin) -> str:
    """Return the web address of the file holding the matrix.

    Args:
        pin: The matrix.

    Returns:
        ``https://huggingface.co/<repo>/resolve/<revision>/<file>``.
    """
    return f"{HUB_URL}/{BF16.repo_id}/resolve/{BF16.revision}/{pin.file}"


def check(pin: MatrixPin, tensor: Tensor, data: bytes) -> None:
    """Make sure the bytes are the pinned matrix.

    Args:
        pin: What the matrix must be.
        tensor: What the header said about it.
        data: Its bytes.

    Raises:
        MatrixError: If the type, the shape or the fingerprint differ.
    """
    if tensor.dtype != pin.dtype or tensor.shape != pin.shape:
        raise MatrixError(
            f"{pin.name} is {tensor.dtype} {tensor.shape}, expected "
            f"{pin.dtype} {pin.shape}"
        )
    found = hashlib.sha256(data).hexdigest()
    if found != pin.sha256:
        raise MatrixError(
            f"{pin.name}: SHA-256 starts {found[:FINGERPRINT_SHOWN]}, expected "
            f"{pin.sha256[:FINGERPRINT_SHOWN]}: not the pinned matrix"
        )


def read_local(pin: MatrixPin, folder: Path) -> bytes:
    """Read the matrix from a downloaded copy of the repository.

    Args:
        pin: The matrix.
        folder: The repository's folder.

    Returns:
        The matrix's bytes, checked.

    Raises:
        MatrixError: If the file is unreadable or the bytes fail the check.
    """
    path = folder / pin.file
    try:
        tensor = read_header(path).find(pin.name)
        data = read_tensor_bytes(path, tensor)
    except (OSError, HeaderError, KeyError) as error:
        raise MatrixError(f"could not read {pin.name} from {path}: {error}") from error
    check(pin, tensor, data)
    return data


def urlopen_range(url: str, first: int, last: int) -> bytes:
    """Ask a web server for bytes ``first`` to ``last`` of a file.

    Args:
        url: The file's address.
        first: The first byte wanted.
        last: The last byte wanted (included).

    Returns:
        Those bytes.

    Raises:
        MatrixError: If the server sends the whole file instead, or the wrong
            number of bytes.
    """
    wanted = {RANGE_HEADER: f"bytes={first}-{last}"}
    request = urllib.request.Request(url, headers=wanted)
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        if response.status != PARTIAL_CONTENT:
            status = response.status
            raise MatrixError(f"the server ignored the range request ({status})")
        data = response.read()
    if len(data) != last - first + 1:
        raise MatrixError(f"asked for {last - first + 1:,} bytes, got {len(data):,}")
    return data


def fetch(pin: MatrixPin, read_range: RangeReader = urlopen_range) -> bytes:
    """Fetch only the matrix from the Hub: the header first, then its bytes.

    Args:
        pin: The matrix.
        read_range: Reads a stretch of a file over the network.

    Returns:
        The matrix's bytes, checked.

    Raises:
        MatrixError: If the Hub cannot be reached, or what arrives is not the
            pinned matrix.
    """
    url = file_url(pin)
    try:
        length = header_length(read_range(url, 0, HEADER_LENGTH_BYTES - 1))
        raw = read_range(url, HEADER_LENGTH_BYTES, HEADER_LENGTH_BYTES + length - 1)
        tensor = parse_header(raw).find(pin.name)
        data = read_range(url, tensor.start, tensor.start + tensor.nbytes - 1)
    except NETWORK_ERRORS as error:
        raise MatrixError(
            f"could not fetch {pin.name} from the Hugging Face Hub\n"
            f"  {url}\n  reason: {error}\n"
            "Check your connection (or proxy) and run it again."
        ) from error
    except (HeaderError, KeyError) as error:
        raise MatrixError(f"the Hub's copy of {pin.file}: {error}") from error
    check(pin, tensor, data)
    return data


def cache_path(pin: MatrixPin) -> Path:
    """Return where a fetched matrix is kept for the next run.

    Args:
        pin: The matrix.

    Returns:
        ``$COURSE_HOME/11-quantize/<tensor name>.bf16``.
    """
    return chapter_home() / (pin.name + CACHE_SUFFIX)


def keep(path: Path, data: bytes) -> None:
    """Write fetched bytes to the cache, never leaving a half-written file.

    Args:
        path: Where they go.
        data: The bytes.

    Raises:
        MatrixError: If the file cannot be written.
    """
    part = path.with_name(path.name + PART_SUFFIX)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(data)
        part.replace(path)
    except OSError as error:
        raise MatrixError(f"could not write {short_path(part)}: {error}") from error
    finally:
        part.unlink(missing_ok=True)


def load_matrix(
    fast: bool,
    pin: MatrixPin = W1_LAYER_10,
    read_range: RangeReader = urlopen_range,
) -> Matrix:
    """Find the matrix: chapter 10's download, else (with ``--fast``) the Hub.

    Args:
        fast: True to fetch the matrix alone when the download is missing.
        pin: The matrix.
        read_range: Reads a stretch of a file over the network.

    Returns:
        The matrix, checked against its fingerprint.

    Raises:
        MatrixError: If it cannot be found or fetched.
        SystemExit: Without ``--fast``, if chapter 10's download is missing,
            naming both ways to go on.
    """
    folder = model_folder(BF16)
    if (folder / pin.file).is_file():
        source = f"{short_path(folder / pin.file)} (chapter 10's download)"
        return Matrix(pin, read_local(pin, folder), source)
    if not fast:
        raise SystemExit(
            f"chapter 10's bfloat16 download is not in {short_path(folder)}.\n"
            "Run `uv run download.py` in 10-meet-lfm (5.41 GB), or add --fast to "
            "fetch only this matrix from the Hub (44 MB)."
        )
    cached = cache_path(pin)
    note = ""
    if cached.is_file():
        try:
            return Matrix(
                pin,
                read_kept(pin, cached),
                f"{short_path(cached)} (fetched on an earlier run)",
            )
        except (OSError, MatrixError):
            cached.unlink(missing_ok=True)  # damaged: fetch it again
            note = " (the kept copy was damaged, so it was fetched again)"
    data = fetch(pin, read_range)
    keep(cached, data)
    source = f"the Hub, {BF16.repo_id} {pin.file}: this matrix only"
    return Matrix(pin, data, f"{source}; kept in {short_path(cached)}{note}")


def read_kept(pin: MatrixPin, path: Path) -> bytes:
    """Read a matrix kept by an earlier --fast run, checking it again.

    Args:
        pin: The matrix.
        path: The kept copy.

    Returns:
        Its bytes, checked.

    Raises:
        OSError: If the file cannot be read.
        MatrixError: If the bytes are not the pinned matrix.
    """
    data = path.read_bytes()
    check(pin, Tensor(pin.name, pin.dtype, pin.shape, 0, len(data)), data)
    return data
