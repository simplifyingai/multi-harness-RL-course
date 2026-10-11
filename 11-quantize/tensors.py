"""Read a safetensors file's header and its tensors' bytes, one at a time.

A safetensors file starts with 8 bytes that give the length of its header,
then the header: JSON naming every tensor with its number type, its shape and
where its bytes sit after the header (chapter 10 read these headers to count
the model's numbers). Knowing where a tensor sits, a script can read that
tensor alone, or fingerprint every tensor one by one, without loading the
model. This module needs only the standard library.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

HEADER_LENGTH_BYTES = 8  # a little-endian unsigned 64-bit integer
HEADER_LENGTH_FORMAT = "<Q"
MAX_HEADER_BYTES = 100_000_000  # the safetensors format's own limit
METADATA_KEY = "__metadata__"
BYTES_PER_DTYPE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I64": 8}
READ_CHUNK = 1 << 24  # hash big tensors 16 MiB at a time


class HeaderError(ValueError):
    """Raised when bytes are not a readable safetensors header."""


@dataclass(frozen=True)
class Tensor:
    """One tensor as a safetensors header describes it.

    Attributes:
        name: Its name, such as ``model.layers.10.feed_forward.w1.weight``.
        dtype: Its number type as stored, such as ``BF16`` or ``U32``.
        shape: Its shape as stored.
        start: Where its bytes start, counted from the start of the file.
        nbytes: How many bytes it takes.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    nbytes: int

    def stored(self) -> int:
        """Count the values as stored (a packed word counts as one).

        Returns:
            The product of the shape.
        """
        count = 1
        for size in self.shape:
            count *= size
        return count


@dataclass(frozen=True)
class Header:
    """What a safetensors file's header says.

    Attributes:
        header_bytes: The header's length, plus the 8 bytes that give it.
        metadata: The free-form ``__metadata__`` entry, if any.
        tensors: Every tensor, in the order the header lists them.
    """

    header_bytes: int
    metadata: dict[str, str]
    tensors: list[Tensor]

    def find(self, name: str) -> Tensor:
        """Look a tensor up by its name.

        Args:
            name: The tensor's name.

        Returns:
            The tensor.

        Raises:
            KeyError: If the header lists no such tensor.
        """
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def data_bytes(self) -> int:
        """Add up the bytes of every tensor.

        Returns:
            The bytes after the header.
        """
        return sum(tensor.nbytes for tensor in self.tensors)


def header_length(prefix: bytes, file_size: int | None = None) -> int:
    """Read the header's length from a file's first 8 bytes.

    Args:
        prefix: At least the file's first 8 bytes.
        file_size: The whole file's size, if known, to check the length fits.

    Returns:
        The header's length in bytes (not counting the 8).

    Raises:
        HeaderError: If there are fewer than 8 bytes or the length is absurd.
    """
    if len(prefix) < HEADER_LENGTH_BYTES:
        raise HeaderError("too short to be a safetensors file")
    (length,) = struct.unpack(HEADER_LENGTH_FORMAT, prefix[:HEADER_LENGTH_BYTES])
    limit = MAX_HEADER_BYTES
    if file_size is not None:
        limit = min(limit, file_size - HEADER_LENGTH_BYTES)
    if length > limit:
        raise HeaderError(f"says its header is {length:,} bytes: not safetensors")
    return length


def parse_header(raw: bytes) -> Header:
    """Turn a header's JSON bytes into a Header.

    Args:
        raw: The header's bytes, after the 8-byte length.

    Returns:
        The header.

    Raises:
        HeaderError: If the JSON or a tensor entry does not make sense.
    """
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HeaderError(f"the header is not JSON ({error})") from error
    if not isinstance(entries, dict):
        raise HeaderError("the header is not a JSON object")
    data_start = HEADER_LENGTH_BYTES + len(raw)
    tensors = [
        tensor_from(name, entry, data_start)
        for name, entry in entries.items()
        if name != METADATA_KEY
    ]
    metadata = entries.get(METADATA_KEY) or {}
    if not isinstance(metadata, dict):
        raise HeaderError("its __metadata__ entry is not a JSON object")
    return Header(data_start, dict(metadata), tensors)


def tensor_from(name: str, entry: Any, data_start: int) -> Tensor:
    """Turn one header entry into a Tensor, checking its byte count.

    Args:
        name: The tensor's name.
        entry: Its entry: ``dtype``, ``shape`` and ``data_offsets``.
        data_start: Where the data starts in the file (just after the header).

    Returns:
        The tensor.

    Raises:
        HeaderError: If the entry is malformed or its bytes do not fit its
            shape.
    """
    try:
        first, last = entry["data_offsets"]
        shape = tuple(int(size) for size in entry["shape"])
        dtype = str(entry["dtype"])
    except (KeyError, TypeError, ValueError) as error:
        raise HeaderError(f"{name}: a malformed entry ({error!r})") from error
    whole = isinstance(first, int) and isinstance(last, int)
    if not whole or first < 0 or last < first:
        raise HeaderError(f"{name}: its byte offsets {first}..{last} make no sense")
    tensor = Tensor(name, dtype, shape, data_start + first, last - first)
    width = BYTES_PER_DTYPE.get(tensor.dtype)
    if width is not None and tensor.nbytes != tensor.stored() * width:
        raise HeaderError(f"{name}: {tensor.nbytes:,} bytes do not fit its shape")
    return tensor


def read_header(path: Path) -> Header:
    """Read a safetensors file's header, and nothing after it.

    Args:
        path: The file.

    Returns:
        Its header.

    Raises:
        HeaderError: If the file is not a readable safetensors file.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        length = header_length(handle.read(HEADER_LENGTH_BYTES), size)
        raw = handle.read(length)
    try:
        return parse_header(raw)
    except HeaderError as error:
        raise HeaderError(f"{path.name}: {error}") from error


def read_tensor_bytes(path: Path, tensor: Tensor) -> bytes:
    """Read one tensor's bytes from a file, and no others.

    Args:
        path: The safetensors file.
        tensor: The tensor, from that file's header.

    Returns:
        Its raw bytes.

    Raises:
        HeaderError: If the file ends before the tensor does.
    """
    with path.open("rb") as handle:
        handle.seek(tensor.start)
        data = handle.read(tensor.nbytes)
    if len(data) != tensor.nbytes:
        raise HeaderError(f"{path.name}: ends inside {tensor.name}")
    return data


def _hash_range(handle: BinaryIO, start: int, size: int) -> str:
    """Fingerprint one stretch of an open file with MD5.

    Args:
        handle: The file, opened for reading bytes.
        start: Where the stretch starts.
        size: How many bytes it is.

    Returns:
        The MD5 fingerprint, as 32 hex characters.

    Raises:
        HeaderError: If the file ends before the stretch does.
    """
    digest = hashlib.md5(usedforsecurity=False)
    handle.seek(start)
    left = size
    while left:
        chunk = handle.read(min(READ_CHUNK, left))
        if not chunk:
            raise HeaderError(f"the file ends {left:,} bytes before the data does")
        digest.update(chunk)
        left -= len(chunk)
    return digest.hexdigest()


def tensor_md5s(path: Path) -> Iterator[tuple[Tensor, str]]:
    """Fingerprint every tensor of a file with MD5, one tensor at a time.

    MD5 is used as a fingerprint, not for security: two tensors with the same
    MD5 hold the same bytes, and one changed byte changes the MD5.

    Args:
        path: The safetensors file.

    Yields:
        Each tensor, with its MD5, in the order the header lists them.

    Raises:
        HeaderError: If the file is not a readable safetensors file.
    """
    header = read_header(path)
    with path.open("rb") as handle:
        for tensor in header.tensors:
            yield tensor, _hash_range(handle, tensor.start, tensor.nbytes)


def file_md5(path: Path) -> str:
    """Fingerprint a whole file with MD5.

    Args:
        path: The file.

    Returns:
        Its MD5 fingerprint, as 32 hex characters.

    Raises:
        HeaderError: If the file shrinks while it is read.
    """
    with path.open("rb") as handle:
        return _hash_range(handle, 0, path.stat().st_size)
