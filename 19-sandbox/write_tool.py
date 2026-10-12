# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Pi's write tool, step for step, so the chapter can run it inside the box.

``python write_tool.py PATH CONTENT``

Pi 1.0.4 (the harness the project measured) writes a file in three steps
(``pi-coding-agent/dist/core/tools/write.js`` and ``file-mutation-queue.js``):

1. resolve the path with ``realpath``, to queue writes to the same file one
   after another; a file that does not exist yet is fine (``ENOENT`` or
   ``ENOTDIR``), any other error fails the tool;
2. make the parent folders;
3. write the content.

Node's ``fs.promises.realpath`` calls the C library's ``realpath(3)``, and so
does this script, through ``ctypes``. That function walks up through every
parent folder of the path. When the box hides a parent folder (the first,
too-strict profile), it fails with ``EPERM`` instead of ``ENOENT``, step 1
fails, and the answer is never written.

Standard library only, and no imports from the chapter, so it runs as is
inside the box.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

# Pi treats these as "the file is not there yet", which is fine.
MISSING_ERRORS = (errno.ENOENT, errno.ENOTDIR)
EXIT_FAILED = 1
USAGE = "usage: python write_tool.py PATH CONTENT"
ARGS_EXPECTED = 3
# The system call each step makes, as Node names it in an error.
STEP_REALPATH = "realpath"
STEP_MKDIR = "mkdir"
STEP_OPEN = "open"
TEXT_ENCODING = "utf-8"


def realpath_c(path: str) -> str:
    """Resolve a path with the C library's ``realpath(3)``, as Node does.

    Python's own ``os.path.realpath`` does not call it, so ``ctypes`` does.

    Args:
        path: An absolute path.

    Returns:
        The resolved path.

    Raises:
        OSError: With the errno ``realpath(3)`` set, if it fails.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.realpath.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.realpath.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    pointer = libc.realpath(os.fsencode(path), None)
    if not pointer:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), path)
    try:
        return os.fsdecode(ctypes.string_at(pointer))
    finally:
        libc.free(pointer)


def queue_key(absolute: str) -> str:
    """Pi's ``getMutationQueueKey``: the real path, or the path if missing.

    Args:
        absolute: The absolute path the tool was asked to write.

    Returns:
        The key Pi queues writes under.

    Raises:
        OSError: For any ``realpath`` error other than a missing file.
    """
    try:
        return realpath_c(absolute)
    except OSError as exc:
        if exc.errno in MISSING_ERRORS:
            return absolute
        raise


class ToolError(Exception):
    """One step of the write tool failed.

    Attributes:
        step: The system call that failed (``realpath``, ``mkdir``, ``open``).
        error: The operating system's error.
    """

    def __init__(self, step: str, error: OSError) -> None:
        """Remember the failed step and its error.

        Args:
            step: The system call that failed.
            error: The operating system's error.
        """
        super().__init__(f"{step}: {error}")
        self.step = step
        self.error = error

    def node_message(self) -> str:
        """Format the error the way Node (and so Pi's log) prints it.

        Returns:
            For example ``EPERM: operation not permitted, realpath '/a/b'``.
        """
        code = self.error.errno or errno.EIO
        name = errno.errorcode.get(code, "EIO")
        reason = (self.error.strerror or os.strerror(code)).lower()
        return f"{name}: {reason}, {self.step} '{self.error.filename}'"


def write(path: str, content: str) -> str:
    """Write a file the way Pi's write tool does.

    Args:
        path: The path the model asked for, relative to the working folder
            or absolute.
        content: The text to write.

    Returns:
        Pi's success message.

    Raises:
        ToolError: If any step fails, naming the step.
    """
    absolute = Path(os.getcwd(), path)
    try:
        queue_key(str(absolute))
    except OSError as exc:
        raise ToolError(STEP_REALPATH, exc) from exc
    try:
        absolute.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(STEP_MKDIR, exc) from exc
    try:
        absolute.write_text(content, encoding=TEXT_ENCODING)
    except OSError as exc:
        raise ToolError(STEP_OPEN, exc) from exc
    return f"Successfully wrote to {path}"


def main() -> None:
    """Write CONTENT to PATH and print Pi's message, or Node's error text."""
    if len(sys.argv) != ARGS_EXPECTED:
        print(USAGE, file=sys.stderr)
        sys.exit(EXIT_FAILED)
    try:
        print(write(sys.argv[1], sys.argv[2]))
    except ToolError as exc:
        # The same words Pi's log showed: "EPERM: operation not permitted, ..."
        print(exc.node_message())
        sys.exit(EXIT_FAILED)


if __name__ == "__main__":
    main()
