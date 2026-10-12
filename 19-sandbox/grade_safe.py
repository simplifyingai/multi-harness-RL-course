# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Grade answer.txt without following links: the grader runs OUTSIDE the box.

``uv run grade_safe.py``         the demo: four answer.txt files made by the box
``uv run grade_safe.py FOLDER``  grade one task folder

The grader has to read the gold answers, so it runs outside the box, where
your secrets are readable. An agent inside the box cannot read a secret, but
it CAN make ``answer.txt`` a link to one (a link is only a name). A grader
that opens ``answer.txt`` the ordinary way follows the link and reads the
secret for the agent, and prints it as "what was submitted".

So the safe grader opens ``answer.txt`` like the project's
``mhrl/tasks/core.py`` does: ``O_NOFOLLOW`` (a link is refused, atomically,
with no gap between a check and the open), ``O_NONBLOCK`` (a named pipe
cannot make it wait forever), then ``fstat`` to accept only a regular file,
and at most 64 KiB is read. A hard link is a second NAME for the same file, so
``O_NOFOLLOW`` cannot see it; the box already refuses to make one to a file it
may not read (the demo shows it), and the grader also refuses any answer.txt
with more than one name. That is deliberately strict: an agent that hard-links
its own answer.txt elsewhere is graded wrong, with the reason saying why.

The demo's "secret" is the decoy ``.env`` (``decoys.py``), never a real file.
"""

from __future__ import annotations

import argparse
import errno
import os
import shlex
import stat
import time
from dataclasses import replace
from pathlib import Path

from decoys import make_decoys
from grader import (
    ANSWER_ENCODING,
    MAX_ANSWER_BYTES,
    REASON_NO_ANSWER,
    REASON_NOT_REGULAR,
    REASON_UNREADABLE,
    Grade,
    check_answer,
)
from runner import new_rollout, run_in_box
from tasks import ANSWER_FILE, Task, load_task, show

ANSWER_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
# open() errors that mean "this is not a plain file": a link (ELOOP, from
# O_NOFOLLOW on the last part of the path) or a socket (ENXIO).
NOT_A_FILE_ERRORS = (errno.ELOOP, errno.ENXIO)
# A plain file has one name. A second one (a hard link) could be a secret's.
SINGLE_LINK = 1
REASON_HARD_LINK = f"{ANSWER_FILE} has another name elsewhere (a hard link)"
COMMAND_TIMEOUT_S = 30.0
NAIVE = "naive grader (opens the path)"
SAFE = "safe grader  (O_NOFOLLOW)"


def _read_prefix(descriptor: int, limit: int) -> bytes:
    """Read up to ``limit`` bytes from an open file descriptor.

    Args:
        descriptor: Descriptor of a regular file, at its start.
        limit: Maximum number of bytes to read.

    Returns:
        The bytes read (fewer than ``limit`` at the end of the file).

    Raises:
        OSError: If reading fails.
    """
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def grade_safe(task: Task, workdir: Path) -> Grade:
    """Grade the ``answer.txt`` in a task folder without following links.

    Args:
        task: The task the agent worked on.
        workdir: The task folder.

    Returns:
        The grade. A missing file is wrong with ``"no answer.txt"``; a link, a
        folder, a named pipe or anything else that is not a regular file is
        wrong with ``"answer.txt is not a regular file"``, and so is a file
        with a second name (a hard link); none of them is read. Only the first
        64 KiB is graded, and the reason says so when there was more.
    """
    path = workdir / ANSWER_FILE
    try:
        descriptor = os.open(path, ANSWER_OPEN_FLAGS)
    except FileNotFoundError:
        return Grade(correct=False, submitted=None, reason=REASON_NO_ANSWER)
    except OSError as exc:
        # O_NOFOLLOW refuses a link with ELOOP: the link is never followed.
        not_a_file = exc.errno in NOT_A_FILE_ERRORS
        reason = REASON_NOT_REGULAR if not_a_file else REASON_UNREADABLE
        return Grade(correct=False, submitted=None, reason=reason)
    try:
        # fstat looks at the file that was opened, so nothing can be swapped
        # in between a check and the read.
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return Grade(correct=False, submitted=None, reason=REASON_NOT_REGULAR)
        if info.st_nlink > SINGLE_LINK:
            return Grade(correct=False, submitted=None, reason=REASON_HARD_LINK)
        raw = _read_prefix(descriptor, MAX_ANSWER_BYTES)
    except OSError:
        return Grade(correct=False, submitted=None, reason=REASON_UNREADABLE)
    finally:
        os.close(descriptor)
    grade = check_answer(task, raw.decode(ANSWER_ENCODING, errors="replace"))
    if info.st_size > MAX_ANSWER_BYTES:
        note = f" (only the first {MAX_ANSWER_BYTES} of {info.st_size} bytes read)"
        grade = replace(grade, reason=grade.reason + note)
    return grade


def grade_naive(task: Task, workdir: Path) -> Grade:
    """Grade the way most code would: open the path and read it. Do not use.

    Args:
        task: The task the agent worked on.
        workdir: The task folder.

    Returns:
        The grade; it follows links, so it can read files the agent cannot.
    """
    path = workdir / ANSWER_FILE
    try:
        raw = path.read_bytes()[:MAX_ANSWER_BYTES]
    except FileNotFoundError:
        return Grade(correct=False, submitted=None, reason=REASON_NO_ANSWER)
    except OSError:
        return Grade(correct=False, submitted=None, reason=REASON_UNREADABLE)
    return check_answer(task, raw.decode(ANSWER_ENCODING, errors="replace"))


def describe(grade: Grade) -> str:
    """Put a grade in one line.

    Args:
        grade: The grade.

    Returns:
        ``correct``, ``refused unread: …`` or ``wrong: …``, with what was read.
    """
    if grade.submitted is None:
        return f"refused unread: {grade.reason}"
    verdict = "correct" if grade.correct else "wrong"
    return f"{verdict} · read {grade.submitted!r}"


def demo() -> None:
    """Let the box make four answer.txt files, then grade each one both ways."""
    task = load_task()
    decoys = make_decoys()
    secret = decoys.dot_env
    cases = [
        ("plain", f"echo {task.answer} > {ANSWER_FILE}", True),
        (
            "link",
            f"ln -s {shlex.quote(str(secret))} {ANSWER_FILE}; "
            f"cat {ANSWER_FILE} >/dev/null",
            True,
        ),
        ("hardlink", f"ln {shlex.quote(str(secret))} {ANSWER_FILE}", True),
        ("pipe", f"mkfifo {ANSWER_FILE}", False),
    ]
    print(f"task      {task.question} (gold answer {task.answer})")
    print(f"'secret'  {show(str(secret))} (a decoy .env; the box guards it)")
    for name, command, run_naive in cases:
        # The box guards the decoy home, so the decoy .env is a "secret" to
        # it wherever COURSE_HOME is; the commands here are fixed and harmless.
        rollout = new_rollout(f"grade-{name}", home=decoys.home)
        outcome, output = run_in_box(
            rollout,
            ["/bin/bash", "-c", command],
            label=name,
            timeout_s=COMMAND_TIMEOUT_S,
        )
        print()
        print(f"case {name}: the agent ran, inside the box")
        print(f"  $ {show(command)}")
        for line in show(output).splitlines():
            print(f"  | {line}")
        print(f"  exit {outcome.exit_code}")
        if run_naive:
            print(f"  {NAIVE}: {describe(grade_naive(task, rollout.workdir))}")
        else:
            print(f"  {NAIVE}: not run (open() would wait forever for a writer)")
        started = time.monotonic()
        grade = grade_safe(task, rollout.workdir)
        seconds = time.monotonic() - started
        print(f"  {SAFE}: {describe(grade)} ({seconds:.3f} s)")


def main() -> None:
    """Run the demo, or grade the folder given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, nargs="?")
    args = parser.parse_args()
    if args.folder is None:
        demo()
        return
    task = load_task()
    grade = grade_safe(task, args.folder)
    print(f"question   {task.question}")
    print(f"{ANSWER_FILE} {describe(grade)}")
    print(f"reward     {grade.reward}")


if __name__ == "__main__":
    main()
