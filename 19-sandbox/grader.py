"""Chapter 1's answer check: a program, not an AI, decides whether an answer is right.

A copy of the project's ``mhrl/tasks/core.py`` answer check (``check_answer``),
unchanged from chapter 16. How ``answer.txt`` is OPENED is this chapter's
subject and lives in ``grade_safe.py``. The check is lenient about format and
strict about content:

* whitespace, quotes and a trailing period are ignored, and for number and
  text answers only the first non-empty line counts;
* numbers may carry a currency sign and thousands separators, and must match
  within 0.01;
* text matches ignoring case and extra spaces, and a leading "Dr." or "the"
  is ignored on both sides;
* lists may come in any order, split by commas, semicolons or new lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tasks import ANSWER_FILE, Task

NUMBER_TOLERANCE = 0.01
FLOAT_EPSILON = 1e-9
MAX_ANSWER_BYTES = 64 * 1024
MAX_SHOWN_CHARS = 500
ANSWER_ENCODING = "utf-8-sig"  # tolerates a byte-order mark

# The three shapes an answer can take; each is checked its own way.
NUMBER = "number"
TEXT = "text"
LIST = "list"
ANSWER_TYPES = (NUMBER, TEXT, LIST)

REASON_CORRECT = "correct"
REASON_NO_ANSWER = f"no {ANSWER_FILE}"
REASON_NOT_REGULAR = f"{ANSWER_FILE} is not a regular file"
REASON_UNREADABLE = f"{ANSWER_FILE} could not be read"

QUOTE_CHARS = "\"'`"
CURRENCY_CHARS = "$€£"
LIST_BRACKETS = "[]"
NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")
LIST_SEPARATOR_RE = re.compile(r"[,;\n]")
BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
LEADING_AND_RE = re.compile(r"^and\s+", re.IGNORECASE)
LEADING_TITLE_RE = re.compile(r"^(?:dr\.?|the)\s+", re.IGNORECASE)
FENCE_RE = re.compile(r"^`{3,}\w*$")


@dataclass(frozen=True)
class Grade:
    """The verdict on one answer.

    Attributes:
        correct: Whether the answer matches.
        submitted: What was judged, or None when there was no answer file.
        reason: ``"correct"`` or why the answer was rejected.
    """

    correct: bool
    submitted: str | None
    reason: str

    @property
    def reward(self) -> int:
        """The reward an RL trainer would get: 1 for correct, 0 otherwise."""
        return 1 if self.correct else 0


def check_answer(task: Task, text: str) -> Grade:
    """Grade an answer given as text.

    Args:
        task: The task being answered.
        text: What the agent wrote.

    Returns:
        The grade.

    Raises:
        ValueError: If the task has an unknown answer type, or a number task's
            own answer is not a number.
    """
    if task.answer_type not in ANSWER_TYPES:
        raise ValueError(f"unknown answer type {task.answer_type!r}")
    if task.answer_type == LIST:
        submitted = text.strip()
    else:
        submitted = first_line(text)
    if not clean(submitted):
        return Grade(correct=False, submitted="", reason=REASON_NO_ANSWER)
    shown = submitted[:MAX_SHOWN_CHARS]
    wrong = f"expected {task.answer!r}, got {shown!r}"
    if task.answer_type == NUMBER:
        value = parse_number(submitted)
        if value is None:
            return Grade(
                correct=False, submitted=shown, reason=f"not a number: {shown!r}"
            )
        gold = parse_number(task.answer)
        if gold is None:
            raise ValueError(f"task {task.task_id} has a non-numeric answer")
        correct = abs(value - gold) <= NUMBER_TOLERANCE + FLOAT_EPSILON
    elif task.answer_type == TEXT:
        correct = normalize_text(submitted) == normalize_text(task.answer)
    else:
        given, expected = list_items(submitted), list_items(task.answer)
        correct = given == expected
        missing, extra = len(expected - given), len(given - expected)
        wrong = f"{wrong} ({missing} missing, {extra} extra)"
    if correct:
        return Grade(correct=True, submitted=shown, reason=REASON_CORRECT)
    return Grade(correct=False, submitted=shown, reason=wrong)


def clean(text: str) -> str:
    """Strip whitespace, surrounding quotes and trailing periods, repeatedly.

    Args:
        text: Raw text.

    Returns:
        The cleaned text.
    """
    previous = None
    while previous != text:
        previous = text
        text = text.strip().strip(QUOTE_CHARS).rstrip(".")
    return text


def first_line(text: str) -> str:
    """Return the first non-empty line that is not a Markdown code fence.

    Args:
        text: Raw text.

    Returns:
        The stripped line, or "" when there is none.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not FENCE_RE.match(stripped):
            return stripped
    return ""


def parse_number(text: str) -> float | None:
    """Read "1,234", "$12.50", "12.5" and the like.

    Args:
        text: One answer line.

    Returns:
        The value, or None when the text is not a plain number.
    """
    value = clean(clean(text).strip(CURRENCY_CHARS))
    if not NUMBER_RE.fullmatch(value):
        return None
    return float(value.replace(",", ""))


def normalize_text(text: str) -> str:
    """Put a name or value in a form where formatting no longer matters.

    Args:
        text: Raw text.

    Returns:
        Lower-case text with single spaces and no leading "Dr." or "the".
    """
    collapsed = " ".join(clean(text).split()).casefold()
    return LEADING_TITLE_RE.sub("", collapsed)


def list_items(text: str) -> set[str]:
    """Split a list answer into a set of normalized items.

    Args:
        text: Items separated by commas, semicolons or new lines.

    Returns:
        The items, ignoring order, bullets, numbering and a final "and".
    """
    items = set()
    for part in LIST_SEPARATOR_RE.split(clean(text).strip(LIST_BRACKETS)):
        item = clean(part)
        if not item or FENCE_RE.match(item):
            continue
        item = LEADING_AND_RE.sub("", BULLET_RE.sub("", item))
        normalized = normalize_text(item)
        if normalized:
            items.add(normalized)
    return items
