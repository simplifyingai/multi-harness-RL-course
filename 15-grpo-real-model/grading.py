"""Grade one reply: find its SQL, run it, write answer.txt, ask chapter 1's grader.

The reply's query is whatever follows its last ``</think>``, in one of three
forms, and which one is recorded:

- a **tool call** in LFM2.5's own syntax,
  ``<|tool_call_start|>[sql('SELECT ...')]<|tool_call_end|>``: given a system
  rule, the base model answers this way even though no tool exists. The call
  is parsed with ``ast`` as the project's engine parses tool calls
  (``mhrl/engine/toolcalls.py``), and its first string argument is the query;
- a fenced **code block**;
- **plain text**, the shape the training rows teach.

The query runs on a read-only
copy of the question's database; its first column becomes an answer the way
chapter 1's ``make_tasks.py`` formats one (copied from chapter 11's
``sql_questions.py``); that answer is written to ``answer.txt`` in the
question's own folder, and chapter 1's ``grade`` reads it from there, as it
reads an agent's.

This module needs only the standard library.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from database import QueryError, run_query
from grader import grade
from tasks import ANSWER_FILE, DATA_DIR, LIST, NUMBER, Task

THINK_END = "</think>"
SQL_BLOCK_RE = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_BLOCK_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)
END_OF_TURN = "<|im_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL
)
FINISH_STOP = "stop"  # the model wrote its end-of-turn token
FINISH_LENGTH = "length"  # the length cap cut the reply off
FORM_TOOL = "tool call"
FORM_BLOCK = "code block"
FORM_TEXT = "plain text"
DECIMALS = 2  # chapter 1's make_tasks.py writes float answers to 2 places

REASON_NO_SQL = "no SQL in the reply"
REASON_NO_ROWS = "the query returned nothing"
REASON_CUT_OFF = "cut off by the length cap while still reasoning"


@dataclass(frozen=True)
class Verdict:
    """The verdict on one reply.

    Attributes:
        correct: Whether the answer matches.
        sql: The query found in the reply, or None.
        answer: What was written to answer.txt, or None if nothing was.
        reason: ``"correct"`` or why the reply was wrong.
        form: ``"tool call"``, ``"code block"`` or ``"plain text"``: how the
            query was written, or None when there was none.
    """

    correct: bool
    sql: str | None
    answer: str | None
    reason: str
    form: str | None = None


def tool_call_sql(source: str) -> str | None:
    """Read the query out of one tool call, without running anything.

    Args:
        source: The text between the tool-call markers, such as
            ``[sql('SELECT 1')]``.

    Returns:
        The first string argument of the first call, or None if the text is
        not a call with a string argument.
    """
    try:
        tree = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        return None
    nodes = tree.elts if isinstance(tree, ast.List | ast.Tuple) else [tree]
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            try:
                value = ast.literal_eval(argument)  # a literal only: never runs code
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def extract_sql(reply: str) -> tuple[str | None, str | None]:
    """Find the query in a reply: what follows its last ``</think>``.

    Args:
        reply: The model's whole reply, reasoning included.

    Returns:
        The query and the form it came in, or ``(None, None)`` if nothing
        follows the reasoning.
    """
    answer = reply.rsplit(THINK_END, 1)[-1].replace(END_OF_TURN, "")
    for call in reversed(TOOL_BLOCK_RE.findall(answer)):
        sql = tool_call_sql(call)
        if sql:
            return sql, FORM_TOOL
    for pattern in (SQL_BLOCK_RE, ANY_BLOCK_RE):
        blocks = pattern.findall(answer)
        if blocks and blocks[-1].strip():
            return blocks[-1].strip(), FORM_BLOCK
    text = answer.strip()
    return (text, FORM_TEXT) if text else (None, None)


def format_answer(rows: list[tuple[Any, ...]], answer_type: str) -> str | None:
    """Turn a query result into answer text, as chapter 1's make_tasks.py does.

    Args:
        rows: The rows the query returned.
        answer_type: ``"number"``, ``"text"`` or ``"list"``.

    Returns:
        The answer text, or None when the result is empty.
    """
    values = [row[0] for row in rows if row and row[0] is not None]
    if not values:
        return None
    if answer_type == LIST:
        return ", ".join(str(value) for value in values)
    value = values[0]
    if answer_type == NUMBER and isinstance(value, float):
        return f"{value:.{DECIMALS}f}"
    return str(value)


def fresh_folder(folder: Path) -> Path:
    """Empty a question's folder, so an old answer.txt cannot be graded again.

    Args:
        folder: The folder.

    Returns:
        The folder, existing and empty.

    Raises:
        OSError: If the folder cannot be removed or made.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def grade_reply(
    task: Task,
    reply: str,
    workdir: Path,
    cut_off: bool = False,
    data: Path = DATA_DIR,
) -> Verdict:
    """Run a reply's SQL, write its result to answer.txt, and grade that file.

    Args:
        task: The question the reply answers.
        reply: The model's whole reply.
        workdir: The question's own folder (emptied first).
        cut_off: True if the length cap ended the reply. A reply cut off
            before its reasoning ended has no answer yet: a query drafted
            inside the reasoning is not taken as one.
        data: The data folder holding the databases.

    Returns:
        The verdict.

    Raises:
        OSError: If the question's folder or answer.txt cannot be written.
    """
    fresh_folder(workdir)
    if cut_off and THINK_END not in reply:
        return Verdict(False, None, None, REASON_CUT_OFF)
    sql, form = extract_sql(reply)
    if sql is None:
        return Verdict(False, None, None, REASON_NO_SQL)
    try:
        rows = run_query(data / task.db_file, sql)
    except QueryError as error:
        return Verdict(False, sql, None, str(error), form)
    answer = format_answer(rows, task.answer_type)
    if answer is None:
        return Verdict(False, sql, None, REASON_NO_ROWS, form)
    (workdir / ANSWER_FILE).write_text(answer + "\n", encoding="utf-8")
    result = grade(task, workdir)  # chapter 1's grader reads answer.txt
    return Verdict(result.correct, sql, answer, result.reason, form)


@dataclass(frozen=True)
class Answered:
    """One question answered and graded, as eval_sql.py saves it.

    Attributes:
        task_id: The question.
        correct: Whether chapter 1's grader accepted answer.txt.
        reason: ``"correct"`` or why not.
        sql: The query found in the reply, or None.
        answer: What was written to answer.txt, or None.
        reply: The whole reply, as text.
        tokens: How many tokens the reply has.
        finish_reason: ``"stop"`` or ``"length"``.
        seconds: How long writing the reply took (varies from run to run).
        form: How the query was written (see ``Verdict``), or None.
        cap: The reply length cap it was answered under.
        adapter: A fingerprint of the adapter file it was answered with
            (``"none"`` for the base model), so saved answers are reused only
            for the same adapter bytes.
    """

    task_id: str
    correct: bool
    reason: str
    sql: str | None
    answer: str | None
    reply: str
    tokens: int
    finish_reason: str
    seconds: float
    form: str | None = None
    cap: int | None = None
    adapter: str | None = None


def write_answers(path: Path, answers: list[Answered]) -> None:
    """Save an evaluation's answers as JSON lines, keeping comparable ones.

    The file's answers to other questions are kept when they were made under
    the same cap with the same adapter, so a 10-question run never erases a
    50-question one; anything else in the file is replaced.

    Args:
        path: The file.
        answers: One per question, all under one cap and one adapter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: dict[str, Answered] = {}
    if answers:
        same = (answers[0].cap, answers[0].adapter)
        saved = read_answers(path).items()
        kept = {i: a for i, a in saved if (a.cap, a.adapter) == same}
    kept.update({answered.task_id: answered for answered in answers})
    lines = [json.dumps(asdict(answered)) for answered in kept.values()]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def read_answers(path: Path) -> dict[str, Answered]:
    """Read an evaluation saved by ``write_answers``.

    Args:
        path: The file.

    Returns:
        Each question's answer, by task id; empty if the file does not exist
        or is not an evaluation file.
    """
    if not path.is_file():
        return {}
    answers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            answered = Answered(**json.loads(line))
        except (json.JSONDecodeError, TypeError):
            return {}  # an older or damaged file: answer the questions again
        answers[answered.task_id] = answered
    return answers
