"""Turn a data question into a training row: token ids and a loss mask.

A row is one chat the model reads, followed by the reply it should write:

- the chat: a system rule, then the database's tables and the question,
  rendered with the model's chat template and ending with the generation
  prompt (``<|im_start|>assistant`` and ``<think>``), exactly what the
  project's engine shows the model before it replies;
- the reply: a one-line reason, ``</think>``, then the gold SQL, and the
  end-of-turn token, as the model would write it.

The **loss mask** says which positions are the reply (1) and which were only
read (0). Training scores only the 1s. ``TrainItem`` and
``item_from_sequence`` are the project's (``mhrl/training/common.py``),
copied unchanged: they turn ``input_ids`` + ``loss_mask`` into the positions
the trainer scores, and they skip position 0, which no position comes before.

This module needs only the standard library: the tokenizer is passed in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from database import schema_text
from grading import THINK_END
from tasks import DATA_DIR, Task

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
# The rule every row starts with. Like chapter 11's prompt, it asks for the
# answer alone, because the answer is read from the result's first column.
SYSTEM_RULE = (
    "You answer questions about a SQLite database with one SQL query. Its result "
    "must be the answer and nothing else: one column, one row per item if the "
    "answer is a list. Reply with the query alone, after your reasoning."
)
USER_TEMPLATE = "The database has these tables:\n\n{schema}\n\n{question}"
# Two kinds of reason. "steps" (the default) describes the gold query; "fixed"
# is one sentence for every row, the line the project's answer-key teacher
# writes before it runs its query (mhrl/engine/oracle.py).
REASON_STEPS = "steps"
REASON_FIXED = "fixed"
REASON_STYLES = (REASON_STEPS, REASON_FIXED)
FIXED_REASON = "Now I can answer the question with one query."
# A reason is written from the gold SQL itself, so every row gets one.
TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", re.IGNORECASE)
GROUP_START = re.compile(r"\bGROUP BY\s+", re.IGNORECASE)
# What ends a GROUP BY's expression, outside any brackets.
GROUP_END = re.compile(r"\s+(?:HAVING|ORDER BY|LIMIT|WHERE)\b", re.IGNORECASE)
JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)
EXCLUDE_PATTERN = re.compile(r"\bNOT\s+(?:IN|EXISTS)\s*\(", re.IGNORECASE)
ORDER_PATTERN = re.compile(r"\bORDER BY\s+(.+?)(?=\s+LIMIT\b|$)", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
WHERE_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)
HAVING_PATTERN = re.compile(r"\bHAVING\b", re.IGNORECASE)
DESCENDING_PATTERN = re.compile(r"\bDESC\b", re.IGNORECASE)
DIRECTION_PATTERN = re.compile(r"\s+(?:ASC|DESC)\b", re.IGNORECASE)
AGGREGATE_PATTERN = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
# What the outer query returns: the text between its SELECT and its first FROM.
SELECT_PATTERN = re.compile(r"^SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM\b", re.IGNORECASE)
COLUMN_PATTERN = re.compile(r"^[A-Za-z_][\w.]*$")  # a plain column, like p.name
# What each aggregate is called when sorting by it, and when it is the answer.
SORT_WORDS = {
    "COUNT": "the count",
    "SUM": "the total",
    "AVG": "the average",
    "MIN": "the smallest",
    "MAX": "the largest",
}
FINAL_WORDS = {
    "COUNT": "count them",
    "SUM": "add them up",
    "AVG": "average them",
    "MIN": "take the smallest",
    "MAX": "take the largest",
}
ONE_ROW = 1


class ChatTokenizer(Protocol):
    """What this module needs from a tokenizer (mlx-lm's, or a test's fake)."""

    eos_token_id: int

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Render a chat with the model's template."""
        ...

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn text into token ids."""
        ...


@dataclass(frozen=True)
class Row:
    """One training row: what the model reads, then the reply it learns.

    Attributes:
        task_id: The question it comes from.
        input_ids: The chat's tokens, then the reply's.
        loss_mask: 1 at the reply's positions, 0 at the chat's.
    """

    task_id: str
    input_ids: list[int]
    loss_mask: list[int]

    @property
    def prompt_length(self) -> int:
        """The number of tokens before the reply.

        Returns:
            The position of the reply's first token.
        """
        return self.loss_mask.index(1) if 1 in self.loss_mask else len(self.loss_mask)

    @property
    def reply_tokens(self) -> int:
        """The number of reply tokens (mask 1).

        Returns:
            The count.
        """
        return sum(self.loss_mask)

    def as_sequence(self) -> dict[str, list[int]]:
        """Give the row in the shape the project's capture proxy records.

        Returns:
            ``{"input_ids": ..., "loss_mask": ...}``.
        """
        return {"input_ids": self.input_ids, "loss_mask": self.loss_mask}


@dataclass
class TrainItem:
    """One training row (the project's, from ``mhrl/training/common.py``).

    Attributes:
        input_ids: Context and sampled tokens.
        positions: Indices of the sampled (trainable) tokens in ``input_ids``.
        old_logprobs: The engine's logprob for each trainable token.
        advantage: Group-relative advantage (1.0 for SFT rows).
    """

    input_ids: list[int]
    positions: list[int]
    old_logprobs: list[float]
    advantage: float = 1.0

    @property
    def n_tokens(self) -> int:
        """Number of trainable tokens."""
        return len(self.positions)


def item_from_sequence(sequence: dict, advantage: float = 1.0) -> TrainItem | None:
    """Turn one captured sequence into a training row (the project's code).

    Args:
        sequence: ``{input_ids, loss_mask, logprobs}`` from the capture proxy.
        advantage: The rollout's advantage.

    Returns:
        The row, or None when it has no trainable tokens.
    """
    ids = sequence["input_ids"]
    mask = sequence["loss_mask"]
    logprobs = sequence.get("logprobs") or [0.0] * len(ids)
    # i > 0: a token is scored from the position before it, and nothing comes
    # before position 0, so the first token can never be a target.
    positions = [i for i, m in enumerate(mask) if m and i > 0]
    if not positions:
        return None
    old = [float(logprobs[i] or 0.0) for i in positions]
    return TrainItem(ids, positions, old, advantage)


def balanced(tasks: list[Task], limit: int) -> list[Task]:
    """Take up to ``limit`` tasks, round-robin over (database, difficulty).

    A copy of the project's ``mhrl/cli.py::balanced`` (via chapter 11), which
    its ``--limit`` flags use: a plain first 40 would be lopsided.

    Args:
        tasks: Candidate tasks, in file order.
        limit: How many to keep.

    Returns:
        A balanced subset, the same for the same input order.
    """
    if limit >= len(tasks):
        return list(tasks)
    buckets: dict[tuple[str, str], list[Task]] = {}
    for task in tasks:
        buckets.setdefault((task.domain, task.difficulty), []).append(task)
    queues = [list(buckets[key]) for key in sorted(buckets)]
    picked: list[Task] = []
    while len(picked) < limit:
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def one_line(sql: str) -> str:
    """Write a query on one line, with single spaces.

    Args:
        sql: The query.

    Returns:
        The same query, its whitespace collapsed.
    """
    return " ".join(sql.split())


def listed(names: list[str]) -> str:
    """Join names in plain English.

    Args:
        names: One or more names.

    Returns:
        Such as ``"a"``, ``"a and b"`` or ``"a, b and c"``.
    """
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def group_expression(sql: str) -> str | None:
    """Find what a query groups by, reading brackets so a subquery ends it.

    Args:
        sql: The query, on one line.

    Returns:
        The GROUP BY expression, such as ``city`` or ``strftime('%m', day)``,
        or None if the query does not group.
    """
    found = GROUP_START.search(sql)
    if found is None:
        return None
    rest = sql[found.end() :]
    depth = 0
    for index, char in enumerate(rest):
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:  # the subquery the GROUP BY sits in has ended
                return rest[:index].strip()
            depth -= 1
        elif depth == 0 and GROUP_END.match(rest, index):
            return rest[:index].strip()
    return rest.strip()


def sort_step(order: str, limit: int | None) -> str:
    """Describe an ORDER BY (and its LIMIT) in words.

    Args:
        order: The ORDER BY clause's expression, such as ``COUNT(*) DESC``.
        limit: The LIMIT's number, or None.

    Returns:
        Such as ``"sort by the count, highest first, and keep the top row"``.
    """
    aggregate = AGGREGATE_PATTERN.search(order)
    if aggregate:
        key = SORT_WORDS[aggregate[1].upper()]
    else:
        key = DIRECTION_PATTERN.sub("", order).strip()
    if limit is None:
        return f"sort by {key}"
    first = "highest" if DESCENDING_PATTERN.search(order) else "lowest"
    keep = "the top row" if limit == ONE_ROW else f"the top {limit} rows"
    return f"sort by {key}, {first} first, and keep {keep}"


def reason_for(gold_sql: str) -> str:
    """Write a one-line reason for a query: which tables, and what it does.

    Args:
        gold_sql: The query.

    Returns:
        Such as ``"Use the customers table: keep the matching rows, then
        count them."``.
    """
    sql = one_line(gold_sql)
    tables = list(dict.fromkeys(TABLE_PATTERN.findall(sql)))
    noun = "table" if len(tables) == 1 else "tables"
    steps = []
    if JOIN_PATTERN.search(sql):
        steps.append("join them")
    if WHERE_PATTERN.search(sql):
        steps.append("keep the matching rows")
    if EXCLUDE_PATTERN.search(sql):
        steps.append("leave out the rows the inner query finds")
    group = group_expression(sql)
    if group:
        steps.append(f"group by {group}")
    if HAVING_PATTERN.search(sql):
        steps.append("keep the groups that qualify")
    order = ORDER_PATTERN.search(sql)
    limit = LIMIT_PATTERN.search(sql)
    returned = SELECT_PATTERN.search(sql)
    column = returned[1] if returned else ""
    aggregate = AGGREGATE_PATTERN.search(column)
    if order:
        steps.append(sort_step(order[1], int(limit[1]) if limit else None))
    elif aggregate:
        steps.append(FINAL_WORDS[aggregate[1].upper()])
    elif COLUMN_PATTERN.match(column):
        steps.append(f"return {column}")
    else:
        steps.append("return the column that answers it")
    said = steps[-1]
    if len(steps) > 1:
        said = ", ".join(steps[:-1]) + ", then " + said
    return f"Use the {listed(tables)} {noun}: {said}."


def reply_text(task: Task, style: str = REASON_STEPS) -> str:
    """Write the reply a row teaches: a reason, the end of thinking, the SQL.

    Args:
        task: The question.
        style: ``"steps"`` (a reason written from the gold SQL) or ``"fixed"``
            (the project's teacher's one sentence).

    Returns:
        ``"<reason></think><gold SQL on one line>"``.

    Raises:
        ValueError: If the style is not one of REASON_STYLES.
    """
    if style not in REASON_STYLES:
        raise ValueError(f"unknown reason style {style!r}")
    reason = reason_for(task.gold_sql) if style == REASON_STEPS else FIXED_REASON
    return f"{reason}{THINK_END}{one_line(task.gold_sql)}"


def chat_messages(task: Task, folder: Path = DATA_DIR) -> list[dict[str, str]]:
    """Write the chat the model reads: the rule, the tables, the question.

    Args:
        task: The question.
        folder: The data folder holding its database.

    Returns:
        A system message and a user message.
    """
    schema = schema_text(folder / task.db_file)
    question = USER_TEMPLATE.format(schema=schema, question=task.question)
    return [
        {"role": SYSTEM_ROLE, "content": SYSTEM_RULE},
        {"role": USER_ROLE, "content": question},
    ]


def encode(tokenizer: ChatTokenizer, text: str) -> list[int]:
    """Turn text into token ids, adding nothing (the template wrote the markers).

    Args:
        tokenizer: The model's tokenizer.
        text: The text.

    Returns:
        Its token ids.
    """
    return list(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> str:
    """Render a chat into the text the model reads, ready for its reply.

    Args:
        tokenizer: The model's tokenizer.
        messages: The chat so far.

    Returns:
        The text, ending with the generation prompt
        (``<|im_start|>assistant`` and ``<think>``).
    """
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # end with the assistant's turn opening
        tokenize=False,
        preserve_thinking=True,  # keep reasoning, as the project's engine renders
    )


def render_finished(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> str:
    """Render a whole chat, the assistant's reply included, with the template.

    Args:
        tokenizer: The model's tokenizer.
        messages: The chat, ending with the assistant's reply.

    Returns:
        The text the template makes of it.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, preserve_thinking=True
    )


def prompt_ids(
    tokenizer: ChatTokenizer, task: Task, folder: Path = DATA_DIR
) -> list[int]:
    """Render a question's chat into the token ids the model reads.

    Args:
        tokenizer: The model's tokenizer.
        task: The question.
        folder: The data folder.

    Returns:
        The prompt's token ids, ending with ``<think>``.
    """
    return encode(tokenizer, render_prompt(tokenizer, chat_messages(task, folder)))


def build_row(
    tokenizer: ChatTokenizer,
    task: Task,
    folder: Path = DATA_DIR,
    style: str = REASON_STEPS,
) -> Row:
    """Build one training row, as the project's engine would record it.

    The row is the prompt the model read, then the tokens it wrote: the
    reply's tokens and the end-of-turn token. The chat is never rendered
    again with the reply in it (see ``template_difference``).

    Args:
        tokenizer: The model's tokenizer.
        task: The question.
        folder: The data folder.
        style: Which kind of reason the reply starts with.

    Returns:
        The row, with 0s over the prompt and 1s over the reply.
    """
    prompt = prompt_ids(tokenizer, task, folder)
    reply = encode(tokenizer, reply_text(task, style)) + [tokenizer.eos_token_id]
    return Row(task.task_id, prompt + reply, [0] * len(prompt) + [1] * len(reply))


def template_difference(
    tokenizer: ChatTokenizer,
    task: Task,
    row: Row,
    folder: Path = DATA_DIR,
    style: str = REASON_STEPS,
) -> tuple[list[int], int]:
    """Render the finished chat with the template, and compare it with the row.

    Args:
        tokenizer: The model's tokenizer.
        task: The question.
        row: Its row from ``build_row``.
        folder: The data folder.
        style: The kind of reason the row's reply starts with.

    Returns:
        The template's token ids, and how many positions of the row they
        share from the start before the first difference.
    """
    messages = chat_messages(task, folder)
    messages.append({"role": ASSISTANT_ROLE, "content": reply_text(task, style)})
    rendered = encode(tokenizer, render_finished(tokenizer, messages))
    shared = 0
    for mine, theirs in zip(row.input_ids, rendered, strict=False):
        if mine != theirs:
            break
        shared += 1
    return rendered, shared


def write_rows(path: Path, rows: Iterable[Row]) -> int:
    """Write rows as JSON lines, replacing the file.

    Args:
        path: The file.
        rows: The rows.

    Returns:
        How many rows were written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {"task_id": row.task_id, **row.as_sequence()}
            handle.write(json.dumps(record) + "\n")
            count += 1
    return count


def read_rows(path: Path) -> list[Row]:
    """Read rows written by ``write_rows``.

    Args:
        path: The file.

    Returns:
        The rows, in file order.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a line is not JSON or not a row.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            row = Row(record["task_id"], record["input_ids"], record["loss_mask"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path.name} has a line that is not a row") from error
        if len(row.input_ids) != len(row.loss_mask):
            raise ValueError(f"{path.name}: {row.task_id}'s mask is the wrong length")
        rows.append(row)
    return rows
