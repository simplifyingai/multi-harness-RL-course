"""What a GRPO run leaves behind, the numbers read from it, and the length guard.

Every rollout of a GRPO run is one line of ``rollouts.jsonl`` (the project's
``RolloutRecord``, ``mhrl/rollout.py``) and every step one line of
``metrics.jsonl``. This module reads both, from this chapter's runs, from the
project's own files (``--records``), or from the slim copies bundled in
``data/project_grpo/``, and defines each number this chapter prints ONCE:

- **sampled tokens** of a rollout: the tokens the model generated across all
  its calls, the project's ``n_sampled``. The first recorded run (attempt 1)
  predates that field, so for it they are counted from the captured rows:
  each reply's tokens, every shared prefix once (what ``dedupe_shared_
  prefixes`` keeps). On all 48 rollouts of attempt 2, which has both, the
  count equals ``n_sampled`` exactly (the tests check it);
- **tokens per call** of a rollout: its sampled tokens / its model calls;
- **tokens per call of a step**: the MEDIAN over the step's rollouts of their
  tokens per call. The length guard reads this one, as the project's
  ``train_grpo`` does. The MEAN over the same rollouts is shown beside it,
  labelled, because one ballooned rollout moves the mean and not the median;
- **what ``mhrl.cli steps`` printed**: the mean, and for records without
  ``n_sampled`` only each rollout's FIRST captured row (the project's
  ``print_steps``). It is reproduced here only to explain the numbers in the
  project's logs, never used for anything else.

The **length guard** is the project's: when a step's rollouts come back with
tokens per call above the guard (150), the update that made them is the one
that went wrong, so training stops and keeps the adapter from before it.

The bundled copies keep only what these numbers need: no prompt, no reply
text, no answer and no path, just each row's length and where each reply sits
in it (its start, its length, whether it holds a tool call, whether it ended
with the end-of-turn token). This module needs only the standard library.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lfm_files import short_path
from rewards import group_advantages

HERE = Path(__file__).resolve().parent
BUNDLE_DIR = HERE / "data" / "project_grpo"
LOGS_DIR = HERE / "data" / "project_logs"  # the project's printed outputs, summaries
ROLLOUTS_FILE = "rollouts.jsonl"
METRICS_FILE = "metrics.jsonl"
# LFM2.5's special tokens (its tokenizer.json, added_tokens): a reply that
# calls a tool holds <|tool_call_start|>, and a reply that ended by itself
# ends with <|im_end|>. A reply cut by the 1,024-token cap has no <|im_end|>.
TOOL_CALL_START_ID = 124_905
END_OF_TURN_ID = 124_900
REPLY_CAP = 1_024  # the project's max_completion_tokens: one reply's limit
ROW_CAP = 6_144  # the project's MAX_TRAIN_TOKENS (training.py): longer rows skipped
GUARD_TOKENS = 150  # the project's length_guard_tokens (mhrl/config.py)
GROUP_SIZE = 4  # rollouts per task (the project's --group)
TASKS_PER_STEP = 2  # tasks per step (the project's --tasks-per-step)
ROLLOUTS_PER_STEP = GROUP_SIZE * TASKS_PER_STEP
STAGE = "grpo"
ADAPTER_STEP_RE = re.compile(r"adapter_step(\d+)")  # adapter_step004.safetensors
OUTPUT_FOLDER = "grpo"  # grpo_harness.py's runs: COURSE_HOME/26-grpo-in-harnesses/grpo/
RUN_NAME = "grpo"  # its default run
SECONDS_DIGITS = 3  # rollout times are kept to the millisecond


class RecordsError(Exception):
    """Raised when a run's files are missing, empty or not a GRPO run's."""


@dataclass(frozen=True)
class Reply:
    """One model reply inside a captured row: a run of mask-1 tokens.

    Attributes:
        start: Its first token's position in the row.
        length: How many tokens the model sampled for it.
        tool_call: Whether it holds a tool call.
        ended: Whether it ended with the end-of-turn token (False when the
            1,024-token cap cut it).
    """

    start: int
    length: int
    tool_call: bool
    ended: bool

    @property
    def capped(self) -> bool:
        """Whether the reply reached the 1,024-token cap."""
        return self.length >= REPLY_CAP


@dataclass(frozen=True)
class Row:
    """One captured row of a rollout, without its tokens.

    Attributes:
        tokens: The row's length (prompt, tool results and replies).
        shared: How many of its first tokens an earlier row of the same
            rollout already holds (a forked rollout repeats its early turns).
        replies: The model's replies in it, in order.
    """

    tokens: int
    shared: int
    replies: tuple[Reply, ...]

    def new_tokens(self) -> int:
        """Sampled tokens of this row that no earlier row holds.

        Returns:
            The reply tokens at or past ``shared``.
        """
        return sum(
            max(0, reply.start + reply.length - max(reply.start, self.shared))
            for reply in self.replies
        )


@dataclass(frozen=True)
class Rollout:
    """One rollout as this chapter reads it.

    Attributes:
        step: The step it belongs to, from 1.
        task_id: The task.
        harness: The harness.
        correct: Whether the grader accepted ``answer.txt``.
        n_calls: The harness's model calls.
        n_sampled: Tokens the model generated, or None when the record
            predates the field (attempt 1).
        reward: The reward GRPO trained on.
        seconds: Wall time.
        rows: Its captured rows, in the proxy's order.
    """

    step: int
    task_id: str
    harness: str
    correct: bool
    n_calls: int
    n_sampled: int | None
    reward: float
    seconds: float
    rows: tuple[Row, ...] = field(default_factory=tuple)

    def sampled_tokens(self) -> int:
        """The tokens the model generated across all its calls.

        Returns:
            ``n_sampled``, or for older records the captured rows' reply
            tokens with every shared prefix counted once.
        """
        if self.n_sampled is not None:
            return self.n_sampled
        return sum(row.new_tokens() for row in self.rows)

    def tokens_per_call(self) -> float | None:
        """Sampled tokens per model call.

        Returns:
            The ratio, or None for a rollout that made no call.
        """
        return self.sampled_tokens() / self.n_calls if self.n_calls else None

    def replies(self) -> list[Reply]:
        """Every reply the rollout's rows hold, each shared prefix once.

        Returns:
            The replies that start at or past their row's shared prefix.
        """
        return [
            reply
            for row in self.rows
            for reply in row.replies
            if reply.start + reply.length > row.shared
        ]


@dataclass(frozen=True)
class StepLength:
    """Tokens per call over one step's rollouts, both ways.

    Attributes:
        median: The median over the rollouts (the guard reads this).
        mean: The mean over the same rollouts.
        rollouts: How many rollouts made at least one call.
    """

    median: float
    mean: float
    rollouts: int


@dataclass
class Exposure:
    """What one harness gave GRPO's updates.

    Attributes:
        groups: Its groups.
        contrast: Of those, groups with contrast (the others teach nothing).
        rows: Rows trained.
        tokens: Their total length: everything the trainer reads.
        trained: Their trained (sampled) tokens.
    """

    groups: int = 0
    contrast: int = 0
    rows: int = 0
    tokens: int = 0
    trained: int = 0

    def add(self, other: Exposure) -> None:
        """Add another step's counts to these.

        Args:
            other: The other counts.
        """
        self.groups += other.groups
        self.contrast += other.contrast
        self.rows += other.rows
        self.tokens += other.tokens
        self.trained += other.trained


@dataclass(frozen=True)
class GuardVerdict:
    """What the length guard decided after one step's rollouts.

    Attributes:
        stop: True when tokens per call passed the guard.
        median: The step's tokens per call (the median).
        kept: The adapter training keeps (the one before the last update).
        blamed: The adapter whose rollouts these were (the last update's).
    """

    stop: bool
    median: float
    kept: Path | str | None = None
    blamed: Path | str | None = None


def step_length(rollouts: Iterable[Rollout]) -> StepLength:
    """Tokens per call of a step: the median and the mean over its rollouts.

    Args:
        rollouts: The step's rollouts.

    Returns:
        Both statistics (0 when no rollout made a call).
    """
    per_call = [
        value for value in (r.tokens_per_call() for r in rollouts) if value is not None
    ]
    if not per_call:
        return StepLength(0.0, 0.0, 0)
    return StepLength(
        statistics.median(per_call), statistics.fmean(per_call), len(per_call)
    )


def length_guard(
    rollouts: Iterable[Rollout],
    history: Sequence[Path | str | None],
    guard_tokens: float = GUARD_TOKENS,
) -> GuardVerdict:
    """The project's length guard, on one step's rollouts.

    The rollouts were sampled with the newest adapter, so if they ballooned,
    the newest update is to blame: keep the one before it.

    Args:
        rollouts: The step's rollouts.
        history: Every adapter so far, oldest first: the starting one, then
            one per update. The last one sampled these rollouts.
        guard_tokens: The most tokens per call (the median) allowed.

    Returns:
        The verdict: whether to stop, and which adapter to keep.
    """
    median = step_length(rollouts).median
    if median <= guard_tokens:
        return GuardVerdict(False, median)
    # As the project's train_grpo: with no update yet, keep the start.
    kept = history[-2] if len(history) > 1 else history[-1]
    return GuardVerdict(True, median, kept=kept, blamed=history[-1])


def steps_printed(rollouts: Sequence[Rollout]) -> float:
    """The tokens-per-call figure the project's ``mhrl.cli steps`` printed.

    Its ``print_steps``: the MEAN over rollouts, each rollout counted as
    ``n_sampled`` or, for records without it, its first captured row's
    sampled tokens only. Reproduced to explain the project's logs.

    Args:
        rollouts: One step's rollouts.

    Returns:
        That mean (0 when no rollout counts).
    """
    per_call = []
    for rollout in rollouts:
        if not rollout.n_calls or not (rollout.n_sampled or rollout.rows):
            continue
        first = sum(reply.length for reply in rollout.rows[0].replies)
        per_call.append((rollout.n_sampled or first) / rollout.n_calls)
    return statistics.fmean(per_call) if per_call else 0.0


def by_step(rollouts: Sequence[Rollout]) -> list[list[Rollout]]:
    """Group rollouts by their step, in step order.

    Args:
        rollouts: A run's rollouts.

    Returns:
        One list per step.
    """
    steps: dict[int, list[Rollout]] = {}
    for rollout in rollouts:
        steps.setdefault(rollout.step, []).append(rollout)
    return [steps[number] for number in sorted(steps)]


def exposure(
    step: Sequence[Rollout], group_size: int = GROUP_SIZE, row_cap: int = ROW_CAP
) -> dict[str, Exposure]:
    """What each harness gave one step's update, read from its records alone.

    The rows ``grpo.step_rows`` trains on: only groups with contrast, every
    shared prefix once, rows over the cap left out. On both recorded runs
    the rows and trained tokens equal the project's ``train_rows`` and
    ``train_tokens`` at every step (the tests check it).

    Args:
        step: One step's rollouts, group by group.
        group_size: Rollouts per group.
        row_cap: Rows longer than this are left out.

    Returns:
        Per harness, what it gave.
    """
    out: dict[str, Exposure] = {}
    for start in range(0, len(step), group_size):
        group = step[start : start + group_size]
        mine = out.setdefault(group[0].harness, Exposure())
        mine.groups += 1
        if group_advantages([rollout.reward for rollout in group]) is None:
            continue
        mine.contrast += 1
        for rollout in group:
            for row in rollout.rows:
                trained = row.new_tokens()
                if trained == 0 or row.tokens > row_cap:
                    continue
                mine.rows += 1
                mine.tokens += row.tokens
                mine.trained += trained
    return out


# --------------------------------------------------------------- reading


def shared_prefixes(sequences: Sequence[dict[str, Any]]) -> list[int]:
    """For each captured row, the longest start an earlier row also has.

    The comparison ``dedupe_shared_prefixes`` makes (chapter 21), kept as a
    number so the slim copy needs no token ids.

    Args:
        sequences: The rollout's rows, each with ``input_ids``.

    Returns:
        One length per row (0 for the first).
    """
    seen: list[list[int]] = []
    out = []
    for sequence in sequences:
        ids = sequence["input_ids"]
        shared = 0
        for earlier in seen:
            limit = min(len(ids), len(earlier))
            length = 0
            while length < limit and ids[length] == earlier[length]:
                length += 1
            shared = max(shared, length)
        out.append(shared)
        seen.append(ids)
    return out


def replies_of(sequence: dict[str, Any]) -> list[Reply]:
    """Find the model's replies in a row: each run of mask-1 tokens.

    Args:
        sequence: ``{input_ids, loss_mask}``.

    Returns:
        The replies, in order.
    """
    ids, mask = sequence["input_ids"], sequence["loss_mask"]
    found, start = [], None
    for index, value in enumerate([*mask, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            tokens = ids[start:index]
            found.append(
                Reply(
                    start=start,
                    length=index - start,
                    tool_call=TOOL_CALL_START_ID in tokens,
                    ended=tokens[-1] == END_OF_TURN_ID,
                )
            )
            start = None
    return found


def slim(record: dict[str, Any], step: int) -> dict[str, Any]:
    """Cut a full rollout record to what this chapter reads.

    Args:
        record: A record as the project (or ``grpo_harness.py``) wrote it,
            with its ``sequences``.
        step: The step it belongs to.

    Returns:
        The slim record: numbers, the task and the harness, and each row's
        length and replies, never a token id or any text.
    """
    sequences = record.get("sequences") or []
    rows = [
        {
            "tokens": len(sequence["input_ids"]),
            "shared": shared,
            "replies": [
                [reply.start, reply.length, reply.tool_call, reply.ended]
                for reply in replies_of(sequence)
            ],
        }
        for sequence, shared in zip(sequences, shared_prefixes(sequences), strict=True)
    ]
    return {
        "step": step,
        "task_id": record["task_id"],
        "harness": record["harness"],
        "correct": bool(record["correct"]),
        "n_calls": int(record["n_calls"]),
        "n_sampled": record.get("n_sampled"),
        "reward": float(record["reward"]),
        "seconds": round(float(record["seconds"]), SECONDS_DIGITS),
        "rows": rows,
    }


def from_slim(data: dict[str, Any]) -> Rollout:
    """Build a rollout from its slim record.

    Args:
        data: A slim record (``slim``'s output).

    Returns:
        The rollout.
    """
    rows = tuple(
        Row(
            tokens=row["tokens"],
            shared=row["shared"],
            replies=tuple(Reply(*reply) for reply in row["replies"]),
        )
        for row in data["rows"]
    )
    return Rollout(
        step=data["step"],
        task_id=data["task_id"],
        harness=data["harness"],
        correct=data["correct"],
        n_calls=data["n_calls"],
        n_sampled=data["n_sampled"],
        reward=data["reward"],
        seconds=data["seconds"],
        rows=rows,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file that must exist and hold records.

    Args:
        path: The file.

    Returns:
        Its records.

    Raises:
        RecordsError: If it is missing, empty, or a line is not JSON.
    """
    if not path.is_file():
        raise RecordsError(f"no file {short_path(path)}")
    try:  # a UnicodeDecodeError and a JSONDecodeError are both ValueErrors
        text = path.read_text("utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
    except ValueError as error:
        raise RecordsError(f"{short_path(path)}: not JSON lines ({error})") from error
    if not records:
        raise RecordsError(f"{short_path(path)} holds no records")
    return records


def read_metrics(folder: Path) -> list[dict[str, Any]]:
    """Read a run's step metrics (the lines that describe a step).

    Args:
        folder: The run's folder.

    Returns:
        One record per step, in order (a guard's stop line is left out).

    Raises:
        RecordsError: If the file is missing or holds no step.
    """
    rows = [
        row
        for row in read_jsonl(folder / METRICS_FILE)
        if row.get("stage") == STAGE and "accuracy" in row
    ]
    if not rows:
        raise RecordsError(f"{short_path(folder / METRICS_FILE)} holds no GRPO step")
    return rows


def rollouts_per_step(folder: Path, records: int) -> int:
    """How many rollouts each step ran, from the run's metrics.

    Every step that ran its rollouts left a line: a step's metrics, or the
    guard's stop line (whose rollouts were recorded too).

    Args:
        folder: The run's folder.
        records: How many rollouts it recorded.

    Returns:
        Rollouts per step.

    Raises:
        RecordsError: If the metrics hold no step, or do not divide the
            rollouts evenly.
    """
    steps = [
        row
        for row in read_jsonl(folder / METRICS_FILE)
        if row.get("stage") == STAGE and "step" in row
    ]
    if not steps or records % len(steps):
        raise RecordsError(
            f"{short_path(folder)}: {records} rollouts do not split evenly into "
            f"{len(steps)} steps"
        )
    return records // len(steps)


def read_rollouts(folder: Path, per_step: int | None = None) -> list[Rollout]:
    """Read a run's rollouts, full records or slim ones.

    A full record (``sequences``) is slimmed as it is read; one without a
    ``step`` gets it from its position, ``per_step`` rollouts to a step, as
    the project's ``print_steps`` does.

    Args:
        folder: The run's folder.
        per_step: Rollouts per step, for records without ``step``; None
            works it out from the metrics (``len(rollouts) // steps``).

    Returns:
        The rollouts, in file order.

    Raises:
        RecordsError: If a file is missing or a record is not a rollout.
    """
    records = read_jsonl(folder / ROLLOUTS_FILE)
    if per_step is None and any("step" not in r for r in records):
        per_step = rollouts_per_step(folder, len(records))
    out = []
    for index, record in enumerate(records):
        try:
            step = record.get("step") or index // (per_step or 1) + 1
            data = record if "rows" in record else slim(record, step)
            out.append(from_slim({**data, "step": step}))
        except (KeyError, TypeError, ValueError) as error:
            raise RecordsError(
                f"{short_path(folder / ROLLOUTS_FILE)}: record {index + 1} is not a "
                f"rollout "
                f"({error!r})"
            ) from error
    return out


def write_slim(source: Path, target: Path, say: Callable[[str], None]) -> None:
    """Write a run's slim copy: its metrics unchanged, its rollouts slimmed.

    Args:
        source: The run's folder (the project's ``runs/main/grpo``).
        target: The bundle folder to write.
        say: Where to report what was written.
    """
    rollouts = read_jsonl(source / ROLLOUTS_FILE)
    steps = len(read_metrics(source))
    per_step = rollouts_per_step(source, len(rollouts))
    target.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(slim(record, index // per_step + 1))
        for index, record in enumerate(rollouts)
    ]
    (target / ROLLOUTS_FILE).write_text("\n".join(lines) + "\n", "utf-8")
    metrics = (source / METRICS_FILE).read_bytes()
    (target / METRICS_FILE).write_bytes(metrics)
    say(f"wrote {short_path(target)}: {len(lines)} rollouts, {steps} steps")


def named_adapter(summary: Path) -> str:
    """The adapter file an evaluation summary names (the project's summary_*.json).

    Args:
        summary: The summary file.

    Returns:
        Its ``adapter`` field, as the project wrote it.

    Raises:
        RecordsError: If the summary is missing or names no adapter.
    """
    try:
        adapter = json.loads(summary.read_text("utf-8"))["adapter"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RecordsError(f"{short_path(summary)}: no adapter named") from error
    if not isinstance(adapter, str) or not adapter:
        raise RecordsError(f"{short_path(summary)}: no adapter named")
    return adapter


def adapter_step(name: str) -> int:
    """The step whose update saved an adapter, read from its name.

    Args:
        name: Such as ``adapter_step004`` or ``adapter_step004.safetensors``.

    Returns:
        Its step, such as 4.

    Raises:
        RecordsError: If the name holds no step number.
    """
    found = ADAPTER_STEP_RE.search(name)
    if found is None:
        raise RecordsError(f"{name} names no step's adapter")
    return int(found[1])
