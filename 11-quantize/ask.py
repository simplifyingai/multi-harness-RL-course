# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Ask our 4-bit LFM2.5-2.6B a question on this laptop, and see what it costs.

The model is our own build from convert.py (Liquid's export when ours is not
built yet), the one every later chapter loads. The question is rendered with
the chat template and the reply is printed token by token, greedily, as
chapter 10's first_reply.py did with the bfloat16 original; then the time per
token and the peak memory, beside chapter 10's saved "before" picture of the
same question. Last, the budget that matters on a Mac: its unified memory,
one pool the CPU and the GPU share, and how much of it the GPU may use.

Run it with ``uv run ask.py "How many customers live in Paris? Answer with
one SQL query."`` after ``uv run convert.py``. ``--fast`` uses Liquid's
export instead.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import report
from lfm import LoadedModel, chat_prompt, load_folder
from lfm_files import (
    CHAPTER_10,
    FINISH_LENGTH,
    FINISH_STOP,
    FOUR_BIT,
    MILLISECONDS,
    course_home,
    model_for_later_chapters,
    require_repo,
)
from mlx_lm import stream_generate
from sql_questions import extract_sql

SCRIPT = "ask.py"
TITLE = "our 4-bit LFM2.5-2.6B answers, on this laptop"
QUESTION = "How many customers live in Paris? Answer with one SQL query."
MAX_TOKENS = 1_024  # the project's output cap in training and evaluation
MIN_TOKENS = 1  # the smallest length cap mlx-lm accepts
WARM_UP_TOKENS = 1
# Chapter 10's first_reply.py saved its bfloat16 run here: the "before" picture.
BEFORE_FILE = "first_reply-bfloat16.json"
# The fields of that file this script reads; a file without them is not used.
BEFORE_KEYS = (
    "question",
    "reply",
    "weight_bytes",
    "peak_memory_bytes",
    "ms_per_token",
    "reply_tokens",
)
DEVICE_MEMORY_KEY = "memory_size"
GPU_LIMIT_KEY = "max_recommended_working_set_size"


class Step(Protocol):
    """What ``stream_generate`` yields for each step (its GenerationResponse)."""

    text: str
    token: int
    finish_reason: str | None


@dataclass(frozen=True)
class Timed:
    """A whole reply, timed token by token.

    Attributes:
        prompt_tokens: The prompt's length.
        texts: Each reply token's text, in order (end of turn not included).
        seconds: Each reply token's time; the first includes reading the prompt.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        end_seconds: The end-of-turn step's time (0 if the cap stopped it).
        peak_bytes: MLX's peak memory over loading and answering.
    """

    prompt_tokens: int
    texts: list[str]
    seconds: list[float]
    finish_reason: str
    end_seconds: float
    peak_bytes: int

    def text(self) -> str:
        """Join the reply's pieces.

        Returns:
            The reply's text.
        """
        return "".join(self.texts)

    def ms_per_token(self) -> float:
        """Average time per token once the prompt is read (chapter 10's measure).

        Returns:
            Milliseconds per token, or 0 for a one-token reply.
        """
        steps = self.seconds[1:]
        if self.finish_reason == FINISH_STOP:
            steps = [*steps, self.end_seconds]
        return MILLISECONDS * sum(steps) / len(steps) if steps else 0.0


@dataclass(frozen=True)
class Budget:
    """This Mac's memory, as Metal reports it.

    Attributes:
        total_bytes: All of its unified memory.
        gpu_limit_bytes: How much of it Metal recommends the GPU keep in use.
    """

    total_bytes: int
    gpu_limit_bytes: int


def collect(
    steps: Iterable[Step],
    show: Callable[[str], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[str], list[float], str, float]:
    """Time each step of a generation and show its text the moment it arrives.

    Args:
        steps: The generation's steps.
        show: Prints a piece of text at once.
        clock: Seconds from a fixed point.

    Returns:
        The reply's pieces, their times, the finish reason and the end-of-turn
        step's time.
    """
    texts: list[str] = []
    seconds: list[float] = []
    finish, end_seconds = FINISH_LENGTH, 0.0
    last = clock()
    for step in steps:
        now = clock()
        took, last = now - last, now
        show(step.text)
        if step.finish_reason == FINISH_STOP:
            finish, end_seconds = FINISH_STOP, took  # the end of turn, not the reply
            continue
        texts.append(step.text)
        seconds.append(took)
    return texts, seconds, finish, end_seconds


def warm_up(loaded: LoadedModel, ids: list[int]) -> float:
    """Run one untimed token: the first pass after loading is slower (chapter 10).

    Args:
        loaded: The model.
        ids: The prompt.

    Returns:
        Its duration in seconds.
    """
    start = time.perf_counter()
    for _ in stream_generate(
        loaded.model, loaded.tokenizer, ids, max_tokens=WARM_UP_TOKENS
    ):
        pass
    return time.perf_counter() - start


def answer(loaded: LoadedModel, question: str, max_tokens: int) -> Timed:
    """Generate the reply token by token, printing each piece as it comes.

    Args:
        loaded: The model.
        question: What the user asks.
        max_tokens: The length cap.

    Returns:
        The timed reply.
    """
    ids = chat_prompt(loaded.tokenizer, question)
    report.print_warm_up(warm_up(loaded, ids))
    report.print_question(question, len(ids))
    steps = stream_generate(loaded.model, loaded.tokenizer, ids, max_tokens=max_tokens)
    texts, seconds, finish, end_seconds = collect(steps, report.show_piece)
    report.end_stream()
    return Timed(len(ids), texts, seconds, finish, end_seconds, mx.get_peak_memory())


def read_before(question: str, home: Path) -> dict[str, Any] | None:
    """Read chapter 10's saved bfloat16 run, if it asked the same question.

    Args:
        question: The question asked here.
        home: The course folder.

    Returns:
        Chapter 10's measurement, or None if it is missing, incomplete or
        asked something else.
    """
    path = home / CHAPTER_10 / BEFORE_FILE
    try:
        before = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(before, dict) or any(key not in before for key in BEFORE_KEYS):
        return None
    return before if before["question"] == question else None


def same_query(first: str, second: str) -> str | None:
    """Return the SQL both replies end with, if it is the same query.

    Args:
        first: One reply.
        second: The other.

    Returns:
        The shared query, or None if either has none or they differ.
    """
    first_sql, second_sql = extract_sql(first), extract_sql(second)
    if first_sql is None or second_sql is None:
        return None
    left, right = " ".join(first_sql.split()), " ".join(second_sql.split())
    return left if left == right else None


def memory_budget() -> Budget:
    """Read this Mac's unified memory and the GPU's share of it from Metal.

    Returns:
        The budget.
    """
    info = mx.device_info()
    return Budget(int(info[DEVICE_MEMORY_KEY]), int(info[GPU_LIMIT_KEY]))


def main() -> None:
    """Load the 4-bit model, stream its reply, and print time and memory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default=QUESTION, help="what to ask")
    parser.add_argument(
        "--fast", action="store_true", help="use Liquid's export instead of ours"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"the length cap (default {MAX_TOKENS:,})",
    )
    args = parser.parse_args()
    if args.max_tokens < MIN_TOKENS:
        parser.error(f"--max-tokens must be at least {MIN_TOKENS}")
    if args.fast:
        folder, which = require_repo(FOUR_BIT), "Liquid's export (--fast)"
    else:
        folder, which = model_for_later_chapters()
    mx.reset_peak_memory()  # so the peak covers loading and answering
    loaded = load_folder(folder)

    report.print_title(SCRIPT, TITLE)
    report.print_model(loaded, which)
    timed = answer(loaded, args.question, args.max_tokens)
    report.print_timing(timed)
    before = read_before(args.question, course_home())
    shared = same_query(before["reply"], timed.text()) if before else None
    report.print_before_after(before, timed, loaded, shared)
    report.print_budget(memory_budget(), timed.peak_bytes, before)


if __name__ == "__main__":
    main()
