# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""LFM2.5-2.6B answers its first question on this laptop, one token at a time.

The question is rendered with the chat template, and mlx-lm's
``stream_generate`` asks the model for the next token again and again, each
time adding the new token to what the model reads. Every piece of text is
printed the moment its token is predicted, until the model writes
``<|im_end|>`` (its end of turn) or the length cap is reached. The script
times each token, reads MLX's peak memory and saves the measurement under
``$COURSE_HOME/10-meet-lfm``: the "before" picture that chapter 11 shrinks.

Decoding is greedy (always the most likely token), so the reply is the same
on every run; chapter 12 shows how a token is drawn instead. Run it with
``uv run first_reply.py "How many customers live in Paris? Answer with one
SQL query."`` after ``uv run download.py``. ``--fast`` uses the 4-bit export.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import mlx.core as mx
import report
from lfm import LoadedModel, chat_prompt, load_model
from lfm_files import FINISH_LENGTH, FINISH_STOP, chapter_home
from mlx_lm import stream_generate

SCRIPT = "first_reply.py"
TITLE = "LFM2.5-2.6B answers, one token at a time"
QUESTION = "How many customers live in Paris? Answer with one SQL query."
MAX_TOKENS = 1_024  # the project's output cap in training and evaluation
RECORD_NAME = "first_reply-{label}.json"
MILLISECONDS = 1_000
WARM_UP_TOKENS = 1
MIN_TOKENS = 1  # the smallest length cap mlx-lm accepts


class Step(Protocol):
    """What ``stream_generate`` yields for each step (its GenerationResponse)."""

    text: str
    token: int
    finish_reason: str | None


@dataclass(frozen=True)
class Piece:
    """One predicted token, as it arrived.

    Attributes:
        token: Its id.
        text: The text it added (empty while a character is incomplete).
        seconds: Time since the previous step (since the start, for the first).
    """

    token: int
    text: str
    seconds: float


@dataclass(frozen=True)
class Reply:
    """A whole reply, timed token by token.

    Attributes:
        question: What was asked.
        prompt_tokens: How many tokens the prompt is.
        pieces: The reply's tokens, in order (the end-of-turn token not
            included).
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        end_seconds: Time the end-of-turn step took, or 0 if the cap stopped it.
        peak_bytes: MLX's peak memory over the whole run, weights included.
    """

    question: str
    prompt_tokens: int
    pieces: list[Piece]
    finish_reason: str
    end_seconds: float
    peak_bytes: int

    def text(self) -> str:
        """Join the pieces into the reply's text.

        Returns:
            The text.
        """
        return "".join(piece.text for piece in self.pieces)

    def first_seconds(self) -> float:
        """Time to the first token: reading the prompt, then one prediction.

        Returns:
            Seconds.
        """
        return self.pieces[0].seconds if self.pieces else 0.0

    def decode_steps(self) -> list[float]:
        """List the time of every step after the first, end of turn included.

        Returns:
            Seconds per step.
        """
        steps = [piece.seconds for piece in self.pieces[1:]]
        if self.finish_reason == FINISH_STOP:
            steps.append(self.end_seconds)
        return steps

    def ms_per_token(self) -> float:
        """Average time per token once the prompt is read.

        Returns:
            Milliseconds per token, or 0 for a one-token reply.
        """
        steps = self.decode_steps()
        return MILLISECONDS * sum(steps) / len(steps) if steps else 0.0


def collect(
    steps: Iterable[Step],
    show: Callable[[str], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[Piece], str, float]:
    """Time each step of a generation and show its text the moment it arrives.

    ``stream_generate`` yields one step per predicted token. The last step
    has a finish reason: for ``"stop"`` its token is the end-of-turn token,
    which is not part of the reply; for ``"length"`` it is the reply's last
    token.

    Args:
        steps: The generation's steps.
        show: Prints a piece of text at once.
        clock: Seconds from a fixed point (``time.perf_counter``).

    Returns:
        The reply's pieces, the finish reason, and the end-of-turn step's time.
    """
    pieces = []
    finish = FINISH_LENGTH
    end_seconds = 0.0
    last = clock()
    for step in steps:
        now = clock()
        seconds, last = now - last, now
        show(step.text)
        if step.finish_reason is None or step.finish_reason == FINISH_LENGTH:
            pieces.append(Piece(step.token, step.text, seconds))
        if step.finish_reason is not None:
            finish = step.finish_reason
            if finish == FINISH_STOP:
                end_seconds = seconds
    return pieces, finish, end_seconds


def warm_up(
    loaded: LoadedModel,
    ids: list[int],
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Run one short, untimed generation, so the timed one measures steady speed.

    The first pass through a freshly loaded model is slower than every later
    one: MLX does one-time set-up work on the GPU. Measured on this laptop,
    the first token of the first pass took over a second; of the second, a
    fraction of one.

    Args:
        loaded: The model.
        ids: The prompt's token ids.
        clock: Seconds from a fixed point (``time.perf_counter``).

    Returns:
        How long the warm-up took, in seconds.
    """
    start = clock()
    for _ in stream_generate(
        loaded.model, loaded.tokenizer, ids, max_tokens=WARM_UP_TOKENS
    ):
        pass
    return clock() - start


def answer(loaded: LoadedModel, question: str, max_tokens: int) -> Reply:
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
    pieces, finish, end_seconds = collect(steps, report.show_piece)
    report.end_stream()
    return Reply(question, len(ids), pieces, finish, end_seconds, mx.get_peak_memory())


class Measurement(TypedDict):
    """The saved measurement: the fields chapter 11 reads back.

    Attributes:
        repo_id: The Hub repository the model came from.
        revision: Its pinned commit.
        storage: How its numbers are stored (``bfloat16`` or ``4-bit``).
        weight_bytes: The weights' bytes in memory.
        question: What was asked.
        prompt_tokens: The prompt's length in tokens.
        reply: The reply's text.
        reply_tokens: The reply's length in tokens.
        finish_reason: ``"stop"`` or ``"length"``.
        first_token_s: Seconds to the first token.
        ms_per_token: Milliseconds per token after the first.
        peak_memory_bytes: MLX's peak memory over the run.
        pieces: Every token: its id, its text and its seconds.
    """

    repo_id: str
    revision: str
    storage: str
    weight_bytes: int
    question: str
    prompt_tokens: int
    reply: str
    reply_tokens: int
    finish_reason: str
    first_token_s: float
    ms_per_token: float
    peak_memory_bytes: int
    pieces: list[dict[str, Any]]


def record(loaded: LoadedModel, reply: Reply) -> Measurement:
    """Gather the measurement chapter 11 compares against.

    Args:
        loaded: The model.
        reply: The timed reply.

    Returns:
        A JSON-ready dict.
    """
    return Measurement(
        repo_id=loaded.repo.repo_id,
        revision=loaded.repo.revision,
        storage=loaded.repo.label,
        weight_bytes=loaded.weight_bytes,
        question=reply.question,
        prompt_tokens=reply.prompt_tokens,
        reply=reply.text(),
        reply_tokens=len(reply.pieces),
        finish_reason=reply.finish_reason,
        first_token_s=reply.first_seconds(),
        ms_per_token=reply.ms_per_token(),
        peak_memory_bytes=reply.peak_bytes,
        pieces=[asdict(piece) for piece in reply.pieces],
    )


def save(data: Measurement, label: str) -> Path:
    """Write the measurement into this chapter's folder under COURSE_HOME.

    Args:
        data: The measurement.
        label: How the model is stored, for the file name.

    Returns:
        The file written.
    """
    folder = chapter_home()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RECORD_NAME.format(label=label)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Load the model, stream its reply, and print the timing and memory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default=QUESTION, help="what to ask")
    parser.add_argument(
        "--fast", action="store_true", help="use the 4-bit export instead of bfloat16"
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
    mx.reset_peak_memory()  # so the peak covers loading and generating
    loaded = load_model(args.fast)

    report.print_title(SCRIPT, f"{TITLE} ({loaded.repo.label})")
    report.print_model(loaded)
    reply = answer(loaded, args.question, args.max_tokens)
    report.print_timing(reply, loaded)
    path = save(record(loaded, reply), loaded.repo.label)
    report.print_saved(path, loaded.repo)


if __name__ == "__main__":
    main()
