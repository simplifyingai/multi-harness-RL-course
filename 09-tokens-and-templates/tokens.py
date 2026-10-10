# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2", "jinja2==3.1.6"]
# ///
"""Text to token ids and back, the vocabulary, and the special tokens.

A language model never sees letters. Its tokenizer cuts text into tokens
(small pieces of text, each with an integer id), the model reads and writes
ids, and the tokenizer turns ids back into text. This prints that round trip
for a question from chapter 1, counts LFM2.5-2.6B's vocabulary, and lists the
special tokens that mark where a message, a reasoning block or a tool call
starts and ends.

Run it with ``uv run tokens.py`` or ``uv run tokens.py "any text"``. The first
run downloads the tokenizer files (about 18 MB) into ``COURSE_HOME``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import report
from chats import QUESTION
from lfm import LfmTokenizer, load_tokenizer

# The model's embedding table has one row per id, and it has 128,000 rows
# (vocab_size in the model's config.json, which chapter 10 downloads). That is
# more rows than the tokenizer has entries; the extra rows are never used.
MODEL_ROWS = 128_000

# The special tokens this course uses, in the order the chapter meets them,
# each with what it means.
SPECIAL_TOKENS = (
    ("<|startoftext|>", "the start of every prompt"),
    ("<|im_start|>", "a message starts; its role follows"),
    ("<|im_end|>", "a message ends: the model's end of turn"),
    ("<think>", "the model's reasoning starts"),
    ("</think>", "its reasoning ends"),
    ("<|tool_call_start|>", "a tool call starts"),
    ("<|tool_call_end|>", "a tool call ends"),
    ("<|pad|>", "filler that lines up sequences of different lengths"),
)
# How the added tokens for images start: <image>, <|image_start|>,
# <|img_row_1_col_1|> and the like.
IMAGE_PREFIXES = ("<image", "<|image", "<|img")
SCRIPT = "tokens.py"
TITLE = "LFM2.5-2.6B's tokenizer: text to token ids and back"


@dataclass(frozen=True)
class RoundTrip:
    """A text cut into tokens and put back together.

    Attributes:
        text: The text.
        ids: Its token ids.
        pieces: The text of each token.
        back: The ids decoded again.
    """

    text: str
    ids: list[int]
    pieces: list[str]
    back: str


@dataclass(frozen=True)
class Vocabulary:
    """How many ids the tokenizer has, and where they come from.

    Attributes:
        size: Every entry, learned and added.
        learned: Pieces learned from text, ids 0 up.
        added: Tokens added on top of them.
        model_rows: Rows in the model's embedding table.
    """

    size: int
    learned: int
    added: int
    model_rows: int


@dataclass(frozen=True)
class SpecialToken:
    """One special token: its text, its id and what it marks.

    Attributes:
        text: The token written out.
        token_id: Its id.
        meaning: What it marks, in plain words.
    """

    text: str
    token_id: int
    meaning: str


@dataclass(frozen=True)
class OtherAdded:
    """The added tokens this course never uses.

    Attributes:
        count: How many there are.
        images: How many of them mark images.
    """

    count: int
    images: int


def round_trip(tokenizer: LfmTokenizer, text: str) -> RoundTrip:
    """Encode a text, split it into pieces and decode it again.

    Args:
        tokenizer: The model's tokenizer.
        text: Any text.

    Returns:
        The round trip.
    """
    ids = tokenizer.encode(text)
    return RoundTrip(text, ids, tokenizer.pieces(ids), tokenizer.decode(ids))


def vocabulary(tokenizer: LfmTokenizer) -> Vocabulary:
    """Count the tokenizer's entries.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        The counts.
    """
    size = tokenizer.vocabulary_size()
    learned = tokenizer.learned_size()
    return Vocabulary(size, learned, size - learned, MODEL_ROWS)


def special_tokens(tokenizer: LfmTokenizer) -> list[SpecialToken]:
    """Look up the id of each special token the course uses.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        One row per token, in the chapter's order.

    Raises:
        KeyError: If a token is missing from tokenizer.json.
    """
    return [
        SpecialToken(text, tokenizer.require_id(text), meaning)
        for text, meaning in SPECIAL_TOKENS
    ]


def other_added(tokenizer: LfmTokenizer) -> OtherAdded:
    """Count the added tokens the course never uses, and those for images.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        The counts.
    """
    used = {text for text, _ in SPECIAL_TOKENS}
    others = [text for text in tokenizer.added_tokens().values() if text not in used]
    images = [text for text in others if text.startswith(IMAGE_PREFIXES)]
    return OtherAdded(len(others), len(images))


def main() -> None:
    """Print the round trip, the vocabulary and the special tokens."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", default=QUESTION, help="text to encode")
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    pause_s = report.NO_PAUSE_S if args.fast else report.PAUSE_S
    tokenizer = load_tokenizer()

    report.print_title(SCRIPT, TITLE)
    report.print_round_trip(round_trip(tokenizer, args.text))
    report.pause(pause_s)
    report.print_vocabulary(vocabulary(tokenizer))
    report.pause(pause_s)
    report.print_special_tokens(special_tokens(tokenizer), other_added(tokenizer))


if __name__ == "__main__":
    main()
