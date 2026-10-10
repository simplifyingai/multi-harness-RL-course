# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2", "jinja2==3.1.6"]
# ///
"""One text, many token spellings.

The tokenizer always cuts a text the same way, but it is not the only way:
'answer.txt' is 'answer' + '.txt' to the tokenizer, and also 'answer' + '.'
+ 'txt', and 166 other sequences of pieces from the vocabulary. A model
writes one token at a time, so it can produce any of them, and each decodes to
the same text. Encode that text again and you get the tokenizer's ids, not the
ones the model wrote (chapter 15 shows why that matters for training).

This finds every spelling of a text, counts the spellings of some texts an
agent writes, and counts how many vocabulary pieces are two smaller pieces
glued together, which is where all the spellings come from.

Run it with ``uv run spellings.py`` or ``uv run spellings.py "any text"``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

import report
from chats import QUESTION
from lfm import LfmTokenizer, load_tokenizer

SCRIPT = "spellings.py"
TITLE = "one text, many token spellings"
ANSWER_FILE = "answer.txt"
# Texts the agents in this course write: the question, and bash commands.
AGENT_TEXTS = (
    QUESTION,
    "sqlite3 data.db .tables",
    "SELECT COUNT(*) FROM customers WHERE city = 'Paris'",
    "echo 13 > answer.txt",
    "cat answer.txt",
)
MAX_LISTED = 8  # spellings listed in full; the rest are only counted
# Longer texts are refused: the search goes one level deeper per token, and a
# paragraph's count of spellings has hundreds of digits anyway.
MAX_TEXT_CHARS = 200
REPLACEMENT = "\ufffd"  # what decoding shows for a token that is part of a character


@dataclass(frozen=True)
class PieceTable:
    """The vocabulary as text: each learned piece's text and its id.

    Attributes:
        ids: The id of each piece, by its text.
        longest: The length of the longest piece, in characters.
        partial: How many pieces hold only part of a character, left out.
    """

    ids: dict[str, int]
    longest: int
    partial: int


@dataclass(frozen=True)
class TextSpellings:
    """Every way to spell one text with vocabulary pieces.

    Attributes:
        text: The text.
        tokenizer_ids: The tokenizer's own spelling.
        by_length: How many spellings use each number of tokens.
        one_longer: Spellings with one token more than the tokenizer's.
        decoded: What each of those decodes to.
        encoded_again: What encoding each decoded text gives.
    """

    text: str
    tokenizer_ids: list[int]
    by_length: dict[int, int]
    one_longer: list[list[int]]
    decoded: list[str]
    encoded_again: list[list[int]]

    @property
    def total(self) -> int:
        """Count every spelling."""
        return sum(self.by_length.values())


def piece_table(tokenizer: LfmTokenizer) -> PieceTable:
    """Collect the text of every learned piece (added marker tokens left out).

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        The pieces whose text is whole characters, by text.
    """
    texts = tokenizer.pieces(list(range(tokenizer.learned_size())))
    ids = {}
    partial = 0
    for token_id, text in enumerate(texts):
        if REPLACEMENT in text:
            partial += 1
            continue
        ids.setdefault(text, token_id)
    # The vocabulary is never empty, so max() always has pieces to measure.
    return PieceTable(ids, max(map(len, ids)), partial)


def tail_counts(text: str, table: PieceTable) -> list[Counter[int]]:
    """For every position, count the spellings of the rest of the text, by length.

    Works backwards from the end: the spellings of text[i:] are each piece
    that starts at i, followed by any spelling of what is left after it.

    Args:
        text: The text.
        table: The vocabulary as text.

    Returns:
        One counter per position (and one for the end): number of tokens ->
        how many spellings of the text from there use that many.
    """
    size = len(text)
    tails: list[Counter[int]] = [Counter() for _ in range(size + 1)]
    tails[size][0] = 1  # the empty rest has one spelling: no tokens
    for start in range(size - 1, -1, -1):
        for end in range(start + 1, min(size, start + table.longest) + 1):
            if text[start:end] in table.ids:
                for tokens, ways in tails[end].items():
                    tails[start][tokens + 1] += ways
    return tails


def count_by_length(text: str, table: PieceTable) -> dict[int, int]:
    """Count the spellings of a text, by how many tokens each uses.

    Args:
        text: The text.
        table: The vocabulary as text.

    Returns:
        Number of tokens -> how many spellings use that many.
    """
    return dict(sorted(tail_counts(text, table)[0].items()))


def spellings_of_length(
    text: str, table: PieceTable, tokens: int, limit: int
) -> list[list[int]]:
    """List spellings of a text that use exactly a given number of tokens.

    It walks piece by piece, and only down paths that can still end with
    exactly ``tokens`` pieces (the counts from ``tail_counts`` say which), so
    it never wanders through the billions of spellings it is not asked for.

    Args:
        text: The text.
        table: The vocabulary as text.
        tokens: How many tokens each spelling must use.
        limit: Stop after this many.

    Returns:
        Up to ``limit`` spellings, as id lists.
    """
    tails = tail_counts(text, table)
    found: list[list[int]] = []

    def extend(start: int, chosen: list[int]) -> None:
        """Try every piece that starts at ``start``, then the rest."""
        if len(found) >= limit or tokens - len(chosen) not in tails[start]:
            return
        if start == len(text):
            found.append(list(chosen))
            return
        for end in range(start + 1, min(len(text), start + table.longest) + 1):
            piece = text[start:end]
            if piece in table.ids:
                chosen.append(table.ids[piece])
                extend(end, chosen)
                chosen.pop()

    extend(0, [])
    return found


def text_spellings(
    tokenizer: LfmTokenizer, table: PieceTable, text: str
) -> TextSpellings:
    """Find a text's spellings and round-trip the ones a token longer.

    Args:
        tokenizer: The model's tokenizer.
        table: The vocabulary as text.
        text: The text.

    Returns:
        The tokenizer's spelling, the counts, and the one-token-longer ones.
    """
    tokenizer_ids = tokenizer.encode(text)
    by_length = count_by_length(text, table)
    longer = spellings_of_length(text, table, len(tokenizer_ids) + 1, MAX_LISTED)
    decoded = [tokenizer.decode(ids) for ids in longer]
    again = [tokenizer.encode(back) for back in decoded]
    return TextSpellings(text, tokenizer_ids, by_length, longer, decoded, again)


def glued_pieces(table: PieceTable) -> int:
    """Count the pieces that are also two shorter pieces put side by side.

    Args:
        table: The vocabulary as text.

    Returns:
        How many pieces can be split into two pieces of the vocabulary.
    """
    glued = 0
    for text in table.ids:
        if any(
            text[:cut] in table.ids and text[cut:] in table.ids
            for cut in range(1, len(text))
        ):
            glued += 1
    return glued


def main() -> None:
    """Spell one text every way, then count spellings across texts and pieces."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", default=ANSWER_FILE, help="text to spell")
    parser.add_argument("--fast", action="store_true", help="no pauses")
    args = parser.parse_args()
    pause_s = report.NO_PAUSE_S if args.fast else report.PAUSE_S
    if len(args.text) > MAX_TEXT_CHARS:
        raise SystemExit(f"give a text of at most {MAX_TEXT_CHARS} characters")
    tokenizer = load_tokenizer()
    table = piece_table(tokenizer)

    report.print_title(SCRIPT, TITLE)
    found = text_spellings(tokenizer, table, args.text)
    report.print_text_spellings(found, tokenizer.pieces)
    report.pause(pause_s)
    rows = [
        (text, len(tokenizer.encode(text)), sum(count_by_length(text, table).values()))
        for text in AGENT_TEXTS
    ]
    report.print_spelling_counts(rows)
    report.pause(pause_s)
    report.print_glued(glued_pieces(table), len(table.ids), table.partial)


if __name__ == "__main__":
    main()
