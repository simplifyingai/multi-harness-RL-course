"""How the chapter's scripts print: round trips, prompts, counts and tables.

Nothing here computes anything. It only turns the scripts' results into lines
a person can read on screen, so each script can stay about tokens.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type hints only: these modules import this one
    from count import Call, Overflow
    from spellings import TextSpellings
    from tokens import OtherAdded, RoundTrip, SpecialToken, Vocabulary

PAUSE_S = 0.4
NO_PAUSE_S = 0.0
INDENT = "  "
PERCENT = 100
ROLE_WIDTH = 11  # "assistant" plus room
ID_WIDTH = 8
ID_COLUMN = ID_WIDTH + len(INDENT)  # ids right-aligned under the text, indented
TOKEN_WIDTH = 22  # the longest special token, <|tool_call_start|>, plus room
COUNT_WIDTH = 8
TEXT_WIDTH = 54  # the longest agent text plus room
SPELLINGS_WIDTH = 18
CALL_WIDTH = 6
LABEL_WIDTH = 10  # "spellings" plus room, the row labels of the length table
CELL_WIDTH = 6  # one column of the length table
NUMBER_WIDTH = 4  # a call's number, right-aligned under "call"
TOOL_CALL_RE = re.compile(r"<\|tool_call_start\|>\[(.*)\]<\|tool_call_end\|>")
EMPTY_OUTPUT = "(nothing)"


def pause(seconds: float) -> None:
    """Wait a moment between sections, so a recording can follow along.

    Args:
        seconds: How long; 0 for no pause.
    """
    if seconds > 0:
        time.sleep(seconds)


def print_title(script: str, title: str) -> None:
    """Print the script's first line.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    print(f"{script} · {title}")


def quoted(pieces: Sequence[str]) -> str:
    """Write token pieces side by side, each in quotes so spaces show.

    Args:
        pieces: The text of each token.

    Returns:
        The pieces as one line.
    """
    return " ".join(repr(piece) for piece in pieces)


def print_round_trip(trip: RoundTrip) -> None:
    """Print a text, its tokens one per line, its ids, and the text decoded back.

    Args:
        trip: The round trip.
    """
    print(f"\ntext    {trip.text}")
    for token_id, piece in zip(trip.ids, trip.pieces, strict=True):
        print(f"{token_id:>{ID_COLUMN}}  {piece!r}")
    print(f"ids     {trip.ids}")
    same = "the same text" if trip.back == trip.text else "a different text"
    print(f"back    {trip.back}  ({same})")
    print(f"{len(trip.ids)} tokens for {len(trip.text)} characters")


def print_vocabulary(vocabulary: Vocabulary) -> None:
    """Print how many entries the vocabulary has, and the model's rows.

    Args:
        vocabulary: The counts.
    """
    last = vocabulary.size - 1
    print(f"\nthe vocabulary: {vocabulary.size:,} entries, ids 0 to {last:,}")
    print(
        f"{INDENT}{vocabulary.learned:>7,} pieces learned from text, "
        f"ids 0 to {vocabulary.learned - 1:,}"
    )
    print(
        f"{INDENT}{vocabulary.added:>7,} tokens added on top, "
        f"ids {vocabulary.learned:,} to {last:,}"
    )
    spare = vocabulary.model_rows - vocabulary.size
    print(
        f"the model keeps {vocabulary.model_rows:,} rows, one per id: "
        f"{spare:,} are never used"
    )


def print_special_tokens(rows: Sequence[SpecialToken], others: OtherAdded) -> None:
    """Print the special tokens the course uses, with their ids and meanings.

    Args:
        rows: The tokens, in the chapter's order.
        others: The added tokens the course never uses.
    """
    print("\nspecial tokens: each is one token, never cut into pieces")
    for row in rows:
        name = f"{row.text:<{TOKEN_WIDTH}}"
        print(f"{INDENT}{name}{row.token_id:<{ID_WIDTH}}{row.meaning}")
    print(
        f"{INDENT}and {others.count} more the course never uses "
        f"({others.images} for images)"
    )


def print_messages(messages: Sequence[dict[str, Any]]) -> None:
    """Print a chat as one line per message: its role, then its content.

    Args:
        messages: OpenAI-style messages.
    """
    print("messages:")
    for message in messages:
        print(f"{INDENT}{message['role']:<{ROLE_WIDTH}}{message['content']}")


def print_prompt(label: str, text: str, tokens: int) -> None:
    """Print a rendered prompt exactly as the model reads it.

    Args:
        label: What this render is.
        text: The prompt.
        tokens: How many tokens it is.
    """
    print(f"{label}, {tokens} tokens:")
    print(text)


def print_plain(
    messages: Sequence[dict[str, Any]],
    text: str,
    ids: list[int],
    cue_pieces: list[str],
) -> None:
    """Print the first chat: one user message, rendered.

    Args:
        messages: The chat.
        text: Its prompt.
        ids: The prompt's ids.
        cue_pieces: The generation prompt, token by token.
    """
    print("\n1. one user message")
    print_messages(messages)
    print_prompt("the prompt", text, len(ids))
    print(
        f"the last {len(cue_pieces)} tokens are the generation prompt "
        "(add_generation_prompt=True):"
    )
    print(f"{quoted(cue_pieces)}, the cue that the model's reply comes next")


def print_with_tool(text: str, tokens: int, plain_tokens: int, think_id: int) -> None:
    """Print the second chat: the same message with a bash tool.

    Args:
        text: Its prompt.
        tokens: The prompt's length in tokens.
        plain_tokens: The first chat's length, for the difference.
        think_id: The id of ``<think>``.
    """
    print("\n2. the same message, with a bash tool (mini-swe-agent's)")
    print_prompt("the prompt", text, tokens)
    print(
        f"the tool list goes into a system message: {tokens - plain_tokens} tokens more"
    )
    print(
        f"the prompt ends with <think> ({think_id}): "
        "every reply starts inside its reasoning"
    )


def print_drop(
    messages: Sequence[dict[str, Any]],
    default: tuple[str, int],
    kept: tuple[str, int],
    shared: int,
) -> None:
    """Print the third chat with and without preserve_thinking, and the rule.

    Args:
        messages: The chat.
        default: The default render's text and length.
        kept: The preserve_thinking render's text and length.
        shared: How many ids the two share from the start.
    """
    print("\n3. three user turns: is the reasoning of earlier replies kept?")
    print_messages(messages)
    print_prompt("the template's default", *default)
    print_prompt("with preserve_thinking=True, as the project renders", *kept)
    print(
        f"the ids agree for the first {shared} tokens, then part: "
        f"the default dropped {kept[1] - default[1]} tokens of reasoning"
    )
    print(
        "the rule: a reply keeps its <think> block only if it comes after the "
        "last user message;"
    )
    print("preserve_thinking=True keeps every reply exactly as the model wrote it")


def tool_call(reply_text: str) -> str:
    """Pull the tool call out of a reply, for a one-line summary.

    Args:
        reply_text: A reply as the model wrote it.

    Returns:
        The call, such as ``bash(command='cat answer.txt')``.
    """
    found = TOOL_CALL_RE.search(reply_text)
    return found.group(1) if found else reply_text


def print_calls(
    calls: Sequence[Call], decode: Callable[[list[int]], str], same: bool
) -> None:
    """Print what each call read and wrote, and how the prompts grow.

    Args:
        calls: One record per call; at least two, to compare the first two.
        decode: Turns ids back into text.
        same: Whether the template's default gives the same prompts.
    """
    print("\nthree calls of a tiny agent on the shop database")
    print("(the replies are scripted; the tool outputs are what sqlite3 prints)")
    header = f"{'call':<{CALL_WIDTH}}{'reads':>{CALL_WIDTH}}{'writes':>{COUNT_WIDTH}}"
    print(f"{header}  tool call")
    for call in calls:
        print(
            f"{call.number:>{NUMBER_WIDTH}}  {len(call.prompt):>{CALL_WIDTH}}"
            f"{len(call.reply):>{COUNT_WIDTH}}  {tool_call(decode(call.reply))}"
        )
    first, second = calls[0], calls[1]
    added = len(second.prompt) - len(first.prompt) - len(first.reply)
    print(
        f"call 2 reads call 1's {len(first.prompt)} tokens, its "
        f"{len(first.reply)}-token reply, and {added} more: the tool's output "
        "and the markers around it"
    )
    kept = all(call.prefix_kept for call in calls if call.prefix_kept is not None)
    answer = "yes" if kept else "no"
    print(f"each prompt starts with the whole call before it, reply included: {answer}")
    read = sum(len(call.prompt) for call in calls)
    written = sum(len(call.reply) for call in calls)
    print(f"{len(calls)} calls read {read} tokens in all, to write {written}")
    if same:
        print(
            "the template's default gives the same ids here: every reply comes "
            "after the only user message"
        )
    else:
        print("the template's default gives different ids here")


def print_last_prompt(number: int, text: str) -> None:
    """Print the last call's whole prompt.

    Args:
        number: The call's number.
        text: The prompt.
    """
    print(f"\ncall {number}'s prompt, in full:")
    print(text)


def print_window(number: int, last_prompt: int, positions: int, cap: int) -> None:
    """Print the context window and how much of it the last call used.

    Args:
        number: The last call's number.
        last_prompt: Its prompt length.
        positions: How many positions the model can read.
        cap: The engine's prompt limit.
    """
    print("\nthe context window: the most tokens the model reads in one call")
    print(f"{INDENT}LFM2.5-2.6B reads up to {positions:,} positions, one token each")
    print(f"{INDENT}the project's engine refuses any prompt over {cap:,} tokens")
    share = PERCENT * last_prompt / cap
    print(
        f"{INDENT}call {number}'s {last_prompt} tokens use {share:.1f}% "
        "of the engine's cap"
    )


def print_overflow(result: Overflow, positions: int, cap: int) -> None:
    """Print what happens when a tool's output is the whole database file.

    Args:
        result: The measured file and prompt.
        positions: How many positions the model can read.
        cap: The engine's prompt limit.
    """
    print("\nthe command that blew the window: in Pi, the untrained model often")
    print("read the binary data.db itself (here with bash: cat data.db)")
    print(
        f"{INDENT}the file (shop.db here) is {result.file_bytes:,} bytes; "
        "read as text it is "
        f"{result.file_tokens:,} tokens"
    )
    times = result.prompt_tokens / cap
    print(
        f"{INDENT}the next prompt is {result.prompt_tokens:,} tokens, "
        f"{times:.1f} times the engine's cap"
    )
    # The engine's own words (mhrl/engine/server.py), numbers unformatted.
    print(
        f"{INDENT}the engine answers: prompt is {result.prompt_tokens} tokens; "
        f"the limit is {cap}"
    )
    fits = "fits" if result.prompt_tokens <= positions else "does not fit"
    print(f"{INDENT}(it {fits} in the model's own {positions:,} positions)")


def print_text_spellings(
    found: TextSpellings, pieces: Callable[[list[int]], list[str]]
) -> None:
    """Print a text's spellings: the tokenizer's, the ones a token longer, the count.

    Args:
        found: The text's spellings.
        pieces: Turns ids into the text of each token.
    """
    print(f"\n{found.text!r}")
    print(
        f"{INDENT}the tokenizer writes {found.tokenizer_ids}: "
        f"{quoted(pieces(found.tokenizer_ids))}"
    )
    longer = len(found.tokenizer_ids) + 1
    count = found.by_length.get(longer, 0)
    print(f"{INDENT}{count} spellings use {longer} tokens:")
    width = max((len(str(ids)) for ids in found.one_longer), default=0)
    for ids in found.one_longer:
        print(f"{INDENT * 2}{str(ids):<{width}}  {quoted(pieces(ids))}")
    if count > len(found.one_longer):
        print(f"{INDENT * 2}and {count - len(found.one_longer)} more")
    same_text = all(back == found.text for back in found.decoded)
    same_ids = all(ids == found.tokenizer_ids for ids in found.encoded_again)
    if same_text and same_ids:
        print(
            f"{INDENT}each decodes to {found.text!r}; encode that again and you "
            f"get {found.tokenizer_ids}"
        )
    else:
        print(f"{INDENT}decoded and encoded again, some do not come back the same")
    print(f"{INDENT}in all, {found.total:,} token sequences spell {found.text!r}:")
    lengths = "".join(f"{tokens:>{CELL_WIDTH}}" for tokens in found.by_length)
    ways = "".join(f"{count:>{CELL_WIDTH},}" for count in found.by_length.values())
    print(f"{INDENT * 2}{'tokens':<{LABEL_WIDTH}}{lengths}")
    print(f"{INDENT * 2}{'spellings':<{LABEL_WIDTH}}{ways}")
    if not found.text.isascii():
        print(
            f"{INDENT}(pieces that hold part of a character are left out, so for "
            "this text the counts are a lower bound)"
        )


def print_spelling_counts(rows: Sequence[tuple[str, int, int]]) -> None:
    """Print how many spellings some texts an agent writes have.

    Args:
        rows: (text, tokens in the tokenizer's spelling, spellings in all).
    """
    print("\nhow many ways to spell what an agent writes")
    header = f"{'text':<{TEXT_WIDTH}}{'tokens':>{CALL_WIDTH}}"
    print(f"{INDENT}{header}{'spellings':>{SPELLINGS_WIDTH}}")
    for text, tokens, spellings in rows:
        print(
            f"{INDENT}{text:<{TEXT_WIDTH}}{tokens:>{CALL_WIDTH}}"
            f"{spellings:>{SPELLINGS_WIDTH},}"
        )


def print_glued(glued: int, whole: int, partial: int) -> None:
    """Print how many vocabulary pieces are two smaller pieces glued together.

    Args:
        glued: Pieces that split into two pieces.
        whole: Pieces whose text is whole characters.
        partial: Pieces that hold only part of a character.
    """
    print("\nwhere the spellings come from: BPE (byte pair encoding) built the")
    print("vocabulary by gluing pairs of pieces that often sit side by side")
    print(f"{INDENT}{glued:,} of the {whole:,} pieces made of whole characters")
    print(f"{INDENT}are two shorter pieces glued together")
    print(f"{INDENT}({partial:,} more hold part of a character: emoji, other scripts)")
    print("the tokenizer always picks one spelling; a model that writes one token at")
    print("a time can produce any of them")
