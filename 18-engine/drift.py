# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi==0.142.2",
#     "uvicorn==0.54.0",
#     "jinja2==3.1.6",
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Template round-trip drift: what a harness's copy of a reply does to the tokens.

A harness never sends a reply back as the model wrote it. It keeps OpenAI's
JSON copy: the visible text, and each tool call's name and arguments. Put
through the chat template on the next call, that copy is not the tokens the
model sampled, for two reasons:

1. **the dropped <think>**: the copy has no reasoning, so the template writes
   the turn without it, starting where the previous prompt ended with
   ``<think>``;
2. **the re-written tool call**: the copy's call is JSON, and the template
   writes it back in its own spelling (its quotes, its escapes), which is not
   always the model's.

``uv run drift.py`` samples one real reply (clinic-easy-001, call 1,
temperature 0), keeps its ids, builds the copy a harness sends back,
re-renders the conversation and prints the first position where the ids
leave the sampled ones: once as the copy comes, once with its reasoning
given back (which leaves only the tool call). Then it puts the model's own
text back, as engine.py does, and checks that the next prompt continues the
previous call.

Text is not ids: the same text can be cut into other tokens (chapter 9).
``--recorded`` reads two replies the project's engine sampled at temperature
1 in its GRPO run (``data/recorded_replies.json``): one whose tool call the
template re-spells, and one whose own text cuts into other tokens, which
putting the text back cannot fix. It loads only the tokenizer.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine import ChatRequest, Engine, harness_copy
from lfm_files import OUTPUT_CAP, SAMPLE_SEED, pick_model, short_path, work_folder
from policy import GREEDY, Model, RenderError, Sampling
from tasks import DATA_DIR, find_task, prepare_workdir
from tiny_harness_http import (
    ASSISTANT,
    DEFAULT_TASK,
    TOOL_ROLE,
    USER,
    bash_schema,
    execute,
    start_messages,
)
from toolcalls import (
    THINK_CLOSE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    openai_tool_calls,
    split_reply,
)

SCRIPT = "drift.py"
LABEL = "drift"  # this script's work folder
RECORDED_FILE = DATA_DIR / "recorded_replies.json"
RESPELLED, RECUT = "re-spelled", "re-cut"
WINDOW = 3  # tokens shown on each side of a difference
STAND_IN = "(a recorded reply follows; the project's prompt is not bundled)"
NOT_RERUN = "(the recorded tool result is not bundled)"
REASONING = "reasoning_content"  # where OpenAI-style APIs put a reply's reasoning
INDENT = "  "


@dataclass(frozen=True)
class Drift:
    """Where re-rendered ids leave a chain of sampled ones.

    Attributes:
        position: The first index where they differ, or None when the
            re-rendered ids continue the chain.
        reply_start: Where the reply begins in the chain (the prompt's length).
    """

    position: int | None
    reply_start: int


@dataclass(frozen=True)
class Recorded:
    """One reply the project's engine sampled.

    Attributes:
        name: ``re-spelled`` or ``re-cut``.
        task_id: The question.
        harness: The harness that asked for it.
        call: Which call of the rollout it was.
        token_ids: Its exact token ids, the end of turn included.
        logprobs: Each token's log-probability, as sampled.
    """

    name: str
    task_id: str
    harness: str
    call: int
    token_ids: list[int]
    logprobs: list[float]


def first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Find the first position where two token lists differ.

    Args:
        a: One list.
        b: The other.

    Returns:
        The first index where they differ, the shorter one's length when one
        is a strict prefix of the other, or None when they are equal.
    """
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(chain: list[int], rendered: list[int], reply_start: int) -> Drift:
    """Check whether a re-rendered prompt continues a sampled chain.

    Args:
        chain: The previous prompt and the reply sampled after it.
        rendered: The next prompt, as some renderer built it.
        reply_start: The previous prompt's length.

    Returns:
        Where the next prompt leaves the chain, if it does.
    """
    return Drift(first_difference(chain, rendered[: len(chain)]), reply_start)


def where(model: Model, reply: list[int], offset: int) -> str:
    """Say which part of a reply a token falls in.

    Args:
        model: The model (to read the reply's text).
        reply: The reply's token ids.
        offset: The token's index in the reply.

    Returns:
        ``in the reasoning``, ``in the tool-call span`` or ``in the visible text``.
    """
    before = model.decode(reply[:offset])
    reasoning, closed, answer = before.partition(THINK_CLOSE)
    if not closed:
        return "in the reasoning"
    if answer.count(TOOL_CALL_START) > answer.count(TOOL_CALL_END):
        return "in the tool-call span"
    return "in the visible text"


def window(model: Model, ids: list[int], position: int) -> str:
    """Show the tokens around a position, a bar before the one at it.

    Args:
        model: The model (to write each token).
        ids: The tokens.
        position: The index to show.

    Returns:
        Such as ``'assistant' '\\n' | '<think>' 'The'``.
    """
    start = max(0, position - WINDOW)
    pieces = [repr(model.decode([token])) for token in ids[start : position + WINDOW]]
    cut = position - start
    return " ".join([*pieces[:cut], "|", *pieces[cut:]])


def describe(
    model: Model, drift: Drift, chain: list[int], rendered: list[int]
) -> list[str]:
    """Write where a re-rendered prompt leaves the chain, token by token.

    Args:
        model: The model.
        drift: From ``compare``.
        chain: The previous prompt and reply.
        rendered: The re-rendered prompt.

    Returns:
        The lines to print.
    """
    if drift.position is None:
        return ["the same tokens: the next prompt continues the previous call"]
    if drift.position < drift.reply_start:
        place = (
            f"token {drift.position + 1:,} of the previous prompt's "
            f"{drift.reply_start:,}: the reply is not even reached"
        )
    else:
        offset = drift.position - drift.reply_start
        reply = chain[drift.reply_start :]
        place = f"token {offset + 1:,} of the reply, {where(model, reply, offset)}"
    return [
        f"first difference: {place}",
        f"{INDENT}sampled      {window(model, chain, drift.position)}",
        f"{INDENT}re-rendered  {window(model, rendered, drift.position)}",
    ]


def print_lines(lines: list[str], indent: str = INDENT) -> None:
    """Print lines under a heading.

    Args:
        lines: The lines.
        indent: Put before each.
    """
    for line in lines:
        print(indent + line)


def copy_of(content: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the copy of a reply a harness keeps and sends back.

    Args:
        content: The visible text.
        calls: The tool calls, as OpenAI JSON.

    Returns:
        ``{"role": "assistant", "content": ..., "tool_calls": [...]}``.
    """
    copy: dict[str, Any] = {"role": ASSISTANT, "content": content or None}
    if calls:
        copy["tool_calls"] = calls
    return copy


def rerender(model: Model, messages: list[dict[str, Any]], tools: list) -> list[int]:
    """Render a conversation with no help from the engine's memory.

    Args:
        model: The model.
        messages: The conversation.
        tools: The tools offered.

    Returns:
        The prompt's token ids.
    """
    return model.encode(model.render(messages, tools))


def run_live(fast: bool) -> None:
    """Sample a real reply and measure both causes, then the engine's fix.

    Args:
        fast: Use Liquid's export of the model.
    """
    from lfm import LfmModel  # MLX loads here, so the rest runs without it

    folder, which = pick_model(fast)
    model = LfmModel(folder, SAMPLE_SEED)
    engine = Engine(model)
    print(f"{INDENT}model: {short_path(folder)} ({which}), {model.seconds:.1f} s")
    task = find_task(DEFAULT_TASK)
    tools = [bash_schema()]
    messages = start_messages(task)
    sampling = Sampling(OUTPUT_CAP, GREEDY)
    completion = engine.complete(ChatRequest(messages, tools, sampling))
    sampled = completion.sampled.token_ids
    chain = completion.prompt_ids + sampled
    start = len(completion.prompt_ids)
    print(f"\n1. the reply the model sampled: {len(sampled)} tokens after a")
    print(f"{INDENT}{start:,}-token prompt ({task.task_id}, call 1, temperature 0)")
    print(f"{INDENT}<think>{model.decode(sampled)}")
    copy = copy_of(completion.reply.content, completion.tool_calls)
    workdir = prepare_workdir(task, work_folder(LABEL, task.task_id))
    results = [
        {"role": TOOL_ROLE, "tool_call_id": call["id"]}
        | {"content": execute(call, tools, workdir).output}
        for call in completion.tool_calls
    ]
    print("\n2. the copy a harness keeps and sends back (OpenAI JSON)")
    print(f"{INDENT}{json.dumps(copy, ensure_ascii=False)}")
    try:
        model.render([*messages, copy, *results], tools)
        print(f"{INDENT}the template renders it as it is")
    except RenderError as error:
        print(f"{INDENT}as it is, LFM2.5's template refuses it: {error}")
        print(f"{INDENT}so its arguments are read back into an object first")
    plain = harness_copy(copy)
    dropped = rerender(model, [*messages, plain, *results], tools)
    print("\n3. cause 1, the dropped <think>: the copy as it comes")
    print_lines(describe(model, compare(chain, dropped, start), chain, dropped))
    given = plain | {REASONING: completion.reply.reasoning}
    rewritten = rerender(model, [*messages, given, *results], tools)
    print("\n4. cause 2, the re-written tool call: the copy, its reasoning given back")
    if not completion.tool_calls:
        print(f"{INDENT}this reply calls no tool")
    print_lines(describe(model, compare(chain, rewritten, start), chain, rewritten))
    restored = engine.render([*messages, copy, *results], tools)
    kept = compare(chain, restored, start).position is None
    print("\n5. the engine's way: the model's own text, found by the call's id")
    print(f"{INDENT}continues the previous call: {'yes' if kept else 'no'}")
    if not kept:
        print_lines(describe(model, compare(chain, restored, start), chain, restored))


def load_recorded(path: Path = RECORDED_FILE) -> list[Recorded]:
    """Read the bundled replies the project's engine sampled.

    Args:
        path: The JSON file.

    Returns:
        The replies.

    Raises:
        OSError: If the file cannot be read.
        KeyError: If a reply lacks a field.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        Recorded(
            name=reply["name"],
            task_id=reply["task_id"],
            harness=reply["harness"],
            call=reply["call"],
            token_ids=reply["token_ids"],
            logprobs=reply["logprobs"],
        )
        for reply in document["replies"]
    ]


def respelled(model: Model, reply: Recorded) -> list[str]:
    """Measure cause 2 on a recorded reply: re-render its copy, reasoning given back.

    Args:
        model: The tokenizer and template.
        reply: The recorded reply.

    Returns:
        The lines to print.
    """
    tools = [bash_schema()]
    messages = [{"role": USER, "content": STAND_IN}]
    prompt = rerender(model, messages, tools)
    chain = prompt + reply.token_ids
    written = [t for t in reply.token_ids if t not in model.eos_ids]
    parsed = split_reply(model.decode(written), tools)
    calls = openai_tool_calls(parsed.calls)
    copy = harness_copy(copy_of(parsed.content, calls)) | {REASONING: parsed.reasoning}
    results = [
        {"role": TOOL_ROLE, "tool_call_id": call["id"], "content": NOT_RERUN}
        for call in calls
    ]
    rendered = rerender(model, [*messages, copy, *results], tools)
    return describe(model, compare(chain, rendered, len(prompt)), chain, rendered)


def recut(model: Model, reply: Recorded) -> list[str]:
    """Show a recorded reply whose own text cuts into other tokens.

    Args:
        model: The tokenizer.
        reply: The recorded reply.

    Returns:
        The lines to print.
    """
    again = model.encode(model.decode(reply.token_ids))
    position = first_difference(reply.token_ids, again)
    if position is None:
        return ["its text cuts into the same tokens"]
    drawn = [
        f"{model.decode([token])!r} ({logprob:.2f})"
        for token, logprob in zip(
            reply.token_ids[position - 1 : position + 2],
            reply.logprobs[position - 1 : position + 2],
            strict=True,
        )
    ]
    cut = [repr(model.decode([token])) for token in again[position - 1 : position + 1]]
    return [
        f"{len(reply.token_ids)} tokens sampled; its text encodes to {len(again)}",
        f"first difference: token {position + 1} of the reply",
        f"{INDENT}the model drew   {' '.join(drawn)}",
        f"{INDENT}the text cuts to {' '.join(cut)}",
        "putting the text back cannot give these ids back: only the ids could",
    ]


def run_recorded(fast: bool) -> None:
    """Show both recorded replies, with the tokenizer alone.

    Args:
        fast: Use Liquid's export's tokenizer.
    """
    from lfm_text import load_text  # the tokenizer loads here; no weights

    folder, which = pick_model(fast)
    model = load_text(folder)
    print(f"{INDENT}tokenizer: {short_path(folder)} ({which})")
    shown = {RESPELLED: respelled, RECUT: recut}
    for number, reply in enumerate(load_recorded(), start=1):
        text = model.decode(reply.token_ids)
        where_from = f"{reply.task_id}, {reply.harness}, call {reply.call}"
        print(f"\n{number}. {reply.name}: {where_from}")
        print(f"{INDENT}sampled at temperature 1 by the project's engine (GRPO run)")
        print(f"{INDENT}<think>{text}")
        print_lines(shown[reply.name](model, reply))


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recorded", action="store_true", help="the project's two recorded replies"
    )
    parser.add_argument("--fast", action="store_true", help="Liquid's export")
    return parser.parse_args()


def main() -> None:
    """Measure template round-trip drift on a fresh reply, or on recorded ones."""
    args = parse_args()
    print(f"{SCRIPT} · what a harness's copy of a reply does to the tokens")
    if args.recorded:
        run_recorded(args.fast)
    else:
        run_live(args.fast)


if __name__ == "__main__":
    main()
