# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Drive the engine by hand: one reply's receipts, the cache, the queue, the limits.

Four demonstrations, each a few plain HTTP requests:

- ``turns`` (the default): the first call of a question, shown as the JSON
  the engine returns, token fields and all; then call 2 of the same
  conversation, whose prompt begins with call 1's prompt and reply, so a
  prompt cache already holds that work (``cached_tokens``) and the first
  token comes sooner;
- ``queue``: four requests at once, as four rollouts would send them; the
  engine's one worker thread answers them one after another;
- ``too-long``: a prompt of 20,000 words, refused before the model reads it;
- ``logprobs``: each sampled token's log-probability with 5 alternatives,
  then chapter 20's probe, OpenEnv's own test (``probe_logprobs_mode``): the
  gap between the two likeliest first tokens at temperature 1 and at 2. If
  the log-probabilities are the sampling distribution's, ``scores / T``, the
  gap halves; if they are the raw scores', it stays.

Run ``uv run client.py [turns|queue|too-long|logprobs]`` while ``engine.py``
runs; ``--base-url`` for another address. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from lfm_files import EVAL_TEMPERATURE, work_folder
from tasks import find_task, load_tasks, prepare_workdir
from tiny_harness_http import (
    ASSISTANT,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_TASK,
    MODEL_NAME,
    TOOL_ROLE,
    USER,
    Chat,
    EngineError,
    bash_schema,
    execute,
    http_chat,
    start_messages,
)

SCRIPT = "client.py"
DEMOS = ("turns", "queue", "too-long", "logprobs")
LABEL = "client"  # this script's work folders
QUEUE_SIZE = 4  # the project runs four rollouts at once
LONG_WORDS = 20_000
LONG_WORD = "data"
GO_ON = "Go on."  # call 2's nudge, only if call 1 called no tool
SHOWN_IDS = 6  # ids of a long list printed before the cut
SHOWN_ENTRIES = 2  # logprob entries printed before the cut
LOGPROB_TOKENS = 4  # tokens drawn to show their log-probabilities
TOP_SHOWN = 5  # alternatives asked for per position
TOO_MANY = 21  # one above the ceiling of 20
DRAW_TEMPERATURE = 1.0  # training's temperature (chapter 15)
# Chapter 20's probe (OpenEnv's probe_logprobs_mode), with its own numbers.
PROBE_PROMPT = "Reply with exactly: hello"
PROBE_TEMPERATURES = (1.0, 2.0)
PROBE_TOP = 5
PROBE_MIN_GAP = 0.5  # flatter than this, the ratio would be noise
SENTINEL = 100.0  # a "logprob" this large is an engine's stand-in for -inf
PROCESSED, RAW, UNKNOWN = "processed", "raw", "unknown"
INDENT = "  "


@dataclass(frozen=True)
class Probe:
    """What chapter 20's probe measured.

    Attributes:
        gaps: The gap between the two likeliest first tokens, per temperature.
        ratio: The second gap over the first, or None when unmeasurable.
        verdict: ``processed`` (tempered), ``raw`` or ``unknown``.
    """

    gaps: list[float]
    ratio: float | None
    verdict: str


def request(messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """Write a chat request for the engine.

    Args:
        messages: The conversation.
        **fields: Anything else (tools, temperature, max_tokens...).

    Returns:
        The request body.
    """
    return {"model": MODEL_NAME, "messages": messages} | fields


def cut_list(values: list[Any], shown: int = SHOWN_IDS) -> list[Any]:
    """Keep the start of a long list, saying how long it was.

    Args:
        values: The list.
        shown: How many items to keep.

    Returns:
        The list, or its first items and a note.
    """
    if len(values) <= shown:
        return values
    return [*values[:shown], f"... {len(values):,} in all"]


def abbreviate(response: dict[str, Any]) -> dict[str, Any]:
    """Shorten a response's long lists for printing; everything else stays.

    Args:
        response: The engine's answer.

    Returns:
        A copy whose token lists show their start and their length.
    """
    shown = json.loads(json.dumps(response))  # a deep copy
    shown["prompt_token_ids"] = cut_list(shown["prompt_token_ids"])
    choice = shown["choices"][0]
    choice["token_ids"] = cut_list(choice["token_ids"])
    entries = choice["logprobs"]["content"]
    choice["logprobs"]["content"] = cut_list(entries, SHOWN_ENTRIES)
    return shown


def follow_up(
    response: dict[str, Any], tools: list[dict[str, Any]], task_id: str
) -> list[dict[str, Any]]:
    """Write what a harness appends after a reply: the reply, then each result.

    Args:
        response: The engine's answer.
        tools: The tools offered.
        task_id: The question (its work folder runs the commands).

    Returns:
        The messages to append.
    """
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    kept = {"role": ASSISTANT, "content": message.get("content")}
    if not calls:
        return [kept, {"role": USER, "content": GO_ON}]
    workdir = prepare_workdir(find_task(task_id), work_folder(LABEL, task_id))
    results = [
        {"role": TOOL_ROLE, "tool_call_id": call["id"]}
        | {"content": execute(call, tools, workdir).output}
        for call in calls
    ]
    return [kept | {"tool_calls": calls}, *results]


def continues(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Say whether a call's prompt begins with the previous prompt and reply.

    Args:
        before: The earlier response.
        after: The later response.

    Returns:
        True when the later prompt extends the earlier chain exactly.
    """
    chain = before["prompt_token_ids"] + before["choices"][0]["token_ids"]
    return after["prompt_token_ids"][: len(chain)] == chain


def call_line(name: str, response: dict[str, Any]) -> str:
    """Write one call's tokens and timings on a line.

    Args:
        name: Such as ``call 1``.
        response: Its response.

    Returns:
        The line.
    """
    usage, timings = response["usage"], response["timings"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return (
        f"{INDENT}{name}: read {usage['prompt_tokens']:,} tokens, {cached:,} cached"
        f" · first token after {timings['first_token_s']:.2f} s · whole reply "
        f"{timings['total_s']:.2f} s"
    )


def demo_turns(chat: Chat) -> None:
    """Show one reply's JSON, then call 2 of the same conversation.

    Args:
        chat: Sends one request.
    """
    task = find_task(DEFAULT_TASK)
    tools = [bash_schema()]
    messages = start_messages(task)
    fields = {"tools": tools, "temperature": EVAL_TEMPERATURE}
    print(f"\n1. {task.task_id}, call 1: the engine's answer (long lists cut)")
    first = chat(request(messages, **fields))
    print(json.dumps(abbreviate(first), indent=2, ensure_ascii=False))
    messages += follow_up(first, tools, task.task_id)
    print("\n2. call 2 of the same conversation: call 1's reply, then its result")
    second = chat(request(messages, **fields))
    print(call_line("call 1", first))
    print(call_line("call 2", second))
    verdict = "yes" if continues(first, second) else "no"
    print(f"{INDENT}call 2's prompt begins with call 1's prompt and reply: {verdict}")


def demo_queue(chat: Chat) -> None:
    """Send four requests at once and show the worker answer them in turn.

    Args:
        chat: Sends one request.
    """
    tasks = load_tasks()[:QUEUE_SIZE]
    tools = [bash_schema()]
    gate = threading.Barrier(QUEUE_SIZE)  # all four leave at the same moment
    start = time.perf_counter()

    def one(task_id: str) -> tuple[str, float, dict[str, Any]]:
        """Send one question's first call; return when it was answered."""
        body = request(
            start_messages(find_task(task_id)),
            tools=tools,
            temperature=EVAL_TEMPERATURE,
        )
        gate.wait()
        response = chat(body)
        return task_id, time.perf_counter() - start, response

    print(f"\n{QUEUE_SIZE} requests at once, one worker thread")
    with ThreadPoolExecutor(max_workers=QUEUE_SIZE) as pool:
        done = list(pool.map(one, [task.task_id for task in tasks]))
    done.sort(key=lambda item: item[1])
    print(f"{INDENT}{'question':<20}{'waited':>8}{'model':>8}{'answered':>10}")
    for task_id, answered, response in done:
        timings = response["timings"]
        print(
            f"{INDENT}{task_id:<20}{timings['waited_s']:>7.2f}s"
            f"{timings['total_s']:>7.2f}s{answered:>9.2f}s"
        )
    model = sum(response["timings"]["total_s"] for _, _, response in done)
    print(
        f"{INDENT}the four model times add up to {model:.2f} s; the last answer "
        f"came after {done[-1][1]:.2f} s: one request at a time"
    )


def demo_too_long(chat: Chat) -> None:
    """Send a prompt over the limit and show the refusal.

    Args:
        chat: Sends one request.
    """
    words = " ".join([LONG_WORD] * LONG_WORDS)
    print(f"\na prompt of {LONG_WORDS:,} words ({LONG_WORD!r} again and again)")
    try:
        chat(request([{"role": USER, "content": words}], max_tokens=1))
    except EngineError as error:
        if error.status is None:  # nobody answered: not a refusal
            raise
        print(f"{INDENT}{error}")
        return
    print(f"{INDENT}answered: the engine did not refuse it")


def token_text(entry: dict[str, Any]) -> str:
    """Write a logprob entry's token as text.

    Args:
        entry: ``{"token", "logprob", "bytes"}``.

    Returns:
        Its text, quoted.
    """
    return repr(bytes(entry["bytes"]).decode("utf-8", errors="replace"))


def probe(chat: Chat) -> Probe:
    """Run chapter 20's probe: does the gap between the top two tokens halve?

    Args:
        chat: Sends one request.

    Returns:
        What it measured.
    """
    gaps = []
    for temperature in PROBE_TEMPERATURES:
        body = request(
            [{"role": USER, "content": PROBE_PROMPT}],
            max_tokens=1,
            temperature=temperature,
            logprobs=True,
            top_logprobs=PROBE_TOP,
        )
        entries = chat(body)["choices"][0]["logprobs"]["content"]
        first = entries[0]["top_logprobs"] if entries else []
        values = sorted((entry["logprob"] for entry in first), reverse=True)
        if len(values) < 2 or abs(values[1]) > SENTINEL:
            return Probe(gaps, None, UNKNOWN)
        gaps.append(values[0] - values[1])
    if gaps[0] < PROBE_MIN_GAP:
        return Probe(gaps, None, UNKNOWN)
    ratio = gaps[1] / gaps[0]
    expected = PROBE_TEMPERATURES[0] / PROBE_TEMPERATURES[1]  # 0.5 when tempered
    verdict = RAW if abs(ratio - 1.0) < abs(ratio - expected) else PROCESSED
    return Probe(gaps, ratio, verdict)


def demo_logprobs(chat: Chat) -> None:
    """Show log-probabilities and alternatives, then run chapter 20's probe.

    Args:
        chat: Sends one request.
    """
    print(
        f"\n1. {LOGPROB_TOKENS} tokens drawn at temperature {DRAW_TEMPERATURE}, "
        f"{TOP_SHOWN} alternatives each"
    )
    body = request(
        [{"role": USER, "content": PROBE_PROMPT}],
        max_tokens=LOGPROB_TOKENS,
        temperature=DRAW_TEMPERATURE,
        top_logprobs=TOP_SHOWN,
    )
    entries = chat(body)["choices"][0]["logprobs"]["content"]
    for number, entry in enumerate(entries, start=1):
        others = " · ".join(
            f"{token_text(alt)} {alt['logprob']:.3f}" for alt in entry["top_logprobs"]
        )
        print(f"{INDENT}token {number}: {token_text(entry)} {entry['logprob']:.3f}")
        print(f"{INDENT * 2}likeliest: {others}")
    print("\n2. chapter 20's probe: the gap between the two likeliest first tokens")
    measured = probe(chat)
    for temperature, gap in zip(PROBE_TEMPERATURES, measured.gaps, strict=False):
        print(f"{INDENT}temperature {temperature}: gap {gap:.4f}")
    ratio = "unmeasurable" if measured.ratio is None else f"{measured.ratio:.3f}"
    print(f"{INDENT}ratio {ratio} -> {measured.verdict}")
    print(f"\n3. asking for {TOO_MANY} alternatives")
    try:
        chat(body | {"top_logprobs": TOO_MANY})
    except EngineError as error:
        if error.status is None:  # nobody answered: not a refusal
            raise
        print(f"{INDENT}{error}")
        return
    print(f"{INDENT}answered: the engine did not refuse it")


DEMO_FUNCTIONS = {
    "turns": demo_turns,
    "queue": demo_queue,
    "too-long": demo_too_long,
    "logprobs": demo_logprobs,
}


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("demo", nargs="?", choices=DEMOS, default=DEMOS[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    return parser.parse_args()


def main() -> None:
    """Run one demonstration against the engine."""
    args = parse_args()
    print(f"{SCRIPT} {args.demo} · {args.base_url}")
    chat = http_chat(args.base_url, args.api_key)
    try:
        DEMO_FUNCTIONS[args.demo](chat)
    except EngineError as error:
        raise SystemExit(f"the engine did not answer: {error}") from error


if __name__ == "__main__":
    main()
