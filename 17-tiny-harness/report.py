"""How the chapter's scripts print: the setup, each call, the grade, the cost.

Nothing here decides anything. It only turns the harness's records into lines
a person can read on screen (counting and adding up at most), so the harness
and the experiments can stay about the loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from lfm_files import MAX_CALLS, MAX_UNREADABLE, OUTPUT_CAP, gigabytes, short_path
from tasks import ANSWER_FILE, DATABASE_FILE, Task
from toolcalls import THINK_CLOSE

if TYPE_CHECKING:  # for type hints only: these modules print through this one
    from experiments import RecordedEvent, TextRun, TextTurn
    from tiny_harness import Rollout, Turn

INDENT = "  "
LABEL_WIDTH = 13  # "observation" and its gap, so values line up
THINK_OPEN = "<think>"
REASONING_HEAD = 160  # characters of a long reasoning shown before the cut
REASONING_TAIL = 80  # and after it
OUTPUT_LINES = 8  # lines of an observation shown per call
COMPACT_LINES = 2  # and in the one-line-per-call output
ERROR_LINES = 5  # lines of a format error shown, blank ones left out
FULL_LINES = 10_000  # --full: every line
TEXT_SHOWN = 100  # characters of a visible answer shown in compact output
NONE = "none"
STOP_SEPARATOR = ": "  # each stop rule reads "short name: why"
REWARD_RIGHT = 1
REWARD_WRONG = 0
TASK_WIDTH = 20
LEVEL_WIDTH = 8
ANSWER_WIDTH = 22
GRADE_WIDTH = 8
NUMBER_WIDTH = 7
# Plain words for Claude Code's own messages (chapter 1's replay_agent.py).
GLOSSES = {
    "Output token limit hit.": (
        f"(a reply hit the {OUTPUT_CAP:,}-token cap; Claude Code asked it to go on)"
    ),
}


class _Sections:
    """Numbers the sections of one script's output: 1, 2, 3..."""

    count = 0


def print_title(script: str, title: str) -> None:
    """Print the script's first line, and start counting sections again.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    _Sections.count = 0
    print(f"{script} · {title}")


def heading(text: str) -> None:
    """Print the next numbered section's heading after a blank line.

    Args:
        text: What the section shows.
    """
    _Sections.count += 1
    print(f"\n{_Sections.count}. {text}")


def labelled(label: str, text: str, indent: str = INDENT * 2) -> None:
    """Print a label and a value, the value's later lines lined up under it.

    Args:
        label: Such as ``reply`` or ``observation``.
        text: The value, possibly several lines.
        indent: The label's indent.
    """
    lines = text.splitlines() or [""]
    pad = indent + " " * LABEL_WIDTH
    print(f"{indent}{label:<{LABEL_WIDTH}}{lines[0]}")
    for line in lines[1:]:
        print(pad + line)


def first_lines(text: str, count: int) -> str:
    """Keep the first few lines of an output, saying how many were left out.

    Args:
        text: The output.
        count: The most lines kept.

    Returns:
        The lines kept, and a last line counting the rest.
    """
    if not text:
        return "(no output)"
    lines = text.splitlines()
    if len(lines) <= count:
        return text
    return "\n".join([*lines[:count], f"... {len(lines) - count} more lines"])


def show_reply(text: str, full: bool) -> str:
    """Write a reply as the model wrote it, after the prompt's ``<think>``.

    Args:
        text: The reply, special tokens kept.
        full: Print it whole; otherwise a long reasoning is cut in the middle.

    Returns:
        The reply as printed.
    """
    reasoning, closed, rest = text.partition(THINK_CLOSE)
    if not full and len(reasoning) > REASONING_HEAD + REASONING_TAIL:
        left_out = len(reasoning) - REASONING_HEAD - REASONING_TAIL
        reasoning = (
            f"{reasoning[:REASONING_HEAD]}[... {left_out:,} characters ...]"
            f"{reasoning[-REASONING_TAIL:]}"
        )
    return THINK_OPEN + reasoning + closed + rest


def call_header(number: int, reply: object) -> str:
    """Write the first line of a call: its number, its tokens, how it ended.

    Args:
        number: The call's number.
        reply: Its ``policy.Reply``.

    Returns:
        Such as ``call 1 · read 287 tokens, wrote 96 · stop · 2.4 s``.
    """
    return (
        f"{INDENT}call {number} · read {reply.prompt_tokens:,} tokens, wrote "
        f"{reply.reply_tokens:,} · {reply.finish_reason} · {reply.seconds:.1f} s"
    )


def split_line(reasoning: str, content: str, calls: int) -> str:
    """Say what the reply split into.

    Args:
        reasoning: The reasoning.
        content: The visible text.
        calls: How many tool calls parsed.

    Returns:
        One line.
    """
    visible = repr(content) if content else NONE
    plural = "" if calls == 1 else "s"
    return (
        f"reasoning {len(reasoning):,} characters · visible text: {visible} · "
        f"{calls} tool call{plural}"
    )


def print_setup(
    model_line: str, tasks: Sequence[Task], temperature: float, tool_name: str
) -> None:
    """Print the model, the decoding and the harness's rules.

    Args:
        model_line: Which build, and how long loading took.
        tasks: The questions this run asks.
        temperature: The decoding temperature (0: the likeliest token).
        tool_name: The name the tool is offered under.
    """
    heading("the setup")
    print(f"{INDENT}model: {model_line}")
    decoding = (
        "greedy, the likeliest token every time (temperature 0, the project's "
        "evaluation setting)"
        if temperature == 0
        else f"temperature {temperature:g}"
    )
    print(f"{INDENT}decoding: {decoding}; at most {OUTPUT_CAP:,} tokens a reply")
    print(
        f"{INDENT}stop rules: a reply with no tool call · {MAX_CALLS} calls · "
        f"{MAX_UNREADABLE} unreadable replies in a row"
    )
    print(
        f"{INDENT}the tool: one, {tool_name!r}, that runs sqlite3 (read only), ls and "
        f"echo > {ANSWER_FILE}, in the task folder, nothing else"
    )
    if len(tasks) > 1:
        print(f"{INDENT}questions: the first {len(tasks)} held-out ones, in file order")


def print_task(task: Task, workdir: Path) -> None:
    """Print one question, its correct answer and its work folder.

    Args:
        task: The question.
        workdir: Its folder.
    """
    heading("the question")
    print(f"{INDENT}{task.task_id} ({task.difficulty}): {task.question}")
    print(f"{INDENT}the correct answer, which the model never sees: {task.answer}")
    database = Path(task.db_file).name
    print(
        f"{INDENT}work folder: {short_path(workdir)} "
        f"(a fresh copy of {database} as {DATABASE_FILE})"
    )


def print_first_prompt(prompt: str) -> None:
    """Print the first call's prompt exactly as the chat template rendered it.

    Args:
        prompt: The rendered text.
    """
    heading("what the model reads on the first call: the chat template's render")
    for line in prompt.splitlines():
        print(INDENT + line)
    print(
        f"{INDENT}(the template printed the tool's schema into the system prompt, "
        f"'List of tools: [...]'; the prompt ends with {THINK_OPEN}, so every "
        "reply starts inside the model's reasoning)"
    )


def turn_printer(full: bool) -> Callable[[Turn], None]:
    """Make the function that prints each call in full, as it happens.

    Args:
        full: Print each reply and observation whole.

    Returns:
        The printer.
    """

    def show(turn: Turn) -> None:
        """Print one call: the raw reply, its parts, the actions and results.

        Args:
            turn: The call.
        """
        parsed = turn.parsed
        print(call_header(turn.number, turn.reply))
        labelled("reply", show_reply(turn.reply.text, full))
        split = split_line(parsed.reasoning, parsed.content, len(parsed.calls))
        labelled("split", split)
        for action in turn.actions:
            labelled("tool call", action.call.as_json())
            lines = FULL_LINES if full else OUTPUT_LINES
            labelled("observation", first_lines(action.result.output, lines))
        if turn.note is not None:
            labelled("harness", f"unreadable; it tells the model: {turn.note}")
        elif not turn.actions:
            labelled("harness", "no tool call: this is the final answer, so it stops")

    return show


def print_task_line(task: Task) -> None:
    """Print one question's heading in the several-questions output.

    Args:
        task: The question.
    """
    print(f"\n{INDENT}{task.task_id} ({task.difficulty}): {task.question}")
    print(f"{INDENT}correct answer: {task.answer}")


def print_turn_line(turn: Turn) -> None:
    """Print one call in a line or two: what it did and the start of the result.

    Args:
        turn: The call.
    """
    head = f"{INDENT}call {turn.number} · wrote {turn.reply.reply_tokens:,} tokens · "
    pad = " " * len(head)
    if turn.note is not None:
        reason = turn.note.split("(", 1)[-1].split(")", 1)[0]
        print(f"{head}unreadable: {reason}")
        return
    if not turn.actions:
        answer = turn.parsed.content[:TEXT_SHOWN] or NONE
        print(f"{head}no tool call, visible text: {answer!r}")
        return
    for index, action in enumerate(turn.actions):
        print(f"{head if index == 0 else pad}{action.call.as_json()}")
        for line in first_lines(action.result.output, COMPACT_LINES).splitlines():
            print(pad + INDENT + line)


def answer_text(workdir: Path) -> str | None:
    """Read the answer file an attempt left, if any (for printing only).

    Args:
        workdir: The attempt's folder.

    Returns:
        Its first line, or None when there is no answer file.
    """
    path = workdir / ANSWER_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text.splitlines()[0] if text else ""


def print_grade(rollout: Rollout, one: bool) -> None:
    """Print the grader's verdict, the reward and why the attempt stopped.

    Args:
        rollout: The attempt.
        one: True for the one-question output (a section of its own).
    """
    verdict = "correct" if rollout.grade.correct else f"wrong ({rollout.grade.reason})"
    reward = REWARD_RIGHT if rollout.grade.correct else REWARD_WRONG
    calls = f"{len(rollout.turns)} of {MAX_CALLS}"
    if not one:
        short = rollout.stop.split(STOP_SEPARATOR, 1)[0]
        print(f"{INDENT}grade: {verdict} · reward {reward} · {calls} calls · {short}")
        return
    heading("the grade")
    written = answer_text(rollout.workdir)
    print(f"{INDENT}{ANSWER_FILE}: {NONE if written is None else repr(written)}")
    print(f"{INDENT}chapter 1's grader: {verdict} · reward {reward}")
    print(f"{INDENT}calls used: {calls} · stop: {rollout.stop}")


def print_summary(rollouts: Sequence[Rollout]) -> None:
    """Print one row per question, then how many were solved.

    Args:
        rollouts: The attempts.
    """
    heading(f"the {len(rollouts)} questions")
    print(
        f"{INDENT}{'question':<{TASK_WIDTH}}{'level':<{LEVEL_WIDTH}}"
        f"{ANSWER_FILE:<{ANSWER_WIDTH}}{'grade':<{GRADE_WIDTH}}"
        f"{'calls':>{NUMBER_WIDTH}}{'written':>{NUMBER_WIDTH + 2}}  stop"
    )
    for rollout in rollouts:
        written = answer_text(rollout.workdir)
        shown = NONE if written is None else written[: ANSWER_WIDTH - 2]
        verdict = "correct" if rollout.grade.correct else "wrong"
        tokens = sum(turn.reply.reply_tokens for turn in rollout.turns)
        short = rollout.stop.split(STOP_SEPARATOR, 1)[0]
        print(
            f"{INDENT}{rollout.task.task_id:<{TASK_WIDTH}}"
            f"{rollout.task.difficulty:<{LEVEL_WIDTH}}{shown:<{ANSWER_WIDTH}}"
            f"{verdict:<{GRADE_WIDTH}}{len(rollout.turns):>{NUMBER_WIDTH}}"
            f"{tokens:>{NUMBER_WIDTH + 2},}  {short}"
        )
    solved = sum(rollout.grade.correct for rollout in rollouts)
    print(f"{INDENT}solved {solved} of {len(rollouts)}")


def print_cost(rollouts: Sequence[Rollout], peak: int | None, saved: Path) -> None:
    """Print the time, the memory and where every call was saved.

    Args:
        rollouts: The attempts.
        peak: MLX's peak memory in bytes, or None without a model.
        saved: One saved attempt's file (the others are beside it).
    """
    turns = [turn for rollout in rollouts for turn in rollout.turns]
    seconds = sum(turn.reply.seconds for turn in turns)
    heading("what it cost (varies from run to run)")
    print(
        f"{INDENT}{len(turns)} calls took {seconds:.1f} s to read their prompts and "
        "write their replies (every call reads the whole conversation again)"
    )
    if peak is not None:
        print(f"{INDENT}peak memory: {gigabytes(peak)}, the weights included")
    where = saved if len(rollouts) == 1 else saved.parent
    print(f"{INDENT}every call is saved in {short_path(where)}")


def print_name_summary(rollout: Rollout, mark: str) -> None:
    """Count the calls the name experiment spent on a tool that does not exist.

    Args:
        rollout: The attempt with the tool offered as 'Bash'.
        mark: The words that identify the error.
    """
    actions = [action for turn in rollout.turns for action in turn.actions]
    wasted = [action for action in actions if mark in action.result.output]
    names = sorted({action.call.name for action in wasted})
    print(
        f"{INDENT}tool calls: {len(actions)}, of which {len(wasted)} answered "
        f"'{mark}'" + (f" (called as {', '.join(map(repr, names))})" if names else "")
    )


def print_recorded(events: Sequence[RecordedEvent], path: Path, mark: str) -> None:
    """Print the recorded Claude Code run, step by step, and count its errors.

    Args:
        events: The run's steps.
        path: The transcript file.
        mark: The words that identify a tool that does not exist.
    """
    heading(
        "the recorded Claude Code run: the same model, as released, on "
        "clinic-medium-001"
    )
    print(
        f"{INDENT}Claude Code 2.1.290 offered one tool, 'Bash'; recorded by the "
        f"project on 2026-10-06 with greedy decoding and an 8-call budget "
        f"(data/{path.name})"
    )
    calls = [event for event in events if event.kind == "call"]
    for event in events:
        if event.kind == "call":
            labelled(f"{event.tool}", event.command)
            labelled("", first_lines(event.result, COMPACT_LINES))
        elif event.kind == "harness":
            first = event.result.split(".", 1)[0] + "."
            labelled("Claude Code", f"{first} {GLOSSES.get(first, '')}".rstrip())
        else:
            labelled("reply", event.result)
    wasted = Counter(event.tool for event in calls if mark in event.result)
    names = ", ".join(f"{name!r} {count}" for name, count in sorted(wasted.items()))
    print(
        f"{INDENT}tool calls: {len(calls)}, of which {wasted.total()} answered "
        f"'{mark}' ({names}); the last reply is not the model's: the project's "
        "proxy writes it when the 8-call budget is used up"
    )


def print_text_setup(model_line: str, max_calls: int, max_errors: int) -> None:
    """Print the model and the text protocol's rules.

    Args:
        model_line: Which build, and how long loading took.
        max_calls: The step limit.
        max_errors: Format errors in a row that abort the run.
    """
    heading("the setup")
    print(f"{INDENT}model: {model_line}")
    print(
        f"{INDENT}decoding: greedy (temperature 0); at most {OUTPUT_CAP:,} tokens "
        "a reply"
    )
    print(
        f"{INDENT}protocol: mini-swe-agent's text protocol, as the project's "
        "mini_swe_agent_textbased.yaml sets it up: no tool is offered, and each "
        "reply must hold exactly one ```bash block in its visible text"
    )
    print(
        f"{INDENT}limits: {max_calls} calls; the run aborts after {max_errors} "
        "format errors in a row (RepeatedFormatError)"
    )


def print_text_rule(system_prompt: str) -> None:
    """Print the text protocol's system prompt, word for word.

    Args:
        system_prompt: The rendered system template.
    """
    heading("the rule the model reads first (mini-swe-agent's system template)")
    for line in system_prompt.splitlines():
        print(INDENT + line)


def text_turn_printer(full: bool, max_errors: int) -> Callable[[TextTurn], None]:
    """Make the function that prints each text-protocol call as it happens.

    Args:
        full: Print each reply and answer whole.
        max_errors: Format errors in a row that abort the run.

    Returns:
        The printer.
    """

    def show(turn: TextTurn) -> None:
        """Print one call: the raw reply, the visible text, what was found.

        Args:
            turn: The call.
        """
        print(call_header(turn.number, turn.reply))
        labelled("reply", show_reply(turn.reply.text, full))
        visible = turn.parsed.content or NONE
        labelled("visible text", f"{visible}  (the only part mini-swe-agent reads)")
        if len(turn.actions) == 1:
            labelled("bash block", turn.actions[0])
            if turn.submitted:
                labelled("harness", "the finishing command: the run is submitted")
                return
            lines = FULL_LINES if full else OUTPUT_LINES
            labelled("observation", first_lines(turn.answer, lines))
            return
        errors = f"format error {turn.format_errors} of {max_errors}"
        labelled("bash blocks", f"{len(turn.actions)} found: {errors}")
        for call in turn.parsed.calls:
            labelled("unseen", f"a tool call in LFM2.5's own syntax: {call.as_json()}")
        # Blank lines left out, the first few lines hold the error itself.
        message = (
            turn.answer if full else "\n".join(filter(None, turn.answer.splitlines()))
        )
        labelled(
            "harness says", first_lines(message, FULL_LINES if full else ERROR_LINES)
        )

    return show


def print_text_result(run: TextRun, max_calls: int) -> None:
    """Print how the text-protocol run ended and the grade.

    Args:
        run: The attempt.
        max_calls: The step limit.
    """
    heading("the result")
    unseen = sum(len(turn.parsed.calls) for turn in run.turns)
    used = f"{len(run.turns)} of {max_calls} calls"
    print(f"{INDENT}exit status: {run.exit_status} after {used}")
    print(
        f"{INDENT}tool calls written in LFM2.5's own syntax, which this protocol "
        f"never reads: {unseen}"
    )
    verdict = "correct" if run.grade.correct else f"wrong ({run.grade.reason})"
    reward = REWARD_RIGHT if run.grade.correct else REWARD_WRONG
    print(f"{INDENT}chapter 1's grader: {verdict} · reward {reward}")


def print_text_cost(run: TextRun, peak: int | None, saved: Path) -> None:
    """Print the text-protocol run's time, memory and where it was saved.

    Args:
        run: The attempt.
        peak: MLX's peak memory in bytes, or None without a model.
        saved: The saved attempt's file.
    """
    seconds = sum(turn.reply.seconds for turn in run.turns)
    heading("what it cost (varies from run to run)")
    print(f"{INDENT}{len(run.turns)} calls took {seconds:.1f} s")
    if peak is not None:
        print(f"{INDENT}peak memory: {gigabytes(peak)}, the weights included")
    print(f"{INDENT}the conversation is saved in {short_path(saved)}")
