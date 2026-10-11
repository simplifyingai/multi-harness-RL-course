"""How the chapter's scripts print: rows, masks, steps, scores and replies.

Nothing here measures anything. It only turns the scripts' results into lines
a person can read on screen (working out a share or an average at most), so
each script can stay about training and grading.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grading import FINISH_LENGTH, THINK_END, Answered
from lfm_files import (
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    OUTPUT_CAP,
    SFT_BATCH,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    gigabytes,
    short_path,
)
from project_run import RecordedStep, first_below
from rows import FIXED_REASON, REASON_FIXED, SYSTEM_RULE, Row, TrainItem
from stats import Comparison, error_bar, median, pass_at_1
from tasks import Task

if TYPE_CHECKING:  # for type hints only: these modules use MLX
    from sft import Step

INDENT = "  "
WRAP_WIDTH = 88
REPLY_SHOWN = 240  # characters shown of a reply's answer
REASON_SHOWN = 46  # characters shown of a verdict's reason, per question
BEFORE_BOUNDARY = 4  # read tokens shown before the reply starts
REPLY_HEAD = 4  # reply tokens shown from its start
REPLY_TAIL = 2  # and from its end
NEAR_ZERO = 0.10  # "loss near zero", the comment's 0.10
LOSS_DECIMALS = 2
# Colours for --show on a terminal (ANSI escape codes); plain text otherwise.
DIM = "\x1b[2m"
HIGHLIGHT = "\x1b[1;7m"  # bold, reversed
RESET = "\x1b[0m"
RIGHT = "right"
WRONG = "wrong"
DIFFICULTIES = ("easy", "medium", "hard")
# The project's two recorded SFT runs (data/project_run/), as their files and
# mhrl/cli.py describe them: rows trained on, and the step the comment names.
MAIN_RUN_ROWS = 120  # 60 tasks x 2 training harnesses
DRY_RUN_ROWS = 48
COMMENT_STEP = 4  # "loss 0.10 by step 4 of 6"


@dataclass(frozen=True)
class SplitSummary:
    """What one split's rows hold.

    Attributes:
        split: ``"train"`` or ``"test"``.
        rows: How many rows.
        tokens: Their tokens, all positions.
        reply: Their reply tokens (mask 1).
        longest: The longest row's length.
        path: Where they were written.
    """

    split: str
    rows: int
    tokens: int
    reply: int
    longest: int
    path: Path


@dataclass(frozen=True)
class Unrelated:
    """One unrelated prompt, answered before and after training.

    Attributes:
        prompt: What was asked.
        base: The base model's reply text.
        base_tokens: Its length in tokens.
        base_finish: Why it ended.
        tuned: The reply with the adapter.
        tuned_tokens: Its length in tokens.
        tuned_finish: Why it ended.
        base_score: The base reply's log-probability per token, base model.
        tuned_score: The same reply's log-probability per token, with the adapter.
    """

    prompt: str
    base: str
    base_tokens: int
    base_finish: str
    tuned: str
    tuned_tokens: int
    tuned_finish: str
    base_score: float
    tuned_score: float


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


def wrapped(text: str, indent: str = INDENT) -> str:
    """Wrap a long line, indenting its continuation lines a little more.

    Args:
        text: One line of text.
        indent: The indent of the first line.

    Returns:
        The text as one or more lines.
    """
    return textwrap.fill(
        text,
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + INDENT,
        break_on_hyphens=False,
        break_long_words=False,  # a path stays whole, even past the margin
    )


def wrapped_block(text: str, indent: str) -> str:
    """Wrap text that has its own line breaks, keeping each of them.

    Args:
        text: The text.
        indent: The indent of every line.

    Returns:
        The text, each of its lines wrapped.
    """
    return "\n".join(
        wrapped(line, indent) if line.strip() else indent.rstrip()
        for line in text.split("\n")
    )


def short(text: str, limit: int) -> str:
    """Cut text to a length, marking the cut.

    Args:
        text: The text.
        limit: The most characters to keep.

    Returns:
        The text, or its start and ``...``.
    """
    return text if len(text) <= limit else text[: limit - 3] + "..."


def answer_part(reply: str) -> str:
    """Keep what a reply says after its reasoning.

    Args:
        reply: The whole reply.

    Returns:
        The text after the last ``</think>``, or ``""`` if there is none.
    """
    return reply.rsplit(THINK_END, 1)[1].strip() if THINK_END in reply else ""


def setup_line(name: str, text: str) -> None:
    """Print one line of a script's setup, its name in a column.

    Args:
        name: Such as ``"model"``.
        text: What to say.
    """
    lines = textwrap.wrap(
        text, width=WRAP_WIDTH - 10, break_on_hyphens=False, break_long_words=False
    )
    print(f"{name:<10}{lines[0]}")
    for line in lines[1:]:
        print(f"{INDENT}{line}")


# --------------------------------------------------------------- make_rows.py


def print_questions(train: Sequence[Task], test: Sequence[Task]) -> None:
    """Count the bundled questions by split, database and difficulty.

    Args:
        train: The training questions.
        test: The held-out questions.
    """
    total = len(train) + len(test)
    databases = len({task.db_file for task in [*train, *test]})
    heading(f"the questions: {total}, bundled in data/ with {databases} databases")
    print(f"{INDENT}split  databases                easy  medium  hard  total")
    for split, tasks in (("train", train), ("test", test)):
        domains = ", ".join(sorted({task.domain for task in tasks}))
        counts = [
            sum(task.difficulty == level for task in tasks) for level in DIFFICULTIES
        ]
        print(
            f"{INDENT}{split:<5}  {domains:<23}  {counts[0]:>4}  {counts[1]:>6}"
            f"  {counts[2]:>4}  {len(tasks):>5}"
        )
    print(
        wrapped(
            "the held-out databases (clinic, airline) appear in no training row: "
            "their tables and questions are new to the model"
        )
    )


def reason_words(style: str) -> str:
    """Say how a style of reason is made.

    Args:
        style: ``"steps"`` or ``"fixed"``.

    Returns:
        A few words for the output.
    """
    if style == REASON_FIXED:
        return (
            f"one fixed sentence for every row, {FIXED_REASON!r}, the line the "
            "project's answer-key teacher writes before its query (--reason fixed)"
        )
    return "written from the gold SQL: its tables, then what the query does"


def print_shape(example: Task, reply: str, style: str) -> None:
    """Show the chat and reply every row is made of.

    Args:
        example: A question to show the reply for.
        reply: Its reply text.
        style: The kind of reason the replies start with.
    """
    heading("each question becomes a chat and a reply")
    print(wrapped(f"system  {SYSTEM_RULE}"))
    print(
        wrapped(
            "user    'The database has these tables:', its CREATE TABLE lines, then "
            "the question"
        )
    )
    print(wrapped("reply   a one-line reason, </think>, the gold SQL, <|im_end|>"))
    print(wrapped(f"the reason: {reason_words(style)}"))
    print(wrapped(f"for {example.task_id}: {reply!r}"))


def print_rows_summary(summaries: Sequence[SplitSummary]) -> None:
    """Count what the rows hold, split by split.

    Args:
        summaries: One per split.
    """
    heading("the rows: input_ids and a loss_mask per question")
    print(
        f"{INDENT}split  rows  tokens  read (mask 0)  reply (mask 1)  reply share  "
        "longest"
    )
    for item in summaries:
        read = item.tokens - item.reply
        share = 100 * item.reply / item.tokens if item.tokens else 0.0
        print(
            f"{INDENT}{item.split:<5}  {item.rows:>4}  {item.tokens:>6,}  {read:>13,}"
            f"  {item.reply:>14,}  {share:>10.1f}%  {item.longest:>7}"
        )
    print(
        wrapped(
            "training scores only the reply tokens: the loss mask is 1 there and 0 "
            "over everything the model only reads"
        )
    )
    for item in summaries:
        print(wrapped(f"{item.split}: {short_path(item.path)}"))


def token_text(decode: Callable[[list[int]], str], token: int) -> str:
    """Write one token as quoted text.

    Args:
        decode: Turns token ids into text.
        token: The token id.

    Returns:
        Such as ``"' the'"``.
    """
    return repr(decode([token]))


def print_row(
    row: Row,
    number: int,
    decode: Callable[[list[int]], str],
    color: bool,
) -> None:
    """Show one row: the part read (mask 0) and the reply (mask 1).

    Args:
        row: The row.
        number: Its number, from 1, in the training rows.
        decode: Turns token ids into text.
        color: Dim the read part and highlight the reply (on a terminal).
    """
    start = row.prompt_length
    heading(f"row {number} of the training rows: {row.task_id}")
    read = decode(row.input_ids[:start])
    reply = decode(row.input_ids[start:])
    on, off = (DIM, RESET) if color else ("", "")
    print(f"{INDENT}read (mask 0), {start} tokens{', dimmed' if color else ''}:")
    print(on + wrapped_block(read, INDENT * 2) + off)
    print(
        f"{INDENT}reply (mask 1), {row.reply_tokens} tokens"
        f"{', highlighted' if color else ''}:"
    )
    on = HIGHLIGHT if color else ""
    print(on + wrapped_block(reply, INDENT * 2) + off)


def print_boundary(
    row: Row, decode: Callable[[list[int]], str], item: TrainItem, forced: TrainItem
) -> None:
    """Show the tokens where the reply begins, and why position 0 is never a target.

    Args:
        row: The row.
        decode: Turns token ids into text.
        item: The row as the project's ``item_from_sequence`` turns it.
        forced: The same, from a mask with a 1 at position 0 as well.
    """
    start = row.prompt_length
    end = len(row.input_ids)
    shown = [
        *range(start - BEFORE_BOUNDARY, start + REPLY_HEAD),
        *range(end - REPLY_TAIL, end),
    ]
    print(f"{INDENT}where the reply begins, token by token:")
    print(f"{INDENT}{INDENT} pos      id  token            mask  its score is read at")
    previous = None
    for position in shown:
        if previous is not None and position != previous + 1:
            print(f"{INDENT * 2}  ...")
        previous = position
        token = row.input_ids[position]
        mask = row.loss_mask[position]
        before = token_text(decode, row.input_ids[position - 1])
        source = f"position {position - 1} ({before})" if mask else "(read only)"
        print(
            f"{INDENT * 2}{position:>4}  {token:>6}  "
            f"{token_text(decode, token):<15}  {mask:>4}  {source}"
        )
    first = token_text(decode, row.input_ids[0])
    print(
        wrapped(
            f"position 0, {first}: mask 0, and never a target in any row: a token's "
            "log-probability is read from the scores at the position before it, and "
            "nothing comes before position 0"
        )
    )
    print(
        wrapped(
            "the project's item_from_sequence keeps the positions with mask 1 and "
            "i > 0: "
            f"{item.n_tokens} here, the reply's {row.reply_tokens}, from position "
            f"{item.positions[0]}; given a mask with a 1 at position 0 too, it keeps "
            f"{forced.n_tokens}, from position {forced.positions[0]}"
        )
    )


def print_template_check(
    row: Row, rendered: list[int], shared: int, decode: Callable[[list[int]], str]
) -> None:
    """Compare the row with the template's rendering of the finished chat.

    Args:
        row: The row.
        rendered: The template's token ids for the chat with the reply in it.
        shared: How many leading positions the two agree on.
        decode: Turns token ids into text.
    """
    if rendered == row.input_ids:
        print(wrapped("the template renders the finished chat to exactly this row"))
        return
    if shared >= min(len(rendered), len(row.input_ids)):  # one holds the other
        longer = "the template's" if len(rendered) > len(row.input_ids) else "the row"
        difference = f"{longer} goes on past the end of the other"
    else:
        mine = token_text(decode, row.input_ids[shared])
        theirs = token_text(decode, rendered[shared])
        difference = f"the template has {theirs} where the row has {mine}"
        if shared == row.prompt_length - 1:  # the token the reply was written after
            difference += f" (it drops the {mine} that ends the prompt)"
    tail = token_text(decode, rendered[-1])
    print(
        wrapped(
            f"rendering the finished chat with the template instead gives "
            f"{len(rendered)} tokens; the first {shared} match this row, then "
            f"{difference}, and it ends with {tail}. The row is the prompt the model "
            "read, then the tokens it wrote, as the project's engine records them; "
            "nothing is rendered again"
        )
    )


# --------------------------------------------------------------- train_sft.py


def print_train_setup(
    folder: Path,
    which: str,
    rows: Sequence[Row],
    source: str,
    epochs: int,
    lr: float,
    steps: int,
) -> None:
    """Say which model and rows are used, and how they are trained.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        rows: The training rows.
        source: Where the rows came from.
        epochs: Passes over the rows.
        lr: The learning rate.
        steps: How many steps that makes.
    """
    notes = []
    if lr != SFT_LEARNING_RATE:
        notes.append(f"--lr; the project's is {SFT_LEARNING_RATE:g}")
    if epochs != SFT_EPOCHS:
        notes.append(f"--epochs; the project's is {SFT_EPOCHS}")
    note = f" ({'; '.join(notes)})" if notes else ""
    epoch_word = "epoch" if epochs == 1 else "epochs"
    setup_line("model", f"{short_path(folder)}, {which}")
    reply = sum(row.reply_tokens for row in rows)
    setup_line("rows", f"{len(rows)} training rows ({source}), {reply:,} reply tokens")
    setup_line(
        "training",
        f"{epochs} {epoch_word}, {SFT_BATCH} rows a step: {steps} AdamW steps at "
        f"learning rate {lr:g}, bias correction on{note}; the project's SFT trainer "
        "and settings (mhrl/training/sft.py, mhrl/cli.py)",
    )


def print_reason_style(
    style: str, row: Row, decode: Callable[[list[int]], str]
) -> None:
    """Say which kind of reason the rows' replies start with, and show one reply.

    Args:
        style: ``"steps"`` or ``"fixed"``.
        row: The first training row.
        decode: Turns token ids into text.
    """
    setup_line("reason", reason_words(style))
    setup_line("e.g.", repr(decode(row.input_ids[row.prompt_length :])))


def print_fresh(
    pairs: int, trainable: int, zeros: tuple[int, int], difference: float, shape: tuple
) -> None:
    """Show that the fresh adapter changes nothing, before any training.

    Args:
        pairs: How many adapter pairs were added.
        trainable: How many numbers they hold.
        zeros: All-zero B matrices, and how many B matrices there are.
        difference: Max logit difference against the model before LoRA.
        shape: The scores' shape (positions, vocabulary).
    """
    heading(
        f"a fresh adapter built the chapter-13 way: rank {LORA_RANK}, scale "
        f"{LORA_SCALE}, the last {LORA_LAYERS} layers, seed 0"
    )
    zero, count = zeros
    print(
        wrapped(
            f"{pairs} adapter pairs, {trainable:,} trainable numbers; B: {zero} of "
            f"{count} matrices all zeros"
        )
    )
    positions, vocabulary = shape
    print(
        wrapped(
            f"max |logit difference| from the model before LoRA, over row 1's "
            f"{positions} positions x {vocabulary:,} scores: {difference}; checked "
            "before training: the adapter starts as a no-op"
        )
    )


def print_steps_header() -> None:
    """Start the table of training steps."""
    heading("training: one forward pass per row, gradients added up row by row")
    print(f"{INDENT}step  epoch  rows  reply tokens  loss per token  seconds  peak")


def print_step(step: Step) -> None:
    """Print one training step as soon as it is done.

    Args:
        step: What the step did.
    """
    print(
        f"{INDENT}{step.step:>4}  {step.epoch:>5}  {step.rows:>4}  {step.tokens:>12,}"
        f"  {step.loss_per_token:>14.4f}  {step.seconds:>7.1f}"
        f"  {gigabytes(step.peak_bytes):>4}",
        flush=True,
    )


def print_steps_footer(steps: Sequence[Step]) -> None:
    """Explain the table, and say what varies from run to run.

    Args:
        steps: Every step taken.
    """
    print(
        wrapped(
            "loss per token: minus the sum of the reply tokens' log-probabilities, "
            "divided by the step's reply-token count, before the step's update"
        )
    )
    if not steps:
        return
    seconds = sum(step.seconds for step in steps)
    peak = max(step.peak_bytes for step in steps)
    print(
        wrapped(
            f"varies from run to run: the seconds ({len(steps)} steps took "
            f"{seconds:.0f} s) and the peak memory ({gigabytes(peak)}, the weights "
            "included)"
        )
    )


def print_epochs(steps: Sequence[Step]) -> None:
    """Average the loss over each epoch, when there was more than one.

    Args:
        steps: Every step taken.
    """
    epochs = sorted({step.epoch for step in steps})
    if len(epochs) < 2:
        return
    heading("loss per token, epoch by epoch (the same rows each time, reshuffled)")
    for epoch in epochs:
        mine = [step for step in steps if step.epoch == epoch]
        tokens = sum(step.tokens for step in mine)
        loss = sum(step.loss_per_token * step.tokens for step in mine) / tokens
        print(
            f"{INDENT}epoch {epoch}: {loss:.4f} over its {tokens:,} reply tokens, "
            f"last step {mine[-1].loss_per_token:.4f}"
        )


def print_saved(adapter: Path, size: int, metrics: Path) -> None:
    """Say where the adapter and the step log went, and what to run next.

    Args:
        adapter: The adapter file.
        size: Its size in bytes.
        metrics: The step log.
    """
    heading("saved")
    print(wrapped(f"{short_path(adapter)} ({size:,} bytes)"))
    print(wrapped(f"{short_path(metrics)} (one line per step)"))
    print(wrapped(f"next: uv run eval_sql.py --adapter {adapter.name}"))


def losses(steps: Sequence[RecordedStep]) -> str:
    """List a recorded run's losses, two decimals each.

    Args:
        steps: The run's steps.

    Returns:
        Such as ``"2.55 2.28 1.77"``.
    """
    return " ".join(f"{step.loss_per_token:.{LOSS_DECIMALS}f}" for step in steps)


def print_project_runs(
    main: Sequence[RecordedStep], dry: Sequence[RecordedStep]
) -> None:
    """Show the project's own SFT runs, from their recorded metrics.

    Args:
        main: The main run's steps.
        dry: The dry run's steps.
    """
    heading("the project's runs (recorded, not run here)")
    seconds = [step.seconds for step in main]
    peaks = [step.peak_gb for step in main]
    tokens = [step.tokens for step in main]
    print(
        wrapped(
            f"main run, runs/main/sft/metrics.jsonl: the same trainer, {SFT_EPOCHS} "
            f"epoch, {SFT_BATCH} rows a step, lr {SFT_LEARNING_RATE:g}, on "
            f"{MAIN_RUN_ROWS} rows captured through mini-swe-agent and Claude Code "
            f"(chapter 25): {len(main)} steps"
        )
    )
    print(wrapped(f"loss per token: {losses(main)}", INDENT * 2))
    print(
        wrapped(
            f"{min(tokens):,} to {max(tokens):,} reply tokens a step; "
            f"{min(seconds):.2f} to {max(seconds):.2f} s a step "
            f"({sum(seconds) / 60:.1f} minutes); peak {min(peaks):.2f} to "
            f"{max(peaks):.2f} GB",
            INDENT * 2,
        )
    )
    reached = first_below(dry, NEAR_ZERO)
    named = next((step for step in dry if step.step == COMMENT_STEP), None)
    print(
        wrapped(
            f"dry run, runs/dry/sft/metrics.jsonl: {DRY_RUN_ROWS} rows, {len(dry)} "
            "steps; lr 1e-4 according to the comment in mhrl/cli.py (the file "
            "records no learning rate)"
        )
    )
    print(wrapped(f"loss per token: {losses(dry)}", INDENT * 2))
    if named is not None:
        when = f"first at step {reached}" if reached else "at no step"
        print(
            wrapped(
                f"the comment says 'loss 0.10 by step {COMMENT_STEP} of 6'; the file "
                f"says {named.loss_per_token:.2f} at step {COMMENT_STEP} and "
                f"{NEAR_ZERO:.2f} {when}",
                INDENT * 2,
            )
        )


# --------------------------------------------------------------- eval_sql.py


def print_eval_setup(
    folder: Path, which: str, adapter: Path | None, tasks: Sequence[Task], cap: int
) -> None:
    """Say which model, adapter and questions the evaluation uses.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        adapter: The adapter file, or None for the base model.
        tasks: The held-out questions asked.
        cap: The reply length cap.
    """
    setup_line("model", f"{short_path(folder)}, {which}")
    setup_line(
        "adapter",
        "none: the base model" if adapter is None else short_path(adapter),
    )
    domains = " and ".join(sorted({task.domain for task in tasks}))
    setup_line(
        "questions",
        f"{len(tasks)} held out, over {domains}: databases no training row comes from",
    )
    cap_note = "the project's cap" if cap == OUTPUT_CAP else "--cap"
    setup_line(
        "replies",
        f"greedy, one per question, up to {cap:,} tokens ({cap_note}), to the chat "
        "make_rows.py builds",
    )


def print_answers_header(label: str) -> None:
    """Start the question-by-question table.

    Args:
        label: Which model answers (``base``, or the adapter's name).
    """
    heading(
        f"{label}: each reply's SQL run, its result written to answer.txt, graded by "
        "chapter 1's grader"
    )
    print(f"{INDENT}task                 verdict  tokens  why")


def print_answer(answered: Answered) -> None:
    """Print one question's verdict as soon as it is graded.

    Args:
        answered: The graded answer.
    """
    verdict = RIGHT if answered.correct else WRONG
    print(
        f"{INDENT}{answered.task_id:<19}  {verdict:<7}  {answered.tokens:>6}  "
        f"{short(answered.reason, REASON_SHOWN)}",
        flush=True,
    )


def print_pass(
    label: str, answers: Sequence[Answered], tasks: Sequence[Task], seconds: float
) -> None:
    """Print pass@1 with its error bar, and how it splits by database and difficulty.

    Args:
        label: Which model answered.
        answers: One per question, in question order.
        tasks: The questions, in the same order.
        seconds: How long all the replies took.
    """
    correct = sum(answered.correct for answered in answers)
    total = len(answers)
    heading(
        f"{label}: pass@1 {correct} of {total} = {pass_at_1(correct, total):.1f}% "
        f"± {error_bar(correct, total):.1f} points"
    )
    print(
        wrapped(
            "pass@1: the share answered right on the one greedy try; error bar: ± one "
            "standard error, sqrt(p x (1 - p) / n), how far the score would typically "
            f"move by luck alone if another {total} questions like these were asked"
        )
    )
    by_task = {task.task_id: task for task in tasks}
    for name, key in (("database", "domain"), ("difficulty", "difficulty")):
        groups: dict[str, list[bool]] = {}
        for answered in answers:
            value = getattr(by_task[answered.task_id], key)
            groups.setdefault(value, []).append(answered.correct)
        parts = [f"{value} {sum(v)}/{len(v)}" for value, v in sorted(groups.items())]
        print(wrapped(f"by {name}: {', '.join(parts)}"))
    forms: dict[str, int] = {}
    for answered in answers:
        name = answered.form or "no query"
        forms[name] = forms.get(name, 0) + 1
    parts = [f"{name} {count}" for name, count in sorted(forms.items())]
    print(wrapped(f"the query came as: {', '.join(parts)}"))
    tokens = [answered.tokens for answered in answers] or [0]
    capped = sum(answered.finish_reason == FINISH_LENGTH for answered in answers)
    print(
        wrapped(
            f"reply length: median {median(tokens):.0f} tokens, longest {max(tokens):,}"
            f"; {capped} cut off at the cap"
        )
    )
    print(wrapped(f"varies from run to run: the {total} replies took {seconds:.0f} s"))


def print_replies(
    title: str, answers: Sequence[Answered], tasks: Mapping[str, Task]
) -> None:
    """Show some replies in full: the question, the answer part and the verdict.

    Args:
        title: The section's heading.
        answers: The replies to show.
        tasks: Every question, by id.
    """
    heading(title)
    for answered in answers:
        task = tasks[answered.task_id]
        verdict = RIGHT if answered.correct else WRONG
        print(f"{INDENT}{answered.task_id}: {verdict}")
        print(wrapped(f"question  {task.question}", INDENT * 2))
        said = answer_part(answered.reply)
        if not said:
            said = f"(no </think> in {answered.tokens} tokens: still reasoning)"
        print(
            wrapped(
                f"after </think> ({answered.tokens} tokens in all): "
                f"{short(said, REPLY_SHOWN)!r}",
                INDENT * 2,
            )
        )
        if answered.sql is not None:
            print(wrapped(f"query ({answered.form})  {answered.sql}", INDENT * 2))
        if answered.correct:
            print(wrapped(f"answer.txt  {answered.answer}", INDENT * 2))
        else:
            print(wrapped(f"why  {answered.reason}", INDENT * 2))
            print(wrapped(f"gold SQL  {task.gold_sql}", INDENT * 2))


def print_comparison(label: str, comparison: Comparison) -> None:
    """Compare the adapter's answers with the base model's, question by question.

    Args:
        label: The adapter's label.
        comparison: The two runs, compared.
    """
    gained, lost = len(comparison.gained), len(comparison.lost)
    heading(f"base against {label}, question by question")
    print(
        wrapped(
            f"gained {gained} (wrong before, right now), lost {lost} (right before, "
            f"wrong now), right both times {comparison.both}, wrong both times "
            f"{comparison.neither}"
        )
    )
    print(
        wrapped(
            f"exact p = {comparison.p_value:.4f}: the chance of a split at least this "
            f"uneven among the {gained + lost} questions that changed, if the two "
            "models were equally good (McNemar's exact test)"
        )
    )
    for name, ids in (("gained", comparison.gained), ("lost", comparison.lost)):
        if ids:
            print(wrapped(f"{name}: {', '.join(ids)}"))


def print_unrelated(results: Sequence[Unrelated]) -> None:
    """Show what else the adapter changed: replies to prompts that are not SQL.

    Args:
        results: One per prompt.
    """
    heading("what else the adapter changed: five prompts that are not about SQL")
    print(
        wrapped(
            "each asked as a plain question (no system rule, no tables), greedy, "
            "before training (a fresh adapter: every B is zero, so this is the base "
            "model exactly) and after"
        )
    )
    for result in results:
        print(f"{INDENT}{result.prompt!r}")
        for name, text, tokens, finish in (
            ("before", result.base, result.base_tokens, result.base_finish),
            ("after ", result.tuned, result.tuned_tokens, result.tuned_finish),
        ):
            said = answer_part(text) or "(no </think>: still reasoning at the cap)"
            print(
                wrapped(
                    f"{name} ({tokens} tokens, '{finish}'): "
                    f"{short(said, REPLY_SHOWN)!r}",
                    INDENT * 2,
                )
            )
        print(
            wrapped(
                f"the before reply's log-probability per token: {result.base_score:.4f}"
                f" before, {result.tuned_score:.4f} after",
                INDENT * 2,
            )
        )
    if not results:
        return
    sql = sum("SELECT" in answer_part(result.tuned).upper() for result in results)
    change = sum(r.tuned_score - r.base_score for r in results) / len(results)
    direction = "lower" if change < 0 else "higher"
    print(
        wrapped(
            f"after training, {sql} of {len(results)} replies contain SELECT; the "
            "base model's own replies became, on average, "
            f"{abs(change):.4f} {direction} in log-probability per token"
        )
    )
