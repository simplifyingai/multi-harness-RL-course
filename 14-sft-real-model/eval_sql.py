# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Measure pass@1 on the 50 held-out questions, with or without an SFT adapter.

One greedy reply per held-out question (clinic and airline, databases no
training row comes from), to the same chat make_rows.py builds, up to the
project's 1,024-token cap. The reply's SQL runs on a read-only copy of the
database, its result is written to ``answer.txt`` in the question's own
folder, and chapter 1's grader judges that file. It prints every verdict,
pass@1 with its error bar, and a few replies.

With ``--adapter sft.safetensors`` (a file train_sft.py saved), the same
questions are answered with the adapter and compared with the base model's
answers question by question: gained, lost, and the exact p-value. Then
five prompts that have nothing to do with SQL are answered before and after,
to show what else the adapter changed.

Run it with ``uv run eval_sql.py`` (several minutes: the base model reasons
at length) and ``uv run eval_sql.py --adapter sft.safetensors``. ``--fast``
uses Liquid's export, 10 questions and no unrelated prompts; ``--limit N``
asks N balanced questions. Results go under
``$COURSE_HOME/14-sft-real-model``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import mlx.nn as nn
import report
from grading import (
    FINISH_LENGTH,
    Answered,
    grade_reply,
    read_answers,
    write_answers,
)
from lfm import Reply, greedy_reply, load_folder
from lfm_files import (
    BASE_LABEL,
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    OUTPUT_CAP,
    adapter_path,
    eval_label,
    eval_path,
    pick_model,
    short_path,
    work_folder,
)
from lora import apply_lora, load_adapter
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rows import USER_ROLE, balanced, encode, prompt_ids, render_prompt
from sft import reply_logprob
from stats import compare
from tasks import TEST_FILE, Task, load_tasks

SCRIPT = "eval_sql.py"
TITLE = "pass@1 on the held-out questions"
FAST_QUESTIONS = 10
SHOWN_RIGHT = 2  # replies shown: the base model's first right ones...
SHOWN_WRONG = 2  # ...and its first wrong ones, the same questions after training
SHOWN_CHANGED = 2  # and up to this many gained and lost
# Prompts unlike the training rows: no system rule, no tables, no SQL.
UNRELATED_PROMPTS = (
    "Write a haiku about the sea.",
    "Translate into French: The library opens at nine in the morning.",
    "What is the capital of Australia?",
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. "
    "When does it arrive?",
    "Give three short tips for sleeping well.",
)


def answer_all(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    tasks: Sequence[Task],
    label: str,
    cap: int,
) -> tuple[list[Answered], float]:
    """Answer every question greedily, grade each reply, and save the answers.

    Args:
        model: The network, with or without an adapter.
        tokenizer: Its tokenizer.
        tasks: The questions.
        label: The evaluation's name (``base`` or the adapter's).
        cap: The most tokens a reply may have.

    Returns:
        One graded answer per question, and how long they all took.
    """
    report.print_answers_header(label)
    answers = []
    start = time.perf_counter()
    for task in tasks:
        reply = greedy_reply(model, tokenizer, prompt_ids(tokenizer, task), cap)
        text = tokenizer.decode(reply.tokens)
        cut_off = reply.finish_reason == FINISH_LENGTH
        verdict = grade_reply(task, text, work_folder(label, task.task_id), cut_off)
        answered = Answered(
            task_id=task.task_id,
            correct=verdict.correct,
            reason=verdict.reason,
            sql=verdict.sql,
            answer=verdict.answer,
            reply=text,
            tokens=len(reply.tokens),
            finish_reason=reply.finish_reason,
            seconds=reply.seconds,
            form=verdict.form,
            cap=cap,
        )
        report.print_answer(answered)
        answers.append(answered)
    seconds = time.perf_counter() - start
    # The base model's record keeps answers to other questions (same cap), so
    # a 10-question run never erases the 50-question one; an adapter's file
    # is replaced, since a retrained adapter keeps its name.
    write_answers(eval_path(label), answers, keep_others=label == BASE_LABEL)
    return answers, seconds


def shown_ids(answers: Sequence[Answered]) -> list[str]:
    """Pick the replies to show: the first right ones and the first wrong ones.

    Args:
        answers: The base model's answers, in question order.

    Returns:
        Their task ids, right ones first.
    """
    right = [a.task_id for a in answers if a.correct][:SHOWN_RIGHT]
    wrong = [a.task_id for a in answers if not a.correct][:SHOWN_WRONG]
    return right + wrong


def plain_prompt(tokenizer: TokenizerWrapper, text: str) -> list[int]:
    """Render a plain question, with no system rule, as chapters 10 to 12 did.

    Args:
        tokenizer: The model's tokenizer.
        text: The question.

    Returns:
        The prompt's token ids.
    """
    messages = [{"role": USER_ROLE, "content": text}]
    return encode(tokenizer, render_prompt(tokenizer, messages))


def per_token(model: nn.Module, prompt: list[int], reply: Reply) -> float:
    """Score a reply under the model's current weights, per token.

    Args:
        model: The network.
        prompt: The prompt's token ids.
        reply: The reply.

    Returns:
        Its log-probability divided by its length.
    """
    return reply_logprob(model, prompt, reply.tokens) / max(len(reply.tokens), 1)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter", default=None, help="an adapter file train_sft.py saved"
    )
    parser.add_argument("--limit", type=int, default=None, help="ask N questions")
    parser.add_argument("--cap", type=int, default=OUTPUT_CAP, help="reply cap")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Liquid's export, {FAST_QUESTIONS} questions, no unrelated prompts",
    )
    args = parser.parse_args()
    if args.fast and args.limit is None:
        args.limit = FAST_QUESTIONS
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.cap < 1:
        parser.error("--cap must be at least 1")
    return args


def evaluate_base(
    model: nn.Module, tokenizer: TokenizerWrapper, tasks: list[Task], cap: int
) -> None:
    """Answer the questions with the base model and print what it got right.

    Args:
        model: The base model.
        tokenizer: Its tokenizer.
        tasks: The questions.
        cap: The reply cap.
    """
    answers, seconds = answer_all(model, tokenizer, tasks, BASE_LABEL, cap)
    report.print_pass(BASE_LABEL, answers, tasks, seconds)
    by_id = {a.task_id: a for a in answers}
    report.print_replies(
        "a few replies: the first right ones and the first wrong ones",
        [by_id[i] for i in shown_ids(answers)],
        {task.task_id: task for task in tasks},
    )
    print(report.wrapped(f"saved: {short_path(eval_path(BASE_LABEL))}"))


def base_answers(
    model: nn.Module, tokenizer: TokenizerWrapper, tasks: list[Task], cap: int
) -> dict[str, Answered]:
    """Read the base model's saved answers, or answer now if they are missing.

    Args:
        model: The model with a fresh adapter (the base model exactly).
        tokenizer: Its tokenizer.
        tasks: The questions.
        cap: The reply cap.

    Returns:
        The base model's answer to every question, by task id.
    """
    saved = read_answers(eval_path(BASE_LABEL))
    name = eval_path(BASE_LABEL).name
    if all(t.task_id in saved and saved[t.task_id].cap == cap for t in tasks):
        print(
            report.wrapped(
                f"base answers: read from {name} (eval_sql.py without --adapter, the "
                f"same {cap:,}-token cap)"
            )
        )
        return saved
    print(
        report.wrapped(
            f"no saved base answers to these questions under a {cap:,}-token cap in "
            f"{name}: the base model answers them first (a fresh adapter, every B zero)"
        )
    )
    answers, seconds = answer_all(model, tokenizer, tasks, BASE_LABEL, cap)
    report.print_pass(BASE_LABEL, answers, tasks, seconds)
    return {a.task_id: a for a in answers}


def evaluate_adapter(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    tasks: list[Task],
    adapter: Path,
    args: argparse.Namespace,
) -> None:
    """Answer with the adapter, compare with the base model, show what else changed.

    Args:
        model: The loaded model, before LoRA.
        tokenizer: Its tokenizer.
        tasks: The questions.
        adapter: The adapter file.
        args: The command line.
    """
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)  # fresh: still the base
    base = base_answers(model, tokenizer, tasks, args.cap)
    before = []
    if not args.fast:  # the base model's replies to the unrelated prompts
        for text in UNRELATED_PROMPTS:
            prompt = plain_prompt(tokenizer, text)
            reply = greedy_reply(model, tokenizer, prompt, args.cap)
            before.append((prompt, reply, per_token(model, prompt, reply)))
    load_adapter(model, adapter)
    label = eval_label(adapter)
    answers, seconds = answer_all(model, tokenizer, tasks, label, args.cap)
    report.print_pass(label, answers, tasks, seconds)
    by_task = {task.task_id: task for task in tasks}
    mine = {a.task_id: a for a in answers}
    ordered = [base[task.task_id] for task in tasks]
    report.print_replies(
        "the same questions as the base run showed, after training",
        [mine[i] for i in shown_ids(ordered)],
        by_task,
    )
    comparison = compare(
        {t.task_id: base[t.task_id].correct for t in tasks},
        {t.task_id: mine[t.task_id].correct for t in tasks},
    )
    report.print_comparison(label, comparison)
    changed = [*comparison.gained[:SHOWN_CHANGED], *comparison.lost[:SHOWN_CHANGED]]
    if changed:
        report.print_replies(
            "some questions that changed, after training",
            [mine[i] for i in changed],
            by_task,
        )
    print(report.wrapped(f"saved: {short_path(eval_path(label))}"))
    if args.fast:
        report.heading("five prompts that are not about SQL: skipped with --fast")
        return
    results = []
    for text, (prompt, old, old_score) in zip(UNRELATED_PROMPTS, before, strict=True):
        new = greedy_reply(model, tokenizer, prompt, args.cap)
        results.append(
            report.Unrelated(
                prompt=text,
                base=tokenizer.decode(old.tokens),
                base_tokens=len(old.tokens),
                base_finish=old.finish_reason,
                tuned=tokenizer.decode(new.tokens),
                tuned_tokens=len(new.tokens),
                tuned_finish=new.finish_reason,
                base_score=old_score,
                tuned_score=per_token(model, prompt, old),
            )
        )
    report.print_unrelated(results)


def main() -> None:
    """Answer the held-out questions, grade them, and report.

    Raises:
        SystemExit: If the adapter file asked for does not exist.
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    tasks = load_tasks(TEST_FILE)
    if args.limit is not None:
        tasks = balanced(tasks, args.limit)
    adapter = adapter_path(args.adapter) if args.adapter else None
    if adapter is not None and not adapter.is_file():
        raise SystemExit(f"no adapter at {short_path(adapter)}: run train_sft.py first")
    loaded = load_folder(folder)
    report.print_title(SCRIPT, TITLE)
    report.print_eval_setup(folder, which, adapter, tasks, args.cap)
    if adapter is None:
        evaluate_base(loaded.model, loaded.tokenizer, tasks, args.cap)
    else:
        evaluate_adapter(loaded.model, loaded.tokenizer, tasks, adapter, args)


if __name__ == "__main__":
    main()
