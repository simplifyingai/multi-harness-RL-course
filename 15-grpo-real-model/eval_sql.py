# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Compare base, SFT and SFT+GRPO on the 50 held-out questions, question by question.

One greedy reply per held-out question (clinic and airline, databases no
training question comes from), to chapter 14's chat, up to the project's
1,024-token cap, from three models: the base model (a fresh adapter, every B
zero), chapter 14's SFT adapter, and the adapter grpo_sql.py trained from it.
Each reply's SQL runs on a read-only copy of the database, its result is
written to ``answer.txt`` in the question's own folder, and chapter 1's
grader judges that file. It prints every verdict, each model's pass@1 with
its error bar, gained and lost with the exact p-value for each pair, and how
long the replies are.

Run it with ``uv run eval_sql.py`` after ``uv run grpo_sql.py`` (tens of
minutes the first time; answers already saved for the same adapter bytes and
cap are read back). ``--adapter`` names another adapter grpo_sql.py saved,
``--init`` another SFT adapter of chapter 14's, ``--fast`` uses Liquid's
export and 10 questions. Results go under ``$COURSE_HOME/15-grpo-real-model``.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import mlx.nn as nn
import report
from grading import FINISH_LENGTH, Answered, grade_reply, read_answers, write_answers
from lfm import greedy_reply, load_folder
from lfm_files import (
    BASE_LABEL,
    LORA_LAYERS,
    LORA_RANK,
    LORA_SCALE,
    OUTPUT_CAP,
    SFT_ADAPTER,
    SFT_LABEL,
    adapter_path,
    eval_path,
    init_path,
    pick_model,
    require_init,
    run_name,
    short_path,
    work_folder,
)
from lora import apply_lora, load_adapter, zero_adapter
from mlx_lm.tokenizer_utils import TokenizerWrapper
from report import ModelAnswers
from rows import balanced, prompt_ids
from stats import compare
from tasks import TEST_FILE, Task, load_tasks

SCRIPT = "eval_sql.py"
TITLE = "base, SFT and SFT+GRPO on the held-out questions"
FAST_QUESTIONS = 10
AFTER_LABEL = "sft+grpo"
NO_ADAPTER = "none"  # the base model's fingerprint
FINGERPRINT_CHARS = 16  # of the adapter file's SHA-256, enough to tell files apart
READ_CHUNK = 1 << 20  # bytes read at a time while fingerprinting
SHOWN_CHANGED = 2  # gained and lost questions shown, each
EVAL_WORK = "eval"  # work/eval/<label>/<task>/answer.txt


def fingerprint(path: Path | None) -> str:
    """Name an adapter file by its bytes, so saved answers match only it.

    Args:
        path: The adapter file, or None for the base model.

    Returns:
        The start of its SHA-256 in hex, or ``"none"``.
    """
    if path is None:
        return NO_ADAPTER
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()[:FINGERPRINT_CHARS]


def answer_all(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    tasks: Sequence[Task],
    label: str,
    adapter: str,
) -> list[Answered]:
    """Answer every question greedily and grade each reply.

    Args:
        model: The network, its adapter already set.
        tokenizer: Its tokenizer.
        tasks: The questions.
        label: The evaluation's file label (``base``, ``sft``, or the adapter's).
        adapter: The adapter's fingerprint.

    Returns:
        One graded answer per question.
    """
    answers = []
    for task in tasks:
        reply = greedy_reply(model, tokenizer, prompt_ids(tokenizer, task), OUTPUT_CAP)
        text = tokenizer.decode(reply.tokens)
        cut_off = reply.finish_reason == FINISH_LENGTH
        where = work_folder(EVAL_WORK, label, task.task_id)
        verdict = grade_reply(task, text, where, cut_off)
        answers.append(
            Answered(
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
                cap=OUTPUT_CAP,
                adapter=adapter,
            )
        )
    return answers


def gather(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    tasks: Sequence[Task],
    label: str,
    path: Path | None,
) -> tuple[dict[str, Answered], str]:
    """Read one model's saved answers, or set its adapter and answer now.

    Args:
        model: The network with LoRA layers.
        tokenizer: Its tokenizer.
        tasks: The questions.
        label: The evaluation's file label.
        path: The adapter file, or None for the base model.

    Returns:
        Each question's answer by task id, and a note saying where they came
        from.
    """
    mark = fingerprint(path)
    saved = read_answers(eval_path(label))
    if all(
        task.task_id in saved
        and saved[task.task_id].cap == OUTPUT_CAP
        and saved[task.task_id].adapter == mark
        for task in tasks
    ):
        name = eval_path(label).name
        return saved, f"read from {name} (the same adapter bytes and cap)"
    if path is None:
        zero_adapter(model)  # every B zero: the base model exactly
    else:
        load_adapter(model, path)
    start = time.perf_counter()
    answers = answer_all(model, tokenizer, tasks, label, mark)
    seconds = time.perf_counter() - start
    write_answers(eval_path(label), answers)
    note = (
        f"answered now, saved to {eval_path(label).name}; varies from run to run: "
        f"the {len(tasks)} replies took {seconds:.0f} s"
    )
    return {answered.task_id: answered for answered in answers}, note


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter", default=run_name([]), help="an adapter grpo_sql.py saved"
    )
    parser.add_argument(
        "--init", default=SFT_ADAPTER, help="the SFT adapter it started from"
    )
    parser.add_argument("--limit", type=int, default=None, help="ask N questions")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Liquid's export, {FAST_QUESTIONS} questions",
    )
    args = parser.parse_args()
    if args.fast and args.limit is None:
        args.limit = FAST_QUESTIONS
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    """Answer the held-out questions with all three models, and compare them.

    Raises:
        SystemExit: If an adapter file asked for does not exist, or its name
            would share the base model's or the SFT adapter's saved answers.
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    init = require_init(init_path(args.init))
    after = adapter_path(args.adapter)
    if not after.is_file():
        raise SystemExit(
            f"no adapter at {short_path(after)}: run grpo_sql.py first, or name the "
            "one it saved with --adapter (--fast saves grpo_steps4.safetensors)"
        )
    sft_label = SFT_LABEL if args.init == SFT_ADAPTER else init.stem
    if sft_label == BASE_LABEL or after.stem in (BASE_LABEL, SFT_LABEL, sft_label):
        raise SystemExit(
            f"--adapter {args.adapter} with --init {args.init}: two models would "
            "share one file of saved answers; rename one of the adapters"
        )
    tasks = load_tasks(TEST_FILE)
    if args.limit is not None:
        tasks = balanced(tasks, args.limit)
    loaded = load_folder(folder)
    model, tokenizer = loaded.model, loaded.tokenizer
    apply_lora(model, LORA_RANK, LORA_SCALE, LORA_LAYERS)
    plan = (
        (BASE_LABEL, BASE_LABEL, None, "a fresh adapter: every B zero, the base model"),
        (SFT_LABEL, sft_label, init, f"{short_path(init)}, chapter 14's"),
        (AFTER_LABEL, after.stem, after, f"{short_path(after)}, grpo_sql.py's"),
    )
    models = []
    for label, file_label, path, file in plan:
        answers, note = gather(model, tokenizer, tasks, file_label, path)
        models.append(ModelAnswers(label, file, answers, note))
    report.print_title(SCRIPT, TITLE)
    report.print_eval_setup(folder, which, models, tasks)
    report.print_verdicts(models, tasks)
    report.print_scores(models, tasks)
    right = {
        model.label: {
            task.task_id: model.answers[task.task_id].correct for task in tasks
        }
        for model in models
    }
    base, sft, grpo = models
    pairs = [
        (before.label, later.label, compare(right[before.label], right[later.label]))
        for before, later in ((sft, grpo), (base, grpo), (base, sft))
    ]
    report.print_comparisons(pairs)
    report.print_lengths(models, tasks)
    changed = pairs[0][2]
    shown = [*changed.gained[:SHOWN_CHANGED], *changed.lost[:SHOWN_CHANGED]]
    report.print_changed(sft, grpo, shown, {task.task_id: task for task in tasks})


if __name__ == "__main__":
    main()
