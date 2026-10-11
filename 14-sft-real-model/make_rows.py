# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Turn the 230 bundled questions into training rows, each with a loss mask.

Every question becomes a chat (a system rule, then the database's tables and
the question) rendered with LFM2.5's chat template, followed by the reply
the model should write: a one-line reason, ``</think>``, the gold SQL and
the end-of-turn token. The row records which positions are the reply: the
**loss mask**, 1 for the reply and 0 for everything the model only reads.
Only the tokenizer is loaded, not the model's weights.

Run it with ``uv run make_rows.py --show 1`` (a few seconds): it writes
``rows_train.jsonl`` (180 rows) and ``rows_test.jsonl`` (50) under
``$COURSE_HOME/14-sft-real-model`` and prints row 1, the read part and the
reply apart, dimmed and highlighted on a terminal. ``--fast`` takes the
tokenizer from Liquid's export (the same file as ours); ``--reason fixed``
starts every reply with the project's teacher's one sentence instead of a
reason written from the query (``rows_train_fixed.jsonl``).
"""

from __future__ import annotations

import argparse
import os
import sys

import report
from lfm import load_tokenizer_only
from lfm_files import pick_model, rows_path, short_path
from rows import (
    REASON_STEPS,
    REASON_STYLES,
    Row,
    build_row,
    item_from_sequence,
    reply_text,
    template_difference,
    write_rows,
)
from tasks import TEST_FILE, TRAIN_FILE, load_tasks

SCRIPT = "make_rows.py"
TITLE = "questions into training rows, with a loss mask"
SPLITS = (("train", TRAIN_FILE), ("test", TEST_FILE))
NO_COLOR_ENV = "NO_COLOR"  # the common convention for turning colours off


def use_color() -> bool:
    """Colour the row only on a terminal, and never when NO_COLOR is set.

    Returns:
        True when the output is a terminal and colours are allowed.
    """
    return sys.stdout.isatty() and not os.environ.get(NO_COLOR_ENV)


def summarize(split: str, rows: list[Row], style: str) -> report.SplitSummary:
    """Count what one split's rows hold, and write them.

    Args:
        split: ``"train"`` or ``"test"``.
        rows: Its rows.
        style: The kind of reason the replies start with.

    Returns:
        The counts, and where the rows went.
    """
    path = rows_path(split, style)
    write_rows(path, rows)
    return report.SplitSummary(
        split=split,
        rows=len(rows),
        tokens=sum(len(row.input_ids) for row in rows),
        reply=sum(row.reply_tokens for row in rows),
        longest=max((len(row.input_ids) for row in rows), default=0),
        path=path,
    )


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments, checked.

    Raises:
        SystemExit: From argparse, with the reason, if a flag is impossible.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--show", type=int, default=None, help="print this training row (from 1)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="the tokenizer of Liquid's export"
    )
    parser.add_argument(
        "--reason",
        choices=REASON_STYLES,
        default=REASON_STEPS,
        help="each reply's reason: steps from the gold SQL, or one fixed sentence",
    )
    return parser.parse_args()


def main() -> None:
    """Build every row, write them, and show one if asked.

    Raises:
        SystemExit: If ``--show`` names a row that does not exist.
    """
    args = parse_args()
    folder, which = pick_model(args.fast)
    tokenizer = load_tokenizer_only(folder)
    tasks = {split: load_tasks(name) for split, name in SPLITS}
    train = tasks["train"]
    if args.show is not None and not 1 <= args.show <= len(train):
        raise SystemExit(f"--show must be between 1 and {len(train)}")
    report.print_title(SCRIPT, TITLE)
    report.setup_line("tokenizer", f"{short_path(folder)}, {which}")
    report.print_questions(train, tasks["test"])
    report.print_shape(train[0], reply_text(train[0], args.reason), args.reason)
    rows = {
        split: [build_row(tokenizer, task, style=args.reason) for task in split_tasks]
        for split, split_tasks in tasks.items()
    }
    summaries = [summarize(split, rows[split], args.reason) for split, _ in SPLITS]
    report.print_rows_summary(summaries)
    if args.show is None:
        return
    row = rows["train"][args.show - 1]
    report.print_row(row, args.show, tokenizer.decode, use_color())
    item = item_from_sequence(row.as_sequence())
    # The same row with a 1 at position 0 as well: it still cannot be a target.
    first_too = {**row.as_sequence(), "loss_mask": [1, *row.loss_mask[1:]]}
    forced = item_from_sequence(first_too)
    report.print_boundary(row, tokenizer.decode, item, forced)
    rendered, shared = template_difference(
        tokenizer, train[args.show - 1], row, style=args.reason
    )
    report.print_template_check(row, rendered, shared, tokenizer.decode)


if __name__ == "__main__":
    main()
