# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay two recorded attempts at the same task by the same small model.

The project ran LFM2.5-2.6B (as released, no training) inside two agent
programs on one clinic question. This prints what the model asked for, what
each program did with it, and how the grader scored the result. Then it
prints the project's scores for that model in three programs.

Run it with ``uv run replay_agent.py`` (``--fast`` skips the pauses).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from grader import REASON_NO_ANSWER, Grade, check_answer
from tasks import ANSWER_FILE, DATA_DIR, find_task

TASK_ID = "clinic-medium-001"
TRANSCRIPTS = DATA_DIR / "transcripts" / f"{TASK_ID}.json"
RESULTS = DATA_DIR / "results_as_released.json"
RUN_KEYS = ("mini-swe-agent", "claude-code")
SCORES = "scores"
PAUSE_S = 0.8
MAX_OUTPUT_LINES = 3
NUMBER_WIDTH = 4
KIND_WIDTH = 10
INDENT = " " * (NUMBER_WIDTH + KIND_WIDTH)
NAME_WIDTH = 18
SOLVED_WIDTH = 10
SCORE_WIDTH = 8
BOX_WIDTH = 50
PERCENT = 100
REPLY_KINDS = {"command", "rejected"}
# Plain words for the program's own messages (the raw text is in data/).
GLOSSES = {
    "Output token limit hit": (
        "[a reply hit the 1,024-token cap; Claude Code asked it to go on]"
    ),
    "Step budget exhausted": "[the 8-call budget is used up, so the run ends]",
}


def indent_block(text: str) -> str:
    """Indent output lines under a step, keeping at most a few.

    Args:
        text: The output, possibly many lines.

    Returns:
        The lines to print, with a note when some are left out.
    """
    lines = text.splitlines()
    shown = lines[:MAX_OUTPUT_LINES]
    if len(lines) > MAX_OUTPUT_LINES:
        shown.append(f"... {len(lines) - MAX_OUTPUT_LINES} more lines")
    return "\n".join(INDENT + line for line in shown)


def describe(event: dict[str, Any]) -> tuple[str, str]:
    """Turn one recorded event into a label and a line of text.

    Args:
        event: A ``command``, ``rejected`` or ``harness`` event.

    Returns:
        The label (``runs``, ``rejected``, ``program``) and the text.

    Raises:
        KeyError: If the event is missing a field its kind needs.
    """
    if event["kind"] == "command":
        return "runs", event["command"]
    if event["kind"] == "rejected":
        tool = event.get("tool")
        if tool is None:
            return "rejected", event["message"]
        asked = event["input"].get("command") or event["input"].get("file", "")
        return "rejected", f"tool {tool!r}: {asked}\n{INDENT}{event['message']}"
    for start, gloss in GLOSSES.items():
        if event["message"].startswith(start):
            return "program", gloss
    return "program", event["message"]


def replay_and_grade(run: dict[str, Any], question: str, pause_s: float) -> Grade:
    """Print one recorded run, then grade the answer file it left.

    Args:
        run: One run from the transcript file.
        question: The task's question.
        pause_s: Seconds to wait between steps, so a viewer can follow.

    Returns:
        The grade the project's grader gives that run.
    """
    print(f"\n=== {run['harness']} {run['version']} ===")
    print(f"question: {question}\n")
    number = 0
    for event in run["events"]:
        label, text = describe(event)
        if event["kind"] in REPLY_KINDS:
            number += 1
            marker = str(number)
        else:
            marker = "."
        print(f"{marker:>{NUMBER_WIDTH - 1}} {label:<{KIND_WIDTH}}{text}")
        if event["kind"] == "command" and event.get("output"):
            print(indent_block(event["output"]))
        time.sleep(pause_s)
    written = run["answer_file"]
    if written:
        result = check_answer(find_task(TASK_ID), written)
    else:
        result = Grade(correct=False, submitted=None, reason=REASON_NO_ANSWER)
    verdict = "correct" if result.correct else f"wrong ({result.reason})"
    print(f"\n{ANSWER_FILE}: {written or '(never written)'}")
    print(f"grade: {verdict}, reward {result.reward}")
    return result


def print_scores() -> None:
    """Print the project's scores for the as-released model, and the article's."""
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    print(f"\n=== {results['title']} ===")
    print(
        f"{'program':<{NAME_WIDTH}}{'solved':>{SOLVED_WIDTH}}{'score':>{SCORE_WIDTH}}"
    )
    for row in results["harnesses"]:
        solved = f"{row['solved']} / {row['tasks']}"
        score = f"{round(PERCENT * row['solved'] / row['tasks'])}%"
        print(
            f"{row['name']:<{NAME_WIDTH}}{solved:>{SOLVED_WIDTH}}{score:>{SCORE_WIDTH}}"
        )
    print(results["note"])
    article = results["article"]
    title = f"+-- {article['label']} "
    print("\n" + title + "-" * max(0, BOX_WIDTH - len(title)))
    for line in article["lines"]:
        print(f"| {line}")
    print("+" + "-" * (BOX_WIDTH - 1))


def main() -> None:
    """Replay both runs, then print the scores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="no pauses")
    parser.add_argument("--only", choices=(*RUN_KEYS, SCORES))
    args = parser.parse_args()
    pause_s = 0.0 if args.fast else PAUSE_S
    recorded = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    if args.only != SCORES:
        print(f"model: {recorded['model']}")
        print(f"recorded: {recorded['recorded']}")
    for run in recorded["runs"]:
        if args.only in (None, run["key"]):
            replay_and_grade(run, recorded["question"], pause_s)
    if args.only in (None, SCORES):
        print_scores()


if __name__ == "__main__":
    main()
