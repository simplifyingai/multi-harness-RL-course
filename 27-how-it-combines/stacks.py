# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""The article's stack beside ours: the same pipeline on a cluster and a laptop.

Each row is one job in the pipeline. The left side is the article's own
sentences, quoted exactly (fetched on the date below); the right side is the
project file that does the same job here. Two of our facts are measured, not
typed: the project's recorded GRPO run, read from its metrics file (a copy is
bundled in ``data/``), and this machine, read from ``sysctl``.

``uv run stacks.py`` prints the panel. ``--verify`` fetches the article again
and checks that every quoted sentence is still in it, word for word. Only
Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = "stacks.py"
ARTICLE_URL = "https://fineenvs-multi-harness-rl.hf.space/"
SPACE_URL = "https://huggingface.co/spaces/FineEnvs/multi-harness-rl"
FETCHED = "2026-10-13"
GRPO_METRICS = HERE / "data" / "project_grpo_metrics.jsonl"
GRPO_METRICS_SOURCE = "runs/main/grpo/metrics.jsonl"
FETCH_TIMEOUT_S = 60
USER_AGENT = "mhrl-course-stacks/1"
SECONDS_PER_MINUTE = 60
BYTES_PER_GB = 1024**3
SYSCTL_TIMEOUT_S = 5
INDENT = "   "
WIDTH = 100  # wrap here, so the panel fits a recording's screen
EXIT_CHANGED = 1


@dataclass(frozen=True)
class Row:
    """One job in the pipeline, on both stacks.

    Attributes:
        job: What the job is.
        article: The article's sentences about it, exactly as published.
        ours: What does it here.
        files: The project files that do it.
    """

    job: str
    article: tuple[str, ...]
    ours: str
    files: tuple[str, ...]


ROWS = (
    Row(
        "tasks and sandboxes",
        (
            "Harbor supplies the tasks and the sandboxes.",
            "The reported LFM runs used Harbor for both policies and E2B sandboxes.",
            "The current tutorial uses Daytona and includes native OpenCode as a "
            "separate comparison, so it is not an exact replay of those historical "
            "runs.",
        ),
        "SQLite questions, one fresh folder per rollout; macOS sandbox-exec",
        ("mhrl/tasks/core.py", "configs/harnesses/sandbox.sb"),
    ),
    Row(
        "inference engine",
        (
            "One H100 trains while a second serves the policy with vLLM (Kwon et "
            "al., 2023) and keeps producing rollouts, so training never waits for "
            "the slowest agent.",
        ),
        "an MLX engine on the same laptop, the 4-bit model plus the adapter",
        ("mhrl/engine/server.py", "mhrl/engine/policy.py"),
    ),
    Row(
        "capture",
        (
            "OpenEnv is the interface the harness, the environment and the trainer "
            "all connect to, and its capture proxy records the tokens.",
        ),
        "OpenEnv's capture proxy, unchanged",
        ("scripts/serve.sh", "pyproject.toml"),
    ),
    Row(
        "trainer",
        (
            "Async GRPO in TRL for 1,000 steps.",
            "The price is that rollouts come from weights usually two optimizer "
            "steps old, and never more than four.",
            "TRL trains on what the proxy recorded.",
        ),
        "GRPO in MLX, in turn: sample with the last adapter, then update",
        ("mhrl/training/grpo.py",),
    ),
    Row(
        "hardware and time",
        (
            "Two H100s per run on the Hugging Face cluster, and one E2B sandbox per "
            "rollout with 1 CPU and 4 GB of memory.",
            "Training took about 32 hours for the OpenCode-only run and 46 hours "
            "for the multi-harness run, not counting evaluation, queueing and "
            "restarts.",
        ),
        "one laptop (below)",
        (),
    ),
)


def article_text(page: str) -> str:
    """Turn the article's HTML into the text a reader sees, on one line.

    Tags become spaces, then the spaces a tag leaves before punctuation and
    inside brackets are closed up: ``vLLM ( Kwon et al., 2023 ) and`` reads
    ``vLLM (Kwon et al., 2023) and`` on the page.

    Args:
        page: The HTML.

    Returns:
        The text.
    """
    page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)!?])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


def missing_quotes(text: str) -> list[str]:
    """Return the quoted sentences the article's text no longer holds.

    Args:
        text: The article, as ``article_text`` gives it.

    Returns:
        Every quote not found word for word.
    """
    return [quote for row in ROWS for quote in row.article if quote not in text]


def fetch_article(url: str = ARTICLE_URL) -> str:
    """Download the article's page.

    Args:
        url: Its address.

    Returns:
        The HTML.

    Raises:
        urllib.error.URLError: If it cannot be fetched.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        return response.read().decode("utf-8", errors="replace")


def recorded_grpo(path: Path = GRPO_METRICS) -> str:
    """Summarise the project's recorded GRPO run from its metrics.

    Args:
        path: The metrics file (one JSON line per step).

    Returns:
        Its steps, rollouts, summed step time and peak memory, in one line.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    steps = [row for row in rows if "step_seconds" in row]
    minutes = sum(row["step_seconds"] for row in steps) / SECONDS_PER_MINUTE
    rollout_groups = statistics.fmean(row["groups"] for row in steps)
    peak = max(row["peak_gb"] for row in steps)
    return (
        f"{len(steps)} steps of {rollout_groups:.0f} groups of 4, {minutes:.1f} "
        f"minutes of steps, trainer peak {peak:.2f} GB"
    )


def sysctl(name: str) -> str:
    """Read one value from macOS's ``sysctl``.

    Args:
        name: The key, e.g. ``hw.memsize``.

    Returns:
        Its value, or "" when it cannot be read (not a Mac).
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=SYSCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def this_machine() -> str:
    """Describe this machine: its chip and its memory.

    Returns:
        For example ``Apple M3 Pro, 18 GB``.
    """
    chip = sysctl("machdep.cpu.brand_string") or "unknown chip"
    memory = sysctl("hw.memsize")
    if not memory.isdigit():
        return chip
    return f"{chip}, {int(memory) / BYTES_PER_GB:.0f} GB"


def panel(machine: str, grpo: str) -> list[str]:
    """Print the side panel.

    Args:
        machine: This machine, as ``this_machine`` describes it.
        grpo: The recorded run, as ``recorded_grpo`` summarises it.

    Returns:
        The lines.
    """
    lines = [
        f"{SCRIPT} · the article's stack beside ours",
        f"article   {SPACE_URL}",
        f"          its text fetched {FETCHED} from {ARTICLE_URL}",
        "",
    ]
    for row in ROWS:
        lines.append(row.job)
        for quote in row.article:
            lines += textwrap.wrap(
                f'"{quote}"',
                WIDTH,
                initial_indent=f"{INDENT}article  ",
                subsequent_indent=f"{INDENT}         ",
                break_on_hyphens=False,
            )
        where = f" ({', '.join(row.files)})" if row.files else ""
        lines += textwrap.wrap(
            f"{row.ours}{where}",
            WIDTH,
            initial_indent=f"{INDENT}ours     ",
            subsequent_indent=f"{INDENT}         ",
            break_on_hyphens=False,
        )
    lines += [
        "",
        f"this machine   {machine}",
        f"our GRPO run   {grpo} ({GRPO_METRICS_SOURCE})",
    ]
    return lines


def main(argv: list[str] | None = None, say: Callable[[str], None] = print) -> int:
    """Print the panel, or check the quotes against the article.

    Args:
        argv: The arguments, or None for ``sys.argv``.
        say: Where each line goes.

    Returns:
        0, or 1 when ``--verify`` finds a quote the article no longer holds.

    Raises:
        SystemExit: If ``--verify`` cannot fetch the article.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify", action="store_true", help="fetch the article and check each quote"
    )
    args = parser.parse_args(argv)
    if not args.verify:
        for line in panel(this_machine(), recorded_grpo()):
            say(line)
        return 0
    try:
        text = article_text(fetch_article())
    except (urllib.error.URLError, TimeoutError) as error:
        raise SystemExit(f"{SCRIPT}: cannot fetch {ARTICLE_URL}: {error}") from error
    quotes = [quote for row in ROWS for quote in row.article]
    missing = missing_quotes(text)
    say(f"{SCRIPT} --verify · {ARTICLE_URL}, fetched now")
    for quote in quotes:
        mark = "missing" if quote in missing else "found  "
        say(f'{INDENT}{mark}  "{quote}"')
    say(f"{len(quotes) - len(missing)} of {len(quotes)} quotes found word for word")
    return EXIT_CHANGED if missing else 0


if __name__ == "__main__":
    sys.exit(main())
