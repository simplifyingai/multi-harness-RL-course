"""Chapter 27 end to end: one_step.py through the project's real pipeline.

No 2.6B is loaded. ``stack.sh scripted`` starts the project's own capture
proxy in front of the project's own answer-key teacher (``mhrl.engine.server
--oracle``, which loads only the tokenizer), and ``one_step_scripted_27.py``
runs ``one_step.py`` with a tiny LFM2 in place of the 2.6B trainer. Every
rollout is the real, pinned Claude Code in the project's sandbox, captured
by the real proxy, graded by the project's grader, and the step is the
project's ``train_grpo``.

The teacher writes the same replies in all four rollouts, so a group has no
contrast: the first test checks that one_step says so and draws the next
task. The second marks every second rollout wrong, so the update path runs:
the rows, the loss, the AdamW step, the adapter file and the POST. Both
check that no session id (the harness's API key) is ever printed whole and
that the project's checkout is left exactly as it was.

The stack uses ``MHRL_ENGINE_PORT`` and ``MHRL_PROXY_PORT`` when they are
set, else free ports. Skipped without a checkout of the project
(``MHRL_PROJECT``, default ``~/sai_work/multi-harness-rl``) that has its
environment, its pinned Claude Code and the model's tokenizer.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from chapters import ROOT

CHAPTER = ROOT / "27-how-it-combines"
DRIVER = ROOT / "tests" / "one_step_scripted_27.py"
PROJECT = Path(
    os.environ.get("MHRL_PROJECT") or Path.home() / "sai_work" / "multi-harness-rl"
)
PYTHON = PROJECT / ".venv" / "bin" / "python"
CLAUDE = PROJECT / "tools" / "claude-code" / "2.1.290" / "claude"
TOKENIZER = PROJECT / "models" / "LFM2.5-2.6B-MLX-4bit" / "tokenizer.json"
NEEDED = (PYTHON, CLAUDE, TOKENIZER)
SESSION_ID = re.compile(r"\bs[0-9a-f]{24}\b")
RUN_TIMEOUT_S = 600
START_TIMEOUT_S = 300

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in NEEDED),
    reason="needs the project's checkout with .venv, pinned Claude Code, tokenizer",
)


def free_port() -> int:
    """Ask the system for a free port (never the project's 8100/8200).

    Returns:
        A port nothing listens on.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def git_status() -> str:
    """Return the project checkout's ``git status``, to prove it unchanged.

    Returns:
        The porcelain status.
    """
    return subprocess.run(
        ["git", "-C", str(PROJECT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Start the scripted stack once for this file, and stop it afterwards.

    Args:
        tmp_path_factory: pytest's folder maker.

    Yields:
        The environment that points the project at it.
    """
    env = {
        **os.environ,
        "MHRL_PROJECT": str(PROJECT),
        # The ports in the environment when set (the run log's 8129 and
        # 8209), else free ones the system hands out.
        "MHRL_ENGINE_PORT": os.environ.get("MHRL_ENGINE_PORT") or str(free_port()),
        "MHRL_PROXY_PORT": os.environ.get("MHRL_PROXY_PORT") or str(free_port()),
        "COURSE_HOME": str(tmp_path_factory.mktemp("stack-home")),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = str(CHAPTER / "stack.sh")
    started = subprocess.run(
        [script, "scripted"],
        env=env,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
        check=False,
    )
    try:
        assert started.returncode == 0, started.stdout + started.stderr
        assert "capture level: tokens" in started.stdout
        yield env
    finally:
        subprocess.run([script, "stop"], env=env, timeout=120, check=False)


def run_step(env: dict[str, str], home: Path, *args: str) -> tuple[int, str]:
    """Run one_step.py through the driver, with the project's Python.

    Args:
        env: The stack's environment.
        home: The test's COURSE_HOME.
        *args: The driver's arguments (``--contrast``), then ``--`` and
            one_step's.

    Returns:
        The exit code and everything printed.
    """
    result = subprocess.run(
        [str(PYTHON), str(DRIVER), *args],
        env={**env, "COURSE_HOME": str(home)},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def test_a_group_without_contrast_draws_the_next_task(
    stack: dict[str, str], course_home: Path
) -> None:
    before = git_status()
    code, output = run_step(stack, course_home, "--", "--max-draws", "2")
    assert code == 2, output
    assert "capture level: tokens" in output
    assert "draw 1 · " in output and "draw 2 · " in output
    assert output.count("no contrast: all 4 rewards are equal") == 2
    assert "no update: drawing the next task in train_grpo's order" in output
    assert "load_policy  the model loaded for the last draw, reused" in output
    assert re.search(r"\[r1\] folder \S+ holds data\.db", output)
    assert "[r1] API key = the session id s" in output
    assert "ANTHROPIC_API_KEY=<redacted>" in output
    assert "state $TMPDIR/mhrl-harness-claude-code-" in output
    assert "call 1 · prompt " in output
    assert not SESSION_ID.search(output), "a whole session id was printed"
    assert git_status() == before


def test_one_step_with_contrast_updates_saves_and_swaps_the_adapter(
    stack: dict[str, str], course_home: Path
) -> None:
    before = git_status()
    code, output = run_step(stack, course_home, "--contrast", "--", "--max-draws", "1")
    assert code == 0, output
    advantages = re.search(r"advantages \(reward - mean\) / spread: (.*)", output)
    assert advantages and sorted(advantages.group(1).split()) == [
        "+1.00",
        "+1.00",
        "-1.00",
        "-1.00",
    ]
    assert "position  mask  old logprob  token" in output
    assert re.search(r"rows   4 trained, [\d,]+ tokens", output)
    assert "AdamW  one update of " in output and "bias correction on" in output
    folders = list((course_home / "27-how-it-combines" / "one_step").glob("draw1-*"))
    assert len(folders) == 1
    adapter = folders[0] / "adapter_step001.safetensors"
    assert adapter.is_file()
    posts = [line for line in output.splitlines() if "/admin/adapter" in line]
    assert posts[0].endswith('{"path": null}')  # train_grpo starts on the base
    assert posts[-1].endswith('adapter_step001.safetensors"}')
    records = (folders[0] / "rollouts.jsonl").read_text().splitlines()
    assert len(records) == 4
    assert {json.loads(line)["harness"] for line in records} == {"claude-code"}
    metrics = json.loads((folders[0] / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["train_rows"] == 4 and metrics["groups_without_contrast"] == 0
    assert not SESSION_ID.search(output), "a whole session id was printed"
    assert git_status() == before
