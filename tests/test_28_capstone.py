"""Chapter 28: capstone.sh's plan, its guards, and the pipeline end to end.

``capstone.sh plan`` prints every command ``capstone.sh`` runs, and runs
none; its output is pinned for FULL and FAST. The plan's pipeline is checked
against the project's own ``scripts/train_from_scratch.sh``: the same
subcommands in the same order, with the same flags, plus only
``--teacher-port`` on ``demos``.

The end-to-end test loads no 2.6B. ``STACK=scripted`` starts the project's
answer-key teacher on the engine port and the project's capture proxy in
front of it, and ``MHRL_CLI`` is the project's CLI with a tiny LFM2 in place
of the trainers' model (``capstone_cli_scripted_28.py``). Every rollout is a
real harness in the project's sandbox. It writes only under ``tmp_path``
(``MHRL_RUNS_DIR``) and leaves the project's checkout exactly as it was.
Skipped without a checkout of the project (``MHRL_PROJECT``, default
``~/sai_work/multi-harness-rl``) that has its environment, its pinned Claude
Code and the model's tokenizer.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
from pathlib import Path

import pytest
from chapters import ROOT

CHAPTER = ROOT / "28-capstone"
SCRIPT = CHAPTER / "capstone.sh"
DRIVER = ROOT / "tests" / "capstone_cli_scripted_28.py"
DATA = ROOT / "tests" / "data"
PROJECT = Path(
    os.environ.get("MHRL_PROJECT") or Path.home() / "sai_work" / "multi-harness-rl"
)
PYTHON = PROJECT / ".venv" / "bin" / "python"
CLAUDE = PROJECT / "tools" / "claude-code" / "2.1.290" / "claude"
MODEL = PROJECT / "models" / "LFM2.5-2.6B-MLX-4bit"
TRAIN_SCRIPT = PROJECT / "scripts" / "train_from_scratch.sh"
SESSION_ID = re.compile(r"\bs[0-9a-f]{24}\b")
CLI_MARK = "-m mhrl.cli "
BANNERS = (
    "1/5 baseline: the base model in every harness",
    "2/5 answer-key demonstrations, then supervised fine-tuning",
    "3/5 evaluate the SFT model",
    "4/5 GRPO through the training harnesses",
    "5/5 evaluate the RL model",
    "results",
    "done",
)
RUN_TIMEOUT_S = 1200


def clean_env(course_home: Path, **extra: str) -> dict[str, str]:
    """An environment with none of the caller's MHRL_* or pipeline knobs.

    Args:
        course_home: The test's COURSE_HOME.
        **extra: Variables to add.

    Returns:
        The environment.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(Path.home()),
        "COURSE_HOME": str(course_home),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if "TMPDIR" in os.environ:
        env["TMPDIR"] = os.environ["TMPDIR"]
    env.update(extra)
    return env


def capstone(env: dict[str, str], *args: str, timeout: int = 60) -> tuple[int, str]:
    """Run capstone.sh.

    Args:
        env: Its environment.
        *args: Its arguments.
        timeout: Seconds before the test gives up.

    Returns:
        The exit code and everything printed.
    """
    result = subprocess.run(
        [str(SCRIPT), *args],
        env=env,
        cwd=CHAPTER,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("fast", "pinned"), [("", "28_plan_full.txt"), ("1", "28_plan_fast.txt")]
)
def test_the_plan_is_pinned(course_home: Path, fast: str, pinned: str) -> None:
    extra = {"FAST": fast} if fast else {}
    code, output = capstone(clean_env(course_home, **extra), "plan")
    assert code == 0, output
    assert output == (DATA / pinned).read_text(encoding="utf-8")
    assert not (course_home / "28-capstone").exists(), "plan must create nothing"


def test_the_full_plan_sets_six_grpo_steps_and_our_ports(course_home: Path) -> None:
    _, output = capstone(clean_env(course_home), "plan")
    assert "--steps 6 --tasks-per-step 2 --group 4" in output
    assert "--teacher-port 8159" in output
    assert "engine 8149, teacher 8159, proxy 8219" in output
    assert "serve.sh" not in output.replace("scripts/serve.sh", "")


def plan_commands(output: str) -> list[list[str]]:
    """The mhrl.cli commands in a plan, as argument lists after the module.

    Args:
        output: ``capstone.sh plan``'s output.

    Returns:
        One list per command.
    """
    return [
        shlex.split(line.split(CLI_MARK, 1)[1])
        for line in output.splitlines()
        if CLI_MARK in line
    ]


def script_commands(text: str) -> list[list[str]]:
    """The ``$CLI`` commands of train_from_scratch.sh, continuations joined.

    Args:
        text: The script.

    Returns:
        One list per command, without ``$CLI``.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    return [
        shlex.split(line.strip().removeprefix("$CLI "))
        for line in joined.splitlines()
        if line.strip().startswith("$CLI ")
    ]


def flags(argv: list[str]) -> list[str]:
    """The subcommand and the names of its flags.

    Args:
        argv: A CLI command's arguments.

    Returns:
        ``[subcommand, --flag, ...]``.
    """
    return [argv[0], *(arg for arg in argv[1:] if arg.startswith("--"))]


@pytest.mark.skipif(not TRAIN_SCRIPT.is_file(), reason="needs the project's checkout")
def test_the_pipeline_is_train_from_scratch_command_for_command(
    course_home: Path,
) -> None:
    _, output = capstone(clean_env(course_home), "plan")
    ours = [flags(argv) for argv in plan_commands(output) if argv[0] != "steps"]
    theirs = []
    for argv in script_commands(TRAIN_SCRIPT.read_text()):
        argv = [a for a in argv if not a.startswith('"${') and a != "]"]
        listed = flags(argv)
        if listed[0] == "eval":  # "${EVAL_ARGS[@]}" holds --run --harnesses --limit
            listed[1:1] = ["--run", "--harnesses", "--limit"]
        if listed[0] == "report":  # "${REPORT_ARGS[@]}" holds --run --harnesses
            listed[1:] = ["--run", "--harnesses"]
        if listed[0] == "demos":  # the one flag the capstone adds
            listed.append("--teacher-port")
        theirs.append(listed)
    assert ours == theirs


def test_steps_that_write_refuse_another_checkout(
    course_home: Path, tmp_path: Path
) -> None:
    other = tmp_path / "someone-elses-checkout"
    other.mkdir()
    env = clean_env(course_home, MHRL_PROJECT=str(other))
    for step in ("setup", "tasks", "tests"):
        code, output = capstone(env, step)
        assert code == 1
        assert f"capstone: {step} writes into the project" in output
    code, output = capstone(env, "pipeline")
    assert code == 1 and "needs MHRL_RUNS_DIR outside it" in output
    inside = clean_env(
        course_home, MHRL_PROJECT=str(other), MHRL_RUNS_DIR=str(other / "runs")
    )
    code, output = capstone(inside, "pipeline")
    assert code == 1 and "MHRL_RUNS_DIR must be outside" in output
    for runs in ("runs", f"{other}/../{other.name}/runs"):
        code, output = capstone(
            clean_env(course_home, MHRL_PROJECT=str(other), MHRL_RUNS_DIR=runs),
            "pipeline",
        )
        assert code == 1 and ("absolute" in output or "must not hold" in output)
    assert list(other.iterdir()) == []


def test_an_unknown_step_prints_the_usage(course_home: Path) -> None:
    code, output = capstone(clean_env(course_home), "train")
    assert code == 2 and output.startswith("usage: ")


def test_the_network_filter_keeps_only_connections_off_the_machine() -> None:
    lsof = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "python 1 me 3u IPv4 0x1 0t0 TCP 127.0.0.1:8149 (LISTEN)\n"
        "python 1 me 4u IPv4 0x2 0t0 TCP 127.0.0.1:5100->127.0.0.1:8219 (ESTABLISHED)\n"
        "node 2 me 5u IPv6 0x3 0t0 TCP [::1]:6000->[::1]:8219 (ESTABLISHED)\n"
        "node 2 me 6u IPv4 0x4 0t0 TCP 127.0.0.1:6001->160.79.104.10:443 (SYN_SENT)\n"
        "python 1 me 7u IPv4 0x5 0t0 TCP *:9000 (LISTEN)\n"
    )
    function = subprocess.run(
        ["sed", "-n", "/^external_lines() {/,/^}/p", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result = subprocess.run(
        ["bash", "-c", f"{function}\nexternal_lines"],
        input=lsof,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "node 2 me 6u IPv4 0x4 0t0 TCP 127.0.0.1:6001->160.79.104.10:443 (SYN_SENT)",
        "python 1 me 7u IPv4 0x5 0t0 TCP *:9000 (LISTEN)",
    ]


# ---------------------------------------------------------------- end to end


def free_port() -> int:
    """Ask the system for a free port.

    Returns:
        A port nothing listens on.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def git_status() -> str:
    """The project checkout's ``git status``, to prove it unchanged.

    Returns:
        The porcelain status.
    """
    return subprocess.run(
        ["git", "-C", str(PROJECT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def changed_since(marker: Path) -> list[str]:
    """Every file in the project's checkout written after ``marker``.

    ``git status`` cannot see ignored folders (runs/, models/, tools/, .venv),
    and those are where a leak would land, so the files' times are read.
    Git's own folder is left out: ``git status`` refreshes its index.

    Args:
        marker: A file made before the run.

    Returns:
        The paths, relative to the checkout.
    """
    found = subprocess.run(
        [
            "find",
            str(PROJECT),
            "-newer",
            str(marker),
            "-not",
            "-path",
            "*/.git",
            "-not",
            "-path",
            "*/.git/*",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(str(Path(path).relative_to(PROJECT)) for path in found)


@pytest.mark.skipif(
    not all(p.exists() for p in (PYTHON, CLAUDE, MODEL / "tokenizer.json")),
    reason="needs the project's checkout with .venv, pinned Claude Code, tokenizer",
)
def test_the_whole_pipeline_runs_scripted(course_home: Path, tmp_path: Path) -> None:
    before = git_status()
    marker = tmp_path / "before-the-run"
    marker.touch()
    runs = tmp_path / "runs"
    env = clean_env(
        course_home,
        MHRL_PROJECT=str(PROJECT),
        MHRL_RUNS_DIR=str(runs),
        MHRL_MODEL_PATH=str(MODEL),
        MHRL_ENGINE_PORT=str(free_port()),
        MHRL_TEACHER_PORT=str(free_port()),
        MHRL_PROXY_PORT=str(free_port()),
        MHRL_CLI=f"{PYTHON} {DRIVER}",
        STACK="scripted",
        RUN="scripted",
        EVAL_LIMIT="1",
        DEMO_LIMIT="1",
        GRPO_STEPS="1",
        EVAL_HARNESSES="mini-swe-agent,claude-code",
        MEMORY_EVERY_S="2",
        NETWORK_EVERY_S="2",
    )
    env["MHRL_TEACHER_PORT"] = env["MHRL_ENGINE_PORT"]  # scripted: one server
    code, started = capstone(env, "start", timeout=300)
    try:
        assert code == 0, started
        assert "capture level: tokens" in started
        code, output = capstone(env, "pipeline", timeout=RUN_TIMEOUT_S)
    finally:
        capstone(env, "stop", timeout=120)
    assert code == 0, output
    positions = [output.index(f"=== {name} ===") for name in BANNERS]
    assert positions == sorted(positions)
    out = runs / "scripted"
    stages = (out / "stages.txt").read_text().splitlines()
    assert [line.strip("= ") for line in stages[::2]] == list(BANNERS)
    for label in ("base", "sft", "rl"):
        records = (out / f"eval_{label}.jsonl").read_text().splitlines()
        assert len(records) == 2
        assert {json.loads(r)["harness"] for r in records} == {
            "mini-swe-agent",
            "claude-code",
        }
    assert (out / "sft" / "adapter.safetensors").is_file()
    assert (out / "grpo" / "adapter_final.safetensors").is_file()
    assert "[demos] 2/2 demonstrations usable" in output
    assert "[grpo] step 1/1" in output
    assert "step  correct  calls  tokens/call" in output
    network = (out / "network_check.txt").read_text()
    assert network.startswith("# Network check during the RL evaluation")
    assert re.search(r"one look every 2 s: [1-9]\d* looks", network)
    memory = (out / "memory.tsv").read_text()
    assert "\tengine\t" in memory and "\tstage\t" in memory
    assert (out / "results.json").is_file()
    assert "━━ the final table" in output and "━━ every pair of checkpoints" in output
    assert not (course_home / "28-capstone" / "multi-harness-rl").exists()
    assert not SESSION_ID.search(output), "a whole session id was printed"
    assert git_status() == before
    assert changed_since(marker) == [], "the run wrote into the project's checkout"
