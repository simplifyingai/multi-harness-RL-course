"""Chapter 27: the map, the side panel, the hooks and the stack script.

Nothing here needs the project or a model. The map is pinned from the
bundled snapshot (``tests/data/27_map.txt``) and checked against every
chapter's own README; the hooks print from fake rollouts and fake modules;
the stack script is checked for the two promises it makes beyond the
project's ``serve.sh``: it refuses a port already in use, and it stops only
what it started. ``test_27_project.py`` runs the real pipeline.
"""

from __future__ import annotations

import ast
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("27-how-it-combines")

import course_map  # noqa: E402
import hooks  # noqa: E402
import map as course_map_script  # noqa: E402
import one_step  # noqa: E402
import paths  # noqa: E402
import project_scan  # noqa: E402
import stacks  # noqa: E402

MAP_FIXTURE = ROOT / "tests" / "data" / "27_map.txt"
PROJECT = Path(
    os.environ.get(paths.PROJECT_ENV) or Path.home() / "sai_work" / "multi-harness-rl"
)
HAVE_PROJECT = (PROJECT / paths.PROJECT_MARKER).is_file()
needs_project = pytest.mark.skipif(
    not HAVE_PROJECT, reason="no checkout of the project (set MHRL_PROJECT)"
)
SESSION = "s0123456789abcdef01234567"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BACKTICKED = re.compile(r"`([^`]+)`")


def run_map(*args: str) -> tuple[int, list[str]]:
    """Run map.py's main and keep what it printed.

    Args:
        *args: Its arguments.

    Returns:
        Its exit code and its lines.
    """
    lines: list[str] = []
    code = course_map_script.main(list(args), lines.append)
    return code, lines


@pytest.fixture
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no checkout is named, so map.py reads its snapshot.

    Args:
        monkeypatch: pytest's environment patcher.
    """
    monkeypatch.delenv(paths.PROJECT_ENV, raising=False)


# ------------------------------------------------------------------ the map


def test_the_map_prints_the_pinned_table_from_its_snapshot(
    no_project_env: None,
) -> None:
    code, lines = run_map()
    assert code == 0
    assert "\n".join(lines) + "\n" == MAP_FIXTURE.read_text()


def test_every_published_chapter_is_mapped_to_its_real_folder() -> None:
    numbers = [entry.number for entry in course_map.CHAPTERS]
    assert numbers == list(range(1, 29))
    for entry in course_map.CHAPTERS:
        assert entry.kind in course_map.KINDS
        assert entry.mirrors, entry.folder
        if entry.number in course_map.PUBLISHED or entry.number == 27:
            assert (ROOT / entry.folder).is_dir(), entry.folder
            for name in entry.files:
                assert (ROOT / entry.folder / name).is_file(), (entry.folder, name)


def readme_mirrors(folder: str) -> str | None:
    """Return a chapter README's "Mirrors in the project" section, if any.

    Args:
        folder: The chapter's folder.

    Returns:
        The section's text, or None.
    """
    text = (ROOT / folder / "README.md").read_text()
    match = re.search(
        r"^## Mirrors in the project\n(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    return match.group(1) if match else None


def test_the_map_names_everything_each_readme_says_its_chapter_mirrors() -> None:
    checked = 0
    for entry in course_map.CHAPTERS:
        if entry.number not in course_map.PUBLISHED:
            continue
        section = readme_mirrors(entry.folder)
        if section is None:
            continue
        checked += 1
        paths_mapped = {m.path for m in entry.mirrors}
        parts = {p for m in entry.mirrors for name in m.names for p in name.split(".")}
        parts |= {c for m in entry.mirrors for c in m.commands}
        for token in BACKTICKED.findall(section):
            word = token.split(" ")[0]
            if word.startswith(("mhrl/", "scripts/", "configs/")):
                # a folder (``mhrl/harnesses/``) is mapped by the files in it
                found = any(p == word or p.startswith(word) for p in paths_mapped)
                assert found and (word.endswith("/") or word in paths_mapped), word
            elif word.endswith(".py") and "/" not in word:
                in_package = any(
                    p.endswith("/" + word) or p == word for p in paths_mapped
                )
                assert in_package, (entry.folder, word)
            elif IDENTIFIER.match(word) and not word.isdigit():
                assert word in parts, (entry.folder, word)
    assert checked == 8  # chapters 18 and 20-26 have the section


def test_names_are_measured_for_functions_methods_attributes_and_nested() -> None:
    source = (
        "X = 1\n"
        "class Settings:\n"
        "    port: int = 8100\n"
        "    def url(self):\n"
        "        return 1\n"
        "def outer():\n"
        "    def inner():\n"
        "        return 2\n"
        "    return inner\n"
    )
    measure = project_scan.python_name_lines
    assert measure(source, "X") == 1
    assert measure(source, "Settings") == 4
    assert measure(source, "Settings.port") == 1
    assert measure(source, "Settings.url") == 2
    assert measure(source, "outer.inner") == 2
    assert measure(source, "outer") == 4
    assert measure(source, "missing") is None
    assert measure(source, "Settings.missing") is None


def test_shell_functions_and_cli_commands_are_found() -> None:
    script = "start() {\n  echo hi\n}\nstop() { echo bye; }\n"
    assert project_scan.shell_function_lines(script, "start") == 3
    assert project_scan.shell_function_lines(script, "stop") == 1
    assert project_scan.shell_function_lines(script, "status") is None
    cli = "p.add_argument('command', choices=['eval', 'grpo'])\n"
    assert project_scan.cli_commands(cli) == ("eval", "grpo")
    assert project_scan.count_lines("a\nb\n") == 2


def test_a_checkout_missing_a_name_is_listed_and_the_map_fails(tmp_path: Path) -> None:
    (tmp_path / "mhrl").mkdir()
    (tmp_path / "mhrl" / "config.py").write_text("class Settings:\n    pass\n")
    code, lines = run_map("--project", str(tmp_path))
    assert code == course_map_script.EXIT_MISSING
    text = "\n".join(lines)
    assert "MISSING from the checkout" in text
    assert "project:mhrl/config.py Settings.max_model_calls" in text
    assert "project:mhrl/rollout.py (file)" in text


def test_a_folder_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a checkout"):
        run_map("--project", str(tmp_path))


def test_one_chapter_or_one_part_can_be_printed(no_project_env: None) -> None:
    _, lines = run_map("--chapter", "23")
    labels = [line for line in lines if re.match(r"^\d\d ", line)]
    assert labels == [
        "23 three-harnesses · cut down · wire.py, run_one.py, mini_swe_agent.yaml"
    ]
    _, lines = run_map("--part", "6")
    labels = [line[:2] for line in lines if re.match(r"^\d\d ", line)]
    assert labels == ["20", "21", "22", "23"]
    assert not any(line.startswith("the project's code") for line in lines)
    with pytest.raises(SystemExit, match="no part 9"):
        run_map("--part", "9")


def test_differences_name_what_changed() -> None:
    snapshot = project_scan.load_snapshot()
    files = dict(snapshot.files)
    files["project:mhrl/cli.py"] += 1
    names = {key: dict(value) for key, value in snapshot.names.items()}
    names["project:mhrl/rollout.py"]["reward_for"] = 99
    changed = project_scan.Measured(
        snapshot.commit,
        snapshot.openenv_commit,
        files,
        names,
        snapshot.commands,
        snapshot.code,
        (),
    )
    found = project_scan.differences(changed, snapshot)
    assert found == [
        "lines project:mhrl/cli.py: 322 -> 323",
        "project:mhrl/rollout.py reward_for: 17 -> 99",
    ]
    assert project_scan.differences(snapshot, snapshot) == []


def test_the_snapshot_cannot_be_written_with_names_missing() -> None:
    snapshot = project_scan.load_snapshot()
    broken = project_scan.Measured(
        "x", "y", {}, {}, (), {}, ("project:mhrl/cli.py main",)
    )
    with pytest.raises(project_scan.ProjectMismatchError):
        project_scan.to_json(broken)
    assert project_scan.from_json(project_scan.to_json(snapshot)) == snapshot


@needs_project
def test_the_bundled_snapshot_is_the_checkouts_measurement() -> None:
    live = project_scan.measure(PROJECT)
    assert live.missing == ()
    assert project_scan.differences(live, project_scan.load_snapshot()) == []


@needs_project
def test_the_live_map_says_every_name_is_there_and_the_snapshot_agrees() -> None:
    code, lines = run_map("--project", str(PROJECT))
    assert code == 0
    assert "every mirrored file and name is there; the bundled snapshot" in lines[2]
    assert lines[2].endswith("agrees")
    _, snapshot_lines = run_map()
    assert lines[3:] == snapshot_lines[3:]  # after the "project" lines


# ------------------------------------------------------------- side panel


def test_the_panel_quotes_the_article_and_measures_our_run() -> None:
    lines = stacks.panel("Apple M9, 64 GB", stacks.recorded_grpo())
    text = " ".join(line.strip() for line in lines)
    for row in stacks.ROWS:
        for quote in row.article:
            assert f'"{quote}"' in text
    assert lines[-1] == (
        "our GRPO run   6 steps of 2 groups of 4, 12.6 minutes of steps, trainer "
        "peak 6.88 GB (runs/main/grpo/metrics.jsonl)"
    )
    assert "this machine   Apple M9, 64 GB" in lines
    assert max(len(line) for line in lines[:-1]) <= stacks.WIDTH + len(stacks.INDENT)


def article_page(*, drop: str = "") -> str:
    """A small page holding every quote, split by tags as the article's are.

    Args:
        drop: A quote to leave out.

    Returns:
        The HTML.
    """
    paragraphs = []
    for row in stacks.ROWS:
        for quote in row.article:
            if quote == drop:
                continue
            tagged = quote.replace(
                "(Kwon et al., 2023)", "( <a>Kwon et al., 2023</a> )"
            )
            paragraphs.append(f"<p>{tagged.replace('OpenEnv ', '<b>OpenEnv</b> ')}</p>")
    return "<html><script>var x;</script>" + "".join(paragraphs) + "</html>"


def test_the_verifier_reads_the_page_as_a_reader_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(stacks, "fetch_article", lambda url=None: article_page())
    assert stacks.main(["--verify"], lines.append) == 0
    total = sum(len(row.article) for row in stacks.ROWS)
    assert lines[-1] == f"{total} of {total} quotes found word for word"
    gone = stacks.ROWS[1].article[0]
    monkeypatch.setattr(
        stacks, "fetch_article", lambda url=None: article_page(drop=gone)
    )
    lines.clear()
    assert stacks.main(["--verify"], lines.append) == stacks.EXIT_CHANGED
    assert f'   missing  "{gone}"' in lines


# ------------------------------------------------------------------ hooks


def test_the_key_is_redacted_by_name_and_by_value() -> None:
    env = {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8209",
        "ANTHROPIC_API_KEY": SESSION,
        "GITHUB_TOKEN": "ghp_x",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1024",
        "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        "NOTE": f"key={SESSION}",
    }
    shown = hooks.redact_env(env, SESSION)
    assert shown["ANTHROPIC_API_KEY"] == hooks.REDACTED
    assert shown["GITHUB_TOKEN"] == hooks.REDACTED
    assert shown["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1024"
    assert shown["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"
    assert shown["NOTE"] == "key=<redacted>"
    argv = ["claude", "-p", "task", f"--api-key={SESSION}", SESSION]
    assert hooks.redact_argv(argv, SESSION) == [
        "claude",
        "-p",
        "task",
        "--api-key=<redacted>",
        "<redacted>",
    ]
    assert hooks.short_id(SESSION) == "s0123456…"
    printed = hooks.argv_lines(argv, SESSION, "task") + hooks.env_lines(env, SESSION)
    assert SESSION not in "\n".join(printed)


def test_the_command_line_shortens_the_task_and_wraps() -> None:
    task = "There is a SQLite database named data.db in the current folder. " * 3
    argv = ["/x/claude", "-p", task, "--tools", "Bash"]
    lines = hooks.argv_lines(argv, SESSION, task)
    assert lines[0].startswith("argv  /x/claude -p")
    assert f"({len(task)} characters)" in "\n".join(lines)
    assert all(len(line) <= hooks.WIDTH for line in lines)


def test_temp_folders_are_written_as_tmpdir(tmp_path: Path) -> None:
    import tempfile

    inside = Path(tempfile.gettempdir()).resolve() / "mhrl-harness-claude-code-ab12"
    assert hooks.shorten(str(inside)) == "$TMPDIR/mhrl-harness-claude-code-ab12"
    assert hooks.shorten(str(Path.home() / "x")) == "~/x"


DOCUMENT = {
    "turns": [
        {
            "n_prompt": 2400,
            "n_sampled": 28,
            "n_tools": 1,
            "finish_reason": "tool_calls",
        },
        {"n_prompt": 2528, "n_sampled": 19, "n_tools": 1, "finish_reason": "stop"},
        {"n_prompt": 90, "n_sampled": 5, "n_tools": 0, "discarded": True},
    ],
    "stats": {"n_roots": 1, "n_sequences": 2, "n_trainable_sequences": 1},
    "trainable": True,
}


def test_each_captured_call_and_the_rollout_shape_are_printed() -> None:
    assert hooks.call_lines(DOCUMENT) == [
        "call 1 · prompt 2,400 tokens · sampled 28 · tools 1 · tool_calls",
        "call 2 · prompt 2,528 tokens · sampled 19 · tools 1 · stop",
        "call 3 · prompt 90 tokens · sampled 5 · tools 0 · no finish reason · "
        "auxiliary, not trained",
    ]
    assert hooks.rollout_summary(DOCUMENT) == (
        "captured 3 calls · roots 1 · rows 1 trainable of 2 · trainable true"
    )


def test_the_reward_line_shows_its_arithmetic() -> None:
    right = hooks.reward_line(True, "73.89", "correct", 177, 1.09115, 2000, 0.1)
    assert right == (
        "grade correct (answer '73.89') · reward 1 + 0.1 x (1 - 177/2000) = 1.0912"
    )
    wrong = hooks.reward_line(False, None, "no answer.txt", 50, 0.0, 2000, 0.1)
    assert wrong == "grade wrong (no answer.txt) · reward 0: a wrong answer earns 0"


def test_a_group_without_contrast_says_why_it_teaches_nothing() -> None:
    lines = hooks.group_lines([1.09, 1.09, 1.09, 1.09], None)
    text = " ".join(lines)
    assert "no contrast" in text and "chapter 5" in text and "no update" in text
    lines = hooks.group_lines([0.0, 1.09, 1.09, 0.0], [-1.0, 1.0, 1.0, -1.0])
    assert (
        lines[0] == "rewards 0.0000 1.0900 1.0900 0.0000 · mean 0.5450 · spread 0.5450"
    )
    assert lines[1].endswith("-1.00 +1.00 +1.00 -1.00")


def test_each_labelled_rollout_gets_its_own_advantage() -> None:
    rewards = [0.0, 1.09, 1.08, 0.0]
    advantages = [-1.0, 1.01, 0.99, -1.0]
    labelled = [("r1", 1.08), ("r2", 0.0), ("r3", 1.09), ("r4", 0.0)]
    assert hooks.matched_advantages(rewards, advantages, labelled) == [
        ("r1", 1.08, 0.99),
        ("r2", 0.0, -1.0),
        ("r3", 1.09, 1.01),
        ("r4", 0.0, -1.0),
    ]


def test_a_row_window_shows_the_mask_and_old_logprobs() -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = [0, 0, 0, 1, 1, 0]
    logprobs = [0.0, 0.0, 0.0, -0.25, -1.5, 0.0]
    lines = hooks.row_window(ids, mask, logprobs, lambda token: f"t{token}")
    assert lines[0] == "position  mask  old logprob  token"
    assert lines[1] == "       0     0               't10'"
    assert lines[4] == "       3     1      -0.2500  't13'"
    assert lines[5] == "       4     1      -1.5000  't14'"
    assert hooks.row_window([1], [0], [0.0], str) == ["(no trained token)"]


def test_ratios_loss_and_the_engines_counters_are_summarised() -> None:
    line = hooks.ratio_line(1, [1.0, 0.999, 1.002], -0.5)
    assert line == (
        "row 1: 3 ratios, first 1.000 0.999 1.002 · min 0.999 max 1.002 "
        "mean 1.0003 · loss -0.5000"
    )
    before = {"requests": 2, "prompt_tokens": 100, "cached_tokens": 0}
    after = {
        "requests": 22,
        "prompt_tokens": 50_100,
        "cached_tokens": 40_000,
        "completion_tokens": 700,
        "model_seconds": 12.5,
    }
    assert hooks.engine_delta(before, after) == (
        "engine  20 requests · 50,000 prompt tokens, 40,000 of them from its prompt "
        "cache (80%) · 700 sampled · 12.5 s of model time"
    )


def test_patched_names_come_back_even_after_an_error() -> None:
    module = types.ModuleType("fake")
    module.value = 1
    with pytest.raises(RuntimeError), hooks.patched(module, value=2):
        assert module.value == 2
        raise RuntimeError("boom")
    assert module.value == 1
    with pytest.raises(AttributeError), hooks.patched(module, renamed=3):
        pass
    namespace = hooks.Namespace(module, value=5)
    assert namespace.value == 5
    module.other = "x"
    assert namespace.other == "x"


def fake_grpo_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Fake ``mhrl.training.grpo`` and ``mhrl.rollout`` with every hand-off name.

    Returns:
        The two modules.
    """
    grpo = types.ModuleType("fake_grpo")
    rollout = types.ModuleType("fake_rollout")

    class FakeHarness:
        def command(self, ctx: Any) -> list[str]:
            return ["claude", "-p", ctx.instruction]

        def environment(self, ctx: Any) -> dict[str, str]:
            return {"ANTHROPIC_API_KEY": ctx.api_key, "ANTHROPIC_MODEL": "m"}

    class FakeClient:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def rollout(self, session_id: str) -> dict[str, Any]:
            return DOCUMENT

    def run_rollout(*args: Any, on_session: Any = None, **kwargs: Any) -> Any:
        workdir = args[5]
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "data.db").write_text("")
        on_session(SESSION, workdir)
        harness = rollout.get_harness("claude-code")
        ctx = types.SimpleNamespace(
            instruction="task",
            api_key=SESSION,
            state_dir=Path("/tmp/state"),
        )
        harness.environment(ctx)
        harness.command(ctx)
        args[4].rollout(SESSION)
        return types.SimpleNamespace(
            correct=True,
            submitted=f"7 {SESSION}",  # a model that echoed its key
            reason="correct",
            n_sampled=47,
            reward=1.0977,
            seconds=1.0,
        )

    grpo.Defaults = types.SimpleNamespace(MAX_TOOL_BONUS=0.1)
    grpo.load_policy = lambda settings: (None, None)
    grpo.use_adapter = lambda settings, path: None
    grpo.CaptureClient = FakeClient
    grpo.run_rollout = run_rollout
    grpo.group_advantages = lambda rewards: None
    grpo.dedupe_shared_prefixes = lambda sequences: sequences
    grpo.item_from_sequence = lambda sequence, advantage=1.0: None
    grpo.save_adapter = lambda model, path: path
    grpo.nn = types.SimpleNamespace(value_and_grad=lambda model, fn: fn)
    grpo.optim = types.SimpleNamespace(AdamW=type("AdamW", (), {}))
    rollout.get_harness = lambda name: FakeHarness()
    return grpo, rollout


def test_the_tracer_prints_each_hand_off_and_restores_every_name(
    tmp_path: Path,
) -> None:
    grpo, rollout = fake_grpo_modules()
    originals = dict(vars(grpo)), dict(vars(rollout))
    settings = types.SimpleNamespace(
        token_budget=2000, engine_url="http://127.0.0.1:9", proxy_url="x"
    )
    said: list[str] = []
    tracer = hooks.Tracer(settings, said.append)
    with tracer.listening(grpo, rollout):
        assert grpo.run_rollout is not originals[0]["run_rollout"]
        tracer.start_draw()
        client = grpo.CaptureClient(settings)
        record = grpo.run_rollout(
            None, "claude-code", "train", settings, client, tmp_path
        )
        grpo.group_advantages([record.reward] * 4)
    assert vars(grpo) == originals[0] and vars(rollout) == originals[1]
    text = "\n".join(said)
    assert f"[r1] folder {tmp_path} holds data.db" in text
    assert (
        "[r1] API key = the session id s0123456… (POST /sessions on the proxy)" in text
    )
    assert "argv  claude -p task" in text
    assert "ANTHROPIC_API_KEY=<redacted>" in text
    assert "call 2 · prompt 2,528 tokens · sampled 19 · tools 1 · stop" in text
    assert "reward 1 + 0.1 x (1 - 47/2000) = 1.0977" in text
    assert "no contrast" in text
    assert SESSION not in text
    assert tracer.outcome.flat_groups == 1 and not tracer.outcome.updated


# ---------------------------------------------------------------- one_step


def test_tasks_come_in_train_grpos_order() -> None:
    import random

    tasks = list(range(10))
    expected = list(tasks)
    random.Random(0).shuffle(expected)
    assert one_step.train_order(tasks) == expected
    assert tasks == list(range(10))  # the caller's list is untouched


def test_the_settings_say_where_each_value_came_from() -> None:
    settings = types.SimpleNamespace(
        engine_port=8129,
        proxy_port=8209,
        runs_dir=Path.home() / "runs",
        model_path=Path("/m"),
        max_model_calls=8,
        train_temperature=1.0,
        max_completion_tokens=1024,
        token_budget=2000,
        length_guard_tokens=150,
        concurrency=4,
    )
    lines = one_step.settings_lines(settings, {"MHRL_ENGINE_PORT": "8129"})
    assert lines[1].split() == ["engine_port", "8129", "MHRL_ENGINE_PORT=8129"]
    assert lines[2].split() == ["proxy_port", "8209", "default"]
    assert lines[3].split()[1] == "~/runs"


def test_the_metrics_lines_read_train_grpos_record() -> None:
    metrics = {
        "train_rows": 4,
        "train_tokens": 708,
        "skipped_long_rows": 0,
        "untrainable_rollouts": 0,
        "importance_ratio_mean": 0.99987,
        "loss_per_token": 0.01234,
        "tokens_per_call": 35.4,
        "step_seconds": 61.0,
        "peak_gb": 4.2,
    }
    lines = one_step.metrics_lines(metrics)
    assert lines[0].endswith(
        "rows 4 · tokens 708 · rows over the cap 0 · untrainable rollouts 0"
    )
    assert lines[1].strip() == (
        "ratio 0.9999 · loss per token +0.0123 · tokens per call 35 · step 61.0 s · "
        "peak 4.2 GB"
    )
    assert one_step.metrics_lines({}) == ["metrics   none written"]


def test_the_start_adapter_is_base_or_an_existing_file(tmp_path: Path) -> None:
    assert one_step.start_adapter("base") is None
    adapter = tmp_path / "a.safetensors"
    adapter.write_bytes(b"x")
    assert one_step.start_adapter(str(adapter)) == adapter.resolve()
    with pytest.raises(SystemExit, match="no such adapter"):
        one_step.start_adapter(str(tmp_path / "missing.safetensors"))


def test_without_mhrl_a_checkout_must_be_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.PROJECT_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "mhrl", None)  # import mhrl fails
    with pytest.raises(SystemExit, match="name the project's checkout"):
        one_step.ensure_project_python(None, [], print)
    with pytest.raises(SystemExit, match="not a checkout"):
        one_step.ensure_project_python(str(tmp_path), [], print)
    (tmp_path / "mhrl").mkdir()
    (tmp_path / "mhrl" / "config.py").write_text("")
    with pytest.raises(SystemExit, match="run `uv sync` in the project"):
        one_step.ensure_project_python(str(tmp_path), [], print)


def test_the_rerun_with_the_projects_python_happens_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mhrl").mkdir()
    (tmp_path / "mhrl" / "config.py").write_text("")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setitem(sys.modules, "mhrl", None)
    monkeypatch.delenv(one_step.REEXEC_ENV, raising=False)
    calls: list[list[str]] = []

    def fake_execv(path: object, argv: list[str]) -> None:
        calls.append(argv)
        raise SystemExit(0)

    monkeypatch.setattr(one_step.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        one_step.ensure_project_python(str(tmp_path), ["--max-draws", "1"], print)
    assert calls[0][-2:] == ["--max-draws", "1"]
    assert os.environ[one_step.REEXEC_ENV] == "1"
    calls.clear()
    with pytest.raises(SystemExit, match="still cannot import its mhrl"):
        one_step.ensure_project_python(str(tmp_path), [], print)
    assert calls == []  # the second pass stops instead of re-running


def test_the_tracer_hides_a_known_key_on_any_line() -> None:
    settings = types.SimpleNamespace(token_budget=2000, engine_url="x")
    said: list[str] = []
    tracer = hooks.Tracer(settings, said.append)
    tracer.secrets.add(SESSION)
    tracer.emit([f"grade wrong (answer '{SESSION}')"])
    assert said == ["grade wrong (answer '<redacted>')"]
    assert tracer.redact(f"GET /sessions/{SESSION}/rollout") == (
        "GET /sessions/<redacted>/rollout"
    )
    assert hooks.short_id("s123") == "s1…"


class HealthHandler(http.server.BaseHTTPRequestHandler):
    """Answers /health with whatever the test put on the server."""

    def do_GET(self) -> None:  # noqa: N802 - the http.server hook's name
        body = json.dumps(self.server.health).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def health_server() -> Iterator[http.server.HTTPServer]:
    """A local server answering /health, on a free port.

    Yields:
        The server; set ``server.health`` to what it should answer.
    """
    server = http.server.HTTPServer(("127.0.0.1", 0), HealthHandler)
    server.health = {}  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def fake_project(engine_url: str, proxy_url: str, installed: bool = True) -> Any:
    """A Project with fake settings and a fake Claude Code.

    Args:
        engine_url: Where the engine answers.
        proxy_url: Where the proxy answers.
        installed: Whether Claude Code is installed.

    Returns:
        The project.
    """

    class Missing(Exception):
        pass

    class Harness:
        pinned_version = "2.1.290"
        binary = "/x/claude"

        def check_installed(self) -> None:
            if not installed:
                raise Missing("pinned claude 2.1.290 not found")

    settings = types.SimpleNamespace(engine_url=engine_url, proxy_url=proxy_url)
    return one_step.Project(
        settings, None, None, [], None, lambda name: Harness(), Missing
    )


def test_the_stack_check_refuses_a_server_that_is_not_the_engine(
    health_server: http.server.HTTPServer,
) -> None:
    url = f"http://127.0.0.1:{health_server.server_port}"
    health_server.health = {"status": "ok", "service": "auth"}  # type: ignore[attr-defined]
    with pytest.raises(one_step.StackNotReadyError, match="not the project's engine"):
        one_step.check_stack(fake_project(url, url))
    health_server.health = {  # type: ignore[attr-defined]
        "ok": True,
        "adapter": "base",
        "upstream": "http://127.0.0.1:1",
        "capture_level": "tokens",
    }
    with pytest.raises(one_step.StackNotReadyError, match="forwards to"):
        one_step.check_stack(fake_project(url, url))
    health_server.health["upstream"] = url  # type: ignore[attr-defined]
    lines = one_step.check_stack(fake_project(url, url))
    assert lines[1].endswith(f"capture level: tokens · forwards to {url}")
    with pytest.raises(one_step.StackNotReadyError, match="not found"):
        one_step.check_stack(fake_project(url, url, installed=False))


def test_the_stack_check_says_how_to_start_a_silent_stack() -> None:
    url = "http://127.0.0.1:1"
    with pytest.raises(one_step.StackNotReadyError, match="./stack.sh start"):
        one_step.check_stack(fake_project(url, url))


def test_the_hand_off_names_are_the_ones_train_grpo_calls() -> None:
    """Every name the tracer swaps is imported by the project's grpo.py.

    Read from the snapshot's view of grpo.py would not show imports, so this
    reads the hooks' own list and checks it against the project's module
    when there is a checkout.
    """
    tree = ast.parse((CHAPTER / "hooks.py").read_text())
    patched_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "patched":
            patched_names |= {keyword.arg for keyword in node.keywords if keyword.arg}
    assert patched_names == {
        "load_policy",
        "use_adapter",
        "CaptureClient",
        "run_rollout",
        "group_advantages",
        "dedupe_shared_prefixes",
        "item_from_sequence",
        "save_adapter",
        "nn",
        "optim",
        "get_harness",
    }
    if HAVE_PROJECT:
        grpo_source = (PROJECT / "mhrl" / "training" / "grpo.py").read_text()
        rollout_source = (PROJECT / "mhrl" / "rollout.py").read_text()
        for name in patched_names - {"get_harness"}:
            assert re.search(rf"\b{name}\b", grpo_source), name
        assert "get_harness(harness_name)" in rollout_source


# ------------------------------------------------------------------ stack.sh


def free_port() -> int:
    """Ask the system for a free port.

    Returns:
        A port nothing listens on.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def fake_checkout(tmp_path: Path) -> Path:
    """A folder that passes stack.sh's first check (an executable Python).

    Args:
        tmp_path: pytest's folder.

    Returns:
        The fake checkout.
    """
    project = tmp_path / "project"
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").symlink_to(sys.executable)
    return project


def test_the_stack_refuses_a_port_already_in_use(tmp_path: Path) -> None:
    project = fake_checkout(tmp_path)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = subprocess.run(
            [str(CHAPTER / "stack.sh"), "scripted"],
            env={
                **os.environ,
                "MHRL_PROJECT": str(project),
                "MHRL_ENGINE_PORT": str(port),
                "MHRL_PROXY_PORT": str(free_port()),
                "COURSE_HOME": str(tmp_path / "home"),
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert result.returncode == 1
    assert f"port {port} is already in use" in result.stderr
    assert not (tmp_path / "home" / "27-how-it-combines" / "stack").exists()


def run_stop(project: Path, home: Path) -> subprocess.CompletedProcess[str]:
    """Run ``stack.sh stop``.

    Args:
        project: The fake checkout.
        home: The COURSE_HOME whose pid files it reads.

    Returns:
        The finished process.
    """
    return subprocess.run(
        [str(CHAPTER / "stack.sh"), "stop"],
        env={**os.environ, "MHRL_PROJECT": str(project), "COURSE_HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def fake_server(port: int) -> subprocess.Popen[bytes]:
    """A process whose command line looks like one of the project's servers.

    Args:
        port: The ``--port`` it shows.

    Returns:
        The running process (it only sleeps).
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "-m",
            "mhrl.engine.server",
            "--port",
            str(port),
        ]
    )


def start_time(pid: int) -> str:
    """Return a process's start time as ``ps`` prints it (what stack.sh keeps).

    Args:
        pid: The process.

    Returns:
        Its start time.
    """
    return subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True
    ).stdout


def record_server(logs: Path, name: str, pid: int, port: int, start: str) -> None:
    """Write the pid, port and start files stack.sh writes at launch.

    Args:
        logs: Its log folder.
        name: The server's name.
        pid: Its pid.
        port: The port it was started on.
        start: Its start time.
    """
    (logs / f"{name}.pid").write_text(f"{pid}\n")
    (logs / f"{name}.port").write_text(f"{port}\n")
    (logs / f"{name}.start").write_text(start)


def test_the_stack_stops_only_its_own_servers(tmp_path: Path) -> None:
    project = fake_checkout(tmp_path)
    home = tmp_path / "home"
    logs = home / "27-how-it-combines" / "stack"
    logs.mkdir(parents=True)
    stranger = subprocess.Popen(["sleep", "60"])  # not a server at all
    other_port = fake_server(4321)  # a server like ours, on another port
    restarted = fake_server(5432)  # our port, but it started at another time
    ours = fake_server(6543)  # the one this script started
    try:
        time.sleep(0.5)
        record_server(logs, "engine", stranger.pid, 1111, start_time(stranger.pid))
        record_server(logs, "teacher", other_port.pid, 1234, start_time(other_port.pid))
        record_server(logs, "proxy", restarted.pid, 5432, "Thu Jan  1 00:00:00 1970\n")
        result = run_stop(project, home)
        assert result.returncode == 0, result.stderr
        assert stranger.poll() is None
        assert other_port.poll() is None
        assert restarted.poll() is None
        assert result.stdout == ""
        assert not list(logs.glob("*.pid"))
        record_server(logs, "engine", ours.pid, 6543, start_time(ours.pid))
        result = run_stop(project, home)
        assert result.stdout.startswith(f"stopped engine (pid {ours.pid})")
        assert ours.wait(timeout=30) is not None
    finally:
        for process in (stranger, other_port, restarted, ours):
            process.kill()
            process.wait()


def test_the_stack_never_prints_a_session_id_from_a_log(tmp_path: Path) -> None:
    project = fake_checkout(tmp_path)
    home = tmp_path / "home"
    logs = home / "27-how-it-combines" / "stack"
    logs.mkdir(parents=True)
    (logs / "proxy.pid").write_text("999999\n")  # long gone
    (logs / "proxy.log").write_text(
        f'INFO: 127.0.0.1:1 - "GET /sessions/{SESSION}/rollout HTTP/1.1" 200 OK\n'
        f'INFO: 127.0.0.1:1 - "DELETE /sessions/{SESSION} HTTP/1.1" 200 OK\n'
    )
    result = run_stop(project, home)
    assert result.returncode == 0
    assert SESSION not in result.stdout
    assert "/sessions/<redacted>/rollout" in result.stdout
    assert '"DELETE /sessions/<redacted> HTTP/1.1"' in result.stdout
