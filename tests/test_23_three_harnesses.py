"""Chapter 23 without a model: three real harnesses, wired by configuration only.

No test here loads weights or needs MLX. The unit tests check the pins, the
wiring each harness reads, the network watch, the stop rules and the
recorded transcripts (``replay.py``'s output is pinned in
``tests/data/23_replay.txt``). The end-to-end tests run each pinned harness
as a learner does: inside the sandbox, through OpenEnv's real capture proxy
(started by ``start.sh``), in front of the chapter's scripted engine served
over real HTTP. They are skipped when the proxy or the harnesses are not on
disk (``./start.sh install`` and ``uv run install.py``); prompt sizes are
checked in LFM2.5's tokens when its tokenizer is on disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from chapters import use_chapter

CHAPTER = use_chapter("23-three-harnesses")

import capture_client  # noqa: E402
import engine  # noqa: E402
import install  # noqa: E402
import lfm_files  # noqa: E402
import loopback  # noqa: E402
import replay  # noqa: E402
import run_one  # noqa: E402
import scripted_engine  # noqa: E402
import scripted_replies  # noqa: E402
import tasks  # noqa: E402
import wire  # noqa: E402
from policy import Sampling  # noqa: E402

START_SH = CHAPTER / "start.sh"
REPLAY_GOLDEN = Path(__file__).parent / "data" / "23_replay.txt"
# Worked out at import, before conftest points COURSE_HOME at a temporary
# folder: where the proxy, the harnesses and LFM2.5 are on this machine.
_HOME = Path(os.environ.get("COURSE_HOME") or Path.home() / ".cache" / "mhrl-course")
PINNED = re.search(r"^PROXY_COMMIT=(\w+)", START_SH.read_text(), re.MULTILINE)
START_TIMEOUT_S = 300
CHAPTERS_WITH_PROXY = (
    "23-three-harnesses",
    "22-silent-capture-failures",
    "21-calls-to-rows",
)
REAL_INSTALLS = _HOME / "23-three-harnesses"
TASK = tasks.find_task(run_one.DEFAULT_TASK)


def installed_commit(env: Path) -> str | None:
    """The OpenEnv commit installed in an environment, read from its metadata.

    Args:
        env: The environment's folder.

    Returns:
        The commit, or None when OpenEnv is not installed there.
    """
    for record in env.glob("lib/python*/site-packages/openenv-*.dist-info"):
        found = json.loads((record / "direct_url.json").read_text())
        return found["vcs_info"]["commit_id"]
    return None


def find_proxy_env() -> Path | None:
    """Find a proxy environment at the pinned commit.

    Returns:
        The environment, or None.
    """
    named = os.environ.get("PROXY_ENV")
    candidates = [Path(named)] if named else []
    candidates += [_HOME / chapter / "proxy-env" for chapter in CHAPTERS_WITH_PROXY]
    for env in candidates:
        if PINNED is not None and installed_commit(env) == PINNED[1]:
            return env
    return None


def harnesses_on_disk() -> bool:
    """Say whether all three pinned harnesses are installed in the real home.

    Returns:
        True when each one's install folder is there.
    """
    return all(
        (REAL_INSTALLS / install.PINS[name].folder).is_dir()
        for name in install.HARNESSES
    )


def find_tokenizer() -> Path | None:
    """Find LFM2.5's checked tokenizer.json (and its template) in the real home.

    Returns:
        The file, or None.
    """
    for name in lfm_files.TOKENIZER_FOLDERS:
        path = _HOME / lfm_files.MODELS_FOLDER / name / lfm_files.TOKENIZER_FILE
        template = path.parent / "chat_template.jinja"
        if (
            path.is_file()
            and template.is_file()
            and lfm_files.sha256_of(path) == lfm_files.TOKENIZER_SHA256
        ):
            return path
    return None


REAL_PROXY_ENV = find_proxy_env()
LFM_TOKENIZER_FILE = find_tokenizer()
needs_stack = pytest.mark.skipif(
    REAL_PROXY_ENV is None or not harnesses_on_disk(),
    reason="needs the capture proxy (./start.sh install) and the three harnesses "
    "(uv run install.py)",
)
needs_tokenizer = pytest.mark.skipif(
    LFM_TOKENIZER_FILE is None, reason="LFM2.5's tokenizer files are not on disk"
)


# ------------------------------------------------- the pins and the installs


def test_each_harness_is_pinned_to_the_projects_version() -> None:
    assert {name: pin.version for name, pin in install.PINS.items()} == {
        "mini-swe-agent": "2.4.6",
        "claude-code": "2.1.290",
        "pi": "1.0.4",
    }


@pytest.mark.parametrize("name", install.HARNESSES)
def test_any_other_version_is_refused(name: str) -> None:
    install.check_version(name, install.PINS[name].version)
    with pytest.raises(install.HarnessVersionError, match="pinned to"):
        install.check_version(name, "9.9.9")
    with pytest.raises(install.HarnessVersionError, match="nothing"):
        install.check_version(name, "")


def test_a_version_is_the_first_word_a_program_prints() -> None:
    assert install.parse_version("2.1.290 (Claude Code)\n") == "2.1.290"
    assert install.parse_version("1.0.4\n") == "1.0.4"
    assert install.parse_version("") == ""


def test_npm_installs_with_private_settings_and_the_public_registry(
    course_home: Path,
) -> None:
    env = install.npm_environment()
    chapter = course_home / "23-three-harnesses"
    assert not (chapter / "npm").exists()  # working it out writes nothing
    install.prepare_installer(install.NPM)
    assert env["npm_config_userconfig"] == str(chapter / "npm" / "npmrc")
    assert env["npm_config_globalconfig"] == env["npm_config_userconfig"]
    assert (chapter / "npm" / "npmrc").read_text() == ""  # nothing of yours
    assert env["npm_config_registry"] == "https://registry.npmjs.org/"
    assert env["HOME"] == str(chapter / "npm")
    assert str(Path.home()) not in env["npm_config_cache"]
    for name in ("claude-code", "pi"):
        (command,) = install.install_commands(name)
        pin = install.PINS[name]
        assert command[:4] == ["npm", "install", "--prefix", str(chapter / pin.folder)]
        assert command[-1] == f"{pin.package}@{pin.version}"
        assert "--save-exact" in command


def test_mini_swe_agent_installs_its_pinned_list_without_resolving(
    course_home: Path,
) -> None:
    venv, pip = install.install_commands("mini-swe-agent")
    assert venv[:2] == ["uv", "venv"] and venv[-1].endswith("mini-swe-agent-env")
    assert venv[venv.index("--python") + 1] == "3.13.7" and "--managed-python" in venv
    assert pip[pip.index("--default-index") + 1] == "https://pypi.org/simple"
    assert "--no-deps" in pip
    pinned = install.MINI_REQUIREMENTS.read_text().splitlines()
    assert "mini-swe-agent==2.4.6" in pinned
    assert "litellm==1.104.0" in pinned
    requirements = [line for line in pinned if line and not line.startswith("#")]
    assert all("==" in line for line in requirements)


def test_uv_installs_with_none_of_your_settings(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_INDEX_URL", "https://token@private.example/simple")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://private.example/simple")
    env = install.uv_environment()
    folder = course_home / "23-three-harnesses" / "uv"
    assert env["UV_NO_CONFIG"] == "1" and env["HOME"] == str(folder)
    assert env["UV_CACHE_DIR"].startswith(str(folder))
    assert env["UV_PYTHON_INSTALL_DIR"].startswith(str(folder))
    assert not any(key.startswith(("UV_INDEX", "UV_EXTRA")) for key in env)
    venv, _ = install.install_commands("mini-swe-agent")
    shown = install.shown_command(venv, install.PYPI)
    assert shown.startswith("UV_NO_CONFIG=1 UV_CACHE_DIR=")
    assert str(Path.home()) not in shown


def fake_install(name: str, version: str) -> None:
    """Leave a harness's version record on disk, as its install would.

    Args:
        name: The harness.
        version: The version to record.
    """
    if name == "mini-swe-agent":
        site = install.install_folder(name) / "lib" / "python3.13" / "site-packages"
        (site / f"mini_swe_agent-{version}.dist-info").mkdir(parents=True)
        return
    folder = install.package_folder(name)
    folder.mkdir(parents=True)
    (folder / "package.json").write_text(json.dumps({"version": version}))


@pytest.mark.parametrize("name", install.HARNESSES)
def test_the_version_on_disk_is_read_from_the_packages_own_record(name: str) -> None:
    assert install.resolved_version(name) == ""
    fake_install(name, install.PINS[name].version)
    assert install.resolved_version(name) == install.PINS[name].version


def test_a_wrong_version_on_disk_is_refused_never_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_install("pi", "1.0.5")
    monkeypatch.setattr(install, "run_install", lambda name: pytest.fail("installed"))
    with pytest.raises(install.HarnessVersionError, match="1.0.5"):
        install.ensure("pi", True, {})


def test_a_missing_harness_is_not_installed_by_a_check() -> None:
    with pytest.raises(install.HarnessInstallError, match="uv run install.py"):
        install.ensure("claude-code", False, {})


def test_a_harness_found_on_disk_says_its_record_was_not_made_by_install() -> None:
    fake_install("claude-code", "2.1.290")
    records: dict[str, dict] = {}
    found = install.ensure("claude-code", False, records)
    assert found.note == install.FOUND_EARLIER
    assert records["claude-code"]["resolved"] == "2.1.290"
    assert json.loads((install.install_root() / install.RECORD_FILE).read_text())


# ------------------------------------------------- the wiring


def run_folders(root: Path) -> wire.RunFolders:
    """Make a run's folders under a temporary root.

    Args:
        root: Any folder.

    Returns:
        The folders, created.
    """
    folders = wire.RunFolders(task=root / "task", state=root / "state")
    folders.make()
    return folders


def assert_nothing_of_yours(env: dict[str, str], folders: wire.RunFolders) -> None:
    """Check an environment holds none of the caller's variables or folders.

    Args:
        env: The harness's environment.
        folders: The run's folders.
    """
    assert env["HOME"] == str(folders.home)
    assert env["TMPDIR"] == str(folders.tmp)
    assert "COURSE_HOME" not in env and "VIRTUAL_ENV" not in env
    for value in env.values():
        assert value.startswith(str(folders.state)) or str(Path.home()) not in value


def test_mini_swe_agent_reads_a_yaml_overlay_pointing_at_the_proxy(
    tmp_path: Path,
) -> None:
    folders = run_folders(tmp_path)
    wiring = wire.wire("mini-swe-agent", folders, "the task", "http://p:1/", "s1")
    overlay = json.loads(wiring.files[folders.state / wire.MINI_OVERLAY])
    assert overlay["model"] == {
        "model_name": "openai/lfm2.5-2.6b",
        "model_kwargs": {"api_base": "http://p:1/v1", "api_key": "s1"},
    }
    assert overlay["agent"]["step_limit"] == 8
    assert wiring.argv.count("-c") == 2 and str(wire.MINI_CONFIG) in wiring.argv
    assert wiring.argv[wiring.argv.index("-t") + 1] == "the task"
    assert wiring.env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert_nothing_of_yours(wiring.env, folders)


def test_mini_swe_agents_fixed_config_is_the_projects() -> None:
    text = wire.MINI_CONFIG.read_text()
    body = text[text.index("agent:\n") :]
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in body
    assert "model_class: litellm" in body and "step_limit: 8" in body
    assert "Every action is executed in a new subshell." in body
    assert "temperature" not in body.split("model:")[1].split("observation")[0]


def test_claude_code_reads_its_environment_and_the_projects_flags(
    tmp_path: Path,
) -> None:
    folders = run_folders(tmp_path)
    wiring = wire.wire("claude-code", folders, "the task", "http://p:1", "s1")
    env = wiring.env
    assert env["ANTHROPIC_BASE_URL"] == "http://p:1"  # it adds /v1/messages
    assert env["ANTHROPIC_API_KEY"] == "s1"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "lfm2.5-2.6b"
    assert env["CLAUDE_CONFIG_DIR"].startswith(str(folders.state))
    assert env["CLAUDE_CODE_TMPDIR"] == str(folders.tmp)
    assert env["CLAUDE_CODE_TOTAL_TOKENS_REMINDER"] == "off"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1024"
    switches = ("DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING", "DISABLE_AUTOUPDATER")
    assert all(env[switch] == "1" for switch in switches)
    argv = wiring.argv
    assert argv[argv.index("--tools") + 1] == "Bash"
    assert argv[argv.index("--setting-sources") + 1] == ""
    for flag in (
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ):
        assert flag in argv
    assert "--bare" not in argv and wiring.files == {}
    assert_nothing_of_yours(env, folders)


def test_claude_codes_variants_change_only_their_flag(tmp_path: Path) -> None:
    folders = run_folders(tmp_path)

    def argv(setup: str) -> list[str]:
        options = wire.Options(claude_setup=setup)
        return wire.wire("claude-code", folders, "t", "u", "k", options).argv

    assert argv("bare") == [*argv("project"), "--bare"]
    assert "--tools" not in argv("default-tools")
    with pytest.raises(ValueError, match="unknown Claude Code setup"):
        argv("everything")


def test_pi_reads_a_models_json_naming_its_api(tmp_path: Path) -> None:
    folders = run_folders(tmp_path)
    wiring = wire.wire("pi", folders, "the task", "http://p:1", "s1")
    agent = folders.state / wire.PI_AGENT_FOLDER
    models = json.loads(wiring.files[agent / wire.PI_MODELS_FILE])
    assert models["providers"] == {
        "intercept": {
            "baseUrl": "http://p:1/v1",
            "api": "openai-completions",
            "apiKey": "s1",
            "models": [
                {
                    "id": "lfm2.5-2.6b",
                    "contextWindow": 16_384,
                    "maxTokens": 1_024,
                    "input": ["text"],
                }
            ],
        }
    }
    settings = json.loads(wiring.files[agent / wire.PI_SETTINGS_FILE])
    assert settings["compaction"]["reserveTokens"] == 1_024
    assert wiring.env["PI_CODING_AGENT_DIR"] == str(agent)
    assert wiring.env["PI_OFFLINE"] == "1" and wiring.env["PI_TELEMETRY"] == "0"
    argv = wiring.argv
    assert argv[argv.index("--provider") + 1] == "intercept"
    assert "--offline" in argv and "--no-context-files" in argv
    assert_nothing_of_yours(wiring.env, folders)


def test_pi_with_its_api_left_to_it_writes_no_api_field(tmp_path: Path) -> None:
    options = wire.Options(pi_provider="openai", pi_api="default")
    wiring = wire.wire("pi", run_folders(tmp_path), "t", "u", "k", options)
    models = next(
        json.loads(text)
        for path, text in wiring.files.items()
        if path.name == wire.PI_MODELS_FILE
    )
    assert "api" not in models["providers"]["openai"]
    assert "(api not set: Pi chooses)" in wiring.settings[0]


def test_wire_writes_each_file_and_refuses_an_unknown_harness(tmp_path: Path) -> None:
    folders = run_folders(tmp_path)
    wiring = wire.wire("pi", folders, "t", "u", "k")
    wiring.write()
    assert all(path.read_text() == text for path, text in wiring.files.items())
    with pytest.raises(ValueError, match="unknown harness"):
        wire.wire("opencode", folders, "t", "u", "k")


# ------------------------------------------------- the network watch


def test_lsofs_field_output_is_read_socket_by_socket() -> None:
    output = (
        "p101\ncclaude\nf23\nn127.0.0.1:50000->127.0.0.1:8200\n"
        "f24\nn*:*\np102\ncnode\nf5\nn10.0.0.2:5000->151.101.1.1:443\n"
    )
    assert loopback.parse_lsof(output) == [
        ("claude", "127.0.0.1:50000->127.0.0.1:8200"),
        ("claude", "*:*"),
        ("node", "10.0.0.2:5000->151.101.1.1:443"),
    ]


def test_a_process_tree_is_found_through_parent_ids_and_its_group() -> None:
    # pid, parent, group: 13 lost its parent (adopted by 1) but kept the group.
    table = (
        "  1     0     1\n 10     1    10\n 11    10    10\n 12    11    10\n"
        " 13     1    10\n 20     1    20\n bad line\n"
    )
    assert loopback.watched_pids(10, table) == [10, 11, 12, 13]


@pytest.mark.parametrize(
    ("address", "loop"),
    [
        ("127.0.0.1:8200", True),
        ("[::1]:8200", True),
        ("localhost:8200", True),
        ("151.101.1.1:443", False),
        ("[2606:4700::1]:443", False),
        ("127.0.0.2:8200", True),
        ("[::ffff:127.0.0.1]:8200", True),
        ("example.com:443", False),
    ],
)
def test_only_this_machines_addresses_count_as_loopback(
    address: str, loop: bool
) -> None:
    assert loopback.is_loopback(address) is loop


def test_the_watch_says_what_it_saw_and_names_anything_outside() -> None:
    report = loopback.NetworkReport(samples=12, seconds=1.2)
    assert "no connection open at any sample" in loopback.describe(report, "x")
    report.connections = {"127.0.0.1:8200": {"claude"}}
    line = loopback.describe(report, "127.0.0.1:8200")
    assert "127.0.0.1:8200 (the proxy) only: loopback, nothing else" in line
    report.connections["151.101.1.1:443"] = {"node"}
    assert report.outside == ["151.101.1.1:443"] and not report.loopback_only
    assert "NOT loopback only: 151.101.1.1:443" in loopback.describe(report, "x")


def test_a_watch_that_could_not_look_is_never_loopback_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not loopback.NetworkReport().loopback_only  # no sample at all
    monkeypatch.setattr(loopback, "PS", "/no/such/ps")
    watch = loopback.NetworkWatch(os.getpid())
    watch.sample()
    assert (watch.report.samples, watch.report.failed) == (0, 1)
    assert not watch.report.loopback_only
    assert "WATCH FAILED: 1 samples could not be taken" in loopback.describe(
        watch.report, "x"
    )
    watch.report.connections = {"151.101.1.1:443": {"node"}}
    assert loopback.describe(watch.report, "x").endswith(
        "seen off this machine: 151.101.1.1:443"
    )
    assert not loopback.NetworkWatch(1).stop().samples  # never started: no error


def test_the_watch_samples_a_live_process_tree() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        start_new_session=True,
    )
    try:
        watch = loopback.NetworkWatch(child.pid, interval_s=0.05)
        watch.start()
        time.sleep(0.3)
        report = watch.stop()
    finally:
        child.kill()
        child.wait()
    assert report.samples >= 3 and report.connections == {}


# ------------------------------------------------- the scripts and the stop rules


def written(reply: scripted_replies.Reply) -> str:
    """The tool call or answer a scripted reply writes, markers removed.

    Args:
        reply: The reply.

    Returns:
        Its text after ``</think>``.
    """
    text = reply.text.split(scripted_replies.THINK_CLOSE, 1)[1]
    for marker in ("<|tool_call_start|>", "<|tool_call_end|>", "<|im_end|>"):
        text = text.replace(marker, "")
    return text


def test_the_solve_script_uses_each_harnesss_own_tool_names_and_ending() -> None:
    mini, claude, pi = (run_one.solve_script(h, TASK) for h in install.HARNESSES)
    assert written(mini[0]) == "[bash(command='sqlite3 data.db .tables')]"
    assert written(claude[0]) == "[Bash(command='sqlite3 data.db .tables')]"
    assert written(mini[-1]) == (
        "[bash(command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')]"
    )
    assert written(claude[-1]) == written(pi[-1]) == "General Practice"
    # LFM2.5 writes text arguments in single quotes, so the shell's are escaped.
    assert written(pi[2]) == (
        "[bash(command='echo \\'General Practice\\' > answer.txt')]"
    )


def test_the_mistakes_script_makes_the_untrained_models_mistakes() -> None:
    claude = [written(r) for r in run_one.mistakes_script("claude-code", TASK)]
    assert claude[0].startswith("[bash(") and claude[1].startswith("[Bash(")
    assert claude[2].startswith("[write(file='answer.txt'")
    (pi,) = run_one.mistakes_script("pi", TASK)
    assert written(pi) == "[read(path='data.db')]"
    mini = [written(r) for r in run_one.mistakes_script("mini-swe-agent", TASK)]
    assert mini[0].startswith("[sqlite3(") and "(" not in mini[1] + mini[2]


def test_scripts_run_only_on_the_scripted_engine() -> None:
    assert run_one.chosen_script("auto", "scripted") == "solve"
    assert run_one.chosen_script("auto", "model") == "none"
    assert run_one.chosen_script("none", "model") == "none"
    with pytest.raises(SystemExit, match="needs the scripted engine"):
        run_one.chosen_script("mistakes", "model")


def test_a_provider_name_must_be_plain(course_home: Path) -> None:
    assert run_one.provider_name("intercept") == "intercept"
    for bad in ("x/../../y", "", "1abc", "a b"):
        with pytest.raises(run_one.argparse.ArgumentTypeError):
            run_one.provider_name(bad)
    with pytest.raises(ValueError, match="is not inside"):
        run_one.prepare_run(course_home / "elsewhere", TASK)


def test_a_runs_folder_is_named_after_what_it_changed() -> None:
    plain = wire.Options()
    assert run_one.variant_name("claude-code", "solve", plain) == "claude-code"
    bare = wire.Options(claude_setup="bare")
    assert run_one.variant_name("claude-code", "none", bare) == "claude-code-bare"
    pi = wire.Options(pi_provider="openai", pi_api="default")
    assert run_one.variant_name("pi", "mistakes", pi) == (
        "pi-provider-openai-api-default-mistakes"
    )


def rollout(turns: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A rollout document as the proxy writes one, cut to what run_one reads.

    Args:
        turns: The turns.
        **extra: Other top-level fields.

    Returns:
        The document.
    """
    return {"turns": turns, "validation": [], **extra}


def process(exit_code: int | None = 0) -> run_one.Process:
    """A finished process with an empty network report.

    Args:
        exit_code: Its exit status (None: killed at the time limit).

    Returns:
        The process.
    """
    return run_one.Process(exit_code, 1.0, loopback.NetworkReport())


def test_facts_count_agent_calls_apart_from_side_calls(tmp_path: Path) -> None:
    document = rollout(
        [
            {"n_tools": 0, "n_prompt": 90},
            {"n_tools": 1, "n_prompt": 2_279},
            {"n_tools": 1, "n_prompt": 2_317},
            {"n_tools": 1, "n_prompt": 9, "discarded": True},
        ],
        metadata={"api_type": "anthropic"},
    )
    facts = run_one.read_facts("claude-code", document, tmp_path, process())
    assert (facts.agent_calls, facts.side_calls, facts.first_prompt) == (2, 1, 2_279)
    assert facts.dialect.startswith("anthropic")
    assert facts.stop == "a reply with no tool call: the harness's loop ends there"


def test_a_refused_prompt_is_read_from_the_proxys_finding(tmp_path: Path) -> None:
    finding = (
        "[WARN] context_budget_exhausted: upstream 400: {'error': {'message': "
        "'prompt is 30,895 tokens; the limit is 16,384'}}"
    )
    document = rollout(
        [{"n_tools": 4, "n_prompt": 1_504}],
        validation=[finding, "[INFO] something"],
        budget_stop_count=1,
        metadata={"api_type": "openai_chat"},
    )
    facts = run_one.read_facts("pi", document, tmp_path, process())
    assert facts.refused_tokens == 30_895 and facts.findings == [finding]
    assert facts.stop.startswith("the context window: the engine refused a 30,895")
    assert run_one.first_call(facts) == (
        "call 1 read 1,504 prompt tokens (system prompt, tools, task)"
    )
    facts.first_prompt = None
    assert run_one.first_call(facts).startswith("call 1 was refused: 30,895")


@pytest.mark.parametrize(
    ("status", "words"),
    [
        ("Submitted", "finishing command"),
        ("LimitsExceeded", "step_limit (8 calls)"),
        ("RepeatedFormatError", "max_consecutive_format_errors"),
        ("Odd", "ended with 'Odd'"),
    ],
)
def test_mini_swe_agents_own_exit_status_names_its_stop(
    status: str, words: str
) -> None:
    facts = run_one.Facts("openai_chat", 1, 0, 529, 0, None)
    assert words in run_one.stop_rule("mini-swe-agent", facts, process(), status)


def test_the_strongest_evidence_names_the_stop_first() -> None:
    facts = run_one.Facts("anthropic", 8, 0, 2_279, 1, None)
    assert run_one.stop_rule("claude-code", facts, process(None), "") == (
        "the time limit: the whole process group was killed"
    )
    assert "8-call budget" in run_one.stop_rule("claude-code", facts, process(), "")
    facts.budget_stops = 0
    assert "exited with status 1" in run_one.stop_rule("pi", facts, process(1), "")


def test_each_harnesss_own_log_is_read_in_its_own_format(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted", "model_stats": {"api_calls": 4}},
                "messages": [
                    {
                        "content": "Tool call error:\n\n<error>\nUnknown tool 'x'.\n"
                        "</error>\n\nHere is general guidance",
                        "extra": {"interrupt_type": "FormatError"},
                    }
                ],
            }
        )
    )
    assert run_one.read_mini_log(trajectory) == (
        "exit_status Submitted · api_calls 4",
        ["Unknown tool 'x'."],
        "Submitted",
    )
    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        "not json\n"
        + json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "is_error": True,
                            "content": "<tool_use_error>Error: No such tool "
                            "available: write</tool_use_error>",
                        }
                    ]
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "num_turns": 4,
                "stop_reason": "end_turn",
            }
        )
    )
    says, errors, _ = run_one.read_claude_log(stream)
    assert says == "result success · 4 turns · stop_reason end_turn"
    assert errors == ["Error: No such tool available: write"]
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"type": "tool_execution_start", "toolName": "read"})
        + "\n"
        + json.dumps({"type": "agent_end"})
    )
    assert run_one.read_pi_log(events)[0] == "agent_end reached · tools run: read"
    assert run_one.read_mini_log(tmp_path / "missing.json")[0] == "no trajectory saved"


def boxed_shell(root: Path, script: str) -> tuple[wire.Wiring, wire.RunFolders]:
    """A wiring that runs a shell script, for run_boxed's own tests.

    Args:
        root: A folder for the run (inside COURSE_HOME).
        script: The shell script.

    Returns:
        The wiring and the run's folders.
    """
    folders = run_folders(root)
    env = wire.base_environment(folders)
    wiring = wire.Wiring("test", ["/bin/sh", "-c", script], env, stdout_name="out.txt")
    return wiring, folders


def test_no_sandbox_means_no_run(
    course_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_box(*args: Any, **kwargs: Any) -> list[str]:
        raise run_one.SandboxError("not on this machine")

    monkeypatch.setattr(run_one, "sandbox_command", no_box)
    monkeypatch.setattr(run_one.subprocess, "Popen", lambda *a, **k: pytest.fail())
    wiring, folders = boxed_shell(course_home / "run", "echo hi")
    with pytest.raises(run_one.HarnessInstallError, match="no sandbox, so no run"):
        run_one.run_boxed(wiring, folders, 5)


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS sandbox-exec")
def test_the_time_limit_kills_the_whole_process_group(course_home: Path) -> None:
    marker = "mhrl-ch23-time-limit-test"
    script = f"(exec -a {marker} sleep 60) & sleep 61"  # the marked one is a child
    wiring, folders = boxed_shell(course_home / "run", script)
    started = time.perf_counter()
    process = run_one.run_boxed(wiring, folders, 1)
    assert process.exit_code is None and time.perf_counter() - started < 15
    assert process.network.samples >= 1
    left = subprocess.run(["pgrep", "-f", marker], capture_output=True, check=False)
    assert left.stdout == b""  # the background sleep went with the group


def test_a_connection_off_this_machine_fails_the_run(
    course_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    report = loopback.NetworkReport(samples=5, seconds=0.5)
    report.connections = {"151.101.1.1:443": {"node"}}
    session = capture_client.Session("s1", "tokens", "eval", 8)
    monkeypatch.setattr(run_one, "detect_engine", lambda c: ("http://e", "model", "x"))
    monkeypatch.setattr(run_one, "checked_version", lambda *a: "1.0.4")
    monkeypatch.setattr(
        run_one, "run_boxed", lambda *a: run_one.Process(0, 1.0, report)
    )
    client = run_one.CaptureClient
    monkeypatch.setattr(client, "open_session", lambda self, *a: session)
    monkeypatch.setattr(client, "rollout", lambda self, sid: rollout([]))
    monkeypatch.setattr(client, "close", lambda self, sid: True)
    monkeypatch.setattr(sys, "argv", ["run_one.py", "pi"])
    with pytest.raises(SystemExit) as stop:
        run_one.main()
    assert stop.value.code == 1
    assert "NOT loopback only: 151.101.1.1:443" in capsys.readouterr().out
    saved = course_home / "23-three-harnesses" / "runs" / "pi" / run_one.RUN_FILE
    assert json.loads(saved.read_text())["network"]["loopback_only"] is False


# ------------------------------------------------- the scripted engine


def test_a_reply_may_take_seconds_and_the_engine_checks_them() -> None:
    reply = scripted_engine.read_reply({"text": "ok", "seconds": 0.2})
    assert reply == scripted_replies.Reply("ok", True, 0.2)
    for bad in ({"text": "ok", "seconds": -1}, {"text": "ok", "seconds": True}):
        with pytest.raises(ValueError, match="seconds"):
            scripted_engine.read_reply(bad)
    model = scripted_engine.ScriptedModel()
    model.push([scripted_replies.Reply("Done.</think>ok<|im_end|>", True, 0.2)])
    started = time.perf_counter()
    sampled = model.sample(
        [], model.new_cache(), Sampling(max_tokens=64, temperature=0.0)
    )
    assert time.perf_counter() - started >= 0.2
    assert model.decode(sampled.token_ids).endswith("ok<|im_end|>")


@needs_tokenizer
def test_the_scripted_engine_counts_a_prompt_in_lfm25s_tokens() -> None:
    cutter = scripted_engine.LfmCutter(LFM_TOKENIZER_FILE)
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
    messages = [{"role": "user", "content": "How many patients live in Murcia?"}]
    text = cutter.render(messages, [tool])
    assert text.startswith("<|startoftext|><|im_start|>system\nList of tools: ")
    assert text.endswith("<|im_start|>assistant\n<think>")
    ids = cutter.encode(text)
    assert cutter.encode("<|im_end|>") == sorted(cutter.eos_ids)
    assert 50 < len(ids) < 150  # one tool and one question, not characters


def test_start_sh_offers_the_scripted_engine_and_this_chapters_folder() -> None:
    text = START_SH.read_text()
    assert 'STATE="$COURSE_HOME/23-three-harnesses"' in text
    assert "usage: $0 [start|scripted|proxy|stop|status|install]" in text
    assert "replay" not in text


# ------------------------------------------------- the recorded transcripts


def test_replay_prints_the_three_recorded_runs_exactly(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    replay.main()
    assert capsys.readouterr().out == REPLAY_GOLDEN.read_text(encoding="utf-8")


def test_the_recorded_runs_show_why_the_harnesses_differ() -> None:
    mini = replay.read_mini(replay.MINI_LOG)
    claude = replay.read_claude(replay.CLAUDE_LOG)
    pi = replay.read_pi(replay.PI_LOG)
    assert len(mini.calls) == 8 and "finishing command" in mini.calls[-1].note
    refused = [c.result for c in claude.calls if c.error]
    assert refused == [
        "Error: No such tool available: bash. Tool names are case-sensitive: "
        "call Bash instead.",
        "Error: No such tool available: bash. Tool names are case-sensitive: "
        "call Bash instead.",
        "Error: No such tool available: write",
    ]
    assert len(claude.calls) == 7 and "proxy's reply" in claude.ending
    assert [c.tool for c in pi.calls] == ["read"]
    assert replay.CUT_RE.search(pi.calls[0].result).group(2) == "36,864"


def test_the_transcripts_carry_no_personal_paths_or_names() -> None:
    for path in replay.TRANSCRIPTS.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text and "/private/var/" not in text, path.name
        assert Path.home().name not in text, path.name


def test_the_pi_copy_says_what_it_cut() -> None:
    text = replay.PI_LOG.read_text(encoding="utf-8")
    cuts = replay.CUT_RE.findall(text)
    assert len(cuts) == 6 and set(cuts) == {("36,464", "36,864")}


# ------------------------------------------------- end to end, the real harnesses


def free_port() -> int:
    """Ask the system for a port nobody listens on.

    Returns:
        The port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def live_engine(cutter: scripted_engine.Cutter | None) -> Iterator[int]:
    """Serve the scripted engine over real HTTP.

    Args:
        cutter: Its tokenizer; the scripted one when None.

    Yields:
        Its port.
    """
    app, _ = scripted_engine.create_app(cutter)
    sock = engine.claim_port(0)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("the scripted engine did not start within 10 s")
    try:
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10)


def run_start_sh(
    command: str, engine_port: int, proxy_port: int, home: Path
) -> subprocess.CompletedProcess[str]:
    """Run start.sh as a learner does, with its settings in the environment.

    Args:
        command: ``proxy`` or ``stop``.
        engine_port: ENGINE_PORT.
        proxy_port: PROXY_PORT.
        home: COURSE_HOME (logs and process ids go there).

    Returns:
        The finished process, its output captured.
    """
    env = {
        **os.environ,
        "ENGINE_PORT": str(engine_port),
        "PROXY_PORT": str(proxy_port),
        "COURSE_HOME": str(home),
        "PROXY_ENV": str(REAL_PROXY_ENV),
    }
    return subprocess.run(
        ["bash", str(START_SH), command],
        env=env,
        capture_output=True,
        text=True,
        timeout=START_TIMEOUT_S,
        check=False,
    )


@dataclass
class Stack:
    """A scripted engine, and the real proxy start.sh put in front of it.

    Attributes:
        proxy_url: The proxy's root.
        lfm_tokens: Whether prompts are counted in LFM2.5's tokens.
    """

    proxy_url: str
    lfm_tokens: bool


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The real proxy, started by start.sh, in front of a scripted engine.

    Args:
        tmp_path_factory: pytest's temporary folders.

    Yields:
        The stack, shared by the module's end-to-end tests.
    """
    if REAL_PROXY_ENV is None or not harnesses_on_disk():
        pytest.skip("needs the capture proxy and the three harnesses")
    home = tmp_path_factory.mktemp("stack")
    cutter = None
    if LFM_TOKENIZER_FILE is not None:
        cutter = scripted_engine.LfmCutter(LFM_TOKENIZER_FILE)
    with live_engine(cutter) as engine_port:
        proxy_port = free_port()
        started = run_start_sh("proxy", engine_port, proxy_port, home)
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            yield Stack(f"http://127.0.0.1:{proxy_port}", cutter is not None)
        finally:
            run_start_sh("stop", engine_port, proxy_port, home)


@pytest.fixture
def harness_installs(course_home: Path) -> None:
    """Point this test's COURSE_HOME at the real harness installs.

    Args:
        course_home: The test's temporary course home.
    """
    chapter = course_home / "23-three-harnesses"
    chapter.mkdir(parents=True)
    for name in install.HARNESSES:
        folder = install.PINS[name].folder
        (chapter / folder).symlink_to(REAL_INSTALLS / folder)


def run_harness(
    stack: Stack,
    harness: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    *flags: str,
) -> tuple[str, dict[str, Any]]:
    """Run ``run_one.py`` as from the command line.

    Args:
        stack: The proxy and its scripted engine.
        harness: The harness.
        monkeypatch: pytest's patcher.
        capsys: pytest's output capture.
        *flags: More arguments.

    Returns:
        What it printed, and its run.json.
    """
    argv = ["run_one.py", harness, "--proxy-url", stack.proxy_url, *flags]
    monkeypatch.setattr(sys, "argv", argv)
    run_one.main()
    out = capsys.readouterr().out
    kept = re.search(r"^kept\s+(\S+)$", out, re.MULTILINE)
    folder = Path(kept.group(1).replace("~", str(Path.home()), 1))
    return out, json.loads((folder / run_one.RUN_FILE).read_text())


def line(out: str, label: str) -> str:
    """Find the line a report starts with a label.

    Args:
        out: The report.
        label: Such as ``stop``.

    Returns:
        The rest of the line.
    """
    found = re.search(rf"^{label}\s+(.*)$", out, re.MULTILINE)
    assert found, f"no {label!r} line in:\n{out}"
    return found.group(1)


def assert_solved_inside_the_box(
    out: str, run: dict[str, Any], harness: str, stack: Stack
) -> None:
    """Check a solve run: pinned, through the proxy, correct, loopback only.

    Args:
        out: What run_one printed.
        run: Its run.json.
        harness: The harness.
        stack: The proxy it ran through.
    """
    pinned = install.PINS[harness].version
    assert line(out, "version").startswith(f"{pinned}, asked inside the sandbox")
    assert line(out, "calls").startswith("4 agent calls · 0 side calls · 0 answered")
    assert line(out, "answer") == "answer.txt 'General Practice' · correct"
    assert "loopback, nothing else" in line(out, "network")
    assert run["network"]["loopback_only"] and run["network"]["samples"] >= 3
    assert list(run["network"]["connections"]) == [
        run_one.proxy_address(stack.proxy_url)
    ]
    assert run["unused_replies"] == 0


def first_tokens(run: dict[str, Any]) -> int:
    """The first agent call's prompt tokens, from run.json.

    Args:
        run: The run.

    Returns:
        The count.
    """
    return int(run["facts"]["first_prompt"])


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_mini_swe_agent_solves_through_the_proxy_and_finishes_its_own_way(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, run = run_harness(stack, "mini-swe-agent", monkeypatch, capsys)
    assert_solved_inside_the_box(out, run, "mini-swe-agent", stack)
    assert line(out, "dialect").startswith("openai_chat")
    assert line(out, "stop") == (
        "mini-swe-agent's finishing command: "
        "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`"
    )
    assert line(out, "its log") == "exit_status Submitted · api_calls 4"
    if stack.lfm_tokens:
        assert 480 <= first_tokens(run) <= 580


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_claude_code_solves_through_the_proxy_speaking_anthropic(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, run = run_harness(stack, "claude-code", monkeypatch, capsys)
    assert_solved_inside_the_box(out, run, "claude-code", stack)
    assert line(out, "dialect").startswith("anthropic")
    assert line(out, "stop").startswith("a reply with no tool call")
    assert line(out, "its log").startswith("result success · 4 turns")
    if stack.lfm_tokens:  # the project's "about 2,300"
        assert 2_200 <= first_tokens(run) <= 2_400


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_pi_solves_through_the_proxy_speaking_openai_chat(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, run = run_harness(stack, "pi", monkeypatch, capsys)
    assert_solved_inside_the_box(out, run, "pi", stack)
    assert line(out, "dialect").startswith("openai_chat")
    assert line(out, "its log") == "agent_end reached · tools run: bash, bash, bash"
    if stack.lfm_tokens:
        assert 1_400 <= first_tokens(run) <= 1_600


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_the_untrained_models_mistakes_meet_each_harnesss_own_rule(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, run = run_harness(
        stack, "claude-code", monkeypatch, capsys, "--script", "mistakes"
    )
    assert run["facts"]["tool_errors"] == [
        "Error: No such tool available: bash. Tool names are case-sensitive: "
        "call Bash instead.",
        "Error: No such tool available: write",
    ]
    assert line(out, "answer") == "answer.txt none · no answer.txt"
    out, run = run_harness(stack, "pi", monkeypatch, capsys, "--script", "mistakes")
    assert run["facts"]["agent_calls"] == 1 and run["facts"]["budget_stops"] == 1
    assert run["facts"]["refused_tokens"] > wire.MAX_PROMPT_TOKENS
    assert line(out, "stop").startswith("the context window: the engine refused")
    assert line(out, "its log") == "agent_end reached · tools run: read"
    out, run = run_harness(
        stack, "mini-swe-agent", monkeypatch, capsys, "--script", "mistakes"
    )
    assert run["facts"]["tool_errors"][0] == (
        "Unknown tool 'sqlite3'.Missing 'command' argument in bash tool call."
    )
    assert line(out, "its log") == "exit_status RepeatedFormatError · api_calls 3"


@needs_stack
@needs_tokenizer
@pytest.mark.usefixtures("harness_installs")
def test_claude_codes_first_call_with_bare_and_with_every_tool(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, run = run_harness(
        stack, "claude-code", monkeypatch, capsys, "--claude-setup", "bare"
    )
    assert 480 <= first_tokens(run) <= 600  # the project's "about 530"
    assert line(out, "answer").endswith("· correct")
    out, run = run_harness(
        stack, "claude-code", monkeypatch, capsys, "--claude-setup", "default-tools"
    )
    # Every tool does not fit the 16,384-token window: call 1 is refused.
    assert run["facts"]["agent_calls"] == 0
    assert run["facts"]["refused_tokens"] > wire.MAX_PROMPT_TOKENS
    assert line(out, "first").startswith("call 1 was refused:")


@needs_stack
@pytest.mark.usefixtures("harness_installs")
def test_pis_api_field_not_its_providers_name_picks_the_dialect(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    out, _ = run_harness(stack, "pi", monkeypatch, capsys, "--pi-provider", "openai")
    assert line(out, "dialect").startswith("openai_chat")
    out, _ = run_harness(
        stack,
        "pi",
        monkeypatch,
        capsys,
        "--pi-provider",
        "openai",
        "--pi-api",
        "default",
    )
    assert line(out, "dialect").startswith("openai_responses")
    assert line(out, "answer").endswith("· correct")


def test_a_missing_harness_is_refused_before_any_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        run_one, "detect_engine", lambda client: ("http://e", "scripted", "scripted")
    )

    def no_session(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a session was opened")

    monkeypatch.setattr(run_one.CaptureClient, "open_session", no_session)
    monkeypatch.setattr(sys, "argv", ["run_one.py", "pi"])
    with pytest.raises(SystemExit, match="REFUSED · pi is not installed"):
        run_one.main()
