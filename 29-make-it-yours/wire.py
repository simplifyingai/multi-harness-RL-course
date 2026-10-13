# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Wire each harness to the capture proxy through its own settings, nothing else.

No harness is modified. Each is pointed at the proxy the way its own
documentation says to point it at any model provider, and the capture
session's id goes where it expects an API key, so every call it makes lands
in that session (chapter 20). This module writes those settings, as the
project does (``mhrl/harnesses/``):

* **mini-swe-agent 2.4.6** reads YAML. The chapter's ``mini_swe_agent.yaml``
  (the project's) holds everything fixed: one native ``bash`` tool, a fresh
  subshell per command, finishing with ``echo
  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``. A small overlay holds the rest:
  model ``openai/lfm2.5-2.6b`` (litellm's route to any OpenAI-style server),
  ``api_base`` the proxy's ``/v1``, ``api_key`` the session id, and
  ``step_limit`` 8.
* **Claude Code 2.1.290** reads environment variables and flags.
  ``ANTHROPIC_BASE_URL`` is the proxy's root (it adds ``/v1/messages``),
  ``ANTHROPIC_API_KEY`` the session id, its small-model setting the same
  model, a private ``CLAUDE_CONFIG_DIR`` and ``CLAUDE_CODE_TMPDIR``, the
  total-tokens note off, replies capped at 1,024 tokens, telemetry, error
  reports and updates off; ``-p``, ``--tools Bash``, no MCP servers, no
  settings files, no slash commands, no saved session.
* **Pi 1.0.4** reads a ``models.json``: a provider named ``intercept`` with
  ``api: openai-completions`` (a provider named ``openai`` is sent to
  OpenAI's other API, Responses), the proxy's ``/v1`` and the session id;
  a ``settings.json`` sizes compaction to the 16K window. Flags switch off
  everything that reads the machine's configuration, and ``--offline``.
* **Qwen Code 0.24.7**, the fourth, reads environment variables, as its
  documentation says for any server that speaks OpenAI's chat API:
  ``OPENAI_BASE_URL`` the proxy's ``/v1``, ``OPENAI_API_KEY`` the session
  id, ``OPENAI_MODEL`` the model. Flags: the instruction as its one-shot
  prompt, ``--bare`` (no settings files, skills, hooks or memory, and a
  shorter system prompt), ``--approval-mode yolo`` (the sandbox is the
  boundary, as ``bypassPermissions`` is for Claude Code), its log as
  ``stream-json``, and its agent and planning tools excluded. Its usage
  statistics, telemetry and the connection it opens early to the API are
  switched off: measured without them, it called a server of its own on
  port 443 while it ran.

Every harness gets an environment with nothing of yours in it: a private
``HOME``, ``TMPDIR`` and config folders inside the run's state folder.

``uv run wire.py`` writes all four for an example session under
``COURSE_HOME/29-make-it-yours/wired/`` and prints what each harness
reads. Only Python's standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from install import (
    CLAUDE,
    HARNESSES,
    MINI,
    PI,
    QWEN,
    SAFE_PATH,
    install_folder,
    program,
)
from lfm_files import MAX_CALLS, OUTPUT_CAP, chapter_home, short_path, tilde
from sandbox import DEFAULT_PROXY_PORT

HERE = Path(__file__).resolve().parent
MODEL_NAME = "lfm2.5-2.6b"  # the name the engine serves
MAX_PROMPT_TOKENS = 16_384  # the engine's limit (engine.py), the project's window
OPENAI_PATH = "/v1"  # mini-swe-agent and Pi speak OpenAI's chat API under it
REQUEST_TIMEOUT_S = 600  # the project's: up to four rollouts share one engine
REQUEST_TIMEOUT_MS = REQUEST_TIMEOUT_S * 1000
ENABLED, DISABLED = "1", "0"
ROLLOUT_TIMEOUT_S = 300  # the project's rollout_timeout_s
EXAMPLE_PROXY = "http://127.0.0.1:8200"
EXAMPLE_KEY = "s0123456789abcdef01234567"  # what a session id looks like
WIRED_FOLDER = "wired"

# mini-swe-agent
MINI_CONFIG = HERE / "mini_swe_agent.yaml"  # the project's, unchanged
MINI_OVERLAY = "mini_swe_agent.run.yaml"  # mini reads only .yaml; JSON is YAML
MINI_TRAJECTORY = "trajectory.json"
MINI_GLOBAL_DIR = "mini-swe-agent"
LITELLM_ROUTE = "openai/"  # litellm sends openai/<name> to any OpenAI-style server
WALL_TIME_MARGIN_S = 15  # mini stops itself, and saves its trajectory, first
MIN_WALL_TIME_S = 1
MODEL_ATTEMPTS = 3

# Claude Code
CLAUDE_CONFIG_FOLDER = "claude"
CLAUDE_TOOLS = "Bash"
CLAUDE_SETUPS = ("project", "bare", "default-tools")
BASH_OUTPUT_MAX_CHARS = 4_000
CLAUDE_FLAG_SETTINGS = {
    "attribution": {"commit": "", "pr": ""},
    "includeGitInstructions": False,
    "bashOutputMaxChars": BASH_OUTPUT_MAX_CHARS,
}

# Pi
PI_AGENT_FOLDER = "pi-agent"
PI_MODELS_FILE = "models.json"
PI_SETTINGS_FILE = "settings.json"
PI_PROVIDER = "intercept"  # not "openai": Pi sends that name to the Responses API
PI_API = "openai-completions"
PI_API_DEFAULT = "default"  # write no api field: Pi picks the provider's own
PI_APIS = (PI_API, PI_API_DEFAULT)
PI_TOOLS = "read,bash,edit,write"  # its built-in tools, named so none changes
KEEP_RECENT_TOKENS = 8_192

# Qwen Code
# Its tools for sub-agents, goals, skills, worktrees and messages: none is
# needed to answer a question, and each one's description is in every prompt.
# Never read_file or edit: Qwen Code turns an excluded tool into a rule that
# also refuses shell commands of the same kind (measured: with read_file
# excluded, `head -3 data.csv` through run_shell_command was "denied by
# permission rules. Matching deny rule: \"read_file\"").
QWEN_EXCLUDED = (
    "agent",
    "list_agents",
    "send_message",
    "task_stop",
    "get_goal",
    "update_goal",
    "tool_call",
    "tool_search",
    "skill",
    "report_findings",
    "enter_worktree",
    "exit_worktree",
    "record_artifact",
    "notebook_edit",
)
QWEN_OFF = "false"

# What each harness writes as it runs, and where its stdout is kept.
STDOUT_FILES = {
    MINI: "output.txt",
    CLAUDE: "stream.jsonl",
    PI: "events.jsonl",
    QWEN: "qwen-stream.jsonl",
}


@dataclass(frozen=True)
class RunFolders:
    """The folders of one run.

    Attributes:
        task: The task folder: the harness's working folder (data.csv,
            answer.txt).
        state: The run's private folder: HOME, TMPDIR, config files.
    """

    task: Path
    state: Path

    @property
    def home(self) -> Path:
        """The harness's HOME."""
        return self.state / "home"

    @property
    def tmp(self) -> Path:
        """The harness's TMPDIR."""
        return self.state / "tmp"

    def make(self) -> None:
        """Create the folders (the task folder may already hold data.csv)."""
        for folder in (self.task, self.home, self.tmp):
            folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Options:
    """The few settings a learner may change.

    Attributes:
        claude_setup: ``project`` (``--tools Bash``), ``bare`` (also
            ``--bare``) or ``default-tools`` (no ``--tools``: every tool).
        pi_provider: The provider name in Pi's models.json.
        pi_api: ``openai-completions``, or ``default`` to write no ``api``
            field and let Pi choose (its built-in ``openai`` provider chooses
            the Responses API).
        timeout_s: The run's time limit (mini-swe-agent stops 15 s before).
    """

    claude_setup: str = "project"
    pi_provider: str = PI_PROVIDER
    pi_api: str = PI_API
    timeout_s: int = ROLLOUT_TIMEOUT_S


@dataclass
class Wiring:
    """Everything one harness needs to run once, against one session.

    Attributes:
        harness: Its name.
        argv: Its command line (before the sandbox wraps it).
        env: Its whole environment.
        files: Each settings file it reads, with its content.
        settings: The lines that point it at the proxy, for printing.
        own_log: The log the harness keeps of the run.
        stdout_name: The file its standard output goes to.
        proxy_port: The proxy's port: the one loopback port the sandbox lets
            the run connect to.
    """

    harness: str
    argv: list[str]
    env: dict[str, str]
    files: dict[Path, str] = field(default_factory=dict)
    settings: list[str] = field(default_factory=list)
    own_log: Path | None = None
    stdout_name: str = ""
    proxy_port: int = DEFAULT_PROXY_PORT

    def write(self) -> None:
        """Write the settings files."""
        for path, text in self.files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def base_environment(folders: RunFolders) -> dict[str, str]:
    """The environment every harness starts from: nothing inherited.

    No credential in your own environment can reach a shell the model drives.

    Args:
        folders: The run's folders.

    Returns:
        Environment variables.
    """
    user = os.environ.get("USER", "learner")
    home = folders.home
    return {
        "PATH": SAFE_PATH,
        "HOME": str(home),
        "TMPDIR": str(folders.tmp),
        "TMPPREFIX": str(folders.tmp / "zsh"),  # zsh ignores TMPDIR for heredocs
        "USER": user,
        "LOGNAME": user,
        "SHELL": "/bin/bash",
        "LANG": "en_US.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": ENABLED,
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
    }


def mini_overlay(base_url: str, api_key: str, timeout_s: int) -> dict[str, Any]:
    """mini-swe-agent's per-run settings, merged over mini_swe_agent.yaml.

    Args:
        base_url: The proxy's root.
        api_key: The capture session's id.
        timeout_s: The run's time limit.

    Returns:
        The overlay.
    """
    return {
        "model": {
            "model_name": LITELLM_ROUTE + MODEL_NAME,
            "model_kwargs": {"api_base": base_url + OPENAI_PATH, "api_key": api_key},
        },
        "agent": {
            "step_limit": MAX_CALLS,
            "wall_time_limit_seconds": max(
                MIN_WALL_TIME_S, timeout_s - WALL_TIME_MARGIN_S
            ),
        },
    }


def wire_mini(
    folders: RunFolders, instruction: str, base_url: str, api_key: str, options: Options
) -> Wiring:
    """Wire mini-swe-agent: a YAML overlay and its own environment switches.

    Args:
        folders: The run's folders.
        instruction: The task.
        base_url: The proxy's root.
        api_key: The capture session's id.
        options: The run's options.

    Returns:
        The wiring.
    """
    overlay = folders.state / MINI_OVERLAY
    trajectory = folders.state / MINI_TRAJECTORY
    # A copy in the run's own folder: the box denies reading this repository.
    config = folders.state / MINI_CONFIG.name
    settings = mini_overlay(base_url, api_key, options.timeout_s)
    argv = [
        *program(MINI),
        "-c", str(config),
        "-c", str(overlay),
        "-t", instruction,
        "--yolo",
        "--exit-immediately",
        "-o", str(trajectory),
    ]  # fmt: skip
    env = base_environment(folders) | {
        # Skip the first-run questions and the banner.
        "MSWEA_CONFIGURED": "true",
        "MSWEA_SILENT_STARTUP": ENABLED,
        "MSWEA_COST_TRACKING": "ignore_errors",  # a local model has no price
        "MSWEA_GLOBAL_CONFIG_DIR": str(folders.state / MINI_GLOBAL_DIR),
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(MODEL_ATTEMPTS),
        # litellm would otherwise fetch its price list from GitHub at start.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONDONTWRITEBYTECODE": ENABLED,
        "PYTHONUNBUFFERED": ENABLED,
    }
    model = settings["model"]
    return Wiring(
        harness=MINI,
        argv=argv,
        env=env,
        files={
            config: MINI_CONFIG.read_text(encoding="utf-8"),
            overlay: json.dumps(settings, indent=2) + "\n",
        },
        settings=[
            f"model.model_name             {model['model_name']}",
            f"model.model_kwargs.api_base  {model['model_kwargs']['api_base']}",
            f"model.model_kwargs.api_key   {api_key}  (the session id)",
            f"agent.step_limit             {settings['agent']['step_limit']}",
        ],
        own_log=trajectory,
        stdout_name=STDOUT_FILES[MINI],
    )


def claude_flags(instruction: str, setup: str) -> list[str]:
    """Claude Code's flags: the project's, or a variant to measure.

    Args:
        instruction: The task, ``-p``'s argument.
        setup: One of ``CLAUDE_SETUPS``.

    Returns:
        The flags.

    Raises:
        ValueError: For an unknown setup.
    """
    if setup not in CLAUDE_SETUPS:
        raise ValueError(f"unknown Claude Code setup {setup!r}: {CLAUDE_SETUPS}")
    tools = [] if setup == "default-tools" else ["--tools", CLAUDE_TOOLS]
    flags = [
        "-p", instruction,
        *tools,
        "--strict-mcp-config",
        "--setting-sources", "",
        "--settings", json.dumps(CLAUDE_FLAG_SETTINGS),
        "--permission-mode", "bypassPermissions",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--model", MODEL_NAME,
        "--output-format", "stream-json",
        "--verbose",
    ]  # fmt: skip
    if setup == "bare":
        flags.append("--bare")
    return flags


def wire_claude(
    folders: RunFolders, instruction: str, base_url: str, api_key: str, options: Options
) -> Wiring:
    """Wire Claude Code: environment variables and flags, no files.

    Args:
        folders: The run's folders.
        instruction: The task.
        base_url: The proxy's root (Claude Code adds ``/v1/messages``).
        api_key: The capture session's id.
        options: The run's options.

    Returns:
        The wiring.
    """
    proxy = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_MODEL": MODEL_NAME,
        # Any "small fast model" call goes to the same model.
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL_NAME,
    }
    env = (
        base_environment(folders)
        | proxy
        | {
            "CLAUDE_CONFIG_DIR": str(folders.state / CLAUDE_CONFIG_FOLDER),
            # Its temp root is this or a fixed /tmp folder the sandbox denies.
            "CLAUDE_CODE_TMPDIR": str(folders.tmp),
            # Its per-call <total_tokens> note would split every rollout.
            "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(OUTPUT_CAP),
            "API_TIMEOUT_MS": str(REQUEST_TIMEOUT_MS),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": ENABLED,
            "DISABLE_TELEMETRY": ENABLED,
            "DISABLE_ERROR_REPORTING": ENABLED,
            "DISABLE_AUTOUPDATER": ENABLED,
        }
    )
    tools = "every tool" if options.claude_setup == "default-tools" else "Bash"
    return Wiring(
        harness=CLAUDE,
        argv=[*program(CLAUDE), *claude_flags(instruction, options.claude_setup)],
        env=env,
        settings=[
            f"ANTHROPIC_BASE_URL             {base_url}",
            f"ANTHROPIC_API_KEY              {api_key}  (the session id)",
            f"ANTHROPIC_DEFAULT_HAIKU_MODEL  {MODEL_NAME}  (side calls too)",
            f"tools                          {tools}"
            + ("  (+ --bare)" if options.claude_setup == "bare" else ""),
        ],
        own_log=None,  # its own log is its standard output, stream.jsonl
        stdout_name=STDOUT_FILES[CLAUDE],
    )


def pi_models(
    base_url: str, api_key: str, provider: str, api: str = PI_API
) -> dict[str, Any]:
    """Pi's models.json: one provider, the proxy, one model.

    Args:
        base_url: The proxy's root.
        api_key: The capture session's id.
        provider: The provider's name.
        api: The API it speaks, or ``default`` to leave the field out.

    Returns:
        The file's content.
    """
    entry: dict[str, Any] = {"baseUrl": base_url + OPENAI_PATH}
    if api != PI_API_DEFAULT:
        entry["api"] = api
    entry["apiKey"] = api_key
    entry["models"] = [
        {
            "id": MODEL_NAME,
            "contextWindow": MAX_PROMPT_TOKENS,
            "maxTokens": OUTPUT_CAP,
            "input": ["text"],
        }
    ]
    return {"providers": {provider: entry}}


def pi_settings() -> dict[str, Any]:
    """Pi's settings.json: timeouts, and compaction sized to the window.

    Returns:
        The file's content.
    """
    return {
        "httpIdleTimeoutMs": REQUEST_TIMEOUT_MS,
        "retry": {"provider": {"timeoutMs": REQUEST_TIMEOUT_MS}},
        # Pi compacts when the context passes the window minus this reserve.
        "compaction": {
            "reserveTokens": OUTPUT_CAP,
            "keepRecentTokens": KEEP_RECENT_TOKENS,
        },
        "quietStartup": True,
    }


def wire_pi(
    folders: RunFolders, instruction: str, base_url: str, api_key: str, options: Options
) -> Wiring:
    """Wire Pi: models.json, settings.json and flags.

    Args:
        folders: The run's folders.
        instruction: The task.
        base_url: The proxy's root.
        api_key: The capture session's id.
        options: The run's options.

    Returns:
        The wiring.
    """
    agent = folders.state / PI_AGENT_FOLDER
    provider = options.pi_provider
    argv = [
        *program(PI),
        "-p", instruction,
        "--provider", provider,
        "--model", MODEL_NAME,
        "--tools", PI_TOOLS,
        "--mode", "json",  # the log then holds every event
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-mcp",
        "--no-approve",
        "--offline",
    ]  # fmt: skip
    env = base_environment(folders) | {
        "PI_CODING_AGENT_DIR": str(agent),
        "PI_OFFLINE": ENABLED,
        "PI_SKIP_VERSION_CHECK": ENABLED,
        "PI_TELEMETRY": DISABLED,
    }
    models = pi_models(base_url, api_key, provider, options.pi_api)
    entry = models["providers"][provider]
    api = entry.get("api", "not set: Pi chooses")
    return Wiring(
        harness=PI,
        argv=argv,
        env=env,
        files={
            agent / PI_MODELS_FILE: json.dumps(models, indent=2) + "\n",
            agent / PI_SETTINGS_FILE: json.dumps(pi_settings(), indent=2) + "\n",
        },
        settings=[
            f"provider  {provider}  (api {api})",
            f"baseUrl   {entry['baseUrl']}",
            f"apiKey    {api_key}  (the session id)",
            f"window    {MAX_PROMPT_TOKENS:,} tokens, reserve {OUTPUT_CAP:,}",
        ],
        own_log=None,  # its own log is its standard output, events.jsonl
        stdout_name=STDOUT_FILES[PI],
    )


def wire_qwen(
    folders: RunFolders, instruction: str, base_url: str, api_key: str, options: Options
) -> Wiring:
    """Wire Qwen Code: three environment variables, as its documentation says.

    Qwen Code speaks OpenAI's chat API to any server given a base URL, a key
    and a model name. ``--bare`` makes it read no settings file at all (so
    its requests ask for 32,000 output tokens, and the proxy and the engine
    cap every reply at 1,024 anyway).

    Args:
        folders: The run's folders.
        instruction: The task, its one-shot prompt.
        base_url: The proxy's root.
        api_key: The capture session's id.
        options: The run's options (none is Qwen Code's own).

    Returns:
        The wiring.
    """
    del options  # every Qwen Code run is wired the same way
    proxy = {
        "OPENAI_BASE_URL": base_url + OPENAI_PATH,
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL": MODEL_NAME,
    }
    env = (
        base_environment(folders)
        | proxy
        | {
            # Off: usage statistics (on by default), telemetry, the early
            # connection to the API, and the side call that titles a session.
            "QWEN_USAGE_STATISTICS_ENABLED": QWEN_OFF,
            "QWEN_TELEMETRY_ENABLED": QWEN_OFF,
            "QWEN_CODE_DISABLE_PRECONNECT": ENABLED,
            "QWEN_DISABLE_AUTO_TITLE": ENABLED,
            # Its notice that yolo runs with no sandbox of its own: ours is on.
            "QWEN_CODE_SUPPRESS_YOLO_WARNING": ENABLED,
        }
    )
    argv = [
        *program(QWEN),
        instruction,  # its one-shot prompt (-p is deprecated in 0.24.7)
        "--bare",
        "--approval-mode", "yolo",
        "--output-format", "stream-json",
        "--exclude-tools", ",".join(QWEN_EXCLUDED),
        "--model", MODEL_NAME,
    ]  # fmt: skip
    return Wiring(
        harness=QWEN,
        argv=argv,
        env=env,
        settings=[
            f"OPENAI_BASE_URL  {proxy['OPENAI_BASE_URL']}",
            f"OPENAI_API_KEY   {api_key}  (the session id)",
            f"OPENAI_MODEL     {MODEL_NAME}",
            "tools            edit, read_file, run_shell_command (--bare, "
            f"{len(QWEN_EXCLUDED)} excluded)",
        ],
        own_log=None,  # its own log is its standard output, qwen-stream.jsonl
        stdout_name=STDOUT_FILES[QWEN],
    )


WIRERS = {MINI: wire_mini, CLAUDE: wire_claude, PI: wire_pi, QWEN: wire_qwen}


def wire(
    harness: str,
    folders: RunFolders,
    instruction: str,
    base_url: str,
    api_key: str,
    options: Options | None = None,
) -> Wiring:
    """Write nothing yet: work out how to point one harness at the proxy.

    Args:
        harness: One of ``HARNESSES``.
        folders: The run's folders.
        instruction: The task.
        base_url: The proxy's root, without a trailing slash.
        api_key: The capture session's id.
        options: The run's options (the defaults when None).

    Returns:
        The wiring; ``Wiring.write`` writes its files.

    Raises:
        ValueError: For an unknown harness.
    """
    if harness not in WIRERS:
        raise ValueError(f"unknown harness {harness!r}: choose from {HARNESSES}")
    wirer = WIRERS[harness]
    root = base_url.rstrip("/")
    wiring = wirer(folders, instruction, root, api_key, options or Options())
    wiring.proxy_port = urlsplit(root).port or DEFAULT_PROXY_PORT
    return wiring


def show(wiring: Wiring, folders: RunFolders) -> None:
    """Print what a harness reads: its proxy settings, files and command.

    Args:
        wiring: The wiring.
        folders: The run's folders (to tell its own variables from the base).
    """
    print(f"━━ {wiring.harness}")
    for line in wiring.settings:
        print(f"  {line}")
    for path in wiring.files:
        print(f"  file      {short_path(path)}")
    # main() wires with "<the task>" as the instruction, so it prints as such.
    command = shlex.join(wiring.argv).replace(str(HERE) + "/", "")
    print(f"  command   {tilde(command)}")
    ours = sorted(set(wiring.env) - set(base_environment(folders)))
    print(f"  env       {len(wiring.env)} variables; its own: {', '.join(ours)}")


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("harness", nargs="*", help=f"any of {', '.join(HARNESSES)}")
    parser.add_argument("--proxy-url", default=EXAMPLE_PROXY)
    parser.add_argument("--api-key", default=EXAMPLE_KEY)
    args = parser.parse_args()
    unknown = sorted(set(args.harness) - set(HARNESSES))
    if unknown:
        parser.error(f"unknown harness {', '.join(unknown)}: choose {HARNESSES}")
    return args


def main() -> None:
    """Write each harness's settings for an example session and print them."""
    args = parse_args()
    print(f"proxy {args.proxy_url} · session id (the API key) {args.api_key}")
    for harness in args.harness or HARNESSES:
        root = chapter_home() / WIRED_FOLDER / harness
        folders = RunFolders(task=root / "task", state=root / "state")
        folders.make()
        wiring = wire(harness, folders, "<the task>", args.proxy_url, args.api_key)
        wiring.write()
        show(wiring, folders)
    print(f"installed in {short_path(install_folder(MINI).parent)}")


if __name__ == "__main__":
    main()
