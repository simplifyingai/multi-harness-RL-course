# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One real GRPO step through the project's own code, printed at every hand-off.

Every earlier chapter rebuilt one piece. This runs the project itself: its
``train_grpo`` (``mhrl/training/grpo.py``) for one step, one task, one
harness (Claude Code) and a group of 4 rollouts, against the project's own
engine and capture proxy. ``hooks.py`` listens at each hand-off and prints
it: the fresh folder; the session id as the API key; Claude Code's command
and environment (the key redacted); each call the proxy captured; the grade
and the reward; the group's rewards and advantages; one row with its mask
and the engine's logprobs; the importance ratios and the loss; the AdamW
update; the adapter file; and the POST that makes the engine sample with it.

A group whose 4 rewards are equal has no contrast and teaches nothing
(chapter 5): ``train_grpo`` skips it and makes no update. Then this draws
the next task in ``train_grpo``'s own order and tries again, up to
``--max-draws`` times.

It imports ``mhrl``, so it runs with the project's Python: ``uv run
one_step.py --project PATH`` (or ``MHRL_PROJECT``) re-runs itself with
``PATH/.venv/bin/python``. Start the stack first (``./stack.sh start``).
The project's settings come from ``MHRL_*`` environment variables
(``mhrl/config.py``): ``MHRL_ENGINE_PORT`` and ``MHRL_PROXY_PORT`` say where
the engine and the proxy listen, and ``MHRL_RUNS_DIR`` (set here to
``COURSE_HOME/27-how-it-combines/runs`` unless you set it) holds the task
folders. The step's records and adapter go to
``COURSE_HOME/27-how-it-combines/one_step/draw<N>-<task>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hooks import INDENT, Tracer, get_json
from paths import (
    PROJECT_PYTHON,
    ProjectNotFoundError,
    chapter_home,
    check_project,
    named_project,
    short_path,
    tilde,
)

SCRIPT = "one_step.py"
HARNESS = "claude-code"
GROUP_SIZE = 4
TASKS_PER_STEP = 1
STEPS = 1
TASK_SEED = 0  # train_grpo's default seed for its task order
DEFAULT_MAX_DRAWS = 5
RUNS_ENV = "MHRL_RUNS_DIR"
# Set before re-running with the project's Python, so a second failure to
# import mhrl stops instead of re-running forever.
REEXEC_ENV = "MHRL_COURSE_REEXEC"
RUNS_FOLDER = "runs"
STEP_FOLDER = "one_step"
BASE = "base"
EXIT_FAILED = 1
EXIT_NO_CONTRAST = 2
# The settings shown, with the environment variable that overrides each.
SHOWN_SETTINGS = (
    ("engine_port", "MHRL_ENGINE_PORT"),
    ("proxy_port", "MHRL_PROXY_PORT"),
    ("runs_dir", "MHRL_RUNS_DIR"),
    ("model_path", "MHRL_MODEL_PATH"),
    ("max_model_calls", "MHRL_MAX_MODEL_CALLS"),
    ("train_temperature", "MHRL_TRAIN_TEMPERATURE"),
    ("max_completion_tokens", "MHRL_MAX_COMPLETION_TOKENS"),
    ("token_budget", "MHRL_TOKEN_BUDGET"),
    ("length_guard_tokens", "MHRL_LENGTH_GUARD_TOKENS"),
    ("concurrency", "MHRL_CONCURRENCY"),
)


class StackNotReadyError(Exception):
    """The engine, the proxy or Claude Code is not what the step needs."""


@dataclass(frozen=True)
class Project:
    """The parts of the project the step uses, imported once.

    Attributes:
        settings: ``mhrl.config.get_settings()``.
        grpo: ``mhrl.training.grpo``.
        rollout: ``mhrl.rollout``.
        tasks: The training tasks, in ``train_grpo``'s order.
        cli_defaults: ``mhrl.cli.CliDefaults``.
        get_harness: ``mhrl.harnesses.base.get_harness``.
        not_installed: ``mhrl.harnesses.errors.HarnessNotInstalledError``.
        failures: What a step can fail with: the harnesses' errors, the
            HTTP client's, and the operating system's.
    """

    settings: Any
    grpo: Any
    rollout: Any
    tasks: list[Any]
    cli_defaults: Any
    get_harness: Callable[[str], Any]
    not_installed: type[Exception]
    failures: tuple[type[Exception], ...] = ()


def project_of_module(module: Any) -> Path:
    """Return the checkout an imported ``mhrl`` package belongs to.

    Args:
        module: The ``mhrl`` package.

    Returns:
        The folder holding it.
    """
    package = Path(module.__file__).parent if module.__file__ else None
    package = package or Path(next(iter(module.__path__)))
    return package.resolve().parent


def ensure_project_python(
    flag: str | None, arguments: list[str], say: Callable[[str], None]
) -> None:
    """Re-run this script with the project's Python unless ``mhrl`` imports.

    Args:
        flag: The ``--project`` value, or None.
        arguments: This run's arguments, handed to the re-run unchanged.
        say: Where the note goes.

    Raises:
        SystemExit: If ``mhrl`` cannot be imported and no checkout with an
            environment is named.
    """
    project = named_project(flag)
    try:
        import mhrl
    except ImportError:
        mhrl = None
    if mhrl is not None and (project is None or project_of_module(mhrl) == project):
        return
    if os.environ.get(REEXEC_ENV):
        found = short_path(project_of_module(mhrl)) if mhrl else "no mhrl at all"
        raise SystemExit(
            f"{SCRIPT}: the project's Python still cannot import its mhrl "
            f"(it found {found}); run `uv sync` in the project"
        )
    if project is None:
        raise SystemExit(
            f"{SCRIPT}: mhrl is not importable here; name the project's checkout "
            "with --project PATH or MHRL_PROJECT"
        )
    try:
        check_project(project)
    except ProjectNotFoundError as error:
        raise SystemExit(f"{SCRIPT}: {error}") from error
    python = project / PROJECT_PYTHON
    if not python.is_file():
        raise SystemExit(
            f"{SCRIPT}: {short_path(python)} is missing; run `uv sync` in the project"
        )
    say(f"{SCRIPT} imports the project, so it runs with {short_path(python)}")
    sys.stdout.flush()
    os.environ[REEXEC_ENV] = "1"
    os.execv(python, [str(python), str(Path(__file__).resolve()), *arguments])


def train_order(tasks: list[Any], seed: int = TASK_SEED) -> list[Any]:
    """Put the training tasks in the order ``train_grpo`` visits them.

    ``python -m mhrl.cli grpo`` hands ``train_grpo`` every training task in
    file order, and ``train_grpo`` shuffles them once with ``random.Random``
    and its seed (0), then takes them in turn.

    Args:
        tasks: The training tasks, in file order.
        seed: The shuffle seed.

    Returns:
        The tasks in ``train_grpo``'s order.
    """
    order = list(tasks)
    random.Random(seed).shuffle(order)
    return order


def import_project() -> Project:
    """Import the project's modules and settings (with ``MHRL_RUNS_DIR`` set).

    Returns:
        The parts the step uses.
    """
    os.environ.setdefault(RUNS_ENV, str(chapter_home() / RUNS_FOLDER))
    import httpx
    from mhrl import cli, rollout
    from mhrl.config import get_settings
    from mhrl.harnesses.base import get_harness
    from mhrl.harnesses.errors import HarnessError, HarnessNotInstalledError
    from mhrl.tasks.core import load_tasks
    from mhrl.training import grpo

    return Project(
        settings=get_settings(),
        grpo=grpo,
        rollout=rollout,
        tasks=train_order(cli.balanced(load_tasks("train"), None)),
        cli_defaults=cli.CliDefaults,
        get_harness=get_harness,
        not_installed=HarnessNotInstalledError,
        failures=(HarnessError, httpx.HTTPError, OSError, ValueError),
    )


def settings_lines(settings: Any, environ: dict[str, str]) -> list[str]:
    """Print the settings the step runs with, and where each came from.

    Args:
        settings: The project's ``Settings``.
        environ: The environment (``MHRL_*`` variables override defaults).

    Returns:
        The lines.
    """
    lines = ["settings  mhrl/config.py Settings; an MHRL_* variable overrides each"]
    for name, variable in SHOWN_SETTINGS:
        value = getattr(settings, name)
        source = f"{variable}={tilde(environ[variable])}" if variable in environ else ""
        shown = tilde(str(value))
        lines.append(f"{INDENT}{name:<22} {shown:<44} {source or 'default'}")
    return lines


def check_stack(project: Project) -> list[str]:
    """Check the engine, the proxy and Claude Code before any rollout.

    The engine must be the project's (its ``/health`` names an adapter), the
    proxy must forward to that same engine (or a new adapter would go to one
    engine while another samples), and the pinned Claude Code must be there.

    Args:
        project: The imported project.

    Returns:
        What was found, as lines.

    Raises:
        StackNotReadyError: If any of the three is wrong.
    """
    settings = project.settings
    try:
        engine = get_json(f"{settings.engine_url}/health")
        proxy = get_json(f"{settings.proxy_url}/health")
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        raise StackNotReadyError(
            f"the engine or the proxy is not answering ({error}); start them "
            "with ./stack.sh start"
        ) from error
    if "adapter" not in engine:
        raise StackNotReadyError(
            f"{settings.engine_url} is not the project's engine: /health says {engine}"
        )
    upstream = str(proxy.get("upstream", "")).rstrip("/")
    if upstream != settings.engine_url:
        raise StackNotReadyError(
            f"the proxy forwards to {upstream}, not to the engine at "
            f"{settings.engine_url}"
        )
    harness = project.get_harness(HARNESS)
    try:
        harness.check_installed()
    except project.not_installed as error:
        raise StackNotReadyError(str(error)) from error
    return [
        f"engine    {settings.engine_url} · adapter {tilde(str(engine['adapter']))!r}",
        f"proxy     {settings.proxy_url} · capture level: {proxy.get('capture_level')}"
        f" · forwards to {upstream}",
        f"harness   Claude Code {harness.pinned_version} at "
        f"{short_path(Path(harness.binary))}",
    ]


def read_metrics(folder: Path) -> dict[str, Any]:
    """Read the last line ``train_grpo`` wrote to its metrics file.

    Args:
        folder: The step's output folder.

    Returns:
        That line, or an empty dict when there is none.
    """
    path = folder / "metrics.jsonl"
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def metrics_lines(metrics: dict[str, Any]) -> list[str]:
    """Print the step's own record, as ``train_grpo`` wrote it.

    Args:
        metrics: Its last metrics line.

    Returns:
        The lines.
    """
    if not metrics:
        return ["metrics   none written"]
    ratio = metrics.get("importance_ratio_mean")
    loss = metrics.get("loss_per_token")
    return [
        f"metrics   train_grpo's record: rows {metrics.get('train_rows')} · "
        f"tokens {metrics.get('train_tokens')} · rows over the cap "
        f"{metrics.get('skipped_long_rows')} · untrainable rollouts "
        f"{metrics.get('untrainable_rollouts')}",
        f"{INDENT}ratio {'n/a' if ratio is None else f'{ratio:.4f}'} · loss per "
        f"token {'n/a' if loss is None else f'{loss:+.4f}'} · tokens per call "
        f"{metrics.get('tokens_per_call', 0):.0f} · step "
        f"{metrics.get('step_seconds')} s · peak {metrics.get('peak_gb')} GB",
    ]


def start_adapter(value: str) -> Path | None:
    """Turn ``--init`` into an adapter path (None for the base model).

    Args:
        value: ``base`` or a path.

    Returns:
        The adapter, or None.

    Raises:
        SystemExit: If the file does not exist.
    """
    if value == BASE:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{SCRIPT}: --init {value}: no such adapter file")
    return path


def run_draws(
    project: Project,
    tracer: Tracer,
    init: Path | None,
    max_draws: int,
    say: Callable[[str], None],
) -> int:
    """Run ``train_grpo`` one task at a time until a group has contrast.

    Args:
        project: The imported project.
        tracer: The hooks.
        init: The adapter to start from (None for the base model).
        max_draws: How many tasks to try.
        say: Where each line goes.

    Returns:
        0 after an update, 1 if the length guard stopped the step, 2 if no
        group had contrast.
    """
    learning_rate = project.cli_defaults.GRPO_LR
    for draw, task in enumerate(project.tasks[:max_draws], start=1):
        out_dir = chapter_home() / STEP_FOLDER / f"draw{draw}-{task.task_id}"
        say("")
        say(f"draw {draw} · {task.task_id} ({task.difficulty}) · {task.question}")
        say(
            f"{INDENT}train_grpo(steps={STEPS}, tasks_per_step={TASKS_PER_STEP}, "
            f"group_size={GROUP_SIZE}, harnesses=[{HARNESS!r}], "
            f"learning_rate={learning_rate:g})"
        )
        tracer.start_draw()
        try:
            project.grpo.train_grpo(
                project.settings,
                [task],
                [HARNESS],
                steps=STEPS,
                tasks_per_step=TASKS_PER_STEP,
                group_size=GROUP_SIZE,
                learning_rate=learning_rate,
                out_dir=out_dir,
                init_adapter=init,
                max_tokens=project.cli_defaults.MAX_TRAIN_TOKENS,
                seed=TASK_SEED,
            )
        except RuntimeError as error:  # the guard tripped with nothing to keep
            say(tracer.redact(f"stopped   {error}"))
            return EXIT_FAILED
        metrics = read_metrics(out_dir)
        for line in metrics_lines(metrics):
            say(line)
        if metrics.get("stopped") == "length_guard":
            say("stopped   the length guard: replies ballooned (chapter 26)")
            return EXIT_FAILED
        if tracer.outcome.updated:
            say(f"output    {short_path(out_dir)}")
            say(
                f"{INDENT}the next call the proxy forwards is sampled with "
                f"{tilde(tracer.outcome.engine_adapter)}"
            )
            return 0
        say("no update: drawing the next task in train_grpo's order")
    say(f"no group had contrast in {max_draws} draws; no update was made")
    return EXIT_NO_CONTRAST


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: The arguments, or None for ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", help="the project's checkout (or MHRL_PROJECT)")
    parser.add_argument(
        "--init",
        default=BASE,
        help="adapter to start from, as `mhrl.cli grpo --init` takes it (default base)",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=DEFAULT_MAX_DRAWS,
        help="tasks to try until a group has contrast",
    )
    parser.add_argument(
        "--all-calls",
        action="store_true",
        help="print every rollout's command and calls, not only the first's",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, say: Callable[[str], None] = print) -> int:
    """Run one GRPO step through the project, printing every hand-off.

    Args:
        argv: The arguments, or None for ``sys.argv``.
        say: Where each line goes.

    Returns:
        0 after an update, 1 on a failure, 2 if no group had contrast.
    """
    arguments = sys.argv[1:] if argv is None else argv
    args = parse_args(arguments)
    ensure_project_python(args.project, arguments, say)
    init = start_adapter(args.init)
    project = import_project()
    import mhrl

    say(f"{SCRIPT} · one GRPO step through the project's own code")
    say(f"project   {short_path(project_of_module(mhrl))} (mhrl imported from it)")
    for line in settings_lines(project.settings, dict(os.environ)):
        say(line)
    try:
        for line in check_stack(project):
            say(line)
    except StackNotReadyError as error:
        say(f"not ready {error}")
        return EXIT_FAILED
    start = short_path(init) if init else "the base model (no adapter)"
    say(f"start     {start}")
    say(
        f"tasks     train_grpo's order: the {len(project.tasks)} training tasks "
        f"shuffled with seed {TASK_SEED}"
    )
    tracer = Tracer(project.settings, say, all_calls=args.all_calls)
    try:
        with tracer.listening(project.grpo, project.rollout):
            return run_draws(project, tracer, init, args.max_draws, say)
    except project.failures as error:
        # An error's message can hold a session URL: redact it like a line.
        say(tracer.redact(f"stopped   {type(error).__name__}: {error}"))
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
