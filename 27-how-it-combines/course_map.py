"""Which project file and function each chapter of the course mirrors.

One entry per chapter, 1 to 26 as the course stands, plus this chapter and
the next. Each names the chapter's own files, how they relate to the
project's code, and the project files and names behind them. A name is a
top-level function, class or constant (``reward_for``), a method or class
attribute (``PolicyEngine.render``, ``Settings.max_model_calls``), or a
function defined inside another (``train_grpo.loss_fn``). ``commands`` are
``python -m mhrl.cli`` subcommands, and an empty ``names`` means the whole
file. ``map.py`` checks every one against a real checkout.

Read the chapters' own README files for the detail: each one says, in its
own words, what of the project it copies or runs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Where a mirrored file lives.
PROJECT = "project"  # the multi-harness-rl checkout
OPENENV = "openenv"  # OpenEnv's capture proxy, installed in the project's .venv

# How a chapter's code relates to the project's.
COPY = "copy"  # the project's code, line for line (comments added)
CUT_DOWN = "cut down"  # the same steps, with less around them
TOY = "toy"  # the idea on chapters 2-8's toy; the project's file is full size
RUNS = "runs"  # the chapter runs or reads the project's file itself
KINDS = (COPY, CUT_DOWN, TOY, RUNS)

PARTS = (
    (1, "The job, and RL from zero on a toy analyst", range(1, 5)),
    (2, "The project's algorithms, still on the toy", range(5, 9)),
    (3, "The model on your laptop", range(9, 13)),
    (4, "Training the real model where we own the loop", range(13, 16)),
    (5, "Inside a harness", range(16, 20)),
    (6, "Capturing real harnesses", range(20, 24)),
    (7, "Training inside real harnesses, measured honestly", range(24, 29)),
)


@dataclass(frozen=True)
class Mirror:
    """One project file a chapter mirrors.

    Attributes:
        path: The file, relative to the project (or to OpenEnv's capture
            package when ``origin`` is ``OPENENV``).
        names: The functions, classes and constants in it; empty means the
            whole file.
        commands: ``python -m mhrl.cli`` subcommands (for ``mhrl/cli.py``).
        origin: ``PROJECT`` or ``OPENENV``.
    """

    path: str
    names: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    origin: str = PROJECT


@dataclass(frozen=True)
class Chapter:
    """One chapter and what it mirrors.

    Attributes:
        number: The chapter number.
        folder: Its folder in the course repository.
        kind: How its code relates to the project's (one of ``KINDS``).
        files: Its own files that do the mirroring.
        mirrors: The project files behind them.
        note: One short line on what is mirrored.
    """

    number: int
    folder: str
    kind: str
    files: tuple[str, ...]
    mirrors: tuple[Mirror, ...]
    note: str


# OpenEnv's capture graph, as chapters 21 and 22 read it (OpenEnv 3f14355).
CAPTURE_GRAPH = (
    Mirror("graph.py", origin=OPENENV),
    Mirror("export.py", origin=OPENENV),
    Mirror("validate.py", origin=OPENENV),
)
# How the project turns a rollout document into rows (chapters 21 and 22).
DOCUMENT_TO_ROWS = (
    Mirror("mhrl/rollout.py", ("count_calls", "run_rollout")),
    Mirror("mhrl/training/common.py", ("dedupe_shared_prefixes", "item_from_sequence")),
)

CHAPTERS: tuple[Chapter, ...] = (
    Chapter(
        1,
        "01-the-job",
        COPY,
        ("tasks.py", "grader.py", "make_tasks.py"),
        (
            Mirror(
                "mhrl/tasks/core.py",
                (
                    "load_tasks",
                    "task_instruction",
                    "prepare_workdir",
                    "grade",
                    "check_answer",
                ),
            ),
            Mirror("mhrl/tasks/generate.py", ("generate_tasks",)),
            Mirror("mhrl/tasks/questions.py", ("draw_instances", "resolve_answer")),
            Mirror("mhrl/tasks/databases.py", ("build_database",)),
        ),
        "the tasks, the grader, the generator, the five databases",
    ),
    Chapter(
        2,
        "02-plain-rl",
        TOY,
        ("bandit.py",),
        (
            Mirror("mhrl/rollout.py", ("reward_for",)),
            Mirror(
                "mhrl/config.py",
                ("Settings.train_temperature", "Settings.eval_temperature"),
            ),
        ),
        "a reward of 1 or 0; sample at temperature 1, measure greedily",
    ),
    Chapter(
        3,
        "03-policy-gradient",
        TOY,
        ("reinforce.py", "baseline.py"),
        (
            Mirror("mhrl/training/grpo.py", ("train_grpo.loss_fn",)),
            Mirror("mhrl/training/common.py", ("token_logprobs",)),
        ),
        "REINFORCE with a baseline: the GRPO loss at ratio 1",
    ),
    Chapter(
        4,
        "04-episodes",
        TOY,
        ("minianalyst.py", "environment.py"),
        (
            Mirror("mhrl/rollout.py", ("run_rollout",)),
            Mirror("mhrl/config.py", ("Settings.max_model_calls",)),
            Mirror("mhrl/training/common.py", ("item_from_sequence",)),
        ),
        "8 calls, one reward; every call shares the rollout's advantage",
    ),
    Chapter(
        5,
        "05-grpo-groups",
        COPY,
        ("grpo.py", "grpo_groups.py"),
        (
            Mirror("mhrl/training/grpo.py", ("group_advantages", "train_grpo")),
            Mirror("mhrl/config.py", ("Defaults.ADVANTAGE_EPS",)),
        ),
        "group advantages; a group without contrast is skipped",
    ),
    Chapter(
        6,
        "06-ratio-and-clip",
        COPY,
        ("grpo.py", "grpo_clip.py"),
        (
            Mirror("mhrl/training/grpo.py", ("train_grpo.loss_fn",)),
            Mirror("mhrl/config.py", ("Defaults.CLIP_EPSILON",)),
        ),
        "the importance ratio and the clip at 0.8-1.2",
    ),
    Chapter(
        7,
        "07-reward-design",
        COPY,
        ("rewards.py", "reward_lab.py", "real_run.py"),
        (
            Mirror("mhrl/rollout.py", ("tool_bonus", "token_bonus", "reward_for")),
            Mirror(
                "mhrl/config.py",
                ("Settings.token_budget", "Settings.length_guard_tokens"),
            ),
            Mirror("mhrl/cli.py", ("CliDefaults.GRPO_LR",)),
        ),
        "the call bonus that backfired and the token bonus that replaced it",
    ),
    Chapter(
        8,
        "08-sft-toy",
        TOY,
        ("sft.py", "teacher.py", "sft_toy.py"),
        (
            Mirror("mhrl/engine/oracle.py", ("AnswerKeyTeacher._steps",)),
            Mirror("mhrl/training/sft.py", ("train_sft.loss_fn",)),
            Mirror("mhrl/training/common.py", ("batches",)),
            Mirror("mhrl/cli.py", ("CliDefaults.SFT_LR",)),
        ),
        "imitation: the teacher's four paths and the SFT loss",
    ),
    Chapter(
        9,
        "09-tokens-and-templates",
        CUT_DOWN,
        ("template.py", "count.py"),
        (
            Mirror("mhrl/engine/policy.py", ("PolicyEngine.render",)),
            Mirror(
                "mhrl/config.py",
                (
                    "Defaults.END_OF_TURN",
                    "Defaults.TOOL_CALL_START",
                    "Defaults.TOOL_CALL_END",
                    "Settings.max_prompt_tokens",
                ),
            ),
            Mirror("mhrl/engine/server.py", ("create_app.chat",)),
        ),
        "the chat template and the 16,384-token prompt limit",
    ),
    Chapter(
        10,
        "10-meet-lfm",
        RUNS,
        ("download.py", "inside.py"),
        (
            Mirror("scripts/setup.sh"),
            Mirror("mhrl/config.py", ("Settings.model_path",)),
        ),
        "the download setup.sh makes, and the model the engine loads",
    ),
    Chapter(
        11,
        "11-quantize",
        RUNS,
        ("convert.py",),
        (
            Mirror("scripts/setup.sh"),
            Mirror("mhrl/config.py", ("Settings.model_path",)),
        ),
        "builds the 4-bit file the project downloads ready-made",
    ),
    Chapter(
        12,
        "12-generate",
        CUT_DOWN,
        ("sample_lab.py", "timing.py"),
        (
            Mirror(
                "mhrl/engine/policy.py",
                ("PolicyEngine._sampler", "PolicyEngine._sample", "_to_float32"),
            ),
            Mirror(
                "mhrl/config.py",
                ("Defaults.PREFILL_STEP", "Settings.max_completion_tokens"),
            ),
        ),
        "full-distribution sampling, float32 logprobs, the 1,024 cap",
    ),
    Chapter(
        13,
        "13-lora",
        COPY,
        ("lora.py", "lora_lab.py", "adam_steps.py"),
        (
            Mirror(
                "mhrl/lora.py",
                ("apply_lora", "save_adapter", "load_adapter", "zero_adapter"),
            ),
            Mirror("mhrl/engine/policy.py", ("PolicyEngine.set_adapter",)),
            Mirror(
                "mhrl/config.py",
                ("Settings.lora_rank", "Settings.lora_scale", "Settings.lora_layers"),
            ),
        ),
        "the adapter: rank 16, scale 2.0, the last 12 layers",
    ),
    Chapter(
        14,
        "14-sft-real-model",
        CUT_DOWN,
        ("sft.py", "rows.py", "train_sft.py"),
        (
            Mirror("mhrl/training/sft.py", ("train_sft", "demo_items")),
            Mirror(
                "mhrl/training/common.py",
                ("TrainItem", "item_from_sequence", "token_logprobs"),
            ),
        ),
        "rows with a loss mask, scored in one pass",
    ),
    Chapter(
        15,
        "15-grpo-real-model",
        CUT_DOWN,
        ("grpo.py", "grpo_sql.py", "rewards.py"),
        (
            Mirror("mhrl/training/grpo.py", ("train_grpo", "group_advantages")),
            Mirror("mhrl/rollout.py", ("reward_for", "token_bonus")),
            Mirror("mhrl/training/common.py", ("token_logprobs",)),
        ),
        "GRPO with the harness taken out: one reply per rollout",
    ),
    Chapter(
        16,
        "16-inside-claude-code",
        RUNS,
        ("run_claude.sh", "stand_in.py"),
        (
            Mirror(
                "mhrl/harnesses/claude_code.py",
                ("PINNED_VERSION", "ClaudeCode.command", "ClaudeCode.environment"),
            ),
        ),
        "the real Claude Code 2.1.290 with the project's flags",
    ),
    Chapter(
        17,
        "17-tiny-harness",
        CUT_DOWN,
        ("toolcalls.py", "tiny_harness.py"),
        (
            Mirror("mhrl/engine/toolcalls.py", ("split_reply", "parse_calls")),
            Mirror("configs/harnesses/mini_swe_agent_textbased.yaml"),
        ),
        "a tool call read out of the model's text",
    ),
    Chapter(
        18,
        "18-engine",
        CUT_DOWN,
        ("engine.py", "policy.py"),
        (
            Mirror(
                "mhrl/engine/policy.py",
                (
                    "PolicyEngine.render",
                    "PolicyEngine.generate",
                    "PolicyEngine._remember",
                    "PolicyEngine._restore",
                ),
            ),
            Mirror(
                "mhrl/engine/server.py",
                ("to_openai", "create_app.chat", "MLX_THREADS"),
            ),
        ),
        "an OpenAI-style engine that returns token ids and logprobs",
    ),
    Chapter(
        19,
        "19-sandbox",
        COPY,
        ("sandbox.sb", "sandbox.py", "process.py", "grade_safe.py"),
        (
            Mirror("configs/harnesses/sandbox.sb"),
            Mirror(
                "mhrl/harnesses/sandbox.py",
                ("sandbox_command", "sandbox_params", "check_writable_root"),
            ),
            Mirror("mhrl/harnesses/process.py", ("run_logged", "kill_tree")),
            Mirror("mhrl/tasks/core.py", ("grade",)),
            Mirror("mhrl/config.py", ("Settings.rollout_timeout_s",)),
        ),
        "sandbox-exec, the process tree, a grader that follows no link",
    ),
    Chapter(
        20,
        "20-capture-proxy",
        CUT_DOWN,
        ("capture_client.py", "start.sh"),
        (
            Mirror(
                "mhrl/rollout.py",
                (
                    "CaptureClient.open_session",
                    "CaptureClient.rollout",
                    "CaptureClient.close",
                ),
            ),
            Mirror("scripts/serve.sh", ("start",)),
        ),
        "a session per rollout; its id is the harness's API key",
    ),
    Chapter(
        21,
        "21-calls-to-rows",
        CUT_DOWN,
        ("training_rows.py", "rollout_view.py"),
        (*CAPTURE_GRAPH, *DOCUMENT_TO_ROWS),
        "the rollout document, cut into training rows",
    ),
    Chapter(
        22,
        "22-silent-capture-failures",
        CUT_DOWN,
        ("breakages.py", "crosscheck.py"),
        (
            *CAPTURE_GRAPH,
            *DOCUMENT_TO_ROWS,
            Mirror("mhrl/harnesses/smoke.py", ("smoke_once",)),
            Mirror("mhrl/harnesses/claude_code.py", ("REMINDER_OFF",)),
        ),
        "a broken prefix splits a rollout into many rows",
    ),
    Chapter(
        23,
        "23-three-harnesses",
        CUT_DOWN,
        ("wire.py", "run_one.py", "mini_swe_agent.yaml"),
        (
            Mirror(
                "mhrl/harnesses/base.py",
                ("Harness.run", "base_environment", "get_harness"),
            ),
            Mirror("mhrl/harnesses/mini_swe_agent.py", ("MiniSweAgent",)),
            Mirror("mhrl/harnesses/claude_code.py", ("ClaudeCode",)),
            Mirror("mhrl/harnesses/pi.py", ("Pi",)),
            Mirror("configs/harnesses/mini_swe_agent.yaml"),
        ),
        "three unmodified harnesses, wired by configuration only",
    ),
    Chapter(
        24,
        "24-measuring",
        COPY,
        ("evaluate.py", "baseline_eval.py", "noise.py"),
        (
            Mirror("mhrl/evaluate.py", ("evaluate", "summarize")),
            Mirror("mhrl/cli.py", ("balanced",), commands=("eval",)),
        ),
        "pass@1 per harness on a balanced held-out set",
    ),
    Chapter(
        25,
        "25-sft-in-harnesses",
        CUT_DOWN,
        ("teacher.py", "collect.py", "train_sft.py", "memory_probe.py"),
        (
            Mirror("mhrl/engine/oracle.py", ("AnswerKeyTeacher",)),
            Mirror(
                "mhrl/training/sft.py",
                ("collect_demonstrations", "demo_items", "train_sft"),
            ),
            Mirror(
                "mhrl/training/common.py",
                ("chunked_attention", "load_policy", "token_logprobs"),
            ),
        ),
        "the answer-key teacher, SFT on captured rows, attention in blocks",
    ),
    Chapter(
        26,
        "26-grpo-in-harnesses",
        CUT_DOWN,
        ("grpo_harness.py", "steps.py", "replay.py"),
        (
            Mirror("mhrl/training/grpo.py", ("train_grpo", "group_advantages")),
            Mirror("mhrl/rollout.py", ("run_rollout", "reward_for", "tool_bonus")),
            Mirror("mhrl/cli.py", ("print_steps",), commands=("steps",)),
        ),
        "GRPO through the harnesses, and the length guard",
    ),
    Chapter(
        27,
        "27-how-it-combines",
        RUNS,
        ("map.py", "one_step.py", "stack.sh"),
        (
            Mirror("mhrl/config.py", ("Settings", "Defaults", "get_settings")),
            Mirror(
                "mhrl/cli.py",
                ("main", "CliDefaults"),
                commands=("demo", "eval", "demos", "sft", "grpo", "steps", "report"),
            ),
            Mirror("mhrl/training/grpo.py", ("train_grpo", "use_adapter")),
            Mirror("mhrl/rollout.py", ("run_rollout",)),
            Mirror("scripts/serve.sh", ("start", "stop", "status")),
            Mirror("mhrl/harnesses/smoke.py", ("smoke_once", "main")),
            Mirror("tests/test_core.py"),
            Mirror("tests/test_harnesses.py"),
            Mirror("tests/test_oracle.py"),
            Mirror("tests/test_tasks.py"),
        ),
        "this chapter: the settings, the command line, one real step",
    ),
    Chapter(
        28,
        "28-capstone",
        RUNS,
        (),
        (
            Mirror("scripts/train_from_scratch.sh"),
            Mirror("mhrl/evaluate.py", ("format_table",)),
            Mirror("mhrl/cli.py", ("markdown_results",), commands=("report",)),
        ),
        "next: the whole pipeline, from a fresh clone",
    ),
)

# The chapters the course has published so far; 27 and 28 are this one and
# the next.
PUBLISHED = range(1, 27)


def chapter(number: int) -> Chapter:
    """Return one chapter's entry.

    Args:
        number: The chapter number.

    Returns:
        Its entry.

    Raises:
        KeyError: If the map has no such chapter.
    """
    for entry in CHAPTERS:
        if entry.number == number:
            return entry
    raise KeyError(f"no chapter {number} in the map")


def part_of(number: int) -> tuple[int, str]:
    """Return the course part a chapter belongs to.

    Args:
        number: The chapter number.

    Returns:
        The part's number and title.

    Raises:
        KeyError: If no part holds the chapter.
    """
    for part, title, numbers in PARTS:
        if number in numbers:
            return part, title
    raise KeyError(f"chapter {number} is in no part")
