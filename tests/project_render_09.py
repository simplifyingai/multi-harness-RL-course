"""Record the multi-harness-rl project's own render of chapter 9's chats.

This is not a test and not part of the chapter: it runs inside the project,
with the project's Python, and prints JSON. Each case is one chat rendered by
the project's ``PolicyEngine.render`` (mhrl/engine/policy.py), loaded with its
tokenizer only, from its own copy of the model files. Cases with
``preserve_thinking`` off use the same tokenizer's ``apply_chat_template`` with
that flag off, which is what transformers renders by default.

``tests/data/09_project_render.json`` is this script's output, made with::

    cd <multi-harness-rl> && .venv/bin/python <course>/tests/project_render_09.py \\
        > <course>/tests/data/09_project_render.json

``test_09_tokens_and_templates.py`` checks the chapter's render against it, and
re-runs this script when ``MHRL_PROJECT`` names a checkout of the project.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

CHAPTER = Path(__file__).resolve().parent.parent / "09-tokens-and-templates"
sys.path.insert(0, str(CHAPTER))

import chats  # noqa: E402  (the chapter's chats: plain Python, no imports)
from mhrl.config import get_settings  # noqa: E402
from mhrl.engine.policy import PolicyEngine  # noqa: E402

PACKAGES = ("transformers", "tokenizers", "jinja2", "mlx-lm")
UNKNOWN = "unknown"
BINARY_BYTES = 1500  # the start of shop.db, read as text
JSON_INDENT = 1

# An OpenAI-style tool call and result, with quotes, a backslash and a newline
# in the arguments, and reasoning in a separate field.
TOOL_CALL_CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "There is a SQLite database named data.db."},
            {"type": "text", "text": " " + chats.QUESTION},
        ],
    },
    {
        "role": "assistant",
        "reasoning_content": "Look at the tables first.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": {"command": "sqlite3 data.db '.tables'\necho \\ok"},
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "customers  orders"},
    {"role": "user", "content": "Now count Paris."},
    {"role": "assistant", "content": "<think>Count it.</think>13"},
    {"role": "tool", "content": "13"},
]
WIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "läs",
        "description": "Läs en fil <b>&</b> 'x' — 東京",
        "parameters": {
            "type": "object",
            "properties": {"z": {"type": "integer"}, "a": {"type": "string"}},
        },
    },
}


def cases() -> list[dict[str, Any]]:
    """Build every chat to render: the chapter's own, then harder ones.

    Returns:
        One dict per case: name, messages, tools and preserve_thinking.
    """
    raw = (CHAPTER / "data" / "shop.db").read_bytes()[:BINARY_BYTES]
    binary = raw.decode("utf-8", errors="replace")
    read_database = [
        chats.assistant(
            "<think>Let me look at the database file.</think>"
            "<|tool_call_start|>[bash(command='cat data.db')]<|tool_call_end|>"
        ),
        chats.tool_result(binary),
    ]
    tools = [chats.BASH_TOOL]
    built = [
        ("one-question", chats.one_question(), None, True),
        ("one-question-bash", chats.one_question(), tools, True),
        ("three-user-turns", chats.three_user_turns(), None, True),
        ("three-user-turns-default", chats.three_user_turns(), None, False),
    ]
    for calls_done in range(len(chats.AGENT_CALLS)):
        messages = chats.agent_chat(calls_done)
        built.append((f"agent-call-{calls_done + 1}", messages, tools, True))
        built.append((f"agent-call-{calls_done + 1}-default", messages, tools, False))
    built += [
        ("tool-calls", TOOL_CALL_CHAT, tools, True),
        ("tool-calls-default", TOOL_CALL_CHAT, tools, False),
        (
            "wide-characters",
            [
                {"role": "system", "content": ""},
                chats.user("Zürich — 東京 ✓ <b>&amp;</b> 'quote' <|im_end|>"),
            ],
            [WIDE_TOOL, chats.BASH_TOOL],
            True,
        ),
        ("binary-output", chats.agent_chat(0) + read_database, tools, True),
    ]
    return [
        {"name": name, "messages": chat, "tools": chat_tools, "preserve_thinking": keep}
        for name, chat, chat_tools, keep in built
    ]


def project_commit() -> str:
    """Return the project's git commit, or "unknown" without git or a checkout.

    Returns:
        The commit SHA.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:  # no git on this machine
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN


def no_reply(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """A scripted policy that never writes; it makes the engine load no weights.

    Args:
        messages: The chat (unused).
        tools: The tools (unused).

    Returns:
        An empty reply.
    """
    del messages, tools
    return ""


def main() -> None:
    """Render every case with the project's engine and print the JSON."""
    engine = PolicyEngine(get_settings(), scripted=no_reply)
    rendered = []
    for case in cases():
        if case["preserve_thinking"]:
            ids = engine.render(case["messages"], case["tools"])
        else:
            text = engine.tokenizer.apply_chat_template(
                case["messages"],
                tools=case["tools"] or None,
                add_generation_prompt=True,
                tokenize=False,
                preserve_thinking=False,
            )
            ids = list(engine.tokenizer.encode(text, add_special_tokens=False))
        rendered.append({**case, "ids": ids})
    made_with = {
        "project_commit": project_commit(),
        "model_path": Path(get_settings().model_path).name,
        **{package: version(package) for package in PACKAGES},
    }
    json.dump(
        {"made_with": made_with, "cases": rendered}, sys.stdout, indent=JSON_INDENT
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
