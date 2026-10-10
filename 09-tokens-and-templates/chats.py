"""The chats this chapter renders: one question, three user turns, three calls.

Messages are OpenAI-style: a list of dicts, each with a ``role`` (system, user,
assistant or tool) and its ``content``. An assistant message holds the model's
whole reply, reasoning included, written the way LFM2.5 writes it:
``<think>reasoning</think>`` and then the answer or a tool call. That is also
how the project's engine puts earlier replies back into a prompt.
"""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]
Tool = dict[str, Any]

# Two of LFM2.5's special tokens, as they appear in its replies.
THINK_OPEN = "<think>"
END_OF_TURN = "<|im_end|>"

QUESTION = "How many customers live in Paris?"

# mini-swe-agent's bash tool, exactly as it sends it (version 2.4.6, the one the
# project runs: minisweagent/models/utils/actions_toolcall.py). A tool is a
# function the model may ask the program to run; chapter 16 shows the loop.
BASH_TOOL: Tool = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def user(text: str) -> Message:
    """Build a user message.

    Args:
        text: What the person (or the harness) writes.

    Returns:
        The message.
    """
    return {"role": "user", "content": text}


def assistant(reply: str) -> Message:
    """Build an assistant message from the model's whole reply.

    Args:
        reply: The reply as the model wrote it, ``<think>...</think>`` first.

    Returns:
        The message.
    """
    return {"role": "assistant", "content": reply}


def tool_result(output: str) -> Message:
    """Build the message that carries a tool's output back to the model.

    Args:
        output: What the command printed.

    Returns:
        The message.
    """
    return {"role": "tool", "content": output}


def one_question() -> list[Message]:
    """Return a chat of one user message.

    Returns:
        The messages.
    """
    return [user(QUESTION)]


def three_user_turns() -> list[Message]:
    """Return a chat with three user turns and two earlier replies.

    Returns:
        The messages, ending with the third user turn.
    """
    return [
        user(QUESTION),
        assistant(
            "<think>The customers table has a city column.</think>"
            "SELECT COUNT(*) FROM customers WHERE city = 'Paris';"
        ),
        user("And in Berlin?"),
        assistant(
            "<think>The same query, another city.</think>"
            "SELECT COUNT(*) FROM customers WHERE city = 'Berlin';"
        ),
        user("Which city has the most customers?"),
    ]


# The job for the three calls: chapter 1's, as a user message.
AGENT_TASK = (
    "There is a SQLite database named data.db. "
    f"{QUESTION} Write the answer to answer.txt."
)

# Three calls of a tiny agent on the shop database: what the model wrote each
# time, and what the bash command printed for shop.db (trailing spaces and the
# final newline trimmed).
AGENT_CALLS: tuple[tuple[str, str], ...] = (
    (
        "<think>First, which tables are in this database?</think>"
        "<|tool_call_start|>[bash(command='sqlite3 data.db .tables')]"
        "<|tool_call_end|>",
        "customers    order_items  orders       products",
    ),
    (
        "<think>The customers table should have a city column.</think>"
        "<|tool_call_start|>[bash(command='sqlite3 data.db "
        "\"SELECT COUNT(*) FROM customers WHERE city = \\'Paris\\'\"')]"
        "<|tool_call_end|>",
        "13",
    ),
    (
        "<think>The answer is 13. I will save it to answer.txt.</think>"
        "<|tool_call_start|>[bash(command='echo 13 > answer.txt')]"
        "<|tool_call_end|>",
        "",
    ),
)


def agent_chat(calls_done: int) -> list[Message]:
    """Return the conversation as it stands before a call.

    Args:
        calls_done: How many calls have already happened (0 for the first).

    Returns:
        The messages the harness sends for the next call.
    """
    messages = [user(AGENT_TASK)]
    for reply, output in AGENT_CALLS[:calls_done]:
        messages.append(assistant(reply))
        messages.append(tool_result(output))
    return messages
