# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""A tiny MCP server with one tool, for the "your setup" run.

MCP (the Model Context Protocol) is how people plug extra tools into Claude
Code. A server is any program that answers a few JSON-RPC requests, one JSON
object per line on its standard input and output:

* ``initialize``: say hello and what it offers (tools);
* ``tools/list``: describe each tool (name, description, input schema);
* ``tools/call``: run one.

Claude Code starts the server, lists its tools, and offers each one to the
model as ``mcp__<server>__<tool>``, beside its own tools, on every call. That
is the cost this chapter measures: a tool the model never uses still rides
along in every request. ``run_claude.sh --setup yours`` registers this file
with ``--mcp-config``; nothing here is ever called by the scripted run.
"""

from __future__ import annotations

import json
import sys
from typing import Any

SERVER_NAME = "notes"
SERVER_VERSION = "1.0.0"
DEFAULT_PROTOCOL = "2025-06-18"
JSONRPC = "2.0"
# JSON-RPC's error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

NOTES = {
    "patients": "patients(id, name, city, birth_date); one row per patient.",
    "doctors": "doctors(id, name, specialty); one row per doctor.",
    "appointments": "appointments(id, patient_id, doctor_id, appointment_date, fee).",
}
READ_NOTE = {
    "name": "read_note",
    "description": "Read the team's note about one table of the clinic database.",
    "inputSchema": {
        "type": "object",
        "properties": {"table": {"type": "string", "enum": sorted(NOTES)}},
        "required": ["table"],
    },
}


def error_reply(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    Args:
        request_id: The id of the request it answers (None if unknown).
        code: The JSON-RPC error code.
        message: What went wrong.

    Returns:
        The response.
    """
    error = {"code": code, "message": message}
    return {"jsonrpc": JSONRPC, "id": request_id, "error": error}


def answer(request: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one JSON-RPC message.

    Args:
        request: The decoded message.

    Returns:
        The response, or None for a notification (a message without an id),
        which gets no reply.
    """
    if "id" not in request:
        return None
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": params.get("protocolVersion", DEFAULT_PROTOCOL),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    elif method == "tools/list":
        result = {"tools": [READ_NOTE]}
    elif method == "tools/call":
        if params.get("name") != READ_NOTE["name"]:
            message = f"unknown tool {params.get('name')!r}"
            return error_reply(request["id"], INVALID_PARAMS, message)
        table = (params.get("arguments") or {}).get("table", "")
        text = NOTES.get(table, f"no note about {table!r}")
        result = {"content": [{"type": "text", "text": text}]}
    elif method == "ping":
        result = {}
    else:
        message = f"unknown method {method!r}"
        return error_reply(request["id"], METHOD_NOT_FOUND, message)
    return {"jsonrpc": JSONRPC, "id": request["id"], "result": result}


def main() -> None:
    """Serve requests from standard input until it closes."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            response = error_reply(None, PARSE_ERROR, f"not JSON: {error}")
        else:
            if isinstance(request, dict):
                response = answer(request)
            else:
                message = "a request is one JSON object"
                response = error_reply(None, INVALID_REQUEST, message)
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
