"""The `tools` module a script run by execute_code imports.

Copied into the execution directory at run time and imported by the script as
`tools`; it is never imported by this package. Kept as a real file rather than a
string constant inside tools.py so it can be read, linted and tested like any
other code, and so its own docstrings do not have to survive being nested inside
another string literal.

The script talks to the agent's tool registry over a loopback socket with a
one-time token. Every call still goes through ToolRegistry.call on the other
side, so the per-tool guards, the path confinement, the argument aliasing and
the tool_calls audit trail all apply exactly as they do to a model-issued call.
"""

from __future__ import annotations

import json as _json
import os as _os
import socket as _socket


class ToolError(RuntimeError):
    """A tool refused the call, or the bridge is gone."""


def _rpc(payload: dict) -> str:
    host = _os.environ["LLM_TOOL_HOST"]
    port = int(_os.environ["LLM_TOOL_PORT"])
    payload = dict(payload, token=_os.environ["LLM_TOOL_TOKEN"])
    with _socket.create_connection((host, port), timeout=60) as sock:
        sock.sendall((_json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    reply = _json.loads(buf.decode("utf-8", "replace") or "{}")
    if not reply.get("ok"):
        raise ToolError(reply.get("error") or "tool call failed")
    return reply.get("result", "")


def call(tool: str, **args) -> str:
    """Call one of the agent's tools and return its result as a string."""
    return _rpc({"tool": tool, "args": args})


def names() -> list[str]:
    """The tools this script is allowed to call."""
    listing = _rpc({"tool": "__names__", "args": {}})
    return [name for name in listing.split(",") if name]


def __getattr__(name: str):
    """Sugar: tools.read_file(path="x") means tools.call("read_file", path="x").

    Defined as a module __getattr__ so the script does not have to know which
    tools exist at import time; an unknown name fails at call time with the
    registry's own message, which lists what is available.
    """
    if name.startswith("_"):
        raise AttributeError(name)

    def _bound(**args) -> str:
        return call(name, **args)

    _bound.__name__ = name
    return _bound
