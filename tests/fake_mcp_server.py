#!/usr/bin/env python3
"""A minimal MCP server over stdio, for testing the client.

Real enough to exercise the parts that matter: it emits an unsolicited
notification and a plain log line before answering, so the client is forced to
match replies by id instead of trusting the first line back.
"""
import json
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text back",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "what to echo"}},
            "required": ["text"],
        },
    },
    {
        "name": "boom",
        "description": "Always reports an error",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue                     # a notification; nothing to answer
        # Noise before every reply: a log line and an unrelated notification.
        sys.stdout.write("starting to think about it\n")
        sys.stdout.flush()
        send({"jsonrpc": "2.0", "method": "notifications/progress",
              "params": {"progress": 1}})
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake", "version": "1"},
                "capabilities": {"tools": {}}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"echo: {args.get('text')}"}]}})
            elif name == "boom":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "it went wrong"}],
                    "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32601, "message": f"no tool {name}"}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
