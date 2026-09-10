"""A minimal Model Context Protocol client, so external MCP servers show up as tools.

Scope is deliberate: stdio transport, `initialize`, `tools/list`, `tools/call`.
That is the part of MCP that turns somebody else's server into a tool this agent
can use, and it needs no dependency beyond the standard library. Resources,
prompts, sampling and the HTTP transports are not implemented; a server that
offers them still works, its tools are just all we take.

Synchronous by design. ToolRegistry handlers are synchronous and the agent
already runs them off the event loop with asyncio.to_thread, so an async client
here would only add a second concurrency model to reason about.

Processes are shared PROCESS-WIDE, not per registry. A ToolRegistry is built for
every subagent and every task run, and starting a fresh `npx` server each time
would cost seconds per turn and leak processes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403


PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPToolSpec:
    """One remote tool, in the shape ToolRegistry needs."""

    server: str
    remote_name: str
    local_name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)


def _local_name(server: str, tool: str) -> str:
    """A registry-safe name that says where the tool came from.

    Prefixed rather than flattened so an MCP server cannot shadow a built-in
    tool: a server called "core" offering "read_file" must not become the
    read_file the agent already relies on.
    """
    def clean(text: str) -> str:
        out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(text).lower())
        return out.strip("_")[:40]
    return f"mcp_{clean(server)}_{clean(tool)}"


def _describe_schema(schema: Any) -> tuple[dict[str, str], list[str]]:
    """Turn a JSON Schema into (parameter descriptions, required names).

    The registry advertises parameters to the model as one line of prose each,
    so the useful part of a schema is the description and the type, not the
    constraints. Anything unparseable degrades to "no arguments" rather than
    breaking the tool.
    """
    if not isinstance(schema, dict):
        return {}, []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}, []
    parameters: dict[str, str] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            parameters[str(name)] = "value"
            continue
        kind = spec.get("type") or "value"
        text = str(spec.get("description") or "").strip()
        parameters[str(name)] = f"{text} ({kind})" if text else str(kind)
    required = [str(r) for r in (schema.get("required") or []) if isinstance(r, str)]
    return parameters, required


class MCPServer:
    """One MCP server subprocess, spoken to over JSON-RPC on stdio."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None,
                 timeout: float = 30.0):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = max(1.0, float(timeout))
        self.proc: subprocess.Popen | None = None
        self.tools: list[MCPToolSpec] = []
        self.error: str = ""
        self._lock = threading.Lock()
        self._next_id = 0

    # ------------------------------------------------------------ lifecycle --
    def start(self) -> bool:
        """Start the process and complete the handshake. False on any failure.

        Never raises. An MCP server that will not start is a missing capability,
        not a broken agent, so it degrades to a logged warning and no tools.
        """
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return True
            if shutil.which(self.command) is None and not Path(self.command).exists():
                self.error = f"{self.command!r} is not on PATH"
                log(f"MCP server {self.name!r}: {self.error}", logging.WARNING)
                return False
            try:
                self.proc = subprocess.Popen(
                    [self.command, *self.args],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True, bufsize=1,
                    cwd=self.cwd or None,
                    env={**os.environ, **self.env},
                )
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                log(f"MCP server {self.name!r} would not start: {self.error}",
                    logging.WARNING)
                self.proc = None
                return False
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": APP_NAME, "version": str(UI_BUILD)},
            })
            self._notify("notifications/initialized", {})
        except Exception as exc:
            self.error = f"handshake failed: {exc}"
            log(f"MCP server {self.name!r}: {self.error}", logging.WARNING)
            self.stop()
            return False
        self.error = ""
        return True

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass      # already gone

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -------------------------------------------------------------- protocol --
    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("server is not running")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        """One request/response round trip.

        Reads until the reply with OUR id arrives, discarding notifications and
        anything unparseable in between: a server is free to log progress and
        send unrelated notifications while we wait, and treating the first line
        back as the answer is how that turns into random failures.
        """
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id,
                        "method": method, "params": params})
            proc = self.proc
            if proc is None or proc.stdout is None:
                raise RuntimeError("server is not running")
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed its output")
                try:
                    message = json.loads(line)
                except Exception:
                    continue          # a log line, not protocol
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue          # a notification or someone else's reply
                if message.get("error"):
                    detail = message["error"]
                    text = (detail.get("message") if isinstance(detail, dict)
                            else str(detail))
                    raise RuntimeError(str(text)[:300])
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            raise TimeoutError(f"no reply within {self.timeout:.0f}s")

    # ----------------------------------------------------------------- tools --
    def list_tools(self) -> list[MCPToolSpec]:
        if not self.start():
            return []
        try:
            result = self._request("tools/list", {})
        except Exception as exc:
            self.error = f"tools/list failed: {exc}"
            log(f"MCP server {self.name!r}: {self.error}", logging.WARNING)
            return []
        specs: list[MCPToolSpec] = []
        for entry in (result.get("tools") or []):
            if not isinstance(entry, dict):
                continue
            remote = str(entry.get("name") or "").strip()
            if not remote:
                continue
            parameters, required = _describe_schema(entry.get("inputSchema"))
            specs.append(MCPToolSpec(
                server=self.name,
                remote_name=remote,
                local_name=_local_name(self.name, remote),
                description=(str(entry.get("description") or remote).strip()
                             + f" (via the {self.name} MCP server)"),
                parameters=parameters,
                required=required,
            ))
        self.tools = specs
        return specs

    def call(self, remote_name: str, arguments: dict) -> str:
        """Call a remote tool and flatten its content blocks to text."""
        if not self.start():
            return f"(the {self.name} MCP server is unavailable: {self.error})"
        try:
            result = self._request("tools/call",
                                   {"name": remote_name, "arguments": arguments or {}})
        except Exception as exc:
            # A dead server should be retried on the next call, not permanently
            # written off: `npx` servers get killed by all sorts of things.
            self.stop()
            raise ValueError(f"{self.name} MCP server: {exc}") from None
        parts: list[str] = []
        for block in (result.get("content") or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "resource":
                resource = block.get("resource") or {}
                parts.append(str(resource.get("text") or resource.get("uri") or ""))
            else:
                parts.append(f"[{block.get('type')} content omitted]")
        text = "\n".join(p for p in parts if p).strip()
        if result.get("isError"):
            raise ValueError(text or "the MCP tool reported an error")
        return text or "(the tool returned no content)"


class MCPManager:
    """Process-wide registry of MCP servers.

    A single instance, because ToolRegistry is constructed per subagent and per
    task run: starting a fresh `npx` server for each would cost seconds a turn
    and leak processes. Tool lists are cached for the life of the process; a
    server that changes its tool set needs a restart, which is the same contract
    as the rest of this project's server-level settings.
    """

    def __init__(self) -> None:
        self.servers: dict[str, MCPServer] = {}
        self._lock = threading.Lock()
        self._configured: str = ""

    def configure(self, spec: str, timeout: float = 30.0) -> None:
        """Adopt a server list, given as JSON. Idempotent for the same JSON."""
        with self._lock:
            if spec == self._configured:
                return
            self._configured = spec
            for server in self.servers.values():
                server.stop()
            self.servers = {}
            for entry in parse_mcp_spec(spec):
                name = entry["name"]
                self.servers[name] = MCPServer(
                    name=name, command=entry["command"], args=entry.get("args") or [],
                    env=entry.get("env") or {}, cwd=entry.get("cwd"), timeout=timeout)

    def tools(self) -> list[MCPToolSpec]:
        """Every remote tool, listing each server at most once per process."""
        out: list[MCPToolSpec] = []
        for server in list(self.servers.values()):
            if not server.tools and not server.error:
                server.list_tools()
            out.extend(server.tools)
        return out

    def call(self, server: str, remote_name: str, arguments: dict) -> str:
        target = self.servers.get(server)
        if target is None:
            raise ValueError(f"no MCP server named {server!r}")
        return target.call(remote_name, arguments)

    def status(self) -> list[dict]:
        return [{"name": s.name, "command": s.command, "args": s.args,
                 "alive": s.alive, "tools": [t.local_name for t in s.tools],
                 "error": s.error}
                for s in self.servers.values()]

    def stop_all(self) -> None:
        for server in list(self.servers.values()):
            server.stop()


def parse_mcp_spec(spec: str) -> list[dict]:
    """Validate a server list. Bad entries are dropped with a warning.

    Accepts either the array form or the `{"mcpServers": {...}}` object form that
    other MCP clients use, so a config file can be copied across unchanged.
    """
    text = (spec or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception as exc:
        log(f"MCP_SERVERS is not valid JSON ({exc}); no MCP servers configured.",
            logging.WARNING)
        return []
    entries: list[dict] = []
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        for name, body in data["mcpServers"].items():
            if isinstance(body, dict):
                entries.append({**body, "name": name})
    elif isinstance(data, list):
        entries = [e for e in data if isinstance(e, dict)]
    elif isinstance(data, dict):
        entries = [{**body, "name": name} for name, body in data.items()
                   if isinstance(body, dict)]
    out: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.get("enabled") is False:
            continue
        name = str(entry.get("name") or "").strip()
        command = str(entry.get("command") or "").strip()
        if not name or not command:
            log(f"Ignoring an MCP server entry with no name or command: "
                f"{str(entry)[:120]}", logging.WARNING)
            continue
        if name in seen:
            log(f"Ignoring a duplicate MCP server named {name!r}.", logging.WARNING)
            continue
        seen.add(name)
        args = entry.get("args") or []
        out.append({
            "name": name,
            "command": command,
            "args": [str(a) for a in args] if isinstance(args, list) else [],
            "env": entry.get("env") if isinstance(entry.get("env"), dict) else {},
            "cwd": str(entry["cwd"]) if entry.get("cwd") else None,
        })
    return out


# The one manager for this process.
MCP = MCPManager()


__all__ = [
    'MCP',
    'MCPManager',
    'MCPServer',
    'MCPToolSpec',
    'PROTOCOL_VERSION',
    'parse_mcp_spec',
]
