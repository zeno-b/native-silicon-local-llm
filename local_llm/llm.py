"""Prompt assembly, reasoning splitting and tool-call parsing.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import html
import hashlib
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import sqlite3
import traceback
import shlex
import subprocess
import sys
import textwrap
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field, asdict, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403


def note_prefix(prompt: str) -> None:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    if PREFIX_STATE["hash"] == digest:
        return
    first = PREFIX_STATE["hash"] is None
    PREFIX_STATE["hash"] = digest
    PREFIX_STATE["changed_at"] = iso(utc_now())
    PREFIX_STATE["generation"] += 1
    if not first:
        log(
            "Agent prompt prefix changed (system prompt or tool set). Any cached "
            "prefix is now invalid and the next few steps will re-prefill in full.",
            logging.WARNING,
        )


REASONING_INSTRUCTION = (
    "\n\nThink before you answer. Begin your reply with your reasoning enclosed in "
    "<think> and </think> tags: work through the problem step by step, consider "
    "edge cases, and question your assumptions there. After the closing </think> "
    "tag, give your final answer (or, if you are calling a tool, the single JSON "
    "object). Put ONLY reasoning inside the tags and never the tool JSON."
)


def build_agent_system_prompt(base_prompt: str, registry: ToolRegistry,
                              reasoning: bool = False) -> str:
    lines = []
    for tool in registry.specs():
        params = ", ".join(
            f"{name} ({desc})" for name, desc in tool["parameters"].items()
        ) or "no arguments"
        required = ", ".join(tool["required"]) or "none"
        lines.append(f"- {tool['name']}: {tool['description']}\n  args: {params}\n  required: {required}")
    prompt = f"{base_prompt}\n\n{TOOL_PROTOCOL}" + "\n".join(lines)
    if reasoning:
        prompt += REASONING_INSTRUCTION
    note_prefix(prompt)
    return prompt


def extract_json_object(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply.

    Small models wrap the object in prose or a code fence at random, so scanning
    for balanced braces beats expecting a clean response.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:index + 1])
                start = -1
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


THINK_BLOCK = re.compile(r"(?is)<(think|thinking|reasoning)>.*?</\1>")
OPEN_THINK = re.compile(r"(?is)<(think|thinking|reasoning)>.*\Z")


class ThinkSplitter:
    """Streaming splitter that separates a model's <think> reasoning from its
    answer as tokens arrive.

    A reasoning model emits <think>...</think> before its answer. We want the
    reasoning shown live in its own visible "thinking" area, not stripped and
    hidden and not dumped raw into the answer. Tokens can split a tag across
    chunk boundaries, so a partial tag at the end of a feed is held over to the
    next one. feed() returns a list of ("think"|"answer", text) pieces.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.carry = ""

    @staticmethod
    def _prefix_tail(data: str, tag: str) -> int:
        """Length of a trailing slice of data that is a proper prefix of tag.

        So "...</thi" holds back 4 chars until the rest of </think> arrives.
        """
        for size in range(min(len(tag) - 1, len(data)), 0, -1):
            if tag.startswith(data[-size:]):
                return size
        return 0

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        data = self.carry + (text or "")
        self.carry = ""
        while data:
            if self.in_think:
                end = data.find(self.CLOSE)
                if end == -1:
                    keep = self._prefix_tail(data, self.CLOSE)
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("think", data))
                    data = ""
                else:
                    if end:
                        out.append(("think", data[:end]))
                    data = data[end + len(self.CLOSE):]
                    self.in_think = False
            else:
                start = data.find(self.OPEN)
                if start == -1:
                    keep = self._prefix_tail(data, self.OPEN)
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("answer", data))
                    data = ""
                else:
                    if start:
                        out.append(("answer", data[:start]))
                    data = data[start + len(self.OPEN):]
                    self.in_think = True
        return out


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's chain-of-thought block.

    Qwen3.5 and later emit <think>...</think> before the answer. extract_json_object
    scans for the first balanced JSON object anywhere in the reply, so reasoning
    that talks through candidate arguments gets parsed as the tool call itself.
    An unterminated block is stripped too, because mid-stream the closing tag
    has not arrived yet and the agent tests the buffer on every token.
    """
    if not text or "<" not in text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = OPEN_THINK.sub("", text)
    return text.strip()


def parse_tool_call(text: str, known: set[str] | None = None) -> tuple[str, dict] | None:
    """Return (tool name, args) if the reply is a tool call, else None.

    `known` is the set of registered tool names. When it is supplied, a JSON
    object whose name is not a real tool is only treated as a call if it used
    the explicit "tool" key. That stops a final answer that happens to contain
    JSON (a config snippet, a parsed record) from being executed as a tool call.
    """
    parsed = extract_json_object(strip_reasoning(text))
    if not parsed:
        return None
    explicit = "tool" in parsed or "tool_name" in parsed
    name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    name = name.strip()
    if known is not None and name not in known and not explicit:
        return None
    args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"query": args}
    if not isinstance(args, dict):
        args = {}
    return name, args


@dataclass
class GenerationStats:
    """What one model call actually cost.

    prompt_tokens is the number that matters most on this hardware: an agent
    re-sends its whole prompt every step, so prefill, not decode, is where a
    multi-step run spends its time. Watching this climb step over step within a
    single run is how you see the quadratic.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    from_server: bool = False

    @property
    def decode_tps(self) -> float:
        decode_ms = max(1.0, self.total_ms - self.ttft_ms)
        return self.completion_tokens / (decode_ms / 1000.0)

    def as_event(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "decode_tps": round(self.decode_tps, 2),
            "estimated": not self.from_server,
        }


_CONNECTIVITY: dict = {"online": True, "checked_at": 0.0}


async def has_internet(recheck_after: float = 30.0) -> bool:
    """Best-effort connectivity check, cached briefly.

    Used to decide whether lookups are even possible. When offline the agent
    answers from its own knowledge instead of attempting a search that would
    fail slowly. The result is cached for recheck_after seconds so it costs at
    most one tiny request per window. Any failure is read as offline.
    """
    now = time.time()
    if now - _CONNECTIVITY["checked_at"] < recheck_after:
        return _CONNECTIVITY["online"]
    online = False
    try:
        import httpx
        # A couple of reliable, lightweight endpoints; success on either is enough.
        async with httpx.AsyncClient(timeout=3.0) as client:
            for url in ("https://1.1.1.1", "https://dns.google"):
                try:
                    resp = await client.head(url)
                    if resp.status_code < 500:
                        online = True
                        break
                except Exception:
                    continue
    except Exception:
        online = False
    _CONNECTIVITY.update(online=online, checked_at=now)
    return online



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'GenerationStats',
    'OPEN_THINK',
    'REASONING_INSTRUCTION',
    'THINK_BLOCK',
    'ThinkSplitter',
    '_CONNECTIVITY',
    'build_agent_system_prompt',
    'extract_json_object',
    'has_internet',
    'note_prefix',
    'parse_tool_call',
    'strip_reasoning',
]
