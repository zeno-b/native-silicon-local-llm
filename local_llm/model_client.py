"""HTTP client for the local model server, with retries.

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
from .llm import *  # noqa: F401,F403


class ModelClient:
    """Thin async client for the local OpenAI-compatible model server."""

    def __init__(self, config: Config):
        self.config = config

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.model_port}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"http://127.0.0.1:{self.config.model_port}/v1/models"

    async def wait_until_ready(self, timeout: float = 40.0) -> bool:
        """Wait until the server can actually GENERATE, or until timeout.

        A /v1/models ping only proves the process is up. It does not prove the
        generation thread is alive: a backend crash (e.g. mlx-lm dying inside
        _step) leaves the HTTP server answering 200 on /v1/models while every
        completion stalls forever. So probe with a real one-token completion.
        If that returns, the server can generate; if it errors or times out, it
        cannot, and the caller degrades honestly instead of retrying into a
        corpse. Returns True only on a genuine completion.
        """
        import httpx
        deadline = time.time() + timeout
        delay = 0.5
        probe_body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=min(15.0, timeout)) as client:
                    resp = await client.post(self.url, json=probe_body)
                if resp.status_code < 500:
                    data = resp.json()
                    # A live generation thread returns a choices array; a crashed
                    # backend that still serves HTTP will not.
                    if data.get("choices"):
                        return True
            except Exception:
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)
        return False

    @staticmethod
    def classify_error(exc: Exception) -> str:
        """A short, honest label for a generation failure.

        The retry notice used to say "memory limit" for every exception, which
        hid stalls, resets and Cloudflare-style pages behind a wrong cause. This
        names what actually happened so the UI and logs are truthful.
        """
        import httpx
        name = type(exc).__name__
        text = str(exc).lower()
        if isinstance(exc, httpx.ReadTimeout) or "readtimeout" in name.lower() or "timed out" in text:
            return "the model stalled (no output in time)"
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)) or \
                "connection" in text or "reset" in text or "refused" in text:
            return "the model server dropped, likely out of memory and restarting"
        if "memory" in text or "oom" in text or "alloc" in text:
            return "the model server ran out of memory"
        return f"a generation error ({name})"

    def payload(
        self,
        messages: list[dict],
        stream: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        body = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        # mlx_lm.server reads these from the request body; they are not OpenAI
        # parameters. They are OMITTED by default: on recent mlx-lm the
        # repetition-penalty path builds a null logits-processor list and the
        # generation thread crashes on every request. Only send them when
        # explicitly re-enabled.
        if (self.config.repetition_penalty_enabled
                and self.config.repetition_penalty
                and self.config.repetition_penalty != 1.0):
            body["repetition_penalty"] = self.config.repetition_penalty
            body["repetition_context_size"] = self.config.repetition_context_size
        if stream:
            # Ask for usage on the final chunk. Servers that do not know the
            # option ignore it, and the estimate below covers them.
            body["stream_options"] = {"include_usage": True}
        if self.config.disable_thinking and not self.config.reasoning_visible:
            # Qwen3.5 and friends default to thinking-on. Chain of thought is a
            # bad trade in a tool loop: it burns the KV cache on tokens the
            # protocol discards, and the reasoning text confuses JSON parsing.
            # Both spellings are in circulation; unknown keys are ignored.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["enable_thinking"] = False
        return body

    def _stats_from_usage(self, usage: dict | None, messages: list[dict], text: str,
                          started: float, first_token_at: float | None) -> GenerationStats:
        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_token_at - started) * 1000) if first_token_at else total_ms
        if usage:
            return GenerationStats(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                ttft_ms=ttft_ms, total_ms=total_ms, from_server=True,
            )
        return GenerationStats(
            prompt_tokens=messages_tokens(messages),
            completion_tokens=estimate_tokens(text),
            ttft_ms=ttft_ms, total_ms=total_ms, from_server=False,
        )

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        text, _ = await self.complete_with_stats(messages, max_tokens, temperature)
        return text

    async def complete_with_stats(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, GenerationStats]:
        import httpx
        started = time.time()
        timeout = httpx.Timeout(self.config.stall_timeout, connect=15.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.url, json=self.payload(messages, False, max_tokens, temperature))
            if resp.status_code != 200:
                fallback = self.payload(messages, False, max_tokens, temperature)
                fallback.pop("max_tokens", None)
                resp = await client.post(self.url, json=fallback)
            if resp.status_code != 200:
                raise RuntimeError(f"model server returned {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text, self._stats_from_usage(data.get("usage"), messages, text, started, None)

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stats: GenerationStats | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield content deltas. When `stats` is passed it is filled in place.

        In place rather than returned because this is an async generator and the
        caller needs the numbers even when it breaks out of the loop early,
        which the agent does on every completed tool call.
        """
        import httpx
        started = time.time()
        first_token_at: float | None = None
        usage: dict | None = None
        text_len = 0
        try:
            timeout = httpx.Timeout(self.config.stall_timeout, connect=15.0, pool=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", self.url, json=self.payload(messages, True, max_tokens, temperature)
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise RuntimeError(f"model server returned {resp.status_code}: {body[:300]}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(chunk)
                        except Exception:
                            continue
                        if data.get("usage"):
                            usage = data["usage"]
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = (choices[0].get("delta") or {}).get("content", "")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.time()
                            text_len += len(delta)
                            yield delta
        finally:
            if stats is not None:
                measured = self._stats_from_usage(
                    usage, messages, "x" * text_len, started, first_token_at
                )
                stats.prompt_tokens = measured.prompt_tokens
                stats.completion_tokens = measured.completion_tokens
                stats.ttft_ms = measured.ttft_ms
                stats.total_ms = measured.total_ms
                stats.from_server = measured.from_server


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------
# The old approach tried to recognise every phrasing of "search this" or "what
# is the weather in X" with regular expressions. That is a losing game: users
# phrase things in unbounded ways, and each new phrasing needed another pattern.
# The general fix is to make the *model* the router. It emits one structured
# decision (a small JSON object) saying how to handle the message and, for a
# lookup, what to search or which place and day it is about. Deterministic code
# then executes that decision. The model does what models are good at (reading
# intent from free text); the code does what code is good at (reliably running
# the chosen tool). This module holds the two deterministic shortcuts kept for
# cost reasons, plus the JSON extraction the router relies on.

# A bare arithmetic expression: only digits, spaces and operators. Routed to the
# calculator without a model call, since "17*23" needs no interpretation.
ARITHMETIC_ONLY = re.compile(r"^[\d\s+\-*/().,^%]+$")
# A message that is nothing but a URL. Routed straight to fetch_url.
BARE_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.I)

# Greetings and acknowledgements that are not worth a routing round trip. These
# get a plain reply; everything longer is eligible for model routing.
TRIVIAL_MESSAGE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup)(?:\s+(?:there|all|everyone|folks|claude))?"
    r"|^\s*(?:thanks|thank you|thx|ok|okay|cool|nice|got it|great|perfect|"
    r"lol|haha|bye|goodbye|good (?:morning|evening|night)|"
    r"how are you|what's up|whats up)\b",
    re.I,
)
_TRIVIAL_TAIL = re.compile(r"[\s!.?]*$")



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ARITHMETIC_ONLY',
    'BARE_URL',
    'ModelClient',
    'TRIVIAL_MESSAGE',
    '_TRIVIAL_TAIL',
]
