"""The ReAct-style agent loop: routing, tools, memory, verification.

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
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403


class Agent:
    """A ReAct-style loop over the local model.

    The protocol is plain JSON in the message body rather than the OpenAI tools
    field, because mlx_lm.server support for native tool calling varies by
    version and small local models emit malformed tool_calls more often than
    they emit malformed JSON text.
    """

    def __init__(self, config: Config, registry: ToolRegistry, client: ModelClient):
        self.config = config
        self.registry = registry
        self.client = client
        # Open partial-output file handles, keyed by conversation. Reused across a
        # turn's many appends so we do not pay an open()/close() per write. An LRU
        # cap keeps the fd count bounded on a long-running server.
        self._partials: dict[str, dict] = {}
        self._partials_dir = DATA_DIR / "partials"

    async def route(self, message: str, history: list[dict] | None = None) -> dict:
        """Ask the model how to handle a message, as one structured decision.

        The router menu is generated from the registry: every tool that declares
        route_hint becomes a selectable action. Adding a routable tool therefore
        needs no change here and no new regex lane, which is what keeps routing
        maintainable as the toolset grows over the years.

        Returns {"action": "answer"} or {"action": "<tool name>", ...tool args}.
        The model reads the message (with a little history so follow-ups resolve)
        and picks. On any parse failure it biases to a web search when one exists,
        so a genuine lookup is never silently answered from stale weights.
        """
        routable = self.registry.routable()
        if not routable:
            # Nothing to route to; the model answers everything itself.
            return {"action": "answer"}

        # The menu: an answer option plus one line per routable tool, taken
        # straight from each tool's route_hint.
        # When a real codebase is attached, "answer" must NOT claim to cover code:
        # a request to fix the user's own files cannot be satisfied from weights.
        has_project = bool(self.config.project_dir)
        if has_project:
            answer_option = ('{"action":"answer"} — the DEFAULT for general questions '
                             "you can answer from your own knowledge: facts, "
                             "explanations, definitions, writing, math, reasoning, and "
                             "code written from scratch that does NOT touch the user's "
                             "project. If the user refers to THEIR code, this project, "
                             "a file, a bug, or says 'fix this', do NOT answer — use a "
                             "file tool to look at the real files first.")
        else:
            answer_option = ('{"action":"answer"} — the DEFAULT. Use it whenever you '
                             "can answer from your own knowledge: general facts, "
                             "explanations, definitions, writing, math, reasoning, and "
                             "all code. Most questions are answer.")
        options = [answer_option] + [t.route_hint for t in routable]
        system = (
            "You are a router. Read the user's latest message and reply with "
            "exactly ONE JSON object and nothing else. Prefer answering from your "
            "own knowledge; only choose a lookup tool when the question truly "
            "needs current, real-time, or external facts you cannot be confident "
            "about (today's events, prices, scores, the latest version of "
            "something, or a specific named entity you do not know). If it is "
            "general knowledge you already know, choose answer. Options:\n"
            + "\n".join(f"- {opt}" for opt in options)
            + "\nFill the fields from the user's own words. Reply with only the "
            "JSON object."
        )

        # A few recent turns so "look it up" / "and tomorrow?" resolve in context.
        context: list[dict] = [{"role": "system", "content": system}]
        for turn in (history or [])[-4:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                context.append({"role": role, "content": str(turn["content"])[:500]})
        context.append({"role": "user", "content": message[:1000]})

        # If the reply is unusable, answer from own knowledge rather than
        # defaulting to a search. Prefer the model's knowledge unless it clearly
        # asked for a tool.
        fallback = {"action": "answer"}

        try:
            text, _ = await self.client.complete_with_stats(
                context, max_tokens=64, temperature=0.0
            )
        except Exception as exc:
            log(f"Router call failed ({exc}); falling back.", logging.WARNING)
            return fallback

        decision = extract_json_object(text) or {}
        action = str(decision.get("action", "")).lower().strip()

        if action == "answer":
            return {"action": "answer"}

        # A tool action: it must name a routable tool, and after alias-mapping its
        # required arguments must be present. Anything missing falls back safely.
        tool = self.registry.get(action)
        if tool is not None and tool.routable:
            args = {k: v for k, v in decision.items() if k != "action"}
            args = self.registry.normalise_args(tool, args)
            if all(r in args and str(args[r]).strip() for r in tool.required):
                # Trim over-long string args defensively.
                args = {k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()}
                return {"action": action, **args}

        return fallback

    def with_retrieved_context(self, user_message: str) -> str:
        """Prepend the most relevant knowledge-base passages to the question.

        This is the RAG step: when documents have been indexed, the best-matching
        passages are injected so the model answers from the user's own material
        (with source paths) instead of guessing. Silent no-op when the knowledge
        base is empty, so behaviour is unchanged until documents are added.
        """
        db = getattr(self.registry, "db", None)
        if not self.config.rag_enabled or db is None:
            return user_message
        if not getattr(db, "fts_enabled", False):
            return user_message
        try:
            scope = [p.strip() for p in (self.config.rag_scope or "").split(",") if p.strip()]
            # Scope retrieval to the acting user's own + shared documents so one
            # user's imported history never surfaces in another's answers.
            hits = db.search_documents(user_message, limit=self.config.rag_passages,
                                       only=scope or None, user_id=get_acting_user())
        except Exception as exc:
            log(f"knowledge-base lookup skipped: {exc}", logging.DEBUG)
            return user_message
        if not hits:
            return user_message
        budget = max(400, int(self.config.context_size * 0.25) * 4)
        blocks, used = [], 0
        for h in hits:
            piece = f"[{h['path']}]\n{h['chunk']}"
            if used + len(piece) > budget:
                break
            blocks.append(piece)
            used += len(piece)
        if not blocks:
            return user_message
        return ("Relevant passages from the user's indexed documents:\n\n"
                + "\n\n".join(blocks)
                + "\n\nUsing those passages where they apply (cite the [path] when you do), "
                  "answer:\n" + user_message)

    def build_base(
        self,
        history: list[dict],
        user_message: str,
        reserve: int,
    ) -> tuple[list[dict], int]:
        """Assemble [system, trimmed history, user]. This prefix is never cut later."""
        system = {
            "role": "system",
            "content": build_agent_system_prompt(self.config.system_prompt_with_identity, self.registry,
                                              reasoning=self.config.reasoning_visible),
        }
        user = {"role": "user", "content": self.with_retrieved_context(user_message)}
        return trim_to_context(
            system,
            [{"role": m["role"], "content": m["content"]} for m in history],
            user,
            self.config.context_size,
            reserve,
        )

    SUMMARY_MARKER = "EARLIER STEPS (condensed):"

    def compact(self, base: list[dict], scratch: list[dict], reserve: int) -> int:
        """Shrink an overlong trace in place. Returns how many entries collapsed.

        Mutating scratch rather than recomputing a view each step is the whole
        point. A prefix cache matches on the token prefix, so what it needs is
        for step k+1's prompt to *start with* step k's prompt. Appending to a
        stable list gives exactly that. Recomputing a summary every step does
        not: the summary text changes as more is folded into it, which moves
        every token after it and invalidates the cache on every single step.

        So the collapse happens once, when the budget is actually exceeded, and
        the run then extends cleanly again until the next one. Long runs of
        cache hits punctuated by rare misses, instead of a miss every step.
        """
        budget = max(256, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        fixed = messages_tokens(base)
        if fixed + messages_tokens(scratch) <= budget:
            return 0

        collapsed = 0
        carried: list[str] = []
        # Reclaim to 60% of budget so the next few steps fit without another
        # collapse. Collapsing to exactly the limit would re-trigger next step.
        target = int(budget * 0.6)
        while scratch and fixed + messages_tokens(scratch) > target:
            oldest = scratch.pop(0)
            collapsed += 1
            content = (oldest.get("content") or "").strip()
            if not content:
                continue
            if content.startswith(self.SUMMARY_MARKER):
                # Fold a previous summary in rather than nesting them.
                carried = [line[2:] for line in content.splitlines()[1:]] + carried
            elif oldest.get("role") == "user":
                carried.append(content.split("\n")[0][:120])

        if carried:
            scratch.insert(0, {
                "role": "user",
                "content": self.SUMMARY_MARKER + "\n" + "\n".join(f"- {line}" for line in carried[-12:]),
            })
        return collapsed

    def assemble(self, base: list[dict], scratch: list[dict], reserve: int) -> tuple[list[dict], int]:
        """base + the trace. base is fixed and can never be evicted.

        With stable_prefix off this drops the oldest entries from a copy every
        step, which is correct but cache-hostile. With it on, compact() has
        already made the list fit, so this is a concatenation and the prompt
        strictly extends between collapses.
        """
        if self.config.stable_prefix:
            collapsed = self.compact(base, scratch, reserve)
            return [*base, *scratch], collapsed

        budget = max(256, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        fixed = messages_tokens(base)
        kept = list(scratch)
        dropped = 0
        while kept and fixed + messages_tokens(kept) > budget:
            kept.pop(0)
            dropped += 1
        return [*base, *kept], dropped

    def _tool_budget(self, reserve: int) -> int:
        """Characters of a single tool result allowed into the context.

        A result added at step k is re-prefilled at every step after it, so the
        real cost is this number times the steps remaining. The share of context
        is deliberately smaller than it looks reasonable to allow.
        """
        room = max(512, (self.config.context_size - reserve) // 6) * CHARS_PER_TOKEN
        return int(min(self.config.tool_result_chars, room))

    async def compress_tool_result(self, name: str, result: str, budget: int) -> tuple[str, bool]:
        """Shrink an oversized tool result, preferring a summary over a hard cut.

        Truncation keeps the navigation chrome at the top of a page and throws
        away the part that answers the question. One cheap summarisation call
        pays for itself the moment two more steps follow.
        """
        if len(result) <= budget:
            return result, False
        if not self.config.summarise_tool_results or len(result) <= self.config.summarise_over_chars:
            return result[:budget] + "\n[truncated]", False
        prompt = [
            {"role": "system", "content":
                "You compress tool output. Reply with only the facts the caller asked for, "
                "in at most 8 short lines. Keep numbers, names, dates and URLs exactly. "
                "Do not add commentary, and do not invent anything."},
            {"role": "user", "content":
                f"Tool: {name}\nCompress this output:\n\n{result[:12000]}"},
        ]
        summary = await self.resilient_complete(
            prompt, max_tokens=min(400, budget // CHARS_PER_TOKEN), temperature=0.0
        )
        summary = strip_reasoning(summary).strip()
        if not summary:
            return result[:budget] + "\n[truncated]", False
        return f"[condensed from {len(result)} chars]\n{summary[:budget]}", True

    async def resilient_complete(self, messages: list[dict], max_tokens: int,
                                 temperature: float | None) -> str:
        """A non-streaming completion that never raises.

        On failure (a server OOM kill and watchdog restart present as a dropped
        connection here), it waits briefly for the server to come back and
        retries with a smaller token budget. If every attempt fails it returns an
        empty string, so callers degrade instead of erroring. Used for the router,
        the tool-result summariser, and the forced final answer.
        """
        tokens = max_tokens
        for attempt in range(self.config.resilient_retries + 1):
            try:
                text, _ = await self.client.complete_with_stats(messages, tokens, temperature)
                return text
            except Exception as exc:
                if attempt >= self.config.resilient_retries:
                    log(f"resilient_complete gave up after {attempt + 1} tries: {exc}",
                        logging.WARNING)
                    return ""
                # Wait for the (possibly restarting) server to be ready, then
                # retry with roughly half the tokens (floored), which also halves
                # the KV cache the reply needs.
                await self.client.wait_until_ready(timeout=self.config.ready_wait_timeout)
                tokens = max(self.config.min_max_tokens, tokens // 2)

    # Soft ceiling on a partial file. Past this we stop growing it (the reader is
    # tail-biased anyway), so a runaway task cannot fill the disk.
    PARTIAL_MAX_BYTES = 4_000_000
    # Most open partial handles to keep at once before closing the least-recent.
    PARTIAL_MAX_OPEN = 8

    def _partial_key(self, conversation_id: str | None) -> str:
        return "".join(c for c in (conversation_id or "scratch")
                       if c.isalnum() or c in "-_")[:60] or "scratch"

    def _partial_path(self, conversation_id: str | None) -> Path:
        """Path to the partial file. Does not touch the filesystem."""
        return self._partials_dir / f"{self._partial_key(conversation_id)}.md"

    def _close_partial(self, key: str) -> None:
        entry = self._partials.pop(key, None)
        if entry:
            try:
                entry["fh"].close()
            except Exception:
                pass

    def partial_begin(self, conversation_id: str | None, question: str) -> Path:
        """Open (truncate) the partial-output file for this turn and keep the
        handle open for the whole turn.

        Every finding, conclusion and chunk note is appended to this one open
        handle as it is produced, so if RAM runs out before the model can
        synthesise, the work is already on disk and can be handed back. Keeping
        the handle open avoids an open()/close() per append; a flush() after each
        write pushes the data to the OS page cache, which survives an OOM-kill of
        this process without the cost of an fsync.
        """
        key = self._partial_key(conversation_id)
        self._close_partial(key)  # a new turn for this conversation starts fresh
        try:
            self._partials_dir.mkdir(parents=True, exist_ok=True)  # once per turn
            fh = self._partial_path(conversation_id).open(
                "w", encoding="utf-8", buffering=1 << 16)
            fh.write(f"# Working notes\n\nRequest: {question[:500]}\n\n")
            fh.flush()
            self._partials[key] = {"fh": fh, "bytes": 0, "capped": False}
            # Bound open handles on a long-running server: close the oldest.
            while len(self._partials) > self.PARTIAL_MAX_OPEN:
                oldest = next(iter(self._partials))
                self._close_partial(oldest)
        except Exception as exc:
            log(f"could not open partial file: {exc}", logging.WARNING)
        return self._partial_path(conversation_id)

    def partial_add(self, conversation_id: str | None, text: str) -> None:
        """Append one piece of progress to the already-open partial file.

        One buffered write plus a cheap flush, no reopen. Stops growing the file
        past PARTIAL_MAX_BYTES so a runaway task cannot fill the disk; the reader
        keeps the head and tail regardless.
        """
        text = (text or "").strip()
        if not text:
            return
        key = self._partial_key(conversation_id)
        entry = self._partials.get(key)
        try:
            if entry is None:
                # Defensive: add without begin. Open in append mode once.
                self._partials_dir.mkdir(parents=True, exist_ok=True)
                fh = self._partial_path(conversation_id).open(
                    "a", encoding="utf-8", buffering=1 << 16)
                entry = {"fh": fh, "bytes": fh.tell(), "capped": False}
                self._partials[key] = entry
            if entry["capped"]:
                return
            chunk = text + "\n\n"
            entry["fh"].write(chunk)
            entry["fh"].flush()  # to OS cache: cheap, survives an OOM-kill
            entry["bytes"] += len(chunk.encode("utf-8", "ignore"))
            if entry["bytes"] >= self.PARTIAL_MAX_BYTES:
                entry["fh"].write("\n\n[partial truncated: size cap reached]\n")
                entry["fh"].flush()
                entry["capped"] = True
        except Exception as exc:
            log(f"could not append partial: {exc}", logging.DEBUG)

    def partial_read(self, conversation_id: str | None, max_chars: int = 8000) -> str:
        """Read back the accumulated partial work, tail-biased and bounded.

        Flushes the open handle first so our own read sees buffered writes, and
        seeks to read only the head and tail of a large file instead of loading
        the whole thing into memory.
        """
        key = self._partial_key(conversation_id)
        entry = self._partials.get(key)
        path = self._partial_path(conversation_id)
        try:
            if entry is not None:
                entry["fh"].flush()
            size = path.stat().st_size
            if size <= max_chars:
                data = path.read_text(encoding="utf-8", errors="replace").strip()
            else:
                # Read a head slice and a tail slice, skip the middle.
                head_n = 600
                tail_n = max_chars - head_n
                with path.open("rb") as fh:
                    head = fh.read(head_n)
                    fh.seek(-tail_n, 2)
                    tail = fh.read()
                data = (head.decode("utf-8", "replace").strip()
                        + "\n\n[...]\n\n"
                        + tail.decode("utf-8", "replace").strip())
            return data
        except Exception:
            return ""

    def salvage(self, conversation_id: str | None, note: str) -> str:
        """Build a useful answer from saved work when synthesis cannot run."""
        saved = self.partial_read(conversation_id)
        if saved:
            return (note + "\n\nHere is what I gathered before running low on "
                    "memory (also saved to disk):\n\n" + saved)
        return note

    async def run_iterating(self, message, history, conversation_id, max_tokens, temperature):
        """Run the agent, then verify any code it changed and, if the checks fail,
        let it see the errors and try again — up to auto_iterate_rounds times.

        Verification is layered so it works with or without the sandbox:
        - Always (no execution needed): syntax-check changed Python files.
        - When execution is enabled and a test command exists: run the tests.
        A failure at either layer sends the agent back with the exact errors.
        Only the final round's answer is surfaced as the turn's answer.
        """
        rounds = self.config.auto_iterate_rounds
        verify = rounds > 0
        current = message
        # Snapshot once, before any round: everything changed during this whole
        # turn (across rounds) is what we verify. Capturing per-round would miss a
        # file the fix re-edits, since it is already in changed_files by then.
        turn_start_changed = set(self.registry.changed_files)
        for round_i in range(rounds + 1):
            last_final = None
            async for ev in self.run(current, history, conversation_id, max_tokens, temperature):
                if ev.get("type") == "final":
                    last_final = ev
                    if not verify:
                        yield ev
                else:
                    yield ev
            if not verify:
                return

            new_files = sorted(self.registry.changed_files - turn_start_changed)
            if not new_files:
                if last_final:
                    yield last_final
                return

            problems = []
            # Layer 1: syntax check (always, safe without execution).
            syntax_errors = self.registry.syntax_check(new_files)
            if syntax_errors:
                problems.append("Syntax errors:\n" + syntax_errors)

            # Layer 2: tests, only if the sandbox is available.
            test_cmd = self.registry._detect_test_command()
            if self.config.allow_shell and test_cmd:
                yield {"type": "notice", "info": True,
                       "message": f"verifying in the sandbox (round {round_i + 1}/{rounds + 1}): {test_cmd}"}
                result = await asyncio.to_thread(self.registry._run_tests, "")
                if self.config.show_internals:
                    yield {"type": "detail", "message": "test output:\n" + result[:1000]}
                if "[exit code 0]" not in result:
                    problems.append("Test failures:\n" + result[:1500])
            elif test_cmd and not self.config.allow_shell:
                yield {"type": "notice", "info": True,
                       "message": "tests found but execution is off; ran a syntax "
                                  "check only. Start with --allow-shell to auto-run tests."}

            if not problems:
                yield {"type": "notice", "info": True,
                       "message": "changes verified \u2713 (" +
                                  ("syntax + tests" if (self.config.allow_shell and test_cmd) else "syntax")
                                  + ")"}
                if last_final:
                    yield last_final
                return

            if round_i >= rounds:
                yield {"type": "notice", "info": True,
                       "message": "checks still failing after the last round; returning the "
                                  "latest attempt (review the diff before using it)"}
                if last_final:
                    yield last_final
                return

            yield {"type": "notice", "info": True,
                   "message": "verification failed; fixing and re-checking"}
            current = ("Your edits did not pass verification. Fix the code so it passes. "
                       "Problems found:\n\n" + "\n\n".join(problems)[:2500])

    def running_summary(self, scratch: list[dict]) -> str:
        """A compact plain-text digest of the work so far.

        Used to keep memory flat on a big task: instead of carrying the whole
        transcript into every step (which grows the prompt and the KV cache until
        an 8GB machine OOMs), the transcript is periodically collapsed to this
        summary so each step's working set stays bounded. Slower, but it does not
        stop.
        """
        lines: list[str] = []
        for turn in scratch:
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            role = turn.get("role")
            # Keep tool results (they carry the facts) and the model's own notes,
            # trimmed hard; drop the boilerplate directives.
            if content.startswith("TOOL RESULT") or content.startswith("PAGE TEXT"):
                lines.append(" ".join(content[:600].split()))
            elif role == "assistant":
                lines.append("note: " + " ".join(content[:300].split()))
        return "\n".join(lines[-12:])

    async def plan_steps(self, question: str) -> list[str]:
        """Break a hard question into a short ordered list of sub-questions.

        One small model call. Returns 2..reasoning_max_steps concise steps. On any
        failure it returns a single step (answer the question directly), so the
        caller degrades to a normal answer rather than erroring.
        """
        prompt = [
            {"role": "system", "content":
                "Break the user's question into a short ordered list of sub-questions "
                "to work through, each on its own line, numbered. Between 2 and "
                f"{self.config.reasoning_max_steps} steps. Each step is one concrete "
                "thing to figure out. No preamble, just the numbered list."},
            {"role": "user", "content": question[:1000]},
        ]
        text = await self.resilient_complete(prompt, max_tokens=200, temperature=0.0)
        steps: list[str] = []
        for line in strip_reasoning(text).splitlines():
            line = line.strip()
            # Accept "1. x", "1) x", "- x", or a bare line.
            m = re.match(r"^(?:\d+[.)]|[-*])\s*(.+)$", line)
            step = (m.group(1) if m else line).strip()
            if step and len(step) > 3:
                steps.append(step[:200])
        steps = steps[:self.config.reasoning_max_steps]
        return steps or [question[:200]]

    async def reason_step(self, question: str, notes: list[str], step: str) -> str:
        """Answer one sub-question given only the compact notes so far.

        The working set is [question, a few prior conclusions, this step], which
        is small and constant regardless of how many steps have run. Returns a
        short conclusion to carry forward.
        """
        notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
        prompt = [
            {"role": "system", "content":
                "You are working through a hard question one step at a time. Use the "
                "findings so far, address only the current step, and reply with a "
                "short concrete conclusion in at most 4 sentences. Do not restate the "
                "whole problem."},
            {"role": "user", "content":
                f"Question: {question[:600]}\n\nFindings so far:\n{notes_text}\n\n"
                f"Current step: {step}\n\nYour conclusion for this step:"},
        ]
        text = await self.resilient_complete(prompt, max_tokens=self.config.reasoning_tokens, temperature=0.0)
        return strip_reasoning(text).strip()

    async def run(
        self,
        user_message: str,
        history: list[dict],
        conversation_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cancel: asyncio.Event | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Yield events: context, step, token, tool_call, tool_result, final, error, cancelled."""
        started = time.time()
        reserve = max_tokens or self.config.max_tokens
        known = set(self.registry.names())

        base, dropped = self.build_base(history, user_message, reserve)
        if dropped:
            yield {"type": "context", "dropped": dropped, "tokens": messages_tokens(base)}

        scratch: list[dict] = []
        seen_calls: list[str] = []
        trace: list[dict] = []
        nudges = 0
        prompt_tokens_total = 0
        yield {"type": "phase", "label": "preparing"}
        # Open the on-disk partial file so any work produced this turn is saved
        # as it goes and can be handed back if RAM runs out before synthesis.
        self.partial_begin(conversation_id, user_message)
        completion_tokens_total = 0
        # Snapshot of files changed before this turn, so the final event can show
        # a diff of exactly what this turn changed for you to review before push.
        changed_before = set(self.registry.changed_files)

        def detail(message: str) -> dict | None:
            """A verbose under-the-hood line, only emitted when show_internals is on."""
            return {"type": "detail", "message": message} if self.config.show_internals else None

        def done(answer: str, step: int, truncated: bool = False) -> dict:
            new_files = sorted(self.registry.changed_files - changed_before)
            diff = self.registry.git_diff(new_files) if new_files else ""
            return {
                "type": "final",
                "answer": answer,
                "steps": step,
                "trace": trace,
                "tools_used": [entry["name"] for entry in trace],
                "elapsed_ms": round((time.time() - started) * 1000),
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "truncated": truncated,
                "changed_files": new_files,
                "diff": diff,
            }

        # A tiny helper that runs a tool and seeds its result into the loop so
        # the model answers *from* the result instead of dumping it raw. Used by
        # both the deterministic shortcuts and the router below. Yields UI events
        # as it goes; returns True if the result was seeded (model should now
        # synthesise), False if the tool failed.
        async def run_and_seed(name: str, args: dict, note: str, directive: str) -> bool:
            yield {"type": "tool_call", "name": name, "args": args, "step": 0}
            result, error = await asyncio.to_thread(
                self.registry.call, name, args, conversation_id
            )
            yield {"type": "tool_result", "name": name, "result": result,
                   "error": error, "step": 0}
            if error:
                # A failed tool is not fatal: record it and let the model proceed.
                scratch.append({"role": "assistant", "content": f"I tried {name} and it failed."})
                scratch.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{result}"})
                yield {"__seeded__": False}
                return
            trace.append({"name": name, "args": args, "result": result[:1000], "error": None})
            # Record the call so the loop's dedup guard catches an immediate repeat.
            seen_calls.append(
                f"{name}:" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
            )
            budget = self._tool_budget(reserve)
            seeded, _ = await self.compress_tool_result(name, result, budget)
            # The assistant turn is prose, never a JSON tool call: a greedy small
            # model that sees its own previous turn was a tool call tends to emit
            # the same call again instead of answering.
            scratch.append({"role": "assistant", "content": note})
            scratch.append({"role": "user",
                            "content": f"TOOL RESULT [{name}]:\n{seeded}\n\n{directive}"})

            # Retrieval pipeline: for a web search, automatically fetch the top
            # result page(s) and give the model their full text. This is what
            # lets one generic search answer domain-specific questions whose
            # answer is on the page but not in the snippet (a price, a score, a
            # forecast, a version), so the app never needs a per-domain tool.
            if name == "web_search" and self.config.auto_fetch_results > 0 \
                    and self.registry.get("fetch_url") is not None:
                # Step-by-step multi-source read. Rather than dumping whole pages
                # into the prompt (which OOMs on 8GB with two sources), fetch each
                # of the top results, extract just the findings relevant to the
                # question in its own bounded, streamed pass, and accumulate short
                # notes. Then seed the notes plus a directive to compare the
                # sources and answer. Memory stays flat (one page at a time), the
                # work is visible as steps, and the model compares before it
                # answers.
                # Pull extra candidates so that skipping aggregator/thin pages
                # still leaves enough good sources to reach auto_fetch_results.
                want = self.config.auto_fetch_results
                candidates = top_result_urls(result, want + 4)
                source_notes: list[str] = []
                retrieval_started = time.time()
                good = 0
                idx = 0
                for url in candidates:
                    if good >= want:
                        break
                    # Skip listing/aggregator/JS-shell URLs before spending a
                    # fetch on them; their HTML is navigation, not article text.
                    if is_low_value_url(url):
                        yield {"type": "notice", "info": True,
                               "message": f"skipping a listing/aggregator page: {url[:60]}"}
                        continue
                    signature = "fetch_url:" + json.dumps({"url": url}, sort_keys=True,
                                                          ensure_ascii=False, default=str)
                    if signature in seen_calls:
                        continue
                    yield {"type": "tool_call", "name": "fetch_url",
                           "args": {"url": url}, "step": 0, "auto": True}
                    page, page_err = await asyncio.to_thread(
                        self.registry.call, "fetch_url", {"url": url}, conversation_id
                    )
                    yield {"type": "tool_result", "name": "fetch_url", "result": page,
                           "error": page_err, "step": 0, "auto": True}
                    seen_calls.append(signature)
                    if page_err:
                        continue  # a dead link is not fatal; the snippets remain
                    # If the fetched page is mostly markup/nav with little prose,
                    # treat it as a failed fetch: do not extract from it and do
                    # not let it into the synthesis prompt (empty extractions plus
                    # a junk-filled context are exactly what stalls on 8GB).
                    if is_thin_page(page):
                        yield {"type": "notice", "info": True,
                               "message": f"source had little readable text, skipping: {url[:60]}"}
                        continue
                    trace.append({"name": "fetch_url", "args": {"url": url},
                                  "result": page[:1000], "error": None})
                    good += 1
                    idx = good
                    page_budget = min(budget, self.config.auto_fetch_char_cap)
                    page_text, _ = await self.compress_tool_result(
                        "fetch_url", page[:self.config.auto_fetch_char_cap * 4], page_budget)
                    ev = detail(f"source {idx}: fetched {len(page)} chars, reading "
                                f"{len(page_text)} into a {self.config.reasoning_tokens}-token pass")
                    if ev:
                        yield ev
                    # Extraction is best-effort and fast-fail: stream with a
                    # per-source time cap, and on a stall keep whatever streamed
                    # and MOVE ON. It must never fall into a retry-with-wait
                    # (that compounding is what turned a stall into minutes).
                    yield {"type": "reason_step", "step": idx, "total": want,
                           "label": f"reading source {idx}: {url[:70]}"}
                    extract_messages = [
                        {"role": "system", "content":
                            "Read this one source and note only what is relevant to "
                            "answering the question, concisely. If the source does not "
                            "address it, say so in a few words."},
                        {"role": "user", "content":
                            f"Question: {user_message[:500]}\n\nSource {idx} "
                            f"({url}):\n{page_text}\n\nRelevant findings from this source:"},
                    ]
                    finding = ""
                    estats = GenerationStats()
                    estream = self.client.stream(extract_messages, self.config.reasoning_tokens, 0.0, estats)
                    started_src = time.time()
                    try:
                        async for tok in estream:
                            if cancel is not None and cancel.is_set():
                                break
                            finding += tok
                            yield {"type": "reason_token", "step": idx, "token": tok}
                            if time.time() - started_src > self.config.reasoning_step_timeout:
                                yield {"type": "notice", "info": True,
                                       "message": f"source {idx} slow; keeping partial and moving on"}
                                break
                    except Exception as exc:
                        # Best-effort: no retry. Whatever streamed is kept.
                        yield {"type": "notice", "info": True,
                               "message": f"source {idx} could not be read ({self.client.classify_error(exc)}); skipping"}
                    finally:
                        await estream.aclose()
                    finding = strip_reasoning(finding).strip()
                    took = time.time() - started_src
                    yield {"type": "reason_done", "step": idx, "conclusion": finding[:200]}
                    ev = detail(f"source {idx}: extracted {len(finding)} chars in {took:.1f}s")
                    if ev:
                        yield ev
                    if finding:
                        source_notes.append(f"[{idx}] {url}: {finding[:400]}")
                        self.partial_add(conversation_id, f"## Source {idx}: {url}\n{finding}")
                    # Total retrieval budget: if we have spent too long across all
                    # sources, stop fetching more and work with what we have.
                    if time.time() - retrieval_started > self.config.retrieval_deadline:
                        yield {"type": "notice", "info": True,
                               "message": "retrieval time budget reached; answering with what I have"}
                        break

                if source_notes:
                    joined = "\n".join(source_notes)
                    scratch.append({"role": "assistant",
                                    "content": f"I read {len(source_notes)} source(s) and noted the key points."})
                    scratch.append({"role": "user",
                                    "content": f"SOURCES:\n{joined}\n\nCompare these sources, "
                                               "note any agreement or conflict, then answer the "
                                               "original question. Cite the source URLs."})
                else:
                    # No source yielded usable findings. Do NOT synthesise over the
                    # empty notes plus big pages (that is what stalled). Answer
                    # briefly from the search snippets already seeded, or say so.
                    yield {"type": "notice", "info": True,
                           "message": "no usable content extracted from the pages; "
                                      "answering from the search snippets instead"}
                    scratch.append({"role": "user",
                                    "content": "The linked pages could not be read. Answer the "
                                               "question briefly from the search snippets above. "
                                               "If they do not contain the answer, say you could "
                                               "not find it rather than guessing."})
            yield {"__seeded__": True}

        # Step 0: oversized prompt. If the user's input alone is too large to
        # prefill in one pass on this machine, process it in parts: extract
        # findings from each chunk into bounded notes, then synthesise. Each pass
        # sees one chunk plus short notes, so memory stays flat regardless of how
        # big the input is. Done before routing, because a prompt this large
        # cannot survive a single generation to be routed normally.
        est_tokens = len(user_message) // CHARS_PER_TOKEN
        trigger_tokens = int(self.config.context_size * self.config.chunk_trigger_ratio)
        if (self.config.chunk_large_prompts and est_tokens > trigger_tokens
                and len(user_message) > 2000):
            chunk_tokens = max(256, int(self.config.context_size * self.config.chunk_size_ratio))
            chunk_chars = chunk_tokens * CHARS_PER_TOKEN
            parts = chunk_text(user_message, chunk_chars, overlap=chunk_chars // 10)
            # The instruction usually sits at the very start or end of a big
            # paste; keep both ends visible to every pass and the synthesis.
            hint = user_message[:400]
            if len(user_message) > 900:
                hint = user_message[:400] + " [...] " + user_message[-300:]
            yield {"type": "phase", "label": f"input is large; reading it in {len(parts)} parts"}
            yield {"type": "notice",
                   "message": f"prompt is ~{est_tokens} tokens; processing in "
                              f"{len(parts)} parts to fit memory", "info": True}
            notes: list[str] = []
            for i, part in enumerate(parts, 1):
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled", "step": i, "trace": trace}
                    return
                yield {"type": "reason_step", "step": i, "total": len(parts),
                       "label": f"reading part {i}/{len(parts)}"}
                notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
                map_messages = [
                    {"role": "system", "content":
                        "You are reading one part of a long input to help answer the "
                        "user's request. Note only what is relevant to the request "
                        "from this part, concisely. If nothing here is relevant, say so."},
                    {"role": "user", "content":
                        f"Request: {hint}\n\nNotes so far:\n{notes_text}\n\n"
                        f"Part {i} of {len(parts)}:\n{part}\n\nRelevant notes from this part:"},
                ]
                finding = ""
                mstats = GenerationStats()
                mstream = self.client.stream(map_messages, self.config.reasoning_tokens, 0.0, mstats)
                started_part = time.time()
                try:
                    async for tok in mstream:
                        if cancel is not None and cancel.is_set():
                            break
                        finding += tok
                        yield {"type": "reason_token", "step": i, "token": tok}
                        if time.time() - started_part > self.config.reasoning_step_timeout:
                            break
                except Exception:
                    if not strip_reasoning(finding).strip():
                        finding = await self.resilient_complete(map_messages, self.config.reasoning_tokens, 0.0)
                finally:
                    await mstream.aclose()
                finding = strip_reasoning(finding).strip()
                yield {"type": "reason_done", "step": i, "conclusion": finding[:200]}
                if finding:
                    notes.append(f"part {i}: {finding[:400]}")
                    self.partial_add(conversation_id, f"## Part {i}\n{finding}")

            # Reduce: answer the request from the gathered notes, streamed.
            yield {"type": "phase", "label": "writing the answer"}
            joined = "\n".join(notes) or "(no relevant content found)"
            reduce_messages = [
                {"role": "system", "content": self.config.system_prompt_with_identity},
                {"role": "user", "content":
                    f"Request: {hint}\n\nNotes gathered from the full input, in order:\n"
                    f"{joined}\n\nNow give the complete answer to the request in plain text."},
            ]
            answer_buf = ""
            rstats = GenerationStats()
            rstream = self.client.stream(reduce_messages, reserve, temperature, rstats)
            try:
                async for tok in rstream:
                    if cancel is not None and cancel.is_set():
                        break
                    answer_buf += tok
                    yield {"type": "token", "token": tok, "step": len(parts)}
            except Exception:
                answer_buf = await self.resilient_complete(reduce_messages, reserve, temperature)
            finally:
                await rstream.aclose()
            answer = strip_reasoning(answer_buf).strip() or ("Notes from the input:\n" + joined)
            yield done(answer, len(parts))
            return

        # Step 1: deterministic shortcuts. A bare URL or a pure arithmetic
        # expression needs no model call at all. calculator and fetch_url produce
        # the answer itself (a number, a page), so we can return it directly.
        shortcut = quick_tool(user_message) if self.config.fast_path else None
        if shortcut and self.registry.get(shortcut[0]) is not None:
            name, args = shortcut
            yield {"type": "tool_call", "name": name, "args": args, "step": 0, "fast_path": True}
            result, error = await asyncio.to_thread(
                self.registry.call, name, args, conversation_id
            )
            yield {"type": "tool_result", "name": name, "result": result,
                   "error": error, "step": 0, "fast_path": True}
            if not error:
                trace.append({"name": name, "args": args, "result": result[:1000], "error": None})
                yield done(result.strip(), 0)
                return
            # A failed shortcut falls through to normal model handling.
            scratch.append({"role": "assistant", "content": f"I tried {name} and it failed."})
            scratch.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{result}"})

        # Step 2: model routing. For any substantive message the shortcuts did
        # not handle, ask the model how to handle it. This one structured call
        # replaces all the intent regexes: it decides answer vs search vs
        # weather, and extracts the query or the place and day from free text.
        elif self.config.knowledge_triage and is_substantive(user_message):
            # Decide how to handle a substantive message, then execute the
            # decision generically. The decision is either {"action":"answer"}
            # or {"action":"<tool name>", ...tool args}.
            #
            # Code requests are handled without a router call: a self-contained
            # one answers directly, and one that depends on current or external
            # information (a recent API, "latest" anything, security-research
            # topics like recon or CVEs) searches first and then writes the code.
            # Everything else goes to the registry-driven router.
            # If there is no internet, lookups cannot succeed, so answer from own
            # knowledge and say so once. This also makes the router moot offline.
            online = await has_internet()
            for_code = False
            if not online:
                yield {"type": "notice", "info": True,
                       "message": "working offline — answering from my own knowledge"}
                decision = {"action": "answer"}
            elif is_code_request(user_message):
                # With a project attached, code requests must go through the
                # router so it can pick a file tool: "fix this code" is about the
                # user's real files and cannot be answered from weights alone.
                if self.config.project_dir:
                    yield {"type": "phase", "label": "deciding how to handle this"}
                    decision = await self.route(user_message, history)
                elif (CODE_NEEDS_LOOKUP.search(user_message)
                        and self.registry.get("web_search") is not None):
                    decision = {"action": "web_search",
                                "query": code_search_topic(user_message)}
                    for_code = True
                else:
                    decision = {"action": "answer"}
            else:
                yield {"type": "phase", "label": "deciding how to handle this"}
                decision = await self.route(user_message, history)

            action = decision.get("action")
            tool = None if action == "answer" else self.registry.get(action or "")
            # When the router chose to answer from the model's own knowledge, do
            # not honor a lookup tool the model tries to call on its own. The
            # router already judged no external facts are needed; a self-issued
            # web_search here is the "searches all the time" leak. Calculator and
            # the like stay available; only the network lookups are withheld.
            LOOKUP_TOOLS = {"web_search", "fetch_url"}
            answer_routed = (action == "answer" and not for_code)
            answered_retry = False
            lookup_used = False
            ev = detail(f"router decision: {json.dumps(decision, ensure_ascii=False)[:200]}"
                        + ("" if online else " (offline)"))
            if ev:
                yield ev
            if action == "answer" and not scratch:
                yield {"type": "notice", "info": True,
                       "message": "decided to answer from my own knowledge"}

            if tool is not None and tool.routable:
                # Generic execution for any routable tool. Terminal tools (a
                # calculator, a page fetch) return their result as the answer;
                # non-terminal tools (search, weather) seed the result and let
                # the model answer from it.
                args = {k: v for k, v in decision.items() if k != "action"}
                if tool.terminal:
                    yield {"type": "tool_call", "name": action, "args": args, "step": 0}
                    result, error = await asyncio.to_thread(
                        self.registry.call, action, args, conversation_id
                    )
                    yield {"type": "tool_result", "name": action, "result": result,
                           "error": error, "step": 0}
                    if not error:
                        trace.append({"name": action, "args": args,
                                      "result": result[:1000], "error": None})
                        yield done(result.strip(), 0)
                        return
                    scratch.append({"role": "assistant", "content": f"I tried {action} and it failed."})
                    scratch.append({"role": "user", "content": f"TOOL RESULT [{action}]:\n{result}"})
                else:
                    # Directive: the code path overrides it to ask for code; every
                    # other tool uses its own seed_directive (or a sane default).
                    if for_code:
                        directive = ("Use these results as reference, then write the "
                                     "code the user asked for. Prefer standard-library "
                                     "approaches and note briefly if anything may be "
                                     "version-dependent. If a page is needed call "
                                     "fetch_url; do not repeat the search.")
                        note = "I looked up current references before writing this."
                    else:
                        directive = (tool.seed_directive
                                     or "Answer my original question using this result.")
                        note = f"I used {action} to get this."
                    async for event in run_and_seed(action, args, note, directive):
                        if "__seeded__" not in event:
                            yield event
            # action == "answer" (or an unavailable/unknown tool): nothing seeded,
            # the loop below answers directly from the model's own knowledge.

            # Incremental reasoning: if the question is a hard analytical one and
            # nothing was seeded (a pure "answer" that isn't code), decompose it
            # and work through it step by step from a bounded, growing set of
            # conclusions, then stream the synthesis. This keeps the working set
            # small on 8GB and lets a 3B reason in depth by taking its time.
            if (not scratch and action == "answer"
                    and self.config.incremental_reasoning
                    and not is_code_request(user_message)
                    and is_reasoning_question(user_message)):
                yield {"type": "phase", "label": "planning the approach"}
                steps = await self.plan_steps(user_message)
                if len(steps) >= 2:
                    plan_lines = "; ".join(f"{i}) {st}" for i, st in enumerate(steps, 1))
                    yield {"type": "notice", "info": True,
                           "message": f"plan ({len(steps)} steps): {plan_lines[:400]}"}
                    notes: list[str] = []
                    for i, sub in enumerate(steps, 1):
                        if cancel is not None and cancel.is_set():
                            yield {"type": "cancelled", "step": i, "trace": trace}
                            return
                        yield {"type": "phase", "label": f"reasoning step {i}/{len(steps)}"}
                        # Stream each step live so thinking is never a frozen
                        # label: the user sees tokens appear as the model works.
                        yield {"type": "reason_step", "step": i, "total": len(steps), "label": sub[:120]}
                        notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
                        step_messages = [
                            {"role": "system", "content":
                                "You are working through a hard question one step at a time. "
                                "Use the findings so far, address only the current step, and "
                                "reply with a short concrete conclusion in at most 4 sentences."},
                            {"role": "user", "content":
                                f"Question: {user_message[:600]}\n\nFindings so far:\n{notes_text}"
                                f"\n\nCurrent step: {sub}\n\nYour conclusion:"},
                        ]
                        conclusion = ""
                        rstats = GenerationStats()
                        rstream = self.client.stream(step_messages, self.config.reasoning_tokens, 0.0, rstats)
                        started_step = time.time()
                        try:
                            async for tok in rstream:
                                if cancel is not None and cancel.is_set():
                                    break
                                conclusion += tok
                                yield {"type": "reason_token", "step": i, "token": tok}
                                # Per-step wall-clock cap: keep what streamed and
                                # move on rather than letting one step wedge.
                                if time.time() - started_step > self.config.reasoning_step_timeout:
                                    yield {"type": "notice", "message":
                                           f"step {i} taking long; moving on with partial", "info": False}
                                    break
                        except Exception:
                            # Streaming failed; fall back to a resilient non-stream.
                            if not strip_reasoning(conclusion).strip():
                                conclusion = await self.reason_step(user_message, notes, sub)
                        finally:
                            await rstream.aclose()
                        conclusion = strip_reasoning(conclusion).strip()
                        yield {"type": "reason_done", "step": i, "conclusion": conclusion[:200]}
                        if conclusion:
                            notes.append(f"{i}. {sub}: {conclusion[:300]}")
                            self.partial_add(conversation_id, f"## Step {i}: {sub}\n{conclusion}")
                    # Synthesise the final answer from the conclusions, streamed.
                    yield {"type": "phase", "label": "writing the answer"}
                    joined = "\n".join(notes)
                    final_messages = [
                        {"role": "system", "content": self.config.system_prompt_with_identity},
                        {"role": "user", "content":
                            f"{user_message}\n\nYou worked through this and reached these "
                            f"conclusions:\n{joined}\n\nNow give the complete final answer "
                            "in plain text, drawing them together. Do not number the steps."},
                    ]
                    answer_buf = ""
                    fstats = GenerationStats()
                    fstream = self.client.stream(final_messages, reserve, temperature, fstats)
                    try:
                        async for tok in fstream:
                            if cancel is not None and cancel.is_set():
                                break
                            answer_buf += tok
                            yield {"type": "token", "token": tok, "step": len(steps)}
                    except Exception:
                        # Fall back to a non-streaming resilient synthesis.
                        answer_buf = await self.resilient_complete(
                            final_messages, reserve, temperature)
                    finally:
                        await fstream.aclose()
                    answer = strip_reasoning(answer_buf).strip()
                    if not answer:
                        answer = "Here is what I worked out:\n" + joined
                    yield done(answer, len(steps))
                    return

        for step in range(1, self.config.agent_max_steps + 1):
            if cancel is not None and cancel.is_set():
                yield {"type": "cancelled", "step": step, "trace": trace}
                return

            messages, condensed = self.assemble(base, scratch, reserve)
            yield {"type": "step", "step": step, "max_steps": self.config.agent_max_steps,
                   "prompt_tokens": messages_tokens(messages), "condensed": condensed}
            ev = detail(f"step {step}: prompt {messages_tokens(messages)} tokens, "
                        f"reply budget {reserve}, {len(scratch)} scratch turns"
                        + (f", condensed {condensed}" if condensed else ""))
            if ev:
                yield ev

            buffer = ""
            cancelled = False
            stats = GenerationStats()
            # Tool-selection steps want deterministic JSON. Only the answer the
            # user reads should get the configured temperature, and we do not
            # know which this is until it parses, so bias towards valid JSON and
            # let the final-answer pass below use the warmer setting.
            step_temperature = (
                self.config.tool_temperature if temperature is None else temperature
            )
            # Generate this step, retrying with a smaller budget on failure rather
            # than surfacing an error. A model-server OOM kill and watchdog restart
            # look like a dropped stream from here, so a shrink-and-retry both
            # rides out the restart and asks for a reply small enough to fit.
            gen_reserve = reserve
            ctx_reserve = reserve
            step_failed = False
            for attempt in range(self.config.resilient_retries + 1):
                buffer = ""
                stats = GenerationStats()
                # On a retry, re-assemble with a much larger reserve, which
                # collapses the *prompt* budget and trims the trace hard. The
                # failure on 8GB is prefill of an oversized prompt (a stall, no
                # first token), so the input is the lever, not the reply length.
                # Each attempt cuts the prompt to roughly half of the previous,
                # so the three attempts are genuinely distinct rather than
                # bouncing off the reply-token floor.
                if attempt > 0:
                    # Leave only ~attempt/(attempt+1) of the window as reserve,
                    # i.e. cut the prompt to about 1/2, 1/3, 1/4 ... of the
                    # context on successive attempts. Monotonic and distinct.
                    ctx_reserve = min(self.config.context_size - self.config.min_max_tokens,
                                      int(self.config.context_size * (attempt / (attempt + 1))))
                    messages, _ = self.assemble(base, scratch, ctx_reserve)
                stream = self.client.stream(messages, gen_reserve, step_temperature, stats)
                # Split the model's <think> reasoning from its answer as it
                # streams, so the reasoning shows in its own visible thinking
                # area instead of being hidden or dumped raw into the answer.
                splitter = ThinkSplitter()
                try:
                    async for token in stream:
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break
                        buffer += token
                        for kind, piece in splitter.feed(token):
                            if not piece:
                                continue
                            if kind == "think":
                                yield {"type": "think_token", "token": piece, "step": step}
                            else:
                                yield {"type": "token", "token": piece, "step": step}
                        visible = strip_reasoning(buffer).lstrip()
                        if visible.startswith(("{", "```")) and parse_tool_call(buffer, known):
                            break
                    step_failed = False
                    break
                except Exception as exc:
                    await stream.aclose()
                    log(f"generation step {step} attempt {attempt + 1} failed: {exc}",
                        logging.DEBUG)
                    # If usable text already streamed, keep it rather than redoing
                    # work; the loop below can act on a partial answer or call.
                    if strip_reasoning(buffer).strip():
                        step_failed = False
                        break
                    if attempt >= self.config.resilient_retries:
                        step_failed = True
                        break
                    # Shrink the reply a little too, but the prompt cut above is
                    # the real lever. Report the prompt shrink, which is what
                    # actually changes between attempts.
                    gen_reserve = max(self.config.min_max_tokens, int(gen_reserve * 0.75))
                    reason = self.client.classify_error(exc)
                    yield {"type": "notice", "step": step,
                           "message": f"{reason}; cutting the prompt hard and retrying "
                                      f"(attempt {attempt + 2}). Work so far is saved."}
                    await self.client.wait_until_ready(timeout=self.config.ready_wait_timeout)
                    continue
                finally:
                    # Breaking out early leaves the HTTP response open until the
                    # generator is collected, which on a local server means a
                    # socket per abandoned step.
                    await stream.aclose()

            if step_failed:
                # Every retry failed. Distinguish a genuine memory limit from a
                # backend that is up but not generating (e.g. the mlx-lm
                # generation thread crashed): probe with a real one-token
                # completion. An honest message points at the real fix.
                can_generate = await self.client.wait_until_ready(timeout=8.0)
                if can_generate:
                    note = ("I ran low on memory before I could finish this in one "
                            "pass. Try a smaller or more specific request, or raise "
                            "the RAM headroom.")
                else:
                    note = ("The model server is up but not generating — this is a "
                            "backend error, not a memory limit. Check logs/model_server.log "
                            "and restart the model (Settings → Restart model).")
                yield done(self.salvage(conversation_id, note), step, truncated=True)
                return

            prompt_tokens_total += stats.prompt_tokens
            completion_tokens_total += stats.completion_tokens
            yield {"type": "usage", "step": step, **stats.as_event()}

            if cancelled:
                yield {"type": "cancelled", "step": step, "partial": buffer.strip(), "trace": trace}
                return

            call = parse_tool_call(buffer, known)

            # On an answer-routed turn, a self-issued network lookup is allowed
            # ONCE, and only when the question is genuinely time-sensitive (the
            # same signal the router searches on). That lets the model fetch
            # fresh facts when it truly needs them while still blocking the
            # reflexive "search everything" behaviour for general knowledge.
            # parse_tool_call honors an explicit "tool" key regardless of the
            # allowed set, so the gate must be here, at execution.
            if (call is not None and answer_routed and call[0] in LOOKUP_TOOLS
                    and is_time_sensitive(user_message) and not lookup_used):
                lookup_used = True
                yield {"type": "notice", "info": True,
                       "message": "the question looks time-sensitive; allowing one lookup"}
                # fall through to normal tool execution below
            elif call is not None and answer_routed and call[0] in LOOKUP_TOOLS:
                if not answered_retry:
                    answered_retry = True
                    yield {"type": "notice", "info": True,
                           "message": "answering from my own knowledge (no lookup needed)"}
                    scratch.append({"role": "user", "content":
                        "Answer the question directly from your own knowledge. "
                        "Do NOT call any tool and do NOT output JSON; just give the answer."})
                    continue
                # It insisted again: force a plain, tool-free answer.
                nudge = scratch + [{"role": "user", "content":
                    "Answer in prose from your own knowledge. No tools, no JSON."}]
                direct_msgs, _ = self.assemble(base, nudge, reserve)
                direct = await self.resilient_complete(direct_msgs, reserve, temperature)
                yield done(strip_reasoning(direct).strip()
                           or "I don't have enough to answer that confidently.", step)
                return

            if call is None:
                answer = strip_reasoning(buffer).strip()
                if answer:
                    yield done(answer, step)
                    return
                # An empty reply is a hiccup, not an answer. Nudge once.
                if nudges == 0 and step < self.config.agent_max_steps:
                    nudges += 1
                    scratch.append({"role": "assistant", "content": "(empty)"})
                    scratch.append({
                        "role": "user",
                        "content": "That reply was empty. Answer the question in plain text, "
                                   "or call exactly one tool as a JSON object.",
                    })
                    continue
                yield done("", step)
                return

            name, args = call

            if name == "final_answer":
                answer = str(args.get("answer") or "").strip()
                if answer:
                    yield {"type": "tool_call", "name": name, "args": args, "step": step}
                    yield done(answer, step)
                    return
                scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
                scratch.append({
                    "role": "user",
                    "content": "final_answer needs a non-empty answer argument. "
                               "Reply again with the full answer.",
                })
                continue

            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            yield {"type": "tool_call", "name": name, "args": args, "step": step}

            if signature in seen_calls:
                result = (
                    f"You already called {name} with these arguments and received a result. "
                    "Do not repeat it. Answer the user now with what you have, or call a "
                    "different tool."
                )
                error = None
            else:
                seen_calls.append(signature)
                result, error = await asyncio.to_thread(
                    self.registry.call, name, args, conversation_id
                )

            budget = self._tool_budget(reserve)
            result_for_model, was_summarised = await self.compress_tool_result(name, result, budget)

            trace.append({"name": name, "args": args, "result": result[:1000], "error": error})
            yield {"type": "tool_result", "name": name, "result": result, "error": error,
                   "step": step, "context_chars": len(result_for_model),
                   "summarised": was_summarised}
            ev = detail(f"tool {name}: {len(result)} chars returned, "
                        f"{len(result_for_model)} into context"
                        + (" (summarised)" if was_summarised else "")
                        + (f", error: {error}" if error else ""))
            if ev:
                yield ev

            scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
            scratch.append({
                "role": "user",
                "content": f"TOOL RESULT [{name}]:\n{result_for_model}\n\n"
                           "Use this to answer the original question, or call one more tool "
                           "if you genuinely still need it.",
            })

        # Ordinary step budget exhausted without a final answer. Rather than
        # stopping, keep going in bounded batches: collapse the work so far into a
        # compact running summary (so memory stays flat and an 8GB machine does
        # not OOM), then grant another batch of steps, up to hard_step_cap. This
        # is the "slow down but do not stop" path for a task too big for one pass.
        extra_batches = 0
        while (self.config.agent_max_steps * (extra_batches + 1) < self.config.hard_step_cap
               and (cancel is None or not cancel.is_set())):
            extra_batches += 1
            summary = self.running_summary(scratch)
            # Reset the working set to just the summary: constant memory regardless
            # of how much has already happened.
            scratch = [{
                "role": "user",
                "content": f"PROGRESS SO FAR (continue the task, do not restart):\n{summary}\n\n"
                           "Keep going one step at a time. Answer in plain text when done, "
                           "or call one tool as JSON to make progress.",
            }]
            yield {"type": "notice", "step": self.config.agent_max_steps * extra_batches,
                   "message": "task is large; continuing step by step from a summary"}

            batch_progress = False
            for extra in range(1, self.config.agent_max_steps + 1):
                step = self.config.agent_max_steps * extra_batches + extra
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled", "step": step, "trace": trace}
                    return
                messages, _ = self.assemble(base, scratch, reserve)
                yield {"type": "step", "step": step, "max_steps": self.config.hard_step_cap,
                       "prompt_tokens": messages_tokens(messages)}
                buffer = await self.resilient_complete(messages, reserve, temperature) or ""
                call = parse_tool_call(buffer, known)
                if call is None:
                    answer = strip_reasoning(buffer).strip()
                    if answer:
                        yield done(answer, step, truncated=True)
                        return
                    continue
                batch_progress = True  # a tool call is forward motion
                name, args = call
                if name == "final_answer":
                    answer = str(args.get("answer") or "").strip()
                    if answer:
                        yield done(answer, step, truncated=True)
                        return
                    continue
                signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
                yield {"type": "tool_call", "name": name, "args": args, "step": step}
                if signature in seen_calls:
                    result, error = ("Already ran that; use the result you have or try "
                                     "another tool.", None)
                else:
                    seen_calls.append(signature)
                    result, error = await asyncio.to_thread(
                        self.registry.call, name, args, conversation_id
                    )
                budget = self._tool_budget(reserve)
                result_for_model, _ = await self.compress_tool_result(name, result, budget)
                trace.append({"name": name, "args": args, "result": result[:1000], "error": error})
                yield {"type": "tool_result", "name": name, "result": result,
                       "error": error, "step": step}
                scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
                scratch.append({"role": "user",
                                "content": f"TOOL RESULT [{name}]:\n{result_for_model}"})
                self.partial_add(conversation_id, f"Tool {name}: {str(result)[:500]}")

            if not batch_progress:
                # A whole batch produced no answer and no tool call — almost
                # always repeated stalls. Continuing would only stretch a stall
                # into minutes (the 789-second grind). Stop and salvage instead.
                yield {"type": "notice", "info": True,
                       "message": "no progress in the last batch; wrapping up with what I have"}
                break

        # Reached the hard cap, or was cancelled, or a batch stalled out. Force one
        # plain answer, never an error, from the compact summary so the reply
        # always closes cleanly.
        summary = self.running_summary(scratch)
        final_messages = [
            {"role": "system", "content": self.config.system_prompt_with_identity},
            {"role": "user", "content":
                f"{user_message}\n\nWork so far:\n{summary}\n\n"
                "Give your best final answer now in plain text. Do not call any tool."},
        ]
        answer = await self.resilient_complete(final_messages, reserve, temperature)
        answer = strip_reasoning(answer).strip()
        if not answer:
            # Synthesis itself could not run — hand back the saved work from disk
            # (falling back to the in-memory summary) so the turn still delivers.
            answer = self.salvage(
                conversation_id,
                "I reached the step limit before finishing in one pass.")
            if answer.strip() == "I reached the step limit before finishing in one pass.":
                answer += "\n\nHere is as far as I got:\n" + summary
        yield done(answer, self.config.hard_step_cap, truncated=True)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Agent',
]
