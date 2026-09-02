"""Message classifiers, chunking and routing heuristics.

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
from .model_client import *  # noqa: F401,F403


def _is_trivial(text: str) -> bool:
    """True if the whole message is just a greeting or acknowledgement."""
    match = TRIVIAL_MESSAGE.match(text)
    # The pattern matches a leading greeting; require that only punctuation and
    # whitespace follow it, so "hi there" is trivial but "hi, fix this bug" is not.
    return bool(match and _TRIVIAL_TAIL.fullmatch(text[match.end():]))


def is_substantive(message: str) -> bool:
    """True if a message is a real question or request worth routing.

    Used by the chat handlers to keep chit-chat on the cheap plain-reply path
    while sending anything substantive through the model router.
    """
    text = (message or "").strip()
    if len(text) < 8 or _is_trivial(text):
        return False
    return True


def _last_user_turn(history: list[dict] | None) -> str | None:
    """The most recent user message in a history list.

    The router is given recent history, but this is also used to log or inspect
    the previous turn when resolving follow-ups like "look it up".
    """
    for turn in reversed(history or []):
        if turn.get("role") == "user" and (turn.get("content") or "").strip():
            return turn["content"]
    return None


# Requests to write or modify code. These are answered from the model's own
# knowledge and must never be routed to a web search: a code-generation prompt
# ("write a python script...") sent to a search engine returns tutorials at best
# and, as with "os recon" being read as OCR, irrelevant junk at worst. Detected
# deterministically so it also skips the router call entirely.
CODE_INTENT = re.compile(
    r"\b(write|create|generate|build|implement|code|program|fix|debug|"
    r"refactor|optimi[sz]e|complete|extend|port|convert|translate|add|modify|"
    r"update|rewrite|snippet)\b",
    re.I,
)
CODE_OBJECT = re.compile(
    r"\b(script|scripts|function|functions|program|programme|code|class|classes|"
    r"method|methods|module|snippet|regex|regexp|cli|parser|app|application|"
    r"component|query|loop|algorithm|algorithms|unit ?tests?|test|tests|api|"
    r"endpoint|schema|decorator|generator|command|one[- ]?liner)\b",
    re.I,
)
# Programming languages and runtimes worth treating as a code signal on their own
# when paired with a code verb.
CODE_LANGUAGE = re.compile(
    r"\b(python|py|javascript|js|typescript|ts|rust|go|golang|c\+\+|cpp|c#|java|"
    r"bash|shell|sh|zsh|ruby|php|perl|sql|html|css|react|node|node\.js)\b",
    re.I,
)


# A code request that plausibly depends on external or current information, so
# the model should look it up before writing. Without this, "write a script
# using the latest X API" or "a recon script covering current techniques" would
# be answered from stale weights.
CODE_NEEDS_LOOKUP = re.compile(
    r"\b(latest|current|recent|newest|modern|up[- ]?to[- ]?date|today|this year|"
    r"as of|version|changelog|"
    # Security-research nouns: these move fast and benefit from current sources.
    r"recon|reconnaissance|exploit|exploits|vulnerabilit|cve|payload|"
    r"enumeration|privilege escalation|pentest|attack)\b",
    re.I,
)

# Time-sensitive signal: the same notion the router is told to search on (today's
# events, prices, scores, the latest version of something, a named current
# holder). Used to let a single self-issued search through on an answer-routed
# turn when the question genuinely needs fresh, external facts, while still
# blocking reflexive searches for general knowledge.
TIME_SENSITIVE = re.compile(
    r"\b(today|todays|tonight|yesterday|right now|currently|current|latest|"
    r"newest|recent|recently|this (?:week|month|year|morning|season)|"
    r"as of|so far this|"
    r"price|prices|cost|stock|shares|exchange rate|rate today|"
    r"news|headline|breaking|score|scores|standings|fixture|"
    r"weather|forecast|temperature|"
    r"release|released|version|update|changelog|"
    r"who is the (?:current|new)|latest version|out yet|release date)\b",
    re.I,
)


# Implicit chat feedback. A short reply that is essentially praise or a plain
# rejection is a rating of the previous answer: "good job" approves it as a
# training example, "no, wrong" marks it as a bad answer. Kept conservative so a
# substantive message that merely starts with "no" is not misread; longer
# messages must contain an explicit multi-word phrase.
_POS_LEAD = re.compile(
    r"^\s*(?:that'?s\s+)?(great|perfect|excellent|awesome|amazing|nice|good|"
    r"correct|exactly|right|yes+|yep|yeah|thanks?|thank\s+you|ty|helpful|"
    r"brilliant|love\s+it|works|spot\s+on)\b", re.I)
_NEG_LEAD = re.compile(
    r"^\s*(?:no+|nope|nah|wrong|incorrect|bad|false|not\s+(?:right|correct|quite|good))\b",
    re.I)
_POS_PHRASE = re.compile(
    r"\b(good\s+job|well\s+done|that'?s\s+(?:right|correct|perfect)|exactly\s+right|"
    r"perfect\s+answer|that\s+works|great\s+answer|that'?s\s+it)\b", re.I)
_NEG_PHRASE = re.compile(
    r"\b(that'?s\s+(?:wrong|incorrect|not\s+right|false)|wrong\s+answer|"
    r"you'?re\s+wrong|not\s+(?:what\s+i|correct)|that'?s\s+not\s+it|bad\s+answer)\b",
    re.I)
# Guards: leading tokens that look negative/positive but are not feedback.
_NOT_FEEDBACK = re.compile(
    r"^\s*(?:no\s+(?:way|idea|one|problem|worries|clue)|not\s+sure|"
    r"right\s+(?:now|away)|yes\s+(?:and|but|please)\b)", re.I)


def classify_implicit_feedback(message: str) -> int | None:
    """+1 if the message praises the prior answer, -1 if it rejects it, else None."""
    m = (message or "").strip()
    if not m:
        return None
    if _NOT_FEEDBACK.match(m):
        return None
    short = len(m.split()) <= 6
    if _POS_PHRASE.search(m):
        return 1
    if _NEG_PHRASE.search(m):
        return -1
    if short and _POS_LEAD.match(m):
        return 1
    if short and _NEG_LEAD.match(m):
        return -1
    return None


def build_reusable_dataset(rows: list[dict], fmt: str, system_prompt: str = "") -> tuple[str, int]:
    """Turn feedback rows into one of several reusable dataset formats.

    The point is portability: the data you collect should train ANY model later,
    not just this one. Formats:

    - "chat": messages with this assistant's system prompt. Trains a model to be
      THIS assistant. What the built-in LoRA loop uses.
    - "bare": messages with NO system prompt, just user/assistant. Model-neutral;
      use it to train a different base model or a different persona.
    - "preference": {prompt, chosen, rejected} triples for DPO-style preference
      tuning, built from corrected answers and from good/bad answers to the same
      prompt. This is what makes the rejected ("no, wrong") examples pay off.
    - "raw": every column as JSONL. A lossless archive you can reshape into any
      format in the future.

    Returns (text, example_count).
    """
    lines: list[str] = []

    if fmt == "raw":
        for r in rows:
            lines.append(json.dumps(r, ensure_ascii=False, default=str))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    if fmt == "preference":
        seen = set()
        # 1) corrected answers: original is rejected, correction is chosen.
        for r in rows:
            corrected = (r.get("corrected_response") or "").strip()
            original = (r.get("assistant_response") or "").strip()
            prompt = (r.get("user_prompt") or "").strip()
            if corrected and prompt and corrected != original:
                key = (prompt, corrected, original)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": corrected, "rejected": original},
                    ensure_ascii=False))
        # 2) same prompt with an approved answer and a rejected answer.
        approved_by_prompt: dict[str, str] = {}
        rejected_by_prompt: dict[str, str] = {}
        for r in rows:
            prompt = (r.get("user_prompt") or "").strip()
            ans = (r.get("assistant_response") or "").strip()
            if not prompt or not ans:
                continue
            if r.get("approved_for_training"):
                approved_by_prompt.setdefault(prompt, ans)
            elif (r.get("rating") or 0) < 0:
                rejected_by_prompt.setdefault(prompt, ans)
        for prompt, chosen in approved_by_prompt.items():
            rejected = rejected_by_prompt.get(prompt)
            if rejected and rejected != chosen:
                key = (prompt, chosen, rejected)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    # "chat" or "bare": supervised messages. Prefer the corrected answer as the
    # target when present; skip rejected-only rows (nothing good to imitate).
    seen_msgs = set()
    for r in rows:
        prompt = (r.get("user_prompt") or "").strip()
        target = (r.get("corrected_response") or r.get("assistant_response") or "").strip()
        if not prompt or not target:
            continue
        if not r.get("approved_for_training") and not (r.get("corrected_response") or "").strip():
            continue  # a bad answer with no correction is not a target to imitate
        key = (prompt, target)
        if key in seen_msgs:
            continue
        seen_msgs.add(key)
        messages = []
        if fmt == "chat" and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": target})
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


def is_time_sensitive(message: str) -> bool:
    """True if a question plausibly needs current/external facts to answer well.

    Deliberately errs toward the router's own wording so the gate and the router
    agree on what 'needs a lookup' means.
    """
    return bool(TIME_SENSITIVE.search(message or ""))


# Framing words to strip so the search topic is the subject, not "write a python
# script to ...". Applied only when building a query for a code lookup.
_CODE_FRAMING = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:write|create|generate|build|implement|make|code|program|give me|show me)\s+"
    r"(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:python|py|javascript|js|typescript|ts|rust|go|golang|bash|shell|sh|ruby|"
    r"php|perl|sql|c\+\+|cpp|java)?\s*"
    r"(?:script|program|function|snippet|tool|cli|code|app|module|class)?\s*"
    r"(?:that|which|to|for|attempting|covering|using|demonstrating|showing)?\s*",
    re.I,
)


def code_search_topic(message: str) -> str:
    """Strip 'write a python script to ...' framing down to the search subject."""
    topic = _CODE_FRAMING.sub("", (message or "").strip()).strip(" .?!\t")
    return topic or (message or "").strip()[:200]


# A hard analytical question that benefits from being broken into steps. Requires
# an analytical signal (compare, why, how would, evaluate, design, tradeoffs...)
# AND some heft (length, several clauses, or an explicit "step by step"), so it
# does not fire on simple factual or definitional questions that answer in one
# shot. Lookups and code are excluded upstream, so this only sees "answer"-class
# questions.
REASONING_SIGNAL = re.compile(
    r"\b(compare|contrast|versus|vs\.?|trade[- ]?offs?|pros and cons|"
    r"why (?:is|are|does|do|would|should|did)|how would|how do i|how should|"
    r"analy[sz]e|evaluate|assess|weigh|design|architect|derive|prove|"
    r"implications|consequences|reason through|think through|step by step|"
    r"walk me through|work out|figure out|explain why|justify|"
    r"what (?:would|if) )\b",
    re.I,
)


def is_reasoning_question(message: str) -> bool:
    """True if a question is worth decomposing into incremental reasoning steps."""
    text = (message or "").strip()
    signals = REASONING_SIGNAL.findall(text)
    if not signals:
        return False
    # Two or more analytical cues (e.g. "compare ... tradeoffs ... vs") means a
    # genuinely multi-faceted question regardless of length.
    if len(signals) >= 2:
        return True
    # A single cue plus an explicit step-by-step request, length, or several
    # clauses. A lone short cue ("why is the sky blue") stays one-shot.
    if re.search(r"step by step|think through|walk me through", text, re.I):
        return True
    clauses = text.count(" and ") + text.count(", ") + text.count("?")
    return len(text) >= 80 or clauses >= 2


def is_code_request(message: str) -> bool:
    """True if the message asks to write, fix, or modify code.

    Requires a code verb plus either a code object (script, function, ...) or a
    language name, or a fenced code block in the message. Kept deliberately
    strict so "write up the latest news" (verb, but no code object or language)
    does not match and can still be routed to a search.
    """
    text = message or ""
    if "```" in text:
        return True
    has_verb = bool(CODE_INTENT.search(text))
    has_object = bool(CODE_OBJECT.search(text))
    has_language = bool(CODE_LANGUAGE.search(text))
    # A code verb plus an object or a language ("write a python script"), or a
    # language and an object together even without an imperative verb ("python
    # script to pull CVEs"), both count as a code request.
    return (has_verb and (has_object or has_language)) or (has_language and has_object)


# Result lines from _web_search look like "N. Title\n   URL\n   snippet". Pull
# the result URLs in order so the retrieval pipeline can fetch the top ones.
_RESULT_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.M)


# Aggregator, listing and JS-shell URLs whose fetched HTML is mostly navigation
# and script, not article text. Extracting from them wastes a fetch-and-read
# cycle and, on 8GB, the junk-filled prompt is what stalls. Skip them and use the
# next result (or the search snippet) instead.
_LOW_VALUE_HOST = re.compile(
    r"(?:^|\.)(?:news\.google\.|news\.yahoo\.|flipboard\.|reddit\.com|"
    r"twitter\.com|x\.com|facebook\.com|pinterest\.|quora\.com)", re.I)
_LOW_VALUE_PATH = re.compile(
    r"/(?:category|categories|tag|tags|topics?|section|sections|feed|feeds|"
    r"latest|trending|search|archive)(?:/|$|\?)", re.I)


def is_low_value_url(url: str) -> bool:
    """True for aggregator/listing/JS-shell pages unlikely to yield article text."""
    try:
        from urllib.parse import urlparse
        parts = urlparse(url)
    except Exception:
        return False
    host = parts.netloc.lower()
    path = parts.path or "/"
    if _LOW_VALUE_HOST.search(host):
        return True
    if _LOW_VALUE_PATH.search(path):
        return True
    # A bare domain root (no real path) is a homepage/shell, not an article.
    if path in ("", "/") and not parts.query:
        return True
    return False


def extractable_text_len(page: str) -> int:
    """Rough count of article-like text in a fetched page (already tag-stripped).

    Aggregator shells strip down to a pile of short link fragments; a real
    article has long prose lines. Count characters only from lines that read like
    prose (long, or sentence-punctuated) so a wall of two-word nav links scores
    near zero even when the raw length is large.
    """
    total = 0
    for line in (page or "").splitlines():
        line = line.strip()
        if len(line) >= 60 or (len(line) >= 30 and any(c in line for c in ".!?")):
            total += len(line)
    return total


def is_thin_page(page: str, min_chars: int = 400) -> bool:
    """True if a fetched page has too little real prose to be worth extracting."""
    return extractable_text_len(page) < min_chars


def top_result_urls(search_text: str, limit: int) -> list[str]:
    """The first `limit` result URLs from a web_search result block, de-duped."""
    seen: list[str] = []
    for match in _RESULT_URL.finditer(search_text or ""):
        url = match.group(1)
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-character chunks with a little overlap.

    Overlap keeps a sentence that straddles a boundary from being lost to both
    chunks. Prefers to break on a newline or space near the boundary so chunks
    fall on natural seams rather than mid-word.
    """
    text = text or ""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the nearest newline or space within the last 15%.
            window = text.rfind("\n", start + int(size * 0.85), end)
            if window == -1:
                window = text.rfind(" ", start + int(size * 0.85), end)
            if window != -1:
                end = window
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def quick_tool(message: str) -> tuple[str, dict] | None:
    """Deterministic shortcuts that need no model call at all.

    Only two cases qualify: a message that is purely a URL, and one that is
    purely arithmetic. Both are unambiguous and common enough that spending a
    router call on them would be wasteful. Everything else returns None and is
    handled by the model router.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    url = BARE_URL.match(text)
    if url:
        return "fetch_url", {"url": url.group(1)}
    expression = text.rstrip("=?").strip()
    # Require an operator and a digit so a bare number or a year ("2026") is not
    # mistaken for arithmetic.
    if (ARITHMETIC_ONLY.match(expression) and any(op in expression for op in "+-*/^%")
            and any(char.isdigit() for char in expression)):
        return "calculator", {"expression": expression.replace("^", "**").replace(",", "")}
    return None


# Backwards-compatible alias. Older call sites and tests refer to fast_path_call;
# it now covers only the deterministic shortcuts. The prev_user parameter is kept
# for signature compatibility but is unused, because follow-up resolution ("look
# it up") is now handled by the model router, which sees the conversation.
def fast_path_call(message: str, prev_user: str | None = None) -> tuple[str, dict] | None:
    return quick_tool(message)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'CODE_INTENT',
    'CODE_LANGUAGE',
    'CODE_NEEDS_LOOKUP',
    'CODE_OBJECT',
    'REASONING_SIGNAL',
    'TIME_SENSITIVE',
    '_CODE_FRAMING',
    '_LOW_VALUE_HOST',
    '_LOW_VALUE_PATH',
    '_NEG_LEAD',
    '_NEG_PHRASE',
    '_NOT_FEEDBACK',
    '_POS_LEAD',
    '_POS_PHRASE',
    '_RESULT_URL',
    '_is_trivial',
    '_last_user_turn',
    'build_reusable_dataset',
    'chunk_text',
    'classify_implicit_feedback',
    'code_search_topic',
    'extractable_text_len',
    'fast_path_call',
    'is_code_request',
    'is_low_value_url',
    'is_reasoning_question',
    'is_substantive',
    'is_thin_page',
    'is_time_sensitive',
    'quick_tool',
    'top_result_urls',
]
