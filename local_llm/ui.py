"""The embedded single-page UI and its render/validation helpers.

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
from .ui_styles import UI_CSS
from .ui_markup import UI_BODY, UI_HEAD, UI_TAIL
from .ui_script_chat import UI_JS_CHAT
from .ui_script_models import UI_JS_MODELS
from .ui_script_panels import UI_JS_PANELS
from .ui_script_tasks import UI_JS_TASKS
from .ui_script_views import UI_JS_VIEWS
from .ui_script_auth import UI_JS_AUTH


def render_ui() -> str:
    """Return the HTML the browser receives, with the build marker substituted.

    HTML_PAGE is a raw string on purpose. In a normal Python string, a `\\n`
    written inside embedded JavaScript becomes a real line break, which splits a
    JS string literal across two lines and makes the whole <script> fail to
    parse. The page then renders but nothing works: the status panel sits on its
    hardcoded "Starting..." text and the Send button does nothing.
    """
    # Render the logo as an <img> when it looks like a URL/path, otherwise inline
    # it as text or an emoji. Escape the name so a stray < in APP_NAME cannot
    # break the header markup.
    import html as _html
    name = _html.escape(APP_NAME)
    logo = APP_LOGO.strip()
    if logo.startswith(("http://", "https://", "/")):
        logo_html = f'<img src="{_html.escape(logo)}" alt="{name} logo" class="brand-logo">'
    elif logo:
        logo_html = f'<span class="brand-mark">{_html.escape(logo)}</span>'
    else:
        logo_html = '<span class="brand-mark">◆</span>'
    return (HTML_PAGE
            .replace("{{UI_BUILD}}", UI_BUILD)
            .replace("{{APP_NAME}}", name)
            .replace("{{APP_LOGO}}", logo_html))


def check_ui_syntax() -> list[str]:
    """Cheap structural check on the rendered <script>, no Node required.

    Catches the failure above by scanning the script as a character stream and
    reporting a real newline inside a single- or double-quoted literal. The
    previous version counted quotes per line, which flagged any correct line
    containing an apostrophe ("it's") and then refused to start the server. A
    scanner that tracks comments, escapes and template literals has no such
    false positive. Returns a list of problems; empty means well formed.
    """
    script = re.search(r"<script>(.*?)</script>", render_ui(), re.S)
    if not script:
        return ["no <script> block found in HTML_PAGE"]
    return scan_js_strings(script.group(1))


def scan_js_strings(body: str) -> list[str]:
    """Report string literals broken by a real newline, and unclosed comments."""
    problems: list[str] = []
    quote: str | None = None
    quote_line = 0
    line = 1
    index = 0
    length = len(body)

    while index < length:
        char = body[index]
        nxt = body[index + 1] if index + 1 < length else ""

        if char == "\n":
            line += 1
            if quote in ('"', "'"):
                problems.append(f"line {quote_line}: unterminated {quote} string literal")
                quote = None
            index += 1
            continue

        if quote is None:
            if char == "/" and nxt == "/":
                while index < length and body[index] != "\n":
                    index += 1
                continue
            if char == "/" and nxt == "*":
                end = body.find("*/", index + 2)
                if end == -1:
                    problems.append(f"line {line}: unterminated block comment")
                    break
                line += body.count("\n", index, end)
                index = end + 2
                continue
            if char in ('"', "'", "`"):
                quote = char
                quote_line = line
                index += 1
                continue
            if char == "/":
                # A regex literal. Distinguish it from division by looking at the
                # previous significant character: after a value (identifier, digit,
                # closing bracket) a slash is division; otherwise it starts a regex.
                prev = ""
                back = index - 1
                while back >= 0 and body[back] in " \t\n":
                    back -= 1
                if back >= 0:
                    prev = body[back]
                if prev and (prev.isalnum() or prev in ")]_$"):
                    index += 1
                    continue
                scan = index + 1
                in_class = False
                while scan < len(body):
                    ch = body[scan]
                    if ch == "\\":
                        scan += 2
                        continue
                    if ch == "\n":
                        break
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        break
                    scan += 1
                if scan < len(body) and body[scan] == "/":
                    line += body.count("\n", index, scan)
                    index = scan + 1
                    continue
            index += 1
            continue

        if char == "\\":
            index += 2
            continue
        if char == quote:
            quote = None
        index += 1

    if quote is not None:
        problems.append(f"line {quote_line}: unterminated {quote} string literal at end of script")
    return problems

HTML_PAGE = (
    UI_HEAD
    + UI_CSS
    + UI_BODY
    # One <script> block, assembled from parts. JavaScript hoists function
    # declarations across the whole script, so splitting the source here changes
    # nothing about how it runs — the browser still receives a single script.
    + UI_JS_CHAT
    + "\n" + UI_JS_VIEWS
    + "\n" + UI_JS_TASKS
    + "\n" + UI_JS_MODELS
    + "\n" + UI_JS_AUTH
    + "\n" + UI_JS_PANELS
    + UI_TAIL
)



# Request/response models are bound at module scope by _define_api_models().
# They cannot be plain module-level class statements because pydantic is not
# installed until bootstrap() has run.
ChatRequest: Any = None
FeedbackRequest: Any = None
ChatResponse: Any = None
ConfigRequest: Any = None
ToolRequest: Any = None
MemoryRequest: Any = None
TaskRequest: Any = None
TaskUpdateRequest: Any = None
ModelSelectRequest: Any = None



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ChatRequest',
    'ChatResponse',
    'ConfigRequest',
    'FeedbackRequest',
    'HTML_PAGE',
    'MemoryRequest',
    'ModelSelectRequest',
    'TaskRequest',
    'TaskUpdateRequest',
    'ToolRequest',
    'check_ui_syntax',
    'render_ui',
    'scan_js_strings',
]
