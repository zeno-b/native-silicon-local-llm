"""Web search backends, URL guards and HTML stripping.

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


TOOL_TRAINING_PREAMBLE = (
    "You call tools by replying with a single JSON object and nothing else, in "
    'the form {"tool": "tool_name", "args": {"arg": "value"}}.'
)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"

# Hard ceiling on a single fetch_url download, independent of tool_result_chars.
MAX_FETCH_BYTES = 2_000_000


def guard_public_url(url: str) -> None:
    """Reject URLs that resolve to the local machine or a private network.

    Without this, a page the agent fetches can instruct it to fetch
    http://127.0.0.1:8000/api/config or a LAN device, and the model will comply:
    the tool loop treats page text as input, and small models follow it. The
    check runs against the resolved address, so a hostname pointing at 127.0.0.1
    is caught too.
    """
    import ipaddress

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("url has no host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve {host}: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast):
            raise ValueError(
                f"refusing to fetch {host} ({address}): private or loopback address. "
                "Set ALLOW_LOCAL_FETCH=1 if this is deliberate."
            )


def strip_html(raw: str) -> str:
    """Turn a HTML fragment into readable plain text."""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*", "\n\n", raw)
    return raw.strip()


def _ddg_unwrap(href: str) -> str:
    """DuckDuckGo hands back a redirect wrapper; pull the real target out."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchBackend:
    """Web search with pluggable providers.

    ddg needs no key and is the default. It scrapes the HTML endpoint, so it can
    break without warning when DuckDuckGo changes markup; brave, tavily and
    searxng are the stable options when a key or a local instance is available.
    """

    def __init__(self, config: Config):
        self.config = config

    def search(self, query: str, num_results: int | None = None) -> list[SearchResult]:
        count = num_results or self.config.search_results
        backend = (self.config.search_backend or "ddg").lower()
        if backend == "brave":
            return self._brave(query, count)
        if backend == "tavily":
            return self._tavily(query, count)
        if backend == "searxng":
            return self._searxng(query, count)
        return self._ddg(query, count)

    def _client(self):
        import httpx
        return httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def _ddg(self, query: str, count: int) -> list[SearchResult]:
        endpoints = [
            "https://html.duckduckgo.com/html/",
            "https://lite.duckduckgo.com/lite/",
        ]
        last_error = "no response"
        with self._client() as client:
            for endpoint in endpoints:
                try:
                    resp = client.post(endpoint, data={"q": query, "kl": "wt-wt"})
                except Exception as exc:
                    last_error = str(exc)
                    continue
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}"
                    continue
                results = self._parse_ddg(resp.text, count)
                if results:
                    return results
                last_error = "no parsable results"
        raise RuntimeError(f"DuckDuckGo search failed: {last_error}")

    @staticmethod
    def _parse_ddg(page: str, count: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'(?:.*?class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
            re.S,
        )
        for match in pattern.finditer(page):
            url = _ddg_unwrap(match.group(1))
            title = strip_html(match.group(2) or "")
            snippet = strip_html(match.group(3) or "")
            if url and title:
                results.append(SearchResult(title, url, snippet))
            if len(results) >= count:
                break
        if results:
            return results
        # lite.duckduckgo.com uses a plain table, no result__a class at all.
        for match in re.finditer(r'<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', page, re.S):
            url = _ddg_unwrap(match.group(1))
            title = strip_html(match.group(2))
            if "duckduckgo.com" in url or not title:
                continue
            results.append(SearchResult(title, url, ""))
            if len(results) >= count:
                break
        return results

    def _brave(self, query: str, count: int) -> list[SearchResult]:
        key = os.environ.get("BRAVE_API_KEY", "")
        if not key:
            raise RuntimeError("SEARCH_BACKEND=brave but BRAVE_API_KEY is not set.")
        with self._client() as client:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"Accept": "application/json", "X-Subscription-Token": key},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [
            SearchResult(
                item.get("title", ""),
                item.get("url", ""),
                strip_html(item.get("description", "")),
            )
            for item in payload.get("web", {}).get("results", [])[:count]
        ]

    def _tavily(self, query: str, count: int) -> list[SearchResult]:
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise RuntimeError("SEARCH_BACKEND=tavily but TAVILY_API_KEY is not set.")
        with self._client() as client:
            resp = client.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": count},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [
            SearchResult(item.get("title", ""), item.get("url", ""), item.get("content", ""))
            for item in payload.get("results", [])[:count]
        ]

    def _searxng(self, query: str, count: int) -> list[SearchResult]:
        base = os.environ.get("SEARXNG_URL", "").rstrip("/")
        if not base:
            raise RuntimeError("SEARCH_BACKEND=searxng but SEARXNG_URL is not set.")
        with self._client() as client:
            resp = client.get(
                f"{base}/search",
                params={"q": query, "format": "json"},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [
            SearchResult(item.get("title", ""), item.get("url", ""), item.get("content", ""))
            for item in payload.get("results", [])[:count]
        ]


# Ceilings for the calculator. Exponentiation and factorial are the only
# whitelisted operations whose cost is not bounded by the length of the
# expression: 9**9**9 is seven characters and allocates until the kernel kills



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'MAX_FETCH_BYTES',
    'SearchBackend',
    'SearchResult',
    'TOOL_TRAINING_PREAMBLE',
    'USER_AGENT',
    '_ddg_unwrap',
    'guard_public_url',
    'strip_html',
]
