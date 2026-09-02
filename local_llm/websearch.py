"""Web search backend, URL guards, and HTML stripping.

No API keys are required. Search uses DuckDuckGo's public HTML endpoint with
the lite endpoint as a fallback.

Split out of the original single-file deploy.py; public names are kept
compatible with the original module.
"""

from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
import urllib.parse
from dataclasses import dataclass

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


TOOL_TRAINING_PREAMBLE = (
    "You call tools by replying with a single JSON object and nothing else, in "
    'the form {"tool": "tool_name", "args": {"arg": "value"}}.'
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)

# Hard ceiling on a single fetch_url download, independent of tool_result_chars.
MAX_FETCH_BYTES = 2_000_000

_ALLOWED_FETCH_SCHEMES = {"http", "https"}


def guard_public_url(url: str) -> None:
    """Reject URLs that resolve to localhost or private/reserved networks.

    DNS is resolved before the request so hostnames pointing at private
    addresses are rejected too.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme.lower() not in _ALLOWED_FETCH_SCHEMES:
        raise ValueError(
            f"unsupported URL scheme: {parsed.scheme!r}; "
            "only http and https are allowed"
        )

    host = parsed.hostname
    if not host:
        raise ValueError("url has no host")

    # Reject obvious IP literals before DNS resolution.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        _reject_non_public_address(host, literal)
        return

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"invalid URL port: {exc}") from exc

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve {host}: {exc}") from exc

    if not infos:
        raise ValueError(f"could not resolve {host}")

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        _reject_non_public_address(host, address)


def _reject_non_public_address(host: str, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Reject addresses that must never be fetched by an external URL tool."""
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ValueError(
            f"refusing to fetch {host} ({address}): "
            "private or non-public address"
        )


def strip_html(raw: str) -> str:
    """Turn an HTML fragment into readable plain text."""
    raw = re.sub(
        r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>",
        " ",
        raw,
    )
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(
        r"(?i)</(p|div|li|tr|h[1-6])>",
        "\n",
        raw,
    )
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*", "\n\n", raw)
    return raw.strip()


def _ddg_unwrap(href: str) -> str:
    """Unwrap a DuckDuckGo redirect URL."""
    if href.startswith("//"):
        href = "https:" + href

    parsed = urllib.parse.urlparse(href)

    if (
        parsed.netloc.endswith("duckduckgo.com")
        and parsed.path.startswith("/l/")
    ):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)

    return href


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchBackend:
    """API-key-free web search.

    DuckDuckGo's public HTML endpoint is the primary backend. Its lightweight
    endpoint is used as a fallback when the primary endpoint changes or fails.

    The old Brave/Tavily API-key configuration is intentionally no longer
    supported.
    """

    ENDPOINTS = (
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    )

    def __init__(self, config: Config):
        self.config = config

    def search(
        self,
        query: str,
        num_results: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        count = num_results or self.config.search_results
        count = max(1, min(count, 50))

        return self._ddg(query.strip(), count)

    def _client(self):
        import httpx

        return httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def _ddg(self, query: str, count: int) -> list[SearchResult]:
        last_error = "no response"

        with self._client() as client:
            for endpoint in self.ENDPOINTS:
                try:
                    response = client.post(
                        endpoint,
                        data={
                            "q": query,
                            "kl": "wt-wt",
                        },
                    )
                    response.raise_for_status()
                except Exception as exc:
                    last_error = str(exc)
                    continue

                results = self._parse_ddg(response.text, count)

                if results:
                    return results

                last_error = "no parsable results"

        raise RuntimeError(
            f"DuckDuckGo search failed: {last_error}"
        )

    @staticmethod
    def _parse_ddg(page: str, count: int) -> list[SearchResult]:
        """Parse both DDG HTML and lite result formats."""
        results: list[SearchResult] = []

        # Standard DDG HTML endpoint.
        pattern = re.compile(
            r"""
            <a
                [^>]*?
                class=["'][^"']*\bresult__a\b[^"']*["']
                [^>]*?
                href=["']([^"']+)["']
                [^>]*?
            >
                (.*?)
            </a>
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        for match in pattern.finditer(page):
            url = _ddg_unwrap(html.unescape(match.group(1)))
            title = strip_html(match.group(2))

            if not url or not title:
                continue

            snippet = SearchBackend._find_ddg_snippet(
                page,
                match.end(),
            )

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                )
            )

            if len(results) >= count:
                return results

        if results:
            return results

        # DDG lite fallback.
        for match in re.finditer(
            r'<a[^>]+href=["\'](http[^"\']+)["\'][^>]*>(.*?)</a>',
            page,
            re.IGNORECASE | re.DOTALL,
        ):
            url = _ddg_unwrap(html.unescape(match.group(1)))
            title = strip_html(match.group(2))

            if not url or not title:
                continue

            parsed = urllib.parse.urlparse(url)

            # Avoid returning DDG's own navigation links.
            if parsed.netloc.endswith("duckduckgo.com"):
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet="",
                )
            )

            if len(results) >= count:
                break

        return results

    @staticmethod
    def _find_ddg_snippet(page: str, start: int) -> str:
        """Extract the next standard DDG result snippet, if present."""
        remaining = page[start:]

        match = re.search(
            r"""
            class=["'][^"']*\bresult__snippet\b[^"']*["']
            [^>]*>
            (.*?)
            </(?:a|div)
            """,
            remaining,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        if not match:
            return ""

        return strip_html(match.group(1))


# Explicitly preserve the original flat-module exports.
__all__ = [
    "MAX_FETCH_BYTES",
    "SearchBackend",
    "SearchResult",
    "TOOL_TRAINING_PREAMBLE",
    "USER_AGENT",
    "_ddg_unwrap",
    "guard_public_url",
    "strip_html",
]
