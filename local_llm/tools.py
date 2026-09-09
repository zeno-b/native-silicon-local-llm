"""Tool definitions and the registry the agent calls into.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .calculator import *  # noqa: F401,F403


def guarded_thread(fn, *args, **kwargs) -> threading.Thread:
    """A daemon thread whose target can never die silently.

    A bare threading.Thread(target=fn) that raises leaves no trace and silently
    stops whatever background work it was doing (a restart, a retrain). This
    wraps the target so any exception is logged with a traceback instead.
    """
    def _target():
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - top of a thread, must catch all
            log(f"Background thread {getattr(fn, '__name__', fn)!r} failed: {exc}",
                logging.ERROR)
            log(traceback.format_exc(), logging.DEBUG)
    return threading.Thread(target=_target, daemon=True)


def resolve_in_workspace(path: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside ./workspace."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = (WORKSPACE_DIR / path.lstrip("/")).resolve()
    workspace = WORKSPACE_DIR.resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError("path escapes the workspace directory")
    return candidate


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    required: list[str]
    handler: Callable[..., str]
    # --- Routing metadata (all optional, so old Tool(...) calls still work) ---
    # When routable is True, the model router may choose this tool by name and
    # supply its arguments. route_hint is the one-line menu entry shown to the
    # router; without it the tool is callable in the agent loop but not offered
    # as a routing action. This is what makes new capabilities future-proof: add
    # a tool with these two fields and it is routable with no new routing code.
    routable: bool = False
    route_hint: str | None = None
    # terminal tools produce the answer themselves (a number, a page), so their
    # result is returned directly. Non-terminal tools return reference material
    # the model must read, so the result is seeded and the model then answers.
    terminal: bool = False
    # Directive appended after a non-terminal tool's result, telling the model
    # what to do with it. A sensible default is used when this is None.
    seed_directive: str | None = None

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
        }


class ToolRegistry:
    """The tools the agent can call, built from the live config."""

    # Argument names small models reach for that are not the ones in the spec.
    ARG_ALIASES = {
        "q": "query", "search_query": "query", "search": "query", "keywords": "query",
        "link": "url", "href": "url", "uri": "url", "address": "url",
        "file": "path", "filename": "path", "file_path": "path", "filepath": "path",
        "dir": "path", "directory": "path", "folder": "path",
        "text": "content", "body": "content", "data": "content",
        "expr": "expression", "equation": "expression", "math": "expression",
        "cmd": "command", "shell": "command", "script": "code", "source": "code",
        "old": "find", "old_str": "find", "new": "replace", "new_str": "replace",
        "regex": "pattern", "tz": "timezone", "name": "key", "note": "value",
        "response": "answer", "result": "answer", "final": "answer",
    }

    def __init__(self, config: Config, db: Database | None = None):
        self.config = config
        self.db = db
        self.search = SearchBackend(config)
        self._tools: dict[str, Tool] = {}
        # Files the tools have created or modified this session, so you can see at
        # a glance what changed before reviewing with git.
        self.changed_files: set[str] = set()
        # Set per request so memory writes can record where they came from.
        self.conversation_id: str | None = None
        self._register_defaults()

    def normalise_args(self, tool: Tool, args: dict) -> dict:
        """Map common argument-name mistakes onto the tool's real parameters.

        A 0.5B model calls web_search with {"q": ...} often enough that dropping
        the argument and reporting a missing one wastes a whole agent step.
        """
        clean: dict[str, Any] = {}
        for key, value in args.items():
            if key in tool.parameters:
                clean[key] = value
                continue
            alias = self.ARG_ALIASES.get(str(key).lower())
            if alias and alias in tool.parameters and alias not in clean:
                clean[alias] = value
        # A single unnamed value against a single-required-argument tool.
        if not clean and len(tool.required) == 1 and len(args) == 1:
            only = next(iter(args.values()))
            if isinstance(only, (str, int, float)):
                clean[tool.required[0]] = only
        return clean

    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def routable(self) -> list[Tool]:
        """Tools the model router is allowed to choose, in registration order."""
        return [t for t in self._tools.values() if t.routable and t.route_hint]

    def _register_defaults(self) -> None:
        self._add(Tool(
            name="web_search",
            description="Search the web and return titles, URLs and snippets. Use for anything current or outside your training data.",
            parameters={"query": "search terms", "num_results": "how many results, 1-10, default 5"},
            required=["query"],
            handler=self._web_search,
            routable=True,
            route_hint=('{"action":"web_search","query":"<good search terms>"} if it '
                        "needs current events, real-time data, recent releases or "
                        "versions, prices, scores, or facts that may have changed."),
            seed_directive=("Answer my original question using these results and cite "
                            "the URLs. If the snippets lack the detail needed, call "
                            "fetch_url on the most relevant URL. Do not repeat the search."),
        ))
        self._add(Tool(
            name="fetch_url",
            description="Download a web page and return its readable text. Use after web_search when a snippet is not enough.",
            parameters={"url": "absolute http or https URL", "max_chars": "truncate the page to this many characters"},
            required=["url"],
            handler=self._fetch_url,
        ))
        self._add(Tool(
            name="weather",
            description=(
                "Get the current conditions and multi-day forecast for a place. "
                "Use this for any weather question instead of web_search: it returns "
                "actual temperatures and conditions, which search snippets do not."
            ),
            parameters={"location": "city or place name, for example Brussels",
                        "when": "today, tomorrow, or a weekday; default today"},
            required=["location"],
            handler=self._weather,
            routable=True,
            route_hint=('{"action":"weather","location":"<place>","when":"today|'
                        'tomorrow|<weekday>"} for any weather or forecast request.'),
            seed_directive="Answer my original question from this forecast.",
        ))
        self._add(Tool(
            name="calculator",
            description="Evaluate an arithmetic expression. Supports + - * / % ** and functions such as sqrt, log, sin, cos.",
            parameters={"expression": "for example (2+3)*sqrt(16)"},
            required=["expression"],
            handler=self._calculator,
        ))
        self._add(Tool(
            name="current_time",
            description="Return the current date and time.",
            parameters={"timezone": "IANA name such as Europe/Brussels, default UTC"},
            required=[],
            handler=self._current_time,
        ))
        self._add(Tool(
            name="list_files",
            description="List files in the project directory (the local codebase, or the sandbox workspace if none is set).",
            parameters={
                "path": "subdirectory, default the workspace root",
                "recursive": "true to walk subdirectories, default false",
            },
            required=[],
            handler=self._list_files,
            routable=bool(self.config.project_dir),
            route_hint=('{"action":"list_files","recursive":"true"} if the user asks '
                        "about their project, codebase or repository in general terms "
                        '("what is in my project", "fix this code") and you do not yet '
                        "know which files exist. ALWAYS start here for codebase work."),
        ))
        self._add(Tool(
            name="read_file",
            description=("Read a file from the project directory. Handles text and code, and also "
                         "previews structured and document types: .csv/.tsv (columns + sample rows), "
                         ".json (pretty or shape), .ipynb (cells), .pdf and .docx (extracted text if "
                         "the reader library is installed). Binary files are described, not dumped."),
            parameters={"path": "file path relative to the project"},
            required=["path"],
            handler=self._read_file,
            routable=bool(self.config.project_dir),
            route_hint=('{"action":"read_file","path":"<path>"} if the user names a '
                        "specific file in their project to look at, fix or change."),
        ))
        self._add(Tool(
            name="file_info",
            description="Report a file's size, extension, text/binary kind, and line count without dumping its contents.",
            parameters={"path": "file or directory path relative to the project"},
            required=["path"],
            handler=self._file_info,
        ))
        if self.db is not None and getattr(self.db, "fts_enabled", False):
            self._add(Tool(
                name="index_url",
                description=(
                    "Fetch a web page and add it to the knowledge base so it can be searched "
                    "later without fetching again. Use for documentation you will refer back to."
                ),
                parameters={"url": "page to fetch and index"},
                required=["url"],
                handler=self._index_url,
            ))
            self._add(Tool(
                name="search_docs",
                description=(
                    "Search the indexed knowledge base (documents you have added) and return "
                    "the most relevant passages with their source paths. Use this to answer "
                    "questions about the user's own documents instead of guessing."
                ),
                parameters={"query": "what to look for", "limit": "max passages (default 5)"},
                required=["query"],
                handler=self._search_docs,
            ))
            self._add(Tool(
                name="index_docs",
                description=(
                    "Add a file or a whole directory from the project into the knowledge base "
                    "so it can be searched later. Reads PDFs, docx, csv, notebooks and text."
                ),
                parameters={"path": "file or directory to index"},
                required=["path"],
                handler=self._index_docs,
            ))
        self._add(Tool(
            name="write_file",
            description="Create or overwrite a text file in the project directory. Edits happen in place; review with git before pushing.",
            parameters={"path": "file path relative to the workspace", "content": "full file contents"},
            required=["path", "content"],
            handler=self._write_file,
        ))
        self._add(Tool(
            name="edit_file",
            description=(
                "Replace an exact snippet inside a project file. Prefer this over "
                "write_file when changing part of a file you have already read. Edits "
                "happen in place; review with git before pushing."
            ),
            parameters={
                "path": "file path relative to the workspace",
                "find": "exact text to replace, must appear in the file",
                "replace": "replacement text",
                "count": "how many occurrences to replace, default all",
            },
            required=["path", "find", "replace"],
            handler=self._edit_file,
        ))
        self._add(Tool(
            name="search_files",
            description="Search project files for a regular expression and return matching lines.",
            routable=bool(self.config.project_dir),
            route_hint=('{"action":"search_files","pattern":"<regex>"} if the user asks '
                        "where something is defined or used in their project."),
            parameters={
                "pattern": "regular expression",
                "path": "subdirectory to search, default the workspace root",
                "max_results": "how many matching lines, default 40",
            },
            required=["pattern"],
            handler=self._search_files,
        ))
        self._add(Tool(
            name="recall_feedback",
            description="Search stored user feedback for earlier questions and corrected answers.",
            parameters={"query": "text to look for", "limit": "how many rows, default 5"},
            required=["query"],
            handler=self._recall_feedback,
        ))
        self._add(Tool(
            name="remember",
            description=(
                "Store a durable note under a short key. Survives restarts and is "
                "visible in later conversations. Use for facts the user tells you "
                "about themselves, their setup, or their preferences."
            ),
            parameters={"key": "short identifier, for example user.timezone",
                        "value": "the note to store"},
            required=["key", "value"],
            handler=self._remember,
        ))
        self._add(Tool(
            name="recall_memory",
            description="Look up notes stored earlier with remember. Omit the query to list the most recent.",
            parameters={"query": "text to match against keys and values",
                        "limit": "how many notes, default 10"},
            required=[],
            handler=self._recall_memory,
        ))
        self._add(Tool(
            name="forget",
            description="Delete a note stored with remember.",
            parameters={"key": "the key to delete"},
            required=["key"],
            handler=self._forget,
        ))
        self._add(Tool(
            name="final_answer",
            description=(
                "End the loop and give the user your answer. Call this when you have "
                "enough information. The answer argument is shown verbatim."
            ),
            parameters={"answer": "the complete answer, in plain text"},
            required=["answer"],
            handler=lambda answer: str(answer),
        ))
        if self.config.allow_python:
            self._add(Tool(
                name="run_python",
                description="Run a short Python script in the workspace directory and return stdout.",
                parameters={"code": "Python source to execute"},
                required=["code"],
                handler=self._run_python,
            ))
        if self.config.allow_shell:
            self._add(Tool(
                name="run_shell",
                description="Run a shell command in the project directory (confined runner) and return its output.",
                parameters={"command": "shell command"},
                required=["command"],
                handler=self._run_shell,
            ))
            self._add(Tool(
                name="run_tests",
                description=(
                    "Run the project's test suite in the project directory and return the "
                    "output, so you can check whether your edits work and fix failures. "
                    "Auto-detects the command (pytest/npm/cargo/go/make) or set TEST_COMMAND."
                ),
                parameters={"command": "optional explicit test command"},
                required=[],
                handler=self._run_tests,
            ))

        # Office 365 (Microsoft Graph) tools. Registered ONLY when an Azure AD app
        # is configured: every registered tool's description is re-sent in the
        # agent system prompt on every step, so three unusable tools were pure
        # prefill cost on an 8GB machine. The "office365" capability allowlists
        # them once they exist.
        if self._o365_configured():
            self._register_o365()

        allow = {name.strip() for name in (self.config.agent_tools or "").split(",") if name.strip()}
        if allow:
            # Some tools only exist under the right configuration (shell/python
            # execution, a connected Office 365 app). An agent whose capabilities
            # include them is not misconfigured, so their absence is expected and
            # must not be reported as a bad tool name.
            conditional = {"run_shell", "run_python", "run_tests",
                           "o365_mail", "o365_files", "o365_calendar"}
            unknown = allow - set(self._tools) - conditional
            if unknown:
                log(f"AGENT_TOOLS names tools that do not exist: {', '.join(sorted(unknown))}",
                    logging.WARNING)
            # final_answer is the loop's exit condition, never filtered out.
            self._tools = {
                name: tool for name, tool in self._tools.items()
                if name in allow or name == "final_answer"
            }

    def _web_search(self, query: str, num_results: Any = None) -> str:
        try:
            count = int(num_results) if num_results else self.config.search_results
        except (TypeError, ValueError):
            count = self.config.search_results
        count = min(10, max(1, count))
        results = self.search.search(query, count)
        if not results:
            return "No results."
        lines = []
        for index, item in enumerate(results, 1):
            lines.append(f"{index}. {item.title}\n   {item.url}\n   {item.snippet}".rstrip())
        return "\n".join(lines)

    def _fetch_url(self, url: str, max_chars: Any = None) -> str:
        import httpx
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        if not self.config.allow_local_fetch:
            guard_public_url(url)
        limit = self.config.tool_raw_chars
        try:
            if max_chars:
                limit = min(self.config.tool_raw_chars, max(200, int(max_chars)))
        except (TypeError, ValueError):
            pass      # the model passed a non-number: keep the configured cap
        # Read a bounded number of bytes rather than resp.text. An unbounded
        # read of a large file is how a tool call turns into an out-of-memory
        # kill on a machine with 8GB shared between the app and the model.
        byte_cap = min(MAX_FETCH_BYTES, max(4096, limit * 8))
        # Follow redirects MANUALLY so every hop is SSRF-checked BEFORE it is
        # fetched. With httpx's follow_redirects=True the guard only saw the first
        # and the final URL, so a public link that 30x-redirected to
        # 169.254.169.254 or 127.0.0.1:<port> was actually connected to before any
        # validation. Now each Location is validated first, with a hop cap.
        chunks: list[bytes] = []
        total = 0
        content_type = ""
        encoding = "utf-8"
        current = url
        is_pdf = False
        cap_bytes = byte_cap
        with httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _hop in range(10):
                if not self.config.allow_local_fetch:
                    guard_public_url(current)
                with client.stream("GET", current) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        current = str(resp.url.join(resp.headers["location"]))
                        continue  # validate + fetch the next hop on the next loop
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "").lower()
                    # Decide what to read BEFORE reading it. Headers alone rule
                    # out two whole classes of waste: an unsupported type is
                    # rejected without pulling a byte, and HTML/JSON stops at
                    # byte_cap (~160KB) instead of the 2MB PDF ceiling, which is
                    # all that was ever kept anyway. Deciding after the download
                    # meant a large page cost ~12x the bandwidth, wall-clock and
                    # peak memory of the text it contributed.
                    path_only = str(resp.url).lower().split("?", 1)[0].split("#", 1)[0]
                    is_pdf = "pdf" in content_type or path_only.endswith(".pdf")
                    if not is_pdf and content_type and not any(
                        kind in content_type
                        for kind in ("text/", "json", "xml", "html", "javascript", "csv")
                    ):
                        return f"{url}\n\n[skipped: unsupported content type {content_type}]"
                    # A PDF's extractable text is far smaller than its raw bytes,
                    # so it alone is allowed the full MAX_FETCH_BYTES.
                    cap_bytes = MAX_FETCH_BYTES if is_pdf else byte_cap
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= cap_bytes:
                            break
                    encoding = resp.encoding or "utf-8"
                    break
            else:
                raise ValueError("too many redirects")
        data = b"".join(chunks)
        # Binary content with an empty/missing Content-Type slips past the
        # header check above and would decode into replacement-character
        # garbage; detect it by NUL bytes in the body instead.
        if not is_pdf and b"\x00" in data[:4096]:
            return f"{url}\n\n[skipped: binary content]"
        if len(data) > cap_bytes:
            data = data[:cap_bytes]        # the final chunk can overshoot
        total = len(data)
        if is_pdf:
            import io
            text = self._extract_pdf(io.BytesIO(data), limit)
            truncated = "\n\n[truncated: PDF exceeded the fetch byte cap]" if total >= cap_bytes else ""
            return (f"{url}\n\n" + text)[:limit] + truncated
        body = data.decode(encoding, "replace")
        text = strip_html(body) if "html" in content_type or "<" in body[:200] else body
        title = ""
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        if match:
            title = strip_html(match.group(1))
        header = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
        truncated = "\n\n[truncated]" if total >= cap_bytes else ""
        return (header + text)[:limit] + truncated

    def _calculator(self, expression: str) -> str:
        return str(safe_eval(expression))

    def _weather(self, location: str, when: str = "today") -> str:
        """Plain-text forecast from wttr.in, which returns data an LLM can relay.

        Search snippets for "weather" describe weather websites; they carry no
        actual forecast. wttr.in returns the numbers directly as JSON, so the
        model has something to answer from instead of a page of navigation.
        """
        import httpx
        place = (location or "").strip()
        if not place:
            raise ValueError("location must not be empty")
        when_key = (when or "today").strip().lower()
        url = f"https://wttr.in/{urllib.parse.quote(place)}?format=j1"
        try:
            with httpx.Client(timeout=self.config.tool_timeout,
                              headers={"User-Agent": "curl/8"}) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return f"Could not fetch weather for {place}: {exc}"

        days = data.get("weather") or []
        if not days:
            return f"No forecast returned for {place}."
        # Map the request to a day index. wttr.in gives today plus two days.
        index = {"today": 0, "tomorrow": 1}.get(when_key)
        if index is None:
            # A weekday name: match it against each day's date.
            import datetime as _dt
            wanted = when_key[:3]
            index = 0
            for i, day in enumerate(days):
                try:
                    d = _dt.date.fromisoformat(day.get("date", ""))
                    if d.strftime("%a").lower() == wanted:
                        index = i
                        break
                except ValueError:
                    continue
        index = max(0, min(index, len(days) - 1))
        day = days[index]

        label = {0: "today", 1: "tomorrow"}.get(index, day.get("date", ""))
        lines = [f"Weather for {place} ({label}, {day.get('date','')}):"]
        lines.append(
            f"  min {day.get('mintempC','?')}C / max {day.get('maxtempC','?')}C, "
            f"sunrise {day.get('astronomy',[{}])[0].get('sunrise','?')}, "
            f"sunset {day.get('astronomy',[{}])[0].get('sunset','?')}"
        )
        # A few representative hours rather than all 24, to stay inside budget.
        for slot in day.get("hourly", []):
            hour = int(slot.get("time", "0") or 0) // 100
            if hour not in (9, 12, 15, 18):
                continue
            desc = (slot.get("weatherDesc") or [{}])[0].get("value", "").strip()
            lines.append(
                f"  {hour:02d}:00  {slot.get('tempC','?')}C, {desc}, "
                f"rain {slot.get('chanceofrain','?')}%, wind {slot.get('windspeedKmph','?')} km/h"
            )
        lines.append("Source: wttr.in")
        return "\n".join(lines)

    def _current_time(self, timezone: str = "UTC") -> str:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
        except Exception:
            tz = None
        now = datetime.now(tz) if tz else datetime.now(timezone_utc())
        label = timezone if tz else "UTC"
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({label}), %A")

    def syntax_check(self, files: list[str]) -> str:
        """Compile changed Python files in-process to catch syntax errors.

        Safe without --allow-shell: compile() parses but does not execute the
        code, so this verification runs even when execution is disabled. Returns
        an empty string when all files parse, or a description of the errors.
        """
        problems = []
        root = self._root()
        for rel in files:
            if not rel.endswith(".py"):
                continue
            target = root / rel
            try:
                src = target.read_text(encoding="utf-8", errors="replace")
                compile(src, rel, "exec")
            except SyntaxError as exc:
                problems.append(f"{rel}:{exc.lineno}: {exc.msg}")
            except FileNotFoundError:
                problems.append(f"{rel}: file disappeared before it could be checked")
            except (PermissionError, OSError) as exc:
                problems.append(f"{rel}: could not be read ({exc})")
            except Exception as exc:
                problems.append(f"{rel}: could not be checked ({type(exc).__name__}: {exc})")
        return "\n".join(problems)

    def git_diff(self, files: list[str] | None = None, limit: int = 40000) -> str:
        """Uncommitted git diff in the project root, optionally scoped to files."""
        root = self._root()
        args = ["git", "-C", str(root), "diff", "--no-color"]
        if files:
            args += ["--", *files]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                return proc.stdout[:limit]
        except Exception:
            pass      # no git, or not a repo: fall back to the plain search below
        return ""

    def _root(self) -> Path:
        """The directory the file tools operate in: the configured project, or the
        sandboxed workspace when none is set. Falls back to the workspace if the
        configured project path does not exist."""
        pd = (self.config.project_dir or "").strip()
        if pd:
            root = Path(pd).expanduser()
            if root.is_dir():
                return root.resolve()
            log(f"PROJECT_DIR {pd!r} is not a directory; using ./workspace", logging.WARNING)
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        return WORKSPACE_DIR.resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve a model-supplied path inside the active root, refusing escapes
        and never touching the .git directory."""
        root = self._root()
        candidate = (root / str(path).lstrip("/")).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("path escapes the project directory")
        parts = candidate.relative_to(root).parts if candidate != root else ()
        if ".git" in parts:
            raise ValueError("refusing to touch the .git directory")
        return candidate

    def _rel(self, target: Path) -> str:
        try:
            return str(target.relative_to(self._root()))
        except ValueError:
            return str(target)

    def _note_change(self, target: Path) -> None:
        self.changed_files.add(self._rel(target))

    # Directory names never worth walking for listing/search.
    _IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
                    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
                    ".next", ".gradle", "target", ".idea", ".tox", ".cache"}

    def _ignored_set(self) -> set[str] | None:
        """Relative paths git considers ignored, cached briefly. None if not a repo
        or git is unavailable — callers then fall back to _IGNORE_DIRS."""
        root = self._root()
        now = time.time()
        cache = getattr(self, "_ignore_cache", None)
        if cache and cache[0] == str(root) and now - cache[1] < 5.0:
            return cache[2]
        result: set[str] | None = None
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-o", "-i", "--exclude-standard",
                 "--directory"],
                capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                result = {line.strip().rstrip("/") for line in proc.stdout.splitlines() if line.strip()}
        except Exception:
            result = None
        self._ignore_cache = (str(root), now, result)
        return result

    def _is_ignored(self, item: Path, ignored: set[str] | None) -> bool:
        parts = set(item.relative_to(self._root()).parts) if item != self._root() else set()
        if parts & self._IGNORE_DIRS:
            return True
        if ignored:
            rel = str(item.relative_to(self._root()))
            if rel in ignored or any(rel.startswith(ig + "/") for ig in ignored):
                return True
        return False

    def _list_files(self, path: str = "", recursive: Any = False) -> str:
        target = self._resolve(path)
        if not target.exists():
            return "Directory does not exist."
        if target.is_file():
            return f"{target.name} ({target.stat().st_size} bytes)"
        if str(recursive).lower() in ("1", "true", "yes"):
            root = self._root()
            ignored = self._ignored_set()
            entries = sorted(
                (item for item in target.rglob("*") if not self._is_ignored(item, ignored)),
                key=lambda item: str(item),
            )
            lines = [
                f"{item.relative_to(root)}{'/' if item.is_dir() else ''}"
                f"{'' if item.is_dir() else f' ({item.stat().st_size} bytes)'}"
                for item in entries[:400]
            ]
            return "\n".join(lines) or "Empty directory."
        entries = sorted(target.iterdir())
        if not entries:
            return "Empty directory."
        return "\n".join(
            f"{entry.name}{'/' if entry.is_dir() else ''} ({entry.stat().st_size} bytes)"
            for entry in entries[:200]
        )

    def _read_file(self, path: str) -> str:
        target = self._resolve(path)
        if target.is_dir():
            raise ValueError(f"{path} is a directory, not a file. Use list_files to see what is inside.")
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        cap = self.config.tool_raw_chars
        ext = target.suffix.lower()
        try:
            if ext == ".ipynb":
                return self._read_notebook(target, cap)
            if ext in (".csv", ".tsv"):
                return self._read_tabular(target, cap, "\t" if ext == ".tsv" else ",")
            if ext == ".json":
                return self._read_json(target, cap)
            if ext == ".pdf":
                return self._read_pdf(target, cap)
            if ext in (".docx",):
                return self._read_docx(target, cap)
            if ext in (".xlsx", ".xlsm"):
                return self._read_xlsx(target, cap)
            if ext in (".yaml", ".yml"):
                return self._read_yaml(target, cap)
            if ext == ".toml":
                return self._read_toml(target, cap)
            try:
                raw = target.read_bytes()[:cap * 2]
            except PermissionError:
                raise ValueError(f"no permission to read {path}") from None
            except OSError as exc:
                raise ValueError(f"could not read {path}: {exc}") from None
            if b"\x00" in raw[:4096]:
                kb = target.stat().st_size / 1024
                return (f"[binary file: {target.name}, {kb:.1f} KB, type {ext or 'unknown'}]. "
                        "Not shown as text. Use file_info for details.")
            return raw.decode("utf-8", errors="replace")[:cap]
        except Exception as exc:
            # Never fail a read outright: fall back to raw text with a note.
            try:
                return (f"[could not parse {ext or 'file'} ({type(exc).__name__}: {exc}); "
                        "showing raw text]\n\n"
                        + target.read_text(encoding="utf-8", errors="replace")[:cap])
            except Exception:
                raise ValueError(f"could not read {path}: {exc}") from exc

    def _read_notebook(self, target: Path, cap: int) -> str:
        nb = json.loads(target.read_text(encoding="utf-8", errors="replace"))
        out = []
        for i, cell in enumerate(nb.get("cells", [])):
            kind = cell.get("cell_type", "?")
            src = "".join(cell.get("source", []))
            out.append(f"# --- cell {i} [{kind}] ---\n{src}")
        return (f"[notebook: {len(nb.get('cells', []))} cells]\n\n" + "\n\n".join(out))[:cap]

    def _read_tabular(self, target: Path, cap: int, delim: str) -> str:
        import csv as _csv
        rows = []
        with target.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = _csv.reader(fh, delimiter=delim)
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= 50:
                    break
        total = sum(1 for _ in target.open(encoding="utf-8", errors="replace"))
        if not rows:
            return "[tabular: file is empty]"
        header = rows[0]
        preview = "\n".join(delim.join(r) for r in rows[:20])
        return (f"[tabular: ~{total} rows, {len(header)} columns]\n"
                f"columns: {', '.join(header)}\n\nfirst rows:\n{preview}")[:cap]

    def _read_json(self, target: Path, cap: int) -> str:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return (f"[invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}; "
                    "showing raw text]\n\n" + text[:cap])
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        if len(pretty) <= cap:
            return pretty
        # Too big: describe the shape instead of dumping.
        def shape(v, depth=0):
            if isinstance(v, dict):
                keys = list(v.keys())[:20]
                return "{" + ", ".join(f"{k}: {type(v[k]).__name__}" for k in keys) + \
                       (", ..." if len(v) > 20 else "") + "}"
            if isinstance(v, list):
                return f"[{len(v)} items of {type(v[0]).__name__ if v else 'empty'}]"
            return type(v).__name__
        return (f"[json too large to show fully; {len(pretty)} chars]\n"
                f"top-level: {shape(data)}\n\nhead:\n{pretty[:cap - 200]}")

    def _read_pdf(self, target: Path, cap: int) -> str:
        return self._extract_pdf(str(target), cap)

    @staticmethod
    def _extract_pdf(source: Any, cap: int) -> str:
        """Extract text from a PDF given a path string or a binary stream
        (BytesIO). Shared by the file reader and the URL fetcher."""
        for mod, fn in (("pypdf", "PdfReader"), ("PyPDF2", "PdfReader")):
            try:
                m = __import__(mod)
            except ImportError:
                continue  # try the next library
            try:
                reader = getattr(m, fn)(source)
                text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
                if not text:
                    return f"[pdf: {len(reader.pages)} pages, no extractable text (scanned?)]"
                return (f"[pdf: {len(reader.pages)} pages]\n\n" + text)[:cap]
            except Exception as exc:
                return f"[could not extract pdf text: {exc}]"
        return "[pdf detected but no PDF library installed. `pip install pypdf` to read PDFs.]"

    def _read_docx(self, target: Path, cap: int) -> str:
        try:
            import docx  # python-docx
        except ImportError:
            return "[docx detected but python-docx not installed. `pip install python-docx` to read.]"
        try:
            doc = docx.Document(str(target))
            text = "\n".join(p.text for p in doc.paragraphs)
            return (f"[docx: {len(doc.paragraphs)} paragraphs]\n\n" + text.strip())[:cap]
        except Exception as exc:
            return f"[could not read docx: {exc}]"

    def _read_xlsx(self, target: Path, cap: int) -> str:
        try:
            import openpyxl
        except ImportError:
            return "[xlsx detected but openpyxl not installed. `pip install openpyxl` to read.]"
        try:
            wb = openpyxl.load_workbook(str(target), read_only=True, data_only=True)
            out = []
            for ws in wb.worksheets:
                out.append(f"# sheet: {ws.title} ({ws.max_row} rows x {ws.max_column} cols)")
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    out.append(", ".join("" if c is None else str(c) for c in row))
                    if i >= 20:
                        out.append("...")
                        break
            wb.close()
            return "\n".join(out)[:cap]
        except Exception as exc:
            return f"[could not read xlsx: {exc}]"

    def _read_yaml(self, target: Path, cap: int) -> str:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            return "[yaml] (PyYAML not installed; showing raw text)\n\n" + text[:cap]
        except Exception as exc:
            return f"[yaml parse error: {exc}; showing raw text]\n\n" + text[:cap]
        return (f"[yaml, top-level {type(data).__name__}]\n\n" + text)[:cap]

    def _read_toml(self, target: Path, cap: int) -> str:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            import tomllib  # Python 3.11+
            data = tomllib.loads(text)
            keys = ", ".join(list(data.keys())[:20])
            return (f"[toml, top-level keys: {keys}]\n\n" + text)[:cap]
        except ModuleNotFoundError:
            return "[toml] (tomllib unavailable; showing raw text)\n\n" + text[:cap]
        except Exception as exc:
            return f"[toml parse error: {exc}; showing raw text]\n\n" + text[:cap]

    def _index_url(self, url: str) -> str:
        """Fetch a web page and add its text to the knowledge base."""
        if self.db is None or not getattr(self.db, "fts_enabled", False):
            return "Knowledge base unavailable (SQLite FTS5 not enabled)."
        if not str(url).lower().startswith(("http://", "https://")):
            return (f"{url!r} is not a web address. Give a full URL starting with "
                    "http:// or https://, or use index_docs for a local file.")
        try:
            text = self._fetch_url(url)
        except Exception as exc:
            return f"Could not fetch {url}: {type(exc).__name__}: {exc}"
        if not text or len(text.strip()) < 50:
            return f"Nothing substantial to index from {url}."
        n = self.db.index_document(url, text, url, user_id=get_acting_user())
        return f"Indexed {url} into the knowledge base ({n} passages)."

    def _search_docs(self, query: str, limit: Any = None) -> str:
        try:
            k = min(10, max(1, int(limit)))
        except (TypeError, ValueError):
            k = 5
        # Scope to the acting user's own + shared documents so the tool never
        # surfaces another user's imported/indexed material.
        hits = (self.db.search_documents(query, limit=k, user_id=get_acting_user())
                if self.db else [])
        if not hits:
            return "No matching passages in the knowledge base."
        out = []
        for h in hits:
            out.append(f"[{h['path']}]\n{h['chunk']}")
        return "\n\n".join(out)[:self.config.tool_raw_chars]

    def _index_docs(self, path: str) -> str:
        """Index a file or directory into the knowledge base, reusing the
        type-aware readers so PDFs, notebooks and spreadsheets are handled."""
        if self.db is None or not getattr(self.db, "fts_enabled", False):
            return "Knowledge base unavailable (SQLite FTS5 not enabled)."
        target = self._resolve(path)
        if not target.exists():
            raise ValueError(f"no such path: {path}")
        files = [target] if target.is_file() else [
            f for f in sorted(target.rglob("*"))
            if f.is_file() and not self._is_ignored(f, self._ignored_set())]
        indexed, skipped, chunks = 0, 0, 0
        for f in files[:200]:
            try:
                text = self._read_file(self._rel(f))
            except Exception:
                skipped += 1
                continue
            if not text or text.startswith("[binary file"):
                skipped += 1
                continue
            n = self.db.index_document(self._rel(f), text, f.name,
                                       user_id=get_acting_user())
            chunks += n
            indexed += 1
        if not files:
            return f"Nothing to index: {path} contains no readable files."
        if not indexed:
            return (f"Indexed nothing from {path}: all {skipped} file(s) were binary, "
                    "empty or unreadable.")
        note = f", skipped {skipped}" if skipped else ""
        more = " (only the first 200 files were considered)" if len(files) > 200 else ""
        return (f"Indexed {indexed} file(s) into the knowledge base "
                f"({chunks} passages){note}.{more}")

    def _file_info(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ValueError(f"no such path: {path}")
        if target.is_dir():
            n = sum(1 for _ in target.iterdir())
            return f"{self._rel(target)}: directory with {n} entries"
        size = target.stat().st_size
        ext = target.suffix.lower() or "(none)"
        raw = target.read_bytes()[:4096]
        binary = b"\x00" in raw
        info = [f"path: {self._rel(target)}", f"size: {size} bytes ({size/1024:.1f} KB)",
                f"extension: {ext}", f"kind: {'binary' if binary else 'text'}"]
        if not binary:
            try:
                lines = sum(1 for _ in target.open(encoding="utf-8", errors="replace"))
                info.append(f"lines: {lines}")
            except Exception:
                pass  # unreadable: report the rest of the metadata, not an error
        return "\n".join(info)

    def _write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        if target.is_dir():
            raise ValueError(f"{path} is a directory; give a file path to write to.")
        existed = target.is_file()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        except PermissionError:
            raise ValueError(f"no permission to write {path}") from None
        except OSError as exc:
            raise ValueError(f"could not write {path}: {exc}") from None
        self._note_change(target)
        verb = "Updated" if existed else "Created"
        note = self._validate_written(target, str(content))
        return f"{verb} {self._rel(target)} ({len(str(content))} characters){note}"

    def _validate_written(self, target: Path, content: str) -> str:
        """After writing a structured file, check it parses and report the result
        inline so the agent catches malformed JSON/YAML/TOML immediately."""
        ext = target.suffix.lower()
        try:
            if ext == ".json":
                json.loads(content)
                return "  [valid JSON \u2713]"
            if ext in (".yaml", ".yml"):
                import yaml
                yaml.safe_load(content)
                return "  [valid YAML \u2713]"
            if ext == ".toml":
                import tomllib
                tomllib.loads(content)
                return "  [valid TOML \u2713]"
        except ImportError:
            return ""
        except Exception as exc:
            return f"  [\u26a0 warning: not valid {ext[1:].upper()}: {exc}]"
        return ""

    def _edit_file(self, path: str, find: str, replace: str, count: Any = None) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        original = target.read_text(encoding="utf-8", errors="replace")
        occurrences = original.count(find)
        if occurrences == 0:
            raise ValueError(
                "the find text does not appear in the file. Read the file first "
                "and copy the snippet exactly, including indentation."
            )
        try:
            limit = int(count) if count not in (None, "") else occurrences
        except (TypeError, ValueError):
            limit = occurrences
        updated = original.replace(find, str(replace), max(1, limit))
        target.write_text(updated, encoding="utf-8")
        self._note_change(target)
        replaced = min(occurrences, max(1, limit))
        return f"Replaced {replaced} of {occurrences} occurrence(s) in {self._rel(target)}"

    def _search_files(self, pattern: str, path: str = "", max_results: Any = None) -> str:
        root = self._resolve(path)
        if not root.exists():
            return "Directory does not exist."
        try:
            limit = min(200, max(1, int(max_results)))
        except (TypeError, ValueError):
            limit = 40
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        base = self._root()
        ignored = self._ignored_set()
        targets = [root] if root.is_file() else sorted(
            item for item in root.rglob("*") if not self._is_ignored(item, ignored))
        hits: list[str] = []
        for item in targets:
            if not item.is_file() or item.stat().st_size > 2_000_000:
                continue
            try:
                text = item.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{item.relative_to(base)}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + "\n[result limit reached]"
        return "\n".join(hits) or "No matches."

    def _recall_feedback(self, query: str, limit: Any = 5) -> str:
        if self.db is None:
            return "Feedback store unavailable."
        try:
            count = min(20, max(1, int(limit)))
        except (TypeError, ValueError):
            count = 5
        # Only the acting user's own feedback — never another user's prompts/answers.
        rows = self.db.list_feedback(limit=count, search=query, user_id=get_acting_user())
        if not rows:
            return "No matching feedback."
        lines = []
        for row in rows:
            answer = row.get("corrected_response") or row.get("assistant_response") or ""
            lines.append(f"Q: {row.get('user_prompt', '')}\nA: {answer}")
        return "\n\n".join(lines)

    def _register_o365(self) -> None:
        """Register the Graph-backed tools. Called only when O365_* is configured."""
        self._add(Tool(
            name="o365_mail",
            description="Read or send Office 365 / Outlook mail via Microsoft Graph.",
            parameters={"action": "read | send", "query": "search/filter for read",
                        "to": "recipient (send)", "subject": "subject (send)",
                        "body": "message body (send)"},
            required=["action"],
            handler=self._o365_mail,
        ))
        self._add(Tool(
            name="o365_files",
            description="Browse or fetch OneDrive / SharePoint files via Microsoft Graph.",
            parameters={"action": "list | get", "path": "folder or file path"},
            required=["action"],
            handler=self._o365_files,
        ))
        self._add(Tool(
            name="o365_calendar",
            description="Read Office 365 calendar events via Microsoft Graph.",
            parameters={"window": "e.g. today | week | a date range"},
            required=[],
            handler=self._o365_calendar,
        ))

    # ---- Office 365 / Microsoft Graph (framework; connect later) --------- #
    def _o365_configured(self) -> bool:
        c = self.config
        return bool(c.o365_tenant_id and c.o365_client_id and c.o365_client_secret)

    def _o365(self, service: str, **kwargs: Any) -> str:
        """Shared Office 365 entrypoint.

        The connection framework is in place (config + capability + tools); live
        Microsoft Graph calls are not wired yet, so this reports its state clearly
        instead of failing. When O365_* is unset it tells the user how to connect;
        when set it acknowledges the request and names what a live build would do.
        """
        if not self._o365_configured():
            return ("Office 365 is not connected. An admin must register an Azure AD "
                    "app and set O365_TENANT_ID, O365_CLIENT_ID and O365_CLIENT_SECRET "
                    "(Settings -> Office 365), then this agent can use it.")
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v)
        return (f"[office365:{service}] request accepted ({detail or 'no args'}). "
                f"Connected app {self.config.o365_client_id[:8]}... on tenant "
                f"{self.config.o365_tenant_id[:8]}... with scopes "
                f"{self.config.o365_scopes}. Microsoft Graph calls are not enabled "
                "in this build yet; the connection is configured and ready to be wired.")

    def _o365_mail(self, action: str, query: str = "", to: str = "",
                   subject: str = "", body: str = "") -> str:
        return self._o365("mail", action=action, query=query, to=to,
                          subject=subject, body=body)

    def _o365_files(self, action: str, path: str = "") -> str:
        return self._o365("files", action=action, path=path)

    def _o365_calendar(self, window: str = "week") -> str:
        return self._o365("calendar", window=window)

    def _remember(self, key: str, value: str) -> str:
        if self.db is None:
            return "Memory store unavailable."
        key = str(key).strip()[:120]
        if not key:
            raise ValueError("key must not be empty")
        # Scope the note to the acting user (contextvar) so it is private to them.
        # conversation id also comes from a contextvar: the registry is shared, so
        # reading an instance attribute would race across concurrent requests.
        self.db.remember(key, str(value), get_acting_conversation(),
                         user_id=get_acting_user())
        return f"Stored under {key}."

    def _recall_memory(self, query: str = "", limit: Any = 10) -> str:
        if self.db is None:
            return "Memory store unavailable."
        try:
            count = min(50, max(1, int(limit)))
        except (TypeError, ValueError):
            count = 10
        rows = self.db.recall(query or None, count, user_id=get_acting_user())
        if not rows:
            return "No stored notes." if not query else f"No stored notes matching {query!r}."
        return "\n".join(f"{row['key']}: {row['value']}" for row in rows)

    def _forget(self, key: str) -> str:
        if self.db is None:
            return "Memory store unavailable."
        return (f"Deleted {key}."
                if self.db.forget(str(key), user_id=get_acting_user())
                else f"No note called {key}.")

    def _exec(self, cmd, shell: bool = False) -> str:
        """Run a command in the project root and return captured, capped output.

        Two backends. "local" is a CONFINED subprocess (fixed working directory,
        timeout, size-capped output) but runs with your user privileges, which is
        why it is gated behind --allow-shell / --allow-python. "docker" runs the
        command inside a container with the project mounted at /work and no
        network, which is a real isolation boundary. Either way you review the
        diff on the host and push manually.
        """
        root = self._root()
        if (self.config.exec_backend or "local").lower() == "docker":
            return self._exec_docker(cmd, shell, root)
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=str(root),
                capture_output=True, text=True, timeout=self.config.tool_timeout,
            )
        except subprocess.TimeoutExpired:
            return (f"(timed out after {self.config.tool_timeout}s; the command was "
                    "still running and was stopped)")
        except FileNotFoundError:
            first = cmd if isinstance(cmd, str) else (cmd[0] if cmd else "")
            return f"(command not found: {first!r})"
        except Exception as exc:
            return f"(could not run the command: {type(exc).__name__}: {exc})"
        out = (proc.stdout or "")
        if proc.stderr:
            out += ("\n[stderr]\n" + proc.stderr)
        out = out.strip() or "(no output)"
        out += f"\n[exit code {proc.returncode}]"
        return out[:self.config.tool_raw_chars]

    def _exec_docker(self, cmd, shell: bool, root: Path) -> str:
        """Run the command inside a container with the project mounted at /work.

        The container is removed after the run (--rm), has the project as its
        working directory, and runs with no network. Requires Docker installed
        and running; on any Docker failure this returns a clear message rather
        than silently falling back to unsandboxed local execution.
        """
        image = self.config.docker_image or "python:3.12-slim"
        inner = cmd if shell else " ".join(shlex.quote(str(c)) for c in cmd)
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{root}:/work",
            "-w", "/work",
            "--network", "none",
            image, "sh", "-lc", inner,
        ]
        try:
            proc = subprocess.run(
                docker_cmd, capture_output=True, text=True,
                timeout=self.config.tool_timeout + 30,
            )
        except subprocess.TimeoutExpired:
            return f"(docker run timed out after ~{self.config.tool_timeout}s)"
        except FileNotFoundError:
            return ("(docker is not installed or not on PATH; install Docker or set "
                    "EXEC_BACKEND=local)")
        except Exception as exc:
            return f"(could not run docker: {type(exc).__name__}: {exc})"
        out = (proc.stdout or "")
        if proc.stderr:
            out += ("\n[stderr]\n" + proc.stderr)
        out = out.strip() or "(no output)"
        out += f"\n[exit code {proc.returncode}] [sandbox: docker {image}]"
        return out[:self.config.tool_raw_chars]

    def _run_python(self, code: str) -> str:
        return self._exec([sys.executable, "-c", code])

    def _run_shell(self, command: str) -> str:
        return self._exec(command, shell=True)

    def _detect_test_command(self) -> str:
        """Pick a sensible test command from the project's files."""
        root = self._root()
        if self.config.test_command:
            return self.config.test_command
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() \
                or (root / "tests").is_dir() or list(root.glob("test_*.py")):
            return "python -m pytest -q"
        if (root / "package.json").exists():
            return "npm test --silent"
        if (root / "Cargo.toml").exists():
            return "cargo test"
        if (root / "go.mod").exists():
            return "go test ./..."
        if (root / "Makefile").exists():
            return "make test"
        return ""

    def _run_tests(self, command: str = "") -> str:
        cmd = (command or "").strip() or self._detect_test_command()
        if not cmd:
            return ("No test command found. Set TEST_COMMAND or pass one, e.g. "
                    "'python -m pytest -q'.")
        return f"$ {cmd}\n" + self._exec(cmd, shell=True)

    def call(self, name: str, args: dict, conversation_id: str | None = None) -> tuple[str, str | None]:
        """Run a tool. Returns (result text, error message or None)."""
        tool = self.get(name)
        start = time.time()
        # Bind on the shared instance for backward compat, but the source of
        # truth for tool handlers is the per-context contextvar (race-free).
        self.conversation_id = conversation_id
        set_acting_conversation(conversation_id)
        if tool is None:
            error = f"Unknown tool: {name}. Available: {', '.join(self.names())}"
            if self.db:
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error,
                                      user_id=get_acting_user())
            return error, error

        normalised = self.normalise_args(tool, args)
        missing = [
            key for key in tool.required
            if key not in normalised or normalised[key] in (None, "")
        ]
        if missing:
            expected = ", ".join(f"{k} ({v})" for k, v in tool.parameters.items())
            error = (f"Missing required argument(s) for {name}: {', '.join(missing)}. "
                     f"Expected arguments: {expected}")
            if self.db:
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error,
                                      user_id=get_acting_user())
            return error, error

        clean = self.normalise_args(tool, args)
        try:
            result = str(tool.handler(**clean))
            error = None
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            error = result
        duration = (time.time() - start) * 1000
        # Keep the full-ish result here. The agent decides separately how much
        # of it is worth spending context on, after a summarisation pass.
        result = result[:self.config.tool_raw_chars]
        if self.db:
            self.db.log_tool_call(conversation_id, name, clean, result, duration, error,
                                  user_id=get_acting_user())
        return result, error



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Tool',
    'ToolRegistry',
    'guarded_thread',
    'resolve_in_workspace',
]
