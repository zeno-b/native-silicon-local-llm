#!/usr/bin/env python3
"""
deploy.py

All-in-one local LLM trainer/server/feedback loop for Apple Silicon.

Designed for:
- M1 Mac with 8GB RAM
- Native arm64 Python
- Fast local operation
- Full loop:
   serve model
   chat web UI
   collect feedback
   export feedback
   LoRA retrain
   restart server

Usage:
   python3 deploy.py --seed-demo
   python3 deploy.py --export-only --export-format csv
   python3 deploy.py --list-feedback
   python3 deploy.py --system-prompt "You are a pirate"
   python3 deploy.py --agent --context-size 8192 --max-tokens 512
   python3 deploy.py --tool-test web_search --tool-args '{"query": "mlx lora"}'

Environment overrides:
   MODEL_ID, MODEL_PORT, WEB_PORT, TRAIN_ITERS, TRAIN_LR, TRAIN_SEQ_LEN,
   MAX_TOKENS, SYSTEM_PROMPT, AUTO_RETRAIN_THRESHOLD,
   CONTEXT_SIZE, MAX_KV_SIZE, TEMPERATURE, HISTORY_TURNS,
   AGENT_ENABLED, AGENT_MAX_STEPS, ALLOW_PYTHON,
   SEARCH_BACKEND (ddg|brave|tavily|searxng), SEARCH_RESULTS,
   BRAVE_API_KEY, TAVILY_API_KEY, SEARXNG_URL

Notes:
- This script creates ./.venv and installs dependencies on first run.
- It stores data in ./data and logs in ./logs.
- Agent tools that touch the filesystem are confined to ./workspace.
- Retraining stops the model server temporarily, trains, then restarts it.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import html
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
import subprocess
import sys
import textwrap
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SFT_DIR = DATA_DIR / "sft"
LOG_DIR = ROOT / "logs"
ADAPTER_DIR = ROOT / "adapters" / "latest"
ADAPTER_BACKUP_DIR = ROOT / "adapters" / "backups"
WORKSPACE_DIR = ROOT / "workspace"
DB_PATH = DATA_DIR / "feedback.db"

DEFAULT_MODEL = os.environ.get(
    "MODEL_ID",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
)
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant.",
)

# Roughly four characters per token for English. Good enough for budgeting a
# context window without pulling a tokenizer into the web process, which would
# mean loading the model a second time in a machine with 8GB of RAM.
CHARS_PER_TOKEN = 4
# Tokens held back from the context budget for the chat template, the tool
# instructions, and the model's own reply.
CONTEXT_SAFETY_MARGIN = 256

# Bump when HTML_PAGE changes. Shown in the header and returned by /api/health so
# a stale browser cache is immediately visible rather than silently misleading.
UI_BUILD = "2026-08-06.4-agent"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [local-llm] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("local_llm")


def log(msg: str, level: int = logging.INFO) -> None:
    logger.log(level, msg)


def run_cmd(cmd: list[str], check: bool = True) -> None:
    log("$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=check)


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def bootstrap() -> None:
    """Create venv and install dependencies if needed. Re-executes inside venv."""
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")

    if platform.system() != "Darwin":
        log("Warning: this script is intended for macOS.", logging.WARNING)
    if platform.machine() != "arm64":
        log("Warning: this script is intended for native Apple Silicon arm64.", logging.WARNING)

    if not in_venv():
        venv_dir = ROOT / ".venv"
        venv_python = venv_dir / "bin" / "python"

        if not venv_python.exists():
            log("Creating virtual environment...")
            run_cmd([sys.executable, "-m", "venv", str(venv_dir)])

        log("Upgrading pip...")
        try:
            run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        except Exception as exc:
            log(f"Warning: pip upgrade failed: {exc}", logging.WARNING)

        log("Installing dependencies...")
        run_cmd([
            str(venv_python), "-m", "pip", "install",
            "mlx-lm", "fastapi", "uvicorn", "httpx", "pydantic",
        ])

        log("Restarting script inside virtual environment...")
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:],
        )

    missing = []
    for module_name in ["mlx_lm", "fastapi", "uvicorn", "httpx", "pydantic"]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        packages = ["mlx-lm" if m == "mlx_lm" else m for m in missing]
        log("Installing missing dependencies: " + ", ".join(packages))
        run_cmd([sys.executable, "-m", "pip", "install", *packages])
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
        )


def ensure_dirs() -> None:
    for d in [DATA_DIR, SFT_DIR, LOG_DIR, ADAPTER_DIR.parent, ADAPTER_BACKUP_DIR, WORKSPACE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokenizer."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def messages_tokens(messages: list[dict]) -> int:
    # Four tokens per message covers the chat template role markers.
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


def trim_to_context(
    system: dict,
    history: list[dict],
    user: dict,
    context_size: int,
    reserve: int,
) -> tuple[list[dict], int]:
    """Drop the oldest history until the request fits the context window.

    Returns the message list actually sent and the number of dropped messages.
    The system prompt and the current user turn are never dropped: if they alone
    exceed the budget the caller has a configuration problem, not a history
    problem, and silently truncating them would hide it.
    """
    budget = max(256, context_size - reserve - CONTEXT_SAFETY_MARGIN)
    fixed = messages_tokens([system, user])
    kept = list(history)
    dropped = 0
    while kept and fixed + messages_tokens(kept) > budget:
        kept.pop(0)
        dropped += 1
    return [system, *kept, user], dropped


class Database:
    """Thread-safe SQLite manager with one connection per thread."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.append(conn)
        return conn

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_prompt TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    rating INTEGER DEFAULT 0,
                    corrected_response TEXT,
                    approved_for_training INTEGER DEFAULT 0,
                    session_id TEXT,
                    model_id TEXT,
                    trained_at TIMESTAMP
                )
            """)
            # Migration: older databases predate trained_at.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(feedback)")}
            if "trained_at" not in columns:
                log("Migrating feedback table: adding trained_at column.")
                conn.execute("ALTER TABLE feedback ADD COLUMN trained_at TIMESTAMP")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_approved 
                ON feedback(approved_for_training)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_untrained
                ON feedback(approved_for_training, trained_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created 
                ON feedback(created_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    duration_ms REAL,
                    status_code INTEGER,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    est_tokens INTEGER DEFAULT 0,
                    meta TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    name TEXT NOT NULL,
                    args TEXT,
                    result TEXT,
                    duration_ms REAL,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation
                ON tool_calls(conversation_id, id)
            """)
            conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._connection()
        return conn.execute(sql, params)

    def commit(self) -> None:
        self._connection().commit()

    def close(self) -> None:
        """Close every connection this Database has handed out, across all threads."""
        with self._conns_lock:
            conns, self._all_conns = self._all_conns, []
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    def seed_demo(self) -> int:
        count = self.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()["cnt"]
        if count > 0:
            log("Feedback table not empty, skipping demo seed.")
            return 0

        examples = [
            ("What is this app?", "This is a local LLM chat app that can learn from your feedback."),
            ("How do I retrain the model?", "Give feedback on answers, then press the Retrain button."),
            ("What machine is this optimized for?", "This configuration is optimized for an 8GB Apple Silicon Mac."),
        ]
        for user_prompt, assistant_response in examples:
            self.execute(
                """INSERT INTO feedback
                   (user_prompt, assistant_response, rating, corrected_response, approved_for_training)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_prompt, assistant_response, 1, assistant_response, 1),
            )
        self.commit()
        log(f"Inserted {len(examples)} demo feedback examples.")
        return len(examples)

    def list_feedback(
        self,
        limit: int = 50,
        approved_only: bool = False,
        search: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []
        if approved_only:
            sql += " AND approved_for_training = 1"
        if search:
            sql += " AND (user_prompt LIKE ? OR assistant_response LIKE ? OR corrected_response LIKE ?)"
            params.extend([f"%{search}%"] * 3)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, int]:
        total = self.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()["cnt"]
        approved = self.execute("SELECT COUNT(*) as cnt FROM feedback WHERE approved_for_training = 1").fetchone()["cnt"]
        positive = self.execute("SELECT COUNT(*) as cnt FROM feedback WHERE rating > 0").fetchone()["cnt"]
        negative = self.execute("SELECT COUNT(*) as cnt FROM feedback WHERE rating < 0").fetchone()["cnt"]
        corrected = self.execute("SELECT COUNT(*) as cnt FROM feedback WHERE corrected_response IS NOT NULL").fetchone()["cnt"]
        untrained = self.execute(
            "SELECT COUNT(*) as cnt FROM feedback "
            "WHERE approved_for_training = 1 AND trained_at IS NULL"
        ).fetchone()["cnt"]
        return {
            "total": total,
            "approved": approved,
            "untrained": untrained,
            "positive": positive,
            "negative": negative,
            "corrected": corrected,
        }

    def get_untrained_count(self) -> int:
        """Count feedback approved for training that has not been trained on yet."""
        return self.execute(
            "SELECT COUNT(*) as cnt FROM feedback "
            "WHERE approved_for_training = 1 AND trained_at IS NULL"
        ).fetchone()["cnt"]

    def mark_trained(self, feedback_ids: list[int]) -> int:
        """Stamp rows as trained so auto-retrain does not fire on them again."""
        if not feedback_ids:
            return 0
        stamp = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in feedback_ids)
        cursor = self.execute(
            f"UPDATE feedback SET trained_at = ? WHERE id IN ({placeholders})",
            (stamp, *feedback_ids),
        )
        self.commit()
        return cursor.rowcount

    def delete_feedback(self, feedback_id: int) -> bool:
        cursor = self.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        self.commit()
        return cursor.rowcount > 0

    def clear_feedback(self) -> int:
        cursor = self.execute("DELETE FROM feedback")
        self.commit()
        return cursor.rowcount

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> int:
        cursor = self.execute(
            "INSERT INTO messages (conversation_id, role, content, est_tokens, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                estimate_tokens(content),
                json.dumps(meta) if meta else None,
            ),
        )
        self.commit()
        return int(cursor.lastrowid or 0)

    def get_messages(self, conversation_id: str, limit: int = 200) -> list[dict]:
        """Return the tail of a conversation in chronological order."""
        rows = self.execute(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_conversation(self, conversation_id: str) -> int:
        cursor = self.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        self.execute("DELETE FROM tool_calls WHERE conversation_id = ?", (conversation_id,))
        self.commit()
        return cursor.rowcount

    def list_conversations(self, limit: int = 50) -> list[dict]:
        rows = self.execute(
            "SELECT conversation_id, COUNT(*) AS messages, MAX(created_at) AS last_at "
            "FROM messages GROUP BY conversation_id ORDER BY last_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def log_tool_call(
        self,
        conversation_id: str | None,
        name: str,
        args: dict,
        result: str,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        self.execute(
            "INSERT INTO tool_calls (conversation_id, name, args, result, duration_ms, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                name,
                json.dumps(args, ensure_ascii=False)[:4000],
                (result or "")[:8000],
                duration_ms,
                error,
            ),
        )
        self.commit()

    def list_tool_calls(self, limit: int = 100, conversation_id: str | None = None) -> list[dict]:
        if conversation_id:
            rows = self.execute(
                "SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        else:
            rows = self.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def log_metric(self, endpoint: str, duration_ms: float, status_code: int, error: str | None = None) -> None:
        self.execute(
            "INSERT INTO metrics (endpoint, duration_ms, status_code, error) VALUES (?, ?, ?, ?)",
            (endpoint, duration_ms, status_code, error),
        )
        self.commit()


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def get_free_port(preferred: int, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    for port in range(preferred, preferred + 100):
        if port in exclude:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port near {preferred}")


def wait_for_port(port: int, timeout: int = 300, proc: subprocess.Popen | None = None) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Model server process exited early. Check logs/model_server.log.")
        if port_open(port):
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for port {port}. Check logs/model_server.log.")


def wait_for_model_ready(
    model_id: str,
    port: int,
    timeout: int = 900,
    proc: subprocess.Popen | None = None,
) -> None:
    """Block until the model actually answers a completion.

    wait_for_port only proves the socket accepts connections. mlx_lm.server binds
    the port before the weights finish loading, so treating an open port as
    readiness makes the first real request hang or fail against a server that is
    still initialising. A one-token completion is the only cheap proof of life.
    """
    import urllib.error
    import urllib.request

    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }).encode("utf-8")

    start = time.time()
    last_error: str = "no response"
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Model server process exited while loading. Check logs/model_server.log.")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            # A 4xx means the server is up and routing, just unhappy with us.
            body = exc.read(200).decode("utf-8", "replace")
            raise RuntimeError(
                f"Model server rejected the readiness probe: HTTP {exc.code} {body}"
            )
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(
        f"Model did not become ready within {timeout}s (last: {last_error}). "
        "Check logs/model_server.log."
    )


def render_ui() -> str:
    """Return the HTML the browser receives, with the build marker substituted.

    HTML_PAGE is a raw string on purpose. In a normal Python string, a `\\n`
    written inside embedded JavaScript becomes a real line break, which splits a
    JS string literal across two lines and makes the whole <script> fail to
    parse. The page then renders but nothing works: the status panel sits on its
    hardcoded "Starting..." text and the Send button does nothing.
    """
    return HTML_PAGE.replace("{{UI_BUILD}}", UI_BUILD)


def check_ui_syntax() -> list[str]:
    """Cheap structural check on the rendered <script>, no Node required.

    Catches the failure above by finding string literals broken by a real
    newline. Returns a list of problems; empty means the script is well formed.
    """
    import re as _re

    script = _re.search(r"<script>(.*?)</script>", render_ui(), _re.S)
    if not script:
        return ["no <script> block found in HTML_PAGE"]

    problems = []
    for lineno, line in enumerate(script.group(1).splitlines(), 1):
        stripped = _re.sub(r"\\.", "", line)          # drop escaped chars
        stripped = _re.sub(r"//.*$", "", stripped)     # drop line comments
        for quote in ('"', "'"):
            if stripped.count(quote) % 2:
                problems.append(f"line {lineno}: unterminated {quote} string: {line.strip()[:70]}")
                break
    return problems


def help_cmd(module: str) -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", module, "--help"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return ""


def add_if_supported(
    cmd: list[str],
    help_text: str,
    names: list[str],
    value: str | None = None,
) -> bool:
    if not help_text:
        return False
    for name in names:
        if name in help_text:
            cmd.append(name)
            if value is not None:
                cmd.append(str(value))
            return True
    return False


def adapter_ready() -> bool:
    return ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir())


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_port: int = field(default_factory=lambda: int(os.environ.get("MODEL_PORT", "8080")))
    web_port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8000")))
    train_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_ITERS", "30")))
    train_lr: str = field(default_factory=lambda: os.environ.get("TRAIN_LR", "1e-4"))
    train_seq_len: str = field(default_factory=lambda: os.environ.get("TRAIN_SEQ_LEN", "256"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "512")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    # Context window. context_size is the budget this process enforces when it
    # assembles a request; max_kv_size is what the model server is told to
    # allocate. They are separate because the server flag is optional and
    # changing it needs a restart, while context_size takes effect immediately.
    context_size: int = field(default_factory=lambda: int(os.environ.get("CONTEXT_SIZE", "4096")))
    max_kv_size: int = field(default_factory=lambda: int(os.environ.get("MAX_KV_SIZE", "0")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.7")))
    history_turns: int = field(default_factory=lambda: int(os.environ.get("HISTORY_TURNS", "20")))

    # Agent
    agent_enabled: bool = field(default_factory=lambda: os.environ.get("AGENT_ENABLED", "0") == "1")
    agent_max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "6")))
    allow_python: bool = field(default_factory=lambda: os.environ.get("ALLOW_PYTHON", "0") == "1")
    allow_shell: bool = field(default_factory=lambda: os.environ.get("ALLOW_SHELL", "0") == "1")

    # Tools
    search_backend: str = field(default_factory=lambda: os.environ.get("SEARCH_BACKEND", "ddg"))
    search_results: int = field(default_factory=lambda: int(os.environ.get("SEARCH_RESULTS", "5")))
    tool_timeout: int = field(default_factory=lambda: int(os.environ.get("TOOL_TIMEOUT", "30")))
    tool_result_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RESULT_CHARS", "4000")))

    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"

    # Settings the web UI is allowed to change at runtime. Anything not listed
    # here needs a process restart and is rejected by /api/config.
    MUTABLE = (
        "system_prompt", "max_tokens", "temperature", "context_size",
        "history_turns", "agent_enabled", "agent_max_steps",
        "search_backend", "search_results", "tool_result_chars",
    )

    def public(self) -> dict:
        data = {k: v for k, v in asdict(self).items()}
        data["mutable"] = list(self.MUTABLE)
        return data

    def apply(self, updates: dict) -> list[str]:
        """Apply a settings patch. Returns the names of the fields changed."""
        changed = []
        for key, value in updates.items():
            if key not in self.MUTABLE or value is None:
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            else:
                value = str(value)
            if value != current:
                setattr(self, key, value)
                changed.append(key)
        if self.context_size < 512:
            self.context_size = 512
        if self.max_tokens < 16:
            self.max_tokens = 16
        if self.max_tokens >= self.context_size:
            self.max_tokens = max(16, self.context_size // 2)
        self.temperature = min(2.0, max(0.0, self.temperature))
        self.agent_max_steps = min(20, max(1, self.agent_max_steps))
        return changed


class ModelServerManager:
    """Manages the MLX model server with health probes and auto-restart."""

    def __init__(self, model_id: str, model_port: int, adapter_dir: Path, max_kv_size: int = 0):
        self.model_id = model_id
        self.model_port = model_port
        self.adapter_dir = adapter_dir
        self.max_kv_size = max_kv_size
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"
        self.lock = threading.RLock()
        self._server_help = help_cmd("mlx_lm.server")
        self._log_file: Any = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    def _build_cmd(self, use_adapter: bool) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm.server"]
        if not add_if_supported(cmd, self._server_help, ["--model", "--hf-path", "--mlx-path"], self.model_id):
            cmd.extend(["--model", self.model_id])
        if not add_if_supported(cmd, self._server_help, ["--port"], str(self.model_port)):
            cmd.extend(["--port", str(self.model_port)])
        if self.max_kv_size > 0:
            # Caps the rotating KV cache. Older mlx-lm builds do not have it, so
            # this is best-effort rather than an error.
            if not add_if_supported(
                cmd, self._server_help,
                ["--max-kv-size", "--max_kv_size"], str(self.max_kv_size),
            ):
                log(
                    "Installed mlx_lm.server has no --max-kv-size; the KV cache stays unbounded.",
                    logging.WARNING,
                )
        if use_adapter and adapter_ready():
            add_if_supported(cmd, self._server_help, ["--adapter-path", "--adapter"], str(self.adapter_dir))
        return cmd

    def _start_watchdog(self) -> None:
        # A restart triggered from inside the watchdog thread must not spawn a second one.
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_stop.clear()
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Auto-restart model server if it crashes unexpectedly."""
        while not self._watchdog_stop.wait(5):
            with self.lock:
                if self.status == "ready" and self.proc is not None and self.proc.poll() is not None:
                    log("Watchdog: Model server crashed, restarting...", logging.WARNING)
                    self.status = "restarting"
                    try:
                        self._start_internal()
                    except Exception as exc:
                        log(f"Watchdog restart failed: {exc}", logging.ERROR)
                        self.status = f"error: {exc}"

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        # stop() can be reached from inside the watchdog loop; joining self raises.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
            self._watchdog_thread = None

    def stop(self) -> None:
        self._stop_watchdog()
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                log("Stopping model server...")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        log("Model server did not exit after SIGKILL.", logging.WARNING)
            self.proc = None
            self.status = "stopped"
            self._close_log_file()
        # Give the OS a moment to release the port. Done outside the lock.
        time.sleep(0.5)

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _start_internal(self) -> None:
        """Internal start without lock acquisition (caller must hold lock)."""
        candidates = []
        if adapter_ready():
            candidates.append(self._build_cmd(True))
        candidates.append(self._build_cmd(False))

        last_error: Exception | None = None
        for cmd in candidates:
            log("Starting model server: " + " ".join(cmd))
            self._close_log_file()
            self._log_file = open(LOG_DIR / "model_server.log", "ab")
            proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)
            time.sleep(2)

            if proc.poll() is None:
                self.proc = proc
                try:
                    wait_for_port(self.model_port, timeout=300, proc=proc)
                    self.status = "loading"
                    log(f"Port {self.model_port} open. Waiting for weights to load...")
                    wait_for_model_ready(self.model_id, self.model_port, timeout=900, proc=proc)
                    self.status = "ready"
                    log(f"Model loaded and responding at http://127.0.0.1:{self.model_port}")
                    self._start_watchdog()
                    return
                except Exception as exc:
                    last_error = exc
                    self.stop()
            else:
                self._close_log_file()
                last_error = RuntimeError(
                    f"Model server exited immediately with code {proc.returncode}. "
                    "Check logs/model_server.log."
                )

        self.status = f"error: {last_error}"
        raise last_error or RuntimeError("Failed to start model server.")

    def start(self) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.status = "ready"
                return
            self.status = "starting"
            self._start_internal()

    def restart(self) -> None:
        self.stop()
        self.start()

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def health_probe(self) -> bool:
        """Actually ping the model server to verify it's responsive."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Try a minimal chat completion as health check
                resp = await client.post(
                    f"http://127.0.0.1:{self.model_port}/v1/chat/completions",
                    json={
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False


class RetrainManager:
    """Manages LoRA retraining with adapter backup/rollback."""

    def __init__(self, db: Database, model_manager: ModelServerManager, config: Config):
        self.db = db
        self.model_manager = model_manager
        self.config = config
        self.lock = threading.Lock()
        self.status = {"running": False, "message": "idle"}
        self._lora_help = help_cmd("mlx_lm.lora")

    def _backup_adapter(self) -> Path | None:
        """Backup current adapter before retraining."""
        if not adapter_ready():
            return None
        ADAPTER_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = ADAPTER_BACKUP_DIR / backup_id
        # Two retrains inside the same second would collide on the timestamp.
        suffix = 1
        while backup_path.exists():
            backup_path = ADAPTER_BACKUP_DIR / f"{backup_id}_{suffix}"
            suffix += 1
        shutil.copytree(ADAPTER_DIR, backup_path)
        log(f"Adapter backed up to {backup_path}")
        return backup_path

    def _rollback_adapter(self, backup_path: Path | None) -> None:
        """Restore the pre-training adapter after a failed run."""
        if backup_path is None or not backup_path.exists():
            return
        try:
            if ADAPTER_DIR.exists():
                shutil.rmtree(ADAPTER_DIR)
            shutil.copytree(backup_path, ADAPTER_DIR)
            log(f"Rolled back adapter from {backup_path}", logging.WARNING)
        except Exception as exc:
            log(f"Adapter rollback failed: {exc}", logging.ERROR)

    def _build_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm.lora"]
        if not add_if_supported(cmd, self._lora_help, ["--model", "--hf-path", "--mlx-path"], self.config.model):
            cmd.extend(["--model", self.config.model])
        if not add_if_supported(cmd, self._lora_help, ["--train"]):
            cmd.append("--train")
        if not add_if_supported(cmd, self._lora_help, ["--data"], str(SFT_DIR)):
            cmd.extend(["--data", str(SFT_DIR)])
        if not add_if_supported(cmd, self._lora_help, ["--adapter-path", "--adapter"], str(ADAPTER_DIR)):
            cmd.extend(["--adapter-path", str(ADAPTER_DIR)])

        add_if_supported(cmd, self._lora_help, ["--iters", "--iterations"], str(self.config.train_iters))
        add_if_supported(cmd, self._lora_help, ["--batch-size"], "1")
        add_if_supported(cmd, self._lora_help, ["--learning-rate", "-lr"], self.config.train_lr)
        add_if_supported(cmd, self._lora_help, ["--grad-checkpoint", "--gradient-checkpoint"])
        add_if_supported(cmd, self._lora_help, ["--max-seq-length", "--seq-length", "--max-seq-len"], self.config.train_seq_len)
        return cmd

    def export_feedback(self) -> tuple[int, list[int]]:
        """Write train/valid JSONL. Returns (example count, contributing feedback ids)."""
        rows = self.db.execute("""
            SELECT id, user_prompt, assistant_response, corrected_response, rating
            FROM feedback WHERE approved_for_training = 1
            ORDER BY id
        """).fetchall()

        examples = []
        exported_ids: list[int] = []
        seen = set()
        for row in rows:
            user_prompt = (row["user_prompt"] or "").strip()
            assistant_response = (row["corrected_response"] or row["assistant_response"] or "").strip()
            if not user_prompt or not assistant_response:
                continue
            key = (user_prompt, assistant_response)
            if key in seen:
                # Duplicate content still counts as consumed, or it retriggers forever.
                exported_ids.append(row["id"])
                continue
            seen.add(key)
            exported_ids.append(row["id"])
            examples.append({
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            })

        if not examples:
            return 0, []

        random.seed(42)
        random.shuffle(examples)

        split = max(1, int(len(examples) * 0.9))
        train_examples = examples[:split]
        valid_examples = examples[split:] or examples[:1]

        train_path = SFT_DIR / "train.jsonl"
        valid_path = SFT_DIR / "valid.jsonl"

        train_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in train_examples) + "\n"
        valid_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_examples) + "\n"

        train_path.write_text(train_text, encoding="utf-8")
        valid_path.write_text(valid_text, encoding="utf-8")

        return len(examples), exported_ids

    def run(self, trigger: str = "manual") -> None:
        if not self.lock.acquire(blocking=False):
            return

        backup_path: Path | None = None
        try:
            self.status = {"running": True, "message": f"Retraining started from {trigger}"}

            count, exported_ids = self.export_feedback()
            if count == 0:
                self.status = {"running": False, "message": "No approved feedback available for training."}
                return

            backup_path = self._backup_adapter()

            self.status["message"] = f"Exported {count} examples. Stopping model server."
            self.model_manager.stop()

            cmd = self._build_cmd()
            train_log = LOG_DIR / "train.log"

            self.status["message"] = "Training LoRA adapter..."

            with open(train_log, "a", encoding="utf-8") as lf:
                lf.write(f"\n\n{datetime.now(timezone.utc).isoformat()} Training command:\n{' '.join(cmd)}\n")
                lf.flush()

                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    raise RuntimeError(f"Training failed with exit code {proc.returncode}. See logs/train.log.")

            self.db.mark_trained(exported_ids)

            self.status["message"] = "Training complete. Restarting model server."
            self.model_manager.restart()

            self.status = {"running": False, "message": f"Retraining complete on {count} examples."}

        except Exception as exc:
            log(f"Retrain failed: {exc}", logging.ERROR)
            self._rollback_adapter(backup_path)
            self.status = {"running": False, "message": f"Retrain error: {exc}"}
            try:
                self.model_manager.start()
            except Exception as restart_exc:
                self.status["message"] += f" Restart error: {restart_exc}"
        finally:
            self.lock.release()


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"


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


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in ("pi", "e", "tau", "sqrt", "log", "log2", "log10", "exp", "sin",
                 "cos", "tan", "asin", "acos", "atan", "atan2", "floor", "ceil",
                 "factorial", "degrees", "radians", "hypot", "fabs")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without exposing the interpreter.

    eval() on model-produced text is a remote code execution hole, so this walks
    the AST and rejects anything that is not a literal, an operator, or a
    whitelisted math function.
    """
    if len(expression) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _SAFE_NAMES.get(node.func.id)
            if not callable(func):
                raise ValueError(f"unknown function: {node.func.id}")
            return func(*[evaluate(arg) for arg in node.args])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]
        raise ValueError(f"disallowed expression element: {type(node).__name__}")

    return evaluate(tree)


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

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
        }


class ToolRegistry:
    """The tools the agent can call, built from the live config."""

    def __init__(self, config: Config, db: Database | None = None):
        self.config = config
        self.db = db
        self.search = SearchBackend(config)
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def _register_defaults(self) -> None:
        self._add(Tool(
            name="web_search",
            description="Search the web and return titles, URLs and snippets. Use for anything current or outside your training data.",
            parameters={"query": "search terms", "num_results": "how many results, 1-10, default 5"},
            required=["query"],
            handler=self._web_search,
        ))
        self._add(Tool(
            name="fetch_url",
            description="Download a web page and return its readable text. Use after web_search when a snippet is not enough.",
            parameters={"url": "absolute http or https URL", "max_chars": "truncate the page to this many characters"},
            required=["url"],
            handler=self._fetch_url,
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
            description="List files in the workspace directory.",
            parameters={"path": "subdirectory, default the workspace root"},
            required=[],
            handler=self._list_files,
        ))
        self._add(Tool(
            name="read_file",
            description="Read a text file from the workspace directory.",
            parameters={"path": "file path relative to the workspace"},
            required=["path"],
            handler=self._read_file,
        ))
        self._add(Tool(
            name="write_file",
            description="Write a text file into the workspace directory, replacing it if it exists.",
            parameters={"path": "file path relative to the workspace", "content": "full file contents"},
            required=["path", "content"],
            handler=self._write_file,
        ))
        self._add(Tool(
            name="recall_feedback",
            description="Search stored user feedback for earlier questions and corrected answers.",
            parameters={"query": "text to look for", "limit": "how many rows, default 5"},
            required=["query"],
            handler=self._recall_feedback,
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
                description="Run a shell command in the workspace directory and return its output.",
                parameters={"command": "shell command"},
                required=["command"],
                handler=self._run_shell,
            ))

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
        limit = self.config.tool_result_chars
        try:
            if max_chars:
                limit = min(20000, max(200, int(max_chars)))
        except (TypeError, ValueError):
            pass
        with httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            body = resp.text
        text = strip_html(body) if "html" in content_type or "<" in body[:200] else body
        title = ""
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        if match:
            title = strip_html(match.group(1))
        header = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
        return (header + text)[:limit]

    def _calculator(self, expression: str) -> str:
        return str(safe_eval(expression))

    def _current_time(self, timezone: str = "UTC") -> str:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
        except Exception:
            tz = None
        now = datetime.now(tz) if tz else datetime.now(timezone_utc())
        label = timezone if tz else "UTC"
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({label}), %A")

    def _list_files(self, path: str = "") -> str:
        target = resolve_in_workspace(path)
        if not target.exists():
            return "Directory does not exist."
        if target.is_file():
            return f"{target.name} ({target.stat().st_size} bytes)"
        entries = sorted(target.iterdir())
        if not entries:
            return "Empty directory."
        return "\n".join(
            f"{entry.name}{'/' if entry.is_dir() else ''} ({entry.stat().st_size} bytes)"
            for entry in entries[:200]
        )

    def _read_file(self, path: str) -> str:
        target = resolve_in_workspace(path)
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")[:self.config.tool_result_chars]

    def _write_file(self, path: str, content: str) -> str:
        target = resolve_in_workspace(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Wrote {len(str(content))} characters to {target.relative_to(WORKSPACE_DIR.resolve())}"

    def _recall_feedback(self, query: str, limit: Any = 5) -> str:
        if self.db is None:
            return "Feedback store unavailable."
        try:
            count = min(20, max(1, int(limit)))
        except (TypeError, ValueError):
            count = 5
        rows = self.db.list_feedback(limit=count, search=query)
        if not rows:
            return "No matching feedback."
        lines = []
        for row in rows:
            answer = row.get("corrected_response") or row.get("assistant_response") or ""
            lines.append(f"Q: {row.get('user_prompt', '')}\nA: {answer}")
        return "\n\n".join(lines)

    def _run_python(self, code: str) -> str:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=self.config.tool_timeout,
        )
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return (out.strip() or f"(no output, exit code {proc.returncode})")[:self.config.tool_result_chars]

    def _run_shell(self, command: str) -> str:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=self.config.tool_timeout,
        )
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return (out.strip() or f"(no output, exit code {proc.returncode})")[:self.config.tool_result_chars]

    def call(self, name: str, args: dict, conversation_id: str | None = None) -> tuple[str, str | None]:
        """Run a tool. Returns (result text, error message or None)."""
        tool = self.get(name)
        start = time.time()
        if tool is None:
            error = f"Unknown tool: {name}. Available: {', '.join(self.names())}"
            if self.db:
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error)
            return error, error

        missing = [key for key in tool.required if key not in args or args[key] in (None, "")]
        if missing:
            error = f"Missing required argument(s) for {name}: {', '.join(missing)}"
            if self.db:
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error)
            return error, error

        clean = {k: v for k, v in args.items() if k in tool.parameters}
        try:
            result = str(tool.handler(**clean))
            error = None
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            error = result
        duration = (time.time() - start) * 1000
        result = result[:self.config.tool_result_chars]
        if self.db:
            self.db.log_tool_call(conversation_id, name, clean, result, duration, error)
        return result, error


def timezone_utc():
    return timezone.utc


TOOL_PROTOCOL = textwrap.dedent("""\
    You can call tools. To call one, reply with a single JSON object and nothing
    else:
    {"tool": "tool_name", "args": {"arg": "value"}}
    You will then receive a message beginning with TOOL RESULT. Use it to answer.
    Call one tool at a time. When you can answer, reply with plain text and no
    JSON. Never invent tool output. Never claim you searched unless a TOOL RESULT
    says so.

    Available tools:
    """)


def build_agent_system_prompt(base_prompt: str, registry: ToolRegistry) -> str:
    lines = []
    for tool in registry.specs():
        params = ", ".join(
            f"{name} ({desc})" for name, desc in tool["parameters"].items()
        ) or "no arguments"
        required = ", ".join(tool["required"]) or "none"
        lines.append(f"- {tool['name']}: {tool['description']}\n  args: {params}\n  required: {required}")
    return f"{base_prompt}\n\n{TOOL_PROTOCOL}" + "\n".join(lines)


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


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Return (tool name, args) if the reply is a tool call, else None."""
    parsed = extract_json_object(text)
    if not parsed:
        return None
    name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
    if not isinstance(name, str) or not name:
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


class ModelClient:
    """Thin async client for the local OpenAI-compatible model server."""

    def __init__(self, config: Config):
        self.config = config

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.model_port}/v1/chat/completions"

    def payload(self, messages: list[dict], stream: bool, max_tokens: int | None = None) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(self.url, json=self.payload(messages, False, max_tokens))
            if resp.status_code != 200:
                fallback = self.payload(messages, False, max_tokens)
                fallback.pop("max_tokens", None)
                resp = await client.post(self.url, json=fallback)
            if resp.status_code != 200:
                raise RuntimeError(f"model server returned {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncGenerator[str, None]:
        import httpx
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", self.url, json=self.payload(messages, True, max_tokens)) as resp:
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
                        delta = data["choices"][0]["delta"].get("content", "")
                    except Exception:
                        continue
                    if delta:
                        yield delta


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

    def build_messages(self, history: list[dict], user_message: str) -> tuple[list[dict], int]:
        system = {
            "role": "system",
            "content": build_agent_system_prompt(self.config.system_prompt, self.registry),
        }
        user = {"role": "user", "content": user_message}
        return trim_to_context(
            system,
            [{"role": m["role"], "content": m["content"]} for m in history],
            user,
            self.config.context_size,
            self.config.max_tokens,
        )

    async def run(
        self,
        user_message: str,
        history: list[dict],
        conversation_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Yield events: step, token, tool_call, tool_result, final, error."""
        messages, dropped = self.build_messages(history, user_message)
        if dropped:
            yield {"type": "context", "dropped": dropped, "tokens": messages_tokens(messages)}

        seen_calls: list[str] = []
        trace: list[dict] = []

        for step in range(1, self.config.agent_max_steps + 1):
            yield {"type": "step", "step": step, "max_steps": self.config.agent_max_steps}
            buffer = ""
            stream = self.client.stream(messages)
            try:
                async for token in stream:
                    buffer += token
                    yield {"type": "token", "token": token, "step": step}
                    # A tool call is complete as soon as the JSON object closes;
                    # letting the model ramble past it wastes seconds per step.
                    if buffer.lstrip().startswith(("{", "```")) and parse_tool_call(buffer):
                        break
            except Exception as exc:
                yield {"type": "error", "error": str(exc)}
                return
            finally:
                # Breaking out early leaves the HTTP response open until the
                # generator is collected, which on a local server means a socket
                # per abandoned step.
                await stream.aclose()

            call = parse_tool_call(buffer)
            if call is None:
                answer = buffer.strip()
                yield {"type": "final", "answer": answer, "steps": step, "trace": trace}
                return

            name, args = call
            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            yield {"type": "tool_call", "name": name, "args": args, "step": step}

            if signature in seen_calls:
                result = (
                    f"You already called {name} with these arguments and received a result. "
                    "Answer the user now using what you have."
                )
                error = None
            else:
                seen_calls.append(signature)
                result, error = await asyncio.to_thread(
                    self.registry.call, name, args, conversation_id
                )

            trace.append({"name": name, "args": args, "result": result[:1000], "error": error})
            yield {"type": "tool_result", "name": name, "result": result, "error": error, "step": step}

            messages.append({"role": "assistant", "content": buffer.strip()})
            messages.append({
                "role": "user",
                "content": f"TOOL RESULT [{name}]:\n{result}\n\n"
                           "Answer the original question now, or call one more tool if you truly need it.",
            })
            messages, _ = trim_to_context(
                messages[0], messages[1:-1], messages[-1],
                self.config.context_size, self.config.max_tokens,
            )

        # Step budget exhausted. Force a plain answer with tools withheld.
        final_messages = [
            {"role": "system", "content": self.config.system_prompt},
            *messages[1:],
            {"role": "user", "content": "Give your best final answer now. Do not call any tool."},
        ]
        try:
            answer = await self.client.complete(final_messages)
        except Exception as exc:
            yield {"type": "error", "error": str(exc)}
            return
        yield {
            "type": "final",
            "answer": answer.strip(),
            "steps": self.config.agent_max_steps,
            "trace": trace,
            "truncated": True,
        }


HTML_PAGE = r"""
<!doctype html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>Local LLM</title>
 <style>
   :root {
     --bg: #111;
     --fg: #eee;
     --accent: #0a84ff;
     --surface: #1a1a1a;
     --border: #333;
     --error: #ff6b6b;
     --warn: #ffd93d;
     --success: #9be29b;
     --tool: #b48ead;
   }
   * { box-sizing: border-box; }
   body {
     font-family: -apple-system, BlinkMacSystemFont, sans-serif;
     margin: 0;
     padding: 0;
     background: var(--bg);
     color: var(--fg);
     height: 100vh;
     display: flex;
     flex-direction: column;
   }
   header {
     padding: 10px 16px;
     border-bottom: 1px solid var(--border);
     display: flex;
     justify-content: space-between;
     gap: 12px;
     align-items: center;
     flex-wrap: wrap;
   }
   header .actions { display: flex; gap: 6px; }
   #status {
     font-size: 12px;
     color: var(--success);
     white-space: pre-wrap;
     font-family: ui-monospace, monospace;
     flex: 1;
     min-width: 220px;
   }
   #status.error { color: var(--error); }
   #status.warn { color: var(--warn); }
   #main { flex: 1; display: flex; overflow: hidden; }
   #chat {
     flex: 1;
     overflow-y: auto;
     padding: 16px;
     display: flex;
     flex-direction: column;
     gap: 8px;
   }
   .msg {
     max-width: 80%;
     padding: 10px 12px;
     border-radius: 12px;
     white-space: pre-wrap;
     line-height: 1.35;
     word-wrap: break-word;
   }
   .user { align-self: flex-end; background: var(--accent); color: white; }
   .assistant { align-self: flex-start; background: var(--surface); border: 1px solid #444; }
   .assistant.pending { opacity: 0.75; }
   .tool-card {
     align-self: flex-start;
     max-width: 80%;
     background: #16121a;
     border: 1px solid var(--tool);
     border-radius: 10px;
     padding: 8px 10px;
     font-family: ui-monospace, monospace;
     font-size: 12px;
   }
   .tool-card .name { color: var(--tool); font-weight: 600; }
   .tool-card .args { color: #999; margin: 4px 0; white-space: pre-wrap; }
   .tool-card pre {
     margin: 6px 0 0;
     white-space: pre-wrap;
     max-height: 220px;
     overflow: auto;
     color: #ccc;
   }
   .tool-card.failed { border-color: var(--error); }
   .feedback {
     align-self: flex-start;
     display: flex;
     gap: 6px;
     margin-left: 6px;
     margin-bottom: 10px;
   }
   .feedback button {
     background: var(--surface);
     color: var(--fg);
     border: 1px solid #555;
     border-radius: 8px;
     padding: 4px 8px;
     cursor: pointer;
     font-size: 13px;
   }
   .feedback button:hover { background: #333; }
   .feedback button:disabled { opacity: 0.4; cursor: default; }
   footer {
     display: flex;
     gap: 8px;
     padding: 10px 16px;
     border-top: 1px solid var(--border);
     align-items: flex-end;
   }
   textarea, input[type=text], input[type=number], select {
     padding: 9px 11px;
     border-radius: 10px;
     border: 1px solid #444;
     background: var(--surface);
     color: var(--fg);
     outline: none;
     font-family: inherit;
     font-size: 14px;
   }
   #input { flex: 1; resize: vertical; min-height: 42px; max-height: 200px; }
   button {
     padding: 9px 13px;
     border-radius: 10px;
     border: 1px solid #555;
     background: #2c2c2c;
     color: var(--fg);
     cursor: pointer;
     font-size: 13px;
   }
   button:hover { background: #3a3a3a; }
   button:disabled { opacity: 0.5; cursor: not-allowed; }
   button.primary { background: var(--accent); border-color: var(--accent); color: white; }
   .system-msg { align-self: center; color: #888; font-size: 12px; margin: 4px 0; }
   .agent-toggle {
     display: flex;
     align-items: center;
     gap: 6px;
     font-size: 13px;
     color: #bbb;
     white-space: nowrap;
   }
   #settings {
     width: 320px;
     border-left: 1px solid var(--border);
     padding: 14px 16px;
     overflow-y: auto;
     display: none;
     background: #131313;
   }
   #settings.open { display: block; }
   #settings h3 { margin: 0 0 10px; font-size: 14px; }
   #settings label {
     display: block;
     font-size: 12px;
     color: #999;
     margin: 10px 0 4px;
   }
   #settings input, #settings select, #settings textarea { width: 100%; }
   #settings textarea { min-height: 70px; resize: vertical; }
   .row { display: flex; gap: 8px; }
   .row > div { flex: 1; }
   .meter { height: 6px; background: #222; border-radius: 3px; overflow: hidden; margin-top: 6px; }
   .meter > div { height: 100%; background: var(--accent); width: 0%; }
   .hint { font-size: 11px; color: #777; margin-top: 6px; line-height: 1.4; }
   .tools-list { font-size: 11px; color: #888; font-family: ui-monospace, monospace; line-height: 1.6; }
 </style>
</head>
<body>
 <header>
   <div><strong>Local LLM</strong> <span style="color:#666;font-size:11px">build {{UI_BUILD}}</span></div>
   <div id="status">Starting...</div>
   <div class="actions">
     <button onclick="newChat()">New chat</button>
     <button onclick="retrain()">Retrain</button>
     <button onclick="toggleSettings()">Settings</button>
   </div>
 </header>

 <div id="main">
   <div id="chat"></div>
   <aside id="settings">
     <h3>Generation</h3>
     <label>System prompt</label>
     <textarea id="cfgSystem"></textarea>
     <div class="row">
       <div>
         <label>Max tokens</label>
         <input id="cfgMaxTokens" type="number" min="16" max="32768" step="16">
       </div>
       <div>
         <label>Temperature</label>
         <input id="cfgTemperature" type="number" min="0" max="2" step="0.05">
       </div>
     </div>
     <div class="row">
       <div>
         <label>Context size (tokens)</label>
         <input id="cfgContext" type="number" min="512" max="131072" step="512">
       </div>
       <div>
         <label>History turns</label>
         <input id="cfgHistory" type="number" min="0" max="200" step="1">
       </div>
     </div>
     <div class="meter"><div id="ctxBar"></div></div>
     <div class="hint" id="ctxHint">Context usage in this conversation.</div>

     <h3 style="margin-top:18px">Agent</h3>
     <div class="row">
       <div>
         <label>Enabled by default</label>
         <select id="cfgAgent">
           <option value="false">off</option>
           <option value="true">on</option>
         </select>
       </div>
       <div>
         <label>Max steps</label>
         <input id="cfgAgentSteps" type="number" min="1" max="20" step="1">
       </div>
     </div>
     <div class="row">
       <div>
         <label>Search backend</label>
         <select id="cfgSearchBackend">
           <option value="ddg">ddg</option>
           <option value="brave">brave</option>
           <option value="tavily">tavily</option>
           <option value="searxng">searxng</option>
         </select>
       </div>
       <div>
         <label>Search results</label>
         <input id="cfgSearchResults" type="number" min="1" max="10" step="1">
       </div>
     </div>
     <label>Tool result limit (chars)</label>
     <input id="cfgToolChars" type="number" min="200" max="40000" step="200">

     <div style="margin-top:14px" class="row">
       <button class="primary" onclick="saveConfig()">Apply</button>
       <button onclick="restartModel()">Restart model</button>
     </div>
     <div class="hint">
       Context size takes effect on the next message. The KV cache size passed to
       the model server only changes on restart.
     </div>

     <h3 style="margin-top:18px">Tools</h3>
     <div class="tools-list" id="toolsList">loading...</div>
   </aside>
 </div>

 <footer>
   <textarea id="input" placeholder="Send a message. Shift+Enter for a new line." rows="1"></textarea>
   <label class="agent-toggle"><input id="agentToggle" type="checkbox"> agent</label>
   <button id="sendBtn" class="primary" onclick="send()">Send</button>
 </footer>

 <script>
   var chat = document.getElementById("chat");
   var input = document.getElementById("input");
   var sendBtn = document.getElementById("sendBtn");
   var statusEl = document.getElementById("status");
   var agentToggle = document.getElementById("agentToggle");
   var settingsEl = document.getElementById("settings");

   var conversationId = localStorage.getItem("llm_conversation") || randomId();
   localStorage.setItem("llm_conversation", conversationId);
   var contextSize = 4096;
   var usedTokens = 0;
   var busy = false;

   function randomId() {
     return Math.random().toString(36).slice(2, 12);
   }

   function estimateTokens(text) {
     return Math.max(1, Math.ceil((text || "").length / 4));
   }

   input.addEventListener("keydown", function(e) {
     if (e.key === "Enter" && !e.shiftKey) {
       e.preventDefault();
       send();
     }
   });

   function scrollDown() {
     chat.scrollTop = chat.scrollHeight;
   }

   function addMessage(role, text) {
     var div = document.createElement("div");
     div.className = "msg " + role;
     div.textContent = text;
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   function addSystem(text) {
     var div = document.createElement("div");
     div.className = "system-msg";
     div.textContent = text;
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   function addToolCard(name, args) {
     var card = document.createElement("div");
     card.className = "tool-card";
     var head = document.createElement("div");
     head.className = "name";
     head.textContent = "tool: " + name;
     var argsEl = document.createElement("div");
     argsEl.className = "args";
     argsEl.textContent = JSON.stringify(args);
     var body = document.createElement("pre");
     body.textContent = "running...";
     card.appendChild(head);
     card.appendChild(argsEl);
     card.appendChild(body);
     chat.appendChild(card);
     scrollDown();
     return { card: card, body: body };
   }

   function addFeedbackBar(userText, botText) {
     var bar = document.createElement("div");
     bar.className = "feedback";

     var up = document.createElement("button");
     up.textContent = "yes";
     up.title = "Good answer";
     var down = document.createElement("button");
     down.textContent = "no";
     down.title = "Bad answer";
     var correct = document.createElement("button");
     correct.textContent = "edit";
     correct.title = "Provide a corrected answer";

     up.onclick = function() { vote(userText, botText, 1); up.disabled = true; down.disabled = true; };
     down.onclick = function() { vote(userText, botText, -1); up.disabled = true; down.disabled = true; };
     correct.onclick = function() { correctAnswer(userText, botText); };

     bar.appendChild(up);
     bar.appendChild(down);
     bar.appendChild(correct);
     chat.appendChild(bar);
     scrollDown();
   }

   async function fetchJSON(url, options) {
     var res = await fetch(url, options);
     var raw = await res.text();
     if (!raw) return { res: res, data: {} };
     try {
       return { res: res, data: JSON.parse(raw) };
     } catch (err) {
       throw new Error(
         "HTTP " + res.status + " from " + (res.url || url) +
         " returned non-JSON, so this page is probably talking to the model " +
         "server instead of the web UI. Body: " + raw.slice(0, 200)
       );
     }
   }

   function updateContextMeter(delta) {
     usedTokens += delta;
     var pct = Math.min(100, Math.round((usedTokens / contextSize) * 100));
     document.getElementById("ctxBar").style.width = pct + "%";
     document.getElementById("ctxHint").textContent =
       "About " + usedTokens + " of " + contextSize + " tokens used. Oldest turns drop out automatically.";
   }

   async function send() {
     if (busy) return;
     var message = input.value.trim();
     if (!message) return;

     input.value = "";
     busy = true;
     sendBtn.disabled = true;
     addMessage("user", message);
     updateContextMeter(estimateTokens(message));

     var bubble = addMessage("assistant", "");
     bubble.classList.add("pending");
     var answered = false;
     var currentTool = null;

     try {
       var res = await fetch("/api/chat/stream", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           message: message,
           conversation_id: conversationId,
           agent: agentToggle.checked
         })
       });

       if (!res.ok || !res.body) {
         var text = await res.text();
         bubble.textContent = "Error: " + text.slice(0, 400);
         bubble.classList.remove("pending");
         return;
       }

       var reader = res.body.getReader();
       var decoder = new TextDecoder();
       var buffer = "";

       while (true) {
         var chunk = await reader.read();
         if (chunk.done) break;
         buffer += decoder.decode(chunk.value, { stream: true });
         var parts = buffer.split("\n\n");
         buffer = parts.pop();

         for (var i = 0; i < parts.length; i++) {
           var line = parts[i].trim();
           if (line.indexOf("data: ") !== 0) continue;
           var payload = line.slice(6);
           if (payload === "[DONE]") continue;
           var event;
           try {
             event = JSON.parse(payload);
           } catch (err) {
             continue;
           }
           handleEvent(event);
         }
       }

       function handleEvent(event) {
         if (event.type === "start") {
           conversationId = event.conversation_id || conversationId;
           localStorage.setItem("llm_conversation", conversationId);
         } else if (event.type === "context") {
           addSystem("Trimmed " + event.dropped + " old messages to fit the context window.");
         } else if (event.type === "step") {
           if (event.step > 1) bubble.textContent = "";
         } else if (event.type === "token") {
           bubble.textContent += event.token;
           scrollDown();
         } else if (event.type === "tool_call") {
           bubble.textContent = "";
           currentTool = addToolCard(event.name, event.args);
         } else if (event.type === "tool_result") {
           if (currentTool) {
             currentTool.body.textContent = event.result;
             if (event.error) currentTool.card.classList.add("failed");
             currentTool = null;
           }
           bubble = addMessage("assistant", "");
           bubble.classList.add("pending");
         } else if (event.type === "final") {
           bubble.textContent = event.answer;
           bubble.classList.remove("pending");
           answered = true;
           updateContextMeter(estimateTokens(event.answer));
           addFeedbackBar(message, event.answer);
           if (event.truncated) {
             addSystem("Agent hit the step limit and answered with what it had.");
           }
         } else if (event.type === "error") {
           bubble.textContent = "Error: " + event.error;
           bubble.classList.remove("pending");
           bubble.classList.add("failed");
           answered = true;
         }
       }

       if (!answered) {
         bubble.classList.remove("pending");
         if (!bubble.textContent) bubble.textContent = "(no response)";
       }
     } catch (err) {
       bubble.textContent = "Error: " + err.message;
       bubble.classList.remove("pending");
     } finally {
       busy = false;
       sendBtn.disabled = false;
       input.focus();
     }
   }

   async function vote(userPrompt, assistantResponse, rating) {
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: rating,
           corrected_response: null
         })
       });
       addSystem(out.data.status || "Feedback saved.");
     } catch (err) {
       addSystem("Feedback error: " + err.message);
     }
   }

   async function correctAnswer(userPrompt, assistantResponse) {
     var corrected = window.prompt("Corrected answer:", assistantResponse);
     if (corrected === null) return;
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: 1,
           corrected_response: corrected
         })
       });
       addSystem(out.data.status || "Correction saved.");
     } catch (err) {
       addSystem("Correction error: " + err.message);
     }
   }

   async function retrain() {
     if (!window.confirm("Start retraining? The model server will be stopped temporarily.")) return;
     try {
       var out = await fetchJSON("/api/retrain", { method: "POST" });
       addSystem(out.data.status || JSON.stringify(out.data));
     } catch (err) {
       addSystem("Retrain error: " + err.message);
     }
   }

   async function newChat() {
     conversationId = randomId();
     localStorage.setItem("llm_conversation", conversationId);
     chat.innerHTML = "";
     usedTokens = 0;
     updateContextMeter(0);
     addSystem("New conversation started.");
   }

   function toggleSettings() {
     settingsEl.classList.toggle("open");
   }

   async function restartModel() {
     try {
       var out = await fetchJSON("/api/model/restart", { method: "POST" });
       addSystem("Model server restarting. KV cache: " + out.data.max_kv_size);
     } catch (err) {
       addSystem("Restart error: " + err.message);
     }
   }

   async function loadConfig() {
     try {
       var out = await fetchJSON("/api/config");
       var cfg = out.data.config;
       contextSize = cfg.context_size;
       document.getElementById("cfgSystem").value = cfg.system_prompt;
       document.getElementById("cfgMaxTokens").value = cfg.max_tokens;
       document.getElementById("cfgTemperature").value = cfg.temperature;
       document.getElementById("cfgContext").value = cfg.context_size;
       document.getElementById("cfgHistory").value = cfg.history_turns;
       document.getElementById("cfgAgent").value = cfg.agent_enabled ? "true" : "false";
       document.getElementById("cfgAgentSteps").value = cfg.agent_max_steps;
       document.getElementById("cfgSearchBackend").value = cfg.search_backend;
       document.getElementById("cfgSearchResults").value = cfg.search_results;
       document.getElementById("cfgToolChars").value = cfg.tool_result_chars;
       agentToggle.checked = cfg.agent_enabled;
       var names = out.data.tools.map(function(t) { return t.name; });
       document.getElementById("toolsList").textContent = names.join("\n");
       updateContextMeter(0);
     } catch (err) {
       addSystem("Could not load settings: " + err.message);
     }
   }

   async function saveConfig() {
     var body = {
       system_prompt: document.getElementById("cfgSystem").value,
       max_tokens: Number(document.getElementById("cfgMaxTokens").value),
       temperature: Number(document.getElementById("cfgTemperature").value),
       context_size: Number(document.getElementById("cfgContext").value),
       history_turns: Number(document.getElementById("cfgHistory").value),
       agent_enabled: document.getElementById("cfgAgent").value === "true",
       agent_max_steps: Number(document.getElementById("cfgAgentSteps").value),
       search_backend: document.getElementById("cfgSearchBackend").value,
       search_results: Number(document.getElementById("cfgSearchResults").value),
       tool_result_chars: Number(document.getElementById("cfgToolChars").value)
     };
     try {
       var out = await fetchJSON("/api/config", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       contextSize = out.data.config.context_size;
       updateContextMeter(0);
       addSystem(out.data.changed.length
         ? "Updated: " + out.data.changed.join(", ")
         : "No settings changed.");
     } catch (err) {
       addSystem("Settings error: " + err.message);
     }
   }

   async function refreshHealth() {
     try {
       var out = await fetchJSON("/api/health");
       var data = out.data;
       var msg = "Model: " + (data.model_status || "unknown");
       if (data.ui_build && data.ui_build !== "{{UI_BUILD}}") {
         msg = "STALE PAGE: server is build " + data.ui_build +
               ", this tab is {{UI_BUILD}}. Hard-reload.\n" + msg;
       }
       msg += "  |  agent " + (data.agent_enabled ? "on" : "off") +
              ", ctx " + data.context_size + ", max " + data.max_tokens +
              ", tools " + (data.tools ? data.tools.length : 0);
       msg += "\nRetrain: " + ((data.retrain && data.retrain.message) || "idle");
       if (data.stats) {
         msg += "  |  feedback " + data.stats.total + " total, " + data.stats.approved
              + " approved, " + data.stats.untrained + " untrained";
       }
       statusEl.textContent = msg;
       statusEl.className = "";
       if (data.model_status && data.model_status.indexOf("error") === 0) statusEl.className = "error";
       else if (data.model_status === "starting" || data.model_status === "loading") statusEl.className = "warn";
     } catch (err) {
       statusEl.textContent = "Status unavailable: " + err.message;
       statusEl.className = "error";
     }
   }

   loadConfig();
   setInterval(refreshHealth, 3000);
   refreshHealth();
 </script>
</body>
</html>
"""


# Request/response models are bound at module scope by _define_api_models().
# They cannot be plain module-level class statements because pydantic is not
# installed until bootstrap() has run.
ChatRequest: Any = None
FeedbackRequest: Any = None
ChatResponse: Any = None
ConfigRequest: Any = None
ToolRequest: Any = None


def _define_api_models() -> None:
    """Define the Pydantic models in the module namespace.

    This module uses `from __future__ import annotations`, so every parameter
    annotation is a string at runtime. FastAPI resolves those strings against the
    endpoint function's __globals__, which is the module namespace. Models defined
    inside create_app() are invisible there, so FastAPI silently falls back to
    treating the body parameter as a query parameter and every POST returns 422.
    """
    global ChatRequest, FeedbackRequest, ChatResponse, ConfigRequest, ToolRequest
    if ChatRequest is not None:
        return
    from pydantic import BaseModel, Field

    class ChatRequest(BaseModel):  # noqa: F811
        message: str = Field(..., min_length=1, max_length=32000)
        conversation_id: str | None = Field(None, max_length=64)
        agent: bool | None = None
        max_tokens: int | None = Field(None, ge=16, le=32768)
        temperature: float | None = Field(None, ge=0.0, le=2.0)
        use_history: bool = True

    class FeedbackRequest(BaseModel):  # noqa: F811
        user_prompt: str = Field(..., min_length=1)
        assistant_response: str = Field(..., min_length=1)
        rating: int = Field(0, ge=-1, le=1)
        corrected_response: str | None = None

    class ChatResponse(BaseModel):  # noqa: F811
        answer: str

    class ConfigRequest(BaseModel):  # noqa: F811
        system_prompt: str | None = Field(None, max_length=8000)
        max_tokens: int | None = Field(None, ge=16, le=32768)
        temperature: float | None = Field(None, ge=0.0, le=2.0)
        context_size: int | None = Field(None, ge=512, le=1048576)
        history_turns: int | None = Field(None, ge=0, le=200)
        agent_enabled: bool | None = None
        agent_max_steps: int | None = Field(None, ge=1, le=20)
        search_backend: str | None = Field(None, max_length=20)
        search_results: int | None = Field(None, ge=1, le=10)
        tool_result_chars: int | None = Field(None, ge=200, le=40000)

    class ToolRequest(BaseModel):  # noqa: F811
        name: str = Field(..., min_length=1, max_length=64)
        args: dict = Field(default_factory=dict)
        conversation_id: str | None = Field(None, max_length=64)


def create_app(
    config: Config,
    db: Database,
    model_manager: ModelServerManager,
    retrain_manager: RetrainManager,
    registry: ToolRegistry | None = None,
):
    """Create and configure the FastAPI application with Pydantic validation."""
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    _define_api_models()

    registry = registry or ToolRegistry(config, db)
    model_client = ModelClient(config)
    agent = Agent(config, registry, model_client)

    app = FastAPI(title="Local LLM")

    # The UI is same-origin, so only the loopback origins this process serves are allowed.
    # A wildcard here would let any site the user visits drive /api/retrain and
    # DELETE /api/feedback on their machine.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://127.0.0.1:{config.web_port}",
            f"http://localhost:{config.web_port}",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # The UI is embedded in this file, so a cached copy silently defeats every
    # edit to HTML_PAGE. Nothing here is worth caching on a local dev server.
    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(content=render_ui())

    @app.get("/api/health")
    async def health():
        model_healthy = await model_manager.health_probe() if model_manager.is_alive() else False
        return {
            "ui_build": UI_BUILD,
            "web_port": config.web_port,
            "model_port": config.model_port,
            "model_status": model_manager.status,
            "model_process_alive": model_manager.is_alive(),
            "model_healthy": model_healthy,
            "retrain": retrain_manager.status,
            "stats": db.get_stats(),
            "agent_enabled": config.agent_enabled,
            "agent_max_steps": config.agent_max_steps,
            "context_size": config.context_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "tools": registry.names(),
            "search_backend": config.search_backend,
        }

    @app.get("/api/feedback")
    def list_feedback(
        limit: int = Query(50, ge=1, le=500),
        approved_only: bool = False,
        search: str | None = Query(None, max_length=100),
    ):
        return {"feedback": db.list_feedback(limit, approved_only, search)}

    @app.delete("/api/feedback/{feedback_id}")
    def delete_feedback(feedback_id: int):
        success = db.delete_feedback(feedback_id)
        return {"deleted": success}

    def not_ready() -> JSONResponse | None:
        if retrain_manager.status.get("running"):
            return JSONResponse(content={"error": "Retraining in progress. Please wait."}, status_code=503)
        if model_manager.status != "ready":
            return JSONResponse(
                content={"error": f"Model not ready. Status: {model_manager.status}"},
                status_code=503,
            )
        return None

    def load_history(request) -> tuple[str, list[dict]]:
        conversation_id = request.conversation_id or str(uuid.uuid4())[:12]
        if not request.use_history or config.history_turns <= 0:
            return conversation_id, []
        rows = db.get_messages(conversation_id, limit=config.history_turns * 2)
        return conversation_id, [{"role": r["role"], "content": r["content"]} for r in rows]

    def overrides(request) -> tuple[int, float]:
        max_tokens = request.max_tokens or config.max_tokens
        temperature = config.temperature if request.temperature is None else request.temperature
        return max_tokens, temperature

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        start_time = time.time()
        blocked = not_ready()
        if blocked is not None:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, "not ready")
            return blocked

        conversation_id, history = load_history(request)
        max_tokens, _ = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent

        try:
            if use_agent:
                answer = ""
                trace: list[dict] = []
                error: str | None = None
                async for event in agent.run(request.message, history, conversation_id):
                    if event["type"] == "final":
                        answer = event["answer"]
                        trace = event.get("trace", [])
                    elif event["type"] == "error":
                        error = event["error"]
                if error:
                    db.log_metric("chat", (time.time() - start_time) * 1000, 502, error)
                    return JSONResponse(content={"error": error}, status_code=502)
            else:
                system = {"role": "system", "content": config.system_prompt}
                user = {"role": "user", "content": request.message}
                messages, _ = trim_to_context(
                    system, history, user, config.context_size, max_tokens
                )
                answer = await model_client.complete(messages, max_tokens)
                trace = []

            db.add_message(conversation_id, "user", request.message)
            db.add_message(
                conversation_id, "assistant", answer,
                meta={"trace": trace} if trace else None,
            )
            db.log_metric("chat", (time.time() - start_time) * 1000, 200)
            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "agent": use_agent,
                "trace": trace,
            }
        except Exception as exc:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, str(exc))
            return JSONResponse(content={"error": str(exc)}, status_code=503)

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """Streaming chat over Server-Sent Events.

        Emits the same event vocabulary whether or not the agent is on, so the
        browser has one code path: step, token, tool_call, tool_result, final.
        """
        blocked = not_ready()
        if blocked is not None:
            return blocked

        conversation_id, history = load_history(request)
        max_tokens, _ = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent

        async def sse(event: dict) -> str:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        async def event_generator() -> AsyncGenerator[str, None]:
            answer = ""
            trace: list[dict] = []
            start_time = time.time()
            try:
                yield await sse({"type": "start", "conversation_id": conversation_id, "agent": use_agent})
                if use_agent:
                    async for event in agent.run(request.message, history, conversation_id):
                        yield await sse(event)
                        if event["type"] == "final":
                            answer = event["answer"]
                            trace = event.get("trace", [])
                else:
                    system = {"role": "system", "content": config.system_prompt}
                    user = {"role": "user", "content": request.message}
                    messages, dropped = trim_to_context(
                        system, history, user, config.context_size, max_tokens
                    )
                    if dropped:
                        yield await sse({"type": "context", "dropped": dropped,
                                         "tokens": messages_tokens(messages)})
                    async for token in model_client.stream(messages, max_tokens):
                        answer += token
                        yield await sse({"type": "token", "token": token, "step": 1})
                    yield await sse({"type": "final", "answer": answer, "steps": 1, "trace": []})

                if answer:
                    db.add_message(conversation_id, "user", request.message)
                    db.add_message(
                        conversation_id, "assistant", answer,
                        meta={"trace": trace} if trace else None,
                    )
                db.log_metric("chat_stream", (time.time() - start_time) * 1000, 200)
            except Exception as exc:
                db.log_metric("chat_stream", (time.time() - start_time) * 1000, 503, str(exc))
                yield await sse({"type": "error", "error": str(exc)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @app.get("/api/config")
    def get_config():
        return {"config": config.public(), "tools": registry.specs()}

    @app.post("/api/config")
    def set_config(request: ConfigRequest):
        changed = config.apply(request.model_dump(exclude_none=True))
        if changed:
            log(f"Config updated from web UI: {', '.join(changed)}")
        return {
            "changed": changed,
            "config": config.public(),
            "note": "context_size applies immediately. max_kv_size needs a model server restart.",
        }

    @app.get("/api/tools")
    def list_tools():
        return {
            "tools": registry.specs(),
            "search_backend": config.search_backend,
            "agent_enabled": config.agent_enabled,
            "agent_max_steps": config.agent_max_steps,
        }

    @app.post("/api/tools/call")
    async def call_tool(request: ToolRequest):
        """Run a tool directly. Useful for testing one without the model."""
        result, error = await asyncio.to_thread(
            registry.call, request.name, request.args, request.conversation_id
        )
        return {"name": request.name, "result": result, "error": error}

    @app.get("/api/tools/calls")
    def tool_calls(
        limit: int = Query(50, ge=1, le=500),
        conversation_id: str | None = Query(None, max_length=64),
    ):
        return {"calls": db.list_tool_calls(limit, conversation_id)}

    @app.get("/api/conversations")
    def conversations(limit: int = Query(50, ge=1, le=200)):
        return {"conversations": db.list_conversations(limit)}

    @app.get("/api/conversation/{conversation_id}")
    def conversation(conversation_id: str, limit: int = Query(200, ge=1, le=1000)):
        messages = db.get_messages(conversation_id, limit)
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "est_tokens": sum(m.get("est_tokens") or 0 for m in messages),
            "context_size": config.context_size,
        }

    @app.delete("/api/conversation/{conversation_id}")
    def clear_conversation(conversation_id: str):
        return {"deleted": db.clear_conversation(conversation_id)}

    @app.post("/api/model/restart")
    def restart_model():
        """Restart the model server, picking up a changed KV cache size."""
        model_manager.max_kv_size = config.max_kv_size
        threading.Thread(target=model_manager.restart, daemon=True).start()
        return {"status": "restarting", "max_kv_size": config.max_kv_size}

    @app.post("/api/feedback")
    async def feedback(request: FeedbackRequest):
        approved = 1 if (request.corrected_response or request.rating > 0) else 0
        session_id = str(uuid.uuid4())[:8]

        db.execute(
            """INSERT INTO feedback
               (user_prompt, assistant_response, rating, corrected_response, approved_for_training, session_id, model_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request.user_prompt,
                request.assistant_response,
                request.rating,
                request.corrected_response,
                approved,
                session_id,
                config.model,
            ),
        )
        db.commit()

        # Auto-retrain check
        if config.auto_retrain_threshold > 0:
            untrained = db.get_untrained_count()
            if untrained >= config.auto_retrain_threshold and not retrain_manager.status.get("running"):
                log(f"Auto-retrain triggered: {untrained} approved feedback items")
                threading.Thread(target=retrain_manager.run, kwargs={"trigger": "auto"}, daemon=True).start()

        return {
            "status": "feedback saved",
            "approved_for_training": bool(approved),
            "session_id": session_id,
        }

    @app.post("/api/retrain")
    def retrain():
        if retrain_manager.status.get("running"):
            return {"status": "already running", "detail": retrain_manager.status}

        threading.Thread(target=retrain_manager.run, kwargs={"trigger": "web"}, daemon=True).start()
        return {"status": "started", "detail": retrain_manager.status}

    @app.get("/api/metrics")
    def metrics(limit: int = Query(100, ge=1, le=1000)):
        rows = db.execute(
            "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"metrics": [dict(r) for r in rows]}

    return app


def export_to_csv(db: Database, path: Path) -> int:
    """Export all feedback to CSV."""
    rows = db.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    if not rows:
        return 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return len(rows)


def doctor(web_port: int, model_port: int) -> None:
    """Probe the loopback ports and report what is actually answering on each."""
    import urllib.error
    import urllib.request

    def probe(port: int, path: str) -> str:
        url = f"http://127.0.0.1:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return f"{resp.status} {resp.headers.get('Content-Type')} {resp.read(90)!r}"
        except urllib.error.HTTPError as exc:
            return f"{exc.code} {exc.headers.get('Content-Type')} {exc.read(90)!r}"
        except Exception as exc:
            return f"no response ({exc})"

    print(f"python      : {sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    print(f"script      : {Path(__file__).resolve()}")
    print(f"ui build    : {UI_BUILD}")
    print(f"in venv     : {in_venv()}  ({sys.prefix})")

    print(f"\nweb UI port {web_port}: {'LISTENING' if port_open(web_port) else 'CLOSED'}")
    if port_open(web_port):
        for path in ["/", "/api/health"]:
            print(f"  GET {path:14} -> {probe(web_port, path)}")

    print(f"\nmodel port {model_port}: {'LISTENING' if port_open(model_port) else 'CLOSED'}")
    if port_open(model_port):
        print(f"  GET {'/v1/models':14} -> {probe(model_port, '/v1/models')}")
        print(f"  GET {'/':14} -> {probe(model_port, '/')}")

    others = [p for p in range(8000, 8101) if p not in (web_port, model_port) and port_open(p)]
    print(f"\nother loopback listeners in 8000-8100: {others or 'none'}")
    print("\nA healthy web UI answers GET / with 200 text/html and")
    print("GET /api/health with 200 application/json. Anything else is the wrong port.")


def selftest() -> int:
    """Verify this file is intact and internally consistent. Returns an exit code.

    Runs before bootstrap() and uses only the standard library, so it works on a
    machine with no dependencies installed.
    """
    import hashlib
    import tempfile

    path = Path(__file__).resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    failures: list[str] = []

    print(f"file     : {path}")
    print(f"sha256   : {hashlib.sha256(raw).hexdigest()}")
    print(f"lines    : {text.count(chr(10))}")
    print(f"ui build : {UI_BUILD}")
    print(f"python   : {sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    print()

    # A pasted-over-instead-of-replaced file shows up as repeated top-of-file
    # markers. Match whole lines, and build the shebang from parts so this file
    # contains exactly one literal copy of it.
    shebang = "#!" + "/usr/bin/env python3"
    future_line = "from __future__ import annotations"
    source_lines = text.splitlines()
    shebangs = sum(1 for line in source_lines if line == shebang)
    futures = sum(1 for line in source_lines if line == future_line)
    if shebangs != 1:
        failures.append(f"file looks duplicated: {shebangs} shebangs, expected 1")
    if futures != 1:
        failures.append(f"file looks duplicated: {futures} __future__ imports, expected 1")

    failures.extend(f"embedded UI: {p}" for p in check_ui_syntax())

    page = render_ui()
    if "{{UI_BUILD}}" in page:
        failures.append("UI_BUILD placeholder left unsubstituted in the rendered page")
    if "<script>" not in page or "</script>" not in page:
        failures.append("rendered page is missing its <script> block")

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "selftest.db")
        db.execute(
            "INSERT INTO feedback (user_prompt, assistant_response, approved_for_training) "
            "VALUES ('a', 'b', 1)"
        )
        db.commit()
        if db.get_untrained_count() != 1:
            failures.append("untrained count did not reflect a newly approved row")
        db.mark_trained([1])
        if db.get_untrained_count() != 0:
            failures.append("mark_trained did not clear the untrained count")

        db.add_message("conv1", "user", "hello")
        db.add_message("conv1", "assistant", "hi there")
        history = db.get_messages("conv1")
        if [m["role"] for m in history] != ["user", "assistant"]:
            failures.append("conversation history did not round-trip in order")
        db.log_tool_call("conv1", "calculator", {"expression": "1+1"}, "2", 1.0)
        if not db.list_tool_calls(conversation_id="conv1"):
            failures.append("tool call log did not round-trip")
        if db.clear_conversation("conv1") != 2:
            failures.append("clear_conversation did not remove both messages")
        db.close()

    # Tool call parsing has to survive whatever a small model wraps around it.
    parse_cases = [
        ('{"tool": "web_search", "args": {"query": "x"}}', ("web_search", {"query": "x"})),
        ('Sure!\n```json\n{"tool": "calculator", "args": {"expression": "2+2"}}\n```', ("calculator", {"expression": "2+2"})),
        ('{"tool": "current_time", "arguments": {}}', ("current_time", {})),
        ('I think the answer is 4.', None),
        ('The set {a, b} is not JSON.', None),
    ]
    for text, expected in parse_cases:
        if parse_tool_call(text) != expected:
            failures.append(f"parse_tool_call mishandled: {text[:40]!r}")

    if abs(safe_eval("(2+3)*sqrt(16)") - 20.0) > 1e-9:
        failures.append("safe_eval returned the wrong result")
    for bad in ["__import__('os').system('ls')", "open('/etc/passwd').read()", "1 if x else 2"]:
        try:
            safe_eval(bad)
        except Exception:
            continue
        failures.append(f"safe_eval accepted unsafe input: {bad}")

    try:
        resolve_in_workspace("../../etc/passwd")
        failures.append("resolve_in_workspace allowed a path outside the workspace")
    except ValueError:
        pass

    system = {"role": "system", "content": "s" * 400}
    history = [{"role": "user", "content": "h" * 4000} for _ in range(10)]
    user = {"role": "user", "content": "u" * 400}
    trimmed, dropped = trim_to_context(system, history, user, 2048, 512)
    if dropped == 0 or trimmed[0] is not system or trimmed[-1] is not user:
        failures.append("trim_to_context did not drop history while keeping system and user turns")

    config = Config()
    registry = ToolRegistry(config)
    for expected_tool in ["web_search", "fetch_url", "calculator", "read_file", "write_file"]:
        if registry.get(expected_tool) is None:
            failures.append(f"tool registry is missing {expected_tool}")
    if registry.get("run_shell") is not None:
        failures.append("run_shell is exposed without --allow-shell")
    result, error = registry.call("calculator", {"expression": "6*7"})
    if error or result.strip() != "42":
        failures.append(f"calculator tool returned {result!r}")
    _, error = registry.call("nonexistent_tool", {})
    if not error:
        failures.append("unknown tool did not report an error")
    prompt = build_agent_system_prompt("base", registry)
    if "web_search" not in prompt or "TOOL RESULT" not in prompt:
        failures.append("agent system prompt is missing the tool protocol")

    changed = config.apply({"max_tokens": 99999, "context_size": 1024, "model": "evil/model"})
    if config.model == "evil/model":
        failures.append("config.apply changed a field outside MUTABLE")
    if config.max_tokens >= config.context_size:
        failures.append("config.apply left max_tokens above the context size")
    if "context_size" not in changed:
        failures.append("config.apply did not report context_size as changed")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS  single copy of the file")
    print("PASS  embedded UI parses and renders")
    print("PASS  database schema round-trips")
    print("PASS  tool call parsing, calculator sandbox, workspace confinement")
    print("PASS  context trimming and runtime config guardrails")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one local LLM server, chat UI, feedback, and retraining loop."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model-port", type=int, default=int(os.environ.get("MODEL_PORT", "8080")))
    parser.add_argument("--web-port", type=int, default=int(os.environ.get("WEB_PORT", "8000")))
    parser.add_argument("--train-iters", type=int, default=int(os.environ.get("TRAIN_ITERS", "30")))
    parser.add_argument("--train-lr", default=os.environ.get("TRAIN_LR", "1e-4"))
    parser.add_argument("--train-seq-len", default=os.environ.get("TRAIN_SEQ_LEN", "256"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "512")))
    parser.add_argument("--auto-retrain-threshold", type=int, default=int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    parser.add_argument("--context-size", type=int, default=int(os.environ.get("CONTEXT_SIZE", "4096")),
                        help="Token budget this process enforces when assembling a request.")
    parser.add_argument("--max-kv-size", type=int, default=int(os.environ.get("MAX_KV_SIZE", "0")),
                        help="KV cache cap passed to mlx_lm.server. 0 leaves it unbounded.")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.7")))
    parser.add_argument("--history-turns", type=int, default=int(os.environ.get("HISTORY_TURNS", "20")),
                        help="How many past messages to replay. 0 disables conversation memory.")

    parser.add_argument("--agent", action="store_true", default=os.environ.get("AGENT_ENABLED") == "1",
                        help="Enable the tool-calling agent loop by default.")
    parser.add_argument("--agent-max-steps", type=int, default=int(os.environ.get("AGENT_MAX_STEPS", "6")))
    parser.add_argument("--allow-python", action="store_true", default=os.environ.get("ALLOW_PYTHON") == "1",
                        help="Expose a run_python tool. The model gets code execution on this machine.")
    parser.add_argument("--allow-shell", action="store_true", default=os.environ.get("ALLOW_SHELL") == "1",
                        help="Expose a run_shell tool. The model gets shell access on this machine.")
    parser.add_argument("--search-backend", default=os.environ.get("SEARCH_BACKEND", "ddg"),
                        choices=["ddg", "brave", "tavily", "searxng"])
    parser.add_argument("--search-results", type=int, default=int(os.environ.get("SEARCH_RESULTS", "5")))
    parser.add_argument("--list-tools", action="store_true", help="Print the tool catalogue and exit.")
    parser.add_argument("--tool-test", metavar="NAME", help="Run one tool directly and exit.")
    parser.add_argument("--tool-args", default="{}", help="JSON arguments for --tool-test.")
    parser.add_argument("--seed-demo", action="store_true")
    parser.add_argument("--retrain-now", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--list-feedback", action="store_true")
    parser.add_argument("--doctor", action="store_true",
                        help="Probe the ports and report what is actually listening, then exit.")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify this file is intact and the embedded UI parses, then exit.")
    parser.add_argument("--export-format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    # Must run before get_free_port(), which deliberately returns *unused* ports
    # and would therefore report on the wrong ones.
    if args.doctor:
        doctor(args.web_port, args.model_port)
        return

    bootstrap()
    import uvicorn

    ensure_dirs()

    ui_problems = check_ui_syntax()
    if ui_problems:
        log("Embedded UI is malformed; the page would load but do nothing:", logging.ERROR)
        for problem in ui_problems:
            log(f"  {problem}", logging.ERROR)
        sys.exit("Refusing to serve a broken UI.")

    config = Config(
        model=args.model,
        system_prompt=args.system_prompt,
        model_port=args.model_port,
        web_port=args.web_port,
        train_iters=args.train_iters,
        train_lr=args.train_lr,
        train_seq_len=args.train_seq_len,
        max_tokens=args.max_tokens,
        auto_retrain_threshold=args.auto_retrain_threshold,
        context_size=args.context_size,
        max_kv_size=args.max_kv_size,
        temperature=args.temperature,
        history_turns=args.history_turns,
        agent_enabled=args.agent,
        agent_max_steps=args.agent_max_steps,
        allow_python=args.allow_python,
        allow_shell=args.allow_shell,
        search_backend=args.search_backend,
        search_results=args.search_results,
        seed_demo=args.seed_demo,
        retrain_now=args.retrain_now,
        export_only=args.export_only,
        list_feedback=args.list_feedback,
        export_format=args.export_format,
    )

    config.model_port = get_free_port(config.model_port)
    config.web_port = get_free_port(config.web_port, exclude={config.model_port})

    db = Database(DB_PATH)
    registry = ToolRegistry(config, db)

    if args.list_tools:
        for tool in registry.specs():
            print(f"{tool['name']}: {tool['description']}")
            for name, desc in tool["parameters"].items():
                flag = " (required)" if name in tool["required"] else ""
                print(f"    {name}{flag}: {desc}")
        db.close()
        return

    if args.tool_test:
        try:
            tool_args = json.loads(args.tool_args)
        except json.JSONDecodeError as exc:
            db.close()
            sys.exit(f"--tool-args is not valid JSON: {exc}")
        result, error = registry.call(args.tool_test, tool_args)
        print(result)
        db.close()
        sys.exit(1 if error else 0)

    if config.seed_demo:
        db.seed_demo()

    if config.list_feedback:
        import pprint
        stats = db.get_stats()
        print("\n=== Feedback Statistics ===")
        pprint.pprint(stats)
        print("\n=== Recent Feedback ===")
        for item in db.list_feedback(limit=100):
            pprint.pprint(item)
        db.close()
        return

    model_manager = ModelServerManager(config.model, config.model_port, ADAPTER_DIR, config.max_kv_size)
    retrain_manager = RetrainManager(db, model_manager, config)

    if config.export_only:
        count, _ = retrain_manager.export_feedback()
        log(f"Exported {count} training examples to {SFT_DIR}")
        if config.export_format == "csv":
            csv_count = export_to_csv(db, DATA_DIR / "feedback_export.csv")
            log(f"Exported {csv_count} rows to {DATA_DIR / 'feedback_export.csv'}")
        db.close()
        return

    app = create_app(config, db, model_manager, retrain_manager, registry)

    def handle_sigterm(signum, frame):
        log("Received SIGTERM, shutting down...")
        model_manager.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    log(f"Model: {config.model}")
    log(f"System prompt: {config.system_prompt[:60]}...")
    log(f"Context: {config.context_size} tokens, max_tokens {config.max_tokens}, temperature {config.temperature}")
    log(f"Agent: {'on' if config.agent_enabled else 'off'}, max steps {config.agent_max_steps}, "
        f"tools: {', '.join(registry.names())}")
    if config.allow_shell or config.allow_python:
        log("Code execution tools are enabled. The model can run commands on this machine.", logging.WARNING)
    log("=" * 62)
    log(f"  OPEN THIS IN YOUR BROWSER:  http://127.0.0.1:{config.web_port}")
    log(f"  Model backend (not a UI):   http://127.0.0.1:{config.model_port}")
    log("=" * 62)
    log("Ports shift automatically when the preferred one is busy, so use the URL above.")
    log("Starting model server. First run may download the model.")

    try:
        model_manager.start()
    except Exception as exc:
        log(f"Model server failed to start: {exc}", logging.ERROR)
        log("Web UI will still start. Check logs/model_server.log.", logging.WARNING)

    if config.retrain_now:
        def delayed_retrain():
            time.sleep(3)
            retrain_manager.run("cli")
        threading.Thread(target=delayed_retrain, daemon=True).start()

    try:
        uvicorn.run(app, host="127.0.0.1", port=config.web_port, log_level="warning", access_log=False)
    except KeyboardInterrupt:
        log("Shutting down...")
    finally:
        model_manager.stop()
        db.close()


if __name__ == "__main__":
    main()
