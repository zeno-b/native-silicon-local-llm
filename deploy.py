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
   python3 deploy.py --list-models
   python3 deploy.py --bench
   python3 deploy.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter none
   python3 deploy.py --add-task '{"name": "news", "goal": "Search for MLX news", "interval_seconds": 3600}'

Performance notes for an 8GB machine:
- The dominant cost in a multi-step agent run is re-prefill, not decode. The
  whole prompt is re-sent every step, so total prefill is quadratic in the step
  count. Run --bench to see the curve on your hardware, and watch
  /api/metrics/run/<conversation_id> for prompt tokens climbing step over step.
- What attacks that: a small tool catalogue (AGENT_TOOLS, or per-task tools),
  small tool results in context (TOOL_RESULT_CHARS), summarising long results
  instead of truncating them, and a stable prompt prefix (STABLE_PREFIX) so a
  server-side prompt cache can reuse the previous step.
- Reasoning models (Qwen3.5 and later) think by default. In a tool loop that is
  pure cost, so DISABLE_THINKING is on and any <think> block that arrives is
  stripped before the reply is parsed.
- Quantization damages format adherence before it damages knowledge, and the
  tool loop depends on format adherence. An 8-bit small model can beat a 4-bit
  larger one at agent work. Both are in the catalogue so you can measure it.
- Tasks may name their own model. Interactive chat wants a small fast model; a
  scheduled 3am task has nobody waiting and can afford a bigger one. The task
  swaps up, runs, and swaps back.

Environment overrides:
   MODEL_ID, MODEL_PORT, WEB_PORT, TRAIN_ITERS, TRAIN_LR, TRAIN_SEQ_LEN,
   MAX_TOKENS, SYSTEM_PROMPT, AUTO_RETRAIN_THRESHOLD,
   CONTEXT_SIZE, MAX_KV_SIZE, TEMPERATURE, HISTORY_TURNS,
   AGENT_ENABLED, AGENT_MAX_STEPS, ALLOW_PYTHON, ALLOW_SHELL,
   AGENT_TOOLS (comma-separated allowlist), ALLOW_LOCAL_FETCH,
   ADAPTER (latest|none|<backup id>), MODEL_CATALOG,
   MAX_CONCURRENT_TASKS, TASK_POLL_SECONDS,
   SEARCH_BACKEND (ddg|brave|tavily|searxng), SEARCH_RESULTS,
   TOOL_TIMEOUT, TOOL_RESULT_CHARS, TOOL_RAW_CHARS,
   DISABLE_THINKING, TOOL_TEMPERATURE, FAST_PATH, STABLE_PREFIX,
   SUMMARISE_TOOL_RESULTS, SUMMARISE_OVER_CHARS, CHAT_IDLE_SECONDS,
   KV_BITS, KV_GROUP_SIZE, QUANTIZED_KV_START, PROMPT_CACHE_DIR,
   TRAIN_NUM_LAYERS, TRAIN_ON_TOOL_CALLS, TRAIN_TOOL_EXAMPLES,
   BRAVE_API_KEY, TAVILY_API_KEY, SEARXNG_URL

Notes:
- This script creates ./.venv and installs dependencies on first run.
- It stores data in ./data and logs in ./logs.
- Agent tools that touch the filesystem are confined to ./workspace.
- fetch_url refuses loopback and private addresses unless ALLOW_LOCAL_FETCH=1.
- run_python and run_shell are off unless you pass --allow-python/--allow-shell.
  They are not sandboxed: the model gets the same rights as the user running it.
- Retraining stops the model server temporarily, trains, then restarts it.
- Training now also learns from logged tool calls, not just thumbs-up feedback.
  Format adherence is what small models fail at, and those rows are this app's
  own schema in its own wording.
- Tasks are agent runs that happen on their own: give one a goal and an
  interval, and the scheduler runs it and keeps every step of every run. The
  Tasks tab in the web UI streams a live run and replays finished ones.
- Models can be swapped from the Models tab without restarting this process.
  Uncached weights download on first use; watch the model server log for it.
- Run --selftest after editing this file. It checks the embedded UI, the
  database schema, tool parsing and the workspace guard without touching the
  network or loading a model.
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SFT_DIR = DATA_DIR / "sft"
LOG_DIR = ROOT / "logs"
ADAPTER_DIR = ROOT / "adapters" / "latest"
ADAPTER_BACKUP_DIR = ROOT / "adapters" / "backups"
WORKSPACE_DIR = ROOT / "workspace"
DB_PATH = DATA_DIR / "feedback.db"

def _detect_total_ram_gb() -> float:
    """Physical RAM in GB, so the default model can scale to the machine.

    Uses sysctl hw.memsize on macOS (the target platform), falls back to
    os.sysconf where available, and returns 8.0 if neither works. Any failure is
    non-fatal: a wrong guess only affects which default model is chosen, and an
    explicit MODEL_ID always overrides it.
    """
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        return 8.0


def _default_model_for_ram(ram_gb: float) -> str:
    """Pick a coding model that fits comfortably alongside the KV cache, the web
    process and headroom for training, given the machine's RAM.

    The thresholds are deliberately conservative because unified memory is shared
    with the OS and the GPU wired limit. Bigger machines get a bigger, stronger
    model; 8GB Macs stay on the 3B that has carried this setup. Override any of
    this with MODEL_ID.
    """
    if ram_gb >= 48:
        return "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"   # ~18GB
    if ram_gb >= 24:
        return "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"   # ~8GB
    if ram_gb >= 14:
        return "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"    # ~4.3GB
    return "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"        # ~1.9GB


def _default_context_for_ram(ram_gb: float) -> int:
    """Pick a context window sized to the machine.

    Context sits in the KV cache, which shares unified memory with the model
    weights, the web process, and the OS. Because bigger machines also run bigger
    models (see _default_model_for_ram), these tiers account for the heavier
    weights and still leave headroom: an 8GB Mac stays at a safe 4k, roomier
    machines get proportionally more so they chunk less and keep more history.
    The chunking thresholds derive from this value, so making context RAM-aware
    makes chunk sizing RAM-aware too. Override with CONTEXT_SIZE.
    """
    if ram_gb >= 48:
        return 32768
    if ram_gb >= 24:
        return 16384
    if ram_gb >= 14:
        return 8192
    return 4096


def _default_reasoning_tokens(ram_gb: float) -> int:
    """Per-step token budget for reasoning, chunk and source-extraction passes.

    Small on 8GB to keep each pass cheap and memory-light; larger on roomy
    machines so they can think in more depth per step. Override REASONING_TOKENS.
    """
    if ram_gb >= 48:
        return 1024
    if ram_gb >= 24:
        return 768
    if ram_gb >= 14:
        return 512
    return 256


def _default_fetch_cap(ram_gb: float) -> int:
    """Hard character cap on a fetched page entering the prompt.

    Scales with RAM so bigger machines read more of a source per pass while an
    8GB machine stays safe. Override AUTO_FETCH_CHAR_CAP.
    """
    if ram_gb >= 48:
        return 24000
    if ram_gb >= 24:
        return 16000
    if ram_gb >= 14:
        return 10000
    return 6000


# Total RAM is detected once at import. MODEL_ID overrides the RAM-based choice.
TOTAL_RAM_GB = _detect_total_ram_gb()
DEFAULT_MODEL = os.environ.get("MODEL_ID") or _default_model_for_ram(TOTAL_RAM_GB)
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a local coding and research assistant. Prefer complete, runnable "
    "code over prose. State the language and any assumptions in one line, then "
    "give the code. Say plainly when you do not know an API rather than "
    "inventing one.\n"
    # The routing layer sends look-up and current-events requests to web_search
    # before you see them, so when a TOOL RESULT block is present, the work is
    # to read those results and answer from them, with sources, not to decide
    # whether the topic is allowed. Security topics are in scope for this tool: "
    "When a message includes web search results, base your answer on them and "
    "cite the URLs. Do not refuse to report on publicly documented software "
    "vulnerabilities, CVEs, or the existence of published exploits and tools "
    "(for example Metasploit modules or exploit-db entries): this is standard "
    "defensive security research and the information is already public. Explain "
    "what a vulnerability is, how it was fixed, and what public tooling exists, "
    "and point to primary sources. Decline only a request to produce a working "
    "exploit against a specified live target, and offer the defensive framing "
    "instead.",
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
UI_BUILD = "2026-08-07.9-glass"
# Branding. Set APP_NAME to change the title shown in the header and browser tab.
# Set APP_LOGO to a URL or a local path (rendered as an image) or to an emoji or
# short text (rendered as-is). Both are safe to leave unset.
APP_NAME = os.environ.get("APP_NAME", "Local LLM")
APP_LOGO = os.environ.get("APP_LOGO", "")

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


REQUIRED_MODULES = ["mlx_lm", "fastapi", "uvicorn", "httpx", "pydantic"]
REQUIRED_PACKAGES = ["mlx-lm", "fastapi", "uvicorn", "httpx", "pydantic"]
# Guards the install-then-re-exec handoff. A package that installs cleanly but
# still cannot be imported (mlx-lm off Apple Silicon, a broken wheel) otherwise
# sends bootstrap round the same install and exec forever, with nothing on
# stdout but pip repeating itself.
BOOTSTRAP_MARKER = "LOCAL_LLM_BOOTSTRAP_ATTEMPT"


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

        # Probe before installing. The previous version ran a pip upgrade and a
        # full install on every launch, which is several seconds on every
        # --list-models, --list-tasks or --doctor even when nothing is missing.
        probe = subprocess.run(
            [str(venv_python), "-c", "import " + ", ".join(REQUIRED_MODULES)],
            capture_output=True,
        )
        if probe.returncode != 0:
            log("Upgrading pip...")
            try:
                run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
            except Exception as exc:
                log(f"Warning: pip upgrade failed: {exc}", logging.WARNING)

            log("Installing dependencies...")
            run_cmd([
                str(venv_python), "-m", "pip", "install", *REQUIRED_PACKAGES,
            ])

        os.environ[BOOTSTRAP_MARKER] = str(int(os.environ.get(BOOTSTRAP_MARKER, "0")) + 1)
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:],
        )

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        attempt = int(os.environ.get(BOOTSTRAP_MARKER, "0"))
        if attempt >= 2:
            sys.exit(
                "Cannot import " + ", ".join(missing) + " even after installing "
                "them into " + sys.prefix + ".\n"
                "mlx-lm only imports on Apple Silicon: on any other machine this "
                "script can still run --selftest, --doctor and --list-models, but "
                "not the model server.\n"
                "Otherwise delete ./.venv and try again, or install by hand with:\n"
                f"  {sys.executable} -m pip install " + " ".join(REQUIRED_PACKAGES)
            )
        packages = ["mlx-lm" if m == "mlx_lm" else m for m in missing]
        log("Installing missing dependencies: " + ", ".join(packages))
        run_cmd([sys.executable, "-m", "pip", "install", *packages])
        os.environ[BOOTSTRAP_MARKER] = str(attempt + 1)
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
        )


def ensure_dirs() -> None:
    for d in [DATA_DIR, SFT_DIR, LOG_DIR, ADAPTER_DIR.parent, ADAPTER_BACKUP_DIR, WORKSPACE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


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
    pinned: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Drop the oldest history until the request fits the context window.

    Returns the message list actually sent and the number of dropped messages.
    The system prompt, anything in `pinned`, and the current user turn are never
    dropped: if those alone exceed the budget the caller has a configuration
    problem, not a history problem, and silently truncating them would hide it.

    `pinned` exists for the agent loop. Without it the oldest message dropped
    from a long tool trace is the original question, and the model ends up
    summarising tool output while no longer knowing what was asked.
    """
    pinned = list(pinned or [])
    budget = max(256, context_size - reserve - CONTEXT_SAFETY_MARGIN)
    fixed = messages_tokens([system, *pinned, user])
    kept = list(history)
    dropped = 0
    while kept and fixed + messages_tokens(kept) > budget:
        kept.pop(0)
        dropped += 1
    return [system, *pinned, *kept, user], dropped


class Database:
    """Thread-safe SQLite manager with one connection per thread."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        # Bumped by close(). A thread holding a connection from an older
        # generation reconnects instead of raising ProgrammingError, which is
        # what happened when close() ran on the main thread while a worker
        # thread still had its own handle cached.
        self._generation = 0
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None and getattr(self._local, "generation", -1) != self._generation:
            conn = None
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
            self._local.generation = self._generation
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
                    error TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    ttft_ms REAL,
                    decode_tps REAL,
                    model TEXT,
                    step INTEGER,
                    conversation_id TEXT
                )
            """)
            # Migration: older databases have the four-column metrics table.
            metric_columns = {row["name"] for row in conn.execute("PRAGMA table_info(metrics)")}
            for column, ddl in (
                ("prompt_tokens", "INTEGER"), ("completion_tokens", "INTEGER"),
                ("ttft_ms", "REAL"), ("decode_tps", "REAL"), ("model", "TEXT"),
                ("step", "INTEGER"), ("conversation_id", "TEXT"),
            ):
                if column not in metric_columns:
                    conn.execute(f"ALTER TABLE metrics ADD COLUMN {column} {ddl}")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_endpoint
                ON metrics(endpoint, timestamp)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    interval_seconds INTEGER DEFAULT 0,
                    max_steps INTEGER DEFAULT 6,
                    tools TEXT DEFAULT '',
                    system_prompt TEXT,
                    use_history INTEGER DEFAULT 0,
                    model TEXT,
                    next_task_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_run_at TIMESTAMP,
                    next_run_at TIMESTAMP,
                    run_count INTEGER DEFAULT 0,
                    last_status TEXT,
                    last_answer TEXT
                )
            """)
            # Migration: model override and chaining came later.
            task_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
            for column in ("model", "next_task_id"):
                if column not in task_columns:
                    conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    trigger TEXT,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    steps INTEGER DEFAULT 0,
                    answer TEXT,
                    error TEXT,
                    elapsed_ms REAL,
                    tools_used TEXT,
                    model TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_runs_task
                ON task_runs(task_id, started_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    type TEXT NOT NULL,
                    payload TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_events_run
                ON task_events(run_id, seq)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    conversation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
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
            self._generation += 1
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

    def remember(self, key: str, value: str, conversation_id: str | None = None) -> None:
        """Upsert a durable note the agent can read back in a later session."""
        stamp = datetime.now(timezone.utc).isoformat()
        self.execute(
            "INSERT INTO memories (key, value, conversation_id, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "conversation_id = excluded.conversation_id, updated_at = excluded.updated_at",
            (key, value, conversation_id, stamp),
        )
        self.commit()

    def recall(self, query: str | None = None, limit: int = 10) -> list[dict]:
        if query:
            rows = self.execute(
                "SELECT * FROM memories WHERE key LIKE ? OR value LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        else:
            rows = self.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------------- tasks --

    TASK_FIELDS = (
        "name", "goal", "enabled", "interval_seconds", "max_steps",
        "tools", "system_prompt", "use_history", "model", "next_task_id",
    )

    def create_task(self, **fields: Any) -> dict:
        task_id = str(uuid.uuid4())[:12]
        interval = int(fields.get("interval_seconds") or 0)
        enabled = 1 if fields.get("enabled", True) else 0
        # A repeating task is due immediately so the operator sees a first run
        # rather than waiting out a one-hour interval to find out it is wrong.
        next_run = iso(utc_now()) if (enabled and interval > 0) else None
        self.execute(
            "INSERT INTO tasks (id, name, goal, enabled, interval_seconds, max_steps, "
            "tools, system_prompt, use_history, model, next_task_id, next_run_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                str(fields.get("name") or "task").strip()[:120],
                str(fields.get("goal") or "").strip(),
                enabled,
                interval,
                int(fields.get("max_steps") or 6),
                str(fields.get("tools") or ""),
                fields.get("system_prompt"),
                1 if fields.get("use_history") else 0,
                fields.get("model") or None,
                fields.get("next_task_id") or None,
                next_run,
            ),
        )
        self.commit()
        return self.get_task(task_id) or {}

    def get_task(self, task_id: str) -> dict | None:
        row = self.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self) -> list[dict]:
        rows = self.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_task(self, task_id: str, updates: dict) -> dict | None:
        task = self.get_task(task_id)
        if task is None:
            return None
        sets, params = [], []
        for key in self.TASK_FIELDS:
            if key not in updates or updates[key] is None:
                continue
            value = updates[key]
            if key in ("enabled", "use_history"):
                value = 1 if value else 0
            elif key in ("interval_seconds", "max_steps"):
                value = int(value)
            elif key in ("model", "next_task_id"):
                value = str(value).strip() or None
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return task

        # Re-arm or disarm the schedule to match the new settings, rather than
        # leaving a stale next_run_at that fires a task the user just disabled.
        enabled = updates.get("enabled", task["enabled"])
        interval = int(updates.get("interval_seconds", task["interval_seconds"]) or 0)
        if not enabled or interval <= 0:
            next_run = None
        elif task["next_run_at"] and int(task["interval_seconds"] or 0) == interval and task["enabled"]:
            next_run = task["next_run_at"]
        else:
            next_run = iso(utc_now())
        sets.append("next_run_at = ?")
        params.append(next_run)
        sets.append("updated_at = ?")
        params.append(iso(utc_now()))
        params.append(task_id)
        self.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        self.commit()
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> bool:
        run_ids = [row["id"] for row in
                   self.execute("SELECT id FROM task_runs WHERE task_id = ?", (task_id,)).fetchall()]
        for run_id in run_ids:
            self.execute("DELETE FROM task_events WHERE run_id = ?", (run_id,))
        self.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        cursor = self.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.commit()
        return cursor.rowcount > 0

    def due_tasks(self) -> list[dict]:
        rows = self.execute(
            "SELECT * FROM tasks WHERE enabled = 1 AND next_run_at IS NOT NULL "
            "AND next_run_at <= ? ORDER BY next_run_at",
            (iso(utc_now()),),
        ).fetchall()
        return [dict(row) for row in rows]

    def schedule_next(self, task_id: str, interval_seconds: int) -> None:
        from datetime import timedelta
        next_run = (iso(utc_now() + timedelta(seconds=interval_seconds))
                    if interval_seconds > 0 else None)
        self.execute("UPDATE tasks SET next_run_at = ? WHERE id = ?", (next_run, task_id))
        self.commit()

    # ----------------------------------------------------------------- runs --

    def create_run(self, task_id: str, trigger: str, model: str) -> str:
        run_id = str(uuid.uuid4())[:16]
        self.execute(
            "INSERT INTO task_runs (id, task_id, trigger, status, started_at, model) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            (run_id, task_id, trigger, iso(utc_now()), model),
        )
        self.execute(
            "UPDATE tasks SET last_run_at = ?, last_status = 'running', "
            "run_count = run_count + 1 WHERE id = ?",
            (iso(utc_now()), task_id),
        )
        self.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        answer: str = "",
        error: str | None = None,
        steps: int = 0,
        elapsed_ms: float = 0.0,
        tools_used: list[str] | None = None,
    ) -> None:
        self.execute(
            "UPDATE task_runs SET status = ?, finished_at = ?, steps = ?, answer = ?, "
            "error = ?, elapsed_ms = ?, tools_used = ? WHERE id = ?",
            (status, iso(utc_now()), steps, answer, error, elapsed_ms,
             ",".join(tools_used or []), run_id),
        )
        row = self.execute("SELECT task_id FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        if row:
            self.execute(
                "UPDATE tasks SET last_status = ?, last_answer = ? WHERE id = ?",
                (status, answer, row["task_id"]),
            )
        self.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, task_id: str | None = None, limit: int = 20) -> list[dict]:
        if task_id:
            rows = self.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT ?",
                (task_id, limit),
            ).fetchall()
        else:
            rows = self.execute(
                "SELECT * FROM task_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def append_event(self, run_id: str, seq: int, event_type: str, payload: dict) -> None:
        self.execute(
            "INSERT INTO task_events (run_id, seq, type, payload) VALUES (?, ?, ?, ?)",
            (run_id, seq, event_type, json.dumps(payload, ensure_ascii=False, default=str)[:20000]),
        )
        self.commit()

    def run_events(self, run_id: str, after_seq: int = 0, limit: int = 500) -> list[dict]:
        rows = self.execute(
            "SELECT seq, type, payload, created_at FROM task_events "
            "WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (run_id, after_seq, limit),
        ).fetchall()
        events = []
        for row in rows:
            try:
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except json.JSONDecodeError:
                payload = {}
            payload.update({"seq": row["seq"], "type": row["type"], "at": row["created_at"]})
            events.append(payload)
        return events

    def prune_runs(self, task_id: str, keep: int = 25) -> int:
        stale = self.execute(
            "SELECT id FROM task_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT -1 OFFSET ?",
            (task_id, keep),
        ).fetchall()
        for row in stale:
            self.execute("DELETE FROM task_events WHERE run_id = ?", (row["id"],))
            self.execute("DELETE FROM task_runs WHERE id = ?", (row["id"],))
        self.commit()
        return len(stale)

    def reset_orphan_runs(self) -> int:
        """Mark runs that were live when the process died, so nothing shows as running forever."""
        cursor = self.execute(
            "UPDATE task_runs SET status = 'interrupted', finished_at = ?, "
            "error = 'process exited during this run' WHERE status = 'running'",
            (iso(utc_now()),),
        )
        self.execute("UPDATE tasks SET last_status = 'interrupted' WHERE last_status = 'running'")
        self.commit()
        return cursor.rowcount

    def count_memories(self) -> int:
        return self.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]

    def forget(self, key: str) -> bool:
        cursor = self.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.commit()
        return cursor.rowcount > 0

    def log_metric(
        self,
        endpoint: str,
        duration_ms: float,
        status_code: int,
        error: str | None = None,
        stats: "GenerationStats | None" = None,
        model: str | None = None,
        step: int | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.execute(
            "INSERT INTO metrics (endpoint, duration_ms, status_code, error, prompt_tokens, "
            "completion_tokens, ttft_ms, decode_tps, model, step, conversation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint, duration_ms, status_code, error,
                stats.prompt_tokens if stats else None,
                stats.completion_tokens if stats else None,
                round(stats.ttft_ms, 1) if stats and stats.ttft_ms else None,
                round(stats.decode_tps, 2) if stats and stats.decode_tps else None,
                model, step, conversation_id,
            ),
        )
        self.commit()

    def metric_summary(self, endpoint: str | None = None, limit: int = 500) -> dict:
        """Aggregate the recent metrics. This is what tells you if a change helped."""
        clause = "WHERE endpoint = ?" if endpoint else ""
        params: tuple = (endpoint, limit) if endpoint else (limit,)
        rows = self.execute(
            f"SELECT * FROM (SELECT * FROM metrics {clause} ORDER BY id DESC LIMIT ?)",
            params,
        ).fetchall()
        if not rows:
            return {"samples": 0}

        def mean(key: str) -> float | None:
            values = [row[key] for row in rows if row[key] is not None]
            return round(sum(values) / len(values), 2) if values else None

        prompts = [row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None]
        return {
            "samples": len(rows),
            "avg_duration_ms": mean("duration_ms"),
            "avg_prompt_tokens": mean("prompt_tokens"),
            "avg_completion_tokens": mean("completion_tokens"),
            "avg_ttft_ms": mean("ttft_ms"),
            "avg_decode_tps": mean("decode_tps"),
            "total_prompt_tokens": sum(prompts) if prompts else 0,
            "errors": sum(1 for row in rows if row["error"]),
        }

    def run_step_metrics(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Per-step prompt token counts, so re-prefill growth is visible as a curve."""
        rows = self.execute(
            "SELECT step, prompt_tokens, completion_tokens, ttft_ms, decode_tps, duration_ms "
            "FROM metrics WHERE conversation_id = ? AND step IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


LOG_FILES = {"model": "model_server.log", "train": "train.log", "tasks": "tasks.log"}


def tail_log(name: str, lines: int = 120) -> str:
    """Return the last N lines of one of the app's log files.

    The model server writes its download progress here, so the UI can show why a
    swap to an uncached model is taking four minutes instead of appearing hung.
    """
    filename = LOG_FILES.get(name)
    if filename is None:
        raise ValueError(f"unknown log: {name}")
    path = LOG_DIR / filename
    if not path.exists():
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        block = min(size, max(4096, lines * 200))
        handle.seek(size - block)
        raw = handle.read()
    text = raw.decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


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


class StartupCancelled(RuntimeError):
    """Raised when a stop request arrives while the model server is still loading."""


def wait_for_port(
    port: int,
    timeout: int = 300,
    proc: subprocess.Popen | None = None,
    cancel: threading.Event | None = None,
) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if cancel is not None and cancel.is_set():
            raise StartupCancelled("startup cancelled")
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
    cancel: threading.Event | None = None,
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
        if cancel is not None and cancel.is_set():
            raise StartupCancelled("startup cancelled")
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
            body = exc.read(200).decode("utf-8", "replace")
            # 5xx and 429 are what a server that is up but still loading weights
            # returns. Only a genuine routing or contract error is fatal, so the
            # probe no longer aborts a working startup on one transient 500.
            if exc.code >= 500 or exc.code == 429:
                last_error = f"HTTP {exc.code} {body[:120]}"
            else:
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

        if char == "\\":
            index += 2
            continue
        if char == quote:
            quote = None
        index += 1

    if quote is not None:
        problems.append(f"line {quote_line}: unterminated {quote} string literal at end of script")
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


def adapter_ready(path: Path | None = None) -> bool:
    target = path or ADAPTER_DIR
    return target.exists() and any(target.iterdir())


# Written into an adapter directory at the end of a successful training run.
# A LoRA adapter only fits the base model it was trained on: handing a Qwen
# adapter to a Llama server is a shape mismatch, and the start would fall back
# to the base model silently. Recording the base makes the mismatch visible and
# skippable instead.
ADAPTER_BASE_FILE = "base_model.txt"


def adapter_base(path: Path) -> str | None:
    """The model an adapter was trained against, or None for adapters predating this."""
    marker = path / ADAPTER_BASE_FILE
    try:
        return marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_adapter_base(path: Path, model_id: str) -> None:
    try:
        (path / ADAPTER_BASE_FILE).write_text(model_id + "\n", encoding="utf-8")
    except OSError as exc:
        log(f"Could not record the adapter base model: {exc}", logging.WARNING)


def adapter_fits(path: Path, model_id: str) -> bool:
    """An untagged adapter is trusted; a tagged one must match the current model."""
    recorded = adapter_base(path)
    return recorded is None or recorded == model_id


def list_adapters() -> list[dict]:
    """Every adapter that can be loaded: the live one plus every backup."""
    entries: list[dict] = []
    if adapter_ready(ADAPTER_DIR):
        stat = ADAPTER_DIR.stat()
        entries.append({
            "id": "latest",
            "path": str(ADAPTER_DIR),
            "base_model": adapter_base(ADAPTER_DIR),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    if ADAPTER_BACKUP_DIR.exists():
        for item in sorted(ADAPTER_BACKUP_DIR.iterdir(), reverse=True):
            if item.is_dir() and adapter_ready(item):
                entries.append({
                    "id": item.name,
                    "path": str(item),
                    "base_model": adapter_base(item),
                    "modified": datetime.fromtimestamp(
                        item.stat().st_mtime, timezone.utc).isoformat(),
                })
    return entries


def resolve_adapter_quietly(choice: str | None) -> Path | None:
    """resolve_adapter without raising, for read-only reporting."""
    try:
        return resolve_adapter(choice)
    except ValueError:
        return None


def resolve_adapter(choice: str | None) -> Path | None:
    """Map an adapter id from the UI onto a directory, or None for the base model."""
    if not choice or choice == "none":
        return None
    if choice == "latest":
        return ADAPTER_DIR if adapter_ready(ADAPTER_DIR) else None
    candidate = (ADAPTER_BACKUP_DIR / choice).resolve()
    if ADAPTER_BACKUP_DIR.resolve() not in candidate.parents:
        raise ValueError("adapter id escapes the backups directory")
    if not adapter_ready(candidate):
        raise ValueError(f"no adapter called {choice}")
    return candidate


# Small enough to run on 8GB alongside the web process. Anything on Hugging
# Face works too; the UI accepts a free-text repo id, so this list is a
# convenience rather than a limit. Sizes are the 4-bit download, roughly.
#
# Two notes that matter more than the ordering:
#   - Quantization damages instruction-following and strict format adherence
#     earlier than it damages world knowledge, and format adherence is exactly
#     what the tool loop depends on. An 8-bit 1.5B can beat a 4-bit 4B at
#     agent work even when every general benchmark says otherwise. The 8-bit
#     entries below are here to make that easy to test.
#   - Qwen3.5 and later are reasoning models with thinking on by default.
#     disable_thinking (on by default) turns it off, and the agent strips any
#     <think> block that arrives anyway.
DEFAULT_MODEL_CATALOG = [
    # Coding-specialised. Sizes are resident weights; add ~0.5-1.5GB for the KV
    # cache at the context lengths this app uses.
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",  # ~4.3GB, best code quality that
                                                     # loads on 8GB. Inference only:
                                                     # raise the wired limit first and
                                                     # do not retrain against it.
    "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",  # ~1.9GB, default. Trains fine.
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",# ~1.0GB, fast completion-style
    # General purpose.
    "mlx-community/Qwen3.5-4B-OptiQ-4bit",       # ~2.5GB, mixed precision, agent-calibrated
    "mlx-community/Qwen3.5-4B-MLX-4bit",         # ~2.5GB, stock uniform 4-bit
    "mlx-community/Qwen2.5-3B-Instruct-4bit",    # ~1.7GB, no thinking mode
    "mlx-community/Qwen2.5-1.5B-Instruct-8bit",  # ~1.6GB, 8-bit: better format adherence
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",  # ~0.9GB
    "mlx-community/Llama-3.2-3B-Instruct-4bit",  # ~1.8GB
    "mlx-community/Llama-3.2-1B-Instruct-4bit",  # ~0.8GB
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # ~0.4GB, fastest, weakest at tools
]


def hf_cache_dir() -> Path:
    """Where huggingface_hub keeps downloaded repos."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_is_cached(model_id: str) -> bool:
    """True when the weights are already on disk, so switching is instant.

    A miss is not an error: mlx_lm.server downloads on first use. The UI shows
    this so a switch that is about to pull several gigabytes says so up front.
    """
    if Path(model_id).expanduser().is_dir():
        return True
    folder = "models--" + model_id.replace("/", "--")
    path = hf_cache_dir() / folder / "snapshots"
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def model_catalog(config: "Config") -> list[dict]:
    ids = [item.strip() for item in (config.model_catalog or "").split(",") if item.strip()]
    for known in DEFAULT_MODEL_CATALOG:
        if known not in ids:
            ids.append(known)
    if config.model not in ids:
        ids.insert(0, config.model)
    return [
        {"id": item, "cached": model_is_cached(item), "current": item == config.model}
        for item in ids
    ]


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_port: int = field(default_factory=lambda: int(os.environ.get("MODEL_PORT", "8080")))
    web_port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8000")))
    # 30 iterations is a warmup, not a fine-tune. A format tune needs a few
    # hundred passes over a hundred-odd examples to change anything.
    train_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_ITERS", "300")))
    train_lr: str = field(default_factory=lambda: os.environ.get("TRAIN_LR", "3e-5"))
    train_seq_len: str = field(default_factory=lambda: os.environ.get("TRAIN_SEQ_LEN", "512"))
    # Tuning only the top layers is what keeps a 3B trainable on 8GB while the
    # web process and the page cache are also resident.
    train_num_layers: int = field(default_factory=lambda: int(os.environ.get("TRAIN_NUM_LAYERS", "8")))
    train_on_tool_calls: bool = field(default_factory=lambda: os.environ.get("TRAIN_ON_TOOL_CALLS", "1") == "1")
    train_tool_examples: int = field(default_factory=lambda: int(os.environ.get("TRAIN_TOOL_EXAMPLES", "400")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "512")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    # Context window. context_size is the budget this process enforces when it
    # assembles a request; max_kv_size is what the model server is told to
    # allocate. They are separate because the server flag is optional and
    # changing it needs a restart, while context_size takes effect immediately.
    context_size: int = field(default_factory=lambda: int(
        os.environ.get("CONTEXT_SIZE") or _default_context_for_ram(TOTAL_RAM_GB)))
    max_kv_size: int = field(default_factory=lambda: int(os.environ.get("MAX_KV_SIZE", "0")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.7")))
    history_turns: int = field(default_factory=lambda: int(os.environ.get("HISTORY_TURNS", "20")))

    # Agent
    agent_enabled: bool = field(default_factory=lambda: os.environ.get("AGENT_ENABLED", "0") == "1")
    # Before answering a substantive question, ask the model in one word whether
    # it needs to look the answer up. This is the general form of the fast-path
    # regexes: instead of enumerating every phrasing, let the model judge, but
    # in a shape a small model handles well (a single SEARCH/ANSWER token) and
    # then seed the search deterministically so it does not depend on the model
    # emitting tool-call JSON. Biased toward SEARCH when unsure, since a needless
    # search is cheaper than a confident wrong answer or a refusal.
    knowledge_triage: bool = field(default_factory=lambda: os.environ.get("KNOWLEDGE_TRIAGE", "1") == "1")
    # After a lookup search, automatically fetch this many of the top result
    # pages and give the model their full text, not just the snippet. This is
    # what makes one generic search path answer domain-specific questions (a
    # stock price, a score, a forecast, a version number): the answer is usually
    # on the page even when the snippet omits it. 0 disables auto-fetch and falls
    # back to snippet-only plus the model choosing to call fetch_url itself.
    auto_fetch_results: int = field(default_factory=lambda: int(os.environ.get("AUTO_FETCH_RESULTS", "2")))
    # Hardest ceiling on characters of a fetched page that may enter the prompt.
    # A large docs page would otherwise dominate context and OOM prefill on an
    # 8GB machine. Roughly char_cap/4 tokens.
    auto_fetch_char_cap: int = field(default_factory=lambda: int(
        os.environ.get("AUTO_FETCH_CHAR_CAP") or _default_fetch_cap(TOTAL_RAM_GB)))
    agent_max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "6")))
    # Resilience under memory pressure. When a generation errors (a model-server
    # OOM kill and watchdog restart look like a dropped connection from here), the
    # turn retries with a smaller token budget rather than surfacing an error.
    # resilient_retries is how many times; min_max_tokens is the floor it shrinks
    # to. hard_step_cap lets a big task keep going past agent_max_steps by
    # compacting progress into a running summary, so it slows down but does not
    # stop. These exist to satisfy "never error or stop; step down and continue".
    resilient_retries: int = field(default_factory=lambda: int(os.environ.get("RESILIENT_RETRIES", "3")))
    min_max_tokens: int = field(default_factory=lambda: int(os.environ.get("MIN_MAX_TOKENS", "128")))
    # If the model server sends nothing for this many seconds mid-generation, the
    # request is treated as stalled: it raises, and the resilient loop retries
    # with a smaller budget instead of hanging. This is what turns "stuck" into
    # visible "retrying" rather than minutes of dead air waiting on a wedged or
    # OOM-killed server.
    stall_timeout: int = field(default_factory=lambda: int(os.environ.get("STALL_TIMEOUT", "60")))
    # How long a retry waits for a restarting model server to become ready again
    # before giving up on that attempt. Longer helps slow cold-start reloads.
    ready_wait_timeout: float = field(default_factory=lambda: float(os.environ.get("READY_WAIT_TIMEOUT", "40")))
    hard_step_cap: int = field(default_factory=lambda: int(os.environ.get("HARD_STEP_CAP", "24")))
    # Incremental reasoning: for a hard analytical question with no tool to call,
    # decompose it into sub-steps and solve them one at a time, carrying only
    # short conclusions forward. Each pass is small, so the working set stays
    # inside 8GB no matter how deep the reasoning goes, and decomposition makes a
    # small model reason better than one shot. It is slower (several small calls)
    # by design: it takes its time instead of failing or answering shallowly.
    incremental_reasoning: bool = field(default_factory=lambda: os.environ.get("INCREMENTAL_REASONING", "1") == "1")
    # Chunk an oversized prompt (a pasted file or long document) and process it
    # part by part, so a single input larger than the context never has to be
    # prefilled in one pass. This is what lets an 8GB machine handle a large
    # prompt: split it, extract findings per chunk into bounded notes, then
    # synthesise. Chunk size and trigger are derived from context_size.
    chunk_large_prompts: bool = field(default_factory=lambda: os.environ.get("CHUNK_LARGE_PROMPTS", "1") == "1")
    # Emit verbose "under the hood" detail events (prompt sizes, per-step timing,
    # extracted lengths, fallback reasons) so the whole pipeline is visible.
    show_internals: bool = field(default_factory=lambda: os.environ.get("SHOW_INTERNALS", "1") == "1")
    # Hard wall-clock ceiling for the whole multi-source retrieval phase (fetch +
    # read all sources). Prevents a stalling extraction from grinding for minutes.
    retrieval_deadline: float = field(default_factory=lambda: float(os.environ.get("RETRIEVAL_DEADLINE", "90")))
    # Fraction of the context above which a prompt is chunked, and the fraction
    # of the context each chunk targets. Derived from context_size so they scale
    # with RAM; exposed so the thresholds themselves can be tuned per machine.
    chunk_trigger_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_TRIGGER_RATIO", "0.6")))
    chunk_size_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_SIZE_RATIO", "0.4")))
    reasoning_max_steps: int = field(default_factory=lambda: int(os.environ.get("REASONING_MAX_STEPS", "6")))
    # Hard wall-clock cap per reasoning step. Distinct from stall_timeout (which
    # only fires on zero output): this bounds a step that streams slowly but
    # never finishes, so a single step can never wedge the whole chain.
    reasoning_step_timeout: int = field(default_factory=lambda: int(os.environ.get("REASONING_STEP_TIMEOUT", "45")))
    # Per-step generation budget for reasoning, chunk and source-extraction
    # passes. RAM-scaled default; small on 8GB, larger on roomy machines.
    reasoning_tokens: int = field(default_factory=lambda: int(
        os.environ.get("REASONING_TOKENS") or _default_reasoning_tokens(TOTAL_RAM_GB)))
    allow_python: bool = field(default_factory=lambda: os.environ.get("ALLOW_PYTHON", "0") == "1")
    allow_shell: bool = field(default_factory=lambda: os.environ.get("ALLOW_SHELL", "0") == "1")
    # Comma-separated allowlist. Empty means every registered tool is offered.
    agent_tools: str = field(default_factory=lambda: os.environ.get("AGENT_TOOLS", ""))
    # Which LoRA adapter the model server loads: latest, none, or a backup id.
    adapter: str = field(default_factory=lambda: os.environ.get("ADAPTER", "latest"))
    # Extra model ids to offer in the switcher, on top of DEFAULT_MODEL_CATALOG.
    model_catalog: str = field(default_factory=lambda: os.environ.get("MODEL_CATALOG", ""))
    # How many background task runs may execute at once. The model server
    # serves one request at a time, so more than one mostly adds queueing.
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_TASKS", "1")))
    task_poll_seconds: int = field(default_factory=lambda: int(os.environ.get("TASK_POLL_SECONDS", "2")))
    # Seconds of interactive quiet before a scheduled run is allowed to start.
    chat_idle_seconds: int = field(default_factory=lambda: int(os.environ.get("CHAT_IDLE_SECONDS", "45")))
    # fetch_url refuses loopback and RFC1918 targets unless this is on, so a
    # prompt-injected page cannot make the agent read the machine's own
    # services (including this app's API) and hand the result back.
    allow_local_fetch: bool = field(default_factory=lambda: os.environ.get("ALLOW_LOCAL_FETCH", "0") == "1")

    # Reasoning-mode models (Qwen3.5 and later) emit a <think> block by default.
    # In a tool loop that is pure cost: the reasoning is discarded by the
    # protocol, it inflates the KV cache, and JSON inside it confuses parsing.
    disable_thinking: bool = field(default_factory=lambda: os.environ.get("DISABLE_THINKING", "1") == "1")
    # Tool-selection steps want deterministic JSON; only the final answer wants
    # the configured temperature. One value for both costs malformed calls.
    tool_temperature: float = field(default_factory=lambda: float(os.environ.get("TOOL_TEMPERATURE", "0.0")))
    # Multiplicative penalty on tokens already in the window. Small quantised
    # models fall into verbatim repetition loops, and greedy decoding
    # (tool_temperature 0.0) has no way out of one: the argmax that produced the
    # loop keeps producing it until max_tokens runs out. 1.0 disables it.
    repetition_penalty: float = field(default_factory=lambda: float(os.environ.get("REPETITION_PENALTY", "1.1")))
    repetition_context_size: int = field(default_factory=lambda: int(os.environ.get("REPETITION_CONTEXT_SIZE", "64")))
    # Answer arithmetic and bare URLs without a model round trip at all.
    fast_path: bool = field(default_factory=lambda: os.environ.get("FAST_PATH", "1") == "1")
    # When the trace outgrows the context, collapse the oldest steps into one
    # summary message instead of dropping them off the front. Dropping shifts
    # every following token and invalidates any server-side prefix cache at the
    # exact point a run is longest.
    stable_prefix: bool = field(default_factory=lambda: os.environ.get("STABLE_PREFIX", "1") == "1")

    # KV cache quantization, passed through to mlx_lm.server when the installed
    # build accepts the flags. 0 leaves the cache in fp16.
    kv_bits: int = field(default_factory=lambda: int(os.environ.get("KV_BITS", "0")))
    kv_group_size: int = field(default_factory=lambda: int(os.environ.get("KV_GROUP_SIZE", "64")))
    quantized_kv_start: int = field(default_factory=lambda: int(os.environ.get("QUANTIZED_KV_START", "1024")))
    prompt_cache_dir: str = field(default_factory=lambda: os.environ.get("PROMPT_CACHE_DIR", ""))

    # Tools
    search_backend: str = field(default_factory=lambda: os.environ.get("SEARCH_BACKEND", "ddg"))
    search_results: int = field(default_factory=lambda: int(os.environ.get("SEARCH_RESULTS", "5")))
    tool_timeout: int = field(default_factory=lambda: int(os.environ.get("TOOL_TIMEOUT", "30")))
    # Two different caps, and the difference matters.
    #
    # tool_raw_chars is how much a tool may return at all. It is what gets
    # logged and shown in the UI, and what the summariser reads.
    #
    # tool_result_chars is how much may enter the model's context. A result is
    # re-sent on every later step, so a 4000-character page is not a 4000-token
    # cost, it is that times the number of steps that follow.
    #
    # Collapsing these two into one number means the raw result is destroyed
    # before anything can summarise it, and the summariser becomes dead code.
    tool_raw_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RAW_CHARS", "20000")))
    tool_result_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RESULT_CHARS", "1500")))
    # Results longer than this get one cheap summarisation pass before they
    # enter the context. Worth it even at 15 tok/s because of the multiplier.
    summarise_tool_results: bool = field(default_factory=lambda: os.environ.get("SUMMARISE_TOOL_RESULTS", "1") == "1")
    summarise_over_chars: int = field(default_factory=lambda: int(os.environ.get("SUMMARISE_OVER_CHARS", "2500")))

    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"

    # Settings the web UI is allowed to change at runtime. Anything not listed
    # here needs a process restart and is rejected by /api/config.
    MUTABLE = (
        "system_prompt", "max_tokens", "temperature", "repetition_penalty",
        "repetition_context_size", "context_size",
        "history_turns", "agent_enabled", "agent_max_steps",
        "search_backend", "search_results", "tool_result_chars", "tool_raw_chars", "auto_fetch_results",
        "disable_thinking", "tool_temperature", "fast_path", "stable_prefix", "knowledge_triage",
        "summarise_tool_results", "summarise_over_chars",
        # Safeguards, all tunable live so a machine can be dialled in without a
        # restart or an env edit.
        "incremental_reasoning", "reasoning_max_steps", "reasoning_step_timeout",
        "reasoning_tokens", "chunk_large_prompts", "chunk_trigger_ratio",
        "chunk_size_ratio", "auto_fetch_char_cap", "stall_timeout", "ready_wait_timeout",
        "resilient_retries", "min_max_tokens", "hard_step_cap",
        "show_internals", "retrieval_deadline",
    )

    SEARCH_BACKENDS = ("ddg", "brave", "tavily", "searxng")

    def __post_init__(self) -> None:
        # Clamp at construction too, not only on live edits, so a bad value from
        # an environment variable at startup is corrected the same way the UI's
        # live edits are. apply({}) runs every guardrail with no other effect.
        self.apply({})

    def public(self) -> dict:
        data = {k: v for k, v in asdict(self).items()}
        data["mutable"] = list(self.MUTABLE)
        return data

    def apply(self, updates: dict) -> list[str]:
        """Apply a settings patch. Returns the names of the fields changed.

        Values are clamped afterwards, and any field the clamp moved is reported
        as changed too, so the UI never shows a setting the process did not
        actually adopt.
        """
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
            if key == "search_backend":
                value = str(value).lower()
                if value not in self.SEARCH_BACKENDS:
                    log(f"Ignoring unknown search backend: {value}", logging.WARNING)
                    continue
            if value != current:
                setattr(self, key, value)
                changed.append(key)

        before = {name: getattr(self, name) for name in
                  ("context_size", "max_tokens", "temperature", "agent_max_steps",
                   "history_turns", "search_results", "tool_result_chars",
                   "tool_temperature", "summarise_over_chars")}
        self.context_size = min(131072, max(512, self.context_size))
        self.max_tokens = max(16, self.max_tokens)
        if self.max_tokens >= self.context_size:
            self.max_tokens = max(16, self.context_size // 2)
        self.temperature = min(2.0, max(0.0, self.temperature))
        self.tool_temperature = min(2.0, max(0.0, self.tool_temperature))
        self.summarise_over_chars = max(500, self.summarise_over_chars)
        self.agent_max_steps = min(20, max(1, self.agent_max_steps))
        self.resilient_retries = min(6, max(0, self.resilient_retries))
        self.min_max_tokens = min(512, max(32, self.min_max_tokens))
        self.stall_timeout = min(600, max(10, self.stall_timeout))
        self.auto_fetch_char_cap = min(60000, max(1000, self.auto_fetch_char_cap))
        # The cap can never be below the ordinary step budget.
        self.hard_step_cap = min(60, max(self.agent_max_steps, self.hard_step_cap))
        self.reasoning_max_steps = min(10, max(2, self.reasoning_max_steps))
        self.reasoning_step_timeout = min(300, max(10, self.reasoning_step_timeout))
        self.retrieval_deadline = min(600.0, max(15.0, self.retrieval_deadline))
        self.reasoning_tokens = min(2048, max(64, self.reasoning_tokens))
        self.ready_wait_timeout = min(300.0, max(2.0, self.ready_wait_timeout))
        # Ratios kept in sane bands so a bad value cannot break chunking: the
        # trigger must leave room for a reply, and a chunk must be smaller than
        # the trigger or it could never fit.
        self.chunk_trigger_ratio = min(0.9, max(0.2, self.chunk_trigger_ratio))
        self.chunk_size_ratio = min(self.chunk_trigger_ratio, max(0.1, self.chunk_size_ratio))
        self.history_turns = min(200, max(0, self.history_turns))
        self.search_results = min(10, max(1, self.search_results))
        self.tool_result_chars = min(40000, max(200, self.tool_result_chars))
        self.tool_raw_chars = min(200000, max(self.tool_result_chars, self.tool_raw_chars))
        for name, old in before.items():
            if getattr(self, name) != old and name not in changed:
                changed.append(name)
        return changed


class ModelServerManager:
    """Manages the MLX model server with health probes and auto-restart."""

    def __init__(
        self,
        model_id: str,
        model_port: int,
        adapter_dir: Path,
        max_kv_size: int = 0,
        adapter_choice: str = "latest",
        kv_bits: int = 0,
        kv_group_size: int = 64,
        quantized_kv_start: int = 1024,
        prompt_cache_dir: str = "",
    ):
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.prompt_cache_dir = prompt_cache_dir
        self.model_id = model_id
        self.model_port = model_port
        self.adapter_dir = adapter_dir
        self.adapter_choice = adapter_choice
        self.max_kv_size = max_kv_size
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"
        self.lock = threading.RLock()
        self._server_help = help_cmd("mlx_lm.server")
        self._log_file: Any = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        # Set by stop(). Both readiness waits poll it, so a stop or a retrain no
        # longer blocks behind a start that is 900 seconds from timing out.
        self._cancel = threading.Event()

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
        adapter_path = self.adapter_path()
        if self.kv_bits > 0:
            # Quantized KV cache roughly halves cache memory at 8 bits, which is
            # the difference between a usable context and swapping on 8GB.
            # Server support is genuinely unsettled across builds, so this is
            # best effort and the run continues without it.
            if add_if_supported(cmd, self._server_help, ["--kv-bits"], str(self.kv_bits)):
                add_if_supported(cmd, self._server_help, ["--kv-group-size"], str(self.kv_group_size))
                add_if_supported(
                    cmd, self._server_help,
                    ["--quantized-kv-start"], str(self.quantized_kv_start),
                )
            else:
                log(
                    "Installed mlx_lm.server has no --kv-bits; the KV cache stays in fp16. "
                    "Support for this flag varies by build.",
                    logging.WARNING,
                )
        if self.prompt_cache_dir:
            # The agent re-sends an extending prefix every step, which is the
            # best case for prompt caching. Worth far more here than in chat.
            if not add_if_supported(
                cmd, self._server_help,
                ["--prompt-cache-dir", "--prompt-cache"], self.prompt_cache_dir,
            ):
                log("Installed mlx_lm.server has no prompt cache flag; ignoring PROMPT_CACHE_DIR.",
                    logging.WARNING)
        if use_adapter and adapter_path is not None:
            add_if_supported(cmd, self._server_help, ["--adapter-path", "--adapter"], str(adapter_path))
        return cmd

    def adapter_path(self) -> Path | None:
        """The adapter directory the next start will use, or None for the base model."""
        try:
            path = resolve_adapter(self.adapter_choice)
        except ValueError as exc:
            log(f"Adapter {self.adapter_choice!r} unusable ({exc}); falling back to the base model.",
                logging.WARNING)
            return None
        if path is not None and not adapter_fits(path, self.model_id):
            log(
                f"Adapter {self.adapter_choice!r} was trained on {adapter_base(path)}, "
                f"not {self.model_id}. Serving the base model instead. Retrain to get "
                "an adapter for this model.",
                logging.WARNING,
            )
            return None
        return path

    def describe(self) -> dict:
        selected = resolve_adapter_quietly(self.adapter_choice)
        active = self.adapter_path()
        return {
            "model": self.model_id,
            "adapter": self.adapter_choice,
            "adapter_path": str(active or ""),
            "adapter_active": active is not None,
            "adapter_base": adapter_base(selected) if selected is not None else None,
            "adapter_mismatch": bool(
                selected is not None and not adapter_fits(selected, self.model_id)
            ),
            "max_kv_size": self.max_kv_size,
            "kv_bits": self.kv_bits,
            "prompt_cache_dir": self.prompt_cache_dir,
            "status": self.status,
            "cached": model_is_cached(self.model_id),
        }

    def swap(self, model_id: str | None = None, adapter_choice: str | None = None) -> bool:
        """Point at a different model or adapter. Returns True if a restart is due."""
        changed = False
        with self.lock:
            if model_id and model_id != self.model_id:
                self.model_id = model_id
                changed = True
            if adapter_choice is not None and adapter_choice != self.adapter_choice:
                # Validate before adopting, so a bad id cannot leave the manager
                # pointing at something that fails on every future start.
                if adapter_choice not in ("latest", "none"):
                    resolve_adapter(adapter_choice)
                self.adapter_choice = adapter_choice
                changed = True
        return changed

    def _start_watchdog(self) -> None:
        # A restart triggered from inside the watchdog thread must not spawn a second one.
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_stop.clear()
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Auto-restart the model server if it crashes unexpectedly."""
        while not self._watchdog_stop.wait(5):
            # A timed acquire, not a blocking one: a retrain holds this lock for
            # the length of a training run, and a watchdog parked on it would
            # never see its own stop event.
            if not self.lock.acquire(timeout=1):
                continue
            try:
                if self._watchdog_stop.is_set():
                    return
                if self.status == "ready" and self.proc is not None and self.proc.poll() is not None:
                    log("Watchdog: Model server crashed, restarting...", logging.WARNING)
                    self.status = "restarting"
                    try:
                        self._start_internal()
                    except StartupCancelled:
                        log("Watchdog restart cancelled by an explicit stop.", logging.WARNING)
                        self.status = "stopped"
                    except Exception as exc:
                        log(f"Watchdog restart failed: {exc}", logging.ERROR)
                        self.status = f"error: {exc}"
            finally:
                self.lock.release()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        # stop() can be reached from inside the watchdog loop; joining self raises.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
            self._watchdog_thread = None

    def _terminate_proc(self) -> None:
        """Kill the child and release its log handle. Caller holds the lock."""
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
        self._close_log_file()

    def stop(self) -> None:
        # Signal before taking the lock. Whoever holds it is inside a readiness
        # wait and will bail out on the next poll instead of holding us for
        # minutes.
        self._cancel.set()
        self._stop_watchdog()
        with self.lock:
            self._terminate_proc()
            self.status = "stopped"
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
        self._cancel.clear()
        candidates = []
        if self.adapter_path() is not None:
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
                    wait_for_port(self.model_port, timeout=300, proc=proc, cancel=self._cancel)
                    self.status = "loading"
                    log(f"Port {self.model_port} open. Waiting for weights to load...")
                    wait_for_model_ready(
                        self.model_id, self.model_port,
                        timeout=900, proc=proc, cancel=self._cancel,
                    )
                    self.status = "ready"
                    log(f"Model loaded and responding at http://127.0.0.1:{self.model_port}")
                    self._start_watchdog()
                    return
                except StartupCancelled:
                    # Do not fall through to the no-adapter candidate: the caller
                    # asked for a stop, not for a different command line.
                    self._terminate_proc()
                    self.status = "stopped"
                    raise
                except Exception as exc:
                    last_error = exc
                    # _terminate_proc, not stop(): stop() would set the cancel
                    # flag and abort the very retry we are about to make.
                    self._terminate_proc()
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
        add_if_supported(cmd, self._lora_help, ["--num-layers"], str(self.config.train_num_layers))
        add_if_supported(cmd, self._lora_help, ["--learning-rate", "-lr"], self.config.train_lr)
        add_if_supported(cmd, self._lora_help, ["--grad-checkpoint", "--gradient-checkpoint"])
        add_if_supported(cmd, self._lora_help, ["--max-seq-length", "--seq-length", "--max-seq-len"], self.config.train_seq_len)
        return cmd

    def export_tool_traces(self) -> list[dict]:
        """Turn logged tool calls into supervised examples of the call format.

        Small models fail at format adherence far more than at reasoning, and
        format adherence is what the loop depends on: a run dies when the model
        emits {"tool": "search"} instead of this app's schema, not when it
        misremembers a date. These rows are the app's own schema, in the app's
        own wording, which is exactly the supervision that is missing.
        """
        if not self.config.train_on_tool_calls:
            return []
        rows = self.db.execute(
            "SELECT t.name, t.args, t.error, m.content AS question "
            "FROM tool_calls t LEFT JOIN messages m "
            "  ON m.conversation_id = t.conversation_id AND m.role = 'user' "
            "  AND m.id = (SELECT MAX(id) FROM messages m2 "
            "              WHERE m2.conversation_id = t.conversation_id "
            "                AND m2.role = 'user' AND m2.created_at <= t.created_at) "
            "WHERE t.error IS NULL AND t.conversation_id IS NOT NULL "
            "ORDER BY t.id DESC LIMIT ?",
            (self.config.train_tool_examples,),
        ).fetchall()

        examples: list[dict] = []
        seen: set[tuple] = set()
        for row in rows:
            question = (row["question"] or "").strip()
            if not question:
                continue
            try:
                args = json.loads(row["args"] or "{}")
            except Exception:
                continue
            call = json.dumps({"tool": row["name"], "args": args},
                              ensure_ascii=False, sort_keys=True)
            key = (question, call)
            if key in seen:
                continue
            seen.add(key)
            examples.append({
                "messages": [
                    {"role": "system", "content": TOOL_TRAINING_PREAMBLE},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": call},
                ]
            })
        return examples

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

        tool_examples = self.export_tool_traces()
        if tool_examples:
            log(f"Adding {len(tool_examples)} tool-call examples to the training set.")
            examples.extend(tool_examples)

        if not examples:
            return 0, []

        # A private Random keeps the split reproducible without reseeding the
        # process-wide generator, which every other caller of random shares.
        rng = random.Random(42)
        rng.shuffle(examples)

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

            write_adapter_base(ADAPTER_DIR, self.config.model)
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
# the process. safe_eval runs in the web process, on the request thread, with
# no subprocess timeout around it the way run_python and run_shell have, so an
# unbounded intermediate takes the whole app down rather than one tool call.
# 65536 bits is a 19,728-digit number, past any real calculator use.
# Calculator safety bounds. These cap the only two whitelisted operations whose
# cost is not bounded by expression length (exponentiation and factorial), so a
# seven-character expression cannot allocate until the process dies. Configurable
# for anyone who needs bigger numbers on a bigger machine.
MAX_RESULT_BITS = int(os.environ.get("CALC_MAX_RESULT_BITS", str(1 << 16)))
MAX_FACTORIAL_INPUT = int(os.environ.get("CALC_MAX_FACTORIAL", "1000"))


def _guarded_pow(base: Any, exponent: Any) -> Any:
    """base ** exponent, refusing results too large to hold in memory."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        # bit_length() * exponent is the exact width of the result, computed
        # without building it. 0 and 1 have no growth, so exempt them.
        if base not in (0, 1, -1) and base.bit_length() * exponent > MAX_RESULT_BITS:
            raise ValueError(
                f"refusing to compute a number with about "
                f"{base.bit_length() * exponent // 3.32:.0f} digits"
            )
    try:
        return operator.pow(base, exponent)
    except OverflowError as exc:
        raise ValueError(f"result out of range: {exc}") from exc


def _guarded_factorial(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("factorial needs a whole number")
    if not 0 <= value <= MAX_FACTORIAL_INPUT:
        raise ValueError(f"factorial argument must be between 0 and {MAX_FACTORIAL_INPUT}")
    return math.factorial(value)


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
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
# math.factorial is the other unbounded-cost entry: factorial(9**7) never returns.
_SAFE_NAMES["factorial"] = _guarded_factorial


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
            result = _SAFE_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
            # Repeated multiplication reaches the same place as ** by a longer
            # road, so bound every intermediate rather than only the pow.
            if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
                raise ValueError("intermediate result too large")
            return result
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
            description="List files in the workspace directory.",
            parameters={
                "path": "subdirectory, default the workspace root",
                "recursive": "true to walk subdirectories, default false",
            },
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
            name="edit_file",
            description=(
                "Replace an exact snippet inside a workspace file. Prefer this over "
                "write_file when changing part of a file you have already read."
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
            description="Search workspace files for a regular expression and return matching lines.",
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
                description="Run a shell command in the workspace directory and return its output.",
                parameters={"command": "shell command"},
                required=["command"],
                handler=self._run_shell,
            ))

        allow = {name.strip() for name in (self.config.agent_tools or "").split(",") if name.strip()}
        if allow:
            unknown = allow - set(self._tools)
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
            pass
        # Read a bounded number of bytes rather than resp.text. An unbounded
        # read of a large file is how a tool call turns into an out-of-memory
        # kill on a machine with 8GB shared between the app and the model.
        byte_cap = min(MAX_FETCH_BYTES, max(4096, limit * 8))
        with httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                if not self.config.allow_local_fetch:
                    guard_public_url(str(resp.url))
                content_type = resp.headers.get("content-type", "").lower()
                if content_type and not any(
                    kind in content_type
                    for kind in ("text/", "json", "xml", "html", "javascript", "csv")
                ):
                    return f"{url}\n\n[skipped: unsupported content type {content_type}]"
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= byte_cap:
                        break
                encoding = resp.encoding or "utf-8"
        body = b"".join(chunks).decode(encoding, "replace")
        text = strip_html(body) if "html" in content_type or "<" in body[:200] else body
        title = ""
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
        if match:
            title = strip_html(match.group(1))
        header = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
        truncated = "\n\n[truncated]" if total >= byte_cap else ""
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

    def _list_files(self, path: str = "", recursive: Any = False) -> str:
        target = resolve_in_workspace(path)
        if not target.exists():
            return "Directory does not exist."
        if target.is_file():
            return f"{target.name} ({target.stat().st_size} bytes)"
        if str(recursive).lower() in ("1", "true", "yes"):
            root = WORKSPACE_DIR.resolve()
            entries = sorted(
                (item for item in target.rglob("*")),
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
        target = resolve_in_workspace(path)
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")[:self.config.tool_raw_chars]

    def _write_file(self, path: str, content: str) -> str:
        target = resolve_in_workspace(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Wrote {len(str(content))} characters to {target.relative_to(WORKSPACE_DIR.resolve())}"

    def _edit_file(self, path: str, find: str, replace: str, count: Any = None) -> str:
        target = resolve_in_workspace(path)
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
        replaced = min(occurrences, max(1, limit))
        return (f"Replaced {replaced} of {occurrences} occurrence(s) in "
                f"{target.relative_to(WORKSPACE_DIR.resolve())}")

    def _search_files(self, pattern: str, path: str = "", max_results: Any = None) -> str:
        root = resolve_in_workspace(path)
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

        workspace = WORKSPACE_DIR.resolve()
        targets = [root] if root.is_file() else sorted(root.rglob("*"))
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
                    hits.append(f"{item.relative_to(workspace)}:{lineno}: {line.strip()[:200]}")
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
        rows = self.db.list_feedback(limit=count, search=query)
        if not rows:
            return "No matching feedback."
        lines = []
        for row in rows:
            answer = row.get("corrected_response") or row.get("assistant_response") or ""
            lines.append(f"Q: {row.get('user_prompt', '')}\nA: {answer}")
        return "\n\n".join(lines)

    def _remember(self, key: str, value: str) -> str:
        if self.db is None:
            return "Memory store unavailable."
        key = str(key).strip()[:120]
        if not key:
            raise ValueError("key must not be empty")
        self.db.remember(key, str(value), self.conversation_id)
        return f"Stored under {key}."

    def _recall_memory(self, query: str = "", limit: Any = 10) -> str:
        if self.db is None:
            return "Memory store unavailable."
        try:
            count = min(50, max(1, int(limit)))
        except (TypeError, ValueError):
            count = 10
        rows = self.db.recall(query or None, count)
        if not rows:
            return "No stored notes." if not query else f"No stored notes matching {query!r}."
        return "\n".join(f"{row['key']}: {row['value']}" for row in rows)

    def _forget(self, key: str) -> str:
        if self.db is None:
            return "Memory store unavailable."
        return f"Deleted {key}." if self.db.forget(str(key)) else f"No note called {key}."

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
        return (out.strip() or f"(no output, exit code {proc.returncode})")[:self.config.tool_raw_chars]

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
        return (out.strip() or f"(no output, exit code {proc.returncode})")[:self.config.tool_raw_chars]

    def call(self, name: str, args: dict, conversation_id: str | None = None) -> tuple[str, str | None]:
        """Run a tool. Returns (result text, error message or None)."""
        tool = self.get(name)
        start = time.time()
        self.conversation_id = conversation_id
        if tool is None:
            error = f"Unknown tool: {name}. Available: {', '.join(self.names())}"
            if self.db:
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error)
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
                self.db.log_tool_call(conversation_id, name, args, "", 0.0, error)
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
            self.db.log_tool_call(conversation_id, name, clean, result, duration, error)
        return result, error


def timezone_utc():
    return timezone.utc


TOOL_PROTOCOL = textwrap.dedent("""\
    You can call tools. To call one, reply with a single JSON object and nothing
    else:
    {"tool": "tool_name", "args": {"arg": "value"}}

    Rules:
    - One tool per reply. Use the exact argument names listed below.
    - You will then receive a message beginning with TOOL RESULT. Read it before
      deciding what to do next.
    - When you have enough information, either reply in plain text with no JSON,
      or call final_answer with your complete answer.
    - Never invent tool output, and never say you searched, read or ran anything
      unless a TOOL RESULT above shows it.
    - If a tool returns an error, fix the arguments and try once more, or answer
      without it. Do not repeat an identical call.
    - Prefer web_search then fetch_url for anything current. Prefer the
      calculator over doing arithmetic yourself.

    Available tools:
    """)


# Every agent request begins with this exact text, so it is the prefix any
# server-side prompt cache keys on. When it changes, every cached prefix on the
# machine becomes worthless. That is invisible otherwise: you edit the system
# prompt in the UI, latency doubles, and nothing says why.
PREFIX_STATE: dict[str, Any] = {"hash": None, "changed_at": None, "generation": 0}


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


def build_agent_system_prompt(base_prompt: str, registry: ToolRegistry) -> str:
    lines = []
    for tool in registry.specs():
        params = ", ".join(
            f"{name} ({desc})" for name, desc in tool["parameters"].items()
        ) or "no arguments"
        required = ", ".join(tool["required"]) or "none"
        lines.append(f"- {tool['name']}: {tool['description']}\n  args: {params}\n  required: {required}")
    prompt = f"{base_prompt}\n\n{TOOL_PROTOCOL}" + "\n".join(lines)
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
        """Poll the model server until it answers, or until timeout.

        After an out-of-memory kill the watchdog restarts the server, but
        reloading a model takes far longer than a fixed sleep. A retry that fires
        into a still-loading server fails instantly and wastes the attempt. This
        polls the lightweight /v1/models endpoint so a retry waits for a live
        server to shrink its prompt into, rather than sleeping a guessed few
        seconds. Returns True once the server responds.
        """
        import httpx
        deadline = time.time() + timeout
        delay = 0.5
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(self.models_url)
                if resp.status_code < 500:
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
        # parameters, and any server that does not know them ignores them.
        if self.config.repetition_penalty and self.config.repetition_penalty != 1.0:
            body["repetition_penalty"] = self.config.repetition_penalty
            body["repetition_context_size"] = self.config.repetition_context_size
        if stream:
            # Ask for usage on the final chunk. Servers that do not know the
            # option ignore it, and the estimate below covers them.
            body["stream_options"] = {"include_usage": True}
        if self.config.disable_thinking:
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

    def build_base(
        self,
        history: list[dict],
        user_message: str,
        reserve: int,
    ) -> tuple[list[dict], int]:
        """Assemble [system, trimmed history, user]. This prefix is never cut later."""
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

        def detail(message: str) -> dict | None:
            """A verbose under-the-hood line, only emitted when show_internals is on."""
            return {"type": "detail", "message": message} if self.config.show_internals else None

        def done(answer: str, step: int, truncated: bool = False) -> dict:
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
                {"role": "system", "content": self.config.system_prompt},
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
                if (CODE_NEEDS_LOOKUP.search(user_message)
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
                        {"role": "system", "content": self.config.system_prompt},
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
                # Every retry failed. Hand back whatever was gathered on disk so a
                # long, hard request still returns something useful, rather than a
                # bare apology. If nothing was gathered, fall back to the note.
                yield done(
                    self.salvage(
                        conversation_id,
                        "I ran low on memory before I could finish this in one pass. "
                        "Try a smaller or more specific request, or raise the RAM headroom."),
                    step, truncated=True,
                )
                return

            prompt_tokens_total += stats.prompt_tokens
            completion_tokens_total += stats.completion_tokens
            yield {"type": "usage", "step": step, **stats.as_event()}

            if cancelled:
                yield {"type": "cancelled", "step": step, "partial": buffer.strip(), "trace": trace}
                return

            call = parse_tool_call(buffer, known)

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
            {"role": "system", "content": self.config.system_prompt},
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


class TaskRun:
    """Live state for one run: the event buffer plus everyone tailing it.

    Events are buffered in memory so a browser that connects halfway through
    still sees the whole run, and the interesting ones are also persisted so a
    run survives a page reload or a restart. Tokens are deliberately not
    persisted: one run would otherwise write thousands of rows.
    """

    BUFFER_LIMIT = 4000

    def __init__(self, run_id: str, task_id: str, task_name: str):
        self.run_id = run_id
        self.task_id = task_id
        self.task_name = task_name
        self.seq = 0
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.cancel = asyncio.Event()
        self.done = False
        self.status = "running"
        self.answer = ""
        self.started = time.time()

    def publish(self, event: dict) -> dict:
        self.seq += 1
        stamped = dict(event)
        stamped["seq"] = self.seq
        stamped["run_id"] = self.run_id
        stamped["task_id"] = self.task_id
        self.events.append(stamped)
        if len(self.events) > self.BUFFER_LIMIT:
            # Drop the oldest tokens first; the structural events are the record.
            self.events = ([e for e in self.events if e["type"] != "token"][-self.BUFFER_LIMIT:]
                           or self.events[-self.BUFFER_LIMIT:])
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(stamped)
            except asyncio.QueueFull:
                # A tab that cannot keep up loses tokens, not the run.
                pass
        return stamped

    def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass


class TaskManager:
    """Runs agent tasks in the background, on demand or on an interval.

    One scheduler coroutine on the web process's event loop polls for due tasks.
    Runs are gated by a semaphore because the model server answers one request
    at a time: firing five tasks at once just queues them inside mlx_lm with no
    way to see the queue.
    """

    # Everything except tokens goes to the database.
    PERSISTED = {"start", "context", "step", "tool_call", "tool_result",
                 "final", "error", "cancelled"}

    def __init__(
        self,
        config: Config,
        db: Database,
        model_manager: ModelServerManager,
        retrain_manager: RetrainManager,
        client: ModelClient,
    ):
        self.config = config
        self.db = db
        self.model_manager = model_manager
        self.retrain_manager = retrain_manager
        self.client = client
        self.active: dict[str, TaskRun] = {}
        self.by_task: dict[str, TaskRun] = {}
        self.recent: dict[str, TaskRun] = {}
        self._scheduler: asyncio.Task | None = None
        self._semaphore: asyncio.Semaphore | None = None
        # Strong references to the in-flight run coroutines. The event loop only
        # holds a weak reference to a Task, so a run whose handle is dropped can
        # be garbage collected mid-execution and simply vanish. self.active
        # holds the TaskRun record, not the Task, so it does not protect this.
        self._runners: set[asyncio.Task] = set()
        self._stopping = False
        # Wall clock of the most recent interactive request. A scheduled run
        # landing mid-conversation puts two inference requests into a server
        # that holds one KV cache comfortably and two only by swapping, and on
        # 8GB the failure mode is not an error, it is macOS compressing memory
        # while tok/s quietly collapses.
        self.last_chat_at = 0.0
        self.chat_in_flight = 0

    def note_chat_activity(self) -> None:
        self.last_chat_at = time.time()

    def chat_is_busy(self) -> bool:
        if self.chat_in_flight > 0:
            return True
        return (time.time() - self.last_chat_at) < self.config.chat_idle_seconds

    # ------------------------------------------------------------ lifecycle --

    async def start(self) -> None:
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_tasks))
        orphans = self.db.reset_orphan_runs()
        if orphans:
            log(f"Marked {orphans} task run(s) interrupted by the previous shutdown.",
                logging.WARNING)
        self._scheduler = asyncio.create_task(self._scheduler_loop())
        log("Task scheduler started.")

    async def stop(self) -> None:
        self._stopping = True
        for run in list(self.active.values()):
            run.cancel.set()
        if self._scheduler is not None:
            self._scheduler.cancel()
            try:
                await self._scheduler
            except (asyncio.CancelledError, Exception):
                pass
        # Wait on the run tasks themselves. Polling self.active only observed
        # the bookkeeping dict, so a runner still unwinding its finally block
        # could outlive shutdown and write to a closed database.
        if self._runners:
            await asyncio.wait(set(self._runners), timeout=5)
        for runner in list(self._runners):
            runner.cancel()

    async def _scheduler_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(max(1, self.config.task_poll_seconds))
                if self.model_manager.status != "ready":
                    continue
                if self.retrain_manager.status.get("running"):
                    continue
                # Interactive use wins. A task deferred by a few seconds costs
                # nobody anything; a task that halves your chat throughput does.
                if self.chat_is_busy():
                    continue
                for task in self.db.due_tasks():
                    if task["id"] in self.by_task:
                        continue
                    # Re-arm before running: a task whose run outlives its own
                    # interval must not stack up a backlog of overdue firings.
                    self.db.schedule_next(task["id"], int(task["interval_seconds"] or 0))
                    await self.launch(task, trigger="schedule")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log(f"Task scheduler error: {exc}", logging.ERROR)
                await asyncio.sleep(5)

    # ---------------------------------------------------------------- runs --

    async def launch(self, task: dict, trigger: str = "manual") -> TaskRun:
        if task["id"] in self.by_task:
            raise ValueError("this task is already running")
        run_id = self.db.create_run(task["id"], trigger, self.config.model)
        run = TaskRun(run_id, task["id"], task["name"])
        self.active[run_id] = run
        self.by_task[task["id"]] = run
        runner = asyncio.create_task(self._execute(task, run, trigger))
        self._runners.add(runner)
        runner.add_done_callback(self._runners.discard)
        return run

    def _task_config(self, task: dict) -> Config:
        """A config copy scoped to one task, so its tools and step budget are its own."""
        return dataclass_replace(
            self.config,
            model=task.get("model") or self.config.model,
            system_prompt=task.get("system_prompt") or self.config.system_prompt,
            agent_tools=task.get("tools") or self.config.agent_tools,
            agent_max_steps=int(task.get("max_steps") or self.config.agent_max_steps),
        )

    async def _swap_for_task(self, task: dict) -> str | None:
        """Load a task's own model, returning the one to restore afterwards.

        Latency tolerance differs by workload. Interactive chat wants a small
        model that answers now; a 3am research task has nobody waiting and can
        afford a bigger one, or a reasoning model whose chain of thought would
        be intolerable in a chat box. On 8GB you cannot hold both, so swap.
        """
        wanted = (task.get("model") or "").strip()
        if not wanted or wanted == self.model_manager.model_id:
            return None
        previous = self.model_manager.model_id
        log(f"Task {task['name']}: swapping {previous} -> {wanted}")
        self.model_manager.swap(wanted)
        await asyncio.to_thread(self.model_manager.restart)
        if self.model_manager.status != "ready":
            self.model_manager.swap(previous)
            await asyncio.to_thread(self.model_manager.restart)
            raise RuntimeError(f"could not load {wanted} for this task")
        return previous

    async def _restore_model(self, previous: str | None) -> None:
        if not previous or previous == self.model_manager.model_id:
            return
        log(f"Restoring model {previous} after task run")
        self.model_manager.swap(previous)
        try:
            await asyncio.to_thread(self.model_manager.restart)
        except Exception as exc:
            log(f"Could not restore {previous}: {exc}", logging.ERROR)

    async def _execute(self, task: dict, run: TaskRun, trigger: str) -> None:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_tasks))
        conversation_id = f"task:{task['id']}"
        answer = ""
        error: str | None = None
        steps = 0
        tools_used: list[str] = []
        status = "ok"
        previous_model: str | None = None

        try:
            async with self._semaphore:
                if run.cancel.is_set():
                    raise asyncio.CancelledError
                previous_model = await self._swap_for_task(task)
                self._emit(run, {"type": "start", "task": task["name"], "trigger": trigger,
                                 "model": task.get("model") or self.config.model,
                                 "swapped": previous_model is not None})
                append_task_log(f"run {run.run_id} start: {task['name']} ({trigger})")

                task_config = self._task_config(task)
                registry = ToolRegistry(task_config, self.db)
                agent = Agent(task_config, registry, ModelClient(task_config))

                history: list[dict] = []
                if task.get("use_history"):
                    rows = self.db.get_messages(conversation_id, limit=self.config.history_turns * 2)
                    history = [{"role": r["role"], "content": r["content"]} for r in rows]

                async for event in agent.run(
                    task["goal"], history, conversation_id, cancel=run.cancel
                ):
                    self._emit(run, event)
                    if event["type"] == "final":
                        answer = event["answer"]
                        steps = event.get("steps", 0)
                        tools_used = event.get("tools_used", [])
                    elif event["type"] == "error":
                        error = event["error"]
                        status = "error"
                    elif event["type"] == "cancelled":
                        status = "cancelled"

                if task.get("use_history") and answer:
                    self.db.add_message(conversation_id, "user", task["goal"])
                    self.db.add_message(conversation_id, "assistant", answer)

        except asyncio.CancelledError:
            status = "cancelled"
            self._emit(run, {"type": "cancelled", "reason": "shutdown or cancel request"})
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            log(f"Task {task['name']} failed: {error}", logging.ERROR)
            self._emit(run, {"type": "error", "error": error})
        finally:
            try:
                await self._restore_model(previous_model)
            except Exception as exc:
                log(f"Model restore failed: {exc}", logging.ERROR)
            elapsed = (time.time() - run.started) * 1000
            run.status = status
            run.answer = answer
            self.db.finish_run(run.run_id, status, answer, error, steps, elapsed, tools_used)
            self.db.prune_runs(task["id"], keep=25)
            append_task_log(
                f"run {run.run_id} {status} in {elapsed:.0f}ms"
                + (f": {error}" if error else f": {answer[:120]}")
            )
            self._emit(run, {"type": "done", "status": status, "answer": answer,
                             "elapsed_ms": round(elapsed), "error": error})
            run.close()
            self.active.pop(run.run_id, None)
            if self.by_task.get(task["id"]) is run:
                self.by_task.pop(task["id"], None)
            self.recent[run.run_id] = run
            while len(self.recent) > 20:
                self.recent.pop(next(iter(self.recent)))

            # Chaining. Three narrow tasks passing state through the workspace
            # beat one task with a compound goal at this model size, so the
            # dependency is first class rather than something you fake with two
            # schedules and a file.
            if status == "ok" and task.get("next_task_id") and not self._stopping:
                await self._chain(task, answer)

    async def _chain(self, task: dict, answer: str) -> None:
        follow_on = self.db.get_task(task["next_task_id"])
        if follow_on is None:
            log(f"Task {task['name']} points at a missing next task.", logging.WARNING)
            return
        if follow_on["id"] in self.by_task:
            log(f"Chained task {follow_on['name']} is already running; skipping.", logging.WARNING)
            return
        if follow_on["id"] == task["id"]:
            log("Refusing to chain a task to itself.", logging.WARNING)
            return
        # Hand the upstream answer over on disk rather than in the goal text, so
        # a long result does not become a giant prompt for the next task.
        try:
            handoff = resolve_in_workspace(f"chain/{task['id']}.txt")
            handoff.parent.mkdir(parents=True, exist_ok=True)
            handoff.write_text(answer, encoding="utf-8")
            relative = handoff.relative_to(WORKSPACE_DIR.resolve())
        except Exception as exc:
            log(f"Could not write the chain handoff: {exc}", logging.WARNING)
            return
        chained = dict(follow_on)
        chained["goal"] = (
            f"{follow_on['goal']}\n\n"
            f"The previous task ({task['name']}) wrote its result to {relative}. "
            "Read that file first with read_file."
        )
        log(f"Chaining {task['name']} -> {follow_on['name']}")
        await self.launch(chained, trigger=f"chain:{task['id']}")

    def _emit(self, run: TaskRun, event: dict) -> None:
        stamped = run.publish(event)
        if event["type"] in self.PERSISTED:
            payload = {k: v for k, v in stamped.items() if k not in ("run_id", "task_id")}
            try:
                self.db.append_event(run.run_id, stamped["seq"], event["type"], payload)
            except Exception as exc:
                log(f"Could not persist task event: {exc}", logging.WARNING)

    # -------------------------------------------------------------- control --

    def cancel_task(self, task_id: str) -> bool:
        run = self.by_task.get(task_id)
        if run is None:
            return False
        run.cancel.set()
        return True

    def cancel_run(self, run_id: str) -> bool:
        run = self.active.get(run_id)
        if run is None:
            return False
        run.cancel.set()
        return True

    def live_status(self, task_id: str) -> dict | None:
        run = self.by_task.get(task_id)
        if run is None:
            return None
        last_step = next((e for e in reversed(run.events) if e["type"] == "step"), None)
        last_tool = next((e for e in reversed(run.events) if e["type"] == "tool_call"), None)
        return {
            "run_id": run.run_id,
            "step": (last_step or {}).get("step", 0),
            "max_steps": (last_step or {}).get("max_steps", 0),
            "tool": (last_tool or {}).get("name"),
            "elapsed_ms": round((time.time() - run.started) * 1000),
        }

    async def subscribe(self, run_id: str) -> AsyncGenerator[dict, None]:
        """Replay a run then follow it live. Falls back to the database when finished."""
        run = self.active.get(run_id) or self.recent.get(run_id)
        if run is None:
            for event in self.db.run_events(run_id, limit=2000):
                yield event
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        # Subscribe before snapshotting, then drop anything the snapshot already
        # covered. The other order loses events published in between.
        run.subscribers.add(queue)
        try:
            backlog = list(run.events)
            highest = backlog[-1]["seq"] if backlog else 0
            for event in backlog:
                yield event
            if run.done:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                if event["seq"] <= highest:
                    continue
                yield event
        finally:
            run.subscribers.discard(queue)


def append_task_log(line: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / LOG_FILES["tasks"], "a", encoding="utf-8") as handle:
            handle.write(f"[{iso(utc_now())}] {line}\n")
    except OSError:
        pass


HTML_PAGE = r"""
<!doctype html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>{{APP_NAME}}</title>
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
   /* Glass box: the agent's work as a live vertical trace. */
   .turn { align-self: stretch; display: flex; flex-direction: column; gap: 8px; }
   .trace {
     align-self: flex-start;
     max-width: 88%;
     margin-left: 6px;
     padding-left: 14px;
     border-left: 2px solid var(--border);
     display: flex;
     flex-direction: column;
     gap: 10px;
   }
   .trace:empty { display: none; }
   .gnode { position: relative; font-size: 13px; }
   .gnode::before {
     content: ""; position: absolute; left: -19px; top: 5px;
     width: 8px; height: 8px; border-radius: 50%;
     background: var(--surface); border: 1.5px solid var(--border);
   }
   .gnode.router::before { border-color: var(--accent); }
   .gnode.ok::before { border-color: var(--success); background: var(--success); }
   .gnode.failed::before { border-color: var(--error); background: var(--error); }
   .gnode.notice::before { border-color: var(--warn); background: var(--warn); }
   .gnode .ghead {
     display: flex; align-items: center; gap: 8px;
     color: #9aa0a6; font-family: ui-monospace, monospace; font-size: 12px;
   }
   .gnode.tool .ghead { cursor: pointer; user-select: none; }
   .gnode .gtool { color: var(--tool); font-weight: 600; }
   .gnode .gpill {
     font-size: 11px; padding: 1px 7px; border-radius: 999px;
     background: #22331f; color: var(--success);
   }
   .gnode.failed .gpill { background: #331f1f; color: var(--error); }
   .gnode .gcaret { margin-left: auto; transition: transform 0.15s; color: #666; }
   .gnode.open .gcaret { transform: rotate(90deg); }
   .gnode .gargs { color: #8a8f94; font-family: ui-monospace, monospace; font-size: 12px; margin-top: 3px; }
   .gnode .gbody {
     margin-top: 6px; white-space: pre-wrap; font-family: ui-monospace, monospace;
     font-size: 12px; color: #cfcfcf; max-height: 240px; overflow: auto;
     background: #16121a; border-radius: 8px; padding: 8px 10px;
   }
   .gnode.tool:not(.open) .gbody { display: none; }
   .gnode.notice .gmsg { color: var(--warn); }
   .gnode.notice.info::before { border-color: var(--tool); background: var(--tool); }
   .gnode.notice.info .gmsg { color: var(--tool); }
   .gnode.answer { border-left: 0; }
   .gnode.answer .gtext {
     background: var(--surface); border: 1px solid #444; border-radius: 12px;
     padding: 10px 12px; white-space: pre-wrap; line-height: 1.4; font-size: 14px; color: var(--fg);
   }
   .gnode.answer.pending .gtext { opacity: 0.75; }
   .gmeta { color: #777; font-size: 11px; font-family: ui-monospace, monospace; margin: 2px 0 0 6px; }
   .gactivity {
     display: flex; align-items: center; gap: 8px; margin: 2px 0 0 6px;
     font-family: ui-monospace, monospace; font-size: 12px; color: var(--accent);
   }
   .gspin {
     width: 11px; height: 11px; border-radius: 50%;
     border: 2px solid #2a3b4d; border-top-color: var(--accent);
     animation: gspin 0.7s linear infinite; flex: 0 0 auto;
   }
   @keyframes gspin { to { transform: rotate(360deg); } }
   .gactivity .gelapsed { color: #888; }
   .gactivity.stalled { color: var(--warn); }
   .gactivity.stalled .gspin { border-top-color: var(--warn); }
   .gnode.tool.running .gpill { background: #1f2a33; color: var(--accent); }
   .gnode.reason::before { border-color: var(--accent); }
   .gnode.reason .gbody { color: #b9c2cc; max-height: 160px; }
   .gnode.reason.done::before { border-color: var(--success); background: var(--success); }
   .gnode.thinking::before { border-color: #8a8f94; }
   .gnode.thinking .gtool { color: #9aa0a6; }
   .gnode.thinking .gbody { color: #9aa0a6; font-style: italic; max-height: 200px; }
   .gnode.thinking:not(.open) .gbody { display: none; }
   .gdetail {
     margin: 1px 0 1px 22px; font-family: ui-monospace, monospace;
     font-size: 11px; color: #6b7075; white-space: pre-wrap;
   }
   .gdetail::before { content: "\2699 "; opacity: 0.6; }
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
   /* Branding in the header. */
   .brand { display: flex; align-items: center; gap: 8px; }
   .brand-slot { display: inline-flex; align-items: center; }
   .brand-logo { height: 22px; width: auto; border-radius: 5px; display: block; }
   .brand-mark { font-size: 18px; line-height: 1; color: var(--accent); }
   .brand .build { color: #666; font-size: 11px; }
   /* Hover tooltips. Any element with data-tip shows a styled bubble on hover
      and on keyboard focus, so every control can explain itself. */
   [data-tip] { position: relative; }
   [data-tip]:hover::after, [data-tip]:focus-visible::after {
     content: attr(data-tip);
     position: absolute; left: 50%; bottom: calc(100% + 8px);
     transform: translateX(-50%);
     background: #000; color: #eee; border: 1px solid #444;
     padding: 6px 9px; border-radius: 7px; font-size: 12px; font-weight: 400;
     line-height: 1.35; white-space: normal; width: max-content; max-width: 240px;
     text-align: left; z-index: 50; pointer-events: none;
     box-shadow: 0 6px 20px rgba(0,0,0,0.45);
   }
   [data-tip]:hover::before, [data-tip]:focus-visible::before {
     content: ""; position: absolute; left: 50%; bottom: calc(100% + 3px);
     transform: translateX(-50%);
     border: 5px solid transparent; border-top-color: #444; z-index: 50;
     pointer-events: none;
   }
   /* Tooltips that would clip at the top of the screen flip below the element. */
   [data-tip-below]:hover::after, [data-tip-below]:focus-visible::after {
     bottom: auto; top: calc(100% + 8px);
   }
   [data-tip-below]:hover::before, [data-tip-below]:focus-visible::before {
     bottom: auto; top: calc(100% + 3px); border-top-color: transparent; border-bottom-color: #444;
   }
   header { overflow: visible; }
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

   nav.views { display: flex; gap: 4px; }
   nav.views button { padding: 6px 12px; }
   nav.views button.active { background: var(--accent); border-color: var(--accent); color: white; }
   .view { flex: 1; display: flex; overflow: hidden; }
   .view.hidden { display: none; }

   #tasksView { flex-direction: row; }
   .task-list { width: 340px; min-width: 300px; border-right: 1px solid var(--border); overflow-y: auto; padding: 14px; }
   .task-monitor { flex: 1; overflow-y: auto; padding: 14px 16px; }
   .card {
     border: 1px solid var(--border);
     border-radius: 10px;
     padding: 10px 12px;
     margin-bottom: 10px;
     background: var(--surface);
     cursor: pointer;
   }
   .card.selected { border-color: var(--accent); }
   .card h4 { margin: 0 0 4px; font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }
   .card .goal { font-size: 12px; color: #aaa; line-height: 1.4; max-height: 48px; overflow: hidden; }
   .card .meta { font-size: 11px; color: #777; margin-top: 6px; font-family: ui-monospace, monospace; }
   .card .card-actions { display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }
   .card .card-actions button { padding: 3px 8px; font-size: 12px; }
   .pill {
     font-size: 10px;
     padding: 1px 7px;
     border-radius: 999px;
     border: 1px solid #555;
     color: #bbb;
     white-space: nowrap;
     font-family: ui-monospace, monospace;
   }
   .pill.ok { border-color: var(--success); color: var(--success); }
   .pill.error, .pill.interrupted { border-color: var(--error); color: var(--error); }
   .pill.running { border-color: var(--accent); color: var(--accent); }
   .pill.cancelled { border-color: var(--warn); color: var(--warn); }
   .form-grid label { display: block; font-size: 12px; color: #999; margin: 10px 0 4px; }
   .form-grid input, .form-grid textarea, .form-grid select { width: 100%; }
   .form-grid textarea { min-height: 64px; resize: vertical; }
   #runFeed { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
   .feed-line { font-size: 12px; color: #999; font-family: ui-monospace, monospace; }
   .feed-answer {
     border: 1px solid var(--success);
     border-radius: 10px;
     padding: 10px 12px;
     white-space: pre-wrap;
     line-height: 1.4;
   }
   .feed-partial { color: #ccc; white-space: pre-wrap; font-size: 13px; line-height: 1.4; }
   .logbox {
     background: #0c0c0c;
     border: 1px solid var(--border);
     border-radius: 8px;
     padding: 10px;
     font-family: ui-monospace, monospace;
     font-size: 11px;
     white-space: pre-wrap;
     max-height: 260px;
     overflow: auto;
     color: #bbb;
   }
   table.models { width: 100%; border-collapse: collapse; font-size: 13px; }
   table.models td { padding: 7px 6px; border-bottom: 1px solid #262626; }
   table.models tr:last-child td { border-bottom: none; }
   table.models td.id { font-family: ui-monospace, monospace; font-size: 12px; word-break: break-all; }
   .panel { max-width: 760px; padding: 16px; overflow-y: auto; flex: 1; }
   .panel h3 { margin: 18px 0 8px; font-size: 14px; }
   .panel h3:first-child { margin-top: 0; }
 </style>
</head>
<body>
 <header>
   <div class="brand"><span class="brand-slot">{{APP_LOGO}}</span><strong>{{APP_NAME}}</strong> <span class="build" title="UI build version">build {{UI_BUILD}}</span></div>
   <div id="status">Starting...</div>
   <nav class="views">
     <button id="navChat" class="active" onclick="showView('chat')" data-tip-below data-tip="Talk to the model. Ask questions, paste code or documents, run tools." title="Chat view">Chat</button>
     <button id="navTasks" onclick="showView('tasks')" data-tip-below data-tip="Scheduled or saved jobs the agent runs on demand or on a timer." title="Tasks view">Tasks</button>
     <button id="navModels" onclick="showView('models')" data-tip-below data-tip="Pick or download a model, and attach a fine-tuned adapter." title="Models view">Models</button>
   </nav>
   <div class="actions">
     <button onclick="newChat()" data-tip-below data-tip="Start a fresh conversation. Clears the current thread from view." title="New chat">New chat</button>
     <button onclick="retrain()" data-tip-below data-tip="Fine-tune the model on your thumbs-up/down feedback so far (LoRA)." title="Retrain on feedback">Retrain</button>
     <button onclick="toggleSettings()" data-tip-below data-tip="Model, tools, generation and memory settings you can change live." title="Open settings">Settings</button>
   </div>
 </header>

 <div id="main">
   <div id="chatView" class="view"><div id="chat"></div></div>

   <div id="tasksView" class="view hidden">
     <div class="task-list">
       <div style="display:flex;justify-content:space-between;align-items:center">
         <strong style="font-size:13px">Tasks</strong>
         <button onclick="toggleTaskForm()">New task</button>
       </div>

       <div id="taskForm" class="form-grid" style="display:none;margin-top:10px">
         <label>Name</label>
         <input id="tfName" type="text" placeholder="Morning news sweep">
         <label>Goal, written as an instruction to the agent</label>
         <textarea id="tfGoal" placeholder="Search for news about MLX released in the last day and summarise anything new in three bullets."></textarea>
         <div class="row">
           <div>
             <label>Repeat every (seconds)</label>
             <input id="tfInterval" type="number" min="0" step="60" value="0">
           </div>
           <div>
             <label data-tip="Tool/reasoning steps per turn before the agent must answer. Big tasks continue past this from a summary.">Max steps</label>
             <input id="tfSteps" type="number" min="1" max="20" value="6">
           </div>
         </div>
         <label>Tools (comma separated, blank for all)</label>
         <input id="tfTools" type="text" placeholder="web_search, fetch_url, remember">
         <label>System prompt override (optional)</label>
         <textarea id="tfSystem" placeholder="Leave blank to use the global system prompt."></textarea>
         <div class="row" style="margin-top:10px">
           <div>
             <label>Keep conversation history</label>
             <select id="tfHistory">
               <option value="false">no, each run is fresh</option>
               <option value="true">yes, runs build on each other</option>
             </select>
           </div>
         </div>
         <div class="row" style="margin-top:12px">
           <button class="primary" onclick="createTask()">Create</button>
           <button onclick="toggleTaskForm()">Cancel</button>
         </div>
         <div class="hint">
           Interval 0 means the task only runs when you press Run. A repeating
           task fires once as soon as it is created, so you find out quickly
           whether the goal is worded well.
         </div>
       </div>

       <div id="taskCards" style="margin-top:12px">loading...</div>
     </div>

     <div class="task-monitor">
       <div id="monitorHeader" style="color:#888;font-size:13px">
         Select a task to watch its runs, or create one.
       </div>
       <div id="runControls" style="display:none;margin-top:10px">
         <div class="row" style="max-width:520px">
           <div>
             <label style="font-size:12px;color:#999">Run</label>
             <select id="runPicker" onchange="openRun(this.value)"></select>
           </div>
           <div style="flex:0 0 auto;display:flex;align-items:flex-end;gap:6px">
             <button onclick="runSelectedTask()">Run now</button>
             <button onclick="cancelSelectedTask()">Cancel</button>
           </div>
         </div>
       </div>
       <div id="runFeed"></div>
     </div>
   </div>

   <div id="modelsView" class="view hidden">
     <div class="panel">
       <h3>Current</h3>
       <div id="modelCurrent" class="logbox" style="max-height:none">loading...</div>

       <h3>Switch model</h3>
       <table class="models"><tbody id="modelTable"></tbody></table>
       <label style="display:block;font-size:12px;color:#999;margin:12px 0 4px">
         Or any Hugging Face repo id
       </label>
       <div class="row">
         <div><input id="modelCustom" type="text" placeholder="mlx-community/Qwen2.5-7B-Instruct-4bit"></div>
         <div style="flex:0 0 auto"><button onclick="useModel(document.getElementById('modelCustom').value.trim())" data-tip="Queue this model id (Hugging Face / mlx-community) to load." title="Use this model">Use</button></div>
       </div>

       <h3>Adapter and cache</h3>
       <div class="row">
         <div>
           <label style="font-size:12px;color:#999">LoRA adapter</label>
           <select id="adapterSelect"></select>
         </div>
         <div>
           <label style="font-size:12px;color:#999">KV cache cap (0 = unbounded)</label>
           <input id="kvSize" type="number" min="0" step="512">
         </div>
       </div>
       <div class="row" style="margin-top:12px">
         <button class="primary" onclick="applyModel()" data-tip="Load the selected model and adapter. Restarts the model server." title="Apply and restart">Apply and restart</button>
         <button onclick="loadModels()" data-tip="Reload the list of available and cached models." title="Refresh model list">Refresh</button>
       </div>
       <div class="hint">
         Switching to a model that is not cached downloads it on first use, which
         can take minutes and several gigabytes. The log below is the model
         server's own output, including download progress.
       </div>

       <h3>Model server log</h3>
       <div id="modelLog" class="logbox">loading...</div>
     </div>
   </div>

   <aside id="settings">
     <h3>Generation</h3>
     <label>System prompt</label>
     <textarea id="cfgSystem"></textarea>
     <div class="row">
       <div>
         <label data-tip="Longest reply the model may generate, in tokens. Higher = longer answers but more memory and time.">Max tokens</label>
         <input id="cfgMaxTokens" type="number" min="16" max="32768" step="16">
       </div>
       <div>
         <label data-tip="Randomness of replies. 0 = deterministic and focused; higher = more varied and creative.">Temperature</label>
         <input id="cfgTemperature" type="number" min="0" max="2" step="0.05">
       </div>
     </div>
     <div class="row">
       <div>
         <label data-tip="Total working window (prompt + reply). Auto-sized to your RAM; larger holds more history but uses more memory. Above ~60% of this, large prompts are chunked.">Context size (tokens)</label>
         <input id="cfgContext" type="number" min="512" max="131072" step="512">
       </div>
       <div>
         <label data-tip="How many past messages to carry into each request. Fewer = less memory, less continuity.">History turns</label>
         <input id="cfgHistory" type="number" min="0" max="200" step="1">
       </div>
     </div>
     <div class="meter"><div id="ctxBar"></div></div>
     <div class="hint" id="ctxHint">Context usage in this conversation.</div>

     <h3 style="margin-top:18px">Agent</h3>
     <div class="row">
       <div>
         <label data-tip="Whether new chats start in agent mode (tools + step reasoning) or as plain single replies.">Enabled by default</label>
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
         <label data-tip="Which web search provider the search tool uses. ddg needs no key; brave/tavily/searxng may need one.">Search backend</label>
         <select id="cfgSearchBackend">
           <option value="ddg">ddg</option>
           <option value="brave">brave</option>
           <option value="tavily">tavily</option>
           <option value="searxng">searxng</option>
         </select>
       </div>
       <div>
         <label data-tip="How many results the search tool returns per query.">Search results</label>
         <input id="cfgSearchResults" type="number" min="1" max="10" step="1">
       </div>
     </div>
     <label data-tip="Max characters a tool result may contribute before it is summarised or truncated.">Tool result limit (chars)</label>
     <input id="cfgToolChars" type="number" min="200" max="40000" step="200">

     <div style="margin-top:14px" class="row">
       <button class="primary" onclick="saveConfig()" data-tip="Save these settings. Most apply immediately, no restart needed." title="Apply settings">Apply</button>
       <button onclick="restartModel()" data-tip="Restart the local model server. Use if it becomes unresponsive." title="Restart model server">Restart model</button>
     </div>
     <div class="hint">
       Context size takes effect on the next message. The KV cache size passed to
       the model server only changes on restart.
     </div>

     <h3 style="margin-top:18px">Performance</h3>
     <div class="row">
       <div>
         <label data-tip="How much of a tool result is fed back to the model as context.">Tool result into context (chars)</label>
         <input id="cfgToolChars2" type="number" min="200" max="40000" step="100">
       </div>
       <div>
         <label data-tip="Temperature used only when the model is choosing a tool. 0 keeps tool-calls deterministic.">Tool step temperature</label>
         <input id="cfgToolTemp" type="number" min="0" max="2" step="0.05">
       </div>
     </div>
     <label class="agent-toggle" style="margin-top:8px" data-tip="Skip the model's hidden &lt;think&gt; phase. Faster replies; may reduce reasoning quality on hard questions.">
       <input id="cfgThinking" type="checkbox"> disable thinking mode
     </label>
     <label class="agent-toggle" data-tip="Route obvious requests (a URL, arithmetic) directly to a tool without a model call. Faster and cheaper.">
       <input id="cfgFastPath" type="checkbox"> deterministic fast path
     </label>
     <label class="agent-toggle" data-tip="Keep the prompt prefix stable between steps so the model server can reuse its cache. Faster, uses a bit more memory.">
       <input id="cfgStablePrefix" type="checkbox"> stable prefix (cache friendly)
     </label>
     <label class="agent-toggle" data-tip="Condense oversized tool results before feeding them back, to save context on smaller machines.">
       <input id="cfgSummarise" type="checkbox"> summarise long tool results
     </label>
     <div class="row" style="margin-top:10px">
       <button onclick="saveConfig()">Apply</button>
       <button onclick="loadPerf()">Refresh stats</button>
     </div>
     <div class="tools-list" id="perfStats" style="margin-top:8px">no samples yet</div>
     <div id="prefillCurve" style="margin-top:10px"></div>
     <div class="hint">
       Prompt tokens climbing step over step within one run is re-prefill cost.
       Flat means the prefix is being reused.
     </div>
     <div class="hint" id="prefixWarn" style="display:none"></div>

     <h3 style="margin-top:18px">Memory</h3>
     <div class="row">
       <div><input id="memKey" type="text" placeholder="key"></div>
       <div><input id="memValue" type="text" placeholder="value"></div>
     </div>
     <div class="row" style="margin-top:8px">
       <button onclick="saveMemory()">Store</button>
       <button onclick="loadMemory()">Refresh</button>
     </div>
     <div class="tools-list" id="memoryList" style="margin-top:8px">loading...</div>
     <div class="hint">
       Notes the agent stores with the remember tool, and anything you add here.
       They persist across restarts.
     </div>

     <h3 style="margin-top:18px">Tools</h3>
     <div class="tools-list" id="toolsList">loading...</div>
   </aside>
 </div>

 <footer id="composer">
   <textarea id="input" placeholder="Send a message. Shift+Enter for a new line." rows="1" data-tip="Type here. Enter sends, Shift+Enter adds a line. Paste large files freely; they are processed in chunks." title="Message input"></textarea>
   <label class="agent-toggle" data-tip="Agent mode lets the model call tools (search, fetch, weather, calculator) and reason in steps. Off = a plain single reply." title="Toggle agent mode"><input id="agentToggle" type="checkbox"> agent</label>
   <button id="stopBtn" onclick="stopStream()" disabled data-tip="Stop the current response. Keeps whatever streamed so far." title="Stop generating">Stop</button>
   <button id="sendBtn" class="primary" onclick="send()" data-tip="Send your message (or press Enter)." title="Send message">Send</button>
 </footer>

 <script>
   var chat = document.getElementById("chat");
   var input = document.getElementById("input");
   var sendBtn = document.getElementById("sendBtn");
   var stopBtn = document.getElementById("stopBtn");
   var statusEl = document.getElementById("status");
   var agentToggle = document.getElementById("agentToggle");
   var settingsEl = document.getElementById("settings");

   var conversationId = localStorage.getItem("llm_conversation") || randomId();
   localStorage.setItem("llm_conversation", conversationId);
   var contextSize = 4096;
   var usedTokens = 0;
   var busy = false;
   var controller = null;

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

   input.addEventListener("input", function() {
     input.style.height = "auto";
     input.style.height = Math.min(200, input.scrollHeight) + "px";
   });

   function stopStream() {
     if (controller) controller.abort();
   }

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

   // --- Glass box: build one trace timeline per assistant turn. ---
   function startTrace() {
     var turn = document.createElement("div");
     turn.className = "turn";
     var trace = document.createElement("div");
     trace.className = "trace";
     turn.appendChild(trace);
     // Always-visible activity line: a spinner, the current state, and a clock
     // that keeps ticking so a slow step never looks frozen.
     var activity = document.createElement("div");
     activity.className = "gactivity";
     var spin = document.createElement("span"); spin.className = "gspin";
     var label = document.createElement("span"); label.className = "glabel"; label.textContent = "starting\u2026";
     var elapsed = document.createElement("span"); elapsed.className = "gelapsed";
     activity.appendChild(spin); activity.appendChild(label); activity.appendChild(elapsed);
     turn.appendChild(activity);
     chat.appendChild(turn);
     scrollDown();
     var t = { turn: turn, trace: trace, answer: null, tool: null,
               activity: activity, label: label, elapsed: elapsed,
               started: Date.now(), stepLabel: "", timer: null };
     // Tick the clock four times a second. This is the liveness proof: as long
     // as this number moves, the turn is not dead.
     t.timer = setInterval(function() {
       var secs = (Date.now() - t.started) / 1000;
       t.elapsed.textContent = secs.toFixed(1) + "s";
       // If a single step runs long, flag it visually so a real stall is obvious.
       activity.classList.toggle("stalled", secs > 25 && !t.answered);
     }, 250);
     return t;
   }

   function setActivity(t, text) {
     if (!t.activity) return;
     t.label.textContent = t.stepLabel ? (t.stepLabel + " \u00b7 " + text) : text;
   }

   function stopActivity(t) {
     if (t.timer) { clearInterval(t.timer); t.timer = null; }
     t.answered = true;
     if (t.activity && t.activity.parentNode) t.activity.parentNode.removeChild(t.activity);
   }

   function bumpActivity(t) {
     // Keep the activity line as the last child of the turn as nodes are added.
     if (t.activity) t.turn.appendChild(t.activity);
   }

   function traceRouter(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode router";
     var head = document.createElement("div");
     head.className = "ghead";
     var loc = args && (args.location || args.query);
     head.textContent = "router \u2192 " + name + (loc ? " (" + String(loc).slice(0, 60) + ")" : "");
     node.appendChild(head);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceTool(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode tool";
     var head = document.createElement("div");
     head.className = "ghead";
     var label = document.createElement("span");
     label.className = "gtool";
     label.textContent = name;
     var pill = document.createElement("span");
     pill.className = "gpill";
     pill.textContent = "running";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "\u25b8";
     head.appendChild(label);
     head.appendChild(pill);
     head.appendChild(caret);
     var argsEl = document.createElement("div");
     argsEl.className = "gargs";
     argsEl.textContent = JSON.stringify(args || {});
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(argsEl);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, pill: pill, body: body };
   }

   function traceThinking(t) {
     if (t.thinking) return t.thinking;
     var node = document.createElement("div");
     node.className = "gnode thinking open";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "thinking";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "\u25b8";
     head.appendChild(tag);
     head.appendChild(caret);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     t.thinking = { node: node, body: body };
     return t.thinking;
   }

   function traceReasonStep(t, step, total, label) {
     var node = document.createElement("div");
     node.className = "gnode reason";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "reasoning " + step + "/" + total;
     var lab = document.createElement("span");
     lab.style.color = "#9aa0a6";
     lab.textContent = label || "";
     head.appendChild(tag); head.appendChild(lab);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head); node.appendChild(body);
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, body: body };
   }

   function traceDetail(t, message) {
     var el = document.createElement("div");
     el.className = "gdetail";
     el.textContent = message;
     t.turn.appendChild(el);
     scrollDown();
   }

   function traceNotice(t, message, info) {
     var node = document.createElement("div");
     node.className = "gnode notice" + (info ? " info" : "");
     var msg = document.createElement("div");
     msg.className = "gmsg";
     msg.textContent = message;
     node.appendChild(msg);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceAnswer(t) {
     if (t.answer) return t.answer;
     var node = document.createElement("div");
     node.className = "gnode answer pending";
     var text = document.createElement("div");
     text.className = "gtext";
     node.appendChild(text);
     t.turn.appendChild(node);
     t.answer = { node: node, text: text };
     scrollDown();
     return t.answer;
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
     input.style.height = "auto";
     busy = true;
     sendBtn.disabled = true;
     stopBtn.disabled = false;
     controller = new AbortController();
     addMessage("user", message);
     updateContextMeter(estimateTokens(message));

     var trace = startTrace();
     var answered = false;
     var firstTool = true;

     try {
       var res = await fetch("/api/chat/stream", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         signal: controller.signal,
         body: JSON.stringify({
           message: message,
           conversation_id: conversationId,
           agent: agentToggle.checked
         })
       });

       if (!res.ok || !res.body) {
         var text = await res.text();
         var a0 = traceAnswer(trace);
         a0.text.textContent = "Error: " + text.slice(0, 400);
         a0.node.classList.remove("pending");
         a0.node.classList.add("failed");
         stopActivity(trace);
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
         } else if (event.type === "detail") {
           traceDetail(trace, event.message || "");
           bumpActivity(trace);
         } else if (event.type === "phase") {
           setActivity(trace, event.label || "working\u2026");
           bumpActivity(trace);
         } else if (event.type === "reason_step") {
           trace.reason = traceReasonStep(trace, event.step, event.total, event.label);
           bumpActivity(trace);
           setActivity(trace, "reasoning " + event.step + "/" + event.total + "\u2026");
         } else if (event.type === "reason_token") {
           if (trace.reason) { trace.reason.body.textContent += event.token; scrollDown(); }
         } else if (event.type === "reason_done") {
           if (trace.reason) {
             trace.reason.node.classList.add("done");
             if (event.conclusion) trace.reason.body.textContent = event.conclusion;
             trace.reason = null;
           }
         } else if (event.type === "context") {
           traceNotice(trace, "trimmed " + event.dropped + " old messages to fit the context window", true);
           bumpActivity(trace);
         } else if (event.type === "step") {
           // Show which step is active and out of how many, always.
           trace.stepLabel = "step " + event.step + (event.max_steps ? "/" + event.max_steps : "");
           setActivity(trace, "thinking\u2026");
           if (trace.answer && trace.answer.node.classList.contains("pending")) {
             trace.answer.text.textContent = "";
           }
         } else if (event.type === "think_token") {
           traceThinking(trace).body.textContent += event.token;
           bumpActivity(trace);
           setActivity(trace, "thinking\u2026");
           scrollDown();
         } else if (event.type === "token") {
           // First answer token: the thinking phase is over, collapse it so the
           // answer is the focus but the reasoning stays one click away.
           if (trace.thinking && !trace.thinking.done) {
             trace.thinking.done = true;
             trace.thinking.node.classList.remove("open");
           }
           traceAnswer(trace).text.textContent += event.token;
           bumpActivity(trace);
           setActivity(trace, "generating\u2026");
           scrollDown();
         } else if (event.type === "tool_call") {
           if (firstTool) { traceRouter(trace, event.name, event.args); firstTool = false; }
           trace.tool = traceTool(trace, event.name, event.args);
           trace.tool.node.classList.add("running");
           bumpActivity(trace);
           setActivity(trace, "running " + event.name + "\u2026");
         } else if (event.type === "tool_result") {
           if (trace.tool) {
             trace.tool.node.classList.remove("running");
             trace.tool.body.textContent = event.result || "";
             if (event.error) {
               trace.tool.node.classList.add("failed");
               trace.tool.pill.textContent = "failed";
             } else {
               trace.tool.node.classList.add("ok");
               trace.tool.pill.textContent = "ok";
             }
             trace.tool = null;
           }
           trace.answer = null;
           trace.thinking = null;
           setActivity(trace, "thinking\u2026");
         } else if (event.type === "notice") {
           traceNotice(trace, event.message || "", !!event.info);
           bumpActivity(trace);
           setActivity(trace, event.message || "working\u2026");
         } else if (event.type === "final") {
           var ans = traceAnswer(trace);
           ans.text.textContent = event.answer || "(no answer)";
           ans.node.classList.remove("pending");
           answered = true;
           updateContextMeter(estimateTokens(event.answer || ""));
           var meta = document.createElement("div");
           meta.className = "gmeta";
           var bits = [];
           if (event.tools_used && event.tools_used.length) bits.push(event.tools_used.join(", "));
           bits.push((event.steps || 1) + " step" + ((event.steps || 1) === 1 ? "" : "s"));
           bits.push(Math.round((event.elapsed_ms || 0) / 100) / 10 + "s");
           if (event.truncated) bits.push("continued from summary");
           meta.textContent = bits.join(" \u00b7 ");
           stopActivity(trace);
           trace.turn.appendChild(meta);
           addFeedbackBar(message, event.answer || "");
           loadRunCurve(conversationId);
           loadPerf();
         } else if (event.type === "cancelled") {
           traceNotice(trace, "stopped", false);
           stopActivity(trace);
           answered = true;
         } else if (event.type === "error") {
           traceNotice(trace, "error: " + event.error, false);
           stopActivity(trace);
           answered = true;
         }
       }

       if (!answered) {
         var a = traceAnswer(trace);
         a.node.classList.remove("pending");
         if (!a.text.textContent) a.text.textContent = "(no response)";
       }
       stopActivity(trace);
     } catch (err) {
       var a2 = traceAnswer(trace);
       a2.node.classList.remove("pending");
       if (err.name === "AbortError") {
         traceNotice(trace, "stopped", false);
       } else {
         // A broken stream on a local single-box setup almost always means the
         // model server ran out of memory and dropped the connection. Say that
         // plainly instead of showing a raw stream error.
         if (!a2.text.textContent) {
           a2.text.textContent = "The connection dropped, most likely the local "
             + "model server ran low on memory. Try a shorter request, or lower "
             + "AUTO_FETCH_RESULTS / the context size.";
         }
         a2.node.classList.add("failed");
       }
       stopActivity(trace);
     } finally {
       busy = false;
       controller = null;
       sendBtn.disabled = false;
       stopBtn.disabled = true;
       input.focus();
     }
   }

   async function loadMemory() {
     var list = document.getElementById("memoryList");
     try {
       var out = await fetchJSON("/api/memory?limit=50");
       var items = out.data.memories || [];
       list.innerHTML = "";
       if (!items.length) {
         list.textContent = "No stored notes.";
         return;
       }
       items.forEach(function(item) {
         var row = document.createElement("div");
         var del = document.createElement("button");
         del.textContent = "x";
         del.title = "Forget this note";
         del.style.marginRight = "6px";
         del.style.padding = "0 6px";
         del.onclick = function() { forgetMemory(item.key); };
         var label = document.createElement("span");
         label.textContent = item.key + ": " + item.value;
         row.appendChild(del);
         row.appendChild(label);
         list.appendChild(row);
       });
     } catch (err) {
       list.textContent = "Memory unavailable: " + err.message;
     }
   }

   async function saveMemory() {
     var key = document.getElementById("memKey").value.trim();
     var value = document.getElementById("memValue").value.trim();
     if (!key || !value) return;
     try {
       await fetchJSON("/api/memory", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ key: key, value: value })
       });
       document.getElementById("memKey").value = "";
       document.getElementById("memValue").value = "";
       loadMemory();
     } catch (err) {
       addSystem("Memory error: " + err.message);
     }
   }

   async function forgetMemory(key) {
     try {
       await fetchJSON("/api/memory/" + encodeURIComponent(key), { method: "DELETE" });
       loadMemory();
     } catch (err) {
       addSystem("Memory error: " + err.message);
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
       document.getElementById("cfgToolChars2").value = cfg.tool_result_chars;
       document.getElementById("cfgToolTemp").value = cfg.tool_temperature;
       document.getElementById("cfgThinking").checked = !!cfg.disable_thinking;
       document.getElementById("cfgFastPath").checked = !!cfg.fast_path;
       document.getElementById("cfgStablePrefix").checked = !!cfg.stable_prefix;
       document.getElementById("cfgSummarise").checked = !!cfg.summarise_tool_results;
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
       tool_result_chars: Number(document.getElementById("cfgToolChars2").value ||
                                 document.getElementById("cfgToolChars").value),
       tool_temperature: Number(document.getElementById("cfgToolTemp").value),
       disable_thinking: document.getElementById("cfgThinking").checked,
       fast_path: document.getElementById("cfgFastPath").checked,
       stable_prefix: document.getElementById("cfgStablePrefix").checked,
       summarise_tool_results: document.getElementById("cfgSummarise").checked
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
       if (typeof data.memories === "number") {
         msg += ", " + data.memories + " notes";
       }
       var warn = document.getElementById("prefixWarn");
       if (data.prefix && data.prefix.generation > 1) {
         warn.style.display = "";
         warn.textContent = "Prompt prefix changed " + (data.prefix.generation - 1) +
           " time(s) this session (last " + (data.prefix.changed_at || "") +
           "). Each change invalidates any cached prefix.";
       } else if (warn) {
         warn.style.display = "none";
       }
       if (data.tasks) {
         var running = data.tasks.running || [];
         msg += "\nTasks: " + data.tasks.total + " defined, " + running.length + " running";
         if (running.length) {
           msg += " (" + running.map(function(item) {
             return "step " + item.step + "/" + item.max_steps;
           }).join(", ") + ")";
         }
       }
       statusEl.textContent = msg;
       statusEl.className = "";
       statusEl.setAttribute("data-tip-below", "");
       statusEl.setAttribute("data-tip",
         "Live server status. Model: " + (data.model || "?")
         + " \u00b7 context " + (data.context_size || contextSize) + " tokens"
         + (data.ram_gb ? " \u00b7 " + data.ram_gb + "GB RAM" : ""));
       if (data.model_status && data.model_status.indexOf("error") === 0) statusEl.className = "error";
       else if (data.model_status === "starting" || data.model_status === "loading") statusEl.className = "warn";
     } catch (err) {
       statusEl.textContent = "Status unavailable: " + err.message;
       statusEl.className = "error";
     }
   }

   // ------------------------------------------------------------- views ---

   var currentView = "chat";
   var selectedTask = null;
   var runSource = null;
   var currentRunId = null;
   var tasksTimer = null;
   var modelLogTimer = null;

   function showView(name) {
     currentView = name;
     ["chat", "tasks", "models"].forEach(function(view) {
       var el = document.getElementById(view + "View");
       if (el) el.classList.toggle("hidden", view !== name);
       var nav = document.getElementById("nav" + view.charAt(0).toUpperCase() + view.slice(1));
       if (nav) nav.classList.toggle("active", view === name);
     });
     document.getElementById("composer").style.display = name === "chat" ? "flex" : "none";

     if (tasksTimer) { clearInterval(tasksTimer); tasksTimer = null; }
     if (modelLogTimer) { clearInterval(modelLogTimer); modelLogTimer = null; }

     if (name === "tasks") {
       loadTasks();
       tasksTimer = setInterval(loadTasks, 3000);
     } else if (name === "models") {
       loadModels();
       loadModelLog();
       modelLogTimer = setInterval(loadModelLog, 4000);
     }
   }

   // ------------------------------------------------------------- tasks ---

   function toggleTaskForm() {
     var form = document.getElementById("taskForm");
     form.style.display = form.style.display === "none" ? "block" : "none";
   }

   function statusPill(status) {
     var pill = document.createElement("span");
     pill.className = "pill " + (status || "");
     pill.textContent = status || "never run";
     return pill;
   }

   function relativeTime(value) {
     if (!value) return "never";
     var then = new Date(value.indexOf("Z") < 0 && value.indexOf("+") < 0 ? value + "Z" : value);
     var seconds = Math.round((Date.now() - then.getTime()) / 1000);
     if (isNaN(seconds)) return value;
     if (seconds < 0) return "in " + Math.abs(seconds) + "s";
     if (seconds < 60) return seconds + "s ago";
     if (seconds < 3600) return Math.round(seconds / 60) + "m ago";
     if (seconds < 86400) return Math.round(seconds / 3600) + "h ago";
     return Math.round(seconds / 86400) + "d ago";
   }

   function renderPrefillCurve(steps) {
     var box = document.getElementById("prefillCurve");
     box.innerHTML = "";
     if (!steps || steps.length < 2) return;

     var values = steps.map(function(s) { return s.prompt_tokens || 0; });
     var peak = Math.max.apply(null, values) || 1;
     var total = values.reduce(function(a, b) { return a + b; }, 0);

     var title = document.createElement("div");
     title.style.fontSize = "11px";
     title.style.opacity = "0.75";
     title.style.marginBottom = "4px";
     title.textContent = "prompt tokens per step (" + total + " total prefill)";
     box.appendChild(title);

     steps.forEach(function(s, i) {
       var row = document.createElement("div");
       row.style.display = "flex";
       row.style.alignItems = "center";
       row.style.gap = "6px";
       row.style.fontSize = "11px";
       row.style.lineHeight = "1.5";

       var label = document.createElement("span");
       label.style.opacity = "0.6";
       label.style.minWidth = "18px";
       label.textContent = "s" + (s.step === null ? i + 1 : s.step);

       var track = document.createElement("span");
       track.style.flex = "1";
       track.style.height = "9px";
       track.style.borderRadius = "3px";
       track.style.background = "rgba(127,127,127,0.18)";
       track.style.overflow = "hidden";

       var fill = document.createElement("span");
       fill.style.display = "block";
       fill.style.height = "100%";
       fill.style.width = Math.round(((s.prompt_tokens || 0) / peak) * 100) + "%";
       // Growth across steps is the thing to notice, so colour by it.
       var growing = i > 0 && (s.prompt_tokens || 0) > values[i - 1] * 1.15;
       fill.style.background = growing ? "#c2703a" : "#5a8f6f";
       track.appendChild(fill);

       var value = document.createElement("span");
       value.style.opacity = "0.7";
       value.style.minWidth = "40px";
       value.style.textAlign = "right";
       value.textContent = String(s.prompt_tokens || 0);

       row.appendChild(label);
       row.appendChild(track);
       row.appendChild(value);
       box.appendChild(row);
     });

     var verdict = document.createElement("div");
     verdict.style.fontSize = "11px";
     verdict.style.marginTop = "5px";
     var growth = values[values.length - 1] / (values[0] || 1);
     if (growth > 1.5) {
       verdict.style.color = "#c2703a";
       verdict.textContent = "Prompt grew " + growth.toFixed(1) +
         "x across the run. Shrink tool results or the tool list.";
     } else {
       verdict.style.opacity = "0.7";
       verdict.textContent = "Prompt stayed flat across steps.";
     }
     box.appendChild(verdict);
   }

   async function loadRunCurve(convId) {
     if (!convId) return;
     try {
       var out = await fetchJSON("/api/metrics/run/" + encodeURIComponent(convId));
       renderPrefillCurve(out.data.steps || []);
     } catch (err) {
       /* the run had no agent steps; nothing to draw */
     }
   }

   async function loadPerf() {
     var box = document.getElementById("perfStats");
     try {
       var out = await fetchJSON("/api/metrics/summary");
       var chat = out.data.chat || {};
       var step = out.data.agent_step || {};
       var lines = [];
       if (chat.samples) {
         lines.push("chat: " + chat.samples + " samples, " +
                    Math.round(chat.avg_duration_ms) + "ms avg");
       }
       if (step.samples) {
         lines.push("agent steps: " + step.samples);
         lines.push("  prompt tokens avg " + Math.round(step.avg_prompt_tokens || 0));
         lines.push("  ttft avg " + Math.round(step.avg_ttft_ms || 0) + "ms");
         lines.push("  decode " + (step.avg_decode_tps || 0).toFixed(1) + " tok/s");
         lines.push("  total prefill " + (step.total_prompt_tokens || 0) + " tokens");
       }
       box.textContent = lines.join("\n") || "no samples yet";
     } catch (err) {
       box.textContent = "stats unavailable: " + err.message;
     }
   }

   async function loadTasks() {
     try {
       var out = await fetchJSON("/api/tasks");
       renderTasks(out.data.tasks || []);
     } catch (err) {
       document.getElementById("taskCards").textContent = "Could not load tasks: " + err.message;
     }
   }

   function renderTasks(items) {
     var host = document.getElementById("taskCards");
     host.innerHTML = "";
     if (!items.length) {
       host.textContent = "No tasks yet.";
       return;
     }
     items.forEach(function(task) {
       var card = document.createElement("div");
       card.className = "card" + (selectedTask === task.id ? " selected" : "");
       card.onclick = function(e) {
         if (e.target.tagName === "BUTTON") return;
         selectTask(task.id);
       };

       var head = document.createElement("h4");
       var title = document.createElement("span");
       title.textContent = task.name;
       head.appendChild(title);
       head.appendChild(statusPill(task.live ? "running" : task.last_status));
       card.appendChild(head);

       var goal = document.createElement("div");
       goal.className = "goal";
       goal.textContent = task.goal;
       card.appendChild(goal);

       var meta = document.createElement("div");
       meta.className = "meta";
       var parts = [];
       parts.push(task.enabled ? "enabled" : "disabled");
       parts.push(task.interval_seconds > 0 ? "every " + task.interval_seconds + "s" : "manual");
       parts.push(task.run_count + " runs");
       parts.push("last " + relativeTime(task.last_run_at));
       if (task.live) {
         parts.push("step " + task.live.step + "/" + task.live.max_steps);
         if (task.live.tool) parts.push("tool " + task.live.tool);
       } else if (task.next_run_at) {
         parts.push("next " + relativeTime(task.next_run_at));
       }
       meta.textContent = parts.join(" | ");
       card.appendChild(meta);

       var actions = document.createElement("div");
       actions.className = "card-actions";
       actions.appendChild(taskButton("Run", function() { runTask(task.id); }));
       actions.appendChild(taskButton("Cancel", function() { cancelTask(task.id); }));
       actions.appendChild(taskButton(task.enabled ? "Disable" : "Enable", function() {
         updateTask(task.id, { enabled: !task.enabled });
       }));
       actions.appendChild(taskButton("Delete", function() { deleteTask(task.id, task.name); }));
       card.appendChild(actions);

       host.appendChild(card);
     });
   }

   function taskButton(label, handler) {
     var button = document.createElement("button");
     button.textContent = label;
     button.onclick = handler;
     return button;
   }

   async function createTask() {
     var body = {
       name: document.getElementById("tfName").value.trim(),
       goal: document.getElementById("tfGoal").value.trim(),
       interval_seconds: Number(document.getElementById("tfInterval").value) || 0,
       max_steps: Number(document.getElementById("tfSteps").value) || 6,
       tools: document.getElementById("tfTools").value.trim(),
       system_prompt: document.getElementById("tfSystem").value.trim() || null,
       use_history: document.getElementById("tfHistory").value === "true",
       enabled: true
     };
     if (!body.name || !body.goal) {
       alert("A task needs a name and a goal.");
       return;
     }
     try {
       var out = await fetchJSON("/api/tasks", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       document.getElementById("tfName").value = "";
       document.getElementById("tfGoal").value = "";
       toggleTaskForm();
       selectTask(out.data.task.id);
       loadTasks();
     } catch (err) {
       alert("Could not create the task: " + err.message);
     }
   }

   async function updateTask(taskId, patch) {
     try {
       await fetchJSON("/api/tasks/" + taskId, {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(patch)
       });
       loadTasks();
     } catch (err) {
       alert("Could not update the task: " + err.message);
     }
   }

   async function deleteTask(taskId, name) {
     if (!window.confirm("Delete " + name + " and its run history?")) return;
     try {
       await fetchJSON("/api/tasks/" + taskId, { method: "DELETE" });
       if (selectedTask === taskId) {
         selectedTask = null;
         closeRun();
         document.getElementById("runControls").style.display = "none";
         document.getElementById("monitorHeader").textContent = "Select a task to watch its runs.";
         document.getElementById("runFeed").innerHTML = "";
       }
       loadTasks();
     } catch (err) {
       alert("Could not delete the task: " + err.message);
     }
   }

   async function runTask(taskId) {
     try {
       var out = await fetchJSON("/api/tasks/" + taskId + "/run", { method: "POST" });
       if (out.data.error) {
         alert(out.data.error);
         return;
       }
       selectedTask = taskId;
       await loadRuns(taskId);
       openRun(out.data.run_id);
       loadTasks();
     } catch (err) {
       alert("Could not start the run: " + err.message);
     }
   }

   async function cancelTask(taskId) {
     try {
       await fetchJSON("/api/tasks/" + taskId + "/cancel", { method: "POST" });
       loadTasks();
     } catch (err) {
       alert("Could not cancel: " + err.message);
     }
   }

   function runSelectedTask() { if (selectedTask) runTask(selectedTask); }
   function cancelSelectedTask() { if (selectedTask) cancelTask(selectedTask); }

   async function selectTask(taskId) {
     selectedTask = taskId;
     document.getElementById("runControls").style.display = "block";
     try {
       var out = await fetchJSON("/api/tasks/" + taskId);
       var task = out.data.task;
       document.getElementById("monitorHeader").textContent =
         task.name + " -- " + task.goal;
       renderRunPicker(out.data.runs || []);
       if (out.data.runs && out.data.runs.length) {
         openRun(out.data.runs[0].id);
       } else {
         closeRun();
         document.getElementById("runFeed").innerHTML = "";
       }
       loadTasks();
     } catch (err) {
       document.getElementById("monitorHeader").textContent = "Could not load task: " + err.message;
     }
   }

   async function loadRuns(taskId) {
     try {
       var out = await fetchJSON("/api/tasks/" + taskId + "/runs?limit=20");
       renderRunPicker(out.data.runs || []);
     } catch (err) {
       // Leave the picker as it is; the stream is the important part.
     }
   }

   function renderRunPicker(runs) {
     var picker = document.getElementById("runPicker");
     picker.innerHTML = "";
     runs.forEach(function(run) {
       var option = document.createElement("option");
       option.value = run.id;
       option.textContent = run.status + " -- " + relativeTime(run.started_at) +
                            " -- " + run.trigger;
       picker.appendChild(option);
     });
     if (currentRunId) picker.value = currentRunId;
   }

   function closeRun() {
     if (runSource) {
       runSource.close();
       runSource = null;
     }
     currentRunId = null;
   }

   function feedLine(text, className) {
     var div = document.createElement("div");
     div.className = className || "feed-line";
     div.textContent = text;
     document.getElementById("runFeed").appendChild(div);
     return div;
   }

   function openRun(runId) {
     if (!runId) return;
     closeRun();
     currentRunId = runId;
     document.getElementById("runPicker").value = runId;
     var feed = document.getElementById("runFeed");
     feed.innerHTML = "";
     var partial = null;
     var currentCard = null;

     runSource = new EventSource("/api/runs/" + runId + "/stream");
     runSource.onmessage = function(message) {
       if (message.data === "[DONE]") {
         closeRun();
         loadTasks();
         return;
       }
       var event;
       try {
         event = JSON.parse(message.data);
       } catch (err) {
         return;
       }

       if (event.type === "start") {
         feedLine("started by " + event.trigger + " on " + event.model);
       } else if (event.type === "context") {
         feedLine("trimmed " + event.dropped + " old messages to fit the context");
       } else if (event.type === "step") {
         partial = null;
         feedLine("step " + event.step + " of " + event.max_steps);
       } else if (event.type === "token") {
         if (!partial) partial = feedLine("", "feed-partial");
         partial.textContent += event.token;
       } else if (event.type === "tool_call") {
         partial = null;
         currentCard = addToolCardTo(feed, event.name, event.args);
       } else if (event.type === "tool_result") {
         if (currentCard) {
           currentCard.body.textContent = event.result;
           if (event.error) currentCard.card.classList.add("failed");
           currentCard = null;
         }
       } else if (event.type === "final") {
         partial = null;
         feedLine(event.answer || "(empty answer)", "feed-answer");
         if (event.tools_used && event.tools_used.length) {
           feedLine("tools: " + event.tools_used.join(", ") +
                    " | steps: " + event.steps +
                    " | " + Math.round((event.elapsed_ms || 0) / 100) / 10 + "s");
         }
       } else if (event.type === "cancelled") {
         feedLine("cancelled");
       } else if (event.type === "error") {
         feedLine("error: " + event.error, "feed-answer");
       } else if (event.type === "done") {
         feedLine("run finished: " + event.status);
         closeRun();
         loadTasks();
       }
       feed.scrollIntoView({ block: "end" });
     };
     runSource.onerror = function() {
       // The server closes the stream when the run ends. Do not let EventSource
       // reconnect and replay the whole run in a loop.
       closeRun();
     };
   }

   function addToolCardTo(host, name, args) {
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
     host.appendChild(card);
     return { card: card, body: body };
   }

   // ------------------------------------------------------------ models ---

   async function loadModels() {
     try {
       var out = await fetchJSON("/api/models");
       var current = out.data.current;
       document.getElementById("modelCurrent").textContent =
         "model    : " + current.model + (current.cached ? "  (cached)" : "  (not downloaded)") +
         "\nadapter  : " + current.adapter + (current.adapter_path ? "  " + current.adapter_path : "") +
         "\nkv cache : " + (current.max_kv_size || "unbounded") +
         "\nstatus   : " + current.status +
         "\ncache dir: " + out.data.cache_dir;

       var table = document.getElementById("modelTable");
       table.innerHTML = "";
       (out.data.catalog || []).forEach(function(item) {
         var row = document.createElement("tr");
         var id = document.createElement("td");
         id.className = "id";
         id.textContent = item.id;
         var state = document.createElement("td");
         state.style.width = "110px";
         state.appendChild(statusPill(item.current ? "running" : (item.cached ? "ok" : "")));
         state.lastChild.textContent = item.current ? "in use" : (item.cached ? "cached" : "download");
         var action = document.createElement("td");
         action.style.width = "70px";
         var button = document.createElement("button");
         button.textContent = "Use";
         button.disabled = item.current;
         button.onclick = function() { useModel(item.id); };
         action.appendChild(button);
         row.appendChild(id);
         row.appendChild(state);
         row.appendChild(action);
         table.appendChild(row);
       });

       var select = document.getElementById("adapterSelect");
       select.innerHTML = "";
       (out.data.adapters || []).forEach(function(item) {
         var option = document.createElement("option");
         option.value = item.id;
         option.textContent = item.id + (item.modified ? "  (" + relativeTime(item.modified) + ")" : "");
         select.appendChild(option);
       });
       select.value = current.adapter;
       document.getElementById("kvSize").value = current.max_kv_size || 0;
     } catch (err) {
       document.getElementById("modelCurrent").textContent = "Could not load models: " + err.message;
     }
   }

   async function postModel(body) {
     try {
       var out = await fetchJSON("/api/models/select", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(body)
       });
       if (out.data.error) {
         alert(out.data.error);
         return;
       }
       loadModels();
       loadModelLog();
     } catch (err) {
       alert("Could not switch: " + err.message);
     }
   }

   function useModel(modelId) {
     if (!modelId) return;
     if (!window.confirm("Switch to " + modelId + "? The model server restarts, " +
                         "and an uncached model downloads first.")) return;
     postModel({ model: modelId, restart: true });
   }

   function applyModel() {
     postModel({
       adapter: document.getElementById("adapterSelect").value,
       max_kv_size: Number(document.getElementById("kvSize").value) || 0,
       restart: true
     });
   }

   async function loadModelLog() {
     try {
       var out = await fetchJSON("/api/logs/model?lines=120");
       var box = document.getElementById("modelLog");
       box.textContent = out.data.text || "(no output yet)";
       box.scrollTop = box.scrollHeight;
     } catch (err) {
       document.getElementById("modelLog").textContent = "Log unavailable: " + err.message;
     }
   }

   loadConfig();
   loadMemory();
   loadPerf();
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
MemoryRequest: Any = None
TaskRequest: Any = None
TaskUpdateRequest: Any = None
ModelSelectRequest: Any = None


def _define_api_models() -> None:
    """Define the Pydantic models in the module namespace.

    This module uses `from __future__ import annotations`, so every parameter
    annotation is a string at runtime. FastAPI resolves those strings against the
    endpoint function's __globals__, which is the module namespace. Models defined
    inside create_app() are invisible there, so FastAPI silently falls back to
    treating the body parameter as a query parameter and every POST returns 422.
    """
    global ChatRequest, FeedbackRequest, ChatResponse, ConfigRequest, ToolRequest
    global MemoryRequest, TaskRequest, TaskUpdateRequest, ModelSelectRequest
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
        tool_raw_chars: int | None = Field(None, ge=200, le=200000)
        tool_temperature: float | None = Field(None, ge=0.0, le=2.0)
        disable_thinking: bool | None = None
        fast_path: bool | None = None
        stable_prefix: bool | None = None
        summarise_tool_results: bool | None = None
        summarise_over_chars: int | None = Field(None, ge=500, le=100000)

    class ToolRequest(BaseModel):  # noqa: F811
        name: str = Field(..., min_length=1, max_length=64)
        args: dict = Field(default_factory=dict)
        conversation_id: str | None = Field(None, max_length=64)

    class MemoryRequest(BaseModel):  # noqa: F811
        key: str = Field(..., min_length=1, max_length=120)
        value: str = Field(..., min_length=1, max_length=8000)

    class TaskRequest(BaseModel):  # noqa: F811
        name: str = Field(..., min_length=1, max_length=120)
        goal: str = Field(..., min_length=1, max_length=8000)
        enabled: bool = True
        interval_seconds: int = Field(0, ge=0, le=2_592_000)
        max_steps: int = Field(6, ge=1, le=20)
        tools: str = Field("", max_length=500)
        system_prompt: str | None = Field(None, max_length=8000)
        use_history: bool = False
        # Run this task on its own model, swapping back afterwards. Empty means
        # whatever the server is already serving.
        model: str | None = Field(None, max_length=200)
        # Run another task when this one succeeds, handing the answer over
        # through a workspace file.
        next_task_id: str | None = Field(None, max_length=64)

    class TaskUpdateRequest(BaseModel):  # noqa: F811
        name: str | None = Field(None, min_length=1, max_length=120)
        goal: str | None = Field(None, min_length=1, max_length=8000)
        enabled: bool | None = None
        interval_seconds: int | None = Field(None, ge=0, le=2_592_000)
        max_steps: int | None = Field(None, ge=1, le=20)
        tools: str | None = Field(None, max_length=500)
        system_prompt: str | None = Field(None, max_length=8000)
        use_history: bool | None = None
        model: str | None = Field(None, max_length=200)
        next_task_id: str | None = Field(None, max_length=64)

    class ModelSelectRequest(BaseModel):  # noqa: F811
        model: str | None = Field(None, min_length=1, max_length=200)
        adapter: str | None = Field(None, max_length=100)
        max_kv_size: int | None = Field(None, ge=0, le=1_048_576)
        restart: bool = True


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

    from contextlib import asynccontextmanager

    registry = registry or ToolRegistry(config, db)
    model_client = ModelClient(config)
    agent = Agent(config, registry, model_client)
    tasks = TaskManager(config, db, model_manager, retrain_manager, model_client)

    @asynccontextmanager
    async def lifespan(_app):
        await tasks.start()
        try:
            yield
        finally:
            await tasks.stop()

    app = FastAPI(title="Local LLM", lifespan=lifespan)
    # Exposed so tests and the CLI can reach the scheduler without a global.
    app.state.tasks = tasks

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
            "prefix": dict(PREFIX_STATE),
            "stats": db.get_stats(),
            "agent_enabled": config.agent_enabled,
            "agent_max_steps": config.agent_max_steps,
            "model": config.model,
            "ram_gb": round(TOTAL_RAM_GB),
            "context_size": config.context_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "tools": registry.names(),
            "search_backend": config.search_backend,
            "memories": db.count_memories(),
            "adapter": model_manager.adapter_choice,
            "tasks": {
                "total": len(db.list_tasks()),
                "running": [
                    {"task_id": task_id, **(tasks.live_status(task_id) or {})}
                    for task_id in list(tasks.by_task)
                ],
            },
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
        max_tokens, temperature = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent
        # Mirror the streaming handler: route substantive messages through the
        # agent so the model router runs, even with the agent toggle off.
        if not use_agent and config.fast_path and quick_tool(request.message):
            use_agent = True
        elif not use_agent and config.knowledge_triage and is_substantive(request.message):
            use_agent = True

        # Recorded before generating, so a failed or empty run still leaves the
        # question in the transcript instead of silently dropping the turn.
        db.add_message(conversation_id, "user", request.message)
        tasks.note_chat_activity()
        tasks.chat_in_flight += 1

        try:
            stats: GenerationStats | None = None
            if use_agent:
                answer = ""
                trace: list[dict] = []
                error: str | None = None
                async for event in agent.run(
                    request.message, history, conversation_id, max_tokens, temperature
                ):
                    if event["type"] == "final":
                        answer = event["answer"]
                        trace = event.get("trace", [])
                    elif event["type"] == "usage":
                        db.log_metric(
                            "agent_step", event["total_ms"], 200,
                            stats=GenerationStats(
                                prompt_tokens=event["prompt_tokens"],
                                completion_tokens=event["completion_tokens"],
                                ttft_ms=event["ttft_ms"], total_ms=event["total_ms"],
                            ),
                            model=config.model, step=event.get("step"),
                            conversation_id=conversation_id,
                        )
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
                answer, stats = await model_client.complete_with_stats(
                    messages, max_tokens, temperature
                )
                trace = []

            db.add_message(
                conversation_id, "assistant", answer,
                meta={"trace": trace} if trace else None,
            )
            db.log_metric(
                "chat", (time.time() - start_time) * 1000, 200,
                stats=stats, model=config.model, conversation_id=conversation_id,
            )
            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "agent": use_agent,
                "trace": trace,
                "usage": stats.as_event() if stats else None,
            }
        except Exception as exc:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, str(exc))
            return JSONResponse(content={"error": str(exc)}, status_code=503)
        finally:
            tasks.chat_in_flight = max(0, tasks.chat_in_flight - 1)
            tasks.note_chat_activity()

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
        max_tokens, temperature = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent
        # Route substantive messages through the agent even when the agent
        # toggle is off, so the model router can decide answer vs search vs
        # weather. Deterministic shortcuts (bare URL, arithmetic) also need the
        # agent path to run. Greetings stay on the cheap plain path below.
        if not use_agent and config.fast_path and quick_tool(request.message):
            use_agent = True
        elif not use_agent and config.knowledge_triage and is_substantive(request.message):
            use_agent = True

        async def sse(event: dict) -> str:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        async def event_generator() -> AsyncGenerator[str, None]:
            answer = ""
            trace: list[dict] = []
            start_time = time.time()
            tasks.note_chat_activity()
            tasks.chat_in_flight += 1
            try:
                yield await sse({"type": "start", "conversation_id": conversation_id, "agent": use_agent})
                await asyncio.to_thread(db.add_message, conversation_id, "user", request.message)
                if use_agent:
                    async for event in agent.run(
                        request.message, history, conversation_id, max_tokens, temperature
                    ):
                        yield await sse(event)
                        if event["type"] == "final":
                            answer = event["answer"]
                            trace = event.get("trace", [])
                        elif event["type"] == "usage":
                            await asyncio.to_thread(
                                db.log_metric,
                                "agent_step", event["total_ms"], 200,
                                stats=GenerationStats(
                                    prompt_tokens=event["prompt_tokens"],
                                    completion_tokens=event["completion_tokens"],
                                    ttft_ms=event["ttft_ms"], total_ms=event["total_ms"],
                                ),
                                model=config.model, step=event.get("step"),
                                conversation_id=conversation_id,
                            )
                else:
                    system = {"role": "system", "content": config.system_prompt}
                    user = {"role": "user", "content": request.message}
                    messages, dropped = trim_to_context(
                        system, history, user, config.context_size, max_tokens
                    )
                    if dropped:
                        yield await sse({"type": "context", "dropped": dropped,
                                         "tokens": messages_tokens(messages)})
                    plain_stats = GenerationStats()
                    # Accumulate tokens in a list and join once: repeated string
                    # concatenation in a hot loop is O(n^2) and drags on long replies.
                    answer_parts: list[str] = []
                    async for token in model_client.stream(
                        messages, max_tokens, temperature, plain_stats
                    ):
                        answer_parts.append(token)
                        yield await sse({"type": "token", "token": token, "step": 1})
                    answer = "".join(answer_parts)
                    yield await sse({"type": "usage", "step": 1, **plain_stats.as_event()})
                    await asyncio.to_thread(
                        db.log_metric,
                        "chat_stream_gen", plain_stats.total_ms, 200, stats=plain_stats,
                        model=config.model, step=1, conversation_id=conversation_id,
                    )
                    yield await sse({"type": "final", "answer": answer, "steps": 1, "trace": []})

                if answer:
                    await asyncio.to_thread(
                        db.add_message,
                        conversation_id, "assistant", answer,
                        meta={"trace": trace} if trace else None,
                    )
                await asyncio.to_thread(
                    db.log_metric, "chat_stream", (time.time() - start_time) * 1000, 200)
            except Exception as exc:
                db.log_metric("chat_stream", (time.time() - start_time) * 1000, 503, str(exc))
                yield await sse({"type": "error", "error": str(exc)})
            finally:
                tasks.chat_in_flight = max(0, tasks.chat_in_flight - 1)
                tasks.note_chat_activity()
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

    def task_view(task: dict) -> dict:
        view = dict(task)
        view["live"] = tasks.live_status(task["id"])
        return view

    @app.get("/api/tasks")
    def list_tasks():
        return {"tasks": [task_view(task) for task in db.list_tasks()]}

    @app.post("/api/tasks")
    def create_task(request: TaskRequest):
        task = db.create_task(**request.model_dump())
        log(f"Task created: {task['name']} ({task['id']})")
        return {"task": task_view(task)}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        task = db.get_task(task_id)
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        return {"task": task_view(task), "runs": db.list_runs(task_id, 20)}

    @app.post("/api/tasks/{task_id}")
    def update_task(task_id: str, request: TaskUpdateRequest):
        task = db.update_task(task_id, request.model_dump(exclude_none=True))
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        return {"task": task_view(task)}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str):
        tasks.cancel_task(task_id)
        return {"deleted": db.delete_task(task_id)}

    @app.post("/api/tasks/{task_id}/run")
    async def run_task(task_id: str):
        task = db.get_task(task_id)
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        blocked = not_ready()
        if blocked is not None:
            return blocked
        try:
            run = await tasks.launch(task, trigger="manual")
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=409)
        return {"run_id": run.run_id, "task_id": task_id}

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        return {"cancelling": tasks.cancel_task(task_id)}

    @app.get("/api/tasks/{task_id}/runs")
    def task_runs(task_id: str, limit: int = Query(20, ge=1, le=200)):
        return {"runs": db.list_runs(task_id, limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = db.get_run(run_id)
        if run is None:
            return JSONResponse(content={"error": "no such run"}, status_code=404)
        return {"run": run, "events": db.run_events(run_id, limit=2000)}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        return {"cancelling": tasks.cancel_run(run_id)}

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        """Replay a run from the start, then follow it live until it finishes."""
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in tasks.subscribe(run_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @app.get("/api/models")
    def list_models():
        return {
            "current": model_manager.describe(),
            "catalog": model_catalog(config),
            "adapters": [{"id": "none", "path": "", "modified": None}] + list_adapters(),
            "cache_dir": str(hf_cache_dir()),
        }

    @app.post("/api/models/select")
    def select_model(request: ModelSelectRequest):
        """Point the model server at a different model or adapter and restart it."""
        if retrain_manager.status.get("running"):
            return JSONResponse(
                content={"error": "Retraining is running and owns the model server."},
                status_code=409,
            )
        running = list(tasks.by_task)
        if running:
            return JSONResponse(
                content={"error": f"{len(running)} task run(s) in flight. Cancel them first.",
                         "running": running},
                status_code=409,
            )
        try:
            changed = model_manager.swap(request.model, request.adapter)
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

        if request.model:
            config.model = request.model
        if request.adapter is not None:
            config.adapter = request.adapter
        if request.max_kv_size is not None and request.max_kv_size != config.max_kv_size:
            config.max_kv_size = request.max_kv_size
            model_manager.max_kv_size = request.max_kv_size
            changed = True

        if changed or request.restart:
            log(f"Switching to {config.model} (adapter: {config.adapter}). Restarting model server.")
            threading.Thread(target=model_manager.restart, daemon=True).start()
        return {
            "restarting": bool(changed or request.restart),
            "changed": changed,
            "current": model_manager.describe(),
            "note": "Weights download on first use. Watch /api/logs/model for progress.",
        }

    @app.get("/api/adapters")
    def adapters():
        return {"adapters": list_adapters(), "current": model_manager.adapter_choice}

    @app.get("/api/logs/{name}")
    def logs(name: str, lines: int = Query(120, ge=1, le=2000)):
        try:
            return {"name": name, "text": tail_log(name, lines)}
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=404)

    @app.get("/api/memory")
    def list_memory(
        limit: int = Query(50, ge=1, le=500),
        search: str | None = Query(None, max_length=100),
    ):
        return {"memories": db.recall(search, limit)}

    @app.post("/api/memory")
    def set_memory(request: MemoryRequest):
        db.remember(request.key, request.value)
        return {"stored": request.key}

    @app.delete("/api/memory/{key}")
    def delete_memory(key: str):
        return {"deleted": db.forget(key)}

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
        if retrain_manager.status.get("running"):
            return JSONResponse(
                content={"error": "Retraining is running and already owns the model server."},
                status_code=409,
            )
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

    @app.get("/api/metrics/summary")
    def metrics_summary(endpoint: str | None = Query(None, max_length=40)):
        return {
            "overall": db.metric_summary(endpoint),
            "chat": db.metric_summary("chat"),
            "agent_step": db.metric_summary("agent_step"),
        }

    @app.get("/api/metrics/run/{conversation_id}")
    def metrics_for_run(conversation_id: str):
        """Per-step prompt tokens for one conversation.

        A rising curve here is the agent re-sending its whole prompt each step.
        Flat means a prefix cache is doing its job.
        """
        steps = db.run_step_metrics(conversation_id)
        return {
            "conversation_id": conversation_id,
            "steps": steps,
            "total_prompt_tokens": sum(s["prompt_tokens"] or 0 for s in steps),
        }

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


def benchmark(
    config: Config,
    prompts: int = 3,
    prompt_tokens: int = 512,
    baseline_path: Path | None = None,
    save: bool = False,
) -> int:
    """Measure prefill and decode against a running model server.

    Reports time to first token separately from decode throughput, because on
    this hardware they respond to completely different fixes: TTFT is prompt
    processing and scales with how much context the agent re-sends, decode is
    memory bandwidth and scales with model size. A change that helps one often
    does nothing for the other.
    """
    import urllib.error
    import urllib.request

    if not port_open(config.model_port):
        print(f"No model server on port {config.model_port}. Start the app first.")
        return 1

    sizes = [64, prompt_tokens, prompt_tokens * 4]
    # Size the filler for the LARGEST prompt, or the biggest sample silently
    # runs short and the prefill curve looks flatter than it is.
    sentence = "The quick brown fox jumps over the lazy dog. "
    repeats = (max(sizes) * CHARS_PER_TOKEN) // len(sentence) + 2
    filler = sentence * repeats
    print(f"model : {config.model}")
    print(f"port  : {config.model_port}")
    print()
    print(f"{'prompt tok':>11}  {'ttft ms':>8}  {'decode tok/s':>12}  {'total ms':>9}")

    failures = 0
    samples: list[dict] = []
    for size in sizes:
        text = filler[: size * CHARS_PER_TOKEN]
        body = json.dumps({
            "model": config.model,
            "messages": [
                {"role": "user", "content": text + "\n\nReply with exactly one short sentence."}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{config.model_port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.time()
        first_at: float | None = None
        completion = 0
        prompt_reported = 0
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    if data.get("usage"):
                        prompt_reported = int(data["usage"].get("prompt_tokens") or 0)
                        completion = int(data["usage"].get("completion_tokens") or completion)
                    choices = data.get("choices") or []
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        if first_at is None:
                            first_at = time.time()
                        completion = completion or 0
                        completion += 1
        except Exception as exc:
            print(f"{size:>11}  request failed: {exc}")
            failures += 1
            continue

        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_at - started) * 1000) if first_at else total_ms
        decode_ms = max(1.0, total_ms - ttft_ms)
        tps = completion / (decode_ms / 1000.0)
        samples.append({
            "requested": size,
            "prompt_tokens": prompt_reported or size,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 2),
            "total_ms": round(total_ms, 1),
        })
        print(f"{prompt_reported or size:>11}  {ttft_ms:>8.0f}  {tps:>12.1f}  {total_ms:>9.0f}")

    print()
    if baseline_path is not None:
        _report_against_baseline(baseline_path, config.model, samples, save)

    print("TTFT rising steeply with prompt size is re-prefill cost. That is what")
    print("prompt caching, smaller tool results and a tighter tool catalogue attack.")
    print("Flat TTFT with low tok/s is memory bandwidth: use a smaller model.")
    return 1 if failures == len(sizes) else 0


def _report_against_baseline(path: Path, model: str, samples: list[dict], save: bool) -> None:
    """Compare this run to a saved one. Optimizing without this is guessing."""
    previous: dict = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"Could not read the benchmark baseline: {exc}", logging.WARNING)

    prior = previous.get("samples") if previous.get("model") == model else None
    if prior:
        print(f"vs baseline from {previous.get('recorded_at', 'unknown')}:")
        print(f"{'prompt tok':>11}  {'ttft delta':>12}  {'tok/s delta':>12}")
        by_size = {sample["requested"]: sample for sample in prior}
        for sample in samples:
            was = by_size.get(sample["requested"])
            if not was:
                continue
            ttft_delta = sample["ttft_ms"] - was["ttft_ms"]
            tps_delta = sample["tps"] - was["tps"]
            print(f"{sample['prompt_tokens']:>11}  {ttft_delta:>+11.0f}ms  {tps_delta:>+12.1f}")
        print()
    elif previous:
        print(f"(baseline is for {previous.get('model')}, not comparing)\n")

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": model,
            "recorded_at": iso(utc_now()),
            "samples": samples,
        }, indent=2), encoding="utf-8")
        print(f"Baseline saved to {path}. Re-run --bench after a change to compare.")


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

        db.remember("user.city", "Brussels", "conv1")
        db.remember("user.city", "Ghent", "conv1")
        notes = db.recall("city")
        if len(notes) != 1 or notes[0]["value"] != "Ghent":
            failures.append("memory upsert did not replace the earlier value")
        if not db.forget("user.city") or db.recall("city"):
            failures.append("forget did not delete the note")
        db.close()
        # close() must not leave this thread holding a dead handle.
        try:
            db.get_stats()
        except Exception as exc:
            failures.append(f"database did not reconnect after close(): {exc}")
        finally:
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

    # The JS scanner must not flag an apostrophe inside a well-formed literal,
    # because check_ui_syntax failing means main() refuses to start.
    if scan_js_strings('var msg = "it\'s fine"; // don\'t\n'):
        failures.append("scan_js_strings flagged a valid apostrophe")
    if not scan_js_strings('var broken = "starts here\nand ends there";\n'):
        failures.append("scan_js_strings missed a string split across lines")

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

    for local in ["http://127.0.0.1:8000/api/config", "http://localhost:8080/v1/models"]:
        try:
            guard_public_url(local)
            failures.append(f"guard_public_url allowed {local}")
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
    # Deterministic shortcuts: only URL and arithmetic route without a model.
    for text, expected in [
        ("17*23", "calculator"),
        ("(2+3)*4", "calculator"),
        ("https://example.com", "fetch_url"),
        ("look up CVEs online", None),        # now handled by the model router
        ("what is the weather in brussels", None),  # ditto
        ("hi", None),
    ]:
        routed = quick_tool(text)
        if (routed[0] if routed else None) != expected:
            failures.append(f"quick_tool({text!r}) routed to {routed!r}, expected {expected!r}")

    # Code requests answer directly; news/weather/lookups do not count as code.
    for text, want_code in [
        ("write a python script attempting 5 os recon techniques", True),
        ("python script to pull the newest CVEs", True),
        ("fix this function", True),
        ("implement a binary search in rust", True),
        ("write up the latest news on Apple", False),
        ("what are the latest CVEs for openssl?", False),
        ("whats tomorrows weather in brussels", False),
    ]:
        if is_code_request(text) != want_code:
            failures.append(f"is_code_request({text!r}) != {want_code}")
    # Self-contained code answers directly; code needing current info searches first.
    for text, want_lookup in [
        ("write a python script attempting 5 os recon techniques", True),
        ("write a script using the latest OpenAI API", True),
        ("implement a binary search in rust", False),
        ("write a regex for emails", False),
    ]:
        got = bool(is_code_request(text) and CODE_NEEDS_LOOKUP.search(text))
        if got != want_lookup:
            failures.append(f"code-needs-lookup({text!r}) != {want_lookup}")

    # The substantive gate keeps chit-chat off the router.
    for msg, want_substantive in [("hi", False), ("hey there", False), ("thanks!", False),
                                  ("what are the latest CVEs?", True),
                                  ("explain how a hash map works", True)]:
        if is_substantive(msg) != want_substantive:
            failures.append(f"is_substantive({msg!r}) != {want_substantive}")

    # chunk_text splits an oversized prompt into bounded, overlapping parts.
    _big = "para " * 4000  # ~20k chars
    _parts = chunk_text(_big, 6000, 600)
    if not _parts or any(len(pt) > 6000 for pt in _parts):
        failures.append("chunk_text produced an oversized chunk")
    if chunk_text("short", 6000, 600) != ["short"]:
        failures.append("chunk_text split a short string")

    # ThinkSplitter separates <think> reasoning from the answer, even when a tag
    # is split across streamed tokens.
    _sp = ThinkSplitter()
    _pieces = []
    for _tok in ["<thi", "nk>weigh", "ing it</thi", "nk>Answer."]:
        _pieces += _sp.feed(_tok)
    _think = "".join(c for k, c in _pieces if k == "think")
    _ans = "".join(c for k, c in _pieces if k == "answer")
    if _think != "weighing it" or _ans != "Answer.":
        failures.append(f"ThinkSplitter mis-split across tokens: {_think!r} / {_ans!r}")
    _sp2 = ThinkSplitter()
    if any(k == "think" for k, _ in _sp2.feed("a plain answer")):
        failures.append("ThinkSplitter invented a think block")

    # Retrieval guards: skip aggregator/listing/shell URLs and thin pages.
    for u in ("https://news.google.com/topics/abc", "https://techcrunch.com/category/ai/",
              "https://www.exploit-db.com/", "https://reddit.com/r/x"):
        if not is_low_value_url(u):
            failures.append(f"is_low_value_url missed {u}")
    for u in ("https://en.wikipedia.org/wiki/Transformer",
              "https://arstechnica.com/ai/2026/08/real-article/"):
        if is_low_value_url(u):
            failures.append(f"is_low_value_url wrongly flagged {u}")
    if not is_thin_page("\n".join(["Home", "News", "Login"] * 30)):
        failures.append("is_thin_page missed a navigation shell")
    if is_thin_page("A substantial article sentence with real prose content here. " * 8):
        failures.append("is_thin_page wrongly flagged an article")

    # The retrieval pipeline extracts the top result URLs from a search block.
    _sample = "1. A\n   https://a.com/x\n   s\n2. B\n   https://b.com/y\n   s"
    if top_result_urls(_sample, 1) != ["https://a.com/x"]:
        failures.append("top_result_urls did not return the first result URL")
    if top_result_urls(_sample, 5) != ["https://a.com/x", "https://b.com/y"]:
        failures.append("top_result_urls did not return URLs in order")

    # The router's JSON extractor must survive prose and code fences around the
    # object, and reject replies with no object.
    for raw, want in [
        ('{"action":"answer"}', {"action": "answer"}),
        ('Sure!\n```json\n{"action":"search","query":"x"}\n```', {"action": "search", "query": "x"}),
        ('here: {"action":"weather","location":"Paris","when":"tomorrow"} ok',
         {"action": "weather", "location": "Paris", "when": "tomorrow"}),
        ("no json here", None),
    ]:
        if extract_json_object(raw) != want:
            failures.append(f"extract_json_object({raw!r}) != {want!r}")

    for expr in ["9**9**9", "factorial(9**7)", "(10**20000)*(10**20000)"]:
        try:
            safe_eval(expr)
            failures.append(f"safe_eval({expr!r}) was not refused: unbounded intermediate")
        except ValueError:
            pass

    for expected_tool in ["web_search", "fetch_url", "calculator", "read_file",
                          "write_file", "edit_file", "search_files", "remember",
                          "recall_memory", "final_answer"]:
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

    # Argument aliasing: a model that says {"q": ...} should still get a search.
    search_tool = registry.get("web_search")
    if registry.normalise_args(search_tool, {"q": "mlx lora"}) != {"query": "mlx lora"}:
        failures.append("normalise_args did not map q onto query")
    if registry.normalise_args(search_tool, {"nonsense": "x"}) != {"query": "x"}:
        failures.append("normalise_args did not adopt a lone value for a single-argument tool")

    # An allowlist must never remove the loop's exit tool.
    limited = ToolRegistry(Config(agent_tools="calculator"))
    if set(limited.names()) != {"calculator", "final_answer"}:
        failures.append(f"AGENT_TOOLS allowlist produced {limited.names()}")

    known = set(registry.names())
    if parse_tool_call('Here is the config: {"name": "gpt", "temperature": 0.7}', known) is not None:
        failures.append("parse_tool_call treated a JSON answer as a tool call")
    if parse_tool_call('{"tool": "web_search", "args": {"query": "x"}}', known) is None:
        failures.append("parse_tool_call rejected a valid call against the known set")

    # Workspace file tools, against a throwaway workspace.
    original_workspace = WORKSPACE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        globals()["WORKSPACE_DIR"] = Path(tmp) / "workspace"
        try:
            registry.call("write_file", {"path": "notes/a.txt", "content": "alpha\nbeta\n"})
            body, error = registry.call("read_file", {"path": "notes/a.txt"})
            if error or "alpha" not in body:
                failures.append(f"write_file/read_file round-trip failed: {body!r}")
            _, error = registry.call(
                "edit_file", {"path": "notes/a.txt", "find": "beta", "replace": "gamma"}
            )
            body, _ = registry.call("read_file", {"path": "notes/a.txt"})
            if error or "gamma" not in body or "beta" in body:
                failures.append("edit_file did not apply the replacement")
            _, error = registry.call(
                "edit_file", {"path": "notes/a.txt", "find": "absent", "replace": "x"}
            )
            if not error:
                failures.append("edit_file accepted a snippet that is not in the file")
            hits, error = registry.call("search_files", {"pattern": "gam+a"})
            if error or "a.txt" not in hits:
                failures.append(f"search_files did not find the match: {hits!r}")
        finally:
            globals()["WORKSPACE_DIR"] = original_workspace

    # Tasks: creation, scheduling, the run lifecycle, and event replay.
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "tasks.db")
        task = db.create_task(name="nightly", goal="check the news", interval_seconds=3600)
        if not task["id"] or task["next_run_at"] is None:
            failures.append("a scheduled task was not armed on creation")
        if [t["id"] for t in db.due_tasks()] != [task["id"]]:
            failures.append("a task armed for now did not come back as due")

        db.schedule_next(task["id"], 3600)
        if db.due_tasks():
            failures.append("schedule_next did not push the next run into the future")

        if db.update_task(task["id"], {"enabled": False})["next_run_at"] is not None:
            failures.append("disabling a task left it armed")
        manual = db.create_task(name="manual", goal="do a thing", interval_seconds=0)
        if manual["next_run_at"] is not None:
            failures.append("a manual task was armed anyway")

        run_id = db.create_run(task["id"], "manual", "test-model")
        db.append_event(run_id, 1, "step", {"step": 1})
        db.append_event(run_id, 2, "final", {"answer": "done"})
        events = db.run_events(run_id)
        if [e["type"] for e in events] != ["step", "final"]:
            failures.append("task events did not round-trip in order")
        if db.run_events(run_id, after_seq=1)[0]["type"] != "final":
            failures.append("after_seq did not skip replayed events")
        db.finish_run(run_id, "ok", "done", None, 2, 12.0, ["calculator"])
        if db.get_run(run_id)["status"] != "ok":
            failures.append("finish_run did not record the outcome")
        if db.get_task(task["id"])["last_status"] != "ok":
            failures.append("finish_run did not update the task summary")

        stuck = db.create_run(task["id"], "manual", "test-model")
        if db.reset_orphan_runs() != 1 or db.get_run(stuck)["status"] != "interrupted":
            failures.append("reset_orphan_runs left a run marked running")

        if not db.delete_task(task["id"]) or db.get_run(run_id) is not None:
            failures.append("deleting a task left its runs behind")
        db.close()

    # Model and adapter selection.
    if resolve_adapter("none") is not None or resolve_adapter("") is not None:
        failures.append("resolve_adapter returned a path for the base model")
    try:
        resolve_adapter("../../etc")
        failures.append("resolve_adapter allowed an id outside the backups directory")
    except ValueError:
        pass
    # An adapter must never follow a model swap onto a base it does not fit.
    with tempfile.TemporaryDirectory() as tmp:
        original_dirs = (ADAPTER_DIR, ADAPTER_BACKUP_DIR)
        globals()["ADAPTER_DIR"] = Path(tmp) / "adapters" / "latest"
        globals()["ADAPTER_BACKUP_DIR"] = Path(tmp) / "adapters" / "backups"
        try:
            ADAPTER_DIR.mkdir(parents=True)
            (ADAPTER_DIR / "adapters.safetensors").write_bytes(b"stub")
            manager = ModelServerManager("org/base-a", 0, ADAPTER_DIR)
            manager.adapter_choice = "latest"
            if manager.adapter_path() != ADAPTER_DIR:
                failures.append("an untagged adapter was refused")

            write_adapter_base(ADAPTER_DIR, "org/base-a")
            if manager.adapter_path() != ADAPTER_DIR:
                failures.append("a matching adapter was refused")

            manager.swap("org/base-b")
            if manager.adapter_path() is not None:
                failures.append("a mismatched adapter survived a model swap")
            if not manager.describe()["adapter_mismatch"]:
                failures.append("describe() did not report the adapter mismatch")
            if any(flag in manager._build_cmd(True) for flag in ("--adapter-path", "--adapter")):
                failures.append("the server command still passed a mismatched adapter")
        finally:
            globals()["ADAPTER_DIR"], globals()["ADAPTER_BACKUP_DIR"] = original_dirs

    # A reasoning model's chain of thought must not be executed as a tool call.
    thinking = ('<think>Maybe {"tool": "web_search", "args": {"query": "no"}} fits.</think>'
                '{"tool": "calculator", "args": {"expression": "2+2"}}')
    parsed = parse_tool_call(thinking, {"calculator", "web_search"})
    if parsed != ("calculator", {"expression": "2+2"}):
        failures.append(f"a <think> block hijacked the tool call: {parsed}")
    if parse_tool_call('<think>considering {"tool": "web_search"', {"web_search"}) is not None:
        failures.append("an unterminated <think> block produced a tool call mid-stream")
    if strip_reasoning("<reasoning>hidden</reasoning>visible") != "visible":
        failures.append("strip_reasoning left the reasoning block in place")
    if strip_reasoning("no tags here") != "no tags here":
        failures.append("strip_reasoning mangled ordinary text")

    # Deterministic routing, including the cases that must NOT route.
    for text, expected in [
        ("17*23", ("calculator", {"expression": "17*23"})),
        ("https://example.com/a", ("fetch_url", {"url": "https://example.com/a"})),
        ("what is the capital of France?", None),
        ("2024", None),
    ]:
        if fast_path_call(text) != expected:
            failures.append(f"fast_path_call({text!r}) returned {fast_path_call(text)!r}")

    # Prefix stability: step k+1's prompt must start with step k's, so an
    # extending prompt keeps matching a server-side cache.
    def trace_prompts(stable: bool) -> list[list[str]]:
        cfg = Config(context_size=1200, stable_prefix=stable)
        probe = Agent(cfg, ToolRegistry(cfg), ModelClient(cfg))
        base_msgs = [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "the question"}]
        scratch: list[dict] = []
        out = []
        for index in range(12):
            scratch.append({"role": "assistant", "content": f"call {index}"})
            scratch.append({"role": "user", "content": "TOOL RESULT:\n" + "z" * 700})
            messages, _ = probe.assemble(base_msgs, scratch, 128)
            if messages[:2] != base_msgs:
                failures.append("assemble evicted the pinned system prompt or question")
            out.append([m["content"] for m in messages])
        return out

    def extend_hits(prompts: list[list[str]]) -> int:
        return sum(1 for i in range(len(prompts) - 1)
                   if prompts[i + 1][:len(prompts[i])] == prompts[i])

    stable_hits = extend_hits(trace_prompts(True))
    drop_hits = extend_hits(trace_prompts(False))
    if stable_hits <= drop_hits:
        failures.append(f"stable_prefix did not improve prefix reuse ({stable_hits} vs {drop_hits})")

    # The raw tool cap must stay above the context cap, or the summariser never
    # sees enough text to summarise and silently becomes dead code.
    sized = Config()
    sized.apply({"tool_result_chars": 4000, "tool_raw_chars": 500})
    if sized.tool_raw_chars < sized.tool_result_chars:
        failures.append("tool_raw_chars was allowed below tool_result_chars")

    # Generation accounting.
    sample = GenerationStats(prompt_tokens=900, completion_tokens=30, ttft_ms=500, total_ms=2500)
    if abs(sample.decode_tps - 15.0) > 0.1:
        failures.append(f"decode_tps computed {sample.decode_tps}, expected 15")
    if sample.as_event()["estimated"] is not True:
        failures.append("stats without server usage were not marked estimated")

    catalog = model_catalog(Config(model="someone/custom-model"))
    if not any(entry["current"] and entry["id"] == "someone/custom-model" for entry in catalog):
        failures.append("model_catalog did not include the model currently in use")
    if len(catalog) != len({entry["id"] for entry in catalog}):
        failures.append("model_catalog returned duplicates")

    # The live event buffer must number events and survive a token flood.
    live = TaskRun("run1", "task1", "demo")
    live.publish({"type": "start"})
    for index in range(TaskRun.BUFFER_LIMIT + 200):
        live.publish({"type": "token", "token": str(index)})
    live.publish({"type": "final", "answer": "x"})
    if live.seq != TaskRun.BUFFER_LIMIT + 202:
        failures.append("TaskRun did not number every published event")
    if not any(e["type"] == "start" for e in live.events):
        failures.append("TaskRun buffer dropped a structural event under token pressure")
    if len(live.events) > TaskRun.BUFFER_LIMIT + 2:
        failures.append("TaskRun buffer grew past its limit")

    # The agent must keep the system prompt and the question no matter how long
    # the tool trace gets.
    agent = Agent(config, registry, ModelClient(config))

    # Disk-backed partial sink: one reused handle, byte cap, tail-biased read.
    agent.PARTIAL_MAX_BYTES = 1500
    agent.partial_begin("selftest-conv", "a test request")
    for _i in range(30):
        agent.partial_add("selftest-conv", f"finding {_i}: " + "y" * 80)
    _entry = agent._partials.get(agent._partial_key("selftest-conv"))
    if not _entry or not _entry["capped"]:
        failures.append("partial sink did not enforce its byte cap")
    if len(agent._partials) != 1:
        failures.append("partial sink reopened the handle per write")
    _sal = agent.salvage("selftest-conv", "ran low")
    if "finding 0" not in _sal or "[...]" not in agent.partial_read("selftest-conv", 400):
        failures.append("partial sink read-back/salvage lost the saved work")
    agent.partial_begin("selftest-conv", "second turn")
    if "finding 0" in agent.partial_read("selftest-conv"):
        failures.append("partial sink did not truncate on a new turn")
    agent._close_partial("selftest-conv")

    base, _ = agent.build_base([], "what is the capital of France?", 256)
    long_scratch = [{"role": "user", "content": "t" * 8000} for _ in range(8)]
    assembled, cut = agent.assemble(base, long_scratch, 256)
    if cut == 0:
        failures.append("agent.assemble kept a trace that cannot fit the context")
    if assembled[: len(base)] != base:
        failures.append("agent.assemble dropped part of the pinned question or system prompt")

    changed = config.apply({"max_tokens": 99999, "context_size": 1024, "model": "evil/model"})
    if config.model == "evil/model":
        failures.append("config.apply changed a field outside MUTABLE")
    if config.max_tokens >= config.context_size:
        failures.append("config.apply left max_tokens above the context size")
    if "context_size" not in changed:
        failures.append("config.apply did not report context_size as changed")

    # Safeguards clamp at construction and on live edit, and stay in valid bands.
    bad = Config(chunk_size_ratio=0.99, chunk_trigger_ratio=0.3,
                 ready_wait_timeout=99999, reasoning_tokens=10 ** 9)
    if bad.chunk_size_ratio > bad.chunk_trigger_ratio:
        failures.append("chunk_size_ratio not clamped below the trigger at construction")
    if not (2.0 <= bad.ready_wait_timeout <= 300.0):
        failures.append("ready_wait_timeout not clamped at construction")
    if not (64 <= bad.reasoning_tokens <= 2048):
        failures.append("reasoning_tokens not clamped at construction")
    edited = Config()
    edited.apply({"reasoning_tokens": 10 ** 9, "ready_wait_timeout": -5})
    if not (64 <= edited.reasoning_tokens <= 2048 and edited.ready_wait_timeout >= 2.0):
        failures.append("safeguards not clamped on live edit")
    for safeguard in ("reasoning_tokens", "chunk_trigger_ratio", "stall_timeout",
                      "ready_wait_timeout", "auto_fetch_char_cap"):
        if safeguard not in Config.MUTABLE:
            failures.append(f"safeguard {safeguard} is not live-editable")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS  single copy of the file")
    print("PASS  embedded UI parses and renders")
    print("PASS  database schema round-trips")
    print("PASS  tool call parsing, calculator sandbox, workspace confinement")
    print("PASS  file tools, argument aliasing, tool allowlist, fetch guard")
    print("PASS  task scheduling, run lifecycle, event replay, model catalogue")
    print("PASS  model swapping keeps mismatched adapters out of the server")
    print("PASS  reasoning stripping, fast path, prefix reuse, token accounting")
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
    parser.add_argument("--allow-local-fetch", action="store_true",
                        default=os.environ.get("ALLOW_LOCAL_FETCH") == "1",
                        help="Let fetch_url reach loopback and private addresses. Off by default.")
    parser.add_argument("--agent-tools", default=os.environ.get("AGENT_TOOLS", ""),
                        help="Comma-separated allowlist of tool names. Empty offers all of them.")
    parser.add_argument("--adapter", default=os.environ.get("ADAPTER", "latest"),
                        help="LoRA adapter to load: latest, none, or a backup id.")
    parser.add_argument("--model-catalog", default=os.environ.get("MODEL_CATALOG", ""),
                        help="Extra model ids to offer in the Models tab, comma separated.")
    parser.add_argument("--max-concurrent-tasks", type=int,
                        default=int(os.environ.get("MAX_CONCURRENT_TASKS", "1")),
                        help="How many background task runs may execute at once.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the model catalogue and which weights are cached, then exit.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print the defined tasks and their last run, then exit.")
    parser.add_argument("--add-task", metavar="JSON",
                        help='Create a task and exit, e.g. \'{"name": "n", "goal": "g", '
                             '"interval_seconds": 3600}\'')
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
    parser.add_argument("--bench", action="store_true",
                        help="Measure prefill and decode against a running model server, then exit.")
    parser.add_argument("--bench-save", action="store_true",
                        help="With --bench, save the result as the baseline to compare against later.")
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

    if args.bench:
        ensure_dirs()
        sys.exit(benchmark(
            Config(model=args.model, model_port=args.model_port),
            baseline_path=DATA_DIR / "bench_baseline.json",
            save=args.bench_save,
        ))

    if args.list_models or args.list_tasks or args.add_task:
        ensure_dirs()
        early_db = Database(DB_PATH)
        try:
            if args.list_models:
                catalog_config = Config(model=args.model, model_catalog=args.model_catalog)
                for entry in model_catalog(catalog_config):
                    marker = "*" if entry["current"] else " "
                    state = "cached" if entry["cached"] else "not downloaded"
                    print(f"{marker} {entry['id']}  ({state})")
                adapters = list_adapters()
                print("\nadapters: " + (", ".join(
                    a["id"] + (f" [{a['base_model']}]" if a.get("base_model") else "")
                    for a in adapters) or "none"))
                print(f"cache   : {hf_cache_dir()}")
            if args.add_task:
                try:
                    spec = json.loads(args.add_task)
                except json.JSONDecodeError as exc:
                    sys.exit(f"--add-task is not valid JSON: {exc}")
                if not spec.get("name") or not spec.get("goal"):
                    sys.exit("--add-task needs at least a name and a goal.")
                created = early_db.create_task(**spec)
                print(f"Created task {created['id']}: {created['name']}")
            if args.list_tasks:
                for task in early_db.list_tasks():
                    schedule = (f"every {task['interval_seconds']}s"
                                if task["interval_seconds"] else "manual")
                    print(f"{task['id']}  {task['name']}  [{schedule}] "
                          f"{'enabled' if task['enabled'] else 'disabled'}  "
                          f"runs={task['run_count']}  last={task['last_status'] or 'never'}")
                    print(f"    goal: {task['goal'][:120]}")
        finally:
            early_db.close()
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
        agent_tools=args.agent_tools,
        allow_local_fetch=args.allow_local_fetch,
        adapter=args.adapter,
        model_catalog=args.model_catalog,
        max_concurrent_tasks=args.max_concurrent_tasks,
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

    # Announce the model and where the RAM-based default came from, so it is
    # obvious on a new machine why a particular model was chosen.
    if os.environ.get("MODEL_ID"):
        log(f"Model: {config.model} (from MODEL_ID)")
    elif args.model == DEFAULT_MODEL:
        log(f"Model: {config.model} (auto-selected for {TOTAL_RAM_GB:.0f}GB RAM; "
            f"set MODEL_ID to override)")
    else:
        log(f"Model: {config.model}")
    if os.environ.get("CONTEXT_SIZE"):
        log(f"Context: {config.context_size} tokens (from CONTEXT_SIZE)")
    else:
        log(f"Context: {config.context_size} tokens "
            f"(auto-sized for {TOTAL_RAM_GB:.0f}GB RAM; large prompts chunk above "
            f"~{int(config.context_size * config.chunk_trigger_ratio)} tokens)")

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

    model_manager = ModelServerManager(
        config.model, config.model_port, ADAPTER_DIR, config.max_kv_size, config.adapter,
        kv_bits=config.kv_bits, kv_group_size=config.kv_group_size,
        quantized_kv_start=config.quantized_kv_start,
        prompt_cache_dir=config.prompt_cache_dir,
    )
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

    shutting_down = threading.Event()

    def handle_signal(signum, frame):
        if shutting_down.is_set():
            return
        shutting_down.set()
        log(f"Received signal {signum}, shutting down...")
        model_manager.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"Model: {config.model}")
    log(f"System prompt: {config.system_prompt[:60]}...")
    log(f"Context: {config.context_size} tokens, max_tokens {config.max_tokens}, temperature {config.temperature}")
    log(f"Agent: {'on' if config.agent_enabled else 'off'}, max steps {config.agent_max_steps}, "
        f"tools: {', '.join(registry.names())}")
    log(f"Adapter: {config.adapter} ({model_manager.adapter_path() or 'base model'})")
    scheduled = [t for t in db.list_tasks() if t["enabled"] and t["interval_seconds"]]
    log(f"Tasks: {len(db.list_tasks())} defined, {len(scheduled)} on a schedule, "
        f"{config.max_concurrent_tasks} at a time")
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
