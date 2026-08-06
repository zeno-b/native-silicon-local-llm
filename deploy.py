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

Environment overrides:
   MODEL_ID, MODEL_PORT, WEB_PORT, TRAIN_ITERS, TRAIN_LR, TRAIN_SEQ_LEN,
   MAX_TOKENS, SYSTEM_PROMPT, AUTO_RETRAIN_THRESHOLD

Notes:
- This script creates ./.venv and installs dependencies on first run.
- It stores data in ./data and logs in ./logs.
- Retraining stops the model server temporarily, trains, then restarts it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SFT_DIR = DATA_DIR / "sft"
LOG_DIR = ROOT / "logs"
ADAPTER_ROOT = ROOT / "adapters"
LEGACY_ADAPTER_DIR = ADAPTER_ROOT / "latest"
ADAPTER_BACKUP_DIR = ADAPTER_ROOT / "backups"
DB_PATH = DATA_DIR / "feedback.db"

DEFAULT_MODEL = os.environ.get(
    "MODEL_ID",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
)
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant.",
)

# Bump when HTML_PAGE changes. Shown in the header and returned by /api/health so
# a stale browser cache is immediately visible rather than silently misleading.
UI_BUILD = "2026-08-06.4"

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
    for d in [DATA_DIR, SFT_DIR, LOG_DIR, ADAPTER_ROOT, ADAPTER_BACKUP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    model_id TEXT NOT NULL,
                    adapter_path TEXT,
                    example_count INTEGER DEFAULT 0,
                    iters INTEGER,
                    learning_rate TEXT,
                    status TEXT DEFAULT 'running',
                    trigger TEXT,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    run_id INTEGER NOT NULL,
                    feedback_id INTEGER NOT NULL,
                    PRIMARY KEY (run_id, feedback_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_examples_feedback
                ON training_examples(feedback_id)
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
        untrained_for: str | None = None,
        needs_correction: bool = False,
        with_runs: bool = False,
    ) -> list[dict]:
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []
        if approved_only:
            sql += " AND approved_for_training = 1"
        if needs_correction:
            # Thumbs-down with no correction is unusable for training: there is a
            # rejected answer but no target. Surfacing it turns dead rows into work.
            sql += " AND rating < 0 AND (corrected_response IS NULL OR corrected_response = '')"
        if untrained_for:
            sql += """ AND NOT EXISTS (
                SELECT 1 FROM training_examples te
                JOIN training_runs tr ON tr.id = te.run_id
                WHERE te.feedback_id = feedback.id
                  AND tr.model_id = ? AND tr.status = 'success')"""
            params.append(untrained_for)
        if search:
            sql += " AND (user_prompt LIKE ? OR assistant_response LIKE ? OR corrected_response LIKE ?)"
            params.extend([f"%{search}%"] * 3)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = [dict(row) for row in self.execute(sql, tuple(params)).fetchall()]
        if with_runs:
            for row in rows:
                row["trained_by"] = self.runs_for_feedback(row["id"])
        return rows

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
        needs_correction = self.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE rating < 0 "
            "AND (corrected_response IS NULL OR corrected_response = '')"
        ).fetchone()["cnt"]
        return {
            "total": total,
            "approved": approved,
            "needs_correction": needs_correction,
            "untrained": untrained,
            "positive": positive,
            "negative": negative,
            "corrected": corrected,
        }

    def get_untrained_count(self, model_id: str | None = None) -> int:
        """Approved feedback not yet used in a successful run for this model.

        Per-model, because switching to a new base model means none of the
        corpus has been trained into it yet even though the rows are old.
        """
        if model_id is None:
            return self.execute(
                "SELECT COUNT(*) as cnt FROM feedback "
                "WHERE approved_for_training = 1 AND trained_at IS NULL"
            ).fetchone()["cnt"]
        return self.execute(
            """
            SELECT COUNT(*) as cnt FROM feedback f
            WHERE f.approved_for_training = 1
              AND NOT EXISTS (
                SELECT 1 FROM training_examples te
                JOIN training_runs tr ON tr.id = te.run_id
                WHERE te.feedback_id = f.id
                  AND tr.model_id = ?
                  AND tr.status = 'success'
              )
            """,
            (model_id,),
        ).fetchone()["cnt"]

    def start_run(
        self,
        model_id: str,
        adapter_path: str,
        iters: int,
        learning_rate: str,
        trigger: str,
    ) -> int:
        cursor = self.execute(
            """INSERT INTO training_runs
               (model_id, adapter_path, iters, learning_rate, trigger, status)
               VALUES (?, ?, ?, ?, ?, 'running')""",
            (model_id, adapter_path, iters, learning_rate, trigger),
        )
        self.commit()
        return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        example_count: int = 0,
        error: str | None = None,
    ) -> None:
        self.execute(
            "UPDATE training_runs SET status = ?, example_count = ?, error = ?, "
            "finished_at = ? WHERE id = ?",
            (status, example_count, error, datetime.now(timezone.utc).isoformat(), run_id),
        )
        self.commit()

    def record_run_examples(self, run_id: int, feedback_ids: list[int]) -> None:
        for feedback_id in feedback_ids:
            self.execute(
                "INSERT OR IGNORE INTO training_examples (run_id, feedback_id) VALUES (?, ?)",
                (run_id, feedback_id),
            )
        self.commit()

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self.execute(
            "SELECT * FROM training_runs ORDER BY started_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict | None:
        row = self.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        examples = self.execute(
            """SELECT f.* FROM feedback f
               JOIN training_examples te ON te.feedback_id = f.id
               WHERE te.run_id = ? ORDER BY f.id""",
            (run_id,),
        ).fetchall()
        run["examples"] = [dict(e) for e in examples]
        return run

    def runs_for_feedback(self, feedback_id: int) -> list[dict]:
        rows = self.execute(
            """SELECT tr.id, tr.model_id, tr.status, tr.started_at
               FROM training_runs tr
               JOIN training_examples te ON te.run_id = tr.id
               WHERE te.feedback_id = ? ORDER BY tr.id""",
            (feedback_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def known_models(self) -> list[str]:
        """Every model this corpus has been collected under or trained into."""
        rows = self.execute(
            """SELECT DISTINCT model_id FROM (
                   SELECT model_id FROM training_runs WHERE model_id IS NOT NULL
                   UNION SELECT model_id FROM feedback WHERE model_id IS NOT NULL
               ) ORDER BY model_id"""
        ).fetchall()
        return [r["model_id"] for r in rows]

    def update_feedback(
        self,
        feedback_id: int,
        approved: bool | None = None,
        corrected_response: str | None = None,
        rating: int | None = None,
    ) -> dict | None:
        """Curate a row. Editing the answer re-opens it for every model."""
        sets, params = [], []
        if approved is not None:
            sets.append("approved_for_training = ?")
            params.append(1 if approved else 0)
        if corrected_response is not None:
            sets.append("corrected_response = ?")
            params.append(corrected_response.strip() or None)
        if rating is not None:
            sets.append("rating = ?")
            params.append(rating)
        if not sets:
            return self.get_feedback(feedback_id)

        if corrected_response is not None:
            # The answer changed, so prior training on it no longer represents
            # this row. Clear the provenance rather than silently keeping a
            # "trained into X" badge that refers to different text.
            self.execute("DELETE FROM training_examples WHERE feedback_id = ?", (feedback_id,))
            sets.append("trained_at = NULL")

        params.append(feedback_id)
        self.execute(f"UPDATE feedback SET {', '.join(sets)} WHERE id = ?", tuple(params))
        self.commit()
        return self.get_feedback(feedback_id)

    def get_feedback(self, feedback_id: int) -> dict | None:
        row = self.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
        return dict(row) if row else None

    def export_corpus(self) -> str:
        """Approved feedback as JSONL, portable to another machine or install."""
        rows = self.execute(
            "SELECT user_prompt, assistant_response, corrected_response, rating, model_id "
            "FROM feedback WHERE approved_for_training = 1 ORDER BY id"
        ).fetchall()
        return "".join(
            json.dumps({
                "user_prompt": r["user_prompt"],
                "assistant_response": r["assistant_response"],
                "corrected_response": r["corrected_response"],
                "rating": r["rating"],
                "model_id": r["model_id"],
            }, ensure_ascii=False) + "\n"
            for r in rows
        )

    def import_corpus(self, jsonl: str) -> dict[str, int]:
        """Merge JSONL feedback. Skips rows already present verbatim."""
        added = skipped = invalid = 0
        for line in jsonl.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                prompt = (item.get("user_prompt") or "").strip()
                answer = (item.get("assistant_response") or "").strip()
                corrected = (item.get("corrected_response") or "") or None
                if not prompt or not answer:
                    raise ValueError("missing prompt or answer")
            except Exception:
                invalid += 1
                continue
            exists = self.execute(
                "SELECT 1 FROM feedback WHERE user_prompt = ? AND assistant_response = ? "
                "AND IFNULL(corrected_response, '') = ?",
                (prompt, answer, corrected or ""),
            ).fetchone()
            if exists:
                skipped += 1
                continue
            self.execute(
                """INSERT INTO feedback
                   (user_prompt, assistant_response, corrected_response, rating,
                    approved_for_training, model_id)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (prompt, answer, corrected, int(item.get("rating") or 0), item.get("model_id")),
            )
            added += 1
        self.commit()
        return {"added": added, "skipped": skipped, "invalid": invalid}

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


def render_ui(page: str = "chat") -> str:
    """Return the HTML the browser receives, with the build marker substituted.

    The page constants are raw strings on purpose. In a normal Python string, a
    `\\n` written inside embedded JavaScript becomes a real line break, which
    splits a JS string literal across two lines and makes the whole <script>
    fail to parse. The page then renders but nothing works: the status panel
    sits on its hardcoded "Starting..." text and the buttons do nothing.
    """
    source = FEEDBACK_PAGE if page == "feedback" else HTML_PAGE
    return source.replace("{{UI_BUILD}}", UI_BUILD)


def check_ui_syntax() -> list[str]:
    """Cheap structural check on every rendered <script>, no Node required.

    Catches the failure above by finding string literals broken by a real
    newline. Returns a list of problems; empty means the scripts are well formed.
    """
    import re as _re

    problems = []
    for page in ("chat", "feedback"):
        script = _re.search(r"<script>(.*?)</script>", render_ui(page), _re.S)
        if not script:
            problems.append(f"{page} page: no <script> block found")
            continue
        for lineno, line in enumerate(script.group(1).splitlines(), 1):
            stripped = _re.sub(r"\\.", "", line)          # drop escaped chars
            stripped = _re.sub(r"//.*$", "", stripped)     # drop line comments
            for quote in ('"', "'"):
                if stripped.count(quote) % 2:
                    problems.append(
                        f"{page} page line {lineno}: unterminated {quote} string: {line.strip()[:70]}"
                    )
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


def model_slug(model_id: str) -> str:
    """Filesystem-safe single path component for a model id.

    Model ids arrive from request bodies, so this must not be able to escape the
    adapter root. Replacing separators is not enough on its own: dots survive the
    substitution, and a bare ".." would resolve to the parent directory.
    """
    import re as _re
    slug = _re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._-")
    return slug[:120] if slug else "unnamed"


def adapter_dir_for(model_id: str) -> Path:
    """Adapters are per-model.

    A LoRA adapter encodes the shapes of the base model it was trained against,
    so handing one model's adapter to another fails or silently corrupts output.
    Keeping a single shared directory made switching models impossible.
    """
    path = (ADAPTER_ROOT / model_slug(model_id)).resolve()
    root = ADAPTER_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Refusing adapter path outside {root}: {model_id!r}")
    return path


def adapter_ready(model_id_or_path: str | Path) -> bool:
    path = (
        model_id_or_path
        if isinstance(model_id_or_path, Path)
        else adapter_dir_for(model_id_or_path)
    )
    return path.exists() and any(path.iterdir())


def migrate_legacy_adapter(model_id: str) -> None:
    """Adopt a pre-namespacing adapters/latest directory for the given model."""
    target = adapter_dir_for(model_id)
    if target.exists() or not (LEGACY_ADAPTER_DIR.exists() and any(LEGACY_ADAPTER_DIR.iterdir())):
        return
    shutil.copytree(LEGACY_ADAPTER_DIR, target)
    log(f"Adopted legacy adapters/latest for {model_id} -> {target}")
    log("If that adapter was trained on a different model, delete it.", logging.WARNING)


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model_port: int = field(default_factory=lambda: int(os.environ.get("MODEL_PORT", "8080")))
    web_port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8000")))
    train_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_ITERS", "30")))
    train_lr: str = field(default_factory=lambda: os.environ.get("TRAIN_LR", "1e-4"))
    train_seq_len: str = field(default_factory=lambda: os.environ.get("TRAIN_SEQ_LEN", "256"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "128")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))
    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"


class ModelServerManager:
    """Manages the MLX model server with health probes and auto-restart."""

    def __init__(self, model_id: str, model_port: int, adapter_dir: Path | None = None):
        self.model_id = model_id
        self.model_port = model_port
        # Kept only so an explicit override still works; normally derived per model.
        self._adapter_override = adapter_dir
        self.proc: subprocess.Popen | None = None
        self.status = "stopped"
        self.lock = threading.RLock()
        self._server_help = help_cmd("mlx_lm.server")
        self._log_file: Any = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    @property
    def adapter_dir(self) -> Path:
        return self._adapter_override or adapter_dir_for(self.model_id)

    def switch_model(self, model_id: str) -> None:
        """Point the server at a different base model and its own adapter."""
        with self.lock:
            if model_id == self.model_id:
                return
            log(f"Switching model: {self.model_id} -> {model_id}")
            self._adapter_override = None
            self.model_id = model_id
        self.restart()

    def _build_cmd(self, use_adapter: bool) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm.server"]
        if not add_if_supported(cmd, self._server_help, ["--model", "--hf-path", "--mlx-path"], self.model_id):
            cmd.extend(["--model", self.model_id])
        if not add_if_supported(cmd, self._server_help, ["--port"], str(self.model_port)):
            cmd.extend(["--port", str(self.model_port)])
        if use_adapter and adapter_ready(self.adapter_dir):
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
        if adapter_ready(self.adapter_dir):
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

    def _backup_adapter(self, model_id: str) -> Path | None:
        """Backup a model's current adapter before retraining it."""
        adapter_dir = adapter_dir_for(model_id)
        if not adapter_ready(adapter_dir):
            return None
        backup_root = ADAPTER_BACKUP_DIR / model_slug(model_id)
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_root / backup_id
        # Two retrains inside the same second would collide on the timestamp.
        suffix = 1
        while backup_path.exists():
            backup_path = backup_root / f"{backup_id}_{suffix}"
            suffix += 1
        shutil.copytree(adapter_dir, backup_path)
        log(f"Adapter backed up to {backup_path}")
        return backup_path

    def _rollback_adapter(self, model_id: str, backup_path: Path | None) -> None:
        """Restore the pre-training adapter after a failed run."""
        if backup_path is None or not backup_path.exists():
            return
        adapter_dir = adapter_dir_for(model_id)
        try:
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            shutil.copytree(backup_path, adapter_dir)
            log(f"Rolled back {model_id} adapter from {backup_path}", logging.WARNING)
        except Exception as exc:
            log(f"Adapter rollback failed: {exc}", logging.ERROR)

    def resolve_iters(self, example_count: int, requested: int | None = None) -> int:
        """Pick an iteration count.

        mlx_lm.lora at batch size 1 sees one example per iteration, so a fixed 30
        iterations over a 200-example corpus never reaches most of the data. 0 or
        None means auto: roughly four passes, floored and capped for sanity.
        """
        iters = self.config.train_iters if requested is None else requested
        if not iters:
            iters = max(30, min(4 * example_count, 1000))
            log(f"Auto iterations: {iters} for {example_count} examples (~4 passes).")
        elif iters < example_count:
            log(
                f"iters={iters} is below the corpus size ({example_count}); "
                "the model will not see every example even once.",
                logging.WARNING,
            )
        return int(iters)

    def _build_cmd(self, model_id: str, iters: int) -> list[str]:
        adapter_dir = adapter_dir_for(model_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "mlx_lm.lora"]
        if not add_if_supported(cmd, self._lora_help, ["--model", "--hf-path", "--mlx-path"], model_id):
            cmd.extend(["--model", model_id])
        if not add_if_supported(cmd, self._lora_help, ["--train"]):
            cmd.append("--train")
        if not add_if_supported(cmd, self._lora_help, ["--data"], str(SFT_DIR)):
            cmd.extend(["--data", str(SFT_DIR)])
        if not add_if_supported(cmd, self._lora_help, ["--adapter-path", "--adapter"], str(adapter_dir)):
            cmd.extend(["--adapter-path", str(adapter_dir)])

        add_if_supported(cmd, self._lora_help, ["--iters", "--iterations"], str(iters))
        add_if_supported(cmd, self._lora_help, ["--batch-size"], "1")
        add_if_supported(cmd, self._lora_help, ["--learning-rate", "-lr"], self.config.train_lr)
        add_if_supported(cmd, self._lora_help, ["--grad-checkpoint", "--gradient-checkpoint"])
        add_if_supported(cmd, self._lora_help, ["--max-seq-length", "--seq-length", "--max-seq-len"], self.config.train_seq_len)
        return cmd

    def export_feedback(self) -> tuple[int, list[int]]:
        """Write train/valid JSONL. Returns (example count, contributing feedback ids).

        One target per prompt, newest wins. Keying on (prompt, answer) instead
        meant that correcting the same prompt twice kept both answers, training
        the model on two contradictory targets for identical input.
        """
        rows = self.db.execute("""
            SELECT id, user_prompt, assistant_response, corrected_response, rating
            FROM feedback WHERE approved_for_training = 1
            ORDER BY id
        """).fetchall()

        latest: dict[str, dict] = {}
        exported_ids: list[int] = []
        superseded = 0
        for row in rows:
            user_prompt = (row["user_prompt"] or "").strip()
            assistant_response = (row["corrected_response"] or row["assistant_response"] or "").strip()
            if not user_prompt or not assistant_response:
                continue
            key = " ".join(user_prompt.lower().split())
            if key in latest:
                superseded += 1
            # Every row is consumed even when superseded, or it retriggers forever.
            exported_ids.append(row["id"])
            latest[key] = {
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        examples = list(latest.values())
        if not examples:
            return 0, []
        if superseded:
            log(f"Corpus: {superseded} older answer(s) superseded by newer ones for the same prompt.")

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

    def run(
        self,
        trigger: str = "manual",
        target_model: str | None = None,
        switch: bool = False,
        iters: int | None = None,
    ) -> None:
        """Train a LoRA adapter for target_model on the approved feedback corpus.

        target_model defaults to the model currently being served. Passing a
        different one trains the same corpus into a new base model without
        disturbing the running server until switch=True asks for the swap.
        """
        if not self.lock.acquire(blocking=False):
            return

        model_id = target_model or self.config.model
        backup_path: Path | None = None
        run_id: int | None = None
        serving_model = self.model_manager.model_id
        try:
            self.status = {
                "running": True,
                "message": f"Retraining {model_id} (from {trigger})",
                "model": model_id,
            }

            count, exported_ids = self.export_feedback()
            if count == 0:
                self.status = {"running": False, "message": "No approved feedback available for training."}
                return

            resolved_iters = self.resolve_iters(count, iters)
            run_id = self.db.start_run(
                model_id=model_id,
                adapter_path=str(adapter_dir_for(model_id)),
                iters=resolved_iters,
                learning_rate=self.config.train_lr,
                trigger=trigger,
            )
            self.status["run_id"] = run_id

            backup_path = self._backup_adapter(model_id)

            # Training and serving both want the GPU, so free it first.
            self.status["message"] = f"Exported {count} examples. Stopping model server."
            self.model_manager.stop()

            cmd = self._build_cmd(model_id, resolved_iters)
            train_log = LOG_DIR / "train.log"

            self.status["message"] = f"Training LoRA adapter for {model_id}..."

            with open(train_log, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n\n{datetime.now(timezone.utc).isoformat()} "
                    f"run {run_id} model {model_id}\n{' '.join(cmd)}\n"
                )
                lf.flush()

                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    raise RuntimeError(f"Training failed with exit code {proc.returncode}. See logs/train.log.")

            self.db.record_run_examples(run_id, exported_ids)
            self.db.finish_run(run_id, "success", example_count=count)
            self.db.mark_trained(exported_ids)

            if switch and model_id != serving_model:
                self.status["message"] = f"Training complete. Switching to {model_id}."
                self.config.model = model_id
                self.model_manager.model_id = model_id
                self.model_manager._adapter_override = None
            else:
                self.status["message"] = "Training complete. Restarting model server."

            self.model_manager.restart()

            served = self.model_manager.model_id
            self.status = {
                "running": False,
                "message": f"Trained {model_id} on {count} examples. Now serving {served}.",
                "model": served,
                "run_id": run_id,
            }

        except Exception as exc:
            log(f"Retrain failed: {exc}", logging.ERROR)
            self._rollback_adapter(model_id, backup_path)
            if run_id is not None:
                self.db.finish_run(run_id, "failed", error=str(exc))
            self.status = {"running": False, "message": f"Retrain error: {exc}", "run_id": run_id}
            try:
                self.model_manager.start()
            except Exception as restart_exc:
                self.status["message"] += f" Restart error: {restart_exc}"
        finally:
            self.lock.release()


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
   }
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
     padding: 12px 16px;
     border-bottom: 1px solid var(--border);
     display: flex;
     justify-content: space-between;
     gap: 12px;
     align-items: center;
     flex-wrap: wrap;
   }
   #status {
     font-size: 12px;
     color: var(--success);
     white-space: pre-wrap;
     font-family: ui-monospace, monospace;
   }
   #status.error { color: var(--error); }
   #status.warn { color: var(--warn); }
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
     animation: fadeIn 0.2s ease;
   }
   @keyframes fadeIn {
     from { opacity: 0; transform: translateY(4px); }
     to { opacity: 1; transform: translateY(0); }
   }
   .user {
     align-self: flex-end;
     background: var(--accent);
     color: white;
   }
   .assistant {
     align-self: flex-start;
     background: var(--surface);
     border: 1px solid #444;
   }
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
     padding: 12px 16px;
     border-top: 1px solid var(--border);
   }
   input[type=text] {
     flex: 1;
     padding: 10px 12px;
     border-radius: 10px;
     border: 1px solid #444;
     background: var(--surface);
     color: var(--fg);
     outline: none;
   }
   button {
     padding: 10px 14px;
     border-radius: 10px;
     border: 1px solid #555;
     background: #2c2c2c;
     color: var(--fg);
     cursor: pointer;
   }
   button:hover { background: #3a3a3a; }
   button:disabled { opacity: 0.5; cursor: not-allowed; }
   .system-msg {
     align-self: center;
     color: #888;
     font-size: 12px;
     margin: 4px 0;
   }
   .typing {
     align-self: flex-start;
     background: var(--surface);
     border: 1px solid #444;
     padding: 10px 12px;
     border-radius: 12px;
     color: #888;
   }
   .typing::after {
     content: "...";
     animation: blink 1s infinite;
   }
   @keyframes blink {
     0%, 100% { opacity: 1; }
     50% { opacity: 0.3; }
   }
 </style>
</head>
<body>
 <header>
   <div><strong>Local LLM</strong> <span style="color:#666;font-size:11px">build {{UI_BUILD}}</span></div>
   <div id="status">Starting...</div>
   <div style="display:flex;gap:8px">
     <a href="/feedback" style="color:var(--accent);text-decoration:none;font-size:13px;align-self:center">Feedback &amp; training</a>
     <button onclick="retrain()">Retrain</button>
   </div>
 </header>

 <div id="chat"></div>

 <footer>
   <input id="input" type="text" placeholder="Send a message..." autocomplete="off">
   <button id="sendBtn" onclick="send()">Send</button>
 </footer>

 <script>
   const chat = document.getElementById("chat");
   const input = document.getElementById("input");
   const sendBtn = document.getElementById("sendBtn");
   const statusEl = document.getElementById("status");

   input.addEventListener("keydown", function(e) {
     if (e.key === "Enter") send();
   });

   function addMessage(role, text) {
     const div = document.createElement("div");
     div.className = "msg " + role;
     div.textContent = text;
     chat.appendChild(div);
     chat.scrollTop = chat.scrollHeight;
     return div;
   }

   function addSystem(text) {
     const div = document.createElement("div");
     div.className = "system-msg";
     div.textContent = text;
     chat.appendChild(div);
     chat.scrollTop = chat.scrollHeight;
   }

   function addAssistant(userText, botText) {
     addMessage("assistant", botText);

     const bar = document.createElement("div");
     bar.className = "feedback";

     const up = document.createElement("button");
     up.textContent = "👍";
     up.title = "Good answer";
     up.onclick = function() { vote(userText, botText, 1); up.disabled = true; down.disabled = true; };

     const down = document.createElement("button");
     down.textContent = "👎";
     down.title = "Bad answer";
     down.onclick = function() { vote(userText, botText, -1); up.disabled = true; down.disabled = true; };

     const correct = document.createElement("button");
     correct.textContent = "✏️ Correct";
     correct.title = "Provide corrected answer";
     correct.onclick = function() { correctAnswer(userText, botText); };

     bar.appendChild(up);
     bar.appendChild(down);
     bar.appendChild(correct);
     chat.appendChild(bar);
     chat.scrollTop = chat.scrollHeight;
   }

   // Any non-JSON body must surface as a readable message. A bare "Not Found"
   // means the request reached the mlx model server, not this web UI.
   async function fetchJSON(url, options) {
     const res = await fetch(url, options);
     const raw = await res.text();
     if (!raw) return {res: res, data: {}};
     try {
       return {res: res, data: JSON.parse(raw)};
     } catch (err) {
       throw new Error(
         "HTTP " + res.status + " from " + (res.url || url) +
         " returned non-JSON, so this page is probably talking to the model " +
         "server instead of the web UI. Body: " + raw.slice(0, 200)
       );
     }
   }

   async function send() {
     const message = input.value.trim();
     if (!message) return;
     if (message.length > 4000) {
       alert("Message too long (max 4000 chars)");
       return;
     }

     input.value = "";
     sendBtn.disabled = true;
     addMessage("user", message);

     const typing = document.createElement("div");
     typing.className = "typing";
     typing.textContent = "Thinking";
     chat.appendChild(typing);
     chat.scrollTop = chat.scrollHeight;

     try {
       const {res, data} = await fetchJSON("/api/chat", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({message: message})
       });
       typing.remove();

       if (!res.ok || data.error) {
         addMessage("assistant", "Error: " + (data.error || res.statusText));
       } else {
         addAssistant(message, data.answer);
       }
     } catch (err) {
       typing.remove();
       addMessage("assistant", "Error: " + err.message);
     } finally {
       sendBtn.disabled = false;
       input.focus();
     }
   }

   async function vote(userPrompt, assistantResponse, rating) {
     try {
       const {data} = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: rating,
           corrected_response: null
         })
       });
       addSystem(data.status || "Feedback saved.");
     } catch (err) {
       addSystem("Feedback error: " + err.message);
     }
   }

   async function correctAnswer(userPrompt, assistantResponse) {
     const corrected = window.prompt("Corrected answer:", assistantResponse);
     if (corrected === null) return;
     try {
       const {data} = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: 1,
           corrected_response: corrected
         })
       });
       addSystem(data.status || "Correction saved.");
     } catch (err) {
       addSystem("Correction error: " + err.message);
     }
   }

   async function retrain() {
     if (!window.confirm("Start retraining? The model server will be stopped temporarily.")) return;
     try {
       const {data} = await fetchJSON("/api/retrain", {method: "POST"});
       addSystem(data.status || JSON.stringify(data));
     } catch (err) {
       addSystem("Retrain error: " + err.message);
     }
   }

   async function refreshHealth() {
     try {
       const {data} = await fetchJSON("/api/health");
       let msg = "Model: " + (data.model ? data.model.split("/").pop() : "unknown") +
                 " (" + (data.model_status || "unknown") + ")";
       if (data.ui_build && data.ui_build !== "{{UI_BUILD}}") {
         msg = "STALE PAGE: server is build " + data.ui_build +
               ", this tab is {{UI_BUILD}}. Hard-reload.\n" + msg;
       }
       msg += "\nRetrain: " + (data.retrain?.message || "idle");
       if (data.stats) {
         msg += "\nFeedback: " + data.stats.total + " total, " + data.stats.approved
              + " approved, " + data.stats.untrained + " untrained";
       }
       statusEl.textContent = msg;
       statusEl.className = "";
       if (data.model_status?.startsWith("error")) statusEl.className = "error";
       else if (data.model_status === "starting") statusEl.className = "warn";
     } catch (err) {
       statusEl.textContent = "Status unavailable: " + err.message;
       statusEl.className = "error";
     }
   }

   setInterval(refreshHealth, 3000);
   refreshHealth();
 </script>
</body>
</html>
"""

FEEDBACK_PAGE = r"""
<!doctype html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>Feedback & training</title>
 <style>
   :root {
     --bg: #111; --fg: #eee; --accent: #0a84ff; --surface: #1a1a1a;
     --border: #333; --error: #ff6b6b; --warn: #ffd93d; --success: #9be29b;
     --muted: #888;
   }
   * { box-sizing: border-box; }
   body {
     font-family: -apple-system, BlinkMacSystemFont, sans-serif;
     margin: 0; background: var(--bg); color: var(--fg); font-size: 14px;
   }
   header {
     padding: 12px 16px; border-bottom: 1px solid var(--border);
     display: flex; justify-content: space-between; align-items: center;
     gap: 12px; flex-wrap: wrap; position: sticky; top: 0; background: var(--bg); z-index: 5;
   }
   a { color: var(--accent); text-decoration: none; }
   main { padding: 16px; max-width: 1100px; margin: 0 auto; }
   h2 { font-size: 15px; margin: 24px 0 8px; font-weight: 600; }
   h2:first-of-type { margin-top: 0; }
   .card {
     background: var(--surface); border: 1px solid var(--border);
     border-radius: 10px; padding: 14px; margin-bottom: 14px;
   }
   .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
   input[type=text], input[type=number], select {
     padding: 8px 10px; border-radius: 8px; border: 1px solid #444;
     background: #151515; color: var(--fg); outline: none; font-size: 13px;
   }
   input[type=text] { min-width: 240px; }
   button {
     padding: 8px 12px; border-radius: 8px; border: 1px solid #555;
     background: #2c2c2c; color: var(--fg); cursor: pointer; font-size: 13px;
   }
   button:hover { background: #3a3a3a; }
   button:disabled { opacity: 0.5; cursor: not-allowed; }
   button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
   button.primary:hover { background: #0a6fd8; }
   table { width: 100%; border-collapse: collapse; }
   th, td { text-align: left; padding: 8px; border-bottom: 1px solid #262626; vertical-align: top; }
   th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
   td.msg { max-width: 340px; }
   .prompt { color: var(--fg); }
   .answer { color: var(--muted); margin-top: 4px; white-space: pre-wrap; }
   .corrected { color: var(--success); margin-top: 4px; white-space: pre-wrap; }
   .pill {
     display: inline-block; padding: 2px 7px; border-radius: 999px;
     font-size: 11px; border: 1px solid #444; margin: 2px 3px 2px 0; white-space: nowrap;
   }
   .pill.ok { border-color: #2f6b2f; color: var(--success); }
   .pill.no { border-color: #6b2f2f; color: var(--error); }
   .pill.wait { border-color: #6b5f2f; color: var(--warn); }
   .muted { color: var(--muted); }
   .stat { display: inline-block; margin-right: 18px; }
   .stat b { font-size: 18px; display: block; font-weight: 600; }
   .stat span { color: var(--muted); font-size: 12px; }
   .banner { padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; display: none; }
   .banner.show { display: block; }
   .banner.info { background: #10233a; border: 1px solid #1d4a7a; }
   .banner.bad { background: #2c1414; border: 1px solid #6b2f2f; color: var(--error); }
   .hint { color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.5; }
   details summary { cursor: pointer; color: var(--accent); font-size: 13px; }
   .empty { color: var(--muted); padding: 18px; text-align: center; }
 </style>
</head>
<body>
 <header>
   <div><strong>Feedback &amp; training</strong> <span class="muted" style="font-size:11px">build {{UI_BUILD}}</span></div>
   <div id="serving" class="muted">loading...</div>
   <a href="/">Back to chat</a>
 </header>

 <main>
   <div id="banner" class="banner"></div>

   <div class="card">
     <div id="stats"><span class="muted">Loading stats...</span></div>
   </div>

   <h2>Train another model on this feedback</h2>
   <div class="card">
     <div class="row">
       <input id="targetModel" type="text" list="knownModels" placeholder="mlx-community/Llama-3.2-1B-Instruct-4bit">
       <datalist id="knownModels"></datalist>
       <label class="muted">iters <input id="iters" type="number" min="1" max="5000" placeholder="auto" style="width:80px"></label>
       <label class="muted"><input id="switchAfter" type="checkbox"> switch to it when done</label>
       <button id="trainBtn" class="primary" onclick="trainOther()">Train this model</button>
     </div>
     <div class="hint">
       Trains the whole approved corpus below into the model you name. Adapters are
       stored per model, so training a new one never touches the adapter you are
       already serving. Leave the model box empty to retrain the model currently in
       use, and leave iters empty for about four passes over the corpus.
     </div>
   </div>

   <h2>Move this corpus elsewhere</h2>
   <div class="card">
     <div class="row">
       <a href="/api/corpus/export"><button>Download corpus as JSONL</button></a>
       <button onclick="document.getElementById('importBox').style.display='block'">Import JSONL</button>
     </div>
     <div id="importBox" style="display:none;margin-top:10px">
       <textarea id="importText" rows="6" placeholder='{"user_prompt": "...", "assistant_response": "..."}'
         style="width:100%;background:#151515;color:var(--fg);border:1px solid #444;border-radius:8px;padding:10px;font-family:ui-monospace,monospace;font-size:12px"></textarea>
       <div class="row" style="margin-top:8px">
         <button class="primary" onclick="importCorpus()">Import</button>
         <span class="muted">One JSON object per line. Exact duplicates are skipped.</span>
       </div>
     </div>
     <div class="hint">
       Feedback is the durable asset here, not the adapter. Exporting lets you carry
       it to another machine or rebuild any model from scratch.
     </div>
   </div>

   <h2>Training runs</h2>
   <div class="card" id="runs"><span class="muted">Loading runs...</span></div>

   <h2>Feedback</h2>
   <div class="card">
     <div class="row">
       <input id="search" type="text" placeholder="Search prompts and answers..." onkeydown="if(event.key==='Enter')loadFeedback()">
       <label class="muted"><input id="approvedOnly" type="checkbox" onchange="loadFeedback()"> approved only</label>
       <label class="muted"><input id="untrainedOnly" type="checkbox" onchange="loadFeedback()"> not yet trained into served model</label>
       <label class="muted"><input id="needsFix" type="checkbox" onchange="loadFeedback()"> needs correction</label>
       <button onclick="loadFeedback()">Apply</button>
     </div>
   </div>
   <div class="card" id="feedback"><span class="muted">Loading feedback...</span></div>
 </main>

 <script>
   let servingModel = null;

   function esc(s) {
     return String(s == null ? "" : s)
       .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
   }

   function banner(text, kind) {
     const el = document.getElementById("banner");
     el.textContent = text;
     el.className = "banner show " + (kind || "info");
     if (kind !== "bad") setTimeout(function() { el.className = "banner"; }, 6000);
   }

   async function fetchJSON(url, options) {
     const res = await fetch(url, options);
     const raw = await res.text();
     if (!raw) return {res: res, data: {}};
     try {
       return {res: res, data: JSON.parse(raw)};
     } catch (err) {
       throw new Error(
         "HTTP " + res.status + " from " + (res.url || url) +
         " returned non-JSON, so this page is probably talking to the model " +
         "server instead of the web UI. Body: " + raw.slice(0, 200)
       );
     }
   }

   function shortModel(m) {
     if (!m) return "unknown";
     const parts = String(m).split("/");
     return parts[parts.length - 1];
   }

   function when(ts) {
     if (!ts) return "";
     return String(ts).replace("T", " ").slice(0, 19);
   }

   async function loadHealth() {
     try {
       const {data} = await fetchJSON("/api/health");
       servingModel = data.model || null;
       document.getElementById("serving").textContent =
         "serving " + shortModel(servingModel) + " (" + (data.model_status || "?") + ")";
       const s = data.stats || {};
       document.getElementById("stats").innerHTML =
         '<div class="stat"><b>' + (s.total || 0) + '</b><span>total</span></div>' +
         '<div class="stat"><b>' + (s.approved || 0) + '</b><span>approved for training</span></div>' +
         '<div class="stat"><b>' + (s.untrained_for_model || 0) + '</b><span>not yet in ' + esc(shortModel(servingModel)) + '</span></div>' +
         '<div class="stat"><b>' + (s.needs_correction || 0) + '</b><span>need a correction</span></div>' +
         '<div class="stat"><b>' + (s.corrected || 0) + '</b><span>corrected</span></div>';
       const list = document.getElementById("knownModels");
       list.innerHTML = (data.known_models || [])
         .map(function(m) { return '<option value="' + esc(m) + '">'; }).join("");
     } catch (err) {
       banner(err.message, "bad");
     }
   }

   async function loadRuns() {
     const box = document.getElementById("runs");
     try {
       const {data} = await fetchJSON("/api/runs?limit=25");
       const runs = data.runs || [];
       if (!runs.length) {
         box.innerHTML = '<div class="empty">No training runs yet. Approve some feedback, then train.</div>';
         return;
       }
       let html = "<table><tr><th>Run</th><th>Model</th><th>Started</th>" +
                  "<th>Examples</th><th>Status</th><th>Trigger</th><th></th></tr>";
       for (const r of runs) {
         const cls = r.status === "success" ? "ok" : (r.status === "failed" ? "no" : "wait");
         html += "<tr>" +
           "<td>#" + r.id + "</td>" +
           "<td>" + esc(shortModel(r.model_id)) + "</td>" +
           '<td class="muted">' + esc(when(r.started_at)) + "</td>" +
           "<td>" + (r.example_count || 0) + "</td>" +
           '<td><span class="pill ' + cls + '">' + esc(r.status) + "</span></td>" +
           '<td class="muted">' + esc(r.trigger || "") + "</td>" +
           '<td><button onclick="showRun(' + r.id + ')">Examples</button></td>' +
           "</tr>";
         if (r.error) {
           html += '<tr><td colspan="7" class="muted">' + esc(r.error) + "</td></tr>";
         }
         html += '<tr id="run-' + r.id + '" style="display:none"><td colspan="7"></td></tr>';
       }
       box.innerHTML = html + "</table>";
     } catch (err) {
       box.innerHTML = '<span class="muted">' + esc(err.message) + "</span>";
     }
   }

   async function showRun(runId) {
     const row = document.getElementById("run-" + runId);
     if (row.style.display !== "none") { row.style.display = "none"; return; }
     row.style.display = "";
     const cell = row.firstChild;
     cell.textContent = "Loading...";
     try {
       const {data} = await fetchJSON("/api/runs/" + runId);
       const ex = (data.run && data.run.examples) || [];
       if (!ex.length) { cell.innerHTML = '<span class="muted">No examples recorded.</span>'; return; }
       cell.innerHTML = '<div class="muted" style="margin-bottom:6px">' + ex.length +
         " example(s) trained into " + esc(data.run.model_id) + "</div>" +
         ex.map(function(e) {
           const answer = e.corrected_response || e.assistant_response;
           return '<div style="margin-bottom:8px"><div class="prompt">#' + e.id + " " +
                  esc(e.user_prompt) + '</div><div class="answer">' + esc(answer) + "</div></div>";
         }).join("");
     } catch (err) {
       cell.textContent = err.message;
     }
   }

   async function loadFeedback() {
     const box = document.getElementById("feedback");
     const params = new URLSearchParams();
     params.set("limit", "200");
     params.set("with_runs", "true");
     const q = document.getElementById("search").value.trim();
     if (q) params.set("search", q);
     if (document.getElementById("approvedOnly").checked) params.set("approved_only", "true");
     if (document.getElementById("untrainedOnly").checked) params.set("untrained_for_served", "true");
     if (document.getElementById("needsFix").checked) params.set("needs_correction", "true");
     try {
       const {data} = await fetchJSON("/api/feedback?" + params.toString());
       const rows = data.feedback || [];
       window.__lastRows = rows;
       if (!rows.length) {
         box.innerHTML = '<div class="empty">No feedback matches these filters.</div>';
         return;
       }
       let html = "<table><tr><th>#</th><th>Exchange</th><th>Rating</th>" +
                  "<th>Approved</th><th>Trained into</th><th></th></tr>";
       for (const f of rows) {
         const answer = f.corrected_response || f.assistant_response;
         const trained = (f.trained_by || []).filter(function(r) { return r.status === "success"; });
         const badges = trained.length
           ? trained.map(function(r) {
               return '<span class="pill ok" title="run #' + r.id + '">' +
                      esc(shortModel(r.model_id)) + "</span>";
             }).join("")
           : '<span class="pill wait">not trained</span>';
         html += "<tr>" +
           "<td>" + f.id + '<div class="muted" style="font-size:11px">' + esc(when(f.created_at)) + "</div></td>" +
           '<td class="msg"><div class="prompt">' + esc(f.user_prompt) + "</div>" +
             (f.corrected_response
               ? '<div class="corrected">corrected: ' + esc(f.corrected_response) + "</div>"
               : '<div class="answer">' + esc(answer) + "</div>") +
           "</td>" +
           "<td>" + (f.rating > 0 ? "up" : (f.rating < 0 ? "down" : "-")) + "</td>" +
           '<td>' + (f.approved_for_training
               ? '<span class="pill ok">yes</span>'
               : '<span class="pill no">no</span>') + "</td>" +
           "<td>" + badges + "</td>" +
           '<td style="white-space:nowrap">' +
             '<button onclick="toggleApproved(' + f.id + ',' + (f.approved_for_training ? "false" : "true") + ')">' +
               (f.approved_for_training ? "Unapprove" : "Approve") + "</button> " +
             '<button onclick="editAnswer(' + f.id + ')">Edit</button> ' +
             '<button onclick="removeFeedback(' + f.id + ')">Delete</button>' +
           "</td></tr>";
       }
       box.innerHTML = html + "</table>";
     } catch (err) {
       box.innerHTML = '<span class="muted">' + esc(err.message) + "</span>";
     }
   }

   async function toggleApproved(id, approved) {
     try {
       await fetchJSON("/api/feedback/" + id, {
         method: "PATCH",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({approved_for_training: approved})
       });
       await Promise.all([loadFeedback(), loadHealth()]);
     } catch (err) {
       banner(err.message, "bad");
     }
   }

   async function editAnswer(id) {
     const row = (window.__lastRows || []).filter(function(f) { return f.id === id; })[0];
     const current = row ? (row.corrected_response || row.assistant_response) : "";
     const next = window.prompt("Corrected answer for #" + id + ":", current);
     if (next === null) return;
     try {
       const {data} = await fetchJSON("/api/feedback/" + id, {
         method: "PATCH",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({corrected_response: next, approved_for_training: true})
       });
       banner("Saved #" + id + ". Its training history was cleared, so it will be included in the next run.");
       await Promise.all([loadFeedback(), loadHealth(), loadRuns()]);
     } catch (err) {
       banner(err.message, "bad");
     }
   }

   async function importCorpus() {
     const text = document.getElementById("importText").value;
     if (!text.trim()) { banner("Nothing to import.", "bad"); return; }
     try {
       const {data} = await fetchJSON("/api/corpus/import", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({jsonl: text})
       });
       banner("Imported " + data.added + " new, skipped " + data.skipped +
              " duplicate, " + data.invalid + " unparseable.");
       document.getElementById("importText").value = "";
       await Promise.all([loadFeedback(), loadHealth()]);
     } catch (err) {
       banner(err.message, "bad");
     }
   }

   async function removeFeedback(id) {
     if (!window.confirm("Delete feedback #" + id + "? Training history keeps the reference.")) return;
     try {
       await fetchJSON("/api/feedback/" + id, {method: "DELETE"});
       await Promise.all([loadFeedback(), loadHealth()]);
     } catch (err) {
       banner(err.message, "bad");
     }
   }

   async function trainOther() {
     const model = document.getElementById("targetModel").value.trim();
     const iters = parseInt(document.getElementById("iters").value, 10);
     const doSwitch = document.getElementById("switchAfter").checked;
     const label = model || "the model currently in use";
     if (!window.confirm(
         "Train " + label + " on the approved feedback?\n\n" +
         "The model server stops during training" +
         (doSwitch ? " and will come back on the new model." : "."))) return;
     const btn = document.getElementById("trainBtn");
     btn.disabled = true;
     try {
       const body = {switch_after: doSwitch};
       if (model) body.model = model;
       if (iters > 0) body.iters = iters;
       const {data} = await fetchJSON("/api/retrain", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify(body)
       });
       banner(data.detail && data.detail.message ? data.detail.message : (data.status || "Started."));
     } catch (err) {
       banner(err.message, "bad");
     } finally {
       btn.disabled = false;
     }
   }

   async function refreshAll() {
     await loadHealth();
     await loadRuns();
   }

   loadHealth();
   loadRuns();
   loadFeedback();
   setInterval(refreshAll, 5000);
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
RetrainRequest: Any = None
FeedbackUpdate: Any = None
CorpusImport: Any = None


def _define_api_models() -> None:
    """Define the Pydantic models in the module namespace.

    This module uses `from __future__ import annotations`, so every parameter
    annotation is a string at runtime. FastAPI resolves those strings against the
    endpoint function's __globals__, which is the module namespace. Models defined
    inside create_app() are invisible there, so FastAPI silently falls back to
    treating the body parameter as a query parameter and every POST returns 422.
    """
    global ChatRequest, FeedbackRequest, ChatResponse, RetrainRequest
    global FeedbackUpdate, CorpusImport
    if ChatRequest is not None:
        return
    from pydantic import BaseModel, Field

    class ChatRequest(BaseModel):  # noqa: F811
        message: str = Field(..., min_length=1, max_length=4000)

    class FeedbackRequest(BaseModel):  # noqa: F811
        user_prompt: str = Field(..., min_length=1)
        assistant_response: str = Field(..., min_length=1)
        rating: int = Field(0, ge=-1, le=1)
        corrected_response: str | None = None

    class ChatResponse(BaseModel):  # noqa: F811
        answer: str

    class RetrainRequest(BaseModel):  # noqa: F811
        model: str | None = Field(None, max_length=200)
        iters: int | None = Field(None, ge=1, le=5000)
        switch_after: bool = False

    class FeedbackUpdate(BaseModel):  # noqa: F811
        approved_for_training: bool | None = None
        corrected_response: str | None = Field(None, max_length=8000)
        rating: int | None = Field(None, ge=-1, le=1)

    class CorpusImport(BaseModel):  # noqa: F811
        jsonl: str = Field(..., max_length=5_000_000)


def create_app(config: Config, db: Database, model_manager: ModelServerManager, retrain_manager: RetrainManager):
    """Create and configure the FastAPI application with Pydantic validation."""
    import httpx
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    _define_api_models()

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
        return HTMLResponse(content=render_ui("chat"))

    @app.get("/feedback", response_class=HTMLResponse)
    def feedback_page():
        return HTMLResponse(content=render_ui("feedback"))

    @app.get("/api/health")
    async def health():
        model_healthy = await model_manager.health_probe() if model_manager.is_alive() else False
        served = model_manager.model_id
        stats = db.get_stats()
        stats["untrained_for_model"] = db.get_untrained_count(served)
        return {
            "ui_build": UI_BUILD,
            "web_port": config.web_port,
            "model_port": config.model_port,
            "model": served,
            "adapter_dir": str(model_manager.adapter_dir),
            "adapter_present": adapter_ready(model_manager.adapter_dir),
            "known_models": db.known_models(),
            "model_status": model_manager.status,
            "model_process_alive": model_manager.is_alive(),
            "model_healthy": model_healthy,
            "retrain": retrain_manager.status,
            "stats": stats,
        }

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, ge=1, le=200)):
        return {"runs": db.list_runs(limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int):
        run = db.get_run(run_id)
        if run is None:
            return JSONResponse(content={"error": f"No run {run_id}"}, status_code=404)
        return {"run": run}

    @app.get("/api/models")
    def list_models():
        return {
            "serving": model_manager.model_id,
            "known": db.known_models(),
            "adapters": sorted(
                p.name for p in ADAPTER_ROOT.iterdir()
                if p.is_dir() and p.name != "backups" and any(p.iterdir())
            ) if ADAPTER_ROOT.exists() else [],
        }

    @app.get("/api/feedback")
    def list_feedback(
        limit: int = Query(50, ge=1, le=500),
        approved_only: bool = False,
        search: str | None = Query(None, max_length=100),
        untrained_for: str | None = Query(None, max_length=200),
        untrained_for_served: bool = False,
        needs_correction: bool = False,
        with_runs: bool = False,
    ):
        target = model_manager.model_id if untrained_for_served else untrained_for
        return {
            "feedback": db.list_feedback(
                limit=limit,
                approved_only=approved_only,
                search=search,
                untrained_for=target,
                needs_correction=needs_correction,
                with_runs=with_runs,
            )
        }

    @app.delete("/api/feedback/{feedback_id}")
    def delete_feedback(feedback_id: int):
        success = db.delete_feedback(feedback_id)
        return {"deleted": success}

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        start_time = time.time()
        if retrain_manager.status.get("running"):
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, "retraining")
            return JSONResponse(content={"error": "Retraining in progress. Please wait."}, status_code=503)

        if model_manager.status != "ready":
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, "model not ready")
            return JSONResponse(
                content={"error": f"Model not ready. Status: {model_manager.status}"},
                status_code=503,
            )

        url = f"http://127.0.0.1:{config.model_port}/v1/chat/completions"
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": request.message},
            ],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": config.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    fallback = dict(payload)
                    fallback.pop("max_tokens", None)
                    response = await client.post(url, json=fallback)

                if response.status_code != 200:
                    db.log_metric("chat", (time.time() - start_time) * 1000, 502)
                    return JSONResponse(content={"error": response.text}, status_code=502)

                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                db.log_metric("chat", (time.time() - start_time) * 1000, 200)
                return ChatResponse(answer=answer)

        except Exception as exc:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, str(exc))
            return JSONResponse(content={"error": str(exc)}, status_code=503)

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """Streaming chat endpoint using Server-Sent Events."""
        if retrain_manager.status.get("running"):
            return JSONResponse(content={"error": "Retraining in progress."}, status_code=503)
        if model_manager.status != "ready":
            return JSONResponse(
                content={"error": f"Model not ready. Status: {model_manager.status}"},
                status_code=503,
            )

        url = f"http://127.0.0.1:{config.model_port}/v1/chat/completions"
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": request.message},
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": config.max_tokens,
        }

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async with httpx.AsyncClient(timeout=600) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            body = (await response.aread()).decode("utf-8", "replace")
                            yield f"data: {json.dumps({'error': body})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                chunk = line[6:]
                                if chunk == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                    break
                                try:
                                    data = json.loads(chunk)
                                    delta = data["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield f"data: {json.dumps({'token': delta})}\n\n"
                                except Exception:
                                    continue
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

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
            untrained = db.get_untrained_count(model_manager.model_id)
            if untrained >= config.auto_retrain_threshold and not retrain_manager.status.get("running"):
                log(f"Auto-retrain triggered: {untrained} approved feedback items")
                threading.Thread(target=retrain_manager.run, kwargs={"trigger": "auto"}, daemon=True).start()

        return {
            "status": "feedback saved",
            "approved_for_training": bool(approved),
            "session_id": session_id,
        }

    @app.patch("/api/feedback/{feedback_id}")
    def update_feedback(feedback_id: int, request: FeedbackUpdate):
        updated = db.update_feedback(
            feedback_id,
            approved=request.approved_for_training,
            corrected_response=request.corrected_response,
            rating=request.rating,
        )
        if updated is None:
            return JSONResponse(content={"error": f"No feedback {feedback_id}"}, status_code=404)
        return {"feedback": updated}

    @app.get("/api/corpus/export")
    def export_corpus():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            db.export_corpus(),
            headers={"Content-Disposition": 'attachment; filename="feedback_corpus.jsonl"'},
        )

    @app.post("/api/corpus/import")
    def import_corpus(request: CorpusImport):
        result = db.import_corpus(request.jsonl)
        return {"status": "imported", **result}

    @app.post("/api/retrain")
    def retrain(request: RetrainRequest | None = None):
        if retrain_manager.status.get("running"):
            return {"status": "already running", "detail": retrain_manager.status}

        target = (request.model or "").strip() if request else ""
        target = target or None
        switch_after = bool(request.switch_after) if request else False
        # Per-run, not a mutation of shared config that outlives the request.
        iters = request.iters if request else None

        threading.Thread(
            target=retrain_manager.run,
            kwargs={
                "trigger": "web",
                "target_model": target,
                "switch": switch_after,
                "iters": iters,
            },
            daemon=True,
        ).start()
        label = target or model_manager.model_id
        return {
            "status": "started",
            "model": label,
            "switch_after": switch_after,
            "iters": iters or "auto",
            "detail": {"message": f"Training {label} on the approved feedback."},
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
        db.close()

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS  single copy of the file")
    print("PASS  embedded UI parses and renders")
    print("PASS  database schema round-trips")
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
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "128")))
    parser.add_argument("--auto-retrain-threshold", type=int, default=int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))
    parser.add_argument("--seed-demo", action="store_true")
    parser.add_argument("--retrain-now", action="store_true")
    parser.add_argument("--retrain-model", default=None,
                        help="Train this model on the approved feedback instead of the served one.")
    parser.add_argument("--switch-model", action="store_true",
                        help="With --retrain-model, serve that model once training succeeds.")
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
        seed_demo=args.seed_demo,
        retrain_now=args.retrain_now,
        export_only=args.export_only,
        list_feedback=args.list_feedback,
        export_format=args.export_format,
    )

    config.model_port = get_free_port(config.model_port)
    config.web_port = get_free_port(config.web_port, exclude={config.model_port})

    db = Database(DB_PATH)

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

    migrate_legacy_adapter(config.model)
    model_manager = ModelServerManager(config.model, config.model_port)
    retrain_manager = RetrainManager(db, model_manager, config)

    if config.export_only:
        count, _ = retrain_manager.export_feedback()
        log(f"Exported {count} training examples to {SFT_DIR}")
        if config.export_format == "csv":
            csv_count = export_to_csv(db, DATA_DIR / "feedback_export.csv")
            log(f"Exported {csv_count} rows to {DATA_DIR / 'feedback_export.csv'}")
        db.close()
        return

    app = create_app(config, db, model_manager, retrain_manager)

    def handle_sigterm(signum, frame):
        log("Received SIGTERM, shutting down...")
        model_manager.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    log(f"Model: {config.model}")
    log(f"System prompt: {config.system_prompt[:60]}...")
    log("=" * 62)
    log(f"  OPEN THIS IN YOUR BROWSER:  http://127.0.0.1:{config.web_port}")
    log(f"  Feedback & training:        http://127.0.0.1:{config.web_port}/feedback")
    log(f"  Model backend (not a UI):   http://127.0.0.1:{config.model_port}")
    log("=" * 62)
    log("Ports shift automatically when the preferred one is busy, so use the URL above.")
    log("Starting model server. First run may download the model.")

    try:
        model_manager.start()
    except Exception as exc:
        log(f"Model server failed to start: {exc}", logging.ERROR)
        log("Web UI will still start. Check logs/model_server.log.", logging.WARNING)

    if config.retrain_now or args.retrain_model:
        def delayed_retrain():
            time.sleep(3)
            retrain_manager.run(
                "cli",
                target_model=args.retrain_model,
                switch=args.switch_model,
            )
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
