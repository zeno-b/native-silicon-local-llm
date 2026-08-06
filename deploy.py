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
ADAPTER_DIR = ROOT / "adapters" / "latest"
ADAPTER_BACKUP_DIR = ROOT / "adapters" / "backups"
DB_PATH = DATA_DIR / "feedback.db"

DEFAULT_MODEL = os.environ.get(
    "MODEL_ID",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
)
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant.",
)

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
    for d in [DATA_DIR, SFT_DIR, LOG_DIR, ADAPTER_DIR.parent, ADAPTER_BACKUP_DIR]:
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
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "128")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))
    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"


class ModelServerManager:
    """Manages the MLX model server with health probes and auto-restart."""

    def __init__(self, model_id: str, model_port: int, adapter_dir: Path):
        self.model_id = model_id
        self.model_port = model_port
        self.adapter_dir = adapter_dir
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
                    self.status = "ready"
                    log(f"Model server ready at http://127.0.0.1:{self.model_port}")
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


HTML_PAGE = """
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
   <div><strong>Local LLM</strong></div>
   <div id="status">Starting...</div>
   <button onclick="retrain()">Retrain</button>
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
       const res = await fetch("/api/chat", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({message: message})
       });
       typing.remove();

       const data = await res.json();
       if (!res.ok || data.error) {
         addMessage("assistant", "Error: " + (data.error || res.statusText));
       } else {
         addAssistant(message, data.answer);
       }
     } catch (err) {
       typing.remove();
       addMessage("assistant", "Error: " + err);
     } finally {
       sendBtn.disabled = false;
       input.focus();
     }
   }

   async function vote(userPrompt, assistantResponse, rating) {
     try {
       const res = await fetch("/api/feedback", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: rating,
           corrected_response: null
         })
       });
       const data = await res.json();
       addSystem(data.status || "Feedback saved.");
     } catch (err) {
       addSystem("Feedback error: " + err);
     }
   }

   async function correctAnswer(userPrompt, assistantResponse) {
     const corrected = window.prompt("Corrected answer:", assistantResponse);
     if (corrected === null) return;
     try {
       const res = await fetch("/api/feedback", {
         method: "POST",
         headers: {"Content-Type": "application/json"},
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: 1,
           corrected_response: corrected
         })
       });
       const data = await res.json();
       addSystem(data.status || "Correction saved.");
     } catch (err) {
       addSystem("Correction error: " + err);
     }
   }

   async function retrain() {
     if (!window.confirm("Start retraining? The model server will be stopped temporarily.")) return;
     try {
       const res = await fetch("/api/retrain", {method: "POST"});
       const data = await res.json();
       addSystem(data.status || JSON.stringify(data));
     } catch (err) {
       addSystem("Retrain error: " + err);
     }
   }

   async function refreshHealth() {
     try {
       const res = await fetch("/api/health");
       const data = await res.json();
       let msg = "Model: " + (data.model_status || "unknown");
       msg += "\\nRetrain: " + (data.retrain?.message || "idle");
       if (data.stats) {
         msg += "\\nFeedback: " + data.stats.total + " total, " + data.stats.approved
              + " approved, " + data.stats.untrained + " untrained";
       }
       statusEl.textContent = msg;
       statusEl.className = "";
       if (data.model_status?.startsWith("error")) statusEl.className = "error";
       else if (data.model_status === "starting") statusEl.className = "warn";
     } catch (err) {
       statusEl.textContent = "Status unavailable";
       statusEl.className = "error";
     }
   }

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


def _define_api_models() -> None:
    """Define the Pydantic models in the module namespace.

    This module uses `from __future__ import annotations`, so every parameter
    annotation is a string at runtime. FastAPI resolves those strings against the
    endpoint function's __globals__, which is the module namespace. Models defined
    inside create_app() are invisible there, so FastAPI silently falls back to
    treating the body parameter as a query parameter and every POST returns 422.
    """
    global ChatRequest, FeedbackRequest, ChatResponse
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

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/health")
    async def health():
        model_healthy = await model_manager.health_probe() if model_manager.is_alive() else False
        return {
            "model_status": model_manager.status,
            "model_process_alive": model_manager.is_alive(),
            "model_healthy": model_healthy,
            "retrain": retrain_manager.status,
            "stats": db.get_stats(),
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
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--list-feedback", action="store_true")
    parser.add_argument("--export-format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()

    bootstrap()
    import uvicorn

    ensure_dirs()

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

    model_manager = ModelServerManager(config.model, config.model_port, ADAPTER_DIR)
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
    log(f"Model port: {config.model_port}")
    log(f"Web UI: http://127.0.0.1:{config.web_port}")
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
