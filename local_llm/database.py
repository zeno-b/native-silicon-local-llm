"""SQLite storage: messages, feedback, documents, prompts, tasks, metrics.

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
                    trained_at TIMESTAMP,
                    source TEXT DEFAULT 'button',
                    reviewed INTEGER DEFAULT 0
                )
            """)
            # Migration: older databases predate trained_at.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(feedback)")}
            if "trained_at" not in columns:
                log("Migrating feedback table: adding trained_at column.")
                conn.execute("ALTER TABLE feedback ADD COLUMN trained_at TIMESTAMP")
            # Provenance and curation, so a dataset stays reusable later: where a
            # row came from (button / implicit chat / demo / import) and whether a
            # human has reviewed it.
            if "source" not in columns:
                log("Migrating feedback table: adding source column.")
                conn.execute("ALTER TABLE feedback ADD COLUMN source TEXT DEFAULT 'button'")
            if "reviewed" not in columns:
                log("Migrating feedback table: adding reviewed column.")
                conn.execute("ALTER TABLE feedback ADD COLUMN reviewed INTEGER DEFAULT 0")
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
                CREATE TABLE IF NOT EXISTS conversation_meta (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    pinned INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    name TEXT PRIMARY KEY,
                    body TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    title TEXT,
                    chars INTEGER,
                    chunks INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # FTS5 gives real ranked (BM25) retrieval with no extra dependency and
            # no embedding model, which matters on an 8GB machine where a second
            # model would compete for memory with the LLM itself.
            #
            # Some Python builds (notably several Homebrew and python.org macOS
            # builds) ship a SQLite compiled without FTS5. Rather than disable the
            # knowledge base there, fall back to a plain table plus scoring done
            # in Python: slower on a very large corpus, identical in behaviour for
            # the size a local knowledge base actually reaches.
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks
                    USING fts5(path, chunk)
                """)
                self.fts_enabled = True
                self.search_mode = "fts5"
            except sqlite3.OperationalError as exc:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS doc_chunks_plain (
                        id INTEGER PRIMARY KEY,
                        path TEXT,
                        chunk TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_doc_chunks_plain_path
                    ON doc_chunks_plain(path)
                """)
                self.fts_enabled = True          # the knowledge base still works
                self.search_mode = "fallback"
                log(f"SQLite FTS5 is not in this Python build ({exc}); using the "
                    "built-in ranked search instead. The knowledge base works "
                    "normally.", logging.WARNING)
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
                   (user_prompt, assistant_response, rating, corrected_response,
                    approved_for_training, source, reviewed)
                   VALUES (?, ?, ?, ?, ?, 'demo', 1)""",
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

    def index_document(self, path: str, text: str, title: str = "",
                       chunk_chars: int = 1200, overlap: int = 150) -> int:
        """Store a document as overlapping chunks for retrieval. Returns chunk count."""
        if not getattr(self, "fts_enabled", False):
            return 0
        self.remove_document(path)
        chunks = []
        chunk_chars = max(200, int(chunk_chars or 1200))
        overlap = min(max(0, int(overlap or 0)), chunk_chars // 2)
        step = max(100, chunk_chars - overlap)
        for start in range(0, max(1, len(text)), step):
            piece = text[start:start + chunk_chars].strip()
            if piece:
                chunks.append(piece)
        table = self._chunk_table()
        for piece in chunks:
            self.execute(f"INSERT INTO {table} (path, chunk) VALUES (?, ?)", (path, piece))
        self.execute(
            "INSERT OR REPLACE INTO documents (path, title, chars, chunks) VALUES (?, ?, ?, ?)",
            (path, title or Path(path).name, len(text), len(chunks)))
        self.commit()
        return len(chunks)

    def _chunk_table(self) -> str:
        """Which table holds chunks: the FTS5 virtual table, or the plain
        fallback used when this Python's SQLite lacks FTS5."""
        return "doc_chunks" if getattr(self, "search_mode", "fts5") == "fts5" else "doc_chunks_plain"

    def remove_document(self, path: str) -> None:
        if not getattr(self, "fts_enabled", False):
            return
        self.execute(f"DELETE FROM {self._chunk_table()} WHERE path = ?", (path,))
        self.execute("DELETE FROM documents WHERE path = ?", (path,))
        self.commit()

    def search_documents(self, query: str, limit: int = 5, only: list[str] | None = None) -> list[dict]:
        """BM25-ranked chunk search. Returns [{path, chunk}] best first."""
        if not getattr(self, "fts_enabled", False):
            return []
        # FTS5 treats punctuation as syntax; reduce the query to bare terms and
        # OR them so a natural-language question still matches.
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query or "") if len(t) > 2]
        if not terms:
            return []
        if getattr(self, "search_mode", "fts5") != "fts5":
            return self._search_documents_fallback(terms, limit, only)
        expr = " OR ".join(terms[:12])
        sql = "SELECT path, chunk FROM doc_chunks WHERE doc_chunks MATCH ?"
        params: list[Any] = [expr]
        if only:
            # Scope retrieval to chosen documents ("chat with this document").
            sql += " AND path IN (" + ",".join("?" for _ in only) + ")"
            params.extend(only)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"path": r["path"], "chunk": r["chunk"]} for r in rows]

    def _search_documents_fallback(self, terms: list[str], limit: int,
                                   only: list[str] | None) -> list[dict]:
        """Ranked search without FTS5.

        Scores each chunk the way BM25 broadly does: a term is worth more when it
        is rare across the corpus and appears often in a chunk, with longer chunks
        discounted so a big chunk cannot win on length alone. Candidates are
        narrowed with SQL LIKE first so only plausibly matching rows are scored.
        """
        sql = "SELECT path, chunk FROM doc_chunks_plain"
        params: list[Any] = []
        clauses = []
        if only:
            clauses.append("path IN (" + ",".join("?" for _ in only) + ")")
            params.extend(only)
        if terms:
            like_bits = []
            for term in terms[:12]:
                like_bits.append("chunk LIKE ? ESCAPE '\\'")
                safe = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{safe}%")
            clauses.append("(" + " OR ".join(like_bits) + ")")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Bounded: a local knowledge base is thousands of chunks, not millions.
        sql += " LIMIT 5000"
        try:
            rows = self.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        lowered = [t.lower() for t in terms[:12]]
        docs = [(r["path"], r["chunk"], (r["chunk"] or "").lower()) for r in rows]
        total = len(docs)
        # How many chunks contain each term, for the rarity weighting.
        containing = {t: sum(1 for _, _, low in docs if t in low) or 1 for t in lowered}
        avg_len = sum(len(low) for _, _, low in docs) / total or 1.0
        scored = []
        for path, chunk, low in docs:
            score = 0.0
            for term in lowered:
                freq = low.count(term)
                if not freq:
                    continue
                rarity = math.log(1 + total / containing[term])
                length_penalty = 1.0 + (len(low) / avg_len) * 0.5
                score += rarity * (freq / length_penalty)
            if score > 0:
                scored.append((score, path, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"path": p, "chunk": c} for _, p, c in scored[:limit]]

    def document_stats(self) -> dict:
        if not getattr(self, "fts_enabled", False):
            return {"enabled": False, "documents": 0, "chunks": 0, "items": []}
        rows = self.execute(
            "SELECT path, title, chars, chunks, indexed_at FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
        items = [dict(r) for r in rows]
        return {"enabled": True, "documents": len(items),
                "chunks": sum(int(i["chunks"] or 0) for i in items), "items": items[:50],
                "search_mode": getattr(self, "search_mode", "fts5")}

    def clear_documents(self) -> int:
        if not getattr(self, "fts_enabled", False):
            return 0
        n = self.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        self.execute(f"DELETE FROM {self._chunk_table()}")
        self.execute("DELETE FROM documents")
        self.commit()
        return int(n or 0)

    def search_conversations(self, query: str, limit: int = 30) -> list[dict]:
        """Find conversations containing a phrase, newest first, with a snippet."""
        # Escape LIKE wildcards so searching for "%" or "_" looks for those
        # characters instead of matching every conversation.
        safe = (query or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{safe}%"
        rows = self.execute(
            """SELECT conversation_id, role, content, MAX(created_at) AS created_at
               FROM messages WHERE content LIKE ? ESCAPE '\\'
               GROUP BY conversation_id ORDER BY created_at DESC LIMIT ?""",
            (like, limit)).fetchall()
        out = []
        for r in rows:
            content = r["content"] or ""
            idx = content.lower().find(query.lower())
            start = max(0, idx - 60)
            snippet = ("..." if start else "") + content[start:start + 200].strip()
            out.append({"conversation_id": r["conversation_id"], "role": r["role"],
                        "snippet": snippet, "created_at": r["created_at"]})
        return out

    def export_conversation(self, conversation_id: str, fmt: str = "markdown") -> str:
        rows = self.get_messages(conversation_id, limit=10000)
        if not rows:
            raise ValueError(f"conversation {conversation_id!r} has no messages to export")
        if fmt == "json":
            return json.dumps([dict(r) for r in rows], indent=2, default=str)
        lines = [f"# Conversation {conversation_id}", ""]
        for r in rows:
            who = "You" if r["role"] == "user" else "Assistant"
            lines.append(f"## {who}")
            lines.append((r["content"] or "").strip())
            lines.append("")
        return "\n".join(lines)

    def save_prompt(self, name: str, body: str) -> None:
        if not (name or "").strip():
            raise ValueError("a prompt needs a name")
        if not (body or "").strip():
            raise ValueError("a prompt needs a body")
        self.execute(
            "INSERT OR REPLACE INTO prompts (name, body, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (name.strip(), body))
        self.commit()

    def list_prompts(self) -> list[dict]:
        return [dict(r) for r in self.execute(
            "SELECT name, body, updated_at FROM prompts ORDER BY name ASC").fetchall()]

    def delete_prompt(self, name: str) -> bool:
        cur = self.execute("DELETE FROM prompts WHERE name = ?", (name,))
        self.commit()
        return cur.rowcount > 0

    def drop_last_exchange(self, conversation_id: str) -> str | None:
        """Remove the last assistant reply (and return the user prompt that led to
        it) so the turn can be regenerated."""
        rows = self.get_messages(conversation_id, limit=50)
        if not rows:
            return None
        last_user = None
        to_delete = []
        for r in reversed(rows):
            if r["role"] == "assistant" and not to_delete:
                to_delete.append(r["id"])
                continue
            if r["role"] == "user":
                last_user = r["content"]
                break
        for mid in to_delete:
            self.execute("DELETE FROM messages WHERE id = ?", (mid,))
        self.commit()
        return last_user

    def export_backup(self) -> dict:
        """Everything the user created: conversations, prompts, feedback, docs.

        Plain JSON so a backup stays readable and restorable even if the schema
        moves on. Model weights and adapters are not included; those are large
        and re-downloadable.
        """
        def rows(sql: str) -> list[dict]:
            try:
                return [dict(r) for r in self.execute(sql).fetchall()]
            except sqlite3.OperationalError:
                return []
        data = {
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "messages": rows("SELECT * FROM messages ORDER BY id ASC"),
            "conversation_meta": rows("SELECT * FROM conversation_meta"),
            "prompts": rows("SELECT * FROM prompts"),
            "feedback": rows("SELECT * FROM feedback ORDER BY id ASC"),
            "documents": rows("SELECT * FROM documents"),
        }
        if getattr(self, "fts_enabled", False):
            data["doc_chunks"] = rows(f"SELECT path, chunk FROM {self._chunk_table()}")
        return data

    def import_backup(self, data: dict) -> dict:
        """Merge a backup back in. Additive: existing rows are left alone and
        conversations are keyed by their original ids."""
        counts = {"messages": 0, "prompts": 0, "feedback": 0, "documents": 0}
        if not isinstance(data, dict):
            raise ValueError("backup must be a JSON object; got "
                             f"{type(data).__name__}")
        version = data.get("version")
        if version is not None and not isinstance(version, int):
            raise ValueError("backup 'version' must be a number")
        if version is not None and version > 1:
            raise ValueError(f"backup version {version} is newer than this app understands (1)")

        def section(key: str) -> list:
            value = data.get(key, [])
            if not isinstance(value, list):
                log(f"backup section {key!r} is not a list; skipping", logging.WARNING)
                return []
            return [row for row in value if isinstance(row, dict)]
        for msg in section("messages"):
            try:
                self.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
                    (msg.get("conversation_id"), msg.get("role"), msg.get("content"),
                     msg.get("created_at")))
                counts["messages"] += 1
            except Exception:
                continue
        for meta in section("conversation_meta"):
            try:
                if meta.get("title"):
                    self.set_conversation_title(meta["conversation_id"], meta["title"])
                if meta.get("pinned"):
                    self.set_conversation_pinned(meta["conversation_id"], True)
            except Exception:
                continue
        for pr in section("prompts"):
            try:
                self.save_prompt(pr.get("name", ""), pr.get("body", ""))
                counts["prompts"] += 1
            except Exception:
                continue
        for fb in section("feedback"):
            try:
                self.record_feedback(
                    fb.get("user_prompt", ""), fb.get("assistant_response", ""),
                    int(fb.get("rating") or 0), int(fb.get("approved_for_training") or 0),
                    fb.get("corrected_response"), fb.get("session_id"), fb.get("model_id"),
                    fb.get("source") or "import")
                counts["feedback"] += 1
            except Exception:
                continue
        by_path: dict[str, list[str]] = {}
        for ch in section("doc_chunks"):
            by_path.setdefault(ch.get("path", ""), []).append(ch.get("chunk", ""))
        for path, chunks in by_path.items():
            if not path:
                continue
            try:
                self.index_document(path, "\n\n".join(chunks), Path(path).name)
                counts["documents"] += 1
            except Exception:
                continue
        self.commit()
        return counts

    def fork_conversation(self, conversation_id: str, upto_message_id: int | None = None) -> str:
        """Copy a conversation (optionally only up to a message) into a new one,
        so you can explore a different direction without losing the original."""
        existing = self.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?",
            (conversation_id,)).fetchone()
        if not existing or not existing["c"]:
            raise ValueError(f"conversation {conversation_id!r} has no messages to fork")
        new_id = f"fork-{uuid.uuid4().hex[:10]}"
        sql = "SELECT role, content FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if upto_message_id:
            sql += " AND id <= ?"
            params.append(upto_message_id)
        sql += " ORDER BY id ASC"
        for row in self.execute(sql, tuple(params)).fetchall():
            self.add_message(new_id, row["role"], row["content"])
        base = self.conversation_title(conversation_id) or conversation_id[:8]
        self.set_conversation_title(new_id, f"{base} (fork)"[:120])
        return new_id

    def record_feedback(self, user_prompt: str, assistant_response: str, rating: int,
                        approved: int, corrected: str | None = None,
                        session_id: str | None = None, model_id: str | None = None,
                        source: str = "button") -> None:
        """Insert one feedback row (used by the button and by implicit chat feedback)."""
        self.execute(
            """INSERT INTO feedback
               (user_prompt, assistant_response, rating, corrected_response,
                approved_for_training, session_id, model_id, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_prompt, assistant_response, rating, corrected, approved,
             session_id or "implicit", model_id, source),
        )
        self.commit()

    def dataset_stats(self) -> dict:
        """Counts that describe how reusable the collected dataset is."""
        def scalar(sql: str, params: tuple = ()) -> int:
            row = self.execute(sql, params).fetchone()
            return int((row[0] if row else 0) or 0)
        total = scalar("SELECT COUNT(*) FROM feedback")
        approved = scalar("SELECT COUNT(*) FROM feedback WHERE approved_for_training = 1")
        rejected = scalar("SELECT COUNT(*) FROM feedback WHERE rating < 0")
        reviewed = scalar("SELECT COUNT(*) FROM feedback WHERE reviewed = 1")
        corrected = scalar("SELECT COUNT(*) FROM feedback WHERE corrected_response IS NOT NULL AND corrected_response != ''")
        by_source = {}
        for row in self.execute("SELECT COALESCE(source,'button') AS s, COUNT(*) AS c FROM feedback GROUP BY s").fetchall():
            by_source[row["s"]] = int(row["c"])
        # Preference pairs available: prompts with a corrected answer, plus prompts
        # that have both an approved and a rejected answer.
        pref = scalar(
            "SELECT COUNT(*) FROM feedback WHERE corrected_response IS NOT NULL "
            "AND corrected_response != '' AND corrected_response != assistant_response")
        pref += scalar(
            "SELECT COUNT(DISTINCT a.user_prompt) FROM feedback a "
            "JOIN feedback b ON a.user_prompt = b.user_prompt "
            "WHERE a.approved_for_training = 1 AND b.rating < 0")
        return {"total": total, "approved": approved, "rejected": rejected,
                "reviewed": reviewed, "corrected": corrected,
                "preference_pairs": pref, "by_source": by_source}

    def set_reviewed(self, feedback_id: int, reviewed: bool) -> bool:
        cur = self.execute("UPDATE feedback SET reviewed = ? WHERE id = ?",
                           (1 if reviewed else 0, feedback_id))
        self.commit()
        return cur.rowcount > 0

    def export_rows(self, approved_only: bool, reviewed_only: bool) -> list[dict]:
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []
        if approved_only:
            sql += " AND approved_for_training = 1"
        if reviewed_only:
            sql += " AND reviewed = 1"
        sql += " ORDER BY id ASC"
        return [dict(r) for r in self.execute(sql, tuple(params)).fetchall()]

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
        out = []
        for row in rows:
            item = dict(row)
            cid = item["conversation_id"]
            meta = self.execute(
                "SELECT title, pinned FROM conversation_meta WHERE conversation_id = ?",
                (cid,)).fetchone()
            custom = (meta["title"] if meta else None)
            item["pinned"] = bool(meta["pinned"]) if meta else False
            if custom:
                item["title"] = custom
            else:
                # Fall back to the first user message as a readable title.
                first = self.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' "
                    "ORDER BY id ASC LIMIT 1", (cid,)).fetchone()
                item["title"] = ((first["content"] or "").strip()[:90] if first else "") or "(no messages)"
            out.append(item)
        # Pinned conversations float to the top, newest first within each group.
        out.sort(key=lambda c: (not c["pinned"], c.get("last_at") or ""), reverse=False)
        out.sort(key=lambda c: c["pinned"], reverse=True)
        return out

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        self.execute(
            "INSERT INTO conversation_meta (conversation_id, title, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(conversation_id) DO UPDATE SET title = excluded.title, "
            "updated_at = CURRENT_TIMESTAMP",
            (conversation_id, title.strip()[:120]))
        self.commit()

    def set_conversation_pinned(self, conversation_id: str, pinned: bool) -> None:
        self.execute(
            "INSERT INTO conversation_meta (conversation_id, pinned, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(conversation_id) DO UPDATE SET pinned = excluded.pinned, "
            "updated_at = CURRENT_TIMESTAMP",
            (conversation_id, 1 if pinned else 0))
        self.commit()

    def conversation_title(self, conversation_id: str) -> str | None:
        row = self.execute("SELECT title FROM conversation_meta WHERE conversation_id = ?",
                           (conversation_id,)).fetchone()
        return row["title"] if row else None

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



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Database',
]
