"""SQLite storage: messages, feedback, documents, prompts, tasks, metrics.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403


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
                    reviewed INTEGER DEFAULT 0,
                    pending_approval INTEGER DEFAULT 0
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
            # A non-admin's rating queues a row instead of approving it: the LoRA
            # adapter is shared by every user, so unreviewed submissions must not
            # be able to steer it. See the /api/feedback handler.
            if "pending_approval" not in columns:
                log("Migrating feedback table: adding pending_approval column.")
                conn.execute(
                    "ALTER TABLE feedback ADD COLUMN pending_approval INTEGER DEFAULT 0")
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
            # The active task and its artifact, one row per conversation. Kept
            # OUT of the messages table on purpose: this is state, not
            # transcript, and it has to survive exactly the trimming that
            # removes the messages it was derived from.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_task_state (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    state TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    body TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, name)
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
            # Memories are per-user: uniqueness is (user_id, key), not key alone,
            # so one user's note can never overwrite or read another's. Fresh
            # databases get this shape directly; the migration below upgrades an
            # older single-user memories table (key as the sole primary key).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    conversation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, key)
                )
            """)

            self._migrate_multiuser(conn)
            conn.commit()

    # ---------------------------------------------------------- migrations --

    def _migrate_multiuser(self, conn: sqlite3.Connection) -> None:
        """Additive, idempotent migration to the multi-user / multi-node schema.

        Adds the users/sessions/imports/routing_events tables and stamps a
        `user_id` owner column (backfilled to the sentinel local user) onto every
        user-owned table. Safe to run on every startup and on any older database:
        it only creates what is missing and only backfills NULLs.
        """
        sentinel = SENTINEL_LOCAL_USER

        # --- Auth: users and sessions ------------------------------------- #
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                source TEXT NOT NULL DEFAULT 'local',
                email TEXT,
                display_name TEXT,
                oidc_subject TEXT,
                disabled INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_oidc ON users(oidc_subject)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip TEXT,
                user_agent TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at)")

        # --- Claude-history import jobs ------------------------------------ #
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT,
                size_bytes INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                progress INTEGER NOT NULL DEFAULT 0,
                counts TEXT,
                warnings TEXT,
                error TEXT,
                artifacts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP
            )
        """)
        # `artifacts` (the exact conversations/docs an import created, so removal
        # is precise) was added after the table; add it to older imports tables.
        _imp_cols = {row["name"] for row in conn.execute("PRAGMA table_info(imports)")}
        if "artifacts" not in _imp_cols:
            conn.execute("ALTER TABLE imports ADD COLUMN artifacts TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_user ON imports(user_id, created_at DESC)")

        # --- Node routing observability ----------------------------------- #
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_events (
                id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                correlation_id TEXT,
                task_id TEXT,
                user_id TEXT,
                conversation_id TEXT,
                kind TEXT,
                requested_model TEXT,
                selected_model TEXT,
                selected_node TEXT,
                reason TEXT,
                candidates TEXT,
                status TEXT,
                attempt INTEGER DEFAULT 0,
                duration_ms REAL,
                error TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_created ON routing_events(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_task ON routing_events(task_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_node ON routing_events(selected_node, created_at DESC)")

        # --- Agent profiles (named capability sets) ----------------------- #
        # An agent is a saved profile: a name, a description, and the set of
        # capabilities it is allowed to use (JSON list of capability keys, see
        # config.CAPABILITY_GROUPS). The runtime turns capabilities into a tool
        # allowlist. Shared across the install (admin-managed), like models.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                capabilities TEXT NOT NULL DEFAULT '[]',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)")

        # --- Owner column on every user-owned table ----------------------- #
        # ADD COLUMN ... DEFAULT is constant so existing rows read as the
        # sentinel owner; the explicit UPDATE covers any pre-existing NULLs.
        for table in ("messages", "conversation_meta", "feedback", "tasks",
                      "tool_calls", "metrics", "documents", "task_runs"):
            cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not cols:
                continue  # table absent on this build
            if "user_id" not in cols:
                # Only announce it as a migration when there is real data to
                # stamp; a brand-new table is just being born with the column.
                existing = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
                if existing and existing["c"]:
                    log(f"Migrating {table}: adding user_id owner column "
                        f"({existing['c']} rows -> owner '{sentinel}').")
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT '{sentinel}'")
                conn.execute(
                    f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
                    (sentinel,))
        for table, col in (("messages", "user_id"), ("feedback", "user_id"),
                           ("tasks", "user_id"), ("documents", "user_id")):
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_user ON {table}({col})")
        # The conversation list groups a user's messages by conversation and
        # orders by recency. Covering all three columns lets SQLite walk the
        # index instead of building a temporary b-tree over every message the
        # user has ever sent, which is the one query that grows without bound.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_conv "
                     "ON messages(user_id, conversation_id, created_at)")

        # --- memories: migrate the old (key-only PK) table if present ------ #
        mem_cols = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        if mem_cols and "user_id" not in mem_cols:
            log("Migrating memories to per-user keys ((user_id, key) unique).")
            conn.execute("ALTER TABLE memories RENAME TO memories_legacy")
            conn.execute("""
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    conversation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, key)
                )
            """)
            conn.execute(
                "INSERT INTO memories (user_id, key, value, conversation_id, created_at, updated_at) "
                "SELECT ?, key, value, conversation_id, created_at, updated_at FROM memories_legacy",
                (sentinel,))
            conn.execute("DROP TABLE memories_legacy")

        # --- prompts: same story, name was the sole PRIMARY KEY ------------ #
        # A global prompt store means every user reads, overwrites and deletes
        # everyone else's saved prompts, and two users cannot both keep a
        # "code-review". Rebuild with (user_id, name) unique.
        prompt_cols = {row["name"] for row in conn.execute("PRAGMA table_info(prompts)")}
        if prompt_cols and "user_id" not in prompt_cols:
            log("Migrating prompts to per-user names ((user_id, name) unique).")
            conn.execute("ALTER TABLE prompts RENAME TO prompts_legacy")
            conn.execute("""
                CREATE TABLE prompts (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    body TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, name)
                )
            """)
            conn.execute(
                "INSERT INTO prompts (user_id, name, body, updated_at) "
                "SELECT ?, name, body, updated_at FROM prompts_legacy", (sentinel,))
            conn.execute("DROP TABLE prompts_legacy")

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
                pass      # already closed, or closed from its owning thread
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
        user_id: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM feedback WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
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
        """Feedback counters for the admin status bar.

        One pass with conditional sums rather than six COUNT queries: /api/health
        polls this every few seconds, and six separate scans of the same table is
        six times the work for the same numbers. SUM() over an empty table gives
        NULL, hence the `or 0`.
        """
        row = self.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(approved_for_training = 1) AS approved,"
            " SUM(approved_for_training = 1 AND trained_at IS NULL) AS untrained,"
            " SUM(rating > 0) AS positive,"
            " SUM(rating < 0) AS negative,"
            " SUM(corrected_response IS NOT NULL) AS corrected"
            " FROM feedback").fetchone()
        return {key: int(row[key] or 0) for key in
                ("total", "approved", "untrained", "positive", "negative", "corrected")}

    def index_document(self, path: str, text: str, title: str = "",
                       chunk_chars: int = 1200, overlap: int = 150,
                       user_id: str | None = None) -> int:
        """Store a document as overlapping chunks for retrieval. Returns chunk count.

        The owner (user_id, defaulting to the sentinel local user) is recorded on
        the documents row so retrieval can be scoped: a user sees their own
        documents plus anything owned by the shared owner.
        """
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
            "INSERT OR REPLACE INTO documents (path, title, chars, chunks, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (path, title or Path(path).name, len(text), len(chunks),
             user_id or SENTINEL_LOCAL_USER))
        self.commit()
        return len(chunks)

    def _owned_paths_clause(self, user_id: str | None) -> tuple[str, list[Any]]:
        """SQL fragment restricting chunk paths to those a user may retrieve
        (their own documents plus shared ones). Empty when user_id is None."""
        if user_id is None:
            return "", []
        return (" AND path IN (SELECT path FROM documents WHERE user_id IN (?, ?))",
                [user_id, SHARED_OWNER])

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

    def search_documents(self, query: str, limit: int = 5, only: list[str] | None = None,
                         user_id: str | None = None) -> list[dict]:
        """BM25-ranked chunk search. Returns [{path, chunk}] best first.

        When user_id is given, results are restricted to documents that user owns
        plus shared documents, so one user's imported material never leaks into
        another user's retrieval.
        """
        if not getattr(self, "fts_enabled", False):
            return []
        # FTS5 treats punctuation as syntax; reduce the query to bare terms and
        # OR them so a natural-language question still matches.
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query or "") if len(t) > 2]
        if not terms:
            return []
        if getattr(self, "search_mode", "fts5") != "fts5":
            return self._search_documents_fallback(terms, limit, only, user_id)
        expr = " OR ".join(terms[:12])
        sql = "SELECT path, chunk FROM doc_chunks WHERE doc_chunks MATCH ?"
        params: list[Any] = [expr]
        if only:
            # Scope retrieval to chosen documents ("chat with this document").
            sql += " AND path IN (" + ",".join("?" for _ in only) + ")"
            params.extend(only)
        owner_clause, owner_params = self._owned_paths_clause(user_id)
        sql += owner_clause
        params.extend(owner_params)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"path": r["path"], "chunk": r["chunk"]} for r in rows]

    def _search_documents_fallback(self, terms: list[str], limit: int,
                                   only: list[str] | None,
                                   user_id: str | None = None) -> list[dict]:
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
        if user_id is not None:
            clauses.append("path IN (SELECT path FROM documents WHERE user_id IN (?, ?))")
            params.extend([user_id, SHARED_OWNER])
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

    def document_stats(self, user_id: str | None = None) -> dict:
        if not getattr(self, "fts_enabled", False):
            return {"enabled": False, "documents": 0, "chunks": 0, "items": []}
        where, params = "", []
        if user_id is not None:
            where = "WHERE user_id IN (?, ?)"
            params = [user_id, SHARED_OWNER]
        rows = self.execute(
            f"SELECT path, title, chars, chunks, indexed_at FROM documents {where} "
            f"ORDER BY indexed_at DESC", tuple(params)
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

    def search_conversations(self, query: str, limit: int = 30,
                             user_id: str | None = None) -> list[dict]:
        """Find conversations containing a phrase, newest first, with a snippet."""
        # Escape LIKE wildcards so searching for "%" or "_" looks for those
        # characters instead of matching every conversation.
        safe = (query or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{safe}%"
        owner = ""
        params: list[Any] = [like]
        if user_id is not None:
            owner = "AND m.user_id = ?"
            params.append(user_id)
        params.append(limit)
        # LEFT JOIN the title/pin metadata in one pass. Fetching it per row was an
        # N+1: a 30-result search issued 31 queries.
        rows = self.execute(
            f"""SELECT m.conversation_id AS conversation_id, m.role AS role,
                      m.content AS content, MAX(m.created_at) AS created_at,
                      cm.title AS meta_title, cm.pinned AS meta_pinned
               FROM messages m
               LEFT JOIN conversation_meta cm
                      ON cm.conversation_id = m.conversation_id
               WHERE m.content LIKE ? ESCAPE '\\' {owner}
               GROUP BY m.conversation_id ORDER BY created_at DESC LIMIT ?""",
            tuple(params)).fetchall()
        out = []
        for r in rows:
            content = r["content"] or ""
            idx = content.lower().find(query.lower())
            start = max(0, idx - 60)
            snippet = ("..." if start else "") + content[start:start + 200].strip()
            # Same title/pinned fields the plain list provides: search rows go
            # through the identical row renderer in the UI.
            out.append({"conversation_id": r["conversation_id"], "role": r["role"],
                        "snippet": snippet, "created_at": r["created_at"],
                        "title": r["meta_title"] or "",
                        "pinned": bool(r["meta_pinned"])})
        return out

    def export_conversation(self, conversation_id: str, fmt: str = "markdown",
                            user_id: str | None = None) -> str:
        rows = self.get_messages(conversation_id, limit=10000, user_id=user_id)
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

    # Prompts are per-user. They were one global store, so on a multi-user
    # install every user saw, overwrote and deleted everyone else's prompts.
    # user_id=None keeps the single-user (auth-off) behaviour unscoped.
    def save_prompt(self, name: str, body: str, user_id: str | None = None) -> None:
        if not (name or "").strip():
            raise ValueError("a prompt needs a name")
        if not (body or "").strip():
            raise ValueError("a prompt needs a body")
        self.execute(
            "INSERT OR REPLACE INTO prompts (name, body, user_id, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (name.strip(), body, user_id or SENTINEL_LOCAL_USER))
        self.commit()

    def list_prompts(self, user_id: str | None = None) -> list[dict]:
        if user_id is None:
            rows = self.execute(
                "SELECT name, body, updated_at FROM prompts ORDER BY name ASC").fetchall()
        else:
            rows = self.execute(
                "SELECT name, body, updated_at FROM prompts WHERE user_id = ? "
                "ORDER BY name ASC", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_prompt(self, name: str, user_id: str | None = None) -> bool:
        if user_id is None:
            cur = self.execute("DELETE FROM prompts WHERE name = ?", (name,))
        else:
            cur = self.execute("DELETE FROM prompts WHERE name = ? AND user_id = ?",
                               (name, user_id))
        self.commit()
        return cur.rowcount > 0

    def drop_last_exchange(self, conversation_id: str, user_id: str | None = None) -> str | None:
        """Remove the last assistant reply (and return the user prompt that led to
        it) so the turn can be regenerated."""
        rows = self.get_messages(conversation_id, limit=50, user_id=user_id)
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
                # Carry the owner across: the export row has it, and dropping it
                # would restore every user's prompts into the shared library.
                self.save_prompt(pr.get("name", ""), pr.get("body", ""),
                                 user_id=pr.get("user_id"))
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

    def fork_conversation(self, conversation_id: str, upto_message_id: int | None = None,
                          user_id: str | None = None) -> str:
        """Copy a conversation (optionally only up to a message) into a new one,
        so you can explore a different direction without losing the original.

        When user_id is given the source is filtered by owner and the new
        conversation is owned by that user, so a fork stays within one account.
        """
        check = "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?"
        check_params: list[Any] = [conversation_id]
        if user_id is not None:
            check += " AND user_id = ?"
            check_params.append(user_id)
        existing = self.execute(check, tuple(check_params)).fetchone()
        if not existing or not existing["c"]:
            raise ValueError(f"conversation {conversation_id!r} has no messages to fork")
        new_id = f"fork-{uuid.uuid4().hex[:10]}"
        sql = "SELECT role, content FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if upto_message_id:
            sql += " AND id <= ?"
            params.append(upto_message_id)
        sql += " ORDER BY id ASC"
        for row in self.execute(sql, tuple(params)).fetchall():
            self.add_message(new_id, row["role"], row["content"], user_id=user_id)
        base = self.conversation_title(conversation_id) or conversation_id[:8]
        self.set_conversation_title(new_id, f"{base} (fork)"[:120])
        return new_id

    def record_feedback(self, user_prompt: str, assistant_response: str, rating: int,
                        approved: int, corrected: str | None = None,
                        session_id: str | None = None, model_id: str | None = None,
                        source: str = "button", user_id: str | None = None) -> None:
        """Insert one feedback row (used by the button and by implicit chat feedback)."""
        self.execute(
            """INSERT INTO feedback
               (user_prompt, assistant_response, rating, corrected_response,
                approved_for_training, session_id, model_id, source, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_prompt, assistant_response, rating, corrected, approved,
             session_id or "implicit", model_id, source, user_id or SENTINEL_LOCAL_USER),
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
        # Rows a non-admin submitted that an admin has not cleared for training
        # yet. Surfaced so the queue is visible rather than silently growing.
        pending = scalar("SELECT COUNT(*) FROM feedback "
                         "WHERE approved_for_training = 0 AND pending_approval = 1")
        return {"total": total, "approved": approved, "rejected": rejected,
                "reviewed": reviewed, "corrected": corrected, "pending": pending,
                "preference_pairs": pref, "by_source": by_source}

    def set_reviewed(self, feedback_id: int, reviewed: bool) -> bool:
        cur = self.execute("UPDATE feedback SET reviewed = ? WHERE id = ?",
                           (1 if reviewed else 0, feedback_id))
        self.commit()
        return cur.rowcount > 0

    def set_approved(self, feedback_id: int, approved: bool) -> bool:
        """Admin decision on one queued row. Clears pending_approval either way:
        an explicit reject is a decision too, and should leave the queue."""
        cur = self.execute(
            "UPDATE feedback SET approved_for_training = ?, pending_approval = 0 "
            "WHERE id = ?", (1 if approved else 0, feedback_id))
        self.commit()
        return cur.rowcount > 0

    def approve_pending(self) -> int:
        """Clear the whole queue in one go. Returns how many rows were approved."""
        cur = self.execute(
            "UPDATE feedback SET approved_for_training = 1, pending_approval = 0 "
            "WHERE approved_for_training = 0 AND pending_approval = 1")
        self.commit()
        return cur.rowcount

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

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
        user_id: str | None = None,
    ) -> int:
        cursor = self.execute(
            "INSERT INTO messages (conversation_id, role, content, est_tokens, meta, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                estimate_tokens(content),
                json.dumps(meta) if meta else None,
                user_id or SENTINEL_LOCAL_USER,
            ),
        )
        self.commit()
        return int(cursor.lastrowid or 0)

    def get_messages(self, conversation_id: str, limit: int = 200,
                     user_id: str | None = None) -> list[dict]:
        """Return the tail of a conversation in chronological order.

        When user_id is given, the conversation is filtered by owner too, so a
        client cannot read another user's conversation by guessing its id.
        """
        sql = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        rows = self.execute(
            f"SELECT * FROM ({sql} ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def can_access_conversation(self, conversation_id: str, user_id: str | None) -> bool:
        """True if the conversation is empty (new) or owned by this user.

        A None user_id means "no scoping" (admin/global or auth disabled) and
        always passes. An empty conversation belongs to whoever writes first.
        """
        if user_id is None:
            return True
        row = self.execute(
            "SELECT user_id, COUNT(*) AS c FROM messages WHERE conversation_id = ?",
            (conversation_id,)).fetchone()
        if not row or not row["c"]:
            return True
        return row["user_id"] == user_id

    # --- Active task state ------------------------------------------------- #
    # The structured task/artifact record for a conversation (see taskstate.py).
    # Stored as one JSON blob rather than columns: the shape is owned by
    # TaskState and adding a field there must not need a migration here.

    def save_task_state(self, conversation_id: str, state: dict,
                        user_id: str | None = None) -> None:
        if not conversation_id:
            return
        self.execute(
            "INSERT INTO conversation_task_state (conversation_id, user_id, state, "
            "updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(conversation_id) DO UPDATE SET state = excluded.state, "
            "user_id = excluded.user_id, updated_at = CURRENT_TIMESTAMP",
            (conversation_id, user_id or SENTINEL_LOCAL_USER,
             json.dumps(state, ensure_ascii=False, default=str)))
        self.commit()

    def load_task_state(self, conversation_id: str,
                        user_id: str | None = None) -> dict | None:
        """The stored task state, or None. Owner-scoped like get_messages."""
        if not conversation_id:
            return None
        sql = "SELECT state FROM conversation_task_state WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        row = self.execute(sql, tuple(params)).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["state"])
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def clear_task_state(self, conversation_id: str) -> None:
        self.execute("DELETE FROM conversation_task_state WHERE conversation_id = ?",
                     (conversation_id,))
        self.commit()

    def clear_conversation(self, conversation_id: str, user_id: str | None = None) -> int:
        sql = "DELETE FROM messages WHERE conversation_id = ?"
        params: list[Any] = [conversation_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        cursor = self.execute(sql, tuple(params))
        self.execute("DELETE FROM tool_calls WHERE conversation_id = ?", (conversation_id,))
        # Clearing the transcript clears the task: leaving the row behind would
        # brief the model on an artifact the user has just deleted.
        self.execute("DELETE FROM conversation_task_state WHERE conversation_id = ?",
                     (conversation_id,))
        self.commit()
        return cursor.rowcount

    def list_conversations(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        where = ""
        params: list[Any] = []
        if user_id is not None:
            where = "WHERE user_id = ?"
            params.append(user_id)
        rows = self.execute(
            f"SELECT conversation_id, COUNT(*) AS messages, MAX(created_at) AS last_at "
            f"FROM messages {where} GROUP BY conversation_id ORDER BY last_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        # Title + pin metadata and the fallback title come from two more queries
        # PER ROW originally: listing 50 conversations cost 101 queries. Both are
        # resolved in one extra pass over just the ids on this page.
        ids = [r["conversation_id"] for r in rows]
        meta_by_id: dict[str, dict] = {}
        first_by_id: dict[str, str] = {}
        if ids:
            marks = ",".join("?" for _ in ids)
            for m in self.execute(
                    f"SELECT conversation_id, title, pinned FROM conversation_meta "
                    f"WHERE conversation_id IN ({marks})", tuple(ids)).fetchall():
                meta_by_id[m["conversation_id"]] = dict(m)
            # One row per conversation: the earliest user message, as a fallback title.
            for f in self.execute(
                    f"SELECT conversation_id, content FROM messages WHERE id IN ("
                    f"  SELECT MIN(id) FROM messages WHERE role = 'user' "
                    f"  AND conversation_id IN ({marks}) GROUP BY conversation_id)",
                    tuple(ids)).fetchall():
                first_by_id[f["conversation_id"]] = f["content"] or ""
        out = []
        for row in rows:
            item = dict(row)
            cid = item["conversation_id"]
            meta = meta_by_id.get(cid)
            custom = (meta or {}).get("title")
            item["pinned"] = bool((meta or {}).get("pinned"))
            item["title"] = (custom
                             or (first_by_id.get(cid, "").strip()[:90])
                             or "(no messages)")
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

    def skills_used(self, conversation_id: str | None, limit: int = 20) -> list[str]:
        """Skill names loaded in a conversation, most recent first.

        Derived from tool_calls rather than a new column: every load_skill call
        is already persisted with its arguments, so attribution needs no schema
        change and works retroactively on conversations that predate the
        learning loop. This is what lets a rating -- which arrives minutes after
        the turn, from a different request -- be credited to the procedures that
        produced the answer.
        """
        if not conversation_id:
            return []
        rows = self.execute(
            "SELECT args FROM tool_calls WHERE conversation_id = ? AND name = 'load_skill' "
            "AND (error IS NULL OR error = '') ORDER BY id DESC LIMIT ?",
            (conversation_id, max(1, int(limit))),
        ).fetchall()
        names: list[str] = []
        for row in rows:
            try:
                args = json.loads(row["args"] or "{}")
            except Exception:
                continue
            name = str(args.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def log_tool_call(
        self,
        conversation_id: str | None,
        name: str,
        args: dict,
        result: str,
        duration_ms: float,
        error: str | None = None,
        user_id: str | None = None,
    ) -> None:
        # Redact secrets from both args and result before they hit disk: a tool
        # can receive a URL with embedded credentials or return a page that
        # echoes a token. Persisting them raw would defeat log redaction.
        safe_args = json.dumps(redact_obj(args or {}), ensure_ascii=False)[:4000]
        safe_result = redact_text(result or "")[:8000]
        self.execute(
            "INSERT INTO tool_calls (conversation_id, name, args, result, duration_ms, error, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                name,
                safe_args,
                safe_result,
                duration_ms,
                redact_text(error) if error else None,
                user_id or SENTINEL_LOCAL_USER,
            ),
        )
        self.commit()

    def list_tool_calls(self, limit: int = 100, conversation_id: str | None = None,
                        user_id: str | None = None) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.execute(
            f"SELECT * FROM tool_calls {where} ORDER BY id DESC LIMIT ?", tuple(params)
        ).fetchall()
        return [dict(row) for row in rows]

    def remember(self, key: str, value: str, conversation_id: str | None = None,
                 user_id: str | None = None) -> None:
        """Upsert a durable note the agent can read back in a later session.

        Scoped per user: the conflict target is (user_id, key), so one user's
        note can never overwrite another's under the same key.
        """
        stamp = datetime.now(timezone.utc).isoformat()
        self.execute(
            "INSERT INTO memories (user_id, key, value, conversation_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
            "conversation_id = excluded.conversation_id, updated_at = excluded.updated_at",
            (user_id or SENTINEL_LOCAL_USER, key, value, conversation_id, stamp),
        )
        self.commit()

    def recall(self, query: str | None = None, limit: int = 10,
               user_id: str | None = None) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if query:
            clauses.append("(key LIKE ? OR value LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.execute(
            f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---------------------------------------------------------------- tasks --

    TASK_FIELDS = (
        "name", "goal", "enabled", "interval_seconds", "max_steps",
        "tools", "system_prompt", "use_history", "model", "next_task_id",
    )

    def create_task(self, user_id: str | None = None, **fields: Any) -> dict:
        task_id = str(uuid.uuid4())[:12]
        interval = int(fields.get("interval_seconds") or 0)
        enabled = 1 if fields.get("enabled", True) else 0
        # A repeating task is due immediately so the operator sees a first run
        # rather than waiting out a one-hour interval to find out it is wrong.
        next_run = iso(utc_now()) if (enabled and interval > 0) else None
        self.execute(
            "INSERT INTO tasks (id, name, goal, enabled, interval_seconds, max_steps, "
            "tools, system_prompt, use_history, model, next_task_id, next_run_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                user_id or SENTINEL_LOCAL_USER,
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

    def count_tasks(self) -> int:
        """How many tasks exist. /api/health only needs the number, and building
        a dict per row every few seconds to call len() on it is pure waste."""
        return self.execute("SELECT COUNT(*) AS cnt FROM tasks").fetchone()["cnt"]

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

    def count_memories(self, user_id: str | None = None) -> int:
        if user_id is not None:
            return self.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ?",
                (user_id,)).fetchone()["cnt"]
        return self.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]

    def forget(self, key: str, user_id: str | None = None) -> bool:
        sql = "DELETE FROM memories WHERE key = ?"
        params: list[Any] = [key]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        cursor = self.execute(sql, tuple(params))
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
        user_id: str | None = None,
    ) -> None:
        self.execute(
            "INSERT INTO metrics (endpoint, duration_ms, status_code, error, prompt_tokens, "
            "completion_tokens, ttft_ms, decode_tps, model, step, conversation_id, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint, duration_ms, status_code, error,
                stats.prompt_tokens if stats else None,
                stats.completion_tokens if stats else None,
                round(stats.ttft_ms, 1) if stats and stats.ttft_ms else None,
                round(stats.decode_tps, 2) if stats and stats.decode_tps else None,
                model, step, conversation_id, user_id or SENTINEL_LOCAL_USER,
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

    # ------------------------------------------------------------- users --

    def create_user(self, username: str, password_hash: str | None = None,
                    role: str = "user", source: str = "local",
                    email: str | None = None, display_name: str | None = None,
                    oidc_subject: str | None = None, user_id: str | None = None) -> dict:
        uid = user_id or uuid.uuid4().hex
        self.execute(
            "INSERT INTO users (id, username, password_hash, role, source, email, "
            "display_name, oidc_subject) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, username, password_hash, role, source, email, display_name, oidc_subject))
        self.commit()
        return self.get_user(uid) or {}

    def get_user(self, user_id: str) -> dict | None:
        row = self.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict | None:
        row = self.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def get_user_by_oidc(self, subject: str) -> dict | None:
        if not subject:
            return None
        row = self.execute("SELECT * FROM users WHERE oidc_subject = ?", (subject,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        return [dict(r) for r in self.execute(
            "SELECT * FROM users ORDER BY created_at ASC").fetchall()]

    def count_users(self, role: str | None = None) -> int:
        if role:
            return self.execute("SELECT COUNT(*) AS c FROM users WHERE role = ?",
                                (role,)).fetchone()["c"]
        return self.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]

    def count_enabled_admins(self) -> int:
        """Admins who can actually log in — the count the lockout guard must use
        (a disabled admin does not keep you from being locked out)."""
        return self.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role = 'admin' AND disabled = 0"
        ).fetchone()["c"]

    def update_user(self, user_id: str, **fields: Any) -> dict | None:
        allowed = {"password_hash", "role", "email", "display_name", "disabled",
                   "username", "oidc_subject", "source"}
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "disabled":
                value = 1 if value else 0
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return self.get_user(user_id)
        sets.append("updated_at = ?")
        params.append(iso(utc_now()))
        params.append(user_id)
        self.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", tuple(params))
        self.commit()
        return self.get_user(user_id)

    def touch_login(self, user_id: str) -> None:
        self.execute("UPDATE users SET last_login_at = ? WHERE id = ?",
                     (iso(utc_now()), user_id))
        self.commit()

    def delete_user(self, user_id: str) -> bool:
        cur = self.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------ agents --

    @staticmethod
    def _agent_row(row) -> dict:
        d = dict(row)
        try:
            d["capabilities"] = json.loads(d.get("capabilities") or "[]")
        except (TypeError, ValueError):
            d["capabilities"] = []
        d["enabled"] = bool(d.get("enabled"))
        return d

    def create_agent(self, name: str, description: str = "",
                     capabilities: list | None = None, enabled: bool = True) -> dict:
        aid = uuid.uuid4().hex
        self.execute(
            "INSERT INTO agents (id, name, description, capabilities, enabled) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, name.strip()[:120], (description or "").strip()[:2000],
             json.dumps(list(capabilities or [])), 1 if enabled else 0))
        self.commit()
        return self.get_agent(aid) or {}

    def get_agent(self, agent_id: str) -> dict | None:
        row = self.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._agent_row(row) if row else None

    def list_agents(self) -> list[dict]:
        return [self._agent_row(r) for r in self.execute(
            "SELECT * FROM agents ORDER BY name COLLATE NOCASE ASC").fetchall()]

    def update_agent(self, agent_id: str, **fields: Any) -> dict | None:
        sets, params = [], []
        if "name" in fields:
            sets.append("name = ?"); params.append(str(fields["name"]).strip()[:120])
        if "description" in fields:
            sets.append("description = ?"); params.append(str(fields["description"] or "").strip()[:2000])
        if "capabilities" in fields:
            sets.append("capabilities = ?"); params.append(json.dumps(list(fields["capabilities"] or [])))
        if "enabled" in fields:
            sets.append("enabled = ?"); params.append(1 if fields["enabled"] else 0)
        if not sets:
            return self.get_agent(agent_id)
        sets.append("updated_at = ?"); params.append(iso(utc_now()))
        params.append(agent_id)
        self.execute(f"UPDATE agents SET {', '.join(sets)} WHERE id = ?", tuple(params))
        self.commit()
        return self.get_agent(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        cur = self.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        self.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------------- sessions --

    def create_session(self, token_hash: str, user_id: str, expires_at: str,
                       ip: str | None = None, user_agent: str | None = None) -> None:
        self.execute(
            "INSERT OR REPLACE INTO sessions (token_hash, user_id, expires_at, ip, user_agent) "
            "VALUES (?, ?, ?, ?, ?)",
            (token_hash, user_id, expires_at, ip, (user_agent or "")[:400]))
        self.commit()

    def get_session(self, token_hash: str) -> dict | None:
        row = self.execute("SELECT * FROM sessions WHERE token_hash = ?",
                           (token_hash,)).fetchone()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> bool:
        cur = self.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        self.commit()
        return cur.rowcount > 0

    def delete_user_sessions(self, user_id: str) -> int:
        cur = self.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self.commit()
        return cur.rowcount

    def purge_expired_sessions(self) -> int:
        cur = self.execute("DELETE FROM sessions WHERE expires_at < ?", (iso(utc_now()),))
        self.commit()
        return cur.rowcount

    # ----------------------------------------------------------- imports --

    def create_import(self, import_id: str, user_id: str, filename: str,
                      size_bytes: int) -> dict:
        self.execute(
            "INSERT INTO imports (id, user_id, filename, size_bytes, status, progress) "
            "VALUES (?, ?, ?, ?, 'pending', 0)",
            (import_id, user_id, filename, size_bytes))
        self.commit()
        return self.get_import(import_id) or {}

    def update_import(self, import_id: str, **fields: Any) -> None:
        allowed = {"status", "progress", "counts", "warnings", "error",
                   "finished_at", "artifacts"}
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("counts", "warnings", "artifacts") and not isinstance(value, str):
                value = json.dumps(value, default=str)
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return
        sets.append("updated_at = ?")
        params.append(iso(utc_now()))
        params.append(import_id)
        self.execute(f"UPDATE imports SET {', '.join(sets)} WHERE id = ?", tuple(params))
        self.commit()

    def get_import(self, import_id: str) -> dict | None:
        row = self.execute("SELECT * FROM imports WHERE id = ?", (import_id,)).fetchone()
        return dict(row) if row else None

    def list_imports(self, user_id: str | None = None, limit: int = 50) -> list[dict]:
        if user_id is not None:
            rows = self.execute(
                "SELECT * FROM imports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)).fetchall()
        else:
            rows = self.execute(
                "SELECT * FROM imports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete_import(self, import_id: str, user_id: str | None = None) -> bool:
        sql = "DELETE FROM imports WHERE id = ?"
        params: list[Any] = [import_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        cur = self.execute(sql, tuple(params))
        self.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------- routing events --

    def log_routing_event(self, **fields: Any) -> None:
        cols = ["correlation_id", "task_id", "user_id", "conversation_id", "kind",
                "requested_model", "selected_model", "selected_node", "reason",
                "candidates", "status", "attempt", "duration_ms", "error"]
        values: list[Any] = []
        for col in cols:
            value = fields.get(col)
            if col == "candidates" and value is not None and not isinstance(value, str):
                value = json.dumps(value, default=str)[:8000]
            values.append(value)
        self.execute(
            f"INSERT INTO routing_events ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})", tuple(values))
        self.commit()

    def list_routing_events(self, limit: int = 100, node: str | None = None,
                            task_id: str | None = None) -> list[dict]:
        clauses, params = [], []
        if node:
            clauses.append("selected_node = ?")
            params.append(node)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.execute(
            f"SELECT * FROM routing_events {where} ORDER BY id DESC LIMIT ?",
            tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def routing_summary(self, limit: int = 500) -> dict:
        rows = self.execute(
            "SELECT selected_node, status, COUNT(*) AS c FROM "
            "(SELECT * FROM routing_events ORDER BY id DESC LIMIT ?) "
            "GROUP BY selected_node, status", (limit,)).fetchall()
        by_node: dict[str, dict] = {}
        for r in rows:
            node = r["selected_node"] or "?"
            by_node.setdefault(node, {})[r["status"] or "?"] = r["c"]
        return {"by_node": by_node, "samples": sum(
            r["c"] for r in rows)}



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Database',
]
