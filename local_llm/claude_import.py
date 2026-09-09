"""Claude conversation-history import: validate → extract → parse → store.

An admin uploads a Claude data-export ZIP (as produced by Claude's "Export data"
feature: a README, conversations.json, projects.json, users.json, plus any
supporting files). The archive is UNTRUSTED, so extraction is hardened against
path traversal (zip-slip), symlinks, zip bombs, oversized uploads and excessive
file counts. Parsed content is normalised, de-duplicated, and stored so it is
usable three ways without ever being dumped wholesale into a prompt:

* **Historical conversations** -> written as real conversations (owned by the
  importing user) that show up in History and can be reopened.
* **Reusable knowledge / historical context** -> each conversation and project
  document is indexed into the user's knowledge base, so the existing BM25
  retrieval surfaces only the passages relevant to a new question.
* **Reusable skills / preferences** -> project instructions become saved prompts;
  explicit preferences become durable memories.

The pipeline runs in a background thread and reports progress/counts/errors via
the ``imports`` table so the admin Settings UI can show status and retry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403


_import_log = get_logger("import")

# Where uploaded archives are staged and extracted, one directory per import id.
IMPORTS_DIR = DATA_DIR / "imports"

# Files we recognise as text/knowledge when attaching supporting artifacts.
_TEXT_EXT = {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".py", ".js",
             ".ts", ".html", ".htm", ".yaml", ".yml", ".toml", ".rst"}


class ImportError_(Exception):
    """A validation/processing failure surfaced to the admin UI."""


class ImportManager:
    """Validate, extract, parse and store a Claude history export."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db

    # ---- entry points ----------------------------------------------------- #
    def stage_upload(self, user_id: str, filename: str, data: bytes) -> str:
        """Validate size and write the upload to a fresh staging dir. Returns id."""
        if not data:
            raise ImportError_("the uploaded file is empty")
        if len(data) > self.config.import_max_zip_bytes:
            raise ImportError_(
                f"upload is {len(data)} bytes; the limit is "
                f"{self.config.import_max_zip_bytes} (IMPORT_MAX_ZIP_BYTES)")
        if not zipfile.is_zipfile(_BytesFile(data)):
            raise ImportError_("the upload is not a valid ZIP archive")
        import_id = uuid.uuid4().hex[:16]
        staging = self._staging(import_id)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "upload.zip").write_bytes(data)
        self.db.create_import(import_id, user_id, (filename or "export.zip")[:200], len(data))
        log_event(_import_log, 20, "import.staged", import_id=import_id,
                  user_id=user_id, size_bytes=len(data), filename=filename)
        return import_id

    def _staging(self, import_id: str) -> Path:
        return IMPORTS_DIR / import_id

    # ---- the pipeline ----------------------------------------------------- #
    def process(self, import_id: str, user_id: str) -> dict:
        """Run the full pipeline. Records status/counts/warnings; returns counts."""
        counts = {"conversations": 0, "messages": 0, "knowledge_docs": 0,
                  "skills": 0, "files": 0, "duplicates": 0}
        warnings: list[str] = []
        # The exact conversations and knowledge docs this import created, so that
        # removing THIS import deletes only its own artifacts (never the user's
        # other imports).
        artifacts: dict = {"conversations": [], "docs": [], "prompts": []}
        staging = self._staging(import_id)
        extract_dir = staging / "extracted"
        try:
            self.db.update_import(import_id, status="validating", progress=5)
            zip_path = staging / "upload.zip"
            if not zip_path.exists():
                raise ImportError_("staged upload is missing; re-upload the file")

            self.db.update_import(import_id, status="extracting", progress=15)
            self._safe_extract(zip_path, extract_dir, warnings)

            self.db.update_import(import_id, status="parsing", progress=45)
            conversations, projects, prefs = self._discover(extract_dir, warnings)

            self.db.update_import(import_id, status="storing", progress=65)
            self._store_conversations(user_id, conversations, counts, warnings, artifacts)
            self._store_projects(user_id, projects, counts, warnings, artifacts)
            self._store_preferences(user_id, prefs, counts, warnings)
            self._attach_files(user_id, extract_dir, counts, warnings, artifacts)

            self.db.update_import(
                import_id, status="completed", progress=100, counts=counts,
                warnings=warnings, artifacts=artifacts, finished_at=iso(utc_now()))
            log_event(_import_log, 20, "import.completed", import_id=import_id,
                      user_id=user_id, **counts)
        except Exception as exc:
            message = redact_text(f"{type(exc).__name__}: {exc}")
            self.db.update_import(import_id, status="failed", error=message,
                                  counts=counts, warnings=warnings,
                                  finished_at=iso(utc_now()))
            # A rejected hostile/invalid archive (ImportError_: zip-slip, zip
            # bomb, oversize, not-a-zip) is the guard doing its job -> WARNING.
            # Only an unexpected processing crash is an ERROR worth alerting on.
            rejected = isinstance(exc, ImportError_)
            log_event(_import_log, 30 if rejected else 40,
                      "import.rejected" if rejected else "import.failed",
                      import_id=import_id, user_id=user_id, error=message)
        finally:
            # Extracted tree can be large; keep only the original upload for retry.
            try:
                if extract_dir.exists():
                    shutil.rmtree(extract_dir, ignore_errors=True)
            except Exception:
                pass      # cleanup only; the import result already stands
        return counts

    # ---- hardened extraction --------------------------------------------- #
    def _safe_extract(self, zip_path: Path, dest: Path, warnings: list[str]) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        dest_resolved = dest.resolve()
        total_uncompressed = 0   # declared (header) sizes, for the pre-check
        actual_written = 0       # real bytes written, the authoritative aggregate
        file_count = 0
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > self.config.import_max_files:
                raise ImportError_(
                    f"archive has {len(infos)} entries; the limit is "
                    f"{self.config.import_max_files} (IMPORT_MAX_FILES)")
            for info in infos:
                name = info.filename
                # Reject absolute paths and drive letters.
                if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
                    raise ImportError_(f"unsafe absolute path in archive: {name!r}")
                # Reject symlinks (mode high bits 0xA000).
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    warnings.append(f"skipped symlink entry: {name}")
                    continue
                if info.is_dir():
                    continue
                # Resolve the target and confirm it stays inside dest (zip-slip).
                target = (dest / name).resolve()
                if os.path.commonpath([dest_resolved, target]) != str(dest_resolved):
                    raise ImportError_(f"path traversal blocked for entry: {name!r}")
                # Per-file and total uncompressed caps (zip-bomb defence).
                if info.file_size > self.config.import_max_file_bytes:
                    warnings.append(
                        f"skipped oversized entry ({info.file_size} bytes): {name}")
                    continue
                total_uncompressed += info.file_size
                if total_uncompressed > self.config.import_max_uncompressed_bytes:
                    raise ImportError_(
                        "archive expands beyond the uncompressed limit "
                        f"({self.config.import_max_uncompressed_bytes} bytes, "
                        "IMPORT_MAX_UNCOMPRESSED_BYTES); refusing (possible zip bomb)")
                # Compression-ratio guard for a single entry.
                if info.compress_size > 0 and info.file_size / info.compress_size > 200:
                    warnings.append(
                        f"skipped entry with suspicious compression ratio: {name}")
                    continue
                file_count += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    # Stream-copy counting ACTUAL bytes, not the declared header
                    # size: a crafted zip can understate file_size to slip past
                    # the pre-check. Enforce both the per-file cap and the running
                    # aggregate cap on real bytes written.
                    remaining = self.config.import_max_file_bytes
                    while True:
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        if remaining < 0:
                            out.close()
                            target.unlink(missing_ok=True)
                            warnings.append(f"truncated entry exceeding size cap: {name}")
                            break
                        actual_written += len(chunk)
                        if actual_written > self.config.import_max_uncompressed_bytes:
                            out.close()
                            target.unlink(missing_ok=True)
                            raise ImportError_(
                                "archive expands beyond the uncompressed limit while "
                                f"extracting ({self.config.import_max_uncompressed_bytes} "
                                "bytes, IMPORT_MAX_UNCOMPRESSED_BYTES); refusing "
                                "(possible zip bomb)")
                        out.write(chunk)
        log_event(_import_log, 20, "import.extracted", files=file_count,
                  declared_bytes=total_uncompressed, actual_bytes=actual_written)

    # ---- discovery / parsing --------------------------------------------- #
    def _discover(self, root: Path, warnings: list[str]) -> tuple[list, list, dict]:
        conversations: list[dict] = []
        projects: list[dict] = []
        prefs: dict = {}
        for path in root.rglob("*.json"):
            name = path.name.lower()
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                warnings.append(f"could not parse {path.name}: {exc}")
                continue
            if "conversation" in name:
                conversations.extend(self._parse_conversations(data, warnings))
            elif "project" in name:
                projects.extend(self._parse_projects(data, warnings))
            elif "user" in name or "account" in name or "preference" in name:
                if isinstance(data, dict):
                    prefs.update(self._parse_preferences(data))
            else:
                # Unknown JSON that still looks like conversations.
                if isinstance(data, list) and data and isinstance(data[0], dict) and (
                        "chat_messages" in data[0] or "messages" in data[0]
                        or "mapping" in data[0]):
                    conversations.extend(self._parse_conversations(data, warnings))
        return conversations, projects, prefs

    @staticmethod
    def _msg_text(msg: dict) -> str:
        """Text of one message, across export dialects.

        Claude puts it in `text` or a `content` list; OpenAI/DeepSeek wrap it as
        content={"parts": [...]}; some exports use a bare string. Anything whose
        parts are non-text (images, tool payloads) contributes nothing.
        """
        if isinstance(msg.get("text"), str) and msg["text"].strip():
            return msg["text"]
        parts = msg.get("content") or []
        if isinstance(parts, dict):
            # OpenAI/DeepSeek: {"content_type": "text", "parts": ["..."]}
            parts = parts.get("parts") or parts.get("text") or []
            if isinstance(parts, str):
                parts = [parts]
        chunks = []
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part, str):
                    chunks.append(part)
        elif isinstance(parts, str):
            chunks.append(parts)
        return "\n".join(c for c in chunks if c).strip()

    @staticmethod
    def _messages_from_mapping(mapping: dict) -> list[dict]:
        """Flatten an OpenAI/DeepSeek-style `mapping` graph into ordered messages.

        Those exports store the thread as a node graph keyed by id, not a list, so
        a parser that only looks for `messages`/`chat_messages` finds nothing and
        silently drops every conversation. Walk parent -> children from the root,
        then append any orphans in timestamp order.
        """
        nodes = {k: v for k, v in mapping.items() if isinstance(v, dict)}
        if not nodes:
            return []
        roots = [k for k, v in nodes.items() if not v.get("parent")] or [next(iter(nodes))]
        order: list[str] = []
        seen: set = set()
        stack = list(roots)
        while stack:
            nid = stack.pop(0)
            if nid in seen or nid not in nodes:
                continue
            seen.add(nid)
            order.append(nid)
            kids = [c for c in (nodes[nid].get("children") or []) if c in nodes]
            stack = kids + stack                      # follow the branch we are on
        rest = [k for k in nodes if k not in seen]

        def _when(node_id: str):
            msg = nodes[node_id].get("message") or {}
            return msg.get("create_time") or msg.get("created_at") or 0

        rest.sort(key=_when)
        order.extend(rest)
        out = []
        for nid in order:
            msg = nodes[nid].get("message")
            if isinstance(msg, dict):
                out.append(msg)
        return out

    @staticmethod
    def _msg_role(msg: dict) -> str:
        """user | assistant | other, across dialects (Claude sender, OpenAI author.role)."""
        author = msg.get("author")
        raw = (msg.get("sender") or msg.get("role")
               or (author.get("role") if isinstance(author, dict) else "")
               or "")
        raw = str(raw).lower()
        if raw in ("assistant", "ai", "claude", "model", "bot", "gpt", "grok", "deepseek"):
            return "assistant"
        if raw in ("human", "user"):
            return "user"
        return raw or "user"

    def _parse_conversations(self, data: Any, warnings: list[str]) -> list[dict]:
        items = data if isinstance(data, list) else data.get("conversations", []) \
            if isinstance(data, dict) else []
        out = []
        for conv in items:
            if not isinstance(conv, dict):
                continue
            source_id = str(conv.get("uuid") or conv.get("id") or uuid.uuid4().hex)
            title = (conv.get("name") or conv.get("title")
                     or conv.get("conversation_title") or "").strip()
            raw_msgs = conv.get("chat_messages") or conv.get("messages") or []
            if not raw_msgs and isinstance(conv.get("mapping"), dict):
                raw_msgs = self._messages_from_mapping(conv["mapping"])
            messages = []
            for msg in raw_msgs:
                if not isinstance(msg, dict):
                    continue
                role = self._msg_role(msg)
                # A browsable transcript is the human/assistant exchange; system
                # and tool frames are plumbing and would read as the user talking.
                if role not in ("user", "assistant"):
                    continue
                text = self._msg_text(msg)
                if not text:
                    continue
                messages.append({"role": role, "content": text,
                                 "created_at": msg.get("created_at") or msg.get("create_time")})
            if messages:
                out.append({"source_id": source_id,
                            "title": title or (messages[0]["content"][:80]),
                            "created_at": (conv.get("created_at")
                                           or conv.get("create_time")
                                           or conv.get("inserted_at")),
                            "messages": messages})
        return out

    def _parse_projects(self, data: Any, warnings: list[str]) -> list[dict]:
        items = data if isinstance(data, list) else data.get("projects", []) \
            if isinstance(data, dict) else []
        out = []
        for proj in items:
            if not isinstance(proj, dict):
                continue
            name = (proj.get("name") or proj.get("uuid") or "project").strip()
            instructions = (proj.get("prompt_template") or proj.get("instructions")
                            or proj.get("custom_instructions") or "")
            docs = []
            for doc in (proj.get("docs") or proj.get("documents") or []):
                if isinstance(doc, dict):
                    text = doc.get("content") or doc.get("text") or ""
                    if text:
                        docs.append({"name": doc.get("filename") or doc.get("uuid") or "doc",
                                     "text": text})
            out.append({"name": name, "instructions": instructions, "docs": docs})
        return out

    def _parse_preferences(self, data: dict) -> dict:
        prefs = {}
        for key in ("custom_instructions", "preferences", "about_me", "role",
                    "conversation_preferences"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                prefs[key] = value.strip()
        return prefs

    # ---- storage / integration ------------------------------------------- #
    def _store_conversations(self, user_id: str, conversations: list[dict],
                             counts: dict, warnings: list[str], artifacts: dict) -> None:
        owner = _slug(user_id)
        for conv in conversations:
            # Conversation ids and KB paths are namespaced by owner so two users
            # importing the same Claude uuid never collide or clobber each other.
            conversation_id = f"claude-{owner}-{conv['source_id'][:24]}"
            # Dedup: if this user already has this imported conversation, skip.
            if not self.db.can_access_conversation(conversation_id, user_id):
                counts["duplicates"] += 1
                continue
            existing = self.db.get_messages(conversation_id, limit=1, user_id=user_id)
            if existing:
                counts["duplicates"] += 1
                continue
            body_parts = []
            for msg in conv["messages"]:
                self.db.add_message(conversation_id, msg["role"], msg["content"],
                                    user_id=user_id)
                counts["messages"] += 1
                who = "User" if msg["role"] == "user" else "Assistant"
                body_parts.append(f"## {who}\n{msg['content']}")
            try:
                self.db.set_conversation_title(conversation_id,
                                               f"[imported] {conv['title']}"[:120])
            except Exception:
                pass      # an untitled import is still a usable import
            counts["conversations"] += 1
            artifacts["conversations"].append(conversation_id)
            # Index the whole conversation as one retrievable knowledge document.
            doc_path = f"claude-history/{owner}/{conv['source_id'][:24]}.md"
            try:
                indexed = self.db.index_document(
                    doc_path,
                    (conv["title"] + "\n\n" + "\n\n".join(body_parts)),
                    title=f"[imported] {conv['title']}"[:120], user_id=user_id)
                if indexed:
                    counts["knowledge_docs"] += 1
                    artifacts["docs"].append(doc_path)
            except Exception as exc:
                warnings.append(f"could not index conversation {conv['source_id']}: {exc}")

    def _store_projects(self, user_id: str, projects: list[dict],
                        counts: dict, warnings: list[str], artifacts: dict) -> None:
        owner = _slug(user_id)
        for proj in projects:
            # Project instructions are reusable skills -> saved prompts.
            if proj.get("instructions"):
                try:
                    name = f"[imported] {proj['name']}"[:120]
                    # Owned by the importer: imported artefacts are private to
                    # them, and prompt names are unique per user, so two people
                    # importing the same project must not collide.
                    self.db.save_prompt(name, proj["instructions"], user_id=user_id)
                    counts["skills"] += 1
                    artifacts["prompts"].append(name)
                except Exception as exc:
                    warnings.append(f"could not save project prompt {proj['name']}: {exc}")
            for doc in proj.get("docs", []):
                try:
                    path = f"claude-projects/{owner}/{_slug(proj['name'])}/{_slug(doc['name'])}.md"
                    if self.db.index_document(path, doc["text"],
                                              title=doc["name"], user_id=user_id):
                        counts["knowledge_docs"] += 1
                        artifacts["docs"].append(path)
                except Exception as exc:
                    warnings.append(f"could not index project doc {doc.get('name')}: {exc}")

    def _store_preferences(self, user_id: str, prefs: dict,
                           counts: dict, warnings: list[str]) -> None:
        for key, value in prefs.items():
            try:
                self.db.remember(f"claude_import.{key}", value, user_id=user_id)
                counts["skills"] += 1
            except Exception as exc:
                warnings.append(f"could not store preference {key}: {exc}")

    def _attach_files(self, user_id: str, root: Path,
                      counts: dict, warnings: list[str], artifacts: dict) -> None:
        """Index supporting text artifacts (not the JSON we already parsed)."""
        owner = _slug(user_id)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                continue  # already handled by discovery
            counts["files"] += 1
            if path.suffix.lower() in _TEXT_EXT:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if not text.strip():
                    continue
                rel = path.relative_to(root)
                doc_path = f"claude-files/{owner}/{rel}"
                try:
                    if self.db.index_document(doc_path, text[:200000],
                                              title=path.name, user_id=user_id):
                        counts["knowledge_docs"] += 1
                        artifacts["docs"].append(doc_path)
                except Exception as exc:
                    warnings.append(f"could not index file {rel}: {exc}")

    def remove_import(self, import_id: str, user_id: str | None = None) -> bool:
        """Delete THIS import's record and exactly the conversations and KB docs
        it created — never another of the user's imports."""
        record = self.db.get_import(import_id)
        if not record:
            return False
        owner = record["user_id"]
        if user_id is not None and owner != user_id:
            return False
        artifacts = record.get("artifacts")
        if isinstance(artifacts, str):
            try:
                artifacts = json.loads(artifacts)
            except Exception:
                artifacts = None
        artifacts = artifacts or {}
        # Remove precisely the artifacts recorded for this import.
        for conversation_id in artifacts.get("conversations", []):
            self.db.clear_conversation(conversation_id, user_id=owner)
        for path in artifacts.get("docs", []):
            self.db.remove_document(path)
        for prompt_name in artifacts.get("prompts", []):
            # Scoped to the owner: an unscoped delete would take every user's
            # prompt of that name, and imported names are far from unique.
            try:
                self.db.delete_prompt(prompt_name, user_id=owner)
            except Exception:
                pass          # best effort: a missing prompt must not abort removal
        staging = self._staging(import_id)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        return self.db.delete_import(import_id, user_id=user_id)


class _BytesFile:
    """Minimal seekable wrapper so zipfile can sniff bytes without a temp file."""

    def __init__(self, data: bytes):
        import io
        self._io = io.BytesIO(data)

    def __getattr__(self, name):
        return getattr(self._io, name)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "x")).strip("-")[:60] or "x"


__all__ = [
    "ImportManager",
    "IMPORTS_DIR",
]
