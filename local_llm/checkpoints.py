"""Undo for a turn that edited files.

The agent already reported which files it touched and showed you a git diff, but
"here is what I broke" is not the same as "put it back". This records the
ORIGINAL bytes of every file a turn is about to modify, before the write, so one
call restores the tree to how it was.

Copy-on-write per file rather than a snapshot of the whole directory. Snapshotting
the working tree costs the size of the project on every turn that writes a single
line, and would be unusable on a real repository; copying only what is about to
change costs exactly what was at risk. It also works in a directory that is not a
git repository, which `git checkout` does not.

A checkpoint is a directory holding a manifest and the original bytes:

    data/checkpoints/<id>/manifest.json
    data/checkpoints/<id>/blobs/0001

Files that did NOT exist before the turn are recorded with no blob, and rolling
back deletes them. That distinction is the whole reason a manifest exists rather
than just a pile of copies.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .core import *  # noqa: F401,F403


@dataclass
class Checkpoint:
    """One turn's worth of undo."""

    id: str
    root: str
    created_at: str
    conversation_id: str | None
    entries: list[dict]

    @property
    def files(self) -> list[str]:
        return [entry["path"] for entry in self.entries]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "root": self.root,
            "created_at": self.created_at,
            "conversation_id": self.conversation_id,
            "files": self.files,
            "created": [e["path"] for e in self.entries if not e.get("existed")],
            "modified": [e["path"] for e in self.entries if e.get("existed")],
        }


class CheckpointStore:
    """Records pre-write state and puts it back.

    One store per process. Deliberately holds no open handles and keeps nothing
    in memory between calls: a rollback may well be the thing you reach for
    after the process that made the mess has been restarted.
    """

    def __init__(self, root: Path | str, keep: int = 20, max_file_bytes: int = 4_000_000):
        self.root = Path(root)
        self.keep = max(1, int(keep))
        # A file bigger than this is recorded as "not captured" rather than
        # copied. Silently copying a 2GB artefact on every turn would be worse
        # than admitting one file is not covered.
        self.max_file_bytes = max(0, int(max_file_bytes))

    # ----------------------------------------------------------- recording --
    def _dir(self, checkpoint_id: str) -> Path | None:
        """The directory for an id, or None when the id is not safe.

        Checkpoint ids reach here from an API path parameter, so the same rule
        applies as everywhere else a caller-supplied string becomes a path.
        """
        slug = "".join(c for c in str(checkpoint_id or "") if c.isalnum() or c in "-_")
        if not slug or slug != str(checkpoint_id):
            return None
        candidate = (self.root / slug).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def new_id(self, conversation_id: str | None = None) -> str:
        stem = "".join(c for c in str(conversation_id or "adhoc")
                       if c.isalnum() or c in "-_")[:24] or "adhoc"
        return f"{int(time.time() * 1000)}-{stem}"

    def capture(self, checkpoint_id: str, project_root: Path, target: Path,
                conversation_id: str | None = None) -> None:
        """Record `target`'s current state, once per checkpoint.

        Called BEFORE the write. Idempotent per file: the first capture in a
        turn is the one that matters, because that is the state the turn
        started from. Never raises -- a failed capture must not stop the edit
        the user asked for, it just means that file is not covered, and the
        manifest says so.
        """
        directory = self._dir(checkpoint_id)
        if directory is None:
            return
        try:
            rel = str(target.resolve().relative_to(project_root.resolve()))
        except Exception:
            rel = target.name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            manifest = self._read_manifest(manifest_path) or {
                "root": str(project_root.resolve()),
                "created_at": iso(utc_now()),
                "conversation_id": conversation_id,
                "entries": [],
            }
            if any(entry["path"] == rel for entry in manifest["entries"]):
                return                       # already captured this turn
            entry: dict = {"path": rel, "existed": target.is_file()}
            if entry["existed"]:
                size = target.stat().st_size
                if size > self.max_file_bytes:
                    entry["captured"] = False
                    entry["reason"] = f"{size} bytes is over the checkpoint limit"
                else:
                    blobs = directory / "blobs"
                    blobs.mkdir(exist_ok=True)
                    blob = blobs / f"{len(manifest['entries']):04d}"
                    shutil.copy2(target, blob)
                    entry["captured"] = True
                    entry["blob"] = blob.name
            else:
                entry["captured"] = True     # nothing to copy; rollback deletes
            manifest["entries"].append(entry)
            self._write_manifest(manifest_path, manifest)
        except Exception as exc:
            log(f"Could not checkpoint {target}: {exc}", logging.WARNING)

    @staticmethod
    def _read_manifest(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and "entries" in data else None
        except Exception:
            return None

    @staticmethod
    def _write_manifest(path: Path, manifest: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -------------------------------------------------------------- reading --
    def get(self, checkpoint_id: str) -> Checkpoint | None:
        directory = self._dir(checkpoint_id)
        if directory is None:
            return None
        manifest = self._read_manifest(directory / "manifest.json")
        if manifest is None:
            return None
        return Checkpoint(
            id=directory.name,
            root=str(manifest.get("root") or ""),
            created_at=str(manifest.get("created_at") or ""),
            conversation_id=manifest.get("conversation_id"),
            entries=list(manifest.get("entries") or []),
        )

    def list(self, limit: int = 20) -> list[Checkpoint]:
        """Newest first. Ids start with a millisecond timestamp, so name order
        is time order without reading a single manifest to sort."""
        if not self.root.exists():
            return []
        out: list[Checkpoint] = []
        for entry in sorted(self.root.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            found = self.get(entry.name)
            if found is not None:
                out.append(found)
            if len(out) >= max(1, limit):
                break
        return out

    # ------------------------------------------------------------- undoing --
    def rollback(self, checkpoint_id: str) -> tuple[list[str], list[str]]:
        """Put the files back. Returns (restored, problems).

        Deliberately does NOT delete the checkpoint afterwards. A rollback that
        turns out to be the wrong call should itself be recoverable, and the
        retention limit will clear it soon enough.
        """
        checkpoint = self.get(checkpoint_id)
        directory = self._dir(checkpoint_id)
        if checkpoint is None or directory is None:
            return [], [f"no checkpoint {checkpoint_id!r}"]
        project_root = Path(checkpoint.root)
        restored: list[str] = []
        problems: list[str] = []
        for entry in checkpoint.entries:
            rel = entry.get("path") or ""
            target = project_root / rel
            try:
                # Containment again: the manifest is a file on disk and could
                # have been edited between capture and rollback.
                resolved = target.resolve()
                if (resolved != project_root.resolve()
                        and project_root.resolve() not in resolved.parents):
                    problems.append(f"{rel}: outside the recorded project root")
                    continue
                if not entry.get("captured"):
                    problems.append(f"{rel}: was not captured "
                                    f"({entry.get('reason') or 'unknown reason'})")
                    continue
                if entry.get("existed"):
                    blob = directory / "blobs" / str(entry.get("blob") or "")
                    if not blob.is_file():
                        problems.append(f"{rel}: its saved copy is missing")
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(blob, target)
                    restored.append(rel)
                else:
                    # The turn created it; putting it back means removing it.
                    if target.is_file():
                        target.unlink()
                    restored.append(rel)
            except Exception as exc:
                problems.append(f"{rel}: {type(exc).__name__}: {exc}")
        return restored, problems

    def prune(self, reserve: int = 0) -> int:
        """Drop all but the newest `keep` checkpoints. Returns how many went.

        `reserve` leaves room for checkpoints about to be created. Callers prune
        at the START of a turn, which is once per turn instead of once per file
        written; without a reserve of one, the fresh checkpoint then makes
        `keep + 1`, so the documented limit would be quietly wrong by one.
        """
        if not self.root.exists():
            return 0
        directories = sorted((d for d in self.root.iterdir() if d.is_dir()),
                             reverse=True)
        limit = max(0, self.keep - max(0, int(reserve)))
        removed = 0
        for stale in directories[limit:]:
            try:
                shutil.rmtree(stale, ignore_errors=True)
                removed += 1
            except Exception:
                pass
        return removed


__all__ = [
    'Checkpoint',
    'CheckpointStore',
]
