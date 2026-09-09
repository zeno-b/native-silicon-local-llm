"""Structured, production-grade logging: JSON records, correlation IDs,
secret redaction, rotation, retention, and configurable chat-content levels.

This module upgrades the plain text logger set up in core.py into an
aggregation-ready pipeline without changing any existing ``log(...)`` call: it
reconfigures the shared ``local_llm`` logger in place, so every legacy call now
carries a correlation id and (optionally) JSON structure. New code emits richer
records through :func:`log_event` and the per-domain loggers
(``request``/``chat``/``tool``/``model``/``routing``).

Design notes
------------
* **Correlation id.** Bound per request in a ``contextvars.ContextVar`` so it
  follows the async task and threads spawned from it (when they copy context).
  Every record gets it via a logging ``Filter``.
* **Redaction.** A filter scrubs secrets from the rendered message of *every*
  record, and :func:`redact_text` / :func:`redact_obj` are reused for tool
  args/results and DB rows so nothing sensitive reaches disk.
* **Levels.** Standard ERROR/WARN/INFO/DEBUG plus a custom TRACE (5) for the
  most verbose operational tracing.
* **Rotation & retention.** Size-based ``RotatingFileHandler`` prevents a log
  from filling the disk; old rotations past the retention window are pruned at
  startup.
* **Chat content.** ``disabled`` / ``metadata`` / ``full`` decide whether user
  and assistant text is ever written, and content is always redacted+truncated.

Split-friendly: like the rest of the package it star-imports its dependencies so
the self-test's name-resolution check passes and the single-file bundle works.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403


# --------------------------------------------------------------------------- #
# TRACE level                                                                  #
# --------------------------------------------------------------------------- #
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Numeric level lookup that also understands "TRACE" and plain integers.
_LEVELS = {
    "TRACE": TRACE, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
    "WARN": logging.WARNING, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
}


def level_from_name(name: str | int | None, default: int = logging.INFO) -> int:
    """Resolve a level name/number to a logging level, tolerating junk."""
    if isinstance(name, int):
        return name
    if not name:
        return default
    text = str(name).strip().upper()
    if text.isdigit():
        return int(text)
    return _LEVELS.get(text, default)


# --------------------------------------------------------------------------- #
# Correlation id + bound structured context                                   #
# --------------------------------------------------------------------------- #
# The id that ties every log line, metric and routing decision of one request
# together. Also surfaced to the client as a response header so a user can quote
# it in a bug report and an admin can grep for it.
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None)
# Extra structured fields bound for the duration of a request (user id, route,
# chat id, node, ...). Kept immutable-by-copy so nested binds do not leak out.
_log_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "log_context", default={})


# The id of the user on whose behalf the current request is acting. Bound by the
# auth dependency and read by request-scoped code (RAG retrieval, memory tools)
# so per-user data scoping works without threading user_id through the whole
# agent. contextvars are per-async-task and are copied into asyncio.to_thread
# workers, so this is race-free across concurrent requests.
_acting_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "acting_user", default=None)


def set_acting_user(user_id: str | None) -> None:
    _acting_user.set(user_id)


def get_acting_user() -> str | None:
    return _acting_user.get()


# Which conversation the current unit of work belongs to. Bound per tool call in
# a contextvar (not on the shared registry instance) so concurrent requests --
# each running its tools in its own ``asyncio.to_thread`` copied context -- never
# clobber each other's conversation id.
_acting_conversation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "acting_conversation", default=None)


def set_acting_conversation(conversation_id: str | None) -> None:
    _acting_conversation.set(conversation_id)


def get_acting_conversation() -> str | None:
    return _acting_conversation.get()


def new_correlation_id() -> str:
    """A fresh short, URL-safe correlation id."""
    return uuid.uuid4().hex[:16]


def set_correlation_id(cid: str | None) -> str:
    """Bind (or clear) the correlation id for the current context. Returns it."""
    cid = cid or new_correlation_id()
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def bind_context(**fields: Any) -> dict:
    """Merge structured fields into the current logging context.

    Returns the previous context so a caller can restore it (the FastAPI
    middleware resets both this and the correlation id after each request).
    """
    current = dict(_log_context.get() or {})
    previous = current
    merged = dict(current)
    for key, value in fields.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    _log_context.set(merged)
    return previous


def reset_context(previous: dict | None = None) -> None:
    _log_context.set(dict(previous or {}))


def clear_context() -> None:
    _correlation_id.set(None)
    _log_context.set({})


# --------------------------------------------------------------------------- #
# Redaction                                                                    #
# --------------------------------------------------------------------------- #
_REDACT = "***redacted***"

# key: value / key = value where the key looks sensitive.
_KV_SECRET = re.compile(
    r"""(?ix)
    \b(
      authorization | api[-_ ]?key | access[-_ ]?token | refresh[-_ ]?token |
      id[-_ ]?token | client[-_ ]?secret | secret | password | passwd | pwd |
      cookie | set-cookie | session | bearer | token | private[-_ ]?key |
      x-api-key | node[-_ ]?token
    )
    (\s*[:=]\s*|"\s*:\s*"?|'\s*:\s*'?)
    ([^\s,;"'}\]]+)
    """
)
# Well-known token shapes, redacted wherever they appear.
_TOKEN_SHAPES = [
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),            # OpenAI-style
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),   # Slack
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),           # GitHub PAT
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}"),  # JWT
    re.compile(r"(?i)\baws_secret_access_key\b\S*"),
]
# Keys whose *value* must always be dropped when redacting a dict/JSON object.
_SECRET_KEYS = {
    "authorization", "api_key", "apikey", "access_token", "refresh_token",
    "id_token", "client_secret", "secret", "password", "passwd", "pwd",
    "cookie", "set-cookie", "session", "session_token", "bearer", "token",
    "private_key", "x-api-key", "node_token", "password_hash", "salt",
    "tavily_api_key", "brave_api_key",
}


def redact_text(text: str | None) -> str:
    """Scrub secrets from a free-text string. Safe on any input."""
    if not text:
        return "" if text is None else text
    out = str(text)
    out = _KV_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACT}", out)
    for pattern in _TOKEN_SHAPES:
        out = pattern.sub(_REDACT, out)
    return out


def redact_obj(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact a JSON-ish structure: secret keys are dropped and
    string values are scrubbed. Bounded depth so a cyclic/huge object cannot
    wedge logging."""
    if _depth > 6:
        return "..."
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if str(key).lower() in _SECRET_KEYS:
                out[key] = _REDACT
            else:
                out[key] = redact_obj(value, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, _depth + 1) for v in obj][:200]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


class RedactionFilter(logging.Filter):
    """Redact the fully-rendered message and any string args of every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact_text(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: (redact_text(v) if isinstance(v, str) else v)
                                   for k, v in record.args.items()}
                else:
                    record.args = tuple(
                        redact_text(a) if isinstance(a, str) else a
                        for a in record.args)
        except Exception:
            # Logging must never raise; if redaction trips, drop the message
            # body rather than risk leaking or crashing.
            record.msg = _REDACT
            record.args = ()
        return True


class ContextFilter(logging.Filter):
    """Attach the correlation id and bound context to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        record.context = dict(_log_context.get() or {})
        # Promote a few common context fields to top-level attributes so both
        # the JSON and text formatters can show them without extra work.
        ctx = record.context
        record.user_id = ctx.get("user_id")
        record.route = ctx.get("route")
        record.event = getattr(record, "event", ctx.get("event"))
        return True


# --------------------------------------------------------------------------- #
# Formatters                                                                   #
# --------------------------------------------------------------------------- #
# The attributes a bare LogRecord already has. A structured field that collides
# with one of these cannot be passed via extra= (logging refuses to overwrite),
# so log_event renames such keys. This set is also used by the JSON formatter to
# skip standard attributes when promoting extras.
_RECORD_ATTRS = set(vars(logging.makeLogRecord({})).keys())
# Standard LogRecord attributes we never want to duplicate into the JSON "extra".
_RESERVED = _RECORD_ATTRS | {
    "correlation_id", "context", "user_id", "route", "event", "message",
    "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, aggregation-ready (Loki/ELK/Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = getattr(record, "correlation_id", None)
        if cid:
            payload["correlation_id"] = cid
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        ctx = getattr(record, "context", None)
        if ctx:
            payload.update(ctx)
        # Any explicit extra=... fields on the record.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable, correlation-id-aware text for local development."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", None)
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        head = f"[{stamp}] [local-llm]"
        if cid:
            head += f" [{cid}]"
        line = f"{head} {record.levelname}: {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
_CONFIGURED = {"done": False, "chat_content": "metadata", "level": logging.INFO}


def _log_dir(config: Any = None) -> Path:
    override = ""
    if config is not None:
        override = str(getattr(config, "log_dir", "") or "")
    override = override or os.environ.get("LOG_DIR", "")
    return Path(override).expanduser() if override else LOG_DIR


def configure_logging(config: Any = None, *, force: bool = False) -> None:
    """Install the structured logging pipeline on the shared ``local_llm`` logger.

    Idempotent: safe to call more than once (e.g. tests, a re-exec). Reads its
    settings from ``config`` when present, falling back to environment variables
    and then sane defaults, so it works before the Config fields exist too.
    """
    if _CONFIGURED["done"] and not force:
        return

    def cfg(attr: str, env: str, default: str) -> str:
        if config is not None:
            value = getattr(config, attr, None)
            if value is not None and str(value) != "":
                return str(value)
        return os.environ.get(env, default)

    level = level_from_name(cfg("log_level", "LOG_LEVEL", "INFO"))
    fmt = cfg("log_format", "LOG_FORMAT", "json").strip().lower()
    chat_content = cfg("log_chat_content", "LOG_CHAT_CONTENT", "metadata").strip().lower()
    if chat_content not in ("disabled", "metadata", "full"):
        chat_content = "metadata"
    try:
        max_bytes = int(cfg("log_max_bytes", "LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    except ValueError:
        max_bytes = 10 * 1024 * 1024
    try:
        backup_count = int(cfg("log_backup_count", "LOG_BACKUP_COUNT", "10"))
    except ValueError:
        backup_count = 10
    try:
        retention_days = int(cfg("log_retention_days", "LOG_RETENTION_DAYS", "14"))
    except ValueError:
        retention_days = 14

    _CONFIGURED["chat_content"] = chat_content
    _CONFIGURED["level"] = level

    log_dir = _log_dir(config)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass          # read-only volume: keep console logging rather than dying

    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else TextFormatter()
    redaction = RedactionFilter()
    context = ContextFilter()

    # Reconfigure the package logger in place. propagate=False so records do not
    # also hit the root handler installed by core's basicConfig (double logging).
    root = logging.getLogger("local_llm")
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(redaction)
    console.addFilter(context)
    root.addHandler(console)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=max(0, max_bytes),
            backupCount=max(0, backup_count), encoding="utf-8", delay=True)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction)
        file_handler.addFilter(context)
        root.addHandler(file_handler)
    except Exception as exc:  # pragma: no cover - disk/perm issues are non-fatal
        root.warning("could not open app.log for rotation: %s", exc)

    # Quiet chatty third-party loggers that would otherwise emit an INFO line
    # per outbound call (httpx logs every request; hpack/httpcore are verbose at
    # DEBUG). Their warnings/errors still surface.
    for _noisy in ("httpx", "httpcore", "hpack", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    _prune_old_logs(log_dir, retention_days)
    _CONFIGURED["done"] = True
    log_event(get_logger("boot"), logging.INFO, "logging.configured",
              log_format=fmt, configured_level=logging.getLevelName(level),
              chat_content=chat_content, dir=str(log_dir),
              max_bytes=max_bytes, backup_count=backup_count,
              retention_days=retention_days)


def _prune_old_logs(log_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 86400
    try:
        for path in log_dir.glob("*.log*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
    except Exception:
        pass          # pruning is housekeeping; never let it block startup


def chat_content_level() -> str:
    return _CONFIGURED["chat_content"]


# --------------------------------------------------------------------------- #
# Emit helpers                                                                 #
# --------------------------------------------------------------------------- #
def get_logger(domain: str) -> logging.Logger:
    """A child of the package logger, e.g. ``local_llm.chat``."""
    return logging.getLogger(f"local_llm.{domain}")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured record: an ``event`` name plus arbitrary fields.

    Field values are redacted defensively; the JSON formatter promotes them to
    top-level keys and the text formatter shows the event name in the message.
    """
    # Redact every field value (secret-named keys are dropped, strings scrubbed)
    # so nothing sensitive reaches the JSON "extra" columns, not just the message.
    # Rename any key that collides with a standard LogRecord attribute (e.g.
    # "filename", "module", "name"): logging refuses to overwrite those.
    safe = {}
    for key, value in redact_obj(dict(fields)).items():
        if key in _RECORD_ATTRS or key in ("message", "asctime"):
            key = key + "_"
        safe[key] = value
    msg = event
    if safe:
        # A compact human hint for the text formatter; JSON gets the real fields.
        hint = " ".join(f"{k}={v}" for k, v in safe.items()
                        if isinstance(v, (str, int, float, bool)))
        if hint:
            msg = f"{event} {hint}"
    logger.log(level, msg, extra={"event": event, **safe})


def trace(logger: logging.Logger, msg: str, **fields: Any) -> None:
    if fields:
        log_event(logger, TRACE, msg, **fields)
    else:
        logger.log(TRACE, msg)


def content_for_log(text: str | None, *, force_level: str | None = None) -> Any:
    """Render chat content for logging per the configured content level.

    Returns ``None`` when content logging is disabled, a small metadata dict
    (length + sha-ish fingerprint, never the text) at ``metadata``, and the
    redacted, truncated text at ``full``. Never returns credentials.
    """
    level = force_level or _CONFIGURED["chat_content"]
    if not text:
        text = ""
    if level == "disabled":
        return None
    if level == "metadata":
        return {"chars": len(text), "fingerprint": _fingerprint(text)}
    # full
    redacted = redact_text(text)
    if len(redacted) > 8000:
        redacted = redacted[:8000] + f"...[+{len(redacted) - 8000} chars]"
    return redacted


def _fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


__all__ = [
    "TRACE",
    "JsonFormatter",
    "TextFormatter",
    "RedactionFilter",
    "ContextFilter",
    "level_from_name",
    "new_correlation_id",
    "set_correlation_id",
    "get_correlation_id",
    "set_acting_user",
    "get_acting_user",
    "set_acting_conversation",
    "get_acting_conversation",
    "bind_context",
    "reset_context",
    "clear_context",
    "redact_text",
    "redact_obj",
    "configure_logging",
    "chat_content_level",
    "content_for_log",
    "get_logger",
    "log_event",
    "trace",
]
