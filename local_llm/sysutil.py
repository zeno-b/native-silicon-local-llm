"""Ports, log tailing, adapters and the model catalogue.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import *  # noqa: F401,F403


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



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ADAPTER_BASE_FILE',
    'DEFAULT_MODEL_CATALOG',
    'LOG_FILES',
    'StartupCancelled',
    'adapter_base',
    'adapter_fits',
    'adapter_ready',
    'add_if_supported',
    'get_free_port',
    'help_cmd',
    'hf_cache_dir',
    'list_adapters',
    'model_catalog',
    'model_is_cached',
    'port_open',
    'resolve_adapter',
    'resolve_adapter_quietly',
    'tail_log',
    'wait_for_model_ready',
    'wait_for_port',
    'write_adapter_base',
]
