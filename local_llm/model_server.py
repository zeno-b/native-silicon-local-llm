"""Supervises the mlx_lm.server subprocess (start, stop, watchdog).

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


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
        # Cache the last live health-probe result so the UI's 3-second poll does
        # not trigger a real inference on the model server every time.
        self._probe_cache: tuple[float, bool] = (0.0, False)

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
            try:
                self._watchdog_tick()
            except Exception as exc:
                # Never let the watchdog thread die: if it does, nothing restarts
                # a crashed model server for the rest of the process's life.
                log(f"Watchdog tick error (continuing): {exc}", logging.ERROR)

    def _watchdog_tick(self) -> None:
        # A timed acquire, not a blocking one: a retrain holds this lock for
        # the length of a training run, and a watchdog parked on it would
        # never see its own stop event.
        if not self.lock.acquire(timeout=1):
            return
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
                pass      # best effort on shutdown; the OS closes it regardless
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

    async def health_probe_cached(self, max_age: float = 15.0) -> bool:
        """A cheap health signal for frequent polling (e.g. the UI status bar).

        Returns the last live-probe result while it is fresher than `max_age`
        seconds, and only runs a real one-token completion when the cache is
        stale. This turns a per-3-second inference into one every ~15 seconds,
        so the status bar no longer competes with real chat for the model.
        """
        now = time.time()
        cached_at, value = self._probe_cache
        if now - cached_at < max_age:
            return value
        value = await self.health_probe() if self.is_alive() else False
        self._probe_cache = (time.time(), value)
        return value



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ModelServerManager',
]
