"""Supervises the mlx_lm.server subprocess (start, stop, watchdog).

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


# Bytes read from the backend log per watchdog tick. Bounded so a burst of
# output cannot make one tick read megabytes.
_LOG_SCAN_CHUNK = 256 * 1024

# Failures that kill mlx_lm.server's single generation thread while leaving the
# HTTP server answering 200 on /v1/models and /health. There is no HTTP-visible
# symptom at all: completions simply hang forever. The traceback goes to the
# child's stdout/stderr, which is the log file below, and nowhere else.
_FATAL_BACKEND_PATTERNS = (
    re.compile(rb"Exception in thread [^\n]*\(_generate\)"),
    re.compile(rb"Command buffer execution failed"),
    re.compile(rb"kIOGPUCommandBufferCallbackErrorOutOfMemory"),
)


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
        *,
        # mlx-lm >= 0.29 memory levers. Keyword-only and defaulted so the
        # three-argument construction used by the tests and the selftest keeps
        # working unchanged.
        prefill_step_size: int = 0,
        prompt_cache_size: int = 0,
        prompt_cache_bytes: int = 0,
        decode_concurrency: int = 0,
        prompt_concurrency: int = 0,
        # Watchdog tuning. See _watchdog_tick.
        probe_interval: float = 30.0,
        probe_timeout: float = 25.0,
        probe_failures: int = 3,
        max_restarts: int = 3,
        restart_window: float = 600.0,
    ):
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.prompt_cache_dir = prompt_cache_dir
        self.prefill_step_size = prefill_step_size
        self.prompt_cache_size = prompt_cache_size
        self.prompt_cache_bytes = prompt_cache_bytes
        self.decode_concurrency = decode_concurrency
        self.prompt_concurrency = prompt_concurrency
        self.probe_interval = probe_interval
        self.probe_timeout = probe_timeout
        self.probe_failures = probe_failures
        self.max_restarts = max_restarts
        self.restart_window = restart_window
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
        # Watchdog state. _log_scan_pos is a byte offset into
        # logs/model_server.log: the child inherits that fd and writes to it
        # directly, so new bytes are proof the backend is doing work, and the
        # generation thread's traceback lands there and nowhere else.
        self._log_scan_pos = 0
        self._log_scan_tail = b""
        self._probe_failures = 0
        self._last_probe_at = 0.0
        self._probe_grace_until = 0.0
        self._restart_times: list[float] = []
        self._auto_restart_disabled = False

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
        # Count the memory levers that the installed build actually accepted.
        # Zero of them on an 8GB machine is worth saying out loud: it means the
        # server runs with no cap on prefill peak or on resident prompt caches,
        # which is how a marginal machine ends up with a dead generation thread.
        accepted: list[str] = []
        if self.kv_bits > 0:
            # Quantized KV cache roughly halves cache memory at 8 bits, which is
            # the difference between a usable context and swapping on 8GB.
            # Server support is genuinely unsettled across builds, so this is
            # best effort and the run continues without it.
            if add_if_supported(cmd, self._server_help, ["--kv-bits"], str(self.kv_bits)):
                accepted.append("--kv-bits")
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
        # --- mlx-lm >= 0.29 levers ----------------------------------------- #
        # These replaced --max-kv-size/--kv-bits and are the ones that exist on
        # a current build. --prefill-step-size is the important one: its default
        # of 2048 makes any shorter prompt a single forward pass, and the peak
        # of that pass is what fails the Metal command buffer on 8GB.
        if self.prefill_step_size > 0:
            if add_if_supported(cmd, self._server_help,
                                ["--prefill-step-size"], str(self.prefill_step_size)):
                accepted.append("--prefill-step-size")
        if self.prompt_cache_size > 0:
            if add_if_supported(cmd, self._server_help,
                                ["--prompt-cache-size"], str(self.prompt_cache_size)):
                accepted.append("--prompt-cache-size")
        if self.prompt_cache_bytes > 0:
            if add_if_supported(cmd, self._server_help,
                                ["--prompt-cache-bytes"], str(self.prompt_cache_bytes)):
                accepted.append("--prompt-cache-bytes")
        if self.decode_concurrency > 0:
            if add_if_supported(cmd, self._server_help,
                                ["--decode-concurrency"], str(self.decode_concurrency)):
                accepted.append("--decode-concurrency")
        if self.prompt_concurrency > 0:
            if add_if_supported(cmd, self._server_help,
                                ["--prompt-concurrency"], str(self.prompt_concurrency)):
                accepted.append("--prompt-concurrency")
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
        if accepted:
            log("Model server memory levers in effect: " + ", ".join(accepted), logging.DEBUG)
        elif self._server_help:
            log(
                "Installed mlx_lm.server accepted none of the memory-limit flags this "
                "build knows about, so prefill peak and resident prompt caches are "
                "uncapped. On a small machine that risks a Metal out-of-memory that "
                "kills the server's generation thread. Check `mlx_lm.server --help` "
                "for the flag names this version uses.",
                logging.WARNING,
            )
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
        """Auto-restart the model server when it stops being able to generate."""
        while not self._watchdog_stop.wait(5):
            try:
                self._watchdog_tick()
            except Exception as exc:
                # Never let the watchdog thread die: if it does, nothing restarts
                # a crashed model server for the rest of the process's life.
                log(f"Watchdog tick error (continuing): {exc}", logging.ERROR)

    def _watchdog_tick(self) -> None:
        """One health check, in three escalating signals.

        1. The process exited. Cheap, and the only thing this used to check.
        2. The backend log grew, or it contains a generation-thread traceback.
           Growth is free proof of life; a traceback is proof that no future
           request will ever be answered, because mlx_lm.server runs generation
           on ONE thread and does not restart it. This is the signal that
           matters: the process stays alive and keeps returning 200 from
           /v1/models and /health while every completion hangs forever.
        3. The log has gone quiet, so ask for one real token. A completion is
           the only evidence that generation works; no status endpoint reports
           it. Requires several consecutive failures so a slow server is not
           mistaken for a dead one.

        The probe runs with NO lock held: it can block for probe_timeout, and a
        stop() or a retrain must not queue behind it.
        """
        needs_probe = self._watchdog_cheap_checks()
        if not needs_probe:
            return
        self._watchdog_record_probe(self._probe_generation())

    def _watchdog_cheap_checks(self) -> bool:
        """Signals 1 and 2. Returns True when a live probe is warranted."""
        # A timed acquire, not a blocking one: a retrain holds this lock for
        # the length of a training run, and a watchdog parked on it would
        # never see its own stop event.
        if not self.lock.acquire(timeout=1):
            return False
        try:
            if self._watchdog_stop.is_set():
                return False
            if self.status != "ready" or self.proc is None:
                return False
            if self.proc.poll() is not None:
                self._watchdog_restart("model server process exited")
                return False

            advanced, fatal = self._scan_backend_log()
            if fatal:
                self._watchdog_restart(
                    f"model server generation thread died ({fatal})"
                )
                return False
            if advanced:
                # The backend is writing prefill progress, cache stats or access
                # lines, so it is demonstrably working. Nothing to probe.
                self._probe_failures = 0
                self._last_probe_at = time.time()
                return False

            now = time.time()
            if now < self._probe_grace_until:
                return False
            if now - self._last_probe_at < self.probe_interval:
                return False
            self._last_probe_at = now
            return True
        finally:
            self.lock.release()

    def _watchdog_record_probe(self, ok: bool) -> None:
        """Signal 3's verdict, applied under the lock."""
        if not self.lock.acquire(timeout=1):
            return
        try:
            if self._watchdog_stop.is_set() or self.status != "ready":
                return
            if ok:
                self._probe_failures = 0
                return
            self._probe_failures += 1
            log(
                "Watchdog: model server answered nothing to a one-token probe "
                f"({self._probe_failures}/{self.probe_failures}).",
                logging.WARNING,
            )
            if self._probe_failures >= self.probe_failures:
                self._watchdog_restart("model server is up but not generating")
        finally:
            self.lock.release()

    def _scan_backend_log(self) -> tuple[bool, str]:
        """(did the backend write since the last scan, fatal reason or "").

        The child inherits the log file descriptor and writes to it directly, so
        the byte offset is an honest activity counter that costs one stat().
        """
        path = LOG_DIR / "model_server.log"
        try:
            size = path.stat().st_size
        except OSError:
            return False, ""
        if size < self._log_scan_pos:      # rotated or truncated under us
            self._log_scan_pos = 0
            self._log_scan_tail = b""
        if size <= self._log_scan_pos:
            return False, ""
        try:
            with path.open("rb") as handle:
                handle.seek(self._log_scan_pos)
                chunk = handle.read(_LOG_SCAN_CHUNK)
        except OSError:
            return False, ""
        if not chunk:
            return False, ""
        self._log_scan_pos += len(chunk)
        # A pattern can straddle two reads, so carry a small tail across scans.
        window = self._log_scan_tail + chunk
        self._log_scan_tail = window[-512:]
        for pattern in _FATAL_BACKEND_PATTERNS:
            found = pattern.search(window)
            if found:
                return True, found.group(0).decode("utf-8", "replace").strip()
        return True, ""

    def _probe_generation(self) -> bool:
        """True only if the server actually produced a token.

        Synchronous on purpose: this runs on the watchdog thread, which has no
        event loop. /health returns a hard-coded {"status": "ok"} and
        /v1/models lists repositories, so neither proves anything about the
        generation thread. Only a completion does.
        """
        import httpx
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{self.model_port}/v1/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": "ok"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=self.probe_timeout,
            )
        except Exception:
            return False
        if resp.status_code != 200:
            return False
        try:
            return bool(resp.json().get("choices"))
        except Exception:
            return False

    def _watchdog_restart(self, reason: str) -> None:
        """Restart the backend, unless it has already failed too often.

        Caller holds the lock. Restarting forever into an immediate OOM is worse
        than stopping and saying so: the message a user gets should name the
        real fix rather than a spinner.
        """
        if self._auto_restart_disabled:
            return
        now = time.time()
        self._restart_times = [
            stamp for stamp in self._restart_times if now - stamp < self.restart_window
        ]
        if len(self._restart_times) >= self.max_restarts:
            self._auto_restart_disabled = True
            self.status = (
                f"error: the model server stopped generating {len(self._restart_times) + 1} "
                f"times in {int(self.restart_window)}s ({reason}). This is not "
                "transient. See logs/model_server.log, then lower CONTEXT_SIZE or "
                "PREFILL_STEP_SIZE, or choose a smaller model. Restart the model "
                "from Settings to try again."
            )
            log("Watchdog: " + self.status, logging.ERROR)
            return
        self._restart_times.append(now)
        log(
            f"Watchdog: {reason}; restarting model server (attempt "
            f"{len(self._restart_times)} of {self.max_restarts} within "
            f"{int(self.restart_window)}s).",
            logging.WARNING,
        )
        self.status = "restarting"
        # A wedged child is still ALIVE and still holding the port, unlike the
        # crashed-process case this watchdog originally handled. Reap it first,
        # or the replacement exits immediately with "address already in use".
        if self.proc is not None and self.proc.poll() is None:
            self._terminate_proc()
            time.sleep(0.5)      # let the OS release the listening socket
        try:
            self._start_internal()
        except StartupCancelled:
            log("Watchdog restart cancelled by an explicit stop.", logging.WARNING)
            self.status = "stopped"
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
            log_path = LOG_DIR / "model_server.log"
            self._log_file = open(log_path, "ab")
            # Baseline the watchdog's log scan at the end of the PREVIOUS child's
            # output, so a traceback already on disk cannot make the new child
            # look dead and start a restart loop.
            try:
                self._log_scan_pos = log_path.stat().st_size
            except OSError:
                self._log_scan_pos = 0
            self._log_scan_tail = b""
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
                    # A freshly loaded server is warm but idle, and the readiness
                    # wait already proved it generates. Skip probing for a beat so
                    # a restart is never immediately followed by another.
                    self._probe_failures = 0
                    self._last_probe_at = time.time()
                    self._probe_grace_until = time.time() + max(60.0, self.probe_interval)
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
            # An explicit start is the user overruling the watchdog's give-up:
            # clear the latch and the restart history so auto-restart works again.
            self._auto_restart_disabled = False
            self._restart_times = []
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
