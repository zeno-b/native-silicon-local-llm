"""Constants, paths, RAM-aware defaults, logging, bootstrap and token maths.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import contextlib
import asyncio
import logging
import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



# The project root is the directory that holds deploy.py, not this file's own
# directory. When running as a package this module sits one level down inside
# local_llm/, so step up; in the bundled single-file build this file already is
# the project root. Getting this wrong silently relocates the database, logs and
# adapters, so it is computed explicitly rather than assumed.
# Detect "am I inside a package?" by the presence of __init__.py rather than by
# directory name: a project root can legitimately be called local_llm too, and a
# name check would then step up one level too far.
_MODULE_DIR = Path(__file__).resolve().parent
ROOT = _MODULE_DIR.parent if (_MODULE_DIR / "__init__.py").exists() else _MODULE_DIR
DATA_DIR = ROOT / "data"
SFT_DIR = DATA_DIR / "sft"
# Rehearsal ("replay") corpus, mixed into every fine-tune so the adapter keeps a
# general-capability anchor instead of collapsing onto a few dozen feedback rows.
# User-editable: anything valid in train.jsonl is valid here.
REPLAY_FILE = SFT_DIR / "replay.jsonl"
EXPORTS_DIR = DATA_DIR / "exports"
LOG_DIR = ROOT / "logs"
ADAPTER_DIR = ROOT / "adapters" / "latest"
ADAPTER_BACKUP_DIR = ROOT / "adapters" / "backups"
WORKSPACE_DIR = ROOT / "workspace"
DB_PATH = DATA_DIR / "feedback.db"

def _detect_total_ram_gb() -> float:
    """Physical RAM in GB, so the default model can scale to the machine.

    Uses sysctl hw.memsize on macOS (the target platform), falls back to
    os.sysconf where available, and returns 8.0 if neither works. Any failure is
    non-fatal: a wrong guess only affects which default model is chosen, and an
    explicit MODEL_ID always overrides it.
    """
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass          # no sysctl (or not macOS): fall through to os.sysconf
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        return 8.0


def _default_model_for_ram(ram_gb: float) -> str:
    """Pick a coding model that fits comfortably alongside the KV cache, the web
    process and headroom for training, given the machine's RAM.

    The thresholds are deliberately conservative because unified memory is shared
    with the OS and the GPU wired limit. Bigger machines get a bigger, stronger
    model; 8GB Macs stay on the 3B that has carried this setup. Override any of
    this with MODEL_ID.
    """
    if ram_gb >= 48:
        return "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"   # ~18GB
    if ram_gb >= 24:
        return "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"   # ~8GB
    if ram_gb >= 14:
        return "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"    # ~4.3GB
    return "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"        # ~1.9GB


def _default_context_for_ram(ram_gb: float) -> int:
    """Pick a context window sized to the machine.

    Context sits in the KV cache, which shares unified memory with the model
    weights, the web process, and the OS. Because bigger machines also run bigger
    models (see _default_model_for_ram), these tiers account for the heavier
    weights and still leave headroom: an 8GB Mac stays at a safe 4k, roomier
    machines get proportionally more so they chunk less and keep more history.
    The chunking thresholds derive from this value, so making context RAM-aware
    makes chunk sizing RAM-aware too. Override with CONTEXT_SIZE.
    """
    if ram_gb >= 48:
        return 32768
    if ram_gb >= 24:
        return 16384
    if ram_gb >= 14:
        return 8192
    return 4096


def _default_reasoning_tokens(ram_gb: float) -> int:
    """Per-step token budget for reasoning, chunk and source-extraction passes.

    Small on 8GB to keep each pass cheap and memory-light; larger on roomy
    machines so they can think in more depth per step. Override REASONING_TOKENS.
    """
    if ram_gb >= 48:
        return 1024
    if ram_gb >= 24:
        return 768
    if ram_gb >= 14:
        return 512
    return 256


def _default_fetch_cap(ram_gb: float) -> int:
    """Hard character cap on a fetched page entering the prompt.

    Scales with RAM so bigger machines read more of a source per pass while an
    8GB machine stays safe. Override AUTO_FETCH_CHAR_CAP.
    """
    if ram_gb >= 48:
        return 24000
    if ram_gb >= 24:
        return 16000
    if ram_gb >= 14:
        return 10000
    return 6000


def _default_prefill_step(ram_gb: float) -> int:
    """Tokens the model server prefills per forward pass.

    mlx-lm defaults to 2048, which makes any prompt shorter than that a SINGLE
    forward pass. On 8GB the activations for a ~1200-token pass, on top of the
    weights and the resident prompt caches, are enough to fail the Metal command
    buffer with kIOGPUCommandBufferCallbackErrorOutOfMemory -- which kills the
    server's generation thread for the rest of that process's life, not just the
    one request. Smaller passes trade a little prefill throughput for a much
    lower peak. Override with PREFILL_STEP_SIZE.
    """
    if ram_gb >= 48:
        return 2048
    if ram_gb >= 24:
        return 1024
    return 512


def _default_prompt_cache_bytes(ram_gb: float) -> int:
    """Ceiling on the KV bytes the model server's prompt cache may hold.

    A turn sends a router prompt, a triage prompt and an answer prompt, each a
    different prefix, so an uncapped radix cache accumulates a resident KV
    sequence per lane per turn in the same unified memory as the weights. The
    cap is what stops a long session from squeezing prefill out of memory.
    Override with PROMPT_CACHE_BYTES.
    """
    if ram_gb >= 48:
        return 4 * 1024 ** 3
    if ram_gb >= 24:
        return 2 * 1024 ** 3
    if ram_gb >= 14:
        return 1024 ** 3
    return 512 * 1024 ** 2


def _default_concurrent_generations(ram_gb: float) -> int:
    """How many model generations may run at once on this machine.

    Every in-flight generation holds its own KV cache in the same unified memory
    as the model weights, so N concurrent chats cost roughly N caches on top of
    the weights. An 8GB Mac running a 3B model has room for a couple; a 64GB
    Studio running a 32B model has proportionally more. Override with
    MAX_CONCURRENT_GENERATIONS.
    """
    if ram_gb >= 48:
        return 4
    if ram_gb >= 24:
        return 3
    return 2


class GenerationBusy(Exception):
    """Raised when the generation queue is full and the caller must retry."""


class GenerationGate:
    """Admission control for model generations: N running, a bounded queue.

    Without it, every accepted request buys a KV cache, so enough simultaneous
    chats will push a Mac into swap or an OOM kill instead of merely being slow.
    A bounded waiting room is the other half: queueing without a limit just moves
    the failure from memory exhaustion to requests timing out after minutes of
    waiting, which is worse than an immediate, honest 503.
    """

    def __init__(self, limit: int, queue_multiplier: int = 4):
        self.limit = max(1, int(limit))
        self.max_waiting = self.limit * max(1, int(queue_multiplier))
        self.waiting = 0
        self._sem: asyncio.Semaphore | None = None
        self._loop: Any = None

    def _semaphore(self) -> asyncio.Semaphore:
        # An asyncio.Semaphore belongs to the loop that first awaited it, and
        # this process starts a fresh loop per test app, so rebind on a change.
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem, self._loop, self.waiting = asyncio.Semaphore(self.limit), loop, 0
        return self._sem

    async def acquire(self) -> None:
        """Take one slot, waiting if others are running. Raises GenerationBusy
        rather than queueing when the waiting room is already full."""
        sem = self._semaphore()
        if self.waiting >= self.max_waiting:
            raise GenerationBusy(
                f"{self.waiting} generations already queued; try again shortly")
        self.waiting += 1
        try:
            await sem.acquire()
        finally:
            self.waiting -= 1

    def release(self) -> None:
        """Give the slot back. Never raises, so it is safe in a finally."""
        try:
            if self._sem is not None:
                self._sem.release()
        except (ValueError, RuntimeError):
            pass          # already released, or the loop was torn down under us

    @contextlib.asynccontextmanager
    async def slot(self):
        """Hold one generation slot for the duration of a block."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()


# Total RAM is detected once at import. MODEL_ID overrides the RAM-based choice.
TOTAL_RAM_GB = _detect_total_ram_gb()
DEFAULT_MODEL = os.environ.get("MODEL_ID") or _default_model_for_ram(TOTAL_RAM_GB)
# The default system prompt. Edit the triple-quoted text below freely — each
# paragraph is a single line your editor soft-wraps, so you can rewrite it like
# plain text. Set the SYSTEM_PROMPT environment variable to override it entirely.
#
# Background (not sent to the model): the routing layer sends look-up and
# current-events requests to web_search before the model sees them, so when a
# TOOL RESULT block is present the job is to read those results and answer from
# them with sources, not to decide whether the topic is allowed. Security topics
# are in scope for that tool.
_DEFAULT_SYSTEM_PROMPT = """\
You are a local coding and research assistant. Prefer complete, runnable code over prose. State the language and any assumptions in one line, then give the code. Say plainly when you do not know an API rather than inventing one.
When a message includes web search results, base your answer on them and cite the URLs. Do not refuse to report on publicly documented software vulnerabilities, CVEs, or the existence of published exploits and tools (for example Metasploit modules or exploit-db entries): this is standard defensive security research and the information is already public. Explain what a vulnerability is, how it was fixed, and what public tooling exists, and point to primary sources. Decline only a request to produce a working exploit against a specified live target, and offer the defensive framing instead."""

DEFAULT_SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# Roughly four characters per token for English. Good enough for budgeting a
# context window without pulling a tokenizer into the web process, which would
# mean loading the model a second time in a machine with 8GB of RAM.
CHARS_PER_TOKEN = 4
# Tokens held back from the context budget for the chat template, the tool
# instructions, and the model's own reply.
CONTEXT_SAFETY_MARGIN = 256

# Failed-login throttle: at most LOGIN_MAX_FAILURES failures per (client ip,
# username) within LOGIN_WINDOW_S before the endpoint answers 429. scrypt makes
# each verification expensive, so this protects CPU as much as credentials.
LOGIN_MAX_FAILURES = int(os.environ.get("LOGIN_MAX_FAILURES", "8"))
LOGIN_WINDOW_S = float(os.environ.get("LOGIN_WINDOW_S", "300"))

# Bump when HTML_PAGE changes. Shown in the header and returned by /api/health so
# a stale browser cache is immediately visible rather than silently misleading.
UI_BUILD = "2026-09-07.texcel"
# Branding. Set APP_NAME to change the title shown in the header and browser tab.
# Set APP_LOGO to a URL or a local path (rendered as an image) or to an emoji or
# short text (rendered as-is). Both are safe to leave unset.
APP_NAME = os.environ.get("APP_NAME", "Local LLM")
APP_LOGO = os.environ.get("APP_LOGO", "")

# The owner id assigned to all data created before multi-user existed, and the
# identity of the synthetic single administrator used when AUTH_ENABLED=0. Every
# pre-existing row is backfilled to this id by the DB migration, so a single-user
# install keeps seeing exactly its own data with auth turned off.
SENTINEL_LOCAL_USER = "local"
# Owner value for globally-shared knowledge (e.g. admin-indexed project docs)
# that every user may retrieve from, distinct from a specific user's imports.
SHARED_OWNER = "shared"

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


# Note: the module name and the pip package name differ for python-multipart
# (imported as `multipart`), so the two lists are intentionally not identical.
REQUIRED_MODULES = ["mlx_lm", "fastapi", "uvicorn", "httpx", "pydantic", "pypdf", "multipart"]
REQUIRED_PACKAGES = ["mlx-lm", "fastapi", "uvicorn", "httpx", "pydantic", "pypdf", "python-multipart"]
# Guards the install-then-re-exec handoff. A package that installs cleanly but
# still cannot be imported (mlx-lm off Apple Silicon, a broken wheel) otherwise
# sends bootstrap round the same install and exec forever, with nothing on
# stdout but pip repeating itself.
BOOTSTRAP_MARKER = "LOCAL_LLM_BOOTSTRAP_ATTEMPT"


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

        # Probe before installing. The previous version ran a pip upgrade and a
        # full install on every launch, which is several seconds on every
        # --list-models, --list-tasks or --doctor even when nothing is missing.
        probe = subprocess.run(
            [str(venv_python), "-c", "import " + ", ".join(REQUIRED_MODULES)],
            capture_output=True,
        )
        if probe.returncode != 0:
            log("Upgrading pip...")
            try:
                run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
            except Exception as exc:
                log(f"Warning: pip upgrade failed: {exc}", logging.WARNING)

            log("Installing dependencies...")
            run_cmd([
                str(venv_python), "-m", "pip", "install", *REQUIRED_PACKAGES,
            ])

        os.environ[BOOTSTRAP_MARKER] = str(int(os.environ.get(BOOTSTRAP_MARKER, "0")) + 1)
        os.execv(
            str(venv_python),
            [str(venv_python)] + _entry_command() + sys.argv[1:],
        )

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        attempt = int(os.environ.get(BOOTSTRAP_MARKER, "0"))
        if attempt >= 2:
            sys.exit(
                "Cannot import " + ", ".join(missing) + " even after installing "
                "them into " + sys.prefix + ".\n"
                "mlx-lm only imports on Apple Silicon: on any other machine this "
                "script can still run --selftest, --doctor and --list-models, but "
                "not the model server.\n"
                "Otherwise delete ./.venv and try again, or install by hand with:\n"
                f"  {sys.executable} -m pip install " + " ".join(REQUIRED_PACKAGES)
            )
        packages = ["mlx-lm" if m == "mlx_lm" else m for m in missing]
        log("Installing missing dependencies: " + ", ".join(packages))
        run_cmd([sys.executable, "-m", "pip", "install", *packages])
        os.environ[BOOTSTRAP_MARKER] = str(attempt + 1)
        os.execv(
            sys.executable,
            [sys.executable] + _entry_command() + sys.argv[1:],
        )


def _entry_command() -> list[str]:
    """How to re-launch this app, whatever way it was started.

    Bootstrap re-execs the process after creating a venv or installing deps.
    Re-execing __file__ would run this module directly, which has no main() and
    would exit silently; and running __main__.py by path breaks its relative
    imports. So reconstruct the original entry: `-m local_llm` when started as a
    module, otherwise the script the user actually ran.
    """
    argv0 = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if argv0 is not None and argv0.name == "__main__.py" and (argv0.parent / "__init__.py").exists():
        return ["-m", argv0.parent.name]
    if argv0 is not None and argv0.is_file():
        return [str(argv0)]
    # Last resort: the package's module form, which always works when installed.
    return ["-m", __package__ or "local_llm"]


def ensure_dirs() -> None:
    for d in [DATA_DIR, SFT_DIR, LOG_DIR, ADAPTER_DIR.parent, ADAPTER_BACKUP_DIR, WORKSPACE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokenizer."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def messages_tokens(messages: list[dict]) -> int:
    # Four tokens per message covers the chat template role markers.
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


def trim_to_context(
    system: dict,
    history: list[dict],
    user: dict,
    context_size: int,
    reserve: int,
    pinned: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Drop the oldest history until the request fits the context window.

    Returns the message list actually sent and the number of dropped messages.
    The system prompt, anything in `pinned`, and the current user turn are never
    dropped: if those alone exceed the budget the caller has a configuration
    problem, not a history problem, and silently truncating them would hide it.

    `pinned` exists for the agent loop. Without it the oldest message dropped
    from a long tool trace is the original question, and the model ends up
    summarising tool output while no longer knowing what was asked.
    """
    pinned = list(pinned or [])
    budget = max(256, context_size - reserve - CONTEXT_SAFETY_MARGIN)
    fixed = messages_tokens([system, *pinned, user])
    kept = list(history)
    dropped = 0
    while kept and fixed + messages_tokens(kept) > budget:
        kept.pop(0)
        dropped += 1

    # If trimming took the MOST RECENT exchange, put an abbreviated version back.
    # A follow-up almost always refers to the last turn ("it doesn't have a main
    # function"), and one long answer -- a program, a big tool result -- is enough
    # to evict it wholesale. The model then has no idea what is being discussed
    # and answers something like "no". A shortened tail is far better than none:
    # it keeps the thread while still respecting the budget.
    if dropped and not kept:
        room = budget - fixed
        recovered: list[dict] = []
        for message in reversed(history[-2:]):
            if room <= 96:                      # not enough left to say anything
                break
            text = str(message.get("content") or "")
            # ~4 characters per token, minus a little for the marker itself.
            allowed_chars = max(0, (room - 32) * 4)
            if len(text) > allowed_chars:
                text = (text[:allowed_chars].rstrip()
                        + "\n[... truncated to fit the context window ...]")
            candidate = {"role": message.get("role", "user"), "content": text}
            cost = messages_tokens([candidate])
            if cost > room:
                continue
            recovered.insert(0, candidate)
            room -= cost
        if recovered:
            kept = recovered
            dropped = max(0, dropped - len(recovered))
    return [system, *pinned, *kept, user], dropped


def timezone_utc():
    return timezone.utc


TOOL_PROTOCOL = textwrap.dedent("""\
    You can call tools. To call one, reply with a single JSON object and nothing
    else:
    {"tool": "tool_name", "args": {"arg": "value"}}

    Rules:
    - One tool per reply. Use the exact argument names listed below.
    - You will then receive a message beginning with TOOL RESULT. Read it before
      deciding what to do next.
    - When you have enough information, either reply in plain text with no JSON,
      or call final_answer with your complete answer. In a plain-text reply the
      text IS the answer: never add a "Final answer:" label or a
      <final_answer> tag around it.
    - Never invent tool output, and never say you searched, read or ran anything
      unless a TOOL RESULT above shows it.
    - If a tool returns an error, fix the arguments and try once more, or answer
      without it. Do not repeat an identical call.
    - Prefer web_search then fetch_url for anything current. Prefer the
      calculator over doing arithmetic yourself.

    Available tools:
    """)


# Every agent request begins with this exact text, so it is the prefix any
# server-side prompt cache keys on. When it changes, every cached prefix on the
# machine becomes worthless. That is invisible otherwise: you edit the system
# prompt in the UI, latency doubles, and nothing says why.
# Two lanes now share this machine's prompt cache: the tool-calling prompt and
# the prose-only prompt used on a turn routing has already decided needs no
# tool. Each has its own stable prefix, so each is tracked separately under
# "lanes"; the top-level keys mirror the tool lane for existing consumers.
PREFIX_STATE: dict[str, Any] = {"hash": None, "changed_at": None, "generation": 0,
                                "lanes": {}}



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    '_entry_command',
    'ADAPTER_BACKUP_DIR',
    'ADAPTER_DIR',
    'APP_LOGO',
    'APP_NAME',
    'BOOTSTRAP_MARKER',
    'CHARS_PER_TOKEN',
    'CONTEXT_SAFETY_MARGIN',
    'LOGIN_MAX_FAILURES',
    'LOGIN_WINDOW_S',
    'DATA_DIR',
    'DB_PATH',
    'DEFAULT_MODEL',
    'DEFAULT_SYSTEM_PROMPT',
    'EXPORTS_DIR',
    'REPLAY_FILE',
    'LOG_DIR',
    'PREFIX_STATE',
    'REQUIRED_MODULES',
    'REQUIRED_PACKAGES',
    'ROOT',
    'SENTINEL_LOCAL_USER',
    'SHARED_OWNER',
    'SFT_DIR',
    'TOOL_PROTOCOL',
    'TOTAL_RAM_GB',
    'UI_BUILD',
    'WORKSPACE_DIR',
    '_DEFAULT_SYSTEM_PROMPT',
    '_default_context_for_ram',
    '_default_fetch_cap',
    '_default_concurrent_generations',
    '_default_prefill_step',
    '_default_prompt_cache_bytes',
    'GenerationBusy',
    'GenerationGate',
    '_default_model_for_ram',
    '_default_reasoning_tokens',
    '_detect_total_ram_gb',
    'bootstrap',
    'ensure_dirs',
    'estimate_tokens',
    'in_venv',
    'iso',
    'log',
    'logger',
    'messages_tokens',
    'run_cmd',
    'timezone_utc',
    'trim_to_context',
    'utc_now',
]
