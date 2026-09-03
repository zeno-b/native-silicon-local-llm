"""The Config dataclass: every setting, its env var, and its clamps.

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
from .sysutil import *  # noqa: F401,F403


# The search provider is locked to DuckDuckGo Lite (see websearch.SearchBackend).
# These are the only spellings accepted for the SEARCH_BACKEND env var / CLI flag;
# every one of them resolves to the same single provider. Anything else is a
# request for a different engine and is refused.
_DDG_LITE_ALIASES = {
    "duckduckgo_lite", "duckduckgo-lite", "duckduckgo", "ddg", "ddg-lite",
    "ddg_lite", "ddglite", "lite", "",
}


def _normalize_search_backend(value: str | None) -> str:
    """Force the search provider to DuckDuckGo Lite.

    Historical aliases (ddg, duckduckgo, lite, ...) are accepted silently; a
    value that names any other engine (google, bing, brave, tavily, searxng, ...)
    is warned about once and still normalised to duckduckgo_lite. This is the
    config-layer enforcement of the "DuckDuckGo Lite only" constraint: no other
    provider can be selected here, so nothing downstream ever has to choose one.
    """
    name = str(value or "").strip().lower()
    if name and name not in _DDG_LITE_ALIASES:
        log(f"Search provider is locked to DuckDuckGo Lite; ignoring requested "
            f"backend {value!r}.", logging.WARNING)
    return "duckduckgo_lite"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Identity line prepended to the system prompt so the model states who it is
    # rather than falling back on the base model's pretrained identity (e.g.
    # "created by Alibaba Cloud" for Qwen). Overriding identity is a prompt job,
    # not a weights job. Set ASSISTANT_IDENTITY to "" to disable, or to your own
    # text. Defaults to the app's brand name.
    identity: str = field(default_factory=lambda: os.environ.get(
        "ASSISTANT_IDENTITY",
        f"You are {APP_NAME}, a private local AI assistant running on the user's "
        f"Apple Silicon Mac. If asked who or what you are, or who made you, say "
        f"you are {APP_NAME}, a local assistant; do not claim to be built by any "
        f"particular company or name an underlying base model.").strip())
    # Refuse to fine-tune on fewer than this many approved examples: a tiny set
    # overfits and causes catastrophic forgetting rather than a useful shift.
    train_min_examples: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MIN_EXAMPLES", "16")))
    # Knowledge base (RAG): when documents are indexed, the best-matching passages
    # are prepended to the question so answers come from your own material with
    # citations. Uses SQLite FTS5 (BM25) — no embedding model, so it costs no
    # extra memory on an 8GB machine. No-op while the index is empty.
    rag_enabled: bool = field(default_factory=lambda: os.environ.get("RAG_ENABLED", "1") == "1")
    rag_passages: int = field(default_factory=lambda: int(os.environ.get("RAG_PASSAGES", "4")))
    # Comma-separated document paths to restrict retrieval to. Empty = search the
    # whole knowledge base. Set from the UI to "chat with this document".
    rag_scope: str = field(default_factory=lambda: os.environ.get("RAG_SCOPE", "").strip())
    # Local codebase the file tools read and edit. When empty, tools stay in the
    # sandboxed ./workspace. When set to a project directory, the agent can read
    # and change that project's files in place; you review with git and decide
    # what to push. Confined to this directory; .git is never written.
    project_dir: str = field(default_factory=lambda: os.environ.get("PROJECT_DIR", "").strip())
    # Command run by the run_tests tool (in the project dir). Empty = auto-detect
    # (pytest / npm test / cargo test / go test) from the project's files.
    test_command: str = field(default_factory=lambda: os.environ.get("TEST_COMMAND", "").strip())
    # Execution backend for run_shell/run_python/run_tests. "local" is the
    # confined subprocess (fast, but runs with your privileges). "docker" runs
    # the command inside a container with the project mounted, which IS a real
    # boundary: the agent can iterate freely and the blast radius stays in the
    # container. You still review the diff on the host and push manually.
    # After an agent turn edits files, optionally run the tests and, if they fail,
    # let the agent see the failures and try again, up to this many rounds (0 =
    # off). Needs execution enabled (--allow-shell) and a detectable test command.
    auto_iterate_rounds: int = field(default_factory=lambda: int(os.environ.get("AUTO_ITERATE_ROUNDS", "2")))
    exec_backend: str = field(default_factory=lambda: os.environ.get("EXEC_BACKEND", "local").strip().lower())
    docker_image: str = field(default_factory=lambda: os.environ.get("DOCKER_IMAGE", "python:3.12-slim").strip())
    model_port: int = field(default_factory=lambda: int(os.environ.get("MODEL_PORT", "8080")))
    web_port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8000")))
    # 30 iterations is a warmup, not a fine-tune. A format tune needs a few
    # hundred passes over a hundred-odd examples to change anything.
    train_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_ITERS", "300")))
    train_lr: str = field(default_factory=lambda: os.environ.get("TRAIN_LR", "3e-5"))
    train_seq_len: str = field(default_factory=lambda: os.environ.get("TRAIN_SEQ_LEN", "512"))
    # Tuning only the top layers is what keeps a 3B trainable on 8GB while the
    # web process and the page cache are also resident.
    train_num_layers: int = field(default_factory=lambda: int(os.environ.get("TRAIN_NUM_LAYERS", "8")))
    # Fine-tuning method. "lora" (default) trains small adapters and fits on 8GB.
    # "dora" is a slightly heavier LoRA variant. "full" fine-tunes all weights and
    # needs far more memory than an 8GB Mac has — set it (and TRAIN_NUM_LAYERS=-1)
    # only once you move to bigger hardware. The dataset format is identical
    # across all three, so the training set you build now is already future-proof.
    train_fine_tune_type: str = field(default_factory=lambda: os.environ.get("TRAIN_FINE_TUNE_TYPE", "lora"))
    train_on_tool_calls: bool = field(default_factory=lambda: os.environ.get("TRAIN_ON_TOOL_CALLS", "1") == "1")
    train_tool_examples: int = field(default_factory=lambda: int(os.environ.get("TRAIN_TOOL_EXAMPLES", "400")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "512")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    # Context window. context_size is the budget this process enforces when it
    # assembles a request; max_kv_size is what the model server is told to
    # allocate. They are separate because the server flag is optional and
    # changing it needs a restart, while context_size takes effect immediately.
    context_size: int = field(default_factory=lambda: int(
        os.environ.get("CONTEXT_SIZE") or _default_context_for_ram(TOTAL_RAM_GB)))
    max_kv_size: int = field(default_factory=lambda: int(os.environ.get("MAX_KV_SIZE", "0")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.7")))
    history_turns: int = field(default_factory=lambda: int(os.environ.get("HISTORY_TURNS", "20")))

    # Agent
    agent_enabled: bool = field(default_factory=lambda: os.environ.get("AGENT_ENABLED", "0") == "1")
    # Before answering a substantive question, ask the model in one word whether
    # it needs to look the answer up. This is the general form of the fast-path
    # regexes: instead of enumerating every phrasing, let the model judge, but
    # in a shape a small model handles well (a single SEARCH/ANSWER token) and
    # then seed the search deterministically so it does not depend on the model
    # emitting tool-call JSON. Biased toward SEARCH when unsure, since a needless
    # search is cheaper than a confident wrong answer or a refusal.
    knowledge_triage: bool = field(default_factory=lambda: os.environ.get("KNOWLEDGE_TRIAGE", "1") == "1")
    # After a lookup search, automatically fetch this many of the top result
    # pages and give the model their full text, not just the snippet. This is
    # what makes one generic search path answer domain-specific questions (a
    # stock price, a score, a forecast, a version number): the answer is usually
    # on the page even when the snippet omits it. 0 disables auto-fetch and falls
    # back to snippet-only plus the model choosing to call fetch_url itself.
    auto_fetch_results: int = field(default_factory=lambda: int(os.environ.get("AUTO_FETCH_RESULTS", "2")))
    # Hardest ceiling on characters of a fetched page that may enter the prompt.
    # A large docs page would otherwise dominate context and OOM prefill on an
    # 8GB machine. Roughly char_cap/4 tokens.
    auto_fetch_char_cap: int = field(default_factory=lambda: int(
        os.environ.get("AUTO_FETCH_CHAR_CAP") or _default_fetch_cap(TOTAL_RAM_GB)))
    agent_max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "6")))
    # Resilience under memory pressure. When a generation errors (a model-server
    # OOM kill and watchdog restart look like a dropped connection from here), the
    # turn retries with a smaller token budget rather than surfacing an error.
    # resilient_retries is how many times; min_max_tokens is the floor it shrinks
    # to. hard_step_cap lets a big task keep going past agent_max_steps by
    # compacting progress into a running summary, so it slows down but does not
    # stop. These exist to satisfy "never error or stop; step down and continue".
    resilient_retries: int = field(default_factory=lambda: int(os.environ.get("RESILIENT_RETRIES", "3")))
    min_max_tokens: int = field(default_factory=lambda: int(os.environ.get("MIN_MAX_TOKENS", "128")))
    # If the model server sends nothing for this many seconds mid-generation, the
    # request is treated as stalled: it raises, and the resilient loop retries
    # with a smaller budget instead of hanging. This is what turns "stuck" into
    # visible "retrying" rather than minutes of dead air waiting on a wedged or
    # OOM-killed server.
    stall_timeout: int = field(default_factory=lambda: int(os.environ.get("STALL_TIMEOUT", "60")))
    # How long a retry waits for a restarting model server to become ready again
    # before giving up on that attempt. Longer helps slow cold-start reloads.
    ready_wait_timeout: float = field(default_factory=lambda: float(os.environ.get("READY_WAIT_TIMEOUT", "40")))
    hard_step_cap: int = field(default_factory=lambda: int(os.environ.get("HARD_STEP_CAP", "24")))
    # Incremental reasoning: for a hard analytical question with no tool to call,
    # decompose it into sub-steps and solve them one at a time, carrying only
    # short conclusions forward. Each pass is small, so the working set stays
    # inside 8GB no matter how deep the reasoning goes, and decomposition makes a
    # small model reason better than one shot. It is slower (several small calls)
    # by design: it takes its time instead of failing or answering shallowly.
    incremental_reasoning: bool = field(default_factory=lambda: os.environ.get("INCREMENTAL_REASONING", "1") == "1")
    # Chunk an oversized prompt (a pasted file or long document) and process it
    # part by part, so a single input larger than the context never has to be
    # prefilled in one pass. This is what lets an 8GB machine handle a large
    # prompt: split it, extract findings per chunk into bounded notes, then
    # synthesise. Chunk size and trigger are derived from context_size.
    chunk_large_prompts: bool = field(default_factory=lambda: os.environ.get("CHUNK_LARGE_PROMPTS", "1") == "1")
    # Emit verbose "under the hood" detail events (prompt sizes, per-step timing,
    # extracted lengths, fallback reasons) so the whole pipeline is visible.
    show_internals: bool = field(default_factory=lambda: os.environ.get("SHOW_INTERNALS", "1") == "1")
    # Hard wall-clock ceiling for the whole multi-source retrieval phase (fetch +
    # read all sources). Prevents a stalling extraction from grinding for minutes.
    retrieval_deadline: float = field(default_factory=lambda: float(os.environ.get("RETRIEVAL_DEADLINE", "90")))
    # Fraction of the context above which a prompt is chunked, and the fraction
    # of the context each chunk targets. Derived from context_size so they scale
    # with RAM; exposed so the thresholds themselves can be tuned per machine.
    chunk_trigger_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_TRIGGER_RATIO", "0.6")))
    chunk_size_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_SIZE_RATIO", "0.4")))
    reasoning_max_steps: int = field(default_factory=lambda: int(os.environ.get("REASONING_MAX_STEPS", "6")))
    # Hard wall-clock cap per reasoning step. Distinct from stall_timeout (which
    # only fires on zero output): this bounds a step that streams slowly but
    # never finishes, so a single step can never wedge the whole chain.
    reasoning_step_timeout: int = field(default_factory=lambda: int(os.environ.get("REASONING_STEP_TIMEOUT", "45")))
    # Per-step generation budget for reasoning, chunk and source-extraction
    # passes. RAM-scaled default; small on 8GB, larger on roomy machines.
    reasoning_tokens: int = field(default_factory=lambda: int(
        os.environ.get("REASONING_TOKENS") or _default_reasoning_tokens(TOTAL_RAM_GB)))
    allow_python: bool = field(default_factory=lambda: os.environ.get("ALLOW_PYTHON", "0") == "1")
    allow_shell: bool = field(default_factory=lambda: os.environ.get("ALLOW_SHELL", "0") == "1")
    # Comma-separated allowlist. Empty means every registered tool is offered.
    agent_tools: str = field(default_factory=lambda: os.environ.get("AGENT_TOOLS", ""))
    # Which LoRA adapter the model server loads: latest, none, or a backup id.
    adapter: str = field(default_factory=lambda: os.environ.get("ADAPTER", "latest"))
    # Extra model ids to offer in the switcher, on top of DEFAULT_MODEL_CATALOG.
    model_catalog: str = field(default_factory=lambda: os.environ.get("MODEL_CATALOG", ""))
    # How many background task runs may execute at once. The model server
    # serves one request at a time, so more than one mostly adds queueing.
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_TASKS", "1")))
    task_poll_seconds: int = field(default_factory=lambda: int(os.environ.get("TASK_POLL_SECONDS", "2")))
    # Seconds of interactive quiet before a scheduled run is allowed to start.
    chat_idle_seconds: int = field(default_factory=lambda: int(os.environ.get("CHAT_IDLE_SECONDS", "45")))
    # fetch_url refuses loopback and RFC1918 targets unless this is on, so a
    # prompt-injected page cannot make the agent read the machine's own
    # services (including this app's API) and hand the result back.
    allow_local_fetch: bool = field(default_factory=lambda: os.environ.get("ALLOW_LOCAL_FETCH", "0") == "1")

    # Reasoning-mode models (Qwen3.5 and later) emit a <think> block by default.
    # In a tool loop that is pure cost: the reasoning is discarded by the
    # protocol, it inflates the KV cache, and JSON inside it confuses parsing.
    disable_thinking: bool = field(default_factory=lambda: os.environ.get("DISABLE_THINKING", "0") == "1")
    # Make the model reason out loud before answering, and show that reasoning in
    # the trace. Adds a <think> instruction to the prompt so even non-reasoning
    # models produce visible step-by-step thinking. Costs tokens; turn off for
    # speed. On by default per request for maximum transparency.
    reasoning_visible: bool = field(default_factory=lambda: os.environ.get("REASONING_VISIBLE", "1") == "1")
    # Tool-selection steps want deterministic JSON; only the final answer wants
    # the configured temperature. One value for both costs malformed calls.
    tool_temperature: float = field(default_factory=lambda: float(os.environ.get("TOOL_TEMPERATURE", "0.0")))
    # Multiplicative penalty on tokens already in the window. Small quantised
    # models fall into verbatim repetition loops, and greedy decoding
    # (tool_temperature 0.0) has no way out of one: the argmax that produced the
    # loop keeps producing it until max_tokens runs out. 1.0 disables it.
    repetition_penalty: float = field(default_factory=lambda: float(os.environ.get("REPETITION_PENALTY", "1.1")))
    repetition_context_size: int = field(default_factory=lambda: int(os.environ.get("REPETITION_CONTEXT_SIZE", "64")))
    # OFF by default. Recent mlx-lm builds (see generate.py _step) crash with
    # "TypeError: 'NoneType' object is not iterable" over self.logits_processors
    # when the repetition_penalty request fields are present, killing the
    # generation thread so every request stalls. Sending no logits-processor
    # fields avoids the null list entirely. Set REPETITION_PENALTY_ENABLED=1 to
    # re-enable once upstream is fixed.
    repetition_penalty_enabled: bool = field(default_factory=lambda: os.environ.get("REPETITION_PENALTY_ENABLED", "0") == "1")
    # Answer arithmetic and bare URLs without a model round trip at all.
    fast_path: bool = field(default_factory=lambda: os.environ.get("FAST_PATH", "1") == "1")
    # When the trace outgrows the context, collapse the oldest steps into one
    # summary message instead of dropping them off the front. Dropping shifts
    # every following token and invalidates any server-side prefix cache at the
    # exact point a run is longest.
    stable_prefix: bool = field(default_factory=lambda: os.environ.get("STABLE_PREFIX", "1") == "1")

    # KV cache quantization, passed through to mlx_lm.server when the installed
    # build accepts the flags. 0 leaves the cache in fp16.
    kv_bits: int = field(default_factory=lambda: int(os.environ.get("KV_BITS", "0")))
    kv_group_size: int = field(default_factory=lambda: int(os.environ.get("KV_GROUP_SIZE", "64")))
    quantized_kv_start: int = field(default_factory=lambda: int(os.environ.get("QUANTIZED_KV_START", "1024")))
    prompt_cache_dir: str = field(default_factory=lambda: os.environ.get("PROMPT_CACHE_DIR", ""))

    # Tools
    # Search provider is LOCKED to DuckDuckGo Lite. SEARCH_BACKEND is honoured
    # only insofar as it names a DuckDuckGo-Lite alias; any request for another
    # engine is ignored (with a warning) and normalised back to duckduckgo_lite.
    search_backend: str = field(default_factory=lambda: _normalize_search_backend(
        os.environ.get("SEARCH_BACKEND", "duckduckgo_lite")))
    search_results: int = field(default_factory=lambda: int(os.environ.get("SEARCH_RESULTS", "5")))
    tool_timeout: int = field(default_factory=lambda: int(os.environ.get("TOOL_TIMEOUT", "30")))
    # Two different caps, and the difference matters.
    #
    # tool_raw_chars is how much a tool may return at all. It is what gets
    # logged and shown in the UI, and what the summariser reads.
    #
    # tool_result_chars is how much may enter the model's context. A result is
    # re-sent on every later step, so a 4000-character page is not a 4000-token
    # cost, it is that times the number of steps that follow.
    #
    # Collapsing these two into one number means the raw result is destroyed
    # before anything can summarise it, and the summariser becomes dead code.
    tool_raw_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RAW_CHARS", "20000")))
    tool_result_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RESULT_CHARS", "1500")))
    # Results longer than this get one cheap summarisation pass before they
    # enter the context. Worth it even at 15 tok/s because of the multiplier.
    summarise_tool_results: bool = field(default_factory=lambda: os.environ.get("SUMMARISE_TOOL_RESULTS", "1") == "1")
    summarise_over_chars: int = field(default_factory=lambda: int(os.environ.get("SUMMARISE_OVER_CHARS", "2500")))

    # ----------------------------------------------------------------- #
    # Structured logging (see obslog.py). All env-overridable.
    # ----------------------------------------------------------------- #
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    # json for aggregation (Loki/ELK/Datadog), text for local reading.
    log_format: str = field(default_factory=lambda: os.environ.get("LOG_FORMAT", "json"))
    # Override the log directory; empty uses ./logs under the project root.
    log_dir: str = field(default_factory=lambda: os.environ.get("LOG_DIR", "").strip())
    # How much chat content is written: disabled | metadata (length+fingerprint,
    # never the text) | full (redacted, truncated). Production default: metadata.
    log_chat_content: str = field(default_factory=lambda: os.environ.get("LOG_CHAT_CONTENT", "metadata"))
    log_max_bytes: int = field(default_factory=lambda: int(os.environ.get("LOG_MAX_BYTES", str(10 * 1024 * 1024))))
    log_backup_count: int = field(default_factory=lambda: int(os.environ.get("LOG_BACKUP_COUNT", "10")))
    log_retention_days: int = field(default_factory=lambda: int(os.environ.get("LOG_RETENTION_DAYS", "14")))

    # ----------------------------------------------------------------- #
    # Authentication and multi-user (see auth.py). OFF by default so a
    # single-user local install behaves exactly as before (a synthetic
    # 'local' admin owns everything). Turn on with AUTH_ENABLED=1.
    # ----------------------------------------------------------------- #
    auth_enabled: bool = field(default_factory=lambda: os.environ.get("AUTH_ENABLED", "0") == "1")
    auth_session_ttl_hours: int = field(default_factory=lambda: int(os.environ.get("AUTH_SESSION_TTL_HOURS", "168")))
    auth_cookie_name: str = field(default_factory=lambda: os.environ.get("AUTH_COOKIE_NAME", "llm_session"))
    # Set 1 when served over HTTPS (behind a reverse proxy) so the cookie is
    # marked Secure. Leave 0 for plain-HTTP local dev or the cookie won't be set.
    auth_cookie_secure: bool = field(default_factory=lambda: os.environ.get("AUTH_COOKIE_SECURE", "0") == "1")
    # First-run admin bootstrap. The password is used ONCE to create the admin
    # then must not persist; it is never stored in the DB in plaintext and is
    # redacted from any config dump. If unset and no admin exists, the app prints
    # a one-time generated password to the log at startup.
    admin_username: str = field(default_factory=lambda: os.environ.get("AUTH_ADMIN_USERNAME", "admin").strip())
    admin_password: str = field(default_factory=lambda: os.environ.get("AUTH_ADMIN_PASSWORD", ""))
    # A convenience non-admin account for dev/testing. NOT a backdoor: it exists
    # only when explicitly enabled and with an explicit password, and the README
    # documents removing it for production.
    allow_test_user: bool = field(default_factory=lambda: os.environ.get("AUTH_ALLOW_TEST_USER", "0") == "1")
    test_username: str = field(default_factory=lambda: os.environ.get("AUTH_TEST_USERNAME", "test").strip())
    test_password: str = field(default_factory=lambda: os.environ.get("AUTH_TEST_PASSWORD", ""))

    # ----------------------------------------------------------------- #
    # Microsoft Entra ID / OpenID Connect (authorization-code flow).
    # ----------------------------------------------------------------- #
    oidc_enabled: bool = field(default_factory=lambda: os.environ.get("OIDC_ENABLED", "0") == "1")
    oidc_tenant_id: str = field(default_factory=lambda: os.environ.get("OIDC_TENANT_ID", "").strip())
    oidc_client_id: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_ID", "").strip())
    oidc_client_secret: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_SECRET", ""))
    oidc_redirect_uri: str = field(default_factory=lambda: os.environ.get("OIDC_REDIRECT_URI", "").strip())
    # Defaults to https://login.microsoftonline.com/{tenant}/v2.0 when empty.
    oidc_authority: str = field(default_factory=lambda: os.environ.get("OIDC_AUTHORITY", "").strip())
    oidc_scopes: str = field(default_factory=lambda: os.environ.get("OIDC_SCOPES", "openid profile email").strip())
    # Role mapping. Any of these that matches an incoming token grants admin.
    oidc_admin_emails: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_EMAILS", "").strip())
    oidc_admin_groups: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_GROUPS", "").strip())
    oidc_admin_roles: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_ROLES", "").strip())
    oidc_default_role: str = field(default_factory=lambda: os.environ.get("OIDC_DEFAULT_ROLE", "user").strip())

    # ----------------------------------------------------------------- #
    # Mac Mini (primary) / Mac Studio (secondary) cluster + routing
    # (see cluster.py). Single-node by default: with no STUDIO_NODE_URL the
    # router has one node (the local model server) and behaves as before.
    # ----------------------------------------------------------------- #
    node_role: str = field(default_factory=lambda: os.environ.get("NODE_ROLE", "primary").strip().lower())
    node_name: str = field(default_factory=lambda: os.environ.get("NODE_NAME", "").strip())
    # The Studio's OpenAI-compatible generation base URL, e.g.
    # http://studio.local:8080 . Empty disables the secondary node entirely.
    studio_node_url: str = field(default_factory=lambda: os.environ.get("STUDIO_NODE_URL", "").strip())
    # A secondary advertises where its router/primary is (used for heartbeats).
    primary_node_url: str = field(default_factory=lambda: os.environ.get("PRIMARY_NODE_URL", "").strip())
    # Shared secret for inter-node calls (sent as a bearer token). SECRET.
    node_token: str = field(default_factory=lambda: os.environ.get("NODE_TOKEN", ""))
    # Configurable routing factors. No arbitrary hard-coded thresholds.
    route_max_active_per_node: int = field(default_factory=lambda: int(os.environ.get("ROUTE_MAX_ACTIVE", "2")))
    route_queue_depth: int = field(default_factory=lambda: int(os.environ.get("ROUTE_QUEUE_DEPTH", "4")))
    route_cpu_pct: float = field(default_factory=lambda: float(os.environ.get("ROUTE_CPU_PCT", "85")))
    route_mem_pct: float = field(default_factory=lambda: float(os.environ.get("ROUTE_MEM_PCT", "85")))
    # If the primary's recent latency exceeds this SLA (ms), eligible work spills
    # to the Studio. 0 disables the SLA factor.
    route_sla_ms: int = field(default_factory=lambda: int(os.environ.get("ROUTE_SLA_MS", "0")))
    # Substrings that mark a model as "large" (needs the high-memory Studio).
    large_model_markers: str = field(default_factory=lambda: os.environ.get(
        "LARGE_MODEL_MARKERS", "14B,32B,70B,72B").strip())
    heartbeat_interval: float = field(default_factory=lambda: float(os.environ.get("HEARTBEAT_INTERVAL", "10")))
    heartbeat_timeout: float = field(default_factory=lambda: float(os.environ.get("HEARTBEAT_TIMEOUT", "30")))
    # Timeout for a node health probe / heartbeat request.
    node_probe_timeout: float = field(default_factory=lambda: float(os.environ.get("NODE_PROBE_TIMEOUT", "8")))

    # ----------------------------------------------------------------- #
    # Claude-history ZIP import limits (see claude_import.py). The upload is
    # untrusted: these bound the damage a hostile archive can do.
    # ----------------------------------------------------------------- #
    import_max_zip_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_ZIP_BYTES", str(200 * 1024 * 1024))))
    import_max_files: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_FILES", "20000")))
    import_max_uncompressed_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_UNCOMPRESSED_BYTES", str(1024 * 1024 * 1024))))
    import_max_file_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_FILE_BYTES", str(50 * 1024 * 1024))))

    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"

    # Fields whose value must never be returned by public() or written to a log.
    SECRET_FIELDS = (
        "admin_password", "test_password", "oidc_client_secret", "node_token",
    )

    # Settings the web UI is allowed to change at runtime. Anything not listed
    # here needs a process restart and is rejected by /api/config.
    MUTABLE = (
        "system_prompt", "identity", "train_min_examples", "project_dir",
        "rag_enabled", "rag_passages", "rag_scope",
        "max_tokens", "temperature", "repetition_penalty",
        "repetition_context_size", "repetition_penalty_enabled", "context_size",
        "history_turns", "agent_enabled", "agent_max_steps",
        # search_backend is intentionally NOT mutable: the provider is locked to
        # DuckDuckGo Lite and cannot be changed from the UI or the API.
        "search_results", "tool_result_chars", "tool_raw_chars", "auto_fetch_results",
        "disable_thinking", "reasoning_visible", "tool_temperature", "fast_path", "stable_prefix", "knowledge_triage",
        "summarise_tool_results", "summarise_over_chars",
        # Safeguards, all tunable live so a machine can be dialled in without a
        # restart or an env edit.
        "incremental_reasoning", "reasoning_max_steps", "reasoning_step_timeout",
        "reasoning_tokens", "chunk_large_prompts", "chunk_trigger_ratio",
        "chunk_size_ratio", "auto_fetch_char_cap", "stall_timeout", "ready_wait_timeout",
        "resilient_retries", "min_max_tokens", "hard_step_cap",
        "show_internals", "retrieval_deadline", "exec_backend", "docker_image",
        "test_command", "auto_iterate_rounds",
        # Logging: enabling DEBUG/TRACE and disabling content logs at runtime.
        "log_level", "log_format", "log_chat_content",
        # Routing factors, tunable live so a two-Mac cluster can be dialled in.
        # (studio_node_url is NOT here: adding/removing a node is a topology
        # change that rebuilds the registry, so it needs a restart.)
        "route_max_active_per_node", "route_queue_depth", "route_cpu_pct",
        "route_mem_pct", "route_sla_ms", "large_model_markers",
        "heartbeat_interval", "heartbeat_timeout",
    )

    # The only supported search provider. DuckDuckGo Lite, exclusively.
    SEARCH_BACKENDS = ("duckduckgo_lite",)

    @property
    def system_prompt_with_identity(self) -> str:
        """The system prompt with the identity line prepended, if set."""
        if self.identity:
            return f"{self.identity}\n\n{self.system_prompt}"
        return self.system_prompt

    def __post_init__(self) -> None:
        # Clamp at construction too, not only on live edits, so a bad value from
        # an environment variable at startup is corrected the same way the UI's
        # live edits are. apply({}) runs every guardrail with no other effect.
        self.apply({})

    def public(self) -> dict:
        data = {k: v for k, v in asdict(self).items()}
        # Never expose secrets through the API/UI or a config dump: report only
        # whether each is set. This is what makes GET /api/config safe to serve.
        for secret in self.SECRET_FIELDS:
            if secret in data:
                data[secret] = "***set***" if data.get(secret) else ""
        data["mutable"] = list(self.MUTABLE)
        return data

    def apply(self, updates: dict) -> list[str]:
        """Apply a settings patch. Returns the names of the fields changed.

        Values are clamped afterwards, and any field the clamp moved is reported
        as changed too, so the UI never shows a setting the process did not
        actually adopt.
        """
        changed = []
        for key, value in updates.items():
            if key not in self.MUTABLE or value is None:
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                # A malformed value (e.g. "abc" for a number). Skip this field
                # rather than crash the whole settings update.
                log(f"Ignoring invalid value for {key}: {value!r}", logging.WARNING)
                continue
            if value != current:
                setattr(self, key, value)
                changed.append(key)

        before = {name: getattr(self, name) for name in
                  ("context_size", "max_tokens", "temperature", "agent_max_steps",
                   "history_turns", "search_results", "tool_result_chars",
                   "tool_temperature", "summarise_over_chars")}
        self.context_size = min(131072, max(512, self.context_size))
        self.max_tokens = max(16, self.max_tokens)
        if self.max_tokens >= self.context_size:
            self.max_tokens = max(16, self.context_size // 2)
        self.temperature = min(2.0, max(0.0, self.temperature))
        self.tool_temperature = min(2.0, max(0.0, self.tool_temperature))
        self.summarise_over_chars = max(500, self.summarise_over_chars)
        self.agent_max_steps = min(20, max(1, self.agent_max_steps))
        self.resilient_retries = min(6, max(0, self.resilient_retries))
        self.min_max_tokens = min(512, max(32, self.min_max_tokens))
        self.stall_timeout = min(600, max(10, self.stall_timeout))
        self.auto_fetch_char_cap = min(60000, max(1000, self.auto_fetch_char_cap))
        # The cap can never be below the ordinary step budget.
        self.hard_step_cap = min(60, max(self.agent_max_steps, self.hard_step_cap))
        self.reasoning_max_steps = min(10, max(2, self.reasoning_max_steps))
        self.reasoning_step_timeout = min(300, max(10, self.reasoning_step_timeout))
        self.retrieval_deadline = min(600.0, max(15.0, self.retrieval_deadline))
        self.auto_iterate_rounds = min(5, max(0, self.auto_iterate_rounds))
        self.rag_passages = min(20, max(1, self.rag_passages))
        # A scope of unusable entries would silently return nothing; normalise it.
        self.rag_scope = ",".join(
            part.strip() for part in str(self.rag_scope or "").split(",") if part.strip())
        self.reasoning_tokens = min(2048, max(64, self.reasoning_tokens))
        self.ready_wait_timeout = min(300.0, max(2.0, self.ready_wait_timeout))
        # Ratios kept in sane bands so a bad value cannot break chunking: the
        # trigger must leave room for a reply, and a chunk must be smaller than
        # the trigger or it could never fit.
        self.chunk_trigger_ratio = min(0.9, max(0.2, self.chunk_trigger_ratio))
        self.chunk_size_ratio = min(self.chunk_trigger_ratio, max(0.1, self.chunk_size_ratio))
        self.history_turns = min(200, max(0, self.history_turns))
        self.search_results = min(10, max(1, self.search_results))
        self.tool_result_chars = min(40000, max(200, self.tool_result_chars))
        self.tool_raw_chars = min(200000, max(self.tool_result_chars, self.tool_raw_chars))
        # The search provider is immutable: whatever arrived (env, CLI, a stale
        # constructor value), it is forced back to DuckDuckGo Lite here.
        self.search_backend = _normalize_search_backend(self.search_backend)

        # --- Logging -------------------------------------------------------- #
        self.log_level = str(self.log_level or "INFO").strip().upper() or "INFO"
        if str(self.log_format).strip().lower() not in ("json", "text"):
            self.log_format = "json"
        else:
            self.log_format = str(self.log_format).strip().lower()
        if str(self.log_chat_content).strip().lower() not in ("disabled", "metadata", "full"):
            self.log_chat_content = "metadata"
        else:
            self.log_chat_content = str(self.log_chat_content).strip().lower()
        self.log_max_bytes = max(0, self.log_max_bytes)
        self.log_backup_count = min(1000, max(0, self.log_backup_count))
        self.log_retention_days = min(3650, max(0, self.log_retention_days))

        # --- Auth ----------------------------------------------------------- #
        self.auth_session_ttl_hours = min(8760, max(1, self.auth_session_ttl_hours))
        if str(self.oidc_default_role).strip().lower() not in ("user", "admin"):
            self.oidc_default_role = "user"

        # --- Cluster / routing ---------------------------------------------- #
        if str(self.node_role).strip().lower() not in ("primary", "secondary"):
            self.node_role = "primary"
        else:
            self.node_role = str(self.node_role).strip().lower()
        self.route_max_active_per_node = min(256, max(1, self.route_max_active_per_node))
        self.route_queue_depth = min(100000, max(0, self.route_queue_depth))
        self.route_cpu_pct = min(100.0, max(1.0, self.route_cpu_pct))
        self.route_mem_pct = min(100.0, max(1.0, self.route_mem_pct))
        self.route_sla_ms = min(3600000, max(0, self.route_sla_ms))
        self.heartbeat_interval = min(3600.0, max(1.0, self.heartbeat_interval))
        self.heartbeat_timeout = min(86400.0, max(2.0, self.heartbeat_timeout))
        self.node_probe_timeout = min(120.0, max(1.0, self.node_probe_timeout))

        # --- Import limits -------------------------------------------------- #
        self.import_max_zip_bytes = max(1024, self.import_max_zip_bytes)
        self.import_max_files = min(10_000_000, max(1, self.import_max_files))
        self.import_max_uncompressed_bytes = max(1024, self.import_max_uncompressed_bytes)
        self.import_max_file_bytes = max(1024, self.import_max_file_bytes)

        for name, old in before.items():
            if getattr(self, name) != old and name not in changed:
                changed.append(name)
        return changed



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Config',
    '_DDG_LITE_ALIASES',
    '_normalize_search_backend',
]
