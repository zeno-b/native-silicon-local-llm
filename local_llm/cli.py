"""Argument parsing and the main entry point.

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
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .ui import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .tasks import *  # noqa: F401,F403
from .api import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .diagnostics import *  # noqa: F401,F403
from .selftest import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403


def build_config(args) -> Config:
    """Build a Config from parsed CLI args. Shared by serving and the
    --print-config / --dump-prompt inspection flags so they see the same
    configuration the server would actually run with."""
    return Config(
        model=args.model,
        system_prompt=args.system_prompt,
        model_port=args.model_port,
        web_port=args.web_port,
        train_iters=args.train_iters,
        train_lr=args.train_lr,
        train_seq_len=args.train_seq_len,
        max_tokens=args.max_tokens,
        auto_retrain_threshold=args.auto_retrain_threshold,
        context_size=args.context_size,
        max_kv_size=args.max_kv_size,
        temperature=args.temperature,
        history_turns=args.history_turns,
        agent_enabled=args.agent,
        agent_max_steps=args.agent_max_steps,
        allow_python=args.allow_python,
        allow_shell=args.allow_shell,
        agent_tools=args.agent_tools,
        allow_local_fetch=args.allow_local_fetch,
        adapter=args.adapter,
        model_catalog=args.model_catalog,
        max_concurrent_tasks=args.max_concurrent_tasks,
        search_backend=args.search_backend,
        search_results=args.search_results,
        seed_demo=args.seed_demo,
        retrain_now=args.retrain_now,
        export_only=args.export_only,
        list_feedback=args.list_feedback,
        export_format=args.export_format,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one local LLM server, chat UI, feedback, and retraining loop."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model-port", type=int, default=int(os.environ.get("MODEL_PORT", "8080")))
    parser.add_argument("--web-port", type=int, default=int(os.environ.get("WEB_PORT", "8000")))
    parser.add_argument("--train-iters", type=int, default=int(os.environ.get("TRAIN_ITERS", "30")))
    parser.add_argument("--train-lr", default=os.environ.get("TRAIN_LR", "1e-4"))
    parser.add_argument("--train-seq-len", default=os.environ.get("TRAIN_SEQ_LEN", "256"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "512")))
    parser.add_argument("--auto-retrain-threshold", type=int, default=int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    parser.add_argument("--context-size", type=int, default=int(os.environ.get("CONTEXT_SIZE", "4096")),
                        help="Token budget this process enforces when assembling a request.")
    parser.add_argument("--max-kv-size", type=int, default=int(os.environ.get("MAX_KV_SIZE", "0")),
                        help="KV cache cap passed to mlx_lm.server. 0 leaves it unbounded.")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.7")))
    parser.add_argument("--history-turns", type=int, default=int(os.environ.get("HISTORY_TURNS", "20")),
                        help="How many past messages to replay. 0 disables conversation memory.")

    parser.add_argument("--agent", action="store_true", default=os.environ.get("AGENT_ENABLED") == "1",
                        help="Enable the tool-calling agent loop by default.")
    parser.add_argument("--agent-max-steps", type=int, default=int(os.environ.get("AGENT_MAX_STEPS", "6")))
    parser.add_argument("--allow-python", action="store_true", default=os.environ.get("ALLOW_PYTHON") == "1",
                        help="Expose a run_python tool. The model gets code execution on this machine.")
    parser.add_argument("--allow-shell", action="store_true", default=os.environ.get("ALLOW_SHELL") == "1",
                        help="Expose a run_shell tool. The model gets shell access on this machine.")
    parser.add_argument("--allow-local-fetch", action="store_true",
                        default=os.environ.get("ALLOW_LOCAL_FETCH") == "1",
                        help="Let fetch_url reach loopback and private addresses. Off by default.")
    parser.add_argument("--agent-tools", default=os.environ.get("AGENT_TOOLS", ""),
                        help="Comma-separated allowlist of tool names. Empty offers all of them.")
    parser.add_argument("--adapter", default=os.environ.get("ADAPTER", "latest"),
                        help="LoRA adapter to load: latest, none, or a backup id.")
    parser.add_argument("--model-catalog", default=os.environ.get("MODEL_CATALOG", ""),
                        help="Extra model ids to offer in the Models tab, comma separated.")
    parser.add_argument("--max-concurrent-tasks", type=int,
                        default=int(os.environ.get("MAX_CONCURRENT_TASKS", "1")),
                        help="How many background task runs may execute at once.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the model catalogue and which weights are cached, then exit.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print the defined tasks and their last run, then exit.")
    parser.add_argument("--add-task", metavar="JSON",
                        help='Create a task and exit, e.g. \'{"name": "n", "goal": "g", '
                             '"interval_seconds": 3600}\'')
    # The search provider is locked to DuckDuckGo Lite. The flag is kept for
    # backward compatibility with existing scripts, but any value other than a
    # DuckDuckGo-Lite alias is ignored (normalised in Config) rather than
    # selecting a different engine.
    parser.add_argument("--search-backend", default=os.environ.get("SEARCH_BACKEND", "duckduckgo_lite"),
                        help="Locked to DuckDuckGo Lite; other values are ignored.")
    parser.add_argument("--search-results", type=int, default=int(os.environ.get("SEARCH_RESULTS", "5")))
    parser.add_argument("--list-tools", action="store_true", help="Print the tool catalogue and exit.")
    parser.add_argument("--tool-test", metavar="NAME", help="Run one tool directly and exit.")
    parser.add_argument("--tool-args", default="{}", help="JSON arguments for --tool-test.")
    parser.add_argument("--seed-demo", action="store_true")
    parser.add_argument("--retrain-now", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--list-feedback", action="store_true")
    parser.add_argument("--doctor", action="store_true",
                        help="Probe the ports and report what is actually listening, then exit.")
    parser.add_argument("--bench", action="store_true",
                        help="Measure prefill and decode against a running model server, then exit.")
    parser.add_argument("--bench-save", action="store_true",
                        help="With --bench, save the result as the baseline to compare against later.")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify this file is intact and the embedded UI parses, then exit.")
    parser.add_argument("--dump-prompt", action="store_true",
                        help="Print the exact system prompt (identity + prompt + tool preamble) sent to the model, then exit.")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the resolved configuration (after env, flags, and clamps), then exit.")
    parser.add_argument("--export-format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.print_config:
        cfg = build_config(args)
        import json as _json
        data = cfg.public()
        data.pop("mutable", None)
        print(_json.dumps(data, indent=2, default=str))
        return

    if args.dump_prompt:
        cfg = build_config(args)
        reg = ToolRegistry(cfg, None)
        print("=== identity ===")
        print(cfg.identity or "(none)")
        print("\n=== system prompt (as sent, with identity + tool preamble) ===")
        print(build_agent_system_prompt(cfg.system_prompt_with_identity, reg,
                                        reasoning=cfg.reasoning_visible))
        print(f"\n=== model: {cfg.model} | context: {cfg.context_size} | "
              f"project: {cfg.project_dir or '(sandbox workspace)'} ===")
        return

    # Must run before get_free_port(), which deliberately returns *unused* ports
    # and would therefore report on the wrong ones.
    if args.doctor:
        doctor(args.web_port, args.model_port)
        return

    if args.bench:
        ensure_dirs()
        sys.exit(benchmark(
            Config(model=args.model, model_port=args.model_port),
            baseline_path=DATA_DIR / "bench_baseline.json",
            save=args.bench_save,
        ))

    if args.list_models or args.list_tasks or args.add_task:
        ensure_dirs()
        early_db = Database(DB_PATH)
        try:
            if args.list_models:
                catalog_config = Config(model=args.model, model_catalog=args.model_catalog)
                for entry in model_catalog(catalog_config):
                    marker = "*" if entry["current"] else " "
                    state = "cached" if entry["cached"] else "not downloaded"
                    print(f"{marker} {entry['id']}  ({state})")
                adapters = list_adapters()
                print("\nadapters: " + (", ".join(
                    a["id"] + (f" [{a['base_model']}]" if a.get("base_model") else "")
                    for a in adapters) or "none"))
                print(f"cache   : {hf_cache_dir()}")
            if args.add_task:
                try:
                    spec = json.loads(args.add_task)
                except json.JSONDecodeError as exc:
                    sys.exit(f"--add-task is not valid JSON: {exc}")
                if not spec.get("name") or not spec.get("goal"):
                    sys.exit("--add-task needs at least a name and a goal.")
                created = early_db.create_task(**spec)
                print(f"Created task {created['id']}: {created['name']}")
            if args.list_tasks:
                for task in early_db.list_tasks():
                    schedule = (f"every {task['interval_seconds']}s"
                                if task["interval_seconds"] else "manual")
                    print(f"{task['id']}  {task['name']}  [{schedule}] "
                          f"{'enabled' if task['enabled'] else 'disabled'}  "
                          f"runs={task['run_count']}  last={task['last_status'] or 'never'}")
                    print(f"    goal: {task['goal'][:120]}")
        finally:
            early_db.close()
        return

    bootstrap()
    import uvicorn

    ensure_dirs()

    ui_problems = check_ui_syntax()
    if ui_problems:
        log("Embedded UI is malformed; the page would load but do nothing:", logging.ERROR)
        for problem in ui_problems:
            log(f"  {problem}", logging.ERROR)
        sys.exit("Refusing to serve a broken UI.")

    config = build_config(args)

    # Upgrade the plain bootstrap logger into the structured, redacting,
    # rotating pipeline now that we have the resolved configuration.
    configure_logging(config, force=True)

    config.model_port = get_free_port(config.model_port)
    config.web_port = get_free_port(config.web_port, exclude={config.model_port})

    # Announce the model and where the RAM-based default came from, so it is
    # obvious on a new machine why a particular model was chosen.
    if os.environ.get("MODEL_ID"):
        log(f"Model: {config.model} (from MODEL_ID)")
    elif args.model == DEFAULT_MODEL:
        log(f"Model: {config.model} (auto-selected for {TOTAL_RAM_GB:.0f}GB RAM; "
            f"set MODEL_ID to override)")
    else:
        log(f"Model: {config.model}")
    if os.environ.get("CONTEXT_SIZE"):
        log(f"Context: {config.context_size} tokens (from CONTEXT_SIZE)")
    else:
        log(f"Context: {config.context_size} tokens "
            f"(auto-sized for {TOTAL_RAM_GB:.0f}GB RAM; large prompts chunk above "
            f"~{int(config.context_size * config.chunk_trigger_ratio)} tokens)")

    db = Database(DB_PATH)
    registry = ToolRegistry(config, db)
    log("Structured error handling active: unhandled API errors return JSON, not bare 500s.")
    if config.project_dir:
        root = registry._root()
        in_workspace = root == WORKSPACE_DIR.resolve()
        if in_workspace:
            log(f"PROJECT_DIR {config.project_dir!r} not found; file tools use ./workspace",
                logging.WARNING)
        else:
            is_repo = (root / ".git").exists()
            log(f"Codebase: {root} (file tools edit here in place)")
            if not is_repo:
                log("Codebase is not a git repository; you will have no easy way "
                    "to review or revert the agent's edits. A git repo is strongly "
                    "recommended.", logging.WARNING)

    if args.list_tools:
        for tool in registry.specs():
            print(f"{tool['name']}: {tool['description']}")
            for name, desc in tool["parameters"].items():
                flag = " (required)" if name in tool["required"] else ""
                print(f"    {name}{flag}: {desc}")
        db.close()
        return

    if args.tool_test:
        try:
            tool_args = json.loads(args.tool_args)
        except json.JSONDecodeError as exc:
            db.close()
            sys.exit(f"--tool-args is not valid JSON: {exc}")
        result, error = registry.call(args.tool_test, tool_args)
        print(result)
        db.close()
        sys.exit(1 if error else 0)

    if config.seed_demo:
        db.seed_demo()

    if config.list_feedback:
        import pprint
        stats = db.get_stats()
        print("\n=== Feedback Statistics ===")
        pprint.pprint(stats)
        print("\n=== Recent Feedback ===")
        for item in db.list_feedback(limit=100):
            pprint.pprint(item)
        db.close()
        return

    model_manager = ModelServerManager(
        config.model, config.model_port, ADAPTER_DIR, config.max_kv_size, config.adapter,
        kv_bits=config.kv_bits, kv_group_size=config.kv_group_size,
        quantized_kv_start=config.quantized_kv_start,
        prompt_cache_dir=config.prompt_cache_dir,
    )
    retrain_manager = RetrainManager(db, model_manager, config)

    if config.export_only:
        count, _ = retrain_manager.export_feedback()
        log(f"Exported {count} training examples to {SFT_DIR}")
        if config.export_format == "csv":
            csv_count = export_to_csv(db, DATA_DIR / "feedback_export.csv")
            log(f"Exported {csv_count} rows to {DATA_DIR / 'feedback_export.csv'}")
        db.close()
        return

    app = create_app(config, db, model_manager, retrain_manager, registry)

    shutting_down = threading.Event()

    def handle_signal(signum, frame):
        if shutting_down.is_set():
            return
        shutting_down.set()
        log(f"Received signal {signum}, shutting down...")
        model_manager.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"Model: {config.model}")
    log(f"System prompt: {config.system_prompt[:60]}...")
    log(f"Context: {config.context_size} tokens, max_tokens {config.max_tokens}, temperature {config.temperature}")
    log(f"Agent: {'on' if config.agent_enabled else 'off'}, max steps {config.agent_max_steps}, "
        f"tools: {', '.join(registry.names())}")
    log(f"Adapter: {config.adapter} ({model_manager.adapter_path() or 'base model'})")
    scheduled = [t for t in db.list_tasks() if t["enabled"] and t["interval_seconds"]]
    log(f"Tasks: {len(db.list_tasks())} defined, {len(scheduled)} on a schedule, "
        f"{config.max_concurrent_tasks} at a time")
    if config.allow_shell or config.allow_python:
        log("Code execution tools are enabled. The model can run commands on this machine.", logging.WARNING)
    log("=" * 62)
    log(f"  OPEN THIS IN YOUR BROWSER:  http://127.0.0.1:{config.web_port}")
    log(f"  Model backend (not a UI):   http://127.0.0.1:{config.model_port}")
    log("=" * 62)
    log("Ports shift automatically when the preferred one is busy, so use the URL above.")
    log("Starting model server. First run may download the model.")

    try:
        model_manager.start()
    except Exception as exc:
        log(f"Model server failed to start: {exc}", logging.ERROR)
        log("Web UI will still start. Check logs/model_server.log.", logging.WARNING)

    if config.retrain_now:
        def delayed_retrain():
            time.sleep(3)
            retrain_manager.run("cli")
        guarded_thread(delayed_retrain).start()

    try:
        uvicorn.run(app, host="127.0.0.1", port=config.web_port, log_level="warning", access_log=False)
    except KeyboardInterrupt:
        log("Shutting down...")
    finally:
        model_manager.stop()
        db.close()


if __name__ == "__main__":
    main()



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'build_config',
    'main',
]
