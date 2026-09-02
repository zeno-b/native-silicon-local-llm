"""Doctor, benchmark and CSV export.

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
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403


def export_to_csv(db: Database, path: Path) -> int:
    """Export all feedback to CSV."""
    rows = db.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    if not rows:
        return 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return len(rows)


def doctor(web_port: int, model_port: int) -> None:
    """Probe the loopback ports and report what is actually answering on each."""
    import urllib.error
    import urllib.request

    def probe(port: int, path: str) -> str:
        url = f"http://127.0.0.1:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return f"{resp.status} {resp.headers.get('Content-Type')} {resp.read(90)!r}"
        except urllib.error.HTTPError as exc:
            return f"{exc.code} {exc.headers.get('Content-Type')} {exc.read(90)!r}"
        except Exception as exc:
            return f"no response ({exc})"

    print(f"python      : {sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    # Report what the user actually ran, not whichever module this line lives in.
    _entry = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path(__file__).resolve()
    print(f"entry point : {_entry}")
    print(f"code        : {Path(__file__).resolve().parent}")
    print(f"ui build    : {UI_BUILD}")
    print(f"in venv     : {in_venv()}  ({sys.prefix})")

    print(f"\nweb UI port {web_port}: {'LISTENING' if port_open(web_port) else 'CLOSED'}")
    if port_open(web_port):
        for path in ["/", "/api/health"]:
            print(f"  GET {path:14} -> {probe(web_port, path)}")

    print(f"\nmodel port {model_port}: {'LISTENING' if port_open(model_port) else 'CLOSED'}")
    if port_open(model_port):
        print(f"  GET {'/v1/models':14} -> {probe(model_port, '/v1/models')}")
        print(f"  GET {'/':14} -> {probe(model_port, '/')}")

    others = [p for p in range(8000, 8101) if p not in (web_port, model_port) and port_open(p)]
    print(f"\nother loopback listeners in 8000-8100: {others or 'none'}")
    print("\nA healthy web UI answers GET / with 200 text/html and")
    print("GET /api/health with 200 application/json. Anything else is the wrong port.")


def benchmark(
    config: Config,
    prompts: int = 3,
    prompt_tokens: int = 512,
    baseline_path: Path | None = None,
    save: bool = False,
) -> int:
    """Measure prefill and decode against a running model server.

    Reports time to first token separately from decode throughput, because on
    this hardware they respond to completely different fixes: TTFT is prompt
    processing and scales with how much context the agent re-sends, decode is
    memory bandwidth and scales with model size. A change that helps one often
    does nothing for the other.
    """
    import urllib.error
    import urllib.request

    if not port_open(config.model_port):
        print(f"No model server on port {config.model_port}. Start the app first.")
        return 1

    sizes = [64, prompt_tokens, prompt_tokens * 4]
    # Size the filler for the LARGEST prompt, or the biggest sample silently
    # runs short and the prefill curve looks flatter than it is.
    sentence = "The quick brown fox jumps over the lazy dog. "
    repeats = (max(sizes) * CHARS_PER_TOKEN) // len(sentence) + 2
    filler = sentence * repeats
    print(f"model : {config.model}")
    print(f"port  : {config.model_port}")
    print()
    print(f"{'prompt tok':>11}  {'ttft ms':>8}  {'decode tok/s':>12}  {'total ms':>9}")

    failures = 0
    samples: list[dict] = []
    for size in sizes:
        text = filler[: size * CHARS_PER_TOKEN]
        body = json.dumps({
            "model": config.model,
            "messages": [
                {"role": "user", "content": text + "\n\nReply with exactly one short sentence."}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{config.model_port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.time()
        first_at: float | None = None
        completion = 0
        prompt_reported = 0
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    if data.get("usage"):
                        prompt_reported = int(data["usage"].get("prompt_tokens") or 0)
                        completion = int(data["usage"].get("completion_tokens") or completion)
                    choices = data.get("choices") or []
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        if first_at is None:
                            first_at = time.time()
                        completion = completion or 0
                        completion += 1
        except Exception as exc:
            print(f"{size:>11}  request failed: {exc}")
            failures += 1
            continue

        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_at - started) * 1000) if first_at else total_ms
        decode_ms = max(1.0, total_ms - ttft_ms)
        tps = completion / (decode_ms / 1000.0)
        samples.append({
            "requested": size,
            "prompt_tokens": prompt_reported or size,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 2),
            "total_ms": round(total_ms, 1),
        })
        print(f"{prompt_reported or size:>11}  {ttft_ms:>8.0f}  {tps:>12.1f}  {total_ms:>9.0f}")

    print()
    if baseline_path is not None:
        _report_against_baseline(baseline_path, config.model, samples, save)

    print("TTFT rising steeply with prompt size is re-prefill cost. That is what")
    print("prompt caching, smaller tool results and a tighter tool catalogue attack.")
    print("Flat TTFT with low tok/s is memory bandwidth: use a smaller model.")
    return 1 if failures == len(sizes) else 0


def _report_against_baseline(path: Path, model: str, samples: list[dict], save: bool) -> None:
    """Compare this run to a saved one. Optimizing without this is guessing."""
    previous: dict = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"Could not read the benchmark baseline: {exc}", logging.WARNING)

    prior = previous.get("samples") if previous.get("model") == model else None
    if prior:
        print(f"vs baseline from {previous.get('recorded_at', 'unknown')}:")
        print(f"{'prompt tok':>11}  {'ttft delta':>12}  {'tok/s delta':>12}")
        by_size = {sample["requested"]: sample for sample in prior}
        for sample in samples:
            was = by_size.get(sample["requested"])
            if not was:
                continue
            ttft_delta = sample["ttft_ms"] - was["ttft_ms"]
            tps_delta = sample["tps"] - was["tps"]
            print(f"{sample['prompt_tokens']:>11}  {ttft_delta:>+11.0f}ms  {tps_delta:>+12.1f}")
        print()
    elif previous:
        print(f"(baseline is for {previous.get('model')}, not comparing)\n")

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": model,
            "recorded_at": iso(utc_now()),
            "samples": samples,
        }, indent=2), encoding="utf-8")
        print(f"Baseline saved to {path}. Re-run --bench after a change to compare.")



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    '_report_against_baseline',
    'benchmark',
    'doctor',
    'export_to_csv',
]
