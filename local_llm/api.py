"""FastAPI application: every HTTP endpoint.

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
from .llm import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .tasks import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403
from .cluster import *  # noqa: F401,F403
from .claude_import import *  # noqa: F401,F403

# Request/Response must be resolvable at MODULE scope: with `from __future__
# import annotations` every endpoint annotation is a string that FastAPI resolves
# against this module's globals. If Request/Response are only imported inside
# create_app(), FastAPI cannot see them and treats `request`/`response`
# parameters as query fields (every such endpoint 422s). Guarded so the module
# still imports for --selftest on a machine without FastAPI installed.
try:
    from fastapi import Request, Response  # noqa: F401
except Exception:  # pragma: no cover
    Request = Any  # type: ignore
    Response = Any  # type: ignore


def _define_api_models() -> None:
    """Define the Pydantic models in the module namespace.

    This module uses `from __future__ import annotations`, so every parameter
    annotation is a string at runtime. FastAPI resolves those strings against the
    endpoint function's __globals__, which is the module namespace. Models defined
    inside create_app() are invisible there, so FastAPI silently falls back to
    treating the body parameter as a query parameter and every POST returns 422.
    """
    global ChatRequest, FeedbackRequest, ChatResponse, ConfigRequest, ToolRequest
    global MemoryRequest, TaskRequest, TaskUpdateRequest, ModelSelectRequest
    if ChatRequest is not None:
        return
    from pydantic import BaseModel, Field

    class ChatRequest(BaseModel):  # noqa: F811
        message: str = Field(..., min_length=1, max_length=32000)
        conversation_id: str | None = Field(None, max_length=64)
        agent: bool | None = None
        max_tokens: int | None = Field(None, ge=16, le=32768)
        temperature: float | None = Field(None, ge=0.0, le=2.0)
        use_history: bool = True

    class FeedbackRequest(BaseModel):  # noqa: F811
        user_prompt: str = Field(..., min_length=1)
        assistant_response: str = Field(..., min_length=1)
        rating: int = Field(0, ge=-1, le=1)
        corrected_response: str | None = None

    class ChatResponse(BaseModel):  # noqa: F811
        answer: str

    class ConfigRequest(BaseModel):  # noqa: F811
        system_prompt: str | None = Field(None, max_length=8000)
        max_tokens: int | None = Field(None, ge=16, le=32768)
        temperature: float | None = Field(None, ge=0.0, le=2.0)
        context_size: int | None = Field(None, ge=512, le=1048576)
        history_turns: int | None = Field(None, ge=0, le=200)
        agent_enabled: bool | None = None
        agent_max_steps: int | None = Field(None, ge=1, le=20)
        # search_backend is intentionally absent: the provider is locked to
        # DuckDuckGo Lite and cannot be set via the API.
        search_results: int | None = Field(None, ge=1, le=10)
        tool_result_chars: int | None = Field(None, ge=200, le=40000)
        tool_raw_chars: int | None = Field(None, ge=200, le=200000)
        tool_temperature: float | None = Field(None, ge=0.0, le=2.0)
        disable_thinking: bool | None = None
        fast_path: bool | None = None
        stable_prefix: bool | None = None
        summarise_tool_results: bool | None = None
        summarise_over_chars: int | None = Field(None, ge=500, le=100000)

    class ToolRequest(BaseModel):  # noqa: F811
        name: str = Field(..., min_length=1, max_length=64)
        args: dict = Field(default_factory=dict)
        conversation_id: str | None = Field(None, max_length=64)

    class MemoryRequest(BaseModel):  # noqa: F811
        key: str = Field(..., min_length=1, max_length=120)
        value: str = Field(..., min_length=1, max_length=8000)

    class TaskRequest(BaseModel):  # noqa: F811
        name: str = Field(..., min_length=1, max_length=120)
        goal: str = Field(..., min_length=1, max_length=8000)
        enabled: bool = True
        interval_seconds: int = Field(0, ge=0, le=2_592_000)
        max_steps: int = Field(6, ge=1, le=20)
        tools: str = Field("", max_length=500)
        system_prompt: str | None = Field(None, max_length=8000)
        use_history: bool = False
        # Run this task on its own model, swapping back afterwards. Empty means
        # whatever the server is already serving.
        model: str | None = Field(None, max_length=200)
        # Run another task when this one succeeds, handing the answer over
        # through a workspace file.
        next_task_id: str | None = Field(None, max_length=64)

    class TaskUpdateRequest(BaseModel):  # noqa: F811
        name: str | None = Field(None, min_length=1, max_length=120)
        goal: str | None = Field(None, min_length=1, max_length=8000)
        enabled: bool | None = None
        interval_seconds: int | None = Field(None, ge=0, le=2_592_000)
        max_steps: int | None = Field(None, ge=1, le=20)
        tools: str | None = Field(None, max_length=500)
        system_prompt: str | None = Field(None, max_length=8000)
        use_history: bool | None = None
        model: str | None = Field(None, max_length=200)
        next_task_id: str | None = Field(None, max_length=64)

    class ModelSelectRequest(BaseModel):  # noqa: F811
        model: str | None = Field(None, min_length=1, max_length=200)
        adapter: str | None = Field(None, max_length=100)
        max_kv_size: int | None = Field(None, ge=0, le=1_048_576)
        restart: bool = True


def create_app(
    config: Config,
    db: Database,
    model_manager: ModelServerManager,
    retrain_manager: RetrainManager,
    registry: ToolRegistry | None = None,
):
    """Create and configure the FastAPI application with Pydantic validation."""
    from fastapi import FastAPI, Query, UploadFile, File, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (HTMLResponse, JSONResponse, StreamingResponse,
                                   Response, RedirectResponse)

    _define_api_models()

    from contextlib import asynccontextmanager

    # ---- Cluster router (Mac Mini primary / Mac Studio secondary) ---------- #
    node_registry = NodeRegistry(config)
    router = ClusterRouter(config, node_registry, db)
    health_monitor = HealthMonitor(
        config, node_registry,
        local_status=lambda: {"status": model_manager.status, "model": config.model})

    registry = registry or ToolRegistry(config, db)
    model_client = ModelClient(config, cluster=router)
    agent = Agent(config, registry, model_client)
    tasks = TaskManager(config, db, model_manager, retrain_manager, model_client)
    # Task runs build their own ModelClient per run; give them the same router so
    # scheduled work is load-balanced and fails over just like interactive chat.
    tasks.cluster = router

    # ---- Auth + Claude-history import ------------------------------------- #
    auth = Auth(config, db)
    auth.bootstrap()
    importer = ImportManager(config, db)

    @asynccontextmanager
    async def lifespan(_app):
        health_monitor.start()
        await tasks.start()
        try:
            yield
        finally:
            await tasks.stop()
            health_monitor.stop()

    app = FastAPI(title=f"{APP_NAME}", lifespan=lifespan)
    # Exposed so tests and the CLI can reach the scheduler without a global.
    app.state.tasks = tasks
    app.state.auth = auth
    app.state.router = router
    app.state.node_registry = node_registry
    app.state.importer = importer

    # RBAC dependency shortcuts. USER = any authenticated user (synthetic local
    # admin when auth is disabled); ADMIN = must have the admin role. Enforced
    # server-side on every protected route, not merely hidden in the UI.
    USER = Depends(auth.require_user)
    ADMIN = Depends(auth.require_admin)

    # Catch-all so no endpoint can ever return a bare, bodyless 500. Any
    # unhandled error becomes a structured JSON body with the request path and a
    # short reason, and is logged with a traceback for debugging. HTTPExceptions
    # (deliberate 4xx/404 etc.) keep their own handling and are not caught here.
    from starlette.requests import Request as _Request

    _req_log = get_logger("request")

    @app.exception_handler(Exception)
    async def _unhandled(request: _Request, exc: Exception):
        cid = get_correlation_id()
        # Redact the exception text: a stack/message can carry a URL with
        # embedded credentials, a token echoed from a tool, etc.
        safe = redact_text(f"{type(exc).__name__}: {exc}")
        log_event(_req_log, logging.ERROR, "http.unhandled_error",
                  method=request.method, route=request.url.path,
                  error=safe, correlation_id=cid)
        _req_log.log(logging.DEBUG, redact_text(traceback.format_exc()))
        return JSONResponse(
            status_code=500,
            content={
                "error": safe,
                "path": request.url.path,
                "correlation_id": cid,
                "hint": "This was logged. Quote the correlation_id when reporting it.",
            },
            headers={"X-Correlation-ID": cid or ""},
        )

    # The UI is same-origin, so only the loopback origins this process serves are allowed.
    # A wildcard here would let any site the user visits drive /api/retrain and
    # DELETE /api/feedback on their machine.
    # Same-origin by default (loopback). Cookie-based sessions need credentials
    # enabled. Behind a reverse proxy set ALLOWED_ORIGINS to the public origin(s)
    # (comma-separated). A wildcard is never combined with credentials.
    _origins = [
        f"http://127.0.0.1:{config.web_port}",
        f"http://localhost:{config.web_port}",
    ]
    for extra in os.environ.get("ALLOWED_ORIGINS", "").split(","):
        extra = extra.strip()
        if extra and extra not in _origins:
            _origins.append(extra)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
    )

    # The UI is embedded in this file, so a cached copy silently defeats every
    # edit to HTML_PAGE. Nothing here is worth caching on a local dev server.
    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    # Correlation-id + request-lifecycle logging. Registered after no_cache so it
    # is the OUTERMOST middleware: the id is bound before any handler runs and
    # every response carries it back (X-Correlation-ID) for support/debugging.
    # A client may supply its own id via X-Correlation-ID / X-Request-ID.
    @app.middleware("http")
    async def request_context(request, call_next):
        incoming = (request.headers.get("x-correlation-id")
                    or request.headers.get("x-request-id"))
        cid = set_correlation_id(incoming)
        bind_context(route=request.url.path, method=request.method)
        # Expose to handlers/dependencies without relying on contextvar
        # propagation across the middleware boundary.
        request.state.correlation_id = cid
        started = time.time()
        try:
            response = await call_next(request)
        except Exception:
            log_event(_req_log, logging.ERROR, "http.request",
                      method=request.method, route=request.url.path, status=500,
                      duration_ms=round((time.time() - started) * 1000, 1),
                      client=(request.client.host if request.client else None),
                      correlation_id=cid)
            clear_context()
            raise
        response.headers["X-Correlation-ID"] = cid
        # The index page and SSE streams are high-volume/low-signal at INFO.
        path = request.url.path
        level = logging.DEBUG if (path == "/" or path.endswith("/stream")) else logging.INFO
        log_event(_req_log, level, "http.request",
                  method=request.method, route=path,
                  status=response.status_code,
                  duration_ms=round((time.time() - started) * 1000, 1),
                  client=(request.client.host if request.client else None),
                  correlation_id=cid)
        clear_context()
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(content=render_ui())

    # A local-only client for the node-worker endpoint: it must talk to THIS
    # machine's mlx server directly, never route back through the cluster.
    node_worker_client = ModelClient(config)

    def _safe_user(user: dict | None) -> dict | None:
        if not user:
            return None
        return {k: v for k, v in user.items() if k not in ("password_hash",)}

    # -------------------------------------------------------------- auth --- #
    @app.get("/api/auth/config")
    def auth_config():
        """Public: what the login page needs to render (no secrets)."""
        return public_auth_config(config, auth)

    @app.post("/api/auth/login")
    async def auth_login(body: dict, request: Request, response: Response):
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        user = await asyncio.to_thread(auth.authenticate_local, username, password)
        if not user:
            log_event(get_logger("auth"), logging.WARNING, "auth.login_failed",
                      username=username,
                      client=(request.client.host if request.client else None))
            return JSONResponse({"error": "invalid username or password"}, status_code=401)
        token = auth.create_session(user, request)
        auth.set_cookie(response, token)
        log_event(get_logger("auth"), logging.INFO, "auth.login",
                  user_id=user["id"], username=user["username"], role=user["role"])
        return {"user": _safe_user(user)}

    @app.post("/api/auth/logout")
    def auth_logout(request: Request, response: Response):
        token = auth._token_from_request(request)
        auth.logout(token)
        auth.clear_cookie(response)
        return {"ok": True}

    @app.get("/api/auth/me")
    def auth_me(request: Request):
        user = auth.user_for_request(request)
        if not user:
            return {"authenticated": False, "auth_enabled": config.auth_enabled,
                    "oidc_enabled": auth.oidc_enabled}
        return {"authenticated": True, "auth_enabled": config.auth_enabled,
                "oidc_enabled": auth.oidc_enabled, "user": _safe_user(user)}

    @app.get("/api/auth/oidc/login")
    def oidc_login(request: Request):
        if not auth.oidc_enabled:
            return JSONResponse({"error": "OIDC/Entra is not configured"}, status_code=400)
        redirect_uri = (config.oidc_redirect_uri
                        or f"{request.url.scheme}://{request.url.netloc}/api/auth/oidc/callback")
        try:
            url = auth.oidc_authorize_url(redirect_uri)
        except Exception as exc:
            return JSONResponse({"error": f"OIDC discovery failed: {redact_text(str(exc))}"},
                                status_code=502)
        return RedirectResponse(url)

    @app.get("/api/auth/oidc/callback")
    async def oidc_callback(request: Request, code: str = "", state: str = "",
                            error: str = ""):
        if error or not code:
            return HTMLResponse(f"<p>Login failed: {html.escape(error or 'no code')}. "
                                "<a href='/'>Back</a></p>", status_code=400)
        redirect_uri = (config.oidc_redirect_uri
                        or f"{request.url.scheme}://{request.url.netloc}/api/auth/oidc/callback")
        try:
            user = await asyncio.to_thread(auth.oidc_exchange, code, state, redirect_uri)
        except Exception as exc:
            return HTMLResponse(f"<p>Login failed: {html.escape(redact_text(str(exc)))}. "
                                "<a href='/'>Back</a></p>", status_code=400)
        token = auth.create_session(user, request)
        resp = RedirectResponse("/", status_code=303)
        auth.set_cookie(resp, token)
        log_event(get_logger("auth"), logging.INFO, "auth.oidc_login",
                  user_id=user["id"], role=user["role"])
        return resp

    # ------------------------------------------------------ user admin ---- #
    @app.get("/api/users")
    def users_list(_a: dict = ADMIN):
        return {"users": [_safe_user(u) for u in db.list_users()]}

    @app.post("/api/users")
    def users_create(body: dict, _a: dict = ADMIN):
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        role = "admin" if body.get("role") == "admin" else "user"
        if not username or not password:
            return JSONResponse({"error": "username and password are required"}, status_code=400)
        if db.get_user_by_username(username):
            return JSONResponse({"error": "username already exists"}, status_code=409)
        user = db.create_user(username, password_hash=hash_password(password),
                              role=role, source="local")
        log_event(get_logger("auth"), logging.INFO, "auth.user_created",
                  username=username, role=role, user_id=user["id"])
        return {"user": _safe_user(user)}

    @app.post("/api/users/{user_id}")
    def users_update(user_id: str, body: dict, _a: dict = ADMIN):
        target = db.get_user(user_id)
        if not target:
            return JSONResponse({"error": "no such user"}, status_code=404)
        updates: dict = {}
        if body.get("role") in ("admin", "user"):
            updates["role"] = body["role"]
        if "disabled" in body:
            updates["disabled"] = bool(body["disabled"])
        if body.get("password"):
            updates["password_hash"] = hash_password(str(body["password"]))
        if not updates:
            return JSONResponse({"error": "nothing to update"}, status_code=400)
        # Never lock everyone out: keep at least one enabled admin.
        losing_admin = (target["role"] == "admin"
                        and (updates.get("role") == "user" or updates.get("disabled")))
        if losing_admin and db.count_users(role="admin") <= 1:
            return JSONResponse({"error": "cannot demote or disable the last admin"},
                                status_code=400)
        user = db.update_user(user_id, **updates)
        if updates.get("disabled") or updates.get("role") == "user" or updates.get("password_hash"):
            db.delete_user_sessions(user_id)  # force re-login on privilege change
        return {"user": _safe_user(user)}

    @app.delete("/api/users/{user_id}")
    def users_delete(user_id: str, _a: dict = ADMIN):
        target = db.get_user(user_id)
        if not target:
            return JSONResponse({"error": "no such user"}, status_code=404)
        if target["role"] == "admin" and db.count_users(role="admin") <= 1:
            return JSONResponse({"error": "cannot delete the last admin"}, status_code=400)
        return {"deleted": db.delete_user(user_id)}

    # --------------------------------------------- cluster / routing ------ #
    @app.get("/api/cluster/nodes")
    def cluster_nodes(_a: dict = ADMIN):
        return {"nodes": node_registry.snapshot(),
                "multi_node": node_registry.multi_node,
                "node_role": config.node_role,
                "routing_summary": db.routing_summary()}

    @app.get("/api/routing/events")
    def routing_events(limit: int = Query(100, ge=1, le=1000),
                       node: str | None = Query(None, max_length=64),
                       task_id: str | None = Query(None, max_length=64),
                       _a: dict = ADMIN):
        return {"events": db.list_routing_events(limit, node, task_id)}

    @app.get("/api/routing/summary")
    def routing_summary(_a: dict = ADMIN):
        return db.routing_summary()

    # -------------------------------------------------- node worker ------- #
    # Called by a primary to run a raw generation on THIS node's local model.
    # Guarded by the shared NODE_TOKEN so the mlx server itself stays unexposed.
    @app.get("/api/node/health")
    def node_health(request: Request):
        if config.node_token and not auth.check_node_token(request):
            return JSONResponse({"error": "node token required"}, status_code=401)
        cpu, mem = sample_local_load()
        return {"state": model_manager.status, "model_status": model_manager.status,
                "model": config.model, "cpu_pct": cpu, "mem_pct": mem,
                "node": config.node_name or config.node_role,
                "active": node_registry.local_node().active}

    @app.post("/api/node/generate")
    async def node_generate(body: dict, request: Request):
        if not config.node_token or not auth.check_node_token(request):
            return JSONResponse({"error": "node token required"}, status_code=401)
        messages = body.get("messages") or []
        max_tokens = int(body.get("max_tokens") or config.max_tokens)
        temperature = float(body.get("temperature", config.temperature))
        made = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        if body.get("stream"):
            async def gen():
                def chunk(delta, finish=None, usage=None):
                    payload = {"id": made, "object": "chat.completion.chunk",
                               "created": created, "model": config.model,
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage:
                        payload["usage"] = usage
                    return f"data: {json.dumps(payload)}\n\n"
                yield chunk({"role": "assistant", "content": ""})
                stats = GenerationStats()
                try:
                    async for tok in node_worker_client.stream(messages, max_tokens, temperature, stats):
                        yield chunk({"content": tok})
                except Exception as exc:
                    yield chunk({"content": f"\n[node error: {type(exc).__name__}]"})
                yield chunk({}, finish="stop",
                            usage={"prompt_tokens": stats.prompt_tokens,
                                   "completion_tokens": stats.completion_tokens})
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        text, stats = await node_worker_client.complete_with_stats(messages, max_tokens, temperature)
        return {"id": made, "object": "chat.completion", "created": created,
                "model": config.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": stats.prompt_tokens,
                          "completion_tokens": stats.completion_tokens}}

    # --------------------------------------------- Claude history import -- #
    @app.post("/api/import")
    async def import_upload(request: Request, _a: dict = ADMIN):
        # Read the raw ZIP body (no multipart dependency). Reject early on an
        # oversized declared length before buffering it.
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared and declared > config.import_max_zip_bytes:
            return JSONResponse(
                {"error": f"upload exceeds IMPORT_MAX_ZIP_BYTES ({config.import_max_zip_bytes})"},
                status_code=413)
        data = await request.body()
        filename = request.headers.get("x-filename", "export.zip")
        user_id = get_acting_user() or SENTINEL_LOCAL_USER
        try:
            import_id = await asyncio.to_thread(importer.stage_upload, user_id, filename, data)
        except Exception as exc:
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        guarded_thread(importer.process, import_id, user_id).start()
        log_event(get_logger("import"), logging.INFO, "import.requested",
                  import_id=import_id, user_id=user_id, size_bytes=len(data))
        return {"import_id": import_id, "status": "processing"}

    @app.get("/api/imports")
    def imports_list(_a: dict = ADMIN):
        return {"imports": db.list_imports(limit=100)}

    @app.get("/api/imports/{import_id}")
    def import_status(import_id: str, _a: dict = ADMIN):
        record = db.get_import(import_id)
        if not record:
            return JSONResponse({"error": "no such import"}, status_code=404)
        return {"import": record}

    @app.post("/api/imports/{import_id}/retry")
    def import_retry(import_id: str, _a: dict = ADMIN):
        record = db.get_import(import_id)
        if not record:
            return JSONResponse({"error": "no such import"}, status_code=404)
        guarded_thread(importer.process, import_id, record["user_id"]).start()
        return {"import_id": import_id, "status": "processing"}

    @app.delete("/api/imports/{import_id}")
    def import_delete(import_id: str, _a: dict = ADMIN):
        return {"deleted": importer.remove_import(import_id)}

    @app.get("/api/health")
    async def health(user: dict = USER):
        is_admin = user.get("role") == "admin"
        model_healthy = await model_manager.health_probe() if model_manager.is_alive() else False
        # Everyone may see whether the service is up and which model/agent is
        # active; only admins see internal telemetry (feedback stats, prefix
        # cache state, retrain progress, per-node/task detail).
        payload = {
            "ui_build": UI_BUILD,
            "model_status": model_manager.status,
            "model_healthy": model_healthy,
            "agent_enabled": config.agent_enabled,
            "agent_max_steps": config.agent_max_steps,
            "model": config.model,
            "context_size": config.context_size,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "tools": registry.names(),
            "search_backend": config.search_backend,
            "memories": db.count_memories(user_id=user["id"]),
            "role": user.get("role"),
            "username": user.get("username"),
            "auth_enabled": config.auth_enabled,
        }
        if is_admin:
            payload.update({
                "web_port": config.web_port,
                "model_port": config.model_port,
                "model_process_alive": model_manager.is_alive(),
                "retrain": retrain_manager.status,
                "prefix": dict(PREFIX_STATE),
                "stats": db.get_stats(),
                "ram_gb": round(TOTAL_RAM_GB),
                "adapter": model_manager.adapter_choice,
                "node_role": config.node_role,
                "multi_node": node_registry.multi_node,
                "tasks": {
                    "total": len(db.list_tasks()),
                    "running": [
                        {"task_id": task_id, **(tasks.live_status(task_id) or {})}
                        for task_id in list(tasks.by_task)
                    ],
                },
            })
        return payload

    @app.get("/api/feedback")
    def list_feedback(
        limit: int = Query(50, ge=1, le=500),
        approved_only: bool = False,
        search: str | None = Query(None, max_length=100),
        _a: dict = ADMIN,
    ):
        return {"feedback": db.list_feedback(limit, approved_only, search)}

    @app.delete("/api/feedback/{feedback_id}")
    def delete_feedback(feedback_id: int, _a: dict = ADMIN):
        success = db.delete_feedback(feedback_id)
        return {"deleted": success}

    @app.post("/api/feedback/{feedback_id}/reviewed")
    def set_reviewed(feedback_id: int, reviewed: bool = True, _a: dict = ADMIN):
        return {"updated": db.set_reviewed(feedback_id, reviewed)}

    @app.get("/api/dataset/stats")
    def dataset_stats(_a: dict = ADMIN):
        return db.dataset_stats()

    @app.get("/api/docs/stats")
    def docs_stats(user: dict = USER):
        return db.document_stats(user_id=user["id"])

    @app.post("/api/docs/index")
    def docs_index(path: str = Query(...), _a: dict = ADMIN):
        try:
            return {"result": registry._index_docs(path), "stats": db.document_stats()}
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    @app.get("/api/docs/search")
    def docs_search(q: str = Query(...), limit: int = 5, user: dict = USER):
        return {"hits": db.search_documents(q, limit=limit, user_id=user["id"])}

    @app.post("/api/docs/clear")
    def docs_clear(_a: dict = ADMIN):
        return {"cleared": db.clear_documents()}

    try:
        import multipart  # noqa: F401  (python-multipart, required by FastAPI for uploads)
        _uploads_ok = True
    except ImportError:
        _uploads_ok = False
        log("File upload disabled: `pip install python-multipart` to enable drag-and-drop.",
            logging.WARNING)

    if _uploads_ok:
        @app.post("/api/docs/upload")
        async def docs_upload(file: UploadFile = File(...), _a: dict = ADMIN):
            """Accept a file from the browser and index it into the knowledge base.

            Written into the project's uploads/ folder first so the type-aware
            readers (pdf, docx, csv, notebook, ...) can parse it the same way they
            parse any other project file.
            """
            try:
                root = registry._root()
                dest_dir = root / "uploads"
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Take only the base name so "../../etc/passwd" cannot escape, and
                # reject a name that is empty or purely dots after stripping.
                name = Path(file.filename or "").name.strip()
                if not name or set(name) <= {"."}:
                    name = f"upload-{uuid.uuid4().hex[:8]}.bin"
                dest = dest_dir / name
                if dest.resolve().parent != dest_dir.resolve():
                    return JSONResponse({"error": "invalid file name"}, status_code=400)
                data = await file.read()
                if not data:
                    return JSONResponse({"error": "the uploaded file is empty"}, status_code=400)
                if len(data) > 25 * 1024 * 1024:
                    return JSONResponse({"error": "file is larger than 25 MB"}, status_code=400)
                dest.write_bytes(data)
                result = registry._index_docs(f"uploads/{name}")
                return {"result": result, "path": f"uploads/{name}", "stats": db.document_stats()}
            except Exception as exc:
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    @app.get("/api/uploads/enabled")
    def uploads_enabled(user: dict = USER):
        return {"enabled": _uploads_ok}

    @app.post("/api/docs/index_url")
    def docs_index_url(url: str = Query(...), _a: dict = ADMIN):
        try:
            return {"result": registry._index_url(url), "stats": db.document_stats()}
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    @app.get("/api/conversations/search")
    def conversations_search(q: str = Query(...), limit: int = 30, user: dict = USER):
        return {"results": db.search_conversations(q, limit=limit, user_id=user["id"])}

    @app.get("/api/conversation/{conversation_id}/export")
    def conversation_export(conversation_id: str,
                            format: str = Query("markdown", pattern="^(markdown|json)$"),
                            user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            text = db.export_conversation(conversation_id, format, user_id=user["id"])
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        ext = "md" if format == "markdown" else "json"
        return Response(
            content=text,
            media_type="text/markdown" if format == "markdown" else "application/json",
            headers={"Content-Disposition": f'attachment; filename="conversation-{conversation_id}.{ext}"'},
        )

    @app.get("/api/prompts")
    def prompts_list(user: dict = USER):
        return {"prompts": db.list_prompts()}

    @app.post("/api/prompts")
    def prompts_save(name: str = Query(...), body: str = Query(...), user: dict = USER):
        try:
            db.save_prompt(name, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"saved": name.strip(), "prompts": db.list_prompts()}

    @app.delete("/api/prompts/{name}")
    def prompts_delete(name: str, user: dict = USER):
        return {"deleted": db.delete_prompt(name)}

    @app.get("/api/backup")
    def backup_export(_a: dict = ADMIN):
        data = db.export_backup()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Response(
            content=json.dumps(data, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="backup-{stamp}.json"'},
        )

    @app.post("/api/backup/restore")
    async def backup_restore(body: dict, _a: dict = ADMIN):
        try:
            counts = db.import_backup(body)
            if not any(counts.values()):
                return JSONResponse(
                    {"error": "nothing was restored; the file did not contain any "
                              "recognisable conversations, prompts, feedback or documents",
                     "restored": counts},
                    status_code=400)
            return {"restored": counts}
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    @app.get("/api/browse")
    def browse_directories(path: str = Query(""), _a: dict = ADMIN):
        """List sub-directories so the UI can offer a real folder picker.

        The browser cannot hand back an absolute path (it only exposes file
        contents, never locations), so directory selection has to be served from
        this side. Read-only: names and paths, never file contents.
        """
        try:
            base = Path(path).expanduser() if path.strip() else Path.home()
            base = base.resolve()
            if not base.is_dir():
                return JSONResponse({"error": f"{base} is not a directory"}, status_code=400)
            entries = []
            for child in sorted(base.iterdir(), key=lambda c: c.name.lower()):
                if child.name.startswith(".") and child.name != ".git":
                    continue
                try:
                    if child.is_dir():
                        entries.append({
                            "name": child.name,
                            "path": str(child),
                            "is_git_repo": (child / ".git").exists(),
                        })
                except OSError:
                    continue
                if len(entries) >= 300:
                    break
            return {
                "path": str(base),
                "parent": str(base.parent) if base.parent != base else None,
                "home": str(Path.home()),
                "is_git_repo": (base / ".git").exists(),
                "entries": entries,
            }
        except PermissionError:
            return JSONResponse({"error": f"no permission to read {path}"}, status_code=403)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    @app.post("/api/conversation/{conversation_id}/fork")
    def conversation_fork(conversation_id: str, upto: int | None = None, user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            return {"conversation_id": db.fork_conversation(conversation_id, upto,
                                                            user_id=user["id"])}
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/conversation/{conversation_id}/title")
    def conversation_title(conversation_id: str, title: str = Query(...), user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        db.set_conversation_title(conversation_id, title)
        return {"conversation_id": conversation_id, "title": title}

    @app.post("/api/conversation/{conversation_id}/pin")
    def conversation_pin(conversation_id: str, pinned: bool = True, user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        db.set_conversation_pinned(conversation_id, pinned)
        return {"conversation_id": conversation_id, "pinned": pinned}

    @app.post("/api/docs/scope")
    def docs_scope(paths: str = Query(""), _a: dict = ADMIN):
        """Restrict retrieval to selected documents ('' = whole knowledge base)."""
        wanted = [p.strip() for p in (paths or "").split(",") if p.strip()]
        known = {d["path"] for d in db.document_stats().get("items", [])}
        unknown = [p for p in wanted if p not in known]
        if unknown:
            # Scoping to a path that was never indexed would silently return no
            # passages, which reads as "the model ignored my documents".
            return JSONResponse(
                {"error": "these documents are not in the knowledge base: "
                          + ", ".join(unknown[:5]),
                 "indexed": sorted(known)[:20]},
                status_code=400)
        changed = config.apply({"rag_scope": ",".join(wanted)})
        return {"scope": config.rag_scope, "changed": changed}

    @app.get("/api/docs/scope")
    def docs_scope_get(_a: dict = ADMIN):
        return {"scope": config.rag_scope}

    @app.post("/api/conversation/{conversation_id}/regenerate")
    def conversation_regenerate(conversation_id: str, user: dict = USER):
        """Drop the last answer and hand back the prompt that produced it, so the
        client can re-send and get a fresh response."""
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        prompt = db.drop_last_exchange(conversation_id, user_id=user["id"])
        if not prompt:
            return JSONResponse({"error": "nothing to regenerate"}, status_code=400)
        return {"prompt": prompt}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(body: dict, user: dict = USER):
        """OpenAI-compatible endpoint backed by this app's agent.

        Lets external tools (SDKs, editors, other front-ends) drive the full
        stack — routing, tools, knowledge base, retries — not just the raw model.
        Supports stream=true (OpenAI-shaped SSE deltas) and non-streaming.
        """
        try:
            messages = body.get("messages") or []
            prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    prompt = msg.get("content") or ""
                    break
            if not prompt:
                return JSONResponse({"error": {"message": "no user message provided"}},
                                    status_code=400)
            history = [{"role": m.get("role"), "content": m.get("content") or ""}
                       for m in messages[:-1] if m.get("role") in ("user", "assistant")]
            max_tokens = int(body.get("max_tokens") or config.max_tokens)
            temperature = float(body.get("temperature", config.temperature))
            cid = f"openai-{uuid.uuid4().hex[:8]}"

            # Streaming: emit OpenAI-shaped SSE deltas so editors and SDK clients
            # that expect stream=True (Continue, Zed, the OpenAI SDK) work.
            if body.get("stream"):
                async def sse_stream():
                    made = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    created = int(time.time())

                    def chunk(delta: dict, finish=None) -> str:
                        payload = {
                            "id": made, "object": "chat.completion.chunk",
                            "created": created, "model": config.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                        return f"data: {json.dumps(payload)}\n\n"

                    yield chunk({"role": "assistant", "content": ""})
                    sent = ""
                    try:
                        async for event in agent.run_iterating(prompt, history, cid,
                                                               max_tokens, temperature):
                            if event.get("type") == "token":
                                piece = event.get("token") or ""
                                if piece:
                                    sent += piece
                                    yield chunk({"content": piece})
                            elif event.get("type") == "final":
                                full = event.get("answer") or ""
                                # If the answer never streamed as tokens, send it now.
                                if full and not sent:
                                    yield chunk({"content": full})
                    except Exception as exc:
                        yield chunk({"content": f"\n[error: {type(exc).__name__}: {exc}]"})
                    yield chunk({}, finish="stop")
                    yield "data: [DONE]\n\n"

                return StreamingResponse(sse_stream(), media_type="text/event-stream")

            answer, used = "", []
            async for event in agent.run_iterating(prompt, history, cid,
                                                   max_tokens, temperature):
                if event.get("type") == "final":
                    answer = event.get("answer") or ""
                    used = event.get("tools_used") or []
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "x_tools_used": used,
            }
        except Exception as exc:
            return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}},
                                status_code=500)

    def _git(args: list[str], limit: int = 60000) -> tuple[bool, str]:
        """Run a read-only git command in the project dir. Returns (ok, output)."""
        root = registry._root()
        try:
            proc = subprocess.run(["git", "-C", str(root), *args],
                                  capture_output=True, text=True, timeout=15)
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, out[:limit]
        except Exception as exc:
            return False, str(exc)

    @app.get("/api/project/status")
    def project_status(_a: dict = ADMIN):
        root = str(registry._root())
        is_repo, _ = _git(["rev-parse", "--is-inside-work-tree"])
        branch = ""
        if is_repo:
            ok, out = _git(["rev-parse", "--abbrev-ref", "HEAD"])
            branch = out.strip() if ok else ""
        ok, status = _git(["status", "--porcelain"]) if is_repo else (False, "")
        return {
            "project_dir": config.project_dir,
            "root": root,
            "is_git_repo": is_repo,
            "branch": branch,
            "status": status,
            "changed_this_session": sorted(registry.changed_files),
            "allow_shell": config.allow_shell,
            "allow_python": config.allow_python,
            "test_command": registry._detect_test_command(),
        }

    @app.get("/api/project/diff")
    def project_diff(path: str = "", _a: dict = ADMIN):
        is_repo, _ = _git(["rev-parse", "--is-inside-work-tree"])
        if not is_repo:
            return {"is_git_repo": False, "diff": ""}
        args = ["diff", "--no-color"]
        if path:
            args += ["--", path]
        ok, diff = _git(args)
        return {"is_git_repo": True, "diff": diff}

    @app.post("/api/project/revert")
    def project_revert(_a: dict = ADMIN):
        """Undo this session's edits: restore modified tracked files and delete
        files created this session. Destructive; the UI confirms first."""
        root = registry._root()
        is_repo, _ = _git(["rev-parse", "--is-inside-work-tree"])
        if not is_repo:
            return JSONResponse({"error": "not a git repository; cannot revert"}, status_code=400)
        restored, removed, failed = [], [], []
        for rel in sorted(registry.changed_files):
            target = (root / rel)
            tracked, _ = _git(["ls-files", "--error-unmatch", rel])
            if tracked:
                ok, _ = _git(["checkout", "--", rel])
                (restored if ok else failed).append(rel)
            else:
                try:
                    if target.is_file():
                        target.unlink()
                    removed.append(rel)
                except Exception:
                    failed.append(rel)
        registry.changed_files.clear()
        return {"restored": restored, "removed": removed, "failed": failed}

    @app.get("/api/dataset/export")
    def dataset_export(
        format: str = Query("chat", pattern="^(chat|bare|preference|raw)$"),
        approved_only: bool = True,
        reviewed_only: bool = False,
        _a: dict = ADMIN,
    ):
        # For preference/raw we need rejected rows too, so don't force approved.
        want_approved = approved_only and format in ("chat", "bare")
        try:
            rows = db.export_rows(approved_only=want_approved, reviewed_only=reviewed_only)
            text, count = build_reusable_dataset(rows, format, config.system_prompt_with_identity)
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = EXPORTS_DIR / f"dataset_{format}.jsonl"
            path.write_text(text, encoding="utf-8")
        except Exception as exc:
            return JSONResponse(
                {"error": f"export failed: {type(exc).__name__}: {exc}"}, status_code=500)
        filename = f"dataset_{format}.jsonl"
        return Response(
            content=text,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Example-Count": str(count),
            },
        )

    def not_ready() -> JSONResponse | None:
        if retrain_manager.status.get("running"):
            return JSONResponse(content={"error": "Retraining in progress. Please wait."}, status_code=503)
        if model_manager.status != "ready":
            return JSONResponse(
                content={"error": f"Model not ready. Status: {model_manager.status}"},
                status_code=503,
            )
        return None

    def load_history(request, user_id: str) -> tuple[str, list[dict]]:
        conversation_id = request.conversation_id or str(uuid.uuid4())[:12]
        # Ownership: never let one user append to (or read) another user's
        # conversation by guessing its id. If the id belongs to someone else,
        # start a fresh conversation for this user instead.
        if not db.can_access_conversation(conversation_id, user_id):
            conversation_id = str(uuid.uuid4())[:12]
        if not request.use_history or config.history_turns <= 0:
            return conversation_id, []
        rows = db.get_messages(conversation_id, limit=config.history_turns * 2, user_id=user_id)
        return conversation_id, [{"role": r["role"], "content": r["content"]} for r in rows]

    def overrides(request) -> tuple[int, float]:
        max_tokens = request.max_tokens or config.max_tokens
        temperature = config.temperature if request.temperature is None else request.temperature
        return max_tokens, temperature

    def maybe_autotitle(conversation_id: str, first_message: str) -> None:
        """Title a conversation from its opening question, once.

        Uses the question itself rather than a model call: on an 8GB machine a
        second generation per conversation is real latency for a cosmetic gain.
        """
        try:
            if db.conversation_title(conversation_id):
                return
            text = " ".join((first_message or "").split())
            if not text:
                return
            title = text[:60].rstrip()
            if len(text) > 60:
                cut = title.rfind(" ")
                title = (title[:cut] if cut > 20 else title) + "..."
            db.set_conversation_title(conversation_id, title)
        except Exception as exc:
            log(f"auto-title skipped: {exc}", logging.DEBUG)

    def note_implicit_feedback(conversation_id: str, message: str,
                               user_id: str | None = None) -> str | None:
        """Turn a short "good job" / "no, wrong" into feedback on the prior answer.

        Praise stores the previous (prompt, answer) pair as an approved training
        example; a rejection stores it as a bad answer (rating -1, not approved),
        which is excluded from LoRA training but kept for future preference
        tuning. Returns a short label for a UI notice, or None.
        """
        sign = classify_implicit_feedback(message)
        if sign is None:
            return None
        rows = db.get_messages(conversation_id, limit=6, user_id=user_id)
        prompt = answer = None
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["role"] == "assistant" and rows[i]["content"].strip():
                answer = rows[i]["content"]
                for j in range(i - 1, -1, -1):
                    if rows[j]["role"] == "user":
                        prompt = rows[j]["content"]
                        break
                break
        if not prompt or not answer:
            return None
        try:
            db.record_feedback(prompt, answer, rating=sign,
                               approved=1 if sign > 0 else 0,
                               session_id="implicit", model_id=config.model,
                               source="implicit", user_id=user_id)
        except Exception as exc:
            log(f"implicit feedback not recorded: {exc}", logging.DEBUG)
            return None
        return "approved the previous answer for training" if sign > 0 else \
               "marked the previous answer as a bad example"

    @app.post("/api/chat")
    async def chat(request: ChatRequest, user: dict = USER):
        start_time = time.time()
        uid = user["id"]
        blocked = not_ready()
        if blocked is not None:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, "not ready")
            return blocked

        conversation_id, history = load_history(request, uid)
        max_tokens, temperature = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent
        # Mirror the streaming handler: route substantive messages through the
        # agent so the model router runs, even with the agent toggle off.
        if not use_agent and config.fast_path and quick_tool(request.message):
            use_agent = True
        elif not use_agent and config.knowledge_triage and is_substantive(request.message):
            use_agent = True
        elif not use_agent and config.project_dir and is_code_request(request.message):
            use_agent = True  # codebase edits need the tool loop

        # Recorded before generating, so a failed or empty run still leaves the
        # question in the transcript instead of silently dropping the turn.
        note_implicit_feedback(conversation_id, request.message, uid)
        db.add_message(conversation_id, "user", request.message, user_id=uid)
        tasks.note_chat_activity()
        tasks.chat_in_flight += 1
        log_event(get_logger("chat"), logging.INFO, "chat.start",
                  conversation_id=conversation_id, user_id=uid, agent=use_agent,
                  content=content_for_log(request.message))

        try:
            stats: GenerationStats | None = None
            if use_agent:
                answer = ""
                trace: list[dict] = []
                error: str | None = None
                async for event in agent.run_iterating(
                    request.message, history, conversation_id, max_tokens, temperature
                ):
                    if event["type"] == "final":
                        answer = event["answer"]
                        trace = event.get("trace", [])
                    elif event["type"] == "usage":
                        db.log_metric(
                            "agent_step", event["total_ms"], 200,
                            stats=GenerationStats(
                                prompt_tokens=event["prompt_tokens"],
                                completion_tokens=event["completion_tokens"],
                                ttft_ms=event["ttft_ms"], total_ms=event["total_ms"],
                            ),
                            model=config.model, step=event.get("step"),
                            conversation_id=conversation_id, user_id=uid,
                        )
                    elif event["type"] == "error":
                        error = event["error"]
                if error:
                    db.log_metric("chat", (time.time() - start_time) * 1000, 502, error, user_id=uid)
                    return JSONResponse(content={"error": error}, status_code=502)
            else:
                _sys = config.system_prompt_with_identity + (
                    REASONING_INSTRUCTION if config.reasoning_visible else "")
                system = {"role": "system", "content": _sys}
                user_msg = {"role": "user", "content": request.message}
                messages, _ = trim_to_context(
                    system, history, user_msg, config.context_size, max_tokens
                )
                answer, stats = await model_client.complete_with_stats(
                    messages, max_tokens, temperature, conversation_id=conversation_id
                )
                trace = []

            db.add_message(
                conversation_id, "assistant", answer,
                meta={"trace": trace} if trace else None, user_id=uid,
            )
            db.log_metric(
                "chat", (time.time() - start_time) * 1000, 200,
                stats=stats, model=config.model, conversation_id=conversation_id, user_id=uid,
            )
            log_event(get_logger("chat"), logging.INFO, "chat.completed",
                      conversation_id=conversation_id, user_id=uid, agent=use_agent,
                      tool_count=len(trace),
                      content=content_for_log(answer))
            return {
                "answer": answer,
                "conversation_id": conversation_id,
                "agent": use_agent,
                "trace": trace,
                "usage": stats.as_event() if stats else None,
            }
        except Exception as exc:
            db.log_metric("chat", (time.time() - start_time) * 1000, 503, str(exc), user_id=uid)
            return JSONResponse(content={"error": str(exc)}, status_code=503)
        finally:
            tasks.chat_in_flight = max(0, tasks.chat_in_flight - 1)
            tasks.note_chat_activity()

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest, user: dict = USER):
        """Streaming chat over Server-Sent Events.

        Emits the same event vocabulary whether or not the agent is on, so the
        browser has one code path: step, token, tool_call, tool_result, final.
        """
        blocked = not_ready()
        if blocked is not None:
            return blocked

        uid = user["id"]
        conversation_id, history = load_history(request, uid)
        max_tokens, temperature = overrides(request)
        use_agent = config.agent_enabled if request.agent is None else request.agent
        # Route substantive messages through the agent even when the agent
        # toggle is off, so the model router can decide answer vs search vs
        # weather. Deterministic shortcuts (bare URL, arithmetic) also need the
        # agent path to run. Greetings stay on the cheap plain path below.
        if not use_agent and config.fast_path and quick_tool(request.message):
            use_agent = True
        elif not use_agent and config.knowledge_triage and is_substantive(request.message):
            use_agent = True
        elif not use_agent and config.project_dir and is_code_request(request.message):
            use_agent = True  # codebase edits need the tool loop

        async def sse(event: dict) -> str:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        async def event_generator() -> AsyncGenerator[str, None]:
            answer = ""
            trace: list[dict] = []
            start_time = time.time()
            # Re-bind the acting user in this generator's context so RAG retrieval
            # and memory tools scope correctly during streaming.
            set_acting_user(uid)
            tasks.note_chat_activity()
            tasks.chat_in_flight += 1
            log_event(get_logger("chat"), logging.INFO, "chat.start",
                      conversation_id=conversation_id, user_id=uid, agent=use_agent,
                      streaming=True, content=content_for_log(request.message))
            try:
                yield await sse({"type": "start", "conversation_id": conversation_id, "agent": use_agent})
                # Turn a short "good job" / "no, wrong" into feedback on the
                # prior answer before recording this message as the new turn.
                fb_label = await asyncio.to_thread(note_implicit_feedback, conversation_id, request.message, uid)
                if fb_label:
                    yield await sse({"type": "notice", "info": True, "message": fb_label})
                await asyncio.to_thread(db.add_message, conversation_id, "user", request.message, user_id=uid)
                if use_agent:
                    async for event in agent.run_iterating(
                        request.message, history, conversation_id, max_tokens, temperature
                    ):
                        yield await sse(event)
                        if event["type"] == "final":
                            answer = event["answer"]
                            trace = event.get("trace", [])
                        elif event["type"] == "usage":
                            await asyncio.to_thread(
                                db.log_metric,
                                "agent_step", event["total_ms"], 200,
                                stats=GenerationStats(
                                    prompt_tokens=event["prompt_tokens"],
                                    completion_tokens=event["completion_tokens"],
                                    ttft_ms=event["ttft_ms"], total_ms=event["total_ms"],
                                ),
                                model=config.model, step=event.get("step"),
                                conversation_id=conversation_id, user_id=uid,
                            )
                else:
                    _sys = config.system_prompt_with_identity + (
                        REASONING_INSTRUCTION if config.reasoning_visible else "")
                    system = {"role": "system", "content": _sys}
                    user_msg = {"role": "user", "content": request.message}
                    messages, dropped = trim_to_context(
                        system, history, user_msg, config.context_size, max_tokens
                    )
                    if dropped:
                        yield await sse({"type": "context", "dropped": dropped,
                                         "tokens": messages_tokens(messages)})
                    plain_stats = GenerationStats()
                    # Accumulate tokens in a list and join once: repeated string
                    # concatenation in a hot loop is O(n^2) and drags on long replies.
                    answer_parts: list[str] = []
                    async for token in model_client.stream(
                        messages, max_tokens, temperature, plain_stats,
                        conversation_id=conversation_id
                    ):
                        answer_parts.append(token)
                        yield await sse({"type": "token", "token": token, "step": 1})
                    answer = "".join(answer_parts)
                    yield await sse({"type": "usage", "step": 1, **plain_stats.as_event()})
                    await asyncio.to_thread(
                        db.log_metric,
                        "chat_stream_gen", plain_stats.total_ms, 200, stats=plain_stats,
                        model=config.model, step=1, conversation_id=conversation_id, user_id=uid,
                    )
                    yield await sse({"type": "final", "answer": answer, "steps": 1, "trace": []})

                if answer:
                    await asyncio.to_thread(
                        db.add_message,
                        conversation_id, "assistant", answer,
                        meta={"trace": trace} if trace else None, user_id=uid,
                    )
                    # Give a brand-new conversation a short title derived from the
                    # opening question, so the history list is readable at a glance.
                    await asyncio.to_thread(maybe_autotitle, conversation_id, request.message)
                    log_event(get_logger("chat"), logging.INFO, "chat.completed",
                              conversation_id=conversation_id, user_id=uid,
                              agent=use_agent, streaming=True, tool_count=len(trace),
                              content=content_for_log(answer))
                await asyncio.to_thread(
                    db.log_metric, "chat_stream", (time.time() - start_time) * 1000, 200,
                    user_id=uid)
            except Exception as exc:
                db.log_metric("chat_stream", (time.time() - start_time) * 1000, 503, str(exc), user_id=uid)
                yield await sse({"type": "error", "error": str(exc)})
            finally:
                tasks.chat_in_flight = max(0, tasks.chat_in_flight - 1)
                tasks.note_chat_activity()
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @app.get("/api/config")
    def get_config(_a: dict = ADMIN):
        return {"config": config.public(), "tools": registry.specs()}

    @app.post("/api/config")
    def set_config(request: ConfigRequest, _a: dict = ADMIN):
        try:
            requested = request.model_dump(exclude_none=True)
            changed = config.apply(requested)
        except Exception as exc:
            return JSONResponse(
                {"error": f"could not apply settings: {type(exc).__name__}: {exc}"},
                status_code=400)
        if changed:
            log(f"Config updated from web UI: {', '.join(changed)}")
            # Re-apply structured logging when its settings change at runtime.
            # (Routing factors are read live by the router, so they need no
            # re-sync; changing the node topology needs a restart.)
            if any(k in changed for k in ("log_level", "log_format", "log_chat_content")):
                configure_logging(config, force=True)
        # Report any requested mutable field that was rejected (e.g. a bad value)
        # so the UI can tell the user rather than silently dropping it.
        ignored = [k for k in requested
                   if k in Config.MUTABLE and k not in changed
                   and str(requested[k]) != str(getattr(config, k, ""))]
        return {
            "changed": changed,
            "ignored": ignored,
            "config": config.public(),
            "note": "context_size applies immediately. max_kv_size needs a model server restart.",
        }

    @app.get("/api/tools")
    def list_tools(user: dict = USER):
        return {
            "tools": registry.specs(),
            "search_backend": config.search_backend,
            "agent_enabled": config.agent_enabled,
            "agent_max_steps": config.agent_max_steps,
        }

    @app.post("/api/tools/call")
    async def call_tool(request: ToolRequest, _a: dict = ADMIN):
        """Run a tool directly. Useful for testing one without the model."""
        result, error = await asyncio.to_thread(
            registry.call, request.name, request.args, request.conversation_id
        )
        return {"name": request.name, "result": result, "error": error}

    @app.get("/api/tools/calls")
    def tool_calls(
        limit: int = Query(50, ge=1, le=500),
        conversation_id: str | None = Query(None, max_length=64),
        _a: dict = ADMIN,
    ):
        return {"calls": db.list_tool_calls(limit, conversation_id)}

    @app.get("/api/conversations")
    def conversations(limit: int = Query(50, ge=1, le=200), user: dict = USER):
        return {"conversations": db.list_conversations(limit, user_id=user["id"])}

    def task_view(task: dict) -> dict:
        view = dict(task)
        view["live"] = tasks.live_status(task["id"])
        return view

    @app.get("/api/tasks")
    def list_tasks(_a: dict = ADMIN):
        return {"tasks": [task_view(task) for task in db.list_tasks()]}

    @app.post("/api/tasks")
    def create_task(request: TaskRequest, _a: dict = ADMIN):
        task = db.create_task(user_id=_a["id"], **request.model_dump())
        log(f"Task created: {task['name']} ({task['id']})")
        return {"task": task_view(task)}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, _a: dict = ADMIN):
        task = db.get_task(task_id)
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        return {"task": task_view(task), "runs": db.list_runs(task_id, 20)}

    @app.post("/api/tasks/{task_id}")
    def update_task(task_id: str, request: TaskUpdateRequest, _a: dict = ADMIN):
        task = db.update_task(task_id, request.model_dump(exclude_none=True))
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        return {"task": task_view(task)}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str, _a: dict = ADMIN):
        tasks.cancel_task(task_id)
        return {"deleted": db.delete_task(task_id)}

    @app.post("/api/tasks/{task_id}/run")
    async def run_task(task_id: str, _a: dict = ADMIN):
        task = db.get_task(task_id)
        if task is None:
            return JSONResponse(content={"error": "no such task"}, status_code=404)
        blocked = not_ready()
        if blocked is not None:
            return blocked
        try:
            run = await tasks.launch(task, trigger="manual")
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=409)
        return {"run_id": run.run_id, "task_id": task_id}

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, _a: dict = ADMIN):
        return {"cancelling": tasks.cancel_task(task_id)}

    @app.get("/api/tasks/{task_id}/runs")
    def task_runs(task_id: str, limit: int = Query(20, ge=1, le=200), _a: dict = ADMIN):
        return {"runs": db.list_runs(task_id, limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, _a: dict = ADMIN):
        run = db.get_run(run_id)
        if run is None:
            return JSONResponse(content={"error": "no such run"}, status_code=404)
        return {"run": run, "events": db.run_events(run_id, limit=2000)}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, _a: dict = ADMIN):
        return {"cancelling": tasks.cancel_run(run_id)}

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str, _a: dict = ADMIN):
        """Replay a run from the start, then follow it live until it finishes."""
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in tasks.subscribe(run_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @app.get("/api/models")
    def list_models(_a: dict = ADMIN):
        return {
            "current": model_manager.describe(),
            "catalog": model_catalog(config),
            "adapters": [{"id": "none", "path": "", "modified": None}] + list_adapters(),
            "cache_dir": str(hf_cache_dir()),
        }

    @app.post("/api/models/select")
    def select_model(request: ModelSelectRequest, _a: dict = ADMIN):
        """Point the model server at a different model or adapter and restart it."""
        if retrain_manager.status.get("running"):
            return JSONResponse(
                content={"error": "Retraining is running and owns the model server."},
                status_code=409,
            )
        running = list(tasks.by_task)
        if running:
            return JSONResponse(
                content={"error": f"{len(running)} task run(s) in flight. Cancel them first.",
                         "running": running},
                status_code=409,
            )
        try:
            changed = model_manager.swap(request.model, request.adapter)
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

        if request.model:
            config.model = request.model
        if request.adapter is not None:
            config.adapter = request.adapter
        if request.max_kv_size is not None and request.max_kv_size != config.max_kv_size:
            config.max_kv_size = request.max_kv_size
            model_manager.max_kv_size = request.max_kv_size
            changed = True

        if changed or request.restart:
            log(f"Switching to {config.model} (adapter: {config.adapter}). Restarting model server.")
            guarded_thread(model_manager.restart).start()
        return {
            "restarting": bool(changed or request.restart),
            "changed": changed,
            "current": model_manager.describe(),
            "note": "Weights download on first use. Watch /api/logs/model for progress.",
        }

    @app.get("/api/adapters")
    def adapters(_a: dict = ADMIN):
        return {"adapters": list_adapters(), "current": model_manager.adapter_choice}

    @app.get("/api/logs/{name}")
    def logs(name: str, lines: int = Query(120, ge=1, le=2000), _a: dict = ADMIN):
        try:
            return {"name": name, "text": tail_log(name, lines)}
        except ValueError as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=404)

    @app.get("/api/memory")
    def list_memory(
        limit: int = Query(50, ge=1, le=500),
        search: str | None = Query(None, max_length=100),
        user: dict = USER,
    ):
        return {"memories": db.recall(search, limit, user_id=user["id"])}

    @app.post("/api/memory")
    def set_memory(request: MemoryRequest, user: dict = USER):
        db.remember(request.key, request.value, user_id=user["id"])
        return {"stored": request.key}

    @app.delete("/api/memory/{key}")
    def delete_memory(key: str, user: dict = USER):
        return {"deleted": db.forget(key, user_id=user["id"])}

    @app.get("/api/conversation/{conversation_id}")
    def conversation(conversation_id: str, limit: int = Query(200, ge=1, le=1000),
                     user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        messages = db.get_messages(conversation_id, limit, user_id=user["id"])
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "est_tokens": sum(m.get("est_tokens") or 0 for m in messages),
            "context_size": config.context_size,
        }

    @app.delete("/api/conversation/{conversation_id}")
    def clear_conversation(conversation_id: str, user: dict = USER):
        if not db.can_access_conversation(conversation_id, user["id"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"deleted": db.clear_conversation(conversation_id, user_id=user["id"])}

    @app.post("/api/model/restart")
    def restart_model(_a: dict = ADMIN):
        """Restart the model server, picking up a changed KV cache size."""
        if retrain_manager.status.get("running"):
            return JSONResponse(
                content={"error": "Retraining is running and already owns the model server."},
                status_code=409,
            )
        model_manager.max_kv_size = config.max_kv_size
        guarded_thread(model_manager.restart).start()
        return {"status": "restarting", "max_kv_size": config.max_kv_size}

    @app.post("/api/feedback")
    async def feedback(request: FeedbackRequest, user: dict = USER):
        approved = 1 if (request.corrected_response or request.rating > 0) else 0
        session_id = str(uuid.uuid4())[:8]

        db.execute(
            """INSERT INTO feedback
               (user_prompt, assistant_response, rating, corrected_response,
                approved_for_training, session_id, model_id, source, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'button', ?)""",
            (
                request.user_prompt,
                request.assistant_response,
                request.rating,
                request.corrected_response,
                approved,
                session_id,
                config.model,
                user["id"],
            ),
        )
        db.commit()

        # Auto-retrain check
        if config.auto_retrain_threshold > 0:
            untrained = db.get_untrained_count()
            if untrained >= config.auto_retrain_threshold and not retrain_manager.status.get("running"):
                log(f"Auto-retrain triggered: {untrained} approved feedback items")
                guarded_thread(retrain_manager.run, trigger="auto").start()

        return {
            "status": "feedback saved",
            "approved_for_training": bool(approved),
            "session_id": session_id,
        }

    @app.post("/api/retrain")
    def retrain(_a: dict = ADMIN):
        if retrain_manager.status.get("running"):
            return {"status": "already running", "detail": retrain_manager.status}

        guarded_thread(retrain_manager.run, trigger="web").start()
        return {"status": "started", "detail": retrain_manager.status}

    @app.get("/api/metrics/summary")
    def metrics_summary(endpoint: str | None = Query(None, max_length=40), _a: dict = ADMIN):
        return {
            "overall": db.metric_summary(endpoint),
            "chat": db.metric_summary("chat"),
            "agent_step": db.metric_summary("agent_step"),
        }

    @app.get("/api/metrics/run/{conversation_id}")
    def metrics_for_run(conversation_id: str, _a: dict = ADMIN):
        """Per-step prompt tokens for one conversation.

        A rising curve here is the agent re-sending its whole prompt each step.
        Flat means a prefix cache is doing its job.
        """
        steps = db.run_step_metrics(conversation_id)
        return {
            "conversation_id": conversation_id,
            "steps": steps,
            "total_prompt_tokens": sum(s["prompt_tokens"] or 0 for s in steps),
        }

    @app.get("/api/metrics")
    def metrics(limit: int = Query(100, ge=1, le=1000), _a: dict = ADMIN):
        rows = db.execute(
            "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return {"metrics": [dict(r) for r in rows]}

    return app



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    '_define_api_models',
    'create_app',
]
