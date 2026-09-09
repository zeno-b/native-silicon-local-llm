"""HTTP integration tests: auth, RBAC, cross-user isolation, import, routing.

Complements the offline ``python deploy.py --selftest`` suite (which covers the
unit-level invariants). These exercise the real FastAPI app end to end with a
test client and a throwaway database — no model server required.

Run with pytest::

    pip install pytest
    python -m pytest tests/ -q

or standalone (no pytest needed)::

    python tests/test_app.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402


def build_app(**overrides):
    """A fresh app + temp DB. Auth on, with a known admin, by default."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token")
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    return app, cfg, db


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_auth_config_is_public():
    app, _, _ = build_app()
    with TestClient(app) as c:
        r = c.get("/api/auth/config")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is True


def test_unauthenticated_is_blocked():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 401
        assert c.get("/api/conversations").status_code == 401
        assert c.get("/api/config").status_code == 401


def test_invalid_credentials_rejected():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "wrong").status_code == 401
        assert login(c, "ghost", "whatever").status_code == 401


def test_login_logout_flow():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        assert c.get("/api/health").status_code == 200
        assert c.post("/api/auth/logout").status_code == 200
        assert c.get("/api/health").status_code == 401


def test_test_user_created_when_enabled():
    app, _, db = build_app(allow_test_user=True, test_username="qa", test_password="qapass12")
    with TestClient(app):
        u = db.get_user_by_username("qa")
        assert u and u["role"] == "user"


def test_login_throttle_blocks_brute_force():
    """Repeated bad passwords earn a 429 + Retry-After, and a success clears it.

    The limit is patched down so the test does not pay for LOGIN_MAX_FAILURES
    real scrypt verifications.
    """
    import local_llm.api as api_mod
    original = api_mod.LOGIN_MAX_FAILURES
    api_mod.LOGIN_MAX_FAILURES = 3
    try:
        app, _, _ = build_app()
        with TestClient(app) as c:
            for _ in range(3):
                assert login(c, "admin", "wrong").status_code == 401
            r = login(c, "admin", "wrong")
            assert r.status_code == 429, r.status_code
            assert int(r.headers["Retry-After"]) > 0
            # The correct password is refused too: the throttle runs first.
            assert login(c, "admin", "adminpw123").status_code == 429

        # A fresh app (fresh counters) still lets the right password through,
        # and one success wipes the failures recorded before it.
        app, _, _ = build_app()
        with TestClient(app) as c:
            assert login(c, "admin", "wrong").status_code == 401
            assert login(c, "admin", "adminpw123").status_code == 200
            assert c.post("/api/auth/logout").status_code == 200
            for _ in range(3):
                assert login(c, "admin", "wrong").status_code == 401
            assert login(c, "admin", "wrong").status_code == 429
    finally:
        api_mod.LOGIN_MAX_FAILURES = original


def test_login_throttle_also_stops_a_username_spray():
    """Rotating usernames from one client hits the per-IP ceiling (3x the
    per-username limit), which the per-username key alone would never trip."""
    import local_llm.api as api_mod
    original = api_mod.LOGIN_MAX_FAILURES
    api_mod.LOGIN_MAX_FAILURES = 1          # per-IP ceiling becomes 3
    try:
        app, _, _ = build_app()
        with TestClient(app) as c:
            for i in range(3):
                assert login(c, f"ghost{i}", "pw").status_code == 401
            assert login(c, "ghost9", "pw").status_code == 429
    finally:
        api_mod.LOGIN_MAX_FAILURES = original


def test_removing_an_import_spares_other_users_prompts():
    """Import removal is owner-scoped. Prompt names are unique only per user, so
    an unscoped delete would take an identically named prompt from everyone."""
    app, _, db = build_app()
    with TestClient(app):
        alice = db.create_user("alice", role="user")["id"]
        bob = db.create_user("bob", role="user")["id"]
        db.save_prompt("[imported] Shared", "alice body", user_id=alice)
        db.save_prompt("[imported] Shared", "bob body", user_id=bob)

        db.create_import("imp1", alice, "export.zip", 10)
        db.update_import("imp1", artifacts={"conversations": [], "docs": [],
                                            "prompts": ["[imported] Shared"]})
        import_id = "imp1"
        assert app.state.importer.remove_import(import_id, user_id=alice)

        assert db.list_prompts(user_id=alice) == []
        assert [p["body"] for p in db.list_prompts(user_id=bob)] == ["bob body"]


def test_generation_gate_limits_and_sheds_load():
    """N generations run, a bounded queue waits, everything past it is refused.

    Exercised directly: driving it through /api/chat would need a live model.
    """
    import asyncio as _asyncio
    from local_llm.core import GenerationBusy, GenerationGate

    async def scenario():
        # 2 run at a time; up to 4 more may queue behind them.
        gate = GenerationGate(limit=2, queue_multiplier=2)
        running = _asyncio.Event()
        holders = []

        async def hold():
            async with gate.slot():
                holders.append(1)
                await running.wait()

        started = [_asyncio.create_task(hold()) for _ in range(6)]
        for _ in range(4):
            await _asyncio.sleep(0)      # let everyone reach the semaphore
        assert len(holders) == 2, holders    # only the limit runs concurrently
        assert gate.waiting == 4             # the rest are queued, not running

        # The waiting room is full, so the seventh is shed at once instead of
        # queueing behind a wait that could last minutes.
        try:
            await gate.acquire()
            raise AssertionError("a full queue should refuse admission")
        except GenerationBusy:
            pass

        running.set()
        await _asyncio.gather(*started)
        assert len(holders) == 6          # the queued four ran once slots freed
        assert gate.waiting == 0

        # Slots are returned, so the gate is reusable afterwards.
        async with gate.slot():
            pass

    _asyncio.run(_asyncio.wait_for(scenario(), timeout=5))


def test_generation_gate_releases_on_failure():
    """An exception inside a slot must still give the slot back, or the server
    silently loses generation capacity every time a chat errors."""
    import asyncio as _asyncio
    from local_llm.core import GenerationGate

    async def scenario():
        gate = GenerationGate(limit=1)
        for _ in range(3):
            try:
                async with gate.slot():
                    raise RuntimeError("model server died")
            except RuntimeError:
                pass
        async with gate.slot():          # would hang forever if a slot leaked
            pass

    _asyncio.run(_asyncio.wait_for(scenario(), timeout=5))


def test_non_admin_feedback_cannot_reach_the_training_set():
    """A shared LoRA adapter means a non-admin thumbs-up must not enter the
    training corpus by itself: it queues for review. Admin ratings still do."""
    app, _, db = build_app()
    payload = {"user_prompt": "hi", "assistant_response": "hello", "rating": 1}
    with TestClient(app) as admin:
        assert login(admin, "admin", "adminpw123").status_code == 200
        admin.post("/api/users", json={"username": "mallory",
                                       "password": "mallorypw1", "role": "user"})
        with TestClient(app) as user:
            assert login(user, "mallory", "mallorypw1").status_code == 200
            out = user.post("/api/feedback", json=payload).json()
            assert out["approved_for_training"] is False
            assert out["pending_approval"] is True
            # A correction is no different: still queued, never auto-approved.
            out = user.post("/api/feedback", json={**payload, "rating": 0,
                                                   "corrected_response": "poison"}).json()
            assert out["approved_for_training"] is False
            # And the user cannot approve their own rows.
            assert user.post("/api/feedback/approve-pending").status_code == 403
            assert user.post("/api/feedback/1/approved").status_code == 403
        assert db.get_untrained_count() == 0          # nothing reached the corpus
        assert db.dataset_stats()["pending"] == 2

        # The admin's own rating is approved immediately, as before.
        out = admin.post("/api/feedback", json=payload).json()
        assert out["approved_for_training"] is True and out["pending_approval"] is False
        assert db.get_untrained_count() == 1

        # Approving the queue releases exactly the queued rows.
        assert admin.post("/api/feedback/approve-pending").json()["approved"] == 2
        assert db.get_untrained_count() == 3
        assert db.dataset_stats()["pending"] == 0


def test_cors_refuses_a_wildcard_origin():
    """ALLOWED_ORIGINS=* must not reach CORSMiddleware: Starlette echoes the
    caller's Origin when a wildcard is combined with credentials, which would
    hand every site the user visits an authenticated session."""
    import os as _os
    from fastapi.middleware.cors import CORSMiddleware
    previous = _os.environ.get("ALLOWED_ORIGINS")
    _os.environ["ALLOWED_ORIGINS"] = "*,https://llm.example.com"
    try:
        app, cfg, _ = build_app()
        origins = []
        for mw in app.user_middleware:
            if mw.cls is CORSMiddleware:
                origins = list(mw.kwargs.get("allow_origins", []))
        assert origins, "CORS middleware not installed"
        assert "*" not in origins
        assert "https://llm.example.com" in origins       # real origins survive
        assert f"http://127.0.0.1:{cfg.web_port}" in origins
        with TestClient(app) as c:
            r = c.get("/api/auth/config", headers={"Origin": "https://evil.example"})
            assert r.headers.get("access-control-allow-origin") is None
    finally:
        if previous is None:
            _os.environ.pop("ALLOWED_ORIGINS", None)
        else:
            _os.environ["ALLOWED_ORIGINS"] = previous


def test_prompts_are_per_user():
    """Saved prompts are owned, not global: one user's library is invisible to
    another, and identical names in two accounts do not collide."""
    app, _, db = build_app()
    with TestClient(app) as admin:
        assert login(admin, "admin", "adminpw123").status_code == 200
        admin.post("/api/users", json={"username": "bob", "password": "bobpass123",
                                       "role": "user"})
        assert admin.post("/api/prompts",
                          params={"name": "shared", "body": "admin body"}).status_code == 200
        def bodies(client):
            return {p["name"]: p["body"] for p in client.get("/api/prompts").json()["prompts"]}

        with TestClient(app) as bob:
            assert login(bob, "bob", "bobpass123").status_code == 200
            assert bodies(bob) == {}                     # admin's library is invisible
            assert bob.post("/api/prompts",
                            params={"name": "shared", "body": "bob body"}).status_code == 200
            assert bodies(bob) == {"shared": "bob body"}   # same name, no collision
            # Deleting bob's copy must not touch the admin's.
            assert bob.delete("/api/prompts/shared").status_code == 200
            assert bodies(bob) == {}
        assert bodies(admin) == {"shared": "admin body"}


# --------------------------------------------------------------------------- #
# Authorization (RBAC) — admin endpoints blocked for non-admins and anon.
# --------------------------------------------------------------------------- #
ADMIN_GET = ["/api/config", "/api/users", "/api/models", "/api/metrics",
             "/api/logs/model", "/api/tasks", "/api/routing/events",
             "/api/cluster/nodes", "/api/imports", "/api/dataset/stats",
             "/api/backup", "/api/project/status", "/api/browse"]


def test_admin_endpoints_blocked_for_non_admin():
    app, _, _ = build_app()
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        admin.post("/api/users", json={"username": "bob", "password": "bobpass12", "role": "user"})
    with TestClient(app) as bob:
        login(bob, "bob", "bobpass12")
        for path in ADMIN_GET:
            assert bob.get(path).status_code == 403, f"{path} not admin-gated for a user"
        # Mutating admin routes too.
        assert bob.post("/api/config", json={"max_tokens": 64}).status_code == 403
        assert bob.post("/api/retrain").status_code == 403
        assert bob.post("/api/import", content=b"x", headers={"x-filename": "e.zip"}).status_code == 403


def test_admin_can_reach_admin_endpoints():
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        for path in ["/api/config", "/api/users", "/api/models", "/api/cluster/nodes",
                     "/api/routing/events", "/api/imports"]:
            assert c.get(path).status_code == 200, f"admin blocked from {path}"


def test_agents_crud_and_capability_gating():
    app, _, _ = build_app()
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        caps = admin.get("/api/agents/capabilities").json()["capabilities"]
        keys = [c["key"] for c in caps]
        assert "file_ops" in keys and "office365" in keys
        # Create an agent with two capabilities; unknown keys are dropped.
        r = admin.post("/api/agents", json={"name": "Coder", "description": "writes code",
                                            "capabilities": ["file_ops", "code_exec", "bogus"]})
        assert r.status_code == 200, r.text
        agent = r.json()["agent"]
        assert agent["capabilities"] == ["file_ops", "code_exec"]  # bogus filtered, order canonical
        aid = agent["id"]
        assert any(a["id"] == aid for a in admin.get("/api/agents").json()["agents"])
        # Update capabilities + disable.
        r = admin.post(f"/api/agents/{aid}", json={"capabilities": ["web_api"], "enabled": False})
        assert r.json()["agent"]["capabilities"] == ["web_api"]
        assert r.json()["agent"]["enabled"] is False
        assert admin.delete(f"/api/agents/{aid}").json()["deleted"] is True


def test_project_dir_can_be_set_from_the_ui():
    """The Codebase panel's Set button must actually change project_dir.

    It POSTs {"project_dir": ...} to /api/config; ConfigRequest used to omit the
    field, so Pydantic dropped it, the server answered 200 with changed: [] and
    the UI reported success while nothing changed.
    """
    import tempfile, os
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        target = tempfile.mkdtemp()
        r = c.post("/api/config", json={"project_dir": target})
        assert r.status_code == 200, r.text
        assert "project_dir" in r.json()["changed"], r.json()
        status = c.get("/api/project/status").json()
        assert status["project_dir"] == os.path.realpath(target), status
        assert status["root"] == os.path.realpath(target)
        # A path that is not a directory is refused, not silently stored.
        assert c.post("/api/config", json={"project_dir": "/no/such/dir"}).status_code == 400
        # And it can be cleared back to the sandbox workspace.
        assert "project_dir" in c.post("/api/config", json={"project_dir": ""}).json()["changed"]


def test_project_diff_withholds_the_enclosing_repo():
    """With no project set the diff panel must not show the app's own source."""
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        d = c.get("/api/project/diff").json()
        assert d.get("project_dir") == "", d          # UI can gate on this
        assert d.get("diff") == "", "leaked the enclosing repo's diff"


def test_conversation_search_returns_title_and_pinned():
    """Search rows go through the same renderer as the list, so they need the
    same fields; without them every hit rendered as a raw conversation id."""
    app, _, db = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        uid = [u for u in db.list_users() if u["username"] == "admin"][0]["id"]
        db.add_message("conv-search", "user", "a distinctive needle", user_id=uid)
        db.set_conversation_title("conv-search", "Named Chat")
        db.set_conversation_pinned("conv-search", True)
        db.add_message("conv-plain", "user", "another needle", user_id=uid)
        rows = c.get("/api/conversations/search?q=needle").json()["results"]
        by_id = {r["conversation_id"]: r for r in rows}
        assert by_id["conv-search"]["title"] == "Named Chat"
        assert by_id["conv-search"]["pinned"] is True
        # A conversation with no meta row must still render (empty title, unpinned).
        assert by_id["conv-plain"]["title"] == ""
        assert by_id["conv-plain"]["pinned"] is False


def test_agents_reject_malformed_capabilities():
    """A bad capabilities payload is a 400, never a 500 or a silent empty set."""
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        for bad in (5, "file_ops", {"file_ops": True}):
            r = c.post("/api/agents", json={"name": "x", "capabilities": bad})
            assert r.status_code == 400, f"capabilities={bad!r} gave {r.status_code}"
        ok = c.post("/api/agents", json={"name": "ok", "capabilities": ["file_ops"]})
        aid = ok.json()["agent"]["id"]
        assert c.post(f"/api/agents/{aid}", json={"capabilities": 5}).status_code == 400


def test_agents_run_route_is_not_shadowed():
    """/api/agents/run must be matched by the run handler.

    It is declared alongside /api/agents/{agent_id}; if that catch-all is
    registered first it swallows "run" and the endpoint silently 404s, which is
    exactly the bug this guards. Its own validation errors prove it is reached.
    """
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        aid = c.post("/api/agents", json={"name": "R", "capabilities": []}).json()["agent"]["id"]
        r = c.post("/api/agents/run", json={"agent_ids": [aid]})
        assert r.status_code == 400 and "prompt" in r.json()["error"], r.text
        r = c.post("/api/agents/run", json={"prompt": "hi"})
        assert r.status_code == 400 and "agent" in r.json()["error"], r.text
        r = c.post("/api/agents/run", json={"prompt": "hi", "agent_ids": ["nope"]})
        assert r.status_code == 400 and "matched" in r.json()["error"], r.text


def test_agents_admin_gated_but_listable():
    app, _, _ = build_app()
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        admin.post("/api/users", json={"username": "carol", "password": "carolpw12", "role": "user"})
        admin.post("/api/agents", json={"name": "Shared", "capabilities": ["web_api"]})
    with TestClient(app) as carol:
        login(carol, "carol", "carolpw12")
        # A non-admin may list/select agents and read capabilities...
        assert carol.get("/api/agents").status_code == 200
        assert carol.get("/api/agents/capabilities").status_code == 200
        # ...but cannot create, edit or delete them.
        assert carol.post("/api/agents", json={"name": "X"}).status_code == 403
        some = carol.get("/api/agents").json()["agents"]
        if some:
            assert carol.delete(f"/api/agents/{some[0]['id']}").status_code == 403


# --------------------------------------------------------------------------- #
# Cross-user data isolation
# --------------------------------------------------------------------------- #
def test_cross_user_isolation():
    app, _, db = build_app()
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        admin.post("/api/users", json={"username": "alice", "password": "alicepw12", "role": "user"})
        admin.post("/api/users", json={"username": "bob", "password": "bobpass12", "role": "user"})

    with TestClient(app) as alice:
        login(alice, "alice", "alicepw12")
        alice.post("/api/memory", json={"key": "diary", "value": "alice-secret"})
        # Seed a conversation directly for alice via the DB, then confirm access.
        alice_id = alice.get("/api/auth/me").json()["user"]["id"]
        db.add_message("conv-alice", "user", "alice private", user_id=alice_id)

    with TestClient(app) as bob:
        login(bob, "bob", "bobpass12")
        # Bob's memory list must not contain alice's note.
        keys = [m["key"] for m in bob.get("/api/memory").json()["memories"]]
        assert "diary" not in keys
        # Bob cannot read alice's conversation by id.
        assert bob.get("/api/conversation/conv-alice").status_code == 404
        assert bob.get("/api/conversation/conv-alice/export").status_code == 404
        # Bob's own conversation list is empty.
        assert bob.get("/api/conversations").json()["conversations"] == []


# --------------------------------------------------------------------------- #
# Node worker endpoints
# --------------------------------------------------------------------------- #
def test_node_endpoints_require_token():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert c.get("/api/node/health").status_code == 401
        assert c.post("/api/node/generate", json={"messages": []}).status_code == 401
        ok = c.get("/api/node/health", headers={"Authorization": "Bearer secret-node-token"})
        assert ok.status_code == 200
        assert c.get("/api/node/health",
                     headers={"Authorization": "Bearer wrong"}).status_code == 401


# --------------------------------------------------------------------------- #
# Search engine constraint
# --------------------------------------------------------------------------- #
def test_search_backend_locked():
    from local_llm.websearch import SearchBackend
    assert SearchBackend.ENDPOINTS == ("https://lite.duckduckgo.com/lite/",)
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        # POST /api/config has no search_backend field; sending it is ignored.
        r = c.post("/api/config", json={"search_backend": "google"})
        assert r.status_code == 200
        cfg = c.get("/api/config").json()["config"]
        assert cfg["search_backend"] == "duckduckgo_lite"


# --------------------------------------------------------------------------- #
# Claude history import
# --------------------------------------------------------------------------- #
def _export_zip():
    conv = [{"uuid": "abc", "name": "T", "chat_messages": [
        {"sender": "human", "text": "hi"}, {"sender": "assistant", "text": "hello"}]}]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("conversations.json", json.dumps(conv))
    return buf.getvalue()


def _wait_import(client, import_id, tries=50):
    import time
    for _ in range(tries):
        rec = client.get(f"/api/imports/{import_id}").json()["import"]
        if rec["status"] in ("completed", "failed"):
            return rec
        time.sleep(0.1)
    return rec


def test_import_valid_export():
    app, _, _ = build_app()
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        r = c.post("/api/import", content=_export_zip(), headers={"x-filename": "export.zip"})
        assert r.status_code == 200
        rec = _wait_import(c, r.json()["import_id"])
        assert rec["status"] == "completed", rec
        counts = json.loads(rec["counts"]) if isinstance(rec["counts"], str) else rec["counts"]
        assert counts["conversations"] == 1 and counts["messages"] == 2
        # The imported conversation (owner-namespaced id) is browsable by the importer.
        convs = [x["conversation_id"] for x in c.get("/api/conversations").json()["conversations"]]
        assert any(cid.startswith("claude-") and cid.endswith("-abc") for cid in convs), convs


def test_import_rejects_zip_slip():
    app, _, _ = build_app()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "pwned")
    with TestClient(app) as c:
        login(c, "admin", "adminpw123")
        r = c.post("/api/import", content=buf.getvalue(), headers={"x-filename": "evil.zip"})
        assert r.status_code == 200  # accepted for processing...
        rec = _wait_import(c, r.json()["import_id"])
        assert rec["status"] == "failed"  # ...then rejected during extraction
        assert "traversal" in (rec["error"] or "").lower()


# --------------------------------------------------------------------------- #
# Concurrency: many users at once stay isolated, no state leaks.
# --------------------------------------------------------------------------- #
def test_concurrent_users_isolated():
    app, _, _ = build_app()
    names = [f"user{i}" for i in range(5)]
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        for n in names:
            admin.post("/api/users", json={"username": n, "password": n + "pass12", "role": "user"})

    results = {}
    errors = []

    def worker(name):
        try:
            with TestClient(app) as c:
                login(c, name, name + "pass12")
                c.post("/api/memory", json={"key": "mine", "value": name + "-value"})
                mems = c.get("/api/memory").json()["memories"]
                results[name] = [m["value"] for m in mems]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for name in names:
        assert results[name] == [name + "-value"], f"{name} saw {results[name]}"


# --------------------------------------------------------------------------- #
# Backward compatibility: auth disabled behaves as a single local admin.
# --------------------------------------------------------------------------- #
def test_auth_disabled_is_open_single_user():
    app, _, _ = build_app(auth_enabled=False)
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/config").status_code == 200  # admin route, no login
        me = c.get("/api/auth/me").json()
        assert me["user"]["role"] == "admin" and me["user"]["id"] == "local"


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest required).
# --------------------------------------------------------------------------- #
def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
