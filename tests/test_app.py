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
        # The imported conversation is browsable by the admin importer.
        convs = [x["conversation_id"] for x in c.get("/api/conversations").json()["conversations"]]
        assert "claude-abc" in convs


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
