"""Authentication, sessions, role-based access control, and Entra ID / OIDC.

Design
------
* **Optional by default.** With ``AUTH_ENABLED=0`` (the default) the whole layer
  is inert: every request resolves to a synthetic ``local`` administrator, so a
  single-user install behaves exactly as it did before multi-user existed. Turn
  it on with ``AUTH_ENABLED=1``.
* **Password hashing.** ``hashlib.scrypt`` (standard library, memory-hard). No
  third-party dependency. Stored as ``scrypt$n$r$p$salt$hash``; verified with a
  constant-time compare.
* **Sessions are cookies.** An opaque random token is set as an HttpOnly cookie;
  only its SHA-256 is stored, so a database leak yields no usable tokens. Cookies
  ride along with fetch, EventSource and file downloads alike, which header-only
  bearer auth cannot. A ``Bearer`` token is also accepted for API clients.
* **RBAC is server-side.** ``require_user`` / ``require_admin`` are FastAPI
  dependencies enforced on every protected route; the UI only *hides* admin
  surfaces, it never guards them.
* **Entra ID / OIDC.** Standard authorization-code flow. The ID token is fetched
  directly from the token endpoint over TLS (confidential client), so claims are
  trusted per the OIDC spec; if ``cryptography`` is present the signature is also
  verified against the tenant JWKS as defence in depth.

First-run admin credentials are NEVER hardcoded: they come from
``AUTH_ADMIN_USERNAME`` / ``AUTH_ADMIN_PASSWORD`` or, if unset, a random password
is generated and printed to the log exactly once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
import urllib.parse
from datetime import timedelta
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403

# FastAPI is a hard dependency of the server, but keep the module importable
# (for --selftest on a bare machine) if it is not installed yet.
try:
    from fastapi import Request, HTTPException, Response  # noqa: F401
except Exception:  # pragma: no cover
    Request = Any  # type: ignore
    Response = Any  # type: ignore

    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int = 400, detail: str = ""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)


_auth_log = get_logger("auth")


# --------------------------------------------------------------------------- #
# Password hashing (scrypt)                                                    #
# --------------------------------------------------------------------------- #
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1


def hash_password(password: str) -> str:
    """Hash a password with scrypt. Returns ``scrypt$n$r$p$salt$hash`` (hex)."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time verify a password against a stored scrypt hash."""
    if not stored or not password:
        return False
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(hash_hex)))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


def _b64url_json(segment: str) -> dict:
    """Decode a base64url JWT segment into a dict (no signature check)."""
    padding = "=" * (-len(segment) % 4)
    raw = base64.urlsafe_b64decode(segment + padding)
    return json.loads(raw.decode("utf-8"))


# --------------------------------------------------------------------------- #
# The Auth manager                                                            #
# --------------------------------------------------------------------------- #
class Auth:
    """Sessions, RBAC dependencies, bootstrap and OIDC for one Database."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        # Short-lived OIDC state -> nonce, to defend the callback against CSRF
        # and token replay. In-process (single web node); entries expire.
        self._oidc_state: dict[str, tuple[float, str, str]] = {}
        self._oidc_lock = threading.Lock()
        self._oidc_discovery: dict | None = None
        self._oidc_discovery_at = 0.0

    # ---- basic properties ------------------------------------------------- #
    @property
    def enabled(self) -> bool:
        return bool(self.config.auth_enabled)

    def synthetic_local_user(self) -> dict:
        """The implicit administrator used when auth is disabled."""
        return {"id": SENTINEL_LOCAL_USER, "username": SENTINEL_LOCAL_USER,
                "role": "admin", "source": "local", "disabled": 0,
                "display_name": "Local", "email": None, "synthetic": True}

    # ---- bootstrap -------------------------------------------------------- #
    def bootstrap(self) -> None:
        """Ensure an admin exists, and (optionally) the test user.

        Never resets an existing admin's password. If no admin exists and no
        password was configured, a strong one is generated and logged ONCE.
        """
        if not self.enabled:
            log("Auth disabled (AUTH_ENABLED=0): a single local administrator "
                "owns everything. Set AUTH_ENABLED=1 to require login.")
            return
        self.db.purge_expired_sessions()
        if self.db.count_users(role="admin") == 0:
            username = (self.config.admin_username or "admin").strip() or "admin"
            password = self.config.admin_password
            generated = False
            if not password:
                password = secrets.token_urlsafe(12)
                generated = True
            # Username might already exist as a non-admin; upgrade or create.
            existing = self.db.get_user_by_username(username)
            if existing:
                self.db.update_user(existing["id"], role="admin",
                                    password_hash=hash_password(password))
                admin_id = existing["id"]
            else:
                admin_id = self.db.create_user(
                    username, password_hash=hash_password(password),
                    role="admin", source="local",
                    display_name="Administrator")["id"]
            if generated:
                log("=" * 62, )
                log(f"  FIRST-RUN ADMIN CREATED: username '{username}'")
                log(f"  GENERATED PASSWORD: {password}")
                log("  Save it now and change it after logging in. It will NOT be "
                    "shown again. Set AUTH_ADMIN_PASSWORD to control it.")
                log("=" * 62)
            else:
                log(f"First-run admin '{username}' created from AUTH_ADMIN_PASSWORD.")
            log_event(_auth_log, 20, "auth.admin_bootstrapped",
                      username=username, generated=generated, user_id=admin_id)

        # Optional test user (dev/testing convenience, not a backdoor).
        if self.config.allow_test_user:
            username = (self.config.test_username or "test").strip() or "test"
            password = self.config.test_password
            if not password:
                log("AUTH_ALLOW_TEST_USER=1 but AUTH_TEST_PASSWORD is empty; "
                    "not creating a passwordless test account.", 30)
            else:
                existing = self.db.get_user_by_username(username)
                if existing:
                    self.db.update_user(existing["id"],
                                        password_hash=hash_password(password))
                else:
                    self.db.create_user(username, password_hash=hash_password(password),
                                        role="user", source="test",
                                        display_name="Test User")
                    log(f"Test user '{username}' created (role: user). Remove it for "
                        "production: set AUTH_ALLOW_TEST_USER=0 or delete via the "
                        "admin users panel.")

    # ---- sessions --------------------------------------------------------- #
    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, user: dict, request: Any = None) -> str:
        token = secrets.token_urlsafe(32)
        expires = utc_now() + timedelta(hours=self.config.auth_session_ttl_hours)
        ip = None
        ua = None
        try:
            if request is not None:
                ip = request.client.host if request.client else None
                ua = request.headers.get("user-agent")
        except Exception:
            pass
        self.db.create_session(self._hash_token(token), user["id"], iso(expires), ip, ua)
        self.db.touch_login(user["id"])
        return token

    def resolve_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        row = self.db.get_session(self._hash_token(token))
        if not row:
            return None
        # Expiry check (ISO strings compare lexicographically in UTC).
        if row.get("expires_at") and str(row["expires_at"]) < iso(utc_now()):
            self.db.delete_session(row["token_hash"])
            return None
        user = self.db.get_user(row["user_id"])
        if not user or user.get("disabled"):
            return None
        return user

    def logout(self, token: str | None) -> None:
        if token:
            self.db.delete_session(self._hash_token(token))

    def set_cookie(self, response: Any, token: str) -> None:
        response.set_cookie(
            key=self.config.auth_cookie_name, value=token, httponly=True,
            samesite="lax", secure=bool(self.config.auth_cookie_secure),
            max_age=self.config.auth_session_ttl_hours * 3600, path="/")

    def clear_cookie(self, response: Any) -> None:
        response.delete_cookie(key=self.config.auth_cookie_name, path="/")

    # ---- local authentication -------------------------------------------- #
    def authenticate_local(self, username: str, password: str) -> dict | None:
        user = self.db.get_user_by_username((username or "").strip())
        if not user or user.get("disabled"):
            # Still run a hash to blunt username-enumeration timing.
            verify_password(password or "x", None)
            return None
        if user.get("source") not in ("local", "test"):
            return None
        if verify_password(password or "", user.get("password_hash")):
            return user
        return None

    # ---- request -> user -------------------------------------------------- #
    def _token_from_request(self, request: Any) -> str | None:
        try:
            cookie = request.cookies.get(self.config.auth_cookie_name)
        except Exception:
            cookie = None
        if cookie:
            return cookie
        try:
            authz = request.headers.get("authorization") or ""
        except Exception:
            authz = ""
        if authz.lower().startswith("bearer "):
            return authz[7:].strip()
        return None

    def user_for_request(self, request: Any) -> dict | None:
        """Resolve the acting user, or None. Synthetic admin when auth is off."""
        if not self.enabled:
            return self.synthetic_local_user()
        return self.resolve_token(self._token_from_request(request))

    # ---- FastAPI dependencies -------------------------------------------- #
    async def require_user(self, request: Request) -> dict:
        user = self.user_for_request(request)
        if not user:
            raise HTTPException(status_code=401, detail="authentication required")
        # Bind identity for structured logging AND for request-scoped data
        # isolation (RAG retrieval, memory tools read the acting user).
        bind_context(user_id=user["id"], username=user.get("username"),
                     role=user.get("role"))
        set_acting_user(user["id"])
        request.state.user = user
        return user

    async def require_admin(self, request: Request) -> dict:
        user = await self.require_user(request)
        if user.get("role") != "admin":
            log_event(_auth_log, 30, "auth.forbidden",
                      route=str(getattr(request, "url", "")), user_id=user["id"],
                      role=user.get("role"))
            raise HTTPException(status_code=403, detail="administrator access required")
        return user

    async def optional_user(self, request: Request) -> dict | None:
        user = self.user_for_request(request)
        if user:
            bind_context(user_id=user["id"], username=user.get("username"),
                         role=user.get("role"))
            set_acting_user(user["id"])
            request.state.user = user
        return user

    # ---- inter-node token ------------------------------------------------- #
    def check_node_token(self, request: Any) -> bool:
        """True if the request carries the shared inter-node bearer token."""
        expected = self.config.node_token
        if not expected:
            return False
        try:
            authz = request.headers.get("authorization") or ""
        except Exception:
            return False
        supplied = authz[7:].strip() if authz.lower().startswith("bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied, expected)

    # ---- OIDC / Entra ID -------------------------------------------------- #
    @property
    def oidc_enabled(self) -> bool:
        return bool(self.config.oidc_enabled and self.config.oidc_client_id
                    and (self.config.oidc_tenant_id or self.config.oidc_authority))

    def _authority(self) -> str:
        if self.config.oidc_authority:
            return self.config.oidc_authority.rstrip("/")
        return f"https://login.microsoftonline.com/{self.config.oidc_tenant_id}/v2.0"

    def oidc_discovery(self) -> dict:
        """Fetch and cache the OpenID discovery document."""
        now = time.time()
        if self._oidc_discovery and (now - self._oidc_discovery_at) < 3600:
            return self._oidc_discovery
        import httpx
        url = self._authority() + "/.well-known/openid-configuration"
        with httpx.Client(timeout=self.config.node_probe_timeout or 10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            self._oidc_discovery = resp.json()
            self._oidc_discovery_at = now
        return self._oidc_discovery

    def oidc_authorize_url(self, redirect_uri: str) -> str:
        """Build the Entra authorize URL and remember the state/nonce."""
        disc = self.oidc_discovery()
        state = secrets.token_urlsafe(24)
        nonce = secrets.token_urlsafe(24)
        with self._oidc_lock:
            # Drop expired states first (10 min TTL).
            cutoff = time.time() - 600
            self._oidc_state = {k: v for k, v in self._oidc_state.items() if v[0] > cutoff}
            self._oidc_state[state] = (time.time(), nonce, redirect_uri)
        params = {
            "client_id": self.config.oidc_client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "response_mode": "query",
            "scope": self.config.oidc_scopes or "openid profile email",
            "state": state,
            "nonce": nonce,
        }
        return disc["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)

    def oidc_exchange(self, code: str, state: str, redirect_uri: str) -> dict:
        """Exchange an auth code for tokens and return the verified user record.

        Raises ValueError on any validation failure.
        """
        with self._oidc_lock:
            entry = self._oidc_state.pop(state, None)
        if not entry:
            raise ValueError("unknown or expired login state")
        _ts, nonce, saved_redirect = entry
        redirect_uri = redirect_uri or saved_redirect
        disc = self.oidc_discovery()
        import httpx
        data = {
            "client_id": self.config.oidc_client_id,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "scope": self.config.oidc_scopes or "openid profile email",
        }
        if self.config.oidc_client_secret:
            data["client_secret"] = self.config.oidc_client_secret
        with httpx.Client(timeout=15) as client:
            resp = client.post(disc["token_endpoint"], data=data)
        if resp.status_code != 200:
            raise ValueError(f"token exchange failed ({resp.status_code})")
        tokens = resp.json()
        id_token = tokens.get("id_token")
        if not id_token:
            raise ValueError("no id_token in token response")
        claims = self._verify_id_token(id_token, nonce, disc)
        return self.oidc_upsert_user(claims)

    def _verify_id_token(self, id_token: str, nonce: str, disc: dict) -> dict:
        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("malformed id_token")
        claims = _b64url_json(parts[1])
        # Audience must be this client.
        aud = claims.get("aud")
        if aud != self.config.oidc_client_id and self.config.oidc_client_id not in (
                aud if isinstance(aud, list) else [aud]):
            raise ValueError("id_token audience mismatch")
        # Issuer must match discovery.
        if disc.get("issuer") and claims.get("iss") not in (disc["issuer"],):
            # Entra's issuer contains the tenant; allow the exact discovery issuer.
            raise ValueError("id_token issuer mismatch")
        # Expiry.
        if int(claims.get("exp", 0)) < int(time.time()) - 60:
            raise ValueError("id_token expired")
        # Nonce replay protection.
        if nonce and claims.get("nonce") and claims["nonce"] != nonce:
            raise ValueError("id_token nonce mismatch")
        # Optional signature verification if cryptography is available.
        self._maybe_verify_signature(id_token, disc)
        return claims

    def _maybe_verify_signature(self, id_token: str, disc: dict) -> None:
        """Best-effort JWKS signature check; skipped if libraries absent.

        The token was fetched directly from the TLS-protected token endpoint of a
        confidential client, so per the OIDC spec signature validation is not
        strictly required. When available we verify anyway as defence in depth.
        """
        try:
            import jwt  # PyJWT
            from jwt import PyJWKClient
        except Exception:
            return
        try:
            jwks_uri = disc.get("jwks_uri")
            if not jwks_uri:
                return
            signing_key = PyJWKClient(jwks_uri).get_signing_key_from_jwt(id_token)
            jwt.decode(id_token, signing_key.key,
                       algorithms=["RS256"], audience=self.config.oidc_client_id,
                       options={"verify_exp": True})
        except Exception as exc:
            raise ValueError(f"id_token signature verification failed: {exc}")

    def role_for_oidc(self, claims: dict) -> str:
        """Map Entra claims to an app role using the configured admin lists."""
        def _split(value: str) -> set[str]:
            return {p.strip().lower() for p in (value or "").split(",") if p.strip()}

        admin_emails = _split(self.config.oidc_admin_emails)
        admin_groups = _split(self.config.oidc_admin_groups)
        admin_roles = _split(self.config.oidc_admin_roles)

        email = (claims.get("email") or claims.get("preferred_username")
                 or claims.get("upn") or "").lower()
        if email and email in admin_emails:
            return "admin"
        groups = claims.get("groups") or []
        if isinstance(groups, str):
            groups = [groups]
        if admin_groups and {str(g).lower() for g in groups} & admin_groups:
            return "admin"
        roles = claims.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        if admin_roles and {str(r).lower() for r in roles} & admin_roles:
            return "admin"
        return (self.config.oidc_default_role or "user").lower()

    def oidc_upsert_user(self, claims: dict) -> dict:
        subject = claims.get("oid") or claims.get("sub")
        if not subject:
            raise ValueError("id_token has no subject")
        email = (claims.get("email") or claims.get("preferred_username")
                 or claims.get("upn"))
        name = claims.get("name") or email or subject
        role = self.role_for_oidc(claims)
        existing = self.db.get_user_by_oidc(subject)
        if existing:
            # Refresh role/profile from the IdP each login (IdP is source of truth).
            self.db.update_user(existing["id"], role=role, email=email,
                                display_name=name, disabled=0)
            return self.db.get_user(existing["id"]) or existing
        username = (email or f"entra-{subject[:8]}")
        # Avoid a username collision with a local account.
        if self.db.get_user_by_username(username):
            username = f"{username}#{subject[:6]}"
        return self.db.create_user(username, password_hash=None, role=role,
                                   source="oidc", email=email, display_name=name,
                                   oidc_subject=subject)


def public_auth_config(config: Config, auth: "Auth") -> dict:
    """Non-secret auth facts the login page needs to render itself."""
    return {
        "auth_enabled": bool(config.auth_enabled),
        "oidc_enabled": bool(auth.oidc_enabled),
        "allow_test_user": bool(config.allow_test_user),
        "app_name": APP_NAME,
    }


__all__ = [
    "Auth",
    "hash_password",
    "verify_password",
    "public_auth_config",
]
