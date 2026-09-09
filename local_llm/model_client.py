"""HTTP client for the local model server, with retries.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403


class ModelClient:
    """Async client for the OpenAI-compatible model backend.

    When a cluster (ClusterRouter) is attached it becomes node-aware: each
    generation is routed to the best node (Mac Mini primary / Mac Studio
    secondary) and fails over to the next candidate on a connection failure,
    recording every decision. With no cluster, or a single-node cluster, it
    targets the local mlx server exactly as before.
    """

    def __init__(self, config: Config, cluster: Any = None):
        self.config = config
        self.cluster = cluster

    # One pooled HTTP client for the whole process, not one per request and not
    # one per ModelClient. A fresh AsyncClient per request means a fresh TCP
    # connection per request: on a multi-step agent run that is a connect
    # handshake before every token stream, and the non-streaming path throws the
    # pool away between its two calls. It is a class attribute because
    # /api/agents/run builds a ModelClient per agent, so per-instance pools would
    # give the fan-out no reuse at all. Every call site passes its own timeout,
    # so nothing config-specific is baked into the shared client.
    _shared_http: Any = None
    _shared_loop: Any = None

    @classmethod
    def _client(cls):
        """The shared AsyncClient, created on first use inside the running loop.

        A connection pool belongs to the event loop that created it, and this
        process starts a fresh loop per test app, so the client is rebound
        whenever the running loop changes; otherwise it would hand out sockets
        the current loop cannot await.
        """
        import httpx
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        client = cls._shared_http
        if client is None or cls._shared_loop is not loop or client.is_closed:
            cls._shared_http = httpx.AsyncClient(
                # Generous per-host pool: concurrent chats and agent fan-out all
                # target the same one or two nodes.
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16,
                                    keepalive_expiry=120.0),
                timeout=httpx.Timeout(120.0, connect=15.0, pool=15.0),
            )
            cls._shared_loop = loop
        return cls._shared_http

    @classmethod
    async def aclose(cls) -> None:
        """Release the pooled connections on shutdown. Safe to call twice."""
        client, cls._shared_http, cls._shared_loop = cls._shared_http, None, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass

    def _local_chat_url(self) -> str:
        return f"http://127.0.0.1:{self.config.model_port}/v1/chat/completions"

    @property
    def url(self) -> str:
        return self._local_chat_url()

    def _targets(self, kind: str = "chat"):
        """Ordered list of (node, chat_url, headers, decision) to try.

        A single-node / no-cluster client yields exactly one local target, so the
        generation path is byte-for-byte the old behaviour.
        """
        cluster = self.cluster
        if cluster is None or not getattr(cluster.registry, "multi_node", False):
            return [(None, self._local_chat_url(), {}, None)]
        req = cluster.classify(model=self.config.model, kind=kind)
        decision = cluster.select(req)
        targets = []
        for node in decision.candidates:
            if node.is_local:
                targets.append((node, self._local_chat_url(), {}, decision))
            else:
                headers = {}
                if self.config.node_token:
                    headers["Authorization"] = f"Bearer {self.config.node_token}"
                url = node.remote_url.rstrip("/") + "/api/node/generate"
                targets.append((node, url, headers, decision))
        return targets

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """True for failures where trying another node is the right move.

        Covers transport errors (dropped/refused/timed-out connection) AND a 5xx
        from the model server itself: a 500/502/503 usually means the primary mlx
        server crashed or is reloading, so a healthy secondary should be tried
        rather than hammering the dead primary.
        """
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                            httpx.RemoteProtocolError, httpx.ReadTimeout,
                            httpx.PoolTimeout)):
            return True
        text = str(exc).lower()
        if re.search(r"returned 5\d\d", text):  # "model server returned 5xx: ..."
            return True
        return any(s in text for s in ("connection", "refused", "reset",
                                       "timed out", "unreachable"))

    async def wait_until_ready(self, timeout: float = 40.0) -> bool:
        """Wait until the server can actually GENERATE, or until timeout.

        A /v1/models ping only proves the process is up. It does not prove the
        generation thread is alive: a backend crash (e.g. mlx-lm dying inside
        _step) leaves the HTTP server answering 200 on /v1/models while every
        completion stalls forever. So probe with a real one-token completion.
        If that returns, the server can generate; if it errors or times out, it
        cannot, and the caller degrades honestly instead of retrying into a
        corpse. Returns True only on a genuine completion.
        """
        import httpx
        deadline = time.time() + timeout
        delay = 0.5
        probe_body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=min(15.0, timeout)) as client:
                    resp = await client.post(self.url, json=probe_body)
                if resp.status_code < 500:
                    data = resp.json()
                    # A live generation thread returns a choices array; a crashed
                    # backend that still serves HTTP will not.
                    if data.get("choices"):
                        return True
            except Exception:
                pass      # still starting: retry until the deadline below
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)
        return False

    @staticmethod
    def classify_error(exc: Exception) -> str:
        """A short, honest label for a generation failure.

        The retry notice used to say "memory limit" for every exception, which
        hid stalls, resets and Cloudflare-style pages behind a wrong cause. This
        names what actually happened so the UI and logs are truthful.
        """
        import httpx
        name = type(exc).__name__
        text = str(exc).lower()
        if isinstance(exc, httpx.ReadTimeout) or "readtimeout" in name.lower() or "timed out" in text:
            return "the model stalled (no output in time)"
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)) or \
                "connection" in text or "reset" in text or "refused" in text:
            return "the model server dropped, likely out of memory and restarting"
        if "memory" in text or "oom" in text or "alloc" in text:
            return "the model server ran out of memory"
        return f"a generation error ({name})"

    def payload(
        self,
        messages: list[dict],
        stream: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        body = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        # mlx_lm.server reads these from the request body; they are not OpenAI
        # parameters. They are OMITTED by default: on recent mlx-lm the
        # repetition-penalty path builds a null logits-processor list and the
        # generation thread crashes on every request. Only send them when
        # explicitly re-enabled.
        if (self.config.repetition_penalty_enabled
                and self.config.repetition_penalty
                and self.config.repetition_penalty != 1.0):
            body["repetition_penalty"] = self.config.repetition_penalty
            body["repetition_context_size"] = self.config.repetition_context_size
        if stream:
            # Ask for usage on the final chunk. Servers that do not know the
            # option ignore it, and the estimate below covers them.
            body["stream_options"] = {"include_usage": True}
        if self.config.disable_thinking and not self.config.reasoning_visible:
            # Qwen3.5 and friends default to thinking-on. Chain of thought is a
            # bad trade in a tool loop: it burns the KV cache on tokens the
            # protocol discards, and the reasoning text confuses JSON parsing.
            # Both spellings are in circulation; unknown keys are ignored.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["enable_thinking"] = False
        return body

    def _stats_from_usage(self, usage: dict | None, messages: list[dict], text: str,
                          started: float, first_token_at: float | None) -> GenerationStats:
        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_token_at - started) * 1000) if first_token_at else total_ms
        if usage:
            return GenerationStats(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                ttft_ms=ttft_ms, total_ms=total_ms, from_server=True,
            )
        return GenerationStats(
            prompt_tokens=messages_tokens(messages),
            completion_tokens=estimate_tokens(text),
            ttft_ms=ttft_ms, total_ms=total_ms, from_server=False,
        )

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        text, _ = await self.complete_with_stats(messages, max_tokens, temperature)
        return text

    async def complete_with_stats(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
    ) -> tuple[str, GenerationStats]:
        import httpx
        import uuid as _uuid
        targets = self._targets(kind)
        task_id = _uuid.uuid4().hex[:16]
        cid = get_correlation_id()
        last_error: Exception | None = None
        active_node = None
        node_ended = True
        if self.cluster is not None:
            self.cluster.claim(task_id)
        try:
            for attempt, (node, url, headers, decision) in enumerate(targets):
                started = time.time()
                active_node, node_ended = node, False
                if self.cluster is not None and node is not None:
                    self.cluster.begin(node)
                try:
                    timeout = httpx.Timeout(self.config.stall_timeout, connect=15.0, pool=15.0)
                    client = self._client()
                    resp = await client.post(url, headers=headers, timeout=timeout,
                                             json=self.payload(messages, False, max_tokens, temperature))
                    if resp.status_code != 200:
                        fallback = self.payload(messages, False, max_tokens, temperature)
                        fallback.pop("max_tokens", None)
                        resp = await client.post(url, headers=headers, timeout=timeout,
                                                 json=fallback)
                    if resp.status_code != 200:
                        raise RuntimeError(f"model server returned {resp.status_code}: {resp.text[:300]}")
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    stats = self._stats_from_usage(data.get("usage"), messages, text, started, None)
                    stats.finish_reason = str(
                        (data["choices"][0] or {}).get("finish_reason") or "")
                    dur = (time.time() - started) * 1000
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, True, dur)
                        node_ended = True
                        self.cluster.record(decision, node,
                                            status=("ok" if attempt == 0 else "failover_ok"),
                                            attempt=attempt, duration_ms=dur,
                                            correlation_id=cid, task_id=task_id,
                                            conversation_id=conversation_id, user_id=get_acting_user())
                    return text, stats
                except Exception as exc:
                    last_error = exc
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, False)
                        node_ended = True
                        self.cluster.record(decision, node, status="failed", attempt=attempt,
                                            error=self.classify_error(exc), correlation_id=cid,
                                            task_id=task_id, conversation_id=conversation_id, user_id=get_acting_user())
                    if (node is not None and attempt + 1 < len(targets)
                            and self._is_connection_error(exc)):
                        log_event(get_logger("model"), 30, "model.failover",
                                  from_node=node.name, error=self.classify_error(exc),
                                  correlation_id=cid, task_id=task_id)
                        continue
                    raise
            raise last_error or RuntimeError("no model node available")
        finally:
            # Safety net for a cancellation (CancelledError is BaseException, not
            # caught above): decrement the active count so the node is not left
            # "busy". A consumer leaving is NOT a node failure, so this end is
            # recorded as success -- exactly as stream() does. Real failures are
            # handled in the except above and set node_ended, so they are never
            # double-counted here.
            if self.cluster is not None:
                if active_node is not None and not node_ended:
                    self.cluster.end(active_node, True)
                self.cluster.release(task_id)

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stats: GenerationStats | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield content deltas. When `stats` is passed it is filled in place.

        In place rather than returned because this is an async generator and the
        caller needs the numbers even when it breaks out of the loop early, which
        the agent does on every completed tool call. Fails over to another node
        only before the first token arrives (never mid-stream).
        """
        import httpx
        import uuid as _uuid
        targets = self._targets(kind)
        task_id = _uuid.uuid4().hex[:16]
        cid = get_correlation_id()
        started = time.time()
        first_token_at: float | None = None
        usage: dict | None = None
        finish_reason = ""
        text_len = 0
        last_error: Exception | None = None
        active_node = None
        node_ended = True
        streaming_started = False
        if self.cluster is not None:
            self.cluster.claim(task_id)
        try:
            for attempt, (node, url, headers, decision) in enumerate(targets):
                started = time.time()
                first_token_at = None
                usage = None
                text_len = 0
                streaming_started = False
                active_node, node_ended = node, False
                if self.cluster is not None and node is not None:
                    self.cluster.begin(node)
                try:
                    timeout = httpx.Timeout(self.config.stall_timeout, connect=15.0, pool=15.0)
                    client = self._client()
                    async with client.stream(
                        "POST", url, headers=headers, timeout=timeout,
                        json=self.payload(messages, True, max_tokens, temperature)
                    ) as resp:
                        if resp.status_code != 200:
                            body = (await resp.aread()).decode("utf-8", "replace")
                            raise RuntimeError(f"model server returned {resp.status_code}: {body[:300]}")
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                            except Exception:
                                continue
                            if data.get("usage"):
                                usage = data["usage"]
                            choices = data.get("choices") or []
                            if not choices:
                                continue
                            if choices[0].get("finish_reason"):
                                finish_reason = str(choices[0]["finish_reason"])
                            delta = (choices[0].get("delta") or {}).get("content", "")
                            if delta:
                                if first_token_at is None:
                                    first_token_at = time.time()
                                    streaming_started = True
                                text_len += len(delta)
                                yield delta
                    # Completed this node successfully.
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, True, (time.time() - started) * 1000)
                        node_ended = True
                        self.cluster.record(decision, node,
                                            status=("ok" if attempt == 0 else "failover_ok"),
                                            attempt=attempt, duration_ms=(time.time() - started) * 1000,
                                            correlation_id=cid, task_id=task_id,
                                            conversation_id=conversation_id, user_id=get_acting_user())
                    return
                except Exception as exc:
                    last_error = exc
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, False)
                        node_ended = True
                        self.cluster.record(decision, node, status="failed", attempt=attempt,
                                            error=self.classify_error(exc), correlation_id=cid,
                                            task_id=task_id, conversation_id=conversation_id, user_id=get_acting_user())
                    if (node is not None and not streaming_started
                            and attempt + 1 < len(targets) and self._is_connection_error(exc)):
                        log_event(get_logger("model"), 30, "model.failover",
                                  from_node=node.name, error=self.classify_error(exc),
                                  correlation_id=cid, task_id=task_id)
                        continue
                    raise
            if last_error:
                raise last_error
        finally:
            # Safety net for early consumer close (GeneratorExit / cancellation):
            # decrement the active count so a node is never left "busy" after the
            # agent stops reading. A consumer leaving is NOT a node failure, so
            # this end is recorded as success (real failures are handled above and
            # set node_ended). Also releases the idempotency claim.
            if (self.cluster is not None and active_node is not None and not node_ended):
                self.cluster.end(active_node, True, (time.time() - started) * 1000)
            if self.cluster is not None:
                self.cluster.release(task_id)
            if stats is not None:
                measured = self._stats_from_usage(
                    usage, messages, "x" * text_len, started, first_token_at
                )
                stats.prompt_tokens = measured.prompt_tokens
                stats.completion_tokens = measured.completion_tokens
                stats.finish_reason = finish_reason
                stats.ttft_ms = measured.ttft_ms
                stats.total_ms = measured.total_ms
                stats.from_server = measured.from_server


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------
# The old approach tried to recognise every phrasing of "search this" or "what
# is the weather in X" with regular expressions. That is a losing game: users
# phrase things in unbounded ways, and each new phrasing needed another pattern.
# The general fix is to make the *model* the router. It emits one structured
# decision (a small JSON object) saying how to handle the message and, for a
# lookup, what to search or which place and day it is about. Deterministic code
# then executes that decision. The model does what models are good at (reading
# intent from free text); the code does what code is good at (reliably running
# the chosen tool). This module holds the two deterministic shortcuts kept for
# cost reasons, plus the JSON extraction the router relies on.

# A bare arithmetic expression: only digits, spaces and operators. Routed to the
# calculator without a model call, since "17*23" needs no interpretation.
ARITHMETIC_ONLY = re.compile(r"^[\d\s+\-*/().,^%]+$")
# A message that is nothing but a URL. Routed straight to fetch_url.
BARE_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.I)

# Greetings and acknowledgements that are not worth a routing round trip. These
# get a plain reply; everything longer is eligible for model routing.
TRIVIAL_MESSAGE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup)(?:\s+(?:there|all|everyone|folks|claude))?"
    r"|^\s*(?:thanks|thank you|thx|ok|okay|cool|nice|got it|great|perfect|"
    r"lol|haha|bye|goodbye|good (?:morning|evening|night)|"
    r"how are you|what's up|whats up)\b",
    re.I,
)
_TRIVIAL_TAIL = re.compile(r"[\s!.?]*$")



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ARITHMETIC_ONLY',
    'BARE_URL',
    'ModelClient',
    'TRIVIAL_MESSAGE',
    '_TRIVIAL_TAIL',
]
