"""Automatic Mac Mini (primary) / Mac Studio (secondary) routing and failover.

This is a real scheduler/router subsystem, not an ``if machine == ...`` switch.
Users never pick a machine; work is classified, nodes are evaluated on live
load/health/capability, one is selected with a recorded reason, and execution
fails over to another node without re-running committed work.

Pieces
------
* :class:`Node` — one execution target (the local mlx server, or a remote peer),
  with capabilities, capacity and live health state.
* :class:`NodeRegistry` — builds the node set from config: the primary is always
  the local model server; a secondary (Studio) exists only when ``STUDIO_NODE_URL``
  is set. With no secondary the registry has one node and routing is a no-op.
* :class:`ClusterRouter` — classifies a task, scores candidate nodes against
  configurable factors (active requests, queue depth, CPU/memory, node health,
  model capability, SLA), selects an ordered candidate list with a human reason,
  and records every decision to ``routing_events`` for admin observability.
* :class:`HealthMonitor` — a background heartbeat loop that probes each node and
  moves it between healthy / degraded / overloaded / draining / unavailable /
  starting, so a failed Studio never breaks the Mini and eligible Mini work
  shifts to the Studio when it is overloaded or down.

The router is exercised offline by the self-test with injected node snapshots and
a fake executor: overloaded-primary, unavailable-primary, large-model,
unavailable-secondary and recovery all have deterministic outcomes.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


_cluster_log = get_logger("routing")

# Node health states, from best to worst for scheduling.
HEALTHY = "healthy"
DEGRADED = "degraded"       # responding but slow / recent failures
OVERLOADED = "overloaded"   # at/over capacity; avoid unless nothing else
DRAINING = "draining"       # finishing in-flight work, take no new work
STARTING = "starting"       # coming up (loading weights)
UNAVAILABLE = "unavailable"  # not reachable / failing

# Order used when otherwise-equal: lower is preferred.
_STATE_RANK = {HEALTHY: 0, DEGRADED: 1, OVERLOADED: 2, STARTING: 3,
               DRAINING: 4, UNAVAILABLE: 5}


@dataclass
class Node:
    """One routable execution target."""
    name: str
    role: str                    # "primary" | "secondary"
    is_local: bool               # local mlx server vs a remote peer app
    capabilities: set[str] = field(default_factory=set)
    remote_url: str = ""         # base URL for a remote node (empty if local)
    # Live state (guarded by the registry lock).
    state: str = STARTING
    active: int = 0              # in-flight generations dispatched here
    last_latency_ms: float | None = None
    consecutive_failures: int = 0
    last_heartbeat: float = 0.0
    cpu_pct: float | None = None
    mem_pct: float | None = None
    # 1-minute load average divided by core count. A saturation RATIO (1.0 ==
    # fully committed), deliberately NOT reported as a percentage: it routinely
    # exceeds 1.0 on a healthy machine and used to be mislabelled as "cpu 100%",
    # which flagged an idle laptop as overloaded.
    load_ratio: float | None = None
    model: str | None = None
    # Detected hardware, so the cluster panel shows the real machine.
    machine: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    detail: str = ""
    # Circuit breaker: when the node trips to UNAVAILABLE it is skipped until this
    # timestamp, after which one half-open trial request is allowed through.
    cooldown_until: float = 0.0

    def can_serve(self, requirements: "Requirements") -> bool:
        if requirements.needs_large_model and "large_model" not in self.capabilities:
            return False
        for cap in requirements.required_capabilities:
            if cap not in self.capabilities:
                return False
        return True

    def snapshot(self) -> dict:
        return {
            "name": self.name, "role": self.role, "is_local": self.is_local,
            "state": self.state, "active": self.active,
            "capabilities": sorted(self.capabilities),
            "last_latency_ms": self.last_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "cpu_pct": self.cpu_pct, "mem_pct": self.mem_pct,
            "load_ratio": self.load_ratio, "machine": self.machine,
            "cores": self.cores, "ram_gb": self.ram_gb,
            "model": self.model, "detail": self.detail,
            "cooldown_remaining_s": (round(self.cooldown_until - time.time(), 1)
                                     if self.cooldown_until > time.time() else None),
            "last_heartbeat_age_s": (round(time.time() - self.last_heartbeat, 1)
                                     if self.last_heartbeat else None),
        }


@dataclass
class Requirements:
    """What a task needs, derived by classification."""
    kind: str = "chat"                       # chat | reasoning | code | task
    needs_large_model: bool = False
    required_capabilities: tuple[str, ...] = ()
    # Soft preference (not a hard requirement): heavy work advertises the
    # capabilities that make a node a better fit (e.g. "reasoning" for the
    # Studio), so the router steers toward it when it is ready and has capacity.
    prefer_capabilities: tuple[str, ...] = ()
    requested_model: str | None = None
    complexity: str = "normal"               # low | normal | high


@dataclass
class RoutingDecision:
    """The result of a selection: an ordered candidate list plus the why."""
    candidates: list[Node]
    reason: str
    requirements: Requirements
    snapshot: list[dict]

    @property
    def primary_choice(self) -> Node | None:
        return self.candidates[0] if self.candidates else None


# Apple model identifiers -> a short, readable node name.
_MODEL_NAMES = (
    ("macbookpro", "macbook-pro"), ("macbookair", "macbook-air"), ("macbook", "macbook"),
    ("macmini", "mac-mini"), ("macstudio", "mac-studio"), ("macpro", "mac-pro"),
    ("imacpro", "imac-pro"), ("imac", "imac"), ("virtualmac", "mac-vm"),
)

_MACHINE: dict | None = None


def friendly_node_name(model: str, hostname: str) -> str:
    """A short node name from the hardware model, falling back to the hostname."""
    key = re.sub(r"[^a-z]", "", (model or "").lower())
    for prefix, nice in _MODEL_NAMES:
        if key.startswith(prefix):
            return nice
    host = re.sub(r"[^a-z0-9-]", "", (hostname or "").lower())
    return host or (key or "local-node")


def detect_machine() -> dict:
    """What machine is this, really? Cached; probed once per process.

    The node used to be hardcoded to "mac-mini" regardless of hardware, which
    is simply wrong on any other Mac. This reads the actual model identifier so
    the registry and the admin panel describe the real machine.
    """
    global _MACHINE
    if _MACHINE is not None:
        return _MACHINE
    model = ""
    try:
        model = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True,
                               text=True, timeout=2).stdout.strip()
    except Exception:
        model = ""
    if not model:
        try:
            model = platform.machine()
        except Exception:
            model = ""
    host = ""
    try:
        host = socket.gethostname().split(".")[0]
    except Exception:
        host = ""
    _MACHINE = {
        "model": model,
        "hostname": host,
        "cores": os.cpu_count() or 1,
        "ram_gb": float(TOTAL_RAM_GB or 0),
        "name": friendly_node_name(model, host),
    }
    return _MACHINE


def capabilities_for_machine(ram_gb: float) -> set:
    """Capabilities a local node can honestly advertise for its RAM.

    Only a roomy machine may claim the high-memory / large-model / deep-reasoning
    capabilities the router steers heavy work to; an 8GB laptop must not.
    """
    caps = {"chat", "code", "default"}
    if ram_gb >= 32:
        caps |= {"high_memory", "large_model", "reasoning"}
    return caps


def sample_load_ratio() -> float | None:
    """1-minute load average per core. A ratio, not a percentage."""
    try:
        return round(os.getloadavg()[0] / (os.cpu_count() or 1), 2)
    except (OSError, AttributeError):
        return None


def sample_local_load() -> tuple[float | None, float | None]:
    """Best-effort (cpu_pct, mem_pct) for this machine.

    cpu_pct is a REAL utilisation sample and is None when we cannot measure one.
    It used to be derived from the load average (load/cores*100), which is not a
    utilisation figure at all: on an idle 8-core laptop a 1-minute load of 17
    produced "cpu 100%" and tripped the overload threshold. Better to report
    nothing than to report a number that means something else -- the router
    skips a factor it cannot measure (see sample_load_ratio for saturation).
    """
    cpu_pct = None
    mem_pct = None
    try:
        import psutil  # optional
        cpu_pct = float(psutil.cpu_percent(interval=None))
        mem_pct = float(psutil.virtual_memory().percent)
    except Exception:
        pass      # psutil absent: both stay None, i.e. "unmeasured", not "idle"
    return cpu_pct, mem_pct


class NodeRegistry:
    """The set of nodes, built from config, with thread-safe state updates."""

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.RLock()
        self.nodes: list[Node] = []
        self._build()

    def _build(self) -> None:
        cfg = self.config
        machine = detect_machine()
        primary = Node(
            name=cfg.node_name or machine["name"],
            role="primary", is_local=True,
            capabilities=capabilities_for_machine(machine["ram_gb"]),
            state=STARTING,
            machine=machine["model"], cores=machine["cores"], ram_gb=machine["ram_gb"],
        )
        self.nodes = [primary]
        if cfg.studio_node_url:
            studio = Node(
                name="mac-studio", role="secondary", is_local=False,
                remote_url=cfg.studio_node_url.rstrip("/"),
                # The Studio is a superset: everything the Mini does, plus the
                # high-memory / large-model / deep-reasoning capabilities.
                capabilities={"chat", "code", "default", "large_model",
                              "high_memory", "reasoning"},
                state=STARTING,
            )
            self.nodes.append(studio)

    @property
    def multi_node(self) -> bool:
        return len(self.nodes) > 1

    def local_node(self) -> Node:
        for node in self.nodes:
            if node.is_local:
                return node
        return self.nodes[0]

    def by_name(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def update(self, name: str, **fields: Any) -> None:
        with self._lock:
            node = self.by_name(name)
            if not node:
                return
            for key, value in fields.items():
                if hasattr(node, key):
                    setattr(node, key, value)

    def begin(self, node: Node) -> None:
        with self._lock:
            node.active += 1

    def end(self, node: Node, ok: bool, latency_ms: float | None = None) -> None:
        with self._lock:
            node.active = max(0, node.active - 1)
            if ok:
                # A success closes the breaker: a half-open trial that worked, or
                # ordinary healthy traffic, both clear the failure state.
                node.consecutive_failures = 0
                node.cooldown_until = 0.0
                if latency_ms is not None:
                    node.last_latency_ms = round(latency_ms, 1)
                if node.state in (UNAVAILABLE, STARTING, DEGRADED):
                    node.state = HEALTHY
            else:
                node.consecutive_failures += 1
                if node.consecutive_failures >= 2:
                    node.state = UNAVAILABLE
                    # Open the breaker: skip this node until the cooldown elapses,
                    # then allow one half-open trial (see ClusterRouter._eligible).
                    node.cooldown_until = time.time() + max(
                        0.0, getattr(self.config, "route_cooldown_s", 20.0))
                elif node.state == HEALTHY:
                    node.state = DEGRADED

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [n.snapshot() for n in self.nodes]


class ClusterRouter:
    """Classify a task, pick nodes, record the decision, and fail over."""

    def __init__(self, config: Config, registry: NodeRegistry, db: Any = None):
        self.config = config
        self.registry = registry
        self.db = db
        # In-flight task ids, so an accidental re-dispatch of the same unit of
        # work is detected rather than silently duplicated.
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    # ---- classification --------------------------------------------------- #
    def classify(self, *, model: str | None = None, kind: str = "chat",
                 complexity: str = "normal") -> Requirements:
        model = model or self.config.model
        markers = [m.strip().lower() for m in
                   (self.config.large_model_markers or "").split(",") if m.strip()]
        needs_large = any(m in (model or "").lower() for m in markers)
        req_caps: tuple[str, ...] = ()
        prefer: tuple[str, ...] = ()
        if kind == "reasoning" or complexity == "high":
            # Deep reasoning is eligible for (not required on) the Studio: a soft
            # preference, so heavy work steers toward the high-capability node
            # when it is ready, but light chat is never forced off the primary.
            prefer = ("reasoning",)
        return Requirements(kind=kind, needs_large_model=needs_large,
                            required_capabilities=req_caps,
                            prefer_capabilities=prefer,
                            requested_model=model, complexity=complexity)

    # ---- capacity / health helpers --------------------------------------- #
    def _is_overloaded(self, node: Node) -> tuple[bool, str]:
        cfg = self.config
        if node.active >= cfg.route_max_active_per_node:
            return True, f"active {node.active}>={cfg.route_max_active_per_node}"
        if node.cpu_pct is not None and node.cpu_pct >= cfg.route_cpu_pct:
            return True, f"cpu {node.cpu_pct}%>={cfg.route_cpu_pct}%"
        if node.mem_pct is not None and node.mem_pct >= cfg.route_mem_pct:
            return True, f"mem {node.mem_pct}%>={cfg.route_mem_pct}%"
        if (node.load_ratio is not None and cfg.route_load_ratio
                and node.load_ratio >= cfg.route_load_ratio):
            return True, f"load {node.load_ratio}x>={cfg.route_load_ratio}x per core"
        if (cfg.route_sla_ms and node.last_latency_ms is not None
                and node.last_latency_ms >= cfg.route_sla_ms):
            return True, f"latency {node.last_latency_ms}ms>=SLA {cfg.route_sla_ms}ms"
        return False, ""

    def _eligible(self, node: Node, req: Requirements) -> bool:
        if not node.can_serve(req) or node.state == DRAINING:
            return False
        if node.state == UNAVAILABLE:
            # Half-open: a tripped node is eligible again only after its cooldown
            # elapses, and even then it scores worst (see _score), so it is used
            # only as a single trial when nothing healthier can serve the work.
            return node.cooldown_until > 0 and time.time() >= node.cooldown_until
        return True

    def _is_half_open(self, node: Node) -> bool:
        return (node.state == UNAVAILABLE and node.cooldown_until > 0
                and time.time() >= node.cooldown_until)

    def _score(self, node: Node) -> float:
        """Lower is better. Orders eligible nodes by health, then live load.

        Health dominates (a healthy node always beats a degraded one); within a
        health tier the least-busy, lowest-latency, lowest-utilisation node wins,
        so a second request does not pile onto a node already working.
        """
        score = _STATE_RANK.get(node.state, 9) * 1000.0
        score += node.active * 50.0
        if node.cpu_pct is not None:
            score += node.cpu_pct
        if node.mem_pct is not None:
            score += node.mem_pct
        if node.last_latency_ms is not None:
            score += node.last_latency_ms / 50.0
        return score

    # ---- selection -------------------------------------------------------- #
    def select(self, req: Requirements) -> RoutingDecision:
        """Return an ordered candidate list (best first) and the reason."""
        nodes = self.registry.nodes
        snapshot = self.registry.snapshot()

        # Single-node install: the local node is the only answer.
        if len(nodes) == 1:
            only = nodes[0]
            reason = "single node (no secondary configured)"
            if not only.can_serve(req):
                reason = ("single node cannot meet requirement "
                          f"(needs_large_model={req.needs_large_model}); using it anyway")
            return RoutingDecision([only], reason, req, snapshot)

        primary = self.registry.local_node()

        eligible = [n for n in nodes if self._eligible(n, req)]
        reasons: list[str] = []

        # Hard capability need (large model): only capable nodes qualify.
        if req.needs_large_model:
            capable = [n for n in eligible if "large_model" in n.capabilities]
            if capable:
                capable.sort(key=self._score)
                order = capable + sorted((n for n in eligible if n not in capable),
                                         key=self._score)
                reasons.append(f"large model {req.requested_model!r} requires "
                               "high-memory node")
                return RoutingDecision(order, "; ".join(reasons), req, snapshot)
            # No capable node up: fall through to best-effort below.
            reasons.append("no high-memory node available for large model; best effort")

        # Soft capability preference (heavy reasoning): steer toward a ready
        # capability node (the Studio) with spare capacity, keeping the primary
        # as the immediate fallback. Light chat has no preference and falls
        # through to the primary-first path below.
        if req.prefer_capabilities:
            preferred = [
                n for n in eligible
                if n is not primary
                and all(c in n.capabilities for c in req.prefer_capabilities)
                and n.state in (HEALTHY, DEGRADED)
                and not self._is_overloaded(n)[0]
            ]
            if preferred:
                preferred.sort(key=self._score)
                chosen = preferred[0]
                rest = sorted((n for n in eligible if n is not chosen), key=self._score)
                reasons.append(f"heavy {req.kind} work -> capability node "
                               f"{sorted(req.prefer_capabilities)}")
                return RoutingDecision([chosen] + rest, "; ".join(reasons), req, snapshot)

        # Ordinary case: prefer the primary unless it is unfit. A node still
        # loading weights (STARTING) is eligible but not ready to be *preferred*
        # over a healthy secondary, so it is treated like an overloaded primary.
        if primary in eligible:
            over, why = self._is_overloaded(primary)
            ready = primary.state in (HEALTHY, DEGRADED)
            if not over and ready:
                order = [primary] + sorted((n for n in eligible if n is not primary),
                                           key=self._score)
                reasons.append("primary healthy and within capacity")
                return RoutingDecision(order, "; ".join(reasons), req, snapshot)
            reasons.append(f"primary overloaded ({why})" if over
                           else f"primary not ready ({primary.state})")
            # Offload to the least-loaded eligible secondary if one can take it.
            others = sorted((n for n in eligible if n is not primary), key=self._score)
            takeable = [n for n in others if not self._is_overloaded(n)[0]]
            if takeable:
                pick = takeable[0]
                reasons.append("secondary has capacity -> offload"
                               + (" (half-open trial)" if self._is_half_open(pick) else ""))
                rest = [primary] + [n for n in others if n is not pick]
                return RoutingDecision([pick] + rest, "; ".join(reasons), req, snapshot)
            if others:
                reasons.append("secondary also loaded; keep on primary")
                return RoutingDecision([primary] + others, "; ".join(reasons), req, snapshot)
            reasons.append("secondary unavailable; keep on primary")
            return RoutingDecision([primary], "; ".join(reasons), req, snapshot)

        # Primary not eligible (down/incapable): use the best eligible secondary.
        others = sorted((n for n in eligible if n is not primary), key=self._score)
        if others:
            pick = others[0]
            reasons.append("primary unavailable -> secondary"
                           + (" (half-open trial)" if self._is_half_open(pick) else ""))
            return RoutingDecision(others, "; ".join(reasons), req, snapshot)

        # Nothing eligible: return the least-bad node so the caller degrades
        # honestly rather than 500ing with no target.
        fallback = sorted(nodes, key=self._score)
        reasons.append("no healthy node; best-effort fallback")
        return RoutingDecision(fallback, "; ".join(reasons), req, snapshot)

    # ---- observability ---------------------------------------------------- #
    def record(self, decision: RoutingDecision, node: Node, *, status: str,
               attempt: int = 0, duration_ms: float | None = None,
               error: str | None = None, correlation_id: str | None = None,
               task_id: str | None = None, user_id: str | None = None,
               conversation_id: str | None = None) -> None:
        log_event(_cluster_log, 20, "routing.decision",
                  node=node.name, status=status, attempt=attempt,
                  kind=decision.requirements.kind,
                  needs_large_model=decision.requirements.needs_large_model,
                  requested_model=decision.requirements.requested_model,
                  reason=decision.reason, duration_ms=duration_ms,
                  correlation_id=correlation_id, task_id=task_id)
        if self.db is not None:
            try:
                self.db.log_routing_event(
                    correlation_id=correlation_id, task_id=task_id, user_id=user_id,
                    conversation_id=conversation_id, kind=decision.requirements.kind,
                    requested_model=decision.requirements.requested_model,
                    selected_model=node.model or decision.requirements.requested_model,
                    selected_node=node.name, reason=decision.reason,
                    candidates=decision.snapshot, status=status, attempt=attempt,
                    duration_ms=duration_ms, error=error)
            except Exception as exc:  # never let telemetry break a request
                log_event(_cluster_log, 30, "routing.record_failed", error=str(exc))

    # ---- idempotency ------------------------------------------------------ #
    def claim(self, task_id: str) -> bool:
        """Register a unit of work. False if it is already in flight (a dup)."""
        if not task_id:
            return True
        with self._inflight_lock:
            if task_id in self._inflight:
                return False
            self._inflight.add(task_id)
            return True

    def release(self, task_id: str) -> None:
        if not task_id:
            return
        with self._inflight_lock:
            self._inflight.discard(task_id)


class HealthMonitor:
    """Background heartbeat loop that keeps node states current."""

    def __init__(self, config: Config, registry: NodeRegistry,
                 local_status: Callable[[], dict] | None = None):
        self.config = config
        self.registry = registry
        # Callable returning {"status": <mlx status>, "model": <id>} for the
        # local model server (wired to ModelServerManager in create_app).
        self.local_status = local_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="cluster-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _loop(self) -> None:
        # Probe once promptly, then on the configured interval.
        while True:
            try:
                self.tick()
            except Exception as exc:  # never let the monitor thread die
                log_event(_cluster_log, 40, "heartbeat.error", error=str(exc))
            if self._stop.wait(max(1.0, self.config.heartbeat_interval)):
                return

    def tick(self) -> None:
        """One heartbeat pass over every node."""
        cpu_pct, mem_pct = sample_local_load()
        load_ratio = sample_load_ratio()
        for node in list(self.registry.nodes):
            if node.is_local:
                self._probe_local(node, cpu_pct, mem_pct, load_ratio)
            else:
                self._probe_remote(node)

    def _probe_local(self, node: Node, cpu_pct, mem_pct, load_ratio=None) -> None:
        state = HEALTHY
        detail = ""
        model = node.model
        if self.local_status is not None:
            try:
                info = self.local_status() or {}
                status = str(info.get("status", ""))
                model = info.get("model") or model
                if status == "ready":
                    state = HEALTHY
                elif status in ("loading", "starting", "restarting"):
                    state = STARTING
                elif status == "stopped":
                    state = UNAVAILABLE
                else:
                    state = DEGRADED
                    detail = status
            except Exception as exc:
                state, detail = DEGRADED, str(exc)
        # Capacity pressure downgrades an otherwise-healthy node. Every factor is
        # skipped when it cannot be measured -- an unmeasured signal must never
        # read as "over the limit" (that is what made an idle laptop overloaded).
        if state == HEALTHY:
            cfg = self.config
            over_cpu = cpu_pct is not None and cpu_pct >= cfg.route_cpu_pct
            over_mem = mem_pct is not None and mem_pct >= cfg.route_mem_pct
            over_load = (load_ratio is not None and cfg.route_load_ratio
                         and load_ratio >= cfg.route_load_ratio)
            over_active = node.active >= cfg.route_max_active_per_node
            if over_cpu or over_mem or over_load or over_active:
                state = OVERLOADED
        fields = dict(state=state, cpu_pct=cpu_pct, mem_pct=mem_pct,
                      load_ratio=load_ratio, model=model,
                      detail=detail, last_heartbeat=time.time())
        # A healthy probe closes the breaker, exactly as _probe_remote and
        # end(ok=True) do: otherwise a primary that tripped to UNAVAILABLE and
        # recovered keeps consecutive_failures>=2, so the next single failure
        # re-trips it straight to UNAVAILABLE instead of degrading first.
        if state == HEALTHY:
            fields["consecutive_failures"] = 0
            fields["cooldown_until"] = 0.0
        self.registry.update(node.name, **fields)

    def _probe_remote(self, node: Node) -> None:
        import httpx
        url = node.remote_url.rstrip("/") + "/api/node/health"
        headers = {}
        if self.config.node_token:
            headers["Authorization"] = f"Bearer {self.config.node_token}"
        started = time.time()
        try:
            with httpx.Client(timeout=self.config.node_probe_timeout) as client:
                resp = client.get(url, headers=headers)
            latency = (time.time() - started) * 1000
            if resp.status_code == 200:
                info = resp.json()
                remote_state = str(info.get("state") or info.get("model_status") or "")
                model = info.get("model")
                cpu = info.get("cpu_pct")
                mem = info.get("mem_pct")
                state = HEALTHY
                if remote_state in ("loading", "starting", "restarting"):
                    state = STARTING
                elif remote_state in ("stopped", "unavailable"):
                    state = UNAVAILABLE
                elif remote_state in ("overloaded",):
                    state = OVERLOADED
                self.registry.update(node.name, state=state, model=model,
                                     cpu_pct=cpu, mem_pct=mem,
                                     last_latency_ms=round(latency, 1),
                                     consecutive_failures=0,
                                     last_heartbeat=time.time(), detail="")
            else:
                self._mark_unreachable(node, f"health {resp.status_code}")
        except Exception as exc:
            self._mark_unreachable(node, str(exc))

    def _mark_unreachable(self, node: Node, detail: str) -> None:
        failures = node.consecutive_failures + 1
        state = UNAVAILABLE if failures >= 2 else DEGRADED
        fields = dict(state=state, consecutive_failures=failures,
                      last_heartbeat=time.time(), detail=detail[:200])
        # Opening the breaker from the heartbeat too, so the router's half-open
        # trial logic works whether the failure was seen by a request or a probe.
        if state == UNAVAILABLE:
            fields["cooldown_until"] = time.time() + max(
                0.0, getattr(self.config, "route_cooldown_s", 20.0))
        self.registry.update(node.name, **fields)
        log_event(_cluster_log, 30, "heartbeat.node_unreachable",
                  node=node.name, detail=detail[:200], state=state)


__all__ = [
    "Node",
    "NodeRegistry",
    "Requirements",
    "RoutingDecision",
    "ClusterRouter",
    "HealthMonitor",
    "sample_local_load",
    "sample_load_ratio",
    "detect_machine",
    "friendly_node_name",
    "capabilities_for_machine",
    "HEALTHY", "DEGRADED", "OVERLOADED", "DRAINING", "STARTING", "UNAVAILABLE",
]
