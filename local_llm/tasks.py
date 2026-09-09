"""Scheduled and background task runs.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace as dataclass_replace
from typing import AsyncGenerator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403


class TaskRun:
    """Live state for one run: the event buffer plus everyone tailing it.

    Events are buffered in memory so a browser that connects halfway through
    still sees the whole run, and the interesting ones are also persisted so a
    run survives a page reload or a restart. Tokens are deliberately not
    persisted: one run would otherwise write thousands of rows.
    """

    BUFFER_LIMIT = 4000

    def __init__(self, run_id: str, task_id: str, task_name: str):
        self.run_id = run_id
        self.task_id = task_id
        self.task_name = task_name
        self.seq = 0
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.cancel = asyncio.Event()
        self.done = False
        self.status = "running"
        self.answer = ""
        self.started = time.time()

    def publish(self, event: dict) -> dict:
        self.seq += 1
        stamped = dict(event)
        stamped["seq"] = self.seq
        stamped["run_id"] = self.run_id
        stamped["task_id"] = self.task_id
        self.events.append(stamped)
        if len(self.events) > self.BUFFER_LIMIT:
            # Drop the oldest tokens first; the structural events are the record.
            self.events = ([e for e in self.events if e["type"] != "token"][-self.BUFFER_LIMIT:]
                           or self.events[-self.BUFFER_LIMIT:])
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(stamped)
            except asyncio.QueueFull:
                # A tab that cannot keep up loses tokens, not the run.
                pass
        return stamped

    def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass  # a subscriber too slow to take the sentinel is already gone


class TaskManager:
    """Runs agent tasks in the background, on demand or on an interval.

    One scheduler coroutine on the web process's event loop polls for due tasks.
    Runs are gated by a semaphore because the model server answers one request
    at a time: firing five tasks at once just queues them inside mlx_lm with no
    way to see the queue.
    """

    # Everything except tokens goes to the database.
    PERSISTED = {"start", "context", "step", "tool_call", "tool_result",
                 "final", "error", "cancelled"}

    def __init__(
        self,
        config: Config,
        db: Database,
        model_manager: ModelServerManager,
        retrain_manager: RetrainManager,
        client: ModelClient,
    ):
        self.config = config
        self.db = db
        self.model_manager = model_manager
        self.retrain_manager = retrain_manager
        self.client = client
        # Cluster router, set by create_app so per-run model clients route and
        # fail over across nodes exactly like interactive chat. None = single node.
        self.cluster = getattr(client, "cluster", None)
        self.active: dict[str, TaskRun] = {}
        self.by_task: dict[str, TaskRun] = {}
        self.recent: dict[str, TaskRun] = {}
        self._scheduler: asyncio.Task | None = None
        self._semaphore: asyncio.Semaphore | None = None
        # Strong references to the in-flight run coroutines. The event loop only
        # holds a weak reference to a Task, so a run whose handle is dropped can
        # be garbage collected mid-execution and simply vanish. self.active
        # holds the TaskRun record, not the Task, so it does not protect this.
        self._runners: set[asyncio.Task] = set()
        self._stopping = False
        # Wall clock of the most recent interactive request. A scheduled run
        # landing mid-conversation puts two inference requests into a server
        # that holds one KV cache comfortably and two only by swapping, and on
        # 8GB the failure mode is not an error, it is macOS compressing memory
        # while tok/s quietly collapses.
        self.last_chat_at = 0.0
        self.chat_in_flight = 0

    def note_chat_activity(self) -> None:
        self.last_chat_at = time.time()

    def chat_is_busy(self) -> bool:
        if self.chat_in_flight > 0:
            return True
        return (time.time() - self.last_chat_at) < self.config.chat_idle_seconds

    # ------------------------------------------------------------ lifecycle --

    async def start(self) -> None:
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_tasks))
        orphans = self.db.reset_orphan_runs()
        if orphans:
            log(f"Marked {orphans} task run(s) interrupted by the previous shutdown.",
                logging.WARNING)
        self._scheduler = asyncio.create_task(self._scheduler_loop())
        log("Task scheduler started.")

    async def stop(self) -> None:
        self._stopping = True
        for run in list(self.active.values()):
            run.cancel.set()
        if self._scheduler is not None:
            self._scheduler.cancel()
            try:
                await self._scheduler
            except (asyncio.CancelledError, Exception):
                pass      # we asked it to stop; whatever it raises is not news
        # Wait on the run tasks themselves. Polling self.active only observed
        # the bookkeeping dict, so a runner still unwinding its finally block
        # could outlive shutdown and write to a closed database.
        if self._runners:
            await asyncio.wait(set(self._runners), timeout=5)
        for runner in list(self._runners):
            runner.cancel()

    async def _scheduler_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(max(1, self.config.task_poll_seconds))
                if self.model_manager.status != "ready":
                    continue
                if self.retrain_manager.status.get("running"):
                    continue
                # Interactive use wins. A task deferred by a few seconds costs
                # nobody anything; a task that halves your chat throughput does.
                if self.chat_is_busy():
                    continue
                for task in self.db.due_tasks():
                    if task["id"] in self.by_task:
                        continue
                    # Re-arm before running: a task whose run outlives its own
                    # interval must not stack up a backlog of overdue firings.
                    self.db.schedule_next(task["id"], int(task["interval_seconds"] or 0))
                    await self.launch(task, trigger="schedule")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log(f"Task scheduler error: {exc}", logging.ERROR)
                await asyncio.sleep(5)

    # ---------------------------------------------------------------- runs --

    async def launch(self, task: dict, trigger: str = "manual") -> TaskRun:
        if task["id"] in self.by_task:
            raise ValueError("this task is already running")
        run_id = self.db.create_run(task["id"], trigger, self.config.model)
        run = TaskRun(run_id, task["id"], task["name"])
        self.active[run_id] = run
        self.by_task[task["id"]] = run
        runner = asyncio.create_task(self._execute(task, run, trigger))
        self._runners.add(runner)
        runner.add_done_callback(self._runners.discard)
        return run

    def _task_config(self, task: dict) -> Config:
        """A config copy scoped to one task, so its tools and step budget are its own."""
        return dataclass_replace(
            self.config,
            model=task.get("model") or self.config.model,
            system_prompt=task.get("system_prompt") or self.config.system_prompt,
            agent_tools=task.get("tools") or self.config.agent_tools,
            agent_max_steps=int(task.get("max_steps") or self.config.agent_max_steps),
        )

    async def _swap_for_task(self, task: dict) -> str | None:
        """Load a task's own model, returning the one to restore afterwards.

        Latency tolerance differs by workload. Interactive chat wants a small
        model that answers now; a 3am research task has nobody waiting and can
        afford a bigger one, or a reasoning model whose chain of thought would
        be intolerable in a chat box. On 8GB you cannot hold both, so swap.
        """
        wanted = (task.get("model") or "").strip()
        if not wanted or wanted == self.model_manager.model_id:
            return None
        previous = self.model_manager.model_id
        log(f"Task {task['name']}: swapping {previous} -> {wanted}")
        self.model_manager.swap(wanted)
        await asyncio.to_thread(self.model_manager.restart)
        if self.model_manager.status != "ready":
            self.model_manager.swap(previous)
            await asyncio.to_thread(self.model_manager.restart)
            raise RuntimeError(f"could not load {wanted} for this task")
        return previous

    async def _restore_model(self, previous: str | None) -> None:
        if not previous or previous == self.model_manager.model_id:
            return
        log(f"Restoring model {previous} after task run")
        self.model_manager.swap(previous)
        try:
            await asyncio.to_thread(self.model_manager.restart)
        except Exception as exc:
            log(f"Could not restore {previous}: {exc}", logging.ERROR)

    async def _execute(self, task: dict, run: TaskRun, trigger: str) -> None:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_tasks))
        conversation_id = f"task:{task['id']}"
        answer = ""
        error: str | None = None
        steps = 0
        tools_used: list[str] = []
        status = "ok"
        previous_model: str | None = None

        try:
            async with self._semaphore:
                if run.cancel.is_set():
                    raise asyncio.CancelledError
                previous_model = await self._swap_for_task(task)
                self._emit(run, {"type": "start", "task": task["name"], "trigger": trigger,
                                 "model": task.get("model") or self.config.model,
                                 "swapped": previous_model is not None})
                append_task_log(f"run {run.run_id} start: {task['name']} ({trigger})")

                task_config = self._task_config(task)
                registry = ToolRegistry(task_config, self.db)
                agent = Agent(task_config, registry,
                              ModelClient(task_config, cluster=self.cluster))

                # Attribute this run to the task's owner so RAG retrieval, memory
                # writes and knowledge indexing scope to that user, not 'local'.
                owner = task.get("user_id") or SENTINEL_LOCAL_USER
                set_acting_user(owner)

                history: list[dict] = []
                if task.get("use_history"):
                    rows = self.db.get_messages(conversation_id,
                                                limit=self.config.history_turns * 2,
                                                user_id=owner)
                    history = [{"role": r["role"], "content": r["content"]} for r in rows]

                async for event in agent.run(
                    task["goal"], history, conversation_id, cancel=run.cancel
                ):
                    self._emit(run, event)
                    if event["type"] == "final":
                        answer = event["answer"]
                        steps = event.get("steps", 0)
                        tools_used = event.get("tools_used", [])
                    elif event["type"] == "error":
                        error = event["error"]
                        status = "error"
                    elif event["type"] == "cancelled":
                        status = "cancelled"

                if task.get("use_history") and answer:
                    self.db.add_message(conversation_id, "user", task["goal"], user_id=owner)
                    self.db.add_message(conversation_id, "assistant", answer, user_id=owner)

        except asyncio.CancelledError:
            status = "cancelled"
            self._emit(run, {"type": "cancelled", "reason": "shutdown or cancel request"})
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            log(f"Task {task['name']} failed: {error}", logging.ERROR)
            self._emit(run, {"type": "error", "error": error})
        finally:
            try:
                await self._restore_model(previous_model)
            except Exception as exc:
                log(f"Model restore failed: {exc}", logging.ERROR)
            elapsed = (time.time() - run.started) * 1000
            run.status = status
            run.answer = answer
            self.db.finish_run(run.run_id, status, answer, error, steps, elapsed, tools_used)
            self.db.prune_runs(task["id"], keep=25)
            append_task_log(
                f"run {run.run_id} {status} in {elapsed:.0f}ms"
                + (f": {error}" if error else f": {answer[:120]}")
            )
            self._emit(run, {"type": "done", "status": status, "answer": answer,
                             "elapsed_ms": round(elapsed), "error": error})
            run.close()
            self.active.pop(run.run_id, None)
            if self.by_task.get(task["id"]) is run:
                self.by_task.pop(task["id"], None)
            self.recent[run.run_id] = run
            while len(self.recent) > 20:
                self.recent.pop(next(iter(self.recent)))

            # Chaining. Three narrow tasks passing state through the workspace
            # beat one task with a compound goal at this model size, so the
            # dependency is first class rather than something you fake with two
            # schedules and a file.
            if status == "ok" and task.get("next_task_id") and not self._stopping:
                await self._chain(task, answer)

    async def _chain(self, task: dict, answer: str) -> None:
        follow_on = self.db.get_task(task["next_task_id"])
        if follow_on is None:
            log(f"Task {task['name']} points at a missing next task.", logging.WARNING)
            return
        if follow_on["id"] in self.by_task:
            log(f"Chained task {follow_on['name']} is already running; skipping.", logging.WARNING)
            return
        if follow_on["id"] == task["id"]:
            log("Refusing to chain a task to itself.", logging.WARNING)
            return
        # Hand the upstream answer over on disk rather than in the goal text, so
        # a long result does not become a giant prompt for the next task.
        try:
            handoff = resolve_in_workspace(f"chain/{task['id']}.txt")
            handoff.parent.mkdir(parents=True, exist_ok=True)
            handoff.write_text(answer, encoding="utf-8")
            relative = handoff.relative_to(WORKSPACE_DIR.resolve())
        except Exception as exc:
            log(f"Could not write the chain handoff: {exc}", logging.WARNING)
            return
        chained = dict(follow_on)
        chained["goal"] = (
            f"{follow_on['goal']}\n\n"
            f"The previous task ({task['name']}) wrote its result to {relative}. "
            "Read that file first with read_file."
        )
        log(f"Chaining {task['name']} -> {follow_on['name']}")
        await self.launch(chained, trigger=f"chain:{task['id']}")

    def _emit(self, run: TaskRun, event: dict) -> None:
        stamped = run.publish(event)
        if event["type"] in self.PERSISTED:
            payload = {k: v for k, v in stamped.items() if k not in ("run_id", "task_id")}
            try:
                self.db.append_event(run.run_id, stamped["seq"], event["type"], payload)
            except Exception as exc:
                log(f"Could not persist task event: {exc}", logging.WARNING)

    # -------------------------------------------------------------- control --

    def cancel_task(self, task_id: str) -> bool:
        run = self.by_task.get(task_id)
        if run is None:
            return False
        run.cancel.set()
        return True

    def cancel_run(self, run_id: str) -> bool:
        run = self.active.get(run_id)
        if run is None:
            return False
        run.cancel.set()
        return True

    def live_status(self, task_id: str) -> dict | None:
        run = self.by_task.get(task_id)
        if run is None:
            return None
        last_step = next((e for e in reversed(run.events) if e["type"] == "step"), None)
        last_tool = next((e for e in reversed(run.events) if e["type"] == "tool_call"), None)
        return {
            "run_id": run.run_id,
            "step": (last_step or {}).get("step", 0),
            "max_steps": (last_step or {}).get("max_steps", 0),
            "tool": (last_tool or {}).get("name"),
            "elapsed_ms": round((time.time() - run.started) * 1000),
        }

    async def subscribe(self, run_id: str) -> AsyncGenerator[dict, None]:
        """Replay a run then follow it live. Falls back to the database when finished."""
        run = self.active.get(run_id) or self.recent.get(run_id)
        if run is None:
            for event in self.db.run_events(run_id, limit=2000):
                yield event
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        # Subscribe before snapshotting, then drop anything the snapshot already
        # covered. The other order loses events published in between.
        run.subscribers.add(queue)
        try:
            backlog = list(run.events)
            highest = backlog[-1]["seq"] if backlog else 0
            for event in backlog:
                yield event
            if run.done:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                if event["seq"] <= highest:
                    continue
                yield event
        finally:
            run.subscribers.discard(queue)


def append_task_log(line: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / LOG_FILES["tasks"], "a", encoding="utf-8") as handle:
            handle.write(f"[{iso(utc_now())}] {line}\n")
    except OSError:
        pass          # a task log write must never fail the task itself



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'TaskManager',
    'TaskRun',
    'append_task_log',
]
