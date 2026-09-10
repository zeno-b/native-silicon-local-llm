"""The ReAct-style agent loop: routing, tools, memory, verification.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, AsyncGenerator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .taskstate import *  # noqa: F401,F403


# Tokens held back for conversation history when sizing the retrieval block.
# Without a floor, retrieval can spend everything trim_to_context was going to
# give the last exchange, and a follow-up loses the turn it refers to.
RAG_HISTORY_FLOOR = 256


class Agent:
    """A ReAct-style loop over the local model.

    The protocol is plain JSON in the message body rather than the OpenAI tools
    field, because mlx_lm.server support for native tool calling varies by
    version and small local models emit malformed tool_calls more often than
    they emit malformed JSON text.
    """

    def __init__(self, config: Config, registry: ToolRegistry, client: ModelClient):
        self.config = config
        self.registry = registry
        self.client = client
        # Open partial-output file handles, keyed by conversation. Reused across a
        # turn's many appends so we do not pay an open()/close() per write. An LRU
        # cap keeps the fd count bounded on a long-running server.
        self._partials: dict[str, dict] = {}
        self._partials_dir = DATA_DIR / "partials"
        # Every in-flight subagent holds its own KV cache in the same unified
        # memory as the weights, so this is a memory limit as much as a
        # concurrency one. Created lazily: an asyncio.Semaphore binds to the
        # loop that first awaits it, and this process starts a fresh loop per
        # test app.
        self._subagent_sem: asyncio.Semaphore | None = None
        self._subagent_loop: Any = None

    async def forced_prose_answer(self, history: list[dict], user_message: str,
                                  reserve: int, temperature: float | None,
                                  task: "TaskState | None" = None,
                                  stats: "GenerationStats | None" = None) -> str:
        """One last tool-free pass, on the prose prompt.

        Used when a step produced nothing usable -- an empty reply, or a tool
        call the model never filled in. The alternative was returning "" and
        rendering a blank assistant turn.
        """
        # The forced pass is a last resort, which is exactly when the active
        # task matters most: without the brief this lane answered from the bare
        # follow-up ("add more checks") and had no idea what it was adding to.
        brief = self.task_brief(task, reserve)
        base, _ = self.build_base(history, user_message, reserve, tools=False,
                                  extra_context=brief,
                                  active_code_task=bool(brief and task
                                                        and task.is_code_task()))
        nudge = [{"role": "user", "content":
                  "Answer the question above directly, in plain text. Do not output "
                  "JSON and do not call any tool."}]
        messages, _ = self.assemble(base, nudge, reserve)
        self.trace_prompt("forced-prose", messages, reply_budget=reserve,
                          task=(task.summary() if task is not None else ""))
        text = strip_reasoning(
            await self.resilient_complete(messages, reserve, temperature, stats)).strip()
        return "" if is_degenerate_tool_call(text) else text

    # A tool name the model wrote that is really the tool's TARGET. Small models
    # do this constantly: {"tool": "https://github.com/libusb/libusb"} instead of
    # {"tool": "fetch_url", "args": {"url": "..."}}.
    _URLISH = re.compile(r"(?i)^(?:https?://|www\.)\S+$")
    # Wrappers some models put around a real name: functions.web_search,
    # `read_file`, "web_search()", tools/list_files.
    _NAME_NOISE = re.compile(r"(?i)^[`'\"\s]*(?:functions?|tools?)?[./]?([a-z_][a-z0-9_]*)"
                             r"\s*(?:\(\s*\))?[`'\"\s]*$")

    def repair_tool_call(self, name: str, args: dict, known: set[str]) -> tuple[str, dict] | None:
        """Map an invalid tool name onto the tool the model obviously meant.

        parse_tool_call lets an EXPLICIT {"tool": ...} through even when the name
        is not registered, so that a JSON reply which is clearly a call attempt
        is not mistaken for prose. That is right for classification and wrong for
        dispatch: handing the name to the registry spends a step, logs a failed
        tool call, and feeds "Unknown tool: <name>. Available: <19 names>" back
        into a 4096-token context as a TOOL RESULT -- which then risks being
        harvested as a training trace, teaching the model that URLs are tool
        names. Repair what is unambiguous, reject the rest without dispatching.

        Returns the corrected (name, args), or None when there is no honest
        repair.
        """
        if name in known:
            return name, args
        # A URL in the name slot means fetch_url with that URL. Nothing else.
        if self._URLISH.match(name) and "fetch_url" in known:
            merged = dict(args)
            merged.setdefault("url", name)
            return "fetch_url", merged
        # Strip a namespace or call syntax off an otherwise real name.
        match = self._NAME_NOISE.match(name)
        if match and match.group(1) in known:
            return match.group(1), args
        return None

    # --- The closed learning loop ----------------------------------------- #
    # Ratings arrive minutes after a turn, from a different request, which is
    # why this is a method taking plain strings rather than something wired into
    # run(): the caller (the feedback endpoint) reconstructs the turn and calls
    # in. Everything here is best-effort and must never raise into a request.

    SKILL_AUTHOR_PROMPT = (
        "You are writing a short reusable SKILL document for yourself, so that "
        "next time a task like this one takes fewer steps and fewer mistakes.\n\n"
        "WHAT THE USER ASKED:\n{prompt}\n\n"
        "THE ANSWER THAT WORKED:\n{answer}\n\n"
        "Reply with ONLY a JSON object and nothing else:\n"
        '{{"name": "kebab-case-name", "description": "one line saying when to use '
        'this", "body": "the procedure as numbered markdown steps, under 150 words"}}\n\n'
        "The body must be the GENERAL procedure for this KIND of task, not a copy "
        "of this one answer, and not a summary of what you just said. If there is "
        "no general procedure worth writing down, reply with exactly: NONE"
    )

    SKILL_LESSON_PROMPT = (
        "A procedure you followed produced an answer the user had to correct.\n\n"
        "PROCEDURE: {name} - {description}\n"
        "WHAT THE USER ASKED:\n{prompt}\n\n"
        "THE USER'S CORRECTION:\n{correction}\n\n"
        "In ONE short sentence, state the lesson to add to that procedure so the "
        "same mistake is not repeated. Reply with the sentence only: no preamble, "
        "no JSON, no quotes."
    )

    async def learn_from_turn(
        self,
        user_prompt: str,
        answer: str,
        correction: str = "",
        used: "list[str] | tuple[str, ...]" = (),
    ) -> str | None:
        """Turn one rated turn into procedural memory. Returns a short note.

        Two cases, one model call at most:

        - A skill was used and the user had to CORRECT the answer. The procedure
          led the model astray, so distil the correction into a single lesson
          and append it. Appending rather than rewriting is deliberate: asking a
          3B model to regenerate a working document from memory is how a good
          skill gets worse.
        - No skill was used and the answer was rated good. There is a procedure
          here that was worked out from scratch; write it down so the next
          equivalent task starts from it. If something close already exists,
          refine that instead of growing a near-duplicate library.
        """
        if not (self.config.skills_enabled and self.config.skills_autolearn):
            return None
        library = getattr(self.registry, "skills", None)
        if library is None:
            return None
        prompt = (user_prompt or "").strip()
        answer = (answer or "").strip()
        # Nothing to learn from chit-chat, and nothing to learn from an answer
        # too short to contain a procedure.
        if not is_substantive(prompt) or len(answer) < 200:
            return None
        try:
            if used and correction.strip():
                return await self._teach_correction(
                    library, list(used), prompt, correction.strip())
            if not used:
                return await self._author_skill(library, prompt, answer)
        except Exception as exc:      # never break a rating on a learning failure
            log(f"Skill learning failed: {exc}", logging.WARNING)
        return None

    async def _teach_correction(self, library, used: list[str],
                                prompt: str, correction: str) -> str | None:
        """Append the lesson from a correction to the skill that misled."""
        skill = library.get(used[0])
        if skill is None:
            return None
        lesson = await self.client.complete(
            [{"role": "system", "content": self.config.system_prompt_with_identity},
             {"role": "user", "content": self.SKILL_LESSON_PROMPT.format(
                 name=skill.name, description=skill.description,
                 prompt=prompt[:600], correction=correction[:800])}],
            max_tokens=96, temperature=0.0,
        )
        lesson = strip_reasoning(lesson or "").strip().strip('"').split("\n")[0]
        if not lesson or len(lesson) < 12:
            return None
        updated = library.refine(skill.name, lesson)
        if updated is None:
            return None
        log(f"Skill {skill.name!r} refined to v{updated.version} from a correction.")
        return f"refined skill {skill.name} (v{updated.version})"

    async def _author_skill(self, library, prompt: str, answer: str) -> str | None:
        """Write a new skill, or refine a close existing one."""
        # A near-duplicate is worse than nothing: it splits the track record and
        # doubles the catalogue cost for the same procedure.
        nearby = library.search(prompt, limit=1)
        if nearby:
            existing = nearby[0]
            overlap = len(set(re.findall(r"[a-z0-9]+", prompt.lower()))
                          & set(re.findall(r"[a-z0-9]+", existing.description.lower())))
            if overlap >= 3:
                log(f"Not authoring a skill: {existing.name!r} already covers this.",
                    logging.DEBUG)
                return None
        raw = await self.client.complete(
            [{"role": "system", "content": self.config.system_prompt_with_identity},
             {"role": "user", "content": self.SKILL_AUTHOR_PROMPT.format(
                 prompt=prompt[:800], answer=answer[:2000])}],
            max_tokens=self.config.skill_author_tokens, temperature=0.0,
        )
        text = strip_reasoning(raw or "").strip()
        if not text or text.upper().startswith("NONE"):
            return None
        parsed = extract_json_object(text) or {}
        name = str(parsed.get("name") or "").strip()
        description = str(parsed.get("description") or "").strip()
        body = str(parsed.get("body") or "").strip()
        # A body shorter than this is a description, not a procedure, and would
        # cost catalogue space forever while teaching the model nothing.
        if not name or not description or len(body) < 80:
            log(f"Discarded an unusable authored skill: {text[:120]!r}", logging.DEBUG)
            return None
        skill = library.save(name, description, body)
        if skill is None:
            return None
        log(f"Authored skill {skill.name!r} (v{skill.version}) from a rated turn.")
        return f"learned skill {skill.name} (v{skill.version})"

    def record_skill_outcomes(self, used: "list[str] | tuple[str, ...]",
                              ok: bool) -> list[str]:
        """Credit or blame the procedures behind a rated answer, and retire the

        ones with a real track record of doing harm. Synchronous and model-free,
        so it can run on the request thread. Returns names retired.
        """
        library = getattr(self.registry, "skills", None)
        if library is None or not self.config.skills_enabled:
            return []
        retired: list[str] = []
        for name in used:
            skill = library.record_outcome(name, ok)
            if skill is None or ok:
                continue
            if library.should_retire(skill, self.config.skill_retire_min_rated,
                                     self.config.skill_retire_loss_rate):
                reason = (f"{skill.losses} of {skill.wins + skill.losses} rated uses "
                          "went badly")
                if library.retire(skill.name, reason):
                    retired.append(skill.name)
        return retired

    @property
    def subagents(self) -> "asyncio.Semaphore":
        """Admission control for child agents, rebound per event loop."""
        loop = asyncio.get_running_loop()
        if self._subagent_sem is None or self._subagent_loop is not loop:
            self._subagent_sem = asyncio.Semaphore(
                max(1, self.config.subagent_max_concurrent))
            self._subagent_loop = loop
        return self._subagent_sem

    async def run_subagent(self, task: str, capabilities: str = "",
                           parent_conversation: str | None = None) -> str:
        """Run one task in a fresh agent with its own context. Returns its answer.

        The value on this hardware is CONTEXT ISOLATION, not parallelism: the
        parent pays for the child's conclusion instead of for the twelve files
        the child had to read to reach it. Which is also why the result is
        capped -- a subagent that returns everything it saw has saved nothing.

        The child cannot delegate again. Recursion is prevented by construction
        (delegate_task is removed from its allowlist) rather than by a depth
        counter, because a depth counter still pays for the generation that
        discovers it has run out of depth.
        """
        allowed = [name for name in tools_for_capabilities(
            [part.strip() for part in (capabilities or "").split(",") if part.strip()])
            if name != "delegate_task"]
        child_config = dataclass_replace(
            self.config,
            agent_tools=",".join(allowed),
            agent_max_steps=min(self.config.agent_max_steps,
                                self.config.subagent_max_steps),
            # A child that hands work to yet another child, or writes skills of
            # its own from an unrated turn, is not what was asked for.
            delegation_enabled=False,
            skills_autolearn=False,
            # Its own history is empty by construction, and re-running the
            # parent's routing on a task the parent already scoped is a wasted
            # generation on a 3B model.
            knowledge_triage=False,
            incremental_reasoning=False,
            # The child shares the parent's conversation id (so its tool calls
            # are attributed there) but it is doing a scoped sub-task, not the
            # user's task. Letting it load and then overwrite the parent's task
            # state would replace the artifact under construction with the
            # helper's fragment.
            task_state_enabled=False,
        )
        child_registry = ToolRegistry(child_config, self.registry.db)
        child = Agent(child_config, child_registry, self.client)
        answer = ""
        async for event in child.run(task, [], conversation_id=parent_conversation,
                                     max_tokens=self.config.max_tokens):
            if event.get("type") == "final":
                answer = event.get("answer") or ""
        # Unwrapped before it reaches the parent: a helper's "Final answer:"
        # label becomes part of the parent's prompt otherwise, which is how the
        # parent learns to write one too.
        return unwrap_answer(answer)[:self.config.subagent_result_chars]

    # --- Active task state ------------------------------------------------ #
    # The task is a first-class object (taskstate.TaskState), not something the
    # model is expected to remember from a transcript that gets trimmed. Loaded
    # at the top of a turn, put in front of the model as a block that trimming
    # cannot reach, and saved at the end with whatever artifact the turn
    # produced.

    def load_task(self, conversation_id: str | None,
                  history: list[dict] | None) -> TaskState:
        """The active task: the persisted record merged with the visible history."""
        if not self.config.task_state_enabled:
            return TaskState()
        stored = None
        db = getattr(self.registry, "db", None)
        if db is not None and conversation_id:
            try:
                stored = db.load_task_state(conversation_id, get_acting_user())
            except Exception as exc:
                log(f"task state could not be read: {exc}", logging.DEBUG)
        return load_task_state(stored, history)

    def save_task(self, conversation_id: str | None, task: TaskState) -> None:
        """Persist the task. Best-effort: never fail a request over bookkeeping."""
        if not self.config.task_state_enabled or not conversation_id:
            return
        db = getattr(self.registry, "db", None)
        if db is None or not task.is_active():
            return
        try:
            db.save_task_state(conversation_id, task.to_dict(), get_acting_user())
        except Exception as exc:
            log(f"task state could not be saved: {exc}", logging.DEBUG)

    def clear_task(self, conversation_id: str | None) -> None:
        """Forget the stored task for this conversation. Best-effort."""
        db = getattr(self.registry, "db", None)
        if db is None or not conversation_id:
            return
        try:
            db.clear_task_state(conversation_id)
        except Exception as exc:
            log(f"task state could not be cleared: {exc}", logging.DEBUG)

    def reply_reserve(self, message: str, requested: int,
                      task: "TaskState | None" = None) -> int:
        """The reply budget this turn deserves, widened for code.

        A program does not fit in the default 512-token budget: it gets cut off
        mid-function. Factored out of run() because the continuation path needs
        the budget of the ORIGINAL request, not of the word "continue".

        The active task is what makes this work across turns. "add more checks
        and error handling" names no language and no code object, so
        is_code_request says no and the turn got the 512-token chat budget --
        for a request to re-emit a whole shell script. When the task says we are
        writing code, a follow-up on it is a code turn, and a turn that has to
        reproduce an existing artifact gets at least what that artifact costs.
        """
        code = is_code_request(message)
        if not code and task is not None and task.is_code_task():
            code = bool(is_modification_request(message)
                        or is_continue_request(message)
                        or parse_correction(message))
        if not code:
            return requested
        wanted = max(requested, self.config.code_max_tokens)
        if task is not None and task.artifact:
            wanted = max(wanted, estimate_tokens(task.artifact)
                         + self.config.artifact_reply_headroom)
        ceiling = max(self.config.min_max_tokens,
                      self.config.context_size - CONTEXT_SAFETY_MARGIN - 768)
        return max(1, min(wanted, ceiling))

    def artifact_chars_for(self, reserve: int) -> int:
        """Characters of the artifact allowed into the prompt for this budget.

        prompt + reply + margin has to fit the context window, so the artifact
        block is sized against what is actually left rather than a fixed number:
        on a 4096-token window a 1536-token reply leaves the artifact about
        1150 tokens, and the rest goes to the system prompt, the brief and
        recent conversation. Past that the brief truncates the middle and says
        so, which is honest; letting it overflow is not.
        """
        room = max(0, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        return int(min(self.config.task_artifact_chars, (room // 2) * CHARS_PER_TOKEN))

    def task_brief(self, task: "TaskState | None", reserve: int) -> str:
        """The ACTIVE TASK block for this turn, or "" when there is no task."""
        if not self.config.task_state_enabled or task is None or not task.is_active():
            return ""
        allowed = self.artifact_chars_for(reserve)
        if task.artifact and len(task.artifact) > allowed:
            log(f"active artifact is {len(task.artifact)} chars but only {allowed} "
                f"fit alongside a {reserve}-token reply in a "
                f"{self.config.context_size}-token context; the middle is omitted "
                "and the model is told to patch rather than rewrite.", logging.INFO)
        verify = bool(self.config.allow_shell and self.registry.get("run_shell"))
        return task.brief(artifact_chars=allowed, verify_hint=verify)

    def trace_prompt(self, where: str, messages: list[dict], **fields: Any) -> None:
        """Log the prompt that is actually about to be sent, behind debug_prompts.

        The logs already showed token counts and trimming decisions, which says
        nothing about what the model semantically received -- the whole reason
        the drift was invisible. Content goes through content_for_log, so it
        honours log_chat_content and is redacted; with content logging disabled
        this still records the shape (lanes, budgets, task, artifact version).
        """
        if not self.config.debug_prompts:
            return
        # content_for_log with no override, so the operator's log_chat_content
        # decides: "disabled" logs nothing of the text, "metadata" logs its
        # length and fingerprint, "full" logs it redacted and truncated. Turning
        # debug_prompts on must not silently start writing conversation text.
        body = content_for_log(
            "\n\n".join(f"[{m.get('role')}]\n{m.get('content', '')}"
                        for m in messages))
        log_event(get_logger("prompt"), logging.DEBUG, "prompt.assembled",
                  where=where, messages=len(messages),
                  prompt_tokens=messages_tokens(messages), prompt=body, **fields)

    DRIFT_DIRECTIVE = (
        "Your previous attempt at this ignored the active task: {problem}. That "
        "output is discarded. Do it again for the task described above: same "
        "language, same artifact, no new files, no unrelated classes or "
        "frameworks. Reply with the artifact only."
    )

    async def redirect_drift(self, task: "TaskState | None", answer: str,
                             user_message: str, history: list[dict], reserve: int,
                             temperature: float | None,
                             stats: "GenerationStats | None" = None) -> tuple[str, str]:
        """Catch an answer that contradicts the active task. Returns (answer, note).

        Narrow by design (see taskstate.detect_drift): it fires on a strong
        contradiction only -- every code block in the reply is a language the
        task is not. On a hit, one corrective regeneration with the task brief
        and an explicit "that was wrong, here is the task" directive. If that
        drifts too, the turn asks rather than silently handing back unrelated
        output, because two drifts in a row usually means the request itself was
        ambiguous.
        """
        if not self.config.drift_check_enabled or task is None or not answer:
            return answer, ""
        problem = detect_drift(task, answer)
        if not problem:
            return answer, ""
        log(f"task drift: {problem}; regenerating with the task brief.", logging.WARNING)
        brief = self.task_brief(task, reserve)
        directive = self.DRIFT_DIRECTIVE.format(problem=problem)
        base, _ = self.build_base(history, user_message, reserve, tools=False,
                                  extra_context=f"{brief}\n\n{directive}",
                                  active_code_task=True)
        messages, _ = self.assemble(base, [], reserve)
        self.trace_prompt("drift-retry", messages, task=task.summary(),
                          reply_budget=reserve, problem=problem)
        retry = strip_reasoning(
            await self.resilient_complete(messages, reserve, temperature, stats)).strip()
        if retry and not detect_drift(task, retry):
            return retry, (f"that reply drifted ({problem}); regenerated it as "
                           f"{task.language}")
        # Still wrong, or nothing came back. Say so and ask, keeping the text so
        # nothing is thrown away silently.
        kept = retry or answer
        question = (
            f"Before I go further: {problem}. The task I have recorded is "
            f"{task.language} — {task.artifact_name or 'the current artifact'}"
            + (f" on {task.platform}" if task.platform else "")
            + ". Do you want me to keep working on that, or switch? "
              "Here is what I produced, in case it is what you wanted:\n\n" + kept)
        return question, f"could not stay on task ({problem}); asking you instead"

    @staticmethod
    def continuation_target(history: list[dict] | None,
                            task: "TaskState | None" = None) -> tuple[str, str] | None:
        """(original request, partial answer) when the last answer was cut off.

        The truncation note the UI appended is the durable evidence in the
        transcript that an answer stopped at the token budget rather than at its
        natural end. The task state is the second source, and the one that
        survives trimming: it records the artifact and whether it was complete,
        so a "continue" whose partial answer has already been evicted from the
        visible history still resumes the right thing instead of being answered
        as a new question.
        """
        turns = list(history or [])
        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            if turn.get("role") != "assistant":
                continue
            content = str(turn.get("content") or "")
            if not was_truncated(content):
                return None          # newest answer finished; nothing to resume
            partial = strip_truncation_note(content)
            if not partial.strip():
                return None
            # Walk back past any earlier "continue" turns: the request that
            # matters is the one that started the answer, and a second
            # continuation would otherwise resume with "continue" as its prompt
            # and lose the task entirely.
            request = ""
            for earlier in range(index - 1, -1, -1):
                if turns[earlier].get("role") != "user":
                    continue
                candidate = str(turns[earlier].get("content") or "").strip()
                if not candidate or is_continue_request(candidate):
                    continue
                request = candidate
                break
            # The recorded objective is a better anchor than a bare "add more
            # checks now", which is what the walk-back finds after a few
            # modification turns and which says nothing about the task.
            if task is not None and task.objective and (
                    not request or len(request) < 24):
                request = f"{task.objective}\n\nMost recent request: {request}" \
                    if request else task.objective
            return request, partial
        # Nothing in the visible history, but the state remembers an artifact
        # that was cut off: resume that.
        if task is not None and task.artifact and not task.artifact_complete:
            return task.objective, task.artifact
        return None

    # How much of a trailing incomplete line is worth discarding to give the
    # model a clean boundary to resume from. Bounded so trimming can never throw
    # away a whole paragraph of prose.
    MAX_DISCARDED_FRAGMENT = 160

    @classmethod
    def clean_boundary(cls, partial: str) -> str:
        """Trim a trailing half-written line so the continuation starts cleanly.

        The budget cuts mid-token: "r = libusb_get_device". Asked to resume from
        there, the model does not finish the identifier -- it starts the
        statement again on a new line, leaving the fragment stranded above. Ending
        the partial at the last complete line instead costs a few tokens to
        regenerate and removes the whole class of seam artefact.
        """
        index = partial.rfind("\n")
        if index <= 0:
            return partial
        fragment = partial[index + 1:]
        if not fragment.strip():
            return partial          # already ends on a line break
        if len(fragment) > cls.MAX_DISCARDED_FRAGMENT:
            return partial          # too much to throw away; resume mid-line
        # A line that already closes a statement or block is a fine boundary.
        if fragment.rstrip().endswith((";", "{", "}", ":", ",", ")", "*/", ">", ".")):
            return partial
        return partial[:index]

    @staticmethod
    def stitch_continuation(partial: str, extra: str) -> str:
        """Join a partial answer to its continuation without duplicating the seam.

        Even told not to, a small model usually re-emits the last line or two,
        and when it was cut off inside a fenced code block it either re-opens the
        fence or immediately CLOSES it -- which strands the rest of the program
        outside the block and leaves the answer with an odd number of fences.
        All of that is cheap to detect here and impossible to fix by asking more
        politely.
        """
        extra = extra or ""
        # Inside an unclosed fence, a fence at the very START of the continuation
        # is wrong either way: an opening one duplicates the block, a closing one
        # ends it mid-statement. The block can only legitimately close at the end.
        if partial.count("```") % 2 == 1:
            extra = re.sub(r"^\s*```[A-Za-z0-9+#.-]*[ \t]*\n?", "", extra, count=1)
        # The longest suffix of the partial that the continuation repeats.
        window = partial[-400:]
        for size in range(len(window), 15, -1):
            if extra.startswith(window[-size:]):
                extra = extra[size:]
                break
        if not extra.strip():
            return partial
        # No separator: a continuation resumes mid-line by design.
        return partial + extra

    CONTINUE_INSTRUCTION = (
        "Continue your previous answer from exactly where it stops. It ended with:\n\n"
        "<<<{tail}>>>\n\n"
        "Resume immediately after that, mid-line if necessary. Do NOT repeat any of "
        "it, do NOT start over, do NOT re-introduce the topic and do NOT add a "
        "preamble. If it is code, stay inside the same code block and finish the "
        "program."
    )

    def continuation_contract(self, task: "TaskState | None", request: str,
                              partial: str) -> str:
        """What the model is resuming, stated explicitly.

        Built from the task state where there is one, and from the partial text
        itself where there is not (a conversation that predates the state, or a
        prose answer): the language of the code it was writing is recoverable
        from the code, and saying it beats hoping.
        """
        language = ""
        kind = "prose"
        name = ""
        platform = ""
        if task is not None and task.is_active():
            language, name, platform = task.language, task.artifact_name, task.platform
            kind = "code" if task.is_code_task() else "prose"
        if not language:
            blocks = extract_code_blocks(partial)
            if blocks:
                info, code = blocks[-1]
                language = canonical_language(info) or code_language(code)
                kind = "code" if language else kind
            elif is_code_request(request):
                language = detect_language(request)
                kind = "code" if language else kind
        if kind != "code" and not language:
            return ""
        lines = ["YOU ARE RESUMING ONE UNFINISHED ANSWER, NOT STARTING A NEW ONE."]
        if language:
            lines.append(f"It is {language}" + (f" for {platform}" if platform else "")
                         + (f", the file {name}" if name else "") + ".")
        lines.append("Continue that exact artifact. Do not change topic, language or "
                     "file, do not restart it, and do not describe a different "
                     "design or architecture.")
        if partial.count("```") % 2 == 1:
            lines.append("The code block is still open: stay inside it and close it "
                         "only when the artifact is finished.")
        return "\n".join(lines)

    async def continue_answer(
        self,
        resume: tuple[str, str],
        history: list[dict],
        requested: int,
        temperature: float | None,
        cancel: "asyncio.Event | None" = None,
        conversation_id: str | None = None,
        task: "TaskState | None" = None,
    ):
        """Resume a truncated answer instead of answering "continue" as a question.

        Skips routing entirely: the lane was already decided on the original
        request, and re-routing a bare "continue" is what sent it through the
        answer lane as a brand new prompt. The reply budget comes from the
        original request too, so a continued program gets the code budget rather
        than the 512-token chat default.
        """
        started = time.time()
        request, partial = resume
        reserve = self.reply_reserve(request, requested, task)
        yield {"type": "phase", "label": "continuing the previous answer"}
        detail_budget = reserve

        partial = self.clean_boundary(partial)
        tail = partial[-600:]
        system = build_plain_system_prompt(
            self.config.system_prompt_with_identity,
            reasoning=self.config.reasoning_visible)
        messages = [{"role": "system", "content": system}]
        if request:
            messages.append({"role": "user", "content": request})
        # The task contract, stated rather than inferred. "continuation lane:
        # resuming a 648-token partial answer" was true and still produced a
        # different topic, because the only thing the model had to work out what
        # it was resuming was the partial text itself: no language, no artifact,
        # no objective. This is the fix -- the contract is explicit, so the model
        # never has to rediscover the task from its own truncated output.
        contract = self.continuation_contract(task, request, partial)
        if contract:
            messages.append({"role": "user", "content": contract})
        # The partial goes in as the assistant turn it actually was, so the model
        # sees its own voice rather than a quoted transcript. Trimmed from the
        # FRONT if it is long: continuation needs the end, not the beginning.
        head_room = max(400, (self.config.context_size - reserve
                              - CONTEXT_SAFETY_MARGIN - estimate_tokens(system)
                              - estimate_tokens(request) - estimate_tokens(contract)
                              - 400)) * 4
        messages.append({"role": "assistant", "content": partial[-head_room:]})
        messages.append({"role": "user",
                         "content": self.CONTINUE_INSTRUCTION.format(tail=tail)})
        detail_budget = self.client.reply_budget(messages, reserve, quiet=True)
        if self.config.show_internals:
            yield {"type": "detail", "message":
                   f"continuation lane: resuming a {estimate_tokens(partial)}-token "
                   f"partial answer, reply budget {detail_budget}"
                   + (f" (asked for {reserve})" if detail_budget != reserve else "")
                   + f", {'code' if reserve != requested else 'chat'} budget"
                   + (f"; task {task.summary()}" if task is not None
                      and task.is_active() else "")}
        self.trace_prompt("continuation", messages, continuation=True,
                          reply_budget=detail_budget, requested_budget=reserve,
                          generation_id=(task.generation_id if task else ""),
                          artifact_version=(task.artifact_version if task else 0),
                          task=(task.summary() if task is not None else ""))

        stats = GenerationStats()
        extra = ""
        stream = self.client.stream(messages, reserve, temperature, stats, kind="code")
        try:
            async for token in stream:
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled", "step": 1, "partial": extra, "trace": []}
                    return
                extra += token
                yield {"type": "token", "token": token, "step": 1}
        except Exception as exc:
            log(f"continuation stream failed ({exc}); falling back to non-streaming.",
                logging.WARNING)
            extra = await self.resilient_complete(messages, reserve, temperature)
        finally:
            await stream.aclose()

        reasoned = strip_reasoning(extra)
        unwrapped = unwrap_answer(reasoned)
        if unwrapped != reasoned.strip():
            # The continuation opened with a label or a tag, and unwrap_answer
            # removed it. Welding "Final answer:" into the middle of a partial
            # program is worse than losing the seam whitespace with it.
            extra = unwrapped
        else:
            # rstrip only: LEADING whitespace is part of the seam. A
            # continuation that resumes on a new line, or after a space, carries
            # that in its first token, and stripping it welded "part one" onto
            # "and part two".
            extra = reasoned.rstrip()
        yield {"type": "usage", "step": 1, **stats.as_event()}
        if not extra:
            yield {"type": "notice", "info": True,
                   "message": "the model had nothing more to add"}
            answer = partial
        else:
            answer = self.stitch_continuation(partial, extra)
        # Mark it again when the continuation was itself cut off, so a further
        # "continue" still has the evidence it needs. This composes.
        if getattr(stats, "finish_reason", "") == "length":
            answer = answer.rstrip() + truncation_note(
                self.client.reply_budget(messages, reserve, quiet=True))
        self.partial_add(conversation_id, extra)
        if task is not None:
            # The stitched artifact is the new current one, and whether it is
            # still truncated is what a further "continue" depends on.
            task.note_answer(answer)
            task.artifact_complete = getattr(stats, "finish_reason", "") != "length"
            self.save_task(conversation_id, task)
        yield {
            "type": "final",
            "answer": answer,
            "steps": 1,
            "trace": [],
            "tools_used": [],
            "elapsed_ms": round((time.time() - started) * 1000),
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "truncated": getattr(stats, "finish_reason", "") == "length",
            "changed_files": [],
            "diff": "",
            "continued": True,
        }

    async def route(self, message: str, history: list[dict] | None = None) -> dict:
        """Ask the model how to handle a message, as one structured decision.

        The router menu is generated from the registry: every tool that declares
        route_hint becomes a selectable action. Adding a routable tool therefore
        needs no change here and no new regex lane, which is what keeps routing
        maintainable as the toolset grows over the years.

        Returns {"action": "answer"} or {"action": "<tool name>", ...tool args}.
        The model reads the message (with a little history so follow-ups resolve)
        and picks. On any parse failure it biases to a web search when one exists,
        so a genuine lookup is never silently answered from stale weights.
        """
        routable = self.registry.routable()
        if not routable:
            # Nothing to route to; the model answers everything itself.
            return {"action": "answer"}

        # The menu: an answer option plus one line per routable tool, taken
        # straight from each tool's route_hint.
        # When a real codebase is attached, "answer" must NOT claim to cover code:
        # a request to fix the user's own files cannot be satisfied from weights.
        has_project = bool(self.config.project_dir)
        if has_project:
            answer_option = ('{"action":"answer"} — the DEFAULT for general questions '
                             "you can answer from your own knowledge: facts, "
                             "explanations, definitions, writing, math, reasoning, and "
                             "code written from scratch that does NOT touch the user's "
                             "project. If the user refers to THEIR code, this project, "
                             "a file, a bug, or says 'fix this', do NOT answer — use a "
                             "file tool to look at the real files first.")
        else:
            answer_option = ('{"action":"answer"} — the DEFAULT. Use it whenever you '
                             "can answer from your own knowledge: general facts, "
                             "explanations, definitions, writing, math, reasoning, and "
                             "all code. Most questions are answer.")
        options = [answer_option] + [t.route_hint for t in routable]
        system = (
            "You are a router. Read the user's latest message and reply with "
            "exactly ONE JSON object and nothing else. Prefer answering from your "
            "own knowledge; only choose a lookup tool when the question truly "
            "needs current, real-time, or external facts you cannot be confident "
            "about (today's events, prices, scores, the latest version of "
            "something, or a specific named entity you do not know). If it is "
            "general knowledge you already know, choose answer. Options:\n"
            + "\n".join(f"- {opt}" for opt in options)
            + "\nFill the fields from the user's own words. Reply with only the "
            "JSON object."
        )

        # A few recent turns so "look it up" / "and tomorrow?" resolve in context.
        context: list[dict] = [{"role": "system", "content": system}]
        for turn in (history or [])[-4:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                context.append({"role": role, "content": str(turn["content"])[:500]})
        context.append({"role": "user", "content": message[:1000]})

        # If the reply is unusable, answer from own knowledge rather than
        # defaulting to a search. Prefer the model's knowledge unless it clearly
        # asked for a tool.
        fallback = {"action": "answer"}

        try:
            text, _ = await self.client.complete_with_stats(
                context, max_tokens=64, temperature=0.0
            )
        except Exception as exc:
            log(f"Router call failed ({exc}); falling back.", logging.WARNING)
            return fallback

        decision = extract_json_object(text) or {}
        action = str(decision.get("action", "")).lower().strip()

        if action == "answer":
            return {"action": "answer"}

        # A tool action: it must name a routable tool, and after alias-mapping its
        # required arguments must be present. Anything missing falls back safely.
        tool = self.registry.get(action)
        if tool is not None and tool.routable:
            args = {k: v for k, v in decision.items() if k != "action"}
            args = self.registry.normalise_args(tool, args)
            if all(r in args and str(args[r]).strip() for r in tool.required):
                # Trim over-long string args defensively.
                args = {k: (v[:200] if isinstance(v, str) else v) for k, v in args.items()}
                return {"action": action, **args}

        return fallback

    def with_retrieved_context(self, user_message: str, max_chars: int | None = None,
                               active_code_task: bool = False) -> str:
        """Prepend the most relevant knowledge-base passages to the question.

        This is the RAG step: when documents have been indexed, the best-matching
        passages are injected so the model answers from the user's own material
        (with source paths) instead of guessing. Silent no-op when the knowledge
        base is empty, so behaviour is unchanged until documents are added.

        `max_chars` caps the block to what is actually left of the context
        window (see build_base). Sizing it against the whole window instead is
        how a 4KB block of unrelated reference material pushed prompt + reply
        past n_ctx on a six-word question.
        """
        db = getattr(self.registry, "db", None)
        if not self.config.rag_enabled or db is None:
            return user_message
        if not getattr(db, "fts_enabled", False):
            return user_message
        if max_chars is not None and max_chars < 400:
            # Not enough window left for a passage worth reading.
            return user_message
        # Subject words only: intent verbs ("write", "program") match nearly
        # every indexed document and are not evidence of anything.
        tokens = rag_query_tokens(user_message)
        if not tokens:
            return user_message
        # "write a basic c++ crud program" is answered from the model's own
        # knowledge. Retrieval belongs on a code request only when the user is
        # pointing at their own files, where the real code is the answer.
        if is_code_request(user_message) and not refers_to_project(user_message):
            return user_message
        # Mid-task, a follow-up like "add more checks and error handling" is
        # about the artifact in the task state and nothing else. Retrieval on it
        # is actively harmful: the FTS query ORs every term, so "checks",
        # "handling" and "validator" matched indexed Python documents, and those
        # passages are where skill_validator.py and seo_health_scorer.py came
        # from in a conversation that was writing a Bash script.
        if active_code_task and not refers_to_project(user_message):
            return user_message
        # A remark about the previous answer ("doesnt have a main function") is
        # answered from the conversation, never from the knowledge base.
        if is_followup_remark(user_message):
            return user_message
        try:
            scope = [p.strip() for p in (self.config.rag_scope or "").split(",") if p.strip()]
            # Scope retrieval to the acting user's own + shared documents so one
            # user's imported history never surfaces in another's answers.
            hits = db.search_documents(user_message, limit=self.config.rag_passages,
                                       only=scope or None, user_id=get_acting_user())
        except Exception as exc:
            log(f"knowledge-base lookup skipped: {exc}", logging.DEBUG)
            return user_message
        # Keep only passages that genuinely overlap the question. The FTS query
        # ORs every term, so one common word ("message", "first") is enough to
        # match a totally unrelated document -- which is how "its not reconsider
        # the first message in this convo" pulled in someone else's transcript and
        # the model answered by inventing a conversation. Requiring real overlap
        # makes retrieval fire only when it has something to contribute.
        need = 2 if len(tokens) >= 2 else 1
        hits = [h for h in hits
                if sum(1 for t in tokens
                       if t in (str(h.get("chunk", "")) + " "
                                + str(h.get("path", ""))).lower()) >= need]
        if not hits:
            return user_message
        budget = max(400, int(self.config.context_size * 0.25) * 4)
        if max_chars is not None:
            budget = min(budget, max_chars)
        blocks, used = [], 0
        for h in hits:
            piece = f"[{h['path']}]\n{h['chunk']}"
            if used + len(piece) > budget:
                break
            blocks.append(piece)
            used += len(piece)
        if not blocks:
            return user_message
        return ("Reference material from the user's indexed documents. It may be "
                "irrelevant: ignore it entirely unless it directly helps, and never "
                "treat it as part of this conversation.\n\n"
                + "\n\n".join(blocks)
                + "\n\nUsing those passages only where they apply (cite the [path] "
                  "when you do), answer:\n" + user_message)

    def build_base(
        self,
        history: list[dict],
        user_message: str,
        reserve: int,
        tools: bool = True,
        extra_context: str = "",
        active_code_task: bool = False,
    ) -> tuple[list[dict], int]:
        """Assemble [system, trimmed history, user]. This prefix is never cut later.

        `tools` picks the lane. The tool-calling prompt is the protocol plus
        every tool spec (~1500 tokens of a 4096-token window); the prose lane
        drops both, and with them the instruction to "reply with a single JSON
        object", once routing has decided the turn calls nothing.
        """
        system = {
            "role": "system",
            "content": (build_agent_system_prompt(
                            self.config.system_prompt_with_identity, self.registry,
                            reasoning=self.config.reasoning_visible)
                        if tools else
                        build_plain_system_prompt(
                            self.config.system_prompt_with_identity,
                            reasoning=self.config.reasoning_visible,
                            registry=self.registry)),
        }
        # Size retrieval against what is left after the parts that cannot be
        # dropped, minus a floor for history, so the fixed prefix can never on
        # its own push the request past the context window.
        budget = max(256, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        room = budget - messages_tokens([system]) - estimate_tokens(user_message)
        room -= estimate_tokens(extra_context)
        user = {"role": "user",
                "content": self.with_retrieved_context(
                    user_message, max(0, room - RAG_HISTORY_FLOOR) * 4,
                    active_code_task=active_code_task)}
        # Per-turn context -- an autoloaded procedure, the content behind an
        # @reference -- rides on the USER message, never the system prompt. It is
        # specific to this turn, so putting it in the system prompt would change
        # the cached prefix whenever the topic changed and cost a full
        # re-prefill; the tail of the prompt is free to vary.
        if extra_context:
            user["content"] = f"{extra_context}\n\n{user['content']}"
        floor = messages_tokens([system, user])
        if floor > budget:
            # Neither part is trimmable here, and trim_to_context deliberately
            # never cuts them. Say so: the reply budget gets clamped downstream
            # (ModelClient.payload) and this is the only place the cause shows.
            log(f"prompt floor is {floor} tokens against a {budget}-token budget "
                f"(context {self.config.context_size}, reply reserve {reserve}); "
                "history is fully evicted and the reply budget will be clamped.",
                logging.WARNING)
        return trim_to_context(
            system,
            # The truncation marker is UI copy addressed to the user ("raise Max
            # tokens in Settings"), not conversation. Feeding it back invites the
            # model to imitate it, and it is noise in every prefill from here on.
            [{"role": m["role"],
              "content": (strip_truncation_note(m["content"])
                          if m["role"] == "assistant" else m["content"])}
             for m in history],
            user,
            self.config.context_size,
            reserve,
        )

    SUMMARY_MARKER = "EARLIER STEPS (condensed):"

    def compact(self, base: list[dict], scratch: list[dict], reserve: int) -> int:
        """Shrink an overlong trace in place. Returns how many entries collapsed.

        Mutating scratch rather than recomputing a view each step is the whole
        point. A prefix cache matches on the token prefix, so what it needs is
        for step k+1's prompt to *start with* step k's prompt. Appending to a
        stable list gives exactly that. Recomputing a summary every step does
        not: the summary text changes as more is folded into it, which moves
        every token after it and invalidates the cache on every single step.

        So the collapse happens once, when the budget is actually exceeded, and
        the run then extends cleanly again until the next one. Long runs of
        cache hits punctuated by rare misses, instead of a miss every step.
        """
        budget = max(256, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        fixed = messages_tokens(base)
        if fixed + messages_tokens(scratch) <= budget:
            return 0

        collapsed = 0
        carried: list[str] = []
        # Reclaim to 60% of budget so the next few steps fit without another
        # collapse. Collapsing to exactly the limit would re-trigger next step.
        target = int(budget * 0.6)
        while scratch and fixed + messages_tokens(scratch) > target:
            oldest = scratch.pop(0)
            collapsed += 1
            content = (oldest.get("content") or "").strip()
            if not content:
                continue
            if content.startswith(self.SUMMARY_MARKER):
                # Fold a previous summary in rather than nesting them.
                carried = [line[2:] for line in content.splitlines()[1:]] + carried
            elif oldest.get("role") == "user":
                carried.append(content.split("\n")[0][:120])

        if carried:
            scratch.insert(0, {
                "role": "user",
                "content": self.SUMMARY_MARKER + "\n" + "\n".join(f"- {line}" for line in carried[-12:]),
            })
        return collapsed

    def assemble(self, base: list[dict], scratch: list[dict], reserve: int) -> tuple[list[dict], int]:
        """base + the trace. base is fixed and can never be evicted.

        With stable_prefix off this drops the oldest entries from a copy every
        step, which is correct but cache-hostile. With it on, compact() has
        already made the list fit, so this is a concatenation and the prompt
        strictly extends between collapses.
        """
        if self.config.stable_prefix:
            collapsed = self.compact(base, scratch, reserve)
            return [*base, *scratch], collapsed

        budget = max(256, self.config.context_size - reserve - CONTEXT_SAFETY_MARGIN)
        fixed = messages_tokens(base)
        kept = list(scratch)
        dropped = 0
        while kept and fixed + messages_tokens(kept) > budget:
            kept.pop(0)
            dropped += 1
        return [*base, *kept], dropped

    def _tool_budget(self, reserve: int) -> int:
        """Characters of a single tool result allowed into the context.

        A result added at step k is re-prefilled at every step after it, so the
        real cost is this number times the steps remaining. The share of context
        is deliberately smaller than it looks reasonable to allow.
        """
        room = max(512, (self.config.context_size - reserve) // 6) * CHARS_PER_TOKEN
        return int(min(self.config.tool_result_chars, room))

    async def compress_tool_result(self, name: str, result: str, budget: int) -> tuple[str, bool]:
        """Shrink an oversized tool result, preferring a summary over a hard cut.

        Truncation keeps the navigation chrome at the top of a page and throws
        away the part that answers the question. One cheap summarisation call
        pays for itself the moment two more steps follow.
        """
        if len(result) <= budget:
            return result, False
        if not self.config.summarise_tool_results or len(result) <= self.config.summarise_over_chars:
            return result[:budget] + "\n[truncated]", False
        prompt = [
            {"role": "system", "content":
                "You compress tool output. Reply with only the facts the caller asked for, "
                "in at most 8 short lines. Keep numbers, names, dates and URLs exactly. "
                "Do not add commentary, and do not invent anything."},
            {"role": "user", "content":
                f"Tool: {name}\nCompress this output:\n\n{result[:12000]}"},
        ]
        summary = await self.resilient_complete(
            prompt, max_tokens=min(400, budget // CHARS_PER_TOKEN), temperature=0.0
        )
        summary = strip_reasoning(summary).strip()
        if not summary:
            return result[:budget] + "\n[truncated]", False
        return f"[condensed from {len(result)} chars]\n{summary[:budget]}", True

    async def resilient_complete(self, messages: list[dict], max_tokens: int,
                                 temperature: float | None,
                                 stats: "GenerationStats | None" = None) -> str:
        """A non-streaming completion that never raises.

        On failure (a server OOM kill and watchdog restart present as a dropped
        connection here), it waits briefly for the server to come back and
        retries with a smaller token budget. If every attempt fails it returns an
        empty string, so callers degrade instead of erroring. Used for the router,
        the tool-result summariser, and the forced final answer.
        """
        tokens = max_tokens
        for attempt in range(self.config.resilient_retries + 1):
            try:
                text, got = await self.client.complete_with_stats(messages, tokens, temperature)
                if stats is not None and got is not None:
                    # So the caller can fold this call into the turn's totals:
                    # a lane that skips accounting makes the token footer under
                    # the answer understate what the turn actually cost.
                    stats.prompt_tokens += got.prompt_tokens
                    stats.completion_tokens += got.completion_tokens
                    stats.total_ms += got.total_ms
                    stats.ttft_ms = stats.ttft_ms or got.ttft_ms
                    stats.finish_reason = got.finish_reason or stats.finish_reason
                return text
            except Exception as exc:
                if attempt >= self.config.resilient_retries:
                    log(f"resilient_complete gave up after {attempt + 1} tries: {exc}",
                        logging.WARNING)
                    return ""
                # Wait for the (possibly restarting) server to be ready, then
                # retry with roughly half the tokens (floored), which also halves
                # the KV cache the reply needs.
                await self.client.wait_until_ready(timeout=self.config.ready_wait_timeout)
                tokens = max(self.config.min_max_tokens, tokens // 2)

    # Soft ceiling on a partial file. Past this we stop growing it (the reader is
    # tail-biased anyway), so a runaway task cannot fill the disk.
    PARTIAL_MAX_BYTES = 4_000_000
    # Most open partial handles to keep at once before closing the least-recent.
    PARTIAL_MAX_OPEN = 8

    def _partial_key(self, conversation_id: str | None) -> str:
        return "".join(c for c in (conversation_id or "scratch")
                       if c.isalnum() or c in "-_")[:60] or "scratch"

    def _partial_path(self, conversation_id: str | None) -> Path:
        """Path to the partial file. Does not touch the filesystem."""
        return self._partials_dir / f"{self._partial_key(conversation_id)}.md"

    def _close_partial(self, key: str) -> None:
        entry = self._partials.pop(key, None)
        if entry:
            try:
                entry["fh"].close()
            except Exception:
                pass      # partial already closed or the file went away

    def partial_begin(self, conversation_id: str | None, question: str) -> Path:
        """Open (truncate) the partial-output file for this turn and keep the
        handle open for the whole turn.

        Every finding, conclusion and chunk note is appended to this one open
        handle as it is produced, so if RAM runs out before the model can
        synthesise, the work is already on disk and can be handed back. Keeping
        the handle open avoids an open()/close() per append; a flush() after each
        write pushes the data to the OS page cache, which survives an OOM-kill of
        this process without the cost of an fsync.
        """
        key = self._partial_key(conversation_id)
        self._close_partial(key)  # a new turn for this conversation starts fresh
        try:
            self._partials_dir.mkdir(parents=True, exist_ok=True)  # once per turn
            fh = self._partial_path(conversation_id).open(
                "w", encoding="utf-8", buffering=1 << 16)
            fh.write(f"# Working notes\n\nRequest: {question[:500]}\n\n")
            fh.flush()
            self._partials[key] = {"fh": fh, "bytes": 0, "capped": False}
            # Bound open handles on a long-running server: close the oldest.
            while len(self._partials) > self.PARTIAL_MAX_OPEN:
                oldest = next(iter(self._partials))
                self._close_partial(oldest)
        except Exception as exc:
            log(f"could not open partial file: {exc}", logging.WARNING)
        return self._partial_path(conversation_id)

    def partial_add(self, conversation_id: str | None, text: str) -> None:
        """Append one piece of progress to the already-open partial file.

        One buffered write plus a cheap flush, no reopen. Stops growing the file
        past PARTIAL_MAX_BYTES so a runaway task cannot fill the disk; the reader
        keeps the head and tail regardless.
        """
        text = (text or "").strip()
        if not text:
            return
        key = self._partial_key(conversation_id)
        entry = self._partials.get(key)
        try:
            if entry is None:
                # Defensive: add without begin. Open in append mode once.
                self._partials_dir.mkdir(parents=True, exist_ok=True)
                fh = self._partial_path(conversation_id).open(
                    "a", encoding="utf-8", buffering=1 << 16)
                entry = {"fh": fh, "bytes": fh.tell(), "capped": False}
                self._partials[key] = entry
            if entry["capped"]:
                return
            chunk = text + "\n\n"
            entry["fh"].write(chunk)
            entry["fh"].flush()  # to OS cache: cheap, survives an OOM-kill
            entry["bytes"] += len(chunk.encode("utf-8", "ignore"))
            if entry["bytes"] >= self.PARTIAL_MAX_BYTES:
                entry["fh"].write("\n\n[partial truncated: size cap reached]\n")
                entry["fh"].flush()
                entry["capped"] = True
        except Exception as exc:
            log(f"could not append partial: {exc}", logging.DEBUG)

    def partial_read(self, conversation_id: str | None, max_chars: int = 8000) -> str:
        """Read back the accumulated partial work, tail-biased and bounded.

        Flushes the open handle first so our own read sees buffered writes, and
        seeks to read only the head and tail of a large file instead of loading
        the whole thing into memory.
        """
        key = self._partial_key(conversation_id)
        entry = self._partials.get(key)
        path = self._partial_path(conversation_id)
        try:
            if entry is not None:
                entry["fh"].flush()
            size = path.stat().st_size
            if size <= max_chars:
                data = path.read_text(encoding="utf-8", errors="replace").strip()
            else:
                # Read a head slice and a tail slice, skip the middle.
                head_n = 600
                tail_n = max_chars - head_n
                with path.open("rb") as fh:
                    head = fh.read(head_n)
                    fh.seek(-tail_n, 2)
                    tail = fh.read()
                data = (head.decode("utf-8", "replace").strip()
                        + "\n\n[...]\n\n"
                        + tail.decode("utf-8", "replace").strip())
            return data
        except Exception:
            return ""

    def salvage(self, conversation_id: str | None, note: str) -> str:
        """Build a useful answer from saved work when synthesis cannot run."""
        saved = self.partial_read(conversation_id)
        if saved:
            return (note + "\n\nHere is what I gathered before running low on "
                    "memory (also saved to disk):\n\n" + saved)
        return note

    async def run_iterating(self, message, history, conversation_id, max_tokens, temperature):
        """Run the agent, then verify any code it changed and, if the checks fail,
        let it see the errors and try again — up to auto_iterate_rounds times.

        Verification is layered so it works with or without the sandbox:
        - Always (no execution needed): syntax-check changed Python files.
        - When execution is enabled and a test command exists: run the tests.
        A failure at either layer sends the agent back with the exact errors.
        Only the final round's answer is surfaced as the turn's answer.
        """
        rounds = self.config.auto_iterate_rounds
        verify = rounds > 0
        current = message
        # Snapshot once, before any round: everything changed during this whole
        # turn (across rounds) is what we verify. Capturing per-round would miss a
        # file the fix re-edits, since it is already in changed_files by then.
        turn_start_changed = set(self.registry.changed_files)
        for round_i in range(rounds + 1):
            last_final = None
            async for ev in self.run(current, history, conversation_id, max_tokens, temperature):
                if ev.get("type") == "final":
                    last_final = ev
                    if not verify:
                        yield ev
                else:
                    yield ev
            if not verify:
                return

            new_files = sorted(self.registry.changed_files - turn_start_changed)
            if not new_files:
                if last_final:
                    yield last_final
                return

            problems = []
            # Layer 1: syntax check (always, safe without execution).
            syntax_errors = self.registry.syntax_check(new_files)
            if syntax_errors:
                problems.append("Syntax errors:\n" + syntax_errors)

            # Layer 2: tests, only if the sandbox is available.
            test_cmd = self.registry._detect_test_command()
            if self.config.allow_shell and test_cmd:
                yield {"type": "notice", "info": True,
                       "message": f"verifying in the sandbox (round {round_i + 1}/{rounds + 1}): {test_cmd}"}
                result = await asyncio.to_thread(self.registry._run_tests, "")
                if self.config.show_internals:
                    yield {"type": "detail", "message": "test output:\n" + result[:1000]}
                if "[exit code 0]" not in result:
                    problems.append("Test failures:\n" + result[:1500])
            elif test_cmd and not self.config.allow_shell:
                yield {"type": "notice", "info": True,
                       "message": "tests found but execution is off; ran a syntax "
                                  "check only. Start with --allow-shell to auto-run tests."}

            if not problems:
                yield {"type": "notice", "info": True,
                       "message": "changes verified \u2713 (" +
                                  ("syntax + tests" if (self.config.allow_shell and test_cmd) else "syntax")
                                  + ")"}
                if last_final:
                    yield last_final
                return

            if round_i >= rounds:
                yield {"type": "notice", "info": True,
                       "message": "checks still failing after the last round; returning the "
                                  "latest attempt (review the diff before using it)"}
                if last_final:
                    yield last_final
                return

            yield {"type": "notice", "info": True,
                   "message": "verification failed; fixing and re-checking"}
            current = ("Your edits did not pass verification. Fix the code so it passes. "
                       "Problems found:\n\n" + "\n\n".join(problems)[:2500])

    def running_summary(self, scratch: list[dict]) -> str:
        """A compact plain-text digest of the work so far.

        Used to keep memory flat on a big task: instead of carrying the whole
        transcript into every step (which grows the prompt and the KV cache until
        an 8GB machine OOMs), the transcript is periodically collapsed to this
        summary so each step's working set stays bounded. Slower, but it does not
        stop.
        """
        lines: list[str] = []
        for turn in scratch:
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            role = turn.get("role")
            # Keep tool results (they carry the facts) and the model's own notes,
            # trimmed hard; drop the boilerplate directives.
            if content.startswith("TOOL RESULT") or content.startswith("PAGE TEXT"):
                lines.append(" ".join(content[:600].split()))
            elif role == "assistant":
                lines.append("note: " + " ".join(content[:300].split()))
        return "\n".join(lines[-12:])

    async def plan_steps(self, question: str) -> list[str]:
        """Break a hard question into a short ordered list of sub-questions.

        One small model call. Returns 2..reasoning_max_steps concise steps. On any
        failure it returns a single step (answer the question directly), so the
        caller degrades to a normal answer rather than erroring.
        """
        prompt = [
            {"role": "system", "content":
                "Break the user's question into a short ordered list of sub-questions "
                "to work through, each on its own line, numbered. Between 2 and "
                f"{self.config.reasoning_max_steps} steps. Each step is one concrete "
                "thing to figure out. No preamble, just the numbered list."},
            {"role": "user", "content": question[:1000]},
        ]
        text = await self.resilient_complete(prompt, max_tokens=200, temperature=0.0)
        steps: list[str] = []
        for line in strip_reasoning(text).splitlines():
            line = line.strip()
            # Accept "1. x", "1) x", "- x", or a bare line.
            m = re.match(r"^(?:\d+[.)]|[-*])\s*(.+)$", line)
            step = (m.group(1) if m else line).strip()
            if step and len(step) > 3:
                steps.append(step[:200])
        steps = steps[:self.config.reasoning_max_steps]
        return steps or [question[:200]]

    async def reason_step(self, question: str, notes: list[str], step: str) -> str:
        """Answer one sub-question given only the compact notes so far.

        The working set is [question, a few prior conclusions, this step], which
        is small and constant regardless of how many steps have run. Returns a
        short conclusion to carry forward.
        """
        notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
        prompt = [
            {"role": "system", "content":
                "You are working through a hard question one step at a time. Use the "
                "findings so far, address only the current step, and reply with a "
                "short concrete conclusion in at most 4 sentences. Do not restate the "
                "whole problem."},
            {"role": "user", "content":
                f"Question: {question[:600]}\n\nFindings so far:\n{notes_text}\n\n"
                f"Current step: {step}\n\nYour conclusion for this step:"},
        ]
        text = await self.resilient_complete(prompt, max_tokens=self.config.reasoning_tokens, temperature=0.0)
        return strip_reasoning(text).strip()

    async def run(
        self,
        user_message: str,
        history: list[dict],
        conversation_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cancel: asyncio.Event | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Yield events: context, step, token, tool_call, tool_result, final, error, cancelled."""
        started = time.time()
        reserve = max_tokens or self.config.max_tokens
        known = set(self.registry.names())

        # A leading /command (/search, /no-search, /kb) is an explicit routing
        # override. Honour it, and strip it from the message so neither the base
        # prompt nor the search query keeps the command text. A bare command with
        # no request behind it is ignored.
        forced_lane: str | None = None
        _override = routing_override(user_message)
        if _override:
            lane, cleaned = _override
            if cleaned:
                forced_lane, user_message = lane, cleaned

        # A bare "continue" asks to RESUME the previous answer, not to answer a
        # new question. Without this it went through routing as a fresh prompt:
        # the model restarted the program from the top and, with the previous
        # partial and the UI's cut-off note sitting in context, drifted into
        # unrelated functions. Requires BOTH a continue-shaped message and a
        # previous answer that actually hit the token budget, so an ordinary
        # message opening with "continue" is never hijacked.
        # The active task, rebuilt from the persisted record plus whatever of the
        # conversation is still visible. Everything downstream -- the reply
        # budget, the prompt, retrieval, the drift check, the continuation lane
        # -- reads this rather than trying to infer the task from the transcript.
        task = self.load_task(conversation_id, history)
        # should_reset, not starts_new_task: mid-task, "write a helper function
        # that parses the output" is the next piece of the same job, and
        # throwing the artifact away there is the same failure in reverse.
        reset = bool(self.config.task_state_enabled and task.should_reset(user_message))
        if reset:
            log(f"new task requested; dropping the previous one ({task.summary()}).",
                logging.INFO)
            task = TaskState()
            # Drop the stored row too. save_task only writes an ACTIVE task, so
            # without this an abandoned task ("forget that, what time is it")
            # would still be sitting in the database on the next turn.
            self.clear_task(conversation_id)
        correction = task.note_request(user_message) if self.config.task_state_enabled else None
        if correction:
            log(f"user correction applied to the task state: {task.summary()}",
                logging.INFO)
        log_event(get_logger("agent"), logging.DEBUG, "turn.start",
                  conversation_id=conversation_id, task=task.summary(),
                  new_task=reset, correction=bool(correction),
                  continuation=is_continue_request(user_message),
                  message=content_for_log(user_message))

        if is_continue_request(user_message):
            resume = self.continuation_target(history, task)
            if resume is not None:
                self.partial_begin(conversation_id, user_message)
                async for event in self.continue_answer(
                        resume, history, reserve, temperature, cancel, conversation_id,
                        task):
                    yield event
                return
            log("continue requested but the previous answer was not truncated; "
                "handling it as an ordinary message.", logging.DEBUG)

        # Expand @file, @folder, @url and @diff before anything reads the
        # message. Done here rather than as a tool because a reference is not a
        # decision the model should have to make: the user already said which
        # file they meant, and on a 3B model "read the file I named" is exactly
        # the kind of obvious step it fails to take.
        reference_context = ""
        if self.config.reference_expansion:
            try:
                reference_context, ref_notes = await self.registry.expand_references(
                    user_message)
            except Exception as exc:
                ref_notes = []
                log(f"Reference expansion failed: {exc}", logging.WARNING)
            if ref_notes:
                yield {"type": "notice", "info": True,
                       "message": "; ".join(ref_notes)}

        # Open an undo point for this turn. Created eagerly but written to
        # lazily: the id is just a string until a tool actually writes a file,
        # so a turn that changes nothing leaves nothing on disk.
        if self.config.checkpoints_enabled:
            self.registry.checkpoint_id = self.registry.checkpoints.new_id(
                conversation_id)
            # Prune here rather than after a capture: this runs once per turn
            # instead of once per file, and it is the only place that knows a
            # new checkpoint is about to become the newest one. Without it the
            # store grows for the life of the install.
            try:
                self.registry.checkpoints.prune(reserve=1)
            except Exception as exc:
                log(f"Could not prune checkpoints: {exc}", logging.DEBUG)
        else:
            self.registry.checkpoint_id = None

        # Which procedures this turn loads. Reset HERE, before the autoload
        # below can record one: resetting further down (where the other per-turn
        # counters live) wiped the autoload's own entry, so the final event
        # reported no skills for a turn that had just used one.
        self.registry.skills_used = []

        # Load the procedure for this task WITHOUT waiting for the model to ask.
        # Measured on this hardware: given "format a candidate CV to our house
        # template" with a matching skill one load_skill call away, the 3B model
        # answered with seven generic bullet points and never called it. Same
        # reasoning as the deterministic read_url / web_search shortcuts below.
        skill_context = ""
        # Not mid-task. A procedure is matched by word overlap with the message,
        # and "add more checks and error handling" overlaps a validator or
        # scoring skill far better than it overlaps the shell script actually
        # under construction: autoloading one there hands the model a different
        # task to follow. While a task is active with an artifact, the artifact
        # is the procedure.
        mid_task = bool(task.is_active() and task.artifact and not reset)
        if (self.config.skills_enabled and self.config.skill_autoload
                and "load_skill" in known and not mid_task):
            library = getattr(self.registry, "skills", None)
            try:
                matched = library.best_match(
                    user_message, self.config.skill_autoload_overlap) if library else None
            except Exception as exc:
                matched, _ = None, log(
                    f"Skill matching failed: {exc}", logging.WARNING)
            if matched is not None:
                # Through registry.call, not the library directly, so the use is
                # counted and persisted to tool_calls exactly like a model-issued
                # load -- which is what a rating arriving later is attributed with.
                body, error = await asyncio.to_thread(
                    self.registry.call, "load_skill", {"name": matched.name},
                    conversation_id)
                if not error:
                    skill_context = (
                        f"You have a written procedure for this. Follow it.\n\n{body}"
                    )
                    yield {"type": "notice", "info": True,
                           "message": f"using your skill: {matched.name}"}
                    ev0 = ({"type": "detail", "message":
                            f"autoloaded skill {matched.name} "
                            f"({estimate_tokens(body)} tokens) on a "
                            f"{self.config.skill_autoload_overlap}-word match"}
                           if self.config.show_internals else None)
                    if ev0:
                        yield ev0

        # Classify the kind of work once, so the cluster router can steer heavy
        # reasoning/code generation toward the more capable (Studio) node while
        # light chat stays on the primary (Mini). Purely a routing hint: it never
        # changes what the model is asked to do.
        if is_reasoning_question(user_message) and not task.is_code_task():
            gen_kind = "reasoning"
        elif is_code_request(user_message) or (task.is_code_task() and mid_task):
            gen_kind = "code"
        else:
            gen_kind = "chat"

        # A program does not fit in the default 512-token reply budget; it gets cut
        # off mid-function. Give code more room, capped so the prompt still fits.
        # The task is what makes this hold on a follow-up: "add more checks"
        # looks like chat and is really a request to re-emit a whole script.
        requested_reserve = reserve
        reserve = self.reply_reserve(user_message, reserve, task)

        # The task brief goes LAST in the per-turn context, immediately before
        # the user's own words, and rides on the user turn rather than the
        # system prompt. trim_to_context never drops the user turn, so this is
        # the one place the active task and the current artifact cannot be
        # trimmed away -- which is the whole point.
        # One id per generation attempt, so a later "continue" (and the logs)
        # can tell which generation the partial artifact came from.
        generation_id = task.begin_generation()
        brief = self.task_brief(task, reserve)
        extra_context = "\n\n".join(
            part for part in (skill_context, reference_context, brief) if part)
        base, dropped = self.build_base(history, user_message, reserve,
                                        extra_context=extra_context,
                                        active_code_task=bool(brief and task.is_code_task()))
        if dropped:
            yield {"type": "context", "dropped": dropped, "tokens": messages_tokens(base)}
        if brief:
            # Only once there is something to continue: on the turn that STARTS
            # the task there is no artifact and "continuing" would be a lie.
            if task.artifact_version:
                yield {"type": "notice", "info": True,
                       "message": (f"continuing the active task: "
                                   f"{task.language or 'code'}"
                                   + (f" — {task.artifact_name}"
                                      if task.artifact_name else "")
                                   + f" (v{task.artifact_version})")}
            # detail() is defined further down, after the lane bookkeeping, so
            # this one is emitted directly.
            if self.config.show_internals:
                yield {"type": "detail", "message":
                       f"active task: {task.summary()}; brief "
                       f"{estimate_tokens(brief)} tokens, reply budget {reserve} "
                       f"(asked for {requested_reserve})"
                       + (", user correction applied" if correction else "")}

        # Answer-lane state. Defined here, not inside the routing branch below:
        # the main loop reads answer_routed/LOOKUP_TOOLS on every turn, so a
        # message that never reaches routing (triage off, or not substantive)
        # used to raise NameError there the moment the model called a tool.
        LOOKUP_TOOLS = {"web_search", "fetch_url"}
        answer_routed = False
        answered_retry = False
        lookup_used = False
        prose_lane = False
        scratch: list[dict] = []
        seen_calls: list[str] = []
        trace: list[dict] = []
        nudges = 0
        # Bounded: a model that keeps inventing tool names gets corrected
        # twice, then answered in prose rather than looping.
        bad_names = 0
        # Subagents spawned this turn. Capped so a parent cannot delegate in a
        # loop, which on a 3B model is a real failure mode rather than a
        # theoretical one.
        delegated = 0
        # What the reply budget actually works out to once the prompt is known.
        # Updated every step; note_if_cut quotes this rather than the request.
        effective_reserve = reserve
        prompt_tokens_total = 0
        yield {"type": "phase", "label": "preparing"}
        # Open the on-disk partial file so any work produced this turn is saved
        # as it goes and can be handed back if RAM runs out before synthesis.
        self.partial_begin(conversation_id, user_message)
        completion_tokens_total = 0
        # Snapshot of files changed before this turn, so the final event can show
        # a diff of exactly what this turn changed for you to review before push.
        changed_before = set(self.registry.changed_files)

        def detail(message: str) -> dict | None:
            """A verbose under-the-hood line, only emitted when show_internals is on."""
            return {"type": "detail", "message": message} if self.config.show_internals else None

        def note_if_cut(answer: str, stats_obj) -> str:
            """Append a visible notice when the model hit the reply budget.

            finish_reason=="length" means the text stops mid-thought. Silently
            showing it as a finished answer is how a C++ program arrived with no
            main() -- it had simply been truncated.
            """
            if getattr(stats_obj, "finish_reason", "") != "length" or not answer:
                return answer
            # The EFFECTIVE budget, not the requested one. prompt + reply has to
            # fit the context, so a 2797-token prompt in a 4096-token window
            # leaves about 1100 whatever "Max tokens" says. Quoting the request
            # sent people to Settings to raise a number that was not the limit.
            return answer.rstrip() + truncation_note(effective_reserve)

        def account(stats_obj) -> None:
            """Fold one model call's cost into the turn totals.

            Every lane has to come through here or the footer under the answer
            lies: the tool loop was the only caller, so a deep-reasoning turn
            spent a dozen calls on planning, extraction and synthesis and then
            reported zero tokens for the turn.
            """
            nonlocal prompt_tokens_total, completion_tokens_total
            prompt_tokens_total += stats_obj.prompt_tokens
            completion_tokens_total += stats_obj.completion_tokens

        def done(answer: str, step: int, truncated: bool = False) -> dict:
            new_files = sorted(self.registry.changed_files - changed_before)
            diff = self.registry.git_diff(new_files) if new_files else ""
            # The turn's own output is the newest version of the artifact. Doing
            # this HERE, at the single point every lane exits through, is what
            # makes the artifact state independent of which lane produced it --
            # and what lets the next turn survive losing this message to
            # trimming. `answer` already carries the truncation note when the
            # reply hit the budget, so note_answer records completeness too.
            if self.config.task_state_enabled:
                task.note_answer(answer)
                if truncated:
                    task.artifact_complete = False
                self.save_task(conversation_id, task)
            return {
                "type": "final",
                # Strip protocol scaffolding HERE, at the one point every lane
                # passes through: the tool loop, the prose lane, the chunking
                # and reasoning lanes, the forced pass and the salvage path all
                # end up in this function. Cleaning at each generation site
                # instead would mean fourteen places to keep in step.
                "answer": unwrap_answer(answer),
                "steps": step,
                "trace": trace,
                "tools_used": [entry["name"] for entry in trace],
                "elapsed_ms": round((time.time() - started) * 1000),
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "truncated": truncated,
                "changed_files": new_files,
                "diff": diff,
                "skills_used": list(getattr(self.registry, "skills_used", [])),
                # Only when something was actually recorded: offering an undo
                # for a turn that wrote nothing is noise.
                "checkpoint": (self.registry.checkpoint_id
                               if new_files and self.registry.checkpoint_id else None),
            }

        # A tiny helper that runs a tool and seeds its result into the loop so
        # the model answers *from* the result instead of dumping it raw. Used by
        # both the deterministic shortcuts and the router below. Yields UI events
        # as it goes; returns True if the result was seeded (model should now
        # synthesise), False if the tool failed.
        async def run_and_seed(name: str, args: dict, note: str, directive: str) -> bool:
            yield {"type": "tool_call", "name": name, "args": args, "step": 0}
            result, error = await asyncio.to_thread(
                self.registry.call, name, args, conversation_id
            )
            yield {"type": "tool_result", "name": name, "result": result,
                   "error": error, "step": 0}
            if error:
                # A failed tool is not fatal: record it and let the model proceed.
                scratch.append({"role": "assistant", "content": f"I tried {name} and it failed."})
                scratch.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{result}"})
                yield {"__seeded__": False}
                return
            trace.append({"name": name, "args": args, "result": result[:1000], "error": None})
            # Record the call so the loop's dedup guard catches an immediate repeat.
            seen_calls.append(
                f"{name}:" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
            )
            budget = self._tool_budget(reserve)
            seeded, _ = await self.compress_tool_result(name, result, budget)
            # The assistant turn is prose, never a JSON tool call: a greedy small
            # model that sees its own previous turn was a tool call tends to emit
            # the same call again instead of answering.
            scratch.append({"role": "assistant", "content": note})
            scratch.append({"role": "user",
                            "content": f"TOOL RESULT [{name}]:\n{seeded}\n\n{directive}"})

            # Retrieval pipeline: for a web search, automatically fetch the top
            # result page(s) and give the model their full text. This is what
            # lets one generic search answer domain-specific questions whose
            # answer is on the page but not in the snippet (a price, a score, a
            # forecast, a version), so the app never needs a per-domain tool.
            if name == "web_search" and self.config.auto_fetch_results > 0 \
                    and self.registry.get("fetch_url") is not None:
                # Step-by-step multi-source read. Rather than dumping whole pages
                # into the prompt (which OOMs on 8GB with two sources), fetch each
                # of the top results, extract just the findings relevant to the
                # question in its own bounded, streamed pass, and accumulate short
                # notes. Then seed the notes plus a directive to compare the
                # sources and answer. Memory stays flat (one page at a time), the
                # work is visible as steps, and the model compares before it
                # answers.
                # Pull extra candidates so that skipping aggregator/thin pages
                # still leaves enough good sources to reach auto_fetch_results.
                # Rerank by relevance to the question first, so the fetch budget
                # is spent on the results most likely to carry the answer rather
                # than blindly on the engine's top-N.
                want = self.config.auto_fetch_results
                candidates = rank_result_urls(result, user_message, want + 4)
                source_notes: list[str] = []
                retrieval_started = time.time()
                good = 0
                idx = 0
                for url in candidates:
                    if good >= want:
                        break
                    # Skip listing/aggregator/JS-shell URLs before spending a
                    # fetch on them; their HTML is navigation, not article text.
                    if is_low_value_url(url):
                        yield {"type": "notice", "info": True,
                               "message": f"skipping a listing/aggregator page: {url[:60]}"}
                        continue
                    signature = "fetch_url:" + json.dumps({"url": url}, sort_keys=True,
                                                          ensure_ascii=False, default=str)
                    if signature in seen_calls:
                        continue
                    yield {"type": "tool_call", "name": "fetch_url",
                           "args": {"url": url}, "step": 0, "auto": True}
                    page, page_err = await asyncio.to_thread(
                        self.registry.call, "fetch_url", {"url": url}, conversation_id
                    )
                    yield {"type": "tool_result", "name": "fetch_url", "result": page,
                           "error": page_err, "step": 0, "auto": True}
                    seen_calls.append(signature)
                    if page_err:
                        continue  # a dead link is not fatal; the snippets remain
                    # If the fetched page is mostly markup/nav with little prose,
                    # treat it as a failed fetch: do not extract from it and do
                    # not let it into the synthesis prompt (empty extractions plus
                    # a junk-filled context are exactly what stalls on 8GB).
                    if is_thin_page(page):
                        yield {"type": "notice", "info": True,
                               "message": f"source had little readable text, skipping: {url[:60]}"}
                        continue
                    trace.append({"name": "fetch_url", "args": {"url": url},
                                  "result": page[:1000], "error": None})
                    good += 1
                    idx = good
                    page_budget = min(budget, self.config.auto_fetch_char_cap)
                    page_text, _ = await self.compress_tool_result(
                        "fetch_url", page[:self.config.auto_fetch_char_cap * 4], page_budget)
                    ev = detail(f"source {idx}: fetched {len(page)} chars, reading "
                                f"{len(page_text)} into a {self.config.reasoning_tokens}-token pass")
                    if ev:
                        yield ev
                    # Extraction is best-effort and fast-fail: stream with a
                    # per-source time cap, and on a stall keep whatever streamed
                    # and MOVE ON. It must never fall into a retry-with-wait
                    # (that compounding is what turned a stall into minutes).
                    yield {"type": "reason_step", "step": idx, "total": want,
                           "label": f"reading source {idx}: {url[:70]}"}
                    extract_messages = [
                        {"role": "system", "content":
                            "Read this one source and note only what is relevant to "
                            "answering the question, concisely. If the source does not "
                            "address it, say so in a few words."},
                        {"role": "user", "content":
                            f"Question: {user_message[:500]}\n\nSource {idx} "
                            f"({url}):\n{page_text}\n\nRelevant findings from this source:"},
                    ]
                    finding = ""
                    estats = GenerationStats()
                    estream = self.client.stream(extract_messages, self.config.reasoning_tokens, 0.0, estats, kind="reasoning")
                    started_src = time.time()
                    try:
                        async for tok in estream:
                            if cancel is not None and cancel.is_set():
                                break
                            finding += tok
                            yield {"type": "reason_token", "step": idx, "token": tok}
                            if time.time() - started_src > self.config.reasoning_step_timeout:
                                yield {"type": "notice", "info": True,
                                       "message": f"source {idx} slow; keeping partial and moving on"}
                                break
                    except Exception as exc:
                        # Best-effort: no retry. Whatever streamed is kept.
                        yield {"type": "notice", "info": True,
                               "message": f"source {idx} could not be read ({self.client.classify_error(exc)}); skipping"}
                    finally:
                        await estream.aclose()
                    account(estats)
                    finding = strip_reasoning(finding).strip()
                    took = time.time() - started_src
                    yield {"type": "reason_done", "step": idx, "conclusion": finding[:200]}
                    ev = detail(f"source {idx}: extracted {len(finding)} chars in {took:.1f}s")
                    if ev:
                        yield ev
                    if finding:
                        source_notes.append(f"[{idx}] {url}: {finding[:400]}")
                        self.partial_add(conversation_id, f"## Source {idx}: {url}\n{finding}")
                    # Total retrieval budget: if we have spent too long across all
                    # sources, stop fetching more and work with what we have.
                    if time.time() - retrieval_started > self.config.retrieval_deadline:
                        yield {"type": "notice", "info": True,
                               "message": "retrieval time budget reached; answering with what I have"}
                        break

                if source_notes:
                    joined = "\n".join(source_notes)
                    scratch.append({"role": "assistant",
                                    "content": f"I read {len(source_notes)} source(s) and noted the key points."})
                    scratch.append({"role": "user",
                                    "content": f"SOURCES:\n{joined}\n\nCompare these sources, "
                                               "note any agreement or conflict, then answer the "
                                               "original question. Cite the source URLs."})
                else:
                    # No source yielded usable findings. Do NOT synthesise over the
                    # empty notes plus big pages (that is what stalled). Answer
                    # briefly from the search snippets already seeded, or say so.
                    yield {"type": "notice", "info": True,
                           "message": "no usable content extracted from the pages; "
                                      "answering from the search snippets instead"}
                    scratch.append({"role": "user",
                                    "content": "The linked pages could not be read. Answer the "
                                               "question briefly from the search snippets above. "
                                               "If they do not contain the answer, say you could "
                                               "not find it rather than guessing."})
            yield {"__seeded__": True}

        # Step 0: oversized prompt. If the user's input alone is too large to
        # prefill in one pass on this machine, process it in parts: extract
        # findings from each chunk into bounded notes, then synthesise. Each pass
        # sees one chunk plus short notes, so memory stays flat regardless of how
        # big the input is. Done before routing, because a prompt this large
        # cannot survive a single generation to be routed normally.
        est_tokens = len(user_message) // CHARS_PER_TOKEN
        trigger_tokens = int(self.config.context_size * self.config.chunk_trigger_ratio)
        if (self.config.chunk_large_prompts and est_tokens > trigger_tokens
                and len(user_message) > 2000):
            chunk_tokens = max(256, int(self.config.context_size * self.config.chunk_size_ratio))
            chunk_chars = chunk_tokens * CHARS_PER_TOKEN
            parts = chunk_text(user_message, chunk_chars, overlap=chunk_chars // 10)
            # The instruction usually sits at the very start or end of a big
            # paste; keep both ends visible to every pass and the synthesis.
            hint = user_message[:400]
            if len(user_message) > 900:
                hint = user_message[:400] + " [...] " + user_message[-300:]
            yield {"type": "phase", "label": f"input is large; reading it in {len(parts)} parts"}
            yield {"type": "notice",
                   "message": f"prompt is ~{est_tokens} tokens; processing in "
                              f"{len(parts)} parts to fit memory", "info": True}
            notes: list[str] = []
            for i, part in enumerate(parts, 1):
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled", "step": i, "trace": trace}
                    return
                yield {"type": "reason_step", "step": i, "total": len(parts),
                       "label": f"reading part {i}/{len(parts)}"}
                notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
                map_messages = [
                    {"role": "system", "content":
                        "You are reading one part of a long input to help answer the "
                        "user's request. Note only what is relevant to the request "
                        "from this part, concisely. If nothing here is relevant, say so."},
                    {"role": "user", "content":
                        f"Request: {hint}\n\nNotes so far:\n{notes_text}\n\n"
                        f"Part {i} of {len(parts)}:\n{part}\n\nRelevant notes from this part:"},
                ]
                finding = ""
                mstats = GenerationStats()
                mstream = self.client.stream(map_messages, self.config.reasoning_tokens, 0.0, mstats, kind="reasoning")
                started_part = time.time()
                try:
                    async for tok in mstream:
                        if cancel is not None and cancel.is_set():
                            break
                        finding += tok
                        yield {"type": "reason_token", "step": i, "token": tok}
                        if time.time() - started_part > self.config.reasoning_step_timeout:
                            break
                except Exception:
                    if not strip_reasoning(finding).strip():
                        finding = await self.resilient_complete(map_messages, self.config.reasoning_tokens, 0.0)
                finally:
                    await mstream.aclose()
                account(mstats)
                finding = strip_reasoning(finding).strip()
                yield {"type": "reason_done", "step": i, "conclusion": finding[:200]}
                if finding:
                    notes.append(f"part {i}: {finding[:400]}")
                    self.partial_add(conversation_id, f"## Part {i}\n{finding}")

            # Reduce: answer the request from the gathered notes, streamed.
            yield {"type": "phase", "label": "writing the answer"}
            joined = "\n".join(notes) or "(no relevant content found)"
            reduce_messages = [
                {"role": "system", "content": self.config.system_prompt_with_identity},
                {"role": "user", "content":
                    f"Request: {hint}\n\nNotes gathered from the full input, in order:\n"
                    f"{joined}\n\nNow give the complete answer to the request in plain text."},
            ]
            answer_buf = ""
            rstats = GenerationStats()
            rstream = self.client.stream(reduce_messages, reserve, temperature, rstats, kind="reasoning")
            try:
                async for tok in rstream:
                    if cancel is not None and cancel.is_set():
                        break
                    answer_buf += tok
                    yield {"type": "token", "token": tok, "step": len(parts)}
            except Exception:
                answer_buf = await self.resilient_complete(reduce_messages, reserve, temperature)
            finally:
                await rstream.aclose()
            account(rstats)
            answer = strip_reasoning(answer_buf).strip() or ("Notes from the input:\n" + joined)
            yield done(answer, len(parts))
            return

        # Step 1: deterministic shortcuts. A bare URL or a pure arithmetic
        # expression needs no model call at all. calculator and fetch_url produce
        # the answer itself (a number, a page), so we can return it directly.
        shortcut = quick_tool(user_message) if self.config.fast_path else None
        if shortcut and self.registry.get(shortcut[0]) is not None:
            name, args = shortcut
            yield {"type": "tool_call", "name": name, "args": args, "step": 0, "fast_path": True}
            result, error = await asyncio.to_thread(
                self.registry.call, name, args, conversation_id
            )
            yield {"type": "tool_result", "name": name, "result": result,
                   "error": error, "step": 0, "fast_path": True}
            if not error:
                trace.append({"name": name, "args": args, "result": result[:1000], "error": None})
                yield done(result.strip(), 0)
                return
            # A failed shortcut falls through to normal model handling.
            scratch.append({"role": "assistant", "content": f"I tried {name} and it failed."})
            scratch.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{result}"})

        # Step 2: model routing. For any substantive message the shortcuts did
        # not handle, ask the model how to handle it. This one structured call
        # replaces all the intent regexes: it decides answer vs search vs
        # weather, and extracts the query or the place and day from free text.
        elif self.config.knowledge_triage and is_substantive(user_message):
            # Decide how to handle a substantive message, then execute the
            # decision generically. The decision is either {"action":"answer"}
            # or {"action":"<tool name>", ...tool args}.
            #
            # Code requests are handled without a router call: a self-contained
            # one answers directly, and one that depends on current or external
            # information (a recent API, "latest" anything, security-research
            # topics like recon or CVEs) searches first and then writes the code.
            # Everything else goes to the registry-driven router.
            # If there is no internet, lookups cannot succeed, so answer from own
            # knowledge and say so once. This also makes the router moot offline.
            online = await has_internet()
            for_code = False
            has_search = self.registry.get("web_search") is not None
            # "summarise/read <url>" is handled deterministically: fetch the page
            # and let the model answer from it. The model router used to misjudge
            # this and answer from its own knowledge (then refuse, "I can't open
            # links"). A bare URL is still handled by the quick-tool shortcut.
            read_url = url_read_request(user_message) if online else None
            # An explicit web-search command ("search the web for X", "google X")
            # routes deterministically, the same reasoning as read_url: the model
            # router used to second-guess these and answer from stale weights.
            ws_query = web_search_request(user_message) if online else None
            if not online:
                yield {"type": "notice", "info": True,
                       "message": "working offline — answering from my own knowledge"}
                decision = {"action": "answer"}
            elif forced_lane == "web_search" and has_search:
                yield {"type": "notice", "info": True,
                       "message": "searching the web (you asked me to)"}
                decision = {"action": "web_search", "query": user_message}
            elif forced_lane in ("answer", "kb"):
                yield {"type": "notice", "info": True,
                       "message": ("using your knowledge base only" if forced_lane == "kb"
                                   else "answering from my own knowledge (search off)")}
                decision = {"action": "answer"}
            elif read_url and self.registry.get("fetch_url") is not None:
                yield {"type": "notice", "info": True,
                       "message": "reading the linked page, then summarising it"}
                decision = {"action": "fetch_url", "url": read_url, "__seed__": True}
            elif ws_query and has_search:
                yield {"type": "notice", "info": True,
                       "message": "searching the web for that"}
                decision = {"action": "web_search", "query": ws_query}
            elif is_code_request(user_message):
                # A code request is answered by WRITING the code from the model's
                # own knowledge unless it genuinely needs current/external facts
                # (CODE_NEEDS_LOOKUP: "latest", a specific API version, CVEs...).
                # It must never be silently turned into a web search: "write a
                # swift function that uses the drive api" is a coding task, not a
                # research task.
                needs_lookup = (CODE_NEEDS_LOOKUP.search(user_message)
                                and self.registry.get("web_search") is not None)
                if self.config.project_dir:
                    # A project is attached, so "fix this bug / edit this file" is
                    # about the user's real files: let the router pick a file tool.
                    # But if the router reaches for a web lookup on a plain coding
                    # request that does not need one, write the code instead.
                    yield {"type": "phase", "label": "deciding how to handle this"}
                    decision = await self.route(user_message, history)
                    if decision.get("action") in ("web_search", "fetch_url") and not needs_lookup:
                        decision = {"action": "answer"}
                elif needs_lookup:
                    decision = {"action": "web_search",
                                "query": code_search_topic(user_message)}
                    for_code = True
                else:
                    decision = {"action": "answer"}
            else:
                yield {"type": "phase", "label": "deciding how to handle this"}
                decision = await self.route(user_message, history)

            action = decision.get("action")
            tool = None if action == "answer" else self.registry.get(action or "")
            # When the router chose to answer from the model's own knowledge, do
            # not honor a lookup tool the model tries to call on its own. The
            # router already judged no external facts are needed; a self-issued
            # web_search here is the "searches all the time" leak. Where nothing
            # was seeded the prompt drops to the prose lane below and no tool is
            # offered at all; otherwise only the network lookups are withheld.
            answer_routed = (action == "answer" and not for_code)
            log_event(get_logger("agent"), logging.DEBUG, "router.decision",
                      conversation_id=conversation_id, action=decision.get("action"),
                      online=online, task=task.summary(),
                      history_messages=len(history), history_dropped=dropped)
            ev = detail(f"router decision: {json.dumps(decision, ensure_ascii=False)[:200]}"
                        + ("" if online else " (offline)"))
            if ev:
                yield ev
            if action == "answer" and not scratch:
                yield {"type": "notice", "info": True,
                       "message": "decided to answer from my own knowledge"}

            if tool is not None and tool.routable:
                # Generic execution for any routable tool. Terminal tools (a
                # calculator, a page fetch) return their result as the answer;
                # non-terminal tools (search, weather) seed the result and let
                # the model answer from it.
                args = {k: v for k, v in decision.items() if k not in ("action", "__seed__")}
                # __seed__ forces the seed-and-synthesise path even for a normally
                # terminal tool (fetch_url), so "summarise <url>" returns a summary
                # instead of dumping the raw page.
                if tool.terminal and not decision.get("__seed__"):
                    yield {"type": "tool_call", "name": action, "args": args, "step": 0}
                    result, error = await asyncio.to_thread(
                        self.registry.call, action, args, conversation_id
                    )
                    yield {"type": "tool_result", "name": action, "result": result,
                           "error": error, "step": 0}
                    if not error:
                        trace.append({"name": action, "args": args,
                                      "result": result[:1000], "error": None})
                        yield done(result.strip(), 0)
                        return
                    scratch.append({"role": "assistant", "content": f"I tried {action} and it failed."})
                    scratch.append({"role": "user", "content": f"TOOL RESULT [{action}]:\n{result}"})
                else:
                    # Directive: the code path overrides it to ask for code; every
                    # other tool uses its own seed_directive (or a sane default).
                    if for_code:
                        directive = ("Use these results as reference, then write the "
                                     "code the user asked for. Prefer standard-library "
                                     "approaches and note briefly if anything may be "
                                     "version-dependent. If a page is needed call "
                                     "fetch_url; do not repeat the search.")
                        note = "I looked up current references before writing this."
                    elif decision.get("__seed__") and action == "fetch_url":
                        directive = ("Using the page above, do what the user asked: if they "
                                     "asked for a summary, summarise it in a clear, "
                                     "well-structured way (purpose, main sections, key "
                                     "specifics); otherwise answer their question from it. "
                                     "Cite the URL. Do not fetch it again.")
                        note = "I read the linked page."
                    else:
                        directive = (tool.seed_directive
                                     or "Answer my original question using this result.")
                        note = f"I used {action} to get this."
                    async for event in run_and_seed(action, args, note, directive):
                        if "__seeded__" not in event:
                            yield event
            # action == "answer" (or an unavailable/unknown tool): nothing seeded,
            # the loop below answers directly from the model's own knowledge.
            # No tool is going to run, so rebuild the prompt without the tool
            # protocol and the tool list: it is ~1500 tokens of a 4096-token
            # window telling the model to reply with a single JSON object, which
            # is exactly the reply a request for a C++ program came back with.
            # Retrieval is re-sized against the bigger window as a side effect.
            if answer_routed and not scratch:
                before = messages_tokens(base)
                base, dropped_prose = self.build_base(
                    history, user_message, reserve, tools=False,
                    extra_context=extra_context,
                    active_code_task=bool(brief and task.is_code_task()))
                prose_lane = True
                # Only re-report a trim that is still a trim: the UI renders
                # this as "trimmed N old messages", and N=0 is not a trim.
                if dropped_prose and dropped_prose != dropped:
                    yield {"type": "context", "dropped": dropped_prose,
                           "tokens": messages_tokens(base)}
                ev = detail(f"answer lane: prose prompt, no tools ({before} -> "
                            f"{messages_tokens(base)} prompt tokens, {dropped} -> "
                            f"{dropped_prose} history messages dropped)")
                if ev:
                    yield ev

            # Incremental reasoning: if the question is a hard analytical one and
            # nothing was seeded (a pure "answer" that isn't code), decompose it
            # and work through it step by step from a bounded, growing set of
            # conclusions, then stream the synthesis. This keeps the working set
            # small on 8GB and lets a 3B reason in depth by taking its time.
            if (not scratch and action == "answer"
                    and self.config.incremental_reasoning
                    and not is_code_request(user_message)
                    and is_reasoning_question(user_message)):
                yield {"type": "phase", "label": "planning the approach"}
                steps = await self.plan_steps(user_message)
                if len(steps) >= 2:
                    plan_lines = "; ".join(f"{i}) {st}" for i, st in enumerate(steps, 1))
                    yield {"type": "notice", "info": True,
                           "message": f"plan ({len(steps)} steps): {plan_lines[:400]}"}
                    notes: list[str] = []
                    for i, sub in enumerate(steps, 1):
                        if cancel is not None and cancel.is_set():
                            yield {"type": "cancelled", "step": i, "trace": trace}
                            return
                        yield {"type": "phase", "label": f"reasoning step {i}/{len(steps)}"}
                        # Stream each step live so thinking is never a frozen
                        # label: the user sees tokens appear as the model works.
                        yield {"type": "reason_step", "step": i, "total": len(steps), "label": sub[:120]}
                        notes_text = "\n".join(notes[-6:]) if notes else "(nothing yet)"
                        step_messages = [
                            {"role": "system", "content":
                                "You are working through a hard question one step at a time. "
                                "Use the findings so far, address only the current step, and "
                                "reply with a short concrete conclusion in at most 4 sentences."},
                            {"role": "user", "content":
                                f"Question: {user_message[:600]}\n\nFindings so far:\n{notes_text}"
                                f"\n\nCurrent step: {sub}\n\nYour conclusion:"},
                        ]
                        conclusion = ""
                        rstats = GenerationStats()
                        rstream = self.client.stream(step_messages, self.config.reasoning_tokens, 0.0, rstats, kind="reasoning")
                        started_step = time.time()
                        try:
                            async for tok in rstream:
                                if cancel is not None and cancel.is_set():
                                    break
                                conclusion += tok
                                yield {"type": "reason_token", "step": i, "token": tok}
                                # Per-step wall-clock cap: keep what streamed and
                                # move on rather than letting one step wedge.
                                if time.time() - started_step > self.config.reasoning_step_timeout:
                                    yield {"type": "notice", "message":
                                           f"step {i} taking long; moving on with partial", "info": False}
                                    break
                        except Exception:
                            # Streaming failed; fall back to a resilient non-stream.
                            if not strip_reasoning(conclusion).strip():
                                conclusion = await self.reason_step(user_message, notes, sub)
                        finally:
                            await rstream.aclose()
                        account(rstats)
                        conclusion = strip_reasoning(conclusion).strip()
                        yield {"type": "reason_done", "step": i, "conclusion": conclusion[:200]}
                        if conclusion:
                            notes.append(f"{i}. {sub}: {conclusion[:300]}")
                            self.partial_add(conversation_id, f"## Step {i}: {sub}\n{conclusion}")
                    # Synthesise the final answer from the conclusions, streamed.
                    yield {"type": "phase", "label": "writing the answer"}
                    joined = "\n".join(notes)
                    final_messages = [
                        {"role": "system", "content": self.config.system_prompt_with_identity},
                        {"role": "user", "content":
                            f"{user_message}\n\nYou worked through this and reached these "
                            f"conclusions:\n{joined}\n\nNow give the complete final answer "
                            "in plain text, drawing them together. Do not number the steps."},
                    ]
                    answer_buf = ""
                    fstats = GenerationStats()
                    fstream = self.client.stream(final_messages, reserve, temperature, fstats, kind=gen_kind)
                    try:
                        async for tok in fstream:
                            if cancel is not None and cancel.is_set():
                                break
                            answer_buf += tok
                            yield {"type": "token", "token": tok, "step": len(steps)}
                    except Exception:
                        # Fall back to a non-streaming resilient synthesis.
                        answer_buf = await self.resilient_complete(
                            final_messages, reserve, temperature)
                    finally:
                        await fstream.aclose()
                    account(fstats)
                    answer = strip_reasoning(answer_buf).strip()
                    if not answer:
                        answer = "Here is what I worked out:\n" + joined
                    yield done(answer, len(steps))
                    return

        for step in range(1, self.config.agent_max_steps + 1):
            if cancel is not None and cancel.is_set():
                yield {"type": "cancelled", "step": step, "trace": trace}
                return

            messages, condensed = self.assemble(base, scratch, reserve)
            # quiet=True: payload() runs the same calculation for real and logs
            # the clamp there; this call is only so the trace and the truncation
            # note can quote the number that will actually apply.
            effective_reserve = self.client.reply_budget(messages, reserve, quiet=True)
            yield {"type": "step", "step": step, "max_steps": self.config.agent_max_steps,
                   "prompt_tokens": messages_tokens(messages), "condensed": condensed,
                   "reply_budget": effective_reserve, "requested_budget": reserve}
            self.trace_prompt(f"step-{step}", messages, generation_id=generation_id,
                              lane=("prose" if prose_lane else "tools"),
                              task=task.summary(), artifact_version=task.artifact_version,
                              reply_budget=effective_reserve, requested_budget=reserve,
                              gen_kind=gen_kind, temperature=(temperature
                                                              if temperature is not None
                                                              else self.config.tool_temperature),
                              condensed=condensed, history_dropped=dropped,
                              continuation=False)
            ev = detail(f"step {step}: prompt {messages_tokens(messages)} tokens, "
                        f"reply budget {effective_reserve}"
                        + (f" (asked for {reserve}, clamped to fit the "
                           f"{self.config.context_size}-token context)"
                           if effective_reserve != reserve else "")
                        + f", {len(scratch)} scratch turns"
                        + (f", condensed {condensed}" if condensed else ""))
            if ev:
                yield ev

            buffer = ""
            cancelled = False
            stats = GenerationStats()
            # Tool-selection steps want deterministic JSON. Only the answer the
            # user reads should get the configured temperature, and we do not
            # know which this is until it parses, so bias towards valid JSON and
            # let the final-answer pass below use the warmer setting.
            step_temperature = (
                self.config.tool_temperature if temperature is None else temperature
            )
            # Generate this step, retrying with a smaller budget on failure rather
            # than surfacing an error. A model-server OOM kill and watchdog restart
            # look like a dropped stream from here, so a shrink-and-retry both
            # rides out the restart and asks for a reply small enough to fit.
            gen_reserve = reserve
            ctx_reserve = reserve
            step_failed = False
            # Set by the retry loop when a readiness probe proves the backend
            # cannot generate, so the failure message below need not re-probe.
            backend_dead = False
            for attempt in range(self.config.resilient_retries + 1):
                buffer = ""
                stats = GenerationStats()
                # On a retry, re-assemble with a much larger reserve, which
                # collapses the *prompt* budget and trims the trace hard. The
                # failure on 8GB is prefill of an oversized prompt (a stall, no
                # first token), so the input is the lever, not the reply length.
                # Each attempt cuts the prompt to roughly half of the previous,
                # so the three attempts are genuinely distinct rather than
                # bouncing off the reply-token floor.
                if attempt > 0:
                    # Leave only ~attempt/(attempt+1) of the window as reserve,
                    # i.e. cut the prompt to about 1/2, 1/3, 1/4 ... of the
                    # context on successive attempts. Monotonic and distinct.
                    ctx_reserve = min(self.config.context_size - self.config.min_max_tokens,
                                      int(self.config.context_size * (attempt / (attempt + 1))))
                    messages, _ = self.assemble(base, scratch, ctx_reserve)
                stream = self.client.stream(messages, gen_reserve, step_temperature, stats, kind=gen_kind)
                # Split the model's <think> reasoning from its answer as it
                # streams, so the reasoning shows in its own visible thinking
                # area instead of being hidden or dumped raw into the answer.
                splitter = ThinkSplitter()
                try:
                    async for token in stream:
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break
                        buffer += token
                        for kind, piece in splitter.feed(token):
                            if not piece:
                                continue
                            if kind == "think":
                                yield {"type": "think_token", "token": piece, "step": step}
                            else:
                                yield {"type": "token", "token": piece, "step": step}
                        visible = strip_reasoning(buffer).lstrip()
                        if visible.startswith(("{", "```")) and parse_tool_call(buffer, known):
                            break
                    step_failed = False
                    break
                except Exception as exc:
                    await stream.aclose()
                    log(f"generation step {step} attempt {attempt + 1} failed: {exc}",
                        logging.DEBUG)
                    # If usable text already streamed, keep it rather than redoing
                    # work; the loop below can act on a partial answer or call.
                    if strip_reasoning(buffer).strip():
                        step_failed = False
                        break
                    if attempt >= self.config.resilient_retries:
                        step_failed = True
                        break
                    reason = self.client.classify_error(exc)
                    # Ask whether the backend can generate AT ALL before deciding
                    # to retry, and ACT on the answer. mlx_lm.server runs
                    # generation on one thread; when that thread dies the process
                    # keeps answering 200 on /v1/models while every completion
                    # hangs. Shrinking the prompt cannot help that, and spending
                    # the remaining attempts on it is minutes of dead air (three
                    # attempts of a 60s stall plus a 40s probe each). The probe
                    # also gives the manager's watchdog time to restart a backend
                    # that is coming back, so a genuine OOM-and-restart still
                    # rides out transparently.
                    backend_dead = not await self.client.wait_until_ready(
                        timeout=self.config.ready_wait_timeout)
                    if backend_dead:
                        log(f"generation step {step}: model server still cannot generate "
                            f"after {self.config.ready_wait_timeout}s; abandoning the "
                            "remaining retries rather than stalling on a dead backend.",
                            logging.WARNING)
                        step_failed = True
                        break
                    # It is generating again, so the prompt is the remaining
                    # suspect. Shrink the reply a little too, but the prompt cut
                    # above is the real lever; report that, since it is what
                    # actually changes between attempts.
                    gen_reserve = max(self.config.min_max_tokens, int(gen_reserve * 0.75))
                    yield {"type": "notice", "step": step,
                           "message": f"{reason}; the model server is answering again, "
                                      f"cutting the prompt hard and retrying (attempt "
                                      f"{attempt + 2}). Work so far is saved."}
                    continue
                finally:
                    # Breaking out early leaves the HTTP response open until the
                    # generator is collected, which on a local server means a
                    # socket per abandoned step.
                    await stream.aclose()

            if step_failed:
                # Every retry failed. Distinguish a genuine memory limit from a
                # backend that is up but not generating (e.g. the mlx-lm
                # generation thread crashed). When the retry loop already proved
                # the latter, reuse that verdict instead of paying for another
                # probe. An honest message points at the real fix.
                can_generate = (False if backend_dead
                                else await self.client.wait_until_ready(timeout=8.0))
                if can_generate:
                    note = ("I ran low on memory before I could finish this in one "
                            "pass. Try a smaller or more specific request, or raise "
                            "the RAM headroom.")
                else:
                    note = ("The model server is up but not generating — this is a "
                            "backend error, not a memory limit. The watchdog should "
                            "have restarted it; if this persists it has given up, so "
                            "check logs/model_server.log and restart the model "
                            "(Settings → Restart model).")
                yield done(self.salvage(conversation_id, note), step, truncated=True)
                return

            account(stats)
            yield {"type": "usage", "step": step, **stats.as_event()}

            if cancelled:
                yield {"type": "cancelled", "step": step, "partial": buffer.strip(), "trace": trace}
                return

            call = parse_tool_call(buffer, known)

            # Canonicalise the tool name BEFORE any policy check runs on it.
            # parse_tool_call honours an explicit {"tool": ...} key whatever the
            # name, so a model that writes the TARGET in the name slot --
            # {"tool": "https://github.com/libusb/libusb"} -- produced a call
            # whose name was in no allow-list and no deny-list. It slipped past
            # the answer-lane lookup gate directly below (which matches on
            # LOOKUP_TOOLS) and then reached the registry, which replied with its
            # entire catalogue: a spent step, a failed tool call in the DB, and
            # 286 characters of "Available: ..." fed back into a 4096-token
            # context as a TOOL RESULT. Repair or reject once, here, so every
            # check downstream is looking at a real tool name.
            if call is not None:
                repaired = self.repair_tool_call(call[0], call[1], known)
                if repaired is None:
                    bad_names += 1
                    log(f"step {step}: model asked for an unregistered tool "
                        f"{call[0]!r}; correcting without dispatching it.",
                        logging.WARNING)
                    if bad_names > 2:
                        forced = await self.forced_prose_answer(
                            history, user_message, reserve, temperature, task)
                        yield done(forced or "I could not produce an answer for that. "
                                             "Try rephrasing it, or ask me to continue.",
                                   step)
                        return
                    yield {"type": "notice", "step": step, "info": True,
                           "message": f"{call[0]!r} is not one of my tools; asking again"}
                    # The tool lane's system prompt already carries every tool
                    # spec, so re-listing the catalogue here would only spend
                    # context on something the model has already been told.
                    scratch.append({"role": "assistant", "content": "(invalid tool name)"})
                    scratch.append({"role": "user", "content":
                                    f"{call[0]!r} is not a tool. Use one of the tool names "
                                    "from your instructions, exactly as written, and put "
                                    "the target in args. Or answer in plain text."})
                    continue
                if repaired != call:
                    log(f"step {step}: repaired tool call {call[0]!r} -> "
                        f"{repaired[0]!r}", logging.INFO)
                    ev = detail(f"repaired tool name: {call[0]} -> {repaired[0]}")
                    if ev:
                        yield ev
                call = repaired

            # On an answer-routed turn, a self-issued network lookup is allowed
            # ONCE, and only when the question is genuinely time-sensitive (the
            # same signal the router searches on). That lets the model fetch
            # fresh facts when it truly needs them while still blocking the
            # reflexive "search everything" behaviour for general knowledge.
            # parse_tool_call honors an explicit "tool" key regardless of the
            # allowed set, so the gate must be here, at execution.
            if (call is not None and answer_routed and call[0] in LOOKUP_TOOLS
                    and is_time_sensitive(user_message) and not lookup_used
                    and forced_lane not in ("answer", "kb")):
                lookup_used = True
                yield {"type": "notice", "info": True,
                       "message": "the question looks time-sensitive; allowing one lookup"}
                # fall through to normal tool execution below
            elif call is not None and answer_routed and call[0] in LOOKUP_TOOLS:
                if not answered_retry:
                    answered_retry = True
                    yield {"type": "notice", "info": True,
                           "message": "answering from my own knowledge (no lookup needed)"}
                    scratch.append({"role": "user", "content":
                        "Answer the question directly from your own knowledge. "
                        "Do NOT call any tool and do NOT output JSON; just give the answer."})
                    continue
                # It insisted again: force a plain, tool-free answer on a prompt
                # that never mentions tools in the first place.
                direct = await self.forced_prose_answer(
                    history, user_message, reserve, temperature, task)
                yield done(direct or "I don't have enough to answer that confidently.", step)
                return

            if call is None:
                answer = strip_reasoning(buffer).strip()
                # A reply that is nothing but an unfilled tool call ("{}") is not
                # an answer. parse_tool_call refuses to execute it and this branch
                # used to hand the raw text to the user as the reply -- and then
                # store it, so the next turn had "{}" in context as a worked
                # example and produced it again in two seconds.
                degenerate = bool(answer) and is_degenerate_tool_call(answer)
                if degenerate:
                    log(f"discarded a degenerate JSON reply at step {step} "
                        f"({'prose' if prose_lane else 'tool'} lane): {answer[:80]!r}",
                        logging.WARNING)
                    yield {"type": "notice", "step": step, "info": True,
                           "message": "that reply was an empty tool call, not an answer; "
                                      "asking again for plain text"}
                    answer = ""
                if answer:
                    # Last gate before the user sees it: does this answer still
                    # belong to the task? A strong contradiction (every code
                    # block in a language the task is not) gets one corrective
                    # regeneration rather than being returned.
                    drift_stats = GenerationStats()
                    fixed, drift_note = await self.redirect_drift(
                        task, answer, user_message, history, reserve, temperature,
                        drift_stats)
                    cost = stats
                    if drift_note:
                        yield {"type": "notice", "info": True, "message": drift_note}
                        # The regeneration is a real model call: account for it,
                        # or the token footer under the answer understates the
                        # turn, and report the cut-off state of the text the user
                        # actually gets rather than of the discarded one.
                        account(drift_stats)
                        yield {"type": "usage", "step": step, **drift_stats.as_event()}
                        answer, cost = fixed, drift_stats
                    yield done(note_if_cut(answer, cost), step)
                    return
                # An empty reply -- or an unfilled tool call -- is a hiccup, not
                # an answer. Nudge once. The rejected text is NEVER appended to
                # the trace: putting it there is what teaches the model to repeat
                # it. A placeholder keeps the turn structure without the example.
                if nudges == 0 and step < self.config.agent_max_steps:
                    nudges += 1
                    scratch.append({"role": "assistant", "content": "(no answer)"})
                    scratch.append({
                        "role": "user",
                        "content": ("That reply was an empty JSON object, not an answer. "
                                    "Answer the question in plain text. Do not output JSON."
                                    if degenerate else
                                    "That reply was empty. Answer the question in plain text, "
                                    "or call exactly one tool as a JSON object."),
                    })
                    continue
                # Out of nudges: one forced tool-free pass beats handing back a
                # blank turn, which is what yielding "" here did.
                forced = await self.forced_prose_answer(
                    history, user_message, reserve, temperature, task)
                yield done(forced or "I could not produce an answer for that. Try "
                                     "rephrasing it, or ask me to continue.", step)
                return

            name, args = call

            if name == "final_answer":
                answer = str(args.get("answer") or "").strip()
                if answer:
                    # No tool_call event: final_answer is how the loop exits, not
                    # an action worth a trace node. Emitting it rendered a stray
                    # "final_answer:" label above the reply.
                    yield done(note_if_cut(answer, stats), step)
                    return
                scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
                scratch.append({
                    "role": "user",
                    "content": "final_answer needs a non-empty answer argument. "
                               "Reply again with the full answer.",
                })
                continue

            if name == "delegate_task":
                # Handled here rather than in the registry because running a
                # child agent means awaiting a coroutine on this event loop, and
                # a registry handler is synchronous.
                # Named sub_task, not task: `task` in this scope is the
                # conversation's active TaskState.
                sub_task = str(args.get("task") or "").strip()
                caps = str(args.get("capabilities") or "")
                if not sub_task:
                    scratch.append({"role": "assistant", "content": "(empty delegation)"})
                    scratch.append({"role": "user", "content":
                                    "delegate_task needs a self-contained task. Give "
                                    "the helper the full instruction, or do it yourself."})
                    continue
                if delegated >= self.config.subagent_max_per_turn:
                    result = (f"You have already delegated "
                              f"{self.config.subagent_max_per_turn} times this turn. "
                              "Finish with what you have.")
                    error = None
                else:
                    delegated += 1
                    yield {"type": "tool_call", "name": name, "args": args, "step": step}
                    yield {"type": "phase", "label": f"helper {delegated}: {sub_task[:60]}"}
                    try:
                        async with self.subagents:
                            result = await asyncio.wait_for(
                                self.run_subagent(sub_task, caps, conversation_id),
                                timeout=self.config.agent_run_timeout)
                        error = None
                    except asyncio.TimeoutError:
                        result = (f"(the helper ran out of time after "
                                  f"{self.config.agent_run_timeout:.0f}s)")
                        error = result
                    except Exception as exc:
                        result = f"(the helper failed: {type(exc).__name__}: {exc})"
                        error = result
                        log(f"subagent failed: {exc}", logging.WARNING)
                    if not result:
                        result = "(the helper came back with nothing)"
                    trace.append({"name": name, "args": args,
                                  "result": str(result)[:1000], "error": error})
                    yield {"type": "tool_result", "name": name, "result": result,
                           "error": error, "step": step}
                scratch.append({"role": "assistant",
                                "content": strip_reasoning(buffer).strip()})
                scratch.append({"role": "user", "content":
                                f"HELPER RESULT:\n{result}\n\nUse this to answer the "
                                "original question. You cannot see the work it did."})
                continue

            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            yield {"type": "tool_call", "name": name, "args": args, "step": step}

            if signature in seen_calls:
                result = (
                    f"You already called {name} with these arguments and received a result. "
                    "Do not repeat it. Answer the user now with what you have, or call a "
                    "different tool."
                )
                error = None
            else:
                seen_calls.append(signature)
                result, error = await asyncio.to_thread(
                    self.registry.call, name, args, conversation_id
                )

            budget = self._tool_budget(reserve)
            result_for_model, was_summarised = await self.compress_tool_result(name, result, budget)

            trace.append({"name": name, "args": args, "result": result[:1000], "error": error})
            yield {"type": "tool_result", "name": name, "result": result, "error": error,
                   "step": step, "context_chars": len(result_for_model),
                   "summarised": was_summarised}
            ev = detail(f"tool {name}: {len(result)} chars returned, "
                        f"{len(result_for_model)} into context"
                        + (" (summarised)" if was_summarised else "")
                        + (f", error: {error}" if error else ""))
            if ev:
                yield ev

            scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
            scratch.append({
                "role": "user",
                "content": f"TOOL RESULT [{name}]:\n{result_for_model}\n\n"
                           "Use this to answer the original question, or call one more tool "
                           "if you genuinely still need it.",
            })

        # Ordinary step budget exhausted without a final answer. Rather than
        # stopping, keep going in bounded batches: collapse the work so far into a
        # compact running summary (so memory stays flat and an 8GB machine does
        # not OOM), then grant another batch of steps, up to hard_step_cap. This
        # is the "slow down but do not stop" path for a task too big for one pass.
        extra_batches = 0
        while (self.config.agent_max_steps * (extra_batches + 1) < self.config.hard_step_cap
               and (cancel is None or not cancel.is_set())):
            extra_batches += 1
            summary = self.running_summary(scratch)
            # Reset the working set to just the summary: constant memory regardless
            # of how much has already happened.
            scratch = [{
                "role": "user",
                "content": f"PROGRESS SO FAR (continue the task, do not restart):\n{summary}\n\n"
                           "Keep going one step at a time. Answer in plain text when done, "
                           "or call one tool as JSON to make progress.",
            }]
            yield {"type": "notice", "step": self.config.agent_max_steps * extra_batches,
                   "message": "task is large; continuing step by step from a summary"}

            batch_progress = False
            for extra in range(1, self.config.agent_max_steps + 1):
                step = self.config.agent_max_steps * extra_batches + extra
                if cancel is not None and cancel.is_set():
                    yield {"type": "cancelled", "step": step, "trace": trace}
                    return
                messages, _ = self.assemble(base, scratch, reserve)
                yield {"type": "step", "step": step, "max_steps": self.config.hard_step_cap,
                       "prompt_tokens": messages_tokens(messages)}
                buffer = await self.resilient_complete(messages, reserve, temperature) or ""
                call = parse_tool_call(buffer, known)
                if call is None:
                    answer = strip_reasoning(buffer).strip()
                    if answer and is_degenerate_tool_call(answer):
                        answer = ""      # an unfilled call, not an answer
                    if answer:
                        yield done(answer, step, truncated=True)
                        return
                    continue
                name, args = call
                repaired = self.repair_tool_call(name, args, known)
                if repaired is None:
                    # Same guard as the main loop: correct, never dispatch. Not
                    # forward motion either, so a batch of these still stalls out
                    # and salvages instead of grinding to the hard cap.
                    log(f"step {step}: model asked for an unregistered tool {name!r}; "
                        "correcting without dispatching it.", logging.WARNING)
                    scratch.append({"role": "assistant", "content": "(invalid tool name)"})
                    scratch.append({"role": "user", "content":
                                    f"{name!r} is not a tool. Use a tool name from your "
                                    "instructions exactly as written, or answer in plain text."})
                    continue
                name, args = repaired
                batch_progress = True  # a real tool call is forward motion
                if name == "final_answer":
                    answer = str(args.get("answer") or "").strip()
                    if answer:
                        yield done(answer, step, truncated=True)
                        return
                    continue
                signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
                yield {"type": "tool_call", "name": name, "args": args, "step": step}
                if signature in seen_calls:
                    result, error = ("Already ran that; use the result you have or try "
                                     "another tool.", None)
                else:
                    seen_calls.append(signature)
                    result, error = await asyncio.to_thread(
                        self.registry.call, name, args, conversation_id
                    )
                budget = self._tool_budget(reserve)
                result_for_model, _ = await self.compress_tool_result(name, result, budget)
                trace.append({"name": name, "args": args, "result": result[:1000], "error": error})
                yield {"type": "tool_result", "name": name, "result": result,
                       "error": error, "step": step}
                scratch.append({"role": "assistant", "content": strip_reasoning(buffer).strip()})
                scratch.append({"role": "user",
                                "content": f"TOOL RESULT [{name}]:\n{result_for_model}"})
                self.partial_add(conversation_id, f"Tool {name}: {str(result)[:500]}")

            if not batch_progress:
                # A whole batch produced no answer and no tool call — almost
                # always repeated stalls. Continuing would only stretch a stall
                # into minutes (the 789-second grind). Stop and salvage instead.
                yield {"type": "notice", "info": True,
                       "message": "no progress in the last batch; wrapping up with what I have"}
                break

        # Reached the hard cap, or was cancelled, or a batch stalled out. Force one
        # plain answer, never an error, from the compact summary so the reply
        # always closes cleanly.
        summary = self.running_summary(scratch)
        final_messages = [
            {"role": "system", "content": self.config.system_prompt_with_identity},
            {"role": "user", "content":
                f"{user_message}\n\nWork so far:\n{summary}\n\n"
                "Give your best final answer now in plain text. Do not call any tool."},
        ]
        answer = await self.resilient_complete(final_messages, reserve, temperature)
        answer = strip_reasoning(answer).strip()
        if is_degenerate_tool_call(answer):
            answer = ""
        if not answer:
            # Synthesis itself could not run — hand back the saved work from disk
            # (falling back to the in-memory summary) so the turn still delivers.
            answer = self.salvage(
                conversation_id,
                "I reached the step limit before finishing in one pass.")
            if answer.strip() == "I reached the step limit before finishing in one pass.":
                answer += "\n\nHere is as far as I got:\n" + summary
        yield done(answer, self.config.hard_step_cap, truncated=True)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Agent',
]
