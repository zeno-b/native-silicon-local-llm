"""Prompt assembly, reasoning splitting and tool-call parsing.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403


def note_prefix(prompt: str, lane: str = "tools") -> None:
    """Track the cached prompt prefix, per lane.

    There are two stable prefixes, not one: the tool-calling prompt and the
    prose-only prompt used on an answer-routed turn. Keying the check per lane
    means ordinary alternation between them is not reported as the prefix
    changing (and does not bump a generation counter every turn) -- a real
    change is an edit to the system prompt or the tool set within one lane.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    lanes = PREFIX_STATE.setdefault("lanes", {})
    state = lanes.setdefault(lane, {"hash": None, "changed_at": None, "generation": 0})
    if state["hash"] == digest:
        return
    first = state["hash"] is None
    state["hash"] = digest
    state["changed_at"] = iso(utc_now())
    state["generation"] += 1
    if lane == "tools":
        # Mirror the tool lane at the top level: /api telemetry and the UI have
        # read those keys since before there was a second lane.
        PREFIX_STATE["hash"] = state["hash"]
        PREFIX_STATE["changed_at"] = state["changed_at"]
        PREFIX_STATE["generation"] = state["generation"]
    if not first:
        log(
            f"Agent prompt prefix changed ({lane} lane: system prompt or tool set). "
            "Any cached prefix for that lane is now invalid and the next few steps "
            "will re-prefill in full.",
            logging.WARNING,
        )


# NOTE on wording: this used to end "give your final answer", immediately after
# describing a pair of XML tags. A 3B model completed the symmetry and wrapped
# its reply in <final_answer> tags, or labelled it "Final answer:" -- measured at
# roughly one reply in eight. Saying "the answer itself" and forbidding a label
# costs a few tokens and removes the invitation. unwrap_answer still cleans the
# output, because prompt wording is a suggestion, not a guarantee.
REASONING_INSTRUCTION = (
    "\n\nThink before you answer. Begin your reply with your reasoning enclosed in "
    "<think> and </think> tags: work through the problem step by step, consider "
    "edge cases, and question your assumptions there. After the closing </think> "
    "tag, give the answer itself (or, if you are calling a tool, the single JSON "
    "object) with no heading, no label and no tags around it. Put ONLY reasoning "
    "inside the tags and never the tool JSON."
)


# The same instruction for the prose lane, with the tool-call clause removed:
# mentioning a "single JSON object" at the very end of the prompt is the most
# available thing in the model's context when it starts writing.
PROSE_REASONING_INSTRUCTION = (
    "\n\nThink before you answer. Begin your reply with your reasoning enclosed in "
    "<think> and </think> tags: work through the problem step by step, consider "
    "edge cases, and question your assumptions there. After the closing </think> "
    "tag, give the answer itself in plain text, with no heading, no label and no "
    "tags around it. Put ONLY reasoning inside the tags."
)


SKILL_PREAMBLE = (
    "\n\nYOUR SKILLS (name and summary only). Call load_skill with a name to read "
    "the procedure BEFORE doing that kind of task; do not guess one you have a "
    "skill for.\n"
)


def project_block(registry: ToolRegistry) -> str:
    """Project instruction files, or "". Safe to call with anything."""
    try:
        return registry.project_context()
    except Exception as exc:      # a hostile or unreadable file cannot break a turn
        log(f"Project context unavailable: {exc}", logging.WARNING)
        return ""


def skills_block(registry: ToolRegistry) -> str:
    """The catalogue lines for the prompt, or "" when there is nothing to add.

    This is the visible half of progressive disclosure and the only part the
    prompt pays for. Returns "" on a fresh install, so a machine with no skills
    spends not one token on the feature.

    Takes no query on purpose: the block is part of the system prompt, and a
    block that varied with the message would change the prefix every turn and
    cost a full re-prefill. Relevance selection lives in search_skills, where it
    is paid for only when used.
    """
    config = getattr(registry, "config", None)
    library = getattr(registry, "skills", None)
    if library is None or config is None or not getattr(config, "skills_enabled", False):
        return ""
    if "load_skill" not in registry.names():
        return ""          # tools disabled: advertising skills would be a lie
    try:
        catalogue = library.catalogue(limit=config.skills_in_prompt)
        hidden = library.overflowing(config.skills_in_prompt)
    except Exception as exc:      # a broken skill file must never break a turn
        log(f"Skill catalogue unavailable: {exc}", logging.WARNING)
        return ""
    if not catalogue:
        return ""
    block = SKILL_PREAMBLE + catalogue
    if hidden:
        block += (f"\n({hidden} more not listed: use search_skills to find them.)")
    return block


def build_agent_system_prompt(base_prompt: str, registry: ToolRegistry,
                              reasoning: bool = False) -> str:
    lines = []
    for tool in registry.specs():
        params = ", ".join(
            f"{name} ({desc})" for name, desc in tool["parameters"].items()
        ) or "no arguments"
        required = ", ".join(tool["required"]) or "none"
        lines.append(f"- {tool['name']}: {tool['description']}\n  args: {params}\n  required: {required}")
    prompt = f"{base_prompt}\n\n{TOOL_PROTOCOL}" + "\n".join(lines)
    prompt += project_block(registry)
    prompt += skills_block(registry)
    if reasoning:
        prompt += REASONING_INSTRUCTION
    note_prefix(prompt, lane="tools")
    return prompt


def build_plain_system_prompt(base_prompt: str, reasoning: bool = False,
                              registry: ToolRegistry | None = None) -> str:
    """The prompt for a turn that will not call a tool.

    The tool protocol plus every tool spec is ~1500 tokens of a 4096-token
    window, and its first instruction is "reply with a single JSON object and
    nothing else". Once routing has decided the turn needs no tool, that text
    only costs context and invites the failure it describes: a request to write
    a C++ program answered with "{}". This lane says nothing about tools at all.
    """
    prompt = base_prompt
    # Skills belong on THIS lane most of all. The prose lane is where the model
    # answers from its own knowledge, which is exactly where a written-down
    # procedure beats a 3B model's recall. The registry is optional so callers
    # that have no tools at all still work.
    if registry is not None:
        prompt += project_block(registry)
        block = skills_block(registry)
        if block:
            # One extra sentence, because this lane is told nothing about tools:
            # without it the model has a list it has no way to act on.
            prompt += block + (
                "\nTo read one, reply with exactly this and nothing else: "
                '{"tool": "load_skill", "args": {"name": "<skill-name>"}}'
            )
    if reasoning:
        prompt += PROSE_REASONING_INSTRUCTION
    note_prefix(prompt, lane="prose")
    return prompt


def extract_json_object(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply.

    Small models wrap the object in prose or a code fence at random, so scanning
    for balanced braces beats expecting a clean response.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:index + 1])
                start = -1
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


THINK_BLOCK = re.compile(r"(?is)<(think|thinking|reasoning)>.*?</\1>")
OPEN_THINK = re.compile(r"(?is)<(think|thinking|reasoning)>.*\Z")


class ThinkSplitter:
    """Streaming splitter that separates a model's <think> reasoning from its
    answer as tokens arrive.

    A reasoning model emits <think>...</think> before its answer. We want the
    reasoning shown live in its own visible "thinking" area, not stripped and
    hidden and not dumped raw into the answer. Tokens can split a tag across
    chunk boundaries, so a partial tag at the end of a feed is held over to the
    next one. feed() returns a list of ("think"|"answer", text) pieces.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.carry = ""

    @staticmethod
    def _prefix_tail(data: str, tag: str) -> int:
        """Length of a trailing slice of data that is a proper prefix of tag.

        So "...</thi" holds back 4 chars until the rest of </think> arrives.
        """
        for size in range(min(len(tag) - 1, len(data)), 0, -1):
            if tag.startswith(data[-size:]):
                return size
        return 0

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        data = self.carry + (text or "")
        self.carry = ""
        while data:
            if self.in_think:
                end = data.find(self.CLOSE)
                if end == -1:
                    keep = self._prefix_tail(data, self.CLOSE)
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("think", data))
                    data = ""
                else:
                    if end:
                        out.append(("think", data[:end]))
                    data = data[end + len(self.CLOSE):]
                    self.in_think = False
            else:
                start = data.find(self.OPEN)
                if start == -1:
                    keep = self._prefix_tail(data, self.OPEN)
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("answer", data))
                    data = ""
                else:
                    if start:
                        out.append(("answer", data[:start]))
                    data = data[start + len(self.OPEN):]
                    self.in_think = True
        return out


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's chain-of-thought block.

    Qwen3.5 and later emit <think>...</think> before the answer. extract_json_object
    scans for the first balanced JSON object anywhere in the reply, so reasoning
    that talks through candidate arguments gets parsed as the tool call itself.
    An unterminated block is stripped too, because mid-stream the closing tag
    has not arrived yet and the agent tests the buffer on every token.
    """
    if not text or "<" not in text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = OPEN_THINK.sub("", text)
    return text.strip()


def parse_tool_call(text: str, known: set[str] | None = None) -> tuple[str, dict] | None:
    """Return (tool name, args) if the reply is a tool call, else None.

    `known` is the set of registered tool names. When it is supplied, a JSON
    object whose name is not a real tool is only treated as a call if it used
    the explicit "tool" key. That stops a final answer that happens to contain
    JSON (a config snippet, a parsed record) from being executed as a tool call.
    """
    parsed = extract_json_object(strip_reasoning(text))
    if not parsed:
        return None
    explicit = "tool" in parsed or "tool_name" in parsed
    name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    name = name.strip()
    if known is not None and name not in known and not explicit:
        return None
    args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"query": args}
    if not isinstance(args, dict):
        args = {}
    return name, args


# Keys that only ever belong to the tool-call protocol. A reply made up of
# nothing but these (or of nothing at all) is an unfilled tool call.
_CALL_ONLY_KEYS = frozenset(
    {"tool", "tool_name", "name", "args", "arguments", "parameters", "action"})


def is_degenerate_tool_call(text: str) -> bool:
    """True if the reply is a malformed tool call rather than an answer.

    "{}" -- or {"args": {}}, or {"action": "answer"} echoed back from the router
    -- is a tool call the model never filled in. parse_tool_call rightly refuses
    to execute it, and the loop then treated the raw text as the reply: a request
    for a C++ program that arrived as "{}". A JSON object carrying real content
    keys is somebody asking for JSON, so it is left alone.
    """
    visible = strip_reasoning(text or "").strip()
    if not visible:
        return False
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", visible, re.S)
    if fenced:
        visible = fenced.group(1).strip()
    if not (visible.startswith("{") and visible.endswith("}")):
        return False
    try:
        parsed = json.loads(visible)     # the WHOLE reply must be one object
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if parse_tool_call(visible) is not None:
        return False                     # a real call: the caller decides
    return all(str(k).lower() in _CALL_ONLY_KEYS for k in parsed)


@dataclass
class GenerationStats:
    """What one model call actually cost.

    prompt_tokens is the number that matters most on this hardware: an agent
    re-sends its whole prompt every step, so prefill, not decode, is where a
    multi-step run spends its time. Watching this climb step over step within a
    single run is how you see the quadratic.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    from_server: bool = False
    # Why generation stopped: "stop" (the model finished) or "length" (it hit the
    # reply budget and was CUT OFF). Nothing read this before, so a truncated
    # answer -- a program ending mid-function -- was shown as if it were complete.
    finish_reason: str = ""

    @property
    def decode_tps(self) -> float:
        decode_ms = max(1.0, self.total_ms - self.ttft_ms)
        return self.completion_tokens / (decode_ms / 1000.0)

    def as_event(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "decode_tps": round(self.decode_tps, 2),
            "estimated": not self.from_server,
        }


_CONNECTIVITY: dict = {"online": True, "checked_at": 0.0}


async def has_internet(recheck_after: float = 30.0) -> bool:
    """Best-effort connectivity check, cached briefly.

    Used to decide whether lookups are even possible. When offline the agent
    answers from its own knowledge instead of attempting a search that would
    fail slowly. The result is cached for recheck_after seconds so it costs at
    most one tiny request per window. Any failure is read as offline.
    """
    now = time.time()
    if now - _CONNECTIVITY["checked_at"] < recheck_after:
        return _CONNECTIVITY["online"]
    online = False
    try:
        import httpx
        # A couple of reliable, lightweight endpoints; success on either is enough.
        async with httpx.AsyncClient(timeout=3.0) as client:
            for url in ("https://1.1.1.1", "https://dns.google"):
                try:
                    resp = await client.head(url)
                    if resp.status_code < 500:
                        online = True
                        break
                except Exception:
                    continue
    except Exception:
        online = False
    _CONNECTIVITY.update(online=online, checked_at=now)
    return online



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'SKILL_PREAMBLE',
    'project_block',
    'skills_block',
    'GenerationStats',
    'OPEN_THINK',
    'REASONING_INSTRUCTION',
    'THINK_BLOCK',
    'ThinkSplitter',
    '_CONNECTIVITY',
    'build_agent_system_prompt',
    'extract_json_object',
    'is_degenerate_tool_call',
    '_CALL_ONLY_KEYS',
    'build_plain_system_prompt',
    'PROSE_REASONING_INSTRUCTION',
    'has_internet',
    'note_prefix',
    'parse_tool_call',
    'strip_reasoning',
]
