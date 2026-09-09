"""Message classifiers, chunking and routing heuristics.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import re

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403


def _is_trivial(text: str) -> bool:
    """True if the whole message is just a greeting or acknowledgement."""
    match = TRIVIAL_MESSAGE.match(text)
    # The pattern matches a leading greeting; require that only punctuation and
    # whitespace follow it, so "hi there" is trivial but "hi, fix this bug" is not.
    return bool(match and _TRIVIAL_TAIL.fullmatch(text[match.end():]))


def is_substantive(message: str) -> bool:
    """True if a message is a real question or request worth routing.

    Used by the chat handlers to keep chit-chat on the cheap plain-reply path
    while sending anything substantive through the model router.
    """
    text = (message or "").strip()
    if len(text) < 8 or _is_trivial(text):
        return False
    return True


def _last_user_turn(history: list[dict] | None) -> str | None:
    """The most recent user message in a history list.

    The router is given recent history, but this is also used to log or inspect
    the previous turn when resolving follow-ups like "look it up".
    """
    for turn in reversed(history or []):
        if turn.get("role") == "user" and (turn.get("content") or "").strip():
            return turn["content"]
    return None


# Requests to write or modify code. These are answered from the model's own
# knowledge and must never be routed to a web search: a code-generation prompt
# ("write a python script...") sent to a search engine returns tutorials at best
# and, as with "os recon" being read as OCR, irrelevant junk at worst. Detected
# deterministically so it also skips the router call entirely.
CODE_INTENT = re.compile(
    r"\b(write|create|generate|build|implement|code|program|fix|debug|"
    r"refactor|optimi[sz]e|complete|extend|port|convert|translate|add|modify|"
    r"update|rewrite|snippet)\b",
    re.I,
)
CODE_OBJECT = re.compile(
    r"\b(script|scripts|function|functions|program|programme|code|class|classes|"
    r"method|methods|module|snippet|regex|regexp|cli|parser|app|application|"
    r"component|query|loop|algorithm|algorithms|unit ?tests?|test|tests|api|"
    r"endpoint|schema|decorator|generator|command|one[- ]?liner)\b",
    re.I,
)
# Programming languages and runtimes worth treating as a code signal on their own
# when paired with a code verb.
CODE_LANGUAGE = re.compile(
    r"\b(python|py|javascript|js|typescript|ts|rust|go|golang|c\+\+|cpp|c#|java|"
    r"bash|shell|sh|zsh|ruby|php|perl|sql|html|css|react|node|node\.js|"
    r"swift|swiftui|kotlin|dart|flutter|scala|objective-c|objc|lua|r|matlab|"
    r"powershell|ps1|julia|haskell|elixir|clojure|solidity)\b",
    re.I,
)


# A code request that plausibly depends on external or current information, so
# the model should look it up before writing. Without this, "write a script
# using the latest X API" or "a recon script covering current techniques" would
# be answered from stale weights.
CODE_NEEDS_LOOKUP = re.compile(
    r"\b(latest|current|recent|newest|modern|up[- ]?to[- ]?date|today|this year|"
    r"as of|version|changelog|"
    # Security-research nouns: these move fast and benefit from current sources.
    r"recon|reconnaissance|exploit|exploits|vulnerabilit|cve|payload|"
    r"enumeration|privilege escalation|pentest|attack)\b",
    re.I,
)

# Time-sensitive signal: the same notion the router is told to search on (today's
# events, prices, scores, the latest version of something, a named current
# holder). Used to let a single self-issued search through on an answer-routed
# turn when the question genuinely needs fresh, external facts, while still
# blocking reflexive searches for general knowledge.
TIME_SENSITIVE = re.compile(
    r"\b(today|todays|tonight|yesterday|right now|currently|current|latest|"
    r"newest|recent|recently|this (?:week|month|year|morning|season)|"
    r"as of|so far this|"
    r"price|prices|cost|stock|shares|exchange rate|rate today|"
    r"news|headline|breaking|score|scores|standings|fixture|"
    r"weather|forecast|temperature|"
    r"release|released|version|update|changelog|"
    r"who is the (?:current|new)|latest version|out yet|release date)\b",
    re.I,
)


# Implicit chat feedback. A short reply that is essentially praise or a plain
# rejection is a rating of the previous answer: "good job" approves it as a
# training example, "no, wrong" marks it as a bad answer. Kept conservative so a
# substantive message that merely starts with "no" is not misread; longer
# messages must contain an explicit multi-word phrase.
_POS_LEAD = re.compile(
    r"^\s*(?:that'?s\s+)?(great|perfect|excellent|awesome|amazing|nice|good|"
    r"correct|exactly|right|yes+|yep|yeah|thanks?|thank\s+you|ty|helpful|"
    r"brilliant|love\s+it|works|spot\s+on)\b", re.I)
_NEG_LEAD = re.compile(
    r"^\s*(?:no+|nope|nah|wrong|incorrect|bad|false|not\s+(?:right|correct|quite|good))\b",
    re.I)
_POS_PHRASE = re.compile(
    r"\b(good\s+job|well\s+done|that'?s\s+(?:right|correct|perfect)|exactly\s+right|"
    r"perfect\s+answer|that\s+works|great\s+answer|that'?s\s+it)\b", re.I)
_NEG_PHRASE = re.compile(
    r"\b(that'?s\s+(?:wrong|incorrect|not\s+right|false)|wrong\s+answer|"
    r"you'?re\s+wrong|not\s+(?:what\s+i|correct)|that'?s\s+not\s+it|bad\s+answer)\b",
    re.I)
# Guards: leading tokens that look negative/positive but are not feedback.
_NOT_FEEDBACK = re.compile(
    r"^\s*(?:no\s+(?:way|idea|one|problem|worries|clue)|not\s+sure|"
    r"right\s+(?:now|away)|yes\s+(?:and|but|please)\b)", re.I)


def classify_implicit_feedback(message: str) -> int | None:
    """+1 if the message praises the prior answer, -1 if it rejects it, else None."""
    m = (message or "").strip()
    if not m:
        return None
    if _NOT_FEEDBACK.match(m):
        return None
    short = len(m.split()) <= 6
    if _POS_PHRASE.search(m):
        return 1
    if _NEG_PHRASE.search(m):
        return -1
    if short and _POS_LEAD.match(m):
        return 1
    if short and _NEG_LEAD.match(m):
        return -1
    return None


def build_reusable_dataset(rows: list[dict], fmt: str, system_prompt: str = "") -> tuple[str, int]:
    """Turn feedback rows into one of several reusable dataset formats.

    The point is portability: the data you collect should train ANY model later,
    not just this one. Formats:

    - "chat": messages with this assistant's system prompt. Trains a model to be
      THIS assistant. What the built-in LoRA loop uses.
    - "bare": messages with NO system prompt, just user/assistant. Model-neutral;
      use it to train a different base model or a different persona.
    - "preference": {prompt, chosen, rejected} triples for DPO-style preference
      tuning, built from corrected answers and from good/bad answers to the same
      prompt. This is what makes the rejected ("no, wrong") examples pay off.
    - "raw": every column as JSONL. A lossless archive you can reshape into any
      format in the future.

    Returns (text, example_count).
    """
    lines: list[str] = []

    if fmt == "raw":
        for r in rows:
            lines.append(json.dumps(r, ensure_ascii=False, default=str))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    if fmt == "preference":
        seen = set()
        # 1) corrected answers: original is rejected, correction is chosen.
        for r in rows:
            corrected = (r.get("corrected_response") or "").strip()
            original = (r.get("assistant_response") or "").strip()
            prompt = (r.get("user_prompt") or "").strip()
            if corrected and prompt and corrected != original:
                key = (prompt, corrected, original)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": corrected, "rejected": original},
                    ensure_ascii=False))
        # 2) same prompt with an approved answer and a rejected answer.
        approved_by_prompt: dict[str, str] = {}
        rejected_by_prompt: dict[str, str] = {}
        for r in rows:
            prompt = (r.get("user_prompt") or "").strip()
            ans = (r.get("assistant_response") or "").strip()
            if not prompt or not ans:
                continue
            if r.get("approved_for_training"):
                approved_by_prompt.setdefault(prompt, ans)
            elif (r.get("rating") or 0) < 0:
                rejected_by_prompt.setdefault(prompt, ans)
        for prompt, chosen in approved_by_prompt.items():
            rejected = rejected_by_prompt.get(prompt)
            if rejected and rejected != chosen:
                key = (prompt, chosen, rejected)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    # "chat" or "bare": supervised messages. Prefer the corrected answer as the
    # target when present; skip rejected-only rows (nothing good to imitate).
    seen_msgs = set()
    for r in rows:
        prompt = (r.get("user_prompt") or "").strip()
        target = (r.get("corrected_response") or r.get("assistant_response") or "").strip()
        if not prompt or not target:
            continue
        if not r.get("approved_for_training") and not (r.get("corrected_response") or "").strip():
            continue  # a bad answer with no correction is not a target to imitate
        key = (prompt, target)
        if key in seen_msgs:
            continue
        seen_msgs.add(key)
        messages = []
        if fmt == "chat" and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": target})
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


# A short remark ABOUT the previous answer rather than a new question:
# "doesnt have a main function", "no", "i meant ...", "its not ...". These refer
# to the conversation, so retrieving documents for them is wrong -- the terms
# match unrelated indexed material and the model answers from that instead.
_FOLLOWUP_LEAD = re.compile(
    r"^\s*(?:no+|nope|nah|yes+|yeah|ok|okay|wrong|incorrect|still|again|but|also|"
    r"i\s+meant|i\s+mean|that'?s|thats|it'?s|its|you\s+(?:just|didn'?t|did\s+not)|"
    r"doesn'?t|does\s+not|didn'?t|did\s+not|isn'?t|is\s+not|aren'?t|won'?t|can'?t|"
    r"cannot|reconsider|redo|retry|nevermind|never\s+mind)\b",
    re.I,
)


def is_followup_remark(message: str) -> bool:
    """True for a short comment on the previous turn, not a standalone question.

    Used to keep knowledge-base retrieval from firing on conversational repair
    ("doesnt have a main function"), where the useful context is the previous
    exchange, never an indexed document.
    """
    text = (message or "").strip()
    if not text or len(text.split()) > 20:
        return False
    return bool(_FOLLOWUP_LEAD.match(text))


def is_time_sensitive(message: str) -> bool:
    """True if a question plausibly needs current/external facts to answer well.

    Deliberately errs toward the router's own wording so the gate and the router
    agree on what 'needs a lookup' means.
    """
    return bool(TIME_SENSITIVE.search(message or ""))


# Framing words to strip so the search topic is the subject, not "write a python
# script to ...". Applied only when building a query for a code lookup.
_CODE_FRAMING = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:write|create|generate|build|implement|make|code|program|give me|show me)\s+"
    r"(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:python|py|javascript|js|typescript|ts|rust|go|golang|bash|shell|sh|ruby|"
    r"php|perl|sql|c\+\+|cpp|java)?\s*"
    r"(?:script|program|function|snippet|tool|cli|code|app|module|class)?\s*"
    r"(?:that|which|to|for|attempting|covering|using|demonstrating|showing)?\s*",
    re.I,
)


def code_search_topic(message: str) -> str:
    """Strip 'write a python script to ...' framing down to the search subject."""
    topic = _CODE_FRAMING.sub("", (message or "").strip()).strip(" .?!\t")
    return topic or (message or "").strip()[:200]


# A hard analytical question that benefits from being broken into steps. Requires
# an analytical signal (compare, why, how would, evaluate, design, tradeoffs...)
# AND some heft (length, several clauses, or an explicit "step by step"), so it
# does not fire on simple factual or definitional questions that answer in one
# shot. Lookups and code are excluded upstream, so this only sees "answer"-class
# questions.
REASONING_SIGNAL = re.compile(
    r"\b(compare|contrast|versus|vs\.?|trade[- ]?offs?|pros and cons|"
    r"why (?:is|are|does|do|would|should|did)|how would|how do i|how should|"
    r"analy[sz]e|evaluate|assess|weigh|design|architect|derive|prove|"
    r"implications|consequences|reason through|think through|step by step|"
    r"walk me through|work out|figure out|explain why|justify|"
    r"what (?:would|if) )\b",
    re.I,
)


def is_reasoning_question(message: str) -> bool:
    """True if a question is worth decomposing into incremental reasoning steps."""
    text = (message or "").strip()
    signals = REASONING_SIGNAL.findall(text)
    if not signals:
        return False
    # Two or more analytical cues (e.g. "compare ... tradeoffs ... vs") means a
    # genuinely multi-faceted question regardless of length.
    if len(signals) >= 2:
        return True
    # A single cue plus an explicit step-by-step request, length, or several
    # clauses. A lone short cue ("why is the sky blue") stays one-shot.
    if re.search(r"step by step|think through|walk me through", text, re.I):
        return True
    clauses = text.count(" and ") + text.count(", ") + text.count("?")
    return len(text) >= 80 or clauses >= 2


def is_code_request(message: str) -> bool:
    """True if the message asks to write, fix, or modify code.

    Requires a code verb plus either a code object (script, function, ...) or a
    language name, or a fenced code block in the message. Kept deliberately
    strict so "write up the latest news" (verb, but no code object or language)
    does not match and can still be routed to a search.
    """
    text = message or ""
    if "```" in text:
        return True
    has_verb = bool(CODE_INTENT.search(text))
    has_object = bool(CODE_OBJECT.search(text))
    has_language = bool(CODE_LANGUAGE.search(text))
    # A code verb plus an object or a language ("write a python script"), or a
    # language and an object together even without an imperative verb ("python
    # script to pull CVEs"), both count as a code request.
    return (has_verb and (has_object or has_language)) or (has_language and has_object)


# Result lines from _web_search look like "N. Title\n   URL\n   snippet". Pull
# the result URLs in order so the retrieval pipeline can fetch the top ones.
_RESULT_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.M)


# Aggregator, listing and JS-shell URLs whose fetched HTML is mostly navigation
# and script, not article text. Extracting from them wastes a fetch-and-read
# cycle and, on 8GB, the junk-filled prompt is what stalls. Skip them and use the
# next result (or the search snippet) instead.
_LOW_VALUE_HOST = re.compile(
    r"(?:^|\.)(?:news\.google\.|news\.yahoo\.|flipboard\.|reddit\.com|"
    r"twitter\.com|x\.com|facebook\.com|pinterest\.|quora\.com)", re.I)
_LOW_VALUE_PATH = re.compile(
    r"/(?:category|categories|tag|tags|topics?|section|sections|feed|feeds|"
    r"latest|trending|search|archive)(?:/|$|\?)", re.I)


def is_low_value_url(url: str) -> bool:
    """True for aggregator/listing/JS-shell pages unlikely to yield article text."""
    try:
        from urllib.parse import urlparse
        parts = urlparse(url)
    except Exception:
        return False
    host = parts.netloc.lower()
    path = parts.path or "/"
    if _LOW_VALUE_HOST.search(host):
        return True
    if _LOW_VALUE_PATH.search(path):
        return True
    # A bare domain root (no real path) is a homepage/shell, not an article.
    if path in ("", "/") and not parts.query:
        return True
    return False


def extractable_text_len(page: str) -> int:
    """Rough count of article-like text in a fetched page (already tag-stripped).

    Aggregator shells strip down to a pile of short link fragments; a real
    article has long prose lines. Count characters only from lines that read like
    prose (long, or sentence-punctuated) so a wall of two-word nav links scores
    near zero even when the raw length is large.
    """
    total = 0
    for line in (page or "").splitlines():
        line = line.strip()
        if len(line) >= 60 or (len(line) >= 30 and any(c in line for c in ".!?")):
            total += len(line)
    return total


def is_thin_page(page: str, min_chars: int = 400) -> bool:
    """True if a fetched page has too little real prose to be worth extracting."""
    return extractable_text_len(page) < min_chars


def top_result_urls(search_text: str, limit: int) -> list[str]:
    """The first `limit` result URLs from a web_search result block, de-duped."""
    seen: list[str] = []
    for match in _RESULT_URL.finditer(search_text or ""):
        url = match.group(1)
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


# Result blocks look like "N. Title\n   URL\n   snippet" (see _web_search). This
# splits a result block into (title, url, snippet) records in engine order.
_RESULT_HEAD = re.compile(r"^\s*\d+\.\s+(.*)$")
_STOPWORDS = frozenset(
    "the a an of to for and or in on at is are be with how what why when who "
    "which that this from into your you my our their his her its as by".split())


def _parse_search_results(search_text: str) -> list[tuple[str, str, str]]:
    """Parse a web_search result block into (title, url, snippet) records."""
    records: list[tuple[str, str, str]] = []
    title = url = ""
    snippet_lines: list[str] = []

    def flush() -> None:
        if url:
            records.append((title, url, " ".join(snippet_lines).strip()))

    for raw in (search_text or "").splitlines():
        head = _RESULT_HEAD.match(raw)
        if head:
            flush()
            title, url, snippet_lines = head.group(1).strip(), "", []
            continue
        urlm = _RESULT_URL.match(raw)
        if urlm and not url:
            url = urlm.group(1)
            continue
        if raw.strip():
            snippet_lines.append(raw.strip())
    flush()
    return records


def _query_tokens(query: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (query or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


def rank_result_urls(search_text: str, query: str, limit: int) -> list[str]:
    """Rerank a web_search result block by relevance before auto-fetching.

    Engine order is a weak prior; this promotes results whose title/snippet
    actually overlap the question and demotes aggregator/listing hosts, so the
    bounded number of pages we spend a fetch-and-read on are the most likely to
    carry the answer. Falls back to engine order when there is nothing to score
    (no query tokens, or an unparseable block).
    """
    records = _parse_search_results(search_text)
    if not records:
        return top_result_urls(search_text, limit)
    tokens = _query_tokens(query)

    def score(title: str, url: str, snippet: str, rank: int) -> float:
        s = -0.1 * rank  # keep engine order as the tiebreak
        if tokens:
            tl = title.lower()
            sn = snippet.lower()
            s += 2.0 * sum(1 for t in tokens if t in tl)
            s += 1.0 * sum(1 for t in tokens if t in sn)
        if is_low_value_url(url):
            s -= 5.0
        return s

    # Sort on (-score, rank) so a higher score wins and engine order breaks ties.
    scored: list[tuple[float, int, str]] = []
    for rank, (title, url, snippet) in enumerate(records):
        scored.append((-score(title, url, snippet, rank), rank, url))
    scored.sort()
    out: list[str] = []
    for _neg, _rank, url in scored:
        if url not in out:
            out.append(url)
        if len(out) >= limit:
            break
    return out


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-character chunks with a little overlap.

    Overlap keeps a sentence that straddles a boundary from being lost to both
    chunks. Prefers to break on a newline or space near the boundary so chunks
    fall on natural seams rather than mid-word.
    """
    text = text or ""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the nearest newline or space within the last 15%.
            window = text.rfind("\n", start + int(size * 0.85), end)
            if window == -1:
                window = text.rfind(" ", start + int(size * 0.85), end)
            if window != -1:
                end = window
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def quick_tool(message: str) -> tuple[str, dict] | None:
    """Deterministic shortcuts that need no model call at all.

    Only two cases qualify: a message that is purely a URL, and one that is
    purely arithmetic. Both are unambiguous and common enough that spending a
    router call on them would be wasteful. Everything else returns None and is
    handled by the model router.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    url = BARE_URL.match(text)
    if url:
        return "fetch_url", {"url": url.group(1)}
    expression = text.rstrip("=?").strip()
    # Require an operator and a digit so a bare number or a year ("2026") is not
    # mistaken for arithmetic.
    if (ARITHMETIC_ONLY.match(expression) and any(op in expression for op in "+-*/^%")
            and any(char.isdigit() for char in expression)):
        return "calculator", {"expression": expression.replace("^", "**").replace(",", "")}
    return None


# A URL anywhere in a message, and the words that signal "go read this page and
# tell me about it" (as opposed to merely mentioning a link). Used to send
# "summarise https://…" to fetch_url instead of letting the router guess.
_URL_IN_TEXT = re.compile(r'https?://[^\s<>()\[\]{}"\']+', re.I)
# Prefix stems with a LEADING word boundary only (no trailing \b): "summar"
# must match summary/summarize/summarise, so a trailing boundary would break it.
URL_READ_INTENT = re.compile(
    r"\b(?:summar|tl;?dr|read|open|fetch|retriev|scrap|extract|explain|describ|"
    r"analy[sz]e|review|brief|digest|go to|look at|check|"
    r"what(?:'s| is| are| does| say)|contents? of|summary of)",
    re.I)


def url_read_request(message: str) -> str | None:
    """The single URL to fetch when a message asks to read/summarise a page.

    A message that is nothing but a URL is handled by quick_tool (it returns the
    page verbatim). This covers the very common "summarise <url>", "read this:
    <url>", "what does <url> say", "tl;dr <url>", where the user wants the model
    to fetch the page and then answer *from* it. Returns None unless there is
    exactly one URL and a clear read/summarise intent in a short instruction, so
    an ordinary message that merely mentions a link falls through to the router.
    """
    text = (message or "").strip()
    if not text:
        return None
    urls = _URL_IN_TEXT.findall(text)
    if len(urls) != 1:
        return None
    url = urls[0].rstrip('.,;:!?)\'"')
    rest = _URL_IN_TEXT.sub(" ", text).strip()
    if not rest:
        return None  # a bare URL: quick_tool handles it
    if len(rest.split()) <= 20 and URL_READ_INTENT.search(rest):
        return url
    return None


# Power-user routing overrides: a leading slash command that forces a lane and
# bypasses the model router entirely. Deterministic and explicit, so the user is
# never surprised by the router's judgement when they have already made the call.
# Lanes: "web_search" (force a search), "answer" (never search; own knowledge +
# knowledge base), "kb" (answer, but say the knowledge base is the source).
_OVERRIDE = re.compile(
    r"^\s*/(search|web|websearch|nosearch|no-search|answer|local|kb|docs)\b[ \t]*",
    re.I)
_OVERRIDE_LANE = {
    "search": "web_search", "web": "web_search", "websearch": "web_search",
    "nosearch": "answer", "no-search": "answer", "answer": "answer", "local": "answer",
    "kb": "kb", "docs": "kb",
}


def routing_override(message: str) -> tuple[str, str] | None:
    """Parse a leading /command override. Returns (lane, cleaned_message) or None.

    The command is stripped from the message so the downstream tool or the model
    sees only the real request ("/search foo bar" -> ("web_search", "foo bar")).
    """
    m = _OVERRIDE.match(message or "")
    if not m:
        return None
    lane = _OVERRIDE_LANE[m.group(1).lower().replace("-", "")]
    rest = (message[m.end():]).strip()
    return lane, rest


# Unambiguous "search the web" imperatives. Unlike TIME_SENSITIVE (a soft hint
# that a lookup might help), these are direct commands to search, so they route
# straight to web_search with no model-router call. Kept strict: a bare
# "search X" often means the attached codebase, so a web marker or a
# search-engine name is required -- except when the verb IS a search engine.
_SEARCH_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:can|could|would)?\s*(?:you\s+)?"
    r"(?:"
    r"web[- ]?search(?:\s+for)?|"
    r"search\s+(?:the\s+)?(?:web|internet|online)(?:\s+for)?|"
    r"search\s+up|"
    r"google|bing|duckduckgo|ddg"
    r")\b[:,]?\s+",
    re.I)
# A trailing "... online" / "on the web" that turns a plain lookup into a search.
_ONLINE_TRAIL = re.compile(r"\b(?:online|on the web|on the internet)\b\s*[.?!]*$", re.I)
_LOOKUP_LEAD = re.compile(r"^\s*(?:please\s+)?(?:find|look\s+up|search\s+for)\b\s+", re.I)


def web_search_request(message: str) -> str | None:
    """The query to search for when a message is an explicit web-search command.

    Covers "search the web for X", "google X", "web search X", and "find/look up
    X online". Returns None for anything not unambiguously a web search, so an
    ordinary question still goes through the model router.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    m = _SEARCH_LEAD.match(text)
    if m:
        return text[m.end():].strip(" .?!\t") or None
    # "find X online" / "look up X on the web": a lookup verb with a web marker.
    if _LOOKUP_LEAD.match(text) and _ONLINE_TRAIL.search(text):
        q = _LOOKUP_LEAD.sub("", text, count=1)
        return _ONLINE_TRAIL.sub("", q).strip(" .?!\t") or None
    return None


# Backwards-compatible alias. Older call sites and tests refer to fast_path_call;
# it now covers only the deterministic shortcuts. The prev_user parameter is kept
# for signature compatibility but is unused, because follow-up resolution ("look
# it up") is now handled by the model router, which sees the conversation.
def fast_path_call(message: str, prev_user: str | None = None) -> tuple[str, dict] | None:
    return quick_tool(message)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'CODE_INTENT',
    'CODE_LANGUAGE',
    'CODE_NEEDS_LOOKUP',
    'CODE_OBJECT',
    'REASONING_SIGNAL',
    'TIME_SENSITIVE',
    'URL_READ_INTENT',
    '_URL_IN_TEXT',
    'url_read_request',
    'routing_override',
    'web_search_request',
    '_OVERRIDE',
    '_SEARCH_LEAD',
    '_ONLINE_TRAIL',
    '_LOOKUP_LEAD',
    '_CODE_FRAMING',
    '_LOW_VALUE_HOST',
    '_LOW_VALUE_PATH',
    '_NEG_LEAD',
    '_NEG_PHRASE',
    '_NOT_FEEDBACK',
    '_POS_LEAD',
    '_POS_PHRASE',
    '_RESULT_URL',
    '_is_trivial',
    '_last_user_turn',
    'build_reusable_dataset',
    'chunk_text',
    'classify_implicit_feedback',
    'code_search_topic',
    'extractable_text_len',
    'fast_path_call',
    'is_code_request',
    'is_low_value_url',
    'is_reasoning_question',
    'is_substantive',
    'is_thin_page',
    'is_time_sensitive',
    'is_followup_remark',
    '_FOLLOWUP_LEAD',
    'quick_tool',
    'top_result_urls',
    'rank_result_urls',
    '_parse_search_results',
    '_query_tokens',
    '_RESULT_HEAD',
]
