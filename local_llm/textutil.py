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
    r"endpoint|schema|decorator|generator|command|one[- ]?liner|"
    # Systems-programming objects. "write a mouse driver ... in c" matched no
    # object and no language, so it was not treated as code: it got the 512-token
    # chat budget and was cut off mid-function. Deliberately excludes ambiguous
    # business words (service, interface, library) whose false positives would
    # skip the router on a genuine lookup question.
    r"driver|drivers|daemon|makefile|header|headers|struct|shader|firmware|"
    r"binding|bindings|wrapper|middleware|dockerfile|kernel module)\b",
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
# C, which cannot go in CODE_LANGUAGE as a bare \bc\b without matching "vitamin
# c" and every stray initial. These are the phrasings that unambiguously mean
# the language.
CODE_C_LANGUAGE = re.compile(
    r"(?i)\b(?:in|using|with|of)\s+c\b(?!\+|#|\w)|\bansi\s+c\b|\bpure\s+c\b|"
    r"\bc\s*(?:89|90|99|11|17|23)\b|"
    r"\bc\s+(?:program|programme|code|source|function|header|library|driver|app)\b")


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
    r"right\s+(?:now|away)|yes\s+(?:and|but|please)\b|"
    # "no" as a determiner, not a verdict: "no timeout is needed here" is a
    # statement about the design, and treating it as a rejection re-asks the
    # previous request instead of acting on it.
    r"no\s+\w+\s+(?:is|are|was|were|will|would|should|needs?|needed|required)\b)",
    re.I)

# What may follow a lead word and still leave the message pure feedback. Anything
# else means the word was doing grammatical work -- "GOOD, now add retries",
# "CORRECT the typo on line 4", "NO timeout is needed here" -- rather than
# judging the answer. A lead-word verdict therefore has to consume the WHOLE
# message, the same reasoning as the continuation pattern: a verdict that is
# only a prefix is a guess about a sentence that went on to say something else.
_FEEDBACK_TAIL = re.compile(
    r"(?i)^[\s,.!]*(?:that|this|it|thanks?|thank\s+you|ty|cheers|"
    r"mate|man|perfect|great|now|indeed|much\s+better|"
    r"very\s+much|a\s+lot|so\s+much|"
    r"answer|one|work[s]?|"
    # A second verdict word is still just a verdict: "yes exactly", "no, wrong".
    r"exactly|right|correct|wrong|incorrect|nope|no+|yes+|"
    r"is\s+(?:right|correct|wrong|incorrect)|"
    r"was\s+(?:right|correct|wrong|incorrect))*[\s,.!?]*$")

# Praise that is immediately withdrawn. "thanks, but it still fails" was being
# stored as a POSITIVE training example of the very answer the user had just
# said did not work -- and positive/negative labels here are not advisory, they
# become the LoRA dataset.
_WITHDRAWN = re.compile(
    r"(?i)\b(?:but|however|though|except|although)\b|"
    r"\b(?:still|again)\s+(?:\w+\s+){0,2}"
    r"(?:fail|fails|failing|wrong|broken|error|errors|crash|crashes|off)\b|"
    r"\b(?:does\s*n[o']?t|doesn'?t|did\s*n[o']?t|didn'?t|is\s*n[o']?t|isn'?t|"
    r"won'?t|can'?t|cannot)\b")


def classify_implicit_feedback(message: str) -> int | None:
    """+1 if the message praises the prior answer, -1 if it rejects it, else None.

    Conservative by design, because this is not advisory: a verdict here is
    written to the feedback table and becomes a training example. A wrong label
    teaches the model that a bad answer was good, and a wrong NEGATIVE also
    makes the chat handler re-ask the previous request and throw away what the
    user actually just said. None is always the safe answer.
    """
    m = (message or "").strip()
    if not m:
        return None
    if _NOT_FEEDBACK.match(m):
        return None
    withdrawn = bool(_WITHDRAWN.search(m))
    if _POS_PHRASE.search(m):
        return None if withdrawn else 1
    if _NEG_PHRASE.search(m):
        return -1
    short = len(m.split()) <= 6
    if not short:
        return None
    # Asymmetric on purpose. A false POSITIVE teaches the model that a bad
    # answer was good and there is nothing downstream to catch it, so praise has
    # to consume the whole message: "good, now add retries" is an instruction
    # with a courtesy on the front, not an endorsement.
    match = _POS_LEAD.match(m)
    if match and _FEEDBACK_TAIL.match(m[match.end():]):
        return None if withdrawn else 1
    # A rejection is left looser. "no c++ just an empty json?" names what was
    # wrong in the same breath, and the retry path carries the user's own words
    # to the model as the complaint, so the instruction is not lost the way a
    # mislabelled positive is.
    if _NEG_LEAD.match(m):
        return -1
    return None


def rejection_retry_message(prompt: str, answer: str, complaint: str) -> str:
    """What to send the model after the user rejects an answer.

    A rejection is about the PREVIOUS request, so the request is re-asked with
    the complaint attached. Answering the complaint as a fresh question is how
    "no c++ just an empty json?" produced a second empty JSON object: the model
    was asked about the remark, not about the C++ program, while the rejected
    answer sat in the history as the only worked example in sight. Naming it as
    rejected is the point.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return (complaint or "").strip()
    rejected = " ".join((answer or "").split())[:300] or "(empty)"
    return (f"{prompt}\n\n[Your previous answer to this was REJECTED by the user, who "
            f"said: \"{(complaint or '').strip()}\". The rejected answer was: {rejected}\n"
            "Answer the original request again, in full, in plain text, fixing what "
            "they complained about. Do not repeat the rejected answer, and do not "
            "reply with JSON.]")


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


# The model saying it cannot do the job without something it was not given: the
# web, the user's files, a shell. This is the model itself reporting that the
# tool-free lane was the wrong lane, and it is far stronger evidence than the
# router's one-shot guess -- the router decides before seeing the question's
# difficulty, this arrives after the model has tried.
#
# Split in two because the two halves want different tools. NEEDS_LOOKUP is
# answered by web_search/fetch_url; NEEDS_FILES by the file and shell tools.
_NEEDS_LOOKUP = re.compile(
    r"(?i)\b(?:"
    r"(?:i\s+)?(?:do\s*n[o']?t|don'?t|cannot|can'?t|could\s+not|couldn'?t|"
    r"am\s+(?:un|not\s+)able|'?m\s+(?:un|not\s+)able|unable)\s+"
    r"(?:to\s+)?(?:\w+\s+){0,3}?"
    r"(?:access|browse|reach|retrieve|look\s+(?:it|that|this)?\s*up|search|"
    r"fetch|download|open\s+(?:the\s+)?(?:link|url|page|website)|"
    r"real[- ]?time|live|current|up[- ]?to[- ]?date|internet|web)"
    r"|"
    r"(?:my|the)\s+(?:knowledge|training)\s+(?:cut[- ]?off|cutoff|data)\b"
    r"|"
    r"as\s+of\s+my\s+(?:last\s+)?(?:knowledge|training|update)"
    r"|"
    r"i\s+(?:do\s*n[o']?t|don'?t)\s+have\s+(?:access\s+to\s+)?"
    r"(?:real[- ]?time|live|current|up[- ]?to[- ]?date|internet|the\s+web)"
    r"|"
    r"(?:you|you'?ll|please)\s+(?:should\s+|will\s+|need\s+to\s+|can\s+)?"
    r"(?:check|visit|consult|see)\s+(?:the\s+)?(?:official|vendor|their)\s+"
    r"(?:website|site|docs|documentation|page)"
    r")\b"
)
_NEEDS_FILES = re.compile(
    r"(?i)\b(?:"
    r"(?:i\s+)?(?:do\s*n[o']?t|don'?t|cannot|can'?t|could\s+not|couldn'?t|"
    r"am\s+(?:un|not\s+)able|'?m\s+(?:un|not\s+)able|unable)\s+"
    r"(?:to\s+)?(?:\w+\s+){0,3}?"
    r"(?:read|open|see|view|inspect|list|run|execute)\s+"
    r"(?:your|the|any|this|that|these|those)?\s*"
    r"(?:file|files|code|codebase|repo|repository|project|directory|folder|"
    r"script|source|command|tests?)"
    r"|"
    r"without\s+(?:seeing|reading|access\s+to)\s+(?:your|the)\s+"
    r"(?:file|files|code|codebase|repo|repository|project|source)"
    r"|"
    r"(?:please\s+)?(?:paste|share|show|send)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:file|files|code|contents|source|error|output)\b"
    r")"
)

# How much text may surround a refusal before it stops being a refusal. A real
# answer that opens "I don't have real-time access, but as of 2024 ..." and then
# answers is NOT a refusal, and escalating it would spend a search on a question
# already answered. A refusal is short because there was nothing to say.
REFUSAL_MAX_CHARS = 700


def missing_capability(answer: str) -> str | None:
    """Which capability an answer says it lacked: "lookup", "files", or None.

    Used to switch a tool-free turn into agent mode after the fact. Returns None
    for any answer long enough to be a real answer, so a hedged-but-complete
    reply is never mistaken for a refusal.
    """
    text = (answer or "").strip()
    if not text or len(text) > REFUSAL_MAX_CHARS:
        return None
    # A reply carrying a code block or a table did work; the caveat in it is a
    # caveat, not a refusal.
    if "```" in text:
        return None
    if _NEEDS_FILES.search(text):
        return "files"
    if _NEEDS_LOOKUP.search(text):
        return "lookup"
    return None


# A request for a document FILE, and which format it asks for. Detected
# deterministically, for the same reason is_code_request is: "write me a PDF
# report on the Q3 numbers" reaches the router as an ordinary request, the router
# quite reasonably answers "answer", and the turn is then rebuilt with no tools
# at all -- so the model writes a lovely report into the chat and no file is ever
# produced. The user asked for a file and got a message.
#
# Order matters: the first pattern that matches wins, so the more specific
# formats are listed before the ones whose words appear inside them.
_DOC_FORMAT_WORDS = (
    (".docx", r"word\s+(?:document|doc|file|version)|docx|ms\s*word|"
              r"microsoft\s+word"),
    (".xlsx", r"excel|spread\s?sheet|xlsx|workbook"),
    (".pptx", r"power\s?point|pptx|slide\s*deck|slides|presentation|deck"),
    (".pdf", r"pdf"),
    (".csv", r"csv"),
    (".html", r"html|web\s*page"),
    (".md", r"markdown|\.md\b"),
    (".txt", r"(?:plain\s*)?text\s+file|\.txt\b"),
)
# Verbs that mean "produce one", as opposed to reading one the user already has.
_DOC_MAKE = (r"(?:creat\w*|generat\w*|mak\w*|writ\w*|produc\w*|export\w*|sav\w*|"
             r"build\w*|draft\w*|prepar\w*|render\w*|render|put|compile|"
             r"give\s+me|send\s+me|i\s+(?:need|want)|turn\s+(?:it|this|that|them)\s+into|"
             r"convert\s+(?:it|this|that|them)\s+(?:in)?to)")
# A filename the user spelled out wins over any phrasing: "save it as notes.docx"
# names both the format and the file.
# Format words that are safe ONLY inside a target phrase. "to word" means Word;
# a bare "word" anywhere else is an English word.
_DOC_TARGET_EXTRA = {".docx": r"word", ".xlsx": r"sheets?", ".pptx": r"slides?"}
_DOC_FILENAME = re.compile(
    r"(?i)\b[\w][\w \-.]{0,60}?(\.(?:pdf|docx|xlsx|pptx|csv|html?|md|txt))\b")


# Verbs that mean the user already HAS the file and wants it read. A format word
# under one of these is not a request to produce anything: "summarise this pdf",
# "read the excel file I uploaded". Only an explicit output phrase ("... into a
# word document") overrides it.
_DOC_READ = re.compile(
    r"(?i)\b(?:read|open|parse|extract|summari[sz]e|analy[sz]e|review|index|"
    r"import|uploaded?|attached?|look\s+at|check|what(?:'s|\s+is)\s+in)\b")


def _doc_patterns() -> tuple[list, list]:
    """Two matchers per format, built once.

    `target` is the unambiguous one: "as/into/to a <fmt>" states the OUTPUT and
    therefore beats a filename mentioned elsewhere in the same sentence, which is
    what makes "convert report.pdf to word" a Word request rather than a PDF one.
    `other` covers "<fmt> file/report/..." and "<make verb> ... a <fmt>", both of
    which are requests to produce one but neither of which can outrank a
    filename the user spelled out.
    """
    target, other = [], []
    for suffix, words in _DOC_FORMAT_WORDS:
        extra = _DOC_TARGET_EXTRA.get(suffix)
        target.append((suffix, re.compile(
            r"(?i)\b(?:as|in-?to|to)\s+(?:an?\s+)?(?:\w+\s+){0,2}?"
            rf"(?:{words}{'|' + extra if extra else ''})\b")))
        other.append((suffix, re.compile(
            r"(?i)(?:"
            rf"\b(?:in)\s+(?:an?\s+)?(?:new\s+)?(?:{words})\b"
            rf"|\b(?:{words})\s+(?:file|document|doc|report|version|copy|export|"
            r"deck|presentation|summary|output)\b"
            rf"|\b{_DOC_MAKE}\s+(?:me\s+)?(?:\w+\s+){{0,3}}?(?:an?|the)?\s*(?:{words})\b"
            r")")))
    return target, other


_DOC_TARGET, _DOC_OTHER = _doc_patterns()


def document_request(message: str) -> str | None:
    """The file extension a message asks to be produced, or None.

    Returns ".pdf", ".docx", ... so the caller can name the format back to the
    user and steer the tool. None for a message that merely mentions a format
    ("summarise this pdf"), which is a request to READ one -- acting on that
    would have the agent overwrite the very file it was asked to look at.
    """
    text = (message or "").strip()
    if not text:
        return None
    for suffix, pattern in _DOC_TARGET:
        if pattern.search(text):
            return suffix
    if _DOC_READ.search(text):
        return None
    named = _DOC_FILENAME.search(text)
    if named:
        suffix = named.group(1).lower()
        return ".html" if suffix == ".htm" else suffix
    for suffix, pattern in _DOC_OTHER:
        if pattern.search(text):
            return suffix
    return None


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


def is_reasoning_question(message: str, signals: int = 2, min_chars: int = 80) -> bool:
    """True if a question is worth decomposing into incremental reasoning steps.

    `signals` is how many analytical cues make it automatic, and `min_chars` the
    length at which a single cue is enough. Both are exposed as config so the
    bar can be moved: decomposition is the most expensive thing this pipeline
    does (one model call per step plus a synthesis), so the right threshold
    depends on the machine and on how much latency the user will trade for
    depth. A message with no cue at all never decomposes, whatever they are set
    to -- the cue list is what distinguishes a question with work in it from
    one that just needs an answer.
    """
    text = (message or "").strip()
    found = REASONING_SIGNAL.findall(text)
    if not found:
        return False
    # Enough analytical cues (e.g. "compare ... tradeoffs ... vs") means a
    # genuinely multi-faceted question regardless of length.
    if len(found) >= max(1, signals):
        return True
    # A single cue plus an explicit step-by-step request, length, or several
    # clauses. A lone short cue ("why is the sky blue") stays one-shot.
    if re.search(r"step by step|think through|walk me through", text, re.I):
        return True
    clauses = text.count(" and ") + text.count(", ") + text.count("?")
    return len(text) >= min_chars or clauses >= 2


# Protocol scaffolding a model wraps its answer in, which must never reach the
# user. Two shapes, both measured on this hardware rather than guessed at:
#
#   <final_answer>...</final_answer>     an XML tag invented by symmetry with the
#                                        <think>...</think> tags the reasoning
#                                        instruction asks for
#   **Answer**: ...  /  Final answer: ...  a label, sometimes dressed as markdown
#
# Both come from the same place: an instruction that describes XML tags and then
# says "give your final answer". The prompt no longer phrases it that way, but a
# 3B model treats prompt wording as a suggestion, so the output is cleaned too.
_ANSWER_TAG = re.compile(
    r"(?is)\A\s*<\s*(final[_ ]?answer|answer|response|output)\s*>\s*(.*?)"
    r"(?:\s*<\s*/\s*\1\s*>\s*)?\Z")
# A label at the very START of the reply only. A mid-text "Answer:" is somebody
# formatting their prose, and rewriting that would be worse than the leak.
#
# Two forms, because the model produces both: an inline label ending in a colon,
# and a markdown heading on its own line. The heading form requires the word
# "final", since a bare "## Answer" section heading is plausibly the user's own
# structure whereas "## Final Answer" is the protocol talking.
_ANSWER_LABEL = re.compile(
    r"(?i)\A\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*final[_ ]?answer\s*(?:\*\*|__)?\s*[:\-]\s*"
    r"|\A\s*(?:#{1,6}\s*)?(?:\*\*|__)\s*answer\s*(?:\*\*|__)\s*[:\-]\s*")
_ANSWER_HEADING = re.compile(
    r"(?i)\A\s*(?:#{1,6}\s*|\*\*|__)\s*final[_ ]?answer\s*(?:\*\*|__)?\s*:?[ \t]*\r?\n+")


def unwrap_answer(text: str) -> str:
    """Strip protocol scaffolding a model wrapped its answer in.

    Applied where an answer becomes the reply, not where it is generated, so
    every lane is covered by one rule: the tool loop, the prose lane, the
    continuation, a subagent's result and the plain non-agent chat all end up
    here.

    Conservative by design. Tags are removed only when one WRAPS the whole
    reply, and a label only at the very start, because "<final_answer>" inside
    an answer about XML, or an "Answer:" halfway down a formatted reply, are the
    user's content rather than protocol noise.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    for _ in range(3):        # a tag inside a label, or the reverse
        before = cleaned
        match = _ANSWER_TAG.match(cleaned)
        if match:
            cleaned = match.group(2).strip()
        cleaned = _ANSWER_HEADING.sub("", cleaned, count=1)
        cleaned = _ANSWER_LABEL.sub("", cleaned, count=1).strip()
        # A JSON object whose only content is the answer. parse_tool_call
        # ignores it (no tool name) so it used to be shown to the user raw.
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                keys = {str(k).lower() for k in parsed}
                if keys and keys <= {"answer", "final_answer", "response", "text"}:
                    values = [str(v) for v in parsed.values() if isinstance(v, str)]
                    if values:
                        cleaned = "\n".join(values).strip()
        # A dangling close tag left by a wrapper whose open tag was in the
        # reasoning block, or by truncation.
        cleaned = re.sub(
            r"(?is)\s*<\s*/\s*(final[_ ]?answer|answer|response|output)\s*>\s*\Z",
            "", cleaned).strip()
        if cleaned == before:
            break
    return cleaned or (text or "").strip()


# The visible marker appended to a reply that hit the token budget. It is the
# ONLY durable record that an answer was cut off: the conversation stores the
# text, not the finish_reason, so a "continue" arriving in a later request has
# nothing else to go on. Generation and detection therefore share one definition,
# and the pattern is loose about the punctuation so notes written by an older
# build still match.
TRUNCATION_NOTE = re.compile(r"\n*\[cut off at the \d+-token reply limit[^\]]*\]\s*$")


def truncation_note(limit: int, requested: int = 0, context_size: int = 0) -> str:
    """The marker note_if_cut appends. Keep in sync with TRUNCATION_NOTE.

    The advice has to be true. "Raise Max tokens in Settings" was printed on
    turns where the harness had already widened the user's 512 to 1536 on its
    own and the real ceiling was the context window, so following it changed
    nothing. When the effective limit is not the user's setting, say which one
    actually bit.
    """
    if requested and limit > requested:
        reason = (f" \u2014 your Max tokens is {requested}, already widened to "
                  f"{limit} for code; ask me to continue")
    elif context_size and limit + 256 >= context_size:
        reason = (f" \u2014 the {context_size}-token context window is the limit "
                  "here, not Max tokens; ask me to continue")
    else:
        reason = " \u2014 raise \"Max tokens\" in Settings, or ask me to continue"
    return f"\n\n[cut off at the {limit}-token reply limit{reason}]"


def was_truncated(text: str) -> bool:
    """True if this stored assistant message ends in the truncation marker."""
    return bool(TRUNCATION_NOTE.search(text or ""))


def append_below_answer(text: str, extra: str) -> str:
    """Append a footnote while keeping the truncation note last.

    was_truncated anchors to the END of the string, and it is the only evidence
    the continuation lane has that an answer stopped at the budget rather than
    at its natural end. Anything appended after it therefore does not merely
    look untidy: it turns "continue" back into an ordinary question.
    """
    body = text or ""
    if not (extra or "").strip():
        return body
    match = TRUNCATION_NOTE.search(body)
    if not match:
        return body.rstrip() + extra
    return body[:match.start()].rstrip() + extra + "\n" + match.group(0).strip()


def strip_truncation_note(text: str) -> str:
    """Drop the marker.

    Called on the way INTO every prompt as well as on the continuation path: the
    marker is UI copy addressed to the user ("raise Max tokens in Settings"), and
    feeding it back as conversation invites the model to imitate it.
    """
    return TRUNCATION_NOTE.sub("", text or "").rstrip()


# A short "carry on" follow-up. Loose about wording, strict about length: the
# caller requires that the previous answer was ACTUALLY truncated as well, so
# both signals must agree before a turn is treated as a continuation and a long
# instruction that merely opens with "continue" is never hijacked.
# A continuation request carries NO new instruction, and the whole string has to
# be consumed to prove it. As a prefix match this hijacked real work: "more
# tests please", "finish the parser" and "go ahead and add retries" were all
# read as "resume the previous answer", so whenever the previous reply had been
# cut off -- which on a small local model is most of them -- the user's actual
# instruction was dropped on the floor and never reached the model.
CONTINUE_REQUEST = re.compile(
    r"(?i)^\s*(?:please\s+|ok(?:ay)?[,\s]+|yes[,\s]+|and\s+|just\s+)*"
    r"(?:continue|carry\s+on|keep\s+going|go\s+on|go\s+ahead|resume|"
    r"finish\s+(?:it|that|this|up)?|(?:the\s+)?rest|more)"
    # Only continuation-shaped tails, never a new object.
    r"(?:\s+(?:of\s+(?:it|that|this)|from\s+where\s+(?:you|it)\s+\w+|"
    r"where\s+you\s+left\s+off|with\s+(?:it|that|this)|for\s+me|now|"
    r"pl(?:ease|z)|then))*"
    r"[\s,.!]*$")


def is_continue_request(message: str) -> bool:
    """True for a short "continue" style follow-up. Pair with was_truncated.

    A full match, not a prefix one: anything trailing the continue word is a new
    instruction, and resuming instead of reading it silently discards it.
    """
    text = (message or "").strip()
    if not text or len(text) > 80:
        return False
    return bool(CONTINUE_REQUEST.match(text))


# A question ABOUT code rather than a request to change it. The failure this
# exists for: mid-task, "whats the logic behind SetHardwareInfo(...)" is a
# request for an explanation, but every gate upstream saw an active code task
# and briefed the model to re-emit the whole file -- into a 512-token chat
# budget, so it was cut off mid-function and the question was never answered.
_EXPLAIN_LEAD = re.compile(
    r"(?i)^\s*(?:and\s+|so\s+|but\s+|ok(?:ay)?[,\s]+|please\s+|"
    r"can\s+you\s+|could\s+you\s+)*"
    r"(?:what(?:'?s| is| are| does| do| was| were)?|why|how\s+(?:does|do|did|is|are|"
    r"come)|explain|describe|walk\s+me\s+through|tell\s+me\s+(?:about|why|how|what)|"
    r"where\s+(?:does|do|is|are)|when\s+(?:does|do|is|are)|which\s+\w+\s+(?:does|do|is|are)|"
    r"is\s+(?:that|this|it|there)|are\s+(?:those|these|they)|does\s+(?:that|this|it)|"
    r"do\s+(?:those|these|they)|should\s+(?:i|we|that|this|it))\b")

# Asking for a change, even when the sentence opens like a question. "can you
# add error handling" is a modification; "what does this do" is not.
_EXPLAIN_NOT = re.compile(
    r"(?i)\b(add|append|extend|include|implement|rewrite|refactor|port|convert|"
    r"translate|fix|correct|change|modify|update|remove|delete|drop|replace|"
    r"rename|split|merge|optimi[sz]e|simplify|harden|improve|make\s+it|"
    r"give\s+me|show\s+me\s+the\s+(?:code|file|script)|write)\b")


def is_explanation_request(message: str) -> bool:
    """True for a question ABOUT the work rather than a change to it.

    Gates two things that have to agree: whether the reply budget is the chat
    budget, and whether the task brief demands the whole file back. They
    disagreed, and a question mid-task got a "reply with the complete updated
    file" instruction on a 512-token budget.

    Deliberately conservative: a message that also asks for a change ("why is
    that slow, can you fix it") is a change, because answering it as prose only
    would drop half the request.
    """
    text = " ".join((message or "").split())
    if not text or len(text) > 400:
        return False
    if not _EXPLAIN_LEAD.match(text) and "?" not in text:
        return False
    if _EXPLAIN_NOT.search(text):
        return False
    # A quoted code fragment is normal in "what does this do: <code>", so a
    # fenced block is not by itself evidence of a code request here.
    return bool(_EXPLAIN_LEAD.match(text) or text.rstrip().endswith("?"))


# Pointing at work already in progress: a pronoun, or a definite reference to
# the thing being built. Used to keep the router's hands off a follow-up.
_ACTIVE_TASK_REFERENCE = re.compile(
    r"(?i)(?:^|\b)(?:it|its|it'?s|that|this|these|those|them|"
    r"the\s+(?:\w+\s+){0,2}(?:script|code|file|program|programme|module|"
    r"function|port|thing|one|artifact|above)|"
    r"(?:the|your|my)\s+(?:last|previous|earlier|first|second)\b|"
    r"what\s+(?:you|we)\s+(?:wrote|produced|made|had|did))\b")


def refers_to_active_task(message: str) -> bool:
    """True when a short message is about work already under way.

    Length-bounded: a long message carries enough of its own subject that the
    router can judge it, while "how about the c port" carries none at all and
    becomes a literal two-word search query the moment it leaves this context.
    """
    text = " ".join((message or "").split())
    if not text or len(text.split()) > 14:
        return False
    if _URL_IN_TEXT.search(text):
        return False                    # a link is a lookup, whatever else it says
    if is_time_sensitive(text):
        # "the latest version of X" points at the world, not at the artifact,
        # however definite the article is.
        return False
    return bool(_ACTIVE_TASK_REFERENCE.search(text))


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
    has_language = bool(CODE_LANGUAGE.search(text) or CODE_C_LANGUAGE.search(text))
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
# The user pointing at their own material: their project, a repo, a path, a
# filename. A code request WITHOUT any of these ("write a basic c++ crud
# program") wants code from the model's own knowledge, and pulling indexed
# documents into it only spends context and derails the answer.
_PROJECT_REFERENCE = re.compile(
    r"(\bthis (?:project|repo|repository|codebase|file|function|class|module|"
    r"script|code|bug|error|test|package)\b"
    r"|\b(?:my|our) (?:project|repo|repository|codebase|code ?base|code|file|"
    r"files|script|scripts|module|function|class|tests?)\b"
    r"|\b(?:in|from|of|open|read|edit|patch|review) (?:the )?(?:file|files|repo|"
    r"repository|codebase|project)\b"
    r"|\b[\w-]+\.(?:py|js|ts|tsx|jsx|rs|go|c|h|cc|cpp|hpp|cs|java|rb|php|swift|"
    r"kt|sh|zsh|sql|html|css|json|toml|yaml|yml|md|txt|ipynb)\b"
    r"|[~.]?/[\w.-]+/[\w.-]+)",
    re.I,
)


def refers_to_project(message: str) -> bool:
    """True if the message points at the user's own files, repo or project."""
    return bool(_PROJECT_REFERENCE.search(message or ""))


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
# knowledge base), "kb" (answer, but say the knowledge base is the source),
# "agent" (skip the router and go straight to the tool loop).
_OVERRIDE = re.compile(
    r"^\s*/(search|web|websearch|nosearch|no-search|answer|local|kb|docs|"
    r"agent|tools|act)\b[ \t]*",
    re.I)
_OVERRIDE_LANE = {
    "search": "web_search", "web": "web_search", "websearch": "web_search",
    "nosearch": "answer", "no-search": "answer", "answer": "answer", "local": "answer",
    "kb": "kb", "docs": "kb",
    "agent": "agent", "tools": "agent", "act": "agent",
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
    'CODE_C_LANGUAGE',
    'CODE_INTENT',
    'CODE_LANGUAGE',
    'CONTINUE_REQUEST',
    'TRUNCATION_NOTE',
    'is_continue_request',
    'unwrap_answer',
    'truncation_note',
    'was_truncated',
    'strip_truncation_note',
    'append_below_answer',
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
    '_FEEDBACK_TAIL',
    '_WITHDRAWN',
    '_POS_LEAD',
    '_POS_PHRASE',
    '_RESULT_URL',
    '_is_trivial',
    '_last_user_turn',
    'build_reusable_dataset',
    'chunk_text',
    'classify_implicit_feedback',
    'rejection_retry_message',
    'code_search_topic',
    'extractable_text_len',
    'fast_path_call',
    'is_code_request',
    'is_explanation_request',
    'refers_to_active_task',
    '_ACTIVE_TASK_REFERENCE',
    '_EXPLAIN_LEAD',
    '_EXPLAIN_NOT',
    'is_low_value_url',
    'is_reasoning_question',
    'is_substantive',
    'is_thin_page',
    'is_time_sensitive',
    'document_request',
    '_DOC_FORMAT_WORDS',
    '_DOC_TARGET',
    '_DOC_OTHER',
    '_DOC_READ',
    '_DOC_FILENAME',
    '_DOC_TARGET_EXTRA',
    '_DOC_MAKE',
    'missing_capability',
    'REFUSAL_MAX_CHARS',
    '_NEEDS_LOOKUP',
    '_NEEDS_FILES',
    'is_followup_remark',
    '_FOLLOWUP_LEAD',
    'quick_tool',
    'top_result_urls',
    'rank_result_urls',
    '_parse_search_results',
    '_query_tokens',
    'rag_query_tokens',
    '_RAG_GENERIC',
    'refers_to_project',
    '_PROJECT_REFERENCE',
    '_RESULT_HEAD',
]
