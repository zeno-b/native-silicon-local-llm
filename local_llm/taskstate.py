"""Structured task and artifact state that survives context trimming.

The failure this exists for: task identity lived only in the raw transcript.
"write a bash script for macOS disk health" is one user message, and it is the
OLDEST one, so it is the first thing trim_to_context evicts. Three turns later
the model saw nothing but "add more checks and error handling" -- no language,
no platform, no artifact -- and answered with unrelated Python. Nothing in the
pipeline could notice, because nothing in the pipeline knew what the task was.

So the task is represented explicitly here, rebuilt deterministically from the
conversation on every turn (no model call), persisted per conversation, and
injected into the prompt as a block that trimming cannot reach because it rides
on the current user turn. Old messages then become genuinely disposable:

    conversation history -> TaskState -> artifact -> relevant context -> LLM

Everything in this module is a pure function or a plain dataclass, so it is
cheap to test and cannot fail a request.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from .core import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .codecheck import definitions, looks_damaged


# --------------------------------------------------------------------------- #
# Language and platform identification                                        #
# --------------------------------------------------------------------------- #

# Canonical name per alias. Only unambiguous spellings: a bare "r", "c" or "go"
# in prose is far more often a word than a language, and a wrong language in the
# task state is worse than none (it becomes a drift check against the wrong
# thing).
_LANG_ALIASES = {
    "bash": "bash", "shell": "bash", "shell script": "bash", "shellscript": "bash",
    "zsh": "bash", "posix sh": "bash", "sh": "bash",
    "python": "python", "python3": "python", "py": "python",
    "javascript": "javascript", "js": "javascript", "node": "javascript",
    "node.js": "javascript", "nodejs": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "rust": "rust", "golang": "go",
    # Bare "c" is safe HERE because canonical_language only ever sees a code
    # fence's info string, never prose. detect_language keeps refusing it in
    # free text, where "c" is an initial far more often than a language.
    "c": "c", "objective-c": "c", "objc": "c",
    "c++": "c++", "cpp": "c++", "cxx": "c++",
    "c#": "c#", "csharp": "c#",
    "java": "java", "ruby": "ruby", "php": "php", "perl": "perl",
    "sql": "sql", "html": "html", "css": "css",
    "swift": "swift", "swiftui": "swift", "kotlin": "kotlin", "dart": "dart",
    "scala": "scala", "lua": "lua", "haskell": "haskell", "elixir": "elixir",
    "powershell": "powershell", "ps1": "powershell",
    "dockerfile": "dockerfile", "yaml": "yaml", "yml": "yaml", "json": "json",
    "makefile": "make", "make": "make", "terraform": "terraform", "hcl": "terraform",
}

# Longest-first so "javascript" is not read as "java" and "python3" not as
# "python". Word-ish boundaries that tolerate the +/# in c++ and c#.
_LANG_IN_TEXT = re.compile(
    r"(?i)(?<![\w+#.])("
    r"shell\s+script|posix\s+sh|powershell|javascript|typescript|node\.js|nodejs|"
    r"dockerfile|terraform|makefile|haskell|elixir|python3|python|kotlin|"
    r"swiftui|swift|scala|golang|bash|zsh|rust|ruby|perl|php|java|html|css|sql|"
    r"yaml|json|lua|dart|c\+\+|cpp|c#|csharp"
    r")(?![\w+#])")

# The file extension each language's artifact gets, for a readable artifact name.
_LANG_EXT = {
    "bash": "sh", "python": "py", "javascript": "js", "typescript": "ts",
    "rust": "rs", "go": "go", "c++": "cpp", "c#": "cs", "java": "java",
    "ruby": "rb", "php": "php", "perl": "pl", "sql": "sql", "html": "html",
    "css": "css", "swift": "swift", "kotlin": "kt", "dart": "dart",
    "scala": "scala", "lua": "lua", "haskell": "hs", "elixir": "ex",
    "powershell": "ps1", "yaml": "yaml", "json": "json", "make": "mk",
    "terraform": "tf", "dockerfile": "Dockerfile", "c": "c",
}

# What to call the artifact in the brief, so the model is told it is editing a
# shell script rather than an anonymous blob.
_LANG_ARTIFACT_TYPE = {
    "bash": "shell_script", "python": "python_module", "javascript": "js_module",
    "typescript": "ts_module", "sql": "sql_query", "html": "html_document",
    "dockerfile": "dockerfile", "yaml": "yaml_document", "json": "json_document",
    "make": "makefile", "terraform": "terraform_config", "powershell": "powershell_script",
}

# Languages close enough that producing one for the other is not drift.
_LANG_FAMILIES = [{"bash", "sh"}, {"javascript", "typescript"}, {"c", "c++"}]

_PLATFORM_IN_TEXT = re.compile(
    r"(?i)\b(macos|mac\s?os\s?x|mac\s?os|osx|os\s+x|darwin|apple\s+silicon|"
    r"linux|ubuntu|debian|rhel|centos|alpine|windows|wsl|freebsd|android|ios)\b")

_PLATFORM_CANON = {
    "macos": "macOS", "macosx": "macOS", "mac os": "macOS", "mac osx": "macOS",
    "mac os x": "macOS", "osx": "macOS", "os x": "macOS", "darwin": "macOS",
    "apple silicon": "macOS", "linux": "Linux", "ubuntu": "Linux",
    "debian": "Linux", "rhel": "Linux", "centos": "Linux", "alpine": "Linux",
    "windows": "Windows", "wsl": "Linux", "freebsd": "FreeBSD",
    "android": "Android", "ios": "iOS",
}


# Commands that only exist (or only behave as written) on one platform. A shell
# script's platform is usually stated nowhere and implied entirely by what it
# calls: the disk script that started this was full of diskutil, so "macOS" was
# knowable from the artifact even though no message ever said it.
_PLATFORM_IN_CODE = [
    ("macOS", re.compile(r"\b(diskutil|sw_vers|system_profiler|launchctl|pmset|"
                         r"scutil|osascript|networksetup|/Volumes/)\b")),
    ("Linux", re.compile(r"\b(lsblk|apt-get|systemctl|journalctl|dmidecode|"
                         r"/proc/(?:mounts|cpuinfo)|udevadm)\b")),
    ("Windows", re.compile(r"\b(Get-WmiObject|Get-CimInstance|wmic|reg\.exe)\b")),
]


def platform_from_code(code: str) -> str:
    """The platform a script's own commands imply, or "" when it is portable."""
    for name, pattern in _PLATFORM_IN_CODE:
        if pattern.search(code or ""):
            return name
    return ""


def canonical_language(name: str | None) -> str:
    """Map a language name or code-fence info string onto a canonical name."""
    text = (name or "").strip().lower()
    if not text:
        return ""
    # A fence info string can carry extras: ```bash title=x
    text = re.split(r"[\s,;:{]", text, 1)[0]
    return _LANG_ALIASES.get(text, "")


# The single-letter languages, which _LANG_IN_TEXT deliberately refuses: a bare
# "c" or "go" in prose is a word far more often than a language. After an
# explicit preposition, in a message that is already about code, it is not
# ambiguous -- and "port it to c" being unreadable is what pinned a whole
# session to PowerShell while the user kept asking for C.
_BARE_LANG_AFTER_PREP = re.compile(
    r"(?i)\b(?:in|into|to|using|as|toward[s]?)\s+(c|go|r)\b(?![\w+#])")

_BARE_LANG_CANON = {"c": "c", "go": "go", "r": "r"}


def detect_language(text: str, allow_bare: bool = False) -> str:
    """The language a message is about, or "" when it names none.

    `allow_bare` opens the single-letter languages ("port it to c"). It is off
    by default because detect_language runs on every message, including ones
    that have nothing to do with code, and "I need to go" must not retarget a
    task to Golang. Callers that already know the message is a port or a
    correction turn it on.
    """
    match = _LANG_IN_TEXT.search(text or "")
    if match:
        return _LANG_ALIASES.get(match.group(1).lower().replace("  ", " "), "")
    # CODE_C_LANGUAGE is already tuned to the phrasings that unambiguously mean
    # the C language ("in c", "ansi c", "a c program"), so it needs no opt-in.
    if CODE_C_LANGUAGE.search(text or ""):
        return "c"
    if allow_bare:
        bare = _BARE_LANG_AFTER_PREP.search(text or "")
        if bare:
            return _BARE_LANG_CANON.get(bare.group(1).lower(), "")
    return ""


def detect_platform(text: str) -> str:
    """The target platform a message names, canonicalised ("macOS", "Linux")."""
    match = _PLATFORM_IN_TEXT.search(text or "")
    if not match:
        return ""
    key = " ".join(match.group(1).lower().split())
    return _PLATFORM_CANON.get(key, _PLATFORM_CANON.get(key.replace(" ", ""), ""))


_FENCE = re.compile(r"```([A-Za-z0-9+#._-]*)[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)", re.S)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced block in a reply as (info string, code). Tolerates an
    unclosed final fence, which is exactly what a truncated answer leaves."""
    return [(m.group(1) or "", m.group(2)) for m in _FENCE.finditer(text or "")]


# Signature evidence per language. Deliberately narrow: this decides whether an
# answer contradicts the active task, so a guess is worse than a shrug.
_SIGNATURES: list[tuple[str, re.Pattern]] = [
    # Shell-specific variable forms only. A bare `$name` is shared with
    # PowerShell, and counting it here is what let a PowerShell module score as
    # much shell evidence as it did PowerShell evidence and come back
    # unidentified -- which in turn let a real drift through unnoticed.
    ("bash", re.compile(r"^#!.*\b(bash|sh|zsh)\b|^\s*(?:fi|esac|done)\s*$|"
                        r"\[\[\s|\$\{\w|\$[(@*?#]|\$\d|"
                        r"^\s*local\s+\w+=|"
                        r"^\s*(?:function\s+)?\w+\s*\(\)\s*\{", re.M)),
    ("python", re.compile(r"^#!.*\bpython|^\s*(?:from\s+[\w.]+\s+)?import\s+[\w.]|"
                          r"^\s*def\s+\w+\s*\(.*\)\s*(?:->[^:]+)?:|^\s*class\s+\w+"
                          r"\s*[:\(]|^\s*if\s+__name__\s*==", re.M)),
    ("javascript", re.compile(r"^\s*(?:const|let|var)\s+\w+\s*=|=>\s*\{|"
                              r"\bfunction\s+\w+\s*\(|\brequire\(|\bconsole\.log\(", re.M)),
    ("dockerfile", re.compile(r"^\s*FROM\s+\S+|^\s*RUN\s+\S+|^\s*ENTRYPOINT\s*\[", re.M)),
    ("sql", re.compile(r"^\s*(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|"
                       r"CREATE\s+(?:TABLE|VIEW|INDEX))\b", re.I | re.M)),
    ("html", re.compile(r"<!DOCTYPE\s+html|<html[\s>]|<div[\s>]|<body[\s>]", re.I)),
    ("go", re.compile(r"^\s*package\s+\w+|^\s*func\s+\w+\s*\(|\bfmt\.Print", re.M)),
    ("rust", re.compile(r"^\s*fn\s+\w+\s*\(|\blet\s+mut\s+|println!\(", re.M)),
    # C and C++ share a family (same_language), so telling them apart never
    # changes a drift verdict -- but it decides the file extension and what the
    # notice says, and calling a plain C program "c++" is how a drift message
    # came to name a language the user had never mentioned.
    ("c++", re.compile(r"\bstd::\w+|#include\s*<(?:iostream|vector|string|map|"
                       r"memory|algorithm)>|\bnullptr\b|\btemplate\s*<|"
                       r"\bnamespace\s+\w+|\bpublic:|\bcout\s*<<", re.M)),
    ("c", re.compile(r"#include\s*<(?:stdio|stdlib|string|unistd|errno|"
                     r"sys/\w+)\.h>|\bprintf\s*\(|\bmalloc\s*\(|"
                     r"\btypedef\s+struct\b|\bsize_t\b", re.M)),
    ("java", re.compile(r"\bpublic\s+(?:static\s+)?(?:class|void)\b|\bSystem\.out\.", re.M)),
    # Verb-Noun cmdlets are the most distinctive thing about PowerShell and they
    # appear mid-line far more often than at the start of one, which is why
    # anchoring to the line start left a module scoring no higher than the
    # generic `$var` shell signature and coming back unidentified.
    ("powershell", re.compile(r"\b(?:Get|Set|New|Remove|Start|Stop|Write|Test|"
                              r"Invoke|Import|Export|Select|Where|ForEach|Add|"
                              r"Clear|Restart|ConvertTo|ConvertFrom)-[A-Z]\w+|"
                              r"\$PSVersionTable|\$PSItem|\bparam\s*\(\s*\[|"
                              r"-ErrorAction\b|\[string\]\s*\$|"
                              r"^\s*function\s+\w+-\w+", re.M)),
    ("json", re.compile(r'^\s*[\[{]\s*$|^\s*"[\w-]+"\s*:\s*(?:"|\d|\[|\{|true|'
                        r'false|null)', re.M)),
]

# How much clearer the winner has to be than the runner-up. A classifier whose
# verdict gates a corrective regeneration has to shrug more readily than it
# guesses: it called a C program "c++" and a C program "json" in one session,
# and each wrong verdict cost a full regeneration and a question to the user.
SIGNATURE_MARGIN = 2


def code_language(code: str) -> str:
    """The language a block of code is written in, or "" when unclear.

    Scores every signature and requires a clear winner, because a shell script
    full of `$(...)` and a Python script full of `import` are easy, and anything
    in between should return "" rather than a coin flip.
    """
    text = code or ""
    if not text.strip():
        return ""
    scores: dict[str, int] = {}
    for name, pattern in _SIGNATURES:
        hits = len(pattern.findall(text))
        if hits:
            scores[name] = hits
    if not scores:
        return ""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, count = ranked[0]
    if len(ranked) > 1:
        runner = ranked[1][1]
        # Not just "ahead": clearly ahead. One stray `int main(` in a shell
        # script should not outvote four lines of shell. Either twice the
        # evidence or two clear hits more, so the rule does not become brittle
        # once both languages have several signals.
        if count < runner * SIGNATURE_MARGIN and count - runner < SIGNATURE_MARGIN:
            # Two members of the same family both scoring is agreement, not a
            # tie: `#include <stdio.h>` and `std::string` in one file is still
            # a C-family file, and the stronger reading wins.
            if not same_language(best, ranked[1][0]):
                return ""
        if count < 2 and runner >= 1:
            return ""                  # a single hit each way decides nothing
    return best


def same_language(left: str, right: str) -> bool:
    """True when two language names are the same or close enough to not be drift."""
    if not left or not right or left == right:
        return True
    return any(left in fam and right in fam for fam in _LANG_FAMILIES)


_FILENAME = re.compile(r"\b([A-Za-z0-9][\w.-]{0,60}\.(?:sh|bash|zsh|py|js|ts|rs|go|"
                       r"cpp|cc|c|h|hpp|cs|java|rb|php|pl|sql|html|css|swift|kt|"
                       r"dart|lua|ps1|yaml|yml|json|tf|mk))\b")


# Lines that name someone ELSE's file. Reading one as the artifact's own name is
# how a C program came to be called "stdio.h" for the rest of a session.
_IMPORT_LINE = re.compile(
    r"(?im)^\s*(?:#\s*(?:include|import)\b|(?:from|import)\s|"
    r"using\s+\w|require\s*\(|\.\s*load\b|source\s+|Import-Module\b)")


def detect_filename(text: str) -> str:
    """A filename the conversation itself named, so the artifact keeps its name.

    Import and include lines are skipped: `#include <stdio.h>` names a system
    header, not the file being written, and adopting it puts a name in the brief
    that the model then tries to live up to.
    """
    for line in (text or "").split("\n"):
        if _IMPORT_LINE.match(line):
            continue
        match = _FILENAME.search(line)
        if match:
            return match.group(1)
    return ""


# --------------------------------------------------------------------------- #
# Intent classification: new task, modification, correction                    #
# --------------------------------------------------------------------------- #

# An explicit "drop what we were doing".
# Anchored where anchoring is what distinguishes abandoning the task from
# editing it. "use a hashtable INSTEAD OF the array" is one of the commonest
# ways to ask for a small change, and it was resetting the state: objective,
# requirements, corrections and artifact discarded, and the database row
# deleted. A reset is a leading statement about the job as a whole.
_RESET = re.compile(r"(?i)\b(forget\s+(?:that|it|this|the\s+\w+|about\s+\w+)|"
                    r"new\s+task|different\s+task|start\s+over|scrap\s+(?:that|it)|"
                    r"change\s+of\s+plan)\b|"
                    r"^\s*(?:ok(?:ay)?[,\s]+|no[,\s]+)?instead\s+of\s+"
                    r"(?:that|this|it|all\s+that)\b")

# Creating something new, as opposed to changing what exists.
_CREATE_VERB = re.compile(r"(?i)\b(write|create|generate|build|implement|produce|"
                          r"make\s+me|give\s+me|draft)\b")

# Pointing at the thing that already exists.
_REFERS_TO_EXISTING = re.compile(
    r"(?i)\b(it|its|it's|that|this|the\s+(?:script|code|file|function|program|query|"
    r"same)|same\s+(?:script|file|thing)|above|previous|existing)\b")

# Asking for a change to the current artifact.
_MODIFY_LEAD = re.compile(
    r"(?i)^\s*(?:and\s+|also\s+|now\s+|then\s+|please\s+|can\s+you\s+)*"
    r"(add|adds|append|extend|include|improve|enhance|harden|fix|correct|update|"
    r"modify|change|refactor|rename|remove|drop|handle|support|make|more|expand|"
    r"tidy|clean|optimi[sz]e|simplify|document|comment|split|merge|wrap|"
    r"validate|check)\b")

# A correction of the assistant's last answer. "stop you were asked a bash
# script" is not another conversational turn: it is a state update, and it has
# to be treated as one or the model apologises and drifts again.
# A correction becomes a permanent, HIGHEST-PRIORITY line in the brief, so a
# false positive here is expensive and never expires. "stop" and "no" therefore
# have to be the LEADING word -- the user interrupting -- rather than any
# occurrence: "add a stop button", "make it stop on the first error" and "the
# script should stop when the disk is full" are all ordinary feature requests,
# and all three were being filed as corrections of a mistake nobody made.
_CORRECTION = re.compile(
    r"(?i)^\s*(?:stop\b|no[,!.\s]+(?:i|it|that|the|we)\b)|"
    r"(\byou\s+(?:were\s+asked|asked|wrote|gave|produced|switched)|"
    r"\bi\s+(?:asked|said|wanted|requested)\b|\bthat'?s?\s+not\b|\bthats\s+not\b|"
    r"\bnot\s+what\s+i\b|\bwrong\s+(?:language|file|script|thing)\b|"
    r"\bwe\s+(?:were|are)\s+(?:doing|working)\b|"
    r"\bgo\s+back\s+to\b|\bback\s+to\s+the\b|\bunrelated\b)")


# Re-expressing the SAME job in another language. "port it to c" is neither a
# new task (the objective and every requirement still stand) nor a modification
# (the language is exactly what changes), and classifying it as the latter is
# what kept a PowerShell task active through four consecutive requests for C.
#
# The object between the verb and the preposition is restricted to a reference
# to the artifact, so "convert the output to json" -- a change to what the code
# PRODUCES -- is not read as a change to what the code IS WRITTEN IN.
_PORT_REQUEST = re.compile(
    r"(?i)\b(?:port|re-?port|convert|translate|rewrite|re-?write|re-?implement|"
    r"reimplement|migrate|re-?do)\s+"
    r"(?:(?:it|this|that|them|these|those|everything|all\s+of\s+(?:it|this|that)|"
    # "the script", "the whole thing", "the powershell module": up to two words
    # of qualifier before a noun that means the artifact. The noun list is the
    # gate -- "convert the OUTPUT to json" changes what the code produces, not
    # what it is written in, and must not match.
    r"the\s+(?:\w+\s+){0,2}(?:script|code|file|program|programme|module|"
    r"function|thing))\s+)?"
    r"(?:in|into|to|as|using)\s+")

# A bare single-letter language at the very start of the tail, where the
# preposition has already been consumed and there is nothing ambiguous left.
_BARE_LANG_HEAD = re.compile(r"(?i)^\s*(c|go|r)\b(?![\w+#])")


def parse_port_request(message: str) -> str:
    """The language a "port it to X" message retargets the task to, or "".

    Returns only the language: a port keeps the objective, the requirements and
    the corrections, so nothing else about the state changes.
    """
    text = " ".join((message or "").split())
    if not text or len(text) > 300:
        return ""
    match = _PORT_REQUEST.search(text)
    if not match:
        return ""
    # The tail only, so "rewrite the python script in rust" retargets to rust
    # rather than reading the language it is being ported AWAY from.
    tail = text[match.end():match.end() + 60]
    named = detect_language(tail, allow_bare=True)
    if named:
        return named
    bare = _BARE_LANG_HEAD.match(tail)
    return _BARE_LANG_CANON.get(bare.group(1).lower(), "") if bare else ""


def is_reset_request(message: str) -> bool:
    """True for an explicit "forget that" style abandonment of the task."""
    return bool(_RESET.search(message or ""))


def starts_new_task(message: str) -> bool:
    """True when this message asks for a NEW artifact rather than a change.

    This is the boundary that keeps a legitimate task switch working: "now
    forget that; write a python script" must reset the state, while "add more
    checks" must not. A creation verb plus a code request, and no reference to
    the existing artifact, is a new task.
    """
    text = (message or "").strip()
    if not text:
        return False
    if is_reset_request(text):
        return True
    if not is_code_request(text) or is_continue_request(text):
        return False
    if parse_correction(text):
        return False                    # a correction names a language too
    if not _CREATE_VERB.search(text):
        return False
    # "write the rest of it", "build it out" are the same artifact.
    head = text[:120]
    if _REFERS_TO_EXISTING.search(head) and not detect_language(head):
        return False
    return True


def is_modification_request(message: str) -> bool:
    """True for "add more checks", "also handle X": a change to the artifact."""
    text = (message or "").strip()
    if not text or len(text) > 600:
        return False
    if starts_new_task(text):
        return False
    if _MODIFY_LEAD.match(text):
        return True
    # Pointing at the artifact is only a modification when something is being
    # asked OF it: "fix it", "make that portable". A question about it ("is that
    # right?") is conversation, and recording it as a requirement would put it in
    # front of the model on every later turn.
    return bool(_REFERS_TO_EXISTING.search(text[:60]) and CODE_INTENT.search(text))


def parse_correction(message: str) -> dict | None:
    """A user correction, as the state updates it implies.

    Returns None for an ordinary message. Otherwise a dict with the corrected
    fields (only those the message actually states) plus the message text, so
    the brief can quote the user's own words back at the model.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    if not _CORRECTION.search(text):
        return None
    if is_reset_request(text):
        return None                    # abandoning the task, not correcting it
    update: dict[str, Any] = {"text": text}
    language = detect_language(text)
    if language:
        update["language"] = language
    platform = detect_platform(text)
    if platform:
        update["platform"] = platform
    return update


# --------------------------------------------------------------------------- #
# The state object                                                             #
# --------------------------------------------------------------------------- #

# Requirements/corrections kept. Enough to carry a working session, bounded so
# the brief cannot grow without limit.
MAX_REQUIREMENTS = 12
MAX_CORRECTIONS = 4

# Complete versions kept for rollback. Small: this is persisted per conversation
# and its only job is to survive the handful of turns in which a bad generation
# is noticed.
ARTIFACT_HISTORY = 3

# A replacement this much smaller than what is already held is treated as a
# possible loss rather than an edit, and is only accepted when it still parses.
# Chosen from the real regression: 3642 chars became 1993, a 0.55 ratio, and the
# 1993 did not parse.
SHRINK_RATIO = 0.7


@dataclass
class TaskState:
    """What the model is currently working on, independent of the transcript."""

    task_type: str = ""                 # code_generation | "" (unknown/chat)
    language: str = ""
    platform: str = ""
    objective: str = ""                 # the user's original ask, verbatim
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    latest_request: str = ""
    artifact: str = ""                  # the current code/document itself
    artifact_name: str = ""
    artifact_type: str = ""
    artifact_version: int = 0
    artifact_complete: bool = True      # False when the last one was cut off
    # The last few complete versions, newest last, so a turn that destroys the
    # artifact can be undone. A truncated generation never lands here.
    artifact_history: list[dict] = field(default_factory=list)
    # A cut-off generation, kept BESIDE the last complete artifact rather than
    # on top of it. Overwriting `artifact` with a partial is how a known-good
    # 3642-character module became 1116 characters of truncated text that the
    # next turn then "continued" into a spliced duplicate.
    partial_artifact: str = ""
    # Why the most recent replacement was refused, for the turn to report.
    rejected_reason: str = ""
    # Continuation bookkeeping: which generation produced the partial artifact
    # and how far it got, so "continue" does not have to rediscover the task.
    generation_id: str = ""
    partial_chars: int = 0
    output_kind: str = "prose"          # code | document | prose
    # Where the last create_document call wrote, and in what format. The
    # artifact fields carry the document's Markdown SOURCE, because the rendered
    # .pdf/.docx is bytes the model can never read back -- so the source is what
    # a follow-up edits, and these two say which file to write it to again.
    document_path: str = ""
    document_format: str = ""
    updated_at: float = 0.0

    # -- lifecycle ------------------------------------------------------- #

    def is_active(self) -> bool:
        """True when there is a task worth telling the model about."""
        return bool(self.task_type and (self.objective or self.artifact))

    def is_code_task(self) -> bool:
        return self.task_type == "code_generation"

    def is_document_task(self) -> bool:
        return self.task_type == "document"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskState":
        if not data:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        clean = {k: v for k, v in dict(data).items() if k in fields}
        # Lists arrive as lists from JSON, but be defensive: a corrupted row
        # must not raise inside a request.
        for key in ("requirements", "constraints", "corrections"):
            value = clean.get(key)
            clean[key] = [str(v) for v in value] if isinstance(value, list) else []
        history = clean.get("artifact_history")
        clean["artifact_history"] = [
            v for v in history if isinstance(v, dict) and v.get("artifact")
        ][-ARTIFACT_HISTORY:] if isinstance(history, list) else []
        return cls(**clean)

    # -- updates --------------------------------------------------------- #

    def add_requirement(self, text: str) -> None:
        line = " ".join((text or "").split())[:200]
        if not line or line in self.requirements:
            return
        self.requirements.append(line)
        del self.requirements[:-MAX_REQUIREMENTS]

    def add_correction(self, text: str) -> None:
        line = " ".join((text or "").split())[:200]
        if not line or line in self.corrections:
            return
        self.corrections.append(line)
        del self.corrections[:-MAX_CORRECTIONS]

    def set_objective(self, message: str) -> None:
        """Adopt this message as the task's objective.

        Only a code request starts a task today: that is the failure mode this
        module was written for, and inventing a task type for ordinary chat
        would put a brief in front of every greeting. Everything else leaves the
        state inactive, so those turns behave exactly as they did before.
        """
        if not is_code_request(message):
            return
        self.objective = " ".join((message or "").split())[:800]
        self.task_type = "code_generation"
        language = detect_language(message)
        if language:
            self.language = language
        platform = detect_platform(message)
        if platform:
            self.platform = platform
        name = detect_filename(message)
        if name:
            self.artifact_name = name
        if self.is_code_task():
            self.output_kind = "code"

    def start_document(self, message: str, suffix: str) -> None:
        """Adopt this message as a DOCUMENT task.

        set_objective deliberately starts a task only for a code request, so
        "create a pdf for construction invoice" left the state inactive -- and
        the follow-up "now make it in the style of texcel.be" then arrived with
        no objective, no artifact and nothing saying a document was under
        construction. It was answered as a fresh question, in prose, with no
        tools, and no file was produced.
        """
        if self.is_active() and not self.is_document_task():
            return                      # a code task is already running
        self.task_type = "document"
        self.output_kind = "document"
        if not self.objective:
            self.objective = " ".join((message or "").split())[:800]
        if suffix and not self.document_format:
            self.document_format = suffix
        if not self.document_path:
            self.document_path = self.default_document_name()
            self.artifact_name = self.document_path
        if suffix and suffix != self.document_format:
            # A format change on the same document ("now make it a word
            # document") keeps the source and moves the file, so the brief, the
            # rules and the next create_document call all name the new path
            # rather than telling the model to write a .docx to a .pdf.
            if self.document_path and "." in self.document_path:
                self.document_path = self.document_path.rsplit(".", 1)[0] + suffix
                self.artifact_name = self.document_path
            self.document_format = suffix
            self.artifact_type = ""
        self.updated_at = time.time()

    def record_document(self, path: str, suffix: str, source: str,
                        type_name: str = "") -> None:
        """Store the Markdown a create_document call rendered, as the artifact.

        Versioned through the same history as a code artifact, so the previous
        wording of a document is recoverable after a bad edit. Deliberately does
        NOT go through record_artifact: that one is about code, and its loss
        guard, filename detection and language bookkeeping would all be applied
        to prose they were not written for.
        """
        body = (source or "").strip()
        if not body:
            return
        self.task_type = "document"
        self.output_kind = "document"
        if path:
            self.document_path = path
            self.artifact_name = path
        if suffix:
            self.document_format = suffix
        if type_name:
            self.artifact_type = type_name
        self.rejected_reason = ""
        if body == self.artifact:
            return                      # same source rendered again
        self._push_history()
        self.artifact = body
        self.artifact_version += 1
        self.artifact_complete = True
        self.partial_artifact = ""
        self.partial_chars = 0
        self.updated_at = time.time()

    def apply_correction(self, update: dict) -> None:
        """A correction is a state update, not just another message."""
        if update.get("language"):
            self.language = update["language"]
            self.task_type = self.task_type or "code_generation"
            self.output_kind = "code"
            # The artifact type follows the language, and a stale one ("python
            # module" on a bash task) is exactly the contradiction that made the
            # model apologise and then drift again.
            self.artifact_type = _LANG_ARTIFACT_TYPE.get(self.language, self.artifact_type)
        if update.get("platform"):
            self.platform = update["platform"]
        self.add_correction(update.get("text") or "")

    def _push_history(self) -> None:
        """Keep the version being replaced, so a bad turn can be undone."""
        if not self.artifact or not self.artifact_complete:
            return
        self.artifact_history.append({
            "artifact": self.artifact,
            "version": self.artifact_version,
            "language": self.language,
            "artifact_name": self.artifact_name,
        })
        del self.artifact_history[:-ARTIFACT_HISTORY]

    def _is_loss(self, body: str) -> bool:
        """True when accepting this replacement would destroy working code.

        The test is structural damage, not size. An edit is allowed to make a
        file much smaller -- the user asked to cut something -- and it is
        allowed to be wrong in ways no parser can see. What is refused is a
        replacement with unbalanced braces or a block spliced in twice, because
        storing that makes it "the current artifact" and every later turn is
        briefed to keep building on it.
        """
        if not self.artifact_complete:
            return False                # nothing whole is at risk
        if not looks_damaged(body, self.language):
            return False
        shrunk = (f", and {len(body)} chars against v{self.artifact_version}'s "
                  f"{len(self.artifact)}"
                  if len(body) < len(self.artifact) * SHRINK_RATIO else "")
        self.rejected_reason = (
            f"the new version has unclosed or duplicated blocks{shrunk}; keeping "
            f"v{self.artifact_version}")
        return True

    def rollback(self) -> bool:
        """Restore the previous complete version. True when one was restored."""
        if not self.artifact_history:
            return False
        previous = self.artifact_history.pop()
        self.artifact = str(previous.get("artifact") or "")
        self.artifact_version = int(previous.get("version") or 0)
        self.language = str(previous.get("language") or self.language)
        self.artifact_name = str(previous.get("artifact_name") or self.artifact_name)
        self.artifact_complete = True
        self.partial_artifact = ""
        self.partial_chars = 0
        self.updated_at = time.time()
        return True

    def resumable(self) -> str:
        """The text a "continue" should resume: the partial, else a cut-off artifact."""
        if self.partial_artifact:
            return self.partial_artifact
        return self.artifact if not self.artifact_complete else ""

    def retarget(self, language: str) -> bool:
        """Switch the task to another language, keeping the job it is doing.

        A port is neither a reset nor a modification. Resetting would throw away
        the objective and every requirement the user has accumulated; treating
        it as a modification is what recorded "port it to c" as a requirement ON
        a PowerShell task and then flagged every C reply as drift for the rest
        of the session. So: same objective, same requirements, new language, and
        the old file retired to history because it is not the new file.
        """
        if not language or same_language(language, self.language):
            return False
        previous = self.language
        self._push_history()
        self.language = language
        self.task_type = self.task_type or "code_generation"
        self.output_kind = "code"
        self.artifact_type = _LANG_ARTIFACT_TYPE.get(language, "")
        self.artifact = ""
        self.partial_artifact = ""
        self.artifact_version = 0
        self.artifact_complete = True
        self.partial_chars = 0
        self.artifact_name = ""
        self.artifact_name = self.default_artifact_name()
        if previous:
            self.add_requirement(
                f"this is a port of the {previous} version; keep the same "
                f"behaviour, expressed idiomatically in {language}")
        self.updated_at = time.time()
        return True

    def record_artifact(self, code: str, language: str = "", complete: bool = True) -> None:
        """Store a produced artifact as the current one, bumping the version.

        Guarded, because this is where a bad turn used to destroy a good file.
        A truncated generation is parked in partial_artifact beside the last
        complete version instead of replacing it, and a complete replacement
        that is both much smaller and structurally broken is refused outright.
        Everything accepted pushes the version it replaces onto artifact_history
        so the loss is recoverable either way.
        """
        body = (code or "").strip()
        if not body:
            return
        self.rejected_reason = ""
        if body == self.artifact and complete == self.artifact_complete:
            return                      # same artifact seen again; not a new version
        if not complete:
            # A cut-off generation. It is what a "continue" resumes, but it is
            # not the artifact until it finishes -- unless there is nothing
            # better, or it is already longer than what we hold, which makes it
            # genuine progress rather than a loss. artifact_complete keeps
            # describing `artifact`, never the partial: conflating the two is
            # what let a truncated reply mark a whole file as truncated and then
            # be overwritten by the next bad generation.
            self.partial_artifact = body
            self.partial_chars = len(body)
            if self.artifact and self.artifact_complete \
                    and len(body) < len(self.artifact):
                # Normal operation, not a rejection: the answer already carries
                # a truncation note saying it was cut off, and a second notice
                # explaining that the whole version was kept is noise on top of
                # it. rejected_reason stays for the case the user cannot see
                # for themselves -- a COMPLETE reply that destroyed the file.
                return
        elif self.artifact and self._is_loss(body):
            return
        self._push_history()
        self.artifact = body
        self.artifact_version += 1
        self.artifact_complete = complete
        self.partial_chars = 0 if complete else len(body)
        if complete:
            self.partial_artifact = ""
        if language:
            self.language = language
            self.artifact_type = _LANG_ARTIFACT_TYPE.get(language, self.artifact_type)
        if not self.artifact_type and self.language:
            self.artifact_type = _LANG_ARTIFACT_TYPE.get(self.language, "")
        if not self.platform:
            # Only from the artifact's own commands, and only when nothing has
            # been stated: a guess must never override what the user said.
            self.platform = platform_from_code(body)
        if not self.artifact_name:
            self.artifact_name = detect_filename(body) or self.default_artifact_name()
        self.output_kind = "code"
        self.updated_at = time.time()

    # Words that say what KIND of file it is, not what it is about. A document
    # named "word-document-for.docx" tells the user nothing.
    _DOC_NAME_FILLER = {
        "create", "make", "write", "draft", "generate", "produce", "build",
        "prepare", "give", "need", "want", "please", "document", "documents",
        "file", "report", "word", "excel", "powerpoint", "spreadsheet",
        "workbook", "deck", "presentation", "markdown", "text", "about",
        "with", "from", "into", "that", "this", "them", "they", "look",
        "much", "find", "then", "your", "yours", "some", "very", "also",
        "for", "you", "can", "the", "and", "are", "its", "our", "all",
        "one", "new", "out", "use", "www", "http", "https", "com", "please",
    }

    def default_document_name(self) -> str:
        """A short descriptive file name derived from the objective.

        The model should not have to invent one: told to pick a name, a 3B model
        picks whatever the rules happen to call it, which is how a turn produced
        `"path": "the document.docx"`.
        """
        suffix = self.document_format or ".pdf"
        seen: list[str] = []
        for word in re.findall(r"[a-z0-9]{3,}", (self.objective or "").lower()):
            if word not in self._DOC_NAME_FILLER and word not in seen:
                seen.append(word)
        stem = "-".join(seen[:4]) if seen else "document"
        return f"{stem[:60].strip('-') or 'document'}{suffix}"

    def default_artifact_name(self) -> str:
        extension = _LANG_EXT.get(self.language, "txt")
        if extension == "Dockerfile":
            return "Dockerfile"
        stem = "artifact"
        filler = {"write", "create", "script", "scripts", "program", "the", "and",
                  "for", "that", "with", "output", "clear", "format", "able",
                  "generate", "build", "make", "give", "which", "what", "some",
                  "please", "using", "into", "from", "file", "code", "reason",
                  "exist", "exists", "etc", "are", "have", "having", "this"}
        words = [w for w in re.findall(r"[a-z]{3,}", (self.objective or "").lower())
                 if w not in filler and w not in _LANG_ALIASES]
        if words:
            stem = "_".join(words[:3])
        return f"{stem}.{extension}"

    def note_request(self, message: str) -> dict | None:
        """Fold the newest user message into the state.

        Returns the correction dict when the message was a correction, so the
        caller can log it and widen the reply budget accordingly.
        """
        self.latest_request = " ".join((message or "").split())[:400]
        correction = parse_correction(message)
        if correction is not None:
            self.apply_correction(correction)
            return correction
        # A port, checked before the modification path: "port it to c" reads as
        # a modification (it points at the artifact and carries a code verb),
        # and note_request deliberately refuses to change the language on a
        # modification. Retargeting has to happen here or not at all.
        if self.is_active():
            target = parse_port_request(message)
            if target and self.retarget(target):
                self.add_requirement(message)
                platform = detect_platform(message)
                if platform:
                    self.platform = platform
                return None
        if not self.objective:
            self.set_objective(message)
            if self.objective:
                return None
        if is_modification_request(message) and not is_continue_request(message):
            self.add_requirement(message)
            # A modification may still name a platform ("...on linux too") but
            # never silently changes the language: that is what starts_new_task
            # is for, and a language word inside "add a python-style docstring"
            # must not retarget a bash task.
            platform = detect_platform(message)
            if platform:
                self.platform = platform
        return None

    def reset(self) -> None:
        """Forget the task. Used when the user starts a genuinely new one."""
        fresh = TaskState()
        for name in self.__dataclass_fields__:                  # noqa: SLF001
            setattr(self, name, getattr(fresh, name))

    def should_reset(self, message: str) -> bool:
        """True when this message abandons the task rather than changing it.

        Stricter than starts_new_task on purpose. Mid-task, "write a helper
        function that parses the smartctl output" reads as a creation request
        and is really the next piece of the same job; throwing the artifact away
        there is the same class of failure as forgetting it. So a reset needs an
        explicit "forget that", a different language, or no artifact to lose.
        """
        if not self.is_active():
            return False
        if is_reset_request(message):
            return True
        # A port changes the language without abandoning the job, and
        # note_request handles it. Resetting here would drop the objective and
        # every requirement the user has built up, which is the same loss the
        # reset guard exists to prevent.
        if parse_port_request(message):
            return False
        if not starts_new_task(message):
            return False
        language = detect_language(message)
        if language and self.language and not same_language(language, self.language):
            return True
        return not self.artifact

    def absorb_history(self, history: list[dict] | None) -> "TaskState":
        """Fold the visible transcript into this state, oldest turn first.

        Deterministic and idempotent: running it every turn over the same
        history yields the same state, which is what makes it safe to run over
        the PERSISTED state rather than beside it. Requirements dedupe, an
        artifact identical to the one already held does not bump the version,
        and a reset inside the history clears what came before it.

        Only the newest assistant turn is read for the artifact: earlier ones
        are superseded by it, and re-recording each of them every turn would
        inflate the version number without changing the file.
        """
        turns = [t for t in (history or [])
                 if t.get("role") in ("user", "assistant") and t.get("content")]
        answers: list[str] = []
        for turn in turns:
            content = str(turn["content"])
            if turn["role"] == "user":
                if self.should_reset(content):
                    self.reset()
                    answers.clear()
                self.note_request(content)
            else:
                answers.append(content)
        # Newest artifact wins, so scan back for the most recent answer that
        # actually carries one: the newest turn is often prose, a question or a
        # drifted reply, and the artifact is a few turns above it. Exactly one
        # record_artifact call, so replaying the same history does not inflate
        # the version.
        for content in reversed(answers):
            found = self._artifact_in(content)
            if found is not None:
                self.record_artifact(*found)
                break
        return self

    def _artifact_in(self, answer: str) -> "tuple[str, str, bool] | None":
        """(code, language, complete) for the artifact in one assistant turn.

        None when the turn carries no code, or carries code that contradicts the
        task. That second case matters: a drifted answer is a mistake, not a new
        version, and adopting it would make the state describe the mistake --
        the Python the model wandered into becomes "the current bash artifact"
        and the next turn is briefed to keep modifying it.
        """
        text = answer or ""
        complete = not was_truncated(text)
        blocks = extract_code_blocks(strip_truncation_note(text))
        if not blocks:
            return None
        # Patch reply first: on a long artifact the model is asked for only the
        # sections it changes, so the reply is legitimately a fraction of the
        # file and must be merged rather than allowed to replace it. apply_patch
        # returns None unless every block names a function the artifact already
        # has, so a genuine rewrite still falls through to the path below.
        if complete and self.artifact:
            bodies = [code for _, code in blocks]
            if sum(len(b) for b in bodies) < len(self.artifact) * SHRINK_RATIO:
                merged = self.apply_patch(bodies)
                if merged is not None:
                    return merged, self.language, True
        # Prefer a block in the task's language; otherwise the longest one. A
        # reply often shows a usage example alongside the artifact itself.
        best_code, best_lang, best_score = "", "", -1
        for info, code in blocks:
            lang = canonical_language(info) or code_language(code)
            score = len(code) + (100_000 if self.language and same_language(
                lang, self.language) else 0)
            if score > best_score:
                best_code, best_lang, best_score = code, lang, score
        if not best_code.strip():
            return None
        if self.language and best_lang and not same_language(best_lang, self.language):
            return None
        # Only adopt a language the block genuinely evidences: an unlabelled
        # fence must not blank out a known one.
        adopt = best_lang if (best_lang and (not self.language
                                             or same_language(best_lang, self.language))) else ""
        return best_code, adopt, complete

    def apply_patch(self, blocks: list[str]) -> str | None:
        """Splice changed functions back into the artifact, or None if unsafe.

        The patch lane's other half. Asking a 3B model to re-emit a 3600-char
        module to add a try/catch costs 45 seconds and risks the whole file on
        every turn; asking for just the changed functions costs a fraction of
        that, but only pays off if the reply can be merged back deterministically
        rather than replacing what it does not contain.

        Confidence is the whole design. Every definition in every block must
        already exist in the artifact, and the blocks together must not cover
        the whole file (that is a rewrite, not a patch). Anything else returns
        None and the caller falls back to treating the reply as a full file.
        """
        if not self.artifact or not self.language or not blocks:
            return None
        current = definitions(self.artifact, self.language)
        if not current:
            return None
        index = {name: (start, end) for name, start, end in current}
        edits: dict[str, str] = {}
        for block in blocks:
            body = (block or "").strip()
            if not body:
                continue
            found = definitions(body, self.language)
            if not found:
                return None             # not a definition: cannot place it
            for name, start, end in found:
                if name not in index:
                    return None         # new function: a rewrite, not a patch
                edits[name] = body[start:end].rstrip()
        if not edits or len(edits) >= len(index):
            return None                 # touches everything: just take the file
        merged = self.artifact
        # Apply back-to-front so earlier offsets stay valid.
        for name, (start, end) in sorted(index.items(), key=lambda kv: -kv[1][0]):
            if name in edits:
                merged = merged[:start] + edits[name] + merged[end:]
        return merged

    def note_answer(self, answer: str) -> None:
        """Fold an assistant turn into the state: its artifact and completeness."""
        # A document task's artifact is the Markdown handed to create_document,
        # recorded from the tool call itself. The visible answer is one sentence
        # about it, and a stray fenced block in that sentence must not be
        # adopted as "the document" -- that would replace the real source with a
        # fragment, and the next edit would build on the fragment.
        if self.is_document_task():
            return
        found = self._artifact_in(answer)
        if found is not None:
            self.record_artifact(*found)

    def begin_generation(self) -> str:
        self.generation_id = uuid.uuid4().hex[:12]
        return self.generation_id

    # -- the prompt block ------------------------------------------------ #

    def brief(self, artifact_chars: int = 6000, verify_hint: bool = False,
              mode: str = "build") -> str:
        """The ACTIVE TASK block: what the model must be told on every turn.

        Rides on the current user turn rather than the system prompt, for two
        reasons: trim_to_context never drops the user turn (so trimming can
        never remove the only copy of the artifact), and the tail of the prompt
        is the part a server-side prefix cache is free to vary without a full
        re-prefill.
        """
        if not self.is_active():
            return ""
        lines = ["ACTIVE TASK (this is what you are working on; it survives "
                 "trimmed history)"]
        pretty = {"code_generation": "code generation",
                  "document": "document"}.get(self.task_type, self.task_type)
        lines.append(f"Type: {pretty}")
        if self.language:
            lines.append(f"Language: {self.language}")
        if self.platform:
            lines.append(f"Platform: {self.platform}")
        if self.artifact_name or self.artifact_type:
            state = ("complete" if self.artifact_complete else "TRUNCATED (was cut "
                     "off at the reply limit)")
            if self.artifact_complete and self.partial_artifact:
                state = ("complete, but a later attempt was cut off and is "
                         "waiting to be finished")
            version = f"v{self.artifact_version}" if self.artifact_version else "not written yet"
            lines.append(f"Artifact: {self.artifact_name or '(unnamed)'} "
                         f"({self.artifact_type or 'file'}, {version}, {state})")
        if self.objective:
            lines.append("\nORIGINAL OBJECTIVE:\n" + self.objective)
        if self.requirements:
            lines.append("\nCURRENT REQUIREMENTS (all of them still apply):\n"
                         + "\n".join(f"- {r}" for r in self.requirements))
        if self.constraints:
            lines.append("\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in self.constraints))
        if self.corrections:
            lines.append("\nCORRECTIONS FROM THE USER (highest priority, do not "
                         "repeat these mistakes):\n"
                         + "\n".join(f"- {c}" for c in self.corrections))
        if self.latest_request:
            lines.append("\nLATEST USER REQUEST:\n" + self.latest_request)
        if self.artifact:
            document = self.is_document_task()
            fence = "markdown" if document else (self.language or "")
            body = self._artifact_for_prompt(artifact_chars)
            purpose = ("this is what the user's question is about"
                       if mode == "explain" else "this is the thing to modify")
            label = ("CURRENT DOCUMENT SOURCE" if document else "CURRENT ARTIFACT")
            lines.append(f"\n{label} (v{self.artifact_version}) — {purpose}:"
                         f"\n```{fence}\n{body}\n```")
        lines.append("\n" + self.rules(verify_hint=verify_hint, mode=mode))
        return "\n".join(lines)

    def _artifact_for_prompt(self, artifact_chars: int) -> str:
        """The artifact, trimmed to fit, keeping both ends when it must be cut.

        The head carries the shebang, the options and the helper definitions;
        the tail carries whatever was being written when the budget ran out.
        Cutting the middle and saying so beats cutting either end silently.
        """
        body = self.artifact
        if artifact_chars <= 0 or len(body) <= artifact_chars:
            return body
        head = int(artifact_chars * 0.6)
        tail = artifact_chars - head
        if tail <= 0:
            return body[:artifact_chars]
        omitted = len(body) - head - tail
        marker = (f"\n\n# ... {omitted} characters omitted to fit the context window. "
                  "Do NOT rewrite the omitted part from memory: keep it as it is and "
                  "output only the sections you change, saying where they go ...\n\n")
        return body[:head].rstrip() + marker + body[-tail:].lstrip()

    def rules(self, verify_hint: bool = False, mode: str = "build") -> str:
        """The non-negotiables, derived from the state rather than hard-coded.

        `mode` is what the turn is actually for, and it has to be honest:

            build     produce or update the whole artifact (the default)
            explain   answer a question ABOUT the artifact, in prose
            patch     emit only the sections that change

        The failure this parameter exists for: the "reply with the complete
        updated file" rule was unconditional on a code task, so a question
        ("what is the logic behind this function?") was briefed to dump a
        3642-character module -- into the 512-token chat budget the same turn
        had been given, because the budget was conditional on the message and
        the rule was not. It was cut off mid-function every time and the
        question was never answered.
        """
        language = self.language or "the same language"
        rules = ["RULES FOR THIS REPLY:"]
        if mode == "explain":
            rules.append("- The user asked a QUESTION about the work, not for a "
                         "change to it. Answer the question directly, in prose.")
            rules.append("- Do NOT re-emit the artifact. Quote at most the few "
                         "lines you are actually talking about.")
            rules.append("- If the honest answer is that the code above is wrong "
                         "or does not do what its comments claim, say so plainly.")
            return "\n".join(rules)
        if self.is_document_task():
            # There is only a document to UPDATE once one has been written. On
            # the first turn these rules used to fire anyway, with the path
            # falling back to the literal words "the document" -- so the model
            # was told to update a file that did not exist, at a made-up name,
            # and obediently wrote `"path": "the document.docx"`.
            name = self.document_path
            if self.artifact and name:
                rules.append(f"- You are updating an existing document: {name}. Its "
                             "current Markdown source is above.")
                rules.append("- Apply the user's latest request to that source, then "
                             f"call create_document with path \"{name}\" and the "
                             "COMPLETE revised Markdown in content. Keep everything "
                             "the user did not ask you to change, including the "
                             "existing headings, tables and placeholder fields.")
            else:
                rules.append(f"- Write the document the objective asks for and call "
                             f"create_document once, with path \"{name}\"."
                             if name else
                             "- Write the document the objective asks for and call "
                             "create_document once, with a short descriptive file "
                             f"name ending in {self.document_format or '.pdf'}.")
                rules.append("- Put the WHOLE document in the content argument as "
                             "Markdown: # headings, **bold**, - bullets, | tables |. "
                             "Start with a # title.")
            rules.append("- Do NOT write the document into your reply and do NOT use "
                         "write_file. Once the tool succeeds, say in one or two "
                         "sentences what you made or changed.")
            return "\n".join(rules)
        if self.artifact:
            rules.append(f"- Continue working on the existing {language} artifact above. "
                         "Preserve its existing behaviour unless the user asked to "
                         "change it.")
        else:
            rules.append(f"- Produce the {language} artifact the objective asks for.")
        rules.append(f"- Stay in {language}. Do not switch language, and do not "
                     "introduce unrelated files, classes, frameworks or architectures.")
        if self.is_code_task() and mode == "patch":
            rules.append(f"- The file is long and most of it is not changing. Reply "
                         f"with ONLY the functions or sections you actually change, "
                         f"each complete, each in its own ```{self.language or ''} "
                         f"block, and name where it goes. Do NOT reprint the "
                         f"unchanged parts and do NOT rewrite them from memory.")
        elif self.is_code_task():
            rules.append(f"- Reply with the complete updated file in ONE ```{self.language or ''} "
                         "block, plus at most two lines of explanation.")
            target = f" on {self.platform}" if self.platform else ""
            rules.append(f"- Use only commands, flags, fields and APIs you are certain "
                         f"exist{target}. Do not invent command-line options or output "
                         "field names. Where a value has to be parsed out of another "
                         "command's output, parse it defensively and handle the field "
                         "being absent, and say in a comment when something is "
                         "version-dependent.")
            if verify_hint:
                rules.append("- You may run a read-only command (for example a --help or "
                             "a plain list command) to confirm a flag or field exists "
                             "before relying on it.")
        return "\n".join(rules)

    def summary(self) -> str:
        """One compact line for logs and the internals trace."""
        bits = [f"type={self.task_type or '-'}", f"lang={self.language or '-'}"]
        if self.platform:
            bits.append(f"platform={self.platform}")
        bits.append(f"artifact={self.artifact_name or '-'}")
        bits.append(f"v{self.artifact_version}")
        bits.append("complete" if self.artifact_complete else "truncated")
        bits.append(f"{len(self.artifact)} chars")
        if self.requirements:
            bits.append(f"{len(self.requirements)} requirement(s)")
        if self.corrections:
            bits.append(f"{len(self.corrections)} correction(s)")
        return " ".join(bits)


def load_task_state(persisted: dict | None, history: list[dict] | None) -> TaskState:
    """The active task, from the persisted copy plus the visible transcript.

    Both sources are needed. The transcript is authoritative while it still
    contains the task, and it self-heals (a hand-edited history, an imported
    conversation). The persisted copy is what survives once trimming or the
    history-turns limit has removed the original request from the transcript
    entirely -- which is the case this whole module exists for.

    Absorbing the history INTO the stored state rather than merging two states
    keeps one code path: absorb_history is idempotent, so replaying the visible
    turns over what is already known adds only what is new.
    """
    return TaskState.from_dict(persisted).absorb_history(history)


def drifted_languages(task: TaskState, answer: str) -> list[str]:
    """The languages an answer is in, when none of them is the task's.

    Empty when the answer is fine, when it carries no identifiable code, or when
    there is no task to contradict. Callers that only want a yes/no use
    detect_drift; this exists so the turn can NAME the language it would switch
    to without parsing its own error message back apart.
    """
    if not task.is_code_task() or not task.language or not answer:
        return []
    found: list[str] = []
    for info, code in extract_code_blocks(answer):
        language = canonical_language(info) or code_language(code)
        if language:
            found.append(language)
    if not found:
        # No fenced code: check the bare body, for a model that skipped fences.
        language = code_language(strip_truncation_note(answer))
        if not language:
            return []
        found = [language]
    if any(same_language(language, task.language) for language in found):
        return []
    return sorted(set(found))


def detect_drift(task: TaskState, answer: str) -> str:
    """Why an answer contradicts the active task, or "" when it does not.

    Deliberately narrow. It fires only when the answer contains code whose
    language is confidently identifiable AND no part of it is in the task's
    language: "you asked for Bash and every block here is Python". A reply with
    no code, an unlabelled block, or a mix that includes the right language is
    left alone, because a validator that fires on ambiguity would block more
    good answers than bad ones.
    """
    found = drifted_languages(task, answer)
    if not found:
        return ""
    return (f"the reply is {'/'.join(found)} but the active task is "
            f"{task.language}")


__all__ = [
    "TaskState",
    "load_task_state",
    "detect_drift",
    "drifted_languages",
    "detect_language",
    "detect_platform",
    "platform_from_code",
    "canonical_language",
    "code_language",
    "same_language",
    "extract_code_blocks",
    "SIGNATURE_MARGIN",
    "detect_filename",
    "parse_correction",
    "starts_new_task",
    "parse_port_request",
    "ARTIFACT_HISTORY",
    "SHRINK_RATIO",
    "is_modification_request",
    "is_reset_request",
    "MAX_REQUIREMENTS",
    "MAX_CORRECTIONS",
]
