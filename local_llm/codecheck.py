"""Cheap correctness checks on generated code, before the user sees it.

The failure this exists for: a session produced ten versions of a PowerShell
module and a C port, and not one of them would run. `Set-WmiObject` is not a
cmdlet. `function Start-Service { ... Start-Service -Name $n }` shadows the
builtin and recurses until the stack dies. `Set-NetAdapter -Name="x"` is not
PowerShell syntax. The C had `IP_ADDR_INFO`, a `std::to_string(x) * y`, and its
own `StartService` shadowing the Win32 one. Nothing in the pipeline looked at
any of it, so every turn built on code that had never been parsed.

Two layers, cheapest first:

    structural_findings()   pure text, always runs, no tools, cannot fail
    syntax_findings()       a real parser (bash -n, py_compile, pwsh, cc), only
                            when shell execution is enabled and the tool exists

Everything returns a list of Finding and never raises: a check that breaks a
request is worse than no check.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import *  # noqa: F401,F403

# A finding is advisory. Nothing here blocks an answer; it annotates one.
SEVERITY_ORDER = {"error": 0, "warning": 1}

# How long a real parser gets. These are syntax-only invocations on a few KB of
# source, so a second is generous; the cap is there so a wedged toolchain cannot
# hold a chat turn open.
CHECK_TIMEOUT = 6.0

# Minimum run of identical lines before it is called a duplicated block. Six is
# above the noise floor for repeated boilerplate (closing braces, blank lines)
# and below the size of the smallest function a stitched continuation duplicates.
DUPLICATE_RUN = 6


@dataclass(frozen=True)
class Finding:
    """One problem found in generated code."""

    severity: str           # error | warning
    message: str
    line: int = 0

    def render(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        return f"{where}{self.message}"


# --------------------------------------------------------------------------- #
# Structural checks: pure text, no tools, every language                       #
# --------------------------------------------------------------------------- #

# Languages whose blocks are delimited by braces, so an imbalance is a real
# syntax error rather than a style. Excludes anything where braces appear inside
# ordinary strings often enough to make counting unreliable.
_BRACED = {"c", "c++", "c#", "java", "javascript", "typescript", "rust", "go",
           "swift", "kotlin", "scala", "php", "dart", "powershell", "json"}

# Strings and comments, so braces inside them are not counted. Ordered: a
# comment marker inside a string must not win, so strings come first.
_NOISE = {
    "c": re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.S),
    "powershell": re.compile(r'"(?:`.|[^"`])*"|\'[^\']*\'|<#.*?#>|#[^\n]*', re.S),
    "default": re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|#[^\n]*', re.S),
}
for _alias in ("c++", "c#", "java", "javascript", "typescript", "rust", "go",
               "swift", "kotlin", "scala", "php", "dart"):
    _NOISE[_alias] = _NOISE["c"]


def _strip_noise(code: str, language: str) -> str:
    """Blank out strings and comments, preserving line numbers."""
    pattern = _NOISE.get(language, _NOISE["default"])

    def blank(match: re.Match) -> str:
        # Keep the newlines so reported line numbers stay true.
        return "\n" * match.group(0).count("\n")

    try:
        return pattern.sub(blank, code)
    except Exception:
        return code


def _mask_noise(code: str, language: str) -> str:
    """Blank strings and comments to spaces, preserving LENGTH and newlines.

    _strip_noise collapses a match to its newlines, which is fine for counting
    but destroys offsets. Anything that has to slice the original text (finding
    where a function starts and ends) needs this instead.
    """
    pattern = _NOISE.get(language, _NOISE["default"])

    def blank(match: re.Match) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    try:
        return pattern.sub(blank, code)
    except Exception:
        return code


def _matching_brace(masked: str, open_at: int) -> int:
    """Index just past the `}` that closes the `{` at open_at, or -1."""
    depth = 0
    for index in range(open_at, len(masked)):
        char = masked[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def definitions(code: str, language: str) -> list[tuple[str, int, int]]:
    """(name, start, end) for each top-level definition this file declares.

    Empty when the language is not one whose definitions can be delimited
    reliably. Callers treat an empty list as "cannot patch, replace the whole
    file", so a language being unsupported costs correctness nothing.
    """
    lang = (language or "").lower()
    header = _DEFINITIONS.get(lang)
    if header is None or not (code or "").strip():
        return []
    masked = _mask_noise(code, lang)
    found: list[tuple[str, int, int]] = []
    try:
        if lang == "python":
            return _python_definitions(code, header)
        for match in header.finditer(masked):
            brace = masked.find("{", match.end() - 1)
            if brace < 0:
                continue
            end = _matching_brace(masked, brace)
            if end < 0:
                continue
            found.append((match.group(1), match.start(), end))
    except Exception as exc:
        log(f"definition scan failed for {lang}: {exc}", logging.DEBUG)
        return []
    return found


def _python_definitions(code: str, header: re.Pattern) -> list[tuple[str, int, int]]:
    """Python definitions, delimited by indentation rather than braces."""
    lines = code.split("\n")
    offset = 0
    offsets: list[int] = []
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
    offsets.append(offset)
    found: list[tuple[str, int, int]] = []
    for index, line in enumerate(lines):
        match = header.match(line)
        if not match:
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for after in range(index + 1, len(lines)):
            probe = lines[after]
            if probe.strip() and (len(probe) - len(probe.lstrip())) <= indent:
                end = after
                break
        found.append((match.group(1), offsets[index], offsets[end]))
    return found


def _balance_findings(code: str, language: str) -> list[Finding]:
    """Unbalanced braces/brackets/parens, with the line the imbalance opened on."""
    if language not in _BRACED:
        return []
    text = _strip_noise(code, language)
    pairs = {"}": "{", ")": "(", "]": "["}
    stack: list[tuple[str, int]] = []
    findings: list[Finding] = []
    for number, line in enumerate(text.split("\n"), 1):
        for char in line:
            if char in "{([":
                stack.append((char, number))
            elif char in pairs:
                if not stack or stack[-1][0] != pairs[char]:
                    findings.append(Finding(
                        "error", f"unbalanced {char!r}: nothing here opens it", number))
                    return findings          # one report; the rest is noise
                stack.pop()
    if stack:
        opener, number = stack[0]
        findings.append(Finding(
            "error",
            f"{opener!r} opened here is never closed "
            f"({len(stack)} unclosed altogether) — the file is incomplete",
            number))
    return findings


# A function definition, per language family. Only the shapes that let the name
# be read reliably; a language whose definitions are not matched here simply
# gets no recursion or shadowing check.
_DEFINITIONS = {
    # [ \t]* rather than \s*: \s includes newlines, so an anchored \s* swallows
    # the blank lines ABOVE the definition and reports a start offset inside the
    # previous function. Harmless for counting, wrong for slicing.
    "powershell": re.compile(r"(?im)^[ \t]*function\s+([A-Za-z][\w-]*)\s*(?:\(|\{)"),
    "bash": re.compile(r"(?m)^[ \t]*(?:function\s+)?([A-Za-z_]\w*)[ \t]*\(\)[ \t]*\{"),
    "python": re.compile(r"(?m)^[ \t]*def\s+([A-Za-z_]\w*)\s*\("),
    "c": re.compile(r"(?m)^[A-Za-z_][\w \t*&:<>,]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{"),
}
_DEFINITIONS["c++"] = _DEFINITIONS["c"]

# Names that already exist in the shell/runtime, where defining a function with
# the same name and then calling it inside itself is unbounded recursion rather
# than a call to the original. This is the Start-Service bug verbatim.
_BUILTINS = {
    "powershell": {
        "start-service", "stop-service", "restart-service", "get-service",
        "set-service", "get-item", "set-item", "new-item", "remove-item",
        "get-content", "set-content", "add-content", "copy-item", "move-item",
        "write-output", "write-error", "write-host", "get-process",
        "stop-process", "start-process", "get-date", "test-path", "invoke-command",
        "get-childitem", "select-object", "where-object", "foreach-object",
    },
    "bash": {"cd", "echo", "test", "read", "set", "export", "kill", "printf",
             "pwd", "exit", "trap", "type", "command", "eval", "source"},
    "c": {"open", "close", "read", "write", "malloc", "free", "exit", "signal",
          "startservice", "controlservice", "openservice", "main"},
}
_BUILTINS["c++"] = _BUILTINS["c"]


def _shadowed_recursion_findings(code: str, language: str) -> list[Finding]:
    """A function that shadows a builtin and then calls that name inside itself.

    `function Start-Service { ... Start-Service -Name $n ... }` reads as a
    wrapper and runs as an infinite loop, because the name now resolves to the
    wrapper. The check is narrow on purpose: the name must be a known builtin
    AND appear in its own body in call position, so ordinary recursion (a
    helper calling itself with a base case) is never reported.
    """
    definition = _DEFINITIONS.get(language)
    builtins = _BUILTINS.get(language)
    if definition is None or not builtins:
        return []
    text = _strip_noise(code, language)
    findings: list[Finding] = []
    matches = list(definition.finditer(text))
    for index, match in enumerate(matches):
        name = match.group(1)
        if name.lower() not in builtins:
            continue
        # The body runs to the next definition, which is a coarse but safe
        # bound: a call to itself anywhere before the next function is inside it.
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        called = re.search(rf"(?<![\w-]){re.escape(name)}\b(?!\s*[({{])"
                           if language == "powershell"
                           else rf"(?<![\w]){re.escape(name)}\s*\(", body)
        if called:
            line = text[:match.start()].count("\n") + 1
            findings.append(Finding(
                "error",
                f"{name} shadows the built-in {name} and calls that name inside "
                f"its own body — this recurses forever, it does not wrap it. "
                f"Rename the function, or call the built-in by its full path.",
                line))
    return findings


def _duplicate_block_findings(code: str) -> list[Finding]:
    """A run of identical lines appearing twice: the stitched-continuation bug.

    A continuation that resumes from the wrong seam re-emits a block it already
    wrote. The result parses (sometimes) and is silently wrong, so length alone
    does not catch it.
    """
    lines = [line.rstrip() for line in (code or "").split("\n")]
    meaningful = [(index, line) for index, line in enumerate(lines)
                  if len(line.strip()) > 3]
    if len(meaningful) < DUPLICATE_RUN * 2:
        return []
    seen: dict[tuple[str, ...], int] = {}
    for position in range(len(meaningful) - DUPLICATE_RUN + 1):
        window = tuple(line for _, line in meaningful[position:position + DUPLICATE_RUN])
        # A run of the SAME line repeated is ordinary code (a block of printfs,
        # a table of asserts), not a spliced duplicate. What this looks for is a
        # distinct block that appears twice, so the window has to carry some
        # variety before it counts as one.
        if len(set(window)) < 3:
            continue
        first = seen.get(window)
        if first is None:
            seen[window] = meaningful[position][0] + 1
            continue
        return [Finding(
            "error",
            f"{DUPLICATE_RUN} or more consecutive lines from line {first} are "
            f"repeated verbatim here — the file looks spliced together rather "
            f"than written through",
            meaningful[position][0] + 1)]
    return []


# Per-language syntax that is wrong often enough, and mechanically, to be worth
# matching literally. Each entry is (pattern, message).
_LITERAL_ERRORS = {
    "powershell": [
        # No space before the `=`: `-Name value` is the only spelling, and
        # `-Name=value` is what a model that has been writing C reaches for.
        (re.compile(r"(?m)(?<=\s)(-[A-Za-z][\w]*)="),
         "PowerShell parameters are passed as `-Name value`, not `-Name=value`"),
        (re.compile(r"(?<![\w-])Set-WmiObject(?![\w-])"),
         "Set-WmiObject is not a cmdlet. Use Set-CimInstance, or the WMI "
         "object's own .Put() method"),
        (re.compile(r"(?m)^\s*function\s+[\w-]+\s*\{\s*$\n\s*\{"),
         "a second `{` immediately follows the function's opening brace — the "
         "parameter block is missing"),
    ],
    "c": [
        (re.compile(r"(?<![\w_])popen\s*\("),
         "popen/pclose are _popen/_pclose in the Microsoft CRT; this will not "
         "link on Windows"),
        (re.compile(r"(?<![\w_])gets\s*\("),
         "gets() has no bounds check and was removed in C11; use fgets()"),
    ],
}
_LITERAL_ERRORS["c++"] = _LITERAL_ERRORS["c"]


def _literal_findings(code: str, language: str) -> list[Finding]:
    rules = _LITERAL_ERRORS.get(language)
    if not rules:
        return []
    text = _strip_noise(code, language)
    findings: list[Finding] = []
    for pattern, message in rules:
        match = pattern.search(text)
        if match:
            findings.append(Finding("error", message,
                                    text[:match.start()].count("\n") + 1))
    return findings


def structural_findings(code: str, language: str) -> list[Finding]:
    """Every tool-free check, for any language. Never raises."""
    body = code or ""
    if not body.strip():
        return []
    lang = (language or "").lower()
    try:
        return (_balance_findings(body, lang)
                + _shadowed_recursion_findings(body, lang)
                + _literal_findings(body, lang)
                + _duplicate_block_findings(body))
    except Exception as exc:                # a check must never break a reply
        log(f"structural code check failed: {exc}", logging.DEBUG)
        return []


# --------------------------------------------------------------------------- #
# Real parsers, when the host has them and shell execution is allowed          #
# --------------------------------------------------------------------------- #

# (executable, argv builder, source extension). Syntax-only in every case: none
# of these runs the program.
_PARSERS: dict[str, tuple[str, list[str], str]] = {
    "bash": ("bash", ["-n"], ".sh"),
    "python": ("python3", ["-m", "py_compile"], ".py"),
    "javascript": ("node", ["--check"], ".js"),
    "php": ("php", ["-l"], ".php"),
    "ruby": ("ruby", ["-c"], ".rb"),
    "c": ("cc", ["-fsyntax-only"], ".c"),
    "c++": ("c++", ["-fsyntax-only"], ".cpp"),
    "go": ("gofmt", ["-e"], ".go"),
}


def parser_for(language: str) -> str:
    """The syntax checker available on this host for a language, or ""."""
    entry = _PARSERS.get((language or "").lower())
    if entry is None:
        return ""
    return shutil.which(entry[0]) or ""


def syntax_findings(code: str, language: str) -> list[Finding]:
    """Parse the code with a real parser. "" findings when none is installed.

    Callers gate this on config.allow_shell: it starts a subprocess, and a host
    that has said no to shell execution has said no to this too. PowerShell is
    checked in-process by pwsh's own parser, which does not execute the script.
    """
    lang = (language or "").lower()
    body = code or ""
    if not body.strip():
        return []
    if lang == "powershell":
        return _powershell_findings(body)
    entry = _PARSERS.get(lang)
    if entry is None or not shutil.which(entry[0]):
        return []
    executable, flags, extension = entry
    try:
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / f"artifact{extension}"
            source.write_text(body, encoding="utf-8")
            done = subprocess.run(                      # noqa: S603
                [executable, *flags, str(source)],
                capture_output=True, text=True, timeout=CHECK_TIMEOUT,
                cwd=work, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if done.returncode == 0:
            return []
        return _parser_output_findings(done.stderr or done.stdout, str(source))
    except Exception as exc:
        log(f"{executable} syntax check did not run: {exc}", logging.DEBUG)
        return []


def _powershell_findings(code: str) -> list[Finding]:
    """PowerShell's own parser, which reports errors without running anything."""
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        return []
    # [Parser]::ParseInput populates a token/error array; nothing is executed.
    script = (
        "$src = [Console]::In.ReadToEnd(); $errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseInput("
        "$src, [ref]$null, [ref]$errors); "
        "foreach ($e in $errors) { "
        "Write-Output ($e.Extent.StartLineNumber.ToString() + ': ' + $e.Message) }"
    )
    try:
        done = subprocess.run(                          # noqa: S603
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
            input=code, capture_output=True, text=True, timeout=CHECK_TIMEOUT)
    except Exception as exc:
        log(f"pwsh syntax check did not run: {exc}", logging.DEBUG)
        return []
    findings: list[Finding] = []
    for line in (done.stdout or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        number, _, message = line.partition(": ")
        findings.append(Finding("error", message.strip() or line,
                                int(number) if number.isdigit() else 0))
        if len(findings) >= 3:
            break
    return findings


_PARSER_LINE = re.compile(r"^(?:.*?):(\d+)(?::\d+)?:\s*(?:fatal\s+)?error:\s*(.*)$",
                          re.I | re.M)


def _parser_output_findings(output: str, source_path: str) -> list[Finding]:
    """The first few real errors out of a compiler's stderr."""
    findings: list[Finding] = []
    for match in _PARSER_LINE.finditer(output or ""):
        findings.append(Finding("error", match.group(2).strip(), int(match.group(1))))
        if len(findings) >= 3:
            break
    if not findings:
        # A parser that does not use the file:line:col: error: convention
        # (py_compile, gofmt) still said something; keep its first real line.
        for line in (output or "").strip().split("\n"):
            text = line.strip().replace(source_path, "the file")
            if text and not text.startswith(("Traceback", "  File", "Sorry:")):
                findings.append(Finding("error", text[:200]))
                break
    return findings


def check_code(code: str, language: str, deep: bool = False) -> list[Finding]:
    """Structural checks, plus a real parse when `deep` and one is available.

    Findings are deduplicated by message: a real parser and the brace counter
    both notice an unclosed function, and saying it twice helps nobody.
    """
    findings = list(structural_findings(code, language))
    if deep:
        findings.extend(syntax_findings(code, language))
    seen: set[str] = set()
    unique: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9),
                                                   f.line)):
        key = finding.message[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique[:4]


def findings_note(findings: list[Finding]) -> str:
    """The block appended under an answer, or "" when the code checked out."""
    if not findings:
        return ""
    lines = ["", "---",
             f"**{len(findings)} problem{'s' if len(findings) > 1 else ''} found "
             f"in this code before showing it to you:**"]
    lines += [f"- {finding.render()}" for finding in findings]
    lines.append("Ask me to fix these and I will work from this list.")
    return "\n".join(lines)


def looks_damaged(code: str, language: str) -> bool:
    """True when the code is structurally broken enough not to be stored.

    Used by the task state to refuse a replacement artifact: a spliced or
    unclosed file must not overwrite a version that was whole.
    """
    return any(f.severity == "error" for f in structural_findings(code, language)
               if "never closed" in f.message or "repeated verbatim" in f.message
               or "nothing here opens it" in f.message)


__all__ = [
    "Finding",
    "definitions",
    "check_code",
    "structural_findings",
    "syntax_findings",
    "findings_note",
    "looks_damaged",
    "parser_for",
    "CHECK_TIMEOUT",
    "DUPLICATE_RUN",
]
