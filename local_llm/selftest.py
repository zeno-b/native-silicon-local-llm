"""The self-test suite run by --selftest.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import importlib

import json
import platform
import re
import sys
import time
from pathlib import Path

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403
from .cluster import *  # noqa: F401,F403
from .claude_import import *  # noqa: F401,F403
from .ui import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .tasks import *  # noqa: F401,F403
from .api import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .calculator import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .diagnostics import *  # noqa: F401,F403


def selftest() -> int:
    """Verify this file is intact and internally consistent. Returns an exit code.

    Runs before bootstrap() and uses only the standard library, so it works on a
    machine with no dependencies installed.
    """
    import hashlib
    import tempfile

    path = Path(__file__).resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    failures: list[str] = []

    # Fingerprint the WHOLE build, not just this module: the point of the hash is
    # "am I running the version I think I am", and after the split one module's
    # digest answers that for 1/26th of the app.
    if __package__:
        _sources = sorted(path.parent.glob("*.py"))
        _blob = b"".join(p.read_bytes() for p in _sources)
        _lines = sum(p.read_text(encoding="utf-8", errors="replace").count("\n")
                     for p in _sources)
        print(f"package  : {path.parent}  ({len(_sources)} modules)")
        print(f"sha256   : {hashlib.sha256(_blob).hexdigest()}  (whole package)")
        print(f"lines    : {_lines}")
    else:
        print(f"file     : {path}")
        print(f"sha256   : {hashlib.sha256(raw).hexdigest()}")
        print(f"lines    : {text.count(chr(10))}")
    print(f"ui build : {UI_BUILD}")
    print(f"python   : {sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    print()

    # A pasted-over-instead-of-replaced file shows up as repeated top-of-file
    # markers. Match whole lines, and build the shebang from parts so this file
    # contains exactly one literal copy of it.
    future_line = "from __future__ import annotations"
    source_lines = text.splitlines()
    futures = sum(1 for line in source_lines if line == future_line)
    if futures != 1:
        failures.append(f"file looks duplicated: {futures} __future__ imports, expected 1")
    # Package integrity: every module imports cleanly and the expected set is
    # present. This replaces the old "one shebang" check, which only made sense
    # when the whole app was a single pasted-over file.
    # Only meaningful when running as a package; the bundled single-file build
    # has no modules to import, and that is fine.
    _pkg_dir = Path(__file__).resolve().parent
    _expected = set() if not __package__ else {
        "core", "obslog", "database", "sysutil", "ui", "config", "model_server",
        "training", "websearch", "calculator", "tools", "llm", "model_client",
        "textutil", "agent", "tasks", "auth", "cluster", "claude_import", "api",
        "diagnostics", "selftest", "cli",
    }
    _found = {p.stem for p in _pkg_dir.glob("*.py")} - {"__init__", "__main__"}
    if _missing := sorted(_expected - _found):
        failures.append(f"package is missing modules: {', '.join(_missing)}")
    for _name in sorted(_expected):
        try:
            importlib.import_module(f"{__package__}.{_name}")
        except Exception as _exc:
            failures.append(f"module {_name} does not import: {type(_exc).__name__}: {_exc}")
    # Importing cleanly is not enough: a module can be missing a dependency edge
    # and only fail when the relevant line runs (help_cmd in training.py did
    # exactly that). Check that every global name each module references is
    # actually resolvable in its namespace.
    if __package__:
        import ast as _ast
        import builtins as _bi
        _builtin = set(dir(_bi))
        for _name in sorted(_expected):
            _mod = importlib.import_module(f"{__package__}.{_name}")
            _tree = _ast.parse((_pkg_dir / f"{_name}.py").read_text(encoding="utf-8"))
            _ns, _bound, _used = set(vars(_mod)), set(), set()
            for _node in _ast.walk(_tree):
                if isinstance(_node, _ast.Name):
                    (_used if isinstance(_node.ctx, _ast.Load) else _bound).add(_node.id)
                elif isinstance(_node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    _bound.add(_node.name)
                    _args = _node.args
                    for _a in _args.args + _args.kwonlyargs + _args.posonlyargs:
                        _bound.add(_a.arg)
                    if _args.vararg:
                        _bound.add(_args.vararg.arg)
                    if _args.kwarg:
                        _bound.add(_args.kwarg.arg)
                elif isinstance(_node, _ast.ClassDef):
                    _bound.add(_node.name)
                elif isinstance(_node, (_ast.Import, _ast.ImportFrom)):
                    for _a in _node.names:
                        _bound.add((_a.asname or _a.name).split(".")[0])
                elif isinstance(_node, _ast.ExceptHandler) and _node.name:
                    _bound.add(_node.name)
            _missing = sorted(n for n in _used - _ns - _builtin - _bound
                              if not n.startswith("__") and len(n) > 1)
            if _missing:
                failures.append(f"module {_name} references unresolved names: "
                                + ", ".join(_missing[:6]))

    # The UI is assembled from parts (styles, markup, five script sections). If a
    # part goes missing or the order changes, the page silently loses behaviour,
    # so assert the seams here rather than discovering it in the browser.
    for _fragment, _why in (
        ("<!doctype html>", "document type"),
        ("<style>", "opening style tag"),
        ("</style>", "closing style tag"),
        ("<script>", "opening script tag"),
        ("</script>", "closing script tag"),
        ("</html>", "closing html tag"),
    ):
        if HTML_PAGE.count(_fragment) != 1:
            failures.append(f"assembled UI has {HTML_PAGE.count(_fragment)} copies of the {_why}")
    if HTML_PAGE.index("<style>") > HTML_PAGE.index("<script>"):
        failures.append("assembled UI puts the script before the styles")
    for _marker, _part in (
        ("function renderMarkdown", "chat script"),
        ("function showView", "views script"),
        ("function loadTasks", "tasks script"),
        ("function loadModels", "models script"),
        ("function loadHistory", "panels script"),
    ):
        if _marker not in HTML_PAGE:
            failures.append(f"assembled UI is missing the {_part}")

    # ROOT must be the directory holding deploy.py, not the package directory.
    # If this slips, the database, logs and adapters silently move one level down
    # and an existing install appears to lose all its data.
    # The invariant is that the project root is not itself a Python package
    # directory. Checking the folder's NAME would misfire when the project is
    # legitimately called local_llm too, which is a normal way to lay this out.
    if (ROOT / "__init__.py").exists():
        failures.append(f"ROOT resolved inside a package ({ROOT}); data would move")
    if DATA_DIR.parent != ROOT or DB_PATH.parent != DATA_DIR:
        failures.append("data paths are not anchored to the project root")

    failures.extend(f"embedded UI: {p}" for p in check_ui_syntax())

    page = render_ui()
    if "{{UI_BUILD}}" in page:
        failures.append("UI_BUILD placeholder left unsubstituted in the rendered page")
    if "<script>" not in page or "</script>" not in page:
        failures.append("rendered page is missing its <script> block")

    # Structural checks that catch a broken layout (an unbalanced <div> leaves a
    # view unclosed and collapses the page) and dangling element references (JS
    # that reads an element the markup never defines). These are exactly the
    # class of bug that JS syntax checking alone cannot see.
    _divs_open = len(re.findall(r"<div\b", page))
    _divs_close = len(re.findall(r"</div>", page))
    if _divs_open != _divs_close:
        failures.append(f"UI <div> tags are unbalanced ({_divs_open} open, {_divs_close} close)")
    # The UI JS lives in Python RAW strings, so a newline escape must be written
    # `\n`. Writing `\\n` reaches JS as an escaped backslash and renders the two
    # characters "\n" to the user instead of a line break -- exactly what made the
    # Training data and Codebase panels show "Total examples: 1\nApproved: 1".
    # Nothing in this UI legitimately emits a literal backslash-n, so any is a bug.
    if "\\\\n" in page:
        _bad = page.count("\\\\n")
        failures.append(f"UI JS has {_bad} double-escaped newline(s) (\\\\n in a raw "
                        "string renders a literal \\n; use a single backslash)")

    _script = re.search(r"<script>(.*?)</script>", page, re.S)
    if _script:
        _body = _script.group(1)
        _defined = set(re.findall(r'\bid="([A-Za-z0-9_]+)"', page))
        _defined |= set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)\s*\.\w+\s*=", _body))
        _refs = set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)", _body))
        _missing = sorted(r for r in _refs if r not in _defined)
        if _missing:
            failures.append("UI references elements with no matching id: " + ", ".join(_missing))

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "selftest.db")
        db.execute(
            "INSERT INTO feedback (user_prompt, assistant_response, approved_for_training) "
            "VALUES ('a', 'b', 1)"
        )
        db.commit()
        if db.get_untrained_count() != 1:
            failures.append("untrained count did not reflect a newly approved row")
        db.mark_trained([1])
        if db.get_untrained_count() != 0:
            failures.append("mark_trained did not clear the untrained count")

        db.add_message("conv1", "user", "hello")
        db.add_message("conv1", "assistant", "hi there")
        history = db.get_messages("conv1")
        if [m["role"] for m in history] != ["user", "assistant"]:
            failures.append("conversation history did not round-trip in order")
        db.log_tool_call("conv1", "calculator", {"expression": "1+1"}, "2", 1.0)
        if not db.list_tool_calls(conversation_id="conv1"):
            failures.append("tool call log did not round-trip")
        if db.clear_conversation("conv1") != 2:
            failures.append("clear_conversation did not remove both messages")

        db.remember("user.city", "Brussels", "conv1")
        db.remember("user.city", "Ghent", "conv1")
        notes = db.recall("city")
        if len(notes) != 1 or notes[0]["value"] != "Ghent":
            failures.append("memory upsert did not replace the earlier value")
        if not db.forget("user.city") or db.recall("city"):
            failures.append("forget did not delete the note")
        db.close()
        # close() must not leave this thread holding a dead handle.
        try:
            db.get_stats()
        except Exception as exc:
            failures.append(f"database did not reconnect after close(): {exc}")
        finally:
            db.close()

    # Tool call parsing has to survive whatever a small model wraps around it.
    parse_cases = [
        ('{"tool": "web_search", "args": {"query": "x"}}', ("web_search", {"query": "x"})),
        ('Sure!\n```json\n{"tool": "calculator", "args": {"expression": "2+2"}}\n```', ("calculator", {"expression": "2+2"})),
        ('{"tool": "current_time", "arguments": {}}', ("current_time", {})),
        ('I think the answer is 4.', None),
        ('The set {a, b} is not JSON.', None),
    ]
    for text, expected in parse_cases:
        if parse_tool_call(text) != expected:
            failures.append(f"parse_tool_call mishandled: {text[:40]!r}")

    # The JS scanner must not flag an apostrophe inside a well-formed literal,
    # because check_ui_syntax failing means main() refuses to start.
    if scan_js_strings('var msg = "it\'s fine"; // don\'t\n'):
        failures.append("scan_js_strings flagged a valid apostrophe")
    if not scan_js_strings('var broken = "starts here\nand ends there";\n'):
        failures.append("scan_js_strings missed a string split across lines")

    if abs(safe_eval("(2+3)*sqrt(16)") - 20.0) > 1e-9:
        failures.append("safe_eval returned the wrong result")
    for bad in ["__import__('os').system('ls')", "open('/etc/passwd').read()", "1 if x else 2"]:
        try:
            safe_eval(bad)
        except Exception:
            continue
        failures.append(f"safe_eval accepted unsafe input: {bad}")

    try:
        resolve_in_workspace("../../etc/passwd")
        failures.append("resolve_in_workspace allowed a path outside the workspace")
    except ValueError:
        pass

    for local in ["http://127.0.0.1:8000/api/config", "http://localhost:8080/v1/models"]:
        try:
            guard_public_url(local)
            failures.append(f"guard_public_url allowed {local}")
        except ValueError:
            pass

    system = {"role": "system", "content": "s" * 400}
    history = [{"role": "user", "content": "h" * 4000} for _ in range(10)]
    user = {"role": "user", "content": "u" * 400}
    trimmed, dropped = trim_to_context(system, history, user, 2048, 512)
    if dropped == 0 or trimmed[0] is not system or trimmed[-1] is not user:
        failures.append("trim_to_context did not drop history while keeping system and user turns")

    config = Config()
    registry = ToolRegistry(config)
    # Deterministic shortcuts: only URL and arithmetic route without a model.
    for text, expected in [
        ("17*23", "calculator"),
        ("(2+3)*4", "calculator"),
        ("https://example.com", "fetch_url"),
        ("look up CVEs online", None),        # now handled by the model router
        ("what is the weather in brussels", None),  # ditto
        ("hi", None),
    ]:
        routed = quick_tool(text)
        if (routed[0] if routed else None) != expected:
            failures.append(f"quick_tool({text!r}) routed to {routed!r}, expected {expected!r}")

    # Code requests answer directly; news/weather/lookups do not count as code.
    for text, want_code in [
        ("write a python script attempting 5 os recon techniques", True),
        ("python script to pull the newest CVEs", True),
        ("fix this function", True),
        ("implement a binary search in rust", True),
        # A Swift coding task must read as code (not a research query), so it is
        # answered by writing code rather than routed to a web search.
        ("write a swift function that uses drive api without api key", True),
        ("write a kotlin coroutine example", True),
        ("write up the latest news on Apple", False),
        ("what are the latest CVEs for openssl?", False),
        ("whats tomorrows weather in brussels", False),
    ]:
        if is_code_request(text) != want_code:
            failures.append(f"is_code_request({text!r}) != {want_code}")
    # Self-contained code answers directly; code needing current info searches first.
    for text, want_lookup in [
        ("write a python script attempting 5 os recon techniques", True),
        ("write a script using the latest OpenAI API", True),
        ("implement a binary search in rust", False),
        ("write a regex for emails", False),
        # "drive api without api key" is a coding task, not a lookup: no search.
        ("write a swift function that uses drive api without api key", False),
    ]:
        got = bool(is_code_request(text) and CODE_NEEDS_LOOKUP.search(text))
        if got != want_lookup:
            failures.append(f"code-needs-lookup({text!r}) != {want_lookup}")

    # Follow-up remarks refer to the previous answer, so knowledge-base retrieval
    # must not fire on them (it matched unrelated transcripts and the model then
    # answered by inventing a conversation that never happened).
    for _m, _want in [
        ("doesnt have a main function", True),
        ("its not reconsider the first message in this convo", True),
        ("I meant the c programm you just wrote does not have a main function", True),
        ("no", True),
        ("i meant the other one", True),
        ("what is the eviction policy?", False),
        ("how does rate limiting and session storage work in our queue?", False),
        ("write a c++ crud basic program", False),
    ]:
        if is_followup_remark(_m) != _want:
            failures.append(f"is_followup_remark({_m!r}) != {_want}")

    # The substantive gate keeps chit-chat off the router.
    for msg, want_substantive in [("hi", False), ("hey there", False), ("thanks!", False),
                                  ("what are the latest CVEs?", True),
                                  ("explain how a hash map works", True)]:
        if is_substantive(msg) != want_substantive:
            failures.append(f"is_substantive({msg!r}) != {want_substantive}")

    # chunk_text splits an oversized prompt into bounded, overlapping parts.
    _big = "para " * 4000  # ~20k chars
    _parts = chunk_text(_big, 6000, 600)
    if not _parts or any(len(pt) > 6000 for pt in _parts):
        failures.append("chunk_text produced an oversized chunk")
    if chunk_text("short", 6000, 600) != ["short"]:
        failures.append("chunk_text split a short string")

    # Identity is prepended to the system prompt and disappears when cleared.
    _idcfg = Config(identity="You are TestBot.")
    if not _idcfg.system_prompt_with_identity.startswith("You are TestBot."):
        failures.append("identity not prepended to system prompt")
    if _idcfg.system_prompt not in _idcfg.system_prompt_with_identity:
        failures.append("identity prefix dropped the base system prompt")
    _noid = Config(identity="")
    if _noid.system_prompt_with_identity != _noid.system_prompt:
        failures.append("empty identity should leave the system prompt unchanged")
    if "identity" not in Config.MUTABLE or "train_min_examples" not in Config.MUTABLE:
        failures.append("identity/train_min_examples not live-editable")
    # A malformed setting value is skipped, not crashed on, and good values still apply.
    _rc = Config()
    _changed = _rc.apply({"max_tokens": "not-a-number", "temperature": 0.42})
    if "max_tokens" in _changed or abs(_rc.temperature - 0.42) > 1e-9:
        failures.append("apply() did not skip a malformed value while keeping valid ones")

    # Project-dir confinement: escapes blocked, .git protected, changes tracked.
    import tempfile as _tf
    _proj = Path(_tf.mkdtemp())
    (_proj / "sub").mkdir()
    _pcfg = Config(project_dir=str(_proj))
    _preg = ToolRegistry(_pcfg, None)
    if _preg._root() != _proj.resolve():
        failures.append("project_dir not used as the file-tool root")
    for _bad in ("../escape.txt", "sub/../../escape", "../../etc/passwd"):
        try:
            _preg._resolve(_bad)
            failures.append(f"project confinement allowed an escape: {_bad}")
        except ValueError:
            pass
    try:
        _preg._resolve(".git/config")
        failures.append("project confinement allowed a .git write")
    except ValueError:
        pass
    _preg._write_file("sub/new.txt", "hi")
    if "sub/new.txt" not in _preg.changed_files:
        failures.append("write_file did not record a changed file")
    # A guarded thread logs and dies quietly instead of crashing silently, and a
    # raising target never propagates.
    _hit = {}
    def _boom():
        raise RuntimeError("intentional")
    _t = guarded_thread(_boom)
    _t.start()
    _t.join(2)
    if _t.is_alive():
        failures.append("guarded_thread did not terminate a failing target")
    def _okfn(v):
        _hit["v"] = v
    _t2 = guarded_thread(_okfn, 7)
    _t2.start()
    _t2.join(2)
    if _hit.get("v") != 7:
        failures.append("guarded_thread did not run a normal target with args")

    # Docker exec backend builds an isolated, no-network container command.
    _dcfg = Config(project_dir=str(_proj), exec_backend="docker", docker_image="python:3.12-slim")
    _dreg = ToolRegistry(_dcfg, None)
    _dout = _dreg._exec(["echo", "hi"])
    # Docker almost certainly is not present in CI; we assert a CLEAR message, not a crash.
    if "docker" not in _dout.lower() and "hi" not in _dout:
        failures.append("docker backend did not return a clear message")

    # auto_iterate_rounds is clamped to a sane band.
    if not (0 <= Config(auto_iterate_rounds=999).auto_iterate_rounds <= 5):
        failures.append("auto_iterate_rounds not clamped")
    # Type-aware read_file: structured/binary types handled, not dumped as garbage.
    import json as _json2
    _ftp = Path(_tf.mkdtemp())
    _ftreg = ToolRegistry(Config(project_dir=str(_ftp)), None)
    (_ftp / "d.csv").write_text("name,age\nA,1\nB,2\n")
    (_ftp / "c.json").write_text(_json2.dumps({"k": 1, "l": [1, 2]}))
    (_ftp / "n.ipynb").write_text(_json2.dumps({"cells": [{"cell_type": "code", "source": ["x=1\n"]}]}))
    (_ftp / "b.bin").write_bytes(bytes([0, 1, 2, 0]) * 50)
    if "columns: name, age" not in _ftreg._read_file("d.csv"):
        failures.append("read_file did not preview CSV columns")
    if '"k": 1' not in _ftreg._read_file("c.json"):
        failures.append("read_file did not pretty-print JSON")
    if "cell 0 [code]" not in _ftreg._read_file("n.ipynb"):
        failures.append("read_file did not extract notebook cells")
    if "binary file" not in _ftreg._read_file("b.bin"):
        failures.append("read_file dumped a binary file instead of describing it")
    if "kind: binary" not in _ftreg._file_info("b.bin"):
        failures.append("file_info did not detect a binary file")
    # TOML preview (stdlib) and structured write validation need no third-party libs.
    (_ftp / "p.toml").write_text('[tool]\nk = "v"\n')
    if "top-level keys: tool" not in _ftreg._read_file("p.toml"):
        failures.append("read_file did not preview TOML keys")
    if "valid JSON" not in _ftreg._write_file("ok.json", '{"a": 1}'):
        failures.append("write_file did not confirm valid JSON")
    if "not valid JSON" not in _ftreg._write_file("bad.json", "{oops}"):
        failures.append("write_file did not warn on invalid JSON")

    # Knowledge base: index, BM25 search, replace-on-reindex, and clear.
    with tempfile.TemporaryDirectory() as _kbtmp:
        _kbdb = Database(Path(_kbtmp) / "kb.db")
        if getattr(_kbdb, "fts_enabled", False):
            _kbdb.index_document("d/a.md", "Retrieval uses BM25 ranking over chunks.", "a")
            _kbdb.index_document("d/b.md", "Deployment targets Apple Silicon with mlx.", "b")
            _h = _kbdb.search_documents("bm25 ranking")
            if not _h or "a.md" not in _h[0]["path"]:
                failures.append("knowledge base did not rank the matching document first")
            if not any("b.md" in x["path"] for x in _kbdb.search_documents("apple silicon")):
                failures.append("knowledge base missed a second document")
            _kbdb.index_document("d/a.md", "short", "a")
            if _kbdb.document_stats()["documents"] != 2:
                failures.append("re-indexing a path should replace, not duplicate")
            if _kbdb.clear_documents() != 2 or _kbdb.document_stats()["documents"] != 0:
                failures.append("clearing the knowledge base did not empty it")
            # RAG injection is a no-op on an empty index and cites paths when not.
            _ragcfg = Config()
            _ragreg = ToolRegistry(_ragcfg, _kbdb)
            _ragag = Agent(_ragcfg, _ragreg, ModelClient(_ragcfg))
            if _ragag.with_retrieved_context("anything") != "anything":
                failures.append("RAG should be a no-op when nothing is indexed")
            _kbdb.index_document("d/c.md", "The cache eviction policy is least-recently-used.", "c")
            _aug = _ragag.with_retrieved_context("what is the eviction policy?")
            if "d/c.md" not in _aug or "eviction" not in _aug:
                failures.append("RAG did not inject the matching passage with its source")

    # Conversation search, export, prompt library, and regenerate.
    with tempfile.TemporaryDirectory() as _convtmp:
        _cdb = Database(Path(_convtmp) / "c.db")
        _cdb.add_message("cx", "user", "how do I tune the KV cache?")
        _cdb.add_message("cx", "assistant", "Lower max_kv_size to save memory.")
        _hits = _cdb.search_conversations("kv cache")
        if not _hits or _hits[0]["conversation_id"] != "cx":
            failures.append("conversation search did not find a matching chat")
        _md = _cdb.export_conversation("cx")
        if "## You" not in _md or "max_kv_size" not in _md:
            failures.append("markdown export missing content or speaker headings")
        if '"role"' not in _cdb.export_conversation("cx", "json"):
            failures.append("json export did not produce structured messages")
        _cdb.save_prompt("review", "Review this code:")
        _cdb.save_prompt("review", "Review this code carefully:")
        if len(_cdb.list_prompts()) != 1:
            failures.append("saving the same prompt name should update, not duplicate")
        if not _cdb.delete_prompt("review") or _cdb.list_prompts():
            failures.append("prompt deletion did not work")
        _again = _cdb.drop_last_exchange("cx")
        if not _again or "KV cache" not in _again:
            failures.append("regenerate did not return the prompt that produced the answer")
        if [r["role"] for r in _cdb.get_messages("cx")] != ["user"]:
            failures.append("regenerate did not remove the previous answer")

    # Conversation list carries a readable title for the history panel.
    with tempfile.TemporaryDirectory() as _htmp:
        _hdb = Database(Path(_htmp) / "h.db")
        _hdb.add_message("h1", "user", "How do I tune the KV cache?")
        _hdb.add_message("h1", "assistant", "Lower max_kv_size.")
        _convs = _hdb.list_conversations()
        if not _convs or "title" not in _convs[0]:
            failures.append("conversation list is missing a title for the history panel")
        elif "KV cache" not in _convs[0]["title"]:
            failures.append("conversation title should come from the first user message")
        if _convs and _convs[0].get("messages") != 2:
            failures.append("conversation list lost its message count")

    # UI: the new panels and their handlers must exist and resolve.
    _page = render_ui()
    for _needed in ("historyView", "promptsPanel", "dropZone", "palette", "paletteInput",
                    "historyList", "promptList", "fileInput"):
        if f'id="{_needed}"' not in _page:
            failures.append(f"UI is missing the {_needed} element")
    _scriptm = re.search(r"<script>(.*?)</script>", _page, re.S)
    if _scriptm:
        for _fn in ("loadHistory", "runHistorySearch", "openConversation", "togglePrompts",
                    "loadPrompts", "savePrompt", "uploadFiles", "wireDropZone",
                    "openPalette", "closePalette", "addMessageActions"):
            if not re.search(rf"function\s+{_fn}\s*\(", _scriptm.group(1)):
                failures.append(f"UI handler {_fn} is not defined")

    # Conversation titles, pinning, and scoped retrieval.
    with tempfile.TemporaryDirectory() as _mtmp:
        _mdb = Database(Path(_mtmp) / "m.db")
        _mdb.add_message("m1", "user", "first chat")
        _mdb.add_message("m2", "user", "second chat")
        _mdb.set_conversation_title("m1", "Renamed thread")
        _mdb.set_conversation_pinned("m1", True)
        _ml = _mdb.list_conversations()
        if not _ml or _ml[0]["conversation_id"] != "m1":
            failures.append("pinned conversations should sort to the top")
        if _ml and _ml[0]["title"] != "Renamed thread":
            failures.append("custom conversation title was not used")
        if _mdb.conversation_title("m2") is not None:
            failures.append("untitled conversation should report no custom title")
        if getattr(_mdb, "fts_enabled", False):
            _mdb.index_document("x.md", "BM25 ranking scores retrieval results.", "x")
            _mdb.index_document("y.md", "BM25 ranking appears here too.", "y")
            if len({h["path"] for h in _mdb.search_documents("bm25 ranking")}) != 2:
                failures.append("unscoped search should span the knowledge base")
            _scoped = _mdb.search_documents("bm25 ranking", only=["y.md"])
            if {h["path"] for h in _scoped} != {"y.md"}:
                failures.append("scoped search should return only the chosen document")

    # UI: markdown rendering helpers must exist alongside the panels.
    _page2 = render_ui()
    _sm = re.search(r"<script>(.*?)</script>", _page2, re.S)
    if _sm:
        for _fn in ("renderMarkdown", "renderInline", "makeCodeBlock", "renderScope",
                    "applyScope", "clearScope"):
            if not re.search(rf"function\s+{_fn}\s*\(", _sm.group(1)):
                failures.append(f"UI helper {_fn} is not defined")
        # Model output must never be injected as raw HTML.
        if "innerHTML = text" in _sm.group(1) or "innerHTML=text" in _sm.group(1):
            failures.append("model output is being assigned as innerHTML (injection risk)")
    for _needed in ("docScope",):
        if f'id="{_needed}"' not in _page2:
            failures.append(f"UI is missing the {_needed} element")

    # Every element that ships with the `hidden` state class must be matched by
    # a rule that actually hides IT. Checking only that `.hidden` appears
    # somewhere in the CSS is not enough: the sheet had `#palette.hidden`,
    # `.view.hidden` and friends but no rule covering `.acct-pop.hidden`, so the
    # account popup sat permanently open over the header with no error anywhere.
    _css = re.search(r"<style>(.*?)</style>", _page2, re.S)
    if _css:
        _sheet = re.sub(r"/\*.*?\*/", " ", _css.group(1), flags=re.S)   # comments first
        _rules = re.findall(r"([^{}]+)\{[^{}]*display\s*:\s*none", _sheet)
        # The captured "selector" runs back to the previous rule's closing brace,
        # so keep only the last line of it, then split a selector list on commas.
        _selectors = {sel.strip()
                      for rule in _rules
                      for sel in rule.strip().split("\n")[-1].split(",")}
        for _tag in re.finditer(r"<(\w+)([^>]*\bclass=\"[^\"]*\bhidden\b[^\"]*\"[^>]*)>",
                                _page2):
            _attrs = _tag.group(2)
            _classes = set((re.search(r'class="([^"]*)"', _attrs).group(1)).split())
            _id_match = re.search(r'id="([\w-]+)"', _attrs)
            _eid = _id_match.group(1) if _id_match else ""
            _ok = {".hidden", f"#{_eid}.hidden" if _eid else None}
            _ok |= {f".{c}.hidden" for c in _classes if c != "hidden"}
            _ok |= {f".hidden.{c}" for c in _classes if c != "hidden"}
            if not (_ok & _selectors):
                failures.append(
                    f"<{_tag.group(1)} id={_eid or '?'}> carries class 'hidden' but no "
                    "CSS rule hides it")

    # Backup round-trips and forking preserves the original conversation.
    with tempfile.TemporaryDirectory() as _btmp:
        _bdb = Database(Path(_btmp) / "b1.db")
        _bdb.add_message("k1", "user", "first question")
        _bdb.add_message("k1", "assistant", "first answer")
        _bdb.add_message("k1", "user", "second question")
        _bdb.set_conversation_title("k1", "Kept name")
        _bdb.save_prompt("p1", "body")
        _payload = _bdb.export_backup()
        if not _payload.get("messages") or "version" not in _payload:
            failures.append("backup export produced no messages or no version")
        _bdb2 = Database(Path(_btmp) / "b2.db")
        _restored = _bdb2.import_backup(_payload)
        if _restored["messages"] != 3 or _restored["prompts"] != 1:
            failures.append("backup restore did not bring back messages and prompts")
        if not _bdb2.list_conversations() or _bdb2.list_conversations()[0]["title"] != "Kept name":
            failures.append("backup restore lost the conversation title")
        _msgs = _bdb.get_messages("k1")
        _fork = _bdb.fork_conversation("k1", _msgs[1]["id"])
        if len(_bdb.get_messages(_fork)) != 2:
            failures.append("fork should copy only messages up to the chosen point")
        if len(_bdb.get_messages("k1")) != 3:
            failures.append("fork must not modify the original conversation")

    # UI: highlighting, streaming render, theme and backup helpers exist.
    _page3 = render_ui()
    _sm3 = re.search(r"<script>(.*?)</script>", _page3, re.S)
    if _sm3:
        for _fn in ("highlightInto", "scheduleMarkdown", "applyTheme", "toggleTheme",
                    "downloadBackup", "restoreBackup"):
            if not re.search(rf"function\s+{_fn}\s*\(", _sm3.group(1)):
                failures.append(f"UI helper {_fn} is not defined")
    if "html.light" not in _page3:
        failures.append("light theme styles are missing")
    if 'id="restoreInput"' not in _page3 or 'id="themeBtn"' not in _page3:
        failures.append("backup/theme controls are missing from the UI")

    # Error handling: malformed input is rejected clearly instead of crashing.
    with tempfile.TemporaryDirectory() as _etmp:
        _edb = Database(Path(_etmp) / "e.db")
        for _bad, _why in ((None, "None"), ([1, 2], "a list"), ("text", "a string")):
            try:
                _edb.import_backup(_bad)
                failures.append(f"import_backup accepted {_why} instead of rejecting it")
            except ValueError:
                pass
            except Exception as _exc:
                failures.append(f"import_backup({_why}) raised {type(_exc).__name__}, not ValueError")
        try:
            _edb.import_backup({"version": 99})
            failures.append("import_backup accepted a newer backup version")
        except ValueError:
            pass
        # A section of the wrong shape is skipped, not fatal.
        _edb.import_backup({"messages": "not-a-list"})
        for _call, _label in (
            (lambda: _edb.fork_conversation("ghost"), "fork of an empty conversation"),
            (lambda: _edb.export_conversation("ghost"), "export of an empty conversation"),
            (lambda: _edb.save_prompt("", "body"), "prompt with no name"),
            (lambda: _edb.save_prompt("name", "  "), "prompt with no body"),
        ):
            try:
                _call()
                failures.append(f"{_label} should have been refused")
            except ValueError:
                pass
        # LIKE wildcards are matched literally, not as "match everything".
        _edb.add_message("w1", "user", "100% sure")
        _edb.add_message("w2", "user", "nothing special")
        if [r["conversation_id"] for r in _edb.search_conversations("%")] != ["w1"]:
            failures.append("a '%' search should match literally, not every conversation")

        _eproj = Path(_etmp) / "proj"
        (_eproj / "sub").mkdir(parents=True)
        _ereg = ToolRegistry(Config(project_dir=str(_eproj)), _edb)
        for _call, _label in (
            (lambda: _ereg._read_file("sub"), "reading a directory"),
            (lambda: _ereg._write_file("sub", "x"), "writing onto a directory"),
        ):
            try:
                _call()
                failures.append(f"{_label} should have been refused")
            except ValueError:
                pass
        (_eproj / "empty.csv").write_text("")
        if "empty" not in _ereg._read_file("empty.csv"):
            failures.append("an empty CSV should say so rather than show a bogus preview")
        (_eproj / "bad.json").write_text("{nope")
        if "invalid JSON" not in _ereg._read_file("bad.json"):
            failures.append("invalid JSON should be reported with its position")
        if "not a web address" not in _ereg._index_url("not-a-url"):
            failures.append("index_url should explain that a bare word is not a URL")
        if "disappeared" not in _ereg.syntax_check(["gone.py"]):
            failures.append("syntax_check should report an unreadable file, not skip it")
    # Safeguards clamp the retrieval settings too.
    if Config(rag_passages=-3).rag_passages < 1 or Config(rag_passages=999).rag_passages > 20:
        failures.append("rag_passages is not clamped to a usable range")
    if Config(rag_scope=" , ,a.md , ").rag_scope != "a.md":
        failures.append("rag_scope should drop blank entries")

    # Codebase work must be routable: with a project attached the router has to be
    # able to choose a file tool, or "fix this code" is answered from weights and
    # the user's files are never touched.
    with tempfile.TemporaryDirectory() as _rtmp:
        _rproj = Path(_rtmp) / "proj"
        _rproj.mkdir()
        _withproj = ToolRegistry(Config(project_dir=str(_rproj)), None)
        _routable = {t.name for t in _withproj.routable()}
        for _needed in ("list_files", "read_file", "search_files"):
            if _needed not in _routable:
                failures.append(f"{_needed} must be routable when a project is set")
        _noproj = ToolRegistry(Config(project_dir=""), None)
        if "list_files" in {t.name for t in _noproj.routable()}:
            failures.append("file tools should not be routable without a project")
    if not is_code_request("fix this code"):
        failures.append("'fix this code' must be recognised as a code request")

    # The knowledge base must work even where SQLite has no FTS5 (several macOS
    # Python builds). Exercise the fallback ranking directly.
    with tempfile.TemporaryDirectory() as _fbtmp:
        _fbdb = Database(Path(_fbtmp) / "fb.db")
        _fbdb.search_mode = "fallback"
        _fbdb.execute("""CREATE TABLE IF NOT EXISTS doc_chunks_plain (
                             id INTEGER PRIMARY KEY, path TEXT, chunk TEXT)""")
        _fbdb.index_document("a.md", "BM25 ranking scores retrieval results. " * 4, "a")
        _fbdb.index_document("b.md", "Apple Silicon runs the model through mlx. " * 4, "b")
        _fbhits = _fbdb.search_documents("ranking retrieval")
        if not _fbhits or _fbhits[0]["path"] != "a.md":
            failures.append("fallback search did not rank the matching document first")
        if not any(h["path"] == "b.md" for h in _fbdb.search_documents("apple silicon")):
            failures.append("fallback search missed a second document")
        if {h["path"] for h in _fbdb.search_documents("ranking", only=["b.md"])} - {"b.md"}:
            failures.append("fallback search ignored the document scope")
        if _fbdb.document_stats().get("search_mode") != "fallback":
            failures.append("document stats did not report the fallback search mode")

    # In-process syntax check catches broken Python without executing it.
    _syproj = Path(_tf.mkdtemp())
    _syreg = ToolRegistry(Config(project_dir=str(_syproj)), None)
    (_syproj / "good.py").write_text("def f():\n    return 1\n")
    (_syproj / "bad.py").write_text("def f(:\n    pass\n")
    if _syreg.syntax_check(["good.py"]):
        failures.append("syntax_check flagged valid Python")
    if "bad.py" not in _syreg.syntax_check(["bad.py"]):
        failures.append("syntax_check missed a syntax error")
    # Reasoning is visible by default and injected into the prompt.
    if not Config().reasoning_visible:
        failures.append("reasoning_visible should default on")
    _rp = build_agent_system_prompt("base", ToolRegistry(Config(), None), reasoning=True)
    if "<think>" not in _rp:
        failures.append("reasoning instruction not added to the prompt")

    # Confined runner executes in the project root and reports an exit code.
    _pcfg.allow_python = True
    _preg2 = ToolRegistry(_pcfg, None)
    _pcfg.tool_timeout = 15
    _out = _preg2._run_python("print(6*7)")
    if "42" not in _out or "[exit code 0]" not in _out:
        failures.append("confined run_python did not execute in the project")
    # Test-command auto-detection picks pytest for a python project.
    (_proj / "pyproject.toml").write_text("[tool]\n")
    if "pytest" not in _preg2._detect_test_command():
        failures.append("run_tests did not auto-detect pytest for a python project")

    # Reusable dataset export: formats behave and stay model-portable.
    _rows = [
        {"user_prompt": "q1", "assistant_response": "a-good", "corrected_response": None,
         "rating": 1, "approved_for_training": 1, "source": "button"},
        {"user_prompt": "q2", "assistant_response": "a-bad", "corrected_response": "a-fixed",
         "rating": -1, "approved_for_training": 0, "source": "button"},
    ]
    _chat, _n = build_reusable_dataset(_rows, "chat", "SYS")
    if '"role": "system"' not in _chat or "SYS" not in _chat:
        failures.append("chat export missing system prompt")
    _bare, _ = build_reusable_dataset(_rows, "bare", "SYS")
    if "SYS" in _bare or '"role": "user"' not in _bare:
        failures.append("bare export should be model-neutral (no system prompt)")
    _pref, _pn = build_reusable_dataset(_rows, "preference", "")
    if '"chosen": "a-fixed"' not in _pref or '"rejected": "a-bad"' not in _pref:
        failures.append("preference export did not build a correction pair")
    _raw, _rn = build_reusable_dataset(_rows, "raw", "")
    if _rn != 2:
        failures.append("raw export should be lossless (one line per row)")
    if "a-bad" in _chat:  # a rejected answer must never be a target to imitate
        failures.append("chat export leaked a rejected answer as a target")

    # Implicit chat feedback: praise approves, rejection marks bad, prose is None.
    for _m in ("good job", "perfect, thanks!", "that's exactly right", "nice one", "thanks"):
        if classify_implicit_feedback(_m) != 1:
            failures.append(f"implicit feedback should be positive: {_m}")
    for _m in ("no", "nope", "wrong", "that's wrong", "you're wrong"):
        if classify_implicit_feedback(_m) != -1:
            failures.append(f"implicit feedback should be negative: {_m}")
    for _m in ("no way to do this without a loop?", "no idea, can you explain?",
               "explain how recursion works", "right now show me the code"):
        if classify_implicit_feedback(_m) is not None:
            failures.append(f"implicit feedback false positive: {_m}")

    # Time-sensitivity gate: general knowledge stays local, fresh-fact questions
    # are allowed a lookup.
    for _q in ("best way to store cinnamon", "explain how recursion works",
               "write a python script to sort a list"):
        if is_time_sensitive(_q):
            failures.append(f"is_time_sensitive wrongly flagged general knowledge: {_q}")
    for _q in ("bitcoin price today", "who is the current ceo of openai",
               "latest version of python", "today's news about AI"):
        if not is_time_sensitive(_q):
            failures.append(f"is_time_sensitive missed a time-sensitive query: {_q}")

    # ThinkSplitter separates <think> reasoning from the answer, even when a tag
    # is split across streamed tokens.
    _sp = ThinkSplitter()
    _pieces = []
    for _tok in ["<thi", "nk>weigh", "ing it</thi", "nk>Answer."]:
        _pieces += _sp.feed(_tok)
    _think = "".join(c for k, c in _pieces if k == "think")
    _ans = "".join(c for k, c in _pieces if k == "answer")
    if _think != "weighing it" or _ans != "Answer.":
        failures.append(f"ThinkSplitter mis-split across tokens: {_think!r} / {_ans!r}")
    _sp2 = ThinkSplitter()
    if any(k == "think" for k, _ in _sp2.feed("a plain answer")):
        failures.append("ThinkSplitter invented a think block")

    # Retrieval guards: skip aggregator/listing/shell URLs and thin pages.
    for u in ("https://news.google.com/topics/abc", "https://techcrunch.com/category/ai/",
              "https://www.exploit-db.com/", "https://reddit.com/r/x"):
        if not is_low_value_url(u):
            failures.append(f"is_low_value_url missed {u}")
    for u in ("https://en.wikipedia.org/wiki/Transformer",
              "https://arstechnica.com/ai/2026/08/real-article/"):
        if is_low_value_url(u):
            failures.append(f"is_low_value_url wrongly flagged {u}")
    if not is_thin_page("\n".join(["Home", "News", "Login"] * 30)):
        failures.append("is_thin_page missed a navigation shell")
    if is_thin_page("A substantial article sentence with real prose content here. " * 8):
        failures.append("is_thin_page wrongly flagged an article")

    # The retrieval pipeline extracts the top result URLs from a search block.
    _sample = "1. A\n   https://a.com/x\n   s\n2. B\n   https://b.com/y\n   s"
    if top_result_urls(_sample, 1) != ["https://a.com/x"]:
        failures.append("top_result_urls did not return the first result URL")
    if top_result_urls(_sample, 5) != ["https://a.com/x", "https://b.com/y"]:
        failures.append("top_result_urls did not return URLs in order")

    # The router's JSON extractor must survive prose and code fences around the
    # object, and reject replies with no object.
    for raw, want in [
        ('{"action":"answer"}', {"action": "answer"}),
        ('Sure!\n```json\n{"action":"search","query":"x"}\n```', {"action": "search", "query": "x"}),
        ('here: {"action":"weather","location":"Paris","when":"tomorrow"} ok',
         {"action": "weather", "location": "Paris", "when": "tomorrow"}),
        ("no json here", None),
    ]:
        if extract_json_object(raw) != want:
            failures.append(f"extract_json_object({raw!r}) != {want!r}")

    for expr in ["9**9**9", "factorial(9**7)", "(10**20000)*(10**20000)"]:
        try:
            safe_eval(expr)
            failures.append(f"safe_eval({expr!r}) was not refused: unbounded intermediate")
        except ValueError:
            pass

    for expected_tool in ["web_search", "fetch_url", "calculator", "read_file",
                          "write_file", "edit_file", "search_files", "remember",
                          "recall_memory", "final_answer"]:
        if registry.get(expected_tool) is None:
            failures.append(f"tool registry is missing {expected_tool}")
    if registry.get("run_shell") is not None:
        failures.append("run_shell is exposed without --allow-shell")
    result, error = registry.call("calculator", {"expression": "6*7"})
    if error or result.strip() != "42":
        failures.append(f"calculator tool returned {result!r}")
    _, error = registry.call("nonexistent_tool", {})
    if not error:
        failures.append("unknown tool did not report an error")
    prompt = build_agent_system_prompt("base", registry)
    if "web_search" not in prompt or "TOOL RESULT" not in prompt:
        failures.append("agent system prompt is missing the tool protocol")

    # Argument aliasing: a model that says {"q": ...} should still get a search.
    search_tool = registry.get("web_search")
    if registry.normalise_args(search_tool, {"q": "mlx lora"}) != {"query": "mlx lora"}:
        failures.append("normalise_args did not map q onto query")
    if registry.normalise_args(search_tool, {"nonsense": "x"}) != {"query": "x"}:
        failures.append("normalise_args did not adopt a lone value for a single-argument tool")

    # An allowlist must never remove the loop's exit tool.
    limited = ToolRegistry(Config(agent_tools="calculator"))
    if set(limited.names()) != {"calculator", "final_answer"}:
        failures.append(f"AGENT_TOOLS allowlist produced {limited.names()}")

    known = set(registry.names())
    if parse_tool_call('Here is the config: {"name": "gpt", "temperature": 0.7}', known) is not None:
        failures.append("parse_tool_call treated a JSON answer as a tool call")
    if parse_tool_call('{"tool": "web_search", "args": {"query": "x"}}', known) is None:
        failures.append("parse_tool_call rejected a valid call against the known set")

    # Workspace file tools, against a throwaway workspace.
    original_workspace = WORKSPACE_DIR
    # Patch the constant in the module that actually reads it. Rebinding it only
    # here would leave the tools writing into the real ./workspace, which is both
    # a dirty test and a way to clobber a user's files.
    _ws_home = sys.modules[ToolRegistry.__module__]
    with tempfile.TemporaryDirectory() as tmp:
        globals()["WORKSPACE_DIR"] = Path(tmp) / "workspace"
        _ws_home.WORKSPACE_DIR = globals()["WORKSPACE_DIR"]
        try:
            registry.call("write_file", {"path": "notes/a.txt", "content": "alpha\nbeta\n"})
            body, error = registry.call("read_file", {"path": "notes/a.txt"})
            if error or "alpha" not in body:
                failures.append(f"write_file/read_file round-trip failed: {body!r}")
            _, error = registry.call(
                "edit_file", {"path": "notes/a.txt", "find": "beta", "replace": "gamma"}
            )
            body, _ = registry.call("read_file", {"path": "notes/a.txt"})
            if error or "gamma" not in body or "beta" in body:
                failures.append("edit_file did not apply the replacement")
            _, error = registry.call(
                "edit_file", {"path": "notes/a.txt", "find": "absent", "replace": "x"}
            )
            if not error:
                failures.append("edit_file accepted a snippet that is not in the file")
            hits, error = registry.call("search_files", {"pattern": "gam+a"})
            if error or "a.txt" not in hits:
                failures.append(f"search_files did not find the match: {hits!r}")
        finally:
            globals()["WORKSPACE_DIR"] = original_workspace
            _ws_home.WORKSPACE_DIR = original_workspace

    # Tasks: creation, scheduling, the run lifecycle, and event replay.
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "tasks.db")
        task = db.create_task(name="nightly", goal="check the news", interval_seconds=3600)
        if not task["id"] or task["next_run_at"] is None:
            failures.append("a scheduled task was not armed on creation")
        if [t["id"] for t in db.due_tasks()] != [task["id"]]:
            failures.append("a task armed for now did not come back as due")

        db.schedule_next(task["id"], 3600)
        if db.due_tasks():
            failures.append("schedule_next did not push the next run into the future")

        if db.update_task(task["id"], {"enabled": False})["next_run_at"] is not None:
            failures.append("disabling a task left it armed")
        manual = db.create_task(name="manual", goal="do a thing", interval_seconds=0)
        if manual["next_run_at"] is not None:
            failures.append("a manual task was armed anyway")

        run_id = db.create_run(task["id"], "manual", "test-model")
        db.append_event(run_id, 1, "step", {"step": 1})
        db.append_event(run_id, 2, "final", {"answer": "done"})
        events = db.run_events(run_id)
        if [e["type"] for e in events] != ["step", "final"]:
            failures.append("task events did not round-trip in order")
        if db.run_events(run_id, after_seq=1)[0]["type"] != "final":
            failures.append("after_seq did not skip replayed events")
        db.finish_run(run_id, "ok", "done", None, 2, 12.0, ["calculator"])
        if db.get_run(run_id)["status"] != "ok":
            failures.append("finish_run did not record the outcome")
        if db.get_task(task["id"])["last_status"] != "ok":
            failures.append("finish_run did not update the task summary")

        stuck = db.create_run(task["id"], "manual", "test-model")
        if db.reset_orphan_runs() != 1 or db.get_run(stuck)["status"] != "interrupted":
            failures.append("reset_orphan_runs left a run marked running")

        if not db.delete_task(task["id"]) or db.get_run(run_id) is not None:
            failures.append("deleting a task left its runs behind")
        db.close()

    # Model and adapter selection.
    if resolve_adapter("none") is not None or resolve_adapter("") is not None:
        failures.append("resolve_adapter returned a path for the base model")
    try:
        resolve_adapter("../../etc")
        failures.append("resolve_adapter allowed an id outside the backups directory")
    except ValueError:
        pass
    # An adapter must never follow a model swap onto a base it does not fit.
    with tempfile.TemporaryDirectory() as tmp:
        original_dirs = (ADAPTER_DIR, ADAPTER_BACKUP_DIR)
        # These constants live in their defining module; patch there (and here)
        # so both this suite and the code under test see the temp directories.
        _adapter_home = sys.modules[ModelServerManager.__module__]
        _const_home = sys.modules[resolve_adapter.__module__]
        globals()["ADAPTER_DIR"] = Path(tmp) / "adapters" / "latest"
        globals()["ADAPTER_BACKUP_DIR"] = Path(tmp) / "adapters" / "backups"
        for _mod in {_adapter_home, _const_home}:
            _mod.ADAPTER_DIR = globals()["ADAPTER_DIR"]
            _mod.ADAPTER_BACKUP_DIR = globals()["ADAPTER_BACKUP_DIR"]
        try:
            ADAPTER_DIR.mkdir(parents=True)
            (ADAPTER_DIR / "adapters.safetensors").write_bytes(b"stub")
            manager = ModelServerManager("org/base-a", 0, ADAPTER_DIR)
            manager.adapter_choice = "latest"
            if manager.adapter_path() != ADAPTER_DIR:
                failures.append("an untagged adapter was refused")

            write_adapter_base(ADAPTER_DIR, "org/base-a")
            if manager.adapter_path() != ADAPTER_DIR:
                failures.append("a matching adapter was refused")

            manager.swap("org/base-b")
            if manager.adapter_path() is not None:
                failures.append("a mismatched adapter survived a model swap")
            if not manager.describe()["adapter_mismatch"]:
                failures.append("describe() did not report the adapter mismatch")
            if any(flag in manager._build_cmd(True) for flag in ("--adapter-path", "--adapter")):
                failures.append("the server command still passed a mismatched adapter")
        finally:
            globals()["ADAPTER_DIR"], globals()["ADAPTER_BACKUP_DIR"] = original_dirs
            for _mod in {_adapter_home, _const_home}:
                _mod.ADAPTER_DIR, _mod.ADAPTER_BACKUP_DIR = original_dirs

    # A reasoning model's chain of thought must not be executed as a tool call.
    thinking = ('<think>Maybe {"tool": "web_search", "args": {"query": "no"}} fits.</think>'
                '{"tool": "calculator", "args": {"expression": "2+2"}}')
    parsed = parse_tool_call(thinking, {"calculator", "web_search"})
    if parsed != ("calculator", {"expression": "2+2"}):
        failures.append(f"a <think> block hijacked the tool call: {parsed}")
    if parse_tool_call('<think>considering {"tool": "web_search"', {"web_search"}) is not None:
        failures.append("an unterminated <think> block produced a tool call mid-stream")
    if strip_reasoning("<reasoning>hidden</reasoning>visible") != "visible":
        failures.append("strip_reasoning left the reasoning block in place")
    if strip_reasoning("no tags here") != "no tags here":
        failures.append("strip_reasoning mangled ordinary text")

    # Deterministic routing, including the cases that must NOT route.
    for text, expected in [
        ("17*23", ("calculator", {"expression": "17*23"})),
        ("https://example.com/a", ("fetch_url", {"url": "https://example.com/a"})),
        ("what is the capital of France?", None),
        ("2024", None),
    ]:
        if fast_path_call(text) != expected:
            failures.append(f"fast_path_call({text!r}) returned {fast_path_call(text)!r}")

    # Explicit web-search commands route deterministically (no model-router call
    # that could second-guess them); ordinary questions do not.
    for text, expected in [
        ("search the web for mlx 0.9 release notes", "mlx 0.9 release notes"),
        ("google apple silicon npu benchmarks", "apple silicon npu benchmarks"),
        ("web search latest cve for openssl", "latest cve for openssl"),
        ("find the current bitcoin price online", "the current bitcoin price"),
        ("what is a monad?", None),          # a plain question stays on the router
        ("search my codebase for the login handler", None),  # not a web command
    ]:
        if web_search_request(text) != expected:
            failures.append(f"web_search_request({text!r}) returned {web_search_request(text)!r}")

    # Power-user /overrides force a lane and strip the command from the message.
    for text, expected in [
        ("/search who won the 2025 tour de france", ("web_search", "who won the 2025 tour de france")),
        ("/no-search explain tail latency", ("answer", "explain tail latency")),
        ("/kb what does our runbook say about failover", ("kb", "what does our runbook say about failover")),
        ("no leading slash here", None),
        ("/search", ("web_search", "")),     # bare command: caller ignores empty rest
    ]:
        if routing_override(text) != expected:
            failures.append(f"routing_override({text!r}) returned {routing_override(text)!r}")

    # Web-search rerank: the on-topic result is fetched before the off-topic one
    # that the engine happened to rank first, and aggregator hosts are demoted.
    _sr = (
        "1. Random blog\n   https://example.com/misc\n   an unrelated post about gardening\n"
        "2. MLX 0.9 release notes\n   https://ml-explore.github.io/mlx/build/html/notes.html\n"
        "   the mlx 0.9 release adds new quantization kernels\n"
        "3. Reddit thread\n   https://reddit.com/r/foo/comments/x\n   people discussing mlx"
    )
    _ranked = rank_result_urls(_sr, "mlx 0.9 release quantization", 3)
    if not _ranked or "ml-explore.github.io" not in _ranked[0]:
        failures.append(f"rank_result_urls did not promote the on-topic result: {_ranked}")
    if _ranked and "reddit.com" in _ranked[0]:
        failures.append("rank_result_urls failed to demote an aggregator host")
    # With no query tokens it must fall back to engine order (never drop results).
    if rank_result_urls(_sr, "", 3)[:1] != top_result_urls(_sr, 1):
        failures.append("rank_result_urls did not fall back to engine order without a query")

    # Prefix stability: step k+1's prompt must start with step k's, so an
    # extending prompt keeps matching a server-side cache.
    def trace_prompts(stable: bool) -> list[list[str]]:
        cfg = Config(context_size=1200, stable_prefix=stable)
        probe = Agent(cfg, ToolRegistry(cfg), ModelClient(cfg))
        base_msgs = [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "the question"}]
        scratch: list[dict] = []
        out = []
        for index in range(12):
            scratch.append({"role": "assistant", "content": f"call {index}"})
            scratch.append({"role": "user", "content": "TOOL RESULT:\n" + "z" * 700})
            messages, _ = probe.assemble(base_msgs, scratch, 128)
            if messages[:2] != base_msgs:
                failures.append("assemble evicted the pinned system prompt or question")
            out.append([m["content"] for m in messages])
        return out

    def extend_hits(prompts: list[list[str]]) -> int:
        return sum(1 for i in range(len(prompts) - 1)
                   if prompts[i + 1][:len(prompts[i])] == prompts[i])

    stable_hits = extend_hits(trace_prompts(True))
    drop_hits = extend_hits(trace_prompts(False))
    if stable_hits <= drop_hits:
        failures.append(f"stable_prefix did not improve prefix reuse ({stable_hits} vs {drop_hits})")

    # The raw tool cap must stay above the context cap, or the summariser never
    # sees enough text to summarise and silently becomes dead code.
    sized = Config()
    sized.apply({"tool_result_chars": 4000, "tool_raw_chars": 500})
    if sized.tool_raw_chars < sized.tool_result_chars:
        failures.append("tool_raw_chars was allowed below tool_result_chars")

    # Generation accounting.
    sample = GenerationStats(prompt_tokens=900, completion_tokens=30, ttft_ms=500, total_ms=2500)
    if abs(sample.decode_tps - 15.0) > 0.1:
        failures.append(f"decode_tps computed {sample.decode_tps}, expected 15")
    if sample.as_event()["estimated"] is not True:
        failures.append("stats without server usage were not marked estimated")

    catalog = model_catalog(Config(model="someone/custom-model"))
    if not any(entry["current"] and entry["id"] == "someone/custom-model" for entry in catalog):
        failures.append("model_catalog did not include the model currently in use")
    if len(catalog) != len({entry["id"] for entry in catalog}):
        failures.append("model_catalog returned duplicates")

    # The live event buffer must number events and survive a token flood.
    live = TaskRun("run1", "task1", "demo")
    live.publish({"type": "start"})
    for index in range(TaskRun.BUFFER_LIMIT + 200):
        live.publish({"type": "token", "token": str(index)})
    live.publish({"type": "final", "answer": "x"})
    if live.seq != TaskRun.BUFFER_LIMIT + 202:
        failures.append("TaskRun did not number every published event")
    if not any(e["type"] == "start" for e in live.events):
        failures.append("TaskRun buffer dropped a structural event under token pressure")
    if len(live.events) > TaskRun.BUFFER_LIMIT + 2:
        failures.append("TaskRun buffer grew past its limit")

    # The agent must keep the system prompt and the question no matter how long
    # the tool trace gets.
    agent = Agent(config, registry, ModelClient(config))

    # Disk-backed partial sink: one reused handle, byte cap, tail-biased read.
    agent.PARTIAL_MAX_BYTES = 1500
    agent.partial_begin("selftest-conv", "a test request")
    for _i in range(30):
        agent.partial_add("selftest-conv", f"finding {_i}: " + "y" * 80)
    _entry = agent._partials.get(agent._partial_key("selftest-conv"))
    if not _entry or not _entry["capped"]:
        failures.append("partial sink did not enforce its byte cap")
    if len(agent._partials) != 1:
        failures.append("partial sink reopened the handle per write")
    _sal = agent.salvage("selftest-conv", "ran low")
    if "finding 0" not in _sal or "[...]" not in agent.partial_read("selftest-conv", 400):
        failures.append("partial sink read-back/salvage lost the saved work")
    agent.partial_begin("selftest-conv", "second turn")
    if "finding 0" in agent.partial_read("selftest-conv"):
        failures.append("partial sink did not truncate on a new turn")
    agent._close_partial("selftest-conv")

    base, _ = agent.build_base([], "what is the capital of France?", 256)
    long_scratch = [{"role": "user", "content": "t" * 8000} for _ in range(8)]
    assembled, cut = agent.assemble(base, long_scratch, 256)
    if cut == 0:
        failures.append("agent.assemble kept a trace that cannot fit the context")
    if assembled[: len(base)] != base:
        failures.append("agent.assemble dropped part of the pinned question or system prompt")

    changed = config.apply({"max_tokens": 99999, "context_size": 1024, "model": "evil/model"})
    if config.model == "evil/model":
        failures.append("config.apply changed a field outside MUTABLE")
    if config.max_tokens >= config.context_size:
        failures.append("config.apply left max_tokens above the context size")
    if "context_size" not in changed:
        failures.append("config.apply did not report context_size as changed")

    # Safeguards clamp at construction and on live edit, and stay in valid bands.
    bad = Config(chunk_size_ratio=0.99, chunk_trigger_ratio=0.3,
                 ready_wait_timeout=99999, reasoning_tokens=10 ** 9)
    if bad.chunk_size_ratio > bad.chunk_trigger_ratio:
        failures.append("chunk_size_ratio not clamped below the trigger at construction")
    if not (2.0 <= bad.ready_wait_timeout <= 300.0):
        failures.append("ready_wait_timeout not clamped at construction")
    if not (64 <= bad.reasoning_tokens <= 2048):
        failures.append("reasoning_tokens not clamped at construction")
    edited = Config()
    edited.apply({"reasoning_tokens": 10 ** 9, "ready_wait_timeout": -5})
    if not (64 <= edited.reasoning_tokens <= 2048 and edited.ready_wait_timeout >= 2.0):
        failures.append("safeguards not clamped on live edit")
    for safeguard in ("reasoning_tokens", "chunk_trigger_ratio", "stall_timeout",
                      "ready_wait_timeout", "auto_fetch_char_cap"):
        if safeguard not in Config.MUTABLE:
            failures.append(f"safeguard {safeguard} is not live-editable")

    # ===================================================================== #
    # Search engine constraint: DuckDuckGo Lite, and ONLY DuckDuckGo Lite.
    # ===================================================================== #
    if SearchBackend.PROVIDER != "duckduckgo_lite":
        failures.append(f"search provider is {SearchBackend.PROVIDER!r}, not duckduckgo_lite")
    if tuple(SearchBackend.ENDPOINTS) != ("https://lite.duckduckgo.com/lite/",):
        failures.append(f"search endpoints are not lite-only: {SearchBackend.ENDPOINTS}")
    for _ep in SearchBackend.ENDPOINTS:
        if "lite.duckduckgo.com" not in _ep:
            failures.append(f"search endpoint is not DuckDuckGo Lite: {_ep}")
    # The websearch module must not name any other provider anywhere. Only
    # checkable in the split package; the single-file bundle has no such file.
    _ws_file = Path(__file__).resolve().parent / "websearch.py"
    if __package__ and _ws_file.exists():
        _ws_src = _ws_file.read_text(encoding="utf-8")
        for _banned in ("google.com/search", "bing.com", "brave", "tavily", "searxng",
                        "html.duckduckgo.com"):
            if _banned in _ws_src:
                failures.append(f"websearch.py still references a non-Lite provider: {_banned}")
    if Config().search_backend != "duckduckgo_lite":
        failures.append("default search_backend is not duckduckgo_lite")
    for _other in ("google", "bing", "brave", "tavily", "searxng", "kagi"):
        if Config(search_backend=_other).search_backend != "duckduckgo_lite":
            failures.append(f"Config accepted a non-Lite backend: {_other}")
        if _normalize_search_backend(_other) != "duckduckgo_lite":
            failures.append(f"_normalize_search_backend accepted {_other}")
    for _alias in ("ddg", "duckduckgo", "lite", "ddg-lite", ""):
        if _normalize_search_backend(_alias) != "duckduckgo_lite":
            failures.append(f"_normalize_search_backend rejected the alias {_alias!r}")
    if "search_backend" in Config.MUTABLE:
        failures.append("search_backend must not be runtime-mutable (would allow provider swap)")
    _c = Config()
    _c.apply({"search_backend": "google"})
    if _c.search_backend != "duckduckgo_lite":
        failures.append("apply() let search_backend change away from DuckDuckGo Lite")
    # The lite parser must unwrap a DDG redirect link and skip DDG's own nav.
    _lite_html = ('<a rel="nofollow" class="result-link" '
                  'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=x">Example</a>'
                  '<td class="result-snippet">A snippet.</td>'
                  '<a href="//duckduckgo.com/settings">Settings</a>')
    _res = SearchBackend._parse_lite(_lite_html, 5)
    if not _res or _res[0].url != "https://example.com/page":
        failures.append(f"DDG lite parser did not unwrap the redirect link: {_res}")
    if any("duckduckgo.com" in r.url for r in _res):
        failures.append("DDG lite parser returned a duckduckgo.com nav link")

    # ===================================================================== #
    # Authentication: hashing, sessions, bootstrap, OIDC role mapping.
    # ===================================================================== #
    _h1 = hash_password("hunter2")
    _h2 = hash_password("hunter2")
    if _h1 == _h2:
        failures.append("password hashes are not salted (identical for same input)")
    if not _h1.startswith("scrypt$"):
        failures.append("password hash is not scrypt")
    if not verify_password("hunter2", _h1):
        failures.append("verify_password rejected the correct password")
    if verify_password("wrong", _h1) or verify_password("hunter2", None):
        failures.append("verify_password accepted a wrong/empty credential")
    with tempfile.TemporaryDirectory() as _atmp:
        _adb = Database(Path(_atmp) / "auth.db")
        _acfg = Config(auth_enabled=True, admin_username="root", admin_password="rootpw123",
                       allow_test_user=True, test_username="t", test_password="tpw12345")
        _auth = Auth(_acfg, _adb)
        _auth.bootstrap()
        if _adb.count_users(role="admin") != 1:
            failures.append("bootstrap did not create exactly one admin")
        if _auth.authenticate_local("root", "rootpw123") is None:
            failures.append("admin cannot authenticate with the bootstrapped password")
        if _auth.authenticate_local("root", "nope") is not None:
            failures.append("admin authenticated with the wrong password")
        _tu = _adb.get_user_by_username("t")
        if not _tu or _tu["role"] != "user":
            failures.append("test user was not created as a non-admin")
        _admin = _adb.get_user_by_username("root")
        _tok = _auth.create_session(_admin)
        if not _auth.resolve_token(_tok) or _auth.resolve_token(_tok)["id"] != _admin["id"]:
            failures.append("session token did not resolve to its user")
        _auth.logout(_tok)
        if _auth.resolve_token(_tok) is not None:
            failures.append("session survived logout")
        if _auth.resolve_token("bogus-token") is not None:
            failures.append("a bogus token resolved to a user")
        # bootstrap must never reset an existing admin's password.
        _auth.bootstrap()
        if _auth.authenticate_local("root", "rootpw123") is None:
            failures.append("re-running bootstrap reset the admin password")
        # Auth disabled -> synthetic local admin.
        _off = Auth(Config(auth_enabled=False), _adb)
        if _off.user_for_request(None) is None or _off.user_for_request(None)["role"] != "admin":
            failures.append("auth-disabled did not yield a synthetic local admin")
        # OIDC role mapping by email and by app role.
        _ocfg = Config(oidc_admin_emails="boss@corp.com", oidc_admin_roles="Admins")
        _oauth = Auth(_ocfg, _adb)
        if _oauth.role_for_oidc({"email": "boss@corp.com"}) != "admin":
            failures.append("OIDC admin-by-email mapping failed")
        if _oauth.role_for_oidc({"roles": ["Admins"]}) != "admin":
            failures.append("OIDC admin-by-role mapping failed")
        if _oauth.role_for_oidc({"email": "nobody@corp.com"}) != "user":
            failures.append("OIDC default role should be user")

    # RBAC dependency: require_admin blocks a non-admin, allows an admin.
    class _FakeReq:
        def __init__(self, user):
            self._user = user
            self.cookies = {}
            self.headers = {}
            self.client = type("C", (), {"host": "127.0.0.1"})()
            self.state = type("S", (), {})()

    with tempfile.TemporaryDirectory() as _rtmp:
        _rdb = Database(Path(_rtmp) / "rbac.db")
        _rauth = Auth(Config(auth_enabled=True), _rdb)
        _admin_u = {"id": "a1", "username": "a", "role": "admin"}
        _user_u = {"id": "u1", "username": "u", "role": "user"}
        # Resolve the acting user straight from the fake request's fixed user.
        def _fixed_user(req):
            return req._user

        def _no_user(req):
            return None

        _orig = _rauth.user_for_request
        _rauth.user_for_request = _fixed_user  # type: ignore
        import asyncio as _aio
        try:
            _ok = _aio.run(_rauth.require_admin(_FakeReq(_admin_u)))
            if _ok["role"] != "admin":
                failures.append("require_admin rejected an admin")
            _denied = False
            try:
                _aio.run(_rauth.require_admin(_FakeReq(_user_u)))
            except Exception as _exc:
                _denied = getattr(_exc, "status_code", None) == 403
            if not _denied:
                failures.append("require_admin did not 403 a non-admin")
            _unauth = False
            _rauth.user_for_request = _no_user  # type: ignore
            try:
                _aio.run(_rauth.require_user(_FakeReq(None)))
            except Exception as _exc:
                _unauth = getattr(_exc, "status_code", None) == 401
            if not _unauth:
                failures.append("require_user did not 401 an anonymous request")
        finally:
            _rauth.user_for_request = _orig

    # ===================================================================== #
    # Multi-user data isolation at the database layer.
    # ===================================================================== #
    with tempfile.TemporaryDirectory() as _itmp:
        _idb = Database(Path(_itmp) / "iso.db")
        _idb.add_message("cA", "user", "alice private", user_id="alice")
        _idb.add_message("cB", "user", "bob private", user_id="bob")
        if [c["conversation_id"] for c in _idb.list_conversations(user_id="alice")] != ["cA"]:
            failures.append("conversation list is not user-scoped")
        if _idb.get_messages("cA", user_id="bob"):
            failures.append("a user could read another user's conversation by id")
        if _idb.can_access_conversation("cA", "bob"):
            failures.append("can_access_conversation allowed a cross-user read")
        if not _idb.can_access_conversation("cA", "alice"):
            failures.append("can_access_conversation denied the owner")
        if not _idb.can_access_conversation("brand-new", "bob"):
            failures.append("a new conversation should be claimable by any user")
        _idb.remember("k", "alice-val", user_id="alice")
        _idb.remember("k", "bob-val", user_id="bob")
        if {m["value"] for m in _idb.recall("k", user_id="alice")} != {"alice-val"}:
            failures.append("per-user memory leaked across users")
        if getattr(_idb, "fts_enabled", False):
            _idb.index_document("a.md", "alpha zulu", user_id="alice")
            _idb.index_document("s.md", "shared zulu", user_id=SHARED_OWNER)
            _bob_hits = {h["path"] for h in _idb.search_documents("zulu", user_id="bob")}
            if _bob_hits != {"s.md"}:
                failures.append(f"knowledge base not user-scoped (bob saw {_bob_hits})")
            _alice_hits = {h["path"] for h in _idb.search_documents("zulu", user_id="alice")}
            if _alice_hits != {"a.md", "s.md"}:
                failures.append(f"owner+shared retrieval wrong (alice saw {_alice_hits})")

    # ===================================================================== #
    # Structured logging: redaction (message + fields + objects), levels.
    # ===================================================================== #
    _red = redact_text("Authorization: Bearer sk-abcdef1234567890abcd key api_key=SECRETVALUE123456")
    if "sk-abcdef" in _red or "SECRETVALUE" in _red or "Bearer sk" in _red:
        failures.append(f"redact_text leaked a secret: {_red}")
    _robj = redact_obj({"password": "hunter2", "note": "token=abcdefghijklmnop12345", "n": 3})
    if _robj.get("password") == "hunter2" or "abcdefghijklmnop" in str(_robj.get("note")):
        failures.append(f"redact_obj leaked a secret: {_robj}")
    if _robj.get("n") != 3:
        failures.append("redact_obj mangled a non-secret value")
    _pub = Config(oidc_client_secret="topsecret", node_token="ntok").public()
    if _pub.get("oidc_client_secret") == "topsecret" or _pub.get("node_token") == "ntok":
        failures.append("Config.public() leaked a secret field")
    if _pub.get("oidc_client_secret") != "***set***":
        failures.append("Config.public() did not mark a set secret")
    # Reconfigure into a throwaway log dir so running --selftest never writes to
    # (or pollutes) the real logs/app.log.
    _logtmp = _tf.mkdtemp()
    configure_logging(Config(log_chat_content="disabled", log_dir=_logtmp), force=True)
    if content_for_log("hello") is not None:
        failures.append("content logging disabled still returned content")
    configure_logging(Config(log_chat_content="metadata", log_dir=_logtmp), force=True)
    _meta = content_for_log("hello world")
    if not isinstance(_meta, dict) or "chars" not in _meta or "hello" in str(_meta):
        failures.append("metadata content logging returned the raw text")
    configure_logging(Config(log_chat_content="full", log_dir=_logtmp), force=True)
    if content_for_log("password=hunter2 hi") is None or "hunter2" in content_for_log("password=hunter2 hi"):
        failures.append("full content logging did not redact")
    configure_logging(Config(log_dir=_logtmp), force=True)  # restore defaults (temp dir)
    _cid = set_correlation_id(None)
    if not _cid or get_correlation_id() != _cid:
        failures.append("correlation id did not round-trip")
    set_acting_user("u9")
    if get_acting_user() != "u9":
        failures.append("acting-user contextvar did not round-trip")
    set_acting_user(None)

    # ===================================================================== #
    # Cluster routing: node selection, overload/failover, large-model, dedup.
    # ===================================================================== #
    _rcfg = Config(studio_node_url="http://studio.local:8080", route_max_active_per_node=2)
    _reg = NodeRegistry(_rcfg)
    if not _reg.multi_node or len(_reg.nodes) != 2:
        failures.append("cluster did not build a two-node registry from STUDIO_NODE_URL")
    _crouter = ClusterRouter(_rcfg, _reg)
    _prim = _reg.local_node()
    _sec = next(n for n in _reg.nodes if not n.is_local)
    _reg.update(_prim.name, state=HEALTHY)
    _reg.update(_sec.name, state=HEALTHY)
    if _crouter.select(_crouter.classify(model="Qwen2.5-Coder-7B")).primary_choice.name != _prim.name:
        failures.append("router did not prefer a healthy primary")
    _reg.update(_prim.name, active=2)  # overloaded
    if _crouter.select(_crouter.classify(model="Qwen2.5-Coder-7B")).primary_choice.name != _sec.name:
        failures.append("router did not offload an overloaded primary to the secondary")
    _reg.update(_prim.name, active=0, state=UNAVAILABLE)
    if _crouter.select(_crouter.classify(model="x")).primary_choice.name != _sec.name:
        failures.append("router did not fail over an unavailable primary")
    _reg.update(_prim.name, state=HEALTHY)
    _big = _crouter.select(_crouter.classify(model="Qwen2.5-Coder-32B-Instruct"))
    if _big.primary_choice.name != _sec.name or [n.name for n in _big.candidates] != [_sec.name]:
        failures.append("large model was not pinned to the high-memory secondary")
    _reg.update(_sec.name, state=UNAVAILABLE)
    if _crouter.select(_crouter.classify(model="Qwen2.5-Coder-32B")).primary_choice is None:
        failures.append("router gave no best-effort node when the secondary was down")
    # A primary still loading weights (STARTING) must not be preferred over a
    # healthy secondary — it is not ready to serve.
    _reg.update(_sec.name, state=HEALTHY)
    _reg.update(_prim.name, active=0, state=STARTING)
    if _crouter.select(_crouter.classify(model="x")).primary_choice.name != _sec.name:
        failures.append("router preferred a STARTING primary over a healthy secondary")
    _reg.update(_prim.name, state=HEALTHY)
    # Soft capability preference: heavy reasoning steers to the Studio (which
    # advertises "reasoning") even when the primary is healthy and idle, while
    # ordinary chat stays on the primary.
    _reg.update(_prim.name, active=0, state=HEALTHY)
    _reg.update(_sec.name, active=0, state=HEALTHY)
    if _crouter.select(_crouter.classify(kind="reasoning")).primary_choice.name != _sec.name:
        failures.append("router did not steer heavy reasoning to the capability node")
    if _crouter.select(_crouter.classify(kind="chat")).primary_choice.name != _prim.name:
        failures.append("router steered light chat off the primary")
    # A down Studio must not capture reasoning work: it falls back to the primary.
    _reg.update(_sec.name, state=UNAVAILABLE)
    if _crouter.select(_crouter.classify(kind="reasoning")).primary_choice.name != _prim.name:
        failures.append("reasoning preference ignored the Studio being down")
    _reg.update(_sec.name, state=HEALTHY)

    # Circuit breaker: two dispatch failures trip a node to UNAVAILABLE with a
    # cooldown, during which it is not selected; after the cooldown it is offered
    # once as a half-open trial, and a success closes the breaker.
    _cbcfg = Config(studio_node_url="http://studio.local:8080", route_cooldown_s=0.2)
    _cbreg = NodeRegistry(_cbcfg)
    _cbrouter = ClusterRouter(_cbcfg, _cbreg)
    _cbp = _cbreg.local_node()
    _cbs = next(n for n in _cbreg.nodes if not n.is_local)
    _cbreg.update(_cbp.name, state=HEALTHY)
    _cbreg.update(_cbs.name, state=HEALTHY)
    _cbreg.end(_cbs, ok=False)
    _cbreg.end(_cbs, ok=False)
    if _cbs.state != UNAVAILABLE or _cbs.cooldown_until <= time.time():
        failures.append("circuit breaker did not open with a cooldown after repeated failures")
    if _cbs.name in [n.name for n in _cbrouter.select(_cbrouter.classify()).candidates]:
        failures.append("tripped node was selected during its cooldown")
    time.sleep(0.25)
    if _cbs.name not in [n.name for n in _cbrouter.select(_cbrouter.classify()).candidates]:
        failures.append("tripped node never got a half-open trial after cooldown")
    _cbreg.end(_cbs, ok=True)
    if _cbs.state != HEALTHY or _cbs.cooldown_until != 0.0:
        failures.append("a successful half-open trial did not close the breaker")
    # A healthy local heartbeat must also close the breaker (reset the failure
    # count), so a recovered primary degrades rather than instantly re-tripping on
    # its next single failure.
    _cbreg.update(_cbp.name, state=UNAVAILABLE, consecutive_failures=2,
                  cooldown_until=time.time() + 999)
    def _ready_status():
        return {"status": "ready"}
    _hmon = HealthMonitor(_cbcfg, _cbreg, local_status=_ready_status)
    _hmon._probe_local(_cbp, 1.0, 1.0)
    if _cbp.state != HEALTHY or _cbp.consecutive_failures != 0 or _cbp.cooldown_until != 0.0:
        failures.append("a healthy local heartbeat did not close the circuit breaker")
    # Machine detection: the local node must describe THIS Mac, not a hardcoded
    # "mac-mini", and may only advertise high-memory capabilities it really has.
    if friendly_node_name("MacBookPro17,1", "whatever") != "macbook-pro":
        failures.append("friendly_node_name did not map a MacBookPro model id")
    if friendly_node_name("Macmini9,1", "x") != "mac-mini":
        failures.append("friendly_node_name did not map a Mac mini model id")
    if friendly_node_name("Mac14,13", "Studio-Host") != "studio-host":
        failures.append("friendly_node_name did not fall back to the hostname")
    if "large_model" in capabilities_for_machine(8):
        failures.append("an 8GB machine must not advertise large_model")
    if "large_model" not in capabilities_for_machine(64):
        failures.append("a 64GB machine should advertise large_model")
    _det = detect_machine()
    if not _det.get("cores") or _det.get("ram_gb") is None:
        failures.append("detect_machine did not report cores/ram")
    _lreg = NodeRegistry(Config())
    if _lreg.local_node().name == "mac-mini" and "macmini" not in \
            re.sub(r"[^a-z]", "", (_det.get("model") or "").lower()):
        failures.append("local node is still hardcoded to mac-mini on non-mini hardware")

    # An unmeasurable or merely-busy signal must NOT read as overloaded: a load
    # average of 2.2x per core on an idle laptop used to be reported as
    # "cpu 100%" and tripped the threshold.
    _ohcfg = Config(route_load_ratio=4.0)
    _ohreg = NodeRegistry(_ohcfg)
    _ohn = _ohreg.local_node()
    def _ready_ok():
        return {"status": "ready"}
    _ohmon = HealthMonitor(_ohcfg, _ohreg, local_status=_ready_ok)
    _ohmon._probe_local(_ohn, None, None, 2.2)          # busy but not saturated
    if _ohn.state != HEALTHY:
        failures.append(f"a node at 2.2x load was marked {_ohn.state}, expected healthy")
    _ohmon._probe_local(_ohn, None, None, 9.0)          # genuinely saturated
    if _ohn.state != OVERLOADED:
        failures.append("a node at 9x load per core was not marked overloaded")
    _ohmon._probe_local(_ohn, None, None, None)         # nothing measurable
    if _ohn.state != HEALTHY:
        failures.append("a node with no measurable load was not treated as healthy")

    # Single-node: exactly one candidate, and it is local.
    _sreg = NodeRegistry(Config())
    _srouter = ClusterRouter(Config(), _sreg)
    _sd = _srouter.select(_srouter.classify())
    if _sreg.multi_node or len(_sd.candidates) != 1 or not _sd.candidates[0].is_local:
        failures.append("single-node routing is not a local no-op")
    # Idempotent claim.
    if not _crouter.claim("task-x") or _crouter.claim("task-x"):
        failures.append("router.claim did not detect a duplicate task")
    _crouter.release("task-x")
    if not _crouter.claim("task-x"):
        failures.append("router.release did not free a task id")

    # ===================================================================== #
    # Claude history import: zip-slip blocked, real export parses, dedup.
    # ===================================================================== #
    import io as _io
    import zipfile as _zip
    with tempfile.TemporaryDirectory() as _imtmp:
        _imdb = Database(Path(_imtmp) / "imp.db")
        _mgr = ImportManager(Config(), _imdb)
        # Point the staging dir at the temp tree so the test leaves no litter.
        _stage_base = Path(_imtmp)

        def _stage(iid):
            return _stage_base / iid

        _mgr._staging = _stage  # type: ignore
        # zip-slip must be refused.
        _evil = _io.BytesIO()
        with _zip.ZipFile(_evil, "w") as _z:
            _z.writestr("../../evil.txt", "pwned")
        _eid = _mgr.stage_upload("alice", "evil.zip", _evil.getvalue())
        _mgr.process(_eid, "alice")
        if (_imdb.get_import(_eid) or {}).get("status") != "failed":
            failures.append("zip-slip archive was not rejected")
        # A real Claude export imports conversations + knowledge, isolated per user.
        _conv = [{"uuid": "u1", "name": "Rust", "chat_messages": [
            {"sender": "human", "text": "read a file in rust"},
            {"sender": "assistant", "text": "std::fs::read_to_string"}]}]
        _good = _io.BytesIO()
        with _zip.ZipFile(_good, "w") as _z:
            _z.writestr("conversations.json", json.dumps(_conv))
            _z.writestr("projects.json", json.dumps(
                [{"name": "P", "prompt_template": "Be terse."}]))
        _gid = _mgr.stage_upload("alice", "export.zip", _good.getvalue())
        _counts = _mgr.process(_gid, "alice")
        if _counts["conversations"] != 1 or _counts["messages"] != 2:
            failures.append(f"import did not store the conversation: {_counts}")
        if _counts["skills"] < 1:
            failures.append("import did not save the project instruction as a skill")
        # Conversation ids are namespaced by owner (no cross-user collision).
        _alice_convs = [c["conversation_id"] for c in _imdb.list_conversations(user_id="alice")]
        if not any(c.endswith("-u1") for c in _alice_convs):
            failures.append(f"imported conversation is not owned by the importer: {_alice_convs}")
        if _imdb.list_conversations(user_id="bob"):
            failures.append("another user can see an imported conversation")
        # Re-import is de-duplicated.
        _c2 = _mgr.process(_mgr.stage_upload("alice", "export.zip", _good.getvalue()), "alice")
        if _c2["duplicates"] < 1 or _c2["conversations"] != 0:
            failures.append(f"re-import was not de-duplicated: {_c2}")
        # Bob imports the SAME uuid: must NOT be seen as a duplicate of alice's,
        # and must not clobber alice's knowledge-base doc (owner-namespaced path).
        _bid = _mgr.stage_upload("bob", "export.zip", _good.getvalue())
        _bc = _mgr.process(_bid, "bob")
        if _bc["conversations"] != 1:
            failures.append(f"bob's same-uuid import was wrongly de-duped: {_bc}")
        if getattr(_imdb, "fts_enabled", False):
            if not _imdb.search_documents("read a file in rust", user_id="alice"):
                failures.append("alice's imported KB doc was clobbered by bob's import")
        # Removing bob's import must leave alice's conversation intact.
        _mgr.remove_import(_bid)
        if not any(c.endswith("-u1") for c in
                   [x["conversation_id"] for x in _imdb.list_conversations(user_id="alice")]):
            failures.append("removing one import deleted another user's conversation")
        if _imdb.list_conversations(user_id="bob"):
            failures.append("remove_import did not delete its own conversation")

    # ===================================================================== #
    # "summarise/read <url>" routes to fetch_url (the misroute that answered
    # from the model's own knowledge and then refused, "I can't open links").
    # ===================================================================== #
    for _m, _want in [
        ("summarize https://example.com/a.pdf", "https://example.com/a.pdf"),
        ("read this: https://example.com/x", "https://example.com/x"),
        ("what does https://example.com/y say?", "https://example.com/y"),
        ("tl;dr https://example.com/z", "https://example.com/z"),
        ("give me a summary of https://example.com/w", "https://example.com/w"),
        ("https://example.com/bare", None),   # bare URL -> quick_tool, not this
        ("i was reading a book yesterday, nothing to do with links", None),
    ]:
        if url_read_request(_m) != _want:
            failures.append(f"url_read_request({_m!r}) -> {url_read_request(_m)!r}, want {_want!r}")
    # A bare URL still routes to fetch_url via quick_tool.
    if quick_tool("https://example.com/bare") != ("fetch_url", {"url": "https://example.com/bare"}):
        failures.append("quick_tool no longer handles a bare URL")
    # A message merely mentioning two links is not a single-page read request.
    if url_read_request("compare https://a.com and https://b.com please") is not None:
        failures.append("url_read_request fired on a message with two URLs")

    # ===================================================================== #
    # The CLI must not shadow config.py. Every argparse default that repeated a
    # literal silently beat the RAM-aware default: a 64GB machine ran a 4096
    # context instead of 32768 and trained 30 iterations instead of 300.
    # ===================================================================== #
    # cli imports selftest (for --selftest), so import it lazily here to avoid a
    # cycle. In the bundled single-file build both names are already global.
    _bc = globals().get("build_config")
    _bp = globals().get("build_parser")
    if _bc is None or _bp is None:
        from .cli import build_config as _bc, build_parser as _bp
    _cli_cfg = _bc(_bp().parse_args([]))
    _plain = Config()
    for _f in ("context_size", "train_iters", "train_lr", "train_seq_len",
               "max_tokens", "temperature", "history_turns", "model",
               "reasoning_tokens", "auto_fetch_char_cap"):
        if getattr(_cli_cfg, _f) != getattr(_plain, _f):
            failures.append(f"CLI default shadows config.py for {_f}: "
                            f"{getattr(_cli_cfg, _f)!r} != {getattr(_plain, _f)!r}")
    # An explicitly supplied flag must still win.
    if _bc(_bp().parse_args(["--context-size", "2048"])).context_size != 2048:
        failures.append("an explicit --context-size no longer overrides the default")

    # ===================================================================== #
    # Agents: capability -> tool mapping, office365 stubs, profile CRUD.
    # ===================================================================== #
    _caps_tools = tools_for_capabilities(["file_ops", "web_api"])
    if "read_file" not in _caps_tools or "web_search" not in _caps_tools:
        failures.append("tools_for_capabilities did not unlock file/web tools")
    if "run_shell" in _caps_tools:
        failures.append("tools_for_capabilities leaked a code-exec tool without the capability")
    if "final_answer" not in _caps_tools or "calculator" not in _caps_tools:
        failures.append("tools_for_capabilities dropped an always-on tool")
    if not rag_for_capabilities(["knowledge"]) or rag_for_capabilities(["file_ops"]):
        failures.append("rag_for_capabilities did not follow the knowledge capability")
    if [c["key"] for c in public_capabilities()] != CAPABILITY_ORDER:
        failures.append("public_capabilities is out of order")
    # O365 tools are registered ONLY when an Azure app is configured: an
    # unusable tool's description would otherwise ride in every agent system
    # prompt, which is pure prefill cost on a small machine.
    _oreg = ToolRegistry(Config(project_dir="/tmp"), None)
    if any(n.startswith("o365") for n in _oreg.names()):
        failures.append("office365 tools registered without an Azure app configured")
    _oreg2 = ToolRegistry(Config(project_dir="/tmp", o365_tenant_id="t",
                                 o365_client_id="c", o365_client_secret="s"), None)
    if "o365_mail" not in _oreg2.names():
        failures.append("office365 tools missing even though O365_* is configured")
    elif "not enabled in this build" not in _oreg2.get("o365_mail").handler(action="read"):
        failures.append("office365 stub did not report its state")
    # An agent whose capabilities name a conditional tool is not misconfigured.
    _cond = ToolRegistry(Config(agent_tools=",".join(tools_for_capabilities(["office365"]))), None)
    if "final_answer" not in _cond.names():
        failures.append("capability allowlist dropped final_answer")
    import tempfile as _agtf
    _adb = Database(Path(_agtf.mkdtemp()) / "agents.db")
    _ag = _adb.create_agent("Coder", "writes code", ["file_ops", "code_exec"])
    if _ag["capabilities"] != ["file_ops", "code_exec"] or not _ag["enabled"]:
        failures.append("create_agent did not persist capabilities/enabled")
    _ag2 = _adb.update_agent(_ag["id"], capabilities=["web_api"], enabled=False)
    if _ag2["capabilities"] != ["web_api"] or _ag2["enabled"]:
        failures.append("update_agent did not apply capability/enabled changes")
    if len(_adb.list_agents()) != 1:
        failures.append("list_agents returned the wrong count")
    if not _adb.delete_agent(_ag["id"]) or _adb.list_agents():
        failures.append("delete_agent did not remove the profile")
    _adb.close()

    # ===================================================================== #
    # Import is provider-agnostic: Claude, OpenAI/ChatGPT, DeepSeek, xAI.
    # OpenAI/DeepSeek store the thread as a `mapping` GRAPH, not a list, so a
    # parser that only looks for messages/chat_messages silently drops them all.
    # ===================================================================== #
    _prov = {
        "claude": [{"uuid": "c1", "name": "C", "chat_messages": [
            {"sender": "human", "text": "q-claude"},
            {"sender": "assistant", "content": [{"type": "text", "text": "a-claude"}]}]}],
        "openai": [{"title": "O", "mapping": {
            "root": {"parent": None, "children": ["a"], "message": None},
            "a": {"parent": "root", "children": ["b"], "message": {
                "author": {"role": "system"}, "content": {"parts": ["sys noise"]}, "create_time": 1}},
            "b": {"parent": "a", "children": ["c"], "message": {
                "author": {"role": "user"}, "content": {"parts": ["q-openai"]}, "create_time": 2}},
            "c": {"parent": "b", "children": [], "message": {
                "author": {"role": "assistant"}, "content": {"parts": ["a-openai"]}, "create_time": 3}}}}],
        "deepseek": [{"title": "D", "mapping": {
            "m1": {"parent": None, "children": ["m2"], "message": {
                "role": "user", "content": "q-deepseek", "create_time": 1}},
            "m2": {"parent": "m1", "children": [], "message": {
                "role": "assistant", "content": "a-deepseek", "create_time": 2}}}}],
        "grok": [{"conversation_title": "G", "messages": [
            {"role": "user", "content": "q-grok"},
            {"role": "assistant", "content": [{"text": "a-grok"}]}]}],
    }
    for _name, _payload in _prov.items():
        _pdb = Database(Path(_agtf.mkdtemp()) / f"{_name}.db")
        _pim = ImportManager(Config(), _pdb)
        _pbuf = _io.BytesIO()
        with _zip.ZipFile(_pbuf, "w") as _pz:
            _pz.writestr("conversations.json", json.dumps(_payload))
        _pres = _pim.process(_pim.stage_upload("u", f"{_name}.zip", _pbuf.getvalue()), "u")
        if _pres.get("conversations") != 1:
            failures.append(f"{_name} export did not import (got {_pres.get('conversations')})")
            _pdb.close()
            continue
        _prow = _pdb.search_conversations(f"q-{_name}", limit=3, user_id="u")
        if not _prow:
            failures.append(f"{_name} import stored no findable message")
        else:
            _pmsgs = _pdb.get_messages(_prow[0]["conversation_id"], user_id="u")
            if [m["role"] for m in _pmsgs] != ["user", "assistant"]:
                failures.append(f"{_name} import roles = {[m['role'] for m in _pmsgs]}")
            if not any(f"a-{_name}" in (m["content"] or "") for m in _pmsgs):
                failures.append(f"{_name} import lost the assistant text")
            if any("sys noise" in (m["content"] or "") for m in _pmsgs):
                failures.append(f"{_name} import leaked a system frame into the transcript")
        _pdb.close()

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS  single copy of the file")
    print("PASS  embedded UI parses and renders")
    print("PASS  database schema round-trips")
    print("PASS  tool call parsing, calculator sandbox, workspace confinement")
    print("PASS  file tools, argument aliasing, tool allowlist, fetch guard")
    print("PASS  task scheduling, run lifecycle, event replay, model catalogue")
    print("PASS  model swapping keeps mismatched adapters out of the server")
    print("PASS  reasoning stripping, fast path, prefix reuse, token accounting")
    print("PASS  context trimming and runtime config guardrails")
    print("PASS  search engine locked to DuckDuckGo Lite only")
    print("PASS  auth: password hashing, sessions, bootstrap, OIDC roles, RBAC deps")
    print("PASS  multi-user data isolation (conversations, memory, knowledge base)")
    print("PASS  structured logging: secret redaction and content-log levels")
    print("PASS  cluster routing: selection, overload/failover, large-model, dedup")
    print("PASS  Claude import: zip-slip blocked, export parsed, de-duplicated")
    print("PASS  'summarise/read <url>' routes to fetch_url (no more link refusal)")
    print("PASS  agents: capability->tool mapping, office365 stubs, profile CRUD")
    print("PASS  import works for Claude, OpenAI, DeepSeek and xAI exports")
    return 0



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'selftest',
]
