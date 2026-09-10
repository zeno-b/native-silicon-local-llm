#!/usr/bin/env python3
"""Regenerate a single-file deploy_bundled.py from the local_llm package.

The split improves readability; this keeps the old "one file you can copy
anywhere and run" deployment. It concatenates the modules in dependency order,
drops the intra-package imports and the generated __all__ blocks, and keeps one
copy of the shared import header.

    python3 bundle.py            -> writes deploy_bundled.py
    python3 deploy_bundled.py --selftest
"""

from __future__ import annotations

import ast
from pathlib import Path

ORDER = [
    "core", "obslog", "database", "sysutil",
    # UI parts must precede ui.py, which assembles HTML_PAGE from them.
    "ui_styles", "ui_markup", "ui_script_chat", "ui_script_views",
    "ui_script_tasks", "ui_script_models", "ui_script_auth", "ui_script_panels",
    "ui", "config", "model_server", "training",
    "websearch", "calculator", "tools", "llm", "model_client", "textutil",
    "taskstate", "agent", "tasks", "auth", "cluster", "claude_import", "api",
    "diagnostics", "selftest", "cli",
]

PKG = Path(__file__).resolve().parent / "local_llm"
OUT = Path(__file__).resolve().parent / "deploy_bundled.py"


def strip_module(text: str) -> tuple[str, list[str]]:
    """Return (body, stdlib_import_lines) with package plumbing removed.

    Uses the parse tree to find the module docstring and the top-level imports,
    so docstrings *inside* functions and classes are never touched.
    """
    tree = ast.parse(text)
    lines = text.split("\n")
    drop: set[int] = set()          # 1-indexed line numbers to remove
    imports: list[str] = []

    body_nodes = list(tree.body)
    if (body_nodes and isinstance(body_nodes[0], ast.Expr)
            and isinstance(body_nodes[0].value, ast.Constant)
            and isinstance(body_nodes[0].value.value, str)):
        doc = body_nodes[0]
        drop.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:      # from .x import *
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
            drop.update(span)
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                imports.extend(lines[i - 1] for i in span)
        elif isinstance(node, ast.Assign):                        # generated __all__
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    start = node.lineno
                    # take the explanatory comment above it too
                    while start > 1 and lines[start - 2].startswith("#"):
                        start -= 1
                    drop.update(range(start, (node.end_lineno or node.lineno) + 1))

    body = [line for i, line in enumerate(lines, 1) if i not in drop]
    return "\n".join(body).strip("\n"), imports


def merge_imports(lines: list[str]) -> list[str]:
    """Collapse the collected import lines into one statement per module.

    Modules import only what they use, so the same module arrives here through
    several different lines ("from dataclasses import dataclass" from one file,
    "from dataclasses import dataclass, field" from another). De-duplicating by
    line text alone would emit all of them: correct, but three redundant imports
    where one belongs. Union the names per module instead.
    """
    plain: list[str] = []                       # "import x" / "import x as y"
    froms: dict[str, list[str]] = {}            # module -> imported names
    for line in lines:
        node = ast.parse(line.strip()).body[0]
        if isinstance(node, ast.Import):
            for alias in node.names:
                label = ("import " + alias.name
                         + (f" as {alias.asname}" if alias.asname else ""))
                if label not in plain:
                    plain.append(label)
        else:                                    # ast.ImportFrom
            module = "." * node.level + (node.module or "")
            names = froms.setdefault(module, [])
            for alias in node.names:
                label = alias.name + (f" as {alias.asname}" if alias.asname else "")
                if label not in names:
                    names.append(label)
    merged = sorted(plain)
    merged += [f"from {module} import " + ", ".join(sorted(names))
               for module, names in sorted(froms.items())]
    return merged


def main() -> None:
    seen_imports: list[str] = []
    chunks: list[str] = []
    for name in ORDER:
        body, imports = strip_module((PKG / f"{name}.py").read_text())
        for imp in imports:
            if imp not in seen_imports:
                seen_imports.append(imp)
        chunks.append(f"# {'=' * 70}\n# {name}\n# {'=' * 70}\n\n{body}")
    header = (
        "#!/usr/bin/env python3\n"
        '"""Local LLM — single-file build.\n\n'
        "Generated by bundle.py from the local_llm package. Edit the package, not\n"
        "this file; re-run `python3 bundle.py` to rebuild.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        + "\n".join(merge_imports(seen_imports)) + "\n\n\n"
    )
    footer = '\n\nif __name__ == "__main__":\n    main()\n'
    OUT.write_text(header + "\n\n\n".join(chunks) + footer)
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
