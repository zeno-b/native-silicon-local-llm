"""Local LLM: a self-contained trainer, server and agent for Apple Silicon.

The code was originally one ~12,700-line deploy.py. It is split here into modules
that mirror how the system actually layers, so a change can be found and read
without scrolling a single enormous file:

    core            constants, paths, RAM-aware defaults, logging
    database        SQLite storage
    sysutil         ports, logs, adapters, model catalogue
    ui              assembles the embedded page from the parts below
      ui_styles         CSS
      ui_markup         HTML skeleton (head, body, tail)
      ui_script_chat    state, helpers, markdown, streaming trace
      ui_script_views   view switching
      ui_script_tasks   the tasks view
      ui_script_models  models, dataset, knowledge base, codebase
      ui_script_panels  history, prompts, palette, theme, backup, boot
    config          every setting and its clamps
    model_server    the mlx_lm.server subprocess supervisor
    training        LoRA retraining from feedback
    websearch       search backends and URL guards
    calculator      the sandboxed arithmetic evaluator
    tools           tool definitions and the registry
    llm             prompt assembly and tool-call parsing
    model_client    HTTP client for the model server
    textutil        classifiers, chunking, routing heuristics
    agent           the agent loop
    tasks           scheduled runs
    api             the FastAPI app
    diagnostics     doctor, benchmark, CSV export
    selftest        the --selftest suite
    cli             argument parsing and main()

Splitting costs no runtime: Python imports each module once at startup and the
whole package still loads in about a tenth of a second. Nothing is imported per
request, and the UI parts are concatenated once at import time, so the browser
still receives exactly the same single page.

Run it with `python3 deploy.py ...` or `python -m local_llm ...`. To produce a
single-file build for deployment, run `python3 bundle.py`.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
