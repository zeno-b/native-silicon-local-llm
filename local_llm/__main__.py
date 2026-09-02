"""Allows `python -m local_llm` to run the app exactly like the old script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
