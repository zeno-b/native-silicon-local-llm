#!/usr/bin/env python3
"""Entry point. Keeps `python3 deploy.py ...` working after the split.

Every flag behaves exactly as before; the implementation now lives in the
local_llm package next to this file.
"""

from __future__ import annotations

from local_llm.cli import main

if __name__ == "__main__":
    main()
