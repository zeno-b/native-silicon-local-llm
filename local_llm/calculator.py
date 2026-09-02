"""The sandboxed arithmetic evaluator.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import html
import hashlib
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import sqlite3
import traceback
import shlex
import subprocess
import sys
import textwrap
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field, asdict, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from .core import *  # noqa: F401,F403


# the process. safe_eval runs in the web process, on the request thread, with
# no subprocess timeout around it the way run_python and run_shell have, so an
# unbounded intermediate takes the whole app down rather than one tool call.
# 65536 bits is a 19,728-digit number, past any real calculator use.
# Calculator safety bounds. These cap the only two whitelisted operations whose
# cost is not bounded by expression length (exponentiation and factorial), so a
# seven-character expression cannot allocate until the process dies. Configurable
# for anyone who needs bigger numbers on a bigger machine.
MAX_RESULT_BITS = int(os.environ.get("CALC_MAX_RESULT_BITS", str(1 << 16)))
MAX_FACTORIAL_INPUT = int(os.environ.get("CALC_MAX_FACTORIAL", "1000"))


def _guarded_pow(base: Any, exponent: Any) -> Any:
    """base ** exponent, refusing results too large to hold in memory."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        # bit_length() * exponent is the exact width of the result, computed
        # without building it. 0 and 1 have no growth, so exempt them.
        if base not in (0, 1, -1) and base.bit_length() * exponent > MAX_RESULT_BITS:
            raise ValueError(
                f"refusing to compute a number with about "
                f"{base.bit_length() * exponent // 3.32:.0f} digits"
            )
    try:
        return operator.pow(base, exponent)
    except OverflowError as exc:
        raise ValueError(f"result out of range: {exc}") from exc


def _guarded_factorial(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("factorial needs a whole number")
    if not 0 <= value <= MAX_FACTORIAL_INPUT:
        raise ValueError(f"factorial argument must be between 0 and {MAX_FACTORIAL_INPUT}")
    return math.factorial(value)


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, Any] = {
    name: getattr(math, name)
    for name in ("pi", "e", "tau", "sqrt", "log", "log2", "log10", "exp", "sin",
                 "cos", "tan", "asin", "acos", "atan", "atan2", "floor", "ceil",
                 "factorial", "degrees", "radians", "hypot", "fabs")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
# math.factorial is the other unbounded-cost entry: factorial(9**7) never returns.
_SAFE_NAMES["factorial"] = _guarded_factorial


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without exposing the interpreter.

    eval() on model-produced text is a remote code execution hole, so this walks
    the AST and rejects anything that is not a literal, an operator, or a
    whitelisted math function.
    """
    if len(expression) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            result = _SAFE_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
            # Repeated multiplication reaches the same place as ** by a longer
            # road, so bound every intermediate rather than only the pow.
            if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
                raise ValueError("intermediate result too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _SAFE_NAMES.get(node.func.id)
            if not callable(func):
                raise ValueError(f"unknown function: {node.func.id}")
            return func(*[evaluate(arg) for arg in node.args])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]
        raise ValueError(f"disallowed expression element: {type(node).__name__}")

    return evaluate(tree)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'MAX_FACTORIAL_INPUT',
    'MAX_RESULT_BITS',
    '_SAFE_NAMES',
    '_SAFE_OPERATORS',
    '_guarded_factorial',
    '_guarded_pow',
    'safe_eval',
]
