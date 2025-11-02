# File: sitecustomize.py
# -*- coding: utf-8 -*-
"""
Local developer convenience hook.

Goal:
- Ensure "src/" is on sys.path whenever Python is started from the repo root,
  so `import tsl` works without requiring `pip install -e .` or manual PYTHONPATH.

Behavior:
- Only acts if this file is on sys.path (i.e., when running from repo root).
- No stdout noise; fully silent on success/failure.
"""

from __future__ import annotations
import sys
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
    _SRC = _ROOT / "src"
    _PKG = _SRC / "tsl"
    if _SRC.is_dir() and _PKG.is_dir():
        sp = str(_SRC)
        if sp not in sys.path:
            # Prepend to prioritize live sources over any installed dist.
            sys.path.insert(0, sp)
except Exception:
    # Be permissive—never crash interpreter startup.
    pass
