# File: src/tsl/utils/logging.py
# -*- coding: utf-8 -*-
"""
Lightweight structured logging utilities.

Public API:
- jlog(event: str, **fields) -> None
    Emit a single JSON line to stdout:
    {"event": "<event>", "ts_ms": <epoch_ms>, ...fields}

Design:
- Minimal dependency surface (stdlib only).
- Safe to import in tests; does not configure global logging handlers.
- Intended for piping into collectors (jq, json.tool, etc.).
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict

__all__ = ["jlog"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def jlog(event: str, /, **fields: Any) -> None:
    """
    Emit a structured JSON log line to stdout.

    Args:
        event: Logical event name (e.g., "ingest_summary").
        **fields: Arbitrary serializable fields to include.

    Contract:
        - Always contains "event" and "ts_ms".
        - Never raises on serialization; falls back to str(value) if needed.
        - Writes exactly one line and flushes.

    Example:
        jlog("ingest_summary", run_id="abcd1234", rows=42, dry_run=True)
    """
    payload: Dict[str, Any] = {"event": event, "ts_ms": _now_ms()}
    # Make best-effort to JSON-serialize; fallback to string repr per field.
    for k, v in fields.items():
        try:
            json.dumps(v)
            payload[k] = v
        except Exception:
            payload[k] = str(v)

    try:
        line = json.dumps(payload, ensure_ascii=False)
    except Exception:
        # Last-resort: stringify every value
        payload = {k: (str(v)) for k, v in payload.items()}
        line = json.dumps(payload, ensure_ascii=False)

    print(line, file=sys.stdout, flush=True)
