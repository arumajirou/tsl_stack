# File: src/tsl/ingest/parser.py
# -*- coding: utf-8 -*-
"""
TSL Ingest Log Parser

Public API
- parse_log_text(text: str) -> ParsedLog

Responsibilities
- Parse log text emitted by NF auto-run to extract:
  - run_dir_rel: e.g., "nf_auto_runs/runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__abcd1234"
  - run_id: the trailing token after the last "__"
  - model_name: the token between "runs/" and first "__" (e.g., "AutoRNN")
  - backend: token after "backend-" prefix if present (e.g., "optuna")

Notes
- Be lenient to absolute paths; downstream normalizes from "nf_auto_runs/".
- Return None for fields that cannot be inferred.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

__all__ = ["ParsedLog", "parse_log_text"]


# A tolerant pattern that captures any run path containing nf_auto_runs/runs/<...>
# We grab the path segment after nf_auto_runs/runs/ up to whitespace or line ending.
_RUNPATH_RE = re.compile(
    r"(?:^|[\s:=])(?P<full>(?:/?[\w./-]*?)?nf_auto_runs/(?P<sub>runs/[^\s]+))",
    re.IGNORECASE | re.MULTILINE,
)

# Common Japanese/English hints we may see in logs:
#   "[INFO] 保存先(短縮): nf_auto_runs/runs/AutoRNN__..."
#   "run_dir: /abs/path/to/nf_auto_runs/runs/AutoRNN__..."
# We don't depend on the prefix text; we just extract the nf_auto_runs path.


@dataclass
class ParsedLog:
    run_dir_rel: str
    run_id: Optional[str]
    model_name: Optional[str]
    backend: Optional[str]


def _extract_model_name(rel_runs_path: str) -> Optional[str]:
    # rel_runs_path is like: "runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__abcd1234"
    try:
        after_runs = rel_runs_path.split("/", 1)[1]
        return after_runs.split("__", 1)[0] if "__" in after_runs else after_runs
    except Exception:
        return None


def _extract_backend(rel_runs_path: str) -> Optional[str]:
    # Find "backend-<name>" between double-underscore separators.
    m = re.search(r"(?:^|__)backend-([^_]+)(?:__|$)", rel_runs_path)
    return m.group(1) if m else None


def _extract_run_id(rel_runs_path: str) -> Optional[str]:
    # The run_id is the trailing token after the last "__"
    if "__" in rel_runs_path:
        return rel_runs_path.rsplit("__", 1)[-1]
    return None


def parse_log_text(text: str) -> ParsedLog:
    """
    Parse the given log text and return a ParsedLog structure.

    Strategy
    - Search for the last occurrence of a path containing "nf_auto_runs/runs/...".
    - Extract model_name, backend, and run_id heuristically from that relative part.
    """
    last_match = None
    for m in _RUNPATH_RE.finditer(text or ""):
        last_match = m
    if not last_match:
        # No recognizable path; return empty fields to avoid hard failure.
        return ParsedLog(run_dir_rel="", run_id=None, model_name=None, backend=None)

    rel_runs_path = last_match.group("sub")  # e.g., "runs/AutoRNN__...__abcd1234"
    run_dir_rel = "nf_auto_runs/" + rel_runs_path

    model_name = _extract_model_name(rel_runs_path)
    backend = _extract_backend(rel_runs_path)
    run_id = _extract_run_id(rel_runs_path)

    return ParsedLog(
        run_dir_rel=run_dir_rel,
        run_id=run_id,
        model_name=model_name,
        backend=backend,
    )
