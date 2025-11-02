# File: tsl_integrated_pkg/src/tsl/ingest/pipeline.py
# -*- coding: utf-8 -*-
"""
Packaged TSL Ingest Pipeline (kept consistent with src/ version).

Public API
- ingest(base, artifacts_out=None, dry_run=True)
- ingest_path(base, run_dir_rel)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tsl.ingest.parser import parse_log_text
from tsl.utils.logging import jlog

__all__ = ["ingest", "ingest_path"]


def _latest_log(logs_dir: Path) -> Path:
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")
    if not logs_dir.is_dir():
        raise NotADirectoryError(f"logs is not a directory: {logs_dir}")

    candidates = sorted(logs_dir.glob("*.log"))
    if not candidates:
        candidates = sorted(p for p in logs_dir.iterdir() if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"no log files in: {logs_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _to_json_text(obj: Any) -> str:
    try:
        import orjson  # type: ignore
        return orjson.dumps(obj).decode()
    except Exception:
        return json.dumps(obj, ensure_ascii=False, indent=2)


def ingest_path(base: Union[Path, str], run_dir_rel: Union[Path, str]) -> Path:
    base_p = Path(base) if not isinstance(base, Path) else base
    rel = str(run_dir_rel)
    idx = rel.find("nf_auto_runs/")
    norm = rel[idx:] if idx >= 0 else rel
    return base_p / Path(norm).relative_to("nf_auto_runs")


def ingest(
    base: Union[Path, str],
    artifacts_out: Optional[Union[Path, str]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    base_p = Path(base) if not isinstance(base, Path) else base
    art_p: Optional[Path]
    if artifacts_out is None:
        art_p = None
    else:
        art_p = Path(artifacts_out) if not isinstance(artifacts_out, Path) else artifacts_out

    logs_dir = base_p / "logs"
    runs_dir = base_p / "runs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    logp = _latest_log(logs_dir)
    parsed = parse_log_text(logp.read_text(encoding="utf-8"))

    run_rel_str = getattr(parsed, "run_dir_rel", "")
    idx = run_rel_str.find("nf_auto_runs/")
    run_rel_norm = run_rel_str[idx:] if idx >= 0 else run_rel_str
    run_dir = base_p / Path(run_rel_norm).relative_to("nf_auto_runs")

    pred_csv = run_dir / "pred.csv"
    rows = 0
    if pred_csv.exists():
        with pred_csv.open("r", newline="", encoding="utf-8") as f:
            rows = max(0, sum(1 for _ in csv.reader(f)) - 1)

    rec: Dict[str, Any] = {
        "run_id": getattr(parsed, "run_id", None),
        "run_dir_rel": run_rel_norm,
        "model_name": getattr(parsed, "model_name", None),
        "backend": getattr(parsed, "backend", None),
        "pred_csv": str(pred_csv.resolve()),
        "pred_rows": rows,
    }

    jlog("ingest_summary", **rec, dry_run=dry_run)

    out: Dict[str, Any] = {"dry_run": dry_run, "records": [rec]}

    if art_p is not None and not dry_run:
        art_p.mkdir(parents=True, exist_ok=True)
        (art_p / f"ingest_{rec.get('run_id')}.json").write_text(
            _to_json_text(out), encoding="utf-8"
        )

    return out
