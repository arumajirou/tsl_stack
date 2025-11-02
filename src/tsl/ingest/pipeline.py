# File: src/tsl/ingest/pipeline.py
# -*- coding: utf-8 -*-
"""
TSL Ingest Pipeline

Features:
- ingest(base, artifacts_out=None, dry_run=True)
  Parse latest log under nf_auto_runs/ and summarize pred.csv rows.

- ingest_path(path, engine=None, dataset_name=None, dry_run=True)
  Load a CSV file, count rows, and optionally persist into a canonical table
  'tsl_observations' via SQLAlchemy engine.

Conventions:
- All path-like arguments accept either str or pathlib.Path.
- dry_run=True guarantees no external side-effects (no DB writes / no files).
- Logging uses tsl.utils.logging.jlog for structured JSON logs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tsl.ingest.parser import parse_log_text
from tsl.utils.logging import jlog


# ---------- helpers ----------

def _latest_log(logs_dir: Path) -> Path:
    """Return the newest (by mtime) log file in logs_dir. Raises if none."""
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
    """Serialize to JSON, prefer orjson when available."""
    try:
        import orjson  # type: ignore
        return orjson.dumps(obj).decode()
    except Exception:
        return json.dumps(obj, ensure_ascii=False, indent=2)


# ---------- public APIs ----------

def ingest(
    base: Union[Path, str],
    artifacts_out: Optional[Union[Path, str]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Analyze the latest nf_auto_runs log and summarize the corresponding pred.csv.

    Args:
        base: Path to nf_auto_runs directory.
        artifacts_out: If provided and dry_run is False, write JSON to this dir.
        dry_run: When True, do not write any files.

    Returns:
        Dict like {"dry_run": bool, "records": [ {run_id, run_dir_rel, model_name, backend, pred_csv, pred_rows} ] }
    """
    if not isinstance(base, Path):
        base = Path(base)
    if artifacts_out is not None and not isinstance(artifacts_out, Path):
        artifacts_out = Path(artifacts_out)

    logs_dir = base / "logs"
    runs_dir = base / "runs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    # parse latest log
    logp = _latest_log(logs_dir)
    parsed = parse_log_text(logp.read_text(encoding="utf-8"))

    # normalize run_dir_rel to "nf_auto_runs/..."
    run_rel_str = getattr(parsed, "run_dir_rel", "")
    run_rel_idx = run_rel_str.find("nf_auto_runs/")
    run_rel = run_rel_str[run_rel_idx:] if run_rel_idx >= 0 else run_rel_str

    # resolve actual run directory under base
    run_dir = base / Path(run_rel).relative_to("nf_auto_runs")

    # count rows in pred.csv (excluding header)
    pred_csv = run_dir / "pred.csv"
    rows = 0
    if pred_csv.exists():
        with pred_csv.open("r", newline="", encoding="utf-8") as f:
            rows = max(0, sum(1 for _ in csv.reader(f)) - 1)

    rec: Dict[str, Any] = {
        "run_id": getattr(parsed, "run_id", None),
        "run_dir_rel": run_rel,
        "model_name": getattr(parsed, "model_name", None),
        "backend": getattr(parsed, "backend", None),
        "pred_csv": str(pred_csv.resolve()),
        "pred_rows": rows,
    }

    jlog("ingest_summary", **rec, dry_run=dry_run)

    out: Dict[str, Any] = {"dry_run": dry_run, "records": [rec]}

    if artifacts_out and not dry_run:
        artifacts_out.mkdir(parents=True, exist_ok=True)
        (artifacts_out / f"ingest_{rec.get('run_id')}.json").write_text(
            _to_json_text(out), encoding="utf-8"
        )

    return out


def ingest_path(
    path: Union[str, Path],
    engine: Any = None,
    dataset_name: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Load a CSV file and optionally store it into a SQL database via SQLAlchemy engine.

    The canonical sink table is 'tsl_observations' with columns:
      - dataset (TEXT) — provided via dataset_name or derived from CSV basename
      - unique_id (TEXT)
      - ds (TIMESTAMP/DATE)
      - y (REAL/NUMERIC)

    Args:
        path: CSV file path to ingest (must include columns unique_id, ds, y).
        engine: SQLAlchemy Engine/Connection. If provided and dry_run=False, rows are written.
        dataset_name: Logical dataset identifier; defaults to CSV stem.
        dry_run: If True, no DB writes are performed.

    Returns:
        Dict:
          {
            "status": "ok",
            "dry_run": bool,
            "path": "<abs path>",
            "rows": <int>,
            "table": "tsl_observations" or None,
            "written_rows": <int>
          }
    """
    from pathlib import Path as _Path
    import pandas as pd

    TARGET_TABLE = "tsl_observations"

    csv_path = _Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate minimal schema
    missing = [c for c in ("unique_id", "ds", "y") if c not in df.columns]
    if missing:
        raise ValueError(f"csv missing required columns: {missing}")

    # Normalize types
    df_norm = df.copy()
    df_norm["ds"] = pd.to_datetime(df_norm["ds"], utc=False, errors="coerce")
    if df_norm["ds"].isna().any():
        raise ValueError("invalid datetime values in 'ds' column")

    rows = int(len(df_norm))

    table = None
    written_rows = 0
    if (engine is not None) and (not dry_run):
        table = TARGET_TABLE
        dataset = dataset_name or csv_path.stem
        out_df = df_norm[["unique_id", "ds", "y"]].copy()
        out_df.insert(0, "dataset", dataset)
        # Create or replace for simplicity in ephemeral tests
        out_df.to_sql(table, con=engine, if_exists="replace", index=False)
        written_rows = rows

    payload = {
        "status": "ok",
        "dry_run": dry_run,
        "path": str(csv_path.resolve()),
        "rows": rows,
        "table": table,
        "written_rows": written_rows,
    }

    jlog("ingest_path", **payload)
    return payload
