\
from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
from sqlalchemy import create_engine, text
import json, time, socket, getpass, platform

def record_nf_run(*, db_url: str, model_dir: Path, status: str, pred_rows: int,
                  started_at: float, finished_at: float, backend: str="optuna") -> Optional[str]:
    """Store a minimal run record into nf_runs/nf_artifacts. Idempotent: no. Thread-safe: cond."""
    eng = create_engine(db_url, future=True)
    with eng.begin() as cx:
        cx.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS nf_runs(
            run_id TEXT PRIMARY KEY,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            duration_sec DOUBLE PRECISION,
            status TEXT,
            backend TEXT,
            model_dir TEXT,
            extra JSONB
        )""")
        cx.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS nf_artifacts(
            artifact_id BIGSERIAL PRIMARY KEY,
            run_id TEXT,
            kind TEXT,
            rel_path TEXT,
            size_bytes BIGINT
        )""")
        run_id = str(int(started_at*1000))
        cx.execute(text("""
            INSERT INTO nf_runs(run_id, started_at, finished_at, duration_sec, status, backend, model_dir, extra)
            VALUES(:r, to_timestamp(:s), to_timestamp(:f), :d, :st, :b, :m, :e)
            ON CONFLICT (run_id) DO NOTHING
        """), {"r": run_id, "s": started_at, "f": finished_at, "d": finished_at-started_at,
               "st": status, "b": backend, "m": str(model_dir), "e": json.dumps({"pred_rows": pred_rows})})
        pred = model_dir / "pred.csv"
        if pred.exists():
            cx.execute(text("""
            INSERT INTO nf_artifacts(run_id, kind, rel_path, size_bytes)
            VALUES(:r, 'pred', :p, :sz)
            ON CONFLICT DO NOTHING
            """), {"r": run_id, "p": str(pred), "sz": pred.stat().st_size})
    return run_id
