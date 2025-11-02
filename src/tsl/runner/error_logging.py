# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def _db_path() -> Path:
    # プロジェクトルート直下の preds.sqlite を既定に
    return Path.cwd() / "preds.sqlite"


def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS nf_errors (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%SZ','now')),
      run_id TEXT,
      model_name TEXT,
      error_type TEXT,
      message TEXT
    )
    """)
    conn.commit()


def log_error_row(*, run_id: str, model_name: str, exc: Exception):
    try:
        db = _db_path()
        conn = sqlite3.connect(str(db))
        try:
            _ensure_table(conn)
            etype = type(exc).__name__
            msg = str(exc)
            conn.execute(
                "INSERT INTO nf_errors (run_id, model_name, error_type, message) VALUES (?,?,?,?)",
                (run_id, model_name, etype, msg)
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # DB ログが失敗しても学習系の失敗情報はコンソールで出るので致命ではない
        pass
