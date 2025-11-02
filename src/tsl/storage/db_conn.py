# /mnt/e/env/ts/zip/tsl_stack/src/tsl/storage/db_conn.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# 既定はリポジトリ直下 preds.sqlite （環境変数で上書き可）
_DB_PATH = os.environ.get(
    "TSL_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "preds.sqlite"),
)

_DDL = """
CREATE TABLE IF NOT EXISTS nf_errors (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at   TEXT NOT NULL,
  run_id       TEXT,
  model_name   TEXT,
  data_csv     TEXT,
  device       TEXT,
  h            INTEGER,
  backend      TEXT,
  error_type   TEXT,
  message      TEXT,
  traceback    TEXT,
  extra_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_nf_errors_created_at ON nf_errors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nf_errors_runid ON nf_errors(run_id);
"""

def get_sqlite_conn() -> sqlite3.Connection:
    """sqlite接続を取得。必要なら nf_errors テーブルを自動作成。"""
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    # テーブルが無ければ作成
    for stmt in filter(None, _DDL.split(";")):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")
    conn.commit()
    return conn
