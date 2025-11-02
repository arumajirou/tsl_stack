# /mnt/e/env/ts/zip/tsl_stack/scripts/check_errors.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DB = os.environ.get("TSL_DB_PATH", str(Path(__file__).resolve().parents[1] / "preds.sqlite"))

def main():
    con = sqlite3.connect(DB)
    q = (
        "SELECT id, created_at, run_id, model_name, error_type, "
        "substr(message,1,120) AS message "
        "FROM nf_errors ORDER BY created_at DESC LIMIT 50"
    )
    try:
        df = pd.read_sql_query(q, con)
    except Exception as e:
        print("nf_errors テーブルが見つかりません。エラー:", e)
        return
    if df.empty:
        print("nf_errors は空です。直近の実行でエラーは記録されていません。")
    else:
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
