# tests/test_cli_ingest_db_apply.py
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.heavy  # デフォでは実行されない。-m heavy で実行。


def _run_cli(args, cwd: Path = None, env: dict | None = None):
    cmd = [sys.executable, "-m", "tsl.cli.tsl"] + args
    return subprocess.run(cmd, text=True, capture_output=True, cwd=cwd, env=env)


@pytest.mark.order(after="tests/test_heavy_train_save_reload.py::test_lightning_train_save_reload_and_make_pred_csv")
def test_cli_diagnose_and_ingest_apply_sqlite(tmp_path: Path, monkeypatch):
    # 1) 事前に pred.csv を置いた runs ディレクトリを用意（heavyテストの生成物を再現）
    base = tmp_path / "nf_auto_runs"
    runs = base / "runs" / "manual_run"
    runs.mkdir(parents=True, exist_ok=True)

    # 最小の pred.csv を作成（万一前テストの成果物が無い場合でも単独で動く）
    import pandas as pd
    t0 = pd.Timestamp.utcnow().normalize()
    df = pd.DataFrame({
        "unique_id": ["tiny"] * 5,
        "ds": [t0 + pd.Timedelta(days=i) for i in range(5)],
        "y_hat": [0.1 * i for i in range(5)],
    })
    pred_csv = runs / "pred.csv"
    df.to_csv(pred_csv, index=False)

    # 2) diagnose（JSONに "runs": true / "logs": (true|false) を含む）
    r1 = _run_cli(["diagnose", "--base", str(base)])
    assert r1.returncode == 0, r1.stderr
    assert '"runs": true' in r1.stdout

    # 3) ingest --apply with SQLite
    #    SQLiteはファイルURLでOK: sqlite:////absolute/path/to/db.sqlite
    db_file = tmp_path / "preds.sqlite"
    db_url = f"sqlite:////{db_file}"
    env = os.environ.copy()
    env["TSL_DB_URL"] = db_url
    env["TSL_DB_TABLE"] = "nf_predictions_test"

    r2 = _run_cli(["ingest", "--base", str(base), "--apply"], env=env)
    assert r2.returncode == 0, r2.stderr

    last = r2.stdout.strip().splitlines()[-1]
    j = json.loads(last)
    assert j.get("ingest") in ("applied", "dry")
    assert j.get("pred_rows") == 5
    # DBへ書けたことを確認
    assert j.get("db_url") == db_url
    assert j.get("db_table") == "nf_predictions_test"
    assert j.get("written_rows") == 5

    # 4) SQLで確認
    from sqlalchemy import create_engine, text
    eng = create_engine(db_url, future=True)
    with eng.begin() as cx:
        cnt = cx.execute(text("SELECT count(*) FROM nf_predictions_test")).scalar()
        assert cnt == 5
