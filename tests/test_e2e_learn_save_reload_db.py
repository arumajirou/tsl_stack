import json
import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.heavy  # 明示実行: pytest -m heavy

def _pyexec_module(args, cwd, env):
    return subprocess.run(
        [sys.executable, "-m", "tsl.cli.tsl", *args],
        cwd=cwd, text=True, capture_output=True, env=env
    )

def _last_json(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip().startswith("{") and ln.strip().endswith("}")]
    return json.loads(lines[-1]) if lines else {}

def test_e2e_learn_save_reload_db(tmp_path: pathlib.Path):
    repo = pathlib.Path(__file__).resolve().parents[1]

    # 実行環境（GPUなし・MLflow無効で軽量に）
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo/'src'}:{env.get('PYTHONPATH','')}"
    env["CUDA_VISIBLE_DEVICES"] = ""     # GPUを隠す
    env["TSL_ENABLE_MLFLOW"] = "0"

    # 1) 最小CSVデータ作成（80行）
    data_csv = tmp_path / "data.csv"
    pd.DataFrame({
        "unique_id": ["u1"] * 80,
        "ds": pd.date_range("2024-01-01", periods=80, freq="D"),
        "y": [float(i % 9) for i in range(80)],
    }).to_csv(data_csv, index=False)

    # 2) ワークスペース初期化
    r0 = _pyexec_module(["workspace-clean", "--all"], cwd=repo, env=env)
    assert r0.returncode == 0, r0.stderr

    # 3) 学習 + 保存（pred.csv 生成確認）
    r1 = _pyexec_module(
        ["run-auto", "--data-csv", str(data_csv), "--num-samples", "1", "--save-model"],
        cwd=repo, env=env
    )
    assert r1.returncode == 0, r1.stderr or r1.stdout
    j1 = _last_json(r1.stdout)
    assert j1.get("status") in ("ok", "gpu-not-available"), j1
    pred_csv = pathlib.Path(j1["pred_csv"])
    assert pred_csv.exists() and pred_csv.stat().st_size > 0

    # モデル保存物の存在（models_full or lightning_logs）
    models_dir = repo / "nf_auto_runs" / "models_full"
    logs_dir = repo / "lightning_logs"
    assert models_dir.exists() or logs_dir.exists()

    # 4) “再学習”として同条件でもう一度実行（パイプラインが安定して回ることを確認）
    r2 = _pyexec_module(
        ["run-auto", "--data-csv", str(data_csv), "--num-samples", "1"],
        cwd=repo, env=env
    )
    assert r2.returncode == 0, r2.stderr or r2.stdout

    # 5) ingest --apply で DB に保存（SQLite）
    db_path = tmp_path / "preds.sqlite"
    env_ing = env.copy()
    env_ing["TSL_DB_URL"] = f"sqlite:///{db_path}"
    env_ing["TSL_DB_TABLE"] = "nf_predictions_test"  # 任意名

    r3 = _pyexec_module(
        ["ingest", "--base", str(repo / "nf_auto_runs"), "--apply"],
        cwd=repo, env=env_ing
    )
    assert r3.returncode == 0, r3.stderr or r3.stdout
    j3 = _last_json(r3.stdout)
    assert j3.get("pred_rows", 0) > 0, j3

    # 6) DB 件数を直接検証
    eng = create_engine(env_ing["TSL_DB_URL"], future=True)
    with eng.begin() as cx:
        count = cx.execute(text("SELECT COUNT(*) FROM nf_predictions_test")).scalar() or 0
    assert count >= j3["pred_rows"]
