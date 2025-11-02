# -*- coding: utf-8 -*-
"""
tsl_stack: 基本/応用/統合を段階的に検証する総合テスト
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest


# ---------- 共通ヘルパ ----------

def _rand_run_id(n: int = 8) -> str:
    import string
    return "".join(random.choices("0123456789abcdef", k=n))

def _write_minimal_pred_csv(p: Path, rows: int = 5) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "unique_id": ["u1"] * rows,
        "ds": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "y_hat": list(range(rows)),
    })
    df.to_csv(p, index=False)

def _write_log_with_runpath(log_file: Path, run_dir_rel: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_text = f"[INFO] 保存先(短縮): {run_dir_rel}\n"
    log_file.write_text(log_text, encoding="utf-8")


# ---------- 1) 基本ユニット（軽量） ----------

@pytest.mark.parametrize("line,expect", [
    ("保存先(短縮): nf_auto_runs/runs/foo__bar__abcd1234", "nf_auto_runs/runs/foo__bar__abcd1234"),
    ("[INFO] 保存先(短縮): nf_auto_runs/runs/X__Y__Z__00112233", "nf_auto_runs/runs/X__Y__Z__00112233"),
    ("no path here", None),
])
def test_log_line_contains_runs_path(line, expect):
    m = re.search(r"(nf_auto_runs/runs/[^\s]+)", line)
    got = m.group(1) if m else None
    assert got == expect


def test_import_modules_smoke():
    __import__("tsl")
    __import__("tsl.ingest.parser")
    __import__("tsl.ingest.pipeline")
    __import__("tsl.cli.tsl")


# ---------- 2) 取り込み系（DB不要） ----------

def test_ingest_dry_run_on_dummy_workspace(tmp_path: Path, monkeypatch):
    """
    nf_auto_runs/ 以下に“ダミーの run ディレクトリと pred.csv”を作り、
    logs にそのパスが書かれたログを置き、ingest(dry-run) が行数を検出できることを確認。
    """
    try:
        from tsl.ingest import pipeline
    except Exception as e:
        pytest.skip(f"tsl.ingest.pipeline import error: {e}")

    base = tmp_path / "nf_auto_runs"
    logs = base / "logs"
    runs = base / "runs"

    run_id = _rand_run_id()
    run_dir_rel = f"nf_auto_runs/runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__{run_id}"
    run_dir_abs = tmp_path / run_dir_rel

    # pred.csv（中身は最小）
    pred_csv = run_dir_abs / "pred.csv"
    _write_minimal_pred_csv(pred_csv, rows=7)

    # ログ1本（パーサが拾える文言で）
    log_file = logs / "nf_auto_run_dummy.log"
    _write_log_with_runpath(log_file, run_dir_rel)

    # ★ 修正ポイント: Path のまま渡す（str にしない）
    result = pipeline.ingest(base=base, artifacts_out=None, dry_run=True)

    as_json = json.loads(json.dumps(result, default=str))
    text = json.dumps(as_json, ensure_ascii=False)
    assert "dry_run" in text and "true" in text.lower()
    assert run_id in text
    assert '"pred_rows": 7' in text or '"pred_rows": "7"' in text


# ---------- 3) CLI 統合（軽量, DB不要） ----------

def _run_cli(args: list[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
    py = sys.executable
    cmd = [py, "-m", "tsl.cli.tsl", *args]
    env_all = os.environ.copy()
    # PYTHONPATH を伝播（テスト環境で src 優先のため）
    if "PYTHONPATH" not in env_all:
        here = Path(__file__).resolve().parents[1] / "src"
        if here.exists():
            env_all["PYTHONPATH"] = str(here)
    if env:
        env_all.update(env)
    return subprocess.run(cmd, env=env_all, capture_output=True, text=True, timeout=180)

def test_cli_help_smoke():
    r = _run_cli(["--help"])
    assert r.returncode == 0
    assert "TSL CLI" in r.stdout

def test_cli_diagnose_and_ingest_dry(tmp_path: Path):
    base = tmp_path / "nf_auto_runs"
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "runs").mkdir(parents=True, exist_ok=True)

    run_id = _rand_run_id()
    run_dir_rel = f"nf_auto_runs/runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__{run_id}"
    run_dir_abs = tmp_path / run_dir_rel
    _write_minimal_pred_csv(run_dir_abs / "pred.csv", rows=3)
    _write_log_with_runpath(base / "logs" / "latest.log", run_dir_rel)

    r1 = _run_cli(["diagnose", "--base", str(base)])
    assert r1.returncode == 0
    assert '"logs": true' in r1.stdout
    assert '"runs": true' in r1.stdout

    r2 = _run_cli(["ingest", "--base", str(base)])
    assert r2.returncode == 0
    assert "pred_rows" in r2.stdout and "3" in r2.stdout


# ---------- 4) 応用：DB 伴う apply / migrate（任意, 無ければ skip） ----------

@pytest.mark.db
def test_cli_migrate_and_ingest_apply_with_db(tmp_path: Path):
    db_url = os.getenv("TSL_DB_URL")
    if not db_url:
        pytest.skip("TSL_DB_URL が未設定のため DB 系テストは skip")

    base = tmp_path / "nf_auto_runs"
    run_id = _rand_run_id()
    run_dir_rel = f"nf_auto_runs/runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__{run_id}"
    run_dir_abs = tmp_path / run_dir_rel
    _write_minimal_pred_csv(run_dir_abs / "pred.csv", rows=1)
    _write_log_with_runpath(base / "logs" / "apply.log", run_dir_rel)

    r0 = _run_cli(["migrate"])
    assert r0.returncode == 0, f"migrate failed: {r0.stderr or r0.stdout}"

    r1 = _run_cli(["ingest", "--base", str(base), "--apply"])
    assert r1.returncode == 0, f"ingest --apply failed: {r1.stderr or r1.stdout}"


# ---------- 5) E2E（任意/遅い） run-auto → ingest ----------

@pytest.mark.e2e
def test_e2e_run_auto_then_ingest(tmp_path: Path):
    """
    小さなCSVを作って run-auto を実行（GPUは既定でOFF）、そのログに基づいて ingest まで流す。
    前提パッケージ（torch, neuralforecast）が import できない環境では自動 skip。
    """
    # 前提が無ければ skip（環境差に強く）
    try:
        import torch  # noqa: F401
        import neuralforecast  # noqa: F401
    except Exception as e:
        pytest.skip(f"E2E 前提ライブラリが無いため skip: {e}")

    # 1) 入力CSV（最小構成）
    data_csv = tmp_path / "data.csv"
    pd.DataFrame({
        "unique_id": ["u1"] * 50,
        "ds": pd.date_range("2024-01-01", periods=50, freq="D"),
        "y": [float(i % 7) for i in range(50)],
    }).to_csv(data_csv, index=False)

    # 2) 実行環境（GPU無効化＆お試し設定）
    env = {
        "NF_DATA_CSV": str(data_csv),
        "NF_SAVE_MODEL": "0",
        "NF_TRIAL_NUM_SAMPLES": "1",
        "TSL_ENABLE_MLFLOW": "0",
        "CUDA_VISIBLE_DEVICES": "",  # GPUを隠す
    }

    # 3) run-auto
    r = _run_cli(["run-auto"], env=env)
    assert r.returncode == 0, f"run-auto failed: {r.stderr or r.stdout}"

    # 4) 出力ベース（CWD配下に nf_auto_runs/ 想定）
    base = Path.cwd() / "nf_auto_runs"
    assert base.exists(), f"nf_auto_runs not found at {base}"

    # 5) ingest (dry)
    r2 = _run_cli(["ingest", "--base", str(base)])
    assert r2.returncode == 0, f"ingest dry failed: {r2.stderr or r2.stdout}"
    assert "pred_rows" in r2.stdout
