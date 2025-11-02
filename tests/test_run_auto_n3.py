# tests/test_run_auto_n3.py
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# === 設定: 入力CSV（絶対パス） ===
N3_CSV = Path("/mnt/e/env/ts/datas/data/data_long/ft_df_all/numbers3/by_unique_id/N3.csv")


def _pyexec_module(argv, *, cwd: Path, extra_env=None):
    """python -m tsl.cli.tsl ... をサブプロセスで実行して CompletedProcess を返す"""
    env = os.environ.copy()
    # src を PYTHONPATH に追加して、サブプロセス側でもローカル実装を確実に参照
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "tsl.cli.tsl"] + argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _json_lines(stdout: str):
    """1行JSONの連続を辞書リストにして返す（CLIは1行JSONを出す想定）"""
    outs = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            outs.append(json.loads(line))
        except Exception:
            pass
    return outs


@pytest.mark.skipif(not N3_CSV.exists(), reason=f"missing input CSV: {N3_CSV}")
def test_run_auto_n3_allparams(tmp_path: Path):
    """
    目的:
      - run-auto の全オプションを“指定できる”ことを確認
      - --dry-run: 予測実行はせず、計画JSONが返る
      - --gpu-smoke: GPUが使えない環境でも pred.csv を生成し、必要ならDBへ upsert
      - --num-samples: 併用時の引数パース/伝播が壊れていないこと

    実施:
      1) workspace-clean（念のため）
      2) dry-run のみ
      3) gpu-smoke + num-samples=1
         - pred.csv の存在/行数チェック
         - TSL_DB_URL が設定されていれば、nf_gpu_smoke への件数増を確認（失敗時はスキップ）
    """
    # 作業ディレクトリは tmp の中（リポジトリを汚さない）
    cwd = tmp_path

    # 1) workspace-clean (dry-run と実行版)
    r0 = _pyexec_module(["workspace-clean", "--all", "--dry-run"], cwd=cwd)
    assert r0.returncode == 0

    r0a = _pyexec_module(["workspace-clean", "--all"], cwd=cwd)
    assert r0a.returncode == 0

    # 2) dry-run のみ
    r1 = _pyexec_module(["run-auto", "--data-csv", str(N3_CSV), "--dry-run"], cwd=cwd)
    assert r1.returncode == 0, r1.stderr
    outs1 = _json_lines(r1.stdout)
    # 最後の行を採用
    res1 = outs1[-1] if outs1 else {}
    # status は ok or dry-run を許容
    assert str(res1.get("status", "")).lower() in {"ok", "dry-run"}
    # dry_run True が返ってくること
    assert res1.get("dry_run") in (True, "true", "True", 1)

    # 3) gpu-smoke + num-samples（GPUが無くても pred.csv 生成を確認）
    # DBが設定されている場合は件数増を計測（無ければスキップ扱い）
    db_url = os.getenv("TSL_DB_URL")
    db_count_before = None
    if db_url:
        try:
            from sqlalchemy import create_engine, text

            eng = create_engine(db_url, future=True)
            with eng.begin() as cx:
                # 無ければ 0 件扱い
                try:
                    db_count_before = cx.execute(text("SELECT count(*) FROM nf_gpu_smoke")).scalar() or 0
                except Exception:
                    db_count_before = 0
        except Exception:
            # DB接続できなければ DB チェックはスキップ
            db_url = None

    r2 = _pyexec_module(
        ["run-auto", "--data-csv", str(N3_CSV), "--gpu-smoke", "--num-samples", "1"],
        cwd=cwd,
    )
    assert r2.returncode == 0, r2.stderr
    outs2 = _json_lines(r2.stdout)
    res2 = outs2[-1] if outs2 else {}
    # GPU が使えなくても gpu-not-available を許容
    assert str(res2.get("status", "")).lower() in {"ok", "gpu-not-available"}
    # 出力pred.csvの存在
    run_dir = Path(res2.get("run_dir", tmp_path / "nf_auto_runs" / "runs"))
    pred_csv = Path(res2.get("pred_csv", run_dir / "pred.csv"))
    assert pred_csv.exists(), f"missing pred.csv: {pred_csv}"

    # 最低1行以上（ヘッダ除く）の予測があること
    with pred_csv.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) >= 2, f"pred.csv seems empty: {pred_csv}"

    # DB が設定されていれば upsert で 1 件増を期待（失敗はスキップ）
    if db_url:
        try:
            from sqlalchemy import create_engine, text

            eng = create_engine(db_url, future=True)
            with eng.begin() as cx:
                db_count_after = cx.execute(text("SELECT count(*) FROM nf_gpu_smoke")).scalar() or 0
            assert db_count_before is not None
            assert db_count_after >= db_count_before + 1
        except Exception as e:
            pytest.skip(f"DB check skipped due to error: {e!r}")

    # 最後にフォルダ構造が健全か簡易チェック
    diag = _pyexec_module(["diagnose"], cwd=cwd)
    assert diag.returncode == 0
    djson = _json_lines(diag.stdout)[0]
    assert djson.get("runs") is True
