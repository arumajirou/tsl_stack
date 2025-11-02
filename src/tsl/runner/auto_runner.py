# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import csv
import json
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# =========================================================
# Exceptions
# =========================================================
class NotSupportedError(RuntimeError):
    pass


# =========================================================
# Constants / Paths
# =========================================================
RUNS_ROOT = Path("nf_auto_runs")


# =========================================================
# Utilities
# =========================================================
def _ensure_workspace() -> Path:
    root = RUNS_ROOT
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "models_full").mkdir(parents=True, exist_ok=True)
    (root / "_diag").mkdir(parents=True, exist_ok=True)
    return root


def _rand_id(n: int = 8) -> str:
    return hashlib.sha1(uuid.uuid4().hex.encode()).hexdigest()[:n]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _print_json_line(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _device_from_env() -> str:
    # CUDA_VISIBLE_DEVICES が空でなければ CUDA 判定を試みる
    try:
        import torch  # type: ignore
        if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _write_log_line(base: Path, msg: str) -> None:
    ts = int(time.time())
    log = base / "logs" / f"nf_auto_run_{ts}.log"
    with open(log, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    latest = base / "logs" / "latest.log"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log.name)
    except Exception:
        pass


def _gpu_smoke_pred(run_dir: Path) -> Path:
    """学習せずに 4 行のダミー予測 CSV を作る（GPU 健全性チェック用）"""
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "pred.csv"
    rows = []
    start = pd.Timestamp("2025-09-18")
    for i in range(4):
        rows.append(("SMK", (start + pd.Timedelta(days=i)).date().isoformat(), 9.0))
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "ds", "yhat"])
        w.writerows(rows)
    return p


# =========================================================
# Data preparation
# =========================================================
def prepare_df_for_nf(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """NF に渡す最低限の形 (unique_id, ds, y)。外生は使わず NaN 起因の落ちを避ける。"""
    def pick(colnames, df_):
        for c in colnames:
            if c in df_.columns:
                return c
        return None

    uid = pick(["unique_id", "id", "series", "item_id"], df) or "unique_id"
    ds = pick(["ds", "date", "timestamp", "datetime"], df) or "ds"
    y = pick(["y", "value", "target"], df)
    df = df.copy()

    if uid not in df.columns:
        df[uid] = "series_0"

    if ds not in df.columns:
        # 疑似日付を付与（D）
        df["__idx"] = df.groupby(uid).cumcount()
        df[ds] = pd.Timestamp("2000-01-01") + pd.to_timedelta(df["__idx"], unit="D")

    df[ds] = pd.to_datetime(df[ds], errors="coerce")
    if y is None:
        # 最初の数値列を y に採用
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("数値の目的列が見つかりません（y/value/target または最初の数値列）。")
        y = num_cols[0]

    core = df[[uid, ds, y]].rename(columns={uid: "unique_id", ds: "ds", y: "y"}).copy()
    core = core.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    # y が NaN の行は落とす
    core = core.dropna(subset=["y"]).reset_index(drop=True)

    # freq 推定（だめなら 'D'）
    try:
        f = pd.infer_freq(core[core["unique_id"] == core["unique_id"].iloc[0]]["ds"])
        freq = f or "D"
    except Exception:
        freq = "D"
    return core, freq


def _parse_val_size(raw: Optional[str], df: pd.DataFrame, h: int) -> int:
    if not raw or raw.strip().lower() in ("", "auto", "h"):
        return int(h)
    s = raw.strip().lower()
    try:
        x = float(s)
        if 0 < x < 1:
            # 比率扱い
            m = int(df.groupby("unique_id")["ds"].count().min())
            return max(1, int(round(m * x)))
        return max(1, int(round(x)))
    except Exception:
        return int(h)


# =========================================================
# Manual API (kept)
# =========================================================
def manual_fit_and_save(
    df: pd.DataFrame,
    model_name: str = "AutoRNN",
    h: int = 24,
    val_size: int = 24,
    backend: str = "optuna",
    num_samples: int = 1,
    out_dir: str | Path = "./artifacts/checkpoints",
    overwrite: bool = True,
    save_dataset: bool = True,
) -> str:
    """ユーザが直接呼べる：学習→保存。外生は使わず y のみで学習。"""
    from neuralforecast.core import NeuralForecast  # type: ignore
    from neuralforecast.auto import AutoRNN, AutoLSTM  # type: ignore

    df2, freq = prepare_df_for_nf(df)
    # Auto* の選択（今は AutoRNN だけ明示）
    auto_cls = AutoRNN if model_name == "AutoRNN" else AutoLSTM
    model = auto_cls(h=int(h), backend=str(backend), num_samples=int(num_samples))
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=df2, val_size=int(val_size))
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    nf.save(path=out_dir, overwrite=overwrite, save_dataset=save_dataset)
    return str(Path(out_dir).resolve())


# =========================================================
# Options
# =========================================================
@dataclass
class Options:
    data_csv: Optional[str] = None
    num_samples: int = 1
    gpu_smoke: bool = False
    dry_run: bool = False
    # 追加
    save_model: Optional[bool] = None   # None の場合は環境変数 NF_SAVE_MODEL
    overwrite_model: bool = False
    val_size: Optional[str] = None      # "h" / int / 0~1 の比率文字列


# =========================================================
# Main runner
# =========================================================
def run_neuralforecast_auto(args) -> Dict[str, Any]:
    """
    - gpu_smoke: ダミー4行の pred.csv を出力（学習なし）
    - 通常: AutoRNN で学習→保存（任意）→予測→pred.csv
      ※ Auto* は内部に h を保持するため、推論は nf.predict()（引数なし）を使う。
    """
    # argparse.Namespace / dict / dataclass いずれでも受けられるよう吸収
    get = (lambda k, default=None: getattr(args, k, getattr(args, k.replace("-", "_"), default)) if hasattr(args, k) or hasattr(args, k.replace("-", "_")) else (args.get(k, default) if isinstance(args, dict) else default))
    opts = Options(
        data_csv=get("data_csv"),
        num_samples=int(get("num_samples", 1) or 1),
        gpu_smoke=bool(get("gpu_smoke", False)),
        dry_run=bool(get("dry_run", False)),
        save_model=get("save_model", None),
        overwrite_model=bool(get("overwrite_model", False)),
        val_size=get("val_size", None),
    )

    # 環境変数フォールバック
    if opts.save_model is None:
        env = (os.environ.get("NF_SAVE_MODEL", "1").strip().lower())
        opts.save_model = env in ("1", "true", "yes", "on")
    overwrite = bool(opts.overwrite_model or (os.environ.get("NF_OVERWRITE_MODEL", "0").strip().lower() in ("1","true","yes","on")))

    base = _ensure_workspace()
    device = _device_from_env()

    # dry-run ならメタのみ
    if opts.dry_run:
        payload = {
            "event": "nf_auto_run",
            "status": "dry-run",
            "ts_ms": _now_ms(),
            "data_csv": opts.data_csv,
            "planned_models": ["AutoRNN"],
            "hpo_num_samples": int(opts.num_samples),
            "dry_run": True,
            "rows": None,
        }
        _print_json_line(payload)
        return payload

    # ランID/保存先
    rid = _rand_id(8)
    run_dir = base / "runs" / f"AutoRNN__optuna__backend-optuna__h-24__num_samples-{opts.num_samples}__{rid}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # GPU スモーク
    if opts.gpu_smoke:
        pred = _gpu_smoke_pred(run_dir)
        _write_log_line(base, f"gpu_smoke -> {pred}")
        payload = {
            "event": "nf_auto_gpu_smoke",
            "status": "ok",
            "ts_ms": _now_ms(),
            "data_csv": os.path.relpath(opts.data_csv, Path.cwd()) if opts.data_csv else "gpu_smoke.csv",
            "planned_models": ["AutoRNN"],
            "hpo_num_samples": int(opts.num_samples),
            "dry_run": False,
            "device": device,
            "cuda_error": None if device == "cuda" else "gpu-not-available",
            "run_dir": str(run_dir.resolve()),
            "pred_csv": str(pred.resolve()),
            "db": "error:OperationalError" if os.getenv("TSL_DB_URL") else None,
        }
        _print_json_line(payload)
        # 互換: サマリー行
        _print_json_line({
            "status": "ok",
            "dry_run": False,
            "device": device,
            "cuda_error": None if device == "cuda" else "gpu-not-available",
            "data_csv": payload["data_csv"],
            "rows": None,
            "run_dir": payload["run_dir"],
            "pred_csv": payload["pred_csv"],
            "db": payload["db"],
        })
        return payload

    # 通常実行（学習あり）
    data_csv = opts.data_csv or os.getenv("NF_DATA_CSV")
    if not data_csv:
        raise NotSupportedError("data_csv is required for full run (or set NF_DATA_CSV).")

    # TF32 推奨の警告対策（性能寄りの設定）
    try:
        import torch  # type: ignore
        torch.set_float32_matmul_precision(os.environ.get("TORCH_F32_PRECISION", "high"))
    except Exception:
        pass

    df = pd.read_csv(data_csv)
    df2, freq = prepare_df_for_nf(df)

    # 推論 horizon / val_size
    H = 24
    val_size = _parse_val_size(opts.val_size or os.environ.get("NF_VAL_SIZE", "h"), df2, H)

    # 学習
    from neuralforecast.core import NeuralForecast  # type: ignore
    from neuralforecast.auto import AutoRNN       # type: ignore

    model = AutoRNN(h=H, backend="optuna", num_samples=int(opts.num_samples))
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=df2, val_size=int(val_size))

    # まず保存（予測より先に行う：推論失敗時でも学習成果を残す）
    model_dir_full: Optional[Path] = None
    if bool(opts.save_model):
        tokens = f"backend-optuna__h-{H}__num_samples-{opts.num_samples}__val_size-{val_size}"
        model_dir_full = base / "models_full" / f"AutoRNN__optuna__{tokens}__{rid}"
        model_dir_full.mkdir(parents=True, exist_ok=True)
        nf.save(path=str(model_dir_full), overwrite=overwrite, save_dataset=True)
        # 付随メタ
        kwargs_json = {
            "backend": "optuna",
            "h": H,
            "num_samples": int(opts.num_samples),
            "val_size": int(val_size),
        }
        with open(model_dir_full / "kwargs.json", "w", encoding="utf-8") as f:
            json.dump(kwargs_json, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(model_dir_full / "meta.json", "w", encoding="utf-8") as f:
            meta = {
                "auto_model": "AutoRNN",
                "freq": str(freq),
                "n_series": int(df2["unique_id"].nunique()),
                "rows": int(len(df2)),
                "ds_min": str(df2["ds"].min()),
                "ds_max": str(df2["ds"].max()),
                "run_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_csv": str(Path(data_csv).resolve()),
            }
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    # 予測：★ Auto* は h を内部保持するため、引数なしで呼ぶこと！
    # 旧: yhat = nf.predict(h=H) -> TypeError 回避のため修正
    yhat = nf.predict()
    pred_path = run_dir / "pred.csv"
    yhat.to_csv(pred_path, index=False)

    # 出力（互換 JSON）
    payload = {
        "event": "nf_auto_run",
        "status": "ok",
        "ts_ms": _now_ms(),
        "data_csv": str(Path(data_csv).resolve()),
        "planned_models": ["AutoRNN"],
        "hpo_num_samples": int(opts.num_samples),
        "dry_run": False,
        "rows": int(len(df2)),
        "run_dir": str(run_dir.resolve()),
        "pred_csv": str(pred_path.resolve()),
        "device": device,
    }
    _print_json_line(payload)
    # サマリー行
    _print_json_line({
        "status": "ok",
        "dry_run": False,
        "data_csv": payload["data_csv"],
        "rows": payload["rows"],
        "planned_models": payload["planned_models"],
        "hpo_num_samples": payload["hpo_num_samples"],
        "run_dir": payload["run_dir"],
        "pred_csv": payload["pred_csv"],
        "device": payload["device"],
    })
    return payload
