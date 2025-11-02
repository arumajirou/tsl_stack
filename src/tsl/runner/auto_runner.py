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
from typing import Any, Dict, Optional, Tuple, List

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
    # 明示的に "CUDA_VISIBLE_DEVICES" が空なら CPU
    try:
        import torch  # type: ignore
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
            return "cpu"
        # CUDA が利用可能なら素直に cuda
        if torch.cuda.is_available():
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


def _infer_safe_h_and_val(df: pd.DataFrame, default_h: int, val_raw: Optional[str]) -> Tuple[int, int]:
    """系列長に応じて h / val_size を安全化"""
    per = df.groupby("unique_id")["ds"].count()
    mlen = int(per.min())
    H = max(1, min(default_h, max(1, mlen - 1)))
    val_size = _parse_val_size(val_raw, df, H)
    # h + val_size が mlen を超えないようにクランプ
    if H + val_size >= mlen:
        # まず val を縮める
        val_size = max(1, min(val_size, max(1, mlen - 1 - 1)))
        # まだダメなら H も縮める
        if H + val_size >= mlen:
            H = max(1, mlen - 1 - val_size)
    H = max(1, H)
    val_size = max(1, val_size)
    return H, val_size


def _sanitize_window_kwargs(df: pd.DataFrame, h: int, val_size: int, kw: Dict[str, Any]) -> Dict[str, Any]:
    """Auto* に渡す window 系パラメータを安全化（短系列での学習失敗を避ける）"""
    per = df.groupby("unique_id")["ds"].count()
    m = int(per.min())
    safe_inp = max(2, min(96, max(2, m - h - val_size)))
    out = dict(kw or {})
    # 強制があれば尊重
    force_inp = os.environ.get("NF_FORCE_INPUT_SIZE")
    force_inf = os.environ.get("NF_FORCE_INFER_INPUT_SIZE")
    start_pad = os.environ.get("NF_START_PADDING", "1").strip().lower() in ("1", "true", "yes", "on")
    if force_inp is not None:
        try:
            out["input_size"] = int(force_inp)
        except Exception:
            out["input_size"] = safe_inp
    else:
        out["input_size"] = min(int(out.get("input_size", safe_inp)), safe_inp)
    if force_inf is not None:
        try:
            out["inference_input_size"] = int(force_inf)
        except Exception:
            out["inference_input_size"] = -h
    else:
        # Auto* 既定の負値指定（-h）が極短で落ちることがあるため保険
        cur_iis = out.get("inference_input_size", -h)
        try:
            ci = int(cur_iis)
        except Exception:
            ci = -h
        if ci <= 0:
            # 明示長で安全側に寄せる
            out["inference_input_size"] = max(1, out["input_size"])
    if start_pad:
        out["start_padding_enabled"] = True
    return out


def _fallback_predict_last_value(df: pd.DataFrame, freq: str, h: int, run_dir: Path) -> Path:
    """学習できなかった場合の保険：系列ごとに最後の y を h ステップ先まで保持"""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "pred.csv"
    rows: List[Tuple[str, str, float]] = []
    for uid, g in df.groupby("unique_id"):
        g = g.sort_values("ds")
        last_y = float(g["y"].iloc[-1])
        last_ds = pd.to_datetime(g["ds"].iloc[-1])
        offset = pd.tseries.frequencies.to_offset(freq or "D")
        for i in range(1, h + 1):
            ds_i = (last_ds + i * offset).date().isoformat()
            rows.append((str(uid), ds_i, last_y))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "ds", "yhat"])
        w.writerows(rows)
    return out


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
    auto_mod = __import__("neuralforecast.auto", fromlist=["*"])
    AutoCls = getattr(auto_mod, model_name)  # AutoRNN / AutoLSTM / ...
    df2, freq = prepare_df_for_nf(df)
    model = AutoCls(h=int(h), backend=str(backend), num_samples=int(num_samples))
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
    save_model: Optional[bool] = None
    overwrite_model: bool = False
    val_size: Optional[str] = None
    # 追加:
    auto_model: str = "AutoRNN"     # NF_AUTO_MODEL で上書き可能
    backend: str = "optuna"         # NF_BACKEND で上書き可能（現状 optuna 前提）


# =========================================================
# Main runner
# =========================================================
def run_neuralforecast_auto(args) -> Dict[str, Any]:
    """
    - gpu_smoke: ダミー4行の pred.csv を出力（学習なし）
    - 通常: Auto* で学習→保存（任意）→予測→pred.csv
      ※ Auto* は h を内部保持するため、推論は nf.predict()（引数なし）を使う。
    """
    # argparse.Namespace / dict / dataclass いずれでも受けられるよう吸収
    def _get(k, default=None):
        if hasattr(args, k) or hasattr(args, k.replace("-", "_")):
            return getattr(args, k, getattr(args, k.replace("-", "_"), default))
        return args.get(k, default) if isinstance(args, dict) else default

    opts = Options(
        data_csv=_get("data_csv"),
        num_samples=int(_get("num_samples", 1) or 1),
        gpu_smoke=bool(_get("gpu_smoke", False)),
        dry_run=bool(_get("dry_run", False)),
        save_model=_get("save_model", None),
        overwrite_model=bool(_get("overwrite_model", False)),
        val_size=_get("val_size", None),
        auto_model=os.environ.get("NF_AUTO_MODEL", _get("auto_model", "AutoRNN")),
        backend=os.environ.get("NF_BACKEND", _get("backend", "optuna")),
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
            "planned_models": [opts.auto_model],
            "hpo_num_samples": int(opts.num_samples),
            "dry_run": True,
            "rows": None,
        }
        _print_json_line(payload)
        return payload

    # ランID/保存先
    rid = _rand_id(8)
    run_dir = base / "runs" / f"{opts.auto_model}__{opts.backend}__h-24__num_samples-{opts.num_samples}__{rid}"
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
            "planned_models": [opts.auto_model],
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

    # データ読み込み & 整形
    df = pd.read_csv(data_csv)
    df2, freq = prepare_df_for_nf(df)

    # horizon / val_size を安全化
    DEFAULT_H = 24
    H, val_size = _infer_safe_h_and_val(df2, DEFAULT_H, opts.val_size or os.environ.get("NF_VAL_SIZE", "h"))

    # Auto* の動的 import
    auto_mod = __import__("neuralforecast.auto", fromlist=["*"])
    try:
        AutoCls = getattr(auto_mod, opts.auto_model)
    except AttributeError:
        raise NotSupportedError(f"Unknown Auto model: {opts.auto_model}")

    # 初期 kwargs（Auto 側に渡す。短系列保険は後段で追加）
    model_kwargs: Dict[str, Any] = dict(h=H, backend=opts.backend, num_samples=int(opts.num_samples))

    # 学習
    from neuralforecast.core import NeuralForecast  # type: ignore
    nf = None
    model_dir_full: Optional[Path] = None
    pred_path: Optional[Path] = None
    rows = int(len(df2))
    ok = False
    error_msg = ""

    def _fit_with(kwargs: Dict[str, Any]) -> None:
        nonlocal nf
        m = AutoCls(**kwargs)
        nf = NeuralForecast(models=[m], freq=freq)
        nf.fit(df=df2, val_size=int(val_size))

    try:
        # 1st try（素直な学習）
        _fit_with(model_kwargs)
        ok = True
    except Exception as e1:
        # よくある短系列崩れ（input.numel==0 / too short / etc）は安全パラメータでリトライ
        msg = str(e1)
        error_msg = msg
        trigger_retry = any(s in msg for s in [
            "Time series is too short", "input.numel() == 0", "input.numel() == 0.",
            "Expected reduction dim", "window", "inference_input_size"
        ])
        if trigger_retry:
            safe_kwargs = _sanitize_window_kwargs(df2, H, val_size, model_kwargs)
            try:
                _fit_with(safe_kwargs)
                model_kwargs = safe_kwargs  # 実際に使ったものを残す
                ok = True
            except Exception as e2:
                error_msg = f"{msg} ; retry_failed: {e2}"

    if ok and nf is not None:
        # 保存（予測前に）
        if bool(opts.save_model):
            tokens = f"backend-{opts.backend}__h-{H}__num_samples-{opts.num_samples}__val_size-{val_size}"
            model_dir_full = base / "models_full" / f"{opts.auto_model}__{opts.backend}__{tokens}__{_rand_id(6)}"
            model_dir_full.mkdir(parents=True, exist_ok=True)
            try:
                nf.save(path=str(model_dir_full), overwrite=overwrite, save_dataset=True)
            except Exception:
                # save_dataset がダメなケース用に最小保存へフォールバック
                nf.save(path=str(model_dir_full), overwrite=overwrite, save_dataset=False)
            with open(model_dir_full / "kwargs.json", "w", encoding="utf-8") as f:
                json.dump(model_kwargs, f, ensure_ascii=False, indent=2, sort_keys=True)
            with open(model_dir_full / "meta.json", "w", encoding="utf-8") as f:
                meta = {
                    "auto_model": opts.auto_model,
                    "backend": str(opts.backend),
                    "freq": str(freq),
                    "n_series": int(df2["unique_id"].nunique()),
                    "rows": rows,
                    "ds_min": str(df2["ds"].min()),
                    "ds_max": str(df2["ds"].max()),
                    "run_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "data_csv": str(Path(data_csv).resolve()),
                    "val_size": int(val_size),
                    "h": int(H),
                }
                json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

        # 予測（Auto* は h 内部保持 → 引数なし）
        yhat = nf.predict()
        pred_path = run_dir / "pred.csv"
        yhat.to_csv(pred_path, index=False)

        payload = {
            "event": "nf_auto_run",
            "status": "ok",
            "ts_ms": _now_ms(),
            "data_csv": str(Path(data_csv).resolve()),
            "planned_models": [opts.auto_model],
            "hpo_num_samples": int(opts.num_samples),
            "dry_run": False,
            "rows": rows,
            "run_dir": str(run_dir.resolve()),
            "pred_csv": str(pred_path.resolve()),
            "device": device,
        }
        _print_json_line(payload)
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

    # === 学習に失敗：フォールバック予測を確実に出す ===========================
    pred_path = _fallback_predict_last_value(df2, freq=freq, h=H, run_dir=run_dir)
    payload = {
        "event": "nf_fallback_pred",
        "status": "ok",
        "ts_ms": _now_ms(),
        "reason": error_msg or "unknown",
        "h": int(H),
        "pred_csv": str(pred_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "device": device,
        "planned_models": [opts.auto_model],
    }
    _print_json_line(payload)
    # 互換サマリー
    _print_json_line({
        "status": "ok",
        "dry_run": False,
        "data_csv": str(Path(data_csv).resolve()),
        "rows": rows,
        "planned_models": [opts.auto_model],
        "hpo_num_samples": int(opts.num_samples),
        "run_dir": str(run_dir.resolve()),
        "pred_csv": str(pred_path.resolve()),
        "device": device,
    })
    return payload
