# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
import os
import time
import hashlib

import pandas as pd

from . import preprocess as pp
from . import kwargs_builder as kb
from .nf_compat import safe_nf_predict, set_torch_matmul_precision_if_available


# ---- ユーティリティ ----------------------------------------------------------

def _hash8_from_obj(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str,int,float,bool)) or obj is None:
        return obj
    if isinstance(obj, (list,tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p


# ---- 実行単位 ---------------------------------------------------------------

@dataclass
class RunConfig:
    data_csv: str
    output_dir: Path = Path("./nf_auto_runs")
    auto_model_name: str = os.getenv("NF_AUTO_MODEL", "AutoLSTM")
    backend: str = os.getenv("NF_BACKEND", "optuna")
    val_size_env: str = os.getenv("NF_VAL_SIZE", "h")
    num_samples: int = int(os.getenv("NF_TRIAL_NUM_SAMPLES", "1"))
    max_steps: int = int(os.getenv("NF_TRIAL_MAX_STEPS", "50"))
    random_state: int = int(os.getenv("NF_RANDOM_STATE", "2077"))
    save_model: bool = (os.getenv("NF_SAVE_MODEL", "1").strip().lower() in ("1","true","yes"))
    overwrite_model: bool = (os.getenv("NF_OVERWRITE_MODEL", "0").strip().lower() in ("1","true","yes"))
    df_strict_cols: bool = (os.getenv("NF_DF_STRICT_COLS", "1").strip().lower() in ("1","true","yes"))
    impute_policy: str = os.getenv("NF_IMPUTE_POLICY", "ffill_bfill_zero")  # drop_rows/drop_cols など
    dir_tokens_maxlen: int = int(os.getenv("NF_DIR_TOKENS_MAXLEN", "200"))


class SingleAutoRun:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        _ensure_dir(cfg.output_dir / "runs")
        _ensure_dir(cfg.output_dir / "models_full")
        _ensure_dir(cfg.output_dir / "logs")
        set_torch_matmul_precision_if_available()

    # -- ディレクトリ命名（短縮 / フル保存） --
    def _kwargs_hash(self, kw: Dict[str, Any]) -> str:
        return _hash8_from_obj(kw)

    def _build_run_dir(self, kw: Dict[str, Any]) -> Path:
        tok = "__".join(
            f"{k}-{str(v)[:24]}" for k, v in sorted(kw.items()) if k not in {"callbacks"}
        )
        tok = tok[:120]
        dirname = f"{self.cfg.auto_model_name}__{self.cfg.backend}__{tok}__{self._kwargs_hash(kw)}"
        if len(dirname) > 160:
            dirname = dirname[:160]
        return self.cfg.output_dir / "runs" / dirname

    def _build_model_dir(self, kw: Dict[str, Any]) -> Path:
        # フル kwargs 版
        parts = [f"{k}-{str(v)[:24]}" for k, v in sorted(kw.items())]
        tok = "__".join(parts)
        if len(tok) > self.cfg.dir_tokens_maxlen:
            tok = tok[:self.cfg.dir_tokens_maxlen]
        dirname = f"{self.cfg.auto_model_name}__{self.cfg.backend}__{tok}__{self._kwargs_hash(kw)}"
        if len(dirname) > 240:
            dirname = dirname[:240]
        return self.cfg.output_dir / "models_full" / dirname

    # -- 実行本体 --
    def run(self) -> Tuple[Path, Optional[Path]]:
        import importlib
        from neuralforecast.core import NeuralForecast

        # 1) データ
        df = pp.load_csv_and_standardize(self.cfg.data_csv)
        df = pp.encode_exogs_by_prefix(df)

        freq, _ = pp.infer_global_freq(df)
        h_val = pp.infer_h(df, int(os.getenv("NF_DEFAULT_H", "24")))
        val_size = pp.parse_val_size(df, h_val)

        # 2) Auto クラスの解決
        auto_mod = importlib.import_module("neuralforecast.auto")
        auto_cls = getattr(auto_mod, self.cfg.auto_model_name)  # 例: AutoLSTM
        base_n_series = int(df["unique_id"].nunique())

        # 3) exog 抽出
        futr_cols, hist_cols, stat_cols = pp.split_exog_by_prefix(df)

        # 4) loss/scaler/early_stop/search
        loss_choice = os.getenv("NF_LOSS", "auto")
        loss_obj = kb.build_loss_instance(loss_choice)
        scaler_choice = os.getenv("NF_SCALER") or None
        early_stop_choice_env = os.getenv("NF_EARLY_STOP", "auto")
        if early_stop_choice_env == "auto":
            early_stop = max(5, h_val // 2)
        elif early_stop_choice_env in ("disabled","off","none"):
            early_stop = None
        else:
            try: early_stop = max(1, int(float(early_stop_choice_env)))
            except Exception: early_stop = max(5, h_val // 2)

        search_alg = os.getenv("NF_SEARCH_ALG") or None

        # 5) Auto* kwargs
        kw = kb.safe_kwargs_for_auto(
            auto_cls,
            backend=self.cfg.backend,
            h_val=h_val,
            df_n_series=base_n_series,
            futr_cols=futr_cols, hist_cols=hist_cols, stat_cols=stat_cols,
            chosen_loss=loss_obj,
            chosen_scaler=scaler_choice,
            chosen_early_stop=early_stop,
            chosen_search_alg=search_alg,
            chosen_val_size=val_size,
            random_state=self.cfg.random_state,
            num_samples=self.cfg.num_samples,
            max_steps=self.cfg.max_steps,
        )

        # 6) 列制限 + 欠損補完（要求に応じて）
        used_cols = ["unique_id", "ds", "y"]
        used_cols += kw.get("hist_exog_list", []) + kw.get("futr_exog_list", []) + kw.get("stat_exog_list", [])
        used_cols = list(dict.fromkeys([c for c in used_cols if c in df.columns]))

        if self.cfg.df_strict_cols:
            df = df[used_cols].copy()

        df, impute_info = pp.impute_or_drop_nan(df, used_exog_cols=[c for c in used_cols if c not in ("unique_id","ds","y")],
                                                policy=self.cfg.impute_policy)
        # 最終チェック
        pp.assert_no_missing(df, [c for c in used_cols if c not in ("unique_id","ds","y")], "impute")

        # 7) 出力先
        run_dir = _ensure_dir(self._build_run_dir({"h": h_val, "num_samples": self.cfg.num_samples}))
        model_dir = self._build_model_dir({"h": h_val, "num_samples": self.cfg.num_samples, "val_size": val_size})

        # 8) 情報の保存（df 列/欠損補完情報/choices など）
        with open(run_dir / "df_columns.json", "w", encoding="utf-8") as f:
            json.dump({"used": used_cols, "all": list(df.columns)}, f, ensure_ascii=False, indent=2)
        with open(run_dir / "choices.json", "w", encoding="utf-8") as f:
            json.dump({
                "auto_model": self.cfg.auto_model_name,
                "backend": self.cfg.backend,
                "loss": os.getenv("NF_LOSS", "auto"),
                "scaler": scaler_choice,
                "early_stop": early_stop_choice_env,
                "search_alg": search_alg,
                "val_size": val_size,
                "impute": impute_info,
            }, f, ensure_ascii=False, indent=2)

        # 9) 学習・予測・保存
        auto_inst = auto_cls(**kw)
        nf = NeuralForecast(models=[auto_inst], freq=freq)

        # fit
        nf.fit(df=df, val_size=int(val_size))

        # save (任意)
        saved_model_dir = None
        if self.cfg.save_model:
            saved_model_dir = _ensure_dir(model_dir)
            nf.save(path=str(saved_model_dir), overwrite=self.cfg.overwrite_model)
            # meta/kwargs 追記
            with open(saved_model_dir / "kwargs.json", "w", encoding="utf-8") as f:
                json.dump(_jsonable(kw), f, ensure_ascii=False, indent=2)
            with open(saved_model_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "auto_model": self.cfg.auto_model_name,
                    "backend": self.cfg.backend,
                    "data_csv": self.cfg.data_csv,
                    "freq": freq,
                    "h": h_val,
                    "val_size": val_size,
                    "n_series": int(df["unique_id"].nunique()),
                    "rows": int(len(df)),
                    "ds_min": str(df["ds"].min()),
                    "ds_max": str(df["ds"].max()),
                    "run_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, ensure_ascii=False, indent=2)

        # predict（安全フォールバック付き）
        yhat = safe_nf_predict(nf, h=h_val)

        out_csv = run_dir / "pred.csv"
        yhat.to_csv(out_csv, index=False)

        return run_dir, (saved_model_dir if self.cfg.save_model else None)
