# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import inspect
import json
import types
import os

def signature_params_defaults(callable_obj):
    try:
        sig = inspect.signature(callable_obj)
        return sig.parameters, {k: v.default for k, v in sig.parameters.items()}
    except Exception:
        return {}, {}

def _is_installed(modname: str) -> bool:
    try:
        __import__(modname); return True
    except Exception:
        return False

HAS_OPTUNA = _is_installed("optuna")
HAS_RAY    = _is_installed("ray")

def build_loss_instance(loss_choice: str | None):
    # "auto"/None -> MSE()
    if loss_choice is None or str(loss_choice).strip().lower() in ("", "auto", "none"):
        try:
            from neuralforecast.losses.pytorch import MSE
            return MSE()
        except Exception:
            return None
    # 具体的クラス名
    try:
        from neuralforecast import losses as _losses_pkg
        from neuralforecast.losses import pytorch as pt
        if hasattr(pt, loss_choice):
            cls = getattr(pt, loss_choice)
            if inspect.isclass(cls):
                return cls()
        # distribution 系
        if loss_choice.startswith("dist:"):
            from neuralforecast.losses.pytorch import DistributionLoss
            dist_name = loss_choice.split(":", 1)[1]
            # 検索
            for modp in [
                "neuralforecast.losses.pytorch",
                "neuralforecast.losses.pytorch.distributions",
                "neuralforecast.losses.pytorch.distribution",
            ]:
                try:
                    m = __import__(modp, fromlist=["*"])
                    if hasattr(m, dist_name) and inspect.isclass(getattr(m, dist_name)):
                        return DistributionLoss(distribution=getattr(m, dist_name))
                except Exception:
                    pass
    except Exception:
        pass
    # fallback
    try:
        from neuralforecast.losses.pytorch import MSE
        return MSE()
    except Exception:
        return None


def safe_kwargs_for_auto(
    auto_cls,
    *,
    backend: str,
    h_val: int,
    df_n_series: int,
    futr_cols: List[str],
    hist_cols: List[str],
    stat_cols: List[str],
    chosen_loss: Any = None,
    chosen_scaler: Optional[str] = None,
    chosen_early_stop: Optional[int] = None,
    chosen_search_alg: Optional[str] = None,
    chosen_val_size: Optional[int] = None,
    random_state: int = 2077,
    num_samples: int = 1,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Auto* クラスへ渡す kwargs を安全に構築。存在しない引数は渡さない。"""
    params, defaults = signature_params_defaults(auto_cls)
    pnames = set(params.keys())
    kw: Dict[str, Any] = {}

    # --- backend / search ---
    if "backend" in pnames: kw["backend"] = backend
    if chosen_search_alg is not None:
        for cand in ("search_alg","search_algorithm","sampler","searcher"):
            if cand in pnames:
                kw[cand] = chosen_search_alg; break

    # --- 代表的パラメータ
    if "h" in pnames:
        kw["h"] = int(h_val)
    for cand in ("random_state","seed","random_seed"):
        if cand in pnames:
            kw[cand] = int(random_state); break
    for cand in ("num_samples","n_trials","n_samples"):
        if cand in pnames:
            kw[cand] = int(num_samples); break
    for cand in ("max_steps","max_epochs","max_train_steps","max_iters"):
        if cand in pnames:
            kw[cand] = int(max_steps); break
    for cand in ("val_size","valid_size","validation_size"):
        if cand in pnames:
            kw[cand] = int(chosen_val_size if chosen_val_size is not None else h_val); break

    # --- loss/scaler
    if ("loss" in pnames) and (chosen_loss is not None):
        kw["loss"] = chosen_loss
        if "valid_loss" in pnames:
            kw["valid_loss"] = chosen_loss
    scaler_param = next((c for c in ("scaler_type","local_scaler_type") if c in pnames), None)
    if scaler_param and chosen_scaler is not None:
        kw[scaler_param] = chosen_scaler

    # --- early stop
    if "early_stop_patience_steps" in pnames and chosen_early_stop is not None:
        kw["early_stop_patience_steps"] = int(chosen_early_stop)

    # --- exog list（存在する場合だけ）
    if "futr_exog_list" in pnames and futr_cols:
        kw["futr_exog_list"] = futr_cols
    if "hist_exog_list" in pnames and hist_cols:
        kw["hist_exog_list"] = hist_cols
    if "stat_exog_list" in pnames and stat_cols:
        kw["stat_exog_list"] = stat_cols

    if "n_series" in pnames:
        kw["n_series"] = int(df_n_series)

    # verbose/verbosity
    for cand in ("verbose","verbosity"):
        if cand in pnames:
            kw[cand] = True

    return kw
