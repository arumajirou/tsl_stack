# -*- coding: utf-8 -*-
"""
nf_compat.py
NeuralForecast のバージョン差異・PyTorch 2.6 の安全ロード仕様変更に吸収レイヤを提供。

公開関数:
- safe_nf_load(path, *, map_location=None, allow_unsafe_env="NF_UNPICKLE_UNSAFE_OK", **kwargs) -> NeuralForecast
- safe_nf_predict(nf_obj, h: int | None = None, **kwargs) -> pandas.DataFrame

使い方:
from tsl.runner.nf_compat import safe_nf_load, safe_nf_predict
nf = safe_nf_load("nf_auto_runs/models_full/...")
yhat = safe_nf_predict(nf, h=24)  # 新APIなら h は無視され、旧APIなら使用される
"""
from __future__ import annotations

from typing import Any, Optional
import inspect

# --- PyTorch 2.6 safety: allowlist 必要なクラスを登録 -------------------------
try:
    from torch.serialization import add_safe_globals, safe_globals  # PyTorch >= 2.6
except Exception:  # 古い PyTorch では存在しない
    add_safe_globals = None
    from contextlib import nullcontext as safe_globals  # type: ignore

def _install_torch_safe_globals() -> None:
    """NF の checkpoint で頻出する型を allowlist に追加。"""
    if add_safe_globals is None:
        return
    allow: list[type] = []
    # lightning_fabric AttributeDict
    try:
        from lightning_fabric.utilities.data import AttributeDict
        allow.append(AttributeDict)  # type: ignore[misc]
    except Exception:
        pass
    # NF の Loss クラス (例: MSE)
    try:
        from neuralforecast.losses.pytorch import MSE
        allow.append(MSE)  # type: ignore[misc]
    except Exception:
        pass
    if allow:
        add_safe_globals(allow)

# --- Load 互換 ---------------------------------------------------------------
def safe_nf_load(path: str, *, map_location: Optional[str] = None,
                 allow_unsafe_env: str = "NF_UNPICKLE_UNSAFE_OK", **kwargs):
    """
    NeuralForecast.load(path) の安全ローダ。
    1) allowlist を登録して weights_only=True のまま読み込みを試行
    2) それでも失敗し、かつ環境変数 allow_unsafe_env=1 のときのみ weights_only=False 再試行
    """
    from neuralforecast.core import NeuralForecast
    import os
    _install_torch_safe_globals()

    # まずは安全に（weights_only 既定 True のまま）
    try:
        with safe_globals([]):  # 明示しておく（古い環境では no-op）
            return NeuralForecast.load(path=path, map_location=map_location, **kwargs)
    except Exception as e:
        # 最終手段: 信頼できる自前のチェックポイントのみ許可
        if os.getenv(allow_unsafe_env, "0").strip().lower() in ("1", "true", "yes"):
            try:
                return NeuralForecast.load(path=path, map_location=map_location, weights_only=False, **kwargs)
            except Exception:
                raise
        raise

# --- Predict 互換 ------------------------------------------------------------
def _method_accepts_kw(name: str, fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    return name in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

def _infer_h_from_models(nf_obj) -> Optional[int]:
    try:
        models = getattr(nf_obj, "models", None)
        if models and len(models) > 0:
            return getattr(models[0], "h", None)
    except Exception:
        pass
    return None

def safe_nf_predict(nf_obj, h: Optional[int] = None, **kwargs):
    """
    NF の API 差異を吸収して予測を実行:
      - 新API系: nf.predict() で次 h ステップ（Auto* で h は内部保持）
      - 旧API系: nf.predict(h=h) / nf.forecast(h=h)
    引数:
      h: 旧APIや forecast 用の保険。新APIでは未使用。
      kwargs: df=..., futr_df=... などを自前で渡したいときに利用可。
    返り値: pandas.DataFrame (columns: unique_id, ds, yhat[, yhat_qX])
    """
    # 優先1: 新API（predict 引数なし）— Auto* ではドキュメント上この形を推奨
    # https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/hyperparameter_tuning.html （nf.predict() を使用）
    pred = getattr(nf_obj, "predict", None)
    if callable(pred):
        # まず h を渡さずに呼ぶ（新API）
        try:
            return pred(**kwargs)
        except TypeError:
            # 旧API: h をキーワードで受け付けるか？
            if h is not None and _method_accepts_kw("h", pred):
                try:
                    return pred(h=h, **kwargs)
                except TypeError:
                    pass
            # 旧API: 位置引数 h のケース（最後の手段）
            if h is not None:
                try:
                    return pred(h, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    pass

    # 優先2: forecast(h=...) を持つ版
    fore = getattr(nf_obj, "forecast", None)
    if callable(fore):
        hh = h if h is not None else _infer_h_from_models(nf_obj)
        if hh is not None:
            try:
                if _method_accepts_kw("h", fore):
                    return fore(h=hh, **kwargs)
                else:
                    return fore(hh, **kwargs)  # type: ignore[arg-type]
            except TypeError:
                pass

    # 優先3: predict(df=...) を明示で使うケース（ユーザが kwargs で df を渡す）
    if callable(pred):
        return pred(**kwargs)

    raise TypeError(
        "safe_nf_predict: predict/forecast の呼び分けに失敗しました。"
        "お使いの NF 版では df=... を渡す必要がある可能性があります。"
    )
