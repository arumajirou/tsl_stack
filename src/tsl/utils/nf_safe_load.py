# -*- coding: utf-8 -*-
"""
NeuralForecast のチェックポイントを PyTorch 2.6+ で安全にロードするヘルパー。
- まず torch.serialization.add_safe_globals(...) で、NF が内部で使うクラス類を許可リストに追加
- それでも失敗した場合は、最後の手段として weights_only=False でロード（自分で保存した信頼できるファイル限定）

使い方:
    from tsl.utils.nf_safe_load import load_neuralforecast
    nf = load_neuralforecast("nf_auto_runs/models_full/....")
    yhat = nf.predict()
"""

from __future__ import annotations
import importlib
import inspect
from typing import Iterable, List, Optional

from torch.serialization import add_safe_globals
from neuralforecast.core import NeuralForecast


def _collect_classes(module_names: Iterable[str]) -> List[type]:
    classes: List[type] = []
    for modname in module_names:
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in vars(m).items():
            if inspect.isclass(obj):
                classes.append(obj)
    return classes


def register_nf_safe_globals() -> int:
    """NeuralForecast が pickle 内で参照する可能性のあるクラスを allowlist に追加する。
    戻り値は登録したクラス数。
    """
    # 代表的に必要になるもの（losses / distributions / lightning の AttributeDict）
    modules = [
        "neuralforecast.losses.pytorch",
        "neuralforecast.losses.pytorch.distributions",
        # 互換のため（環境によりモジュール名が異なる場合）
        "neuralforecast.losses",
    ]
    safe_types: List[type] = _collect_classes(modules)

    # Lightning の AttributeDict
    try:
        from lightning_fabric.utilities.data import AttributeDict  # type: ignore
        safe_types.append(AttributeDict)
    except Exception:
        pass

    # 重複を除いて登録
    uniq = []
    seen = set()
    for t in safe_types:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    if uniq:
        add_safe_globals(uniq)
    return len(uniq)


def load_neuralforecast(
    path: str,
    *,
    prefer_safe: bool = True,
    map_location: Optional[str] = None,
):
    """NeuralForecast の保存ディレクトリからモデルをロード。
    prefer_safe=True: まず safe_globals を登録して通常ロードを試す。ダメなら weights_only=False に自動フォールバック。
    prefer_safe=False: いきなり weights_only=False でロード（“自分で保存した信頼できる ckpt”のみで使用推奨）。
    """
    if not prefer_safe:
        # 旧挙動に戻してロード（安全上の注意: 自分で保存した ckpt に限定）
        return NeuralForecast.load(path=path, map_location=map_location, weights_only=False)

    # 1) allowlist でできる限り安全に読み込み
    try:
        _ = register_nf_safe_globals()
        return NeuralForecast.load(path=path, map_location=map_location)
    except Exception as e1:
        # 2) それでも失敗したら、信頼できる ckpt 前提で weights_only=False へフォールバック
        try:
            return NeuralForecast.load(path=path, map_location=map_location, weights_only=False)
        except Exception as e2:
            # どちらも失敗した場合は元の例外情報をまとめて投げ直す
            raise RuntimeError(
                "NeuralForecast checkpoint のロードに失敗しました。\n"
                f"[safe_globals 登録後の例外] {type(e1).__name__}: {e1}\n"
                f"[weights_only=False の例外]  {type(e2).__name__}: {e2}\n"
                "→ ckpt の破損やバージョン不整合の可能性を確認してください。"
            ) from e2
