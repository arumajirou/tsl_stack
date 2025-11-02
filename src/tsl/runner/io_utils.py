# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _sanitize_token(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.\\-]+", "", s)
    return s or "x"


def _kwargs_preview_tokens(kw: Dict[str, Any]) -> str:
    keys_order = [
        "backend", "h", "val_size", "num_samples", "max_steps",
        "scaler_type", "local_scaler_type", "early_stop_patience_steps",
        "search_alg", "search_algorithm", "sampler",
    ]
    picks = []
    for k in keys_order:
        if k in kw:
            v = kw[k]
            if isinstance(v, (list, tuple, set, dict)):
                v = str(type(v).__name__)
            picks.append(f"{_sanitize_token(k)}-{_sanitize_token(str(v))}")
    tok = "__".join(picks) if picks else "default"
    return tok[:120]


def _kwargs_hash(kw: Dict[str, Any]) -> str:
    raw = json.dumps(_jsonable(kw), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)


def build_run_dir(root: Path, auto_name: str, backend: str, kw: Dict[str, Any], maxlen: int) -> Path:
    tokens = _kwargs_preview_tokens(kw)
    hh = _kwargs_hash(kw)
    dirname = f"{auto_name}__{backend}__{tokens}__{hh}"
    if len(dirname) > maxlen:
        dirname = dirname[:maxlen]
    return root / "runs" / dirname


def build_model_dir(root: Path, auto_name: str, backend: str, kw: Dict[str, Any]) -> Path:
    hh = _kwargs_hash(kw)
    dirname = f"{auto_name}__{backend}__full__{hh}"
    return root / "models_full" / dirname


def save_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def pretty_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(p.resolve())
