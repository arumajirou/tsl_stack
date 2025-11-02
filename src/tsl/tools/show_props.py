# File: src/tsl/tools/show_props.py
# -*- coding: utf-8 -*-
"""
Show system and project properties as JSON to stdout.

Usage:
  python -m tsl.tools.show_props | python -m json.tool
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def _maybe_import_version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not-installed"


def _torch_gpu() -> Dict[str, Any]:
    meta = {"available": False, "count": 0, "devices": []}
    try:
        import torch  # type: ignore
        meta["available"] = bool(torch.cuda.is_available())
        if meta["available"]:
            n = torch.cuda.device_count()
            meta["count"] = n
            meta["devices"] = [torch.cuda.get_device_name(i) for i in range(n)]
    except Exception:
        pass
    return meta


def collect() -> Dict[str, Any]:
    cwd = str(Path.cwd())
    repo_root = str(Path(__file__).resolve().parents[3])
    env_keys = [k for k in os.environ.keys() if k.startswith(("TSL_", "NF_", "MLFLOW_", "CUDA_VISIBLE_DEVICES"))]
    env_sample = {k: os.environ.get(k, "") for k in sorted(env_keys)}
    nf_base = Path(repo_root) / "nf_auto_runs"
    return {
        "system": {"platform": platform.platform(), "python": sys.version.split()[0], "executable": sys.executable},
        "paths": {"cwd": cwd, "repo_root": repo_root, "pythonpath_head": sys.path[:5], "nf_auto_runs": str(nf_base)},
        "env": env_sample,
        "packages": {
            "torch": _maybe_import_version("torch"),
            "neuralforecast": _maybe_import_version("neuralforecast"),
            "pytorch_lightning": _maybe_import_version("pytorch_lightning"),
            "optuna": _maybe_import_version("optuna"),
            "ray": _maybe_import_version("ray"),
            "pandas": _maybe_import_version("pandas"),
        },
        "gpu": _torch_gpu(),
        "db": {"TSL_DB_URL_set": bool(os.environ.get("TSL_DB_URL"))},
    }


def main() -> None:
    print(json.dumps(collect(), ensure_ascii=False))


if __name__ == "__main__":
    main()
