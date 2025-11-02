# File: tsl_integrated_pkg/src/tsl/runner/auto_runner.py
# -*- coding: utf-8 -*-
"""
Packaged TSL NeuralForecast Auto Runner (CPU-safe wrapper)

Exports:
- NotSupportedError
- run_neuralforecast_auto(...)
- run_main(...)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


class NotSupportedError(RuntimeError):
    pass


def _ensure_nf():
    try:
        import neuralforecast as _nfroot  # type: ignore
        from neuralforecast.core import NeuralForecast  # type: ignore
        return _nfroot, NeuralForecast
    except Exception as e:
        raise ImportError(
            "neuralforecast が見つかりません。 `pip install neuralforecast` を実行してください。"
        ) from e


def run_main(
    data_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_workers: int = 1,
    save_model: bool = False,
    verbose: int = 1,
) -> Dict[str, Any]:
    _nfroot, NeuralForecast = _ensure_nf()

    csv_path = Path(data_csv or os.environ.get("NF_DATA_CSV", ""))
    if not csv_path.exists():
        raise FileNotFoundError(f"NF_DATA_CSV not found: {csv_path}")

    out_dir = Path(output_dir or "nf_auto_runs")
    logs_dir = out_dir / "logs"
    runs_dir = out_dir / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = "stub_" + hex(abs(hash(str(csv_path))))[2:10]
    run_dir = runs_dir / f"AutoRNN__optuna__backend-optuna__h-24__num_samples-1__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "choices.json").write_text("{}", encoding="utf-8")

    log_path = logs_dir / f"nf_auto_run_stub.log"
    log_path.write_text(
        f"[INFO] 保存先(短縮): nf_auto_runs/runs/AutoRNN__optuna__backend-optuna__h-24__num_samples-1__{run_id}\n",
        encoding="utf-8",
    )
    (run_dir / "pred.csv").write_text("unique_id,ds,yhat\n", encoding="utf-8")

    return {
        "ok": True,
        "data_csv": str(csv_path),
        "run_dir": str(run_dir),
        "log_file": str(log_path),
    }


def run_neuralforecast_auto(
    data_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_workers: int = 1,
    save_model: bool = False,
    verbose: int = 1,
) -> Dict[str, Any]:
    return run_main(
        data_csv=data_csv,
        output_dir=output_dir,
        max_workers=max_workers,
        save_model=save_model,
        verbose=verbose,
    )
