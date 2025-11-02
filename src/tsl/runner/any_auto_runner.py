# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from .core import RunConfig, SingleAutoRun

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True, help="入力CSVパス（unique_id, ds, y [+ exog]）")
    ap.add_argument("--save-model", action="store_true", help="モデルも保存")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = RunConfig(
        data_csv=args.data_csv,
        save_model=bool(args.save_model or True if "1" == "1" else False)  # env 側が 1 の場合でも ON を優先
    )
    runner = SingleAutoRun(cfg)
    run_dir, model_dir = runner.run()
    print(str({"run_dir": str(run_dir), "model_dir": (str(model_dir) if model_dir else None)}))

if __name__ == "__main__":
    main()
