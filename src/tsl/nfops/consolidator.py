\
from __future__ import annotations
import os, shutil, logging, time
from pathlib import Path
from typing import Optional, Tuple, Dict

class ArtifactConsolidator:
    """Move pred.csv and logs into a canonical MODEL_DIR. Idempotent: yes. Thread-safe: no (fs)."""
    def __init__(self, model_dir: Path, start_ts: float):
        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.model_dir / "logs"; self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir   = self.model_dir / "lightning_logs"; self.tb_dir.mkdir(parents=True, exist_ok=True)
        self._start_ts = float(start_ts)

    def attach_combo_file_handler(self, logger: logging.Logger) -> logging.Handler:
        fh = logging.FileHandler(str(self.logs_dir / "combo.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(fh)
        return fh

    @staticmethod
    def detach_handler(logger: logging.Logger, handler: Optional[logging.Handler]):
        if handler is None: return
        try: logger.removeHandler(handler)
        finally:
            try: handler.flush()
            finally:
                try: handler.close()
                except Exception: pass

    def move_pred_and_configs(self, combo_dir: Path) -> Tuple[Path, Dict[str,str]]:
        combo_dir = Path(combo_dir)
        moved = {}
        for name in ["pred.csv","kwargs.json","meta.json","choices.json","error.txt"]:
            src = combo_dir / name
            if src.exists():
                dst = self.model_dir / name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                moved[name] = str(dst)
        return (self.model_dir / "pred.csv"), moved

    def move_lightning_logs_since_start(self):
        root = Path.cwd() / "lightning_logs"
        if not root.exists(): return
        for vdir in sorted([p for p in root.glob("version_*") if p.is_dir()]):
            try:
                mtime = float(vdir.stat().st_mtime)
                if mtime >= self._start_ts - 1.0:
                    dst = self.tb_dir / vdir.name
                    if dst.exists():
                        # avoid overwrite
                        i=2
                        while (self.tb_dir / f"{vdir.name}__{i}").exists(): i+=1
                        dst = self.tb_dir / f"{vdir.name}__{i}"
                    shutil.move(str(vdir), str(dst))
            except Exception:
                logging.getLogger("tsl.nfops").warning("Failed to move lightning log: %s", vdir, exc_info=False)
