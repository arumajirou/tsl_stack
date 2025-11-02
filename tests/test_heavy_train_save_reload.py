# tests/test_heavy_train_save_reload.py
import os
import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy  # デフォでは実行されない。-m heavy で実行。


def _have_torch_lightning():
    try:
        import torch  # noqa
        import pytorch_lightning as pl  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_torch_lightning(), reason="torch または pytorch_lightning 未インストール")
def test_lightning_train_save_reload_and_make_pred_csv(tmp_path: Path, monkeypatch):
    # 依存 import
    import torch
    import torch.nn as nn
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd

    # ----- 1) おもちゃデータ（線形回帰 y = 2x + 1 + ε） -----
    torch.manual_seed(0)
    n = 256
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2.0 * x + 1.0 + 0.1 * torch.randn_like(x)

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    # ----- 2) シンプルな LightningModule -----
    class TinyReg(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
            self.loss = nn.MSELoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, _batch_idx):
            xb, yb = batch
            pred = self(xb)
            loss = self.loss(pred, yb)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-2)

    # ----- 3) 1回目の学習（短時間） & checkpoint 保存 -----
    run_root = tmp_path / "nf_auto_runs" / "runs" / f"tiny_run_{int(time.time())}"
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt = run_root / "model.ckpt"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=3,  # かなり軽量
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,  # 自前で保存
    )
    m1 = TinyReg()
    trainer.fit(m1, dl)
    trainer.save_checkpoint(str(ckpt))
    assert ckpt.exists(), "checkpoint が保存されていません"

    # ----- 4) チェックポイントからロードして“再学習” -----
    m2 = TinyReg.load_from_checkpoint(str(ckpt))
    trainer2 = pl.Trainer(
        max_epochs=2,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )
    trainer2.fit(m2, dl)

    # ----- 5) 予測生成して pred.csv を保存（CLI ingest が拾えるカラム） -----
    with torch.no_grad():
        xs = torch.linspace(-1.5, 1.5, 30).unsqueeze(1)
        ys = m2(xs)

    # ingest が期待する最小構成: unique_id, ds, y_hat
    base_time = pd.Timestamp.utcnow().normalize()
    df_pred = pd.DataFrame({
        "unique_id": ["tiny"] * len(xs),
        "ds": [base_time + pd.Timedelta(days=i) for i in range(len(xs))],
        "y_hat": ys.squeeze().cpu().numpy().tolist(),
    })
    pred_csv = run_root / "pred.csv"
    df_pred.to_csv(pred_csv, index=False)
    assert pred_csv.exists(), "pred.csv が保存されていません"

    # ついでに “最新行 JSON 形式” を run-dir にも書いておく（後続テストで読む用ではないが記録）
    (run_root / "last.json").write_text(json.dumps({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pred_rows": int(df_pred.shape[0]),
        "run_dir": str(run_root),
        "pred_csv": str(pred_csv),
    }), encoding="utf-8")
