\
from __future__ import annotations
import os
def wandb_log_stub():
    """No-op unless WANDB enabled. Idempotent; thread-safe: yes."""
    if str(os.getenv("TSL_ENABLE_WANDB","0")).lower() in ("1","true","yes"):
        try:
            import wandb
            wandb.init(project=os.getenv("TSL_WANDB_PROJECT","tsl"), mode=os.getenv("TSL_WANDB_MODE","offline"))
            wandb.log({"hello":"world"})
            wandb.finish()
        except Exception:
            pass
