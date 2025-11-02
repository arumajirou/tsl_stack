# sitecustomize.py
# - src レイアウトを最優先
# - もし sys.path に tsl_integrated_pkg/src が残っていたら取り除く

import os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
src = repo_root / "src"
legacy = repo_root / "tsl_integrated_pkg" / "src"

# 先頭に src をセット
src_str = str(src)
if src.exists():
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)

# 旧パスは除去
legacy_str = str(legacy)
try:
    while legacy_str in sys.path:
        sys.path.remove(legacy_str)
except Exception:
    pass

# 便利: TSL_BASE の既定
os.environ.setdefault("TSL_BASE", str(repo_root / "nf_auto_runs"))
