#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/6] 旧 tsl_integrated_pkg をバックアップして削除（存在すれば）"
if [[ -d tsl_integrated_pkg ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  mv tsl_integrated_pkg "_backup_tsl_integrated_pkg_${TS}"
fi

echo "[2/6] キャッシュ削除"
find "$REPO_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -rf "$REPO_ROOT"/src/*.egg-info || true

echo "[3/6] 旧 editable をアンインストール（入っていなければOK）"
python -m pip uninstall -y tsl-stack || true

echo "[4/6] サイトパッケージの残骸を除去（egg-info / egg-link）"
python - << 'PY'
import site, sys, pathlib, re, os
cands = []
for base in set(site.getsitepackages() + [site.getusersitepackages()]):
    p = pathlib.Path(base)
    if p.exists():
        for q in p.iterdir():
            if re.search(r"^tsl[_-]stack.*\.(egg-info|egg-link)$", q.name):
                cands.append(q)
for q in cands:
    try:
        if q.is_dir():
            os.system(f"rm -rf {q}")
        else:
            q.unlink()
        print(f"removed: {q}")
    except Exception as e:
        print(f"skip: {q} -> {e}", file=sys.stderr)
PY

echo "[5/6] 再インストール（editable）"
python -m pip install -U pip
python -m pip install -e .

echo "[6/6] 動作確認"
python - << 'PY'
import sys, tsl
import tsl.ingest.pipeline as P
import tsl.runner.auto_runner as R
print("ok: import tsl", tsl.__file__)
print("ok: ingest.pipeline at", P.__file__)
print("ok: runner.auto_runner at", R.__file__)
PY

echo "DONE."
