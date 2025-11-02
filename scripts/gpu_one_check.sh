#!/usr/bin/env bash

# === Config ===
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONPATH

# GPUを必ず可視化（必要に応じて 0 を変更）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIAG_DIR="nf_auto_runs/_diag"
mkdir -p "$DIAG_DIR"

# _diag の所有者が root になっていると tee で Permission denied になるためガード
# sudo を使わず、失敗しても続行（所有者が自分なら chown は成功する）
if command -v id >/dev/null 2>&1; then
  chown "$(id -u)":"$(id -g)" "$DIAG_DIR" 2>/dev/null || true
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${DIAG_DIR}/run_${TS}.stdout"
JSON="${DIAG_DIR}/run_${TS}.json"

echo "== GPU smoke one =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi | head -n 10 || true

# 1) GPUスモーク: 全出力をまずログへ保存（tail禁止）
python -m tsl.cli.tsl run-auto \
  --data-csv "$(pwd)/gpu_smoke.csv" \
  --gpu-smoke \
  --num-samples 1 | tee "$LOG"

# 2) ログの「最後に現れた pred_csv を含む JSON」を安全に抽出
python - "$LOG" "$JSON" <<'PY'
import json, pathlib, sys
log = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore').splitlines()
sel = None
for line in reversed(log):
    line = line.strip()
    if not (line.startswith('{') and line.endswith('}')):
        continue
    try:
        j = json.loads(line)
    except Exception:
        continue
    if 'pred_csv' in j and 'device' in j:
        sel = j
        break

if not sel:
    print("ERROR: JSON with pred_csv/device not found in log", file=sys.stderr)
    sys.exit(1)

pathlib.Path(sys.argv[2]).write_text(json.dumps(sel, ensure_ascii=False), encoding='utf-8')
print(f"device: {sel.get('device')}, pred_csv: {sel.get('pred_csv')}")
PY

# 3) 生成物の実在チェック
python - "$JSON" <<'PY'
import json, pathlib, sys
j = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
pred = pathlib.Path(j['pred_csv'])
print("pred exists:", pred.exists(), pred)
if not pred.exists():
    sys.exit(2)
print("✅ GPU 単発スモーク OK")
PY
