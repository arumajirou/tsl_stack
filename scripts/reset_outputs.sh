#!/usr/bin/env bash
# scripts/reset_outputs.sh
# 出力物の初期化 + __pycache__ / *.py[co] も削除
# 既定: runs/models_full/lightning_logs を空に（logsとpreds.sqliteは保持）
# オプション:
#   --hard       : nf_auto_runs ごと削除して完全初期化（logs も消える）
#   --keep-logs  : logs を必ず保持
#   --reset-db   : preds.sqlite も削除
#   --dry-run    : 実行せず削除対象だけ表示
#   --no-pyc     : __pycache__ / *.py[co] を削除しない（既定は削除）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HARD=0
KEEP_LOGS=0
RESET_DB=0
DRY_RUN=0
CLEAN_PYCACHE=1

for a in "$@"; do
  case "$a" in
    --hard) HARD=1 ;;
    --keep-logs) KEEP_LOGS=1 ;;
    --reset-db) RESET_DB=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --no-pyc) CLEAN_PYCACHE=0 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

NF_DIR="${ROOT_DIR}/nf_auto_runs"
RUNS_DIR="${NF_DIR}/runs"
MODELS_DIR="${NF_DIR}/models_full"
LOGS_DIR="${NF_DIR}/logs"
DIAG_DIR="${NF_DIR}/_diag"
LL_DIR="${ROOT_DIR}/lightning_logs"
DB_PATH="${ROOT_DIR}/preds.sqlite"

case "${ROOT_DIR}" in
  *"/tsl_stack") : ;;
  *"/tsl_stack/"*) : ;;
  *) echo "[ABORT] ROOT_DIR が tsl_stack 配下ではありません: ${ROOT_DIR}"; exit 1 ;;
esac

say() { echo "[reset] $*"; }
run() { if [[ "$DRY_RUN" == "1" ]]; then echo "  DRY: $*"; else eval "$@"; fi; }

if [[ "$HARD" == "1" ]]; then
  say "HARD 初期化: ${NF_DIR}, ${LL_DIR} を丸ごと削除 → 再作成します（logs も消えます）。"
  run "rm -rf '${NF_DIR}' '${LL_DIR}'"
else
  say "ソフト初期化: runs/models_full/lightning_logs を空にし、logs は保持します。"
  run "rm -rf '${RUNS_DIR}' '${MODELS_DIR}' '${LL_DIR}'"
fi

run "mkdir -p '${RUNS_DIR}' '${MODELS_DIR}' '${DIAG_DIR}'"
run "mkdir -p '${LOGS_DIR}'"
run "mkdir -p '${LL_DIR}'"

if [[ "$RESET_DB" == "1" ]]; then
  if [[ -f "${DB_PATH}" ]]; then
    say "preds.sqlite を削除します: ${DB_PATH}"
    run "rm -f '${DB_PATH}'"
  else
    say "preds.sqlite は見つかりません（スキップ）"
  fi
else
  say "preds.sqlite は保持します（--reset-db 指定で削除可）。"
fi

# ▼ 追加: __pycache__ と *.py[co] の掃除（既定ON）
if [[ "$CLEAN_PYCACHE" == "1" ]]; then
  say "__pycache__ と *.py[co] を削除します（リポジトリ配下のみ）。"
  # __pycache__ ディレクトリの削除
  run "find '${ROOT_DIR}' -type d -name '__pycache__' -not -path '${ROOT_DIR}/.git/*' -exec rm -rf {} +"
  # .pyc / .pyo の削除
  run "find '${ROOT_DIR}' -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -not -path '${ROOT_DIR}/.git/*' -delete"
else
  say "__pycache__ と *.py[co] の削除はスキップ（--no-pyc 指定）。"
fi

say "完了: 出力物とキャッシュを初期化しました。構成:"
say "  - ${RUNS_DIR}"
say "  - ${MODELS_DIR}"
say "  - ${LOGS_DIR}"
say "  - ${DIAG_DIR}"
say "  - ${LL_DIR}"
[[ "$DRY_RUN" == "1" ]] && say "(dry-run のため実際の削除は行っていません)"
