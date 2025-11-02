#!/usr/bin/env bash
# Safe tidy script: 常に成功終了。--dry-run / --aggressive / --move-sample-data / --no-symlink
set -u  # -e/-o pipefail は使わない（非ゼロを拾って落ちないため）

DRY=0
AGGR=0
MOVE=0
NO_SYM=0

for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --aggressive) AGGR=1 ;;
    --move-sample-data) MOVE=1 ;;
    --no-symlink) NO_SYM=1 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_DIR="$ROOT/archive/$(date +%Y%m%d_%H%M%S)"

say() { printf '%s\n' "[tidy] $*"; }
run() {
  if [ "$DRY" -eq 1 ]; then
    printf '  DRY: %s\n' "$*"
    return 0
  else
    # どこかで失敗しても続ける
    bash -c "$*" || true
  fi
}

say "ROOT: $ROOT"
say "ARCHIVE: $ARCHIVE_DIR (バックアップ退避先)"

# 1) lightning_logs を nf_auto_runs へ集約し、必要なら symlink
TARGET_LL="$ROOT/nf_auto_runs/lightning_logs"
SRC_LL="$ROOT/lightning_logs"
say "集約: lightning_logs → $TARGET_LL"
run "mkdir -p '$TARGET_LL'"
if [ "$NO_SYM" -eq 0 ]; then
  say "symlink 作成: lightning_logs -> nf_auto_runs/lightning_logs"
  # 既存がファイル/ディレクトリ/リンクでも確実に置換
  if [ "$DRY" -eq 1 ]; then
    run "rm -f '$SRC_LL'"
    run "ln -s 'nf_auto_runs/lightning_logs' '$SRC_LL'"
  else
    [ -e "$SRC_LL" ] || [ -L "$SRC_LL" ] && rm -rf "$SRC_LL" || true
    ln -s 'nf_auto_runs/lightning_logs' "$SRC_LL" || true
  fi
fi

# 2) ルートの out*.json を削除
say "削除: ルートの out*.json"
run "find '$ROOT' -maxdepth 1 -type f -name 'out*.json' -delete"

# 3) バックアップ系を archive/ に退避
BK1="$ROOT/_backup_tsl_integrated_pkg_"*
BK2="$ROOT/_migration_backup_"*
if compgen -G "$BK1" >/dev/null || compgen -G "$BK2" >/dev/null; then
  run "mkdir -p '$ARCHIVE_DIR'"
  for p in $BK1 $BK2; do
    [ -e "$p" ] || continue
    say "退避: $p → $ARCHIVE_DIR/"
    run "mv '$p' '$ARCHIVE_DIR/'"
  done
else
  say "バックアップ系は見つかりませんでした（スキップ）"
fi

# 4) aggressive: egg-info も削除
if [ "$AGGR" -eq 1 ]; then
  if compgen -G "$ROOT/src/*.egg-info" >/dev/null || [ -d "$ROOT/src/tsl_stack.egg-info" ]; then
    for eg in "$ROOT"/src/*.egg-info "$ROOT"/src/tsl_stack.egg-info; do
      [ -e "$eg" ] || continue
      say "削除(aggressive): $eg（再生成可: pip install -e .）"
      run "rm -rf '$eg'"
    done
  else
    say "egg-info は見つかりません（スキップ）"
  fi
fi

# 5) 掃除: __pycache__, *.pyc, *.pyo
say "掃除: __pycache__, *.pyc, *.pyo"
run "find '$ROOT' -type d -name '__pycache__' -prune -exec rm -rf {} +"
run "find '$ROOT' -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -delete"

# 6) サンプル csv の移動
if [ "$MOVE" -eq 1 ]; then
  if [ -e "$ROOT/gpu_smoke.csv" ]; then
    say "移動: gpu_smoke.csv → tests/data/"
    run "mkdir -p '$ROOT/tests/data'"
    run "git mv -k '$ROOT/gpu_smoke.csv' '$ROOT/tests/data/' || mv '$ROOT/gpu_smoke.csv' '$ROOT/tests/data/'"
  else
    say "gpu_smoke.csv は見つかりません（スキップ）"
  fi
fi

# 7) 空ディレクトリ削除（archive 配下は残す）
say "空ディレクトリを整理中…"
run "find '$ROOT' -type d -empty -not -path '$ROOT' -not -path '$ROOT/archive' -not -path '$ROOT/archive/*' -delete"

say "完了。必要なら: pip install -e . で egg-info を再生成できます。"
# 必ず成功終了
exit 0
