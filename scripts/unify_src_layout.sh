#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src/tsl"
INT_DIR="${REPO_ROOT}/tsl_integrated_pkg/src/tsl"
BACKUP_DIR="${REPO_ROOT}/_migration_backup_$(date +%Y%m%d_%H%M%S)"
DIFF_LIST="${BACKUP_DIR}/diff_list.txt"

echo "[info] repo: ${REPO_ROOT}"
echo "[info] src:  ${SRC_DIR}"
echo "[info] int:  ${INT_DIR}"

if [[ ! -d "${SRC_DIR}" || ! -d "${INT_DIR}" ]]; then
  echo "[error] src/tsl または tsl_integrated_pkg/src/tsl が見つかりません。" >&2
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

echo "[step] 差分一覧を作成"
diff -rq "${INT_DIR}" "${SRC_DIR}" | tee "${DIFF_LIST}" || true

echo "[step] src に存在しないファイルを取り込み（rsync --ignore-existing）"
rsync -av --ignore-existing "${INT_DIR}/" "${SRC_DIR}/"

echo "[step] 同名で内容が異なるファイルを退避コピー（*.from_integrated）"
# diff -rq の "Files A and B differ" 行だけ処理
grep '^Files ' "${DIFF_LIST}" | sed 's/^Files \(.*\) and \(.*\) differ.*/\1|\2/' | while IFS='|' read -r INTF SRCF; do
  rel="${INTF#${INT_DIR}/}"
  dst="${SRC_DIR}/${rel}.from_integrated"
  mkdir -p "$(dirname "${dst}")"
  cp -a "${INTF}" "${dst}"
  echo "  -> ${rel} に差分。統合候補を ${dst} に退避しました"
done

echo "[step] 統合前の統合元をバックアップへコピー"
mkdir -p "${BACKUP_DIR}/tsl_integrated_pkg"
cp -a "${REPO_ROOT}/tsl_integrated_pkg" "${BACKUP_DIR}/"

echo "[step] tsl_integrated_pkg を削除"
rm -rf "${REPO_ROOT}/tsl_integrated_pkg"

echo "[done] 統合完了。差分は ${BACKUP_DIR} を参照し、必要なら *.from_integrated を手でマージしてください。"
