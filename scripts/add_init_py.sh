#!/usr/bin/env bash

# 走査ルート（必要に応じて追加）
ROOTS=( "src" "tsl_integrated_pkg/src" )

# 対象にしたいトップレベル・パッケージ名
# 例: src/tsl/**, tsl_integrated_pkg/src/tsl/**
TARGET_PREFIXES=( "tsl" )

# 除外ディレクトリ（build成果物・キャッシュなど）
# ※ find へは 1 回の式で渡します（-prune の定石）
EXCLUDE_FIND_EXPR=(
  \( -name __pycache__ -o -name '*.egg-info' -o -name build -o -name dist
     -o -name .venv -o -name .mypy_cache -o -name .pytest_cache
     -o -name .git -o -name lightning_logs -o -name nf_auto_runs \) -prune -o
)

HEADER=$'# -*- coding: utf-8 -*-\n"""Package marker: created by scripts/add_init_py.sh"""\n'

DRY=${DRY_RUN:-0}
created=0
checked=0

is_target_dir () {
  local d="$1"
  for prefix in "${TARGET_PREFIXES[@]}"; do
    # ルート直下 or それ以下に prefix が含まれていれば対象
    if [[ "$d" == */"$prefix" || "$d" == */"$prefix"/* ]]; then
      return 0
    fi
  done
  return 1
}

# 直下 or 下位に .py が一つでもあれば“将来モジュール候補”とみなす
looks_like_pkg () {
  local d="$1"
  # 直下に .py
  if find "$d" -maxdepth 1 -type f -name '*.py' -quit | grep -q .; then
    return 0
  fi
  # さらに下位
  if find "$d" -mindepth 2 -type f -name '*.py' -quit | grep -q .; then
    return 0
  fi
  return 1
}

for root in "${ROOTS[@]}"; do
  [[ -d "$root" ]] || continue

  # -print0 で NUL 区切り → while で安全に読む
  while IFS= read -r -d '' dir; do
    ((checked++))

    is_target_dir "$dir" || continue
    looks_like_pkg "$dir" || continue

    init="$dir/__init__.py"
    if [[ ! -e "$init" ]]; then
      if [[ "$DRY" == "1" ]]; then
        echo "[DRY] would create: $init"
      else
        printf "%s" "$HEADER" > "$init"
        echo "created: $init"
      fi
      ((created++))
    fi
  done < <(find "$root" "${EXCLUDE_FIND_EXPR[@]}" -type d -print0)
done

echo "checked=$checked, created=$created"
