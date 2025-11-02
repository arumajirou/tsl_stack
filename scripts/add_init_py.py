#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensure __init__.py exists for planned Python packages under given roots.

Usage:
  python scripts/add_init_py.py --apply
  python scripts/add_init_py.py --dry-run
  python scripts/add_init_py.py --roots src tsl_integrated_pkg/src
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

DEFAULT_ROOTS = ["src", "tsl_integrated_pkg/src"]
PKG_DIR_PATTERNS = ["tsl"]  # "tsl" 配下を対象。必要なら増やす: ["tsl", "yourpkg"]

EXCLUDES = {
    "__pycache__", ".mypy_cache", ".pytest_cache", ".venv",
    "build", "dist", ".git", "lightning_logs", "nf_auto_runs"
}

HEADER = """# -*- coding: utf-8 -*-
\"\"\"Package marker: created by scripts/add_init_py.py\"\"\"
"""

def is_excluded(p: Path) -> bool:
    return any(part in EXCLUDES for part in p.parts)

def looks_like_pkg_dir(d: Path) -> bool:
    """直下 or 下位に .py が一つでもあればパッケージ候補とみなす"""
    for p in d.glob("*.py"):
        return True
    for sub in d.rglob("*.py"):
        try:
            rel = sub.relative_to(d)
        except Exception:
            continue
        if len(rel.parts) >= 2:
            return True
    return False

def should_visit(d: Path) -> bool:
    # 例: src/tsl/**, tsl_integrated_pkg/src/tsl/**
    # ルート直下のパッケージ名と一致する経路だけ対象
    parts = d.parts
    for name in PKG_DIR_PATTERNS:
        if name in parts:
            return True
    return False

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Search roots")
    ap.add_argument("--apply", action="store_true", help="Create files for real")
    ap.add_argument("--dry-run", action="store_true", help="Only show what would happen")
    args = ap.parse_args(argv)

    dry = args.dry_run and not args.apply
    roots = [Path(r).resolve() for r in args.roots]

    checked = 0
    created = 0
    for root in roots:
        if not root.exists():
            continue
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            if is_excluded(d):
                continue
            if not should_visit(d):
                continue
            checked += 1
            init = d / "__init__.py"
            if init.exists():
                continue
            if not looks_like_pkg_dir(d):
                continue
            if dry:
                print(f"[DRY] would create: {init}")
            else:
                init.write_text(HEADER, encoding="utf-8")
                print(f"created: {init}")
            created += 1

    print(f"checked={checked}, created={created}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
