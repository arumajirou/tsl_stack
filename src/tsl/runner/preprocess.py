# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import os
import re

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


# ---- CSV ロード & 標準化 -----------------------------------------------------

def load_csv_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    def _first_present(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    uid = _first_present(["unique_id", "id", "series", "item_id"])
    ds = _first_present(["ds", "date", "timestamp", "datetime"])
    y = _first_present(["y", "value", "target"])

    if y is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("目標列(y/value/target)が見つかりません。")
        y = num_cols[0]

    if ds is None:
        df = df.copy()
        if uid is None:
            df["unique_id"] = "series_0"; uid = "unique_id"
        df["__idx"] = df.groupby(uid).cumcount()
        df["ds"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(df["__idx"], unit="D")
        ds = "ds"
    else:
        df[ds] = pd.to_datetime(df[ds], errors="coerce")
        if df[ds].isna().any():
            bad = df[df[ds].isna()].head(3)
            raise ValueError(f"日付列 {ds} に日時化できない値があります。例:\n{bad}")

    if uid is None:
        df = df.copy()
        df["unique_id"] = "series_0"; uid = "unique_id"

    keep = [uid, ds, y] + [c for c in df.columns if c not in (uid, ds, y)]
    df = df[keep].rename(columns={uid: "unique_id", ds: "ds", y: "y"})
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


# ---- exog の簡易エンコード（接頭辞グループのみ） -----------------------------

def encode_exogs_by_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """hist_/futr_/stat_ のみカテゴリ等を数値化。"""
    core = {"unique_id", "ds", "y"}
    exog_cols = [c for c in df.columns
                 if c not in core and (c.startswith("futr_") or c.startswith("hist_") or c.startswith("stat_"))]

    if not exog_cols:
        return df

    qpat = re.compile(r"^(\d{4})Q([1-4])$")
    to_drop: List[str] = []

    for c in exog_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.astype(str)
        # "YYYYQd" -> year, q に展開
        if vals.map(lambda x: bool(qpat.match(x))).all():
            df[f"{c}__year"] = vals.map(lambda x: int(qpat.match(x).group(1)))
            df[f"{c}__q"] = vals.map(lambda x: int(qpat.match(x).group(2)))
            to_drop.append(c)
        else:
            cat = pd.Categorical(vals)
            df[c] = cat.codes.astype("int32")

    if to_drop:
        df.drop(columns=to_drop, inplace=True)

    return df


# ---- exog の抽出（接頭辞だけで判定） ----------------------------------------

def split_exog_by_prefix(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cols = [c for c in df.columns if c not in ("unique_id", "ds", "y")]
    futr = [c for c in cols if c.startswith("futr_")]
    hist = [c for c in cols if c.startswith("hist_")]
    stat = [c for c in cols if c.startswith("stat_")]
    return futr, hist, stat


# ---- freq / h / val_size ----------------------------------------------------

def _infer_freq_from_index(idx: pd.DatetimeIndex) -> Optional[str]:
    if len(idx) < 3:
        return None
    off = pd.infer_freq(idx)
    if off:
        return off
    diffs = (idx[1:] - idx[:-1]).to_series(index=idx[1:]).dropna()
    if diffs.empty:
        return None
    mode_delta = diffs.mode().iloc[0]
    try:
        return to_offset(mode_delta).freqstr
    except Exception:
        return None


def infer_global_freq(df: pd.DataFrame) -> Tuple[str, Dict[str, int]]:
    freq_counts: Dict[str, int] = {}
    for _, g in df.groupby("unique_id"):
        idx = pd.DatetimeIndex(g["ds"].sort_values().values)
        f = _infer_freq_from_index(idx)
        if f:
            freq_counts[f] = freq_counts.get(f, 0) + 1
    if not freq_counts:
        return "D", {}
    global_freq = sorted(freq_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return global_freq, freq_counts


def infer_h(df: pd.DataFrame, default_h: int) -> int:
    """系列の最短長から安全な h を推定。"""
    lens = df.groupby("unique_id")["ds"].count()
    if lens.empty:
        return max(1, default_h)
    min_len = int(lens.min())
    if min_len <= 5:
        return 1
    h_cap = max(1, int(min_len * float(os.getenv("NF_H_RATIO", "0.1"))))
    # ← ここのカッコを 1 個減らしました（SyntaxError 修正）
    return int(max(1, min(default_h, h_cap, max(1, min_len - 1))))


def _min_series_len(df: pd.DataFrame) -> int:
    return int(df.groupby("unique_id")["ds"].count().min())


def parse_val_size(df: pd.DataFrame, h_val: int) -> int:
    raw = os.getenv("NF_VAL_SIZE", "h").strip().lower()
    if raw in ("", "auto", "h"):
        return int(h_val)
    try:
        f = float(raw)
        if 0 < f < 1:
            m = _min_series_len(df)
            return max(1, int(round(m * f)))
    except Exception:
        pass
    try:
        return max(1, int(float(raw)))
    except Exception:
        return int(h_val)


# ---- 列制限 & 欠損対策 ------------------------------------------------------

def columns_used_from_kwargs(df: pd.DataFrame, kw: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    keep = ["unique_id", "ds", "y"]
    for k in ("hist_exog_list", "futr_exog_list", "stat_exog_list"):
        if k in kw and kw[k]:
            keep += [c for c in kw[k] if c in df.columns]
    # 一意化
    keep = list(dict.fromkeys(keep))
    missing = [c for c in keep if c not in df.columns]
    drop = [c for c in df.columns if c not in keep]
    return keep, drop, missing


def restrict_df_columns(df: pd.DataFrame, keep: List[str]) -> pd.DataFrame:
    return df[keep].copy()


def impute_or_drop_nan(
    df: pd.DataFrame,
    used_exog_cols: List[str],
    policy: str = "ffill_bfill_zero"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    policy = (policy or "ffill_bfill_zero").strip().lower()
    info = {"policy": policy, "dropped_rows": 0, "dropped_cols": [], "filled_cols": []}

    if not used_exog_cols:
        return df, info

    uid = "unique_id"

    if policy == "drop_cols":
        bad_cols = [c for c in used_exog_cols if df[c].isna().any()]
        info["dropped_cols"] = bad_cols
        return df.drop(columns=bad_cols), info

    if policy == "drop_rows":
        def _trim(g):
            mask = g[["y"] + used_exog_cols].notna().all(axis=1)
            if not mask.any():
                return g.iloc[0:0]
            first_idx = mask.idxmax()
            return g.loc[first_idx:]

        before = len(df)
        out = df.groupby(uid, group_keys=False, sort=False).apply(lambda g: _trim(g.sort_values("ds")))
        info["dropped_rows"] = before - len(out)
        return out.reset_index(drop=True), info

    # default: ffill -> bfill -> 0 fill
    def _fill(g):
        g = g.sort_values("ds")
        g[used_exog_cols] = g[used_exog_cols].ffill().bfill()
        return g

    df2 = df.groupby(uid, group_keys=False, sort=False).apply(_fill).reset_index(drop=True)
    still_na = [c for c in used_exog_cols if df2[c].isna().any()]
    if still_na:
        df2[still_na] = df2[still_na].fillna(0)
        info["filled_cols"] = still_na
    return df2, info


def assert_no_missing(df: pd.DataFrame, cols: List[str], context: str):
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        head = bad[:8]
        sfx = "..." if len(bad) > 8 else ""
        raise ValueError(f"[{context}] Missing values remain in columns: {head}{sfx}")
