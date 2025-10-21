"""Prepare SE triplet CSVs only at repaired timestamps.

This utility extracts exactly the timestamps present in a repaired tail-rows
CSV and emits three aligned CSVs (Normal/Attacked/Repaired) with identical
schema and row ordering for MATLAB state-estimation analysis.

Usage:
  python scripts/prepare_se_triplet.py \
      --normal_csv data/case14/case14_acopf_all_rows_noisy.csv \
      --attacked_csv data/case14/case14_fdia_2025-07_2025-08_noisy.csv \
      --repaired_tail_csv data/case14/..._repaired_tail_rows_L24_steps288_xxx.csv \
      --out_dir se_triplet/2025-08-14

Outputs in --out_dir:
  - normal_subset.csv
  - attacked_subset.csv
  - repaired_subset.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"[error] failed to read CSV: {path}: {e}")


def _canon_time(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise SystemExit(f"[error] timestamp column '{ts_col}' not found in df")
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors='coerce').dt.strftime('%Y/%m/%d %H:%M')
    if out[ts_col].isna().any():
        raise SystemExit('[error] failed to canonicalize timestamps')
    return out


def _first_col(df: pd.DataFrame) -> str:
    if df.shape[1] < 2:
        raise SystemExit('[error] CSV must have at least 2 columns (timestamp + 1 feature)')
    return str(df.columns[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--normal_csv', type=str, required=True)
    p.add_argument('--attacked_csv', type=str, required=True)
    p.add_argument('--repaired_tail_csv', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    args = p.parse_args()

    for path in (args.normal_csv, args.attacked_csv, args.repaired_tail_csv):
        if not os.path.exists(path):
            raise SystemExit(f"[error] file not found: {path}")

    df_n = _load_csv(args.normal_csv)
    df_a = _load_csv(args.attacked_csv)
    df_r = _load_csv(args.repaired_tail_csv)

    ts_na = _first_col(df_a)
    ts_nn = _first_col(df_n)
    if ts_na != ts_nn:
        # unify first-column header name to attacked's header
        cols = list(df_n.columns)
        cols[0] = ts_na
        df_n.columns = cols
    ts_col = ts_na
    # ensure repaired first column matches ts_col
    if _first_col(df_r) != ts_col:
        cols = list(df_r.columns)
        cols[0] = ts_col
        df_r.columns = cols

    # canonicalize timestamps
    df_n = _canon_time(df_n, ts_col)
    df_a = _canon_time(df_a, ts_col)
    df_r = _canon_time(df_r, ts_col)

    # base timestamps = repaired set
    rep_ts = set(df_r[ts_col].astype(str).values)
    if not rep_ts:
        raise SystemExit('[error] repaired_tail_csv has no timestamps')

    # feature columns order = attacked
    feat_cols: List[str] = [c for c in df_a.columns if c != ts_col]

    # restrict all to repaired timestamps and identical ordering
    def _subset(df: pd.DataFrame) -> pd.DataFrame:
        d = df[df[ts_col].astype(str).isin(rep_ts)].copy()
        d = d.sort_values(by=ts_col).reset_index(drop=True)
        # align feature columns to attacked order (subset to intersection if needed)
        common = [c for c in feat_cols if c in d.columns]
        if len(common) != len(feat_cols):
            # unlikely, but keep robustness
            sys.stderr.write('[warn] columns differ; restricting to common intersection\n')
        d = d[[ts_col] + common]
        return d

    out_n = _subset(df_n)
    out_a = _subset(df_a)
    out_r = _subset(df_r)

    # final sanity: identical shapes/headers
    if list(out_n.columns) != list(out_a.columns) or list(out_n.columns) != list(out_r.columns):
        raise SystemExit('[error] mismatched headers after alignment')
    if out_n.shape[0] != out_a.shape[0] or out_n.shape[0] != out_r.shape[0]:
        raise SystemExit('[error] mismatched row counts after filtering')

    os.makedirs(args.out_dir, exist_ok=True)
    out_n.to_csv(os.path.join(args.out_dir, 'normal_subset.csv'), index=False, encoding='utf-8')
    out_a.to_csv(os.path.join(args.out_dir, 'attacked_subset.csv'), index=False, encoding='utf-8')
    out_r.to_csv(os.path.join(args.out_dir, 'repaired_subset.csv'), index=False, encoding='utf-8')
    print('[ok] Wrote triplet to', args.out_dir, 'rows=', out_n.shape[0])


if __name__ == '__main__':
    main()

