"""Timestamp format checker for all CSVs under the input directory.

This script scans every CSV recursively, parses the first column as timestamps,
and reports basic diagnostics:
- total rows and invalid timestamp count
- unique seconds and minute modulo 5 values
- unique time-step sizes in minutes (between consecutive rows)

Usage:
    python scripts/check_timestamps.py            # default root: input
    python scripts/check_timestamps.py --root input/case118
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def _format_list(values: Sequence) -> str:
    """Format a small list for display."""
    if not values:
        return "[]"
    return "[" + ", ".join(str(v) for v in values) + "]"


def analyze_csv(path: Path) -> Dict[str, object]:
    """Load timestamp column from a CSV and compute diagnostics."""
    df = pd.read_csv(path, usecols=[0], dtype=str)
    col = df.columns[0]
    ts_raw = df[col].astype(str)
    ts_parsed = pd.to_datetime(ts_raw, errors="coerce")
    invalid_mask = ts_parsed.isna()
    invalid_count = int(invalid_mask.sum())
    ts_valid = ts_parsed.dropna()
    rows_total = int(len(ts_raw))
    rows_valid = int(len(ts_valid))

    if rows_valid == 0:
        return {
            "path": path,
            "rows_total": rows_total,
            "rows_valid": rows_valid,
            "invalid_count": invalid_count,
            "seconds_unique": [],
            "minute_mod5": [],
            "step_minutes": [],
            "min_ts": None,
            "max_ts": None,
        }

    seconds_unique = sorted(ts_valid.dt.second.unique().tolist())
    minute_mod5 = sorted((ts_valid.dt.minute % 5).unique().tolist())
    deltas = ts_valid.diff().dropna().dt.total_seconds() / 60.0
    step_minutes = sorted(round(x, 3) for x in deltas.unique().tolist())

    return {
        "path": path,
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "invalid_count": invalid_count,
        "seconds_unique": seconds_unique,
        "minute_mod5": minute_mod5,
        "step_minutes": step_minutes,
        "min_ts": ts_valid.min(),
        "max_ts": ts_valid.max(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check timestamp format for CSV files.")
    parser.add_argument(
        "--root",
        default="input",
        help="Root directory to scan recursively for CSV files (default: input).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[error] Root not found: {root}")

    csv_files: List[Path] = sorted(root.rglob("*.csv"))
    if not csv_files:
        print(f"[warn] No CSV files found under {root}")
        return

    overall_invalid = 0
    non_mod5_files: List[Path] = []
    print(f"Scanning {len(csv_files)} CSV files under {root} ...\n")
    for p in csv_files:
        try:
            info = analyze_csv(p)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"- {p}: FAILED to analyze ({exc})")
            continue

        overall_invalid += info["invalid_count"]  # type: ignore[index]
        if info["minute_mod5"] not in ([0], []):  # type: ignore[index]
            non_mod5_files.append(p)

        print(
            f"- {p} | rows={info['rows_total']} valid={info['rows_valid']} "
            f"invalid={info['invalid_count']} seconds={_format_list(info['seconds_unique'])} "
            f"minute%5={_format_list(info['minute_mod5'])} "
            f"steps(min)={_format_list(info['step_minutes'])}"
        )

    print("\nSummary:")
    print(f"- total files: {len(csv_files)}")
    print(f"- total invalid timestamps: {overall_invalid}")
    if non_mod5_files:
        print(f"- files with minute%5 != 0: {len(non_mod5_files)}")
        for p in non_mod5_files:
            print(f"  * {p}")
    else:
        print("- all files have minute%5 == 0")


if __name__ == "__main__":
    main()
