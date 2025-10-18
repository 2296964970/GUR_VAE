"""
Check if timestamps in CSV file are ordered and detect duplicates.

Validates chronological ordering of timestamps and reports any out-of-order
entries or duplicate timestamps.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import load_csv, count_duplicates


def check_order(csv_path: str) -> None:
    """
    Check timestamp ordering in CSV file.

    Args:
        csv_path: Path to CSV file to check
    """
    df = load_csv(csv_path)
    ts_col = df.iloc[:, 0]

    # Parse as datetime
    ts_parsed = pd.to_datetime(ts_col, errors='coerce')

    # Check for parse failures
    if ts_parsed.isna().any():
        print(f"WARNING: {ts_parsed.isna().sum()} timestamps could not be parsed")

    # Find all disordered positions
    disorder = []
    for i in range(1, len(ts_parsed)):
        if ts_parsed.iloc[i] < ts_parsed.iloc[i-1]:
            disorder.append(i)

    if len(disorder) == 0:
        print("OK - Timestamps are fully ordered")
    else:
        print(f"ERROR - Found {len(disorder)} disordered timestamps:")
        for idx in disorder[:10]:  # Show first 10 only
            print(f"  row{idx-1}: {ts_col.iloc[idx-1]} -> row{idx}: {ts_col.iloc[idx]}")
        if len(disorder) > 10:
            print(f"  ... and {len(disorder)-10} more")

    # Check for duplicate timestamps
    dups = count_duplicates(ts_col)
    if dups > 0:
        print(f"ERROR - Found {dups} duplicate timestamps")
    else:
        print("OK - No duplicate timestamps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file path")
    args = parser.parse_args()

    print(f"Checking file: {args.csv}")
    check_order(args.csv)
