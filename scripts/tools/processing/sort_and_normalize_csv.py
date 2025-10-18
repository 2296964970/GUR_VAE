"""
Sort CSV by timestamp and normalize format to YYYY/MM/DD HH:MM.

Optionally removes duplicate timestamps.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import load_csv, normalize_and_sort, save_csv


def sort_and_normalize(path_in: str, path_out: str, dedup: bool = False) -> None:
    """
    Sort and normalize timestamps in CSV file.

    Args:
        path_in: Input CSV path
        path_out: Output CSV path
        dedup: If True, remove duplicate timestamps
    """
    df = load_csv(path_in)
    df = normalize_and_sort(df)

    if dedup:
        df = df.drop_duplicates(subset=[df.columns[0]], keep="first")

    save_csv(df, path_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sort by timestamp and normalize format")
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    sort_and_normalize(args.in_path, args.out_path, dedup=args.dedup)
    print("wrote:", args.out_path)


if __name__ == "__main__":
    main()
