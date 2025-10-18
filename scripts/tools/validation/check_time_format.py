"""
Check time format consistency against normalized YYYY/MM/DD HH:MM format.

Validates that timestamp columns conform to expected format and reports
any format mismatches with samples.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import load_csv, check_format_mismatch


def analyze_time_format(path: str, sample: int = 10) -> None:
    """
    Analyze time format in CSV file.

    Args:
        path: Path to CSV file
        sample: Number of sample mismatches to show
    """
    df = load_csv(path)
    col = df.columns[0]
    mism_cnt, samples = check_format_mismatch(df.iloc[:, 0])

    print(f"FILE: {path}")
    print(f" rows: {len(df)} format_mismatch: {mism_cnt}")

    if mism_cnt > 0:
        print(" samples (original -> normalized):")
        for orig, norm in samples[:sample]:
            print(f"  - {orig} -> {norm}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check time format consistency against normalized YYYY/MM/DD HH:MM")
    ap.add_argument("paths", nargs="+", help="CSV paths to check")
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()

    for p in args.paths:
        analyze_time_format(p, sample=args.sample)


if __name__ == "__main__":
    main()
