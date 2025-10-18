"""
Compare timestamp sets between two CSV files.

Identifies differences in timestamp sets between files, reporting unique
timestamps in each file and optional summary output.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import load_csv, normalize_and_sort


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare timestamp sets between two CSVs")
    ap.add_argument("--a", required=True, help="CSV A")
    ap.add_argument("--b", required=True, help="CSV B")
    ap.add_argument("--out", default=None, help="Optional TSV to write summary")
    args = ap.parse_args()

    A = normalize_and_sort(load_csv(args.a))
    B = normalize_and_sort(load_csv(args.b))

    sa = set(A.iloc[:, 0])
    sb = set(B.iloc[:, 0])

    only_a = sorted(list(sa - sb))[:50]
    only_b = sorted(list(sb - sa))[:50]

    print("len(A)", len(A), "len(B)", len(B))
    print("only_in_A", len(sa - sb))
    print("only_in_B", len(sb - sa))

    if only_a:
        print("sample only_in_A:", only_a[:10])
    if only_b:
        print("sample only_in_B:", only_b[:10])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("key\tvalue\n")
            f.write(f"len_A\t{len(A)}\n")
            f.write(f"len_B\t{len(B)}\n")
            f.write(f"only_in_A\t{len(sa - sb)}\n")
            f.write(f"only_in_B\t{len(sb - sa)}\n")


if __name__ == "__main__":
    main()
