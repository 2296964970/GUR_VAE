"""
Check alignment between attacked data and labels CSV files.

Validates that timestamps match and data is properly aligned between
attack data and corresponding label files.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import (
    load_csv,
    normalize_and_sort,
    is_string_sorted,
    is_time_sorted,
    count_duplicates,
    compare_timestamps
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check alignment between attacked and labels CSV")
    ap.add_argument("--attacked", required=True, help="Path to attacked CSV")
    ap.add_argument("--labels", required=True, help="Path to labels CSV")
    ap.add_argument("--normal", default=None, help="Optional normal CSV for header/time reference")
    args = ap.parse_args()

    a_raw = load_csv(args.attacked)
    l_raw = load_csv(args.labels)

    print("RAW INFO (string-based)")
    print("attacked shape:", a_raw.shape)
    print("labels   shape:", l_raw.shape)
    print("raw headers equal:", list(a_raw.columns) == list(l_raw.columns))
    print("raw string_sorted attacked:", is_string_sorted(a_raw.iloc[:, 0]), "labels:", is_string_sorted(l_raw.iloc[:, 0]))
    print("raw dups attacked:", count_duplicates(a_raw.iloc[:, 0]), "labels:", count_duplicates(l_raw.iloc[:, 0]))

    equal_raw, idx_raw = compare_timestamps(a_raw.iloc[:, 0].astype(str), l_raw.iloc[:, 0].astype(str))
    print("raw timestamps equal (string):", equal_raw)
    if not equal_raw:
        print("raw first mismatch idx:", idx_raw)
        if len(idx_raw) > 0:
            i = idx_raw[0]
            print("attacked ts at i:", a_raw.iloc[i, 0])
            print("labels   ts at i:", l_raw.iloc[i, 0])

    # Time-based sortedness (after parsing)
    a_time_sorted, a_nat = is_time_sorted(a_raw.iloc[:, 0])
    l_time_sorted, l_nat = is_time_sorted(l_raw.iloc[:, 0])
    print("time_sorted attacked:", a_time_sorted, "nat:", a_nat, "; labels:", l_time_sorted, "nat:", l_nat)

    a_norm = normalize_and_sort(a_raw)
    l_norm = normalize_and_sort(l_raw)

    print("\nNORMALIZED-SORTED INFO (time-based)")
    print("headers equal:", list(a_norm.columns) == list(l_norm.columns))
    print("string_sorted attacked:", is_string_sorted(a_norm.iloc[:, 0]), "labels:", is_string_sorted(l_norm.iloc[:, 0]))
    print("dups attacked:", count_duplicates(a_norm.iloc[:, 0]), "labels:", count_duplicates(l_norm.iloc[:, 0]))

    equal_ns, idx_ns = compare_timestamps(a_norm.iloc[:, 0], l_norm.iloc[:, 0])
    print("timestamps equal (normalized):", equal_ns)
    if not equal_ns:
        print("first mismatch idx:", idx_ns)
        if len(idx_ns) > 0:
            i = idx_ns[0]
            print("attacked ts at i:", a_norm.iloc[i, 0])
            print("labels   ts at i:", l_norm.iloc[i, 0])

    if args.normal:
        n_raw = load_csv(args.normal)
        print("\nNORMAL (reference)")
        print("normal shape:", n_raw.shape)
        print("normal string_sorted:", is_string_sorted(n_raw.iloc[:, 0]), "dups:", count_duplicates(n_raw.iloc[:, 0]))


if __name__ == "__main__":
    main()
