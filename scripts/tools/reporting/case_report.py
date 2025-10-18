"""
Generate comprehensive case report for dataset quality.

Reports shapes, string-sorted vs time-sorted status, parseability, duplicates,
format differences, and alignment between normal, attacked, and label files.
"""

import argparse
import os
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
    check_format_mismatch
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Case report: shapes, string-sorted vs time-sorted, parseability, duplicates, format diffs, alignment"
    )
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--case", required=True)
    args = ap.parse_args()

    dd = args.data_dir
    case = args.case

    p_norm = os.path.join(dd, case, f"{case}_acopf_all_rows_noisy.csv")
    p_att = os.path.join(dd, case, f"{case}_fdia_2025-07_2025-08_noisy.csv")
    p_lab = os.path.join(dd, case, f"{case}_fdia_2025-07_2025-08_labels.csv")

    N = load_csv(p_norm)
    A = load_csv(p_att)
    L = load_csv(p_lab)

    for name, DF in [("NORMAL", N), ("ATTACK", A), ("LABELS", L)]:
        col = DF.iloc[:, 0]

        s_sorted = is_string_sorted(col)
        t_sorted, nat = is_time_sorted(col)
        dups = count_duplicates(col)
        mism, samples = check_format_mismatch(col)

        print(f"{name}: {DF.shape} string_sorted:{s_sorted} time_sorted:{t_sorted} nat:{nat} dups:{dups} format_mismatch:{mism}")

        if mism and samples:
            print(f"  format_mismatch_samples (original -> normalized) up to 5:")
            for a, b in samples[:5]:
                print(f"   - {a} -> {b}")

    # Alignment after normalization and sort
    An = normalize_and_sort(A)
    Ln = normalize_and_sort(L)

    headers_equal = list(An.columns) == list(Ln.columns)
    ts_equal_norm = (An.iloc[:, 0].values == Ln.iloc[:, 0].values).all()
    ts_equal_raw = (A.iloc[:, 0].astype(str).values == L.iloc[:, 0].astype(str).values).all()

    print("ALIGNMENT")
    print("  headers_equal:", headers_equal)
    print("  raw_timestamps_equal:", ts_equal_raw)
    print("  normalized_timestamps_equal:", ts_equal_norm)


if __name__ == "__main__":
    main()
