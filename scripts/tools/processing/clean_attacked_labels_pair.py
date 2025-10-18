"""
Produce aligned attacked/labels pair on timestamp intersection.

Takes attacked and labels CSV files, finds common timestamps, removes duplicates,
and produces clean aligned output files.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import load_csv, normalize_and_sort, save_csv


def intersect_align(attacked: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align attacked and labels DataFrames on timestamp intersection.

    Args:
        attacked: Attacked data DataFrame
        labels: Labels DataFrame

    Returns:
        (aligned_attacked, aligned_labels) tuple
    """
    a = normalize_and_sort(attacked)
    l = normalize_and_sort(labels)

    key = a.columns[0]

    # Remove duplicates, keeping first occurrence
    a2 = a.drop_duplicates(subset=[key], keep="first")
    l2 = l.drop_duplicates(subset=[key], keep="first")

    # Find intersection of timestamps
    inter = sorted(set(a2[key]).intersection(set(l2[key])))

    # Filter to intersection and sort
    a_al = a2[a2[key].isin(inter)].sort_values(key).reset_index(drop=True)
    l_al = l2[l2[key].isin(inter)].sort_values(key).reset_index(drop=True)

    return a_al, l_al


def main() -> None:
    ap = argparse.ArgumentParser(description="Produce aligned attacked/labels pair on timestamp intersection")
    ap.add_argument("--attacked", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--case", required=True, help="case name prefix for outputs")
    ap.add_argument("--normal", default=None, help="optional normal CSV to copy as-is")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    a = load_csv(args.attacked)
    l = load_csv(args.labels)

    a_al, l_al = intersect_align(a, l)

    out_a = os.path.join(args.out_dir, f"{args.case}_fdia_2025-07_2025-08_noisy.csv")
    out_l = os.path.join(args.out_dir, f"{args.case}_fdia_2025-07_2025-08_labels.csv")

    save_csv(a_al, out_a)
    save_csv(l_al, out_l)

    print("wrote:", out_a)
    print("wrote:", out_l)

    if args.normal:
        import shutil
        shutil.copy2(args.normal, os.path.join(args.out_dir, os.path.basename(args.normal)))


if __name__ == "__main__":
    main()
