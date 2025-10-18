"""
Common utility functions for data processing and validation.

Provides timestamp processing, data validation, and other shared utilities
to avoid code duplication across tools.
"""

from typing import Tuple, List
import pandas as pd


def normalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and sort DataFrame by first column (timestamp).

    Args:
        df: DataFrame to process, first column should be timestamp

    Returns:
        Processed DataFrame with timestamps formatted as YYYY/MM/DD HH:MM and sorted by time
    """
    df2 = df.copy()
    ts = pd.to_datetime(df2.iloc[:, 0], errors="coerce")

    if not ts.isna().any():
        # Time is parseable, sort by time and format
        df2 = df2.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)
        df2.iloc[:, 0] = df2["_ts"].dt.strftime("%Y/%m/%d %H:%M")
        df2 = df2.drop(columns=["_ts"])
    else:
        # Has unparseable times, sort as strings
        df2 = df2.sort_values(df2.columns[0]).reset_index(drop=True)
        df2.iloc[:, 0] = df2.iloc[:, 0].astype(str)

    return df2


def is_string_sorted(col: pd.Series) -> bool:
    """Check if column is sorted as strings."""
    s = col.astype(str)
    return bool(s.is_monotonic_increasing)


def is_time_sorted(col: pd.Series) -> Tuple[bool, int]:
    """
    Check if column is sorted as timestamps.

    Returns:
        (is_sorted, count_of_unparseable_times)
    """
    ts = pd.to_datetime(col.astype(str), errors="coerce")
    nat_count = int(ts.isna().sum())

    if nat_count > 0:
        return False, nat_count

    return bool(ts.is_monotonic_increasing), 0


def count_duplicates(col: pd.Series) -> int:
    """Count duplicate values in column."""
    return int(col.astype(str).duplicated().sum())


def check_format_mismatch(col: pd.Series) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Check count of timestamp format mismatches.

    Returns:
        (mismatch_count, [(original, normalized)] sample list)
    """
    s = col.astype(str)
    ts = pd.to_datetime(s, errors="coerce")
    normalized = ts.dt.strftime("%Y/%m/%d %H:%M")

    mismatch_mask = (s != normalized)
    mismatch_count = int(mismatch_mask.sum())

    samples = []
    if mismatch_count > 0:
        # Take first 10 mismatched samples
        mismatch_indices = mismatch_mask[mismatch_mask].index[:10]
        for idx in mismatch_indices:
            samples.append((s.iloc[idx], normalized.iloc[idx]))

    return mismatch_count, samples


def compare_timestamps(a: pd.Series, b: pd.Series) -> Tuple[bool, List[int]]:
    """
    Compare if two timestamp series are equal.

    Returns:
        (are_equal, list_of_mismatch_indices)
    """
    equal_mask = (a.values == b.values)

    if equal_mask.all():
        return True, []

    # Return first 20 mismatched indices
    mismatch_indices = [int(i) for i in (~equal_mask).nonzero()[0][:20]]
    return False, mismatch_indices


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save DataFrame as CSV."""
    df.to_csv(path, index=index)
