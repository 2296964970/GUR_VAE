"""
Convert timestamp format in CSV files from YYYY/MM/DD HH:MM to YYYY-MM-DD HH:MM:SS
"""
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def convert_timestamp(ts_str):
    """
    Convert timestamp from YYYY/MM/DD HH:MM to YYYY-MM-DD HH:MM:SS
    Supports multiple input formats and skips already converted timestamps.
    """
    if pd.isna(ts_str):
        return ts_str

    ts_str = str(ts_str).strip()

    # Try multiple formats
    formats_to_try = [
        "%Y/%m/%d %H:%M",        # Old format: 2025/09/01 09:30
        "%Y-%m-%d %H:%M:%S",     # Already in target format
        "%Y-%m-%d %H:%M",        # Target format without seconds
    ]

    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(ts_str, fmt)
            # Always return target format
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    # If no format matches, return as-is
    return ts_str


def convert_csv_file(input_path, output_path=None, backup=False):
    """
    Convert a single CSV file's timestamp format.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, will overwrite input)
        backup: If True and overwriting, create a backup with .bak extension
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
        if backup:
            backup_path = input_path.with_suffix(input_path.suffix + '.bak')
            if backup_path.exists():
                print(f"Backup already exists: {backup_path}")
            else:
                import shutil
                shutil.copy2(input_path, backup_path)
                print(f"Created backup: {backup_path}")

    print(f"Processing: {input_path}")

    # Read CSV
    df = pd.read_csv(input_path, encoding='utf-8')

    # Check if first column is timestamp
    first_col = df.columns[0]
    if first_col.lower() not in ['timestimp', 'timestamp', 'time']:
        print(f"Warning: First column '{first_col}' doesn't look like a timestamp column")

    # Convert timestamps
    original_count = len(df)
    df[first_col] = df[first_col].apply(convert_timestamp)

    # Write back to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[OK] Converted {original_count} rows, saved to: {output_path}")

    return output_path


def convert_directory(dir_path, pattern="*.csv", recursive=True, backup=False):
    """
    Convert all CSV files in a directory.

    Args:
        dir_path: Directory path
        pattern: File pattern to match (default: *.csv)
        recursive: If True, search subdirectories
        backup: If True, create backups before overwriting
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return

    # Find all CSV files
    if recursive:
        csv_files = list(dir_path.rglob(pattern))
    else:
        csv_files = list(dir_path.glob(pattern))

    if not csv_files:
        print(f"No CSV files found in {dir_path}")
        return

    print(f"Found {len(csv_files)} CSV files")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for csv_file in csv_files:
        try:
            convert_csv_file(csv_file, backup=backup)
            success_count += 1
            print()
        except Exception as e:
            print(f"[ERROR] Error processing {csv_file}: {e}")
            fail_count += 1
            print()

    print("=" * 60)
    print(f"Conversion complete: {success_count} succeeded, {fail_count} failed")


def main():
    """
    Main entry point for the script.

    Usage:
        python convert_timestamp_format.py <path> [--backup] [--no-recursive]

    Examples:
        # Convert all CSV files in input/ directory (overwrite directly)
        python convert_timestamp_format.py input/

        # Convert a single file with backup
        python convert_timestamp_format.py input/case14/train/data.csv --backup

        # Convert only files in the specified directory (not subdirectories)
        python convert_timestamp_format.py input/ --no-recursive
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_timestamp_format.py <path> [--backup] [--no-recursive]")
        print()
        print("Examples:")
        print("  python convert_timestamp_format.py input/")
        print("  python convert_timestamp_format.py input/case14/train/data.csv --backup")
        print("  python convert_timestamp_format.py input/ --no-recursive")
        sys.exit(1)

    path = Path(sys.argv[1])
    backup = "--backup" in sys.argv
    recursive = "--no-recursive" not in sys.argv

    if path.is_file():
        # Convert single file
        convert_csv_file(path, backup=backup)
    elif path.is_dir():
        # Convert directory
        convert_directory(path, recursive=recursive, backup=backup)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
