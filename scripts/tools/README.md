# Data Processing and Validation Tools

A collection of utilities for processing, validating, and generating datasets for online VAE experiments.

## Directory Structure

```
tools/
├── common_utils.py          # Shared utility functions
├── validation/              # Data validation tools
├── processing/              # Data processing tools
├── reporting/               # Report generation tools
└── generators/              # Data generation tools
```

## Common Utilities

The `common_utils.py` module provides shared functions used across all tools:

- **normalize_and_sort()**: Normalize timestamps to YYYY/MM/DD HH:MM format and sort
- **is_string_sorted()**: Check if column is sorted as strings
- **is_time_sorted()**: Check if column is sorted chronologically
- **count_duplicates()**: Count duplicate values in column
- **check_format_mismatch()**: Check timestamp format consistency
- **compare_timestamps()**: Compare two timestamp series
- **load_csv() / save_csv()**: CSV I/O helpers

## Tool Categories

### 1. Validation Tools (`validation/`)

Tools for validating data quality and consistency.

- **check_alignment.py**: Verify alignment between attacked data and labels
- **check_time_format.py**: Check timestamp format consistency
- **check_time_order.py**: Verify chronological ordering and detect duplicates
- **compare_timestamps.py**: Compare timestamp sets between files

### 2. Processing Tools (`processing/`)

Tools for data transformation and cleaning.

- **clean_attacked_labels_pair.py**: Align attacked/labels files on timestamp intersection
- **sort_and_normalize_csv.py**: Sort and normalize timestamp formats

### 3. Reporting Tools (`reporting/`)

Tools for generating data quality reports.

- **case_report.py**: Comprehensive dataset quality report

### 4. Generation Tools (`generators/`)

Tools for generating synthetic datasets.

- **mixed_dataset_generator/**: Generate mixed normal/attack datasets

## Usage Examples

### Check Data Alignment

```bash
python validation/check_alignment.py \
    --attacked data/case1/case1_fdia_noisy.csv \
    --labels data/case1/case1_fdia_labels.csv
```

### Sort and Normalize Timestamps

```bash
python processing/sort_and_normalize_csv.py \
    --in_path data/raw.csv \
    --out_path data/normalized.csv \
    --dedup
```

### Generate Case Report

```bash
python reporting/case_report.py \
    --data_dir data \
    --case case118
```

### Clean and Align Label Pairs

```bash
python processing/clean_attacked_labels_pair.py \
    --attacked data/attacked.csv \
    --labels data/labels.csv \
    --out_dir data/cleaned \
    --case case1
```

## Development Notes

- All tools use common utilities from `common_utils.py` to avoid code duplication
- Scripts add parent directory to path for importing common utilities
- Timestamps are standardized to YYYY/MM/DD HH:MM format
- Tools follow consistent error reporting and validation patterns

## Requirements

- pandas
- Python 3.10+

## File Naming Conventions

Expected file naming patterns:
- Normal data: `{case}_acopf_all_rows_noisy.csv`
- Attacked data: `{case}_fdia_2025-07_2025-08_noisy.csv`
- Labels: `{case}_fdia_2025-07_2025-08_labels.csv`
