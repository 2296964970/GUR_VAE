# Data Processing Tools

Tools for cleaning, transforming, and normalizing datasets.

## Tools

### clean_attacked_labels_pair.py

Align attacked and labels CSV files on timestamp intersection.

**Features:**
- Normalizes timestamps to YYYY/MM/DD HH:MM
- Sorts by timestamp
- Removes duplicate timestamps
- Finds intersection of timestamps
- Produces aligned output files

**Usage:**
```bash
python clean_attacked_labels_pair.py \
    --attacked path/to/attacked.csv \
    --labels path/to/labels.csv \
    --out_dir path/to/output \
    --case case_name \
    [--normal path/to/normal.csv]
```

**Output files:**
- `{case}_fdia_2025-09_noisy.csv` - Cleaned attacked data
- `{case}_fdia_2025-09_labels.csv` - Cleaned labels (if applicable)
- Optional copy of normal file if provided

### sort_and_normalize_csv.py

Sort CSV by timestamp and normalize format.

**Features:**
- Parse and normalize timestamps to YYYY/MM/DD HH:MM
- Sort chronologically
- Optional deduplication

**Usage:**
```bash
python sort_and_normalize_csv.py \
    --in_path input.csv \
    --out_path output.csv \
    [--dedup]
```

**Options:**
- `--dedup`: Remove duplicate timestamps (keeps first occurrence)

## Workflow Examples

### Clean Attack/Label Pair
```bash
# Before: misaligned timestamps, duplicates, format inconsistencies
# After: aligned, normalized, deduplicated

python clean_attacked_labels_pair.py \
    --attacked raw/attacked.csv \
    --labels raw/labels.csv \
    --out_dir cleaned \
    --case case118
```

### Simple Normalization
```bash
# Normalize and deduplicate a single file
python sort_and_normalize_csv.py \
    --in_path messy_data.csv \
    --out_path clean_data.csv \
    --dedup
```

## Processing Pipeline

Recommended order for data cleaning:

1. **Validation**: Use validation tools to assess data quality
2. **Processing**: Apply appropriate processing tools
3. **Re-validation**: Verify processed data quality
4. **Reporting**: Generate quality report

```bash
# 1. Check initial quality
python ../validation/check_alignment.py --attacked raw_a.csv --labels raw_l.csv

# 2. Clean and align
python clean_attacked_labels_pair.py \
    --attacked raw_a.csv --labels raw_l.csv \
    --out_dir cleaned --case case1

# 3. Verify
python ../validation/check_alignment.py \
    --attacked cleaned/case1_fdia_2025-09_noisy.csv \
    --labels cleaned/case1_fdia_2025-09_labels.csv

# 4. Generate report
python ../reporting/case_report.py --data_dir cleaned --case case1
```
