# Data Validation Tools

Tools for validating data quality and consistency.

## Tools

### check_alignment.py

Verify alignment between attacked data and labels.

**Features:**
- Check header consistency
- Compare timestamp alignment (raw and normalized)
- Verify shape matching
- Detect misalignment issues

**Usage:**
```bash
python check_alignment.py \
    --attacked path/to/attacked.csv \
    --labels path/to/labels.csv
```

**Output:**
- Reports header equality
- Shows raw timestamp alignment status
- Shows normalized timestamp alignment status
- Indicates whether files are properly aligned

### check_time_format.py

Check timestamp format consistency.

**Features:**
- Parse timestamps and normalize to YYYY/MM/DD HH:MM
- Identify unparseable timestamps (NaT)
- Detect format inconsistencies
- Report format mismatch statistics

**Usage:**
```bash
python check_time_format.py --csv path/to/data.csv
```

**Output:**
- Count of unparseable timestamps
- Count of format mismatches
- Sample mismatched formats with corrections

### check_time_order.py

Verify chronological ordering and detect duplicates.

**Features:**
- Check if timestamps are sorted as strings
- Check if timestamps are chronologically sorted
- Count duplicate timestamps
- Identify sorting and duplication issues

**Usage:**
```bash
python check_time_order.py --csv path/to/data.csv
```

**Output:**
- String sorting status
- Chronological sorting status
- Duplicate count

### compare_timestamps.py

Compare timestamp sets between files.

**Features:**
- Load timestamps from two CSV files
- Normalize both to YYYY/MM/DD HH:MM format
- Find intersection and differences
- Report timestamp set relationships

**Usage:**
```bash
python compare_timestamps.py \
    --csv1 path/to/file1.csv \
    --csv2 path/to/file2.csv
```

**Output:**
- Timestamp counts for each file
- Intersection size
- Timestamps only in file1
- Timestamps only in file2

## Validation Workflow

Recommended validation sequence:

### 1. Individual File Validation
```bash
# Check time order
python check_time_order.py --csv data/attacked.csv

# Check time format
python check_time_format.py --csv data/attacked.csv
```

### 2. Paired File Validation
```bash
# Check alignment between attacked and labels
python check_alignment.py \
    --attacked data/attacked.csv \
    --labels data/labels.csv

# Compare timestamps
python compare_timestamps.py \
    --csv1 data/attacked.csv \
    --csv2 data/labels.csv
```

### 3. Multi-file Validation
```bash
# Compare normal vs attacked timestamps
python compare_timestamps.py \
    --csv1 data/normal.csv \
    --csv2 data/attacked.csv
```

## Common Issues and Solutions

### Issue: Timestamps Not Aligned
```bash
# Use check_alignment.py to identify the problem
python check_alignment.py --attacked a.csv --labels l.csv

# Solution: Use processing tools to align
python ../processing/clean_attacked_labels_pair.py \
    --attacked a.csv --labels l.csv --out_dir cleaned --case case1
```

### Issue: Time Not Sorted
```bash
# Use check_time_order.py to verify
python check_time_order.py --csv data.csv

# Solution: Use processing tools to sort
python ../processing/sort_and_normalize_csv.py \
    --in_path data.csv --out_path sorted.csv
```

### Issue: Format Inconsistencies
```bash
# Use check_time_format.py to identify
python check_time_format.py --csv data.csv

# Solution: Use processing tools to normalize
python ../processing/sort_and_normalize_csv.py \
    --in_path data.csv --out_path normalized.csv
```

## Integration with Other Tools

### Pre-processing Pipeline
```bash
# 1. Validate raw data
python validation/check_alignment.py --attacked raw_a.csv --labels raw_l.csv

# 2. Process if needed
python processing/clean_attacked_labels_pair.py \
    --attacked raw_a.csv --labels raw_l.csv \
    --out_dir cleaned --case case1

# 3. Re-validate processed data
python validation/check_alignment.py \
    --attacked cleaned/case1_fdia_2025-07_2025-08_noisy.csv \
    --labels cleaned/case1_fdia_2025-07_2025-08_labels.csv

# 4. Generate quality report
python reporting/case_report.py --data_dir cleaned --case case1
```

## Requirements

- pandas
- Python 3.10+

## Notes

- All tools use common utilities from `../common_utils.py`
- Timestamps are normalized to YYYY/MM/DD HH:MM format for comparison
- Tools follow consistent error reporting patterns
