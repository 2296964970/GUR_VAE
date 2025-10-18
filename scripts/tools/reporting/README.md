# Report Generation Tools

Tools for generating comprehensive data quality reports.

## Tools

### case_report.py

Generate comprehensive quality report for a dataset case.

**Features:**
- Shape analysis for normal, attacked, and labels
- String vs time-based sorting validation
- Timestamp parseability check
- Duplicate detection
- Format mismatch identification with samples
- Alignment verification between files

**Usage:**
```bash
python case_report.py \
    --data_dir path/to/data \
    --case case_name
```

**Expected file structure:**
```
data_dir/
└── case_name/
    ├── {case}_acopf_all_rows_noisy.csv      # Normal data
    ├── {case}_fdia_2025-07_2025-08_noisy.csv # Attack data
    └── {case}_fdia_2025-07_2025-08_labels.csv # Labels
```

**Output example:**
```
NORMAL: (10000, 119) string_sorted:True time_sorted:True nat:0 dups:0 format_mismatch:0
ATTACK: (8000, 119) string_sorted:True time_sorted:True nat:0 dups:0 format_mismatch:0
LABELS: (8000, 119) string_sorted:True time_sorted:True nat:0 dups:0 format_mismatch:0
  format_mismatch_samples (original -> normalized) up to 5:
   - 2025/7/1 0:00 -> 2025/07/01 00:00
   - 2025/7/1 0:01 -> 2025/07/01 00:01
ALIGNMENT
  headers_equal: True
  raw_timestamps_equal: False
  normalized_timestamps_equal: True
```

## Report Metrics

### Per-file Metrics

- **Shape**: (rows, columns)
- **string_sorted**: Whether timestamps are sorted as strings
- **time_sorted**: Whether timestamps are chronologically sorted
- **nat**: Count of unparseable timestamps (NaT = Not a Time)
- **dups**: Count of duplicate timestamps
- **format_mismatch**: Count of timestamps not matching YYYY/MM/DD HH:MM format

### Alignment Metrics

- **headers_equal**: Column names match between attacked and labels
- **raw_timestamps_equal**: Raw timestamp strings match exactly
- **normalized_timestamps_equal**: Timestamps match after normalization

## Use Cases

### Pre-processing Assessment
```bash
# Check raw data quality before processing
python case_report.py --data_dir raw_data --case case118
```

### Post-processing Verification
```bash
# Verify data quality after cleaning
python case_report.py --data_dir cleaned_data --case case118
```

### Batch Quality Check
```bash
# Check multiple cases
for case in case14 case30 case57 case118; do
    echo "=== $case ==="
    python case_report.py --data_dir data --case $case
done
```

## Interpreting Results

### Good Quality Indicators
- ✓ string_sorted: True
- ✓ time_sorted: True
- ✓ nat: 0
- ✓ dups: 0
- ✓ format_mismatch: 0
- ✓ normalized_timestamps_equal: True

### Issues to Address
- ✗ time_sorted: False → Need to sort data
- ✗ nat > 0 → Invalid timestamps present
- ✗ dups > 0 → Need deduplication
- ✗ format_mismatch > 0 → Inconsistent formats
- ✗ normalized_timestamps_equal: False → Need alignment
