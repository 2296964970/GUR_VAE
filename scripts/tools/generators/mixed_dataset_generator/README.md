# Mixed Dataset Generator

Tool for generating mixed datasets from normal and attack data to evaluate model detection performance.

## Directory Structure

```
mixed_dataset_generator/
├── README.md                   # This document
├── generate_mixed_dataset.py   # Core generation script
├── batch_generate.py           # Batch generation script
├── verify_dataset.py           # Verification script
├── example.py                  # Usage example
└── test_generator.py           # Test suite
```

## Key Features

- ✅ **Preserve temporal order** - Maintains time sequence relationships without shuffling data
- ✅ **Random mixing** - Randomly selects time points as attack/normal data
- ✅ **Automatic label generation** - Normal samples have all labels as 1, attack samples use labels from file
- ✅ **Efficient implementation** - Optimized code for fast processing of large datasets
- ✅ **Batch generation** - Generate multiple datasets with different attack ratios at once
- ✅ **Complete validation** - Includes verification scripts to ensure data quality
- ✅ **Test coverage** - Complete test suite to ensure functionality

## Quick Start

### 1. Basic Usage

```bash
python generate_mixed_dataset.py \
    --normal-data path/to/normal_data.csv \
    --attack-data path/to/attack_data.csv \
    --attack-labels path/to/attack_labels.csv \
    --output-dir path/to/output \
    --attack-ratio 0.3
```

### 2. Usage in Python Code

```python
from generate_mixed_dataset import generate_mixed_dataset

mixed_data, mixed_labels = generate_mixed_dataset(
    normal_data_path="input/normal.csv",
    attack_data_path="input/attack.csv",
    attack_labels_path="input/labels.csv",
    output_dir="output/mixed",
    attack_ratio=0.3,
    random_seed=42
)
```

### 3. Batch Generate Multiple Datasets

```bash
python batch_generate.py \
    --normal-data path/to/normal_data.csv \
    --attack-data path/to/attack_data.csv \
    --attack-labels path/to/attack_labels.csv \
    --output-base-dir path/to/output \
    --attack-ratios 0.1 0.2 0.3 0.4 0.5
```

### 4. Verify Generated Dataset

```bash
python verify_dataset.py \
    --data output/mixed_data.csv \
    --labels output/mixed_labels.csv
```

### 5. Run Example

```bash
python example.py
```

### 6. Run Tests

```bash
python test_generator.py
```

## Parameters

### generate_mixed_dataset.py

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--normal-data` | str | Required | Path to normal data CSV file |
| `--attack-data` | str | Required | Path to attack data CSV file |
| `--attack-labels` | str | Required | Path to attack labels CSV file |
| `--output-dir` | str | Required | Output directory |
| `--attack-ratio` | float | 0.3 | Attack data ratio (between 0-1) |
| `--random-seed` | int | 42 | Random seed |

### batch_generate.py

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--normal-data` | str | Required | Path to normal data CSV file |
| `--attack-data` | str | Required | Path to attack data CSV file |
| `--attack-labels` | str | Required | Path to attack labels CSV file |
| `--output-base-dir` | str | Required | Base output directory |
| `--attack-ratios` | float[] | [0.1, 0.2, 0.3, 0.4, 0.5] | List of attack ratios |
| `--random-seed` | int | 42 | Random seed |

## Output Files

Each generation creates two files in the output directory:

1. **`mixed_data.csv`** - Mixed dataset
   - Contains mixture of normal and attack data
   - Maintains temporal order
   - Includes timestamp column and all feature columns

2. **`mixed_labels.csv`** - Label matrix
   - Each row corresponds to the same row in mixed_data.csv
   - Label values: 1 for normal, 0 for attacked
   - Normal samples: all feature labels are 1
   - Attack samples: attacked features have label 0, others are 1

## Label Description

### Normal Time Points
Label matrix is all 1s, indicating all features are normal:
```
timestamp,feature_1,feature_2,feature_3,...
2025-01-01 00:00,1,1,1,...
```

### Attack Time Points
Labels are obtained from attack label file, 0 indicates the feature is attacked:
```
timestamp,feature_1,feature_2,feature_3,...
2025-01-01 00:05,1,0,1,...
```

## Working Principle

1. **Read data**: Load normal data, attack data, and attack labels
2. **Determine mixing ratio**: Calculate normal/attack sample counts based on attack_ratio
3. **Random selection**: Randomly select which time points use attack data
4. **Sequential mixing**: Insert attack data at selected positions in temporal order
5. **Generate labels**: Normal samples have all labels as 1, attack samples use actual labels
6. **Save files**: Save mixed data and corresponding labels

## Temporal Order Explanation

**Important**: This tool maintains temporal order and does not shuffle time sequence relationships.

- Data is arranged in chronological order
- Randomly select certain time points as attack points
- Ensures temporal models can correctly learn time dependencies

## Example Output

```
Reading data...
Normal data: 17856 rows
Attack data: 17856 rows

Generating mixed dataset:
  Total samples: 17856
  Normal samples: 12500 (70.0%)
  Attack samples: 5356 (30.0%)
  Maintaining temporal order, randomly selecting time points as attack/normal

Dataset saved:
  Mixed data: output/mixed_data.csv
  Label data: output/mixed_labels.csv

Label statistics:
  Label 0 count (attacked): 373032
  Label 1 count (normal): 21089880
  Attack feature ratio: 1.74%

Temporal order verification:
  First timestamp: 2025/07/01 00:00
  Last timestamp: 2025/08/13 09:35
```

## Testing

Run test suite to verify tool functionality:

```bash
python test_generator.py
```

Test coverage includes:
- ✅ Basic dataset generation
- ✅ Temporal order preservation
- ✅ Different attack ratios
- ✅ Label integrity

## FAQ

### Q: Why not shuffle temporal order?
A: For time-series data (such as power system data), temporal order contains important temporal dependency relationships. Shuffling would break these relationships and affect model learning.

### Q: Will the attack ratio be exactly precise?
A: Due to rounding, the actual attack ratio may have minor differences from the set value (typically within 1%).

### Q: Can I reproduce results using the same random seed?
A: Yes. Using the same random seed and parameters will generate identical datasets.

### Q: What data formats are supported?
A: Currently supports CSV format, requiring a timestamp column with other columns as feature columns.

### Q: How to ensure data quality?
A: Use the verify_dataset.py script to validate generated datasets. It checks shape, labels, timestamp matching, etc.

## Dependencies

- pandas
- numpy
- pathlib (Python standard library)
- argparse (Python standard library)

## License

This tool is part of an academic research project.

## Changelog

### v1.0.0 (2025-10-14)
- ✨ Initial version
- ✅ Mixed data generation with temporal order preservation
- ✅ Batch generation functionality
- ✅ Verification tool
- ✅ Complete test suite
