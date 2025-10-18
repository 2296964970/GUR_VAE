# Data Generation Tools

Tools for generating synthetic datasets for experiments.

## Tools

### mixed_dataset_generator/

Generate mixed normal/attack datasets with configurable parameters.

See [mixed_dataset_generator/README.md](mixed_dataset_generator/README.md) for detailed documentation.

**Quick start:**
```bash
cd mixed_dataset_generator
python example.py
```

## Overview

The mixed dataset generator creates synthetic datasets that combine:
- Normal operational data
- Attack (FDIA) data
- Corresponding labels

This is useful for:
- Testing detection algorithms
- Creating balanced training/validation sets
- Simulating various attack scenarios
- Generating reproducible experimental datasets

## Common Patterns

### Single Dataset Generation
```bash
cd mixed_dataset_generator
python generate_mixed_dataset.py \
    --normal_csv path/to/normal.csv \
    --attack_csv path/to/attack.csv \
    --labels_csv path/to/labels.csv \
    --output_dir output \
    --config config.json
```

### Batch Generation
```bash
cd mixed_dataset_generator
python batch_generate.py --config_dir configs
```

### Verification
```bash
cd mixed_dataset_generator
python verify_dataset.py --dataset_dir path/to/generated
```

## Integration with Other Tools

After generating datasets, use validation and reporting tools:

```bash
# Generate dataset
cd generators/mixed_dataset_generator
python generate_mixed_dataset.py [options]

# Validate generated data
cd ../../validation
python check_alignment.py \
    --attacked ../generators/mixed_dataset_generator/output/mixed_attacked.csv \
    --labels ../generators/mixed_dataset_generator/output/mixed_labels.csv

# Generate quality report
cd ../reporting
python case_report.py \
    --data_dir ../generators/mixed_dataset_generator/output \
    --case mixed
```
