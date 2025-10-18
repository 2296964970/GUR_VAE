# Experiment Results - October 17, 2025

**Date & Time**: 2025-10-17 16:45

## Training Configuration

### Model Training (Case14)
- **Dataset**: case14
- **Time Length**: 96
- **Stride**: 48
- **Batch Size**: 64
- **Epochs**: 40
- **Experiment Name**: gru_case14_ep40
- **Device**: cuda (fallback to CPU)
- **Model Path**: `models\gru_case14_ep40\ckpt.pt`

### Training Results

| Epoch | Train Loss | Train NLL | Train KL | Val Loss | Val NLL | Val KL | MSE Miss Val |
|-------|-----------|-----------|----------|----------|---------|--------|--------------|
| 001 | 17.7444 | 15.6428 | 21.0163 | 19.8278 | 17.9455 | 18.8237 | 1.135565 |
| 010 | 12.8072 | 12.4549 | 3.5230 | 13.3268 | 12.9571 | 3.6978 | 0.840400 |
| 020 | 11.0782 | 10.8457 | 2.3256 | 11.5860 | 11.3171 | 2.6890 | 0.752394 |
| 030 | 10.8867 | 10.6757 | 2.1100 | 11.3037 | 11.0679 | 2.3579 | 0.740939 |
| 040 | 10.3665 | 10.1708 | 1.9577 | 11.2983 | 11.0980 | 2.0029 | 0.755623 |

**Final Performance**:
- Train Loss: 10.3665 (NLL: 10.1708, KL: 1.9577)
- Validation Loss: 11.2983 (NLL: 11.0980, KL: 2.0029)
- MSE Missing Value: 0.755623

---

## Experiment 1: Tail-Only Label Repair

### Configuration
- **Script**: `tail_only_label_repair.py`
- **Normal Data**: `data/case14/case14_acopf_all_rows_noisy.csv`
- **Attacked Data**: `data/case14/case14_fdia_2025-07_2025-08_noisy.csv`
- **Labels**: `data/case14/case14_fdia_2025-07_2025-08_labels.csv`
- **Time Length**: 24
- **Attack Timestamp**: 2025/08/01 17:00
- **Checkpoint**: `models/gru_case14_ep40/ckpt.pt`
- **Device**: cpu

### Attack Details
- **Window**: s=9109, e=9133
- **Tail Index**: 9132
- **Timestamp**: 2025/08/01 17:00
- **Attacked Tail Features**: 46

### Performance Metrics

| Metric | Baseline | Repaired | Unit |
|--------|----------|----------|------|
| **MSE** | 0.143242 | 0.000628 | - |
| **MAE** | 0.270985 | 0.015364 | - |
| **RMSE** | 0.378473 | 0.025052 | - |
| **NRMSE** | 14.290922 | 0.667082 | % |
| **NMAE** | 8.242787 | 0.456316 | % |

### Improvement
- **MSE Improvement**: 99.56%
- **RMSE Improvement**: 93.38%

---

## Experiment 2: Tail-Only Locate and Repair

### Configuration
- **Script**: `tail_only_locate_and_repair.py`
- **Normal Data**: `data/case14/case14_acopf_all_rows_noisy.csv`
- **Attacked Data**: `data/case14/case14_fdia_2025-07_2025-08_noisy.csv`
- **Time Length**: 24
- **Attack Timestamp**: 2025/08/01 17:00
- **Sliding Steps**: 1
- **Alpha**: 0.01
- **Threshold Type**: per_feature
- **Checkpoint**: `models/gru_case14_ep40/ckpt.pt`
- **Device**: cpu

### Repair Details
- **Mode**: history reuses all previously repaired tails (others from Normal)
- **Tail Index**: 9132
- **Timestamp**: 2025/08/01 17:00
- **Window**: s=9109, e=9133

### Detection and Repair Results

| Step | Tail Idx | Timestamp | Observed | Dropped | Drop Rate (%) |
|------|----------|-----------|----------|---------|---------------|
| 0 | 9132 | 2025/08/01 17:00 | 144 | 41 | 28.47 |

### Performance Metrics

| Metric | Attacked | Repaired |
|--------|----------|----------|
| **MSE** | 0.045802 | 0.000393 |
| **RMSE** | 0.214015 | 0.019812 |
| **NRMSE** | 8.084038 | 0.631094 |

### Improvement
- **MSE Improvement**: 99.14%
- **RMSE Improvement**: 90.74%

### Output
- **Repaired Data**: `data/case14\case14_fdia_2025-07_2025-08_noisy_repaired_tail_rows_L24_steps1.csv` (1 row)

---

## Summary

Both repair methods demonstrate excellent performance in recovering attacked data:

1. **Tail-Only Label Repair** (with labels):
   - Achieves 99.56% MSE improvement
   - Repairs all 46 attacked features
   - Uses ground truth labels for validation

2. **Tail-Only Locate and Repair** (blind detection):
   - Achieves 99.14% MSE improvement
   - Successfully detects 28.47% of features as attacked (41 out of 144 observed)
   - No labels required during repair process

Both methods show that the trained GRU-VAE model (40 epochs) is capable of accurately reconstructing attacked power system measurements with over 90% RMSE improvement.
