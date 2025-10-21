# 1. Training

## 1.1 Training Command

```
python scripts/train.py \
  --data_dir input \
  --case case14 \
  --epochs 40 \
  --device cpu \
  --exp_name gru_base_ep40 \
  --time_length 96 \
  --stride 48 \
  --batch_size 64 \
  --mask_rate 0.3 \
  --mask_mode iid \
  --latent_dim 32 \
  --gru_hidden 256 \
  --gru_layers 1 \
  --beta 0.1 \
  --obs_init_logvar -3.5 \
  --learning_rate 3e-4
```

Notes
- Training data is located under `input/case14/train/` (2025/07–08 clean). Filenames may vary; do not use any path containing `fdia` or `2025-09` for training.
- Checkpoint saved to: `output/case14/models/gru_base_ep40/ckpt.pt`.

## 1.2 Training Results

- Device: CPU (CUDA requested but not available; fell back to CPU)
- Final validation loss: 11.0990 (Epoch 40)
- Best checkpoint: saved automatically on lowest validation loss during training


# 2. Inference

## 2.1 Inference Command

Defaults are set to the best configuration:
- `--time_length 96` (matches training; avoids short-context cold start)
- `--sliding_steps 12`
- `--alpha 0.02`
- `--stats_source training` (use mean/std from 07–08 training normal set)

Minimal command (auto‑detects 2025/09 normal/attacked CSVs under `input/case14/infer/`)
```
python scripts/tail_only_locate_and_repair.py \
  --case case14 \
  --attack_timestamp "2025/09/18 16:45" \
  --ckpt output/case14/models/gru_base_ep40/ckpt.pt
```

## 2.2 Inference Results (Metrics)

Best configuration (window=96, stats=training 07–08, alpha=0.02)

| Metric | Attacked | Repaired | Gain |
|---|---:|---:|---:|
| MSE | 0.006048 | 0.000657 | +89.13% |
| RMSE | 0.072774 | 0.025563 | +64.87% |
| MSE@Dropped (predicted anomalies only) | — | 0.000871 | — |
| RMSE@Dropped | — | 0.029266 | — |
| NRMSE@Dropped | — | 0.774588 | — |

Artifacts
- Repaired tail rows (12 timestamps):
  - `output/case14/repaired/*repaired_tail_rows_L96_steps12*.csv`
- Triplet for SE (12 timestamps, strict timestamp slicing only):
  - `D:/VsCodeProject/matlabproject/loadseries/output/case14/experiment/model_gated_recurrent_unit_baseline_epochs_40_alpha_0_0200/triplet_csv/normal_subset.csv`
  - `D:/VsCodeProject/matlabproject/loadseries/output/case14/experiment/model_gated_recurrent_unit_baseline_epochs_40_alpha_0_0200/triplet_csv/attacked_subset.csv`
  - `D:/VsCodeProject/matlabproject/loadseries/output/case14/experiment/model_gated_recurrent_unit_baseline_epochs_40_alpha_0_0200/triplet_csv/repaired_subset.csv`

This document is the baseline for future experiments.


# 3. State Estimation Results

Not executed in this iteration (focus: inference tuning). To run SE figures/metrics on the saved triplet, use the MATLAB pipeline under `loadseries/scripts/estimation/exper`.
