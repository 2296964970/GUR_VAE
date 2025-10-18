# GRU-VAE (Online)

Lightweight GRU-VAE for FDIA repair.

Install: `pip install -e .`

## Training

Example (IEEE14, CPU):

```
python scripts/train.py \
  --data_dir data \
  --case case14 \
  --time_length 96 \
  --stride 48 \
  --batch_size 64 \
  --epochs 40 \
  --exp_name gru_case14_ep40 \
  --device cpu
```

Artifacts are saved to `models/gru_case14_ep40/` (e.g., `ckpt.pt`, `training_curve.tsv`).

## Evaluation

Three CPU examples on IEEE14:

- Ideal labels, tail-only repair

```
python scripts/tail_only_label_repair.py \
  --data_dir data \
  --case case14 \
  --normal_csv data/case14/case14_acopf_all_rows_noisy.csv \
  --attacked_csv data/case14/case14_fdia_2025-07_2025-08_noisy.csv \
  --labels_csv data/case14/case14_fdia_2025-07_2025-08_labels.csv \
  --time_length 24 \
  --attack_timestamp "2025/08/01 00:00" \
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```

- Locate + Repair (tail-only, single step)

```
python scripts/tail_only_locate_and_repair.py \
  --data_dir data \
  --case case14 \
  --normal_csv data/case14/case14_acopf_all_rows_noisy.csv \
  --attacked_csv data/case14/case14_fdia_2025-07_2025-08_noisy.csv \
  --time_length 24 \
  --attack_timestamp "2025/08/01 00:00" \
  --sliding_steps 1 \
  --alpha 0.01 \
  --threshold_type per_feature \
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```

- Locate + Repair (tail-only, sliding multi-step)

```
python scripts/tail_only_locate_and_repair.py \
  --data_dir data \
  --case case14 \
  --normal_csv data/case14/case14_acopf_all_rows_noisy.csv \
  --attacked_csv data/case14/case14_fdia_2025-07_2025-08_noisy.csv \
  --time_length 24 \
  --attack_timestamp "2025/08/01 00:00" \
  --sliding_steps 12 \
  --alpha 0.01 \
  --threshold_type per_feature \
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```
