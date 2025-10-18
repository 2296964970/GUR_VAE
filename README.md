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

Note: training always applies synthetic corruption on selected masked positions (no zero-drop path).

Artifacts are saved to `models/gru_case14_ep40/` (e.g., `ckpt.pt`, `training_curve.tsv`).

## Evaluation

Examples on IEEE14:

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
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```

- FDIA inference (full timeline)

```
python scripts/fdia_infer.py \
  --data_dir data \
  --case case14 \
  --attacked_csv data/case14/case14_fdia_2025-07_2025-08_noisy.csv \
  --time_length 24 \
  --alpha 0.01 \
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```

- Simulated attacks (point/window) validation

```
python scripts/validate_simulated_attacks.py \
  --data_dir data \
  --case case14 \
  --time_length 24 \
  --attack_mode point \
  --point_rate 0.005 \
  --noise_kind gaussian \
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```

Training supports `--mask_mode` aliases: `point` (same as `iid`) and `window` (same as `block`).

## Notes

- Inference/localization entry points:
  - `scripts/fdia_infer.py` for full-timeline localization+repair on attacked CSVs
  - `scripts/validate_simulated_attacks.py` to simulate attacks and evaluate
  - Tail-only evaluation via `scripts/tail_only_locate_and_repair.py`

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
  --ckpt models/gru_case14_ep40/ckpt.pt \
  --device cpu
```
