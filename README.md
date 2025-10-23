# GRU-VAE: Generative Reconstruction for FDIA

End-to-end GRU-Variational Autoencoder (GRU-VAE) for power-system measurement streams. The model learns nominal trajectories and reconstructs attacked inputs by aligning them with the paired normal sequence at the same timestamps, using observed-only losses.

---

## Repository Layout
- `gru_vae/` Core library (data loaders, model, trainer, metrics, config).
- `scripts/train.py` Train on paired attacked/normal windows with observed-only NLL + KL.
- `scripts/infer_reconstruct.py` Reconstruct attacked sequences over a specified window and report observed-space metrics.
- `input/` Example datasets and structure description.
- `output/` All generated artifacts.

---

## Installation
- Python >= 3.10
- PyTorch (CPU or CUDA)
- Install dependencies: `pip install -r requirements.txt`

---

## Datasets and Schema
- Files are UTF-8 without BOM and include a header row.
- First column: `timestimp` (string), normalized to `YYYY/MM/DD HH:MM`.
- Remaining columns: numeric features (float), `NaN` denotes missing.
- Feature dimensions by benchmark:
  - IEEE 14-bus: 144 features
  - IEEE 57-bus: 575 features
  - IEEE 118-bus: 1202 features
- Row count: 17,857 rows per file (1 header + 17,856 time steps).

Data locations (paired and aligned)
- Training (2025/07–08): under `input/{case}/train/`
  - Paired CSVs with identical timestamps: `*_2025-07_2025-08_clean_noisy.csv` (normal) and `*_fdia_2025-07_2025-08_noisy.csv` (attacked).
- Inference (2025/09): under `input/{case}/infer/`
  - Paired CSVs: `*_2025-09_clean_noisy.csv` (normal) and `*_fdia_2025-09_noisy.csv` (attacked).

---

## Training and Inference Policy (No Data Leakage)
- Training uses 2025/07–08 only with paired attacked/normal CSVs.
- Training normal CSV must not contain `fdia` nor `2025-09`.
- Training attacked CSV may contain `fdia` but must not contain `2025-09`.
- Any `2025-09` path is forbidden in training configuration.
- Standardization stats (`mean.npy`, `std.npy`) are computed from the training normal split only and saved next to the checkpoint.

Observed-only reconstruction loss
- The model consumes attacked inputs and computes NLL against the aligned normal targets.
- Only observed positions (mask == 1) contribute to loss/metrics.

---

## Quickstart (Single Config)

All parameters are configured in a single `config.yaml` at the project root. The loader enforces schema/timestamp alignment between normal and attacked CSVs.

1) Configure
- Edit `config.yaml` and set:
  - `global.case`, `global.data_dir`
  - `train.normal_csv`, `train.attacked_csv` (07–08)
  - `infer.normal_csv`, `infer.attacked_csv` (09)
  - `infer.end_timestamp` (inclusive end of window), `infer.length`
  - `model`, `train`, `window`

2) Train
```
python scripts/train.py
```
Artifacts go to `output/<case>/models/<exp_name>/`:
- `ckpt.pt` (with minimal hyperparameters)
- `mean.npy`, `std.npy` (training-normal statistics)
- `training_curve.tsv` (two lines: train loss, val loss sequences)

3) Inference (Reconstruction)
```
python scripts/infer_reconstruct.py
```
Outputs are written under `output/<case>/infer/<exp_name>/`:
- `metrics.csv` (per-timestamp MSE/MAE/RMSE/MAPE/MSPE on observed positions, plus a final mean row)
- `reconstructed_window.csv` (model reconstruction on original scale)
- `attacked_window.csv` (attacked slice for the same window)
- `normal_window.csv` (normal slice for the same window)

---

## Model Overview
- Encoder: causal GRU produces per-step diagonal Gaussian `q(z_t|x_<=t)`.
- Prior: AR(1) state-space model with optional low-rank noise.
- Decoder: MLP outputs per-step Gaussian mean and log-variance.
- Training objective: observed-only Gaussian NLL + β·KL (with warm-up).

---

## Output Directory Policy
- Training: `output/<case>/models/<exp_name>/`.
- Inference: `output/<case>/infer/<exp_name>/`.

---

## Encoding and Language
- All files use UTF-8 without BOM.
- Code/comments: English. Conversation with AI agents: Chinese (Simplified).

---

## License
Research use within the GRU-VAE FDIA defense project. See forthcoming license documentation for details.

