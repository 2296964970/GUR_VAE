# TCN-VAE: Generative Reconstruction for FDIA

End-to-end Temporal Convolutional Network Variational Autoencoder (TCN-VAE) for power-system measurement streams. The model learns nominal trajectories and reconstructs attacked inputs by aligning them with paired normal sequences at identical timestamps, using observed-only losses. The pipeline is streamlined: two-phase training on normal-only data and metrics-only inference (no CSV outputs).

---

## Repository Layout
- `tcn_vae/` Core library (data loaders, model, trainer, metrics, config).
- `scripts/train.py` Two-phase training on normal-only windows: Phase-1 identity pretraining (ELBO only), Phase-2 robust training (sparse contiguous Laplace attacks + anchor loss).
- `scripts/infer_reconstruct.py` Full-series reconstruction for September paired data; prints observed-space metrics only.
- `scripts/infer_reconstruct_mc_median16.py` MC-median reconstruction variant for experimentation; adapted to the new config loader.
- `scripts/verify_case118_logistic_issue.py`, `scripts/verify_overrepair_mc_median16.py` Utilities to assess over-repair on clean sequences; adapted to the new config loader.
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
- Feature dimensions:
  - IEEE 14-bus: 144
  - IEEE 57-bus: 575
  - IEEE 118-bus: 1202
- Row count: 17,857 rows per file (1 header + 17,856 time steps).

Data locations
- Training (2025/07–08): `input/{case}/train/`
  - Normal CSV only: e.g., `*_2025-07_2025-08_clean_noisy.csv`.
- Inference (2025/09): `input/{case}/infer/`
  - Paired CSVs: `*_2025-09_clean_noisy.csv` (normal) and `*_fdia_2025-09_noisy.csv` (attacked).

---

## Training and Inference Policy (No Data Leakage)
- Training uses 2025/07–08 normal CSV only; no September files are used for training.
- Training normal CSV must not contain `fdia` nor `2025-09` (enforced by config validation).
- Standardization uses slot-wise robust (hour-of-day) statistics computed from the training split and saved next to the checkpoint; reused for inference without re-fitting.

Observed-only reconstruction and masks
- Only observed positions (mask == 1) contribute to loss/metrics; missing entries are not penalized.
- Observability mask is derived from `NaN` positions and is enforced identical across paired CSVs.
- Encoder inputs are mask-aware via concatenation: `[x*mask, mask]`.

---

## Training Scheme (Two Phases)

- Phase-1: Identity Pretraining (Normal → Normal)
  - Use only clean normal windows; no synthetic attacks.
  - Objective: ELBO (observed-only Gaussian NLL + KL with linear warm-up).

- Phase-2: Robust Training (Sparse, Contiguous Laplace Attacks + Anchor)
  - For each batch sample, with probability `robust.clean_fraction`, keep the whole window clean (strong fidelity constraint).
  - Otherwise sample a single contiguous time segment `L ∈ [seg_len_min, seg_len_max]` and a feature subset fraction in `[dims_fraction_min, dims_fraction_max]`.
  - Add Laplace(0, b) offsets on observed entries within that rectangle, with `b` drawn from `robust.laplace_scales`.
  - Add anchor loss on non-attacked observed positions: `λ · mean((ŷ - x)^2 | a=0, m=1)` with `λ = train.anchor_lambda`, suppressing over-repair on clean inputs.

Inference uses September paired normal/attacked CSVs and prints observed-space metrics. No synthetic noise is used during inference.

---

## Quickstart (Single Config)

All parameters are configured in a single `config.yaml` at the project root.

1) Configure
- Edit `config.yaml` and set:
  - `global.case`
  - `train.normal_csv`, `train.epochs`, `train.phase1_epochs`, `train.phase2_epochs`, `train.anchor_lambda`
  - `robust.clean_fraction`, `robust.seg_len` (min,max), `robust.dims_fraction` (min,max), `robust.laplace_scales`
  - `infer.normal_csv`, `infer.attacked_csv`
  - `model`, `window`, `preprocess`

2) Train
```
python scripts/train.py
```
Artifacts go to `output/<case>/models/<exp_name>/`:
- `ckpt.pt` (with minimal hyperparameters)
- `slot_stats.npz` (slot-wise robust standardization stats for hour-of-day)
- `training_curve.tsv` (four lines: P1 train, P1 val, P2 train, P2 val)

3) Inference (Metrics Only)
```
python scripts/infer_reconstruct.py
```
Console output includes:
- Reconstruction metrics (Observed Only): top/worst timestamps by MSE improvement (att - rep)
- NRMSE metrics (Observed Only): attacked vs normal; repaired vs normal; optional clean self vs normal
No CSVs are written during inference.

---

## Model Overview
- Encoder: causal TCN produces per-step diagonal Gaussian `q(z_t|x_<=t)`.
- Prior: AR(1) state-space model with low-rank noise (always enabled).
- Decoder: MLP outputs per-step Gaussian mean and log-variance.
- Training objective: observed-only Gaussian NLL + KL regularization (with warm-up).

---

## Output Directory Policy
- Training: `output/<case>/models/<exp_name>/`.
- Inference: `output/<case>/infer/<exp_name>/`.

---

## Encoding and Language
- All files use UTF-8 without BOM.
- Code/comments: English. Conversation with AI agents: Chinese (Simplified).

---

## Inference Specifics
- The script standardizes the full 2025/09 series, reconstructs with the posterior mean path and decoder variance, applies confidence-guided blending on observed entries, then unstandardizes to original scale.
- A separate MC-median script is provided for experimentation; both follow the same metric definitions.
- Terminal summary prints top/worst timestamps by MSE improvement and global NRMSE statistics. No files are written.

## Configuration Notes
- `model.tcn_channels`: comma-separated string or YAML list (e.g., "256,256,256").
- `model.tcn_kernel_size`: positive int (e.g., 3).
- `model.tcn_dropout`: float in [0,1] (default 0.0).
- Encoder inputs use concatenation `[x*mask, mask]`; the mask is identical across paired CSVs.
- Decoder predicts time-varying mean and log-variance with clamped bounds.

---

## License
Research use within the TCN-VAE FDIA defense project. See forthcoming license documentation for details.

