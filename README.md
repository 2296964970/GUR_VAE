# GRU-VAE: Generative Reconstruction for FDIA

End-to-end GRU-Variational Autoencoder (GRU-VAE) for power-system measurement streams. The model learns nominal trajectories and reconstructs attacked inputs by aligning them with the paired normal sequence at the same timestamps, using observed-only losses.

---

## Repository Layout
- `gru_vae/` Core library (data loaders, model, trainer, metrics, config).
- `scripts/train.py` Train on normal-only windows with FDIA noise injection (per-step sparse attacks) and observed-only NLL + KL.
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

- Data locations
- Training (2025/07–08): under `input/{case}/train/`
  - Normal CSV: `*_2025-07_2025-08_clean_noisy.csv` (used to fit and train with FDIA injection on-the-fly).
- Inference (2025/09): under `input/{case}/infer/`
  - Paired CSVs: `*_2025-09_clean_noisy.csv` (normal) and `*_fdia_2025-09_noisy.csv` (attacked).

---

## Training and Inference Policy (No Data Leakage)
- Training uses 2025/07–08 normal CSV only; FDIA attacks are simulated by per-step sparse Gaussian injection in standardized domain.
- Training normal CSV must not contain `fdia` nor `2025-09`. Any `2025-09` path is forbidden in training configuration.
- Standardization uses slot-wise robust (hour-of-day) statistics computed from the training normal split only and saved next to the checkpoint; these statistics are reused for inference (no re-fitting on 2025/09).

Observed-only reconstruction and masks
- The model consumes attacked inputs and computes NLL against the aligned normal targets.
- Only observed positions (mask == 1) contribute to loss/metrics; missing entries are not penalized.
- The observability mask is derived from `NaN` positions and is enforced identical across paired CSVs. Encoder inputs are mask-aware via concatenation: `[x*mask, mask]`.

---

## FDIA Injection (Training)

During training and validation, False Data Injection Attacks (FDIA) are simulated on-the-fly in the standardized domain.

- Notation: `x_target ∈ R^{B×T×H}` (standardized clean inputs), `m ∈ {0,1}^{B×T×H}` (observability mask).
- Per-step sparse attacks: for each batch `b` and time step `t`, pick a sparse subset of observed features to attack.
- Attack rate: fixed fraction `q = 0.15` (no config switch). If no observed feature is selected, force-select `ceil(q · N_obs)` (or 1 if `N_obs > 0`). If `N_obs=0`, fall back to full domain (will be masked out by `m`).
- Noise: i.i.d. Gaussian `ε ~ N(0, σ^2)` with `σ = noise.strength`.

Formula (elementwise Hadamard products):

```
Given x_target, m, strength σ, rate q, for each (b, t):
  obs_idx = { j | m[b,t,j] = 1 }
  A[b,t,:] = 0
  Sample S ⊆ obs_idx by Bernoulli(q); if S = ∅ and |obs_idx|>0, pick k=ceil(q·|obs_idx|) random indices; set A[b,t,S]=1
  Sample ε[b,t,:] ~ Normal(0, σ^2 I)
  x_input[b,t,:] = x_target[b,t,:] + ε[b,t,:] ⊙ A[b,t,:] ⊙ m[b,t,:]
```

- The model receives `x_input` and is optimized to reconstruct `x_target` using observed-only Gaussian NLL + KL.
- Training and validation both use the same injection scheme (with `noise.seed` for reproducibility).

Inference is unchanged: paired (attacked/normal) data from 2025/09 are used for evaluation only.

---

## Quickstart (Single Config)

All parameters are configured in a single `config.yaml` at the project root.

1) Configure
- Edit `config.yaml` and set:
  - `global.case`, `global.data_dir`
  - `train.normal_csv` (07–08)
  - `noise.strength`, `noise.seed` (training-time FDIA injection)
  - `infer.normal_csv`, `infer.attacked_csv` (09)
  - `model`, `train`, `window`

2) Train
```
python scripts/train.py
```
Artifacts go to `output/<case>/models/<exp_name>/`:
- `ckpt.pt` (with minimal hyperparameters)
- `slot_stats.npz` (slot-wise robust standardization stats for hour-of-day)
- `training_curve.tsv` (two lines: train loss, val loss sequences)

3) Inference (Reconstruction)
```
python scripts/infer_reconstruct.py
```
Outputs are written under `output/<case>/infer/<exp_name>/`:
- `reconstructed.csv` (full-series reconstruction on original scale)

---

## Model Overview
- Encoder: causal GRU produces per-step diagonal Gaussian `q(z_t|x_<=t)`.
- Prior: AR(1) state-space model with optional low-rank noise.
- Decoder: MLP outputs per-step Gaussian mean and log-variance.
- Training objective: observed-only Gaussian NLL + KL regularization (with warm-up).
  - KL warm-up: linear 0 → `beta` during the first ~20% of epochs; always enabled (no configuration switch).

---

## Output Directory Policy
- Training: `output/<case>/models/<exp_name>/`.
- Inference: `output/<case>/infer/<exp_name>/`.

---

## Encoding and Language
- All files use UTF-8 without BOM.
- Code/comments: English. Conversation with AI agents: Chinese (Simplified).

---

## Inference Specifics (Current Implementation)
- Full-series reconstruction: the script standardizes the entire 2025/09 attacked series, runs causal reconstruction, and unstandardizes to original scale.
- No window/MC: window selection and MC averaging are removed to keep the pipeline minimal.
- No passthrough of observed inputs: reconstructed values are model predictions at all positions.
- Standardization: uses slot-wise robust (hour-of-day) statistics computed from 2025/07–08 normal training split; no re-fitting on 2025/09.
- Terminal summary: prints Top-10 and Worst-10 timestamps by MSE repair effect (att - rep) on observed positions only.

## Configuration Notes
- `infer.mc_samples` (int): number of MC samples for inference averaging (default: 8).
- Encoder inputs are mask-aware via concatenation `[x*mask, mask]`; the mask derives from `NaN` positions and is enforced identical across paired CSVs.
- Decoder predicts time-varying mean and log-variance; a hard upper bound on log-variance prevents variance blow-up.

---

## License
Research use within the GRU-VAE FDIA defense project. See forthcoming license documentation for details.
