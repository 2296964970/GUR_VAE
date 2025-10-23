# GRU-VAE: Sequence-Aware FDIA Localization and Repair

End-to-end GRU-Variational Autoencoder pipeline for power-system measurement streams. The model learns the manifold of clean operating trajectories from physics-informed simulations and repairs stealth FDIA (False Data Injection Attack) contaminations at inference time.

Key features
- Sliding-window recurrent encoder-decoder with variational latent space.
- Self-supervised masking with corruption on masked inputs.
- Tail-only locate-and-repair utility for attacked streams (no external labels required).
- Strict training data policy to prevent leakage.

---

## Repository Layout
- `gru_vae/` Core library (data loaders, model, trainer, metrics).
- `scripts/train.py` Train on clean baselines only.
- `scripts/tail_only_locate_and_repair.py` Calibrate thresholds on clean tail, detect + repair attacked tails.
- `scripts/tools/` Validation and preprocessing helpers for CSVs.
- `interop/` Optional MATLAB-Python interop helpers.
- `input/` Example datasets and structure description.
- `output/` All generated artifacts (see Output Policy below).

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

Data locations (do not rely on exact filenames)
- Training (2025/07–08, clean only): under `input/{case}/train/`
  - Contains clean baselines for July–August. Filenames may vary; no path should contain `fdia` or `2025-09`.
- Inference (2025/09): under `input/{case}/infer/`
  - Contains two CSVs with identical timestamps: one non‑`fdia` (clean baseline), one with `fdia` (attacked).
  - Scripts can auto‑detect these when `--normal_csv`/`--attacked_csv` are omitted.

---

## Training Policy (No Data Leakage)
- Train only on the 2025/07–08 clean baseline located under `input/{case}/train/`.
- Never pass any path containing `fdia` or `2025-09` to training utilities (guardrails enforce this).
- Attacked CSVs are used only for inference and evaluation.

---

## Quickstart (Single Config)

All parameters are configured in a single `config.yaml` at the repo root. No command‑line flags are required or supported by the scripts anymore.

1) Configure
- Open `config.yaml` and adjust:
  - `global.case`, `global.data_dir`
  - `data.*` paths (or leave blank to auto-detect where supported)
  - `model`, `train`, `window`, `mask`, `noise`
  - `inference_tail.attack_timestamp` for the target attacked time

2) Train
```
python scripts/train.py
```
Artifacts go to `output/<case>/models/<exp_name>/`.

3) Tail-only Locate + Repair
```
python scripts/tail_only_locate_and_repair.py
```
Notes
- If `data.normal_csv` or `data.attacked_csv` is empty, the script searches under `input/<case>/infer/` for a clean 2025/09 file (non‑`fdia`) and an attacked 2025/09 file (`fdia`).
Outputs are written under `output/<case>/...`:
- `output/<case>/repaired/` — repaired tail rows CSV, and optional wide-format repaired/attack/true rows.
- `output/<case>/tail_scores/` — wide-format tail scores per step when enabled.

---

## Interop with MATLAB (optional)
Set `matlab_eval.*` fields in `config.yaml` and run:
```
python scripts/run_state_estimation_triplet.py
```
See `interop/README.md` for repository-specific MATLAB details.

---

## Output Directory Policy (Mandatory)

All script outputs must be written under the repository `output/` directory, grouped by power-system case and artifact type.

- Case folder: `output/<case>/`
- Subfolders (created on demand):
  - `models/` — training checkpoints, logs and curves.
  - `repaired/` — repaired tail rows and wide-format repaired/attack/true rows.
  - `thresholds/` — per-feature threshold CSVs.
  - `tail_scores/` — wide-format tail scores (per step; score/threshold/keep_pred/is_anom).

Enforcement
- Training: `scripts/train.py` saves to `output/<case>/models/<exp_name>/`.
- Inference: `scripts/tail_only_locate_and_repair.py` always writes thresholds, tail scores, and repaired rows into the corresponding `output/<case>/...` subfolders regardless of input CSV locations. Legacy options for custom output folders are removed.

---

## Encoding and Language
- All files use UTF-8 without BOM.
- Code/comments: English. Conversation with AI agents: Chinese (Simplified).

---

## License
Research use within the GRU-VAE FDIA defense project. See forthcoming license documentation for details.
