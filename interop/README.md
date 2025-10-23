# MATLAB-Python Interop (GRU-VAE)

Purpose
- Bridge MATLAB time-series (e.g., loadseries) with this Python GRU-VAE pipeline for training and reconstruction-based evaluation.

Data Contract (CSV)
- First column: timestamp string, normalized to `YYYY/MM/DD HH:MM`.
- Remaining columns: numeric features (float). Use `NaN` for missing.
- Header row required; feature names must be ASCII identifiers.
- File encoding: UTF-8 without BOM.
- Training vs Inference data:
  - Training (2025/07-08, normal only): `input/{case}/{case}_acopf_2025-07_2025-08_noisy.csv`
  - Inference (2025/09):
    - Normal: `input/{case}/{case}_acopf_2025-09_noisy.csv`
    - Attacked: `input/{case}/{case}_fdia_2025-09_noisy.csv`

MATLAB Export
- Use `interop/matlab/export_to_gru_vae_csv.m` to export arrays to a compliant CSV.
- Example:
  - `[T,H]` numeric matrix `X`, vector `t` (datetime or serial date), `featureNames` cellstr.
  - `export_to_gru_vae_csv(t, X, featureNames, 'input/case14/case14_acopf_2025-07_2025-08_noisy.csv');`

Preprocess & Validate (optional but recommended)
- Normalize/Sort timestamps in CSV:
  - `python scripts/tools/processing/sort_and_normalize_csv.py --in_path <in.csv> --out_path <out.csv> --dedup`
- Validate time format/order:
  - `python scripts/tools/validation/check_time_format.py <csv>`
  - `python scripts/tools/validation/check_time_order.py <csv>`

Training (Python)
- Install deps: `pip install -r requirements.txt`
- Train on paired CSVs (attacked as input, normal as target, July–August only):
  - Configure `config.yaml` with `train.normal_csv` and `train.attacked_csv`.
  - Run: `python scripts/train.py`
  - Checkpoint saved to `output/<case>/models/<exp_name>/ckpt.pt` with `mean.npy/std.npy`.

Windowed Reconstruction (Python CLI)
- Script: `scripts/infer_reconstruct.py`
- Example:
  - Configure `infer.normal_csv`, `infer.attacked_csv`, `infer.end_timestamp`, `infer.length` in `config.yaml`.
  - Run: `python scripts/infer_reconstruct.py`
- Outputs under `output/<case>/infer/<exp_name>/`:
  - `metrics.csv` with per-timestamp MSE/MAE/RMSE/MAPE/MSPE and a bottom mean row.
  - `reconstructed_window.csv`, `attacked_window.csv`, `normal_window.csv`.

MATLAB Wrapper (optional)
- Example templates in `interop/matlab/` can be adapted to call `scripts/train.py` and `scripts/infer_reconstruct.py` via `system()`.

Environment Notes
- Ensure Python is visible to MATLAB (`pyenv`), or pass full path to the `python` executable in `run_tail_repair`.
- On Windows, keep double-quoted paths; this README uses forward slashes which Python supports on Windows.

Training Data Policy
- Train on 2025/07–08 data. Training normal CSV must not contain `fdia` or `2025-09`. Training attacked CSV may include `fdia` but must not contain `2025-09`.

