# MATLAB-Python Interop (GRU-VAE)

Purpose
- Bridge MATLAB time-series (e.g., loadseries) with this Python GRU-VAE pipeline for training and FDIA repair.

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
- Train strictly on Normal CSV (no FDIA paths):
  - `python scripts/train.py --data_dir input --case case14 --epochs 40 --device cpu`
  - Checkpoint saved to `output/<case>/models/<exp_name>/ckpt.pt`.

Tail-Only Repair (Python CLI)
- CLI: `scripts/tail_only_locate_and_repair.py`
- Minimal example:
  - `python scripts/tail_only_locate_and_repair.py --data_dir input --case case14 --normal_csv input/case14/case14_acopf_2025-09_noisy.csv --attacked_csv input/case14/case14_fdia_2025-09_noisy.csv --time_length 24 --sliding_steps 6 --attack_timestamp "2025/09/14 12:00" --ckpt output/case14/models/gru_base_ep40/ckpt.pt --tail_scores_wide`
- Outputs:
  - Repaired tail rows CSV under `output/<case>/repaired/` (one row per step).
  - Optional wide CSVs written to `output/<case>/tail_scores/` (scores/threshold/keep_pred/is_anom) and to `output/<case>/repaired/` (attacked/true/repaired tails) when `--tail_scores_wide` is set.

MATLAB Wrapper (system call)
- Use `interop/matlab/run_tail_repair.m` to invoke the Python CLI from MATLAB.
- Example:
  - `[status, cmdout] = run_tail_repair('python', pwd, ...
      'DataDir','input', 'Case','case14', 'NormalCsv','input/case14/case14_acopf_2025-09_noisy.csv', ...
      'AttackedCsv','input/case14/case14_fdia_2025-09_noisy.csv', ...
      'Ckpt','output/case14/models/gru_base_ep40/ckpt.pt', 'AttackTimestamp','2025/09/14 12:00', ...
      'TimeLength',24, 'SlidingSteps',6, 'TailScoresWide',true);`

Environment Notes
- Ensure Python is visible to MATLAB (`pyenv`), or pass full path to the `python` executable in `run_tail_repair`.
- On Windows, keep double-quoted paths; this README uses forward slashes which Python supports on Windows.

Training Data Policy
- Do not train on any `fdia` CSV. Training uses only the clean baseline file.

