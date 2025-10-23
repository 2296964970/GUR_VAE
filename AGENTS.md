**Branch Note:** The experimental-playground branch is reserved for experimental ideas and may introduce large structural changes.

# Agent Constraints and Guidelines

This document defines coding standards, language conventions, and technical constraints for the GRU-VAE project.

---

## File Encoding

**All files MUST use UTF-8 encoding without BOM.**

- Python source files: UTF-8
- Configuration files (JSON, YAML, TOML): UTF-8
- Data files (CSV, TXT): UTF-8
- Documentation (Markdown, RST): UTF-8

**Rationale**: Ensures cross-platform compatibility and consistent handling of international characters.

---

## Language Conventions

### Human-Agent Communication
- **Language**: Chinese (Simplified)
- **Context**: All conversations, explanations, and clarifications between users and AI agents MUST be conducted in Chinese.

### Code and Documentation
- **Code**: English only
  - Variable names: English (e.g., `time_length`, `alpha`, `pred_mask`)
  - Function names: English (e.g., `compute_tail_scores`, `calibrate_thresholds`)
  - Class names: English (e.g., `OnlineGPVAE`, `SlidingWindowDataset`)

- **Comments**: English
  - Inline comments: English
  - Docstrings: English
  - Type hints: Standard Python/English conventions

- **Documentation**: English
  - README files: English
  - API documentation: English
  - Technical specifications: English 



**Rationale**: English code ensures broader accessibility, easier collaboration with international developers, and compatibility with most coding standards and linters.

---

## Character Encoding in Code

### Preferred Characters
- **Identifiers (variables, functions, classes)**: ASCII only (`[a-zA-Z0-9_]`)
- **String literals**: UTF-8 supported, but prefer ASCII where possible
- **File paths in code**: Use forward slashes `/` or `os.path.join()` for cross-platform compatibility

---

## Project Background

State estimation underlies operational decision making in modern power systems, yet supervisory control and data acquisition (SCADA) telemetry is increasingly exposed to false data injection attacks (FDIAs). Traditional residual-based detectors fail when attackers exploit knowledge of network topology or measurement redundancy, allowing corrupted observations to pass undetected and destabilize downstream automation. The GRU-VAE project explores sequence-aware generative models as a defense mechanism, leveraging recurrent encoders and variational latent spaces to learn the manifold of clean operating trajectories. By modeling nominal dynamics directly from physics-informed simulations, the framework targets accurate localization and repair of stealthy FDIAs without relying on attacker signatures or labeled anomalies.

## Current Project Description

The repository implements an end-to-end GRU-Variational Autoencoder pipeline tailored to IEEE benchmark grids. The codebase now uses explicit paired datasets (attacked and normal) with strict timestamp alignment for both training and inference. Data utilities in `gru_vae/data.py` provide:

- `load_paired_timeseries(normal_csv, attacked_csv)`: validates identical headers, shapes, timestamps, and observability masks.
- `create_paired_loaders(...)`: builds sliding-window datasets where the model takes attacked inputs and learns to reconstruct the aligned normal targets.

Training optimizes a causal GRU encoder + Gaussian decoder with KL regularization. The supervised ELBO uses observed-only reconstruction loss: only positions marked observable contribute to the negative log-likelihood. Inference scripts reconstruct attacked sequences over a user-specified time window and report observed-space metrics against the aligned normal slice. A single `config.yaml` controls data paths, windowing, model hyperparameters, training schedule, and inference windowing.

## Experimental Scope
Experiments are conducted on three benchmark power system models: IEEE14, IEEE57, and IEEE118. For each system, five datasets provide complementary perspectives on normal operation, attacked measurements, and idealized ground-truth labels.

## Dataset Overview
- **Source and structure:** Each dataset is generated from optimal power flow calculations on the corresponding IEEE system. The CSV files share an identical schema with a header row. The first column is `timestimp`, followed by feature columns (144 for IEEE14, 575 for IEEE57, and 1202 for IEEE118). IEEE118 files contain 17,857 rows (including the header), with 17,856 chronological time steps; IEEE14 and IEEE57 follow the same row count but adjust the feature dimension.
- **Training data (2025/07-08):** Clean baseline measurements for July-August 2025 (normal only).
- **Inference data (2025/09):**
  - Normal: Clean baseline measurements for September 2025 (all time steps are normal).
  - FDIA: Measurements corrupted by FDIAs (all time steps are attacked).

---

## Training and Inference Data Policy (No Data Leakage)

To prevent leakage across calendar periods while supporting supervised reconstruction, the current implementation uses paired attacked/normal data from 2025/07–2025/08 for training and reserves 2025/09 for evaluation only.

- Training data (2025/07–08, paired and aligned)
  - Normal pattern (example): `input/{case}/train/*_2025-07_2025-08_clean_noisy.csv`
  - Attacked pattern (example): `input/{case}/train/*_fdia_2025-07_2025-08_noisy.csv`
  - The two CSVs must have identical headers and timestamps and share the same observability mask.
  - Loss is computed only on observed positions, with the model learning to map attacked inputs to normal targets.

- Inference data (2025/09, do NOT use for training)
  - Normal pattern (example): `input/{case}/infer/*_2025-09_clean_noisy.csv`
  - Attacked pattern (example): `input/{case}/infer/*_fdia_2025-09_noisy.csv`
  - Inference selects a window by `infer.end_timestamp` and `infer.length` in `config.yaml`, reconstructs attacked inputs, and evaluates metrics against the aligned normal slice on observed positions only.

Enforcement guidance
- Training uses `gru_vae/data.py:create_paired_loaders`, which expects explicit normal and attacked CSVs for 07–08 and validates alignment.
- Configuration validation in `gru_vae/config.py:_validate_training_policy` forbids any `2025-09` paths in training and forbids `fdia` in the training normal CSV. Attacked CSVs are allowed to contain `fdia` only for the 07–08 training period.
- Both CSVs must share schema and timestamps; the loader aborts on mismatch.

Rationale
- Supervised pairing (attacked→normal) trains the model to reconstruct nominal trajectories while preserving generalization to 09 by isolating evaluation to the inference period. Observed-only losses avoid penalizing unobserved entries.
