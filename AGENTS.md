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

The repository implements an end-to-end GRU-Variational Autoencoder pipeline tailored to IEEE benchmark grids. Data loaders in `gru_vae/data.py` curate sliding-window sequences solely from the clean `*_acopf_all_rows_noisy.csv` files to prevent leakage. The training scripts optimize a gated recurrent encoder-decoder with reconstruction and KL terms, producing latent priors that characterize normal measurement evolution. Inference utilities apply the trained model to FDIA-contaminated streams, score each timestamp-feature pair, and generate repair masks that replace suspicious readings with reconstructed estimates. Configuration files and notebooks document hyperparameters, windowing strategy, and evaluation workflows so that new experiments on IEEE14, IEEE57, and IEEE118 cases can be reproduced consistently across the experimental-playground branch.

## Experimental Scope
Experiments are conducted on three benchmark power system models: IEEE14, IEEE57, and IEEE118. For each system, five datasets provide complementary perspectives on normal operation, attacked measurements, and idealized ground-truth labels.

## Dataset Overview
- **Source and structure:** Each dataset is generated from optimal power flow calculations on the corresponding IEEE system. The CSV files share an identical schema with a header row. The first column is `timestimp`, followed by feature columns (144 for IEEE14, 575 for IEEE57, and 1202 for IEEE118). IEEE118 files contain 17,857 rows (including the header), with 17,856 chronological time steps; IEEE14 and IEEE57 follow the same row count but adjust the feature dimension.
- **Dataset1 - Normal data:** Clean baseline measurements for every time step.
- **Dataset2 - FDIA data:** Measurements corrupted by FDIAs, derived from Dataset1. Attack locations and magnitudes vary across time steps.
- **Dataset3 - Ideal mask labels for Dataset2:** A binary matrix matching Dataset2's dimensions; `1` denotes an unaltered measurement, `0` indicates an injected value.
- **Mixed datasets are not currently provided.**

---

## Training Data Policy (No Data Leakage)

To prevent data leakage and preserve the validity of anomaly localization and repair, training MUST use only the clean baseline ("normal") datasets. Attacked datasets and their labels are for inference/evaluation only.

- Normal data (training source)
  - Pattern (all cases): `data/{case}/{case}_acopf_all_rows_noisy.csv`
  - IEEE 14-bus (current repo): `data/case14/case14_acopf_all_rows_noisy.csv`

- Attacked data (do NOT use for training)
  - Pattern: `data/{case}/{case}_fdia_*_noisy.csv`
  - IEEE 14-bus (example): `data/case14/case14_fdia_2025-07_2025-08_noisy.csv`

- Labels (for evaluation only)
  - Pattern: `data/{case}/{case}_fdia_*_labels.csv`
  - IEEE 14-bus (example): `data/case14/case14_fdia_2025-07_2025-08_labels.csv`

Enforcement guidance
- The canonical training loader `gru_vae/data.py:create_normal_loaders` reads the normal file (`*_acopf_all_rows_noisy.csv`). Do not change this to any FDIA file.
- Training scripts must not accept or silently substitute any path containing `fdia` for training.
- Recommended guardrail: if a training CLI argument points to a file path matching `*fdia*`, raise an error and exit.

Rationale
- Using attacked CSVs during training would normalize anomalies and degrade both localization and repair. Keeping training strictly on clean baselines ensures the model learns the normal manifold and treats FDIA as distributional deviations at inference time.
