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
This repository supports academic research on defending against false data injection attacks (FDIAs) in power system state estimation. We continue to assume that incoming measurements have already been flagged as potentially corrupted, and the workflow now branches into two operating cases. In the first case, localization is driven by ideal external labels that identify the corrupted entries, yielding an upper-bound reference when those entries are excised before reconstruction. In the second case, a trained model performs localization and recovery without external attack masks: it highlights suspicious values, removes them, and reconstructs the resulting gaps. Across both cases the primary objective is high-quality recovery, so the pipeline tolerates localization errors as long as the reconstructed sequence remains consistent with attack-free behavior.

## Current Project Description
The current GRU-VAE project focuses on building a generative reconstruction pipeline that couples gated recurrent encoders with variational decoders to recover trustworthy measurements after FDIA cleaning. The workflow ingests partially observed time-series produced by upstream detectors, models temporal dynamics with GRU-based encoders, and learns latent representations that enable high-fidelity synthesis of missing or corrupted values. One branch consumes ideal localization masks to study the upper-bound reconstruction performance, while the autonomous branch relies on model-inferred anomaly masks before reconstruction. Emphasis is placed on scalable training across IEEE14, IEEE57, and IEEE118 benchmarks, robust handling of varying anomaly densities, and comprehensive evaluation against ground-truth clean datasets. The resulting artifacts include reusable preprocessing utilities, configurable model checkpoints, and experiment scripts designed for reproducible comparison of defense strategies.

## Experimental Scope
Experiments are conducted on three benchmark power system models: IEEE14, IEEE57, and IEEE118. For each system, five datasets provide complementary perspectives on normal operation, attacked measurements, and idealized ground-truth labels.

## Dataset Overview
- **Source and structure:** Each dataset is generated from optimal power flow calculations on the corresponding IEEE system. The CSV files share an identical schema with a header row. The first column is `timestimp`, followed by feature columns (144 for IEEE14, 575 for IEEE57, and 1202 for IEEE118). IEEE118 files contain 17,857 rows (including the header), with 17,856 chronological time steps; IEEE14 and IEEE57 follow the same row count but adjust the feature dimension.
- **Dataset1 - Normal data:** Clean baseline measurements for every time step.
- **Dataset2 - FDIA data:** Measurements corrupted by FDIAs, derived from Dataset1. Attack locations and magnitudes vary across time steps.
- **Dataset3 - Ideal mask labels for Dataset2:** A binary matrix matching Dataset2's dimensions; `1` denotes an unaltered measurement, `0` indicates an injected value.
- **Mixed datasets are not currently provided.**
