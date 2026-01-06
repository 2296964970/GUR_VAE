# Project Context

<!--
文档元数据:
- 日期: 2026-01-01
- 执行者: Codex (GPT-5.2)
-->

## Purpose
- 本项目用于电力系统测量流（带缺失观测）的 FDIA（False Data Injection Attack）场景：学习正常轨迹并对攻击输入进行重构/修复。
- 当前主模型为 `LGSSM-VAE`，训练/推理均以配置驱动（`config.yaml`）。

## Tech Stack
- Python (>=3.9)
- PyTorch
- NumPy / Pandas
- PyYAML

## Project Conventions

### Code Style
- 代码尽量精简、可读；优先 SoC 与 DRY。
- 文档/注释使用简体中文、UTF-8（无 BOM）。

### Architecture Patterns
- 数据加载/标准化/窗口切片：集中在 `LGSSM_VAE/data/io.py`
- 训练循环：集中在 `LGSSM_VAE/pipeline/trainer.py`
- 推理与指标：集中在 `LGSSM_VAE/pipeline/inference.py`
- CLI 脚本：`scripts/train.py`、`scripts/infer_range.py`

### Testing Strategy
- 使用 `pytest`（见 `tests/`）验证配置严格性、模型/推理/指标的关键行为。

### Git Workflow
- 未在仓库内强制约定；建议保持小步提交与可回滚改动。

## Domain Context
- 数据以 CSV 形式提供，第一列为时间戳，后续为数值特征；缺失以 `NaN` 表示。
- 训练使用特定月份的 normal 数据，推理使用另一个月份的 paired normal/attacked 数据；需避免数据泄漏。

## Important Constraints
- 标准化统计量必须仅由训练 normal 数据得到，并在推理时复用。
- paired normal/attacked CSV 必须时间戳对齐且 mask 完全一致。

## External Dependencies
- 无外部服务依赖；所有输入来自本地 `input/` 目录（或用户提供路径）。
