<!--
文档元数据:
- 日期: 2026-01-02
- 执行者: Codex (GPT-5.2)
-->

# Change: 重构 baselines（移除 encoder/decoder 架构）

## Why
当前两个 AE baselines 采用典型的“编码器 + 解码器”Autoencoder 结构。现需求为：除 `LGSSM-VAE` 与 `mlp_vae` 外，其余 baseline **必须彻底移除** encoder/decoder 架构，并且**不保留任何兼容**（旧配置/旧 checkpoint 均不再支持）。

## What Changes
- **BREAKING**：移除旧 AE baselines 的支持（含其 checkpoint 结构与重建逻辑）。
- **BREAKING**：删除旧 AE baselines（以及任何 `encoder/decoder` 拆分），替换为新 baselines：`TCN` 与 `LSTM`。
- 新 baselines `TCN`/`LSTM` 必须满足：
  - 代码结构上不再出现 `encoder/decoder` 模块拆分
  - **无 latent / 无 reparameterize**（不再存在 VAE 风格隐变量采样）
  - 仍实现统一训练/推理协议（`TrainableModel`）
- **不允许破坏**：`LGSSM-VAE` 与 `mlp_vae` 的现有模型结构、训练/推理接口与 checkpoint 行为必须保持可用。

## Impact
- Affected code
  - `LGSSM_VAE/modeling/baselines.py`
  - `LGSSM_VAE/modeling/registry.py`
  - `LGSSM_VAE/modeling/__init__.py`
  - `LGSSM_VAE/config/schema.py`
  - `LGSSM_VAE/pipeline/inference.py`（checkpoint 重建路径）
- Affected configs
  - `configs/case14/tcn.yaml`（新增）
  - `configs/case14/lstm.yaml`（新增）
- Affected tests/docs
  - `tests/test_baselines.py`
  - `docs/模型对比与重构方案.md`

## Out of Scope
- 不引入新的数据处理/指标定义（沿用当前 pipeline）。
- 不为旧 checkpoint 或旧 config 提供迁移兼容层（仅提供迁移说明）。

## Acceptance
- `LGSSM-VAE` 与 `mlp_vae`：
  - 训练/推理路径保持可用
  - 旧 checkpoint 仍可重建并推理
- `TCN` 与 `LSTM`：
  - 满足“无 encoder/decoder、无 latent、无 reparameterize”的结构要求
  - 满足 `TrainableModel` 协议，并可被 registry 从 config 与 checkpoint 正确构建
- 使用旧配置或旧 checkpoint 将报错（不提供兼容与迁移）。
