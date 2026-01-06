<!--
文档元数据:
- 日期: 2026-01-02
- 执行者: Codex (GPT-5.2)
-->

# 设计说明：TCN/LSTM Baselines（无 encoder/decoder、无 latent）

## 1. 目标
- 仅对 baselines 做破坏性调整：彻底移除旧的 AE baselines（含其 checkpoint/config 支持）。
- 新增/替换 baselines：`TCN` 与 `LSTM`。
- `TCN`/`LSTM` 必须满足：
  - 代码结构上不再有 `encoder/decoder` 拆分（不保留字段名/模块名/类职责）
  - 无 latent / 无 reparameterize（不再出现 `mu/logvar`、`z` 采样等 VAE 结构）
  - 仍遵循统一协议 `TrainableModel`（`training_step` / `reconstruct`）
- `LGSSM-VAE` 与 `mlp_vae` 不允许破坏（接口与 checkpoint 行为保持）。

## 2. Baseline 行为定义（统一接口）
`TCN`/`LSTM` 的输出仍为窗口级序列重构：
- 输入：`x_input[B,T,H]`、`mask_keep[B,T,H]`、`x_target[B,T,H]`
- 输出：`mean[B,T,H]`（重构均值序列）
- 损失：仅在观测点上计算的 MSE（observed-only）
- 不确定性：默认 `logvar_x=None`（推理融合若需要 sigma，则由 `infer.fixed_sigma_std` 提供）

## 3. 关于“mask-aware 输入”（用于处理缺失观测）
数据中存在缺失观测，`mask_keep` 表示该点是否观测到（1=观测，0=缺失）。

为让模型知道“哪些是缺失值”，有两种输入方式：
1) **方式 A（推荐）**：输入 `concat([x_input * mask_keep, mask_keep])`，形状为 `[B,T,2H]`
2) **方式 B**：只输入 `x_input * mask_keep`，形状为 `[B,T,H]`（模型无法显式区分“真实 0”与“缺失置 0”）

本变更建议采用方式 A 以保持缺失信息显式可见；若你希望采用方式 B，可在任务中将该选择定稿。

## 4. 关于“logvar_x/不确定性输出”（用于融合策略）
推理阶段存在“观测-重构融合”，需要一个 sigma（不确定性）来决定更相信观测还是重构：
- 若模型输出 `logvar_x`（即每个点的方差），推理会用它计算 sigma；
- 若模型不输出，则推理使用固定超参 `infer.fixed_sigma_std` 作为 sigma。

本变更中 `TCN`/`LSTM` 默认不输出不确定性（`logvar_x=None`），由 `infer.fixed_sigma_std` 控制融合强度。

## 5. 架构草图（示意）
### 5.1 TCN baseline（示意）
`[B,T,2H]` → causal TCN 堆叠（1D conv over time）→ 1x1 conv 投影 → `mean[B,T,H]`

### 5.2 LSTM baseline（示意）
`[B,T,2H]` → LSTM（batch_first）→ 线性投影 → `mean[B,T,H]`

两者均为“直接从输入到输出”的 deterministic 映射，不存在 latent/reparameterize。

