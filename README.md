# LGSSM-VAE：面向 FDIA 的生成式重构

<!--
文档元数据:
- 日期: 2026-01-01
- 执行者: Codex (GPT-5.2)
-->

面向电力系统测量时间序列的端到端 Temporal Convolutional Network Variational Autoencoder（LGSSM-VAE）。模型学习正常（nominal）轨迹，并通过与相同时间戳的配对正常序列对齐，在仅对观测位置计算损失（observed-only losses）的约束下，对受攻击输入进行重构。整体流程尽量精简：仅使用正常数据进行两阶段训练，并支持按时间范围推理与可选 CSV 导出。

---

## 仓库结构
- `LGSSM_VAE/` 核心库（数据加载、模型、训练器、指标、配置）。
- `scripts/train.py` 仅使用正常窗口进行两阶段训练：Phase-1 身份映射预训练（仅 ELBO），Phase-2 鲁棒训练（稀疏连续 Laplace 攻击 + anchor loss）。
- `scripts/infer_range.py` 在时间范围内执行 fixed-history window 推理。
- `input/` 示例数据集与结构说明。
- `output/` 全部生成产物。

---

## 安装
- Python >= 3.10
- PyTorch（CPU 或 CUDA）
- 安装依赖：`pip install -r requirements.txt`

---

## 数据集与格式
- 文件为 UTF-8（无 BOM），包含表头行。
- 第一列：`timestamp`（字符串），规范化为 `YYYY/MM/DD HH:MM`。
- 其余列：数值特征（float），`NaN` 表示缺失。
- 特征维度：
  - IEEE 14-bus: 144
  - IEEE 57-bus: 575
  - IEEE 118-bus: 1202
- 行数：每文件 17,857 行（1 行表头 + 17,856 个时间步）。

数据位置
- 训练（2025/07–08）：`input/{case}/train/`
  - 仅正常 CSV，例如：`*_2025-07_2025-08_clean_noisy.csv`。
- 推理（2025/09）：`input/{case}/infer/`
  - 成对 CSV：`*_2025-09_clean_noisy.csv`（正常）与 `*_fdia_2025-09_noisy.csv`（受攻击）。

---

## 训练与推理策略（避免数据泄漏）
- 训练仅使用 2025/07–08 的正常 CSV；不使用 9 月文件。
- 训练正常 CSV 不得包含 `fdia` 或 `2025-09`（由配置校验强制）。
- 标准化使用按时间槽（每天 5 分钟粒度）的鲁棒统计量：从训练划分计算并与 checkpoint 同目录保存；推理复用，不重新拟合。

仅观测位置重构与掩码
- 仅观测位置（mask == 1）参与 loss/metrics；缺失值不计入惩罚。
- 可观测掩码由 `NaN` 位置导出，并强制在成对 CSV 之间完全一致。
- 编码器输入通过拼接 `[x*mask, mask]` 让模型感知 mask。

---

## 两阶段训练方案

- Phase-1：身份映射预训练（Normal → Normal）
  - 仅使用干净正常窗口；不做合成攻击。
  - 目标：ELBO（仅观测位置的高斯 NLL + KL，带线性 warm-up）。

- Phase-2：鲁棒训练（稀疏、连续段 Laplace 攻击 + Anchor）
  - 对每个 batch 样本，以概率 `robust.clean_fraction` 保持整段窗口为干净（强化保真约束）。
  - 否则采样一个连续时间片段 `L ∈ [seg_len_min, seg_len_max]`，以及一个特征子集比例 `∈ [dims_fraction_min, dims_fraction_max]`。
  - 在该矩形区域内的观测位置上添加 Laplace(0, b) 偏移，`b` 从 `robust.laplace_scales` 中抽取。
  - 在未攻击的观测位置上添加 anchor loss：`λ · mean((ŷ - x)^2 | a=0, m=1)`，其中 `λ = train.anchor_lambda`，用于抑制对干净输入的过度修复。

---

## 快速开始（推荐：按 case×model 选择配置）

本仓库为每个系统（case）和每个模型各提供 1 份可直接运行的配置文件：`configs/<case>/*.yaml`。

两种使用方式（二选一）：

1) 直接指定配置文件（无需创建新文件）
```powershell
$env:LGSSMVAE_CONFIG="configs/case14/lgssm_vae.yaml"
python -m scripts.train
```

2) 只维护根目录 `config.yaml`（复制一份模板后再改）
```powershell
Copy-Item configs/case14/lgssm_vae.yaml config.yaml
python -m scripts.train
```

如果你想在不改模板文件的情况下做很多实验，也可以在根目录新建一个很小的 `config.yaml` 覆盖少量字段：
```yaml
extends: configs/case14/lgssm_vae.yaml

train:
  model_dir: output/case14/models/exp_name
  device: cuda
  learning_rate: 3.0e-4
```

备注
- `infer.ckpt` 必须显式配置（不再从 `train.model_dir` 推断）。
- 许多底层参数（decoder clamp、prior jitter/floors 等）都有安全默认值，可省略。
- 模型模板（case14）：`configs/case14/lgssm_vae.yaml`、`configs/case14/tcn.yaml`、`configs/case14/lstm.yaml`、`configs/case14/mlp_vae.yaml`

合并语义
- dict：深度合并
- list：整体替换
- scalar：override 覆盖 base

推荐优先调整的少数参数
- 路径：`train.normal_csv`、`infer.normal_csv`、`infer.attacked_csv`、`infer.ckpt`
- 运行：`train.device`、`window.batch_size`、`window.num_workers`
- 实验记录：`train.model_dir`、`train.seed`
- 训练速度/质量权衡：`train.phase1_epochs`、`train.phase2_epochs`、`train.learning_rate`

通常保持默认（除非做研究/消融）
- `model.dec_hidden`、`model.tcn_channels`、`model.decoder.*`、`model.prior.*`
- `robust.*`（Phase-2 合成攻击分布）
- `preprocess.*` 与 `window.time_length/stride`（标准化 + 窗口定义）
- `infer.blend_*`（修复融合行为）

2) 训练
```
python -m scripts.train
```

产物输出到 `train.model_dir`：
- `ckpt.pt`（包含最小超参数）
- `slot_stats.npz`（按时间槽的鲁棒标准化统计量，5 分钟粒度）
- `training_curve.tsv`（四条曲线：P1 train、P1 val、P2 train、P2 val）

3) 推理 / 评估

如果你的推理 CSV 从某月的第一个时间戳开始（例如 9 月 1 日 00:00），fixed-history window 推理在最开始的 `T-1` 个时间步需要至少 `T-1` 行历史数据才能产生输出。本仓库在推理时会在内存中从 `train.normal_csv` 补齐这段缺失历史（不会生成额外的 CSV 文件）。

然后运行范围推理：

```
python -m scripts.infer_range --start "2025-09-01 00:00" --end "2025-09-30 23:55"
```

---

## 模型概览
本仓库通过 `model.name` 支持多种模型对比：
- `LGSSM-VAE`（主模型）：因果 TCN 编码器 + 低秩 VAR(1) 先验 + 高斯解码器。
- `TCN`：确定性 TCN baseline（仅观测位置 MSE）。
- `LSTM`：确定性 LSTM baseline（仅观测位置 MSE）。
- `mlp_vae`：窗口级 MLP-VAE 基线（高斯 NLL + KL）。

各模型保持一致的推理行为（对齐比较口径）：
- 在时间范围内执行 fixed-history window 推理。
- 每个时间步的输出使用该窗口的“最后一步”重构结果。
- 融合修复输出 `recon_blend`：若模型提供 sigma 则使用其 sigma；否则回退到 `infer.fixed_sigma_std`。

---

## 输出目录约定
- 训练：`output/<case>/models/<exp_name>/`。
- 推理：`output/<case>/infer/`。

---

## 编码与语言
- 所有文件使用 UTF-8（无 BOM）。
- 代码/注释：英文为主；与 AI 代理对话：中文（简体）。
- 本文档：中文；英文版见 `README.en.md`。

---

## 配置说明
- `model.dec_hidden`：YAML 整数列表（例如 `[256, 256]`）。
- `model.tcn_channels`：YAML 整数列表（例如 `[256, 256, 256]`）。
- `model.tcn_kernel_size`：正整数（例如 3）。
- `model.tcn_dropout`：区间 [0,1] 的浮点数（默认 0.0）。
- 编码器输入使用拼接 `[x*mask, mask]`；mask 在成对 CSV 中完全一致。
- 解码器预测随时间变化的均值与 log-variance，并进行 clamp 限幅。

---

## 许可
仅供 LGSSM-VAE FDIA 防御项目研究使用。许可文档将另行提供。
