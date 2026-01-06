<!--
文档元数据:
- 日期: 2026-01-02
- 执行者: Codex (GPT-5.2)
-->

# Tasks: 重构 baselines（移除 encoder/decoder 架构）

## 0. 需求定稿（来自用户确认）
- [x] 0.1 `LGSSM-VAE` 与 `mlp_vae` 不允许破坏（保持现有结构/接口/checkpoint 行为）。
- [x] 0.2 彻底移除旧 AE baselines（不做任何兼容）。
- [x] 0.3 新 baselines 命名：`TCN` 与 `LSTM`。
- [x] 0.4 新 baselines 结构约束：代码结构无 `encoder/decoder`，且无 latent/无 reparameterize。

## 1. 仍需你确认（用更直白的表述）
- [x] 1.1 缺失观测的 mask 作为输入特征：采用 `concat([x*mask, mask])`。
- [x] 1.2 `TCN/LSTM` 不输出不确定性（`logvar_x=None`），推理使用 `infer.fixed_sigma_std` 作为固定 sigma。

## 2. Baselines 实现（Breaking）
- [x] 2.1 删除旧 AE baselines，新增 `TCN`、`LSTM`（类内不出现 `encoder/decoder` 拆分）。
- [x] 2.2 `TCN/LSTM` 不再包含 latent/reparameterize；输出为 deterministic `mean[B,T,H]`。
- [x] 2.3 `TCN/LSTM` 实现 `training_step()` 与 `reconstruct()`，输出遵循 `LGSSM_VAE/foundation/interfaces.py:StepOutput` 约定。
- [x] 2.4 统一 observed-only loss 策略（`TCN/LSTM`: MSE；`mlp_vae/LGSSM-VAE` 保持现状）。

## 3. Registry 与 checkpoint（Breaking）
- [x] 3.1 更新 `LGSSM_VAE/modeling/registry.py`：移除旧 AE baselines 的 hparams/dataclass/builder；新增 `TCN/LSTM` 的 hparams/builder。
- [x] 3.2 更新 `_MODEL_BUILDERS_FROM_CONFIG/_MODEL_BUILDERS_FROM_HPARAMS`：仅支持 `LGSSM-VAE`、`mlp_vae`、`TCN`、`LSTM`。
- [x] 3.3 checkpoint 重建：旧 AE baselines 的 checkpoint 不再支持；`LGSSM-VAE/mlp_vae` 旧 checkpoint 仍可用。

## 4. 配置 Schema 与模板（Breaking）
- [x] 4.1 更新 `LGSSM_VAE/config/schema.py`：确保不会影响 `LGSSM-VAE/mlp_vae`；并移除/清理仅服务于旧 AE baselines 的配置说明与模板痕迹。
- [x] 4.2 新增模板：`configs/case14/tcn.yaml`、`configs/case14/lstm.yaml`。

## 5. 对外导出与引用点清理
- [x] 5.1 更新 `LGSSM_VAE/modeling/__init__.py` 的导出：导出 `TCN/LSTM`。
- [x] 5.2 全仓检索并删除旧 AE baselines 的残留引用（tests/docs/scripts/configs）。

## 6. 测试与文档
- [x] 6.1 更新 `tests/test_baselines.py`：删除旧 AE 测试，新增 `TCN/LSTM` 的 shape/反传/重构测试；保留 `MLPVAE` 测试。
- [x] 6.2 更新 `docs/模型对比与重构方案.md`：baseline 清单与说明同步更新；补充破坏性变更与迁移说明。

## 7. 验证
- [x] 7.1 运行 `pytest -q`。
- [x] 7.2 至少用一个 case 的 config 进行一次端到端 dry-run（训练可缩短 epochs；重点验证 config→build→train→save ckpt→infer_range 能走通）。
