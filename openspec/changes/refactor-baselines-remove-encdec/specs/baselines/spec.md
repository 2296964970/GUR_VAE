# Baselines

## ADDED Requirements

### Requirement: 非 encoder/decoder baselines
The system SHALL provide baseline models (other than `LGSSM-VAE` and `mlp_vae`) that do not use an explicit encoder/decoder split.

#### Scenario: Baseline exposes unified training/inference API
- **WHEN** a non-encoder/decoder baseline model is constructed via the model registry
- **THEN** it SHALL implement `training_step()` and `reconstruct()` as defined by `TrainableModel`
- **AND** it SHALL return `mean` with shape `[B,T,H]`
- **AND** it SHALL return `logvar_x=None` (sigma comes from `infer.fixed_sigma_std`)

### Requirement: No latent / no reparameterize
The baselines `TCN` and `LSTM` SHALL be deterministic direct-mapping models and SHALL NOT contain latent variables or reparameterization.

#### Scenario: No VAE-style sampling is present
- **WHEN** reviewing the `TCN` or `LSTM` baseline implementation
- **THEN** it SHALL NOT contain `mu/logvar` latent parameterization
- **AND** it SHALL NOT sample `z` via reparameterization
- **AND** it SHALL NOT expose encoder/decoder modules

### Requirement: Supported model.name set
The system SHALL accept only the following `model.name` values for baselines in this change:
- `LGSSM-VAE`
- `mlp_vae`
- `TCN`
- `LSTM`

#### Scenario: Unsupported model names are rejected
- **WHEN** `model.name` is not one of the supported values above
- **THEN** the system SHALL raise a configuration/build error

## REMOVED Requirements

### Requirement: 支持 encoder/decoder 的旧 AE baselines
**Reason**: 本次变更要求除 `LGSSM-VAE` 与 `mlp_vae` 外，其余 baseline 必须彻底移除 encoder/decoder 架构，并且不保留兼容。

**Migration**:
- Use `model.name: TCN` or `model.name: LSTM`

#### Scenario: Checkpoint compatibility is not preserved
- **WHEN** a checkpoint contains an unsupported `model_name`
- **THEN** the system SHALL fail with a clear error indicating the model is no longer supported