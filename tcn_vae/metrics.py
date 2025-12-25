"""Common metrics for observed-only evaluation.

This module centralizes per-timestamp and global metrics used across
experimental scripts to avoid duplication and keep behavior consistent.
"""

from __future__ import annotations

import numpy as np

from .data import masked_robust_slot_stats
import torch


def mse_timestep(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-timestamp observed-only MSE.

    Shapes: y_true/y_hat [T,H], mask [T,H]. Returns [T] with NaN where no observed entries.
    """
    if y_true.shape != y_hat.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_hat, mask must share shape [T,H]")
    T, H = y_true.shape
    out = np.full((T,), np.nan, dtype=np.float64)
    for t in range(T):
        obs = mask[t] > 0.5
        cnt = int(obs.sum())
        if cnt > 0:
            diff = y_hat[t, obs].astype(np.float64) - y_true[t, obs].astype(np.float64)
            out[t] = float(np.mean(diff * diff))
    return out


def rmse_timestep(mse_per_t: np.ndarray) -> np.ndarray:
    """Elementwise sqrt for per-timestamp MSE array [T] -> RMSE [T]."""
    return np.sqrt(mse_per_t)


def rmse_global(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> float:
    """Observed-only global RMSE in original scale."""
    if y_true.shape != y_hat.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_hat, mask must share shape [T,H]")
    obs = mask > 0.5
    if not np.any(obs):
        return float("nan")
    diff2_obs = ((y_hat.astype(np.float64) - y_true.astype(np.float64)) ** 2)[obs]
    return float(np.sqrt(np.mean(diff2_obs)))


def nrmse_timestep(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    mask: np.ndarray,
    sigma_feat: np.ndarray,
) -> np.ndarray:
    """Per-timestamp NRMSE over observed positions normalized by per-feature sigma.

    Shapes: y_true/y_hat [T,H], mask [T,H], sigma_feat [H]. Returns [T].
    """
    if y_true.shape != y_hat.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_hat, mask must share shape [T,H]")
    T, H = y_true.shape
    if sigma_feat.shape[0] != H:
        raise ValueError("sigma_feat must have shape [H]")
    out = np.full((T,), np.nan, dtype=np.float64)
    inv_sigma2 = (1.0 / (sigma_feat.astype(np.float64) ** 2))
    for t in range(T):
        obs = mask[t] > 0.5
        cnt = int(obs.sum())
        if cnt > 0:
            diff = y_hat[t, obs].astype(np.float64) - y_true[t, obs].astype(np.float64)
            se_norm = (diff * diff) * inv_sigma2[obs]
            out[t] = float(np.sqrt(np.mean(se_norm)))
    return out


def nrmse_global(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    mask: np.ndarray,
    sigma_feat: np.ndarray,
) -> float:
    """Observed-only global NRMSE normalized by per-feature sigma."""
    if y_true.shape != y_hat.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_hat, mask must share shape [T,H]")
    if sigma_feat.ndim != 1 or sigma_feat.shape[0] != y_true.shape[1]:
        raise ValueError("sigma_feat must be [H]")
    obs = mask > 0.5
    if not np.any(obs):
        return float("nan")
    diff = (y_hat.astype(np.float64) - y_true.astype(np.float64))
    sigma2 = (sigma_feat.astype(np.float64) ** 2).reshape(1, -1)
    se_norm = (diff * diff) / sigma2
    se_norm_obs = se_norm[obs]
    return float(np.sqrt(np.mean(se_norm_obs)))


def compute_global_sigma_per_feature(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust per-feature std across all time steps using observed entries only.

    Computes IQR/1.349 with a single slot over the entire series.
    Returns an array [H] in float64.
    """
    if x.shape != mask.shape:
        raise ValueError("x and mask must share shape [T,H]")
    T_all = x.shape[0]
    slots_all = np.zeros((T_all,), dtype=np.int64)
    _, std_ = masked_robust_slot_stats(x, mask, slots_all, slot_count=1)
    return std_[0].astype(np.float64)


__all__ = [
    "mse_timestep",
    "rmse_timestep",
    "rmse_global",
    "nrmse_timestep",
    "nrmse_global",
    "compute_global_sigma_per_feature",
]

# ---------------------------------------------------------------------------
# Torch-based losses used by the model (kept here for backward compatibility)
# ---------------------------------------------------------------------------


def gaussian_nll_observed(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Observed-only Gaussian NLL per batch (mean over observed time/features).

    Inputs have shape [B,T,H]. Returns [B].
    The constant term log(2*pi) is included.
    """
    if not (mean.shape == logvar.shape == x.shape == mask.shape):
        raise ValueError("mean, logvar, x, mask must share shape [B,T,H]")
    diff = x - mean
    inv_var = torch.exp(-logvar)
    # 0.5 * ( (x-mu)^2 / var + logvar + log(2*pi) )
    nll_elem = 0.5 * (diff * diff * inv_var + logvar + np.log(2.0 * np.pi))
    nll_masked = nll_elem * mask
    # Mean over observed positions per batch element. This keeps the scale
    # roughly invariant to window length and feature count, which simplifies
    # comparison across cases and configurations.
    nll_flat = nll_masked.flatten(1)
    mask_flat = mask.flatten(1)
    nll_sum = nll_flat.sum(dim=1)
    obs_count = mask_flat.sum(dim=1)
    # Avoid division by zero in degenerate all-missing cases.
    nll_mean = torch.where(obs_count > 0, nll_sum / obs_count, torch.zeros_like(nll_sum))
    return nll_mean


def _masked_mse_per_batch(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    diff2 = (x - y) ** 2
    num = (diff2 * w).flatten(1).sum(dim=1)
    den = w.flatten(1).sum(dim=1)
    out = torch.where(den > 0, num / den, torch.zeros_like(num))
    return out


def mse_missing(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-batch MSE over missing positions (mask <= 0.5). Returns [B]."""
    if not (x.shape == y.shape == mask.shape):
        raise ValueError("x, y, mask must share shape [B,T,H]")
    w = (mask <= 0.5).to(x.dtype)
    return _masked_mse_per_batch(x, y, w)


def mse_observed(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-batch MSE over observed positions (mask > 0.5). Returns [B]."""
    if not (x.shape == y.shape == mask.shape):
        raise ValueError("x, y, mask must share shape [B,T,H]")
    w = (mask > 0.5).to(x.dtype)
    return _masked_mse_per_batch(x, y, w)


__all__ += [
    "gaussian_nll_observed",
    "mse_missing",
    "mse_observed",
]


def batch_metrics(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Batch-level metrics computed on observed entries."""
    mse_obs = mse_observed(mean, target, mask)
    return {"mse_obs": mse_obs}


__all__ += ["batch_metrics"]
