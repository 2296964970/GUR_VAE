"""Observed-only losses and lightweight metrics.

This module contains the torch-based losses used by the model/trainer and a
small set of helper metrics, to keep behavior consistent across the codebase.
"""

from __future__ import annotations

import numpy as np
import torch

__all__: list[str] = []

# Torch-based losses used by the model.
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
