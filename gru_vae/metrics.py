import math
from typing import Dict

import torch


def gaussian_nll_observed(mean: torch.Tensor, logvar: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not (mean.shape == logvar.shape == x.shape == mask.shape):
        raise ValueError('mean, logvar, x, mask must share shape [B,T,H]')
    log2pi = math.log(2.0 * math.pi)
    inv_var = torch.exp(-logvar)
    nll_elem = 0.5 * (log2pi + logvar + (x - mean) ** 2 * inv_var)
    mask = mask.to(dtype=mean.dtype)
    nll_sum = (nll_elem * mask).sum(dim=(1, 2))
    obs_count = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return nll_sum / obs_count


def mse_missing(x_true: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not (x_true.shape == x_hat.shape == mask.shape):
        raise ValueError('x_true, x_hat, mask must share shape [B,T,H]')
    miss = (mask <= 0.5).to(dtype=x_true.dtype)
    se = (x_true - x_hat) ** 2
    miss_sum = (se * miss).sum(dim=(1, 2))
    miss_count = miss.sum(dim=(1, 2))
    denom = torch.where(miss_count > 0, miss_count, torch.ones_like(miss_count))
    mse = miss_sum / denom
    mse = torch.where(miss_count > 0, mse, torch.zeros_like(mse))
    return mse


def mse_observed(x_true: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not (x_true.shape == x_hat.shape == mask.shape):
        raise ValueError('x_true, x_hat, mask must share shape [B,T,H]')
    obs = (mask > 0.5).to(dtype=x_true.dtype)
    se = (x_true - x_hat) ** 2
    obs_sum = (se * obs).sum(dim=(1, 2))
    obs_count = obs.sum(dim=(1, 2)).clamp_min(1.0)
    return obs_sum / obs_count


def batch_metrics(mean: torch.Tensor, logvar: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    nll = gaussian_nll_observed(mean, logvar, x, mask)
    x_hat = mean
    return {
        'nll': nll,
        'mse_miss': mse_missing(x, x_hat, mask),
        'mse_obs': mse_observed(x, x_hat, mask),
    }


__all__ = ['gaussian_nll_observed', 'mse_missing', 'mse_observed', 'batch_metrics']
