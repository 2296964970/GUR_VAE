import torch
from gru_vae.metrics import gaussian_nll_observed, mse_missing, mse_observed
from gru_vae.data import SelfSupervisedMaskingDataset
import numpy as np


def test_metrics_mask_semantics():
    B, T, H = 2, 5, 3
    mean = torch.zeros(B, T, H)
    logvar = torch.zeros(B, T, H)
    x = torch.zeros(B, T, H)
    mask = torch.ones(B, T, H)
    nll = gaussian_nll_observed(mean, logvar, x, mask)
    assert nll.shape == (B,)
    # no-missing case -> mse_missing == 0
    mm = mse_missing(x, mean, mask)
    assert torch.allclose(mm, torch.zeros(B))
    # observed mse computed only over ones
    mo = mse_observed(x, mean + 1.0, mask)
    assert torch.allclose(mo, torch.ones(B))


def test_self_supervised_dataset_tuple_shapes():
    T, H = 20, 4
    x = np.random.randn(T, H).astype(np.float32)
    m = np.ones((T, H), dtype=np.float32)
    ds = SelfSupervisedMaskingDataset(x, m, window=8, stride=4, mask_rate=0.5, seed=123)
    xm, ms, xt = ds[0]
    assert xm.shape == (8, H)
    assert ms.shape == (8, H)
    assert xt.shape == (8, H)
