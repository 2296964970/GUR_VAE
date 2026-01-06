import numpy as np
import pandas as pd
import pytest
import torch

from LGSSM_VAE.foundation.errors import DataError
from LGSSM_VAE.foundation.metrics import gaussian_nll_observed, mse_missing, mse_observed
from LGSSM_VAE.data import PairedSlidingWindowDataset, load_paired_timeseries


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


def test_paired_dataset_tuple_shapes():
    T, H = 20, 4
    xa = np.random.randn(T, H).astype(np.float32)
    xn = np.random.randn(T, H).astype(np.float32)
    m = np.ones((T, H), dtype=np.float32)
    ds = PairedSlidingWindowDataset(xa, xn, m, window=8, stride=4)
    xa_w, m_w, xn_w = ds[0]
    assert xa_w.shape == (8, H)
    assert m_w.shape == (8, H)
    assert xn_w.shape == (8, H)


def test_load_paired_timeseries_rejects_mismatched_headers(tmp_path):
    normal_csv = tmp_path / "normal.csv"
    attacked_csv = tmp_path / "attacked.csv"

    df_normal = pd.DataFrame(
        {
            "ts": ["2025/09/15 00:00", "2025/09/15 00:05"],
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        }
    )
    df_attacked = pd.DataFrame(
        {
            "ts": ["2025/09/15 00:00", "2025/09/15 00:05"],
            "a": [1.0, 2.0],
            "c": [3.0, 4.0],
        }
    )

    df_normal.to_csv(normal_csv, index=False)
    df_attacked.to_csv(attacked_csv, index=False)

    with pytest.raises(DataError) as exc:
        load_paired_timeseries(str(normal_csv), str(attacked_csv))
    assert "特征列不一致" in str(exc.value)
