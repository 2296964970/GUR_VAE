import numpy as np
import pandas as pd
import torch

from LGSSM_VAE.pipeline import RangeInferInputs, RangeInferRequest, infer_sequence_over_range


class DummyModel(torch.nn.Module):
    """Minimal model stub that returns fixed standardized mean/logvar."""

    def __init__(self, mean_last: np.ndarray, logvar_last: np.ndarray):
        super().__init__()
        self._mean_last = torch.as_tensor(mean_last, dtype=torch.float32)
        self._logvar_last = torch.as_tensor(logvar_last, dtype=torch.float32)

    @torch.no_grad()
    def reconstruct(self, x, mask, use_mean=True, *, return_logvar=False):
        # Return constant outputs with the right shape.
        B, T, H = x.shape
        mean_seq = self._mean_last.view(1, 1, H).expand(B, T, H).clone()
        logvar_seq = self._logvar_last.view(1, 1, H).expand(B, T, H).clone()
        if return_logvar:
            return mean_seq, logvar_seq
        return mean_seq


class DummyModelNoLogvar(torch.nn.Module):
    """Minimal model stub that returns fixed standardized mean (no logvar)."""

    def __init__(self, mean_last: np.ndarray):
        super().__init__()
        self._mean_last = torch.as_tensor(mean_last, dtype=torch.float32)

    @torch.no_grad()
    def reconstruct(self, x, mask, use_mean=True, *, return_logvar=False):
        batch_size, time_steps, num_features = x.shape
        mean_seq = self._mean_last.view(1, 1, num_features).expand(batch_size, time_steps, num_features).clone()
        if return_logvar:
            return mean_seq, None
        return mean_seq


def _run_infer(
    model: torch.nn.Module,
    *,
    x_obs: np.ndarray,
    x_obs_std: np.ndarray,
    mask: np.ndarray,
    slots: np.ndarray,
    slot_mean: np.ndarray,
    slot_std: np.ndarray,
    ts: pd.Series,
    time_length: int,
    start_time: str,
    end_time: str,
    blend_k_sigma: float,
    blend_softness: float,
    blend_sigma_temperature: float,
    device: torch.device,
    clean_target=None,
    fixed_sigma_std: float = 0.5,
    disable_blend: bool = False,
):
    inputs = RangeInferInputs(
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=mask,
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        clean_target=clean_target,
    )
    request = RangeInferRequest(
        time_length=int(time_length),
        start_time=str(start_time),
        end_time=str(end_time),
        blend_k_sigma=float(blend_k_sigma),
        blend_softness=float(blend_softness),
        blend_sigma_temperature=float(blend_sigma_temperature),
        device=device,
        fixed_sigma_std=float(fixed_sigma_std),
        disable_blend=bool(disable_blend),
    )
    return infer_sequence_over_range(model, inputs=inputs, request=request)


def test_inference_unstandardizes_and_blends_in_raw_domain():
    # Small synthetic setup
    T, H = 5, 2
    time_length = 3

    # Slots per timestep (must be within slot_count)
    slots = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    # Per-slot robust stats (mean=median, std=robust std)
    slot_mean = np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ],
        dtype=np.float32,
    )
    slot_std = np.array(
        [
            [2.0, 5.0],
            [4.0, 10.0],
            [6.0, 15.0],
        ],
        dtype=np.float32,
    )

    # Dummy standardized model outputs
    mean_last_std = np.array([1.0, -2.0], dtype=np.float32)
    sigma_std = np.array([0.5, 0.5], dtype=np.float32)
    logvar_last_std = np.log(sigma_std ** 2).astype(np.float32)
    model = DummyModel(mean_last_std, logvar_last_std)

    # Build raw-domain observations so that z==2 => w==0.5 for observed entries.
    x_obs = np.zeros((T, H), dtype=np.float32)
    for t_idx in (2, 3, 4):
        s = slots[t_idx]
        mu_raw = mean_last_std * slot_std[s] + slot_mean[s]
        sigma_raw = sigma_std * slot_std[s]
        x_obs[t_idx] = mu_raw + 2.0 * sigma_raw

    mask = np.ones_like(x_obs, dtype=np.float32)

    # Standardized input windows are unused by DummyModel, but must have correct shape.
    x_obs_std = np.zeros_like(x_obs, dtype=np.float32)

    # Timestamps every 5 minutes
    ts = pd.Series(
        pd.date_range("2025-09-15 00:00", periods=T, freq="5min").strftime("%Y/%m/%d %H:%M")
    )

    out = _run_infer(
        model,
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=mask,
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        time_length=time_length,
        start_time="2025-09-15 00:00",
        end_time="2025-09-15 00:20",
        blend_k_sigma=2.0,
        blend_softness=0.1,
        blend_sigma_temperature=1.0,
        device=torch.device("cpu"),
    )

    # Indices should start at time_length-1=2
    assert out.indices.tolist() == [2, 3, 4]

    # Check unstandardized mean/sigma and blend in raw domain.
    for i, t_idx in enumerate(out.indices):
        s = slots[t_idx]
        mu_raw = mean_last_std * slot_std[s] + slot_mean[s]
        sigma_raw = sigma_std * slot_std[s]
        # recon_mean and sigma_raw should be raw-domain
        assert np.allclose(out.recon_mean[i], mu_raw, atol=1e-6)
        assert np.allclose(out.sigma_raw[i], sigma_raw, atol=1e-6)
        # With z==2 and k_sigma==2 => w==0.5 => blend = mu_raw + sigma_raw
        expected_blend = mu_raw + sigma_raw
        assert np.allclose(out.recon_blend[i], expected_blend, atol=1e-5)


def test_inference_uses_fixed_sigma_when_model_has_no_logvar():
    num_steps, num_features = 5, 2
    time_length = 3

    slots = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    slot_mean = np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ],
        dtype=np.float32,
    )
    slot_std = np.array(
        [
            [2.0, 5.0],
            [4.0, 10.0],
            [6.0, 15.0],
        ],
        dtype=np.float32,
    )

    mean_last_std = np.array([1.0, -2.0], dtype=np.float32)
    model = DummyModelNoLogvar(mean_last_std)

    fixed_sigma_std = 0.5
    sigma_std = np.array([fixed_sigma_std, fixed_sigma_std], dtype=np.float32)

    x_obs = np.zeros((num_steps, num_features), dtype=np.float32)
    for t_idx in (2, 3, 4):
        s = slots[t_idx]
        mu_raw = mean_last_std * slot_std[s] + slot_mean[s]
        sigma_raw = sigma_std * slot_std[s]
        x_obs[t_idx] = mu_raw + 2.0 * sigma_raw

    mask = np.ones_like(x_obs, dtype=np.float32)
    x_obs_std = np.zeros_like(x_obs, dtype=np.float32)
    ts = pd.Series(
        pd.date_range("2025-09-15 00:00", periods=num_steps, freq="5min").strftime("%Y/%m/%d %H:%M")
    )

    out = _run_infer(
        model,
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=mask,
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        time_length=time_length,
        start_time="2025-09-15 00:00",
        end_time="2025-09-15 00:20",
        blend_k_sigma=2.0,
        blend_softness=0.1,
        blend_sigma_temperature=1.0,
        fixed_sigma_std=fixed_sigma_std,
        device=torch.device("cpu"),
    )

    assert out.indices.tolist() == [2, 3, 4]
    for i, t_idx in enumerate(out.indices):
        s = slots[t_idx]
        mu_raw = mean_last_std * slot_std[s] + slot_mean[s]
        sigma_raw = sigma_std * slot_std[s]
        assert np.allclose(out.recon_mean[i], mu_raw, atol=1e-6)
        assert np.allclose(out.sigma_raw[i], sigma_raw, atol=1e-6)
        expected_blend = mu_raw + sigma_raw
        assert np.allclose(out.recon_blend[i], expected_blend, atol=1e-5)


def test_inference_reports_rmse_before_and_repair_vs_clean():
    # Small synthetic setup
    T, H = 5, 2
    time_length = 3

    slots = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    slot_mean = np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ],
        dtype=np.float32,
    )
    slot_std = np.array(
        [
            [2.0, 5.0],
            [4.0, 10.0],
            [6.0, 15.0],
        ],
        dtype=np.float32,
    )

    mean_last_std = np.array([1.0, -2.0], dtype=np.float32)
    sigma_std = np.array([0.5, 0.5], dtype=np.float32)
    logvar_last_std = np.log(sigma_std ** 2).astype(np.float32)
    model = DummyModel(mean_last_std, logvar_last_std)

    # Raw observations: obs = mu_raw + 2*sigma_raw (so z==2 on observed entries)
    x_obs = np.zeros((T, H), dtype=np.float32)
    clean = np.zeros_like(x_obs, dtype=np.float32)
    for t_idx in (2, 3, 4):
        s = slots[t_idx]
        mu_raw = mean_last_std * slot_std[s] + slot_mean[s]
        sigma_raw = sigma_std * slot_std[s]
        x_obs[t_idx] = mu_raw + 2.0 * sigma_raw
        # Set clean target exactly at the model mean for deterministic RMSE checks.
        clean[t_idx] = mu_raw

    mask = np.ones_like(x_obs, dtype=np.float32)
    x_obs_std = np.zeros_like(x_obs, dtype=np.float32)
    ts = pd.Series(
        pd.date_range("2025-09-15 00:00", periods=T, freq="5min").strftime("%Y/%m/%d %H:%M")
    )

    out = _run_infer(
        model,
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=mask,
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        time_length=time_length,
        start_time="2025-09-15 00:00",
        end_time="2025-09-15 00:20",
        blend_k_sigma=2.0,
        blend_softness=0.1,
        blend_sigma_temperature=1.0,
        device=torch.device("cpu"),
        clean_target=clean,
    )

    assert isinstance(out.rmse_vs_clean, np.ndarray)
    assert isinstance(out.rmse_obs_vs_clean, np.ndarray)
    assert out.rmse_vs_clean.shape == out.rmse_obs_vs_clean.shape

    for i, t_idx in enumerate(out.indices):
        s = slots[t_idx]
        sigma_raw = sigma_std * slot_std[s]
        # Baseline: obs vs clean => diff = 2*sigma_raw
        expected_rmse_obs = float(np.sqrt(np.mean((2.0 * sigma_raw) ** 2)))
        # Blend: mu + sigma vs clean(mu) => diff = sigma_raw
        expected_rmse_blend = float(np.sqrt(np.mean((sigma_raw) ** 2)))
        assert np.allclose(out.rmse_obs_vs_clean[i], expected_rmse_obs, atol=1e-6)
        assert np.allclose(out.rmse_vs_clean[i], expected_rmse_blend, atol=1e-6)
