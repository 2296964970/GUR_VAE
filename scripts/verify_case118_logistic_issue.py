import os
from typing import Tuple

import numpy as np
import torch

from tcn_vae.config import load_config
from tcn_vae.data import (
    load_timeseries,
    apply_standardization_slotwise,
    times_to_hour_index,
    masked_robust_slot_stats,
)
from tcn_vae.model import TCNVAE
from tcn_vae.utils import resolve_device


def mc_median_reconstruct(
    model: TCNVAE,
    x_in: torch.Tensor,
    m_in: torch.Tensor,
    *,
    samples: int = 16,
) -> np.ndarray:
    """Return MC-median predictions (standardized domain) with no blending.

    Inputs x_in/m_in are tensors of shape [1, T, H]. Returns [T, H].
    """
    preds = []
    with torch.no_grad():
        for _ in range(int(samples)):
            y_s = model.reconstruct(x_in, m_in, use_mean=False)
            preds.append(y_s)
    y_stack = torch.stack(preds, dim=0)  # [K,1,T,H]
    y_med = torch.median(y_stack, dim=0).values.squeeze(0)  # [T,H]
    return y_med.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    # Load config and device
    args = load_config()
    device = resolve_device(args.device)

    # Resolve checkpoint directory
    model_dir = os.path.dirname(args.ckpt)

    # Training stats
    stats_path = os.path.join(model_dir, 'slot_stats.npz')
    if not os.path.exists(stats_path):
        raise SystemExit('[error] slot_stats.npz not found next to checkpoint; run training first')
    stats = np.load(stats_path, allow_pickle=True)
    slot_mean = stats['mean']  # [S,H]
    slot_std = stats['std']    # [S,H]
    clip_k = float(stats['clip_k']) if 'clip_k' in stats else 5.0

    # Single clean CSV
    Xc, M, ts = load_timeseries(args.infer_normal_csv)
    H = Xc.shape[1]

    # Model
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location='cpu')

    latent_dim = ckpt.get('latent_dim', getattr(args, 'latent_dim', 32))
    tcn_channels_raw = ckpt.get('tcn_channels', getattr(args, 'tcn_channels', '256,256,256'))
    if isinstance(tcn_channels_raw, str):
        tcn_channels = tuple(int(x) for x in tcn_channels_raw.split(',') if x)
    else:
        tcn_channels = tuple(int(x) for x in tcn_channels_raw)
    tcn_kernel_size = int(ckpt.get('tcn_kernel_size', getattr(args, 'tcn_kernel_size', 3)))
    tcn_dropout = float(ckpt.get('tcn_dropout', getattr(args, 'tcn_dropout', 0.0)))
    dec_hidden = tuple(int(x) for x in str(getattr(args, 'dec_hidden', '256,256')).split(',') if x)

    enc_diag_eps = ckpt.get('enc_diag_eps', getattr(args, 'enc_diag_eps', 1e-4))
    dec_eps = ckpt.get('dec_eps', getattr(args, 'dec_eps', 1e-6))
    dec_logvar_min = ckpt.get('dec_logvar_min', getattr(args, 'dec_logvar_min', -5.0))
    dec_logvar_max = ckpt.get('dec_logvar_max', getattr(args, 'dec_logvar_max', 2.302585092994046))
    prior_rank = ckpt.get('prior_rank', getattr(args, 'prior_rank', 4))
    prior_a_init = ckpt.get('prior_a_init', getattr(args, 'prior_a_init', 0.95))
    prior_q_init = ckpt.get('prior_q_init', getattr(args, 'prior_q_init', 0.1))
    prior_m0_init = ckpt.get('prior_m0_init', getattr(args, 'prior_m0_init', 0.0))
    prior_P0_init = ckpt.get('prior_P0_init', getattr(args, 'prior_P0_init', 1.0))
    prior_jitter = ckpt.get('prior_jitter', getattr(args, 'prior_jitter', 1e-6))
    prior_variance_floor = ckpt.get('prior_variance_floor', getattr(args, 'prior_variance_floor', 1e-6))

    model = TCNVAE(
        input_dim=H,
        output_dim=H,
        latent_dim=latent_dim,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        dec_hidden=dec_hidden,
        beta=getattr(args, 'beta', 0.1),
        obs_init_logvar=getattr(args, 'obs_init_logvar', -3.5),
        enc_diag_eps=enc_diag_eps,
        dec_eps=dec_eps,
        dec_logvar_min=dec_logvar_min,
        dec_logvar_max=dec_logvar_max,
        prior_rank=prior_rank,
        prior_a_init=prior_a_init,
        prior_q_init=prior_q_init,
        prior_m0_init=prior_m0_init,
        prior_P0_init=prior_P0_init,
        prior_jitter=prior_jitter,
        prior_variance_floor=prior_variance_floor,
    ).to(device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Standardize clean series and run MC-median reconstruction (no blending)
    hours_all = times_to_hour_index(ts)
    Xc_s = apply_standardization_slotwise(Xc, M, hours_all, slot_mean, slot_std, clip_k=clip_k)
    x_in = torch.from_numpy(Xc_s).unsqueeze(0).to(device)
    m_in = torch.from_numpy(M).unsqueeze(0).to(device)
    y_med_s = mc_median_reconstruct(model, x_in, m_in, samples=16)

    # Unstandardize
    mean_t = slot_mean[hours_all]
    std_t = slot_std[hours_all]
    y_pred = (y_med_s * std_t + mean_t).astype(np.float32)

    # Over-repair metrics (observed-only) against original clean series
    def mse_timestep(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        T_, H_ = y_true.shape
        out = np.full((T_), np.nan, dtype=np.float64)
        for t in range(T_):
            obs = mask[t] > 0.5
            cnt = int(obs.sum())
            if cnt > 0:
                diff = y_hat[t, obs].astype(np.float64) - y_true[t, obs].astype(np.float64)
                out[t] = float(np.mean(diff * diff))
        return out

    mse_rep = mse_timestep(Xc, y_pred, M)

    def compute_sigma_per_feature(x_clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
        T_all = x_clean.shape[0]
        slots_all = np.zeros((T_all,), dtype=np.int64)
        _, std_ = masked_robust_slot_stats(x_clean, mask, slots_all, slot_count=1)
        return std_[0].astype(np.float64)

    sigma_feat = compute_sigma_per_feature(Xc, M)

    def nrmse_global(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray, sigma_feat: np.ndarray) -> float:
        obs = mask > 0.5
        if not np.any(obs):
            return float('nan')
        diff = (y_hat.astype(np.float64) - y_true.astype(np.float64))
        sigma2 = (sigma_feat.astype(np.float64) ** 2).reshape(1, -1)
        se_norm = (diff * diff) / sigma2
        se_norm_obs = se_norm[obs]
        return float(np.sqrt(np.mean(se_norm_obs)))

    nrmse_rep_global = nrmse_global(Xc, y_pred, M, sigma_feat)

    # Console report: over-repair metrics only
    ts_s = ts.astype(str).values

    def fmt(v: float, w: int = 10, p: int = 6) -> str:
        try:
            return f"{float(v):{w}.{p}f}"
        except Exception:
            return str(v)

    # Over-repair: global metrics
    print('\n===== Over-Repair Metrics (Observed Only) =====')
    obs = M > 0.5
    if np.any(obs):
        diff = (y_pred.astype(np.float64) - Xc.astype(np.float64))[obs]
        mse_global = float(np.mean(diff * diff))
    else:
        mse_global = float('nan')
    print(f"- Global MSE (prediction vs original): {mse_global:.6f}")

    print('\n===== NRMSE Metrics (Observed Only, input feature-scale) =====')
    print(f"- NRMSE (prediction vs original): {nrmse_rep_global:.6f}")

    # Per-timestamp worst/best by MSE
    valid = ~np.isnan(mse_rep)
    idx = np.where(valid)[0]
    if idx.size == 0:
        print('- Per-timestamp MSE: no valid entries (all-missing).')
    else:
        order_desc = np.argsort(mse_rep[idx])[::-1]
        order_asc = np.argsort(mse_rep[idx])
        top_idx = idx[order_desc][:min(10, len(order_desc))]
        best_idx = idx[order_asc][:min(10, len(order_asc))]
        print('\n- Top 10 by MSE (prediction vs original)')
        print('  Rank  Timestamp              mse_pred')
        for r, t in enumerate(top_idx, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(mse_rep[t])}")
        print('\n- Best 10 by MSE (prediction vs original)')
        print('  Rank  Timestamp              mse_pred')
        for r, t in enumerate(best_idx, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(mse_rep[t])}")


if __name__ == '__main__':
    main()
