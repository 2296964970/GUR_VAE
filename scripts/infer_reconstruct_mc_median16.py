import os
import numpy as np
import pandas as pd
import torch

from tcn_vae.config import load_config
from tcn_vae.data import (
    load_paired_timeseries,
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

    Inputs x_in/m_in are tensors of shape [1, T, H]. Returns a numpy array [T, H].
    """
    preds = []
    with torch.no_grad():
        for _ in range(int(samples)):
            y_s = model.reconstruct(x_in, m_in, use_mean=False)  # [1,T,H]
            preds.append(y_s)
    y_stack = torch.stack(preds, dim=0)  # [K,1,T,H]
    y_med = torch.median(y_stack, dim=0).values.squeeze(0)  # [T,H]
    return y_med.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = load_config()
    device = resolve_device(args.device)

    # Derive experiment name from checkpoint directory to keep output/log style aligned
    model_dir = os.path.dirname(args.ckpt)
    exp_name = os.path.basename(model_dir) if model_dir else 'default'

    # Load stats from training
    stats_path = os.path.join(model_dir, 'slot_stats.npz')
    if not os.path.exists(stats_path):
        raise SystemExit('[error] slot_stats.npz not found next to checkpoint; run training first')
    stats = np.load(stats_path, allow_pickle=True)
    slot_mean = stats['mean']  # [S,H]
    slot_std = stats['std']    # [S,H]
    clip_k = float(stats['clip_k']) if 'clip_k' in stats else 5.0

    # Data: paired (full series)
    Xa, Xn, M, ts = load_paired_timeseries(args.infer_normal_csv, args.infer_attacked_csv)

    # Model
    H = Xa.shape[1]
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location='cpu')
    latent_dim = ckpt.get('latent_dim', getattr(args, 'latent_dim', 32))
    tcn_channels_str = ckpt.get('tcn_channels', getattr(args, 'tcn_channels', '256,256,256'))
    if isinstance(tcn_channels_str, str):
        tcn_channels = tuple(int(x) for x in tcn_channels_str.split(',') if x)
    else:
        tcn_channels = tuple(int(x) for x in tcn_channels_str)
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

    # Standardize full series
    hours_all = times_to_hour_index(ts)
    Xa_s = apply_standardization_slotwise(Xa, M, hours_all, slot_mean, slot_std, clip_k=clip_k)
    x_in = torch.from_numpy(Xa_s).unsqueeze(0).to(device)
    m_in = torch.from_numpy(M).unsqueeze(0).to(device)

    # MC-median predictions in standardized domain (no blending)
    y_med_s = mc_median_reconstruct(model, x_in, m_in, samples=16)

    # Unstandardize to original scale (replace all entries by predictions)
    mean_t = slot_mean[hours_all]
    std_t = slot_std[hours_all]
    y_pred = (y_med_s * std_t + mean_t).astype(np.float32)

    # Compute per-timestamp observed-only MSE for attacked/repaired vs normal
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

    mse_rep = mse_timestep(Xn, y_pred, M)
    mse_att = mse_timestep(Xn, Xa, M)

    # Compute repair NRMSE (Observed Only), normalization from September normal per-feature across all time steps
    def compute_sigma09_per_feature(x_normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return per-feature robust std over the entire month [H]."""
        T_all = x_normal.shape[0]
        slots_all = np.zeros((T_all,), dtype=np.int64)
        _, std_ = masked_robust_slot_stats(x_normal, mask, slots_all, slot_count=1)
        return std_[0].astype(np.float64)

    sigma09 = compute_sigma09_per_feature(Xn, M)

    def nrmse_timestep(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray, sigma_feat: np.ndarray) -> np.ndarray:
        T_, H_ = y_true.shape
        if sigma_feat.shape[0] != H_:
            raise ValueError('sigma_feat must have shape [H]')
        out = np.full((T_), np.nan, dtype=np.float64)
        inv_sigma2 = (1.0 / (sigma_feat.astype(np.float64) ** 2))
        for t in range(T_):
            obs = mask[t] > 0.5
            cnt = int(obs.sum())
            if cnt > 0:
                diff = y_hat[t, obs].astype(np.float64) - y_true[t, obs].astype(np.float64)
                se_norm = (diff * diff) * inv_sigma2[obs]
                out[t] = float(np.sqrt(np.mean(se_norm)))
        return out

    nrmse_rep = nrmse_timestep(Xn, y_pred, M, sigma09)
    nrmse_att = nrmse_timestep(Xn, Xa, M, sigma09)

    def nrmse_global(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray, sigma_feat: np.ndarray) -> float:
        obs = mask > 0.5
        if not np.any(obs):
            return float('nan')
        diff = (y_hat.astype(np.float64) - y_true.astype(np.float64))
        sigma2 = (sigma_feat.astype(np.float64) ** 2).reshape(1, -1)
        se_norm = (diff * diff) / sigma2
        se_norm_obs = se_norm[obs]
        return float(np.sqrt(np.mean(se_norm_obs)))

    nrmse_rep_global = nrmse_global(Xn, y_pred, M, sigma09)
    nrmse_att_global = nrmse_global(Xn, Xa, M, sigma09)

    # Terminal summary: Top-10 and Worst-10 by MSE repair effect (att - rep)
    df_a = pd.read_csv(args.infer_attacked_csv)
    ts_col = df_a.columns[0]
    ts_s = ts.astype(str).values

    def fmt(v: float, w: int = 10, p: int = 6) -> str:
        try:
            return f"{float(v):{w}.{p}f}"
        except Exception:
            return str(v)

    valid = (~np.isnan(mse_att)) & (~np.isnan(mse_rep))
    idx = np.where(valid)[0]
    if idx.size == 0:
        print('\n===== Reconstruction Metrics (Observed Only) =====')
        print('- MSE: no valid entries (all-missing).')
    else:
        impr = mse_att[idx] - mse_rep[idx]
        order_desc = np.argsort(impr)[::-1]
        order_asc = np.argsort(impr)
        top_idx = idx[order_desc][:min(10, len(order_desc))]
        worst_idx = idx[order_asc][:min(10, len(order_asc))]
        print('\n===== Reconstruction Metrics (Observed Only) =====')
        print('- Top 10 by repair MSE (att - rep)')
        print('  Rank  Timestamp              mse_att      mse_rep      improve')
        for r, t in enumerate(top_idx, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(mse_att[t])}  {fmt(mse_rep[t])}  {fmt(mse_att[t]-mse_rep[t])}")
        print('\n- Worst 10 by repair MSE (att - rep)')
        print('  Rank  Timestamp              mse_att      mse_rep      improve')
        for r, t in enumerate(worst_idx, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(mse_att[t])}  {fmt(mse_rep[t])}  {fmt(mse_att[t]-mse_rep[t])}")

    # Report NRMSE (Observed Only) for attacked/repaired series
    print('\n===== NRMSE Metrics (Observed Only, Sept feature-scale) =====')
    print(f"- NRMSE (attacked vs normal): {nrmse_att_global:.6f}")
    print(f"- NRMSE (repaired vs normal): {nrmse_rep_global:.6f}")
    if np.isfinite(nrmse_att_global) and np.isfinite(nrmse_rep_global):
        abs_imp = nrmse_att_global - nrmse_rep_global
        rel_red = (1.0 - (nrmse_rep_global / max(nrmse_att_global, 1e-12))) * 100.0
        print(f"- Absolute improvement (att - rep): {abs_imp:.6f}")
        print(f"- Relative reduction: {rel_red:.2f}%")
    else:
        print('- Global NRMSE summary unavailable (insufficient valid entries).')

    # Per-timestamp NRMSE: Top/Worst by improvement (att - rep)
    valid_np = (~np.isnan(nrmse_att)) & (~np.isnan(nrmse_rep))
    idx_np = np.where(valid_np)[0]
    if idx_np.size == 0:
        print('- Per-timestamp NRMSE: no valid entries (all-missing).')
    else:
        impr_n = nrmse_att[idx_np] - nrmse_rep[idx_np]
        order_desc_n = np.argsort(impr_n)[::-1]
        order_asc_n = np.argsort(impr_n)
        top_idx_n = idx_np[order_desc_n][:min(10, len(order_desc_n))]
        worst_idx_n = idx_np[order_asc_n][:min(10, len(order_asc_n))]
        print('\n- Top 10 by repair NRMSE (att - rep)')
        print('  Rank  Timestamp              nrmse_att    nrmse_rep      improve')
        for r, t in enumerate(top_idx_n, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(nrmse_att[t])}  {fmt(nrmse_rep[t])}  {fmt(nrmse_att[t]-nrmse_rep[t])}")
        print('\n- Worst 10 by repair NRMSE (att - rep)')
        print('  Rank  Timestamp              nrmse_att    nrmse_rep      improve')
        for r, t in enumerate(worst_idx_n, 1):
            print(f"  {r:>4d}  {ts_s[int(t)]:>19s}  {fmt(nrmse_att[t])}  {fmt(nrmse_rep[t])}  {fmt(nrmse_att[t]-nrmse_rep[t])}")


if __name__ == '__main__':
    main()

