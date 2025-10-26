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


def main() -> None:
    args = load_config()
    device = resolve_device(args.device)

    # Paths
    outdir = os.path.join('output', args.case, 'infer', args.exp_name)
    os.makedirs(outdir, exist_ok=True)
    # Load stats from training
    model_dir = os.path.dirname(args.ckpt)
    stats_path = os.path.join(model_dir, 'slot_stats.npz')
    if not os.path.exists(stats_path):
        raise SystemExit('[error] slot_stats.npz not found next to checkpoint; run training first')
    stats = np.load(stats_path, allow_pickle=True)
    slot_mean = stats['mean']  # [S,H]
    slot_std = stats['std']    # [S,H]
    # Optional metadata
    clip_k = float(stats['clip_k']) if 'clip_k' in stats else 5.0

    # Data: infer paired (full series)
    Xa, Xn, M, ts = load_paired_timeseries(args.infer_normal_csv, args.infer_attacked_csv)

    # Model
    # Infer H from data
    H = Xa.shape[1]
    # Safe load: prefer weights_only=True to avoid arbitrary code execution via pickle
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        # Older PyTorch without weights_only: fallback to standard load
        ckpt = torch.load(args.ckpt, map_location='cpu')
    # Minimal config: latent and TCN sizes; saved during training
    latent_dim = ckpt.get('latent_dim', getattr(args, 'latent_dim', 32))
    tcn_channels_str = ckpt.get('tcn_channels', getattr(args, 'tcn_channels', '256,256,256'))
    if isinstance(tcn_channels_str, str):
        tcn_channels = tuple(int(x) for x in tcn_channels_str.split(',') if x)
    else:
        # allow list from YAML
        tcn_channels = tuple(int(x) for x in tcn_channels_str)
    tcn_kernel_size = int(ckpt.get('tcn_kernel_size', getattr(args, 'tcn_kernel_size', 3)))
    tcn_dropout = float(ckpt.get('tcn_dropout', getattr(args, 'tcn_dropout', 0.0)))
    dec_hidden = tuple(int(x) for x in str(getattr(args, 'dec_hidden', '256,256')).split(',') if x)
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
    ).to(device)
    # Accept either full checkpoint dict with 'model' key or raw state_dict
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Standardize full series and run reconstruction (batch=1)
    hours_all = times_to_hour_index(ts)
    Xa_s = apply_standardization_slotwise(Xa, M, hours_all, slot_mean, slot_std, clip_k=clip_k)
    x_in = torch.from_numpy(Xa_s).unsqueeze(0).to(device)
    m_in = torch.from_numpy(M).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred_s = model.reconstruct(x_in, m_in, use_mean=True).squeeze(0).cpu().numpy().astype(np.float32)

    # Unstandardize to original scale
    mean_t = slot_mean[hours_all]
    std_t = slot_std[hours_all]
    y_pred = (y_pred_s * std_t + mean_t).astype(np.float32)

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
    # Scale sigma_09[j] computed on Xn[:, j] at observed positions only; robust via IQR/1.349 with floor and fallback.
    def compute_sigma09_per_feature(x_normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return per-feature robust std over the entire month [H]."""
        T_all = x_normal.shape[0]
        slots_all = np.zeros((T_all,), dtype=np.int64)  # single slot across the whole month
        _, std_ = masked_robust_slot_stats(x_normal, mask, slots_all, slot_count=1)
        # std_ has shape [1, H]; squeeze to [H]
        return std_[0].astype(np.float64)

    sigma09 = compute_sigma09_per_feature(Xn, M)  # [H]

    def nrmse_timestep(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray, sigma_feat: np.ndarray) -> np.ndarray:
        """Per-timestep NRMSE over observed positions only, normalized by per-feature sigma.

        Shapes: y_true/y_hat [T,H], mask [T,H], sigma_feat [H].
        """
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
                # mean of ((diff/sigma)^2) over observed features at this timestamp
                se_norm = (diff * diff) * inv_sigma2[obs]
                out[t] = float(np.sqrt(np.mean(se_norm)))
        return out

    nrmse_rep = nrmse_timestep(Xn, y_pred, M, sigma09)
    nrmse_att = nrmse_timestep(Xn, Xa, M, sigma09)

    # Optional global NRMSE (Observed Only) as a single summary scalar for repaired series
    def nrmse_global(y_true: np.ndarray, y_hat: np.ndarray, mask: np.ndarray, sigma_feat: np.ndarray) -> float:
        obs = mask > 0.5
        if not np.any(obs):
            return float('nan')
        diff = (y_hat.astype(np.float64) - y_true.astype(np.float64))
        # broadcast sigma over time
        sigma2 = (sigma_feat.astype(np.float64) ** 2).reshape(1, -1)
        se_norm = (diff * diff) / sigma2
        se_norm_obs = se_norm[obs]
        return float(np.sqrt(np.mean(se_norm_obs)))

    nrmse_rep_global = nrmse_global(Xn, y_pred, M, sigma09)
    nrmse_att_global = nrmse_global(Xn, Xa, M, sigma09)
    nrmse_improve = nrmse_att - nrmse_rep

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

    # Save single reconstructed CSV for full series
    cols = list(df_a.columns)
    rec_df = pd.DataFrame(columns=cols)
    rec_df[ts_col] = ts
    rec_df.iloc[:, 1:] = y_pred.astype(np.float32)
    rec_path = os.path.join(outdir, 'reconstructed.csv')
    tmp_path = rec_path + '.tmp'
    rec_df.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, rec_path)
    except Exception:
        print(f"[warn] Could not replace reconstructed.csv; wrote {os.path.basename(tmp_path)} instead.")

    print('Inference finished. Reconstructed series saved to:', rec_path)


if __name__ == '__main__':
    main()
