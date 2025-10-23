import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from gru_vae.config import load_config
from gru_vae.data import load_paired_timeseries, apply_standardization_slotwise, times_to_hour_index
from gru_vae.model import OnlineGPVAE
from gru_vae.utils import resolve_device


def _normalized_rmse_window(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, scale: np.ndarray) -> float:
    """Compute window-level NRMSE on observed positions using fixed scale.

    Shapes: y_true/y_pred/mask/scale: [L, H]
    - mask: 1 for observed, 0 for missing (only observed contribute)
    - scale: per-time, per-feature normalization scale, typically slot_std[hour]
    Returns a single scalar NRMSE = sqrt(mean(((y_pred - y_true)/scale)^2 over observed positions)).
    """
    if not (y_true.shape == y_pred.shape == mask.shape == scale.shape):
        raise ValueError('y_true, y_pred, mask, scale must share shape [L,H]')
    obs = (mask > 0.5).astype(np.float64)
    # Guard against zero/inf scale
    scale_s = np.clip(scale.astype(np.float64), 1e-6, None)
    err = (y_pred.astype(np.float64) - y_true.astype(np.float64)) / scale_s
    se = (err * err) * obs
    num = float(se.sum())
    den = float(obs.sum())
    if den <= 0:
        return float('nan')
    return float(np.sqrt(num / den))


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

    # Data: infer paired
    Xa, Xn, M, ts = load_paired_timeseries(args.infer_normal_csv, args.infer_attacked_csv)

    # Slice window by end timestamp and length
    end_ts = str(args.infer_end_timestamp or '').strip()
    L = int(args.infer_length)
    if not end_ts:
        raise SystemExit('[error] infer.end_timestamp must be provided in config.yaml')
    if L <= 0:
        raise SystemExit('[error] infer.length must be positive')
    # Find end index
    try:
        idx_end = int(np.where(ts.astype(str).values == end_ts)[0][0])
    except Exception:
        raise SystemExit(f"[error] end timestamp not found in series: {end_ts}")
    idx_start = idx_end - L + 1
    if idx_start < 0:
        raise SystemExit(f"[error] Window start < 0. Need length {L} ending at {end_ts}")
    Xa_w = Xa[idx_start:idx_end+1]
    Xn_w = Xn[idx_start:idx_end+1]
    M_w = M[idx_start:idx_end+1]
    ts_w = ts.iloc[idx_start:idx_end+1].reset_index(drop=True)

    # Warm-up context: run the model on preceding context (up to one window length)
    warmup_len = int(min(idx_start, L))
    s_warm = idx_start - warmup_len
    Xa_ext = Xa[s_warm:idx_end+1]
    M_ext = M[s_warm:idx_end+1]

    # Model
    # Infer H from data
    H = Xa.shape[1]
    # Safe load: prefer weights_only=True to avoid arbitrary code execution via pickle
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        # Older PyTorch without weights_only: fallback to standard load
        ckpt = torch.load(args.ckpt, map_location='cpu')
    # Minimal config: we must know latent and sizes; save them via training script
    # Fallback to defaults if unavailable in checkpoint
    latent_dim = ckpt.get('latent_dim', getattr(args, 'latent_dim', 32))
    gru_hidden = ckpt.get('gru_hidden', getattr(args, 'gru_hidden', 256))
    gru_layers = ckpt.get('gru_layers', getattr(args, 'gru_layers', 1))
    dec_hidden = tuple(int(x) for x in str(getattr(args, 'dec_hidden', '256,256')).split(',') if x)
    model = OnlineGPVAE(
        input_dim=H,
        output_dim=H,
        latent_dim=latent_dim,
        enc_hidden_size=gru_hidden,
        enc_layers=gru_layers,
        dec_hidden=dec_hidden,
        beta=getattr(args, 'beta', 0.1),
        obs_init_logvar=getattr(args, 'obs_init_logvar', -3.5),
    ).to(device)
    # Accept either full checkpoint dict with 'model' key or raw state_dict
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Standardize only the extended context and run inference (batch=1)
    hours_ext = times_to_hour_index(ts.iloc[s_warm:idx_end+1].reset_index(drop=True))
    Xa_ext_s = apply_standardization_slotwise(Xa_ext, M_ext, hours_ext, slot_mean, slot_std, clip_k=clip_k)
    x_in = torch.from_numpy(Xa_ext_s).unsqueeze(0).to(device)
    m_in = torch.from_numpy(M_ext).unsqueeze(0).to(device)
    # Monte Carlo averaging over posterior z (default enabled, no switch)
    with torch.no_grad():
        B = int(x_in.shape[0])
        # MC samples from config (infer.mc_samples), default 8, clamp >=1
        K = max(1, int(getattr(args, 'infer_mc_samples', 8)))
        x_rep = x_in.repeat_interleave(K, dim=0)
        m_rep = m_in.repeat_interleave(K, dim=0)
        out = model.elbo_sequence(x_rep, m_rep)
        mean_rep = out['mean']  # [B*K, T, H]
        mean_rep = mean_rep.view(B, K, mean_rep.shape[1], mean_rep.shape[2])
        mean_seq = mean_rep.mean(dim=1).squeeze(0).cpu().numpy()  # [T, H]

    # Unstandardize reconstruction to original scale; evaluate on target window only
    # Unstandardize per-timestep using slot stats
    mean_t = slot_mean[hours_ext]
    std_t = slot_std[hours_ext]
    rec_ext = mean_seq * std_t + mean_t
    rec_target = rec_ext[-L:]
    y_true = Xn_w.astype(np.float64)
    y_pred = rec_target.astype(np.float64)
    y_att = Xa_w.astype(np.float64)

    # Compute window-level NRMSE using fixed training scale (slot-wise std)
    # Scale per time/feature uses the same slot stats as standardization
    scale_win = std_t[-L:].astype(np.float64)
    nrmse_rep = _normalized_rmse_window(y_true, y_pred, M_w, scale_win)
    nrmse_att = _normalized_rmse_window(y_true, y_att, M_w, scale_win)
    nrmse_impr = float(nrmse_att - nrmse_rep) if np.isfinite(nrmse_att) and np.isfinite(nrmse_rep) else float('nan')

    # Per-timestep metrics on observed positions; NRMSE uses fixed training scale
    def _masked_metrics_per_timestep_fixedscale(
        y_t: np.ndarray, y_p: np.ndarray, m: np.ndarray, sc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not (y_t.shape == y_p.shape == m.shape == sc.shape):
            raise ValueError('y_true, y_pred, mask, scale must share shape [L,H]')
        L_, H_ = y_t.shape
        mse = np.full((L_,), np.nan, dtype=np.float64)
        mae = np.full((L_,), np.nan, dtype=np.float64)
        smape = np.full((L_,), np.nan, dtype=np.float64)
        nrmse = np.full((L_,), np.nan, dtype=np.float64)
        eps_sm = 1e-6
        for t in range(L_):
            obs = m[t] > 0.5
            if not np.any(obs):
                continue
            yt = y_t[t, obs].astype(np.float64)
            yp = y_p[t, obs].astype(np.float64)
            sc_t = np.clip(sc[t, obs].astype(np.float64), eps_sm, None)
            diff = yp - yt
            se = diff * diff
            ae = np.abs(diff)
            mse[t] = se.mean()
            mae[t] = ae.mean()
            denom = np.abs(yt) + np.abs(yp) + eps_sm
            smape[t] = 200.0 * np.mean(ae / denom)
            nrmse[t] = float(np.sqrt(np.mean((diff / sc_t) ** 2)))
        return mse, mae, smape, nrmse

    mse_rep, mae_rep, smape_rep, nrmse_rep_ts = _masked_metrics_per_timestep_fixedscale(y_true, y_pred, M_w, scale_win)
    mse_att, mae_att, smape_att, nrmse_att_ts = _masked_metrics_per_timestep_fixedscale(y_true, y_att, M_w, scale_win)

    # Save detailed metrics.csv with per-timestamp rows and a bottom mean row
    df_a = pd.read_csv(args.infer_attacked_csv)
    ts_col = df_a.columns[0]
    start_ts = str(ts_w.iloc[0])
    end_ts_s = str(ts_w.iloc[-1])
    metrics_detail = pd.DataFrame({
        ts_col: ts_w,
        'mse_att': mse_att, 'mae_att': mae_att, 'smape_att': smape_att, 'nrmse_att': nrmse_att_ts,
        'mse_rep': mse_rep, 'mae_rep': mae_rep, 'smape_rep': smape_rep, 'nrmse_rep': nrmse_rep_ts,
    })
    mean_row_d = {
        ts_col: 'mean',
        'mse_att': float(np.nanmean(mse_att)), 'mae_att': float(np.nanmean(mae_att)), 'smape_att': float(np.nanmean(smape_att)), 'nrmse_att': float(np.nanmean(nrmse_att_ts)),
        'mse_rep': float(np.nanmean(mse_rep)), 'mae_rep': float(np.nanmean(mae_rep)), 'smape_rep': float(np.nanmean(smape_rep)), 'nrmse_rep': float(np.nanmean(nrmse_rep_ts)),
    }
    metrics_detail = pd.concat([metrics_detail, pd.DataFrame([mean_row_d])], ignore_index=True)
    metrics_path = os.path.join(outdir, 'metrics.csv')
    tmp_path = metrics_path + '.tmp'
    metrics_detail.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, metrics_path)
    except Exception:
        print(f"[warn] Could not replace metrics.csv; wrote {os.path.basename(tmp_path)} instead.")
    # Clean up legacy name if any
    legacy_path = os.path.join(outdir, 'metrics_detailed.csv')
    try:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
    except Exception:
        pass

    # Pretty terminal summary: Top-10 by repair effect (att - rep) + overall means and window-NRMSE
    def fmt(v: float, w: int = 8, p: int = 6) -> str:
        try:
            return f"{float(v):{w}.{p}f}"
        except Exception:
            return str(v)
    print("\n===== Reconstruction Metrics (Observed Only) =====")
    print(f"Window: [{start_ts} -> {end_ts_s}]  Length: {L}")
    def top_k_improve(name: str, att: np.ndarray, rep: np.ndarray, k: int = 10) -> None:
        valid = (~np.isnan(att)) & (~np.isnan(rep))
        idx = np.where(valid)[0]
        if idx.size == 0:
            print(f"- {name}: no valid entries")
            return
        imp = att[idx] - rep[idx]
        order = np.argsort(imp)[::-1]
        take = idx[order][:min(k, len(order))]
        print(f"\n- Top {min(k, len(take))} by repair {name} (att - rep)")
        print("  Rank  Timestamp              att        rep        impr")
        for r, t in enumerate(take, 1):
            ts_s = str(ts_w.iloc[int(t)])
            impr = att[t] - rep[t]
            print(f"  {r:>4d}  {ts_s:>19s}  {fmt(att[t])}  {fmt(rep[t])}  {fmt(impr)}")
    top_k_improve('MSE',   mse_att,   mse_rep)
    top_k_improve('MAE',   mae_att,   mae_rep)
    top_k_improve('NRMSE', nrmse_att_ts, nrmse_rep_ts)
    top_k_improve('sMAPE', smape_att, smape_rep)

    # Summary (means over window)
    att_mean = {
        'MSE': float(np.nanmean(mse_att)), 'MAE': float(np.nanmean(mae_att)), 'sMAPE': float(np.nanmean(smape_att)), 'NRMSE': float(np.nanmean(nrmse_att_ts)),
    }
    rep_mean = {
        'MSE': float(np.nanmean(mse_rep)), 'MAE': float(np.nanmean(mae_rep)), 'sMAPE': float(np.nanmean(smape_rep)), 'NRMSE': float(np.nanmean(nrmse_rep_ts)),
    }
    print("\n- Summary (means over window)")
    print("  Metric      att_mean    rep_mean    impr_mean")
    for m in ('MSE','MAE','NRMSE','sMAPE'):
        impr_mean = float(att_mean[m] - rep_mean[m])
        print(f"  {m:<7s}  {fmt(att_mean[m])}  {fmt(rep_mean[m])}  {fmt(impr_mean)}")

    # Window-level NRMSE summary using fixed scale
    print("\n- Window-level NRMSE (slot-std scale)")
    print(f"  nrmse_att: {fmt(nrmse_att)}  nrmse_rep: {fmt(nrmse_rep)}  improvement: {fmt(nrmse_impr)}")

    # Save three window slices: reconstructed, attacked, normal
    cols = list(df_a.columns)
    # Reconstructed window
    rec_df = pd.DataFrame(columns=cols)
    rec_df[ts_col] = ts_w
    rec_df.iloc[:, 1:] = y_pred.astype(np.float32)
    rec_df.to_csv(os.path.join(outdir, 'reconstructed_window.csv'), index=False)
    # Attacked slice
    attacked_slice = df_a.iloc[idx_start:idx_end+1].reset_index(drop=True)
    attacked_slice.to_csv(os.path.join(outdir, 'attacked_window.csv'), index=False)
    # Normal slice
    df_n = pd.read_csv(args.infer_normal_csv)
    normal_slice = df_n.iloc[idx_start:idx_end+1].reset_index(drop=True)
    normal_slice.to_csv(os.path.join(outdir, 'normal_window.csv'), index=False)

    print('Inference finished. Outputs saved to:', outdir)


if __name__ == '__main__':
    main()
