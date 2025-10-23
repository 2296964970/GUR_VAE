import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from gru_vae.config import load_config
from gru_vae.data import load_paired_timeseries, _apply_standardization
from gru_vae.model import OnlineGPVAE
from gru_vae.utils import resolve_device


def _masked_metrics_per_timestep(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute MSE, MAE, RMSE, MAPE, MSPE per time step on observed positions.

    Shapes: y_true/y_pred/mask: [L, H]
    Returns arrays of shape [L] for each metric; NaN when no valid positions.
    """
    if not (y_true.shape == y_pred.shape == mask.shape):
        raise ValueError('Shapes must match [L,H]')
    L, H = y_true.shape
    mse = np.full((L,), np.nan, dtype=np.float64)
    mae = np.full((L,), np.nan, dtype=np.float64)
    rmse = np.full((L,), np.nan, dtype=np.float64)
    mape = np.full((L,), np.nan, dtype=np.float64)
    mspe = np.full((L,), np.nan, dtype=np.float64)
    for t in range(L):
        obs = mask[t] > 0.5
        if not np.any(obs):
            continue
        yt = y_true[t, obs].astype(np.float64)
        yp = y_pred[t, obs].astype(np.float64)
        diff = yp - yt
        se = diff * diff
        ae = np.abs(diff)
        mse[t] = se.mean()
        mae[t] = ae.mean()
        rmse[t] = np.sqrt(mse[t])
        nz = np.abs(yt) > eps
        if np.any(nz):
            rel = diff[nz] / yt[nz]
            mape[t] = np.mean(np.abs(rel))
            mspe[t] = np.mean(rel * rel)
    return mse, mae, rmse, mape, mspe


def main() -> None:
    args = load_config()
    device = resolve_device(args.device)

    # Paths
    outdir = os.path.join('output', args.case, 'infer', args.exp_name)
    os.makedirs(outdir, exist_ok=True)
    # Load stats from training
    model_dir = os.path.dirname(args.ckpt)
    mean_path = os.path.join(model_dir, 'mean.npy')
    std_path = os.path.join(model_dir, 'std.npy')
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        raise SystemExit('[error] mean.npy/std.npy not found next to checkpoint; run training first')
    mean = np.load(mean_path)
    std = np.load(std_path)

    # Data: infer paired
    Xa, Xn, M, ts = load_paired_timeseries(args.infer_normal_csv, args.infer_attacked_csv)
    Xa_s = _apply_standardization(Xa, M, mean, std)
    Xn_s = _apply_standardization(Xn, M, mean, std)

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

    # Standardize only the selected window and run inference (batch=1)
    Xa_s = _apply_standardization(Xa_w, M_w, mean, std)
    Xn_s = _apply_standardization(Xn_w, M_w, mean, std)
    x_in = torch.from_numpy(Xa_s).unsqueeze(0).to(device)
    m_in = torch.from_numpy(M_w).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.elbo_sequence(x_in, m_in)
        mean_seq = out['mean'].squeeze(0).cpu().numpy()

    # Unstandardize reconstruction to original scale
    rec = mean_seq * std + mean
    y_true = Xn_w.astype(np.float64)
    y_pred = rec.astype(np.float64)
    y_att = Xa_w.astype(np.float64)
    # Repaired vs normal
    mse_rep, mae_rep, rmse_rep, mape_rep, mspe_rep = _masked_metrics_per_timestep(y_true, y_pred, M_w)
    # Attacked vs normal
    mse_att, mae_att, rmse_att, mape_att, mspe_att = _masked_metrics_per_timestep(y_true, y_att, M_w)

    # Save metrics.csv with per-timestamp rows and a bottom mean row
    df_a = pd.read_csv(args.infer_attacked_csv)
    ts_col = df_a.columns[0]
    # Detailed metrics including attacked vs normal and relative improvement (%).
    def _rel_improv(att: np.ndarray, rep: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return 100.0 * (att - rep) / np.maximum(att, eps)

    rel_mse = _rel_improv(mse_att, mse_rep)
    rel_mae = _rel_improv(mae_att, mae_rep)
    rel_rmse = _rel_improv(rmse_att, rmse_rep)
    rel_mape = _rel_improv(mape_att, mape_rep)
    rel_mspe = _rel_improv(mspe_att, mspe_rep)

    metrics_detail = pd.DataFrame({
        ts_col: ts_w,
        'mse_att': mse_att, 'mae_att': mae_att, 'rmse_att': rmse_att, 'mape_att': mape_att, 'mspe_att': mspe_att,
        'mse_rep': mse_rep, 'mae_rep': mae_rep, 'rmse_rep': rmse_rep, 'mape_rep': mape_rep, 'mspe_rep': mspe_rep,
        'rel_improv_mse_%': rel_mse, 'rel_improv_mae_%': rel_mae, 'rel_improv_rmse_%': rel_rmse,
        'rel_improv_mape_%': rel_mape, 'rel_improv_mspe_%': rel_mspe,
    })
    mean_row_d = {
        ts_col: 'mean',
        'mse_att': float(np.nanmean(mse_att)), 'mae_att': float(np.nanmean(mae_att)), 'rmse_att': float(np.nanmean(rmse_att)), 'mape_att': float(np.nanmean(mape_att)), 'mspe_att': float(np.nanmean(mspe_att)),
        'mse_rep': float(np.nanmean(mse_rep)), 'mae_rep': float(np.nanmean(mae_rep)), 'rmse_rep': float(np.nanmean(rmse_rep)), 'mape_rep': float(np.nanmean(mape_rep)), 'mspe_rep': float(np.nanmean(mspe_rep)),
        'rel_improv_mse_%': float(np.nanmean(rel_mse)), 'rel_improv_mae_%': float(np.nanmean(rel_mae)), 'rel_improv_rmse_%': float(np.nanmean(rel_rmse)),
        'rel_improv_mape_%': float(np.nanmean(rel_mape)), 'rel_improv_mspe_%': float(np.nanmean(rel_mspe)),
    }
    metrics_detail = pd.concat([metrics_detail, pd.DataFrame([mean_row_d])], ignore_index=True)
    # Single CSV output (detailed metrics)
    metrics_path = os.path.join(outdir, 'metrics.csv')
    tmp_path = metrics_path + '.tmp'
    metrics_detail.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, metrics_path)
    except Exception:
        # Fallback: keep tmp file if replacement fails
        print(f"[warn] Could not replace metrics.csv; wrote {os.path.basename(tmp_path)} instead.")
    # Clean up legacy metrics file if present
    legacy_path = os.path.join(outdir, 'metrics_detailed.csv')
    try:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
    except Exception:
        pass

    # Pretty terminal summary: Top-10 improvements per metric + overall means
    start_ts = str(ts_w.iloc[0])
    end_ts_s = str(ts_w.iloc[-1])
    def fmt(v: float, w: int = 8, p: int = 6) -> str:
        try:
            return f"{float(v):{w}.{p}f}"
        except Exception:
            return str(v)
    def top_k(name: str, att: np.ndarray, rep: np.ndarray, imp: np.ndarray, k: int = 10) -> None:
        valid = ~np.isnan(imp)
        idx = np.where(valid)[0]
        if idx.size == 0:
            print(f"- {name}: no valid entries")
            return
        order = np.argsort(imp[idx])[::-1]
        take = idx[order][:min(k, len(order))]
        print(f"\n- Top {min(k, len(take))} by rel_improv {name} (%)")
        print("  Rank  Timestamp              att        rep        improv(%)")
        for r, t in enumerate(take, 1):
            ts_s = str(ts_w.iloc[int(t)])
            print(f"  {r:>4d}  {ts_s:>19s}  {fmt(att[t])}  {fmt(rep[t])}  {fmt(imp[t])}")

    print("\n===== Reconstruction Metrics (Observed Only) =====")
    print(f"Window: [{start_ts} -> {end_ts_s}]  Length: {L}")
    top_k('MSE',  mse_att,  mse_rep,  rel_mse)
    top_k('MAE',  mae_att,  mae_rep,  rel_mae)
    top_k('RMSE', rmse_att, rmse_rep, rel_rmse)
    top_k('MAPE', mape_att, mape_rep, rel_mape)
    top_k('MSPE', mspe_att, mspe_rep, rel_mspe)

    # Summary (means over window)
    att_mean = {
        'MSE': float(np.nanmean(mse_att)), 'MAE': float(np.nanmean(mae_att)), 'RMSE': float(np.nanmean(rmse_att)), 'MAPE': float(np.nanmean(mape_att)), 'MSPE': float(np.nanmean(mspe_att)),
    }
    rep_mean = {
        'MSE': float(np.nanmean(mse_rep)), 'MAE': float(np.nanmean(mae_rep)), 'RMSE': float(np.nanmean(rmse_rep)), 'MAPE': float(np.nanmean(mape_rep)), 'MSPE': float(np.nanmean(mspe_rep)),
    }
    imp_mean = {
        'MSE': float(np.nanmean(rel_mse)), 'MAE': float(np.nanmean(rel_mae)), 'RMSE': float(np.nanmean(rel_rmse)), 'MAPE': float(np.nanmean(rel_mape)), 'MSPE': float(np.nanmean(rel_mspe)),
    }
    print("\n- Summary (means over window)")
    print("  Metric      att_mean    rep_mean    improv_mean(%)")
    for m in ('MSE','MAE','RMSE','MAPE','MSPE'):
        print(f"  {m:<7s}  {fmt(att_mean[m])}  {fmt(rep_mean[m])}  {fmt(imp_mean[m])}")

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
