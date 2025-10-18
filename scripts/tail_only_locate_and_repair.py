"""Tail-Only (context Normal, tail Attacked) locate+repair without external labels.

Pipeline (best-only implementation):
- Calibrate per-feature thresholds on Normal tail NLL at (1-alpha) quantile
  (with global-quantile fallback for sparse features).
- For Tail-Only window [s,e), compute tail NLL at e-1, threshold to predict anomalies,
  hard-mask predicted anomalies at the tail, and impute via Online GP-VAE.
- Monte Carlo averaging with 16 samples (fixed) is used during imputation to improve stability.
  Evaluate repair quality against the clean baseline over observed tail features.

Sliding behavior:
- Consecutive-attack simulation: at each sliding step, only the immediately previous
  repaired tail row is fed back as history for the next window; all earlier history
  rows remain from the Normal (clean) sequence.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gru_vae.data import load_timeseries, SlidingWindowDataset, masked_mean_std
from gru_vae.model import OnlineGPVAE
from gru_vae.utils import parse_sizes, resolve_device
def _check_headers_match(a_csv: str, b_csv: str) -> List[str]:
    df_a = pd.read_csv(a_csv, nrows=1)
    df_b = pd.read_csv(b_csv, nrows=1)
    if list(df_a.columns) != list(df_b.columns):
        raise SystemExit('[error] Feature headers differ between reference and target CSVs.')
    return list(df_a.columns)


def _standardize_full(
    X_norm: np.ndarray,
    M_struct_norm: np.ndarray,
    X_att: np.ndarray,
    M_struct_att: np.ndarray,
    *,
    train_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T_norm = X_norm.shape[0]
    train_T = int(T_norm * float(train_ratio))
    mean, std = masked_mean_std(X_norm[:train_T], M_struct_norm[:train_T])
    std_safe = np.where(std > 0, std, 1.0).astype(np.float32)
    Xn_std = (X_norm - mean) / std_safe
    Xa_std = (X_att - mean) / std_safe
    Xn_std = np.where(M_struct_norm > 0.5, Xn_std, 0.0).astype(np.float32)
    Xa_std = np.where(M_struct_att > 0.5, Xa_std, 0.0).astype(np.float32)
    return Xn_std, Xa_std, mean.astype(np.float32), std_safe, M_struct_norm.astype(np.float32), M_struct_att.astype(np.float32)


def _compute_tail_scores_window(
    model: OnlineGPVAE,
    x_w: np.ndarray,
    m_w: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    L, H = x_w.shape
    xb = torch.from_numpy(x_w[None, :, :]).to(device).float()
    mb = torch.from_numpy(m_w[None, :, :]).to(device).float()
    log2pi = math.log(2.0 * math.pi)
    model.eval()
    with torch.no_grad():
        state = model.init_state(1, device=xb.device, dtype=xb.dtype)
        mean_last = None
        logv_last = None
        for t in range(L):
            x_t = xb[:, t, :]
            m_t = mb[:, t, :]
            _yhat_t, state, aux = model.step(x_t, m_t, state, use_mean=True)
            mean_last = aux['mean_t']
            logv_last = aux['logvar_x_t']
        assert mean_last is not None and logv_last is not None
        x_last = xb[:, -1, :]
        m_last = mb[:, -1, :]
        inv_var = torch.exp(-logv_last)
        # Per-feature NLL element as score (higher is more anomalous)
        score_elem = 0.5 * (log2pi + logv_last + (x_last - mean_last) ** 2 * inv_var)
        nll_np = score_elem.squeeze(0).cpu().numpy().astype(np.float64)
        m_np = m_last.squeeze(0).cpu().numpy().astype(np.float64)
        out = np.full((H,), np.nan, dtype=np.float64)
        obs_idx = m_np > 0.5
        out[obs_idx] = nll_np[obs_idx]
        return out


def _calibrate_thresholds_normal(
    model: OnlineGPVAE,
    Xn_std: np.ndarray,
    Mn_struct: np.ndarray,
    L: int,
    *,
    batch_size: int,
    device: torch.device,
    alpha: float,
    min_count_per_feature: int,
    feature_names: List[str],
) -> Dict[str, object]:
    dataset = SlidingWindowDataset(Xn_std, Mn_struct, L, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    T, H = Xn_std.shape
    log2pi = math.log(2.0 * math.pi)
    scores = np.full((T, H), np.nan, dtype=np.float64)
    offset = 0
    model.eval()
    with torch.no_grad():
        for xb, mb in loader:
            xb = xb.to(device).float()
            mb = mb.to(device).float()
            B = xb.shape[0]
            state = model.init_state(B, device=xb.device, dtype=xb.dtype)
            mean_last = None
            logv_last = None
            for t in range(L):
                x_t = xb[:, t, :]
                m_t = mb[:, t, :]
                _yhat_t, state, aux = model.step(x_t, m_t, state, use_mean=True)
                mean_last = aux['mean_t']
                logv_last = aux['logvar_x_t']
            assert mean_last is not None and logv_last is not None
            x_last = xb[:, -1, :]
            m_last = mb[:, -1, :]
            inv_var = torch.exp(-logv_last)
            score_elem = 0.5 * (log2pi + logv_last + (x_last - mean_last) ** 2 * inv_var)
            nll_np = score_elem.cpu().numpy().astype(np.float64)
            m_np = m_last.cpu().numpy().astype(np.float64)
            starts = dataset.starts[offset:offset+B]
            for i in range(B):
                start = int(starts[i])
                t_idx = start + (L - 1)
                obs_idx = m_np[i] > 0.5
                scores[t_idx, obs_idx] = nll_np[i, obs_idx]
            offset += B
    if not (0.0 < alpha < 1.0):
        raise SystemExit('[error] alpha must be in (0,1)')
    mask_valid = (Mn_struct > 0.5) & np.isfinite(scores)
    # Per-feature quantile thresholds (fallback to global quantile for sparse features)
    thr: Dict[str, float] = {}
    for h in range(Xn_std.shape[1]):
        col = scores[:, h]
        mcol = mask_valid[:, h]
        vals = col[mcol]
        if vals.size >= int(min_count_per_feature):
            thr[feature_names[h]] = float(np.quantile(vals, 1.0 - alpha))
    if len(thr) < Xn_std.shape[1]:
        all_vals = scores[mask_valid]
        tau = float(np.quantile(all_vals, 1.0 - alpha))
        for h in range(Xn_std.shape[1]):
            name = feature_names[h]
            if name not in thr:
                thr[name] = tau
    return {'type': 'per_feature', 'alpha': float(alpha), 'thresholds': thr}


def _apply_thresholds(scores_tail: np.ndarray, m_tail: np.ndarray, meta: Dict[str, object], feature_names: List[str]) -> np.ndarray:
    A = np.zeros_like(scores_tail, dtype=np.float64)
    A[:] = np.nan
    obs = m_tail > 0.5
    thr: Dict[str, float] = meta['thresholds']
    for h, name in enumerate(feature_names):
        if not obs[h]:
            continue
        tau_h = float(thr[name])
        A[h] = float(scores_tail[h] <= tau_h)
    return A


MC_SAMPLES = 16


def _impute_tail(
    model: OnlineGPVAE,
    x_w: np.ndarray,
    m_w: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    """Impute tail-window with optional Monte Carlo averaging on latent samples.

    - mc_samples=1: deterministic mean-path imputation.
    - mc_samples>1: average over stochastic paths (use_mean=False) to reduce variance.
    Returns array of shape [L, H] in standardized space.
    """
    xb = torch.from_numpy(x_w[None, :, :]).to(device).float()
    mb = torch.from_numpy(m_w[None, :, :]).to(device).float()
    with torch.no_grad():
        acc = None
        S = int(MC_SAMPLES)
        for _ in range(S):
            ys = model.impute_online(xb, mb, use_mean=False)
            ys_np = ys.squeeze(0).cpu().numpy().astype(np.float64)
            if acc is None:
                acc = ys_np
            else:
                acc += ys_np
        return acc / float(S)


def _select_tail_timestamp_only(*, attack_timestamp: str, timestamps: pd.Series, L: int) -> int:
    if not attack_timestamp:
        raise SystemExit('[error] --attack_timestamp is required')
    min_t = max(0, int(L) - 1)
    ts_series = timestamps.astype(str)
    ts_parsed = pd.to_datetime(ts_series, errors='coerce')
    if ts_parsed.isna().any():
        raise SystemExit('[error] Failed to parse timestamps in data CSVs')
    target_dt = pd.to_datetime(attack_timestamp, errors='coerce')
    if pd.isna(target_dt):
        raise SystemExit('[error] Provided --attack_timestamp cannot be parsed')
    # Exact match
    matches = np.where(ts_series.values == attack_timestamp)[0]
    for idx in matches:
        if idx >= min_t:
            return int(idx)
    if matches.size > 0:
        print('[warn] Provided attack_timestamp exists but violates window length; searching forward')
    else:
        print('[warn] Provided attack_timestamp not found; searching forward by time')
    # First feasible at/after target_dt
    feasible = np.where(ts_parsed >= target_dt)[0]
    feasible = feasible[feasible >= min_t]
    if feasible.size == 0:
        raise SystemExit('[error] No feasible rows at/after the provided time for the requested window length')
    fallback = int(feasible[0])
    print(f'[warn] Fallback tail selected at index {fallback} (min_t={min_t})')
    return fallback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--case', type=str, default='case14')
    p.add_argument('--normal_csv', type=str, default='')
    p.add_argument('--attacked_csv', type=str, default='')
    p.add_argument('--time_length', type=int, default=24)
    p.add_argument('--sliding_steps', type=int, default=6, help='Number of consecutive tail steps to repair (>=1)')
    # Timestamp-only selection
    p.add_argument('--attack_timestamp', type=str, required=True, help='Exact time; if invalid, fallback to first feasible attacked tail at/after this time')
    # Thresholds
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--min_count_per_feature', type=int, default=100)
    # Model
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--dec_hidden', type=str, default='256,256')
    p.add_argument('--gru_hidden', type=int, default=256)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--obs_learn_var', dest='obs_learn_var', action='store_true')
    p.add_argument('--no-obs_learn_var', dest='obs_learn_var', action='store_false')
    p.set_defaults(obs_learn_var=True)
    p.add_argument('--obs_init_logvar', type=float, default=-3.5)
    p.add_argument('--ckpt', type=str, default='models/gru_smoke/ckpt.pt')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()

    if args.time_length <= 0:
        raise SystemExit('[error] --time_length must be positive')
    if args.sliding_steps <= 0:
        raise SystemExit('[error] --sliding_steps must be positive')

    device = resolve_device(args.device)

    # Resolve CSVs
    case_dir = os.path.join(args.data_dir, args.case)
    normal_csv = args.normal_csv or os.path.join(case_dir, f"{args.case}_acopf_all_rows_noisy.csv")
    if not os.path.exists(normal_csv):
        raise SystemExit(f'[error] Normal CSV not found: {normal_csv}')
    attacked_csv = args.attacked_csv
    if not attacked_csv:
        attacked_guess = ''
        if os.path.isdir(case_dir):
            for fname in sorted(os.listdir(case_dir)):
                lower = fname.lower()
                if lower.endswith('.csv') and 'fdia' in lower and 'noisy' in lower:
                    attacked_guess = os.path.join(case_dir, fname)
                    break
        attacked_csv = attacked_guess
    if not attacked_csv:
        raise SystemExit('[error] Could not infer attacked_csv; provide it explicitly')

    headers = _check_headers_match(normal_csv, attacked_csv)
    feat_names = headers[1:]
    ts_col = headers[0]

    # Load sequences
    Xn, Mn_struct, ts = load_timeseries(normal_csv)
    Xa, Ma_struct, ts_a = load_timeseries(attacked_csv)
    if len(ts) != Xn.shape[0] or len(ts_a) != Xa.shape[0]:
        raise SystemExit('[error] Timestamp and data length mismatch across CSVs')
    if not np.all(ts.astype(str).values == ts_a.astype(str).values):
        raise SystemExit('[error] Timestamps are not perfectly aligned between Normal and Attacked CSVs')

    # Standardize
    Xn_std, Xa_std, mean, std_safe, Mn_struct_f, Ma_struct_f = _standardize_full(
        Xn, Mn_struct, Xa, Ma_struct, train_ratio=args.train_ratio
    )

    # Select tail by timestamp-only with fallback to first feasible
    L = int(args.time_length)
    t = _select_tail_timestamp_only(attack_timestamp=args.attack_timestamp, timestamps=ts, L=L)
    s = t - (L - 1)
    e = t + 1
    if s < 0:
        raise SystemExit('[error] Window underflows start; choose a later tail or reduce --time_length')

    # Model
    H = Xn.shape[1]
    dec_hidden = parse_sizes(args.dec_hidden)
    model = OnlineGPVAE(
        input_dim=H,
        output_dim=H,
        latent_dim=args.latent_dim,
        enc_hidden_size=args.gru_hidden,
        enc_layers=args.gru_layers,
        dec_hidden=tuple(dec_hidden),
        beta=args.beta,
        obs_learn_var=args.obs_learn_var,
        obs_init_logvar=args.obs_init_logvar,
    )
    model.to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        try:
            state = torch.load(args.ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        elif isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        print(f'[ok] Loaded checkpoint: {args.ckpt}')
    else:
        if args.ckpt:
            print(f'[warn] Checkpoint not found: {args.ckpt}. Continue with random weights.')

    # Calibrate thresholds on Normal
    thresholds_meta = _calibrate_thresholds_normal(
        model, Xn_std, Mn_struct_f, L, batch_size=args.batch_size, device=device,
        alpha=args.alpha, min_count_per_feature=args.min_count_per_feature, feature_names=feat_names,
    )

    sliding_steps = int(args.sliding_steps)
    max_tail_idx = t + sliding_steps - 1
    if max_tail_idx >= Xn.shape[0]:
        raise SystemExit('[error] Sliding steps exceed available data length for requested start index')

    current_stream = Xa.copy().astype(np.float32)
    repaired_tail_rows: List[Tuple[int, str, np.ndarray]] = []
    repaired_mask = np.zeros(Xn.shape[0], dtype=bool)
    step_results: List[Dict[str, object]] = []

    def rel_improve(a: float, b: float) -> float:
        return 100.0 * (b - a) / max(b, 1e-12)

    print(f"[info] Sliding repair starts at tail index {t} ({ts.iloc[t]}) with L={L} for {sliding_steps} steps")
    print('[info] Mode: history reuses all previously repaired tails (others from Normal)')

    for step_offset in range(sliding_steps):
        tail_idx = t + step_offset
        s_step = tail_idx - (L - 1)
        e_step = tail_idx + 1
        if e_step > Xn.shape[0]:
            raise SystemExit('[error] Sliding window exceeds available data length; reduce --sliding_steps')
        timestamp_str = str(ts.iloc[tail_idx])

        X_T_raw = Xn[s_step:e_step].copy().astype(np.float32)
        # Feed every previously repaired tail row within the history window
        if step_offset > 0:
            history_idx = np.arange(s_step, tail_idx)
            if history_idx.size > 0:
                repaired_idx = history_idx[repaired_mask[history_idx]]
                for idx_hist in repaired_idx:
                    X_T_raw[idx_hist - s_step] = current_stream[idx_hist].astype(np.float32)
        # Tail row is always taken from the current (attacked/repaired) stream
        X_T_raw[-1] = current_stream[tail_idx].astype(np.float32)
        M_T_struct = Mn_struct_f[s_step:e_step].copy().astype(np.float32)
        M_T_struct[-1] = Ma_struct_f[tail_idx].astype(np.float32)
        X_T_std = np.where(M_T_struct > 0.5, (X_T_raw - mean) / std_safe, 0.0).astype(np.float32)

        scores_tail = _compute_tail_scores_window(model, X_T_std, M_T_struct, device=device)
        m_tail = M_T_struct[-1]
        pred_tail = _apply_thresholds(scores_tail, m_tail, thresholds_meta, feat_names)
        m_imp = M_T_struct.copy()
        m_tail_obs = m_tail > 0.5
        if not np.any(m_tail_obs):
            raise SystemExit(f'[error] Sliding step {step_offset} (tail index {tail_idx}) has no observed tail features')
        is_anom = (pred_tail == 0.0) & m_tail_obs
        m_imp[-1, is_anom] = 0.0
        x_in = np.where(m_imp > 0.5, X_T_std, 0.0).astype(np.float32)
        y_std = _impute_tail(model, x_in, m_imp, device=device)
        y = y_std * std_safe + mean
        current_stream[tail_idx] = y[-1].astype(np.float32)
        repaired_tail_rows.append((tail_idx, timestamp_str, y[-1].astype(np.float64)))
        repaired_mask[tail_idx] = True

        x_true_tail = Xn[tail_idx].astype(np.float64)
        x_att_tail = Xa[tail_idx].astype(np.float64)
        y_tail = y[-1].astype(np.float64)
        mask_idx = m_tail_obs
        err_att = x_att_tail[mask_idx] - x_true_tail[mask_idx]
        err_rep = y_tail[mask_idx] - x_true_tail[mask_idx]
        mse_b = float(np.mean(err_att ** 2))
        mae_b = float(np.mean(np.abs(err_att)))
        rmse_b = float(np.sqrt(mse_b))
        mse = float(np.mean(err_rep ** 2))
        mae = float(np.mean(np.abs(err_rep)))
        rmse = float(np.sqrt(mse))
        std_map = std_safe.astype(np.float64)
        std_tail = std_map[mask_idx]
        std_tail = np.where(std_tail > 0.0, std_tail, 1.0)
        nerr_att = err_att / std_tail
        nerr_rep = err_rep / std_tail
        nrmse_b = float(np.sqrt(np.mean(nerr_att ** 2)))
        nmae_b = float(np.mean(np.abs(nerr_att)))
        nrmse = float(np.sqrt(np.mean(nerr_rep ** 2)))
        nmae = float(np.mean(np.abs(nerr_rep)))

        obs_count = int(np.sum(m_tail_obs))
        drop_count = int(np.sum(is_anom))
        drop_rate = 100.0 * drop_count / max(obs_count, 1e-12)
        rim_mse = rel_improve(mse, mse_b) if mse_b > 1e-12 else float('nan')
        rim_rmse = rel_improve(rmse, rmse_b) if rmse_b > 1e-12 else float('nan')

        step_results.append({
            'step': step_offset,
            'tail_idx': tail_idx,
            'window_start': s_step,
            'window_end': e_step,
            'timestamp': timestamp_str,
            'obs_count': obs_count,
            'drop_count': drop_count,
            'drop_rate': drop_rate,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'nmae': nmae,
            'mse_b': mse_b,
            'mae_b': mae_b,
            'rmse_b': rmse_b,
            'nrmse_b': nrmse_b,
            'nmae_b': nmae_b,
            'rim_mse': rim_mse,
            'rim_rmse': rim_rmse,
        })

        print(
            "[info] Step "
            f"{step_offset}: window s={s_step}, e={e_step} (tail index {tail_idx}, timestamp {timestamp_str}), "
            f"observed_tail_count={obs_count}, dropped={drop_count} ({drop_rate:.2f}%)"
        )

    # Always save only repaired tail rows (one row per sliding step)
    base = os.path.splitext(os.path.basename(attacked_csv))[0]
    out_tail_csv = os.path.join(
        os.path.dirname(attacked_csv),
        f"{base}_repaired_tail_rows_L{L}_steps{sliding_steps}.csv",
    )
    if repaired_tail_rows:
        stamps = [it[1] for it in repaired_tail_rows]
        vals = np.stack([it[2] for it in repaired_tail_rows], axis=0)
        df_tail = pd.DataFrame(vals, columns=feat_names)
        df_tail.insert(0, ts_col, stamps)
        df_tail.to_csv(out_tail_csv, index=False, encoding='utf-8')
        print(f"[ok] Saved repaired tail rows CSV: {out_tail_csv} (rows={len(stamps)})")
    else:
        print('[warn] No repaired tail rows to save (empty sliding result).')

    print('\nSummary (lower is better):')
    header = (
        "Step  TailIdx  Timestamp        Obs    Dropped  DropRate(%)  "
        "MSE_att     MSE_rep     RMSE_att    RMSE_rep    NRMSE_att   NRMSE_rep   "
        "Rel.Improv_MSE(%)  Rel.Improv_RMSE(%)"
    )
    print(header)
    for res in step_results:
        rim_mse = res['rim_mse']
        rim_rmse = res['rim_rmse']
        print(
            f"{res['step']:4d}  {res['tail_idx']:7d}  {res['timestamp']:16s}  "
            f"{res['obs_count']:6d}  {res['drop_count']:7d}  {res['drop_rate']:11.2f}  "
            f"{res['mse_b']:10.6f} {res['mse']:10.6f} {res['rmse_b']:11.6f} {res['rmse']:11.6f} "
            f"{res['nrmse_b']:10.6f} {res['nrmse']:10.6f} {rim_mse:18.2f} {rim_rmse:18.2f}"
        )

    if step_results:
        mse_b_mean = float(np.mean([res['mse_b'] for res in step_results]))
        mse_mean = float(np.mean([res['mse'] for res in step_results]))
        rmse_b_mean = float(np.mean([res['rmse_b'] for res in step_results]))
        rmse_mean = float(np.mean([res['rmse'] for res in step_results]))
        agg_gain_mse = rel_improve(mse_mean, mse_b_mean) if mse_b_mean > 1e-12 else float('nan')
        agg_gain_rmse = rel_improve(rmse_mean, rmse_b_mean) if rmse_b_mean > 1e-12 else float('nan')
        print(
            "\nAggregate (mean across steps): "
            f"MSE_att={mse_b_mean:.6f}, MSE_rep={mse_mean:.6f}, RMSE_att={rmse_b_mean:.6f}, "
            f"RMSE_rep={rmse_mean:.6f}, Gain_MSE={agg_gain_mse:.2f}%, Gain_RMSE={agg_gain_rmse:.2f}%"
        )


if __name__ == '__main__':
    main()

