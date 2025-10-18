"""Tail-Only repair using ground-truth attack labels.

Pipeline:
- Load Normal (clean), Attacked, and ideal label CSVs with aligned timestamps.
- Select a tail index where the tail row is attacked while the history rows remain clean.
- Use labels to mask attacked tail features, impute them with Online GP-VAE, and evaluate
  restoration quality against the clean baseline on the attacked features.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from gru_vae.data import load_timeseries, masked_mean_std
from gru_vae.model import OnlineGPVAE
from gru_vae.utils import parse_sizes, resolve_device


def _read_labels_csv(path: str, expected_headers: List[str]) -> Tuple[np.ndarray, pd.Series]:
    if not path:
        raise SystemExit('[error] --labels_csv is required')
    if not os.path.exists(path):
        raise SystemExit(f'[error] Labels CSV not found: {path}')
    df = pd.read_csv(path)
    if list(df.columns) != expected_headers:
        raise SystemExit('[error] Labels CSV headers differ from data CSV')
    ts_col = df.columns[0]
    ts_parsed = pd.to_datetime(df[ts_col], errors='raise')
    sort_idx = ts_parsed.argsort(kind='mergesort')
    df = df.iloc[sort_idx].reset_index(drop=True)
    ts_sorted = ts_parsed.iloc[sort_idx].reset_index(drop=True)
    ts_norm = ts_sorted.dt.strftime('%Y/%m/%d %H:%M')
    df[ts_col] = ts_norm
    labels = df.drop(columns=[ts_col]).to_numpy(dtype=np.float32)
    return labels, ts_norm


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


def _impute_tail(model: OnlineGPVAE, x_w: np.ndarray, m_w: np.ndarray, *, device: torch.device) -> np.ndarray:
    xb = torch.from_numpy(x_w[None, :, :]).to(device).float()
    mb = torch.from_numpy(m_w[None, :, :]).to(device).float()
    with torch.no_grad():
        yb = model.impute_online(xb, mb, use_mean=True)
    return yb.squeeze(0).cpu().numpy().astype(np.float64)


def _select_attacked_tail(
    labels01: np.ndarray,
    *,
    attack_timestamp: str,
    timestamps: pd.Series,
    L: int,
) -> int:
    min_t = max(0, int(L) - 1)
    ts_series = timestamps.astype(str)
    ts_parsed = pd.to_datetime(ts_series, errors='coerce')
    if ts_parsed.isna().any():
        raise SystemExit('[error] Failed to parse timestamps in data CSVs')
    target_dt = pd.to_datetime(attack_timestamp, errors='coerce')
    if pd.isna(target_dt):
        raise SystemExit('[error] Provided --attack_timestamp cannot be parsed')
    matches = np.where(ts_series.values == attack_timestamp)[0]
    for idx in matches:
        if idx >= min_t and np.any(labels01[idx] < 0.5):
            return int(idx)
    if matches.size > 0:
        print('[warn] Provided attack_timestamp exists but violates window length or is not attacked; searching forward')
    else:
        print('[warn] Provided attack_timestamp not found; searching forward by time')
    attacked_rows = np.where(np.any(labels01 < 0.5, axis=1))[0]
    feasible = attacked_rows[(attacked_rows >= min_t)]
    feasible = feasible[ts_parsed.iloc[feasible] >= target_dt]
    if feasible.size == 0:
        raise SystemExit('[error] No attacked tail found at/after requested time for the window length')
    tail_idx = int(feasible[0])
    print(f'[warn] Fallback tail selected at index {tail_idx} (min_t={min_t})')
    return tail_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--case', type=str, default='case14')
    p.add_argument('--normal_csv', type=str, default='')
    p.add_argument('--attacked_csv', type=str, default='')
    p.add_argument('--labels_csv', type=str, default='')
    p.add_argument('--time_length', type=int, default=24)
    p.add_argument('--attack_timestamp', type=str, required=True)
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

    device = resolve_device(args.device)

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
    labels_csv = args.labels_csv
    if not labels_csv:
        if os.path.isdir(case_dir):
            for fname in sorted(os.listdir(case_dir)):
                lower = fname.lower()
                if lower.endswith('.csv') and 'labels' in lower:
                    labels_csv = os.path.join(case_dir, fname)
                    break
    if not labels_csv:
        raise SystemExit('[error] Could not infer labels_csv; provide it explicitly')

    headers = _check_headers_match(normal_csv, attacked_csv)
    headers_l = _check_headers_match(attacked_csv, labels_csv)
    if headers != headers_l:
        raise SystemExit('[error] Headers mismatch across CSVs')
    Xn, Mn_struct, ts = load_timeseries(normal_csv)
    Xa, Ma_struct, ts_a = load_timeseries(attacked_csv)
    labels01, ts_l = _read_labels_csv(labels_csv, headers)
    if len(ts) != Xn.shape[0] or len(ts_a) != Xa.shape[0] or len(ts_l) != labels01.shape[0]:
        raise SystemExit('[error] Timestamp and data length mismatch across CSVs')
    if not (np.all(ts.astype(str).values == ts_a.astype(str).values) and np.all(ts.astype(str).values == ts_l.astype(str).values)):
        raise SystemExit('[error] Timestamps are not aligned across Normal/Attacked/Labels CSVs')

    Xn_std, Xa_std, mean, std_safe, Mn_struct_f, Ma_struct_f = _standardize_full(
        Xn, Mn_struct, Xa, Ma_struct, train_ratio=args.train_ratio
    )

    L = int(args.time_length)
    tail_idx = _select_attacked_tail(labels01, attack_timestamp=args.attack_timestamp, timestamps=ts, L=L)
    s = tail_idx - (L - 1)
    e = tail_idx + 1
    if s < 0:
        raise SystemExit('[error] Window underflows start; choose a later tail or reduce --time_length')

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

    # Build tail-only window
    X_T_raw = Xn[s:e].copy().astype(np.float32)
    X_T_raw[-1] = Xa[tail_idx].astype(np.float32)
    M_T_struct = (~np.isnan(X_T_raw)).astype(np.float32)
    X_T_std = np.where(M_T_struct > 0.5, (X_T_raw - mean) / std_safe, 0.0).astype(np.float32)
    labels_window = labels01[s:e]
    tail_attack_mask = (labels_window[-1] < 0.5) & (M_T_struct[-1] > 0.5)
    attacked_count = int(np.sum(tail_attack_mask))
    if attacked_count == 0:
        raise SystemExit('[error] Selected tail has no attacked features; pick a different timestamp')

    m_imp = M_T_struct.copy()
    m_imp[-1, tail_attack_mask] = 0.0
    x_in = np.where(m_imp > 0.5, X_T_std, 0.0).astype(np.float32)
    y_std = _impute_tail(model, x_in, m_imp, device=device)
    y = y_std * std_safe + mean

    x_true_tail = Xn[tail_idx].astype(np.float64)
    x_att_tail = Xa[tail_idx].astype(np.float64)
    y_tail = y[-1].astype(np.float64)
    err_att = x_att_tail[tail_attack_mask] - x_true_tail[tail_attack_mask]
    err_rep = y_tail[tail_attack_mask] - x_true_tail[tail_attack_mask]
    mse_b = float(np.mean(err_att ** 2))
    mae_b = float(np.mean(np.abs(err_att)))
    rmse_b = float(np.sqrt(mse_b))
    mse = float(np.mean(err_rep ** 2))
    mae = float(np.mean(np.abs(err_rep)))
    rmse = float(np.sqrt(mse))
    std_map = std_safe.astype(np.float64)
    std_tail = std_map[tail_attack_mask]
    std_tail = np.where(std_tail > 0.0, std_tail, 1.0)
    nerr_att = err_att / std_tail
    nerr_rep = err_rep / std_tail
    nrmse_b = float(np.sqrt(np.mean(nerr_att ** 2)))
    nmae_b = float(np.mean(np.abs(nerr_att)))
    nrmse = float(np.sqrt(np.mean(nerr_rep ** 2)))
    nmae = float(np.mean(np.abs(nerr_rep)))

    def rel_improve(new_val: float, base_val: float) -> float:
        if not math.isfinite(base_val) or base_val <= 1e-12:
            return float('nan')
        return 100.0 * (base_val - new_val) / base_val

    rim_mse = rel_improve(mse, mse_b)
    rim_rmse = rel_improve(rmse, rmse_b)

    print(f"[info] Window s={s}, e={e} (tail index {tail_idx}, timestamp {ts.iloc[tail_idx]})")
    print(f"[info] Attacked tail features={attacked_count}")
    print(
        "\nMetrics (lower is better):\n"
        "          MSE         MAE         RMSE        NRMSE       NMAE"
    )
    print(f"Baseline {mse_b:10.6f} {mae_b:10.6f} {rmse_b:10.6f} {nrmse_b:10.6f} {nmae_b:10.6f}")
    print(f"Repaired {mse:10.6f} {mae:10.6f} {rmse:10.6f} {nrmse:10.6f} {nmae:10.6f}")
    print(f"\nRelative improvement: MSE={rim_mse:.2f}%  RMSE={rim_rmse:.2f}%")


if __name__ == '__main__':
    main()
