"""Tail-Only (context Normal, tail Attacked) locate+repair without external labels.

Pipeline (best-only implementation):
- Calibrate per-feature thresholds on Normal tail NLL at (1-alpha) quantile
  (with global-quantile fallback for sparse features).
- For Tail-Only window [s,e), compute tail NLL at e-1, threshold to predict anomalies,
  hard-mask predicted anomalies at the tail, and reconstruct via Online GP-VAE.
- Monte Carlo averaging with 16 samples (fixed) is used during imputation to improve stability.
  Evaluate repair quality against the clean baseline over observed tail features.

Sliding behavior:
- Consecutive-attack simulation: at each sliding step, only the immediately previous
  repaired tail row is fed back as history for the next window; all earlier history
  rows remain from the Normal (clean) sequence.
"""

from __future__ import annotations
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gru_vae.data import load_timeseries, SlidingWindowDataset, masked_mean_std, resolve_training_csv
from gru_vae.model import OnlineGPVAE
from gru_vae.noise import apply_noise
from gru_vae.utils import parse_sizes, resolve_device
from gru_vae.config import load_config

def _ensure_outdir_case(case: str, kind: str) -> str:
    base = os.path.join('output', case, kind)
    os.makedirs(base, exist_ok=True)
    return base
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
) -> Tuple[np.ndarray, np.ndarray]:
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
        logv_tail = logv_last.squeeze(0).cpu().numpy().astype(np.float64)
        return out, logv_tail


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
    # Store empirical samples per feature for BH-FDR p-value computation
    samples: Dict[str, np.ndarray] = {}
    thr: Dict[str, float] = {}
    for h in range(H):
        name = feature_names[h]
        vals = scores[:, h][mask_valid[:, h]]
        if vals.size > 0:
            arr = np.sort(vals.astype(np.float64))
            samples[name] = arr
            # Equivalent (1-alpha) quantile per feature (for reporting only)
            thr[name] = float(np.quantile(arr, 1.0 - alpha))
    # Fallback: global threshold if some features lack samples
    if len(samples) < H:
        all_vals = scores[mask_valid]
        if np.any(np.isfinite(all_vals)):
            tau = float(np.quantile(all_vals, 1.0 - alpha))
        else:
            tau = float('inf')
        for h in range(H):
            name = feature_names[h]
            if name not in thr:
                thr[name] = tau
    return {
        'type': 'empirical',
        'alpha': float(alpha),
        'samples': samples,
        'thresholds': thr,
        'min_count': int(min_count_per_feature),
    }


def _bh_fdr_keep_pred(scores_tail: np.ndarray, m_tail: np.ndarray, meta: Dict[str, object], feature_names: List[str]) -> np.ndarray:
    """Compute keep_pred via BH-FDR using empirical tail-score distributions.

    Returns keep_pred[H] in {0.0,1.0} (np.nan for unobserved), where 0.0 indicates drop/anomaly.
    """
    alpha = float(meta.get('alpha', 0.05))
    samples: Dict[str, np.ndarray] = meta.get('samples', {})  # type: ignore[assignment]
    H = len(feature_names)
    keep = np.full((H,), np.nan, dtype=np.float64)
    obs_idx = np.where(m_tail > 0.5)[0]
    if obs_idx.size == 0:
        return keep
    # Compute empirical upper-tail p-values
    pvals = []
    idx_list = []
    for h in obs_idx:
        name = feature_names[h]
        s_h = float(scores_tail[h])
        arr = samples.get(name, None)
        if arr is None or arr.size == 0:
            # fallback: p-value unknown -> use 1.0 (most conservative keep)
            p = 1.0
        else:
            # p = 1 - F_hat(s)
            # equivalent to proportion of samples > s_h
            # use searchsorted for sorted arr
            k = int(np.searchsorted(arr, s_h, side='right'))
            F = k / max(len(arr), 1)
            p = max(0.0, min(1.0, 1.0 - F))
        pvals.append(p)
        idx_list.append(h)
    pvals = np.array(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return keep
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresh = alpha * (np.arange(1, m + 1) / float(m))
    # Largest k with p_(k) <= thresh_k
    k_max = np.where(sorted_p <= thresh)[0]
    reject = np.zeros_like(sorted_p, dtype=bool)
    if k_max.size > 0:
        k = int(k_max.max())
        reject[:k + 1] = True
    # Assign keep_pred: reject -> anomaly (drop -> 0.0), else keep 1.0
    keep_obs = (~reject).astype(np.float64)
    # Map back to feature indices
    for rank, pos in enumerate(order):
        h = idx_list[pos]
        keep[h] = keep_obs[rank]
    return keep


MC_SAMPLES = 16


def _reconstruct_tail(
    model: OnlineGPVAE,
    x_w: np.ndarray,
    m_w: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    """Reconstruct tail-window with optional Monte Carlo averaging on latent samples.

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
            ys = model.reconstruct_online(xb, mb, use_mean=False)
            ys_np = ys.squeeze(0).cpu().numpy().astype(np.float64)
            if acc is None:
                acc = ys_np
            else:
                acc += ys_np
        return acc / float(S)


def _select_tail_timestamp_only(*, attack_timestamp: str, timestamps: pd.Series, L: int, quiet: bool = False) -> int:
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
        if not quiet:
            print('[warn] Provided attack_timestamp exists but violates window length; searching forward')
    else:
        if not quiet:
            print('[warn] Provided attack_timestamp not found; searching forward by time')
    # First feasible at/after target_dt
    feasible = np.where(ts_parsed >= target_dt)[0]
    feasible = feasible[feasible >= min_t]
    if feasible.size == 0:
        raise SystemExit('[error] No feasible rows at/after the provided time for the requested window length')
    fallback = int(feasible[0])
    if not quiet:
        print(f'[warn] Fallback tail selected at index {fallback} (min_t={min_t})')
    return fallback


def main() -> None:
    # All parameters now come from config.yaml
    args = load_config()

    if args.time_length <= 0:
        raise SystemExit('[error] --time_length must be positive')
    if args.sliding_steps <= 0:
        raise SystemExit('[error] --sliding_steps must be positive')
    if not args.attack_timestamp or not str(args.attack_timestamp).strip():
        raise SystemExit('[error] attack_timestamp must be set in config for tail-only inference')

    device = resolve_device(args.device)

    # Resolve CSVs
    case_dir = os.path.join(args.data_dir, args.case)
    # Normal: prefer 2025-09 clean (non-fdia), search in case dir and 'infer/'
    normal_csv = args.normal_csv
    if not normal_csv:
        normal_guess = ''
        search_dirs = [case_dir, os.path.join(case_dir, 'infer')]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                low = fname.lower()
                if (low.endswith('.csv')) and ('2025-09' in low) and ('fdia' not in low):
                    # prefer acopf_2025-09_noisy over other variants
                    path = os.path.join(d, fname)
                    if 'acopf_2025-09_noisy' in low:
                        normal_guess = path
                        break
                    if not normal_guess:
                        normal_guess = path
            if normal_guess:
                break
        normal_csv = normal_guess
    if not normal_csv or not os.path.exists(normal_csv):
        raise SystemExit('[error] Could not infer normal_csv (clean 2025/09). Provide --normal_csv explicitly.')

    attacked_csv = args.attacked_csv
    if not attacked_csv:
        attacked_guess = ''
        search_dirs = [case_dir, os.path.join(case_dir, 'infer')]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                lower = fname.lower()
                if lower.endswith('.csv') and 'fdia' in lower and '2025-09' in lower:
                    attacked_guess = os.path.join(d, fname)
                    break
            if attacked_guess:
                break
        attacked_csv = attacked_guess
    if not attacked_csv:
        raise SystemExit('[error] Could not infer attacked_csv (fdia 2025/09); provide it explicitly')

    headers = _check_headers_match(normal_csv, attacked_csv)
    feat_names = headers[1:]
    ts_col = headers[0]

    # Load sequences (full normal/attacked for the target month)
    Xn, Mn_struct, ts = load_timeseries(normal_csv)
    Xa, Ma_struct, ts_a = load_timeseries(attacked_csv)
    if len(ts) != Xn.shape[0] or len(ts_a) != Xa.shape[0]:
        raise SystemExit('[error] Timestamp and data length mismatch across CSVs')
    if not np.all(ts.astype(str).values == ts_a.astype(str).values):
        raise SystemExit('[error] Timestamps are not perfectly aligned between Normal and Attacked CSVs')

    # Standardize with consistent mean/std
    # If stats_source == 'training', compute mean/std from auto-detected clean 2025/07-08 training set
    # to match training-time standardization; otherwise compute from the current normal month.
    if args.stats_source == 'training':
        train_csv = resolve_training_csv(args.data_dir, args.case)
        Xtr_full, Mtr_full, _ = load_timeseries(train_csv)
        Ttr = Xtr_full.shape[0]
        train_T = int(Ttr * float(args.train_ratio))
        mean, std = masked_mean_std(Xtr_full[:train_T], Mtr_full[:train_T])
        std_safe = np.where(std > 0, std, 1.0).astype(np.float32)
        Xn_std = np.where(Mn_struct > 0.5, (Xn - mean) / std_safe, 0.0).astype(np.float32)
        Xa_std = np.where(Ma_struct > 0.5, (Xa - mean) / std_safe, 0.0).astype(np.float32)
        Mn_struct_f = Mn_struct.astype(np.float32)
        Ma_struct_f = Ma_struct.astype(np.float32)
    else:
        Xn_std, Xa_std, mean, std_safe, Mn_struct_f, Ma_struct_f = _standardize_full(
            Xn, Mn_struct, Xa, Ma_struct, train_ratio=args.train_ratio
        )

    # Select tail by timestamp-only with fallback to first feasible
    L = int(args.time_length)
    t = _select_tail_timestamp_only(attack_timestamp=args.attack_timestamp, timestamps=ts, L=L, quiet=args.quiet)
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
        if not args.quiet:
            print(f'[ok] Loaded checkpoint: {args.ckpt}')
    else:
        if args.ckpt and not args.quiet:
            print(f'[warn] Checkpoint not found: {args.ckpt}. Continue with random weights.')

    # Calibrate thresholds on Normal (global per-feature)
    thresholds_meta = _calibrate_thresholds_normal(
        model, Xn_std, Mn_struct_f, L,
        batch_size=args.batch_size, device=device,
        alpha=args.alpha, min_count_per_feature=args.min_count_per_feature,
        feature_names=feat_names,
    )
    # Always dump thresholds to output/<case>/thresholds
    thr_map: Dict[str, float] = thresholds_meta['thresholds']  # type: ignore[index]
    df_thr = pd.DataFrame({
        'feature': feat_names,
        'threshold': [float(thr_map[name]) for name in feat_names],
    })
    thr_outdir = _ensure_outdir_case(args.case, 'thresholds')
    thr_name = f"thresholds_L{L}_alpha{str(args.alpha).replace('.', '_')}.csv"
    thr_path = os.path.join(thr_outdir, thr_name)
    df_thr.to_csv(thr_path, index=False, encoding='utf-8')
    if not args.quiet:
        print(f"[ok] Saved thresholds CSV: {thr_path} (features={len(df_thr)})")

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

    if not args.quiet:
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

        scores_tail, logv_tail = _compute_tail_scores_window(model, X_T_std, M_T_struct, device=device)
        m_tail = M_T_struct[-1]
        pred_tail = _bh_fdr_keep_pred(scores_tail, m_tail, thresholds_meta, feat_names)
        m_imp = M_T_struct.copy()
        m_tail_obs = m_tail > 0.5
        if not np.any(m_tail_obs):
            raise SystemExit(f'[error] Sliding step {step_offset} (tail index {tail_idx}) has no observed tail features')
        is_anom = (pred_tail == 0.0) & m_tail_obs
        m_imp[-1, is_anom] = 0.0
        # Diagnostics: Top-K printing and per-step tail scores dump
        if args.print_topk and args.topk > 0:
            obs_idx_arr = np.where(m_tail_obs)[0]
            if obs_idx_arr.size > 0:
                thr_map: Dict[str, float] = thresholds_meta['thresholds']  # type: ignore[index]
                scores_obs = scores_tail[obs_idx_arr]
                names_obs = [feat_names[i] for i in obs_idx_arr]
                order = np.argsort(-scores_obs)  # descending
                k = int(min(args.topk, order.size))
                print(f"[debug] Top-{k} tail scores at step {step_offset} (tail idx {tail_idx}):")
                for rank in range(k):
                    j = int(order[rank])
                    name = names_obs[j]
                    sc = float(scores_obs[j])
                    tau = float(thr_map[name])
                    keep_pred = int(sc <= tau)
                    print(f"  #{rank+1:02d} {name:>20s}  score={sc:.6f}  thr={tau:.6f}  keep_pred={keep_pred}")
        if args.tail_scores_wide:
            thr_map: Dict[str, float] = thresholds_meta['thresholds']  # type: ignore[index]
            # Always emit wide-format CSV only: one row per timestamp, columns match input features
            kind = str(args.tail_scores_value) if hasattr(args, 'tail_scores_value') else 'score'
            if kind == 'score':
                vals = scores_tail.astype(np.float64)
            elif kind == 'threshold':
                vals = np.asarray([float(thr_map[name]) for name in feat_names], dtype=np.float64)
            elif kind == 'keep_pred':
                vals = (scores_tail <= np.asarray([thr_map[name] for name in feat_names], dtype=np.float64)).astype(np.float64)
            else:  # 'is_anom'
                vals = is_anom.astype(np.float64)
            header = [ts_col] + list(feat_names)
            row_vals = [timestamp_str] + [float(x) for x in vals.tolist()]
            wide_outdir = _ensure_outdir_case(args.case, 'tail_scores')
            out_wide = os.path.join(wide_outdir, f"tail_scores_wide_{kind}.csv")
            need_header = not os.path.exists(out_wide)
            with open(out_wide, 'a', encoding='utf-8') as fw:
                if need_header:
                    fw.write(','.join(header) + '\n')
                fw.write(','.join(str(v) for v in row_vals) + '\n')
            if not args.quiet:
                print(f"[ok] Appended wide tail scores ({kind}) to: {out_wide}")
        # Build encoder input by injecting Gaussian noise only on predicted anomalous tail positions (mask=0).
        # This matches training semantics of corrupted masked inputs; no zero-filling here.
        mask_keep_noise = np.ones_like(M_T_struct, dtype=np.float32)
        mask_keep_noise[-1, is_anom] = 0.0
        
        rng = None
        if int(getattr(args, 'noise_seed', -1)) >= 0:
            try:
                rng = np.random.default_rng(int(args.noise_seed) + int(tail_idx))
            except Exception:
                rng = None
        # Determine sigma for anomalies from decoder log-variance (adaptive-only)
        sigma_tail = np.exp(0.5 * logv_tail).astype(np.float64)
        sigma_arr = np.zeros_like(X_T_std, dtype=np.float64)
        sigma_arr[-1, :] = sigma_tail
        sigma_for_apply: float | np.ndarray = sigma_arr
        x_in = apply_noise(
            X_T_std.astype(np.float32),
            mask_keep_noise.astype(np.float32),
            kind='gaussian',
            rng=rng,
            sigma=sigma_for_apply,
        )
        y_std = _reconstruct_tail(model, x_in, m_imp, device=device)
        y = y_std * std_safe + mean
        current_stream[tail_idx] = y[-1].astype(np.float32)
        repaired_tail_rows.append((tail_idx, timestamp_str, y[-1].astype(np.float64)))
        repaired_mask[tail_idx] = True

        # Also append repaired values (wide format): one row per timestamp, columns match input features
        if args.tail_scores_wide:
            rep_wide_dir = _ensure_outdir_case(args.case, 'repaired')
            out_repaired = os.path.join(rep_wide_dir, 'tail_repaired_wide.csv')
            header_rep = [ts_col] + list(feat_names)
            need_header_rep = not os.path.exists(out_repaired)
            y_tail_vec = y[-1].astype(np.float64)
            row_rep = [timestamp_str] + [float(v) for v in y_tail_vec.tolist()]
            with open(out_repaired, 'a', encoding='utf-8') as fw:
                if need_header_rep:
                    fw.write(','.join(header_rep) + '\n')
                fw.write(','.join(str(v) for v in row_rep) + '\n')

        x_true_tail = Xn[tail_idx].astype(np.float64)
        x_att_tail = Xa[tail_idx].astype(np.float64)
        # Also append attacked/true values as wide CSVs for side-by-side comparison
        if args.tail_scores_wide:
            rep_wide_dir = _ensure_outdir_case(args.case, 'repaired')
            header_w = [ts_col] + list(feat_names)
            # attacked
            out_att = os.path.join(rep_wide_dir, 'tail_attacked_wide.csv')
            if not os.path.exists(out_att):
                with open(out_att, 'w', encoding='utf-8') as fw:
                    fw.write(','.join(header_w) + '\n')
            row_att = [timestamp_str] + [float(v) for v in x_att_tail.tolist()]
            with open(out_att, 'a', encoding='utf-8') as fw:
                fw.write(','.join(str(v) for v in row_att) + '\n')
            # true (clean baseline)
            out_true = os.path.join(rep_wide_dir, 'tail_true_wide.csv')
            if not os.path.exists(out_true):
                with open(out_true, 'w', encoding='utf-8') as fw:
                    fw.write(','.join(header_w) + '\n')
            row_true = [timestamp_str] + [float(v) for v in x_true_tail.tolist()]
            with open(out_true, 'a', encoding='utf-8') as fw:
                fw.write(','.join(str(v) for v in row_true) + '\n')
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

        # Drop-only metrics: evaluate reconstruction only over predicted dropped features
        if drop_count > 0:
            mask_idx_drop = is_anom
            err_rep_d = y_tail[mask_idx_drop] - x_true_tail[mask_idx_drop]
            mse_drop = float(np.mean(err_rep_d ** 2))
            rmse_drop = float(np.sqrt(mse_drop))
            std_tail_d = std_map[mask_idx_drop]
            std_tail_d = np.where(std_tail_d > 0.0, std_tail_d, 1.0)
            nrmse_drop = float(np.sqrt(np.mean((err_rep_d / std_tail_d) ** 2)))
        else:
            mse_drop = float('nan')
            rmse_drop = float('nan')
            nrmse_drop = float('nan')

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
            'mse_drop': mse_drop,
            'rmse_drop': rmse_drop,
            'nrmse_drop': nrmse_drop,
        })

        if not args.quiet:
            print(
                "[info] Step "
                f"{step_offset}: window s={s_step}, e={e_step} (tail index {tail_idx}, timestamp {timestamp_str}), "
                f"observed_tail_count={obs_count}, dropped={drop_count} ({drop_rate:.2f}%)"
            )

    # Always save only repaired tail rows (one row per sliding step)
    base = os.path.splitext(os.path.basename(attacked_csv))[0]
    suffix = ("_" + args.out_suffix.strip()) if args.out_suffix and args.out_suffix.strip() else ""
    rep_outdir = _ensure_outdir_case(args.case, 'repaired')
    out_tail_csv = os.path.join(rep_outdir, f"{base}_repaired_tail_rows_L{L}_steps{sliding_steps}{suffix}.csv")
    if repaired_tail_rows:
        stamps = [it[1] for it in repaired_tail_rows]
        vals = np.stack([it[2] for it in repaired_tail_rows], axis=0)
        df_tail = pd.DataFrame(vals, columns=feat_names)
        df_tail.insert(0, ts_col, stamps)
        df_tail.to_csv(out_tail_csv, index=False, encoding='utf-8')
        if not args.quiet:
            print(f"[ok] Saved repaired tail rows CSV: {out_tail_csv} (rows={len(stamps)})")
    else:
        if not args.quiet:
            print('[warn] No repaired tail rows to save (empty sliding result).')

    print('\nSummary (lower is better):')
    header = (
        "Step  TailIdx  Timestamp        Obs    Dropped  DropRate(%)  "
        "MSE_att     MSE_rep     RMSE_att    RMSE_rep    NRMSE_att   NRMSE_rep   "
        "Rel.Improv_MSE(%)  Rel.Improv_RMSE(%)  MSE_rep@Drop  RMSE_rep@Drop  NRMSE_rep@Drop"
    )
    print(header)
    for res in step_results:
        rim_mse = res['rim_mse']
        rim_rmse = res['rim_rmse']
        mse_drop = res.get('mse_drop', float('nan'))
        rmse_drop = res.get('rmse_drop', float('nan'))
        nrmse_drop = res.get('nrmse_drop', float('nan'))
        mse_drop_s = f"{mse_drop:.6f}" if np.isfinite(mse_drop) else "   nan  "
        rmse_drop_s = f"{rmse_drop:.6f}" if np.isfinite(rmse_drop) else "   nan  "
        nrmse_drop_s = f"{nrmse_drop:.6f}" if np.isfinite(nrmse_drop) else "   nan  "
        print(
            f"{res['step']:4d}  {res['tail_idx']:7d}  {res['timestamp']:16s}  "
            f"{res['obs_count']:6d}  {res['drop_count']:7d}  {res['drop_rate']:11.2f}  "
            f"{res['mse_b']:10.6f} {res['mse']:10.6f} {res['rmse_b']:11.6f} {res['rmse']:11.6f} "
            f"{res['nrmse_b']:10.6f} {res['nrmse']:10.6f} {rim_mse:18.2f} {rim_rmse:18.2f}  "
            f"{mse_drop_s:>12s} {rmse_drop_s:>14s} {nrmse_drop_s:>14s}"
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
        # Drop-only aggregates
        mse_drop_vals = [res.get('mse_drop', float('nan')) for res in step_results]
        rmse_drop_vals = [res.get('rmse_drop', float('nan')) for res in step_results]
        nrmse_drop_vals = [res.get('nrmse_drop', float('nan')) for res in step_results]
        arr_m = np.asarray(mse_drop_vals, dtype=np.float64)
        arr_r = np.asarray(rmse_drop_vals, dtype=np.float64)
        arr_n = np.asarray(nrmse_drop_vals, dtype=np.float64)
        m_m = np.isfinite(arr_m)
        m_r = np.isfinite(arr_r)
        m_n = np.isfinite(arr_n)
        if args.print_drop_metrics and (np.any(m_m) or np.any(m_r) or np.any(m_n)):
            mse_drop_mean = float(np.mean(arr_m[m_m])) if np.any(m_m) else float('nan')
            rmse_drop_mean = float(np.mean(arr_r[m_r])) if np.any(m_r) else float('nan')
            nrmse_drop_mean = float(np.mean(arr_n[m_n])) if np.any(m_n) else float('nan')
            print(
                f"Drop-Only Aggregate: MSE_rep@Drop={mse_drop_mean:.6f}, "
                f"RMSE_rep@Drop={rmse_drop_mean:.6f}, NRMSE_rep@Drop={nrmse_drop_mean:.6f}"
            )


if __name__ == '__main__':
    main()

