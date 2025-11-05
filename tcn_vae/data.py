import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _load_csv(path: str) -> pd.DataFrame:
    # Normalize Windows-style backslashes to the current OS separator for portability
    norm_path = path.replace('\\', os.sep)
    if not os.path.exists(norm_path):
        raise FileNotFoundError(path)
    df = pd.read_csv(norm_path)
    if df.shape[1] < 2:
        raise ValueError('CSV must have at least a timestamp column and one feature')
    return df


def load_timeseries(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    df_x = _load_csv(csv_path)
    ts_col = df_x.columns[0]
    ts_parsed = pd.to_datetime(df_x[ts_col], errors='raise')
    sort_idx = ts_parsed.argsort(kind='mergesort')
    df_x = df_x.iloc[sort_idx].reset_index(drop=True)
    ts_sorted = ts_parsed.iloc[sort_idx].reset_index(drop=True)
    ts = ts_sorted.dt.strftime('%Y/%m/%d %H:%M')
    df_x[ts_col] = ts
    X = df_x.drop(columns=[ts_col]).to_numpy(dtype=np.float32)
    M_struct = (~np.isnan(X)).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X, M_struct, ts


# Removed unused legacy standardization helper to keep API minimal.


def _window_indices(T: int, window: int, stride: int) -> np.ndarray:
    if window <= 0 or stride <= 0:
        raise ValueError('window and stride must be positive')
    idx = np.arange(0, max(T - window + 1, 0), stride, dtype=np.int64)
    return idx


def _make_worker_init_fn(base_seed: Optional[int]):
    if base_seed is None:
        return None
    def _fn(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _fn


class NormalSlidingWindowDataset(Dataset):
    """Windows for normal-only training with FDIA injection upstream.

    Returns (x_normal_window, m_struct_window), both [L,H].
    """

    def __init__(self, x_normal: np.ndarray, m_struct: np.ndarray, window: int, stride: int = 1):
        if x_normal.shape != m_struct.shape:
            raise ValueError('x_normal and m_struct must share shape [T,H]')
        self.T, self.H = x_normal.shape
        self.window = int(window)
        self.stride = int(stride)
        self.starts = _window_indices(self.T, self.window, self.stride)
        self.xn = torch.from_numpy(x_normal)
        self.m = torch.from_numpy(m_struct)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s = self.starts[idx]
        e = s + self.window
        return self.xn[s:e], self.m[s:e]


class PairedSlidingWindowDataset(Dataset):
    """Paired windows for supervised reconstruction.

    Returns (x_attack_window, m_struct_window, y_normal_window), all [L,H].
    """

    def __init__(self, x_attack: np.ndarray, x_normal: np.ndarray, m_struct: np.ndarray, window: int, stride: int = 1):
        if not (x_attack.shape == x_normal.shape == m_struct.shape):
            raise ValueError('x_attack, x_normal, m_struct must share shape [T,H]')
        self.T, self.H = x_attack.shape
        self.window = int(window)
        self.stride = int(stride)
        self.starts = _window_indices(self.T, self.window, self.stride)
        self.xa = torch.from_numpy(x_attack)
        self.xn = torch.from_numpy(x_normal)
        self.m = torch.from_numpy(m_struct)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s = self.starts[idx]
        e = s + self.window
        return self.xa[s:e], self.m[s:e], self.xn[s:e]


@dataclass
class DataModule:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    slot_mean: np.ndarray
    slot_std: np.ndarray
    slot_kind: str
    clip_k: float


def times_to_hour_index(ts: pd.Series) -> np.ndarray:
    """Convert timestamps to hour-of-day indices [0..23]."""
    try:
        hours = pd.to_datetime(ts, errors='raise').dt.hour.to_numpy()
    except Exception:
        s = ts.astype(str)
        hours = s.str.slice(11, 13).astype(int).to_numpy()
    return hours.astype(np.int64)



def masked_robust_slot_stats(
    x: np.ndarray,
    m: np.ndarray,
    slots: np.ndarray,
    *,
    slot_count: int = 24,
    eps: float = 1e-6,
    std_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robust per-slot median and std via IQR on observed entries.

    Returns (slot_median[S,H], slot_std[S,H]). Falls back to global robust stats if a slot lacks data.
    """
    if x.shape != m.shape:
        raise ValueError('x and m must share shape [T,H]')
    T, H = x.shape
    if slots.shape[0] != T:
        raise ValueError('slots must have length T')
    X = x.astype(np.float64)
    M = m.astype(np.float64)
    X_obs = np.where(M > 0.5, X, np.nan)
    S = int(slot_count)
    med = np.zeros((S, H), dtype=np.float64)
    q25 = np.zeros_like(med)
    q75 = np.zeros_like(med)
    for s in range(S):
        idx = (slots == s)
        if not np.any(idx):
            med[s, :] = np.nan
            q25[s, :] = np.nan
            q75[s, :] = np.nan
            continue
        Xs = X_obs[idx]
        med[s, :] = np.nanmedian(Xs, axis=0)
        q25[s, :] = np.nanpercentile(Xs, 25.0, axis=0)
        q75[s, :] = np.nanpercentile(Xs, 75.0, axis=0)
    iqr = q75 - q25
    robust_std = (iqr / 1.349)
    # Global fallback
    g_med = np.nanmedian(X_obs, axis=0)
    g_q25 = np.nanpercentile(X_obs, 25.0, axis=0)
    g_q75 = np.nanpercentile(X_obs, 75.0, axis=0)
    g_std = (g_q75 - g_q25) / 1.349
    g_std = np.clip(g_std, std_floor, None)
    med = np.where(np.isnan(med), g_med.reshape(1, H), med)
    robust_std = np.where(np.isnan(robust_std), g_std.reshape(1, H), robust_std)
    robust_std = np.clip(robust_std, std_floor, None)
    return med.astype(np.float32), robust_std.astype(np.float32)


def apply_standardization_slotwise(
    x: np.ndarray,
    m: np.ndarray,
    slots: np.ndarray,
    slot_mean: np.ndarray,
    slot_std: np.ndarray,
    *,
    clip_k: float = 0.0,
) -> np.ndarray:
    """Apply per-slot robust z-score and gate by mask.

    Shapes: x/m [T,H], slots [T], slot_mean/std [S,H].
    """
    if x.shape != m.shape:
        raise ValueError('x and m must share shape [T,H]')
    idx = slots.astype(np.int64)
    mean_t = slot_mean[idx]
    std_t = slot_std[idx]
    x_scaled = (x - mean_t) / std_t
    clip_val = float(clip_k)
    if clip_val > 0.0:
        x_scaled = np.clip(x_scaled, -clip_val, clip_val)
    x_proc = np.where(m > 0.5, x_scaled, 0.0).astype(np.float32)
    return x_proc


def _load_header(csv_path: str) -> List[str]:
    df = _load_csv(csv_path)
    cols = list(df.columns)
    if not cols:
        raise ValueError('CSV has no columns')
    return cols[1:]


def load_paired_timeseries(normal_csv: str, attacked_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Load aligned normal/attacked CSVs and validate.

    Returns (X_attack, X_normal, M_struct, timestamps)
    """
    h_n = _load_header(normal_csv)
    h_a = _load_header(attacked_csv)
    if h_n != h_a:
        raise SystemExit('[error] Normal and attacked CSV headers do not match')
    Xa, Ma, tsa = load_timeseries(attacked_csv)
    Xn, Mn, tsn = load_timeseries(normal_csv)
    if Xa.shape != Xn.shape:
        raise SystemExit('[error] Shape mismatch between normal and attacked arrays')
    if not tsn.equals(tsa):
        raise SystemExit('[error] Timestamp alignment mismatch between normal and attacked')
    if not np.array_equal(Ma, Mn):
        raise SystemExit('[error] Observability masks differ between normal and attacked')
    return Xa, Xn, Mn.astype(np.float32), tsn


def create_normal_loaders(
    *,
    train_normal_csv: str,
    time_length: int = 96,
    stride: int = 48,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = 1337,
    num_workers: int = 0,
    # Standardization controls
    clip_k: float = 0.0,
    std_floor: float = 1e-3,
) -> DataModule:
    Xn, M_struct, ts = load_timeseries(train_normal_csv)
    T, H = Xn.shape
    train_T = int(T * train_ratio)
    val_T = int(T * val_ratio)
    s_train = slice(0, train_T)
    s_val = slice(train_T, train_T + val_T)
    s_test = slice(train_T + val_T, T)

    Xn_tr, M_tr = Xn[s_train], M_struct[s_train]
    Xn_va, M_va = Xn[s_val], M_struct[s_val]
    Xn_te, M_te = Xn[s_test], M_struct[s_test]

    # Slot-wise robust standardization computed from training normal only (hour-of-day)
    hours = times_to_hour_index(ts)
    hours_tr = hours[s_train]
    hours_va = hours[s_val]
    hours_te = hours[s_test]
    slot_mean, slot_std = masked_robust_slot_stats(
        Xn_tr, M_tr, hours_tr, slot_count=24, std_floor=float(std_floor)
    )
    CLIP_K = float(clip_k)
    Xn_tr_s = apply_standardization_slotwise(
        Xn_tr, M_tr, hours_tr, slot_mean, slot_std, clip_k=CLIP_K
    )
    Xn_va_s = apply_standardization_slotwise(
        Xn_va, M_va, hours_va, slot_mean, slot_std, clip_k=CLIP_K
    )
    Xn_te_s = apply_standardization_slotwise(
        Xn_te, M_te, hours_te, slot_mean, slot_std, clip_k=CLIP_K
    )

    ds_train = NormalSlidingWindowDataset(Xn_tr_s, M_tr.astype(np.float32), time_length, stride)
    ds_val = NormalSlidingWindowDataset(Xn_va_s, M_va.astype(np.float32), time_length, stride)
    ds_test = NormalSlidingWindowDataset(Xn_te_s, M_te.astype(np.float32), time_length, stride)

    wif_train = _make_worker_init_fn(seed)
    wif_val = _make_worker_init_fn(None if seed is None else seed + 1)
    wif_test = _make_worker_init_fn(None if seed is None else seed + 2)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=wif_train)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=wif_val)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=wif_test)
    return DataModule(
        train=dl_train,
        val=dl_val,
        test=dl_test,
        slot_mean=slot_mean,
        slot_std=slot_std,
        slot_kind='hour',
        clip_k=CLIP_K,
    )



__all__ = [
    'load_timeseries',
    'load_paired_timeseries',
    'PairedSlidingWindowDataset',
    'NormalSlidingWindowDataset',
    'create_normal_loaders',
    'DataModule',
    'times_to_hour_index',
    'apply_standardization_slotwise',
    'masked_robust_slot_stats',
]
