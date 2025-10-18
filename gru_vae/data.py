import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
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


def _apply_standardization(x: np.ndarray, m: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_scaled = (x - mean) / std
    x_proc = np.where(m > 0.5, x_scaled, 0.0)
    return x_proc.astype(np.float32)


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


class SlidingWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, m: np.ndarray, window: int, stride: int = 1):
        if x.shape != m.shape:
            raise ValueError('x and m must have identical shapes [T,H]')
        self.T, self.H = x.shape
        self.window = int(window)
        self.stride = int(stride)
        self.starts = _window_indices(self.T, self.window, self.stride)
        self.x = torch.from_numpy(x)
        self.m = torch.from_numpy(m)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s = self.starts[idx]
        e = s + self.window
        xw = self.x[s:e]
        mw = self.m[s:e]
        return xw, mw


class SelfSupervisedMaskingDataset(SlidingWindowDataset):
    def __init__(self, x: np.ndarray, m_struct: np.ndarray, window: int, stride: int = 1,
                 *, mask_rate: float = 0.2, seed: Optional[int] = None,
                 mask_mode: str = 'iid',
                 # block mode params
                 block_t_min: int = 2, block_t_max: int = 8,
                 block_f_min: int = 4, block_f_max: int = 32,
                 block_max_blocks: int = 4,
                 # corr mode params (odd kernel recommended)
                 corr_t: int = 7, corr_f: int = 15) -> None:
        super().__init__(x, m_struct, window, stride)
        if not (0.0 <= float(mask_rate) < 1.0):
            raise ValueError('mask_rate must satisfy 0 <= rate < 1')
        self.mask_rate = float(mask_rate)
        self._base_seed = int(seed) if seed is not None else None
        self._gen: Optional[torch.Generator] = None
        self.mask_mode = str(mask_mode).lower()
        if self.mask_mode not in ('iid', 'block', 'corr'):
            raise ValueError("mask_mode must be one of: 'iid', 'block', 'corr'")
        # block params
        self.block_t_min = int(block_t_min)
        self.block_t_max = int(block_t_max)
        self.block_f_min = int(block_f_min)
        self.block_f_max = int(block_f_max)
        self.block_max_blocks = int(block_max_blocks)
        # corr params
        self.corr_t = int(corr_t)
        self.corr_f = int(corr_f)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xw, m_struct_w = super().__getitem__(idx)
        if self.mask_rate == 0.0:
            x_masked = torch.where(m_struct_w > 0.5, xw, torch.zeros_like(xw))
            return x_masked, m_struct_w, xw
        if self._gen is None and self._base_seed is not None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self._base_seed + worker_id)
            self._gen = gen
        mode = self.mask_mode
        if mode == 'iid':
            if self._gen is not None:
                rnd = torch.rand(m_struct_w.shape, generator=self._gen, device=m_struct_w.device, dtype=m_struct_w.dtype)
            else:
                rnd = torch.rand_like(m_struct_w)
            keep = (rnd >= self.mask_rate).to(m_struct_w.dtype)
            m_sampled = m_struct_w * keep
        elif mode == 'block':
            L, H = m_struct_w.shape
            # approximate number of blocks to hit target rate
            avg_area = max((self.block_t_min + self.block_t_max) * 0.5, 1.0) * max((self.block_f_min + self.block_f_max) * 0.5, 1.0)
            target_drop = int(self.mask_rate * L * H)
            n_blocks = max(int(round(target_drop / max(avg_area, 1.0))), 1) if target_drop > 0 else 0
            if self.block_max_blocks > 0:
                n_blocks = min(n_blocks, self.block_max_blocks)
            drop = torch.zeros_like(m_struct_w)
            gen = self._gen
            for _ in range(n_blocks):
                # sizes
                if gen is not None:
                    rt = int(torch.randint(low=self.block_t_min, high=max(self.block_t_max, self.block_t_min + 1), size=(1,), generator=gen).item())
                    rf = int(torch.randint(low=self.block_f_min, high=max(self.block_f_max, self.block_f_min + 1), size=(1,), generator=gen).item())
                else:
                    rt = int(np.random.randint(self.block_t_min, max(self.block_t_max, self.block_t_min + 1)))
                    rf = int(np.random.randint(self.block_f_min, max(self.block_f_max, self.block_f_min + 1)))
                rt = max(1, min(rt, L))
                rf = max(1, min(rf, H))
                # positions
                if gen is not None:
                    t0 = int(torch.randint(low=0, high=max(L - rt + 1, 1), size=(1,), generator=gen).item())
                    f0 = int(torch.randint(low=0, high=max(H - rf + 1, 1), size=(1,), generator=gen).item())
                else:
                    t0 = int(np.random.randint(0, max(L - rt + 1, 1)))
                    f0 = int(np.random.randint(0, max(H - rf + 1, 1)))
                drop[t0:t0+rt, f0:f0+rf] = 1.0
            keep = (1.0 - drop).to(m_struct_w.dtype)
            m_sampled = m_struct_w * keep
        else:  # corr
            L, H = m_struct_w.shape
            if self._gen is not None:
                u = torch.rand((1, 1, L, H), generator=self._gen, device=m_struct_w.device, dtype=m_struct_w.dtype)
            else:
                u = torch.rand((1, 1, L, H), device=m_struct_w.device, dtype=m_struct_w.dtype)
            kt = max(int(self.corr_t), 1)
            kf = max(int(self.corr_f), 1)
            if kt % 2 == 0:
                kt += 1
            if kf % 2 == 0:
                kf += 1
            pad_t, pad_f = (kt - 1) // 2, (kf - 1) // 2
            u_s = F.avg_pool2d(F.pad(u, (pad_f, pad_f, pad_t, pad_t), mode='replicate'), kernel_size=(kt, kf), stride=1)
            u_s = u_s[0, 0]
            struct = (m_struct_w > 0.5)
            vals = u_s[struct]
            if vals.numel() == 0:
                keep = torch.ones_like(m_struct_w)
            else:
                p_keep = float(max(0.0, min(1.0, 1.0 - self.mask_rate)))
                tau = torch.quantile(vals, q=1.0 - torch.as_tensor(p_keep, dtype=vals.dtype, device=vals.device))
                keep = (u_s >= tau).to(dtype=m_struct_w.dtype)
            m_sampled = m_struct_w * keep
        x_masked = torch.where(m_sampled > 0.5, xw, torch.zeros_like(xw))
        return x_masked, m_sampled, xw


@dataclass
class DataModule:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    mean: np.ndarray
    std: np.ndarray


def masked_mean_std(x: np.ndarray, m: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    count = np.maximum(m.sum(axis=0), 1.0)
    mean = (x * m).sum(axis=0) / count
    var = (((x - mean) * m) ** 2).sum(axis=0) / count
    std = np.sqrt(np.maximum(var, eps))
    return mean.astype(np.float32), std.astype(np.float32)


def create_normal_loaders(
    data_dir: str,
    case: str = 'case14',
    *,
    time_length: int = 96,
    stride: int = 48,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    mask_rate: float = 0.2,
    mask_mode: str = 'iid',
    block_t_min: int = 2,
    block_t_max: int = 8,
    block_f_min: int = 4,
    block_f_max: int = 32,
    block_max_blocks: int = 4,
    corr_t: int = 7,
    corr_f: int = 15,
    seed: Optional[int] = 1337,
    num_workers: int = 0,
) -> DataModule:
    csv_path = os.path.join(data_dir, case, f'{case}_acopf_all_rows_noisy.csv')
    X, M_struct, _ = load_timeseries(csv_path)
    T, H = X.shape
    train_T = int(T * train_ratio)
    val_T = int(T * val_ratio)
    s_train = slice(0, train_T)
    s_val = slice(train_T, train_T + val_T)
    s_test = slice(train_T + val_T, T)
    X_train, M_train = X[s_train], M_struct[s_train]
    X_val, M_val = X[s_val], M_struct[s_val]
    X_test, M_test = X[s_test], M_struct[s_test]
    mean, std = masked_mean_std(X_train, M_train)
    def _apply(x, m):
        return _apply_standardization(x, m, mean, std)
    Xtr, Xva, Xte = _apply(X_train, M_train), _apply(X_val, M_val), _apply(X_test, M_test)
    ds_train = SelfSupervisedMaskingDataset(
        Xtr, M_train.astype(np.float32), time_length, stride,
        mask_rate=mask_rate, seed=seed,
        mask_mode=mask_mode,
        block_t_min=block_t_min, block_t_max=block_t_max,
        block_f_min=block_f_min, block_f_max=block_f_max,
        block_max_blocks=block_max_blocks,
        corr_t=corr_t, corr_f=corr_f,
    )
    ds_val = SelfSupervisedMaskingDataset(
        Xva, M_val.astype(np.float32), time_length, stride,
        mask_rate=mask_rate, seed=None if seed is None else seed + 1,
        mask_mode=mask_mode,
        block_t_min=block_t_min, block_t_max=block_t_max,
        block_f_min=block_f_min, block_f_max=block_f_max,
        block_max_blocks=block_max_blocks,
        corr_t=corr_t, corr_f=corr_f,
    )
    ds_test = SelfSupervisedMaskingDataset(
        Xte, M_test.astype(np.float32), time_length, stride,
        mask_rate=mask_rate, seed=None if seed is None else seed + 2,
        mask_mode=mask_mode,
        block_t_min=block_t_min, block_t_max=block_t_max,
        block_f_min=block_f_min, block_f_max=block_f_max,
        block_max_blocks=block_max_blocks,
        corr_t=corr_t, corr_f=corr_f,
    )
    wif_train = _make_worker_init_fn(seed)
    wif_val = _make_worker_init_fn(None if seed is None else seed + 1)
    wif_test = _make_worker_init_fn(None if seed is None else seed + 2)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=wif_train)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=wif_val)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=wif_test)
    return DataModule(train=dl_train, val=dl_val, test=dl_test, mean=mean, std=std)


__all__ = [
    'load_timeseries',
    'SlidingWindowDataset',
    'SelfSupervisedMaskingDataset',
    'create_normal_loaders',
    'DataModule',
]
