import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
                 corr_t: int = 7, corr_f: int = 15,
                 # input noise corruption (always applied on selected masked positions)
                 noise_kind: str = 'gaussian',
                 noise_sigma: float = 1.0,
                 noise_bias_min: float = -1.0, noise_bias_max: float = 1.0,
                 noise_scale_min: float = 0.5, noise_scale_max: float = 1.5,
                 noise_amp_min: float = 3.0, noise_amp_max: float = 6.0,
                 ) -> None:
        super().__init__(x, m_struct, window, stride)
        if not (0.0 <= float(mask_rate) < 1.0):
            raise ValueError('mask_rate must satisfy 0 <= rate < 1')
        self.mask_rate = float(mask_rate)
        self._base_seed = int(seed) if seed is not None else None
        self._gen: Optional[torch.Generator] = None
        # Support aliases: point->iid, window->block
        mm = str(mask_mode).lower()
        if mm == 'point':
            mm = 'iid'
        elif mm == 'window':
            mm = 'block'
        self.mask_mode = mm
        if self.mask_mode not in ('iid', 'block', 'corr'):
            raise ValueError("mask_mode must be one of: 'iid', 'block', 'corr', or aliases 'point'/'window'")
        # block params
        self.block_t_min = int(block_t_min)
        self.block_t_max = int(block_t_max)
        self.block_f_min = int(block_f_min)
        self.block_f_max = int(block_f_max)
        self.block_max_blocks = int(block_max_blocks)
        # corr params
        self.corr_t = int(corr_t)
        self.corr_f = int(corr_f)
        # noise params (always enabled on selected masked positions)
        self.noise_kind = str(noise_kind).lower()
        self.noise_sigma = float(noise_sigma)
        self.noise_bias_min = float(noise_bias_min)
        self.noise_bias_max = float(noise_bias_max)
        self.noise_scale_min = float(noise_scale_min)
        self.noise_scale_max = float(noise_scale_max)
        self.noise_amp_min = float(noise_amp_min)
        self.noise_amp_max = float(noise_amp_max)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xw, m_struct_w = super().__getitem__(idx)
        if self.mask_rate == 0.0:
            # No selected masked positions -> feed original (structurally observed) values as input
            return xw, m_struct_w, xw
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
            # target drop only counts structurally observed cells
            struct = (m_struct_w > 0.5).to(dtype=torch.float32)
            obs_count = float(struct.sum().item())
            target_drop = float(self.mask_rate) * obs_count
            if target_drop <= 0.0 or obs_count <= 0.0:
                m_sampled = m_struct_w.clone()
            else:
                # iteratively place windows until approaching target; then top-up with iid if needed
                avg_area = max((self.block_t_min + self.block_t_max) * 0.5, 1.0) * max((self.block_f_min + self.block_f_max) * 0.5, 1.0)
                approx_blocks = int(np.ceil(target_drop / max(avg_area, 1.0)))
                max_blocks = int(self.block_max_blocks) if self.block_max_blocks > 0 else int(approx_blocks * 4 + 10)
                drop = torch.zeros_like(m_struct_w, dtype=torch.float32)
                gen = self._gen
                placed = 0
                attempts = 0
                max_attempts = max(max_blocks * 5, 50)
                while placed < max_blocks and attempts < max_attempts:
                    attempts += 1
                    # sizes (clamped by window size)
                    if gen is not None:
                        rt = int(torch.randint(low=self.block_t_min, high=max(self.block_t_max, self.block_t_min + 1), size=(1,), generator=gen).item())
                        rf = int(torch.randint(low=self.block_f_min, high=max(self.block_f_max, self.block_f_min + 1), size=(1,), generator=gen).item())
                    else:
                        rt = int(np.random.randint(self.block_t_min, max(self.block_t_max, self.block_t_min + 1)))
                        rf = int(np.random.randint(self.block_f_min, max(self.block_f_max, self.block_f_min + 1)))
                    rt = max(1, min(rt, int(L)))
                    rf = max(1, min(rf, int(H)))
                    # positions
                    if gen is not None:
                        t0 = int(torch.randint(low=0, high=max(int(L) - rt + 1, 1), size=(1,), generator=gen).item())
                        f0 = int(torch.randint(low=0, high=max(int(H) - rf + 1, 1), size=(1,), generator=gen).item())
                    else:
                        t0 = int(np.random.randint(0, max(int(L) - rt + 1, 1)))
                        f0 = int(np.random.randint(0, max(int(H) - rf + 1, 1)))
                    drop[t0:t0+rt, f0:f0+rf] = 1.0
                    placed += 1
                    # check progress periodically to avoid excessive loops
                    if placed % 2 == 0 or placed >= max_blocks:
                        cur_drop = (drop * struct).sum().item()
                        if cur_drop >= target_drop:
                            break
                # top-up with iid on remaining observed cells if still under target
                cur_drop = (drop * struct).sum().item()
                if cur_drop < target_drop:
                    remain = (struct * (1.0 - drop))
                    remain_count = float(remain.sum().item())
                    need = max(target_drop - cur_drop, 0.0)
                    if remain_count > 0.0 and need > 0.0:
                        p_extra = min(max(need / remain_count, 0.0), 1.0)
                        if gen is not None:
                            rnd = torch.rand(remain.shape, generator=gen, device=remain.device, dtype=remain.dtype)
                        else:
                            rnd = torch.rand_like(remain)
                        extra = (rnd < p_extra).to(dtype=remain.dtype)
                        drop = torch.clamp(drop + extra, max=1.0)
                keep = (1.0 - drop).to(dtype=m_struct_w.dtype)
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
        # Input construction: always add noise at selected masked locations (no zero-drop path)
        # build drop mask over structurally observed cells that were sampled as missing
        drop = ((m_struct_w > 0.5) & (m_sampled <= 0.5)).to(dtype=torch.float32)
        x_in = self._maybe_apply_noise(xw, drop)
        return x_in, m_sampled, xw

    def _maybe_apply_noise(self, x: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
        """Apply noise on positions where drop==1, keep original elsewhere.

        Noise is generated using the local torch.Generator for reproducibility when available.
        """
        if drop.dtype != torch.float32:
            drop = drop.to(dtype=torch.float32)
        gen = self._gen
        kind = self.noise_kind
        x_noisy = x.clone()
        if kind == 'gaussian':
            if gen is not None:
                noise = torch.randn(x_noisy.shape, generator=gen, device=x_noisy.device, dtype=x_noisy.dtype) * self.noise_sigma
            else:
                noise = torch.randn_like(x_noisy) * self.noise_sigma
            x_noisy = x_noisy + noise * drop
        elif kind == 'bias':
            low, high = self.noise_bias_min, self.noise_bias_max
            if gen is not None:
                r = torch.rand(x_noisy.shape, generator=gen, device=x_noisy.device, dtype=x_noisy.dtype)
            else:
                r = torch.rand_like(x_noisy)
            b = low + (high - low) * r
            x_noisy = x_noisy + b * drop
        elif kind == 'scale':
            a, bmax = self.noise_scale_min, self.noise_scale_max
            if gen is not None:
                r = torch.rand(x_noisy.shape, generator=gen, device=x_noisy.device, dtype=x_noisy.dtype)
            else:
                r = torch.rand_like(x_noisy)
            s = a + (bmax - a) * r
            x_noisy = x_noisy * (1.0 + (s - 1.0) * drop)
        elif kind == 'spike':
            a, bmax = self.noise_amp_min, self.noise_amp_max
            if gen is not None:
                r1 = torch.rand(x_noisy.shape, generator=gen, device=x_noisy.device, dtype=x_noisy.dtype)
                r2 = torch.rand(x_noisy.shape, generator=gen, device=x_noisy.device, dtype=x_noisy.dtype)
            else:
                r1 = torch.rand_like(x_noisy)
                r2 = torch.rand_like(x_noisy)
            amp = a + (bmax - a) * r1
            sign = torch.where(r2 >= 0.5, 1.0, -1.0)
            x_noisy = x_noisy + sign * amp * drop
        else:
            # unknown kind -> fallback to drop zeros
            x_noisy = torch.where(drop > 0.5, torch.zeros_like(x_noisy), x_noisy)
        # Ensure structurally missing cells remain zeroed
        return torch.where(drop > 0.5, x_noisy, x)


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


def _list_csv_candidates(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in os.listdir(folder):
        if name.lower().endswith('.csv'):
            out.append(os.path.join(folder, name))
    return out


def resolve_training_csv(data_dir: str, case: str, train_csv: Optional[str] = None) -> str:
    """Resolve training CSV path without hardcoding a specific filename.

    Policy:
    - If ``train_csv`` is provided, enforce guardrails and return if valid.
    - Otherwise, search under ``{data_dir}/{case}`` and ``{data_dir}/{case}/train``
      for a clean 2025/07-08 file (exclude any path containing 'fdia' or '2025-09').
      Prefer filenames matching 'acopf_2025-07_2025-08_noisy'; fallback to
      '*2025-07_2025-08*_noisy' (e.g., 'clean_noisy'). Prefer files inside '/train/'.
    """
    if train_csv:
        p = os.path.abspath(train_csv)
        if not os.path.exists(p):
            raise SystemExit(f"[error] Training CSV not found: {p}")
        low = p.lower()
        if ('fdia' in low) or ('2025-09' in low):
            raise SystemExit('[error] Training CSV must not contain fdia or 2025-09')
        return p

    case_dir = os.path.join(data_dir, case)
    if not os.path.isdir(case_dir):
        raise SystemExit(f"[error] Case directory not found: {case_dir}")

    candidates: List[str] = []
    candidates += _list_csv_candidates(case_dir)
    candidates += _list_csv_candidates(os.path.join(case_dir, 'train'))

    filt: List[str] = []
    for p in candidates:
        low = p.lower()
        if ('2025-07_2025-08' in low) and ('fdia' not in low) and ('2025-09' not in low):
            filt.append(p)
    if not filt:
        raise SystemExit(
            '[error] Could not locate a clean 2025/07-08 training CSV under '
            f"{case_dir}. Provide --train_csv explicitly."
        )

    def _rank(path: str) -> Tuple[int, int, str]:
        low = os.path.basename(path).lower()
        pri_name = 0 if 'acopf_2025-07_2025-08_noisy' in low else (1 if '2025-07_2025-08_clean_noisy' in low else 2)
        pri_dir = 0 if (os.sep + 'train' + os.sep) in path else 1
        return (pri_name, pri_dir, low)

    filt.sort(key=_rank)
    return os.path.abspath(filt[0])


def create_normal_loaders(
    data_dir: str,
    case: str = 'case14',
    *,
    train_csv: Optional[str] = None,
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
    # input corruption (always enabled on selected masked positions)
    noise_kind: str = 'gaussian',
    noise_sigma: float = 1.0,
    noise_bias_min: int | float = -1.0,
    noise_bias_max: int | float = 1.0,
    noise_scale_min: int | float = 0.5,
    noise_scale_max: int | float = 1.5,
    noise_amp_min: int | float = 3.0,
    noise_amp_max: int | float = 6.0,
    seed: Optional[int] = 1337,
    num_workers: int = 0,
) -> DataModule:
    csv_path = resolve_training_csv(data_dir, case, train_csv)
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
        noise_kind=noise_kind,
        noise_sigma=float(noise_sigma),
        noise_bias_min=float(noise_bias_min), noise_bias_max=float(noise_bias_max),
        noise_scale_min=float(noise_scale_min), noise_scale_max=float(noise_scale_max),
        noise_amp_min=float(noise_amp_min), noise_amp_max=float(noise_amp_max),
    )
    ds_val = SelfSupervisedMaskingDataset(
        Xva, M_val.astype(np.float32), time_length, stride,
        mask_rate=mask_rate, seed=None if seed is None else seed + 1,
        mask_mode=mask_mode,
        block_t_min=block_t_min, block_t_max=block_t_max,
        block_f_min=block_f_min, block_f_max=block_f_max,
        block_max_blocks=block_max_blocks,
        corr_t=corr_t, corr_f=corr_f,
        noise_kind=noise_kind,
        noise_sigma=float(noise_sigma),
        noise_bias_min=float(noise_bias_min), noise_bias_max=float(noise_bias_max),
        noise_scale_min=float(noise_scale_min), noise_scale_max=float(noise_scale_max),
        noise_amp_min=float(noise_amp_min), noise_amp_max=float(noise_amp_max),
    )
    ds_test = SelfSupervisedMaskingDataset(
        Xte, M_test.astype(np.float32), time_length, stride,
        mask_rate=mask_rate, seed=None if seed is None else seed + 2,
        mask_mode=mask_mode,
        block_t_min=block_t_min, block_t_max=block_t_max,
        block_f_min=block_f_min, block_f_max=block_f_max,
        block_max_blocks=block_max_blocks,
        corr_t=corr_t, corr_f=corr_f,
        noise_kind=noise_kind,
        noise_sigma=float(noise_sigma),
        noise_bias_min=float(noise_bias_min), noise_bias_max=float(noise_bias_max),
        noise_scale_min=float(noise_scale_min), noise_scale_max=float(noise_scale_max),
        noise_amp_min=float(noise_amp_min), noise_amp_max=float(noise_amp_max),
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
    'resolve_training_csv',
    'DataModule',
]
