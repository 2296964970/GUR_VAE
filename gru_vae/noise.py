"""Noise injection policies for self-supervised training and simulation.

This module provides configurable policies to generate corruption masks and
apply synthetic attacks to clean data for evaluation. The mask convention is
aligned with Dataset3 in the project: 1 = unaltered (observed), 0 = injected.

Policies
- point: random independent points across [T,H] according to a target rate
- window: contiguous windows over time with optional feature subsets

Noise kinds (applied at masked positions)
- gaussian: additive Gaussian N(0, sigma)
- bias: additive uniform bias in [bias_min, bias_max]
- scale: multiplicative factor in [scale_min, scale_max]
- spike: replace with outlier spikes with magnitude in [amp_min, amp_max]

All functions are deterministic under a provided numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PointPolicy:
    """Point-wise corruption policy.

    Attributes:
        rate: Fraction of entries to corrupt in [0,1).
    """
    rate: float = 0.01


@dataclass
class WindowPolicy:
    """Sliding-window corruption policy along time with feature subset.

    Attributes:
        window_t: Tuple[min_len, max_len] inclusive bounds for time length.
        windows_per_seq: Number of windows to place per sequence (approx).
        feature_ratio: Fraction of features per window to corrupt (0-1].
        allow_overlap: Whether multiple windows may overlap.
    """
    window_t: Tuple[int, int] = (5, 20)
    windows_per_seq: int = 2
    feature_ratio: float = 1.0
    allow_overlap: bool = True


def sample_point_mask(struct_mask: np.ndarray, rate: float, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample an independent point mask given a structural mask.

    Args:
        struct_mask: Structural availability mask [T,H], 1=valid cell.
        rate: Target corruption rate in [0,1).
        rng: Optional numpy Generator for determinism.
    Returns:
        Binary mask [T,H] with 1=kept, 0=corrupted, respecting struct_mask.
    """
    if not (0.0 <= float(rate) < 1.0):
        raise ValueError("rate must satisfy 0 <= rate < 1")
    if rng is None:
        rng = np.random.default_rng()
    keep = rng.random(struct_mask.shape, dtype=np.float64) >= float(rate)
    m = (struct_mask.astype(np.float64) > 0.5) & keep
    return m.astype(np.float32)


def sample_window_mask(
    struct_mask: np.ndarray,
    *,
    window_t: Tuple[int, int] = (5, 20),
    windows_per_seq: int = 2,
    feature_ratio: float = 1.0,
    allow_overlap: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a time-window mask with optional feature subset per window.

    Each window selects a time span [t0, t0+L) and a subset of features; cells
    inside the selection are marked as corrupted (0). The rest keep the
    structural mask value.

    Args:
        struct_mask: Structural availability mask [T,H].
        window_t: (min_len, max_len) inclusive bounds for time length.
        windows_per_seq: Number of windows to place (approximate upper bound).
        feature_ratio: Fraction of features per window in (0,1].
        allow_overlap: Whether windows may overlap; if False, naive re-sampling
            is used to reduce overlap (best-effort).
        rng: Optional numpy Generator for determinism.
    Returns:
        Binary mask [T,H] with 1=kept, 0=corrupted, respecting struct_mask.
    """
    T, H = struct_mask.shape
    if rng is None:
        rng = np.random.default_rng()
    f_count = max(1, int(round(float(feature_ratio) * H)))
    t_min, t_max = int(window_t[0]), int(window_t[1])
    t_min = max(1, t_min)
    t_max = max(t_min, t_max)
    m = (struct_mask > 0.5).astype(np.float32)
    drop = np.zeros_like(m, dtype=np.float32)
    placed = 0
    attempts = 0
    max_attempts = max(10 * windows_per_seq, 50)
    while placed < windows_per_seq and attempts < max_attempts:
        attempts += 1
        L = int(rng.integers(low=t_min, high=t_max + 1))
        t0 = int(rng.integers(low=0, high=max(T - L + 1, 1)))
        feats = np.sort(rng.choice(H, size=f_count, replace=False))
        if not allow_overlap:
            region = drop[t0:t0+L, :]
            if np.any(region[:, feats] > 0.5):
                continue
        drop[t0:t0+L, feats] = 1.0
        placed += 1
    keep = (1.0 - drop)
    return (m * keep).astype(np.float32)


def apply_noise(
    x: np.ndarray,
    mask_keep: np.ndarray,
    *,
    kind: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
    sigma: float = 1.0,
    bias_min: float = -1.0,
    bias_max: float = 1.0,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    amp_min: float = 3.0,
    amp_max: float = 6.0,
) -> np.ndarray:
    """Apply synthetic corruption to x at positions where mask_keep == 0.

    Args:
        x: Clean array [T,H].
        mask_keep: Binary mask [T,H], 1=keep, 0=corrupt.
        kind: One of {'gaussian','bias','scale','spike'}.
        rng: Optional numpy Generator for determinism.
        sigma, bias_min, bias_max, scale_min, scale_max, amp_min, amp_max: parameters.
    Returns:
        Corrupted array with the same shape as x.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = x.astype(np.float32, copy=True)
    keep = (mask_keep.astype(np.float32) > 0.5)
    if kind == "gaussian":
        noise = rng.normal(loc=0.0, scale=float(sigma), size=x.shape).astype(np.float32)
        x[~keep] = x[~keep] + noise[~keep]
    elif kind == "bias":
        b = rng.uniform(low=float(bias_min), high=float(bias_max), size=x.shape).astype(np.float32)
        x[~keep] = x[~keep] + b[~keep]
    elif kind == "scale":
        s = rng.uniform(low=float(scale_min), high=float(scale_max), size=x.shape).astype(np.float32)
        x[~keep] = x[~keep] * s[~keep]
    elif kind == "spike":
        a = rng.uniform(low=float(amp_min), high=float(amp_max), size=x.shape).astype(np.float32)
        sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
        x[~keep] = x[~keep] + sign[~keep] * a[~keep]
    else:
        raise ValueError("Unknown noise kind: %r" % (kind,))
    return x


__all__ = [
    "PointPolicy",
    "WindowPolicy",
    "sample_point_mask",
    "sample_window_mask",
    "apply_noise",
]

