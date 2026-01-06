from __future__ import annotations

from typing import Tuple, Sequence

import torch
import math


def softplus_inverse(value: float, eps: float = 1e-6) -> float:
    """Inverse of softplus for positive scalar values with epsilon floor."""
    v = max(float(value), eps)
    return float(math.log(math.expm1(v)))


def resolve_device(arg: str) -> torch.device:
    dev = arg.lower().strip()
    if dev == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA requested but not available. Falling back to CPU.')
        dev = 'cpu'
    return torch.device(dev)


def first_batch_or_exit(loader, err_msg: str):
    batch = next(iter(loader), None)
    if batch is None:
        raise SystemExit(err_msg)
    return batch


# Shape validation helpers to reduce repetition across modules

def validate_bth(x: torch.Tensor, name: str = 'x') -> Tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, H]")
    B, T, H = x.shape
    return int(B), int(T), int(H)



def _sample_laplace(shape: torch.Size, scale: float, *, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Sample Laplace(0, b) using inverse CDF with an explicit RNG generator (CPU).

    Uses: X = b * sign(U - 0.5) * log(1 - 2*|U - 0.5|), where U ~ Uniform(0, 1).
    """
    U = torch.rand(shape, generator=generator, dtype=dtype)
    # Clamp away from exactly 0 or 1 to avoid log(0)
    U = U.clamp(1e-8, 1 - 1e-8)
    left = U < 0.5
    out = torch.empty_like(U)
    out[left] = scale * torch.log(2 * U[left])
    out[~left] = -scale * torch.log(2 * (1 - U[~left]))
    return out


def apply_segment_laplace_attacks(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
    clean_fraction: float = 0.4,
    seg_len_min: int = 8,
    seg_len_max: int = 96,
    dims_fraction_min: float = 0.05,
    dims_fraction_max: float = 0.3,
    laplace_scales: Sequence[float] = (0.1, 0.3, 0.7, 1.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply sparse, contiguous-time Laplace attacks and return (x_attacked, attack_mask).

    - x/mask: standardized inputs [B,T,H], observed mask in {0,1}
    - clean_fraction: probability of keeping a sample window entirely clean in the batch
    - seg_len_min/max: inclusive length bounds for a single attacked segment per sample
    - dims_fraction_min/max: fraction range of feature dimensions to attack within that segment
    - laplace_scales: candidate Laplace scales b; a value is sampled per sample

    Only observed entries within the chosen segment and feature subset are modified.
    """
    if x.shape != mask.shape:
        raise ValueError("x and mask must share shape [B,T,H]")
    if not (0.0 <= clean_fraction <= 1.0):
        raise ValueError("clean_fraction must be in [0,1]")
    B, T, H = x.shape
    if seg_len_min <= 0 or seg_len_max <= 0:
        raise ValueError("seg_len_min/max must be positive")
    if seg_len_min > seg_len_max:
        raise ValueError("seg_len_min must be <= seg_len_max")
    if dims_fraction_min < 0.0 or dims_fraction_max > 1.0 or dims_fraction_min > dims_fraction_max:
        raise ValueError("dims_fraction_min/max must satisfy 0<=min<=max<=1")

    scales = tuple(float(s) for s in laplace_scales) or (1.0,)

    device = x.device
    dtype = x.dtype
    # Work on CPU for deterministic generator; move back to device at end
    x_cpu = x.detach().to("cpu")
    m_cpu = mask.detach().to("cpu")
    out_cpu = x_cpu.clone()
    attack_mask = torch.zeros_like(x_cpu, dtype=torch.float32)

    for b in range(B):
        u_keep = torch.rand((), generator=generator).item()
        if u_keep < float(clean_fraction):
            continue

        Lmin = int(seg_len_min)
        Lmax = max(Lmin, int(min(seg_len_max, T)))
        L = int(torch.randint(low=Lmin, high=Lmax + 1, size=(1,), generator=generator).item())

        max_start = max(int(T - L), 0)
        s = int(torch.randint(low=0, high=max_start + 1, size=(1,), generator=generator).item())
        e = min(int(s + L), int(T))
        L_eff = int(e - s)

        fmin = float(dims_fraction_min)
        fmax = float(dims_fraction_max)
        frac = fmin + (fmax - fmin) * torch.rand((), generator=generator).item()
        k = max(1, int(math.ceil(frac * H)))

        obs_any = (m_cpu[b, s:e, :] > 0.5).any(dim=0)
        cand_idx = obs_any.nonzero(as_tuple=False).flatten()
        cand_idx = cand_idx if cand_idx.numel() else torch.arange(H)

        cand_n = int(cand_idx.numel())
        k = min(int(k), cand_n)
        perm = torch.randperm(cand_n, generator=generator)
        dims_sel = cand_idx[perm[:k]]

        idx_scale = int(torch.randint(low=0, high=len(scales), size=(1,), generator=generator).item())
        b_scale = float(scales[idx_scale])

        noise_rect = _sample_laplace(
            torch.Size([L_eff, k]),
            b_scale,
            generator=generator,
            dtype=x_cpu.dtype,
        )

        for ti, t in enumerate(range(s, e)):
            obs = (m_cpu[b, t, dims_sel] > 0.5)
            sel = dims_sel[obs]
            out_cpu[b, t, sel] = x_cpu[b, t, sel] + noise_rect[ti, obs]
            attack_mask[b, t, sel] = 1.0

    return out_cpu.to(device=device, dtype=dtype), attack_mask.to(device=device)

__all__ = [
    "apply_segment_laplace_attacks",
    "first_batch_or_exit",
    "resolve_device",
    "softplus_inverse",
    "validate_bth",
]
