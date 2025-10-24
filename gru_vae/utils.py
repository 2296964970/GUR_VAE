from __future__ import annotations

from typing import Tuple

import torch


def parse_sizes(s: str) -> Tuple[int, ...]:
    if not s:
        return tuple()
    return tuple(int(part) for part in s.split(',') if part.strip())


def resolve_device(arg: str) -> torch.device:
    dev = arg.lower().strip()
    if dev == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA requested but not available. Falling back to CPU.')
        dev = 'cpu'
    return torch.device(dev)


def first_batch_or_exit(loader, err_msg: str):
    try:
        return next(iter(loader))
    except StopIteration:
        raise SystemExit(err_msg)


# Shape validation helpers to reduce repetition across modules

def validate_bth(x: torch.Tensor, name: str = 'x') -> Tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, H]")
    B, T, H = x.shape
    return int(B), int(T), int(H)


def validate_bh(x: torch.Tensor, name: str = 'x_t') -> Tuple[int, int]:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [B,H]")
    B, H = x.shape
    return int(B), int(H)

__all__ = ['parse_sizes', 'resolve_device', 'first_batch_or_exit']
 
def apply_fdia_noise(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float,
    generator: torch.Generator,
    fraction: float = 0.15,
) -> torch.Tensor:
    """Apply FDIA-like sparse Gaussian noise per time step on observed features.

    - x: standardized input [B,T,H]
    - mask: observability mask [B,T,H] (1 observed, 0 missing)
    - strength: stddev of Gaussian noise in standardized domain
    - generator: RNG for reproducibility (CPU generator is acceptable; noise will be moved to device)
    - fraction: per-step feature fraction to attack (constant, not exposed to config)

    For each batch b and time t, sample an attack subset over observed features with Bernoulli(fraction).
    If no observed feature is selected, force-select ceil(fraction * N_obs) random observed features (or 1 if N_obs>0).
    Noise is added only on attacked observed positions.
    """
    if x.shape != mask.shape:
        raise ValueError('x and mask must share shape [B,T,H]')
    if not (0.0 <= fraction <= 1.0):
        raise ValueError('fraction must be in [0,1]')
    B, T, H = x.shape
    device = x.device
    dtype = x.dtype
    # Work on CPU for RNG generator compatibility, then move back to device
    x_cpu = x.detach().to('cpu')
    m_cpu = mask.detach().to('cpu')
    out_cpu = x_cpu.clone()
    frac = float(fraction)
    for b in range(B):
        for t in range(T):
            obs_idx = (m_cpu[b, t, :] > 0.5).nonzero(as_tuple=False).flatten()
            n_obs = int(obs_idx.numel())
            if n_obs == 0:
                # No observed features; fall back to full domain sampling (will be masked out later)
                # Still generate noise vector for shape consistency
                noise_vec = torch.randn(H, generator=generator, dtype=dtype)
                out_cpu[b, t, :] = x_cpu[b, t, :] + strength * noise_vec * m_cpu[b, t, :]
                continue
            # Bernoulli sampling over observed set
            p = torch.full((n_obs,), frac, dtype=dtype)
            bern = torch.bernoulli(p, generator=generator)
            sel = bern > 0.5
            if int(sel.sum().item()) == 0:
                # Force-select ceil(frac * n_obs), at least 1
                k = max(1, int((frac * n_obs + 0.9999) // 1))
                perm = torch.randperm(n_obs, generator=generator)
                take = perm[:k]
                att_idx = obs_idx[take]
            else:
                att_idx = obs_idx[sel]
            attack_mask = torch.zeros(H, dtype=dtype)
            attack_mask[att_idx] = 1.0
            noise_vec = torch.randn(H, generator=generator, dtype=dtype)
            out_cpu[b, t, :] = x_cpu[b, t, :] + strength * noise_vec * attack_mask
    return out_cpu.to(device=device)

__all__ += ['apply_fdia_noise']
