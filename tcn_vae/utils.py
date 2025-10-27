from __future__ import annotations

from typing import Tuple

import torch
import math


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

__all__ = ['parse_sizes', 'resolve_device', 'first_batch_or_exit']
 
def apply_fdia_noise(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    strength: float,
    generator: torch.Generator,
    fraction: float = 0.15,
    fraction_choices: torch.Tensor | None = None,
    strength_choices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply FDIA-like sparse Gaussian noise per time step on observed features.

    - x: standardized input [B,T,H]
    - mask: observability mask [B,T,H] (1 observed, 0 missing)
    - strength: base stddev of Gaussian noise in standardized domain
    - generator: RNG for reproducibility (CPU generator is acceptable; noise will be moved to device)
    - fraction: per-step feature fraction to attack (constant, not exposed to config)
    - fraction_choices: optional tensor of candidate per-step fractions; when provided,
      a value is sampled (with replacement) for each (batch, time) pair.
    - strength_choices: optional tensor of candidate per-step noise stddev values (non-negative);
      when provided, a value is sampled (with replacement) per (batch, time).

    For each batch b and time t, sample an attack subset over observed features with Bernoulli(fraction).
    If no observed feature is selected, force-select ceil(fraction * N_obs) random observed features (or 1 if N_obs>0).
    
    Strength sampling (default randomized): if no strength_choices are provided, the per-step
    strength is sampled from a uniform window around the base strength: U[0.5*strength, 1.5*strength].
    """
    if x.shape != mask.shape:
        raise ValueError('x and mask must share shape [B,T,H]')
    if not (0.0 <= fraction <= 1.0):
        raise ValueError('fraction must be in [0,1]')
    if fraction_choices is not None:
        if fraction_choices.ndim != 1:
            raise ValueError('fraction_choices must be a 1D tensor if provided')
        if fraction_choices.numel() == 0:
            fraction_choices = None
        else:
            if (fraction_choices < 0.0).any() or (fraction_choices > 1.0).any():
                raise ValueError('fraction_choices values must lie within [0, 1]')
            fraction_choices = fraction_choices.to(dtype=torch.float32, device='cpu')
    if strength_choices is not None:
        if strength_choices.ndim != 1:
            raise ValueError('strength_choices must be a 1D tensor if provided')
        if strength_choices.numel() == 0:
            strength_choices = None
        else:
            if (strength_choices < 0.0).any():
                raise ValueError('strength_choices values must be non-negative')
            strength_choices = strength_choices.to(dtype=torch.float32, device='cpu')
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
            frac_bt = frac
            if fraction_choices is not None:
                choice_idx = torch.randint(
                    low=0,
                    high=fraction_choices.numel(),
                    size=(1,),
                    generator=generator,
                    dtype=torch.int64,
                ).item()
                frac_bt = float(fraction_choices[choice_idx].item())
                frac_bt = max(0.0, min(1.0, frac_bt))
            # Determine per-(b,t) noise strength
            if strength_choices is not None:
                s_idx = torch.randint(
                    low=0,
                    high=strength_choices.numel(),
                    size=(1,),
                    generator=generator,
                    dtype=torch.int64,
                ).item()
                strength_bt = max(0.0, float(strength_choices[s_idx].item()))
            else:
                # Default: sample around base strength using U[0.5*strength, 1.5*strength]
                scale = 0.5 + torch.rand((), generator=generator, dtype=dtype).item()
                strength_bt = max(0.0, float(strength) * scale)
            obs_idx = (m_cpu[b, t, :] > 0.5).nonzero(as_tuple=False).flatten()
            n_obs = int(obs_idx.numel())
            if n_obs == 0:
                continue
            if frac_bt <= 0.0:
                continue
            cluster_size = max(1, int(math.ceil(frac_bt * n_obs)))
            perm = torch.randperm(n_obs, generator=generator)
            sel_idx = obs_idx[perm[:cluster_size]]
            # Generate a coordinated shift shared across the attacked subset
            shared_shift = torch.randn((), generator=generator, dtype=dtype) * strength_bt
            small_jitter = torch.randn(cluster_size, generator=generator, dtype=dtype) * (0.1 * strength_bt)
            attack_values = shared_shift + small_jitter
            out_cpu[b, t, sel_idx] = x_cpu[b, t, sel_idx] + attack_values
    return out_cpu.to(device=device)

__all__ += ['apply_fdia_noise']
