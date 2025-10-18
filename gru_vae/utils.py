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
