from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import torch
from torch.utils.data import DataLoader

from .model import TCNVAE
from .metrics import batch_metrics
from .utils import apply_segment_laplace_attacks


@dataclass
class EpochStats:
    loss: float
    nll: float
    kl: float
    mse_obs: float


def _run_epoch(
    *,
    model: TCNVAE,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    beta: Optional[float],
    train: bool,
    # Robust phase controls (None => identity phase)
    robust: bool = False,
    anchor_lambda: float = 1.0,
    clean_fraction: float = 0.4,
    seg_len_min: int = 8,
    seg_len_max: int = 96,
    dims_fraction_min: float = 0.05,
    dims_fraction_max: float = 0.3,
    laplace_scales: Optional[Sequence[float]] = None,
    rng: Optional[torch.Generator] = None,
) -> EpochStats:
    if train and optimizer is None:
        raise ValueError('optimizer must be provided when train=True')
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_nll = torch.zeros_like(total_loss)
    total_kl = torch.zeros_like(total_loss)
    total_mse_obs = torch.zeros_like(total_loss)
    num_batches = 0
    for batch in loader:
        # Training strictly supports normal-only batches: (x_normal, mask).
        if len(batch) != 2:
            raise ValueError('Expected batch of 2 tensors: (x_normal, mask)')
        x, m = batch
        x_target = x.float().to(device)
        m = m.float().to(device)
        if robust:
            if rng is None:
                raise ValueError('rng must be provided for robust training')
            with torch.no_grad():
                x_input, a_mask = apply_segment_laplace_attacks(
                    x_target,
                    m,
                    generator=rng,
                    clean_fraction=float(clean_fraction),
                    seg_len_min=int(seg_len_min),
                    seg_len_max=int(seg_len_max),
                    dims_fraction_min=float(dims_fraction_min),
                    dims_fraction_max=float(dims_fraction_max),
                    laplace_scales=tuple(laplace_scales) if laplace_scales else (0.5, 1.0),
                )
        else:
            x_input = x_target
            a_mask = torch.zeros_like(x_target)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            out = model.elbo_sequence_supervised(x_input, m, x_target, beta=beta)
            loss = out['loss']
            if robust:
                # Anchor loss on non-attacked observed positions only
                with torch.no_grad():
                    keep_mask = (1.0 - a_mask).clamp(min=0.0, max=1.0) * m
                    denom = keep_mask.sum().clamp_min(1.0)
                anchor = ((out['mean'] - x_target) ** 2 * keep_mask).sum() / denom
                loss = loss + float(anchor_lambda) * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            with torch.no_grad():
                metrics = batch_metrics(out['mean'], out['logvar_x'], x_target, m)
                total_loss += loss.detach().to(dtype=total_loss.dtype)
                total_nll += out['nll'].detach().to(dtype=total_loss.dtype)
                total_kl += out['kl'].detach().to(dtype=total_loss.dtype)
                total_mse_obs += metrics['mse_obs'].mean().to(dtype=total_loss.dtype)
                num_batches += 1
        else:
            out = model.elbo_sequence_supervised(x_input, m, x_target, beta=beta)
            metrics = batch_metrics(out['mean'], out['logvar_x'], x_target, m)
            total_loss += out['loss'].detach().to(dtype=total_loss.dtype)
            total_nll += out['nll'].detach().to(dtype=total_loss.dtype)
            total_kl += out['kl'].detach().to(dtype=total_loss.dtype)
            total_mse_obs += metrics['mse_obs'].mean().to(dtype=total_loss.dtype)
            num_batches += 1
    denom = max(num_batches, 1)
    return EpochStats(
        loss=float((total_loss / denom).item()),
        nll=float((total_nll / denom).item()),
        kl=float((total_kl / denom).item()),
        mse_obs=float((total_mse_obs / denom).item()),
    )


class OnlineTrainer:
    """Two-phase trainer with optional robust phase using Laplace segment attacks and anchor loss."""

    def __init__(
        self,
        model: TCNVAE,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[torch.device] = None,
        grad_clip: float = 1e4,
        beta: Optional[float] = None,
        # Robust phase params
        anchor_lambda: float = 1.0,
        clean_fraction: float = 0.4,
        seg_len_min: int = 8,
        seg_len_max: int = 96,
        dims_fraction_min: float = 0.05,
        dims_fraction_max: float = 0.3,
        laplace_scales: Optional[Tuple[float, ...]] = None,
        rng_seed: Optional[int] = 1337,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = float(grad_clip)
        self.beta = beta
        self.model.to(self.device)
        # Robust controls
        self.anchor_lambda = float(anchor_lambda)
        self.clean_fraction = float(clean_fraction)
        self.seg_len_min = int(seg_len_min)
        self.seg_len_max = int(seg_len_max)
        self.dims_fraction_min = float(dims_fraction_min)
        self.dims_fraction_max = float(dims_fraction_max)
        self.laplace_scales = tuple(laplace_scales) if laplace_scales else (0.1, 0.3, 0.7, 1.2)
        # RNG (CPU) for deterministic augmentation
        if rng_seed is not None:
            g = torch.Generator(device='cpu')
            g.manual_seed(int(rng_seed))
            self.rng = g
        else:
            self.rng = torch.Generator(device='cpu')

    def train_epoch(self, loader: DataLoader, *, robust: bool = False) -> EpochStats:
        self.model.train()
        return _run_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=self.grad_clip,
            beta=self.beta,
            train=True,
            robust=robust,
            anchor_lambda=self.anchor_lambda,
            clean_fraction=self.clean_fraction,
            seg_len_min=self.seg_len_min,
            seg_len_max=self.seg_len_max,
            dims_fraction_min=self.dims_fraction_min,
            dims_fraction_max=self.dims_fraction_max,
            laplace_scales=self.laplace_scales,
            rng=self.rng,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochStats:
        self.model.eval()
        # Validation always runs without synthetic attacks
        return _run_epoch(
            model=self.model,
            loader=loader,
            optimizer=None,
            device=self.device,
            grad_clip=self.grad_clip,
            beta=self.beta,
            train=False,
            robust=False,
            anchor_lambda=self.anchor_lambda,
            clean_fraction=self.clean_fraction,
            seg_len_min=self.seg_len_min,
            seg_len_max=self.seg_len_max,
            dims_fraction_min=self.dims_fraction_min,
            dims_fraction_max=self.dims_fraction_max,
            laplace_scales=self.laplace_scales,
            rng=self.rng,
        )

    # No legacy bulk-imputation APIs are provided.
