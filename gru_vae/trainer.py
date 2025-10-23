from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .model import OnlineGPVAE
from .metrics import batch_metrics


@dataclass
class EpochStats:
    loss: float
    nll: float
    kl: float
    mse_obs: float


def _run_epoch(
    *,
    model: OnlineGPVAE,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    beta: Optional[float],
    train: bool,
) -> EpochStats:
    elbo_fn = model.elbo_sequence
    if train and optimizer is None:
        raise ValueError('optimizer must be provided when train=True')
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_nll = torch.zeros_like(total_loss)
    total_kl = torch.zeros_like(total_loss)
    total_mse_obs = torch.zeros_like(total_loss)
    num_batches = 0
    for batch in loader:
        if len(batch) == 3:
            x_input, m, x_target = batch
        elif len(batch) == 2:
            x_input, m = batch
            x_target = x_input
        else:
            raise ValueError('Expected batch of 2 or 3 tensors')
        x_input = x_input.float().to(device)
        m = m.float().to(device)
        x_target = x_target.float().to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            out = model.elbo_sequence_supervised(x_input, m, x_target, beta=beta)
            loss = out['loss']
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
    def __init__(
        self,
        model: OnlineGPVAE,
        optimizer: torch.optim.Optimizer,
        *,
        device: Optional[torch.device] = None,
        grad_clip: float = 1e4,
        beta: Optional[float] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = float(grad_clip)
        self.beta = beta
        self.model.to(self.device)

    def _to_device(self, *tensors: torch.Tensor):
        return tuple(t.to(self.device) for t in tensors)

    def train_epoch(self, loader: DataLoader) -> EpochStats:
        self.model.train()
        return _run_epoch(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=self.grad_clip,
            beta=self.beta,
            train=True,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochStats:
        self.model.eval()
        return _run_epoch(
            model=self.model,
            loader=loader,
            optimizer=None,
            device=self.device,
            grad_clip=self.grad_clip,
            beta=self.beta,
            train=False,
        )

    # No legacy bulk-imputation APIs are provided.
