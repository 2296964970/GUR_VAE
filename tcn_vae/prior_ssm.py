from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(value: float, eps: float = 1e-6) -> float:
    v = max(float(value), eps)
    return math.log(math.expm1(v))


class SSMPrior(nn.Module):
    """Sparse low-rank VAR(1) state-space prior.

    Dynamics: z_t = A z_{t-1} + eps_t, eps_t ~ N(0, D + U U^T).
    A is masked to enforce sparsity and stabilised to satisfy rho(A) < 1.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        a_init: float = 0.95,
        q_init: float = 0.1,
        m0_init: float = 0.0,
        P0_init: float = 1.0,
        rank: int = 4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError('latent_dim must be positive')
        if q_init <= 0.0:
            raise ValueError('q_init must be positive')
        if P0_init <= 0.0:
            raise ValueError('P0_init must be positive')
        self.D = int(latent_dim)
        self.rank = int(rank) if rank is not None else 0
        self.jitter = 1e-6
        self.variance_floor = 1e-6
        dev = device if device is not None else torch.device('cpu')
        dt = dtype

        self.register_buffer('sparsity_mask', self._build_band_mask(self.D, bandwidth=min(4, max(self.D - 1, 0))))

        raw_A = torch.zeros(self.D, self.D, device=dev, dtype=dt)
        if abs(float(a_init)) < 0.999:
            raw_diag = 0.5 * math.log((1.0 + float(a_init)) / (1.0 - float(a_init)))
        else:
            raw_diag = 0.0
        raw_A.fill_(0.0)
        raw_A.diagonal().fill_(raw_diag)
        if self.D > 1:
            noise = torch.zeros_like(raw_A)
            nn.init.normal_(noise, mean=0.0, std=0.05)
            raw_A = raw_A + noise * self.sparsity_mask
            raw_A.diagonal().fill_(raw_diag)
        self.raw_A = nn.Parameter(raw_A)

        raw_q_init = _softplus_inverse(q_init)
        raw_P0_init = _softplus_inverse(P0_init)
        self.raw_q = nn.Parameter(torch.full((self.D,), raw_q_init, device=dev, dtype=dt))
        self.m0 = nn.Parameter(torch.full((self.D,), float(m0_init), device=dev, dtype=dt))
        self.raw_P0 = nn.Parameter(torch.full((self.D,), raw_P0_init, device=dev, dtype=dt))

        if self.rank > 0:
            U = torch.zeros(self.D, self.rank, device=dev, dtype=dt)
            nn.init.normal_(U, mean=0.0, std=0.05)
            self.U = nn.Parameter(U)
        else:
            self.register_parameter('U', None)

    @staticmethod
    def _build_band_mask(dim: int, bandwidth: int) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros(0, 0)
        ar = torch.arange(dim)
        mask = (ar.unsqueeze(1) - ar.unsqueeze(0)).abs() <= bandwidth
        return mask.to(dtype=torch.float32)

    def _mask_matrix(self, mat: torch.Tensor) -> torch.Tensor:
        mask = self.sparsity_mask.to(device=mat.device, dtype=mat.dtype)
        return mat * mask

    def _stabilize(self, A: torch.Tensor) -> torch.Tensor:
        singular_vals = torch.linalg.svdvals(A)
        sigma_max = torch.max(singular_vals)
        scale = torch.clamp(0.99 / (sigma_max + 1e-8), max=1.0)
        return A * scale

    def transition_matrices(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = device if device is not None else self.raw_A.device
        dt = dtype if dtype is not None else self.raw_A.dtype
        A_raw = self.raw_A.to(device=dev, dtype=dt)
        A_masked = self._mask_matrix(torch.tanh(A_raw))
        A = self._stabilize(A_masked)

        q_diag = F.softplus(self.raw_q).clamp_min(self.variance_floor).to(device=dev, dtype=dt)
        Q = torch.diag(q_diag)
        if self.rank > 0 and self.U is not None:
            U = self.U.to(device=dev, dtype=dt)
            Q = Q + torch.matmul(U, U.t())
        Q = Q + self.jitter * torch.eye(self.D, device=dev, dtype=dt)

        m0 = self.m0.to(device=dev, dtype=dt)
        P0_diag = F.softplus(self.raw_P0).clamp_min(self.variance_floor).to(device=dev, dtype=dt)
        P0 = torch.diag(P0_diag) + self.jitter * torch.eye(self.D, device=dev, dtype=dt)
        return A, Q, m0, P0

    @torch.no_grad()
    def init_filter_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        A, Q, m0, P0 = self.transition_matrices(device=device, dtype=dtype)
        dev = m0.device
        dt = m0.dtype
        m = m0.view(1, -1).expand(batch_size, self.D).clone()
        P = P0.unsqueeze(0).expand(batch_size, self.D, self.D).clone()
        return m.to(device=dev, dtype=dt), P.to(device=dev, dtype=dt)

    def predict(
        self,
        m_prev: torch.Tensor,
        P_prev: torch.Tensor,
        *,
        A: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if m_prev.ndim != 2:
            raise ValueError('m_prev must have shape [B,D]')
        if P_prev.ndim != 3:
            raise ValueError('P_prev must have shape [B,D,D]')
        if m_prev.shape[0] != P_prev.shape[0] or P_prev.shape[1] != self.D or P_prev.shape[2] != self.D:
            raise ValueError('P_prev must be [B,D,D] matching m_prev and latent_dim')
        dev = m_prev.device
        dt = m_prev.dtype
        if A is None or Q is None:
            A, Q, _, _ = self.transition_matrices(device=dev, dtype=dt)
        else:
            A = A.to(device=dev, dtype=dt)
            Q = Q.to(device=dev, dtype=dt)

        P_prev = P_prev.to(device=dev, dtype=dt)
        m_pred = torch.matmul(m_prev, A.t())
        AP = torch.matmul(A, P_prev)
        P_pred = torch.matmul(AP, A.t()) + Q.unsqueeze(0)
        P_pred = 0.5 * (P_pred + P_pred.transpose(1, 2))
        return m_pred, P_pred

    def kl_q_prior(
        self,
        mu_q: torch.Tensor,
        chol_q: torch.Tensor,
        m_pred: torch.Tensor,
        P_pred: torch.Tensor,
    ) -> torch.Tensor:
        if mu_q.ndim != 2 or chol_q.ndim != 3:
            raise ValueError('mu_q must be [B,D] and chol_q must be [B,D,D]')
        if mu_q.shape[0] != chol_q.shape[0] or mu_q.shape[1] != self.D or chol_q.shape[1] != self.D or chol_q.shape[2] != self.D:
            raise ValueError('chol_q must have shape [B,D,D] matching mu_q')
        if m_pred.shape != mu_q.shape:
            raise ValueError('m_pred must have shape [B,D]')
        if P_pred.ndim != 3 or P_pred.shape[0] != mu_q.shape[0] or P_pred.shape[1] != self.D or P_pred.shape[2] != self.D:
            raise ValueError('P_pred must have shape [B,D,D]')

        device, dtype = mu_q.device, mu_q.dtype
        eye = torch.eye(self.D, device=device, dtype=dtype)
        Sigma_q = torch.matmul(chol_q, chol_q.transpose(-1, -2))
        Sigma_p = P_pred + self.jitter * eye

        L_p = torch.linalg.cholesky(Sigma_p)
        logdet_p = 2.0 * torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)).sum(dim=-1)
        L_q_diag = torch.diagonal(chol_q, dim1=-2, dim2=-1)
        logdet_q = 2.0 * torch.log(L_q_diag.clamp_min(self.variance_floor)).sum(dim=-1)

        inv_p_Sigma_q = torch.cholesky_solve(Sigma_q, L_p)
        tr_term = torch.diagonal(inv_p_Sigma_q, dim1=-2, dim2=-1).sum(dim=-1)

        diff = (mu_q - m_pred).unsqueeze(-1)
        inv_p_diff = torch.cholesky_solve(diff, L_p)
        quad = torch.sum(diff.squeeze(-1) * inv_p_diff.squeeze(-1), dim=-1)

        return 0.5 * (logdet_p - logdet_q - float(self.D) + tr_term + quad)


__all__ = ['SSMPrior']
