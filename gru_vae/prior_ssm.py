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
    """AR(1) state-space prior with optional low-rank coupling of process noise.

    Dynamics (per latent dim d): z_t[d] = a[d] * z_{t-1}[d] + eps_t[d].
    Process noise covariance Q = diag(q) + U U^T, where U has rank r (r=0 -> diagonal).
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
        dev = device if device is not None else torch.device('cpu')
        dt = dtype
        import math as _math
        if abs(float(a_init)) < 0.999:
            raw_a_init = 0.5 * _math.log((1.0 + float(a_init)) / (1.0 - float(a_init)))
        else:
            raw_a_init = 0.0
        self.raw_a = nn.Parameter(torch.full((self.D,), float(raw_a_init), device=dev, dtype=dt))
        raw_q_init = _softplus_inverse(q_init)
        raw_P0_init = _softplus_inverse(P0_init)
        self.raw_q = nn.Parameter(torch.full((self.D,), raw_q_init, device=dev, dtype=dt))
        self.m0 = nn.Parameter(torch.full((self.D,), float(m0_init), device=dev, dtype=dt))
        self.raw_P0 = nn.Parameter(torch.full((self.D,), raw_P0_init, device=dev, dtype=dt))
        # Low-rank factor U for process noise coupling (optional)
        if self.rank > 0:
            U = torch.zeros(self.D, self.rank, device=dev, dtype=dt)
            # small random init
            nn.init.normal_(U, mean=0.0, std=0.05)
            self.U = nn.Parameter(U)
        else:
            self.register_parameter('U', None)

    def transition_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.tanh(self.raw_a)
        q = F.softplus(self.raw_q).clamp_min(1e-6)
        P0 = F.softplus(self.raw_P0).clamp_min(1e-6)
        return a, q, self.m0, P0

    @torch.no_grad()
    def init_filter_state(self, batch_size: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        a, q, m0, P0 = self.transition_params()
        dev = device if device is not None else m0.device
        dt = dtype if dtype is not None else m0.dtype
        m = m0.view(1, -1).expand(batch_size, self.D).to(device=dev, dtype=dt)
        P = P0.view(1, -1).expand(batch_size, self.D).to(device=dev, dtype=dt)
        return m, P

    def predict(self, m_prev: torch.Tensor, P_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if m_prev.ndim != 2 or P_prev.ndim != 2:
            raise ValueError('m_prev and P_prev must have shape [B,D]')
        if m_prev.shape != P_prev.shape or m_prev.shape[1] != self.D:
            raise ValueError('m_prev/P_prev must both be [B,D] with D=latent_dim')
        a, q, _m0, _P0 = self.transition_params()
        a2 = a * a
        m_pred = m_prev * a
        # Diagonal prediction for the state variance component
        P_pred = P_prev * a2 + q
        return m_pred, P_pred

    @staticmethod
    def _kl_diag(mu_q: torch.Tensor, s2_q: torch.Tensor, m_p: torch.Tensor, s2_p: torch.Tensor) -> torch.Tensor:
        s2_q = s2_q.clamp_min(1e-12)
        s2_p = s2_p.clamp_min(1e-12)
        logdet_p = torch.log(s2_p).sum(dim=-1)
        logdet_q = torch.log(s2_q).sum(dim=-1)
        inv_p = 1.0 / s2_p
        tr = (s2_q * inv_p).sum(dim=-1)
        diff = (mu_q - m_p)
        quad = (diff * diff * inv_p).sum(dim=-1)
        D = mu_q.shape[-1]
        return 0.5 * (logdet_p - logdet_q - float(D) + tr + quad)

    def kl_q_prior(self, mu_q: torch.Tensor, logvar_q: torch.Tensor, m_pred: torch.Tensor, P_pred: torch.Tensor) -> torch.Tensor:
        if not (mu_q.shape == logvar_q.shape == m_pred.shape == P_pred.shape):
            raise ValueError('All inputs must share shape [B,D]')
        s2_q = torch.exp(logvar_q)
        # If no low-rank factor, fall back to diagonal KL
        if (self.rank <= 0) or (self.U is None):
            return self._kl_diag(mu_q, s2_q, m_pred, P_pred)

        # Sigma_p = diag(P_pred) + U U^T (low-rank update)
        B, D = mu_q.shape
        r = int(self.rank)
        # Clamp for numerical stability
        Ddiag = P_pred.clamp_min(1e-12)
        Dinv = 1.0 / Ddiag
        # Expand U across batch
        U = self.U  # [D, r]
        U_b = U.unsqueeze(0).expand(B, -1, -1)  # [B, D, r]
        # A = D^{-1} U
        A = Dinv.unsqueeze(-1) * U_b  # [B, D, r]
        # M = I + U^T D^{-1} U = I + U^T A
        UT_A = torch.einsum('bdr,bdq->brq', U_b, A)  # [B, r, r]
        I = torch.eye(r, device=UT_A.device, dtype=UT_A.dtype).unsqueeze(0).expand(B, -1, -1)
        M = I + UT_A
        Minv = torch.linalg.inv(M)  # [B, r, r]

        # log|Sigma_p| = log|D| + log|M|
        logdet_D = torch.log(Ddiag).sum(dim=-1)  # [B]
        # batched logdet via slogdet
        sign, logabs = torch.linalg.slogdet(M)
        logdet_M = logabs
        logdet_p = logdet_D + logdet_M  # [B]

        # log|Sigma_q|
        logdet_q = torch.log(s2_q.clamp_min(1e-12)).sum(dim=-1)

        # diag(inv(Sigma_p)) = D^{-1} - diag(A Minv A^T)
        tmp = torch.einsum('bdr,brq->bdq', A, Minv)  # [B, D, r]
        diagK = torch.sum(tmp * A, dim=-1)  # [B, D]
        diag_inv = Dinv - diagK  # [B, D]

        # tr(inv(Sigma_p) Sigma_q) where Sigma_q=diag(s2_q)
        tr_term = (diag_inv * s2_q).sum(dim=-1)  # [B]

        # quadratic term (mu_q - m_p)^T inv(Sigma_p) (mu_q - m_p)
        diff = (mu_q - m_pred)
        v = Dinv * diff  # [B, D]
        b_vec = torch.einsum('bdr,bd->br', A, diff)  # [B, r]
        w = torch.linalg.solve(M, b_vec)  # [B, r]
        Aw = torch.einsum('bdr,br->bd', A, w)  # [B, D]
        inv_times_diff = v - Aw  # [B, D]
        quad = (diff * inv_times_diff).sum(dim=-1)  # [B]

        Dn = D
        kl = 0.5 * (logdet_p - logdet_q - float(Dn) + tr_term + quad)
        return kl


__all__ = ['SSMPrior']
