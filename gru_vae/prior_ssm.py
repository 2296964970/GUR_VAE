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
    """Diagonal AR(1) state-space prior over latent dimensions.

    z_t[d] = a[d] * z_{t-1}[d] + eps_t[d],   eps_t[d] ~ N(0, q[d])
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        a_init: float = 0.95,
        q_init: float = 0.1,
        m0_init: float = 0.0,
        P0_init: float = 1.0,
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
        return self._kl_diag(mu_q, s2_q, m_pred, P_pred)


__all__ = ['SSMPrior']
