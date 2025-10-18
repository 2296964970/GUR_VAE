from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import validate_bth, validate_bh


def raw_to_logvar(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = F.softplus(raw) + eps
    return torch.log(var)


class CausalGRUEncoder(nn.Module):
    """Causal GRU encoder producing diagonal Gaussian parameters per step.

    Sequence API:
      - forward(x[B,T,H]) -> (mu[B,Z,T], logvar[B,Z,T])
    Online API:
      - step(x_t[B,H], h) -> (mu_t[B,Z], logvar_t[B,Z], h_new)
    """

    def __init__(
        self,
        input_dim: int,
        z_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive')
        if z_size <= 0:
            raise ValueError('z_size must be positive')
        if hidden_size <= 0:
            raise ValueError('hidden_size must be positive')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive')

        self.input_dim = int(input_dim)
        self.z_size = int(z_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_size, 2 * self.z_size)

    def _split_params(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.head(h_t)
        mu, raw = torch.split(params, self.z_size, dim=-1)
        logvar = raw_to_logvar(raw)
        return mu, logvar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = validate_bth(x, 'x')
        if H != self.input_dim:
            raise ValueError('x.shape[2] does not match input_dim')
        out, _h = self.gru(x)
        params = self.head(out)
        mu, raw = torch.split(params, self.z_size, dim=-1)
        logvar = raw_to_logvar(raw)
        return mu.transpose(1, 2).contiguous(), logvar.transpose(1, 2).contiguous()

    @torch.no_grad()
    def init_hidden(self, batch_size: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        dev = device if device is not None else torch.device('cpu')
        dt = dtype if dtype is not None else torch.float32
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=dev, dtype=dt)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H = validate_bh(x_t, 'x_t')
        if H != self.input_dim:
            raise ValueError('x_t.shape[1] does not match input_dim')
        if h.ndim != 3 or h.shape[0] != self.num_layers or h.shape[2] != self.hidden_size or h.shape[1] != B:
            raise ValueError('h must have shape [num_layers,B,hidden_size] and match batch size')
        out, h_new = self.gru(x_t.unsqueeze(1), h)
        h_t = out[:, -1, :]
        mu_t, logvar_t = self._split_params(h_t)
        return mu_t, logvar_t, h_new


__all__ = ['CausalGRUEncoder']
