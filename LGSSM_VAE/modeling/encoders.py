from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from LGSSM_VAE.foundation.utils import validate_bth
class CausalConv1d(nn.Module):
    """1D causal convolution using left padding only.

    Given input x[B,C,T], applies left padding of size d*(k-1) then a conv1d with
    dilation=d, kernel_size=k, stride=1, padding=0, ensuring output at t depends
    only on x[:, :, :t+1].
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size <= 0 or dilation <= 0:
            raise ValueError('kernel_size and dilation must be positive')
        self.k = int(kernel_size)
        self.d = int(dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.k, dilation=self.d, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('x must have shape [B,C,T]')
        pad = self.d * (self.k - 1)
        x_pad = F.pad(x, (pad, 0))  # pad left only
        y = self.conv(x_pad)
        return y


class TCNBlock(nn.Module):
    """Residual TCN block with two causal convs and ReLU.

    - No BatchNorm to avoid potential leakage through batch statistics.
    - Optional dropout can be added outside if desired; default kept minimal.
    """

    def __init__(self, channels_in: int, channels_out: int, *, kernel_size: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels_in, channels_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels_out, channels_out, kernel_size, dilation)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv1d(channels_in, channels_out, kernel_size=1) if channels_in != channels_out else nn.Identity()
        p = float(dropout)
        self.dp1 = nn.Dropout(p) if p > 0.0 else nn.Identity()
        self.dp2 = nn.Dropout(p) if p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.dp1(h)
        h = self.act(self.conv2(h))
        h = self.dp2(h)
        return h + self.proj(x)


class CausalTCNEncoder(nn.Module):
    """Causal TCN encoder producing mean and Cholesky factor per timestep.

    API:
      - forward(x[B,T,H]) -> (mu[B,Z,T], chol[B,T,Z,Z])
    """

    def __init__(
        self,
        input_dim: int,
        z_size: int,
        channels: Sequence[int] = (256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.0,
        diag_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError('input_dim must be positive')
        if z_size <= 0:
            raise ValueError('z_size must be positive')
        if kernel_size <= 0:
            raise ValueError('kernel_size must be positive')
        self.input_dim = int(input_dim)
        self.z_size = int(z_size)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.diag_eps = float(diag_eps)

        layers = []
        c_in = self.input_dim
        dilation = 1
        for c_out in self.channels:
            layers.append(TCNBlock(c_in, c_out, kernel_size=self.kernel_size, dilation=dilation, dropout=self.dropout_p))
            c_in = c_out
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        tril_size = self.z_size * (self.z_size + 1) // 2
        self.tril_size = int(tril_size)
        self.head = nn.Conv1d(c_in, self.z_size + self.tril_size, kernel_size=1)
        idx = torch.tril_indices(self.z_size, self.z_size, offset=0)
        self.register_buffer('tril_rows', idx[0])
        self.register_buffer('tril_cols', idx[1])

    def _build_cholesky(self, raw_tril: torch.Tensor) -> torch.Tensor:
        B, T, _ = raw_tril.shape
        device, dtype = raw_tril.device, raw_tril.dtype
        flat = raw_tril.reshape(B * T, self.tril_size)
        L = torch.zeros(B * T, self.z_size, self.z_size, device=device, dtype=dtype)
        L[:, self.tril_rows, self.tril_cols] = flat
        diag_idx = torch.arange(self.z_size, device=device)
        diag = L[:, diag_idx, diag_idx]
        diag = F.softplus(diag) + self.diag_eps
        L[:, diag_idx, diag_idx] = diag
        L = L.view(B, T, self.z_size, self.z_size)
        return L

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = validate_bth(x, 'x')
        if H != self.input_dim:
            raise ValueError('x.shape[2] does not match input_dim')
        x_bct = x.transpose(1, 2).contiguous()  # [B,H,T]
        h = self.tcn(x_bct)
        params = self.head(h)  # [B, Z + tril, T]
        mu_ch = params[:, :self.z_size, :].contiguous()
        raw_tril = params[:, self.z_size:, :].transpose(1, 2).contiguous()  # [B,T,tril]
        chol = self._build_cholesky(raw_tril)
        return mu_ch, chol


__all__ = ['CausalTCNEncoder']
