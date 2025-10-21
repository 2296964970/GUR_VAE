from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class GaussianDecoder(nn.Module):
    """MLP decoder mapping z[B,Z,T] -> x params [B,T,H].

    This simplified version always uses learnable per-feature log-variance
    (time-invariant), removing the fixed-variance branch.
    """

    def __init__(
        self,
        output_dim: int,
        z_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        *,
        init_logvar: float = -2.0,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError('output_dim must be positive')
        if z_size <= 0:
            raise ValueError('z_size must be positive')
        self.output_dim = int(output_dim)
        self.z_size = int(z_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.output_activation = output_activation

        layers = []
        in_f = self.z_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU(inplace=True))
            in_f = h
        layers.append(nn.Linear(in_f, self.output_dim))
        self.net = nn.Sequential(*layers)

        # Learnable per-feature log-variance (broadcast over [B,T])
        self.logvar_param = nn.Parameter(torch.full((self.output_dim,), float(init_logvar)))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 3:
            raise ValueError('z must have shape [B, Z, T]')
        B, Z, T = z.shape
        if Z != self.z_size:
            raise ValueError('z.shape[1] does not match z_size')
        z_bt_z = z.transpose(1, 2).contiguous()  # [B,T,Z]
        y = self.net(z_bt_z.view(B * T, Z))
        if self.output_activation is not None:
            y = self.output_activation(y)
        mean = y.view(B, T, self.output_dim)
        logvar = self.logvar_param.view(1, 1, self.output_dim).expand_as(mean)
        return mean, logvar


__all__ = ['GaussianDecoder']
