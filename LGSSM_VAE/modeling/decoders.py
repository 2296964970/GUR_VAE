from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LGSSM_VAE.foundation.utils import softplus_inverse
class GaussianDecoder(nn.Module):
    """MLP decoder mapping z[B,Z,T] -> time-varying Gaussian params [B,T,H].

    Changes:
    - Predict per-step mean and log-variance (time-varying). The final linear
      layer outputs 2*H; the second half is transformed via softplus->log.
    - No time-invariant variance parameter is used.
    """

    def __init__(
        self,
        output_dim: int,
        z_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        *,
        init_logvar: float = -2.0,
        output_activation: Optional[nn.Module] = None,
        eps: float = 1e-6,
        logvar_min: float = -5.0,
        logvar_max: float = 2.302585092994046,
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
        self.eps = float(eps)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        layers = []
        in_f = self.z_size
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU(inplace=True))
            in_f = h
        # Output 2*H: [mean, raw_var]
        out = nn.Linear(in_f, 2 * self.output_dim)
        # Initialize raw_var bias so that softplus(raw_var)->exp(init_logvar)
        with torch.no_grad():
            # target variance = exp(init_logvar)
            target_var = float(torch.exp(torch.tensor(float(init_logvar))).item())
            raw_init = softplus_inverse(target_var, eps=self.eps)
            out.bias[self.output_dim:].fill_(raw_init)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 3:
            raise ValueError('z must have shape [B, Z, T]')
        B, Z, T = z.shape
        if Z != self.z_size:
            raise ValueError('z.shape[1] does not match z_size')
        z_bt_z = z.transpose(1, 2).contiguous()  # [B,T,Z]
        y = self.net(z_bt_z.view(B * T, Z))  # [B*T, 2H]
        y = y.view(B, T, 2 * self.output_dim)
        mean_raw, raw = torch.split(y, self.output_dim, dim=-1)
        mean = self.output_activation(mean_raw) if self.output_activation is not None else mean_raw
        # logvar = log(softplus(raw) + eps), then clamp to prevent under/over-confidence
        logvar = torch.log(F.softplus(raw) + self.eps)
        # Bounds on log-variance to keep decoder uncertainty reasonable
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        return mean, logvar


__all__ = ['GaussianDecoder']
