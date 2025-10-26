from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(value: float, eps: float = 1e-6) -> float:
    v = max(float(value), eps)
    return float(torch.log(torch.expm1(torch.tensor(v))).item())


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
            raw_init = _softplus_inverse(target_var, eps=self.eps)
            if out.bias is not None:
                # Second half bias corresponds to raw_var
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
        if self.output_activation is not None:
            # Apply optional activation to the mean part only after split
            pass
        y = y.view(B, T, 2 * self.output_dim)
        mean_raw, raw = torch.split(y, self.output_dim, dim=-1)
        if self.output_activation is not None:
            mean = self.output_activation(mean_raw)
        else:
            mean = mean_raw
        # logvar = log(softplus(raw) + eps), then clamp to prevent variance blow-up
        logvar = torch.log(F.softplus(raw) + self.eps)
        # Hard upper bound on log-variance: ~ log(10.0)
        LOGVAR_MAX = 2.302585092994046
        logvar = torch.clamp(logvar, max=LOGVAR_MAX)
        return mean, logvar


__all__ = ['GaussianDecoder']

