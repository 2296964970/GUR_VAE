from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TCNBlock
from LGSSM_VAE.foundation.interfaces import StepOutput
from LGSSM_VAE.foundation.metrics import gaussian_nll_observed, mse_observed
from LGSSM_VAE.foundation.utils import softplus_inverse, validate_bth


def _make_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim/output_dim 必须为正数")
    layers: list[nn.Module] = []
    in_features = int(input_dim)
    for hidden in hidden_sizes:
        layers.append(nn.Linear(in_features, int(hidden)))
        layers.append(nn.ReLU(inplace=True))
        in_features = int(hidden)
    layers.append(nn.Linear(in_features, int(output_dim)))
    return nn.Sequential(*layers)


def _mask_aware_input(x: torch.Tensor, mask: torch.Tensor, *, input_dim: int) -> torch.Tensor:
    validate_bth(x, "x")
    if x.shape != mask.shape:
        raise ValueError("x 与 mask 形状必须一致 [B,T,H]")
    if x.shape[2] != int(input_dim):
        raise ValueError("x.shape[2] 与 input_dim 不一致")
    return torch.cat([x * mask, mask], dim=-1)


class TCN(nn.Module):
    """确定性 TCN baseline（直接映射、无隐变量采样）。

    - 输入：mask-aware 的 concat([x*mask, mask])
    - 输出：mean[B,T,H]
    - Loss：observed-only MSE
    """

    def __init__(
        self,
        *,
        input_dim: int,
        tcn_channels: Sequence[int],
        tcn_kernel_size: int,
        tcn_dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正数")
        self.input_dim = int(input_dim)
        self.model_input_dim = self.input_dim * 2

        channels = tuple(int(c) for c in tcn_channels)
        if not channels:
            raise ValueError("tcn_channels 不能为空")

        layers: list[nn.Module] = []
        channels_in = self.model_input_dim
        dilation = 1
        for channels_out in channels:
            layers.append(
                TCNBlock(
                    channels_in,
                    int(channels_out),
                    kernel_size=int(tcn_kernel_size),
                    dilation=dilation,
                    dropout=float(tcn_dropout),
                )
            )
            channels_in = int(channels_out)
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Conv1d(channels_in, self.input_dim, kernel_size=1)

    def _forward_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_in = _mask_aware_input(x, mask, input_dim=self.input_dim)
        x_bct = x_in.transpose(1, 2).contiguous()
        h = self.tcn(x_bct)
        return self.proj(h).transpose(1, 2).contiguous()

    def training_step(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> StepOutput:
        mean = self._forward_mean(x_input, mask_keep)
        recon_b = mse_observed(mean, x_target, mask_keep)
        recon = recon_b.mean()
        zero = torch.zeros((), device=recon.device, dtype=recon.dtype)
        return StepOutput(
            loss=recon,
            recon=recon,
            kl=zero,
            kl_used=zero,
            mean=mean,
            logvar_x=None,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_mean: bool = True,
        *,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean = self._forward_mean(x, mask)
        return (mean, None) if return_logvar else mean


class LSTM(nn.Module):
    """确定性 LSTM baseline（直接映射、无隐变量采样）。

    - 输入：mask-aware 的 concat([x*mask, mask])
    - 输出：mean[B,T,H]
    - Loss：observed-only MSE
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0:
            raise ValueError("input_dim/hidden_size 必须为正数")
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正数")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.model_input_dim = self.input_dim * 2

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=self.model_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=float(lstm_dropout),
            batch_first=True,
        )
        self.proj = nn.Linear(self.hidden_size, self.input_dim)

    def _forward_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_in = _mask_aware_input(x, mask, input_dim=self.input_dim)
        h, _ = self.rnn(x_in)
        mean = self.proj(h.reshape(-1, self.hidden_size)).view_as(x)
        return mean

    def training_step(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> StepOutput:
        mean = self._forward_mean(x_input, mask_keep)
        recon_b = mse_observed(mean, x_target, mask_keep)
        recon = recon_b.mean()
        zero = torch.zeros((), device=recon.device, dtype=recon.dtype)
        return StepOutput(
            loss=recon,
            recon=recon,
            kl=zero,
            kl_used=zero,
            mean=mean,
            logvar_x=None,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_mean: bool = True,
        *,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean = self._forward_mean(x, mask)
        return (mean, None) if return_logvar else mean


class MLPVAE(nn.Module):
    """Window-level MLP-VAE（probabilistic）。

    - Encoder: MLP（展平窗口输入，mask-aware）
    - Latent: window-level diagonal Gaussian q(z|x)
    - Decoder: MLP 输出整段窗口的 mean/logvar
    - Loss: observed-only Gaussian NLL + beta * KL
    """

    def __init__(
        self,
        *,
        input_dim: int,
        time_length: int,
        latent_dim: int,
        hidden_sizes: Sequence[int],
        dec_eps: float,
        dec_logvar_min: float,
        dec_logvar_max: float,
        obs_init_logvar: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or time_length <= 0 or latent_dim <= 0:
            raise ValueError("input_dim/time_length/latent_dim 必须为正数")
        self.input_dim = int(input_dim)
        self.time_length = int(time_length)
        self.latent_dim = int(latent_dim)
        self.enc_input_dim = self.input_dim * 2
        self.flat_in = self.time_length * self.enc_input_dim
        self.flat_out = self.time_length * self.input_dim
        self.dec_eps = float(dec_eps)
        self.dec_logvar_min = float(dec_logvar_min)
        self.dec_logvar_max = float(dec_logvar_max)

        hidden = tuple(int(h) for h in hidden_sizes)
        self.encoder = _make_mlp(self.flat_in, hidden, 2 * self.latent_dim)

        self.decoder_net = _make_mlp(self.latent_dim, hidden, 2 * self.flat_out)
        out = self.decoder_net[-1]
        with torch.no_grad():
            target_var = float(torch.exp(torch.tensor(float(obs_init_logvar))).item())
            raw_init = softplus_inverse(target_var, eps=self.dec_eps)
            out.bias[self.flat_out :].fill_(raw_init)

    def _encode_window(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_in = _mask_aware_input(x, mask, input_dim=self.input_dim)  # [B,T,2H]
        if enc_in.shape[1] != self.time_length:
            raise ValueError("MLPVAE 仅支持固定 time_length 的窗口输入")

        flat = enc_in.reshape(enc_in.shape[0], -1)
        params = self.encoder(flat)
        mu, logvar = torch.split(params, self.latent_dim, dim=-1)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode_window(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.decoder_net(z)
        mean_raw, raw = torch.split(y, self.flat_out, dim=-1)
        mean = mean_raw.view(z.shape[0], self.time_length, self.input_dim)
        logvar = torch.log(F.softplus(raw) + self.dec_eps)
        logvar = torch.clamp(logvar, min=self.dec_logvar_min, max=self.dec_logvar_max)
        logvar = logvar.view_as(mean)
        return mean, logvar

    def training_step(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> StepOutput:
        mu_z, logvar_z = self._encode_window(x_input, mask_keep)
        z = self._reparameterize(mu_z, logvar_z)
        mean, logvar_x = self._decode_window(z)

        nll_b = gaussian_nll_observed(mean, logvar_x, x_target, mask_keep)
        nll = nll_b.mean()

        kl_elem = 0.5 * (torch.exp(logvar_z) + mu_z * mu_z - 1.0 - logvar_z)
        kl_sum_b = kl_elem.sum(dim=-1)
        kl = kl_sum_b.mean()
        kl_used = (kl_sum_b / self.latent_dim).mean()

        beta_val = float(1.0 if beta is None else beta)
        loss = nll + beta_val * kl_used
        return StepOutput(
            loss=loss,
            recon=nll,
            kl=kl,
            kl_used=kl_used,
            mean=mean,
            logvar_x=logvar_x,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_mean: bool = True,
        *,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        mu_z, logvar_z = self._encode_window(x, mask)
        z = mu_z if use_mean else self._reparameterize(mu_z, logvar_z)
        mean, logvar_x = self._decode_window(z)
        return (mean, logvar_x) if return_logvar else mean


__all__ = ["TCN", "LSTM", "MLPVAE"]

