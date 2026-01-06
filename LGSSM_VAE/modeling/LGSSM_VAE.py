from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import CausalTCNEncoder
from .decoders import GaussianDecoder
from .prior_ssm import SSMPrior
from LGSSM_VAE.foundation.metrics import gaussian_nll_observed
from LGSSM_VAE.foundation.interfaces import StepOutput
from LGSSM_VAE.foundation.utils import validate_bth


class LGSSMVAE(nn.Module):
    """GP-VAE with causal TCN encoder and sparse low-rank VAR(1) prior.

    API:
      - elbo_sequence(x, mask, beta=None) -> dict with scalar loss and tensors.
      - elbo_sequence_supervised(x_input, mask_keep, x_target) -> dict.
      - reconstruct(x, mask, use_mean=True) -> [B,T,H] predictions.

    Shapes:
      - Inputs: x [B,T,H], mask [B,T,H].
      - Latents: [B,D,T].
      - Decoder outputs: mean/logvar_x [B,T,H].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        *,
        tcn_channels: Tuple[int, ...] = (256, 256, 256),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.0,
        dec_hidden: Tuple[int, ...] = (256, 256),
        beta: float = 1.0,
        obs_init_logvar: float = -2.0,
        # Encoder/decoder fine controls
        enc_diag_eps: float = 1e-4,
        dec_eps: float = 1e-6,
        dec_logvar_min: float = -5.0,
        dec_logvar_max: float = 2.302585092994046,
        # Prior controls
        prior_rank: int = 4,
        prior_a_init: float = 0.95,
        prior_q_init: float = 0.1,
        prior_m0_init: float = 0.0,
        prior_P0_init: float = 1.0,
        prior_jitter: float = 1e-6,
        prior_variance_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or latent_dim <= 0:
            raise ValueError('input_dim, output_dim, latent_dim must be positive')
        if input_dim != output_dim:
            raise ValueError('LGSSMVAE expects input_dim == output_dim')
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        # Mask-aware encoder input: concat [x*mask, mask]
        self.enc_input_dim = self.input_dim * 2
        self.encoder: nn.Module = CausalTCNEncoder(
            input_dim=self.enc_input_dim,
            z_size=self.latent_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            diag_eps=float(enc_diag_eps),
        )
        self.decoder = GaussianDecoder(
            output_dim=self.output_dim,
            z_size=self.latent_dim,
            hidden_sizes=dec_hidden,
            init_logvar=obs_init_logvar,
            eps=float(dec_eps),
            logvar_min=float(dec_logvar_min),
            logvar_max=float(dec_logvar_max),
        )
        self.prior = SSMPrior(
            latent_dim=self.latent_dim,
            a_init=float(prior_a_init),
            q_init=float(prior_q_init),
            m0_init=float(prior_m0_init),
            P0_init=float(prior_P0_init),
            rank=int(prior_rank),
        )
        # Override numerical floors if provided
        self.prior.jitter = float(prior_jitter)
        self.prior.variance_floor = float(prior_variance_floor)


    def _validate_x_mask(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int, int]:
        B, T, H = validate_bth(x, "x")
        if x.shape != mask.shape:
            raise ValueError("x and mask must have shape [B,T,H] and match")
        if T <= 0:
            raise ValueError("x.shape[1] (T) must be positive")
        if H != self.output_dim:
            raise ValueError("x.shape[2] does not match model output_dim")
        return int(B), int(T), int(H)

    @staticmethod
    def _validate_mu_chol(mu: torch.Tensor, chol: torch.Tensor) -> Tuple[int, int, int]:
        if mu.ndim != 3 or chol.ndim != 4:
            raise ValueError("mu must be [B,D,T] and chol must be [B,T,D,D]")
        B, D, T = mu.shape
        if chol.shape != (B, T, D, D):
            raise ValueError("chol must have shape [B,T,D,D] matching mu")
        return int(B), int(D), int(T)

    def _reparameterize(self, mu: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
        B, D, T = self._validate_mu_chol(mu, chol)
        eps = torch.randn(B, T, D, device=mu.device, dtype=mu.dtype)
        z_t = mu.transpose(1, 2) + torch.matmul(chol, eps.unsqueeze(-1)).squeeze(-1)
        return z_t.transpose(1, 2)

    def _kl_over_time(self, mu: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
        """Accumulate KL_t against the LGSSM prior.

        - t=0 uses the initial prior N(m0, P0)
        - t>=1 uses the one-step predicted prior from the previous posterior

        mu: [B,D,T], chol: [B,T,D,D]
        Returns: kl_sum [B]
        """
        B, D, T = self._validate_mu_chol(mu, chol)
        device, dtype = mu.device, mu.dtype
        A, Q, m0, P0 = self.prior.transition_matrices(device=device, dtype=dtype)
        jitter_eye = self.prior.jitter * torch.eye(D, device=device, dtype=dtype)

        kl_acc = torch.zeros(B, device=device, dtype=dtype)

        m_prior0 = m0.unsqueeze(0).expand(B, -1).contiguous()
        P_prior0 = P0.unsqueeze(0).expand(B, -1, -1).contiguous()

        mu_0 = mu[:, :, 0]
        chol_0 = chol[:, 0, :, :]
        kl_acc += self.prior.kl_q_prior(mu_0, chol_0, m_prior0, P_prior0)

        m_prev = mu_0.detach()
        Sigma_0 = torch.matmul(chol_0, chol_0.transpose(-1, -2)) + jitter_eye
        P_prev = Sigma_0.detach()

        for t in range(1, T):
            m_pred, P_pred = self.prior.predict(m_prev, P_prev, A=A, Q=Q)
            mu_t = mu[:, :, t]
            chol_t = chol[:, t, :, :]
            kl_acc += self.prior.kl_q_prior(mu_t, chol_t, m_pred, P_pred)

            m_prev = mu_t.detach()
            Sigma_t = torch.matmul(chol_t, chol_t.transpose(-1, -2)) + jitter_eye
            P_prev = Sigma_t.detach()

        return kl_acc

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_eff = x * mask
        enc_in = torch.cat([x_eff, mask], dim=-1)
        mu, chol = self.encoder(enc_in)
        return mu, chol

    def elbo_sequence(self, x: torch.Tensor, mask: torch.Tensor, beta: Optional[float] = None) -> Dict[str, torch.Tensor]:
        _, T, _ = self._validate_x_mask(x, mask)
        mu, chol = self._encode(x, mask)  # mu [B,D,T], chol [B,T,D,D]
        z = self._reparameterize(mu, chol)
        mean_seq, logvar_x_seq = self.decoder(z)
        nll_b = gaussian_nll_observed(mean_seq, logvar_x_seq, x, mask)
        kl_sum_b = self._kl_over_time(mu, chol)
        # Use per-step average KL in the objective to keep ELBO scale stable across time_length.
        kl_used_b = kl_sum_b / T
        beta_val = float(self.beta if beta is None else beta)
        loss_b = nll_b + beta_val * kl_used_b
        return {
            "loss": loss_b.mean(),
            "nll": nll_b.mean(),
            "kl": kl_sum_b.mean(),
            # KL actually used in the objective (mean per timestep).
            "kl_used": kl_used_b.mean(),
            "mean": mean_seq,
            "logvar_x": logvar_x_seq,
            "mu": mu,
            "chol": chol,
        }

    def elbo_sequence_supervised(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        _, T, _ = self._validate_x_mask(x_input, mask_keep)
        if x_target.ndim != 3 or x_target.shape != x_input.shape:
            raise ValueError("x_input, mask_keep, x_target must share shape [B,T,H]")

        mu, chol = self._encode(x_input, mask_keep)
        z = self._reparameterize(mu, chol)
        mean_seq, logvar_x_seq = self.decoder(z)
        nll_b = gaussian_nll_observed(mean_seq, logvar_x_seq, x_target, mask_keep)
        kl_sum_b = self._kl_over_time(mu, chol)
        # Use per-step average KL in the objective to keep ELBO scale stable across time_length.
        kl_used_b = kl_sum_b / T
        beta_val = float(self.beta if beta is None else beta)
        loss_b = nll_b + beta_val * kl_used_b
        return {
            "loss": loss_b.mean(),
            "nll": nll_b.mean(),
            "kl": kl_sum_b.mean(),
            # KL actually used in the objective (mean per timestep).
            "kl_used": kl_used_b.mean(),
            "mean": mean_seq,
            "logvar_x": logvar_x_seq,
            "mu": mu,
            "chol": chol,
        }

    def training_step(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> StepOutput:
        """统一训练接口：返回 StepOutput，供 trainer 做模型无关训练。"""
        out = self.elbo_sequence_supervised(x_input, mask_keep, x_target, beta=beta)
        return StepOutput(
            loss=out["loss"],
            recon=out["nll"],
            kl=out["kl"],
            kl_used=out["kl_used"],
            mean=out["mean"],
            logvar_x=out["logvar_x"],
        )

    def reconstruct(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_mean: bool = True,
        *,
        return_logvar: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_x_mask(x, mask)
        mu, chol = self._encode(x, mask)
        z = mu if use_mean else self._reparameterize(mu, chol)
        mean_seq, logvar_x = self.decoder(z)
        return (mean_seq, logvar_x) if return_logvar else mean_seq


__all__ = ["LGSSMVAE"]
