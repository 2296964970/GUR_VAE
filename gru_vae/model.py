from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoders import CausalGRUEncoder
from .decoders import GaussianDecoder
from .prior_ssm import SSMPrior
from .metrics import gaussian_nll_observed
from .utils import validate_bh


class OnlineGPVAE(nn.Module):
    """Online GP-VAE with causal GRU encoder and diagonal AR(1) SSM prior.

    API:
      - init_state(B) -> state: encoder hidden, filtered (m,P), step index.
      - step(x_t, mask_t, state, use_mean=True) -> (yhat_t, state, aux)
      - elbo_sequence(x, mask, beta=None) -> dict with scalar loss and tensors.

    Shapes:
      - Inputs: x [B,T,H], mask [B,T,H]; Online step uses x_t/mask_t [B,H].
      - Latents: [B,D,T].
      - Decoder outputs: mean/logvar_x [B,T,H].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        *,
        enc_hidden_size: int = 256,
        enc_layers: int = 1,
        dec_hidden: Tuple[int, ...] = (256, 256),
        beta: float = 1.0,
        obs_init_logvar: float = -2.0,
        enc_use_mask: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or latent_dim <= 0:
            raise ValueError('input_dim, output_dim, latent_dim must be positive')
        if input_dim != output_dim:
            raise ValueError('OnlineGPVAE expects input_dim == output_dim')
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.enc_use_mask = bool(enc_use_mask)
        # Encoder consumes [x*mask, mask] if enc_use_mask, else only x
        self.enc_input_dim = (self.input_dim * 2) if self.enc_use_mask else self.input_dim
        self.encoder: nn.Module = CausalGRUEncoder(
            input_dim=self.enc_input_dim,
            z_size=self.latent_dim,
            hidden_size=enc_hidden_size,
            num_layers=enc_layers,
        )
        self.decoder = GaussianDecoder(
            output_dim=self.output_dim,
            z_size=self.latent_dim,
            hidden_sizes=dec_hidden,
            init_logvar=obs_init_logvar,
        )
        self.prior = SSMPrior(latent_dim=self.latent_dim)

    @torch.no_grad()
    def init_state(self, batch_size: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device('cpu')
        dt = dtype if dtype is not None else torch.float32
        h = self.encoder.init_hidden(batch_size, device=dev, dtype=dt)  # type: ignore[attr-defined]
        m0, P0 = self.prior.init_filter_state(batch_size, device=dev, dtype=dt)
        return {'h': h, 'm': m0, 'P': P0, 't': torch.zeros((), device=dev, dtype=torch.long)}

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def step(
        self,
        x_t: torch.Tensor,
        mask_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
        *,
        use_mean: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if x_t.ndim != 2 or mask_t.ndim != 2:
            raise ValueError('x_t and mask_t must have shape [B,H]')
        B, H = validate_bh(x_t, 'x_t')
        if H != self.input_dim:
            raise ValueError('x_t.shape[1] does not match input_dim')
        h = state['h']
        m_prev, P_prev = state['m'], state['P']
        # Prepare encoder input: gate and concat mask if enabled
        if self.enc_use_mask:
            x_eff = x_t * mask_t
            enc_in = torch.cat([x_eff, mask_t], dim=-1)
        else:
            enc_in = x_t
        mu_t, logvar_t, h_new = self.encoder.step(enc_in, h)  # type: ignore[attr-defined]
        m_pred, P_pred = self.prior.predict(m_prev, P_prev)
        kl_t = self.prior.kl_q_prior(mu_t, logvar_t, m_pred, P_pred)
        z_t = mu_t if use_mean else self._reparameterize(mu_t, logvar_t)
        z_seq = z_t.unsqueeze(-1)
        mean_1, logvar_x_1 = self.decoder(z_seq)
        mean_t = mean_1[:, 0, :]
        logvar_x_t = logvar_x_1[:, 0, :]
        nll_t = gaussian_nll_observed(mean_1, logvar_x_1, x_t.unsqueeze(1), mask_t.unsqueeze(1))
        m_filt = mu_t.detach()
        P_filt = torch.exp(logvar_t).detach()
        state_new = {'h': h_new, 'm': m_filt, 'P': P_filt, 't': state['t'] + 1}
        aux = {
            'kl_t': kl_t,
            'nll_t': nll_t,
            'mean_t': mean_t,
            'logvar_x_t': logvar_x_t,
            'mu_t': mu_t,
            'logvar_t': logvar_t,
            'z_t': z_t,
        }
        return mean_t, state_new, aux

    def elbo_sequence(self, x: torch.Tensor, mask: torch.Tensor, beta: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if x.ndim != 3 or mask.ndim != 3:
            raise ValueError('x and mask must have shape [B,T,H]')
        if x.shape != mask.shape:
            raise ValueError('x and mask must share the same shape')
        B, T, H = x.shape
        if H != self.output_dim:
            raise ValueError('x.shape[2] does not match model output_dim')
        device, dtype = x.device, x.dtype
        state = self.init_state(B, device=device, dtype=dtype)
        mean_seq = torch.empty(B, T, H, device=device, dtype=dtype)
        logvar_x_seq = torch.empty_like(mean_seq)
        mu_seq = torch.empty(B, self.latent_dim, T, device=device, dtype=dtype)
        logvar_seq = torch.empty_like(mu_seq)
        kl_acc = torch.zeros(B, device=device, dtype=dtype)
        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            _yhat_t, state, aux = self.step(x_t, m_t, state, use_mean=False)
            mean_seq[:, t, :] = aux['mean_t']
            logvar_x_seq[:, t, :] = aux['logvar_x_t']
            mu_seq[:, :, t] = aux['mu_t']
            logvar_seq[:, :, t] = aux['logvar_t']
            kl_acc += aux['kl_t']
        nll_b = gaussian_nll_observed(mean_seq, logvar_x_seq, x, mask)
        beta_val = float(self.beta if beta is None else beta)
        loss_b = nll_b + beta_val * kl_acc
        return {
            'loss': loss_b.mean(),
            'nll': nll_b.mean(),
            'kl': kl_acc.mean(),
            'mean': mean_seq,
            'logvar_x': logvar_x_seq,
            'mu': mu_seq,
            'logvar': logvar_seq,
        }

    def elbo_sequence_supervised(
        self,
        x_input: torch.Tensor,
        mask_keep: torch.Tensor,
        x_target: torch.Tensor,
        *,
        supervise: str = 'miss',
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """ELBO with decoupled input/target for self-supervised training.

        - Encoder/step consumes `x_input` with `mask_keep`.
        - NLL is computed against `x_target` using either observed (keep) or
          missing (1-keep) positions depending on `supervise`.
        """
        if not (x_input.shape == mask_keep.shape == x_target.shape):
            raise ValueError('x_input, mask_keep, x_target must share shape [B,T,H]')
        B, T, H = x_input.shape
        if H != self.output_dim:
            raise ValueError('x_input.shape[2] does not match model output_dim')
        device, dtype = x_input.device, x_input.dtype
        state = self.init_state(B, device=device, dtype=dtype)
        mean_seq = torch.empty(B, T, H, device=device, dtype=dtype)
        logvar_x_seq = torch.empty_like(mean_seq)
        mu_seq = torch.empty(B, self.latent_dim, T, device=device, dtype=dtype)
        logvar_seq = torch.empty_like(mu_seq)
        kl_acc = torch.zeros(B, device=device, dtype=dtype)
        for t in range(T):
            x_t = x_input[:, t, :]
            m_t = mask_keep[:, t, :]
            _yhat_t, state, aux = self.step(x_t, m_t, state, use_mean=False)
            mean_seq[:, t, :] = aux['mean_t']
            logvar_x_seq[:, t, :] = aux['logvar_x_t']
            mu_seq[:, :, t] = aux['mu_t']
            logvar_seq[:, :, t] = aux['logvar_t']
            kl_acc += aux['kl_t']
        if supervise not in ('obs', 'miss'):
            raise ValueError("supervise must be 'obs' or 'miss'")
        mask_loss = mask_keep if supervise == 'obs' else (mask_keep <= 0.5).to(dtype=mask_keep.dtype)
        nll_b = gaussian_nll_observed(mean_seq, logvar_x_seq, x_target, mask_loss)
        beta_val = float(self.beta if beta is None else beta)
        loss_b = nll_b + beta_val * kl_acc
        return {
            'loss': loss_b.mean(),
            'nll': nll_b.mean(),
            'kl': kl_acc.mean(),
            'mean': mean_seq,
            'logvar_x': logvar_x_seq,
            'mu': mu_seq,
            'logvar': logvar_seq,
        }

    @torch.no_grad()
    def reconstruct_online(self, x: torch.Tensor, mask: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        if x.ndim != 3 or mask.ndim != 3 or x.shape != mask.shape:
            raise ValueError('x and mask must have shape [B,T,H] and match')
        B, T, H = x.shape
        device, dtype = x.device, x.dtype
        state = self.init_state(B, device=device, dtype=dtype)
        out = torch.empty_like(x)
        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            yhat_t, state, _ = self.step(x_t, m_t, state, use_mean=use_mean)
            out[:, t, :] = torch.where(m_t > 0.5, x_t, yhat_t)
        return out


__all__ = ['OnlineGPVAE']
