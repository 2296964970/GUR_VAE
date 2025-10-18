import torch
from gru_vae.model import OnlineGPVAE


def test_model_elbo_and_shapes():
    B, T, H, Z = 2, 6, 4, 3
    model = OnlineGPVAE(input_dim=H, output_dim=H, latent_dim=Z, enc_hidden_size=16)
    x = torch.randn(B, T, H)
    m = torch.ones(B, T, H)
    out = model.elbo_sequence(x, m)
    for k in ('loss', 'nll', 'kl', 'mean', 'logvar_x', 'mu', 'logvar'):
        assert k in out
    assert out['mean'].shape == (B, T, H)
    assert out['mu'].shape == (B, Z, T)


def test_model_reconstruct_online_passes_observed():
    B, T, H, Z = 1, 4, 3, 2
    model = OnlineGPVAE(input_dim=H, output_dim=H, latent_dim=Z, enc_hidden_size=8)
    x = torch.randn(B, T, H)
    m = torch.zeros(B, T, H)
    m[:, :, 0] = 1.0
    y = model.reconstruct_online(x, m, use_mean=True)
    assert torch.allclose(y[:, :, 0], x[:, :, 0])
