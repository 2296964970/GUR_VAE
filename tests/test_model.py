import torch
from tcn_vae.model import TCNVAE


def test_model_elbo_and_shapes():
    B, T, H, Z = 2, 6, 4, 3
    model = TCNVAE(input_dim=H, output_dim=H, latent_dim=Z, tcn_channels=(16,), tcn_kernel_size=3)
    x = torch.randn(B, T, H)
    m = torch.ones(B, T, H)
    out = model.elbo_sequence(x, m)
    for k in ('loss', 'nll', 'kl', 'mean', 'logvar_x', 'mu', 'chol'):
        assert k in out
    assert out['mean'].shape == (B, T, H)
    assert out['mu'].shape == (B, Z, T)
    assert out['chol'].shape == (B, T, Z, Z)


def test_model_reconstruct_predicts_all_positions():
    B, T, H, Z = 1, 4, 3, 2
    model = TCNVAE(input_dim=H, output_dim=H, latent_dim=Z, tcn_channels=(8,), tcn_kernel_size=3)
    x = torch.randn(B, T, H)
    m = torch.zeros(B, T, H)
    m[:, :, 0] = 1.0  # observed column
    y = model.reconstruct(x, m, use_mean=True)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.isnan(y).any()
