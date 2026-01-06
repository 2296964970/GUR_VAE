import torch
from LGSSM_VAE.modeling.encoders import CausalTCNEncoder


def test_tcn_encoder_shapes():
    B, T, H, Z = 2, 5, 7, 3
    enc = CausalTCNEncoder(input_dim=H, z_size=Z, channels=(11, 13), kernel_size=3, dropout=0.0)
    x = torch.randn(B, T, H)
    mu, chol = enc(x)
    assert mu.shape == (B, Z, T)
    assert chol.shape == (B, T, Z, Z)
    # Cholesky factors must be lower-triangular with positive diagonal
    for t in range(T):
        L_t = chol[:, t]
        assert torch.allclose(L_t, torch.tril(L_t), atol=1e-6)
        assert torch.all(torch.diagonal(L_t, dim1=-2, dim2=-1) > 0)


def test_tcn_encoder_causality():
    # Construct two inputs that share the same prefix, different suffix.
    B, T, H, Z = 1, 16, 4, 2
    enc = CausalTCNEncoder(input_dim=H, z_size=Z, channels=(8, 8), kernel_size=3, dropout=0.0)
    enc.eval()
    x1 = torch.randn(B, T, H)
    x2 = x1.clone()
    # Make suffix (after t0) different
    t0 = T // 2
    x2[:, t0:, :] = torch.randn(B, T - t0, H)
    mu1, _ = enc(x1)
    mu2, _ = enc(x2)
    # Outputs up to and including t0-1 must be identical
    assert torch.allclose(mu1[:, :, :t0], mu2[:, :, :t0], atol=1e-6, rtol=0.0)
