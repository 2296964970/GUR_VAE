import torch
from gru_vae.encoders import CausalGRUEncoder


def test_gru_encoder_shapes():
    B, T, H, Z = 2, 5, 7, 3
    enc = CausalGRUEncoder(input_dim=H, z_size=Z, hidden_size=11, num_layers=2)
    x = torch.randn(B, T, H)
    mu, logvar = enc(x)
    assert mu.shape == (B, Z, T)
    assert logvar.shape == (B, Z, T)


def test_gru_encoder_step_and_init():
    B, H, Z = 2, 7, 3
    enc = CausalGRUEncoder(input_dim=H, z_size=Z, hidden_size=13, num_layers=1)
    h0 = enc.init_hidden(B)
    x_t = torch.randn(B, H)
    mu_t, lv_t, h1 = enc.step(x_t, h0)
    assert mu_t.shape == (B, Z)
    assert lv_t.shape == (B, Z)
    assert h1.shape == h0.shape


def test_gru_encoder_invalid_shapes():
    B, H, Z = 2, 4, 3
    enc = CausalGRUEncoder(input_dim=H, z_size=Z, hidden_size=8, num_layers=1)
    h0 = enc.init_hidden(B)
    bad_x = torch.randn(B, H + 1)
    try:
        enc.step(bad_x, h0)
        assert False, 'expected ValueError for bad input dim'
    except ValueError:
        pass
