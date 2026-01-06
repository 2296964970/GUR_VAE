import torch

from LGSSM_VAE.modeling import LSTM, MLPVAE, TCN


def test_tcn_step_and_reconstruct_shapes():
    batch_size, time_length, num_features = 2, 6, 4
    model = TCN(
        input_dim=num_features,
        tcn_channels=(8,),
        tcn_kernel_size=3,
        tcn_dropout=0.0,
    )
    x = torch.randn(batch_size, time_length, num_features)
    m = torch.ones(batch_size, time_length, num_features)
    out = model.training_step(x, m, x)
    assert out.mean.shape == (batch_size, time_length, num_features)
    assert out.logvar_x is None
    out.loss.backward()

    mean, logvar = model.reconstruct(x, m, return_logvar=True)
    assert mean.shape == (batch_size, time_length, num_features)
    assert logvar is None



def test_lstm_step_and_reconstruct_shapes():
    batch_size, time_length, num_features = 2, 6, 4
    model = LSTM(
        input_dim=num_features,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    x = torch.randn(batch_size, time_length, num_features)
    m = torch.ones(batch_size, time_length, num_features)
    out = model.training_step(x, m, x)
    assert out.mean.shape == (batch_size, time_length, num_features)
    assert out.logvar_x is None
    out.loss.backward()

    mean, logvar = model.reconstruct(x, m, return_logvar=True)
    assert mean.shape == (batch_size, time_length, num_features)
    assert logvar is None



def test_mlp_vae_step_and_reconstruct_shapes():
    batch_size, time_length, num_features, latent_dim = 2, 6, 4, 3
    model = MLPVAE(
        input_dim=num_features,
        time_length=time_length,
        latent_dim=latent_dim,
        hidden_sizes=(16,),
        dec_eps=1e-6,
        dec_logvar_min=-5.0,
        dec_logvar_max=2.302585092994046,
        obs_init_logvar=-2.0,
    )
    x = torch.randn(batch_size, time_length, num_features)
    m = torch.ones(batch_size, time_length, num_features)
    out = model.training_step(x, m, x, beta=0.5)
    assert out.mean.shape == (batch_size, time_length, num_features)
    assert isinstance(out.logvar_x, torch.Tensor)
    out.loss.backward()

    mean, logvar = model.reconstruct(x, m, return_logvar=True)
    assert mean.shape == (batch_size, time_length, num_features)
    assert isinstance(logvar, torch.Tensor)
