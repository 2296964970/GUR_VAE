import torch
from LGSSM_VAE.modeling.prior_ssm import SSMPrior


def test_ssm_prior_shapes_and_kl():
    B, D = 3, 5
    prior = SSMPrior(latent_dim=D)
    m0, P0 = prior.init_filter_state(B)
    assert m0.shape == (B, D)
    assert P0.shape == (B, D, D)
    mu_pred, P_pred = prior.predict(m0, P0)
    assert mu_pred.shape == (B, D)
    assert P_pred.shape == (B, D, D)
    mu_q = torch.zeros(B, D)
    chol_q = torch.eye(D).expand(B, D, D).clone()
    kl = prior.kl_q_prior(mu_q, chol_q, mu_pred, P_pred)
    assert kl.shape == (B,)
    assert torch.all(kl >= 0)
