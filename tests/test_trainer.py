import torch
from torch.utils.data import DataLoader, TensorDataset
from LGSSM_VAE.modeling import LGSSMVAE
from LGSSM_VAE.pipeline import OnlineTrainer


def test_trainer_single_epoch_smoke():
    B, T, H, Z = 8, 6, 4, 3
    # simple synthetic dataset: x==0, mask==1
    x = torch.zeros(B, T, H)
    m = torch.ones(B, T, H)
    ds = TensorDataset(x, m)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = LGSSMVAE(input_dim=H, output_dim=H, latent_dim=Z, tcn_channels=(16,), tcn_kernel_size=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optim, device=torch.device('cpu'))

    stats = trainer.train_epoch(loader)
    assert isinstance(stats.loss, float)
    assert isinstance(stats.nll, float)
    assert isinstance(stats.kl, float)
    assert isinstance(stats.kl_used, float)
