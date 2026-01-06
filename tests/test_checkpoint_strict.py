import torch
import pytest

from LGSSM_VAE.foundation.errors import CheckpointError
from LGSSM_VAE.pipeline import build_model_from_checkpoint


def test_build_model_from_checkpoint_requires_metadata(tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": {}}, ckpt_path)

    with pytest.raises(CheckpointError) as exc:
        build_model_from_checkpoint(str(ckpt_path), input_dim=4)
    assert "检查点缺少必要字段" in str(exc.value)


def test_build_model_from_checkpoint_rejects_string_sizes(tmp_path):
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt = {
        "model": {},
        "model_name": "LGSSM-VAE",
        "model_hparams": {
            "latent_dim": 4,
            "tcn_channels": "256,256,256",
            "tcn_kernel_size": 3,
            "tcn_dropout": 0.0,
            "dec_hidden": [8],
            "enc_diag_eps": 1e-4,
            "dec_eps": 1e-6,
            "dec_logvar_min": -5.0,
            "dec_logvar_max": 2.302585092994046,
            "prior_rank": 1,
            "prior_a_init": 0.95,
            "prior_q_init": 0.1,
            "prior_m0_init": 0.0,
            "prior_P0_init": 1.0,
            "prior_jitter": 1e-6,
            "prior_variance_floor": 1e-6,
        },
    }
    torch.save(ckpt, ckpt_path)

    with pytest.raises(CheckpointError) as exc:
        build_model_from_checkpoint(str(ckpt_path), input_dim=4)
    assert "tcn_channels" in str(exc.value)
