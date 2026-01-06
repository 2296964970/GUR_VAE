import os

import yaml
import pytest

from LGSSM_VAE.config import load_config
from LGSSM_VAE.foundation.errors import ConfigError


def _load_case14_template() -> dict:
    with open("configs/case14/lgssm_vae.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_load_config_accepts_repo_templates():
    cfg = load_config("configs/case14/lgssm_vae.yaml")
    assert cfg.case == "case14"
    assert isinstance(cfg.model.dec_hidden, tuple) and len(cfg.model.dec_hidden) > 0
    assert isinstance(cfg.model.tcn_channels, tuple) and len(cfg.model.tcn_channels) > 0
    assert isinstance(cfg.robust.laplace_scales, tuple) and len(cfg.robust.laplace_scales) > 0


def test_load_config_rejects_string_sizes(tmp_path):
    raw = _load_case14_template()
    raw["model"]["dec_hidden"] = "256,256"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(str(p))
    assert "model.dec_hidden" in str(exc.value)


def test_load_config_rejects_missing_required_key(tmp_path):
    raw = _load_case14_template()
    del raw["global"]["case"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(str(p))
    assert "global.case" in str(exc.value)


def test_load_config_supports_extends_and_overrides(tmp_path):
    base_dir = tmp_path / "configs" / "case14"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / "lgssm_vae.yaml"
    base_path.write_text(yaml.safe_dump(_load_case14_template(), sort_keys=False), encoding="utf-8")

    override = {
        "extends": "configs/case14/lgssm_vae.yaml",
        "train": {"device": "cpu"},
        "model": {"tcn_channels": [16, 16, 16]},
    }
    override_path = tmp_path / "config.yaml"
    override_path.write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")

    cfg = load_config(str(override_path))
    assert cfg.case == "case14"
    assert cfg.train.device == "cpu"
    assert cfg.model.latent_dim == 64  # from base
    assert cfg.model.tcn_channels == (16, 16, 16)


def test_load_config_extends_rejects_unknown_keys(tmp_path):
    override = {
        "extends": os.path.abspath("configs/case14/lgssm_vae.yaml"),
        "train": {"learing_rate": 0.1},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(str(p))
    assert "Unknown config key(s) at train" in str(exc.value)


def test_load_config_extends_cycle_detected(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"extends": "b.yaml"}, sort_keys=False), encoding="utf-8")
    b.write_text(yaml.safe_dump({"extends": "a.yaml"}, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_config(str(a))
    assert "extends cycle detected" in str(exc.value)
