"""Unified configuration loader for the GRU-VAE project.

This module loads a single YAML file (default: ``config.yaml`` at repo root),
merges it with sensible defaults, performs minimal validation, and exposes
attributes expected by existing scripts so we can remove argparse everywhere.

Usage
-----
from gru_vae.config import load_config
cfg = load_config()  # returns a SimpleNamespace with flat attributes

By design, the returned object provides attributes matching the previous
command-line flags used by scripts, e.g., ``cfg.case``, ``cfg.data_dir``,
``cfg.time_length``, ``cfg.latent_dim``, etc., even if the YAML is nested.

Environment override
--------------------
If the environment variable ``GRUVAE_CONFIG`` is set to a path, it will be used
instead of the default ``config.yaml`` at the project root.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required. Please install PyYAML or run `pip install -r requirements.txt`."
    ) from e


def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def _defaults() -> Dict[str, Any]:
    # Defaults for pared-down paired training/inference
    return {
        # Data roots (kept for output layout compatibility)
        'data_dir': 'input',
        'case': 'case14',
        # Explicit paired CSVs
        'train_normal_csv': '',
        'train_attacked_csv': '',
        'infer_normal_csv': '',
        'infer_attacked_csv': '',
        # Windowing/common
        'time_length': 96,
        'stride': 48,
        'batch_size': 64,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        # Model
        'latent_dim': 32,
        'dec_hidden': '256,256',
        'gru_hidden': 256,
        'gru_layers': 1,
        'beta': 0.1,
        'obs_init_logvar': -3.5,
        # Train
        'epochs': 40,
        'seed': 1337,
        'learning_rate': 3e-4,
        'grad_clip': 1e4,
        'exp_name': 'gru_base_ep40',
        'device': 'cpu',
        # Checkpoint
        'ckpt': '',
        # Inference window
        'infer_end_timestamp': '',
        'infer_length': 96,
        'infer_mc_samples': 8,
    }


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML into a flat dict for paired training.

    Suggested YAML structure:
    - global: { data_dir, case }
    - train: { normal_csv, attacked_csv, epochs, seed, learning_rate, grad_clip, exp_name, device }
    - infer: { normal_csv, attacked_csv }
    - window: { time_length, stride, batch_size, train_ratio, val_ratio }
    - model: { latent_dim, dec_hidden, gru_hidden, gru_layers, beta, obs_init_logvar }
    """
    d = _defaults()

    # Global
    d['data_dir'] = _deep_get(raw, 'global.data_dir', d['data_dir'])
    d['case'] = _deep_get(raw, 'global.case', d['case'])

    # Paired CSVs
    d['train_normal_csv'] = _deep_get(raw, 'train.normal_csv', d['train_normal_csv'])
    d['train_attacked_csv'] = _deep_get(raw, 'train.attacked_csv', d['train_attacked_csv'])
    d['infer_normal_csv'] = _deep_get(raw, 'infer.normal_csv', d['infer_normal_csv'])
    d['infer_attacked_csv'] = _deep_get(raw, 'infer.attacked_csv', d['infer_attacked_csv'])
    d['infer_end_timestamp'] = str(_deep_get(raw, 'infer.end_timestamp', d['infer_end_timestamp']))
    d['infer_length'] = int(_deep_get(raw, 'infer.length', d['infer_length']))
    d['infer_mc_samples'] = int(_deep_get(raw, 'infer.mc_samples', d['infer_mc_samples']))

    # Window
    d['time_length'] = int(_deep_get(raw, 'window.time_length', d['time_length']))
    d['stride'] = int(_deep_get(raw, 'window.stride', d['stride']))
    d['batch_size'] = int(_deep_get(raw, 'window.batch_size', d['batch_size']))
    d['train_ratio'] = float(_deep_get(raw, 'window.train_ratio', d['train_ratio']))
    d['val_ratio'] = float(_deep_get(raw, 'window.val_ratio', d['val_ratio']))

    # No masking/noise options in paired setting

    # Model
    # Allow dec_hidden as list[int] or comma string
    dec_hidden = _deep_get(raw, 'model.dec_hidden', d['dec_hidden'])
    if isinstance(dec_hidden, list):
        d['dec_hidden'] = ','.join(str(int(x)) for x in dec_hidden)
    else:
        d['dec_hidden'] = str(dec_hidden)
    d['latent_dim'] = int(_deep_get(raw, 'model.latent_dim', d['latent_dim']))
    d['gru_hidden'] = int(_deep_get(raw, 'model.gru_hidden', d['gru_hidden']))
    d['gru_layers'] = int(_deep_get(raw, 'model.gru_layers', d['gru_layers']))
    d['beta'] = float(_deep_get(raw, 'model.beta', d['beta']))
    d['obs_init_logvar'] = float(_deep_get(raw, 'model.obs_init_logvar', d['obs_init_logvar']))

    # Train
    d['epochs'] = int(_deep_get(raw, 'train.epochs', d['epochs']))
    d['seed'] = int(_deep_get(raw, 'train.seed', d['seed']))
    d['learning_rate'] = float(_deep_get(raw, 'train.learning_rate', d['learning_rate']))
    d['grad_clip'] = float(_deep_get(raw, 'train.grad_clip', d['grad_clip']))
    d['exp_name'] = str(_deep_get(raw, 'train.exp_name', d['exp_name']))
    d['device'] = str(_deep_get(raw, 'train.device', d['device']))

    # No tail-only or MATLAB sections in pared-down config
    d['ckpt'] = str(_deep_get(raw, 'train.ckpt', d['ckpt']))

    # If ckpt is empty, derive from exp_name
    if not d.get('ckpt'):
        d['ckpt'] = os.path.join('output', d['case'], 'models', d['exp_name'], 'ckpt.pt')

    return d


def _validate_training_policy(flat: Dict[str, Any]) -> None:
    """Guardrails for experimental paired training.

    - Allow train_attacked_csv to contain 'fdia' only for 2025-07_2025-08 period.
    - Forbid any '2025-09' in training CSVs.
    - Forbid 'fdia' in train_normal_csv.
    """
    n = str(flat.get('train_normal_csv') or '').strip().lower()
    a = str(flat.get('train_attacked_csv') or '').strip().lower()
    if n:
        if 'fdia' in n:
            raise SystemExit('[error] train.normal_csv must not contain fdia')
        if '2025-09' in n:
            raise SystemExit('[error] train.normal_csv must not contain 2025-09')
    if a:
        if '2025-09' in a:
            raise SystemExit('[error] train.attacked_csv must not contain 2025-09')


def load_config(path: str | None = None) -> SimpleNamespace:
    """Load YAML configuration and return a SimpleNamespace with flat attributes."""
    cfg_path = path or os.environ.get('GRUVAE_CONFIG') or 'config.yaml'
    if not os.path.exists(cfg_path):
        raise SystemExit(f"[error] Config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise SystemExit('[error] Config YAML must map to a dictionary at the top level')
    flat = _flatten_config(raw)
    _validate_training_policy(flat)
    return _to_namespace(flat)


__all__ = ['load_config']
