"""Unified configuration loader for the TCN-VAE project.

This module loads a single YAML file (default: ``config.yaml`` at repo root),
merges it with sensible defaults, performs minimal validation, and exposes
attributes expected by existing scripts so we can remove argparse everywhere.

Usage
-----
from tcn_vae.config import load_config
cfg = load_config()  # returns a SimpleNamespace with flat attributes

By design, the returned object provides attributes matching the previous
command-line flags used by scripts, e.g., ``cfg.case``, ``cfg.data_dir``,
``cfg.time_length``, ``cfg.latent_dim``, etc., even if the YAML is nested.

Environment override
--------------------
If the environment variable ``TCNVAE_CONFIG`` is set to a path, it will be used
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
        # Preprocess / standardization
        'clip_k': 0.0,
        'std_floor': 1e-3,
        # Explicit CSVs
        'train_normal_csv': '',
        'infer_normal_csv': '',
        'infer_attacked_csv': '',
        # Windowing/common
        'time_length': 96,
        'stride': 48,
        'batch_size': 64,
        'num_workers': 0,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        # Model
        'latent_dim': 32,
        'dec_hidden': '256,256',
        'tcn_channels': '256,256,256',
        'tcn_kernel_size': 3,
        'tcn_dropout': 0.0,
        # Encoder/decoder fine controls
        'enc_diag_eps': 1e-4,
        'dec_eps': 1e-6,
        'dec_logvar_min': -5.0,
        'dec_logvar_max': 2.302585092994046,
        # Prior controls
        'prior_rank': 4,
        'prior_a_init': 0.95,
        'prior_q_init': 0.1,
        'prior_m0_init': 0.0,
        'prior_P0_init': 1.0,
        'prior_jitter': 1e-6,
        'prior_variance_floor': 1e-6,
        'beta': 0.1,
        'obs_init_logvar': -3.5,
        # Train
        'epochs': 40,
        # Two-phase schedule
        'phase1_epochs': 10,
        'phase2_epochs': 30,
        # Robust training controls
        'anchor_lambda': 1.0,
        'clean_fraction': 0.4,
        'seg_len_min': 8,
        'seg_len_max': 96,
        'dims_fraction_min': 0.05,
        'dims_fraction_max': 0.3,
        'robust_laplace_scales': '0.1,0.3,0.7,1.2',
        'seed': 1337,
        'learning_rate': 3e-4,
        'grad_clip': 1e4,
        'warmup_frac': 0.2,
        'model_dir': '',
        'device': 'cpu',
        # Checkpoint
        'ckpt': '',
        # Inference blend controls
        'blend_k_sigma': 1.0,
        'blend_softness': 0.5,
    }


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML into a flat dict for paired training.

    Suggested YAML structure:
    - global: { data_dir, case }
    - train: { normal_csv, attacked_csv, epochs, seed, learning_rate, grad_clip, exp_name, device }
    - infer: { normal_csv, attacked_csv }
    - window: { time_length, stride, batch_size, train_ratio, val_ratio }
    - model: { latent_dim, dec_hidden, tcn_channels, tcn_kernel_size, tcn_dropout, beta, obs_init_logvar }
    """
    d = _defaults()

    # Global
    d['data_dir'] = _deep_get(raw, 'global.data_dir', d['data_dir'])
    d['case'] = _deep_get(raw, 'global.case', d['case'])
    # Single model directory (Plan A). Default to output/<case>/models
    d['model_dir'] = str(_deep_get(raw, 'train.model_dir', os.path.join('output', d['case'], 'models')))

    # CSV paths
    d['train_normal_csv'] = _deep_get(raw, 'train.normal_csv', d['train_normal_csv'])
    d['infer_normal_csv'] = _deep_get(raw, 'infer.normal_csv', d['infer_normal_csv'])
    d['infer_attacked_csv'] = _deep_get(raw, 'infer.attacked_csv', d['infer_attacked_csv'])
    # No window/MC flattening for inference in simplified pipeline

    # Preprocess
    d['clip_k'] = float(_deep_get(raw, 'preprocess.clip_k', d['clip_k']))
    d['std_floor'] = float(_deep_get(raw, 'preprocess.std_floor', d['std_floor']))

    # Window
    d['time_length'] = int(_deep_get(raw, 'window.time_length', d['time_length']))
    d['stride'] = int(_deep_get(raw, 'window.stride', d['stride']))
    d['batch_size'] = int(_deep_get(raw, 'window.batch_size', d['batch_size']))
    d['num_workers'] = int(_deep_get(raw, 'window.num_workers', d['num_workers']))
    d['train_ratio'] = float(_deep_get(raw, 'window.train_ratio', d['train_ratio']))
    d['val_ratio'] = float(_deep_get(raw, 'window.val_ratio', d['val_ratio']))

    # Two-phase schedule and robust controls
    d['phase1_epochs'] = int(_deep_get(raw, 'train.phase1_epochs', d['phase1_epochs']))
    d['phase2_epochs'] = int(_deep_get(raw, 'train.phase2_epochs', d['phase2_epochs']))
    d['anchor_lambda'] = float(_deep_get(raw, 'train.anchor_lambda', d['anchor_lambda']))
    d['clean_fraction'] = float(_deep_get(raw, 'robust.clean_fraction', d['clean_fraction']))
    # Segment length
    seg_len = _deep_get(raw, 'robust.seg_len', None)
    if isinstance(seg_len, (list, tuple)) and len(seg_len) >= 2:
        d['seg_len_min'] = int(seg_len[0])
        d['seg_len_max'] = int(seg_len[1])
    else:
        d['seg_len_min'] = int(_deep_get(raw, 'robust.seg_len_min', d['seg_len_min']))
        d['seg_len_max'] = int(_deep_get(raw, 'robust.seg_len_max', d['seg_len_max']))
    # Dims fraction
    dims_frac = _deep_get(raw, 'robust.dims_fraction', None)
    if isinstance(dims_frac, (list, tuple)) and len(dims_frac) >= 2:
        d['dims_fraction_min'] = float(dims_frac[0])
        d['dims_fraction_max'] = float(dims_frac[1])
    else:
        d['dims_fraction_min'] = float(_deep_get(raw, 'robust.dims_fraction_min', d['dims_fraction_min']))
        d['dims_fraction_max'] = float(_deep_get(raw, 'robust.dims_fraction_max', d['dims_fraction_max']))
    # Laplace scales
    lap_scales = _deep_get(raw, 'robust.laplace_scales', d['robust_laplace_scales'])
    if isinstance(lap_scales, list):
        d['robust_laplace_scales'] = ','.join(str(float(x)) for x in lap_scales)
    elif lap_scales in (None, ''):
        d['robust_laplace_scales'] = ''
    else:
        d['robust_laplace_scales'] = str(lap_scales)

    # Model
    # Allow dec_hidden as list[int] or comma string
    dec_hidden = _deep_get(raw, 'model.dec_hidden', d['dec_hidden'])
    if isinstance(dec_hidden, list):
        d['dec_hidden'] = ','.join(str(int(x)) for x in dec_hidden)
    else:
        d['dec_hidden'] = str(dec_hidden)
    d['latent_dim'] = int(_deep_get(raw, 'model.latent_dim', d['latent_dim']))
    # TCN params
    tcn_channels = _deep_get(raw, 'model.tcn_channels', d['tcn_channels'])
    if isinstance(tcn_channels, list):
        d['tcn_channels'] = ','.join(str(int(x)) for x in tcn_channels)
    else:
        d['tcn_channels'] = str(tcn_channels)
    d['tcn_kernel_size'] = int(_deep_get(raw, 'model.tcn_kernel_size', d['tcn_kernel_size']))
    d['tcn_dropout'] = float(_deep_get(raw, 'model.tcn_dropout', d['tcn_dropout']))
    d['beta'] = float(_deep_get(raw, 'model.beta', d['beta']))
    d['obs_init_logvar'] = float(_deep_get(raw, 'model.obs_init_logvar', d['obs_init_logvar']))
    # Encoder/decoder fine controls
    d['enc_diag_eps'] = float(_deep_get(raw, 'model.encoder.diag_eps', d['enc_diag_eps']))
    d['dec_eps'] = float(_deep_get(raw, 'model.decoder.eps', d['dec_eps']))
    d['dec_logvar_min'] = float(_deep_get(raw, 'model.decoder.logvar_min', d['dec_logvar_min']))
    d['dec_logvar_max'] = float(_deep_get(raw, 'model.decoder.logvar_max', d['dec_logvar_max']))
    # Prior controls
    d['prior_rank'] = int(_deep_get(raw, 'model.prior.rank', d['prior_rank']))
    d['prior_a_init'] = float(_deep_get(raw, 'model.prior.a_init', d['prior_a_init']))
    d['prior_q_init'] = float(_deep_get(raw, 'model.prior.q_init', d['prior_q_init']))
    d['prior_m0_init'] = float(_deep_get(raw, 'model.prior.m0_init', d['prior_m0_init']))
    d['prior_P0_init'] = float(_deep_get(raw, 'model.prior.P0_init', d['prior_P0_init']))
    d['prior_jitter'] = float(_deep_get(raw, 'model.prior.jitter', d['prior_jitter']))
    d['prior_variance_floor'] = float(_deep_get(raw, 'model.prior.variance_floor', d['prior_variance_floor']))

    # Train
    d['epochs'] = int(_deep_get(raw, 'train.epochs', d['epochs']))
    d['seed'] = int(_deep_get(raw, 'train.seed', d['seed']))
    d['learning_rate'] = float(_deep_get(raw, 'train.learning_rate', d['learning_rate']))
    d['grad_clip'] = float(_deep_get(raw, 'train.grad_clip', d['grad_clip']))
    d['device'] = str(_deep_get(raw, 'train.device', d['device']))
    d['warmup_frac'] = float(_deep_get(raw, 'train.warmup_frac', d['warmup_frac']))

    # Effective checkpoint resolution (Plan A)
    # - Train ckpt: derived from model_dir/ckpt.pt
    # - Infer ckpt: infer.ckpt; if empty, fall back to train ckpt
    train_ckpt = os.path.join(d['model_dir'], 'ckpt.pt')
    infer_ckpt = str(_deep_get(raw, 'infer.ckpt', ''))
    d['ckpt'] = infer_ckpt if infer_ckpt else train_ckpt
    # Inference blending controls
    d['blend_k_sigma'] = float(_deep_get(raw, 'infer.blend_k_sigma', d['blend_k_sigma']))
    d['blend_softness'] = float(_deep_get(raw, 'infer.blend_softness', d['blend_softness']))

    return d


def _validate_training_policy(flat: Dict[str, Any]) -> None:
    """Guardrails for experimental paired training.

    - Forbid any '2025-09' in training CSVs.
    - Forbid 'fdia' in train_normal_csv.
    """
    n = str(flat.get('train_normal_csv') or '').strip().lower()
    if n:
        if 'fdia' in n:
            raise SystemExit('[error] train.normal_csv must not contain fdia')
        if '2025-09' in n:
            raise SystemExit('[error] train.normal_csv must not contain 2025-09')


def load_config(path: str | None = None) -> SimpleNamespace:
    """Load YAML configuration and return a SimpleNamespace with flat attributes."""
    cfg_path = path or os.environ.get('TCNVAE_CONFIG') or 'config.yaml'
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
