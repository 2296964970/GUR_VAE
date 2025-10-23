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
    # Defaults reflect prior argparse defaults in scripts
    return {
        # Global/data
        'data_dir': 'input',
        'case': 'case14',
        'train_csv': '',
        'normal_csv': '',
        'attacked_csv': '',
        # Windowing/common
        'time_length': 96,
        'stride': 48,
        'batch_size': 64,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        # Masking
        'mask_rate': 0.3,
        'mask_mode': 'block',  # ['iid','block','corr'] (aliases: point->iid, window->block)
        'block_t_min': 2,
        'block_t_max': 8,
        'block_f_min': 4,
        'block_f_max': 32,
        'block_max_blocks': 4,
        'corr_t': 7,
        'corr_f': 15,
        # Noise (dataset corruption on masked positions)
        'noise_kind': 'gaussian',
        'noise_sigma': 1.0,
        'noise_bias_min': -1.0,
        'noise_bias_max': 1.0,
        'noise_scale_min': 0.5,
        'noise_scale_max': 1.5,
        'noise_amp_min': 3.0,
        'noise_amp_max': 6.0,
        'mask_seed': 1337,
        # Model
        'latent_dim': 32,
        'dec_hidden': '256,256',  # comma-separated string for compatibility
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
        'device': 'cpu',  # ['cpu','cuda']
        # Tail-only inference
        'sliding_steps': 12,
        'attack_timestamp': '',  # required for tail-only; must be provided in YAML
        'alpha': 0.02,
        'min_count_per_feature': 100,
        'print_topk': False,
        'topk': 20,
        'tail_scores_wide': False,
        'tail_scores_value': 'score',  # ['score','threshold','keep_pred','is_anom']
        'print_drop_metrics': True,
        'quiet': True,
        'out_suffix': '',
        'ckpt': '',
        'noise_seed': -1,
        'stats_source': 'training',  # ['training','normal']
        # MATLAB state-estimation evaluation
        'matlab_root': '',
        'matlab_bin': 'matlab',
        'se_attacked_csv': '',
        'se_repaired_csv': '',
        'se_clean_csv': '',
        'se_out_dir': '',
        'format': 'pdf',          # ['pdf','eps']
        'angle_unit': 'rad',      # ['rad','deg']
        'reference_mode': 'eliminate',  # ['eliminate','pseudo','none']
    }


def _flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML into a flat dict matching previous CLI flags.

    YAML structure (suggested):
    - global: { data_dir, case }
    - data: { train_csv, normal_csv, attacked_csv }
    - window: { time_length, stride, batch_size, train_ratio, val_ratio }
    - mask: { mask_rate, mask_mode, block_*, corr_* }
    - noise: { kind, sigma, bias_min, bias_max, scale_min, scale_max, amp_min, amp_max, mask_seed }
    - model: { latent_dim, dec_hidden, gru_hidden, gru_layers, beta, obs_init_logvar }
    - train: { epochs, seed, learning_rate, grad_clip, exp_name, device }
    - inference_tail: { sliding_steps, attack_timestamp, alpha, min_count_per_feature, print_topk, topk, tail_scores_wide, tail_scores_value, print_drop_metrics, quiet, out_suffix, ckpt, batch_size, noise_seed, stats_source, device }
    - matlab_eval: { matlab_root, matlab_bin, attacked_csv, repaired_csv, clean_csv, out_dir, format, angle_unit, reference_mode }
    """
    d = _defaults()

    # Global
    d['data_dir'] = _deep_get(raw, 'global.data_dir', d['data_dir'])
    d['case'] = _deep_get(raw, 'global.case', d['case'])

    # Data
    d['train_csv'] = _deep_get(raw, 'data.train_csv', d['train_csv'])
    d['normal_csv'] = _deep_get(raw, 'data.normal_csv', d['normal_csv'])
    d['attacked_csv'] = _deep_get(raw, 'data.attacked_csv', d['attacked_csv'])

    # Window
    d['time_length'] = int(_deep_get(raw, 'window.time_length', d['time_length']))
    d['stride'] = int(_deep_get(raw, 'window.stride', d['stride']))
    d['batch_size'] = int(_deep_get(raw, 'window.batch_size', d['batch_size']))
    d['train_ratio'] = float(_deep_get(raw, 'window.train_ratio', d['train_ratio']))
    d['val_ratio'] = float(_deep_get(raw, 'window.val_ratio', d['val_ratio']))

    # Mask
    d['mask_rate'] = float(_deep_get(raw, 'mask.mask_rate', d['mask_rate']))
    d['mask_mode'] = str(_deep_get(raw, 'mask.mask_mode', d['mask_mode']))
    d['block_t_min'] = int(_deep_get(raw, 'mask.block_t_min', d['block_t_min']))
    d['block_t_max'] = int(_deep_get(raw, 'mask.block_t_max', d['block_t_max']))
    d['block_f_min'] = int(_deep_get(raw, 'mask.block_f_min', d['block_f_min']))
    d['block_f_max'] = int(_deep_get(raw, 'mask.block_f_max', d['block_f_max']))
    d['block_max_blocks'] = int(_deep_get(raw, 'mask.block_max_blocks', d['block_max_blocks']))
    d['corr_t'] = int(_deep_get(raw, 'mask.corr_t', d['corr_t']))
    d['corr_f'] = int(_deep_get(raw, 'mask.corr_f', d['corr_f']))

    # Noise
    d['noise_kind'] = str(_deep_get(raw, 'noise.kind', d['noise_kind']))
    d['noise_sigma'] = float(_deep_get(raw, 'noise.sigma', d['noise_sigma']))
    d['noise_bias_min'] = float(_deep_get(raw, 'noise.bias_min', d['noise_bias_min']))
    d['noise_bias_max'] = float(_deep_get(raw, 'noise.bias_max', d['noise_bias_max']))
    d['noise_scale_min'] = float(_deep_get(raw, 'noise.scale_min', d['noise_scale_min']))
    d['noise_scale_max'] = float(_deep_get(raw, 'noise.scale_max', d['noise_scale_max']))
    d['noise_amp_min'] = float(_deep_get(raw, 'noise.amp_min', d['noise_amp_min']))
    d['noise_amp_max'] = float(_deep_get(raw, 'noise.amp_max', d['noise_amp_max']))
    d['mask_seed'] = int(_deep_get(raw, 'noise.mask_seed', d['mask_seed']))

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

    # Tail-only inference
    d['sliding_steps'] = int(_deep_get(raw, 'inference_tail.sliding_steps', d['sliding_steps']))
    d['attack_timestamp'] = str(_deep_get(raw, 'inference_tail.attack_timestamp', d['attack_timestamp']))
    d['alpha'] = float(_deep_get(raw, 'inference_tail.alpha', d['alpha']))
    d['min_count_per_feature'] = int(_deep_get(raw, 'inference_tail.min_count_per_feature', d['min_count_per_feature']))
    d['print_topk'] = bool(_deep_get(raw, 'inference_tail.print_topk', d['print_topk']))
    d['topk'] = int(_deep_get(raw, 'inference_tail.topk', d['topk']))
    d['tail_scores_wide'] = bool(_deep_get(raw, 'inference_tail.tail_scores_wide', d['tail_scores_wide']))
    d['tail_scores_value'] = str(_deep_get(raw, 'inference_tail.tail_scores_value', d['tail_scores_value']))
    d['print_drop_metrics'] = bool(_deep_get(raw, 'inference_tail.print_drop_metrics', d['print_drop_metrics']))
    d['quiet'] = bool(_deep_get(raw, 'inference_tail.quiet', d['quiet']))
    d['out_suffix'] = str(_deep_get(raw, 'inference_tail.out_suffix', d['out_suffix']))
    d['ckpt'] = str(_deep_get(raw, 'inference_tail.ckpt', d['ckpt']))
    d['noise_seed'] = int(_deep_get(raw, 'inference_tail.noise_seed', d['noise_seed']))
    d['stats_source'] = str(_deep_get(raw, 'inference_tail.stats_source', d['stats_source']))
    # Allow overriding batch/device for inference separately
    inf_bs = _deep_get(raw, 'inference_tail.batch_size', None)
    if inf_bs is not None:
        d['batch_size'] = int(inf_bs)
    d_inf_dev = _deep_get(raw, 'inference_tail.device', None)
    if d_inf_dev:
        d['device'] = str(d_inf_dev)

    # MATLAB eval
    d['matlab_root'] = str(_deep_get(raw, 'matlab_eval.matlab_root', d['matlab_root']))
    d['matlab_bin'] = str(_deep_get(raw, 'matlab_eval.matlab_bin', d['matlab_bin']))
    d['se_attacked_csv'] = str(_deep_get(raw, 'matlab_eval.attacked_csv', d['se_attacked_csv']))
    d['se_repaired_csv'] = str(_deep_get(raw, 'matlab_eval.repaired_csv', d['se_repaired_csv']))
    d['se_clean_csv'] = str(_deep_get(raw, 'matlab_eval.clean_csv', d['se_clean_csv']))
    d['se_out_dir'] = str(_deep_get(raw, 'matlab_eval.out_dir', d['se_out_dir']))
    d['format'] = str(_deep_get(raw, 'matlab_eval.format', d['format']))
    d['angle_unit'] = str(_deep_get(raw, 'matlab_eval.angle_unit', d['angle_unit']))
    d['reference_mode'] = str(_deep_get(raw, 'matlab_eval.reference_mode', d['reference_mode']))

    # If ckpt is empty, derive from exp_name
    if not d.get('ckpt'):
        d['ckpt'] = os.path.join('output', d['case'], 'models', d['exp_name'], 'ckpt.pt')

    return d


def _validate_training_policy(flat: Dict[str, Any]) -> None:
    """Guardrails: prevent training leakage by disallowing fdia/2025-09 in train_csv."""
    p = str(flat.get('train_csv') or '').strip()
    if p:
        low = p.lower()
        if ('fdia' in low) or ('2025-09' in low):
            raise SystemExit('[error] Training train_csv must not contain fdia or 2025-09')


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
