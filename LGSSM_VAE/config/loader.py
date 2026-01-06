"""Configuration loader for the LGSSM-VAE project.

Features
- Loads YAML (default: repo root `config.yaml`).
- Supports composition via a top-level `extends` key (deep-merge dicts; lists replace).
- Validates and parses into typed dataclasses (see `LGSSM_VAE.config.schema`).

Environment override
- If `LGSSMVAE_CONFIG` is set, it is used as the config path.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from LGSSM_VAE.foundation.errors import ConfigError
from LGSSM_VAE.foundation.validate import require_mapping, require_non_empty_str

from .schema import Config, parse_config


def load_config(path: str | None = None) -> Config:
    """Load YAML configuration and return a validated `Config`."""

    cfg_path = path or os.environ.get("LGSSMVAE_CONFIG") or "config.yaml"
    raw = _load_and_resolve_config(cfg_path)
    return parse_config(raw)


__all__ = ["Config", "load_config"]


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"[error] Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return dict(
        require_mapping(
            raw,
            err="[error] Config YAML must map to a dictionary at the top level",
            exc=ConfigError,
        )
    )


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(out[k], v) if k in out else v
        return out
    # list/scalar: override replaces base entirely
    return override


def _resolve_path(ref: str, *, relative_to: str) -> str:
    if os.path.isabs(ref):
        return ref
    base_dir = os.path.dirname(os.path.abspath(relative_to))
    return os.path.normpath(os.path.join(base_dir, ref))


def _load_and_resolve_config(path: str) -> Dict[str, Any]:
    parts: list[Dict[str, Any]] = []
    stack: list[str] = []

    cur = os.path.abspath(path)
    while True:
        if cur in stack:
            chain = " -> ".join([*stack, cur])
            raise ConfigError(f"[error] extends cycle detected: {chain}")

        stack.append(cur)

        raw = _load_yaml_dict(cur)
        extends = raw.pop("extends", None)
        parts.append(raw)

        if extends is None:
            break

        extends_ref = require_non_empty_str(
            extends,
            err="[error] extends must be a string path",
            exc=ConfigError,
        )
        cur = _resolve_path(extends_ref, relative_to=cur)

    merged: Any = parts[-1]
    for override in reversed(parts[:-1]):
        merged = _deep_merge(merged, override)

    return dict(
        require_mapping(
            merged,
            err="[error] Config YAML must map to a dictionary at the top level",
            exc=ConfigError,
        )
    )
