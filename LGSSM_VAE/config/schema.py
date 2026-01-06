from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Tuple

from LGSSM_VAE.foundation.errors import ConfigError
from LGSSM_VAE.foundation.validate import (
    reject_unknown_keys,
    require_bool,
    require_float,
    require_float_tuple,
    require_int,
    require_int_tuple,
    require_mapping,
    require_non_empty_str,
    require_range2_float,
    require_range2_int,
)


@dataclass(frozen=True)
class GlobalConfig:
    case: str


@dataclass(frozen=True)
class TrainConfig:
    normal_csv: str
    model_dir: str
    device: str
    seed: int
    learning_rate: float
    grad_clip: float
    phase1_epochs: int
    phase2_epochs: int
    warmup_frac: float
    anchor_lambda: float


@dataclass(frozen=True)
class InferConfig:
    normal_csv: str
    attacked_csv: str
    ckpt: str
    start: str
    end: str
    save_outputs: bool
    fixed_sigma_std: float
    disable_blend: bool
    blend_k_sigma: float
    blend_softness: float
    blend_sigma_temperature: float


@dataclass(frozen=True)
class PreprocessConfig:
    clip_k: float
    std_floor: float


@dataclass(frozen=True)
class WindowConfig:
    time_length: int
    stride: int
    batch_size: int
    num_workers: int
    train_ratio: float
    val_ratio: float


@dataclass(frozen=True)
class RobustConfig:
    clean_fraction: float
    seg_len: Tuple[int, int]
    dims_fraction: Tuple[float, float]
    laplace_scales: Tuple[float, ...]

    @property
    def seg_len_min(self) -> int:
        return int(self.seg_len[0])

    @property
    def seg_len_max(self) -> int:
        return int(self.seg_len[1])

    @property
    def dims_fraction_min(self) -> float:
        return float(self.dims_fraction[0])

    @property
    def dims_fraction_max(self) -> float:
        return float(self.dims_fraction[1])


@dataclass(frozen=True)
class EncoderConfig:
    diag_eps: float


@dataclass(frozen=True)
class DecoderConfig:
    eps: float
    logvar_min: float
    logvar_max: float


@dataclass(frozen=True)
class PriorConfig:
    rank: int
    a_init: float
    q_init: float
    m0_init: float
    P0_init: float
    jitter: float
    variance_floor: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    latent_dim: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    beta: float
    dec_hidden: Tuple[int, ...]
    tcn_channels: Tuple[int, ...]
    tcn_kernel_size: int
    tcn_dropout: float
    obs_init_logvar: float
    encoder: EncoderConfig
    decoder: DecoderConfig
    prior: PriorConfig


@dataclass(frozen=True)
class Config:
    global_: GlobalConfig
    train: TrainConfig
    infer: InferConfig
    window: WindowConfig
    preprocess: PreprocessConfig
    robust: RobustConfig
    model: ModelConfig

    @property
    def case(self) -> str:
        return str(self.global_.case)


def _allowed_keys(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _require_key(d: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"[error] Missing config key: {path}.{key}")
    return d[key]


def _optional_mapping(d: Mapping[str, Any], key: str, *, path: str) -> Mapping[str, Any]:
    if key not in d:
        return {}
    return require_mapping(
        d[key],
        err=f"[error] Config key must be a mapping: {path}.{key}",
        exc=ConfigError,
    )




_CASE_PRESETS: dict[str, Dict[str, Any]] = {
    "case118": {
        "latent_dim": 256,
        "beta": 0.1,
        "phase1_epochs": 15,
        "phase2_epochs": 45,
        "warmup_frac": 0.4,
        "dec_logvar_max": 1.0,
    },
    "case57": {"latent_dim": 128},
}

_DEFAULT_CASE_PRESET: Dict[str, Any] = {"latent_dim": 64}

def _case_preset(case: str) -> Dict[str, Any]:
    # Per-case recommended defaults; users can override any of them in YAML.
    return dict(_CASE_PRESETS.get(case, _DEFAULT_CASE_PRESET))


def _validate_training_policy(train_normal_csv: str) -> None:
    """Guardrails for experimental paired training.

    - Forbid any '2025-09' in training CSVs.
    - Forbid 'fdia' in train_normal_csv.
    """
    n = str(train_normal_csv or "").strip().lower()
    if n:
        if "fdia" in n:
            raise ConfigError("[error] train.normal_csv must not contain fdia")
        if "2025-09" in n:
            raise ConfigError("[error] train.normal_csv must not contain 2025-09")


def parse_config(raw: Mapping[str, Any]) -> Config:
    """Validate and parse merged YAML dict into a typed Config."""

    reject_unknown_keys(
        raw,
        allowed={"global", "train", "infer", "preprocess", "window", "robust", "model"},
        path="root",
        exc=ConfigError,
    )

    g_raw = require_mapping(
        _require_key(raw, "global", path="root"),
        err="[error] Config key must be a mapping: global",
        exc=ConfigError,
    )
    reject_unknown_keys(g_raw, allowed=_allowed_keys(GlobalConfig), path="global", exc=ConfigError)

    case = require_non_empty_str(
        _require_key(g_raw, "case", path="global"),
        err="[error] Config key must be a non-empty string: global.case",
        exc=ConfigError,
    )
    global_cfg = GlobalConfig(case=case)

    preset = _case_preset(case)

    tr_raw = _optional_mapping(raw, "train", path="root")
    inf_raw = _optional_mapping(raw, "infer", path="root")
    pre_raw = _optional_mapping(raw, "preprocess", path="root")
    win_raw = _optional_mapping(raw, "window", path="root")
    rob_raw = _optional_mapping(raw, "robust", path="root")
    mod_raw = _optional_mapping(raw, "model", path="root")

    reject_unknown_keys(tr_raw, allowed=_allowed_keys(TrainConfig), path="train", exc=ConfigError)
    reject_unknown_keys(inf_raw, allowed=_allowed_keys(InferConfig), path="infer", exc=ConfigError)
    reject_unknown_keys(pre_raw, allowed=_allowed_keys(PreprocessConfig), path="preprocess", exc=ConfigError)
    reject_unknown_keys(win_raw, allowed=_allowed_keys(WindowConfig), path="window", exc=ConfigError)
    reject_unknown_keys(rob_raw, allowed=_allowed_keys(RobustConfig), path="robust", exc=ConfigError)
    reject_unknown_keys(mod_raw, allowed=_allowed_keys(ModelConfig), path="model", exc=ConfigError)

    # Window
    time_length = require_int(
        win_raw.get("time_length", 96),
        err="[error] Config key must be an integer: window.time_length",
        exc=ConfigError,
    )
    stride = require_int(
        win_raw.get("stride", 24),
        err="[error] Config key must be an integer: window.stride",
        exc=ConfigError,
    )
    batch_size = require_int(
        win_raw.get("batch_size", 64),
        err="[error] Config key must be an integer: window.batch_size",
        exc=ConfigError,
    )
    num_workers = require_int(
        win_raw.get("num_workers", 0),
        err="[error] Config key must be an integer: window.num_workers",
        exc=ConfigError,
    )
    train_ratio = require_float(
        win_raw.get("train_ratio", 0.7),
        err="[error] Config key must be a number: window.train_ratio",
        exc=ConfigError,
    )
    val_ratio = require_float(
        win_raw.get("val_ratio", 0.15),
        err="[error] Config key must be a number: window.val_ratio",
        exc=ConfigError,
    )
    window_cfg = WindowConfig(
        time_length=int(time_length),
        stride=int(stride),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
    )

    # Train
    model_dir = require_non_empty_str(
        tr_raw.get("model_dir", f"output/{case}/models/default"),
        err="[error] Config key must be a non-empty string: train.model_dir",
        exc=ConfigError,
    )
    train_normal_csv = require_non_empty_str(
        tr_raw.get("normal_csv"),
        err="[error] Config key must be a non-empty string: train.normal_csv",
        exc=ConfigError,
    )
    phase1_epochs = require_int(
        tr_raw.get("phase1_epochs", int(preset.get("phase1_epochs", 18))),
        err="[error] Config key must be an integer: train.phase1_epochs",
        exc=ConfigError,
    )
    phase2_epochs = require_int(
        tr_raw.get("phase2_epochs", int(preset.get("phase2_epochs", 42))),
        err="[error] Config key must be an integer: train.phase2_epochs",
        exc=ConfigError,
    )
    warmup_frac = require_float(
        tr_raw.get("warmup_frac", float(preset.get("warmup_frac", 0.2))),
        err="[error] Config key must be a number: train.warmup_frac",
        exc=ConfigError,
    )
    anchor_lambda = require_float(
        tr_raw.get("anchor_lambda", 3.0),
        err="[error] Config key must be a number: train.anchor_lambda",
        exc=ConfigError,
    )
    seed = require_int(
        tr_raw.get("seed", 1337),
        err="[error] Config key must be an integer: train.seed",
        exc=ConfigError,
    )
    learning_rate = require_float(
        tr_raw.get("learning_rate", 3e-4),
        err="[error] Config key must be a number: train.learning_rate",
        exc=ConfigError,
    )
    grad_clip = require_float(
        tr_raw.get("grad_clip", 10000.0),
        err="[error] Config key must be a number: train.grad_clip",
        exc=ConfigError,
    )
    device = require_non_empty_str(
        tr_raw.get("device", "cuda"),
        err="[error] Config key must be a non-empty string: train.device",
        exc=ConfigError,
    )
    train_cfg = TrainConfig(
        normal_csv=train_normal_csv,
        model_dir=model_dir,
        device=device,
        seed=int(seed),
        learning_rate=float(learning_rate),
        grad_clip=float(grad_clip),
        phase1_epochs=int(phase1_epochs),
        phase2_epochs=int(phase2_epochs),
        warmup_frac=float(warmup_frac),
        anchor_lambda=float(anchor_lambda),
    )

    # Infer
    infer_normal_csv = require_non_empty_str(
        inf_raw.get("normal_csv"),
        err="[error] Config key must be a non-empty string: infer.normal_csv",
        exc=ConfigError,
    )
    infer_attacked_csv = require_non_empty_str(
        inf_raw.get("attacked_csv"),
        err="[error] Config key must be a non-empty string: infer.attacked_csv",
        exc=ConfigError,
    )
    ckpt = require_non_empty_str(
        inf_raw.get("ckpt"),
        err="[error] Config key must be a non-empty string: infer.ckpt",
        exc=ConfigError,
    )
    infer_cfg = InferConfig(
        normal_csv=infer_normal_csv,
        attacked_csv=infer_attacked_csv,
        ckpt=ckpt,
        start=str(inf_raw.get("start", "")).strip(),
        end=str(inf_raw.get("end", "")).strip(),
        save_outputs=require_bool(
            inf_raw.get("save_outputs", True),
            err="[error] Config key must be a boolean: infer.save_outputs",
            exc=ConfigError,
        ),
        fixed_sigma_std=require_float(
            inf_raw.get("fixed_sigma_std", 0.5),
            err="[error] Config key must be a number: infer.fixed_sigma_std",
            exc=ConfigError,
        ),
        disable_blend=require_bool(
            inf_raw.get("disable_blend", False),
            err="[error] Config key must be a boolean: infer.disable_blend",
            exc=ConfigError,
        ),
        blend_k_sigma=require_float(
            inf_raw.get("blend_k_sigma", 2.5),
            err="[error] Config key must be a number: infer.blend_k_sigma",
            exc=ConfigError,
        ),
        blend_softness=require_float(
            inf_raw.get("blend_softness", 0.1),
            err="[error] Config key must be a number: infer.blend_softness",
            exc=ConfigError,
        ),
        blend_sigma_temperature=require_float(
            inf_raw.get("blend_sigma_temperature", 1.5),
            err="[error] Config key must be a number: infer.blend_sigma_temperature",
            exc=ConfigError,
        ),
    )

    # Preprocess
    preprocess_cfg = PreprocessConfig(
        clip_k=require_float(
            pre_raw.get("clip_k", 3.0),
            err="[error] Config key must be a number: preprocess.clip_k",
            exc=ConfigError,
        ),
        std_floor=require_float(
            pre_raw.get("std_floor", 0.005),
            err="[error] Config key must be a number: preprocess.std_floor",
            exc=ConfigError,
        ),
    )

    # Robust
    seg_default = (8, int(window_cfg.time_length))
    dims_default = (0.05, 0.30)
    seg_len = require_range2_int(
        rob_raw.get("seg_len", list(seg_default)),
        err="[error] Config key must be a YAML list: robust.seg_len",
        empty_err="[error] Config list must not be empty: robust.seg_len",
        len_err="[error] Config key must be a list of length 2: robust.seg_len",
        range_err="[error] Invalid range (min>max): robust.seg_len",
        item_err=lambda i: f"[error] Config list items must be integers: robust.seg_len[{i}]",
        exc=ConfigError,
    )
    dims_fraction = require_range2_float(
        rob_raw.get("dims_fraction", list(dims_default)),
        err="[error] Config key must be a YAML list: robust.dims_fraction",
        empty_err="[error] Config list must not be empty: robust.dims_fraction",
        len_err="[error] Config key must be a list of length 2: robust.dims_fraction",
        range_err="[error] Invalid range (min>max): robust.dims_fraction",
        item_err=lambda i: f"[error] Config list items must be numbers: robust.dims_fraction[{i}]",
        exc=ConfigError,
    )
    laplace_scales = require_float_tuple(
        rob_raw.get("laplace_scales", [0.1, 0.3, 0.7, 1.2]),
        err="[error] Config key must be a YAML list: robust.laplace_scales",
        empty_err="[error] Config list must not be empty: robust.laplace_scales",
        item_err=lambda i: f"[error] Config list items must be numbers: robust.laplace_scales[{i}]",
        exc=ConfigError,
    )
    robust_cfg = RobustConfig(
        clean_fraction=require_float(
            rob_raw.get("clean_fraction", 0.4),
            err="[error] Config key must be a number: robust.clean_fraction",
            exc=ConfigError,
        ),
        seg_len=seg_len,
        dims_fraction=dims_fraction,
        laplace_scales=laplace_scales,
    )

    # Model (nested)
    enc_raw = _optional_mapping(mod_raw, "encoder", path="model")
    dec_raw = _optional_mapping(mod_raw, "decoder", path="model")
    pri_raw = _optional_mapping(mod_raw, "prior", path="model")
    reject_unknown_keys(enc_raw, allowed=_allowed_keys(EncoderConfig), path="model.encoder", exc=ConfigError)
    reject_unknown_keys(dec_raw, allowed=_allowed_keys(DecoderConfig), path="model.decoder", exc=ConfigError)
    reject_unknown_keys(pri_raw, allowed=_allowed_keys(PriorConfig), path="model.prior", exc=ConfigError)

    name = str(mod_raw.get("name", "LGSSM-VAE")).strip() or "LGSSM-VAE"

    latent_default = int(preset.get("latent_dim", 64))
    latent_dim = require_int(
        mod_raw.get("latent_dim", latent_default),
        err="[error] Config key must be an integer: model.latent_dim",
        exc=ConfigError,
    )
    lstm_hidden_size = require_int(
        mod_raw.get("lstm_hidden_size", int(latent_dim)),
        err="[error] Config key must be an integer: model.lstm_hidden_size",
        exc=ConfigError,
    )
    lstm_num_layers = require_int(
        mod_raw.get("lstm_num_layers", 1),
        err="[error] Config key must be an integer: model.lstm_num_layers",
        exc=ConfigError,
    )
    lstm_dropout = require_float(
        mod_raw.get("lstm_dropout", 0.0),
        err="[error] Config key must be a number: model.lstm_dropout",
        exc=ConfigError,
    )

    beta_default = float(preset.get("beta", 0.5))
    beta = require_float(
        mod_raw.get("beta", beta_default),
        err="[error] Config key must be a number: model.beta",
        exc=ConfigError,
    )

    dec_hidden = require_int_tuple(
        mod_raw.get("dec_hidden", [256, 256]),
        err="[error] Config key must be a YAML list: model.dec_hidden",
        empty_err="[error] Config list must not be empty: model.dec_hidden",
        item_err=lambda i: f"[error] Config list items must be integers: model.dec_hidden[{i}]",
        exc=ConfigError,
    )
    tcn_channels = require_int_tuple(
        mod_raw.get("tcn_channels", [256, 256, 256, 256, 256]),
        err="[error] Config key must be a YAML list: model.tcn_channels",
        empty_err="[error] Config list must not be empty: model.tcn_channels",
        item_err=lambda i: f"[error] Config list items must be integers: model.tcn_channels[{i}]",
        exc=ConfigError,
    )
    tcn_kernel_size = require_int(
        mod_raw.get("tcn_kernel_size", 3),
        err="[error] Config key must be an integer: model.tcn_kernel_size",
        exc=ConfigError,
    )
    tcn_dropout = require_float(
        mod_raw.get("tcn_dropout", 0.0),
        err="[error] Config key must be a number: model.tcn_dropout",
        exc=ConfigError,
    )
    obs_init_logvar = require_float(
        mod_raw.get("obs_init_logvar", -3.5),
        err="[error] Config key must be a number: model.obs_init_logvar",
        exc=ConfigError,
    )

    enc_cfg = EncoderConfig(
        diag_eps=require_float(
            enc_raw.get("diag_eps", 1e-4),
            err="[error] Config key must be a number: model.encoder.diag_eps",
            exc=ConfigError,
        )
    )

    dec_logvar_max_default = float(preset.get("dec_logvar_max", 2.302585092994046))
    dec_cfg = DecoderConfig(
        eps=require_float(
            dec_raw.get("eps", 1e-6),
            err="[error] Config key must be a number: model.decoder.eps",
            exc=ConfigError,
        ),
        logvar_min=require_float(
            dec_raw.get("logvar_min", -5.0),
            err="[error] Config key must be a number: model.decoder.logvar_min",
            exc=ConfigError,
        ),
        logvar_max=require_float(
            dec_raw.get("logvar_max", dec_logvar_max_default),
            err="[error] Config key must be a number: model.decoder.logvar_max",
            exc=ConfigError,
        ),
    )

    pri_cfg = PriorConfig(
        rank=require_int(
            pri_raw.get("rank", 4),
            err="[error] Config key must be an integer: model.prior.rank",
            exc=ConfigError,
        ),
        a_init=require_float(
            pri_raw.get("a_init", 0.95),
            err="[error] Config key must be a number: model.prior.a_init",
            exc=ConfigError,
        ),
        q_init=require_float(
            pri_raw.get("q_init", 0.1),
            err="[error] Config key must be a number: model.prior.q_init",
            exc=ConfigError,
        ),
        m0_init=require_float(
            pri_raw.get("m0_init", 0.0),
            err="[error] Config key must be a number: model.prior.m0_init",
            exc=ConfigError,
        ),
        P0_init=require_float(
            pri_raw.get("P0_init", 1.0),
            err="[error] Config key must be a number: model.prior.P0_init",
            exc=ConfigError,
        ),
        jitter=require_float(
            pri_raw.get("jitter", 1e-6),
            err="[error] Config key must be a number: model.prior.jitter",
            exc=ConfigError,
        ),
        variance_floor=require_float(
            pri_raw.get("variance_floor", 1e-6),
            err="[error] Config key must be a number: model.prior.variance_floor",
            exc=ConfigError,
        ),
    )

    model_cfg = ModelConfig(
        name=name,
        latent_dim=int(latent_dim),
        lstm_hidden_size=int(lstm_hidden_size),
        lstm_num_layers=int(lstm_num_layers),
        lstm_dropout=float(lstm_dropout),
        beta=float(beta),
        dec_hidden=dec_hidden,
        tcn_channels=tcn_channels,
        tcn_kernel_size=int(tcn_kernel_size),
        tcn_dropout=float(tcn_dropout),
        obs_init_logvar=float(obs_init_logvar),
        encoder=enc_cfg,
        decoder=dec_cfg,
        prior=pri_cfg,
    )

    _validate_training_policy(train_cfg.normal_csv)

    return Config(
        global_=global_cfg,
        train=train_cfg,
        infer=infer_cfg,
        window=window_cfg,
        preprocess=preprocess_cfg,
        robust=robust_cfg,
        model=model_cfg,
    )


__all__ = [
    "Config",
    "GlobalConfig",
    "TrainConfig",
    "InferConfig",
    "PreprocessConfig",
    "WindowConfig",
    "RobustConfig",
    "EncoderConfig",
    "DecoderConfig",
    "PriorConfig",
    "ModelConfig",
    "parse_config",
]
