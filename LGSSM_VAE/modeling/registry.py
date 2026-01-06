from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Mapping

import torch.nn as nn

from LGSSM_VAE.config import Config
from LGSSM_VAE.foundation.errors import CheckpointError, ConfigError
from LGSSM_VAE.foundation.validate import (
    reject_unknown_keys,
    require_float,
    require_int,
    require_int_tuple,
    require_mapping,
    require_non_empty_str,
)

from .baselines import LSTM, MLPVAE, TCN
from .LGSSM_VAE import LGSSMVAE


def _allowed_keys(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _require_hparams_int_tuple(hparams: Mapping[str, Any], *, key: str) -> tuple[int, ...]:
    full_key = f"model_hparams.{key}"
    return require_int_tuple(
        hparams.get(key),
        err=f"[错误] {full_key} 必须为非空整数列表",
        empty_err=f"[错误] {full_key} 必须为非空整数列表",
        item_err=lambda i: f"[错误] {full_key}[{i}] 必须为整数",
        exc=CheckpointError,
    )


@dataclass(frozen=True)
class LGSSMVAEHParams:
    latent_dim: int
    tcn_channels: tuple[int, ...]
    tcn_kernel_size: int
    tcn_dropout: float
    dec_hidden: tuple[int, ...]
    enc_diag_eps: float
    dec_eps: float
    dec_logvar_min: float
    dec_logvar_max: float
    prior_rank: int
    prior_a_init: float
    prior_q_init: float
    prior_m0_init: float
    prior_P0_init: float
    prior_jitter: float
    prior_variance_floor: float
    time_length: int | None = None

    def to_checkpoint_hparams(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "latent_dim": int(self.latent_dim),
            "tcn_channels": [int(x) for x in self.tcn_channels],
            "tcn_kernel_size": int(self.tcn_kernel_size),
            "tcn_dropout": float(self.tcn_dropout),
            "dec_hidden": [int(x) for x in self.dec_hidden],
            "enc_diag_eps": float(self.enc_diag_eps),
            "dec_eps": float(self.dec_eps),
            "dec_logvar_min": float(self.dec_logvar_min),
            "dec_logvar_max": float(self.dec_logvar_max),
            "prior_rank": int(self.prior_rank),
            "prior_a_init": float(self.prior_a_init),
            "prior_q_init": float(self.prior_q_init),
            "prior_m0_init": float(self.prior_m0_init),
            "prior_P0_init": float(self.prior_P0_init),
            "prior_jitter": float(self.prior_jitter),
            "prior_variance_floor": float(self.prior_variance_floor),
        }
        if self.time_length is not None:
            out["time_length"] = int(self.time_length)
        return out

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LGSSMVAEHParams":
        reject_unknown_keys(
            raw,
            allowed=_allowed_keys(cls),
            path="model_hparams",
            exc=CheckpointError,
            kind="checkpoint",
            prefix="[错误]",
        )
        time_length = raw.get("time_length")
        time_length_i = None if time_length is None else require_int(
            time_length,
            err="[错误] model_hparams.time_length 必须为整数",
            exc=CheckpointError,
        )
        return cls(
            time_length=time_length_i,
            latent_dim=require_int(raw.get("latent_dim"), err="[错误] model_hparams.latent_dim 必须为整数", exc=CheckpointError),
            dec_hidden=_require_hparams_int_tuple(raw, key="dec_hidden"),
            tcn_channels=_require_hparams_int_tuple(raw, key="tcn_channels"),
            tcn_kernel_size=require_int(
                raw.get("tcn_kernel_size"),
                err="[错误] model_hparams.tcn_kernel_size 必须为整数",
                exc=CheckpointError,
            ),
            tcn_dropout=require_float(raw.get("tcn_dropout"), err="[错误] model_hparams.tcn_dropout 必须为数值", exc=CheckpointError),
            enc_diag_eps=require_float(raw.get("enc_diag_eps"), err="[错误] model_hparams.enc_diag_eps 必须为数值", exc=CheckpointError),
            dec_eps=require_float(raw.get("dec_eps"), err="[错误] model_hparams.dec_eps 必须为数值", exc=CheckpointError),
            dec_logvar_min=require_float(raw.get("dec_logvar_min"), err="[错误] model_hparams.dec_logvar_min 必须为数值", exc=CheckpointError),
            dec_logvar_max=require_float(raw.get("dec_logvar_max"), err="[错误] model_hparams.dec_logvar_max 必须为数值", exc=CheckpointError),
            prior_rank=require_int(raw.get("prior_rank"), err="[错误] model_hparams.prior_rank 必须为整数", exc=CheckpointError),
            prior_a_init=require_float(raw.get("prior_a_init"), err="[错误] model_hparams.prior_a_init 必须为数值", exc=CheckpointError),
            prior_q_init=require_float(raw.get("prior_q_init"), err="[错误] model_hparams.prior_q_init 必须为数值", exc=CheckpointError),
            prior_m0_init=require_float(raw.get("prior_m0_init"), err="[错误] model_hparams.prior_m0_init 必须为数值", exc=CheckpointError),
            prior_P0_init=require_float(raw.get("prior_P0_init"), err="[错误] model_hparams.prior_P0_init 必须为数值", exc=CheckpointError),
            prior_jitter=require_float(raw.get("prior_jitter"), err="[错误] model_hparams.prior_jitter 必须为数值", exc=CheckpointError),
            prior_variance_floor=require_float(
                raw.get("prior_variance_floor"),
                err="[错误] model_hparams.prior_variance_floor 必须为数值",
                exc=CheckpointError,
            ),
        )


@dataclass(frozen=True)
class TCNHParams:
    tcn_channels: tuple[int, ...]
    tcn_kernel_size: int
    tcn_dropout: float

    def to_checkpoint_hparams(self) -> Dict[str, Any]:
        return {
            "tcn_channels": [int(x) for x in self.tcn_channels],
            "tcn_kernel_size": int(self.tcn_kernel_size),
            "tcn_dropout": float(self.tcn_dropout),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TCNHParams":
        reject_unknown_keys(
            raw,
            allowed=_allowed_keys(cls),
            path="model_hparams",
            exc=CheckpointError,
            kind="checkpoint",
            prefix="[错误]",
        )
        return cls(
            tcn_channels=_require_hparams_int_tuple(raw, key="tcn_channels"),
            tcn_kernel_size=require_int(
                raw.get("tcn_kernel_size"),
                err="[错误] model_hparams.tcn_kernel_size 必须为整数",
                exc=CheckpointError,
            ),
            tcn_dropout=require_float(raw.get("tcn_dropout"), err="[错误] model_hparams.tcn_dropout 必须为数值", exc=CheckpointError),
        )



@dataclass(frozen=True)
class LSTMHParams:
    hidden_size: int
    num_layers: int
    dropout: float

    def to_checkpoint_hparams(self) -> Dict[str, Any]:
        return {
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LSTMHParams":
        reject_unknown_keys(
            raw,
            allowed=_allowed_keys(cls),
            path="model_hparams",
            exc=CheckpointError,
            kind="checkpoint",
            prefix="[错误]",
        )
        return cls(
            hidden_size=require_int(raw.get("hidden_size"), err="[错误] model_hparams.hidden_size 必须为整数", exc=CheckpointError),
            num_layers=require_int(raw.get("num_layers"), err="[错误] model_hparams.num_layers 必须为整数", exc=CheckpointError),
            dropout=require_float(raw.get("dropout"), err="[错误] model_hparams.dropout 必须为数值", exc=CheckpointError),
        )



@dataclass(frozen=True)
class MLPVAEHParams:
    time_length: int
    latent_dim: int
    hidden_sizes: tuple[int, ...]
    dec_eps: float
    dec_logvar_min: float
    dec_logvar_max: float
    obs_init_logvar: float

    def to_checkpoint_hparams(self) -> Dict[str, Any]:
        return {
            "time_length": int(self.time_length),
            "latent_dim": int(self.latent_dim),
            "hidden_sizes": [int(x) for x in self.hidden_sizes],
            "dec_eps": float(self.dec_eps),
            "dec_logvar_min": float(self.dec_logvar_min),
            "dec_logvar_max": float(self.dec_logvar_max),
            "obs_init_logvar": float(self.obs_init_logvar),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MLPVAEHParams":
        reject_unknown_keys(
            raw,
            allowed=_allowed_keys(cls),
            path="model_hparams",
            exc=CheckpointError,
            kind="checkpoint",
            prefix="[错误]",
        )
        return cls(
            time_length=require_int(raw.get("time_length"), err="[错误] model_hparams.time_length 必须为整数", exc=CheckpointError),
            latent_dim=require_int(raw.get("latent_dim"), err="[错误] model_hparams.latent_dim 必须为整数", exc=CheckpointError),
            hidden_sizes=_require_hparams_int_tuple(raw, key="hidden_sizes"),
            dec_eps=require_float(raw.get("dec_eps"), err="[错误] model_hparams.dec_eps 必须为数值", exc=CheckpointError),
            dec_logvar_min=require_float(raw.get("dec_logvar_min"), err="[错误] model_hparams.dec_logvar_min 必须为数值", exc=CheckpointError),
            dec_logvar_max=require_float(raw.get("dec_logvar_max"), err="[错误] model_hparams.dec_logvar_max 必须为数值", exc=CheckpointError),
            obs_init_logvar=require_float(
                raw.get("obs_init_logvar"),
                err="[错误] model_hparams.obs_init_logvar 必须为数值",
                exc=CheckpointError,
            ),
        )




def _build_lgssm_vae(
    *,
    num_features: int,
    hp: LGSSMVAEHParams,
    beta: float,
    obs_init_logvar: float,
) -> LGSSMVAE:
    return LGSSMVAE(
        input_dim=int(num_features),
        output_dim=int(num_features),
        latent_dim=int(hp.latent_dim),
        tcn_channels=hp.tcn_channels,
        tcn_kernel_size=int(hp.tcn_kernel_size),
        tcn_dropout=float(hp.tcn_dropout),
        dec_hidden=hp.dec_hidden,
        beta=float(beta),
        obs_init_logvar=float(obs_init_logvar),
        enc_diag_eps=float(hp.enc_diag_eps),
        dec_eps=float(hp.dec_eps),
        dec_logvar_min=float(hp.dec_logvar_min),
        dec_logvar_max=float(hp.dec_logvar_max),
        prior_rank=int(hp.prior_rank),
        prior_a_init=float(hp.prior_a_init),
        prior_q_init=float(hp.prior_q_init),
        prior_m0_init=float(hp.prior_m0_init),
        prior_P0_init=float(hp.prior_P0_init),
        prior_jitter=float(hp.prior_jitter),
        prior_variance_floor=float(hp.prior_variance_floor),
    )


def _build_tcn(*, num_features: int, hp: TCNHParams) -> TCN:
    return TCN(
        input_dim=int(num_features),
        tcn_channels=hp.tcn_channels,
        tcn_kernel_size=int(hp.tcn_kernel_size),
        tcn_dropout=float(hp.tcn_dropout),
    )


def _build_lstm(*, num_features: int, hp: LSTMHParams) -> LSTM:
    return LSTM(
        input_dim=int(num_features),
        hidden_size=int(hp.hidden_size),
        num_layers=int(hp.num_layers),
        dropout=float(hp.dropout),
    )




def _build_mlp_vae(*, num_features: int, hp: MLPVAEHParams) -> MLPVAE:
    return MLPVAE(
        input_dim=int(num_features),
        time_length=int(hp.time_length),
        latent_dim=int(hp.latent_dim),
        hidden_sizes=hp.hidden_sizes,
        dec_eps=float(hp.dec_eps),
        dec_logvar_min=float(hp.dec_logvar_min),
        dec_logvar_max=float(hp.dec_logvar_max),
        obs_init_logvar=float(hp.obs_init_logvar),
    )


def _build_from_config_lgssm_vae(
    cfg: Config, *, num_features: int, window_length: int
) -> tuple[nn.Module, Dict[str, Any]]:
    m = cfg.model
    hp = LGSSMVAEHParams(
        time_length=int(window_length),
        latent_dim=int(m.latent_dim),
        tcn_channels=tuple(int(x) for x in m.tcn_channels),
        tcn_kernel_size=int(m.tcn_kernel_size),
        tcn_dropout=float(m.tcn_dropout),
        dec_hidden=tuple(int(x) for x in m.dec_hidden),
        enc_diag_eps=float(m.encoder.diag_eps),
        dec_eps=float(m.decoder.eps),
        dec_logvar_min=float(m.decoder.logvar_min),
        dec_logvar_max=float(m.decoder.logvar_max),
        prior_rank=int(m.prior.rank),
        prior_a_init=float(m.prior.a_init),
        prior_q_init=float(m.prior.q_init),
        prior_m0_init=float(m.prior.m0_init),
        prior_P0_init=float(m.prior.P0_init),
        prior_jitter=float(m.prior.jitter),
        prior_variance_floor=float(m.prior.variance_floor),
    )
    model = _build_lgssm_vae(
        num_features=num_features,
        hp=hp,
        beta=float(m.beta),
        obs_init_logvar=float(m.obs_init_logvar),
    )
    return model, hp.to_checkpoint_hparams()


def _build_from_config_tcn(cfg: Config, *, num_features: int, window_length: int) -> tuple[nn.Module, Dict[str, Any]]:
    m = cfg.model
    hp = TCNHParams(
        tcn_channels=tuple(int(x) for x in m.tcn_channels),
        tcn_kernel_size=int(m.tcn_kernel_size),
        tcn_dropout=float(m.tcn_dropout),
    )
    return _build_tcn(num_features=num_features, hp=hp), hp.to_checkpoint_hparams()


def _build_from_config_lstm(cfg: Config, *, num_features: int, window_length: int) -> tuple[nn.Module, Dict[str, Any]]:
    m = cfg.model
    hp = LSTMHParams(
        hidden_size=int(m.lstm_hidden_size),
        num_layers=int(m.lstm_num_layers),
        dropout=float(m.lstm_dropout),
    )
    return _build_lstm(num_features=num_features, hp=hp), hp.to_checkpoint_hparams()




def _build_from_config_mlp_vae(
    cfg: Config, *, num_features: int, window_length: int
) -> tuple[nn.Module, Dict[str, Any]]:
    m = cfg.model
    hidden_sizes = tuple(int(x) for x in m.dec_hidden)
    hp = MLPVAEHParams(
        time_length=int(window_length),
        latent_dim=int(m.latent_dim),
        hidden_sizes=hidden_sizes,
        dec_eps=float(m.decoder.eps),
        dec_logvar_min=float(m.decoder.logvar_min),
        dec_logvar_max=float(m.decoder.logvar_max),
        obs_init_logvar=float(m.obs_init_logvar),
    )
    return _build_mlp_vae(num_features=num_features, hp=hp), hp.to_checkpoint_hparams()


def _build_from_hparams_lgssm_vae(hparams: Mapping[str, Any], *, num_features: int) -> nn.Module:
    hp = LGSSMVAEHParams.from_mapping(hparams)
    return _build_lgssm_vae(
        num_features=num_features,
        hp=hp,
        beta=1.0,
        obs_init_logvar=-3.5,
    )


def _build_from_hparams_tcn(hparams: Mapping[str, Any], *, num_features: int) -> nn.Module:
    hp = TCNHParams.from_mapping(hparams)
    return _build_tcn(num_features=num_features, hp=hp)


def _build_from_hparams_lstm(hparams: Mapping[str, Any], *, num_features: int) -> nn.Module:
    hp = LSTMHParams.from_mapping(hparams)
    return _build_lstm(num_features=num_features, hp=hp)




def _build_from_hparams_mlp_vae(hparams: Mapping[str, Any], *, num_features: int) -> nn.Module:
    hp = MLPVAEHParams.from_mapping(hparams)
    return _build_mlp_vae(num_features=num_features, hp=hp)


_MODEL_BUILDERS_FROM_CONFIG: dict[str, Callable[..., tuple[nn.Module, Dict[str, Any]]]] = {
    "LGSSM-VAE": _build_from_config_lgssm_vae,
    "mlp_vae": _build_from_config_mlp_vae,
    "TCN": _build_from_config_tcn,
    "LSTM": _build_from_config_lstm,
}

_MODEL_BUILDERS_FROM_HPARAMS: dict[str, Callable[..., nn.Module]] = {
    "LGSSM-VAE": _build_from_hparams_lgssm_vae,
    "mlp_vae": _build_from_hparams_mlp_vae,
    "TCN": _build_from_hparams_tcn,
    "LSTM": _build_from_hparams_lstm,
}

def build_model_from_config(
    cfg: Config,
    *,
    input_dim: int,
    time_length: int,
) -> tuple[nn.Module, str, Dict[str, Any]]:
    """根据 `Config` 构建模型，并返回 (model, model_name, model_hparams)。

    `model_hparams` 仅包含推理/重建架构所需的最小字段，用于写入 checkpoint。
    """

    name = str(cfg.model.name).strip() or "LGSSM-VAE"
    num_features = int(input_dim)
    window_length = int(time_length)

    def _unsupported(*_args, **_kwargs):
        raise ConfigError(f"[错误] 不支持的 model.name: {name}")

    builder = _MODEL_BUILDERS_FROM_CONFIG.get(name, _unsupported)
    model, model_hparams = builder(cfg, num_features=num_features, window_length=window_length)
    return model, name, model_hparams


def build_model_from_hparams(
    model_name: str,
    model_hparams: Mapping[str, Any],
    *,
    input_dim: int,
) -> nn.Module:
    """根据 checkpoint 中的 (model_name, model_hparams) 重建模型。"""

    name = require_non_empty_str(
        model_name,
        err="[错误] 检查点字段必须为非空字符串: model_name",
        exc=CheckpointError,
    ).strip()
    hparams = require_mapping(
        model_hparams,
        err="[错误] 检查点字段必须为 dict: model_hparams",
        exc=CheckpointError,
    )
    num_features = int(input_dim)

    def _unsupported(*_args, **_kwargs):
        raise CheckpointError(f"[错误] 不支持的 checkpoint model_name: {model_name}")

    builder = _MODEL_BUILDERS_FROM_HPARAMS.get(name, _unsupported)
    return builder(hparams, num_features=num_features)


__all__ = ["build_model_from_config", "build_model_from_hparams"]
