from __future__ import annotations

import os
from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from LGSSM_VAE.config import Config
from LGSSM_VAE.foundation.errors import CheckpointError, DataError, InferenceError
from LGSSM_VAE.foundation.validate import require_int, require_mapping, require_non_empty_str
from LGSSM_VAE.data import (
    apply_standardization_slotwise,
    load_feature_cols,
    load_paired_timeseries,
    load_timeseries,
    times_to_5min_index,
)
from LGSSM_VAE.modeling.registry import build_model_from_hparams


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _infer_angle_mask(feature_cols: Sequence[str], *, num_features: int) -> np.ndarray:
    """Infer angle feature positions from column names.

    Notes:
      - This is intentionally conservative: only known angle-like measurement types.
      - Values are assumed to be in radians.
    """
    if len(feature_cols) != int(num_features):
        raise InferenceError(
            f"feature_cols length mismatch: header={len(feature_cols)} vs num_features={int(num_features)}"
        )
    # Match common angle tokens with underscore (or start/end) boundaries to avoid accidental matches.
    # Examples covered:
    #   va_bus_*, pmu_va_bus_*, if_ang_br_*, it_ang_br_*,
    #   *_angle_*, *_theta_*, *_phase_*, *_phi_*, *_ang_*.
    angle_name_re = re.compile(r"(^|_)(va|ang|angle|theta|phase|phi)(_|$)", re.IGNORECASE)
    mask = np.array([bool(angle_name_re.search(str(c))) for c in feature_cols], dtype=bool)
    return mask


def _blend_angles_on_circle(obs: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Blend angles using circular (unit-circle) interpolation."""
    # Inputs are assumed already wrapped to (-pi, pi].
    s = w * np.sin(obs) + (1.0 - w) * np.sin(mu)
    c = w * np.cos(obs) + (1.0 - w) * np.cos(mu)
    return np.arctan2(s, c).astype(np.float32)


@dataclass(frozen=True)
class RangeInferOutput:
    """范围推理输出（按时间步、raw 域）。"""

    timestamps: pd.Series
    indices: np.ndarray
    recon_mean: np.ndarray
    recon_blend: np.ndarray
    sigma_raw: np.ndarray
    rmse_vs_clean: Optional[np.ndarray] = None
    rmse_obs_vs_clean: Optional[np.ndarray] = None
    rmse_change_vs_obs: Optional[np.ndarray] = None
    mean_weight_obs: Optional[np.ndarray] = None
    mean_abs_change_over_sigma: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RangeInferInputs:
    """将长参数列表打包成结构化输入。"""

    x_obs: np.ndarray
    x_obs_std: np.ndarray
    mask: np.ndarray
    slots: np.ndarray
    slot_mean: np.ndarray
    slot_std: np.ndarray
    ts: pd.Series
    angle_mask: Optional[np.ndarray] = None
    clean_target: Optional[np.ndarray] = None

    def validate(self) -> Tuple[int, int]:
        x_obs = self.x_obs
        mask = self.mask

        if x_obs.ndim != 2 or mask.ndim != 2:
            raise InferenceError("x_obs/mask must have shape [T,H]")
        if x_obs.shape != mask.shape:
            raise InferenceError("x_obs and mask must share shape [T,H]")

        T, H = x_obs.shape

        if self.x_obs_std.shape != (T, H):
            raise InferenceError("x_obs_std must have shape [T,H]")

        slots = self.slots
        if slots.ndim != 1 or slots.shape != (T,):
            raise InferenceError("slots must be 1D with length T")

        slot_mean = self.slot_mean
        slot_std = self.slot_std
        if slot_mean.ndim != 2 or slot_mean.shape[1] != H or slot_std.shape != slot_mean.shape:
            raise InferenceError("slot_mean/slot_std must have shape [S,H] with H matching x_obs")

        if self.clean_target is not None and self.clean_target.shape != (T, H):
            raise InferenceError("clean_target must have shape [T,H]")

        if self.angle_mask is not None:
            angle_mask = np.asarray(self.angle_mask)
            if angle_mask.shape != (H,):
                raise InferenceError("angle_mask must have shape [H]")

        return int(T), int(H)


@dataclass(frozen=True)
class RangeInferRequest:
    """范围推理请求参数（策略无关）。"""

    time_length: int
    start_time: str
    end_time: str
    blend_k_sigma: float
    blend_softness: float
    blend_sigma_temperature: float
    device: torch.device
    eps: float = 1e-6
    fixed_sigma_std: float = 0.5
    disable_blend: bool = False

    def validate(self) -> None:
        time_length = int(self.time_length)
        if time_length <= 0:
            raise InferenceError("time_length must be a positive integer")

        if float(self.blend_softness) <= 0.0:
            raise InferenceError("blend_softness must be > 0")

        if float(self.eps) <= 0.0:
            raise InferenceError("eps must be > 0")

        if float(self.fixed_sigma_std) < 0.0:
            raise InferenceError("fixed_sigma_std must be >= 0")

        if float(self.blend_sigma_temperature) < 0.0:
            raise InferenceError("blend_sigma_temperature must be >= 0")


def _sigma_std_from_logvar_seq(
    logvar_seq: Any, *, num_features: int, fixed_sigma_std: float
) -> np.ndarray:
    if isinstance(logvar_seq, torch.Tensor):
        logvar_last = logvar_seq[:, -1, :].squeeze(0).cpu().numpy()
        return np.exp(0.5 * logvar_last).astype(np.float32)
    return np.full((int(num_features),), float(fixed_sigma_std), dtype=np.float32)


def _compute_blend_weights_sigmoid(
    *,
    residual: np.ndarray,
    sigma_raw: np.ndarray,
    obs_mask: np.ndarray,
    k_sigma: float,
    softness: float,
    eps: float,
) -> np.ndarray:
    z = np.zeros_like(residual, dtype=np.float32)
    denom = sigma_raw + float(eps)
    np.divide(np.abs(residual), denom, out=z, where=obs_mask)

    s_arg = (float(k_sigma) - z) / float(softness)
    s_arg = np.clip(s_arg, -88.0, 88.0)
    return (1.0 / (1.0 + np.exp(-s_arg))).astype(np.float32)


class RangeInferenceStrategy(Protocol):
    def run(
        self,
        model: torch.nn.Module,
        *,
        inputs: RangeInferInputs,
        request: RangeInferRequest,
    ) -> RangeInferOutput: ...


class FixedHistoryWindowInference:
    """默认推理策略：fixed-history window，取窗口末步输出。"""

    def run(
        self,
        model: torch.nn.Module,
        *,
        inputs: RangeInferInputs,
        request: RangeInferRequest,
    ) -> RangeInferOutput:
        T, H = inputs.validate()
        request.validate()

        x_obs = inputs.x_obs
        x_obs_std = inputs.x_obs_std
        mask = inputs.mask
        slots = inputs.slots
        slot_mean = inputs.slot_mean
        slot_std = inputs.slot_std
        ts = inputs.ts
        angle_mask = np.zeros(H, dtype=bool) if inputs.angle_mask is None else np.asarray(inputs.angle_mask, dtype=bool)
        clean_target = inputs.clean_target

        time_length = int(request.time_length)
        target_indices = _select_range_indices(ts, request.start_time, request.end_time, time_length)
        num_steps = int(target_indices.size)

        recon_mean = np.zeros((num_steps, H), dtype=np.float32)
        recon_blend = np.zeros_like(recon_mean)
        sigma_raw = np.zeros_like(recon_mean)

        rmse_change_vec = np.full(num_steps, np.nan, dtype=np.float32)
        mean_weight_vec = np.full(num_steps, np.nan, dtype=np.float32)
        mean_abs_change_over_sigma_vec = np.full(num_steps, np.nan, dtype=np.float32)

        has_clean_target = clean_target is not None
        rmse_vec = np.full(num_steps, np.nan, dtype=np.float32) if has_clean_target else None
        rmse_obs_vec = np.full(num_steps, np.nan, dtype=np.float32) if has_clean_target else None

        x_std_t = torch.from_numpy(x_obs_std).to(device=request.device, dtype=torch.float32)
        m_t = torch.from_numpy(mask.astype(np.float32)).to(device=request.device)

        eps = float(request.eps)
        fixed_sigma_std = float(request.fixed_sigma_std)
        sigma_temp = float(request.blend_sigma_temperature)

        if bool(request.disable_blend):

            def weight_fn(residual: np.ndarray, sigma_raw_i: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
                return np.zeros_like(residual, dtype=np.float32)

        else:
            k_sigma = float(request.blend_k_sigma)
            softness = float(request.blend_softness)

            def weight_fn(residual: np.ndarray, sigma_raw_i: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
                return _compute_blend_weights_sigmoid(
                    residual=residual,
                    sigma_raw=sigma_raw_i,
                    obs_mask=obs_mask,
                    k_sigma=k_sigma,
                    softness=softness,
                    eps=eps,
                )

        model.eval()
        with torch.no_grad():
            for i, t_idx in enumerate(target_indices):
                start_idx = int(t_idx) - int(time_length) + 1
                end_idx = int(t_idx) + 1

                x_win = x_std_t[start_idx:end_idx, :].unsqueeze(0)
                m_win = m_t[start_idx:end_idx, :].unsqueeze(0)

                mean_seq, logvar_seq = model.reconstruct(x_win, m_win, use_mean=True, return_logvar=True)
                mean_last = mean_seq[:, -1, :].squeeze(0).cpu().numpy()

                sigma_std = _sigma_std_from_logvar_seq(
                    logvar_seq,
                    num_features=H,
                    fixed_sigma_std=fixed_sigma_std,
                )
                sigma_std *= sigma_temp

                slot_idx = int(slots[t_idx])
                mean_slot = slot_mean[slot_idx]
                std_slot = slot_std[slot_idx]

                mu_raw = mean_last * std_slot + mean_slot
                sigma_raw_i = sigma_std * std_slot

                sigma_raw[i, :] = sigma_raw_i

                obs_t = x_obs[t_idx]
                obs_mask = mask[t_idx] > 0.5
                obs = obs_t.astype(np.float32, copy=False).copy()
                mu = mu_raw.astype(np.float32, copy=False).copy()

                obs[angle_mask] = _wrap_to_pi(obs[angle_mask])
                mu[angle_mask] = _wrap_to_pi(mu[angle_mask])

                recon_mean[i, :] = mu
                residual = obs - mu
                residual[angle_mask] = _wrap_to_pi(residual[angle_mask])

                w = weight_fn(residual, sigma_raw_i, obs_mask)

                blend = mu.copy()
                lin = obs_mask & (~angle_mask)
                ang = obs_mask & angle_mask
                blend[lin] = w[lin] * obs[lin] + (1.0 - w[lin]) * mu[lin]
                blend[ang] = _blend_angles_on_circle(obs[ang], mu[ang], w[ang])
                recon_blend[i, :] = blend

                if np.any(obs_mask):
                    delta_full = (blend - obs).astype(np.float64)
                    delta_full[angle_mask] = _wrap_to_pi(delta_full[angle_mask])
                    delta = delta_full[obs_mask]
                    rmse_change_vec[i] = float(np.sqrt(np.mean(delta * delta)))
                    mean_weight_vec[i] = float(np.mean(w[obs_mask].astype(np.float64)))
                    denom = sigma_raw_i[obs_mask].astype(np.float64) + eps
                    mean_abs_change_over_sigma_vec[i] = float(np.mean(np.abs(delta) / denom))

                    if has_clean_target:
                        y_clean = clean_target[t_idx]
                        diff_obs_full = (obs - y_clean).astype(np.float64)
                        diff_obs_full[angle_mask] = _wrap_to_pi(diff_obs_full[angle_mask])
                        diff_obs = diff_obs_full[obs_mask]
                        rmse_obs_vec[i] = float(np.sqrt(np.mean(diff_obs * diff_obs)))

                        diff_blend_full = (blend - y_clean).astype(np.float64)
                        diff_blend_full[angle_mask] = _wrap_to_pi(diff_blend_full[angle_mask])
                        diff_blend = diff_blend_full[obs_mask]
                        rmse_vec[i] = float(np.sqrt(np.mean(diff_blend * diff_blend)))

        ts_target = ts.iloc[target_indices].reset_index(drop=True)
        return RangeInferOutput(
            timestamps=ts_target,
            indices=target_indices,
            recon_mean=recon_mean,
            recon_blend=recon_blend,
            sigma_raw=sigma_raw,
            rmse_vs_clean=rmse_vec,
            rmse_obs_vs_clean=rmse_obs_vec,
            rmse_change_vs_obs=rmse_change_vec,
            mean_weight_obs=mean_weight_vec,
            mean_abs_change_over_sigma=mean_abs_change_over_sigma_vec,
        )


def infer_sequence_over_range(
    model: torch.nn.Module,
    *,
    inputs: RangeInferInputs,
    request: RangeInferRequest,
    strategy: Optional[RangeInferenceStrategy] = None,
) -> RangeInferOutput:
    """范围推理入口（默认使用 FixedHistoryWindowInference）。"""

    strat = strategy or FixedHistoryWindowInference()
    return strat.run(model, inputs=inputs, request=request)


def _load_checkpoint_dict(ckpt_path: str) -> Mapping[str, Any]:
    if not os.path.exists(ckpt_path):
        raise CheckpointError(f"[错误] 检查点文件不存在: {ckpt_path}")
    return require_mapping(
        torch.load(ckpt_path, map_location="cpu"),
        err="[错误] 检查点格式错误：期望 dict",
        exc=CheckpointError,
    )


def _try_upgrade_legacy_checkpoint(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    """兼容旧版本 checkpoint（没有 model_name/model_hparams 字段）。

    旧 checkpoint 的超参直接平铺在顶层（如 latent_dim, tcn_channels, dec_hidden...）。
    这里在不改变训练侧严格性的前提下，仅在推理侧做一次“补齐元数据”。
    """

    if "model" not in ckpt:
        return ckpt
    if "model_name" in ckpt and "model_hparams" in ckpt:
        return ckpt

    required_hp = (
        "latent_dim",
        "tcn_channels",
        "tcn_kernel_size",
        "tcn_dropout",
        "dec_hidden",
        "enc_diag_eps",
        "dec_eps",
        "dec_logvar_min",
        "dec_logvar_max",
        "prior_rank",
        "prior_a_init",
        "prior_q_init",
        "prior_m0_init",
        "prior_P0_init",
        "prior_jitter",
        "prior_variance_floor",
    )
    if not all(k in ckpt for k in required_hp):
        return ckpt

    def _parse_int_list(value: Any, *, key: str) -> list[int]:
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",")]
            parts = [p for p in parts if p]
            try:
                return [int(p) for p in parts]
            except Exception as e:  # pragma: no cover
                raise CheckpointError(f"[错误] 旧checkpoint字段 {key} 解析失败: {value}") from e
        if isinstance(value, (list, tuple)):
            try:
                return [int(x) for x in value]
            except Exception as e:  # pragma: no cover
                raise CheckpointError(f"[错误] 旧checkpoint字段 {key} 解析失败: {value}") from e
        raise CheckpointError(f"[错误] 旧checkpoint字段 {key} 格式不支持: {type(value)}")

    model_hparams: dict[str, Any] = {
        "latent_dim": int(ckpt["latent_dim"]),
        "tcn_channels": _parse_int_list(ckpt["tcn_channels"], key="tcn_channels"),
        "tcn_kernel_size": int(ckpt["tcn_kernel_size"]),
        "tcn_dropout": float(ckpt["tcn_dropout"]),
        "dec_hidden": _parse_int_list(ckpt["dec_hidden"], key="dec_hidden"),
        "enc_diag_eps": float(ckpt["enc_diag_eps"]),
        "dec_eps": float(ckpt["dec_eps"]),
        "dec_logvar_min": float(ckpt["dec_logvar_min"]),
        "dec_logvar_max": float(ckpt["dec_logvar_max"]),
        "prior_rank": int(ckpt["prior_rank"]),
        "prior_a_init": float(ckpt["prior_a_init"]),
        "prior_q_init": float(ckpt["prior_q_init"]),
        "prior_m0_init": float(ckpt["prior_m0_init"]),
        "prior_P0_init": float(ckpt["prior_P0_init"]),
        "prior_jitter": float(ckpt["prior_jitter"]),
        "prior_variance_floor": float(ckpt["prior_variance_floor"]),
    }

    out = dict(ckpt)
    out.setdefault("model_name", "LGSSM-VAE")
    out.setdefault("model_hparams", model_hparams)
    return out


def _build_model_from_checkpoint_dict(ckpt: Mapping[str, Any], input_dim: int) -> torch.nn.Module:
    ckpt = _try_upgrade_legacy_checkpoint(ckpt)

    required = ("model", "model_name", "model_hparams")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise CheckpointError(f"[错误] 检查点缺少必要字段: {', '.join(missing)}")

    model_name = require_non_empty_str(
        ckpt.get("model_name"),
        err="[错误] 检查点字段必须为非空字符串: model_name",
        exc=CheckpointError,
    )
    model_hparams = require_mapping(
        ckpt.get("model_hparams"),
        err="[错误] 检查点字段必须为 dict: model_hparams",
        exc=CheckpointError,
    )
    state_dict = require_mapping(
        ckpt.get("model"),
        err="[错误] 检查点字段 'model' 格式错误：期望 state_dict 字典",
        exc=CheckpointError,
    )

    model = build_model_from_hparams(model_name, model_hparams, input_dim=int(input_dim))
    model.load_state_dict(state_dict)
    return model


def build_model_from_checkpoint(ckpt_path: str, input_dim: int) -> torch.nn.Module:
    """从训练检查点重建模型（依赖 checkpoint 自描述元数据）。"""

    ckpt = _load_checkpoint_dict(ckpt_path)
    return _build_model_from_checkpoint_dict(ckpt, input_dim=int(input_dim))


def _load_slot_stats(stats_dir: str) -> Tuple[np.ndarray, np.ndarray, float]:
    stats_path = os.path.join(stats_dir, "slot_stats.npz")
    if not os.path.exists(stats_path):
        raise CheckpointError(f"[错误] 未找到 slot_stats.npz: {stats_dir}")
    stats = np.load(stats_path)
    required = ("mean", "std", "clip_k")
    missing = [k for k in required if k not in stats.files]
    if missing:
        raise CheckpointError(f"[错误] slot_stats.npz 缺少必要字段: {', '.join(missing)}")
    slot_mean = stats["mean"].astype(np.float32)
    slot_std = stats["std"].astype(np.float32)
    if slot_mean.shape[0] != 288 or slot_std.shape[0] != 288:
        raise CheckpointError(
            f"[错误] slot_stats.npz 期望288个时隙（5分钟时隙），"
            f"但实际 mean/std shape={slot_mean.shape}。"
            "此检查点使用了不同的标准化统计量，请使用5分钟时隙统计量训练的检查点。"
        )
    clip_k = float(stats["clip_k"])
    return slot_mean, slot_std, float(clip_k)


def _select_range_indices(
    ts: pd.Series,
    start_time: str,
    end_time: str,
    time_length: int,
) -> np.ndarray:
    if not start_time or not end_time:
        raise InferenceError("[错误] start_time 和 end_time 不能为空")
    ts_dt = pd.to_datetime(ts, errors="raise")
    start_dt = pd.to_datetime(start_time, errors="raise")
    end_dt = pd.to_datetime(end_time, errors="raise")
    if end_dt < start_dt:
        raise InferenceError("[错误] end_time 必须 >= start_time")
    mask = (ts_dt >= start_dt) & (ts_dt <= end_dt)
    idx_all = np.nonzero(mask.to_numpy())[0].astype(np.int64)
    if idx_all.size == 0:
        raise InferenceError(f"[错误] 时间范围 [{start_time}, {end_time}] 内对齐后无有效时间戳")
    min_hist = int(time_length) - 1
    idx_valid = idx_all[idx_all >= min_hist]
    if idx_valid.size == 0:
        raise InferenceError(
            f"[错误] 时间范围 [{start_time}, {end_time}] 内"
            f"没有至少 {time_length} 步历史窗口的时间戳"
        )
    skipped = int(idx_all.size - idx_valid.size)
    if skipped > 0:
        print(f"[警告] 由于历史窗口不足，跳过范围内最早的 {skipped} 个时间步")
    return idx_valid


def _compute_missing_history_steps(ts: pd.Series, start_time: str, time_length: int) -> int:
    if not start_time:
        return 0
    ts_dt = pd.to_datetime(ts, errors="raise")
    start_dt = pd.to_datetime(start_time, errors="raise")
    mask = (ts_dt >= start_dt).to_numpy()
    if not mask.any():
        return 0
    first_idx = int(np.nonzero(mask)[0][0])
    need = (int(time_length) - 1) - first_idx
    return max(0, int(need))


def _prepend_history_from_train(
    *,
    train_normal_csv: str,
    first_infer_ts: str,
    steps: int,
    input_dim: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    X_train, M_train, ts_train = load_timeseries(train_normal_csv)
    if X_train.shape[1] != int(input_dim):
        raise DataError(
            f"[错误] 训练 normal_csv 特征维度不一致: train={X_train.shape[1]} infer={int(input_dim)}"
        )
    ts_train_dt = pd.to_datetime(ts_train, errors="raise")
    first_dt = pd.to_datetime(first_infer_ts, errors="raise")
    eligible = np.nonzero((ts_train_dt < first_dt).to_numpy())[0].astype(np.int64)
    if eligible.size < steps:
        raise DataError(
            f"[错误] 历史不足：需要 {steps} 行（T-1或其子集），但 train.normal_csv 中仅找到 {eligible.size} 行早于推理起始时间"
        )
    idx = eligible[-steps:]
    return X_train[idx], M_train[idx], ts_train.iloc[idx].reset_index(drop=True)


def run_range_inference(
    cfg: Config,
    *,
    series: str,
    start_time: str,
    end_time: str,
    device: torch.device,
    strategy: Optional[RangeInferenceStrategy] = None,
) -> RangeInferOutput:
    """高层 driver：数据读取 + 标准化 + checkpoint 重建 + 调用推理策略。"""

    series_key = str(series).lower().strip()

    def _load_normal() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        x_obs, m_struct, ts = load_timeseries(str(cfg.infer.normal_csv))
        return x_obs, x_obs, m_struct, ts

    def _load_attacked() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        xa, xn, m_struct, ts = load_paired_timeseries(
            str(cfg.infer.normal_csv),
            str(cfg.infer.attacked_csv),
        )
        return xa, xn, m_struct, ts

    def _unsupported_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        raise ValueError('series must be "normal" or "attacked"')

    loader = {"normal": _load_normal, "attacked": _load_attacked}.get(series_key, _unsupported_series)
    x_obs, x_target, m_struct, ts = loader()

    H = int(x_obs.shape[1])
    feature_cols = load_feature_cols(str(cfg.infer.normal_csv))
    angle_mask = _infer_angle_mask(feature_cols, num_features=H)

    ckpt_path = str(cfg.infer.ckpt)
    ckpt = _load_checkpoint_dict(ckpt_path)

    v = ckpt.get("time_length")
    ckpt_time_length = None if v is None else require_int(
        v,
        err="[错误] 检查点字段必须为整数: time_length",
        exc=CheckpointError,
    )
    time_length = int(ckpt_time_length) if ckpt_time_length is not None else int(cfg.window.time_length)

    missing_hist = _compute_missing_history_steps(ts, start_time, time_length)
    if missing_hist > 0:
        x_pre, m_pre, ts_pre = _prepend_history_from_train(
            train_normal_csv=str(cfg.train.normal_csv).strip(),
            first_infer_ts=str(ts.iloc[0]),
            steps=missing_hist,
            input_dim=H,
        )
        x_obs = np.concatenate([x_pre, x_obs], axis=0)
        m_struct = np.concatenate([m_pre, m_struct], axis=0)
        ts = pd.concat([ts_pre, ts], ignore_index=True)

        is_attacked = series_key == "attacked"
        x_target = np.concatenate([x_pre, x_target], axis=0) if is_attacked else x_obs

    ckpt_dir = os.path.dirname(ckpt_path)
    slot_mean, slot_std, clip_k = _load_slot_stats(ckpt_dir)
    slots = times_to_5min_index(ts)

    x_obs_std = apply_standardization_slotwise(
        x_obs.astype(np.float32),
        m_struct.astype(np.float32),
        slots,
        slot_mean,
        slot_std,
        clip_k=float(clip_k),
    )

    model = _build_model_from_checkpoint_dict(ckpt, input_dim=H)
    model.to(device=device)

    inputs = RangeInferInputs(
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=m_struct.astype(np.float32),
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        angle_mask=angle_mask,
        clean_target=x_target,
    )

    request = RangeInferRequest(
        time_length=int(time_length),
        start_time=str(start_time),
        end_time=str(end_time),
        blend_k_sigma=float(cfg.infer.blend_k_sigma),
        blend_softness=float(cfg.infer.blend_softness),
        blend_sigma_temperature=float(cfg.infer.blend_sigma_temperature),
        fixed_sigma_std=float(cfg.infer.fixed_sigma_std),
        disable_blend=bool(cfg.infer.disable_blend),
        device=device,
    )

    return infer_sequence_over_range(model, inputs=inputs, request=request, strategy=strategy)


__all__ = [
    "RangeInferOutput",
    "RangeInferInputs",
    "RangeInferRequest",
    "RangeInferenceStrategy",
    "FixedHistoryWindowInference",
    "build_model_from_checkpoint",
    "infer_sequence_over_range",
    "run_range_inference",
]
