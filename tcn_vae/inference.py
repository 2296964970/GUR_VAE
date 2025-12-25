from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from .data import (
    load_timeseries,
    load_paired_timeseries,
    apply_standardization_slotwise,
    times_to_5min_index,
)
from .model import TCNVAE
from .utils import parse_sizes


@dataclass
class RangeInferOutput:
    """Container for window-based range inference outputs."""

    timestamps: pd.Series
    indices: np.ndarray
    recon_mean: np.ndarray
    recon_blend: np.ndarray
    sigma_raw: np.ndarray
    rmse_vs_clean: Optional[np.ndarray] = None
    rmse_obs_vs_clean: Optional[np.ndarray] = None


def _parse_sizes_any(value: Any) -> Tuple[int, ...]:
    if isinstance(value, str):
        return parse_sizes(value)
    try:
        # Treat generic sequences (list/tuple) as sizes
        from collections.abc import Sequence

        if isinstance(value, Sequence):
            return tuple(int(v) for v in value)
    except Exception:
        pass
    return parse_sizes(str(value))


def build_model_from_checkpoint(ckpt_path: str, input_dim: int, cfg: Any) -> TCNVAE:
    """Instantiate TCNVAE from a training checkpoint plus config fallback."""
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"[错误] 检查点文件不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    latent_dim = int(ckpt.get("latent_dim", getattr(cfg, "latent_dim", 32)))
    dec_hidden = _parse_sizes_any(ckpt.get("dec_hidden", getattr(cfg, "dec_hidden", "256,256")))
    tcn_channels = _parse_sizes_any(ckpt.get("tcn_channels", getattr(cfg, "tcn_channels", "256,256,256")))

    model = TCNVAE(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=latent_dim,
        tcn_channels=tcn_channels,
        tcn_kernel_size=int(ckpt.get("tcn_kernel_size", getattr(cfg, "tcn_kernel_size", 3))),
        tcn_dropout=float(ckpt.get("tcn_dropout", getattr(cfg, "tcn_dropout", 0.0))),
        dec_hidden=dec_hidden,
        beta=float(getattr(cfg, "beta", 1.0)),
        obs_init_logvar=float(
            ckpt.get("obs_init_logvar", getattr(cfg, "obs_init_logvar", -3.5))
        ),
        enc_diag_eps=float(ckpt.get("enc_diag_eps", getattr(cfg, "enc_diag_eps", 1e-4))),
        dec_eps=float(ckpt.get("dec_eps", getattr(cfg, "dec_eps", 1e-6))),
        dec_logvar_min=float(ckpt.get("dec_logvar_min", getattr(cfg, "dec_logvar_min", -5.0))),
        dec_logvar_max=float(
            ckpt.get("dec_logvar_max", getattr(cfg, "dec_logvar_max", 2.302585092994046))
        ),
        prior_rank=int(ckpt.get("prior_rank", getattr(cfg, "prior_rank", 4))),
        prior_a_init=float(ckpt.get("prior_a_init", getattr(cfg, "prior_a_init", 0.95))),
        prior_q_init=float(ckpt.get("prior_q_init", getattr(cfg, "prior_q_init", 0.1))),
        prior_m0_init=float(ckpt.get("prior_m0_init", getattr(cfg, "prior_m0_init", 0.0))),
        prior_P0_init=float(ckpt.get("prior_P0_init", getattr(cfg, "prior_P0_init", 1.0))),
        prior_jitter=float(ckpt.get("prior_jitter", getattr(cfg, "prior_jitter", 1e-6))),
        prior_variance_floor=float(
            ckpt.get("prior_variance_floor", getattr(cfg, "prior_variance_floor", 1e-6))
        ),
    )
    state_dict = ckpt.get("model", None)
    if state_dict is None:
        raise SystemExit("[错误] 检查点缺少 'model' state_dict")
    model.load_state_dict(state_dict)
    return model


def _load_slot_stats(model_dir: str, cfg: Any) -> Tuple[np.ndarray, np.ndarray, float]:
    stats_path = os.path.join(model_dir, "slot_stats.npz")
    if not os.path.exists(stats_path):
        raise SystemExit(f"[错误] 模型目录中未找到 slot_stats.npz: {model_dir}")
    stats = np.load(stats_path)
    slot_mean = stats["mean"].astype(np.float32)
    slot_std = stats["std"].astype(np.float32)
    # Paper-compatible setup: 5-minute within-day slots (288 slots/day).
    if slot_mean.shape[0] != 288 or slot_std.shape[0] != 288:
        raise SystemExit(
            f"[错误] slot_stats.npz 期望288个时隙（5分钟时隙），"
            f"但实际 mean/std shape={slot_mean.shape}。"
            "此检查点使用了不同的标准化统计量，请使用5分钟时隙统计量训练的检查点。"
        )
    clip_k = float(stats["clip_k"]) if "clip_k" in stats.files else float(
        getattr(cfg, "clip_k", 0.0)
    )
    return slot_mean, slot_std, float(clip_k)


def _select_range_indices(
    ts: pd.Series,
    start_time: str,
    end_time: str,
    time_length: int,
) -> np.ndarray:
    """Select indices within [start_time, end_time] that have sufficient history."""
    if not start_time or not end_time:
        raise SystemExit("[错误] start_time 和 end_time 不能为空")
    ts_dt = pd.to_datetime(ts, errors="raise")
    start_dt = pd.to_datetime(start_time, errors="raise")
    end_dt = pd.to_datetime(end_time, errors="raise")
    if end_dt < start_dt:
        raise SystemExit("[错误] end_time 必须 >= start_time")
    mask = (ts_dt >= start_dt) & (ts_dt <= end_dt)
    idx_all = np.nonzero(mask.to_numpy())[0].astype(np.int64)
    if idx_all.size == 0:
        raise SystemExit(
            f"[错误] 时间范围 [{start_time}, {end_time}] 内对齐后无有效时间戳"
        )
    min_hist = int(time_length) - 1
    idx_valid = idx_all[idx_all >= min_hist]
    if idx_valid.size == 0:
        raise SystemExit(
            f"[错误] 时间范围 [{start_time}, {end_time}] 内"
            f"没有至少 {time_length} 步历史窗口的时间戳"
        )
    skipped = idx_all.size - idx_valid.size
    if skipped > 0:
        print(
            f"[警告] 由于历史窗口不足，跳过范围内最早的 {skipped} 个时间步"
        )
    return idx_valid


def infer_sequence_over_range(
    model: TCNVAE,
    *,
    x_obs: np.ndarray,
    x_obs_std: np.ndarray,
    mask: np.ndarray,
    slots: np.ndarray,
    slot_mean: np.ndarray,
    slot_std: np.ndarray,
    ts: pd.Series,
    time_length: int,
    start_time: str,
    end_time: str,
    blend_k_sigma: float,
    blend_softness: float,
    blend_sigma_temperature: float,
    device: torch.device,
    clean_target: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> RangeInferOutput:
    """Run fixed-history window inference over a time range.

    For each target time index t in [start_time, end_time], we build a window
    [t-time_length+1, t] from the ORIGINAL observed sequence x_obs_std and mask,
    without ever feeding back reconstructed values into future windows.

    slot_mean/slot_std are robust statistics computed on the training normal
    split using 5-minute within-day slots (288 slots/day). They are required
    to unstandardize model outputs and keep blending/metrics in the raw data
    domain.
    """
    if x_obs.shape != mask.shape:
        raise ValueError("x_obs and mask must share shape [T,H]")
    T, H = x_obs.shape
    if x_obs_std.shape != (T, H):
        raise ValueError("x_obs_std must have shape [T,H]")
    if slots.shape[0] != T:
        raise ValueError("slots must have length T")
    if slot_mean.shape != slot_std.shape:
        raise ValueError("slot_mean and slot_std must share shape [S,H]")
    if slot_mean.ndim != 2 or slot_mean.shape[1] != H:
        raise ValueError("slot_mean/slot_std must have shape [S,H] with H matching x_obs")

    target_indices = _select_range_indices(ts, start_time, end_time, time_length)
    num_steps = target_indices.size

    recon_mean = np.zeros((num_steps, H), dtype=np.float32)
    recon_blend = np.zeros_like(recon_mean)
    sigma_raw = np.zeros_like(recon_mean)
    rmse_vec: Optional[np.ndarray]
    rmse_obs_vec: Optional[np.ndarray]
    if clean_target is not None:
        if clean_target.shape != (T, H):
            raise ValueError("clean_target must have shape [T,H]")
        rmse_vec = np.full(num_steps, np.nan, dtype=np.float32)
        rmse_obs_vec = np.full(num_steps, np.nan, dtype=np.float32)
    else:
        rmse_vec = None
        rmse_obs_vec = None

    x_std_t = torch.from_numpy(x_obs_std).to(device=device, dtype=torch.float32)
    m_t = torch.from_numpy(mask.astype(np.float32)).to(device=device)

    model.eval()
    with torch.no_grad():
        for i, t_idx in enumerate(target_indices):
            start_idx = int(t_idx) - int(time_length) + 1
            end_idx = int(t_idx) + 1
            if start_idx < 0 or end_idx > T:
                raise RuntimeError("Window indices out of range; validation should prevent this")
            x_win = x_std_t[start_idx:end_idx, :].unsqueeze(0)
            m_win = m_t[start_idx:end_idx, :].unsqueeze(0)

            mean_seq, logvar_seq = model.reconstruct(
                x_win, m_win, use_mean=True, return_logvar=True
            )
            mean_last = mean_seq[:, -1, :].squeeze(0).cpu().numpy()
            logvar_last = logvar_seq[:, -1, :].squeeze(0).cpu().numpy()

            sigma_std = np.exp(0.5 * logvar_last).astype(np.float32)
            sigma_std *= float(blend_sigma_temperature)

            slot_idx = int(slots[t_idx])
            # Slot-wise unstandardization back to raw domain so that
            # blending and metrics are in the same scale as x_obs.
            mean_slot = slot_mean[slot_idx]
            std_slot = slot_std[slot_idx]
            mu_raw = mean_last * std_slot + mean_slot
            sigma_raw_i = sigma_std * std_slot

            recon_mean[i, :] = mu_raw
            sigma_raw[i, :] = sigma_raw_i

            obs_t = x_obs[t_idx]
            m_vec = mask[t_idx]

            # Residual and standardized residual only on observed entries
            r = obs_t - mu_raw
            z = np.zeros_like(r, dtype=np.float32)
            obs_mask = m_vec > 0.5
            if np.any(obs_mask):
                denom = sigma_raw_i[obs_mask] + float(eps)
                z[obs_mask] = np.abs(r[obs_mask]) / denom

            # Confidence weight toward original observation
            s_arg = (float(blend_k_sigma) - z) / float(blend_softness)
            # Clip to avoid overflow in exp (exp(88) ≈ 1.6e38, safe for float64)
            s_arg = np.clip(s_arg, -88.0, 88.0)
            w = 1.0 / (1.0 + np.exp(-s_arg))

            blend = np.where(obs_mask, w * obs_t + (1.0 - w) * mu_raw, mu_raw)
            recon_blend[i, :] = blend

            if rmse_vec is not None:
                y_clean = clean_target[t_idx]
                if np.any(obs_mask):
                    # Baseline-1: use raw observation as-is (no repair)
                    diff_obs = obs_t[obs_mask] - y_clean[obs_mask]
                    mse_obs = float(np.mean(diff_obs * diff_obs))
                    rmse_obs_vec[i] = np.sqrt(mse_obs)

                    # Final: blended repair output
                    diff_blend = blend[obs_mask] - y_clean[obs_mask]
                    mse_blend = float(np.mean(diff_blend * diff_blend))
                    rmse_vec[i] = np.sqrt(mse_blend)

    ts_target = ts.iloc[target_indices].reset_index(drop=True)
    return RangeInferOutput(
        timestamps=ts_target,
        indices=target_indices,
        recon_mean=recon_mean,
        recon_blend=recon_blend,
        sigma_raw=sigma_raw,
        rmse_vs_clean=rmse_vec,
        rmse_obs_vs_clean=rmse_obs_vec,
    )


def run_range_inference(
    cfg: Any,
    *,
    series: str,
    start_time: str,
    end_time: str,
    device: torch.device,
) -> RangeInferOutput:
    """High-level driver for fixed-history window inference.

    Args:
        cfg: SimpleNamespace from tcn_vae.config.load_config.
        series: "normal" or "attacked".
        start_time: inclusive start timestamp (any pandas-parsable format).
        end_time: inclusive end timestamp.
        device: torch.device to run the model on.
    """
    series = series.lower().strip()
    if series not in ("normal", "attacked"):
        raise ValueError('series must be "normal" or "attacked"')

    if series == "normal":
        if not getattr(cfg, "infer_normal_csv", ""):
            raise SystemExit("[错误] config.yaml 中未设置 infer.normal_csv")
        x_obs, m_struct, ts = load_timeseries(cfg.infer_normal_csv)
        x_target = x_obs
    else:
        if not getattr(cfg, "infer_normal_csv", "") or not getattr(cfg, "infer_attacked_csv", ""):
            raise SystemExit(
                "[错误] 攻击序列推理需要同时设置 infer.normal_csv 和 infer.attacked_csv"
            )
        xa, xn, m_struct, ts = load_paired_timeseries(cfg.infer_normal_csv, cfg.infer_attacked_csv)
        x_obs = xa
        x_target = xn

    T, H = x_obs.shape
    slot_mean, slot_std, clip_k = _load_slot_stats(cfg.model_dir, cfg)
    slots = times_to_5min_index(ts)
    x_obs_std = apply_standardization_slotwise(
        x_obs.astype(np.float32),
        m_struct.astype(np.float32),
        slots,
        slot_mean,
        slot_std,
        clip_k=clip_k,
    )

    model = build_model_from_checkpoint(cfg.ckpt, input_dim=H, cfg=cfg)
    model.to(device=device)

    return infer_sequence_over_range(
        model,
        x_obs=x_obs,
        x_obs_std=x_obs_std,
        mask=m_struct.astype(np.float32),
        slots=slots,
        slot_mean=slot_mean,
        slot_std=slot_std,
        ts=ts,
        time_length=int(getattr(cfg, "time_length", 96)),
        start_time=start_time,
        end_time=end_time,
        blend_k_sigma=float(getattr(cfg, "blend_k_sigma", 2.0)),
        blend_softness=float(getattr(cfg, "blend_softness", 0.2)),
        blend_sigma_temperature=float(getattr(cfg, "blend_sigma_temperature", 1.0)),
        device=device,
        clean_target=x_target,
    )


__all__ = [
    "RangeInferOutput",
    "build_model_from_checkpoint",
    "infer_sequence_over_range",
    "run_range_inference",
]
